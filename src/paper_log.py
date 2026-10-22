"""Forward paper-trading log for the Stocks-in-Play ORB candidate (no orders, ever).

Daily workflow (after the validation gate has passed, or just to collect evidence):
    python -m src.paper_log plan      # shortly after 09:20 IST: records the day's plans
    python -m src.paper_log settle    # after 15:30 IST: settles them with the SAME simulator
    python -m src.paper_log summary    # running forward-test statistics

Design rules:
  * Plans come from the same ``compute_features`` / ``select_day_plans`` code the
    backtest uses, so live planning cannot drift from what was validated.
  * Each plan stores the exact parameters used; settlement reads them back from the
    log instead of the current defaults.
  * Appends only (JSON lines). Planning and settling are idempotent.
  * A session is settled only if its data reaches the final exit bar; otherwise it
    is left unsettled and can be retried.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.orb_backtest import SimConfig, prepare_data, simulate_plan
from src.orb_validation import GateCriteria, _clean, bootstrap_mean_ci, compute_metrics
from src.strategies.stocks_in_play_orb import (
    DayPlan, SipOrbParams, compute_features, select_day_plans,
)

LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
DEFAULT_LOG = Path("reports/paper_log.jsonl")

# Forward-test thresholds (declared in advance). Stricter evidence is gathered by
# the historical gate; this only decides whether the live record supports continuing.
FORWARD_MIN_SESSIONS = 20
FORWARD_MIN_TRADES = 40


def plan_session(frames: dict[str, pd.DataFrame], session: date, params: SipOrbParams) -> list[DayPlan]:
    """Plans for ``session`` using whatever candles are available (opening bars suffice)."""
    features = {sym.upper(): compute_features(frame, params) for sym, frame in frames.items()}
    return select_day_plans(features, session, params)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _append(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_clean(record)) + "\n")


def record_plans(
    log_path: Path, plans: list[DayPlan], params: SipOrbParams, *, recorded_at: datetime | None = None
) -> int:
    """Append plans not already logged for (session, symbol). Returns how many were added."""
    existing = {(r["session"], r["symbol"]) for r in _read_records(log_path) if r["type"] == "plan"}
    stamp = (recorded_at or datetime.now(IST)).isoformat()
    new = []
    for plan in plans:
        if (plan.session.isoformat(), plan.symbol) in existing:
            continue
        new.append({"type": "plan", "recorded_at": stamp, "session": plan.session, "symbol": plan.symbol,
                    "side": plan.side, "trigger": plan.trigger, "stop_distance": plan.stop_distance,
                    "rvol": plan.rvol, "atr": plan.atr, "or_high": plan.or_high, "or_low": plan.or_low,
                    "params": asdict(params)})
    _append(log_path, new)
    return len(new)


def settle_session(
    log_path: Path, frames: dict[str, pd.DataFrame], session: date, cfg: SimConfig
) -> list[dict[str, Any]]:
    """Settle every logged, unsettled plan of ``session``. Returns the result records written."""
    records = _read_records(log_path)
    settled = {(r["session"], r["symbol"]) for r in records if r["type"] == "result"}
    pending = [r for r in records if r["type"] == "plan" and r["session"] == session.isoformat()
               and (r["session"], r["symbol"]) not in settled]
    if not pending:
        return []
    params = SipOrbParams(**pending[0]["params"])
    prepared = prepare_data({s: frames[s] for s in {r["symbol"] for r in pending} if s in frames}, params)
    last_needed = int(params.square_off_bar.split(":")[0]) * 60 + int(params.square_off_bar.split(":")[1])
    out: list[dict[str, Any]] = []
    for rec in pending:
        day = prepared.bars.get(rec["symbol"], {}).get(session)
        if day is None or len(day.minutes) == 0 or int(day.minutes[-1]) < last_needed:
            LOGGER.warning("%s %s: session data incomplete; leaving unsettled", rec["symbol"], session)
            continue
        plan = DayPlan(symbol=rec["symbol"], session=session, side=rec["side"], trigger=rec["trigger"],
                       stop_distance=rec["stop_distance"], rvol=rec["rvol"], atr=rec["atr"],
                       or_high=rec["or_high"], or_low=rec["or_low"])
        plan_params = SipOrbParams(**rec["params"])
        trade, reason = simulate_plan(plan, day, plan_params, cfg)
        result: dict[str, Any] = {"type": "result", "settled_at": datetime.now(IST).isoformat(),
                                  "session": session, "symbol": rec["symbol"], "outcome": reason}
        if trade is not None:
            result.update({k: v for k, v in trade.items() if k not in ("symbol", "session")})
        out.append(result)
    _append(log_path, out)
    return out


def load_results(log_path: Path) -> pd.DataFrame:
    rows = [r for r in _read_records(log_path) if r["type"] == "result" and r["outcome"] == "traded"]
    return pd.DataFrame(rows)


def summarize_log(
    log_path: Path, capital: float, *, n_boot: int = 2000, seed: int = 1, criteria: GateCriteria | None = None
) -> dict[str, Any]:
    """Forward-test statistics. ``sessions_settled`` counts sessions with at least one settled
    plan; quiet days with no stock in play produce no plans and are not counted."""
    criteria = criteria or GateCriteria()
    records = _read_records(log_path)
    plans = [r for r in records if r["type"] == "plan"]
    results = [r for r in records if r["type"] == "result"]
    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    trades = load_results(log_path)
    metrics = compute_metrics(trades, capital)
    sessions_settled = len({r["session"] for r in results})
    ci = bootstrap_mean_ci(trades, n_boot, np.random.default_rng(seed), criteria.ci_alpha) if len(trades) else (float("nan"),) * 2
    if sessions_settled < FORWARD_MIN_SESSIONS or metrics["n_trades"] < FORWARD_MIN_TRADES:
        status = "COLLECTING"
        note = (f"Need at least {FORWARD_MIN_SESSIONS} settled sessions and {FORWARD_MIN_TRADES} trades "
                f"(have {sessions_settled} and {metrics['n_trades']}). No conclusion yet.")
    elif metrics["mean_net_pnl"] > 0 and not math.isnan(ci[0]) and ci[0] > 0:
        status = "FORWARD_SUPPORTS_CONTINUING"
        note = "Forward mean net P&L is positive with a positive lower bound. Still a small sample."
    else:
        status = "FORWARD_DOES_NOT_SUPPORT"
        note = "Forward record does not show a net-of-cost edge. Do not trade real money on this strategy."
    return _clean({
        "advisory_only": True, "status": status, "note": note,
        "plans_logged": len(plans), "results_logged": len(results), "outcomes": outcomes,
        "sessions_settled": sessions_settled, "metrics": metrics,
        "bootstrap_ci_mean_net_pnl_90": list(ci),
    })


# ----------------------------------------------------------------------------- CLI
def _cli_frames(args: argparse.Namespace, params: SipOrbParams) -> dict[str, pd.DataFrame]:
    from src.advisory_engine import load_watchlist
    from src.orb_validation import _load_real_frames

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols
               else load_watchlist(Path("config/watchlist.yaml")))[: args.max_symbols]
    days = max(params.rvol_lookback, params.atr_lookback) + 10
    frames, errors = _load_real_frames(symbols, days, Path("reports/backtests/orb_live"), refresh=True)
    for err in errors:
        print(f"data warning: {err}")
    return frames


def main() -> int:
    from src.risk_engine import load_phase2_settings

    parser = argparse.ArgumentParser(description="Forward paper-trading log (no orders are ever placed)")
    parser.add_argument("command", choices=("plan", "settle", "summary"))
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--session", help="YYYY-MM-DD (default: today IST)")
    parser.add_argument("--symbols")
    parser.add_argument("--max-symbols", type=int, default=50)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    settings, charge_rates = load_phase2_settings()
    cfg = SimConfig(capital_inr=settings.capital_inr, risk_per_trade_inr=settings.risk_per_trade_inr,
                    slippage_bps_per_side=settings.slippage_bps_per_side, charge_rates=charge_rates)
    session = date.fromisoformat(args.session) if args.session else datetime.now(IST).date()
    params = SipOrbParams()

    if args.command == "summary":
        print(json.dumps(summarize_log(args.log, cfg.capital_inr), indent=2))
        return 0
    now = datetime.now(IST)
    if args.command == "plan":
        earliest = now.replace(hour=9, minute=15 + 5 * params.opening_bars, second=0, microsecond=0)
        if session == now.date() and now < earliest:
            print(f"Too early: the opening range completes at {earliest:%H:%M} IST. Run again after that.")
            return 1
        frames = _cli_frames(args, params)
        plans = plan_session(frames, session, params)
        added = record_plans(args.log, plans, params)
        for p in plans:
            print(f"{p.symbol:<12} {p.side:<5} trigger={p.trigger:.2f} stop_distance={p.stop_distance:.2f} rvol={p.rvol:.1f}")
        print(f"{len(plans)} plan(s), {added} newly logged. These are conditional paper plans, not orders.")
        return 0
    frames = _cli_frames(args, params)
    written = settle_session(args.log, frames, session, cfg)
    print(f"Settled {len(written)} plan(s) for {session}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
