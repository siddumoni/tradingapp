"""Validation gate for the Stocks-in-Play ORB candidate (research only, no orders).

The question this answers is NOT "did it make money in the backtest?" but "is there
statistically credible evidence of a net-of-cost edge on data the choice of
parameters never saw?". Procedure:

  1. Sessions are split chronologically into development and holdout.
  2. A tiny, pre-declared parameter grid is scored on development ONLY.
  3. The single chosen configuration is evaluated ONCE on the holdout.
  4. The holdout must clear every criterion in ``GateCriteria``:
       - enough trades and sessions to mean anything,
       - a day-block bootstrap interval for mean net P&L per trade that is above 0,
       - a direction-placebo p-value (same stocks, same days, random direction),
       - profit factor and drawdown limits, and a profitable development period.

Verdicts: INSUFFICIENT_DATA, FAIL, PASS_HISTORICAL_GATE. A pass is permission to
start forward paper trading, never approval to trade real money.

Run (from the repo root):
    python -m src.orb_validation --days 59                 # Yahoo, up to ~60 days
    python -m src.orb_validation --synthetic null          # harness self-check
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.orb_backtest import BacktestResult, PreparedData, SimConfig, prepare_data, run_backtest
from src.strategies.stocks_in_play_orb import SipOrbParams

LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
DEFAULT_STOP_GRID = (0.10, 0.25, 0.50)


@dataclass(frozen=True)
class GateCriteria:
    min_holdout_trades: int = 100
    min_holdout_sessions: int = 40
    min_dev_trades: int = 30
    max_placebo_p: float = 0.05
    min_ci_lower: float = 0.0           # INR per trade; lower bound of the bootstrap interval
    ci_alpha: float = 0.10              # two-sided 90% interval (5th percentile lower bound)
    min_profit_factor: float = 1.2
    max_drawdown_fraction: float = 0.25  # of capital


def split_sessions(sessions: list[date], holdout_fraction: float) -> tuple[list[date], list[date]]:
    """Chronological split. The holdout is always the most recent block."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    ordered = sorted(sessions)
    if len(ordered) < 2:
        return ordered, []
    k = min(max(1, int(round(len(ordered) * holdout_fraction))), len(ordered) - 1)
    return ordered[:-k], ordered[-k:]


def compute_metrics(trades: pd.DataFrame, capital: float) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {"n_trades": 0, "n_sessions_traded": 0, "win_rate": None, "mean_net_pnl": None,
                "median_net_pnl": None, "mean_r": None, "total_net_pnl": 0.0, "total_gross_pnl": 0.0,
                "total_costs": 0.0, "profit_factor": None, "max_drawdown_inr": 0.0,
                "max_drawdown_fraction": 0.0, "stop_exit_rate": None}
    ordered = trades.sort_values(["entry_time", "symbol"])
    net = ordered["net_pnl"].to_numpy(float)
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    cumulative = np.cumsum(net)
    drawdown = float(np.max(np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[1:] - cumulative)) if len(net) else 0.0
    return {
        "n_trades": int(len(net)),
        "n_sessions_traded": int(ordered["session"].nunique()),
        "win_rate": float((net > 0).mean()),
        "mean_net_pnl": float(net.mean()),
        "median_net_pnl": float(np.median(net)),
        "mean_r": float(ordered["r_multiple"].mean()),
        "total_net_pnl": float(net.sum()),
        "total_gross_pnl": float(ordered["gross_pnl"].sum()),
        "total_costs": float((ordered["costs"] + ordered["slippage"]).sum()),
        "profit_factor": float(wins / losses) if losses > 0 else None,
        "max_drawdown_inr": drawdown,
        "max_drawdown_fraction": drawdown / capital if capital > 0 else None,
        "stop_exit_rate": float(ordered["exit_reason"].str.startswith("stop").mean()),
    }


def bootstrap_mean_ci(
    trades: pd.DataFrame, n_boot: int, rng: np.random.Generator, alpha: float
) -> tuple[float, float]:
    """Day-block bootstrap of mean net P&L per trade.

    Whole sessions are resampled, because trades taken on the same day share market
    conditions and are not independent. Returns (nan, nan) if too few sessions.
    """
    if trades is None or trades.empty:
        return float("nan"), float("nan")
    per_day = trades.groupby("session")["net_pnl"].agg(["sum", "count"])
    if len(per_day) < 5:
        return float("nan"), float("nan")
    sums, counts = per_day["sum"].to_numpy(float), per_day["count"].to_numpy(float)
    idx = rng.integers(0, len(per_day), size=(n_boot, len(per_day)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def placebo_pvalue(
    prepared: PreparedData, params: SipOrbParams, cfg: SimConfig, sessions: list[date],
    observed_mean: float, n_runs: int, rng: np.random.Generator,
) -> tuple[float, float]:
    """P(random-direction trades on the same stocks/days do at least as well).

    Returns (p_value, placebo_mean_of_means). A strategy whose edge is only that it
    picks volatile, high-volume days (and pays costs on both sides) cannot beat this.
    """
    stats: list[float] = []
    for _ in range(n_runs):
        result = run_backtest(prepared, params, cfg, sessions, rng=rng)
        stats.append(float(result.trades["net_pnl"].mean()) if len(result.trades) else float("-inf"))
    arr = np.array(stats)
    p = (1 + int((arr >= observed_mean).sum())) / (n_runs + 1)
    finite = arr[np.isfinite(arr)]
    return float(p), float(finite.mean()) if len(finite) else float("nan")


def evaluate_gate(
    dev: dict[str, Any], hold: dict[str, Any], ci_low: float, p_value: float | None,
    hold_sessions: int, criteria: GateCriteria,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if hold["n_trades"] < criteria.min_holdout_trades:
        reasons.append(f"holdout has {hold['n_trades']} trades; need at least {criteria.min_holdout_trades}")
    if hold_sessions < criteria.min_holdout_sessions:
        reasons.append(f"holdout has {hold_sessions} sessions; need at least {criteria.min_holdout_sessions}")
    if reasons:
        return "INSUFFICIENT_DATA", reasons

    if dev["mean_net_pnl"] is None or dev["mean_net_pnl"] <= 0:
        reasons.append("chosen configuration was not profitable net of costs in the development period")
    if hold["mean_net_pnl"] is None or hold["mean_net_pnl"] <= 0:
        reasons.append(f"holdout mean net P&L per trade is not positive ({hold['mean_net_pnl']})")
    if math.isnan(ci_low) or ci_low <= criteria.min_ci_lower:
        reasons.append(f"bootstrap lower bound of mean net P&L per trade is {ci_low:.2f}; must exceed {criteria.min_ci_lower}")
    if p_value is None or p_value > criteria.max_placebo_p:
        reasons.append(f"direction-placebo p-value {p_value} exceeds {criteria.max_placebo_p}")
    pf = hold["profit_factor"]
    if pf is None or pf < criteria.min_profit_factor:
        reasons.append(f"holdout profit factor {pf} is below {criteria.min_profit_factor}")
    if hold["max_drawdown_fraction"] is None or hold["max_drawdown_fraction"] > criteria.max_drawdown_fraction:
        reasons.append(f"holdout max drawdown {hold['max_drawdown_fraction']} exceeds {criteria.max_drawdown_fraction} of capital")
    return ("FAIL", reasons) if reasons else ("PASS_HISTORICAL_GATE", ["all criteria met on the holdout"])


def _clean(value: Any) -> Any:
    """Make a report JSON-safe (dates, numpy scalars, NaN/inf)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def run_validation(
    frames: dict[str, pd.DataFrame],
    *,
    cfg: SimConfig | None = None,
    base_params: SipOrbParams | None = None,
    stop_grid: tuple[float, ...] = DEFAULT_STOP_GRID,
    holdout_fraction: float = 0.4,
    criteria: GateCriteria | None = None,
    n_boot: int = 2000,
    n_placebo: int = 200,
    seed: int = 12345,
) -> dict[str, Any]:
    cfg = cfg or SimConfig()
    base = base_params or SipOrbParams()
    criteria = criteria or GateCriteria()
    rng = np.random.default_rng(seed)
    prepared = prepare_data(frames, base)
    sessions = prepared.sessions()
    dev_sessions, hold_sessions = split_sessions(sessions, holdout_fraction)

    report: dict[str, Any] = {
        "generated_at": datetime.now(IST).isoformat(),
        "advisory_only": True,
        "orders_placed_or_modified": False,
        "symbols": sorted(prepared.features),
        "sessions_total": len(sessions),
        "sessions_dev": len(dev_sessions),
        "sessions_holdout": len(hold_sessions),
        "holdout_fraction": holdout_fraction,
        "seed": seed,
        "base_params": base.__dict__,
        "sim_config": {k: v for k, v in cfg.__dict__.items() if k != "charge_rates"},
        "criteria": criteria.__dict__,
    }
    if len(sessions) < 2 or not hold_sessions:
        report.update(verdict="INSUFFICIENT_DATA",
                      reasons=["fewer than 2 usable sessions after the volume/ATR warm-up"],
                      dev_grid=[], holdout=None)
        return _clean(report)

    # --- development: score the pre-declared grid; nothing here touches the holdout ---
    grid: list[dict[str, Any]] = []
    for fraction in stop_grid:
        params = replace(base, stop_atr_fraction=fraction)
        res = run_backtest(prepared, params, cfg, dev_sessions)
        grid.append({"stop_atr_fraction": fraction, "metrics": compute_metrics(res.trades, cfg.capital_inr),
                     "skipped": dict(res.skipped), "plans": res.plans_made})
    eligible = [g for g in grid if g["metrics"]["n_trades"] >= criteria.min_dev_trades and g["metrics"]["mean_r"] is not None]
    report["dev_grid"] = grid
    if not eligible:
        report.update(verdict="INSUFFICIENT_DATA",
                      reasons=[f"no configuration reached {criteria.min_dev_trades} trades in the development period"],
                      holdout=None)
        return _clean(report)
    chosen = max(eligible, key=lambda g: g["metrics"]["mean_r"])
    chosen_params = replace(base, stop_atr_fraction=chosen["stop_atr_fraction"])
    report["chosen"] = {"stop_atr_fraction": chosen["stop_atr_fraction"], "selected_on": "development mean R"}

    # --- holdout: evaluated once ---
    result: BacktestResult = run_backtest(prepared, chosen_params, cfg, hold_sessions)
    hold = compute_metrics(result.trades, cfg.capital_inr)
    ci_low, ci_high = bootstrap_mean_ci(result.trades, n_boot, rng, criteria.ci_alpha)
    p_value, placebo_mean = (None, None)
    if hold["n_trades"] > 0:
        p_value, placebo_mean = placebo_pvalue(prepared, chosen_params, cfg, hold_sessions,
                                               hold["mean_net_pnl"], n_placebo, rng)
    verdict, reasons = evaluate_gate(chosen["metrics"], hold, ci_low, p_value, len(hold_sessions), criteria)

    per_session = hold["n_trades"] / len(hold_sessions) if hold_sessions else 0.0
    needed = None
    if per_session > 0:
        hold_needed = max(math.ceil(criteria.min_holdout_trades / per_session), criteria.min_holdout_sessions)
        warm = max(base.rvol_lookback, base.atr_lookback) + 1
        needed = {"holdout_sessions": hold_needed,
                  "total_sessions_including_dev_and_warmup": math.ceil(hold_needed / holdout_fraction) + warm,
                  "assumes_trades_per_session": round(per_session, 3)}
    plans = result.plans_made
    report.update(
        verdict=verdict, reasons=reasons, holdout=hold,
        holdout_bootstrap_ci_mean_net_pnl=[ci_low, ci_high],
        placebo={"p_value": p_value, "mean_net_pnl_of_random_direction": placebo_mean, "runs": n_placebo},
        holdout_plans=plans, holdout_skipped=dict(result.skipped),
        holdout_cost_screen_rejection_rate=(result.skipped.get("cost_filter", 0) / plans) if plans else None,
        sessions_needed_estimate=needed,
        disclaimer=("A pass permits forward paper trading only. It is not a profit forecast and not approval "
                    "to trade real money. Data quality and survivorship are not checked here."),
    )
    return _clean(report)


def format_report(report: dict[str, Any]) -> str:
    lines = [f"VERDICT: {report['verdict']}"]
    lines += [f"  - {r}" for r in report.get("reasons", [])]
    lines.append(f"Sessions: total {report['sessions_total']}, development {report['sessions_dev']}, holdout {report['sessions_holdout']}")
    chosen = report.get("chosen")
    if chosen:
        lines.append(f"Chosen on development only: stop = {chosen['stop_atr_fraction']} x daily ATR")
    hold = report.get("holdout")
    if hold:
        lines.append(f"Holdout trades: {hold['n_trades']}  win rate: {hold['win_rate']}  mean net P&L/trade: {hold['mean_net_pnl']}")
        lines.append(f"Holdout profit factor: {hold['profit_factor']}  max drawdown: {hold['max_drawdown_fraction']}  costs: {hold['total_costs']}")
        lines.append(f"Bootstrap 90% interval of mean net P&L/trade: {report.get('holdout_bootstrap_ci_mean_net_pnl')}")
        lines.append(f"Placebo p-value: {report.get('placebo', {}).get('p_value')}")
        lines.append(f"Plans rejected by the cost screen: {report.get('holdout_cost_screen_rejection_rate')}")
    if report.get("sessions_needed_estimate"):
        lines.append(f"Estimated sessions needed for a decisive test: {report['sessions_needed_estimate']}")
    lines.append("This is research output. It is not a trading recommendation.")
    return "\n".join(lines)


def _load_real_frames(symbols: list[str], days: int, cache_dir: Path, refresh: bool) -> tuple[dict[str, pd.DataFrame], list[str]]:
    from src.multi_stock_validation import _load_or_fetch_candles

    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for symbol in symbols:
        try:
            frame, _, _ = _load_or_fetch_candles(symbol, days, cache_dir, refresh_data=refresh)
            frames[symbol] = frame
        except Exception as exc:  # data availability varies per symbol
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            LOGGER.warning("Skipping %s: %s", symbol, exc)
    return frames, errors


def main() -> int:
    from src.advisory_engine import load_watchlist
    from src.risk_engine import load_phase2_settings

    parser = argparse.ArgumentParser(description="Validation gate for the Stocks-in-Play ORB candidate (research only)")
    parser.add_argument("--symbols", help="Comma-separated NSE symbols (default: config/watchlist.yaml)")
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--days", type=int, default=59, help="History length for Yahoo (5-minute data is limited to ~60 days)")
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/backtests/orb"))
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--synthetic", choices=("null", "edge"), help="Use synthetic data instead of market data (harness self-check)")
    parser.add_argument("--synthetic-sessions", type=int, default=300)
    parser.add_argument("--holdout", type=float, default=0.4)
    parser.add_argument("--n-placebo", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    settings, charge_rates = load_phase2_settings()
    cfg = SimConfig(capital_inr=settings.capital_inr, risk_per_trade_inr=settings.risk_per_trade_inr,
                    slippage_bps_per_side=settings.slippage_bps_per_side, charge_rates=charge_rates)
    data_errors: list[str] = []
    if args.synthetic:
        from src.synthetic_data import make_synthetic_market
        frames = make_synthetic_market(n_sessions=args.synthetic_sessions, seed=args.seed,
                                       edge=0.05 if args.synthetic == "edge" else 0.0)
        source = f"synthetic-{args.synthetic}"
    else:
        symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols
                   else load_watchlist(Path("config/watchlist.yaml")))[: args.max_symbols]
        frames, data_errors = _load_real_frames(symbols, args.days, args.cache_dir, args.refresh_data)
        source = "yahoo-unofficial"
        if len(frames) < 10:
            print(f"Only {len(frames)} symbols loaded; stock-in-play ranking needs a wide universe (>=30 recommended).")
    if not frames:
        print("No data loaded.")
        return 1
    report = run_validation(frames, cfg=cfg, holdout_fraction=args.holdout, n_boot=args.n_boot,
                            n_placebo=args.n_placebo, seed=args.seed)
    report["data_source"] = source
    report["data_errors"] = data_errors
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"orb_validation_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_report(report))
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
