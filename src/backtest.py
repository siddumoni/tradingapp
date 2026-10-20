"""Conservative, session-aware OHLCV backtesting for advisory research only.

Signals use completed candles and entries occur at the next candle open. OHLC
bars cannot reveal the exact order of intrabar events, so stop-loss wins if both
stop and target are touched in the same bar. Intraday positions are not carried
overnight and are closed by the configured square-off time when data permits.

Capital is modelled conservatively as cash-only notional, not assumed MIS margin.
Transaction charges use the project's configurable estimate; slippage remains a
separate assumption. Neither is a substitute for actual contract notes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import time, timedelta
from math import floor, isfinite
import statistics
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from src.cost_calculator import ChargeRates, estimate_intraday_round_trip_cost

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 20_000.0
    risk_per_trade: float = 200.0
    fee_per_order: float = 20.0  # retained for backwards-compatible config; ChargeRates applies brokerage
    slippage_bps_per_side: float = 5.0
    max_trades_per_day: int = 3
    minimum_net_risk_reward: float = 1.5
    entry_cutoff: str = "14:45"
    square_off_time: str = "15:15"
    bar_duration_minutes: int = 5
    charge_rates: ChargeRates = ChargeRates()


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    stop_loss: float
    target: float
    quantity: int
    gross_pnl: float
    brokerage: float
    stt: float
    exchange_charges: float
    sebi_charges: float
    gst: float
    stamp_duty: float
    ipft: float
    slippage: float
    estimated_costs: float
    net_pnl: float
    exit_reason: str


def _side_value(side: str, entry: float, exit_price: float, qty: int) -> float:
    return (exit_price - entry) * qty if side == "LONG" else (entry - exit_price) * qty


def _as_ist(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)


def _clock(value: str) -> time:
    try:
        hour, minute = value.split(":", maxsplit=1)
        return time(int(hour), int(minute))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Time must be HH:MM, got {value!r}") from exc


def _costs_and_slippage(
    side: str, entry: float, exit_price: float, qty: int, config: BacktestConfig
) -> dict[str, float]:
    charges = estimate_intraday_round_trip_cost(
        side=side,
        entry_price=entry,
        exit_price=exit_price,
        quantity=qty,
        rates=config.charge_rates,
    )
    # Slippage is an independent estimate applied to both entry and exit turnover.
    slippage = (entry + exit_price) * qty * config.slippage_bps_per_side / 10_000.0
    result = {
        "brokerage": charges.brokerage,
        "stt": charges.stt,
        "exchange_charges": charges.exchange_charges,
        "sebi_charges": charges.sebi_charges,
        "gst": charges.gst,
        "stamp_duty": charges.stamp_duty,
        "ipft": charges.ipft,
        "slippage": slippage,
    }
    result["estimated_costs"] = sum(result.values())
    return result


def _trade_record(
    symbol: str, position: dict, exit_price: float, exit_time, reason: str,
    config: BacktestConfig,
) -> Trade:
    entry = float(position["entry"])
    qty = int(position["quantity"])
    side = position["side"]
    gross = _side_value(side, entry, float(exit_price), qty)
    costs = _costs_and_slippage(side, entry, float(exit_price), qty, config)
    return Trade(
        symbol=symbol,
        side=side,
        entry_time=str(position["entry_time"]),
        exit_time=str(exit_time),
        entry=entry,
        exit=float(exit_price),
        stop_loss=float(position["stop_loss"]),
        target=float(position["target"]),
        quantity=qty,
        gross_pnl=gross,
        brokerage=costs["brokerage"],
        stt=costs["stt"],
        exchange_charges=costs["exchange_charges"],
        sebi_charges=costs["sebi_charges"],
        gst=costs["gst"],
        stamp_duty=costs["stamp_duty"],
        ipft=costs["ipft"],
        slippage=costs["slippage"],
        estimated_costs=costs["estimated_costs"],
        net_pnl=gross - costs["estimated_costs"],
        exit_reason=reason,
    )


def backtest_fixed_levels(
    symbol: str,
    candles: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame, int], dict | None],
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Backtest callback-generated signals with intraday cutoffs and net R:R.

    Candle index values are interpreted as candle *start* timestamps. A bar's
    close timestamp is its index plus ``bar_duration_minutes``. Entries are at
    the next bar's open. Intrabar stop/target exits are timestamped with the bar
    label because the precise execution instant is unknowable from OHLC data.
    """
    config = config or BacktestConfig()
    required = {"open", "high", "low", "close"}
    if not required.issubset(candles.columns):
        raise ValueError(f"Missing candle columns: {sorted(required - set(candles.columns))}")
    if config.initial_capital <= 0 or config.risk_per_trade <= 0:
        raise ValueError("Capital and risk must be positive")
    if config.minimum_net_risk_reward <= 0:
        raise ValueError("minimum_net_risk_reward must be positive")
    if config.bar_duration_minutes <= 0:
        raise ValueError("bar_duration_minutes must be positive")
    if config.max_trades_per_day < 1:
        raise ValueError("max_trades_per_day must be at least 1")

    entry_cutoff = _clock(config.entry_cutoff)
    square_off = _clock(config.square_off_time)
    if entry_cutoff >= square_off:
        raise ValueError("entry_cutoff must be earlier than square_off_time")

    df = candles.copy()
    df.index = pd.DatetimeIndex([_as_ist(ts) for ts in df.index])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])

    trades: list[Trade] = []
    position: dict | None = None
    daily_trade_count: dict[object, int] = {}
    rejections = {
        "invalid_signal": 0,
        "invalid_levels_or_gap": 0,
        "zero_quantity_or_cash_limit": 0,
        "net_risk_over_budget": 0,
        "net_rr_below_minimum": 0,
        "entry_cutoff": 0,
        "session_boundary": 0,
        "signals_rejected_entry_cutoff": 0,
        "bars_skipped_entry_cutoff": 0,
        "bars_skipped_session_boundary": 0,
    }
    candidate_best_net_rr: list[float] = []
    candidate_initial_net_rr: list[float] = []
    candidate_gross_rr: list[float] = []
    candidate_estimated_costs: list[float] = []
    candidate_qty: list[int] = []
    candidate_audit: list[dict] = []
    candidates_evaluated = 0
    candidates_accepted = 0
    bar_delta = timedelta(minutes=config.bar_duration_minutes)

    def close_position(price: float, exit_time, reason: str) -> None:
        nonlocal position
        if position is not None:
            trades.append(_trade_record(symbol, position, price, exit_time, reason, config))
            position = None

    for i in range(len(df)):
        bar = df.iloc[i]
        bar_start = _as_ist(df.index[i])
        bar_end = bar_start + bar_delta
        bar_day = bar_start.date()
        bar_is_inside_session = bar_start.time() < square_off and bar_end.time() <= square_off
        # If the next observed bar is past the cutoff or on another date, close
        # at the last valid close recorded before the cutoff (never use future bars).
        if position is not None:
            if bar_day != position["entry_day"] or bar_start.time() >= square_off:
                if position.get("last_close") is not None:
                    close_position(position["last_close"], position["last_close_time"], "SESSION_SQUARE_OFF_DATA_GAP")
                else:
                    close_position(float(bar["open"]), str(bar_start), "SESSION_SQUARE_OFF_DATA_GAP")
            elif bar_is_inside_session:
                side = position["side"]
                stop = position["stop_loss"]
                target = position["target"]
                stop_hit = float(bar["low"]) <= stop if side == "LONG" else float(bar["high"]) >= stop
                target_hit = float(bar["high"]) >= target if side == "LONG" else float(bar["low"]) <= target
                if stop_hit:
                    # A stop-market order can fill beyond its trigger after a gap.
                    if side == "LONG":
                        stop_fill = min(stop, float(bar["open"]))
                    else:
                        stop_fill = max(stop, float(bar["open"]))
                    close_position(stop_fill, str(bar_start), "STOP_LOSS")
                elif target_hit:
                    # Use the target price conservatively; do not assume price improvement on gaps.
                    close_position(target, str(bar_start), "TARGET")
                elif bar_end.time() >= square_off:
                    close_position(float(bar["close"]), str(bar_end), "SESSION_SQUARE_OFF")
                else:
                    position["last_close"] = float(bar["close"])
                    position["last_close_time"] = str(bar_end)
            else:
                # A partial bar crossing the cutoff cannot be used for OHLC exits.
                if position.get("last_close") is not None:
                    close_position(position["last_close"], position["last_close_time"], "SESSION_SQUARE_OFF_DATA_GAP")

        # A closed position may re-enter only if the next bar's open is before
        # cutoff and remains within the same trading session.
        if position is not None or i + 1 >= len(df):
            continue
        next_start = _as_ist(df.index[i + 1])
        if next_start.date() != bar_day:
            rejections["session_boundary"] += 1
            rejections["bars_skipped_session_boundary"] += 1
            continue
        if next_start.time() >= entry_cutoff or next_start.time() >= square_off:
            rejections["entry_cutoff"] += 1
            rejections["bars_skipped_entry_cutoff"] += 1
            # Count only genuine strategy signals as cutoff rejections. This is
            # diagnostic-only; no entry is permitted after the cutoff.
            late_signal = signal_fn(df.iloc[: i + 1].copy(), i)
            if late_signal and late_signal.get("side") in ("LONG", "SHORT"):
                rejections["signals_rejected_entry_cutoff"] += 1
            continue
        if daily_trade_count.get(next_start.date(), 0) >= config.max_trades_per_day:
            continue

        signal = signal_fn(df.iloc[: i + 1].copy(), i)
        if not signal:
            continue
        side = signal.get("side")
        if side not in ("LONG", "SHORT"):
            rejections["invalid_signal"] += 1
            continue
        try:
            entry = float(df.iloc[i + 1]["open"])
            stop, target = float(signal["stop_loss"]), float(signal["target"])
        except (TypeError, ValueError, KeyError):
            rejections["invalid_signal"] += 1
            continue
        if not all(isfinite(x) and x > 0 for x in (entry, stop, target)):
            rejections["invalid_levels_or_gap"] += 1
            continue
        risk_per_share = entry - stop if side == "LONG" else stop - entry
        reward_per_share = target - entry if side == "LONG" else entry - target
        if risk_per_share <= 0 or reward_per_share <= 0:
            rejections["invalid_levels_or_gap"] += 1
            continue

        candidates_evaluated += 1
        gross_rr = reward_per_share / risk_per_share
        candidate_gross_rr.append(gross_rr)
        risk_qty = floor(config.risk_per_trade / risk_per_share)
        cash_qty = floor(config.initial_capital / entry)
        qty = min(risk_qty, cash_qty)
        candidate_qty.append(qty)
        if qty < 1:
            rejections["zero_quantity_or_cash_limit"] += 1
            continue

        accepted = False
        last_failure = "net_risk_over_budget"
        best_net_rr = float("-inf")
        initial_net_rr = float("-inf")
        first_qty = qty
        first_qty_audit: dict = {}
        best_qty_audit: dict = {}
        while qty > 0:
            stop_cost = _costs_and_slippage(side, entry, stop, qty, config)["estimated_costs"]
            target_cost = _costs_and_slippage(side, entry, target, qty, config)["estimated_costs"]
            net_risk = risk_per_share * qty + stop_cost
            net_reward = reward_per_share * qty - target_cost
            net_rr = net_reward / net_risk if net_risk > 0 else float("-inf")
            best_net_rr = max(best_net_rr, net_rr)
            if qty == first_qty:
                initial_net_rr = net_rr
            candidate_estimated_costs.append((stop_cost + target_cost) / 2.0)
            if qty == first_qty:
                first_qty_audit = {
                    "quantity": int(qty),
                    "gross_risk_inr": round(risk_per_share * qty, 2),
                    "gross_reward_inr": round(reward_per_share * qty, 2),
                    "stop_exit_costs": {k: round(v, 4) for k, v in _costs_and_slippage(side, entry, stop, qty, config).items()},
                    "target_exit_costs": {k: round(v, 4) for k, v in _costs_and_slippage(side, entry, target, qty, config).items()},
                    "net_risk_inr": round(net_risk, 2),
                    "net_reward_inr": round(net_reward, 2),
                    "net_rr": round(net_rr, 4) if isfinite(net_rr) else None,
                }
            if not best_qty_audit or net_rr > best_qty_audit.get("net_rr", float("-inf")):
                best_qty_audit = {
                    "quantity": int(qty),
                    "gross_risk_inr": round(risk_per_share * qty, 2),
                    "gross_reward_inr": round(reward_per_share * qty, 2),
                    "stop_exit_costs": {k: round(v, 4) for k, v in _costs_and_slippage(side, entry, stop, qty, config).items()},
                    "target_exit_costs": {k: round(v, 4) for k, v in _costs_and_slippage(side, entry, target, qty, config).items()},
                    "net_risk_inr": round(net_risk, 2),
                    "net_reward_inr": round(net_reward, 2),
                    "net_rr": round(net_rr, 4) if isfinite(net_rr) else None,
                }
            if net_risk > config.risk_per_trade:
                last_failure = "net_risk_over_budget"
                qty -= 1
                continue
            if net_reward <= 0 or net_reward / net_risk < config.minimum_net_risk_reward:
                last_failure = "net_rr_below_minimum"
                qty -= 1
                continue
            accepted = True
            break
        candidate_best_net_rr.append(best_net_rr if isfinite(best_net_rr) else 0.0)
        candidate_initial_net_rr.append(initial_net_rr if isfinite(initial_net_rr) else 0.0)
        accepted_qty_audit: dict = {}
        if accepted:
            accepted_stop_costs = _costs_and_slippage(side, entry, stop, qty, config)
            accepted_target_costs = _costs_and_slippage(side, entry, target, qty, config)
            accepted_net_risk = risk_per_share * qty + accepted_stop_costs["estimated_costs"]
            accepted_net_reward = reward_per_share * qty - accepted_target_costs["estimated_costs"]
            accepted_net_rr = (
                accepted_net_reward / accepted_net_risk if accepted_net_risk > 0 else float("-inf")
            )
            accepted_qty_audit = {
                "quantity": int(qty),
                "gross_risk_inr": round(risk_per_share * qty, 2),
                "gross_reward_inr": round(reward_per_share * qty, 2),
                "stop_exit_costs": {k: round(v, 4) for k, v in accepted_stop_costs.items()},
                "target_exit_costs": {k: round(v, 4) for k, v in accepted_target_costs.items()},
                "net_risk_inr": round(accepted_net_risk, 2),
                "net_reward_inr": round(accepted_net_reward, 2),
                "net_rr": round(accepted_net_rr, 4) if isfinite(accepted_net_rr) else None,
            }
        candidate_audit.append({
            "signal_time": str(bar_start),
            "planned_entry_time": str(next_start),
            "side": side,
            "signal_close": round(float(bar["close"]), 4),
            "planned_entry": round(entry, 4),
            "stop_loss": round(stop, 4),
            "target": round(target, 4),
            "risk_per_share": round(risk_per_share, 4),
            "reward_per_share": round(reward_per_share, 4),
            "gross_rr": round(gross_rr, 4),
            "risk_quantity_cap": int(max(risk_qty, 0)),
            "cash_quantity_cap": int(max(cash_qty, 0)),
            "quantity_before_net_filters": int(first_qty),
            "initial_quantity_calculation": first_qty_audit,
            "best_quantity_calculation": best_qty_audit,
            "accepted_quantity_calculation": accepted_qty_audit,
            "accepted": accepted,
            "rejection_reason": None if accepted else last_failure,
        })
        if not accepted:
            if min(risk_qty, cash_qty) < 1:
                rejections["zero_quantity_or_cash_limit"] += 1
            else:
                rejections[last_failure] += 1
            continue

        candidates_accepted += 1
        position = {
            "side": side,
            "entry": entry,
            "entry_time": str(next_start),
            "entry_day": next_start.date(),
            "stop_loss": stop,
            "target": target,
            "quantity": qty,
            "last_close": None,
            "last_close_time": None,
        }
        daily_trade_count[next_start.date()] = daily_trade_count.get(next_start.date(), 0) + 1

    # Never carry a position beyond the data/session. Use only the last close
    # observed before the configured square-off time for that session.
    if position is not None:
        if position.get("last_close") is not None:
            close_position(position["last_close"], position["last_close_time"], "SESSION_SQUARE_OFF_DATA_GAP")
        elif len(df):
            last = df.iloc[-1]
            last_start = _as_ist(df.index[-1])
            if last_start.date() == position["entry_day"] and last_start.time() < square_off:
                close_position(float(last["close"]), str(last_start + bar_delta), "END_OF_DATA")
            else:
                close_position(float(last["open"]), str(last_start), "SESSION_SQUARE_OFF_DATA_GAP")

    trade_df = pd.DataFrame([asdict(t) for t in trades])
    summary = summarize_trades(trade_df, config.initial_capital)
    summary.update({
        "total_estimated_costs": float(trade_df["estimated_costs"].sum()) if not trade_df.empty else 0.0,
        "gross_pnl": float(trade_df["gross_pnl"].sum()) if not trade_df.empty else 0.0,
        "max_drawdown_pct_initial_capital": float(summary["max_drawdown"] / config.initial_capital * 100.0),
        "rejections": rejections,
        "risk_per_trade": config.risk_per_trade,
        "candidate_diagnostics": {
            "valid_signal_candidates_evaluated": candidates_evaluated,
            "candidates_accepted": candidates_accepted,
            "gross_rr_median": float(statistics.median(candidate_gross_rr)) if candidate_gross_rr else None,
            "gross_rr_max": max(candidate_gross_rr) if candidate_gross_rr else None,
            "initial_quantity_net_rr_median": float(statistics.median(candidate_initial_net_rr)) if candidate_initial_net_rr else None,
            "initial_quantity_net_rr_max": max(candidate_initial_net_rr) if candidate_initial_net_rr else None,
            "best_net_rr_across_tested_quantities_median": float(statistics.median(candidate_best_net_rr)) if candidate_best_net_rr else None,
            "best_net_rr_across_tested_quantities_max": max(candidate_best_net_rr) if candidate_best_net_rr else None,
            "average_estimated_round_trip_cost_per_quantity_test": float(statistics.mean(candidate_estimated_costs)) if candidate_estimated_costs else None,
            "quantity_median_before_net_filters": float(statistics.median(candidate_qty)) if candidate_qty else None,
            "minimum_net_risk_reward_required": config.minimum_net_risk_reward,
            "diagnostic_candidates_recorded": len(candidate_audit),
            "best_candidates_by_net_rr": sorted(
                candidate_audit,
                key=lambda item: (item["best_quantity_calculation"].get("net_rr") is not None,
                                  item["best_quantity_calculation"].get("net_rr") or float("-inf")),
                reverse=True,
            )[:25],
            "diagnostic_note": "Candidate costs are itemized separately at stop and target exit prices. Best quantity is the tested quantity with the highest computed net R:R; it is diagnostic, not a trade recommendation.",
        },
        "entry_cutoff": config.entry_cutoff,
        "square_off_time": config.square_off_time,
        "bar_duration_minutes": config.bar_duration_minutes,
        "minimum_net_risk_reward": config.minimum_net_risk_reward,
        "capital_model": "cash-only notional cap; MIS margin not assumed",
        "cost_model": "project ChargeRates estimates plus configured per-side slippage assumption",
        "intrabar_collision_policy": "stop-loss first when stop and target both touch in one candle",
        "timestamp_policy": "candle index is bar start; square-off uses bar close timestamp; intrabar exit time is represented by candle start",
    })
    return trade_df, summary


def summarize_trades(trades: pd.DataFrame, initial_capital: float = 20_000.0) -> dict:
    """Compute trade metrics using realized net P&L and equity-curve drawdown."""
    if trades.empty:
        return {
            "trades": 0, "win_rate_pct": 0.0, "net_pnl": 0.0, "expectancy_per_trade": 0.0,
            "max_drawdown": 0.0, "profit_factor": 0.0, "ending_equity": initial_capital,
        }
    pnl = trades["net_pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = initial_capital + pnl.cumsum()
    peak = equity.cummax().clip(lower=initial_capital)
    drawdown = peak - equity
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "trades": int(len(trades)),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "net_pnl": float(pnl.sum()),
        "expectancy_per_trade": float(pnl.mean()),
        "max_drawdown": float(drawdown.max()),
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "ending_equity": float(initial_capital + pnl.sum()),
    }
