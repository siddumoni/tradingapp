"""Deterministic simulator for the Stocks-in-Play ORB candidate (research only).

Never places orders. Uses the repository's own cost model so results are net of
brokerage, STT, exchange/SEBI charges, GST, stamp duty and slippage.

Fill assumptions (deliberately pessimistic where a 5-minute bar hides the order
of events):
  * Entry is a stop order at the opening-range extreme. If a bar opens beyond the
    trigger the fill is at that open (gap through), otherwise at the trigger.
  * If the entry bar also trades through the stop, the stop is assumed to hit
    (adverse move after the fill). This overstates losses slightly, never gains.
  * A later bar that opens beyond the stop exits at that open, not at the stop.
  * Stop exits pay extra slippage (stop-market orders slip more than limits).
  * Positions are sized so the stop loses at most ``risk_per_trade_inr`` and are
    capped at ``capital * leverage / top_n`` notional (cash-only by default).
  * No re-entry, one position per symbol per session, everything squared off by
    the final bar.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.cost_calculator import ChargeRates, estimate_intraday_round_trip_cost
from src.strategies.stocks_in_play_orb import (
    DayPlan, Side, SipOrbParams, compute_features, select_day_plans, _to_ist_index,
)


@dataclass(frozen=True)
class SimConfig:
    capital_inr: float = 20_000.0
    risk_per_trade_inr: float = 200.0
    leverage: float = 1.0
    slippage_bps_per_side: float = 5.0
    stop_extra_slippage_bps: float = 5.0
    charge_rates: ChargeRates = field(default_factory=ChargeRates)

    def __post_init__(self) -> None:
        if min(self.capital_inr, self.risk_per_trade_inr, self.leverage) <= 0:
            raise ValueError("capital, risk and leverage must be positive")
        if self.slippage_bps_per_side < 0 or self.stop_extra_slippage_bps < 0:
            raise ValueError("slippage cannot be negative")


@dataclass
class DayBars:
    index: pd.DatetimeIndex
    minutes: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


@dataclass
class PreparedData:
    features: dict[str, pd.DataFrame]
    bars: dict[str, dict[date, DayBars]]

    def sessions(self) -> list[date]:
        """Sessions where at least one symbol has complete rvol and ATR features."""
        found: set[date] = set()
        for feats in self.features.values():
            if feats is not None and not feats.empty:
                ok = feats.dropna(subset=["rvol", "atr"])
                found.update(ok.index)
        return sorted(found)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    skipped: Counter
    sessions: list[date]
    plans_made: int


def _minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def prepare_data(frames: dict[str, pd.DataFrame], params: SipOrbParams) -> PreparedData:
    features: dict[str, pd.DataFrame] = {}
    bars: dict[str, dict[date, DayBars]] = {}
    for symbol, frame in frames.items():
        sym = symbol.upper()
        feats = compute_features(frame, params)
        if feats.empty:
            continue
        features[sym] = feats
        df = _to_ist_index(frame)
        arrays = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in ("open", "high", "low", "close")}
        keys = np.array(df.index.date)
        mins = (df.index.hour * 60 + df.index.minute).to_numpy()
        starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
        ends = np.concatenate((starts[1:], [len(df)]))
        per_day: dict[date, DayBars] = {}
        for a, b in zip(starts, ends):
            per_day[keys[a]] = DayBars(
                index=df.index[a:b], minutes=mins[a:b],
                open=arrays["open"][a:b], high=arrays["high"][a:b],
                low=arrays["low"][a:b], close=arrays["close"][a:b],
            )
        bars[sym] = per_day
    return PreparedData(features=features, bars=bars)


def _slippage(entry: float, exit_price: float, qty: int, exit_reason: str, cfg: SimConfig) -> float:
    base = (entry + exit_price) * qty * cfg.slippage_bps_per_side / 10_000.0
    extra = exit_price * qty * cfg.stop_extra_slippage_bps / 10_000.0 if exit_reason.startswith("stop") else 0.0
    return base + extra


def simulate_plan(
    plan: DayPlan, day: DayBars, params: SipOrbParams, cfg: SimConfig
) -> tuple[dict | None, str]:
    """Simulate one plan. Returns (trade or None, reason)."""
    stop_dist = plan.stop_distance
    side: Side = plan.side
    sign = 1.0 if side == "LONG" else -1.0

    # Pre-trade sizing and cost screen use only the trigger price (known in advance).
    risk_qty = math.floor(cfg.risk_per_trade_inr / stop_dist)
    cap_qty = math.floor(cfg.capital_inr * cfg.leverage / params.top_n / plan.trigger)
    qty = min(risk_qty, cap_qty)
    if qty < 1:
        return None, "qty_below_1"
    planned_stop = plan.trigger - sign * stop_dist
    if planned_stop <= 0:
        return None, "invalid_stop"
    est = estimate_intraday_round_trip_cost(
        side=side, entry_price=plan.trigger, exit_price=planned_stop, quantity=qty, rates=cfg.charge_rates
    ).total
    est += _slippage(plan.trigger, planned_stop, qty, "stop", cfg)
    if est > params.max_cost_fraction_of_risk * stop_dist * qty:
        return None, "cost_filter"

    n_open = params.opening_bars
    cutoff, square_off = _minutes(params.entry_cutoff), _minutes(params.square_off_bar)
    count = len(day.minutes)
    entry_i = None
    fill = float("nan")
    for i in range(n_open, count):
        if day.minutes[i] >= cutoff:
            break
        if side == "LONG" and day.high[i] >= plan.trigger:
            fill, entry_i = max(plan.trigger, day.open[i]), i
            break
        if side == "SHORT" and day.low[i] <= plan.trigger:
            fill, entry_i = min(plan.trigger, day.open[i]), i
            break
    if entry_i is None:
        return None, "no_trigger"

    stop = fill - sign * stop_dist
    exit_px, exit_i, reason = float("nan"), count - 1, "session_end"
    for j in range(entry_i, count):
        o, h, l, c = day.open[j], day.high[j], day.low[j], day.close[j]
        if j > entry_i and ((side == "LONG" and o <= stop) or (side == "SHORT" and o >= stop)):
            exit_px, exit_i, reason = o, j, "stop_gap"
            break
        if (side == "LONG" and l <= stop) or (side == "SHORT" and h >= stop):
            exit_px, exit_i, reason = stop, j, "stop"
            break
        if day.minutes[j] >= square_off:
            exit_px, exit_i, reason = c, j, "square_off"
            break
        if j == count - 1:
            exit_px, exit_i, reason = c, j, "session_end"
    if math.isnan(exit_px):
        exit_px = float(day.close[-1])

    gross = sign * (exit_px - fill) * qty
    charges = estimate_intraday_round_trip_cost(
        side=side, entry_price=fill, exit_price=exit_px, quantity=qty, rates=cfg.charge_rates
    ).total
    slip = _slippage(fill, exit_px, qty, reason, cfg)
    net = gross - charges - slip
    planned_risk = stop_dist * qty
    return {
        "symbol": plan.symbol, "session": plan.session, "side": side, "rvol": plan.rvol,
        "entry_time": str(day.index[entry_i]), "entry": float(fill), "stop": float(stop),
        "exit_time": str(day.index[exit_i]), "exit": float(exit_px), "exit_reason": reason,
        "quantity": int(qty), "gross_pnl": float(gross), "costs": float(charges),
        "slippage": float(slip), "net_pnl": float(net), "planned_risk": float(planned_risk),
        "r_multiple": float(net / planned_risk),
    }, "traded"


def run_backtest(
    prepared: PreparedData,
    params: SipOrbParams,
    cfg: SimConfig,
    sessions: list[date] | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> BacktestResult:
    """Run the strategy over ``sessions``. ``rng`` randomises direction (placebo test)."""
    use = sessions if sessions is not None else prepared.sessions()
    trades: list[dict] = []
    skipped: Counter = Counter()
    plans_made = 0
    for session in use:
        plans = select_day_plans(prepared.features, session, params)
        if rng is not None and plans:
            override = {p.symbol: ("LONG" if rng.random() < 0.5 else "SHORT") for p in plans}
            plans = select_day_plans(prepared.features, session, params, side_override=override)
        plans_made += len(plans)
        for plan in plans:
            day = prepared.bars.get(plan.symbol, {}).get(session)
            if day is None:
                skipped["no_bars"] += 1
                continue
            trade, reason = simulate_plan(plan, day, params, cfg)
            if trade is None:
                skipped[reason] += 1
            else:
                trades.append(trade)
    frame = pd.DataFrame(trades)
    return BacktestResult(trades=frame, skipped=skipped, sessions=list(use), plans_made=plans_made)
