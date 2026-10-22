"""Stocks-in-Play opening-range breakout (research candidate, advisory-only).

Idea (adapted from published work on "stocks in play"; see docs/SIP_ORB_RESEARCH.md):
on any day only a few stocks have unusual interest, visible as abnormal volume in
the first minutes of trading. Trade only those, in the direction of the opening
bar, with a stop sized from the *daily* ATR, and hold until the stop or the
end-of-day square-off.

This module is pure: it computes features and day plans from candles. The exact
same code path is used by the backtest (src/orb_backtest.py) and by live planning,
so a backtest cannot silently use information a live run would not have.

Nothing here places, modifies or cancels orders.

No-lookahead rule: every feature for session D (relative volume, ATR) is computed
only from sessions strictly before D, plus the opening bars of D themselves, which
are complete by the time a plan is made.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
FIRST_BAR = pd.Timedelta(hours=9, minutes=15)
BAR = pd.Timedelta(minutes=5)

Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class SipOrbParams:
    """Strategy parameters. Defaults are fixed in advance, not fitted to data."""

    opening_bars: int = 1            # 5-minute bars forming the opening range
    rvol_lookback: int = 14          # prior sessions used for the volume baseline
    atr_lookback: int = 14           # prior sessions used for daily ATR
    min_rvol: float = 1.5            # opening volume vs. its own recent average
    top_n: int = 5                   # at most this many plans per session
    stop_atr_fraction: float = 0.25  # stop distance = fraction * daily ATR
    min_price: float = 50.0
    min_atr_pct: float = 0.01        # daily ATR must be at least 1% of price
    entry_cutoff: str = "14:45"      # no new entries from this bar start onward
    square_off_bar: str = "15:10"    # exit at the close of the bar starting here
    max_cost_fraction_of_risk: float = 0.35  # skip if costs at stop exceed this share of risk

    def __post_init__(self) -> None:
        if self.opening_bars < 1:
            raise ValueError("opening_bars must be >= 1")
        if self.rvol_lookback < 2 or self.atr_lookback < 2:
            raise ValueError("lookbacks must be >= 2")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if self.stop_atr_fraction <= 0:
            raise ValueError("stop_atr_fraction must be positive")
        if not 0 < self.max_cost_fraction_of_risk <= 1:
            raise ValueError("max_cost_fraction_of_risk must be in (0, 1]")


@dataclass(frozen=True)
class DayPlan:
    """A conditional plan for one symbol and session. Not an order."""

    symbol: str
    session: date
    side: Side
    trigger: float          # stop-entry price: opening-range high (LONG) or low (SHORT)
    stop_distance: float    # price distance from the actual fill to the stop
    rvol: float
    atr: float
    or_high: float
    or_low: float


def _to_ist_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_index()
    idx = pd.DatetimeIndex(out.index)
    out.index = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    return out


def compute_features(frame: pd.DataFrame, params: SipOrbParams) -> pd.DataFrame:
    """Return one row per session with the features needed to build a plan.

    Columns: or_open, or_close, or_high, or_low, or_volume, day_high, day_low,
    day_close, rvol, atr. Sessions whose opening bars are not exactly the expected
    consecutive 5-minute bars starting 09:15 are dropped rather than guessed.
    rvol and atr use only earlier sessions (shifted), so they carry no lookahead.
    """
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns) or frame.empty:
        return pd.DataFrame()
    df = _to_ist_index(frame)
    if df.index.has_duplicates:
        raise ValueError("Duplicate candle timestamps")

    rows: list[dict] = []
    n = params.opening_bars
    # Convert once, then slice per session with numpy: per-session pandas calls were the
    # dominant cost of a backtest.
    cols = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in sorted(required)}
    day_keys = np.array(df.index.date)
    minutes = (df.index.hour * 60 + df.index.minute).to_numpy()
    starts = np.flatnonzero(np.concatenate(([True], day_keys[1:] != day_keys[:-1])))
    ends = np.concatenate((starts[1:], [len(df)]))
    expected = [int(FIRST_BAR.total_seconds() // 60) + 5 * i for i in range(n)]
    for a, b in zip(starts, ends):
        if b - a < n or list(minutes[a:a + n]) != expected:
            continue
        o, h, l, c, v = (cols[k][a:a + n] for k in ("open", "high", "low", "close", "volume"))
        if any(np.isnan(x).any() for x in (o, h, l, c, v)) or min(o.min(), h.min(), l.min(), c.min()) <= 0 or v.min() < 0:
            continue
        rows.append({
            "session": day_keys[a],
            "or_open": float(o[0]),
            "or_close": float(c[-1]),
            "or_high": float(h.max()),
            "or_low": float(l.min()),
            "or_volume": float(v.sum()),
            "day_high": float(np.fmax.reduce(cols["high"][a:b])),
            "day_low": float(np.fmin.reduce(cols["low"][a:b])),
            "day_close": float(cols["close"][b - 1]),
        })
    if not rows:
        return pd.DataFrame()
    daily = pd.DataFrame(rows).set_index("session").sort_index()

    baseline = daily["or_volume"].shift(1).rolling(params.rvol_lookback, min_periods=params.rvol_lookback).mean()
    daily["rvol"] = daily["or_volume"] / baseline.where(baseline > 0)

    prev_close = daily["day_close"].shift(1)
    true_range = pd.concat(
        [
            daily["day_high"] - daily["day_low"],
            (daily["day_high"] - prev_close).abs(),
            (daily["day_low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    daily["atr"] = true_range.shift(1).rolling(params.atr_lookback, min_periods=params.atr_lookback).mean()
    return daily


def select_day_plans(
    features_by_symbol: dict[str, pd.DataFrame],
    session: date,
    params: SipOrbParams,
    *,
    side_override: dict[str, Side] | None = None,
) -> list[DayPlan]:
    """Rank symbols by relative opening volume and return up to ``top_n`` plans.

    ``side_override`` exists only for the placebo test in the validator: it lets
    the direction be randomised while every other rule stays identical.
    """
    candidates: list[DayPlan] = []
    for symbol, feats in features_by_symbol.items():
        if feats is None or feats.empty or session not in feats.index:
            continue
        row = feats.loc[session]
        rvol, atr = row["rvol"], row["atr"]
        if pd.isna(rvol) or pd.isna(atr) or atr <= 0 or rvol < params.min_rvol:
            continue
        price = float(row["or_close"])
        if price < params.min_price or atr / price < params.min_atr_pct:
            continue
        if side_override is not None and symbol in side_override:
            side: Side = side_override[symbol]
        elif row["or_close"] > row["or_open"]:
            side = "LONG"
        elif row["or_close"] < row["or_open"]:
            side = "SHORT"
        else:
            continue  # a doji opening bar gives no direction
        trigger = float(row["or_high"] if side == "LONG" else row["or_low"])
        candidates.append(DayPlan(
            symbol=symbol, session=session, side=side, trigger=trigger,
            stop_distance=float(params.stop_atr_fraction * atr), rvol=float(rvol),
            atr=float(atr), or_high=float(row["or_high"]), or_low=float(row["or_low"]),
        ))
    candidates.sort(key=lambda p: (-p.rvol, p.symbol))
    return candidates[: params.top_n]
