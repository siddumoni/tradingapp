"""Previous-session levels and classic pivot points from daily OHLCV candles."""
from __future__ import annotations
import pandas as pd


def previous_session_levels(daily_candles: pd.DataFrame) -> dict[str, float]:
    """Return previous completed daily candle's high, low, close and classic pivots.

    Input must be sorted or sortable by timestamp and contain high/low/close.
    The caller is responsible for excluding an incomplete current-day candle.
    """
    required = {"high", "low", "close"}
    missing = required.difference(daily_candles.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(daily_candles) < 2:
        raise ValueError("At least two daily candles are required")
    df = daily_candles.sort_index()
    prev = df.iloc[-2]
    high, low, close = float(prev["high"]), float(prev["low"]), float(prev["close"])
    if min(high, low, close) <= 0 or high < low:
        raise ValueError("Invalid previous-session OHLC values")
    pivot = (high + low + close) / 3.0
    return {
        "previous_high": high,
        "previous_low": low,
        "previous_close": close,
        "pivot": pivot,
        "r1": 2 * pivot - low,
        "s1": 2 * pivot - high,
        "r2": pivot + (high - low),
        "s2": pivot - (high - low),
    }
