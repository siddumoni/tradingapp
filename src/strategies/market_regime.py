
"""Simple market regime filter based on an index's latest completed candles."""
from __future__ import annotations

from typing import Literal

import pandas as pd

Regime = Literal["BULLISH", "BEARISH", "CHOPPY", "UNKNOWN"]


def classify_market_regime(index_candles: pd.DataFrame) -> Regime:
    """Classify index direction using EMA alignment, slope and price.

    When VWAP is valid, it is included as confirmation.
    When VWAP is unavailable (for example, Yahoo index candles with zero
    volume), use price relative to EMA 21 instead.

    This is a heuristic filter, not a prediction or guarantee.
    """
    required = {"close", "ema_9", "ema_21"}
    if index_candles.empty or not required.issubset(index_candles.columns):
        return "UNKNOWN"

    # Need at least three usable EMA-9 values to assess the recent slope.
    ema9 = pd.to_numeric(index_candles["ema_9"], errors="coerce")
    tail = ema9.dropna().tail(4)
    if len(tail) < 3:
        return "UNKNOWN"

    row = index_candles.iloc[-1]

    try:
        close = float(row["close"])
        ema_9 = float(row["ema_9"])
        ema_21 = float(row["ema_21"])
    except (TypeError, ValueError):
        return "UNKNOWN"

    if not all(pd.notna(value) for value in (close, ema_9, ema_21)):
        return "UNKNOWN"

    slope_up = tail.iloc[-1] > tail.iloc[0]
    slope_down = tail.iloc[-1] < tail.iloc[0]

    # Determine whether VWAP is usable. Do not manufacture one if it is missing.
    vwap_available = "vwap" in index_candles.columns
    vwap = pd.to_numeric(
        pd.Series([row.get("vwap")]), errors="coerce"
    ).iloc[0]
    vwap_available = vwap_available and pd.notna(vwap)

    if vwap_available:
        bullish_price_confirmation = close > vwap
        bearish_price_confirmation = close < vwap
    else:
        # Price-based fallback for index feeds without usable volume.
        bullish_price_confirmation = close > ema_21
        bearish_price_confirmation = close < ema_21

    bullish = (
        bullish_price_confirmation
        and ema_9 > ema_21
        and slope_up
    )
    bearish = (
        bearish_price_confirmation
        and ema_9 < ema_21
        and slope_down
    )

    if bullish:
        return "BULLISH"
    if bearish:
        return "BEARISH"
    return "CHOPPY"
