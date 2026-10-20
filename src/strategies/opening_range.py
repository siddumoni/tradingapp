"""Opening-range breakout strategy using completed 5-minute candles."""
from __future__ import annotations

import pandas as pd

from .models import StrategySignal
from .market_regime import Regime


_OPENING_TIMES = ("09:15", "09:20", "09:25")


def opening_range_signal(
    symbol: str,
    candles: pd.DataFrame,
    market_regime: Regime = "UNKNOWN",
    volume_multiple: float = 1.2,
) -> StrategySignal | None:
    """Signal an opening-range breakout only from a valid NSE session.

    The first three bars must be the consecutive 09:15, 09:20 and 09:25 IST
    bars. The final row is treated as the latest completed candle; callers must
    not pass a still-forming candle. Duplicate timestamps, missing opening bars,
    invalid opening OHLCV values, and signals before the opening range completes
    are rejected.
    """
    required = {"open", "high", "low", "close", "volume"}
    if len(candles) < 4 or not required.issubset(candles.columns):
        return None
    if not isinstance(candles.index, pd.DatetimeIndex):
        return None
    if candles.index.has_duplicates:
        return None

    df = candles.sort_index()
    dates = pd.Index(df.index.date)
    today = dates[-1]
    session = df.loc[dates == today]
    if len(session) < 4:
        return None

    # The strategy is defined for NSE regular-session timestamps in IST.
    # Do not silently treat a partial/misaligned feed as a valid opening range.
    session_index = pd.DatetimeIndex(session.index)
    if session_index.tz is not None:
        session_index = session_index.tz_convert("Asia/Kolkata")
    opening_times = tuple(ts.strftime("%H:%M") for ts in session_index[:3])
    if opening_times != _OPENING_TIMES:
        return None

    # Ensure the first three opening bars are consecutive five-minute bars.
    first_three = session.iloc[:3]
    if any(
        (session_index[i + 1] - session_index[i]) != pd.Timedelta(minutes=5)
        for i in range(2)
    ):
        return None

    # Reject malformed opening bars rather than computing a range from them.
    opening_ohlcv = first_three[list(required)].apply(pd.to_numeric, errors="coerce")
    if opening_ohlcv.isna().any().any():
        return None
    if (opening_ohlcv[["open", "high", "low", "close"]] <= 0).any().any():
        return None
    if (opening_ohlcv["volume"] < 0).any():
        return None
    if (
        (opening_ohlcv["high"] < opening_ohlcv[["open", "low", "close"]].max(axis=1))
        | (opening_ohlcv["low"] > opening_ohlcv[["open", "high", "close"]].min(axis=1))
    ).any():
        return None

    opening = first_three
    latest = session.iloc[-1]
    latest_values = pd.to_numeric(latest[list(required)], errors="coerce")
    if latest_values.isna().any():
        return None
    if float(latest_values["volume"]) < 0:
        return None
    if float(latest_values["open"]) <= 0 or float(latest_values["high"]) <= 0 or float(latest_values["low"]) <= 0 or float(latest_values["close"]) <= 0:
        return None

    # Only allow a signal after the opening range has completed at 09:30.
    latest_time = session_index[-1].strftime("%H:%M")
    if latest_time < "09:30":
        return None

    range_high = float(opening["high"].max())
    range_low = float(opening["low"].min())
    prior_volume = session["volume"].iloc[:-1].tail(20)
    prior_volume = pd.to_numeric(prior_volume, errors="coerce").dropna()
    baseline = float(prior_volume.mean()) if not prior_volume.empty else 0.0
    if baseline <= 0 or float(latest_values["volume"]) < baseline * volume_multiple:
        return None

    close = float(latest_values["close"])
    if close > range_high and market_regime != "BEARISH":
        return StrategySignal(
            symbol=symbol, side="LONG", strategy="opening_range_breakout",
            trigger=f"Enter only after a completed 5-minute close above ₹{range_high:.2f}",
            invalidation=f"Setup invalid if price closes back below ₹{range_high:.2f}",
            confidence=65 if market_regime == "BULLISH" else 55,
            reason=f"Close ₹{close:.2f} broke opening-range high ₹{range_high:.2f} with volume confirmation; regime={market_regime}.",
        )
    if close < range_low and market_regime != "BULLISH":
        return StrategySignal(
            symbol=symbol, side="SHORT", strategy="opening_range_breakdown",
            trigger=f"Enter only after a completed 5-minute close below ₹{range_low:.2f}",
            invalidation=f"Setup invalid if price closes back above ₹{range_low:.2f}",
            confidence=65 if market_regime == "BEARISH" else 55,
            reason=f"Close ₹{close:.2f} broke opening-range low ₹{range_low:.2f} with volume confirmation; regime={market_regime}.",
        )
    return None
