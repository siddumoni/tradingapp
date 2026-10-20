"""Technical indicators for NSE intraday OHLCV data.

Input columns: open, high, low, close, volume. Index should be candle timestamps.
This module does not fetch data or place orders.
"""
from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def add_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with EMA9/21, RSI14, MACD, ATR14, cumulative session VWAP.

    The caller should pass candles for one symbol and preferably one interval.
    VWAP resets at each calendar session date represented by the index.
    """
    missing = REQUIRED_COLUMNS.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")
    if candles.empty:
        return candles.copy()

    df = candles.copy().sort_index()
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if (df[["high", "low", "close"]].dropna() <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (df["volume"].dropna() < 0).any():
        raise ValueError("Volume cannot be negative")

    df["ema_9"] = EMAIndicator(close=df["close"], window=9, fillna=False).ema_indicator()
    df["ema_21"] = EMAIndicator(close=df["close"], window=21, fillna=False).ema_indicator()
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14, fillna=False).rsi()

    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    # ta's ATR requires enough bars. Use a temporary valid frame and preserve warm-up NaNs.
    if len(df) >= 14:
        atr = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14, fillna=False
        )
        df["atr_14"] = atr.average_true_range()
    else:
        # ta's ATR implementation requires at least `window` observations.
        # Preserve a predictable NaN warm-up result for short test/history windows.
        df["atr_14"] = float("nan")

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    session_key = pd.Index(df.index.date, name="session")
    pv = typical_price * df["volume"]
    cumulative_pv = pv.groupby(session_key).cumsum()
    cumulative_volume = df["volume"].groupby(session_key).cumsum().replace(0, float("nan"))
    df["vwap"] = cumulative_pv / cumulative_volume

    df["volume_sma_20"] = df["volume"].rolling(20, min_periods=5).mean()
    return df
