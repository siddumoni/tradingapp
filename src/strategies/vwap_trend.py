"""VWAP + EMA/momentum confirmation strategy."""
from __future__ import annotations
import pandas as pd
from .models import StrategySignal
from .market_regime import Regime


def vwap_trend_signal(symbol: str, candles: pd.DataFrame, market_regime: Regime = "UNKNOWN") -> StrategySignal | None:
    """Return a trend-continuation signal from the latest completed candle."""
    required = {"close", "vwap", "ema_9", "ema_21", "rsi_14", "macd_histogram", "atr_14"}
    if candles.empty or not required.issubset(candles.columns):
        return None
    row = candles.iloc[-1]
    if row[list(required)].isna().any():
        return None
    close = float(row["close"])
    vwap = float(row["vwap"])
    ema9 = float(row["ema_9"])
    ema21 = float(row["ema_21"])
    rsi = float(row["rsi_14"])
    hist = float(row["macd_histogram"])
    atr = float(row["atr_14"])
    if atr <= 0:
        return None

    long_ok = close > vwap and ema9 > ema21 and 50 <= rsi <= 70 and hist > 0
    short_ok = close < vwap and ema9 < ema21 and 30 <= rsi <= 50 and hist < 0

    if long_ok and market_regime != "BEARISH":
        return StrategySignal(
            symbol=symbol, side="LONG", strategy="vwap_ema_momentum",
            trigger=f"Enter only if a completed 5-minute candle confirms above ₹{close:.2f}",
            invalidation=f"Invalidate if a completed candle closes below VWAP ₹{vwap:.2f} or EMA21 ₹{ema21:.2f}",
            confidence=70 if market_regime == "BULLISH" else 58,
            reason=f"Price above VWAP; EMA9 > EMA21; RSI={rsi:.1f}; MACD histogram positive; ATR14=₹{atr:.2f}.",
        )
    if short_ok and market_regime != "BULLISH":
        return StrategySignal(
            symbol=symbol, side="SHORT", strategy="vwap_ema_momentum",
            trigger=f"Enter only if a completed 5-minute candle confirms below ₹{close:.2f}",
            invalidation=f"Invalidate if a completed candle closes above VWAP ₹{vwap:.2f} or EMA21 ₹{ema21:.2f}",
            confidence=70 if market_regime == "BEARISH" else 58,
            reason=f"Price below VWAP; EMA9 < EMA21; RSI={rsi:.1f}; MACD histogram negative; ATR14=₹{atr:.2f}.",
        )
    return None
