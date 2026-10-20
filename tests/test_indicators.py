import numpy as np
import pandas as pd
import pytest

from src.indicators import add_indicators


def make_candles(n=80):
    idx = pd.date_range("2026-10-12 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    close = np.linspace(100, 120, n)
    return pd.DataFrame({
        "open": close - 0.1, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n, 1000),
    }, index=idx)


def test_add_indicators_adds_expected_columns():
    result = add_indicators(make_candles())
    for col in ["ema_9", "ema_21", "rsi_14", "macd", "macd_signal",
                "macd_histogram", "atr_14", "vwap", "volume_sma_20"]:
        assert col in result.columns
    assert result["ema_9"].notna().any()
    assert result["atr_14"].notna().any()


def test_missing_column_raises():
    with pytest.raises(ValueError, match="Missing required"):
        add_indicators(pd.DataFrame({"close": [100]}))


def test_negative_volume_raises():
    df = make_candles()
    df.iloc[0, df.columns.get_loc("volume")] = -1
    with pytest.raises(ValueError, match="Volume"):
        add_indicators(df)


def test_vwap_resets_each_session():
    first = make_candles(3)
    second = make_candles(3)
    second.index = pd.date_range("2026-10-13 09:15", periods=3, freq="5min", tz="Asia/Kolkata")
    df = pd.concat([first, second])
    out = add_indicators(df)
    assert out.iloc[3]["vwap"] == pytest.approx(
        (out.iloc[3]["high"] + out.iloc[3]["low"] + out.iloc[3]["close"]) / 3
    )
