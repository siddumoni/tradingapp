import pandas as pd

from src.strategies.market_regime import classify_market_regime
from src.strategies.opening_range import opening_range_signal
from src.strategies.vwap_trend import vwap_trend_signal


def test_regime_unknown_for_empty_data():
    assert classify_market_regime(pd.DataFrame()) == "UNKNOWN"


def test_orb_waits_for_four_bars():
    idx = pd.date_range("2026-10-12 09:15", periods=3, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame({"open":[100]*3,"high":[101]*3,"low":[99]*3,"close":[100]*3,"volume":[1000]*3}, index=idx)
    assert opening_range_signal("TEST", df) is None


def test_orb_rejects_long_when_market_bearish():
    idx = pd.date_range("2026-10-12 09:15", periods=5, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame({
        "open":[100,100,100,100,103], "high":[101,101,101,104,105],
        "low":[99,99,99,100,102], "close":[100,100,100,103,104],
        "volume":[1000,1000,1000,1000,2000],
    }, index=idx)
    # Baseline volume includes prior candles; last bar is above range but volume is confirmed.
    assert opening_range_signal("TEST", df, market_regime="BEARISH") is None


def test_vwap_signal_requires_indicator_columns():
    assert vwap_trend_signal("TEST", pd.DataFrame({"close":[100]})) is None


def test_signal_confidence_is_bounded():
    from src.strategies.models import StrategySignal
    import pytest
    with pytest.raises(ValueError):
        StrategySignal("TEST", "LONG", "test", "trigger", "invalid", 101, "reason")
