
import pandas as pd
import pytest

from src.strategies.market_regime import classify_market_regime


def make_index_frame(direction="bullish", include_vwap=False):
    if direction == "bullish":
        ema9 = [100, 101, 102, 103]
        ema21 = [99, 99.5, 100, 101]
        close = [104, 104, 105, 106]
    elif direction == "bearish":
        ema9 = [103, 102, 101, 100]
        ema21 = [104, 104, 103.5, 103]
        close = [102, 102, 101, 99]
    else:
        ema9 = [100, 101, 100, 101]
        ema21 = [100, 100, 100, 100]
        close = [100, 101, 100, 101]

    data = {
        "close": close,
        "ema_9": ema9,
        "ema_21": ema21,
    }
    if include_vwap:
        data["vwap"] = [100, 100, 100, 100]

    return pd.DataFrame(data)


def test_zero_volume_index_can_use_price_based_bullish_fallback():
    frame = make_index_frame("bullish")
    assert classify_market_regime(frame) == "BULLISH"


def test_zero_volume_index_can_use_price_based_bearish_fallback():
    frame = make_index_frame("bearish")
    assert classify_market_regime(frame) == "BEARISH"


def test_valid_vwap_is_used_for_confirmation():
    frame = make_index_frame("bullish", include_vwap=True)
    frame.loc[3, "vwap"] = 107
    assert classify_market_regime(frame) == "CHOPPY"


def test_missing_ema_values_fail_safely():
    frame = make_index_frame("bullish")
    frame.loc[3, "ema_21"] = float("nan")
    assert classify_market_regime(frame) == "UNKNOWN"


def test_empty_frame_returns_unknown():
    assert classify_market_regime(pd.DataFrame()) == "UNKNOWN"
