"""Regression tests for Opening Range session validation."""
import pandas as pd
import pytest

from src.strategies.opening_range import opening_range_signal


def make_session(times=None):
    times = times or [
        "09:15", "09:20", "09:25", "09:30", "09:35",
    ]
    index = pd.to_datetime(
        [f"2026-10-19 {t}:00" for t in times]
    ).tz_localize("Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 101.0, 102.0][:len(index)],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0][:len(index)],
            "low": [99.0, 99.0, 99.0, 100.0, 101.0][:len(index)],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0][:len(index)],
            "volume": [1000, 1000, 1000, 2000, 2000][:len(index)],
        },
        index=index,
    )


def test_rejects_missing_first_opening_candle():
    df = make_session(["09:20", "09:25", "09:30", "09:35"])
    assert opening_range_signal("TEST", df, market_regime="BULLISH") is None


def test_rejects_gap_inside_first_three_opening_candles():
    df = make_session(["09:15", "09:20", "09:30", "09:35"])
    assert opening_range_signal("TEST", df, market_regime="BULLISH") is None


def test_rejects_duplicate_timestamps():
    df = make_session()
    df = pd.concat([df, df.iloc[[2]]]).sort_index()
    assert opening_range_signal("TEST", df, market_regime="BULLISH") is None


def test_rejects_incomplete_opening_range():
    df = make_session(["09:15", "09:20", "09:25"])
    assert opening_range_signal("TEST", df, market_regime="BULLISH") is None


def test_rejects_malformed_opening_ohlc():
    df = make_session()
    df.loc[df.index[1], "high"] = 98
    assert opening_range_signal("TEST", df, market_regime="BULLISH") is None


def test_valid_session_can_produce_breakout_signal():
    df = make_session()
    # Make the final completed candle break above the first-three-bar high,
    # with volume above the preceding-bar average.
    df.iloc[-1, df.columns.get_loc("close")] = 104.5
    df.iloc[-1, df.columns.get_loc("high")] = 105
    df.iloc[-1, df.columns.get_loc("volume")] = 3000
    signal = opening_range_signal("TEST", df, market_regime="BULLISH")
    assert signal is not None
    assert signal.side == "LONG"
