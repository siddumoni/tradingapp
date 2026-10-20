import numpy as np
import pandas as pd
import pytest

from src.strategy_backtest import candles_to_frame, make_signal_callback, backtest_strategy


def sample_candles(n=100):
    idx = pd.date_range("2026-10-12 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    close = np.linspace(100, 120, n)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1000),
    }, index=idx)


def test_candles_to_frame_converts_provider_shape():
    frame = candles_to_frame([{
        "date": pd.Timestamp("2026-10-12 09:15", tz="Asia/Kolkata"),
        "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000,
    }])
    assert len(frame) == 1
    assert frame.iloc[0]["close"] == 100.5


def test_candles_to_frame_empty_has_expected_columns():
    frame = candles_to_frame([])
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_candles_to_frame_rejects_missing_date():
    with pytest.raises(ValueError, match="date"):
        candles_to_frame([{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}])


def test_callback_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unsupported strategy"):
        make_signal_callback("TEST", "not_a_strategy")


def test_backtest_strategy_returns_summary_for_historical_frame():
    trades, summary = backtest_strategy("TEST", sample_candles(), "vwap_trend")
    assert summary["symbol"] == "TEST"
    assert summary["strategy"] == "vwap_trend"
    assert summary["bars"] == 72
    assert summary["bars_before_session_filter"] == 100
    assert "net_pnl" in summary
    assert isinstance(trades, pd.DataFrame)


def test_backtest_strategy_rejects_empty_data():
    with pytest.raises(ValueError, match="No candles"):
        backtest_strategy("TEST", pd.DataFrame(), "vwap_trend")
