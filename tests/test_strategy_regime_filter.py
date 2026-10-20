"""Tests for the optional CHOPPY/UNKNOWN regime filter in strategy backtests."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src import strategy_backtest as sb


@pytest.mark.parametrize("regime", ["CHOPPY", "UNKNOWN"])
def test_filter_rejects_nontrending_regimes_when_enabled(regime):
    assert sb._regime_is_filtered(regime, True) is True


@pytest.mark.parametrize("regime", ["BULLISH", "BEARISH"])
def test_filter_keeps_directional_regimes(regime):
    assert sb._regime_is_filtered(regime, True) is False


@pytest.mark.parametrize("regime", ["CHOPPY", "UNKNOWN", "BULLISH", "BEARISH"])
def test_filter_disabled_preserves_baseline(regime):
    assert sb._regime_is_filtered(regime, False) is False


def test_callback_returns_no_signal_in_choppy_regime_when_filter_enabled(monkeypatch):
    monkeypatch.setattr(sb, "classify_market_regime", lambda history: "CHOPPY")
    called = {"strategy": False}

    def fake_strategy(*args, **kwargs):
        called["strategy"] = True
        return SimpleNamespace(side="LONG", strategy="fake", confidence=60, reason="test")

    monkeypatch.setattr(sb, "vwap_trend_signal", fake_strategy)
    frame = pd.DataFrame(
        {"close": [100.0], "atr_14": [1.0]},
        index=pd.DatetimeIndex(["2026-10-19 10:00:00"], tz="Asia/Kolkata"),
    )
    callback = sb.make_signal_callback("TEST", "vwap_trend", reject_nontrending_regimes=True)
    assert callback(frame, 0) is None
    assert called["strategy"] is False


def test_callback_keeps_baseline_behavior_when_filter_disabled(monkeypatch):
    monkeypatch.setattr(sb, "classify_market_regime", lambda history: "CHOPPY")
    monkeypatch.setattr(
        sb,
        "vwap_trend_signal",
        lambda *args, **kwargs: SimpleNamespace(
            side="LONG", strategy="fake", confidence=60, reason="test"
        ),
    )
    monkeypatch.setattr(
        sb,
        "plan_atr_levels",
        lambda *args, **kwargs: SimpleNamespace(stop_loss=99.0, target_2=102.0),
    )
    frame = pd.DataFrame(
        {"close": [100.0], "atr_14": [1.0]},
        index=pd.DatetimeIndex(["2026-10-19 10:00:00"], tz="Asia/Kolkata"),
    )
    callback = sb.make_signal_callback("TEST", "vwap_trend", reject_nontrending_regimes=False)
    result = callback(frame, 0)
    assert result is not None
    assert result["side"] == "LONG"
