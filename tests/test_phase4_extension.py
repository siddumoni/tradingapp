import pandas as pd
import pytest

from src.strategies.levels import previous_session_levels
from src.strategies.signal_planner import plan_atr_levels
from src.strategies.vwap_pullback import vwap_pullback_signal


def test_previous_session_levels_and_pivots():
    idx = pd.date_range("2026-10-15", periods=2, freq="D")
    df = pd.DataFrame({
        "high": [110, 120], "low": [90, 100], "close": [105, 110]
    }, index=idx)
    result = previous_session_levels(df)
    assert result["previous_high"] == 110
    assert result["previous_low"] == 90
    assert result["previous_close"] == 105
    assert result["pivot"] == pytest.approx((110 + 90 + 105) / 3)


def test_previous_session_requires_two_days():
    df = pd.DataFrame({"high":[100], "low":[90], "close":[95]})
    with pytest.raises(ValueError, match="two daily candles"):
        previous_session_levels(df)


def test_atr_levels_long():
    result = plan_atr_levels("LONG", entry=100, atr=2)
    assert result.stop_loss == 98
    assert result.target_1 == 103
    assert result.target_2 == 104
    assert result.gross_rr_t1 == pytest.approx(1.5)


def test_atr_levels_short():
    result = plan_atr_levels("SHORT", entry=100, atr=2)
    assert result.stop_loss == 102
    assert result.target_1 == 97
    assert result.target_2 == 96


def test_atr_levels_reject_invalid_multiples():
    with pytest.raises(ValueError):
        plan_atr_levels("LONG", entry=100, atr=2, stop_atr_multiple=2, target_1_atr_multiple=1.5)


def test_vwap_pullback_returns_none_without_required_columns():
    assert vwap_pullback_signal("TEST", pd.DataFrame({"close":[100]})) is None
