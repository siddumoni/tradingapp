from __future__ import annotations

import pytest

from src.risk_engine import RiskSettings, assess_trade


def test_missing_margin_fails_closed() -> None:
    result = assess_trade(side="LONG", entry_price=100, stop_loss=99,
        target_1=102, target_2=103, estimated_margin_per_share=None)
    assert not result.accepted and result.quantity == 0
    assert "Margin estimate unavailable" in result.reason


def test_cash_notional_caps_quantity_even_when_margin_is_low() -> None:
    settings = RiskSettings(capital_inr=1_000, risk_per_trade_inr=100)
    result = assess_trade(side="LONG", entry_price=100, stop_loss=90,
        target_1=120, target_2=130, settings=settings,
        estimated_margin_per_share=1)
    assert result.quantity > 0
    assert result.quantity * result.entry_price <= settings.capital_inr
    assert result.estimated_margin_inr <= settings.capital_inr


def test_net_risk_includes_charges_and_slippage() -> None:
    settings = RiskSettings(capital_inr=20_000, risk_per_trade_inr=100,
        slippage_bps_per_side=5)
    result = assess_trade(side="LONG", entry_price=100, stop_loss=99,
        target_1=103, target_2=104, settings=settings,
        estimated_margin_per_share=50)
    assert result.accepted
    assert result.net_risk_inr == pytest.approx(
        result.gross_risk_inr + result.costs_at_stop.total + result.slippage_at_stop_inr
    )
    assert result.net_risk_inr <= settings.risk_per_trade_inr
    assert result.slippage_at_stop_inr > 0


def test_net_rewards_subtract_costs_and_slippage() -> None:
    result = assess_trade(side="LONG", entry_price=100, stop_loss=99,
        target_1=103, target_2=104, estimated_margin_per_share=50)
    assert result.net_reward_t1_inr == pytest.approx(
        result.gross_reward_t1_inr - result.costs_at_t1.total - result.slippage_at_t1_inr
    )
    assert result.net_reward_t2_inr == pytest.approx(
        result.gross_reward_t2_inr - result.costs_at_t2.total - result.slippage_at_t2_inr
    )
    assert result.net_rr_t2 >= result.net_rr_t1


def test_short_trade_risk_and_slippage_are_bounded() -> None:
    result = assess_trade(side="SHORT", entry_price=100, stop_loss=101,
        target_1=97, target_2=96, estimated_margin_per_share=50)
    assert result.accepted
    assert result.net_risk_inr <= RiskSettings().risk_per_trade_inr
    assert result.slippage_at_stop_inr > 0


def test_invalid_slippage_rejected() -> None:
    settings = RiskSettings(slippage_bps_per_side=-1)
    with pytest.raises(ValueError, match="slippage_bps_per_side"):
        assess_trade(side="LONG", entry_price=100, stop_loss=99,
            target_1=102, target_2=103, settings=settings,
            estimated_margin_per_share=50)


def test_invalid_level_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid price levels"):
        assess_trade(side="LONG", entry_price=100, stop_loss=101,
            target_1=102, target_2=103, estimated_margin_per_share=50)


def test_net_rr_filter_rejects_targets_too_close_to_entry() -> None:
    result = assess_trade(side="LONG", entry_price=100, stop_loss=99,
        target_1=100.2, target_2=100.3, estimated_margin_per_share=50)
    assert not result.accepted
    assert "net R:R" in result.reason

@pytest.mark.parametrize("risk_budget", [100.0, 200.0, 500.0])
def test_risk_budget_is_configurable_without_a_hardcoded_ceiling(risk_budget: float) -> None:
    settings = RiskSettings(
        capital_inr=20_000,
        risk_per_trade_inr=risk_budget,
        minimum_net_risk_reward=1.5,
    )
    result = assess_trade(
        side="LONG", entry_price=100, stop_loss=99,
        target_1=103, target_2=104, settings=settings,
        estimated_margin_per_share=50,
    )
    assert result.accepted
    assert result.net_risk_inr <= risk_budget
