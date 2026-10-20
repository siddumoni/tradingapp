"""Advisory-only trade validation, position sizing, and net R:R calculations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import floor, isfinite
from pathlib import Path
from typing import Literal

import yaml

from src.cost_calculator import (
    ChargeRates, CostBreakdown, estimate_intraday_round_trip_cost,
)

TradeSide = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class RiskSettings:
    capital_inr: float = 20_000.0
    risk_per_trade_inr: float = 200.0
    minimum_net_risk_reward: float = 1.5
    require_margin_verification: bool = True
    skip_if_margin_unavailable: bool = True
    slippage_bps_per_side: float = 5.0


@dataclass(frozen=True)
class TradeAssessment:
    accepted: bool
    reason: str
    side: TradeSide
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    quantity: int
    estimated_margin_inr: float | None
    gross_risk_inr: float
    net_risk_inr: float
    gross_reward_t1_inr: float
    gross_reward_t2_inr: float
    net_reward_t1_inr: float
    net_reward_t2_inr: float
    net_rr_t1: float
    net_rr_t2: float
    costs_at_stop: CostBreakdown | None
    costs_at_t1: CostBreakdown | None
    costs_at_t2: CostBreakdown | None
    slippage_at_stop_inr: float = 0.0
    slippage_at_t1_inr: float = 0.0
    slippage_at_t2_inr: float = 0.0


def assess_trade(
    *,
    side: TradeSide,
    entry_price: float,
    stop_loss: float,
    target_1: float,
    target_2: float,
    settings: RiskSettings = RiskSettings(),
    estimated_margin_per_share: float | None,
    charge_rates: ChargeRates = ChargeRates(),
) -> TradeAssessment:
    """Assess a proposed equity MIS signal without placing any orders.

    Quantity starts with floor(risk_budget / stop distance), then is reduced if
    round-trip costs would push the stop-out loss over budget or capital/margin
    constraints. Margin must be supplied by a trusted read-only estimate when
    required; missing margin data fails closed.
    """
    prices = (entry_price, stop_loss, target_1, target_2)
    if side not in ("LONG", "SHORT"):
        raise ValueError("side must be 'LONG' or 'SHORT'")
    if not all(isfinite(p) and p > 0 for p in prices):
        raise ValueError("all prices must be finite and greater than zero")
    if settings.capital_inr <= 0 or settings.risk_per_trade_inr <= 0:
        raise ValueError("capital and risk_per_trade_inr must be positive")
    if settings.minimum_net_risk_reward <= 0:
        raise ValueError("minimum_net_risk_reward must be positive")
    if not isfinite(settings.slippage_bps_per_side) or settings.slippage_bps_per_side < 0:
        raise ValueError("slippage_bps_per_side must be finite and non-negative")

    if side == "LONG":
        valid_levels = stop_loss < entry_price < target_1 <= target_2
    else:
        valid_levels = target_2 <= target_1 < entry_price < stop_loss
    if not valid_levels:
        raise ValueError("invalid price levels for side; check stop-loss and target ordering")

    if estimated_margin_per_share is None or estimated_margin_per_share <= 0:
        if settings.require_margin_verification or settings.skip_if_margin_unavailable:
            return _rejected(side, prices, "Margin estimate unavailable; skipped by fail-closed policy.")
        # Explicitly permitted only when the caller disables fail-closed checks.
        estimated_margin_per_share = entry_price

    if not isfinite(estimated_margin_per_share) or estimated_margin_per_share <= 0:
        return _rejected(side, prices, "Margin estimate invalid; skipped by fail-closed policy.")

    risk_per_share = abs(entry_price - stop_loss)
    # Cash-only notional cap is mandatory; a low MIS margin estimate must not
    # permit a position whose entry notional exceeds available capital.
    risk_qty = floor(settings.risk_per_trade_inr / risk_per_share)
    cash_qty = floor(settings.capital_inr / entry_price)
    margin_qty = floor(settings.capital_inr / estimated_margin_per_share)
    qty = min(risk_qty, cash_qty, margin_qty)
    if qty < 1:
        return _rejected(side, prices, "Quantity is zero after risk and capital/margin limits.")

    # Reduce quantity until worst-case loss including estimated charges remains
    # within the configured risk budget. This is stricter than price-only sizing.
    while qty > 0:
        stop_costs = estimate_intraday_round_trip_cost(
            side=side, entry_price=entry_price, exit_price=stop_loss, quantity=qty, rates=charge_rates
        )
        gross_risk = risk_per_share * qty
        stop_slippage = (entry_price + stop_loss) * qty * settings.slippage_bps_per_side / 10_000.0
        net_risk = gross_risk + stop_costs.total + stop_slippage
        if net_risk <= settings.risk_per_trade_inr:
            break
        qty -= 1
    if qty < 1:
        return _rejected(side, prices, "No quantity fits the risk budget after estimated round-trip costs.")

    margin = qty * estimated_margin_per_share
    if margin > settings.capital_inr:
        return _rejected(side, prices, "Estimated required margin exceeds available capital.")

    stop_costs = estimate_intraday_round_trip_cost(
        side=side, entry_price=entry_price, exit_price=stop_loss, quantity=qty, rates=charge_rates
    )
    t1_costs = estimate_intraday_round_trip_cost(
        side=side, entry_price=entry_price, exit_price=target_1, quantity=qty, rates=charge_rates
    )
    t2_costs = estimate_intraday_round_trip_cost(
        side=side, entry_price=entry_price, exit_price=target_2, quantity=qty, rates=charge_rates
    )
    gross_risk = risk_per_share * qty
    slippage_stop = (entry_price + stop_loss) * qty * settings.slippage_bps_per_side / 10_000.0
    slippage_t1 = (entry_price + target_1) * qty * settings.slippage_bps_per_side / 10_000.0
    slippage_t2 = (entry_price + target_2) * qty * settings.slippage_bps_per_side / 10_000.0
    net_risk = gross_risk + stop_costs.total + slippage_stop
    gross_t1 = abs(target_1 - entry_price) * qty
    gross_t2 = abs(target_2 - entry_price) * qty
    net_t1 = gross_t1 - t1_costs.total - slippage_t1
    net_t2 = gross_t2 - t2_costs.total - slippage_t2
    rr_t1 = net_t1 / net_risk if net_risk > 0 else 0.0
    rr_t2 = net_t2 / net_risk if net_risk > 0 else 0.0

    if rr_t1 < settings.minimum_net_risk_reward:
        return TradeAssessment(
            accepted=False,
            reason=f"Rejected: Target 1 net R:R is below {settings.minimum_net_risk_reward:.2f}.",
            side=side, entry_price=entry_price, stop_loss=stop_loss,
            target_1=target_1, target_2=target_2, quantity=qty,
            estimated_margin_inr=margin, gross_risk_inr=gross_risk,
            net_risk_inr=net_risk, gross_reward_t1_inr=gross_t1,
            gross_reward_t2_inr=gross_t2, net_reward_t1_inr=net_t1,
            net_reward_t2_inr=net_t2, net_rr_t1=rr_t1, net_rr_t2=rr_t2,
            costs_at_stop=stop_costs, costs_at_t1=t1_costs, costs_at_t2=t2_costs,
            slippage_at_stop_inr=slippage_stop, slippage_at_t1_inr=slippage_t1,
            slippage_at_t2_inr=slippage_t2,
        )

    return TradeAssessment(
        accepted=True,
        reason="Accepted by configured risk, margin, and net R:R checks; this is not a trade recommendation.",
        side=side, entry_price=entry_price, stop_loss=stop_loss,
        target_1=target_1, target_2=target_2, quantity=qty,
        estimated_margin_inr=margin, gross_risk_inr=gross_risk,
        net_risk_inr=net_risk, gross_reward_t1_inr=gross_t1,
        gross_reward_t2_inr=gross_t2, net_reward_t1_inr=net_t1,
        net_reward_t2_inr=net_t2, net_rr_t1=rr_t1, net_rr_t2=rr_t2,
        costs_at_stop=stop_costs, costs_at_t1=t1_costs, costs_at_t2=t2_costs,
        slippage_at_stop_inr=slippage_stop, slippage_at_t1_inr=slippage_t1,
        slippage_at_t2_inr=slippage_t2,
    )


def _rejected(side: TradeSide, prices: tuple[float, ...], reason: str) -> TradeAssessment:
    entry, stop, t1, t2 = prices
    return TradeAssessment(
        accepted=False, reason=reason, side=side, entry_price=entry,
        stop_loss=stop, target_1=t1, target_2=t2, quantity=0,
        estimated_margin_inr=None, gross_risk_inr=0.0, net_risk_inr=0.0,
        gross_reward_t1_inr=0.0, gross_reward_t2_inr=0.0,
        net_reward_t1_inr=0.0, net_reward_t2_inr=0.0,
        net_rr_t1=0.0, net_rr_t2=0.0, costs_at_stop=None,
        costs_at_t1=None, costs_at_t2=None,
    )


def load_phase2_settings(
    config_path: Path | None = None,
) -> tuple[RiskSettings, ChargeRates]:
    """Load risk and charge settings from config/settings.yaml."""
    path = config_path or Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        account = raw.get("account", {})
        risk = raw.get("risk", {})
        cost = raw.get("costs", {})
        settings = RiskSettings(
            capital_inr=float(account.get("capital_inr", 20_000)),
            risk_per_trade_inr=float(account.get("risk_per_trade_inr", 200)),
            minimum_net_risk_reward=float(account.get("minimum_net_risk_reward", 1.5)),
            require_margin_verification=bool(risk.get("require_margin_verification", True)),
            skip_if_margin_unavailable=bool(risk.get("skip_if_margin_unavailable", True)),
        )
        rates = ChargeRates(
            brokerage_rate=Decimal(str(cost.get("brokerage_rate", "0.0003"))),
            brokerage_cap_per_order=Decimal(str(cost.get("brokerage_cap_per_order", "20"))),
            stt_sell_rate=Decimal(str(cost.get("stt_sell_rate", "0.00025"))),
            nse_transaction_rate=Decimal(str(cost.get("nse_transaction_rate", "0.0000307"))),
            sebi_rate=Decimal(str(cost.get("sebi_rate", "0.000001"))),
            gst_rate=Decimal(str(cost.get("gst_rate", "0.18"))),
            stamp_buy_rate=Decimal(str(cost.get("stamp_buy_rate", "0.00003"))),
            ipft_rate=Decimal(str(cost.get("ipft_rate", "0"))),
        )
        return settings, rates
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not load Phase 2 settings from {path}: {exc}") from exc
