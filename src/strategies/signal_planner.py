"""ATR-based indicative trade levels. Does not size positions or execute orders."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PlannedLevels:
    side: Literal["LONG", "SHORT"]
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    gross_risk_per_share: float
    gross_reward_t1_per_share: float
    gross_reward_t2_per_share: float
    gross_rr_t1: float
    gross_rr_t2: float


def plan_atr_levels(
    side: Literal["LONG", "SHORT"],
    entry: float,
    atr: float,
    stop_atr_multiple: float = 1.0,
    target_1_atr_multiple: float = 1.5,
    target_2_atr_multiple: float = 2.0,
) -> PlannedLevels:
    """Compute gross price levels using ATR multiples.

    These are indicative gross levels; actual trade approval must still pass
    through the risk engine and cost calculator.
    """
    if side not in ("LONG", "SHORT"):
        raise ValueError("side must be LONG or SHORT")
    if entry <= 0 or atr <= 0:
        raise ValueError("entry and ATR must be positive")
    if not (0 < stop_atr_multiple < target_1_atr_multiple <= target_2_atr_multiple):
        raise ValueError("Require 0 < stop multiple < target1 multiple <= target2 multiple")
    risk = atr * stop_atr_multiple
    reward1 = atr * target_1_atr_multiple
    reward2 = atr * target_2_atr_multiple
    if side == "LONG":
        sl, t1, t2 = entry - risk, entry + reward1, entry + reward2
    else:
        sl, t1, t2 = entry + risk, entry - reward1, entry - reward2
    if min(sl, t1, t2) <= 0:
        raise ValueError("Calculated price levels must remain positive")
    return PlannedLevels(
        side, entry, sl, t1, t2, risk, reward1, reward2,
        reward1 / risk, reward2 / risk,
    )
