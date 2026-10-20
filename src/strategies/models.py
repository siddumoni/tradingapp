"""Shared strategy result models."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class StrategySignal:
    symbol: str
    side: Side
    strategy: str
    trigger: str
    invalidation: str
    confidence: int
    reason: str

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 100:
            raise ValueError("confidence must be between 0 and 100")
        if self.side not in ("LONG", "SHORT"):
            raise ValueError("side must be LONG or SHORT")
