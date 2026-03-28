"""
Shared types used by both the backtesting engine and strategies.
Keeping these in a separate module breaks the circular import between
strategy/ (which needs Signal) and backtesting/ (which needs TrendStrategy).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class TradeDirection(IntEnum):
    LONG = 1
    SHORT = -1


@dataclass
class Signal:
    direction: int          # 1 = long, -1 = short
    stop_loss: float
    take_profit: float
    volume: Optional[float] = None   # None = engine calculates
    pyramid_level: int = 1
