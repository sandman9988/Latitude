"""
Data cleaning — gap fill, outlier removal, OHLC consistency checks.
All operations are symbol-agnostic and non-destructive (returns new data).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional
from core.math_utils import safe_div
from core.numeric import non_negative, is_valid_number, clamp
from core.logger import get_logger

logger = get_logger("cleaner")


@dataclass
class Bar:
    """Single OHLCV bar."""
    timestamp: float   # unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: str = ""

    def is_valid(self) -> bool:
        return (
            all(is_valid_number(v) and v > 0 for v in [self.open, self.high, self.low, self.close])
            and self.high >= self.low
            and self.high >= self.open
            and self.high >= self.close
            and self.low <= self.open
            and self.low <= self.close
            and non_negative(self.volume) >= 0
        )


def clean_bars(bars: List[Bar], symbol: str = "", tf: str = "") -> List[Bar]:
    """
    Full cleaning pipeline:
    1. Remove bars with invalid OHLC relationships
    2. Remove price outliers (> 5 sigma from rolling mean)
    3. Forward-fill gaps (duplicate previous close into O/H/L/C)
    4. Remove duplicate timestamps
    Returns cleaned list.
    """
    if not bars:
        return bars

    bars = _remove_invalid(bars, symbol, tf)
    bars = _remove_duplicates(bars, symbol, tf)
    bars = _remove_outliers(bars, symbol, tf)
    bars = _fill_gaps(bars, symbol, tf)
    return bars


def _remove_invalid(bars: List[Bar], symbol: str, tf: str) -> List[Bar]:
    clean = []
    removed = 0
    for bar in bars:
        if bar.is_valid():
            clean.append(bar)
        else:
            removed += 1
    if removed:
        logger.warning(
            f"Removed {removed} invalid bars",
            symbol=symbol, tf=tf, component="cleaner"
        )
    return clean


def _remove_duplicates(bars: List[Bar], symbol: str, tf: str) -> List[Bar]:
    seen: set[float] = set()
    clean = []
    dupes = 0
    for bar in bars:
        if bar.timestamp not in seen:
            seen.add(bar.timestamp)
            clean.append(bar)
        else:
            dupes += 1
    if dupes:
        logger.warning(
            f"Removed {dupes} duplicate timestamps",
            symbol=symbol, tf=tf, component="cleaner"
        )
    return sorted(clean, key=lambda b: b.timestamp)


def _remove_outliers(bars: List[Bar], symbol: str, tf: str, window: int = 50, sigma: float = 5.0) -> List[Bar]:
    """Remove bars where close is more than sigma std devs from rolling mean."""
    if len(bars) < window:
        return bars

    clean = list(bars[:window])
    removed = 0
    for i in range(window, len(bars)):
        window_closes = [b.close for b in bars[i - window:i]]
        mean = sum(window_closes) / window
        std = math.sqrt(sum((c - mean) ** 2 for c in window_closes) / window) or 1.0
        bar = bars[i]
        z = abs(safe_div(bar.close - mean, std, fallback=0.0))
        if z <= sigma:
            clean.append(bar)
        else:
            removed += 1
            logger.warning(
                f"Outlier bar removed: close={bar.close:.5f} z={z:.2f}",
                symbol=symbol, tf=tf, component="cleaner",
                timestamp=bar.timestamp
            )
    if removed:
        logger.info(f"Removed {removed} outlier bars", symbol=symbol, tf=tf)
    return clean


def _fill_gaps(bars: List[Bar], symbol: str, tf: str) -> List[Bar]:
    """Forward-fill: if a bar has zero volume, carry previous close into OHLC."""
    filled = 0
    for i in range(1, len(bars)):
        if bars[i].volume == 0 and i > 0:
            prev = bars[i - 1]
            bars[i] = Bar(
                timestamp=bars[i].timestamp,
                open=prev.close,
                high=prev.close,
                low=prev.close,
                close=prev.close,
                volume=0.0,
                symbol=bars[i].symbol,
                timeframe=bars[i].timeframe,
            )
            filled += 1
    if filled:
        logger.info(f"Forward-filled {filled} zero-volume bars", symbol=symbol, tf=tf)
    return bars
