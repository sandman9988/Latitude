"""
Trend strength and direction indicators.
All accept a smoother= parameter for injectable lag reduction.
KER + VHF are the primary regime gates.
"""
from __future__ import annotations

import math
from typing import Optional
from core.math_utils import safe_div, safe_abs
from core.numeric import clamp, non_negative, is_valid_number
from core.memory import FloatCircularBuffer
from .smoothing import Smoother, make_smoother


# ---------------------------------------------------------------------------
# KER — Kaufman Efficiency Ratio
# 0 = pure noise/chop, 1 = perfectly efficient trend
# Works reliably on short windows. Primary trend gate.
# ---------------------------------------------------------------------------
class KER:
    """
    Kaufman Efficiency Ratio.
    period: lookback window (default 10)
    Returns 0.0-1.0. Values > threshold indicate trending.
    """

    def __init__(self, period: int = 10) -> None:
        self._period = max(2, period)
        self._prices = FloatCircularBuffer(self._period + 1)
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._prices.full

    def update(self, price: float) -> float:
        if not is_valid_number(price):
            return self._value

        self._prices.push(price)

        if not self._prices.full:
            return self._value

        data = self._prices.to_list()
        direction = safe_abs(data[-1] - data[0])
        volatility = sum(safe_abs(data[i] - data[i - 1]) for i in range(1, len(data)))

        self._value = clamp(safe_div(direction, volatility, fallback=0.0), 0.0, 1.0)
        return self._value

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# VHF — Vertical Horizontal Filter
# 0 = ranging/choppy, 1 = strongly trending
# Detects trend vs chop on 20-50 bar windows.
# ---------------------------------------------------------------------------
class VHF:
    """
    Vertical Horizontal Filter.
    period: lookback window (default 28)
    Returns 0.0-1.0 (normalised). Raw VHF > 0.35 typically = trending.
    """

    def __init__(self, period: int = 28) -> None:
        self._period = max(2, period)
        self._prices = FloatCircularBuffer(self._period)
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._prices.full

    def update(self, price: float) -> float:
        if not is_valid_number(price):
            return self._value

        self._prices.push(price)

        if not self._prices.full:
            return self._value

        data = self._prices.to_list()
        highest = max(data)
        lowest = min(data)
        vertical = safe_abs(highest - lowest)
        horizontal = sum(safe_abs(data[i] - data[i - 1]) for i in range(1, len(data)))

        raw = safe_div(vertical, horizontal, fallback=0.0)
        # Normalise: raw VHF typically 0.1-0.5, clamp to 0-1
        self._value = clamp(raw * 2.5, 0.0, 1.0)
        return self._value

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# AROON — measures how recent the highest high and lowest low are
# aroon_up: 0-100, aroon_down: 0-100, oscillator: -100 to +100
# ---------------------------------------------------------------------------
class AROON:
    """
    Aroon indicator.
    period: lookback (default 25)
    """

    def __init__(self, period: int = 25) -> None:
        self._period = max(2, period)
        self._highs = FloatCircularBuffer(self._period + 1)
        self._lows = FloatCircularBuffer(self._period + 1)
        self._up = 0.0
        self._down = 0.0

    @property
    def up(self) -> float:
        return self._up

    @property
    def down(self) -> float:
        return self._down

    @property
    def oscillator(self) -> float:
        return self._up - self._down

    @property
    def ready(self) -> bool:
        return self._highs.full

    def update(self, high: float, low: float) -> tuple[float, float]:
        if not is_valid_number(high) or not is_valid_number(low):
            return self._up, self._down

        self._highs.push(high)
        self._lows.push(low)

        if not self._highs.full:
            return self._up, self._down

        highs = self._highs.to_list()
        lows = self._lows.to_list()
        n = len(highs)

        bars_since_high = n - 1 - highs.index(max(highs))
        bars_since_low = n - 1 - lows.index(min(lows))

        self._up = safe_div((n - 1 - bars_since_high) * 100.0, n - 1)
        self._down = safe_div((n - 1 - bars_since_low) * 100.0, n - 1)

        return self._up, self._down

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# Chande CMO — Chande Momentum Oscillator
# Raw signed momentum, no lag penalty. Range -100 to +100.
# ---------------------------------------------------------------------------
class ChandeCMO:
    """
    Chande Momentum Oscillator.
    Positive = upward momentum dominant, negative = downward.
    Works well on short windows.
    """

    def __init__(self, period: int = 14) -> None:
        self._period = max(2, period)
        self._prices = FloatCircularBuffer(self._period + 1)
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._prices.full

    def update(self, price: float) -> float:
        if not is_valid_number(price):
            return self._value

        self._prices.push(price)

        if not self._prices.full:
            return self._value

        data = self._prices.to_list()
        up_sum = 0.0
        down_sum = 0.0
        for i in range(1, len(data)):
            diff = data[i] - data[i - 1]
            if diff > 0:
                up_sum += diff
            elif diff < 0:
                down_sum += safe_abs(diff)

        self._value = safe_div((up_sum - down_sum) * 100.0, up_sum + down_sum, fallback=0.0)
        return self._value

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# Fractal Dimension Index (FDIC)
# Detects trending vs chopping better than Hurst on short windows.
# ~1.0 = trending (low fractal dimension), ~2.0 = choppy (high)
# ---------------------------------------------------------------------------
class FractalDimension:
    """
    Fractal Dimension Index.
    period: must be even (default 30)
    Returns 1.0-2.0. Values < 1.5 indicate trending.
    """

    def __init__(self, period: int = 30) -> None:
        self._period = max(4, period + (period % 2))  # ensure even
        self._prices = FloatCircularBuffer(self._period)
        self._value = 1.5

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._prices.full

    def update(self, price: float) -> float:
        if not is_valid_number(price):
            return self._value

        self._prices.push(price)

        if not self._prices.full:
            return self._value

        data = self._prices.to_list()
        n = len(data)
        half = n // 2

        def _range(subset: list[float]) -> float:
            return max(subset) - min(subset)

        r1 = _range(data[:half])
        r2 = _range(data[half:])
        r_all = _range(data)

        if r_all <= 0.0 or (r1 + r2) <= 0.0:
            return self._value

        ratio = safe_div(r1 + r2, r_all, fallback=1.0)
        if ratio <= 0.0:
            return self._value

        fd = 1.0 + safe_div(math.log(r1 + r2) - math.log(r_all), math.log(2.0), fallback=0.5)

        self._value = clamp(fd, 1.0, 2.0)
        return self._value

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# Market Structure — Higher Highs / Higher Lows detector
# ---------------------------------------------------------------------------
class MarketStructure:
    """
    Detects higher highs / higher lows (uptrend) and lower highs / lower lows (downtrend).
    Uses swing point detection with a configurable lookback.
    """

    def __init__(self, swing_period: int = 5) -> None:
        self._swing = max(2, swing_period)
        self._highs = FloatCircularBuffer(self._swing * 2 + 1)
        self._lows = FloatCircularBuffer(self._swing * 2 + 1)
        self._last_swing_high = 0.0
        self._last_swing_low = 0.0
        self._prev_swing_high = 0.0
        self._prev_swing_low = 0.0
        self._trend: int = 0   # 1 = up, -1 = down, 0 = neutral

    @property
    def trend(self) -> int:
        """1 = uptrend, -1 = downtrend, 0 = neutral."""
        return self._trend

    @property
    def ready(self) -> bool:
        return self._highs.full

    def update(self, high: float, low: float) -> int:
        if not is_valid_number(high) or not is_valid_number(low):
            return self._trend

        self._highs.push(high)
        self._lows.push(low)

        if not self._highs.full:
            return self._trend

        highs = self._highs.to_list()
        lows = self._lows.to_list()
        mid = len(highs) // 2

        # Detect swing high: mid is highest in window
        if highs[mid] == max(highs):
            new_sh = highs[mid]
            if self._last_swing_high > 0:
                self._prev_swing_high = self._last_swing_high
            self._last_swing_high = new_sh

        # Detect swing low: mid is lowest in window
        if lows[mid] == min(lows):
            new_sl = lows[mid]
            if self._last_swing_low > 0:
                self._prev_swing_low = self._last_swing_low
            self._last_swing_low = new_sl

        # Classify structure
        hh = self._last_swing_high > self._prev_swing_high > 0
        hl = self._last_swing_low > self._prev_swing_low > 0
        lh = self._last_swing_high < self._prev_swing_high > 0
        ll = self._last_swing_low < self._prev_swing_low > 0

        if hh and hl:
            self._trend = 1
        elif lh and ll:
            self._trend = -1
        else:
            self._trend = 0

        return self._trend

    def reset(self) -> None:
        self.__init__(self._swing)
