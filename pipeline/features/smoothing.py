"""
Low-lag smoothers — injectable pattern.
All smoothers share a common interface:
    smoother.update(value: float) -> float
    smoother.value -> float   (current output)
    smoother.ready -> bool    (warmed up)
    smoother.reset() -> None

Use the factory:
    make_smoother("jma", period=14, phase=0)
    make_smoother("kama", period=10)
    make_smoother("zlema", period=14)
    make_smoother("t3", period=14, vfactor=0.7)
    make_smoother("alma", period=14, sigma=6, offset=0.85)
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable
from core.math_utils import safe_div, safe_sqrt
from core.numeric import clamp, is_valid_number


@runtime_checkable
class Smoother(Protocol):
    @property
    def value(self) -> float: ...
    @property
    def ready(self) -> bool: ...
    def update(self, v: float) -> float: ...
    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# JMA — Jurik Moving Average
# Adaptive, minimal overshoot, phase-correct. Best for entries and runway.
# ---------------------------------------------------------------------------
class JMA:
    """
    Jurik Moving Average.
    phase: -100 to +100 (0 = neutral, positive = less lag more overshoot)
    power: smoothing aggressiveness (default 2)
    """

    def __init__(self, period: int = 14, phase: float = 0.0, power: float = 2.0) -> None:
        self._period = max(1, period)
        self._phase = clamp(phase, -100.0, 100.0)
        self._power = max(0.5, power)
        self._e0 = 0.0
        self._e1 = 0.0
        self._e2 = 0.0
        self._jma = 0.0
        self._det0 = 0.0
        self._det1 = 0.0
        self._count = 0
        self._beta = 0.45 * (self._period - 1) / (0.45 * (self._period - 1) + 2)
        phase_ratio = self._phase / 100.0 + 1.5
        self._alpha = self._beta ** self._power
        self._phase_ratio = phase_ratio

    @property
    def value(self) -> float:
        return self._jma

    @property
    def ready(self) -> bool:
        return self._count >= self._period

    def update(self, v: float) -> float:
        if not is_valid_number(v):
            return self._jma

        if self._count == 0:
            self._e0 = v
            self._e1 = 0.0
            self._e2 = 0.0
            self._jma = v
            self._det0 = 0.0
            self._det1 = 0.0

        self._count += 1
        alpha = self._alpha
        beta = self._beta

        self._e0 = (1 - alpha) * v + alpha * self._e0
        self._e1 = (v - self._e0) * (1 - beta) + beta * self._e1
        self._det0 = self._e0 + self._phase_ratio * self._e1

        self._e2 = (self._det0 - self._jma) * (1 - alpha) ** 2 + alpha ** 2 * self._e2
        self._det1 = self._e2
        self._jma = self._jma + self._det1

        if not is_valid_number(self._jma):
            self._jma = v

        return self._jma

    def reset(self) -> None:
        self.__init__(self._period, self._phase, self._power)


# ---------------------------------------------------------------------------
# KAMA — Kaufman Adaptive Moving Average
# Slows in chop, accelerates in trend. Natural regime adapter.
# ---------------------------------------------------------------------------
class KAMA:
    """
    Kaufman Adaptive Moving Average.
    fast_period: fast EMA period (default 2)
    slow_period: slow EMA period (default 30)
    """

    def __init__(self, period: int = 10, fast_period: int = 2, slow_period: int = 30) -> None:
        self._period = max(1, period)
        self._fast_sc = safe_div(2.0, fast_period + 1)
        self._slow_sc = safe_div(2.0, slow_period + 1)
        self._kama = 0.0
        self._prices: list[float] = []
        self._count = 0

    @property
    def value(self) -> float:
        return self._kama

    @property
    def ready(self) -> bool:
        return self._count > self._period

    def update(self, v: float) -> float:
        if not is_valid_number(v):
            return self._kama

        self._prices.append(v)
        self._count += 1

        if self._count == 1:
            self._kama = v
            return self._kama

        if len(self._prices) > self._period + 1:
            self._prices.pop(0)

        if len(self._prices) < self._period + 1:
            self._kama = v
            return self._kama

        direction = abs(v - self._prices[0])
        volatility = sum(abs(self._prices[i] - self._prices[i - 1]) for i in range(1, len(self._prices)))

        er = safe_div(direction, volatility, fallback=0.0)
        sc = (er * (self._fast_sc - self._slow_sc) + self._slow_sc) ** 2
        self._kama = self._kama + sc * (v - self._kama)

        if not is_valid_number(self._kama):
            self._kama = v

        return self._kama

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# ZLEMA — Zero-Lag EMA
# Subtracts the lag component from a standard EMA.
# ---------------------------------------------------------------------------
class ZLEMA:
    """
    Zero-Lag Exponential Moving Average.
    Reduces lag by subtracting the estimated lag from price before smoothing.
    """

    def __init__(self, period: int = 14) -> None:
        self._period = max(2, period)
        self._alpha = safe_div(2.0, self._period + 1)
        self._lag = math.ceil((self._period - 1) / 2)
        self._ema = 0.0
        self._prices: list[float] = []
        self._count = 0

    @property
    def value(self) -> float:
        return self._ema

    @property
    def ready(self) -> bool:
        return self._count > self._period

    def update(self, v: float) -> float:
        if not is_valid_number(v):
            return self._ema

        self._prices.append(v)
        self._count += 1

        if len(self._prices) > self._lag + 1:
            self._prices.pop(0)

        if self._count == 1:
            self._ema = v
            return self._ema

        lag_price = self._prices[0] if len(self._prices) > self._lag else v
        zlprice = 2.0 * v - lag_price
        self._ema = self._alpha * zlprice + (1.0 - self._alpha) * self._ema

        if not is_valid_number(self._ema):
            self._ema = v

        return self._ema

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# T3 — Tillson T3
# Six EMAs compounded. Very smooth but moderate overshoot risk on breakouts.
# vfactor: 0.0 = pure EMA, 0.7 = standard T3, 1.0 = maximum smoothness
# ---------------------------------------------------------------------------
class T3:
    """Tillson T3 Moving Average."""

    def __init__(self, period: int = 14, vfactor: float = 0.7) -> None:
        self._period = max(2, period)
        self._vfactor = clamp(vfactor, 0.0, 1.0)
        alpha = safe_div(2.0, self._period + 1)
        self._alpha = alpha
        v = self._vfactor
        self._c1 = -(v ** 3)
        self._c2 = 3.0 * v ** 2 + 3.0 * v ** 3
        self._c3 = -6.0 * v ** 2 - 3.0 * v - 3.0 * v ** 3
        self._c4 = 1.0 + 3.0 * v + v ** 3 + 3.0 * v ** 2
        self._e1 = self._e2 = self._e3 = 0.0
        self._e4 = self._e5 = self._e6 = 0.0
        self._count = 0

    @property
    def value(self) -> float:
        return self._c1 * self._e6 + self._c2 * self._e5 + self._c3 * self._e4 + self._c4 * self._e3

    @property
    def ready(self) -> bool:
        return self._count >= self._period * 6

    def update(self, v: float) -> float:
        if not is_valid_number(v):
            return self.value

        if self._count == 0:
            self._e1 = self._e2 = self._e3 = v
            self._e4 = self._e5 = self._e6 = v

        self._count += 1
        a = self._alpha
        self._e1 = a * v + (1 - a) * self._e1
        self._e2 = a * self._e1 + (1 - a) * self._e2
        self._e3 = a * self._e2 + (1 - a) * self._e3
        self._e4 = a * self._e3 + (1 - a) * self._e4
        self._e5 = a * self._e4 + (1 - a) * self._e5
        self._e6 = a * self._e5 + (1 - a) * self._e6

        result = self.value
        if not is_valid_number(result):
            return v
        return result

    def reset(self) -> None:
        self.__init__(self._period, self._vfactor)


# ---------------------------------------------------------------------------
# ALMA — Arnaud Legoux Moving Average
# Gaussian-weighted. Offset controls lag/smoothness tradeoff.
# ---------------------------------------------------------------------------
class ALMA:
    """
    Arnaud Legoux Moving Average.
    offset: 0.0-1.0, higher = less lag (default 0.85)
    sigma: gaussian width, higher = smoother (default 6)
    """

    def __init__(self, period: int = 14, sigma: float = 6.0, offset: float = 0.85) -> None:
        self._period = max(2, period)
        self._weights = self._compute_weights(self._period, sigma, offset)
        self._prices: list[float] = []
        self._count = 0
        self._value = 0.0

    @staticmethod
    def _compute_weights(period: int, sigma: float, offset: float) -> list[float]:
        m = offset * (period - 1)
        s = safe_div(period, sigma, fallback=float(period))
        weights = []
        for i in range(period):
            w = math.exp(-((i - m) ** 2) / (2.0 * s ** 2))
            weights.append(w)
        total = sum(weights)
        if total == 0.0:
            return [safe_div(1.0, period)] * period
        return [safe_div(w, total) for w in weights]

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._count >= self._period

    def update(self, v: float) -> float:
        if not is_valid_number(v):
            return self._value

        self._prices.append(v)
        self._count += 1

        if len(self._prices) > self._period:
            self._prices.pop(0)

        if len(self._prices) < self._period:
            self._value = v
            return self._value

        self._value = sum(self._prices[i] * self._weights[i] for i in range(self._period))

        if not is_valid_number(self._value):
            self._value = v

        return self._value

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_SMOOTHER_MAP = {
    "jma": lambda **kw: JMA(**kw),
    "kama": lambda **kw: KAMA(**kw),
    "zlema": lambda **kw: ZLEMA(**kw),
    "t3": lambda **kw: T3(**kw),
    "alma": lambda **kw: ALMA(**kw),
}


def make_smoother(name: str, **kwargs) -> Smoother:
    """
    Factory for creating smoothers by name.
    Usage: make_smoother("jma", period=14, phase=0)
    """
    key = name.lower()
    if key not in _SMOOTHER_MAP:
        raise ValueError(f"Unknown smoother '{name}'. Choose from: {list(_SMOOTHER_MAP)}")
    return _SMOOTHER_MAP[key](**kwargs)
