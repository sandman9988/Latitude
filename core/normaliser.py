"""
Normalisers — fitted per symbol/timeframe, never cross-contaminated.
All return values in [0, 1] or z-score range with finite guards.
"""
from __future__ import annotations

import math
from typing import Sequence
from .math_utils import safe_div, safe_sqrt
from .numeric import clamp


class MinMaxNormaliser:
    """Fits on a window, transforms to [0, 1]."""

    def __init__(self) -> None:
        self._min: float = 0.0
        self._max: float = 1.0
        self._fitted: bool = False

    def fit(self, values: Sequence[float]) -> "MinMaxNormaliser":
        finite = [v for v in values if math.isfinite(v)]
        if len(finite) < 2:
            return self
        self._min = min(finite)
        self._max = max(finite)
        self._fitted = True
        return self

    def transform(self, value: float) -> float:
        if not self._fitted or not math.isfinite(value):
            return 0.0
        span = self._max - self._min
        return clamp(safe_div(value - self._min, span, fallback=0.0), 0.0, 1.0)

    def fit_transform(self, values: Sequence[float]) -> list[float]:
        self.fit(values)
        return [self.transform(v) for v in values]


class ZScoreNormaliser:
    """Fits on a window, transforms to z-score. Clamps at ±4 sigma."""

    _CLAMP = 4.0

    def __init__(self) -> None:
        self._mean: float = 0.0
        self._std: float = 1.0
        self._fitted: bool = False

    def fit(self, values: Sequence[float]) -> "ZScoreNormaliser":
        finite = [v for v in values if math.isfinite(v)]
        if len(finite) < 2:
            return self
        n = len(finite)
        self._mean = sum(finite) / n
        variance = sum((v - self._mean) ** 2 for v in finite) / n
        self._std = safe_sqrt(variance, fallback=1.0) or 1.0
        self._fitted = True
        return self

    def transform(self, value: float) -> float:
        if not self._fitted or not math.isfinite(value):
            return 0.0
        z = safe_div(value - self._mean, self._std, fallback=0.0)
        return clamp(z, -self._CLAMP, self._CLAMP)

    def fit_transform(self, values: Sequence[float]) -> list[float]:
        self.fit(values)
        return [self.transform(v) for v in values]


class RobustNormaliser:
    """
    Median + IQR-based normaliser — resistant to outliers.
    Better than z-score for fat-tailed financial data.
    """

    def __init__(self) -> None:
        self._median: float = 0.0
        self._iqr: float = 1.0
        self._fitted: bool = False

    def fit(self, values: Sequence[float]) -> "RobustNormaliser":
        finite = sorted(v for v in values if math.isfinite(v))
        if len(finite) < 4:
            return self
        n = len(finite)
        self._median = _percentile(finite, 50)
        q1 = _percentile(finite, 25)
        q3 = _percentile(finite, 75)
        self._iqr = (q3 - q1) or 1.0
        self._fitted = True
        return self

    def transform(self, value: float) -> float:
        if not self._fitted or not math.isfinite(value):
            return 0.0
        return safe_div(value - self._median, self._iqr, fallback=0.0)

    def fit_transform(self, values: Sequence[float]) -> list[float]:
        self.fit(values)
        return [self.transform(v) for v in values]


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Linear interpolation percentile on a pre-sorted list."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    idx = (pct / 100.0) * (n - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])
