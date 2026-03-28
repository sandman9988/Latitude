"""
DTW (Dynamic Time Warping) feature extraction.
Computes similarity scores between recent price windows and a library
of known trend templates. Low DTW distance = close match to template.
"""
from __future__ import annotations

import math
from typing import Sequence
from core.math_utils import safe_div, safe_sqrt
from core.numeric import clamp, is_valid_number


def dtw_distance(seq_a: Sequence[float], seq_b: Sequence[float]) -> float:
    """
    Compute DTW distance between two sequences.
    Lower = more similar. O(n*m) time and space.
    Both sequences are z-score normalised before comparison.
    """
    a = _zscore_normalise(list(seq_a))
    b = _zscore_normalise(list(seq_b))
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    # DTW matrix — use two-row rolling approach for memory efficiency
    INF = float("inf")
    prev = [INF] * (m + 1)
    prev[0] = 0.0
    curr = [INF] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = INF
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [INF] * (m + 1)

    dist = prev[m]
    if not math.isfinite(dist):
        return float("inf")
    return safe_sqrt(dist)


def dtw_similarity(seq_a: Sequence[float], seq_b: Sequence[float]) -> float:
    """
    Normalised similarity score 0.0-1.0.
    1.0 = identical shape, 0.0 = maximally dissimilar.
    """
    dist = dtw_distance(seq_a, seq_b)
    if dist == float("inf"):
        return 0.0
    return math.exp(-dist)


class TrendTemplateMatcher:
    """
    Matches a rolling price window against a library of trend templates.
    Templates are added via add_template(name, price_series).
    Call update(close) each bar. After window fills, scores are available.
    """

    def __init__(self, window: int = 20) -> None:
        self._window = max(4, window)
        self._prices: list[float] = []
        self._templates: dict[str, list[float]] = {}
        self._scores: dict[str, float] = {}

    def add_template(self, name: str, series: Sequence[float]) -> None:
        """Register a trend template. Normalised internally."""
        if len(series) < 2:
            return
        self._templates[name] = list(series)
        self._scores[name] = 0.0

    @property
    def scores(self) -> dict[str, float]:
        """Similarity scores per template, 0.0-1.0."""
        return dict(self._scores)

    @property
    def best_match(self) -> tuple[str, float]:
        """Name and score of highest-matching template."""
        if not self._scores:
            return ("none", 0.0)
        best = max(self._scores, key=lambda k: self._scores[k])
        return (best, self._scores[best])

    @property
    def ready(self) -> bool:
        return len(self._prices) >= self._window

    def update(self, close: float) -> dict[str, float]:
        if not is_valid_number(close):
            return self._scores

        self._prices.append(close)
        if len(self._prices) > self._window:
            self._prices.pop(0)

        if not self.ready:
            return self._scores

        window = self._prices[-self._window:]
        for name, template in self._templates.items():
            self._scores[name] = dtw_similarity(window, template)

        return self._scores

    def reset(self) -> None:
        self._prices.clear()
        self._scores = {k: 0.0 for k in self._templates}


# ---------------------------------------------------------------------------
# Built-in template generators
# ---------------------------------------------------------------------------
def make_impulse_template(length: int = 20) -> list[float]:
    """Strong directional impulse — steady acceleration."""
    return [math.log(1 + i / length) * length for i in range(1, length + 1)]


def make_breakout_template(length: int = 20, consolidation: float = 0.3) -> list[float]:
    """Flat consolidation followed by sharp breakout."""
    flat_bars = int(length * consolidation)
    breakout_bars = length - flat_bars
    flat = [0.0] * flat_bars
    breakout = [i * (1.0 / breakout_bars) * length for i in range(1, breakout_bars + 1)]
    return flat + breakout


def make_pullback_template(length: int = 20, pullback_depth: float = 0.3) -> list[float]:
    """Trend, shallow pullback, continuation."""
    third = length // 3
    up = [i * (1.0 / third) for i in range(third)]
    pb = [up[-1] - i * (pullback_depth / third) for i in range(1, third + 1)]
    cont = [pb[-1] + i * (1.0 / third) for i in range(1, length - 2 * third + 1)]
    return up + pb + cont


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _zscore_normalise(data: list[float]) -> list[float]:
    n = len(data)
    if n < 2:
        return data
    mean = sum(data) / n
    std = safe_sqrt(sum((v - mean) ** 2 for v in data) / n, fallback=1.0) or 1.0
    return [safe_div(v - mean, std, fallback=0.0) for v in data]
