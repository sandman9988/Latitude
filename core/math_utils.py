"""
Safe math primitives — all division, roots, logs go through here.
No raw arithmetic on untrusted values anywhere in the codebase.
"""
from __future__ import annotations

import math
from typing import Union

Number = Union[int, float]

_FLOAT_MAX = 1e15
_FLOAT_MIN = -1e15
_EPSILON = 1e-10


def safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division with zero-denominator guard and overflow clamp."""
    if denominator == 0.0 or abs(denominator) < _EPSILON:
        return fallback
    result = numerator / denominator
    if not math.isfinite(result):
        return fallback
    return max(_FLOAT_MIN, min(_FLOAT_MAX, result))


def safe_sqrt(value: float, fallback: float = 0.0) -> float:
    """Square root — returns fallback for negative input."""
    if value < 0.0:
        return fallback
    if not math.isfinite(value):
        return fallback
    return math.sqrt(value)


def safe_log(value: float, base: float = math.e, fallback: float = 0.0) -> float:
    """Logarithm — returns fallback for non-positive input."""
    if value <= 0.0 or not math.isfinite(value):
        return fallback
    if base <= 0.0 or base == 1.0:
        return fallback
    result = math.log(value) / math.log(base)
    if not math.isfinite(result):
        return fallback
    return result


def safe_exp(value: float, fallback: float = 0.0) -> float:
    """Exponentiation — clamps input to prevent overflow."""
    if not math.isfinite(value):
        return fallback
    clamped = max(-700.0, min(700.0, value))
    result = math.exp(clamped)
    if not math.isfinite(result):
        return fallback
    return result


def safe_pow(base: float, exp: float, fallback: float = 0.0) -> float:
    """Power function with overflow and domain guards."""
    if not math.isfinite(base) or not math.isfinite(exp):
        return fallback
    if base < 0.0 and not float(exp).is_integer():
        return fallback
    try:
        result = math.pow(base, exp)
    except (ValueError, OverflowError):
        return fallback
    if not math.isfinite(result):
        return fallback
    return max(_FLOAT_MIN, min(_FLOAT_MAX, result))


def safe_abs(value: float) -> float:
    """Absolute value — handles non-finite gracefully."""
    if not math.isfinite(value):
        return 0.0
    return abs(value)
