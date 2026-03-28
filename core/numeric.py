"""
Numeric enforcement — clamping, non-negative, precision rounding, step alignment.
All raw values from external sources pass through here before touching strategy logic.
"""
from __future__ import annotations

import math
from .math_utils import safe_div

_PRICE_MAX = 1_000_000.0
_PRICE_MIN = 0.0
_VOLUME_MAX = 1_000_000.0
_VOLUME_MIN = 0.0


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]. Handles NaN/inf by returning lo."""
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def clamp_price(price: float) -> float:
    """Clamp to valid price range. Prices are always positive."""
    return clamp(price, _PRICE_MIN, _PRICE_MAX)


def clamp_volume(volume: float) -> float:
    """Clamp to valid volume range."""
    return clamp(volume, _VOLUME_MIN, _VOLUME_MAX)


def non_negative(value: float, fallback: float = 0.0) -> float:
    """Return value if >= 0, else fallback. Also guards non-finite."""
    if not math.isfinite(value):
        return fallback
    return value if value >= 0.0 else fallback


def round_to_step(value: float, step: float) -> float:
    """
    Round value to nearest step (e.g. lot step 0.01, tick size 0.25).
    Used for broker-compliant volume and price rounding.
    """
    if step <= 0.0 or not math.isfinite(step) or not math.isfinite(value):
        return value
    factor = safe_div(1.0, step, fallback=1.0)
    return round(value * factor) / factor


def pip_value(tick_size: float, tick_value: float, volume: float, pip_size: float = 0.0) -> float:
    """
    Monetary value of one pip move for a given volume.
    tick_value = money per tick_size move per lot.
    pip_size defaults to tick_size when not provided.
    """
    if tick_size <= 0.0 or not math.isfinite(tick_size):
        return 0.0
    effective_pip = pip_size if pip_size > 0.0 and math.isfinite(pip_size) else tick_size
    return non_negative(safe_div(effective_pip, tick_size) * tick_value * volume)


def normalise_01(value: float, lo: float, hi: float) -> float:
    """Normalise value to [0, 1] given known range."""
    span = hi - lo
    if span <= 0.0:
        return 0.0
    return clamp(safe_div(value - lo, span), 0.0, 1.0)


def is_valid_number(value: float) -> bool:
    """True if value is finite and not NaN."""
    return math.isfinite(value)
