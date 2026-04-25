"""
Safe Mathematical Operations
Defensive programming layer for numerical operations
Prevents NaN/Inf crashes and provides validated operations
"""

import logging
import math

import numpy as np

LOG = logging.getLogger(__name__)

# Constants
SAFE_EPSILON = 1e-10
SAFE_SMALL = 1e-100
SAFE_LARGE = 1e100
SAFE_DIV_MIN = 1e-15
LOG_OVERFLOW_GUARD = 100.0
EXP_UPPER_GUARD = 100.0
EXP_LOWER_GUARD = -100.0
MIN_SAMPLE_COUNT = 2


class SafeMath:
    """Safe mathematical operations with validation and default handling"""

    @staticmethod
    def to_decimal(value, digits: int):
        """Convert value to Decimal with instrument-specific digits.

        Args:
            value: Numeric value to convert
            digits: Number of decimal places (0-10)

        Returns:
            Decimal value quantized to specified digits

        Raises:
            ValueError: If value is NaN/Inf or digits out of range
        """
        from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

        # Validate digits range
        if not isinstance(digits, int) or not 0 <= digits <= 10:
            LOG.error("Invalid digits: %s (must be 0-10)", digits)
            raise ValueError(f"Invalid digits: {digits} (must be 0-10)")

        # Handle None
        if value is None:
            LOG.error("to_decimal received None value")
            raise ValueError("value is None")

        # Check for NaN/Inf before conversion
        if isinstance(value, (float, int, np.floating, np.integer)):
            if not math.isfinite(float(value)):
                LOG.error("Non-finite value in to_decimal: %s", value)
                raise ValueError(f"Non-finite value: {value}")

        try:
            dec = Decimal(str(value))

            # Validate result is finite
            if not dec.is_finite():
                LOG.error("Decimal conversion produced non-finite result: %s", dec)
                raise ValueError(f"Decimal conversion produced non-finite result: {dec}")

            quant = Decimal("1").scaleb(-digits)
            result = dec.quantize(quant, rounding=ROUND_HALF_UP)
            return result

        except (InvalidOperation, ValueError, TypeError, OverflowError) as e:
            LOG.error("to_decimal failed for value=%s, digits=%d: %s", value, digits, e)
            raise

    @staticmethod
    def quantize(value, digits: int):
        """Quantize an existing Decimal to instrument-specific digits.

        Args:
            value: Decimal value to quantize
            digits: Number of decimal places (0-10)

        Returns:
            Quantized Decimal value

        Raises:
            ValueError: If digits out of range
        """
        from decimal import ROUND_HALF_UP, Decimal, InvalidOperation  # noqa: PLC0415

        # Validate digits range
        if not isinstance(digits, int) or not 0 <= digits <= 10:
            LOG.error("Invalid digits: %s (must be 0-10)", digits)
            raise ValueError(f"Invalid digits: {digits} (must be 0-10)")

        try:
            quant = Decimal("1").scaleb(-digits)
            result = Decimal(value).quantize(quant, rounding=ROUND_HALF_UP)

            # Validate result is finite
            if not result.is_finite():
                LOG.error("Quantize produced non-finite result: %s", result)
                return Decimal("0").quantize(quant)

            return result

        except (InvalidOperation, ValueError, TypeError) as e:
            LOG.error("quantize failed for value=%s, digits=%d: %s", value, digits, e)
            return Decimal("0").quantize(Decimal("1").scaleb(-digits))

    @staticmethod
    def is_valid(x: float | np.ndarray) -> bool:
        """Check if value is valid (not NaN or Inf)"""
        if isinstance(x, np.ndarray):
            return np.all(np.isfinite(x))
        return math.isfinite(x)

    @staticmethod
    def is_nan(x: float | np.ndarray) -> bool:
        """Check if value is NaN"""
        if isinstance(x, np.ndarray):
            return np.any(np.isnan(x))
        return math.isnan(x)

    @staticmethod
    def is_inf(x: float | np.ndarray) -> bool:
        """Check if value is Inf"""
        if isinstance(x, np.ndarray):
            return np.any(np.isinf(x))
        return math.isinf(x)

    @staticmethod
    def is_zero(x: float, eps: float = SAFE_EPSILON) -> bool:
        """Check if value is effectively zero"""
        return abs(x) < eps

    @staticmethod
    def is_not_zero(x: float, eps: float = SAFE_EPSILON) -> bool:
        """Check if value is effectively non-zero"""
        return abs(x) >= eps

    @staticmethod
    def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = SAFE_EPSILON) -> bool:
        """
        Check if two floats are approximately equal.
        Uses both relative and absolute tolerance like math.isclose().

        Args:
            a: First value
            b: Second value
            rel_tol: Relative tolerance (default 1e-9)
            abs_tol: Absolute tolerance (default SAFE_EPSILON)

        Returns:
            True if values are close enough to be considered equal
        """
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    @staticmethod
    def safe_div(a: float, b: float, default: float = 0.0) -> float:
        """Division with zero protection and logging."""
        if abs(b) < SAFE_DIV_MIN:
            LOG.debug("safe_div: denominator %.2e below threshold, returning default", b)
            return default
        result = a / b
        if not SafeMath.is_valid(result):
            LOG.warning("Non-finite result from div(%.4f, %.4f)", a, b)
            return default
        return result

    @staticmethod
    def safe_log(x: float, default: float = 0.0) -> float:
        """Logarithm with negative/zero protection and logging."""
        if x <= SAFE_SMALL:
            LOG.debug("safe_log: input %.6f below threshold, returning default", x)
            return default
        result = math.log(x)
        if not SafeMath.is_valid(result):
            LOG.warning("Non-finite result from log(%.6f)", x)
            return default
        return result

    @staticmethod
    def safe_log1p(x: float, default: float = 0.0) -> float:
        """Log(1+x) with protection and logging."""
        if x <= -1.0:
            LOG.debug("safe_log1p: input %.6f <= -1.0, returning default", x)
            return default
        result = math.log1p(x)
        if not SafeMath.is_valid(result):
            LOG.warning("Non-finite result from log1p(%.6f)", x)
            return default
        return result

    @staticmethod
    def safe_sqrt(x: float, default: float = 0.0) -> float:
        """Square root with negative protection and logging."""
        if x < 0:
            LOG.debug("safe_sqrt: negative input %.6f, returning default", x)
            return default
        result = math.sqrt(x)
        if not SafeMath.is_valid(result):
            LOG.warning("Non-finite result from sqrt(%.6f)", x)
            return default
        return result

    @staticmethod
    def safe_pow(base: float, exp: float, default: float = 0.0) -> float:
        """Power with overflow protection and logging."""
        try:
            # Prevent overflow
            if abs(exp * math.log(abs(base) + SAFE_SMALL)) > LOG_OVERFLOW_GUARD:
                LOG.debug("safe_pow overflow guard triggered: base=%.4f, exp=%.4f", base, exp)
                return default
            result = math.pow(base, exp)
            if not SafeMath.is_valid(result):
                LOG.warning("Non-finite result from pow(%.4f, %.4f)", base, exp)
                return default
            return result
        except (ValueError, OverflowError) as e:
            LOG.debug("Overflow/error in safe_pow(%.4f, %.4f): %s", base, exp, e)
            return default

    @staticmethod
    def safe_exp(x: float, default: float = 0.0) -> float:
        """Exponential with overflow protection and logging."""
        if x > EXP_UPPER_GUARD:  # e^100 is huge
            LOG.debug("safe_exp: input %.4f above upper guard, returning default", x)
            return default
        if x < EXP_LOWER_GUARD:  # e^-100 is tiny
            return 0.0
        result = math.exp(x)
        if not SafeMath.is_valid(result):
            LOG.warning("Non-finite result from exp(%.4f)", x)
            return default
        return result

    @staticmethod
    def clamp(x: float, min_val: float, max_val: float) -> float:
        """Hard clamp to range [min_val, max_val]"""
        return max(min_val, min(max_val, x))

    @staticmethod
    def soft_clamp(x: float, min_val: float, max_val: float) -> float:
        """Soft clamp using tanh transformation"""
        center = (min_val + max_val) / 2
        range_val = (max_val - min_val) / 2
        return center + range_val * math.tanh(x)

    @staticmethod
    def clamp_positive(x: float, min_val: float = SAFE_SMALL) -> float:
        """Ensure value is positive"""
        return max(min_val, x)

    @staticmethod
    def normalize_angle(degrees: float) -> float:
        """Normalize angle to [0, 360) range."""
        if not SafeMath.is_valid(degrees):
            return 0.0
        normalized = degrees % 360.0
        if normalized < 0:
            normalized += 360.0
        return normalized

    @staticmethod
    def is_equal(a: float, b: float, eps: float = SAFE_EPSILON) -> bool:
        """Tolerance-based equality"""
        return abs(a - b) < eps

    @staticmethod
    def is_greater(a: float, b: float, eps: float = SAFE_EPSILON) -> bool:
        """Tolerance-based greater-than"""
        return a > b + eps

    @staticmethod
    def is_less(a: float, b: float, eps: float = SAFE_EPSILON) -> bool:
        """Tolerance-based less-than"""
        return a < b - eps

    @staticmethod
    def safe_clip(x: float, min_val: float, max_val: float, default: float = 0.0) -> float:
        """Clamp with NaN/Inf protection — returns default for NaN, clips Inf to bounds."""
        if math.isnan(x):
            return default
        # +/- Inf gets clipped to bounds
        return max(min_val, min(max_val, x))

    @staticmethod
    def safe_mean(values, default: float = 0.0) -> float:
        """Mean with NaN-skip and empty-list protection."""
        try:
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                return default
            result = float(np.nanmean(arr))
            return result if SafeMath.is_valid(result) else default
        except Exception:
            return default

    @staticmethod
    def safe_percentile(values, percentile: float, default: float = 0.0) -> float:
        """Percentile with NaN-skip and empty-list protection.

        Args:
            values: Array of values
            percentile: Percentile to compute (0-100)
            default: Default value if calculation fails

        Returns:
            Percentile value or default
        """
        # Validate percentile range
        if not 0 <= percentile <= 100:
            LOG.error("Invalid percentile: %.1f (must be 0-100)", percentile)
            return default

        try:
            arr = np.asarray(values, dtype=float)

            # Remove NaN values
            clean = arr[~np.isnan(arr)]

            if clean.size == 0:
                LOG.debug("No valid values for percentile calculation")
                return default

            result = float(np.percentile(clean, percentile))

            # Validate result
            if not np.isfinite(result):
                LOG.warning("Percentile calculation produced non-finite result")
                return default

            return result

        except (ValueError, TypeError) as e:
            LOG.warning("Percentile calculation failed: %s", e)
            return default
        except Exception as e:
            LOG.error("Unexpected error in percentile calculation: %s", e)
            return default

    @staticmethod
    def safe_min(values, default: float = 0.0) -> float:
        """Min with NaN-skip and empty-list protection."""
        try:
            arr = np.asarray(values, dtype=float)
            clean = arr[~np.isnan(arr)]
            if clean.size == 0:
                return default
            result = float(np.min(clean))
            return result if SafeMath.is_valid(result) else default
        except Exception:
            return default

    @staticmethod
    def safe_max(values, default: float = 0.0) -> float:
        """Max with NaN-skip and empty-list protection."""
        try:
            arr = np.asarray(values, dtype=float)
            clean = arr[~np.isnan(arr)]
            if clean.size == 0:
                return default
            result = float(np.max(clean))
            return result if SafeMath.is_valid(result) else default
        except Exception:
            return default

    @staticmethod
    def normalize_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Safe softmax normalization"""
        if not SafeMath.is_valid(logits):
            return np.ones(len(logits)) / len(logits)

        # Temperature scaling
        scaled = logits / max(temperature, SAFE_EPSILON)

        # Subtract max for numerical stability
        scaled = scaled - np.max(scaled)

        # Exp and normalize
        exp_vals = np.exp(np.clip(scaled, EXP_LOWER_GUARD, EXP_UPPER_GUARD))
        total = np.sum(exp_vals)

        return exp_vals / total

    @staticmethod
    def running_mean_update(old_mean: float, new_value: float, count: int) -> float:
        """Welford's online mean update"""
        return old_mean + (new_value - old_mean) / max(count, 1)

    @staticmethod
    def running_variance_update(
        old_variance: float, old_mean: float, new_mean: float, new_value: float, count: int
    ) -> float:
        """Welford's online variance update"""
        if count < MIN_SAMPLE_COUNT:
            return 0.0
        delta1 = new_value - old_mean
        delta2 = new_value - new_mean
        return old_variance + (delta1 * delta2 - old_variance) / count


class RunningStats:
    """Online statistics computation using Welford's algorithm"""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, value: float):
        """Add new value and update statistics"""
        if not SafeMath.is_valid(value):
            return

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def get_variance(self) -> float:
        """Get variance"""
        if self.count < MIN_SAMPLE_COUNT:
            return 0.0
        return self.m2 / (self.count - 1)

    def get_std(self) -> float:
        """Get standard deviation"""
        return SafeMath.safe_sqrt(self.get_variance())

    def get_mean(self) -> float:
        """Get mean"""
        return self.mean

    def get_z_score(self, value: float) -> float:
        """Get z-score for value"""
        std = self.get_std()
        if std < SAFE_EPSILON:
            return 0.0
        return SafeMath.safe_div(value - self.mean, std, 0.0)

    def reset(self):
        """Reset statistics"""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")


# Module-level utility functions
def safe_array_operation(arr: np.ndarray, operation: str, default: float = 0.0) -> float:
    """Safely apply operation to array"""
    if arr is None or len(arr) == 0:
        return default

    if not SafeMath.is_valid(arr):
        return default

    operations = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
        "sum": np.sum,
        "median": np.median,
    }

    if operation not in operations:
        return default

    try:
        result = operations[operation](arr)
        return result if SafeMath.is_valid(result) else default
    except Exception:
        return default


# ── Rolling statistics (shared by ctrader_ddqn_paper, dual_policy) ────────


def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Simple rolling mean; positions with fewer than *n* samples are NaN."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) >= n:
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1 :] = (cs[n:] - cs[:-n]) / n
    return out


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    """Rolling standard deviation with NaN protection.

    Uses Welford's online algorithm for O(n) complexity instead of O(n²).
    For a 10,000-element array with window 100, this is ~100x faster.
    """
    out = np.full(len(x), np.nan, dtype=np.float64)
    if len(x) < n or n < 2:
        return out

    # Initialize with first window using Welford's algorithm
    count = n
    mean = np.mean(x[:n])
    m2 = np.sum((x[:n] - mean) ** 2)

    out[n - 1] = np.sqrt(m2 / (n - 1)) if m2 >= 0 else 0.0

    # Rolling update: remove oldest, add newest (O(1) per element)
    for i in range(n, len(x)):
        old_val = x[i - n]
        new_val = x[i]

        # Online update of mean and m2
        old_mean = mean
        mean = old_mean + (new_val - old_val) / n

        # Update sum of squared deviations
        # m2_new = m2_old - (old_val - old_mean)*(old_val - mean) + (new_val - old_mean)*(new_val - mean)
        m2 = m2 - (old_val - old_mean) * (old_val - mean) + (new_val - old_mean) * (new_val - mean)
        m2 = max(0.0, m2)  # Numerical stability

        out[i] = np.sqrt(m2 / (n - 1)) if m2 >= 0 else 0.0

    return out


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature for confidence calculation."""
    exp_x = np.exp((x - np.max(x)) / temperature)
    return exp_x / exp_x.sum()
