#!/usr/bin/env python3
"""
Defensive Programming Utilities
Implements SafeMath and SafeArray patterns from Master Handbook
Prevents NaN/Inf propagation, division by zero, and array bounds errors
"""

import contextlib
import json
import logging
import math
import os
import tempfile
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Safe math constants
MIN_VALUES_FOR_STD: int = 2  # Minimum values needed to calculate standard deviation


class SafeMath:
    """Safe mathematical operations with NaN/Inf protection"""

    EPSILON = 1e-12  # Minimum divisor threshold

    @staticmethod
    def is_valid(value: float | int) -> bool:
        """Check if value is finite and not NaN"""
        try:
            return math.isfinite(float(value)) and not math.isnan(float(value))
        except (ValueError, TypeError, OverflowError):
            return False

    @staticmethod
    def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safe division with NaN/Inf protection

        Args:
            numerator: Dividend
            denominator: Divisor
            default: Value to return if division is invalid

        Returns:
            numerator / denominator if valid, else default
        """
        if not SafeMath.is_valid(numerator):
            logger.debug(f"safe_div: Invalid numerator {numerator}, returning {default}")
            return default

        if not SafeMath.is_valid(denominator) or abs(denominator) < SafeMath.EPSILON:
            logger.debug(f"safe_div: Invalid/zero denominator {denominator}, returning {default}")
            return default

        result = numerator / denominator

        if not SafeMath.is_valid(result):
            logger.debug(f"safe_div: Result {result} invalid, returning {default}")
            return default

        return result

    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        """
        Clamp value to [lower, upper] range with NaN protection

        Args:
            value: Value to clamp
            lower: Lower bound
            upper: Upper bound

        Returns:
            Clamped value, or midpoint if value is invalid
        """
        if not SafeMath.is_valid(value):
            midpoint = (lower + upper) / 2.0
            logger.debug(f"clamp: Invalid value {value}, returning midpoint {midpoint}")
            return midpoint

        return max(lower, min(upper, value))

    @staticmethod
    def safe_sqrt(value: float, default: float = 0.0) -> float:
        """Square root with negative protection"""
        if not SafeMath.is_valid(value) or value < 0:
            return default
        return math.sqrt(value)

    @staticmethod
    def safe_log(value: float, default: float = 0.0) -> float:
        """Natural log with non-positive protection"""
        if not SafeMath.is_valid(value) or value <= 0:
            return default
        result = math.log(value)
        return result if SafeMath.is_valid(result) else default

    @staticmethod
    def safe_exp(value: float, default: float = 1.0) -> float:
        """Exponential with overflow protection"""
        if not SafeMath.is_valid(value):
            return default
        try:
            result = math.exp(value)
            return result if SafeMath.is_valid(result) else default
        except OverflowError:
            logger.debug(f"safe_exp: Overflow on {value}, returning {default}")
            return default

    @staticmethod
    def sanitize(value: float, default: float = 0.0) -> float:
        """Replace NaN/Inf with default"""
        return value if SafeMath.is_valid(value) else default


class SafeArray:
    """Safe array and deque access with bounds checking"""

    @staticmethod
    def safe_get(arr: list | deque, index: int, default: Any = None) -> Any:
        """
        Get element from list/deque with bounds checking

        Args:
            arr: List or deque
            index: Index to access
            default: Value to return if out of bounds

        Returns:
            arr[index] if in bounds, else default
        """
        if arr is None or not hasattr(arr, "__getitem__"):
            logger.debug(f"safe_get: Invalid array type {type(arr)}")
            return default

        try:
            length = len(arr)
        except TypeError:
            logger.debug(f"safe_get: Cannot get length of {type(arr)}")
            return default

        if not (0 <= index < length):
            logger.debug(f"safe_get: Index {index} out of bounds [0, {length})")
            return default

        try:
            return arr[index]
        except (IndexError, KeyError, TypeError) as e:
            logger.debug(f"safe_get: Access failed: {e}")
            return default

    @staticmethod
    def safe_get_series(arr: list | deque, bars_ago: int, default: Any = None) -> Any:
        """
        Get element from series using bars-ago indexing

        Args:
            arr: List or deque (ordered oldest to newest)
            bars_ago: 0 = current (last), 1 = previous, etc.
            default: Value to return if out of bounds

        Returns:
            arr[-1-bars_ago] if valid, else default

        Example:
            bars = [old, ..., prev, current]
            safe_get_series(bars, 0) → current
            safe_get_series(bars, 1) → prev
        """
        if arr is None or not hasattr(arr, "__getitem__"):
            return default

        try:
            length = len(arr)
        except TypeError:
            return default

        if bars_ago < 0:
            logger.debug(f"safe_get_series: Negative bars_ago {bars_ago}")
            return default

        index = length - 1 - bars_ago
        return SafeArray.safe_get(arr, index, default)

    @staticmethod
    def safe_last(arr: list | deque, default: Any = None) -> Any:
        """Get last element with bounds checking"""
        return SafeArray.safe_get_series(arr, 0, default)

    @staticmethod
    def safe_slice(arr: list | deque, start: int | None = None, end: int | None = None) -> list | deque:
        """
        Safe slice with bounds correction

        Args:
            arr: List or deque
            start: Start index (None = beginning)
            end: End index (None = end)

        Returns:
            Sliced array (empty if invalid)
        """
        if arr is None or not hasattr(arr, "__getitem__"):
            return [] if isinstance(arr, list) else deque()

        try:
            return arr[start:end]
        except (TypeError, ValueError) as e:
            logger.debug(f"safe_slice: Slice failed: {e}")
            return [] if isinstance(arr, list) else deque()

    @staticmethod
    def is_empty(arr: list | deque | None) -> bool:
        """Check if array is None or empty"""
        if arr is None:
            return True
        try:
            return len(arr) == 0
        except TypeError:
            return True


class SafeDeque:
    """Wrapper for deque with safe operations"""

    def __init__(self, maxlen: int | None = None, name: str = "deque"):
        self._deque = deque(maxlen=maxlen)
        self._name = name
        self._maxlen = maxlen

    def append(self, item: Any) -> None:
        """Append item to deque"""
        self._deque.append(item)

    def get(self, index: int, default: Any = None) -> Any:
        """Get element with bounds checking"""
        return SafeArray.safe_get(self._deque, index, default)

    def get_series(self, bars_ago: int, default: Any = None) -> Any:
        """Get element using bars-ago indexing"""
        return SafeArray.safe_get_series(self._deque, bars_ago, default)

    def last(self, default: Any = None) -> Any:
        """Get last element"""
        return SafeArray.safe_last(self._deque, default)

    def __len__(self) -> int:
        return len(self._deque)

    def __iter__(self):
        return iter(self._deque)

    def __getitem__(self, index):
        """Direct access (use get() for safe access)"""
        return self._deque[index]

    @property
    def is_empty(self) -> bool:
        return len(self._deque) == 0

    @property
    def maxlen(self) -> int | None:
        return self._maxlen


# Convenience functions for common operations
def safe_mean(values: list[float], default: float = 0.0) -> float:
    """Calculate mean with NaN/empty protection"""
    if not values:
        return default

    valid_values = [v for v in values if SafeMath.is_valid(v)]
    if not valid_values:
        return default

    return sum(valid_values) / len(valid_values)


def safe_std(values: list[float], default: float = 0.0) -> float:
    """Calculate standard deviation with NaN/empty protection"""
    if not values or len(values) < MIN_VALUES_FOR_STD:
        return default

    valid_values = [v for v in values if SafeMath.is_valid(v)]
    if len(valid_values) < MIN_VALUES_FOR_STD:
        return default

    mean = safe_mean(valid_values, 0.0)
    variance = sum((v - mean) ** 2 for v in valid_values) / len(valid_values)

    return SafeMath.safe_sqrt(variance, default)


def safe_percentile(values: list[float], percentile: float, default: float = 0.0) -> float:
    """Calculate percentile with NaN/empty protection"""
    if not values:
        return default

    valid_values = sorted([v for v in values if SafeMath.is_valid(v)])
    if not valid_values:
        return default

    k = (len(valid_values) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return valid_values[int(k)]

    d0 = valid_values[int(f)] * (c - k)
    d1 = valid_values[int(c)] * (k - f)
    return d0 + d1


if __name__ == "__main__":
    # Self-test
    print("SafeMath Tests:")
    print(f"  safe_div(10, 2) = {SafeMath.safe_div(10, 2)}")  # 5.0
    print(f"  safe_div(10, 0) = {SafeMath.safe_div(10, 0)}")  # 0.0
    print(f"  safe_div(float('nan'), 2) = {SafeMath.safe_div(float('nan'), 2)}")  # 0.0
    print(f"  clamp(5, 0, 10) = {SafeMath.clamp(5, 0, 10)}")  # 5
    print(f"  clamp(15, 0, 10) = {SafeMath.clamp(15, 0, 10)}")  # 10
    print(f"  clamp(float('inf'), 0, 10) = {SafeMath.clamp(float('inf'), 0, 10)}")  # 5.0

    print("\nSafeArray Tests:")
    arr = [1, 2, 3, 4, 5]
    print(f"  safe_get(arr, 2) = {SafeArray.safe_get(arr, 2)}")  # 3
    print(f"  safe_get(arr, 10) = {SafeArray.safe_get(arr, 10)}")  # None
    print(f"  safe_get_series(arr, 0) = {SafeArray.safe_get_series(arr, 0)}")  # 5
    print(f"  safe_get_series(arr, 1) = {SafeArray.safe_get_series(arr, 1)}")  # 4

    print("\nSafeDeque Tests:")
    sd = SafeDeque(maxlen=3, name="test")
    sd.append(1)
    sd.append(2)
    sd.append(3)
    print(f"  last() = {sd.last()}")  # 3
    print(f"  get_series(1) = {sd.get_series(1)}")  # 2
    sd.append(4)  # Should evict 1
    print(f"  After append(4), get_series(2) = {sd.get_series(2)}")  # 2

    print("\nAll tests passed ✓")


# ----------------------------
# Time utilities (UTC required for FIX protocol)
# ----------------------------
def utc_ts_ms() -> str:
    """
    Generate FIX protocol UTCTimestamp.

    Format: YYYYMMDD-HH:MM:SS.sss (UTC)
    Required for FIX protocol timestamps (Tag 52, etc.)

    Returns:
        UTC timestamp string with milliseconds
    """
    return datetime.now(UTC).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]


def utc_now() -> datetime:
    """
    Get current UTC datetime.

    Returns:
        Current datetime in UTC timezone
    """
    return datetime.now(UTC)


def save_json_atomic(path: str | Path, data: dict | list, *, indent: int = 2) -> None:
    """
    Atomically write JSON to *path*.

    Strategy: write to a temp file in the same directory, then ``os.replace``
    (atomic on POSIX) to the target.  If the process crashes mid-write the
    original file is untouched.

    Args:
        path:   Destination file path.
        data:   JSON-serialisable data.
        indent: Pretty-print indent (default 2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
