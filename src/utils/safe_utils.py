#!/usr/bin/env python3
"""Defensive Programming Utilities — SafeArray, SafeDeque, and safe statistics.

SafeMath is the canonical safe-math implementation in safe_math.py.
Re-exported here for backwards compatibility with existing importers.
"""

import logging
import math
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.persistence.json_io import save_json_atomic as save_json_atomic_impl
from src.utils.safe_math import SafeMath  # canonical source — do not duplicate

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

MIN_VALUES_FOR_STD: int = 2


class SafeArray:
    """Safe array and deque access with bounds checking."""

    @staticmethod
    def safe_get(arr: list | deque, index: int, default: Any = None) -> Any:
        """Get element from list/deque with bounds checking.

        Args:
            arr: List or deque
            index: Index to access
            default: Value to return if out of bounds

        Returns:
            arr[index] if in bounds, else default

        """
        if arr is None or not hasattr(arr, "__getitem__"):
            logger.debug("safe_get: Invalid array type %s", type(arr))
            return default

        try:
            length = len(arr)
        except TypeError:
            logger.debug("safe_get: Cannot get length of %s", type(arr))
            return default

        if not (0 <= index < length):
            logger.debug("safe_get: Index %s out of bounds [0, %s)", index, length)
            return default

        try:
            return arr[index]
        except (IndexError, KeyError, TypeError) as e:
            logger.debug("safe_get: Access failed: %s", e)
            return default

    @staticmethod
    def safe_get_series(arr: list | deque, bars_ago: int, default: Any = None) -> Any:
        """Get element from series using bars-ago indexing.

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
            logger.debug("safe_get_series: Negative bars_ago %s", bars_ago)
            return default

        index = length - 1 - bars_ago
        return SafeArray.safe_get(arr, index, default)

    @staticmethod
    def safe_last(arr: list | deque, default: Any = None) -> Any:
        """Get last element with bounds checking."""
        return SafeArray.safe_get_series(arr, 0, default)

    @staticmethod
    def safe_slice(arr: list | deque, start: int | None = None, end: int | None = None) -> list | deque:
        """Safe slice with bounds correction.

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
            logger.debug("safe_slice: Slice failed: %s", e)
            return [] if isinstance(arr, list) else deque()

    @staticmethod
    def is_empty(arr: list | deque | None) -> bool:
        """Check if array is None or empty."""
        if arr is None:
            return True
        try:
            return len(arr) == 0
        except TypeError:
            return True


class SafeDeque:
    """Wrapper for deque with safe operations."""

    def __init__(self, maxlen: int | None = None, name: str = "deque") -> None:
        self._deque = deque(maxlen=maxlen)
        self._name = name
        self._maxlen = maxlen

    def append(self, item: Any) -> None:
        """Append item to deque."""
        self._deque.append(item)

    def get(self, index: int, default: Any = None) -> Any:
        """Get element with bounds checking."""
        return SafeArray.safe_get(self._deque, index, default)

    def get_series(self, bars_ago: int, default: Any = None) -> Any:
        """Get element using bars-ago indexing."""
        return SafeArray.safe_get_series(self._deque, bars_ago, default)

    def last(self, default: Any = None) -> Any:
        """Get last element."""
        return SafeArray.safe_last(self._deque, default)

    def __len__(self) -> int:
        return len(self._deque)

    def __iter__(self) -> "Iterator[Any]":
        return iter(self._deque)

    def __getitem__(self, index: int) -> Any:
        """Direct access (use get() for safe access)."""
        return self._deque[index]

    @property
    def is_empty(self) -> bool:
        return len(self._deque) == 0

    @property
    def maxlen(self) -> int | None:
        return self._maxlen


# Convenience functions for common operations
def safe_mean(values: list[float], default: float = 0.0) -> float:
    """Calculate mean with NaN/empty protection."""
    if not values:
        return default

    valid_values = [v for v in values if SafeMath.is_valid(v)]
    if not valid_values:
        return default

    return sum(valid_values) / len(valid_values)


def safe_std(values: list[float], default: float = 0.0) -> float:
    """Calculate standard deviation with NaN/empty protection."""
    if not values or len(values) < MIN_VALUES_FOR_STD:
        return default

    valid_values = [v for v in values if SafeMath.is_valid(v)]
    if len(valid_values) < MIN_VALUES_FOR_STD:
        return default

    mean = safe_mean(valid_values, 0.0)
    variance = sum((v - mean) ** 2 for v in valid_values) / len(valid_values)

    return SafeMath.safe_sqrt(variance, default)


def safe_percentile(values: list[float], percentile: float, default: float = 0.0) -> float:
    """Calculate percentile with NaN/empty protection."""
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


# ----------------------------
# Time utilities (UTC required for FIX protocol)
# ----------------------------
def utc_ts_ms() -> str:
    """Generate FIX protocol UTCTimestamp.

    Format: YYYYMMDD-HH:MM:SS.sss (UTC)
    Required for FIX protocol timestamps (Tag 52, etc.)

    Returns:
        UTC timestamp string with milliseconds

    """
    return datetime.now(UTC).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]


def utc_now() -> datetime:
    """Get current UTC datetime.

    Returns:
        Current datetime in UTC timezone

    """
    return datetime.now(UTC)


def save_json_atomic(path: str | Path, data: dict | list, *, indent: int = 2) -> None:
    """Atomically write JSON to *path*.

    Strategy: write to a temp file in the same directory, then ``os.replace``
    (atomic on POSIX) to the target.  If the process crashes mid-write the
    original file is untouched.

    Args:
        path:   Destination file path.
        data:   JSON-serialisable data.
        indent: Pretty-print indent (default 2).

    """
    save_json_atomic_impl(path, data, indent=indent)
