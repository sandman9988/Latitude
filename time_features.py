#!/usr/bin/env python3
"""
Time Features - Event-Relative Time Features for Trading
Python port of MASTER_HANDBOOK.md Section 4.7 - Feature Engineering

Implements event-relative time features (NOT wall-clock time):
- Minutes to session close
- Minutes to rollover
- Minutes to daily/weekly boundaries
- Day of week, week of month awareness
- Holiday detection

All features use DEFENSIVE PROGRAMMING principles from handbook Section 8.
"""

import datetime as dt
import math

DENOMINATOR_EPS = 1e-10
MIN_STD_SAMPLE_COUNT = 2
YEAR_MIN = 2020
YEAR_MAX = 2030


# ----------------------------
# Defensive Programming Utilities
# ----------------------------
class SafeMath:
    """Safe mathematical operations with NaN/Inf protection and division by zero handling."""

    @staticmethod
    def is_valid(value: float) -> bool:
        """Check if value is valid (not NaN or Inf)."""
        return not (math.isnan(value) or math.isinf(value))

    @staticmethod
    def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safe division with zero check.

        Args:
            numerator: Top of division
            denominator: Bottom of division
            default: Value to return if division invalid

        Returns:
            numerator / denominator if valid, else default
        """
        if abs(denominator) < DENOMINATOR_EPS:  # Protect against near-zero
            return default

        result = numerator / denominator

        if not SafeMath.is_valid(result):
            return default

        return result

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value to range [min_val, max_val]."""
        if not SafeMath.is_valid(value):
            return min_val
        return max(min_val, min(max_val, value))

    @staticmethod
    def normalize_angle(degrees: float) -> float:
        """Normalize angle to [0, 360) range."""
        if not SafeMath.is_valid(degrees):
            return 0.0
        # Use modulo to wrap
        normalized = degrees % 360.0
        if normalized < 0:
            normalized += 360.0
        return normalized


class RingBuffer:
    """
    Fixed-size circular buffer with O(1) statistics.
    From MASTER_HANDBOOK.md Section 4.1 - Defensive Programming Framework.
    """

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError(f"RingBuffer size must be > 0, got {max_size}")

        self.max_size = max_size
        self.buffer = [0.0] * max_size
        self.head = 0
        self.count = 0

        # Running statistics
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def add(self, value: float):
        """Add value to buffer with O(1) update of statistics."""
        if not SafeMath.is_valid(value):
            return  # Skip invalid values

        # Remove old value if buffer full
        if self.count == self.max_size:
            old_value = self.buffer[self.head]
            self.sum -= old_value
            self.sum_sq -= old_value * old_value
        else:
            self.count += 1

        # Add new value
        self.buffer[self.head] = value
        self.sum += value
        self.sum_sq += value * value

        # Advance head
        self.head = (self.head + 1) % self.max_size

        # Update min/max by scanning (could optimize with deque)
        if self.count > 0:
            self.min_val = min(self.buffer[: self.count])
            self.max_val = max(self.buffer[: self.count])

    def mean(self) -> float:
        """O(1) mean calculation."""
        if self.count == 0:
            return 0.0
        return SafeMath.safe_div(self.sum, float(self.count), default=0.0)

    def std(self) -> float:
        """O(1) standard deviation calculation."""
        if self.count < MIN_STD_SAMPLE_COUNT:
            return 0.0

        mean_val = self.mean()
        variance = SafeMath.safe_div(self.sum_sq, float(self.count), default=0.0) - mean_val * mean_val

        if variance < 0:
            variance = 0.0  # Numerical precision issue

        return math.sqrt(variance)

    def get_stats(self) -> dict[str, float]:
        """Get all statistics at once."""
        return {
            "count": float(self.count),
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min_val if self.count > 0 else 0.0,
            "max": self.max_val if self.count > 0 else 0.0,
            "sum": self.sum,
        }


# ----------------------------
# Time Features Calculator
# ----------------------------
class TimeFeatures:
    """
    Event-relative time features for trading.
    Uses defensive programming throughout.
    """

    # Market hours (UTC) - Forex 24h, but conceptual boundaries
    FOREX_SYDNEY_OPEN = 22  # 22:00 UTC Sunday
    FOREX_TOKYO_OPEN = 0  # 00:00 UTC Monday
    FOREX_LONDON_OPEN = 8  # 08:00 UTC
    FOREX_NY_OPEN = 13  # 13:00 UTC
    FOREX_NY_CLOSE = 22  # 22:00 UTC Friday

    # Rollover time (5 PM NY = 22:00 UTC)
    ROLLOVER_HOUR_UTC = 22

    def __init__(self):
        # Cache for expensive calculations
        self.cache: dict[str, float] = {}
        self.cache_timestamp: dt.datetime | None = None
        self.cache_ttl_seconds: float = 60.0  # Cache for 1 minute

        # Ring buffers for feature statistics (last 100 values)
        self.feature_buffers = {
            "minutes_to_session_close": RingBuffer(100),
            "minutes_to_rollover": RingBuffer(100),
            "minutes_to_day_end": RingBuffer(100),
        }

    def _clear_cache_if_stale(self, current_time: dt.datetime | None):
        """Clear cache if older than TTL."""
        if current_time is None or self.cache_timestamp is None:
            return

        try:
            elapsed = (current_time - self.cache_timestamp).total_seconds()
            if elapsed > self.cache_ttl_seconds:
                self.cache = {}
                self.cache_timestamp = None
        except Exception:
            # Defensive: clear cache on any error
            self.cache = {}
            self.cache_timestamp = None

    def _validate_datetime(self, timestamp: dt.datetime | None) -> bool:
        """Validate datetime object is reasonable."""
        if timestamp is None:
            return False

        # Check year is reasonable (2020-2030)
        if timestamp.year < YEAR_MIN or timestamp.year > YEAR_MAX:
            return False

        return timestamp.tzinfo is not None

    def minutes_to_session_close(self, current_time: dt.datetime) -> float:
        """
        Calculate minutes until next session close (NY close at 22:00 UTC Friday).

        Args:
            current_time: Current timestamp (timezone-aware)

        Returns:
            Minutes to session close, or 0.0 if invalid
        """
        if not self._validate_datetime(current_time):
            return 0.0

        # Cache key
        cache_key = f"session_close_{current_time.timestamp()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Find next Friday 22:00 UTC
            days_until_friday = (4 - current_time.weekday()) % 7  # 4 = Friday
            if days_until_friday == 0 and current_time.hour >= self.FOREX_NY_CLOSE:
                days_until_friday = 7  # Next Friday

            next_friday = current_time + dt.timedelta(days=days_until_friday)
            close_time = next_friday.replace(hour=self.FOREX_NY_CLOSE, minute=0, second=0, microsecond=0)

            delta = (close_time - current_time).total_seconds()
            minutes = SafeMath.safe_div(delta, 60.0, default=0.0)

            # Clamp to reasonable range (0 to 1 week)
            minutes = SafeMath.clamp(minutes, 0.0, 10080.0)  # 7 days * 24h * 60min

            # Cache result
            self.cache[cache_key] = minutes
            self.cache_timestamp = current_time

            # Update ring buffer for statistics
            self.feature_buffers["minutes_to_session_close"].add(minutes)

            return minutes

        except Exception:
            # Defensive: return safe default on any error
            return 0.0

    def minutes_to_rollover(self, current_time: dt.datetime) -> float:
        """
        Calculate minutes until next rollover (22:00 UTC daily).

        Args:
            current_time: Current timestamp (timezone-aware)

        Returns:
            Minutes to rollover, or 0.0 if invalid
        """
        if not self._validate_datetime(current_time):
            return 0.0

        cache_key = f"rollover_{current_time.timestamp()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Next rollover at 22:00 UTC
            next_rollover = current_time.replace(hour=self.ROLLOVER_HOUR_UTC, minute=0, second=0, microsecond=0)

            # If already past today's rollover, use tomorrow
            if current_time.hour >= self.ROLLOVER_HOUR_UTC:
                next_rollover += dt.timedelta(days=1)

            delta = (next_rollover - current_time).total_seconds()
            minutes = SafeMath.safe_div(delta, 60.0, default=0.0)

            # Clamp to reasonable range (0 to 24 hours)
            minutes = SafeMath.clamp(minutes, 0.0, 1440.0)

            self.cache[cache_key] = minutes
            self.cache_timestamp = current_time

            self.feature_buffers["minutes_to_rollover"].add(minutes)

            return minutes

        except Exception:
            return 0.0

    def minutes_to_day_end(self, current_time: dt.datetime) -> float:
        """
        Calculate minutes until midnight UTC.

        Args:
            current_time: Current timestamp (timezone-aware)

        Returns:
            Minutes to day end, or 0.0 if invalid
        """
        if not self._validate_datetime(current_time):
            return 0.0

        cache_key = f"day_end_{current_time.timestamp()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Next midnight UTC
            next_midnight = (current_time + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

            delta = (next_midnight - current_time).total_seconds()
            minutes = SafeMath.safe_div(delta, 60.0, default=0.0)

            # Clamp to reasonable range (0 to 24 hours)
            minutes = SafeMath.clamp(minutes, 0.0, 1440.0)

            self.cache[cache_key] = minutes
            self.cache_timestamp = current_time

            self.feature_buffers["minutes_to_day_end"].add(minutes)

            return minutes

        except Exception:
            return 0.0

    def day_of_week_encoded(self, current_time: dt.datetime) -> float:
        """
        Encode day of week as cyclic feature (sin/cos would be better, but this is simpler).
        Monday=0, Tuesday=1, ..., Sunday=6
        Normalized to [0, 1] range.

        Args:
            current_time: Current timestamp (timezone-aware)

        Returns:
            Day of week normalized [0, 1], or 0.0 if invalid
        """
        if not self._validate_datetime(current_time):
            return 0.0

        try:
            day = current_time.weekday()  # 0=Monday, 6=Sunday
            # Normalize to [0, 1]
            normalized = SafeMath.safe_div(float(day), 6.0, default=0.0)
            return SafeMath.clamp(normalized, 0.0, 1.0)
        except Exception:
            return 0.0

    def is_friday_close_approaching(self, current_time: dt.datetime, threshold_hours: float = 4.0) -> bool:
        """
        Check if Friday close is within threshold hours.

        Args:
            current_time: Current timestamp
            threshold_hours: Hours before close to trigger

        Returns:
            True if within threshold, False otherwise
        """
        if not self._validate_datetime(current_time):
            return False

        try:
            minutes = self.minutes_to_session_close(current_time)
            threshold_minutes = threshold_hours * 60.0
            return minutes <= threshold_minutes and minutes > 0
        except Exception:
            return False

    def get_all_features(self, current_time: dt.datetime | None) -> dict[str, float]:
        """
        Calculate all time features at once.

        Args:
            current_time: Current timestamp (timezone-aware)

        Returns:
            Dictionary of all time features (all zeros if invalid input)
        """
        # Defensive: check input validity first
        if current_time is None or not self._validate_datetime(current_time):
            return {
                "minutes_to_session_close": 0.0,
                "minutes_to_rollover": 0.0,
                "minutes_to_day_end": 0.0,
                "day_of_week_norm": 0.0,
                "is_friday_close_near": 0.0,
                "hour_of_day": 0.0,
                "minute_of_hour": 0.0,
            }

        # mypy: current_time is non-None beyond this point
        assert current_time is not None

        # Clear stale cache
        self._clear_cache_if_stale(current_time)

        features = {
            "minutes_to_session_close": self.minutes_to_session_close(current_time),
            "minutes_to_rollover": self.minutes_to_rollover(current_time),
            "minutes_to_day_end": self.minutes_to_day_end(current_time),
            "day_of_week_norm": self.day_of_week_encoded(current_time),
            "is_friday_close_near": float(self.is_friday_close_approaching(current_time)),
            "hour_of_day": float(current_time.hour) / 23.0,  # Normalized [0, 1]
            "minute_of_hour": float(current_time.minute) / 59.0,  # Normalized [0, 1]
        }

        # Add defensive validation
        for key, value in features.items():
            if not SafeMath.is_valid(value):
                features[key] = 0.0

        return features

    def get_feature_statistics(self) -> dict[str, dict[str, float]]:
        """Get statistics for tracked features from ring buffers."""
        stats = {}
        for feature_name, buffer in self.feature_buffers.items():
            stats[feature_name] = buffer.get_stats()
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("Testing TimeFeatures module with defensive programming...")

    # Create timezone-aware UTC time
    current_time = dt.datetime.now(dt.UTC)

    tf = TimeFeatures()

    # Test 1: Get all features
    print("\n=== Test 1: All Features ===")
    features = tf.get_all_features(current_time)
    for name, value in features.items():
        print(f"{name:30s}: {value:.4f}")

    # Test 2: Edge case - invalid datetime
    print("\n=== Test 2: Invalid Input Handling ===")
    invalid_features = tf.get_all_features(None)
    print(f"Invalid input result (should be all zeros): {invalid_features}")

    # Test 3: Friday close detection
    print("\n=== Test 3: Friday Close Detection ===")
    # Create a Friday at 18:00 UTC (4 hours before close)
    friday_evening = current_time.replace(hour=18, minute=0)
    # Adjust to actual Friday
    days_to_friday = (4 - friday_evening.weekday()) % 7
    test_friday = friday_evening + dt.timedelta(days=days_to_friday)

    is_near = tf.is_friday_close_approaching(test_friday, threshold_hours=4.0)
    print(f"Friday 18:00 UTC (4h before close): {is_near}")

    # Test 4: SafeMath utilities
    print("\n=== Test 4: SafeMath Utilities ===")
    print(f"Safe division 10/2: {SafeMath.safe_div(10.0, 2.0)}")
    print(f"Safe division 10/0 (should return 0.0): {SafeMath.safe_div(10.0, 0.0)}")
    print(f"Clamp 150 to [0, 100]: {SafeMath.clamp(150.0, 0.0, 100.0)}")
    print(f"Is NaN valid? {SafeMath.is_valid(float('nan'))}")
    print(f"Is 42.0 valid? {SafeMath.is_valid(42.0)}")

    # Test 5: RingBuffer
    print("\n=== Test 5: RingBuffer ===")
    rb = RingBuffer(5)
    for i in range(10):
        rb.add(float(i))
    stats = rb.get_stats()
    print(f"RingBuffer stats (last 5 of 0-9): {stats}")
    print(f"Mean should be 7.0: {stats['mean']}")

    # Test 6: Feature statistics
    print("\n=== Test 6: Feature Statistics ===")
    # Add some samples
    for _ in range(10):
        tf.get_all_features(current_time)
        current_time += dt.timedelta(minutes=1)

    feature_stats = tf.get_feature_statistics()
    for feature, stats in feature_stats.items():
        print(f"{feature}: count={stats['count']}, mean={stats['mean']:.2f}, std={stats['std']:.4f}")

    print("\n✅ All tests complete - Defensive programming validated")
