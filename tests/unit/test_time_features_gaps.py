"""Gap tests for src.features.time_features.

Targets uncovered production lines:
- Line 55: SafeMath.safe_div producing NaN (inf/inf) → return default
- Line 140: RingBuffer.std variance < 0 protection
- Lines 198-201: _clear_cache_if_stale exception path
- Lines 256-258: minutes_to_rollover past-rollover (hour >= 22)
- Lines 298-299: day_of_week_encoded valid path
- Lines 335-336: is_friday_close_approaching true path
- Lines 358-359, 379-380: get_all_features valid paths
- Line 423: get_feature_statistics path
"""

import datetime as dt
import math
from unittest.mock import patch, PropertyMock

import pytest

from src.features.time_features import RingBuffer, SafeMath, TimeFeatures


# ---------------------------------------------------------------------------
# SafeMath.safe_div NaN result (line 55)
# ---------------------------------------------------------------------------
class TestSafeDivNaN:
    def test_inf_over_inf_returns_default(self):
        """inf / inf = nan, which triggers the is_valid check."""
        result = SafeMath.safe_div(float("inf"), float("inf"), default=-1.0)
        assert result == pytest.approx(-1.0)

    def test_nan_numerator_returns_nan_caught(self):
        """NaN numerator with valid denominator → result is NaN → returns default."""
        result = SafeMath.safe_div(float("nan"), 1.0, default=42.0)
        # nan / 1.0 = nan → not valid → default
        assert result == pytest.approx(42.0)

    def test_valid_division_works(self):
        """Sanity: normal division returns correct result."""
        assert SafeMath.safe_div(10.0, 2.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# RingBuffer variance < 0 (line 140)
# ---------------------------------------------------------------------------
class TestRingBufferNegativeVariance:
    def test_forced_negative_variance(self):
        """Force variance < 0 via direct attribute manipulation."""
        buf = RingBuffer(max_size=10)
        # Add values to get past MIN_STD_SAMPLE_COUNT
        for v in [1e15, 1e15 + 1, 1e15 + 2]:
            buf.add(v)

        # Force internal state to create negative variance scenario
        # variance = sum_sq/count - mean^2
        # If sum_sq is artificially low, variance can be negative
        buf.sum_sq = 0.0  # Force sum_sq to 0 while sum is large

        std_val = buf.std()
        assert std_val == pytest.approx(0.0)  # Should clamp negative variance to 0

    def test_normal_std(self):
        """Sanity: normal std computation works."""
        buf = RingBuffer(max_size=100)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            buf.add(v)
        std_val = buf.std()
        assert std_val > 0


# ---------------------------------------------------------------------------
# _clear_cache_if_stale exception path (lines 198-201)
# ---------------------------------------------------------------------------
class TestClearCacheException:
    def test_exception_clears_cache(self):
        """When subtraction raises, cache is cleared defensively."""
        tf = TimeFeatures()

        # Populate cache
        t = dt.datetime(2026, 1, 5, 10, 0, tzinfo=dt.UTC)
        tf.minutes_to_session_close(t)
        assert len(tf.cache) > 0

        # Set cache_timestamp to something that will crash on subtract
        tf.cache_timestamp = "not-a-datetime"  # Will fail: str - datetime

        # This should trigger the exception path and clear cache
        tf._clear_cache_if_stale(dt.datetime(2026, 1, 5, 12, 0, tzinfo=dt.UTC))
        assert len(tf.cache) == 0
        assert tf.cache_timestamp is None


# ---------------------------------------------------------------------------
# minutes_to_rollover past-rollover (lines 256-258)
# ---------------------------------------------------------------------------
class TestRolloverPastHour:
    def test_past_rollover_wraps_to_next_day(self):
        """When hour >= 22 (ROLLOVER_HOUR_UTC), should wrap to next day."""
        tf = TimeFeatures()
        # 23:00 UTC → 1 hour past rollover → next rollover in ~23 hours
        late = dt.datetime(2026, 1, 5, 23, 0, tzinfo=dt.UTC)
        minutes = tf.minutes_to_rollover(late)
        # Should be approximately 23 * 60 = 1380 minutes
        assert 1370 <= minutes <= 1440

    def test_at_rollover_wraps_to_next_day(self):
        """Exactly at 22:00 → next day rollover."""
        tf = TimeFeatures()
        exact = dt.datetime(2026, 1, 5, 22, 0, tzinfo=dt.UTC)
        minutes = tf.minutes_to_rollover(exact)
        # Should be 24 * 60 = 1440 minutes
        assert minutes == pytest.approx(1440.0)

    def test_before_rollover(self):
        """Before 22:00 → same day rollover."""
        tf = TimeFeatures()
        early = dt.datetime(2026, 1, 5, 10, 0, tzinfo=dt.UTC)
        minutes = tf.minutes_to_rollover(early)
        # Should be 12 * 60 = 720 minutes
        assert minutes == pytest.approx(720.0)


# ---------------------------------------------------------------------------
# day_of_week_encoded valid path (lines 298-299)
# ---------------------------------------------------------------------------
class TestDayOfWeekEncoded:
    def test_monday(self):
        """Monday = 0 → normalized = 0/6 = 0.0."""
        tf = TimeFeatures()
        monday = dt.datetime(2026, 1, 5, 12, 0, tzinfo=dt.UTC)
        assert monday.weekday() == 0  # Confirm Monday
        assert tf.day_of_week_encoded(monday) == pytest.approx(0.0)

    def test_sunday(self):
        """Sunday = 6 → normalized = 6/6 = 1.0."""
        tf = TimeFeatures()
        sunday = dt.datetime(2026, 1, 4, 12, 0, tzinfo=dt.UTC)
        assert sunday.weekday() == 6  # Confirm Sunday
        assert tf.day_of_week_encoded(sunday) == pytest.approx(1.0)

    def test_wednesday(self):
        """Wednesday = 2 → normalized = 2/6 ≈ 0.333."""
        tf = TimeFeatures()
        wednesday = dt.datetime(2026, 1, 7, 12, 0, tzinfo=dt.UTC)
        assert wednesday.weekday() == 2  # Confirm Wednesday
        assert tf.day_of_week_encoded(wednesday) == pytest.approx(2.0 / 6.0)


# ---------------------------------------------------------------------------
# is_friday_close_approaching true path (lines 335-336)
# ---------------------------------------------------------------------------
class TestFridayCloseApproaching:
    def test_friday_within_threshold(self):
        """Friday 20:00 → 2h before close → within 4h threshold."""
        tf = TimeFeatures()
        friday = dt.datetime(2026, 1, 9, 20, 0, tzinfo=dt.UTC)
        assert friday.weekday() == 4  # Confirm Friday
        assert tf.is_friday_close_approaching(friday, threshold_hours=4.0) is True

    def test_friday_outside_threshold(self):
        """Friday 08:00 → 14h before close → outside threshold."""
        tf = TimeFeatures()
        friday_morning = dt.datetime(2026, 1, 9, 8, 0, tzinfo=dt.UTC)
        assert tf.is_friday_close_approaching(friday_morning, threshold_hours=4.0) is False

    def test_not_friday(self):
        """Non-Friday → not approaching Friday close (too far away)."""
        tf = TimeFeatures()
        tuesday = dt.datetime(2026, 1, 6, 20, 0, tzinfo=dt.UTC)
        assert tf.is_friday_close_approaching(tuesday, threshold_hours=4.0) is False


# ---------------------------------------------------------------------------
# get_all_features valid paths (lines 358-380)
# ---------------------------------------------------------------------------
class TestGetAllFeaturesValid:
    def test_valid_datetime_returns_all_keys(self):
        """Valid datetime produces all expected feature keys."""
        tf = TimeFeatures()
        t = dt.datetime(2026, 1, 7, 14, 30, tzinfo=dt.UTC)
        features = tf.get_all_features(t)

        expected_keys = {
            "minutes_to_session_close",
            "minutes_to_rollover",
            "minutes_to_day_end",
            "day_of_week_norm",
            "is_friday_close_near",
            "hour_of_day",
            "minute_of_hour",
        }
        assert set(features.keys()) == expected_keys

    def test_hour_of_day_normalized(self):
        """hour_of_day should be hour / 23.0."""
        tf = TimeFeatures()
        t = dt.datetime(2026, 1, 5, 14, 0, tzinfo=dt.UTC)
        features = tf.get_all_features(t)
        assert features["hour_of_day"] == pytest.approx(14.0 / 23.0)

    def test_minute_of_hour_normalized(self):
        """minute_of_hour should be minute / 59.0."""
        tf = TimeFeatures()
        t = dt.datetime(2026, 1, 5, 10, 45, tzinfo=dt.UTC)
        features = tf.get_all_features(t)
        assert features["minute_of_hour"] == pytest.approx(45.0 / 59.0)

    def test_nan_feature_value_sanitized(self):
        """If a feature returns NaN, get_all_features replaces with 0.0."""
        tf = TimeFeatures()
        t = dt.datetime(2026, 1, 5, 10, 0, tzinfo=dt.UTC)

        # Patch minutes_to_session_close to return NaN
        original = tf.minutes_to_session_close
        tf.minutes_to_session_close = lambda ct: float("nan")
        features = tf.get_all_features(t)
        assert features["minutes_to_session_close"] == pytest.approx(0.0)
        tf.minutes_to_session_close = original  # Restore

    def test_invalid_datetime_returns_zeros(self):
        """Out of range datetime → all zeros."""
        tf = TimeFeatures()
        old = dt.datetime(2015, 1, 1, 12, 0, tzinfo=dt.UTC)
        features = tf.get_all_features(old)
        assert all(v == pytest.approx(0.0) for v in features.values())


# ---------------------------------------------------------------------------
# get_feature_statistics (line 423)
# ---------------------------------------------------------------------------
class TestGetFeatureStatistics:
    def test_statistics_after_calls(self):
        """After calling get_all_features, ring buffers should have data."""
        tf = TimeFeatures()
        for hour in range(10, 15):
            t = dt.datetime(2026, 1, 5, hour, 0, tzinfo=dt.UTC)
            tf.get_all_features(t)

        stats = tf.get_feature_statistics()
        assert "minutes_to_session_close" in stats
        assert "minutes_to_rollover" in stats
        assert "minutes_to_day_end" in stats
        assert stats["minutes_to_session_close"]["count"] == pytest.approx(5.0)

    def test_statistics_empty(self):
        """No calls → empty buffers."""
        tf = TimeFeatures()
        stats = tf.get_feature_statistics()
        assert stats["minutes_to_session_close"]["count"] == pytest.approx(0.0)
