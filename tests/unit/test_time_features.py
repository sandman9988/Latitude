"""Tests for src.features.time_features – SafeMath, RingBuffer, TimeFeatures."""

import datetime as dt
import math

import pytest

from src.features.time_features import RingBuffer, SafeMath, TimeFeatures


# ---------------------------------------------------------------------------
# SafeMath
# ---------------------------------------------------------------------------


class TestSafeMath:
    def test_is_valid_normal(self):
        assert SafeMath.is_valid(42.0) is True

    def test_is_valid_nan(self):
        assert SafeMath.is_valid(float("nan")) is False

    def test_is_valid_inf(self):
        assert SafeMath.is_valid(float("inf")) is False

    def test_safe_div_normal(self):
        assert SafeMath.safe_div(10.0, 2.0) == pytest.approx(5.0)

    def test_safe_div_zero_denominator(self):
        assert SafeMath.safe_div(10.0, 0.0) == pytest.approx(0.0)

    def test_safe_div_near_zero(self):
        assert SafeMath.safe_div(10.0, 1e-15) == pytest.approx(0.0)

    def test_safe_div_custom_default(self):
        assert SafeMath.safe_div(1.0, 0.0, default=-1.0) == pytest.approx(-1.0)

    def test_clamp_within_range(self):
        assert SafeMath.clamp(5.0, 0.0, 10.0) == pytest.approx(5.0)

    def test_clamp_below(self):
        assert SafeMath.clamp(-5.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_clamp_above(self):
        assert SafeMath.clamp(15.0, 0.0, 10.0) == pytest.approx(10.0)

    def test_clamp_nan_returns_min(self):
        assert SafeMath.clamp(float("nan"), 0.0, 10.0) == pytest.approx(0.0)

    def test_normalize_angle_positive(self):
        assert SafeMath.normalize_angle(450.0) == pytest.approx(90.0)

    def test_normalize_angle_negative(self):
        assert SafeMath.normalize_angle(-90.0) == pytest.approx(270.0)

    def test_normalize_angle_nan(self):
        assert SafeMath.normalize_angle(float("nan")) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RingBuffer
# ---------------------------------------------------------------------------


class TestRingBuffer:
    def test_init_invalid_size(self):
        with pytest.raises(ValueError, match="must be > 0"):
            RingBuffer(0)

    def test_add_and_mean(self):
        rb = RingBuffer(5)
        for v in [1, 2, 3, 4, 5]:
            rb.add(float(v))
        assert rb.mean() == pytest.approx(3.0)

    def test_wraparound(self):
        rb = RingBuffer(3)
        for v in range(10):
            rb.add(float(v))
        # Buffer should contain [7, 8, 9]
        assert rb.mean() == pytest.approx(8.0)
        assert rb.count == 3

    def test_skip_nan(self):
        rb = RingBuffer(5)
        rb.add(1.0)
        rb.add(float("nan"))
        rb.add(3.0)
        assert rb.count == 2

    def test_skip_inf(self):
        rb = RingBuffer(5)
        rb.add(float("inf"))
        assert rb.count == 0

    def test_std_below_min_sample(self):
        rb = RingBuffer(5)
        rb.add(10.0)
        assert rb.std() == pytest.approx(0.0)

    def test_std_constant_values(self):
        rb = RingBuffer(5)
        for _ in range(5):
            rb.add(7.0)
        assert rb.std() == pytest.approx(0.0, abs=1e-10)

    def test_std_known_values(self):
        rb = RingBuffer(4)
        for v in [2, 4, 4, 4]:
            rb.add(float(v))
        # population std: sqrt(0.75) = 0.866...
        assert rb.std() == pytest.approx(math.sqrt(0.75), abs=0.01)

    def test_get_stats_empty(self):
        rb = RingBuffer(5)
        stats = rb.get_stats()
        assert stats["count"] == pytest.approx(0.0)
        assert stats["min"] == pytest.approx(0.0)  # guarded default
        assert stats["max"] == pytest.approx(0.0)

    def test_get_stats_filled(self):
        rb = RingBuffer(5)
        for v in [1, 2, 3]:
            rb.add(float(v))
        stats = rb.get_stats()
        assert stats["count"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(3.0)
        assert stats["sum"] == pytest.approx(6.0)

    def test_mean_empty(self):
        rb = RingBuffer(3)
        assert rb.mean() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TimeFeatures
# ---------------------------------------------------------------------------


def _utc(year=2026, month=1, day=14, hour=10, minute=0):
    """Create a timezone-aware UTC datetime."""
    return dt.datetime(year, month, day, hour, minute, tzinfo=dt.UTC)


class TestTimeFeatures:

    def test_get_all_features_none_input(self):
        tf = TimeFeatures()
        f = tf.get_all_features(None)
        assert all(v == pytest.approx(0.0) for v in f.values())

    def test_get_all_features_naive_invalid(self):
        tf = TimeFeatures()
        # Naive datetime (no tz) → should be treated as invalid
        naive = dt.datetime(2026, 1, 14, 10, 0)
        f = tf.get_all_features(naive)
        assert all(v == pytest.approx(0.0) for v in f.values())

    def test_get_all_features_keys(self):
        tf = TimeFeatures()
        f = tf.get_all_features(_utc())
        expected_keys = {
            "minutes_to_session_close", "minutes_to_rollover",
            "minutes_to_day_end", "day_of_week_norm",
            "is_friday_close_near", "hour_of_day", "minute_of_hour",
        }
        assert set(f.keys()) == expected_keys

    def test_minutes_to_rollover_before_rollover(self):
        tf = TimeFeatures()
        t = _utc(hour=20)  # 2 hours before 22:00
        mins = tf.minutes_to_rollover(t)
        assert 110 <= mins <= 130  # ~120 minutes

    def test_minutes_to_rollover_after_rollover(self):
        tf = TimeFeatures()
        t = _utc(hour=23)  # 1 hour after 22:00
        mins = tf.minutes_to_rollover(t)
        # Next rollover is tomorrow
        assert 1300 <= mins <= 1400

    def test_minutes_to_day_end(self):
        tf = TimeFeatures()
        t = _utc(hour=23, minute=0)  # 60 minutes to midnight
        mins = tf.minutes_to_day_end(t)
        assert 55 <= mins <= 65

    def test_day_of_week_encoded_monday(self):
        tf = TimeFeatures()
        monday = _utc(year=2026, month=1, day=12, hour=10)  # Monday
        val = tf.day_of_week_encoded(monday)
        assert val == pytest.approx(0.0)  # Monday = 0 / 6 = 0.0

    def test_day_of_week_encoded_sunday(self):
        tf = TimeFeatures()
        sunday = _utc(year=2026, month=1, day=18, hour=10)  # Sunday
        val = tf.day_of_week_encoded(sunday)
        assert val == pytest.approx(1.0)  # Sunday = 6 / 6 = 1.0

    def test_is_friday_close_approaching_true(self):
        tf = TimeFeatures()
        # Friday at 20:00 UTC → ~2h before 22:00 close
        friday = _utc(year=2026, month=1, day=16, hour=20)  # Friday
        assert tf.is_friday_close_approaching(friday, threshold_hours=4.0) is True

    def test_is_friday_close_approaching_false_wrong_day(self):
        tf = TimeFeatures()
        wednesday = _utc(year=2026, month=1, day=14, hour=20)  # Wednesday
        # Should still calculate minutes to next Friday
        result = tf.is_friday_close_approaching(wednesday, threshold_hours=4.0)
        assert result is False  # Too far away

    def test_caching(self):
        tf = TimeFeatures()
        t = _utc()
        # First call populates cache
        tf.minutes_to_rollover(t)
        assert len(tf.cache) > 0
        # Second call should use cache (just verify no error)
        tf.minutes_to_rollover(t)

    def test_cache_cleared_when_stale(self):
        tf = TimeFeatures()
        tf.cache_ttl_seconds = 0.001  # Very short TTL
        t1 = _utc(hour=10)
        tf.minutes_to_rollover(t1)
        # Slightly later time should trigger cache clear
        t2 = _utc(hour=10, minute=5)
        tf._clear_cache_if_stale(t2)
        assert len(tf.cache) == 0

    def test_feature_statistics(self):
        tf = TimeFeatures()
        for h in range(5):
            tf.get_all_features(_utc(hour=10 + h))
        stats = tf.get_feature_statistics()
        assert "minutes_to_rollover" in stats
        assert stats["minutes_to_rollover"]["count"] >= 1
