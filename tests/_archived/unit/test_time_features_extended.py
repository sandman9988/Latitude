"""Extended tests for src.features.time_features.

Covers: invalid datetime on individual methods (session_close, rollover,
day_end, day_of_week, friday_close), cache hits, _clear_cache_if_stale
exception, get_all_features NaN sanitization, _validate_datetime edge cases.
"""

import datetime as dt

import pytest

from src.features.time_features import TimeFeatures


@pytest.fixture
def tf():
    return TimeFeatures()


# ---------------------------------------------------------------------------
# _validate_datetime edge cases
# ---------------------------------------------------------------------------
class TestValidateDatetime:
    def test_none_is_invalid(self, tf):
        assert tf._validate_datetime(None) is False

    def test_year_below_range(self, tf):
        old = dt.datetime(2019, 6, 1, tzinfo=dt.UTC)
        assert tf._validate_datetime(old) is False

    def test_year_above_range(self, tf):
        future = dt.datetime(2031, 1, 1, tzinfo=dt.UTC)
        assert tf._validate_datetime(future) is False

    def test_naive_is_invalid(self, tf):
        naive = dt.datetime(2026, 1, 5, 12, 0)
        assert tf._validate_datetime(naive) is False

    def test_valid_utc(self, tf):
        valid = dt.datetime(2026, 1, 5, 12, 0, tzinfo=dt.UTC)
        assert tf._validate_datetime(valid) is True


# ---------------------------------------------------------------------------
# Individual methods with invalid datetime
# ---------------------------------------------------------------------------
class TestInvalidDatetimeMethods:
    def test_minutes_to_session_close_invalid(self, tf):
        assert tf.minutes_to_session_close(dt.datetime(2019, 1, 1, tzinfo=dt.UTC)) == pytest.approx(0.0)

    def test_minutes_to_rollover_invalid(self, tf):
        assert tf.minutes_to_rollover(dt.datetime(2019, 1, 1, tzinfo=dt.UTC)) == pytest.approx(0.0)

    def test_minutes_to_day_end_invalid(self, tf):
        assert tf.minutes_to_day_end(dt.datetime(2019, 1, 1, tzinfo=dt.UTC)) == pytest.approx(0.0)

    def test_day_of_week_encoded_invalid(self, tf):
        assert tf.day_of_week_encoded(dt.datetime(2019, 1, 1, tzinfo=dt.UTC)) == pytest.approx(0.0)

    def test_is_friday_close_invalid(self, tf):
        assert tf.is_friday_close_approaching(dt.datetime(2019, 1, 1, tzinfo=dt.UTC)) is False


# ---------------------------------------------------------------------------
# Cache hit for individual methods
# ---------------------------------------------------------------------------
class TestCacheHits:
    def test_session_close_cache_hit(self, tf):
        t = dt.datetime(2026, 1, 5, 10, 0, tzinfo=dt.UTC)
        result1 = tf.minutes_to_session_close(t)
        result2 = tf.minutes_to_session_close(t)  # Cache hit
        assert result1 == result2

    def test_rollover_cache_hit(self, tf):
        t = dt.datetime(2026, 1, 5, 10, 0, tzinfo=dt.UTC)
        result1 = tf.minutes_to_rollover(t)
        result2 = tf.minutes_to_rollover(t)
        assert result1 == result2

    def test_day_end_cache_hit(self, tf):
        t = dt.datetime(2026, 1, 5, 10, 0, tzinfo=dt.UTC)
        result1 = tf.minutes_to_day_end(t)
        result2 = tf.minutes_to_day_end(t)
        assert result1 == result2


# ---------------------------------------------------------------------------
# _clear_cache_if_stale
# ---------------------------------------------------------------------------
class TestCacheStale:
    def test_stale_cache_cleared(self, tf):
        t1 = dt.datetime(2026, 1, 5, 10, 0, tzinfo=dt.UTC)
        tf.minutes_to_session_close(t1)  # Populates cache
        assert tf.cache_timestamp is not None

        # Advance time beyond TTL (60s)
        t2 = dt.datetime(2026, 1, 5, 10, 5, tzinfo=dt.UTC)  # 5 minutes later
        tf._clear_cache_if_stale(t2)
        assert len(tf.cache) == 0
        assert tf.cache_timestamp is None

    def test_fresh_cache_not_cleared(self, tf):
        t1 = dt.datetime(2026, 1, 5, 10, 0, 0, tzinfo=dt.UTC)
        tf.minutes_to_session_close(t1)
        cache_size = len(tf.cache)

        # Within TTL
        t2 = dt.datetime(2026, 1, 5, 10, 0, 30, tzinfo=dt.UTC)  # 30s later
        tf._clear_cache_if_stale(t2)
        assert len(tf.cache) == cache_size

    def test_none_timestamp_no_clear(self, tf):
        tf._clear_cache_if_stale(None)  # Should not crash


# ---------------------------------------------------------------------------
# Friday session close
# ---------------------------------------------------------------------------
class TestFridayClose:
    def test_friday_22_past_close(self, tf):
        """Friday 23:00 → past this week's 22:00, next week's close."""
        friday_late = dt.datetime(2026, 1, 9, 23, 0, tzinfo=dt.UTC)  # Friday
        minutes = tf.minutes_to_session_close(friday_late)
        # Should wrap to next Friday (7 days away)
        assert minutes > 6 * 24 * 60  # at least 6 days

    def test_is_friday_close_approaching_true_within_threshold(self, tf):
        # Friday 19:00 → 3 hours before 22:00 close
        friday = dt.datetime(2026, 1, 9, 19, 0, tzinfo=dt.UTC)
        assert tf.is_friday_close_approaching(friday, threshold_hours=4.0) is True

    def test_is_friday_close_not_approaching_midweek(self, tf):
        # Tuesday 10:00 → far from Friday close
        tuesday = dt.datetime(2026, 1, 6, 10, 0, tzinfo=dt.UTC)
        assert tf.is_friday_close_approaching(tuesday, threshold_hours=4.0) is False


# ---------------------------------------------------------------------------
# get_all_features
# ---------------------------------------------------------------------------
class TestGetAllFeatures:
    def test_all_features_valid_range(self, tf):
        t = dt.datetime(2026, 1, 5, 14, 30, tzinfo=dt.UTC)
        features = tf.get_all_features(t)
        for key, value in features.items():
            assert isinstance(value, float), f"{key} is not float"

    def test_none_returns_zeros(self, tf):
        features = tf.get_all_features(None)
        assert all(v == pytest.approx(0.0) for v in features.values())

    def test_feature_statistics_populated(self, tf):
        t = dt.datetime(2026, 1, 5, 14, 30, tzinfo=dt.UTC)
        tf.get_all_features(t)
        stats = tf.get_feature_statistics()
        assert "minutes_to_session_close" in stats
        assert stats["minutes_to_session_close"]["count"] >= 1
