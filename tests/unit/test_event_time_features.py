"""Tests for src.features.event_time_features – EventTimeFeatureEngine."""

from datetime import UTC, datetime

import pytest

from src.features.event_time_features import EventTimeFeatureEngine, SessionTimes


# ---------------------------------------------------------------------------
# SessionTimes dataclass
# ---------------------------------------------------------------------------

class TestSessionTimes:
    def test_get_open_minutes(self):
        s = SessionTimes("Test", 7, 30, 16, 0)
        assert s.get_open_minutes() == 7 * 60 + 30

    def test_get_close_minutes(self):
        s = SessionTimes("Test", 7, 30, 16, 45)
        assert s.get_close_minutes() == 16 * 60 + 45


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    return EventTimeFeatureEngine()


# ---------------------------------------------------------------------------
# calculate_features()
# ---------------------------------------------------------------------------

class TestCalculateFeatures:

    def test_returns_dict(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 12, 0, tzinfo=UTC))
        assert isinstance(f, dict)
        assert len(f) > 20  # Many features expected

    def test_session_proximity_keys(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 12, 0, tzinfo=UTC))
        for session in ("london", "new_york", "tokyo", "sydney"):
            assert f"{session}_mins_to_open" in f
            assert f"{session}_mins_to_close" in f
            assert f"{session}_mins_from_open" in f
            assert f"{session}_mins_from_close" in f
            assert f"{session}_is_active" in f

    def test_overlap_keys(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 14, 0, tzinfo=UTC))
        assert "london_ny_overlap" in f
        assert "tokyo_london_overlap" in f
        assert "sydney_tokyo_overlap" in f

    def test_week_month_keys(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 12, 0, tzinfo=UTC))
        assert "week_progress" in f
        assert "month_progress" in f
        assert "day_of_week" in f
        assert "hour_of_day" in f
        assert "is_weekend" in f

    def test_defaults_to_now(self, engine):
        f = engine.calculate_features()  # No argument
        assert isinstance(f, dict)
        assert len(f) > 0

    def test_naive_datetime_treated_as_utc(self, engine):
        dt_naive = datetime(2026, 1, 14, 12, 0)
        f = engine.calculate_features(dt_naive)
        assert len(f) > 0  # Should not crash

    def test_caching(self, engine):
        dt1 = datetime(2026, 1, 14, 12, 0, tzinfo=UTC)
        f1 = engine.calculate_features(dt1)
        f2 = engine.calculate_features(dt1)
        assert f1 == f2

    def test_cache_pruning(self, engine):
        """Add many distinct minutes to trigger cache pruning."""
        for m in range(50):
            engine.calculate_features(datetime(2026, 1, 14, 0, m % 60, tzinfo=UTC))
        # Should not raise or crash


# ---------------------------------------------------------------------------
# Session activity
# ---------------------------------------------------------------------------

class TestSessionActivity:

    def test_london_active_at_10_utc(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 10, 0, tzinfo=UTC))
        assert f["london_is_active"] == pytest.approx(1.0)

    def test_london_inactive_at_5_utc(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 5, 0, tzinfo=UTC))
        assert f["london_is_active"] == pytest.approx(0.0)

    def test_ny_active_at_15_utc(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 15, 0, tzinfo=UTC))
        assert f["new_york_is_active"] == pytest.approx(1.0)

    def test_london_ny_overlap(self, engine):
        """At 14:00 UTC both London and NY should be active."""
        f = engine.calculate_features(datetime(2026, 1, 14, 14, 0, tzinfo=UTC))
        assert f["london_ny_overlap"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Session overlaps (overnight sessions)
# ---------------------------------------------------------------------------

class TestOvernightSessions:
    def test_sydney_active_at_23_utc(self, engine):
        """Sydney 21:00-06:00 → active at 23:00."""
        f = engine.calculate_features(datetime(2026, 1, 14, 23, 0, tzinfo=UTC))
        assert f["sydney_is_active"] == pytest.approx(1.0)

    def test_sydney_active_at_3_utc(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 3, 0, tzinfo=UTC))
        assert f["sydney_is_active"] == pytest.approx(1.0)

    def test_sydney_inactive_at_10_utc(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 10, 0, tzinfo=UTC))
        assert f["sydney_is_active"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Week / month progress
# ---------------------------------------------------------------------------

class TestProgressFeatures:
    def test_week_progress_range(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 12, 0, tzinfo=UTC))
        assert 0.0 <= f["week_progress"] <= 1.0

    def test_month_progress_range(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 12, 0, tzinfo=UTC))
        assert 0.0 <= f["month_progress"] <= 1.0

    def test_weekend_flag(self, engine):
        # Saturday
        sat = datetime(2026, 1, 17, 12, 0, tzinfo=UTC)
        f = engine.calculate_features(sat)
        assert f["is_weekend"] == pytest.approx(1.0)

        # Wednesday
        wed = datetime(2026, 1, 14, 12, 0, tzinfo=UTC)
        f2 = engine.calculate_features(wed)
        assert f2["is_weekend"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_active_sessions / get_next_major_event / is_high_liquidity_period
# ---------------------------------------------------------------------------

class TestHelperMethods:

    def test_get_active_sessions_during_london(self, engine):
        active = engine.get_active_sessions(datetime(2026, 1, 14, 10, 0, tzinfo=UTC))
        assert "LONDON" in active

    def test_get_active_sessions_defaults_to_now(self, engine):
        active = engine.get_active_sessions()
        assert isinstance(active, list)

    def test_get_next_major_event(self, engine):
        name, mins = engine.get_next_major_event(datetime(2026, 1, 14, 10, 0, tzinfo=UTC))
        assert isinstance(name, str)
        assert mins >= 0

    def test_is_high_liquidity_during_overlap(self, engine):
        dt_overlap = datetime(2026, 1, 14, 14, 0, tzinfo=UTC)
        assert engine.is_high_liquidity_period(dt_overlap) is True

    def test_is_high_liquidity_defaults_to_now(self, engine):
        # Just ensure it runs without error
        result = engine.is_high_liquidity_period()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Rollover features
# ---------------------------------------------------------------------------

class TestRollover:

    def test_rollover_proximity_keys(self, engine):
        f = engine.calculate_features(datetime(2026, 1, 14, 21, 0, tzinfo=UTC))
        assert "mins_to_rollover" in f
        assert "mins_from_rollover" in f

    def test_close_to_rollover(self, engine):
        """At 21:30, rollover at 22:00 → mins_to_rollover should be small positive."""
        f = engine.calculate_features(datetime(2026, 1, 14, 21, 30, tzinfo=UTC))
        # 30 minutes to rollover normalized by 720 ≈ 0.042
        assert f["mins_to_rollover"] > 0
        assert f["mins_to_rollover"] < 0.1
