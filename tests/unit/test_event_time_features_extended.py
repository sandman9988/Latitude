"""Extended tests for src.features.event_time_features.

Covers: cache pruning, week progress before Sunday 21:00, get_next_major_event
edge cases, is_high_liquidity false path, session proximity wrap-around.
"""

from datetime import UTC, datetime

import pytest

from src.features.event_time_features import EventTimeFeatureEngine


# ---------------------------------------------------------------------------
# Cache pruning
# ---------------------------------------------------------------------------
class TestCachePruning:
    def test_cache_pruned_over_1000(self):
        engine = EventTimeFeatureEngine()
        # Fill cache with >1000 entries by using different minutes
        base = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
        for day in range(4):
            for hour in range(24):
                for minute in range(0, 60, 4):  # 15 per hour × 24 × 4 = 1440
                    dt = base.replace(day=5 + day, hour=hour, minute=minute)
                    engine.calculate_features(dt)
        # After 1440 unique entries, pruning should have triggered
        assert len(engine.cache) <= 1001


# ---------------------------------------------------------------------------
# Week progress edge cases
# ---------------------------------------------------------------------------
class TestWeekProgressEdge:
    def test_before_sunday_2100(self):
        """Sunday 20:00 is before FX week start → end of previous week."""
        engine = EventTimeFeatureEngine()
        # Sunday 20:00 UTC is before the 21:00 FX open
        dt = datetime(2026, 1, 4, 20, 0, tzinfo=UTC)  # A Sunday
        features = engine.calculate_features(dt)
        # Should be near end of week (close to 1.0)
        assert features["week_progress"] > 0.9

    def test_sunday_2100_is_week_start(self):
        """Sunday 21:00 is the start of FX week → progress ≈ 0."""
        engine = EventTimeFeatureEngine()
        dt = datetime(2026, 1, 4, 21, 0, tzinfo=UTC)
        features = engine.calculate_features(dt)
        assert features["week_progress"] < 0.01

    def test_midweek_progress(self):
        engine = EventTimeFeatureEngine()
        # Wednesday 12:00 — roughly 2.625 days into 5-day week
        dt = datetime(2026, 1, 7, 12, 0, tzinfo=UTC)
        features = engine.calculate_features(dt)
        assert 0.4 < features["week_progress"] < 0.6


# ---------------------------------------------------------------------------
# Month progress
# ---------------------------------------------------------------------------
class TestMonthProgressEdge:
    def test_first_day_near_zero(self):
        engine = EventTimeFeatureEngine()
        dt = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        features = engine.calculate_features(dt)
        assert features["month_progress"] < 0.05

    def test_last_day_near_one(self):
        engine = EventTimeFeatureEngine()
        dt = datetime(2026, 1, 31, 23, 0, tzinfo=UTC)
        features = engine.calculate_features(dt)
        assert features["month_progress"] > 0.95


# ---------------------------------------------------------------------------
# get_next_major_event
# ---------------------------------------------------------------------------
class TestGetNextMajorEvent:
    def test_returns_tuple_with_name_and_minutes(self):
        engine = EventTimeFeatureEngine()
        dt = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)  # Monday 10 UTC
        name, mins = engine.get_next_major_event(dt)
        assert isinstance(name, str)
        assert mins > 0

    def test_defaults_to_now(self):
        engine = EventTimeFeatureEngine()
        name, mins = engine.get_next_major_event()
        assert isinstance(name, str)

    def test_nearest_event_is_closest(self):
        engine = EventTimeFeatureEngine()
        # At 06:50 UTC, London opens at 07:00 (10 mins away)
        dt = datetime(2026, 1, 5, 6, 50, tzinfo=UTC)
        name, mins = engine.get_next_major_event(dt)
        assert mins <= 15  # Should be very close to London open


# ---------------------------------------------------------------------------
# is_high_liquidity_period
# ---------------------------------------------------------------------------
class TestHighLiquidity:
    def test_not_high_liquidity_outside_sessions(self):
        engine = EventTimeFeatureEngine()
        # 04:00 UTC: Sydney active, Tokyo active → but NOT London or NY
        # Sydney: 21-06, Tokyo: 23-08
        # Check a time when only Sydney/Tokyo are active (no London/NY overlap)
        # Actually Sydney-Tokyo overlap IS tracked. But is_high_liquidity
        # only checks london_ny, tokyo_london, london, and ny — not sydney_tokyo
        dt = datetime(2026, 1, 5, 4, 0, tzinfo=UTC)
        result = engine.is_high_liquidity_period(dt)
        # At 04:00: Sydney active, Tokyo active; London inactive, NY inactive
        # No london_ny overlap, no tokyo_london overlap, no london, no ny
        assert result is False

    def test_high_liquidity_during_london(self):
        engine = EventTimeFeatureEngine()
        dt = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
        assert engine.is_high_liquidity_period(dt) is True

    def test_high_liquidity_during_ny(self):
        engine = EventTimeFeatureEngine()
        dt = datetime(2026, 1, 5, 17, 0, tzinfo=UTC)  # NY active, London closed
        assert engine.is_high_liquidity_period(dt) is True


# ---------------------------------------------------------------------------
# Session proximity wrap
# ---------------------------------------------------------------------------
class TestSessionProximityWrap:
    def test_wrap_around_midnight(self):
        """Event at 23:00, current time 01:00 → event was 2 hours ago."""
        engine = EventTimeFeatureEngine()
        dt = datetime(2026, 1, 5, 1, 0, tzinfo=UTC)
        mins_to, mins_from = engine._calc_session_proximity(dt, 23 * 60)
        # 23:00 was 2 hours ago
        assert mins_from > 0
        assert mins_to < 0

    def test_normalize_minutes_clamps(self):
        engine = EventTimeFeatureEngine()
        assert engine._normalize_minutes(9999) == pytest.approx(1.0)
        assert engine._normalize_minutes(-9999) == pytest.approx(-1.0)
