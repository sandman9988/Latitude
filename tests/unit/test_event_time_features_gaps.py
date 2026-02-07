"""
Tests for event_time_features gap coverage.

Covers:
- get_nearest_session_event "no events" path (line 299)
- is_high_liquidity_period final return (line 320)
"""

from datetime import datetime, UTC
from unittest.mock import patch

import pytest

from src.features.event_time_features import EventTimeFeatureEngine


class TestGetNearestSessionEvent:
    """Test get_nearest_session_event, including the no-events edge case."""

    @pytest.fixture()
    def engine(self):
        return EventTimeFeatureEngine()

    def test_returns_nearest_event(self, engine):
        """Normal case: returns the nearest upcoming event."""
        dt = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)  # Monday noon
        name, minutes = engine.get_next_major_event(dt)
        assert isinstance(name, str)
        assert isinstance(minutes, int)

    def test_no_events_returns_none(self, engine):
        """When _calc_session_proximity returns <= 0 for all, return ('None', 0)."""
        dt = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)

        # Mock _calc_session_proximity to always return (0, x) or negative
        with patch.object(engine, "_calc_session_proximity", return_value=(0, 100)):
            name, minutes = engine.get_next_major_event(dt)
            assert name == "None"
            assert minutes == 0

    def test_no_events_negative_proximity(self, engine):
        """When all proximities are negative, return ('None', 0)."""
        dt = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)

        with patch.object(engine, "_calc_session_proximity", return_value=(-10, 10)):
            name, minutes = engine.get_next_major_event(dt)
            assert name == "None"
            assert minutes == 0


class TestIsHighLiquidityPeriod:
    """Test is_high_liquidity_period, including the NY-only and false paths."""

    @pytest.fixture()
    def engine(self):
        return EventTimeFeatureEngine()

    def test_high_liquidity_during_london_ny_overlap(self, engine):
        """London-NY overlap (14:00 UTC) is high liquidity."""
        dt = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)  # Monday 14:00
        assert engine.is_high_liquidity_period(dt) is True

    def test_high_liquidity_during_london_only(self, engine):
        """London session (09:00 UTC) without overlap is high liquidity."""
        dt = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
        assert engine.is_high_liquidity_period(dt) is True

    def test_high_liquidity_during_ny_only(self, engine):
        """NY session without London (17:00 UTC) should reach the final return."""
        dt = datetime(2026, 1, 5, 17, 0, tzinfo=UTC)  # Monday 17:00
        # At 17:00: London closed (07-16), NY active (13-22)
        # No overlap, no London → falls through to NY check (line 320)
        result = engine.is_high_liquidity_period(dt)
        assert result is True

    def test_not_high_liquidity_early_morning(self, engine):
        """At 04:00 UTC (only Sydney/Tokyo), not high liquidity."""
        dt = datetime(2026, 1, 5, 4, 0, tzinfo=UTC)
        result = engine.is_high_liquidity_period(dt)
        # At 04:00: only Sydney (21-06) and Tokyo (23-08) active
        # No London, no NY, no major overlaps → False via line 320
        assert result is False

    def test_no_args_uses_current_time(self, engine):
        """Calling without args uses current time."""
        result = engine.is_high_liquidity_period()
        assert isinstance(result, bool)

    def test_tokyo_london_overlap_is_high(self, engine):
        """Tokyo-London overlap (07:30 UTC) is high liquidity."""
        dt = datetime(2026, 1, 5, 7, 30, tzinfo=UTC)  # Monday 07:30
        # Tokyo: 23-08, London: 07-16 → overlap at 07:30
        result = engine.is_high_liquidity_period(dt)
        assert result is True
