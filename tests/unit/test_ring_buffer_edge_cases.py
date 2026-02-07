"""
Tests for ring buffer edge cases.

Covers:
- RollingVariance with period=1 (ring_buffer.py lines 195-196)
- RollingMean edge cases
- RollingMinMax edge cases
- RollingStats combined tracker
"""

import math

import pytest

from src.utils.ring_buffer import (
    RingBuffer,
    RollingMean,
    RollingMinMax,
    RollingStats,
    RollingVariance,
)


# ===========================================================================
# RollingVariance period=1 edge case (lines 195-196)
# ===========================================================================

class TestRollingVariancePeriodOne:
    """Period=1 buffer triggers the n_after=0 reset path."""

    def test_period_one_single_update(self):
        """With period=1, first update sets mean and zero variance."""
        rv = RollingVariance(period=1)
        rv.update(5.0)
        assert rv.mean == pytest.approx(5.0)
        # With 1 sample, variance is 0 (less than min_periods=2)
        assert rv.variance == pytest.approx(0.0)
        assert rv.std == pytest.approx(0.0)

    def test_period_one_second_update_triggers_reset(self):
        """Second update with period=1 triggers the n_after=0 else branch."""
        rv = RollingVariance(period=1)
        rv.update(10.0)
        rv.update(20.0)
        # After second update, buffer only holds [20.0]
        assert rv.mean == pytest.approx(20.0)

    def test_period_one_multiple_updates(self):
        """Multiple updates with period=1 keeps only latest value."""
        rv = RollingVariance(period=1)
        for val in [1.0, 2.0, 3.0, 100.0]:
            rv.update(val)
        assert rv.mean == pytest.approx(100.0)

    def test_period_one_is_not_ready(self):
        """With period=1, variance is never 'ready' (min_periods=2)."""
        rv = RollingVariance(period=1)
        rv.update(5.0)
        # min_periods is clamped to at least 2, but period=1 buffer never has 2 elements
        assert not rv.is_ready()

    def test_period_one_repr(self):
        """repr works for period=1."""
        rv = RollingVariance(period=1)
        rv.update(3.14)
        r = repr(rv)
        assert "period=1" in r
        assert "ready=False" in r


# ===========================================================================
# RollingVariance standard edge cases
# ===========================================================================

class TestRollingVarianceEdgeCases:
    """Additional edge cases for RollingVariance."""

    def test_period_two_two_values(self):
        """With period=2, two values gives valid variance."""
        rv = RollingVariance(period=2)
        rv.update(10.0)
        rv.update(20.0)
        assert rv.is_ready()
        # variance of [10, 20] = 50.0
        assert rv.variance == pytest.approx(50.0)
        assert rv.std == pytest.approx(math.sqrt(50.0))

    def test_period_two_three_values_rolls(self):
        """Third value with period=2 replaces the oldest."""
        rv = RollingVariance(period=2)
        rv.update(10.0)
        rv.update(20.0)
        rv.update(30.0)
        # Window: [20, 30], variance = 50.0
        assert rv.mean == pytest.approx(25.0)
        assert rv.variance == pytest.approx(50.0)

    def test_constant_values_zero_variance(self):
        """Constant data has zero variance."""
        rv = RollingVariance(period=5)
        for _ in range(10):
            rv.update(42.0)
        assert rv.variance == pytest.approx(0.0, abs=1e-10)
        assert rv.std == pytest.approx(0.0, abs=1e-10)

    def test_large_values_numerical_stability(self):
        """Large values should not cause numerical issues."""
        rv = RollingVariance(period=10)
        base = 1e12
        for i in range(20):
            rv.update(base + i)
        # Variance of last 10 values [10..19] offset by base
        # Should match numpy
        import numpy as np
        expected = np.var(np.arange(10, 20, dtype=np.float64), ddof=1)
        assert rv.variance == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# RollingMean edge cases
# ===========================================================================

class TestRollingMeanEdgeCases:
    """Edge cases for RollingMean."""

    def test_period_one(self):
        """Period=1 buffer tracks only latest value."""
        rm = RollingMean(period=1)
        rm.update(5.0)
        assert rm.value == pytest.approx(5.0)
        rm.update(10.0)
        assert rm.value == pytest.approx(10.0)
        assert rm.is_ready()

    def test_no_updates_value_zero(self):
        """Before any updates, value is 0."""
        rm = RollingMean(period=5)
        assert rm.value == pytest.approx(0.0)
        assert not rm.is_ready()

    def test_repr(self):
        """repr format is correct."""
        rm = RollingMean(period=3)
        rm.update(6.0)
        r = repr(rm)
        assert "period=3" in r
        assert "ready=False" in r

    def test_accurate_after_overflow(self):
        """Mean remains accurate after buffer wraps around."""
        rm = RollingMean(period=3)
        for v in [1.0, 2.0, 3.0, 10.0, 20.0]:
            rm.update(v)
        # Window: [3, 10, 20]
        assert rm.value == pytest.approx(11.0)


# ===========================================================================
# RollingMinMax edge cases
# ===========================================================================

class TestRollingMinMaxEdgeCases:
    """Edge cases for RollingMinMax."""

    def test_period_one(self):
        """Period=1 keeps only the latest value as both min and max."""
        mm = RollingMinMax(period=1)
        mm.update(5.0)
        assert mm.min_value == pytest.approx(5.0)
        assert mm.max_value == pytest.approx(5.0)
        mm.update(10.0)
        assert mm.min_value == pytest.approx(10.0)
        assert mm.max_value == pytest.approx(10.0)

    def test_not_ready_before_update(self):
        """Before any updates, not ready."""
        mm = RollingMinMax(period=5)
        assert not mm.is_ready()

    def test_descending_then_ascending(self):
        """Both deques must maintain correct order through transitions."""
        mm = RollingMinMax(period=3)
        # Descending
        mm.update(10.0)
        mm.update(5.0)
        mm.update(1.0)
        assert mm.min_value == pytest.approx(1.0)
        assert mm.max_value == pytest.approx(10.0)

        # Now ascending - window becomes [5, 1, 100], then [1, 100, 200]
        mm.update(100.0)
        assert mm.max_value == pytest.approx(100.0)
        assert mm.min_value == pytest.approx(1.0)

        mm.update(200.0)
        assert mm.max_value == pytest.approx(200.0)
        assert mm.min_value == pytest.approx(1.0)

    def test_repr(self):
        """repr format is correct."""
        mm = RollingMinMax(period=3)
        mm.update(7.0)
        r = repr(mm)
        assert "period=3" in r


# ===========================================================================
# RollingStats combined tracker
# ===========================================================================

class TestRollingStatsEdgeCases:
    """Edge cases for the combined RollingStats tracker."""

    def test_all_properties_accessible(self):
        """All properties (mean, std, variance, min, max) work."""
        rs = RollingStats(period=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rs.update(v)

        assert rs.mean == pytest.approx(3.0)
        assert rs.min == pytest.approx(1.0)
        assert rs.max == pytest.approx(5.0)
        assert rs.variance > 0
        assert rs.std > 0

    def test_is_ready_requires_all_trackers(self):
        """is_ready needs all three sub-trackers to be ready."""
        rs = RollingStats(period=5)
        assert not rs.is_ready()
        for v in range(5):
            rs.update(float(v))
        assert rs.is_ready()

    def test_repr(self):
        """repr works."""
        rs = RollingStats(period=3)
        rs.update(1.0)
        r = repr(rs)
        assert "period=3" in r

    def test_period_one_combined(self):
        """Period=1 works for the combined tracker."""
        rs = RollingStats(period=1)
        rs.update(42.0)
        assert rs.mean == pytest.approx(42.0)
        assert rs.min == pytest.approx(42.0)
        assert rs.max == pytest.approx(42.0)
