"""Extended tests for src.utils.ring_buffer.

Covers: RollingStats combined tracker, RollingMinMax expiration,
RollingVariance edge cases, iter/getitem.
"""

import math
import pytest
import numpy as np

from src.utils.ring_buffer import (
    RingBuffer,
    RollingMean,
    RollingMinMax,
    RollingStats,
    RollingVariance,
)


# ---------------------------------------------------------------------------
# RingBuffer extras
# ---------------------------------------------------------------------------
class TestRingBufferExtended:
    def test_iter(self):
        rb = RingBuffer(3)
        rb.append(1.0)
        rb.append(2.0)
        assert list(rb) == [1.0, 2.0]

    def test_iter_after_wrap(self):
        rb = RingBuffer(2)
        for v in [1.0, 2.0, 3.0]:
            rb.append(v)
        assert list(rb) == [2.0, 3.0]

    def test_getitem_negative(self):
        rb = RingBuffer(3)
        rb.append(10.0)
        rb.append(20.0)
        assert rb[-1] == pytest.approx(20.0)

    def test_capacity_one(self):
        rb = RingBuffer(1)
        rb.append(5.0)
        assert rb.is_full()
        rb.append(10.0)
        assert len(rb) == 1
        assert rb[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# RollingMean extras
# ---------------------------------------------------------------------------
class TestRollingMeanExtended:
    def test_empty_value_is_zero(self):
        rm = RollingMean(5)
        assert rm.value == pytest.approx(0.0)

    def test_not_ready_until_full(self):
        rm = RollingMean(3)
        rm.update(1.0)
        rm.update(2.0)
        assert not rm.is_ready()
        rm.update(3.0)
        assert rm.is_ready()

    def test_large_period_accurate(self):
        rm = RollingMean(100)
        vals = list(range(200))
        for v in vals:
            rm.update(float(v))
        expected = np.mean(vals[-100:])
        assert abs(rm.value - expected) < 1e-6


# ---------------------------------------------------------------------------
# RollingVariance extras
# ---------------------------------------------------------------------------
class TestRollingVarianceExtended:
    def test_single_value_zero_variance(self):
        rv = RollingVariance(5)
        rv.update(42.0)
        assert rv.variance == pytest.approx(0.0)
        assert rv.std == pytest.approx(0.0)

    def test_two_values_valid_variance(self):
        rv = RollingVariance(5, min_periods=2)
        rv.update(10.0)
        rv.update(20.0)
        assert rv.is_ready()
        assert rv.variance > 0

    def test_after_overflow_matches_numpy(self):
        period = 5
        rv = RollingVariance(period)
        vals = [3.0, 7.0, 1.0, 9.0, 5.0, 2.0, 8.0, 4.0]
        for v in vals:
            rv.update(v)
        window = vals[-period:]
        assert abs(rv.mean - np.mean(window)) < 1e-6
        assert abs(rv.std - np.std(window, ddof=1)) < 1e-6

    def test_min_periods_clamped_to_two(self):
        rv = RollingVariance(10, min_periods=0)
        assert rv.min_periods == 2

    def test_negative_variance_prevented(self):
        """M2 should never go significantly negative even with many updates."""
        rv = RollingVariance(3)
        for v in [1e10 + i * 0.1 for i in range(100)]:
            rv.update(v)
        assert rv.variance >= 0 or abs(rv.variance) < 1e-6


# ---------------------------------------------------------------------------
# RollingMinMax extras
# ---------------------------------------------------------------------------
class TestRollingMinMaxExtended:
    def test_decreasing_then_increasing(self):
        mm = RollingMinMax(3)
        for v in [5.0, 4.0, 3.0]:
            mm.update(v)
        assert mm.min_value == pytest.approx(3.0)
        assert mm.max_value == pytest.approx(5.0)
        # Now increase: oldest (5) expires
        mm.update(10.0)
        assert mm.max_value == pytest.approx(10.0)
        assert mm.min_value == pytest.approx(3.0)

    def test_all_same_value(self):
        mm = RollingMinMax(5)
        for _ in range(10):
            mm.update(42.0)
        assert mm.min_value == pytest.approx(42.0)
        assert mm.max_value == pytest.approx(42.0)

    def test_window_expiry(self):
        mm = RollingMinMax(3)
        mm.update(100.0)
        mm.update(1.0)
        mm.update(50.0)
        assert mm.min_value == pytest.approx(1.0)
        # Push old min out
        mm.update(60.0)
        mm.update(70.0)
        # Window: [50, 60, 70]
        assert mm.min_value == pytest.approx(50.0)
        assert mm.max_value == pytest.approx(70.0)

    def test_not_ready_when_empty(self):
        mm = RollingMinMax(5)
        assert not mm.is_ready()

    def test_repr_format(self):
        mm = RollingMinMax(3)
        mm.update(1.0)
        r = repr(mm)
        assert "RollingMinMax" in r
        assert "period=3" in r


# ---------------------------------------------------------------------------
# RollingStats combined
# ---------------------------------------------------------------------------
class TestRollingStatsExtended:
    def test_update_propagates(self):
        rs = RollingStats(5)
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            rs.update(v)
        assert rs.mean == pytest.approx(30.0)
        assert rs.min == pytest.approx(10.0)
        assert rs.max == pytest.approx(50.0)
        assert rs.std > 0

    def test_variance_property(self):
        rs = RollingStats(5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rs.update(v)
        assert rs.variance > 0
        assert abs(rs.std - math.sqrt(rs.variance)) < 1e-9

    def test_is_ready_requires_all(self):
        rs = RollingStats(5)
        rs.update(1.0)
        assert not rs.is_ready()
        for v in [2.0, 3.0, 4.0, 5.0]:
            rs.update(v)
        assert rs.is_ready()

    def test_repr(self):
        rs = RollingStats(3)
        rs.update(1.0)
        r = repr(rs)
        assert "RollingStats" in r
        assert "period=3" in r

    def test_sliding_window_correctness(self):
        rs = RollingStats(3)
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        for v in data:
            rs.update(v)
        # Window is [30, 40, 50]
        assert rs.mean == pytest.approx(40.0)
        assert rs.min == pytest.approx(30.0)
        assert rs.max == pytest.approx(50.0)
