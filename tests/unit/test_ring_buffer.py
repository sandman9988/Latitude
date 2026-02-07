"""Tests for src.utils.ring_buffer – RingBuffer, RollingMean, RollingVariance, RollingMinMax, RollingStats."""

import math

import numpy as np
import pytest

from src.utils.ring_buffer import (
    RingBuffer,
    RollingMean,
    RollingMinMax,
    RollingStats,
    RollingVariance,
)


# ---------------------------------------------------------------------------
# RingBuffer
# ---------------------------------------------------------------------------

class TestRingBuffer:
    def test_init_capacity(self):
        rb = RingBuffer(5)
        assert rb.capacity == 5
        assert len(rb) == 0

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            RingBuffer(0)

    def test_append_and_len(self):
        rb = RingBuffer(3)
        rb.append(1.0)
        rb.append(2.0)
        assert len(rb) == 2

    def test_overflow_discards_oldest(self):
        rb = RingBuffer(3)
        for v in [1, 2, 3, 4, 5]:
            rb.append(v)
        assert len(rb) == 3
        assert list(rb) == [3, 4, 5]

    def test_is_full(self):
        rb = RingBuffer(2)
        assert not rb.is_full()
        rb.append(1)
        assert not rb.is_full()
        rb.append(2)
        assert rb.is_full()

    def test_count_tracks_total(self):
        rb = RingBuffer(2)
        for i in range(10):
            rb.append(i)
        assert rb.count == 10
        assert len(rb) == 2

    def test_getitem(self):
        rb = RingBuffer(5)
        for v in [10, 20, 30]:
            rb.append(v)
        assert rb[0] == 10
        assert rb[-1] == 30


# ---------------------------------------------------------------------------
# RollingMean
# ---------------------------------------------------------------------------

class TestRollingMean:
    def test_single_value(self):
        rm = RollingMean(5)
        rm.update(10.0)
        assert rm.value == pytest.approx(10.0)

    def test_full_window(self):
        rm = RollingMean(3)
        for v in [10, 20, 30]:
            rm.update(v)
        assert rm.value == pytest.approx(20.0)

    def test_sliding_window(self):
        rm = RollingMean(3)
        for v in [10, 20, 30, 40]:
            rm.update(v)
        # Window: [20, 30, 40]
        assert rm.value == pytest.approx(30.0)

    def test_is_ready(self):
        rm = RollingMean(3)
        assert not rm.is_ready()
        rm.update(1)
        rm.update(2)
        assert not rm.is_ready()
        rm.update(3)
        assert rm.is_ready()

    def test_repr(self):
        rm = RollingMean(5)
        assert "RollingMean" in repr(rm)

    def test_matches_numpy(self):
        rm = RollingMean(5)
        data = [3.0, 7.0, 1.5, 9.2, 4.8, 6.1]
        for v in data:
            rm.update(v)
        # Window: last 5 = [7.0, 1.5, 9.2, 4.8, 6.1]
        expected = np.mean([7.0, 1.5, 9.2, 4.8, 6.1])
        assert rm.value == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# RollingVariance
# ---------------------------------------------------------------------------

class TestRollingVariance:
    def test_constant_values_zero_variance(self):
        rv = RollingVariance(5)
        for _ in range(5):
            rv.update(10.0)
        assert rv.variance == pytest.approx(0.0)
        assert rv.std == pytest.approx(0.0)

    def test_varying_values(self):
        rv = RollingVariance(5)
        for v in [10, 20, 30, 40, 50]:
            rv.update(v)
        expected_std = np.std([10, 20, 30, 40, 50], ddof=1)
        assert rv.std == pytest.approx(expected_std, abs=1e-4)

    def test_not_ready_before_min_periods(self):
        rv = RollingVariance(10, min_periods=5)
        for v in [1, 2, 3]:
            rv.update(v)
        assert not rv.is_ready()
        assert rv.variance == pytest.approx(0.0)

    def test_ready_after_min_periods(self):
        rv = RollingVariance(10, min_periods=3)
        for v in [1, 2, 3]:
            rv.update(v)
        assert rv.is_ready()
        assert rv.variance > 0

    def test_sliding_window_matches_numpy(self):
        rv = RollingVariance(5)
        data = [3, 7, 1, 9, 4, 6, 2, 8]
        for v in data:
            rv.update(v)
        window = data[-5:]
        expected_std = np.std(window, ddof=1)
        assert rv.std == pytest.approx(expected_std, abs=0.1)

    def test_numerical_stability_large_values(self):
        rv = RollingVariance(10)
        base = 1e10
        for i in range(20):
            rv.update(base + i * 0.1)
        # std should be small despite huge mean
        assert rv.std < 1.0

    def test_repr(self):
        rv = RollingVariance(5)
        assert "RollingVariance" in repr(rv)


# ---------------------------------------------------------------------------
# RollingMinMax
# ---------------------------------------------------------------------------

class TestRollingMinMax:
    def test_single_value(self):
        mm = RollingMinMax(5)
        mm.update(10.0)
        assert mm.min_value == pytest.approx(10.0)
        assert mm.max_value == pytest.approx(10.0)

    def test_increasing_sequence(self):
        mm = RollingMinMax(5)
        for v in [1, 2, 3, 4, 5]:
            mm.update(v)
        assert mm.min_value == pytest.approx(1.0)
        assert mm.max_value == pytest.approx(5.0)

    def test_window_slides(self):
        mm = RollingMinMax(3)
        for v in [10, 20, 30, 40, 50]:
            mm.update(v)
        # Window: [30, 40, 50]
        assert mm.min_value == pytest.approx(30.0)
        assert mm.max_value == pytest.approx(50.0)

    def test_min_drops_from_window(self):
        mm = RollingMinMax(3)
        for v in [1, 5, 3, 7]:
            mm.update(v)
        # Window: [5, 3, 7]
        assert mm.min_value == pytest.approx(3.0)
        assert mm.max_value == pytest.approx(7.0)

    def test_is_ready(self):
        mm = RollingMinMax(5)
        assert not mm.is_ready()
        mm.update(1)
        assert mm.is_ready()

    def test_repr(self):
        mm = RollingMinMax(5)
        mm.update(1)
        assert "RollingMinMax" in repr(mm)


# ---------------------------------------------------------------------------
# RollingStats (composite)
# ---------------------------------------------------------------------------

class TestRollingStats:
    def test_all_properties(self):
        rs = RollingStats(5)
        data = [10, 20, 30, 40, 50]
        for v in data:
            rs.update(v)
        assert rs.mean == pytest.approx(30.0)
        assert rs.min == pytest.approx(10.0)
        assert rs.max == pytest.approx(50.0)
        assert rs.std > 0

    def test_is_ready(self):
        rs = RollingStats(3)
        rs.update(1)
        assert not rs.is_ready()
        rs.update(2)
        rs.update(3)
        assert rs.is_ready()

    def test_repr(self):
        rs = RollingStats(5)
        rs.update(1)
        assert "RollingStats" in repr(rs)
