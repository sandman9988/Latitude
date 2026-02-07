"""Gap tests for src.utils.non_repaint_guards.

Targets:
- Line 194: get_series break when bar closed but data empty (offset=0, bars_back=0)
- NonRepaintIndicator class paths (update/get_current/get_previous completeness)
"""

import pytest

from src.utils.non_repaint_guards import (
    NonRepaintBarAccess,
    NonRepaintIndicator,
)


class TestGetSeriesClosedButEmpty:
    """Hit line 194 — get_series with offset=0, bar closed, but data is empty."""

    def test_get_series_offset_0_closed_empty_returns_empty(self):
        """When bar is closed but data is empty, get_series should return []."""
        series = NonRepaintBarAccess("close")
        series.is_bar_closed = True  # Force closed state without data
        result = series.get_series(3, offset=0)
        assert result == []

    def test_get_series_offset_0_closed_then_historical(self):
        """offset=0 with only 1 bar — returns [bar0], no historical available."""
        series = NonRepaintBarAccess("close")
        series.append(42.0)
        series.mark_bar_closed()
        result = series.get_series(5, offset=0)
        # bar[0]=42 succeeds, bar[1] needs safe_get_previous(1) which fails
        assert result == [42.0]

    def test_get_series_offset_0_multiple_bars(self):
        """offset=0 with several bars — includes current + historicals."""
        series = NonRepaintBarAccess("close")
        for val in [10.0, 20.0, 30.0]:
            series.append(val)
        series.mark_bar_closed()
        # offset=0: bar[0]=30, bar[1]=20, bar[2]=10
        result = series.get_series(3, offset=0)
        assert result == [30.0, 20.0, 10.0]


class TestNonRepaintIndicatorEdgeCases:
    """Additional edge cases for NonRepaintIndicator."""

    def test_update_many_values_deque_maxlen(self):
        """Values deque has maxlen=500; test overflow doesn't crash."""
        ind = NonRepaintIndicator("test", period=5)
        for i in range(600):
            ind.update(float(i))
        assert ind.get_current() == pytest.approx(599.0)
        assert ind.is_ready is True
        # Oldest value should be evicted
        assert ind.get_previous(499) == pytest.approx(100.0)

    def test_get_previous_exact_boundary(self):
        """bars_ago == len(values) - 1 returns oldest value;
        bars_ago == len(values) returns None."""
        ind = NonRepaintIndicator("test", period=2)
        ind.update(1.0)
        ind.update(2.0)
        ind.update(3.0)
        # 3 values: [1.0, 2.0, 3.0]
        assert ind.get_previous(2) == pytest.approx(1.0)  # Oldest
        assert ind.get_previous(3) is None  # Out of range

    def test_is_ready_becomes_true_at_period(self):
        """is_ready flips to True exactly when len(values) == period."""
        ind = NonRepaintIndicator("rsi", period=3)
        ind.update(10.0)
        assert ind.is_ready is False
        ind.update(20.0)
        assert ind.is_ready is False
        ind.update(30.0)
        assert ind.is_ready is True

    def test_get_current_after_single_update(self):
        """get_current returns last even when not ready."""
        ind = NonRepaintIndicator("sma", period=10)
        ind.update(99.0)
        assert ind.get_current() == pytest.approx(99.0)
        assert ind.is_ready is False
