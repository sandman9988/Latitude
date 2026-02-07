"""Extended tests for src.utils.non_repaint_guards.

Covers NonRepaintIndicator class, get_previous(), is_ready logic,
and NonRepaintBarAccess edge cases not in original tests.
"""

import pytest

from src.utils.non_repaint_guards import (
    NonRepaintBarAccess,
    NonRepaintError,
    NonRepaintIndicator,
)


# ---------------------------------------------------------------------------
# NonRepaintIndicator
# ---------------------------------------------------------------------------
class TestNonRepaintIndicator:
    def test_init(self):
        ind = NonRepaintIndicator("rsi", period=14)
        assert ind.name == "rsi"
        assert ind.period == 14
        assert ind.is_ready is False
        assert ind.get_current() is None

    def test_update_and_ready(self):
        ind = NonRepaintIndicator("ema", period=3)
        ind.update(1.0)
        assert ind.is_ready is False
        ind.update(2.0)
        assert ind.is_ready is False
        ind.update(3.0)
        assert ind.is_ready is True
        assert ind.get_current() == pytest.approx(3.0)

    def test_get_previous_valid(self):
        ind = NonRepaintIndicator("sma", period=2)
        ind.update(10.0)
        ind.update(20.0)
        ind.update(30.0)
        assert ind.get_previous(1) == pytest.approx(20.0)
        assert ind.get_previous(2) == pytest.approx(10.0)

    def test_get_previous_out_of_range(self):
        ind = NonRepaintIndicator("stoch", period=2)
        ind.update(5.0)
        assert ind.get_previous(1) is None  # Only 1 value, can't go back 1
        assert ind.get_previous(5) is None

    def test_get_previous_zero_returns_none(self):
        ind = NonRepaintIndicator("atr", period=1)
        ind.update(42.0)
        assert ind.get_previous(0) is None  # bars_ago < 1

    def test_get_current_empty(self):
        ind = NonRepaintIndicator("vol", period=5)
        assert ind.get_current() is None

    def test_max_lookback(self):
        ind = NonRepaintIndicator("test", period=2)
        for i in range(600):
            ind.update(float(i))
        assert len(ind.values) == 500  # default maxlen


# ---------------------------------------------------------------------------
# NonRepaintBarAccess: additional edge cases
# ---------------------------------------------------------------------------
class TestBarAccessExtended:
    def test_len(self):
        ba = NonRepaintBarAccess("close", max_lookback=10)
        assert len(ba) == 0
        ba.append(1.0)
        ba.append(2.0)
        assert len(ba) == 2

    def test_bar_count_tracks_appends(self):
        ba = NonRepaintBarAccess("test")
        ba.append(1.0)
        ba.append(2.0)
        ba.append(3.0)
        assert ba.bar_count == 3

    def test_get_series_with_closed_offset_0(self):
        ba = NonRepaintBarAccess("close")
        for v in [100.0, 101.0, 102.0, 103.0]:
            ba.append(v)
        ba.mark_bar_closed()
        series = ba.get_series(3, offset=0)
        assert series == [103.0, 102.0, 101.0]

    def test_get_series_partial_when_insufficient(self):
        ba = NonRepaintBarAccess("close")
        ba.append(1.0)
        ba.append(2.0)
        ba.mark_bar_closed()
        # Request 10 bars with offset=0, but only 2 available
        series = ba.get_series(10, offset=0)
        assert len(series) <= 2

    def test_get_series_offset_1_safe(self):
        ba = NonRepaintBarAccess("close")
        for v in [10.0, 20.0, 30.0, 40.0]:
            ba.append(v)
        # Bar not closed, but offset=1 is safe
        series = ba.get_series(2, offset=1)
        assert series == [30.0, 20.0]

    def test_get_series_large_offset(self):
        ba = NonRepaintBarAccess("close")
        ba.append(1.0)
        ba.append(2.0)
        # offset=5 > len(data) → empty list
        series = ba.get_series(3, offset=5)
        assert series == []

    def test_safe_get_previous_exact_boundary(self):
        ba = NonRepaintBarAccess("close")
        ba.append(1.0)
        ba.append(2.0)
        ba.append(3.0)
        # bars_ago=2, data[-3] = 1.0
        assert ba.safe_get_previous(2) == pytest.approx(1.0)
        # bars_ago=3 → >= len(data)=3 → None
        assert ba.safe_get_previous(3) is None

    def test_multiple_close_open_cycles(self):
        ba = NonRepaintBarAccess("close")
        ba.append(1.0)
        ba.mark_bar_closed()
        assert ba.get_current() == pytest.approx(1.0)
        ba.mark_bar_opened()
        ba.append(2.0)
        ba.mark_bar_closed()
        assert ba.get_current() == pytest.approx(2.0)
        ba.mark_bar_opened()
        with pytest.raises(NonRepaintError):
            ba.get_current()

    def test_repr_content(self):
        ba = NonRepaintBarAccess("rsi", max_lookback=50)
        ba.append(42.0)
        r = repr(ba)
        assert "rsi" in r
        assert "bars=1" in r
