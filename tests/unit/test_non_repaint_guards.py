"""Tests for NonRepaintBarAccess and NonRepaintIndicator."""

import pytest

from src.utils.non_repaint_guards import (
    NonRepaintBarAccess,
    NonRepaintError,
    NonRepaintIndicator,
)


class TestNonRepaintBarAccess:
    def test_init(self):
        series = NonRepaintBarAccess("close")
        assert series.name == "close"
        assert len(series) == 0
        assert series.is_bar_closed is False

    def test_append(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        assert len(series) == 1
        assert series.bar_count == 1

    def test_current_blocked_before_close(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        with pytest.raises(NonRepaintError):
            series.get_current()

    def test_current_allowed_after_close(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        series.mark_bar_closed()
        assert series.get_current() == pytest.approx(100.0)

    def test_current_with_allow_incomplete(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        assert series.get_current(allow_incomplete=True) == pytest.approx(100.0)

    def test_current_raises_when_empty(self):
        series = NonRepaintBarAccess("close")
        with pytest.raises(IndexError):
            series.get_current()

    def test_mark_bar_opened_resets_flag(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        series.mark_bar_closed()
        series.mark_bar_opened()
        with pytest.raises(NonRepaintError):
            series.get_current()

    def test_safe_get_previous(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        series.append(101.0)
        series.append(102.0)
        # bar[1] = 101.0 (previous bar), bar[2] = 100.0
        assert series.safe_get_previous(1) == pytest.approx(101.0)
        assert series.safe_get_previous(2) == pytest.approx(100.0)

    def test_safe_get_previous_returns_none_for_insufficient_data(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        assert series.safe_get_previous(5) is None

    def test_safe_get_previous_rejects_zero(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        with pytest.raises(ValueError):
            series.safe_get_previous(0)

    def test_get_series_with_closed_bar(self):
        series = NonRepaintBarAccess("close")
        for i in range(5):
            series.append(100.0 + i)
        series.mark_bar_closed()
        # offset=0 includes bar[0]=104, bar[1]=103, bar[2]=102
        result = series.get_series(3, offset=0)
        assert result == [104.0, 103.0, 102.0]

    def test_get_series_offset_1(self):
        series = NonRepaintBarAccess("close")
        for i in range(5):
            series.append(100.0 + i)
        # offset=1 skips bar[0]: bar[1]=103, bar[2]=102
        result = series.get_series(2, offset=1)
        assert result == [103.0, 102.0]

    def test_get_series_offset_0_blocked_when_open(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        with pytest.raises(NonRepaintError):
            series.get_series(1, offset=0)

    def test_get_series_negative_offset_raises(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        with pytest.raises(ValueError):
            series.get_series(1, offset=-1)

    def test_get_series_shorter_than_requested(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        series.append(101.0)
        # Only 1 historical bar available with offset=1
        result = series.get_series(5, offset=1)
        assert len(result) == 1
        assert result[0] == pytest.approx(100.0)

    def test_max_lookback(self):
        series = NonRepaintBarAccess("close", max_lookback=3)
        for i in range(10):
            series.append(float(i))
        assert len(series) == 3

    def test_repr(self):
        series = NonRepaintBarAccess("rsi")
        r = repr(series)
        assert "rsi" in r
        assert "NonRepaintBarAccess" in r

    def test_append_resets_closed_flag(self):
        series = NonRepaintBarAccess("close")
        series.append(100.0)
        series.mark_bar_closed()
        series.append(101.0)  # New bar -> reset
        assert series.is_bar_closed is False


class TestNonRepaintIndicator:
    def test_init(self):
        ind = NonRepaintIndicator("rsi", period=14)
        assert ind.name == "rsi"
        assert ind.period == 14
        assert ind.is_ready is False

    def test_update_and_get_current(self):
        ind = NonRepaintIndicator("sma", period=3)
        ind.update(50.0)
        ind.update(51.0)
        assert ind.get_current() == pytest.approx(51.0)
        assert ind.is_ready is False  # Only 2 values, need 3
        ind.update(52.0)
        assert ind.is_ready is True

    def test_get_current_empty(self):
        ind = NonRepaintIndicator("sma", period=3)
        assert ind.get_current() is None

    def test_get_previous(self):
        ind = NonRepaintIndicator("sma", period=2)
        ind.update(10.0)
        ind.update(20.0)
        ind.update(30.0)
        assert ind.get_previous(1) == pytest.approx(20.0)
        assert ind.get_previous(2) == pytest.approx(10.0)

    def test_get_previous_out_of_range(self):
        ind = NonRepaintIndicator("sma", period=2)
        ind.update(10.0)
        assert ind.get_previous(0) is None  # bars_ago < 1
        assert ind.get_previous(5) is None  # Not enough data
