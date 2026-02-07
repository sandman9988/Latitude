"""Tests for src.utils.safe_utils – SafeMath, SafeArray, SafeDeque, helper functions."""

import math
from collections import deque

import pytest

from src.utils.safe_utils import (
    SafeArray,
    SafeDeque,
    SafeMath,
    safe_mean,
    safe_percentile,
    safe_std,
    utc_now,
    utc_ts_ms,
)


# ---------------------------------------------------------------------------
# SafeMath
# ---------------------------------------------------------------------------


class TestSafeMath:
    # -- is_valid --
    def test_is_valid_normal(self):
        assert SafeMath.is_valid(1.0) is True
        assert SafeMath.is_valid(0) is True
        assert SafeMath.is_valid(-99.9) is True

    def test_is_valid_nan(self):
        assert SafeMath.is_valid(float("nan")) is False

    def test_is_valid_inf(self):
        assert SafeMath.is_valid(float("inf")) is False
        assert SafeMath.is_valid(float("-inf")) is False

    def test_is_valid_none(self):
        assert SafeMath.is_valid(None) is False

    def test_is_valid_string(self):
        assert SafeMath.is_valid("hello") is False

    # -- safe_div --
    def test_safe_div_normal(self):
        assert SafeMath.safe_div(10, 2) == pytest.approx(5.0)

    def test_safe_div_zero_denom(self):
        assert SafeMath.safe_div(10, 0) == pytest.approx(0.0)

    def test_safe_div_nan_numerator(self):
        assert SafeMath.safe_div(float("nan"), 2) == pytest.approx(0.0)

    def test_safe_div_inf_denominator(self):
        assert SafeMath.safe_div(10, float("inf")) == pytest.approx(0.0)

    def test_safe_div_custom_default(self):
        assert SafeMath.safe_div(10, 0, default=-1.0) == pytest.approx(-1.0)

    def test_safe_div_tiny_denominator(self):
        assert SafeMath.safe_div(10, 1e-15) == pytest.approx(0.0)

    # -- clamp --
    def test_clamp_in_range(self):
        assert SafeMath.clamp(5, 0, 10) == 5

    def test_clamp_below(self):
        assert SafeMath.clamp(-5, 0, 10) == 0

    def test_clamp_above(self):
        assert SafeMath.clamp(15, 0, 10) == 10

    def test_clamp_nan_returns_midpoint(self):
        assert SafeMath.clamp(float("nan"), 0, 10) == pytest.approx(5.0)

    def test_clamp_inf_returns_upper(self):
        assert SafeMath.clamp(float("inf"), 0, 10) == pytest.approx(5.0)  # midpoint for invalid

    # -- safe_sqrt --
    def test_safe_sqrt_valid(self):
        assert SafeMath.safe_sqrt(4.0) == pytest.approx(2.0)

    def test_safe_sqrt_zero(self):
        assert SafeMath.safe_sqrt(0.0) == pytest.approx(0.0)

    def test_safe_sqrt_negative(self):
        assert SafeMath.safe_sqrt(-1.0) == pytest.approx(0.0)

    def test_safe_sqrt_nan(self):
        assert SafeMath.safe_sqrt(float("nan")) == pytest.approx(0.0)

    # -- safe_log --
    def test_safe_log_valid(self):
        assert SafeMath.safe_log(math.e) == pytest.approx(1.0)

    def test_safe_log_zero(self):
        assert SafeMath.safe_log(0.0) == pytest.approx(0.0)

    def test_safe_log_negative(self):
        assert SafeMath.safe_log(-1.0) == pytest.approx(0.0)

    # -- safe_exp --
    def test_safe_exp_valid(self):
        assert SafeMath.safe_exp(0.0) == pytest.approx(1.0)
        assert SafeMath.safe_exp(1.0) == pytest.approx(math.e)

    def test_safe_exp_overflow(self):
        assert SafeMath.safe_exp(1000) == pytest.approx(1.0)  # default

    def test_safe_exp_nan(self):
        assert SafeMath.safe_exp(float("nan")) == pytest.approx(1.0)

    # -- sanitize --
    def test_sanitize_valid(self):
        assert SafeMath.sanitize(3.14) == pytest.approx(3.14)

    def test_sanitize_nan(self):
        assert SafeMath.sanitize(float("nan"), 0.0) == pytest.approx(0.0)

    def test_sanitize_inf(self):
        assert SafeMath.sanitize(float("inf"), -1.0) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# SafeArray
# ---------------------------------------------------------------------------


class TestSafeArray:
    # -- safe_get --
    def test_safe_get_normal(self):
        assert SafeArray.safe_get([10, 20, 30], 1) == 20

    def test_safe_get_out_of_bounds(self):
        assert SafeArray.safe_get([10, 20], 5) is None

    def test_safe_get_negative_index_out(self):
        assert SafeArray.safe_get([10], -1) is None  # index < 0 → out of bounds

    def test_safe_get_none_arr(self):
        assert SafeArray.safe_get(None, 0) is None

    def test_safe_get_deque(self):
        d = deque([1, 2, 3])
        assert SafeArray.safe_get(d, 2) == 3

    # -- safe_get_series --
    def test_safe_get_series_current(self):
        # bars_ago=0 → last element
        assert SafeArray.safe_get_series([10, 20, 30], 0) == 30

    def test_safe_get_series_prev(self):
        assert SafeArray.safe_get_series([10, 20, 30], 1) == 20

    def test_safe_get_series_out(self):
        assert SafeArray.safe_get_series([10, 20], 5) is None

    def test_safe_get_series_negative_bars_ago(self):
        assert SafeArray.safe_get_series([1, 2, 3], -1) is None

    # -- safe_last --
    def test_safe_last_nonempty(self):
        assert SafeArray.safe_last([1, 2, 3]) == 3

    def test_safe_last_empty(self):
        assert SafeArray.safe_last([]) is None

    # -- safe_slice --
    def test_safe_slice_normal(self):
        assert SafeArray.safe_slice([1, 2, 3, 4], 1, 3) == [2, 3]

    def test_safe_slice_none(self):
        result = SafeArray.safe_slice(None)
        assert len(result) == 0

    # -- is_empty --
    def test_is_empty_none(self):
        assert SafeArray.is_empty(None) is True

    def test_is_empty_empty(self):
        assert SafeArray.is_empty([]) is True

    def test_is_empty_nonempty(self):
        assert SafeArray.is_empty([1]) is False


# ---------------------------------------------------------------------------
# SafeDeque
# ---------------------------------------------------------------------------


class TestSafeDeque:
    def test_append_and_len(self):
        sd = SafeDeque(maxlen=3)
        sd.append(1)
        sd.append(2)
        assert len(sd) == 2

    def test_maxlen_enforced(self):
        sd = SafeDeque(maxlen=2)
        sd.append(1)
        sd.append(2)
        sd.append(3)
        assert len(sd) == 2
        assert sd.last() == 3

    def test_get_in_bounds(self):
        sd = SafeDeque(maxlen=5)
        sd.append(10)
        sd.append(20)
        assert sd.get(0) == 10
        assert sd.get(1) == 20

    def test_get_out_of_bounds(self):
        sd = SafeDeque(maxlen=5)
        assert sd.get(99, default=-1) == -1

    def test_get_series(self):
        sd = SafeDeque(maxlen=5)
        for v in [10, 20, 30]:
            sd.append(v)
        assert sd.get_series(0) == 30  # current
        assert sd.get_series(1) == 20

    def test_last_empty(self):
        sd = SafeDeque(maxlen=5)
        assert sd.last(default=-1) == -1

    def test_is_empty(self):
        sd = SafeDeque(maxlen=5)
        assert sd.is_empty is True
        sd.append(1)
        assert sd.is_empty is False

    def test_maxlen_property(self):
        sd = SafeDeque(maxlen=7)
        assert sd.maxlen == 7

    def test_iter(self):
        sd = SafeDeque(maxlen=5)
        sd.append(1)
        sd.append(2)
        assert list(sd) == [1, 2]

    def test_getitem(self):
        sd = SafeDeque(maxlen=5)
        sd.append(10)
        assert sd[0] == 10


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    # -- safe_mean --
    def test_safe_mean_normal(self):
        assert safe_mean([1, 2, 3]) == pytest.approx(2.0)

    def test_safe_mean_empty(self):
        assert safe_mean([]) == pytest.approx(0.0)

    def test_safe_mean_with_nan(self):
        result = safe_mean([1, float("nan"), 3])
        assert result == pytest.approx(2.0)

    def test_safe_mean_all_nan(self):
        assert safe_mean([float("nan"), float("nan")]) == pytest.approx(0.0)

    # -- safe_std --
    def test_safe_std_normal(self):
        result = safe_std([10, 10, 10])
        assert result == pytest.approx(0.0)

    def test_safe_std_varied(self):
        result = safe_std([0, 10])
        assert result > 0

    def test_safe_std_too_few(self):
        assert safe_std([1]) == pytest.approx(0.0)

    def test_safe_std_empty(self):
        assert safe_std([]) == pytest.approx(0.0)

    # -- safe_percentile --
    def test_safe_percentile_median(self):
        assert safe_percentile([1, 2, 3, 4, 5], 50) == pytest.approx(3.0)

    def test_safe_percentile_min(self):
        assert safe_percentile([10, 20, 30], 0) == pytest.approx(10.0)

    def test_safe_percentile_max(self):
        assert safe_percentile([10, 20, 30], 100) == pytest.approx(30.0)

    def test_safe_percentile_empty(self):
        assert safe_percentile([], 50) == pytest.approx(0.0)

    def test_safe_percentile_with_nan(self):
        result = safe_percentile([1, float("nan"), 3], 50)
        assert result == pytest.approx(2.0)

    # -- utc helpers --
    def test_utc_ts_ms_format(self):
        ts = utc_ts_ms()
        # Format: YYYYMMDD-HH:MM:SS.sss
        assert len(ts) == 21
        assert "-" in ts

    def test_utc_now_is_aware(self):
        now = utc_now()
        assert now.tzinfo is not None
