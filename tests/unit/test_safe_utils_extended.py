"""Extended tests for src.utils.safe_utils.

Covers: safe_div result NaN/Inf, safe_get no-len type, safe_get access exception,
safe_get_series None arr, safe_get_series no-len, safe_slice None/exception,
safe_percentile empty valid values, safe_percentile f==c path.
"""

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
)


# ---------------------------------------------------------------------------
# SafeMath edge cases
# ---------------------------------------------------------------------------
class TestSafeMathExtended:
    def test_safe_div_result_is_inf(self):
        """Division producing Inf should return default."""
        # float('inf') / 1.0 produces inf, but safe_div checks validity
        result = SafeMath.safe_div(float("inf"), 1.0, default=-1.0)
        # float('inf') numerator is invalid, returns default
        assert result == pytest.approx(-1.0)

    def test_is_valid_with_overflow_string(self):
        assert SafeMath.is_valid("not_a_number") is False

    def test_is_valid_with_none(self):
        assert SafeMath.is_valid(None) is False

    def test_clamp_nan_returns_midpoint(self):
        result = SafeMath.clamp(float("nan"), 10.0, 20.0)
        assert result == pytest.approx(15.0)

    def test_safe_sqrt_nan(self):
        assert SafeMath.safe_sqrt(float("nan"), 99.0) == pytest.approx(99.0)

    def test_safe_log_nan(self):
        assert SafeMath.safe_log(float("nan"), 42.0) == pytest.approx(42.0)

    def test_safe_log_zero(self):
        assert SafeMath.safe_log(0.0, -1.0) == pytest.approx(-1.0)

    def test_safe_exp_overflow(self):
        result = SafeMath.safe_exp(1000.0, 0.5)
        assert result == pytest.approx(0.5)

    def test_safe_exp_nan(self):
        assert SafeMath.safe_exp(float("nan"), 2.0) == pytest.approx(2.0)

    def test_sanitize_nan(self):
        assert SafeMath.sanitize(float("nan"), 7.0) == pytest.approx(7.0)

    def test_sanitize_valid(self):
        assert SafeMath.sanitize(3.14) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# SafeArray edge cases
# ---------------------------------------------------------------------------
class TestSafeArrayExtended:
    def test_safe_get_none_array(self):
        assert SafeArray.safe_get(None, 0, "default") == "default"

    def test_safe_get_no_getitem(self):
        assert SafeArray.safe_get(42, 0, "fallback") == "fallback"

    def test_safe_get_no_len(self):
        """Object with __getitem__ but no __len__."""

        class NoLen:
            def __getitem__(self, idx):
                return idx

        result = SafeArray.safe_get(NoLen(), 0, "nope")
        assert result == "nope"

    def test_safe_get_access_exception(self):
        """Object where __getitem__ raises on access."""

        class BadAccess:
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                raise TypeError("broken")

        result = SafeArray.safe_get(BadAccess(), 0, "err")
        assert result == "err"

    def test_safe_get_series_none(self):
        assert SafeArray.safe_get_series(None, 0, "d") == "d"

    def test_safe_get_series_no_len(self):
        class NoLen:
            def __getitem__(self, idx):
                return idx

        assert SafeArray.safe_get_series(NoLen(), 0, "d") == "d"

    def test_safe_get_series_negative_bars_ago(self):
        assert SafeArray.safe_get_series([1, 2, 3], -1, "d") == "d"

    def test_safe_slice_none(self):
        result = SafeArray.safe_slice(None)
        # None has no isinstance list check, returns [] or deque()
        assert len(result) == 0

    def test_safe_slice_deque_none(self):
        result = SafeArray.safe_slice(deque())
        assert result == deque()

    def test_safe_slice_bad_indices(self):
        result = SafeArray.safe_slice([1, 2, 3], "a", "b")
        assert result == []

    def test_is_empty_none(self):
        assert SafeArray.is_empty(None) is True

    def test_is_empty_non_iterable(self):
        assert SafeArray.is_empty(42) is True

    def test_is_empty_empty_list(self):
        assert SafeArray.is_empty([]) is True

    def test_is_empty_non_empty(self):
        assert SafeArray.is_empty([1]) is False


# ---------------------------------------------------------------------------
# SafeDeque
# ---------------------------------------------------------------------------
class TestSafeDequeExtended:
    def test_iter(self):
        sd = SafeDeque(maxlen=3)
        sd.append(10)
        sd.append(20)
        assert list(sd) == [10, 20]

    def test_getitem(self):
        sd = SafeDeque(maxlen=5)
        sd.append("a")
        sd.append("b")
        assert sd[1] == "b"

    def test_is_empty(self):
        sd = SafeDeque()
        assert sd.is_empty is True
        sd.append(1)
        assert sd.is_empty is False

    def test_maxlen(self):
        sd = SafeDeque(maxlen=10, name="test")
        assert sd.maxlen == 10


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------
class TestModuleFunctions:
    def test_safe_mean_all_nan(self):
        result = safe_mean([float("nan"), float("nan")], default=99.0)
        assert result == pytest.approx(99.0)

    def test_safe_std_all_nan(self):
        result = safe_std([float("nan"), float("nan"), float("nan")], default=5.0)
        assert result == pytest.approx(5.0)

    def test_safe_std_too_few(self):
        assert safe_std([1.0], default=0.0) == pytest.approx(0.0)

    def test_safe_percentile_empty(self):
        assert safe_percentile([], 50, default=7.0) == pytest.approx(7.0)

    def test_safe_percentile_all_nan(self):
        result = safe_percentile([float("nan"), float("nan")], 50, default=-1.0)
        assert result == pytest.approx(-1.0)

    def test_safe_percentile_exact_index(self):
        """When f == c (exact index match)."""
        # 5 elements, percentile 25 → k = (4)*0.25 = 1.0, f==c==1
        result = safe_percentile([10.0, 20.0, 30.0, 40.0, 50.0], 25)
        assert result == pytest.approx(20.0)

    def test_safe_percentile_interpolated(self):
        result = safe_percentile([10.0, 20.0, 30.0, 40.0, 50.0], 30)
        assert 20.0 < result < 30.0
