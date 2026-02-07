"""
Tests for src.utils.safe_math

Coverage targets:
- SafeMath: to_decimal, quantize, is_valid, is_nan, is_inf, is_zero, is_not_zero,
  is_close, safe_div, safe_log, safe_log1p, safe_sqrt, safe_pow, safe_exp,
  clamp, soft_clamp, clamp_positive, is_equal, is_greater, is_less, safe_clip,
  safe_mean, safe_percentile, safe_min, safe_max, normalize_logits,
  running_mean_update, running_variance_update
- RunningStats: update, get_variance, get_std, get_mean, get_z_score, reset
- safe_array_operation (module-level)
"""

import math

import numpy as np
import pytest

from src.utils.safe_math import (
    EXP_LOWER_GUARD,
    EXP_UPPER_GUARD,
    SAFE_EPSILON,
    RunningStats,
    SafeMath,
    safe_array_operation,
)


# ── SafeMath.to_decimal / quantize ──────────────────────────────────────────

class TestToDecimalQuantize:
    def test_to_decimal_basic(self):
        from decimal import Decimal
        result = SafeMath.to_decimal(1.23456, 2)
        assert result == Decimal("1.23")

    def test_to_decimal_five_digits(self):
        from decimal import Decimal
        result = SafeMath.to_decimal(1.234567, 5)
        assert result == Decimal("1.23457")

    def test_to_decimal_invalid(self):
        from decimal import Decimal
        result = SafeMath.to_decimal("not_a_number", 2)
        assert result == Decimal("0.00")

    def test_to_decimal_none(self):
        from decimal import Decimal
        result = SafeMath.to_decimal(None, 3)
        assert result == Decimal("0.000")

    def test_quantize_basic(self):
        from decimal import Decimal
        result = SafeMath.quantize(Decimal("1.23456"), 3)
        assert result == Decimal("1.235")

    def test_quantize_invalid(self):
        from decimal import Decimal
        result = SafeMath.quantize("bad", 2)
        assert result == Decimal("0.00")


# ── SafeMath type checks ───────────────────────────────────────────────────

class TestTypeChecks:
    def test_is_valid_float(self):
        assert SafeMath.is_valid(1.0) is True
        assert SafeMath.is_valid(float("nan")) is False
        assert SafeMath.is_valid(float("inf")) is False
        assert SafeMath.is_valid(float("-inf")) is False

    def test_is_valid_array(self):
        assert SafeMath.is_valid(np.array([1.0, 2.0])) == True
        assert SafeMath.is_valid(np.array([1.0, np.nan])) == False
        assert SafeMath.is_valid(np.array([1.0, np.inf])) == False

    def test_is_nan_float(self):
        assert SafeMath.is_nan(float("nan")) is True
        assert SafeMath.is_nan(1.0) is False

    def test_is_nan_array(self):
        assert SafeMath.is_nan(np.array([1.0, np.nan])) == True
        assert SafeMath.is_nan(np.array([1.0, 2.0])) == False

    def test_is_inf_float(self):
        assert SafeMath.is_inf(float("inf")) is True
        assert SafeMath.is_inf(float("-inf")) is True
        assert SafeMath.is_inf(1.0) is False

    def test_is_inf_array(self):
        assert SafeMath.is_inf(np.array([1.0, np.inf])) == True
        assert SafeMath.is_inf(np.array([1.0, 2.0])) == False

    def test_is_zero(self):
        assert SafeMath.is_zero(0.0) is True
        assert SafeMath.is_zero(1e-12) is True
        assert SafeMath.is_zero(1.0) is False

    def test_is_not_zero(self):
        assert SafeMath.is_not_zero(1.0) is True
        assert SafeMath.is_not_zero(0.0) is False


# ── SafeMath comparison helpers ─────────────────────────────────────────────

class TestComparisonHelpers:
    def test_is_close(self):
        assert SafeMath.is_close(1.0, 1.0 + 1e-12) is True
        assert SafeMath.is_close(1.0, 2.0) is False
        assert SafeMath.is_close(0.0, 1e-12) is True  # within abs_tol

    def test_is_equal(self):
        assert SafeMath.is_equal(1.0, 1.0) is True
        assert SafeMath.is_equal(1.0, 1.0 + 1e-12) is True
        assert SafeMath.is_equal(1.0, 2.0) is False

    def test_is_greater(self):
        assert SafeMath.is_greater(2.0, 1.0) is True
        assert SafeMath.is_greater(1.0, 1.0) is False  # Not greater with epsilon
        assert SafeMath.is_greater(1.0 + 1e-12, 1.0) is False  # Within epsilon

    def test_is_less(self):
        assert SafeMath.is_less(1.0, 2.0) is True
        assert SafeMath.is_less(1.0, 1.0) is False
        assert SafeMath.is_less(1.0 - 1e-12, 1.0) is False


# ── SafeMath safe operations ───────────────────────────────────────────────

class TestSafeOperations:
    def test_safe_div_normal(self):
        assert SafeMath.safe_div(10.0, 2.0) == pytest.approx(5.0)

    def test_safe_div_zero(self):
        assert SafeMath.safe_div(10.0, 0.0) == pytest.approx(0.0)
        assert SafeMath.safe_div(10.0, 0.0, default=-1.0) == pytest.approx(-1.0)

    def test_safe_div_tiny(self):
        assert SafeMath.safe_div(1.0, 1e-20) == pytest.approx(0.0)  # abs(b) < SAFE_DIV_MIN

    def test_safe_log_normal(self):
        assert abs(SafeMath.safe_log(math.e) - 1.0) < 1e-10

    def test_safe_log_zero(self):
        assert SafeMath.safe_log(0.0) == pytest.approx(0.0)

    def test_safe_log_negative(self):
        assert SafeMath.safe_log(-1.0) == pytest.approx(0.0)

    def test_safe_log_custom_default(self):
        assert SafeMath.safe_log(-1.0, default=-99.0) == pytest.approx(-99.0)

    def test_safe_log1p_normal(self):
        assert abs(SafeMath.safe_log1p(0.0)) < 1e-10  # log(1+0) = 0

    def test_safe_log1p_minus_one(self):
        assert SafeMath.safe_log1p(-1.0) == pytest.approx(0.0)  # x <= -1.0

    def test_safe_log1p_below_minus_one(self):
        assert SafeMath.safe_log1p(-2.0) == pytest.approx(0.0)

    def test_safe_sqrt_normal(self):
        assert SafeMath.safe_sqrt(4.0) == pytest.approx(2.0)

    def test_safe_sqrt_negative(self):
        assert SafeMath.safe_sqrt(-1.0) == pytest.approx(0.0)

    def test_safe_sqrt_zero(self):
        assert SafeMath.safe_sqrt(0.0) == pytest.approx(0.0)

    def test_safe_pow_normal(self):
        assert abs(SafeMath.safe_pow(2.0, 3.0) - 8.0) < 1e-8

    def test_safe_pow_overflow(self):
        assert SafeMath.safe_pow(10.0, 1000.0) == pytest.approx(0.0)  # Overflow protection

    def test_safe_pow_value_error(self):
        # Negative base with non-integer exponent
        assert SafeMath.safe_pow(-1.0, 0.5) == pytest.approx(0.0)

    def test_safe_exp_normal(self):
        assert abs(SafeMath.safe_exp(0.0) - 1.0) < 1e-10

    def test_safe_exp_overflow(self):
        assert SafeMath.safe_exp(200.0) == pytest.approx(0.0)

    def test_safe_exp_underflow(self):
        assert SafeMath.safe_exp(-200.0) == pytest.approx(0.0)

    def test_safe_clip_normal(self):
        assert SafeMath.safe_clip(5.0, 0.0, 10.0) == pytest.approx(5.0)

    def test_safe_clip_nan(self):
        assert SafeMath.safe_clip(float("nan"), 0.0, 10.0) == pytest.approx(0.0)

    def test_safe_clip_inf(self):
        assert SafeMath.safe_clip(float("inf"), 0.0, 10.0) == pytest.approx(10.0)
        assert SafeMath.safe_clip(float("-inf"), 0.0, 10.0) == pytest.approx(0.0)


# ── Clamp operations ──────────────────────────────────────────────────────

class TestClampOperations:
    def test_clamp_within_range(self):
        assert SafeMath.clamp(5.0, 0.0, 10.0) == pytest.approx(5.0)

    def test_clamp_below_min(self):
        assert SafeMath.clamp(-5.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_clamp_above_max(self):
        assert SafeMath.clamp(15.0, 0.0, 10.0) == pytest.approx(10.0)

    def test_soft_clamp_center(self):
        # At x=0, tanh(0)=0, so result is center
        result = SafeMath.soft_clamp(0.0, -1.0, 1.0)
        assert abs(result) < 1e-10

    def test_soft_clamp_stays_in_bounds(self):
        for x in [-1000, -10, -1, 0, 1, 10, 1000]:
            result = SafeMath.soft_clamp(float(x), -5.0, 5.0)
            assert -5.0 <= result <= 5.0

    def test_clamp_positive(self):
        assert SafeMath.clamp_positive(5.0) == pytest.approx(5.0)
        assert SafeMath.clamp_positive(-5.0) > 0
        assert SafeMath.clamp_positive(0.0) > 0


# ── Aggregate operations ─────────────────────────────────────────────────

class TestAggregateOperations:
    def test_safe_mean_normal(self):
        assert SafeMath.safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_safe_mean_empty(self):
        assert SafeMath.safe_mean([]) == pytest.approx(0.0)

    def test_safe_mean_with_nan(self):
        result = SafeMath.safe_mean([1.0, float("nan"), 3.0])
        assert abs(result - 2.0) < 1e-10  # nanmean skips NaN

    def test_safe_mean_custom_default(self):
        assert SafeMath.safe_mean([], default=-1.0) == pytest.approx(-1.0)

    def test_safe_percentile_normal(self):
        result = SafeMath.safe_percentile([1, 2, 3, 4, 5], 50)
        assert abs(result - 3.0) < 1e-10

    def test_safe_percentile_empty(self):
        assert SafeMath.safe_percentile([], 50) == pytest.approx(0.0)

    def test_safe_percentile_with_nan(self):
        result = SafeMath.safe_percentile([1.0, float("nan"), 3.0, 5.0], 50)
        assert result == pytest.approx(3.0)

    def test_safe_min_normal(self):
        assert SafeMath.safe_min([3.0, 1.0, 2.0]) == pytest.approx(1.0)

    def test_safe_min_empty(self):
        assert SafeMath.safe_min([]) == pytest.approx(0.0)

    def test_safe_min_with_nan(self):
        assert SafeMath.safe_min([float("nan"), 5.0, 3.0]) == pytest.approx(3.0)

    def test_safe_max_normal(self):
        assert SafeMath.safe_max([1.0, 3.0, 2.0]) == pytest.approx(3.0)

    def test_safe_max_empty(self):
        assert SafeMath.safe_max([]) == pytest.approx(0.0)

    def test_safe_max_with_nan(self):
        assert SafeMath.safe_max([float("nan"), 2.0, 5.0]) == pytest.approx(5.0)


# ── normalize_logits ──────────────────────────────────────────────────────

class TestNormalizeLogits:
    def test_uniform_logits(self):
        result = SafeMath.normalize_logits(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)

    def test_sums_to_one(self):
        result = SafeMath.normalize_logits(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(np.sum(result), 1.0, atol=1e-10)

    def test_temperature_sharpens(self):
        logits = np.array([1.0, 2.0, 3.0])
        sharp = SafeMath.normalize_logits(logits, temperature=0.1)
        flat = SafeMath.normalize_logits(logits, temperature=10.0)
        # Lower temperature → sharper distribution → max prob higher
        assert np.max(sharp) > np.max(flat)

    def test_invalid_logits_returns_uniform(self):
        result = SafeMath.normalize_logits(np.array([np.nan, 1.0, 2.0]))
        expected = np.ones(3) / 3
        np.testing.assert_allclose(result, expected, atol=1e-10)


# ── Running mean/variance ────────────────────────────────────────────────

class TestRunningUpdates:
    def test_running_mean_update(self):
        mean = 0.0
        for i, val in enumerate([2.0, 4.0, 6.0], start=1):
            mean = SafeMath.running_mean_update(mean, val, i)
        assert abs(mean - 4.0) < 1e-10

    def test_running_mean_count_zero(self):
        # count=0 → max(count,1)=1
        result = SafeMath.running_mean_update(5.0, 10.0, 0)
        assert abs(result - 10.0) < 1e-10

    def test_running_variance_update_below_min(self):
        result = SafeMath.running_variance_update(0.0, 0.0, 0.0, 1.0, 1)
        assert result == pytest.approx(0.0)  # count < MIN_SAMPLE_COUNT


# ── RunningStats ─────────────────────────────────────────────────────────

class TestRunningStats:
    def test_empty_stats(self):
        rs = RunningStats()
        assert rs.count == 0
        assert rs.get_mean() == pytest.approx(0.0)
        assert rs.get_variance() == pytest.approx(0.0)
        assert rs.get_std() == pytest.approx(0.0)

    def test_single_value(self):
        rs = RunningStats()
        rs.update(5.0)
        assert rs.count == 1
        assert rs.get_mean() == pytest.approx(5.0)
        assert rs.get_variance() == pytest.approx(0.0)  # count < MIN_SAMPLE_COUNT

    def test_multiple_values(self):
        rs = RunningStats()
        values = [2.0, 4.0, 6.0, 8.0, 10.0]
        for v in values:
            rs.update(v)
        assert abs(rs.get_mean() - 6.0) < 1e-10
        expected_var = np.var(values, ddof=1)
        assert abs(rs.get_variance() - expected_var) < 1e-10

    def test_std(self):
        rs = RunningStats()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rs.update(v)
        expected = np.std([1, 2, 3, 4, 5], ddof=1)
        assert abs(rs.get_std() - expected) < 1e-10

    def test_min_max_tracking(self):
        rs = RunningStats()
        for v in [5.0, 1.0, 9.0, 3.0]:
            rs.update(v)
        assert rs.min_val == pytest.approx(1.0)
        assert rs.max_val == pytest.approx(9.0)

    def test_ignore_nan(self):
        rs = RunningStats()
        rs.update(5.0)
        rs.update(float("nan"))
        assert rs.count == 1

    def test_ignore_inf(self):
        rs = RunningStats()
        rs.update(5.0)
        rs.update(float("inf"))
        assert rs.count == 1

    def test_z_score_normal(self):
        rs = RunningStats()
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            rs.update(v)
        z = rs.get_z_score(30.0)  # Mean = 30, so z = 0
        assert abs(z) < 1e-10

    def test_z_score_zero_std(self):
        rs = RunningStats()
        rs.update(5.0)
        z = rs.get_z_score(10.0)
        assert z == pytest.approx(0.0)  # std < SAFE_EPSILON → returns 0

    def test_reset(self):
        rs = RunningStats()
        for v in [1.0, 2.0, 3.0]:
            rs.update(v)
        rs.reset()
        assert rs.count == 0
        assert rs.mean == pytest.approx(0.0)
        assert rs.m2 == pytest.approx(0.0)
        assert rs.min_val == float("inf")
        assert rs.max_val == float("-inf")


# ── safe_array_operation ────────────────────────────────────────────────────

class TestSafeArrayOperation:
    def test_mean(self):
        assert safe_array_operation(np.array([1.0, 2.0, 3.0]), "mean") == pytest.approx(2.0)

    def test_std(self):
        result = safe_array_operation(np.array([1.0, 2.0, 3.0]), "std")
        assert abs(result - np.std([1, 2, 3])) < 1e-10

    def test_min(self):
        assert safe_array_operation(np.array([3.0, 1.0, 2.0]), "min") == pytest.approx(1.0)

    def test_max(self):
        assert safe_array_operation(np.array([1.0, 3.0, 2.0]), "max") == pytest.approx(3.0)

    def test_sum(self):
        assert safe_array_operation(np.array([1.0, 2.0, 3.0]), "sum") == pytest.approx(6.0)

    def test_median(self):
        assert safe_array_operation(np.array([1.0, 2.0, 3.0]), "median") == pytest.approx(2.0)

    def test_empty_array(self):
        assert safe_array_operation(np.array([]), "mean") == pytest.approx(0.0)

    def test_none_array(self):
        assert safe_array_operation(None, "mean") == pytest.approx(0.0)

    def test_unknown_operation(self):
        assert safe_array_operation(np.array([1.0]), "bogus") == pytest.approx(0.0)

    def test_invalid_array(self):
        assert safe_array_operation(np.array([np.nan, np.inf]), "mean") == pytest.approx(0.0)

    def test_custom_default(self):
        assert safe_array_operation(np.array([]), "mean", default=-99.0) == pytest.approx(-99.0)
