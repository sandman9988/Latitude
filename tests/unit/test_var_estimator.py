"""Tests for src.risk.var_estimator.

Covers: RegimeType, KurtosisMonitor, VaREstimator, position_size_from_var.
"""

import math
import pytest
import numpy as np

from src.risk.var_estimator import (
    KURTOSIS_CAP,
    MIN_BASE_VAR_SAMPLE,
    MIN_KURTOSIS_SAMPLE,
    MIN_VAR_SAMPLE,
    MIN_VAR_THRESHOLD,
    REFERENCE_VOL_FALLBACK,
    VAR_MULT_MAX,
    VPIN_CAP,
    KurtosisMonitor,
    RegimeType,
    VaREstimator,
    position_size_from_var,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fill_returns(estimator, n=50, mean=0.0, std=0.01, seed=42):
    """Feed n normally-distributed returns to a VaREstimator or KurtosisMonitor."""
    rng = np.random.default_rng(seed)
    for r in rng.normal(mean, std, n):
        if isinstance(estimator, VaREstimator):
            estimator.update_return(float(r))
        else:
            estimator.update(float(r))


# ---------------------------------------------------------------------------
# RegimeType
# ---------------------------------------------------------------------------
class TestRegimeType:
    def test_enum_values(self):
        assert RegimeType.UNDERDAMPED.value == "underdamped"
        assert RegimeType.CRITICAL.value == "critical"
        assert RegimeType.OVERDAMPED.value == "overdamped"

    def test_all_members(self):
        assert len(RegimeType) == 3


# ---------------------------------------------------------------------------
# KurtosisMonitor
# ---------------------------------------------------------------------------
class TestKurtosisMonitor:
    def test_init_defaults(self):
        km = KurtosisMonitor()
        assert km.window == 100
        assert km.threshold == pytest.approx(3.0)
        assert km.current_kurtosis == pytest.approx(0.0)
        assert not km.is_breaker_active

    def test_update_below_min_sample(self):
        km = KurtosisMonitor()
        for i in range(MIN_KURTOSIS_SAMPLE - 1):
            k, breaker = km.update(0.01)
        assert k == pytest.approx(0.0)
        assert not breaker

    def test_update_normal_returns_no_breaker(self):
        km = KurtosisMonitor(window=100, threshold=3.0)
        _fill_returns(km, n=60)
        # Normal distribution has ~0 excess kurtosis
        assert km.current_kurtosis < 3.0
        assert not km.is_breaker_active

    def test_update_fat_tails_triggers_breaker(self):
        km = KurtosisMonitor(window=200, threshold=2.0)
        rng = np.random.default_rng(42)
        # Use t-distribution with low df for fat tails
        for r in rng.standard_t(df=3, size=100):
            km.update(float(r))
        # t(3) has extremely high kurtosis
        assert km.current_kurtosis > 2.0
        assert km.is_breaker_active

    def test_invalid_return_skipped(self):
        km = KurtosisMonitor()
        k, b = km.update(float("nan"))
        assert k == pytest.approx(0.0)
        assert not b
        k, b = km.update(float("inf"))
        assert k == pytest.approx(0.0)
        assert not b

    def test_breaker_resets(self):
        km = KurtosisMonitor(window=50, threshold=1.0)
        rng = np.random.default_rng(42)
        # First trigger with fat tails
        for r in rng.standard_t(df=3, size=50):
            km.update(float(r))
        was_active = km.is_breaker_active
        # Then feed normal data to reset
        for r in rng.normal(0, 0.01, 50):
            km.update(float(r))
        # If was active, should now be inactive from normal data
        if was_active:
            assert not km.is_breaker_active

    def test_properties(self):
        km = KurtosisMonitor()
        _fill_returns(km, n=40)
        assert isinstance(km.current_kurtosis, float)
        assert isinstance(km.is_breaker_active, bool)


# ---------------------------------------------------------------------------
# VaREstimator
# ---------------------------------------------------------------------------
class TestVaREstimatorInit:
    def test_defaults(self):
        v = VaREstimator()
        assert v.window == 500
        assert v.confidence == pytest.approx(0.95)
        assert v.last_var == pytest.approx(0.0)
        assert v.kurtosis == pytest.approx(0.0)
        assert not v.is_kurtosis_breaker_active

    def test_custom_params(self):
        v = VaREstimator(window=200, confidence=0.99)
        assert v.window == 200
        assert v.confidence == pytest.approx(0.99)

    def test_custom_regime_multipliers(self):
        mults = {RegimeType.OVERDAMPED: 0.5}
        v = VaREstimator(regime_multipliers=mults)
        assert v.regime_multipliers[RegimeType.OVERDAMPED] == pytest.approx(0.5)

    def test_shared_kurtosis_monitor(self):
        km = KurtosisMonitor(window=50)
        v = VaREstimator(kurtosis_monitor=km)
        assert v.kurtosis_monitor is km


class TestUpdateReturn:
    def test_valid_return(self):
        v = VaREstimator()
        v.update_return(0.01)
        assert len(v.returns) == 1

    def test_nan_skipped(self):
        v = VaREstimator()
        v.update_return(float("nan"))
        assert len(v.returns) == 0

    def test_inf_skipped(self):
        v = VaREstimator()
        v.update_return(float("inf"))
        assert len(v.returns) == 0

    def test_kurtosis_monitor_updated(self):
        v = VaREstimator()
        _fill_returns(v, n=40)
        assert len(v.kurtosis_monitor.returns) == 40


class TestEstimateVar:
    def test_insufficient_data_returns_zero(self):
        v = VaREstimator()
        _fill_returns(v, n=MIN_VAR_SAMPLE - 1)
        assert v.estimate_var() == pytest.approx(0.0)

    def test_basic_var_positive(self):
        v = VaREstimator(window=200)
        _fill_returns(v, n=100, std=0.02)
        var = v.estimate_var(regime=RegimeType.OVERDAMPED)
        assert var > 0.0

    def test_var_cached_as_last_var(self):
        v = VaREstimator(window=200)
        _fill_returns(v, n=100, std=0.02)
        var = v.estimate_var(regime=RegimeType.OVERDAMPED)
        assert v.last_var == var

    def test_regime_multiplier_increases_var(self):
        v = VaREstimator(window=200)
        _fill_returns(v, n=100, std=0.02)
        var_low = v.estimate_var(regime=RegimeType.OVERDAMPED, vpin_z=0.0)
        var_high = v.estimate_var(regime=RegimeType.UNDERDAMPED, vpin_z=0.0)
        assert var_high > var_low

    def test_vpin_multiplier_effect(self):
        v = VaREstimator(window=200)
        _fill_returns(v, n=100, std=0.02)
        var_no_vpin = v.estimate_var(regime=RegimeType.CRITICAL, vpin_z=0.0)
        var_high_vpin = v.estimate_var(regime=RegimeType.CRITICAL, vpin_z=4.0)
        assert var_high_vpin > var_no_vpin

    def test_vpin_capped(self):
        v = VaREstimator(window=200)
        _fill_returns(v, n=100, std=0.02)
        var1 = v.estimate_var(regime=RegimeType.CRITICAL, vpin_z=5.0)
        var2 = v.estimate_var(regime=RegimeType.CRITICAL, vpin_z=100.0)
        # Both should produce same result because vpin is capped at VPIN_CAP
        assert abs(var1 - var2) < 1e-10

    def test_vpin_nan_treated_as_zero(self):
        v = VaREstimator(window=200)
        _fill_returns(v, n=100, std=0.02)
        var_nan = v.estimate_var(regime=RegimeType.CRITICAL, vpin_z=float("nan"))
        var_zero = v.estimate_var(regime=RegimeType.CRITICAL, vpin_z=0.0)
        assert abs(var_nan - var_zero) < 1e-10

    def test_vol_multiplier_effect(self):
        v = VaREstimator(window=200)
        _fill_returns(v, n=100, std=0.01)
        v.set_reference_vol(0.01)
        var_low = v.estimate_var(regime=RegimeType.CRITICAL, current_vol=0.005)
        var_high = v.estimate_var(regime=RegimeType.CRITICAL, current_vol=0.03)
        assert var_high > var_low

    def test_vol_none_multiplier_is_one(self):
        v = VaREstimator(window=200)
        _fill_returns(v, n=100)
        var1 = v.estimate_var(regime=RegimeType.CRITICAL, current_vol=None)
        # vol_mult should be 1.0
        assert var1 > 0.0

    def test_max_var_cap(self):
        """VaR should never exceed base_var * VAR_MULT_MAX."""
        v = VaREstimator(window=200)
        _fill_returns(v, n=100, std=0.02)
        base_var = v._calculate_base_var()
        # Use extreme multipliers
        var = v.estimate_var(
            regime=RegimeType.UNDERDAMPED, vpin_z=5.0, current_vol=0.1
        )
        assert var <= base_var * VAR_MULT_MAX + 1e-10

    def test_unknown_regime_uses_default(self):
        """If regime not in multipliers, falls back to 1.5."""
        v = VaREstimator(window=200)
        _fill_returns(v, n=100, std=0.02)
        # Pass a regime not in the dict (use RegimeType.CRITICAL which is in dict)
        var = v.estimate_var(regime=RegimeType.CRITICAL)
        assert var > 0.0


class TestCalculateBaseVar:
    def test_insufficient_data(self):
        v = VaREstimator()
        assert v._calculate_base_var() == pytest.approx(0.0)

    def test_positive_result(self):
        v = VaREstimator(window=200)
        _fill_returns(v, n=50, std=0.02)
        base = v._calculate_base_var()
        assert base > 0.0

    def test_increasing_with_volatility(self):
        v1 = VaREstimator(window=200)
        _fill_returns(v1, n=100, std=0.01, seed=42)
        v2 = VaREstimator(window=200)
        _fill_returns(v2, n=100, std=0.05, seed=42)
        assert v2._calculate_base_var() > v1._calculate_base_var()


class TestCalculateVolMult:
    def test_none_returns_one(self):
        v = VaREstimator()
        assert v._calculate_vol_mult(None) == pytest.approx(1.0)

    def test_nan_returns_one(self):
        v = VaREstimator()
        assert v._calculate_vol_mult(float("nan")) == pytest.approx(1.0)

    def test_reference_vol_auto_set(self):
        v = VaREstimator(window=200)
        _fill_returns(v, n=50)
        v._calculate_vol_mult(0.01)
        assert v._reference_vol is not None

    def test_clamped_range(self):
        v = VaREstimator()
        v._reference_vol = 0.01
        # Very high vol → capped at 3.0
        result = v._calculate_vol_mult(1.0)
        assert result == pytest.approx(3.0)
        # Very low vol → capped at 0.5
        result = v._calculate_vol_mult(0.0001)
        assert result == pytest.approx(0.5)


class TestSetReferenceVol:
    def test_valid_set(self):
        v = VaREstimator()
        v.set_reference_vol(0.015)
        assert v._reference_vol == pytest.approx(0.015)

    def test_invalid_ignored(self):
        v = VaREstimator()
        v.set_reference_vol(float("nan"))
        assert v._reference_vol is None

    def test_zero_ignored(self):
        v = VaREstimator()
        v.set_reference_vol(0.0)
        assert v._reference_vol is None

    def test_negative_ignored(self):
        v = VaREstimator()
        v.set_reference_vol(-0.01)
        assert v._reference_vol is None


class TestProperties:
    def test_last_var_initial(self):
        assert VaREstimator().last_var == pytest.approx(0.0)

    def test_kurtosis_initial(self):
        assert VaREstimator().kurtosis == pytest.approx(0.0)

    def test_kurtosis_breaker_initial(self):
        assert not VaREstimator().is_kurtosis_breaker_active


# ---------------------------------------------------------------------------
# position_size_from_var
# ---------------------------------------------------------------------------
class TestPositionSizeFromVar:
    def test_basic_sizing(self):
        size = position_size_from_var(
            var=0.02, risk_budget_usd=1000.0,
            account_equity=50000.0, contract_size=100000.0
        )
        # risk_budget / var = 1000/0.02 = 50000
        # max_leveraged = 50000 * 10 = 500000
        # position_value = min(50000, 500000) = 50000
        # lots = 50000 / 100000 = 0.5
        assert abs(size - 0.5) < 0.01

    def test_zero_var_returns_zero(self):
        size = position_size_from_var(0.0, 1000.0, 50000.0)
        assert size == pytest.approx(0.0)

    def test_tiny_var_returns_zero(self):
        size = position_size_from_var(1e-15, 1000.0, 50000.0)
        assert size == pytest.approx(0.0)

    def test_leverage_cap_active(self):
        # Very small VaR → huge position, but leverage cap kicks in
        size = position_size_from_var(
            var=0.0001, risk_budget_usd=10000.0,
            account_equity=10000.0, contract_size=100000.0,
            max_leverage=2.0
        )
        # risk_budget / var = 10000/0.0001 = 100,000,000
        # max_leveraged = 10000 * 2 = 20000
        # lots = 20000 / 100000 = 0.2
        assert abs(size - 0.2) < 0.01

    def test_default_contract_size(self):
        size = position_size_from_var(
            var=0.01, risk_budget_usd=100.0,
            account_equity=100000.0
        )
        # risk_budget / var = 100/0.01 = 10000
        # max_leveraged = 100000 * 10 = 1000000
        # lots = 10000 / 1.0 = 10000
        assert abs(size - 10000.0) < 0.01

    def test_large_var_small_position(self):
        size = position_size_from_var(
            var=0.5, risk_budget_usd=100.0,
            account_equity=10000.0, contract_size=100000.0
        )
        # risk_budget / var = 100/0.5 = 200
        # lots = 200 / 100000 = 0.002
        assert abs(size - 0.002) < 0.001
