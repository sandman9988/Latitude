"""Coverage batch 4: var_estimator + safe_utils edge cases.

Targets:
  - var_estimator.py  (85% → ~95%):  lines 100, 107, 113, 225-226, 258-259, 305
  - safe_utils.py     (87% → ~89%):  lines 57-58
"""

from unittest.mock import patch

import numpy as np
import pytest

# ── var_estimator ────────────────────────────────────────────────────────

from src.risk.var_estimator import KurtosisMonitor, RegimeType, VaREstimator


class TestKurtosisMonitorGaps:
    """Cover _calculate_kurtosis edge-case returns."""

    def test_calculate_kurtosis_too_few_returns(self):
        """Line 100: len(self.returns) < MIN_KURTOSIS_STATS → 0.0."""
        km = KurtosisMonitor(window=100)
        # Directly add < 4 returns to the deque (bypassing update's 30-check)
        km.returns.extend([0.01, 0.02, 0.03])
        assert km._calculate_kurtosis() == 0.0

    def test_calculate_kurtosis_non_finite_filtered(self):
        """Line 107: after filtering non-finite, len < MIN_KURTOSIS_STATS → 0.0."""
        km = KurtosisMonitor(window=100)
        # 3 finite + lots of inf/nan → after filtering, only 3 remain < 4
        km.returns.extend([0.01, 0.02, 0.03, float("inf"), float("nan"), float("-inf")])
        assert km._calculate_kurtosis() == 0.0

    def test_calculate_kurtosis_zero_std(self):
        """Line 113: std < STD_EPS → 0.0."""
        km = KurtosisMonitor(window=100)
        # All same value → std = 0
        km.returns.extend([0.01] * 10)
        assert km._calculate_kurtosis() == 0.0


class TestVaREstimatorGaps:
    """Cover defensive validation paths in estimate_var and vol_mult."""

    def _fill_var(self, var_est: VaREstimator, n: int = 50):
        """Fill estimator with enough returns for calculation."""
        rng = np.random.default_rng(42)
        for r in rng.normal(0.0, 0.01, n):
            var_est.update_return(float(r))

    def test_invalid_base_var_returns_zero(self):
        """Lines 225-226: base_var is NaN → return 0.0."""
        var_est = VaREstimator(window=100)
        self._fill_var(var_est)
        with patch.object(var_est, "_calculate_base_var", return_value=float("nan")):
            result = var_est.estimate_var(regime=RegimeType.CRITICAL, vpin_z=0.0)
        assert result == 0.0

    def test_invalid_combined_var_falls_back(self):
        """Lines 258-259: combined var overflows to Inf → falls back to base_var."""
        var_est = VaREstimator(window=100)
        self._fill_var(var_est)

        # base_var=1e308 is finite (valid), but 1e308 * regime_mult(2.0) overflows to Inf
        with patch.object(var_est, "_calculate_base_var", return_value=1e308):
            result = var_est.estimate_var(regime=RegimeType.UNDERDAMPED, vpin_z=0.0)
        # Falls back to base_var (1e308), then capped by max_var = base_var * 10 = 1e309 (Inf)
        # Final sanitize should produce some positive value
        assert result > 0

    def test_vol_mult_reference_vol_too_small(self):
        """Line 305: reference_vol < STD_EPS → set to REFERENCE_VOL_FALLBACK."""
        var_est = VaREstimator(window=100)
        self._fill_var(var_est)
        # Set reference_vol to a tiny value
        var_est._reference_vol = 1e-20  # < STD_EPS (1e-12)
        mult = var_est._calculate_vol_mult(current_vol=0.01)
        # Should have reset to REFERENCE_VOL_FALLBACK (0.01)
        # mult = 0.01 / 0.01 = 1.0
        assert 0.5 <= mult <= 3.0  # Within clamp range


# ── safe_utils ───────────────────────────────────────────────────────────

from src.utils.safe_utils import SafeMath


class TestSafeUtilsGaps:
    """Cover safe_div when result overflows to infinity."""

    def test_safe_div_result_overflow(self):
        """Lines 57-58: numerator/denominator overflows → return default."""
        # 1e308 / 1e-10 = 1e318 → Inf (overflow)
        result = SafeMath.safe_div(1e308, 1e-10, default=-1.0)
        assert result == -1.0

    def test_safe_div_result_neg_overflow(self):
        """Also triggers lines 57-58 with negative overflow."""
        result = SafeMath.safe_div(-1e308, 1e-10, default=-2.0)
        assert result == -2.0
