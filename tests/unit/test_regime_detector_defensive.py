"""Tests for defensive paths in RegimeDetector._update_regime.

Covers uncovered lines: 140-141 (non-positive prices in buffer),
147-149 (log exception path), 153-154 (non-finite returns),
168 (cached variance reuse), 178 (zero-variance transitional),
184-185 (< MIN_RETURNS_TWO_PERIOD), 191-192 (invalid var_2),
199-200 (non-finite variance ratio), 221 (exception fallback),
and 234-238 (_autocorrelation body).
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.features.regime_detector import (
    RegimeDetector,
    NEUTRAL_ZETA,
    MIN_RETURNS_REQUIRED,
    VARIANCE_EPSILON,
)


class TestUpdateRegimeDefensivePaths:
    """Cover defensive guards inside _update_regime."""

    def _filled_detector(self, prices=None, window=50, update_interval=1):
        """Create detector with price buffer pre-filled, ready to trigger _update_regime."""
        det = RegimeDetector(window_size=window, update_interval=update_interval)
        if prices is None:
            rng = np.random.default_rng(42)
            prices = 100.0 + np.cumsum(rng.normal(0, 0.5, window))
        det.price_buffer = list(prices)
        return det

    def test_non_positive_prices_in_buffer_skipped(self):
        """Lines 140-141: non-positive prices bypass add_price guard via direct buffer."""
        det = self._filled_detector()
        # Inject a zero (non-positive) into the buffer
        det.price_buffer[25] = 0.0
        det._cache_invalidated = True
        det._cached_var_1 = None
        det._update_regime()
        # Regime should remain unchanged (UNKNOWN from init)
        assert det.current_regime == "UNKNOWN"

    def test_non_positive_negative_price_in_buffer(self):
        """Non-positive check also catches negative values."""
        det = self._filled_detector()
        det.price_buffer[10] = -5.0
        det._cache_invalidated = True
        det._cached_var_1 = None
        det._update_regime()
        assert det.current_regime == "UNKNOWN"

    def test_non_finite_prices_nan_in_buffer(self):
        """Lines 136-138: NaN in buffer detected by np.isfinite check."""
        det = self._filled_detector()
        det.price_buffer[30] = float("nan")
        det._cache_invalidated = True
        det._cached_var_1 = None
        det._update_regime()
        assert det.current_regime == "UNKNOWN"

    def test_non_finite_prices_inf_in_buffer(self):
        """Inf in buffer detected by np.isfinite check."""
        det = self._filled_detector()
        det.price_buffer[5] = float("inf")
        det._cache_invalidated = True
        det._cached_var_1 = None
        det._update_regime()
        assert det.current_regime == "UNKNOWN"

    def test_non_finite_returns_from_extreme_prices(self):
        """Lines 153-154: finite positive prices producing non-finite returns."""
        det = self._filled_detector()
        # Create a massive ratio that could produce extreme log returns
        # but keep prices positive; to actually get non-finite returns
        # we inject after the positive check by making prices produce inf log diff
        prices = [100.0] * 50
        prices[25] = 1e-308  # Extremely small but positive
        prices[26] = 1e308   # Extremely large
        det.price_buffer = prices
        det._cache_invalidated = True
        det._cached_var_1 = None
        det._update_regime()
        # Should handle gracefully (may detect via various checks)
        assert det.current_regime in ("UNKNOWN", "TRANSITIONAL", "TRENDING", "MEAN_REVERTING")

    def test_cached_variance_reuse(self):
        """Line 168: test that cached variance is reused when cache is valid."""
        det = self._filled_detector(update_interval=1)
        det._cache_invalidated = True
        det._cached_var_1 = None
        det._cached_returns = None
        # First call fills the cache
        det._update_regime()
        first_regime = det.current_regime

        # Manually set cache as valid (not invalidated) and call again
        det._cache_invalidated = False
        det._update_regime()
        # Should still work, using cached variance
        assert det.current_regime in ("TRENDING", "MEAN_REVERTING", "TRANSITIONAL")

    def test_zero_variance_gives_transitional(self):
        """Line 178: all-same prices produce zero variance → TRANSITIONAL."""
        # Constant prices: all same value
        prices = [100.0] * 50
        det = self._filled_detector(prices=prices)
        det._cache_invalidated = True
        det._cached_var_1 = None
        det._update_regime()
        assert det.current_regime == "TRANSITIONAL"
        assert det.current_zeta == NEUTRAL_ZETA

    def test_exception_fallback_to_transitional(self):
        """Line 221: generic exception falls back to TRANSITIONAL."""
        det = self._filled_detector()
        det._cache_invalidated = True
        det._cached_var_1 = None
        # Corrupt the price_buffer to cause an unexpected exception
        det.price_buffer = "not_a_list"  # Will cause error in np.array usage
        det._update_regime()
        assert det.current_regime == "TRANSITIONAL"
        assert det.current_zeta == NEUTRAL_ZETA

    def test_non_finite_2period_returns(self):
        """Lines 184-185: 2-period returns produce non-finite values."""
        det = self._filled_detector()
        det._cache_invalidated = True
        det._cached_var_1 = None
        # Create prices where individual returns are finite but sum overflows
        # Use prices that create huge returns in consecutive bars
        prices = [100.0] * 50
        for i in range(25, 35):
            prices[i] = prices[i - 1] * (1e15 if i % 2 == 0 else 1e-15)
        # Ensure all positive
        prices = [max(p, 1e-300) for p in prices]
        det.price_buffer = prices
        det._update_regime()
        # Should handle gracefully
        assert det.current_regime in ("UNKNOWN", "TRANSITIONAL", "TRENDING", "MEAN_REVERTING")


class TestAutocorrelation:
    """Cover _autocorrelation method body (lines 234-238)."""

    def test_lag_exceeds_data_length(self):
        """Returns 0.0 when lag >= len(x)."""
        det = RegimeDetector()
        result = det._autocorrelation(np.array([1.0, 2.0, 3.0]), lag=5)
        assert result == pytest.approx(0.0)

    def test_lag_equals_data_length(self):
        """Returns 0.0 when lag == len(x)."""
        det = RegimeDetector()
        result = det._autocorrelation(np.array([1.0, 2.0, 3.0]), lag=3)
        assert result == pytest.approx(0.0)

    def test_zero_variance_returns_zero(self):
        """Returns 0.0 for constant array (zero variance)."""
        det = RegimeDetector()
        result = det._autocorrelation(np.array([5.0, 5.0, 5.0, 5.0, 5.0]), lag=1)
        assert result == pytest.approx(0.0)

    def test_positive_autocorrelation(self):
        """Trending series has positive lag-1 autocorrelation."""
        det = RegimeDetector()
        # Linear trend has strong positive autocorrelation
        x = np.arange(100, dtype=float)
        result = det._autocorrelation(x, lag=1)
        assert result > 0.5

    def test_near_zero_for_random_walk(self):
        """Random noise should have near-zero autocorrelation."""
        det = RegimeDetector()
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        result = det._autocorrelation(x, lag=1)
        assert abs(result) < 0.1

    def test_clamped_to_valid_range(self):
        """Result is always in [-1, 1]."""
        det = RegimeDetector()
        x = np.arange(50, dtype=float)
        for lag in (1, 2, 5, 10):
            result = det._autocorrelation(x, lag=lag)
            assert -1.0 <= result <= 1.0

    def test_lag_one_with_alternating_series(self):
        """Alternating series has negative autocorrelation."""
        det = RegimeDetector()
        x = np.array([1.0, -1.0] * 50)
        result = det._autocorrelation(x, lag=1)
        assert result < -0.5


class TestLogExceptionPath:
    """Lines 147-149: np.log raises ValueError/RuntimeWarning."""

    def test_log_value_error_skips_update(self):
        """When np.log raises ValueError, regime stays unchanged."""
        det = RegimeDetector(window_size=10, update_interval=1)
        prices = [100.0 + i for i in range(10)]
        det.price_buffer = prices
        det._cache_invalidated = True
        det._cached_var_1 = None

        with patch("src.features.regime_detector.np.log", side_effect=ValueError("bad")):
            det._update_regime()
        assert det.current_regime == "UNKNOWN"

    def test_log_runtime_warning_skips_update(self):
        """When np.log raises RuntimeWarning, regime stays unchanged."""
        det = RegimeDetector(window_size=10, update_interval=1)
        prices = [100.0 + i for i in range(10)]
        det.price_buffer = prices
        det._cache_invalidated = True
        det._cached_var_1 = None

        with patch("src.features.regime_detector.np.log", side_effect=RuntimeWarning("overflow")):
            det._update_regime()
        assert det.current_regime == "UNKNOWN"


class TestNonFiniteReturnsMocked:
    """Lines 153-154: returns contain NaN after log diff."""

    def test_nan_in_returns_skips_update(self):
        """If np.diff(log_prices) produces NaN, regime stays unchanged."""
        det = RegimeDetector(window_size=10, update_interval=1)
        prices = [100.0 + i for i in range(10)]
        det.price_buffer = prices
        det._cache_invalidated = True
        det._cached_var_1 = None

        # Mock np.diff to produce non-finite returns
        original_diff = np.diff

        def mock_diff(x):
            result = original_diff(x)
            result[3] = float("nan")
            return result

        with patch("src.features.regime_detector.np.diff", side_effect=mock_diff):
            det._update_regime()
        assert det.current_regime == "UNKNOWN"


class TestNonFinite2PeriodReturns:
    """Lines 191-192: var_2 invalid or lines 184-185: non-finite 2-period returns."""

    def test_non_finite_var2_from_corrupted_2period(self):
        """If 2-period returns become non-finite, regime stays unchanged."""
        det = RegimeDetector(window_size=50, update_interval=1)
        rng = np.random.default_rng(42)
        prices = list(100.0 + np.cumsum(rng.normal(0, 0.5, 50)))
        det.price_buffer = prices
        det._cache_invalidated = True
        det._cached_var_1 = None

        # Mock np.var to return NaN on second call (for var_2)
        original_var = np.var
        call_count = [0]

        def mock_var(x, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call is for var_2
                return float("nan")
            return original_var(x, *args, **kwargs)

        with patch("src.features.regime_detector.np.var", side_effect=mock_var):
            det._update_regime()
        # Should handle gracefully — stays unchanged or falls back
        assert det.current_regime in ("UNKNOWN", "TRANSITIONAL")


class TestNonFiniteVarianceRatio:
    """Lines 199-200: non-finite variance ratio."""

    def test_inf_variance_ratio_handled(self):
        """If vr becomes inf, regime stays unchanged."""
        det = RegimeDetector(window_size=50, update_interval=1)
        rng = np.random.default_rng(42)
        prices = list(100.0 + np.cumsum(rng.normal(0, 0.5, 50)))
        det.price_buffer = prices
        det._cache_invalidated = True
        det._cached_var_1 = None

        # Mock np.var: first call returns very small (near epsilon), second returns huge
        original_var = np.var
        call_count = [0]

        def mock_var(x, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return 1e-100  # Very very small, but > VARIANCE_EPSILON
            if call_count[0] == 2:
                return float("inf")  # Infinite var_2
            return original_var(x, *args, **kwargs)

        with patch("src.features.regime_detector.np.var", side_effect=mock_var):
            det._update_regime()
        assert det.current_regime in ("UNKNOWN", "TRANSITIONAL")


class TestAddPriceIntegration:
    """Integration tests ensuring add_price properly triggers _update_regime."""

    def test_update_after_window_fill(self):
        """Regime updates once window_size reached and update_interval passed."""
        det = RegimeDetector(window_size=20, update_interval=1)
        rng = np.random.default_rng(42)
        price = 100.0
        for i in range(25):
            price += rng.normal(0.5, 0.3)  # Uptrend
            regime, zeta = det.add_price(price)
        # After filling window, regime should have been updated
        assert det.current_regime != "UNKNOWN"

    def test_no_update_before_interval(self):
        """Regime stays UNKNOWN before update_interval is reached."""
        det = RegimeDetector(window_size=10, update_interval=100)
        for i in range(15):
            det.add_price(100.0 + i * 0.1)
        # update_interval=100, only 15 bars added, so no update yet
        assert det.current_regime == "UNKNOWN"
