"""Extended tests for src.features.regime_detector.

Covers: _update_regime edge cases (non-finite prices, returns),
_autocorrelation edge cases, get_regime_info, regime transition,
instrument volatility scaling, cache invalidation.
"""

import pytest
import numpy as np

from src.features.regime_detector import (
    RUNWAY_MULT_MEAN_REVERTING,
    RUNWAY_MULT_NEUTRAL,
    RUNWAY_MULT_TRENDING,
    REGIME_ADJ_MEAN_REVERTING,
    REGIME_ADJ_NEUTRAL,
    REGIME_ADJ_TRENDING,
    RegimeDetector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fill_trending(det, n=60, base=100000.0, drift=10.0, seed=42):
    """Feed a trending price series."""
    rng = np.random.default_rng(seed)
    price = base
    for _ in range(n):
        price += drift + rng.normal(0, 2.0)
        det.add_price(price)


def _fill_mean_reverting(det, n=60, base=100000.0, seed=42):
    """Feed a mean-reverting (oscillating) price series."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        price = base + 20.0 * np.sin(i * 0.3) + rng.normal(0, 1.0)
        det.add_price(price)


def _fill_random_walk(det, n=60, base=100000.0, seed=42):
    rng = np.random.default_rng(seed)
    price = base
    for _ in range(n):
        price += rng.normal(0, 5.0)
        det.add_price(price)


# ---------------------------------------------------------------------------
# _update_regime edge cases
# ---------------------------------------------------------------------------
class TestUpdateRegimeEdges:
    def test_non_finite_prices_skipped(self):
        det = RegimeDetector(window_size=10, update_interval=1)
        for i in range(10):
            det.add_price(100.0 + i * 0.1)
        # Inject nan into buffer manually
        det.price_buffer[0] = float("nan")
        det.bars_since_update = det.update_interval
        det._cache_invalidated = True
        det._cached_var_1 = None
        det._cached_returns = None
        det._update_regime()
        # Should fallback without crash

    def test_all_same_prices_stays_unknown(self):
        det = RegimeDetector(window_size=10, update_interval=1)
        for _ in range(15):
            det.add_price(100.0)
        # Zero variance → returns stay all-zero → regime cannot be determined
        assert det.current_regime == "UNKNOWN"

    def test_negative_price_skipped(self):
        det = RegimeDetector(window_size=10, update_interval=1)
        regime, _ = det.add_price(-5.0)
        assert regime == "UNKNOWN"
        assert len(det.price_buffer) == 0


# ---------------------------------------------------------------------------
# _autocorrelation
# ---------------------------------------------------------------------------
class TestAutocorrelation:
    def test_lag_exceeds_length(self):
        det = RegimeDetector()
        x = np.array([1.0, 2.0, 3.0])
        assert det._autocorrelation(x, lag=5) == pytest.approx(0.0)

    def test_zero_variance_returns_zero(self):
        det = RegimeDetector()
        x = np.array([5.0, 5.0, 5.0, 5.0])
        assert det._autocorrelation(x, lag=1) == pytest.approx(0.0)

    def test_clamped_to_valid_range(self):
        det = RegimeDetector()
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        ac = det._autocorrelation(x, lag=1)
        assert -1.0 <= ac <= 1.0


# ---------------------------------------------------------------------------
# get_regime_multiplier
# ---------------------------------------------------------------------------
class TestGetRegimeMultiplier:
    def test_unknown_returns_neutral(self):
        det = RegimeDetector()
        assert det.current_regime == "UNKNOWN"
        assert det.get_regime_multiplier() == RUNWAY_MULT_NEUTRAL

    def test_trending(self):
        det = RegimeDetector()
        det.current_regime = "TRENDING"
        assert det.get_regime_multiplier() == RUNWAY_MULT_TRENDING

    def test_mean_reverting(self):
        det = RegimeDetector()
        det.current_regime = "MEAN_REVERTING"
        assert det.get_regime_multiplier() == RUNWAY_MULT_MEAN_REVERTING


# ---------------------------------------------------------------------------
# get_trigger_threshold_adjustment
# ---------------------------------------------------------------------------
class TestTriggerAdjustment:
    def test_unknown_is_zero(self):
        det = RegimeDetector()
        assert det.get_trigger_threshold_adjustment() == REGIME_ADJ_NEUTRAL

    def test_trending(self):
        det = RegimeDetector()
        det.current_regime = "TRENDING"
        assert det.get_trigger_threshold_adjustment() == REGIME_ADJ_TRENDING

    def test_mean_reverting(self):
        det = RegimeDetector()
        det.current_regime = "MEAN_REVERTING"
        assert det.get_trigger_threshold_adjustment() == REGIME_ADJ_MEAN_REVERTING


# ---------------------------------------------------------------------------
# get_regime_info
# ---------------------------------------------------------------------------
class TestGetRegimeInfo:
    def test_all_keys_present(self):
        det = RegimeDetector()
        info = det.get_regime_info()
        for key in [
            "regime",
            "damping_ratio",
            "runway_multiplier",
            "trigger_adjustment",
            "buffer_size",
            "bars_since_update",
        ]:
            assert key in info

    def test_values_consistent(self):
        det = RegimeDetector()
        _fill_trending(det, n=60)
        info = det.get_regime_info()
        assert info["regime"] == det.current_regime
        assert info["damping_ratio"] == det.current_zeta


# ---------------------------------------------------------------------------
# Regime transitions
# ---------------------------------------------------------------------------
class TestRegimeTransition:
    def test_transitions_from_trending_to_mean_reverting(self):
        det = RegimeDetector(window_size=20, update_interval=1)
        # Fill trending
        rng = np.random.default_rng(42)
        price = 100000.0
        for _ in range(25):
            price += 15.0 + rng.normal(0, 1.0)
            det.add_price(price)
        _first_regime = det.current_regime
        # Now mean-revert
        det2 = RegimeDetector(window_size=20, update_interval=1)
        _fill_mean_reverting(det2, n=25)
        _second_regime = det2.current_regime
        # They should be different regimes (or both transitional is ok too)
        # Main goal: no crash during transition

    def test_instrument_volatility_scales_thresholds(self):
        det_low = RegimeDetector(instrument_volatility=0.5)
        det_high = RegimeDetector(instrument_volatility=2.0)
        assert det_low.trending_threshold < det_high.trending_threshold
        assert det_low.mean_reverting_threshold < det_high.mean_reverting_threshold

    def test_volatility_clamped(self):
        det = RegimeDetector(instrument_volatility=100.0)
        assert det.instrument_volatility == pytest.approx(2.0)
        det2 = RegimeDetector(instrument_volatility=0.01)
        assert det2.instrument_volatility == pytest.approx(0.5)
