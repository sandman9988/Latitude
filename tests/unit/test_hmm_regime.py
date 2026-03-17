"""Tests for src.features.hmm_regime – HMMRegimeDetector."""

import numpy as np
import pytest

from src.features.hmm_regime import (
    HMM_MIN_OBSERVATIONS,
    HMM_N_STATES,
    HMMRegimeDetector,
    RUNWAY_MULT_MEAN_REVERTING,
    RUNWAY_MULT_NEUTRAL,
    RUNWAY_MULT_TRENDING,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed_prices(det: HMMRegimeDetector, prices: list[float]) -> None:
    for p in prices:
        det.add_price(p)


def _trending_prices(n: int = 80, *, seed: int = 42) -> list[float]:
    """AR(1) with positive coefficient → trending."""
    rng = np.random.default_rng(seed)
    price = 100_000.0
    prices = []
    prev_ret = 0.0
    for _ in range(n):
        ret = 0.7 * prev_ret + rng.normal(0, 8.0)
        price += ret
        price = max(price, 1.0)
        prices.append(price)
        prev_ret = ret
    return prices


def _mean_reverting_prices(n: int = 80, *, seed: int = 42) -> list[float]:
    """AR(1) with negative coefficient → mean-reverting."""
    rng = np.random.default_rng(seed)
    price = 100_000.0
    prices = []
    prev_ret = 0.0
    for _ in range(n):
        ret = -0.7 * prev_ret + rng.normal(0, 8.0)
        price += ret
        price = max(price, 1.0)
        prices.append(price)
        prev_ret = ret
    return prices


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestHMMInit:
    def test_inherits_regime_detector(self):
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        assert hasattr(det, "current_regime")
        assert hasattr(det, "current_zeta")
        assert det._hmm_fitted is False

    def test_initial_probabilities(self):
        det = HMMRegimeDetector()
        probs = det.get_regime_probabilities()
        assert len(probs) == HMM_N_STATES
        assert np.sum(probs) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# HMM fitting
# ---------------------------------------------------------------------------

class TestHMMFitting:
    def test_not_fitted_with_few_observations(self):
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        for p in _trending_prices(HMM_MIN_OBSERVATIONS - 5):
            det.add_price(p)
        assert det._hmm_fitted is False

    def test_fits_with_enough_observations(self):
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        _feed_prices(det, _trending_prices(80))
        assert det._hmm_fitted is True

    def test_posteriors_sum_to_one(self):
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        _feed_prices(det, _trending_prices(80))
        probs = det.get_regime_probabilities()
        assert np.sum(probs) == pytest.approx(1.0, abs=1e-6)
        assert all(p >= 0.0 for p in probs)


# ---------------------------------------------------------------------------
# Blended multiplier
# ---------------------------------------------------------------------------

class TestBlendedMultiplier:
    def test_fallback_before_fit(self):
        """Before HMM is fitted, blended multiplier should match VR regime."""
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        blended = det.get_blended_runway_multiplier()
        discrete = det.get_regime_multiplier()
        assert blended == discrete

    def test_multiplier_in_valid_range(self):
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        _feed_prices(det, _trending_prices(80))
        blended = det.get_blended_runway_multiplier()
        assert RUNWAY_MULT_MEAN_REVERTING <= blended <= RUNWAY_MULT_TRENDING

    def test_blended_is_weighted_sum(self):
        """Multiplier should be the dot product of probs and multiplier map."""
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        _feed_prices(det, _trending_prices(80))
        probs = det.get_regime_probabilities()
        mults = np.array([RUNWAY_MULT_TRENDING, RUNWAY_MULT_MEAN_REVERTING, RUNWAY_MULT_NEUTRAL])
        expected = float(np.dot(probs, mults))
        assert det.get_blended_runway_multiplier() == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# get_regime_info()
# ---------------------------------------------------------------------------

class TestRegimeInfo:
    def test_info_before_fit(self):
        det = HMMRegimeDetector()
        info = det.get_regime_info()
        assert info["hmm_fitted"] is False
        assert "hmm_probs" not in info

    def test_info_after_fit(self):
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        _feed_prices(det, _trending_prices(80))
        info = det.get_regime_info()
        assert info["hmm_fitted"] is True
        assert "hmm_probs" in info
        assert "hmm_blended_multiplier" in info
        probs = info["hmm_probs"]
        assert probs["trending"] + probs["mean_reverting"] + probs["neutral"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Base class compatibility
# ---------------------------------------------------------------------------

class TestBaseCompatibility:
    def test_vr_regime_still_works(self):
        """VR-based regime detection should still function normally."""
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        _feed_prices(det, _trending_prices(80))
        assert det.current_regime in ("TRENDING", "MEAN_REVERTING", "TRANSITIONAL", "UNKNOWN")
        assert 0.1 <= det.current_zeta <= 2.0

    def test_get_trigger_threshold_adjustment(self):
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        _feed_prices(det, _trending_prices(80))
        adj = det.get_trigger_threshold_adjustment()
        assert -0.5 <= adj <= 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestHMMEdgeCases:
    def test_invalid_prices_handled(self):
        det = HMMRegimeDetector()
        det.add_price(0.0)  # Invalid
        det.add_price(-1.0)  # Invalid
        assert det._hmm_fitted is False

    def test_duplicate_prices_no_crash(self):
        """Constant prices → zero variance log-returns."""
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        for _ in range(80):
            det.add_price(100.0)
        # Should not crash; HMM may or may not fit with zero-variance data
        probs = det.get_regime_probabilities()
        assert len(probs) == HMM_N_STATES

    def test_mixed_regime_prices(self):
        """Feeding trending then mean-reverting data should update posteriors."""
        det = HMMRegimeDetector(window_size=50, update_interval=5)
        _feed_prices(det, _trending_prices(60))
        probs_trending = det.get_regime_probabilities().copy()
        _feed_prices(det, _mean_reverting_prices(60))
        probs_mr = det.get_regime_probabilities()
        # The posteriors should have changed
        assert not np.allclose(probs_trending, probs_mr, atol=0.05)
