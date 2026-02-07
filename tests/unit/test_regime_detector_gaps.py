"""Gap tests for src.features.regime_detector – _autocorrelation edge cases.

Targets uncovered production lines:
- Line 178: dead `len(x)` call (standalone statement, unused return)
- Lines 184-185: zero variance → return 0.0
- Lines 199-200: covariance / variance calculation (autocorrelation)

Also tests: get_regime_multiplier, get_trigger_threshold_adjustment, get_regime_info
"""

import numpy as np
import pytest

from src.features.regime_detector import (
    RegimeDetector,
    RUNWAY_MULT_TRENDING,
    RUNWAY_MULT_MEAN_REVERTING,
    RUNWAY_MULT_NEUTRAL,
    TRIGGER_ADJUST_TRENDING,
    TRIGGER_ADJUST_MEAN_REVERTING,
    TRIGGER_ADJUST_NEUTRAL,
)


# ---------------------------------------------------------------------------
# _autocorrelation edge cases
# ---------------------------------------------------------------------------
class TestAutocorrelation:
    def test_short_series_returns_zero(self):
        """Series shorter than lag → returns 0."""
        det = RegimeDetector()
        result = det._autocorrelation(np.array([1.0, 2.0]), lag=5)
        assert result == pytest.approx(0.0)

    def test_constant_series_returns_zero(self):
        """Constant series has zero variance → returns 0."""
        det = RegimeDetector()
        result = det._autocorrelation(np.array([5.0] * 20), lag=1)
        assert result == pytest.approx(0.0)

    def test_positive_autocorrelation(self):
        """Trending data should show positive autocorrelation."""
        det = RegimeDetector()
        # Linear uptrend → strong positive autocorrelation at lag 1
        x = np.arange(50, dtype=float)
        result = det._autocorrelation(x, lag=1)
        assert result > 0.9

    def test_negative_autocorrelation(self):
        """Oscillating data should show negative autocorrelation."""
        det = RegimeDetector()
        # Alternating pattern
        x = np.array([1.0, -1.0] * 25)
        result = det._autocorrelation(x, lag=1)
        assert result < -0.9

    def test_autocorrelation_clamped_to_valid_range(self):
        """Result should always be in [-1, 1]."""
        det = RegimeDetector()
        x = np.random.default_rng(42).standard_normal(100)
        for lag in [1, 2, 5, 10]:
            result = det._autocorrelation(x, lag=lag)
            assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# get_regime_multiplier
# ---------------------------------------------------------------------------
class TestRegimeMultiplier:
    def test_trending_multiplier(self):
        det = RegimeDetector()
        det.current_regime = "TRENDING"
        assert det.get_regime_multiplier() == RUNWAY_MULT_TRENDING

    def test_mean_reverting_multiplier(self):
        det = RegimeDetector()
        det.current_regime = "MEAN_REVERTING"
        assert det.get_regime_multiplier() == RUNWAY_MULT_MEAN_REVERTING

    def test_transitional_multiplier(self):
        det = RegimeDetector()
        det.current_regime = "TRANSITIONAL"
        assert det.get_regime_multiplier() == RUNWAY_MULT_NEUTRAL

    def test_unknown_multiplier(self):
        det = RegimeDetector()
        det.current_regime = "UNKNOWN"
        assert det.get_regime_multiplier() == RUNWAY_MULT_NEUTRAL


# ---------------------------------------------------------------------------
# get_trigger_threshold_adjustment
# ---------------------------------------------------------------------------
class TestTriggerThresholdAdjustment:
    def test_trending_adjustment(self):
        det = RegimeDetector()
        det.current_regime = "TRENDING"
        assert det.get_trigger_threshold_adjustment() == TRIGGER_ADJUST_TRENDING

    def test_mean_reverting_adjustment(self):
        det = RegimeDetector()
        det.current_regime = "MEAN_REVERTING"
        assert det.get_trigger_threshold_adjustment() == TRIGGER_ADJUST_MEAN_REVERTING

    def test_transitional_adjustment(self):
        det = RegimeDetector()
        det.current_regime = "TRANSITIONAL"
        assert det.get_trigger_threshold_adjustment() == TRIGGER_ADJUST_NEUTRAL


# ---------------------------------------------------------------------------
# get_regime_info
# ---------------------------------------------------------------------------
class TestRegimeInfo:
    def test_info_contains_all_fields(self):
        det = RegimeDetector()
        info = det.get_regime_info()
        assert "regime" in info
        assert "damping_ratio" in info
        assert "runway_multiplier" in info
        assert "trigger_adjustment" in info
        assert "buffer_size" in info
        assert "bars_since_update" in info

    def test_info_after_trending_detection(self):
        """Feed trending data and verify info reflects it."""
        det = RegimeDetector(window_size=50, update_interval=1)
        rng = np.random.default_rng(42)

        price = 100.0
        for _ in range(60):
            price += 0.5 + rng.normal(0, 0.1)  # Strong uptrend
            det.add_price(price)

        info = det.get_regime_info()
        assert info["buffer_size"] == 50  # Window capped


# ---------------------------------------------------------------------------
# add_price edge cases
# ---------------------------------------------------------------------------
class TestAddPriceEdgeCases:
    def test_none_price_skipped(self):
        det = RegimeDetector()
        regime, zeta = det.add_price(None)
        assert regime == "UNKNOWN"
        assert len(det.price_buffer) == 0

    def test_zero_price_skipped(self):
        det = RegimeDetector()
        regime, zeta = det.add_price(0.0)
        assert regime == "UNKNOWN"

    def test_negative_price_skipped(self):
        det = RegimeDetector()
        regime, zeta = det.add_price(-100.0)
        assert regime == "UNKNOWN"

    def test_buffer_rolls_at_window_size(self):
        det = RegimeDetector(window_size=10)
        for i in range(1, 20):
            det.add_price(float(100 + i))
        assert len(det.price_buffer) == 10

    def test_regime_update_only_at_interval(self):
        """Regime only recalculates every update_interval bars."""
        det = RegimeDetector(window_size=10, update_interval=5)
        for i in range(1, 12):
            det.add_price(float(100 + i))
        # With 11 bars added, buffer has 10 bars, should have updated once
        # (at bar 10 or 15 depending on when interval aligns)
        assert det.bars_since_update < 5
