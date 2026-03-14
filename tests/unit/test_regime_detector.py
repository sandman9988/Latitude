"""Tests for src.features.regime_detector – RegimeDetector (DSP damping-ratio regime classification)."""

import numpy as np
import pytest

from src.features.regime_detector import RegimeDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feed_prices(det: RegimeDetector, prices: list[float]):
    """Feed a list of prices into the detector, return final (regime, zeta)."""
    result = ("UNKNOWN", 1.0)
    for p in prices:
        result = det.add_price(p)
    return result


def _trending_prices(n: int = 60, *, seed: int = 42) -> list[float]:
    """Generate prices with positively-autocorrelated returns (VR > 1).

    Uses an AR(1) process on returns with positive coefficient so that
    consecutive returns tend to persist → variance ratio > 1 → TRENDING.
    """
    rng = np.random.default_rng(seed)
    price = 100_000.0
    prices = []
    prev_ret = 0.0
    for _ in range(n):
        ret = 0.6 * prev_ret + rng.normal(0, 10.0)
        price += ret
        price = max(price, 1.0)
        prices.append(price)
        prev_ret = ret
    return prices


def _mean_reverting_prices(n: int = 60, *, seed: int = 42) -> list[float]:
    """Generate prices with negatively-autocorrelated returns (VR < 1).

    Uses an AR(1) process on returns with negative coefficient so that
    consecutive returns tend to flip sign → variance ratio < 1 → MEAN_REVERTING.
    """
    rng = np.random.default_rng(seed)
    price = 100_000.0
    prices = []
    prev_ret = 0.0
    for _ in range(n):
        ret = -0.6 * prev_ret + rng.normal(0, 10.0)
        price += ret
        price = max(price, 1.0)
        prices.append(price)
        prev_ret = ret
    return prices


def _random_walk_prices(n: int = 60, *, seed: int = 42) -> list[float]:
    """Generate random-walk prices."""
    rng = np.random.default_rng(seed)
    price = 100_000.0
    prices = []
    for _ in range(n):
        price += rng.normal(0, 10.0)
        prices.append(price)
    return prices


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestRegimeDetectorInit:
    def test_default_values(self):
        d = RegimeDetector()
        assert d.current_regime == "UNKNOWN"
        assert d.current_zeta == pytest.approx(1.0)
        assert d.window_size == 50

    def test_custom_window(self):
        d = RegimeDetector(window_size=30, update_interval=3)
        assert d.window_size == 30
        assert d.update_interval == 3

    def test_volatility_scaling(self):
        d = RegimeDetector(instrument_volatility=2.0)
        assert d.trending_threshold > 0.7  # Scaled up


# ---------------------------------------------------------------------------
# add_price
# ---------------------------------------------------------------------------


class TestAddPrice:
    def test_invalid_price_skipped(self):
        d = RegimeDetector()
        regime, zeta = d.add_price(-1.0)
        assert regime == "UNKNOWN"

    def test_none_price_skipped(self):
        d = RegimeDetector()
        regime, zeta = d.add_price(None)
        assert regime == "UNKNOWN"

    def test_buffer_grows(self):
        d = RegimeDetector(window_size=10)
        for i in range(1, 6):
            d.add_price(100_000.0 + i)
        assert len(d.price_buffer) == 5

    def test_buffer_capped_at_window(self):
        d = RegimeDetector(window_size=10)
        for i in range(1, 20):
            d.add_price(100_000.0 + i)
        assert len(d.price_buffer) == 10

    def test_regime_stays_unknown_until_window_full(self):
        d = RegimeDetector(window_size=50, update_interval=5)
        for i in range(1, 30):
            d.add_price(100_000.0 + i)
        assert d.current_regime == "UNKNOWN"


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


class TestRegimeClassification:
    def test_trending_market(self):
        d = RegimeDetector(window_size=50, update_interval=5)
        regime, zeta = _feed_prices(d, _trending_prices(60))
        assert regime == "TRENDING"

    def test_mean_reverting_market(self):
        d = RegimeDetector(window_size=50, update_interval=5)
        regime, zeta = _feed_prices(d, _mean_reverting_prices(80))
        assert regime == "MEAN_REVERTING"

    def test_random_walk_transitional(self):
        d = RegimeDetector(window_size=50, update_interval=5)
        regime, zeta = _feed_prices(d, _random_walk_prices(60))
        # Random walk → near ζ≈1.0 → TRANSITIONAL
        assert regime in ("TRANSITIONAL", "TRENDING", "MEAN_REVERTING")

    def test_zero_variance_gives_transitional(self):
        # window_size must be > MIN_RETURNS_REQUIRED(10) so that
        # diff(prices) yields enough returns for _update_regime()
        d = RegimeDetector(window_size=12, update_interval=1)
        regime, zeta = _feed_prices(d, [100_000.0] * 20)
        assert regime == "TRANSITIONAL"


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------


class TestRegimeHelpers:
    def test_get_regime_multiplier_trending(self):
        d = RegimeDetector()
        d.current_regime = "TRENDING"
        assert d.get_regime_multiplier() == pytest.approx(1.3)

    def test_get_regime_multiplier_mean_reverting(self):
        d = RegimeDetector()
        d.current_regime = "MEAN_REVERTING"
        assert d.get_regime_multiplier() == pytest.approx(0.7)

    def test_get_regime_multiplier_transitional(self):
        d = RegimeDetector()
        d.current_regime = "TRANSITIONAL"
        assert d.get_regime_multiplier() == pytest.approx(1.0)

    def test_get_trigger_threshold_trending(self):
        d = RegimeDetector()
        d.current_regime = "TRENDING"
        assert d.get_trigger_threshold_adjustment() == pytest.approx(-0.15)

    def test_get_trigger_threshold_mean_reverting(self):
        d = RegimeDetector()
        d.current_regime = "MEAN_REVERTING"
        assert d.get_trigger_threshold_adjustment() == pytest.approx(0.15)

    def test_get_regime_info_keys(self):
        d = RegimeDetector()
        info = d.get_regime_info()
        for key in (
            "regime",
            "damping_ratio",
            "runway_multiplier",
            "trigger_adjustment",
            "buffer_size",
            "bars_since_update",
        ):
            assert key in info


# ---------------------------------------------------------------------------
# Autocorrelation helper
# ---------------------------------------------------------------------------


class TestAutocorrelation:
    def test_lag_exceeds_data(self):
        d = RegimeDetector()
        x = np.array([1.0, 2.0, 3.0])
        assert d._autocorrelation(x, lag=5) == pytest.approx(0.0)

    def test_zero_variance(self):
        d = RegimeDetector()
        x = np.array([5.0, 5.0, 5.0, 5.0])
        assert d._autocorrelation(x, lag=1) == pytest.approx(0.0)

    def test_known_autocorrelation(self):
        d = RegimeDetector()
        # Strong positive autocorrelation for trending data
        x = np.arange(1.0, 51.0)  # linear upward
        ac = d._autocorrelation(x, lag=1)
        assert ac > 0.9  # Should be very high
