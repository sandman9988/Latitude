"""Tests for src.risk.path_geometry – PathGeometry (entry trigger features)."""

from collections import deque

import numpy as np
import pytest

from src.risk.path_geometry import PathGeometry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bars(*closes: float) -> deque:
    """Build a deque of (t, o, h, l, c) tuples from close prices."""
    return deque([(i, c, c + 1, c - 1, c) for i, c in enumerate(closes)])


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestPathGeometryInit:
    def test_defaults(self):
        pg = PathGeometry()
        assert pg.last["efficiency"] == pytest.approx(0.0)
        assert pg.last["feasibility"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_returns_last_when_insufficient_bars(self):
        pg = PathGeometry()
        result = pg.update(deque(), sigma=0.01)
        assert result == pg.last  # Unchanged

    def test_returns_last_when_sigma_zero(self):
        pg = PathGeometry()
        bars = _bars(100, 101, 102)
        result = pg.update(bars, sigma=0.0)
        assert result["efficiency"] == pytest.approx(0.0)  # Not updated

    def test_straight_uptrend_high_efficiency(self):
        pg = PathGeometry()
        bars = _bars(100.0, 102.0, 104.0)  # Straight up
        result = pg.update(bars, sigma=0.01)
        assert result["efficiency"] == pytest.approx(1.0)

    def test_zigzag_low_efficiency(self):
        pg = PathGeometry()
        bars = _bars(100.0, 110.0, 100.0)  # Up then back
        result = pg.update(bars, sigma=0.01)
        assert result["efficiency"] == pytest.approx(0.0, abs=0.01)

    def test_runway_decreases_with_high_sigma(self):
        pg = PathGeometry()
        bars = _bars(100.0, 101.0, 102.0)
        r1 = pg.update(bars, sigma=0.001)
        pg2 = PathGeometry()
        r2 = pg2.update(bars, sigma=0.1)
        assert r1["runway"] > r2["runway"]

    def test_feasibility_in_range(self):
        pg = PathGeometry()
        bars = _bars(100.0, 101.0, 102.0)
        result = pg.update(bars, sigma=0.01)
        assert 0.0 <= result["feasibility"] <= 1.0

    def test_gamma_and_jerk_present(self):
        pg = PathGeometry()
        bars = _bars(100.0, 101.0, 103.0)  # Accelerating
        result = pg.update(bars, sigma=0.01)
        assert "gamma" in result
        assert "jerk" in result

    def test_invalid_prices_returns_last(self):
        pg = PathGeometry()
        bars = _bars(0.0, 101.0, 102.0)  # c0 = 0 → invalid
        result = pg.update(bars, sigma=0.01)
        assert result == pg.last

    def test_multiple_updates(self):
        pg = PathGeometry()
        bars = _bars(100.0, 101.0, 102.0)
        pg.update(bars, sigma=0.01)
        # Second update — jerk should reflect change from previous gamma
        bars2 = _bars(101.0, 102.0, 105.0)
        result2 = pg.update(bars2, sigma=0.01)
        assert result2["jerk"] != 0.0  # Changed from prev gamma

    def test_more_than_3_bars(self):
        """Only last 3 bars used."""
        pg = PathGeometry()
        bars = _bars(90.0, 95.0, 100.0, 101.0, 102.0)
        result = pg.update(bars, sigma=0.01)
        assert result["efficiency"] > 0


# ---------------------------------------------------------------------------
# get_feature_vector()
# ---------------------------------------------------------------------------

class TestGetFeatureVector:
    def test_shape(self):
        pg = PathGeometry()
        vec = pg.get_feature_vector()
        assert vec.shape == (5,)
        assert vec.dtype == np.float32

    def test_matches_last(self):
        pg = PathGeometry()
        bars = _bars(100.0, 102.0, 104.0)
        pg.update(bars, sigma=0.01)
        vec = pg.get_feature_vector()
        assert vec[0] == pytest.approx(pg.last["efficiency"])
        assert vec[4] == pytest.approx(pg.last["feasibility"])


# ---------------------------------------------------------------------------
# _clamp01
# ---------------------------------------------------------------------------

class TestClamp01:
    def test_within_range(self):
        assert PathGeometry._clamp01(0.5) == pytest.approx(0.5)

    def test_below(self):
        assert PathGeometry._clamp01(-0.3) == pytest.approx(0.0)

    def test_above(self):
        assert PathGeometry._clamp01(1.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Multi-horizon vol ratio (Enhancement B)
# ---------------------------------------------------------------------------

class TestVolRatio:
    def test_no_sigma_long_falls_back(self):
        """When sigma_long=0, runway should match the original formula."""
        pg = PathGeometry()
        bars = _bars(100.0, 101.0, 102.0)
        result = pg.update(bars, sigma=0.01, sigma_long=0.0)
        expected = 1.0 / (1.0 + 50.0 * 0.01)
        assert result["runway"] == pytest.approx(expected, rel=1e-6)

    def test_expanding_vol_reduces_runway(self):
        """sigma > sigma_long (expanding vol) should reduce runway."""
        pg1, pg2 = PathGeometry(), PathGeometry()
        bars = _bars(100.0, 101.0, 102.0)
        r_neutral = pg1.update(bars, sigma=0.01, sigma_long=0.01)
        r_expanding = pg2.update(bars, sigma=0.02, sigma_long=0.01)
        assert r_expanding["runway"] < r_neutral["runway"]

    def test_contracting_vol_increases_runway(self):
        """sigma < sigma_long (contracting vol) should increase runway."""
        pg1, pg2 = PathGeometry(), PathGeometry()
        bars = _bars(100.0, 101.0, 102.0)
        r_neutral = pg1.update(bars, sigma=0.01, sigma_long=0.01)
        r_contracting = pg2.update(bars, sigma=0.005, sigma_long=0.01)
        assert r_contracting["runway"] > r_neutral["runway"]

    def test_vol_ratio_adjustment_clamped(self):
        """Extreme vol ratios should be clamped, runway stays in valid range."""
        pg = PathGeometry()
        bars = _bars(100.0, 101.0, 102.0)
        # Extreme ratio: sigma 10x of sigma_long
        result = pg.update(bars, sigma=0.1, sigma_long=0.01)
        assert 0.0 < result["runway"] <= 1.0
        assert 0.0 <= result["feasibility"] <= 1.0
