"""Tests for src.features.runway_labels - forward favorable-excursion labels."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.features.runway_labels import (
    DEFAULT_HORIZON_BARS,
    build_labels,
    forward_favorable_excursion,
    horizon_for_timeframe,
    wilder_atr,
)


def _make_bars(closes, spread=1.0, highs=None, lows=None, opens=None):
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    bars = []
    for i, c in enumerate(closes):
        o = opens[i] if opens is not None else c
        h = highs[i] if highs is not None else c
        low = lows[i] if lows is not None else c
        bars.append((t0 + timedelta(minutes=i), float(o), float(h), float(low), float(c), spread))
    return bars


class TestHorizon:
    def test_known_timeframes(self):
        assert horizon_for_timeframe(1) == DEFAULT_HORIZON_BARS[1]
        assert horizon_for_timeframe(240) == DEFAULT_HORIZON_BARS[240]

    def test_unknown_timeframe_fallback(self):
        assert horizon_for_timeframe(7) == 24


class TestWilderAtr:
    def test_empty(self):
        assert wilder_atr(np.array([]), np.array([]), np.array([])).size == 0

    def test_constant_range(self):
        highs = np.full(50, 11.0)
        lows = np.full(50, 10.0)
        closes = np.full(50, 10.5)
        atr = wilder_atr(highs, lows, closes, period=14)
        assert atr.shape == (50,)
        assert atr[-1] == pytest.approx(1.0, abs=1e-6)

    def test_length_matches(self):
        n = 5
        atr = wilder_atr(np.arange(n) + 2.0, np.arange(n) + 1.0, np.arange(n) + 1.5, period=14)
        assert atr.shape == (n,)


class TestForwardExcursion:
    def test_long_excursion(self):
        highs = np.array([1.0, 2.0, 3.0, 5.0, 1.0])
        lows = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        closes = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        out = forward_favorable_excursion(highs, lows, closes, horizon=2, side="long")
        assert out[0] == pytest.approx(2.0)
        assert out[1] == pytest.approx(4.0)
        assert np.isnan(out[3])

    def test_short_excursion(self):
        highs = np.array([5.0, 5.0, 5.0, 5.0])
        lows = np.array([5.0, 3.0, 1.0, 5.0])
        closes = np.array([5.0, 5.0, 5.0, 5.0])
        out = forward_favorable_excursion(highs, lows, closes, horizon=2, side="short")
        assert out[0] == pytest.approx(4.0)

    def test_no_lookahead_tail_nan(self):
        highs = np.arange(10) + 1.0
        out = forward_favorable_excursion(highs, highs - 0.5, highs - 0.25, horizon=3, side="long")
        assert np.all(np.isnan(out[-3:]))


class TestBuildLabels:
    def test_shapes_and_validity(self):
        rng = np.random.default_rng(0)
        closes = 100 + np.cumsum(rng.normal(0, 1, 200))
        highs = closes + np.abs(rng.normal(0, 1, 200))
        lows = closes - np.abs(rng.normal(0, 1, 200))
        bars = _make_bars(closes, highs=highs, lows=lows)
        lab = build_labels(bars, timeframe_minutes=30)
        n = len(bars)
        for key in ("close", "atr", "mfe_long", "mfe_short", "label_long", "label_short", "valid"):
            assert lab[key].shape == (n,)
        assert lab["valid"].dtype == bool
        valid = lab["valid"]
        assert np.all(lab["label_long"][valid] >= 0)
        assert np.all(np.isfinite(lab["label_long"][valid]))

    def test_empty_bars(self):
        lab = build_labels([], timeframe_minutes=30)
        assert lab["close"].size == 0
