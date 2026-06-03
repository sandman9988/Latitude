"""Tests for src.agents.runway_forecaster - ATR-anchored runway forecaster."""

import numpy as np
import pytest

from src.agents.runway_forecaster import (
    RunwayForecaster,
    build_feature_matrix,
    extract_features,
)


def _synth_series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
    highs = closes + np.abs(rng.normal(0, 0.5, n))
    lows = closes - np.abs(rng.normal(0, 0.5, n))
    opens = closes + rng.normal(0, 0.2, n)
    atr = np.full(n, 0.7)
    return opens, highs, lows, closes, atr


class TestExtractFeatures:
    def test_insufficient_history(self):
        o, h, l, c, atr = _synth_series()
        assert extract_features(o, h, l, c, atr, idx=3) is None

    def test_feature_vector_shape_and_side(self):
        o, h, l, c, atr = _synth_series()
        f_long = extract_features(o, h, l, c, atr, idx=100, side=1)
        f_short = extract_features(o, h, l, c, atr, idx=100, side=-1)
        assert f_long.shape == (9,)
        assert f_long[0] == 1.0
        assert f_long[1] == 1.0
        assert f_short[1] == -1.0


class TestBuildFeatureMatrix:
    def test_matrix_alignment(self):
        o, h, l, c, atr = _synth_series()
        fm, idxs = build_feature_matrix(o, h, l, c, atr, side=1)
        assert fm.shape[0] == idxs.shape[0]
        assert fm.shape[1] == 9
        assert np.all(idxs >= 20)


class TestForecasterFit:
    def _labels(self, idxs, seed=1):
        rng = np.random.default_rng(seed)
        return np.abs(rng.normal(2.0, 1.0, idxs.shape[0]))

    def test_fit_and_predict_quantiles_monotone(self):
        o, h, l, c, atr = _synth_series()
        fm, idxs = build_feature_matrix(o, h, l, c, atr, side=1)
        labels = self._labels(idxs)
        model = RunwayForecaster()
        model.fit(fm, labels)
        assert model.fitted
        q = model.predict_quantiles(fm[0])
        assert q.shape == (3,)
        assert q[0] <= q[1] <= q[2]
        assert np.all(q >= 0)

    def test_predict_runway_scales_with_atr(self):
        o, h, l, c, atr = _synth_series()
        fm, idxs = build_feature_matrix(o, h, l, c, atr, side=1)
        model = RunwayForecaster()
        model.fit(fm, self._labels(idxs))
        r1 = model.predict_runway(fm[0], atr=1.0)
        r2 = model.predict_runway(fm[0], atr=2.0)
        assert r2 == pytest.approx(2.0 * r1, rel=1e-6)

    def test_empty_fit_raises(self):
        model = RunwayForecaster()
        with pytest.raises(ValueError):
            model.fit(np.empty((0, 9)), np.empty(0))

    def test_prob_exceed_bounds(self):
        o, h, l, c, atr = _synth_series()
        fm, idxs = build_feature_matrix(o, h, l, c, atr, side=1)
        model = RunwayForecaster()
        model.fit(fm, self._labels(idxs))
        p = model.prob_exceed(fm[0], atr=0.7, threshold=0.5)
        assert 0.0 <= p <= 1.0
        assert model.prob_exceed(fm[0], atr=0.0, threshold=0.5) == 0.0

    def test_save_load_roundtrip(self, tmp_path):
        o, h, l, c, atr = _synth_series()
        fm, idxs = build_feature_matrix(o, h, l, c, atr, side=1)
        model = RunwayForecaster()
        model.fit(fm, self._labels(idxs))
        path = tmp_path / "rwf.json"
        model.save(path)
        loaded = RunwayForecaster.load(path)
        q1 = model.predict_quantiles(fm[5])
        q2 = loaded.predict_quantiles(fm[5])
        np.testing.assert_allclose(q1, q2, rtol=1e-9)
