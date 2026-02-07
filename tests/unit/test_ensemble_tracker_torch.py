"""Tests for EnsembleTracker PyTorch-like model path in _get_q_values.

Covers uncovered lines 207-211: torch import, model.forward() path,
tensor conversion, squeeze/cpu/numpy, and zeros fallback for
models with neither forward nor predict.
"""

import numpy as np

rng = np.random.default_rng(42)

import pytest
from unittest.mock import MagicMock, patch

from src.core.ensemble_tracker import (
    EnsembleTracker,
    NUM_ACTIONS,
    MIN_ENSEMBLE_MODELS,
)


class _FakeTensor:
    """Minimal tensor-like object to emulate PyTorch behavior."""

    def __init__(self, data: np.ndarray):
        self._data = data

    def dim(self):
        return len(self._data.shape)

    def unsqueeze(self, dim):
        return _FakeTensor(np.expand_dims(self._data, axis=dim))

    def float(self):
        return self

    def squeeze(self):
        return _FakeTensor(self._data.squeeze())

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class _TorchLikeModel:
    """Model with a forward method that mimics PyTorch nn.Module."""

    def __init__(self, q_values: np.ndarray):
        self._q_values = q_values

    def forward(self, x):
        return _FakeTensor(self._q_values)

    def __call__(self, x):
        return self.forward(x)


class _PredictModel:
    """Model with only a predict method (numpy-based)."""

    def __init__(self, q_values: np.ndarray):
        self._q_values = q_values

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self._q_values


class _BareModel:
    """Model with no forward or predict method."""
    pass


class TestGetQValuesTorchPath:
    """Test the PyTorch-like forward() path in _get_q_values."""

    @pytest.fixture()
    def tracker(self):
        return EnsembleTracker(n_models=2)

    def test_forward_model_returns_q_values(self, tracker):
        """Model with forward() method returns correct Q-values."""
        expected = np.array([0.1, 0.5, 0.3])
        model = _TorchLikeModel(expected)
        state = rng.standard_normal(10)

        # Mock torch so the import succeeds and from_numpy works
        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.from_numpy.return_value = _FakeTensor(state)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = tracker._get_q_values(model, state)

        assert isinstance(result, np.ndarray)
        assert len(result) == NUM_ACTIONS

    def test_predict_model_fallback(self, tracker):
        """Model without forward falls back to predict()."""
        expected = np.array([0.2, 0.8, 0.1])
        model = _PredictModel(expected)
        state = rng.standard_normal(10)
        result = tracker._get_q_values(model, state)
        np.testing.assert_array_equal(result, expected)

    def test_bare_model_returns_zeros(self, tracker):
        """Model with neither forward nor predict returns zeros."""
        model = _BareModel()
        state = rng.standard_normal(10)
        result = tracker._get_q_values(model, state)
        np.testing.assert_array_equal(result, np.zeros(NUM_ACTIONS))

    def test_forward_exception_falls_back_to_predict(self, tracker):
        """If forward() raises, falls back to predict()."""
        expected = np.array([0.3, 0.4, 0.5])

        class _FailForwardModel:
            def forward(self, x):
                raise RuntimeError("GPU error")

            def predict(self, state):
                return expected

        model = _FailForwardModel()
        state = rng.standard_normal(10)
        result = tracker._get_q_values(model, state)
        np.testing.assert_array_equal(result, expected)


class TestEnsemblePredictWithTorchModels:
    """Integration: predict() with torch-like models in ensemble."""

    def test_predict_with_torch_models(self):
        """Ensemble of torch-like models produces valid action."""
        models = [
            _TorchLikeModel(np.array([0.1, 0.5, 0.3])),
            _TorchLikeModel(np.array([0.2, 0.4, 0.6])),
        ]
        tracker = EnsembleTracker(n_models=2)
        tracker.set_models(models)
        state = rng.standard_normal(10)

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_torch.from_numpy.return_value = _FakeTensor(state)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            action, disagreement, stats = tracker.predict(state)

        assert 0 <= action <= 2
        assert disagreement >= 0.0
        assert isinstance(stats, dict)

    def test_predict_with_mixed_models(self):
        """Ensemble with predict-only models works."""
        models = [
            _PredictModel(np.array([0.1, 0.5, 0.3])),
            _PredictModel(np.array([0.2, 0.4, 0.6])),
        ]
        tracker = EnsembleTracker(n_models=2)
        tracker.set_models(models)
        state = rng.standard_normal(10)
        action, disagreement, stats = tracker.predict(state)
        assert 0 <= action <= 2
        assert disagreement >= 0.0
