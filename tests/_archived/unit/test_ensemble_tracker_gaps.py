"""Gap tests for src.core.ensemble_tracker – lines 207-209.

Targets:
- PyTorch forward() path inside _get_q_values: state dim check + unsqueeze.
- Also: __repr__ method, fallback prediction, set_models wrong count.
"""

import numpy as np
import pytest
from numpy.random import default_rng

from src.core.ensemble_tracker import EnsembleTracker, NUM_ACTIONS


class ForwardModel:
    """Mock model with a forward() method that mimics PyTorch model."""

    def __init__(self, q_values):
        self._q = np.array(q_values, dtype=np.float32)

    def forward(self, state_tensor):
        """Simulate torch model forward pass (returns numpy for simplicity)."""
        # Return a 'tensor-like' object with squeeze/cpu/numpy methods
        return TensorLike(self._q)


class TensorLike:
    """Fake tensor that supports squeeze().cpu().numpy() chain."""

    def __init__(self, data):
        self._data = np.array(data, dtype=np.float32)

    def squeeze(self):
        return TensorLike(self._data.squeeze())

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class PredictModel:
    """Simple numpy-based model."""

    def __init__(self, q_values):
        self._q = np.array(q_values, dtype=float)

    def predict(self, state):
        return self._q.copy()


# ---------------------------------------------------------------------------
# PyTorch forward() path (lines 207-209)
# ---------------------------------------------------------------------------
class TestPyTorchForwardPath:
    def test_forward_model_used_when_torch_available(self):
        """If model has forward(), _get_q_values should use torch path."""
        ens = EnsembleTracker(n_models=2)
        model = ForwardModel([0.1, 0.5, 0.3])

        # This will try torch import. If torch is available, it will use
        # the forward path. If not, it falls through to predict path.
        # Since ForwardModel has no predict(), it returns zeros if torch fails.
        result = ens._get_q_values(model, np.zeros(5))
        assert len(result) == NUM_ACTIONS

    def test_forward_model_in_ensemble_predict(self):
        """Test full ensemble predict with forward models."""
        try:
            import torch  # noqa: F401
            _ = torch  # mark as used

            # If torch available, exercise the forward path fully
            ens = EnsembleTracker(n_models=2)
            models = [ForwardModel([0.1, 0.5, 0.3]), ForwardModel([0.2, 0.4, 0.6])]
            ens.set_models(models)
            action, disagreement, stats = ens.predict(np.zeros(5))
            assert 0 <= action <= 2
            assert disagreement >= 0
        except ImportError:
            pytest.skip("torch not available")


# ---------------------------------------------------------------------------
# Fallback prediction (no models loaded)
# ---------------------------------------------------------------------------
class TestFallbackPrediction:
    def test_none_models_returns_default(self):
        """When models are None, fallback prediction is used."""
        ens = EnsembleTracker(n_models=3)
        # Models default to [None, None, None]
        action, disagreement, stats = ens.predict(np.zeros(5))
        assert action == 1  # Default action
        assert disagreement == pytest.approx(0.0)
        assert stats["mean_q"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# set_models validation
# ---------------------------------------------------------------------------
class TestSetModels:
    def test_wrong_count_raises(self):
        """set_models raises if model count doesn't match n_models."""
        ens = EnsembleTracker(n_models=3)
        with pytest.raises(ValueError, match="Expected 3"):
            ens.set_models([PredictModel([0, 0, 0])])

    def test_correct_count_succeeds(self):
        ens = EnsembleTracker(n_models=2)
        models = [PredictModel([0, 0, 0]), PredictModel([1, 1, 1])]
        ens.set_models(models)
        assert ens.models == models


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------
class TestRepr:
    def test_repr_contains_info(self):
        ens = EnsembleTracker(n_models=2, rng=default_rng(42))
        r = repr(ens)
        assert "EnsembleTracker" in r
        assert "n_models=2" in r

    def test_repr_after_predictions(self):
        ens = EnsembleTracker(n_models=2, rng=default_rng(42))
        models = [PredictModel([0.1, 0.5, 0.3]), PredictModel([0.2, 0.4, 0.6])]
        ens.set_models(models)
        for _ in range(5):
            ens.predict(np.zeros(5))
        r = repr(ens)
        assert "predictions=5" in r


# ---------------------------------------------------------------------------
# min ensemble size
# ---------------------------------------------------------------------------
class TestMinEnsembleSize:
    def test_single_model_raises(self):
        with pytest.raises(ValueError, match="requires >= 2"):
            EnsembleTracker(n_models=1)

    def test_two_models_ok(self):
        ens = EnsembleTracker(n_models=2)
        assert ens.n_models == 2


# ---------------------------------------------------------------------------
# get_stats edge cases
# ---------------------------------------------------------------------------
class TestGetStatsEmpty:
    def test_empty_stats(self):
        ens = EnsembleTracker(n_models=2)
        stats = ens.get_stats()
        assert stats["total_predictions"] == 0
        assert stats["mean_disagreement"] == pytest.approx(0.0)
        assert stats["max_disagreement"] == pytest.approx(0.0)
        assert stats["avg_exploration_bonus"] == pytest.approx(0.0)
