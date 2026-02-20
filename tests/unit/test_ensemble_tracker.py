"""Tests for EnsembleTracker."""

import numpy as np
import pytest

from src.core.ensemble_tracker import EnsembleTracker


class MockModel:
    """Mock model returning predictable Q-values."""

    def __init__(self, q_values=None):
        self.q_values = q_values or [0.5, 0.3, 0.7]

    def predict(self, state):
        return np.array(self.q_values, dtype=np.float64)


class TestEnsembleTrackerInit:
    def test_default_init(self):
        et = EnsembleTracker(n_models=3)
        assert et.n_models == 3
        assert et.total_predictions == 0
        assert len(et.models) == 0  # models are registered via set_models()

    def test_minimum_models_enforced(self):
        with pytest.raises(ValueError):
            EnsembleTracker(n_models=1)

    def test_set_models(self):
        et = EnsembleTracker(n_models=2)
        models = [MockModel(), MockModel()]
        et.set_models(models)
        assert et.models == models

    def test_set_models_wrong_count(self):
        et = EnsembleTracker(n_models=3)
        with pytest.raises(ValueError):
            et.set_models([MockModel(), MockModel()])


class TestEnsembleTrackerPredict:
    def test_fallback_when_no_models(self):
        et = EnsembleTracker(n_models=2)
        action, disagreement, stats = et.predict(np.zeros(10))
        assert action == 1
        assert disagreement == pytest.approx(0.0)

    def test_predict_with_identical_models(self):
        et = EnsembleTracker(n_models=3)
        et.set_models([MockModel([0.1, 0.2, 0.9])] * 3)
        action, disagreement, stats = et.predict(np.zeros(10))
        assert action == 2  # argmax of [0.1, 0.2, 0.9]
        assert disagreement < 1e-10  # Near-zero disagreement (identical models)

    def test_predict_with_diverse_models(self):
        et = EnsembleTracker(n_models=3)
        et.set_models([
            MockModel([1.0, 0.0, 0.0]),
            MockModel([0.0, 1.0, 0.0]),
            MockModel([0.0, 0.0, 1.0]),
        ])
        action, disagreement, stats = et.predict(np.zeros(10))
        assert disagreement > 0  # Models disagree
        assert "mean_q" in stats
        assert "max_q" in stats

    def test_prediction_tracks_count(self):
        et = EnsembleTracker(n_models=2)
        et.set_models([MockModel(), MockModel()])
        for _ in range(5):
            et.predict(np.zeros(10))
        assert et.total_predictions == 5


class TestExplorationBonus:
    def test_no_bonus_below_threshold(self):
        et = EnsembleTracker(n_models=2, disagreement_threshold=0.5)
        bonus = et.get_exploration_bonus(0.3)
        assert bonus == pytest.approx(0.0)

    def test_bonus_above_threshold(self):
        et = EnsembleTracker(n_models=2, disagreement_threshold=0.5, exploration_scale=0.2)
        bonus = et.get_exploration_bonus(0.8)
        assert bonus > 0.0
        assert bonus <= 0.2  # Capped at exploration_scale

    def test_bonus_capped_at_scale(self):
        et = EnsembleTracker(n_models=2, disagreement_threshold=0.1, exploration_scale=0.2)
        bonus = et.get_exploration_bonus(100.0)  # Extremely high disagreement
        assert bonus == pytest.approx(0.2)  # Should cap


class TestModelWeights:
    def test_initial_weights_uniform(self):
        et = EnsembleTracker(n_models=3)
        np.testing.assert_allclose(et.model_weights, [1/3, 1/3, 1/3])

    def test_weight_update_favors_low_loss(self):
        et = EnsembleTracker(n_models=3, use_weighted_voting=True)
        et.set_models([MockModel(), MockModel(), MockModel()])
        et.update_weights(0, 0.1)  # Best model
        et.update_weights(1, 1.0)  # Worst model
        et.update_weights(2, 0.5)  # Medium model
        assert et.model_weights[0] > et.model_weights[1]
        assert abs(sum(et.model_weights) - 1.0) < 1e-6

    def test_no_weight_update_when_disabled(self):
        et = EnsembleTracker(n_models=2, use_weighted_voting=False)
        original = et.model_weights.copy()
        et.update_weights(0, 0.1)
        np.testing.assert_array_equal(et.model_weights, original)


class TestStats:
    def test_empty_stats(self):
        et = EnsembleTracker(n_models=2)
        stats = et.get_stats()
        assert stats["total_predictions"] == 0
        assert stats["mean_disagreement"] == pytest.approx(0.0)
        assert stats["max_disagreement"] == pytest.approx(0.0)

    def test_stats_after_predictions(self):
        et = EnsembleTracker(n_models=2, disagreement_threshold=0.1)
        et.set_models([MockModel([1.0, 0.0, 0.0]), MockModel([0.0, 1.0, 0.0])])
        for _ in range(10):
            et.predict(np.zeros(10))
        stats = et.get_stats()
        assert stats["total_predictions"] == 10
        assert stats["mean_disagreement"] > 0


class TestShouldExplore:
    def test_no_explore_with_zero_epsilon(self):
        et = EnsembleTracker(n_models=2)
        assert et.should_explore(0.1, epsilon=0.0) is False

    def test_explore_scales_with_disagreement(self):
        rng = np.random.default_rng(42)
        et = EnsembleTracker(n_models=2, disagreement_threshold=0.1, rng=rng)
        # High disagreement + high epsilon -> should explore often
        explore_count = sum(et.should_explore(1.0, epsilon=0.5) for _ in range(100))
        assert explore_count > 30  # Should explore more than half

    def test_repr(self):
        et = EnsembleTracker(n_models=2)
        r = repr(et)
        assert "EnsembleTracker" in r
        assert "n_models=2" in r
