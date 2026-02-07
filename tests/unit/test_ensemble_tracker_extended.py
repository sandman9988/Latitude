"""Extended tests for src.core.ensemble_tracker.

Covers majority-vote selection, _get_q_values fallback paths, update_weights
edge cases, should_explore probability, and stats with exploration bonuses.
"""

import numpy as np
import pytest
from numpy.random import default_rng

from src.core.ensemble_tracker import EnsembleTracker, NUM_ACTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class PredictModel:
    """Model with a predict() method (numpy model)."""
    def __init__(self, q_values):
        self._q = np.array(q_values, dtype=float)

    def predict(self, state):
        return self._q.copy()


class BadModel:
    """Model with no forward or predict method."""
    pass


# ---------------------------------------------------------------------------
# Majority vote (use_weighted_voting=False)
# ---------------------------------------------------------------------------
class TestMajorityVote:
    def test_majority_vote_selects_most_popular(self):
        # Two models prefer action 2, one prefers action 0
        ens = EnsembleTracker(n_models=3, use_weighted_voting=False)
        models = [
            PredictModel([1.0, 0.0, 0.5]),  # argmax=0
            PredictModel([0.0, 0.0, 1.0]),  # argmax=2
            PredictModel([0.0, 0.0, 1.0]),  # argmax=2
        ]
        ens.set_models(models)
        action, dis, stats = ens.predict(np.zeros(5))
        assert action == 2

    def test_weighted_vote_different_from_majority(self):
        """With skewed weights, the weighted path can differ from majority."""
        ens = EnsembleTracker(n_models=3, use_weighted_voting=True)
        models = [
            PredictModel([10.0, 0.0, 0.0]),  # strong action 0
            PredictModel([0.0, 0.0, 1.0]),   # action 2
            PredictModel([0.0, 0.0, 1.0]),   # action 2
        ]
        ens.set_models(models)
        # Give model 0 almost all weight
        ens.model_weights = np.array([0.98, 0.01, 0.01])
        action, _, _ = ens.predict(np.zeros(5))
        assert action == 0  # weighted average dominated by model 0


# ---------------------------------------------------------------------------
# _get_q_values fallback paths
# ---------------------------------------------------------------------------
class TestGetQValues:
    def test_predict_method_model(self):
        ens = EnsembleTracker(n_models=2)
        model = PredictModel([0.3, 0.7, 0.1])
        result = ens._get_q_values(model, np.zeros(5))
        np.testing.assert_array_almost_equal(result, [0.3, 0.7, 0.1])

    def test_bad_model_returns_zeros(self):
        ens = EnsembleTracker(n_models=2)
        model = BadModel()
        result = ens._get_q_values(model, np.zeros(5))
        np.testing.assert_array_equal(result, np.zeros(NUM_ACTIONS))


# ---------------------------------------------------------------------------
# update_weights edge cases
# ---------------------------------------------------------------------------
class TestUpdateWeightsExtended:
    def test_weights_sum_to_one(self):
        ens = EnsembleTracker(n_models=3, use_weighted_voting=True)
        ens.update_weights(0, 0.1)
        ens.update_weights(1, 0.5)
        ens.update_weights(2, 0.3)
        assert np.isclose(np.sum(ens.model_weights), 1.0)

    def test_multiple_losses_averaged(self):
        ens = EnsembleTracker(n_models=2, use_weighted_voting=True)
        ens.update_weights(0, 0.1)
        ens.update_weights(0, 0.2)
        ens.update_weights(1, 1.0)
        # Model 0 has avg loss ~0.15, model 1 has 1.0 → model 0 should have higher weight
        assert ens.model_weights[0] > ens.model_weights[1]

    def test_no_update_when_disabled(self):
        ens = EnsembleTracker(n_models=2, use_weighted_voting=False)
        original = ens.model_weights.copy()
        ens.update_weights(0, 0.01)
        np.testing.assert_array_equal(ens.model_weights, original)

    def test_empty_losses_use_default(self):
        ens = EnsembleTracker(n_models=3, use_weighted_voting=True)
        # Only update one model
        ens.update_weights(0, 0.1)
        # Other models default to loss 1.0, so model 0 should have highest weight
        assert ens.model_weights[0] > ens.model_weights[1]


# ---------------------------------------------------------------------------
# get_exploration_bonus extended
# ---------------------------------------------------------------------------
class TestExplorationBonusExtended:
    def test_bonus_tracked_in_stats(self):
        ens = EnsembleTracker(n_models=2, disagreement_threshold=0.1, exploration_scale=0.5)
        ens.get_exploration_bonus(0.3)
        ens.get_exploration_bonus(0.5)
        stats = ens.get_stats()
        assert stats["avg_exploration_bonus"] > 0

    def test_bonus_zero_below_threshold(self):
        ens = EnsembleTracker(n_models=2, disagreement_threshold=1.0)
        assert ens.get_exploration_bonus(0.5) == pytest.approx(0.0)

    def test_bonus_monotonically_increases_with_disagreement(self):
        ens = EnsembleTracker(n_models=2, disagreement_threshold=0.1, exploration_scale=1.0)
        b1 = ens.get_exploration_bonus(0.2)
        b2 = ens.get_exploration_bonus(0.5)
        assert b2 >= b1


# ---------------------------------------------------------------------------
# should_explore edge cases
# ---------------------------------------------------------------------------
class TestShouldExploreExtended:
    def test_epsilon_zero_low_disagreement(self):
        ens = EnsembleTracker(n_models=2, disagreement_threshold=0.5, rng=default_rng(42))
        # Low disagreement + epsilon=0 → never explore
        assert ens.should_explore(0.1, epsilon=0.0) is False

    def test_high_disagreement_high_epsilon(self):
        ens = EnsembleTracker(n_models=2, disagreement_threshold=0.1, rng=default_rng(42))
        # Run many times, majority should explore
        count = sum(ens.should_explore(1.0, epsilon=0.4) for _ in range(100))
        assert count > 30  # adjusted epsilon is capped at 0.5

    def test_adjusted_epsilon_capped_at_half(self):
        ens = EnsembleTracker(n_models=2, disagreement_threshold=0.1, rng=default_rng(42))
        # Even huge disagreement caps at 0.5
        count = sum(ens.should_explore(100.0, epsilon=0.4) for _ in range(1000))
        # Expect ~50% exploration rate
        assert 400 < count < 600


# ---------------------------------------------------------------------------
# get_stats with data
# ---------------------------------------------------------------------------
class TestGetStatsExtended:
    def test_stats_with_predictions(self):
        ens = EnsembleTracker(n_models=2, rng=default_rng(42))
        models = [PredictModel([0.1, 0.2, 0.3]), PredictModel([0.4, 0.5, 0.6])]
        ens.set_models(models)
        for _ in range(10):
            ens.predict(np.zeros(5))
        stats = ens.get_stats()
        assert stats["total_predictions"] == 10
        assert stats["mean_disagreement"] > 0
        assert len(stats["model_weights"]) == 2

    def test_high_disagreement_rate(self):
        ens = EnsembleTracker(n_models=2, disagreement_threshold=0.0001, rng=default_rng(42))
        models = [PredictModel([0.1, 0.2, 0.3]), PredictModel([0.9, 0.8, 0.7])]
        ens.set_models(models)
        for _ in range(20):
            ens.predict(np.zeros(5))
        stats = ens.get_stats()
        assert stats["high_disagreement_rate"] > 0.5


# ---------------------------------------------------------------------------
# _track_disagreement
# ---------------------------------------------------------------------------
class TestTrackDisagreement:
    def test_count_increments(self):
        ens = EnsembleTracker(n_models=2, disagreement_threshold=0.5)
        ens._track_disagreement(0.6)
        ens._track_disagreement(0.3)
        assert ens.total_predictions == 2
        assert ens.high_disagreement_count == 1
        assert len(ens.recent_disagreements) == 2

    def test_max_history_enforced(self):
        ens = EnsembleTracker(n_models=2)
        for i in range(2000):
            ens._track_disagreement(float(i) * 0.001)
        assert len(ens.recent_disagreements) == 1000


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------
class TestReprExtended:
    def test_repr_after_predictions(self):
        ens = EnsembleTracker(n_models=2, rng=default_rng(42))
        models = [PredictModel([0.1, 0.2, 0.3]), PredictModel([0.3, 0.2, 0.1])]
        ens.set_models(models)
        ens.predict(np.zeros(5))
        r = repr(ens)
        assert "EnsembleTracker" in r
        assert "predictions=1" in r
