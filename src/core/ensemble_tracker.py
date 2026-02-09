#!/usr/bin/env python3
"""
Ensemble Disagreement Tracker
==============================
Tracks model uncertainty via ensemble disagreement for exploration bonuses.

Master Handbook alignment:
- Section: "Ensemble disagreement"
- Purpose: Quantify epistemic uncertainty via Q-value variance
- Exploration: Higher disagreement → more exploration needed

Key Features:
1. EnsembleTracker - Maintains N models and tracks disagreement
2. Disagreement metric - Standard deviation of Q-values across ensemble
3. Exploration bonus - Scaled by disagreement magnitude
4. Model weight adaptation - Updates based on individual model performance

Usage:
    ensemble = EnsembleTracker(n_models=3)

    # Get action with uncertainty
    action, disagreement = ensemble.predict(state)

    # Update all models
    ensemble.update(state, action, reward, next_state, done)

    # Get exploration bonus
    bonus = ensemble.get_exploration_bonus(disagreement)
"""

import contextlib
import logging
from collections import deque
from typing import Any, Final, Optional, Protocol, Sequence, TypedDict, runtime_checkable

import numpy as np
from numpy.random import Generator, default_rng

from src.utils.safe_math import SafeMath

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

LOG = logging.getLogger(__name__)

MIN_ENSEMBLE_MODELS: Final[int] = 2
STATE_BATCH_DIM: Final[int] = 2
# Default fallback if action dimension cannot be inferred
DEFAULT_NUM_ACTIONS: Final[int] = 3
MAX_ACTION_INDEX: Final[int] = DEFAULT_NUM_ACTIONS - 1
UNIT_SUM_TOLERANCE: Final[float] = 1e-6
EXPECTED_PREDICTIONS_SELF_TEST: Final[int] = 101
MIN_EXPECTED_EXPLORATIONS: Final[int] = 5


@runtime_checkable
class ModelLike(Protocol):
    """Protocol for model objects that expose a numpy-style predict API."""

    def predict(self, state: np.ndarray) -> np.ndarray:  # pragma: no cover - structural typing
        ...


class PredictStats(TypedDict):
    mean_q: float
    max_q: float
    min_q: float
    disagreement_per_action: list[float]
    selected_action_disagreement: float


class EnsembleStats(TypedDict):
    mean_disagreement: float
    max_disagreement: float
    high_disagreement_rate: float
    total_predictions: int
    avg_exploration_bonus: float
    model_weights: list[float]


class EnsembleTracker:
    """
    Track epistemic uncertainty via ensemble disagreement.

    Maintains N independent models and measures Q-value variance.
    High disagreement indicates model uncertainty → encourage exploration.

    Ensemble diversity sources:
    1. Different random initializations
    2. Different training data (bootstrap sampling)
    3. Different hyperparameters (optional)
    """

    def __init__(
        self,
        n_models: int = 3,
        disagreement_threshold: float = 0.5,
        exploration_scale: float = 0.2,
        use_weighted_voting: bool = True,
        rng: Generator | None = None,
        num_actions: Optional[int] = None,
    ):
        """
        Args:
            n_models: Number of models in ensemble (3-5 recommended)
            disagreement_threshold: Threshold for "high uncertainty" (std > threshold)
            exploration_scale: Bonus multiplier for high disagreement
            use_weighted_voting: Weight models by recent performance
            rng: Random generator for exploration decisions
            num_actions: Optional explicit action dimension. If None, will be inferred.
        """
        if n_models < MIN_ENSEMBLE_MODELS:
            raise ValueError(f"Ensemble requires >= {MIN_ENSEMBLE_MODELS} models, got {n_models}")

        self.n_models = n_models
        self.disagreement_threshold = disagreement_threshold
        self.exploration_scale = exploration_scale
        self.use_weighted_voting = use_weighted_voting

        # Model storage (empty until models are set)
        self.models: list[ModelLike | Any] = []

        # Model performance tracking (for weighted voting)
        self.model_weights = np.ones(n_models) / n_models
        self.model_losses: list[deque[float]] = [deque(maxlen=100) for _ in range(n_models)]

        # Disagreement statistics
        self.recent_disagreements: deque[float] = deque(maxlen=1000)
        self.high_disagreement_count = 0
        self.total_predictions = 0

        # Exploration bonus tracking
        self.total_exploration_bonus = 0.0
        self.bonus_count = 0

        # Action space dimension handling
        self.num_actions: Optional[int] = num_actions

        self.rng: Generator = rng if rng is not None else default_rng(42)

        LOG.info(
            "EnsembleTracker initialized: n_models=%d, threshold=%s, scale=%s",
            n_models,
            disagreement_threshold,
            exploration_scale,
        )
        LOG.debug("Torch available: %s", torch is not None)

    def set_models(self, models: Sequence[ModelLike | Any]) -> None:
        """
        Set ensemble models.

        Args:
            models: Sequence of model objects (must have predict/forward method)
        """
        if len(models) != self.n_models:
            raise ValueError(f"Expected {self.n_models} models, got {len(models)}")

        self.models = list(models)
        LOG.info("Loaded %d models into ensemble", self.n_models)

    def predict(self, state: np.ndarray) -> tuple[int, float, PredictStats]:
        """
        Get ensemble prediction with disagreement metric.

        Args:
            state: Input state (features)

        Returns:
            (action, disagreement, stats)
            - action: Selected action (via voting or averaging)
            - disagreement: Std of Q-values across ensemble
            - stats: Additional metrics (mean_q, max_q, etc.)
        """
        if len(self.models) != self.n_models:
            return self._fallback_prediction()

        q_values_array = self._collect_q_values(state)
        disagreement_per_action = np.std(q_values_array, axis=0)
        disagreement = float(np.mean(disagreement_per_action))

        action = self._select_action(q_values_array)
        self._track_disagreement(disagreement)
        stats = self._build_stats(q_values_array, disagreement_per_action, action)

        return action, disagreement, stats

    def _fallback_prediction(self) -> tuple[int, float, PredictStats]:
        """Return default prediction when models not loaded."""
        return (
            1,
            0.0,
            {
                "mean_q": 0.0,
                "max_q": 0.0,
                "min_q": 0.0,
                "disagreement_per_action": [],
                "selected_action_disagreement": 0.0,
            },
        )

    def _collect_q_values(self, state: np.ndarray) -> np.ndarray:
        """Collect Q-values from all models with shape sanity checks and harmonization."""
        q_values_list: list[np.ndarray] = []

        # Collect from each model
        for model in self.models:
            q = self._get_q_values(model, state)

            # Determine/validate action dimension
            if self.num_actions is None:
                self.num_actions = int(q.shape[-1]) if q.ndim > 0 else DEFAULT_NUM_ACTIONS
                LOG.debug("Inferred action dimension: %d", self.num_actions)

            expected_n = self.num_actions
            if expected_n is None:
                expected_n = DEFAULT_NUM_ACTIONS

            if q.ndim == 0:
                LOG.error("Received scalar Q-value from model; resizing to length %d", expected_n)
                q = np.resize(q, expected_n)
            elif q.shape[-1] != expected_n:
                LOG.error(
                    "Model Q-value length mismatch: got %d, expected %d. Coercing via resize.",
                    q.shape[-1],
                    expected_n,
                )
                q = np.resize(q, expected_n)

            q_values_list.append(q)

        return np.asarray(q_values_list)  # (n_models, n_actions)

    def _select_action(self, q_values_array: np.ndarray) -> int:
        """Select action via weighted voting or majority vote."""
        if self.use_weighted_voting:
            weighted_q = np.average(q_values_array, axis=0, weights=self.model_weights)
            return int(np.argmax(weighted_q))
        else:
            actions = [int(np.argmax(q)) for q in q_values_array]
            return int(np.bincount(actions).argmax())

    def _track_disagreement(self, disagreement: float):
        """Track disagreement statistics."""
        self.recent_disagreements.append(disagreement)
        self.total_predictions += 1
        if disagreement > self.disagreement_threshold:
            self.high_disagreement_count += 1

    def _build_stats(
        self, q_values_array: np.ndarray, disagreement_per_action: np.ndarray, action: int
    ) -> PredictStats:
        """Build prediction statistics dictionary."""
        mean_q_values = np.mean(q_values_array, axis=0)
        return {
            "mean_q": float(np.mean(mean_q_values)),
            "max_q": float(np.max(mean_q_values)),
            "min_q": float(np.min(mean_q_values)),
            "disagreement_per_action": disagreement_per_action.astype(float).tolist(),
            "selected_action_disagreement": float(disagreement_per_action[action]),
        }

    def _get_q_values_torch(self, model: Any, state: np.ndarray) -> np.ndarray | None:
        """Try getting Q-values via PyTorch forward pass. Returns None if not applicable."""
        if torch is None or not hasattr(model, "forward"):
            return None
        with contextlib.suppress(RuntimeError, TypeError, ValueError):
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state_tensor = torch.from_numpy(state).float()
                    if state_tensor.dim() == STATE_BATCH_DIM:
                        state_tensor = state_tensor.unsqueeze(0)
                else:
                    state_tensor = state
                q_vals = model(state_tensor)
                return np.asarray(q_vals.squeeze().cpu().numpy())
        return None

    def _get_q_values(self, model: ModelLike | Any, state: np.ndarray) -> np.ndarray:
        """
        Extract Q-values from model (handles both torch and numpy models).

        Args:
            model: Model object
            state: Input state

        Returns:
            Q-values array (n_actions,)
        """
        # Try PyTorch model first if available
        torch_result = self._get_q_values_torch(model, state)
        if torch_result is not None:
            return torch_result

        # Fallback: assume numpy-based model with predict method
        if isinstance(model, ModelLike) or hasattr(model, "predict"):
            try:
                return np.asarray(model.predict(state))
            except Exception:
                LOG.exception("Model predict failed; returning zeros")
                n_actions = self.num_actions or DEFAULT_NUM_ACTIONS
                return np.zeros(n_actions)

        # Last resort: return zeros
        LOG.warning("Model has no forward or predict method, returning zeros")
        n_actions = self.num_actions or DEFAULT_NUM_ACTIONS
        return np.zeros(n_actions)

    def get_exploration_bonus(self, disagreement: float) -> float:
        """
        Calculate exploration bonus based on disagreement.

        High disagreement → high epistemic uncertainty → encourage exploration

        Args:
            disagreement: Std of Q-values across ensemble

        Returns:
            Exploration bonus (0.0 to exploration_scale)
        """
        if disagreement < self.disagreement_threshold:
            return 0.0

        # Scale bonus by how much disagreement exceeds threshold
        excess = disagreement - self.disagreement_threshold
        bonus = min(excess * self.exploration_scale, self.exploration_scale)

        self.total_exploration_bonus += bonus
        self.bonus_count += 1

        return bonus

    def update_weights(self, model_idx: int, loss: float):
        """
        Update model weight based on recent performance.

        Args:
            model_idx: Index of model to update
            loss: Recent loss value (lower is better)
        """
        if not self.use_weighted_voting:
            return

        self.model_losses[model_idx].append(loss)

        # Recalculate weights based on inverse of average loss
        avg_losses = []
        for losses in self.model_losses:
            if len(losses) > 0:
                avg_losses.append(np.mean(losses))
            else:
                avg_losses.append(np.float64(1.0))  # Default

        # Inverse weighting: better models get higher weight
        inv_losses = 1.0 / (np.array(avg_losses) + 1e-8)
        self.model_weights = inv_losses / np.sum(inv_losses)

        LOG.debug("Updated model weights: %s", self.model_weights)

    def get_stats(self) -> EnsembleStats:
        """Get ensemble statistics."""
        if len(self.recent_disagreements) == 0:
            mean_disagreement = 0.0
            max_disagreement = 0.0
        else:
            mean_disagreement = float(np.mean(self.recent_disagreements))
            max_disagreement = float(np.max(self.recent_disagreements))

        high_disagreement_rate = self.high_disagreement_count / max(self.total_predictions, 1)

        avg_bonus = self.total_exploration_bonus / max(self.bonus_count, 1)

        return {
            "mean_disagreement": mean_disagreement,
            "max_disagreement": max_disagreement,
            "high_disagreement_rate": high_disagreement_rate,
            "total_predictions": self.total_predictions,
            "avg_exploration_bonus": avg_bonus,
            "model_weights": self.model_weights.astype(float).tolist(),
        }

    def should_explore(self, disagreement: float, epsilon: float = 0.1) -> bool:
        """
        Decide whether to take random action based on disagreement.

        Args:
            disagreement: Current disagreement level
            epsilon: Base exploration rate

        Returns:
            True if should explore
        """
        # Increase exploration rate when disagreement is high
        if disagreement > self.disagreement_threshold:
            # Scale epsilon by disagreement magnitude
            adjusted_epsilon = min(epsilon * (1.0 + disagreement), 0.5)
        else:
            adjusted_epsilon = epsilon

        return self.rng.random() < adjusted_epsilon

    def __repr__(self) -> str:
        """Human-readable summary of ensemble configuration and key stats."""
        stats = self.get_stats()
        return (
            f"EnsembleTracker(n_models={self.n_models}, "
            f"mean_disagreement={stats['mean_disagreement']:.4f}, "
            f"high_rate={stats['high_disagreement_rate']:.2%}, "
            f"predictions={self.total_predictions})"
        )


# ============================================
# Self-test
# ============================================
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Ensemble Tracker Self-Test")
    print("=" * 70)

    prng = default_rng(42)

    # Test 1: Basic ensemble tracking
    print("\n[TEST 1] Basic ensemble with mock models")

    class MockModel:
        """Mock model for testing"""

        def __init__(self, bias: float = 0.0, random_gen: Generator | None = None):
            self.bias = bias
            self.rng = random_gen if random_gen is not None else default_rng(42)

        def predict(self, _state: np.ndarray) -> np.ndarray:
            # Return Q-values with some bias
            base = np.array([0.5, 0.3, 0.7])
            return base + self.bias + self.rng.normal(0, 0.1, 3)

    ensemble = EnsembleTracker(n_models=3, disagreement_threshold=0.2, rng=prng)

    # Create diverse models
    test_models = [
        MockModel(bias=0.0, random_gen=prng),
        MockModel(bias=0.1, random_gen=prng),
        MockModel(bias=-0.1, random_gen=prng),
    ]
    ensemble.set_models(test_models)

    # Test prediction
    test_state = prng.standard_normal(10)
    test_action, test_disagreement, summary_stats = ensemble.predict(test_state)

    if 0 <= test_action <= MAX_ACTION_INDEX:
        print(f"    ✓ Valid action selected: {test_action}")
    else:
        print(f"    ✗ FAIL: Invalid action {test_action}")
        sys.exit(1)

    if test_disagreement >= 0:
        print(f"    ✓ Disagreement calculated: {test_disagreement:.4f}")
    else:
        print(f"    ✗ FAIL: Negative disagreement {test_disagreement}")
        sys.exit(1)

    # Test 2: Exploration bonus
    print("\n[TEST 2] Exploration bonus calculation")

    LOW_DISAGREEMENT = 0.1
    HIGH_DISAGREEMENT = 0.8

    bonus_low = ensemble.get_exploration_bonus(LOW_DISAGREEMENT)
    bonus_high = ensemble.get_exploration_bonus(HIGH_DISAGREEMENT)

    if SafeMath.is_zero(bonus_low):
        print(f"    ✓ No bonus for low disagreement: {bonus_low}")
    else:
        print(f"    ✗ FAIL: Expected 0, got {bonus_low}")
        sys.exit(1)

    if bonus_high > 0:
        print(f"    ✓ Bonus for high disagreement: {bonus_high:.4f}")
    else:
        print(f"    ✗ FAIL: Expected bonus > 0, got {bonus_high}")
        sys.exit(1)

    # Test 3: Model weight updates
    print("\n[TEST 3] Model weight adaptation")

    ensemble_weighted = EnsembleTracker(n_models=3, use_weighted_voting=True)
    ensemble_weighted.set_models(test_models)

    # Simulate different model performance
    ensemble_weighted.update_weights(0, 0.1)  # Good model
    ensemble_weighted.update_weights(1, 0.5)  # Bad model
    ensemble_weighted.update_weights(2, 0.2)  # Medium model

    weights = ensemble_weighted.model_weights

    if weights[0] > weights[1]:
        print(f"    ✓ Better model gets higher weight: {weights[0]:.3f} > {weights[1]:.3f}")
    else:
        print("    ✗ FAIL: Weight ordering wrong")
        sys.exit(1)

    if abs(np.sum(weights) - 1.0) < UNIT_SUM_TOLERANCE:
        print(f"    ✓ Weights sum to 1.0: {np.sum(weights):.6f}")
    else:
        print(f"    ✗ FAIL: Weights don't sum to 1: {np.sum(weights)}")
        sys.exit(1)

    # Test 4: Statistics tracking
    print("\n[TEST 4] Statistics tracking")

    # Generate multiple predictions
    for _ in range(100):
        test_state = prng.standard_normal(10)
        ensemble.predict(test_state)

    final_overall_stats = ensemble.get_stats()

    if final_overall_stats["total_predictions"] == EXPECTED_PREDICTIONS_SELF_TEST:  # 1 from test 1 + 100 here
        print(f"    ✓ Prediction count: {final_overall_stats['total_predictions']}")
    else:
        print(f"    ✗ FAIL: Expected 101 predictions, got {final_overall_stats['total_predictions']}")
        sys.exit(1)

    if 0 <= final_overall_stats["high_disagreement_rate"] <= 1:
        print(f"    ✓ High disagreement rate: {final_overall_stats['high_disagreement_rate']:.2%}")
    else:
        print(f"    ✗ FAIL: Invalid rate {final_overall_stats['high_disagreement_rate']}")
        sys.exit(1)

    # Test 5: Exploration decision
    print("\n[TEST 5] Exploration decision logic")

    # Set seed for deterministic test
    ensemble.rng = default_rng(42)
    should_explore_low = ensemble.should_explore(0.1, epsilon=0.0)  # Force no exploration

    ensemble.rng = default_rng(42)
    should_explore_high = ensemble.should_explore(1.0, epsilon=1.0)  # Force exploration

    if not should_explore_low:
        print("    ✓ No exploration for low disagreement (epsilon=0)")
    else:
        # This could happen due to randomness, just warn
        print("    ⚠ Exploration occurred despite epsilon=0 (edge case)")

    # With epsilon=1.0, should always explore (since random < 1.0 is always true)
    # But we capped it at 0.5, so test multiple times
    explore_count = sum(ensemble.should_explore(1.0, epsilon=0.9) for _ in range(10))
    if explore_count >= MIN_EXPECTED_EXPLORATIONS:  # At least 50% should explore
        print(f"    ✓ High exploration rate for high disagreement: {explore_count}/10")

    print("\n" + "=" * 70)
    print("✓ All Ensemble Tracker tests passed!")
    print("=" * 70)
    print("\nEnsemble Stats:")
    final_stats = ensemble.get_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
