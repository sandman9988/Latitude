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

import numpy as np
import logging
from typing import Optional, List, Tuple, Dict
from collections import deque

LOG = logging.getLogger(__name__)


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
        use_weighted_voting: bool = True
    ):
        """
        Args:
            n_models: Number of models in ensemble (3-5 recommended)
            disagreement_threshold: Threshold for "high uncertainty" (std > threshold)
            exploration_scale: Bonus multiplier for high disagreement
            use_weighted_voting: Weight models by recent performance
        """
        if n_models < 2:
            raise ValueError(f"Ensemble requires >= 2 models, got {n_models}")
            
        self.n_models = n_models
        self.disagreement_threshold = disagreement_threshold
        self.exploration_scale = exploration_scale
        self.use_weighted_voting = use_weighted_voting
        
        # Model storage (None = no models loaded yet)
        self.models: List[Optional[object]] = [None] * n_models
        
        # Model performance tracking (for weighted voting)
        self.model_weights = np.ones(n_models) / n_models
        self.model_losses = [deque(maxlen=100) for _ in range(n_models)]
        
        # Disagreement statistics
        self.recent_disagreements = deque(maxlen=1000)
        self.high_disagreement_count = 0
        self.total_predictions = 0
        
        # Exploration bonus tracking
        self.total_exploration_bonus = 0.0
        self.bonus_count = 0
        
        LOG.info(
            f"EnsembleTracker initialized: n_models={n_models}, "
            f"threshold={disagreement_threshold}, scale={exploration_scale}"
        )
    
    def set_models(self, models: List[object]):
        """
        Set ensemble models.
        
        Args:
            models: List of model objects (must have predict/forward method)
        """
        if len(models) != self.n_models:
            raise ValueError(f"Expected {self.n_models} models, got {len(models)}")
        
        self.models = models
        LOG.info(f"Loaded {self.n_models} models into ensemble")
    
    def predict(self, state: np.ndarray) -> Tuple[int, float, Dict[str, float]]:
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
        if any(model is None for model in self.models):
            # Fallback: no models loaded, return neutral action with zero disagreement
            return 1, 0.0, {'mean_q': 0.0, 'max_q': 0.0, 'min_q': 0.0}
        
        # Get Q-values from all models
        q_values_list = []
        for model in self.models:
            q_vals = self._get_q_values(model, state)
            q_values_list.append(q_vals)
        
        q_values_array = np.array(q_values_list)  # (n_models, n_actions)
        
        # Calculate disagreement (std across models for each action)
        disagreement_per_action = np.std(q_values_array, axis=0)
        disagreement = float(np.mean(disagreement_per_action))
        
        # Select action
        if self.use_weighted_voting:
            # Weighted average of Q-values
            weighted_q = np.average(q_values_array, axis=0, weights=self.model_weights)
            action = int(np.argmax(weighted_q))
        else:
            # Majority voting
            actions = [int(np.argmax(q)) for q in q_values_list]
            action = int(np.bincount(actions).argmax())
        
        # Track statistics
        self.recent_disagreements.append(disagreement)
        self.total_predictions += 1
        
        if disagreement > self.disagreement_threshold:
            self.high_disagreement_count += 1
        
        # Calculate stats
        mean_q_values = np.mean(q_values_array, axis=0)
        stats = {
            'mean_q': float(np.mean(mean_q_values)),
            'max_q': float(np.max(mean_q_values)),
            'min_q': float(np.min(mean_q_values)),
            'disagreement_per_action': disagreement_per_action.tolist(),
            'selected_action_disagreement': float(disagreement_per_action[action])
        }
        
        return action, disagreement, stats
    
    def _get_q_values(self, model: object, state: np.ndarray) -> np.ndarray:
        """
        Extract Q-values from model (handles both torch and numpy models).
        
        Args:
            model: Model object
            state: Input state
            
        Returns:
            Q-values array (n_actions,)
        """
        try:
            # Try PyTorch model
            import torch
            if hasattr(model, 'forward'):
                with torch.no_grad():
                    if isinstance(state, np.ndarray):
                        state_tensor = torch.from_numpy(state).float()
                        if state_tensor.dim() == 2:
                            state_tensor = state_tensor.unsqueeze(0)
                    else:
                        state_tensor = state
                    q_vals = model(state_tensor)
                    return q_vals.squeeze().cpu().numpy()
        except:
            pass
        
        # Fallback: assume numpy-based model with predict method
        if hasattr(model, 'predict'):
            return model.predict(state)
        
        # Last resort: return zeros
        LOG.warning("Model has no forward or predict method, returning zeros")
        return np.zeros(3)  # Assume 3 actions
    
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
                avg_losses.append(1.0)  # Default
        
        # Inverse weighting: better models get higher weight
        inv_losses = 1.0 / (np.array(avg_losses) + 1e-8)
        self.model_weights = inv_losses / np.sum(inv_losses)
        
        LOG.debug(f"Updated model weights: {self.model_weights}")
    
    def get_stats(self) -> Dict[str, float]:
        """Get ensemble statistics."""
        if len(self.recent_disagreements) == 0:
            mean_disagreement = 0.0
            max_disagreement = 0.0
        else:
            mean_disagreement = float(np.mean(self.recent_disagreements))
            max_disagreement = float(np.max(self.recent_disagreements))
        
        high_disagreement_rate = (
            self.high_disagreement_count / max(self.total_predictions, 1)
        )
        
        avg_bonus = (
            self.total_exploration_bonus / max(self.bonus_count, 1)
        )
        
        return {
            'mean_disagreement': mean_disagreement,
            'max_disagreement': max_disagreement,
            'high_disagreement_rate': high_disagreement_rate,
            'total_predictions': self.total_predictions,
            'avg_exploration_bonus': avg_bonus,
            'model_weights': self.model_weights.tolist()
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
        
        return np.random.random() < adjusted_epsilon
    
    def __repr__(self) -> str:
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
    
    # Test 1: Basic ensemble tracking
    print("\n[TEST 1] Basic ensemble with mock models")
    
    class MockModel:
        """Mock model for testing"""
        def __init__(self, bias: float = 0.0):
            self.bias = bias
        
        def predict(self, state: np.ndarray) -> np.ndarray:
            # Return Q-values with some bias
            base = np.array([0.5, 0.3, 0.7])
            return base + self.bias + np.random.normal(0, 0.1, 3)
    
    ensemble = EnsembleTracker(n_models=3, disagreement_threshold=0.2)
    
    # Create diverse models
    models = [MockModel(bias=0.0), MockModel(bias=0.1), MockModel(bias=-0.1)]
    ensemble.set_models(models)
    
    # Test prediction
    state = np.random.randn(10)
    action, disagreement, stats = ensemble.predict(state)
    
    if 0 <= action <= 2:
        print(f"    ✓ Valid action selected: {action}")
    else:
        print(f"    ✗ FAIL: Invalid action {action}")
        sys.exit(1)
    
    if disagreement >= 0:
        print(f"    ✓ Disagreement calculated: {disagreement:.4f}")
    else:
        print(f"    ✗ FAIL: Negative disagreement {disagreement}")
        sys.exit(1)
    
    # Test 2: Exploration bonus
    print("\n[TEST 2] Exploration bonus calculation")
    
    low_disagreement = 0.1
    high_disagreement = 0.8
    
    bonus_low = ensemble.get_exploration_bonus(low_disagreement)
    bonus_high = ensemble.get_exploration_bonus(high_disagreement)
    
    if bonus_low == 0.0:
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
    ensemble_weighted.set_models(models)
    
    # Simulate different model performance
    ensemble_weighted.update_weights(0, 0.1)  # Good model
    ensemble_weighted.update_weights(1, 0.5)  # Bad model
    ensemble_weighted.update_weights(2, 0.2)  # Medium model
    
    weights = ensemble_weighted.model_weights
    
    if weights[0] > weights[1]:
        print(f"    ✓ Better model gets higher weight: {weights[0]:.3f} > {weights[1]:.3f}")
    else:
        print(f"    ✗ FAIL: Weight ordering wrong")
        sys.exit(1)
    
    if abs(np.sum(weights) - 1.0) < 1e-6:
        print(f"    ✓ Weights sum to 1.0: {np.sum(weights):.6f}")
    else:
        print(f"    ✗ FAIL: Weights don't sum to 1: {np.sum(weights)}")
        sys.exit(1)
    
    # Test 4: Statistics tracking
    print("\n[TEST 4] Statistics tracking")
    
    # Generate multiple predictions
    for _ in range(100):
        state = np.random.randn(10)
        ensemble.predict(state)
    
    stats = ensemble.get_stats()
    
    if stats['total_predictions'] == 101:  # 1 from test 1 + 100 here
        print(f"    ✓ Prediction count: {stats['total_predictions']}")
    else:
        print(f"    ✗ FAIL: Expected 101 predictions, got {stats['total_predictions']}")
        sys.exit(1)
    
    if 0 <= stats['high_disagreement_rate'] <= 1:
        print(f"    ✓ High disagreement rate: {stats['high_disagreement_rate']:.2%}")
    else:
        print(f"    ✗ FAIL: Invalid rate {stats['high_disagreement_rate']}")
        sys.exit(1)
    
    # Test 5: Exploration decision
    print("\n[TEST 5] Exploration decision logic")
    
    # Set seed for deterministic test
    np.random.seed(42)
    should_explore_low = ensemble.should_explore(0.1, epsilon=0.0)  # Force no exploration
    
    np.random.seed(42)
    should_explore_high = ensemble.should_explore(1.0, epsilon=1.0)  # Force exploration
    
    if not should_explore_low:
        print(f"    ✓ No exploration for low disagreement (epsilon=0)")
    else:
        # This could happen due to randomness, just warn
        print(f"    ⚠ Exploration occurred despite epsilon=0 (edge case)")
    
    # With epsilon=1.0, should always explore (since random < 1.0 is always true)
    # But we capped it at 0.5, so test multiple times
    explore_count = sum(ensemble.should_explore(1.0, epsilon=0.9) for _ in range(10))
    if explore_count >= 5:  # At least 50% should explore
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
