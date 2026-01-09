"""
Feature Tournament - Automatic Feature Selection

Tournament-based feature selection using survival scoring across
different market regimes and instruments.

Features compete based on:
- Predictive power (correlation with outcomes)
- Stability across regimes
- Non-redundancy (low correlation with other features)

Reference: MQL5 FeatureTournament.mqh
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import deque
import logging

LOG = logging.getLogger(__name__)


class FeatureTournament:
    """
    Tournament-based feature selection.
    
    Each feature competes based on:
    1. Predictive power: Correlation with target outcomes
    2. Regime stability: Consistent performance across market conditions
    3. Diversity: Low correlation with other selected features
    
    Features that consistently rank high across tournaments survive.
    """
    
    def __init__(
        self,
        n_features: int,
        feature_names: List[str],
        tournament_window: int = 100,
        survival_threshold: float = 0.6,
        max_correlation: float = 0.8  # Max correlation between features
    ):
        """
        Initialize feature tournament.
        
        Args:
            n_features: Total number of features
            feature_names: Names of features
            tournament_window: Window size for tournament evaluation
            survival_threshold: Min score to keep feature
            max_correlation: Max allowed correlation between features
        """
        self.n_features = n_features
        self.feature_names = feature_names
        self.tournament_window = tournament_window
        self.survival_threshold = survival_threshold
        self.max_correlation = max_correlation
        
        # Feature tracking
        self.feature_scores = np.ones(n_features)  # Start with all features active
        self.feature_active = np.ones(n_features, dtype=bool)
        
        # Tournament history
        self.feature_values = deque(maxlen=tournament_window)
        self.target_values = deque(maxlen=tournament_window)
        self.regime_tags = deque(maxlen=tournament_window)
        
        # Statistics
        self.tournaments_run = 0
        
        LOG.info(
            "[FEATURE-TOURNAMENT] Initialized: %d features, window=%d, threshold=%.2f",
            n_features, tournament_window, survival_threshold
        )
    
    def add_sample(self, features: np.ndarray, target: float, regime: int = 0):
        """
        Add sample for feature evaluation.
        
        Args:
            features: Feature vector (n_features,)
            target: Target outcome (e.g., trade PnL, direction correctness)
            regime: Market regime identifier
        """
        self.feature_values.append(features.copy())
        self.target_values.append(target)
        self.regime_tags.append(regime)
    
    def run_tournament(self) -> Dict:
        """
        Run feature tournament and update scores.
        
        Returns:
            Dictionary with tournament results
        """
        if len(self.feature_values) < self.tournament_window:
            return {'ready': False, 'n_samples': len(self.feature_values)}
        
        # Convert to arrays
        X = np.array(self.feature_values)  # (window, n_features)
        y = np.array(self.target_values)  # (window,)
        regimes = np.array(self.regime_tags)  # (window,)
        
        # Calculate scores for each feature
        new_scores = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            # Predictive power: Absolute correlation with target
            predictive_power = abs(self._safe_correlation(X[:, i], y))
            
            # Regime stability: Consistent correlation across regimes
            regime_stability = self._calculate_regime_stability(X[:, i], y, regimes)
            
            # Diversity: Low correlation with other high-scoring features
            diversity = self._calculate_diversity(X, i)
            
            # Combined score
            new_scores[i] = (
                0.5 * predictive_power +
                0.3 * regime_stability +
                0.2 * diversity
            )
        
        # Exponential moving average of scores
        alpha = 0.3  # Smoothing factor
        self.feature_scores = alpha * new_scores + (1 - alpha) * self.feature_scores
        
        # Update active features
        self.feature_active = self.feature_scores >= self.survival_threshold
        
        # Ensure at least some features are active
        n_active = np.sum(self.feature_active)
        if n_active == 0:
            # Activate top 3 features
            top_3_idx = np.argsort(self.feature_scores)[-3:]
            self.feature_active[top_3_idx] = True
        
        self.tournaments_run += 1
        
        # Get active feature names
        active_names = [
            self.feature_names[i] for i in range(self.n_features) if self.feature_active[i]
        ]
        
        result = {
            'ready': True,
            'tournament_id': self.tournaments_run,
            'n_active': int(np.sum(self.feature_active)),
            'n_total': self.n_features,
            'active_features': active_names,
            'scores': self.feature_scores.tolist(),
            'top_5_features': [
                self.feature_names[i] 
                for i in np.argsort(self.feature_scores)[-5:][::-1]
            ]
        }
        
        if self.tournaments_run % 10 == 0:
            LOG.info(
                "[FEATURE-TOURNAMENT] Round %d: %d/%d active, top: %s",
                self.tournaments_run, result['n_active'], self.n_features,
                ', '.join(result['top_5_features'][:3])
            )
        
        return result
    
    def _safe_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate correlation with safety checks.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Correlation coefficient (0 if constant)
        """
        # Handle constant arrays
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        
        # Calculate correlation
        corr = np.corrcoef(x, y)[0, 1]
        
        # Handle NaN
        if np.isnan(corr):
            return 0.0
        
        return corr
    
    def _calculate_regime_stability(
        self,
        feature: np.ndarray,
        target: np.ndarray,
        regimes: np.ndarray
    ) -> float:
        """
        Calculate stability of feature across regimes.
        
        Args:
            feature: Feature values
            target: Target values
            regimes: Regime identifiers
            
        Returns:
            Stability score [0, 1]
        """
        unique_regimes = np.unique(regimes)
        
        if len(unique_regimes) <= 1:
            return 1.0  # Only one regime, perfect stability
        
        # Calculate correlation in each regime
        regime_corrs = []
        for regime in unique_regimes:
            mask = regimes == regime
            if np.sum(mask) < 10:  # Need minimum samples
                continue
            
            corr = abs(self._safe_correlation(feature[mask], target[mask]))
            regime_corrs.append(corr)
        
        if len(regime_corrs) == 0:
            return 0.5  # Default middle value
        
        # Stability = 1 - coefficient of variation
        mean_corr = np.mean(regime_corrs)
        std_corr = np.std(regime_corrs)
        
        if mean_corr == 0:
            return 0.0
        
        cv = std_corr / mean_corr
        stability = 1.0 / (1.0 + cv)  # Transform to [0, 1]
        
        return stability
    
    def _calculate_diversity(self, X: np.ndarray, feature_idx: int) -> float:
        """
        Calculate diversity score (low correlation with other features).
        
        Args:
            X: All feature values (window, n_features)
            feature_idx: Index of feature to evaluate
            
        Returns:
            Diversity score [0, 1] (higher = more diverse)
        """
        feature = X[:, feature_idx]
        
        # Calculate correlation with all other high-scoring features
        correlations = []
        for i in range(self.n_features):
            if i == feature_idx:
                continue
            
            # Only check against active high-scoring features
            if self.feature_scores[i] >= self.survival_threshold:
                corr = abs(self._safe_correlation(feature, X[:, i]))
                correlations.append(corr)
        
        if len(correlations) == 0:
            return 1.0  # No other features to compare
        
        # Diversity = 1 - max correlation with other features
        max_corr = np.max(correlations)
        diversity = 1.0 - min(max_corr, 1.0)
        
        return diversity
    
    def get_active_features(self) -> np.ndarray:
        """
        Get boolean mask of active features.
        
        Returns:
            Boolean array indicating active features
        """
        return self.feature_active.copy()
    
    def get_active_indices(self) -> List[int]:
        """
        Get indices of active features.
        
        Returns:
            List of active feature indices
        """
        return list(np.where(self.feature_active)[0])
    
    def filter_features(self, features: np.ndarray) -> np.ndarray:
        """
        Filter feature vector to only active features.
        
        Args:
            features: Full feature vector (n_features,)
            
        Returns:
            Filtered feature vector (n_active,)
        """
        return features[self.feature_active]
    
    def reset(self):
        """Reset tournament state."""
        self.feature_scores = np.ones(self.n_features)
        self.feature_active = np.ones(self.n_features, dtype=bool)
        self.feature_values.clear()
        self.target_values.clear()
        self.regime_tags.clear()
        self.tournaments_run = 0
        LOG.info("[FEATURE-TOURNAMENT] Reset")
