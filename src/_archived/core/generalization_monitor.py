"""
Generalization Monitor - Detect Training-Live Performance Gap

Uses Kolmogorov-Smirnov test to detect distribution shifts between
training and live performance.

Key Metrics:
- Train vs Live reward distribution (KS test)
- Expected Calibration Error (ECE) for probability calibration
- Confidence degradation
- State classification: HEALTHY, OVERFITTING, UNDERFITTING, REGIME_SHIFT

Reference: MQL5 GeneralizationMonitor.mqh
"""

import logging
from collections import deque
from enum import Enum

import numpy as np

# Generalization detection thresholds
OVERFITTING_GAP_THRESHOLD: float = 0.1  # Training significantly better than live
UNDERFITTING_THRESHOLD: float = -0.05  # Both train and live performing poorly

LOG = logging.getLogger(__name__)


class GeneralizationState(Enum):
    """System generalization states."""

    HEALTHY = "healthy"  # Good generalization
    OVERFITTING = "overfitting"  # Train >> Live performance
    UNDERFITTING = "underfitting"  # Both train and live poor
    REGIME_SHIFT = "regime_shift"  # Distribution changed


class GeneralizationMonitor:
    """
    Monitor training-live performance gap for overfitting detection.

    Tracks two windows:
    - Training window: Performance during training episodes
    - Live window: Performance during live trading

    Detects overfitting when training performance significantly
    exceeds live performance (using KS test).
    """

    def __init__(
        self,
        window_size: int = 100,
        ks_threshold: float = 0.3,  # KS statistic threshold for distribution shift
        ece_bins: int = 10,  # Bins for Expected Calibration Error
        min_samples: int = 30,  # Minimum samples before testing
    ):
        """
        Initialize generalization monitor.

        Args:
            window_size: Size of performance windows
            ks_threshold: KS test threshold for overfitting detection
            ece_bins: Number of bins for calibration error
            min_samples: Minimum samples before running tests
        """
        self.window_size = window_size
        self.ks_threshold = ks_threshold
        self.ece_bins = ece_bins
        self.min_samples = min_samples

        # Performance windows
        self.train_rewards = deque(maxlen=window_size)
        self.live_rewards = deque(maxlen=window_size)

        # Confidence calibration tracking
        self.predicted_probs = deque(maxlen=window_size)
        self.actual_outcomes = deque(maxlen=window_size)

        # State tracking
        self.current_state = GeneralizationState.HEALTHY
        self.ks_statistic = 0.0
        self.ks_pvalue = 1.0
        self.ece_score = 0.0

        LOG.info("[GEN-MON] Initialized: window=%d, ks_threshold=%.3f", window_size, ks_threshold)

    def add_train_reward(self, reward: float):
        """Add reward from training episode."""
        self.train_rewards.append(reward)

    def add_live_reward(self, reward: float):
        """Add reward from live trading."""
        self.live_rewards.append(reward)

    def add_prediction(self, predicted_prob: float, actual_outcome: bool):
        """
        Add prediction for calibration monitoring.

        Args:
            predicted_prob: Predicted probability of success [0, 1]
            actual_outcome: Actual outcome (True=success, False=failure)
        """
        self.predicted_probs.append(predicted_prob)
        self.actual_outcomes.append(1.0 if actual_outcome else 0.0)

    def update(self) -> dict:
        """
        Update generalization state and run tests.

        Returns:
            Dictionary with test results and current state
        """
        # Need minimum samples
        if len(self.train_rewards) < self.min_samples or len(self.live_rewards) < self.min_samples:
            return {
                "state": GeneralizationState.HEALTHY.value,
                "ks_statistic": 0.0,
                "ks_pvalue": 1.0,
                "ece": 0.0,
                "train_mean": 0.0,
                "live_mean": 0.0,
                "gap": 0.0,
                "ready": False,
            }

        # Kolmogorov-Smirnov test for distribution difference
        self.ks_statistic, self.ks_pvalue = self._ks_test_2sample(
            np.array(self.train_rewards), np.array(self.live_rewards)
        )

        # Expected Calibration Error
        if len(self.predicted_probs) >= self.min_samples:
            self.ece_score = self._calculate_ece(np.array(self.predicted_probs), np.array(self.actual_outcomes))

        # Calculate performance gap
        train_mean = np.mean(self.train_rewards)
        live_mean = np.mean(self.live_rewards)
        gap = train_mean - live_mean

        # Classify state
        self.current_state = self._classify_state(train_mean, live_mean, gap)

        result = {
            "state": self.current_state.value,
            "ks_statistic": float(self.ks_statistic),
            "ks_pvalue": float(self.ks_pvalue),
            "ece": float(self.ece_score),
            "train_mean": float(train_mean),
            "live_mean": float(live_mean),
            "gap": float(gap),
            "ready": True,
            "overfitting_detected": self.current_state == GeneralizationState.OVERFITTING,
        }

        if self.current_state != GeneralizationState.HEALTHY:
            LOG.warning(
                "[GEN-MON] State=%s, gap=%.4f, KS=%.3f (p=%.3f), ECE=%.3f",
                self.current_state.value,
                gap,
                self.ks_statistic,
                self.ks_pvalue,
                self.ece_score,
            )

        return result

    def _ks_test_2sample(self, sample1: np.ndarray, sample2: np.ndarray) -> tuple[float, float]:
        """
        Two-sample Kolmogorov-Smirnov test.

        Tests null hypothesis that samples are drawn from same distribution.

        Args:
            sample1: First sample (training rewards)
            sample2: Second sample (live rewards)

        Returns:
            Tuple of (KS statistic, p-value)
        """
        n1 = len(sample1)
        n2 = len(sample2)

        # Sort both samples
        data1 = np.sort(sample1)
        data2 = np.sort(sample2)

        # Combine and sort all data points
        all_data = np.concatenate([data1, data2])
        sorted_data = np.sort(all_data)

        # Calculate empirical CDFs at each unique point
        cdf1 = np.searchsorted(data1, sorted_data, side="right") / n1
        cdf2 = np.searchsorted(data2, sorted_data, side="right") / n2

        # KS statistic = max absolute difference between CDFs
        ks_stat = np.max(np.abs(cdf1 - cdf2))

        # Approximate p-value using Kolmogorov distribution
        # For large samples, KS follows: P(D > d) ≈ 2 * exp(-2 * n * d^2)
        n_eff = np.sqrt(n1 * n2 / (n1 + n2))
        pvalue = 2 * np.exp(-2 * n_eff**2 * ks_stat**2)
        pvalue = min(1.0, pvalue)  # Clamp to [0, 1]

        return ks_stat, pvalue

    def _calculate_ece(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """
        Calculate Expected Calibration Error.

        Measures how well predicted probabilities match actual frequencies.
        Perfect calibration: if predict 70%, actual should be 70%.

        Args:
            predicted_probs: Predicted probabilities [0, 1]
            actual_outcomes: Actual outcomes {0, 1}

        Returns:
            ECE score (0 = perfect calibration)
        """
        # Create bins [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
        bin_edges = np.linspace(0, 1, self.ece_bins + 1)

        ece = 0.0
        total_samples = len(predicted_probs)

        for i in range(self.ece_bins):
            # Find predictions in this bin
            in_bin = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])

            if i == self.ece_bins - 1:  # Last bin includes 1.0
                in_bin = (predicted_probs >= bin_edges[i]) & (predicted_probs <= bin_edges[i + 1])

            n_bin = np.sum(in_bin)

            if n_bin == 0:
                continue

            # Average predicted probability in bin
            avg_pred = np.mean(predicted_probs[in_bin])

            # Actual frequency of positive outcomes in bin
            avg_actual = np.mean(actual_outcomes[in_bin])

            # Weighted contribution to ECE
            ece += (n_bin / total_samples) * abs(avg_pred - avg_actual)

        return ece

    def _classify_state(self, train_mean: float, live_mean: float, gap: float) -> GeneralizationState:
        """
        Classify generalization state based on performance metrics.

        Args:
            train_mean: Mean training reward
            live_mean: Mean live reward
            gap: Train - Live gap

        Returns:
            Current generalization state
        """
        # Significant distribution shift detected
        if self.ks_statistic > self.ks_threshold:
            # Train >> Live = Overfitting
            if gap > OVERFITTING_GAP_THRESHOLD:  # Training significantly better
                return GeneralizationState.OVERFITTING
            # Both poor = Underfitting
            elif train_mean < 0 and live_mean < 0:
                return GeneralizationState.UNDERFITTING
            # Distribution shifted but not clear overfitting
            else:
                return GeneralizationState.REGIME_SHIFT

        # Both performing poorly = Underfitting
        if train_mean < UNDERFITTING_THRESHOLD and live_mean < UNDERFITTING_THRESHOLD:
            return GeneralizationState.UNDERFITTING

        # Default: healthy
        return GeneralizationState.HEALTHY

    def get_recommendation(self) -> str:
        """
        Get training recommendation based on current state.

        Returns:
            Recommended action string
        """
        if self.current_state == GeneralizationState.OVERFITTING:
            return "INCREASE_REGULARIZATION"
        elif self.current_state == GeneralizationState.UNDERFITTING:
            return "INCREASE_CAPACITY"
        elif self.current_state == GeneralizationState.REGIME_SHIFT:
            return "COLLECT_MORE_DATA"
        else:
            return "CONTINUE_TRAINING"

    def reset(self):
        """Reset monitoring state."""
        self.train_rewards.clear()
        self.live_rewards.clear()
        self.predicted_probs.clear()
        self.actual_outcomes.clear()
        self.current_state = GeneralizationState.HEALTHY
        LOG.info("[GEN-MON] Reset monitoring state")
