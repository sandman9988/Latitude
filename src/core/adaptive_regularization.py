"""
Adaptive Regularization - Dynamic L2 Weight and Dropout Adjustment

Automatically adjusts regularization based on overfitting signals.

Reference: MQL5 AdaptiveRegularization.mqh
"""

import logging
from collections import deque

import numpy as np

LOG = logging.getLogger(__name__)


class AdaptiveRegularization:
    """
    Dynamically adjust regularization strength based on overfitting signals.

    Increases L2 weight and dropout when overfitting detected.
    Decreases when underfitting detected.
    """

    def __init__(
        self,
        initial_l2: float = 0.0001,
        initial_dropout: float = 0.1,
        l2_range: tuple = (1e-5, 1e-2),
        dropout_range: tuple = (0.0, 0.5),
        adjustment_rate: float = 1.2,  # Multiplicative adjustment factor
    ):
        """
        Initialize adaptive regularization.

        Args:
            initial_l2: Starting L2 weight
            initial_dropout: Starting dropout rate
            l2_range: (min, max) bounds for L2 weight
            dropout_range: (min, max) bounds for dropout
            adjustment_rate: Factor to multiply/divide by when adjusting
        """
        self.l2_weight = initial_l2
        self.dropout_rate = initial_dropout
        self.l2_min, self.l2_max = l2_range
        self.dropout_min, self.dropout_max = dropout_range
        self.adjustment_rate = adjustment_rate

        self.adjustment_history: deque[str] = deque(maxlen=1000)

        LOG.info(
            "[ADAPT-REG] Initialized: L2=%.5f [%.5f, %.5f], Dropout=%.3f [%.3f, %.3f]",
            initial_l2,
            self.l2_min,
            self.l2_max,
            initial_dropout,
            self.dropout_min,
            self.dropout_max,
        )

    def increase_regularization(self):
        """Increase regularization (response to overfitting)."""
        old_l2 = self.l2_weight
        old_dropout = self.dropout_rate

        self.l2_weight = min(self.l2_weight * self.adjustment_rate, self.l2_max)
        self.dropout_rate = min(self.dropout_rate * self.adjustment_rate, self.dropout_max)

        self.adjustment_history.append("increase")

        LOG.info(
            "[ADAPT-REG] INCREASE: L2 %.5f→%.5f, Dropout %.3f→%.3f",
            old_l2,
            self.l2_weight,
            old_dropout,
            self.dropout_rate,
        )

    def decrease_regularization(self):
        """Decrease regularization (response to underfitting)."""
        old_l2 = self.l2_weight
        old_dropout = self.dropout_rate

        self.l2_weight = max(self.l2_weight / self.adjustment_rate, self.l2_min)
        self.dropout_rate = max(self.dropout_rate / self.adjustment_rate, self.dropout_min)

        self.adjustment_history.append("decrease")

        LOG.info(
            "[ADAPT-REG] DECREASE: L2 %.5f→%.5f, Dropout %.3f→%.3f",
            old_l2,
            self.l2_weight,
            old_dropout,
            self.dropout_rate,
        )

    def update_from_signal(self, overfitting_signal: str):
        """
        Update regularization based on overfitting detector signal.

        Args:
            overfitting_signal: One of:
                - "INCREASE_REGULARIZATION" → increase
                - "INCREASE_CAPACITY" → decrease
                - "CONTINUE_TRAINING" → no change
                - "COLLECT_MORE_DATA" → no change
        """
        if overfitting_signal == "INCREASE_REGULARIZATION":
            self.increase_regularization()
        elif overfitting_signal == "INCREASE_CAPACITY":
            self.decrease_regularization()
        # else: no adjustment

    def get_current(self) -> dict:
        """
        Get current regularization parameters.

        Returns:
            Dictionary with l2_weight and dropout_rate
        """
        return {"l2_weight": self.l2_weight, "dropout_rate": self.dropout_rate}

    def reset(self, l2: float | None = None, dropout: float | None = None):
        """
        Reset to specified or initial values.

        Args:
            l2: L2 weight (None = use initial)
            dropout: Dropout rate (None = use initial)
        """
        if l2 is not None:
            self.l2_weight = np.clip(l2, self.l2_min, self.l2_max)
        if dropout is not None:
            self.dropout_rate = np.clip(dropout, self.dropout_min, self.dropout_max)

        self.adjustment_history.clear()
        LOG.info("[ADAPT-REG] Reset: L2=%.5f, Dropout=%.3f", self.l2_weight, self.dropout_rate)
