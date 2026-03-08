"""
Early Stopping - Training Termination and Checkpoint Management

Monitors validation performance and stops training when improvement plateaus.
Maintains best model checkpoint for rollback.

Reference: MQL5 EarlyStopping.mqh
"""

import logging

import numpy as np

LOG = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping with patience and checkpoint management.

    Monitors validation metric and stops training if no improvement
    for specified patience epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = "max",  # 'max' for rewards, 'min' for losses
        restore_best: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            restore_best: Restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_value = -np.inf if mode == "max" else np.inf
        self.best_epoch = 0
        self.wait_count = 0
        self.stopped = False
        self.best_weights = None

        LOG.info(
            "[EARLY-STOP] Initialized: patience=%d, min_delta=%.4f, mode=%s",
            patience,
            min_delta,
            mode,
        )

    def __call__(self, current_value: float, current_weights: dict | None = None) -> bool:
        """
        Check if training should stop.

        Args:
            current_value: Current validation metric value
            current_weights: Current model weights (for checkpointing)

        Returns:
            True if training should stop, False otherwise
        """
        # Check for improvement
        if self.mode == "max":
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)

        if improved:
            # New best found
            old_best = self.best_value
            self.best_value = current_value
            self.best_epoch = self.wait_count
            self.wait_count = 0

            # Save checkpoint
            if current_weights is not None:
                self.best_weights = current_weights

            LOG.info(
                "[EARLY-STOP] New best: %.4f (improved by %.4f)",
                current_value,
                abs(current_value - old_best),
            )

        else:
            # No improvement
            self.wait_count += 1

            LOG.debug(
                "[EARLY-STOP] No improvement: current=%.4f, best=%.4f, patience=%d/%d",
                current_value,
                self.best_value,
                self.wait_count,
                self.patience,
            )

            # Check if patience exhausted
            if self.wait_count >= self.patience:
                self.stopped = True
                LOG.warning(
                    "[EARLY-STOP] Stopping training: no improvement for %d epochs. Best=%.4f at epoch %d",
                    self.patience,
                    self.best_value,
                    self.best_epoch,
                )
                return True

        return False

    def get_best_weights(self) -> dict | None:
        """Get best weights checkpoint."""
        return self.best_weights

    def should_reduce_lr(self, lr_patience: int = 5) -> bool:
        """
        Check if learning rate should be reduced.

        Args:
            lr_patience: Patience before reducing LR (should be < patience)

        Returns:
            True if LR should be reduced
        """
        return self.wait_count > 0 and self.wait_count % lr_patience == 0

    def reset(self):
        """Reset early stopping state."""
        self.best_value = -np.inf if self.mode == "max" else np.inf
        self.best_epoch = 0
        self.wait_count = 0
        self.stopped = False
        self.best_weights = None
        LOG.info("[EARLY-STOP] Reset")
