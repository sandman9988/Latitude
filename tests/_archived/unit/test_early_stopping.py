import pytest

"""Tests for EarlyStopping."""

import numpy as np

from src.core.early_stopping import EarlyStopping


class TestEarlyStoppingMaxMode:
    """Test early stopping in 'max' mode (maximize reward)."""

    def test_init_defaults(self):
        es = EarlyStopping()
        assert es.patience == 10
        assert es.mode == "max"
        assert es.best_value == -np.inf
        assert es.stopped is False
        assert es.wait_count == 0

    def test_improvement_resets_wait(self):
        es = EarlyStopping(patience=3)
        assert es(0.5) is False  # New best
        assert es.wait_count == 0
        assert es(0.4) is False  # No improvement, wait=1
        assert es.wait_count == 1
        assert es(0.6) is False  # New best, wait resets
        assert es.wait_count == 0

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        es(1.0)  # best = 1.0
        assert es(0.9) is False  # wait=1
        assert es(0.8) is False  # wait=2
        assert es(0.7) is True  # wait=3 -> stop
        assert es.stopped is True

    def test_min_delta_respected(self):
        es = EarlyStopping(patience=3, min_delta=0.1)
        es(1.0)  # best = 1.0
        # Tiny improvement below min_delta doesn't count
        assert es(1.05) is False  # Not enough improvement, wait=1
        assert es.wait_count == 1
        # Large improvement above min_delta does count
        assert es(1.2) is False  # New best
        assert es.wait_count == 0

    def test_checkpoint_saved(self):
        es = EarlyStopping(patience=5)
        weights_v1 = {"layer1": [1, 2, 3]}
        weights_v2 = {"layer1": [4, 5, 6]}

        es(0.5, current_weights=weights_v1)
        assert es.get_best_weights() == weights_v1

        es(0.7, current_weights=weights_v2)
        assert es.get_best_weights() == weights_v2

        # No improvement - weights stay at v2
        es(0.3, current_weights={"layer1": [7, 8, 9]})
        assert es.get_best_weights() == weights_v2


class TestEarlyStoppingMinMode:
    """Test early stopping in 'min' mode (minimize loss)."""

    def test_min_mode_init(self):
        es = EarlyStopping(mode="min")
        assert es.best_value == np.inf

    def test_min_mode_improvement(self):
        es = EarlyStopping(patience=3, mode="min", min_delta=0.0)
        assert es(1.0) is False  # New best
        assert es.best_value == pytest.approx(1.0)
        assert es(0.5) is False  # Lower is better -> new best
        assert es.best_value == pytest.approx(0.5)
        assert es(0.8) is False  # Worse, wait=1
        assert es.wait_count == 1

    def test_min_mode_stops(self):
        es = EarlyStopping(patience=2, mode="min", min_delta=0.0)
        es(0.5)
        assert es(0.6) is False  # wait=1
        assert es(0.7) is True  # wait=2 -> stop


class TestEarlyStoppingUtilities:
    """Test utility methods."""

    def test_should_reduce_lr(self):
        es = EarlyStopping(patience=20)
        es(1.0)  # best
        # Simulate no improvement for 10 epochs
        for _ in range(10):
            es(0.5)
        assert es.should_reduce_lr(lr_patience=5) is True
        assert es.should_reduce_lr(lr_patience=3) is False  # 10 % 3 != 0

    def test_reset(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        es(0.5)  # wait=1
        es.reset()
        assert es.best_value == -np.inf
        assert es.wait_count == 0
        assert es.stopped is False
        assert es.best_weights is None

    def test_no_weights_returns_none(self):
        es = EarlyStopping()
        assert es.get_best_weights() is None
