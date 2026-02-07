import pytest

"""Tests for GeneralizationMonitor."""

import numpy as np

from src.core.generalization_monitor import (
    GeneralizationMonitor,
    GeneralizationState,
)


class TestGeneralizationMonitorInit:
    def test_default_init(self):
        gm = GeneralizationMonitor()
        assert gm.window_size == 100
        assert gm.ks_threshold == pytest.approx(0.3)
        assert gm.min_samples == 30
        assert gm.current_state == GeneralizationState.HEALTHY

    def test_custom_init(self):
        gm = GeneralizationMonitor(window_size=50, ks_threshold=0.5, min_samples=10)
        assert gm.window_size == 50
        assert gm.ks_threshold == pytest.approx(0.5)
        assert gm.min_samples == 10


class TestGeneralizationMonitorUpdate:
    def test_not_ready_below_min_samples(self):
        gm = GeneralizationMonitor(min_samples=30)
        for i in range(10):
            gm.add_train_reward(float(i))
            gm.add_live_reward(float(i))
        result = gm.update()
        assert result["ready"] is False
        assert result["state"] == "healthy"

    def test_healthy_when_same_distribution(self):
        gm = GeneralizationMonitor(min_samples=30, ks_threshold=0.3)
        rng = np.random.default_rng(42)
        for _ in range(50):
            val = rng.normal(0.5, 0.1)
            gm.add_train_reward(val)
            gm.add_live_reward(val + rng.normal(0, 0.01))
        result = gm.update()
        assert result["ready"] is True
        assert result["state"] == "healthy"

    def test_overfitting_detected(self):
        gm = GeneralizationMonitor(min_samples=30, ks_threshold=0.3)
        # Training performs much better than live
        for _ in range(50):
            gm.add_train_reward(1.0)  # High training reward
            gm.add_live_reward(-0.5)  # Low live reward
        result = gm.update()
        assert result["ready"] is True
        assert result["overfitting_detected"] is True
        assert result["state"] == "overfitting"

    def test_underfitting_detected(self):
        gm = GeneralizationMonitor(min_samples=30, ks_threshold=0.1)
        # Both train and live performing poorly
        for _ in range(50):
            gm.add_train_reward(-0.5)
            gm.add_live_reward(-0.5)
        result = gm.update()
        assert result["ready"] is True
        assert result["state"] == "underfitting"


class TestECE:
    def test_perfect_calibration(self):
        gm = GeneralizationMonitor(min_samples=10)
        # Add enough rewards for update to work
        for _ in range(30):
            gm.add_train_reward(0.5)
            gm.add_live_reward(0.5)
        # Add perfectly calibrated predictions
        rng = np.random.default_rng(42)
        for _ in range(50):
            prob = rng.uniform(0.0, 1.0)
            outcome = rng.random() < prob  # Calibrated
            gm.add_prediction(prob, outcome)
        result = gm.update()
        # ECE should be low (not necessarily 0 with finite samples)
        assert result["ece"] < 0.3

    def test_poor_calibration(self):
        gm = GeneralizationMonitor(min_samples=10)
        for _ in range(30):
            gm.add_train_reward(0.5)
            gm.add_live_reward(0.5)
        # Always predict high prob but always fail
        for _ in range(50):
            gm.add_prediction(0.9, False)
        result = gm.update()
        assert result["ece"] > 0.5  # Should be high


class TestKSTest:
    def test_identical_distributions(self):
        gm = GeneralizationMonitor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ks_stat, pvalue = gm._ks_test_2sample(data, data)
        assert ks_stat == pytest.approx(0.0)
        assert pvalue >= 0.99

    def test_different_distributions(self):
        gm = GeneralizationMonitor()
        rng = np.random.default_rng(42)
        data1 = rng.normal(0, 1, size=100)
        data2 = rng.normal(5, 1, size=100)
        ks_stat, pvalue = gm._ks_test_2sample(data1, data2)
        assert ks_stat > 0.5
        assert pvalue < 0.01


class TestRecommendations:
    def test_overfitting_recommendation(self):
        gm = GeneralizationMonitor()
        gm.current_state = GeneralizationState.OVERFITTING
        assert gm.get_recommendation() == "INCREASE_REGULARIZATION"

    def test_underfitting_recommendation(self):
        gm = GeneralizationMonitor()
        gm.current_state = GeneralizationState.UNDERFITTING
        assert gm.get_recommendation() == "INCREASE_CAPACITY"

    def test_regime_shift_recommendation(self):
        gm = GeneralizationMonitor()
        gm.current_state = GeneralizationState.REGIME_SHIFT
        assert gm.get_recommendation() == "COLLECT_MORE_DATA"

    def test_healthy_recommendation(self):
        gm = GeneralizationMonitor()
        assert gm.get_recommendation() == "CONTINUE_TRAINING"


class TestReset:
    def test_reset_clears_state(self):
        gm = GeneralizationMonitor(min_samples=5)
        for _ in range(10):
            gm.add_train_reward(1.0)
            gm.add_live_reward(-1.0)
            gm.add_prediction(0.9, False)
        gm.update()
        gm.reset()
        assert len(gm.train_rewards) == 0
        assert len(gm.live_rewards) == 0
        assert len(gm.predicted_probs) == 0
        assert gm.current_state == GeneralizationState.HEALTHY
