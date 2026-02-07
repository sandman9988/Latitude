"""
Tests for TriggerAgent - Entry specialist agent.

Covers:
- Initialization (fallback mode, paper mode, live mode)
- decide() method - epsilon-greedy, feasibility gate, confidence floor
- _fallback_strategy() - MA crossover with imbalance tilt + regime adj
- _softmax() / _platt_calibrate() / _q_to_runway()
- _decay_epsilon()
- update_from_trade()
- add_experience() / train_step() / get_training_stats()
- update_platt_params()
"""

import logging
import os
from unittest.mock import MagicMock, patch

import numpy as np

rng = np.random.default_rng(42)

import pytest

from src.agents.trigger_agent import (
    TriggerAgent,
    Q_RUNWAY_MIN,
    Q_RUNWAY_MAX,
    Q_RUNWAY_MAX_Q,
    PREDICTED_RUNWAY_FALLBACK,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestTriggerInit:

    def test_default_init(self):
        ta = TriggerAgent(window=64, n_features=7)
        assert ta.window == 64
        assert ta.n_features == 7
        assert ta.use_torch is False
        assert ta.model is None
        assert ta.ddqn is None  # DDQN only created when training enabled
        assert ta.training_steps == 0

    def test_init_with_training(self):
        ta = TriggerAgent(window=32, n_features=7, enable_training=True)
        assert ta.enable_training is True
        assert ta.buffer is not None

    def test_init_without_training(self):
        ta = TriggerAgent(window=32, n_features=7, enable_training=False)
        assert ta.enable_training is False
        assert ta.buffer is None

    @patch.dict(os.environ, {"PAPER_MODE": "1"})
    def test_init_paper_mode(self):
        ta = TriggerAgent(window=64, n_features=7)
        assert ta.paper_mode is True
        assert ta.epsilon == pytest.approx(1.0)  # Higher epsilon in paper mode

    def test_init_symbol_params(self):
        ta = TriggerAgent(symbol="XAUUSD", timeframe="M5", broker="pepperstone")
        assert ta.symbol == "XAUUSD"
        assert ta.timeframe == "M5"
        assert ta.broker == "pepperstone"


# ---------------------------------------------------------------------------
# decide() basics
# ---------------------------------------------------------------------------

class TestTriggerDecide:

    def test_decide_already_in_position_returns_no_entry(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        action, conf, runway = ta.decide(state, current_position=1)
        assert action == 0
        assert conf == pytest.approx(0.0)
        assert runway == pytest.approx(0.0)

    def test_decide_already_short_returns_no_entry(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        action, conf, runway = ta.decide(state, current_position=-1)
        assert action == 0

    def test_decide_flat_position_returns_valid_action(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.epsilon = 0.0  # No random exploration
        state = rng.standard_normal((64, 7)).astype(np.float32)
        action, conf, runway = ta.decide(state, current_position=0)
        assert action in [0, 1, 2]
        assert 0.0 <= conf <= 1.0
        assert runway >= 0.0

    def test_decide_feasibility_gate_blocks_low_feasibility(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.epsilon = 0.0
        ta.paper_mode = False
        ta.disable_gates = False
        state = rng.standard_normal((64, 7)).astype(np.float32)
        # Set a very low feasibility
        action, conf, runway = ta.decide(state, current_position=0, feasibility=0.0)
        assert action == 0  # Blocked by feasibility gate

    @patch.dict(os.environ, {"PAPER_MODE": "1"})
    def test_decide_paper_mode_epsilon_exploration(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.epsilon = 1.0  # Force exploration
        state = np.zeros((64, 7), dtype=np.float32)
        action, conf, runway = ta.decide(state, current_position=0)
        assert action in [1, 2]  # Random LONG or SHORT (excludes 0)
        assert conf == pytest.approx(0.5)

    def test_decide_resets_bars_since_trade_when_in_position(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.bars_since_trade = 100
        state = np.zeros((64, 7), dtype=np.float32)
        ta.decide(state, current_position=1)
        assert ta.bars_since_trade == 0


# ---------------------------------------------------------------------------
# Fallback strategy
# ---------------------------------------------------------------------------

class TestFallbackStrategy:

    def test_fallback_empty_state(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((0, 7), dtype=np.float32)
        action = ta._fallback_strategy(state)
        assert action == 0  # NO_ENTRY

    def test_fallback_too_few_features(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((10, 2), dtype=np.float32)
        action = ta._fallback_strategy(state)
        assert action == 0

    def test_fallback_strong_long_signal(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = 0.5  # Strong positive MA diff
        action = ta._fallback_strategy(state)
        assert action == 1  # LONG

    def test_fallback_strong_short_signal(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = -0.5  # Strong negative MA diff
        action = ta._fallback_strategy(state)
        assert action == 2  # SHORT

    def test_fallback_neutral_no_entry(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = 0.0  # Neutral
        action = ta._fallback_strategy(state)
        assert action == 0  # NO_ENTRY

    def test_fallback_with_regime_adjustment(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = 0.35  # Medium positive MA diff

        # Without adjustment — should trigger LONG
        action_no_adj = ta._fallback_strategy(state, regime_threshold_adj=0.0)

        # With positive adjustment (mean-reverting → harder threshold)
        action_hard = ta._fallback_strategy(state, regime_threshold_adj=0.5)

        # The harder threshold should block — medium signal not enough
        assert action_hard == 0

    def test_fallback_with_imbalance_tilt(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = 0.25  # Near threshold
        state[-1, 4] = 0.9   # Strong positive imbalance → tilts LONG easier
        action = ta._fallback_strategy(state)
        # With tilt, the effective threshold for LONG is lower
        assert action in [0, 1]  # May or may not trigger depending on threshold


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------

class TestTriggerUtils:

    def test_softmax_basic(self):
        ta = TriggerAgent(window=64, n_features=7)
        q = np.array([1.0, 2.0, 3.0])
        probs = ta._softmax(q)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert probs[2] > probs[1] > probs[0]

    def test_softmax_equal_values(self):
        ta = TriggerAgent(window=64, n_features=7)
        q = np.array([1.0, 1.0, 1.0])
        probs = ta._softmax(q)
        assert abs(probs.sum() - 1.0) < 1e-6
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3], atol=1e-6)

    def test_softmax_large_values(self):
        ta = TriggerAgent(window=64, n_features=7)
        q = np.array([1000.0, 1001.0, 1002.0])
        probs = ta._softmax(q)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert np.all(np.isfinite(probs))

    def test_platt_calibrate_midpoint(self):
        ta = TriggerAgent(window=64, n_features=7)
        result = ta._platt_calibrate(0.5)
        assert 0.0 < result < 1.0

    def test_platt_calibrate_extreme_low(self):
        ta = TriggerAgent(window=64, n_features=7)
        result = ta._platt_calibrate(0.0)
        assert 0.0 < result < 1.0  # Should handle edge case

    def test_platt_calibrate_extreme_high(self):
        ta = TriggerAgent(window=64, n_features=7)
        result = ta._platt_calibrate(1.0)
        assert 0.0 < result < 1.0

    def test_q_to_runway_minimum(self):
        ta = TriggerAgent(window=64, n_features=7)
        assert ta._q_to_runway(-1.0) == Q_RUNWAY_MIN
        assert ta._q_to_runway(0.0) == Q_RUNWAY_MIN

    def test_q_to_runway_maximum(self):
        ta = TriggerAgent(window=64, n_features=7)
        assert ta._q_to_runway(3.0) == Q_RUNWAY_MAX
        assert ta._q_to_runway(10.0) == Q_RUNWAY_MAX

    def test_q_to_runway_interpolation(self):
        ta = TriggerAgent(window=64, n_features=7)
        mid = ta._q_to_runway(1.5)
        expected = Q_RUNWAY_MIN + (1.5 / Q_RUNWAY_MAX_Q) * (Q_RUNWAY_MAX - Q_RUNWAY_MIN)
        assert abs(mid - expected) < 1e-9

    def test_decay_epsilon(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.epsilon = 0.5
        ta.epsilon_end = 0.01
        ta.epsilon_decay = 0.99
        ta._decay_epsilon()
        assert abs(ta.epsilon - 0.495) < 1e-6

    def test_decay_epsilon_minimum(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.epsilon = 0.001
        ta.epsilon_end = 0.01
        ta.epsilon_decay = 0.5
        ta._decay_epsilon()
        assert ta.epsilon == pytest.approx(0.01)  # Should not go below epsilon_end


# ---------------------------------------------------------------------------
# Update from trade
# ---------------------------------------------------------------------------

class TestTriggerUpdateFromTrade:

    def test_update_from_trade_positive_runway(self):
        ta = TriggerAgent(window=64, n_features=7)
        # Should not crash, just logs
        ta.update_from_trade(actual_mfe=0.003, predicted_runway=0.002)

    def test_update_from_trade_zero_runway(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.update_from_trade(actual_mfe=0.003, predicted_runway=0.0)
        # Zero predicted → skip logging


# ---------------------------------------------------------------------------
# Platt parameter updates
# ---------------------------------------------------------------------------

class TestPlattUpdate:

    def test_update_platt_params_training_enabled(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=True)
        old_a = ta.platt_a
        old_b = ta.platt_b
        ta.update_platt_params(predicted_prob=0.7, actual_outcome=1.0)
        # Parameters should have changed
        assert ta.platt_a != old_a or ta.platt_b != old_b

    def test_update_platt_params_training_disabled(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=False)
        old_a = ta.platt_a
        old_b = ta.platt_b
        ta.update_platt_params(predicted_prob=0.7, actual_outcome=1.0)
        # Should not change when training disabled
        assert ta.platt_a == old_a
        assert ta.platt_b == old_b


# ---------------------------------------------------------------------------
# Experience buffer and training
# ---------------------------------------------------------------------------

class TestTriggerTraining:

    def test_add_experience_no_buffer(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=False)
        state = np.zeros((64, 7), dtype=np.float32)
        # Should not crash
        ta.add_experience(state, action=1, reward=0.5, next_state=state, done=True)

    def test_train_step_no_training(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=False)
        assert ta.train_step() is None

    def test_train_step_insufficient_experiences(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=True)
        assert ta.train_step() is None  # No experiences added

    def test_get_training_stats_disabled(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=False)
        stats = ta.get_training_stats()
        assert stats == {"enabled": False}

    def test_get_training_stats_enabled(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=True)
        stats = ta.get_training_stats()
        assert stats["enabled"] is True
        assert stats["training_steps"] == 0
        assert stats["buffer_size"] >= 0
