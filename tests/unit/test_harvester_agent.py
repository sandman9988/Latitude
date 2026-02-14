"""
Tests for HarvesterAgent - Exit specialist agent.

Covers:
- Initialization (default, with training, with friction)
- decide() - fallback strategy path
- _fallback_strategy() - stop loss, profit target, time stops
- _build_full_state() - market + position features
- quick_exit_check() - tick-level exit checks
- _init_exit_thresholds() / _get_timeframe_scale()
- get_friction_cost_pct()
- _softmax()
- update_from_trade()
- add_experience() / train_step() / get_training_stats()
"""

import logging
from unittest.mock import MagicMock

import numpy as np

rng = np.random.default_rng(42)

import pytest

from src.agents.harvester_agent import (
    HarvesterAgent,
    PCT_SCALE,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestHarvesterInit:

    def test_default_init(self):
        ha = HarvesterAgent(window=64, n_features=10)
        assert ha.window == 64
        assert ha.n_features == 10
        assert ha.use_torch is False
        assert ha.training_steps == 0

    def test_init_with_training(self):
        ha = HarvesterAgent(window=32, n_features=10, enable_training=True)
        assert ha.enable_training is True
        assert ha.buffer is not None

    def test_init_without_training(self):
        ha = HarvesterAgent(window=32, n_features=10, enable_training=False)
        assert ha.buffer is None

    def test_init_exit_thresholds(self):
        ha = HarvesterAgent(window=64, n_features=10)
        # Default thresholds should be set
        assert ha.profit_target_pct > 0
        assert ha.stop_loss_pct > 0
        assert ha.soft_time_stop_bars > 0
        assert ha.hard_time_stop_bars > ha.soft_time_stop_bars

    def test_init_with_friction_calculator(self):
        mock_fc = MagicMock()
        ha = HarvesterAgent(window=64, n_features=10, friction_calculator=mock_fc)
        assert ha.friction_calculator is mock_fc


# ---------------------------------------------------------------------------
# Timeframe scaling
# ---------------------------------------------------------------------------


class TestTimeframeScale:

    def test_m1_timeframe(self):
        ha = HarvesterAgent(window=64, n_features=10, timeframe="M1")
        assert ha._get_timeframe_scale() == pytest.approx(0.3)

    def test_m15_timeframe(self):
        ha = HarvesterAgent(window=64, n_features=10, timeframe="M15")
        assert ha._get_timeframe_scale() == pytest.approx(1.0)

    def test_h1_timeframe(self):
        ha = HarvesterAgent(window=64, n_features=10, timeframe="H1")
        assert ha._get_timeframe_scale() == pytest.approx(2.0)

    def test_unknown_timeframe_defaults_to_1(self):
        ha = HarvesterAgent(window=64, n_features=10, timeframe="W1")
        assert ha._get_timeframe_scale() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fallback strategy
# ---------------------------------------------------------------------------


class TestFallbackStrategy:

    def test_stop_loss_triggered(self):
        ha = HarvesterAgent(window=64, n_features=10)
        entry_price = 100.0
        mfe = entry_price * 0.001
        mae = entry_price * ha.stop_loss_pct / PCT_SCALE * 2  # Well above stop loss
        action = ha._fallback_strategy(mfe, mae, ticks_held=5, entry_price=entry_price)
        assert action == 1  # CLOSE

    def test_profit_target_hit(self):
        ha = HarvesterAgent(window=64, n_features=10)
        entry_price = 100.0
        # Set MFE well above profit target + friction
        mfe = entry_price * (ha.profit_target_pct + 2.0) / PCT_SCALE
        mae = 0.0
        action = ha._fallback_strategy(mfe, mae, ticks_held=5, entry_price=entry_price)
        assert action == 1  # CLOSE

    def test_hold_when_no_conditions_met(self):
        ha = HarvesterAgent(window=64, n_features=10)
        entry_price = 100.0
        mfe = entry_price * 0.0005  # Very small profit (0.05%), below all exit thresholds
        mae = entry_price * 0.0003  # Very small drawdown, below stop
        action = ha._fallback_strategy(mfe, mae, ticks_held=5, entry_price=entry_price)
        assert action == 0  # HOLD

    def test_hard_time_stop(self):
        ha = HarvesterAgent(window=64, n_features=10)
        entry_price = 100.0
        mfe = entry_price * 0.001
        mae = entry_price * 0.001
        action = ha._fallback_strategy(mfe, mae, ticks_held=ha.hard_time_stop_bars + 1, entry_price=entry_price)
        assert action == 1  # CLOSE

    def test_soft_time_stop_with_profit(self):
        ha = HarvesterAgent(window=64, n_features=10)
        entry_price = 100.0
        # MFE large enough so net profit > 0 after friction
        mfe = entry_price * 0.01  # 1% MFE
        mae = entry_price * 0.001
        ticks = ha.soft_time_stop_bars + 1
        action = ha._fallback_strategy(mfe, mae, ticks_held=ticks, entry_price=entry_price)
        assert action == 1  # CLOSE (net positive after friction)

    def test_zero_entry_price_holds(self):
        ha = HarvesterAgent(window=64, n_features=10)
        action = ha._fallback_strategy(mfe=10, mae=5, ticks_held=10, entry_price=0.0)
        assert action == 0  # HOLD (can't evaluate)


# ---------------------------------------------------------------------------
# Build full state
# ---------------------------------------------------------------------------


class TestBuildFullState:

    def test_basic_state_concatenation(self):
        ha = HarvesterAgent(window=64, n_features=10)
        market_state = rng.standard_normal((64, 7)).astype(np.float32)
        full_state = ha._build_full_state(market_state, mfe=50.0, mae=10.0, ticks_held=30, entry_price=1000.0)
        assert full_state.shape == (64, 10)  # 7 market + 3 position

    def test_position_features_normalized(self):
        ha = HarvesterAgent(window=64, n_features=10)
        market_state = np.zeros((64, 7), dtype=np.float32)
        entry_price = 1000.0
        mfe = 50.0
        mae = 10.0
        ticks = 50

        full_state = ha._build_full_state(market_state, mfe, mae, ticks, entry_price)
        # Check position features in last 3 columns
        mfe_norm = (mfe / entry_price) * PCT_SCALE
        mae_norm = (mae / entry_price) * PCT_SCALE
        ticks_norm = min(ticks / 100.0, 1.0)

        assert abs(full_state[0, 7] - mfe_norm) < 1e-4
        assert abs(full_state[0, 8] - mae_norm) < 1e-4
        assert abs(full_state[0, 9] - ticks_norm) < 1e-4

    def test_zero_entry_price_fallback(self):
        ha = HarvesterAgent(window=64, n_features=10)
        market_state = np.zeros((64, 7), dtype=np.float32)
        # entry_price <= 0 → falls back to 1.0
        full_state = ha._build_full_state(market_state, mfe=1.0, mae=0.5, ticks_held=10, entry_price=0.0)
        assert full_state.shape == (64, 10)
        # MFE norm = (1.0/1.0) * 100 = 100.0
        assert abs(full_state[0, 7] - 100.0) < 1e-4


# ---------------------------------------------------------------------------
# Quick exit check
# ---------------------------------------------------------------------------


class TestQuickExitCheck:

    def test_stop_loss_triggers_exit(self):
        ha = HarvesterAgent(window=64, n_features=10)
        entry_price = 100.0
        mfe = 0.0
        mae = entry_price * ha.stop_loss_pct / PCT_SCALE * 2  # Well above stop
        assert ha.quick_exit_check(mfe, mae, entry_price, 95.0, direction=1) is True

    def test_profit_target_triggers_exit(self):
        ha = HarvesterAgent(window=64, n_features=10)
        entry_price = 100.0
        mfe = entry_price * (ha.profit_target_pct + 2.0) / PCT_SCALE  # Well above target
        mae = 0.0
        assert ha.quick_exit_check(mfe, mae, entry_price, 105.0, direction=1) is True

    def test_no_exit_conditions(self):
        ha = HarvesterAgent(window=64, n_features=10)
        entry_price = 100.0
        mfe = entry_price * 0.001
        mae = entry_price * 0.001
        assert ha.quick_exit_check(mfe, mae, entry_price, 100.1, direction=1) is False

    def test_zero_entry_price(self):
        ha = HarvesterAgent(window=64, n_features=10)
        assert ha.quick_exit_check(1.0, 1.0, 0.0, 100.0, direction=1) is False


# ---------------------------------------------------------------------------
# Friction cost
# ---------------------------------------------------------------------------


class TestFrictionCost:

    def test_default_friction_no_calculator(self):
        ha = HarvesterAgent(window=64, n_features=10)
        pct = ha.get_friction_cost_pct(entry_price=100000.0)
        assert abs(pct - 0.0015) < 1e-6  # Default 0.15%

    def test_zero_entry_price_returns_default(self):
        ha = HarvesterAgent(window=64, n_features=10)
        pct = ha.get_friction_cost_pct(entry_price=0.0)
        assert abs(pct - 0.0015) < 1e-6

    def test_with_friction_calculator(self):
        mock_fc = MagicMock()
        mock_costs = MagicMock()
        mock_costs.contract_size = 100
        mock_fc.costs = mock_costs
        mock_fc.calculate_total_friction.return_value = {
            "total": 5.0,
            "total_pips": 5.0,
            "swap": 0.0,
        }
        ha = HarvesterAgent(window=64, n_features=10, friction_calculator=mock_fc)
        pct = ha.get_friction_cost_pct(entry_price=1000.0, quantity=0.1)
        # position_value = 0.1 * 1000.0 * 100 = 10000
        # friction_pct = 5.0 / 10000 = 0.0005
        assert abs(pct - 0.0005) < 1e-6


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


class TestHarvesterSoftmax:

    def test_softmax_sums_to_one(self):
        ha = HarvesterAgent(window=64, n_features=10)
        probs = ha._softmax(np.array([1.0, 2.0]))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_softmax_higher_value_wins(self):
        ha = HarvesterAgent(window=64, n_features=10)
        probs = ha._softmax(np.array([0.0, 5.0]))
        assert probs[1] > probs[0]


# ---------------------------------------------------------------------------
# Decide (integration)
# ---------------------------------------------------------------------------


class TestHarvesterDecide:

    def test_decide_fallback_mode(self):
        ha = HarvesterAgent(window=64, n_features=10)
        market_state = rng.standard_normal((64, 7)).astype(np.float32)
        entry_price = 100000.0
        mfe = entry_price * 0.001
        mae = entry_price * 0.001
        action, conf = ha.decide(market_state, mfe, mae, ticks_held=10, entry_price=entry_price, direction=1)
        assert action in [0, 1]
        assert 0.0 <= conf <= 1.0

    def test_decide_with_ddqn(self):
        ha = HarvesterAgent(window=64, n_features=10)
        # DDQN is always initialized in numpy mode
        market_state = rng.standard_normal((64, 7)).astype(np.float32)
        entry_price = 100000.0
        action, conf = ha.decide(market_state, mfe=100.0, mae=50.0, ticks_held=10, entry_price=entry_price, direction=1)
        assert action in [0, 1]
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Update from trade
# ---------------------------------------------------------------------------


class TestHarvesterUpdateFromTrade:

    def test_update_no_param_manager(self):
        ha = HarvesterAgent(window=64, n_features=10)
        ha.param_manager = None
        # Should not crash
        ha.update_from_trade(capture_ratio=0.8, was_wtl=False)

    def test_update_wtl_trade(self):
        mock_pm = MagicMock()
        mock_pm.update.return_value = 2.0
        ha = HarvesterAgent(window=64, n_features=10, param_manager=mock_pm)
        ha.update_from_trade(capture_ratio=0.0, was_wtl=True)
        mock_pm.update.assert_called_once()
        mock_pm.save.assert_called_once()

    def test_update_high_capture(self):
        mock_pm = MagicMock()
        mock_pm.update.return_value = 2.0
        ha = HarvesterAgent(window=64, n_features=10, param_manager=mock_pm)
        ha.update_from_trade(capture_ratio=0.9, was_wtl=False)
        mock_pm.update.assert_called_once()

    def test_update_low_capture(self):
        mock_pm = MagicMock()
        mock_pm.update.return_value = 3.0
        ha = HarvesterAgent(window=64, n_features=10, param_manager=mock_pm)
        ha.update_from_trade(capture_ratio=0.1, was_wtl=False)
        mock_pm.update.assert_called_once()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestHarvesterTraining:

    def test_add_experience_no_buffer(self):
        ha = HarvesterAgent(window=64, n_features=10, enable_training=False)
        state = np.zeros((64, 10), dtype=np.float32)
        ha.add_experience(state, action=0, reward=0.1, next_state=state, done=False)

    def test_train_step_disabled(self):
        ha = HarvesterAgent(window=64, n_features=10, enable_training=False)
        assert ha.train_step() is None

    def test_train_step_insufficient_data(self):
        ha = HarvesterAgent(window=64, n_features=10, enable_training=True)
        assert ha.train_step() is None

    def test_get_training_stats_disabled(self):
        ha = HarvesterAgent(window=64, n_features=10, enable_training=False)
        stats = ha.get_training_stats()
        assert stats == {"enabled": False}

    def test_get_training_stats_enabled(self):
        ha = HarvesterAgent(window=64, n_features=10, enable_training=True)
        stats = ha.get_training_stats()
        assert stats["enabled"] is True
        assert "buffer_size" in stats
        assert "training_steps" in stats
