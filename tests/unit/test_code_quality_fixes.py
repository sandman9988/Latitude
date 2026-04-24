"""
Tests for code quality fixes: division guards, silent-except logging,
named constants, and edge-case protection across risk/utility modules.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# RiskManager division guards & named constants
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskManagerCapitalAllocation:
    """Test capital allocation division safety."""

    @pytest.fixture
    def risk_mgr(self):
        from src.risk.risk_manager import RiskManager
        from src.risk.circuit_breakers import CircuitBreakerManager
        from src.risk.var_estimator import VaREstimator
        cb = CircuitBreakerManager()
        var = VaREstimator()
        rm = RiskManager(circuit_breakers=cb, var_estimator=var)
        rm.correlation_matrix = None
        rm.returns_history = {}
        return rm

    def test_empty_symbols_returns_empty_dict(self, risk_mgr):
        """allocate_capital_by_correlation([]) must not crash."""
        result = risk_mgr.allocate_capital_by_correlation([], 100_000.0)
        assert result == {}

    def test_single_symbol_equal_allocation(self, risk_mgr):
        result = risk_mgr.allocate_capital_by_correlation(["XAUUSD"], 100_000.0)
        assert result == {"XAUUSD": pytest.approx(100_000.0)}

    def test_multiple_symbols_equal_allocation(self, risk_mgr):
        result = risk_mgr.allocate_capital_by_correlation(["XAUUSD", "EURUSD"], 100_000.0)
        assert result["XAUUSD"] == pytest.approx(50_000.0)
        assert result["EURUSD"] == pytest.approx(50_000.0)

    def test_zero_diversification_scores_fallback(self, risk_mgr):
        """When all diversification scores are 0, must not ZeroDivisionError."""
        from src.risk.risk_manager import MIN_SYMBOLS_FOR_ALLOCATION

        symbols = [f"SYM{i}" for i in range(max(5, MIN_SYMBOLS_FOR_ALLOCATION + 1))]
        risk_mgr.correlation_matrix = np.ones((len(symbols), len(symbols)))
        risk_mgr.returns_history = {s: list(range(100)) for s in symbols}
        # Make _compute_diversification_scores return all zeros
        risk_mgr._compute_diversification_scores = lambda syms: {s: 0.0 for s in syms}

        result = risk_mgr.allocate_capital_by_correlation(symbols, 100_000.0)
        assert len(result) == len(symbols)
        # Should be equal allocation fallback
        expected = 100_000.0 / len(symbols)
        for v in result.values():
            assert v == pytest.approx(expected)


class TestRiskManagerNamedConstants:
    """Verify named constants replaced magic numbers."""

    def test_circuit_breaker_factor_exists(self):
        from src.risk.risk_manager import CIRCUIT_BREAKER_BUDGET_FACTOR
        assert 0 < CIRCUIT_BREAKER_BUDGET_FACTOR < 1

    def test_uncorrelated_reserve_exists(self):
        from src.risk.risk_manager import UNCORRELATED_RESERVE_FRACTION
        assert 0 < UNCORRELATED_RESERVE_FRACTION < 1

    def test_confidence_epsilon_exists(self):
        from src.risk.risk_manager import CONFIDENCE_EPSILON
        assert CONFIDENCE_EPSILON > 0


# ══════════════════════════════════════════════════════════════════════════════
# SumTree batch_size guard
# ══════════════════════════════════════════════════════════════════════════════

class TestSumTreeBatchSizeGuard:
    def test_zero_batch_size_raises(self):
        from src.utils.sum_tree import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100, state_dim=4)
        # Add one experience
        buf.add(np.zeros(4), 0, 1.0, np.zeros(4), False)
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            buf.sample(batch_size=0)

    def test_normal_batch_size_works(self):
        from src.utils.sum_tree import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100, state_dim=4)
        for i in range(10):
            buf.add(np.ones(4) * i, 0, 1.0, np.ones(4) * (i + 1), False)
        result = buf.sample(batch_size=5)
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# FrictionCosts silent except → logged
# ══════════════════════════════════════════════════════════════════════════════

class TestFrictionCostsLogging:
    def test_invalid_digits_logs_warning(self, caplog):
        from src.risk.friction_costs import FrictionCalculator

        model = FrictionCalculator()
        model.costs.digits = "not_a_number"  # will cause int() to fail

        with caplog.at_level(logging.WARNING):
            model._refresh_derived_costs()

        assert any("Failed to derive tick_size" in r.message for r in caplog.records)


# ══════════════════════════════════════════════════════════════════════════════
# _get_live_qty silent except → logged
# ══════════════════════════════════════════════════════════════════════════════

class TestGetLiveQtyLogging:
    def test_trade_manager_exception_logs_warning(self):
        """When trade_manager.get_position() throws, should log warning."""
        from src.core.ctrader_ddqn_paper import CTraderFixApp, LOG

        bot = CTraderFixApp.__new__(CTraderFixApp)
        bot.qty = 0.01
        bot.trade_integration = MagicMock()
        bot.trade_integration.trade_manager.get_position.side_effect = RuntimeError("broker disconnect")

        with patch.object(LOG, "warning") as mock_warn:
            result = bot._get_live_qty()

        assert result == 0.01  # falls back to self.qty
        mock_warn.assert_called_once()
        assert "Failed to read live qty" in mock_warn.call_args[0][0]


class TestHarvesterCloseRewardUnits:
    def test_close_reward_uses_point_pnl_against_point_mfe(self):
        from src.core.ctrader_ddqn_paper import CTraderFixApp

        bot = CTraderFixApp.__new__(CTraderFixApp)
        bot.vol_cap = 0.05
        bot.vpin_z_threshold = 2.5
        bot.entry_var = 0.0
        bot.entry_vpin_z = 0.0
        bot.prev_harvester_state = np.array([1.0, 0.0], dtype=float)
        bot.prev_exit_action = 1
        bot.prev_mfe = 0.0
        bot.prev_mae = 0.0
        bot.bars = []
        bot.entry_action = 1
        bot.was_exploration_entry = False
        bot.entry_imbalance = 0.0

        class RewardSpy:
            def __init__(self):
                self.exit_pnl = None

            def calculate_harvester_reward(self, **kwargs):
                self.exit_pnl = kwargs["exit_pnl"]
                return {"harvester_reward": -1.25}

        reward_spy = RewardSpy()
        bot.reward_shaper = reward_spy
        harvester = MagicMock()
        harvester.last_state = np.array([0.0, 1.0], dtype=float)
        harvester.buffer = None
        bot.policy = MagicMock()
        bot.policy.harvester = harvester
        bot.policy.current_regime = "TEST"
        bot.policy.add_harvester_experience = MagicMock()
        bot._bar_cache = MagicMock()

        summary = {
            "mfe": 0.004,
            "mae": 0.001,
            "winner_to_loser": False,
            "bars_held": 5,
            "bars_from_mfe_to_exit": 2,
            "exit_time": "2026-04-23T00:00:00+00:00",
        }

        bot._add_harvester_experience_for_close(
            summary=summary,
            pnl=350.0,
            entry_price=1.1000,
            exit_price=1.1007,
            pnl_pts=0.0007,
            shaped_rewards={},
            trigger_reward=0.0,
        )

        assert reward_spy.exit_pnl == pytest.approx(0.0007)
        bot.policy.add_harvester_experience.assert_called_once()
        _, kwargs = bot.policy.add_harvester_experience.call_args
        assert kwargs["reward"] == pytest.approx(-1.25)


# ══════════════════════════════════════════════════════════════════════════════
# DDQNNetwork optimizer load → logged
# ══════════════════════════════════════════════════════════════════════════════

class TestDDQNOptimizerLoadLogging:
    def test_bad_optimizer_state_logs_warning(self, tmp_path, caplog):
        import torch
        from src.core.ddqn_network import DDQNNetwork

        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        filepath = str(tmp_path / "weights.pt")
        # Save with bad optimizer state
        torch.save(
            {
                "online": net.online.state_dict(),
                "target": net.target.state_dict(),
                "optimizer": {"bad": "data"},  # will fail load_state_dict
                "training_steps": 5,
            },
            filepath,
        )

        net2 = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        with caplog.at_level(logging.WARNING):
            net2.load_weights(filepath)

        # Should still load model weights successfully
        assert net2.training_steps == 5
        # Should log warning about optimizer
        assert any("Skipping optimizer" in r.message for r in caplog.records)


class TestLiveRiskTuner:
    def test_updates_dynamic_floors_from_risk_manager(self):
        from src.core.ctrader_ddqn_paper import CTraderFixApp

        bot = CTraderFixApp.__new__(CTraderFixApp)
        bot.entry_confidence = 0.72
        bot._last_trigger_conf = 0.72
        bot._last_harvester_conf = 0.61
        bot._last_exit_confidence = 0.61
        bot._entry_conf_dynamic_floor = 0.60
        bot._exit_conf_dynamic_floor = 0.45
        bot.symbol = "XAUUSD"
        bot.timeframe_label = "M5"
        bot.broker = "default"
        bot.performance = MagicMock()
        bot.performance.total_trades = 10
        bot._estimate_account_equity = MagicMock(return_value=10025.0)
        bot.param_manager = MagicMock()
        bot.risk_manager = MagicMock()
        bot.risk_manager.get_rl_recommended_thresholds.return_value = {
            "entry_threshold": 0.66,
            "exit_threshold": 0.50,
            "reason": "test",
        }

        bot._update_risk_feedback_thresholds(pnl=15.0)

        assert bot._entry_conf_dynamic_floor == pytest.approx(0.66)
        assert bot._exit_conf_dynamic_floor == pytest.approx(0.50)
        assert bot.risk_manager.update_decision_outcome.call_count == 3
        bot.risk_manager.on_trade_complete.assert_called_once()
        assert bot.param_manager.set_value.call_count >= 2

    def test_exit_guard_blocks_close_below_dynamic_floor(self):
        from src.core.ctrader_ddqn_paper import CTraderFixApp

        bot = CTraderFixApp.__new__(CTraderFixApp)
        bot._exit_conf_dynamic_floor = 0.55
        bot._obc_max_loss_force_close = MagicMock(return_value=False)
        bot._pending_closes = set()
        bot.bars = []
        bot.policy = MagicMock()
        bot.policy.decide_exit.return_value = (1, 0.40)  # CLOSE, low confidence

        action, conf, already = bot._obc_get_exit_action(
            price=100.0, imbalance=0.0, depth_ratio=1.0, vpin_zscore=0.0, event_features={}
        )

        assert action == 0
        assert conf == pytest.approx(0.40)
        assert already is False

    def test_dynamic_entry_floor_is_sample_gated_and_caps_rl_floor(self):
        from src.core.ctrader_ddqn_paper import CTraderFixApp

        bot = CTraderFixApp.__new__(CTraderFixApp)
        bot.performance = MagicMock()
        bot.performance.total_trades = 5
        bot._conf_calib_err_ema = 0.90
        bot._runway_accuracy_ema = 0.20
        bot._entry_conf_dynamic_floor = 0.90
        bot._lp_get = lambda name, default: {
            "entry_guard_min_trade_samples": 40.0,
            "entry_guard_rl_floor_extra_cap": 0.10,
        }.get(name, default)

        dyn, meta = bot._compute_dynamic_entry_floor(0.70)

        assert dyn == pytest.approx(0.80)
        assert meta["uplift"] == pytest.approx(0.0)
        assert meta["runway_penalty"] == pytest.approx(0.0)
        assert meta["total_trades"] == 5

    def test_dynamic_entry_floor_applies_capped_uplifts_after_min_samples(self):
        from src.core.ctrader_ddqn_paper import CTraderFixApp

        bot = CTraderFixApp.__new__(CTraderFixApp)
        bot.performance = MagicMock()
        bot.performance.total_trades = 120
        bot._conf_calib_err_ema = 0.50
        bot._runway_accuracy_ema = 0.40
        bot._entry_conf_dynamic_floor = 0.75
        bot._lp_get = lambda name, default: {
            "entry_guard_min_trade_samples": 40.0,
            "entry_guard_calib_err_start": 0.30,
            "entry_guard_calib_uplift_cap": 0.08,
            "entry_guard_runway_acc_target": 0.60,
            "entry_guard_runway_penalty_cap": 0.05,
            "entry_guard_rl_floor_extra_cap": 0.10,
        }.get(name, default)

        dyn, meta = bot._compute_dynamic_entry_floor(0.70)

        assert meta["uplift"] == pytest.approx(0.08)
        assert meta["runway_penalty"] == pytest.approx(0.05)
        assert dyn == pytest.approx(0.83)
