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
# ParameterStaleness division guard
# ══════════════════════════════════════════════════════════════════════════════

class TestParameterStalenessGuard:
    def test_empty_regime_history_returns_none(self):
        from src.core.parameter_staleness import ParameterStalenessDetector
        det = ParameterStalenessDetector.__new__(ParameterStalenessDetector)
        det.regime_stability_bars = 50
        det.bars_for_baseline = 5
        # Fill snapshots with mocks that have a regime attribute
        from collections import deque
        mock_snap = MagicMock()
        mock_snap.regime = "normal"
        det.snapshots = deque([mock_snap] * 10)
        det.regime_history = deque()  # empty!

        result = det._check_regime_shift()
        assert result is None  # Should not crash


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
