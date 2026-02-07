"""
Tests for production-critical RiskManager code paths (Tier 1 + Tier 2).

Covers:
  Tier 1:
    - VaR zero/negative → entry rejected (lines 301-304)
    - Position size zero → entry rejected (lines 321-324)
    - Unknown exit_type → exit rejected (lines 461-468)
    - Exit volume <= 0 → exit rejected (lines 477-480)
    - Circuit breaker → 25% budget reduction (lines 661-668)

  Tier 2:
    - No position to close → exit rejected (lines 407-408)
    - Negative risk budget rejected (line 556)
    - Good performance → increase budget (lines 672-679)
    - RL threshold recommendations (lines 1146-1166)
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.risk.circuit_breakers import CircuitBreakerManager
from src.risk.risk_manager import RiskManager, EntryValidation, ExitValidation
from src.risk.var_estimator import VaREstimator, RegimeType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def var_estimator():
    """VaREstimator pre-loaded with sample data."""
    est = VaREstimator(window=100, confidence=0.95)
    rng = np.random.default_rng(42)
    for ret in rng.normal(0.0, 0.01, 100):
        est.update_return(ret)
    return est


@pytest.fixture()
def circuit_breakers():
    return CircuitBreakerManager()


@pytest.fixture()
def rm(circuit_breakers, var_estimator):
    """Standard RiskManager for most tests."""
    return RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=100.0,
        max_position_size=1.0,
        min_confidence_entry=0.6,
        min_confidence_exit=0.5,
    )


# =========================================================================
# TIER 1 — PRODUCTION-CRITICAL
# =========================================================================


class TestVarZeroRejectsEntry:
    """Lines 301-304: VaR <= 0 → entry rejected."""

    def test_var_returns_zero(self, rm):
        rm.var_estimator.estimate_var = MagicMock(return_value=0.0)
        result = rm.validate_entry(action=1, confidence=0.9, current_position=0.0)
        assert result.approved is False
        assert result.qty == pytest.approx(0.0)
        assert "var" in result.reason.lower() or "zero" in result.reason.lower()

    def test_var_returns_negative(self, rm):
        rm.var_estimator.estimate_var = MagicMock(return_value=-0.5)
        result = rm.validate_entry(action=1, confidence=0.9, current_position=0.0)
        assert result.approved is False
        assert result.qty == pytest.approx(0.0)

    def test_rejected_increments_counter(self, rm):
        rm.var_estimator.estimate_var = MagicMock(return_value=0.0)
        before = rm.entries_rejected
        rm.validate_entry(action=1, confidence=0.9, current_position=0.0)
        assert rm.entries_rejected == before + 1


class TestPositionSizeZeroRejectsEntry:
    """Lines 321-324: qty <= 0 from position_size_from_var → entry rejected."""

    def test_position_size_returns_zero(self, rm):
        rm.var_estimator.estimate_var = MagicMock(return_value=0.01)
        with patch("src.risk.var_estimator.position_size_from_var", return_value=0.0):
            result = rm.validate_entry(action=1, confidence=0.9, current_position=0.0)
        assert result.approved is False
        assert "zero" in result.reason.lower() or "negative" in result.reason.lower()

    def test_position_size_returns_negative(self, rm):
        rm.var_estimator.estimate_var = MagicMock(return_value=0.01)
        with patch("src.risk.var_estimator.position_size_from_var", return_value=-0.1):
            result = rm.validate_entry(action=1, confidence=0.9, current_position=0.0)
        assert result.approved is False

    def test_rejected_increments_counter(self, rm):
        rm.var_estimator.estimate_var = MagicMock(return_value=0.01)
        before = rm.entries_rejected
        with patch("src.risk.var_estimator.position_size_from_var", return_value=0.0):
            rm.validate_entry(action=1, confidence=0.9, current_position=0.0)
        assert rm.entries_rejected == before + 1


class TestUnknownExitTypeRejected:
    """Lines 461-468: Unknown exit_type → exit rejected."""

    def test_unknown_exit_type(self, rm):
        result = rm.validate_exit(
            action=1,
            exit_type="UNKNOWN",
            current_position=1.0,
        )
        assert result.approved is False
        assert "unknown" in result.reason.lower()

    def test_bogus_exit_type(self, rm):
        result = rm.validate_exit(
            action=1,
            exit_type="MARKET_ON_CLOSE",
            current_position=1.0,
        )
        assert result.approved is False

    def test_unknown_exit_type_increments_counter(self, rm):
        before = rm.exits_rejected
        rm.validate_exit(action=1, exit_type="INVALID", current_position=1.0)
        assert rm.exits_rejected == before + 1


class TestExitVolumeZeroRejected:
    """Lines 477-480: volume <= 0 → exit rejected."""

    def test_exit_full_with_zero_position(self, rm):
        """FULL exit with 0 position should hit the 'no position' guard first."""
        result = rm.validate_exit(
            action=1,
            exit_type="FULL",
            current_position=0.0,
            min_position_size=0.001,
        )
        assert result.approved is False

    def test_tiny_position_full_exit_zero_volume_rejected(self, rm):
        """Positions < 0.01 lots produce volume=0 after int() conversion.

        The post-conversion guard catches this and rejects the exit,
        preventing a zero-volume close order from reaching the broker.
        """
        result = rm.validate_exit(
            action=1,
            exit_type="FULL",
            current_position=0.002,
            min_position_size=0.001,
        )
        assert result.approved is False
        assert result.volume == 0
        assert "round" in result.reason.lower()


class TestCircuitBreakerBudgetReduction:
    """Lines 661-668: Circuit breaker active → 25% budget cut."""

    def test_breaker_tripped_reduces_budget(self, rm):
        initial = rm.risk_budget_usd
        rm.circuit_breakers.is_any_tripped = MagicMock(return_value=True)
        rm._consider_risk_adaptation(equity=10000.0, win_rate=0.50)
        assert rm.risk_budget_usd == pytest.approx(initial * 0.75)

    def test_breaker_tripped_takes_precedence_over_good_performance(self, rm):
        """Even with excellent performance, tripped breaker triggers reduction."""
        initial = rm.risk_budget_usd
        rm.circuit_breakers.is_any_tripped = MagicMock(return_value=True)
        rm.peak_equity = 8000.0  # Equity growing
        rm._consider_risk_adaptation(equity=12000.0, win_rate=0.80)
        assert rm.risk_budget_usd < initial

    def test_successive_breaker_reductions_compound(self, rm):
        """Multiple trigger events compound the reduction."""
        initial = rm.risk_budget_usd
        rm.circuit_breakers.is_any_tripped = MagicMock(return_value=True)
        rm._consider_risk_adaptation(equity=10000.0, win_rate=0.50)
        rm._consider_risk_adaptation(equity=10000.0, win_rate=0.50)
        assert rm.risk_budget_usd == pytest.approx(initial * 0.75 * 0.75)


# =========================================================================
# TIER 2 — IMPORTANT
# =========================================================================


class TestNoPositionToClose:
    """Lines 407-408: abs(current_position) < min_position_size → rejected."""

    def test_no_position_rejects_exit(self, rm):
        result = rm.validate_exit(
            action=1,
            exit_type="FULL",
            current_position=0.0,
            min_position_size=0.001,
        )
        assert result.approved is False
        assert "no position" in result.reason.lower()

    def test_dust_position_rejects_exit(self, rm):
        result = rm.validate_exit(
            action=1,
            exit_type="FULL",
            current_position=0.0005,
            min_position_size=0.001,
        )
        assert result.approved is False


class TestNegativeRiskBudgetRejected:
    """Line 556: Negative budget rejected in update_risk_budget."""

    def test_negative_budget_ignored(self, rm):
        original = rm.risk_budget_usd
        rm.update_risk_budget(-50.0)
        assert rm.risk_budget_usd == original

    def test_zero_budget_ignored(self, rm):
        original = rm.risk_budget_usd
        rm.update_risk_budget(0.0)
        assert rm.risk_budget_usd == original

    def test_positive_budget_accepted(self, rm):
        rm.update_risk_budget(200.0)
        assert rm.risk_budget_usd == pytest.approx(200.0)


class TestGoodPerformanceIncreasesBudget:
    """Lines 672-679: win_rate > 0.55 and equity growth > 5% → increase budget."""

    def test_strong_performance_increases_budget(self, rm):
        rm.peak_equity = 10000.0
        initial = rm.risk_budget_usd
        rm.circuit_breakers.is_any_tripped = MagicMock(return_value=False)
        # equity > peak * 1.05, win_rate > 0.55
        rm._consider_risk_adaptation(equity=10600.0, win_rate=0.60)
        assert rm.risk_budget_usd > initial

    def test_budget_capped_at_1_5x_initial(self, rm):
        rm.peak_equity = 10000.0
        rm.initial_risk_budget = 100.0
        rm.risk_budget_usd = 140.0  # Already near cap
        rm.circuit_breakers.is_any_tripped = MagicMock(return_value=False)
        rm._consider_risk_adaptation(equity=11000.0, win_rate=0.70)
        assert rm.risk_budget_usd <= rm.initial_risk_budget * 1.5

    def test_poor_performance_decreases_budget(self, rm):
        rm.peak_equity = 10000.0
        initial = rm.risk_budget_usd
        rm.circuit_breakers.is_any_tripped = MagicMock(return_value=False)
        # equity < peak * 0.9, win_rate < 0.45
        rm._consider_risk_adaptation(equity=8500.0, win_rate=0.40)
        assert rm.risk_budget_usd < initial

    def test_budget_floored_at_0_5x_initial(self, rm):
        rm.peak_equity = 10000.0
        rm.initial_risk_budget = 100.0
        rm.risk_budget_usd = 55.0  # Near floor
        rm.circuit_breakers.is_any_tripped = MagicMock(return_value=False)
        rm._consider_risk_adaptation(equity=7000.0, win_rate=0.30)
        assert rm.risk_budget_usd >= rm.initial_risk_budget * 0.5


class TestRLThresholdRecommendations:
    """Lines 1146-1166: Q-table aggregation → threshold recommendations."""

    def test_insufficient_learning_returns_defaults(self, rm):
        rm.q_table = {}  # Empty
        rec = rm.get_rl_recommended_thresholds()
        assert rec["confidence"] == pytest.approx(0.0)
        assert rec["entry_threshold"] == rm.min_confidence_entry
        assert rec["exit_threshold"] == rm.min_confidence_exit
        assert "insufficient" in rec["reason"].lower()

    def test_sufficient_learning_produces_recommendations(self, rm):
        # Populate Q-table with 15 states (> 10 threshold)
        for i in range(15):
            rm.q_table[(i, 0)] = {
                "lower_threshold": 0.5,
                "keep_threshold": 0.1,
                "raise_threshold": -0.3,
            }
        rec = rm.get_rl_recommended_thresholds()
        assert rec["confidence"] > 0.0
        assert 0.5 <= rec["entry_threshold"] <= 0.9
        assert 0.4 <= rec["exit_threshold"] <= 0.8

    def test_raise_threshold_recommendation(self, rm):
        for i in range(15):
            rm.q_table[(i, 0)] = {
                "lower_threshold": -0.5,
                "keep_threshold": 0.0,
                "raise_threshold": 1.0,
            }
        rec = rm.get_rl_recommended_thresholds()
        assert "raise" in rec["reason"].lower()
        # The entry threshold should be higher than default
        assert rec["entry_threshold"] >= rm.min_confidence_entry

    def test_lower_threshold_recommendation(self, rm):
        for i in range(15):
            rm.q_table[(i, 0)] = {
                "lower_threshold": 1.0,
                "keep_threshold": -0.5,
                "raise_threshold": -1.0,
            }
        rec = rm.get_rl_recommended_thresholds()
        assert "lower" in rec["reason"].lower()

    def test_thresholds_bounded(self, rm):
        """Ensure thresholds stay within [0.5, 0.9] entry and [0.4, 0.8] exit."""
        rm.min_confidence_entry = 0.9  # Push to upper bound
        for i in range(15):
            rm.q_table[(i, 0)] = {
                "lower_threshold": -1.0,
                "keep_threshold": -1.0,
                "raise_threshold": 5.0,
            }
        rec = rm.get_rl_recommended_thresholds()
        assert rec["entry_threshold"] <= 0.9
        assert rec["exit_threshold"] <= 0.8
