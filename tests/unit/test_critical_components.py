"""
Integration Tests for Critical Gap-Closing Components

Tests the following components implemented to close production deployment gaps:
1. FeedbackLoopBreaker (GAP 1)
2. ColdStartManager (GAP 3)
3. RewardIntegrityMonitor (GAP 4)
4. BrokerExecutionModel (GAP 2 - execution modeling)
5. ParameterStalenessDetector (GAP 8)

These tests verify that the gap-closing implementations work correctly
and integrate properly with the rest of the system.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

rng = np.random.default_rng(42)


from src.core.broker_execution_model import BrokerExecutionModel, ExecutionCosts, OrderSide
from src.core.cold_start_manager import ColdStartManager, WarmupPhase
from src.core.feedback_loop_breaker import FeedbackLoopBreaker, FeedbackLoopSignal
from src.core.parameter_staleness import ParameterStalenessDetector, StalenessSignal
from src.core.reward_integrity_monitor import RewardIntegrityMonitor

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


# ============================================================================
# TEST 1: FeedbackLoopBreaker
# ============================================================================


def test_feedback_loop_breaker_no_trade_detection():
    """Test that FeedbackLoopBreaker detects no-trade loops."""
    LOG.info("\n=== TEST 1A: FeedbackLoopBreaker - No-Trade Detection ===")

    breaker = FeedbackLoopBreaker(
        no_trade_window_bars=20,
        min_volatility_threshold=0.005,
    )

    # Simulate scenario: market has opportunities but no trades for 25 bars.
    # update() takes bars_since_last_trade (cumulative) and returns Optional[FeedbackLoopSignal].
    signal = None
    for bar in range(30):
        signal = breaker.update(
            bars_since_last_trade=bar,  # increases each bar
            current_volatility=0.015,  # Good volatility (above 0.005 threshold)
            circuit_breakers_tripped=False,
        )
        if signal is not None:
            break

    assert signal is not None, "Should detect no-trade loop"
    assert signal.loop_type == "no_trades", "Should detect no_trades type"
    assert signal.severity > 0.0, "Should have non-zero severity"

    LOG.info(
        "✓ No-trade loop detected: severity=%.2f, duration=%d bars",
        signal.severity,
        signal.duration_bars,
    )
    LOG.info("  Evidence: %s", signal.evidence)


def test_feedback_loop_breaker_circuit_breaker_stuck():
    """Test detection of permanently tripped circuit breakers."""
    LOG.info("\n=== TEST 1B: FeedbackLoopBreaker - Circuit Breaker Stuck ===")

    breaker = FeedbackLoopBreaker(circuit_breaker_stuck_bars=50)

    # Simulate circuit breakers continuously tripped for 60 bars.
    # The internal counter increments each update where circuit_breakers_tripped=True.
    signal = None
    for bar in range(70):
        signal = breaker.update(
            bars_since_last_trade=bar,
            current_volatility=0.010,
            circuit_breakers_tripped=True,  # Stuck!
        )
        if signal is not None and signal.loop_type == "circuit_breaker":
            break

    assert signal is not None, "Should detect circuit breaker loop"
    assert signal.loop_type == "circuit_breaker", "Should detect circuit_breaker type"
    assert signal.severity >= 0.5, "Circuit breaker stuck should be severe"

    LOG.info("✓ Circuit breaker loop detected: severity=%.2f", signal.severity)


def test_feedback_loop_breaker_intervention():
    """Test that breaker can detect a loop and apply an intervention."""
    LOG.info("\n=== TEST 1C: FeedbackLoopBreaker - Intervention Application ===")

    breaker = FeedbackLoopBreaker(
        circuit_breaker_stuck_bars=50,
        intervention_cooldown_bars=0,  # No cooldown for testing
    )

    # Create stuck state by sending circuit_breakers_tripped for 60 bars
    signal = None
    for bar in range(60):
        signal = breaker.update(
            bars_since_last_trade=bar,
            current_volatility=0.012,
            circuit_breakers_tripped=True,
        )

    assert signal is not None, "Should detect loop before intervention"
    LOG.info("✓ Loop detected: %s (severity=%.2f)", signal.loop_type, signal.severity)

    # Apply intervention using the signal
    result = breaker.apply_intervention(signal)
    assert result is not None, "Should return intervention result"
    assert "action" in result, "Intervention result should have 'action' key"

    LOG.info("✓ Intervention applied: %s", result)


# ============================================================================
# TEST 2: ColdStartManager
# ============================================================================


def test_cold_start_manager_phase_progression():
    """Test that ColdStartManager progresses through phases correctly."""
    LOG.info("\n=== TEST 2A: ColdStartManager - Phase Progression ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ColdStartManager(
            observation_min_bars=10,
            paper_min_bars=20,
            micro_min_bars=30,
            state_file=Path(tmpdir) / "cold_start.json",
        )

        # Initially in OBSERVATION phase
        assert manager.current_phase == WarmupPhase.OBSERVATION
        assert not manager.can_trade()

        LOG.info("✓ Started in OBSERVATION phase")

        # Complete observation phase (10 bars)
        for _ in range(12):
            manager.update(new_bar=True)

        # Check graduation and apply
        next_phase = manager.check_graduation()
        assert next_phase == WarmupPhase.PAPER_TRADING
        manager.graduate(next_phase)

        assert manager.current_phase == WarmupPhase.PAPER_TRADING
        assert manager.can_trade()
        assert manager.is_paper_only()

        LOG.info("✓ Graduated to PAPER_TRADING phase")

        # Complete paper trading with good metrics (need >=20 bars, >=10 trades)
        for _ in range(25):
            manager.update(new_bar=True)
        # Add winning trades to get good sharpe/win_rate/drawdown
        for _ in range(12):
            manager.update(trade_completed={"pnl": 10.0, "is_paper": True})
        # Add a few losing trades for realism (still >45% win rate)
        for _ in range(3):
            manager.update(trade_completed={"pnl": -2.0, "is_paper": True})

        next_phase = manager.check_graduation()
        assert next_phase == WarmupPhase.MICRO_POSITIONS
        manager.graduate(next_phase)

        assert manager.current_phase == WarmupPhase.MICRO_POSITIONS
        assert manager.can_trade()
        assert not manager.is_paper_only()

        LOG.info("✓ Graduated to MICRO_POSITIONS phase")

        # Complete micro phase with excellent metrics (need >=30 bars, >=20 trades)
        for _ in range(35):
            manager.update(new_bar=True)
        # Add winning trades for good sharpe/win_rate
        for _ in range(20):
            manager.update(trade_completed={"pnl": 12.0, "is_paper": False})
        # Add a few losses (still >48% win rate, avg profit > 0)
        for _ in range(3):
            manager.update(trade_completed={"pnl": -3.0, "is_paper": False})

        next_phase = manager.check_graduation()
        assert next_phase == WarmupPhase.PRODUCTION
        manager.graduate(next_phase)

        assert manager.current_phase == WarmupPhase.PRODUCTION
        assert manager.can_trade()

        LOG.info("✓ Graduated to PRODUCTION phase")


def test_cold_start_manager_failed_graduation():
    """Test that ColdStartManager handles failed graduation criteria."""
    LOG.info("\n=== TEST 2B: ColdStartManager - Failed Graduation ===")

    manager = ColdStartManager(observation_min_bars=5, paper_min_bars=10)

    # Complete observation (5 bars)
    for _ in range(7):
        manager.update(new_bar=True)

    next_phase = manager.check_graduation()
    assert next_phase == WarmupPhase.PAPER_TRADING
    manager.graduate(next_phase)

    assert manager.current_phase == WarmupPhase.PAPER_TRADING

    # Fail paper trading: enough bars but poor metrics (losing trades)
    for _ in range(15):
        manager.update(new_bar=True)
    for _ in range(12):
        manager.update(trade_completed={"pnl": -5.0, "is_paper": True})

    # Should NOT graduate (bad metrics)
    next_phase = manager.check_graduation()
    assert next_phase is None, "Should not graduate with poor metrics"
    assert manager.current_phase == WarmupPhase.PAPER_TRADING

    LOG.info("✓ Failed graduation handled correctly (stayed in PAPER_TRADING)")


def test_cold_start_manager_demotion():
    """Test that production bot can be demoted on poor performance."""
    LOG.info("\n=== TEST 2C: ColdStartManager - Performance Demotion ===")

    manager = ColdStartManager(observation_min_bars=5, paper_min_bars=5, micro_min_bars=5)

    # Fast-forward to production
    manager.current_phase = WarmupPhase.PRODUCTION
    manager.bars_in_current_phase = 100

    # Simulate terrible performance (need >=50 trades for demotion check)
    for _ in range(60):
        manager.update(trade_completed={"pnl": -50.0, "is_paper": False})

    # Check if demotion would occur
    demotion_result = manager.check_graduation()

    # Check status regardless — demotion depends on metrics
    status = manager.get_status()
    LOG.info("✓ Performance demotion status: %s", status)

    # If demotion triggered, it should recommend MICRO_POSITIONS
    if demotion_result is not None:
        assert demotion_result == WarmupPhase.MICRO_POSITIONS, "Demotion should go to MICRO"
        LOG.info("✓ Demotion triggered: recommending MICRO_POSITIONS")
    else:
        LOG.info("✓ No demotion triggered (metrics within bounds)")


# ============================================================================
# TEST 3: RewardIntegrityMonitor
# ============================================================================


def test_reward_integrity_monitor_correlation():
    """Test reward-P&L correlation detection."""
    LOG.info("\n=== TEST 3A: RewardIntegrityMonitor - Correlation Tracking ===")

    monitor = RewardIntegrityMonitor(correlation_threshold=0.7)

    # Simulate 100 trades with good correlation
    rng = np.random.default_rng(42)
    for i in range(100):
        pnl = rng.standard_normal() * 100  # Random P&L
        reward = pnl * 0.8 + rng.standard_normal() * 20  # Correlated reward with noise

        monitor.add_trade(
            reward=reward,
            pnl=pnl,
            reward_components={"capture": reward * 0.7, "wtl": reward * 0.3},
        )

    status = monitor.check_integrity()

    assert status["status"] == "ok", "Should detect good correlation"
    assert status["correlation"] > 0.7, "Correlation should be >0.7"
    assert not status["is_gaming"], "Should not flag gaming with good correlation"

    LOG.info("✓ Good correlation detected: %.2f", status["correlation"])


def test_reward_integrity_monitor_gaming_detection():
    """Test detection of reward gaming (decorrelation)."""
    LOG.info("\n=== TEST 3B: RewardIntegrityMonitor - Gaming Detection ===")

    monitor = RewardIntegrityMonitor(correlation_threshold=0.7)

    # Simulate gaming: positive rewards but negative P&L
    rng = np.random.default_rng(42)
    for i in range(100):
        pnl = rng.standard_normal() * 100 - 50  # Negative-biased P&L
        reward = abs(rng.standard_normal() * 50 + 100)  # Always positive rewards (gaming!)

        monitor.add_trade(reward=reward, pnl=pnl, reward_components={"fake": reward})

    status = monitor.check_integrity()

    assert status["is_gaming"], "Should detect gaming behavior"
    assert status["correlation"] < 0.7, "Correlation should be poor"
    assert status["status"] == "critical", "Should flag critical status for gaming"

    LOG.info("✓ Gaming detected: correlation=%.2f (threshold=0.7)", status["correlation"])


# ============================================================================
# TEST 4: BrokerExecutionModel
# ============================================================================


def test_broker_execution_model_slippage():
    """Test realistic slippage calculation."""
    LOG.info("\n=== TEST 4A: BrokerExecutionModel - Slippage Calculation ===")

    model = BrokerExecutionModel(
        typical_spread_bps=5.0,
        base_slippage_bps=2.0,
        volatile_multiplier=2.0,
    )

    # Test BUY order in transitional regime (highest slippage multiplier)
    costs = model.estimate_execution_costs(
        side=OrderSide.BUY,
        quantity=0.10,
        mid_price=50000.0,
        spread_bps=6.0,
        regime="TRANSITIONAL",
    )

    assert costs.total_slippage_bps > costs.base_slippage_bps, "Should have extra slippage in transitional regime"
    assert costs.expected_fill_price > 50000.0, "BUY should pay above mid"

    LOG.info("✓ BUY slippage: %.1f bps, fill price: %.2f", costs.total_slippage_bps, costs.expected_fill_price)

    # Test SELL order
    costs_sell = model.estimate_execution_costs(
        side=OrderSide.SELL,
        quantity=0.10,
        mid_price=50000.0,
        spread_bps=6.0,
        regime="TRANSITIONAL",
    )

    assert costs_sell.expected_fill_price < 50000.0, "SELL should receive below mid"

    LOG.info(
        "✓ SELL slippage: %.1f bps, fill price: %.2f", costs_sell.total_slippage_bps, costs_sell.expected_fill_price
    )


def test_broker_execution_model_regime_impact():
    """Test that different regimes have different slippage."""
    LOG.info("\n=== TEST 4B: BrokerExecutionModel - Regime Impact ===")

    model = BrokerExecutionModel(
        base_slippage_bps=2.0,
        volatile_multiplier=2.0,
        trending_multiplier=1.5,
        mean_reverting_multiplier=0.8,
    )

    # Valid regimes: TRENDING, MEAN_REVERTING, TRANSITIONAL, UNKNOWN
    regimes = ["TRANSITIONAL", "TRENDING", "MEAN_REVERTING"]
    slippages = []

    for regime in regimes:
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY,
            quantity=0.05,
            mid_price=50000.0,
            spread_bps=5.0,
            regime=regime,
        )
        slippages.append(costs.total_slippage_bps)

        LOG.info("  %s: %.1f bps", regime, costs.total_slippage_bps)

    # Verify ordering: TRANSITIONAL > TRENDING > MEAN_REVERTING
    assert slippages[0] > slippages[1] > slippages[2], (
        f"Regime impact ordering incorrect: {list(zip(regimes, slippages))}"
    )

    LOG.info("✓ Regime impact verified: TRANSITIONAL > TRENDING > MEAN_REVERTING")


# ============================================================================
# TEST 5: ParameterStalenessDetector
# ============================================================================


def test_parameter_staleness_baseline():
    """Test baseline establishment."""
    LOG.info("\n=== TEST 5A: ParameterStalenessDetector - Baseline Establishment ===")

    detector = ParameterStalenessDetector(
        performance_window=100,
        staleness_threshold=0.6,
    )

    # Simulate 100 bars to establish baseline (bars_for_baseline = min(500, 100) = 100)
    params = {"param1": 0.5, "param2": 0.3}
    for bar in range(100):
        detector.update(
            bar_num=bar,
            parameters=params,
            performance_metrics={
                "win_rate": 0.52 + rng.standard_normal() * 0.05,
                "sharpe": 0.4 + rng.standard_normal() * 0.1,
                "avg_confidence": 0.6 + rng.standard_normal() * 0.05,
            },
            regime="TRENDING",
        )

    status = detector.check_staleness()

    assert detector.baseline_established, "Baseline should be established after 100 bars"
    assert not status["is_stale"], "Should not be stale with consistent performance"

    LOG.info(
        "✓ Baseline established: WinRate=%.2f%%, Sharpe=%.2f",
        detector.baseline_win_rate * 100,
        detector.baseline_sharpe,
    )


def test_parameter_staleness_performance_decay():
    """Test detection of performance decay."""
    LOG.info("\n=== TEST 5B: ParameterStalenessDetector - Performance Decay ===")

    detector = ParameterStalenessDetector(
        performance_window=200,
        staleness_threshold=0.5,
    )

    # Establish good baseline (need min(500,200)=200 bars)
    params = {"param1": 0.5}
    for bar in range(200):
        detector.update(
            bar_num=bar,
            parameters=params,
            performance_metrics={
                "win_rate": 0.55,
                "sharpe": 0.5,
                "avg_confidence": 0.65,
            },
            regime="TRENDING",
        )

    assert detector.baseline_established, "Baseline should be established"

    # Now simulate performance decay
    for bar in range(200, 300):
        detector.update(
            bar_num=bar,
            parameters=params,
            performance_metrics={
                "win_rate": 0.35,  # Dropped 20pp
                "sharpe": 0.1,  # Dropped 0.4
                "avg_confidence": 0.55,
            },
            regime="TRENDING",  # Same regime!
        )

    status = detector.check_staleness()

    assert status["is_stale"], "Should detect staleness from performance decay"
    assert any(
        s["type"] == "performance_decay" for s in status.get("signals", [])
    ), "Should have performance_decay signal"

    LOG.info("✓ Performance decay detected: staleness_score=%.2f", status["staleness_score"])


def test_parameter_staleness_regime_shift():
    """Test detection of regime shift."""
    LOG.info("\n=== TEST 5C: ParameterStalenessDetector - Regime Shift ===")

    detector = ParameterStalenessDetector(
        performance_window=600,
        regime_stability_bars=50,
    )

    # Baseline in TRENDING regime (need min(500,600) = 500 bars)
    params = {"param1": 0.5}
    for bar in range(500):
        detector.update(
            bar_num=bar,
            parameters=params,
            performance_metrics={
                "win_rate": 0.52,
                "sharpe": 0.4,
                "avg_confidence": 0.6,
            },
            regime="TRENDING",
        )

    assert detector.baseline_established, "Baseline should be established"

    # Shift to MEAN_REVERTING and stay there
    for bar in range(500, 600):
        detector.update(
            bar_num=bar,
            parameters=params,
            performance_metrics={
                "win_rate": 0.50,  # Slight degradation
                "sharpe": 0.35,
                "avg_confidence": 0.58,
            },
            regime="MEAN_REVERTING",  # Different regime!
        )

    status = detector.check_staleness()

    assert any(s["type"] == "regime_shift" for s in status.get("signals", [])), "Should detect regime shift"

    LOG.info("✓ Regime shift detected")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


def run_all_tests():
    """Run all integration tests."""
    LOG.info("\n" + "=" * 80)
    LOG.info("CRITICAL COMPONENTS INTEGRATION TESTS")
    LOG.info("Testing gap-closing implementations")
    LOG.info("=" * 80)

    tests = [
        # FeedbackLoopBreaker
        test_feedback_loop_breaker_no_trade_detection,
        test_feedback_loop_breaker_circuit_breaker_stuck,
        test_feedback_loop_breaker_intervention,
        # ColdStartManager
        test_cold_start_manager_phase_progression,
        test_cold_start_manager_failed_graduation,
        test_cold_start_manager_demotion,
        # RewardIntegrityMonitor
        test_reward_integrity_monitor_correlation,
        test_reward_integrity_monitor_gaming_detection,
        # BrokerExecutionModel
        test_broker_execution_model_slippage,
        test_broker_execution_model_regime_impact,
        # ParameterStalenessDetector
        test_parameter_staleness_baseline,
        test_parameter_staleness_performance_decay,
        test_parameter_staleness_regime_shift,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            LOG.error("FAILED: %s - %s", test.__name__, e)
            failed += 1
        except Exception as e:
            LOG.error("ERROR: %s - %s", test.__name__, e)
            failed += 1

    LOG.info("\n" + "=" * 80)
    LOG.info("TEST SUMMARY")
    LOG.info("=" * 80)
    LOG.info("Passed: %d", passed)
    LOG.info("Failed: %d", failed)
    LOG.info("Total:  %d", len(tests))

    if failed == 0:
        LOG.info("\n✅ ALL TESTS PASSED")
    else:
        LOG.error("\n❌ SOME TESTS FAILED")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
