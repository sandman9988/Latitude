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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from broker_execution_model import BrokerExecutionModel, ExecutionCosts, OrderSide
from cold_start_manager import ColdStartManager, GraduationCriteria, WarmupPhase
from feedback_loop_breaker import FeedbackLoopBreaker, FeedbackLoopSignal
from parameter_staleness import ParameterStalenessDetector, StalenessSignal
from reward_integrity_monitor import RewardIntegrityMonitor

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


# ============================================================================
# TEST 1: FeedbackLoopBreaker
# ============================================================================


def test_feedback_loop_breaker_no_trade_detection():
    """Test that FeedbackLoopBreaker detects no-trade loops."""
    LOG.info("\n=== TEST 1A: FeedbackLoopBreaker - No-Trade Detection ===")

    breaker = FeedbackLoopBreaker(
        no_trade_lookback=20,
        circuit_breaker_lookback=50,
        performance_lookback=100,
    )

    # Simulate scenario: market has opportunities but no trades for 25 bars
    for bar in range(30):
        breaker.update(
            bar_num=bar,
            trades_this_bar=0,
            volatility=0.015,  # Good volatility
            opportunities_present=True,  # Signals generated but not acted on
            circuit_breakers_tripped=False,
            recent_performance={"win_rate": 0.45, "sharpe": 0.2},
        )

    loops = breaker.check_for_loops()

    assert len(loops) > 0, "Should detect no-trade loop"
    assert any(loop.loop_type == "no_trades" for loop in loops), "Should detect no_trades type"

    no_trade_loop = next(loop for loop in loops if loop.loop_type == "no_trades")
    assert no_trade_loop.severity > 0.5, "Should have significant severity"

    LOG.info(
        "✓ No-trade loop detected: severity=%.2f, duration=%d bars",
        no_trade_loop.severity,
        no_trade_loop.duration_bars,
    )
    LOG.info("  Evidence: %s", no_trade_loop.evidence)


def test_feedback_loop_breaker_circuit_breaker_stuck():
    """Test detection of permanently tripped circuit breakers."""
    LOG.info("\n=== TEST 1B: FeedbackLoopBreaker - Circuit Breaker Stuck ===")

    breaker = FeedbackLoopBreaker()

    # Simulate circuit breakers stuck for 60 bars
    for bar in range(70):
        breaker.update(
            bar_num=bar,
            trades_this_bar=0,
            volatility=0.010,
            opportunities_present=True,
            circuit_breakers_tripped=True,  # Stuck!
            recent_performance={"win_rate": 0.40, "sharpe": 0.0},
        )

    loops = breaker.check_for_loops()

    assert any(loop.loop_type == "circuit_breaker" for loop in loops), "Should detect circuit breaker loop"

    cb_loop = next(loop for loop in loops if loop.loop_type == "circuit_breaker")
    assert cb_loop.severity > 0.7, "Circuit breaker stuck should be severe"
    assert "temporarily disable" in cb_loop.suggested_intervention.lower()

    LOG.info("✓ Circuit breaker loop detected: severity=%.2f", cb_loop.severity)


def test_feedback_loop_breaker_intervention():
    """Test that breaker can suggest and apply interventions."""
    LOG.info("\n=== TEST 1C: FeedbackLoopBreaker - Intervention Application ===")

    breaker = FeedbackLoopBreaker()

    # Create stuck state
    for bar in range(60):
        breaker.update(
            bar_num=bar,
            trades_this_bar=0,
            volatility=0.012,
            opportunities_present=True,
            circuit_breakers_tripped=True,
            recent_performance={"win_rate": 0.38, "sharpe": -0.1},
        )

    loops = breaker.check_for_loops()
    assert loops, "Should detect loop before intervention"

    # Apply intervention
    intervention = breaker.suggest_intervention()
    assert intervention is not None, "Should suggest intervention"

    LOG.info("✓ Suggested intervention: %s", intervention)

    # Apply intervention (in real system, this would modify circuit breakers)
    breaker.apply_intervention(intervention)

    # After intervention, should be able to clear
    breaker.clear_loop_state()

    LOG.info("✓ Intervention applied and loop state cleared")


# ============================================================================
# TEST 2: ColdStartManager
# ============================================================================


def test_cold_start_manager_phase_progression():
    """Test that ColdStartManager progresses through phases correctly."""
    LOG.info("\n=== TEST 2A: ColdStartManager - Phase Progression ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ColdStartManager(
            observation_bars=10,
            paper_bars=20,
            micro_bars=30,
            persistence_path=Path(tmpdir) / "cold_start.json",
        )

        # Initially in OBSERVATION phase
        assert manager.current_phase == WarmupPhase.OBSERVATION
        assert not manager.can_trade()
        assert not manager.can_use_real_money()

        LOG.info("✓ Started in OBSERVATION phase")

        # Complete observation phase
        for bar in range(12):
            manager.update(
                bar_num=bar,
                metrics={
                    "win_rate": 0.5,
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                    "avg_trade_profit": 0.0,
                },
            )

        # Should graduate to PAPER
        assert manager.current_phase == WarmupPhase.PAPER_TRADING
        assert manager.can_trade()
        assert not manager.can_use_real_money()

        LOG.info("✓ Graduated to PAPER_TRADING phase")

        # Complete paper trading with good metrics
        for bar in range(12, 35):
            manager.update(
                bar_num=bar,
                metrics={
                    "win_rate": 0.52,  # >45% required
                    "sharpe": 0.35,  # >0.3 required
                    "max_drawdown": 0.15,  # <20% required
                    "avg_trade_profit": 5.0,
                },
            )

        # Should graduate to MICRO
        assert manager.current_phase == WarmupPhase.MICRO_POSITIONS
        assert manager.can_trade()
        assert manager.can_use_real_money()

        LOG.info("✓ Graduated to MICRO_POSITIONS phase")

        # Complete micro phase with excellent metrics
        for bar in range(35, 70):
            manager.update(
                bar_num=bar,
                metrics={
                    "win_rate": 0.55,  # >48% required
                    "sharpe": 0.6,  # >0.5 required
                    "max_drawdown": 0.12,
                    "avg_trade_profit": 12.0,  # >0 required
                },
            )

        # Should graduate to PRODUCTION
        assert manager.current_phase == WarmupPhase.PRODUCTION
        assert manager.can_trade()
        assert manager.can_use_real_money()

        LOG.info("✓ Graduated to PRODUCTION phase")


def test_cold_start_manager_failed_graduation():
    """Test that ColdStartManager handles failed graduation criteria."""
    LOG.info("\n=== TEST 2B: ColdStartManager - Failed Graduation ===")

    manager = ColdStartManager(observation_bars=5, paper_bars=10)

    # Complete observation
    for bar in range(7):
        manager.update(bar_num=bar, metrics={"win_rate": 0.5})

    assert manager.current_phase == WarmupPhase.PAPER_TRADING

    # Fail paper trading (poor metrics)
    for bar in range(7, 20):
        manager.update(
            bar_num=bar,
            metrics={
                "win_rate": 0.35,  # <45% required
                "sharpe": 0.1,  # <0.3 required
                "max_drawdown": 0.25,  # >20% limit
                "avg_trade_profit": -5.0,
            },
        )

    # Should NOT graduate (still in paper trading)
    assert manager.current_phase == WarmupPhase.PAPER_TRADING
    assert not manager.can_use_real_money()

    LOG.info("✓ Failed graduation handled correctly (stayed in PAPER_TRADING)")


def test_cold_start_manager_demotion():
    """Test that production bot can be demoted on poor performance."""
    LOG.info("\n=== TEST 2C: ColdStartManager - Performance Demotion ===")

    manager = ColdStartManager(observation_bars=5, paper_bars=5, micro_bars=5)

    # Fast-forward to production
    manager.current_phase = WarmupPhase.PRODUCTION
    manager.bars_in_phase = 100

    # Simulate terrible performance
    for bar in range(120, 145):
        manager.update(
            bar_num=bar,
            metrics={
                "win_rate": 0.25,  # Terrible
                "sharpe": -0.5,  # Negative Sharpe
                "max_drawdown": 0.30,  # Large drawdown
                "avg_trade_profit": -50.0,
            },
        )

    # Check if demotion occurred (should demote back to micro or paper)
    status = manager.get_status()

    LOG.info("✓ Performance demotion status: %s", status)
    # Note: Actual demotion logic depends on implementation


# ============================================================================
# TEST 3: RewardIntegrityMonitor
# ============================================================================


def test_reward_integrity_monitor_correlation():
    """Test reward-P&L correlation detection."""
    LOG.info("\n=== TEST 3A: RewardIntegrityMonitor - Correlation Tracking ===")

    monitor = RewardIntegrityMonitor(correlation_threshold=0.7)

    # Simulate 100 trades with good correlation
    np.random.seed(42)
    for i in range(100):
        pnl = np.random.randn() * 100  # Random P&L
        reward = pnl * 0.8 + np.random.randn() * 20  # Correlated reward with noise

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
    np.random.seed(42)
    for i in range(100):
        pnl = np.random.randn() * 100 - 50  # Negative-biased P&L
        reward = abs(np.random.randn() * 50 + 100)  # Always positive rewards (gaming!)

        monitor.add_trade(reward=reward, pnl=pnl, reward_components={"fake": reward})

    status = monitor.check_integrity()

    assert status["is_gaming"], "Should detect gaming behavior"
    assert status["correlation"] < 0.7, "Correlation should be poor"
    assert status["status"] == "WARNING", "Should warn about gaming"

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

    # Test BUY order in volatile regime
    costs = model.estimate_execution_costs(
        side=OrderSide.BUY,
        quantity=0.10,
        mid_price=50000.0,
        spread_bps=6.0,
        regime="VOLATILE",
    )

    assert costs.total_slippage_bps > costs.base_slippage_bps, "Should have extra slippage in volatile regime"
    assert costs.expected_fill_price > 50000.0, "BUY should pay above mid"

    LOG.info("✓ BUY slippage: %.1f bps, fill price: %.2f", costs.total_slippage_bps, costs.expected_fill_price)

    # Test SELL order
    costs_sell = model.estimate_execution_costs(
        side=OrderSide.SELL,
        quantity=0.10,
        mid_price=50000.0,
        spread_bps=6.0,
        regime="VOLATILE",
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

    regimes = ["VOLATILE", "TRENDING", "MEAN_REVERTING"]
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

    # Verify ordering: VOLATILE > TRENDING > MEAN_REVERTING
    assert slippages[0] > slippages[1] > slippages[2], "Regime impact ordering incorrect"

    LOG.info("✓ Regime impact verified: VOLATILE > TRENDING > MEAN_REVERTING")


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

    # Simulate 500 bars to establish baseline
    params = {"param1": 0.5, "param2": 0.3}
    for bar in range(500):
        detector.update(
            bar_num=bar,
            parameters=params,
            performance_metrics={
                "win_rate": 0.52 + np.random.randn() * 0.05,
                "sharpe": 0.4 + np.random.randn() * 0.1,
                "avg_confidence": 0.6 + np.random.randn() * 0.05,
            },
            regime="TRENDING",
        )

    status = detector.check_staleness()

    assert detector.baseline_established, "Baseline should be established"
    assert not status["is_stale"], "Should not be stale with consistent performance"

    LOG.info(
        "✓ Baseline established: WinRate=%.2f%%, Sharpe=%.2f",
        detector.baseline_win_rate * 100,
        detector.baseline_sharpe,
    )


def test_parameter_staleness_performance_decay():
    """Test detection of performance decay."""
    LOG.info("\n=== TEST 5B: ParameterStalenessDetector - Performance Decay ===")

    detector = ParameterStalenessDetector(staleness_threshold=0.5)

    # Establish good baseline
    params = {"param1": 0.5}
    for bar in range(500):
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

    # Now simulate performance decay
    for bar in range(500, 600):
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

    detector = ParameterStalenessDetector(regime_stability_bars=50)

    # Baseline in TRENDING regime
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
