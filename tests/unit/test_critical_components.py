"""
Integration Tests for Critical Components

Tests the following active components:
1. RewardIntegrityMonitor (GAP 4)
2. BrokerExecutionModel (GAP 2 - execution modeling)
"""

import logging
import sys

import numpy as np

rng = np.random.default_rng(42)


from src.core.broker_execution_model import BrokerExecutionModel, OrderSide
from src.core.reward_integrity_monitor import RewardIntegrityMonitor

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


# ============================================================================
# TEST 1: RewardIntegrityMonitor
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
# MAIN TEST RUNNER
# ============================================================================


def run_all_tests():
    """Run all integration tests."""
    LOG.info("\n" + "=" * 80)
    LOG.info("CRITICAL COMPONENTS INTEGRATION TESTS")
    LOG.info("Testing gap-closing implementations")
    LOG.info("=" * 80)

    tests = [
        # RewardIntegrityMonitor
        test_reward_integrity_monitor_correlation,
        test_reward_integrity_monitor_gaming_detection,
        # BrokerExecutionModel
        test_broker_execution_model_slippage,
        test_broker_execution_model_regime_impact,
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
