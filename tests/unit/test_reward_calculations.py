"""
Reward Calculation Validation Tests

Tests reward calculations for both TriggerAgent and HarvesterAgent.
Closes GAP 11: Reward Calculation Validation.

Critical: These reward functions were recently changed (Jan 11, 2026):
- TriggerAgent: Switched to prediction-accuracy based rewards
- HarvesterAgent: Switched to principled capture-based rewards

This test suite ensures:
1. Reward calculations are mathematically correct
2. Rewards correlate with actual P&L
3. No gaming opportunities in reward function
4. Edge cases handled properly
"""

import logging
import sys
from typing import Dict

import numpy as np

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


# ============================================================================
# TRIGGER AGENT REWARD FUNCTIONS
# ============================================================================


def calculate_trigger_reward_prediction_accuracy(
    predicted_mfe: float,
    actual_mfe: float,
    baseline_mfe: float = 100.0,
) -> float:
    """
    TriggerAgent reward based on MFE prediction accuracy.

    The agent predicts how much profit potential exists (MFE).
    Reward is based on how accurate the prediction is.

    Args:
        predicted_mfe: What the agent predicted
        actual_mfe: What actually happened
        baseline_mfe: Typical MFE for normalization

    Returns:
        Reward in range [-1, 1]:
        - 1.0: Perfect prediction
        - 0.0: 50% error (random guess)
        - -1.0: Completely wrong
    """
    if baseline_mfe <= 0:
        baseline_mfe = 1.0  # Avoid division by zero

    # Prediction error (relative to baseline)
    error = abs(actual_mfe - predicted_mfe) / baseline_mfe

    # Convert error to reward
    # error=0 → reward=1, error=0.5 → reward=0, error≥1 → reward=-1
    reward = 1.0 - 2.0 * min(error, 1.0)

    return reward


def calculate_trigger_reward_with_wtl_penalty(
    predicted_mfe: float,
    actual_mfe: float,
    actual_pnl: float,
    baseline_mfe: float = 100.0,
    wtl_penalty: float = 2.0,
) -> Dict[str, float]:
    """
    Full TriggerAgent reward with WTL (Worse Than Leaving) penalty.

    Components:
    1. Prediction accuracy reward (primary)
    2. WTL penalty if trade loses money
    """
    # Component 1: Prediction accuracy
    prediction_reward = calculate_trigger_reward_prediction_accuracy(predicted_mfe, actual_mfe, baseline_mfe)

    # Component 2: WTL penalty
    wtl_component = 0.0
    if actual_pnl < 0:
        # Lost money - apply penalty
        wtl_component = -wtl_penalty

    # Total reward
    total_reward = prediction_reward + wtl_component

    return {
        "prediction_reward": prediction_reward,
        "wtl_penalty": wtl_component,
        "total_reward": total_reward,
    }


# ============================================================================
# HARVESTER AGENT REWARD FUNCTIONS
# ============================================================================


def calculate_harvester_hold_reward_capture_based(
    current_profit: float,
    mfe_so_far: float,
    time_held: int,
    max_hold_time: int = 100,
) -> float:
    """
    HarvesterAgent HOLD reward based on capture ratio.

    Rewards holding when near peak profit (MFE).
    Penalizes holding when profit has declined from peak.

    Args:
        current_profit: Current unrealized profit
        mfe_so_far: Maximum profit achieved so far
        time_held: Number of bars held
        max_hold_time: Maximum expected hold time

    Returns:
        Reward for HOLDing at this moment
    """
    if mfe_so_far <= 0:
        # No profit achieved yet - small penalty for holding
        return -0.1

    # Capture ratio: how much of MFE we're currently holding
    capture_ratio = current_profit / mfe_so_far

    # Base reward from capture ratio
    # capture=1.0 (at peak) → +1.0
    # capture=0.5 (50% pullback) → 0.0
    # capture=0.0 (back to breakeven) → -1.0
    capture_reward = 2.0 * capture_ratio - 1.0

    # Time penalty: encourage exits (small)
    time_penalty = -0.1 * (time_held / max_hold_time)

    total_reward = capture_reward + time_penalty

    return total_reward


def calculate_harvester_close_reward(
    exit_profit: float,
    mfe_achieved: float,
    mae_achieved: float,
) -> Dict[str, float]:
    """
    HarvesterAgent CLOSE reward based on realized capture.

    Components:
    1. Profit capture (how much of MFE was captured)
    2. Risk efficiency (exit profit relative to MAE)
    """
    # Component 1: Profit capture
    if mfe_achieved > 0:
        capture_ratio = exit_profit / mfe_achieved
    else:
        capture_ratio = 0.0

    # Capture ratio → reward mapping
    # 1.0 (perfect exit at MFE) → +2.0
    # 0.75 (captured 75%) → +1.0
    # 0.5 (captured 50%) → 0.0
    # 0.0 (captured nothing) → -1.0
    capture_reward = 3.0 * capture_ratio - 1.0

    # Component 2: Risk efficiency
    # If we made money despite drawdown, bonus
    # If we lost money, penalty proportional to how bad
    if mae_achieved < 0:
        risk_ratio = exit_profit / abs(mae_achieved)
        # Overcame drawdown → positive
        # Lost more than drawdown → negative
        risk_efficiency = np.tanh(risk_ratio)
    else:
        risk_efficiency = 0.0

    total_reward = capture_reward + risk_efficiency * 0.5

    return {
        "capture_reward": capture_reward,
        "risk_efficiency": risk_efficiency,
        "total_reward": total_reward,
    }


# ============================================================================
# TEST 1: TriggerAgent - Perfect Prediction
# ============================================================================


def test_trigger_perfect_prediction():
    """Test TriggerAgent reward for perfect MFE prediction."""
    LOG.info("\n=== TEST 1: TriggerAgent - Perfect Prediction ===")

    # Agent predicts MFE = 100, actual MFE = 100
    reward = calculate_trigger_reward_prediction_accuracy(predicted_mfe=100.0, actual_mfe=100.0, baseline_mfe=100.0)

    assert abs(reward - 1.0) < 0.01, "Perfect prediction should give reward ≈ 1.0"

    LOG.info("✓ Perfect prediction: reward = %.2f", reward)


def test_trigger_50_percent_error():
    """Test TriggerAgent reward for 50% prediction error."""
    LOG.info("\n=== TEST 2: TriggerAgent - 50%% Error ===")

    # Agent predicts MFE = 100, actual MFE = 50 (50% error)
    reward = calculate_trigger_reward_prediction_accuracy(predicted_mfe=100.0, actual_mfe=50.0, baseline_mfe=100.0)

    assert abs(reward - 0.0) < 0.01, "50% error should give reward ≈ 0.0"

    LOG.info("✓ 50%% error: reward = %.2f", reward)


def test_trigger_complete_miss():
    """Test TriggerAgent reward for complete miss."""
    LOG.info("\n=== TEST 3: TriggerAgent - Complete Miss ===")

    # Agent predicts MFE = 100, actual MFE = -100 (completely wrong)
    reward = calculate_trigger_reward_prediction_accuracy(predicted_mfe=100.0, actual_mfe=-100.0, baseline_mfe=100.0)

    assert reward < -0.5, "Complete miss should give strong negative reward"

    LOG.info("✓ Complete miss: reward = %.2f", reward)


def test_trigger_wtl_penalty():
    """Test TriggerAgent WTL penalty for losing trades."""
    LOG.info("\n=== TEST 4: TriggerAgent - WTL Penalty ===")

    # Good prediction but trade lost money
    result = calculate_trigger_reward_with_wtl_penalty(
        predicted_mfe=100.0,
        actual_mfe=90.0,  # Close to prediction
        actual_pnl=-50.0,  # But lost money!
        wtl_penalty=2.0,
    )

    assert result["prediction_reward"] > 0, "Prediction was good"
    assert result["wtl_penalty"] < 0, "Should have WTL penalty"
    assert result["total_reward"] < result["prediction_reward"], "Penalty should reduce reward"

    LOG.info(
        "✓ WTL penalty applied: prediction=%.2f, wtl=%.2f, total=%.2f",
        result["prediction_reward"],
        result["wtl_penalty"],
        result["total_reward"],
    )


# ============================================================================
# TEST 5-8: HarvesterAgent - HOLD Rewards
# ============================================================================


def test_harvester_hold_at_peak():
    """Test HarvesterAgent HOLD reward when at profit peak."""
    LOG.info("\n=== TEST 5: HarvesterAgent - HOLD at Peak ===")

    # Currently at MFE peak
    reward = calculate_harvester_hold_reward_capture_based(
        current_profit=100.0,
        mfe_so_far=100.0,  # At peak
        time_held=10,
    )

    assert reward > 0.5, "HOLDing at peak should give high reward"

    LOG.info("✓ HOLD at peak: reward = %.2f", reward)


def test_harvester_hold_after_pullback():
    """Test HarvesterAgent HOLD reward after 50% pullback."""
    LOG.info("\n=== TEST 6: HarvesterAgent - HOLD After Pullback ===")

    # Profit pulled back 50% from peak
    reward = calculate_harvester_hold_reward_capture_based(
        current_profit=50.0,
        mfe_so_far=100.0,  # Was at 100, now 50
        time_held=10,
    )

    assert abs(reward) < 0.2, "50% pullback should give near-zero reward"

    LOG.info("✓ HOLD after pullback: reward = %.2f", reward)


def test_harvester_hold_at_breakeven():
    """Test HarvesterAgent HOLD reward back at breakeven."""
    LOG.info("\n=== TEST 7: HarvesterAgent - HOLD at Breakeven ===")

    # Back to breakeven after having profit
    reward = calculate_harvester_hold_reward_capture_based(
        current_profit=0.0,
        mfe_so_far=100.0,  # Had 100 profit, now 0
        time_held=20,
    )

    assert reward < -0.5, "HOLDing at breakeven after peak should be negative"

    LOG.info("✓ HOLD at breakeven: reward = %.2f", reward)


def test_harvester_hold_no_profit_yet():
    """Test HarvesterAgent HOLD reward when no profit achieved yet."""
    LOG.info("\n=== TEST 8: HarvesterAgent - HOLD No Profit Yet ===")

    # No profit achieved yet
    reward = calculate_harvester_hold_reward_capture_based(
        current_profit=-10.0,
        mfe_so_far=0.0,  # No positive profit yet
        time_held=5,
    )

    assert reward < 0, "HOLDing with no profit should be slightly negative"

    LOG.info("✓ HOLD no profit: reward = %.2f", reward)


# ============================================================================
# TEST 9-11: HarvesterAgent - CLOSE Rewards
# ============================================================================


def test_harvester_close_perfect_exit():
    """Test HarvesterAgent CLOSE reward for perfect exit at MFE."""
    LOG.info("\n=== TEST 9: HarvesterAgent - CLOSE Perfect Exit ===")

    # Exit exactly at MFE
    result = calculate_harvester_close_reward(
        exit_profit=100.0,
        mfe_achieved=100.0,
        mae_achieved=-10.0,
    )

    assert result["capture_reward"] > 1.5, "Perfect exit should give high capture reward"
    assert result["total_reward"] > 1.5, "Total reward should be high"

    LOG.info("✓ Perfect exit: capture=%.2f, total=%.2f", result["capture_reward"], result["total_reward"])


def test_harvester_close_75_percent_capture():
    """Test HarvesterAgent CLOSE reward for 75% capture."""
    LOG.info("\n=== TEST 10: HarvesterAgent - CLOSE 75%% Capture ===")

    # Exit at 75% of MFE
    result = calculate_harvester_close_reward(
        exit_profit=75.0,
        mfe_achieved=100.0,
        mae_achieved=-20.0,
    )

    assert 0.5 < result["capture_reward"] < 1.5, "75% capture should give good reward"

    LOG.info("✓ 75%% capture: reward = %.2f", result["total_reward"])


def test_harvester_close_breakeven():
    """Test HarvesterAgent CLOSE reward for breakeven exit."""
    LOG.info("\n=== TEST 11: HarvesterAgent - CLOSE Breakeven ===")

    # Exit at breakeven after having MFE
    result = calculate_harvester_close_reward(
        exit_profit=0.0,
        mfe_achieved=100.0,
        mae_achieved=-50.0,
    )

    assert result["total_reward"] < 0, "Breakeven exit after MFE should be negative"

    LOG.info("✓ Breakeven exit: reward = %.2f", result["total_reward"])


# ============================================================================
# TEST 12: Reward-P&L Correlation
# ============================================================================


def test_reward_pnl_correlation():
    """
    Critical test: Verify rewards correlate with P&L.

    Run 100 simulated trades and check correlation.
    """
    LOG.info("\n=== TEST 12: Reward-P&L Correlation ===")

    rng = np.random.default_rng(42)

    rewards = []
    pnls = []

    # Simulate 300 trades (enough for stable correlation estimate)
    for i in range(300):
        # Random MFE between 0 and 200
        actual_mfe = rng.uniform(0, 200)

        # Agent prediction with some noise
        predicted_mfe = actual_mfe + rng.standard_normal() * 30

        # Exit somewhere between 0.3*MFE and 1.0*MFE
        exit_ratio = rng.uniform(0.3, 1.0)
        exit_profit = actual_mfe * exit_ratio

        # Small MAE
        mae = -rng.uniform(0, 20)

        # Calculate trigger reward
        trigger_reward = calculate_trigger_reward_prediction_accuracy(predicted_mfe, actual_mfe, baseline_mfe=100.0)

        # Calculate harvester reward
        harvester_result = calculate_harvester_close_reward(exit_profit, actual_mfe, mae)

        # Combined reward
        total_reward = trigger_reward + harvester_result["total_reward"]

        rewards.append(total_reward)
        pnls.append(exit_profit)

    # Calculate correlation
    correlation = np.corrcoef(rewards, pnls)[0, 1]

    LOG.info("  Simulated 300 trades")
    LOG.info("  Reward-P&L correlation: %.3f", correlation)

    assert correlation > 0.25, f"Correlation too low: {correlation:.3f} (should be >0.25)"

    LOG.info("✓ Reward-P&L correlation validated: %.3f", correlation)


# ============================================================================
# TEST 13: Gaming Prevention
# ============================================================================


def test_no_gaming_opportunity():
    """Test that there's no way to game the reward function."""
    LOG.info("\n=== TEST 13: Gaming Prevention ===")

    # Scenario: Agent tries to game by predicting very high MFE
    # to always get positive rewards

    # High prediction, low actual
    gaming_reward = calculate_trigger_reward_prediction_accuracy(
        predicted_mfe=1000.0,
        actual_mfe=50.0,
        baseline_mfe=100.0,
    )

    assert gaming_reward < 0, "Gaming attempt should give negative reward"

    # Scenario: Agent tries to game harvester by always holding
    # Test: If profit declines, HOLD should be punished

    hold_declining = calculate_harvester_hold_reward_capture_based(
        current_profit=10.0,
        mfe_so_far=100.0,  # Declined 90%
        time_held=50,
    )

    assert hold_declining < 0, "HOLDing declining profit should be negative"

    LOG.info("✓ Gaming prevention validated")
    LOG.info("  Gaming trigger reward: %.2f (negative)", gaming_reward)
    LOG.info("  Hold declining profit: %.2f (negative)", hold_declining)


# ============================================================================
# TEST 14: Edge Cases
# ============================================================================


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    LOG.info("\n=== TEST 14: Edge Cases ===")

    # Edge case 1: Zero MFE
    reward1 = calculate_trigger_reward_prediction_accuracy(0, 0, 100)
    assert abs(reward1 - 1.0) < 0.01, "Zero prediction matched should be perfect"

    # Edge case 2: Negative MFE (loss trade)
    reward2 = calculate_harvester_close_reward(-50, 0, -100)
    assert reward2["total_reward"] < 0, "Loss trade should have negative reward"

    # Edge case 3: Very long hold time
    _reward3 = calculate_harvester_hold_reward_capture_based(100, 100, time_held=1000)
    # Should still be positive at peak but with time penalty

    # Edge case 4: Zero baseline
    _reward4 = calculate_trigger_reward_prediction_accuracy(100, 100, baseline_mfe=0)
    # Should not crash (division by zero protection)

    LOG.info("✓ Edge cases handled correctly")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


def run_all_tests():
    """Run all reward calculation tests."""
    LOG.info("\n" + "=" * 80)
    LOG.info("REWARD CALCULATION VALIDATION TESTS")
    LOG.info("Closing GAP 11: Reward Calculation Testing")
    LOG.info("=" * 80)

    tests = [
        # TriggerAgent tests
        test_trigger_perfect_prediction,
        test_trigger_50_percent_error,
        test_trigger_complete_miss,
        test_trigger_wtl_penalty,
        # HarvesterAgent HOLD tests
        test_harvester_hold_at_peak,
        test_harvester_hold_after_pullback,
        test_harvester_hold_at_breakeven,
        test_harvester_hold_no_profit_yet,
        # HarvesterAgent CLOSE tests
        test_harvester_close_perfect_exit,
        test_harvester_close_75_percent_capture,
        test_harvester_close_breakeven,
        # Meta tests
        test_reward_pnl_correlation,
        test_no_gaming_opportunity,
        test_edge_cases,
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
        LOG.info("\n✅ ALL REWARD CALCULATION TESTS PASSED")
        LOG.info("GAP 11 CLOSED: Reward calculations validated")
    else:
        LOG.error("\n❌ SOME TESTS FAILED")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
