"""
Test Composite Probability Predictor - Multi-Agent Calibration

Tests the enhanced RiskManager that tracks predictions separately for:
- TriggerAgent (entry decisions)
- HarvesterAgent (exit decisions)
- Composite (combined view)

This creates a composite probability prediction tool that measures each
agent's predicted vs actual outcomes and becomes the main risk management tool.
"""

import numpy as np
from circuit_breakers import CircuitBreakerManager
from risk_manager import RiskManager
from var_estimator import VaREstimator


def test_per_agent_calibration():
    """Test tracking predictions separately for TriggerAgent vs HarvesterAgent"""
    print("\n" + "=" * 70)
    print("TEST: Per-Agent Probability Calibration")
    print("=" * 70)

    var_est = VaREstimator(window=100, confidence=0.95)
    breakers = CircuitBreakerManager()
    risk_mgr = RiskManager(
        circuit_breakers=breakers,
        var_estimator=var_est,
        risk_budget_usd=100.0,
        symbol="BTCUSD",
    )

    print("\n1. TriggerAgent predictions (entry decisions)")
    print("   Simulate well-calibrated trigger (70% confidence → 70% win rate)")

    # TriggerAgent: 70% confidence, 70% actual win rate (WELL CALIBRATED)
    trigger_outcomes = [True] * 7 + [False] * 3
    for i, outcome in enumerate(trigger_outcomes):
        risk_mgr.update_decision_outcome(
            decision_type="entry", confidence=0.7, approved=True, actual_outcome=outcome, agent_id="trigger"
        )
        print(f"   Trigger trade {i+1}: {'Win' if outcome else 'Loss'}")

    # Get TriggerAgent calibration
    trigger_calib = risk_mgr.get_probability_calibration("trigger")
    if 0.7 in trigger_calib:
        c = trigger_calib[0.7]
        print(f"\n   TriggerAgent 70% bucket:")
        print(f"   Predicted: {c.predicted_success_rate:.1%}")
        print(f"   Actual: {c.actual_success_rate:.1%}")
        print(f"   Error: {c.calibration_error:.1%}")
        print(f"   Well calibrated: {c.is_well_calibrated}")
        assert c.agent_id == "trigger"
        assert c.is_well_calibrated

    print("\n2. HarvesterAgent predictions (exit decisions)")
    print("   Simulate poorly-calibrated harvester (90% confidence → 50% win rate)")

    # HarvesterAgent: 90% confidence, 50% actual win rate (OVERCONFIDENT)
    harvester_outcomes = [True] * 5 + [False] * 5
    for i, outcome in enumerate(harvester_outcomes):
        risk_mgr.update_decision_outcome(
            decision_type="exit", confidence=0.9, approved=True, actual_outcome=outcome, agent_id="harvester"
        )
        print(f"   Harvester trade {i+1}: {'Win' if outcome else 'Loss'}")

    # Get HarvesterAgent calibration
    harvester_calib = risk_mgr.get_probability_calibration("harvester")
    if 0.9 in harvester_calib:
        c = harvester_calib[0.9]
        print(f"\n   HarvesterAgent 90% bucket:")
        print(f"   Predicted: {c.predicted_success_rate:.1%}")
        print(f"   Actual: {c.actual_success_rate:.1%}")
        print(f"   Error: {c.calibration_error:.1%}")
        print(f"   Well calibrated: {c.is_well_calibrated}")
        assert c.agent_id == "harvester"
        assert not c.is_well_calibrated  # Should be poorly calibrated

    print("\n3. Verify separation (agents don't interfere)")
    # TriggerAgent should still be well calibrated despite HarvesterAgent being poor
    trigger_calib_after = risk_mgr.get_probability_calibration("trigger")
    assert 0.7 in trigger_calib_after
    assert trigger_calib_after[0.7].is_well_calibrated
    print("   ✓ TriggerAgent still well-calibrated")
    print("   ✓ HarvesterAgent independently tracked")

    print("\n✓ Per-agent calibration tracking PASSED")
    return True


def test_composite_probability_predictor():
    """Test composite predictor that combines both agents"""
    print("\n" + "=" * 70)
    print("TEST: Composite Probability Predictor (Main Risk Tool)")
    print("=" * 70)

    var_est = VaREstimator(window=100, confidence=0.95)
    breakers = CircuitBreakerManager()
    risk_mgr = RiskManager(
        circuit_breakers=breakers,
        var_estimator=var_est,
        risk_budget_usd=100.0,
        symbol="BTCUSD",
    )

    print("\n1. Populate data for both agents")

    # TriggerAgent: Well-calibrated at multiple confidence levels
    print("   TriggerAgent:")
    for conf, win_rate in [(0.6, 0.6), (0.7, 0.7), (0.8, 0.8)]:
        outcomes = [True] * int(win_rate * 10) + [False] * int((1 - win_rate) * 10)
        for outcome in outcomes:
            risk_mgr.update_decision_outcome("entry", conf, True, outcome, "trigger")
        print(f"   - {conf:.0%} confidence: {int(win_rate*100)}% win rate")

    # HarvesterAgent: Overconfident at all levels
    print("   HarvesterAgent (overconfident):")
    for conf in [0.7, 0.8, 0.9]:
        # Always 50% win rate regardless of confidence (poor calibration)
        outcomes = [True] * 5 + [False] * 5
        for outcome in outcomes:
            risk_mgr.update_decision_outcome("exit", conf, True, outcome, "harvester")
        print(f"   - {conf:.0%} confidence: 50% win rate (should be {conf:.0%})")

    print("\n2. Get composite predictor")
    composite = risk_mgr.get_composite_probability_predictor()

    print(f"\n   COMPOSITE ANALYSIS:")
    print(f"   ==================")
    print(f"   TriggerAgent:")
    print(f"   - Overall accuracy: {composite.trigger_overall_accuracy:.1%}")
    print(f"   - Buckets calibrated: {len(composite.trigger_calibration)}")

    print(f"\n   HarvesterAgent:")
    print(f"   - Overall accuracy: {composite.harvester_overall_accuracy:.1%}")
    print(f"   - Buckets calibrated: {len(composite.harvester_calibration)}")

    print(f"\n   Best calibrated: {composite.best_calibrated_agent.upper()}")
    print(f"   Recommendation: {composite.recommendation}")

    # Verify TriggerAgent is better calibrated
    assert composite.best_calibrated_agent == "trigger"
    assert "TriggerAgent" in composite.recommendation or "trust" in composite.recommendation.lower()

    print("\n3. Detailed per-agent calibration")
    print("\n   TriggerAgent Calibration:")
    for bucket, calib in sorted(composite.trigger_calibration.items()):
        status = "✓" if calib.is_well_calibrated else "✗"
        print(
            f"   {status} {bucket:.0%}: pred={calib.predicted_success_rate:.0%} "
            f"act={calib.actual_success_rate:.0%} err={calib.calibration_error:.1%} (n={calib.sample_size})"
        )

    print("\n   HarvesterAgent Calibration:")
    for bucket, calib in sorted(composite.harvester_calibration.items()):
        status = "✓" if calib.is_well_calibrated else "✗"
        print(
            f"   {status} {bucket:.0%}: pred={calib.predicted_success_rate:.0%} "
            f"act={calib.actual_success_rate:.0%} err={calib.calibration_error:.1%} (n={calib.sample_size})"
        )

    print("\n✓ Composite probability predictor PASSED")
    return True


def test_composite_in_risk_assessment():
    """Test that composite predictor is integrated into risk assessment"""
    print("\n" + "=" * 70)
    print("TEST: Composite Predictor in Risk Assessment")
    print("=" * 70)

    var_est = VaREstimator(window=100, confidence=0.95)
    breakers = CircuitBreakerManager()
    risk_mgr = RiskManager(
        circuit_breakers=breakers,
        var_estimator=var_est,
        risk_budget_usd=100.0,
        symbol="BTCUSD",
    )

    print("\n1. Create scenario with different agent performance")

    # TriggerAgent: Good performance
    for i in range(20):
        risk_mgr.update_decision_outcome("entry", 0.75, True, True, "trigger")  # 100% win

    # HarvesterAgent: Poor performance
    for i in range(20):
        risk_mgr.update_decision_outcome("exit", 0.75, True, False, "harvester")  # 0% win

    print("   TriggerAgent: 20 wins")
    print("   HarvesterAgent: 20 losses")

    print("\n2. Run risk assessment")
    risk_mgr.active_positions["BTCUSD"] = 0.5
    assessment = risk_mgr.assess_risk()

    print(f"\n   Portfolio Health: {assessment.portfolio_health}")
    print(f"   Recommendations ({len(assessment.recommendations)}):")
    for rec in assessment.recommendations[:10]:
        print(f"   • {rec}")

    # Verify composite predictor is included
    assert assessment.composite_predictor is not None
    print(f"\n3. Composite predictor in assessment:")
    print(f"   Trigger accuracy: {assessment.composite_predictor.trigger_overall_accuracy:.1%}")
    print(f"   Harvester accuracy: {assessment.composite_predictor.harvester_overall_accuracy:.1%}")
    print(f"   Best agent: {assessment.composite_predictor.best_calibrated_agent}")
    print(f"   Recommendation: {assessment.composite_predictor.recommendation}")

    # Should recommend trusting TriggerAgent
    assert assessment.composite_predictor.best_calibrated_agent == "trigger"
    assert (
        assessment.composite_predictor.trigger_overall_accuracy
        > assessment.composite_predictor.harvester_overall_accuracy
    )

    print("\n✓ Composite predictor integration PASSED")
    return True


def test_adaptive_trust_weighting():
    """Test that system can identify which agent to trust more"""
    print("\n" + "=" * 70)
    print("TEST: Adaptive Trust Weighting")
    print("=" * 70)

    var_est = VaREstimator(window=100, confidence=0.95)
    breakers = CircuitBreakerManager()
    risk_mgr = RiskManager(
        circuit_breakers=breakers,
        var_estimator=var_est,
        risk_budget_usd=100.0,
        symbol="BTCUSD",
    )

    print("\n1. Scenario A: HarvesterAgent outperforms TriggerAgent")

    # TriggerAgent: Overconfident (80% conf → 50% win)
    for _ in range(10):
        risk_mgr.update_decision_outcome("entry", 0.8, True, True, "trigger")
        risk_mgr.update_decision_outcome("entry", 0.8, True, False, "trigger")

    # HarvesterAgent: Well-calibrated (80% conf → 80% win)
    for _ in range(10):
        outcomes = [True] * 8 + [False] * 2
        for outcome in outcomes:
            risk_mgr.update_decision_outcome("exit", 0.8, True, outcome, "harvester")

    composite_a = risk_mgr.get_composite_probability_predictor()
    print(f"\n   Trigger error: {np.mean([c.calibration_error for c in composite_a.trigger_calibration.values()]):.1%}")
    print(
        f"   Harvester error: {np.mean([c.calibration_error for c in composite_a.harvester_calibration.values()]):.1%}"
    )
    print(f"   Best: {composite_a.best_calibrated_agent}")
    print(f"   Recommendation: {composite_a.recommendation}")

    assert composite_a.best_calibrated_agent == "harvester"
    print("   ✓ Correctly identified HarvesterAgent as better calibrated")

    print("\n2. Scenario B: Both equally calibrated")

    # Reset
    risk_mgr.calibration_buckets_trigger = {0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}
    risk_mgr.calibration_buckets_harvester = {0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}

    # Both well-calibrated at 70%
    for agent_id in ["trigger", "harvester"]:
        outcomes = [True] * 7 + [False] * 3
        for outcome in outcomes:
            risk_mgr.update_decision_outcome("entry", 0.7, True, outcome, agent_id)

    composite_b = risk_mgr.get_composite_probability_predictor()
    print(f"\n   Trigger accuracy: {composite_b.trigger_overall_accuracy:.1%}")
    print(f"   Harvester accuracy: {composite_b.harvester_overall_accuracy:.1%}")
    print(f"   Recommendation: {composite_b.recommendation}")

    assert "equally" in composite_b.recommendation.lower() or "both" in composite_b.recommendation.lower()
    print("   ✓ Correctly identified equal calibration")

    print("\n✓ Adaptive trust weighting PASSED")
    return True


def main():
    """Run all composite predictor tests"""
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  COMPOSITE PROBABILITY PREDICTOR TEST SUITE".center(68) + "║")
    print("║" + "  (Multi-Agent Calibration)".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    results = []

    # Run tests
    results.append(("Per-Agent Calibration", test_per_agent_calibration()))
    results.append(("Composite Probability Predictor", test_composite_probability_predictor()))
    results.append(("Composite in Risk Assessment", test_composite_in_risk_assessment()))
    results.append(("Adaptive Trust Weighting", test_adaptive_trust_weighting()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 70)

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("✓ ALL COMPOSITE PREDICTOR TESTS PASSED")
        print("\nThe RiskManager now tracks:")
        print("  • TriggerAgent predictions vs actuals")
        print("  • HarvesterAgent predictions vs actuals")
        print("  • Composite view of both agents")
        print("  • Automatic identification of which agent to trust")
        print("\nThis is the MAIN RISK MANAGEMENT TOOL for probability predictions!")
    else:
        print("✗ SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
