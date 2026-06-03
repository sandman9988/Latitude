"""
Test Composite Probability Predictor - Multi-Agent Calibration

Tests the enhanced RiskManager that tracks predictions separately for:
- TriggerAgent (entry decisions)
- HarvesterAgent (exit decisions)
- Composite (combined view)

This creates a composite probability prediction tool that measures each
agent's predicted vs actual outcomes and becomes the main risk management tool.
"""


from src.risk.circuit_breakers import CircuitBreakerManager
from src.risk.risk_manager import RiskManager
from src.risk.var_estimator import VaREstimator


def test_per_agent_calibration():
    """Test tracking predictions separately for TriggerAgent vs HarvesterAgent"""

    var_est = VaREstimator(window=100, confidence=0.95)
    breakers = CircuitBreakerManager()
    risk_mgr = RiskManager(
        circuit_breakers=breakers,
        var_estimator=var_est,
        risk_budget_usd=100.0,
        symbol="BTCUSD",
    )


    # TriggerAgent: 70% confidence, 70% actual win rate (WELL CALIBRATED)
    trigger_outcomes = [True] * 7 + [False] * 3
    for _i, outcome in enumerate(trigger_outcomes):
        risk_mgr.update_decision_outcome(
            decision_type="entry", confidence=0.7, approved=True, actual_outcome=outcome, agent_id="trigger",
        )

    # Get TriggerAgent calibration
    trigger_calib = risk_mgr.get_probability_calibration("trigger")
    if 0.7 in trigger_calib:
        c = trigger_calib[0.7]
        assert c.agent_id == "trigger"
        assert c.is_well_calibrated


    # HarvesterAgent: 90% confidence, 50% actual win rate (OVERCONFIDENT)
    harvester_outcomes = [True] * 5 + [False] * 5
    for _i, outcome in enumerate(harvester_outcomes):
        risk_mgr.update_decision_outcome(
            decision_type="exit", confidence=0.9, approved=True, actual_outcome=outcome, agent_id="harvester",
        )

    # Get HarvesterAgent calibration
    harvester_calib = risk_mgr.get_probability_calibration("harvester")
    if 0.9 in harvester_calib:
        c = harvester_calib[0.9]
        assert c.agent_id == "harvester"
        assert not c.is_well_calibrated  # Should be poorly calibrated

    # TriggerAgent should still be well calibrated despite HarvesterAgent being poor
    trigger_calib_after = risk_mgr.get_probability_calibration("trigger")
    assert 0.7 in trigger_calib_after
    assert trigger_calib_after[0.7].is_well_calibrated



def test_composite_probability_predictor():
    """Test composite predictor that combines both agents"""

    var_est = VaREstimator(window=100, confidence=0.95)
    breakers = CircuitBreakerManager()
    risk_mgr = RiskManager(
        circuit_breakers=breakers,
        var_estimator=var_est,
        risk_budget_usd=100.0,
        symbol="BTCUSD",
    )


    # TriggerAgent: Well-calibrated at multiple confidence levels
    for conf, win_rate in [(0.6, 0.6), (0.7, 0.7), (0.8, 0.8)]:
        outcomes = [True] * int(win_rate * 10) + [False] * int((1 - win_rate) * 10)
        for outcome in outcomes:
            risk_mgr.update_decision_outcome("entry", conf, True, outcome, "trigger")

    # HarvesterAgent: Overconfident at all levels
    for conf in [0.7, 0.8, 0.9]:
        # Always 50% win rate regardless of confidence (poor calibration)
        outcomes = [True] * 5 + [False] * 5
        for outcome in outcomes:
            risk_mgr.update_decision_outcome("exit", conf, True, outcome, "harvester")

    composite = risk_mgr.get_composite_probability_predictor()




    # Verify TriggerAgent is better calibrated
    assert composite.best_calibrated_agent == "trigger"
    assert "TriggerAgent" in composite.recommendation or "trust" in composite.recommendation.lower()

    for _bucket, _calib in sorted(composite.trigger_calibration.items()):
        pass

    for _bucket, _calib in sorted(composite.harvester_calibration.items()):
        pass



def test_composite_in_risk_assessment():
    """Test that composite predictor is integrated into risk assessment"""

    var_est = VaREstimator(window=100, confidence=0.95)
    breakers = CircuitBreakerManager()
    risk_mgr = RiskManager(
        circuit_breakers=breakers,
        var_estimator=var_est,
        risk_budget_usd=100.0,
        symbol="BTCUSD",
    )


    # TriggerAgent: Good performance
    for _i in range(20):
        risk_mgr.update_decision_outcome("entry", 0.75, True, True, "trigger")  # 100% win

    # HarvesterAgent: Poor performance
    for _i in range(20):
        risk_mgr.update_decision_outcome("exit", 0.75, True, False, "harvester")  # 0% win


    risk_mgr.active_positions["BTCUSD"] = 0.5
    assessment = risk_mgr.assess_risk()

    for _rec in assessment.recommendations[:10]:
        pass

    # Verify composite predictor is included
    assert assessment.composite_predictor is not None

    # Should recommend trusting TriggerAgent
    assert assessment.composite_predictor.best_calibrated_agent == "trigger"
    assert (
        assessment.composite_predictor.trigger_overall_accuracy
        > assessment.composite_predictor.harvester_overall_accuracy
    )



def test_adaptive_trust_weighting():
    """Test that system can identify which agent to trust more"""

    var_est = VaREstimator(window=100, confidence=0.95)
    breakers = CircuitBreakerManager()
    risk_mgr = RiskManager(
        circuit_breakers=breakers,
        var_estimator=var_est,
        risk_budget_usd=100.0,
        symbol="BTCUSD",
    )


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

    assert composite_a.best_calibrated_agent == "harvester"


    # Reset
    risk_mgr.calibration_buckets_trigger = {0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}
    risk_mgr.calibration_buckets_harvester = {0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []}

    # Both well-calibrated at 70%
    for agent_id in ["trigger", "harvester"]:
        outcomes = [True] * 7 + [False] * 3
        for outcome in outcomes:
            risk_mgr.update_decision_outcome("entry", 0.7, True, outcome, agent_id)

    composite_b = risk_mgr.get_composite_probability_predictor()

    assert "equally" in composite_b.recommendation.lower() or "both" in composite_b.recommendation.lower()



def main():
    """Run all composite predictor tests"""

    results = []

    # Run tests
    results.append(("Per-Agent Calibration", test_per_agent_calibration()))
    results.append(("Composite Probability Predictor", test_composite_probability_predictor()))
    results.append(("Composite in Risk Assessment", test_composite_in_risk_assessment()))
    results.append(("Adaptive Trust Weighting", test_adaptive_trust_weighting()))

    # Summary
    for _name, _passed in results:
        pass


    all_passed = all(r[1] for r in results)
    if all_passed:
        pass
    else:
        pass

    return all_passed


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
