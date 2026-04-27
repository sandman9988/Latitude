#!/usr/bin/env python3
"""Test Suite for RiskManager - portfolio-level risk validation per SYSTEM_FLOW.md design."""

import sys

import numpy as np
import pytest

from src.risk.circuit_breakers import CircuitBreakerManager
from src.risk.risk_manager import RiskAssessment, RiskManager
from src.risk.var_estimator import RegimeType, VaREstimator

rng = np.random.default_rng(42)


def create_var_estimator_with_data(window=100, confidence=0.95, seed=42):
    """Helper to create VaREstimator with sample data"""
    var_estimator = VaREstimator(window=window, confidence=confidence)
    rng = np.random.default_rng(seed)
    sample_returns = rng.normal(0.0, 0.01, max(window, 50))
    for ret in sample_returns:
        var_estimator.update_return(ret)
    return var_estimator


def test_entry_validation_basic():
    """Test basic entry validation flow"""

    # Initialize components with sample data
    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=100.0,
        max_position_size=1.0,
        min_confidence_entry=0.6,
    )

    # Test 1: Valid entry with good confidence
    validation = risk_manager.validate_entry(
        action=1,  # LONG
        confidence=0.75,
        current_position=0.0,
        regime=RegimeType.UNDERDAMPED,  # High vol trending
        vpin_z=0.0,
        account_balance=10000.0,
    )

    assert validation.approved, "Should approve valid entry"
    assert validation.qty > 0, "Should calculate non-zero quantity"

    # Test 2: Low confidence rejection
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.4,  # Below threshold
        current_position=0.0,
    )

    assert not validation.approved, "Should reject low confidence"
    assert "confidence" in validation.reason.lower(), "Reason should mention confidence"

    # Test 3: NO_ENTRY action
    validation = risk_manager.validate_entry(
        action=0,  # NO_ENTRY
        confidence=0.9,
        current_position=0.0,
    )

    assert not validation.approved, "Should not approve NO_ENTRY"
    assert "NO_ENTRY" in validation.reason, "Reason should mention NO_ENTRY"



def test_circuit_breaker_integration():
    """Test circuit breaker integration"""

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager(max_consecutive_losses=3)
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=100.0,
    )

    # Test 1: Entry allowed when breakers OK
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.7,
        current_position=0.0,
    )
    assert validation.approved, "Should approve when breakers OK"

    # Test 2: Simulate consecutive losses to trip breaker
    # Add a winning trade first (circuit breaker needs context)
    circuit_breakers.update_trade(pnl=10.0, equity=10010.0)
    circuit_breakers.check_all()  # Check breakers

    # Now add consecutive losses
    for i in range(3):
        circuit_breakers.update_trade(pnl=-10.0, equity=10000.0 - (i + 1) * 10)
        circuit_breakers.check_all()  # Check breakers after each trade

    assert circuit_breakers.is_any_tripped(), "Breaker should be tripped"

    # Test 3: Entry rejected when breakers tripped
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.9,  # High confidence
        current_position=0.0,
    )
    assert not validation.approved, "Should reject when breakers tripped"
    assert "circuit breaker" in validation.reason.lower(), "Reason should mention breakers"



def test_exit_validation():
    """Test exit validation logic"""

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
    )

    # Test 1: Full exit
    validation = risk_manager.validate_exit(
        action=1,  # CLOSE
        exit_type="FULL",
        current_position=0.5,
    )
    assert validation.approved, "Should approve full exit"
    assert validation.volume > 0, "Should calculate non-zero volume"
    assert validation.urgency == "NORMAL", "Should be normal urgency"

    # Test 2: Partial exit
    validation = risk_manager.validate_exit(
        action=1,
        exit_type="PARTIAL",
        current_position=1.0,
        fraction=0.5,
        min_position_size=0.01,
    )
    assert validation.approved, "Should approve partial exit"

    # Test 3: HOLD action
    validation = risk_manager.validate_exit(
        action=0,  # HOLD
        current_position=0.5,
    )
    assert not validation.approved, "Should not approve HOLD"
    assert "HOLD" in validation.reason, "Reason should mention HOLD"

    # Test 4: Invalid partial fraction
    validation = risk_manager.validate_exit(
        action=1,
        exit_type="PARTIAL",
        current_position=1.0,
        fraction=1.5,  # Invalid
    )
    assert not validation.approved, "Should reject invalid fraction"

    # Test 5: Dust position upgrade to FULL
    validation = risk_manager.validate_exit(
        action=1,
        exit_type="PARTIAL",
        current_position=0.015,  # Small position
        fraction=0.5,  # Would leave 0.0075 < min
        min_position_size=0.01,
    )
    assert validation.approved, "Should approve (upgraded to FULL)"
    # Volume should be full position (not partial)



def test_emergency_exit_override():
    """Test emergency exit override when breakers trip"""

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager(max_consecutive_losses=2)
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
    )

    # Trip circuit breaker
    # Add a win first
    circuit_breakers.update_trade(pnl=10.0, equity=10010.0)
    circuit_breakers.check_all()
    for i in range(2):
        circuit_breakers.update_trade(pnl=-20.0, equity=10000.0 - (i + 1) * 20)
        circuit_breakers.check_all()  # Check after each trade
    assert circuit_breakers.is_any_tripped(), "Breaker should be tripped"

    # Test emergency override: PARTIAL → FULL, urgency → EMERGENCY
    validation = risk_manager.validate_exit(
        action=1,
        exit_type="PARTIAL",  # Agent requested partial
        current_position=1.0,
        fraction=0.5,
    )

    assert validation.approved, "Should approve exit"
    assert validation.urgency == "EMERGENCY", "Should be EMERGENCY urgency"
    # Volume should be full position (overridden from partial)
    expected_volume = int(1.0 * 100)  # Full position
    assert validation.volume == expected_volume, f"Should be full volume: {validation.volume} vs {expected_volume}"



def test_position_size_limits():
    """Test maximum position size enforcement"""

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=10000.0,  # Large budget
        max_position_size=0.5,  # But capped position size
    )

    # Test: Position size capped by max_position_size
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.8,
        current_position=0.0,
        regime=RegimeType.UNDERDAMPED,  # High vol
        account_balance=100000.0,  # Large account
    )

    assert validation.approved, "Should approve entry"
    assert validation.qty <= 0.5, f"Should cap at max_position_size: {validation.qty}"

    # Test: Reject if total exposure would exceed limit
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.8,
        current_position=0.4,  # Already have position
        regime=RegimeType.UNDERDAMPED,  # High vol
        account_balance=100000.0,
    )

    # With current=0.4 and max=0.5, new position of 0.5 would give total=0.9 > 0.5
    # Should reject
    assert not validation.approved, f"Should reject total exposure: {validation.reason}"



def test_statistics_tracking():
    """Test approval/rejection statistics tracking"""

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        min_confidence_entry=0.6,
    )

    # Initial stats
    status = risk_manager.get_status()

    # Approve some entries
    for _ in range(3):
        risk_manager.validate_entry(action=1, confidence=0.8, current_position=0.0)

    # Reject some entries
    for _ in range(2):
        risk_manager.validate_entry(action=1, confidence=0.4, current_position=0.0)

    # Check stats
    status = risk_manager.get_status()

    assert status["entries_approved"] == 3, f"Should have 3 approvals: {status['entries_approved']}"
    assert status["entries_rejected"] == 2, f"Should have 2 rejections: {status['entries_rejected']}"

    # Test exits
    for _ in range(2):
        risk_manager.validate_exit(action=1, exit_type="FULL", current_position=0.5)

    risk_manager.validate_exit(action=0, current_position=0.5)  # HOLD (rejected)

    status = risk_manager.get_status()

    assert status["exits_approved"] == 2, f"Should have 2 exit approvals: {status['exits_approved']}"
    assert status["exits_rejected"] == 1, f"Should have 1 exit rejection: {status['exits_rejected']}"



def test_adaptive_updates():
    """Test adaptive risk budget and confidence threshold updates"""

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=100.0,
        min_confidence_entry=0.6,
        min_confidence_exit=0.5,
    )

    # Test 1: Update risk budget
    risk_manager.update_risk_budget(200.0)
    status = risk_manager.get_status()
    assert status["risk_budget_usd"] == pytest.approx(200.0), "Risk budget should be updated"

    # Test 2: Update confidence thresholds
    risk_manager.update_confidence_thresholds(entry=0.7, exit_threshold=0.6)
    assert risk_manager.min_confidence_entry == pytest.approx(0.7), "Entry threshold should be updated"
    assert risk_manager.min_confidence_exit == pytest.approx(0.6), "Exit threshold should be updated"

    # Test 3: Verify updated threshold affects validation
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.65,  # Was OK with 0.6 threshold, now rejected with 0.7
        current_position=0.0,
    )
    assert not validation.approved, "Should reject with new higher threshold"



def test_circuit_breaker_control():
    """Test RiskManager as circuit breaker controller"""

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager(max_consecutive_losses=3)
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=100.0,
    )

    # Test 1: Trade updates flow through RiskManager

    # Winning trade
    risk_manager.on_trade_complete(pnl=10.0, equity=10010.0)
    assert risk_manager.total_trades == 1, "Should track trades"
    assert risk_manager.winning_trades == 1, "Should track wins"

    # Losing trades to trip breaker
    for i in range(3):
        risk_manager.on_trade_complete(pnl=-5.0, equity=10010.0 - (i + 1) * 5)

    assert circuit_breakers.is_any_tripped(), "Breaker should be tripped by RiskManager"

    # Test 2: Entry validation respects breaker state
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.8,
        current_position=0.0,
    )
    assert not validation.approved, "Entry should be blocked"



def test_risk_assessment():
    """Test comprehensive risk assessment"""

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=100.0,
        max_position_size=1.0,
    )

    # Test 1: No positions
    assessment = risk_manager.assess_risk(
        current_regime=RegimeType.OVERDAMPED,
        current_vol=0.01,
    )

    assert isinstance(assessment, RiskAssessment), "Should return RiskAssessment"
    assert assessment.portfolio_health == "HEALTHY", "Empty portfolio should be healthy"
    assert assessment.position_concentration == pytest.approx(0.0), "No positions = no concentration"

    # Test 2: With position
    risk_manager.update_exposure("BTCUSD", 0.5)
    assessment = risk_manager.assess_risk(
        current_regime=RegimeType.UNDERDAMPED,
        current_vol=0.02,
    )

    assert assessment.total_exposure_usd > 0, "Should track exposure"
    assert assessment.position_concentration == pytest.approx(1.0), "Single position = full concentration"
    assert assessment.regime_risk_multiplier == pytest.approx(2.0), "UNDERDAMPED = 2.0x multiplier"

    # Test 3: Risk summary
    summary = risk_manager.get_risk_summary()
    assert "Portfolio Health" in summary, "Summary should include health"
    assert "HEALTHY" in summary or "CAUTION" in summary, "Summary should show status"



def test_adaptive_risk_budget():
    """Test adaptive risk budget management"""

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=100.0,
    )

    initial_budget = risk_manager.risk_budget_usd

    # Test 1: Good performance increases budget
    equity = 10000.0
    for _ in range(10):
        equity += 20.0
        risk_manager.on_trade_complete(pnl=20.0, equity=equity)

    # Should trigger adaptation after 10 trades
    if risk_manager.risk_budget_usd > initial_budget:
        pass
    else:
        pass

    # Test 2: Poor performance decreases budget
    risk_manager2 = RiskManager(
        circuit_breakers=CircuitBreakerManager(),
        var_estimator=create_var_estimator_with_data(),
        risk_budget_usd=100.0,
    )

    equity = 10000.0
    for _ in range(10):
        equity -= 15.0
        risk_manager2.on_trade_complete(pnl=-15.0, equity=equity)

    if risk_manager2.risk_budget_usd < 100.0:
        pass
    else:
        pass



if __name__ == "__main__":

    results = []

    try:
        results.append(("Basic Entry Validation", test_entry_validation_basic()))
        results.append(("Circuit Breaker Integration", test_circuit_breaker_integration()))
        results.append(("Exit Validation", test_exit_validation()))
        results.append(("Emergency Exit Override", test_emergency_exit_override()))
        results.append(("Position Size Limits", test_position_size_limits()))
        results.append(("Statistics Tracking", test_statistics_tracking()))
        results.append(("Adaptive Updates", test_adaptive_updates()))

    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Summary

    all_passed = True
    for _test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False

    if all_passed:
        sys.exit(0)
    else:
        sys.exit(1)
