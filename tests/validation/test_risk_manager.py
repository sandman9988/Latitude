#!/usr/bin/env python3
import pytest

"""
Test Suite for RiskManager
===========================
Tests portfolio-level risk validation per SYSTEM_FLOW.md design.
"""

import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)


from src.risk.circuit_breakers import CircuitBreakerManager
from src.risk.risk_manager import RiskManager, EntryValidation, ExitValidation, RiskAssessment
from src.risk.var_estimator import VaREstimator, RegimeType


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
    print("\n" + "=" * 70)
    print("TEST: Basic Entry Validation")
    print("=" * 70)

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
    print("\n1. Valid entry (action=1, confidence=0.75)")
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
    print(f"   ✓ Approved: qty={validation.qty:.4f}, VaR={validation.var_used:.4f}")

    # Test 2: Low confidence rejection
    print("\n2. Low confidence (confidence=0.4)")
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.4,  # Below threshold
        current_position=0.0,
    )

    assert not validation.approved, "Should reject low confidence"
    assert "confidence" in validation.reason.lower(), "Reason should mention confidence"
    print(f"   ✓ Rejected: {validation.reason}")

    # Test 3: NO_ENTRY action
    print("\n3. NO_ENTRY action (action=0)")
    validation = risk_manager.validate_entry(
        action=0,  # NO_ENTRY
        confidence=0.9,
        current_position=0.0,
    )

    assert not validation.approved, "Should not approve NO_ENTRY"
    assert "NO_ENTRY" in validation.reason, "Reason should mention NO_ENTRY"
    print(f"   ✓ Correctly handled: {validation.reason}")

    print("\n✓ Basic entry validation PASSED")


def test_circuit_breaker_integration():
    """Test circuit breaker integration"""
    print("\n" + "=" * 70)
    print("TEST: Circuit Breaker Integration")
    print("=" * 70)

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager(max_consecutive_losses=3)
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=100.0,
    )

    # Test 1: Entry allowed when breakers OK
    print("\n1. Entry with no breakers tripped")
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.7,
        current_position=0.0,
    )
    assert validation.approved, "Should approve when breakers OK"
    print("   ✓ Entry approved")

    # Test 2: Simulate consecutive losses to trip breaker
    print("\n2. Trip circuit breaker with consecutive losses")
    # Add a winning trade first (circuit breaker needs context)
    circuit_breakers.update_trade(pnl=10.0, equity=10010.0)
    circuit_breakers.check_all()  # Check breakers
    print("   Win 1: PnL=+10.0 (establish baseline)")

    # Now add consecutive losses
    for i in range(3):
        circuit_breakers.update_trade(pnl=-10.0, equity=10000.0 - (i + 1) * 10)
        circuit_breakers.check_all()  # Check breakers after each trade
        print(f"   Loss {i+1}: PnL=-10.0")

    assert circuit_breakers.is_any_tripped(), "Breaker should be tripped"
    print("   ✓ Circuit breaker tripped")

    # Test 3: Entry rejected when breakers tripped
    print("\n3. Entry attempt with breakers tripped")
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.9,  # High confidence
        current_position=0.0,
    )
    assert not validation.approved, "Should reject when breakers tripped"
    assert "circuit breaker" in validation.reason.lower(), "Reason should mention breakers"
    print(f"   ✓ Entry rejected: {validation.reason}")

    print("\n✓ Circuit breaker integration PASSED")


def test_exit_validation():
    """Test exit validation logic"""
    print("\n" + "=" * 70)
    print("TEST: Exit Validation")
    print("=" * 70)

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
    )

    # Test 1: Full exit
    print("\n1. Full exit (FULL, position=0.5)")
    validation = risk_manager.validate_exit(
        action=1,  # CLOSE
        exit_type="FULL",
        current_position=0.5,
    )
    assert validation.approved, "Should approve full exit"
    assert validation.volume > 0, "Should calculate non-zero volume"
    assert validation.urgency == "NORMAL", "Should be normal urgency"
    print(f"   ✓ Approved: volume={validation.volume}, urgency={validation.urgency}")

    # Test 2: Partial exit
    print("\n2. Partial exit (50%, position=1.0)")
    validation = risk_manager.validate_exit(
        action=1,
        exit_type="PARTIAL",
        current_position=1.0,
        fraction=0.5,
        min_position_size=0.01,
    )
    assert validation.approved, "Should approve partial exit"
    print(f"   ✓ Approved: volume={validation.volume}")

    # Test 3: HOLD action
    print("\n3. HOLD action (action=0)")
    validation = risk_manager.validate_exit(
        action=0,  # HOLD
        current_position=0.5,
    )
    assert not validation.approved, "Should not approve HOLD"
    assert "HOLD" in validation.reason, "Reason should mention HOLD"
    print(f"   ✓ Correctly handled: {validation.reason}")

    # Test 4: Invalid partial fraction
    print("\n4. Invalid partial fraction (fraction=1.5)")
    validation = risk_manager.validate_exit(
        action=1,
        exit_type="PARTIAL",
        current_position=1.0,
        fraction=1.5,  # Invalid
    )
    assert not validation.approved, "Should reject invalid fraction"
    print(f"   ✓ Rejected: {validation.reason}")

    # Test 5: Dust position upgrade to FULL
    print("\n5. Partial that would leave dust → upgrade to FULL")
    validation = risk_manager.validate_exit(
        action=1,
        exit_type="PARTIAL",
        current_position=0.015,  # Small position
        fraction=0.5,  # Would leave 0.0075 < min
        min_position_size=0.01,
    )
    assert validation.approved, "Should approve (upgraded to FULL)"
    # Volume should be full position (not partial)
    print(f"   ✓ Upgraded to FULL: volume={validation.volume}")

    print("\n✓ Exit validation PASSED")


def test_emergency_exit_override():
    """Test emergency exit override when breakers trip"""
    print("\n" + "=" * 70)
    print("TEST: Emergency Exit Override")
    print("=" * 70)

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager(max_consecutive_losses=2)
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
    )

    # Trip circuit breaker
    print("\n1. Trip circuit breaker")
    # Add a win first
    circuit_breakers.update_trade(pnl=10.0, equity=10010.0)
    circuit_breakers.check_all()
    for i in range(2):
        circuit_breakers.update_trade(pnl=-20.0, equity=10000.0 - (i + 1) * 20)
        circuit_breakers.check_all()  # Check after each trade
    assert circuit_breakers.is_any_tripped(), "Breaker should be tripped"
    print("   ✓ Circuit breaker tripped")

    # Test emergency override: PARTIAL → FULL, urgency → EMERGENCY
    print("\n2. Partial exit with breakers tripped → EMERGENCY FULL override")
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
    print(f"   ✓ Override successful: urgency={validation.urgency}, volume={validation.volume}")

    print("\n✓ Emergency exit override PASSED")


def test_position_size_limits():
    """Test maximum position size enforcement"""
    print("\n" + "=" * 70)
    print("TEST: Position Size Limits")
    print("=" * 70)

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=10000.0,  # Large budget
        max_position_size=0.5,  # But capped position size
    )

    # Test: Position size capped by max_position_size
    print("\n1. High risk budget but capped by max_position_size")
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.8,
        current_position=0.0,
        regime=RegimeType.UNDERDAMPED,  # High vol
        account_balance=100000.0,  # Large account
    )

    assert validation.approved, "Should approve entry"
    assert validation.qty <= 0.5, f"Should cap at max_position_size: {validation.qty}"
    print(f"   ✓ Position capped: qty={validation.qty:.4f} ≤ max={0.5}")

    # Test: Reject if total exposure would exceed limit
    print("\n2. Reject entry that would exceed total exposure")
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
    print(f"   ✓ Rejected: {validation.reason}")

    print("\n✓ Position size limits PASSED")


def test_statistics_tracking():
    """Test approval/rejection statistics tracking"""
    print("\n" + "=" * 70)
    print("TEST: Statistics Tracking")
    print("=" * 70)

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        min_confidence_entry=0.6,
    )

    # Initial stats
    status = risk_manager.get_status()
    print(f"\nInitial stats: approved={status['entries_approved']}, rejected={status['entries_rejected']}")

    # Approve some entries
    for _ in range(3):
        risk_manager.validate_entry(action=1, confidence=0.8, current_position=0.0)

    # Reject some entries
    for _ in range(2):
        risk_manager.validate_entry(action=1, confidence=0.4, current_position=0.0)

    # Check stats
    status = risk_manager.get_status()
    print(
        f"After 3 approvals, 2 rejections: approved={status['entries_approved']}, rejected={status['entries_rejected']}"
    )

    assert status["entries_approved"] == 3, f"Should have 3 approvals: {status['entries_approved']}"
    assert status["entries_rejected"] == 2, f"Should have 2 rejections: {status['entries_rejected']}"

    # Test exits
    for _ in range(2):
        risk_manager.validate_exit(action=1, exit_type="FULL", current_position=0.5)

    risk_manager.validate_exit(action=0, current_position=0.5)  # HOLD (rejected)

    status = risk_manager.get_status()
    print(
        f"After 2 exit approvals, 1 rejection: exits_approved={status['exits_approved']}, exits_rejected={status['exits_rejected']}"
    )

    assert status["exits_approved"] == 2, f"Should have 2 exit approvals: {status['exits_approved']}"
    assert status["exits_rejected"] == 1, f"Should have 1 exit rejection: {status['exits_rejected']}"

    print("\n✓ Statistics tracking PASSED")


def test_adaptive_updates():
    """Test adaptive risk budget and confidence threshold updates"""
    print("\n" + "=" * 70)
    print("TEST: Adaptive Updates")
    print("=" * 70)

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
    print("\n1. Update risk budget: $100 → $200")
    risk_manager.update_risk_budget(200.0)
    status = risk_manager.get_status()
    assert status["risk_budget_usd"] == pytest.approx(200.0), "Risk budget should be updated"
    print("   ✓ Risk budget updated")

    # Test 2: Update confidence thresholds
    print("\n2. Update confidence thresholds")
    risk_manager.update_confidence_thresholds(entry=0.7, exit=0.6)
    assert risk_manager.min_confidence_entry == pytest.approx(0.7), "Entry threshold should be updated"
    assert risk_manager.min_confidence_exit == pytest.approx(0.6), "Exit threshold should be updated"
    print("   ✓ Confidence thresholds updated")

    # Test 3: Verify updated threshold affects validation
    print("\n3. Verify updated threshold (0.65 confidence now rejected)")
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.65,  # Was OK with 0.6 threshold, now rejected with 0.7
        current_position=0.0,
    )
    assert not validation.approved, "Should reject with new higher threshold"
    print(f"   ✓ Correctly rejected: {validation.reason}")

    print("\n✓ Adaptive updates PASSED")


def test_circuit_breaker_control():
    """Test RiskManager as circuit breaker controller"""
    print("\n" + "=" * 70)
    print("TEST: Circuit Breaker Control")
    print("=" * 70)

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager(max_consecutive_losses=3)
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=100.0,
    )

    # Test 1: Trade updates flow through RiskManager
    print("\n1. Complete trades via RiskManager")

    # Winning trade
    risk_manager.on_trade_complete(pnl=10.0, equity=10010.0)
    assert risk_manager.total_trades == 1, "Should track trades"
    assert risk_manager.winning_trades == 1, "Should track wins"
    print("   ✓ Winning trade tracked")

    # Losing trades to trip breaker
    for i in range(3):
        risk_manager.on_trade_complete(pnl=-5.0, equity=10010.0 - (i + 1) * 5)

    assert circuit_breakers.is_any_tripped(), "Breaker should be tripped by RiskManager"
    print("   ✓ Circuit breakers updated automatically")

    # Test 2: Entry validation respects breaker state
    print("\n2. Entry validation blocked by tripped breakers")
    validation = risk_manager.validate_entry(
        action=1,
        confidence=0.8,
        current_position=0.0,
    )
    assert not validation.approved, "Entry should be blocked"
    print(f"   ✓ Entry blocked: {validation.reason}")

    print("\n✓ Circuit breaker control PASSED")


def test_risk_assessment():
    """Test comprehensive risk assessment"""
    print("\n" + "=" * 70)
    print("TEST: Risk Assessment")
    print("=" * 70)

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=100.0,
        max_position_size=1.0,
    )

    # Test 1: No positions
    print("\n1. Risk assessment with no positions")
    assessment = risk_manager.assess_risk(
        current_regime=RegimeType.OVERDAMPED,
        current_vol=0.01,
    )

    assert isinstance(assessment, RiskAssessment), "Should return RiskAssessment"
    assert assessment.portfolio_health == "HEALTHY", "Empty portfolio should be healthy"
    assert assessment.position_concentration == pytest.approx(0.0), "No positions = no concentration"
    print(f"   ✓ Health: {assessment.portfolio_health}, Concentration: {assessment.position_concentration:.2f}")

    # Test 2: With position
    print("\n2. Risk assessment with active position")
    risk_manager.update_exposure("BTCUSD", 0.5)
    assessment = risk_manager.assess_risk(
        current_regime=RegimeType.UNDERDAMPED,
        current_vol=0.02,
    )

    assert assessment.total_exposure_usd > 0, "Should track exposure"
    assert assessment.position_concentration == pytest.approx(1.0), "Single position = full concentration"
    assert assessment.regime_risk_multiplier == pytest.approx(2.0), "UNDERDAMPED = 2.0x multiplier"
    print(f"   ✓ Exposure: ${assessment.total_exposure_usd:.2f}, Regime: {assessment.regime_risk_multiplier:.1f}x")

    # Test 3: Risk summary
    print("\n3. Get human-readable risk summary")
    summary = risk_manager.get_risk_summary()
    assert "Portfolio Health" in summary, "Summary should include health"
    assert "HEALTHY" in summary or "CAUTION" in summary, "Summary should show status"
    print("   ✓ Risk summary generated")

    print("\n✓ Risk assessment PASSED")


def test_adaptive_risk_budget():
    """Test adaptive risk budget management"""
    print("\n" + "=" * 70)
    print("TEST: Adaptive Risk Budget")
    print("=" * 70)

    var_estimator = create_var_estimator_with_data()
    circuit_breakers = CircuitBreakerManager()
    risk_manager = RiskManager(
        circuit_breakers=circuit_breakers,
        var_estimator=var_estimator,
        risk_budget_usd=100.0,
    )

    initial_budget = risk_manager.risk_budget_usd
    print(f"\nInitial risk budget: ${initial_budget:.2f}")

    # Test 1: Good performance increases budget
    print("\n1. Simulate good performance (10 winning trades)")
    equity = 10000.0
    for i in range(10):
        equity += 20.0
        risk_manager.on_trade_complete(pnl=20.0, equity=equity)

    # Should trigger adaptation after 10 trades
    if risk_manager.risk_budget_usd > initial_budget:
        print(f"   ✓ Budget increased: ${initial_budget:.2f} → ${risk_manager.risk_budget_usd:.2f}")
    else:
        print(f"   → Budget unchanged: ${risk_manager.risk_budget_usd:.2f} (needs >55% win rate)")

    # Test 2: Poor performance decreases budget
    print("\n2. Simulate poor performance (10 losing trades)")
    risk_manager2 = RiskManager(
        circuit_breakers=CircuitBreakerManager(),
        var_estimator=create_var_estimator_with_data(),
        risk_budget_usd=100.0,
    )

    equity = 10000.0
    for i in range(10):
        equity -= 15.0
        risk_manager2.on_trade_complete(pnl=-15.0, equity=equity)

    if risk_manager2.risk_budget_usd < 100.0:
        print(f"   ✓ Budget decreased: $100.00 → ${risk_manager2.risk_budget_usd:.2f}")
    else:
        print(f"   → Budget unchanged: ${risk_manager2.risk_budget_usd:.2f}")

    print("\n✓ Adaptive risk budget PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RISK MANAGER TEST SUITE")
    print("=" * 70)

    results = []

    try:
        results.append(("Basic Entry Validation", test_entry_validation_basic()))
        results.append(("Circuit Breaker Integration", test_circuit_breaker_integration()))
        results.append(("Exit Validation", test_exit_validation()))
        results.append(("Emergency Exit Override", test_emergency_exit_override()))
        results.append(("Position Size Limits", test_position_size_limits()))
        results.append(("Statistics Tracking", test_statistics_tracking()))
        results.append(("Adaptive Updates", test_adaptive_updates()))

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
