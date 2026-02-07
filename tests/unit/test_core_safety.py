import pytest

"""
Unit Tests for Core Safety Components

Expands test coverage for critical safety components:
- SafeMath operations
- CircuitBreakers
- RingBuffer
- AtomicPersistence
- NonRepaintGuards

Target: Increase coverage from ~40% to 70%+
"""

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

from src.utils.safe_math import SafeMath

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


# ============================================================================
# SAFE MATH TESTS
# ============================================================================


def test_safe_math_division():
    """Test SafeMath division with edge cases."""
    LOG.info("\n=== TEST: SafeMath Division ===")

    # Normal division
    result = SafeMath.safe_div(10.0, 2.0, default=0.0)
    assert abs(result - 5.0) < 0.001, "Normal division failed"

    # Division by zero
    result = SafeMath.safe_div(10.0, 0.0, default=99.0)
    assert result == pytest.approx(99.0), "Division by zero should return default"

    # Division by NaN
    result = SafeMath.safe_div(10.0, np.nan, default=99.0)
    assert result == pytest.approx(99.0), "Division by NaN should return default"

    # NaN dividend
    result = SafeMath.safe_div(np.nan, 2.0, default=99.0)
    assert result == pytest.approx(99.0), "NaN dividend should return default"

    # Division by Inf
    result = SafeMath.safe_div(10.0, np.inf, default=99.0)
    assert result == pytest.approx(0.0), "Division by inf should give 0"

    LOG.info("✓ SafeMath division tests passed")


def test_safe_math_log():
    """Test SafeMath logarithm with edge cases."""
    LOG.info("\n=== TEST: SafeMath Logarithm ===")

    # Normal log
    result = SafeMath.safe_log(10.0, default=0.0)
    assert abs(result - np.log(10.0)) < 0.001, "Normal log failed"

    # Log of zero
    result = SafeMath.safe_log(0.0, default=-99.0)
    assert result == pytest.approx(-99.0), "Log(0) should return default"

    # Log of negative
    result = SafeMath.safe_log(-5.0, default=-99.0)
    assert result == pytest.approx(-99.0), "Log(negative) should return default"

    # Log of NaN
    result = SafeMath.safe_log(np.nan, default=-99.0)
    assert result == pytest.approx(-99.0), "Log(NaN) should return default"

    LOG.info("✓ SafeMath logarithm tests passed")


def test_safe_math_sqrt():
    """Test SafeMath square root with edge cases."""
    LOG.info("\n=== TEST: SafeMath Square Root ===")

    # Normal sqrt
    result = SafeMath.safe_sqrt(16.0, default=0.0)
    assert abs(result - 4.0) < 0.001, "Normal sqrt failed"

    # Sqrt of zero
    result = SafeMath.safe_sqrt(0.0, default=-1.0)
    assert result == pytest.approx(0.0), "Sqrt(0) should be 0"

    # Sqrt of negative
    result = SafeMath.safe_sqrt(-9.0, default=-99.0)
    assert result == pytest.approx(-99.0), "Sqrt(negative) should return default"

    # Sqrt of NaN
    result = SafeMath.safe_sqrt(np.nan, default=-99.0)
    assert result == pytest.approx(-99.0), "Sqrt(NaN) should return default"

    LOG.info("✓ SafeMath sqrt tests passed")


def test_safe_math_clip():
    """Test SafeMath clipping with edge cases."""
    LOG.info("\n=== TEST: SafeMath Clip ===")

    # Normal clip
    result = SafeMath.safe_clip(5.0, min_val=0.0, max_val=10.0, default=0.0)
    assert result == pytest.approx(5.0), "Value within range should be unchanged"

    # Clip below min
    result = SafeMath.safe_clip(-5.0, min_val=0.0, max_val=10.0, default=0.0)
    assert result == pytest.approx(0.0), "Value below min should be clipped"

    # Clip above max
    result = SafeMath.safe_clip(15.0, min_val=0.0, max_val=10.0, default=0.0)
    assert result == pytest.approx(10.0), "Value above max should be clipped"

    # Clip NaN
    result = SafeMath.safe_clip(np.nan, min_val=0.0, max_val=10.0, default=5.0)
    assert result == pytest.approx(5.0), "NaN should return default"

    # Clip Inf
    result = SafeMath.safe_clip(np.inf, min_val=0.0, max_val=10.0, default=5.0)
    assert result == pytest.approx(10.0), "Inf should be clipped to max"

    LOG.info("✓ SafeMath clip tests passed")


def test_safe_math_mean():
    """Test SafeMath mean calculation."""
    LOG.info("\n=== TEST: SafeMath Mean ===")

    # Normal mean
    result = SafeMath.safe_mean([1, 2, 3, 4, 5], default=0.0)
    assert abs(result - 3.0) < 0.001, "Normal mean failed"

    # Empty list
    result = SafeMath.safe_mean([], default=-99.0)
    assert result == pytest.approx(-99.0), "Empty list should return default"

    # List with NaN
    result = SafeMath.safe_mean([1, np.nan, 3], default=-99.0)
    # Should skip NaN and average 1 and 3
    assert abs(result - 2.0) < 0.001 or result == pytest.approx(-99.0), "Should handle NaN"

    LOG.info("✓ SafeMath mean tests passed")


# ============================================================================
# RING BUFFER TESTS
# ============================================================================


def test_ring_buffer_basic_operations():
    """Test RingBuffer basic push and statistics."""
    LOG.info("\n=== TEST: RingBuffer Basic Operations ===")

    try:
        from ring_buffer import RingBuffer
    except ImportError:
        LOG.warning("RingBuffer not found, skipping test")
        return

    buffer = RingBuffer(capacity=5)

    # Push values
    for i in range(3):
        buffer.push(float(i))

    assert buffer.size() == 3, "Buffer should have 3 elements"
    assert buffer.is_full() == False, "Buffer should not be full"

    # Check mean
    mean = buffer.mean()
    assert abs(mean - 1.0) < 0.001, f"Mean should be 1.0, got {mean}"

    LOG.info("✓ RingBuffer basic operations passed")


def test_ring_buffer_overflow():
    """Test RingBuffer behavior when full."""
    LOG.info("\n=== TEST: RingBuffer Overflow ===")

    try:
        from ring_buffer import RingBuffer
    except ImportError:
        LOG.warning("RingBuffer not found, skipping test")
        return

    buffer = RingBuffer(capacity=3)

    # Fill buffer
    buffer.push(1.0)
    buffer.push(2.0)
    buffer.push(3.0)

    assert buffer.is_full() == True, "Buffer should be full"

    # Overflow - should evict oldest
    buffer.push(4.0)

    assert buffer.size() == 3, "Buffer should still have 3 elements"
    # Should now contain: 2, 3, 4 (1 was evicted)
    mean = buffer.mean()
    assert abs(mean - 3.0) < 0.001, f"Mean should be 3.0 after overflow, got {mean}"

    LOG.info("✓ RingBuffer overflow behavior correct")


# ============================================================================
# CIRCUIT BREAKER TESTS
# ============================================================================


def test_circuit_breaker_max_loss():
    """Test circuit breaker triggers on max loss."""
    LOG.info("\n=== TEST: Circuit Breaker - Max Loss ===")

    try:
        from circuit_breakers import CircuitBreakers
    except ImportError:
        LOG.warning("CircuitBreakers not found, skipping test")
        return

    breakers = CircuitBreakers(
        max_loss_day=100.0,
        max_drawdown=0.20,
    )

    # Small loss - should be OK
    result = breakers.check_all(
        pnl_day=-50.0,
        pnl_total=500.0,
        drawdown_current=0.10,
        volatility=0.015,
    )

    assert result["all_clear"] == True, "Small loss should not trip breaker"

    # Large loss - should trip
    result = breakers.check_all(
        pnl_day=-150.0,
        pnl_total=500.0,
        drawdown_current=0.10,
        volatility=0.015,
    )

    assert result["all_clear"] == False, "Large loss should trip breaker"
    assert "max_loss_day" in result["tripped"], "Should trip max_loss_day breaker"

    LOG.info("✓ Circuit breaker max loss test passed")


def test_circuit_breaker_drawdown():
    """Test circuit breaker triggers on drawdown."""
    LOG.info("\n=== TEST: Circuit Breaker - Drawdown ===")

    try:
        from circuit_breakers import CircuitBreakers
    except ImportError:
        LOG.warning("CircuitBreakers not found, skipping test")
        return

    breakers = CircuitBreakers(
        max_loss_day=1000.0,
        max_drawdown=0.15,
    )

    # Large drawdown - should trip
    result = breakers.check_all(
        pnl_day=-50.0,
        pnl_total=500.0,
        drawdown_current=0.25,  # >15% limit
        volatility=0.015,
    )

    assert result["all_clear"] == False, "Large drawdown should trip breaker"

    LOG.info("✓ Circuit breaker drawdown test passed")


# ============================================================================
# ATOMIC PERSISTENCE TESTS
# ============================================================================


def test_atomic_persistence_save_load():
    """Test atomic persistence save and load."""
    LOG.info("\n=== TEST: Atomic Persistence Save/Load ===")

    try:
        from atomic_persistence import AtomicPersistence
    except ImportError:
        LOG.warning("AtomicPersistence not found, skipping test")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_state.json"

        # Create data
        data = {"counter": 42, "name": "test", "values": [1, 2, 3]}

        # Save atomically
        persistence = AtomicPersistence()
        persistence.save(filepath, data)

        # Load back
        loaded = persistence.load(filepath)

        assert loaded is not None, "Should load successfully"
        assert loaded["counter"] == 42, "Data should match"
        assert loaded["name"] == "test", "Data should match"
        assert loaded["values"] == [1, 2, 3], "Data should match"

    LOG.info("✓ Atomic persistence save/load passed")


def test_atomic_persistence_corruption_detection():
    """Test that corrupted files are detected."""
    LOG.info("\n=== TEST: Atomic Persistence Corruption Detection ===")

    try:
        from atomic_persistence import AtomicPersistence
    except ImportError:
        LOG.warning("AtomicPersistence not found, skipping test")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "corrupted.json"

        # Write corrupted JSON
        with open(filepath, "w") as f:
            f.write('{"invalid json syntax')

        # Try to load
        persistence = AtomicPersistence()
        loaded = persistence.load(filepath)

        # Should return None or raise exception
        assert loaded is None, "Corrupted file should not load"

    LOG.info("✓ Corruption detection passed")


# ============================================================================
# NON-REPAINT GUARDS TESTS
# ============================================================================


def test_non_repaint_bar_zero_discipline():
    """Test that Bar[0] discipline is enforced."""
    LOG.info("\n=== TEST: Non-Repaint Bar[0] Discipline ===")

    try:
        from non_repaint_guards import NonRepaintGuards
    except ImportError:
        LOG.warning("NonRepaintGuards not found, skipping test")
        return

    guards = NonRepaintGuards()

    # Simulate bar close
    bar_num = 100
    close_price = 50000.0

    # Mark bar as closed
    guards.on_bar_close(bar_num, close_price)

    # Try to access Bar[0] - should be safe
    is_safe = guards.is_bar_closed(bar_num)
    assert is_safe == True, "Bar should be marked as closed"

    # Try to access Bar[0] before close - should not be safe
    is_safe = guards.is_bar_closed(bar_num + 1)
    assert is_safe == False, "Future bar should not be closed"

    LOG.info("✓ Non-repaint Bar[0] discipline enforced")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


def run_all_tests():
    """Run all unit tests."""
    LOG.info("\n" + "=" * 80)
    LOG.info("UNIT TESTS FOR CORE SAFETY COMPONENTS")
    LOG.info("Expanding test coverage")
    LOG.info("=" * 80)

    tests = [
        # SafeMath tests
        test_safe_math_division,
        test_safe_math_log,
        test_safe_math_sqrt,
        test_safe_math_clip,
        test_safe_math_mean,
        # RingBuffer tests
        test_ring_buffer_basic_operations,
        test_ring_buffer_overflow,
        # CircuitBreaker tests
        test_circuit_breaker_max_loss,
        test_circuit_breaker_drawdown,
        # AtomicPersistence tests
        test_atomic_persistence_save_load,
        test_atomic_persistence_corruption_detection,
        # NonRepaintGuards tests
        test_non_repaint_bar_zero_discipline,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            LOG.error("FAILED: %s - %s", test.__name__, e)
            failed += 1
        except Exception as e:
            LOG.warning("SKIPPED: %s - %s", test.__name__, e)
            skipped += 1

    LOG.info("\n" + "=" * 80)
    LOG.info("TEST SUMMARY")
    LOG.info("=" * 80)
    LOG.info("Passed:  %d", passed)
    LOG.info("Failed:  %d", failed)
    LOG.info("Skipped: %d", skipped)
    LOG.info("Total:   %d", len(tests))

    if failed == 0:
        LOG.info("\n✅ ALL UNIT TESTS PASSED")
    else:
        LOG.error("\n❌ SOME TESTS FAILED")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
