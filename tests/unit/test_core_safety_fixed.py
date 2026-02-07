import pytest

"""
Core Safety Component Unit Tests (Fixed)

Tests the actual API of core safety components:
- SafeMath (safe_div, safe_log, safe_sqrt, clamp)
- RingBuffer (append, mean, std)
- AtomicPersistence (save_state, load_state)
"""

import logging
import sys
from pathlib import Path
import tempfile
import os

import numpy as np

from src.utils.safe_math import SafeMath
from src.utils.ring_buffer import RingBuffer, RollingStats
from src.persistence.atomic_persistence import AtomicPersistence

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def test_safe_math_div():
    """Test SafeMath.safe_div with edge cases."""
    LOG.info("\n=== TEST: SafeMath safe_div ===")

    # Normal division
    result = SafeMath.safe_div(10.0, 2.0)
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

    LOG.info("✓ SafeMath safe_div tests passed")


def test_safe_math_log():
    """Test SafeMath.safe_log with edge cases."""
    LOG.info("\n=== TEST: SafeMath safe_log ===")

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

    LOG.info("✓ SafeMath safe_log tests passed")


def test_safe_math_sqrt():
    """Test SafeMath.safe_sqrt with edge cases."""
    LOG.info("\n=== TEST: SafeMath safe_sqrt ===")

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

    LOG.info("✓ SafeMath safe_sqrt tests passed")


def test_safe_math_clamp():
    """Test SafeMath.clamp."""
    LOG.info("\n=== TEST: SafeMath clamp ===")

    # Normal clamp
    result = SafeMath.clamp(5.0, min_val=0.0, max_val=10.0)
    assert result == pytest.approx(5.0), "Value within range should be unchanged"

    # Clamp below min
    result = SafeMath.clamp(-5.0, min_val=0.0, max_val=10.0)
    assert result == pytest.approx(0.0), "Value below min should be clamped"

    # Clamp above max
    result = SafeMath.clamp(15.0, min_val=0.0, max_val=10.0)
    assert result == pytest.approx(10.0), "Value above max should be clamped"

    LOG.info("✓ SafeMath clamp tests passed")


def test_safe_math_is_valid():
    """Test SafeMath.is_valid."""
    LOG.info("\n=== TEST: SafeMath is_valid ===")

    # Valid numbers
    assert SafeMath.is_valid(5.0), "Normal float should be valid"
    assert SafeMath.is_valid(0.0), "Zero should be valid"
    assert SafeMath.is_valid(-100.0), "Negative should be valid"

    # Invalid numbers
    assert not SafeMath.is_valid(np.nan), "NaN should be invalid"
    assert not SafeMath.is_valid(np.inf), "Inf should be invalid"
    assert not SafeMath.is_valid(-np.inf), "-Inf should be invalid"

    # Valid arrays
    arr = np.array([1.0, 2.0, 3.0])
    assert SafeMath.is_valid(arr), "Valid array should be valid"

    # Invalid arrays
    arr = np.array([1.0, np.nan, 3.0])
    assert not SafeMath.is_valid(arr), "Array with NaN should be invalid"

    arr = np.array([1.0, np.inf, 3.0])
    assert not SafeMath.is_valid(arr), "Array with Inf should be invalid"

    LOG.info("✓ SafeMath is_valid tests passed")


def test_ring_buffer_basic():
    """Test RingBuffer basic operations."""
    LOG.info("\n=== TEST: RingBuffer basic ===")

    rb = RingBuffer(capacity=5)

    # Add values
    for i in range(5):
        rb.append(float(i))

    assert len(rb) == 5, "Buffer should have 5 elements"

    # Check we can iterate
    values = list(rb)
    assert values == [0.0, 1.0, 2.0, 3.0, 4.0], f"Values mismatch: {values}"

    LOG.info("✓ RingBuffer basic operations passed")


def test_ring_buffer_overflow():
    """Test RingBuffer overflow behavior."""
    LOG.info("\n=== TEST: RingBuffer overflow ===")

    rb = RingBuffer(capacity=3)

    # Fill buffer
    for i in range(3):
        rb.append(float(i))

    assert len(rb) == 3, "Buffer should be full"

    # Add one more (should overwrite oldest)
    rb.append(100.0)

    assert len(rb) == 3, "Buffer should still be size 3"
    # Should contain [1.0, 2.0, 100.0]
    values = list(rb)
    assert values == [1.0, 2.0, 100.0], f"Values mismatch: {values}"

    LOG.info("✓ RingBuffer overflow tests passed")


def test_rolling_stats():
    """Test RollingStats (combined mean/std/min/max)."""
    LOG.info("\n=== TEST: RollingStats ===")

    rs = RollingStats(period=5)

    values = [5.0, 2.0, 8.0, 1.0, 9.0, 3.0]
    for val in values:
        rs.update(val)

    # After 6 values with period=5, should have last 5: [2, 8, 1, 9, 3]
    assert abs(rs.min - 1.0) < 0.001, f"Min should be 1.0, got {rs.min}"
    assert abs(rs.max - 9.0) < 0.001, f"Max should be 9.0, got {rs.max}"

    expected_mean = (2.0 + 8.0 + 1.0 + 9.0 + 3.0) / 5.0
    assert abs(rs.mean - expected_mean) < 0.001, f"Mean mismatch: {rs.mean} vs {expected_mean}"

    LOG.info("✓ RollingStats tests passed")


def test_atomic_persistence_save_load():
    """Test AtomicPersistence save/load."""
    LOG.info("\n=== TEST: AtomicPersistence save/load ===")

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        ap = AtomicPersistence(tmpdir)

        # Save state
        state = {"counter": 42, "values": [1.0, 2.0, 3.0], "name": "test_state"}

        success = ap.save_json(state, "test_file.json", create_backup=False)
        assert success, "Save should succeed"

        # Load state
        loaded = ap.load_json("test_file.json", verify_crc=True)
        assert loaded is not None, "Load should succeed"
        assert loaded["counter"] == 42, "Counter mismatch"
        assert loaded["name"] == "test_state", "Name mismatch"

    LOG.info("✓ AtomicPersistence save/load tests passed")


def test_atomic_persistence_corruption_detection():
    """Test AtomicPersistence corruption detection."""
    LOG.info("\n=== TEST: AtomicPersistence corruption ===")

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        ap = AtomicPersistence(tmpdir)

        # Save state
        state = {"value": 123}
        ap.save_json(state, "test_file.json", create_backup=False)

        # Manually corrupt the file
        file_path = os.path.join(tmpdir, "test_file.json")
        with open(file_path, "a") as f:
            f.write("corrupted_data")

        # Try to load - should detect corruption
        loaded = ap.load_json("test_file.json", verify_crc=True)
        assert loaded is None, "Should detect corruption and return None"

    LOG.info("✓ AtomicPersistence corruption detection passed")


def test_safe_math_tolerance_comparisons():
    """Test SafeMath tolerance-based comparisons."""
    LOG.info("\n=== TEST: SafeMath tolerance comparisons ===")

    # is_equal
    assert SafeMath.is_equal(1.0, 1.0), "Exact equals should work"
    assert SafeMath.is_equal(1.0, 1.0001, eps=0.001), "Close values should be equal"
    assert not SafeMath.is_equal(1.0, 1.1, eps=0.01), "Distant values should not be equal"

    # is_greater
    assert SafeMath.is_greater(2.0, 1.0), "2 > 1"
    assert not SafeMath.is_greater(1.0, 1.0), "1 not > 1"
    assert not SafeMath.is_greater(1.0001, 1.0, eps=0.01), "Close values not greater"

    # is_less
    assert SafeMath.is_less(1.0, 2.0), "1 < 2"
    assert not SafeMath.is_less(1.0, 1.0), "1 not < 1"
    assert not SafeMath.is_less(1.0, 1.0001, eps=0.01), "Close values not less"

    LOG.info("✓ SafeMath tolerance comparisons passed")


def run_all_tests():
    """Run all tests and report summary."""
    LOG.info("\n" + "=" * 80)
    LOG.info("CORE SAFETY UNIT TESTS (Fixed)")
    LOG.info("Testing actual API implementations")
    LOG.info("=" * 80)

    tests = [
        test_safe_math_div,
        test_safe_math_log,
        test_safe_math_sqrt,
        test_safe_math_clamp,
        test_safe_math_is_valid,
        test_safe_math_tolerance_comparisons,
        test_ring_buffer_basic,
        test_ring_buffer_overflow,
        test_rolling_stats,
        test_atomic_persistence_save_load,
        test_atomic_persistence_corruption_detection,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            LOG.error(f"FAILED: {test.__name__} - {e}")
            failed += 1
        except Exception as e:
            LOG.error(f"ERROR: {test.__name__} - {e}")
            failed += 1

    LOG.info("\n" + "=" * 80)
    LOG.info("TEST SUMMARY")
    LOG.info("=" * 80)
    LOG.info(f"Passed: {passed}")
    LOG.info(f"Failed: {failed}")
    LOG.info(f"Total:  {len(tests)}")

    if failed == 0:
        LOG.info("\n✅ ALL CORE SAFETY TESTS PASSED")
        return 0
    else:
        LOG.error(f"\n❌ {failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
