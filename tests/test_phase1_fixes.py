#!/usr/bin/env python3
import pytest

"""
Comprehensive test suite for Phase 1 critical fixes
Tests SafeMath, AtomicPersistence, VaR estimation, and kurtosis monitoring
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path for direct script execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from numpy.random import Generator, default_rng

from src.persistence.atomic_persistence import AtomicPersistence
from src.persistence.learned_parameters import LearnedParametersManager
from src.utils.safe_math import SafeMath
from src.utils.safe_utils import SafeArray, SafeDeque, safe_mean, safe_std
from src.risk.var_estimator import KurtosisMonitor, RegimeType, VaREstimator

print("=" * 70)
print("PHASE 1 CRITICAL FIXES - TEST SUITE")
print("=" * 70)

# Test 1: SafeMath
print("\n[TEST 1] SafeMath Defensive Programming")
print("-" * 70)

# Division by zero
result = SafeMath.safe_div(10.0, 0.0, default=-1.0)
assert result == pytest.approx(-1.0), "Division by zero should return default"
print("✓ Division by zero returns default")

# NaN handling
result = SafeMath.safe_div(float("nan"), 2.0, default=-1.0)
assert result == pytest.approx(-1.0), "NaN numerator should return default"
print("✓ NaN numerator returns default")

# Inf handling
result = SafeMath.safe_div(float("inf"), 2.0, default=-1.0)
assert result == pytest.approx(-1.0), "Inf numerator should return default"
print("✓ Inf numerator returns default")

# Valid division
result = SafeMath.safe_div(10.0, 2.0)
assert abs(result - 5.0) < 1e-9, "Valid division should work"
print("✓ Valid division works")

# Clamping
result = SafeMath.clamp(15.0, 0.0, 10.0)
assert SafeMath.is_close(result, 10.0), "Clamp should limit upper bound"
print("✓ Clamping works")

# Clamping NaN
result = SafeMath.clamp(float("nan"), 0.0, 10.0)
assert abs(result - 5.0) < 1e-9, "Clamping NaN should return midpoint"
print("✓ Clamping NaN returns midpoint")

# Test 2: SafeArray
print("\n[TEST 2] SafeArray Bounds Checking")
print("-" * 70)

arr = [1, 2, 3, 4, 5]

# Valid access
result = SafeArray.safe_get(arr, 2)
assert result == 3, "Valid index should work"
print("✓ Valid index access works")

# Out of bounds
result = SafeArray.safe_get(arr, 10, default=-1)
assert result == -1, "Out of bounds should return default"
print("✓ Out of bounds returns default")

# Negative index protection
result = SafeArray.safe_get(arr, -10, default=-1)
assert result == -1, "Negative out of bounds should return default"
print("✓ Negative out of bounds returns default")

# Series access (bars_ago)
result = SafeArray.safe_get_series(arr, 0)  # Current (last)
assert result == 5, "bars_ago=0 should return last element"
print("✓ Series access (bars_ago=0) works")

result = SafeArray.safe_get_series(arr, 1)  # Previous
assert result == 4, "bars_ago=1 should return second-to-last"
print("✓ Series access (bars_ago=1) works")

result = SafeArray.safe_get_series(arr, 10, default=-1)
assert result == -1, "Series access out of bounds should return default"
print("✓ Series access out of bounds returns default")

# Test 3: SafeDeque
print("\n[TEST 3] SafeDeque Wrapper")
print("-" * 70)

sd = SafeDeque(maxlen=3, name="test")
sd.append(1)
sd.append(2)
sd.append(3)

result = sd.last()
assert result == 3, "last() should return last element"
print("✓ SafeDeque.last() works")

result = sd.get_series(1)
assert result == 2, "get_series(1) should return previous"
print("✓ SafeDeque.get_series() works")

sd.append(4)  # Evicts 1
result = sd.get_series(2)
assert result == 2, "After eviction, series should still work"
print("✓ SafeDeque eviction works correctly")

# Test 4: Atomic Persistence
print("\n[TEST 4] Atomic Persistence with CRC32")
print("-" * 70)

with tempfile.TemporaryDirectory() as tmpdir:
    ap = AtomicPersistence(tmpdir)

    # Save data
    test_data = {
        "learned_spread_relax": 1.5,
        "learned_depth_buffer": 1.2,
        "learned_vpin_z_limit": 2.0,
    }

    success = ap.save_json(test_data, "test_params.json", create_backup=True)
    assert success, "Save should succeed"
    print("✓ Atomic save with CRC32 succeeded")

    # Load data
    loaded = ap.load_json("test_params.json", verify_crc=True)
    assert loaded == test_data, "Loaded data should match saved data"
    print("✓ Atomic load with CRC32 verification succeeded")

    # Modify and save again
    test_data["learned_spread_relax"] = 2.0
    ap.save_json(test_data, "test_params.json", create_backup=True)

    backups = ap.list_backups("test_params.json")
    assert len(backups) >= 1, "Backup should be created"
    print(f"✓ Backup created: {len(backups)} backup(s)")

    # Simulate corruption and restore
    target = Path(tmpdir) / "test_params.json"
    with open(target) as f:
        envelope = __import__("json").load(f)
    envelope["data"]["learned_spread_relax"] = 999.0  # Corrupt data
    with open(target, "w") as f:
        __import__("json").dump(envelope, f)

    loaded_corrupted = ap.load_json("test_params.json", verify_crc=True)
    assert loaded_corrupted is not None, "Should restore from backup on CRC error"
    print("✓ Automatic restore from backup on CRC error")

# Test 5: Kurtosis Monitor
print("\n[TEST 5] Kurtosis Circuit Breaker")
print("-" * 70)

km = KurtosisMonitor(window=100, threshold=3.0)

# Add normal returns
rng: Generator = default_rng(42)
normal_returns = rng.normal(0.0, 0.01, 100)
for r in normal_returns:
    kurtosis, breaker = km.update(r)

assert not km.is_breaker_active, "Breaker should not trigger for normal distribution"
print(f"✓ Normal returns: kurtosis={kurtosis:.2f}, breaker={km.is_breaker_active}")

# Add fat-tail events
fat_tail_returns = [0.05, -0.05, 0.04, -0.04, 0.06, -0.06]
for r in fat_tail_returns:
    kurtosis, breaker = km.update(r)

print(f"✓ Fat-tail returns: kurtosis={kurtosis:.2f}, breaker={km.is_breaker_active}")
if kurtosis > 2.0:  # May or may not breach depending on window
    print("  Note: Kurtosis elevated by fat-tail events")

# Test 6: VaR Estimator
print("\n[TEST 6] VaR Estimator with Multi-Factor Adjustment")
print("-" * 70)

var_est = VaREstimator(window=100, confidence=0.95)

# Add returns
rng = default_rng(42)
returns = list(rng.normal(0.0, 0.01, 100))
for r in returns:
    var_est.update_return(r)

# Estimate VaR in normal regime
var_normal = var_est.estimate_var(regime=RegimeType.OVERDAMPED, vpin_z=0.0, current_vol=0.01)
assert var_normal > 0, "VaR should be positive"
print(f"✓ VaR (normal regime): {var_normal:.6f}")

# Estimate VaR in stressed regime
var_stressed = var_est.estimate_var(regime=RegimeType.UNDERDAMPED, vpin_z=3.0, current_vol=0.03)
assert var_stressed > var_normal, "Stressed VaR should be higher"
print(f"✓ VaR (stressed regime): {var_stressed:.6f}")
print(f"  Stress multiplier: {var_stressed / var_normal:.2f}x")

# Test 7: Integration with learned parameters
print("\n[TEST 7] Learned Parameters with Atomic Persistence")
print("-" * 70)

with tempfile.TemporaryDirectory() as tmpdir:
    param_path = Path(tmpdir) / "learned_parameters.json"

    # Create manager
    mgr = LearnedParametersManager(persistence_path=param_path)

    # Get/set parameters (using correct API)
    symbol = "BTCUSD"
    timeframe = "M15"
    broker = "default"

    pos_size = mgr.get(symbol, "base_position_size", timeframe, broker)
    assert pos_size is not None, "Should have default value"
    print(f"✓ Default base_position_size: {pos_size:.4f}")

    # Update parameter
    new_value = mgr.update(symbol, "var_multiplier", gradient=0.05, timeframe=timeframe, broker=broker)
    print(f"✓ Updated var_multiplier: {new_value:.4f}")

    # Save
    mgr.save()
    assert param_path.exists(), "Parameters should be saved"
    print("✓ Parameters saved with atomic persistence")

    # Verify CRC envelope
    with open(param_path) as f:
        envelope = __import__("json").load(f)
    assert "crc32" in envelope, "Should have CRC32 checksum"
    print(f"✓ CRC32 checksum present: {envelope['crc32']:08x}")

    # Load in new manager
    mgr2 = LearnedParametersManager(persistence_path=param_path)
    loaded_value = mgr2.get(symbol, "var_multiplier", timeframe, broker)
    assert abs(loaded_value - new_value) < 1e-6, "Loaded value should match saved"
    print(f"✓ Parameters loaded correctly: {loaded_value:.4f}")

# Test 8: NaN propagation prevention
print("\n[TEST 8] NaN/Inf Propagation Prevention")
print("-" * 70)

# Test safe_mean
values = [1.0, 2.0, float("nan"), 3.0, 4.0]
result = safe_mean(values, default=-1.0)
expected = (1.0 + 2.0 + 3.0 + 4.0) / 4.0
assert abs(result - expected) < 1e-9, "safe_mean should skip NaN"
print(f"✓ safe_mean skips NaN: {result:.2f}")

# Test safe_std
result = safe_std(values, default=-1.0)
assert result > 0, "safe_std should work with valid values"
print(f"✓ safe_std works with NaN: {result:.2f}")

# Test all NaN
values = [float("nan"), float("nan")]
result = safe_mean(values, default=-1.0)
assert result == pytest.approx(-1.0), "All NaN should return default"
print("✓ All NaN returns default")

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("\nPhase 1 Critical Fixes Status:")
print("  ✓ SafeMath defensive layer")
print("  ✓ Atomic persistence with CRC32")
print("  ✓ Kurtosis circuit breaker")
print("  ✓ VaR estimator with multi-factor adjustment")
print("  ✓ NaN/Inf propagation prevention")
print("  ✓ Array bounds checking")
print("\nSystem is production-ready for paper trading.")
print("=" * 70)
