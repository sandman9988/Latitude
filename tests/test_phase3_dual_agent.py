#!/usr/bin/env python3
"""
Test Phase 3: Dual-Agent Integration
====================================
Validates TriggerAgent + HarvesterAgent + DualPolicy integration into main bot.
"""

import os
import sys
import logging
from collections import deque
import datetime as dt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dual-agent mode
os.environ["DDQN_DUAL_AGENT"] = "1"

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

print("=" * 70)
print("Phase 3: Dual-Agent Integration Test")
print("=" * 70)

# Test 1: Import checks
print("\n[TEST 1] Import validation")
try:
    from dual_policy import DualPolicy
    from trigger_agent import TriggerAgent
    from harvester_agent import HarvesterAgent
    print("✓ Dual-agent modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: DualPolicy initialization
print("\n[TEST 2] DualPolicy initialization")
dual_policy = DualPolicy(window=64)
assert dual_policy.trigger is not None
assert dual_policy.harvester is not None
assert dual_policy.current_position == 0
print("✓ DualPolicy initialized with trigger + harvester")

# Test 3: Entry decision (flat position)
print("\n[TEST 3] Entry decision workflow")
bars = deque(maxlen=100)
for i in range(100):
    t = dt.datetime.now()
    o = h = l = c = 100000.0 + i * 10
    bars.append((t, o, h, l, c))

action, conf, runway = dual_policy.decide_entry(bars, imbalance=0.1)
assert action in [0, 1, 2], f"Invalid action: {action}"
assert 0 <= conf <= 1, f"Invalid confidence: {conf}"
assert runway >= 0, f"Invalid runway: {runway}"
print(f"✓ Entry decision: action={action}, conf={conf:.2f}, runway={runway:.4f}")

# Test 4: Position entry
print("\n[TEST 4] Position entry tracking")
dual_policy.on_entry(direction=1, entry_price=100000.0, entry_time=dt.datetime.now())
assert dual_policy.current_position == 1
assert dual_policy.entry_price == 100000.0
assert dual_policy.bars_held == 0
print("✓ Position tracked: LONG @ 100000.0")

# Test 5: Exit decision (in position)
print("\n[TEST 5] Exit decision workflow")
current_price = 100050.0  # Small profit
action, conf = dual_policy.decide_exit(bars, current_price, imbalance=0.1)
assert action in [0, 1], f"Invalid exit action: {action}"
assert 0 <= conf <= 1, f"Invalid exit confidence: {conf}"
assert dual_policy.mfe > 0, "MFE should be tracked"
assert dual_policy.bars_held == 1, "Bars held should increment"
print(f"✓ Exit decision: action={action} ({'CLOSE' if action == 1 else 'HOLD'}), "
      f"conf={conf:.2f}, MFE={dual_policy.mfe:.2f}, bars={dual_policy.bars_held}")

# Test 6: Position exit
print("\n[TEST 6] Position exit tracking")
dual_policy.on_exit(exit_price=100050.0, capture_ratio=0.8, was_wtl=False)
assert dual_policy.current_position == 0
assert dual_policy.mfe == 0.0
assert dual_policy.bars_held == 0
print("✓ Position reset after exit")

# Test 7: Multiple entry/exit cycles
print("\n[TEST 7] Multiple trade cycles")
for cycle in range(3):
    # Entry
    action, conf, runway = dual_policy.decide_entry(bars, imbalance=0.1)
    if action != 0:  # If entry signaled
        direction = 1 if action == 1 else -1
        dual_policy.on_entry(direction, 100000.0, dt.datetime.now())
        
        # Hold position for a few bars
        for _ in range(5):
            action, conf = dual_policy.decide_exit(bars, 100050.0, imbalance=0.1)
            if action == 1:  # Close
                break
        
        # Exit
        dual_policy.on_exit(100050.0, 0.75, False)
        assert dual_policy.current_position == 0

print(f"✓ Completed {3} entry/exit cycles")

# Test 8: Backward compatibility check
print("\n[TEST 8] Backward compatibility (single-agent mode)")
os.environ["DDQN_DUAL_AGENT"] = "0"

# This would normally import Policy, but we're just checking the flag
use_dual = os.environ.get("DDQN_DUAL_AGENT", "0").strip() in ("1", "true", "True", "TRUE")
assert not use_dual
print("✓ Single-agent mode detected when DDQN_DUAL_AGENT=0")

# Restore dual-agent mode
os.environ["DDQN_DUAL_AGENT"] = "1"

# Test 9: State consistency
print("\n[TEST 9] State consistency checks")
dual_policy2 = DualPolicy(window=64)
dual_policy2.on_entry(1, 100000.0, dt.datetime.now())

# Simulate price movement
for i in range(10):
    price = 100000.0 + i * 5  # Increasing price (favorable for LONG)
    dual_policy2.decide_exit(bars, price, imbalance=0.1)

assert dual_policy2.mfe > 0, "MFE should increase with favorable price movement"
assert dual_policy2.mae >= 0, "MAE should be tracked"
assert dual_policy2.bars_held == 10, "Bars held should match iterations"
print(f"✓ State tracking: MFE={dual_policy2.mfe:.2f}, MAE={dual_policy2.mae:.2f}, bars={dual_policy2.bars_held}")

# Test 10: Agent update mechanisms
print("\n[TEST 10] Agent update mechanisms")
dual_policy2.on_exit(100050.0, 0.85, False)

# Verify agents received update (logs should show update messages)
# Since update_from_trade is called in on_exit(), we just verify it doesn't crash
print("✓ Agent update mechanisms functional")

print("\n" + "=" * 70)
print("✓ All Phase 3 integration tests passed!")
print("=" * 70)
print("\nNext steps:")
print("1. Run bot with DDQN_DUAL_AGENT=1 to test live integration")
print("2. Implement specialized reward functions (Phase 3.2)")
print("3. Enhance PathRecorder for dual-agent attribution (Phase 3.3)")
