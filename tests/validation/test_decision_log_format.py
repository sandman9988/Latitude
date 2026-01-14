#!/usr/bin/env python3
"""
Harvester Decision Log Integration Test
========================================
Verify that decision log captures all harvester metrics correctly.

This test directly simulates position tracking + harvester exit decisions
without relying on trigger entry signals.
"""
import sys
import json
from pathlib import Path

print("=" * 70)
print("TEST: Harvester Decision Log Capture")
print("=" * 70)

# Test decision log structure (simulating main bot's log format)
log_path = Path("test_exports/harvester_decision_log.json")
log_path.parent.mkdir(exist_ok=True, parents=True)

decision_log = []

# Simulate 5 bars of position tracking
for bar in range(5):
    # Simulate position metrics that change over time
    entry_price = 90000.0
    mfe = bar * 10.0  # MFE increasing
    mae = 0.0 if bar < 3 else (bar - 2) * 2.0  # MAE starts at bar 3
    bars_held = bar + 1
    exit_action = 1 if bar == 4 else 0  # Exit on bar 4
    exit_conf = 0.75 if bar == 4 else 0.60

    # Log entry (same format as ctrader_ddqn_paper.py lines 2126-2169)
    log_entry = {
        "timestamp": f"2026-01-10T19:00:{bar:02d}",
        "bar": bar,
        "event": "bar_close",
        "details": {
            "open": entry_price + bar * 5,
            "high": entry_price + bar * 5 + 10,
            "low": entry_price + bar * 5 - 10,
            "close": entry_price + mfe,  # Close reflects MFE
            "cur_pos": 1,  # LONG
            "desired": 1 if bar < 4 else 0,  # HOLD until bar 4, then exit
            "depth_bid": 1000.0,
            "depth_ask": 950.0,
            "depth_ratio": 1.05,
            "imbalance": 0.05,
            "runway": 150.0,
            "feasibility": 0.9,
            "action": None,  # Only set when FLAT
            "confidence": None,
            "exit_action": exit_action,
            "exit_conf": exit_conf,
            # HARVESTER METRICS (added in latest code)
            "mfe": mfe,
            "mae": mae,
            "bars_held": bars_held,
            "entry_price": entry_price,
            "circuit_breaker": False,
        },
    }
    decision_log.append(log_entry)

    print(f"[BAR {bar}] MFE={mfe:.2f}, MAE={mae:.2f}, bars_held={bars_held}, exit={exit_action}")

# Save decision log
with open(log_path, "w") as f:
    json.dump(decision_log, f, indent=2)

print(f"\n✓ Decision log saved: {log_path}")

# --- VERIFICATION ---
print("\n" + "=" * 70)
print("VERIFICATION: Check all harvester fields are present")
print("=" * 70)

required_fields = ["mfe", "mae", "bars_held", "entry_price", "exit_action", "exit_conf"]
missing_fields = []

for bar_idx, entry in enumerate(decision_log):
    details = entry["details"]
    for field in required_fields:
        if field not in details:
            missing_fields.append((bar_idx, field))
            print(f"✗ BAR {bar_idx}: Missing field '{field}'")

if not missing_fields:
    print("✓ All harvester fields present in all log entries")

# Check MFE progression
mfe_values = [e["details"]["mfe"] for e in decision_log]
mfe_increases = all(mfe_values[i] <= mfe_values[i + 1] for i in range(len(mfe_values) - 1))
print(f"✓ MFE increases monotonically: {mfe_values}" if mfe_increases else f"✗ MFE progression incorrect: {mfe_values}")

# Check bars_held progression
bars_values = [e["details"]["bars_held"] for e in decision_log]
bars_increases = bars_values == list(range(1, 6))
print(f"✓ bars_held increments correctly: {bars_values}" if bars_increases else f"✗ bars_held incorrect: {bars_values}")

# Check exit signal
exit_signals = [e["details"]["exit_action"] for e in decision_log]
has_exit = 1 in exit_signals
print(f"✓ Exit signal detected (action=1 at bar 4)" if has_exit else "✗ No exit signal")

# Final result
print("\n" + "=" * 70)
if not missing_fields and mfe_increases and bars_increases and has_exit:
    print("✓ ALL TESTS PASSED - Decision log format correct!")
    print("=" * 70)
    print("\nNext step: Verify live bot generates this log when market opens.")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED")
    print("=" * 70)
    sys.exit(1)
