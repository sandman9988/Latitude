#!/usr/bin/env python3
"""
Test Harvester Exit Triggers
============================
Verify that harvester generates exit signals when thresholds are hit:
1. Profit target (30% MFE)
2. Stop loss (20% MAE)
3. Soft time stop (50 bars + profit)
4. Hard time stop (80 bars)
"""
import sys
import numpy as np

# Import harvester
from src.agents.harvester_agent import HarvesterAgent


def test_profit_target():
    """Test that harvester exits when profit target is hit."""
    print("\n" + "=" * 70)
    print("TEST 1: Profit Target Exit (30% MFE)")
    print("=" * 70)

    harvester = HarvesterAgent(window=10, n_features=10)

    # Create market state (dummy)
    market_state = np.zeros((10, 7), dtype=np.float32)

    entry_price = 90000.0

    # Simulate MFE building up to 30%
    exit_triggered = False
    # bars_held=15 — past the 12-bar min-hold at M5 (= 1 h) so threshold exits can fire
    for pct in [5, 10, 15, 20, 25, 28, 30, 31]:
        mfe = entry_price * (pct / 100.0)
        mae = 0.0  # No adverse movement
        bars_held = 15

        action, conf = harvester.decide(market_state, mfe, mae, bars_held, entry_price, direction=1)

        if action == 1:
            print(f"✓ EXIT TRIGGERED at MFE={pct:.1f}%, action={action}, conf={conf:.2f}")
            exit_triggered = True
            break
        else:
            print(f"  HOLD at MFE={pct:.1f}%, action={action}")

    assert exit_triggered, "No exit triggered at profit target"


def test_stop_loss():
    """Test that harvester exits when stop loss is hit."""
    print("\n" + "=" * 70)
    print("TEST 2: Stop Loss Exit (20% MAE)")
    print("=" * 70)

    harvester = HarvesterAgent(window=10, n_features=10)

    market_state = np.zeros((10, 7), dtype=np.float32)
    entry_price = 90000.0

    # Simulate MAE building up to 20%
    exit_triggered = False
    # bars_held=15 — past the 12-bar min-hold at M5 so stop-loss exits can fire
    for pct in [5, 10, 15, 18, 20, 22]:
        mae = entry_price * (pct / 100.0)
        mfe = entry_price * 0.05  # Small profit before reversal
        bars_held = 15

        action, conf = harvester.decide(market_state, mfe, mae, bars_held, entry_price, direction=1)

        if action == 1:
            print(f"✓ EXIT TRIGGERED at MAE={pct:.1f}%, action={action}, conf={conf:.2f}")
            exit_triggered = True
            break
        else:
            print(f"  HOLD at MAE={pct:.1f}%, action={action}")

    assert exit_triggered, "No exit triggered at stop loss"


def test_soft_time_stop():
    """Test that harvester exits on soft time stop (50 bars + profit)."""
    print("\n" + "=" * 70)
    print("TEST 3: Soft Time Stop (50 bars + 0.05% profit)")
    print("=" * 70)

    harvester = HarvesterAgent(window=10, n_features=10)

    market_state = np.zeros((10, 7), dtype=np.float32)
    entry_price = 90000.0
    mfe = entry_price * 0.06  # 6% profit (above 0.05% threshold)
    mae = 0.0

    # Test various bar counts
    exit_triggered = False
    for bars in [40, 45, 50, 51, 52]:
        action, conf = harvester.decide(market_state, mfe, mae, bars, entry_price, direction=1)

        if action == 1:
            print(f"✓ EXIT TRIGGERED at bars_held={bars}, MFE=6.0%, action={action}, conf={conf:.2f}")
            exit_triggered = True
            break
        else:
            print(f"  HOLD at bars_held={bars}")

    assert exit_triggered, "No exit triggered at soft time stop"


def test_hard_time_stop():
    """Test that harvester exits on hard time stop (80 bars regardless)."""
    print("\n" + "=" * 70)
    print("TEST 4: Hard Time Stop (80 bars)")
    print("=" * 70)

    harvester = HarvesterAgent(window=10, n_features=10)

    market_state = np.zeros((10, 7), dtype=np.float32)
    entry_price = 90000.0
    mfe = entry_price * 0.02  # Small profit
    mae = entry_price * 0.01  # Small loss

    # Test various bar counts
    exit_triggered = False
    for bars in [70, 75, 79, 80, 81]:
        action, conf = harvester.decide(market_state, mfe, mae, bars, entry_price, direction=1)

        if action == 1:
            print(f"✓ EXIT TRIGGERED at bars_held={bars}, action={action}, conf={conf:.2f}")
            exit_triggered = True
            break
        else:
            print(f"  HOLD at bars_held={bars}")

    assert exit_triggered, "No exit triggered at hard time stop"


if __name__ == "__main__":
    results = []

    results.append(test_profit_target())
    results.append(test_stop_loss())
    results.append(test_soft_time_stop())
    results.append(test_hard_time_stop())

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Profit Target: {'✓ PASS' if results[0] else '✗ FAIL'}")
    print(f"Stop Loss:     {'✓ PASS' if results[1] else '✗ FAIL'}")
    print(f"Soft Time:     {'✓ PASS' if results[2] else '✗ FAIL'}")
    print(f"Hard Time:     {'✓ PASS' if results[3] else '✗ FAIL'}")
    print("=" * 70)

    if all(results):
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
