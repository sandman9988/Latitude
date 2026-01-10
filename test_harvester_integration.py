#!/usr/bin/env python3
"""
Integration Test: Harvester Decision Flow + Logging
===================================================
Verify complete integration:
1. Entry decision (TriggerAgent)
2. Position tracking (MFE, MAE, bars_held)
3. Exit decision (HarvesterAgent)
4. Decision log captures all metrics

This simulates the actual bot flow: FLAT → ENTRY → IN_POSITION → EXIT
"""
import sys
import json
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

from dual_policy import DualPolicy


def simulate_trading_session():
    """Simulate a complete trading session with decision logging."""
    print("=" * 70)
    print("INTEGRATION TEST: Harvester + Decision Logging")
    print("=" * 70)

    # Setup
    policy = DualPolicy(
        window=10,
        symbol="BTCUSD",
        timeframe="M15",
        broker="paper",
        enable_training=False,
    )

    # Prepare decision log (like main bot does)
    log_path = Path("test_exports/decision_log_integration.json")
    log_path.parent.mkdir(exist_ok=True, parents=True)
    decision_log = []

    # Simulate 15 bars of trading
    base_price = 90000.0
    base_time = datetime.utcnow()
    bars = deque(maxlen=64)

    entry_bar = None
    exit_bar = None

    for bar_idx in range(15):
        # Simulate price movement
        if bar_idx == 0:
            # Entry bar
            price = base_price
        elif bar_idx < 8:
            # Price rises (favorable)
            price = base_price + (bar_idx * 15.0)  # +15 per bar
        else:
            # Price consolidates/dips slightly
            price = base_price + 105.0 - ((bar_idx - 7) * 5.0)

        timestamp = base_time + timedelta(minutes=15 * bar_idx)

        # Add bar to history
        o, h, l, c = price, price + 5, price - 5, price
        bars.append([timestamp.timestamp(), o, h, l, c, 1000.0, bar_idx])

        # Determine position state
        cur_pos = policy.current_position

        # Decision variables
        action, confidence, exit_action, exit_conf = None, None, None, None
        runway, feas = None, None

        if cur_pos == 0:
            # FLAT: Check for entry
            action, confidence, runway = policy.decide_entry(
                bars,
                imbalance=0.05,
                vpin_z=1.2,
                depth_ratio=1.1,
            )
            feas = 1.0  # Simplified for test

            # Simulate entry if signal
            if action == 1 and entry_bar is None:  # LONG entry
                policy.on_entry(direction=1, entry_price=c, entry_time=timestamp)
                entry_bar = bar_idx
                print(f"\n[BAR {bar_idx:2d}] ENTRY: LONG @ {c:.2f}, conf={confidence:.2f}, runway={runway:.2f}")
        else:
            # IN POSITION: Check for exit
            exit_action, exit_conf = policy.decide_exit(
                bars,
                current_price=c,
                imbalance=0.05,
                vpin_z=1.0,
                depth_ratio=1.05,
            )

            # Get position metrics
            pos_metrics = policy.get_position_metrics()

            print(
                f"[BAR {bar_idx:2d}] IN POS: C={c:.2f} | "
                f"MFE={pos_metrics['mfe']:.2f}, MAE={pos_metrics['mae']:.2f}, "
                f"bars={pos_metrics['bars_held']}, exit={exit_action}, conf={exit_conf:.2f}"
            )

            # Simulate exit if signal
            if exit_action == 1 and exit_bar is None:
                capture_ratio = pos_metrics["mfe"] / max(pos_metrics["mfe"], 1.0)  # Simplification
                policy.on_exit(exit_price=c, capture_ratio=capture_ratio, was_wtl=False)
                exit_bar = bar_idx
                print(f"\n[BAR {bar_idx:2d}] EXIT: CLOSE @ {c:.2f}, conf={exit_conf:.2f}")

        # --- DECISION LOG ENTRY (like main bot) ---
        pos_metrics = policy.get_position_metrics() if hasattr(policy, "get_position_metrics") else {}

        log_entry = {
            "timestamp": timestamp.isoformat(),
            "bar": bar_idx,
            "event": "bar_close",
            "details": {
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "cur_pos": cur_pos,
                "action": action,
                "confidence": confidence,
                "runway": runway,
                "feasibility": feas,
                "exit_action": exit_action,
                "exit_conf": exit_conf,
                # Harvester metrics
                "mfe": pos_metrics.get("mfe", 0.0),
                "mae": pos_metrics.get("mae", 0.0),
                "bars_held": pos_metrics.get("bars_held", 0),
                "entry_price": pos_metrics.get("entry_price", 0.0),
            },
        }
        decision_log.append(log_entry)

    # Save decision log
    with open(log_path, "w") as f:
        json.dump(decision_log, f, indent=2)
    print(f"\n✓ Decision log saved: {log_path}")

    # --- VERIFICATION ---
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Check entry was logged
    entry_logged = any(
        e["details"].get("action") == 1 and e["details"].get("confidence") is not None for e in decision_log
    )
    print(f"Entry logged: {'✓' if entry_logged else '✗'}")

    # Check in-position metrics were tracked
    in_pos_logged = any(e["details"].get("mfe") is not None and e["details"].get("mfe") > 0 for e in decision_log)
    print(f"Position metrics tracked: {'✓' if in_pos_logged else '✗'}")

    # Check MFE increased over time
    mfe_values = [e["details"].get("mfe", 0) for e in decision_log if e["details"].get("cur_pos") != 0]
    mfe_increased = len(mfe_values) > 1 and max(mfe_values) > min(mfe_values)
    print(f"MFE tracking correct: {'✓' if mfe_increased else '✗'}")

    # Check bars_held incremented
    bars_held_values = [e["details"].get("bars_held", 0) for e in decision_log if e["details"].get("cur_pos") != 0]
    bars_incremented = len(bars_held_values) > 1 and max(bars_held_values) > min(bars_held_values)
    print(f"Bars held incremented: {'✓' if bars_incremented else '✗'}")

    # Check exit was logged
    exit_logged = any(e["details"].get("exit_action") == 1 for e in decision_log)
    print(f"Exit logged: {'✓' if exit_logged else '✗'}")

    # Summary
    all_pass = entry_logged and in_pos_logged and mfe_increased and bars_incremented and exit_logged
    print("\n" + "=" * 70)
    if all_pass:
        print("✓ ALL TESTS PASSED - Harvester integration complete!")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME TESTS FAILED - Review decision log")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(simulate_trading_session())
