#!/usr/bin/env python3
"""
Test Harvester (Exit) Decision Flow
====================================
Tests the complete flow from entry → in-position → exit decision → logging
"""

import sys
from collections import deque
from datetime import UTC, datetime


# Mock the necessary imports
class MockHarvesterAgent:
    def decide(self, market_state, mfe, mae, bars_held, entry_price, direction):
        """Mock harvester decision"""
        # Simple rule: close if MFE > 0.05% of entry or bars_held > 8
        if (entry_price > 0 and mfe / entry_price > 0.0005) or bars_held > 8:
            return 1, 0.8  # CLOSE with 80% confidence
        return 0, 0.6  # HOLD with 60% confidence


class MockDualPolicy:
    def __init__(self) -> None:
        self.harvester = MockHarvesterAgent()
        self.current_position = 0
        self.entry_price = 0
        self.mfe = 0
        self.mae = 0
        self.bars_held = 0

    def decide_entry(self, bars, **kwargs):
        """Mock entry decision"""
        return 1, 0.7, 0.0015  # LONG, 70% confidence, 0.15% runway

    def decide_exit(self, bars, current_price, **kwargs):
        """Mock exit decision using harvester"""
        # Update MFE/MAE
        if self.current_position == 1:  # LONG
            profit = current_price - self.entry_price
            self.mfe = max(self.mfe, profit)
            self.mae = max(self.mae, -profit)

        self.bars_held += 1

        # Call harvester
        import numpy as np

        market_state = np.random.default_rng(42).standard_normal((64, 7))  # Mock state

        action, confidence = self.harvester.decide(
            market_state=market_state,
            mfe=self.mfe,
            mae=self.mae,
            bars_held=self.bars_held,
            entry_price=self.entry_price,
            direction=self.current_position,
        )

        return action, confidence


def test_entry_to_exit_flow():
    """Test complete flow: FLAT → ENTRY → IN_POSITION → EXIT"""

    policy = MockDualPolicy()
    bars = deque(maxlen=100)
    decision_log = []

    # Simulate 10 bars of trading
    base_price = 90500.0

    for i in range(10):
        # Create mock bar
        t = datetime.now(UTC)
        o = base_price + i * 10
        h = o + 20
        lo = o - 15
        c = o + 5

        bars.append((t, o, h, lo, c))

        # Decision logic (simulating on_bar_close)
        action = None
        confidence = None
        exit_action = None
        exit_conf = None
        desired = None

        if policy.current_position == 0:
            # FLAT: Check for entry
            action, confidence, _runway = policy.decide_entry(bars)
            desired = 1 if action == 1 else 0

            # Simulate entry if signaled
            if action == 1:
                policy.current_position = 1
                policy.entry_price = c
                policy.mfe = 0
                policy.mae = 0
                policy.bars_held = 0
        else:
            # IN POSITION: Check for exit
            exit_action, exit_conf = policy.decide_exit(bars, current_price=c)
            desired = 0 if exit_action == 1 else policy.current_position


            # Simulate exit if signaled
            if exit_action == 1:
                c - policy.entry_price
                policy.current_position = 0

        # Log decision
        log_entry = {
            "timestamp": t.isoformat(),
            "bar": i,
            "event": "bar_close",
            "details": {
                "close": c,
                "cur_pos": policy.current_position,
                "desired": desired,
                "action": action,
                "confidence": confidence,
                "exit_action": exit_action,
                "exit_conf": exit_conf,
                "mfe": policy.mfe if policy.current_position != 0 else None,
                "mae": policy.mae if policy.current_position != 0 else None,
                "bars_held": policy.bars_held if policy.current_position != 0 else None,
            },
        }
        decision_log.append(log_entry)

    # Verify flow

    entry_found = False
    in_position_found = False
    exit_found = False

    for entry in decision_log:
        details = entry["details"]

        # Check for entry (action=1 means LONG entry signal)
        if details.get("action") == 1 and details.get("confidence") is not None:
            entry_found = True

        # Check for in-position (harvester active when cur_pos != 0)
        if details.get("mfe") is not None and details.get("exit_action") is not None:
            in_position_found = True

        # Check for exit
        if details.get("exit_action") == 1:
            exit_found = True

    # Summary
    if entry_found and in_position_found and exit_found:
        pass
    else:
        pass

    assert entry_found and in_position_found and exit_found, "Incomplete entry→in_position→exit flow"


def test_decision_log_harvester_fields():
    """Test that harvester-specific fields are captured in decision log"""

    policy = MockDualPolicy()
    policy.current_position = 1  # Already in position
    policy.entry_price = 90500.0
    policy.mfe = 0
    policy.mae = 0
    policy.bars_held = 0

    # Simulate a few bars in position
    for i in range(3):
        current_price = 90500.0 + i * 20  # Price rising
        _exit_action, _exit_conf = policy.decide_exit([], current_price=current_price)


    # Verify harvester fields exist
    required_fields = ["mfe", "mae", "bars_held", "exit_action", "exit_conf"]
    for _field in required_fields:
        pass


if __name__ == "__main__":
    results = []

    results.append(("Entry to Exit Flow", test_entry_to_exit_flow()))
    results.append(("Harvester Fields", test_decision_log_harvester_fields()))


    all_passed = True
    for _test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False


    sys.exit(0 if all_passed else 1)
