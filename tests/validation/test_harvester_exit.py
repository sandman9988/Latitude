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

        action, _conf = harvester.decide(market_state, mfe, mae, bars_held, entry_price, direction=1)

        if action == 1:
            exit_triggered = True
            break

    assert exit_triggered, "No exit triggered at profit target"


def test_stop_loss():
    """Test that harvester exits when stop loss is hit."""

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

        action, _conf = harvester.decide(market_state, mfe, mae, bars_held, entry_price, direction=1)

        if action == 1:
            exit_triggered = True
            break

    assert exit_triggered, "No exit triggered at stop loss"


def test_soft_time_stop():
    """Test that harvester exits on soft time stop (50 bars + profit)."""

    harvester = HarvesterAgent(window=10, n_features=10)

    market_state = np.zeros((10, 7), dtype=np.float32)
    entry_price = 90000.0
    mfe = entry_price * 0.06  # 6% profit (above 0.05% threshold)
    mae = 0.0

    # Test various bar counts
    exit_triggered = False
    for bars in [40, 45, 50, 51, 52]:
        action, _conf = harvester.decide(market_state, mfe, mae, bars, entry_price, direction=1)

        if action == 1:
            exit_triggered = True
            break

    assert exit_triggered, "No exit triggered at soft time stop"


def test_hard_time_stop():
    """Test that harvester exits on hard time stop (80 bars regardless)."""

    harvester = HarvesterAgent(window=10, n_features=10)

    market_state = np.zeros((10, 7), dtype=np.float32)
    entry_price = 90000.0
    mfe = entry_price * 0.02  # Small profit
    mae = entry_price * 0.01  # Small loss

    # Test various bar counts
    exit_triggered = False
    for bars in [70, 75, 79, 80, 81]:
        action, _conf = harvester.decide(market_state, mfe, mae, bars, entry_price, direction=1)

        if action == 1:
            exit_triggered = True
            break

    assert exit_triggered, "No exit triggered at hard time stop"


if __name__ == "__main__":
    results = []

    results.append(test_profit_target())
    results.append(test_stop_loss())
    results.append(test_soft_time_stop())
    results.append(test_hard_time_stop())


    if all(results):
        sys.exit(0)
    else:
        sys.exit(1)
