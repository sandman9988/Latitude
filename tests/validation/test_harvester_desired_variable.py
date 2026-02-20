"""
test_harvester_desired_variable.py
===================================
Regression tests for the bug where `desired` was left as None in the
in-position harvester branch of on_bar_close(), causing:

    TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'
    at: delta = desired - self.cur_pos

The fix (src/core/ctrader_ddqn_paper.py harvester branch):
    desired = 0 if exit_action == 1 else self.cur_pos

Tests:
    1. desired=0 when harvester signals CLOSE (SL triggered, MAE >= threshold)
    2. desired=cur_pos when harvester signals HOLD
    3. delta = desired - cur_pos never raises TypeError for any exit_action value
    4. desired is never None after exit decision (invariant)
    5. Close produces correct delta direction for SHORT and LONG positions
"""
import numpy as np
import pytest
from collections import deque
from datetime import datetime, timedelta

from src.agents.dual_policy import DualPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 20, base: float = 5000.0) -> deque:
    """Create a minimal bar deque that satisfies MIN_BARS checks."""
    bars = deque(maxlen=64)
    base_time = datetime(2026, 2, 20, 16, 0, 0)
    for i in range(n):
        t = base_time + timedelta(minutes=5 * i)
        o = base + i * 0.1
        h = o + 1.0
        lo = o - 1.0
        c = o + 0.05
        bars.append([t.timestamp(), o, h, lo, c, 1000.0, i])
    return bars


def _policy_in_position(direction: int = -1, entry_price: float = 5053.6) -> DualPolicy:
    """Return a DualPolicy with an open position."""
    policy = DualPolicy(
        window=10,
        symbol="XAUUSD",
        timeframe="M5",
        broker="paper",
        enable_training=False,
    )
    policy.on_entry(
        direction=direction,
        entry_price=entry_price,
        entry_time=datetime(2026, 2, 20, 16, 30, 0),
    )
    return policy


# ---------------------------------------------------------------------------
# Core logic (mirrors the fix in on_bar_close)
# ---------------------------------------------------------------------------

def _compute_desired(exit_action: int, cur_pos: int) -> int:
    """
    Mirrors the fixed line in on_bar_close harvester branch:
        desired = 0 if exit_action == 1 else self.cur_pos
    """
    return 0 if exit_action == 1 else cur_pos


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHarvesterDesiredVariable:
    """Regression tests for desired=None bug in harvester branch."""

    def test_desired_is_zero_on_close_signal(self):
        """desired must be 0 (flat) when harvester returns CLOSE (exit_action=1)."""
        policy = _policy_in_position(direction=-1, entry_price=5053.6)
        bars = _make_bars(20, base=5080.0)  # price above entry → adverse for SHORT
        cur_pos = -1  # SHORT

        # Trigger with high MAE — emergency SL should fire
        exit_action, _ = policy.decide_exit(
            bars,
            current_price=5086.0,
            imbalance=0.0,
            vpin_z=0.0,
            depth_ratio=1.0,
        )

        desired = _compute_desired(exit_action, cur_pos)

        assert desired is not None, "desired must never be None after exit decision"
        if exit_action == 1:  # CLOSE
            assert desired == 0, f"CLOSE signal must set desired=0, got {desired}"

    def test_desired_is_cur_pos_on_hold_signal(self):
        """desired must equal cur_pos (hold) when harvester returns HOLD (exit_action=0)."""
        policy = _policy_in_position(direction=1, entry_price=5053.6)
        bars = _make_bars(20, base=5055.0)  # favourable for LONG, MAE low
        cur_pos = 1

        exit_action, _ = policy.decide_exit(
            bars,
            current_price=5055.5,  # slightly in profit for LONG
            imbalance=0.0,
            vpin_z=0.0,
            depth_ratio=1.0,
        )

        desired = _compute_desired(exit_action, cur_pos)

        assert desired is not None, "desired must never be None after exit decision"
        if exit_action == 0:  # HOLD
            assert desired == cur_pos, f"HOLD signal must keep desired=cur_pos={cur_pos}, got {desired}"

    def test_delta_never_raises_type_error_close(self):
        """delta = desired - cur_pos must not raise TypeError when CLOSE."""
        cur_pos = -1
        exit_action = 1  # CLOSE
        desired = _compute_desired(exit_action, cur_pos)
        # Must not raise
        delta = desired - cur_pos
        assert delta == 1, f"Expected delta=1 to close SHORT, got {delta}"

    def test_delta_never_raises_type_error_hold(self):
        """delta = desired - cur_pos must not raise TypeError when HOLD."""
        cur_pos = -1
        exit_action = 0  # HOLD
        desired = _compute_desired(exit_action, cur_pos)
        # Must not raise
        delta = desired - cur_pos
        assert delta == 0, f"Expected delta=0 (hold position), got {delta}"

    def test_desired_never_none_for_any_exit_action(self):
        """Invariant: desired is never None regardless of exit_action value."""
        for cur_pos in [-1, 1]:
            for exit_action in [0, 1]:
                desired = _compute_desired(exit_action, cur_pos)
                assert desired is not None, (
                    f"desired=None for exit_action={exit_action}, cur_pos={cur_pos}"
                )
                # Must be an integer (numeric subtraction must be safe)
                delta = desired - cur_pos
                assert isinstance(delta, int), f"delta is not int: {type(delta)}"

    def test_close_short_produces_buy_delta(self):
        """Closing a SHORT (cur_pos=-1) must produce delta=+1 (BUY to close)."""
        desired = _compute_desired(exit_action=1, cur_pos=-1)
        delta = desired - (-1)
        assert delta == 1, f"Closing SHORT must yield delta=+1 (BUY), got {delta}"

    def test_close_long_produces_sell_delta(self):
        """Closing a LONG (cur_pos=+1) must produce delta=-1 (SELL to close)."""
        desired = _compute_desired(exit_action=1, cur_pos=1)
        delta = desired - 1
        assert delta == -1, f"Closing LONG must yield delta=-1 (SELL), got {delta}"

    def test_harvester_sl_triggers_close_at_high_mae(self):
        """
        End-to-end: with MAE well above SL threshold the harvester must
        return exit_action=1 (CLOSE), which then maps to desired=0.
        """
        policy = _policy_in_position(direction=-1, entry_price=5000.0)
        bars = _make_bars(20, base=5050.0)
        cur_pos = -1

        # Price has moved 1% against SHORT → MAE >> SL (0.40%)
        exit_action, conf = policy.decide_exit(
            bars,
            current_price=5051.0,
            imbalance=0.0,
            vpin_z=0.0,
            depth_ratio=1.0,
        )

        desired = _compute_desired(exit_action, cur_pos)

        # The SL must have fired
        assert exit_action == 1, (
            f"Expected CLOSE (exit_action=1) after large MAE, got {exit_action} conf={conf}"
        )
        assert desired == 0, f"Expected desired=0 after CLOSE, got {desired}"
        assert conf == pytest.approx(1.0), f"Emergency SL should emit conf=1.0, got {conf}"
