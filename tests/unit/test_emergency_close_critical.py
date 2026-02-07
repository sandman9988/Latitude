"""
Tests for emergency_close.py — Tier 1: hedging-mode close via mfe_mae_trackers.

Covers lines 67-75: the mfe_mae_trackers fallback path where
positions are iterated and closed individually.
"""

from unittest.mock import MagicMock

import pytest

from src.risk.emergency_close import EmergencyPositionCloser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade_integration(*, position_tickets=None, mfe_mae_trackers=None, trade_manager=None):
    ti = MagicMock()
    app = MagicMock()

    if position_tickets is not None:
        ti.position_tickets = position_tickets
    else:
        del ti.position_tickets

    if mfe_mae_trackers is not None:
        app.mfe_mae_trackers = mfe_mae_trackers
    else:
        del app.mfe_mae_trackers

    ti.app = app
    ti.trade_manager = trade_manager
    return ti


# ---------------------------------------------------------------------------
# Tier 1: mfe_mae_trackers close path (hedging mode fallback)
# ---------------------------------------------------------------------------

class TestCloseByMfeMaeTrackers:
    """Lines 67-75: Close positions via app.mfe_mae_trackers when tickets unavailable."""

    def test_all_trackers_closed_successfully(self):
        trackers = {"POS_100": MagicMock(), "POS_200": MagicMock(), "POS_300": MagicMock()}
        ti = _make_trade_integration(mfe_mae_trackers=trackers)
        ti.close_position.return_value = True
        closer = EmergencyPositionCloser(ti)

        result = closer.close_all_positions("CIRCUIT_BREAK")
        assert result is True
        assert ti.close_position.call_count == 3

    def test_one_tracker_fails(self):
        trackers = {"POS_100": MagicMock(), "POS_200": MagicMock()}
        ti = _make_trade_integration(mfe_mae_trackers=trackers)
        ti.close_position.side_effect = [True, False]
        closer = EmergencyPositionCloser(ti)

        result = closer.close_all_positions("BREAKER_TRIP")
        assert result is False
        # Both positions attempted
        assert ti.close_position.call_count == 2

    def test_all_trackers_fail(self):
        trackers = {"POS_A": MagicMock(), "POS_B": MagicMock()}
        ti = _make_trade_integration(mfe_mae_trackers=trackers)
        ti.close_position.return_value = False
        closer = EmergencyPositionCloser(ti)

        result = closer.close_all_positions("EMERGENCY")
        assert result is False

    def test_tracker_exception_returns_false(self):
        trackers = {"POS_X": MagicMock()}
        ti = _make_trade_integration(mfe_mae_trackers=trackers)
        ti.close_position.side_effect = RuntimeError("FIX disconnect")
        closer = EmergencyPositionCloser(ti)

        result = closer.close_all_positions("FIX_ERROR")
        assert result is False

    def test_empty_trackers_returns_success(self):
        """Empty tracker dict → no positions to close → success."""
        ti = _make_trade_integration(mfe_mae_trackers={})
        closer = EmergencyPositionCloser(ti)

        result = closer.close_all_positions("CHECK")
        assert result is True

    def test_trackers_take_precedence_over_netting(self):
        """When trackers present, netting mode NOT used even if trade_manager exists."""
        tm = MagicMock()
        trackers = {"POS_1": MagicMock()}
        ti = _make_trade_integration(mfe_mae_trackers=trackers, trade_manager=tm)
        ti.close_position.return_value = True
        closer = EmergencyPositionCloser(ti)

        closer.close_all_positions("DUAL")
        # trade_manager.get_position should NOT be called
        tm.get_position.assert_not_called()

    def test_position_ids_passed_correctly(self):
        """Verify position IDs from tracker keys are forwarded to close_position."""
        trackers = {"POS_ABC": MagicMock()}
        ti = _make_trade_integration(mfe_mae_trackers=trackers)
        ti.close_position.return_value = True
        closer = EmergencyPositionCloser(ti)

        closer.close_all_positions("TEST_REASON")
        ti.close_position.assert_called_once_with(
            position_id="POS_ABC", reason="TEST_REASON"
        )
