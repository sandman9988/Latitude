"""
TradeManagerIntegration unit tests — trailing stop, state persistence.

Verifies the integration layer between CTraderFixApp and TradeManager:
- Trailing stop lifecycle (enable → update → disable)
- Trailing stop uses stop orders (not limit orders)
- Position recovery state
- Ghost position reconciliation
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from src.core.trade_manager import Order, OrderStatus, OrdType, Side, TradeManager
from src.core.trade_manager_integration import TradeManagerIntegration

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app() -> MagicMock:
    """Create a minimal mock CTraderFixApp."""
    app = MagicMock()
    app.symbol = "XAUUSD"
    app.qty = 0.01
    app.mfe_mae_trackers = {}
    app.path_recorders = {}
    app._tracker_lock = MagicMock()
    app.friction_calculator = MagicMock()
    app.friction_calculator.normalize_price = lambda p: p
    app.friction_calculator.costs = MagicMock(digits=2)
    return app


def _make_integration(app=None) -> TradeManagerIntegration:
    """Create integration instance with mocked components."""
    if app is None:
        app = _make_app()
    integration = TradeManagerIntegration(app)
    # Create a mock TradeManager
    tm = MagicMock(spec=TradeManager)
    integration.trade_manager = tm
    return integration


# ---------------------------------------------------------------------------
# Trailing Stop — uses submit_stop_order (not limit)
# ---------------------------------------------------------------------------


class TestTrailingStopOrderType:
    """Verify the P1 bug fix: trailing stop uses stop orders."""

    def test_submit_stop_order_called(self):
        integ = _make_integration()
        integ.position_direction = 1  # LONG
        integ.trailing_stop_order = None  # No existing stop

        integ._submit_stop_order(2850.0)

        integ.trade_manager.submit_stop_order.assert_called_once()
        integ.trade_manager.submit_limit_order.assert_not_called()

    def test_submit_stop_order_side_sell_for_long(self):
        integ = _make_integration()
        integ.position_direction = 1
        integ.trailing_stop_order = None

        integ._submit_stop_order(2850.0)

        call_kwargs = integ.trade_manager.submit_stop_order.call_args
        assert call_kwargs[1]["side"] == Side.SELL or call_kwargs[0][0] == Side.SELL

    def test_submit_stop_order_side_buy_for_short(self):
        integ = _make_integration()
        integ.position_direction = -1
        integ.trailing_stop_order = None

        integ._submit_stop_order(2950.0)

        call_args = integ.trade_manager.submit_stop_order.call_args
        # Check positional or keyword
        side = call_args[1].get("side", call_args[0][0] if call_args[0] else None)
        assert side == Side.BUY

    def test_modify_existing_stop_order(self):
        integ = _make_integration()
        integ.position_direction = 1

        # Set up existing stop order
        existing_order = MagicMock()
        existing_order.clord_id = "STOP_cl_123"
        integ.trailing_stop_order = existing_order

        integ._submit_stop_order(2860.0)

        integ.trade_manager.modify_order.assert_called_once()
        integ.trade_manager.submit_stop_order.assert_not_called()


# ---------------------------------------------------------------------------
# Trailing Stop lifecycle
# ---------------------------------------------------------------------------


class TestTrailingStopLifecycle:
    def test_disable_trailing_stop_cancels_order(self):
        integ = _make_integration()
        existing_order = MagicMock()
        existing_order.clord_id = "STOP_123"
        integ.trailing_stop_order = existing_order
        integ.trailing_stop_active = True

        integ.disable_trailing_stop()

        integ.trade_manager.cancel_order.assert_called_once_with("STOP_123")
        assert integ.trailing_stop_active is False
        assert integ.trailing_stop_order is None

    def test_disable_trailing_stop_no_order(self):
        integ = _make_integration()
        integ.trailing_stop_order = None
        integ.trailing_stop_active = True

        integ.disable_trailing_stop()

        integ.trade_manager.cancel_order.assert_not_called()
        assert integ.trailing_stop_active is False

    def test_disable_clears_price_tracking(self):
        integ = _make_integration()
        integ.highest_price_since_entry = 2950.0
        integ.lowest_price_since_entry = 2850.0
        integ.entry_price = 2900.0
        integ.position_direction = 1

        integ.disable_trailing_stop()

        assert integ.highest_price_since_entry is None
        assert integ.lowest_price_since_entry is None
        assert integ.entry_price is None
        assert integ.position_direction == 0


# ---------------------------------------------------------------------------
# Ghost position reconciliation
# ---------------------------------------------------------------------------


class TestGhostPositionReconciliation:
    def test_no_action_when_in_position(self):
        integ = _make_integration()
        integ.app.cur_pos = 1  # In LONG position
        assert integ.reconcile_ghost_positions() is False

    def test_cleanup_ghost_tickets(self):
        integ = _make_integration()
        integ.app.cur_pos = 0
        integ.position_tickets = {"TICKET1": "pos1", "TICKET2": "pos2"}

        result = integ.reconcile_ghost_positions()
        assert result is True
        assert len(integ.position_tickets) == 0

    def test_has_any_open_positions_with_tickets(self):
        integ = _make_integration()
        integ.position_tickets = {"TICKET1": "pos1"}
        assert integ.has_any_open_positions() is True

    def test_has_any_open_positions_empty(self):
        integ = _make_integration()
        integ.position_tickets = {}
        assert integ.has_any_open_positions() is False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestIntegrationInit:
    def test_defaults(self):
        app = _make_app()
        integ = TradeManagerIntegration(app)
        assert integ.trailing_stop_active is False
        assert integ.trailing_stop_order is None
        assert integ.position_direction == 0
        assert integ.trade_manager is None  # Not initialized until setup_trade_manager
