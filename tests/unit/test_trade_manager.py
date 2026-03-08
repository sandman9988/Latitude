"""
TradeManager unit tests — order lifecycle, paper fill, thread safety.

Tests core trade manager logic in isolation (FIX module is stubbed).
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.trade_manager import (
    Order,
    OrderStatus,
    OrdType,
    Position,
    Side,
    TradeManager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_session_id():
    return MagicMock()


@pytest.fixture()
def manager(mock_session_id):
    """Basic TradeManager with paper_mode enabled."""
    return TradeManager(
        session_id=mock_session_id,
        symbol_id=1,
        paper_mode=True,
        get_price_callback=lambda: (2900.0, 2900.5),
    )


# ---------------------------------------------------------------------------
# Order / Position dataclasses
# ---------------------------------------------------------------------------


class TestOrder:
    def test_order_is_active_when_new(self):
        o = Order("c1", "1", Side.BUY, OrdType.MARKET, 0.01)
        assert o.is_active()
        assert not o.is_terminal()

    def test_order_is_terminal_when_filled(self):
        o = Order("c1", "1", Side.BUY, OrdType.MARKET, 0.01)
        o.status = OrderStatus.FILLED
        assert o.is_terminal()
        assert not o.is_active()

    def test_order_round_trip_dict(self):
        o = Order("c1", "1", Side.BUY, OrdType.MARKET, 0.01, price=100.0)
        d = o.to_dict()
        o2 = Order.from_dict(d)
        assert o2.clord_id == "c1"
        assert o2.side == Side.BUY

    def test_remaining_qty_after_partial_fill(self):
        o = Order("c1", "1", Side.BUY, OrdType.LIMIT, 1.0, price=100.0)
        from src.utils.safe_math import SafeMath
        o.filled_qty = SafeMath.to_decimal(0.3, 2)
        assert float(o.remaining_qty()) == pytest.approx(0.7, abs=0.01)


class TestPosition:
    def test_update_from_fill_buy(self):
        p = Position(symbol="1")
        p.update_from_fill(Side.BUY, 0.5, 100.0)
        assert float(p.long_qty) == pytest.approx(0.5, abs=0.01)
        assert float(p.net_qty) == pytest.approx(0.5, abs=0.01)

    def test_update_from_fill_sell(self):
        p = Position(symbol="1")
        p.update_from_fill(Side.SELL, 0.5, 100.0)
        assert float(p.short_qty) == pytest.approx(0.5, abs=0.01)
        assert float(p.net_qty) == pytest.approx(-0.5, abs=0.01)

    def test_seed_long(self):
        p = Position(symbol="1")
        p.seed(1.0)
        assert float(p.long_qty) == pytest.approx(1.0, abs=0.01)
        assert float(p.net_qty) == pytest.approx(1.0, abs=0.01)

    def test_seed_short(self):
        p = Position(symbol="1")
        p.seed(-0.5)
        assert float(p.short_qty) == pytest.approx(0.5, abs=0.01)
        assert float(p.net_qty) == pytest.approx(-0.5, abs=0.01)


# ---------------------------------------------------------------------------
# TradeManager order submit
# ---------------------------------------------------------------------------


class TestTradeManagerOrders:
    def test_submit_market_order_returns_order(self, manager):
        order = manager.submit_market_order(Side.BUY, 0.01)
        assert order is not None
        assert order.side == Side.BUY
        assert order.clord_id in manager.orders

    def test_submit_limit_order_returns_order(self, manager):
        order = manager.submit_limit_order(Side.SELL, 0.01, price=2900.0)
        assert order is not None
        assert order.ord_type == OrdType.LIMIT

    def test_submit_stop_order_returns_order(self, manager):
        order = manager.submit_stop_order(Side.SELL, 0.01, stop_price=2850.0)
        assert order is not None
        assert order.ord_type == OrdType.STOP

    def test_max_pending_orders_enforcement(self, manager):
        manager.max_pending_orders = 2
        manager.submit_market_order(Side.BUY, 0.01)
        manager.submit_market_order(Side.BUY, 0.01)
        third = manager.submit_market_order(Side.BUY, 0.01)
        assert third is None  # rejected

    def test_clord_id_prefix(self, manager):
        order = manager.submit_market_order(Side.BUY, 0.01, tag_prefix="ENTRY")
        assert order.clord_id.startswith("ENTRY_")

    def test_seed_position(self, manager):
        manager.seed_position(0.5)
        assert float(manager.position.net_qty) == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Paper fill simulation
# ---------------------------------------------------------------------------


class TestPaperFill:
    def test_simulate_paper_fill_updates_position(self, manager):
        order = Order("c1", "1", Side.BUY, OrdType.MARKET, 0.01)
        manager.orders["c1"] = order
        manager.pending_orders["c1"] = {"submitted_at": None, "retries": 0}

        manager._simulate_paper_fill(order, Side.BUY, 0.01)

        assert order.status == OrderStatus.FILLED
        assert float(manager.position.long_qty) == pytest.approx(0.01, abs=0.001)
        assert "c1" not in manager.pending_orders

    def test_paper_fill_calls_callback(self, manager):
        callback = MagicMock()
        manager.on_fill_callback = callback

        order = Order("c1", "1", Side.BUY, OrdType.MARKET, 0.01)
        manager.orders["c1"] = order
        manager.pending_orders["c1"] = {"submitted_at": None, "retries": 0}

        manager._simulate_paper_fill(order, Side.BUY, 0.01)
        callback.assert_called_once_with(order)

    def test_paper_fill_no_callback_when_none(self, manager):
        manager.on_fill_callback = None

        order = Order("c1", "1", Side.BUY, OrdType.MARKET, 0.01)
        manager.orders["c1"] = order
        manager.pending_orders["c1"] = {"submitted_at": None, "retries": 0}

        manager._simulate_paper_fill(order, Side.BUY, 0.01)
        assert order.status == OrderStatus.FILLED

    def test_paper_fill_with_invalid_prices_does_nothing(self, manager):
        manager.get_price_callback = lambda: (0.0, 0.0)

        order = Order("c1", "1", Side.BUY, OrdType.MARKET, 0.01)
        manager.orders["c1"] = order

        manager._simulate_paper_fill(order, Side.BUY, 0.01)
        assert order.status != OrderStatus.FILLED

    def test_paper_fill_without_price_callback_does_nothing(self, manager):
        manager.get_price_callback = None

        order = Order("c1", "1", Side.BUY, OrdType.MARKET, 0.01)
        manager.orders["c1"] = order

        manager._simulate_paper_fill(order, Side.BUY, 0.01)
        assert order.status != OrderStatus.FILLED

    def test_paper_fill_buy_fills_at_ask(self, manager):
        manager.get_price_callback = lambda: (2900.0, 2901.0)

        order = Order("c1", "1", Side.BUY, OrdType.MARKET, 0.01)
        manager.orders["c1"] = order
        manager.pending_orders["c1"] = {"submitted_at": None, "retries": 0}

        manager._simulate_paper_fill(order, Side.BUY, 0.01)
        assert order.avg_price == 2901.0

    def test_paper_fill_sell_fills_at_bid(self, manager):
        manager.get_price_callback = lambda: (2900.0, 2901.0)

        order = Order("c1", "1", Side.SELL, OrdType.MARKET, 0.01)
        manager.orders["c1"] = order
        manager.pending_orders["c1"] = {"submitted_at": None, "retries": 0}

        manager._simulate_paper_fill(order, Side.SELL, 0.01)
        assert order.avg_price == 2900.0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_lock_exists(self, manager):
        assert hasattr(manager, "_lock")
        assert isinstance(manager._lock, type(threading.Lock()))

    def test_concurrent_paper_fills_no_crash(self, manager):
        """Stress test: multiple paper fills from concurrent threads."""
        errors = []

        def fill_order(idx):
            try:
                order = Order(f"c{idx}", "1", Side.BUY, OrdType.MARKET, 0.01)
                manager.orders[f"c{idx}"] = order
                manager.pending_orders[f"c{idx}"] = {"submitted_at": None, "retries": 0}
                manager._simulate_paper_fill(order, Side.BUY, 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fill_order, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"

    def test_check_paper_fill_timeout_skips_already_filled(self, manager):
        """Order no longer in pending_orders → skip."""
        order = Order("c1", "1", Side.BUY, OrdType.MARKET, 0.01)
        manager.orders["c1"] = order
        # Not in pending_orders → should return silently
        manager._check_paper_fill_timeout("c1", Side.BUY, 0.01)
        assert order.status != OrderStatus.FILLED


# ---------------------------------------------------------------------------
# Execution report tracking
# ---------------------------------------------------------------------------


class TestExecReportTracking:
    def test_exec_reports_appended_on_paper_fill(self, manager):
        order = Order("c1", "1", Side.BUY, OrdType.MARKET, 0.01)
        manager.orders["c1"] = order
        manager.pending_orders["c1"] = {"submitted_at": None, "retries": 0}

        manager._simulate_paper_fill(order, Side.BUY, 0.01)

        assert len(manager.exec_reports) == 1
        assert manager.exec_reports[0]["paper_fill"] is True
        assert manager.exec_reports[0]["exec_type"] == "F"
