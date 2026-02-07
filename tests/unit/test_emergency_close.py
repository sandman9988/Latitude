"""Tests for src.risk.emergency_close – EmergencyPositionCloser."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.risk.emergency_close import EmergencyPositionCloser, create_emergency_closer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_trade_integration(
    *,
    position_tickets=None,
    mfe_mae_trackers=None,
    trade_manager=None,
):
    """Build a minimal mock trade_integration."""
    ti = MagicMock()
    app = MagicMock()

    # position_tickets
    if position_tickets is not None:
        ti.position_tickets = position_tickets
    else:
        del ti.position_tickets  # hasattr returns False

    # mfe_mae_trackers on app
    if mfe_mae_trackers is not None:
        app.mfe_mae_trackers = mfe_mae_trackers
    else:
        del app.mfe_mae_trackers

    ti.app = app

    # trade_manager
    if trade_manager is not None:
        ti.trade_manager = trade_manager
    else:
        ti.trade_manager = None

    return ti


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

class TestEmergencyPositionCloserInit:
    def test_init_stores_attrs(self):
        ti = _make_trade_integration()
        closer = EmergencyPositionCloser(ti)
        assert closer.trade_integration is ti
        assert closer.close_attempts == 0
        assert closer.max_retries == 3


# ---------------------------------------------------------------------------
# close_all_positions – hedging mode (position_tickets)
# ---------------------------------------------------------------------------

class TestCloseByTickets:
    def test_close_via_tickets_all_succeed(self):
        ti = _make_trade_integration(position_tickets={"T1": "POS_A", "T2": "POS_B"})
        ti.close_position.return_value = True
        closer = EmergencyPositionCloser(ti)

        result = closer.close_all_positions("TEST")
        assert result is True
        assert ti.close_position.call_count == 2

    def test_close_via_tickets_one_fails(self):
        ti = _make_trade_integration(position_tickets={"T1": "POS_A", "T2": "POS_B"})
        ti.close_position.side_effect = [True, False]
        closer = EmergencyPositionCloser(ti)

        result = closer.close_all_positions("TEST")
        assert result is False

    def test_close_via_tickets_exception(self):
        ti = _make_trade_integration(position_tickets={"T1": "POS_A"})
        ti.close_position.side_effect = RuntimeError("FIX down")
        closer = EmergencyPositionCloser(ti)

        # _close_by_position_id catches exceptions and returns False
        result = closer.close_all_positions("TEST")
        assert result is False


# ---------------------------------------------------------------------------
# close_all_positions – mfe_mae_trackers fallback
# ---------------------------------------------------------------------------

class TestCloseByTrackers:
    def test_close_via_trackers(self):
        ti = _make_trade_integration(mfe_mae_trackers={"POS_1": MagicMock()})
        ti.close_position.return_value = True
        closer = EmergencyPositionCloser(ti)

        result = closer.close_all_positions("BREAKER")
        assert result is True
        ti.close_position.assert_called_once()


# ---------------------------------------------------------------------------
# close_all_positions – netting mode fallback
# ---------------------------------------------------------------------------

class TestCloseNetPosition:
    def test_close_net_position_nonzero(self):
        tm = MagicMock()
        tm.get_position.return_value = SimpleNamespace(net_qty=0.1)
        ti = _make_trade_integration(trade_manager=tm)
        ti.close_position.return_value = True
        closer = EmergencyPositionCloser(ti)

        result = closer.close_all_positions("NET_MODE")
        assert result is True

    def test_no_net_position(self):
        tm = MagicMock()
        tm.get_position.return_value = SimpleNamespace(net_qty=0.0)
        ti = _make_trade_integration(trade_manager=tm)
        closer = EmergencyPositionCloser(ti)

        result = closer.close_all_positions("NOTHING")
        assert result is True  # no positions to close

    def test_close_net_position_exception(self):
        tm = MagicMock()
        tm.get_position.return_value = SimpleNamespace(net_qty=0.5)
        ti = _make_trade_integration(trade_manager=tm)
        ti.close_position.side_effect = RuntimeError("boom")
        closer = EmergencyPositionCloser(ti)

        result = closer.close_all_positions("ERR")
        assert result is False


# ---------------------------------------------------------------------------
# verify_all_closed
# ---------------------------------------------------------------------------

class TestVerifyAllClosed:
    def test_all_clear(self):
        ti = _make_trade_integration()
        ti.trade_manager = None
        closer = EmergencyPositionCloser(ti)
        assert closer.verify_all_closed() is True

    def test_trackers_still_active(self):
        ti = _make_trade_integration(mfe_mae_trackers={"POS_1": MagicMock()})
        closer = EmergencyPositionCloser(ti)
        assert closer.verify_all_closed() is False

    def test_tickets_still_active(self):
        ti = _make_trade_integration(position_tickets={"T1": "POS_A"})
        closer = EmergencyPositionCloser(ti)
        assert closer.verify_all_closed() is False

    def test_trade_manager_still_has_position(self):
        tm = MagicMock()
        tm.get_position.return_value = SimpleNamespace(net_qty=0.1, long_qty=0.1, short_qty=0.0)
        ti = _make_trade_integration(trade_manager=tm)
        closer = EmergencyPositionCloser(ti)
        assert closer.verify_all_closed() is False

    def test_trade_manager_all_zero(self):
        tm = MagicMock()
        tm.get_position.return_value = SimpleNamespace(net_qty=0.0, long_qty=0.0, short_qty=0.0)
        ti = _make_trade_integration(trade_manager=tm)
        closer = EmergencyPositionCloser(ti)
        assert closer.verify_all_closed() is True


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_create_emergency_closer(self):
        ti = _make_trade_integration()
        closer = create_emergency_closer(ti)
        assert isinstance(closer, EmergencyPositionCloser)
        assert closer.trade_integration is ti
