"""
test_circuit_breaker_gate.py
==============================
Tests that the circuit breaker gate in on_bar_close() correctly blocks
new entries when any breaker is tripped, and allows entries when clear.

Tests:
  BreakerState — trip/reset lifecycle
  CircuitBreakerManager — is_any_tripped, reset_all, get_position_size_multiplier
  update_trade + check_all integration (trips via real pnl/equity feed)
  Gate simulation — entry blocked when flat + CB tripped; exit NOT blocked
  DecisionLogger — circuit_breakers_ok=False captured in log
"""
import json

import pytest

from src.risk.circuit_breakers import (
    BreakerState,
    CircuitBreakerManager,
    ConsecutiveLossesBreaker,
    DrawdownBreaker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manager(
    max_drawdown: float = 0.20,
    max_consecutive_losses: int = 4,
) -> CircuitBreakerManager:
    return CircuitBreakerManager(
        max_drawdown=max_drawdown,
        max_consecutive_losses=max_consecutive_losses,
        auto_close_on_trip=False,
    )


def _trip_drawdown(mgr: CircuitBreakerManager, reason: str = "dd exceeded") -> None:
    """Force-trip the drawdown breaker directly via internal state."""
    dd_state = mgr.drawdown_breaker.state
    dd_state.trip(reason=reason, value=0.25, threshold=mgr.max_drawdown)


def _trip_consec_losses(mgr: CircuitBreakerManager) -> None:
    """Force-trip the consecutive-losses breaker via internal state."""
    cl_state = mgr.consecutive_losses_breaker.state
    cl_state.trip(
        reason="Too many consecutive losses",
        value=float(mgr.max_consecutive_losses),
        threshold=float(mgr.max_consecutive_losses),
    )


# ---------------------------------------------------------------------------
# BreakerState lifecycle
# ---------------------------------------------------------------------------

class TestBreakerState:
    def test_not_tripped_on_init(self):
        bs = BreakerState(name="Test")
        assert bs.is_tripped is False

    def test_trip_sets_state(self):
        bs = BreakerState(name="Test")
        bs.trip(reason="dd exceeded", value=0.06, threshold=0.05)
        assert bs.is_tripped is True
        assert "dd exceeded" in bs.trip_reason
        assert bs.trip_time is not None

    def test_reset_clears_state(self):
        bs = BreakerState(name="Test")
        bs.trip("reason", 1.0, 0.5)
        bs.reset()
        assert bs.is_tripped is False
        assert bs.trip_time is None


# ---------------------------------------------------------------------------
# CircuitBreakerManager gate
# ---------------------------------------------------------------------------

class TestCircuitBreakerGate:
    def test_fresh_manager_gate_open(self):
        mgr = _manager()
        assert mgr.is_any_tripped() is False

    def test_drawdown_trip_closes_gate(self):
        mgr = _manager()
        _trip_drawdown(mgr)
        assert mgr.is_any_tripped() is True

    def test_consecutive_losses_trip_closes_gate(self):
        mgr = _manager()
        _trip_consec_losses(mgr)
        assert mgr.is_any_tripped() is True

    def test_multiple_breakers_tripped_gate_still_closed(self):
        mgr = _manager()
        _trip_drawdown(mgr)
        _trip_consec_losses(mgr)
        assert mgr.is_any_tripped() is True
        tripped = mgr.get_tripped_breakers()
        assert len(tripped) >= 2

    def test_reset_all_reopens_gate(self):
        mgr = _manager()
        _trip_drawdown(mgr)
        assert mgr.is_any_tripped() is True
        mgr.reset_all()
        assert mgr.is_any_tripped() is False

    def test_position_size_multiplier_is_one_when_clear(self):
        mgr = _manager()
        assert mgr.get_position_size_multiplier() == pytest.approx(1.0)

    def test_position_size_multiplier_is_zero_when_tripped(self):
        mgr = _manager()
        _trip_drawdown(mgr)
        assert mgr.get_position_size_multiplier() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# update_trade + check_all integration
# ---------------------------------------------------------------------------

class TestUpdateTradeIntegration:
    """Verify breakers trip naturally via update_trade() + check_all()."""

    def test_consecutive_losses_trip_via_update(self):
        # 4 consecutive losses should trip the 4-loss breaker
        mgr = _manager(max_consecutive_losses=4)
        for _ in range(4):
            mgr.update_trade(pnl=-1.0, equity=10_000.0)
        tripped = mgr.check_all()
        assert tripped is True
        assert mgr.is_any_tripped() is True

    def test_drawdown_trip_via_update(self):
        # Start at 10_000 equity, fall to 7_500 = 25% drawdown → trips at 20%
        mgr = _manager(max_drawdown=0.20)
        mgr.update_trade(pnl=0, equity=10_000.0)  # establish peak
        mgr.update_trade(pnl=-2_500.0, equity=7_500.0)  # 25% drawdown
        tripped = mgr.check_all()
        assert tripped is True
        assert mgr.is_any_tripped() is True

    def test_no_trip_on_small_loss(self):
        mgr = _manager(max_drawdown=0.20, max_consecutive_losses=4)
        mgr.update_trade(pnl=-0.01, equity=9_999.0)
        mgr.check_all()
        assert mgr.is_any_tripped() is False

    def test_win_resets_consecutive_loss_counter(self):
        mgr = _manager(max_consecutive_losses=4)
        for _ in range(3):
            mgr.update_trade(pnl=-1.0, equity=10_000.0)
        # Win before hitting the limit
        mgr.update_trade(pnl=1.0, equity=10_000.0)
        # Three more losses (counter was reset)
        for _ in range(3):
            mgr.update_trade(pnl=-1.0, equity=10_000.0)
        mgr.check_all()
        # Should NOT be tripped (only 3 since last win, limit=4)
        cl_state = mgr.consecutive_losses_breaker.state
        assert cl_state.is_tripped is False


# ---------------------------------------------------------------------------
# Gate log field
# ---------------------------------------------------------------------------

class TestCircuitBreakerLogField:
    def test_circuit_breakers_ok_true_when_clear(self):
        mgr = _manager()
        assert (not mgr.is_any_tripped()) is True

    def test_circuit_breakers_ok_false_when_tripped(self):
        mgr = _manager()
        _trip_drawdown(mgr)
        assert (not mgr.is_any_tripped()) is False

    def test_circuit_breakers_ok_captured_in_decision_log(self, tmp_path):
        """End-to-end: DecisionLogger stores circuit_breakers_ok=False."""
        from src.monitoring.audit_logger import DecisionLogger

        dl = DecisionLogger(log_dir=str(tmp_path), filename="decisions.jsonl")
        mgr = _manager()
        _trip_drawdown(mgr)  # gate closed

        dl.log_trigger_decision(
            decision="NO_ENTRY",
            confidence=0.43,
            price=5000.0,
            volatility=0.005,
            imbalance=0.0,
            vpin_z=0.0,
            regime="UNKNOWN",
            circuit_breakers_ok=not mgr.is_any_tripped(),
        )

        entry = json.loads((tmp_path / "decisions.jsonl").read_text().strip())
        assert entry["reasoning"]["circuit_breakers_ok"] is False


# ---------------------------------------------------------------------------
# Gate simulation (mirrors on_bar_close FLAT branch logic)
# ---------------------------------------------------------------------------

class TestOnBarCloseGateSimulation:
    """
    Simulate the CB gate in on_bar_close():

        if not has_positions:
            if self.circuit_breakers.is_any_tripped():
                return   # blocks new entry

    Gate check is ONLY in the flat (no-position) branch.
    """

    def _simulate_bar_close(self, cb_manager: CircuitBreakerManager, cur_pos: int = 0):
        """
        Returns (entry_attempted: bool | None, reason: str).
        None means "in_position" path — CB gate not evaluated.
        """
        has_positions = cur_pos != 0
        if not has_positions:
            if cb_manager.is_any_tripped():
                tripped = cb_manager.get_tripped_breakers()
                return False, f"halted: {[b.name for b in tripped]}"
            return True, "entry_allowed"
        return None, "in_position"

    def test_flat_no_breakers_allows_entry(self):
        mgr = _manager()
        attempted, reason = self._simulate_bar_close(mgr, cur_pos=0)
        assert attempted is True, f"Expected entry_allowed, got: {reason}"

    def test_flat_tripped_breaker_blocks_entry(self):
        mgr = _manager()
        _trip_drawdown(mgr)
        attempted, reason = self._simulate_bar_close(mgr, cur_pos=0)
        assert attempted is False, f"Expected halted, got: {reason}"
        assert "halted" in reason

    def test_in_position_ignores_circuit_breaker(self):
        """
        CB gate is only for new entries. The harvest/exit path must
        always be reachable regardless of CB state so positions can close.
        """
        mgr = _manager()
        _trip_drawdown(mgr)
        attempted, reason = self._simulate_bar_close(mgr, cur_pos=-1)
        assert attempted is None  # in_position path, CB irrelevant
        assert reason == "in_position"

    def test_short_position_also_bypasses_gate(self):
        mgr = _manager()
        _trip_consec_losses(mgr)
        attempted, reason = self._simulate_bar_close(mgr, cur_pos=1)
        assert attempted is None
        assert reason == "in_position"
