"""Extended tests for src.risk.circuit_breakers.

Covers: _resolve_param edge cases (invalid explicit, learned fallback),
_execute_emergency_close paths (no closer, success, failure, exception),
restore_state corrupt data, save/restore roundtrip with actual breaker values.
"""

from unittest.mock import MagicMock

import pytest

from src.risk.circuit_breakers import CircuitBreakerManager


# ---------------------------------------------------------------------------
# _resolve_param
# ---------------------------------------------------------------------------
class TestResolveParam:
    def test_explicit_valid_value(self):
        mgr = CircuitBreakerManager(sortino_threshold=0.8)
        assert mgr.sortino_threshold == pytest.approx(0.8)

    def test_explicit_invalid_value_falls_back(self):
        """Invalid explicit (non-castable) should fall back to default."""
        mgr = CircuitBreakerManager(sortino_threshold="bad_value")
        # Should fall back to default (0.5) since "bad_value" can't be float()
        assert mgr.sortino_threshold == pytest.approx(0.5)

    def test_learned_parameters_used(self, tmp_path):
        """When param_manager is set and no explicit override, learned value is used."""
        from src.persistence.learned_parameters import LearnedParametersManager

        pm = LearnedParametersManager(persistence_path=tmp_path / "params.json")
        inst = pm.get_instrument("BTCUSD", timeframe="M15", broker="default")
        inst.params["sortino_threshold"].value = 1.5  # Override default

        mgr = CircuitBreakerManager(param_manager=pm, symbol="BTCUSD", timeframe="M15")
        # Should pick up the learned value
        assert mgr.sortino_threshold == pytest.approx(1.5)

    def test_param_manager_exception_falls_back(self):
        """If param_manager.get raises, fall back to default."""
        bad_pm = MagicMock()
        bad_pm.get.side_effect = RuntimeError("boom")
        mgr = CircuitBreakerManager(param_manager=bad_pm)
        # Should still initialise with defaults
        assert mgr.sortino_threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _execute_emergency_close
# ---------------------------------------------------------------------------
class TestExecuteEmergencyClose:
    def _make_tripped_mgr(self) -> CircuitBreakerManager:
        mgr = CircuitBreakerManager(max_consecutive_losses=2, auto_close_on_trip=True)
        for _ in range(3):
            mgr.update_trade(-0.05, 10000.0)
        return mgr

    def test_no_closer_configured(self):
        mgr = self._make_tripped_mgr()
        mgr.emergency_closer = None
        mgr.check_all()  # Should log warning but not crash
        assert mgr.positions_closed_on_trip is False

    def test_closer_success(self):
        mgr = self._make_tripped_mgr()
        closer = MagicMock()
        closer.close_all_positions.return_value = True
        mgr.set_emergency_closer(closer)
        mgr.positions_closed_on_trip = False
        mgr.check_all()
        assert mgr.positions_closed_on_trip is True

    def test_closer_failure(self):
        mgr = self._make_tripped_mgr()
        closer = MagicMock()
        closer.close_all_positions.return_value = False
        mgr.set_emergency_closer(closer)
        mgr.positions_closed_on_trip = False
        mgr.check_all()
        assert mgr.positions_closed_on_trip is False

    def test_closer_exception(self):
        mgr = self._make_tripped_mgr()
        closer = MagicMock()
        closer.close_all_positions.side_effect = Exception("network error")
        mgr.set_emergency_closer(closer)
        mgr.positions_closed_on_trip = False
        mgr.check_all()  # Should not raise
        assert mgr.positions_closed_on_trip is False

    def test_no_double_close(self):
        """Once positions_closed_on_trip is True, no second close."""
        mgr = self._make_tripped_mgr()
        closer = MagicMock()
        closer.close_all_positions.return_value = True
        mgr.set_emergency_closer(closer)
        mgr.check_all()
        call_count = closer.close_all_positions.call_count
        mgr.check_all()
        assert closer.close_all_positions.call_count == call_count


# ---------------------------------------------------------------------------
# save_state / restore_state extended
# ---------------------------------------------------------------------------
class TestPersistenceExtended:
    def test_restore_corrupt_json(self, tmp_path):
        filepath = str(tmp_path / "corrupt.json")
        with open(filepath, "w") as f:
            f.write("{broken json")
        mgr = CircuitBreakerManager()
        assert mgr.restore_state(filepath) is False

    def test_save_creates_dirs(self, tmp_path):
        filepath = str(tmp_path / "sub" / "dir" / "state.json")
        mgr = CircuitBreakerManager()
        mgr.save_state(filepath)
        assert (tmp_path / "sub" / "dir" / "state.json").exists()

    def test_roundtrip_preserves_breaker_data(self, tmp_path):
        filepath = str(tmp_path / "cb_state.json")
        mgr = CircuitBreakerManager()
        # Feed some returns to sortino/kurtosis
        for i in range(15):
            mgr.update_trade(0.01 * (i % 3 - 1), 10000.0 + i * 10)
        mgr.save_state(filepath)

        mgr2 = CircuitBreakerManager()
        assert mgr2.restore_state(filepath) is True
        # Sortino returns should be restored
        assert len(mgr2.sortino_breaker.returns) == len(mgr.sortino_breaker.returns)

    def test_restore_partial_state(self, tmp_path):
        """Restoring a file missing some breaker keys should not crash."""
        import json

        filepath = str(tmp_path / "partial.json")
        with open(filepath, "w") as f:
            json.dump({"timestamp": 0, "sortino": {"is_tripped": False, "trip_time": None, "returns": []}}, f)
        mgr = CircuitBreakerManager()
        assert mgr.restore_state(filepath) is True


# ---------------------------------------------------------------------------
# Position multiplier & status
# ---------------------------------------------------------------------------
class TestStatusExtended:
    def test_drawdown_multiplier_without_trip(self):
        mgr = CircuitBreakerManager()
        # Small drawdown
        mgr.drawdown_breaker.update(10000.0)
        mgr.drawdown_breaker.update(9500.0)  # 5% drawdown
        mult = mgr.get_position_size_multiplier()
        assert 0 < mult <= 1.0

    def test_status_dict_has_all_keys(self):
        mgr = CircuitBreakerManager()
        status = mgr.get_status()
        for key in ("any_tripped", "position_multiplier", "sortino", "kurtosis", "drawdown", "consecutive_losses"):
            assert key in status
        assert "size_mult" in status["drawdown"]
