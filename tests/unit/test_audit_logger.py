"""Tests for src.monitoring.audit_logger – TransactionLogger & DecisionLogger."""

import json

import pytest

from src.monitoring.audit_logger import DecisionLogger, TransactionLogger


# ---------------------------------------------------------------------------
# TransactionLogger
# ---------------------------------------------------------------------------

class TestTransactionLogger:
    @pytest.fixture()
    def logger(self, tmp_path):
        return TransactionLogger(log_dir=str(tmp_path), filename="tx.jsonl")

    def _read_entries(self, logger):
        with open(logger.log_file) as f:
            return [json.loads(line) for line in f]

    # -- init --
    def test_session_start_written(self, logger):
        entries = self._read_entries(logger)
        assert len(entries) >= 1
        assert entries[0]["event_type"] == "SESSION_START"
        assert "session_id" in entries[0]["data"]

    def test_session_id_format(self, logger):
        assert logger.session_id.startswith("session_")

    # -- log_event --
    def test_log_event_basic(self, logger):
        logger.log_event("CUSTOM", {"key": "val"}, severity="WARNING")
        entries = self._read_entries(logger)
        last = entries[-1]
        assert last["event_type"] == "CUSTOM"
        assert last["severity"] == "WARNING"
        assert last["data"]["key"] == "val"
        assert "timestamp" in last

    # -- convenience methods --
    def test_log_order_submit(self, logger):
        logger.log_order_submit("ORD1", "BUY", 0.1, 100.0)
        entries = self._read_entries(logger)
        last = entries[-1]
        assert last["event_type"] == "ORDER_SUBMIT"
        assert last["data"]["side"] == "BUY"
        assert last["data"]["quantity"] == pytest.approx(0.1)

    def test_log_order_fill(self, logger):
        logger.log_order_fill("ORD1", 101.0, 0.1, position_id="POS1")
        entries = self._read_entries(logger)
        last = entries[-1]
        assert last["event_type"] == "ORDER_FILL"
        assert last["data"]["fill_price"] == pytest.approx(101.0)

    def test_log_order_reject(self, logger):
        logger.log_order_reject("ORD2", "Insufficient margin")
        entries = self._read_entries(logger)
        last = entries[-1]
        assert last["event_type"] == "ORDER_REJECT"
        assert last["severity"] == "WARNING"

    def test_log_position_update(self, logger):
        logger.log_position_update("POS1", 0.2, 99.5)
        entries = self._read_entries(logger)
        last = entries[-1]
        assert last["event_type"] == "POSITION_UPDATE"

    def test_log_position_close(self, logger):
        logger.log_position_close("POS1", 50.0, 75.0, 25.0)
        entries = self._read_entries(logger)
        last = entries[-1]
        assert last["event_type"] == "POSITION_CLOSE"
        assert last["data"]["pnl"] == pytest.approx(50.0)

    def test_log_circuit_breaker_tripped(self, logger):
        logger.log_circuit_breaker("Sortino", True, 0.3, 0.5)
        entries = self._read_entries(logger)
        last = entries[-1]
        assert last["event_type"] == "CIRCUIT_BREAKER"
        assert last["severity"] == "WARNING"
        assert last["data"]["tripped"] is True

    def test_log_circuit_breaker_not_tripped(self, logger):
        logger.log_circuit_breaker("Sortino", False, 0.8, 0.5)
        entries = self._read_entries(logger)
        last = entries[-1]
        assert last["severity"] == "INFO"
        assert last["data"]["tripped"] is False

    def test_log_session_event(self, logger):
        logger.log_session_event("TRADE", "LOGON", {"ip": "10.0.0.1"})
        entries = self._read_entries(logger)
        last = entries[-1]
        assert last["event_type"] == "SESSION_EVENT"
        assert last["data"]["event"] == "LOGON"

    def test_log_component_health_unhealthy(self, logger):
        logger.log_component_health("FIX", False, error_count=3)
        entries = self._read_entries(logger)
        last = entries[-1]
        assert last["event_type"] == "COMPONENT_HEALTH"
        assert last["severity"] == "ERROR"

    def test_log_component_health_healthy(self, logger):
        logger.log_component_health("FIX", True)
        entries = self._read_entries(logger)
        last = entries[-1]
        assert last["severity"] == "INFO"

    def test_log_dir_created(self, tmp_path):
        sub = tmp_path / "deep" / "nested"
        _logger = TransactionLogger(log_dir=str(sub))
        assert sub.exists()


# ---------------------------------------------------------------------------
# DecisionLogger
# ---------------------------------------------------------------------------

class TestDecisionLogger:
    @pytest.fixture()
    def logger(self, tmp_path):
        return DecisionLogger(log_dir=str(tmp_path), filename="decisions.jsonl")

    def _read_entries(self, logger):
        with open(logger.log_file) as f:
            return [json.loads(line) for line in f]

    def test_log_decision_generic(self, logger):
        logger.log_decision("TestAgent", "BUY", 0.8, {"price": 100}, reasoning={"r": 1})
        entries = self._read_entries(logger)
        assert len(entries) == 1
        assert entries[0]["agent"] == "TestAgent"
        assert entries[0]["confidence"] == pytest.approx(0.8)
        assert entries[0]["reasoning"]["r"] == 1

    def test_log_trigger_decision(self, logger):
        logger.log_trigger_decision(
            decision="LONG",
            confidence=0.75,
            price=100.0,
            volatility=0.005,
            imbalance=0.1,
            vpin_z=0.5,
            regime="TRENDING",
            predicted_runway=150.0,
            feasibility=0.85,
        )
        entries = self._read_entries(logger)
        last = entries[0]
        assert last["agent"] == "TriggerAgent"
        assert last["decision"] == "LONG"
        assert last["context"]["price"] == pytest.approx(100.0)
        assert last["reasoning"]["feasibility"] == pytest.approx(0.85)

    def test_log_harvester_decision(self, logger):
        logger.log_harvester_decision(
            decision="CLOSE",
            confidence=0.9,
            price=100050.0,
            entry_price=100000.0,
            mfe=75.0,
            mae=25.0,
            ticks_held=10,
            unrealized_pnl=50.0,
            capture_ratio=0.67,
        )
        entries = self._read_entries(logger)
        last = entries[0]
        assert last["agent"] == "HarvesterAgent"
        assert last["reasoning"]["capture_ratio"] == pytest.approx(0.67)

    def test_decision_no_reasoning(self, logger):
        logger.log_decision("X", "Y", 0.5, {})
        entries = self._read_entries(logger)
        assert entries[0]["reasoning"] == {}

    def test_session_id_format(self, logger):
        assert logger.session_id.startswith("session_")
