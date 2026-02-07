"""Extended tests for src.monitoring.audit_logger.

Covers DecisionLogger.log_trigger_decision, log_harvester_decision details,
TransactionLogger write-error path, and multi-threaded safety.
"""

import json
import threading
import pytest
from pathlib import Path

from src.monitoring.audit_logger import DecisionLogger, TransactionLogger


# ---------------------------------------------------------------------------
# TransactionLogger extended
# ---------------------------------------------------------------------------
class TestTransactionLoggerExt:
    @pytest.fixture()
    def logger(self, tmp_path):
        return TransactionLogger(log_dir=str(tmp_path), filename="tx.jsonl")

    def _read_entries(self, logger):
        with open(logger.log_file) as f:
            return [json.loads(line) for line in f]

    def test_severity_propagated(self, logger):
        logger.log_event("TEST", {"x": 1}, severity="CRITICAL")
        entries = self._read_entries(logger)
        critical = [e for e in entries if e["severity"] == "CRITICAL"]
        assert len(critical) == 1

    def test_log_session_event(self, logger):
        logger.log_session_event("QUOTE", "LOGON", {"user": "bot"})
        entries = self._read_entries(logger)
        session_events = [e for e in entries if e["event_type"] == "SESSION_EVENT"]
        assert len(session_events) == 1
        assert session_events[0]["data"]["session_type"] == "QUOTE"
        assert session_events[0]["data"]["event"] == "LOGON"

    def test_log_session_event_no_details(self, logger):
        logger.log_session_event("TRADE", "DISCONNECT")
        entries = self._read_entries(logger)
        session_events = [e for e in entries if e["event_type"] == "SESSION_EVENT"]
        assert session_events[0]["data"]["details"] == {}

    def test_log_component_health_unhealthy_severity(self, logger):
        logger.log_component_health("FIX", healthy=False, error_count=5)
        entries = self._read_entries(logger)
        health = [e for e in entries if e["event_type"] == "COMPONENT_HEALTH"]
        assert health[0]["severity"] == "ERROR"
        assert health[0]["data"]["error_count"] == 5

    def test_log_component_health_healthy_severity(self, logger):
        logger.log_component_health("MODEL", healthy=True)
        entries = self._read_entries(logger)
        health = [e for e in entries if e["event_type"] == "COMPONENT_HEALTH"]
        assert health[0]["severity"] == "INFO"

    def test_concurrent_writes(self, logger):
        """Multiple threads writing should not corrupt the file."""
        def write_events(tid):
            for i in range(20):
                logger.log_event("CONCURRENT", {"thread": tid, "idx": i})

        threads = [threading.Thread(target=write_events, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = self._read_entries(logger)
        # 1 session start + 4*20 = 81
        concurrent = [e for e in entries if e["event_type"] == "CONCURRENT"]
        assert len(concurrent) == 80

    def test_write_error_handled(self, tmp_path):
        """If log file path is invalid, log_event should not raise."""
        logger = TransactionLogger.__new__(TransactionLogger)
        logger.log_dir = tmp_path
        logger.log_file = Path("/dev/null/impossible/path.jsonl")
        logger.lock = threading.Lock()
        logger.session_id = "test"
        # Should silently log error, not raise
        logger.log_event("TEST", {"x": 1})


# ---------------------------------------------------------------------------
# DecisionLogger extended
# ---------------------------------------------------------------------------
class TestDecisionLoggerExt:
    @pytest.fixture()
    def logger(self, tmp_path):
        return DecisionLogger(log_dir=str(tmp_path), filename="dec.jsonl")

    def _read_entries(self, logger):
        with open(logger.log_file) as f:
            return [json.loads(line) for line in f]

    def test_log_trigger_decision_full(self, logger):
        logger.log_trigger_decision(
            decision="LONG",
            confidence=0.85,
            price=100000.0,
            volatility=0.005,
            imbalance=0.15,
            vpin_z=1.2,
            regime="TRENDING",
            predicted_runway=200.0,
            feasibility=0.9,
            circuit_breakers_ok=True,
        )
        entries = self._read_entries(logger)
        assert len(entries) == 1
        e = entries[0]
        assert e["agent"] == "TriggerAgent"
        assert e["decision"] == "LONG"
        assert e["confidence"] == pytest.approx(0.85)
        assert e["context"]["price"] == pytest.approx(100000.0)
        assert e["context"]["volatility"] == pytest.approx(0.005)
        assert e["reasoning"]["predicted_runway"] == pytest.approx(200.0)

    def test_log_trigger_decision_defaults(self, logger):
        logger.log_trigger_decision(
            decision="NO_ENTRY", confidence=0.3,
            price=99000.0, volatility=0.01, imbalance=0.0,
            vpin_z=0.0, regime="UNKNOWN",
        )
        entries = self._read_entries(logger)
        e = entries[0]
        assert e["reasoning"]["predicted_runway"] == pytest.approx(0.0)
        assert e["reasoning"]["feasibility"] == pytest.approx(1.0)

    def test_log_harvester_decision_full(self, logger):
        logger.log_harvester_decision(
            decision="CLOSE",
            confidence=0.92,
            price=100050.0,
            entry_price=100000.0,
            mfe=75.0,
            mae=20.0,
            ticks_held=150,
            unrealized_pnl=50.0,
            capture_ratio=0.67,
        )
        entries = self._read_entries(logger)
        assert len(entries) == 1
        e = entries[0]
        assert e["agent"] == "HarvesterAgent"
        assert e["decision"] == "CLOSE"
        assert e["context"]["entry_price"] == pytest.approx(100000.0)
        assert e["reasoning"]["ticks_held"] == 150
        assert e["reasoning"]["capture_ratio"] == pytest.approx(0.67)

    def test_log_harvester_decision_defaults(self, logger):
        logger.log_harvester_decision(
            decision="HOLD", confidence=0.6,
            price=100010.0, entry_price=100000.0,
            mfe=30.0, mae=10.0, ticks_held=50,
            unrealized_pnl=10.0,
        )
        entries = self._read_entries(logger)
        e = entries[0]
        assert e["reasoning"]["capture_ratio"] == pytest.approx(0.0)

    def test_log_decision_generic_no_reasoning(self, logger):
        logger.log_decision("TestAgent", "ACTION", 0.5, {"key": "val"})
        entries = self._read_entries(logger)
        assert entries[0]["reasoning"] == {}

    def test_concurrent_decision_writes(self, logger):
        def write_decisions(tid):
            for i in range(20):
                logger.log_decision("Agent", "ACT", 0.5, {"t": tid, "i": i})

        threads = [threading.Thread(target=write_decisions, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = self._read_entries(logger)
        assert len(entries) == 80

    def test_session_id_format(self, logger):
        assert logger.session_id.startswith("session_")

    def test_log_dir_creation(self, tmp_path):
        new_dir = tmp_path / "nested" / "dir"
        dl = DecisionLogger(log_dir=str(new_dir))
        assert new_dir.exists()
