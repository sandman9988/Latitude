"""
Tests for audit logger write error paths.

Covers:
- DecisionLogger.log_decision error handler (audit_logger.py lines 230-231)
- TradeAuditLogger._write_entry error handler (trade_audit_logger.py lines 113-114)
"""

import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from src.monitoring.audit_logger import DecisionLogger
from src.monitoring.trade_audit_logger import TradeAuditLogger, get_trade_audit_logger


# ===========================================================================
# DecisionLogger write error (audit_logger.py lines 230-231)
# ===========================================================================

class TestDecisionLoggerWriteError:
    """Test that DecisionLogger handles write failures gracefully."""

    @pytest.fixture()
    def logger(self, tmp_path):
        return DecisionLogger(log_dir=str(tmp_path), filename="dec.jsonl")

    def test_log_decision_write_error_handled(self, tmp_path):
        """If the log file write fails, log_decision should not raise."""
        logger = DecisionLogger.__new__(DecisionLogger)
        logger.log_dir = tmp_path
        logger.log_file = Path("/dev/null/impossible/path/decision.jsonl")
        logger.lock = threading.Lock()
        logger.session_id = "test_session"

        # Should not raise, error is caught and logged
        logger.log_decision(
            agent="TriggerAgent",
            decision="BUY",
            confidence=0.85,
            context={"price": 91000.0},
            reasoning={"signal": "strong"},
        )

    def test_log_decision_write_error_with_readonly_dir(self, tmp_path):
        """Test write failure with a read-only directory."""
        log_dir = tmp_path / "readonly"
        log_dir.mkdir()
        log_file = log_dir / "dec.jsonl"

        logger = DecisionLogger.__new__(DecisionLogger)
        logger.log_dir = log_dir
        logger.log_file = log_file
        logger.lock = threading.Lock()
        logger.session_id = "test_session"

        # Make directory read-only
        os.chmod(log_dir, 0o444)
        try:
            # Should not raise
            logger.log_decision(
                agent="HarvesterAgent",
                decision="HOLD",
                confidence=0.5,
                context={},
            )
        finally:
            os.chmod(log_dir, 0o755)

    def test_log_decision_write_error_via_mock(self, logger):
        """Mock open to raise an IOError during log_decision."""
        with patch("builtins.open", side_effect=IOError("disk full")):
            # Should not raise
            logger.log_decision(
                agent="TriggerAgent",
                decision="SELL",
                confidence=0.9,
                context={"price": 90000.0},
            )

    def test_log_harvester_decision_calls_log_decision(self, logger):
        """Ensure log_harvester_decision exercises the full code path."""
        logger.log_harvester_decision(
            decision="EXIT",
            confidence=0.88,
            price=92000.0,
            entry_price=91500.0,
            mfe=600.0,
            mae=150.0,
            ticks_held=42,
            unrealized_pnl=500.0,
            capture_ratio=0.83,
        )
        with open(logger.log_file) as f:
            entries = [json.loads(line) for line in f]
        harvester_entries = [e for e in entries if e.get("agent") == "HarvesterAgent"]
        assert len(harvester_entries) == 1
        entry = harvester_entries[0]
        assert entry["decision"] == "EXIT"
        assert entry["confidence"] == pytest.approx(0.88)
        assert entry["reasoning"]["capture_ratio"] == pytest.approx(0.83)

    def test_log_decision_none_reasoning_defaults(self, logger):
        """Test that reasoning=None is replaced with empty dict."""
        logger.log_decision(
            agent="TestAgent",
            decision="HOLD",
            confidence=0.5,
            context={"price": 100.0},
            reasoning=None,
        )
        with open(logger.log_file) as f:
            entries = [json.loads(line) for line in f]
        assert entries[-1]["reasoning"] == {}


# ===========================================================================
# TradeAuditLogger write error (trade_audit_logger.py lines 113-114)
# ===========================================================================

class TestTradeAuditLoggerWriteError:
    """Test that TradeAuditLogger handles write failures gracefully."""

    def test_write_entry_error_handled(self, tmp_path):
        """If the log file write fails, _write_entry should not raise."""
        logger = TradeAuditLogger.__new__(TradeAuditLogger)
        logger.log_dir = tmp_path
        logger.log_file = Path("/dev/null/impossible/path/audit.jsonl")
        logger.lock = threading.Lock()
        logger.session_id = "test_session"
        logger.sequence = 0

        # Should not raise
        logger._write_entry("TEST_EVENT", {"key": "value"}, "INFO")

    def test_write_entry_readonly_file(self, tmp_path):
        """Test write failure when file is in a read-only directory."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()

        logger = TradeAuditLogger.__new__(TradeAuditLogger)
        logger.log_dir = readonly_dir
        logger.log_file = readonly_dir / "audit.jsonl"
        logger.lock = threading.Lock()
        logger.session_id = "test_session"
        logger.sequence = 0

        # Make directory read-only
        os.chmod(readonly_dir, 0o444)
        try:
            # Should not raise
            logger._write_entry("TEST", {"x": 1}, "INFO")
        finally:
            os.chmod(readonly_dir, 0o755)

    def test_write_entry_open_raises_oserror(self, tmp_path):
        """Mock open to raise OSError."""
        logger = TradeAuditLogger(log_dir=str(tmp_path), filename="test.jsonl")
        with patch("builtins.open", side_effect=OSError("disk full")):
            # Should not raise
            logger._write_entry("DISK_FULL_TEST", {"data": 123}, "ERROR")

    def test_write_entry_fsync_raises(self, tmp_path):
        """Test when fsync fails (rare but possible)."""
        logger = TradeAuditLogger(log_dir=str(tmp_path), filename="test.jsonl")
        with patch("os.fsync", side_effect=OSError("fsync failed")):
            # Should not raise
            logger._write_entry("FSYNC_FAIL", {"data": 1}, "WARNING")

    def test_write_entry_with_ticket(self, tmp_path):
        """Verify ticket is included in entry when provided."""
        logger = TradeAuditLogger(log_dir=str(tmp_path), filename="test.jsonl")
        logger.log_order_submit("ORD1", "BUY", 0.1, 91000.0, ticket="T123")
        with open(logger.log_file) as f:
            entries = [json.loads(line) for line in f]
        # Find the ORDER_SUBMIT entry (session start is first)
        submit = [e for e in entries if e["event_type"] == "ORDER_SUBMIT"]
        assert len(submit) == 1
        assert submit[0]["ticket"] == "T123"

    def test_write_entry_without_ticket(self, tmp_path):
        """Verify ticket is omitted from entry when not provided."""
        logger = TradeAuditLogger(log_dir=str(tmp_path), filename="test.jsonl")
        logger.log_order_reject("ORD2", "Insufficient margin")
        with open(logger.log_file) as f:
            entries = [json.loads(line) for line in f]
        reject = [e for e in entries if e["event_type"] == "ORDER_REJECT"]
        assert len(reject) == 1
        assert "ticket" not in reject[0]

    def test_full_lifecycle_methods(self, tmp_path):
        """Exercise remaining TradeAuditLogger methods for coverage."""
        logger = TradeAuditLogger(log_dir=str(tmp_path), filename="test.jsonl")

        logger.log_order_cancel("ORD1", reason="Timeout")
        logger.log_position_update("POS1", 0.2, 91100.0, "T456", "PartialFill")
        logger.log_state_load("state.json", 3, 0.3, checksum_valid=True)
        logger.log_state_load("state.json", 2, 0.1, checksum_valid=False)
        logger.log_reconciliation(3, 3, [], reconciled=True)
        logger.log_reconciliation(3, 2, ["missing T789"], reconciled=False)
        logger.log_orphaned_position("T999", 0.05, "closed")
        logger.log_error("TIMEOUT", "Order timed out", {"order": "ORD5"}, ticket="T100")

        with open(logger.log_file) as f:
            entries = [json.loads(line) for line in f]

        types = [e["event_type"] for e in entries]
        assert "ORDER_CANCEL" in types
        assert "POSITION_UPDATE" in types
        assert "STATE_LOAD" in types
        assert "RECONCILIATION" in types
        assert "ORPHANED_POSITION" in types
        assert "ERROR" in types


# ===========================================================================
# get_trade_audit_logger singleton
# ===========================================================================

class TestGetTradeAuditLoggerSingleton:
    """Test the singleton factory function."""

    def test_returns_instance(self):
        """get_trade_audit_logger should return a TradeAuditLogger."""
        import src.monitoring.trade_audit_logger as mod
        # Reset singleton for clean test
        old = mod._audit_logger_instance
        try:
            mod._audit_logger_instance = None
            instance = get_trade_audit_logger()
            assert isinstance(instance, TradeAuditLogger)
            # Second call returns same instance
            assert get_trade_audit_logger() is instance
        finally:
            mod._audit_logger_instance = old
