"""Extended tests for src.monitoring.trade_audit_logger.

Covers: singleton pattern, log_orphaned_position, log_error,
log_reconciliation discrepancy severity, concurrent writes,
sequence monotonicity, ticket field in entries.
"""

import json
import threading
import pytest

from src.monitoring.trade_audit_logger import (
    TradeAuditLogger,
    get_trade_audit_logger,
)
import src.monitoring.trade_audit_logger as tal_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_entries(audit):
    with open(audit.log_file) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
class TestSingleton:
    def test_get_trade_audit_logger_returns_instance(self, tmp_path, monkeypatch):
        # Reset singleton
        monkeypatch.setattr(tal_module, "_audit_logger_instance", None)
        monkeypatch.setattr(
            tal_module, "TradeAuditLogger",
            lambda **kw: TradeAuditLogger(log_dir=str(tmp_path), **kw),
        )
        instance = get_trade_audit_logger()
        assert instance is not None

    def test_singleton_returns_same_instance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tal_module, "_audit_logger_instance", None)
        real = TradeAuditLogger(log_dir=str(tmp_path))
        monkeypatch.setattr(tal_module, "_audit_logger_instance", real)
        assert get_trade_audit_logger() is real


# ---------------------------------------------------------------------------
# Lifecycle methods not covered yet
# ---------------------------------------------------------------------------
class TestPositionLifecycle:
    @pytest.fixture()
    def audit(self, tmp_path):
        return TradeAuditLogger(log_dir=str(tmp_path), filename="test.jsonl")

    def test_log_order_cancel(self, audit):
        audit.log_order_cancel("ORD001", reason="Timeout", ticket="T100")
        entries = _read_entries(audit)
        cancel = [e for e in entries if e["event_type"] == "ORDER_CANCEL"]
        assert len(cancel) == 1
        assert cancel[0]["data"]["reason"] == "Timeout"
        assert cancel[0]["ticket"] == "T100"

    def test_log_position_close_with_bars(self, audit):
        audit.log_position_close(
            "POS001", 91900.0, 49.0, 75.0, 25.0, "T200",
            bars_held=34, close_reason="StopLoss"
        )
        entries = _read_entries(audit)
        close = [e for e in entries if e["event_type"] == "POSITION_CLOSE"]
        assert close[0]["data"]["bars_held"] == 34
        assert close[0]["data"]["close_reason"] == "StopLoss"
        assert close[0]["ticket"] == "T200"


# ---------------------------------------------------------------------------
# Ticket tracking
# ---------------------------------------------------------------------------
class TestTicketTracking:
    @pytest.fixture()
    def audit(self, tmp_path):
        return TradeAuditLogger(log_dir=str(tmp_path), filename="test.jsonl")

    def test_log_ticket_tracker_created(self, audit):
        audit.log_ticket_tracker_created("T300", "POS002", "LONG", 91851.0, 0.1)
        entries = _read_entries(audit)
        tracker = [e for e in entries if e["event_type"] == "TRACKER_CREATED"]
        assert len(tracker) == 1
        assert tracker[0]["data"]["direction"] == "LONG"
        assert tracker[0]["ticket"] == "T300"

    def test_log_ticket_tracker_removed(self, audit):
        audit.log_ticket_tracker_removed("T300", "POS002", 75.0, 25.0, 34)
        entries = _read_entries(audit)
        removed = [e for e in entries if e["event_type"] == "TRACKER_REMOVED"]
        assert removed[0]["data"]["final_mfe"] == pytest.approx(75.0)
        assert removed[0]["data"]["bars_held"] == 34


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
class TestStatePersistence:
    @pytest.fixture()
    def audit(self, tmp_path):
        return TradeAuditLogger(log_dir=str(tmp_path), filename="test.jsonl")

    def test_log_state_load_invalid_checksum_severity(self, audit):
        audit.log_state_load("file.json", 3, 0.3, checksum_valid=False)
        entries = _read_entries(audit)
        load = [e for e in entries if e["event_type"] == "STATE_LOAD"]
        assert load[0]["severity"] == "WARNING"

    def test_log_state_load_valid_checksum_severity(self, audit):
        audit.log_state_load("file.json", 3, 0.3, checksum_valid=True)
        entries = _read_entries(audit)
        load = [e for e in entries if e["event_type"] == "STATE_LOAD"]
        assert load[0]["severity"] == "INFO"


# ---------------------------------------------------------------------------
# Reconciliation & errors
# ---------------------------------------------------------------------------
class TestReconciliationAndErrors:
    @pytest.fixture()
    def audit(self, tmp_path):
        return TradeAuditLogger(log_dir=str(tmp_path), filename="test.jsonl")

    def test_log_reconciliation_discrepancy_severity(self, audit):
        audit.log_reconciliation(3, 2, ["missing T100"], reconciled=False)
        entries = _read_entries(audit)
        recon = [e for e in entries if e["event_type"] == "RECONCILIATION"]
        assert recon[0]["severity"] == "WARNING"
        assert recon[0]["data"]["discrepancies"] == ["missing T100"]

    def test_log_reconciliation_ok_severity(self, audit):
        audit.log_reconciliation(3, 3, [], reconciled=True)
        entries = _read_entries(audit)
        recon = [e for e in entries if e["event_type"] == "RECONCILIATION"]
        assert recon[0]["severity"] == "INFO"

    def test_log_orphaned_position(self, audit):
        audit.log_orphaned_position("T500", 0.1, "closed")
        entries = _read_entries(audit)
        orphan = [e for e in entries if e["event_type"] == "ORPHANED_POSITION"]
        assert len(orphan) == 1
        assert orphan[0]["severity"] == "WARNING"
        assert orphan[0]["ticket"] == "T500"
        assert orphan[0]["data"]["action_taken"] == "closed"

    def test_log_error(self, audit):
        audit.log_error("TIMEOUT", "No response", {"order": "ORD1"}, ticket="T600")
        entries = _read_entries(audit)
        errors = [e for e in entries if e["event_type"] == "ERROR"]
        assert errors[0]["severity"] == "ERROR"
        assert errors[0]["ticket"] == "T600"
        assert errors[0]["data"]["error_type"] == "TIMEOUT"

    def test_log_error_no_context(self, audit):
        audit.log_error("UNKNOWN", "Something happened")
        entries = _read_entries(audit)
        errors = [e for e in entries if e["event_type"] == "ERROR"]
        assert errors[0]["data"]["context"] == {}


# ---------------------------------------------------------------------------
# Sequence & threading
# ---------------------------------------------------------------------------
class TestSequenceAndThreading:
    @pytest.fixture()
    def audit(self, tmp_path):
        return TradeAuditLogger(log_dir=str(tmp_path), filename="test.jsonl")

    def test_sequence_monotonically_increases(self, audit):
        for i in range(10):
            audit.log_order_submit(f"ORD{i}", "BUY", 0.1)
        entries = _read_entries(audit)
        sequences = [e["sequence"] for e in entries]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)

    def test_concurrent_writes_no_corruption(self, audit):
        def write_events(tid):
            for i in range(20):
                audit.log_order_submit(f"ORD_{tid}_{i}", "BUY", 0.1)

        threads = [threading.Thread(target=write_events, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = _read_entries(audit)
        # 1 session start + 80 order submits
        submits = [e for e in entries if e["event_type"] == "ORDER_SUBMIT"]
        assert len(submits) == 80

    def test_ticket_field_absent_when_not_provided(self, audit):
        audit.log_order_submit("ORD001", "BUY", 0.1)
        entries = _read_entries(audit)
        submit = [e for e in entries if e["event_type"] == "ORDER_SUBMIT"][0]
        assert "ticket" not in submit

    def test_ticket_field_present_when_provided(self, audit):
        audit.log_order_submit("ORD001", "BUY", 0.1, ticket="T100")
        entries = _read_entries(audit)
        submit = [e for e in entries if e["event_type"] == "ORDER_SUBMIT"][0]
        assert submit["ticket"] == "T100"

    def test_session_id_in_all_entries(self, audit):
        audit.log_order_submit("ORD001", "BUY", 0.1)
        entries = _read_entries(audit)
        for e in entries:
            assert e["session"] == audit.session_id
