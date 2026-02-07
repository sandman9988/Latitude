"""Tests for src.monitoring.trade_audit_logger – TradeAuditLogger & singleton."""

import json
import threading
from pathlib import Path

import pytest

from src.monitoring.trade_audit_logger import TradeAuditLogger


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_entries(log_file: Path):
    with open(log_file) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# TradeAuditLogger
# ---------------------------------------------------------------------------

class TestTradeAuditLogger:
    @pytest.fixture()
    def audit(self, tmp_path):
        return TradeAuditLogger(log_dir=str(tmp_path), filename="audit.jsonl")

    # -- init --
    def test_session_start_logged(self, audit):
        entries = _read_entries(audit.log_file)
        assert entries[0]["event_type"] == "SESSION_START"
        assert entries[0]["sequence"] == 1

    def test_session_id_format(self, audit):
        assert audit.session_id.startswith("session_")

    def test_sequence_increments(self, audit):
        audit.log_order_submit("O1", "BUY", 0.1)
        audit.log_order_submit("O2", "SELL", 0.2)
        entries = _read_entries(audit.log_file)
        seqs = [e["sequence"] for e in entries]
        assert seqs == [1, 2, 3]  # SESSION_START=1, O1=2, O2=3

    # -- order lifecycle --
    def test_log_order_submit(self, audit):
        audit.log_order_submit("ORD1", "BUY", 0.1, 91850.0, ticket="T1", order_type="LIMIT")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "ORDER_SUBMIT"
        assert last["data"]["side"] == "BUY"
        assert last["data"]["quantity"] == pytest.approx(0.1)
        assert last["data"]["order_type"] == "LIMIT"

    def test_log_order_submit_with_ticket(self, audit):
        audit.log_order_submit("ORD1", "BUY", 0.1, ticket="TICKET123")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["ticket"] == "TICKET123"

    def test_log_order_accept(self, audit):
        audit.log_order_accept("ORD1", broker_order_id="BRK1", ticket="T1")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "ORDER_ACCEPT"
        assert last["ticket"] == "T1"

    def test_log_order_reject(self, audit):
        audit.log_order_reject("ORD1", "Insufficient margin", reject_code="103")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "ORDER_REJECT"
        assert last["severity"] == "WARNING"

    def test_log_order_fill(self, audit):
        audit.log_order_fill("ORD1", 91851.0, 0.1, ticket="T1", fill_id="F1")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "ORDER_FILL"
        assert last["data"]["fill_price"] == pytest.approx(91851.0)

    def test_log_order_fill_partial(self, audit):
        audit.log_order_fill("ORD1", 91851.0, 0.05, ticket="T1", is_partial=True)
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["data"]["is_partial"] is True

    def test_log_order_cancel(self, audit):
        audit.log_order_cancel("ORD1", reason="timeout", ticket="T1")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "ORDER_CANCEL"

    # -- position lifecycle --
    def test_log_position_open(self, audit):
        audit.log_position_open("POS1", "LONG", 0.1, 91851.0, ticket="T1")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "POSITION_OPEN"
        assert last["data"]["direction"] == "LONG"

    def test_log_position_update(self, audit):
        audit.log_position_update("POS1", 0.2, 91900.0, ticket="T1")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "POSITION_UPDATE"

    def test_log_position_close(self, audit):
        audit.log_position_close("POS1", 92000.0, 49.0, 75.0, 25.0, "T1", bars_held=34)
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "POSITION_CLOSE"
        assert last["data"]["pnl"] == pytest.approx(49.0)
        assert last["data"]["bars_held"] == 34

    # -- ticket tracking --
    def test_log_ticket_assigned(self, audit):
        audit.log_ticket_assigned("T1", "POS1", order_id="ORD1")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "TICKET_ASSIGNED"
        assert last["ticket"] == "T1"

    def test_log_ticket_tracker_created(self, audit):
        audit.log_ticket_tracker_created("T1", "POS1", "LONG", 91851.0, 0.1)
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "TRACKER_CREATED"

    def test_log_ticket_tracker_removed(self, audit):
        audit.log_ticket_tracker_removed("T1", "POS1", 75.0, 25.0, 34)
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "TRACKER_REMOVED"

    # -- state persistence --
    def test_log_state_save(self, audit):
        audit.log_state_save("state.json", 3, 0.3, checksum="ABC")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "STATE_SAVE"
        assert last["data"]["checksum"] == "ABC"

    def test_log_state_load_valid(self, audit):
        audit.log_state_load("state.json", 3, 0.3, checksum_valid=True)
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "STATE_LOAD"
        assert last["severity"] == "INFO"

    def test_log_state_load_invalid_checksum(self, audit):
        audit.log_state_load("state.json", 3, 0.3, checksum_valid=False)
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["severity"] == "WARNING"

    # -- reconciliation & errors --
    def test_log_reconciliation_ok(self, audit):
        audit.log_reconciliation(3, 3, [], reconciled=True)
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "RECONCILIATION"
        assert last["severity"] == "INFO"

    def test_log_reconciliation_discrepancy(self, audit):
        audit.log_reconciliation(3, 2, ["Missing POS3"], reconciled=False)
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["severity"] == "WARNING"

    def test_log_orphaned_position(self, audit):
        audit.log_orphaned_position("T99", 0.1, "closed")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "ORPHANED_POSITION"
        assert last["severity"] == "WARNING"
        assert last["ticket"] == "T99"

    def test_log_error(self, audit):
        audit.log_error("FILL_TIMEOUT", "No fill after 30s", {"order_id": "O2"}, ticket="T2")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert last["event_type"] == "ERROR"
        assert last["severity"] == "ERROR"
        assert last["ticket"] == "T2"

    # -- no ticket omits field --
    def test_no_ticket_field_when_none(self, audit):
        audit.log_order_reject("ORD1", "bad")
        entries = _read_entries(audit.log_file)
        last = entries[-1]
        assert "ticket" not in last

    # -- thread safety --
    def test_concurrent_writes(self, audit):
        errors = []

        def write_batch(start):
            try:
                for i in range(20):
                    audit.log_order_submit(f"ORD_{start}_{i}", "BUY", 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_batch, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        entries = _read_entries(audit.log_file)
        # SESSION_START + 4*20 = 81
        assert len(entries) == 81
