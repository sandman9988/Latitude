"""
Tests for src.persistence.journaled_persistence

Coverage targets:
- JournalEntry dataclass
- Journal: __init__, _get_last_sequence, log_operation, log_trade_open, log_trade_close,
  log_parameter_update, log_circuit_breaker_trip, log_model_update,
  checkpoint, _should_rotate, _rotate_journal, replay_from_checkpoint,
  get_recent_operations, close, context manager
"""

import json

import pytest

from src.persistence.journaled_persistence import Journal, JournalEntry


# ── JournalEntry ────────────────────────────────────────────────────────────

class TestJournalEntry:
    def test_create(self):
        entry = JournalEntry(seq=1, ts="2025-01-01T00:00:00Z", op="test", data={"k": "v"})
        assert entry.seq == 1
        assert entry.op == "test"
        assert entry.data == {"k": "v"}
        assert entry.checksum is None

    def test_create_with_checksum(self):
        entry = JournalEntry(seq=1, ts="t", op="op", data={}, checksum="abc123")
        assert entry.checksum == "abc123"


# ── Journal init ────────────────────────────────────────────────────────────

class TestJournalInit:
    def test_creates_directory(self, tmp_path):
        jp = tmp_path / "deep" / "nested" / "journal.log"
        j = Journal(str(jp))
        assert jp.parent.exists()
        j.close()

    def test_opens_append(self, tmp_path):
        jp = tmp_path / "journal.log"
        j = Journal(str(jp))
        assert not j.journal_file.closed
        j.close()

    def test_sequence_starts_at_one_for_new(self, tmp_path):
        jp = tmp_path / "journal.log"
        j = Journal(str(jp))
        assert j.sequence_num == 1
        j.close()

    def test_continues_sequence_from_existing(self, tmp_path):
        jp = tmp_path / "journal.log"
        # Write some entries
        j1 = Journal(str(jp), checkpoint_interval=1000)
        j1.log_operation("op1", {"i": 1})
        j1.log_operation("op2", {"i": 2})
        j1.close()
        # Reopen
        j2 = Journal(str(jp), checkpoint_interval=1000)
        assert j2.sequence_num == 3  # Last was 2, next is 3
        j2.close()


# ── _get_last_sequence ──────────────────────────────────────────────────────

class TestGetLastSequence:
    def test_empty_file(self, tmp_path):
        jp = tmp_path / "journal.log"
        jp.write_text("")
        j = Journal(str(jp))
        assert j.sequence_num == 1  # 0 + 1
        j.close()

    def test_corrupt_file(self, tmp_path):
        jp = tmp_path / "journal.log"
        jp.write_text("not json\n")
        j = Journal(str(jp))
        assert j.sequence_num == 1  # Falls back to 0 + 1
        j.close()

    def test_no_file(self, tmp_path):
        jp = tmp_path / "nonexistent.log"
        j = Journal(str(jp))
        assert j.sequence_num == 1
        j.close()


# ── log_operation ───────────────────────────────────────────────────────────

class TestLogOperation:
    def test_returns_sequence(self, tmp_path):
        j = Journal(str(tmp_path / "j.log"), checkpoint_interval=1000)
        seq = j.log_operation("test", {"a": 1})
        assert seq == 1
        seq2 = j.log_operation("test2", {"b": 2})
        assert seq2 == 2
        j.close()

    def test_writes_to_file(self, tmp_path):
        jp = tmp_path / "j.log"
        j = Journal(str(jp), checkpoint_interval=1000)
        j.log_operation("my_op", {"data": 42})
        j.journal_file.flush()
        lines = jp.read_text().strip().split("\n")
        entry = json.loads(lines[-1])
        assert entry["op"] == "my_op"
        assert entry["data"]["data"] == 42
        assert entry["seq"] == 1
        j.close()

    def test_tracks_in_memory(self, tmp_path):
        j = Journal(str(tmp_path / "j.log"), checkpoint_interval=1000)
        j.log_operation("op1", {})
        j.log_operation("op2", {})
        assert len(j.recent_operations) == 2
        j.close()

    def test_auto_checkpoint(self, tmp_path):
        jp = tmp_path / "j.log"
        cp = tmp_path / "j.checkpoint"
        j = Journal(str(jp), checkpoint_interval=3)
        for i in range(4):
            j.log_operation(f"op{i}", {"i": i})
        # Checkpoint should have been triggered
        assert cp.exists()
        j.close()


# ── Typed log helpers ──────────────────────────────────────────────────────

class TestTypedLogHelpers:
    @pytest.fixture
    def journal(self, tmp_path):
        j = Journal(str(tmp_path / "j.log"), checkpoint_interval=1000)
        yield j
        j.close()

    def test_log_trade_open(self, journal):
        import numpy as np
        seq = journal.log_trade_open("ORD1", "BUY", 0.01, 2000.0, entry_state=np.zeros(5))
        assert seq >= 1
        entry = journal.recent_operations[-1]
        assert entry.op == "trade_open"
        assert entry.data["order_id"] == "ORD1"
        assert entry.data["side"] == "BUY"
        assert entry.data["entry_state_shape"] == "(5,)"

    def test_log_trade_open_no_state(self, journal):
        _seq = journal.log_trade_open("ORD2", "SELL", 0.02, 3000.0, entry_state=None)
        assert journal.recent_operations[-1].data["entry_state_shape"] is None

    def test_log_trade_close(self, journal):
        _seq = journal.log_trade_close("ORD1", 2010.0, 10.0, 15.0, -5.0, False)
        entry = journal.recent_operations[-1]
        assert entry.op == "trade_close"
        assert entry.data["pnl"] == pytest.approx(10.0)
        assert entry.data["winner_to_loser"] is False

    def test_log_parameter_update(self, journal):
        journal.log_parameter_update("stop_loss", 0.002, 0.003)
        entry = journal.recent_operations[-1]
        assert entry.op == "parameter_update"
        assert entry.data["param_name"] == "stop_loss"

    def test_log_circuit_breaker_trip(self, journal):
        journal.log_circuit_breaker_trip("sortino", 0.5, 0.3)
        entry = journal.recent_operations[-1]
        assert entry.op == "circuit_breaker_trip"

    def test_log_model_update(self, journal):
        journal.log_model_update("trigger", 0.05, 0.01)
        entry = journal.recent_operations[-1]
        assert entry.op == "model_update"
        assert entry.data["agent"] == "trigger"


# ── Checkpoint ──────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_checkpoint_creates_file(self, tmp_path):
        j = Journal(str(tmp_path / "j.log"), checkpoint_interval=1000)
        j.log_operation("op", {})
        result = j.checkpoint()
        assert result is True
        assert (tmp_path / "j.checkpoint").exists()
        j.close()

    def test_checkpoint_contents(self, tmp_path):
        j = Journal(str(tmp_path / "j.log"), checkpoint_interval=1000)
        j.log_operation("op1", {})
        j.log_operation("op2", {})
        j.checkpoint()
        data = json.loads((tmp_path / "j.checkpoint").read_text())
        assert "seq" in data
        assert "ts" in data
        assert data["seq"] == j.sequence_num
        j.close()

    def test_checkpoint_resets_counter(self, tmp_path):
        j = Journal(str(tmp_path / "j.log"), checkpoint_interval=1000)
        j.log_operation("op", {})
        assert j.operations_since_checkpoint == 1
        j.checkpoint()
        assert j.operations_since_checkpoint == 0
        j.close()


# ── replay_from_checkpoint ──────────────────────────────────────────────────

class TestReplay:
    def test_replay_all_ops(self, tmp_path):
        jp = tmp_path / "j.log"
        j = Journal(str(jp), checkpoint_interval=1000)
        for i in range(5):
            j.log_operation(f"op{i}", {"i": i})
        j.journal_file.close()  # Close without checkpoint
        # Remove checkpoint if close() created one
        cp = tmp_path / "j.checkpoint"
        cp.unlink(missing_ok=True)

        j2 = Journal(str(jp), checkpoint_interval=1000)
        replayed = j2.replay_from_checkpoint()
        assert len(replayed) >= 5
        j2.close()

    def test_replay_from_checkpoint_skips_earlier(self, tmp_path):
        jp = str(tmp_path / "j.log")
        j = Journal(jp, checkpoint_interval=1000)
        for i in range(5):
            j.log_operation(f"op{i}", {"i": i})
        j.checkpoint()
        for i in range(3):
            j.log_operation(f"late_op{i}", {"i": i})
        j.close()

        j2 = Journal(jp, checkpoint_interval=1000)
        replayed = j2.replay_from_checkpoint()
        # Should only replay ops after checkpoint
        ops = [e.op for e in replayed]
        assert all("late_op" in op for op in ops)
        j2.close()

    def test_replay_with_callback(self, tmp_path):
        jp = tmp_path / "j.log"
        j = Journal(str(jp), checkpoint_interval=1000)
        j.log_operation("op1", {})
        j.log_operation("op2", {})
        j.journal_file.close()
        (tmp_path / "j.checkpoint").unlink(missing_ok=True)

        j2 = Journal(str(jp), checkpoint_interval=1000)
        captured = []

        def callback(op, data):
            captured.append(op)
            return True

        j2.replay_from_checkpoint(callback)
        assert len(captured) >= 2
        j2.close()

    def test_replay_callback_stop(self, tmp_path):
        jp = tmp_path / "j.log"
        j = Journal(str(jp), checkpoint_interval=1000)
        for i in range(10):
            j.log_operation(f"op{i}", {})
        j.journal_file.close()
        (tmp_path / "j.checkpoint").unlink(missing_ok=True)

        j2 = Journal(str(jp), checkpoint_interval=1000)
        count = [0]

        def callback(op, data):
            count[0] += 1
            return count[0] < 3  # Stop after 3

        replayed = j2.replay_from_checkpoint(callback)
        assert len(replayed) == 3
        j2.close()

    def test_replay_no_journal(self, tmp_path):
        # Create journal, close it, delete the file
        jp = tmp_path / "j.log"
        j = Journal(str(jp), checkpoint_interval=1000)
        j.close()
        jp.unlink(missing_ok=True)
        j2 = Journal(str(jp), checkpoint_interval=1000)
        replayed = j2.replay_from_checkpoint()
        assert replayed == []
        j2.close()


# ── get_recent_operations ──────────────────────────────────────────────────

class TestGetRecentOperations:
    def test_returns_recent(self, tmp_path):
        j = Journal(str(tmp_path / "j.log"), checkpoint_interval=1000)
        for i in range(5):
            j.log_operation(f"op{i}", {})
        recent = j.get_recent_operations(3)
        assert len(recent) == 3
        assert recent[-1].op == "op4"
        j.close()

    def test_returns_all_when_fewer_than_count(self, tmp_path):
        j = Journal(str(tmp_path / "j.log"), checkpoint_interval=1000)
        j.log_operation("op0", {})
        recent = j.get_recent_operations(100)
        assert len(recent) == 1
        j.close()


# ── Context manager ────────────────────────────────────────────────────────

class TestContextManager:
    def test_with_statement(self, tmp_path):
        jp = str(tmp_path / "j.log")
        with Journal(jp, checkpoint_interval=1000) as j:
            j.log_operation("op", {"v": 1})
        # File should be closed
        assert j.journal_file.closed

    def test_checkpoint_on_close(self, tmp_path):
        jp = str(tmp_path / "j.log")
        with Journal(jp, checkpoint_interval=1000) as j:
            j.log_operation("op", {})
        assert (tmp_path / "j.checkpoint").exists()


# ── Journal rotation ───────────────────────────────────────────────────────

class TestJournalRotation:
    def test_should_rotate_false_for_small(self, tmp_path):
        j = Journal(str(tmp_path / "j.log"), max_journal_size_mb=100)
        j.log_operation("op", {})
        assert j._should_rotate() is False
        j.close()

    def test_rotate_renames_and_creates_new(self, tmp_path):
        jp = tmp_path / "j.log"
        j = Journal(str(jp), checkpoint_interval=1000, max_journal_size_mb=100)
        j.log_operation("op1", {})
        j._rotate_journal()
        # Old journal should be renamed, new one created
        assert jp.exists()  # New journal exists
        # Should be an archive file too
        archives = list(tmp_path.glob("j.*.log"))
        assert len(archives) >= 1
        j.close()
