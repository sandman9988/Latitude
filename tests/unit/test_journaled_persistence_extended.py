"""Extended tests for src.persistence.journaled_persistence.

Covers: _get_last_sequence empty last line, log_trade_open with shape attr,
checkpoint exception, _should_rotate exception, _rotate_journal exception,
replay exception, close exception, context manager __exit__.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.persistence.journaled_persistence import Journal, JournalEntry


# ---------------------------------------------------------------------------
# _get_last_sequence edge cases
# ---------------------------------------------------------------------------
class TestGetLastSequence:
    def test_empty_last_line(self, tmp_path):
        """Journal file with blank trailing line returns 0 sequence."""
        journal_file = tmp_path / "journal.log"
        # Empty last line → _get_last_sequence reads last line which is blank
        journal_file.write_text('{"seq": 5, "ts": "", "op": "x", "data": {}}\n')
        j = Journal(str(journal_file))
        # Last non-empty line has seq=5, so next should be 6
        assert j.sequence_num == 6
        j.close()


# ---------------------------------------------------------------------------
# log_trade_open with numpy-like state
# ---------------------------------------------------------------------------
class TestLogTradeOpen:
    def test_with_shape_attribute(self, tmp_path):
        journal_file = tmp_path / "j.log"
        j = Journal(str(journal_file))
        state = MagicMock()
        state.shape = (1, 64)
        seq = j.log_trade_open("ORD1", "BUY", 0.01, 50000.0, entry_state=state)
        assert seq >= 1

        # Verify written data has shape
        with open(journal_file) as f:
            lines = f.readlines()
        last = json.loads(lines[-1].strip())
        assert last["data"]["entry_state_shape"] == "(1, 64)"
        j.close()

    def test_without_shape_attribute(self, tmp_path):
        journal_file = tmp_path / "j.log"
        j = Journal(str(journal_file))
        seq = j.log_trade_open("ORD2", "SELL", 0.02, 51000.0, entry_state="no_shape")
        assert seq >= 1
        with open(journal_file) as f:
            lines = f.readlines()
        last = json.loads(lines[-1].strip())
        assert last["data"]["entry_state_shape"] is None
        j.close()


# ---------------------------------------------------------------------------
# checkpoint exception path
# ---------------------------------------------------------------------------
class TestCheckpointErrors:
    def test_checkpoint_exception_returns_false(self, tmp_path):
        journal_file = tmp_path / "j.log"
        j = Journal(str(journal_file))
        # Patch open to fail on checkpoint write
        original_open = open

        def patched_open(path, *args, **kwargs):
            if ".tmp" in str(path):
                raise OSError("disk full")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=patched_open):
            result = j.checkpoint()
        assert result is False
        j.journal_file = original_open(journal_file, "a", buffering=1, encoding="utf-8")
        j.close()


# ---------------------------------------------------------------------------
# _should_rotate
# ---------------------------------------------------------------------------
class TestShouldRotate:
    def test_small_file_does_not_rotate(self, tmp_path):
        journal_file = tmp_path / "j.log"
        j = Journal(str(journal_file))
        j.log_operation("test", {"x": 1})
        assert j._should_rotate() is False
        j.close()

    def test_stat_exception_returns_false(self, tmp_path):
        journal_file = tmp_path / "j.log"
        j = Journal(str(journal_file))
        with patch.object(Path, "stat", side_effect=OSError("gone")):
            assert j._should_rotate() is False
        j.close()


# ---------------------------------------------------------------------------
# _rotate_journal exception
# ---------------------------------------------------------------------------
class TestRotateJournal:
    def test_rotation_exception_handled(self, tmp_path):
        journal_file = tmp_path / "j.log"
        j = Journal(str(journal_file))
        j.log_operation("test", {"x": 1})
        with patch.object(Path, "rename", side_effect=OSError("rename failed")):
            j._rotate_journal()  # Should not raise
        j.close()


# ---------------------------------------------------------------------------
# replay exception
# ---------------------------------------------------------------------------
class TestReplayErrors:
    def test_corrupt_checkpoint_skipped(self, tmp_path):
        journal_file = tmp_path / "j.log"
        cp_file = tmp_path / "j.checkpoint"
        cp_file.write_text("NOT JSON")
        j = Journal(str(journal_file))
        j.log_operation("op1", {"v": 1})
        j.journal_file.flush()
        j2 = Journal(str(journal_file))
        replayed = j2.replay_from_checkpoint()
        # Should still replay (checkpoint is corrupt so seq=0)
        assert len(replayed) >= 1
        j.close()
        j2.close()


# ---------------------------------------------------------------------------
# close exception
# ---------------------------------------------------------------------------
class TestCloseErrors:
    def test_close_exception_handled(self, tmp_path):
        journal_file = tmp_path / "j.log"
        j = Journal(str(journal_file))
        j.journal_file.close()  # Pre-close to trigger error
        j.close()  # Should not raise

    def test_context_manager_exit(self, tmp_path):
        journal_file = tmp_path / "j.log"
        with Journal(str(journal_file)) as j:
            j.log_operation("inner", {"val": 42})
        # After __exit__, file should be closed
        assert j.journal_file.closed


# ---------------------------------------------------------------------------
# get_recent_operations
# ---------------------------------------------------------------------------
class TestRecentOperations:
    def test_count_limits_results(self, tmp_path):
        journal_file = tmp_path / "j.log"
        j = Journal(str(journal_file))
        for i in range(10):
            j.log_operation(f"op{i}", {"i": i})
        recent = j.get_recent_operations(count=3)
        assert len(recent) == 3
        j.close()
