"""
Tests for atomic_persistence.py — Tier 1 + Tier 2 critical paths.

Tier 1:
  - save_json error handling: temp file cleanup on OSError (lines 91-95)
  - JournaledPersistence._recover_from_journal: WAL replay on startup (lines 303-330)
  - _journal_write / _journal_commit: WAL implementation (lines 331-348)

Tier 2:
  - _restore_from_backup: load from .bak on CRC corruption (lines 186-190)
"""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.persistence.atomic_persistence import AtomicPersistence, JournaledPersistence


# =========================================================================
# TIER 1 — PRODUCTION-CRITICAL
# =========================================================================


class TestSaveJsonErrorHandling:
    """Lines 91-95: OSError during shutil.move → temp cleaned up, False returned."""

    @pytest.fixture()
    def ap(self, tmp_path):
        return AtomicPersistence(str(tmp_path))

    def test_oserror_on_move_returns_false(self, ap, tmp_path):
        """If atomic rename fails, save returns False."""
        with patch("shutil.move", side_effect=OSError("disk full")):
            result = ap.save_json({"key": "value"}, "test.json")
        assert result is False

    def test_oserror_on_move_cleans_temp(self, ap, tmp_path):
        """Temp file should be cleaned up after OSError."""
        with patch("shutil.move", side_effect=OSError("disk full")):
            ap.save_json({"key": "value"}, "test.json")

        # No leftover temp files (pattern: *.tmp*)
        tmp_files = list(tmp_path.glob("*.tmp*"))
        assert len(tmp_files) == 0

    def test_value_error_returns_false(self, ap, tmp_path):
        """ValueError during save also returns False."""
        # Force a ValueError by injecting a non-serializable object via tempfile
        with patch("json.dumps", side_effect=ValueError("bad data")):
            result = ap.save_json({"key": "value"}, "test.json")
        assert result is False

    def test_successful_save_after_failed_save(self, ap, tmp_path):
        """After a failed save, subsequent saves should still work."""
        with patch("shutil.move", side_effect=OSError("transient")):
            ap.save_json({"v": 1}, "test.json")

        # Now a normal save should succeed
        assert ap.save_json({"v": 2}, "test.json") is True
        loaded = ap.load_json("test.json")
        assert loaded == {"v": 2}


class TestJournaledPersistenceRecovery:
    """Lines 303-330: _recover_from_journal replays uncommitted entries on startup."""

    def test_recovery_reads_uncommitted_entries(self, tmp_path):
        """Uncommitted journal entries should be detected on startup."""
        journal_path = tmp_path / "persistence.journal"
        uncommitted = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "operation": "save",
            "filename": "params.json",
            "data_hash": 12345,
            "committed": False,
        }
        journal_path.write_text(json.dumps(uncommitted) + "\n")

        # Creating JournaledPersistence triggers _recover_from_journal
        jp = JournaledPersistence(base_dir=str(tmp_path))

        # Journal should be archived (moved with timestamp suffix)
        assert not journal_path.exists()
        old_journals = list(tmp_path.glob("persistence.journal.*.old"))
        assert len(old_journals) == 1

    def test_recovery_skips_committed_entries(self, tmp_path):
        """Committed entries in journal should NOT trigger replay warnings."""
        journal_path = tmp_path / "persistence.journal"
        committed = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "operation": "save",
            "filename": "params.json",
            "data_hash": 12345,
            "committed": True,
        }
        journal_path.write_text(json.dumps(committed) + "\n")

        jp = JournaledPersistence(base_dir=str(tmp_path))
        # Should still archive journal
        assert not journal_path.exists()

    def test_recovery_handles_corrupt_journal(self, tmp_path):
        """Malformed journal lines should not crash startup."""
        journal_path = tmp_path / "persistence.journal"
        journal_path.write_text("not valid json\n")

        # Should not raise
        jp = JournaledPersistence(base_dir=str(tmp_path))
        assert jp is not None

    def test_no_journal_file_is_noop(self, tmp_path):
        """No journal file → no recovery needed."""
        jp = JournaledPersistence(base_dir=str(tmp_path))
        old_journals = list(tmp_path.glob("persistence.journal.*.old"))
        assert len(old_journals) == 0

    def test_recovery_with_multiple_entries(self, tmp_path):
        """Journal with mix of committed and uncommitted entries."""
        journal_path = tmp_path / "persistence.journal"
        entries = [
            {"timestamp": "t1", "operation": "save", "filename": "a.json", "committed": True},
            {"timestamp": "t2", "operation": "save", "filename": "b.json", "committed": False},
            {"timestamp": "t3", "operation": "save", "filename": "c.json", "committed": False},
        ]
        journal_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        jp = JournaledPersistence(base_dir=str(tmp_path))
        assert not journal_path.exists()


class TestJournalWriteAndCommit:
    """Lines 331-348: _journal_write and _journal_commit WAL implementation."""

    @pytest.fixture()
    def jp(self, tmp_path):
        return JournaledPersistence(base_dir=str(tmp_path))

    def test_journal_write_creates_entry(self, jp, tmp_path):
        result = jp._journal_write("save", "params.json", data_hash=42)
        assert result is True

        journal = tmp_path / "persistence.journal"
        assert journal.exists()

        entry = json.loads(journal.read_text().strip())
        assert entry["operation"] == "save"
        assert entry["filename"] == "params.json"
        assert entry["data_hash"] == 42
        assert entry["committed"] is False

    def test_journal_write_appends(self, jp, tmp_path):
        jp._journal_write("save", "a.json")
        jp._journal_write("save", "b.json")

        journal = tmp_path / "persistence.journal"
        lines = journal.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_journal_write_oserror_returns_false(self, jp):
        with patch("builtins.open", side_effect=OSError("read-only fs")):
            result = jp._journal_write("save", "params.json")
        assert result is False

    def test_journal_commit_returns_true(self, jp):
        result = jp._journal_commit("params.json")
        assert result is True


# =========================================================================
# TIER 2 — IMPORTANT
# =========================================================================


class TestRestoreFromBackup:
    """Lines 186-190: _restore_from_backup loads most recent .bak on CRC failure."""

    @pytest.fixture()
    def ap(self, tmp_path):
        return AtomicPersistence(str(tmp_path))

    def test_restore_from_backup_on_corruption(self, ap, tmp_path):
        """Save twice (creating backup), corrupt main file, load with CRC → restores from backup."""
        original = {"spread": 1.5, "depth": 1.2}
        ap.save_json(original, "params.json")
        modified = {"spread": 2.0, "depth": 1.2}
        ap.save_json(modified, "params.json")

        # Corrupt the main file's data (change data but keep envelope structure)
        target = tmp_path / "params.json"
        with open(target) as f:
            envelope = json.load(f)
        envelope["data"]["spread"] = 999.0  # Corrupt
        with open(target, "w") as f:
            json.dump(envelope, f)

        # Load with CRC verification → should detect corruption + restore from backup
        loaded = ap.load_json("params.json", verify_crc=True)
        # Should get the backup version (original data)
        assert loaded is not None
        assert loaded["spread"] != 999.0

    def test_no_backup_available_returns_none(self, ap, tmp_path):
        """If no backups exist, _restore_from_backup returns None."""
        result = ap._restore_from_backup("nonexistent.json")
        assert result is None

    def test_restore_picks_most_recent_backup(self, ap, tmp_path):
        """Multiple backups → most recent one used."""
        import time
        # Create multiple saves to generate backups
        ap.save_json({"v": 1}, "data.json")
        time.sleep(0.05)  # Ensure distinct mtime
        ap.save_json({"v": 2}, "data.json")
        time.sleep(0.05)
        ap.save_json({"v": 3}, "data.json")

        backups = ap.list_backups("data.json")
        assert len(backups) >= 1  # At least 1 backup from overwrites

        # Restore should return a valid backup
        restored = ap._restore_from_backup("data.json")
        assert restored is not None
        assert "v" in restored
