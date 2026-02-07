"""Extended tests for src.persistence.atomic_persistence.

Covers: save_json OSError path, _create_backup exception, _cleanup_old_backups
exception, _restore_from_backup OSError/no backups, load_json OSError,
JournaledPersistence subclass (recover, journal_write, journal_commit).
"""

import json
import os
import time
from unittest.mock import patch

from src.persistence.atomic_persistence import AtomicPersistence, JournaledPersistence


# ---------------------------------------------------------------------------
# save_json error paths
# ---------------------------------------------------------------------------
class TestSaveJsonErrors:
    def test_save_returns_false_on_write_error(self, tmp_path):
        ap = AtomicPersistence(str(tmp_path))
        # Make base_dir read-only so tempfile creation fails
        os.chmod(tmp_path, 0o444)
        try:
            result = ap.save_json({"key": "val"}, "test.json")
            assert result is False
        finally:
            os.chmod(tmp_path, 0o755)

    def test_save_backup_failure_proceeds(self, tmp_path):
        """If _create_backup fails, save still proceeds."""
        ap = AtomicPersistence(str(tmp_path))
        ap.save_json({"v": 1}, "data.json")
        with patch.object(ap, "_create_backup", return_value=False):
            result = ap.save_json({"v": 2}, "data.json")
        assert result is True
        loaded = ap.load_json("data.json")
        assert loaded["v"] == 2


# ---------------------------------------------------------------------------
# _create_backup & _cleanup_old_backups edge cases
# ---------------------------------------------------------------------------
class TestBackupEdgeCases:
    def test_create_backup_exception_returns_false(self, tmp_path):
        ap = AtomicPersistence(str(tmp_path))
        ap.save_json({"x": 1}, "f.json")
        with patch("shutil.copy2", side_effect=OSError("disk full")):
            result = ap._create_backup(tmp_path / "f.json")
        assert result is False

    def test_cleanup_exception_swallowed(self, tmp_path):
        ap = AtomicPersistence(str(tmp_path))
        ap.save_json({"x": 1}, "f.json")
        # Create several backups by saving multiple times
        for i in range(5):
            ap.save_json({"x": i}, "f.json")
            time.sleep(0.01)
        # Now patch unlink to raise; cleanup should swallow exception
        with patch("pathlib.Path.unlink", side_effect=OSError("perm denied")):
            ap._cleanup_old_backups(tmp_path / "f.json")
        # Should not raise


# ---------------------------------------------------------------------------
# _restore_from_backup edge cases
# ---------------------------------------------------------------------------
class TestRestoreFromBackup:
    def test_no_backups_returns_none(self, tmp_path):
        ap = AtomicPersistence(str(tmp_path))
        result = ap._restore_from_backup("nofile.json")
        assert result is None

    def test_restore_corrupt_backup_returns_none(self, tmp_path):
        ap = AtomicPersistence(str(tmp_path))
        # Create a corrupt backup file
        backup_name = "data.json.20260101_120000.bak"
        (tmp_path / backup_name).write_text("NOT VALID JSON {{{")
        result = ap._restore_from_backup("data.json")
        assert result is None

    def test_restore_legacy_backup_without_envelope(self, tmp_path):
        """Backup that is plain dict (no envelope wrapper)."""
        ap = AtomicPersistence(str(tmp_path))
        legacy_data = {"legacy_key": 42}
        backup_name = "data.json.20260101_120000.bak"
        with open(tmp_path / backup_name, "w") as f:
            json.dump(legacy_data, f)
        # Also create the target path so copy2 has a destination
        (tmp_path / "data.json").write_text("{}")
        result = ap._restore_from_backup("data.json")
        assert result == legacy_data


# ---------------------------------------------------------------------------
# load_json edge cases
# ---------------------------------------------------------------------------
class TestLoadJsonEdge:
    def test_load_os_error_returns_none(self, tmp_path):
        ap = AtomicPersistence(str(tmp_path))
        ap.save_json({"k": 1}, "test.json")
        with patch("builtins.open", side_effect=OSError("read error")):
            result = ap.load_json("test.json")
        assert result is None


# ---------------------------------------------------------------------------
# JournaledPersistence
# ---------------------------------------------------------------------------
class TestJournaledPersistence:
    def test_init_creates_journal_path(self, tmp_path):
        jp = JournaledPersistence(str(tmp_path))
        assert jp.journal_path == tmp_path / "persistence.journal"

    def test_recover_no_journal_noop(self, tmp_path):
        """No journal file = no recovery needed."""
        jp = JournaledPersistence(str(tmp_path))
        # Should be created without error
        assert jp.base_dir == tmp_path

    def test_recover_with_uncommitted_entries(self, tmp_path):
        """Journal with uncommitted entries gets archived."""
        journal_path = tmp_path / "persistence.journal"
        entry = {"committed": False, "op": "test_save", "filename": "f.json"}
        journal_path.write_text(json.dumps(entry) + "\n")

        _jp = JournaledPersistence(str(tmp_path))
        # Journal should have been archived (moved)
        assert not journal_path.exists()
        archives = list(tmp_path.glob("persistence.journal.*.old"))
        assert len(archives) == 1

    def test_recover_with_committed_entries(self, tmp_path):
        """Journal with committed entries also gets archived."""
        journal_path = tmp_path / "persistence.journal"
        entry = {"committed": True, "op": "test_save", "filename": "f.json"}
        journal_path.write_text(json.dumps(entry) + "\n")

        _jp = JournaledPersistence(str(tmp_path))
        assert not journal_path.exists()

    def test_recover_corrupt_journal(self, tmp_path):
        """Corrupt journal doesn't crash init."""
        journal_path = tmp_path / "persistence.journal"
        journal_path.write_text("NOT JSON AT ALL!!!\n")
        jp = JournaledPersistence(str(tmp_path))
        assert jp is not None

    def test_journal_write_creates_entry(self, tmp_path):
        jp = JournaledPersistence(str(tmp_path))
        result = jp._journal_write("save", "params.json", data_hash=12345)
        assert result is True
        assert jp.journal_path.exists()
        with open(jp.journal_path) as f:
            entry = json.loads(f.readline())
        assert entry["operation"] == "save"
        assert entry["filename"] == "params.json"
        assert entry["committed"] is False

    def test_journal_write_failure(self, tmp_path):
        jp = JournaledPersistence(str(tmp_path))
        with patch("builtins.open", side_effect=OSError("disk full")):
            result = jp._journal_write("save", "f.json")
        assert result is False

    def test_journal_commit(self, tmp_path):
        jp = JournaledPersistence(str(tmp_path))
        result = jp._journal_commit("params.json")
        assert result is True

    def test_inherits_save_load(self, tmp_path):
        """JournaledPersistence inherits AtomicPersistence save/load."""
        jp = JournaledPersistence(str(tmp_path))
        assert jp.save_json({"foo": "bar"}, "test.json") is True
        loaded = jp.load_json("test.json")
        assert loaded == {"foo": "bar"}
