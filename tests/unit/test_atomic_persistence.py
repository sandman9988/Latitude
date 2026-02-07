"""Tests for src.persistence.atomic_persistence – AtomicPersistence."""

import json

import pytest

from src.persistence.atomic_persistence import AtomicPersistence


class TestAtomicPersistence:
    @pytest.fixture()
    def ap(self, tmp_path):
        return AtomicPersistence(str(tmp_path))

    # -- init --
    def test_init_creates_dir(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        AtomicPersistence(str(deep))
        assert deep.exists()

    # -- save_json --
    def test_save_creates_file(self, ap, tmp_path):
        assert ap.save_json({"x": 1}, "test.json") is True
        assert (tmp_path / "test.json").exists()

    def test_save_envelope_format(self, ap, tmp_path):
        ap.save_json({"key": "value"}, "test.json")
        with open(tmp_path / "test.json") as f:
            envelope = json.load(f)
        assert "crc32" in envelope
        assert "timestamp" in envelope
        assert "version" in envelope
        assert envelope["data"] == {"key": "value"}

    def test_save_no_backup_on_first(self, ap, tmp_path):
        ap.save_json({"a": 1}, "test.json")
        backups = list(tmp_path.glob("test.json.*.bak"))
        assert len(backups) == 0

    def test_save_creates_backup_on_overwrite(self, ap, tmp_path):
        ap.save_json({"v": 1}, "test.json")
        ap.save_json({"v": 2}, "test.json")
        backups = list(tmp_path.glob("test.json.*.bak"))
        assert len(backups) == 1

    def test_save_no_backup_when_disabled(self, ap, tmp_path):
        ap.save_json({"v": 1}, "test.json")
        ap.save_json({"v": 2}, "test.json", create_backup=False)
        backups = list(tmp_path.glob("test.json.*.bak"))
        # Only backup from the overwrite, but we said create_backup=False so 0 new
        # First save creates no backup (no previous file), second also disabled
        assert len(backups) == 0

    # -- load_json --
    def test_save_load_roundtrip(self, ap):
        data = {"spread": 1.5, "depth": 1.2, "nested": {"a": [1, 2, 3]}}
        ap.save_json(data, "params.json")
        loaded = ap.load_json("params.json")
        assert loaded == data

    def test_load_nonexistent_returns_none(self, ap):
        assert ap.load_json("nope.json") is None

    def test_load_legacy_format(self, ap, tmp_path):
        """Raw JSON (no envelope) is handled gracefully."""
        raw = {"legacy_key": 42}
        with open(tmp_path / "legacy.json", "w") as f:
            json.dump(raw, f)
        loaded = ap.load_json("legacy.json")
        assert loaded == raw

    def test_load_corrupted_data_restores_backup(self, ap, tmp_path):
        # Save v1 then v2 (v1 becomes backup)
        ap.save_json({"v": 1}, "test.json")
        ap.save_json({"v": 2}, "test.json")

        # Corrupt v2 data (change data but keep old CRC)
        path = tmp_path / "test.json"
        with open(path) as f:
            envelope = json.load(f)
        envelope["data"]["v"] = 999  # corrupts CRC
        with open(path, "w") as f:
            json.dump(envelope, f)

        # Load should detect CRC mismatch and restore from backup (v1)
        loaded = ap.load_json("test.json", verify_crc=True)
        assert loaded is not None
        assert loaded["v"] == 1  # restored from backup

    def test_load_skip_crc_verification(self, ap, tmp_path):
        ap.save_json({"v": 1}, "test.json")
        # Corrupt
        path = tmp_path / "test.json"
        with open(path) as f:
            envelope = json.load(f)
        envelope["data"]["v"] = 999
        with open(path, "w") as f:
            json.dump(envelope, f)
        # No CRC check → return corrupted data
        loaded = ap.load_json("test.json", verify_crc=False)
        assert loaded["v"] == 999

    def test_load_invalid_json(self, ap, tmp_path):
        (tmp_path / "bad.json").write_text("not json{{{")
        loaded = ap.load_json("bad.json")
        assert loaded is None  # no backup either

    # -- backups --
    def test_max_backups_enforced(self, ap, tmp_path):
        import time

        for i in range(6):
            ap.save_json({"v": i}, "test.json")
            time.sleep(0.05)  # ensure distinct timestamps

        backups = sorted(tmp_path.glob("test.json.*.bak"))
        assert len(backups) <= AtomicPersistence.MAX_BACKUPS

    def test_list_backups(self, ap, tmp_path):
        ap.save_json({"v": 1}, "test.json")
        ap.save_json({"v": 2}, "test.json")
        backups = ap.list_backups("test.json")
        assert len(backups) >= 1
        assert all(b.endswith(".bak") for b in backups)

    def test_list_backups_empty(self, ap):
        backups = ap.list_backups("nonexistent.json")
        assert backups == []

    # -- restore_from_backup --
    def test_restore_no_backup_returns_none(self, ap, tmp_path):
        # Create a file with bad CRC and no backup
        ap.save_json({"v": 1}, "solo.json")
        path = tmp_path / "solo.json"
        with open(path) as f:
            envelope = json.load(f)
        envelope["data"]["v"] = 999
        with open(path, "w") as f:
            json.dump(envelope, f)

        # No backup exists → returns None
        loaded = ap.load_json("solo.json", verify_crc=True)
        assert loaded is None

    # -- edge cases --
    def test_save_empty_dict(self, ap):
        assert ap.save_json({}, "empty.json") is True
        loaded = ap.load_json("empty.json")
        assert loaded == {}

    def test_save_large_nested(self, ap):
        data = {"items": [{"id": i, "name": f"item_{i}"} for i in range(1000)]}
        assert ap.save_json(data, "large.json") is True
        loaded = ap.load_json("large.json")
        assert len(loaded["items"]) == 1000
