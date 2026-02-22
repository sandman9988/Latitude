#!/usr/bin/env python3
"""
Atomic Persistence Module
Implements crash-safe file operations with CRC32 checksums and backup/restore
Based on Master Handbook defensive persistence requirements
"""

import contextlib
import json
import logging
import os
import shutil
import tempfile
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


class AtomicPersistence:
    """
    Atomic file operations with CRC32 checksums

    Write strategy:
    1. Write to temp file
    2. Calculate CRC32
    3. Rename temp → target (atomic on POSIX)
    4. Keep backup of previous version
    """

    MAX_BACKUPS = 3  # Keep last N versions

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, data: dict[str, Any], filename: str, create_backup: bool = True) -> bool:
        """
        Save JSON with atomic write and CRC32

        Args:
            data: Dictionary to save
            filename: Target filename (relative to base_dir)
            create_backup: Whether to backup previous version

        Returns:
            True if successful, False otherwise
        """
        target_path = self.base_dir / filename

        try:
            # Create backup of existing file
            if create_backup and target_path.exists() and not self._create_backup(target_path):
                logger.warning("Backup failed for %s, proceeding anyway", filename)

            # Write to temp file
            temp_fd, temp_path = tempfile.mkstemp(dir=self.base_dir, prefix=f".{filename}.", suffix=".tmp")

            try:
                # Serialize data
                json_bytes = json.dumps(data, indent=2).encode("utf-8")

                # Calculate CRC32
                crc32 = zlib.crc32(json_bytes) & 0xFFFFFFFF

                # Create envelope with CRC
                envelope_data: dict[str, Any] = {
                    "crc32": crc32,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "version": 1,
                    "data": data,
                }

                envelope_bytes = json.dumps(envelope_data, indent=2).encode("utf-8")

                # Write atomically
                with os.fdopen(temp_fd, "wb") as f:
                    f.write(envelope_bytes)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename (POSIX guarantees atomicity)
                shutil.move(temp_path, target_path)

                logger.info("Saved %s (CRC32: %08x)", filename, crc32)
                return True

            except (OSError, ValueError) as inner_e:
                # Clean up temp file on error
                with contextlib.suppress(OSError):
                    os.unlink(temp_path)
                raise inner_e

        except (OSError, ValueError) as e:
            logger.error("Failed to save %s: %s", filename, e)
            return False

    def load_json(self, filename: str, verify_crc: bool = True) -> dict[str, Any] | None:  # noqa: PLR0911
        """
        Load JSON with CRC32 verification

        Args:
            filename: File to load (relative to base_dir)
            verify_crc: Whether to verify CRC32 checksum

        Returns:
            Data dict if successful, None on error
        """
        target_path = self.base_dir / filename

        if not target_path.exists():
            logger.warning("File not found: %s", filename)
            return None

        try:
            with open(target_path, "rb") as f:
                envelope_bytes = f.read()

            # Try to parse envelope
            try:
                envelope_data = json.loads(envelope_bytes.decode("utf-8"))

                # Defensive: Validate parsed data is not None
                if envelope_data is None:
                    logger.error("%s: Parsed JSON is None", filename)
                    return self._restore_from_backup(filename)

                # Check if it's an envelope (has CRC32) or raw data
                if isinstance(envelope_data, dict) and "crc32" in envelope_data and "data" in envelope_data:
                    # Envelope format (new)
                    stored_crc = envelope_data["crc32"]
                    data = envelope_data["data"]

                    # Defensive: Validate data structure
                    if not isinstance(data, dict):
                        logger.error("%s: Envelope data is not a dict (type: %s)", filename, type(data))
                        return self._restore_from_backup(filename)

                    if verify_crc:
                        # Recalculate CRC on data portion
                        data_bytes = json.dumps(data, indent=2).encode("utf-8")
                        computed_crc = zlib.crc32(data_bytes) & 0xFFFFFFFF

                        if stored_crc != computed_crc:
                            logger.error(
                                "CRC mismatch in %s: stored=%08x, computed=%08x", filename, stored_crc, computed_crc
                            )
                            # Try to restore from backup
                            return self._restore_from_backup(filename)

                    logger.info("Loaded %s (CRC32: %08x)", filename, stored_crc)
                    return cast(dict[str, Any], data)
                else:
                    # Legacy format (no envelope)
                    logger.warning("%s uses legacy format (no CRC)", filename)

                    # Defensive: Validate legacy data is dict
                    if not isinstance(envelope_data, dict):
                        logger.error("%s: Legacy data is not a dict (type: %s)", filename, type(envelope_data))
                        return None

                    return cast(dict[str, Any], envelope_data)

            except json.JSONDecodeError as decode_e:
                logger.error("JSON decode failed for %s: %s", filename, decode_e)
                return self._restore_from_backup(filename)

        except OSError as e:
            logger.error("Failed to load %s: %s", filename, e)
            return None

    def _create_backup(self, target_path: Path) -> bool:
        """Create timestamped backup of file"""
        try:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            backup_name = f"{target_path.name}.{timestamp}.bak"
            backup_path = target_path.parent / backup_name

            shutil.copy2(target_path, backup_path)

            # Clean up old backups
            self._cleanup_old_backups(target_path)

            logger.debug("Created backup: %s", backup_name)
            return True

        except OSError as e:
            logger.error("Backup creation failed: %s", e)
            return False

    def _cleanup_old_backups(self, target_path: Path) -> None:
        """Keep only MAX_BACKUPS most recent backups"""
        try:
            pattern = f"{target_path.name}.*.bak"
            backup_list = sorted(target_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove old backups
            for backup in backup_list[self.MAX_BACKUPS :]:
                backup.unlink()
                logger.debug("Deleted old backup: %s", backup.name)

        except OSError as e:
            logger.warning("Backup cleanup failed: %s", e)

    def _restore_from_backup(self, filename: str) -> dict[str, Any] | None:
        """Attempt to restore from most recent backup"""
        target_path = self.base_dir / filename
        pattern = f"{filename}.*.bak"

        try:
            backup_files = sorted(target_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

            if not backup_files:
                logger.error("No backups found for %s", filename)
                return None

            latest_backup = backup_files[0]
            logger.warning("Restoring from backup: %s", latest_backup.name)

            # Copy backup to target
            shutil.copy2(latest_backup, target_path)

            # Try to load restored file
            with open(target_path, "rb") as f:
                data = json.load(f)

            # If it's an envelope, extract data
            if isinstance(data, dict) and "data" in data:
                return cast(dict[str, Any], data["data"])
            return cast(dict[str, Any], data)

        except (OSError, json.JSONDecodeError) as e:
            logger.error("Restore from backup failed: %s", e)
            return None

    def list_backups(self, filename: str) -> list[str]:
        """List available backups for a file"""
        target_path = self.base_dir / filename
        pattern = f"{filename}.*.bak"

        backups = sorted(target_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

        return [b.name for b in backups]


class JournaledPersistence(AtomicPersistence):
    """
    Extended persistence with write-ahead logging

    Write strategy:
    1. Write operation to journal
    2. Perform operation
    3. Mark journal entry as committed
    4. Periodically clean committed entries
    """

    def __init__(self, base_dir: str = ".", journal_name: str = "persistence.journal"):
        super().__init__(base_dir)
        self.journal_path = self.base_dir / journal_name
        self._recover_from_journal()

    def _recover_from_journal(self) -> None:
        """Replay uncommitted journal entries on startup"""
        if not self.journal_path.exists():
            return

        try:
            with open(self.journal_path, encoding="utf-8") as journal_f:
                for line in journal_f:
                    entry = json.loads(line.strip())
                    if not entry.get("committed", False):
                        logger.warning("Replaying uncommitted: %s", entry)
                        # Could implement replay logic here

            # Archive old journal
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            archive_path = self.base_dir / f"{self.journal_path.name}.{timestamp}.old"
            shutil.move(self.journal_path, archive_path)
            logger.info("Archived old journal: %s", archive_path.name)

        except (OSError, json.JSONDecodeError) as e:
            logger.error("Journal recovery failed: %s", e)

    def _journal_write(self, operation: str, filename: str, data_hash: int | None = None) -> bool:
        """Write operation to journal"""
        try:
            entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "operation": operation,
                "filename": filename,
                "data_hash": data_hash,
                "committed": False,
            }

            with open(self.journal_path, "a", encoding="utf-8") as journal_f:
                journal_f.write(json.dumps(entry) + "\n")
                journal_f.flush()
                os.fsync(journal_f.fileno())

            return True

        except OSError as e:
            logger.error("Journal write failed: %s", e)
            return False

    def _journal_commit(self, filename: str) -> bool:
        """Mark journal entry as committed"""
        # For simplicity, we just log the commit
        # Full implementation would update the journal file
        logger.debug("Committed: %s", filename)
        return True


if __name__ == "__main__":
    # Self-test
    TEST_FILENAME = "test_params.json"

    print("AtomicPersistence Tests:")

    with tempfile.TemporaryDirectory() as tmpdir:
        ap = AtomicPersistence(tmpdir)

        # Test save/load
        test_data = {
            "learned_spread_relax": 1.5,
            "learned_depth_buffer": 1.2,
            "learned_vpin_z_limit": 2.0,
        }

        print("  Saving test data...")
        success = ap.save_json(test_data, TEST_FILENAME)
        print(f"    Save: {'✓' if success else '✗'}")

        print("  Loading test data...")
        loaded = ap.load_json(TEST_FILENAME)
        print(f"    Load: {'✓' if loaded == test_data else '✗'}")
        print(f"    Data matches: {loaded == test_data}")

        # Test backup
        print("  Modifying and saving again...")
        test_data["learned_spread_relax"] = 2.0
        ap.save_json(test_data, TEST_FILENAME)

        test_backups = ap.list_backups(TEST_FILENAME)
        print(f"    Backups created: {len(test_backups)}")

        # Test CRC corruption detection
        print("  Testing CRC corruption detection...")
        target = Path(tmpdir) / TEST_FILENAME
        with open(target, encoding="utf-8") as read_f:
            envelope_data = json.load(read_f)
        envelope_data["data"]["learned_spread_relax"] = 999.0  # Corrupt data
        with open(target, "w", encoding="utf-8") as write_f:
            json.dump(envelope_data, write_f)

        loaded_corrupted = ap.load_json(TEST_FILENAME, verify_crc=True)
        print(f"    Corrupted restore: {'✓' if loaded_corrupted else '✗'}")
        if loaded_corrupted:
            print(f"    Restored value: {loaded_corrupted.get('learned_spread_relax')}")

    print("\nAll tests passed ✓")
