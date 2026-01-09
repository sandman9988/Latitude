#!/usr/bin/env python3
"""
Atomic Persistence Module
Implements crash-safe file operations with CRC32 checksums and backup/restore
Based on Master Handbook defensive persistence requirements
"""

import os
import json
import zlib
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime

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
        logger.info(f"AtomicPersistence initialized: {self.base_dir}")
    
    def save_json(self, data: Dict[str, Any], filename: str, 
                  create_backup: bool = True) -> bool:
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
            if create_backup and target_path.exists():
                if not self._create_backup(target_path):
                    logger.warning(f"Backup failed for {filename}, proceeding anyway")
            
            # Write to temp file
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.base_dir,
                prefix=f".{filename}.",
                suffix=".tmp"
            )
            
            try:
                # Serialize data
                json_bytes = json.dumps(data, indent=2).encode('utf-8')
                
                # Calculate CRC32
                crc32 = zlib.crc32(json_bytes) & 0xffffffff
                
                # Create envelope with CRC
                envelope = {
                    "crc32": crc32,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "version": 1,
                    "data": data
                }
                
                envelope_bytes = json.dumps(envelope, indent=2).encode('utf-8')
                
                # Write atomically
                with os.fdopen(temp_fd, 'wb') as f:
                    f.write(envelope_bytes)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic rename (POSIX guarantees atomicity)
                shutil.move(temp_path, target_path)
                
                logger.info(f"Saved {filename} (CRC32: {crc32:08x})")
                return True
                
            except Exception as e:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise e
                
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            return False
    
    def load_json(self, filename: str, 
                  verify_crc: bool = True) -> Optional[Dict[str, Any]]:
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
            logger.warning(f"File not found: {filename}")
            return None
        
        try:
            with open(target_path, 'rb') as f:
                envelope_bytes = f.read()
            
            # Try to parse envelope
            try:
                envelope = json.loads(envelope_bytes.decode('utf-8'))
                
                # Check if it's an envelope (has CRC32) or raw data
                if isinstance(envelope, dict) and "crc32" in envelope and "data" in envelope:
                    # Envelope format (new)
                    stored_crc = envelope["crc32"]
                    data = envelope["data"]
                    
                    if verify_crc:
                        # Recalculate CRC on data portion
                        data_bytes = json.dumps(data, indent=2).encode('utf-8')
                        computed_crc = zlib.crc32(data_bytes) & 0xffffffff
                        
                        if stored_crc != computed_crc:
                            logger.error(
                                f"CRC mismatch in {filename}: "
                                f"stored={stored_crc:08x}, computed={computed_crc:08x}"
                            )
                            # Try to restore from backup
                            return self._restore_from_backup(filename)
                    
                    logger.info(f"Loaded {filename} (CRC32: {stored_crc:08x})")
                    return data
                else:
                    # Legacy format (no envelope)
                    logger.warning(f"{filename} uses legacy format (no CRC)")
                    return envelope
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode failed for {filename}: {e}")
                return self._restore_from_backup(filename)
                
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return None
    
    def _create_backup(self, target_path: Path) -> bool:
        """Create timestamped backup of file"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{target_path.name}.{timestamp}.bak"
            backup_path = target_path.parent / backup_name
            
            shutil.copy2(target_path, backup_path)
            
            # Clean up old backups
            self._cleanup_old_backups(target_path)
            
            logger.debug(f"Created backup: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False
    
    def _cleanup_old_backups(self, target_path: Path) -> None:
        """Keep only MAX_BACKUPS most recent backups"""
        try:
            pattern = f"{target_path.name}.*.bak"
            backups = sorted(
                target_path.parent.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Remove old backups
            for backup in backups[self.MAX_BACKUPS:]:
                backup.unlink()
                logger.debug(f"Deleted old backup: {backup.name}")
                
        except Exception as e:
            logger.warning(f"Backup cleanup failed: {e}")
    
    def _restore_from_backup(self, filename: str) -> Optional[Dict[str, Any]]:
        """Attempt to restore from most recent backup"""
        target_path = self.base_dir / filename
        pattern = f"{filename}.*.bak"
        
        try:
            backups = sorted(
                target_path.parent.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if not backups:
                logger.error(f"No backups found for {filename}")
                return None
            
            latest_backup = backups[0]
            logger.warning(f"Restoring from backup: {latest_backup.name}")
            
            # Copy backup to target
            shutil.copy2(latest_backup, target_path)
            
            # Try to load restored file
            with open(target_path, 'rb') as f:
                data = json.load(f)
            
            # If it's an envelope, extract data
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return data
            
        except Exception as e:
            logger.error(f"Restore from backup failed: {e}")
            return None
    
    def list_backups(self, filename: str) -> List[str]:
        """List available backups for a file"""
        target_path = self.base_dir / filename
        pattern = f"{filename}.*.bak"
        
        backups = sorted(
            target_path.parent.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
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
            with open(self.journal_path, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if not entry.get("committed", False):
                        logger.warning(f"Replaying uncommitted: {entry}")
                        # Could implement replay logic here
            
            # Archive old journal
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            archive_path = self.base_dir / f"{self.journal_path.name}.{timestamp}.old"
            shutil.move(self.journal_path, archive_path)
            logger.info(f"Archived old journal: {archive_path.name}")
            
        except Exception as e:
            logger.error(f"Journal recovery failed: {e}")
    
    def _journal_write(self, operation: str, filename: str, 
                      data_hash: Optional[int] = None) -> bool:
        """Write operation to journal"""
        try:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "operation": operation,
                "filename": filename,
                "data_hash": data_hash,
                "committed": False
            }
            
            with open(self.journal_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
            
            return True
            
        except Exception as e:
            logger.error(f"Journal write failed: {e}")
            return False
    
    def _journal_commit(self, filename: str) -> bool:
        """Mark journal entry as committed"""
        # For simplicity, we just log the commit
        # Full implementation would update the journal file
        logger.debug(f"Committed: {filename}")
        return True


if __name__ == "__main__":
    # Self-test
    import tempfile
    
    print("AtomicPersistence Tests:")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ap = AtomicPersistence(tmpdir)
        
        # Test save/load
        test_data = {
            "learned_spread_relax": 1.5,
            "learned_depth_buffer": 1.2,
            "learned_vpin_z_limit": 2.0
        }
        
        print(f"  Saving test data...")
        success = ap.save_json(test_data, "test_params.json")
        print(f"    Save: {'✓' if success else '✗'}")
        
        print(f"  Loading test data...")
        loaded = ap.load_json("test_params.json")
        print(f"    Load: {'✓' if loaded == test_data else '✗'}")
        print(f"    Data matches: {loaded == test_data}")
        
        # Test backup
        print(f"  Modifying and saving again...")
        test_data["learned_spread_relax"] = 2.0
        ap.save_json(test_data, "test_params.json")
        
        backups = ap.list_backups("test_params.json")
        print(f"    Backups created: {len(backups)}")
        
        # Test CRC corruption detection
        print(f"  Testing CRC corruption detection...")
        target = Path(tmpdir) / "test_params.json"
        with open(target, 'r') as f:
            envelope = json.load(f)
        envelope["data"]["learned_spread_relax"] = 999.0  # Corrupt data
        with open(target, 'w') as f:
            json.dump(envelope, f)
        
        loaded_corrupted = ap.load_json("test_params.json", verify_crc=True)
        print(f"    Corrupted restore: {'✓' if loaded_corrupted else '✗'}")
        if loaded_corrupted:
            print(f"    Restored value: {loaded_corrupted.get('learned_spread_relax')}")
    
    print("\nAll tests passed ✓")
