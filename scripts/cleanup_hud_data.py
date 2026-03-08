#!/usr/bin/env python3
"""
Data Cleanup & Archive
======================
Cleans up HUD data directory:
1. Archives files older than 7 days
2. Consolidates learned_parameters backups
3. Removes duplicate risk_metrics files
4. Enforces file freshness policy

Usage:
    python3 scripts/cleanup_hud_data.py [--dry-run] [--archive-dir DATA/ARCHIVE]
    python3 scripts/cleanup_hud_data.py --analyze      # Show what will be archived
    python3 scripts/cleanup_hud_data.py --execute     # Apply cleanup
"""

import argparse
import shutil
from datetime import datetime, UTC, timedelta
from pathlib import Path

DATA_DIR = Path("data")
ARCHIVE_DIR = DATA_DIR / "archive"
STALE_THRESHOLD_DAYS = 7
KEEP_RECENT_BACKUPS = 1

# Files that should be archived if stale
ARCHIVABLE_FILES = {
    "universe.json",
    "learned_parameters.json",
    "test_decision_log.json",
    "decision_log.json",
    "bars_cache.json",
    "order_book.json",
}

# Backup patterns to consolidate (keep only most recent)
BACKUP_PATTERNS = {
    "learned_parameters": ["learned_parameters.json.backup_*", "learned_parameters*.bak"],
}


def analyze_data_dir():
    """Analyze what needs cleanup."""
    if not DATA_DIR.exists():
        print("❌ Data directory not found")
        return

    print("\n" + "=" * 100)
    print("📋 DATA CLEANUP ANALYSIS")
    print("=" * 100)

    now = datetime.now(UTC)
    stale_threshold = now - timedelta(days=STALE_THRESHOLD_DAYS)

    # Find stale files
    print(f"\n🗁 Files to Archive (> {STALE_THRESHOLD_DAYS} days old):")
    stale_files = []
    for json_file in DATA_DIR.glob("*.json"):
        mtime = datetime.fromtimestamp(json_file.stat().st_mtime, UTC)
        age_days = (now - mtime).days

        if json_file.name in ARCHIVABLE_FILES and age_days > STALE_THRESHOLD_DAYS:
            stale_files.append((json_file.name, age_days, mtime))
            print(f"  ❌ {json_file.name:40} | {age_days:3}d old | {mtime.date()}")

    # Find backup clutter
    print(f"\n📦 Backup Files to Consolidate:")
    backups_to_delete = []
    for filename, patterns in BACKUP_PATTERNS.items():
        backup_files = []
        for pattern in patterns:
            backup_files.extend(DATA_DIR.glob(pattern))

        if backup_files:
            # Keep most recent, delete others
            backup_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            kept = backup_files[:KEEP_RECENT_BACKUPS]
            to_delete = backup_files[KEEP_RECENT_BACKUPS:]

            print(f"\n  {filename}:")
            print(f"    Keep:   {kept[0].name if kept else 'none'}")
            if to_delete:
                print(f"    Delete: {len(to_delete)} files")
                for f in to_delete:
                    age = (now - datetime.fromtimestamp(f.stat().st_mtime, UTC)).days
                    print(f"      - {f.name} ({age}d old)")
                    backups_to_delete.extend(to_delete)

    # Summary
    print("\n" + "=" * 100)
    print(f"SUMMARY:")
    print(f"  Files to archive: {len(stale_files)}")
    print(f"  Backup files to delete: {len(backups_to_delete)}")
    print(f"  Archive location: {ARCHIVE_DIR}")
    print("=" * 100 + "\n")

    return stale_files, backups_to_delete


def execute_cleanup(dry_run: bool = True):
    """Execute the cleanup operations."""
    if not DATA_DIR.exists():
        print("❌ Data directory not found")
        return

    stale_files, backups_to_delete = analyze_data_dir()

    if dry_run:
        print("🔄 DRY-RUN MODE: No files will be modified\n")

    # Archive stale files
    if stale_files:
        if not dry_run:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

        print("Archiving stale files:")
        for filename, age_days, mtime in stale_files:
            src = DATA_DIR / filename
            dst = ARCHIVE_DIR / f"{filename}.{mtime.strftime('%Y%m%d_%H%M%S')}"

            if dry_run:
                print(f"  → {filename} → archive/")
            else:
                try:
                    shutil.copy2(src, dst)
                    src.unlink()
                    print(f"  ✓ {filename} → archive/")
                except Exception as e:
                    print(f"  ❌ {filename} - Error: {e}")

    # Delete old backups
    if backups_to_delete:
        print("\nDeleting old backup files:")
        for backup_file in backups_to_delete:
            if dry_run:
                print(f"  → DELETE {backup_file.name}")
            else:
                try:
                    backup_file.unlink()
                    print(f"  ✓ Deleted {backup_file.name}")
                except Exception as e:
                    print(f"  ❌ {backup_file.name} - Error: {e}")

    if not dry_run:
        print(f"\n✓ Cleanup complete!")
        print(f"✓ Created archive: {ARCHIVE_DIR}")
        print(f"✓ Run: python3 scripts/cleanup_hud_data.py --analyze  # Verify")
    else:
        print("\n✓ Dry-run complete. Run with --execute to apply changes.")


def main():
    parser = argparse.ArgumentParser(description="Clean up stale HUD data files")
    parser.add_argument("--analyze", action="store_true", help="Analyze what will be cleaned")
    parser.add_argument("--execute", action="store_true", help="Execute cleanup operations")
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR, help="Archive directory location")

    args = parser.parse_args()

    if not args.analyze and not args.execute:
        args.analyze = True

    if args.analyze:
        analyze_data_dir()

    if args.execute:
        execute_cleanup(dry_run=False)


if __name__ == "__main__":
    main()
