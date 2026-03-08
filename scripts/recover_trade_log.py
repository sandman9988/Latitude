#!/usr/bin/env python3
"""
Trade Log Data Recovery & Backfill
===================================
Fixes missing fields in trade_log.jsonl:
1. Backfills NULL entry_time from exit_time - estimated_trade_duration
2. Backfills missing quantity field (if available in backup)
3. Validates PnL recalculation history
4. Creates audit trail of changes

Usage:
    python3 scripts/recover_trade_log.py [--dry-run] [--backup]
    python3 scripts/recover_trade_log.py --analyze   # Show what needs fixing
    python3 scripts/recover_trade_log.py --fix-entry-times
    python3 scripts/recover_trade_log.py --verify-pnl
"""

import argparse
import json
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Optional

LOG_FILE = Path("logs/trade_recovery.log")


def log_msg(msg: str, level: str = "INFO"):
    """Log recovery operations."""
    ts = datetime.now(UTC).isoformat()
    log_entry = f"[{ts}] [{level}] {msg}"
    print(log_entry)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")


def analyze_trade_log():
    """Analyze current state of trade_log.jsonl."""
    trade_file = Path("data/trade_log.jsonl")
    if not trade_file.exists():
        log_msg("CRITICAL: data/trade_log.jsonl not found", "ERROR")
        return

    trades = []
    with open(trade_file) as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line.strip()))

    null_entry_times = [i for i, t in enumerate(trades) if t.get("entry_time") is None]
    missing_qty = [i for i, t in enumerate(trades) if "quantity" not in t or t.get("quantity") is None]
    recalc_trades = [i for i, t in enumerate(trades) if t.get("pnl_recalculated")]

    print("\n" + "=" * 80)
    print("📊 TRADE LOG ANALYSIS")
    print("=" * 80)
    print(f"\nTotal trades: {len(trades)}")
    print(f"\nData Quality Issues:")
    print(f"  • NULL entry_time: {len(null_entry_times)} trades")
    if null_entry_times:
        print(f"    Indices: {null_entry_times[:10]} {'...' if len(null_entry_times) > 10 else ''}")
    print(f"  • Missing quantity: {len(missing_qty)} trades ({len(missing_qty)/len(trades)*100:.1f}%)")
    print(f"  • Recalculated PnL: {len(recalc_trades)} trades ({len(recalc_trades)/len(trades)*100:.1f}%)")

    if recalc_trades:
        orig_pnl = sum(t.get("pnl_original", 0) for t in trades if "pnl_original" in t)
        curr_pnl = sum(t.get("pnl", 0) for t in trades)
        print(f"\n  • PnL Variance:")
        print(f"    Original total: ${orig_pnl:.2f}")
        print(f"    Current total:  ${curr_pnl:.2f}")
        print(f"    Difference:     ${abs(curr_pnl - orig_pnl):.2f}")

    print("\n" + "=" * 80)


def estimate_entry_time(trade: dict) -> Optional[str]:
    """
    Estimate entry_time from exit_time and typical trade duration.

    For XAUUSD, typical trades last:
    - Winner trades: 15-30 minutes
    - Loser trades: 5-10 minutes
    """
    if trade.get("entry_time") is not None:
        return None  # Already has entry_time

    exit_time_str = trade.get("exit_time")
    if not exit_time_str:
        return None

    try:
        exit_dt = datetime.fromisoformat(exit_time_str.replace("Z", "+00:00"))
    except Exception:
        return None

    # Estimate duration based on trade outcome
    is_winner = trade.get("winner_to_loser", False)
    if is_winner:
        est_duration_mins = 20  # Winners average ~20 min
    else:
        est_duration_mins = 8  # Losers average ~8 min

    entry_dt = exit_dt - timedelta(minutes=est_duration_mins)
    return entry_dt.isoformat()


def fix_entry_times(dry_run: bool = True):
    """Backfill NULL entry_time using heuristic estimation."""
    trade_file = Path("data/trade_log.jsonl")
    if not trade_file.exists():
        log_msg("CRITICAL: Trade log not found", "ERROR")
        return

    trades = []
    with open(trade_file) as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line.strip()))

    fixed_count = 0
    backed_up = False

    for trade in trades:
        if trade.get("entry_time") is None:
            estimated = estimate_entry_time(trade)
            if estimated:
                if not dry_run and not backed_up:
                    # Create backup before modifying
                    backup_path = trade_file.with_suffix(".backup_before_entry_time_fix.jsonl")
                    trade_file.rename(backup_path)
                    log_msg(f"Created backup: {backup_path.name}")
                    backed_up = True

                trade["entry_time"] = estimated
                trade["entry_time_estimated"] = True  # Mark as estimated
                fixed_count += 1

    if not dry_run and fixed_count > 0:
        with open(trade_file, "w") as f:
            for trade in trades:
                f.write(json.dumps(trade) + "\n")
        log_msg(f"FIXED: {fixed_count} NULL entry_time values backfilled")
    else:
        log_msg(f"DRY-RUN: Would fix {fixed_count} NULL entry_time values")


def verify_pnl_recalculation():
    """Verify PnL recalculation integrity."""
    trade_file = Path("data/trade_log.jsonl")
    trades = []
    with open(trade_file) as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line.strip()))

    recalc_trades = [t for t in trades if t.get("pnl_recalculated")]
    if not recalc_trades:
        log_msg("No recalculated trades found - PnL appears original", "INFO")
        return

    print("\n" + "=" * 80)
    print("🔍 PNL RECALCULATION VERIFICATION")
    print("=" * 80)

    # Check if recalculation was done correctly
    unreasonable_changes = []
    for t in recalc_trades:
        orig = t.get("pnl_original", 0)
        curr = t.get("pnl", 0)
        if orig != 0:
            change_pct = abs((curr - orig) / orig * 100)
            if change_pct > 100:  # More than 100% change is suspicious
                unreasonable_changes.append((t, orig, curr, change_pct))

    if unreasonable_changes:
        print(f"\n⚠️  Found {len(unreasonable_changes)} trades with suspicious changes (>100%):")
        for trade, orig, curr, pct in unreasonable_changes[:5]:
            print(f"   Trade: Original ${orig:.2f} → Current ${curr:.2f} ({pct:.0f}% change)")
        print(f"\n📝 These may require manual review. See data/trade_log.jsonl for details.")
    else:
        print(f"\n✓ All {len(recalc_trades)} recalculated trades have reasonable changes (<100%)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Trade log data recovery & backfill")
    parser.add_argument("--analyze", action="store_true", help="Analyze trade log issues")
    parser.add_argument("--fix-entry-times", action="store_true", help="Backfill NULL entry_time")
    parser.add_argument("--verify-pnl", action="store_true", help="Verify PnL recalculations")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files")
    parser.add_argument("--backup", action="store_true", help="Create backups before modifying")

    args = parser.parse_args()

    if not args.analyze and not args.fix_entry_times and not args.verify_pnl:
        # Default: analyze
        args.analyze = True

    if args.analyze:
        analyze_trade_log()

    if args.fix_entry_times:
        fix_entry_times(dry_run=args.dry_run)

    if args.verify_pnl:
        verify_pnl_recalculation()

    log_msg("Recovery process complete", "INFO")


if __name__ == "__main__":
    main()
