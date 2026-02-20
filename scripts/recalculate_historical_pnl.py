#!/usr/bin/env python3
"""
Recalculate P&L for all historical trades.

This script fixes trades affected by the P&L calculation bug where pnl was
overwritten to 0.0 during reward processing.

Usage:
    python3 scripts/recalculate_historical_pnl.py [--dry-run] [--output FILE]

Options:
    --dry-run       Show what would be changed without modifying files
    --output FILE   Output path (default: data/trade_log_corrected.jsonl)
    --backup        Create backup before modifying original file
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def calculate_pnl(entry_price: float, exit_price: float, direction: str, qty: float, contract_size: float) -> float:
    """
    Calculate P&L using correct formula.

    Formula: (exit - entry) * direction_sign * quantity * contract_size

    Args:
        entry_price: Entry execution price
        exit_price: Exit execution price
        direction: "LONG" or "SHORT"
        qty: Position size in lots
        contract_size: Contract size (100.0 for XAUUSD)

    Returns:
        P&L in USD
    """
    direction_sign = 1 if direction == "LONG" else -1
    pnl = (exit_price - entry_price) * direction_sign * qty * contract_size
    return pnl


def analyze_trades(trade_log_path: Path):
    """Analyze trades to identify P&L issues."""
    if not trade_log_path.exists():
        print(f"❌ Trade log not found: {trade_log_path}")
        return

    trades = []
    with open(trade_log_path, "r") as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))

    print(f"📊 Analysis of {len(trades)} trades")
    print("=" * 80)

    zero_pnl_count = 0
    zero_pnl_with_price_diff = 0

    for trade in trades:
        if trade.get("pnl", 0.0) == 0.0:
            zero_pnl_count += 1
            entry = trade.get("entry_price", 0.0)
            exit_p = trade.get("exit_price", 0.0)
            if abs(exit_p - entry) > 0.01:  # Significant price difference
                zero_pnl_with_price_diff += 1

    print(f"Trades with pnl=0.0: {zero_pnl_count} ({zero_pnl_count/len(trades)*100:.1f}%)")
    print(f"Trades with pnl=0.0 but price moved: {zero_pnl_with_price_diff}")
    print(f"Likely affected by bug: {zero_pnl_with_price_diff}")
    print()


def recalculate_trades(
    input_path: Path,
    output_path: Path,
    dry_run: bool = False,
    default_qty: float = 0.1,
    default_contract_size: float = 100.0,
):
    """
    Recalculate P&L for all trades.

    Args:
        input_path: Path to original trade_log.jsonl
        output_path: Path to save corrected trades
        dry_run: If True, show changes without saving
        default_qty: Default quantity (0.1 for most XAUUSD trades)
        default_contract_size: Default contract size (100.0 for XAUUSD)
    """
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    trades = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))

    print(f"🔄 Processing {len(trades)} trades...")
    print()

    corrected_count = 0
    unchanged_count = 0

    for i, trade in enumerate(trades):
        entry_price = trade.get("entry_price", 0.0)
        exit_price = trade.get("exit_price", 0.0)
        direction = trade.get("direction", "UNKNOWN")
        old_pnl = trade.get("pnl", 0.0)

        if entry_price == 0.0 or exit_price == 0.0:
            print(f"⚠️  Trade {i+1}: Skipped (missing prices)")
            unchanged_count += 1
            continue

        # Recalculate P&L
        new_pnl = calculate_pnl(entry_price, exit_price, direction, default_qty, default_contract_size)

        # Update trade record
        if abs(new_pnl - old_pnl) > 0.001:  # Changed
            trade["pnl"] = new_pnl
            trade["pnl_recalculated"] = True
            trade["pnl_recalculated_date"] = datetime.utcnow().isoformat()
            trade["pnl_original"] = old_pnl
            corrected_count += 1

            if dry_run:
                print(f"Trade {i+1}: {direction} {entry_price:.2f}→{exit_price:.2f}")
                print(f"  Old P&L: {old_pnl:.4f}")
                print(f"  New P&L: {new_pnl:.4f}")
                print(f"  Change:  {new_pnl - old_pnl:+.4f}")
                print()
        else:
            unchanged_count += 1

    print("=" * 80)
    print(f"✨ Summary:")
    print(f"  Total trades:     {len(trades)}")
    print(f"  Corrected:        {corrected_count}")
    print(f"  Unchanged:        {unchanged_count}")
    print()

    if not dry_run:
        # Save corrected trades
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for trade in trades:
                f.write(json.dumps(trade, default=str) + "\n")
        print(f"✅ Corrected trades saved to: {output_path}")
    else:
        print("ℹ️  Dry run - no files modified")

    return corrected_count, unchanged_count


def main():
    parser = argparse.ArgumentParser(description="Recalculate P&L for historical trades affected by bug")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without modifying files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/trade_log_corrected.jsonl",
        help="Output file path (default: data/trade_log_corrected.jsonl)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original file before modifying",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/trade_log.jsonl",
        help="Input file path (default: data/trade_log.jsonl)",
    )
    parser.add_argument(
        "--qty",
        type=float,
        default=0.1,
        help="Position quantity in lots (default: 0.1)",
    )
    parser.add_argument(
        "--contract-size",
        type=float,
        default=100.0,
        help="Contract size (default: 100.0 for XAUUSD)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze trades, don't recalculate",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Historical P&L Recalculation" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Analyze trades first
    analyze_trades(input_path)

    if args.analyze_only:
        print("ℹ️  Analysis complete (--analyze-only specified)")
        return

    # Create backup if requested
    if args.backup and not args.dry_run:
        backup_path = input_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        shutil.copy2(input_path, backup_path)
        print(f"💾 Backup created: {backup_path}")
        print()

    # Recalculate
    recalculate_trades(
        input_path=input_path,
        output_path=output_path,
        dry_run=args.dry_run,
        default_qty=args.qty,
        default_contract_size=args.contract_size,
    )

    print()
    print("✨ Done!")


if __name__ == "__main__":
    main()
