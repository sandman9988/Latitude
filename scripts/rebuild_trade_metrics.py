#!/usr/bin/env python3
"""Rebuild history-derived metric snapshots from trade_log.jsonl.

This intentionally does not rewrite paper_stats_*.json.  Those files contain
runtime session counters from the active bots and cannot be faithfully rebuilt
from lifetime trade history without a trusted session-start boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.persistence.trade_log_reader import read_all_trades  # noqa: E402
from src.utils.metrics_calculator import period_metrics  # noqa: E402


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.metrics_backup_{stamp}")
    shutil.copy2(path, backup)
    return backup


def _entry_dt(trade: dict[str, Any]) -> datetime | None:
    return _parse_dt(trade.get("entry_time"))


def _close_or_entry_dt(trade: dict[str, Any]) -> datetime | None:
    return _parse_dt(trade.get("exit_time") or trade.get("entry_time"))


def build_performance_snapshot(
    trades: list[dict[str, Any]],
    *,
    trading_mode: str,
    starting_equity: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the same rolling performance snapshot the paper bot writes."""
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    else:
        now = now.astimezone(UTC)

    cutoff_24h = now - timedelta(hours=24)
    cutoff_7d = now - timedelta(days=7)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    daily: list[dict[str, Any]] = []
    weekly: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    for trade in trades:
        tdt = _entry_dt(trade)
        if tdt is None:
            continue
        if tdt >= cutoff_24h:
            daily.append(trade)
        if tdt >= cutoff_7d:
            weekly.append(trade)
        if tdt >= month_start:
            monthly.append(trade)

    def calc(period_trades: list[dict[str, Any]]) -> dict[str, Any]:
        return period_metrics(period_trades, starting_equity=starting_equity)

    return {
        "trading_mode": trading_mode,
        "rebuilt_at": now.isoformat(),
        "source": "trade_log.jsonl",
        "daily": calc(daily),
        "weekly": calc(weekly),
        "monthly": calc(monthly),
        "lifetime": calc(trades),
    }


def build_epoch_metrics(
    trades: list[dict[str, Any]],
    *,
    epoch_path: Path,
    starting_equity: float,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    if not epoch_path.exists():
        return None
    try:
        epoch_data = json.loads(epoch_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    epoch_raw = epoch_data.get("epoch")
    epoch_dt = _parse_dt(epoch_raw)
    if epoch_dt is None:
        return {
            "epoch": epoch_raw,
            "set_at": epoch_data.get("set_at"),
            "rebuilt_at": (now or datetime.now(UTC)).isoformat(),
            "included_trades": len(trades),
            "excluded_trades": 0,
            "excluded_pnl": 0.0,
            "metrics": period_metrics(trades, starting_equity=starting_equity),
        }

    included: list[dict[str, Any]] = []
    excluded_pnl = 0.0
    for trade in trades:
        tdt = _close_or_entry_dt(trade)
        if tdt is not None and tdt < epoch_dt:
            excluded_pnl += float(trade.get("pnl", 0.0) or 0.0)
        else:
            included.append(trade)

    return {
        "epoch": epoch_dt.isoformat(),
        "set_at": epoch_data.get("set_at"),
        "rebuilt_at": (now or datetime.now(UTC)).isoformat(),
        "included_trades": len(included),
        "excluded_trades": len(trades) - len(included),
        "excluded_pnl": excluded_pnl,
        "metrics": period_metrics(included, starting_equity=starting_equity),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--trade-log", type=Path, default=None)
    parser.add_argument("--trading-mode", default="paper", choices=("paper", "live"))
    parser.add_argument("--starting-equity", type=float, default=10_000.0)
    parser.add_argument("--write", action="store_true", help="Write rebuilt JSON files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir
    trade_log = args.trade_log or (data_dir / "trade_log.jsonl")
    trades = read_all_trades(trade_log)
    now = datetime.now(UTC)

    snapshot = build_performance_snapshot(
        trades,
        trading_mode=args.trading_mode,
        starting_equity=args.starting_equity,
        now=now,
    )
    epoch = build_epoch_metrics(
        trades,
        epoch_path=data_dir / "stats_epoch.json",
        starting_equity=args.starting_equity,
        now=now,
    )

    print(
        json.dumps(
            {
                "trade_log": str(trade_log),
                "trades": len(trades),
                "performance_lifetime_pnl": snapshot["lifetime"]["total_pnl"],
                "performance_lifetime_capture": snapshot["lifetime"]["avg_capture_ratio"],
                "epoch_included_trades": epoch["included_trades"] if epoch else None,
                "epoch_excluded_trades": epoch["excluded_trades"] if epoch else None,
                "write": bool(args.write),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if not args.write:
        return 0

    perf_path = data_dir / "performance_snapshot.json"
    epoch_path = data_dir / "stats_epoch_metrics.json"
    perf_backup = _backup(perf_path)
    epoch_backup = _backup(epoch_path)
    _atomic_write_json(perf_path, snapshot)
    if epoch is not None:
        _atomic_write_json(epoch_path, epoch)

    print(
        json.dumps(
            {
                "wrote": [str(perf_path), str(epoch_path) if epoch is not None else None],
                "backups": [str(p) for p in (perf_backup, epoch_backup) if p is not None],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
