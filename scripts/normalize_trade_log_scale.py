#!/usr/bin/env python3
"""Normalize trade_log.jsonl MFE/MAE scale and capture-derived fields.

The canonical trade-log convention is:
- pnl, mfe, mae: account currency / dollars
- pnl_points, mfe_points, mae_points: price-point movement
- capture_ratio: pnl / mfe using account-currency units

Older records predate explicit *_points fields.  This script backfills those
fields, recomputes capture_ratio, winner_to_loser, zero-MFE diagnostics, and
harvester_quality, then writes a timestamped backup before replacing the log.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_CONTRACT_SIZE_BY_SYMBOL: dict[str, float] = {
    "XAUUSD": 100.0,
}
NORMALIZER_VERSION = 1


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _round_float(value: float, digits: int = 9) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0 else rounded


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-9)
    return left == right


def _records_equal(left: dict, right: dict) -> bool:
    if set(left) != set(right):
        return False
    return all(_values_equal(left[key], right[key]) for key in left)


def _as_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    return symbol or "UNKNOWN"


def read_jsonl(path: Path) -> list[dict]:
    trades: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"{path}:{line_no}: invalid JSON: {exc}"
                raise ValueError(msg) from exc
            if isinstance(rec, dict):
                trades.append(rec)
    return trades


def infer_contract_sizes(trades: list[dict]) -> dict[str, float]:
    """Infer per-symbol contract sizes from records with both scales present."""
    samples: dict[str, list[float]] = {}
    for trade in trades:
        symbol = _as_symbol(trade.get("symbol"))
        qty = abs(_as_float(trade.get("quantity")))
        mfe_dollars = abs(_as_float(trade.get("mfe")))
        mfe_points = abs(_as_float(trade.get("mfe_points")))
        if qty <= 0 or mfe_dollars <= 0 or mfe_points <= 0:
            continue
        contract_size = mfe_dollars / (mfe_points * qty)
        if 0 < contract_size < 10_000_000 and math.isfinite(contract_size):
            samples.setdefault(symbol, []).append(contract_size)

    inferred: dict[str, float] = {}
    for symbol, values in samples.items():
        inferred[symbol] = _round_float(float(statistics.median(values)))
    return inferred


def contract_size_for(symbol: str, inferred: dict[str, float], fallback: float) -> float:
    if symbol in inferred:
        return inferred[symbol]
    return DEFAULT_CONTRACT_SIZE_BY_SYMBOL.get(symbol, fallback)


def _quality_from_capture(pnl: float, mfe: float, winner_to_loser: bool) -> str:
    if winner_to_loser:
        return "POOR_WTL"
    if pnl <= 0:
        return "STOPPED_OUT"
    if mfe <= 0:
        return "N/A"
    capture = pnl / mfe
    if capture >= 0.8:
        return "EXCELLENT"
    if capture >= 0.6:
        return "GOOD"
    if capture >= 0.35:
        return "FAIR"
    return "POOR"


def normalize_trade(
    trade: dict,
    inferred_contract_sizes: dict[str, float],
    *,
    fallback_contract_size: float = 100.0,
    timestamp: str | None = None,
) -> tuple[dict, bool]:
    """Return (normalized_trade, changed)."""
    original = deepcopy(trade)
    out = dict(trade)
    if (
        out.get("scale_normalized") is True
        and int(out.get("scale_normalized_version", 0) or 0) == NORMALIZER_VERSION
        and out.get("scale_normalized_at")
    ):
        timestamp = str(out["scale_normalized_at"])
    else:
        timestamp = timestamp or datetime.now(UTC).isoformat()

    symbol = _as_symbol(out.get("symbol"))
    qty = abs(_as_float(out.get("quantity"), 0.0))
    contract_size = contract_size_for(symbol, inferred_contract_sizes, fallback_contract_size)
    scale = qty * contract_size

    pnl = _as_float(out.get("pnl"), 0.0)
    mfe_dollars = abs(_as_float(out.get("mfe"), 0.0))
    mae_dollars = abs(_as_float(out.get("mae"), 0.0))
    mfe_points = abs(_as_float(out.get("mfe_points"), 0.0))
    mae_points = abs(_as_float(out.get("mae_points"), 0.0))

    if scale > 0:
        if mfe_points > 0:
            mfe_dollars = mfe_points * scale
        else:
            mfe_points = mfe_dollars / scale if mfe_dollars > 0 else 0.0

        if mae_points > 0:
            mae_dollars = mae_points * scale
        else:
            mae_points = mae_dollars / scale if mae_dollars > 0 else 0.0

        out["pnl_points"] = _round_float(pnl / scale)

    out["symbol"] = symbol
    out["mfe"] = _round_float(mfe_dollars)
    out["mae"] = _round_float(mae_dollars)
    out["mfe_points"] = _round_float(mfe_points)
    out["mae_points"] = _round_float(mae_points)

    capture_ratio = (pnl / mfe_dollars) if mfe_dollars > 1e-12 else 0.0
    out["capture_ratio"] = _round_float(capture_ratio)
    out["capture_pct"] = _round_float(capture_ratio * 100.0)

    winner_to_loser = bool(mfe_dollars > 1e-12 and pnl < -1e-12)
    out["winner_to_loser"] = winner_to_loser
    out["diag_zero_mfe_loss"] = bool(mfe_dollars <= 1e-12 and pnl < -1e-12)
    out["harvester_quality"] = _quality_from_capture(pnl, mfe_dollars, winner_to_loser)

    out["scale_normalized"] = True
    out["scale_normalized_version"] = NORMALIZER_VERSION
    out["scale_normalized_at"] = timestamp
    out["scale_contract_size"] = _round_float(contract_size)

    if "mfe_points" not in original and mfe_points > 0:
        out["mfe_points_backfilled"] = True
    if "mae_points" not in original and mae_points > 0:
        out["mae_points_backfilled"] = True

    return out, not _records_equal(out, original)


def normalize_trades(
    trades: list[dict],
    *,
    fallback_contract_size: float = 100.0,
    timestamp: str | None = None,
) -> tuple[list[dict], dict[str, int | float | dict[str, float]]]:
    inferred = infer_contract_sizes(trades)
    normalized: list[dict] = []
    changed = 0
    backfilled_points = 0
    wtl_changed = 0
    quality_changed = 0
    timestamp = timestamp or datetime.now(UTC).isoformat()

    for trade in trades:
        old_wtl = trade.get("winner_to_loser")
        old_quality = trade.get("harvester_quality")
        rec, did_change = normalize_trade(
            trade,
            inferred,
            fallback_contract_size=fallback_contract_size,
            timestamp=timestamp,
        )
        normalized.append(rec)
        changed += int(did_change)
        backfilled_points += int(bool(rec.get("mfe_points_backfilled") or rec.get("mae_points_backfilled")))
        wtl_changed += int(old_wtl != rec.get("winner_to_loser"))
        quality_changed += int(old_quality != rec.get("harvester_quality"))

    return normalized, {
        "total": len(trades),
        "changed": changed,
        "backfilled_points": backfilled_points,
        "wtl_changed": wtl_changed,
        "quality_changed": quality_changed,
        "inferred_contract_sizes": inferred,
    }


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for rec in records:
                handle.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):  # type: ignore[name-defined]
            os.unlink(tmp_name)
        raise


def backup_path_for(path: Path, timestamp: str) -> Path:
    stamp = timestamp.replace("-", "").replace(":", "").replace("+", "_").replace("T", "_").split(".")[0]
    return path.with_name(f"{path.name}.scale_backup_{stamp}")


def normalize_file(path: Path, *, dry_run: bool, fallback_contract_size: float) -> dict:
    trades = read_jsonl(path)
    timestamp = datetime.now(UTC).isoformat()
    normalized, summary = normalize_trades(
        trades,
        fallback_contract_size=fallback_contract_size,
        timestamp=timestamp,
    )
    summary = dict(summary)
    summary["path"] = str(path)
    summary["dry_run"] = dry_run
    summary["backup"] = ""

    if not dry_run and summary["changed"]:
        backup_path = backup_path_for(path, timestamp)
        backup_path.write_bytes(path.read_bytes())
        write_jsonl_atomic(path, normalized)
        summary["backup"] = str(backup_path)

    return summary


def discover_logs(root: Path) -> list[Path]:
    paths = [root / "trade_log.jsonl"]
    paths.extend(sorted(root.glob("paper_*_M*/trade_log.jsonl")))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def print_summary(summary: dict) -> None:
    print(f"{summary['path']}")
    print(f"  total:             {summary['total']}")
    print(f"  changed:           {summary['changed']}")
    print(f"  backfilled points: {summary['backfilled_points']}")
    print(f"  WTL changed:       {summary['wtl_changed']}")
    print(f"  quality changed:   {summary['quality_changed']}")
    print(f"  inferred CS:       {summary['inferred_contract_sizes']}")
    if summary.get("backup"):
        print(f"  backup:            {summary['backup']}")
    if summary.get("dry_run"):
        print("  dry-run:           no file written")


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize trade log MFE/MAE scale and capture fields")
    parser.add_argument("--input", type=Path, default=Path("data/trade_log.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--all-known", action="store_true", help="Normalize root and per-bot trade logs")
    parser.add_argument("--write", action="store_true", help="Write changes; default is dry-run")
    parser.add_argument("--contract-size", type=float, default=100.0, help="Fallback contract size")
    args = parser.parse_args()

    paths = discover_logs(args.data_root) if args.all_known else [args.input]
    if not paths:
        print("No trade logs found.")
        return 1

    dry_run = not args.write
    for path in paths:
        summary = normalize_file(path, dry_run=dry_run, fallback_contract_size=args.contract_size)
        print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
