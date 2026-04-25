"""Centralized trade_log.jsonl reader.

All modules that need to parse ``data/trade_log.jsonl`` should use this module
instead of hand-rolling their own line-by-line JSON parsing.  This avoids
6+ copies of the same pattern and ensures consistent error handling.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
from pathlib import Path

LOG = logging.getLogger(__name__)

_DEFAULT_PATH = Path(os.environ.get("CTRADER_DATA_DIR", "data")) / "trade_log.jsonl"


def read_all_trades(path: Path | str = _DEFAULT_PATH) -> list[dict]:
    """Read every valid JSONL line from the trade log.

    Uses file locking to prevent corruption during concurrent writes.
    Corrupt lines are logged at WARNING level.

    Returns:
        List of trade dicts in file order (oldest first).
    """
    path = Path(path)
    if not path.exists():
        return []
    trades: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            # Acquire shared lock (non-blocking readers, exclusive writers)
            locked = False
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                locked = True
            except OSError:
                # Lock failed, file is being written - proceed with partial read
                LOG.debug("[TRADE_LOG] Could not acquire lock on %s, using partial read", path)

            try:
                for line_no, raw in enumerate(fh, 1):
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        trades.append(json.loads(stripped))
                    except json.JSONDecodeError as e:
                        LOG.warning(
                            "[TRADE_LOG] Corrupted line %d in %s (recovered): %s",
                            line_no,
                            path,
                            e,
                        )
            finally:
                if locked:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        LOG.warning("[TRADE_LOG] Could not read %s: %s", path, exc)
    return trades


def read_recent_trades(
    path: Path | str = _DEFAULT_PATH,
    max_lines: int = 50,
    buf_size: int = 64 * 1024,
) -> list[dict]:
    """Read up to *max_lines* completed trades from the tail of the log.

    Uses a seek-from-end strategy to avoid reading the entire file.
    Only returns records that have both ``entry_time`` and ``exit_time``.
    Uses robust error handling for encoding issues.

    Args:
        path: Path to the JSONL file.
        max_lines: Maximum number of trade records to return.
        buf_size: How many bytes to read from the end of the file.

    Returns:
        List of completed trade dicts (oldest-first within the window).
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            file_size = fh.tell()
            if file_size == 0:
                return []
            read_bytes = min(file_size, buf_size)
            fh.seek(-read_bytes, 2)
            raw_bytes = fh.read(read_bytes)

            # Decode with proper error handling
            try:
                raw = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                LOG.warning("[TRADE_LOG] Unicode decode error in tail-read %s: %s (skipping)", path, e)
                return []

    except OSError as exc:
        LOG.warning("[TRADE_LOG] Could not tail-read %s: %s", path, exc)
        return []

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()][-max_lines:]
    trades: list[dict] = []
    for line_no, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
            # Validate record structure
            if not isinstance(rec, dict):
                LOG.debug("[TRADE_LOG] Non-dict record in tail: type=%s", type(rec))
                continue
            if rec.get("exit_time") and rec.get("entry_time"):
                trades.append(rec)
        except json.JSONDecodeError as e:
            LOG.debug("[TRADE_LOG] Corrupt JSON in tail-read line %d: %s", line_no, e)
            continue
    return trades


class CachedTradeLogReader:
    """Trade log reader with mtime-based caching.

    Re-parses the file only when its on-disk modification time changes.
    Suitable for hot loops like HUD refresh (1 Hz).
    """

    def __init__(self, path: Path | str = _DEFAULT_PATH) -> None:
        self._path = Path(path)
        self._mtime: float = 0.0
        self._trades: list[dict] = []

    @property
    def trades(self) -> list[dict]:
        """Return cached trades, re-parsing only if the file changed."""
        self._refresh()
        return self._trades

    def _refresh(self) -> None:
        """Refresh cache only if file has changed."""
        if not self._path.exists():
            # File deleted - clear cache but note it
            if self._mtime != 0.0:
                LOG.debug("[TRADE_LOG] File disappeared: %s", self._path)
            self._trades = []
            self._mtime = 0.0
            return
        try:
            current_mtime = self._path.stat().st_mtime
        except OSError as e:
            LOG.debug("[TRADE_LOG] Could not stat file (cache unchanged): %s", e)
            return  # Leave cache as-is, will retry next time

        # Check if file was modified
        if current_mtime == self._mtime:
            return  # Cache is current

        # File has changed - reload
        try:
            new_trades = read_all_trades(self._path)
            self._mtime = current_mtime
            self._trades = new_trades
            LOG.debug("[TRADE_LOG] Cache refreshed, %d trades loaded", len(new_trades))
        except Exception as e:
            LOG.error("[TRADE_LOG] Failed to reload trades: %s (cache unchanged)", e)
            # Leave cache as-is, will retry next time

    def invalidate(self) -> None:
        """Force a re-read on the next access."""
        self._mtime = 0.0


def read_all_trades_validated(path: Path | str = _DEFAULT_PATH) -> tuple[list[dict], list[str]]:
    """Read trades with validation of chronological order.

    Returns:
        (trades, warnings) - list of validated trades and list of warning strings
    """
    trades = read_all_trades(path)
    warnings: list[str] = []

    last_entry_time: float | None = None
    seen_ids: set[str] = set()

    validated_trades: list[dict] = []
    for i, trade in enumerate(trades):
        # Check required fields
        if not trade.get("entry_time") or not trade.get("exit_time"):
            warnings.append(f"Trade {i}: Missing entry/exit_time")
            continue

        # Check trade_id uniqueness
        trade_id = trade.get("trade_id")
        if trade_id:
            if trade_id in seen_ids:
                warnings.append(f"Trade {i}: Duplicate trade_id {trade_id}")
                continue
            seen_ids.add(trade_id)

        # Check timestamp ordering
        try:
            entry_time = float(trade["entry_time"])
            if last_entry_time is not None and entry_time < last_entry_time:
                warnings.append(f"Trade {i}: Out of order entry_time {entry_time} < {last_entry_time}")
                continue
            last_entry_time = entry_time
        except (ValueError, TypeError):
            warnings.append(f"Trade {i}: Invalid entry_time format")
            continue

        validated_trades.append(trade)

    if warnings:
        LOG.warning("[TRADE_LOG] Validation found %d issues", len(warnings))

    return validated_trades, warnings
