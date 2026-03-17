"""Centralized trade_log.jsonl reader.

All modules that need to parse ``data/trade_log.jsonl`` should use this module
instead of hand-rolling their own line-by-line JSON parsing.  This avoids
6+ copies of the same pattern and ensures consistent error handling.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

LOG = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/trade_log.jsonl")


def read_all_trades(path: Path | str = _DEFAULT_PATH) -> list[dict]:
    """Read every valid JSONL line from the trade log.

    Corrupt lines are silently skipped (logged at DEBUG).

    Returns:
        List of trade dicts in file order (oldest first).
    """
    path = Path(path)
    if not path.exists():
        return []
    trades: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, 1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    trades.append(json.loads(stripped))
                except json.JSONDecodeError:
                    LOG.debug("[TRADE_LOG] Skipping corrupt line %d in %s", line_no, path)
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
            raw = fh.read(read_bytes).decode("utf-8", errors="replace")
    except OSError as exc:
        LOG.warning("[TRADE_LOG] Could not tail-read %s: %s", path, exc)
        return []

    lines = [ln for ln in raw.splitlines() if ln.strip()][-max_lines:]
    trades: list[dict] = []
    for line in lines:
        try:
            rec = json.loads(line)
            if rec.get("exit_time") and rec.get("entry_time"):
                trades.append(rec)
        except json.JSONDecodeError:
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
        if not self._path.exists():
            self._trades = []
            self._mtime = 0.0
            return
        try:
            current_mtime = self._path.stat().st_mtime
        except OSError:
            return
        if current_mtime == self._mtime:
            return
        self._mtime = current_mtime
        self._trades = read_all_trades(self._path)

    def invalidate(self) -> None:
        """Force a re-read on the next access."""
        self._mtime = 0.0
