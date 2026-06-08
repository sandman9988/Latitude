"""Centralized trade_log.jsonl reader.

All modules that need to parse ``data/trade_log.jsonl`` should use this module
instead of hand-rolling their own line-by-line JSON parsing.  This avoids
6+ copies of the same pattern and ensures consistent error handling.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

_DEFAULT_PATH = Path(os.environ.get("CTRADER_DATA_DIR", "data")) / "trade_log.jsonl"


def read_all_trades(path: Path | str = _DEFAULT_PATH) -> list[dict[str, Any]]:
    """Read every valid JSONL line from the trade log.

    Uses file locking to prevent corruption during concurrent writes.
    Corrupt lines are logged at WARNING level.

    Returns:
        List of trade dicts in file order (oldest first).

    """
    path = Path(path)
    if not path.exists():
        return []
    trades: list[dict[str, Any]] = []
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


class CachedTradeLogReader:
    """Trade log reader with mtime-based caching.

    Re-parses the file only when its on-disk modification time changes.
    Suitable for hot loops like HUD refresh (1 Hz).
    """

    def __init__(self, path: Path | str = _DEFAULT_PATH) -> None:
        self._path: Path = Path(path)
        self._mtime: float = 0.0
        self._trades: list[dict[str, Any]] = []

    @property
    def trades(self) -> list[dict[str, Any]]:
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
            LOG.exception("[TRADE_LOG] Failed to reload trades: %s (cache unchanged)", e)
            # Leave cache as-is, will retry next time

    def invalidate(self) -> None:
        """Force a re-read on the next access."""
        self._mtime = 0.0


