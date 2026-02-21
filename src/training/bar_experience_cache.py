#!/usr/bin/env python3
"""
BarExperienceCache
==================
Write raw-bar trade snapshots to a JSONL file so completed trades can be
replayed through the current feature pipeline for offline DDQN training.

Why raw bars instead of computed state vectors?
   Feature engineering evolves (geometry, event features, n_features changes).
   A state vector computed today becomes incompatible when the pipeline changes.
   Storing raw OHLC bars + metadata lets train_offline.py re-derive states with
   the *current* DualPolicy._build_state() at replay time — always compatible.

Schema (per line, JSONL):
    version         int   — format version, increment on schema changes
    ts_recorded     str   — ISO-8601 UTC timestamp of when this was written
    symbol          str   — instrument (e.g. "XAUUSD")
    timeframe_minutes int — bar timeframe in minutes (e.g. 5 for M5)
    regime          str   — regime label at trade close ("TRENDING", "RANGING", etc.)
    trigger_action  int   — 0=NO_ENTRY, 1=LONG, 2=SHORT
    trigger_reward  float — trigger reward signal (σ-normalised or prediction-accuracy)
    capture_reward  float — harvester reward (shaped total, clipped [-2, 2])
    entry_price     float — position entry price
    exit_price      float — position exit price
    pnl_pts         float — realised P&L in raw price points (pnl / lot_value)
    mfe             float — maximum favourable excursion (price points)
    mae             float — maximum adverse excursion (price points)
    was_explore     bool  — whether the entry was an exploration action
    imbalance       float — order-book imbalance at entry
    vpin_z          float — VPIN z-score at entry
    depth_ratio     float — bid+ask depth ratio at entry
    entry_bars      list  — last N bars at entry time [[iso, o, h, l, c], ...]
    exit_bars       list  — last N bars at trade-close time [[iso, o, h, l, c], ...]

Usage (in CTraderFixApp):
    # At startup:
    self._bar_cache = BarExperienceCache(symbol="XAUUSD", timeframe_minutes=5)

    # At on_entry:
    self._bar_cache.snapshot_entry(self.bars)

    # After experience is added (inside _on_position_closed):
    self._bar_cache.record_trade(
        bars=self.bars,
        trigger_action=..., trigger_reward=..., capture_reward=...,
        entry_price=..., exit_price=..., pnl_pts=..., mfe=..., mae=...,
        regime=..., was_explore=...,
        imbalance=..., vpin_z=..., depth_ratio=...,
    )
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SCHEMA_VERSION: int = 1
DEFAULT_BARS_TO_STORE: int = 120   # Enough for all rolling stats (30-bar MA + headroom)
MAX_CACHE_SIZE_MB: float = 500.0   # Soft cap — log warning when exceeded


def _default_cache_path(symbol: str, timeframe_minutes: int) -> str:
    """
    Build a per-instrument cache path.

    Examples
    --------
    >>> _default_cache_path("XAUUSD", 15)
    'data/training_cache_XAUUSD_M15.jsonl'
    >>> _default_cache_path("BTCUSD", 240)
    'data/training_cache_BTCUSD_M240.jsonl'
    """
    safe = symbol.replace("/", "-").replace("\\", "-")
    return f"data/training_cache_{safe}_M{timeframe_minutes}.jsonl"


class BarExperienceCache:
    """
    Thread-unsafe but exception-safe JSONL writer for bar-level trade experiences.

    Writes one line per completed trade. Errors are always swallowed so the
    live bot is never disrupted by cache I/O failures.

    Cache file defaults to ``data/training_cache_{SYMBOL}_M{TF}.jsonl`` so
    that each instrument/timeframe pair accumulates its own experience file
    and multiple bot processes never interleave each other's data.
    Pass an explicit ``cache_file`` path to override.
    """

    def __init__(
        self,
        symbol: str = "UNKNOWN",
        timeframe_minutes: int = 5,
        cache_file: str | None = None,
        bars_to_store: int = DEFAULT_BARS_TO_STORE,
        enabled: bool = True,
    ) -> None:
        self.symbol = symbol
        self.timeframe_minutes = timeframe_minutes
        # Derive per-instrument path when caller does not supply an explicit one
        _resolved = cache_file if cache_file is not None else _default_cache_path(symbol, timeframe_minutes)
        self.cache_file = Path(_resolved)
        self.bars_to_store = bars_to_store
        self.enabled = enabled

        # Snapshot captured at entry time (set by snapshot_entry)
        self._entry_bars_snapshot: list[list] | None = None

        if enabled:
            self._ensure_dir()
            LOG.info(
                "[CACHE] BarExperienceCache init: %s tf=%dm file=%s",
                symbol, timeframe_minutes, self.cache_file,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def snapshot_entry(self, bars: deque) -> None:
        """
        Capture the current bars window at trade-entry time.

        Call this inside on_entry() *after* the position is confirmed so the
        snapshot reflects the market state the TriggerAgent actually saw.
        """
        if not self.enabled:
            return
        try:
            self._entry_bars_snapshot = self._serialise_bars(bars)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("[CACHE] snapshot_entry failed: %s", exc)
            self._entry_bars_snapshot = None

    def record_trade(
        self,
        *,
        bars: deque,
        trigger_action: int,
        trigger_reward: float,
        capture_reward: float,
        entry_price: float,
        exit_price: float,
        pnl_pts: float,
        mfe: float,
        mae: float,
        regime: str = "UNKNOWN",
        was_explore: bool = False,
        imbalance: float = 0.0,
        vpin_z: float = 0.0,
        depth_ratio: float = 1.0,
    ) -> None:
        """
        Append one trade record to the JSONL cache.

        Call this immediately after adding both trigger and harvester experiences
        to the live PER buffer so all labels are ready.
        """
        if not self.enabled:
            return

        try:
            exit_bars = self._serialise_bars(bars)
            entry_bars = self._entry_bars_snapshot or []

            record: dict[str, Any] = {
                "version": SCHEMA_VERSION,
                "ts_recorded": datetime.now(UTC).isoformat(),
                "symbol": self.symbol,
                "timeframe_minutes": self.timeframe_minutes,
                "regime": str(regime),
                "trigger_action": int(trigger_action),
                "trigger_reward": float(trigger_reward),
                "capture_reward": float(capture_reward),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pnl_pts": float(pnl_pts),
                "mfe": float(mfe),
                "mae": float(mae),
                "was_explore": bool(was_explore),
                "imbalance": float(imbalance),
                "vpin_z": float(vpin_z),
                "depth_ratio": float(depth_ratio),
                "entry_bars": entry_bars,
                "exit_bars": exit_bars,
            }

            self._append(record)

            # Reset entry snapshot so a missed snapshot_entry is obvious at next trade
            self._entry_bars_snapshot = None

        except Exception as exc:  # noqa: BLE001
            LOG.warning("[CACHE] record_trade failed (trade not cached): %s", exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _serialise_bars(self, bars: deque) -> list[list]:
        """Convert the last *bars_to_store* bars to a JSON-serialisable list."""
        tail = list(bars)[-self.bars_to_store :]
        result = []
        for b in tail:
            t, o, h, l, c = b[0], b[1], b[2], b[3], b[4]
            ts = t.isoformat() if hasattr(t, "isoformat") else str(t)
            result.append([ts, float(o), float(h), float(l), float(c)])
        return result

    def _append(self, record: dict[str, Any]) -> None:
        """Write one JSON line; check soft size cap."""
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with open(self.cache_file, "a", encoding="utf-8") as fh:
            fh.write(line)

        # Soft cap check (stat is cheap for a warning-once path)
        try:
            size_mb = self.cache_file.stat().st_size / (1024 * 1024)
            if size_mb > MAX_CACHE_SIZE_MB:
                LOG.warning(
                    "[CACHE] training_cache.jsonl is %.1f MB — consider archiving.", size_mb
                )
        except OSError:
            pass

    def _ensure_dir(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOG.warning("[CACHE] Could not create cache directory: %s", exc)

    # ── Introspection ─────────────────────────────────────────────────────────

    def record_count(self) -> int:
        """Return number of records written (counts newlines — fast)."""
        if not self.cache_file.exists():
            return 0
        try:
            with open(self.cache_file, "rb") as fh:
                return sum(1 for _ in fh)
        except OSError:
            return 0

    def size_mb(self) -> float:
        """Return file size in MB."""
        try:
            return self.cache_file.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0
