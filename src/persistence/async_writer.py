"""Coalescing background writer for latest-wins snapshot files.

Telemetry/HUD snapshot files (paper_stats, current_position, training_stats,
risk_metrics, order_book, ...) are overwrite-style: only the most recent payload
for a given path matters. Writing them synchronously on the reactor thread adds
per-flush disk latency (temp file + fsync-on-rename) to tick/bar processing.

``AsyncJsonWriter`` accepts payloads on the hot thread (cheap: store under a
lock and signal) and performs the actual serialization + atomic write on a
single daemon thread. Because it is coalescing, repeated submissions for the
same path between flushes collapse to the latest payload, so a slow disk can
never make the queue grow without bound.

This is intended only for recoverable snapshot state. Durability-critical
append logs (trade/decision/transaction audit trails) must NOT use this writer.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

LOG = logging.getLogger(__name__)

WriteFn = Callable[[Path, dict, "int | None"], None]


class AsyncJsonWriter:
    def __init__(self, write_fn: WriteFn, name: str = "async-json-writer") -> None:
        self._write_fn = write_fn
        self._name = name
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[Path, dict, int | None]] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def submit(self, path: Path, payload: dict, indent: int | None = None) -> None:
        """Queue the latest payload for *path*; coalesces with any pending one."""
        with self._lock:
            self._pending[str(path)] = (Path(path), payload, indent)
        self._wake.set()
        if self._thread is None or not self._thread.is_alive():
            # No worker (e.g. not started) — fall back to a synchronous write so
            # callers never silently lose a snapshot.
            self._drain()

    def _drain(self) -> None:
        with self._lock:
            batch = list(self._pending.values())
            self._pending.clear()
        for path, payload, indent in batch:
            try:
                self._write_fn(path, payload, indent)
            except Exception as exc:  # never let a telemetry write kill the writer
                LOG.warning("[ASYNC_WRITE] Failed writing %s: %s", path, exc)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            self._drain()
        self._drain()

    def flush(self) -> None:
        """Synchronously write any pending snapshots."""
        self._drain()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._drain()
        self._thread = None
