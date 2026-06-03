"""Characterization of the coalescing async snapshot writer (phase 2).

Telemetry/HUD snapshot files are latest-wins overwrites. ``AsyncJsonWriter``
moves the actual disk write off the hot (reactor) thread while guaranteeing
(a) the most recent payload per path is eventually written, (b) repeated
submissions between flushes coalesce to the latest payload, and (c) a write
still happens even if the worker thread was never started.
"""

import json
import threading

import pytest

from src.persistence.async_writer import AsyncJsonWriter


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _plain_write(path, payload, indent=None):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent)


class TestAsyncJsonWriter:
    def test_started_writer_persists_latest_payload(self, tmp_path):
        w = AsyncJsonWriter(_plain_write)
        w.start()
        try:
            target = tmp_path / "snap.json"
            w.submit(target, {"v": 1})
            w.submit(target, {"v": 2})
            w.flush()
            assert _read(target) == {"v": 2}
        finally:
            w.stop()

    def test_coalesces_repeated_submissions(self, tmp_path):
        writes = []
        lock = threading.Lock()

        def _counting(path, payload, indent=None):
            with lock:
                writes.append(dict(payload))
            _plain_write(path, payload, indent)

        w = AsyncJsonWriter(_counting)
        # Not started: submit coalesces in the pending dict before a drain.
        target = tmp_path / "snap.json"
        with w._lock:
            w._pending[str(target)] = (target, {"v": 1}, None)
            w._pending[str(target)] = (target, {"v": 2}, None)
            w._pending[str(target)] = (target, {"v": 3}, None)
        w.flush()
        # Three submissions for one path collapse to a single write of the last.
        assert writes == [{"v": 3}]

    def test_submit_without_worker_falls_back_to_sync_write(self, tmp_path):
        w = AsyncJsonWriter(_plain_write)
        target = tmp_path / "snap.json"
        w.submit(target, {"v": 42})  # never started
        assert _read(target) == {"v": 42}

    def test_write_failure_does_not_propagate(self, tmp_path):
        def _boom(path, payload, indent=None):
            raise OSError("disk full")

        w = AsyncJsonWriter(_boom)
        w.submit(tmp_path / "snap.json", {"v": 1})  # must not raise
        w.flush()

    def test_stop_flushes_pending(self, tmp_path):
        w = AsyncJsonWriter(_plain_write)
        w.start()
        target = tmp_path / "snap.json"
        w.submit(target, {"v": 7})
        w.stop()
        assert _read(target) == {"v": 7}
