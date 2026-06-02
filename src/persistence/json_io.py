"""Canonical persistence primitives: atomic JSON write and durable JSONL append.

Consolidates the duplicated atomic-write and append-only log helpers that were
previously reimplemented across modules. Existing module-local helpers delegate
here so behaviour stays consistent across the codebase.

- ``save_json_atomic``: write JSON to a temp file in the same directory, then
  ``os.replace`` (atomic on POSIX) onto the target. A crash mid-write leaves the
  original file untouched.
- ``append_jsonl_durable``: append one JSON Lines record with a single
  ``O_APPEND`` write followed by ``fsync`` for durability.
"""

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional


def save_json_atomic(
    path: str | Path,
    data: Any,
    *,
    indent: Optional[int] = None,
    default: Optional[Callable[[Any], Any]] = None,
) -> None:
    """Atomically write *data* as JSON to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=default)
        os.replace(tmp_path, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def append_jsonl_durable(
    path: str | Path,
    entry: dict[str, Any],
    *,
    default: Any = str,
) -> None:
    """Append one JSONL record as a single durable O_APPEND write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(entry, default=default, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(line)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
