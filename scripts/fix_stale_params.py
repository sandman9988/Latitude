#!/usr/bin/env python3
"""Fix stale learned parameters that fell behind constants.py updates.

Fixes:
  - pnl_alignment_multiplier: 0.35 → 1.5 (matches PNL_ALIGNMENT_MULT_DEFAULT)
    max_bound updated to 2.0 to accommodate the new value.

Safe to run while bots are stopped. Uses atomic write + CRC32.
"""
import contextlib
import json
import os
import shutil
import sys
import tempfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

FIXES = {
    "pnl_alignment_multiplier": {
        "old_value": 0.35,
        "new_value": 1.5,
        "new_max_bound": 2.0,
        "reason": "PNL_ALIGNMENT_MULT_DEFAULT updated from 0.35 to 1.5 in reward_shaper.py (2026-04-29)",
    },
}


def _load_envelope(path: Path) -> dict | None:
    with open(path) as f:
        return json.load(f)


def _write_atomic(path: Path, data: dict) -> str:
    json_bytes = json.dumps(data, indent=2).encode("utf-8")
    crc32 = zlib.crc32(json_bytes) & 0xFFFFFFFF
    envelope = {
        "crc32": crc32,
        "timestamp": datetime.now(UTC).isoformat(),
        "version": 1,
        "data": data,
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".lp.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(json.dumps(envelope, indent=2).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        bak = str(path) + f".pre_stale_fix_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.bak"
        shutil.copy2(path, bak)
        shutil.move(tmp, path)
        return f"OK (CRC32: {crc32:08x}, backup: {Path(bak).name})"
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def main() -> int:
    changed_total = 0

    for param_dir in sorted(_REPO.glob("data/paper_*/learned_parameters.json")):
        env = _load_envelope(param_dir)
        if not isinstance(env, dict):
            continue

        data = env.get("data") if "data" in env else env
        if not isinstance(data, dict):
            continue

        instruments = data.get("instruments", {})
        changed_in_file = False

        for inst_key, inst in instruments.items():
            params = inst.get("params", {})
            for pname, fix in FIXES.items():
                if pname not in params:
                    continue
                entry = params[pname]
                if abs(float(entry.get("value", 0)) - fix["old_value"]) < 1e-9:
                    print(f"  {param_dir.parent.name}/{inst_key}: {pname} {fix['old_value']} → {fix['new_value']}")
                    entry["value"] = fix["new_value"]
                    if "new_max_bound" in fix:
                        entry["max_bound"] = fix["new_max_bound"]
                    changed_in_file = True
                    changed_total += 1

        if changed_in_file:
            result = _write_atomic(param_dir, data)
            print(f"    {param_dir.parent.name}/learned_parameters.json: {result}")

    if changed_total == 0:
        print("No stale parameters found. Nothing changed.")
    else:
        print(f"\n{changed_total} parameter(s) updated. Restart bots to pick up changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
