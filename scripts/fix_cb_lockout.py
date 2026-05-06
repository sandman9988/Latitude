#!/usr/bin/env python3
"""Atomic fix for stale CB lockouts and runaway entry thresholds.

Safe to run while bots are live. Uses temp-file + rename (atomic) with backups.

What it fixes (all restorable by self-heal after):
  1. circuit_breakers.json — clears stale return history and trip state.
     Keeps all thresholds so the CB can re-trip on current data.
  2. learned_parameters.json — resets entry_confidence_threshold back to 0.6
     baseline for bots where the risk-tuner feedback loop pushed it to ≥ 0.7.
     Also resets feasibility_threshold for XAUUSD_M5 (was 1.0, impossible gate).

What it does NOT touch:
  - CB threshold values (Sortino ratio, kurtosis, consecutive loss limits)
  - confidence_floor (managed by performance analyzer)
  - exit_confidence_threshold
  - Any other learned param
"""
import json
import os
import shutil
import tempfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_DATA = _REPO / "data"

# Bots to process
BOTS = [
    "XAUUSD_M1", "XAUUSD_M5", "XAUUSD_M15", "XAUUSD_M30", "XAUUSD_M60", "XAUUSD_M240",
    "BTCUSD_M1", "BTCUSD_M5", "BTCUSD_M15", "BTCUSD_M30", "BTCUSD_M60", "BTCUSD_M240",
]

# entry_confidence_threshold baseline — risk tuner re-adjusts from here
ENTRY_CONF_BASELINE = 0.6
ENTRY_CONF_RESET_IF_ABOVE = 0.65  # only reset bots that drifted well above baseline

# feasibility_threshold reset only for the impossible-gate case
FEASIBILITY_IMPOSSIBLE_THRESHOLD = 0.95  # reset if >= this
FEASIBILITY_RESET_TO = 0.5

_TS = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp file + rename; preserves permissions."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".fix.", suffix=".tmp")
    try:
        payload = json.dumps(data, indent=2)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        shutil.copy2(path, str(path) + f".pre_fix_{_TS}.bak")
        shutil.move(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_lp(path: Path, data: dict) -> None:
    """Write learned_parameters.json atomically with CRC32 envelope."""
    json_bytes = json.dumps(data, indent=2).encode("utf-8")
    crc32 = zlib.crc32(json_bytes) & 0xFFFFFFFF
    envelope = {
        "crc32": crc32,
        "timestamp": datetime.now(UTC).isoformat(),
        "version": 1,
        "data": data,
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".fix.", suffix=".tmp")
    try:
        payload = json.dumps(envelope, indent=2)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        shutil.copy2(path, str(path) + f".pre_fix_{_TS}.bak")
        shutil.move(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def fix_circuit_breakers(bot: str, bot_dir: Path) -> bool:
    path = bot_dir / "circuit_breakers.json"
    if not path.exists():
        return False

    with open(path) as f:
        cb = json.load(f)

    changed = False
    cleared = []

    for cb_name in ("sortino", "kurtosis", "drawdown", "consecutive_losses"):
        entry = cb.get(cb_name)
        if not isinstance(entry, dict):
            continue

        had_trip = entry.get("is_tripped", False)
        had_returns = bool(entry.get("returns"))
        had_readings = bool(entry.get("readings"))
        had_losses = entry.get("consecutive_losses", 0) > 0

        if had_trip or had_returns or had_readings or had_losses:
            entry["is_tripped"] = False
            entry["trip_time"] = None
            entry["trip_reason"] = ""
            entry["trip_value"] = 0.0
            if "returns" in entry:
                entry["returns"] = []
            if "readings" in entry:
                entry["readings"] = []
            if "consecutive_losses" in entry:
                entry["consecutive_losses"] = 0
            changed = True
            status = []
            if had_trip:
                status.append("tripped")
            if had_returns or had_readings:
                status.append("returns cleared")
            if had_losses:
                status.append("consec_losses cleared")
            cleared.append(f"{cb_name}({', '.join(status)})")

    if not changed:
        return False

    _atomic_write_json(path, cb)
    print(f"  CB  {', '.join(cleared)}")
    return True


def fix_learned_params(bot: str, bot_dir: Path) -> bool:
    path = bot_dir / "learned_parameters.json"
    if not path.exists():
        return False

    with open(path) as f:
        envelope = json.load(f)

    # Support both plain dict and CRC32-envelope formats
    if "data" in envelope and isinstance(envelope["data"], dict):
        data = envelope["data"]
        is_envelope = True
    else:
        data = envelope
        is_envelope = False

    instruments = data.get("instruments", {})
    changed = False
    changes = []

    for inst_key, inst in instruments.items():
        if not inst_key.startswith(bot.replace("-", "_")):
            continue
        params = inst.get("params", {})

        # Fix 1: entry_confidence_threshold runaway
        ect = params.get("entry_confidence_threshold")
        if ect and isinstance(ect, dict):
            cur = float(ect.get("value", 0))
            if cur >= ENTRY_CONF_RESET_IF_ABOVE:
                ect["value"] = ENTRY_CONF_BASELINE
                # Sync bounds so the tuner has room to re-adjust upward
                ect["min_bound"] = min(float(ect.get("min_bound", 0.3)), ENTRY_CONF_BASELINE)
                ect["max_bound"] = max(float(ect.get("max_bound", 0.9)), ENTRY_CONF_BASELINE + 0.30)
                changed = True
                changes.append(f"entry_conf_thr {cur:.3f}→{ENTRY_CONF_BASELINE}")

        # Fix 2: feasibility_threshold impossible gate
        ft = params.get("feasibility_threshold")
        if ft and isinstance(ft, dict):
            cur = float(ft.get("value", 0))
            if cur >= FEASIBILITY_IMPOSSIBLE_THRESHOLD:
                ft["value"] = FEASIBILITY_RESET_TO
                changed = True
                changes.append(f"feasibility_thr {cur:.3f}→{FEASIBILITY_RESET_TO}")

    if not changed:
        return False

    if is_envelope:
        _atomic_write_lp(path, data)
    else:
        _atomic_write_json(path, data)

    print(f"  LP  {', '.join(changes)}")
    return True


def main() -> None:
    print(f"fix_cb_lockout.py — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Backups tagged: pre_fix_{_TS}\n")

    total_cb = 0
    total_lp = 0

    for bot in BOTS:
        bot_dir = _DATA / f"paper_{bot}"
        if not bot_dir.is_dir():
            continue

        cb_changed = False
        lp_changed = False

        print(f"{bot}:")
        cb_changed = fix_circuit_breakers(bot, bot_dir)
        lp_changed = fix_learned_params(bot, bot_dir)

        if not cb_changed and not lp_changed:
            print("  (nothing to fix)")

        total_cb += cb_changed
        total_lp += lp_changed

    print(f"\nDone — {total_cb} CB files reset, {total_lp} LP files reset.")
    print("Bots will rebuild CB history from live trades and re-adjust thresholds naturally.")


if __name__ == "__main__":
    main()
