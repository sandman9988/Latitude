"""Reset adaptive entry/feasibility thresholds to their schema defaults.

The Phase 1 "bound + decay" fix removes the runaway feedback loops that could
ratchet ``entry_confidence_threshold`` upward (re-reading the mutated persisted
value as its own base) and ``feasibility_threshold`` toward 1.0 (increment-only,
no decay). Persisted values that drifted under the old logic are reset here so the
fixed tuner starts from a sane base.

Usage:
    .venv/bin/python scripts/training/reset_entry_thresholds.py [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.persistence.learned_parameters import LearnedParametersManager

RESET_DEFAULTS = {
    "entry_confidence_threshold": 0.6,
    "feasibility_threshold": 0.5,
}


def _split_key(key: str) -> tuple[str, str, str]:
    symbol, timeframe, broker = key.rsplit("_", 2)
    return symbol, timeframe, broker


def reset_file(path: Path, dry_run: bool) -> int:
    mgr = LearnedParametersManager(persistence_path=path)
    changed = 0
    for key, instrument in mgr.instruments.items():
        symbol, timeframe, broker = _split_key(key)
        for name, default in RESET_DEFAULTS.items():
            param = instrument.params.get(name)
            if param is None or param.value is None:
                continue
            if abs(float(param.value) - default) < 1e-9:
                continue
            print(f"  {key}.{name}: {param.value} -> {default}")
            if not dry_run:
                mgr.set_value(symbol, name, default, timeframe=timeframe, broker=broker)
            changed += 1
    if changed and not dry_run:
        mgr.save()
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    total = 0
    for f in sorted(glob.glob("data/paper_*/learned_parameters.json")):
        path = Path(f)
        print(path)
        total += reset_file(path, args.dry_run)
    print(f"\n{'Would reset' if args.dry_run else 'Reset'} {total} parameter value(s).")


if __name__ == "__main__":
    main()
