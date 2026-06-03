"""Reset the hub's self-calibrating runway bias EMAs after the forecaster cutover.

The legacy Q->runway predictor over-predicted, so the hub accumulated a stale
positive ``runway_delta_ema`` (bias correction) and a low ``runway_accuracy_ema``.
The new calibrated forecaster starts unbiased, so these EMAs must be zeroed so the
self-adapting layer begins as a near-no-op and re-learns from clean signal.

Usage:
    .venv/bin/python scripts/training/reset_runway_bias.py [--dry-run]
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

RESET_PARAMS = ("runway_delta_ema", "runway_accuracy_ema")


def _split_key(key: str) -> tuple[str, str, str]:
    symbol, timeframe, broker = key.rsplit("_", 2)
    return symbol, timeframe, broker


def reset_file(path: Path, dry_run: bool) -> int:
    mgr = LearnedParametersManager(persistence_path=path)
    changed = 0
    for key, instrument in mgr.instruments.items():
        symbol, timeframe, broker = _split_key(key)
        for name in RESET_PARAMS:
            param = instrument.params.get(name)
            if param is None or param.value is None:
                continue
            if abs(float(param.value)) < 1e-12:
                continue
            print(f"  {key}.{name}: {param.value} -> 0.0")
            if not dry_run:
                mgr.set_value(symbol, name, 0.0, timeframe=timeframe, broker=broker)
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
