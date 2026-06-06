#!/usr/bin/env python3
"""List or update learned parameters for one scoped bot.

The learned-parameter store is keyed by ``symbol_timeframe_broker``.  This
script intentionally refuses to create a missing instrument key so an operator
cannot accidentally tune M5 while intending to tune M1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.persistence.learned_parameters import LearnedParametersManager

INSTRUMENT_KEY_PARTS = 3

# Known parameter bounds — prevents threshold runaway via CLI (entry_conf ≥ 1.0 → full lockout)
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "entry_confidence_threshold": (0.0, 0.95),
    "exit_confidence_threshold": (0.0, 0.95),
    "feasibility_threshold": (0.0, 0.95),
    "confidence_floor": (0.0, 0.95),
    "wtl_penalty_multiplier": (0.0, 5.0),
    "runway_cal_alpha": (0.0, 1.0),
    "pnl_alignment_multiplier": (0.0, 10.0),
}


def _split_instrument_key(key: str) -> tuple[str, str, str]:
    parts = str(key or "").rsplit("_", 2)
    if len(parts) != INSTRUMENT_KEY_PARTS or not all(parts):
        msg = "instrument must look like SYMBOL_M5_default"
        raise ValueError(msg)
    return parts[0], parts[1], parts[2]


def _instrument_key(args: argparse.Namespace) -> str:
    if args.instrument:
        return args.instrument
    return f"{args.symbol}_{args.timeframe}_{args.broker}"


def _load_manager(path: Path) -> LearnedParametersManager:
    return LearnedParametersManager(persistence_path=path)


def list_parameters(path: Path, instrument_key: str) -> int:
    manager = _load_manager(path)
    instrument = manager.instruments.get(instrument_key)
    if instrument is None:
        return 1

    for name in sorted(instrument.params):
        instrument.params[name]
    return 0


def update_parameter(path: Path, instrument_key: str, param_name: str, value: float) -> int:
    manager = _load_manager(path)
    instrument = manager.instruments.get(instrument_key)
    if instrument is None:
        return 1
    if param_name not in instrument.params:
        return 1

    if param_name in _PARAM_BOUNDS:
        lo, hi = _PARAM_BOUNDS[param_name]
        if not (lo <= value <= hi):
            print(f"ERROR: {param_name}={value} is outside allowed range [{lo}, {hi}]", file=sys.stderr)
            return 2

    symbol, timeframe, broker = _split_instrument_key(instrument_key)
    manager.set_value(symbol, param_name, value, timeframe=timeframe, broker=broker)
    manager.save()
    return 0


def _matching_files(path: Path, include_backups: bool) -> list[Path]:
    if not include_backups:
        return [path]
    return sorted(path.parent.glob(f"{path.name}*"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=Path("data/learned_parameters.json"))
    parser.add_argument("--instrument", help="Full key, e.g. XAUUSD_M5_default")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--broker", default="default")
    parser.add_argument("--param", help="Parameter name to update")
    parser.add_argument("--value", type=float, help="New parameter value")
    parser.add_argument("--list", action="store_true", help="List parameters for the scoped instrument")
    parser.add_argument("--all-files", action="store_true", help="Also update matching backup files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.file
    instrument_key = _instrument_key(args)

    if args.list:
        return list_parameters(path, instrument_key)

    if not args.param or args.value is None:
        return 2

    rc = 0
    for file_path in _matching_files(path, args.all_files):
        if file_path.is_file():
            rc = max(rc, update_parameter(file_path, instrument_key, args.param, args.value))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
