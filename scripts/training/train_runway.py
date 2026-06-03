#!/usr/bin/env python3
"""Fit and persist a RunwayForecaster per (symbol, timeframe) from history.

For each data/history/{SYMBOL}_{TF}.csv this script:
  1. builds ATR-normalized forward favorable-excursion labels (long & short),
  2. builds market-state features per bar for each side,
  3. fits the quantile forecaster on the full series,
  4. saves the model to data/paper_{SYMBOL}_{TF}/runway_forecaster.json.

Usage:
    .venv/bin/python scripts/training/train_runway.py
    .venv/bin/python scripts/training/train_runway.py --symbol XAUUSD --tf M30
    .venv/bin/python scripts/training/train_runway.py --max-bars 60000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.runway_forecaster import (  # noqa: E402
    RunwayForecaster,
    build_feature_matrix,
)
from src.features.runway_labels import build_labels  # noqa: E402
from src.training.historical_loader import load_csv  # noqa: E402

_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "M60": 60, "M240": 240}


def _parse_name(path: Path) -> tuple[str, str, int]:
    stem = path.stem
    symbol, _, tf = stem.partition("_")
    return symbol, tf, _TF_MINUTES.get(tf, 0)


def _assemble(bars: list, timeframe_minutes: int) -> dict[str, np.ndarray] | None:
    lab = build_labels(bars, timeframe_minutes)
    o = np.asarray([b[1] for b in bars], dtype=np.float64)
    h = np.asarray([b[2] for b in bars], dtype=np.float64)
    l = np.asarray([b[3] for b in bars], dtype=np.float64)
    c = lab["close"]
    atr = lab["atr"]

    feats: list[np.ndarray] = []
    labels: list[float] = []
    for side, lab_key in ((1, "label_long"), (-1, "label_short")):
        fm, idxs = build_feature_matrix(o, h, l, c, atr, side=side)
        for row, bi in zip(fm, idxs):
            if not lab["valid"][bi]:
                continue
            feats.append(row)
            labels.append(float(lab[lab_key][bi]))
    if not feats:
        return None
    return {"features": np.asarray(feats), "labels": np.asarray(labels)}


def train_file(
    path: Path,
    out_dir: Path,
    max_bars: int | None = None,
) -> dict | None:
    symbol, tf_label, tf_min = _parse_name(path)
    if tf_min == 0:
        return None
    bars = load_csv(path, max_bars=max_bars, timeframe_minutes=tf_min)
    if len(bars) < 500:
        print(f"[skip] {path.name}: only {len(bars)} bars")
        return None
    assembled = _assemble(bars, tf_min)
    if assembled is None:
        print(f"[skip] {path.name}: no valid labels")
        return None

    model = RunwayForecaster()
    losses = model.fit(assembled["features"], assembled["labels"])

    dest_dir = out_dir / f"paper_{symbol}_{tf_label}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "runway_forecaster.json"
    model.save(dest)

    return {
        "symbol": symbol,
        "tf": tf_label,
        "n": assembled["features"].shape[0],
        "use_residual": model.use_residual,
        "pinball_q50": losses.get("pinball_q50", float("nan")),
        "dest": str(dest.relative_to(ROOT)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--tf", default=None)
    ap.add_argument("--max-bars", type=int, default=None)
    ap.add_argument("--history-dir", default=str(ROOT / "data" / "history"))
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    hist = Path(args.history_dir)
    out_dir = Path(args.out_dir)
    files = sorted(hist.glob("*.csv"))
    if args.symbol:
        files = [f for f in files if f.stem.startswith(args.symbol)]
    if args.tf:
        files = [f for f in files if f.stem.endswith("_" + args.tf)]
    if not files:
        print(f"No history CSVs found in {hist}")
        return 1

    rows = []
    for f in files:
        try:
            res = train_file(f, out_dir, max_bars=args.max_bars)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {f.name}: {exc}")
            continue
        if res:
            rows.append(res)
            print(
                f"[ok] {res['symbol']:9s} {res['tf']:>4s} "
                f"n={res['n']:>7d} residual={str(res['use_residual']):>5s} "
                f"pin50={res['pinball_q50']:.4f} -> {res['dest']}"
            )

    if not rows:
        print("No models trained.")
        return 1
    print(f"\nTrained {len(rows)} forecaster(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
