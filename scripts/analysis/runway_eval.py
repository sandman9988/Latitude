#!/usr/bin/env python3
"""Runway forecaster evaluation harness.
=====================================
Trains the decoupled :class:`RunwayForecaster` on historical OHLC data and
reports its out-of-sample skill against the legacy runway baseline.

Baseline to beat (measured on live trade history, data/trade_log.jsonl):
    overall corr(predicted_runway_points, realized_mfe_points) ~= 0.19,
    with several high-volume (symbol, timeframe) bots at zero or negative corr.

For each data/history/{SYMBOL}_{TF}.csv this script:
  1. builds ATR-normalized forward favorable-excursion labels (long & short),
  2. builds market-state features per bar for each side,
  3. does a temporal train/test split (no shuffling, no lookahead),
  4. fits the quantile forecaster on the train slice,
  5. reports test corr / pinball / utilization in price (point) units.

Usage:
    .venv/bin/python scripts/analysis/runway_eval.py
    .venv/bin/python scripts/analysis/runway_eval.py --symbol XAUUSD --tf M30
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


def _parse_name(path: Path) -> tuple[str, int]:
    stem = path.stem
    symbol, _, tf = stem.partition("_")
    return symbol, _TF_MINUTES.get(tf, 0)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _assemble(
    bars: list,
    timeframe_minutes: int,
    split: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    lab = build_labels(bars, timeframe_minutes)
    o = np.asarray([b[1] for b in bars], dtype=np.float64)
    h = np.asarray([b[2] for b in bars], dtype=np.float64)
    l = np.asarray([b[3] for b in bars], dtype=np.float64)
    c = lab["close"]
    atr = lab["atr"]
    n = len(c)
    cut = int(n * split)

    def _collect(lo: int, hi: int) -> dict[str, np.ndarray] | None:
        feats: list[np.ndarray] = []
        labels: list[float] = []
        atrs: list[float] = []
        mfes: list[float] = []
        for side, lab_key, mfe_key in ((1, "label_long", "mfe_long"), (-1, "label_short", "mfe_short")):
            fm, idxs = build_feature_matrix(o, h, l, c, atr, side=side)
            mask = (idxs >= lo) & (idxs < hi)
            for row, bi in zip(fm[mask], idxs[mask]):
                if not lab["valid"][bi]:
                    continue
                feats.append(row)
                labels.append(float(lab[lab_key][bi]))
                atrs.append(float(atr[bi]))
                mfes.append(float(lab[mfe_key][bi]))
        if not feats:
            return None
        return {
            "features": np.asarray(feats),
            "labels": np.asarray(labels),
            "atr": np.asarray(atrs),
            "mfe": np.asarray(mfes),
        }

    train = _collect(0, cut)
    test = _collect(cut, n)
    if train is None or test is None:
        return None
    return train, test


def evaluate_file(path: Path, split: float = 0.7, max_bars: int | None = None) -> dict | None:
    symbol, tf_min = _parse_name(path)
    if tf_min == 0:
        return None
    bars = load_csv(path, max_bars=max_bars, timeframe_minutes=tf_min)
    if len(bars) < 500:
        return None
    assembled = _assemble(bars, tf_min, split)
    if assembled is None:
        return None
    train, test = assembled

    model = RunwayForecaster()
    losses = model.fit(train["features"], train["labels"])

    q = model.predict_matrix(test["features"])
    p50_idx = int(np.argmin(np.abs(np.asarray(model.quantiles) - 0.5)))
    pred_norm = q[:, p50_idx]
    pred_price = pred_norm * test["atr"]
    realized_price = test["mfe"]

    corr = _safe_corr(pred_price, realized_price)
    valid_pred = pred_price[pred_price > 1e-9]
    util = (
        float(np.median(realized_price[pred_price > 1e-9] / valid_pred))
        if valid_pred.size
        else float("nan")
    )
    return {
        "symbol": symbol,
        "tf": tf_min,
        "n_train": train["features"].shape[0],
        "n_test": test["features"].shape[0],
        "corr": corr,
        "util_median": util,
        "pinball_q50": losses.get("pinball_q50", float("nan")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--tf", default=None)
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--max-bars", type=int, default=None)
    ap.add_argument("--history-dir", default=str(ROOT / "data" / "history"))
    args = ap.parse_args()

    hist = Path(args.history_dir)
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
            res = evaluate_file(f, split=args.split, max_bars=args.max_bars)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {f.name}: {exc}")
            continue
        if res:
            rows.append(res)

    if not rows:
        print("No evaluable files.")
        return 1

    print(f"\n{'symbol':9s} {'tf':>4s} {'n_train':>8s} {'n_test':>7s} {'corr':>7s} {'util':>7s} {'pin50':>7s}")
    print("-" * 56)
    corrs = []
    for r in sorted(rows, key=lambda x: (x["symbol"], x["tf"])):
        corrs.append(r["corr"])
        print(
            f"{r['symbol']:9s} {r['tf']:>4d} {r['n_train']:>8d} {r['n_test']:>7d} "
            f"{r['corr']:>7.3f} {r['util_median']:>7.2f} {r['pinball_q50']:>7.4f}"
        )
    valid = [c for c in corrs if not np.isnan(c)]
    print("-" * 56)
    print(f"mean corr (new): {np.mean(valid):.3f}   baseline (legacy): 0.190")
    print(f"min corr  (new): {np.min(valid):.3f}   (legacy had negatives)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
