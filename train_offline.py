#!/usr/bin/env python3
"""
train_offline.py
================
Parallelised offline DDQN trainer.

Spawns one worker process per (symbol, timeframe) job, trains each
independently, then copies the best-by-ZOmega weights per symbol to
data/checkpoints/best/.

Usage
-----
    python3 train_offline.py [OPTIONS] FILE_OR_DIR ...

    FILE_OR_DIR can be:
      - A CSV file:           XAUUSD_M5.csv
      - A JSONL cache:        data/training_cache.jsonl
      - A directory:          data/history/  (scans for *.csv / *.jsonl)

Options
-------
    --symbols   SYM [SYM ...]   Only process these symbols (default: all detected)
    --timeframes TF [TF ...]    Timeframes to train  e.g. M1 M5 M15 H1
                                (default: auto-detect from filenames)
    --workers   N               Parallel worker processes (default: CPU count)
    --checkpoint-dir  PATH      Where to save weights (default: data/checkpoints)
    --train-split  0.8          Fraction of bars for training (default: 0.80)
    --train-every  N            bar steps between gradient updates (default: 4)
    --max-bars    N             Cap on bars per job  (default: unlimited)
    --dry-run                   Parse + plan but don't train

Examples
--------
    # Train on all CSVs in data/history/
    python3 train_offline.py data/history/

    # XAUUSD M5 + H1 only, 4 workers
    python3 train_offline.py data/history/XAUUSD_M5.csv data/history/XAUUSD_H1.csv \\
        --workers 4

    # Replay live-captured experience cache
    python3 train_offline.py data/training_cache.jsonl --timeframes M5
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("train_offline")

# HUD-visible status file — written throughout the run so progress can be
# monitored live in the Training tab without parsing the log file.
_STATUS_PATH = Path("data/offline_training_status.json")

# Universe registry — shared with run_universe.py; records trained instruments
# and their current pipeline stage (UNTRAINED → OFFLINE_TRAINING → PAPER → …)
_UNIVERSE_PATH = Path("data/universe.json")

_STAGE_ORDER = ["UNTRAINED", "OFFLINE_TRAINING", "PAPER", "MICRO", "LIVE"]


def _write_status(data: dict) -> None:
    """Atomically write offline training status for HUD consumption."""
    _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATUS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as _f:
        json.dump(data, _f, indent=2)
    tmp.replace(_STATUS_PATH)


def _register_universe(
    symbol: str,
    timeframe_minutes: int,
    z_omega: float,
    weights_path: str,
) -> None:
    """
    Promote a successfully trained instrument to PAPER stage in
    data/universe.json.

    Safe to call concurrently — uses atomic tmp-file rename.
    Never downgrades an instrument already at PAPER or above.
    """
    _UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry: dict = {"version": 1, "instruments": {}}
    if _UNIVERSE_PATH.exists():
        try:
            with open(_UNIVERSE_PATH) as _f:
                registry = json.load(_f)
        except Exception:
            pass

    instruments = registry.setdefault("instruments", {})
    existing = instruments.get(symbol, {})
    current_stage = existing.get("stage", "UNTRAINED")
    current_idx = (
        _STAGE_ORDER.index(current_stage)
        if current_stage in _STAGE_ORDER
        else 0
    )

    if current_idx <= _STAGE_ORDER.index("OFFLINE_TRAINING"):
        instruments[symbol] = {
            **existing,
            "stage":             "PAPER",
            "timeframe_minutes": timeframe_minutes,
            "z_omega":           z_omega,
            "weights_path":      weights_path,
            "promoted_at":       datetime.now(timezone.utc).isoformat(),
            "paper_pid":         None,
            "paper_started_at":  None,
        }
        tmp = _UNIVERSE_PATH.with_suffix(".tmp")
        with open(tmp, "w") as _f:
            json.dump(registry, _f, indent=2)
        tmp.replace(_UNIVERSE_PATH)
        LOG.info(
            "[UNIVERSE] %s M%d → PAPER  (ZOmega=%.4f)  Run: python3 run_universe.py",
            symbol, timeframe_minutes, z_omega,
        )
    else:
        LOG.info(
            "[UNIVERSE] %s already at stage %s — not demoting",
            symbol, current_stage,
        )

# ── Import training modules ────────────────────────────────────────────────────
# Deferred to avoid importing torch/numpy before fork on some platforms
_TF_PATTERN = re.compile(r"[_\-](M15|M30|M5|M1|H1|H4|H12|D1|W1)(?!\d)", re.IGNORECASE)
_SYM_PATTERN = re.compile(r"^([A-Z]{3,8}(?:[A-Z]{3})?)", re.IGNORECASE)

# ── Job descriptor ─────────────────────────────────────────────────────────────

@dataclass
class Job:
    symbol: str
    timeframe_minutes: int
    bars_file: Path
    file_format: str   # "csv" or "jsonl"


# ── Worker function (runs in child process) ────────────────────────────────────

def _run_job(
    symbol: str,
    timeframe_minutes: int,
    bars_file: str,
    file_format: str,
    checkpoint_dir: str,
    train_split: float,
    train_every: int,
    max_bars: int | None,
) -> dict[str, Any]:
    """
    Child-process entry point.  Returns a dict (not a TrainResult) so it
    can be pickled cleanly across the process boundary.
    """
    import logging as _log
    _log.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s][%(process)d] %(name)s: %(message)s")
    logger = _log.getLogger("train_offline.worker")

    from src.training.historical_loader import load_csv, load_jsonl_cache
    from src.training.offline_trainer import OfflineTrainer

    label = f"{symbol}_M{timeframe_minutes}"
    logger.info("[WORKER] Starting %s from %s", label, bars_file)

    try:
        if file_format == "jsonl":
            bars = load_jsonl_cache(bars_file, max_bars=max_bars)
        else:
            bars = load_csv(bars_file, max_bars=max_bars, timeframe_minutes=timeframe_minutes)
    except Exception as exc:
        logger.error("[WORKER] %s: failed to load bars: %s", label, exc)
        return {
            "symbol": symbol, "timeframe_minutes": timeframe_minutes,
            "z_omega": 0.0, "train_trades": 0, "val_trades": 0,
            "total_train_steps": 0, "elapsed_s": 0.0,
            "weights_path": "", "error": str(exc),
        }

    trainer = OfflineTrainer(
        symbol=symbol,
        timeframe_minutes=timeframe_minutes,
        bars=bars,
        checkpoint_dir=checkpoint_dir,
        train_split=train_split,
        train_every=train_every,
    )
    result = trainer.run()
    return {
        "symbol":              result.symbol,
        "timeframe_minutes":   result.timeframe_minutes,
        "z_omega":             result.z_omega,
        "train_trades":        result.train_trades,
        "val_trades":          result.val_trades,
        "total_train_steps":   result.total_train_steps,
        "elapsed_s":           result.elapsed_s,
        "weights_path":        result.weights_path,
        "error":               result.error,
    }


# ── Job discovery ──────────────────────────────────────────────────────────────

_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
               "H1": 60, "H4": 240, "H12": 720, "D1": 1440, "W1": 10080}


def _detect_tf_minutes(filename: str) -> int | None:
    m = _TF_PATTERN.search(filename)
    if m:
        return _TF_MINUTES.get(m.group(1).upper())
    return None


def _detect_symbol(filename: str) -> str | None:
    stem = Path(filename).stem.upper()
    m = _SYM_PATTERN.match(stem)
    return m.group(1) if m else None


def discover_jobs(
    paths: list[str],
    symbol_filter: list[str] | None = None,
    tf_filter: list[str] | None = None,
) -> list[Job]:
    """Expand paths (files + directories) into a list of Job descriptors."""
    tf_minutes_filter = (
        {_TF_MINUTES[t.upper()] for t in tf_filter if t.upper() in _TF_MINUTES}
        if tf_filter else None
    )
    sym_filter_upper = {s.upper() for s in symbol_filter} if symbol_filter else None

    found: list[Job] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files = sorted(p.glob("*.csv")) + sorted(p.glob("*.jsonl"))
        elif p.exists():
            files = [p]
        else:
            LOG.warning("Path not found: %s", raw)
            continue

        for f in files:
            fmt = "jsonl" if f.suffix.lower() == ".jsonl" else "csv"
            sym = _detect_symbol(f.name)
            tf  = _detect_tf_minutes(f.name)

            if sym is None:
                LOG.warning("Could not detect symbol from filename: %s — skipping", f.name)
                continue

            if fmt == "jsonl" and tf is None:
                LOG.warning(
                    "JSONL file %s has no TF in filename; will use --timeframes or skip", f.name
                )

            if tf_minutes_filter:
                if tf is None and tf_minutes_filter:
                    # JSONL without TF in name: create one job per requested TF
                    for tfm in tf_minutes_filter:
                        if sym_filter_upper is None or sym.upper() in sym_filter_upper:
                            found.append(Job(sym.upper(), tfm, f, fmt))
                    continue
                if tf not in tf_minutes_filter:
                    continue

            if sym_filter_upper and sym.upper() not in sym_filter_upper:
                continue

            if tf is None:
                LOG.warning("Could not detect timeframe for %s — skipping", f.name)
                continue

            found.append(Job(sym.upper(), tf, f, fmt))

    # Deduplicate
    seen = set()
    unique = []
    for j in found:
        key = (j.symbol, j.timeframe_minutes, str(j.bars_file))
        if key not in seen:
            seen.add(key)
            unique.append(j)

    return unique


# ── Best-model selection ───────────────────────────────────────────────────────

def select_best(results: list[dict]) -> dict[str, dict]:
    """
    For each symbol, select the timeframe with the highest ZOmega.

    Returns:
        { symbol: best_result_dict }
    """
    best: dict[str, dict] = {}
    for r in results:
        if r.get("error"):
            continue
        sym = r["symbol"]
        score = r.get("z_omega", 0.0)
        if sym not in best or score > best[sym].get("z_omega", -1.0):
            best[sym] = r
    return best


def copy_best_weights(best: dict[str, dict], dest_dir: Path) -> None:
    """Copy winning weight files to dest_dir/{symbol}_{agent}.npz."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for sym, r in best.items():
        paths_str = r.get("weights_path", "")
        if not paths_str:
            continue
        for src_path in paths_str.split(";"):
            src = Path(src_path)
            if not src.exists():
                continue
            # Rename: XAUUSD_M5_trigger_offline.npz → best/XAUUSD_trigger.npz
            name = src.name
            # Strip the _M<N>_ part for the "best" canonical name
            canonical = re.sub(r"_M\d+_", "_", name)
            dst = dest_dir / canonical
            try:
                shutil.copy2(src, dst)
                LOG.info("[BEST] %s → %s  (ZOmega=%.4f)", src.name, dst.name, r["z_omega"])
            except Exception as exc:
                LOG.warning("[BEST] Could not copy %s: %s", src, exc)


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    header = f"{'Symbol':<12} {'TF':>5} {'Trades':>7} {'ValTrades':>9} {'Steps':>7} {'ZOmega':>9} {'Time':>8}  Status"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in sorted(results, key=lambda x: (x["symbol"], x["timeframe_minutes"])):
        label = f"M{r['timeframe_minutes']}"
        status = f"ERROR: {r['error'][:40]}" if r.get("error") else "OK"
        zo = r.get("z_omega", 0.0)
        zo_str = f"{zo:.4f}" if zo != float("inf") else "  +inf"
        print(
            f"{r['symbol']:<12} {label:>5} {r['train_trades']:>7} "
            f"{r['val_trades']:>9} {r['total_train_steps']:>7} "
            f"{zo_str:>9} {r['elapsed_s']:>7.1f}s  {status}"
        )
    print(sep)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parallelised offline DDQN training across multiple instruments / timeframes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Options")[0],
    )
    p.add_argument("inputs", nargs="+", metavar="FILE_OR_DIR",
                   help="CSV files, JSONL caches, or directories containing them")
    p.add_argument("--symbols",       nargs="+", default=None, metavar="SYM",
                   help="Filter to specific symbols (default: all)")
    p.add_argument("--timeframes",    nargs="+", default=None, metavar="TF",
                   help="Timeframe codes to train e.g. M5 H1 (default: all detected)")
    p.add_argument("--workers",       type=int, default=None,
                   help="Worker processes (default: CPU count)")
    p.add_argument("--checkpoint-dir",default="data/checkpoints", metavar="PATH")
    p.add_argument("--train-split",   type=float, default=0.80, metavar="0.8")
    p.add_argument("--train-every",   type=int, default=4, metavar="N")
    p.add_argument("--max-bars",      type=int, default=None, metavar="N")
    p.add_argument("--dry-run",       action="store_true",
                   help="Discover jobs and print plan without training")
    # Universe / paper-trading promotion
    p.add_argument(
        "--auto-promote", action="store_true", default=False,
        help=(
            "After training, promote any symbol whose best ZOmega meets "
            "--paper-threshold to PAPER stage in data/universe.json. "
            "Run `python3 run_universe.py` to launch the paper bots."
        ),
    )
    p.add_argument(
        "--paper-threshold", type=float, default=1.0, metavar="ZO",
        help="Minimum ZOmega score required for --auto-promote (default: 1.0)",
    )
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable DEBUG logging")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Ensure log dir exists before setting up file handler
    Path("log").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("log/train_offline.log", mode="a"),
        ],
    )

    # Discover jobs
    jobs = discover_jobs(args.inputs, args.symbols, args.timeframes)
    if not jobs:
        LOG.error("No jobs found. Check file paths and --symbols/--timeframes filters.")
        return 1

    LOG.info("Discovered %d job(s):", len(jobs))
    for j in jobs:
        LOG.info("  %s M%d  ← %s", j.symbol, j.timeframe_minutes, j.bars_file.name)

    if args.dry_run:
        print("\n[DRY RUN] — no training executed.")
        return 0

    n_workers = args.workers or min(len(jobs), multiprocessing.cpu_count())
    LOG.info("Launching %d worker(s) for %d job(s)", n_workers, len(jobs))

    results: list[dict] = []
    t_start = time.perf_counter()

    # ── Write initial status for HUD ─────────────────────────────────────────
    _ot_status: dict = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "total_jobs": len(jobs),
        "elapsed_s": 0.0,
        "results": [
            {
                "symbol": j.symbol,
                "timeframe_minutes": j.timeframe_minutes,
                "label": f"M{j.timeframe_minutes}",
                "status": "queued",
            }
            for j in jobs
        ],
    }
    _write_status(_ot_status)

    # Use spawn context to avoid CUDA/fork issues
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(
                _run_job,
                j.symbol,
                j.timeframe_minutes,
                str(j.bars_file),
                j.file_format,
                args.checkpoint_dir,
                args.train_split,
                args.train_every,
                args.max_bars,
            ): j
            for j in jobs
        }

        # Mark all submitted jobs as running (they're queued in the pool)
        for entry in _ot_status["results"]:
            entry["status"] = "running"
        _write_status(_ot_status)

        for fut in as_completed(futures):
            job = futures[fut]
            label = f"{job.symbol}_M{job.timeframe_minutes}"
            try:
                res = fut.result()
                results.append(res)
                if res.get("error"):
                    LOG.error("[MAIN] %s failed: %s", label, res["error"])
                else:
                    LOG.info(
                        "[MAIN] %s done — ZOmega=%.4f  trades=%d",
                        label, res["z_omega"], res["val_trades"],
                    )
            except Exception as exc:
                LOG.error("[MAIN] %s raised: %s", label, exc, exc_info=True)
                res = {
                    "symbol": job.symbol, "timeframe_minutes": job.timeframe_minutes,
                    "z_omega": 0.0, "train_trades": 0, "val_trades": 0,
                    "total_train_steps": 0, "elapsed_s": 0.0,
                    "weights_path": "", "error": str(exc),
                }
                results.append(res)

            # ── Update HUD status after each job completes ────────────────────
            _ot_status["elapsed_s"] = time.perf_counter() - t_start
            for entry in _ot_status["results"]:
                if entry["symbol"] == res["symbol"] and entry["timeframe_minutes"] == res["timeframe_minutes"]:
                    entry["status"] = "error" if res.get("error") else "done"
                    entry["z_omega"] = res.get("z_omega", 0.0)
                    entry["train_trades"] = res.get("train_trades", 0)
                    entry["val_trades"] = res.get("val_trades", 0)
                    entry["total_train_steps"] = res.get("total_train_steps", 0)
                    entry["elapsed_s"] = res.get("elapsed_s", 0.0)
                    entry["error"] = res.get("error")
                    break
            _write_status(_ot_status)

    total_s = time.perf_counter() - t_start

    # ── Write final status ────────────────────────────────────────────────────
    _ot_status["status"] = "complete"
    _ot_status["completed_at"] = datetime.now(timezone.utc).isoformat()
    _ot_status["elapsed_s"] = total_s
    _write_status(_ot_status)

    # Print summary table
    print_summary(results)
    LOG.info("All jobs completed in %.1f s", total_s)

    # Copy best-by-ZOmega weights per symbol
    best = select_best(results)
    if best:
        best_dir = Path(args.checkpoint_dir) / "best"
        copy_best_weights(best, best_dir)
        LOG.info("Best weights written to %s/", best_dir)
        print(f"\nBest weights by ZOmega:")
        promoted: list[str] = []
        for sym, r in sorted(best.items()):
            zo = r.get("z_omega", 0.0)
            zo_str = f"{zo:.4f}" if zo != float("inf") else "+inf"
            label = f"M{r['timeframe_minutes']}"
            print(f"  {sym:<12} {label:>5}  ZOmega={zo_str}")

            # Promote to PAPER stage if enabled and threshold met
            if args.auto_promote and zo >= args.paper_threshold:
                _register_universe(
                    symbol=sym,
                    timeframe_minutes=r["timeframe_minutes"],
                    z_omega=zo,
                    weights_path=r.get("weights_path", ""),
                )
                promoted.append(sym)

        if promoted:
            print(
                f"\n[UNIVERSE] {len(promoted)} instrument(s) promoted to PAPER stage: "
                + ", ".join(promoted)
            )
            print("  Launch paper bots with:  python3 run_universe.py --watch")

    n_ok  = sum(1 for r in results if not r.get("error"))
    n_err = len(results) - n_ok
    if n_err:
        LOG.warning("%d job(s) failed.", n_err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
