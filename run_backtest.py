#!/usr/bin/env python3
"""
Latitude — end-to-end backtest runner.

Downloads or loads bar data, resamples to MTF, runs ML-aware walk-forward
validation, and optionally runs Optuna hyperparameter tuning.

Usage examples:
    # Download from cTrader and run walk-forward
    python run_backtest.py --symbol XAUUSD --days 360

    # Load from CSV (skip download)
    python run_backtest.py --symbol XAUUSD --csv data/XAUUSD/M30/XAUUSD_M30_*.csv

    # Full tuning run
    python run_backtest.py --symbol DE40 --days 360 --tune --n-trials 100

    # Quick single-pass (no walk-forward, no ML)
    python run_backtest.py --symbol XAUUSD --days 90 --single-pass

Environment variables (for cTrader download):
    Set up .env.openapi with CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID
    or run: python -m ctrader.auth
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from core.logger import get_logger
from core.validator import BrokerSpec
from pipeline.cleaner import Bar, clean_bars
from pipeline.resampler import build_mtf
from backtesting.pipeline import BacktestPipeline
from backtesting.metrics import compute_metrics

logger = get_logger("run_backtest")


# ---------------------------------------------------------------------------
# Minimal BrokerSpec for offline use (no cTrader connection)
# Overridden by fetched spec when --download is used.
# ---------------------------------------------------------------------------

_DEFAULT_SPECS: dict[str, dict] = {
    "XAUUSD": dict(
        symbol="XAUUSD", digits=2, pip_size=0.01, tick_size=0.01,
        tick_value=1.0, lot_size=100.0, lot_step=0.01,
        min_volume=0.01, max_volume=50.0, margin_rate=0.005,
        spread_pips=2.0, short_selling_enabled=True,
    ),
    "DE40": dict(
        symbol="DE40", digits=1, pip_size=1.0, tick_size=0.5,
        tick_value=25.0, lot_size=1.0, lot_step=0.1,
        min_volume=0.1, max_volume=20.0, margin_rate=0.005,
        spread_pips=1.5, short_selling_enabled=True,
    ),
}


def _make_default_spec(symbol: str) -> BrokerSpec:
    kwargs = _DEFAULT_SPECS.get(symbol.upper())
    if kwargs is None:
        kwargs = dict(
            symbol=symbol.upper(), digits=5, pip_size=0.0001, tick_size=0.00001,
            tick_value=1.0, lot_size=100000.0, lot_step=0.01,
            min_volume=0.01, max_volume=100.0, margin_rate=0.01,
            spread_pips=1.5, short_selling_enabled=True,
        )
    return BrokerSpec(**kwargs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_csv(csv_paths: List[Path], symbol: str) -> List[Bar]:
    from pipeline.downloader import load_bars_from_csv
    bars: List[Bar] = []
    for path in csv_paths:
        bars.extend(load_bars_from_csv(str(path), symbol=symbol, timeframe="M30"))
    bars = clean_bars(bars)
    logger.info(f"Loaded {len(bars)} bars from CSV", symbol=symbol)
    return bars


def download_from_ctrader(symbol: str, days: int, output_dir: Path) -> List[Bar]:
    try:
        from ctrader.connector import CTraderConnector, CTraderCredentials
        from ctrader.spec_fetcher import fetch_spec
        from pipeline.downloader import download_bars
    except ImportError as e:
        logger.error(f"cTrader import failed: {e}")
        sys.exit(1)

    creds = CTraderCredentials.from_env()
    conn = CTraderConnector(creds)

    logger.info("Connecting to cTrader...", symbol=symbol)
    if not conn.start(timeout_s=30.0):
        logger.error("Failed to connect to cTrader")
        sys.exit(1)

    try:
        spec = fetch_spec(conn, symbol)
        logger.info(
            f"Spec: lot={spec.lot_size} pip={spec.pip_size} "
            f"commission={spec.commission_rate} margin={spec.margin_rate:.1%}",
            symbol=symbol,
        )

        end_dt = datetime.now(tz=timezone.utc)
        start_dt = end_dt - timedelta(days=days)
        bars = download_bars(conn, symbol, "M30", days=days)
        bars = clean_bars(bars)
        logger.info(f"Downloaded {len(bars)} M30 bars", symbol=symbol)

        bars_dir = output_dir / symbol / "M30"
        bars_dir.mkdir(parents=True, exist_ok=True)
        return bars, spec
    finally:
        conn.stop()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    if not metrics:
        print("  (no trades)")
        return
    rows = [
        ("Trades",          f"{int(metrics.get('total_trades', 0))}"),
        ("Win Rate",        f"{metrics.get('win_rate', 0):.1%}"),
        ("Profit Factor",   f"{metrics.get('profit_factor', 0):.2f}"),
        ("Expectancy",      f"${metrics.get('expectancy', 0):.2f}"),
        ("Net P&L",         f"${metrics.get('net_pnl', 0):.2f}"),
        ("Return",          f"{metrics.get('return_pct', 0):.1%}"),
        ("Sharpe",          f"{metrics.get('sharpe', 0):.2f}"),
        ("Sortino",         f"{metrics.get('sortino', 0):.2f}"),
        ("Max DD",          f"{metrics.get('max_drawdown_pct', 0):.1%}"),
        ("Avg MFE",         f"{metrics.get('avg_mfe', 0):.4f}"),
        ("MFE/MAE Ratio",   f"{metrics.get('mfe_mae_ratio', 0):.2f}"),
        ("Avg Bars Held",   f"{metrics.get('avg_bars_held', 0):.1f}"),
    ]
    for name, val in rows:
        print(f"  {name:<18} {val}")
    print()


def print_wf_summary(result) -> None:
    print(f"\n{'=' * 60}")
    print("  Walk-Forward Summary")
    print(f"{'=' * 60}")
    print(f"  {'Fold':<6} {'IS Trades':>10} {'IS WR':>8} {'OOS Trades':>11} {'OOS WR':>8} {'ML':>5}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*11} {'-'*8} {'-'*5}")
    for f in result.folds:
        ml_flag = "yes" if f.ml_trained else f"n={f.ml_train_samples}"
        print(
            f"  {f.fold:<6} {f.is_trades:>10} "
            f"{f.is_metrics.get('win_rate', 0):>8.1%} "
            f"{f.oos_trades:>11} "
            f"{f.oos_metrics.get('win_rate', 0):>8.1%} "
            f"{ml_flag:>5}"
        )
    print(f"\n  Efficiency (OOS/IS):  {result.efficiency:.2f}  (target 0.65-0.90)")
    print(f"  Robust:               {'YES' if result.is_robust else 'NO'}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Latitude backtest runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",      default="XAUUSD",  help="Trading symbol")
    p.add_argument("--days",        type=int, default=360, help="Days of history to download")
    p.add_argument("--balance",     type=float, default=10_000.0, help="Starting balance")
    p.add_argument("--output",      default="./data",  help="Output directory for results/CSVs")
    p.add_argument("--folds",       type=int, default=5, help="Walk-forward folds")
    p.add_argument("--anchored",    action="store_true", help="Use anchored walk-forward")
    p.add_argument("--single-pass", action="store_true", help="Single backtest, no WF")
    p.add_argument("--no-ml",       action="store_true", help="Disable ML entry filter")
    p.add_argument("--tune",        action="store_true", help="Run Optuna hyperparameter search")
    p.add_argument("--n-trials",    type=int, default=50, help="Optuna trial count")
    p.add_argument("--timeout",     type=int, default=3600, help="Optuna timeout in seconds")
    p.add_argument("--metric",      default="win_rate",
                   choices=["win_rate", "sharpe", "profit_factor", "calmar"],
                   help="Optuna optimisation target")
    p.add_argument("--csv",         nargs="*", metavar="FILE",
                   help="Load bars from CSV files instead of downloading")
    p.add_argument("--best-params", metavar="FILE",
                   help="JSON file with best params from a previous tune run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    output_dir = Path(args.output)

    # --- Load or download bars ---
    spec: Optional[BrokerSpec] = None

    if args.csv:
        csv_paths = [Path(p) for p in args.csv]
        bars = load_from_csv(csv_paths, symbol)
        spec = _make_default_spec(symbol)
    else:
        result = download_from_ctrader(symbol, args.days, output_dir)
        if isinstance(result, tuple):
            bars, spec = result
        else:
            bars = result
            spec = _make_default_spec(symbol)

    if not bars:
        logger.error("No bars loaded — aborting")
        sys.exit(1)

    print(f"\nLoaded {len(bars)} M30 bars for {symbol}")
    print(f"Range: {_ts_str(bars[0].timestamp)} → {_ts_str(bars[-1].timestamp)}")

    # Resample for MTF context
    mtf = build_mtf(bars)
    for tf, tf_bars in mtf.items():
        print(f"  {tf}: {len(tf_bars)} bars")

    # --- Load best params if provided ---
    from strategy.trend_strategy import StrategyConfig
    config = StrategyConfig()
    if args.best_params:
        import json
        with open(args.best_params) as f:
            saved = json.load(f)
        from backtesting.pipeline import _params_to_config
        config = _params_to_config(saved.get("params", {}))
        print(f"\nLoaded best params from {args.best_params}")

    # --- Build pipeline ---
    pipeline = BacktestPipeline(spec, config, initial_balance=args.balance)

    # --- Single pass ---
    if args.single_pass:
        print("\nRunning single-pass backtest (heuristic only)...")
        t0 = time.time()
        result = pipeline.run_single(bars)
        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.1f}s")
        print_metrics(f"Single-Pass — {symbol}", result.metrics)
        return

    # --- Walk-forward ---
    if not args.tune:
        print(f"\nRunning {args.folds}-fold walk-forward validation...")
        t0 = time.time()
        wf_result = pipeline.run_walk_forward(
            bars,
            n_folds=args.folds,
            anchored=args.anchored,
        )
        elapsed = time.time() - t0
        print(f"Completed in {elapsed:.1f}s")
        print_wf_summary(wf_result)
        print_metrics(f"OOS Aggregate — {symbol}", wf_result.oos_metrics)
        pipeline.save_result(wf_result, output_dir / symbol)
        return

    # --- Optuna tuning ---
    print(f"\nRunning Optuna tuning: {args.n_trials} trials, metric={args.metric}...")
    t0 = time.time()
    tune_result = pipeline.tune(
        bars,
        n_trials=args.n_trials,
        n_folds=min(args.folds, 4),   # fewer folds during tuning for speed
        timeout_s=args.timeout,
        metric=args.metric,
        output_dir=output_dir / symbol,
    )
    elapsed = time.time() - t0
    print(f"\nTuning completed in {elapsed:.1f}s")
    print(f"Best {args.metric}: {tune_result.best_value:.4f}")
    print(f"Trials run:         {tune_result.n_trials}")
    print("\nBest parameters:")
    for k, v in sorted(tune_result.best_params.items()):
        print(f"  {k:<30} {v}")

    # Run final walk-forward with best params
    print(f"\nRunning final walk-forward with best params...")
    from backtesting.pipeline import _params_to_config
    best_config = _params_to_config(tune_result.best_params)
    final_pipeline = BacktestPipeline(spec, best_config, initial_balance=args.balance)
    wf_result = final_pipeline.run_walk_forward(bars, n_folds=args.folds)
    print_wf_summary(wf_result)
    print_metrics(f"OOS Aggregate (Best Params) — {symbol}", wf_result.oos_metrics)
    final_pipeline.save_result(wf_result, output_dir / symbol)


def _ts_str(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
