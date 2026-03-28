#!/usr/bin/env python3
"""
Latitude — live trading session runner.

Downloads warm-up bars, loads (or tunes) a strategy config, and starts
the live session loop connected to cTrader.

Usage examples:
    # Paper trade (log signals, no real orders):
    python run_live.py --symbol XAUUSD --paper

    # Live with best params from a previous tune run:
    python run_live.py --symbol XAUUSD --best-params data/XAUUSD/XAUUSD_best_params.json

    # Live with default config:
    python run_live.py --symbol XAUUSD --warmup-days 90

Environment:
    Credentials read from .env.openapi (see: python -m ctrader.auth)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.logger import get_logger
from core.validator import BrokerSpec
from pipeline.cleaner import clean_bars

logger = get_logger("run_live")


# ---------------------------------------------------------------------------
# Default specs (same as run_backtest.py)
# ---------------------------------------------------------------------------

_DEFAULT_SPECS = {
    "XAUUSD": dict(
        symbol="XAUUSD", digits=2, pip_size=0.01, tick_size=0.01,
        tick_value=1.0, lot_size=100.0, lot_step=0.01,
        min_volume=0.01, max_volume=50.0, margin_rate=0.01,
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
            tick_value=1.0, lot_size=100_000.0, lot_step=0.01,
            min_volume=0.01, max_volume=100.0, margin_rate=0.01,
            spread_pips=1.5, short_selling_enabled=True,
        )
    return BrokerSpec(**kwargs)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latitude live trading session")
    p.add_argument("--symbol",       required=True, help="Symbol to trade (e.g. XAUUSD)")
    p.add_argument("--warmup-days",  type=int, default=90,
                   help="Days of M30 history to download for indicator warm-up (default 90)")
    p.add_argument("--best-params",  metavar="FILE",
                   help="JSON file with best Optuna params (from run_backtest.py --tune)")
    p.add_argument("--bar-source",   choices=["trendbar", "tick"], default="trendbar",
                   help="Live bar source: trendbar (cTrader native) or tick (aggregated)")
    p.add_argument("--lots-per-1000", type=float, default=0.01,
                   help="Lots per $1000 equity for position sizing (default 0.01)")
    p.add_argument("--max-trades",   type=int, default=3,
                   help="Maximum concurrent open positions (default 3)")
    p.add_argument("--max-daily-loss", type=float, default=0.05,
                   help="Stop trading if daily drawdown exceeds this fraction (default 0.05)")
    p.add_argument("--paper",        action="store_true",
                   help="Paper trade: log signals but do not send real orders")
    p.add_argument("--output",       default="data", help="Directory for logs/data (default data)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    symbol = Path(args.symbol.strip().upper()).name or args.symbol.strip().upper()
    output_dir = Path(args.output)

    # --- Connect to cTrader ---
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
        # --- Fetch live spec ---
        try:
            spec = fetch_spec(conn, symbol)
            logger.info(
                f"Live spec: lot={spec.lot_size} pip={spec.pip_size} "
                f"tick={spec.tick_size} tv={spec.tick_value}",
                symbol=symbol,
            )
        except Exception as e:
            logger.warning(f"Could not fetch spec ({e}), using defaults")
            spec = _make_default_spec(symbol)

        # --- Download warm-up bars ---
        warmup_bars = []
        if args.warmup_days > 0:
            logger.info(f"Downloading {args.warmup_days}d M30 warm-up bars...", symbol=symbol)
            try:
                warmup_bars = download_bars(conn, symbol, "M30", days=args.warmup_days)
                warmup_bars = clean_bars(warmup_bars)
                logger.info(f"Warm-up: {len(warmup_bars)} M30 bars", symbol=symbol)
            except Exception as e:
                logger.warning(f"Warm-up download failed: {e} — starting cold")

        # --- Load strategy config ---
        from strategy.trend_strategy import TrendStrategy, StrategyConfig

        config = StrategyConfig()
        if args.best_params:
            params_path = Path(args.best_params)
            if params_path.exists():
                try:
                    with open(params_path, encoding="utf-8") as f:
                        data = json.load(f)
                    params = data.get("best_params", data)
                    # Apply only fields that exist on StrategyConfig
                    for k, v in params.items():
                        if hasattr(config, k):
                            setattr(config, k, v)
                    logger.info(f"Loaded strategy params from {params_path}", symbol=symbol)
                except Exception as e:
                    logger.warning(f"Could not load best-params: {e} — using defaults")

        strategy = TrendStrategy(config=config, spec=spec)

        # --- Build and start session ---
        from live.session import LiveSession, SessionConfig

        session_cfg = SessionConfig(
            symbol=symbol,
            timeframe="M30",
            bar_source=args.bar_source,
            warmup_bars=len(warmup_bars),
            lots_per_1000=args.lots_per_1000,
            max_open_trades=args.max_trades,
            max_daily_loss_pct=args.max_daily_loss,
            paper_mode=args.paper,
        )

        session = LiveSession(
            strategy=strategy,
            spec=spec,
            connector=conn,
            config=session_cfg,
            warmup_bars=warmup_bars or None,
        )

        mode = "PAPER" if args.paper else "LIVE"
        logger.info(f"[{mode}] Starting session — {symbol} M30 via {args.bar_source}")
        session.start()   # blocks until Ctrl-C or SIGTERM

    finally:
        conn.stop()
        logger.info("Session ended")


if __name__ == "__main__":
    main()
