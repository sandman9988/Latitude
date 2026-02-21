#!/usr/bin/env python3
"""
run_universe.py – Universe Orchestrator
=========================================
Reads ``data/universe.json`` and manages paper-trading bot processes for
instruments that have graduated from offline training.

Each instrument follows this pipeline:

    UNTRAINED → OFFLINE_TRAINING → PAPER → MICRO → LIVE

``train_offline.py --auto-promote`` writes instruments to ``universe.json``
when their ZOmega clears the threshold.  This script then launches an
isolated paper bot for every PAPER-stage entry and keeps it alive.

Usage
-----
    # List current universe
    python3 run_universe.py --list

    # Launch paper bots for all PAPER-stage instruments (one-shot)
    python3 run_universe.py

    # Supervisor: keep all PAPER bots alive; pick up new promotions automatically
    python3 run_universe.py --watch

    # Manually promote an already-trained instrument to PAPER
    python3 run_universe.py --promote EURUSD --timeframe 60

    # Demote an instrument back to UNTRAINED (stops its paper bot)
    python3 run_universe.py --demote EURUSD

    # SIGTERM all tracked paper bots
    python3 run_universe.py --stop-all

Bot processes
-------------
Each paper bot is launched as:

    python3 -m src.core.ctrader_ddqn_paper

in its own session (``start_new_session=True``), immune to terminal signals.
Credentials are inherited from the environment; ``.env`` is sourced if
the caller has not already exported them.  Per-instrument overrides
(SYMBOL, SYMBOL_ID, TIMEFRAME_MINUTES, QTY, PAPER_MODE=1) are injected at
launch time so the same credentials file supports the full instrument
universe without modification.

Logs: ``logs/paper_{SYMBOL}_M{TF}.log``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("run_universe")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UNIVERSE_PATH  = Path("data/universe.json")
_SYMBOL_SPECS   = Path("config/symbol_specs.json")
_ENV_PATH       = Path(".env")
_BOT_MODULE     = "src.core.ctrader_ddqn_paper"
_STAGE_ORDER    = ["UNTRAINED", "OFFLINE_TRAINING", "PAPER", "MICRO", "LIVE"]
_PAPER_STAGE    = "PAPER"
_WATCH_INTERVAL = 30   # seconds between supervisor polls

# Paper-mode env defaults (mirror .env.example PAPER_MODE block)
_PAPER_ENV_DEFAULTS: dict[str, str] = {
    "PAPER_MODE":          "1",
    "DISABLE_GATES":       "1",
    "FEAS_THRESHOLD":      "0.0",
    "EPSILON_START":       "0.30",
    "EPSILON_END":         "0.05",
    "EPSILON_DECAY":       "0.9995",
    "EXPLORATION_BOOST":   "0.0",
    "MAX_BARS_INACTIVE":   "1000",
    "FORCE_EXPLORATION":   "0",
    "DDQN_ONLINE_LEARNING": "1",
}

# ---------------------------------------------------------------------------
# Universe I/O
# ---------------------------------------------------------------------------

def _load_universe() -> dict:
    if _UNIVERSE_PATH.exists():
        try:
            with open(_UNIVERSE_PATH) as f:
                return json.load(f)
        except Exception as exc:
            LOG.warning("Could not load %s: %s — starting empty", _UNIVERSE_PATH, exc)
    return {"version": 1, "instruments": {}}


def _save_universe(registry: dict) -> None:
    _UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _UNIVERSE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(registry, f, indent=2)
    tmp.replace(_UNIVERSE_PATH)


# ---------------------------------------------------------------------------
# Symbol spec lookup
# ---------------------------------------------------------------------------

def _load_symbol_specs() -> dict[str, dict]:
    try:
        with open(_SYMBOL_SPECS) as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception as exc:
        LOG.warning("Could not load %s: %s", _SYMBOL_SPECS, exc)
        return {}


# ---------------------------------------------------------------------------
# .env loader (credentials inheritance)
# ---------------------------------------------------------------------------

def _load_dotenv() -> dict[str, str]:
    """
    Parse .env file into a dict without modifying os.environ.
    Caller merges with os.environ so already-exported vars win.
    """
    result: dict[str, str] = {}
    try:
        for line in _ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'\"")
            if key:
                result[key] = val
    except OSError:
        pass
    return result


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)   # signal 0 = existence check only
        return True
    except OSError:
        return False


def _launch_paper_bot(
    symbol: str,
    timeframe_minutes: int,
    symbol_id: int,
    qty: float,
    base_env: dict[str, str],
) -> int:
    """
    Start an isolated paper-bot subprocess.

    The process gets its own session (``start_new_session=True``) so it
    survives terminal close and is immune to SIGINT propagation.

    Returns the child PID.
    """
    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs") / f"paper_{symbol}_M{timeframe_minutes}.log"

    # Build env: dotenv+os.environ base  →  paper defaults  →  per-instrument
    env = {
        **base_env,
        **_PAPER_ENV_DEFAULTS,
        # Per-instrument overrides (highest priority)
        "SYMBOL":            symbol,
        "SYMBOL_ID":         str(symbol_id),
        "TIMEFRAME_MINUTES": str(timeframe_minutes),
        "QTY":               str(qty),
    }

    cmd = [sys.executable, "-m", _BOT_MODULE]
    LOG.info(
        "Launching paper bot  %s M%d  →  %s  (symbol_id=%d, qty=%s)",
        symbol, timeframe_minutes, log_path, symbol_id, qty,
    )

    with open(log_path, "a") as log_fh:
        log_fh.write(
            f"\n{'='*60}\n"
            f"Paper bot started by run_universe  "
            f"{datetime.now(timezone.utc).isoformat()}\n"
            f"Symbol={symbol}  TF=M{timeframe_minutes}  "
            f"SymbolID={symbol_id}  QTY={qty}\n"
            f"{'='*60}\n"
        )
        log_fh.flush()

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,   # own process group/session
        )

    LOG.info("  PID %d  log: %s", proc.pid, log_path)
    return proc.pid


def _stop_pid(pid: int, label: str = "bot") -> None:
    """SIGTERM → wait 10 s → SIGKILL if still alive."""
    if not _pid_alive(pid):
        return
    LOG.info("Stopping %s (PID %d)…", label, pid)
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.5)
            if not _pid_alive(pid):
                LOG.info("  PID %d stopped", pid)
                return
        LOG.warning("  PID %d did not stop after 10 s — sending SIGKILL", pid)
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass   # already gone


# ---------------------------------------------------------------------------
# Core launch loop
# ---------------------------------------------------------------------------

def launch_paper_bots(
    registry: dict,
    specs: dict[str, dict],
    base_env: dict[str, str],
) -> dict:
    """
    Walk registry; for every PAPER-stage instrument that has no live bot,
    launch one and write the PID back.  Returns the (possibly mutated) registry.
    """
    instruments = registry.get("instruments", {})
    changed = False

    for symbol, entry in instruments.items():
        if entry.get("stage") != _PAPER_STAGE:
            continue

        pid = entry.get("paper_pid")
        if _pid_alive(pid):
            LOG.debug(
                "%s M%d — paper bot already running (PID %d)",
                symbol, entry.get("timeframe_minutes", 0), pid,
            )
            continue

        tf = entry.get("timeframe_minutes")
        if not tf:
            LOG.warning("%s — missing timeframe_minutes in universe.json; skipping", symbol)
            continue

        # Resolve symbol_id: universe.json > symbol_specs.json
        spec = specs.get(symbol, {})
        symbol_id = entry.get("symbol_id") or spec.get("symbol_id")
        if not symbol_id:
            LOG.warning(
                "%s — symbol_id not found in symbol_specs.json or universe.json; "
                "add it manually or update config/symbol_specs.json",
                symbol,
            )
            continue

        qty = float(spec.get("min_volume", 0.01))

        try:
            new_pid = _launch_paper_bot(symbol, tf, int(symbol_id), qty, base_env)
            entry["paper_pid"]        = new_pid
            entry["paper_started_at"] = datetime.now(timezone.utc).isoformat()
            entry["paper_log"]        = f"logs/paper_{symbol}_M{tf}.log"
            changed = True
        except Exception as exc:
            LOG.error("Failed to launch paper bot for %s: %s", symbol, exc)

    if changed:
        _save_universe(registry)

    return registry


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_list(registry: dict) -> None:
    instruments = registry.get("instruments", {})
    if not instruments:
        print(
            "Universe is empty.\n"
            "  Populate it with:  python3 train_offline.py <data> --auto-promote\n"
            "  Or manually:       python3 run_universe.py --promote XAUUSD --timeframe 240"
        )
        return

    header = (
        f"{'Symbol':<12} {'Stage':<20} {'TF':>6} {'ZOmega':>9} "
        f"{'PID':>8}  {'Promoted':<22}  Running?"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for sym, entry in sorted(instruments.items()):
        stage = entry.get("stage", "UNTRAINED")
        tf    = entry.get("timeframe_minutes", "?")
        zo    = entry.get("z_omega", 0.0)
        pid   = entry.get("paper_pid")
        prom  = (entry.get("promoted_at") or "")[:19].replace("T", " ")
        alive = "✓ running" if _pid_alive(pid) else ("✗ stopped" if pid else "—")
        zo_str = f"{zo:.4f}" if isinstance(zo, float) else str(zo)
        print(
            f"{sym:<12} {stage:<20} {str(tf):>6} {zo_str:>9} "
            f"{str(pid or '—'):>8}  {prom:<22}  {alive}"
        )
    print(sep)
    print()


def cmd_promote(
    registry: dict,
    symbol: str,
    timeframe_minutes: int,
    z_omega: float = 0.0,
    symbol_id: int | None = None,
) -> dict:
    instruments = registry.setdefault("instruments", {})
    existing = instruments.get(symbol, {})
    instruments[symbol] = {
        **existing,
        "stage":             _PAPER_STAGE,
        "timeframe_minutes": timeframe_minutes,
        "z_omega":           existing.get("z_omega", z_omega),
        "promoted_at":       datetime.now(timezone.utc).isoformat(),
        "paper_pid":         None,
        "paper_started_at":  None,
        **({"symbol_id": symbol_id} if symbol_id else {}),
    }
    _save_universe(registry)
    LOG.info("Promoted %s M%d → PAPER", symbol, timeframe_minutes)
    return registry


def cmd_demote(registry: dict, symbol: str) -> dict:
    instruments = registry.get("instruments", {})
    if symbol not in instruments:
        LOG.warning("%s not found in universe.json", symbol)
        return registry
    entry = instruments[symbol]
    pid = entry.get("paper_pid")
    if _pid_alive(pid):
        _stop_pid(pid, f"{symbol} paper bot")
    entry["stage"]     = "UNTRAINED"
    entry["paper_pid"] = None
    _save_universe(registry)
    LOG.info("Demoted %s → UNTRAINED", symbol)
    return registry


def cmd_stop_all(registry: dict) -> dict:
    for sym, entry in registry.get("instruments", {}).items():
        pid = entry.get("paper_pid")
        if _pid_alive(pid):
            _stop_pid(pid, f"{sym} paper bot")
            entry["paper_pid"] = None
    _save_universe(registry)
    return registry


# ---------------------------------------------------------------------------
# Arg parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Universe Orchestrator — launches and supervises paper-trading bots "
            "for offline-trained instruments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--watch", action="store_true",
        help=(
            f"Supervisor mode: poll every {_WATCH_INTERVAL}s, "
            "restart crashed bots, pick up new promotions from universe.json"
        ),
    )
    p.add_argument(
        "--list", "-l", action="store_true",
        help="Print instrument table and exit",
    )
    p.add_argument(
        "--promote", metavar="SYM",
        help="Manually set SYMBOL to PAPER stage in universe.json",
    )
    p.add_argument(
        "--timeframe", type=int, metavar="MIN",
        help="Timeframe in minutes (required with --promote)",
    )
    p.add_argument(
        "--symbol-id", type=int, metavar="ID",
        help="cTrader symbol ID (optional with --promote; falls back to symbol_specs.json)",
    )
    p.add_argument(
        "--demote", metavar="SYM",
        help="Reset SYMBOL to UNTRAINED stage (stops its paper bot if running)",
    )
    p.add_argument(
        "--stop-all", action="store_true",
        help="SIGTERM all tracked paper bots and exit",
    )
    p.add_argument(
        "--universe", type=Path, default=_UNIVERSE_PATH, metavar="PATH",
        help=f"Universe registry file (default: {_UNIVERSE_PATH})",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    global _UNIVERSE_PATH
    _UNIVERSE_PATH = args.universe

    # Build base environment: .env values first, os.environ overrides on top
    # (already-exported vars always win; per-instrument overrides added at launch)
    dotenv   = _load_dotenv()
    base_env = {**dotenv, **os.environ}

    specs    = _load_symbol_specs()
    registry = _load_universe()

    # ── One-shot commands ──────────────────────────────────────────────────

    if args.list:
        cmd_list(registry)
        return 0

    if args.stop_all:
        cmd_stop_all(registry)
        LOG.info("All tracked paper bots stopped.")
        return 0

    if args.demote:
        registry = cmd_demote(registry, args.demote.upper())
        return 0

    if args.promote:
        if not args.timeframe:
            parser.error("--promote requires --timeframe (in minutes, e.g. --timeframe 240)")
        registry = cmd_promote(
            registry,
            args.promote.upper(),
            args.timeframe,
            symbol_id=args.symbol_id,
        )

    # ── Launch pass ────────────────────────────────────────────────────────

    Path("logs").mkdir(exist_ok=True)

    LOG.info("Universe: %s", _UNIVERSE_PATH)
    registry = launch_paper_bots(registry, specs, base_env)
    cmd_list(registry)

    if not args.watch:
        return 0

    # ── Supervisor loop ────────────────────────────────────────────────────

    LOG.info(
        "Supervisor mode active — polling every %ds.  "
        "Paper bots run in background; Ctrl+C exits supervisor only.",
        _WATCH_INTERVAL,
    )
    try:
        while True:
            time.sleep(_WATCH_INTERVAL)
            # Re-read file so new train_offline.py promotions are picked up
            registry = _load_universe()
            registry = launch_paper_bots(registry, specs, base_env)
    except KeyboardInterrupt:
        LOG.info(
            "Supervisor stopped.  Paper bots continue running in background.\n"
            "  Check status:  python3 run_universe.py --list\n"
            "  Stop all bots: python3 run_universe.py --stop-all"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
