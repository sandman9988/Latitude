#!/usr/bin/env python3
"""
run_universe.py – Universe Orchestrator
=========================================
Reads ``data/universe.json`` and manages paper-trading bot processes for
instruments that have graduated from offline training.

Each instrument follows this pipeline:

    UNTRAINED → OFFLINE_TRAINING → PAPER → MICRO → LIVE

``train_offline.py --auto-promote`` writes instruments to ``universe.json``
when their ZOmega clears the threshold.  By default this script launches an
isolated paper bot for every PAPER-stage entry and keeps it alive.  For
FIX-safe migration work, set ``UNIVERSE_BROKER_TOPOLOGY=shared-symbol`` or
``shared-account`` to allow only one direct-FIX owner while other entries wait
for the shared gateway path.

Usage
-----
    # List current universe
    python3 run_universe.py --list

    # Launch paper bots for all PAPER-stage instruments (one-shot)
    python3 run_universe.py

    # Supervisor: keep all PAPER bots alive; pick up new promotions automatically
    python3 run_universe.py --watch

    # Transitional guard: one direct-FIX owner for all entries on the broker account
    python3 run_universe.py --watch --broker-topology shared-account

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
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

LOG = logging.getLogger("run_universe")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
_UNIVERSE_PATH  = Path("data/universe.json")
_SYMBOL_SPECS   = Path("config/symbol_specs.json")
_ENV_PATH       = Path(".env")
_BOT_MODULE     = "src.core.ctrader_ddqn_paper"
_STAGE_ORDER    = ["UNTRAINED", "OFFLINE_TRAINING", "PAPER", "MICRO", "LIVE"]
_PAPER_STAGE    = "PAPER"
_WATCH_INTERVAL = 30   # seconds between supervisor polls
_CFG_QUOTE_TEMPLATE = Path("config/ctrader_quote.cfg")
_CFG_TRADE_TEMPLATE = Path("config/ctrader_trade.cfg")
_RUNTIME_ROOT = Path("data/paper_runtime")
_LAUNCH_STAGGER_ENV = "UNIVERSE_LAUNCH_STAGGER_SEC"
_BROKER_TOPOLOGY_ENV = "UNIVERSE_BROKER_TOPOLOGY"
_TOPOLOGY_ISOLATED = "isolated"
_TOPOLOGY_SHARED_SYMBOL = "shared-symbol"
_TOPOLOGY_SHARED_ACCOUNT = "shared-account"
_VALID_BROKER_TOPOLOGIES = {
    _TOPOLOGY_ISOLATED,
    _TOPOLOGY_SHARED_SYMBOL,
    _TOPOLOGY_SHARED_ACCOUNT,
}
_PROJECT_VENV_PYTHONS = (
    _PROJECT_ROOT / ".venv/bin/python",
    _PROJECT_ROOT / ".venv/bin/python3",
    _PROJECT_ROOT / "venv/bin/python",
    _PROJECT_ROOT / "venv/bin/python3",
)

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

def _normalize_instruments(raw_instruments: object) -> list[dict]:
    """Normalize legacy/new universe shapes into a list of entry dicts."""
    out: list[dict] = []
    if isinstance(raw_instruments, list):
        for item in raw_instruments:
            if isinstance(item, dict):
                _entry = dict(item)
                if _entry.get("symbol"):
                    _entry["symbol"] = str(_entry["symbol"]).upper()
                out.append(_entry)
        return out
    if isinstance(raw_instruments, dict):
        for sym, item in raw_instruments.items():
            if isinstance(item, list):
                for sub in item:
                    if not isinstance(sub, dict):
                        continue
                    _entry = dict(sub)
                    _entry.setdefault("symbol", str(sym).upper())
                    out.append(_entry)
            elif isinstance(item, dict):
                _entry = dict(item)
                _entry.setdefault("symbol", str(sym).upper())
                out.append(_entry)
    return out


def _load_universe() -> dict:
    if _UNIVERSE_PATH.exists():
        try:
            with open(_UNIVERSE_PATH) as f:
                raw = json.load(f)
            instruments = _normalize_instruments(raw.get("instruments", raw))
            return {"version": int(raw.get("version", 1)), "instruments": instruments}
        except Exception as exc:
            LOG.warning("Could not load %s: %s — starting empty", _UNIVERSE_PATH, exc)
    return {"version": 1, "instruments": []}


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
        for raw_line in _ENV_PATH.read_text().splitlines():
            line = raw_line.strip()
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
# FIX config/runtime isolation helpers
# ---------------------------------------------------------------------------

def _bot_slug(symbol: str, timeframe_minutes: int) -> str:
    _symbol = re.sub(r"[^A-Z0-9]+", "_", str(symbol or "").upper()).strip("_") or "UNKNOWN"
    return f"paper_{_symbol}_M{int(timeframe_minutes)}"


def _checkpoint_slug(symbol: str, timeframe_minutes: int) -> str:
    _symbol = re.sub(r"[^A-Z0-9]+", "_", str(symbol or "").upper()).strip("_") or "UNKNOWN"
    return f"{_symbol}_M{int(timeframe_minutes)}"


def _runtime_checkpoint_dir(symbol: str, timeframe_minutes: int) -> Path:
    return (
        Path("data")
        / _bot_slug(symbol, timeframe_minutes)
        / "checkpoints"
        / _checkpoint_slug(symbol, timeframe_minutes)
    )


def _weight_agent(path: Path) -> str | None:
    name = path.name.lower()
    if "trigger" in name:
        return "trigger"
    if "harvester" in name:
        return "harvester"
    return None


def _promoted_weight_sources(entry: dict) -> dict[str, Path]:
    raw = str(entry.get("weights_path") or "").strip()
    if not raw:
        return {}
    sources: dict[str, Path] = {}
    for item in raw.split(";"):
        text = item.strip()
        if not text:
            continue
        src = Path(text)
        agent = _weight_agent(src)
        if agent is None:
            LOG.warning("Ignoring unrecognised promoted weight path: %s", src)
            continue
        if not src.exists():
            LOG.warning("Promoted %s weight missing: %s", agent, src)
            continue
        sources[agent] = src
    return sources


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_weights_stale(entry: dict, symbol: str, timeframe_minutes: int) -> bool:
    sources = _promoted_weight_sources(entry)
    if not sources:
        return False
    checkpoint_dir = _runtime_checkpoint_dir(symbol, timeframe_minutes)
    for agent, src in sources.items():
        dst = checkpoint_dir / f"{agent}_ddqn_weights{src.suffix or '.pt'}"
        if not dst.exists():
            return True
        try:
            if _sha256(src) != _sha256(dst):
                return True
        except OSError:
            return True
    return False


def _sync_promoted_weights_to_runtime(entry: dict, symbol: str, timeframe_minutes: int) -> bool:
    sources = _promoted_weight_sources(entry)
    if not sources:
        return False
    checkpoint_dir = _runtime_checkpoint_dir(symbol, timeframe_minutes)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    copied = False
    for agent, src in sources.items():
        dst = checkpoint_dir / f"{agent}_ddqn_weights{src.suffix or '.pt'}"
        try:
            if dst.exists() and _sha256(src) == _sha256(dst):
                continue
            shutil.copy2(src, dst)
            copied = True
            LOG.info("[WEIGHTS] Synced promoted %s weights for %s M%d → %s", agent, symbol, timeframe_minutes, dst)
        except OSError as exc:
            LOG.warning(
                "[WEIGHTS] Could not sync promoted %s weights for %s M%d: %s",
                agent, symbol, timeframe_minutes, exc,
            )
    return copied


def _write_isolated_fix_cfg(template_path: Path, output_path: Path, store_dir: Path, log_dir: Path) -> None:
    text = template_path.read_text(encoding="utf-8")
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("FileStorePath="):
            lines.append(f"FileStorePath={store_dir.as_posix()}")
        elif line.startswith("FileLogPath="):
            lines.append(f"FileLogPath={log_dir.as_posix()}")
        else:
            lines.append(line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_bot_runtime(symbol: str, timeframe_minutes: int) -> dict[str, str]:
    slug = _bot_slug(symbol, timeframe_minutes)
    runtime_root = _RUNTIME_ROOT / slug
    cfg_dir = runtime_root / "config"
    quote_cfg = cfg_dir / "ctrader_quote.cfg"
    trade_cfg = cfg_dir / "ctrader_trade.cfg"
    quote_store = Path("store") / slug / "QUOTE"
    quote_log = Path("logs/fix") / slug / "QUOTE"
    trade_store = Path("store") / slug / "TRADE"
    trade_log = Path("logs/fix") / slug / "TRADE"

    quote_store.mkdir(parents=True, exist_ok=True)
    quote_log.mkdir(parents=True, exist_ok=True)
    trade_store.mkdir(parents=True, exist_ok=True)
    trade_log.mkdir(parents=True, exist_ok=True)

    _write_isolated_fix_cfg(
        _CFG_QUOTE_TEMPLATE,
        quote_cfg,
        store_dir=quote_store,
        log_dir=quote_log,
    )
    _write_isolated_fix_cfg(
        _CFG_TRADE_TEMPLATE,
        trade_cfg,
        store_dir=trade_store,
        log_dir=trade_log,
    )

    data_dir = Path("data") / slug
    data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "CTRADER_CFG_QUOTE": quote_cfg.as_posix(),
        "CTRADER_CFG_TRADE": trade_cfg.as_posix(),
        "CTRADER_DATA_DIR": data_dir.as_posix(),
    }


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)   # signal 0 = existence check only
    except OSError:
        return False
    # Reject zombie (defunct) processes — os.kill(pid, 0) succeeds for zombies
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat", "--no-headers"],
            capture_output=True, text=True,
            check=False,
        )
        return bool(result.stdout.strip()) and "Z" not in result.stdout
    except Exception:
        return True  # ps unavailable — assume alive


def _resolve_python_executable(base_env: dict[str, str]) -> str:
    for candidate in _PROJECT_VENV_PYTHONS:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.as_posix()

    env_python = str(base_env.get("VIRTUAL_ENV") or "").strip()
    if env_python:
        for suffix in ("bin/python", "bin/python3"):
            candidate = Path(env_python) / suffix
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate.as_posix()

    explicit_python = str(base_env.get("PYTHON") or "").strip()
    if explicit_python and Path(explicit_python).exists() and os.access(explicit_python, os.X_OK):
        return explicit_python

    return sys.executable


def _launch_stagger_seconds(base_env: dict[str, str]) -> float:
    raw = str(base_env.get(_LAUNCH_STAGGER_ENV) or os.environ.get(_LAUNCH_STAGGER_ENV) or "0").strip()
    try:
        delay = float(raw)
    except ValueError:
        LOG.warning("Ignoring invalid %s=%r", _LAUNCH_STAGGER_ENV, raw)
        return 0.0
    return max(0.0, min(delay, 120.0))


def _broker_topology(base_env: dict[str, str]) -> str:
    raw = str(base_env.get(_BROKER_TOPOLOGY_ENV) or os.environ.get(_BROKER_TOPOLOGY_ENV) or _TOPOLOGY_ISOLATED)
    topology = raw.strip().lower().replace("_", "-")
    aliases = {
        "account": _TOPOLOGY_SHARED_ACCOUNT,
        "gateway": _TOPOLOGY_SHARED_ACCOUNT,
        "shared": _TOPOLOGY_SHARED_ACCOUNT,
        "symbol": _TOPOLOGY_SHARED_SYMBOL,
    }
    topology = aliases.get(topology, topology)
    if topology not in _VALID_BROKER_TOPOLOGIES:
        LOG.warning("Ignoring invalid %s=%r; using %s", _BROKER_TOPOLOGY_ENV, raw, _TOPOLOGY_ISOLATED)
        return _TOPOLOGY_ISOLATED
    return topology


def _read_fix_cfg_value(path: Path, key: str) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    prefix = f"{key}="
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _cfg_path_from_env(base_env: dict[str, str], env_key: str, default_path: Path) -> Path:
    raw = str(base_env.get(env_key) or os.environ.get(env_key) or default_path).strip()
    return Path(raw)


def _broker_identity(base_env: dict[str, str]) -> str:
    quote_cfg = _cfg_path_from_env(base_env, "CTRADER_CFG_QUOTE", _CFG_QUOTE_TEMPLATE)
    trade_cfg = _cfg_path_from_env(base_env, "CTRADER_CFG_TRADE", _CFG_TRADE_TEMPLATE)
    sender = (
        _read_fix_cfg_value(quote_cfg, "SenderCompID")
        or _read_fix_cfg_value(trade_cfg, "SenderCompID")
        or str(base_env.get("CTRADER_USERNAME") or os.environ.get("CTRADER_USERNAME") or "unknown_sender")
    )
    target = (
        _read_fix_cfg_value(trade_cfg, "TargetCompID")
        or _read_fix_cfg_value(quote_cfg, "TargetCompID")
        or "unknown_target"
    )
    return f"{sender}->{target}"


def _entry_tf(entry: dict) -> int | None:
    try:
        value = int(entry.get("timeframe_minutes", 0) or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _entry_fix_key(entry: dict) -> tuple[str, int] | None:
    symbol = str(entry.get("symbol", "") or "").upper()
    tf = _entry_tf(entry)
    if not symbol or tf is None:
        return None
    return symbol, tf


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _fix_owner_rank(entry: dict) -> tuple[int, int, str]:
    key = _entry_fix_key(entry) or ("", 0)
    return (0 if _truthy(entry.get("fix_owner")) else 1, key[1], key[0])


def _set_entry_field(entry: dict, key: str, value: object) -> bool:
    if entry.get(key) == value:
        return False
    entry[key] = value
    return True


def _pop_entry_field(entry: dict, key: str) -> bool:
    if key not in entry:
        return False
    entry.pop(key, None)
    return True


def _fix_scope_key(entry: dict, topology: str, broker_identity: str) -> tuple[str, ...] | None:
    if topology == _TOPOLOGY_ISOLATED:
        return None
    key = _entry_fix_key(entry)
    if key is None:
        return None
    if topology == _TOPOLOGY_SHARED_SYMBOL:
        return (broker_identity, key[0])
    if topology == _TOPOLOGY_SHARED_ACCOUNT:
        return (broker_identity,)
    return None


def _plan_fix_session_owners(
    instruments: list,
    topology: str,
    base_env: dict[str, str],
) -> dict[tuple[str, int], tuple[str, int]]:
    """Map every PAPER entry key to the direct-FIX owner key for its topology scope."""
    if topology == _TOPOLOGY_ISOLATED:
        return {}

    broker_identity = _broker_identity(base_env)
    groups: dict[tuple[str, ...], list[dict]] = {}
    for entry in instruments:
        if not isinstance(entry, dict) or entry.get("stage") != _PAPER_STAGE:
            continue
        scope = _fix_scope_key(entry, topology, broker_identity)
        if scope is None:
            continue
        groups.setdefault(scope, []).append(entry)

    owner_for_entry: dict[tuple[str, int], tuple[str, int]] = {}
    for scope, entries in groups.items():
        owner = min(entries, key=_fix_owner_rank)
        owner_key = _entry_fix_key(owner)
        if owner_key is None:
            continue
        explicit_owners = [e for e in entries if _truthy(e.get("fix_owner"))]
        if len(explicit_owners) > 1:
            LOG.warning(
                "Multiple fix_owner entries in %s scope %s; using %s M%d",
                topology,
                scope,
                owner_key[0],
                owner_key[1],
            )
        for entry in entries:
            key = _entry_fix_key(entry)
            if key is not None:
                owner_for_entry[key] = owner_key

    return owner_for_entry


def _annotate_fix_session_entry(
    entry: dict,
    topology: str,
    owner_key: tuple[str, int] | None,
    entry_key: tuple[str, int],
) -> bool:
    changed = False
    changed |= _set_entry_field(entry, "broker_topology", topology)
    if topology == _TOPOLOGY_ISOLATED or owner_key is None:
        changed |= _set_entry_field(entry, "fix_session_owner", True)
        changed |= _pop_entry_field(entry, "fix_owner_symbol")
        changed |= _pop_entry_field(entry, "fix_owner_timeframe_minutes")
        changed |= _pop_entry_field(entry, "direct_fix_disabled_reason")
        return changed

    is_owner = owner_key == entry_key
    changed |= _set_entry_field(entry, "fix_session_owner", is_owner)
    changed |= _set_entry_field(entry, "fix_owner_symbol", owner_key[0])
    changed |= _set_entry_field(entry, "fix_owner_timeframe_minutes", owner_key[1])
    if is_owner:
        changed |= _pop_entry_field(entry, "direct_fix_disabled_reason")
    else:
        changed |= _set_entry_field(entry, "direct_fix_disabled_reason", "shared_fix_gateway_pending")
    return changed


def _launch_paper_bot(  # noqa: PLR0913
    symbol: str,
    timeframe_minutes: int,
    symbol_id: int,
    qty: float,
    base_env: dict[str, str],
    starting_equity: float | None = None,
) -> int:
    """
    Start an isolated paper-bot subprocess.

    The process gets its own session (``start_new_session=True``) so it
    survives terminal close and is immune to SIGINT propagation.

    Returns the child PID.
    """
    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs") / f"paper_{symbol}_M{timeframe_minutes}.log"

    runtime_env = _prepare_bot_runtime(symbol, timeframe_minutes)

    # Build env: dotenv+os.environ base  →  paper defaults  →  per-instrument
    env = {
        **base_env,
        **_PAPER_ENV_DEFAULTS,
        # Per-instrument overrides (highest priority).
        # Set BOTH the legacy short names (used by run.sh / .env checks) and the
        # CTRADER_-prefixed names that ctrader_ddqn_paper._load_and_validate_config()
        # actually reads.  Without the CTRADER_ keys the bot falls back to its
        # hardcoded defaults (XAUUSD / M1) regardless of what universe.json says.
        "SYMBOL":               symbol,
        "SYMBOL_ID":            str(symbol_id),
        "TIMEFRAME_MINUTES":    str(timeframe_minutes),
        "QTY":                  str(qty),
        "CTRADER_SYMBOL":       symbol,
        "CTRADER_SYMBOL_ID":    str(symbol_id),
        "CTRADER_TIMEFRAME_MIN": str(timeframe_minutes),
        "CTRADER_QTY":          str(qty),
        **runtime_env,
    }
    # Pass real starting equity when available so HUD balance reflects the
    # actual Pepperstone demo account (cTrader FIX doesn't expose it via
    # CollateralInquiry — the user must set starting_equity in universe.json).
    if starting_equity is not None:
        env["CTRADER_STARTING_EQUITY"] = str(starting_equity)

    python_exec = _resolve_python_executable(base_env)
    cmd = [python_exec, "-m", _BOT_MODULE]
    LOG.info(
        "Launching paper bot  %s M%d  →  %s  (symbol_id=%d, qty=%s, python=%s)",
        symbol, timeframe_minutes, log_path, symbol_id, qty, python_exec,
    )

    with open(log_path, "a") as log_fh:
        log_fh.write(
            f"\n{'='*60}\n"
            f"Paper bot started by run_universe  "
            f"{datetime.now(UTC).isoformat()}\n"
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

def launch_paper_bots(  # noqa: PLR0912, PLR0915
    registry: dict,
    specs: dict[str, dict],
    base_env: dict[str, str],
) -> dict:
    """
    Walk registry; for every PAPER-stage instrument that has no live bot,
    launch one and write the PID back.  Returns the (possibly mutated) registry.
    """
    instruments = registry.get("instruments", [])
    changed = False
    launch_stagger_s = _launch_stagger_seconds(base_env)
    topology = _broker_topology(base_env)
    owner_for_entry = _plan_fix_session_owners(instruments, topology, base_env)

    for entry in instruments:
        if not isinstance(entry, dict):
            continue
        symbol = str(entry.get("symbol", "") or "").upper()
        if not symbol:
            continue
        entry["symbol"] = symbol
        if entry.get("stage") != _PAPER_STAGE:
            continue

        tf = _entry_tf(entry)
        if not tf:
            LOG.warning("%s — missing timeframe_minutes in universe.json; skipping", symbol)
            continue

        entry_key = (symbol, tf)
        owner_key = owner_for_entry.get(entry_key)
        changed |= _annotate_fix_session_entry(entry, topology, owner_key, entry_key)

        pid = entry.get("paper_pid")
        direct_fix_allowed = topology == _TOPOLOGY_ISOLATED or owner_key in {None, entry_key}
        if not direct_fix_allowed:
            owner_symbol, owner_tf = owner_key
            if _pid_alive(pid):
                LOG.warning(
                    "%s M%d is still running direct FIX under %s topology; "
                    "restart the universe watcher to hand FIX ownership to %s M%d",
                    symbol,
                    tf,
                    topology,
                    owner_symbol,
                    owner_tf,
                )
                continue
            if pid:
                LOG.warning("%s M%d — clearing stale paper_pid %s", symbol, tf, pid)
                entry["paper_pid"] = None
                entry["paper_started_at"] = None
                changed = True
            LOG.warning(
                "%s M%d waiting for shared FIX gateway owner %s M%d; direct FIX launch disabled by %s=%s",
                symbol,
                tf,
                owner_symbol,
                owner_tf,
                _BROKER_TOPOLOGY_ENV,
                topology,
            )
            continue

        if _pid_alive(pid):
            if _runtime_weights_stale(entry, symbol, tf):
                LOG.info(
                    "%s M%d — promoted weights differ from runtime checkpoint; restarting paper bot (PID %d)",
                    symbol, tf, pid,
                )
                _stop_pid(int(pid), f"{symbol} M{tf} paper bot")
                entry["paper_pid"] = None
                entry["paper_started_at"] = None
                pid = None
                changed = True
            else:
                LOG.debug(
                    "%s M%d — paper bot already running (PID %d)",
                    symbol, tf, pid,
                )
                continue
        if pid:
            LOG.warning("%s M%d — clearing stale paper_pid %s", symbol, tf, pid)
            entry["paper_pid"] = None
            entry["paper_started_at"] = None
            changed = True

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
        starting_equity = entry.get("starting_equity")

        try:
            _sync_promoted_weights_to_runtime(entry, symbol, tf)
            new_pid = _launch_paper_bot(symbol, tf, int(symbol_id), qty, base_env, starting_equity)
            entry["paper_pid"]        = new_pid
            entry["paper_started_at"] = datetime.now(UTC).isoformat()
            entry["paper_log"]        = f"logs/paper_{symbol}_M{tf}.log"
            changed = True
            if launch_stagger_s > 0.0:
                LOG.info("Staggering next paper bot launch by %.1fs to reduce FIX logon contention", launch_stagger_s)
                time.sleep(launch_stagger_s)
        except Exception as exc:
            LOG.error("Failed to launch paper bot for %s M%d: %s", symbol, tf, exc)

    if changed:
        _save_universe(registry)

    return registry


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_list(registry: dict) -> None:
    instruments = registry.get("instruments", [])
    if not instruments:
        print(
            "Universe is empty.\n"
            "  Populate it with:  python3 train_offline.py <data> --auto-promote\n"
            "  Or manually:       python3 run_universe.py --promote XAUUSD --timeframe 240"
        )
        return

    header = (
        f"{'Symbol':<12} {'Stage':<20} {'TF':>6} {'ZOmega':>9} "
        f"{'PID':>8}  {'FIX':<18} {'Promoted':<22}  Running?"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    _rows = sorted(
        [e for e in instruments if isinstance(e, dict)],
        key=lambda x: (str(x.get("symbol", "")), int(x.get("timeframe_minutes", 0) or 0)),
    )
    for entry in _rows:
        sym = str(entry.get("symbol", "?") or "?")
        stage = entry.get("stage", "UNTRAINED")
        tf = entry.get("timeframe_minutes", "?")
        zo = entry.get("z_omega", 0.0)
        pid = entry.get("paper_pid")
        prom = (entry.get("promoted_at") or "")[:19].replace("T", " ")
        alive = "✓ running" if _pid_alive(pid) else ("✗ stopped" if pid else "—")
        zo_str = f"{zo:.4f}" if isinstance(zo, float) else str(zo)
        if entry.get("broker_topology") and entry.get("broker_topology") != _TOPOLOGY_ISOLATED:
            fix_status = "owner" if entry.get("fix_session_owner") else (
                f"wait {entry.get('fix_owner_symbol', '?')} M{entry.get('fix_owner_timeframe_minutes', '?')}"
            )
        elif entry.get("broker_topology") == _TOPOLOGY_ISOLATED:
            fix_status = _TOPOLOGY_ISOLATED
        else:
            fix_status = "—"
        print(
            f"{sym:<12} {stage:<20} {str(tf):>6} {zo_str:>9} "
            f"{str(pid or '—'):>8}  {fix_status:<18} {prom:<22}  {alive}"
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
    instruments = registry.setdefault("instruments", [])
    if not isinstance(instruments, list):
        instruments = _normalize_instruments(instruments)
        registry["instruments"] = instruments

    existing_idx = next(
        (
            i for i, e in enumerate(instruments)
            if isinstance(e, dict)
            and str(e.get("symbol", "")).upper() == symbol
            and int(e.get("timeframe_minutes", 0) or 0) == int(timeframe_minutes)
        ),
        -1,
    )
    existing = instruments[existing_idx] if existing_idx >= 0 else {}
    promoted = {
        **existing,
        "symbol":            symbol,
        "stage":             _PAPER_STAGE,
        "timeframe_minutes": timeframe_minutes,
        "z_omega":           existing.get("z_omega", z_omega),
        "promoted_at":       datetime.now(UTC).isoformat(),
        "paper_pid":         None,
        "paper_started_at":  None,
        **({"symbol_id": symbol_id} if symbol_id else {}),
    }
    if existing_idx >= 0:
        instruments[existing_idx] = promoted
    else:
        instruments.append(promoted)
    _save_universe(registry)
    LOG.info("Promoted %s M%d → PAPER", symbol, timeframe_minutes)
    return registry


def cmd_demote(registry: dict, symbol: str) -> dict:
    instruments = registry.get("instruments", [])
    if not isinstance(instruments, list):
        instruments = _normalize_instruments(instruments)
        registry["instruments"] = instruments
    matches = [
        e for e in instruments
        if isinstance(e, dict) and str(e.get("symbol", "")).upper() == symbol
    ]
    if not matches:
        LOG.warning("%s not found in universe.json", symbol)
        return registry
    for entry in matches:
        pid = entry.get("paper_pid")
        tf = int(entry.get("timeframe_minutes", 0) or 0)
        if _pid_alive(pid):
            _stop_pid(pid, f"{symbol} M{tf} paper bot")
        entry["stage"] = "UNTRAINED"
        entry["paper_pid"] = None
    _save_universe(registry)
    LOG.info("Demoted %s (%d entries) → UNTRAINED", symbol, len(matches))
    return registry


def _process_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "ignore") for part in raw.split(b"\0") if part]


def _process_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    env: dict[str, str] = {}
    for part in raw.split(b"\0"):
        if not part or b"=" not in part:
            continue
        key, value = part.split(b"=", 1)
        env[key.decode("utf-8", "ignore")] = value.decode("utf-8", "ignore")
    return env


def _process_cwd(pid: int) -> Path | None:
    try:
        return Path(f"/proc/{pid}/cwd").resolve()
    except OSError:
        return None


def _iter_managed_paper_bot_pids() -> list[tuple[int, str]]:
    """Return running paper bot PIDs managed by this project.

    This catches orphan paper bots whose PID was not saved in universe.json,
    which can happen if a supervisor is killed while it is launching children.
    """
    found: list[tuple[int, str]] = []
    current_pid = os.getpid()
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == current_pid:
            continue
        argv = _process_cmdline(pid)
        if _BOT_MODULE not in argv:
            continue
        cwd = _process_cwd(pid)
        if cwd is None:
            continue
        env = _process_environ(pid)
        data_dir = str(env.get("CTRADER_DATA_DIR", "") or "").strip()
        if not data_dir:
            continue
        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = cwd / data_path
        try:
            data_path = data_path.resolve()
            project_data = (_PROJECT_ROOT / "data").resolve()
        except OSError:
            continue
        if not data_path.is_relative_to(project_data) or not data_path.name.startswith("paper_"):
            continue
        symbol = env.get("CTRADER_SYMBOL", "?")
        tf = env.get("CTRADER_TIMEFRAME_MIN", "?")
        found.append((pid, f"{symbol} M{tf} orphan paper bot"))
    return found


def _should_scan_orphan_paper_bots() -> bool:
    """Only sweep orphan project bots when operating on the default universe."""
    try:
        return _UNIVERSE_PATH.resolve() == (_PROJECT_ROOT / "data/universe.json").resolve()
    except OSError:
        return False


def cmd_stop_all(registry: dict) -> dict:
    instruments = registry.get("instruments", [])
    if not isinstance(instruments, list):
        instruments = _normalize_instruments(instruments)
        registry["instruments"] = instruments
    stopped_pids: set[int] = set()
    for entry in instruments:
        if not isinstance(entry, dict):
            continue
        sym = str(entry.get("symbol", "?") or "?")
        tf = int(entry.get("timeframe_minutes", 0) or 0)
        pid = entry.get("paper_pid")
        if _pid_alive(pid):
            _stop_pid(pid, f"{sym} M{tf} paper bot")
            stopped_pids.add(int(pid))
            entry["paper_pid"] = None
            entry["paper_started_at"] = None
        elif pid:
            entry["paper_pid"] = None
            entry["paper_started_at"] = None

    if _should_scan_orphan_paper_bots():
        for pid, label in _iter_managed_paper_bot_pids():
            if pid in stopped_pids:
                continue
            if _pid_alive(pid):
                _stop_pid(pid, label)

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
    p.add_argument(
        "--broker-topology",
        choices=sorted(_VALID_BROKER_TOPOLOGIES),
        default=None,
        help=(
            "FIX launch topology: isolated launches one direct FIX bot per entry; "
            "shared-symbol allows one direct FIX owner per symbol; shared-account "
            "allows one direct FIX owner for the broker account while other entries "
            "wait for the shared gateway."
        ),
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

    global _UNIVERSE_PATH  # noqa: PLW0603
    _UNIVERSE_PATH = args.universe

    # Build base environment: .env values first, os.environ overrides on top
    # (already-exported vars always win; per-instrument overrides added at launch)
    dotenv   = _load_dotenv()
    base_env = {**dotenv, **os.environ}
    if args.broker_topology:
        base_env[_BROKER_TOPOLOGY_ENV] = args.broker_topology

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
    LOG.info("Broker topology: %s", _broker_topology(base_env))
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
            try:
                # Re-read file so new train_offline.py promotions are picked up
                registry = _load_universe()
                registry = launch_paper_bots(registry, specs, base_env)
            except Exception as exc:
                LOG.error("Supervisor poll error (will retry in %ds): %s", _WATCH_INTERVAL, exc)
    except KeyboardInterrupt:
        LOG.info(
            "Supervisor stopped.  Paper bots continue running in background.\n"
            "  Check status:  python3 run_universe.py --list\n"
            "  Stop all bots: python3 run_universe.py --stop-all"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
