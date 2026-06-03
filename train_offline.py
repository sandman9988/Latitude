#!/usr/bin/env python3
r"""train_offline.py.
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
    --timeframes TF [TF ...]    Timeframes to train  e.g. M1 M5 M15 M60
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

    # XAUUSD M5 + M60 only, 4 workers
    python3 train_offline.py data/history/XAUUSD_M5.csv data/history/XAUUSD_M60.csv \\
        --workers 4

    # Replay live-captured experience cache
    python3 train_offline.py data/training_cache.jsonl --timeframes M5

"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import multiprocessing
import os

# Must be set before any torch/HIP import — maps gfx1102 → gfx1100 libs.
# Inherited by ProcessPoolExecutor subprocesses via spawn context.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import re
import shutil
import sys
import tempfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

LOG = logging.getLogger("train_offline")

# HUD-visible status file — written throughout the run so progress can be
# monitored live in the Training tab without parsing the log file.
_STATUS_PATH = Path("data/offline_training_status.json")
_OFFLINE_CHAMPIONS_NAME = "offline_champions.json"
_OPTUNA_STORAGE_DIR = Path("data/optuna")
_CAPTURE_RATIO_FLOOR = 1e-9
_OFFLINE_RESUME_ENV = "CTRADER_OFFLINE_RESUME_STATUS"
_OFFLINE_RESTART_COUNT_ENV = "CTRADER_OFFLINE_RESTART_COUNT"

# Universe registry — shared with run_universe.py; records trained instruments
# and their current pipeline stage (UNTRAINED → OFFLINE_TRAINING → PAPER → …)
_UNIVERSE_PATH = Path("data/universe.json")

_STAGE_ORDER = ["UNTRAINED", "OFFLINE_TRAINING", "PAPER", "MICRO", "LIVE"]


def _tf_label(minutes: int) -> str:
    """Human-readable canonical runtime label: M{minutes}.

    Convention: always use M{minutes} format (M60/M240, not H1/H4) per
    project guidelines. Legacy H1/H4 inputs are still accepted by
    _tf_to_minutes(), but generated runtime/status labels stay canonical.
    """
    return f"M{int(minutes)}"


def _offline_supervisor_payload(argv: list[str] | None = None) -> dict[str, Any]:
    """Metadata run_universe.py can use to restart an unfinished training run."""
    script = sys.argv[0] if Path(sys.argv[0]).name == "train_offline.py" else str(Path(__file__).resolve())
    raw_argv = list(sys.argv if argv is None else [script, *argv])
    restart_count_raw = os.environ.get(_OFFLINE_RESTART_COUNT_ENV, "0")
    try:
        restart_count = max(0, int(restart_count_raw))
    except ValueError:
        restart_count = 0
    return {
        "pid": os.getpid(),
        "python": sys.executable,
        "argv": raw_argv,
        "cwd": str(Path.cwd()),
        "restartable": True,
        "restart_count": restart_count,
        "last_heartbeat": datetime.now(UTC).isoformat(),
    }


def _write_status(data: dict) -> None:
    """Atomically write offline training status for HUD consumption."""
    supervisor = data.setdefault("supervisor", _offline_supervisor_payload())
    if isinstance(supervisor, dict):
        supervisor["pid"] = os.getpid()
        supervisor["last_heartbeat"] = datetime.now(UTC).isoformat()
    _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=_STATUS_PATH.parent,
        prefix=f".{_STATUS_PATH.name}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as _f:
            json.dump(data, _f, indent=2)
            _f.flush()
            os.fsync(_f.fileno())
        tmp.replace(_STATUS_PATH)
    finally:
        tmp.unlink(missing_ok=True)


def _progress_path_for(symbol: str, timeframe_minutes: int) -> Path:
    return Path(f"data/offline_progress_{symbol}_M{timeframe_minutes}.json")


def _refresh_running_status(
    ot_status: dict[str, Any],
    jobs: list[Any],
    t_start: float,
) -> None:
    """Keep the HUD/status file live while long worker futures are still running."""
    job_keys = {(j.symbol, j.timeframe_minutes) for j in jobs}
    ot_status["elapsed_s"] = time.perf_counter() - t_start
    for entry in ot_status.get("results") or []:
        if not isinstance(entry, dict) or entry.get("status") != "running":
            continue
        try:
            symbol = str(entry.get("symbol") or "").upper()
            timeframe = int(entry.get("timeframe_minutes"))
        except (TypeError, ValueError):
            continue
        if (symbol, timeframe) not in job_keys:
            continue
        progress_path = _progress_path_for(symbol, timeframe)
        if not progress_path.exists():
            continue
        try:
            progress = json.loads(progress_path.read_text())
        except Exception:
            continue
        if not isinstance(progress, dict):
            continue
        entry["progress_pct"] = progress.get("pct")
        entry["progress_bar"] = progress.get("bar")
        entry["progress_total_bars"] = progress.get("total_bars")
        entry["progress_epoch"] = progress.get("epoch")
        entry["progress_epochs"] = progress.get("n_epochs")
        entry["progress_train_steps"] = progress.get("train_steps")
        entry["progress_trades"] = progress.get("trades")
        entry["progress_epsilon"] = progress.get("epsilon")
        entry["progress_updated_at"] = datetime.fromtimestamp(progress_path.stat().st_mtime, UTC).isoformat()
    _write_status(ot_status)


def _load_offline_status() -> dict[str, Any]:
    if not _STATUS_PATH.exists():
        return {}
    try:
        with open(_STATUS_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _status_job_key(entry: dict[str, Any]) -> tuple[str, int] | None:
    try:
        return str(entry.get("symbol", "")).upper(), int(entry.get("timeframe_minutes", 0))
    except (TypeError, ValueError):
        return None


def _completed_resume_entries(status: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """Completed jobs from a previous interrupted run that should not be repeated."""
    if os.environ.get(_OFFLINE_RESUME_ENV, "1").strip().lower() in {"0", "false", "no", "off"}:
        return {}
    if status.get("status") not in {"running", "interrupted", "stalled"}:
        return {}
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in status.get("results") or []:
        if not isinstance(entry, dict) or entry.get("status") != "done":
            continue
        key = _status_job_key(entry)
        if key is not None:
            completed[key] = dict(entry)
    return completed


def _register_universe(
    symbol: str,
    timeframe_minutes: int,
    z_omega: float,
    weights_path: str,
) -> None:
    """Promote a successfully trained instrument to PAPER stage in
    data/universe.json.

    Safe to call concurrently — uses atomic tmp-file rename.
    Never downgrades an instrument already at PAPER or above.
    """
    _UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry: dict = {"version": 1, "instruments": []}
    if _UNIVERSE_PATH.exists():
        try:
            with open(_UNIVERSE_PATH) as _f:
                registry = json.load(_f)
        except Exception:
            pass

    raw = registry.get("instruments", [])
    instruments: list[dict] = []
    if isinstance(raw, list):
        instruments = [e for e in raw if isinstance(e, dict)]
    elif isinstance(raw, dict):
        for _sym, _entry in raw.items():
            if isinstance(_entry, dict):
                instruments.append({"symbol": str(_sym).upper(), **_entry})

    existing_idx = next(
        (
            i
            for i, e in enumerate(instruments)
            if str(e.get("symbol", "")).upper() == symbol
            and int(e.get("timeframe_minutes", 0) or 0) == int(timeframe_minutes)
        ),
        -1,
    )
    existing = instruments[existing_idx] if existing_idx >= 0 else {}
    current_stage = existing.get("stage", "UNTRAINED")
    current_idx = _STAGE_ORDER.index(current_stage) if current_stage in _STAGE_ORDER else 0

    already_paper = current_idx > _STAGE_ORDER.index("OFFLINE_TRAINING")
    better_score = z_omega > existing.get("z_omega", 0.0)

    if not already_paper or better_score:
        new_stage = current_stage if already_paper else "PAPER"
        updated = {
            **existing,
            "symbol": symbol,
            "stage": new_stage,
            "timeframe_minutes": timeframe_minutes,
            "z_omega": z_omega,
            "weights_path": weights_path,
            "promoted_at": existing.get("promoted_at") or datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "paper_pid": existing.get("paper_pid"),
            "paper_started_at": existing.get("paper_started_at"),
        }
        if existing_idx >= 0:
            instruments[existing_idx] = updated
        else:
            instruments.append(updated)

        registry["instruments"] = instruments
        tmp = _UNIVERSE_PATH.with_suffix(".tmp")
        with open(tmp, "w") as _f:
            json.dump(registry, _f, indent=2)
        tmp.replace(_UNIVERSE_PATH)
        if already_paper:
            LOG.info(
                "[UNIVERSE] %s M%d ZOmega updated  (%.4f → %.4f)  stage=%s preserved",
                symbol,
                timeframe_minutes,
                existing.get("z_omega", 0.0),
                z_omega,
                new_stage,
            )
        else:
            LOG.info(
                "[UNIVERSE] %s M%d → PAPER  (ZOmega=%.4f)  Run: python3 run_universe.py",
                symbol,
                timeframe_minutes,
                z_omega,
            )
    else:
        LOG.info(
            "[UNIVERSE] %s M%d already %s ZΩ=%.4f, new ZΩ=%.4f — keeping best",
            symbol,
            timeframe_minutes,
            current_stage,
            existing.get("z_omega", 0.0),
            z_omega,
        )


# ── Import training modules ────────────────────────────────────────────────────
# Deferred to avoid importing torch/numpy before fork on some platforms
_TF_PATTERN = re.compile(r"[_\-](M\d+|H\d+|D1|W1)(?!\d)", re.IGNORECASE)
_SYM_PATTERN = re.compile(r"^([A-Z]{3,8}(?:[A-Z]{3})?)", re.IGNORECASE)


def _safe_path_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "UNKNOWN"


def _bot_checkpoint_dir(checkpoint_root: str | Path, symbol: str, timeframe_minutes: int) -> Path:
    return Path(checkpoint_root) / f"{_safe_path_token(symbol)}_M{int(timeframe_minutes)}"


def _candidate_checkpoint_dir(
    checkpoint_root: str | Path,
    symbol: str,
    timeframe_minutes: int,
    candidate_id: str = "offline_candidate",
) -> Path:
    return _bot_checkpoint_dir(checkpoint_root, symbol, timeframe_minutes) / _safe_path_token(candidate_id)


def _copy_candidate_to_runtime(weights_path: str, runtime_dir: Path) -> list[str]:
    """Copy accepted offline candidate weights to the runtime names loaded by DualPolicy."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for raw in str(weights_path or "").split(";"):
        src = Path(raw)
        if not src.exists():
            continue
        if "_trigger_" in src.name:
            dst = runtime_dir / "trigger_ddqn_weights.pt"
        elif "_harvester_" in src.name:
            dst = runtime_dir / "harvester_ddqn_weights.pt"
        else:
            continue
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
        copied.append(str(dst))
    return copied


def _deploy_candidate_result(result: dict[str, Any], checkpoint_root: str | Path) -> bool:
    """Copy one selected candidate into the deployed runtime checkpoint names."""
    if result.get("error") or result.get("accepted") is False:
        return False
    symbol = str(result.get("symbol", "") or "").upper()
    try:
        timeframe_minutes = int(result.get("timeframe_minutes", 0) or 0)
    except (TypeError, ValueError):
        return False
    if not symbol or timeframe_minutes <= 0:
        return False
    runtime_dir = _bot_checkpoint_dir(checkpoint_root, symbol, timeframe_minutes)
    copied = _copy_candidate_to_runtime(str(result.get("weights_path", "") or ""), runtime_dir)
    if not copied:
        result["accepted"] = False
        result["accept_reason"] = "candidate_weights_missing_on_deferred_deploy"
        return False
    result["accepted_weights_path"] = ";".join(copied)
    result["deployed_checkpoint_dir"] = str(runtime_dir)
    result["candidate_deploy_deferred"] = False
    LOG.info(
        "[DEPLOY] %s %s candidate %s → %s",
        symbol,
        _tf_label(timeframe_minutes),
        result.get("candidate_id", "offline_candidate"),
        runtime_dir,
    )
    return True


@contextlib.contextmanager
def _isolated_runtime_data_dir(target_dir: Path):
    """Run offline policy construction with isolated mutable runtime state."""
    source_dir = Path(os.environ.get("CTRADER_DATA_DIR", "data"))
    target_dir.mkdir(parents=True, exist_ok=True)
    source_params = source_dir / "learned_parameters.json"
    target_params = target_dir / "learned_parameters.json"
    if source_params.exists() and not target_params.exists():
        shutil.copy2(source_params, target_params)
    original = os.environ.get("CTRADER_DATA_DIR")
    os.environ["CTRADER_DATA_DIR"] = str(target_dir)
    try:
        yield target_dir
    finally:
        if original is None:
            os.environ.pop("CTRADER_DATA_DIR", None)
        else:
            os.environ["CTRADER_DATA_DIR"] = original


def _load_symbol_digits(symbol: str) -> int:
    """Return the decimal-digit count for *symbol* from symbol_specs.json (default 2).

    Strips ECN/variant suffixes (+, .crp, etc.) to find the base symbol spec.
    """
    try:
        _specs_path = Path("config/symbol_specs.json")
        if _specs_path.exists():
            import json as _j  # noqa: PLC0415

            specs = _j.loads(_specs_path.read_text())
            base = re.sub(r"[+.].*$", "", symbol)  # XAUUSD+→XAUUSD, XAUUSD.CRP→XAUUSD
            entry = specs.get(symbol) or specs.get(base)
            if entry and "digits" in entry:
                return int(entry["digits"])
    except Exception:
        pass
    return 2


# ── Job descriptor ─────────────────────────────────────────────────────────────


@dataclass
class Job:
    symbol: str
    timeframe_minutes: int
    bars_file: Path
    file_format: str  # "csv" or "jsonl"
    source_files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class TrainingVariant:
    name: str
    n_epochs: int
    train_every: int
    epsilon_start: float
    epsilon_end: float
    penalty_scale: float
    focused_cap_passes: int
    warm_start: bool
    seed_offset: int


@dataclass(frozen=True)
class OptunaSearchConfig:
    trials: int
    timeout_s: int | None
    min_val_trades: int
    storage_dir: str


@dataclass(frozen=True)
class TimeframeLimit:
    timeframe_minutes: int
    max_bars: int


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reason: str
    guard_z_omega: float
    guard_source: str


def _offline_champions_path(checkpoint_root: str | Path) -> Path:
    return Path(checkpoint_root) / _OFFLINE_CHAMPIONS_NAME


def _champion_key(symbol: str, timeframe_minutes: int) -> str:
    return f"{_safe_path_token(symbol).upper()}_M{int(timeframe_minutes)}"


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _parse_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return score


def _offline_champion_from_registry(
    checkpoint_root: str | Path,
    symbol: str,
    timeframe_minutes: int,
) -> tuple[float | None, str]:
    path = _offline_champions_path(checkpoint_root)
    registry = _load_json_dict(path)
    champions = registry.get("champions", {})
    if not isinstance(champions, dict):
        return None, ""
    entry = champions.get(_champion_key(symbol, timeframe_minutes))
    if not isinstance(entry, dict):
        return None, ""
    score = _parse_score(entry.get("z_omega"))
    return (score, str(path)) if score is not None else (None, "")


def _offline_champion_from_universe(symbol: str, timeframe_minutes: int) -> tuple[float | None, str]:
    registry = _load_json_dict(_UNIVERSE_PATH)
    instruments = registry.get("instruments", [])
    if not isinstance(instruments, list):
        return None, ""
    wanted_symbol = str(symbol).upper()
    wanted_tf = int(timeframe_minutes)
    best: float | None = None
    for entry in instruments:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("symbol", "")).upper() != wanted_symbol:
            continue
        if int(entry.get("timeframe_minutes", -1) or -1) != wanted_tf:
            continue
        score = _parse_score(entry.get("z_omega"))
        if score is not None and (best is None or score > best):
            best = score
    return (best, str(_UNIVERSE_PATH)) if best is not None else (None, "")


def _load_offline_champion(
    checkpoint_root: str | Path,
    symbol: str,
    timeframe_minutes: int,
) -> tuple[float | None, str]:
    loaders = [lambda: _offline_champion_from_registry(checkpoint_root, symbol, timeframe_minutes)]
    if Path(checkpoint_root).resolve() == Path("data/checkpoints").resolve():
        loaders.extend(
            [
                lambda: _offline_champion_from_universe(symbol, timeframe_minutes),
            ],
        )
    for loader in loaders:
        score, source = loader()
        if score is not None:
            return score, source
    return None, ""


def _decide_acceptance(
    candidate_score: float,
    incumbent_score: float,
    incumbent_loaded: bool,
    champion_score: float | None,
    acceptance_margin: float,
) -> AcceptanceDecision:
    margin = max(0.0, float(acceptance_margin))
    guard_score = float(incumbent_score) if incumbent_loaded else 0.0
    guard_source = "incumbent" if incumbent_loaded else ""
    if champion_score is not None and (not guard_source or float(champion_score) > guard_score):
        guard_score = float(champion_score)
        guard_source = "champion"

    if not guard_source:
        return AcceptanceDecision(True, "no_incumbent_checkpoint", 0.0, "none")

    if float(candidate_score) > guard_score + margin:
        return AcceptanceDecision(True, f"candidate_better_than_{guard_source}", guard_score, guard_source)

    return AcceptanceDecision(False, f"candidate_not_better_than_{guard_source}", guard_score, guard_source)


def _record_offline_champion(checkpoint_root: str | Path, result: dict[str, Any]) -> None:
    if result.get("error") or result.get("accepted") is False:
        return
    score = _parse_score(result.get("z_omega"))
    if score is None:
        return

    path = _offline_champions_path(checkpoint_root)
    registry = _load_json_dict(path)
    champions = registry.get("champions", {})
    if not isinstance(champions, dict):
        champions = {}

    symbol = str(result["symbol"]).upper()
    timeframe_minutes = int(result["timeframe_minutes"])
    key = _champion_key(symbol, timeframe_minutes)
    existing = champions.get(key)
    existing_score = _parse_score(existing.get("z_omega")) if isinstance(existing, dict) else None
    if existing_score is not None and score <= existing_score:
        return

    champions[key] = {
        "symbol": symbol,
        "timeframe_minutes": timeframe_minutes,
        "label": _tf_label(timeframe_minutes),
        "z_omega": score,
        "val_trades": int(result.get("val_trades", 0) or 0),
        "weights_path": result.get("accepted_weights_path") or result.get("weights_path", ""),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    registry["version"] = 1
    registry["champions"] = champions
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _prefer_training_result(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    if candidate.get("error") and not incumbent.get("error"):
        return False
    if incumbent.get("error") and not candidate.get("error"):
        return True
    candidate_accepted = candidate.get("accepted") is not False
    incumbent_accepted = incumbent.get("accepted") is not False
    if candidate_accepted != incumbent_accepted:
        return candidate_accepted
    return float(candidate.get("z_omega", -1.0)) > float(incumbent.get("z_omega", -1.0))


def _parse_iso_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _record_capture_ratio(record: dict[str, Any]) -> float | None:
    pnl = _parse_score(record.get("pnl_pts"))
    if pnl is None:
        pnl = _parse_score(record.get("pnl_points"))
    if pnl is None:
        pnl = _parse_score(record.get("pnl"))
    mfe = _parse_score(record.get("mfe"))
    if mfe is None:
        mfe = _parse_score(record.get("mfe_points"))
    if pnl is None or mfe is None or abs(mfe) <= _CAPTURE_RATIO_FLOOR:
        stored = _parse_score(record.get("capture_ratio"))
        if stored is not None:
            return stored
        pct = _parse_score(record.get("capture_pct"))
        return pct / 100.0 if pct is not None else None

    derived = pnl / mfe
    stored = _parse_score(record.get("capture_ratio"))
    if stored is None:
        pct = _parse_score(record.get("capture_pct"))
        if pct is not None:
            stored = pct / 100.0
    if pnl < 0.0 and derived < 0.0 and (stored is None or stored >= 0.0):
        return derived
    return stored if stored is not None else derived


def _record_replay_bars(record: dict[str, Any], parse_bar) -> list:
    seen: dict[Any, Any] = {}
    for row in record.get("entry_bars", []) + record.get("exit_bars", []):
        bar = parse_bar(row)
        if bar:
            seen[bar[0]] = bar
    return sorted(seen.values(), key=lambda bar: bar[0])


def _load_focused_cap_replay_windows(  # noqa: PLR0912
    source_files: list[str],
    symbol: str,
    timeframe_minutes: int,
    lookback_days: float,
    per_side: int,
) -> list[list]:
    if per_side <= 0 or lookback_days <= 0:
        return []

    from src.training.historical_loader import _parse_jsonl_bar  # noqa: PLC0415

    cutoff = datetime.now(UTC) - timedelta(days=float(lookback_days))
    candidates: list[tuple[float, datetime, str, int, list]] = []
    wanted_symbol = str(symbol).upper()
    wanted_tf = int(timeframe_minutes)

    for raw_path in source_files:
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for lineno, raw_line in enumerate(fh, 1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(record.get("symbol", "")).upper() != wanted_symbol:
                        continue
                    if int(record.get("timeframe_minutes", 0) or 0) != wanted_tf:
                        continue
                    ts = (
                        _parse_iso_utc(record.get("ts_recorded"))
                        or _parse_iso_utc(record.get("exit_time"))
                        or _parse_iso_utc(record.get("entry_time"))
                    )
                    if ts is None or ts < cutoff:
                        continue
                    ratio = _record_capture_ratio(record)
                    if ratio is None:
                        continue
                    bars = _record_replay_bars(record, _parse_jsonl_bar)
                    if len(bars) < _MIN_FOCUSED_REPLAY_BARS:
                        continue
                    candidates.append((float(ratio), ts, str(path), lineno, bars))
        except OSError:
            continue

    if not candidates:
        return []

    selected: dict[tuple[str, int], tuple[float, datetime, str, int, list]] = {}
    for item in sorted(candidates, key=lambda row: (row[0], row[1]))[:per_side]:
        selected[(item[2], item[3])] = item
    for item in sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True)[:per_side]:
        selected[(item[2], item[3])] = item

    return [item[4] for item in sorted(selected.values(), key=lambda row: row[1])]


def _clamp_int(value: float, floor: int, ceiling: int | None = None) -> int:
    out = max(floor, int(value))
    return min(out, ceiling) if ceiling is not None else out


def _clamp_float(value: float, floor: float, ceiling: float) -> float:
    return max(floor, min(float(value), ceiling))


def _build_training_variants(args) -> list[TrainingVariant]:
    """Build deterministic training recipes for weekend champion tournaments."""
    count = _clamp_int(getattr(args, "tournament_variants", 1), 1, 12)
    base_epochs = _clamp_int(args.n_epochs, 1)
    base_every = _clamp_int(args.train_every, 1)
    base_focus = _clamp_int(args.focused_cap_passes, 0)
    base_eps_start = _clamp_float(args.epsilon_start, 0.01, 1.0)
    base_eps_end = _clamp_float(args.epsilon_end, 0.0, 0.5)
    base_penalty = _clamp_float(args.penalty_scale, 0.0, 3.0)
    base_warm = bool(args.warm_start)

    variants = [
        TrainingVariant(
            "base",
            base_epochs,
            base_every,
            base_eps_start,
            base_eps_end,
            base_penalty,
            base_focus,
            base_warm,
            0,
        ),
        TrainingVariant(
            "capture_soft",
            base_epochs + 1,
            base_every,
            _clamp_float(max(base_eps_start, 0.35), 0.01, 1.0),
            _clamp_float(min(max(base_eps_end, 0.03), 0.06), 0.0, 0.5),
            _clamp_float(base_penalty * 0.75, 0.0, 3.0),
            base_focus + 1,
            True,
            101,
        ),
        TrainingVariant(
            "explore_high",
            base_epochs,
            base_every,
            _clamp_float(max(base_eps_start, 0.60), 0.01, 1.0),
            _clamp_float(max(base_eps_end, 0.08), 0.0, 0.5),
            base_penalty,
            base_focus + 1,
            True,
            211,
        ),
        TrainingVariant(
            "fast_updates",
            base_epochs + 1,
            _clamp_int(max(1, base_every // 2), 1),
            _clamp_float(max(base_eps_start, 0.45), 0.01, 1.0),
            _clamp_float(max(base_eps_end, 0.05), 0.0, 0.5),
            _clamp_float(base_penalty * 0.85, 0.0, 3.0),
            base_focus + 2,
            True,
            307,
        ),
        TrainingVariant(
            "strict_penalty",
            base_epochs + 1,
            base_every,
            _clamp_float(min(base_eps_start, 0.35), 0.01, 1.0),
            _clamp_float(min(max(base_eps_end, 0.02), 0.04), 0.0, 0.5),
            _clamp_float(max(base_penalty * 1.25, 1.0), 0.0, 3.0),
            base_focus + 1,
            True,
            409,
        ),
        TrainingVariant(
            "fresh_long",
            base_epochs + 2,
            base_every,
            _clamp_float(max(base_eps_start, 0.50), 0.01, 1.0),
            _clamp_float(max(base_eps_end, 0.05), 0.0, 0.5),
            _clamp_float(base_penalty * 0.90, 0.0, 3.0),
            base_focus + 2,
            False,
            503,
        ),
        TrainingVariant(
            "low_explore_long",
            base_epochs + 3,
            base_every,
            _clamp_float(min(base_eps_start, 0.25), 0.01, 1.0),
            _clamp_float(min(max(base_eps_end, 0.02), 0.03), 0.0, 0.5),
            _clamp_float(base_penalty * 0.70, 0.0, 3.0),
            base_focus + 2,
            True,
            601,
        ),
        TrainingVariant(
            "fresh_explore_fast",
            base_epochs + 2,
            _clamp_int(max(1, base_every // 2), 1),
            _clamp_float(max(base_eps_start, 0.70), 0.01, 1.0),
            _clamp_float(max(base_eps_end, 0.10), 0.0, 0.5),
            _clamp_float(base_penalty * 0.60, 0.0, 3.0),
            base_focus + 3,
            False,
            709,
        ),
    ]
    return variants[:count]


def _variant_args(args, variant: TrainingVariant, defer_candidate_deploy: bool):
    variant_args = argparse.Namespace(**vars(args))
    variant_args.n_epochs = variant.n_epochs
    variant_args.train_every = variant.train_every
    variant_args.epsilon_start = variant.epsilon_start
    variant_args.epsilon_end = variant.epsilon_end
    variant_args.penalty_scale = variant.penalty_scale
    variant_args.focused_cap_passes = variant.focused_cap_passes
    variant_args.candidate_id = (
        "offline_candidate"
        if not defer_candidate_deploy and variant.name == "base"
        else f"offline_candidate_{variant.name}"
    )
    variant_args.candidate_seed = int(getattr(args, "tournament_seed", 8675309) or 8675309) + variant.seed_offset
    variant_args.defer_candidate_deploy = defer_candidate_deploy
    return variant_args


def _parse_timeframe_limits(raw: str | None) -> dict[int, int]:
    """Parse per-timeframe max-bars limits like ``M1=500000,M5=750000``."""
    if not raw:
        return {}
    limits: dict[int, int] = {}
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid timeframe limit {item!r}; expected TF=N")
        tf_raw, value_raw = item.split("=", 1)
        tf = _tf_to_minutes(tf_raw.strip())
        if tf is None:
            raise ValueError(f"Invalid timeframe in {item!r}")
        try:
            value = int(value_raw.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid max-bars value in {item!r}") from exc
        if value <= 0:
            raise ValueError(f"Invalid max-bars value in {item!r}; must be > 0")
        limits[tf] = value
    return limits


def _job_max_bars(args, job: Job) -> int | None:
    limits = getattr(args, "_max_bars_by_timeframe", None)
    if limits is None:
        limits = _parse_timeframe_limits(getattr(args, "max_bars_by_timeframe", None))
        setattr(args, "_max_bars_by_timeframe", limits)
    if job.timeframe_minutes in limits:
        return limits[job.timeframe_minutes]
    return getattr(args, "max_bars", None)


def _optuna_storage_path(storage_dir: str | Path, symbol: str, timeframe_minutes: int) -> Path:
    """Per-bot Optuna SQLite store so searches resume independently."""
    root = Path(storage_dir)
    return root / f"offline_{_safe_path_token(symbol).upper()}_M{int(timeframe_minutes)}.db"


def _trial_float(trial, name: str, low: float, high: float, *, step: float | None = None) -> float:
    """Small wrapper so tests can use a fake trial without importing Optuna."""
    try:
        return float(trial.suggest_float(name, low, high, step=step))
    except TypeError:
        return float(trial.suggest_float(name, low, high))


def _build_optuna_trial_spec(trial, args, job: Job, trial_number: int) -> dict[str, Any]:
    """Convert one Optuna trial into the existing serialisable candidate spec."""
    base_epochs = _clamp_int(args.n_epochs, 1)
    base_every = _clamp_int(args.train_every, 1)
    base_focus = _clamp_int(args.focused_cap_passes, 0)
    base_seed = int(getattr(args, "tournament_seed", 8675309) or 8675309)

    epsilon_start = _trial_float(trial, "epsilon_start", 0.20, 1.00, step=0.05)
    epsilon_end_hi = min(0.30, max(0.02, epsilon_start - 0.05))
    epsilon_end = _trial_float(trial, "epsilon_end", 0.01, epsilon_end_hi, step=0.01)
    candidate_id = f"optuna_t{int(trial_number):04d}"
    token = f"{candidate_id}:{job.symbol}:{job.timeframe_minutes}:{base_seed}".encode()

    return {
        "n_epochs": int(trial.suggest_int("n_epochs", base_epochs, max(base_epochs + 4, base_epochs))),
        "train_every": int(trial.suggest_int("train_every", 1, max(base_every * 2, 1))),
        "warm_start": bool(trial.suggest_categorical("warm_start", [True, False])),
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "penalty_scale": _trial_float(trial, "penalty_scale", 0.50, 1.80, step=0.05),
        "focused_cap_passes": int(trial.suggest_int("focused_cap_passes", 0, max(base_focus + 4, 1))),
        "candidate_id": candidate_id,
        "candidate_seed": (base_seed + zlib.crc32(token)) % 2_147_483_647,
        "deploy_candidate": False,
    }


def _bounded_score(value: Any, *, neutral: float = 0.0, cap: float = 3.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return neutral
    if math.isnan(parsed):
        return neutral
    if math.isinf(parsed):
        return cap if parsed > 0 else -cap
    return max(-cap, min(cap, parsed))


def _optuna_objective_score(result: dict[str, Any], min_val_trades: int) -> float:
    """Rank candidates by ZOmega plus validation PnL quality, with low-trade penalties."""
    if result.get("error"):
        return -1_000.0

    z_score = _bounded_score(result.get("z_omega"), cap=5.0)
    pf = _bounded_score(result.get("val_profit_factor"), cap=3.0)
    net_pnl = _bounded_score(result.get("val_net_pnl"), cap=5.0)
    avg_pnl = _bounded_score(result.get("val_avg_pnl"), cap=3.0)
    val_trades = max(0, int(result.get("val_trades", 0) or 0))
    trade_shortfall = max(0, int(min_val_trades) - val_trades)

    return z_score + (0.25 * pf) + (0.20 * net_pnl) + (0.10 * avg_pnl) - (0.30 * trade_shortfall)


def _prefer_optuna_result(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    """Prefer accepted Optuna trials first, then the highest objective score."""
    if incumbent is None:
        return True
    if candidate.get("error") and not incumbent.get("error"):
        return False
    if incumbent.get("error") and not candidate.get("error"):
        return True
    candidate_accepted = candidate.get("accepted") is not False
    incumbent_accepted = incumbent.get("accepted") is not False
    if candidate_accepted != incumbent_accepted:
        return candidate_accepted
    return float(candidate.get("optuna_objective", -float("inf"))) > float(
        incumbent.get("optuna_objective", -float("inf")),
    )


def _candidate_seed(base_seed: int | None, candidate_id: str, job: Job) -> int | None:
    if base_seed is None:
        return None
    token = f"{candidate_id}:{job.symbol}:{job.timeframe_minutes}:{base_seed}".encode()
    return (int(base_seed) + zlib.crc32(token)) % 2_147_483_647


def _seed_worker_rng(seed: int | None) -> None:
    if seed is None:
        return
    import random  # noqa: PLC0415

    random.seed(seed)
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ── Worker function (runs in child process) ────────────────────────────────────


def _run_job(  # noqa: PLR0912, PLR0915
    symbol: str,
    timeframe_minutes: int,
    bars_file: str,
    file_format: str,
    checkpoint_dir: str,
    train_split: float,
    train_every: int,
    max_bars: int | None,
    n_epochs: int = 1,
    warm_start: bool = False,
    epsilon_start: float = 0.4,
    epsilon_end: float = 0.05,
    symbol_digits: int = 2,
    penalty_scale: float = 1.0,
    accept_if_better: bool = True,
    acceptance_margin: float = 0.0,
    source_files: list[str] | None = None,
    focused_cap_replay: bool = True,
    focused_cap_per_side: int = 10,
    focused_cap_lookback_days: float = 7.0,
    focused_cap_passes: int = 1,
    candidate_id: str = "offline_candidate",
    candidate_seed: int | None = None,
    deploy_candidate: bool = True,
) -> dict[str, Any]:
    """Child-process entry point.  Returns a dict (not a TrainResult) so it
    can be pickled cleanly across the process boundary.
    """
    import logging as _log  # noqa: PLC0415

    _log.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s][%(process)d] %(name)s: %(message)s")
    logger = _log.getLogger("train_offline.worker")

    from src.persistence.learned_parameters import LearnedParametersManager  # noqa: PLC0415
    from src.training.historical_loader import load_csv, load_jsonl_cache  # noqa: PLC0415
    from src.training.offline_trainer import OfflineTrainer  # noqa: PLC0415

    # ── Stalled-job resume detection ────────────────────────────────────────────
    # If a progress file exists and is stale (>30 min without update) the previous
    # training run stalled.  Check if a candidate checkpoint already has weights;
    # if so, force warm_start=True so the run picks up from where it left off.
    _progress_path = Path(f"data/offline_progress_{symbol}_M{timeframe_minutes}.json")
    _resume = False
    if _progress_path.exists():
        try:
            _p = json.loads(_progress_path.read_text())
            _pct = _p.get("pct", 0)
            _mtime = _progress_path.stat().st_mtime
            _stale = (time.time() - _mtime) > 1800  # 30 min stall threshold
            if _pct > 0 and _pct < 99 and _stale:
                logger.warning(
                    "[RESUME] %s %s progress=%.1f%% stale since %s — attempting resume",
                    symbol, _tf_label(timeframe_minutes), _pct,
                    datetime.fromtimestamp(_mtime).isoformat())
                _resume = True
        except Exception as _exc:
            logger.debug("[RESUME] Progress check error: %s", _exc)
    if _resume:
        _cp_dir = _candidate_checkpoint_dir(checkpoint_dir, symbol, timeframe_minutes, candidate_id)
        if _cp_dir.exists():
            logger.info("[RESUME] Found checkpoint dir %s — forcing warm_start=True", _cp_dir)
            warm_start = True
            # Also try to restore epsilon from progress for smoother continuation
            try:
                if _progress_path.exists():
                    _p = json.loads(_progress_path.read_text())
                    _eps = _p.get("epsilon")
                    if _eps is not None:
                        epsilon_start = max(float(_eps), epsilon_end)
                        logger.info("[RESUME] Restored epsilon=%.4f from progress file", epsilon_start)
            except Exception:
                pass

    candidate_id = _safe_path_token(candidate_id)
    _seed_worker_rng(candidate_seed)

    label = f"{symbol}_M{timeframe_minutes}"
    sources = source_files or [bars_file]
    logger.info(
        "[WORKER] Starting %s candidate=%s seed=%s from %s",
        label,
        candidate_id,
        candidate_seed if candidate_seed is not None else "none",
        "; ".join(sources),
    )

    try:
        if file_format == "jsonl":
            seen: dict[Any, Any] = {}
            for source in sources:
                for bar in load_jsonl_cache(source, max_bars=None):
                    seen[bar[0]] = bar
            bars = sorted(seen.values(), key=lambda bar: bar[0])
            if max_bars and len(bars) > max_bars:
                bars = bars[-max_bars:]
            if len(sources) > 1:
                logger.info(
                    "[WORKER] %s merged %d JSONL cache(s) into %d unique bars",
                    label,
                    len(sources),
                    len(bars),
                )
            focused_replay_windows = (
                _load_focused_cap_replay_windows(
                    sources,
                    symbol,
                    timeframe_minutes,
                    focused_cap_lookback_days,
                    focused_cap_per_side,
                )
                if focused_cap_replay
                else []
            )
            if focused_replay_windows:
                logger.info(
                    "[WORKER] %s focused CAP replay selected %d weekly best/worst window(s)",
                    label,
                    len(focused_replay_windows),
                )
        else:
            bars = load_csv(bars_file, max_bars=max_bars, timeframe_minutes=timeframe_minutes)
            focused_replay_windows = (
                _load_focused_cap_replay_windows(
                    sources,
                    symbol,
                    timeframe_minutes,
                    focused_cap_lookback_days,
                    focused_cap_per_side,
                )
                if focused_cap_replay
                else []
            )
    except Exception as exc:
        logger.exception("[WORKER] %s: failed to load bars: %s", label, exc)
        return {
            "symbol": symbol,
            "timeframe_minutes": timeframe_minutes,
            "z_omega": 0.0,
            "train_trades": 0,
            "val_trades": 0,
            "total_train_steps": 0,
            "elapsed_s": 0.0,
            "weights_path": "",
            "error": str(exc),
            "candidate_id": candidate_id,
            "candidate_seed": candidate_seed,
        }

    # Load reward-shaping params from learned_parameters.json (auto-backfills defaults)
    _pm = LearnedParametersManager()
    _tf_lbl = _tf_label(timeframe_minutes)

    def _lp(name: str, default: float) -> float:
        return float(_pm.get(symbol, name, timeframe=_tf_lbl, default=default))

    bot_checkpoint_dir = _bot_checkpoint_dir(checkpoint_dir, symbol, timeframe_minutes)
    candidate_checkpoint_dir = _candidate_checkpoint_dir(
        checkpoint_dir,
        symbol,
        timeframe_minutes,
        candidate_id,
    )
    offline_runtime_dir = candidate_checkpoint_dir / "runtime_state"

    trainer = OfflineTrainer(
        symbol=symbol,
        timeframe_minutes=timeframe_minutes,
        bars=bars,
        checkpoint_dir=candidate_checkpoint_dir,
        train_split=train_split,
        train_every=train_every,
        n_epochs=n_epochs,
        warm_start=warm_start,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        symbol_digits=symbol_digits,
        reward_clip_harvester=_lp("reward_clip_harvester", 2.0),
        reward_clip_trigger=_lp("reward_clip_trigger", 0.5),
        capture_baseline=_lp("capture_baseline", 0.5),
        penalty_scale=penalty_scale,
        focused_replay_windows=focused_replay_windows,
        focused_replay_passes=focused_cap_passes if focused_cap_replay else 0,
    )
    incumbent_z_omega = 0.0
    incumbent_val_trades = 0
    incumbent_loaded = False
    champion_z_omega, champion_source = _load_offline_champion(
        checkpoint_dir,
        symbol,
        timeframe_minutes,
    )
    with _isolated_runtime_data_dir(offline_runtime_dir):
        if accept_if_better:
            incumbent_z_omega, incumbent_val_trades, incumbent_loaded = trainer.evaluate_runtime_checkpoint(
                bot_checkpoint_dir,
            )
            legacy_checkpoint_dir = Path(checkpoint_dir)
            if not incumbent_loaded and legacy_checkpoint_dir != bot_checkpoint_dir:
                incumbent_z_omega, incumbent_val_trades, incumbent_loaded = trainer.evaluate_runtime_checkpoint(
                    legacy_checkpoint_dir,
                )

        result = trainer.run()
    accepted = False
    accepted_paths: list[str] = []
    accept_reason = "acceptance_disabled"
    acceptance_guard_z_omega = 0.0
    acceptance_guard_source = "none"
    if not result.error and accept_if_better:
        candidate_score = float(result.z_omega)
        decision = _decide_acceptance(
            candidate_score=candidate_score,
            incumbent_score=float(incumbent_z_omega),
            incumbent_loaded=incumbent_loaded,
            champion_score=champion_z_omega,
            acceptance_margin=acceptance_margin,
        )
        accepted = decision.accepted
        accept_reason = decision.reason
        acceptance_guard_z_omega = decision.guard_z_omega
        acceptance_guard_source = decision.guard_source
        if accepted and deploy_candidate:
            accepted_paths = _copy_candidate_to_runtime(result.weights_path, bot_checkpoint_dir)
            if not accepted_paths:
                accepted = False
                accept_reason = "candidate_weights_missing"
        elif accepted:
            accept_reason = f"{accept_reason}_deferred"
    elif not result.error:
        accepted = True
        accept_reason = "acceptance_disabled_candidate_saved_only"
    return {
        "symbol": result.symbol,
        "timeframe_minutes": result.timeframe_minutes,
        "candidate_id": candidate_id,
        "candidate_seed": candidate_seed,
        "candidate_deploy_deferred": bool(accepted and not deploy_candidate),
        "z_omega": result.z_omega,
        "val_net_pnl": result.val_net_pnl,
        "val_avg_pnl": result.val_avg_pnl,
        "val_profit_factor": result.val_profit_factor,
        "incumbent_z_omega": incumbent_z_omega,
        "incumbent_val_trades": incumbent_val_trades,
        "incumbent_loaded": incumbent_loaded,
        "champion_z_omega": champion_z_omega,
        "champion_source": champion_source,
        "acceptance_guard_z_omega": acceptance_guard_z_omega,
        "acceptance_guard_source": acceptance_guard_source,
        "accepted": accepted,
        "accept_reason": accept_reason,
        "accepted_weights_path": ";".join(accepted_paths),
        "deployed_checkpoint_dir": str(bot_checkpoint_dir),
        "offline_runtime_dir": str(offline_runtime_dir),
        "train_trades": result.train_trades,
        "val_trades": result.val_trades,
        "total_train_steps": result.total_train_steps,
        "elapsed_s": result.elapsed_s,
        "weights_path": result.weights_path,
        "error": result.error,
    }


def _run_job_tournament(
    symbol: str,
    timeframe_minutes: int,
    bars_file: str,
    file_format: str,
    checkpoint_dir: str,
    train_split: float,
    max_bars: int | None,
    accept_if_better: bool,
    acceptance_margin: float,
    source_files: list[str] | None,
    focused_cap_replay: bool,
    focused_cap_per_side: int,
    focused_cap_lookback_days: float,
    symbol_digits: int,
    variant_specs: list[dict],
) -> dict[str, Any]:
    """Run all tournament variants for one (symbol, TF) job in this process.

    Each variant is trained sequentially here; the caller runs multiple jobs in
    parallel so no timeframe waits for another timeframe's variants to finish.
    Returns the best result across all variants by ZOmega.
    """
    import logging as _log  # noqa: PLC0415

    _log.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s][%(process)d] %(name)s: %(message)s")
    logger = _log.getLogger("train_offline.tournament_worker")

    label = f"{symbol}_M{timeframe_minutes}"
    best: dict[str, Any] | None = None

    for idx, v in enumerate(variant_specs):
        logger.info(
            "[TOURNAMENT-WORKER] %s variant %d/%d: %s",
            label,
            idx + 1,
            len(variant_specs),
            v.get("candidate_id", "?"),
        )
        result = _run_job(
            symbol=symbol,
            timeframe_minutes=timeframe_minutes,
            bars_file=bars_file,
            file_format=file_format,
            checkpoint_dir=checkpoint_dir,
            train_split=train_split,
            train_every=v["train_every"],
            max_bars=max_bars,
            n_epochs=v["n_epochs"],
            warm_start=v["warm_start"],
            epsilon_start=v["epsilon_start"],
            epsilon_end=v["epsilon_end"],
            symbol_digits=symbol_digits,
            penalty_scale=v["penalty_scale"],
            accept_if_better=accept_if_better,
            acceptance_margin=acceptance_margin,
            source_files=source_files,
            focused_cap_replay=focused_cap_replay,
            focused_cap_per_side=focused_cap_per_side,
            focused_cap_lookback_days=focused_cap_lookback_days,
            focused_cap_passes=v["focused_cap_passes"],
            candidate_id=v["candidate_id"],
            candidate_seed=v["candidate_seed"],
            deploy_candidate=v["deploy_candidate"],
        )
        if result.get("error"):
            logger.warning(
                "[TOURNAMENT-WORKER] %s variant %s errored: %s",
                label,
                v.get("candidate_id"),
                result["error"],
            )
            continue
        if best is None or _prefer_training_result(result, best):
            best = result
            logger.info(
                "[TOURNAMENT-WORKER] %s new best from %s: ZΩ=%.4f",
                label,
                v.get("candidate_id"),
                result.get("z_omega", 0.0),
            )

    if best is None:
        return {
            "symbol": symbol,
            "timeframe_minutes": timeframe_minutes,
            "z_omega": 0.0,
            "train_trades": 0,
            "val_trades": 0,
            "total_train_steps": 0,
            "elapsed_s": 0.0,
            "weights_path": "",
            "error": "all variants failed",
            "candidate_id": variant_specs[0].get("candidate_id", "tournament") if variant_specs else "tournament",
            "candidate_seed": None,
        }
    return best


def _run_job_optuna(  # noqa: PLR0915
    symbol: str,
    timeframe_minutes: int,
    bars_file: str,
    file_format: str,
    checkpoint_dir: str,
    train_split: float,
    max_bars: int | None,
    accept_if_better: bool,
    acceptance_margin: float,
    source_files: list[str] | None,
    focused_cap_replay: bool,
    focused_cap_per_side: int,
    focused_cap_lookback_days: float,
    symbol_digits: int,
    base_args: dict[str, Any],
    search_config: OptunaSearchConfig,
) -> dict[str, Any]:
    """Run Optuna-generated candidates for one symbol/timeframe job."""
    import logging as _log  # noqa: PLC0415

    _log.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s][%(process)d] %(name)s: %(message)s")
    logger = _log.getLogger("train_offline.optuna_worker")

    try:
        import optuna  # noqa: PLC0415
    except Exception as exc:
        return {
            "symbol": symbol,
            "timeframe_minutes": timeframe_minutes,
            "z_omega": 0.0,
            "train_trades": 0,
            "val_trades": 0,
            "total_train_steps": 0,
            "elapsed_s": 0.0,
            "weights_path": "",
            "error": f"Optuna unavailable: {exc}",
            "candidate_id": "optuna",
            "candidate_seed": None,
        }

    label = f"{symbol}_M{timeframe_minutes}"
    storage_path = _optuna_storage_path(search_config.storage_dir, symbol, timeframe_minutes)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study_name = f"offline_{_safe_path_token(symbol).upper()}_M{int(timeframe_minutes)}"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    args = argparse.Namespace(**base_args)
    job = Job(symbol, timeframe_minutes, Path(bars_file), file_format)
    best: dict[str, Any] | None = None

    def _completed_trial_result(trial) -> dict[str, Any]:
        attrs = trial.user_attrs
        return {
            "symbol": symbol,
            "timeframe_minutes": timeframe_minutes,
            "z_omega": attrs.get("z_omega", 0.0),
            "val_net_pnl": attrs.get("val_net_pnl", 0.0),
            "val_avg_pnl": attrs.get("val_avg_pnl", 0.0),
            "val_profit_factor": attrs.get("val_profit_factor", 0.0),
            "val_trades": attrs.get("val_trades", 0),
            "train_trades": attrs.get("train_trades", 0),
            "total_train_steps": attrs.get("total_train_steps", 0),
            "elapsed_s": attrs.get("elapsed_s", 0.0),
            "weights_path": attrs.get("weights_path", ""),
            "accepted": attrs.get("accepted", False),
            "accept_reason": attrs.get("accept_reason", ""),
            "candidate_id": attrs.get("candidate_id", f"optuna_t{int(trial.number):04d}"),
            "candidate_seed": attrs.get("candidate_seed"),
            "optuna_objective": float(trial.value),
            "optuna_trial_number": trial.number,
            "optuna_params": dict(trial.params),
            "error": "",
        }

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if completed_trials:
        best_trial = completed_trials[0]
        for trial in completed_trials:
            candidate = _completed_trial_result(trial)
            if _prefer_optuna_result(candidate, best):
                best = candidate
                best_trial = trial
        logger.info(
            "[OPTUNA] %s resuming study with %d completed trial(s); best trial=%d objective=%.4f",
            label,
            len(completed_trials),
            best_trial.number,
            float(best_trial.value),
        )
    remaining_trials = max(0, int(search_config.trials) - len(completed_trials))
    if remaining_trials <= 0:
        if best is not None:
            best["optuna_storage"] = str(storage_path)
            best["optuna_study_name"] = study.study_name
            best["optuna_best_value"] = best.get("optuna_objective", 0.0)
            return best
        remaining_trials = 1

    def objective(trial) -> float:
        nonlocal best
        spec = _build_optuna_trial_spec(trial, args, job, trial.number)
        logger.info(
            "[OPTUNA] %s trial=%d candidate=%s epochs=%d train_every=%d eps=%.3f->%.3f penalty=%.3f focused=%d",
            label,
            trial.number,
            spec["candidate_id"],
            spec["n_epochs"],
            spec["train_every"],
            spec["epsilon_start"],
            spec["epsilon_end"],
            spec["penalty_scale"],
            spec["focused_cap_passes"],
        )
        result = _run_job(
            symbol=symbol,
            timeframe_minutes=timeframe_minutes,
            bars_file=bars_file,
            file_format=file_format,
            checkpoint_dir=checkpoint_dir,
            train_split=train_split,
            train_every=spec["train_every"],
            max_bars=max_bars,
            n_epochs=spec["n_epochs"],
            warm_start=spec["warm_start"],
            epsilon_start=spec["epsilon_start"],
            epsilon_end=spec["epsilon_end"],
            symbol_digits=symbol_digits,
            penalty_scale=spec["penalty_scale"],
            accept_if_better=accept_if_better,
            acceptance_margin=acceptance_margin,
            source_files=source_files,
            focused_cap_replay=focused_cap_replay,
            focused_cap_per_side=focused_cap_per_side,
            focused_cap_lookback_days=focused_cap_lookback_days,
            focused_cap_passes=spec["focused_cap_passes"],
            candidate_id=spec["candidate_id"],
            candidate_seed=spec["candidate_seed"],
            deploy_candidate=False,
        )
        score = _optuna_objective_score(result, search_config.min_val_trades)
        result["optuna_objective"] = score
        result["optuna_trial_number"] = trial.number
        result["optuna_params"] = {
            key: spec[key]
            for key in (
                "n_epochs",
                "train_every",
                "warm_start",
                "epsilon_start",
                "epsilon_end",
                "penalty_scale",
                "focused_cap_passes",
            )
        }
        trial.set_user_attr("candidate_id", spec["candidate_id"])
        trial.set_user_attr("z_omega", result.get("z_omega", 0.0))
        trial.set_user_attr("val_trades", result.get("val_trades", 0))
        trial.set_user_attr("val_net_pnl", result.get("val_net_pnl", 0.0))
        trial.set_user_attr("val_avg_pnl", result.get("val_avg_pnl", 0.0))
        trial.set_user_attr("val_profit_factor", result.get("val_profit_factor", 0.0))
        trial.set_user_attr("accepted", result.get("accepted", False))
        trial.set_user_attr("accept_reason", result.get("accept_reason", ""))
        trial.set_user_attr("weights_path", result.get("weights_path", ""))
        trial.set_user_attr("train_trades", result.get("train_trades", 0))
        trial.set_user_attr("total_train_steps", result.get("total_train_steps", 0))
        trial.set_user_attr("elapsed_s", result.get("elapsed_s", 0.0))
        trial.set_user_attr("candidate_seed", result.get("candidate_seed"))

        if not result.get("error") and _prefer_optuna_result(result, best):
            best = result
            logger.info(
                "[OPTUNA] %s new best trial=%d objective=%.4f ZΩ=%.4f val_trades=%d avg_pnl=%.4f",
                label,
                trial.number,
                score,
                result.get("z_omega", 0.0),
                result.get("val_trades", 0),
                result.get("val_avg_pnl", 0.0),
            )
        return score

    study.optimize(
        objective,
        n_trials=remaining_trials,
        timeout=search_config.timeout_s,
        gc_after_trial=True,
    )

    if best is None:
        return {
            "symbol": symbol,
            "timeframe_minutes": timeframe_minutes,
            "z_omega": 0.0,
            "train_trades": 0,
            "val_trades": 0,
            "total_train_steps": 0,
            "elapsed_s": 0.0,
            "weights_path": "",
            "error": "all optuna trials failed",
            "candidate_id": "optuna",
            "candidate_seed": None,
            "optuna_storage": str(storage_path),
        }
    best["optuna_storage"] = str(storage_path)
    best["optuna_study_name"] = study.study_name
    best["optuna_best_value"] = best.get("optuna_objective", 0.0)
    return best


# ── Job discovery ──────────────────────────────────────────────────────────────

_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "H12": 720, "D1": 1440, "W1": 10080}
_MIN_FOCUSED_REPLAY_BARS = 80


def _tf_to_minutes(label: str) -> int | None:
    text = str(label or "").strip().upper()
    if text in _TF_MINUTES:
        return _TF_MINUTES[text]
    if text.startswith("M") and text[1:].isdigit():
        return int(text[1:])
    if text.startswith("H") and text[1:].isdigit():
        return int(text[1:]) * 60
    if text == "D1":
        return 1440
    if text == "W1":
        return 10080
    return None


def _detect_tf_minutes(filename: str) -> int | None:
    m = _TF_PATTERN.search(filename)
    if m:
        return _tf_to_minutes(m.group(1))
    return None


def _detect_symbol(filename: str) -> str | None:
    """Extract broker symbol from filename, preserving ECN/variant suffixes.

    Uses the timeframe marker as a right-boundary delimiter so that symbols
    like XAUUSD+, XAUUSD.crp and BTCUSD are returned in full rather than
    being truncated to their bare alphabetic prefix.

    Examples
    --------
    XAUUSD_M60.csv              → XAUUSD
    XAUUSD+_M60_20240101.csv    → XAUUSD+
    XAUUSD.crp_H4_20240101.csv  → XAUUSD.CRP
    BTCUSD_M15.csv              → BTCUSD

    """
    stem = Path(filename).stem  # drop extension
    if stem.lower().startswith("training_cache_"):
        stem = stem[len("training_cache_") :]
    m = _TF_PATTERN.search(stem)
    if m:
        sym = stem[: m.start()]  # everything before _M15 / _M60 / legacy _H1 / etc.
        if sym:
            return sym.upper()
    # Fallback for filenames without a recognised TF marker
    m2 = _SYM_PATTERN.match(stem.upper())
    return m2.group(1) if m2 else None


def discover_jobs(  # noqa: PLR0912
    paths: list[str],
    symbol_filter: list[str] | None = None,
    tf_filter: list[str] | None = None,
) -> list[Job]:
    """Expand paths (files + directories) into a list of Job descriptors."""
    tf_minutes_filter = {v for v in (_tf_to_minutes(t) for t in tf_filter or []) if v is not None} or None
    sym_filter_upper = {s.upper() for s in symbol_filter} if symbol_filter else None

    found: list[Job] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files = sorted(p.rglob("*.csv")) + sorted(p.rglob("*.jsonl"))
        elif p.exists():
            files = [p]
        else:
            LOG.warning("Path not found: %s", raw)
            continue

        for f in files:
            fmt = "jsonl" if f.suffix.lower() == ".jsonl" else "csv"
            sym = _detect_symbol(f.name)
            tf = _detect_tf_minutes(f.name)

            if sym is None:
                LOG.warning("Could not detect symbol from filename: %s — skipping", f.name)
                continue

            if fmt == "jsonl" and tf is None:
                LOG.warning("JSONL file %s has no TF in filename; will use --timeframes or skip", f.name)

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

    # Deduplicate: one job per (symbol, timeframe_minutes). Duplicate JSONL
    # caches are merged by the worker so retraining sees the complete available
    # per-bot cache instead of whichever path sorted first.
    groups: dict[tuple, list[Job]] = {}
    for j in found:
        key = (j.symbol, j.timeframe_minutes)
        groups.setdefault(key, []).append(j)

    return [_combine_duplicate_jobs(candidates) for candidates in groups.values()]


def _file_source_score(path: Path) -> tuple[int, float, int]:
    try:
        stat = path.stat()
    except OSError:
        return (0, 0.0, 0)
    return (int(stat.st_size), float(stat.st_mtime), -len(str(path)))


def _prefer_discovered_job(candidate: Job, incumbent: Job) -> bool:
    if candidate.bars_file.name == incumbent.bars_file.name:
        return _file_source_score(candidate.bars_file) > _file_source_score(incumbent.bars_file)
    return len(candidate.bars_file.name) < len(incumbent.bars_file.name)


def _job_source_files(job: Job) -> tuple[Path, ...]:
    return job.source_files or (job.bars_file,)


def _combine_duplicate_jobs(candidates: list[Job]) -> Job:
    """Merge duplicate jobs for the same (symbol, TF) into one.

    - JSONL-only duplicates: combine into a single JSONL job with all sources.
    - CSV + JSONL mix: prefer CSV (full history) but include JSONL as extra
      source so the worker can also read the live paper cache.
    """
    jsonl_jobs = [j for j in candidates if j.file_format == "jsonl"]
    csv_jobs = [j for j in candidates if j.file_format == "csv"]

    if jsonl_jobs and csv_jobs:
        # Prefer CSV (full history), include JSONLs as extra source files
        all_sources = {j.bars_file for j in jsonl_jobs}
        all_sources.add(csv_jobs[0].bars_file)
        source_files = tuple(
            sorted(all_sources, key=_file_source_score, reverse=True),
        )
        primary = csv_jobs[0]
        for candidate in csv_jobs[1:]:
            if _prefer_discovered_job(candidate, primary):
                primary = candidate
        return Job(
            primary.symbol,
            primary.timeframe_minutes,
            primary.bars_file,
            primary.file_format,  # csv
            source_files,
        )

    if len(jsonl_jobs) > 1:
        source_files = tuple(
            sorted(
                {j.bars_file for j in jsonl_jobs},
                key=_file_source_score,
                reverse=True,
            ),
        )
        primary = jsonl_jobs[0]
        for candidate in jsonl_jobs[1:]:
            if _prefer_discovered_job(candidate, primary):
                primary = candidate
        return Job(
            primary.symbol,
            primary.timeframe_minutes,
            primary.bars_file,
            primary.file_format,
            source_files,
        )

    primary = candidates[0]
    for candidate in candidates[1:]:
        if _prefer_discovered_job(candidate, primary):
            primary = candidate
    return primary


# ── Pre-flight data integrity check ───────────────────────────────────────────


def preflight_check(jobs: list[Job], min_rows: int = 50) -> tuple[list[Job], list[str]]:  # noqa: PLR0912
    """Validate each job's data file before spawning any workers.

    Checks performed:
      1. Symlink resolves to a real file (catches broken symlinks immediately)
      2. File can be opened and read
      3. Header row is present (CSV) or first line parses as JSON (JSONL)
      4. File contains at least ``min_rows`` data rows

    Returns:
      (good_jobs, error_messages)

    """
    good: list[Job] = []
    errors: list[str] = []

    for j in jobs:
        paths = _job_source_files(j)
        label = f"{j.symbol} {_tf_label(j.timeframe_minutes)}"
        row_count = 0
        source_errors: list[str] = []

        for path in paths:
            # 1. Resolve symlink — Path.exists() follows symlinks
            if not path.exists():
                # Distinguish broken symlink from missing file
                if path.is_symlink():
                    target = path.resolve()
                    source_errors.append(f"broken symlink {path} → {target} (target does not exist)")
                else:
                    source_errors.append(f"file not found: {path}")
                continue

            # 2 + 3 + 4. Open, read header, count rows
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    first = fh.readline()
                    if not first.strip():
                        source_errors.append(f"file is empty: {path}")
                        continue

                    if j.file_format == "jsonl":
                        import json as _json  # noqa: PLC0415

                        try:
                            _json.loads(first)
                        except Exception as exc:
                            source_errors.append(f"first line is not valid JSON ({exc}): {path}")
                            continue
                        row_count += 1
                    # CSV: first line is header — just presence is enough

                    # Count data rows (up to min_rows + 1 to avoid reading huge files)
                    for _ in fh:
                        row_count += 1
                        if row_count >= min_rows:
                            break

                    if row_count >= min_rows:
                        break

            except OSError as exc:
                source_errors.append(f"cannot read {path}: {exc}")
                continue

        if source_errors:
            errors.append(f"[PREFLIGHT] {label}: {'; '.join(source_errors)}")
            continue

        if row_count < min_rows:
            source_desc = "; ".join(str(path) for path in paths)
            errors.append(f"[PREFLIGHT] {label}: only {row_count} data rows (need ≥ {min_rows}): {source_desc}")
            continue

        good.append(j)

    return good, errors


# ── Best-model selection ───────────────────────────────────────────────────────


def select_best(results: list[dict]) -> dict[str, dict]:
    """For each symbol, select the timeframe with the highest ZOmega.

    Returns:
        { symbol: best_result_dict }

    """
    best: dict[str, dict] = {}
    for r in results:
        if r.get("error") or r.get("accepted") is False:
            continue
        sym = r["symbol"]
        score = r.get("z_omega", 0.0)
        if sym not in best or score > best[sym].get("z_omega", -1.0):
            best[sym] = r
    return best


def select_best_per_bot(results: list[dict]) -> dict[tuple[str, int], dict]:
    """For each symbol/timeframe, select the best accepted result."""
    best: dict[tuple[str, int], dict] = {}
    for r in results:
        if r.get("error") or r.get("accepted") is False:
            continue
        key = (r["symbol"], int(r["timeframe_minutes"]))
        score = r.get("z_omega", 0.0)
        if key not in best or score > best[key].get("z_omega", -1.0):
            best[key] = r
    return best


def _best_weight_filename(result: dict, src: Path) -> str | None:
    name = src.name.lower()
    if "trigger" in name:
        agent = "trigger"
    elif "harvester" in name:
        agent = "harvester"
    else:
        return None
    sym = _safe_path_token(str(result.get("symbol", "UNKNOWN")).upper())
    tf = int(result.get("timeframe_minutes", 0) or 0)
    suffix = src.suffix or ".pt"
    return f"{sym}_M{tf}_{agent}_offline{suffix}"


def copy_best_weights(best: dict[tuple[str, int], dict], dest_dir: Path) -> None:
    """Copy accepted per-bot weight files to dest_dir/{symbol}_M{tf}_{agent}_offline.pt."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for r in best.values():
        paths_str = r.get("accepted_weights_path") or r.get("weights_path", "")
        if not paths_str:
            continue
        for src_path in paths_str.split(";"):
            src = Path(src_path)
            if not src.exists():
                continue
            canonical = _best_weight_filename(r, src)
            if canonical is None:
                continue
            dst = dest_dir / canonical
            try:
                shutil.copy2(src, dst)
                LOG.info("[BEST] %s → %s  (ZOmega=%.4f)", src.name, dst.name, r["z_omega"])
            except Exception as exc:
                LOG.warning("[BEST] Could not copy %s: %s", src, exc)


def _retrain_eligible(
    best_result: dict | None,
    threshold: float,
    negative_only: bool,
) -> bool:
    if not best_result or best_result.get("error"):
        return False
    if best_result.get("accepted") is False:
        return True
    z_omega = float(best_result.get("z_omega", 0.0))
    if negative_only:
        return z_omega < 0.0
    return z_omega < threshold


# ── Summary table ──────────────────────────────────────────────────────────────


def print_summary(results: list[dict]) -> None:
    header = (
        f"{'Symbol':<12} {'TF':>5} {'Trades':>7} {'ValTrades':>9} "
        f"{'Steps':>7} {'ZOmega':>9} {'AvgPnL':>9} {'PF':>7} {'Guard':>9} {'Time':>8}  Status"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in sorted(results, key=lambda x: (x["symbol"], x["timeframe_minutes"])):
        label = _tf_label(r["timeframe_minutes"])
        status = f"ERROR: {r['error'][:40]}" if r.get("error") else "OK"
        zo = r.get("z_omega", 0.0)
        zo_str = f"{zo:.4f}" if zo != float("inf") else "  +inf"
        avg_pnl = float(r.get("val_avg_pnl", 0.0) or 0.0)
        pf = r.get("val_profit_factor", 0.0)
        pf_str = f"{float(pf):.2f}" if pf != float("inf") else "+inf"
        inc = r.get("acceptance_guard_z_omega", r.get("incumbent_z_omega", 0.0))
        inc_str = f"{inc:.4f}" if inc != float("inf") else "  +inf"
        if not r.get("error") and r.get("accepted") is False:
            status = f"REJECTED: {r.get('accept_reason', 'not_better')}"
        elif not r.get("error") and r.get("accepted") is True:
            status = f"ACCEPTED: {r.get('accept_reason', 'better')}"
        print(
            f"{r['symbol']:<12} {label:>5} {r['train_trades']:>7} "
            f"{r['val_trades']:>9} {r['total_train_steps']:>7} "
            f"{zo_str:>9} {avg_pnl:>9.4f} {pf_str:>7} {inc_str:>9} {r['elapsed_s']:>7.1f}s  {status}"
        )
    print(sep)


# ── CLI ────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parallelised offline DDQN training across multiple instruments / timeframes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Options")[0],
    )
    p.add_argument(
        "inputs", nargs="+", metavar="FILE_OR_DIR", help="CSV files, JSONL caches, or directories containing them",
    )
    p.add_argument(
        "--symbols", nargs="+", default=None, metavar="SYM", help="Filter to specific symbols (default: all)",
    )
    p.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        metavar="TF",
        help="Timeframe codes to train e.g. M5 M60 (default: all detected; legacy H1 input is accepted)",
    )
    p.add_argument("--workers", type=int, default=None, help="Worker processes (default: CPU count)")
    p.add_argument(
        "--allow-gpu-parallel",
        action="store_true",
        default=False,
        help=(
            "Honor --workers > 1 even when a GPU is visible. Default is one "
            "worker on GPU hosts to avoid VRAM contention."
        ),
    )
    p.add_argument("--checkpoint-dir", default="data/checkpoints", metavar="PATH")
    p.add_argument("--train-split", type=float, default=0.80, metavar="0.8")
    p.add_argument("--train-every", type=int, default=4, metavar="N")
    p.add_argument("--max-bars", type=int, default=None, metavar="N")
    p.add_argument(
        "--max-bars-by-timeframe",
        default=None,
        metavar="TF=N[,TF=N]",
        help=(
            "Per-timeframe bar caps, e.g. M1=500000,M5=750000. "
            "Overrides --max-bars for matching timeframes."
        ),
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=1,
        metavar="N",
        help="Number of training passes over the data per job (default: 1)",
    )
    p.add_argument(
        "--warm-start",
        action="store_true",
        default=False,
        help="Load existing checkpoint weights before training (continue from prior run)",
    )
    p.add_argument(
        "--epsilon-start",
        type=float,
        default=0.4,
        metavar="E",
        help="Epsilon at the start of each training epoch (default: 0.4)",
    )
    p.add_argument(
        "--epsilon-end", type=float, default=0.05, metavar="E", help="Epsilon floor / val epsilon (default: 0.05)",
    )
    p.add_argument(
        "--penalty-scale",
        type=float,
        default=1.0,
        metavar="S",
        help="Scale factor for WTL/timing penalties (0.0=no penalty, 1.0=full, default: 1.0)",
    )
    p.add_argument(
        "--accept-if-better",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate candidate weights against the deployed per-bot checkpoint "
            "and only copy them into runtime checkpoint files if better "
            "(default: enabled)."
        ),
    )
    p.add_argument(
        "--acceptance-margin",
        type=float,
        default=0.0,
        metavar="ZO",
        help="Required ZOmega improvement over incumbent before accepting candidate (default: 0.0)",
    )
    p.add_argument(
        "--focused-cap-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run a bounded training-only replay pass over the 10 best and 10 worst "
            "recent CAP%% cache records for each symbol/timeframe (default: enabled)."
        ),
    )
    p.add_argument(
        "--focused-cap-per-side",
        type=int,
        default=10,
        metavar="N",
        help="Best and worst CAP%% replay records to select per job (default: 10 each)",
    )
    p.add_argument(
        "--focused-cap-lookback-days",
        type=float,
        default=7.0,
        metavar="DAYS",
        help="Lookback window for focused CAP%% replay records (default: 7)",
    )
    p.add_argument(
        "--focused-cap-passes",
        type=int,
        default=1,
        metavar="N",
        help="Training-only passes over selected focused CAP%% replay windows (default: 1)",
    )
    p.add_argument(
        "--tournament-variants",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run N deterministic training recipes per job and deploy only the best candidate "
            "that beats the per-symbol/timeframe guard (default: 1)"
        ),
    )
    p.add_argument(
        "--tournament-seed",
        type=int,
        default=8675309,
        metavar="N",
        help="Base RNG seed for tournament candidates (default: 8675309)",
    )
    p.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run N Optuna-generated candidate trials per symbol/timeframe instead of deterministic "
            "tournament variants (default: 0 = disabled)."
        ),
    )
    p.add_argument(
        "--optuna-timeout-secs",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Optional per-job Optuna timeout in seconds (default: 0 = no timeout).",
    )
    p.add_argument(
        "--optuna-min-val-trades",
        type=int,
        default=5,
        metavar="N",
        help="Minimum validation trades before Optuna stops penalising sparse candidates (default: 5).",
    )
    p.add_argument(
        "--optuna-storage-dir",
        default=str(_OPTUNA_STORAGE_DIR),
        metavar="PATH",
        help="Directory for per-symbol/timeframe Optuna SQLite studies (default: data/optuna).",
    )
    p.add_argument("--dry-run", action="store_true", help="Discover jobs and print plan without training")
    # Universe / paper-trading promotion
    p.add_argument(
        "--auto-promote",
        action="store_true",
        default=False,
        help=(
            "After training, promote any symbol whose best ZOmega meets "
            "--paper-threshold to PAPER stage in data/universe.json. "
            "Run `python3 run_universe.py` to launch the paper bots."
        ),
    )
    p.add_argument(
        "--paper-threshold",
        type=float,
        default=1.0,
        metavar="ZO",
        help="Minimum ZOmega score required for --auto-promote (default: 1.0)",
    )
    p.add_argument(
        "--retrain-rounds",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of warm-start retrain cycles for rejected or below-threshold jobs. "
            "After the first round, any job rejected by the acceptance guard or whose "
            "ZOmega < paper-threshold is re-run with warm_start=True until accepted or N rounds are exhausted "
            "(default: 1 = no auto-retrain)."
        ),
    )
    p.add_argument(
        "--retrain-negative-only",
        action="store_true",
        default=False,
        help=(
            "After round 1, retrain only jobs with negative ZOmega (z_omega < 0). "
            "When enabled, this overrides threshold-based retry selection."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    return p


def _build_variant_specs(
    variants: list,
    base_seed: int | None,
    job: Job,
    accept_if_better: bool,
) -> list[dict]:
    """Build serialisable per-job variant spec dicts for _run_job_tournament."""
    specs = []
    tournament_seed = int(base_seed) if base_seed is not None else 8675309
    for v in variants:
        cid = f"offline_candidate_{v.name}"
        token = f"{cid}:{job.symbol}:{job.timeframe_minutes}:{tournament_seed}".encode()
        seed = (tournament_seed + zlib.crc32(token)) % 2_147_483_647
        specs.append({
            "n_epochs": v.n_epochs,
            "train_every": v.train_every,
            "warm_start": v.warm_start,
            "epsilon_start": v.epsilon_start,
            "epsilon_end": v.epsilon_end,
            "penalty_scale": v.penalty_scale,
            "focused_cap_passes": v.focused_cap_passes,
            "candidate_id": cid,
            "candidate_seed": seed,
            "deploy_candidate": False,  # tournament worker defers; main() deploys winner
        })
    return specs


def _execute_pool(  # noqa: PLR0912, PLR0915
    jobs: list,
    n_workers: int,
    args,
    warm_start: bool,
    ot_status: dict,
    t_start: float,
    best_per_job: dict | None = None,
    variants: list | None = None,
) -> list[dict]:
    """Run a pool of training jobs and return the list of result dicts.
    Updates ot_status in place so the HUD reflects live progress.
    If best_per_job is supplied and args.auto_promote is set, promotes
    winners to universe.json immediately as each job finishes.

    When *variants* is provided (tournament mode) each job runs all its variants
    internally via _run_job_tournament — no inter-job barrier between variant
    rounds.  When None, each job runs a single _run_job call (non-tournament /
    retrain path).
    """
    round_results: list[dict] = []
    ctx = multiprocessing.get_context("spawn")
    candidate_id = str(getattr(args, "candidate_id", "offline_candidate") or "offline_candidate")
    base_seed = getattr(args, "candidate_seed", None)
    deploy_candidate = not bool(getattr(args, "defer_candidate_deploy", False))
    optuna_trials = max(0, int(getattr(args, "optuna_trials", 0) or 0))

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        if optuna_trials > 0:
            timeout_raw = int(getattr(args, "optuna_timeout_secs", 0) or 0)
            search_config = OptunaSearchConfig(
                trials=optuna_trials,
                timeout_s=timeout_raw if timeout_raw > 0 else None,
                min_val_trades=max(0, int(getattr(args, "optuna_min_val_trades", 5) or 0)),
                storage_dir=str(getattr(args, "optuna_storage_dir", _OPTUNA_STORAGE_DIR)),
            )
            base_args = {
                "n_epochs": args.n_epochs,
                "train_every": args.train_every,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
                "penalty_scale": args.penalty_scale,
                "focused_cap_passes": args.focused_cap_passes,
                "tournament_seed": getattr(args, "tournament_seed", 8675309),
            }
            futures = {
                pool.submit(
                    _run_job_optuna,
                    j.symbol,
                    j.timeframe_minutes,
                    str(j.bars_file),
                    j.file_format,
                    args.checkpoint_dir,
                    args.train_split,
                    _job_max_bars(args, j),
                    args.accept_if_better,
                    args.acceptance_margin,
                    [str(path) for path in _job_source_files(j)],
                    args.focused_cap_replay,
                    args.focused_cap_per_side,
                    args.focused_cap_lookback_days,
                    _load_symbol_digits(j.symbol),
                    base_args,
                    search_config,
                ): j
                for j in jobs
            }
            candidate_id = "optuna"
        elif variants is not None:
            # Tournament mode: one future per job, all variants run inside the worker.
            futures = {
                pool.submit(
                    _run_job_tournament,
                    j.symbol,
                    j.timeframe_minutes,
                    str(j.bars_file),
                    j.file_format,
                    args.checkpoint_dir,
                    args.train_split,
                    _job_max_bars(args, j),
                    args.accept_if_better,
                    args.acceptance_margin,
                    [str(path) for path in _job_source_files(j)],
                    args.focused_cap_replay,
                    args.focused_cap_per_side,
                    args.focused_cap_lookback_days,
                    _load_symbol_digits(j.symbol),
                    _build_variant_specs(variants, base_seed, j, args.accept_if_better),
                ): j
                for j in jobs
            }
        else:
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
                    _job_max_bars(args, j),
                    args.n_epochs,
                    warm_start,
                    args.epsilon_start,
                    args.epsilon_end,
                    _load_symbol_digits(j.symbol),
                    args.penalty_scale,
                    args.accept_if_better,
                    args.acceptance_margin,
                    [str(path) for path in _job_source_files(j)],
                    args.focused_cap_replay,
                    args.focused_cap_per_side,
                    args.focused_cap_lookback_days,
                    args.focused_cap_passes,
                    candidate_id,
                    _candidate_seed(base_seed, candidate_id, j),
                    deploy_candidate,
                ): j
                for j in jobs
            }

        # Mark only the first n_workers as running; the rest stay queued.
        # As each job finishes we promote the next queued entry below.
        submitted_list = list(jobs)
        running_keys = set()
        for j in submitted_list[:n_workers]:
            running_keys.add((j.symbol, j.timeframe_minutes))
        for entry in ot_status["results"]:
            key = (entry["symbol"], entry["timeframe_minutes"])
            if key in running_keys:
                entry["status"] = "running"
                entry["candidate_id"] = candidate_id
        _write_status(ot_status)

        pending = dict(futures)
        try:
            while pending:
                try:
                    fut = next(as_completed(pending, timeout=30.0))
                except FuturesTimeoutError:
                    _refresh_running_status(ot_status, jobs, t_start)
                    continue
                job = pending.pop(fut)
                label = f"{job.symbol}_M{job.timeframe_minutes}"
                try:
                    res = fut.result()
                    round_results.append(res)
                    if res.get("error"):
                        LOG.error("[MAIN] %s failed: %s", label, res["error"])
                    else:
                        LOG.info(
                            "[MAIN] %s done — ZOmega=%.4f  trades=%d",
                            label,
                            res["z_omega"],
                            res["val_trades"],
                        )
                except Exception as exc:
                    LOG.error("[MAIN] %s raised: %s", label, exc, exc_info=True)
                    res = {
                        "symbol": job.symbol,
                        "timeframe_minutes": job.timeframe_minutes,
                        "z_omega": 0.0,
                        "train_trades": 0,
                        "val_trades": 0,
                        "total_train_steps": 0,
                        "elapsed_s": 0.0,
                        "weights_path": "",
                        "error": str(exc),
                        "candidate_id": candidate_id,
                        "candidate_seed": _candidate_seed(base_seed, candidate_id, job),
                    }
                    round_results.append(res)

                ot_status["elapsed_s"] = time.perf_counter() - t_start
                for entry in ot_status["results"]:
                    if entry["symbol"] == res["symbol"] and entry["timeframe_minutes"] == res["timeframe_minutes"]:
                        entry["status"] = "error" if res.get("error") else "done"
                        entry["candidate_id"] = res.get("candidate_id", candidate_id)
                        entry["candidate_seed"] = res.get("candidate_seed")
                        entry["z_omega"] = res.get("z_omega", 0.0)
                        entry["optuna_objective"] = res.get("optuna_objective")
                        entry["optuna_trial_number"] = res.get("optuna_trial_number")
                        entry["optuna_storage"] = res.get("optuna_storage", "")
                        entry["val_net_pnl"] = res.get("val_net_pnl", 0.0)
                        entry["val_avg_pnl"] = res.get("val_avg_pnl", 0.0)
                        entry["val_profit_factor"] = res.get("val_profit_factor", 0.0)
                        entry["train_trades"] = res.get("train_trades", 0)
                        entry["val_trades"] = res.get("val_trades", 0)
                        entry["incumbent_z_omega"] = res.get("incumbent_z_omega", 0.0)
                        entry["champion_z_omega"] = res.get("champion_z_omega")
                        entry["champion_source"] = res.get("champion_source", "")
                        entry["acceptance_guard_z_omega"] = res.get("acceptance_guard_z_omega", 0.0)
                        entry["acceptance_guard_source"] = res.get("acceptance_guard_source", "")
                        entry["accepted"] = res.get("accepted", False)
                        entry["accept_reason"] = res.get("accept_reason", "")
                        entry["total_train_steps"] = res.get("total_train_steps", 0)
                        entry["elapsed_s"] = res.get("elapsed_s", 0.0)
                        entry["error"] = res.get("error")
                        break
                # Promote the first queued job to running now that a slot freed up
                for entry in ot_status["results"]:
                    if entry.get("status") == "queued":
                        entry["status"] = "running"
                        entry["candidate_id"] = candidate_id
                        break
                _write_status(ot_status)

                # ── Immediate promotion: don't wait for all jobs to finish ──
                if best_per_job is not None and not res.get("error"):
                    key = (res["symbol"], res["timeframe_minutes"])
                    zo = res.get("z_omega", 0.0)
                    prev = best_per_job.get(key)
                    prev_best = prev.get("z_omega", -1.0) if prev else -1.0
                    if _prefer_training_result(res, prev):
                        best_per_job[key] = res
                    if getattr(args, "defer_candidate_deploy", False):
                        continue
                    _record_offline_champion(args.checkpoint_dir, res)
                    # Promote accepted per-symbol/per-timeframe winners as soon
                    # as each job finishes so parallel timeframes do not mask
                    # one another behind a single symbol-level best result.
                    if (
                        args.auto_promote
                        and bool(res.get("accepted", True))
                        and zo >= args.paper_threshold
                        and zo > prev_best
                    ):
                        sym = res["symbol"]
                        _register_universe(
                            symbol=sym,
                            timeframe_minutes=res["timeframe_minutes"],
                            z_omega=zo,
                            weights_path=res.get("accepted_weights_path") or res.get("weights_path", ""),
                        )
                        LOG.info(
                            "[PROMOTE] %s %s ZΩ=%.4f → promoted to PAPER immediately",
                            sym,
                            _tf_label(res["timeframe_minutes"]),
                            zo,
                        )

        except Exception as pool_exc:
            # BrokenProcessPool or other pool-level failure — mark remaining jobs as failed
            LOG.error("[MAIN] Process pool crashed: %s", pool_exc, exc_info=True)
            completed_syms = {(r["symbol"], r["timeframe_minutes"]) for r in round_results}
            for j in jobs:
                if (j.symbol, j.timeframe_minutes) not in completed_syms:
                    err_res = {
                        "symbol": j.symbol,
                        "timeframe_minutes": j.timeframe_minutes,
                        "z_omega": 0.0,
                        "train_trades": 0,
                        "val_trades": 0,
                        "total_train_steps": 0,
                        "elapsed_s": 0.0,
                        "weights_path": "",
                        "error": f"Pool crash: {pool_exc}",
                        "candidate_id": candidate_id,
                        "candidate_seed": _candidate_seed(base_seed, candidate_id, j),
                    }
                    round_results.append(err_res)
                    for entry in ot_status["results"]:
                        if entry["symbol"] == j.symbol and entry["timeframe_minutes"] == j.timeframe_minutes:
                            entry["status"] = "error"
                            entry["error"] = err_res["error"]
                            break
            ot_status["elapsed_s"] = time.perf_counter() - t_start
            _write_status(ot_status)

    return round_results


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912, PLR0915
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Ensure log dir exists before setting up file handler
    Path("log").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/train_offline.log", mode="a"),
        ],
    )

    # Discover jobs
    jobs = discover_jobs(args.inputs, args.symbols, args.timeframes)
    if not jobs:
        LOG.error("No jobs found. Check file paths and --symbols/--timeframes filters.")
        return 1

    # Filter to known active symbols only (XAUUSD, BTCUSD).
    # This prevents orphan EURUSD / other test symbols from entering the pipeline.
    _KNOWN_SYMBOLS = {"XAUUSD", "BTCUSD"}
    jobs = [j for j in jobs if j.symbol in _KNOWN_SYMBOLS]
    if not jobs:
        LOG.error("All jobs filtered out by symbol whitelist (XAUUSD, BTCUSD).")
        return 1

    LOG.info("Discovered %d job(s):", len(jobs))
    for j in jobs:
        sources = _job_source_files(j)
        suffix = f" (+{len(sources) - 1} merged)" if len(sources) > 1 else ""
        LOG.info("  %s %s  ← %s%s", j.symbol, _tf_label(j.timeframe_minutes), sources[0], suffix)

    # Sort jobs by reverse timeframe (highest first, lowest last).
    # M240 runs first (fewest bars → finishes fastest) so the worker pool
    # is never blocked on a long M1 job while fast TFs sit idle.
    jobs.sort(key=lambda j: -j.timeframe_minutes)
    LOG.info("Jobs sorted by reverse timeframe (highest TF first):")
    for j in jobs:
        LOG.info("  %s %s (%d min)", j.symbol, _tf_label(j.timeframe_minutes), j.timeframe_minutes)

    # ── Pre-flight data integrity check ──────────────────────────────────────
    jobs, pf_errors = preflight_check(jobs)
    if pf_errors:
        for msg in pf_errors:
            LOG.error(msg)
        if not jobs:
            LOG.error("All jobs failed pre-flight — nothing to train.")
            return 1
        LOG.warning(
            "%d job(s) dropped due to data errors; continuing with %d valid job(s).",
            len(pf_errors),
            len(jobs),
        )

    if args.dry_run:
        print("\n[DRY RUN] — no training executed.")
        return 0

    # On GPU systems, limit to 1 worker to avoid CUDA OOM from multiple
    # processes allocating tensors on the same device simultaneously.
    try:
        import torch  # noqa: PLC0415
        _gpu_available = torch.cuda.is_available()
    except Exception:
        _gpu_available = False
    all_jobs = list(jobs)
    resume_entries = _completed_resume_entries(_load_offline_status())
    if resume_entries:
        resume_keys = {
            key
            for key in resume_entries
            if any(j.symbol == key[0] and j.timeframe_minutes == key[1] for j in all_jobs)
        }
        if resume_keys:
            jobs = [j for j in jobs if (j.symbol, j.timeframe_minutes) not in resume_keys]
            LOG.info(
                "[RESUME] Skipping %d already-completed offline job(s) from %s",
                len(resume_keys),
                _STATUS_PATH,
            )
    if not jobs:
        LOG.info("[RESUME] All discovered offline jobs were already complete.")
        _ot_status = {
            "status": "complete",
            "started_at": datetime.now(UTC).isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "total_jobs": len(all_jobs),
            "elapsed_s": 0.0,
            "supervisor": _offline_supervisor_payload(raw_argv),
            "results": [
                resume_entries.get(
                    (j.symbol, j.timeframe_minutes),
                    {
                        "symbol": j.symbol,
                        "timeframe_minutes": j.timeframe_minutes,
                        "label": _tf_label(j.timeframe_minutes),
                        "status": "done",
                    },
                )
                for j in all_jobs
            ],
        }
        _write_status(_ot_status)
        return 0

    allow_gpu_parallel = bool(args.allow_gpu_parallel) or str(
        os.environ.get("CTRADER_OFFLINE_ALLOW_GPU_PARALLEL", ""),
    ).strip().lower() in {"1", "true", "yes", "on"}
    if _gpu_available and (args.workers is None or args.workers > 1) and not allow_gpu_parallel:
        LOG.info("[SPAWN] GPU detected — forcing 1 worker to prevent CUDA contention")
        n_workers = 1
    else:
        n_workers = args.workers or min(len(jobs), multiprocessing.cpu_count())
        if _gpu_available and n_workers > 1:
            LOG.warning(
                "[SPAWN] GPU parallelism explicitly enabled — running %d worker(s); monitor VRAM/OOM",
                n_workers,
            )
    LOG.info("Launching %d worker(s) for %d job(s)", n_workers, len(jobs))

    results: list[dict] = []
    t_start = time.perf_counter()

    # ── Write initial status for HUD ─────────────────────────────────────────
    _ot_status: dict = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "total_jobs": len(all_jobs),
        "elapsed_s": 0.0,
        "supervisor": _offline_supervisor_payload(raw_argv),
        "results": [
            resume_entries.get(
                (j.symbol, j.timeframe_minutes),
                {
                    "symbol": j.symbol,
                    "timeframe_minutes": j.timeframe_minutes,
                    "label": _tf_label(j.timeframe_minutes),
                    "status": "queued",
                },
            )
            for j in all_jobs
        ],
    }
    _write_status(_ot_status)

    training_variants = _build_training_variants(args)
    optuna_mode = max(0, int(getattr(args, "optuna_trials", 0) or 0)) > 0
    tournament_mode = len(training_variants) > 1 and not optuna_mode
    search_mode = tournament_mode or optuna_mode
    if optuna_mode:
        timeout_raw = int(getattr(args, "optuna_timeout_secs", 0) or 0)
        LOG.info(
            "[OPTUNA] Running %d trial(s) per job; timeout=%s; storage=%s",
            args.optuna_trials,
            f"{timeout_raw}s" if timeout_raw > 0 else "none",
            args.optuna_storage_dir,
        )
    if tournament_mode:
        LOG.info("[TOURNAMENT] Running %d candidate recipes per job", len(training_variants))
        for variant in training_variants:
            LOG.info(
                "[TOURNAMENT] %s: epochs=%d train_every=%d eps=%.3f→%.3f penalty=%.3f focused_passes=%d warm_start=%s",
                variant.name,
                variant.n_epochs,
                variant.train_every,
                variant.epsilon_start,
                variant.epsilon_end,
                variant.penalty_scale,
                variant.focused_cap_passes,
                variant.warm_start,
            )

    # ── Round 0 / tournament variants ─────────────────────────────────────────
    best_per_job: dict[tuple, dict] = {}
    if search_mode:
        # Each job runs all its variants internally (no per-variant pool barrier).
        # With --workers N, N jobs process their full variant sequences in parallel.
        _ot_status["status"] = "running"
        _write_status(_ot_status)
        round_results = _execute_pool(
            jobs,
            n_workers,
            args,
            warm_start=False,  # warm_start is per-variant inside _run_job_tournament
            ot_status=_ot_status,
            t_start=t_start,
            best_per_job=best_per_job,
            variants=training_variants if tournament_mode else None,
        )
        results.extend(round_results)
    else:
        variant_args = _variant_args(args, training_variants[0], defer_candidate_deploy=False)
        round_results = _execute_pool(
            jobs,
            n_workers,
            variant_args,
            warm_start=training_variants[0].warm_start,
            ot_status=_ot_status,
            t_start=t_start,
            best_per_job=best_per_job,
        )
        results.extend(round_results)

    if not search_mode:
        # ── Auto-retrain rounds: warm-start re-run for below-threshold jobs ───
        # Keep best result per (symbol, timeframe) across all rounds.
        for r in results:
            key = (r["symbol"], r["timeframe_minutes"])
            if not r.get("error") and _prefer_training_result(r, best_per_job.get(key)):
                best_per_job[key] = r
                _record_offline_champion(args.checkpoint_dir, r)

    for retrain_round in range(1, args.retrain_rounds if not search_mode else 1):
        retry_jobs = [
            j
            for j in jobs
            if _retrain_eligible(
                best_per_job.get((j.symbol, j.timeframe_minutes)),
                threshold=args.paper_threshold,
                negative_only=args.retrain_negative_only,
            )
        ]
        if not retry_jobs:
            if args.retrain_negative_only:
                LOG.info(
                    "[RETRAIN] No jobs with negative ZΩ after round %d — stopping early.",
                    retrain_round,
                )
            else:
                LOG.info(
                    "[RETRAIN] All jobs accepted and met threshold after round %d — stopping early.",
                    retrain_round,
                )
            break
        if args.retrain_negative_only:
            LOG.info(
                "[RETRAIN] Round %d/%d: %d job(s) with negative ZΩ — re-training with warm_start=True",
                retrain_round + 1,
                args.retrain_rounds,
                len(retry_jobs),
            )
        else:
            LOG.info(
                "[RETRAIN] Round %d/%d: %d rejected/below-threshold job(s) — "
                "re-training with warm_start=True (threshold ZΩ=%.2f)",
                retrain_round + 1,
                args.retrain_rounds,
                len(retry_jobs),
                args.paper_threshold,
            )
        # Reset HUD entries for jobs being retrained so they show "running" again
        retry_keys = {(j.symbol, j.timeframe_minutes) for j in retry_jobs}
        for entry in _ot_status["results"]:
            if (entry["symbol"], entry["timeframe_minutes"]) in retry_keys:
                entry["status"] = "queued"
                entry.pop("z_omega", None)
        _ot_status["status"] = "running"
        _write_status(_ot_status)

        retrain_args = _variant_args(args, training_variants[0], defer_candidate_deploy=False)
        round_results = _execute_pool(
            retry_jobs,
            min(n_workers, len(retry_jobs)),
            retrain_args,
            warm_start=True,  # always warm-start on retrain rounds
            ot_status=_ot_status,
            t_start=t_start,
            best_per_job=best_per_job,
        )
        results.extend(round_results)

        for r in round_results:
            key = (r["symbol"], r["timeframe_minutes"])
            if not r.get("error") and _prefer_training_result(r, best_per_job.get(key)):
                best_per_job[key] = r
                _record_offline_champion(args.checkpoint_dir, r)

    total_s = time.perf_counter() - t_start

    # ── Write final status ────────────────────────────────────────────────────
    _ot_status["status"] = "complete"
    _ot_status["completed_at"] = datetime.now(UTC).isoformat()
    _ot_status["elapsed_s"] = total_s
    _write_status(_ot_status)

    # Print summary table — deduplicate to show best result per (symbol, TF)
    deduped = list(best_per_job.values()) if best_per_job else results
    # Include any errored jobs not in best_per_job
    errored = [r for r in results if r.get("error") and (r["symbol"], r["timeframe_minutes"]) not in best_per_job]
    print_summary(deduped + errored)
    LOG.info("All jobs completed in %.1f s", total_s)

    # Copy best-by-ZOmega weights per symbol/timeframe.
    best = select_best_per_bot(deduped + errored)
    if search_mode and best:
        for result in best.values():
            _deploy_candidate_result(result, args.checkpoint_dir)
        best = select_best_per_bot(deduped + errored)
        for result in best.values():
            _record_offline_champion(args.checkpoint_dir, result)
    if best:
        best_dir = Path(args.checkpoint_dir) / "best"
        copy_best_weights(best, best_dir)
        LOG.info("Best weights written to %s/", best_dir)
        print("\nBest weights by ZOmega:")
        for (_sym, _tf), r in sorted(best.items()):
            zo = r.get("z_omega", 0.0)
            zo_str = f"{zo:.4f}" if zo != float("inf") else "+inf"
            label = _tf_label(r["timeframe_minutes"])
            print(f"  {r['symbol']:<12} {label:>5}  ZOmega={zo_str}")

        promoted: list[str] = []
        if args.auto_promote:
            for r in deduped:
                if r.get("error") or not bool(r.get("accepted", True)):
                    continue
                zo = r.get("z_omega", 0.0)
                if zo < args.paper_threshold:
                    continue
                sym = r["symbol"]
                label = _tf_label(r["timeframe_minutes"])
                _register_universe(
                    symbol=sym,
                    timeframe_minutes=r["timeframe_minutes"],
                    z_omega=zo,
                    weights_path=r.get("accepted_weights_path") or r.get("weights_path", ""),
                )
                promoted.append(f"{sym} {label}")

        if promoted:
            print(f"\n[UNIVERSE] {len(promoted)} bot(s) promoted to PAPER stage: " + ", ".join(promoted))
            print("  Launch paper bots with:  python3 run_universe.py --watch")

    n_ok = sum(1 for r in deduped + errored if not r.get("error"))
    n_err = len(deduped + errored) - n_ok
    if n_err:
        LOG.warning("%d job(s) failed.", n_err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
