#!/usr/bin/env bash
# Guarded offline retraining — runs automatically during the weekend market close
# and on-demand via train_btcusd_offline.sh (or any other symbol wrapper).
#
# Market-closed window (UTC):  Friday 22:00 → Sunday 21:30
# The Sunday buffer avoids promoting weights right at market reopen.
#
# Environment variables (all optional):
#
#   WEEKEND_TRAIN_FORCE=1               bypass the market-closed window check
#   WEEKEND_TRAIN_SYMBOLS="BTCUSD"      train only these symbols (space-separated)
#   WEEKEND_TRAIN_DOWNLOAD_HISTORY=0    set to 1 to download/refresh CSVs first
#   FROM_DATE=2024-01-01                history download start date
#   TO_DATE=<today>                     history download end date
#   WEEKEND_TRAIN_WORKERS=2             parallel worker processes
#   WEEKEND_TRAIN_ALLOW_GPU_PARALLEL=1  honor multiple workers on GPU hosts
#   WEEKEND_TRAIN_EPOCHS=3              training epochs per job
#   WEEKEND_TRAIN_PAPER_THRESHOLD=1.0   minimum z_omega to promote
#   WEEKEND_TRAIN_ACCEPTANCE_MARGIN=0.0 extra margin above incumbent z_omega
#   WEEKEND_TRAIN_RETRAIN_ROUNDS=6      retrain attempts when below threshold
#   WEEKEND_TRAIN_TOURNAMENT_VARIANTS=6 hyperparameter variants per job
#   WEEKEND_TRAIN_TOURNAMENT_SEED=8675309
#   WEEKEND_TRAIN_FOCUSED_CAP_PER_SIDE=10
#   WEEKEND_TRAIN_FOCUSED_CAP_LOOKBACK_DAYS=7
#   WEEKEND_TRAIN_FOCUSED_CAP_PASSES=2
#   WEEKEND_TRAIN_MAX_BARS=<unset>      cap bars per job (unset = unlimited)
#   WEEKEND_TRAIN_MAX_BARS_BY_TF=M1=500000 per-timeframe bar caps
#   WEEKEND_TRAIN_REPLACE_EXISTING=1    terminate stale repo-local trainers first
#   WEEKEND_TRAIN_FRESH_STATUS=1        archive old status and train all jobs
#   WEEKEND_TRAIN_INCREMENTAL_SYNC=1    stage accepted weights while jobs finish
#   WEEKEND_TRAIN_INCREMENTAL_SYNC_SECS=300
#   WEEKEND_TRAIN_RESTART_UNIVERSE=1    restart watcher after training
#   WEEKEND_TRAIN_LOG=logs/weekend_offline_training.log
#   WEEKEND_TRAIN_LOCK=data/.weekend_offline_training.lock

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

mkdir -p logs data

LOG_FILE="${WEEKEND_TRAIN_LOG:-logs/weekend_offline_training.log}"
LOCK_FILE="${WEEKEND_TRAIN_LOCK:-data/.weekend_offline_training.lock}"
UNIVERSE_WAS_RUNNING=0

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" | tee -a "$LOG_FILE"
}

# ── Environment loading ───────────────────────────────────────────────────────

load_environment() {
    if [[ -f .env ]]; then
        set -a
        # shellcheck disable=SC1091
        source .env
        set +a
    fi

    # OAuth credentials for history download
    if [[ -f /home/renierdejager/Projects/Kinetra/.env.openapi ]]; then
        set -a
        # shellcheck disable=SC1091
        source /home/renierdejager/Projects/Kinetra/.env.openapi
        set +a
    fi

    # cTrader access token (contains CTRADER_ACCESS_TOKEN required by download script)
    if [[ -f config/cTraderAppTokens ]]; then
        set -a
        # shellcheck disable=SC1091
        source config/cTraderAppTokens
        set +a
    fi

    if [[ -f .venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi

    if [[ -f config/rocm_env.sh ]]; then
        set +u
        # shellcheck disable=SC1091
        source config/rocm_env.sh 2>/dev/null || true
        set -u
    fi
}

# ── Market window check ───────────────────────────────────────────────────────

is_market_closed_window() {
    python3 - <<'PY'
from __future__ import annotations
from datetime import datetime, timezone

now = datetime.now(timezone.utc)
weekday = now.weekday()  # Monday=0 … Sunday=6
minutes = now.hour * 60 + now.minute

closed = (
    (weekday == 4 and minutes >= 22 * 60)
    or weekday == 5
    or (weekday == 6 and minutes < 21 * 60 + 30)
)
print("1" if closed else "0")
PY
}

# ── Data discovery ────────────────────────────────────────────────────────────

discover_training_caches() {
    # When WEEKEND_TRAIN_SYMBOLS is set, restrict to those symbols only.
    if [[ -n "${WEEKEND_TRAIN_SYMBOLS:-}" ]]; then
        for sym in ${WEEKEND_TRAIN_SYMBOLS}; do
            find data -maxdepth 2 -type f \
                \( -name "training_cache_${sym}_M*.jsonl" \
                -o -name "training_cache_${sym}_H*.jsonl" \) 2>/dev/null | sort
        done
    else
        find data -maxdepth 2 -type f \
            \( -name 'training_cache_*_M*.jsonl' \
            -o -name 'training_cache_*_H*.jsonl' \
            -o -name 'training_cache_*_D1.jsonl' \
            -o -name 'training_cache_*_W1.jsonl' \) | sort
    fi
}

discover_history_csvs() {
    # Finds pre-downloaded CSVs in data/history/, filtered by symbol when set.
    [[ -d data/history ]] || return 0
    if [[ -n "${WEEKEND_TRAIN_SYMBOLS:-}" ]]; then
        for sym in ${WEEKEND_TRAIN_SYMBOLS}; do
            find data/history -maxdepth 1 -type f -name "${sym}_M*.csv" 2>/dev/null | sort
        done
    else
        find data/history -maxdepth 1 -type f -name '*.csv' 2>/dev/null | sort
    fi
}

# ── History download (optional) ───────────────────────────────────────────────

download_history() {
    local symbols="${WEEKEND_TRAIN_SYMBOLS:-XAUUSD BTCUSD}"
    local from_date="${FROM_DATE:-2024-01-01}"
    local to_date="${TO_DATE:-$(date '+%Y-%m-%d')}"

    mkdir -p data/history
    log "Downloading history: symbols=[$symbols] from=$from_date to=$to_date"

    # shellcheck disable=SC2086
    python3 scripts/download_ctrader_history.py \
        --symbol $symbols \
        --timeframe 1 5 15 30 60 240 \
        --from "$from_date" \
        --to "$to_date" \
        --demo \
        --output-dir data/history \
        2>&1 | tee -a "$LOG_FILE" \
    || log "History download failed — continuing with existing CSVs (if any)"
}

# ── Process management ────────────────────────────────────────────────────────

managed_paper_bots_running() {
    python3 - <<'PY'
from __future__ import annotations
import os
from pathlib import Path

module = "src.core.ctrader_ddqn_paper"
project_data = (Path.cwd() / "data").resolve()

for proc_dir in Path("/proc").iterdir():
    if not proc_dir.name.isdigit():
        continue
    pid = int(proc_dir.name)
    if pid == os.getpid():
        continue
    try:
        argv = [
            part.decode("utf-8", "ignore")
            for part in (proc_dir / "cmdline").read_bytes().split(b"\0")
            if part
        ]
    except OSError:
        continue
    if module not in argv:
        continue
    try:
        raw_env = (proc_dir / "environ").read_bytes()
    except OSError:
        continue
    env = {}
    for part in raw_env.split(b"\0"):
        if b"=" not in part:
            continue
        key, value = part.split(b"=", 1)
        env[key.decode("utf-8", "ignore")] = value.decode("utf-8", "ignore")
    data_dir = str(env.get("CTRADER_DATA_DIR", "") or "").strip()
    if not data_dir:
        continue
    try:
        cwd = Path(f"/proc/{pid}/cwd").resolve()
        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = cwd / data_path
        data_path = data_path.resolve()
    except OSError:
        continue
    if data_path.name.startswith("paper_") and data_path.is_relative_to(project_data):
        raise SystemExit(0)

raise SystemExit(1)
PY
}

repo_train_offline_pids() {
    python3 - "$PROJECT_ROOT" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()

for proc_dir in Path("/proc").iterdir():
    if not proc_dir.name.isdigit():
        continue
    pid = int(proc_dir.name)
    if pid == os.getpid():
        continue
    try:
        argv = [
            part.decode("utf-8", "ignore")
            for part in (proc_dir / "cmdline").read_bytes().split(b"\0")
            if part
        ]
    except OSError:
        continue
    if not any(Path(part).name == "train_offline.py" or part == "train_offline.py" for part in argv):
        continue
    try:
        cwd = (proc_dir / "cwd").resolve()
    except OSError:
        continue
    if cwd == root:
        print(pid)
PY
}

terminate_repo_trainers() {
    local -a pids=("$@")
    [[ ${#pids[@]} -gt 0 ]] || return 0

    log "Terminating existing repo-local train_offline.py process(es): ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true

    local deadline=$((SECONDS + 45))
    local -a remaining=()
    while (( SECONDS < deadline )); do
        remaining=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                remaining+=("$pid")
            fi
        done
        [[ ${#remaining[@]} -eq 0 ]] && return 0
        sleep 2
    done

    log "Existing trainer(s) did not exit cleanly; forcing: ${remaining[*]}"
    kill -9 "${remaining[@]}" 2>/dev/null || true
}

ensure_no_existing_trainer() {
    local -a pids=()
    mapfile -t pids < <(repo_train_offline_pids)
    [[ ${#pids[@]} -gt 0 ]] || return 0

    if [[ "${WEEKEND_TRAIN_REPLACE_EXISTING:-1}" != "1" ]]; then
        log "Another repo-local train_offline.py is already running (${pids[*]}); skipping."
        return 1
    fi

    terminate_repo_trainers "${pids[@]}"
    mapfile -t pids < <(repo_train_offline_pids)
    if [[ ${#pids[@]} -gt 0 ]]; then
        log "Existing train_offline.py process(es) still running after termination attempt: ${pids[*]}"
        return 1
    fi
}

archive_previous_offline_status() {
    if [[ "${WEEKEND_TRAIN_FRESH_STATUS:-1}" != "1" ]]; then
        export CTRADER_OFFLINE_RESUME_STATUS="${CTRADER_OFFLINE_RESUME_STATUS:-1}"
        return 0
    fi

    export CTRADER_OFFLINE_RESUME_STATUS=0

    local status_file="data/offline_training_status.json"
    [[ -f "$status_file" ]] || return 0

    local stamp
    stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
    local archive="data/offline_training_status.pre_weekend_${stamp}.json"
    mv "$status_file" "$archive"
    log "Archived previous offline status to $archive; starting a fresh weekend queue."
}

sync_accepted_checkpoints_to_runtime() {
    python3 - <<'PY'
from __future__ import annotations
import json
import re
import shutil
import filecmp
from pathlib import Path

root = Path.cwd()
champions_path = root / "data" / "checkpoints" / "offline_champions.json"
if not champions_path.exists():
    raise SystemExit(0)

try:
    payload = json.loads(champions_path.read_text())
except Exception as exc:
    print(f"[WARN] Could not read {champions_path}: {exc}")
    raise SystemExit(0)

champions = payload.get("champions", {})
if not isinstance(champions, dict):
    raise SystemExit(0)

copied = 0
for entry in champions.values():
    if not isinstance(entry, dict):
        continue
    symbol = str(entry.get("symbol", "") or "").upper()
    try:
        timeframe = int(entry.get("timeframe_minutes", 0) or 0)
    except (TypeError, ValueError):
        continue
    if not symbol or timeframe <= 0:
        continue
    weights_path = str(entry.get("weights_path", "") or "")
    if not weights_path:
        continue

    symbol_token = re.sub(r"[^A-Z0-9]+", "_", symbol).strip("_") or "UNKNOWN"
    runtime_dir = root / "data" / f"paper_{symbol_token}_M{timeframe}" / "checkpoints" / f"{symbol_token}_M{timeframe}"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for raw_src in weights_path.split(";"):
        raw_src = raw_src.strip()
        if not raw_src:
            continue
        src = Path(raw_src)
        if not src.is_absolute():
            src = root / src
        if not src.exists():
            print(f"[WARN] Accepted checkpoint missing: {src}")
            continue
        name = src.name.lower()
        if "trigger" in name:
            dst = runtime_dir / f"trigger_ddqn_weights{src.suffix}"
        elif "harvester" in name:
            dst = runtime_dir / f"harvester_ddqn_weights{src.suffix}"
        else:
            dst = runtime_dir / src.name
        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            continue
        shutil.copy2(src, dst)
        copied += 1
        try:
            src_label = src.relative_to(root)
        except ValueError:
            src_label = src
        try:
            dst_label = dst.relative_to(root)
        except ValueError:
            dst_label = dst
        print(f"[SYNC] {src_label} -> {dst_label}")

if copied:
    print(f"[SYNC] Deployed {copied} accepted checkpoint file(s) to runtime bot directories.")
PY
}

runtime_sync_monitor() {
    local train_pid="$1"
    local interval="${WEEKEND_TRAIN_INCREMENTAL_SYNC_SECS:-300}"

    while kill -0 "$train_pid" 2>/dev/null; do
        if ! sync_accepted_checkpoints_to_runtime | tee -a "$LOG_FILE"; then
            log "Incremental checkpoint runtime sync failed; will retry."
        fi
        sleep "$interval"
    done

    if ! sync_accepted_checkpoints_to_runtime | tee -a "$LOG_FILE"; then
        log "Final incremental checkpoint runtime sync failed."
    fi
}

stop_universe_if_running() {
    if pgrep -f "run_universe.py --watch" >/dev/null 2>&1; then
        UNIVERSE_WAS_RUNNING=1
    fi
    if [[ "$UNIVERSE_WAS_RUNNING" != "1" ]] && managed_paper_bots_running; then
        UNIVERSE_WAS_RUNNING=1
        log "Managed paper bots running without universe watcher — stopping before promotion."
    fi

    if [[ "${WEEKEND_TRAIN_RESTART_UNIVERSE:-1}" != "1" || "$UNIVERSE_WAS_RUNNING" != "1" ]]; then
        return 0
    fi

    log "Stopping universe watcher and managed bots before checkpoint promotion."
    python3 run_universe.py --stop-all >> "$LOG_FILE" 2>&1 || true
    pkill -f "run_universe.py --watch" 2>/dev/null || true
    sleep 5
}

start_universe_watcher() {
    # Close the weekend-training flock fd in the child so future runs aren't blocked.
    if command -v setsid >/dev/null 2>&1; then
        setsid bash -c 'exec 9>&-; export UNIVERSE_LAUNCH_STAGGER_SEC="${UNIVERSE_LAUNCH_STAGGER_SEC:-12}"; exec python3 run_universe.py --watch' >> logs/run_universe.log 2>&1 &
    else
        nohup bash -c 'exec 9>&-; export UNIVERSE_LAUNCH_STAGGER_SEC="${UNIVERSE_LAUNCH_STAGGER_SEC:-12}"; exec python3 run_universe.py --watch' >> logs/run_universe.log 2>&1 &
    fi
}

restart_universe_if_needed() {
    if [[ "${WEEKEND_TRAIN_RESTART_UNIVERSE:-1}" != "1" || "$UNIVERSE_WAS_RUNNING" != "1" ]]; then
        return 0
    fi

    if pgrep -f "run_universe.py --watch" >/dev/null 2>&1; then
        log "Universe watcher already running after training."
        return 0
    fi

    log "Restarting universe watcher so accepted checkpoints are loaded."
    start_universe_watcher
    log "Universe watcher restart requested; log: logs/run_universe.log"
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "Offline training already running (lock held); skipping."
        return 0
    fi

    load_environment

    if [[ "${WEEKEND_TRAIN_FORCE:-0}" != "1" && "$(is_market_closed_window)" != "1" ]]; then
        log "Market-closed weekend window is not active; skipping."
        return 0
    fi

    if ! ensure_no_existing_trainer; then
        return 0
    fi

    # ── Optionally refresh historical CSVs ────────────────────────────────────
    if [[ "${WEEKEND_TRAIN_DOWNLOAD_HISTORY:-0}" == "1" ]]; then
        download_history
    fi

    # ── Collect inputs: live caches + historical CSVs ─────────────────────────
    mapfile -t caches < <(discover_training_caches)
    mapfile -t csvs   < <(discover_history_csvs)

    if [[ ${#caches[@]} -eq 0 && ${#csvs[@]} -eq 0 ]]; then
        log "No training inputs found (no caches, no CSVs); skipping."
        return 0
    fi

    log "Inputs: ${#caches[@]} cache(s), ${#csvs[@]} CSV(s)"

    archive_previous_offline_status

    stop_universe_if_running

    # ── Build train_offline.py command ────────────────────────────────────────
    local -a cmd=(
        python3 train_offline.py
        "${csvs[@]}"
        "${caches[@]}"
        --workers "${WEEKEND_TRAIN_WORKERS:-2}"
        --n-epochs "${WEEKEND_TRAIN_EPOCHS:-3}"
        --warm-start
        --accept-if-better
        --auto-promote
        --acceptance-margin "${WEEKEND_TRAIN_ACCEPTANCE_MARGIN:-0.0}"
        --retrain-rounds "${WEEKEND_TRAIN_RETRAIN_ROUNDS:-6}"
        --paper-threshold "${WEEKEND_TRAIN_PAPER_THRESHOLD:-1.0}"
        --focused-cap-per-side "${WEEKEND_TRAIN_FOCUSED_CAP_PER_SIDE:-10}"
        --focused-cap-lookback-days "${WEEKEND_TRAIN_FOCUSED_CAP_LOOKBACK_DAYS:-7}"
        --focused-cap-passes "${WEEKEND_TRAIN_FOCUSED_CAP_PASSES:-2}"
        --tournament-variants "${WEEKEND_TRAIN_TOURNAMENT_VARIANTS:-6}"
        --tournament-seed "${WEEKEND_TRAIN_TOURNAMENT_SEED:-8675309}"
    )

    if [[ "${WEEKEND_TRAIN_ALLOW_GPU_PARALLEL:-1}" == "1" ]]; then
        cmd+=(--allow-gpu-parallel)
    fi

    if [[ -n "${WEEKEND_TRAIN_SYMBOLS:-}" ]]; then
        # shellcheck disable=SC2206
        cmd+=(--symbols ${WEEKEND_TRAIN_SYMBOLS})
    fi

    if [[ -n "${WEEKEND_TRAIN_MAX_BARS:-}" ]]; then
        cmd+=(--max-bars "$WEEKEND_TRAIN_MAX_BARS")
    fi

    if [[ -n "${WEEKEND_TRAIN_MAX_BARS_BY_TF:-M1=500000}" ]]; then
        cmd+=(--max-bars-by-timeframe "${WEEKEND_TRAIN_MAX_BARS_BY_TF:-M1=500000}")
    fi

    log "Starting offline training."
    log "Command: ${cmd[*]}"

    set +e
    "${cmd[@]}" >> "$LOG_FILE" 2>&1 &
    local train_pid=$!
    local sync_pid=""
    if [[ "${WEEKEND_TRAIN_INCREMENTAL_SYNC:-1}" == "1" ]]; then
        runtime_sync_monitor "$train_pid" &
        sync_pid=$!
        log "Incremental accepted-checkpoint sync enabled every ${WEEKEND_TRAIN_INCREMENTAL_SYNC_SECS:-300}s (PID $sync_pid)."
    fi
    wait "$train_pid"
    local status=$?
    if [[ -n "$sync_pid" ]]; then
        kill "$sync_pid" 2>/dev/null || true
        wait "$sync_pid" 2>/dev/null || true
    fi
    set -e

    if [[ $status -eq 0 ]]; then
        log "Offline training completed."
        if ! sync_accepted_checkpoints_to_runtime | tee -a "$LOG_FILE"; then
            log "Checkpoint runtime sync failed; continuing to universe restart."
        fi
    else
        log "Offline training failed with exit status $status."
    fi

    restart_universe_if_needed
    return "$status"
}

main "$@"
