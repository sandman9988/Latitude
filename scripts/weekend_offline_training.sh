#!/usr/bin/env bash
# Run guarded offline retraining during the normal FX/XAU weekend close.
#
# Defaults target the closed window in UTC:
#   Friday 22:00 UTC through Sunday 21:30 UTC.
# The Sunday buffer avoids promoting/restarting bots right at market reopen.
# The training defaults favour champion search over quick retraining: weaker
# candidates remain rejected by train_offline.py's per-symbol/timeframe guard.

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

load_environment() {
    if [[ -f .env ]]; then
        set -a
        # shellcheck disable=SC1091
        source .env
        set +a
    fi

    if [[ -f .venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
}

is_market_closed_window() {
    python3 - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone

now = datetime.now(timezone.utc)
weekday = now.weekday()  # Monday=0 ... Sunday=6
minutes = now.hour * 60 + now.minute

closed = (
    (weekday == 4 and minutes >= 22 * 60)
    or weekday == 5
    or (weekday == 6 and minutes < 21 * 60 + 30)
)
print("1" if closed else "0")
PY
}

discover_training_caches() {
    find data -maxdepth 2 -type f \
        \( -name 'training_cache_*_M*.jsonl' \
        -o -name 'training_cache_*_H*.jsonl' \
        -o -name 'training_cache_*_D1.jsonl' \
        -o -name 'training_cache_*_W1.jsonl' \) | sort
}

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

sync_accepted_checkpoints_to_runtime() {
    python3 - <<'PY'
from __future__ import annotations

import json
import re
import shutil
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

stop_universe_if_running() {
    if pgrep -f "run_universe.py --watch" >/dev/null 2>&1; then
        UNIVERSE_WAS_RUNNING=1
    fi
    if [[ "$UNIVERSE_WAS_RUNNING" != "1" ]] && managed_paper_bots_running; then
        UNIVERSE_WAS_RUNNING=1
        log "Managed paper bots are running without the universe watcher; stopping them before checkpoint promotion."
    fi

    if [[ "${WEEKEND_TRAIN_RESTART_UNIVERSE:-1}" != "1" || "$UNIVERSE_WAS_RUNNING" != "1" ]]; then
        return 0
    fi

    log "Universe watcher or managed paper bots are running; stopping them before checkpoint promotion."
    python3 run_universe.py --stop-all >> "$LOG_FILE" 2>&1 || true
    pkill -f "run_universe.py --watch" 2>/dev/null || true
    sleep 5
}

start_universe_watcher() {
    # Close the weekend-training flock fd in the child. Without this, the
    # long-lived watcher inherits fd 9 and future weekend runs see a held lock.
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

main() {
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "Weekend offline training already running; skipping."
        return 0
    fi

    load_environment

    if [[ "${WEEKEND_TRAIN_FORCE:-0}" != "1" && "$(is_market_closed_window)" != "1" ]]; then
        log "Market-closed weekend window is not active; skipping."
        return 0
    fi

    if pgrep -f "train_offline.py" >/dev/null 2>&1; then
        log "Another offline training process is already running; skipping."
        return 0
    fi

    mapfile -t caches < <(discover_training_caches)
    if [[ ${#caches[@]} -eq 0 ]]; then
        log "No per-bot training caches found under data/training_cache_*_<TF>.jsonl; skipping."
        return 0
    fi

    stop_universe_if_running

    local -a cmd=(
        python3 train_offline.py
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

    if [[ -n "${WEEKEND_TRAIN_MAX_BARS:-}" ]]; then
        cmd+=(--max-bars "$WEEKEND_TRAIN_MAX_BARS")
    fi

    log "Starting guarded weekend offline training for ${#caches[@]} cache(s)."
    log "Command: ${cmd[*]}"

    set +e
    "${cmd[@]}" >> "$LOG_FILE" 2>&1
    local status=$?
    set -e

    if [[ $status -eq 0 ]]; then
        log "Weekend offline training completed."
        if ! sync_accepted_checkpoints_to_runtime | tee -a "$LOG_FILE"; then
            log "Accepted checkpoint runtime sync failed; continuing to universe restart."
        fi
    else
        log "Weekend offline training failed with exit status $status."
    fi

    restart_universe_if_needed
    return "$status"
}

main "$@"
