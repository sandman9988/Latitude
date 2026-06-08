#!/usr/bin/env bash
# Bootstrap initial offline training from 2 years of historical CSV data.
#
# Run this ONCE before starting the paper bot for the first time (or to
# re-seed weights from scratch).  Subsequent weekly retraining is handled
# automatically by scripts/weekend_offline_training.sh.
#
# What this script does:
#   1. Sources ROCm env (GPU auto-detected by ddqn_network.py via global DEVICE)
#   2. Downloads 2 years of OHLCV history per symbol / timeframe via cTrader API
#   3. Runs train_offline.py with --workers 1 (safe for 8 GB VRAM)
#   4. Auto-promotes winning weights into data/universe.json
#
# GPU notes:
#   - ddqn_network._select_device() picks cuda (ROCm) automatically
#   - --workers 1 runs one (symbol, TF) job at a time — each job owns the full
#     GPU.  --workers 2 is possible but risks OOM mid-training on 8 GB VRAM.
#   - M1 jobs are the longest (most bars); M240 are the shortest.
#   - Total wall time at --workers 1 / 8 tournament variants:
#       ~4-8 hours depending on bar count and epoch count.
#
# Rolling training after bootstrap:
#   The paper bot writes data/training_cache_{SYM}_M{TF}.jsonl continuously.
#   weekend_offline_training.sh picks those up every Friday night and runs
#   --warm-start so weights improve incrementally without starting from scratch.
#   Once ZOmega > 1.0 consistently, drop --n-epochs on the weekly run for speed.
#
# Environment variables (override defaults):
#   BOOTSTRAP_FROM          – start date for history download (default: 2 years ago)
#   BOOTSTRAP_TO            – end date (default: today)
#   BOOTSTRAP_SYMBOLS       – space-separated symbols (default: "XAUUSD BTCUSD")
#   BOOTSTRAP_TIMEFRAMES    – space-separated TF minutes (default: "1 5 15 30 60 240")
#   BOOTSTRAP_DEMO          – set to 1 to use demo API endpoint (default: 0 = live)
#   BOOTSTRAP_WORKERS       – train_offline.py --workers (default: 1)
#   BOOTSTRAP_EPOCHS        – --n-epochs per job (default: 10)
#   BOOTSTRAP_TOURNAMENT    – --tournament-variants (default: 8)
#   BOOTSTRAP_PAPER_THRESH  – --paper-threshold ZOmega (default: 1.0)
#   BOOTSTRAP_MAX_BARS      – --max-bars cap (default: unlimited)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

LOG_FILE="${BOOTSTRAP_LOG:-logs/bootstrap_offline_training.log}"
mkdir -p logs data/history

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" | tee -a "$LOG_FILE"
}

# ── Guard: don't run while the live bot is active ────────────────────────────
if pgrep -f "run_universe.py --watch" >/dev/null 2>&1; then
    log "ERROR: run_universe.py --watch is running — stop the bot before bootstrapping."
    exit 1
fi

if pgrep -f "train_offline.py" >/dev/null 2>&1; then
    log "ERROR: train_offline.py is already running."
    exit 1
fi

# ── Environment ──────────────────────────────────────────────────────────────
if [[ -f config/rocm_env.sh ]]; then
    # Source ROCm env — sets HSA_OVERRIDE_GFX_VERSION, PYTORCH_HIP_MEMORY_POOL_SIZE, etc.
    # Child processes from ProcessPoolExecutor(mp_context="spawn") inherit these.
    set +u
    source config/rocm_env.sh
    set -u
    log "ROCm environment loaded (GFX=${HSA_OVERRIDE_GFX_VERSION:-default})"
fi

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Load OAuth credentials from local .env.openapi
LOCAL_ENV=".env.openapi"
if [[ -f "$LOCAL_ENV" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$LOCAL_ENV"
    set +a
    log "Credentials loaded from $LOCAL_ENV"
else
    log "WARNING: No .env.openapi found — download step may fail if tokens not set in environment"
fi

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ── Parameters ───────────────────────────────────────────────────────────────
TODAY="$(date -u '+%Y-%m-%d')"
TWO_YEARS_AGO="$(date -u -d '2 years ago' '+%Y-%m-%d' 2>/dev/null || date -u -v-2y '+%Y-%m-%d')"

FROM_DATE="${BOOTSTRAP_FROM:-$TWO_YEARS_AGO}"
TO_DATE="${BOOTSTRAP_TO:-$TODAY}"
SYMBOLS="${BOOTSTRAP_SYMBOLS:-XAUUSD BTCUSD}"
TIMEFRAMES="${BOOTSTRAP_TIMEFRAMES:-1 5 15 30 60 240}"
DEMO_FLAG=""
[[ "${BOOTSTRAP_DEMO:-0}" == "1" ]] && DEMO_FLAG="--demo"
WORKERS="${BOOTSTRAP_WORKERS:-1}"
EPOCHS="${BOOTSTRAP_EPOCHS:-10}"
TOURNAMENT="${BOOTSTRAP_TOURNAMENT:-8}"
PAPER_THRESH="${BOOTSTRAP_PAPER_THRESH:-1.0}"

log "Bootstrap parameters:"
log "  Date range   : $FROM_DATE → $TO_DATE"
log "  Symbols      : $SYMBOLS"
log "  Timeframes   : $TIMEFRAMES"
log "  Demo mode    : ${BOOTSTRAP_DEMO:-0}"
log "  Workers      : $WORKERS (one GPU job at a time)"
log "  Epochs/job   : $EPOCHS"
log "  Tournament   : $TOURNAMENT variants"
log "  ZΩ threshold : $PAPER_THRESH"

# ── Step 1: Download history ──────────────────────────────────────────────────
log "=== STEP 1: Downloading 2-year historical CSVs ==="

# Build symbol and timeframe arrays for the download command
read -ra SYM_ARRAY <<< "$SYMBOLS"
read -ra TF_ARRAY  <<< "$TIMEFRAMES"

DOWNLOAD_CMD=(
    python3 scripts/download_ctrader_history.py
    --symbol "${SYM_ARRAY[@]}"
    --timeframe "${TF_ARRAY[@]}"
    --from "$FROM_DATE"
    --to   "$TO_DATE"
)
[[ -n "$DEMO_FLAG" ]] && DOWNLOAD_CMD+=("$DEMO_FLAG")

log "Download command: ${DOWNLOAD_CMD[*]}"
set +e
"${DOWNLOAD_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
DL_STATUS=${PIPESTATUS[0]}
set -e

if [[ $DL_STATUS -ne 0 ]]; then
    log "ERROR: History download failed (exit $DL_STATUS) — aborting."
    exit "$DL_STATUS"
fi

# Verify at least some CSV files exist
CSV_COUNT="$(find data/history -name '*.csv' 2>/dev/null | wc -l)"
if [[ "$CSV_COUNT" -eq 0 ]]; then
    log "ERROR: No CSV files found in data/history/ after download — aborting."
    exit 1
fi
log "Downloaded $CSV_COUNT CSV file(s) into data/history/"

# ── Step 2: Offline training ──────────────────────────────────────────────────
log "=== STEP 2: Offline training (GPU, --workers $WORKERS) ==="

mapfile -t CSV_FILES < <(find data/history -name '*.csv' | sort)

TRAIN_CMD=(
    python3 train_offline.py
    "${CSV_FILES[@]}"
    --workers         "$WORKERS"
    --n-epochs        "$EPOCHS"
    --warm-start
    --accept-if-better
    --auto-promote
    --paper-threshold "$PAPER_THRESH"
    --retrain-rounds  6
    --tournament-variants "$TOURNAMENT"
    --tournament-seed 8675309
    --focused-cap-per-side      10
    --focused-cap-lookback-days 30
    --focused-cap-passes        2
)

if [[ -n "${BOOTSTRAP_MAX_BARS:-}" ]]; then
    TRAIN_CMD+=(--max-bars "$BOOTSTRAP_MAX_BARS")
fi

log "Training command: ${TRAIN_CMD[*]}"
log "GPU memory pool : ${PYTORCH_HIP_MEMORY_POOL_SIZE:-unset} MB"
log "GFX version     : ${HSA_OVERRIDE_GFX_VERSION:-unset}"

set +e
"${TRAIN_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

if [[ $TRAIN_STATUS -eq 0 ]]; then
    log "=== Bootstrap training completed successfully ==="
    log ""
    log "Next steps:"
    log "  1. Inspect data/checkpoints/offline_champions.json for accepted weights"
    log "  2. Start the paper bot: bash run.sh (or python3 run_universe.py --watch)"
    log "  3. Rolling retraining runs automatically via scripts/weekend_offline_training.sh"
    log "     Schedule it: bash scripts/setup_weekend_training.sh"
else
    log "WARNING: Training exited with status $TRAIN_STATUS"
    log "Partial results may still be in data/checkpoints/ — check logs above."
fi

exit "$TRAIN_STATUS"
