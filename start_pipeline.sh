#!/bin/bash
# Offline → Paper Trading pipeline with PER RL
#
# STEP 1: Train DualPolicy offline on historical bars (XAUUSD + BTCUSD by default)
#         PER buffer auto-scales to the training set size.
#         Models that clear --paper-threshold are registered in data/universe.json.
#
# STEP 2: Launch run_universe.py --watch to spin up isolated paper bots for every
#         instrument in universe.json and supervise them.
#
# Usage:
#   ./start_pipeline.sh                          # default symbols + timeframes
#   SYMBOLS="XAUUSD" ./start_pipeline.sh         # single symbol
#   THRESHOLD=1.5 ./start_pipeline.sh            # stricter Z-Omega gate
#   ./start_pipeline.sh --no-watch               # train only, skip paper launch
#
# Override any variable inline:
#   WORKERS=4 TIMEFRAMES="H1 H4" ./start_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ── Configurable defaults ──────────────────────────────────────────────────────
SYMBOLS="${SYMBOLS:-XAUUSD BTCUSD}"
TIMEFRAMES="${TIMEFRAMES:-M15 M30 H1 H4}"
WORKERS="${WORKERS:-$(nproc)}"  # default: all logical CPU cores
THRESHOLD="${THRESHOLD:-1.0}"
HISTORY_DIR="${HISTORY_DIR:-/home/renierdejager/Kinetra/data/master_standardized}"
WATCH=1

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --no-watch)  WATCH=0 ;;
        --help|-h)
            sed -n '/^# Usage/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

echo "============================================================"
echo "  OFFLINE → PAPER TRADING PIPELINE"
echo "============================================================"
echo "  Symbols    : $SYMBOLS"
echo "  Timeframes : $TIMEFRAMES"
echo "  Workers    : $WORKERS"
echo "  Z-Ω gate   : $THRESHOLD"
echo "  History    : $HISTORY_DIR"
echo "  Auto-watch : $([ "$WATCH" -eq 1 ] && echo yes || echo no)"
echo "============================================================"
echo ""

if [ ! -d "$HISTORY_DIR" ]; then
    echo "ERROR: History directory '$HISTORY_DIR' not found."
    echo "       Export bars from cTrader or MT5 into $HISTORY_DIR/ and retry."
    exit 1
fi

# ── STEP 1: Offline training ───────────────────────────────────────────────────
echo "[1/2] Starting offline training..."
echo ""

# Build --symbols flag (space-separated list → multiple args)
SYM_ARGS=""
for s in $SYMBOLS; do SYM_ARGS="$SYM_ARGS --symbols $s"; done
# shellcheck disable=SC2206
SYM_ARGS=($SYM_ARGS)

TF_ARGS=""
for t in $TIMEFRAMES; do TF_ARGS="$TF_ARGS --timeframes $t"; done
# shellcheck disable=SC2206
TF_ARGS=($TF_ARGS)

python3 train_offline.py "$HISTORY_DIR" \
    "${SYM_ARGS[@]}" \
    "${TF_ARGS[@]}" \
    --workers "$WORKERS" \
    --auto-promote \
    --paper-threshold "$THRESHOLD"

echo ""
echo "[1/2] Offline training complete."
echo ""

# ── STEP 2: Universe / paper trading ─────────────────────────────────────────
if [ "$WATCH" -eq 0 ]; then
    echo "Skipping paper trading launch (--no-watch)."
    echo ""
    echo "To launch manually:"
    echo "  python3 run_universe.py --watch"
    echo "  python3 run_universe.py --list"
    exit 0
fi

UNIVERSE="data/universe.json"
if [ ! -f "$UNIVERSE" ]; then
    echo "No instruments cleared the Z-Ω gate (threshold=$THRESHOLD)."
    echo "Try a lower --paper-threshold or provide more history data."
    exit 0
fi

PAPER_COUNT=$(python3 -c "
import json
u = json.load(open('$UNIVERSE'))
print(sum(1 for v in u.values() if v.get('stage') == 'PAPER'))
" 2>/dev/null || echo 0)

if [ "$PAPER_COUNT" -eq 0 ]; then
    echo "universe.json exists but no instruments are in PAPER stage."
    echo "  python3 run_universe.py --list  # inspect current stages"
    exit 0
fi

echo "[2/2] Launching $PAPER_COUNT paper bot(s) under run_universe --watch..."
echo "      Press Ctrl-C to stop the supervisor (paper bots survive in setsid)."
echo ""

python3 run_universe.py --watch
