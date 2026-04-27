#!/usr/bin/env bash
# BTCUSD offline training — thin wrapper around weekend_offline_training.sh.
#
# Configures the weekend script for an on-demand BTCUSD run:
#   - bypasses the market-closed window check
#   - downloads/refreshes historical CSVs before training
#   - uses more epochs and a lower promotion threshold (initial training)
#
# Override any default with environment variables before calling:
#   WORKERS=6 FROM_DATE=2023-01-01 bash scripts/train_btcusd_offline.sh
#   SKIP_DOWNLOAD=1 bash scripts/train_btcusd_offline.sh   # reuse existing CSVs

export WEEKEND_TRAIN_FORCE=1
export WEEKEND_TRAIN_SYMBOLS="BTCUSD"
export WEEKEND_TRAIN_DOWNLOAD_HISTORY="${SKIP_DOWNLOAD:+0}"
export WEEKEND_TRAIN_DOWNLOAD_HISTORY="${WEEKEND_TRAIN_DOWNLOAD_HISTORY:-1}"
export WEEKEND_TRAIN_WORKERS="${WORKERS:-4}"
export WEEKEND_TRAIN_EPOCHS="${EPOCHS:-5}"
export WEEKEND_TRAIN_PAPER_THRESHOLD="${PAPER_THRESHOLD:-0.5}"
export WEEKEND_TRAIN_RETRAIN_ROUNDS="${RETRAIN_ROUNDS:-8}"
export WEEKEND_TRAIN_LOG="logs/train_btcusd_offline.log"
export FROM_DATE="${FROM_DATE:-2024-01-01}"
export TO_DATE="${TO_DATE:-$(date '+%Y-%m-%d')}"

exec "$(dirname "$0")/weekend_offline_training.sh" "$@"
