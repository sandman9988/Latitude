#!/usr/bin/env bash
# Phase 1+2 combined: Optuna (8 trials) → Tournament (6 variants)
# for the 5 untrained symbol/TF pairs.
#
# Usage:  bash scripts/optuna_then_tournament.sh
# Logs:   logs/phase1_optuna_*.log, logs/phase2_tournament_*.log
set -euo pipefail

cd /home/renierdejager/Projects/ctrader_trading_bot
source .venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0

COMMON_ARGS="data/ --workers 1 --n-epochs 3 --warm-start --accept-if-better --auto-promote --paper-threshold 1.0 --focused-cap-per-side 10 --focused-cap-lookback-days 7 --focused-cap-passes 2 --tournament-seed 8675309"

echo "================================================================"
echo " PHASE 1: Optuna hyperparameter search (8 trials each)"
echo " Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

echo ""
echo "▶▶▶ XAUUSD M15 + M30 (2 jobs × 8 trials)"
python3 train_offline.py ${COMMON_ARGS} \
--symbols XAUUSD --timeframes M15 M30 \
--optuna-trials 8 \
> logs/phase1_optuna_xauusu_m15m30.log 2>&1
echo "   ✅ XAUUSD Optuna done — $(date '+%H:%M:%S')"

echo ""
echo "▶▶▶ BTCUSD M1 + M5 + M15 (3 jobs × 8 trials)"
python3 train_offline.py ${COMMON_ARGS} \
--symbols BTCUSD --timeframes M1 M5 M15 \
--optuna-trials 8 \
> logs/phase1_optuna_btcusd_m1m5m15.log 2>&1
echo "   ✅ BTCUSD Optuna done — $(date '+%H:%M:%S')"

echo ""
echo "================================================================"
echo " PHASE 2: Tournament variants (6 variants each)"
echo " Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

echo ""
echo "▶▶▶ XAUUSD M15 + M30 (2 jobs × 6 variants)"
python3 train_offline.py ${COMMON_ARGS} \
--symbols XAUUSD --timeframes M15 M30 \
--tournament-variants 6 \
> logs/phase2_tournament_xauusu_m15m30.log 2>&1
echo "   ✅ XAUUSD Tournament done — $(date '+%H:%M:%S')"

echo ""
echo "▶▶▶ BTCUSD M1 + M5 + M15 (3 jobs × 6 variants)"
python3 train_offline.py ${COMMON_ARGS} \
--symbols BTCUSD --timeframes M1 M5 M15 \
--tournament-variants 6 \
> logs/phase2_tournament_btcusd_m1m5m15.log 2>&1
echo "   ✅ BTCUSD Tournament done — $(date '+%H:%M:%S')"

echo ""
echo "================================================================"
echo " ALL PHASES COMPLETE"
echo " Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
echo ""
echo "Check results:"
echo "  cat data/offline_training_status.json"
echo "  cat data/checkpoints/offline_champions.json"
echo "  cat data/universe.json | python3 -m json.tool"
