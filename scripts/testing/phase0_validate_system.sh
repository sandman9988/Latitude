#!/bin/bash
# phase0_validate_system.sh
# Phase 0: System validation ONLY (2-4 hours maximum)
# 
# Quick paper run to verify no crashes before going live.
# Learning DISABLED - we're not trying to train agents here.

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Phase 0: System Validation (Paper Mode)${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}This is a SHORT validation run (2-4 hours max)${NC}"
echo -e "${BLUE}Purpose: Verify system stability before live trading${NC}"
echo ""
echo -e "${GREEN}✓ Paper mode (no real money)${NC}"
echo -e "${GREEN}✓ Learning DISABLED (not training agents)${NC}"
echo -e "${GREEN}✓ All safety systems active${NC}"
echo ""
echo -e "${YELLOW}What we're checking:${NC}"
echo -e "  - No crashes or exceptions${NC}"
echo -e "  - FIX sessions connect properly${NC}"
echo -e "  - Circuit breakers trip correctly${NC}"
echo -e "  - Event time features log correctly${NC}"
echo -e "  - Path geometry calculations work${NC}"
echo -e "  - PER buffer fills (even if not learning)${NC}"
echo ""
echo -e "${RED}⚠ This is NOT for training agents${NC}"
echo -e "${RED}⚠ Do NOT run this for more than 4 hours${NC}"
echo -e "${RED}⚠ After validation passes, move to Phase 1 (live micro)${NC}"
echo ""

# Navigate to project root (two levels up from scripts/testing/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Stop any running instances
echo -e "\n${BLUE}Stopping any running bot instances...${NC}"
pkill -f ctrader_ddqn_paper 2>/dev/null || true
sleep 2

# Load credentials from .env
if [ ! -f .env ]; then
    echo -e "${RED}✗ .env file not found!${NC}"
    echo -e "${YELLOW}Copy .env.example to .env and configure credentials${NC}"
    exit 1
fi

source .env

# Verify required credentials
if [ -z "$CTRADER_USERNAME" ] || [ -z "$CTRADER_PASSWORD_QUOTE" ] || [ -z "$CTRADER_PASSWORD_TRADE" ]; then
    echo -e "${RED}✗ Missing cTrader credentials in .env${NC}"
    exit 1
fi

# Phase 0 Configuration
echo -e "\n${GREEN}Configuring Phase 0: System Validation${NC}"

# CRITICAL: Paper mode with learning DISABLED
export PAPER_MODE=1                  # ✅ PAPER TRADING (no real money)
export DDQN_ONLINE_LEARNING=0        # ✅ LEARNING DISABLED (not training)

# Position sizing (doesn't matter in paper, but keep it small)
export SYMBOL=XAUUSD
export SYMBOL_ID=41
export QTY=0.001                     # Micro size even in paper
export TIMEFRAME_MINUTES=1

# Safety layers (ALL ACTIVE - we're testing these)
export DISABLE_GATES=0               # ✅ All safety gates enabled
export FEAS_THRESHOLD=0.3            # ✅ Path geometry gate
export MAX_BARS_INACTIVE=100         # ✅ Activity monitor

# Exploration (high to test all code paths)
export EPSILON_START=0.5             # ✅ 50% random (test exploration)
export EPSILON_END=0.1               # ✅ Stay somewhat random
export EPSILON_DECAY=0.9999          # ✅ Very slow decay (4 hours)
export EXPLORATION_BOOST=0.2         # ✅ Test activity boost

# Dual-agent architecture
export DDQN_DUAL_AGENT=1             # ✅ TriggerAgent + HarvesterAgent

# cTrader sessions
export CTRADER_QUOTE_CONFIG=config/ctrader_quote.cfg
export CTRADER_TRADE_CONFIG=config/ctrader_trade.cfg

# Create log directory
mkdir -p logs/phase0_validation

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/phase0_validation/validate_${TIMESTAMP}.log"

echo -e "\n${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Configuration Summary:${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "Mode:              ${BOLD}PAPER TRADING${NC}"
echo -e "Learning:          ${BOLD}DISABLED${NC}"
echo -e "Symbol:            ${SYMBOL} (ID=${SYMBOL_ID})"
echo -e "Position Size:     ${QTY} lots (paper)"
echo -e "Timeframe:         M${TIMEFRAME_MINUTES}"
echo -e "Duration:          ${BOLD}2-4 hours maximum${NC}"
echo -e "Safety Gates:      ${BOLD}ALL ACTIVE${NC}"
echo -e "Log File:          ${LOGFILE}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}Starting validation in 3 seconds...${NC}"
echo -e "${YELLOW}Monitor with: tail -f ${LOGFILE}${NC}"
echo -e "${YELLOW}Stop with: Ctrl+C or pkill -f ctrader_ddqn_paper${NC}"
sleep 3

# Set validation timeout (4 hours max)
TIMEOUT_SECONDS=$((4 * 60 * 60))

echo -e "\n${GREEN}✓ Launching bot (${TIMEOUT_SECONDS}s timeout)...${NC}\n"

# Launch bot with timeout via run.sh (handles venv, env vars, FIX sessions)
timeout ${TIMEOUT_SECONDS} ./run.sh --no-hud 2>&1 | tee "$LOGFILE" || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo -e "\n${GREEN}✓ Validation timeout reached (4 hours)${NC}"
        echo -e "${GREEN}✓ If no errors above, system is stable${NC}"
        echo -e "\n${BOLD}NEXT STEP: Start training with ./start_training.sh${NC}"
    else
        echo -e "\n${RED}✗ Bot exited with code ${EXIT_CODE}${NC}"
        echo -e "${YELLOW}Check log for errors: ${LOGFILE}${NC}"
    fi
}

# Validation summary
echo -e "\n${BLUE}Validation Summary:${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"

# Check for common issues in log
ERRORS=$(grep -i "error\|exception\|traceback" "$LOGFILE" | wc -l)
CRASHES=$(grep -i "crash\|fatal\|killed" "$LOGFILE" | wc -l)
CB_TRIPS=$(grep -i "circuit.*breaker.*trip" "$LOGFILE" | wc -l)
TRADES=$(grep -i "order.*filled\|position.*closed" "$LOGFILE" | wc -l)

echo -e "Errors Found:      ${ERRORS}"
echo -e "Crashes:           ${CRASHES}"
echo -e "Circuit Trips:     ${CB_TRIPS}"
echo -e "Trades Executed:   ${TRADES}"
echo ""

if [ $ERRORS -eq 0 ] && [ $CRASHES -eq 0 ]; then
    echo -e "${GREEN}✅ VALIDATION PASSED${NC}"
    echo -e "${GREEN}✅ No critical errors detected${NC}"
    echo -e "\n${BOLD}Ready for training: ./start_training.sh${NC}"
    echo -e "${BOLD}Ready for production: ./start_production.sh${NC}"
else
    echo -e "${RED}❌ VALIDATION FAILED${NC}"
    echo -e "${RED}Found ${ERRORS} errors and ${CRASHES} crashes${NC}"
    echo -e "\n${YELLOW}Fix issues before proceeding to live trading${NC}"
    echo -e "${YELLOW}Review log: ${LOGFILE}${NC}"
fi

echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
