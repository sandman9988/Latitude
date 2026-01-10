#!/bin/bash
# launch_micro_learning.sh
# Phase 1: Live micro-position learning with real friction costs
# 
# This is the RECOMMENDED starting point for the living ecosystem.
# Agents learn from real market conditions with tiny position sizes.

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Phase 1: LIVE Micro-Position Learning${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}⚠ WARNING: This is LIVE TRADING with real money${NC}"
echo -e "${GREEN}✓ Position size: 0.001 lots ($0.10/pip on XAUUSD)${NC}"
echo -e "${GREEN}✓ Max loss per trade: ~$2-3${NC}"
echo -e "${GREEN}✓ Learning friction costs from real market${NC}"
echo -e "${GREEN}✓ Circuit breakers active${NC}"
echo ""
echo -e "${BLUE}This creates a living ecosystem where agents:${NC}"
echo -e "${BLUE}- Make real mistakes with tiny consequences${NC}"
echo -e "${BLUE}- Learn actual friction costs (spread + slippage)${NC}"
echo -e "${BLUE}- Experience real execution delays and requotes${NC}"
echo -e "${BLUE}- Build robust policy from ground truth${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C now if you want to abort...${NC}"
sleep 5

# Navigate to bot directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Stop any running instances
echo -e "\n${BLUE}Stopping any running bot instances...${NC}"
pkill -f ctrader_ddqn_paper.py 2>/dev/null || true
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

# Phase 1 Configuration
echo -e "\n${GREEN}Configuring Phase 1: Live Micro-Position Learning${NC}"

# CRITICAL: Live trading with learning enabled
export PAPER_MODE=0                  # ✅ LIVE TRADING (real money)
export DDQN_ONLINE_LEARNING=1        # ✅ LEARNING ENABLED (build ecosystem)

# Position sizing
export SYMBOL=XAUUSD
export SYMBOL_ID=41
export QTY=0.001                     # ✅ MICRO POSITIONS ($0.10/pip)
export TIMEFRAME_MINUTES=1

# Safety layers (ALL ACTIVE)
export DISABLE_GATES=0               # ✅ All safety gates enabled
export FEAS_THRESHOLD=0.3            # ✅ Path geometry gate
export MAX_BARS_INACTIVE=100         # ✅ Activity monitor

# Exploration strategy (high to learn friction)
export EPSILON_START=0.3             # ✅ HIGH EXPLORATION (30% random)
export EPSILON_END=0.05              # ✅ Decay to 5% over time
export EPSILON_DECAY=0.9995          # ✅ Slow decay (learn thoroughly)
export EXPLORATION_BOOST=0.1         # ✅ Activity boost when stagnant

# Dual-agent architecture
export DDQN_DUAL_AGENT=1             # ✅ TriggerAgent + HarvesterAgent

# cTrader sessions
export CTRADER_QUOTE_CONFIG=config/ctrader_quote.cfg
export CTRADER_TRADE_CONFIG=config/ctrader_trade.cfg

# Create log directory
mkdir -p logs/live_micro

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/live_micro/learning_${TIMESTAMP}.log"

echo -e "\n${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Configuration Summary:${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "Mode:              ${BOLD}LIVE TRADING${NC}"
echo -e "Learning:          ${BOLD}ENABLED${NC}"
echo -e "Symbol:            ${SYMBOL} (ID=${SYMBOL_ID})"
echo -e "Position Size:     ${BOLD}${QTY} lots${NC} (~\$0.10/pip)"
echo -e "Max Loss/Trade:    ${BOLD}~\$2-3${NC}"
echo -e "Timeframe:         M${TIMEFRAME_MINUTES}"
echo -e "Epsilon:           ${EPSILON_START} → ${EPSILON_END} (decay=${EPSILON_DECAY})"
echo -e "Safety Gates:      ${BOLD}ALL ACTIVE${NC}"
echo -e "Log File:          ${LOGFILE}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}Starting live micro-position learning in 5 seconds...${NC}"
echo -e "${YELLOW}Monitor logs with: tail -f ${LOGFILE}${NC}"
sleep 5

# Launch bot with logging
echo -e "\n${GREEN}✓ Launching bot...${NC}\n"
python3 ctrader_ddqn_paper.py 2>&1 | tee "$LOGFILE"

# If we get here, bot exited
echo -e "\n${YELLOW}⚠ Bot stopped${NC}"
echo -e "${BLUE}Check log for details: ${LOGFILE}${NC}"
