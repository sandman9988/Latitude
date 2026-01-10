#!/bin/bash
# quick_test.sh
# Quick test run with high exploration to see agents in action
# Runs in paper mode for 30 minutes to test both agents

set -euo pipefail

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Quick Test - Dual-Agent Performance${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Mode:          PAPER (no real money)"
echo "  Learning:      ENABLED (agents will train)"
echo "  Exploration:   HIGH (80% random initially)"
echo "  Duration:      Until you stop (Ctrl+C)"
echo "  Position Size: 0.001 lots"
echo "  Symbol:        XAUUSD (Gold)"
echo ""
echo -e "${GREEN}This will test TriggerAgent entries and HarvesterAgent exits${NC}"
echo ""
echo -e "${YELLOW}In another terminal, run: ./monitor_agents.sh${NC}"
echo ""
echo "Press Ctrl+C to stop..."
sleep 3

# Stop any running instances
pkill -f ctrader_ddqn_paper.py 2>/dev/null || true
sleep 2

# Load credentials
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi

source .env

# Test configuration
export PAPER_MODE=1                  # Paper trading
export DDQN_ONLINE_LEARNING=1        # Learning enabled
export DDQN_DUAL_AGENT=1             # Use dual-agent system
export SYMBOL=XAUUSD
export SYMBOL_ID=41
export QTY=0.001
export TIMEFRAME_MINUTES=1

# High exploration to generate trades quickly
export EPSILON_START=0.8             # 80% random (high exploration)
export EPSILON_END=0.1
export EPSILON_DECAY=0.999           # Slow decay

# Relaxed gates to allow more trades
export DISABLE_GATES=0               # Keep safety on
export FEAS_THRESHOLD=0.2            # Lower threshold = more entries
export MAX_BARS_INACTIVE=50          # Allow some inactivity

# FIX configs
export CTRADER_QUOTE_CONFIG=config/ctrader_quote.cfg
export CTRADER_TRADE_CONFIG=config/ctrader_trade.cfg

# Create test log directory
mkdir -p logs/quick_test

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/quick_test/test_${TIMESTAMP}.log"

echo ""
echo -e "${GREEN}Starting bot...${NC}"
echo -e "Log: ${LOGFILE}"
echo ""

# Run bot
python3 ctrader_ddqn_paper.py 2>&1 | tee "$LOGFILE"

echo ""
echo -e "${BLUE}Test stopped${NC}"
echo -e "Analyze results with: ./analyze_trades.sh"
