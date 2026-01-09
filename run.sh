#!/bin/bash
# cTrader Trading Bot Launcher
# This script starts the dual FIX session trading bot

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check required environment variables
check_env() {
    local missing=0
    
    if [ -z "$CTRADER_USERNAME" ]; then
        echo -e "${RED}✗ Missing: CTRADER_USERNAME${NC}"
        missing=1
    fi
    
    if [ -z "$CTRADER_PASSWORD_QUOTE" ]; then
        echo -e "${RED}✗ Missing: CTRADER_PASSWORD_QUOTE${NC}"
        missing=1
    fi
    
    if [ -z "$CTRADER_PASSWORD_TRADE" ]; then
        echo -e "${RED}✗ Missing: CTRADER_PASSWORD_TRADE${NC}"
        missing=1
    fi
    
    if [ $missing -eq 1 ]; then
        echo ""
        echo -e "${YELLOW}Please set the required environment variables:${NC}"
        echo "  export CTRADER_USERNAME=\"your_username\""
        echo "  export CTRADER_PASSWORD_QUOTE=\"your_quote_password\""
        echo "  export CTRADER_PASSWORD_TRADE=\"your_trade_password\""
        exit 1
    fi
    
    echo -e "${GREEN}✓ All required environment variables are set${NC}"
}

# Activate virtual environment if it exists
activate_venv() {
    if [ -f "../.venv/bin/activate" ]; then
        echo -e "${GREEN}✓ Activating virtual environment${NC}"
        source ../.venv/bin/activate
    else
        echo -e "${YELLOW}⚠ Virtual environment not found at ../.venv${NC}"
        echo "  You may need to install quickfix Python bindings"
    fi
}

# Set default configuration
export CTRADER_CFG_QUOTE="${CTRADER_CFG_QUOTE:-config/ctrader_quote.cfg}"
export CTRADER_CFG_TRADE="${CTRADER_CFG_TRADE:-config/ctrader_trade.cfg}"
export CTRADER_BTC_SYMBOL_ID="${CTRADER_BTC_SYMBOL_ID:-10028}"
export CTRADER_QTY="${CTRADER_QTY:-0.10}"
export PY_LOGDIR="${PY_LOGDIR:-logs/python}"

echo "=========================================="
echo "  cTrader DDQN Trading Bot"
echo "=========================================="
echo ""

check_env
activate_venv

echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Symbol ID: $CTRADER_BTC_SYMBOL_ID"
echo "  Quantity:  $CTRADER_QTY"
echo "  Quote CFG: $CTRADER_CFG_QUOTE"
echo "  Trade CFG: $CTRADER_CFG_TRADE"
echo ""
echo -e "${YELLOW}Starting trading bot...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Run the bot
python3 ctrader_ddqn_paper.py
