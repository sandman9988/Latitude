#!/bin/bash
# Live Log Streaming for cTrader Trading Bot
# Run this in a separate terminal to monitor bot activity in real-time

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  cTrader Trading Bot - Live Log Monitor${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Detect main log file
LOG_FILE=""
if [ -f "bot_console.log" ]; then
    LOG_FILE="bot_console.log"
elif [ -f "logs/python/bot.log" ]; then
    LOG_FILE="logs/python/bot.log"
elif [ -f "logs/ctrader/app.log" ]; then
    LOG_FILE="logs/ctrader/app.log"
else
    echo -e "${YELLOW}⚠ No log file found yet. Waiting for bot to start...${NC}"
    echo -e "${BLUE}  Watching for log files in:${NC}"
    echo "    - bot_console.log"
    echo "    - logs/python/bot.log"
    echo "    - logs/ctrader/app.log"
    echo ""
    
    # Wait for log file to appear
    while [ -z "$LOG_FILE" ]; do
        sleep 1
        if [ -f "bot_console.log" ]; then
            LOG_FILE="bot_console.log"
        elif [ -f "logs/python/bot.log" ]; then
            LOG_FILE="logs/python/bot.log"
        elif [ -f "logs/ctrader/app.log" ]; then
            LOG_FILE="logs/ctrader/app.log"
        fi
    done
fi

echo -e "${GREEN}✓ Found log file: ${LOG_FILE}${NC}"
echo -e "${BLUE}  Streaming live logs (Ctrl+C to stop)...${NC}"
echo ""
echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"
echo ""

# Color-code log levels in output
tail -f "$LOG_FILE" | while IFS= read -r line; do
    # Color coding based on log level
    if echo "$line" | grep -qi "ERROR\|CRITICAL\|FATAL"; then
        echo -e "${RED}${line}${NC}"
    elif echo "$line" | grep -qi "WARNING\|WARN"; then
        echo -e "${YELLOW}${line}${NC}"
    elif echo "$line" | grep -qi "INFO"; then
        echo -e "${GREEN}${line}${NC}"
    elif echo "$line" | grep -qi "DEBUG"; then
        echo -e "${CYAN}${line}${NC}"
    elif echo "$line" | grep -qi "\[BAR\]\|\[LOGON\]\|\[POSITION\]\|\[TRADE\]"; then
        # Highlight important events
        echo -e "${MAGENTA}${line}${NC}"
    elif echo "$line" | grep -qi "CIRCUIT"; then
        # Circuit breaker events
        echo -e "${RED}${line}${NC}"
    elif echo "$line" | grep -qi "TRAIN\|LEARNING"; then
        # Training/learning events
        echo -e "${BLUE}${line}${NC}"
    else
        echo "$line"
    fi
done
