#!/bin/bash
# start_bot_with_hud.sh - Launch bot and HUD together

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "  cTrader Bot + HUD Launcher"
echo "=========================================="
echo ""

# Check if already running
if pgrep -f "ctrader_ddqn_paper.py" > /dev/null; then
    echo -e "${YELLOW}⚠${NC} Bot already running"
    echo "Kill it first: pkill -f ctrader_ddqn_paper.py"
    exit 1
fi

# Start bot
echo -e "${BLUE}[1/2]${NC} Starting trading bot..."
bash run.sh > bot_console.log 2>&1 &
BOT_LAUNCHER_PID=$!

# Wait for bot to initialize and create data files
echo "Waiting for bot initialization..."
sleep 5

# Check if bot started successfully
if ! pgrep -f "ctrader_ddqn_paper.py" > /dev/null; then
    echo -e "${RED}✗${NC} Bot failed to start"
    echo "Check bot_console.log for errors"
    exit 1
fi

BOT_PID=$(pgrep -f "ctrader_ddqn_paper.py" | head -1)
echo -e "${GREEN}✓${NC} Bot running (PID: $BOT_PID)"
echo "$BOT_PID" > .bot.pid

# Start HUD
echo ""
echo -e "${BLUE}[2/2]${NC} Starting HUD display..."

# Check if data files exist
if [[ -f "data/current_position.json" ]]; then
    echo -e "${GREEN}✓${NC} Bot data files ready"
else
    echo -e "${YELLOW}⚠${NC} Waiting for bot data files..."
    for i in {1..10}; do
        sleep 1
        if [[ -f "data/current_position.json" ]]; then
            echo -e "${GREEN}✓${NC} Bot data files created"
            break
        fi
    done
fi

# Launch HUD in current terminal
echo ""
echo "=========================================="
echo "  Launching HUD..."
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop HUD (bot will continue running)"
echo ""
sleep 2

# Trap Ctrl+C to only stop HUD
trap "echo ''; echo 'HUD stopped. Bot still running (PID: $BOT_PID)'; exit 0" INT

# Run HUD in foreground
python3 -m src.monitoring.hud_tabbed
