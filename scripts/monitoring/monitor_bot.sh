#!/bin/bash
# Monitor cTrader bot output

cd ~/Documents/ctrader_trading_bot

# Find most recent log file
LATEST_LOG=$(ls -t ctrader_py_logs/ctrader_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No log files found in ctrader_py_logs/"
    exit 1
fi

echo "📊 Monitoring: $LATEST_LOG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if bot is running
BOT_PID=$(ps aux | grep "[p]ython3 ctrader_ddqn_paper.py" | awk '{print $2}')
if [ -n "$BOT_PID" ]; then
    echo "✅ Bot running (PID: $BOT_PID)"
else
    echo "⚠️  Bot not running"
fi
echo ""

# Show filter options
echo "Choose filter:"
echo "  1) All events (verbose)"
echo "  2) Important only (BAR, TRADE, MFE/MAE, ERROR)"
echo "  3) Bar closes only"
echo "  4) Trades only"
echo "  q) Quit"
echo ""
read -p "Select [1-4]: " choice

case $choice in
    1)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        tail -f "$LATEST_LOG"
        ;;
    2)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        tail -f "$LATEST_LOG" | grep --line-buffered -E "BAR M|TRADE|MFE/MAE|ERROR|WARNING"
        ;;
    3)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        tail -f "$LATEST_LOG" | grep --line-buffered "BAR M"
        ;;
    4)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        tail -f "$LATEST_LOG" | grep --line-buffered -E "TRADE.*Sent|POS-UPDATE|MFE/MAE"
        ;;
    *)
        echo "Cancelled"
        exit 0
        ;;
esac
