#!/bin/bash
# Monitor bot training progress in real-time

echo "=========================================="
echo "  Training Progress Monitor"
echo "=========================================="
echo ""

# Check if bot is running
if ! pgrep -f "python3 -m src.core.ctrader_ddqn_paper" > /dev/null; then
    echo "❌ Bot not running"
    echo "Start training with: source .env.training && ./run.sh --with-hud"
    exit 1
fi

echo "✓ Bot is running"
echo ""

# Get bot PID and runtime
BOT_PID=$(pgrep -f "python3 -m src.core.ctrader_ddqn_paper")
BOT_RUNTIME=$(ps -p "$BOT_PID" -o etime= | tr -d ' ')
echo "📊 Bot Status:"
echo "   PID: $BOT_PID"
echo "   Runtime: $BOT_RUNTIME"
echo ""

# Check current mode
if [ -n "$PAPER_MODE" ] && [ "$PAPER_MODE" = "1" ]; then
    echo "🎓 Mode: TRAINING (PAPER_MODE=1)"
else
    echo "🚀 Mode: PRODUCTION (PAPER_MODE=0 or unset)"
fi

if [ -n "$EPSILON_START" ]; then
    echo "   Epsilon schedule: $EPSILON_START → $EPSILON_END (decay=$EPSILON_DECAY)"
fi
echo ""

# Training stats from logs
echo "📈 Training Activity (last 100 lines):"
TRAINING_LINES=$(tail -100 logs/*.log 2>/dev/null | grep -i "training step\|experience added\|buffer_size" | tail -5)
if [ -n "$TRAINING_LINES" ]; then
    echo "$TRAINING_LINES" | sed 's/^/   /'
else
    echo "   No training activity in recent logs"
fi
echo ""

# Exploration stats
echo "🔍 Exploration Activity (last 100 lines):"
EXPLORE_LINES=$(tail -100 logs/*.log 2>/dev/null | grep -i "explore\|epsilon" | tail -5)
if [ -n "$EXPLORE_LINES" ]; then
    echo "$EXPLORE_LINES" | sed 's/^/   /'
else
    echo "   No exploration activity (expected in LIVE mode)"
fi
echo ""

# Recent trades
echo "💰 Recent Trades:"
TRADE_COUNT=$(wc -l < data/trade_log.jsonl 2>/dev/null || echo "0")
echo "   Total trades: $TRADE_COUNT"
if [ -f data/trade_log.jsonl ]; then
    RECENT_PNL=$(tail -5 data/trade_log.jsonl | jq -r '.pnl' 2>/dev/null | paste -sd '+' | bc 2>/dev/null || echo "N/A")
    echo "   Last 5 trades P&L: $RECENT_PNL"
fi
echo ""

# Learned parameters
echo "🧠 Learned Parameters (XAUUSD_M15):"
if [ -f data/learned_parameters.json ]; then
    CONFIDENCE_FLOOR=$(jq -r '.data.instruments.XAUUSD_M15_default.params.confidence_floor.value // "N/A"' data/learned_parameters.json 2>/dev/null)
    ENTRY_CONF=$(jq -r '.data.instruments.XAUUSD_M15_default.params.entry_confidence_threshold.value // "N/A"' data/learned_parameters.json 2>/dev/null)
    EXIT_CONF=$(jq -r '.data.instruments.XAUUSD_M15_default.params.exit_confidence_threshold.value // "N/A"' data/learned_parameters.json 2>/dev/null)
    PROFIT_TARGET=$(jq -r '.data.instruments.XAUUSD_M15_default.params.profit_target.value // "N/A"' data/learned_parameters.json 2>/dev/null)
    
    echo "   Confidence floor: $CONFIDENCE_FLOOR"
    echo "   Entry threshold: $ENTRY_CONF"
    echo "   Exit threshold: $EXIT_CONF"
    echo "   Profit target: $PROFIT_TARGET"
else
    echo "   No learned parameters file found"
fi
echo ""

# Buffer stats (if available)
echo "💾 Experience Buffer Status:"
python3 -c "
import json
import sys
try:
    # Try to extract buffer info from training stats if available
    with open('data/training_stats.json', 'r') as f:
        stats = json.load(f)
        print(f'   Trigger buffer: {stats.get(\"trigger_buffer_size\", \"N/A\")}')
        print(f'   Harvester buffer: {stats.get(\"harvester_buffer_size\", \"N/A\")}')
except FileNotFoundError:
    print('   Training stats file not found')
except Exception as e:
    print(f'   Error reading training stats: {e}')
" 2>/dev/null || echo "   Unable to read buffer stats"
echo ""

echo "=========================================="
echo "Live monitoring commands:"
echo "  • Watch logs: tail -f logs/*.log | grep -i 'trigger\|harvest'"
echo "  • Watch trades: tail -f data/trade_log.jsonl | jq -r '.'"
echo "  • Re-run this script: ./scripts/monitor_training.sh"
echo "=========================================="
