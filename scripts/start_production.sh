#!/bin/bash
# Start bot in PRODUCTION mode with learned parameters

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

echo "=========================================="
echo "  Starting Bot in PRODUCTION MODE"
echo "=========================================="
echo ""

# Export production environment variables
export PAPER_MODE=0
export DISABLE_GATES=0
export EPSILON_START=0.05
export EPSILON_END=0.01
export EPSILON_DECAY=0.9995
export EXPLORATION_BOOST=0.0
export FORCE_EXPLORATION=0
export MAX_BARS_INACTIVE=1000
export DDQN_ONLINE_LEARNING=1

echo "✓ Production environment configured:"
echo "   PAPER_MODE=$PAPER_MODE (learned gates enabled)"
echo "   EPSILON: $EPSILON_START → $EPSILON_END (minimal exploration)"
echo "   FORCE_EXPLORATION=$FORCE_EXPLORATION"
echo "   DISABLE_GATES=$DISABLE_GATES (confidence floors ACTIVE)"
echo "   DDQN_ONLINE_LEARNING=$DDQN_ONLINE_LEARNING"
echo ""

# Kill existing bot if running
if pgrep -f "ctrader_ddqn_paper" > /dev/null; then
    echo "⚠️  Stopping existing bot..."
    pkill -f "ctrader_ddqn_paper" || true
    sleep 3
fi

# Check if learned parameters exist
if [ ! -f "data/learned_parameters.json" ]; then
    echo "⚠️  WARNING: No learned parameters found!"
    echo "   Consider training first: ./scripts/start_training.sh"
    echo ""
    read -p "Continue anyway with defaults? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "🚀 Launching bot with HUD..."
echo ""

# Launch with environment variables properly exported
./run.sh --with-hud

echo ""
echo "=========================================="
echo "✓ Production mode started!"
echo ""
echo "Monitor performance:"
echo "  ./scripts/monitor_training.sh"
echo ""
echo "Watch trades:"
echo "  tail -f data/trade_log.jsonl | jq -r '.'"
echo "=========================================="
