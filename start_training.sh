#!/bin/bash
# Start bot in TRAINING mode with exploration and online learning

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=========================================="
echo "  Starting Bot in TRAINING MODE"
echo "=========================================="
echo ""

# Export training environment variables
export PAPER_MODE=1
export DISABLE_GATES=1
export EPSILON_START=1.0
export EPSILON_END=0.1
export EPSILON_DECAY=0.998
export EXPLORATION_BOOST=0.5
export FORCE_EXPLORATION=1
export MAX_BARS_INACTIVE=10
export DDQN_ONLINE_LEARNING=1

echo "✓ Training environment configured:"
echo "   PAPER_MODE=$PAPER_MODE (exploration enabled)"
echo "   EPSILON: $EPSILON_START → $EPSILON_END (decay=$EPSILON_DECAY)"
echo "   FORCE_EXPLORATION=$FORCE_EXPLORATION (max_bars_inactive=$MAX_BARS_INACTIVE)"
echo "   DISABLE_GATES=$DISABLE_GATES (no confidence floors)"
echo "   DDQN_ONLINE_LEARNING=$DDQN_ONLINE_LEARNING"
echo ""

# Kill existing bot if running
if pgrep -f "ctrader_ddqn_paper" > /dev/null; then
    echo "⚠️  Stopping existing bot..."
    pkill -f "ctrader_ddqn_paper" || true
    sleep 3
fi

echo "🚀 Launching bot with HUD..."
echo ""

# Launch with environment variables properly exported
./run.sh --with-hud

echo ""
echo "=========================================="
echo "✓ Training mode started!"
echo ""
echo "Monitor training:"
echo "  ./scripts/monitor_training.sh"
echo ""
echo "Watch logs:"
echo "  tail -f logs/*.log | grep -i 'explore\|training'"
echo "=========================================="
