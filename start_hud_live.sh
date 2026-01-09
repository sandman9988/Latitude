#!/bin/bash
# Quick Start - HUD with Live Bot Data

echo "═══════════════════════════════════════════════════════════════════"
echo "  HUD LIVE DATA - QUICK START"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "This script demonstrates HUD integration with live bot data."
echo ""
echo "OPTION 1: Run with actual trading bot"
echo "--------------------------------------"
echo "Terminal 1:  ./run.sh              # Start trading bot"
echo "Terminal 2:  python3 hud_display.py # View HUD"
echo ""
echo "OPTION 2: Test with simulated live updates"
echo "-------------------------------------------"
echo "Terminal 1:  python3 test_hud_integration.py update  # Simulate bot"
echo "Terminal 2:  python3 hud_display.py                  # View HUD"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Creating data/ directory..."
    mkdir -p data
fi

# Check current state
if [ -f "data/performance_snapshot.json" ]; then
    echo "✓ Bot data files exist"
    echo ""
    echo "Current state:"
    echo "  Trades: $(python3 -c "import json; print(json.load(open('data/performance_snapshot.json'))['lifetime']['total_trades'])")"
    echo "  Status: $(python3 -c "import json; print(json.load(open('data/current_position.json'))['direction'])")"
else
    echo "⏳ No bot data yet - files will be created when bot starts"
    echo ""
    echo "Initializing empty state..."
    python3 -c "
import json
from pathlib import Path

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

# Empty initial state
files = {
    'performance_snapshot.json': {
        'daily': {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0},
        'weekly': {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0},
        'monthly': {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0},
        'lifetime': {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    },
    'current_position.json': {
        'direction': 'FLAT', 'entry_price': 0.0, 'current_price': 0.0,
        'mfe': 0.0, 'mae': 0.0, 'unrealized_pnl': 0.0, 'bars_held': 0
    },
    'training_stats.json': {
        'trigger_buffer_size': 0, 'harvester_buffer_size': 0,
        'trigger_loss': 0.0, 'harvester_loss': 0.0
    },
    'risk_metrics.json': {
        'var': 0.0, 'kurtosis': 0.0, 'circuit_breaker': 'INACTIVE',
        'vpin': 0.0, 'vpin_zscore': 0.0
    }
}

for filename, data in files.items():
    with open(data_dir / filename, 'w') as f:
        json.dump(data, f, indent=2)

print('✓ Initialized empty state')
"
fi

echo ""
echo "Ready! Choose an option above to start."
echo ""
