#!/usr/bin/env python3
"""
Test HUD Integration with Live Data
Creates mock bot data and verifies HUD reads it correctly
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta, UTC

def create_mock_data():
    """Create mock data files that bot would export"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Mock performance snapshot
    performance = {
        'daily': {
            'total_trades': 3,
            'win_rate': 0.667,
            'total_pnl': 125.50,
            'sharpe_ratio': 2.34,
            'max_drawdown': 0.015
        },
        'weekly': {
            'total_trades': 12,
            'win_rate': 0.583,
            'total_pnl': 387.25,
            'sharpe_ratio': 1.87,
            'max_drawdown': 0.032
        },
        'monthly': {
            'total_trades': 45,
            'win_rate': 0.622,
            'total_pnl': 1234.80,
            'sharpe_ratio': 1.95,
            'max_drawdown': 0.067
        },
        'lifetime': {
            'total_trades': 156,
            'win_rate': 0.641,
            'total_pnl': 5678.90,
            'sharpe_ratio': 1.89,
            'max_drawdown': 0.098
        }
    }
    
    with open(data_dir / "performance_snapshot.json", 'w') as f:
        json.dump(performance, f, indent=2)
    
    # Mock current position
    position = {
        'direction': 'LONG',
        'entry_price': 2650.25,
        'current_price': 2652.80,
        'mfe': 3.50,
        'mae': 1.20,
        'unrealized_pnl': 2.55,
        'bars_held': 8
    }
    
    with open(data_dir / "current_position.json", 'w') as f:
        json.dump(position, f, indent=2)
    
    # Mock training stats
    training = {
        'trigger_buffer_size': 456,
        'harvester_buffer_size': 489,
        'trigger_loss': 0.0234,
        'harvester_loss': 0.0198
    }
    
    with open(data_dir / "training_stats.json", 'w') as f:
        json.dump(training, f, indent=2)
    
    # Mock risk metrics
    risk = {
        'var': 0.002340,
        'kurtosis': 2.67,
        'circuit_breaker': 'INACTIVE',
        'vpin': 0.123,
        'vpin_zscore': 0.45
    }
    
    with open(data_dir / "risk_metrics.json", 'w') as f:
        json.dump(risk, f, indent=2)
    
    print("✓ Created mock data files in data/")
    print(f"  - performance_snapshot.json")
    print(f"  - current_position.json")
    print(f"  - training_stats.json")
    print(f"  - risk_metrics.json")


def update_position_price():
    """Simulate live price updates"""
    data_dir = Path("data")
    position_file = data_dir / "current_position.json"
    
    if not position_file.exists():
        return
    
    with open(position_file) as f:
        position = json.load(f)
    
    # Simulate price movement
    import random
    price_change = random.uniform(-0.5, 0.5)
    position['current_price'] += price_change
    
    # Update P&L
    entry = position['entry_price']
    current = position['current_price']
    position['unrealized_pnl'] = current - entry
    
    # Update MFE/MAE
    if position['unrealized_pnl'] > position['mfe']:
        position['mfe'] = position['unrealized_pnl']
    if position['unrealized_pnl'] < -position['mae']:
        position['mae'] = abs(position['unrealized_pnl'])
    
    position['bars_held'] += 1
    
    with open(position_file, 'w') as f:
        json.dump(position, f, indent=2)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        # Update mode: continuously update position
        print("Starting live position updates...")
        print("Press Ctrl+C to stop")
        try:
            while True:
                update_position_price()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n✓ Stopped updates")
    else:
        # Create initial mock data
        create_mock_data()
        print("\nTo simulate live updates, run:")
        print("  python3 test_hud_integration.py update")
        print("\nTo view in HUD, run in another terminal:")
        print("  python3 hud_display.py")
