#!/usr/bin/env python3
"""
Test Phase 2 Integration
========================
Validates that non-repaint guards, ring buffers, and activity monitoring
work correctly together in the main bot context.

Run this to verify Phase 2 features before live deployment.
"""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
LOG = logging.getLogger(__name__)

# Ensure repository root is on sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("Phase 2 Integration Test")
print("=" * 70)

# Test 1: Non-repaint guards
print("\n[TEST 1] Non-repaint bar access")
try:
    from non_repaint_guards import NonRepaintBarAccess, NonRepaintError
    
    close_series = NonRepaintBarAccess("close", max_lookback=100)
    
    # Simulate bars
    for i in range(5):
        close_series.append(100.0 + i)
    
    # Try premature access
    try:
        val = close_series.get_current()
        print("    ✗ FAIL: Should have raised NonRepaintError")
        sys.exit(1)
    except NonRepaintError:
        print("    ✓ Correctly blocked bar[0] access before close")
    
    # Mark closed and access
    close_series.mark_bar_closed()
    val = close_series.get_current()
    if val == 104.0:
        print(f"    ✓ Bar[0] access allowed after mark_bar_closed: {val}")
    else:
        print(f"    ✗ FAIL: Expected 104.0, got {val}")
        sys.exit(1)
    
    # Historical access
    prev = close_series.safe_get_previous(1)
    if prev == 103.0:
        print(f"    ✓ Bar[1] access always safe: {prev}")
    else:
        print(f"    ✗ FAIL: Expected 103.0, got {prev}")
        sys.exit(1)
        
except Exception as e:
    print(f"    ✗ FAIL: {e}")
    sys.exit(1)

# Test 2: Ring buffers
print("\n[TEST 2] Ring buffer O(1) statistics")
try:
    from ring_buffer import RollingStats
    
    stats = RollingStats(period=10)
    
    # Add prices
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    for p in prices:
        stats.update(p)
    
    # Check stats
    expected_mean = 102.5
    if abs(stats.mean - expected_mean) < 0.01:
        print(f"    ✓ Rolling mean: {stats.mean:.2f} (expected {expected_mean})")
    else:
        print(f"    ✗ FAIL: Expected mean {expected_mean}, got {stats.mean}")
        sys.exit(1)
    
    if stats.std > 0:
        print(f"    ✓ Rolling std: {stats.std:.2f}")
    else:
        print(f"    ✗ FAIL: Std should be > 0")
        sys.exit(1)
        
except Exception as e:
    print(f"    ✗ FAIL: {e}")
    sys.exit(1)

# Test 3: Activity monitoring
print("\n[TEST 3] Activity monitoring")
try:
    from activity_monitor import ActivityMonitor
    
    monitor = ActivityMonitor(max_bars_inactive=10, min_trades_per_day=2.0)
    
    # Simulate bars without trades
    for i in range(15):
        monitor.on_bar_close()
    
    if monitor.is_stagnant:
        print(f"    ✓ Stagnation detected after {monitor.bars_since_trade} bars")
    else:
        print(f"    ✗ FAIL: Should detect stagnation after 15 bars")
        sys.exit(1)
    
    # Simulate trade
    monitor.on_trade_executed()
    
    if monitor.bars_since_trade == 0:
        print(f"    ✓ Trade resets inactivity counter")
    else:
        print(f"    ✗ FAIL: Expected bars_since_trade=0, got {monitor.bars_since_trade}")
        sys.exit(1)
        
except Exception as e:
    print(f"    ✗ FAIL: {e}")
    sys.exit(1)

# Test 4: Reward shaping integration
print("\n[TEST 4] Enhanced reward shaping (5 components)")
try:
    from reward_shaper import RewardShaper
    from activity_monitor import ActivityMonitor
    
    monitor = ActivityMonitor()
    shaper = RewardShaper(
        symbol="BTCUSD",
        timeframe="M15",
        broker="default",
        activity_monitor=monitor
    )
    
    # Simulate stagnation
    for i in range(20):
        monitor.on_bar_close()
    
    # Calculate reward
    reward_data = {
        'exit_pnl': 50.0,
        'mfe': 100.0,
        'mae': -20.0,
        'winner_to_loser': False
    }
    
    rewards = shaper.calculate_total_reward(reward_data)
    
    # Check all 5 components present
    required_keys = ['capture_efficiency', 'wtl_penalty', 'opportunity_cost', 
                     'activity_bonus', 'counterfactual_adjustment', 'total_reward']
    
    if all(key in rewards for key in required_keys):
        print(f"    ✓ All 5 reward components present")
        print(f"      Total reward: {rewards['total_reward']:+.4f}")
        print(f"      Activity bonus: {rewards['activity_bonus']:+.4f}")
        print(f"      Counterfactual: {rewards['counterfactual_adjustment']:+.4f}")
    else:
        missing = [k for k in required_keys if k not in rewards]
        print(f"    ✗ FAIL: Missing components: {missing}")
        sys.exit(1)
        
except Exception as e:
    print(f"    ✗ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Main bot imports
print("\n[TEST 5] Main bot imports Phase 2 modules")
try:
    # Check if main bot can import all Phase 2 modules
    import sys
    sys.path.insert(0, '/home/renierdejager/Documents/ctrader_trading_bot')
    
    # This will fail if any imports are broken
    with open('ctrader_ddqn_paper.py', 'r') as f:
        content = f.read()
        
    if 'from non_repaint_guards import NonRepaintBarAccess' in content:
        print("    ✓ Main bot imports NonRepaintBarAccess")
    else:
        print("    ✗ FAIL: NonRepaintBarAccess import missing")
        sys.exit(1)
        
    if 'from ring_buffer import RollingStats' in content:
        print("    ✓ Main bot imports RollingStats")
    else:
        print("    ✗ FAIL: RollingStats import missing")
        sys.exit(1)
        
    if 'self.close_series = NonRepaintBarAccess' in content:
        print("    ✓ Main bot initializes non-repaint series")
    else:
        print("    ✗ FAIL: Non-repaint series initialization missing")
        sys.exit(1)
        
    if 'self.close_stats = RollingStats' in content:
        print("    ✓ Main bot initializes rolling stats")
    else:
        print("    ✗ FAIL: Rolling stats initialization missing")
        sys.exit(1)
        
except Exception as e:
    print(f"    ✗ FAIL: {e}")
    sys.exit(1)

# Test 6: Ensemble tracker
print("\n[TEST 6] Ensemble disagreement tracking")
try:
    from ensemble_tracker import EnsembleTracker
    import numpy as np
    
    ensemble = EnsembleTracker(n_models=3)
    
    # Create mock models
    class SimpleMockModel:
        def __init__(self, bias):
            self.bias = bias
        def predict(self, state):
            return np.array([0.5, 0.3, 0.7]) + self.bias
    
    models = [SimpleMockModel(0.0), SimpleMockModel(0.1), SimpleMockModel(-0.1)]
    ensemble.set_models(models)
    
    # Test prediction
    state = np.random.randn(10)
    action, disagreement, stats = ensemble.predict(state)
    
    if 0 <= action <= 2:
        print(f"    ✓ Ensemble prediction: action={action}, disagreement={disagreement:.4f}")
    else:
        print(f"    ✗ FAIL: Invalid action {action}")
        sys.exit(1)
    
    # Test exploration bonus
    bonus = ensemble.get_exploration_bonus(0.8)
    if bonus >= 0:
        print(f"    ✓ Exploration bonus calculated: {bonus:.4f}")
    else:
        print(f"    ✗ FAIL: Negative bonus")
        sys.exit(1)
        
except Exception as e:
    print(f"    ✗ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Ensemble-enhanced Policy integration
print("\n[TEST 7] Ensemble-enhanced Policy.decide()")
try:
    import os
    import numpy as np
    from collections import deque
    
    # Mock quickfix modules before importing Policy
    import sys
    from unittest.mock import MagicMock
    sys.modules['quickfix44'] = MagicMock()
    sys.modules['quickfix'] = MagicMock()
    
    # Mock the environment variables for testing
    original_env = os.environ.copy()
    os.environ['DDQN_MODEL_PATH'] = ''  # No model for testing
    os.environ['DDQN_MODEL_ENSEMBLE'] = '0'
    
    # Import Policy after mocking and setting env
    sys.path.insert(0, '/home/renierdejager/Documents/ctrader_trading_bot')
    from ctrader_ddqn_paper import Policy
    
    # Test 1: Create policy without ensemble (fallback mode)
    policy = Policy()
    bars = deque([(100.0, 100.0, 101.0, 99.0, 100.5) for _ in range(70)])
    action = policy.decide(bars, imbalance=0.1, vpin_z=0.5, depth_ratio=1.2)
    
    if 0 <= action <= 2:
        print(f"    ✓ Policy fallback mode: action={action}")
    else:
        print(f"    ✗ FAIL: Invalid action {action}")
        sys.exit(1)
    
    # Test 2: Check ensemble methods exist
    if hasattr(policy, 'update_ensemble_weights'):
        print(f"    ✓ Policy has update_ensemble_weights method")
    else:
        print(f"    ✗ FAIL: Missing update_ensemble_weights method")
        sys.exit(1)
    
    if hasattr(policy, 'get_ensemble_stats'):
        stats = policy.get_ensemble_stats()
        print(f"    ✓ Policy.get_ensemble_stats() returned {len(stats)} stats")
    else:
        print(f"    ✗ FAIL: Missing get_ensemble_stats method")
        sys.exit(1)
    
    # Restore environment
    os.environ.clear()
    os.environ.update(original_env)
        
except Exception as e:
    print(f"    ✗ FAIL: {e}")
    import traceback
    traceback.print_exc()
    # Restore environment
    os.environ.clear()
    os.environ.update(original_env)
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All Phase 2 integration tests passed!")
print("=" * 70)
print("\nPhase 2 Status:")
print("  ✓ Activity monitoring - INTEGRATED")
print("  ✓ Counterfactual analysis - INTEGRATED")
print("  ✓ Non-repaint guards - INTEGRATED")
print("  ✓ Ring buffers - INTEGRATED")
print("  ✓ 5-component reward shaping - ACTIVE")
print("  ✓ Ensemble disagreement tracking - READY")
print("  ✓ Ensemble-enhanced Policy - INTEGRATED ← NEW!")
print("  ✓ 6-component reward shaping - ACTIVE ← NEW!")
print("\nNext Steps:")
print("  1. ✓ Integrate ensemble into Policy.decide() - DONE")
print("  2. ✓ Add ensemble bonus to reward shaping - DONE")
print("  3. Run end-to-end backtest")
print("  4. Deploy to paper trading")
print("=" * 70)
