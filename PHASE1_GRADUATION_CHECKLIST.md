# Phase 1 → Phase 2 Graduation Checklist

**Purpose:** Ensure micro-position learning was successful before scaling to mini-lots

---

## Prerequisites (ALL must pass)

### 1. Performance Metrics
- [ ] **Sharpe Ratio > 1.0** (risk-adjusted returns positive)
- [ ] **Sortino Ratio > 0.8** (downside-adjusted returns acceptable)
- [ ] **Win Rate ≥ 45%** (not coin-flip territory)
- [ ] **Max Drawdown < 10%** (capital preservation)
- [ ] **Total Profit > $50** (consistent edge at micro scale)
- [ ] **500+ completed trades** (sufficient sample size)

### 2. System Stability
- [ ] **Uptime > 95%** (no frequent crashes)
- [ ] **Circuit breaker trips < 5%** (safety nets rarely needed)
- [ ] **No manual interventions** for 7 consecutive days
- [ ] **FIX session stability** (minimal disconnects)
- [ ] **No silent failures** (PER buffer updating, logs clean)

### 3. Learning Validation
- [ ] **PER buffer contains friction experiences** (high-slippage trades prioritized)
- [ ] **Epsilon decayed to < 0.10** (exploration → exploitation transition)
- [ ] **TriggerAgent win rate improving** (check last 100 vs first 100 trades)
- [ ] **HarvesterAgent avg profit increasing** (check trend over time)
- [ ] **No degenerate behavior** (not always holding, not always flat)

### 4. Friction Cost Understanding
- [ ] **Average slippage logged** (agents experiencing real 5-20 pip slippage)
- [ ] **Spread cost awareness** (not expecting 2-pip paper spreads)
- [ ] **Requote handling** (agents adapting to rejections)
- [ ] **Execution latency** (agents timing entries with 50-200ms delay)
- [ ] **Overnight rollover** (agents learned session boundaries)

---

## Phase 1 Performance Report Template

```bash
# Generate after 2-4 weeks of micro-position learning
cd ~/Documents/ctrader_trading_bot
python3 -c "
import pandas as pd
import numpy as np
from pathlib import Path

# Load trades
trades_dir = Path('trades')
csv_files = sorted(trades_dir.glob('trades_*.csv'))
if not csv_files:
    print('No trade CSV files found!')
    exit(1)

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Filter Phase 1 only (QTY=0.001)
df_micro = df[df['position_size'] == 0.001].copy()

print('=' * 60)
print('PHASE 1 MICRO-POSITION LEARNING REPORT')
print('=' * 60)
print()
print(f'Total Trades:     {len(df_micro)}')
print(f'Win Rate:         {(df_micro[\"pnl\"] > 0).mean():.1%}')
print(f'Total PnL:        ${df_micro[\"pnl\"].sum():.2f}')
print(f'Avg Profit:       ${df_micro[df_micro[\"pnl\"] > 0][\"pnl\"].mean():.2f}')
print(f'Avg Loss:         ${df_micro[df_micro[\"pnl\"] < 0][\"pnl\"].mean():.2f}')
print(f'Max Drawdown:     ${df_micro[\"pnl\"].cumsum().diff().min():.2f}')
print()

# Sharpe/Sortino
returns = df_micro['pnl'].values
sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
downside = returns[returns < 0].std()
sortino = returns.mean() / (downside + 1e-8) * np.sqrt(252)

print(f'Sharpe Ratio:     {sharpe:.2f}')
print(f'Sortino Ratio:    {sortino:.2f}')
print()

# Friction analysis
if 'slippage_pips' in df_micro.columns:
    print(f'Avg Slippage:     {df_micro[\"slippage_pips\"].mean():.1f} pips')
    print(f'Max Slippage:     {df_micro[\"slippage_pips\"].max():.1f} pips')
print()

# Time analysis
first_100 = df_micro.iloc[:100]
last_100 = df_micro.iloc[-100:]
print(f'First 100 Win %:  {(first_100[\"pnl\"] > 0).mean():.1%}')
print(f'Last 100 Win %:   {(last_100[\"pnl\"] > 0).mean():.1%}')
print(f'Improvement:      {(last_100[\"pnl\"] > 0).mean() - (first_100[\"pnl\"] > 0).mean():.1%}')
print()

# Graduation check
passed = []
failed = []

if sharpe > 1.0:
    passed.append('✓ Sharpe > 1.0')
else:
    failed.append('✗ Sharpe < 1.0')
    
if sortino > 0.8:
    passed.append('✓ Sortino > 0.8')
else:
    failed.append('✗ Sortino < 0.8')
    
if (df_micro['pnl'] > 0).mean() >= 0.45:
    passed.append('✓ Win Rate ≥ 45%')
else:
    failed.append('✗ Win Rate < 45%')
    
if len(df_micro) >= 500:
    passed.append('✓ 500+ trades')
else:
    failed.append('✗ < 500 trades')

print('GRADUATION CHECK:')
print('-' * 60)
for p in passed:
    print(p)
for f in failed:
    print(f)
print()

if not failed:
    print('✅ READY FOR PHASE 2 (QTY=0.01)')
else:
    print('❌ CONTINUE PHASE 1 - Requirements not met')
print('=' * 60)
"
```

---

## If Graduation Criteria NOT Met

**Don't panic.** This is expected. Consider:

### A. Extend Phase 1 Duration
- **Issue:** Only 200 trades completed
- **Fix:** Continue micro-learning for another 2 weeks
- **Reason:** Sample size too small for statistical confidence

### B. Tune Hyperparameters
- **Issue:** Win rate stuck at 35%
- **Fix:** Adjust epsilon decay, learning rate, PER alpha/beta
- **Reason:** Exploration/exploitation imbalance

### C. Analyze PER Buffer
- **Issue:** Agents not learning from mistakes
- **Fix:** Check that high-loss trades have high priority
- **Reason:** Experience replay may be sampling incorrectly

### D. Review Circuit Breakers
- **Issue:** 20% of trades trigger circuit breakers
- **Fix:** Relax Sortino threshold from 0.8 to 0.5
- **Reason:** Safety nets too aggressive for learning phase

### E. Check Data Quality
- **Issue:** Execution logs show many requotes
- **Fix:** Switch broker server or adjust order timing
- **Reason:** Poor execution environment hurting learning

---

## Phase 2 Configuration (ONLY after graduation)

```bash
#!/bin/bash
# launch_mini_learning.sh - Phase 2

export PAPER_MODE=0              # Still live
export DDQN_ONLINE_LEARNING=1    # Still learning
export QTY=0.01                  # 10x increase ($1/pip)

# Tighter safety (scaling up)
export EPSILON_START=0.10        # Less exploration (already learned)
export EPSILON_END=0.02          # Lower floor
export MAX_DRAWDOWN_PCT=8        # Stricter risk control

# All other settings same as Phase 1
```

**Max Loss:** ~$20-30 per trade (10x Phase 1)  
**Duration:** 2-4 weeks  
**Graduation to Phase 3:** Same criteria (Sharpe > 1.0, etc.)

---

## Common Pitfalls

1. **Graduating too early** → Agents not ready for larger positions → blow up
2. **Never graduating** → Waiting for "perfect" metrics → stuck in micro forever
3. **Skipping validation** → Scaling up without checking PER buffer → silent failure
4. **Ignoring stability** → Good metrics but frequent crashes → production disaster

**Balance:** Wait until confident, but don't wait for perfection. If 3 of 4 categories pass after 3 weeks, consider graduation.

---

## Next Steps After Phase 2

- **Phase 3:** Standard lots (QTY=0.10, $10/pip)
- **Requirements:** Even stricter (Sharpe > 1.5, Sortino > 1.0)
- **Duration:** 4-8 weeks before full production
- **End Game:** Disable learning (DDQN_ONLINE_LEARNING=0), freeze weights, scale to multiple symbols
