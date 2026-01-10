# 🚀 Deployment Quickstart - Living Ecosystem Strategy

**Goal:** Get agents learning from real friction costs ASAP while minimizing risk

---

## Prerequisites

1. ✅ Phase 3 integration complete (circuit breakers, event features, safe math)
2. ✅ `.env` file configured with cTrader credentials
3. ✅ Dependencies installed: `pip install -r requirements.txt`
4. ✅ FIX configs in `config/ctrader_quote.cfg` and `config/ctrader_trade.cfg`

---

## 3-Step Launch Process

### Step 1: System Validation (2-4 hours)

**Purpose:** Verify no crashes before risking real money

```bash
./phase0_validate_system.sh
```

**What it does:**
- Runs in PAPER mode with learning DISABLED
- Tests all code paths (circuit breakers, FIX sessions, path geometry)
- Auto-stops after 4 hours
- Logs to `logs/phase0_validation/`

**Success Criteria:**
- ✅ No crashes or exceptions
- ✅ FIX sessions stay connected
- ✅ Circuit breakers trigger correctly
- ✅ At least 10-20 paper trades executed

**If validation fails:** Fix errors, re-run Phase 0

---

### Step 2: Launch Live Micro-Position Learning (2-4 weeks)

**Purpose:** Build living ecosystem with real friction at tiny risk

```bash
./launch_micro_learning.sh
```

**What it does:**
- LIVE trading with `QTY=0.001` ($0.10/pip on XAUUSD)
- Learning ENABLED (agents update weights from real experience)
- Max loss: ~$2-3 per trade
- Logs to `logs/live_micro/`

**Monitor in real-time:**
```bash
./monitor_phase1.sh
```

**Dashboard shows:**
- Total trades, win rate, recent PnL
- Circuit breaker trips
- FIX session status
- Graduation progress

**Run Duration:** 2-4 weeks (target 500-1000 trades)

---

### Step 3: Validate Graduation (after 500+ trades)

**Purpose:** Ensure agents learned successfully before scaling

```bash
cd ~/Documents/ctrader_trading_bot
python3 -c "
import pandas as pd
import numpy as np
from pathlib import Path

# Load Phase 1 trades
trades_dir = Path('trades')
csv_files = sorted(trades_dir.glob('trades_*.csv'))
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df_micro = df[df['position_size'] == 0.001].copy()

# Calculate metrics
returns = df_micro['pnl'].values
sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
downside = returns[returns < 0].std()
sortino = returns.mean() / (downside + 1e-8) * np.sqrt(252)

print('=' * 60)
print('PHASE 1 GRADUATION CHECK')
print('=' * 60)
print(f'Total Trades:     {len(df_micro)}')
print(f'Win Rate:         {(df_micro[\"pnl\"] > 0).mean():.1%}')
print(f'Sharpe Ratio:     {sharpe:.2f}')
print(f'Sortino Ratio:    {sortino:.2f}')
print()

# Check graduation
passed = []
if len(df_micro) >= 500: passed.append('✓ 500+ trades')
if sharpe > 1.0: passed.append('✓ Sharpe > 1.0')
if sortino > 0.8: passed.append('✓ Sortino > 0.8')
if (df_micro['pnl'] > 0).mean() >= 0.45: passed.append('✓ Win Rate ≥ 45%')

print('PASSED:', ', '.join(passed) if passed else 'NONE')

if len(passed) >= 3:
    print()
    print('✅ READY FOR PHASE 2 (QTY=0.01)')
else:
    print()
    print('❌ CONTINUE PHASE 1 - Need 3/4 criteria')
print('=' * 60)
"
```

**Graduation Criteria (need 3 of 4):**
- ✅ Sharpe Ratio > 1.0
- ✅ Sortino Ratio > 0.8
- ✅ Win Rate ≥ 45%
- ✅ 500+ completed trades

**If NOT ready:** Continue Phase 1 for another week, tune hyperparameters if needed

**If ready:** Scale to Phase 2 (see below)

---

## Phase 2: Mini-Lot Learning (after graduation)

**Configuration:**
```bash
export QTY=0.01              # 10x increase ($1/pip)
export EPSILON_START=0.10    # Less exploration (already learned basics)
export MAX_DRAWDOWN_PCT=8    # Stricter risk control
```

**Duration:** 2-4 weeks  
**Max Loss:** ~$20-30 per trade  
**Graduation:** Same criteria (Sharpe > 1.0, Sortino > 0.8)

---

## Phase 3: Standard Lots (production)

**Configuration:**
```bash
export QTY=0.10               # Standard lot ($10/pip)
export DDQN_ONLINE_LEARNING=0 # FREEZE weights (stop learning)
export EPSILON_START=0.02     # Pure exploitation
```

**Requirements:**
- ✅ Sharpe > 1.5 from Phase 2
- ✅ Sortino > 1.0 from Phase 2
- ✅ 4-8 weeks Phase 2 profitability
- ✅ Max drawdown < 5%

---

## File Reference

| File | Purpose |
|------|---------|
| [PAPER_VS_LIVE_CONFIG.md](PAPER_VS_LIVE_CONFIG.md) | Full strategy explanation, RL complacency problem |
| [phase0_validate_system.sh](phase0_validate_system.sh) | 2-4 hour paper validation |
| [launch_micro_learning.sh](launch_micro_learning.sh) | Phase 1 startup script |
| [monitor_phase1.sh](monitor_phase1.sh) | Real-time dashboard |
| [PHASE1_GRADUATION_CHECKLIST.md](PHASE1_GRADUATION_CHECKLIST.md) | Detailed validation guide |

---

## Troubleshooting

### Bot crashes during Phase 0
→ Check logs for specific error, fix, re-run validation

### FIX sessions keep disconnecting
→ Verify credentials in `.env`, check broker server status

### Win rate stuck below 30% in Phase 1
→ Tune hyperparameters (epsilon decay, PER alpha/beta)  
→ Check PER buffer is sampling high-loss experiences  
→ Verify event time features are logging correctly

### Circuit breakers trip >10% of time
→ Relax Sortino threshold: `export CB_SORTINO_MIN=0.5`  
→ Increase consecutive loss limit: `export CB_CONSEC_LOSSES=5`

### Not making any trades
→ Increase exploration: `export EPSILON_START=0.5`  
→ Lower feasibility threshold: `export FEAS_THRESHOLD=0.2`  
→ Check `MAX_BARS_INACTIVE` not too aggressive

---

## Why This Works

**The RL Bootstrapping Paradox:**
> "We need real trades to learn, but we don't learn because we don't make real trades because we don't have experience making real trades."

**Solution:** Micro-positions break the Catch-22
- Agents experience REAL friction (5-20 pip slippage, not 1-2 paper pips)
- Agents learn REAL execution delays, requotes, spread variability
- Max loss is pocket change (~$2-3 per trade)
- Circuit breakers protect from catastrophic mistakes
- PER buffer prioritizes expensive lessons (high slippage = high priority)

**Result:** Living ecosystem that learns from real mistakes at acceptable cost

---

## Expected Timeline

| Phase | Duration | Trades | Max Loss/Trade | Total Risk |
|-------|----------|--------|----------------|------------|
| Phase 0 | 2-4 hours | 10-20 | $0 (paper) | $0 |
| Phase 1 | 2-4 weeks | 500-1000 | $2-3 | $500-1500 |
| Phase 2 | 2-4 weeks | 500-1000 | $20-30 | $5k-15k |
| Phase 3 | Ongoing | - | $100-200 | Production |

**Conservative path:** ~2 months from Phase 0 to Phase 3  
**Aggressive path:** ~1 month (if metrics excellent early)

---

## Key Takeaway

🎯 **START LIVE WITH MICRO POSITIONS FROM DAY ONE**

Don't wait for "perfect" paper performance. Paper trading creates complacency—agents learn to expect fills that don't exist in reality. The living ecosystem approach builds robust policies from ground truth while keeping risk tiny during the learning phase.
