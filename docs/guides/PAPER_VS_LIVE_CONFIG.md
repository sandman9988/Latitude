# Paper Trading vs Live Trading Configuration

**Critical Decision:** Should RL agents learn during paper trading?

**Answer: NO** - Disable online learning during paper trading to prevent complacency.

---

## The RL Bootstrapping Paradox

**The Catch-22:**
- Need experience to trade well
- Need to trade to get experience
- Can't get real experience from paper trading
- Paper trading creates false confidence

**The Solution:**
> **START LIVE WITH MICRO POSITIONS ON DAY ONE**

**Why This Works:**
- ✅ Real friction costs ($2-8 pip spread, variable)
- ✅ Real slippage (5-20 pips depending on volatility)
- ✅ Real requotes and rejections
- ✅ Real execution delays
- ✅ Maximum loss = $2-3 per trade (acceptable learning cost)
- ✅ Agents build robust policy from ground truth
- ✅ No need to "unlearn" paper trading habits
- ✅ Circuit breakers prevent catastrophic mistakes
- ✅ Creates true living ecosystem from day one

---

## Recommended Configuration

### Phase 0: System Validation ONLY (2-4 hours)
**Purpose:** Verify bot doesn't crash - nothing more

**Configuration:**
```bash
export DDQN_ONLINE_LEARNING=0    # ❌ NO LEARNING
export PAPER_MODE=1
export QTY=0.01
export DISABLE_GATES=0
```

**Goal:** Confirm no crashes, then immediately move to Phase 1
**Duration:** 2-4 hours maximum

---

### Phase 1: LIVE Micro-Position Learning ⭐ START HERE
**Purpose:** Build living ecosystem with real friction from day one

**Configuration:**
```bash
export DDQN_ONLINE_LEARNING=1    # ✅ LEARN FROM REALITY
export PAPER_MODE=0              # ✅ LIVE TRADING
export QTY=0.001                 # ✅ MICRO POSITIONS ($0.10/pip)
export DISABLE_GATES=0           # ✅ ALL SAFETY ACTIVE
export EPSILON_START=0.3         # ✅ HIGH EXPLORATION (learn friction)
export EPSILON_END=0.05
export EPSILON_DECAY=0.9995
```

**Why Micro-Positions:**
- ✅ Real spread (2-8 pips variable, not fixed 2)
- ✅ Real slippage (2-20 pips depending on volatility)
- ✅ Real requotes and rejections
- ✅ Real execution delays (50-200ms latency)
- ✅ Agents learn TRUE friction costs
- ✅ Maximum loss = $2-3 per trade (acceptable learning cost)
- ✅ NO complacency from perfect fills

**Behavior:**
- ✅ Agents make real mistakes with tiny consequences
- ✅ PER prioritizes high-cost experiences (friction lessons)
- ✅ Circuit breakers protect from catastrophic learning
- ✅ Event features learn real session characteristics
- ✅ Builds robust policy from ground truth

**Duration:** 2-4 weeks (500-1000 trades at micro scale)
**Goal:** Learn friction, build robust policy, prove profitability at micro scale

---

### Phase 2: LIVE Standard Positions (After Micro Success)
**Purpose:** Scale up after proving profitability at micro level

**Configuration:**
```bash
export DDQN_ONLINE_LEARNING=1    # ✅ CONTINUE LEARNING
export PAPER_MODE=0
export QTY=0.01                  # ✅ STANDARD MINI LOT ($1/pip)
export DISABLE_GATES=0           # ✅ ALL SAFETY ACTIVE
export EPSILON_START=0.15        # ✅ MODERATE EXPLORATION
export EPSILON_END=0.05
export EPSILON_DECAY=0.9995
```

**Prerequisites:**
- ✅ Profitable at QTY=0.001 over 500+ trades
- ✅ Sharpe > 1.0, Sortino > 0.8 at micro scale
- ✅ Max drawdown < 10% at micro scale
- ✅ Circuit breakers triggered < 5% of time
- ✅ Win rate stable > 45%

**Duration:** 2-3 months (1000-2000 trades)
**Goal:** Confirm scalability, continue adaptation

---

### Phase 3: LIVE Full Positions (Production)
**Purpose:** Full production deployment after extensive validation

**Configuration:**
```bash
export DDQN_ONLINE_LEARNING=1    # ✅ CONTINUOUS LEARNING
export PAPER_MODE=0
export QTY=0.10                  # ✅ STANDARD LOT ($10/pip)
export DISABLE_GATES=0
export EPSILON_START=0.10        # ✅ LOW EXPLORATION (confident policy)
export EPSILON_END=0.03
export EPSILON_DECAY=0.9998
```

**Prerequisites:**
- ✅ Profitable at QTY=0.01 over 1000+ trades
- ✅ Sharpe > 1.5, Sortino > 1.2
- ✅ Max drawdown < 15% at mini lot scale
- ✅ 3+ months live validation
- ✅ Consistent monthly profitability

**Duration:** Indefinite (with monitoring)
**Goal:** Production trading with continuous adaptation

---

### Offline Training (Development)
**Purpose:** Pre-train on historical data before any trading

**Configuration:**
```bash
# Run in training mode with historical bars
export DDQN_ONLINE_LEARNING=1
export TRAINING_MODE=1           # Custom: no live connection
export EPSILON_START=1.0         # High initial exploration
export EPSILON_END=0.1
export TRAINING_EPISODES=10000
```

**Behavior:**
- ✅ Learn from historical M1/M5/M15 bars
- ✅ Realistic friction costs from past spreads
- ✅ No execution bias (historical data has true fills)
- ✅ Build initial policy before live risk

**Duration:** Until convergence (may take days)
**Goal:** Warm-start agents before production

---

## Why Paper Learning Fails

### Example: The "Perfect Entry" Trap

**Paper Trading Scenario:**
1. Agent sees setup at 2650.00
2. Sends market buy order
3. Gets filled instantly at 2650.01 (1 pip slippage)
4. MFE reaches 2652.00 (+200 pips profit)
5. Reward = +10.5 (huge)
6. PER stores with high priority
7. Agent learns: "This setup = guaranteed profit"

**Live Trading Reality:**
1. Agent sees same setup at 2650.00
2. Sends market buy order
3. Order gets queued (50ms latency)
4. Price moves to 2650.15 before execution
5. Gets filled at 2650.20 (20 pip slippage + spread)
6. MFE only reaches 2651.50 (+130 pips)
7. Reward = +4.2 (mediocre)
8. Reality doesn't match learned expectation
9. Agent's confidence was misplaced

**The Problem:** Agent learned from 1-pip slippage but faces 20-pip slippage live. The policy optimized for paper won't work in production.

---

## Circuit Breaker Validation (Paper Mode Use Case)

**Acceptable Paper Trading Use:**
- ✅ Test circuit breaker trip thresholds
- ✅ Verify event feature logging
- ✅ Validate path geometry calculations
- ✅ Check SafeMath prevents crashes
- ✅ Measure system latency and throughput

**What NOT to Validate in Paper:**
- ❌ Actual profitability (friction mismatch)
- ❌ Win rate (execution quality mismatch)
- ❌ Position sizing effectiveness (no market impact)
- ❌ RL agent decision quality (no real risk)

---

## Recommended Workflow

### Phase 1: Offline Training (Optional)
```bash
# Train on historical data first
python3 train_offline.py --bars data/XAUUSD_M1_2025.csv --episodes 10000
# Saves: models/trigger_offline.pth, models/harvester_offline.pth
```

### Phase 2: Paper Validation (No Learning)
```bash
# 24-48 hour system validation
export DDQN_ONLINE_LEARNING=0
export PAPER_MODE=1
bash run.sh

# Monitor:
- Circuit breaker trips (should happen during bad streaks)
- Event feature patterns (session awareness)
- System crashes (should be zero with SafeMath)
- Log for errors/warnings
```

### Phase 3: Live Trading (Learning Enabled)
```bash
# Production deployment with learning
export DDQN_ONLINE_LEARNING=1
export PAPER_MODE=0
bash run.sh

# Monitor for 3 months:
- Performance metrics (Sharpe, Sortino, drawdown)
- Circuit breaker effectiveness
- Event feature importance (after 200+ trades)
- Agent weight stability (not oscillating)
```

---

## Environment Variables Reference

| Variable | Paper (Validate) | Live (Learn) | Purpose |
|----------|------------------|--------------|---------|
| `DDQN_ONLINE_LEARNING` | **0** | **1** | Enable/disable RL weight updates |
| `PAPER_MODE` | **1** | **0** | Paper vs live execution |
| `DISABLE_GATES` | **0** | **0** | Keep all safety gates active |
| `EPSILON_START` | 0.1 | 0.15 | Initial exploration rate |
| `EPSILON_END` | 0.05 | 0.05 | Final exploration rate |
| `EPSILON_DECAY` | 0.999 | 0.9995 | Exploration decay rate |
| `FEAS_THRESHOLD` | 0.3 | 0.3 | Path geometry feasibility gate |
| `MAX_BARS_INACTIVE` | 100 | 100 | Activity monitor trigger |

---

## Key Principle

> **"Paper trading validates the SYSTEM. Live trading trains the AGENTS."**

- **System validation** = Does it work without crashing? Do safety layers activate?
- **Agent training** = Can it learn profitable patterns from REAL market conditions?

Paper trading with learning enabled conflates these two goals and creates complacency.

---

## Example: Proper Paper Trading Run

```bash
#!/bin/bash
# paper_validate.sh - System validation ONLY (no learning)

# Stop any running bot
pkill -f ctrader_ddqn_paper.py
sleep 2

# Navigate to bot directory
cd /home/renierdejager/Documents/ctrader_trading_bot

# Configuration for paper validation
export PAPER_MODE=1
export DDQN_ONLINE_LEARNING=0        # ← CRITICAL: NO LEARNING
export DDQN_DUAL_AGENT=1
export DISABLE_GATES=0               # Use all safety gates
export EPSILON_START=0.1             # Low exploration (use policy as-is)
export EPSILON_END=0.05
export EPSILON_DECAY=0.999

# Symbol configuration
export SYMBOL=XAUUSD
export SYMBOL_ID=41
export QTY=0.01
export TIMEFRAME_MINUTES=1

# cTrader credentials
export CTRADER_USERNAME=5179095
export CTRADER_QUOTE_PASSWORD='YourQuotePassword'
export CTRADER_TRADE_PASSWORD='YourTradePassword'
export CTRADER_QUOTE_CONFIG=config/ctrader_quote.cfg
export CTRADER_TRADE_CONFIG=config/ctrader_trade.cfg

# Run with logging
python3 ctrader_ddqn_paper.py 2>&1 | tee paper_validation.log

# After 24-48 hours, analyze logs:
# - grep "CIRCUIT-BREAKER" paper_validation.log
# - grep "EVENT-TIME" paper_validation.log  
# - grep "ERROR\|WARNING" paper_validation.log
```

---

## Example: Proper Live Trading Run

```bash
#!/bin/bash
# live_trade.sh - Production with learning ENABLED

# Stop any running bot
pkill -f ctrader_ddqn_paper.py
sleep 2

# Navigate to bot directory
cd /home/renierdejager/Documents/ctrader_trading_bot

# Configuration for live learning
export PAPER_MODE=0                  # ← LIVE TRADING
export DDQN_ONLINE_LEARNING=1        # ← LEARN FROM REALITY
export DDQN_DUAL_AGENT=1
export DISABLE_GATES=0               # All safety gates active
export EPSILON_START=0.15            # Moderate exploration
export EPSILON_END=0.05
export EPSILON_DECAY=0.9995          # Slow decay

# Symbol configuration
export SYMBOL=XAUUSD
export SYMBOL_ID=41
export QTY=0.01
export TIMEFRAME_MINUTES=1

# cTrader credentials
export CTRADER_USERNAME=5179095
export CTRADER_QUOTE_PASSWORD='YourQuotePassword'
export CTRADER_TRADE_PASSWORD='YourTradePassword'
export CTRADER_QUOTE_CONFIG=config/ctrader_quote.cfg
export CTRADER_TRADE_CONFIG=config/ctrader_trade.cfg

# Run with logging
python3 ctrader_ddqn_paper.py 2>&1 | tee live_trading.log

# Monitor continuously:
# - Performance metrics (Sharpe, Sortino)
# - Circuit breaker trips
# - Agent buffer sizes (should grow)
# - Model checkpoints saved
```

---

## FAQ

**Q: Should I ever enable learning in paper mode?**
A: Only for initial debugging of the learning code itself. Never for validation or policy development.

**Q: How do I test if my RL agents are learning correctly?**
A: Use offline training on historical data, or start with micro-positions live (QTY=0.001) with learning enabled.

**Q: What if circuit breakers never trip in paper mode?**
A: They might not - paper fills are too perfect. That's why you test them live with micro-positions first.

**Q: Can I use paper mode to collect experience for offline batch training?**
A: Technically yes, but the experience will have paper biases. Better to use historical bars with realistic fill simulation.

**Q: How long should live learning run before I trust the agents?**
A: Minimum 3 months (500-1000 trades) to see diverse market conditions and verify statistical significance.

---

**Conclusion:** Paper trading is for system validation. Live trading (with micro positions) is for agent training. Never conflate the two.
