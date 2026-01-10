# SYSTEM ARCHITECTURE & FLOW CHARTS
## Adaptive Trading Bot - cTrader FIX Implementation

**Version:** 3.0.0  
**Date:** 2026-01-09  
**Status:** ✅ Production Ready

---

## SYSTEM OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE TRADING BOT                         │
│                     cTrader FIX + Dual-Agent RL                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │                           │
        ┌───────────▼──────────┐    ┌──────────▼────────────┐
        │   QUOTE SESSION      │    │   TRADE SESSION       │
        │   Market Data        │    │   Order Execution     │
        │   FIX 4.4            │    │   FIX 4.4             │
        └───────────┬──────────┘    └──────────┬────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   MAIN EVENT LOOP         │
                    │   ctrader_ddqn_paper.py   │
                    └─────────────┬─────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
     ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
     │  Bar Data   │      │   Safety    │      │  Learning   │
     │  Processing │      │   Checks    │      │   Updates   │
     └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   DUAL POLICY DECISION    │
                    │   Trigger + Harvester     │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   TRADE EXECUTION         │
                    │   Risk-Adjusted Size      │
                    └───────────────────────────┘
```

---

## DATA FLOW PIPELINE

```
╔══════════════════════════════════════════════════════════════════════╗
║                         MARKET TICK RECEIVED                          ║
╚══════════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. FIX MESSAGE PARSING                                              │
│    - Parse bid/ask from QuoteCancel message                         │
│    - Extract symbol, timestamp, quotes                              │
│    - Validate message integrity                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. BAR AGGREGATION (ring_buffer.py)                                │
│    - Accumulate ticks into M1 bars                                  │
│    - Calculate OHLC from tick data                                  │
│    - Detect bar close events                                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
                    ┌──────────┴──────────┐
                    │   BAR CLOSE EVENT   │
                    └──────────┬──────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. SAFETY VALIDATION                                                │
│    ┌───────────────────────────────────────────────────────────┐   │
│    │ Circuit Breakers (circuit_breakers.py)                    │   │
│    │  ✓ Check Sortino ratio                                    │   │
│    │  ✓ Check kurtosis (fat tails)                             │   │
│    │  ✓ Check drawdown limits                                  │   │
│    │  ✓ Check consecutive losses                               │   │
│    │  → If tripped: HALT TRADING                               │   │
│    └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
                        ┌──────┴──────┐
                        │ Breakers OK? │
                        └──────┬──────┘
                               │ YES
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. FEATURE ENGINEERING                                              │
│    ┌───────────────────────────────────────────────────────────┐   │
│    │ Core Features (feature_engine.py)                         │   │
│    │  • Price momentum (multi-timeframe)                       │   │
│    │  • Volatility (ATR, Rogers-Satchell)                      │   │
│    │  • Volume analysis                                        │   │
│    │  • Technical indicators (50+ features)                    │   │
│    └───────────────────────────────────────────────────────────┘   │
│    ┌───────────────────────────────────────────────────────────┐   │
│    │ Path Features (path_geometry.py)                          │   │
│    │  • Damping ratio (trending vs mean-reverting)             │   │
│    │  • Natural frequency                                      │   │
│    │  • Momentum physics (jerk, acceleration)                  │   │
│    └───────────────────────────────────────────────────────────┘   │
│    ┌───────────────────────────────────────────────────────────┐   │
│    │ Time Features (event_time_features.py) 🆕                 │   │
│    │  • London/NY/Tokyo session proximity                      │   │
│    │  • Rollover time proximity                                │   │
│    │  • Session overlaps (high liquidity)                      │   │
│    │  • Week/month progress                                    │   │
│    └───────────────────────────────────────────────────────────┘   │
│    ┌───────────────────────────────────────────────────────────┐   │
│    │ Regime Features (regime_detector.py)                      │   │
│    │  • Market regime classification                           │   │
│    │  • Volatility regime                                      │   │
│    │  • Trend strength                                         │   │
│    └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. DUAL POLICY DECISION (dual_policy.py)                           │
│                                                                     │
│    ┌─────────────────────┐          ┌─────────────────────┐        │
│    │  TRIGGER AGENT      │          │  HARVESTER AGENT    │        │
│    │  (Entry Specialist) │          │  (Exit Specialist)  │        │
│    ├─────────────────────┤          ├─────────────────────┤        │
│    │ If FLAT:            │          │ If IN POSITION:     │        │
│    │  • Evaluate entry   │          │  • Evaluate exit    │        │
│    │  • Predict runway   │          │  • Check capture    │        │
│    │  • Check confidence │          │  • Prevent WTL      │        │
│    │  → LONG/SHORT/HOLD  │          │  → EXIT/HOLD        │        │
│    └──────────┬──────────┘          └──────────┬──────────┘        │
│               │                                │                    │
│               └────────────┬───────────────────┘                    │
│                            ▼                                        │
│              ┌─────────────────────────┐                            │
│              │  Combined Decision      │                            │
│              │  (weighted by Sharpe)   │                            │
│              └─────────────┬───────────┘                            │
└────────────────────────────┼────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. RISK MANAGEMENT (var_estimator.py)                              │
│    ┌───────────────────────────────────────────────────────────┐   │
│    │ VaR-Based Position Sizing                                 │   │
│    │  • Calculate Value at Risk                                │   │
│    │  • Apply circuit breaker multiplier 🆕                    │   │
│    │  • Account for friction costs                             │   │
│    │  • Normalize to account equity                            │   │
│    │  → Final Position Size                                    │   │
│    └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 7. TRADE EXECUTION                                                  │
│    - Send FIX order (NewOrderSingle)                                │
│    - Track execution (ExecutionReport)                              │
│    - Record slippage, commissions                                   │
│    - Update position state                                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 8. PATH TRACKING (During Position)                                 │
│    - Update MFE (Maximum Favorable Excursion)                       │
│    - Update MAE (Maximum Adverse Excursion)                         │
│    - Track bar-by-bar P&L                                           │
│    - Monitor for exit signals                                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
                    ┌──────────┴──────────┐
                    │   POSITION CLOSED   │
                    └──────────┬──────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 9. REWARD CALCULATION (reward_shaper.py)                           │
│    ┌───────────────────────────────────────────────────────────┐   │
│    │ Asymmetric Reward Components:                             │   │
│    │  • Outcome: Normalized P&L                                │   │
│    │  • Process: Runway accuracy, capture ratio                │   │
│    │  • Penalties: WTL (winner-to-loser), costs               │   │
│    │  • Bonuses: Early exit efficiency, patience               │   │
│    │  → Total Reward (for learning)                            │   │
│    └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 10. LEARNING UPDATE                                                 │
│     ┌──────────────────────────────────────────────────────────┐   │
│     │ Experience Replay (experience_buffer.py)                 │   │
│     │  • Store (s, a, r, s') with priority                     │   │
│     │  • Sample batch with PER                                 │   │
│     │  • Update TD-error priorities                            │   │
│     └──────────────────────────────────────────────────────────┘   │
│     ┌──────────────────────────────────────────────────────────┐   │
│     │ Network Update (ddqn_network.py)                         │   │
│     │  • Calculate TD-error                                    │   │
│     │  • Backpropagate gradients                               │   │
│     │  • Update online network (Adam)                          │   │
│     │  • Soft-update target network                            │   │
│     └──────────────────────────────────────────────────────────┘   │
│     ┌──────────────────────────────────────────────────────────┐   │
│     │ Overfitting Detection                                    │   │
│     │  • Monitor train/live gap                                │   │
│     │  • Check ensemble disagreement                           │   │
│     │  • Adjust regularization 🆕                              │   │
│     │  • Trigger early stopping if needed                      │   │
│     └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 11. PERSISTENCE (atomic_persistence.py)                            │
│     - Save agent weights                                            │
│     - Save learned parameters 🆕                                    │
│     - Save experience buffer                                        │
│     - CRC32 checksums for integrity                                 │
│     - Atomic writes with backups                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## COMPONENT DEPENDENCY GRAPH

```
                         safe_math.py 🆕
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
         ring_buffer.py  learned_params.py  event_time_features.py 🆕
                │         🆕 │                        │
                ▼             ▼                      ▼
         path_geometry.py     │              feature_engine.py
                │             │                      │
                ▼             │                      ▼
         regime_detector.py   │              circuit_breakers.py 🆕
                │             │                      │
                └─────────────┼──────────────────────┘
                              ▼
                      ddqn_network.py
                              │
                ┌─────────────┼─────────────┐
                │                           │
                ▼                           ▼
         trigger_agent.py          harvester_agent.py
                │                           │
                └─────────────┬─────────────┘
                              ▼
                        dual_policy.py
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
         reward_shaper.py  var_estimator.py  friction_costs.py
                │             │             │
                └─────────────┼─────────────┘
                              ▼
                   ctrader_ddqn_paper.py
                       (MAIN LOOP)
```

---

## DECISION FLOW: BAR CLOSE EVENT

```
┌──────────────────────────────────────┐
│    BAR CLOSE DETECTED               │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│ CHECK CIRCUIT BREAKERS 🆕           │
│  - Sortino ratio OK?                │
│  - Kurtosis OK?                     │
│  - Drawdown OK?                     │
│  - Consecutive losses OK?           │
└─────────────┬────────────────────────┘
              │
              ├─ NO → HALT TRADING
              │       └─ Log reason
              │       └─ Wait cooldown
              │
              ▼ YES
┌──────────────────────────────────────┐
│ CALCULATE FEATURES                  │
│  - Technical indicators             │
│  - Path geometry                    │
│  - Event time 🆕                    │
│  - Regime detection                 │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│ GET POSITION STATE                  │
└─────────────┬────────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
    ▼ FLAT              ▼ IN POSITION
┌───────────┐      ┌────────────┐
│  TRIGGER  │      │ HARVESTER  │
│  AGENT    │      │  AGENT     │
└─────┬─────┘      └──────┬─────┘
      │                   │
      ▼                   ▼
┌───────────┐      ┌────────────┐
│ Predict   │      │ Check exit │
│ runway    │      │ conditions │
│           │      │            │
│ Q(LONG)   │      │ Q(EXIT)    │
│ Q(SHORT)  │      │ Q(HOLD)    │
│ Q(HOLD)   │      │            │
└─────┬─────┘      └──────┬─────┘
      │                   │
      ▼                   ▼
┌───────────┐      ┌────────────┐
│ Confidence│      │ Capture    │
│ > thresh? │      │ efficiency │
└─────┬─────┘      └──────┬─────┘
      │                   │
      ├─ NO → HOLD        ├─ EXIT → Execute
      │                   │
      ▼ YES               ▼ HOLD → Track MFE/MAE
┌───────────┐
│ Calculate │
│ VaR size  │
│  × CB mult│ 🆕
└─────┬─────┘
      │
      ▼
┌───────────┐
│  Execute  │
│   Trade   │
└───────────┘
```

---

## LEARNING CYCLE

```
╔══════════════════════════════════════════════════════════════╗
║                    ONLINE LEARNING LOOP                       ║
╚══════════════════════════════════════════════════════════════╝

    ┌──────────────────────────────────────────┐
    │  1. EXPERIENCE COLLECTION                │
    │     Trade executed → outcome observed    │
    │     Store: (state, action, reward, next) │
    └─────────────────┬────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │  2. PRIORITIZED SAMPLING                 │
    │     Sample batch from buffer (PER)       │
    │     Higher priority = larger TD-error    │
    └─────────────────┬────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │  3. GRADIENT CALCULATION                 │
    │     Q_target = r + γ·max(Q'(s',a'))     │
    │     Loss = (Q_target - Q(s,a))²         │
    │     ∇Loss → gradients                    │
    └─────────────────┬────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │  4. NETWORK UPDATE                       │
    │     Apply gradients (Adam optimizer)     │
    │     Update online network weights        │
    │     θ_target ← τ·θ + (1-τ)·θ_target     │
    └─────────────────┬────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │  5. PARAMETER UPDATE 🆕                  │
    │     Update learned parameters:           │
    │      - Confidence thresholds             │
    │      - Position sizing multipliers       │
    │      - Reward shaping weights            │
    │     Via momentum-based gradient descent  │
    └─────────────────┬────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │  6. OVERFITTING CHECK                    │
    │     Monitor train/live gap               │
    │     Check ensemble disagreement          │
    │     Adjust regularization if needed 🆕   │
    │     Save checkpoint if improvement       │
    └─────────────────┬────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────┐
    │  7. PERSISTENCE                          │
    │     Save weights, params, buffer         │
    │     Atomic writes with CRC32             │
    └──────────────────────────────────────────┘
```

---

## SAFETY LAYERS

```
╔════════════════════════════════════════════════════════════════╗
║                    DEFENSE IN DEPTH                             ║
╚════════════════════════════════════════════════════════════════╝

LAYER 1: Input Validation (safe_math.py) 🆕
├─ All divisions protected (SafeDiv)
├─ All logs protected (SafeLog, SafeSqrt)
├─ NaN/Inf detection
└─ Graceful defaults

LAYER 2: Circuit Breakers (circuit_breakers.py) 🆕
├─ Sortino Ratio < 0.5 → HALT
├─ Kurtosis > 5.0 → HALT
├─ Drawdown > 20% → HALT
├─ 5 consecutive losses → HALT
└─ Auto-reset after cooldown

LAYER 3: Position Sizing Limits
├─ VaR-based calculation
├─ Circuit breaker multiplier
├─ Max position size caps
└─ Minimum viability checks

LAYER 4: Execution Validation
├─ Price reasonability checks
├─ Order book depth validation
├─ Spread width limits
└─ Slippage monitoring

LAYER 5: Overfitting Prevention
├─ Train/live gap monitoring
├─ Ensemble disagreement tracking
├─ Adaptive regularization
└─ Early stopping with rollback

LAYER 6: State Persistence
├─ Atomic writes (no partial saves)
├─ CRC32 checksums
├─ Automatic backups
└─ Version compatibility checks

LAYER 7: FIX Protocol Health
├─ Heartbeat monitoring (60s)
├─ Auto-reconnection (exponential backoff)
├─ Session state tracking
└─ Graceful shutdown handling
```

---

## KEY INNOVATIONS

### 🆕 Phase 1-2 Enhancements

1. **Safe Math Operations**
   - All numerical operations validated
   - No more NaN/Inf crashes
   - Welford's online statistics

2. **Learned Parameters**
   - No hardcoded magic numbers
   - Soft bounds via tanh
   - Per-instrument adaptation
   - Momentum-based updates

3. **Event-Relative Time**
   - Session proximity features
   - Rollover awareness
   - Liquidity period detection
   - Better than raw timestamps

4. **Circuit Breakers**
   - Multi-layer safety system
   - Auto-shutdown on risk escalation
   - Progressive size reduction
   - Smart cooldown periods

5. **Adaptive Regularization**
   - Auto-adjusts L2/dropout
   - Responds to overfit signals
   - Checkpoint management
   - Prevents model degradation

---

## FILE STRUCTURE

```
ctrader_trading_bot/
│
├── CORE INFRASTRUCTURE
│   ├── safe_math.py                    🆕 Defensive numerical operations
│   ├── atomic_persistence.py           Crash-safe file I/O
│   ├── ring_buffer.py                  Fixed-capacity circular buffer
│   └── learned_parameters.py           🆕 Self-optimizing parameters
│
├── FEATURE ENGINEERING
│   ├── feature_engine.py               Technical indicators
│   ├── path_geometry.py                Physics-based features
│   ├── event_time_features.py          🆕 Session-relative time
│   ├── regime_detector.py              Market regime classification
│   └── time_features.py                Legacy time features
│
├── AGENTS & POLICY
│   ├── ddqn_network.py                 Neural network (DDQN)
│   ├── trigger_agent.py                Entry specialist
│   ├── harvester_agent.py              Exit specialist
│   ├── dual_policy.py                  Combined policy
│   └── agent_arena.py                  Multi-agent competition
│
├── LEARNING & MEMORY
│   ├── experience_buffer.py            Prioritized Experience Replay
│   ├── reward_shaper.py                Asymmetric reward calculation
│   ├── generalization_monitor.py       Overfitting detection
│   ├── adaptive_regularization.py      🆕 Auto-adjust L2/dropout
│   ├── early_stopping.py               Checkpoint management
│   └── ensemble_tracker.py             Multi-agent tracking
│
├── RISK & SAFETY
│   ├── var_estimator.py                VaR-based position sizing
│   ├── circuit_breakers.py             🆕 Safety shutdown system
│   ├── friction_costs.py               Cost modeling
│   └── non_repaint_guards.py           Data integrity
│
├── EXECUTION & TRACKING
│   ├── ctrader_ddqn_paper.py           ⭐ MAIN BOT (FIX application)
│   ├── order_book.py                   Order book management
│   ├── performance_tracker.py          Trade statistics
│   └── trade_exporter.py               Trade logging
│
├── MONITORING & DEPLOYMENT
│   ├── health_check.sh                 System health monitoring
│   ├── watchdog.sh                     Auto-restart on failure
│   ├── start_bot_with_hud.sh           Integrated launcher
│   └── hud_tabbed.py                   Tabbed real-time dashboard
│
└── DOCUMENTATION
    ├── MASTER_HANDBOOK.md              Original design handbook
    ├── SYSTEM_ARCHITECTURE.md          🆕 This file
    └── README.md                       Quick start guide
```

---

## METRICS & KPIs

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns (target > 1.0)
- **Sortino Ratio**: Downside risk focus (target > 0.5)
- **Capture Ratio**: Realized P&L / MFE (target > 0.7)
- **WTL Rate**: Winner-to-loser rate (target < 0.1)
- **Max Drawdown**: Peak-to-trough loss (limit < 20%)

### Learning Metrics
- **Train/Live Gap**: Generalization quality (target < 0.2)
- **TD Error**: Learning progress indicator
- **Epsilon**: Exploration rate (1.0 → 0.1)
- **Parameter Staleness**: Last update time (< 24h)

### Safety Metrics
- **Circuit Breaker Status**: Active/tripped count
- **Position Size Multiplier**: Drawdown adjustment (0-1)
- **Kurtosis**: Return distribution (normal ≈ 3)
- **Consecutive Losses**: Streak length (limit = 5)

---

## INTEGRATION CHECKLIST

### ✅ Completed Integrations
- [x] safe_math imported in critical modules
- [x] Circuit breakers wired into main bot
- [x] Event time features added to feature engine
- [x] Learned parameters replace hardcoded values
- [x] Adaptive regularization responds to overfit signals
- [x] Graceful shutdown saves all state

### 📋 Pending Integrations
- [ ] Feature tournament for auto-selection
- [ ] Multi-symbol parameter sets
- [ ] Advanced regime-specific strategies
- [ ] Live performance dashboard enhancements

---

## OPERATIONAL NOTES

### Starting the Bot
```bash
# With HUD
bash start_bot_with_hud.sh

# Production mode
bash run.sh

# Paper trading
PAPER_MODE=1 python3 ctrader_ddqn_paper.py
```

### Monitoring
```bash
# Health check
bash health_check.sh

# Live logs
tail -f logs/bot_$(date +%Y%m%d).log

# Performance metrics
python3 performance_tracker.py --summary
```

### Emergency Procedures
```bash
# Graceful shutdown
pkill -SIGINT -f ctrader_ddqn_paper.py

# Force stop
pkill -9 -f ctrader_ddqn_paper.py

# Reset circuit breakers (use caution!)
# Edit data/circuit_breaker_state.json
# Or wait for cooldown period
```

---

## VERSION HISTORY

### v3.0.0 (2026-01-09) - Current
- ✅ Added safe_math.py for defensive operations
- ✅ Added event_time_features.py for session awareness
- ✅ Added circuit_breakers.py for safety shutdowns
- ✅ Enhanced learned_parameters.py with asset-class scaling
- ✅ Enhanced adaptive_regularization.py with auto-adjustment
- ✅ Integrated all new components into main bot
- ✅ Comprehensive testing and validation

### v2.0.0 (2026-01-08)
- Dual-agent architecture implemented
- Experience replay with PER
- Overfitting detection and prevention
- Production deployment infrastructure

### v1.0.0 (2025-12-XX)
- Initial cTrader FIX integration
- Basic DDQN implementation
- Feature engineering pipeline

---

**END OF SYSTEM ARCHITECTURE**

For detailed component documentation, see individual source files.  
For design philosophy, see MASTER_HANDBOOK.md.  
For quick start, see README.md.
