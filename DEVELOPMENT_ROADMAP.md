# Development Roadmap - Incremental Enhancement Strategy

**Project:** cTrader Python FIX Trading Bot  
**Original Design:** MQL5 Adaptive Trading System (see MASTER_HANDBOOK.md)
**Platform Migration:** MQL5 → Python + QuickFIX for cTrader
**Date:** 2026-01-09
**Approach:** Test-driven incremental development following handbook priorities

---

## Current Baseline (v0.1.0)

✅ **Working:**
- Dual FIX sessions (QUOTE + TRADE)
- Live market data streaming (BTC/USD)
- M15 bar building
- DDQN framework (no trained model)
- MA crossover fallback strategy
- Basic position management
- Logging infrastructure

⚠️ **Limitations:**
- No MFE/MAE tracking
- No performance metrics
- No reward shaping
- Single agent only
- No regime detection
- Minimal error recovery

---

## Enhancement Phases

### **PHASE 1: Observability & Metrics (Week 1)** ✅ COMPLETE
Make the invisible visible - track everything

#### Update 1.1: Add MFE/MAE Tracking ✅ COMPLETE (commit 24496c5)
**Goal:** Track maximum favorable/adverse excursion during trades
**Files:** `ctrader_ddqn_paper.py`
**Test:** Open trade, verify MFE/MAE updates in logs
**Success:** ✅ MFEMAETracker class implemented, updates every tick
**Bonus:** ✅ Configurable timeframe (M1 for 15x faster testing)

#### Update 1.2: Add Path Recording ✅ COMPLETE (commit b86edc3)
**Goal:** Record M1 OHLC path during entire trade lifecycle
**Files:** `ctrader_ddqn_paper.py`
**Test:** Open/close trade, verify full path saved to JSON
**Success:** ✅ PathRecorder class saves complete trade journey to trades/

#### Update 1.3: Add Performance Dashboard ✅ COMPLETE (commit 5f090b3)
**Goal:** Real-time performance metrics (Sharpe, win rate, etc.)
**Files:** Create `performance_tracker.py`
**Test:** Run for 10 trades, check metrics
**Success:** ✅ PerformanceTracker calculates 15+ metrics in real-time
**Bonus:** ✅ Dashboard prints every 5 trades

#### Update 1.4: Add Trade Export ✅ COMPLETE (commit 8b584c9)
**Goal:** Export trades to CSV for offline analysis
**Files:** Create `trade_exporter.py`
**Test:** Generate CSV, load in Excel/pandas
**Success:** ✅ TradeExporter with 18-column CSV format
**Bonus:** ✅ Auto-exports every 10 trades

---

### **PHASE 2: Reward Shaping (Week 2)** ⏳
Teach the bot what "good" looks like
*Handbook Reference: Section 4.6 - Reward Shaping + Section 2.3 - Path-Centric Design*

#### Update 2.1: Detect Winner-to-Loser (WTL) ⏳
**Goal:** Flag trades that had MFE > 0 but closed at loss (Omega % concept)
**Files:** `ctrader_ddqn_paper.py`
**Handbook Formula:** `if MFE > 0 and final_pnl < 0: WTL = True`
**Test:** Manually close profitable trade at loss, check flag in logs + JSON
**Success:** WTL flag appears in trade record and path JSON

#### Update 2.2: Implement Asymmetric Reward Shaper ⏳
**Goal:** Create reward_shaper.py with component-based rewards
**Files:** Create `reward_shaper.py` (Python port of RewardShaping.mqh)
**Components:**
  - Capture efficiency: `(exit_pnl / MFE) - target_capture`
  - WTL penalty: `-(mfe_normalized × giveback_ratio × time_penalty)`
  - Opportunity cost: `-(potential_mfe × signal_strength × 0.3)`
**Test:** Compare rewards with/without components
**Success:** Asymmetric rewards calculated correctly

#### Update 2.3: Add Self-Optimizing Reward Parameters ⏳
**Goal:** Make reward weights adaptive per instrument (learned not hardcoded)
**Files:** `reward_shaper.py`
**Handbook Principle:** "NO MAGIC NUMBERS - Every parameter is learned"
**Test:** Track reward weight evolution over time
**Success:** Reward weights adapt based on performance feedback

#### Update 2.4: Integrate Shaped Rewards into DDQN ⏳
**Goal:** Use shaped rewards in training loop with replay buffer
**Files:** `ctrader_ddqn_paper.py`
**Test:** Train for 100 episodes, verify reward components logged
**Success:** Shaped rewards influence policy, check correlation with performance

---

### **PHASE 3: Advanced Features (Week 3)** ⏳
Add market intelligence
*Handbook Reference: Section 4.7 - Feature Engineering + Section 3.3 - Regime Detection*

#### Update 3.1: Add Event-Relative Time Features ⏳
**Goal:** Minutes to session close, rollover, news events (not wall-clock time)
**Files:** Create `time_features.py` (Python port of MarketCalendar.mqh)
**Handbook Insight:** Event-relative > absolute time (handles holidays, DST)
**Test:** Check features at known times (e.g., 4:59 PM EST Friday)
**Success:** Features accurate within 1 minute, handle DST transitions

#### Update 3.2: Add Physics-Based Regime Detection ⏳
**Goal:** Damping ratio (ζ) via DSP to detect trending vs mean-reverting
**Files:** Create `regime_detector.py` (Python port of DSPPipeline.mqh + RegimeDetector.mqh)
**Handbook Formula:**
  - Detrend → Bandpass → Hilbert → Envelope → Decay fit: `A(t) = A₀ × e^(-ζωt)`
  - ζ < 0.3: Trending (underdamped)
  - 0.3 ≤ ζ < 0.7: Transitional (critical)
  - ζ ≥ 0.7: Mean-reverting (overdamped)
**Test:** Test on synthetic trending vs ranging data
**Success:** 80%+ regime classification accuracy

#### Update 3.3: Expand Feature Set with Physics + Volatility ⏳
**Goal:** Add advanced features (Roger-Satchell vol, Omega %, physics-based)
**Files:** Create `feature_engine.py`
**Features to add:**
  - Roger-Satchell volatility (handles trending better than Garman-Klass)
  - Omega % (upside potential / downside risk ratio)
  - ATR (Average True Range)
  - RSI (Relative Strength Index)
  - VWAP (Volume-Weighted Average Price)
  - Physics: momentum, acceleration, jerk
**Handbook Principle:** "Instrument-agnostic normalization (log-returns, BPS)"
**Test:** Verify all features calculate without NaN, check normalization
**Success:** 20+ features normalized correctly (mean≈0, std≈1)

#### Update 3.4: Add Feature Tournament Selection ⏳
**Goal:** Survival tournament to eliminate low-IC features
**Files:** `feature_engine.py` (Python port of FeatureTournament.mqh)
**Handbook Method:** Information Coefficient = `correlation(feature[t], returns[t+horizon])`
**Test:** Run tournament on 50 features, keep top 20
**Success:** IC-ranked features, bottom performers eliminated

---

### **PHASE 4: Risk Management (Week 4)** ⏳
Don't lose the account
*Handbook Reference: Section 4.9 - Risk Management + Section 3.2 - VaR Adjustment*

#### Update 4.1: Add VaR-Based Position Sizing ⏳
**Goal:** Dynamic position sizing with multi-factor VaR adjustment
**Files:** Create `position_sizer.py` (Python port of VaREstimator.mqh)
**Handbook Formula:**
```
Adjusted_VaR = Base_VaR × regime_factor × vpin_factor × kurtosis_factor
Position_Size = (Risk_Budget × Equity) / Adjusted_VaR
```
**Test:** Compare sizes in high vs low volatility regimes
**Success:** Position size inversely proportional to adjusted VaR

#### Update 4.2: Add Multi-Signal Circuit Breakers ⏳
**Goal:** Stop trading on Sortino degradation, kurtosis, VPIN thresholds
**Files:** Create `circuit_breaker.py` (Python port of CircuitBreakers.mqh)
**Handbook Breakers:**
  - Sortino ratio < threshold (downside risk spike)
  - Kurtosis > threshold (fat tails detected)
  - VPIN > threshold (toxic order flow)
  - Max consecutive losses
**Test:** Trigger each breaker independently, verify halt
**Success:** Trading stops when any breaker triggers, logs reason

#### Update 4.3: Add Dynamic Risk Budget ⏳
**Goal:** Adaptive daily/weekly loss limits based on account health
**Files:** `circuit_breaker.py`
**Handbook Principle:** Efficiency over avoidance - don't reward not trading
**Features:**
  - Graduated loss limits (tighter when struggling)
  - Recovery mode with reduced sizing
  - Automatic reset on performance improvement
**Test:** Simulate drawdown scenarios
**Success:** Risk budget adapts correctly, prevents catastrophic loss

#### Update 4.4: Add Crisis Correlation Adjustment ⏳
**Goal:** Detect correlation spikes during stress (multi-asset)
**Files:** Create `correlation_monitor.py` (Python port of DynamicCorrelation.mqh)
**Handbook Warning:** "Correlations spike in crises"
**Test:** Simulate crisis scenario with correlated moves
**Success:** Position sizing reduced when correlation exceeds normal range

---

### **PHASE 5: Dual Agent (Month 2)**
Specialize entry vs exit

#### Update 5.1: Refactor to Agent Interface ⏳
**Goal:** Create IAgent interface
**Files:** Create `agents/base_agent.py`
**Test:** Current agent works via interface
**Success:** No regression in functionality

#### Update 5.2: Create Harvester Agent ⏳
**Goal:** Separate exit specialist
**Files:** Create `agents/harvester_agent.py`
**Test:** Harvester produces exit signals
**Success:** Exit signals independent from entry

#### Update 5.3: Add Agent Arena ⏳
**Goal:** Coordinate multiple agents
**Files:** Create `agents/agent_arena.py`
**Test:** Arena allocates between agents
**Success:** Allocation weights adjust by performance

#### Update 5.4: Separate Reward Functions ⏳
**Goal:** Trigger rewarded on runway, Harvester on capture
**Files:** `reward_shaper.py`
**Test:** Check rewards align with agent goals
**Success:** Each agent optimizes its objective

---

### **PHASE 6: Robustness (Month 3)**
Prepare for production

#### Update 6.1: Add Crash Recovery ⏳
**Goal:** Restore state after restart
**Files:** Create `state_manager.py`
**Test:** Kill process, restart, verify state
**Success:** Positions/state restored correctly

#### Update 6.2: Add Journaled Persistence ⏳
**Goal:** Write-ahead log for state changes
**Files:** `state_manager.py`
**Test:** Crash during write, verify recovery
**Success:** No data loss on crash

#### Update 6.3: Add Health Checks ⏳
**Goal:** Monitor FIX connection, data staleness
**Files:** Create `health_monitor.py`
**Test:** Disconnect network, check alerts
**Success:** Alerts trigger on connection loss

#### Update 6.4: Add Automatic Restart ⏳
**Goal:** Restart on fatal errors
**Files:** Update `run.sh`
**Test:** Trigger error, verify restart
**Success:** Bot restarts automatically

---

## Testing Strategy

### **After Each Update:**

```bash
# 1. Unit test (if applicable)
pytest tests/test_<feature>.py

# 2. Integration test
./run.sh  # Start bot

# 3. Verify logs
tail -f logs/python/ctrader_*.log | grep -i <feature>

# 4. Smoke test (5 minutes)
# Watch for errors, check feature working

# 5. Commit if successful
git add .
git commit -m "Update X.Y: <description>"
git tag vX.Y.Z
```

### **Before Phase Completion:**

```bash
# Full regression test
1. Restart bot fresh
2. Run for 24 hours
3. Check all metrics
4. Review all logs for errors
5. Validate performance vs baseline
```

---

## Rollback Plan

```bash
# If update breaks something:
git log --oneline  # Find last good commit
git checkout <commit_hash>
git checkout -b fix-broken-feature
# Fix issue
# Test
git commit
git checkout main
git merge fix-broken-feature
```

---

## Success Metrics

### **Phase 1:**
- ✅ MFE/MAE tracked on 100% of trades
- ✅ Performance metrics accurate
- ✅ No data loss

### **Phase 2:**
- ✅ WTL detection 100% accurate
- ✅ Rewards improve learning speed by 20%+
- ✅ Policy converges faster

### **Phase 3:**
- ✅ 20 features calculated without errors
- ✅ Regime detection 80%+ accurate
- ✅ Features normalized correctly

### **Phase 4:**
- ✅ No account blow-up
- ✅ Circuit breakers trigger correctly
- ✅ Position sizing adapts to volatility

### **Phase 5:**
- ✅ Dual agents outperform single agent
- ✅ Sharpe ratio improves by 30%+
- ✅ Each agent optimizes its goal

### **Phase 6:**
- ✅ 99.9% uptime
- ✅ Zero data loss on crashes
- ✅ Automatic recovery works

---
4 (Phase 1 Complete)
**Branch:** update-1.1-mfe-mae-tracking-v2
**Next Update:** 2.1 - Detect Winner-to-Loser (WTL) Trades
**Target Date:** 2026-01-10
**Estimated Time:** 1-2 hours

**Recent Achievements:**
- ✅ Phase 1 (4 updates) completed 2026-01-09
- ✅ 1,303 total lines of code (877 main + 228 perf + 198 export)
- ✅ M1 timeframe enabled (15x faster testing)
- ✅ Complete observability stack operationale)
**Next Update:** 1.1 - Add MFE/MAE Tracking
**Target Date:** 2026-01-09 (Today)
**Estimated Time:** 2 hours

---

## Notes

- Each update is small (2-4 hours max)
- Test immediately after each change
- Don't move to next update until current one works
- Keep main branch stable (use feature branches)
- Document issues in GitHub issues or KNOWN_ISSUES.md
