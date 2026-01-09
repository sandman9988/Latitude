# Development Roadmap - Incremental Enhancement Strategy

**Project:** cTrader Python FIX Trading Bot
**Date:** 2026-01-09
**Approach:** Test-driven incremental development

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

### **PHASE 2: Reward Shaping (Week 2)**
Teach the bot what "good" looks like

#### Update 2.1: Detect Winner-to-Loser (WTL) ⏳
**Goal:** Flag trades that had profit but closed at loss
**Files:** `ctrader_ddqn_paper.py`
**Test:** Manually close profitable trade at loss, check flag
**Success:** WTL flag appears in trade record

#### Update 2.2: Implement WTL Penalty ⏳
**Goal:** Punish giving back profits in reward calculation
**Files:** Create `reward_shaper.py`
**Test:** Compare rewards with/without WTL penalty
**Success:** WTL trades have negative reward component

#### Update 2.3: Add Capture Efficiency Reward ⏳
**Goal:** Reward based on (exit_pnl / MFE) ratio
**Files:** `reward_shaper.py`
**Test:** High capture ratio = higher reward
**Success:** Reward correlates with capture ratio

#### Update 2.4: Integrate Shaped Rewards ⏳
**Goal:** Use shaped rewards in DDQN training
**Files:** `ctrader_ddqn_paper.py`
**Test:** Train for 100 episodes, check reward evolution
**Success:** Shaped rewards influence policy

---

### **PHASE 3: Enhanced Features (Week 3)**
Add market intelligence

#### Update 3.1: Add Event-Relative Time Features ⏳
**Goal:** Minutes to session close, rollover, etc.
**Files:** Create `time_features.py`
**Test:** Check features at known times (e.g., 4:59 PM EST)
**Success:** Features accurate within 1 minute

#### Update 3.2: Add Volatility Regime Detection ⏳
**Goal:** Simple fast/slow volatility ratio
**Files:** Create `regime_detector.py`
**Test:** Test on trending vs ranging data
**Success:** Correctly identifies regime type

#### Update 3.3: Expand Feature Set ⏳
**Goal:** Add 10 more features (RSI, ATR, VWAP, etc.)
**Files:** Create `feature_engine.py`
**Test:** Verify all features calculate without NaN
**Success:** 20 total features, no errors

#### Update 3.4: Feature Normalization ⏳
**Goal:** Z-score normalize all features
**Files:** `feature_engine.py`
**Test:** Check mean=0, std=1 for all features
**Success:** Normalized feature distribution correct

---

### **PHASE 4: Risk Management (Week 4)**
Don't lose the account

#### Update 4.1: Add Position Sizing ⏳
**Goal:** Size positions based on volatility
**Files:** Create `position_sizer.py`
**Test:** Compare sizes in high vs low vol
**Success:** Size inversely proportional to volatility

#### Update 4.2: Add Circuit Breakers ⏳
**Goal:** Stop trading on drawdown/loss streak
**Files:** Create `circuit_breaker.py`
**Test:** Trigger drawdown threshold, verify halt
**Success:** Trading stops when breaker triggers

#### Update 4.3: Add Daily Loss Limit ⏳
**Goal:** Maximum % loss per day
**Files:** `circuit_breaker.py`
**Test:** Lose X%, verify no new trades
**Success:** No trades after daily limit hit

#### Update 4.4: Add Correlation Monitor ⏳
**Goal:** Track correlation with other assets (if multi-asset)
**Files:** Create `correlation_monitor.py`
**Test:** Add second symbol, check correlation
**Success:** Correlation calculated correctly

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
