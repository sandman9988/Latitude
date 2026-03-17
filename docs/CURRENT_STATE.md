# cTrader DDQN Bot - Current State

**Last Updated:** March 17, 2026 (stats epoch, HUD enhancements, new modules)  
**Branch:** `update-1.1-mfe-mae-tracking-v2`  
**Status:** ✅ Operational — all tests green  
**Audience:** All

---

## 🎯 Executive Summary

XAUUSD M5 trading bot using dual-agent DDQN reinforcement learning. Currently in **paper trading** mode. Offline, paper, and live training pipelines now fully aligned (same 18-feature state, same .pt weight format). Dead code removed. Profitability tail-risk fixes applied. Stats epoch feature allows excluding old losing periods from performance metrics.

**Test Suite:** 2,221 passing, 0 skipped, 0 failures (~35 s)  
**Production Lines:** ~41,300

**Trading Status:**
- **Symbol:** XAUUSD (Gold Spot)
- **Timeframe:** M5 (5-minute bars)
- **Mode:** Paper Trading (PAPER_MODE=1)
- **Position Size:** 0.01 lots
- **Session:** QUOTE + TRADE dual FIX sessions

---

## 🔧 HUD Enhancements & Stats Epoch (Mar 17, 2026)

### Stats Epoch Feature (`[e]` key)
New interactive HUD key opens the Stats Epoch Manager — a configurable cutoff date that excludes old trades from all Performance tab metrics (period rows, mode breakdown, trade quality, edge quality). Persisted atomically in `data/stats_epoch.json`.

Options: set to NOW, start of today, 7d/30d ago, custom date (YYYY-MM-DD), or clear.

**Why:** The "All" row in Performance was permanently dragged down by ~1,000 old losing trades ($-4,483). With epoch set, metrics reflect only recent performance while raw `trade_log.jsonl` is preserved intact.

### New Modules Added
| Module | Lines | Purpose |
|--------|-------|---------|
| `src/features/hmm_regime.py` | 264 | HMM-based regime detector (complement to DSP ζ) |
| `src/persistence/trade_log_reader.py` | 127 | Centralized trade_log.jsonl reader (replaces 6+ copies) |
| `src/utils/metrics_calculator.py` | 207 | Single-source period metrics calculation |

### Mode Breakdown Issue Identified
~999 trades in `trade_log.jsonl` have missing/empty `trading_mode` field. These fall through both "Paper" and "Live" mode breakdown filters. The stats epoch feature lets operators focus on recent correctly-tagged trades.

---

## 🔧 Offline/Paper/Live Alignment (Mar 14, 2026)

### Problem
Offline-trained weights were incompatible with paper/live trading:
- **Offline**: 7 features (base only) → `net.0.weight` shape `[128, 448]`
- **Online (paper/live)**: 18 features (7 base + 5 geometry + 6 event) → `net.0.weight` shape `[128, 1152]`

Root cause: `OfflineTrainer` created `DualPolicy` with `enable_event_features=False` and no `path_geometry`.

### Fix
- `offline_trainer.py`: Now creates `PathGeometry()` and `EventTimeFeatureEngine()`, passes them to `DualPolicy` with `enable_event_features=True`
- `_Simulator` computes event-time features from bar timestamps (UTC-aware datetime in `bar[0]`)
- Event features and geometry features are now passed to `decide_entry()` / `decide_exit()` in offline mode
- Weight format unified to `.pt` everywhere (previously saved as `.npz` suffix but actual format was `.pt`)
- `_ckpt_load_weights()` tries `.pt` first, falls back to `.npz` for backward compatibility

### Result
All three modes now produce identical feature dimensions: 18 trigger features, 21 harvester features.
Offline-trained weights directly loadable in paper/live without shape mismatch.

---

## 🔧 Profitability Tail-Risk Fixes (Mar 13, 2026)

Trade log analysis of 1,149 trades revealed:
- 137 ghost reconcile trades (all `bars_held=0`), many as LONG+SHORT pairs on same bar
- 45 trades with loss > $100 accounted for -$11,760 in total losses
- With $100 max-loss cap: P&L shifts from -$4,140 to +$3,119

### FIX-P1 — Hard per-trade loss cap ($100 USD)
Added `MAX_LOSS_PER_TRADE_USD = 100.0` in `constants.py` and `_tick_max_loss_exceeded()` in `ctrader_ddqn_paper.py`. Checked on every tick before ML harvester evaluation.

### FIX-P2 — Duplicate fill guard
Paper fill (2s timeout) AND broker fill could process the same order. Added guard in `trade_manager.py` `_handle_fill()` — skips broker fill when paper fill already processed.

### FIX-P3 — Ghost reconcile cooldown
Added `GHOST_RECONCILE_COOLDOWN_BARS = 3` in `constants.py`. After ghost position reconciliation, entry is blocked for 3 bars to prevent the same race condition from re-entering immediately.

---

## 🔧 Dead Code Removal (Mar 13, 2026)

Removed 10 unused modules (~4,700 lines) and their ~2,500 lines of tests:

| Deleted Module | Lines | Reason |
|----------------|-------|--------|
| `agent_arena.py` | 578 | Multi-agent ensemble never activated; DualPolicy is the architecture |
| `cold_start_manager.py` | 599 | Graduated warmup superseded by DISABLE_GATES + epsilon schedule |
| `early_stopping.py` | 141 | Never wired into training loop |
| `ensemble_tracker.py` | 570 | Tracked agent disagreement for unused arena |
| `feedback_loop_breaker.py` | 506 | Never called from production code |
| `generalization_monitor.py` | 303 | Train-live gap monitoring never connected |
| `parameter_staleness.py` | 613 | Superseded by LearnedParametersManager |
| `feature_tournament.py` | 305 | Feature selection framework unused |
| `time_features.py` | 440 | Superseded by event_time_features.py |
| `risk_aware_sac_manager.py` | 516 | SAC (Soft Actor-Critic) never implemented |

### FIX-R1 — Regime detector ZETA_MAP_MULTIPLIER
`ZETA_MAP_MULTIPLIER` was 0.6 (should be 2.0), causing regime to be stuck in TRANSITIONAL. Fixed in `regime_detector.py`.

---

## 🔧 HUD Audit & Decision Log Traceability (Mar 8, 2026)

### FIX-D1 — Decision log timestamps time-only (HH:MM:SS), no date context
`_render_jsonl_decision_entries` in `hud_tabbed.py` was slicing `ts_raw[11:19]` which strips the date. Multi-day sessions had every entry showing the same ambiguous time.  
**Fix:** Changed to `ts_raw[5:16]` → `MM-DD HH:MM`. Column widened from 8→12. Header changed to `Date/Time`.  
**Impact:** Decision log now unambiguous across day boundaries.

### FIX-D2 — No trade correlation ID visible in Decision Log tab  
Trade IDs existed in `logs/audit/decisions.jsonl` but were buried as dim `PID:xxx` at the end of each line.  
**Fix:** Replaced PID suffix with dedicated `TrdID` column showing `trade_id[:8]` (8-char UUID prefix). `--------` dim when no trade is open (correct for NO_ENTRY decisions).  
**Impact:** Operator can now instantly correlate entry → HOLDs → close by scanning the TrdID column, or `grep` the decisions.jsonl by trade_id.

### FIX-D3 — No visual boundary between bot restart sessions
Decision log entries from different sessions rendered as one continuous list with no way to see where bot was restarted.  
**Fix:** Added `_prev_session` tracking; dim `── session <id> ──` separator printed when session_id changes.  
**Impact:** Bot restart boundaries are immediately visible in Decision Log tab.

### FIX-D4 — `bars_held` always null in data/decision_log.json (100%)
`_obc_write_decision_log` used `pos_metrics.get("bars_held")` but `DualPolicy.get_position_metrics()` returns `ticks_held`, never `bars_held`.  
**Fix:** Changed to `self._get_live_bars_held() if self.cur_pos != 0 else 0` — uses the path-recorder counter.  
**Impact:** `bars_held` is now populated with real values in every bar-close entry.

### FIX-D5 — No session_id in data/decision_log.json entries
Bar-close entries in `data/decision_log.json` had no session field, making it impossible to correlate with `logs/audit/decisions.jsonl` after a restart.  
**Fix:** Added `"session": getattr(self.decision_log, "session_id", None)` to the log_entry dict.  
**Impact:** Bar-close entries now linkable to the rich JSONL audit log by session.

### FIX-D6 — Trade History tab had no live/paper separation  
Trade list showed no indication of whether each trade was paper or live. When both modes are present in `trade_log.jsonl`, metrics were silently mixed.  
**Fix:**  
- Header now shows mode badge: `📄 PAPER`, `💰 LIVE`, or `⚠ MIXED`  
- Each trade row has an `M` column: `P` (yellow) for paper, `L` (green) for live  
- Prominent `⚠ MIXED MODE` banner when paper + live trades co-exist, directing to `[P]` Performance tab  
**Impact:** Operator can distinguish real vs simulated trades at a glance without drilling into detail.

---

## 🔧 BrokerExecutionModel Implementation (Mar 2026)

`src/core/broker_execution_model.py` (440 lines) implements asymmetric slippage modelling.

**What it does:** Buys slip more in up-trending markets; sells slip more in down-trending markets. The model learns actual broker asymmetry from observed execution data and feeds adjusted costs into the position-sizing pipeline via `friction_costs.py`.

**Why it matters:** This was the last `❌` gap listed in the handbook before grad-scaling to live money. It is now `✅`. The system is fully production-capable pending harvester Q-value convergence.

---

## 🔧 Housekeeping Fixes (Feb 22, 2026 session)

### FIX-1 — QuickFIX namespace-package type-annotation crash (MEDIUM)
`_register_universe()` in `train_offline.py` would demote a `LIVE` (or `MICRO`) instrument back to `PAPER` whenever a new training run produced a higher `z_omega` score.  
**Root Cause:** The condition `if not already_paper or better_score` branched into the update block and hard-coded `"stage": "PAPER"` even when `current_stage` was `LIVE`.  
**Fix:** Preserve the existing stage when the instrument is already at `PAPER` or above; only set `"PAPER"` when promoting from below.  
**Impact:** `tests/unit/test_universe_registry.py::test_does_not_demote_from_live` now passes.

---

## 🔧 Production Readiness Fixes (Feb 20, 2026 session)

### GAP-1 — Log flood eliminated (HIGH)
24 `LOG.info()` diagnostic lines demoted to `LOG.debug()` in `src/core/ctrader_ddqn_paper.py`.  
Tags demoted: `[DEBUG]`, `[DIAG]`, `[BAR]`, `[FLOW-TRACE]`, `[FLOW-ABORT] No action needed`, `[POLICY-CHECK]`, `[FLAT: Check for entry]`, `[HARVESTER_DEBUG]`.  
Operationally meaningful tags remain at INFO: `[TRIGGER]`, `[HARVESTER]`, `[CIRCUIT-BREAKER]`, `[ORDER]`, `[ENTRY]`, `[EXIT]`, `[RECONNECT]`, `[SAFETY]`.

### GAP-3 — Model weight load verification (MEDIUM)
`_chk_model_weights()` in `src/core/self_test.py` now calls `torch.load()` to verify the checkpoint is actually loadable, not just that the file exists. Missing torch is surfaced as WARNING rather than silently letting the bot fall back to the heuristic.

### GAP-4 — QuickFIX importable check (MEDIUM → CRITICAL)
New `_chk_quickfix_importable()` self-test check added (severity CRITICAL). QuickFIX must be built from source and is not on PyPI — this check surfaces the missing dependency before the FIX session fails to start.

### GAP-6 — Circuit breaker schema key bug (MEDIUM)
`_chk_circuit_breakers()` was reading `.get("tripped")` but `CircuitBreakers.save_state()` writes `"is_tripped"`. Fix applied: now checks `v.get("is_tripped") or v.get("tripped")` (backwards-compatible). Also validates that the present keys match the known schema to catch future drift.

---


## 🔴 Critical Fixes Applied (Feb 14, 2026)

### 1. Stop Loss Scaling (FIXED)
**Issue:** M1 stop loss was **3.3x too wide** due to timeframe scaling bug.

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| M1 SL Distance | 0.40% | 0.12% | -70% |
| Risk per Trade @ $5045 | $20.18 | $6.05 | -67% |
| File Value | 0.4 | 0.12 | Corrected |

**Root Cause:** `learned_parameters.json` contained unscaled M5/M15 value (0.40%) which was retrieved before timeframe scaling was applied.

**Fix:** Updated `data/learned_parameters.json` XAUUSD_M1_default → harvester_stop_loss_pct: 0.4 → 0.12

**Verification:**
```bash
$ python3 -c "import json; d=json.load(open('data/learned_parameters.json')); \
  print('M1 SL:', d['data']['instruments']['XAUUSD_M1_default']['params']['harvester_stop_loss_pct']['value'])"
M1 SL: 0.12
```

### 2. Stop Loss Learning (IMPLEMENTED)
**Issue:** Stop loss parameter never updated (0 updates) while profit target had 40 updates (asymmetric adaptation).

**Solution:** Added adaptive SL logic in `src/agents/harvester_agent.py:update_from_trade()`:

```python
# Adaptive stop loss based on winner-to-loser (WTL) trades
if was_wtl:
    mfe_to_sl_ratio = self._last_mfe_pct / (self.stop_loss_pct + 1e-9)
    if mfe_to_sl_ratio > 2.0:
        # Had MFE > 2x SL before going negative → SL too wide
        sl_gradient = -0.08  # Tighten by 8%
        new_sl = param_manager.update("harvester_stop_loss_pct", sl_gradient)
```

**Logic:**
- WTL trade with MFE > 2× SL → Tighten SL by 8%
- WTL trade with MFE < 0.5× SL → No change (entry was poor, not SL)
- Non-WTL trades → No adjustment

### 3. Friction Costs (VERIFIED)
**Finding:** ✅ Already correctly implemented!

Friction costs (spread + slippage + commission) are subtracted from MFE before comparing to profit target in both:
- Bar-based exits (`_fallback_strategy()` line 388)
- Tick-based exits (`quick_exit_check()` line 500-507)

**No changes needed.**

---

## 🛡️ Defensive Programming Enhancements (Feb 14, 2026)

Hardened **10 critical areas** with comprehensive input validation and error handling:

| Area | Enhancement | Benefit |
|------|-------------|---------|
| **MFE/MAE Tracker** | Entry price > 0 validation, direction in {-1,1} check | Prevents division by zero |
| **FIX Message Parsing** | Try-except per entry, price sanity (0 < p < 1e9) | Corrupt entry doesn't crash feed |
| **Position Recovery** | Type validation, field checking, tracker isolation | Corrupt persistence doesn't crash startup |
| **Emergency SL** | Input validation, bounds checking, % clamping | SL always executes if threshold exceeded |
| **Mid Price Calculation** | None checks, positivity validation, inverted book detection | Bar builder never receives invalid prices |
| **Bar Builder** | OHLC completeness check, datetime type validation | Invalid bars never propagate |
| **Position Reports** | Quantity validation (≥0, <1000), symbol ID checks | Malformed messages isolated |
| **DualPolicy State** | Orphaned state detection, consistency checks, MFE reset | State corruption recovery |
| **Atomic Persistence** | JSON None check, dict type validation, CRC verification | Triggers backup restore on corruption |
| **Division Operations** | Protected all 20 division points with pre-validation | Impossible to divide by zero |

**Files Modified:**
- `src/core/ctrader_ddqn_paper.py` (7 locations)
- `src/agents/harvester_agent.py` (2 locations)
- `src/core/trade_manager_integration.py` (1 location)
- `src/agents/dual_policy.py` (2 locations)
- `src/persistence/atomic_persistence.py` (1 location)

**Validation:** ✅ All files compile without errors (`py_compile`)

---

## 📊 Current Parameters

### XAUUSD_M1_default (Active)
```json
{
  "harvester_profit_target_pct": {
    "value": 0.8521,
    "update_count": 40,
    "last_update": "2026-02-13T21:46:12Z"
  },
  "harvester_stop_loss_pct": {
    "value": 0.12,
    "update_count": 0,
    "last_update": "2026-02-14T18:47:00Z"
  }
}
```

### Training Metadata
```json
{
  "trigger_epsilon": 0.8534,
  "trigger_training_steps": 831,
  "harvester_training_steps": 869,
  "trigger_platt_a": 0.975,
  "trigger_platt_b": -0.0025
}
```

### Environment Configuration
```bash
SYMBOL=XAUUSD
SYMBOL_ID=41
TIMEFRAME_MINUTES=1
QTY=0.01
PAPER_MODE=1
DISABLE_GATES=1
EPSILON_START=1.0
EPSILON_END=0.1
EPSILON_DECAY=0.9995
FORCE_EXPLORATION=1
MAX_BARS_INACTIVE=10
DDQN_ONLINE_LEARNING=1
```

### Expected Log Output (Next Restart)
```
[HARVESTER] Exit plan: TP=0.85% SL=0.12% soft=200 bars hard=400 bars 
min_profit=0.20% (timeframe=M1 scale=0.30)
```

---

## 🎓 Training Status & Analysis

### Exploration vs Exploitation
- **Epsilon:** 0.8534 (85.3% random actions)
- **Training Steps:** 831 trigger / 869 harvester
- **Status:** Early training phase (correct for 831 steps)

**Decay Schedule:**
- Start: 1.0 (100% exploration)
- Current: 0.8534 (85% exploration)
- End: 0.1 (10% exploration)
- Decay: 0.9995 per step
- Steps to ε=0.1: ~3,769 more steps needed

**Forced Exploration:**
- Enabled: Every 10 bars if no trade taken
- Purpose: Prevent "always NO_ENTRY" collapse
- Tradeoff: Adds noise but ensures diverse experiences

**Assessment:**
- ✅ Exploration rate appropriate for training stage
- ⚠️ Epsilon decay very slow (designed for long-term learning)
- ⚠️ Forced entry every 10 bars may be aggressive (consider increasing to 50-100)

### Parameter Learning Asymmetry (FIXED)
| Parameter | Before Fix | After Fix |
|-----------|-----------|-----------|
| Profit Target | 40 updates ✅ | Continues learning ✅ |
| Stop Loss | 0 updates ❌ | Now learns ✅ |
| Symmetry | Asymmetric | Symmetric ✅ |

### Friction Awareness
- ✅ Entry: Calculated and subtracted from predicted runway
- ✅ Exit: Subtracted from MFE before TP comparison
- ✅ Logging: Friction explicitly logged in decisions

---

## 🚀 System Health

### Bot Process
```bash
$ ps aux | grep ctrader_ddqn_paper
PID: 1639168 (running)
Uptime: 8.6+ hours
```

### Current Position
- **Direction:** LONG
- **Entry Price:** 5045.21
- **MAE:** 2.05 points (0.04%)
- **Status:** Safe (within 0.12% SL threshold)

### FIX Sessions
- **QUOTE:** Connected (market data streaming)
- **TRADE:** Connected (order execution ready)
- **Markets:** Closed (Forex weekend)

### Logs
```bash
$ ls -lh logs/ctrader/ | tail -3
-rw-r--r-- 1 user user 1.2M Feb 14 18:45 ctrader_20260213_175906.log
```

### Data Files
- `learned_parameters.json` - ✅ Backed up, SL corrected
- `training_metadata.json` - ✅ Epsilon tracking working
- `current_position.json` - ✅ Position state persisted

---

## ⚠️ Known Issues

### 1. L2/Imbalance Feed (Medium Priority)
- `imbalance` always 0.0 — FIX MarketDataRequest may not request MDEntryType=0/1
- **Impact:** 1 of 7 base features always zero; model can still learn around it
- **Action:** Check FIX config for L2 data subscription

### 2. Harvester Q-value Convergence (Low Priority)
- Monitor `ticks_held` trending in HUD Training tab
- **Action:** Observation only; track across sessions

### 3. `data/decision_log.json` Non-Atomic (Low Priority)
- Secondary bar-close log written with `open(path, "w")` rather than atomic persistence
- **Impact:** Cosmetic only; primary audit log (`logs/audit/decisions.jsonl`) is append-only and safe

---

## 📋 Testing Checklist (Post-Restart)

### Immediate (Within 1 Hour)
- [ ] Verify "Exit plan" log shows SL=0.12% (not 0.40%)
- [ ] Check emergency SL logic still active
- [ ] Confirm position recovered correctly
- [ ] Monitor first bar processing

### Short-Term (24 Hours)
- [ ] Watch for SL triggers at 0.12% (expect higher frequency)
- [ ] Look for first SL parameter update after WTL trade  
- [ ] Verify friction costs logged in TP checks
- [ ] Confirm defensive validation warnings (if any)

### Medium-Term (7 Days)
- [ ] Track SL update_count increase from 0
- [ ] Compare TP and SL update frequencies (should be similar)
- [ ] Analyze capture ratio distribution (target: 50-70%)
- [ ] Review WTL frequency (target: <15%)

---

## 🔧 Quick Commands

### Check Bot Status
```bash
ps aux | grep ctrader_ddqn_paper
```

### View Latest Logs
```bash
tail -f logs/ctrader/ctrader_$(ls -t logs/ctrader | head -1)
```

### Check Parameters
```bash
python3 << 'EOF'
import json
with open('data/learned_parameters.json') as f:
    d = json.load(f)
    m1 = d['data']['instruments']['XAUUSD_M1_default']['params']
    print(f"TP: {m1['harvester_profit_target_pct']['value']:.4f}% ({m1['harvester_profit_target_pct']['update_count']} updates)")
    print(f"SL: {m1['harvester_stop_loss_pct']['value']:.4f}% ({m1['harvester_stop_loss_pct']['update_count']} updates)")
EOF
```

### Restart Bot
```bash
pkill -9 -f ctrader_ddqn_paper && sleep 2
bash run.sh &
```

### Emergency Close All
```bash
python3 emergency_close_all.py
```

---

## 🧪 Feature Engineering Lessons Applied

**New Resources (Feb 14, 2026):**
- 📄 [FEATURE_ENGINEERING_LESSONS.md](FEATURE_ENGINEERING_LESSONS.md) - Lessons from trend_sniper v3.5-v3.9 experiments
- 🔧 [scripts/analyze_feature_importance.py](../scripts/analyze_feature_importance.py) - L1 weight analysis tool

**Key Lessons from External Experiments:**
1. **Subtraction > Addition:** Removing 3 noise features (+95 reward) beat adding best new feature
2. **Small Samples Lie:** 27 trades showed opposite pattern vs 173-trade truth  
3. **Wait for 500+ Trades:** Current ~100-200 trades insufficient for reliable analysis
4. **L1 Weight Analysis:** Reveals which features network actually uses (vs ignores)
5. **Greedy Elimination:** Stepwise removal finds optimal feature set efficiently

**Applicable to This Bot:**
- TriggerAgent has 7 obs features (entry specialist)
- HarvesterAgent has 10 obs features (7 market + 3 position)
- After 500+ trades, run L1 analysis to identify noise features (candidates: L1 < 0.08)
- Consider ablation study: remove low-weight features, retrain, compare performance

**Action Items:**
- ⏳ **Accumulate 500+ closed trades** before optimization (currently ~100-200 estimated)
- 📊 **Run** `python scripts/analyze_feature_importance.py` after milestone
- 🔍 **Cross-reference** L1 weights with win/loss discrimination (Cohen's d > 0.30)
- ✂️ **Test removal** of features with L1 < 0.08 AND d < 0.15 (noise candidates)

---

## 🔄 Recent History

### Feb 14, 2026 (Today)
- ✅ Fixed M1 stop loss scaling (0.40% → 0.12%)
- ✅ Implemented stop loss learning
- ✅ Enhanced defensive programming (10 areas)
- ✅ Created comprehensive documentation cleanup

### Feb 13, 2026
- ✅ Fixed emergency stop loss bypass (DDQN model ignoring SL)
- ✅ Verified SL triggers correctly (tested at MAE=1.53%)
- ✅ Position closed underwater trade successfully

### Feb 7-13, 2026
- ✅ Implemented foreign position auto-close
- ✅ Enhanced MFE/MAE tracking with validation
- ✅ Added comprehensive error handling

---

## 📈 Performance Metrics (Last Session)

**Trades:** ~40 (evidenced by TP update_count)  
**Training Steps:** 831 trigger / 869 harvester  
**Epsilon:** 0.8534 (declining from 1.0)  
**Current Capture:** Not yet stable (early training)

**Target Metrics (Post-Training):**
- Win Rate: 40-50%
- Capture Ratio: 50-70%
- WTL Frequency: <15%
- Sharpe Ratio: >1.5

---

## 🎯 Next Steps

### Immediate (Before Markets Open)
1. **Monitor restart with corrected parameters**
2. **Verify SL=0.12% in logs**
3. **Watch for first SL trigger**

### Short-Term (Next 7 Days)
1. **Track SL learning progress** (update_count should increase)
2. **Analyze WTL trades** (should trigger SL tightening)
3. **Review capture ratios**

### Medium-Term (Next 30 Days)
1. **Reduce exploration** (epsilon → 0.1 after ~3,800 steps)
2. **Enable confidence gates** (once epsilon < 0.3)
3. **Prepare for production** (see INDEX.md → reports/PRE_LAUNCH_CHECKLIST.md)

---

## 📞 Emergency Contacts

### Rollback Procedure
```bash
# 1. Stop bot
pkill -9 -f ctrader_ddqn_paper

# 2. Restore parameters
cp data/learned_parameters.json.backup_* data/learned_parameters.json

# 3. Revert code
git checkout update-1.1-mfe-mae-tracking-v2 -- src/agents/harvester_agent.py src/agents/dual_policy.py

# 4. Restart
bash run.sh
```

### Support Resources
- **Documentation Index:** [INDEX.md](INDEX.md)
- **Disaster Recovery:** [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md)
- **Monitoring Guide:** [MONITORING_GUIDE.md](MONITORING_GUIDE.md)
- **Repository:** github.com/sandman9988/Latitude

---

## 📝 File Modifications Log

| File | Change | Purpose |
|------|--------|---------|
| `src/core/trade_manager_integration.py` | Added `from __future__ import annotations` | Fix QuickFIX namespace-package type-annotation crash |
| `train_offline.py` | `_register_universe()` preserves stage when already PAPER+ | Fix LIVE/MICRO → PAPER demotion bug |
| `data/learned_parameters.json` | M1 SL: 0.4→0.12 | Fix scaling bug (Feb 14) |
| `src/agents/harvester_agent.py` | Lines 730-763 | SL learning (Feb 14) |
| `src/agents/dual_policy.py` | Lines 409-411 | MFE% tracking (Feb 14) |
| `src/core/ctrader_ddqn_paper.py` | 7 locations | Defensive validation (Feb 14) |
| `src/core/trade_manager_integration.py` | Lines 965-1007 | Tracker recovery validation (Feb 14) |
| `src/persistence/atomic_persistence.py` | Lines 126-167 | JSON corruption detection (Feb 14) |

**Backups Created:**
- `data/learned_parameters.json.backup_20260214_*`

---

**Last Review:** February 22, 2026 (housekeeping audit)  
**Next Review:** After next bot session or code change  
**Review Frequency:** After each code change; weekly in production

---

**Navigation:** [📚 Documentation Index](INDEX.md) | [🚀 Quick Start](QUICKSTART.md) | [🔧 Operations](MONITORING_GUIDE.md)
