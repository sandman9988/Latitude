# REMEDIATION ACTION PLAN - Phase Implementation Guide

**Created**: March 8, 2026  
**Status**: 🔴 CRITICAL - Awaiting implementation approval  
**Scope**: Complete architecture refactoring over 3 phases

---

## PHASE 1: IMMEDIATE (Implement This Week)

### Goal: Stop the bleeding
Fix the most critical issues that cause silent data inconsistencies and make future bugs harder to find.

### 1.1 ✅ CREATE CENTRALIZED DATA SOURCE REGISTRY
- **File**: `src/persistence/data_sources.py` ← **CREATED** 
- **What it does**: Replaces all ad-hoc `json.load()` and `glob()` patterns with single, tested interface
- **Classes**:
  - `DataSourceRegistry` - Main abstraction layer  
  - Factory functions: `load_position()`, `load_training_stats()`, `load_risk_metrics()`
  
**Changes Required**:
- [ ] Import `DataSourceRegistry` in `src/monitoring/hud_tabbed.py`
- [ ] Replace lines 649-730 in hud_tabbed.py with registry calls
- [ ] Add logging for which file was actually selected
- [ ] Update tests in `tests/unit/test_hud_plumbing.py` to validate selection logic

**Testing**: 
- [ ] Test with multiple position files present
- [ ] Test with multiple training_stats files (different mtimes)
- [ ] Verify logging shows which file was chosen
- [ ] Run existing HUD tests to ensure behavior unchanged

---

### 1.2 ✅ CREATE UNIFIED EXIT DECISION ENGINE  
- **File**: `src/agents/exit_decision_engine.py` ← **CREATED**
- **What it does**: Consolidates ALL exit logic into single priority-ordered decision tree
- **Classes**:
  - `ExitDecisionEngine` - Single entry point for all exit decisions
  - `ExitSignal` - Standardized exit result

**Changes Required**:
- [ ] Replace `quick_exit_check()` in harvester_agent.py with `engine.evaluate()`
- [ ] Replace `bar_exit_check()` in harvester_agent.py with `engine.evaluate()`
- [ ] Remove individual `_check_*` methods once consolidated
- [ ] Verify exit behavior matches before/after with logs

**Testing**:
- [ ] Test each exit condition in isolation
- [ ] Test priority order (stop loss always first, etc)
- [ ] Verify trade #29 scenario (small winner reversing) catches with micro-winner protection
- [ ] Compare logs before/after to ensure same decisions

---

### 1.3 CENTRALIZE THRESHOLD CONSTANTS
- **Target**: `src/constants.py` (already exists, needs expansion)
- **Constants to move**:
  - `TRAILING_STOP_ACTIVATION_PCT` from harvester_agent.py L61
  - `TRAILING_STOP_DISTANCE_PCT` from harvester_agent.py L62
  - `MICRO_WINNER_MFE_THRESHOLD_PCT` from harvester_agent.py L76
  - `MICRO_WINNER_GIVEBACK_PCT` from harvester_agent.py L77
  - `BREAKEVEN_STOP_PCT` (add new)

**Changes Required**:
- [ ] Add these constants to `src/constants.py` section for Harvester
- [ ] Import in harvester_agent.py from constants.py
- [ ] Import in exit_decision_engine.py from constants.py  
- [ ] Update tests to import from constants.py instead of hardcoding
- [ ] Add comments explaining each constant's purpose

**Testing**:
- [ ] Verify imports work across all modules
- [ ] Run full test suite to ensure no silent imports failing

---

### 1.4 STANDARDIZE LOGGING FORMAT
- **Target**: All `.py` files in `src/`
- **Standard Format**: `[MODULE_TAG] Action description: details`
  
**Examples**:
```python
LOG.info("[DATASRC] Loaded position from per-bot file: current_position_XAUUSD_M5.json")
LOG.info("[EXIT_ENGINE] TRAILING STOP triggered: MFE=0.35%, giveback=0.11% >= 0.10%")
LOG.info("[HARVESTER] Micro-winner exit: MFE=0.065%, current=-0.015%, giveback=125% of MFE")
```

**Modules**:
- `[DATASRC]` - data_sources.py new module
- `[EXIT_ENGINE]` - exit_decision_engine.py new module
- `[HARVESTER]` - harvester_agent.py (already has this)
- `[TRIGGER]` - trigger_agent.py (already has this)
- `[HUD]` - hud_tabbed.py (needs updating)
- `[BOT]` - ctrader_ddqn_paper.py (needs updating)

**Changes Required**:
- [ ] Audit hud_tabbed.py LOG calls, add [HUD] tag to lines without tags
- [ ] Audit ctrader_ddqn_paper.py LOG calls, add [BOT] tag
- [ ] Run grep to find any LOG calls without tags: `grep -n "LOG\.(info|warning|error)" src/**/*.py | grep -v "\["`

**Testing**:
- [ ] Check console output has consistent formatting
- [ ] Verify grep finds all LOG statements have tags

---

## PHASE 2: SHORT-TERM (Implement Next 2-3 Weeks)

### Goal: Enforce single sources of truth throughout

### 2.1 CREATE TRAINING STATS REGISTRY
- **File**: `src/persistence/training_stats_registry.py` (new)
- **Extends**: DataSourceRegistry with ML-specific logic
- **Features**:
  - Version tracking (timestamp, loss curves)
  - Freshness validation (warns if older than N hours)
  - Multi-bot consistency checking
  
---

### 2.2 CREATE RISK METRICS REGISTRY
- **File**: `src/persistence/risk_metrics_registry.py` (new)
- **Similar to**: TrainingStatsRegistry
- **Features**:
  - VPIN consistency checking
  - Order book depth validation
  - Spread/slippage tracking

---

### 2.3 CREATE VERSIONED PARAMETERS MANAGER
- **File**: `src/persistence/versioned_parameters_manager.py` (new)
- **Fixes**: The 6+ backup file chaos
- **Features**:
  - Version metadata (timestamp, why changed)
  - Rollback support (safe restore of previous version)
  - CRC32 validation across versions
  - Automatic cleanup of stale backups

**Structure**:
```
learned_parameters.json (ACTIVE)
learned_parameters_versions/
  v001_20260214_184523.json.bak (reason: "initial_training")
  v002_20260216_105516.json.bak (reason: "architecture_update")
  v003_20260226_113402.json.bak (reason: "feature_expansion")
```

---

### 2.4 UPDATE TESTS TO MATCH NEW ARCHITECTURE
- **Files to update**:
  - `tests/unit/test_hud_plumbing.py` - Add multi-file position tests
  - `tests/unit/test_training_stats.py` - Add selection logic tests
  - `tests/unit/test_learned_parameters_extended.py` - Remove misaligned version tests
  - Add new file: `tests/unit/test_exit_decision_engine.py`
  - Add new file: `tests/unit/test_data_sources_registry.py`

---

## PHASE 3: VALIDATION LAYER (Implement Month 1)

### Goal: No invalid data silently used

### 3.1 CREATE DATA VALIDATOR
- **File**: `src/persistence/data_validator.py` (new)
- **Checks**:
  - Position file: required fields (direction, symbol, entry_price)
  - Training stats: numeric fields are >= 0, not NaN
  - Risk metrics: timestamps are recent (< 1 hour old)
  - Learned parameters: version field matches expected schema

### 3.2 CREATE CONSISTENCY CHECKER
- **File**: `src/monitoring/consistency_checker.py` (new)
- **Compares**:
  - `current_position.json` vs `current_position_SYMBOL_MTF.json` (should be identical)
  - `training_stats.json` vs `training_stats_SYMBOL_MTF.json` (should be identical)
  - Sum of trades in trade_log.jsonl vs cumulative PnL in performance_snapshot.json
  - Per-symbol metrics sum to totals

### 3.3 CREATE AUDIT LOGGER
- **File**: `src/monitoring/audit_logger.py` (new)
- **Features**:
  - Central log file: `logs/audit/full_audit.jsonl`
  - Structured format: timestamp, module, action, details
  - Searchable by module, action, or timerange

---

## VERIFICATION CHECKLIST

### Before Phase 1 Completion:
- [ ] All data loads go through DataSourceRegistry
- [ ] All exit decisions go through ExitDecisionEngine
- [ ] All LOG statements have [TAG] prefix
- [ ] All thresholds in constants.py, imported everywhere
- [ ] Trade #29 scenario is caught by micro-winner protection
- [ ] HUD shows which file was loaded (in debug logs)

### Before Phase 2 Completion:
- [ ] No backup learned_parameters files scattered
- [ ] Version metadata attached to each parameter save
- [ ] Can rollback to previous version safely
- [ ] Tests match implementation (not the reverse)

### Before Phase 3 Completion:
- [ ] Validator runs before any data is used
- [ ] Consistency checker runs on every HUD refresh
- [ ] Alerts if files are older than expected
- [ ] Audit log has complete trace of all data loading

---

## IMPACT ASSESSMENT

### Code Quality Improvements:
- **Reduced Duplication**: ~300 lines of identical exit logic → 100 lines in single class
- **Increased Testability**: Factories and registries easier to mock
- **Better Observability**: Every data load is logged with which file
- **Easier Debugging**: "What data did HUD use?" now easily answered

### Reduced Bugs:
- **Data Conflicts**: Multiple truth sources eliminated
- **Silent Failures**: Invalid data checked before use
- **Lost Changes**: Version tracking prevents accidental rollback
- **Test-Code Mismatch**: Tests updated to match reality

### Performance Impact:
- **Negligible**: DataSourceRegistry uses glob() same as before, just once per refresh
- **Minor Improvement**: ExitDecisionEngine might be slightly faster (single method vs N method calls)

---

## ROLLBACK PLAN

If issues arise:
- Phase 1 changes are **non-breaking**: Old code paths still work, new paths are additive
- Data loading: Revert to manual `json.load()` if registry has issues (single import to remove)
- Exit logic: Revert harvester_agent.py if exit_decision_engine has bugs (single replace)

---

## SUCCESS CRITERIA

Phase 1 COMPLETE when:
1. ✅ DataSourceRegistry created and tested
2. ✅ ExitDecisionEngine created and tested  
3. ✅ All constants centralized
4. ✅ All LOG statements have tags
5. ✅ Zero new bugs in HUD refresh
6. ✅ Trade #29 reversal scenario prevented

Phase 2 COMPLETE when:
1. ✅ Versioned parameter manager deployed
2. ✅ No stale backup files
3. ✅ All tests updated to match code
4. ✅ Safe rollback tested

Phase 3 COMPLETE when:
1. ✅ Validator runs before all data use
2. ✅ Consistency checker detects all mismatches
3. ✅ Audit trail complete and searchable
4. ✅ Zero silent failures

---

## NEXT IMMEDIATE STEPS

**TODAY**:
- [ ] Create todo list for Phase 1 tasks
- [ ] Review this plan with team
- [ ] Assign owners to each subsection

**THIS WEEK**:
- [ ] Integrate DataSourceRegistry into hud_tabbed.py
- [ ] Integrate ExitDecisionEngine into harvester_agent.py
- [ ] Add centralized constants
- [ ] Update logging format

**NEXT WEEK**:
- [ ] Update Phase 1 tests
- [ ] Run full test suite
- [ ] Deploy Phase 1 to staging bot
- [ ] Verify same trades executed as before

---

## DOCUMENTS CREATED

- ✅ `COMPREHENSIVE_CODE_AUDIT.md` - Full problem analysis
- ✅ `src/persistence/data_sources.py` - Registry implementation  
- ✅ `src/agents/exit_decision_engine.py` - Exit consolidation
- ✅ `REMEDIATION_ACTION_PLAN.md` - This document

---

## QUESTIONS FOR REVIEW

1. **Phase 1 Timing**: Can we spend this week on refactoring, or do we need to trade live?
2. **Testing Strategy**: Should we test locally first, then staging, then live?
3. **Rollback Risk**: Is reverting to old code in production acceptable if Phase 1 breaks something?
4. **Data Migration**: For the learned_parameters versioning, how far back should we keep versions?
5. **Logging Volume**: Is logging every data source selection acceptable, or should it be debug-level only?

