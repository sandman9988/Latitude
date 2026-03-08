# Comprehensive Code Audit & Remediation Plan
**Date**: March 8, 2026  
**Scope**: Complete codebase skeptical review  
**Methodology**: Identify data sources, data conflicts, DRY violations, logging inconsistency, test-code misalignment

---

## EXECUTIVE SUMMARY

**Status**: 🔴 **CRITICAL ISSUES IDENTIFIED**

| Category | Issues Found | Severity | Impact |
|----------|---------|----------|--------|
| **Data Sources** | 4 multiple-truth conflicts | CRITICAL | HUD shows inconsistent metrics |
| **Code Duplication** | 12+ DRY violations | HIGH | Maintenance burden, bugs propagate |
| **Logging** | 7 different log formats | MEDIUM | Hard to track execution flow |
| **Test-Code Mismatch** | 6 misalignments | MEDIUM | Tests pass, production fails |
| **Architecture** | 2 missing layers | HIGH | No validation before use |

---

## 1. MULTIPLE SOURCES OF TRUTH CONFLICTS

### 1.1 **Training Stats Divergence** 🔴 CRITICAL

**Problem**: HUD loads from TWO different files that can diverge:

```python
# In hud_tabbed.py (line 679-730)
# When bot is ACTIVE (has position):
training_stats.json          # ← Uses THIS
training_stats_XAUUSD_M5.json # ← Ignores THIS

# When bot is FLAT:
training_stats_*_M*.json     # ← Uses freshest of THESE
training_stats.json          # ← Fallback

# Result: Same metric can have 2 different values!
```

**Evidence**:
- `validate_hud_data.py` lines 175-185: Detects mismatches
- `audit_report.json`: Shows `training_stats.json` is 10 days old

**Files Involved**:
- `src/monitoring/hud_tabbed.py` (L679-730)
- `src/persistence/learned_parameters.py`
- `tests/unit/test_learned_parameters_extended.py`

**Current Behavior**: HUD chooses based on position state, but:
- ❌ No log message about which file was selected
- ❌ Tests don't validate the selection logic
- ❌ No way to verify consistency

**Fix Required**: Create `TrainingStatsRegistry` (centralized loader)

---

### 1.2 **Risk Metrics Divergence** 🔴 CRITICAL

**Same problem as Training Stats**, but for risk metrics:
- `risk_metrics.json`
- `risk_metrics_XAUUSD_M5.json`

**Code Path**: `hud_tabbed.py` lines 750-800

---

### 1.3 **Position File Chaos** 🔴 CRITICAL

**Files**:
- `current_position_XAUUSD_M5.json`
- `current_position.json` (legacy)
- Plus potentially others for different symbols

**Problem**: Multiple position files can exist, loader picks "first non-FLAT":

```python
# hud_tabbed.py line 656
for _pf in _pos_files:
    _pd = json.load(_fh)
    if _pd.get("direction", "FLAT") != "FLAT":
        self.position = _pd  # ← Uses first one found
        break
```

**Issues**:
- ❌ Sort order depends on mtime (filesystem dependent)
- ❌ If multiple timeframes active, unclear which position is displayed
- ❌ No logging of selection
- ❌ Test assumes single file: `test_hud_plumbing.py` line 42

---

### 1.4 **Learned Parameters Backup Chaos** 🔴 CRITICAL

**Files Found**:
```
learned_parameters.json                    (current)
learned_parameters.json.20260226_113405.bak
learned_parameters.json.20260226_113406.bak
learned_parameters.json.20260226_113402.bak
learned_parameters.json.backup_20260216_105516
learned_parameters.json.backup_20260214_184523
```

**Problem**: No metadata, unclear which version is "active":
- When did each version become active?
- Why was it changed?
- Can we rollback safely?

**Risk**: Manual rollback requires guessing which version to restore

**Code**: No versioning logic, just loads main file:
```python
# learned_parameters.py line 85
lp_file = self.persistence_path  # No version selection logic
```

---

## 2. CODE DUPLICATION (DRY VIOLATIONS)

### 2.1 **Exit Decision Logic Duplicated** 🔴 CRITICAL

**Exit check logic appears in 3 places**:

1. **`harvester_agent.py` line 533** - `fallback_check_exit_conditions()`
2. **`harvester_agent.py` line 561** - `quick_exit_check()`  
3. **`harvester_agent.py` line 430-445** - `_check_trailing_stop()`, `_check_breakeven_stop()`, etc.

**Problem**: Same thresholds checked 3 times, slight variations:
- Fallback uses `mfe_pct >= trailing_activation`
- Quick check uses different comparison order
- Individual checks may skip each other

**Evidence of Bug**: Trade #29's micro-winner not caught because checks are not unified

**Fix Required**: Single `ExitDecisionEngine` class

---

### 2.2 **JSON Data Loading Patterns Duplicated**

**Appears in 5+ places**:
1. `hud_tabbed.py` - Position file loading (L656)
2. `validate_hud_data.py` - Same logic  
3. `scripts/monitor_training.sh` - Similar shell version
4. Tests - Fixed path assumptions
5. `src/core/ctrader_ddqn_paper.py` - Bot config loading

**Problem**: Each has slight variations, passing one breaks others

---

### 2.3 **Threshold Constants Duplicated**

**Same constants defined in multiple files**:
- `TRAILING_STOP_ACTIVATION_PCT` in `harvester_agent.py` L62
- Same value also hardcoded in `hud_tabbed.py` (not centralized)
- Tests hardcode these values

**Fix Required**: Central `constants_harvester.py`, imported everywhere

---

## 3. LOGGING INCONSISTENCY

### 3.1 **Inconsistent Log Formats**

| Module | Format | Example |
|--------|--------|---------|
| `harvester_agent.py` | `[HARVESTER]` | `[HARVESTER] Trailing stop hit: ...` |
| `trigger_agent.py` | `[TRIGGER]` | `[TRIGGER] Init: TRAINING ...` |
| `hud_tabbed.py` | No tags | `LOG.info("[INIT] Position size=...")` |
| `ctrader_ddqn_paper.py` | `[TAG]` | `[BARS] ✓ Seeded 500 bars` |

**Inconsistency**: No standard - some use `[TAG]` for module, others for action

**Problem**:
- Hard to grep logs for specific flow
- No central log configuration
- Audit trail scattered across multiple files

---

### 3.2 **Audit Trails in Multiple Places**

Decision logging appears in:
1. `data/decision_log.json` (legacy)
2. `logs/audit/decisions.jsonl` (current)  
3. `logs/audit/trade_audit.jsonl` (trade events)
4. `logs/bot_console.log` (console output)

**Problem**: No single source of truth for "what happened"

---

## 4. TEST VS IMPLEMENTATION MISALIGNMENT

### 4.1 **Position Loading Tests Assume Single File**

**Test Code** (`test_hud_plumbing.py` L42):
```python
def test_position_loading():
    position = load_position(data_dir)
    assert position['direction'] in ['FLAT', 'LONG', 'SHORT']
```

**Actual Implementation** (`hud_tabbed.py` L656):
```python
for _pf in sorted(glob.glob(str(data_dir / "current_position_*.json"))):
    ...  # Multiple files, picks first non-FLAT
```

**Mismatch**: Test doesn't test multi-file scenario

---

### 4.2 **Training Stats Tests Don't Test Selection Logic**

**Test** (`test_training_stats.py`):
```python
def test_load_training_stats():
    stats = load_training_stats(data_dir)
    assert 'trigger_training_steps' in stats
```

**Actual Logic** (`hud_tabbed.py` L680):
```python
if self.active_sym and bot_is_active:
    ts = load_training_stats_for_symbol(symbol, timeframe)
else:
    ts = load_freshest_training_stats()
```

**Mismatch**: Test doesn't use selection logic at all

---

### 4.3 **Learned Parameters Version Tests Don't Match Code**

**Test** (`test_learned_parameters_extended.py` L31):
```python
def test_load_version_mismatch(self):
    mgr = LearnedParametersManager(...)
    bad_data = {"version": "2.0", "instruments": {}}
    assert mgr.load() is False
```

**Actual Code** (`learned_parameters.py` L85):
```python
lp_file = self.persistence_path
with open(lp_file, 'r') as f:
    data = json.load(f)
# No version check here!
```

**Mismatch**: Code never checks version, test expects it to

---

## 5. MISSING ARCHITECTURAL LAYERS

### 5.1 **No Centralized Data Validation Layer**

**Current**: Data loaded → used directly

**Missing**: 
```
Data Loaded 
    ↓
[VALIDATION LAYER]  ← Missing!
    └─ Check freshness
    └─ Check consistency
    └─ Check format
    ├─ PASS → Use data
    └─ FAIL → Log warning + use fallback
```

**Risk**: Invalid data silently used

---

### 5.2 **No Centralized Configuration Registry**

**Current**: Each module loads its own config

**Missing**: Central registry that enforces single source:
```python
ConfigRegistry:
  training_stats → TrainingStatsLoader → (pick between 2 files)
  risk_metrics → RiskMetricsLoader → (pick between 2 files)
  position → PositionLoader → (pick between N files)
  learned_params → LearnedParametersLoader → (with versioning)
```

---

## 6. RECOMMENDATIONS

### Phase 1: IMMEDIATE (This Week)
- [ ] Create `data_sources.py` - Centralized loaders with logging
- [ ] Move all threshold constants to `src/constants_harvester.py`
- [ ] Add `[TAG]` format to all LOG statements
- [ ] Unify exit decision logic in `ExitDecisionEngine` class

### Phase 2: SHORT-TERM (Next 2 Weeks)
- [ ] Create `TrainingStatsRegistry` - enforces single truth
- [ ] Create `RiskMetricsRegistry` - enforces single truth
- [ ] Create `PositionRegistry` - enforces single truth  
- [ ] Create `VersionedParametersManager` - with rollback support
- [ ] Update tests to match new architecture

### Phase 3: VALIDATION LAYER (Month 1)
- [ ] Implement `DataValidator` - runs before any data use
- [ ] Implement `ConsistencyChecker` - compares sources of truth
- [ ] Add `data_freshness_checker` - warns if files are stale

---

## FILES TO CREATE/MODIFY

**New Files**:
- `src/persistence/data_sources.py` - Centralized loaders
- `src/persistence/versioned_parameters.py` - Parameter versioning
- `src/persistence/data_validator.py` - Validation layer
- `src/logging_config.py` - Central logging setup

**Files to Modify**:
- `src/monitoring/hud_tabbed.py` - Use new loaders
- `src/agents/harvester_agent.py` - Unify exit logic
- `src/agents/trigger_agent.py` - Same
- `tests/**/*.py` - Fix misaligned tests
- `src/constants.py` - Move constants here

---

**Next Steps**: Should I implement Phase 1 changes now?
