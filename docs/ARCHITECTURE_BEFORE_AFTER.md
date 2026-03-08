# DATA ARCHITECTURE: BEFORE vs AFTER
**Visual Guide to Remediation**

---

## BEFORE: Current State (Chaotic)

```
HUD REFRESH CYCLE
├─ hud_tabbed.py:649
│  └─ Load position
│     └─ json.load(glob("current_position_*.json")[0])  ← DIRECT JSON LOAD
│        └─ Could pick any file based on mtime
│
├─ hud_tabbed.py:679
│  └─ Load training_stats.json
│     └─ json.load("training_stats.json")
│        └─ Get: trigger_training_steps=500, loss=0.45
│
├─ hud_tabbed.py:693 (OVERWRITES!)
│  └─ IF bot_is_active:
│     └─ Load training_stats_XAUUSD_M5.json
│        └─ OVERWRITES previous data!
│           ├─ Get: trigger_training_steps=750, loss=0.32
│           └─ ⚠️ DIFFERENT VALUES - Silent conflict!
│
├─ hud_tabbed.py:698
│  └─ Load risk_metrics.json
│     └─ Get: vpin=0.65, spread=1.2
│
└─ hud_tabbed.py:715 (OVERWRITES!)
   └─ IF bot_is_active:
      └─ Load risk_metrics_XAUUSD_M5.json
         └─ OVERWRITES with: vpin=0.68, spread=1.1
            └─ ⚠️ DIFFERENT VALUES - Silent conflict!

⚠️ RESULTS:
   - Different files loaded depending on position state
   - No log of which was selected
   - Tests don't cover multi-file scenarios
   - Same metric displayed has 2 possible values
```

---

## BEFORE: Exit Logic (Scattered)

```
harvester_agent.py::quick_exit_check()
├─ Check stop loss
├─ Check profit target
├─ Check micro-winner exit       ← NEW CODE (just added)
├─ Check trailing stop
├─ Check breakeven stop
├─ Check capture decay
└─ Check time stops

harvester_agent.py::bar_exit_check()
├─ Different order of checks!
├─ Does NOT check micro-winner? ← BUG?
├─ Similar but NOT identical to quick_exit_check()
└─ ⚠️ Same condition checked 2 different ways

Individual methods (scattered):
├─ _check_trailing_stop() L459
├─ _check_profit_target() L427
├─ _check_micro_winner_exit() L432
└─ ... others ...

⚠️ RESULTS:
   - Inconsistent exit order potentially misses trades
   - Micro-winner protection might not always trigger
   - Test doesn't verify all paths work
   - Trade #29: Small winners reverse before micro-winner check runs
```

---

## AFTER: Data Loading (Unified)

```
HUD REFRESH CYCLE
├─ DataSourceRegistry.load_position()
│  ├─ Priority 1: current_position_XAUUSD_M5.json (per-bot, if exists)
│  ├─ Priority 2: glob("current_position_*.json") sorted by mtime (any bot, if exists)
│  ├─ Priority 3: current_position.json (legacy fallback)
│  └─ LOG: "[DATASRC] Loaded position from per-bot file: current_position_XAUUSD_M5.json"
│     └─ ✅ SINGLE FILE SELECTED, ALWAYS LOGGED
│
├─ DataSourceRegistry.load_training_stats(active_sym, active_tf, bot_is_active)
│  ├─ Priority 1: training_stats_XAUUSD_M5.json IF bot_is_active (per-bot)
│  ├─ Priority 2: training_stats_*_M*.json freshest IF bot_is_idle (any bot)
│  ├─ Priority 3: training_stats.json (legacy fallback)
│  └─ LOG: "[DATASRC] Loaded training_stats from active bot file: training_stats_XAUUSD_M5.json"
│     └─ ✅ CLEAR PRIORITY ORDER, LOGGED WHICH WAS PICKED
│
├─ DataSourceRegistry.load_risk_metrics(active_sym, active_tf, bot_is_active)
│  ├─ Same priority logic as training_stats
│  └─ LOG: "[DATASRC] Loaded risk_metrics from freshest idle bot file: risk_metrics_BTCUSD_M60.json"
│     └─ ✅ ONLY ONE FILE USED, CONSUMERS KNOW WHICH
│
└─ registry.get_loaded_source("training_stats")
   └─ → (Path("data/training_stats_XAUUSD_M5.json"), {...data...})
      └─ ✅ CAN AUDIT WHICH FILE WAS USED

✅ RESULTS:
   - Single file selected per data reload
   - Priority order explicit and documented
   - Every load is logged with file path
   - Can audit which file was used for which metric
   - Tests can verify priority order
```

---

## AFTER: Exit Logic (Consolidated)

```
harvester_agent.py::REMOVED quick_exit_check(), REMOVED bar_exit_check()

ExitDecisionEngine.evaluate(mae, mfe, profit, ...)
├─ PRIORITY 1: Stop Loss (Mae >= 5%)
│  └─ → EXIT | REASON: "STOP_LOSS"
│
├─ PRIORITY 2: Profit Target (net_profit >= 10%)
│  └─ → EXIT | REASON: "PROFIT_TARGET"
│
├─ PRIORITY 3: Micro-Winner (mfe > 0.05% but giving back > 30% of mfe)
│  └─ │  CASE STUDY: Trade #29
│  └─ │  - Entry: 5313.21 SHORT
│  └─ │  - MFE: +52.27 pts = +0.0984%
│  └─ │  - Current: -0.0149% (reversed)
│  └─ │  - Giveback: 113% of MFE
│  └─ │  → EXIT with reason: "MICRO_WINNER"
│  └─ └─ ✅ CATCHES W2L BEFORE BLEEDING
│
├─ PRIORITY 4: Trailing Stop (mfe >= 35% AND giveback >= 10%)
│  └─ → EXIT | REASON: "TRAILING_STOP"
│
├─ PRIORITY 5: Breakeven Stop (0% < profit < 0.4%)
│  └─ → EXIT | REASON: "BREAKEVEN_STOP"
│
├─ PRIORITY 6: Capture Decay (runway decayed > 2.5% AND profit > 0)
│  └─ → EXIT | REASON: "CAPTURE_DECAY"
│
├─ PRIORITY 7: Time Stop (ticks_held >= max_hold_ticks)
│  └─ → EXIT | REASON: "TIME_STOP"
│
└─ NO MATCH: → HOLD | REASON: "HOLD"

INTEGRATION:
harvester_agent.py:
  engine = ExitDecisionEngine(config)
  signal = engine.evaluate(mae_pct, mfe_pct, current_profit_pct, ...)
  if signal.should_exit:
    LOG.info("[HARVESTER] %s triggered: %s", signal.reason, details)
    place_exit_order(signal.exit_type)

✅ RESULTS:
   - 20+ lines of scattered logic → 1 method call
   - Priority order explicit and guaranteed
   - First match exits (no double-counting)
   - Trade #29 scenario now caught
   - Tests verify priority order, not individual methods
   - Easier to modify thresholds (update in one place)
```

---

## CONSTANT LOCATIONS: BEFORE vs AFTER

### BEFORE: Scattered
```python
# harvester_agent.py
TRAILING_STOP_ACTIVATION_PCT: float = 0.35
TRAILING_STOP_DISTANCE_PCT: float = 0.10
MICRO_WINNER_MFE_THRESHOLD_PCT: float = 0.05
MICRO_WINNER_GIVEBACK_PCT: float = 0.30

# hud_tabbed.py (no constants, hard-coded)
if pct > 0.35:  # What does this magic number mean?
    ...

# tests (duplicated values)
def test_trailing_stop():
    assert trailing_distance == 0.10  # Same value, defined twice

# Other places: scattered references
```

⚠️ **PROBLEMS**:
- Same constant defined in multiple files
- Hard-coded values in tests and HUD
- Changing one breaks the other silently
- Code review: "What's 0.35?"

---

### AFTER: Centralized
```python
# src/constants.py - SINGLE SOURCE OF TRUTH
class HarvesterConstants:
    """Harvester agent thresholds"""
    STOP_LOSS_PCT: float = 0.05  # docs: "Max acceptable loss before mandatory exit"
    PROFIT_TARGET_PCT: float = 0.10  # docs: "Target capture amount"
    TRAILING_STOP_ACTIVATION_PCT: float = 0.35  # docs: "MFE required to activate trailing"
    TRAILING_STOP_DISTANCE_PCT: float = 0.10  # docs: "Max drawback from peak"
    MICRO_WINNER_MFE_THRESHOLD_PCT: float = 0.05  # docs: "Min MFE needing protection"
    MICRO_WINNER_GIVEBACK_PCT: float = 0.30  # docs: "Max giveback before exit"
    BREAKEVEN_STOP_PCT: float = 0.004  # docs: "Minimum profitable exit"

# everywhere else: import and use
from src.constants import HarvesterConstants

# harvester_agent.py
engine = ExitDecisionEngine(
    trailing_activation_pct=HarvesterConstants.TRAILING_STOP_ACTIVATION_PCT,
    trailing_distance_pct=HarvesterConstants.TRAILING_STOP_DISTANCE_PCT,
)

# hud_tabbed.py (if needed)
if mfe_pct > HarvesterConstants.TRAILING_STOP_ACTIVATION_PCT:
    ...

# tests
def test_trailing_stop():
    from src.constants import HarvesterConstants
    assert HarvesterConstants.TRAILING_STOP_DISTANCE_PCT == 0.10
```

✅ **RESULTS**:
- One definition, used everywhere
- Docstrings explain each constant
- Change one place, propagates everywhere
- Tests use same constants as code
- Code review: "HarvesterConstants.TRAILING_STOP_DISTANCE_PCT is 10% of MFE peak"
```

---

## FILE LOADING: SIMPLIFIED FLOW

### BEFORE (Implicit logic):
```
HUD.refresh():
  load training_stats from "training_stats.json"
  IF active_symbol AND active_timeframe:
    load training_stats from f"training_stats_{active_symbol}_M{active_timeframe}.json"
    # Overwrites previous!
  
  # Later check is position from first load or second?
  # Depends on timing, no log, hidden complexity
```

### AFTER (Explicit logic):
```
HUD.refresh():
  registry = DataSourceRegistry(data_dir)
  
  # Load position (explicit priority)
  position = registry.load_position(
    active_symbol="XAUUSD",
    active_timeframe_minutes=5
  )
  # → Returns: (Path("data/current_position_XAUUSD_M5.json"), {...})
  # → Logs: "[DATASRC] Loaded position from per-bot file: current_position_XAUUSD_M5.json"
  
  # Load training stats (explicit priority)
  training_stats = registry.load_training_stats(
    active_symbol="XAUUSD",
    active_timeframe_minutes=5,
    bot_is_active=True  # ← Makes priority explicit
  )
  # → Returns: (Path("data/training_stats_XAUUSD_M5.json"), {...})
  # → Logs: "[DATASRC] Loaded training_stats from active bot file: ..."
  
  # Later code can audit which file was used
  loaded_source = registry.get_loaded_source("training_stats")
  # → (Path, data_dict)
```

---

## LOGGING: BEFORE vs AFTER

### BEFORE (Inconsistent):
```
LOG.info("[INIT] Position size=%d", qty)  # Format 1: [TAG] message
LOG.info("[HARVESTER] Trailing stop hit: MFE=%.2f%%", mfe)  # Format 2: [TAG] action: detail
LOG.warning("[FRICTION] Failed to load: %s", e)  # Format 3: mixed
LOG.debug("Loading from %s", filename)  # Format 4: no tag
```

⚠️ **GREP NIGHTMARE**:
```bash
grep "Loaded training_stats" logs/*.log  # FAILS - different formats
grep "\[DATASRC\]" logs/*.log  # Works for new code only
```

---

### AFTER (Consistent):
```
[DATASRC] Loaded position from per-bot file: current_position_XAUUSD_M5.json
[DATASRC] Loaded training_stats from freshest idle bot file: training_stats_BTCUSD_M60.json
[EXIT_ENGINE] PROFIT TARGET triggered: net_profit=10.15% >= target=10.00%
[EXIT_ENGINE] MICRO-WINNER exit: MFE=0.0984%, current=-0.0149%, giveback=113% of MFE
[HARVESTER] Trailing stop hit: MFE=35.2%, giveback=10.5% >= trail=10%
[HUD] Refresh: trigger_steps=1250, harvester_steps=2100, open_trades=3
```

✅ **GREP FRIENDLY**:
```bash
grep "\[DATASRC\]" logs/*.log | tail -20  # ← ALL loads in order
grep "\[EXIT_ENGINE\]" logs/*.log | grep "MICRO"  # ← Find all micro-winner exits
grep "\[HUD\]" logs/*.log | wc -l  # ← Count HUD refreshes
```

---

## SUMMARY: WHAT CHANGES FOR USERS

### Code Quality Improvements:
- ✅ Better logging: Can now grep for specific actions
- ✅ Fewer edge cases: Single source of truth per data type
- ✅ Easier debugging: Can see which file was loaded
- ✅ Consistent exits: Trade #29 scenario now prevented

### What Users Don't See (But Matters):
- ✅ Easier maintenance: Constants in one place
- ✅ Fewer bugs: Tests now match code
- ✅ Easier features: Adding new thresholds just extends constants
- ✅ Better audits: Complete trace of decisions

### What Stays the Same:
- ✅ Bot still trades the same way
- ✅ Same performance (non-breaking changes)
- ✅ Same HUD display
- ✅ Same risk management

---

## DEPLOYMENT CHECKLIST

### Pre-Deployment:
- [ ] Verify DataSourceRegistry loads same files as old code
- [ ] Verify ExitDecisionEngine makes same exit decisions
- [ ] Run tests with new code for 1 hour, compare logs
- [ ] Compare HUD metrics before/after (should be identical)

### Deployment:
- [ ] Merge Phase 1 code
- [ ] Restart bot with new code
- [ ] Monitor logs for [DATASRC] and [EXIT_ENGINE] messages
- [ ] Spot-check that correct files are being loaded

### Post-Deployment:
- [ ] Monitor for 24 hours (next trading session)
- [ ] Compare trades from old vs new code
- [ ] If logs show same decisions, rollback risk is low

---

**Next**: Read REMEDIATION_ACTION_PLAN.md for Phase 1 implementation details

