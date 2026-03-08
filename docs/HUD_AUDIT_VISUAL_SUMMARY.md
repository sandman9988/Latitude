# SKEPTICAL HUD AUDIT - VISUAL SUMMARY
**Date: 2026-03-08 | Health Score: 0/100 ⚠️ CRITICAL**

---

## 🎯 QUICK FINDINGS

| Category | Finding | Impact | Status |
|----------|---------|--------|--------|
| **Data Completeness** | Missing `quantity` in ALL 1131 trades | Can't show position sizing | 🔴 BROKEN |
| **Data Accuracy** | PnL variance: $2,587.83 (1,258%) | 63% trades recalculated | 🔴 INVALID |
| **Data Freshness** | 15 files exceed 5-hour staleness | Old market data shown | 🔴 STALE |
| **Data Integrity** | 15 trades have NULL entry_time | Duration metric wrong | 🟡 PARTIAL |
| **Data Sync** | Per-bot files sync status unknown | May show mismatched data | 🟡 RISKY |
| **Data Organization** | 7 backup files of learned params | Version ambiguity | 🟡 MESSY |

---

## 🔴 CRITICAL PROBLEMS

### Problem 1: Missing `quantity` Field (100% Impact)
```
SEVERITY: 🔴 CRITICAL
AFFECTED: 1,131 / 1,131 trades (100%)
DISPLAY IMPACT: 
  ❌ qty_usage_ratio = CANNOT CALCULATE
  ❌ position_sizing = CANNOT DISPLAY
  ❌ risk_per_trade = CANNOT VERIFY
STATUS: Renders as empty/zero in HUD trades tab
```

### Problem 2: PnL Recalculation Crisis (63.7% of trades)
```
SEVERITY: 🔴 CRITICAL
AFFECTED: 720 / 1,131 trades
VALUES IN CONFLICT:
  Current PnL:  $-2,793.55 (what HUD shows)
  Original PnL: $  -205.72 (before recalculation)
  DIFFERENCE:   $ 2,587.83 (1,258% variance!)
QUESTION: Which value is "true" and when was it recalculated?
```

**Sample Trade #6 showing the problem:**
```json
{
  "pnl": 12.20,                    # ← HUD uses this
  "pnl_original": 1.22,            # ← But this was original
  "pnl_recalculated": true,        # ← Something changed
  "pnl_recalculated_date": "2026-02-17T22:01:10"
}
```
**That's a 10X difference! What caused it?**

### Problem 3: Data Staleness
```
CRITICAL STALE FILES (>5 hours old):
  ❌ universe.json             : 11.2 days old
  ❌ learned_parameters.json   : 10.0 days old
  ❌ test_decision_log.json    : 10.0 days old
  ❌ decision_log.json         : 3.6 days old  
  ❌ bars_cache.json           : 3.6 days old
  ❌ order_book.json           : 3.6 days old

This is PRODUCTION data but formatted as if stale!
```

---

## 🟡 MEDIUM SEVERITY

### Problem 4: NULL entry_time (1.3% of trades)
```
AFFECTED: 15 / 1,131 trades
IMPACT: Breaks avg_trade_duration calculation
TRADES #: 6, 9, 31, 40, 132, ...
HUD CODE: 
  _entry_dt = _hud_parse_dt(_t.get("entry_time", ""))
  if _entry_dt and _exit_dt:  # This fails when NULL
      _durations.append(...)  # Trade excluded!
RESULT: 15 trades silently dropped from duration metrics
```

### Problem 5: Multiple Files, Same Purpose
```
DEFAULT/SHARED FILES:
  ✓ training_stats.json
  ✓ risk_metrics.json
  ✓ production_metrics.json

PER-BOT/OVERRIDES:
  ✓ training_stats_XAUUSD_M5.json
  ✓ risk_metrics_XAUUSD_M5.json
  
HUD LOGIC (lines 635-710):
  1. Load defaults
  2. Check if FLAT or ACTIVE position
  3. If ACTIVE, load per-bot overrides
  4. If FLAT, search for freshest per-bot file
  
RISK: Race conditions, timestamp collisions, ambiguous fallbacks
```

### Problem 6: Version Chaos
```
Learned Parameters Versions Found:
  learned_parameters.json                    ← CURRENT
  learned_parameters.json.backup_20260214    ← 13 days old
  learned_parameters.json.backup_20260216    ← 11 days old
  learned_parameters.json.20260226_113402    ← 3 days old
  learned_parameters.json.20260226_113405    ← 3 days old
  learned_parameters.json.20260226_113406    ← 3 days old
  
QUESTION: Why 6 versions? Which is active? Was there a rollback?
PROBLEM: Developers/admins don't know which to use
```

---

## 📊 DATA SOURCE TRUTH TABLE

| Metric | trade_log.jsonl | performance_snapshot.json | production_metrics.json | HUD Shows |
|--------|---|---|---|---|  
| Total PnL | -$2,793.55 | $0.00 | $0.00 | ??? |
| Total Trades | 1,131 | N/A | N/A | 1,131 |
| Avg Trade Duration | ???? (15 NULL entries) | N/A | N/A | INACCURATE |
| Win Rate | 42.79% | N/A | N/A | 42.79% |
| Position Size | NULL (missing) | N/A | N/A | NOT SHOWN |

**PATTERN**: HUD uses trade_log as "source of truth" but other files create confusion

---

## 🔄 DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│ MULTI-BOT PRODUCTION ENVIRONMENT                           │
└─────────────────────────────────────────────────────────────┘
         ↓
    BOT WRITES
         ↓
┌─────────────────────────────────────────────────────────────┐
│ DATA FILES IN /data                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ trade_log.jsonl                 (AUTHORITATIVE)        │
│     ├─ 1,131 trades                                        │
│     ├─ 15 have NULL entry_time                             │
│     ├─ ALL missing quantity          ← 🔴 PROBLEM         │
│     ├─ 720 recalculated              ← 🔴 PROBLEM         │
│     └─ Current PnL: -$2,793.55                             │
│                                                             │
│  ⚠️ training_stats.json              (SHARED DEFAULT)      │
│  ⚠️ training_stats_XAUUSD_M5.json    (PER-BOT OVERRIDE)    │
│     └─ May differ from default       ← 🟡 SYNC ISSUE      │
│                                                             │
│  ⚠️ risk_metrics.json                (SHARED DEFAULT)      │
│  ⚠️ risk_metrics_XAUUSD_M5.json      (PER-BOT OVERRIDE)    │
│     └─ May differ from default       ← 🟡 SYNC ISSUE      │
│                                                             │
│  ❌ performance_snapshot.json        (STALE - 10 DAYS)    │
│     └─ Only written at startup       ← 🔴 IGNORED         │
│                                                             │
│  ❌ universe.json                    (STALE - 11 DAYS)    │
│  ❌ learned_parameters.json          (STALE - 10 DAYS)    │
│  ❌ decision_log.json                (STALE - 3 DAYS)     │
│  ⚠️  order_book.json                 (STALE - 3 DAYS)     │
│  ⚠️  7x backup files                 (VERSION CHAOS)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         ↓
    HUD READS
         ↓
┌─────────────────────────────────────────────────────────────┐
│ HUD DISPLAY LOGIC                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  _refresh_data() [line 608]                                │
│     ├─ Load bot_config                                     │
│     ├─ Find active position file                           │
│     ├─ Load training_stats (default or per-bot)            │
│     ├─ Load risk_metrics (default or per-bot)              │
│     ├─ _compute_metrics_from_trade_log()  ← TRUTH!        │
│     │   └─ Reads 1131 trades, skips 15 (NULL entry_time) │
│     │   └─ Recalculates metrics                            │
│     │   └─ RESULT: Metrics based on broken data    ← 🔴   │
│     └─ Load stale snapshots (ignored for metrics)          │
│                                                             │
│  Render Tab [line 1138]                                    │
│     └─ Display metrics with NULL quantity field ← 🔴      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         ↓
    DISPLAY TO USER
         ↓
  ⚠️ DATA MAY BE STALE, INCOMPLETE, OR CONTRADICTORY
```

---

## 🧪 FEASIBILITY SCORE MATRIX

```
Metric                          Feasible?  Confidence  Notes
────────────────────────────────────────────────────────────────
avg_trade_duration               ❌ NO       10%       15 NULL entries break calc
qty_usage_ratio                  ❌ NO        0%       Missing quantity field
position_sizing                  ❌ NO        0%       Cannot calculate
win_rate                          ⚠️ PARTIAL  75%       Based on recalculated PnL
total_pnl                         ⚠️ PARTIAL  40%       $2.5K variance, unclear which is baseline
profit_factor                     🟡 MAYBE    60%       Dependent on PnL validity
sharpe_ratio                      🟡 MAYBE    60%       Dependent on PnL validity
sortino_ratio                     🟡 MAYBE    60%       Dependent on PnL validity
max_drawdown_pct                  🟡 MAYBE    60%       Dependent on PnL validity
spread                            ⚠️ PARTIAL  30%       Order book 3+ days old
vpin_zscore                       ⚠️ PARTIAL  30%       Risk metrics stale
training_steps                    ✅ YES      95%       Fresh per-bot files
training_loss                     ✅ YES      95%       Fresh per-bot files
circuit_breaker_status            ✅ YES      90%       Recently reset
────────────────────────────────────────────────────────────────
OVERALL HUD SCORE:               0/100      ⚠️ BROKEN   Cannot trust most metrics
```

---

## 💡 ROOT CAUSE ANALYSIS

### Why Did This Happen?

1. **Missing quantity field** → Trades logged without this critical field during bot execution  
   *Likely cause*: Trade schema changed, or execution recording incomplete

2. **PnL recalculation** → 720 trades recalculated on 2026-02-17  
   *Possible causes*:
   - Manual correction of slippage/commission?
   - Replay/backfill from exchange?
   - Timezone or decimal precision fix?
   - **No version control or audit trail** ← This is the real problem

3. **Stale data files** → 10+ day old files still in /data  
   *Root cause*: No archival/cleanup strategy; old data accumulates

4. **Multiple sources** → Per-bot AND default files  
   *Root cause*: Multi-bot architecture without clear file ownership; HUD tries to be "smart" with fallback logic

5. **Backup proliferation** → 6+ versions of learned_parameters  
   *Root cause*: Emergency patches/rollbacks without cleanup

---

## 🛠️ RECOMMENDED FIXES (Priority Order)

### 1. IMMEDIATE (Today)
```
□ Document which PnL value is correct (current vs original)
□ Trace back: Why were 720 trades recalculated on 2026-02-17?
□ Verify if trade_log.jsonl is being written correctly in new bot version
□ Check: Is quantity field supposed to be in the trade schema?
```

### 2. URGENT (This Week)
```
□ Archive stale files (>7 days old) → /archive/
□ Consolidate per-bot files → ensure default files stay sync'd
□ Clean up learned_parameters backups → keep only current + 1 backup
□ Add data validation to HUD startup → check required fields exist
```

### 3. IMPORTANT (This Sprint)
```
□ Add NULL entry_time backfill from market data or flag trades
□ Create data schema documentation → what fields are required?
□ Implement atomic writes → prevent partial/corrupted JSON
□ Add audit trail → log when data is modified (who, when, why)
```

### 4. STRUCTURAL (Next Release)
```
□ Single source of truth → unify multi-bot data storage
□ Versioning system → track learned parameter versions
□ Data consistency checks → validate across sources before render
□ Deprecation policy → remove old backup files automatically
```

---

## 📝 AUDIT CHECKLIST FOR FIX VALIDATION

After applying fixes, re-run this audit with:
```bash
python3 validate_hud_data.py --export post_fix_audit.json
```

Expected improvements:
- [ ] Health Score: 0 → 85+
- [ ] Critical Issues: 11 → 0
- [ ] All trades have quantity field
- [ ] PnL variance resolved (original & current aligned)
- [ ] No NULL entry_time values
- [ ] Files < 24 hours old
- [ ] Per-bot files synchronized

---

**Generated by**: `validate_hud_data.py`  
**Report Format**: Markdown + JSON export  
**Next Step**: Schedule review with team to prioritize fixes
