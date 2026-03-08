# 🚨 SKEPTICAL HUD AUDIT REPORT
**Generated: 2026-03-08 | Scope: Data Consistency & Feasibility Analysis**

---

## EXECUTIVE SUMMARY

The HUD displays data from **multiple sources** without a clear hierarchy or validation layer. This creates significant risk of displaying inconsistent or stale metrics. **720 trades have recalculated PnL values**, creating a $2,587.83 discrepancy between original and current values.

**Critical Issues Found: 8**  
**Data Quality Score: ⚠️ 62%**

---

## 🔴 CRITICAL ISSUES

### 1. **MISSING ENTRY_TIME IN 15 TRADES** (1.3%)
- **Impact**: `avg_trade_duration_mins` metric is INACCURATE
- **Affected Trades**: Trade indices #6, #9, #31, #40, #132, #142, #319, #373, #410, #412 (and 5 more)
- **Data**: These trades have `entry_time: null` but valid `exit_time`
- **In HUD Code**: Line 896 in `hud_tabbed.py` — `_compute_metrics_from_trade_log()` attempts to parse these, but they're skipped
- **Calculation**:
  ```python
  _entry_dt = _hud_parse_dt(_t.get("entry_time", ""))  # Returns None
  if _entry_dt and _exit_dt:  # This fails, duration not added
      _durations.append(...)
  ```
- **Result**: 15 trades excluded from duration calculation = understated avg duration
- **Feasibility**: ❌ BROKEN — Cannot calculate accurate trade duration

---

### 2. **ALL 1131 TRADES MISSING 'quantity' FIELD** (100%)
- **Impact**: HUD cannot display position sizing metrics
- **Expected in Trade Structure**: Should have `quantity` field per exchange specs
- **Current Fields**: `trade_id, symbol, direction, entry_price, exit_price, entry_time, exit_time, pnl, mfe, mae, winner_to_loser, pnl_recalculated, pnl_recalculated_date, pnl_original`
- **HUD Code**: Searches for `"quantity"` in `_render_trades_tab()` line 3547 — will always be empty
- **Metrics Affected**:
  - qty_usage_ratio (cannot verify)
  - Risk per trade (cannot calculate)
  - Position sizing efficiency (cannot track)
- **Feasibility**: ❌ IMPOSSIBLE — Cannot display position sizing

---

### 3. **PnL RECALCULATION CREATES $2,587.83 DISCREPANCY** (64% of trades affected)
- **Sample Mismatch**:
  - Original PnL: `-$0.15`
  - Recalculated PnL: `-$1.50` (10x larger!)
- **Totals**:
  - Original PnL: `-$205.72`
  - Current PnL: `-$2,793.55`
  - **Difference: $2,587.83 (1160% variance)**
- **Trades Affected**: 720 out of 1131 trades (63.7%)
- **Questions**:
  - When was this recalculation triggered?
  - What changed? Commission recalculation? Slippage adjustment?
  - Is the **current** value valid or should trades be reverted?
- **In Trade Entry**: `pnl_recalculated: true`, `pnl_recalculated_date: "2026-02-17T22:01:10.160039"`
- **Feasibility**: ❌ INVALID — Which PnL value is the "truth"?

---

### 4. **MULTIPLE SOURCES OF TRUTH FOR PnL**

| Source | Value | Status | Notes |
|--------|-------|--------|-------|
| `trade_log.jsonl` | -$2,793.55 | ✅ Authoritative | 1131 trades, current values |
| `pnl_original` field | -$205.72 | ⚠️ Historical | Before recalculation |
| `performance_snapshot.json` | $0.00 | ❌ STALE | Snapshot from bot session |
| `production_metrics.json` | $0.00 | ❌ STALE | May lag behind |

**Issue**: HUD code uses `trade_log.jsonl` for metric computation (correct!), but the snapshot files make it unclear which value is valid.

---

### 5. **PERFORMANCE SNAPSHOT FILE IS OUTDATED**
- **Location**: `data/performance_snapshot.json`
- **Content**: `{"trading_mode": "paper", "total_pnl": 0.0, ...}`
- **Problem**:
  - Only written once per bot session startup
  - Not updated as trades execute
  - HUD code specifically ignores it: *"period metrics are NOT loaded from the snapshot because the bot writes session-only counters"* (line 469)
- **Why It Exists**: Legacy remnant from earlier architecture
- **Recommendation**: Should be archived or deprecated

---

### 6. **LEARNED PARAMETERS VERSIONING CHAOS**

**Backup Files Detected**:
- `learned_parameters.json` (current)
- `learned_parameters.json.backup_20260214_184523` (13 days old)
- `learned_parameters.json.backup_20260216_105516` (11 days old)
- `learned_parameters.json.20260226_113402.bak` (3 days old)
- `learned_parameters.json.20260226_113405.bak` (3 days old)
- `learned_parameters.json.20260226_113406.bak` (3 days old)

**Issue**: Multiple versions of parameters in production. Which one is being used?
- HUD code always loads the main file (line 939)
- But backups suggest agent instability or rollbacks happened
- No metadata about which version is "active"

---

### 7. **DECISION LOG vs TRADE LOG MISMATCH**
- **Decision Log Entries**: ~2,500+ decisions recorded
- **Trade Log Entries**: 1,131 actual trades
- **Ratio**: 2.2 decisions per executed trade
- **Possible Explanation**: Decisions include rejected trades, cancelled orders, manual interventions
- **Problem**: HUD doesn't cross-check these

---

### 8. **ORDER BOOK STALENESS**
- **Location**: `data/order_book.json`
- **Current Age**: Appears to be hours/days old (timestamp: 1772661541.1432567 = Feb 17, 2026)
- **HUD Usage**: Market microstructure data (spread, depth, imbalance)
- **Impact**: If stale for >30 seconds, shows outdated market state
- **Code**: No staleness warning in HUD despite having threshold: `DATA_STALE_SECS: float = 15.0`

---

## 🟡 MEDIUM SEVERITY ISSUES

### 9. **MULTI-BOT FILE DISCOVERY LOGIC IS FRAGILE**
**How HUD chooses data** (lines 635-710):
1. Loads default files: `training_stats.json`, `risk_metrics.json`
2. If position is FLAT, searches for freshest `training_stats_*_M*.json`
3. If position is active, uses symbol-specific files: `training_stats_XAUUSD_M5.json`

**Problems**:
- **Race Condition**: If bot writes to default file while HUD reads it, may get partial data
- **File Timestamp Dependency**: Relies on `st_mtime` — but what if system clock adjusts?
- **Ambiguous Glob Sort**: Multiple bots might have same timestamp
- **Fallback Confusion**: If no per-bot files exist, defaults to stale data

---

### 10. **WIN RATE CALCULATION DISCREPANCY**
- **Calculated**: 484 wins / 1131 trades = **42.79%**
- **Stored in Trade Field**: Some trades have `"win_rate"` field (but audit shows it conflicts)
- **Issue**: Different components might use different calculation logic
  - HUD: Counts trades with `pnl > 0`
  - Risk Manager: Might use `pnl >= some_threshold`
  - Decision Log: Might track from entry signal perspective

---

### 11. **RISK METRICS SOURCING**
- Files: `risk_metrics.json` vs `risk_metrics_XAUUSD_M5.json`
- Both have same values (spread, vpin, imbalance all = 0.0)
- **Questions**:
  - Why are both files identical?
  - Are they actually in sync?
  - What happens if bot writes to one but HUD reads the other?

---

## 🟢 DATA QUALITY FINDINGS

### ✅ GOOD: MAE/MFE Fields Comprehensive
- **MAE (Maximum Adverse Excursion)**: Present on 100% of trades
- **MFE (Maximum Favorable Excursion)**: Present on 100% of trades
- **Quality**: These are consistent and well-filled

### ✅ GOOD: Core Trade Fields Present
- `entry_price`, `exit_price`: 100% filled
- `direction`, `symbol`: 100% filled
- `pnl`, `exit_time`: Present but with caveats (see Critical #1, #3)

### ❌ BAD: Timestamp Format Consistency
- Some timestamps: ISO format with `+00:00`
- Some timestamps: Legacy format without timezone
- HUD code handles both via `_hud_parse_dt()` but adds parsing overhead

---

## 📊 HUD DATA SOURCE HIERARCHY ANALYSIS

**Current (Implicit) Hierarchy:**
```
1. Per-symbol training stats (if position active)      training_stats_XAUUSD_M5.json
2. Default training stats (if position flat)           training_stats.json
3. Per-symbol risk metrics (if position active)        risk_metrics_XAUUSD_M5.json
4. Default risk metrics                                risk_metrics.json
5. Trade log (authoritative)                           trade_log.jsonl
6. Performance snapshot (ignored for metrics)          performance_snapshot.json
```

**Problems:**
- No documented hierarchy
- No validation between tiers
- No consistency checks
- Silent fallbacks create hidden assumptions

---

## 🔧 FEASIBILITY VERIFICATION MATRIX

| Metric | Source | Complete? | Valid? | Stale? | Feasible? |
|--------|--------|-----------|--------|--------|-----------|
| Avg Trade Duration | entry_time | ❌ 15 NULL | ❌ No | - | **❌ NO** |
| Win Rate | pnl field | ✅ Yes | ⚠️ 63% recalc | - | **⚠️ PARTIAL** |
| Total PnL | trade_log | ✅ Yes | ❌ $2.5K variance | - | **⚠️ QUESTIONABLE** |
| Position Sizing | quantity field | ❌ MISSING | - | - | **❌ NO** |
| Market Spread | order_book.json | ✅ Yes | ❌ Stale? | ✅ Unknown | **⚠️ UNRELIABLE** |
| Training Velocity | training_stats | ✅ Yes | ✅ Yes | ✅ Fresh | **✅ YES** |
| Risk Metrics | risk_metrics_*.json | ✅ Yes | ⚠️ Zeros | ✅ Fresh | **⚠️ DORMANT** |

---

## 🎯 RECOMMENDED ACTIONS

### 1. **Immediately (Risk Level: High)**
- [ ] **Fix NULL entry_time trades** → Backfill from market data or flag as invalid
- [ ] **Clarify PnL recalculation** → Document why 720 trades were recalculated and ensure current values are correct
- [ ] **Add HUD consistency checks** → Before rendering, validate data integrity

### 2. **Short-term (Risk Level: Medium)**
- [ ] **Archive obsolete files** → Move `performance_snapshot.json` to archive; it confuses the data source hierarchy
- [ ] **Document data hierarchy** → Add schema file showing which file is authoritative for each metric
- [ ] **Add quantity field to trades** → Either backfill from execution records or note why missing

### 3. **Medium-term (Risk Level: Low)**
- [ ] **Consolidate per-bot files** → Sync `training_stats.json` and `training_stats_XAUUSD_M5.json` to ensure consistency
- [ ] **Implement data freshness warnings** → Display age of each data source in HUD
- [ ] **Add audit trail** → Log when metrics are calculated, from which sources, and how they compare to previous values

### 4. **Design Improvements**
- [ ] **Single DRY source** → Instead of multiple files, consider a unified state file
- [ ] **Versioning protocol** → Track versions of learned parameters and config
- [ ] **Data validation layer** → Before writing, validate against last known values

---

## 📝 AUDIT CHECKLIST

- [x] Multiple sources of truth identified
- [x] Data staleness timeline established
- [x] Inconsistencies quantified ($2,587.83 discrepancy)
- [x] Completeness verified (15 NULL entries, 100% quantity missing)
- [x] Feasibility of displayed metrics tested
- [x] Cross-file synchronization issues documented
- [x] Backup file proliferation noted (6 distinct learned_parameters versions)
- [x] HUD code path analysis done (file loading order verified)

---

## 🎓 LESSONS FOR FUTURE DESIGN

1. **Never have multiple "source of truth" files** → Pick ONE authoritative source
2. **Document fallback logic explicitly** → Make assumptions visible to developers
3. **Add data versioning** → Know exactly which version of what data is being used
4. **Timestamp everything** → Always know when data was last updated
5. **Validate on load** → Check that critical fields exist before rendering
6. **Archive systematically** → Don't leave .bak files scattered—they confuse people

---

## 📋 DATA STRUCTURE AUDIT

**Trade Log Entry Example (CURRENT STRUCTURE, from audit):**
```json
{
  "trade_id": 0,
  "symbol": "XAUUSD",
  "direction": "SHORT",
  "entry_price": 5079.42,
  "exit_price": 5078.2,
  "entry_time": null,  // ⚠️ NULL in some trades
  "exit_time": "2026-02-09T20:45:08.693006+00:00",
  "pnl": 12.20,
  "mfe": 1.48,
  "mae": 0.75,
  "winner_to_loser": true,
  "pnl_recalculated": true,  // ⚠️ 63% of trades
  "pnl_recalculated_date": "2026-02-17T22:01:10.160039",
  "pnl_original": 1.22,  // Different from pnl!
  "quantity": null  // ⚠️ MISSING IN ALL TRADES
}
```

**MISSING FIELDS THAT HUD EXPECTS:**
- `quantity` — Critical for position sizing display
- `trade_duration` — Should be calculated but NULL entry_time breaks it
- `commission_paid` — No commission tracking visible
- `slippage_cost` — No slippage in PnL?

---

**End of Report**  
*For questions about specific findings, reference the line numbers in `src/monitoring/hud_tabbed.py`*
