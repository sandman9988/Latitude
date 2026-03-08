# HUD CODE LOCATIONS - ISSUE REFERENCE

This document maps audit findings to specific code locations in `src/monitoring/hud_tabbed.py` for debugging and fixing.

---

## 🔴 ISSUE #1: NULL entry_time → Broken Duration Calculation

**Location**: Line 896-907  
**Function**: `_compute_metrics_from_trade_log()`

```python
# CURRENT CODE (BROKEN):
_durations: list[float] = []
for _t in trades:
    _entry_dt = _hud_parse_dt(_t.get("entry_time", ""))  # Returns None for 15 trades
    _exit_dt  = _hud_parse_dt(_t.get("exit_time", ""))
    if _entry_dt and _exit_dt:  # ← SILENTLY SKIPS if entry_dt is None
        _durations.append((_exit_dt - _entry_dt).total_seconds() / 60.0)

# RESULT:
# - 15 trades with NULL entry_time are EXCLUDED
# - Resulting avg_trade_duration is understated
# - No warning logged that data was excluded
```

**Fix Needed**:
- Either backfill `entry_time` from execution records
- Or flag these 15 trades and exclude them from all metrics
- Or log a warning: "Excluding 15 trades from duration calc due to NULL entry_time"

**Audit Evidence**:
```
Trade #6: entry_time: null, exit_time: 2026-02-09T20:45:08.693006+00:00
Trade #9: entry_time: null, exit_time: 2026-02-09T20:51:53.460288+00:00
Trade #31: entry_time: null, exit_time: 2026-02-09T21:38:00.561747+00:00
... (12 more)
```

---

## 🔴 ISSUE #2: Missing 'quantity' Field → Cannot Display Position Sizing

**Location**: Multiple locations across HUD rendering

### 2a. Trade Log Load (Line 689)
```python
# AUDIT FINDING: All 1,131 trades missing 'quantity' field
trades = []
with open(trade_file, encoding="utf-8") as f:
    for raw_line in f:
        stripped = raw_line.strip()
        if stripped:
            trades.append(json.loads(stripped))

# Each trade has: trade_id, symbol, direction, entry_price, exit_price, 
#                 entry_time, exit_time, pnl, mfe, mae, winner_to_loser, 
#                 pnl_recalculated, pnl_recalculated_date, pnl_original
# 
# MISSING: quantity ← Cannot calculate position sizing
```

### 2b. Trades Tab Rendering (Line 3547)
```python
# HUD searches for quantity but always returns empty
def _render_trades_tab(self) -> str:
    # ... rendering code ...
    qty = trade.get("quantity", "N/A")  # ← Always returns "N/A"
    # ... renders row with empty quantity ...
```

**Fix Needed**:
- Backend: Add `quantity` field to trades when logged
- HUD: Either display "N/A" or calculate from entry_price and notional_value
- Do NOT proceed until quantity field is populated in trade_log.jsonl

---

## 🔴 ISSUE #3: PnL Recalculation → $2,587.83 Discrepancy (1,258% variance)

**Location**: Line 842 onwards in `_compute_metrics_from_trade_log()`

```python
def _compute_metrics_from_trade_log(self):
    """Compute performance metrics directly from trade_log.jsonl."""
    # ...
    # AUDIT FINDING: 720 trades have pnl_recalculated = true
    # Original PnL: -$205.72
    # Current PnL:  -$2,793.55
    # Variance:     $2,587.83 (1,258%)
    
    actual_pnl = sum(t.get("pnl", 0) for t in trades)  # ← Uses CURRENT pnl
    # Does NOT validate against pnl_original
    # Does NOT log when recalculation occurred
```

**Evidence from trades**:
```json
{
  "pnl": 12.20,                   # Using current
  "pnl_original": 1.22,           # Ignored
  "pnl_recalculated": true,
  "pnl_recalculated_date": "2026-02-17T22:01:10.160039"
}
```

**Questions Unanswered**:
1. Why was this trade's PnL multiplied by 10?
2. Did commission change? → If so, should be tracked
3. Was there a slippage adjustment? → Needs audit trail
4. Should we revert to original values? → Depends on correctness

**Fix Needed**:
```python
# ADD VALIDATION:
recalc_trades = [t for t in trades if t.get("pnl_recalculated")]
if recalc_trades:
    LOG.warning(f"⚠️  {len(recalc_trades)} trades were recalculated")
    LOG.warning(f"    Original PnL: ${sum(t.get('pnl_original', 0) for t in recalc_trades):.2f}")
    LOG.warning(f"    Current PnL:  ${sum(t.get('pnl', 0) for t in recalc_trades):.2f}")
    LOG.warning(f"    Variance:     ${abs(...):.2f}")
    
# If variance > 5%, require manual verification before using in metrics
```

---

## 🟡 ISSUE #4: Multi-Bot File Priority Logic → Race Conditions

**Location**: Lines 635-710 in `_refresh_data()`

```python
def _refresh_data(self):
    """Refresh all data from bot exports"""
    
    # Step 1: Load defaults
    self._load_json(_BOT_CONFIG_FILE, "bot_config")
    self._load_json("training_stats.json", "training_stats")
    self._load_json("production_metrics.json", "production_metrics")
    
    # Step 2: Determine active position
    _pos_files = sorted(
        _glob.glob(str(self.data_dir / "current_position_*.json")),
        key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0,
        reverse=True,  # ← Sorts by modification time
    )
    
    # Step 3: Load per-bot overrides based on active position
    if self.active_sym and self.active_tf_min:
        _per_train = f"training_stats_{_sym}_M{_tf}.json"
        if (self.data_dir / _per_train).exists():
            self._load_json(_per_train, "training_stats")  # ← Overwrites default!
    else:
        # FLAT position: pick freshest per-bot file
        _ts_candidates = sorted(
            [_p for _p in self.data_dir.glob("training_stats_*_M*.json")],
            key=lambda _p: _p.stat().st_mtime if _p.exists() else 0,
            reverse=True,
        )
        if _ts_candidates:
            self._load_json(_ts_candidates[0].name, "training_stats")
```

**RISKS**:
1. **Race Condition**: If bot writes to `training_stats.json` while HUD reads it, partial data
2. **Timestamp Tie**: Multiple bots finish training at same time → ambiguous sort order
3. **Silent Fallback**: If per-bot file doesn't exist, silently uses stale default
4. **File Deletion**: If file deleted between glob() and load(), silently fails

**Example Scenario**:
```
T=0.0s: Bot A finishes training, writes training_stats_XAUUSD_M5.json
T=0.1s: Bot B finishes training, writes training_stats_XAUUSD_M5.json (overwrites Bot A's)
T=0.2s: HUD refresh sorts by mtime...
        → Both modifications at T=0.1s, sort order ambiguous!
        → Might display Bot A's stale data
```

**Fix Needed**: 
- Atomic writes (write to temp, then rename)
- Explicit bot identity in file metadata
- Version field in JSON to detect stale data
- Consistent mtime (don't rely on it for ordering)

---

## 🟡 ISSUE #5: Stale Files Not Archived

**Location**: Line 608 onwards in `_refresh_data()`

```python
def _refresh_data(self):
    """Refresh all data from bot exports"""
    
    if not self.data_dir.exists():
        self._set_notification(f"⚠️  Data directory not found: {self.data_dir}", ttl=30)
        return
    
    # ← NO CHECK: Are files stale?
    # ← NO ACTION: Archive or warn about old data
    
    # HUD just loads whatever is there, regardless of age
```

**Stale Files Found**:
```
universe.json           : 11.2 days old
learned_parameters.json : 10.0 days old
test_decision_log.json  : 10.0 days old
decision_log.json       : 3.6 days old
bars_cache.json         : 3.6 days old
order_book.json         : 3.6 days old
```

**Fix Needed**:
```python
# ADD STALENESS CHECK:
DATA_STALE_SECS = 86400  # 24 hours

def _check_file_freshness(self):
    """Warn if critical files are stale."""
    critical_files = {
        "trade_log.jsonl": ("Required for metrics", 3600),      # 1 hour
        "training_stats.json": ("Training stats", 300),         # 5 min
        "risk_metrics.json": ("Market metrics", 300),           # 5 min
        "order_book.json": ("Market data", 60),                 # 1 min
    }
    
    for filename, (label, max_age_sec) in critical_files.items():
        path = self.data_dir / filename
        if not path.exists():
            LOG.warning(f"❌ MISSING: {filename}")
            continue
            
        age = datetime.now().timestamp() - path.stat().st_mtime
        if age > max_age_sec:
            LOG.warning(f"⚠️  STALE {label}: {filename} is {age/60:.1f}min old")
```

---

## 🟡 ISSUE #6: Backup File Proliferation

**Files Detected**:
```
learned_parameters.json                      (current)
learned_parameters.json.backup_20260214      └─ 13 days old
learned_parameters.json.backup_20260216      └─ 11 days old
learned_parameters.json.20260226_113402      └─ 3 days old
learned_parameters.json.20260226_113405      └─ 3 days old
learned_parameters.json.20260226_113406      └─ 3 days old
```

**Problem**:
- 6 different versions confuse operators
- No record of which was "active" when
- No way to revert to specific version cleanly

**Fix Needed**:
```python
# Create /data/versions/ directory with structured versioning
data/
  learned_parameters.json              # ← Always the current
  _versions/
    v1/
      learned_parameters.json          # ← Checkpoint from date X
      metadata.json                    # ← What changed, why, who
    v2/
      learned_parameters.json
      metadata.json
```

---

## 🔵 ISSUE #7: Position File Consistency

**Location**: Line 656 onwards in `_refresh_data()`

```python
def _refresh_data(self):
    # Load active position
    _pos_files = sorted(
        _glob.glob(str(self.data_dir / "current_position_*.json")),
        key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0,
        reverse=True,
    )
    self.position = {}
    for _pf in _pos_files:
        try:
            with open(_pf, encoding="utf-8") as _fh:
                _pd = json.load(_fh)
            if _pd.get("direction", "FLAT") != "FLAT":
                self.position = _pd  # ← Use this position
                break
        except Exception:
            pass
    
    # AUDIT FINDING: Position file missing 'quantity' field
    if not position.get("quantity"):
        # HUD cannot verify position size against risk limits
```

**Current Position File**:
```json
{
  "symbol": "XAUUSD",
  "direction": "FLAT",
  "entry_price": 0.0,
  "quantity": ???  // ← MISSING!
}
```

---

## 📊 SUMMARY TABLE: Code Locations

| Issue | File | Line | Function | Severity |
|-------|------|------|----------|----------|
| NULL entry_time | hud_tabbed.py | 896 | _compute_metrics_from_trade_log | 🔴 CRITICAL |
| Missing quantity | hud_tabbed.py | 689 | trade_log load | 🔴 CRITICAL |
| Missing quantity | hud_tabbed.py | 3547 | _render_trades_tab | 🔴 CRITICAL |
| PnL recalc | hud_tabbed.py | 842-920 | _compute_metrics_from_trade_log | 🔴 CRITICAL |
| Multi-bot logic | hud_tabbed.py | 635-710 | _refresh_data | 🟡 MEDIUM |
| No staleness check | hud_tabbed.py | 608 | _refresh_data | 🟡 MEDIUM |
| Position qty | hud_tabbed.py | 656 | _refresh_data → position load | 🟡 MEDIUM |

---

## 🔧 How to Test Fixes

Before/After each fix, run:
```bash
python3 validate_hud_data.py --export pre_fix.json
# Apply fix
python3 validate_hud_data.py --export post_fix.json
# Compare results
diff <(jq '.health_score' pre_fix.json) <(jq '.health_score' post_fix.json)
```

Expected progression:
- Initial: 0/100
- After Issue #1 (entry_time): 20/100
- After Issue #2 (quantity): 40/100
- After Issue #3 (PnL recalc clarified): 60/100
- After Issue #4 (file sync): 75/100
- After Issue #5 (staleness): 85/100
- After Issue #6 (backups cleaned): 90/100
- After Issue #7 (position qty): 100/100

