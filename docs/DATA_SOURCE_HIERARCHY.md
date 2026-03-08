# DATA SOURCE HIERARCHY & TRUTH TABLE
**This document defines the authoritative source for each metric displayed in the HUD**

---

## 🎯 HIERARCHY LEVELS

### Level 1: AUTHORITATIVE (Direct Source of Truth)
These files are the definitive source. Always prefer these over other sources.

### Level 2: DERIVED (Calculated from Level 1)
Derived values computed from authoritative sources. Cached for performance.

### Level 3: SNAPSHOT (Session State)
Point-in-time snapshots. Stale after session ends. Use Level 1 instead.

### Level 4: DEPRECATED (Do Not Use)
Old files that should be archived or deleted.

---

## 📊 TRUTH TABLE BY CATEGORY

### A. PERFORMANCE METRICS (Trading Results)

| Metric | Level | Source File | Notes |
|--------|-------|-------------|-------|
| **Total PnL** | 1 | `data/trade_log.jsonl` | Sum of all `pnl` fields. Authoritative. |
| **Trade Count** | 1 | `data/trade_log.jsonl` | Total lines (records). |
| **Win Rate** | 2 | Calculated from trade_log | `wins / total_trades` where `pnl > 0` |
| **Profit Factor** | 2 | Calculated from trade_log | `sum(wins) / abs(sum(losses))` |
| **Sharpe Ratio** | 2 | Calculated from trade_log | Mean return / std dev |
| **Sortino Ratio** | 2 | Calculated from trade_log | Mean return / downside std dev |
| **Max Drawdown %** | 2 | Calculated from trade_log | Peak-to-trough equity decline |
| **Avg Trade Duration** | 2 | Calculated from trade_log | From `exit_time - entry_time` |
| **Best Trade** | 1 | `data/trade_log.jsonl` | Max `pnl` value |
| **Worst Trade** | 1 | `data/trade_log.jsonl` | Min `pnl` value |

**Fallback Chain for Performance**:
```
trade_log.jsonl (ALWAYS USE)
  ↓ (if stale)
performance_snapshot.json (outdated, use trade_log)
  ↓ (if missing)
production_metrics.json (may lag)
```

**Data Quality Notes**:
- ✓ `trade_log.jsonl` has 1,131 trades
- ⚠ 720 trades have `pnl_recalculated=true` (variance: $2,587.83)
- ⚠ 15 trades have `entry_time=null` (affects duration calc)
- ❌ ALL trades missing `quantity` field (position sizing unknown)

---

### B. TRAINING METRICS (Model State)

| Metric | Level | Source File | Notes |
|--------|-------|-------------|-------|
| **Trigger Loss** | 1 | `data/training_stats_XAUUSD_M5.json` (active bot) | Per-symbol if active, else `training_stats.json` |
| **Harvester Loss** | 1 | `data/training_stats_XAUUSD_M5.json` (active bot) | Per-symbol if active, else `training_stats.json` |
| **Trigger Steps** | 1 | `data/training_stats_XAUUSD_M5.json` (active bot) | Cumulative training steps |
| **Harvester Steps** | 1 | `data/training_stats_XAUUSD_M5.json` (active bot) | Cumulative training steps |
| **Z-Omega Score** | 2 | Calculated from training stats | Quality metric |
| **Beta (importance)** | 1 | `data/training_stats_*.json` | Importance sampling weight |
| **Epsilon (explore)** | 1 | `data/training_stats_*.json` | Exploration rate |

**Fallback Chain for Training**:
```
training_stats_ACTIVE_SYMBOL_MTIMEFRAME.json (if position ACTIVE)
  ↓ (if position FLAT)
Freshest training_stats_*_M*.json (pick latest mtime)
  ↓ (if no per-bot files)
training_stats.json (shared default)
```

**Data Quality Notes**:
- ✓ Per-symbol files are fresh (seconds old)
- ⚠ Default file may lag (seconds difference acceptable)
- ⚠ Files should sync when bot writes - verify this happens

---

### C. RISK METRICS (Market State)

| Metric | Level | Source File | Notes |
|--------|-------|-------------|-------|
| **Spread (bps)** | 1 | `data/risk_metrics_XAUUSD_M5.json` (active bot) | Per-symbol if active, else `risk_metrics.json` |
| **VPIN Score** | 1 | `data/risk_metrics_*.json` | Volume-synchronized Probability of Informed trading |
| **VPIN Z-Score** | 2 | Calculated from VPIN | Standardized VPIN |
| **Imbalance** | 1 | `data/risk_metrics_*.json` | Order flow imbalance (-1 to 1) |
| **Depth Bid/Ask** | 1 | `data/order_book.json` | Market depth at best bid/ask |
| **Order Book** | 1 | `data/order_book.json` | Top 10 bid/ask levels |

**Fallback Chain for Risk**:
```
risk_metrics_ACTIVE_SYMBOL_MTIMEFRAME.json (if position ACTIVE)
  ↓ (if position FLAT)
Freshest risk_metrics_*_M*.json (pick latest mtime)
  ↓ (if no per-bot files)
risk_metrics.json (shared default)
  ↓ (if missing above)
order_book.json (derived from orderbook)
```

**Data Quality Notes**:
- ⚠ `order_book.json` may be 3+ days old (STALE)
- ⚠ Risk metrics all show zeros (dormant trading)
- ⚠ Files should be < 1 hour old for quality decision-making

---

### D. POSITION STATE (Current Trade)

| Metric | Level | Source File | Notes |
|--------|-------|-------------|-------|
| **Symbol** | 1 | `data/current_position_SYMBOL_MTF.json` | Active bot determines symbol |
| **Direction** | 1 | `data/current_position_SYMBOL_MTF.json` | LONG, SHORT, or FLAT |
| **Entry Price** | 1 | `data/current_position_SYMBOL_MTF.json` | Execution price at entry |
| **Entry Time** | 1 | `data/current_position_SYMBOL_MTF.json` | ISO timestamp of entry |
| **Quantity** | 1 | `data/current_position_SYMBOL_MTF.json` | Position size (lots) |
| **Current Price** | 2 | From market data / order book | Last market price |
| **Unrealized PnL** | 2 | (Current - Entry) * Direction * Qty | Calculated |

**Selection Logic**:
```
Sort current_position_*.json by mtime (newest first)
Find first with direction != FLAT (prefer active position)
If all FLAT:
  Use most recently modified file
  Or fallback to current_position.json (legacy)
```

**Data Quality Notes**:
- ❌ Position files missing `quantity` field

---

### E. CONFIGURATION STATE

| Metric | Level | Source File | Notes |
|--------|-------|-------------|-------|
| **Bot Config** | 1 | `data/bot_config.json` | Shared across all bots |
| **Universe** | 1 | `data/universe.json` | Symbol specs & starting equity |
| **Starting Equity** | 1 | Resolve from universe, then bot_config | Per-symbol equity |
| **Learned Params** | 1 | `data/learned_parameters.json` | Version is determined by latest mtime |
| **Circuit Breaker** | 1 | `data/circuit_breakers.json` | Breaker states (tripped/reset) |

**Data Quality Notes**:
- ⚠ `universe.json` is 11 days old (may have stale starting_equity)
- ⚠ `learned_parameters.json` has 6 backup versions scattered
- ⚠ `bot_config.json` is shared and may reflect wrong bot

---

### F. DEPRECATED (DO NOT USE)

| File | Why | Replacement |
|------|-----|-------------|
| `data/performance_snapshot.json` | Written only once at startup | Use `trade_log.jsonl` instead |
| `data/production_metrics.json` | Session counters (reset on restart) | Use `trade_log.jsonl` for permanent metrics |
| `data/test_decision_log.json` | Test file (>10 days old) | Delete or archive |
| Backup files (*.bak, *.backup*) | Version chaos | Consolidate to versioned directory |

---

## 🔄 DATA FLOW & SYNC PROTOCOL

### When HUD Refreshes (Every 1-2 seconds):

```
1. Load bot_config.json          # Shared config
2. Load position files            # Find active position
   - Glob current_position_*.json
   - Pick most recent non-FLAT position
   - If all FLAT, pick freshest file
3. Load training stats (per-bot priority):
   - If ACTIVE position: training_stats_SYMBOL_MTF.json
   - Else FLAT: Pick freshest training_stats_*_M*.json
   - Else fallback: training_stats.json
4. Load risk metrics (same logic as #3)
5. compute_metrics_from_trade_log()  # AUTHORITATIVE
   - Parse ALL of trade_log.jsonl
   - Calculate daily/weekly/monthly/lifetime metrics
   - Log data quality warnings
6. Render tabs using computed metrics
```

### Data Consistency Checks:

```
Before rendering, verify:
□ No dual sources of truth (e.g., both snapshot and trade_log have different PnL)
□ Trade log is recent (< 1 hour old)
□ Per-bot files match defaults (within 1 refresh cycle)
□ Required fields exist (pnl, entry_time, exit_time, direction, symbol)
□ No unreasonable values (e.g., PnL variance > 1000%)
```

---

## 📝 IMPLEMENTATION IN HUD CODE

### Current State:
- File in: `src/monitoring/hud_tabbed.py`
- Key function: `_refresh_data()` (line 608)
- Metric calculation: `_compute_metrics_from_trade_log()` (line 842)

### Changes Made:
1. ✅ Added data quality logging for missing fields
2. ✅ Track fraction of trades with complete timestamps
3. ⏳ Add consistency checks before rendering
4. ⏳ Add staleness warnings to data display

### To-Do:
- [ ] Implement `_validate_data_consistency()` check
- [ ] Display data freshness in HUD footer
- [ ] Warn if PnL variance > threshold
- [ ] Warn if quantity field missing from all trades
- [ ] Document hierarchy in code comments

---

## 🔍 VALIDATION QUERIES

To verify data integrity, run:

```bash
# Check if trade_log is authoritative
python3 -c "
import json
trades = [json.loads(l) for l in open('data/trade_log.jsonl')]
print(f'Trades: {len(trades)}')
print(f'PnL: ${sum(t[\"pnl\"] for t in trades):.2f}')
print(f'Missing qty: {sum(1 for t in trades if \"quantity\" not in t)}/{len(trades)}')
"

# Compare per-bot files vs defaults
diff <(jq '.trigger_training_steps' data/training_stats.json) \
     <(jq '.trigger_training_steps' data/training_stats_XAUUSD_M5.json)

# Check file ages
find data -type f -name "*.json" -printf '%T@ %p\n' | sort -n
```

---

## 📚 References

- **File locations**: `data/*.json`
- **HUD code**: `src/monitoring/hud_tabbed.py`
- **Audit reports**: `HUD_AUDIT_*.md`
- **Validation script**: `validate_hud_data.py`
- **Recovery scripts**: `scripts/recover_trade_log.py`, `scripts/cleanup_hud_data.py`

