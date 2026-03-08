# HUD AUDIT - QUICK REFERENCE CHECKLIST

Print this and post it in your team Slack/tools. Use it to track remediation progress.

---

## 📋 CRITICAL ISSUES TO RESOLVE

### Issue 1: Missing `quantity` Field (100% of trades)
- [ ] Determine if quantity should be in trade_log.jsonl
- [ ] If YES → Backfill historical trades
- [ ] If NO → Update HUD to calculate from other fields
- [ ] Document in schema
- [ ] Test: Run `validate_hud_data.py` → quantity_coverage should be 1.0

**Blocking Metrics**: 
- `qty_usage_ratio`
- `position_sizing_efficiency`
- `risk_per_trade`

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete
**Owner**: _________
**ETA**: _________

---

### Issue 2: PnL Recalculation Variance ($2,587.83 = 1,258%)
- [ ] Root cause analysis: Why 720 trades recalculated on 2026-02-17?
- [ ] Review bot commit history around that date
- [ ] Determine if current or original PnL is correct
- [ ] Document decision with evidence
- [ ] If reverting: restore pnl_original values
- [ ] If keeping: add explanation to trade records
- [ ] Update bot code to log recalculation reasons

**Evidence**:
- Original PnL: -$205.72
- Current PnL: -$2,793.55
- Variance: $2,587.83

**Status**: ☐ Not Started | ☐ Investigating | ☐ Decision Made | ☐ Fixed
**Owner**: _________
**ETA**: _________

---

### Issue 3: NULL entry_time in 15 Trades (1.3%)
- [ ] Find affected trades: #6, #9, #31, #40, #132, #142, #319, #373, #410, #412, ...
- [ ] Source entry_time from market data or execution logs
- [ ] Backfill into trade_log.jsonl
- [ ] Alternative: Flag these trades and exclude from duration calc
- [ ] Test: Run `validate_hud_data.py` → entry_time_coverage should be > 0.95

**Affected Trades**: 15 of 1,131 (see trade_log.jsonl)

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete
**Owner**: _________
**ETA**: _________

---

### Issue 4: Multi-Bot File Synchronization
- [ ] Audit per-bot files vs defaults:
  - [ ] training_stats.json → training_stats_XAUUSD_M5.json (match?)
  - [ ] training_stats.json → training_stats_EURUSD_M5.json (match?)
  - [ ] training_stats.json → training_stats_GBPUSD_M5.json (match?)
  - [ ] risk_metrics.json → risk_metrics_XAUUSD_M5.json (match?)
- [ ] Identify which files are authoritative
- [ ] Implement sync protocol
- [ ] Test: Files should match within 1 refresh cycle

**Status**: ☐ Not Started | ☐ Audit Complete | ☐ Sync Protocol Ready | ☐ Complete
**Owner**: _________
**ETA**: _________

---

## ⚠️ HIGH PRIORITY ISSUES

### Issue 5: Stale Data Files (>5 hours old)
**Affected Files**: 
- [ ] universe.json (11.2 days old)
- [ ] learned_parameters.json (10.0 days old)
- [ ] test_decision_log.json (10.0 days old)
- [ ] decision_log.json (3.6 days old)
- [ ] order_book.json (3.6 days old)
- [ ] bars_cache.json (3.6 days old)

**Actions**:
- [ ] Archive files > 7 days old → /data/archive/
- [ ] Verify active trading is updating market data
- [ ] Add staleness warnings to HUD
- [ ] Define SLA for data freshness (suggest: order_book < 1 min)

**Status**: ☐ Not Started | ☐ Archive Complete | ☐ Warnings Added | ☐ Complete
**Owner**: _________
**ETA**: _________

---

### Issue 6: Backup File Proliferation
**Found 6+ versions of learned_parameters**:
- [ ] learned_parameters.json (current)
- [ ] learned_parameters.json.backup_20260214 (DELETE)
- [ ] learned_parameters.json.backup_20260216 (DELETE)
- [ ] learned_parameters.json.20260226_113402 (DELETE)
- [ ] learned_parameters.json.20260226_113405 (DELETE)
- [ ] learned_parameters.json.20260226_113406 (DELETE)

**Actions**:
- [ ] Keep only current + 1 backup (most recent)
- [ ] Move others to /data/versions/ with metadata
- [ ] Document versioning policy
- [ ] Update deployment scripts to clean up old versions

**Status**: ☐ Not Started | ☐ Cleanup Complete | ☐ Policy Defined | ☐ Complete
**Owner**: _________
**ETA**: _________

---

## 📊 LOWER PRIORITY (Next Sprint)

### Issue 7: Position File Missing quantity
- [ ] Check current_position_XAUUSD_M5.json schema
- [ ] Add quantity field to all position files
- [ ] Update position write code in bot
- [ ] Backfill historical positions if possible

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete
**Owner**: _________
**ETA**: _________

---

## 🏗️ STRUCTURAL IMPROVEMENTS (Architecture)

### Issue 8: Data Source Hierarchy
- [ ] Create TRUTH_TABLE.md mapping each metric to authoritative source
- [ ] Implement as code constants in HUD
- [ ] Add fallback logic with explicit logging
- [ ] Document all assumptions

**Status**: ☐ Not Started | ☐ In Progress | ☐ Complete
**Owner**: _________
**ETA**: _________

---

### Issue 9: Data Validation Layer
- [ ] Add startup checks: required fields present?
- [ ] Add consistency checks: cross-file validation?
- [ ] Add staleness checks: data freshness warnings?
- [ ] Integrate validate_hud_data.py into CI/CD

**Status**: ☐ Not Started | ☐ Prototype | ☐ Integration Complete
**Owner**: _________
**ETA**: _________

---

### Issue 10: Atomic Writes + Versioning
- [ ] Implement temp → rename pattern for all JSON writes
- [ ] Add timestamp to all JSON files at write time
- [ ] Track version history for critical files
- [ ] Document versioning protocol

**Status**: ☐ Not Started | ☐ Design Complete | ☐ Implementation | ☐ Complete
**Owner**: _________
**ETA**: _________

---

## ✅ VALIDATION CHECKLIST

After implementing fixes, verify:

- [ ] Run `python3 validate_hud_data.py --export validation_v1.json`
- [ ] Health score improved from 0 → target (85+)
- [ ] All critical issues resolved
- [ ] No new warnings introduced
- [ ] HUD displays correctly (spot check all tabs)
- [ ] Data consistent across multiple refreshes
- [ ] No error messages in HUD logs
- [ ] Re-run validator daily for 1 week to catch regressions

**Final Health Score**: ____/100

---

## 🚀 COMPLETION TIMELINE

| Phase | Deadline | Owner | Status |
|-------|----------|-------|--------|
| Emergency (Issue 2) | TODAY | _______ | ☐ |
| Critical (Issues 1, 3) | This Week | _______ | ☐ |
| High Priority (Issues 5, 6) | Next 2 weeks | _______ | ☐ |
| Medium Priority (Issue 7) | Next Sprint | _______ | ☐ |
| Architecture (Issues 8-10) | Next Release | _______ | ☐ |

---

## 📞 ESCALATION

If you get stuck, check:

1. **HUD_AUDIT_REPORT.md** → Full technical details
2. **HUD_TECHNICAL_ISSUE_MAP.md** → Code line numbers
3. **validate_hud_data.py** → Run for current status
4. **audit_report.json** → Machine-readable data

---

**Audit Date**: 2026-03-08  
**Current Health**: 🔴 0/100 CRITICAL  
**Target Health**: 🟢 95+ ACCEPTABLE

Print this and stick it somewhere visible. Update regularly!
