# HUD AUDIT - EXECUTIVE SUMMARY & ACTION PLAN

**Date**: 2026-03-08  
**Auditor**: Skeptical Bot Data Validator  
**Status**: 🔴 **CRITICAL FINDINGS DETECTED**  
**Health Score**: 0/100  

---

## TL;DR - What's Wrong?

Your HUD is **displaying data from 7 sources with NO consistency checks**. Three critical issues:

1. **720 trades have PnL recalculated** but we don't know why → $2,587.83 discrepancy
2. **ALL 1,131 trades missing `quantity` field** → cannot show position sizing
3. **Multiple data files, conflicting values** → HUD picks one source, ignores others

**Risk**: **Operators may make trading decisions based on invalid/contradictory data**

---

## What the Audit Found

Ran comprehensive validation on all HUD data sources:

✅ **Trade Log**: 1,131 trades loaded cleanly  
❌ **Data Quality**: 15 trades NULL entry_time, ALL trades missing quantity  
⚠️ **PnL Mismatch**: Current ($-2,793.55) vs Original ($-205.72) = 1,258% variance  
🔴 **File Staleness**: 6+ files >3 days old, some >10 days old  
📊 **Multi-Bot Sync**: Per-bot files may be out of sync with defaults  
🔧 **Version Chaos**: 6 versions of learned_parameters with backups scattered around  

---

## Critical Issues at a Glance

| # | Issue | Impact | Files |
|---|-------|--------|-------|
| 1 | Missing `quantity` | Cannot display position sizing | ALL 1,131 trades |
| 2 | PnL recalculation chaos | $2.6K variance, no audit trail | trade_log.jsonl |
| 3 | NULL entry_time | avg_trade_duration inaccurate | 15 trades |
| 4 | Stale market data | order_book.json is 3+ days old | order_book.json |
| 5 | Multi-bot file conflicts | May display mismatched data | per-symbol files |
| 6 | No data hierarchy | HUD guesses which file to trust | 7 data sources |
| 7 | Backup proliferation | 6 versions of learned params | .bak files |

---

## Generated Audit Reports

Three comprehensive reports have been created:

### 1. **HUD_AUDIT_REPORT.md** (Detailed Technical Analysis)
- Lists all data sources and how they're used
- Quantifies each issue with specific statistics
- Includes recommended fixes for each problem
- **Audience**: Technical leads, architects

### 2. **HUD_AUDIT_VISUAL_SUMMARY.md** (Quick Reference)
- Visual diagrams of data flow
- Color-coded severity levels
- Feasibility matrix for each metric
- Decision tree for prioritizing fixes
- **Audience**: Team leads, PMs

### 3. **HUD_TECHNICAL_ISSUE_MAP.md** (Code-to-Bug Mapping)
- Exact line numbers in hud_tabbed.py
- Before/after code examples
- Specific test cases for validation
- **Audience**: Developers implementing fixes

### 4. **validate_hud_data.py** (Automated Validator)
- Runnable script to re-audit after fixes
- Tracks health score (0-100)
- Exports JSON for CI/CD integration
- **Audience**: DevOps/CI Engineers

### 5. **audit_report.json** (Machine-Readable Results)
- Structured data for automated processing
- Can be imported into dashboards
- Useful for tracking improvements over time

---

## Priority Action Plan

### 🚨 TODAY (Emergency)
```
[ ] Investigate PnL recalculation mystery
    - Why were 720 trades recalculated on 2026-02-17?
    - Is current PnL correct or should we revert?
    - Document decision (this is audit trail!)

[ ] Check if 'quantity' field should exist in trades
    - Look at recent bot commits
    - Check trade execution code
    - Determine: missing by design or bug?

[ ] Run validate_hud_data.py on CI/CD
    - Baseline the 0/100 score
    - Set up monitoring to catch regressions
```

### 📅 THIS WEEK (High Priority - Risk Mitigation)
```
[ ] Data source hierarchy document
    - Create TRUTH_TABLE.md showing authoritative source for each metric
    - Implement in code with explicit constants
    - Add code comments explaining fallback logic

[ ] Add consistency checks to HUD startup
    - Validate required fields exist before rendering
    - Check for stale data and display warnings
    - Fail gracefully or use safe fallbacks

[ ] Archive stale files
    - Move files >7 days old to /data/archive/
    - Keep only current + 1 backup for learned_parameters
    - Clean up decision_log backups
```

### 📊 NEXT SPRINT (Medium Priority - Data Quality)
```
[ ] Backfill quantity field
    - Either: retroactively add to trade_log.jsonl
    - Or: calculate from other fields
    - Or: document why it doesn't apply

[ ] Backfill entry_time for 15 NULL trades
    - Check if market data can source these
    - Or mark trades as "incomplete entry data"

[ ] Implement atomic writes
    - All JSON writes use temp + rename
    - Prevents partial/corrupted files
    - Applies to all data files, not just HUD

[ ] Data versioning system
    - Timestamp all state files
    - Track when values changed (not just current)
    - Useful for post-mortems and debugging
```

### 🏗️ ARCHITECTURE (Next Release)
```
[ ] Unified state file
    - Consider single authoritative JSON for all metrics
    - Eliminate multi-file conflicts
    - Easier to version and validate

[ ] Data audit trail
    - Log: who, what, when, why for data modifications
    - Timestamp all changes
    - Useful for compliance, debugging

[ ] Multi-bot file ownership
    - Clear naming convention (bot-ID in filename)
    - Automatic cleanup of orphaned files
    - Sync verification between bots
```

---

## Validation Checkpoint

**After applying each fix, re-run validation:**

```bash
python3 validate_hud_data.py --export validation_checkpoint.json
```

**Expected improvement trajectory:**

| Phase | Health Score | Key Fixes |
|-------|---|---|
| Current | 0/100 | All critical issues present |
| After PnL clarity | 25/100 | Understand which PnL is correct |
| After quantity field | 45/100 | Position sizing displayable |
| After entry_time fix | 55/100 | Trade duration accurate |
| After file staleness | 70/100 | Data freshness warnings added |
| After multi-bot sync | 80/100 | Per-bot files validated |
| After cleanup | 90/100 | Backups archived, organized |
| Final validation | 95-100/100 | All checks passing |

---

## Risk Assessment

### Current State
```
RISK LEVEL: 🔴 CRITICAL
IMPACT: Operators may make trading decisions based on:
  - Missing data (no position sizing)
  - Contradictory metrics (PnL variance)
  - Stale market data (3+ day old order book)
  
EXAMPLE RISK: "Total PnL shows -$2,793.55, but which is correct?" 
→ Leads to second-guessing, poor decisions, lost confidence in system
```

### Post-Fix State
```
RISK LEVEL: 🟢 LOW
IMPACT: Operators can trust HUD to show:
  - Complete data (all fields present)
  - Fresh data (< 1 hour old)
  - Validated data (consistency checks passed)
  - Clear provenance (knows which file is source of truth)
```

---

## Meetings & Approval

### Recommended Stakeholders to Brief
1. **Trading Team Lead** → Explain data quality issues, risk mitigation
2. **Engineering Lead** → Explain technical complexity, timeline
3. **Risk Manager** → Explain data integrity concerns, controls needed
4. **Operations** → Explain troubleshooting, monitoring needed

### Questions to Answer Before Proceeding
- [ ] Is the recalculated PnL correct, or should we revert?
- [ ] Should `quantity` be in every trade, or is it optional?
- [ ] What's the SLA for data freshness? (1 min, 5 min, 1 hour?)
- [ ] Do we need full audit trail, or just for learned_parameters?

---

## FAQs

**Q: Is the HUD broken RIGHT NOW?**  
A: The HUD still displays data, but some metrics are incomplete or inaccurate. Position sizing is missing, duration is understated, PnL has a large variance. Risk level depends on which metrics you're relying on.

**Q: Should I stop using the HUD?**  
A: No, but verify critical metrics (PnL, position size) using a separate data source until fixes are applied.

**Q: Why wasn't this caught earlier?**  
A: The HUD loads from JSON files and silently handles missing data. There are no consistency checks before rendering. The validators we just wrote are the first "truth check" on the data.

**Q: How long will fixes take?**  
A: Phase 1 (emergency clarification): 1-2 hours. Phases 2-3 (data quality fixes): 2-3 days. Final validation: 1 day.

**Q: Do we need to roll back trades?**  
A: Depends on the answer to "Why were 720 trades recalculated?" That's investigation #1.

---

## Files Delivered

```
ctrader_trading_bot/
├── HUD_AUDIT_REPORT.md               ← Full technical audit
├── HUD_AUDIT_VISUAL_SUMMARY.md       ← Quick reference with diagrams
├── HUD_TECHNICAL_ISSUE_MAP.md        ← Code locations & line numbers
├── validate_hud_data.py              ← Automated validator script
├── audit_report.json                 ← Machine-readable results
└── README.md (this file)
```

---

## Next Steps

1. **TODAY**: Review this audit with team
2. **THIS WEEK**: Investigate PnL recalculation and quantity field
3. **NEXT WEEK**: Implement high-priority fixes
4. **MONTHLY**: Re-validate using validate_hud_data.py

---

**Audit completed**: 2026-03-08 12:02:45 UTC  
**Health Score**: 0/100 🔴 CRITICAL  
**Recommendation**: IMMEDIATE ACTION REQUIRED

For questions about specific findings, see the detailed reports linked above.
