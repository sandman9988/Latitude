# cTrader DDQN Bot — Documentation Index

**Last Updated:** April 25, 2026
**Status:** ✅ Operational — test suite green

---

## Quick Navigation

### New Users Start Here

- [00_START_HERE.md](00_START_HERE.md) — Project overview and first steps
- [QUICKSTART.md](QUICKSTART.md) — Get the bot running end-to-end
- [README.md](../README.md) — Root project documentation

### Operators & Traders

- [CURRENT_STATE.md](CURRENT_STATE.md) — **READ THIS FIRST** — Latest status, recent fixes, known issues
- [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) — Production deployment guide
- [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md) — Emergency procedures

### Developers

- [../AGENTS.md](../AGENTS.md) — Coding-agent operating rules, source-of-truth constraints, test commands
- [../MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md) — Authoritative system design & architecture
- [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) — Technical architecture
- [guides/ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md) — Parameter learning system

### Dual-Agent Architecture

- [TRIGGER_HARVEST_WORKFLOW.md](TRIGGER_HARVEST_WORKFLOW.md) — TriggerAgent + HarvesterAgent workflow
- [architecture/DECISION_FLOW_VERIFICATION.md](architecture/DECISION_FLOW_VERIFICATION.md) — Agent decision logic

---

## Documentation Structure

### `/docs` (Root — Core)

| Document | Purpose | Audience | Freshness |
| -------- | ------- | -------- | --------- |
| [CURRENT_STATE.md](CURRENT_STATE.md) | **Latest system status** | All | ✅ Apr 25 |
| [../AGENTS.md](../AGENTS.md) | Coding-agent instructions and source-of-truth rules | Developers | ✅ Apr 25 |
| [../MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md) | Authoritative system design | Developers | ✅ Apr 25 |
| [archive/REMOVED_LEGACY_CODE.md](archive/REMOVED_LEGACY_CODE.md) | Removed legacy-code manifest | Developers | ✅ Apr 25 |
| [QUICKSTART.md](QUICKSTART.md) | End-to-end setup guide | New users | ✅ Apr 25 |
| [TRAINING_TO_PRODUCTION_GUIDE.md](TRAINING_TO_PRODUCTION_GUIDE.md) | Offline→paper→live champion workflow | Developers | ✅ Apr 25 |

### `/docs/guides` (User Guides)

| Guide | Purpose |
| ----- | ------- |
| [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) | Production deployment |
| [guides/PAPER_VS_LIVE_CONFIG.md](guides/PAPER_VS_LIVE_CONFIG.md) | Configuration strategies |
| [guides/ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md) | Parameter tuning |
| [guides/TRADE_LOGGING_GUIDE.md](guides/TRADE_LOGGING_GUIDE.md) | Trade log analysis |

### `/docs/architecture` (Technical Design)

| Document | Purpose |
| -------- | ------- |
| [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) | Overall system design |
| [architecture/ORDER_EXECUTION_FLOW.md](architecture/ORDER_EXECUTION_FLOW.md) | Order routing |
| [architecture/DECISION_FLOW_VERIFICATION.md](architecture/DECISION_FLOW_VERIFICATION.md) | Agent decision logic |
| [architecture/FIX_GATEWAY_TOPOLOGY.md](architecture/FIX_GATEWAY_TOPOLOGY.md) | FIX session isolation and broker topology |

### `/docs/operations` (Runbooks)

| Runbook | Purpose |
| ------- | ------- |
| [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md) | Emergency procedures |
| [operations/HUD_QUICK_REFERENCE.md](operations/HUD_QUICK_REFERENCE.md) | Dashboard guide |
| [operations/RUNNING_WITH_LOGS.md](operations/RUNNING_WITH_LOGS.md) | Log configuration |

---

## Documentation by Use Case

### "I want to run the bot"

1. [QUICKSTART.md](QUICKSTART.md) — setup and universe supervisor
2. [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) — production deployment
3. [operations/HUD_QUICK_REFERENCE.md](operations/HUD_QUICK_REFERENCE.md) — HUD navigation

### "How does it work?"

1. [../MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md) — design philosophy and RL architecture
2. [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) — technical details
3. [TRIGGER_HARVEST_WORKFLOW.md](TRIGGER_HARVEST_WORKFLOW.md) — dual-agent implementation

### "Something went wrong!"

1. [CURRENT_STATE.md](CURRENT_STATE.md) — check known issues first
2. [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md) — emergency procedures

### "I want to modify the code"

1. [../AGENTS.md](../AGENTS.md) — source-of-truth rules and test commands
2. [../MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md) — design principles
3. [architecture/DECISION_FLOW_VERIFICATION.md](architecture/DECISION_FLOW_VERIFICATION.md) — agent logic
4. [guides/ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md) — parameter system

### "I want to train and promote a model"

1. [TRAINING_TO_PRODUCTION_GUIDE.md](TRAINING_TO_PRODUCTION_GUIDE.md) — full offline→paper pipeline
2. [CURRENT_STATE.md](CURRENT_STATE.md) — current ZΩ scores and champion status

### "I want to understand per-symbol/timeframe scoping"

> Every metric, parameter, checkpoint, decision log, cache, reward monitor output,
> and HUD row is scoped by `(symbol, timeframe_minutes)`. The canonical H4 label
> is `M240` — never a separate H4 runtime path.

- See `AGENTS.md` § "Source-of-Truth Constraints" for the full rule set.
- See `CURRENT_STATE.md` § "Offline Champion Source Of Truth" for the April 25 fix.

---

## Recent Changes

### April 25, 2026 — doc cleanup + code fixes

- **Deleted:** 12 audit-artifact docs (AUDIT_SUMMARY, HUD_AUDIT_*, REMEDIATION_ACTION_PLAN, NEXT_STEPS, TICKET_TRACKING_*, FIX_EXECUTION_REPORT, COMPREHENSIVE_CODE_AUDIT)
- **Archived:** 17 stale snapshot/design docs (Jan 2026 snapshots, RISK_MANAGER_COMPLETE, MONITORING_GUIDE, etc.) to `docs/archive/`
- **Rewrote:** QUICKSTART.md — now reflects universe supervisor, per-bot isolation, broker topology
- **Fixed:** `_sync_kurtosis_monitor_threshold()` called once before loop (not per-bar during preseed)
- **Fixed:** Confidence falsy-zero guard in `_update_risk_feedback_thresholds`
- **Fixed:** Float `==` comparison for kurtosis threshold restore
- **Added:** Smoke tests for `evaluate_runtime_checkpoint` and `_run_focused_replay`
- **Added:** `RiskManager` adaptive confidence threshold feedback (per closed trade)
- **Added:** Scoped `RewardShapingMonitor` output — one JSON per `(symbol, TF)`
- **Synced:** `run_universe.py` promoted weights into isolated paper bot checkpoint dirs

### April 23, 2026 — HUD + docs sync

- **Updated:** HUD quick reference to 7-tab map, arrows/Tab navigation
- **Updated:** Canonical timeframe labels (`M1/M5/M15/M30/M60/M240`)

### February 22, 2026 — housekeeping

- **Fixed:** QuickFIX namespace-package type-annotation crash
- **Fixed:** Universe registry stage-demotion bug (LIVE/MICRO no longer demoted)

---

## Documentation Standards

Document headers should include:

```markdown
# Title
**Last Updated:** YYYY-MM-DD
**Status:** Active | Archived
**Audience:** All | Operators | Developers
```

Freshness guidelines: < 7 days = fresh; 7–30 days = current; > 90 days = archive if superseded.

---

## External References

- [cTrader FIX API Docs](https://help.ctrader.com/fix-api/)
- [QuickFIX/Python](https://github.com/quickfix/quickfix)
- [DDQN Paper](https://arxiv.org/abs/1509.06461) — van Hasselt et al.

---

**Navigation:** [🏠 Root](../README.md) | [📄 Current State](CURRENT_STATE.md) | [🚀 Quick Start](QUICKSTART.md) | [🔧 Operations](operations/DISASTER_RECOVERY_RUNBOOK.md)
