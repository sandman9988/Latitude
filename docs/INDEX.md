# cTrader DDQN Bot - Documentation Index

**Last Updated:** February 14, 2026  
**Branch:** update-1.1-mfe-mae-tracking-v2  
**System Status:** ✅ Operational (Markets Closed - Weekend)

---

## 📍 Quick Navigation

### 🚀 **New Users Start Here**
- [00_START_HERE.md](00_START_HERE.md) - Project overview and first steps
- [QUICKSTART.md](QUICKSTART.md) - Get bot running in 15 minutes
- [README.md](../README.md) - Root project documentation

### 🔧 **Operators & Traders**
- [CURRENT_STATE.md](CURRENT_STATE.md) - **READ THIS FIRST** - Latest system status, recent fixes, known issues
- [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) - Production deployment guide
- [MONITORING_GUIDE.md](MONITORING_GUIDE.md) - System health monitoring
- [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md) - Emergency procedures

### 💻 **Developers**
- [MASTER_HANDBOOK.md](MASTER_HANDBOOK.md) - Original design reference (37 days old)
- [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) - Technical architecture 
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Codebase organization
- [guides/ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md) - Parameter learning system

### 📊 **Phase Implementation** (Historical)
- [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md) - Defensive programming (2024)
- [PHASE2_SUMMARY.md](PHASE2_SUMMARY.md) - Advanced RL features (2025)
- [PHASE3_QUICK_REFERENCE.md](PHASE3_QUICK_REFERENCE.md) - Dual-agent architecture (2026)

---

## 📁 Documentation Structure

### `/docs` (Root - High Priority)
Core documentation that operators and developers reference frequently.

| Document | Purpose | Audience | Freshness |
|----------|---------|----------|-----------|
| **[CURRENT_STATE.md](CURRENT_STATE.md)** | **Latest system status** | All | ✅ Today |
| [MASTER_HANDBOOK.md](MASTER_HANDBOOK.md) | Original design reference | Developers | 37d old |
| [MONITORING_GUIDE.md](MONITORING_GUIDE.md) | Health checks & alerts | Operators | 33d old |
| [QUICKSTART.md](QUICKSTART.md) | Fast setup guide | New users | 33d old |

### `/docs/guides` (User Guides)
Step-by-step instructions for common tasks.

| Guide | Purpose |
|-------|---------|
| [DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) | Production deployment |
| [PAPER_VS_LIVE_CONFIG.md](guides/PAPER_VS_LIVE_CONFIG.md) | Configuration strategies |
| [ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md) | Parameter tuning |
| [TRADE_LOGGING_GUIDE.md](guides/TRADE_LOGGING_GUIDE.md) | Log analysis |

### `/docs/architecture` (Technical Design)
Low-level system architecture and flows.

| Document | Purpose |
|----------|---------|
| [SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) | Overall system design |
| [ORDER_EXECUTION_FLOW.md](architecture/ORDER_EXECUTION_FLOW.md) | Order routing |
| [DECISION_FLOW_VERIFICATION.md](architecture/DECISION_FLOW_VERIFICATION.md) | Agent decision logic |
| [SYSTEM_FLOW.md](architecture/SYSTEM_FLOW.md) | Data flow diagrams |

### `/docs/operations` (Runbooks & Operations)
Day-to-day operational procedures.

| Runbook | Purpose |
|---------|---------|
| [DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md) | Emergency procedures |
| [HUD_QUICK_REFERENCE.md](operations/HUD_QUICK_REFERENCE.md) | Dashboard guide |
| [RUNNING_WITH_LOGS.md](operations/RUNNING_WITH_LOGS.md) | Log configuration |

### `/docs/reports` (Status Reports)
Point-in-time status reports and completion summaries.

| Report | Date | Purpose |
|--------|------|---------|
| [ERROR_REDUCTION_REPORT.md](reports/ERROR_REDUCTION_REPORT.md) | Jan 2026 | Error analysis |
| [PHASE3_INTEGRATION_COMPLETE.md](reports/PHASE3_INTEGRATION_COMPLETE.md) | Jan 2026 | Dual-agent completion |
| [INTEGRATION_STATUS.md](reports/INTEGRATION_STATUS.md) | Jan 2026 | Component integration |
| [PRE_LAUNCH_CHECKLIST.md](reports/PRE_LAUNCH_CHECKLIST.md) | Jan 2026 | Production readiness |

### `/docs/archive` (Historical/Deprecated)
Old documentation kept for reference.

- Phase update summaries (UPDATE_*.md)
- Deprecated contributing guides
- Historical project summaries
- Pause notifications

---

## 🎯 Documentation by Use Case

### "I want to run the bot"
1. [QUICKSTART.md](QUICKSTART.md) - Basic setup
2. [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) - Production deployment
3. [MONITORING_GUIDE.md](MONITORING_GUIDE.md) - Health monitoring

### "How does it work?"
1. [MASTER_HANDBOOK.md](MASTER_HANDBOOK.md) - Design philosophy
2. [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) - Technical details
3. [PHASE3_QUICK_REFERENCE.md](PHASE3_QUICK_REFERENCE.md) - Current implementation

### "Something went wrong!"
1. **[CURRENT_STATE.md](CURRENT_STATE.md)** - Check known issues first
2. [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md) - Emergency procedures
3. [MONITORING_GUIDE.md](MONITORING_GUIDE.md) - Diagnostic procedures

### "I want to modify the code"
1. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Codebase layout
2. [MASTER_HANDBOOK.md](MASTER_HANDBOOK.md) - Design principles
3. [architecture/DECISION_FLOW_VERIFICATION.md](architecture/DECISION_FLOW_VERIFICATION.md) - Agent logic
4. [guides/ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md) - Parameter system

### "I need to understand the parameters"
1. [guides/ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md) - Parameter learning
2. **[CURRENT_STATE.md](CURRENT_STATE.md) - Section "Current Parameters"**
3. [guides/PAPER_VS_LIVE_CONFIG.md](guides/PAPER_VS_LIVE_CONFIG.md) - Configuration strategies

---

## 🔄 Recent Changes (Last 7 Days)

### February 14, 2026 ✅
- **Created:** [CURRENT_STATE.md](CURRENT_STATE.md) - Comprehensive system status
- **Fixed:** M1 stop loss scaling (0.40% → 0.12%, 67% risk reduction)
- **Implemented:** Stop loss adaptive learning (mirrors TP learning)
- **Verified:** Friction costs correctly applied in exits
- **Enhanced:** Defensive programming (10 critical areas hardened)

### February 13, 2026
- Fixed emergency stop loss bypass issue (DDQN model was ignoring SL)
- Verified stop loss now triggers correctly (tested at MAE=1.53%)

### February 7-13, 2026
- Implemented foreign position auto-close (only XAUUSD positions kept)
- Enhanced MFE/MAE tracking with validation
- Added comprehensive error handling and defensive checks

---

## 📝 Documentation Standards

### Document Headers
All docs should include:
```markdown
# Title
**Last Updated:** YYYY-MM-DD
**Status:** [Active|Archived|Deprecated]
**Audience:** [All|Operators|Developers|Traders]
```

### Freshness Guidelines
- **< 7 days:** Fresh, actively referenced
- **7-30 days:** Current, review for updates
- **30-90 days:** Aging, verify accuracy
- **> 90 days:** Consider archiving if superseded

### Archive Criteria
Move to `/docs/archive` if:
- Superseded by newer documentation
- Refers to deprecated features
- Historical value only (no operational use)
- Part of completed phase (not reference material)

---

## 🗂 Deprecated / To Be Archived

**Candidates for archival** (pending review):

### Duplicate Coverage
- `P0_FIXES_IMPLEMENTATION.md` vs `P0_FIXES_IMPLEMENTED.md` (prefer latter)
- `PHASE1_SUMMARY.md` vs `PHASE2_SUMMARY.md` vs handbook (use handbook + quick refs)

### Superseded by CURRENT_STATE.md
- `DEFENSIVE_PROGRAMMING_ENHANCEMENTS.md` (Feb 14) → merged into CURRENT_STATE
- `TRAINING_LOGIC_REVIEW.md` (Feb 14) → merged into CURRENT_STATE
- `P0_FIXES_IMPLEMENTED.md` (Feb 14) → merged into CURRENT_STATE

### Gap Analysis Redundancy
Multiple gap analysis docs exist - consolidate into:
- One historical gap analysis (archive)
- Current gaps in CURRENT_STATE.md

---

## 🔍 Finding Information

### Full Text Search
```bash
# Search all docs
grep -r "stop loss" docs/

# Search active docs only (exclude archive)
grep -r "epsilon" docs/ --exclude-dir=archive

# Find by topic
grep -r "friction" docs/ -l  # List files only
```

### By Date
```bash
# Recently modified
find docs -name "*.md" -mtime -7 -type f

# Older than 30 days
find docs -name "*.md" -mtime +30 -type f
```

### By Size
```bash
# Large docs (> 10KB)
find docs -name "*.md" -size +10k -exec ls -lh {} \;
```

---

## 📞 Contact & Support

- **Repository:** github.com/sandman9988/Latitude
- **Branch:** update-1.1-mfe-mae-tracking-v2
- **Issue Tracking:** See CURRENT_STATE.md "Known Issues" section

---

## 📚 External References

- [cTrader FIX API Docs](https://help.ctrader.com/fix-api/)
- [QuickFIX/Python](https://github.com/quickfix/quickfix)
- [DDQN Paper](https://arxiv.org/abs/1509.06461) - van Hasselt et al.
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/) - Sutton & Barto

---

**Navigation:** [🏠 Root](../README.md) | [📖 Current State](CURRENT_STATE.md) | [🚀 Quick Start](QUICKSTART.md) | [🔧 Operations](MONITORING_GUIDE.md)
