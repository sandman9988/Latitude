# 📚 Deployment & Operations Documentation Index

**Last Updated:** February 22, 2026  
**Purpose:** Quick navigation for deployment-related documents

---

## 🚀 Start Here (Priority Order)

### 1️⃣ Current System Status
**[CURRENT_STATE.md](CURRENT_STATE.md)** — Single source of truth  
- Latest fixes and known issues
- Current trading parameters
- Test suite status (2506 passing)
- Quick commands for operators

### 2️⃣ Deployment Guide
**[guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md)** — Production deployment  
- 3-step launch sequence
- Bash commands ready to copy/paste
- Troubleshooting section

### 3️⃣ Strategy & Philosophy
**[guides/PAPER_VS_LIVE_CONFIG.md](guides/PAPER_VS_LIVE_CONFIG.md)** — Configuration strategies  
- Why micro-positions over paper training
- Three-phase deployment strategy
- Environment variables reference

---

## 📖 Core Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [CURRENT_STATE.md](CURRENT_STATE.md) | Latest status & fixes | All |
| [QUICKSTART.md](QUICKSTART.md) | Get bot running in 15 min | New users |
| [../MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md) | Authoritative system design | Developers |
| [MONITORING_GUIDE.md](MONITORING_GUIDE.md) | Health checks & alerts | Operators |

---

## 📂 Guides (`/docs/guides`)

| Guide | Purpose |
|-------|---------|
| [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) | Production deployment |
| [guides/PAPER_VS_LIVE_CONFIG.md](guides/PAPER_VS_LIVE_CONFIG.md) | Configuration strategies |
| [guides/ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md) | Parameter tuning |
| [guides/TRADE_LOGGING_GUIDE.md](guides/TRADE_LOGGING_GUIDE.md) | Trade log analysis |

---

## 🔧 Operations (`/docs/operations`)

| Runbook | Purpose |
|---------|---------|
| [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md) | Emergency procedures |
| [operations/HUD_QUICK_REFERENCE.md](operations/HUD_QUICK_REFERENCE.md) | Dashboard guide |
| [operations/RUNNING_WITH_LOGS.md](operations/RUNNING_WITH_LOGS.md) | Log configuration |

---

## 🏗 Architecture (`/docs/architecture`)

| Document | Purpose |
|----------|---------|
| [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) | Overall system design |
| [architecture/ORDER_EXECUTION_FLOW.md](architecture/ORDER_EXECUTION_FLOW.md) | Order routing |
| [architecture/DECISION_FLOW_VERIFICATION.md](architecture/DECISION_FLOW_VERIFICATION.md) | Agent decision logic |
| [architecture/SYSTEM_FLOW.md](architecture/SYSTEM_FLOW.md) | Data flow diagrams |
| [architecture/MULTI_POSITION_ANALYSIS.md](architecture/MULTI_POSITION_ANALYSIS.md) | Multi-position handling |
| [architecture/MULTI_POSITION_IMPLEMENTATION.md](architecture/MULTI_POSITION_IMPLEMENTATION.md) | Implementation details |

---

## 🛠 Execution Scripts

| Script | Duration | Purpose |
|--------|----------|---------|
| `scripts/testing/phase0_validate_system.sh` | 2-4 hours | Paper validation before live |
| `scripts/testing/quick_test.sh` | < 5 min | Fast sanity check |
| `run.sh` | Ongoing | Main bot launcher |
| `scripts/monitoring/` | — | Monitoring dashboard suite |

**Note:** `launch_micro_learning.sh` and `monitor_phase1.sh` are not yet created.
See [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) for the equivalent manual launch commands.

---

## 🎯 Use Case Lookup

### "I want to run the bot"
1. [QUICKSTART.md](QUICKSTART.md) — basic setup
2. [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) — production deployment
3. [MONITORING_GUIDE.md](MONITORING_GUIDE.md) — health monitoring

### "Something went wrong!"
1. [CURRENT_STATE.md](CURRENT_STATE.md) — check known issues first
2. [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md) — emergency procedures
3. [MONITORING_GUIDE.md](MONITORING_GUIDE.md) — diagnosis

### "I want to understand the tech"
1. [../MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md) — RL theory & design
2. [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) — technical deep dive
3. [TRIGGER_HARVEST_WORKFLOW.md](TRIGGER_HARVEST_WORKFLOW.md) — dual-agent workflow

### "I want to scale up"
1. Accumulate 500+ closed trades
2. Check metrics: Sharpe > 1.5, Win Rate > 45%, WTL < 15%
3. Update `QTY` in environment and restart

---

**Navigation:** [🏠 Root](../README.md) | [📖 Index](INDEX.md) | [📄 Current State](CURRENT_STATE.md) | [🚀 Quickstart](QUICKSTART.md)
