# Deployment & Operations Documentation Index

**Last Updated:** April 25, 2026
**Purpose:** Quick navigation for deployment-related documents

---

## Start Here (Priority Order)

### 1. Current System Status

[CURRENT_STATE.md](CURRENT_STATE.md) — Single source of truth

- Latest fixes and known issues
- Current trading parameters and ZΩ scores
- Quick commands for operators

### 2. Deployment Guide

[guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) — Production deployment

- 3-step launch sequence
- Bash commands ready to copy/paste
- Troubleshooting section

### 3. Strategy & Philosophy

[guides/PAPER_VS_LIVE_CONFIG.md](guides/PAPER_VS_LIVE_CONFIG.md) — Configuration strategies

- Why micro-positions over paper training
- Three-phase deployment strategy
- Environment variables reference

---

## Core Documentation

| Document | Purpose | Audience |
| -------- | ------- | -------- |
| [CURRENT_STATE.md](CURRENT_STATE.md) | Latest status & fixes | All |
| [QUICKSTART.md](QUICKSTART.md) | End-to-end setup | New users |
| [../AGENTS.md](../AGENTS.md) | Coding-agent instructions and source-of-truth rules | Developers |
| [../MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md) | Authoritative system design | Developers |
| [archive/REMOVED_LEGACY_CODE.md](archive/REMOVED_LEGACY_CODE.md) | Removed legacy-code manifest | Developers |
| [TRAINING_TO_PRODUCTION_GUIDE.md](TRAINING_TO_PRODUCTION_GUIDE.md) | Offline→paper→live pipeline | Developers |

---

## Guides (`/docs/guides`)

| Guide | Purpose |
| ----- | ------- |
| [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) | Production deployment |
| [guides/PAPER_VS_LIVE_CONFIG.md](guides/PAPER_VS_LIVE_CONFIG.md) | Configuration strategies |
| [guides/ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md) | Parameter tuning |
| [guides/TRADE_LOGGING_GUIDE.md](guides/TRADE_LOGGING_GUIDE.md) | Trade log analysis |

---

## Operations (`/docs/operations`)

| Runbook | Purpose |
| ------- | ------- |
| [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md) | Emergency procedures |
| [operations/HUD_QUICK_REFERENCE.md](operations/HUD_QUICK_REFERENCE.md) | Dashboard guide |
| [operations/RUNNING_WITH_LOGS.md](operations/RUNNING_WITH_LOGS.md) | Log configuration |

---

## Architecture (`/docs/architecture`)

| Document | Purpose |
| -------- | ------- |
| [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) | Overall system design |
| [architecture/ORDER_EXECUTION_FLOW.md](architecture/ORDER_EXECUTION_FLOW.md) | Order routing |
| [architecture/DECISION_FLOW_VERIFICATION.md](architecture/DECISION_FLOW_VERIFICATION.md) | Agent decision logic |
| [architecture/FIX_GATEWAY_TOPOLOGY.md](architecture/FIX_GATEWAY_TOPOLOGY.md) | FIX session isolation and broker topology |

---

## Execution Scripts

| Script | Purpose |
| ------ | ------- |
| `scripts/weekend_offline_training.sh` | Guarded per-timeframe offline tournament training |
| `scripts/setup_weekend_training.sh` | Install/update weekend training cron entry |
| `run.sh` | Main bot launcher |

Usage shortcuts:

```bash
./run.sh weekend-train-setup   # install cron entry
./run.sh weekend-train         # run manually (market-closed guard)
./run.sh --hud-only            # open HUD against running bots
python run_universe.py --watch # start universe supervisor
```

---

## Use Case Lookup

### "I want to run the bot"

1. [QUICKSTART.md](QUICKSTART.md) — basic setup and universe supervisor
2. [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md) — production deployment
3. [operations/HUD_QUICK_REFERENCE.md](operations/HUD_QUICK_REFERENCE.md) — HUD navigation

### "Something went wrong!"

1. [CURRENT_STATE.md](CURRENT_STATE.md) — check known issues first
2. [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md) — emergency procedures

### "I want to understand the tech"

1. [../MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md) — RL theory & design
2. [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) — technical deep dive
3. [TRIGGER_HARVEST_WORKFLOW.md](TRIGGER_HARVEST_WORKFLOW.md) — dual-agent workflow

### "I want to scale up"

1. Accumulate 500+ closed trades
2. Verify metrics: Sharpe > 1.5, Win Rate > 45%, WTL < 15%
3. Update `QTY` in environment and restart

### "I want to improve models over the weekend"

1. Read [TRAINING_TO_PRODUCTION_GUIDE.md](TRAINING_TO_PRODUCTION_GUIDE.md) — weekend offline champion workflow
2. Install cron: `./run.sh weekend-train-setup`
3. Run manually: `./run.sh weekend-train` (exits if market is open)
4. Verify accepted candidates in `data/checkpoints/offline_champions.json` and `data/universe.json`

---

**Navigation:** [🏠 Root](../README.md) | [📖 Index](INDEX.md) | [📄 Current State](CURRENT_STATE.md) | [🚀 Quickstart](QUICKSTART.md)
