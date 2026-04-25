# Start Here — cTrader Adaptive Trading Bot

**Last Updated:** April 25, 2026

Recommended entry point for the documentation set.

---

## Essential Documentation

### Current System Status

[CURRENT_STATE.md](CURRENT_STATE.md) — **Single source of truth** for latest changes, fixes, and system status

[INDEX.md](INDEX.md) — **Complete documentation navigation** with all docs organized by topic

### Master References

- [../AGENTS.md](../AGENTS.md) — Coding-agent instructions and source-of-truth rules
- [../MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md) — Authoritative system design & architecture
- [../README.md](../README.md) — Project overview and quick start

---

## Quick Start

- End-to-end setup: [QUICKSTART.md](QUICKSTART.md)
- Removed legacy code manifest: [archive/REMOVED_LEGACY_CODE.md](archive/REMOVED_LEGACY_CODE.md)
- System architecture: [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md)
- FIX session topology: [architecture/FIX_GATEWAY_TOPOLOGY.md](architecture/FIX_GATEWAY_TOPOLOGY.md)

## Operations

- Deployment guide: [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md)
- Running with logs: [operations/RUNNING_WITH_LOGS.md](operations/RUNNING_WITH_LOGS.md)
- HUD reference: [operations/HUD_QUICK_REFERENCE.md](operations/HUD_QUICK_REFERENCE.md)
- Disaster recovery: [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md)

## Training & Models

- Offline→paper→live pipeline: [TRAINING_TO_PRODUCTION_GUIDE.md](TRAINING_TO_PRODUCTION_GUIDE.md)
- Weekend training setup: `./run.sh weekend-train-setup`
- Guarded weekend training run: `./run.sh weekend-train`
- Champion acceptance: `data/checkpoints/offline_champions.json`
- Fleet registry: `data/universe.json`

## Scripts and Commands

```bash
# Universe supervisor (recommended — launches all PAPER-stage bots)
python run_universe.py --watch

# Single paper bot
./run.sh --symbol XAUUSD --timeframe 5 --paper

# HUD only (connects to running bots)
./run.sh --hud-only

# Test suite
python -m pytest tests/ -q
```

## Configuration

- Paper vs live: [guides/PAPER_VS_LIVE_CONFIG.md](guides/PAPER_VS_LIVE_CONFIG.md)
- Adaptive parameters: [guides/ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md)
- FIX configs: `config/ctrader_quote.cfg`, `config/ctrader_trade.cfg`
- Per-bot FIX configs (universe mode): generated under `data/paper_<SYMBOL>_M<TF>/fix/`

## Key Components

- Dual-agent workflow: [TRIGGER_HARVEST_WORKFLOW.md](TRIGGER_HARVEST_WORKFLOW.md)
- Adaptive parameters: [guides/ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md)
- Emergency close: [EMERGENCY_CLOSE.md](EMERGENCY_CLOSE.md)
- Decision flow: [architecture/DECISION_FLOW_VERIFICATION.md](architecture/DECISION_FLOW_VERIFICATION.md)

## Verification & Tests

```bash
python -m pytest tests/ -q        # full suite
python -m pytest tests/unit/ -q   # unit tests only
```

## Security and Compliance

- See [CURRENT_STATE.md](CURRENT_STATE.md) → "Known Issues" for operational risk items
- Never commit `.env`, `cTraderAppTokens`, or `*.cfg` credentials to git
- `.env` template: `.env.example`

## Per-Symbol / Per-Timeframe Scoping Rule

> Every metric, parameter, checkpoint, decision log, reward monitor output, cache,
> and HUD row is scoped by `(symbol, timeframe_minutes)`.  Use `M240` as the
> canonical label for the H4 timeframe — never a separate H4 runtime path.

This rule is enforced throughout `AGENTS.md` and `run_universe.py`.
