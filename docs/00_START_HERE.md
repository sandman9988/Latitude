# Start Here: cTrader Adaptive Trading Bot Documentation Index

**Last Updated:** March 8, 2026  

This page is the recommended entry point for the documentation set.

---

## 📌 Essential Documentation (Start Here)

### **Current System Status**
📄 **[CURRENT_STATE.md](CURRENT_STATE.md)** - **Single source of truth** for latest changes, fixes, and system status  
📁 **[INDEX.md](INDEX.md)** - **Complete documentation navigation** with all docs organized by topic

### **Master References**
- **[../MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md)** - Authoritative system design & architecture
- **[../README.md](../README.md)** - Project overview and quick start

---

## Quick Start

- Project Overview: [../README.md](../README.md)
- Master Handbook (Authoritative): [../MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md)
- System Architecture: [architecture/SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md)
- System Flow: [architecture/SYSTEM_FLOW.md](architecture/SYSTEM_FLOW.md)

## Operations

- Deployment Quickstart: [guides/DEPLOYMENT_QUICKSTART.md](guides/DEPLOYMENT_QUICKSTART.md)
- Running with Logs: [operations/RUNNING_WITH_LOGS.md](operations/RUNNING_WITH_LOGS.md)
- Monitoring Guide: [MONITORING_GUIDE.md](MONITORING_GUIDE.md)
- Disaster Recovery Runbook: [operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md)

## Scripts and Commands

- Phase 0 validation: `scripts/testing/phase0_validate_system.sh` (2-4 hour paper validation)
- Quick test: `scripts/testing/quick_test.sh`
- Monitoring suite: `scripts/monitoring/`
- Stream logs: `scripts/stream_logs.sh`
- Daily health check: `scripts/daily_health_check.sh`

## Configuration

- Paper vs Live Configuration: [guides/PAPER_VS_LIVE_CONFIG.md](guides/PAPER_VS_LIVE_CONFIG.md)
- Config directory: `../config/`
  - Trading config: `../config/ctrader_trade.cfg`
  - Quote config: `../config/ctrader_quote.cfg`
  - App tokens directory: `../config/cTraderAppTokens`

## Key Components and Guides

- Trigger + Harvester Workflow: [TRIGGER_HARVEST_WORKFLOW.md](TRIGGER_HARVEST_WORKFLOW.md)
- Risk Manager: [RISK_MANAGER_COMPLETE.md](RISK_MANAGER_COMPLETE.md)
- Adaptive Parameters: [guides/ADAPTIVE_PARAMETERS_GUIDE.md](guides/ADAPTIVE_PARAMETERS_GUIDE.md)
- Multi-Position Analysis: [architecture/MULTI_POSITION_ANALYSIS.md](architecture/MULTI_POSITION_ANALYSIS.md)
- Multi-Position Implementation: [architecture/MULTI_POSITION_IMPLEMENTATION.md](architecture/MULTI_POSITION_IMPLEMENTATION.md)
- Online Learning Integration: [ONLINE_LEARNING_INTEGRATION.md](ONLINE_LEARNING_INTEGRATION.md)

## Verification & Tests

- Decision Flow Verification: [architecture/DECISION_FLOW_VERIFICATION.md](architecture/DECISION_FLOW_VERIFICATION.md)
- Test suite: `python -m pytest tests/` (2506 passing, 3 skipped)
- Integration tests: `tests/integration/`
- Validation tests: `tests/validation/`

## Security and Compliance

- See [CURRENT_STATE.md](CURRENT_STATE.md) → "Known Issues" for operational risk items
- Never commit `.env`, `cTraderAppTokens`, or `*.cfg` credentials to git
- .env template: ../.env.example
- Token handling: ../config/cTraderAppTokens

## Notes

- The consolidated document is the single source of truth for gaps and current status. Historical analyses remain available for audit but are not authoritative.
- Keep this page current when adding or relocating documents.
