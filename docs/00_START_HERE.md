# Start Here: cTrader Adaptive Trading Bot Documentation Index

**Last Updated:** February 14, 2026  

This page is the recommended entry point for the documentation set.

---

## 📌 Essential Documentation (Start Here)

### **Current System Status**
📄 **[CURRENT_STATE.md](CURRENT_STATE.md)** - **Single source of truth** for latest changes, fixes, and system status  
📁 **[INDEX.md](INDEX.md)** - **Complete documentation navigation** with all 75+ docs organized by topic

### **Master References**
- **[MASTER_HANDBOOK.md](MASTER_HANDBOOK.md)** - Authoritative system design (1108 lines)
- **[README.md](../README.md)** - Project overview and quick start

---

## Legacy Status Documents (Superseded by CURRENT_STATE.md)

- ~~Consolidated Status and Gap Tracking~~ → See [CURRENT_STATE.md](CURRENT_STATE.md)
- ~~Test Coverage Summary~~ → See [CURRENT_STATE.md](CURRENT_STATE.md) → Testing Checklist

## Quick Start

- Project Overview: ../README.md
- Master Handbook (Authoritative): ../MASTER_HANDBOOK.md
- System Architecture: ../SYSTEM_ARCHITECTURE.md
- System Flow: ../SYSTEM_FLOW.md

## Operations

- Deployment Quickstart: guides/DEPLOYMENT_QUICKSTART.md
- Running with Logs: ../RUNNING_WITH_LOGS.md
- Monitoring Guide: guides/MONITORING_GUIDE.md
- Pre-Launch Checklist: reports/PRE_LAUNCH_CHECKLIST.md
- Disaster Recovery Runbook: ../DISASTER_RECOVERY_RUNBOOK.md

## Scripts and Commands

- Start bot with HUD: ../scripts/start_bot_with_hud.sh
- Start HUD live: ../scripts/start_hud_live.sh
- Daily health check: ../scripts/daily_health_check.sh
- Monitoring (suite): ../scripts/monitoring/
- Stream logs: ../scripts/stream_logs.sh

## Configuration

- Paper vs Live Configuration: guides/PAPER_VS_LIVE_CONFIG.md
- Config directory: ../config/
  - Trading config: ../config/ctrader_trade.cfg
  - Quote config: ../config/ctrader_quote.cfg
  - App tokens directory: ../config/cTraderAppTokens

## Key Components and Guides

- Trade Manager Integration: ../TRADEMANAGER_INTEGRATION.md
- Risk Manager Implementation: ../RISK_MANAGER_IMPLEMENTATION.md
- Multi-Position Analysis: ../MULTI_POSITION_ANALYSIS.md
- Multi-Position Implementation: ../MULTI_POSITION_IMPLEMENTATION.md
- Reward Shaper: ../reward_shaper.py
- Risk-Aware SAC Manager: ../risk_aware_sac_manager.py

## Verification & Tests

- Decision Flow Verification: ../DECISION_FLOW_VERIFICATION.md
- P0 Integration Verification: ../P0_INTEGRATION_VERIFICATION.md
- Integration Status: docs/reports/INTEGRATION_STATUS.md
- Test Entry Points:
  - Core Safety: ../test_core_safety.py, ../test_core_safety_fixed.py
  - Reward Calculations: ../test_reward_calculations.py
  - Multi-Position: ../test_multi_position.py, tests/test_phase3_integration.py
  - Risk Manager: ../test_risk_manager.py, ../test_risk_manager_rl.py
  - HUD Integration: scripts/testing/test_hud_integration.py

## Security and Compliance

- Security Practices: see ../CONSOLIDATED_DOCUMENTATION_AND_GAPS.md (Operational Security Practices)
- .env template: ../.env.example
- Token handling: ../config/cTraderAppTokens

## Notes

- The consolidated document is the single source of truth for gaps and current status. Historical analyses remain available for audit but are not authoritative.
- Keep this page current when adding or relocating documents.
