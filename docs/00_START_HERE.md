# Start Here: cTrader Adaptive Trading Bot Documentation Index

This page is the recommended entry point for the documentation set. It links to the canonical status document, primary references, operational guides, and verification resources.

---

## Canonical Status

- Consolidated Status and Gap Tracking: ../CONSOLIDATED_DOCUMENTATION_AND_GAPS.md
- Test Coverage Summary: ../TEST_COVERAGE_SUMMARY.md

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
