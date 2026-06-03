# Removed Legacy Code

**Last Updated:** April 25, 2026
**Status:** Archive manifest
**Audience:** Developers

This file records legacy code removed from the active repository. The source
files themselves are intentionally not kept under `src/` or `tests/` because
they were unreferenced, excluded from normal test collection, or superseded by
the current multi-timeframe DDQN architecture.

## April 25, 2026 Cleanup

Removed tracked archived source modules:

| Removed path | Replacement / reason |
|---|---|
| `src/_archived/agents/agent_arena.py` | Superseded by `src/agents/dual_policy.py`; multi-agent arena was never activated. |
| `src/_archived/core/cold_start_manager.py` | Superseded by per-bot warmup, gating, and epsilon controls. |
| `src/_archived/core/early_stopping.py` | Not wired into the current offline tournament or runtime training loop. |
| `src/_archived/core/ensemble_tracker.py` | Superseded by active trigger/harvester telemetry and reward-shaping monitor outputs. |
| `src/_archived/core/feedback_loop_breaker.py` | Not called from production code. |
| `src/_archived/core/generalization_monitor.py` | Not connected to active train/live telemetry. |
| `src/_archived/core/parameter_staleness.py` | Superseded by `src/persistence/learned_parameters.py`. |
| `src/_archived/features/feature_tournament.py` | Replaced by current offline tournament workflow in `train_offline.py`. |
| `src/_archived/features/time_features.py` | Superseded by `src/features/event_time_features.py`. |
| `src/_archived/risk/risk_aware_sac_manager.py` | SAC was never implemented in the active trading stack. |

Removed tracked archived tests:

| Removed path | Reason |
|---|---|
| `tests/archived/test_calculation_safety.py` | Archived test excluded from normal collection. |
| `tests/archived/test_ddqn_network_legacy.py` | Archived legacy checkpoint test excluded from normal collection. |
| `tests/archived/test_trade_manager_safety.py` | Archived test excluded from normal collection. |

Removed stale documentation:

| Removed path | Replacement / reason |
|---|---|
| `docs/TREE_VIEW.txt` | Static generated tree was stale and listed removed modules. Use `docs/PROJECT_STRUCTURE.md` or `rg --files`. |
| `docs/DISASTER_RECOVERY_RUNBOOK.md` | Legacy duplicate. Canonical runbook is `docs/operations/DISASTER_RECOVERY_RUNBOOK.md`. |

Do not recreate these paths unless a current design explicitly reintroduces the
capability and wires it into runtime code plus tests.
