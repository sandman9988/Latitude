# GitHub Copilot Instructions — cTrader DDQN Trading Bot

> Last updated: 2026-03-14
> Read MASTER_HANDBOOK.md and docs/CURRENT_STATE.md before making structural changes.

---

## Project Identity

Dual-agent DDQN reinforcement learning trading system connected to cTrader via FIX 4.4 protocol.
Active paper trading XAUUSD M5, Pepperstone demo. Python 3.12, ~35 700 production lines, 2 195 tests passing.

---

## Architecture in one paragraph

A **Trigger agent** (entry specialist) and **Harvester agent** (exit specialist) are both Conv1d DDQN networks trained with Prioritized Experience Replay. Market state is built from log-return features + DSP-based regime detector (damping ratio ζ). `DualPolicy` orchestrates them: Trigger scores high-quality entries; Harvester decides when to close. Risk gate (`RiskManager` + `CircuitBreakers` + `VaR`) sizes positions and blocks trading when circuit breakers trip. All decisions are logged to `logs/audit/decisions.jsonl` (rich JSONL) and mirrored to `data/decision_log.json` (bar-close summary). The HUD (`src/monitoring/hud_tabbed.py`) is a 7-tab curses terminal UI.

---

## Key source files

| File                                 | Purpose                                                             |
| ------------------------------------ | ------------------------------------------------------------------- |
| `src/core/ctrader_ddqn_paper.py`     | Main bot orchestrator (6 449 lines)                                 |
| `src/agents/trigger_agent.py`        | Entry DDQN + fallback strategy (857 lines)                          |
| `src/agents/harvester_agent.py`      | Exit DDQN + min-hold guard (911 lines)                              |
| `src/agents/dual_policy.py`          | Orchestrates both agents; feasibility × ζ gate (1 184 lines)        |
| `src/core/ddqn_network.py`           | Conv1dQNet → temporal_pool_size param (396 lines)                   |
| `src/core/reward_shaper.py`          | 5-dim asymmetric rewards: [capture, wtl, runway, opportunity, time] |
| `src/utils/experience_buffer.py`     | PER + IS weights (raw-priority IS, post-loop update)                |
| `src/features/regime_detector.py`    | DSP pipeline → damping ratio ζ                                      |
| `src/risk/risk_manager.py`           | VaR-based sizing; payoff-ratio budget adaptation                    |
| `src/risk/circuit_breakers.py`       | Sortino, Kurtosis, VPIN breakers                                    |
| `src/core/broker_execution_model.py` | Asymmetric slippage model (440 lines)                               |
| `src/persistence/bot_persistence.py` | Atomic + journaled state persistence                                |
| `src/monitoring/hud_tabbed.py`       | 7-tab curses HUD (3 954 lines)                                      |
| `src/monitoring/audit_logger.py`     | `DecisionLogger` → `logs/audit/decisions.jsonl`                     |
| `src/training/offline_trainer.py`    | Walk-forward DDQN training on historical bars (702 lines)           |
| `src/risk/path_geometry.py`          | 5 entry-quality features (efficiency, gamma, jerk, runway, feasibility) |
| `src/features/event_time_features.py`| Session/rollover/week event features (6 broadcast dims)             |

---

## Feature pipeline (offline, paper, live — all aligned)

All three modes now use identical feature dimensions:

| Group    | Count | Features                                                     |
| -------- | ----- | ------------------------------------------------------------ |
| Base     | 7     | ret1, ret5, ma_diff, vol, imbalance, vpin_z, depth_ratio     |
| Geometry | 5     | efficiency, gamma, jerk, runway, feasibility (PathGeometry)   |
| Event    | 6     | london/ny/tokyo_active, overlap, rollover_proximity, week_progress |
| **Trigger total** | **18** | base + geometry + event                              |
| **Harvester total** | **21** | trigger features + MFE + MAE + bars_held           |

Offline trainer extracts event features from bar timestamps; geometry from bar closes + realized vol.

## Weight format

All weights saved as `.pt` files via `ddqn_network.save_weights()`:  
`{"online": state_dict, "target": state_dict, "optimizer": state_dict, "training_steps": int}`  
Load via `ddqn_network.load_weights()` which handles both `.pt` and legacy `.npz`.

---

## Profitability safeguards

- `MAX_LOSS_PER_TRADE_USD = 100.0` — hard per-trade cap checked on every tick
- Duplicate fill guard in `trade_manager.py` — prevents paper fill + broker fill race condition
- `GHOST_RECONCILE_COOLDOWN_BARS = 3` — blocks entry for 3 bars after ghost position reconciliation

---

## Decision log architecture (TWO logs)

| Log                           | Path                         | Format                    | Writer                    | Use                                                                                      |
| ----------------------------- | ---------------------------- | ------------------------- | ------------------------- | ---------------------------------------------------------------------------------------- |
| **Audit log** (primary)       | `logs/audit/decisions.jsonl` | Append-only JSONL         | `DecisionLogger`          | Rich: session_id, agent, decision, confidence, context, reasoning, trade_id, position_id |
| **Bar-close log** (secondary) | `data/decision_log.json`     | JSON list, full overwrite | `_obc_write_decision_log` | OHLC state + bars_held at every bar close; session field added                           |

**Correlation keys:**

- `trade_id` = 8-char UUID prefix; set on entry, propagated to all HOLD/CLOSE for that trade
- `session` = session_id from `DecisionLogger`, now in both logs
- `position_id` = broker IDs (`PAPER_xxx` in paper mode, FIX ticket in live)

---

## HUD tab map

| Key | Tab           | Key data                                                           |
| --- | ------------- | ------------------------------------------------------------------ |
| O   | Overview      | self-test results, position, account balance, risk status, market  |
| T   | Training      | per-agent steps/loss/reward with trend arrows + sparklines         |
| P   | Performance   | period metrics (24h/7d/Mo/All), per-mode breakdown (paper vs live) |
| R   | Risk/Market   | VaR, circuit breakers, regime ζ, VPIN-z, depth ratio               |
| D   | Decision Log  | `MM-DD HH:MM` timestamps, TrdID column, session-break separators   |
| T   | Trade History | paginated list with M badge (P/L), MIXED MODE banner               |
| H   | Health        | connectivity, self-test detail, system metrics                     |

---

## Coding conventions

### Python style

- Type hints on all public function signatures
- `LOG = logging.getLogger(__name__)` at module top; no bare `print()` in bot code
- Log levels: `LOG.debug` for diagnostics; `LOG.info` only for operationally meaningful events (entries, exits, circuit breakers, reconnects)
- All file writes go through `src/persistence/atomic_persistence.py` (temp+rename) or the journaled WAL — **never** `open(path, "w")` directly for state files
- All divisions: use `src/utils/safe_math.py` safe_div helpers
- Test new functions with pytest in `tests/unit/` or `tests/integration/`

### Decision log entries

- Every entry must include: `timestamp` (ISO), `session`, `trading_mode`, `agent`, `decision`, `confidence`
- `trade_id` is set at entry, propagated until close; `None` on NO_ENTRY
- Rich audit log is **append-only** — never truncate `decisions.jsonl`

### Reward shaping (5 dimensions, strictly enforced)

```python
# Index  Name          Direction
#   0    capture       higher better (capture ratio vs MFE)
#   1    wtl           negative (winner-to-loser penalty)
#   2    runway        higher better (predicted vs actual runway)
#   3    opportunity   negative (missed MFE opportunity cost)
#   4    time          small negative (holding cost)
```

Any change to reward dimensions **must** update: `reward_shaper.py`, `ddqn_network.py` (input size), `trigger_agent.py`, `harvester_agent.py`, `dual_policy.py` — all must agree on exactly 5.

### IS weight correction (PER)

IS weights are computed from **raw priorities before normalisation**, updated **after** the full training loop (not inside it). Do not change this — it was a critical bug fix.

---

## Testing requirements

- Run `python -m pytest tests/ -q` before committing — must stay at 2 195 passing
- Unit tests in `tests/unit/`, integration tests in `tests/integration/`, validation in `tests/validation/`
- After modifying reward shaper dims: run `tests/unit/test_reward_calculations.py`
- After modifying IS weights: run `tests/unit/test_experience_buffer.py`
- After modifying risk manager: run `tests/validation/test_risk_manager.py`

---

## Current open items (as of 2026-03-14)

| Item                                | Priority | Notes                                                               |
| ----------------------------------- | -------- | ------------------------------------------------------------------- |
| L2/imbalance feed                   | MEDIUM   | `imbalance` always 0.0; check FIX MarketDataRequest MDEntryType=0/1 |
| Harvester Q-value convergence       | LOW      | Monitor `ticks_held` trending up in HUD Training tab                |
| `data/decision_log.json` non-atomic | LOW      | Secondary log only; does not affect correctness                     |
| Re-run offline training             | HIGH     | Weights must be retrained with 18-feature pipeline                  |

---

## Deleted modules (do NOT recreate)

These were removed as dead code on Mar 13, 2026. Do not recreate or reference them:
`agent_arena.py`, `cold_start_manager.py`, `early_stopping.py`, `ensemble_tracker.py`,
`feedback_loop_breaker.py`, `generalization_monitor.py`, `parameter_staleness.py`,
`feature_tournament.py`, `time_features.py`, `risk_aware_sac_manager.py`

---

## What NOT to do

- Never hardcode parameters — use `learned_parameters.py` with soft bounds
- Never use absolute time features — use event-relative (minutes-to-rollover, etc.)
- Never write state directly with `open(path, "w")` — use atomic persistence
- Never truncate `logs/audit/decisions.jsonl` — it is append-only
- Do not change reward dimensions without updating all 5 files in the reward pipeline
- Do not add `LOG.info()` for per-bar diagnostics — use `LOG.debug()`
