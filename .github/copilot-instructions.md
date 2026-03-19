# GitHub Copilot Instructions — cTrader DDQN Trading Bot

> Last updated: 2026-03-19
> Read MASTER_HANDBOOK.md and docs/CURRENT_STATE.md before making structural changes.

---

## Project Identity

Dual-agent DDQN reinforcement learning trading system connected to cTrader via FIX 4.4 protocol.
Active paper trading XAUUSD M5, Pepperstone demo. Python 3.12, ~41 300 production lines, 2 221 tests passing.

---

## Architecture in one paragraph

A **Trigger agent** (entry specialist) and **Harvester agent** (exit specialist) are both Conv1d DDQN networks trained with Prioritized Experience Replay. Market state is built from log-return features + DSP-based regime detector (damping ratio ζ). `DualPolicy` orchestrates them: Trigger scores high-quality entries; Harvester decides when to close. Risk gate (`RiskManager` + `CircuitBreakers` + `VaR`) sizes positions and blocks trading when circuit breakers trip. All decisions are logged to `logs/audit/decisions.jsonl` (rich JSONL) and mirrored to `data/decision_log.json` (bar-close summary). The HUD (`src/monitoring/hud_tabbed.py`) is a 7-tab curses terminal UI.

---

## Key source files

| File                                  | Purpose                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------- |
| `src/core/ctrader_ddqn_paper.py`      | Main bot orchestrator (6 554 lines)                                     |
| `src/agents/trigger_agent.py`         | Entry DDQN + fallback strategy (931 lines)                              |
| `src/agents/harvester_agent.py`       | Exit DDQN + min-hold guard (972 lines)                                  |
| `src/agents/dual_policy.py`           | Orchestrates both agents; feasibility × ζ gate (1 190 lines)            |
| `src/core/ddqn_network.py`            | Conv1dQNet → temporal_pool_size param (396 lines)                       |
| `src/core/reward_shaper.py`           | 6-dim asymmetric rewards; result-based timing (890 lines)               |
| `src/utils/experience_buffer.py`      | PER + IS weights (raw-priority IS, post-loop update)                    |
| `src/utils/metrics_calculator.py`     | Single-source period metrics (Sharpe, Sortino, PF, MaxDD)               |
| `src/features/regime_detector.py`     | DSP pipeline → damping ratio ζ                                          |
| `src/features/hmm_regime.py`          | HMM-based regime detector (264 lines)                                   |
| `src/risk/risk_manager.py`            | VaR-based sizing; payoff-ratio budget adaptation (1 521 lines)          |
| `src/risk/circuit_breakers.py`        | Sortino, Kurtosis, VPIN breakers (870 lines)                            |
| `src/core/broker_execution_model.py`  | Asymmetric slippage model (440 lines)                                   |
| `src/persistence/bot_persistence.py`  | Atomic + journaled state persistence                                    |
| `src/persistence/trade_log_reader.py` | Centralized trade_log.jsonl reader (127 lines)                          |
| `src/monitoring/hud_tabbed.py`        | 7-tab curses HUD (4 090 lines)                                          |
| `src/monitoring/audit_logger.py`      | `DecisionLogger` → `logs/audit/decisions.jsonl`                         |
| `src/training/offline_trainer.py`     | Walk-forward DDQN training on historical bars (721 lines)               |
| `src/risk/path_geometry.py`           | 5 entry-quality features (efficiency, gamma, jerk, runway, feasibility) |
| `src/features/event_time_features.py` | Session/rollover/week event features (6 broadcast dims)                 |

---

## Feature pipeline (offline, paper, live — all aligned)

All three modes now use identical feature dimensions:

| Group               | Count  | Features                                                           |
| ------------------- | ------ | ------------------------------------------------------------------ |
| Base                | 7      | ret1, ret5, ma_diff, vol, imbalance, vpin_z, depth_ratio           |
| Geometry            | 5      | efficiency, gamma, jerk, runway, feasibility (PathGeometry)        |
| Event               | 6      | london/ny/tokyo_active, overlap, rollover_proximity, week_progress |
| **Trigger total**   | **18** | base + geometry + event                                            |
| **Harvester total** | **21** | trigger features + MFE + MAE + bars_held                           |

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

| Key | Tab           | Key data                                                             |
| --- | ------------- | -------------------------------------------------------------------- |
| 1   | Overview      | fleet status, position, account balance, risk status, market         |
| 2   | Performance   | period metrics (24h/7d/Mo/All), edge quality, prediction convergence |
| 3   | Training      | offline jobs, per-agent steps/loss/reward with trend arrows          |
| 4   | Risk          | VaR, circuit breakers, regime ζ, reward weights, path geometry       |
| 5   | Market        | spread, L2 ladder, VPIN-z, imbalance, signal synthesis               |
| 6   | Decision Log  | `MM-DD HH:MM` timestamps, TrdID column, session-break separators     |
| 7   | Trade History | paginated list with mode badge (P/L), drill-down detail              |

## HUD keyboard shortcuts

| Key     | Action                                                  |
| ------- | ------------------------------------------------------- |
| `1`-`7` | Switch to tab                                           |
| `Tab`   | Cycle forward; `Shift+Tab` backward                     |
| `s`     | Select symbol/timeframe preset                          |
| `r`     | Review & reset tripped circuit breakers                 |
| `e`     | Set/clear stats epoch (exclude old trades from metrics) |
| `h`     | Help screen                                             |
| `Alt+K` | Emergency kill switch (close all + halt)                |
| `q`     | Quit HUD (bot keeps running)                            |

### Stats epoch (`[e]` key)

Configurable cutoff date stored in `data/stats_epoch.json`. Trades before the epoch are excluded from all Performance tab metrics (period rows, mode breakdown, trade quality, edge quality) but the raw `trade_log.jsonl` is never modified. Useful for excluding old losing periods that drag down current performance assessment.

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

### Reward shaping (6 dimensions, strictly enforced)

```python
# Index  Name              Direction
#   0    capture           higher better (capture ratio vs MFE, magnitude-scaled)
#   1    wtl               negative (winner-to-loser penalty)
#   2    opportunity       negative (missed MFE opportunity cost)
#   3    activity          positive (exploration bonus when stagnant)
#   4    counterfactual    signed  (penalty for early exits vs optimal; uses abs(mfe))
#   5    ensemble          positive (epistemic uncertainty bonus)
```

Timing penalty is **result-based** (MAE/MFE drawdown ratio), NOT bar-based. This scales across timeframes (M5, H1, H4).
Zero-MFE entries receive a penalty (-0.3), not neutral.
Capture reward is magnitude-scaled: `min(mfe / baseline_mfe, 2.0)` with floor 0.3.
Session quality multiplier: London/NY overlap ×1.3, solo session ×1.15, off-peak ×0.85.
Harvester gets specialized reward via `calculate_harvester_reward()` with mae + exit_time params.

Any change to reward dimensions **must** update: `reward_shaper.py`, `ddqn_network.py` (input size), `trigger_agent.py`, `harvester_agent.py`, `dual_policy.py` — all must agree on exactly 6.

### IS weight correction (PER)

IS weights are computed from **raw priorities before normalisation**, updated **after** the full training loop (not inside it). Do not change this — it was a critical bug fix.

---

## Testing requirements

- Run `python -m pytest tests/ -q` before committing — must stay at 2 221 passing
- Unit tests in `tests/unit/`, integration tests in `tests/integration/`, validation in `tests/validation/`
- After modifying reward shaper dims: run `tests/unit/test_reward_calculations.py`
- After modifying IS weights: run `tests/unit/test_experience_buffer.py`
- After modifying risk manager: run `tests/validation/test_risk_manager.py`

---

## Current open items (as of 2026-03-19)

| Item                                | Priority | Notes                                                               |
| ----------------------------------- | -------- | ------------------------------------------------------------------- |
| Offline training ZΩ < 1.0           | HIGH     | Best ZΩ=0.867 with penalty_scale=0.5; may need more epochs or ps=0.3 |
| L2/imbalance feed                   | MEDIUM   | `imbalance` always 0.0; check FIX MarketDataRequest MDEntryType=0/1 |
| Mode breakdown missing trades       | MEDIUM   | ~999 trades have missing/empty `trading_mode` field; not shown      |
| Harvester Q-value convergence       | LOW      | Monitor `ticks_held` trending up in HUD Training tab                |
| `data/decision_log.json` non-atomic | LOW      | Secondary log only; does not affect correctness                     |

---

## Deleted modules (do NOT recreate)

These were removed as dead code on Mar 13, 2026. Do not recreate or reference them:
`agent_arena.py`, `cold_start_manager.py`, `early_stopping.py`, `ensemble_tracker.py`,
`feedback_loop_breaker.py`, `generalization_monitor.py`, `parameter_staleness.py`,
`feature_tournament.py`, `time_features.py`, `risk_aware_sac_manager.py`

---

## Paper → Live roadmap

**Current phase:** Paper trading only — focus on reliable profitability first.

When paper is profitable, the plan is to run paper (challenger) + live (champion) side-by-side with weekend weight promotion. Architecture is ~80% ready:

**Already configurable:** FIX config paths (env vars), credentials (env vars), checkpoint dir (parameter), DDQN weight paths (any path accepted).

**Needs implementation:** Parameterize `hud_data_dir` via `BOT_DATA_DIR` env var, mode-suffix trade/decision logs, plumb `LearnedParametersManager` path, create live FIX configs with separate `SenderCompID`, `scripts/promote_weights.py` with validation gate (paper Sharpe > live), extend `run_universe.py` for paper+live of same instrument.

See `docs/CURRENT_STATE.md` § "Paper → Live Roadmap" for full readiness matrix and implementation plan.

---

## What NOT to do

- Never hardcode parameters — use `learned_parameters.py` with soft bounds
- Never use absolute time features — use event-relative (minutes-to-rollover, etc.)
- Never write state directly with `open(path, "w")` — use atomic persistence
- Never truncate `logs/audit/decisions.jsonl` — it is append-only
- Do not change reward dimensions without updating all 5 files in the reward pipeline
- Do not add `LOG.info()` for per-bar diagnostics — use `LOG.debug()`
- Do not use bar-based timing penalties in rewards — use result-based (MAE/MFE ratio); bar counts don't scale across timeframes
- Do not call `path_geometry.update()` from HUD or snapshot code — read `.last` to avoid double-update corruption
