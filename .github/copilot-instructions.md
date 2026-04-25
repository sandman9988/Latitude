# GitHub Copilot Instructions — cTrader DDQN Trading Bot

> Last updated: 2026-04-25
> Read AGENTS.md, MASTER_HANDBOOK.md, and docs/CURRENT_STATE.md before making structural changes.

---

## Project Identity

Dual-agent DDQN reinforcement learning trading system connected to cTrader via FIX 4.4 protocol.
Active paper trading XAUUSD on a **multi-timeframe fleet** (M1, M5, M15, M30, M60, M240) against a Pepperstone demo, supervised by `run_universe.py --watch`. Python 3.12. Validation remains green with a known log/environment-dependent correlation caveat for runway diagnostics.

**GPU Support:** AMD ROCm 7.2+ (gfx1100/gfx1102/Navi 31/33) with native BF16 training, NVIDIA CUDA, and CPU fallback. AMD optimizations auto-detected at startup.

---

## Architecture in one paragraph

A **Trigger agent** (entry specialist) and **Harvester agent** (exit specialist) are both Conv1d DDQN networks trained with Prioritized Experience Replay. Market state is built from log-return features + DSP-based regime detector (damping ratio ζ). `DualPolicy` orchestrates them: Trigger scores high-quality entries; Harvester decides when to close. Risk gate (`RiskManager` + `CircuitBreakers` + `VaR`) sizes positions and blocks trading when circuit breakers trip. All decisions are logged to `logs/audit/decisions.jsonl` (rich JSONL) and mirrored to `data/decision_log.json` (bar-close summary). The HUD (`src/monitoring/hud_tabbed.py`) is a 7-tab terminal UI with low-latency input polling/drain, arrow-key tab switching, and a dedicated Trades tab.

---

## Key source files

| File                                  | Purpose                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------- |
| `src/core/ctrader_ddqn_paper.py`      | Main bot orchestrator                                                    |
| `src/agents/trigger_agent.py`         | Entry DDQN + fallback strategy                                           |
| `src/agents/harvester_agent.py`       | Exit DDQN + min-hold guard                                               |
| `src/agents/dual_policy.py`           | Orchestrates both agents; feasibility × ζ gate                           |
| `src/core/ddqn_network.py`            | Conv1dQNet → temporal_pool_size param, AMD BF16 training support       |
| `src/core/reward_shaper.py`           | 6-dim asymmetric rewards; result-based timing                            |
| `src/constants_amd.py`               | AMD ROCm GPU optimizations (BF16, float16, batch sizes)                |
| `config/rocm_env.sh`                  | ROCm 7.2+ environment configuration for gfx1102/Navi 33                |
| `src/utils/experience_buffer.py`      | PER + IS weights (raw-priority IS, post-loop update)                    |
| `src/utils/metrics_calculator.py`     | Single-source period metrics (Sharpe, Sortino, PF, MaxDD)               |
| `src/features/regime_detector.py`     | DSP pipeline → damping ratio ζ                                          |
| `src/features/hmm_regime.py`          | HMM-based regime detector                                                |
| `src/risk/risk_manager.py`            | VaR-based sizing; payoff-ratio budget adaptation                         |
| `src/risk/circuit_breakers.py`        | Sortino, Kurtosis, VPIN breakers                                         |
| `src/core/broker_execution_model.py`  | Asymmetric slippage model                                                |
| `src/persistence/bot_persistence.py`  | Atomic + journaled state persistence                                    |
| `src/persistence/trade_log_reader.py` | Centralized trade_log.jsonl reader                                       |
| `src/monitoring/hud_tabbed.py`        | 7-tab terminal HUD                                                       |
| `src/monitoring/audit_logger.py`      | `DecisionLogger` → `logs/audit/decisions.jsonl`                         |
| `src/training/offline_trainer.py`     | Walk-forward DDQN training on historical bars                            |
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

## AMD ROCm GPU Optimizations

The system auto-detects AMD GPUs (gfx1100/gfx1102/Navi 31/33) at startup and applies:

| Optimization | Impact | File |
|-------------|--------|------|
| **BF16 training** | 15-25% faster inference, better numerical stability | `src/core/ddqn_network.py` |
| **Float16 state storage** | 50% memory reduction in experience buffer | `src/utils/experience_buffer.py` |
| **Optimal batch sizes** | Better GPU utilization for 8GB VRAM | `src/constants_amd.py` |
| **MIOpen tuning** | 10-20% faster convolutions | `config/rocm_env.sh` |
| **Gradient accumulation** | Larger effective batch without memory increase | `src/constants.py` |

**Key files:**
- `config/rocm_env.sh` — ROCm environment variables (sourced automatically by `run.sh`)
- `src/constants_amd.py` — AMD-specific constants and detection functions
- `src/core/ddqn_network.py` — BF16 autocast context for RDNA 3 GPUs

**Detection flow:**
1. `run.sh` calls `load_rocm_env()` after venv activation
2. `src/core/ddqn_network.py` → `_get_amd_optimizations()` detects GPU
3. BF16 enabled automatically for RDNA 3 (native BF16 support)
4. Experience buffer uses float16 storage on AMD GPUs

**Manual override:**
```python
# Force BF16 on/off in DDQNNetwork
net = DDQNNetwork(state_dim=64, n_actions=3, use_bf16=True)

# Force float16 storage in ExperienceBuffer
buf = ExperienceBuffer(capacity=50000, use_float16=True)
```

---

## Profitability safeguards

- `MAX_LOSS_PER_TRADE_USD = 100.0` — hard per-trade cap checked on every tick
- Duplicate fill guard in `trade_manager.py` — prevents paper fill + broker fill race condition
- `GHOST_RECONCILE_COOLDOWN_BARS = 3` — blocks entry for 3 bars after ghost position reconciliation
- Kurtosis uses a single action threshold path (circuit-breaker threshold, default 5.0); 3.0 remains alert/telemetry level

## Reward monitor scope

- `RewardShapingMonitor` operates per `symbol + timeframe + broker` scope.
- Rolling quality comparison is exported per bot:
  - short window default `24h`
  - baselines default `7d` and `30d`
- Quality-guard recommendations may adjust participation/selectivity and reward weights when 24h metrics materially degrade versus baselines.

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

### Trade-log runway diagnostics fields (points)

- `predicted_runway_net_points_raw` = unadjusted net runway projection in points
- `runway_bias_ema_points` = EMA runway bias estimate used for adaptive correction
- `runway_adjustment_scale` = adaptive multiplier applied to raw runway projection
- `runway_delta_points` = adjusted predicted runway minus realized `mfe_points`
- `mfe_points`, `mae_points` = realized excursion metrics in points for close attribution

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

## Runway-delta adaptation (point-unit contract)

- Adaptive runway correction is driven by `runway_delta_ema` and applied at close attribution time.
- Safety clamps must remain enforced:
  - `RUNWAY_BIAS_LIMIT_POINTS` caps absolute bias correction (points)
  - `RUNWAY_ADJUST_MIN_SCALE` / `RUNWAY_ADJUST_MAX_SCALE` clamp adaptive scale
- Keep all runway adaptation math in point units end-to-end when populating trade attribution and persisted logs.

### HUD convergence runway precedence

When reading trade records for convergence, HUD should resolve runway points in this order:
1. `predicted_runway_net_points` (adjusted)
2. `predicted_runway_net_points_raw` (raw)
3. legacy fallback: `predicted_runway * entry_price`

## HUD keyboard shortcuts

| Key     | Action                                                  |
| ------- | ------------------------------------------------------- |
| `1`-`7` | Switch to tab                                           |
| `Tab`   | Cycle forward; `Shift+Tab` backward                     |
| `←`/`→` | Cycle tabs left/right                                   |
| `s`     | Select symbol/timeframe preset                          |
| `r`     | Review & reset tripped circuit breakers                 |
| `e`     | Set/clear stats epoch (exclude old trades from metrics) |
| `h`     | Help screen                                             |
| `Alt+K` | Emergency kill switch (close all + halt)                |
| `q`     | Quit HUD (bot keeps running)                            |

### Stats epoch (`[e]` key)

Configurable cutoff date stored in `data/stats_epoch.json`. Trades before the epoch are excluded from all Performance tab metrics (period rows, mode breakdown, trade quality, edge quality) but the raw `trade_log.jsonl` is never modified. Useful for excluding old losing periods that drag down current performance assessment.

---

## Operating the paper-bot fleet

The paper-trading workload is a **fleet of per-timeframe bots** supervised by a single watcher. Each entry in `data/universe.json` (a list under `instruments`) becomes a dedicated `src.core.ctrader_ddqn_paper` process with an isolated FIX session directory and log file (`logs/paper_<SYMBOL>_M<TF>.log`).

| Action                        | Command                                                            |
| ----------------------------- | ------------------------------------------------------------------ |
| Start / restart whole fleet   | `./run.sh universe`                                                |
| Show running bots + watcher   | `./run.sh status`                                                  |
| Kill everything               | `pkill -f run_universe ; pkill -f ctrader_ddqn_paper`              |
| Attach HUD to running fleet   | `./run.sh --hud-only` (interactive terminal required)              |
| Manually promote an instrument | `python3 run_universe.py --promote <SYMBOL> --timeframe <MIN>`    |

Watcher semantics:

- Polls every 30 s, re-launches any bot whose PID disappeared, clears stale `paper_pid` fields.
- Adds new instruments from `data/universe.json` as they reach `stage: PAPER`.
- Writes supervisor logs to `logs/run_universe.log`.
- Runs bots with `start_new_session=True` so HUD/terminal signals do not propagate.

Agent caveats:

- The HUD (`src.monitoring.hud_tabbed`) is an **interactive terminal UI** — it cannot be rendered from a non-interactive agent shell. Summarise from `./run.sh status`, `logs/paper_*.log`, and `data/universe.json` instead of trying to launch it in the background.
- Do **not** edit `data/universe.json` as a dict — the canonical schema is `{"version": 1, "instruments": [ {...}, ... ]}` (list of entries). Any status/diagnostic helper must iterate the list.
- Before stopping bots for a hotfix, prefer targeted `pkill -f "paper_<SYMBOL>_M<TF>"` when only one timeframe needs recycling; the watcher will relaunch it on the next poll.

---

## Coding conventions

### Python style

- Type hints on all public function signatures
- `LOG = logging.getLogger(__name__)` at module top; no bare `print()` in bot code
- Log levels: `LOG.debug` for diagnostics; `LOG.info` only for operationally meaningful events (entries, exits, circuit breakers, reconnects)
- All file writes go through `src/persistence/atomic_persistence.py` (temp+rename) or the journaled WAL — **never** `open(path, "w")` directly for state files
- All divisions: use `src/utils/safe_math.py` safe_div helpers
- Test new functions with pytest in `tests/unit/` or `tests/integration/`

### Type annotations and error prevention

**Type annotations:**
- Always use `dict[str, Any]` instead of bare `dict` for return types and parameters
- Always use `list[dict[str, Any]]` instead of bare `list` for JSON-like data
- Always use `deque[Any]` or `deque[SpecificType]` instead of bare `deque`
- Use `float | None` instead of `float = None` for optional float parameters
- Use `dict[str, Any] | None` instead of `dict = None` for optional dict parameters

**NumPy type conversions:**
- Wrap `np.all()`, `np.any()` results with `bool()` when returning Python `bool`
- Wrap numpy float results with `float()` when returning Python `float`
- Example: `return bool(np.all(np.isfinite(x)))` not `return np.all(np.isfinite(x))`

**Mixin pattern:**
- Declare mixin instance attributes at class level with default values
- Use `None`, `False`, `0`, `""` as defaults that are overwritten by host class
- Example: `ddqn: DDQNNetwork | None = None` not just `ddqn: DDQNNetwork`

**None checks:**
- Add assertions for None checks before accessing attributes: `assert self.ddqn is not None`
- Use local variables with `cast()` for repeated access to potentially-None values
- Initialize all instance variables in `__init__` with default values

**Import organization:**
- Imports at top-level only, except for optional dependencies (use local import with try/except)
- Use `from typing import Any, cast` when needed for type annotations
- Use `import math` at module level, not inside functions

**Decimal handling:**
- Use `SafeMath.to_decimal()` for all price/quantity conversions
- Validate `digits` parameter is 0-10 before use
- Check for NaN/Inf before Decimal conversion (raises ValueError now)
- Use `float()` when passing Decimal to functions expecting float

**JSON-like data:**
- Use `dict[str, Any]` for any JSON-like structure
- Use `list[dict[str, Any]]` for lists of JSON objects
- Use `Any` for values that can be multiple types (string, number, bool, null)

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

Timing penalty is **result-based** (MAE/MFE drawdown ratio), NOT bar-based. This scales across timeframes (M5, M60, M240).
Zero-MFE entries receive a penalty (-0.3), not neutral.
Capture reward is magnitude-scaled: `min(mfe / baseline_mfe, 2.0)` with floor 0.3.
Session quality multiplier: London/NY overlap ×1.3, solo session ×1.15, off-peak ×0.85.
Harvester gets specialized reward via `calculate_harvester_reward()` with mae + exit_time params.

Any change to reward dimensions **must** update: `reward_shaper.py`, `ddqn_network.py` (input size), `trigger_agent.py`, `harvester_agent.py`, `dual_policy.py` — all must agree on exactly 6.

### IS weight correction (PER)

IS weights are computed from **raw priorities before normalisation**, updated **after** the full training loop (not inside it). Do not change this — it was a critical bug fix.

---

## Testing requirements

- Run `python -m pytest tests/ -q` before committing — must stay green
- Unit tests in `tests/unit/`, integration tests in `tests/integration/`, validation in `tests/validation/`
- Known caveat: runway-correlation validation can be environment/log-data dependent; treat as a data-quality check when log completeness differs
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

These were removed as dead code and fully deleted from the active repository. Do not recreate or reference them:
`agent_arena.py`, `cold_start_manager.py`, `early_stopping.py`, `ensemble_tracker.py`,
`feedback_loop_breaker.py`, `generalization_monitor.py`, `parameter_staleness.py`,
`feature_tournament.py`, `time_features.py`, `risk_aware_sac_manager.py`

See `docs/archive/REMOVED_LEGACY_CODE.md` for the removal manifest.

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
- Do not use bare `dict`, `list`, or `deque` in type annotations — always specify type arguments (e.g., `dict[str, Any]`, `list[dict[str, Any]]`, `deque[Any]`)
- Do not return numpy types directly — wrap `np.all()`, `np.any()` with `bool()`, and numpy floats with `float()`
- Do not leave mixin attributes undeclared — always declare with default values at class level
- Do not use `import math` inside functions — import at module level
- Do not access potentially-None attributes without assertion — add `assert self.ddqn is not None` before use
- Do not use `float = None` for optional parameters — use `float | None = None`
- Do not pass `Decimal` to functions expecting `float` — convert with `float()` first
