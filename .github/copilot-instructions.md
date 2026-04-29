# GitHub Copilot Instructions — cTrader DDQN Trading Bot

> Last updated: 2026-04-29
> Read AGENTS.md, MASTER_HANDBOOK.md, CLAUDE.md, and docs/CURRENT_STATE.md before making structural changes.

______________________________________________________________________

## Project Identity

Dual-agent DDQN reinforcement learning trading system connected to cTrader via Open API (OpenAPI Hub) and FIX 4.4 protocol.
Active paper trading **XAUUSD + BTCUSD** on a **multi-timeframe fleet** (M1, M5, M15, M30, M60, M240) against a Pepperstone demo, supervised by `run_universe.py --watch`. Python 3.12 on AMD ROCm 6.2 (RX 7600, gfx1102).

**Current topology:** OpenAPI Hub (`src/core/openapi_hub.py`) — one hub process per symbol covering all TFs. `UNIVERSE_BROKER_TOPOLOGY=openapi-hub` is the runtime default. Legacy FIX-based `ctrader_ddqn_paper.py` still exists for reference only.

**Paper-training exploration:** keep live/paper learning deliberately exploratory: `EPSILON_START=1.0`, `EPSILON_END=0.25`, `EPSILON_DECAY=0.9998`, `FORCE_EXPLORATION=1`. All values sourced from `src/constants.py` (`PAPER_EPSILON_*`). Checkpoint metadata may restore current epsilon above the floor, but must not lower the configured paper floor or replace the slower paper decay with stale faster decay.

**GPU Support:** AMD ROCm 7.2+ (gfx1100/gfx1102/Navi 31/33) with native BF16 training, NVIDIA CUDA, and CPU fallback. AMD optimizations auto-detected at startup. Always set `HSA_OVERRIDE_GFX_VERSION=11.0.0`.

______________________________________________________________________

## Architecture in one paragraph

A **Trigger agent** (entry specialist) and **Harvester agent** (exit specialist) are both Conv1d DDQN networks trained with Prioritized Experience Replay. Market state is built from log-return features + DSP-based regime detector (damping ratio ζ). `DualPolicy` orchestrates them: Trigger scores high-quality entries; Harvester decides when to close. Risk gate (`RiskManager` + `CircuitBreakers` + `VaR`) sizes positions and blocks trading when circuit breakers trip. All decisions are logged to `logs/audit/decisions.jsonl` (rich JSONL) and mirrored to `data/decision_log.json` (bar-close summary). The HUD (`src/monitoring/hud_tabbed.py`) is a 7-tab terminal UI with low-latency input polling/drain, arrow-key tab switching, and a dedicated Trades tab.

______________________________________________________________________

## Key source files

| File | Purpose |
| ------------------------------------- | ----------------------------------------------------------------------- |
| `src/core/openapi_hub.py` | **Main hub** — `TFAgent`, all SpotEvents, TrendBars, order flow |
| `src/core/ctrader_ddqn_paper.py` | Legacy FIX-based bot orchestrator (reference only) |
| `src/agents/trigger_agent.py` | Entry DDQN + fallback strategy |
| `src/agents/harvester_agent.py` | Exit DDQN + min-hold guard |
| `src/agents/dual_policy.py` | Orchestrates both agents; feasibility × ζ gate |
| `src/core/ddqn_network.py` | Conv1dQNet → temporal_pool_size param, AMD BF16 training support |
| `src/core/reward_shaper.py` | 6-dim asymmetric rewards; result-based timing |
| `src/constants_amd.py` | AMD ROCm GPU optimizations (BF16, float16, batch sizes) |
| `config/rocm_env.sh` | ROCm 7.2+ environment configuration for gfx1102/Navi 33 |
| `src/utils/experience_buffer.py` | PER + IS weights (raw-priority IS, post-loop update) |
| `src/utils/metrics_calculator.py` | Single-source period metrics (Sharpe, Sortino, PF, MaxDD) |
| `src/features/regime_detector.py` | DSP pipeline → damping ratio ζ |
| `src/features/hmm_regime.py` | HMM-based regime detector |
| `src/risk/risk_manager.py` | VaR-based sizing; payoff-ratio budget adaptation |
| `src/risk/circuit_breakers.py` | Sortino, Kurtosis, VPIN breakers |
| `src/core/broker_execution_model.py` | Asymmetric slippage model |
| `src/persistence/bot_persistence.py` | Atomic + journaled state persistence |
| `src/persistence/trade_log_reader.py` | Centralized trade_log.jsonl reader |
| `src/monitoring/hud_tabbed.py` | 7-tab terminal HUD |
| `src/monitoring/audit_logger.py` | `DecisionLogger` → `logs/audit/decisions.jsonl` |
| `src/training/offline_trainer.py` | Walk-forward DDQN training on historical bars |
| `src/risk/path_geometry.py` | 5 entry-quality features (efficiency, gamma, jerk, runway, feasibility) |
| `src/features/event_time_features.py` | Session/rollover/week event features (6 broadcast dims) |
| `run_universe.py` | Supervisor launching per-symbol hubs, weight sync |
| `scripts/performance_analyzer.py` | Self-healing fleet analyzer — runs every 4h, applies corrections via `LearnedParametersManager` |
| `scripts/optuna_then_tournament.sh` | Chains Optuna HPO search then tournament promotion for a symbol/TF |
| `train_offline.py` | Offline tournament trainer (6 variants, auto-promote) |
| `run.sh` | Shell launcher (sources env + ROCm config) |

______________________________________________________________________

## Wire Scale Invariant (CRITICAL)

cTrader SpotEvent bid/ask are ALWAYS at 10^5 precision. `_scale = 100000` is fixed.
`digits` from `SymbolByIdRes` is display precision only — never use it to set `_scale`.

______________________________________________________________________

## Feature pipeline (offline, paper, live — all aligned)

All three modes now use identical feature dimensions:

| Group | Count | Features |
| ------------------- | ------ | ------------------------------------------------------------------ |
| Base | 7 | ret1, ret5, ma_diff, vol, imbalance, vpin_z, depth_ratio |
| Geometry | 5 | efficiency, gamma, jerk, runway, feasibility (PathGeometry) |
| Event | 6 | london/ny/tokyo_active, overlap, rollover_proximity, week_progress |
| **Trigger total** | **18** | base + geometry + event |
| **Harvester total** | **21** | trigger features + MFE + MAE + bars_held |

Offline trainer extracts event features from bar timestamps; geometry from bar closes + realized vol.

## Weight format

All weights saved as `.pt` files via `ddqn_network.save_weights()`:\
`{"online": state_dict, "target": state_dict, "optimizer": state_dict, "training_steps": int}`\
Load via `ddqn_network.load_weights()` which handles both `.pt` and legacy `.npz`.

______________________________________________________________________

## AMD ROCm GPU Optimizations

The system auto-detects AMD GPUs (gfx1100/gfx1102/Navi 31/33) at startup and applies:

| Optimization | Impact | File |
| ------------------------- | --------------------------------------------------- | -------------------------------- |
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
1. `src/core/ddqn_network.py` → `_get_amd_optimizations()` detects GPU
1. BF16 enabled automatically for RDNA 3 (native BF16 support)
1. Experience buffer uses float16 storage on AMD GPUs

**Manual override:**

```python
# Force BF16 on/off in DDQNNetwork
net = DDQNNetwork(state_dim=64, n_actions=3, use_bf16=True)

# Force float16 storage in ExperienceBuffer
buf = ExperienceBuffer(capacity=50000, use_float16=True)
```

______________________________________________________________________

## Profitability safeguards

- **R-multiple max-loss cap** — hard per-trade cap = `5 × 1R` (position's own expected stop), clamped to [$2, $200]. Computed from `entry_price × STOP_LOSS_PCT_DEFAULT/100 × qty × contract_size × MAX_LOSS_MULT_PER_TRADE`. XAUUSD≈$66, BTCUSD≈$23. Constants: `MAX_LOSS_MULT_PER_TRADE=5.0`, `MIN_CAP_USD=2.0`, `MAX_CAP_USD=200.0`. Replaces the old fixed `MAX_LOSS_PER_TRADE_USD=100.0`.
- **R:R profit floor** — when `MFE ≥ 1R`, trailing floor = `MFE − 1R` (protects captured 1R below peak)
- **Max-loss runs on EVERY tick regardless of pending-close status** (DID-1 fix — removed pending-close exemption)
- **Bar-close max-loss check runs BEFORE pending-close early return** (DID-2 fix)
- **Harvester ML exception escalates to max-loss fallback** (DID-3 fix)
- **Pending-close staleness timeout: 30 seconds** (DID-4 fix — was 120s)
- Duplicate fill guard in `trade_manager.py` — prevents paper fill + broker fill race condition
- `GHOST_RECONCILE_COOLDOWN_BARS = 3` — blocks entry for 3 bars after ghost position reconciliation
- Protective stops (trailing, breakeven, capture decay, micro-winner) run **BEFORE** DDQN decision path (never bypassed — fixed Mar 2026)
- Kurtosis uses a single action threshold path (circuit-breaker threshold, default 5.0); 3.0 remains alert/telemetry level

## Reward monitor scope

- `RewardShapingMonitor` operates per `symbol + timeframe + broker` scope.
- Rolling quality comparison is exported per bot:
  - short window default `24h`
  - baselines default `7d` and `30d`
- Quality-guard recommendations may adjust participation/selectivity and reward weights when 24h metrics materially degrade versus baselines.

______________________________________________________________________

## Decision log architecture (TWO logs)

| Log | Path | Format | Writer | Use |
| ----------------------------- | ---------------------------- | ------------------------- | ------------------------- | ---------------------------------------------------------------------------------------- |
| **Audit log** (primary) | `logs/audit/decisions.jsonl` | Append-only JSONL | `DecisionLogger` | Rich: session_id, agent, decision, confidence, context, reasoning, trade_id, position_id |
| **Bar-close log** (secondary) | `data/decision_log.json` | JSON list, full overwrite | `_obc_write_decision_log` | OHLC state + bars_held at every bar close; session field added |

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

______________________________________________________________________

## HUD tab map

| Key | Tab | Key data |
| --- | ------------- | -------------------------------------------------------------------- |
| 1 | Overview | fleet status, position, account balance, risk status, market |
| 2 | Performance | period metrics (24h/7d/Mo/All), edge quality, prediction convergence |
| 3 | Training | offline jobs, per-agent steps/loss/reward with trend arrows |
| 4 | Risk | VaR, circuit breakers, regime ζ, reward weights, path geometry |
| 5 | Market | spread, L2 ladder, VPIN-z, imbalance, signal synthesis |
| 6 | Decision Log | `MM-DD HH:MM` timestamps, TrdID column, session-break separators |
| 7 | Trade History | paginated list with mode badge (P/L), drill-down detail |

## Runway-delta adaptation (point-unit contract)

- Adaptive runway correction is driven by `runway_delta_ema` and applied at close attribution time.
- Safety clamps must remain enforced:
  - `RUNWAY_BIAS_LIMIT_POINTS` caps absolute bias correction (points)
  - `RUNWAY_ADJUST_MIN_SCALE` / `RUNWAY_ADJUST_MAX_SCALE` clamp adaptive scale
- Keep all runway adaptation math in point units end-to-end when populating trade attribution and persisted logs.
- Runway-friction gate uses multiplier **0.0 by default** (was 1.5, caused trade starvation Mar 2026); env override available.

### HUD convergence runway precedence

When reading trade records for convergence, HUD should resolve runway points in this order:

1. `predicted_runway_net_points` (adjusted)
1. `predicted_runway_net_points_raw` (raw)
1. legacy fallback: `predicted_runway * entry_price`

## HUD keyboard shortcuts

| Key | Action |
| ------- | ------------------------------------------------------- |
| `1`-`7` | Switch to tab |
| `Tab` | Cycle forward; `Shift+Tab` backward |
| `←`/`→` | Cycle tabs left/right |
| `s` | Select symbol/timeframe preset |
| `r` | Review & reset tripped circuit breakers |
| `e` | Set/clear stats epoch (exclude old trades from metrics) |
| `h` | Help screen |
| `Alt+K` | Emergency kill switch (close all + halt) |
| `q` | Quit HUD (bot keeps running) |

### Stats epoch (`[e]` key)

Configurable cutoff date stored in `data/stats_epoch.json`. Trades before the epoch are excluded from all Performance tab metrics (period rows, mode breakdown, trade quality, edge quality) but the raw `trade_log.jsonl` is never modified. Useful for excluding old losing periods that drag down current performance assessment.

______________________________________________________________________

## Operating the paper-bot fleet

The paper-trading workload is a **fleet of per-timeframe bots** supervised by a single watcher. Each entry in `data/universe.json` (a list under `instruments`) becomes a dedicated `src.core.openapi_hub.TFAgent` process with an isolated session directory and log file (`logs/paper_<SYMBOL>_M<TF>.log`).

| Action | Command |
| ------------------------------ | -------------------------------------------------------------- |
| Start / restart whole fleet | `./run.sh universe` |
| Show running bots + watcher | `./run.sh status` |
| Kill everything | `pkill -f run_universe ; pkill -f openapi_hub` |
| Attach HUD to running fleet | `./run.sh --hud-only` (interactive terminal required) |
| Manually promote an instrument | `python3 run_universe.py --promote <SYMBOL> --timeframe <MIN>` |

Watcher semantics:

- Polls every 30 s, re-launches any bot whose PID disappeared, clears stale `paper_pid` fields.
- Adds new instruments from `data/universe.json` as they reach `stage: PAPER`.
- Writes supervisor logs to `logs/run_universe.log`.
- Runs bots with `start_new_session=True` so HUD/terminal signals do not propagate.
- Syncs promoted weights from `data/universe.json` into isolated runtime checkpoint directories.
- Restarts a running bot if its runtime weights are stale.
- Runs `scripts/performance_analyzer.py` every **480 cycles (4 h)** with `--auto-heal`, writing `data/performance_health.json`.

Agent caveats:

- The HUD (`src.monitoring.hud_tabbed`) is an **interactive terminal UI** — it cannot be rendered from a non-interactive agent shell. Summarise from `./run.sh status`, `logs/paper_*.log`, and `data/universe.json` instead of trying to launch it in the background.
- Do **not** edit `data/universe.json` as a dict — the canonical schema is `{"version": 1, "instruments": [ {...}, ... ]}` (list of entries). Any status/diagnostic helper must iterate the list.
- Before stopping bots for a hotfix, prefer targeted `pkill -f "paper_<SYMBOL>_M<TF>"` when only one timeframe needs recycling; the watcher will relaunch it on the next poll.
- Treat `/tmp/ctrader_hud.pid` as possibly stale when restarting the HUD.

______________________________________________________________________

## TFAgent — MFE/MAE Tracking v2 Features (openapi_hub.py)

### trade_id threading

- `_current_trade_id` set to `f"{symbol}_{tf_label}_{uuid.uuid4().hex[:8]}"` on position open
- Cleared to `None` at **top** of `_close_position()` (before processing — prevents ghost HOLDs)
- Threaded through: LONG/SHORT decision log → every HOLD audit entry → CLOSE audit entry → trade_log record

### Ghost HOLD Race Condition (fixed 2026-04-27)

**Bug**: `self.position = None` and `self._current_trade_id = None` were cleared near the END of `_close_position()` (~150 lines in), after reward/ML processing. Any exception left the position live, and the next tick emitted a HOLD with the stale trade_id.

**Fix**: Both clears now happen immediately after snapshotting `pos = self.position` at the TOP of `_close_position()`. Pattern: **snapshot → release → process**.

**Audit log**: `tests/validation/test_decision_log_correlation.py` uses cutoff `2026-04-17T00:00:00+00:00` to skip pre-fix violations.

### HOLD audit logging

`_add_harvester_hold_experience()` writes a HOLD entry to the decision log on every bar while in position.
`DecisionLogger.log_harvester_decision()` guards against `in_position=False` calls (suppresses them).

### Dynamic entry floor

`_compute_dynamic_entry_floor(base_floor)` returns `max(base + cal_uplift + runway_penalty, rl_floor_capped)`.
Applied in `_handle_flat()` after `decide_entry()`. Parameters from `LearnedParametersManager`.

### Exit confidence floor

`_exit_conf_dynamic_floor` applied in `_handle_exit_on_tick()` — blocks exits below the floor.

### Risk feedback thresholds

`_update_risk_feedback_thresholds(pnl_usd)` — win-rate EMA (α=0.15); raises entry floor +0.02 if win_rate < 40%, lowers −0.01 if > 65%. Min-trades guard per TF before kicks in.

### 4-component trigger reward

`_calculate_trigger_reward()` — accuracy + magnitude_bonus − false_positive_penalty − toxic_flow_penalty, clipped to [−1.5, 1.5]. Regime-adjusted capture reward in `_close_position()`.

### Preseed / bars cache

`_PRESEED_STOP_PCT=0.003`, `_PRESEED_TARGET_PCT=0.002`, `_PRESEED_MAX_HOLD=20`.
Last 500 bars persisted to `bars_cache.json` every 10 bars; warm-start preseed on load.

### CB restore at startup

`circuit_breakers.restore_state("data/circuit_breakers.json")` called in `__init__` after `set_emergency_closer`.

### Adaptive regularization TD feedback

In `_maybe_train()`: if `avg_td > 0.5` → `increase_regularization()`; if `< 0.1` → `decrease_regularization()`.

______________________________________________________________________

## Audit Log & Trade Log (Apr 2026 expansion)

The TFAgent writes **65 top-level fields** per trade to `data/trade_log.jsonl` plus
two nested breakdown dicts (`trigger_data` with 34 sub-fields, `reward_*_breakdown`).

The complete field map is defined in `src/core/openapi_hub.py:_write_trade_log()`.
Key groups: identity (7), timing (3), P&L (4), excursions (4), entry conditions (13),
runway prediction (9), reward (7), calibration (7), exit conditions (3), diagnostics (4),
risk state (2), trigger reason snapshot (1 nested dict), reward breakdown (2 nested dicts).

**Every trade is now linked to its trigger entry context.** The `trigger_data` field
captures regime, geometry, HMM probabilities, kurtosis, volatility ratio, gap, returns,
alignment score, bar OHLCV, training state, CB state, drawdown, `entry_confidence`, and
`entry_vpin_z` at the moment of entry.

**`close_reason` is always populated.** `shutdown()` sets `"shutdown"` and
`_PaperEmergencyCloser.close_all_positions()` sets `"circuit_breaker"` before calling
`_close_position()`. No more blank close_reason records.

For retrospective analysis, use `scripts/reconstruct_trade_lifecycle.py` to stitch
trade_log + decisions + cache + transactions + CSV history into a single enriched dataset.

## Real-Data Test Requirements

All price-based tests should use the session-scoped fixtures from `tests/conftest.py`
which load real paper-trading data (`data/training_cache_XAUUSD_M5.jsonl`,
`data/training_cache_XAUUSD_M1.jsonl`, `data/training_cache_BTCUSD_M1.jsonl`).
Tests using these fixtures are automatically skipped if the live cache has insufficient
data (≥200 bars, ≥10 trades required).

After modifying `openapi_hub.py` P&L or trade log paths, run:

```bash
python3 -m pytest tests/unit/test_openapi_hub_pnl.py -v --tb=short
```

After modifying reward shaper or metrics_calculator:

```bash
python3 -m pytest tests/unit/test_metrics_calculator.py tests/unit/test_reward_calculations.py -v
```

______________________________________________________________________

## Harvester Exit Thresholds (src/constants.py)

Cold-start defaults — overridden at runtime by `LearnedParametersManager`. These are the correct
production values restored after the scalping bug fix (2026-04-26). Do NOT halve these again.

```python
BREAKEVEN_TRIGGER_PCT          = 0.30   # MFE % to move stop to breakeven
TRAILING_STOP_ACTIVATION_PCT   = 0.25   # MFE % to activate trailing stop
TRAILING_STOP_DISTANCE_PCT     = 0.12   # Distance to trail behind peak MFE
CAPTURE_DECAY_THRESHOLD        = 0.35   # Exit if current_profit/MFE < this
CAPTURE_DECAY_MIN_MFE_PCT      = 0.10   # Arm capture-decay only above this MFE %
MICRO_WINNER_MFE_THRESHOLD_PCT = 0.10   # Min MFE to activate micro-winner guard
MICRO_WINNER_GIVEBACK_PCT      = 0.40   # Exit if giveback > this fraction of MFE
```

Emergency reset (`_apply_capture_emergency_reset` in openapi_hub.py) applies timeframe scale:

- `trailing_stop_activation_pct = TRAILING_STOP_ACTIVATION_PCT * tf_scale * 0.50`
- `capture_decay_min_mfe_pct    = CAPTURE_DECAY_MIN_MFE_PCT * tf_scale` ← no extra 0.50×
- `capture_decay_threshold      = 0.50`

______________________________________________________________________

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

### Protective stops priority (harvester decide())

1. Emergency SL (even during min-hold)
1. **Protective stops** (trailing, breakeven, capture_decay, micro-winner) — runs BEFORE min-hold gate
1. Min-hold check
1. Hard time stop
1. Soft time stop
1. DDQN / torch / fallback

______________________________________________________________________

## Testing requirements

**Never run `python -m pytest` without a target.** ~2,440 tests across 104 files will OOM-kill (exit 137).

Always target a subset:

```bash
# Single file
python -m pytest tests/unit/test_risk_manager.py -v

# Directory with 2 parallel workers (safe memory limit)
python -m pytest tests/unit/ -n 2 --dist loadfile

# Integration tests (run alone — heavier)
python -m pytest tests/integration/ -v

# Top-level tests
python -m pytest tests/test_*.py -v

# Specific keyword
python -m pytest -k "risk_manager" -v
```

**Avoid DNS-hanging test files** — these spin up HTTPServer that calls `socket.getfqdn()`:

- `tests/unit/test_production_monitor_http.py`
- `tests/unit/test_production_monitor_extended.py`
- `tests/unit/test_prometheus_metrics.py`

Exclude them when doing broad sweeps:

```bash
python -m pytest tests/unit/ --ignore=tests/unit/test_production_monitor_http.py \
  --ignore=tests/unit/test_production_monitor_extended.py \
  --ignore=tests/unit/test_prometheus_metrics.py --timeout=20 -q
```

**Known caveat**: runway-correlation validation can be environment/log-data dependent; treat as a data-quality check when log completeness differs.

After modifying reward shaper dims: run `tests/unit/test_reward_calculations.py`
After modifying IS weights: run `tests/unit/test_experience_buffer.py`
After modifying risk manager: run `tests/validation/test_risk_manager.py`
After modifying run_universe.py or train_offline.py:

```bash
pytest -q tests/unit/test_run_universe.py tests/unit/test_offline_training.py
ruff check run_universe.py train_offline.py tests/unit/test_run_universe.py tests/unit/test_offline_training.py --select E,F,PLR0124,PLW0108,PLR0912,PLR0915
python3 -m py_compile run_universe.py train_offline.py
```

______________________________________________________________________

## Offline Training

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 train_offline.py \
  data/history/XAUUSD_M1.csv ... data/training_cache_XAUUSD_M1.jsonl ... \
  --symbols XAUUSD --workers 2 --n-epochs 3 --warm-start \
  --accept-if-better --auto-promote --retrain-rounds 6 \
  --paper-threshold 1.0 --tournament-variants 6
```

XAUUSD M15/M30/M60 offline training skipped when live cache < 50 rows — needs more paper-trading time.

### Optuna HPO mode

Replace `--tournament-variants` with `--optuna-trials N` for Bayesian HPO:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 train_offline.py \
  data/history/XAUUSD_M5.csv data/training_cache_XAUUSD_M5.jsonl \
  --symbols XAUUSD --workers 2 --optuna-trials 20 \
  --accept-if-better --auto-promote --paper-threshold 1.0
```

- Each `(symbol, timeframe)` gets an isolated SQLite study: `data/optuna/offline_SYMBOL_MTF.db` — studies resume across restarts
- Objective: ZΩ + val PF + val net PnL with trade-shortfall penalty
- Promotion still routes through the incumbent/champion acceptance guard unchanged
- `scripts/optuna_then_tournament.sh` chains Optuna search + tournament promotion

Offline champion acceptance order (Apr 2026 fix):

1. `data/checkpoints/offline_champions.json`
1. `data/universe.json` for the default checkpoint root

**Historical logs are NEVER acceptance guards.** Never scrape `logs/train_offline.log` to decide acceptance.

Run `run_universe.py` syncs promoted weights from `data/universe.json` into isolated runtime checkpoint directories and restarts stale bots.

______________________________________________________________________

## Downloading History Data

Credentials in `.env.openapi` lack `export` — must load with `set -a`:

```bash
cd /home/renierdejager/Projects/Kinetra && set -a && source .env.openapi && set +a
cd /home/renierdejager/Projects/ctrader_trading_bot
python3 scripts/download_ctrader_history.py \
  --symbol XAUUSD --start-date 2024-01-01 \
  --timeframes M1 M5 M15 M30 M60 M240 \
  --account-id 45841299 \
  --out-dir data/history
```

For BTCUSD: batch ≤2 TFs per invocation — demo server drops TCP after ~5 min on BTC-volume downloads.

`ProtoOATrendbar` field names: `b.low` (absolute base, raw ticks), `b.deltaOpen`, `b.deltaHigh`, `b.deltaClose` (deltas from low). Divide all by `_WIRE_SCALE = 100_000` for scaled price.

______________________________________________________________________

## Current Universe State (as of 2026-04-29)

| Symbol | TF | ZΩ | Status |
| ------ | ----------- | ------- | --------------------------------------------- |
| XAUUSD | M1 | 3.159 | Promoted, active |
| XAUUSD | M5 | 1.663 | Promoted, active |
| XAUUSD | M60 | 1.428 | Promoted, active |
| XAUUSD | M240 | 1.601 | Promoted, active |
| XAUUSD | M15, M30 | — | Untrained — no weights (needs >50 live cache rows) |
| BTCUSD | M30 | 1.103 | Promoted, active |
| BTCUSD | M60 | 1.065 | Promoted, active |
| BTCUSD | M240 | 1.057 | Promoted, active |
| BTCUSD | M1, M5, M15 | — | Untrained — offline training in progress |

Both hubs (`hub_XAUUSD.log`, `hub_BTCUSD.log`) active. 7 of 12 bots have promoted weights; 5 untrained. Offline training runs periodically with `--workers 2 --tournament-variants 6`. Weights auto-promoted to `data/universe.json` when ZΩ > 1.0.

**HUD pipeline card ZΩ display**: untrained bots (`z_omega=0.0`, no `weights_path`) show `ZΩ —` (dim) not red `ZΩ 0.0000`. Check `_no_weights = not entry.get("weights_path")` before colouring.

History data: BTCUSD M1=1.18M, M5=237K, M15=79K, M30=39K, M60=19K, M240=5K bars.
XAUUSD M1=816K, M5=163K, M15=54K, M30=27K, M60=13K, M240=3.5K bars (all Jan 2024–Apr 2026).

______________________________________________________________________

## FIX Gateway Topology Migration — COMPLETE

The `openapi-hub` topology is the default (`UNIVERSE_BROKER_TOPOLOGY=openapi-hub`).
`run_universe.py` launches one hub per symbol via `_launch_hub()` — each hub covers all
timeframes for its symbol in a single process. Legacy FIX `isolated` mode still supported.

Modes: `isolated`, `shared-symbol`, `shared-account`, `openapi-hub` (default)

______________________________________________________________________

## Current open items (as of 2026-04-29)

| Item | Priority | Notes |
| ----------------------------------- | -------- | -------------------------------------------------------------------- |
| Harvester train_batch shape mismatch | HIGH | `mat1 (64×1152) × mat2 (1344×128)` — model expects 18 features but gets 21. Happens ~once per bar-close batch. |
| HUD rendering tests (16 failing) | MEDIUM | `test_hud_rendering.py` table alignment regexes need updating for new Comment column header in offline jobs table. |
| HUD timeframe tests (6 failing) | MEDIUM | `test_hud_timeframe_metrics.py` — same Comment column root cause. |
| Problems tab (294 remaining) | LOW | All style/convention: docstrings, type hints, line length, cognitive complexity. Zero bugs. |

______________________________________________________________________

## Deleted modules (do NOT recreate)

These were removed as dead code and fully deleted from the active repository. Do not recreate or reference them:
`agent_arena.py`, `cold_start_manager.py`, `early_stopping.py`, `ensemble_tracker.py`,
`feedback_loop_breaker.py`, `generalization_monitor.py`, `parameter_staleness.py`,
`feature_tournament.py`, `time_features.py`, `risk_aware_sac_manager.py`

See `docs/archive/REMOVED_LEGACY_CODE.md` for the removal manifest.

______________________________________________________________________

## Paper → Live roadmap

**Current phase:** Paper trading only — focus on reliable profitability first.

When paper is profitable, the plan is to run paper (challenger) + live (champion) side-by-side with weekend weight promotion. Architecture is ~80% ready:

**Already configurable:** FIX config paths (env vars), credentials (env vars), checkpoint dir (parameter), DDQN weight paths (any path accepted).

**Needs implementation:** Parameterize `hud_data_dir` via `BOT_DATA_DIR` env var, mode-suffix trade/decision logs, plumb `LearnedParametersManager` path, create live FIX configs with separate `SenderCompID`, `scripts/promote_weights.py` with validation gate (paper Sharpe > live), extend `run_universe.py` for paper+live of same instrument.

See `docs/CURRENT_STATE.md` § "Paper → Live Roadmap" for full readiness matrix and implementation plan.

______________________________________________________________________

## SonarQube MCP Usage

- After finishing code modifications, call SonarQube analysis on changed files
- Project keys: look up via `search_my_sonarqube_projects` — don't guess
- Use USER tokens, not project tokens
- Disable automatic analysis at task start, re-enable when done

______________________________________________________________________

## Recent Fixes (Apr 27–29, 2026, continued)

| Fix | Problem | Solution |
| --- | ------- | -------- |
| Buffer save crash | `np.array()` fails on mixed Trigger(18,64) + Harvester(21,64) states | `.ravel()` before `np.savez_compressed()`; load handles both flat and legacy 2D |
| Harvester preseed | State shape mismatch (18→21 features after event dims added) | Use `_build_full_state()` instead of raw buffer slicing |
| Paper-mode gating | Depth gate + dynamic confidence floor blocked entries before ε-greedy exploration | In paper mode: log the condition, add to `_gated`, but don't block |
| `_write_trade_log` TypeError | `entry_time` passed as `str` instead of `datetime` | Convert with `datetime.fromisoformat()` |
| Ruff unsafe-fix damage | `--fix --unsafe-fixes` deleted 446 `print()` calls + variable assignments from HUD | Restored all damaged methods (6 in `hud_tabbed.py`) |
| `_render_order_book_ladder` | f-strings constructed but never printed (no-op) | Now prints bid/ask rows with correct price precision, uses all params |
| Offline training restart | Stale progress files blocked retraining | Supervisor detects stalled jobs, clears stale metadata before relaunch |
| `_tf_label` H4→M240 | Labels showed "H4" instead of "M240" in HUD and status files | `train_offline.py:_tf_label()` now returns M240 per project convention |
| HUD pipeline ZΩ display | Untrained bots (`z_omega=0.0`, no weights) shown as red `ZΩ 0.0000` | `_render_pipeline_card`: check `not entry.get("weights_path")` → show `ZΩ —` |
| HUD trade quantity | `_excursion_usd_for_trade` read `qty` key (missing) instead of `quantity` | Fallback: `trade.get("quantity") or trade.get("qty", 0.1)` |
| HUD offline enrichment | "done" offline entries missing ZΩ (zo=None) for bots not in current training run | `_enrich_offline_stats_from_champions()` fills ZΩ from `offline_champions.json` |
| AMD buffer wiring | `get_amd_optimized_buffer_capacity()` existed but was never passed to `DualPolicyConfig` — runtime used 2k/10k defaults | Hub now imports and calls `get_amd_optimized_buffer_capacity()` for both agents (50k on AMD) |
| HUD buffer denominators | `_RT_TRIG_CAP`/`_RT_HARV_CAP` hardcoded to 2000/10000 — fill bars showed wrong scale | Module-level constants now call `get_amd_optimized_buffer_capacity()` at import time |
| Reward gradient collapse | 73% of trigger experiences at ±3.0 rail (log-runway reward saturates when predictor uncalibrated); PnL signal at 0.21 effective weight; timing penalty ×1.0 too aggressive | Trigger: fallback to 4-component reward when shaped reward is at rail (abs ≥ 2.99). PnL alignment weight 0.6→1.2, multiplier 0.35→1.5 (effective 0.21→1.80). Activity bonus 0.8→0.2. Timing penalty -1.0→-0.4. |
| Fixed $100 max-loss cap | Instrument-blind: 5.4×R for XAUUSD, 32.5×R for BTCUSD (practically never fired on BTC) | Replaced with 5×R dynamic cap (entry × stop% × lot). XAUUSD≈$66, BTCUSD≈$23. R:R floor at 1R MFE. |
| Buffer save mixed-dim crash | `np.array()` raised inhomogeneous shape when circular buffer mixed offline-training (old dim) + paper-trading (new dim) experiences — save failed silently every checkpoint | `save()` now filters to canonical state size (first entry's flat size) before stacking, matching the filter `load()` already had. Regression tests: `TestSaveLoad::test_save_survives_mixed_state_dims`. |
| Blank `close_reason` records | `shutdown()` and `_PaperEmergencyCloser.close_all_positions()` called `_close_position()` without setting `harvester.last_close_reason` → blank field in trade log | Both paths now set `"shutdown"` / `"circuit_breaker"` on `harvester.last_close_reason` before calling `_close_position()`. |
| Missing `trigger_data` fields | `entry_confidence` and `entry_vpin_z` existed only at top level of trade_log — absent from `trigger_data` sub-dict, so entry context was incomplete for reward shaping analysis | Added both fields to `_entry_trigger_data` dict in `_handle_flat()`; `trigger_data` grows from 32 → 34 sub-fields. |
| Self-healing loop | No automatic detection or correction of fleet anomalies (DDQN win-rate collapse, runway miscalibration, excessive emergency stops) | Added `scripts/performance_analyzer.py` (8 anomaly codes, auto-corrections via `LearnedParametersManager`); supervisor runs it every 4 h with `--auto-heal`. |
| HUD self-heal panel | No visibility into whether the analyzer was running or finding issues | `_render_health_analyzer()` added to Overview SYSTEM HEALTH block — shows fleet health, anomaly codes, and last corrections from `data/performance_health.json`. |
| Epsilon hardcoding | Paper epsilon values (0.1 end, 0.9995 decay) duplicated across paper_mode.py, trigger_agent.py, dual_policy.py, run.sh — checkpoint reload could silently override paper floor | Centralised into `PAPER_EPSILON_*` / `LIVE_EPSILON_*` constants; `load_checkpoint()` preserves paper floor and never replaces paper decay with stale metadata. Paper floor raised 0.1→0.25. |
| Optuna HPO | Offline training only supported fixed tournament variants — no hyperparameter search | `--optuna-trials N` enables Bayesian HPO per bot with persistent SQLite studies; objective scores ZΩ + val PF + net PnL; promotion guard unchanged. |
| Reward monitor gaps | Quality guards did not suggest raising `reward_weight_pnl_alignment` when PF/PnL degraded | Two new suggestion calls added: one on PF+PnL drop, one when capture drops with flat/negative per-trade PnL. |

### What NOT to do (continued from above)

- **Never** run `ruff --unsafe-fixes` on `hud_tabbed.py` or any file with `print()`-based rendering. Ruff treats return-value-less function calls as dead code and strips them. Use `ruff check` (without `--fix`) for lint feedback, then fix manually.

______________________________________________________________________

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
- Do not reintroduce `H4` runtime path — use `M240` everywhere (files, cache keys, metrics, registry entries)
- Do not commit `data/`, `logs/`, `trades/`, `store/`, `.env`, credentials, model artifacts, or live runtime outputs
- Do not recreate deleted modules (see `docs/archive/REMOVED_LEGACY_CODE.md`)
- Do not aggregate across timeframes unless the UI/code path explicitly says it is a portfolio/account view
- Never run `ruff --unsafe-fixes` on `hud_tabbed.py` or any file with `print()`-based terminal rendering — ruff treats return-value-less function calls as dead code and strips them when they're actually the rendering output.
- Do not add hard gates in `_handle_flat()` that block entries before the trigger agent — paper mode needs ε-greedy exploration on ALL bars. Log and add to `_gated`, but let `trigger.decide_entry()` run.
