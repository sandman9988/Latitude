# Agent Instructions

These instructions are for coding agents working in this repository. Keep them
current whenever training, promotion, HUD telemetry, or runtime topology changes.

## Current Operating Model

- The production shape is a multi-timeframe XAUUSD paper fleet supervised by
  `run_universe.py --watch`.
- The canonical timeframe labels are `M1`, `M5`, `M15`, `M30`, `M60`, and
  `M240`. Do not introduce a shadow `H4` runtime path. Use `M240` for files,
  cache keys, metrics, and registry entries.
- Treat `(symbol, timeframe_minutes)` as the identity for metrics, learned
  parameters, caches, decision logs, offline training, reward-shaping monitor
  output, checkpoints, and HUD rows.
- Do not aggregate across timeframes unless the UI or code path explicitly says
  it is a portfolio/account view.

## Source Of Truth

- Learned parameters live in `LearnedParametersManager` and must be read with
  both symbol and timeframe.
- Runtime telemetry must prefer scoped files such as
  `data/paper_XAUUSD_M5/...` or `*_XAUUSD_M5.json` over root fallback files.
- Paper-training HUD/status timestamps come from the runtime training stats
  writer, which reads `DualPolicy.get_training_stats()`. Restored checkpoints
  must preserve a meaningful `last_training_time` from checkpoint metadata
  (`saved_at`, legacy `last_training_time`, or the metadata file mtime) whenever
  restored training steps are non-zero; do not regress resumed agents back to a
  misleading `Never` status.
- Offline champions are sourced from
  `data/checkpoints/offline_champions.json`, then from `data/universe.json` for
  the default checkpoint root.
- Historical logs are not a source of truth for champion ZOmega or promotion
  guards. Never scrape `logs/train_offline.log` to decide acceptance.
- `data/universe.json` records the promoted runtime weight paths. `run_universe.py`
  must sync those promoted weights into each isolated paper runtime checkpoint
  directory before launch and restart a running bot if the runtime weights are
  stale.

## Offline Training

- Weekend offline training should run per symbol/timeframe from the complete
  available cache set discovered under `data/training_cache_*_<TF>.jsonl` and
  per-bot cache directories.
- Completed offline-training rows in `data/offline_training_status.json` are
  resume guards. Restarted weekend runs must skip those `(symbol, timeframe)`
  rows unless `CTRADER_OFFLINE_RESUME_STATUS=0` is explicitly set.
- Long offline jobs write `data/offline_progress_<SYMBOL>_<TF>.json` and
  candidate resume weights under
  `data/checkpoints/<SYMBOL>_<TF>/<candidate_id>/`. Treat those files as the
  live progress/checkpoint source for the currently running job; do not infer a
  cold restart from a stale summary row alone.
- GPU hosts default offline training to one worker to avoid VRAM contention.
  To deliberately parallelize weekend jobs on the same GPU, use
  `ALLOW_GPU_PARALLEL=1 WORKERS=<N> ./run.sh train` or pass
  `--allow-gpu-parallel` to `train_offline.py`, then monitor VRAM and process
  health. CPU-only runs may use `--workers N` directly.
- `train_offline.py` must keep `data/offline_training_status.json`
  restartable while work is unfinished. The status file should include
  supervisor metadata (`pid`, `python`, `argv`, `cwd`, restart count, heartbeat)
  so `run_universe.py --watch` can recover queued/running work after watcher,
  Open API hub, shell, or host restarts.
- `run_universe.py --watch` owns offline-training reconciliation during normal
  operations. It must not launch duplicate training if a live `train_offline.py`
  process exists for this checkout, but it should restart missing or stalled
  unfinished queues until the status is complete. The default stall threshold is
  `UNIVERSE_OFFLINE_STALL_SECS=3600`; autorestart can be disabled with
  `UNIVERSE_OFFLINE_AUTORESTART=0`.
- On resume, completed `(symbol, timeframe_minutes)` entries in the offline
  status remain `done` and should not be retrained unless
  `CTRADER_OFFLINE_RESUME_STATUS=0` is set or the operator starts a fresh run.
- Legacy unfinished status files without supervisor metadata may be rebuilt from
  the status rows plus discovered scoped cache/history inputs, but new runs
  should always write explicit restart metadata.
- Tournament variants are evaluated per symbol/timeframe. Promote only the best
  accepted candidate for that exact pair.
- Optional Optuna search is an offline candidate generator only. It must run per
  `(symbol, timeframe_minutes)`, store studies under scoped files such as
  `data/optuna/offline_XAUUSD_M5.db`, and still route promotion through the
  incumbent/champion acceptance guard.
- Acceptance must beat the evaluated runtime incumbent and the registered
  champion/live universe guard, with the configured acceptance margin.
- If a candidate does not beat both guards, leave the current champion in place
  and run the configured retrain rounds instead of degrading the live pipeline.
- Focused replay should use the 10 best and 10 worst recent capture records per
  symbol/timeframe by default, without contaminating another timeframe.

## Runtime And HUD Rules

- The HUD uses a single global trading-context hierarchy shared by all seven
  tabs: **Portfolio → Symbol → Symbol/TF → Detail** (Levels 1–4).
  Tabs are analytical lenses over the same context, not independent hierarchies.
- The default start level is Portfolio (Level 1). There is no Level 0 mode gate.
  Mode is per-instrument: XAUUSD can be Live while BTCUSD is Paper. Level 1 rows
  are `(symbol, mode)` pairs. On drill, both symbol AND mode are captured from the
  selected row. See `docs/HUD_REDESIGN.md` for the full specification.
- Standard performance periods are `24h`, `7d`, `Month`, `Epoch`, and
  `Lifetime`. Periods are shown as **columns side-by-side** at Levels 1–3
  so the operator can compare across periods without cycling. The `p` key
  cycles the period focus at Level 4 only.
- HUD navigation is consistent across ALL tabs: `Enter` drills down, `Esc`
  drills up, `↑`/`↓` or `j`/`k` moves the cursor, `d` toggles the detail pane.
  `s` jump-scopes Portfolio → Symbol → Symbol/TF → Portfolio.
- HUD result rows must make mode explicit: `Paper` or `Live`. Offline
  training/backtest/champion metrics must be visually separated from
  paper/live account performance so validation results are not confused with
  realized trading PnL.
- Do not mix paper, live, and offline results into one HUD metric row. Compare
  them side by side when useful, but keep their metrics separate.
- `decisions.jsonl` is ~90% CACHED startup entries. All HUD code reading this
  file uses `_tail_meaningful()` which scans backward skipping CACHED/WARMING_UP entries.
  Tab 6 (Decision Log) is fully hierarchical: L1 portfolio→L2 symbol→L3 cursor list→L4 detail
  card. Do not regress this to a flat view. See `docs/HUD_REDESIGN.md`.
- `gated_conditions` (array of gate-rejection reason strings on NO_ENTRY decisions)
  is rendered in Tab 6 Level 3 (inline badge + per-gate expansion) and Level 4 detail
  card (all gates, uncapped), and in Tab 4 Level 3 risk detail. Do not remove this.
- `transactions.jsonl` is routed to Tab 1 (`🔌 SESSION LOG` in system health block —
  per-bot last SESSION_START age, COMPONENT_HEALTH, SESSION_EVENT entries) and to Tab 7
  trade card L4 (`BROKER EVENTS` — POSITION_OPEN/CLOSE looked up by `position_id`).
- Decision logs must include timeframe and symbol, and HUD tabs must render
  timeframe wherever decisions, gates, circuit breakers, training status,
  reward-shaping advice, or cache freshness are shown.
- Spread, system health, risk metrics, kurtosis gates, no-entry behavior,
  reward-shaping diagnostics, runway prediction, trigger metrics, harvester
  metrics, and training stats are per symbol/timeframe unless explicitly shown
  as an account-level summary.
- Circuit breaker thresholds come from the learned per-symbol/timeframe path.
  The kurtosis action threshold is the learned/runtime action threshold, not a
  universal legacy `3.0` alert threshold.
- Paper-training exploration should stay deliberately high: default
  `EPSILON_START=1.0`, `EPSILON_END=0.25`, `EPSILON_DECAY=0.9998`, and
  `FORCE_EXPLORATION=1`. Stale checkpoint metadata must not lower the paper
  exploration floor or speed up paper epsilon decay.
- `DualPolicy.save_checkpoint()` writes `saved_at` into
  `data/checkpoints/<SYMBOL>_<TF>/training_metadata.json`. On restore,
  `DualPolicy._ckpt_load_metadata()` uses that timestamp to repopulate agent
  `last_training_time` after non-zero training steps are restored, so
  `data/training_stats_<SYMBOL>_M<TF>.json` and HUD training panels can
  distinguish resumed learning from a cold start.
- Multiple simultaneously running bots need a portfolio/account view before
  making account-level exposure decisions. Per-timeframe bots may learn
  independently, but order ownership and exposure should be reconciled through
  the broker/account source of truth.
- Any change to training, promotion, reward shaping, trade logging, risk,
  decision logging, learned parameters, runtime metrics, or self-healing
  telemetry must include a HUD impact check in the same change set. Update
  `src/monitoring/hud_tabbed.py`, HUD render tests, and user-facing labels when
  field names, units, periods, scopes, or source-of-truth paths change.
- HUD rows must use consistent names, units, column widths, clipping, and
  alignment. Do not leave deprecated or ambiguous metric pathways visible as
  current operator signals.
- HUD period summary cells must keep fixed visible width. Format large trade
  counts and PnL with compact suffixes instead of truncating commas or letting
  colored values push adjacent period columns out of alignment.
- See `docs/HUD_REDESIGN.md` for the target HUD information architecture and
  drill-down model.

## Operational Safety

- Do not commit `data/`, `logs/`, `trades/`, `store/`, `.env`, credentials,
  model artifacts, or live runtime outputs.
- Before major or risky changes, create an explicit rollback checkpoint first
  (for example an annotated `checkpoint/<name>` tag or clearly named branch).
  Record the pre-change and post-change commit IDs so the operator can quickly
  inspect, revert, or reset if the change causes runtime/HUD issues.
- Prefer targeted tests for touched subsystems before committing. For training
  and supervisor changes, run:

```bash
pytest -q tests/unit/test_run_universe.py tests/unit/test_offline_training.py
ruff check run_universe.py train_offline.py tests/unit/test_run_universe.py tests/unit/test_offline_training.py --select E,F,PLR0124,PLW0108,PLR0912,PLR0915
python3 -m py_compile run_universe.py train_offline.py
```

- When restarting live paper processes, verify flat/current position state first
  unless the user has explicitly accepted weekend/market-closed restart risk.
- Treat `/tmp/ctrader_hud.pid` as possibly stale when restarting the HUD.
- `./run.sh universe` performs a repo-scoped clean restart: it stops stale
  launcher shells, waits for existing `run_universe.py --watch` supervisors to
  exit, stops tracked/orphan paper bots, then verifies exactly one watcher is
  running after relaunch. `run_universe.py --watch` also holds
  `data/run_universe.watch.lock`; a second watcher for this checkout must exit
  instead of supervising the same universe concurrently.
- If stale circuit-breaker state must be cleared as part of a restart, run the
  launcher with `UNIVERSE_FIX_CB_LOCKOUT_ON_RESTART=1 ./run.sh universe` so
  old hubs are stopped before `scripts/fix_cb_lockout.py` edits CB files. Do
  not run the fix before stopping stale hubs, because old hub processes can
  save their in-memory breaker state back over the repaired files on shutdown.
- The performance analyzer may now self-heal a detected scoped `CB_LOCKOUT`
  without a full fleet restart by writing
  `data/paper_<SYMBOL>_<TF>/circuit_breaker_reset.json` with
  `target_timeframes`, then writing scoped `learned_parameters_reload.json`
  files for any normalized runaway gates. The OpenAPI hub must honor
  `target_timeframes` and reset only matching in-process agents.

## Audit Log & Trade Log

The TFAgent writes **66 top-level fields** per trade to `data/trade_log.jsonl` plus
nested lifecycle/breakdown dicts (`trigger_data`, `exit_data`, and `reward_*_breakdown`).

The complete field map is defined in `src/core/openapi_hub.py:_write_trade_log()`.
Key groups: identity (7), timing (3), P&L (4), excursions (4), entry conditions (13),
runway prediction (9), reward (7), calibration (7), exit conditions (3), diagnostics (4),
risk state (2), trigger reason snapshot (1 nested dict), exit reason snapshot (1 nested dict),
reward breakdown (2 nested dicts).
Excursion fields are unit-specific: `mfe` and `mae` are account-currency values
after applying `quantity * contract_size`; `mfe_points` and `mae_points` are raw
price movement. HUD rendering must not treat price points as dollars.

**Every trade is now linked to its trigger entry context.** The `trigger_data` field
captures regime, geometry, HMM probabilities, kurtosis, volatility ratio, gap, returns,
alignment score, L2 depth bid/ask, top-10 bid/ask book snapshot, real-size availability,
bar OHLCV, training state, CB state, and drawdown at the moment of entry.
The `exit_data` field captures exit confidence/floor, close reason, trailing-stop,
breakeven and capture-decay state, close-time regime/risk, spread, excursions,
capture, L2 depth bid/ask plus top-10 book snapshot, VPIN, imbalance,
VaR/kurtosis/volatility, and circuit-breaker state at the moment of exit.

For retrospective analysis, use `scripts/reconstruct_trade_lifecycle.py` to stitch
trade_log + decisions + cache + transactions + CSV history into a single enriched dataset.
The reconstruction output must preserve full linked trigger, HOLD, CLOSE, transaction,
`trigger_data`, `exit_data`, and reward-breakdown objects so top/bottom trade
comparisons can explain exactly what differed.

The audit trail is a learning substrate, not just an operator log. Entry,
in-trade HOLD/CLOSE decisions, broker/transaction events, replay/cache records,
and final trade-log rows must preserve the complete trade lifecycle with stable
`trade_id` / `decision_trade_id`, `symbol`, `timeframe_minutes`, mode, timestamp,
and sequence metadata. Capture every datapoint needed for later meta-analysis
between individual trades: trigger context, all gate states, market microstructure,
risk/circuit-breaker state, confidence/floor values, runway prediction fields,
MFE/MAE in both points and account currency, reward component breakdowns, exit
state, close reason, and learned-parameter/self-healing inputs.

Append-only JSONL audit/trade writes must be serialized as one complete line,
flushed/fsynced, and ordered by the lifecycle sequence: trigger decision before
position open, HOLD updates while in position, CLOSE/transaction event before the
final trade-log summary. Do not reset or reuse lifecycle IDs until the close
record has been durably written.

## Real-Data Test Requirements

All price-based tests should use the session-scoped fixtures from `tests/conftest.py`
which load real paper-trading data (`data/training_cache_XAUUSD_M5.jsonl`,
`data/training_cache_XAUUSD_M1.jsonl`, `data/training_cache_BTCUSD_M1.jsonl`).
Tests using these fixtures are automatically skipped if the live cache has insufficient
data (≥200 bars, ≥10 trades required).

Key fixtures:

- `xauusd_m5_bars` — session-scoped list of ~8,000+ real XAUUSD M5 bars
- `xauusd_m5_trades` — session-scoped list of raw trade records
- `xauusd_m5_bars_100/500` — function-scoped deque subsets

After modifying `openapi_hub.py` P&L or trade log paths, run:

```bash
python3 -m pytest tests/unit/test_openapi_hub_pnl.py -v --tb=short
```

After modifying reward shaper or metrics_calculator:

```bash
python3 -m pytest tests/unit/test_metrics_calculator.py tests/unit/test_reward_calculations.py -v
```

## Known Failure Modes — CB Lockout & Threshold Runaway

### Symptom
Trade rate collapses to near-zero despite bots running (bar_count rising, quote_ok, trade_ok all true).
Best-performing bots (XAUUSD M5, M1) go completely silent while XAUUSD M240 continues.

### Root causes

#### 1. Permanent CB re-trip from stale return history

Each bot's `data/paper_{BOT}/circuit_breakers.json` persists the return history across restarts.
A single large loss (e.g., emergency stop -$37 on a $10k paper account) can permanently
hold the Sortino ratio below threshold. On cooldown expiry, `reset_if_cooldown_elapsed()`
clears the trip but `check_all()` immediately re-trips on the same frozen returns.
The bot takes no trades → no new returns enter the array → infinite trip/reset loop.

The same pattern applies to `consecutive_losses`: 5 back-to-back losses stick in the
history and cause permanent re-tripping if no trades clear them.

#### 2. `entry_confidence_threshold` runaway feedback loop

`_update_risk_feedback_thresholds()` persists the floor to `entry_confidence_threshold` via
`param_manager.set_value()`. The cap is `_base_floor + 0.10` — but `_base_floor` is re-read
from the *already-saved* parameter each call. Each save raises the baseline for the next cap,
compounding with every losing trade. Observed drift: 0.6 → 0.9 within a single losing session.
In paper mode the floor is logged not enforced at the hub level, but the trigger agent's
`_confidence_gate_blocked` uses the persisted value and does enforce it.

#### 3. `feasibility_threshold` stuck at 1.0

The zero-MFE step in `trigger_agent.py` increments `feasibility_threshold` when MFE is zero.
If this runs unchecked it can hit 1.0 — an impossible gate that blocks all model-path entries.

### Diagnosis

```bash
python3 - <<'EOF'
import json, os
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
for bot in ["XAUUSD_M5","XAUUSD_M1","XAUUSD_M15","XAUUSD_M30","XAUUSD_M60","XAUUSD_M240",
            "BTCUSD_M1","BTCUSD_M5","BTCUSD_M30","BTCUSD_M60","BTCUSD_M240"]:
    cb = json.load(open(f"data/paper_{bot}/circuit_breakers.json"))
    lp_raw = json.load(open(f"data/paper_{bot}/learned_parameters.json"))
    lp = lp_raw.get('data', lp_raw)
    inst = lp.get('instruments', {}).get(f"{bot}_default", {}).get('params', {})
    tripped = [k for k in ("sortino","kurtosis","drawdown","consecutive_losses")
               if cb.get(k,{}).get("is_tripped")]
    ect = inst.get('entry_confidence_threshold',{}).get('value','?')
    feas = inst.get('feasibility_threshold',{}).get('value','?')
    print(f"{bot:<20} CB={'TRIPPED:'+','.join(tripped) if tripped else 'CLEAR':<30} entry_conf={ect}  feas={feas}")
EOF
```

### Fix

```bash
python3 scripts/fix_cb_lockout.py
```

Safe to run while bots are live. Atomically:

- Clears return histories and trip state (CB thresholds preserved — self-healing continues)
- Resets `entry_confidence_threshold` to 0.6 for bots where it drifted above 0.65
- Resets `feasibility_threshold` to 0.5 for bots where it hit the impossible-gate range (≥ 0.95)

Creates timestamped `.pre_fix_*.bak` backups beside each modified file.
