# cTrader Adaptive RL Trading Bot — Claude Instructions

## Project Overview

Dual-DDQN paper trading fleet (TriggerAgent + HarvesterAgent) using cTrader Open API.

- Topology: `openapi-hub` — `src/core/openapi_hub.py`, launched via `run_universe.py --watch`
- Symbols: XAUUSD, BTCUSD × 6 TFs (M1, M5, M15, M30, M60, M240)
- GPU: AMD RX 7600 / ROCm — always set `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- Branch: `update-1.1-mfe-mae-tracking-v2`

## Wire Scale Invariant

cTrader SpotEvent bid/ask are ALWAYS at 10^5 precision. `_scale = 100000` is fixed.
`digits` from `SymbolByIdRes` is display precision only — never use it to set `_scale`.

## Audit Log & Trade Log

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

**`close_reason` is always populated.** `shutdown()` sets `last_close_reason = "shutdown"`
and `_PaperEmergencyCloser.close_all_positions()` sets `"circuit_breaker"` before calling
`_close_position()`. No more blank close_reason records.

For retrospective analysis, use `scripts/reconstruct_trade_lifecycle.py` to stitch
trade_log + decisions + cache + transactions + CSV history into a single enriched dataset.

## Real-Data Test Requirements

All price-based tests should use the session-scoped fixtures from `tests/conftest.py`
which load real paper-trading data (`data/training_cache_XAUUSD_M5.jsonl`,
`data/training_cache_XAUUSD_M1.jsonl`, `data/training_cache_BTCUSD_M1.jsonl`).

After modifying `openapi_hub.py` P&L or trade log paths, run:

```bash
python3 -m pytest tests/unit/test_openapi_hub_pnl.py -v --tb=short
```

After modifying reward shaper or metrics_calculator:

```bash
python3 -m pytest tests/unit/test_metrics_calculator.py tests/unit/test_reward_calculations.py -v
```

## Running Tests — CRITICAL

**Never run `python -m pytest` without a target.** 2440 tests across 104 files will OOM-kill (exit 137).

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

When diagnosing failures across the whole suite, run one directory at a time:

1. `python -m pytest tests/unit/ -n 2 --dist loadfile --tb=line -q`
1. `python -m pytest tests/integration/ --tb=short`
1. `python -m pytest tests/test_*.py --tb=short`

## Key Files

| File | Purpose |
| ------ | --------- |
| `src/core/openapi_hub.py` | Main hub — all SpotEvents, TrendBars, order flow |
| `run_universe.py` | Supervisor launching per-symbol hubs |
| `run.sh` | Shell launcher (sources env + ROCm config) |
| `data/universe.json` | Fleet registry — weights paths, symbol/TF config |
| `.env` | Runtime flags (PAPER_MODE, EPSILON, etc.) |
| `/home/renierdejager/Projects/Kinetra/.env.openapi` | OAuth credentials |
| `data/history/` | Downloaded OHLCV CSVs for offline training |
| `train_offline.py` | Offline tournament trainer (6 variants, auto-promote) |
| `scripts/performance_analyzer.py` | Self-healing fleet analyzer — detects anomalies, applies corrections via `LearnedParametersManager`, writes `data/performance_health.json` |

## Architecture Rules

- Every metric scoped by `(symbol, timeframe_minutes)` — e.g. `paper_stats_XAUUSD_M5.json`
- Canonical TF labels: M1, M5, M15, M30, M60, M240 — never H4
- `data/universe.json` is the promoted weight registry

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

## Per-Trade Risk Cap (src/constants.py)

Hard cap is R-multiple-based, not fixed USD. `MAX_LOSS_PER_TRADE_USD` was removed.

```python
MAX_LOSS_MULT_PER_TRADE: float = 5.0   # cap = 5× the position's expected 1R stop-loss
MIN_CAP_USD: float = 2.0               # absolute floor
MAX_CAP_USD: float = 200.0             # absolute ceiling
# cap_usd = entry_price × (STOP_LOSS_PCT_DEFAULT/100) × qty × contract_size × 5.0
# clamped to [2.0, 200.0]
# XAUUSD (0.01 lot, ~$3300): cap ≈ $66.  BTCUSD (0.01 lot, ~$114k): cap ≈ $23.
```

R:R profit floor: when `MFE ≥ 1R`, trailing floor = `MFE − 1R` (protects captured gain).

## Reward Shaping (src/core/reward_shaper.py, as of 2026-04-29)

Key constants after rebalancing — PnL signal raised 8.5×, timing penalty reduced:

```python
WEIGHT_ACTIVITY: float = 0.2            # was 0.8 — activity bonus deprioritised
WEIGHT_PNL_ALIGNMENT: float = 1.2       # was 0.6 — PnL signal weight doubled
PNL_ALIGNMENT_MULT_DEFAULT: float = 1.5 # was 0.35 — effective PnL weight: 0.21 → 1.80
UNDEVELOPED_MFE_PENALTY_SCALE: float = -0.4  # was -1.0 — timing penalty softened
```

Trigger reward saturation fix (openapi_hub.py `_close_position`): the log-based runway
shaped reward is only applied when `abs(shaped_tr) < 2.99`; otherwise falls back to the
4-component reward (`accuracy + magnitude − false_positive − toxic_flow`). This prevents
73%-at-rail gradient collapse when the runway predictor is uncalibrated.

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

For BTCUSD: batch ≤2 TFs per invocation — demo server drops TCP after ~5 min on BTC-volume downloads:

```bash
# Run three separate calls:
python3 scripts/download_ctrader_history.py --symbol BTCUSD --timeframes M1 M5 ...
python3 scripts/download_ctrader_history.py --symbol BTCUSD --timeframes M15 ...
python3 scripts/download_ctrader_history.py --symbol BTCUSD --timeframes M30 M60 M240 ...
```

`ProtoOATrendbar` field names: `b.low` (absolute base, raw ticks), `b.deltaOpen`, `b.deltaHigh`,
`b.deltaClose` (deltas from low). Divide all by `_WIRE_SCALE = 100_000` for scaled price.

## Ghost HOLD Race Condition (fixed 2026-04-27)

**Bug**: `self.position = None` and `self._current_trade_id = None` were cleared near the END of
`_close_position()` (~150 lines in), after reward/ML processing. Any exception in that window left
the position live. The next tick would then emit a HOLD with the stale `trade_id` — a "ghost HOLD".

**Fix**: Both clears now happen immediately after snapshotting `pos = self.position` at the TOP of
`_close_position()`. All downstream work operates on local variables. Pattern: snapshot → release → process.

**Audit log**: `tests/validation/test_decision_log_correlation.py` live-log scan uses a cutoff of
`2026-04-17T00:00:00+00:00` to skip the 4 pre-fix violations in historical data.

## TFAgent — MFE/MAE Tracking v2 Features (as of 2026-04-27)

Key additions to `TFAgent` in `src/core/openapi_hub.py`:

### trade_id threading

- `_current_trade_id` set to `f"{symbol}_{tf_label}_{uuid.uuid4().hex[:8]}"` on position open
- Cleared to `None` at top of `_close_position()` (before processing, preventing ghost HOLDs)
- Threaded through: LONG/SHORT decision log → every HOLD audit entry → CLOSE audit entry → trade_log record

### HOLD audit logging

`_add_harvester_hold_experience()` writes a HOLD entry to the decision log on every bar while in position.
`DecisionLogger.log_harvester_decision()` guards against `in_position=False` calls (suppresses them).

### Dynamic entry floor

`_compute_dynamic_entry_floor(base_floor)` returns `max(base + cal_uplift + runway_penalty, rl_floor_capped)`.
Applied in `_handle_flat()` after `decide_entry()`. Parameters from `LearnedParametersManager`.

### Exit confidence floor

`_exit_conf_dynamic_floor` applied in `_handle_exit_on_tick()` — blocks exits below the floor.

### Risk feedback thresholds

`_update_risk_feedback_thresholds(pnl_usd)` — win-rate EMA (α=0.15); raises entry floor +0.02 if
win_rate < 40%, lowers −0.01 if > 65%. Min-trades guard per TF before kicks in.

### 4-component trigger reward

`_calculate_trigger_reward()` — accuracy + magnitude_bonus − false_positive_penalty − toxic_flow_penalty,
clipped to [−1.5, 1.5]. Regime-adjusted capture reward in `_close_position()`.

### Preseed / bars cache

`_PRESEED_STOP_PCT=0.003`, `_PRESEED_TARGET_PCT=0.002`, `_PRESEED_MAX_HOLD=20`.
Last 500 bars persisted to `bars_cache.json` every 10 bars; warm-start preseed on load.

### CB restore at startup

`circuit_breakers.restore_state("data/circuit_breakers.json")` called in `__init__` after `set_emergency_closer`.

### Adaptive regularization TD feedback

In `_maybe_train()`: if `avg_td > 0.5` → `increase_regularization()`; if `< 0.1` → `decrease_regularization()`.

## Running Tests — Avoid DNS-hanging test files

Some test files spin up `HTTPServer` which calls `socket.getfqdn()` — can hang for minutes:

- `tests/unit/test_production_monitor_http.py`
- `tests/unit/test_production_monitor_extended.py`
- `tests/unit/test_prometheus_metrics.py`

Exclude them when doing broad sweeps:

```bash
python -m pytest tests/unit/ --ignore=tests/unit/test_production_monitor_http.py \
  --ignore=tests/unit/test_production_monitor_extended.py \
  --ignore=tests/unit/test_prometheus_metrics.py --timeout=20 -q
```

## Current Universe State (as of 2026-04-28)

| Symbol | TF | ZΩ | Status |
| -------- | ---- | ------- | -------- |
| XAUUSD | M1 | 3.159 | Promoted, active |
| XAUUSD | M5 | 1.663 | Promoted, active |
| XAUUSD | M60 | 1.428 | Promoted, active |
| XAUUSD | M240 | 1.601 | Promoted, active |
| XAUUSD | M15, M30 | — | Untrained — no weights (needs >50 live cache rows) |
| BTCUSD | M30 | 1.103 | Promoted, active |
| BTCUSD | M60 | 1.065 | Promoted, active |
| BTCUSD | M240 | 1.057 | Promoted, active |
| BTCUSD | M1, M5, M15 | — | Untrained — offline training in progress |

History data in `data/history/`: BTCUSD M1=1.18M, M5=237K, M15=79K, M30=39K, M60=19K, M240=5K bars.
XAUUSD M1=816K, M5=163K, M15=54K, M30=27K, M60=13K, M240=3.5K bars (all Jan 2024–Apr 2026).

## Offline Training

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 train_offline.py \
  data/history/XAUUSD_M1.csv ... data/training_cache_XAUUSD_M1.jsonl ... \
  --symbols XAUUSD --workers 2 --n-epochs 3 --warm-start \
  --accept-if-better --auto-promote --retrain-rounds 6 \
  --paper-threshold 1.0 --tournament-variants 6
```

XAUUSD M15/M30/M60 offline training skipped when live cache < 50 rows — needs more paper-trading time.

### Optuna Hyperparameter Search

Pass `--optuna-trials N` instead of `--tournament-variants` to run Bayesian HPO:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 train_offline.py \
  data/history/XAUUSD_M5.csv data/training_cache_XAUUSD_M5.jsonl \
  --symbols XAUUSD --workers 2 --optuna-trials 20 \
  --accept-if-better --auto-promote --paper-threshold 1.0
```

- Each `(symbol, timeframe)` gets an isolated SQLite study: `data/optuna/offline_SYMBOL_MTF.db`
- Studies persist across restarts — Optuna resumes from completed trials
- Objective: ZΩ + val PF + val net PnL, with trade-shortfall penalty for under-sampled runs
- Promotion still routes through the incumbent/champion acceptance guard unchanged
- `scripts/optuna_then_tournament.sh` runs Optuna search then tournament promotion in sequence

## ExperienceBuffer Save Robustness (src/utils/experience_buffer.py)

`save()` now filters to a canonical state size (determined by the first non-None entry)
before calling `np.array()`. This prevents `"inhomogeneous shape"` crashes when the circular
buffer contains a mix of offline-training experiences (old feature dim) and paper-trading
experiences (new feature dim). Mismatched entries are dropped with a `LOG.warning`; an empty
post-filter result returns `True` (no-op save) rather than crashing.

The matching filter already existed in `load()` — `save()` is now symmetric.

Regression tests: `tests/unit/test_experience_buffer.py::TestSaveLoad`

- `test_round_trip` — canonical round-trip (30 experiences, dim=7)
- `test_save_survives_mixed_state_dims` — 20 dim=7 injected + 30 dim=21 via `add()`; save succeeds, reload recovers 20 canonical entries

## Self-Healing Performance Analyzer (scripts/performance_analyzer.py)

Runs automatically every **4 hours** (480 supervisor cycles) via `run_universe.py --watch`.
Analyzes `data/trade_log.jsonl` and applies corrective parameter adjustments when `--auto-heal`.

**8 anomaly codes with automatic corrections:**

| Code | Threshold | Correction |
| ---- | --------- | ---------- |
| `TRIGGER_SATURATION` | >15% at ±3.0 rail | Code fix flag only |
| `EMERGENCY_RATE_HIGH` | emergency% >5% | raise `confidence_floor` +0.02 |
| `DDQN_WIN_RATE_LOW` | WR <30%, n≥5 | raise `exit_confidence_threshold` +0.04 |
| `BAD_RISK_REWARD` | loser/winner >2.8× | raise `confidence_floor` +0.02 |
| `CAPTURE_EFFICIENCY_LOW` | capture <0.25, n≥8 | raise `exit_confidence_threshold` +0.03 |
| `WTL_PENALTY_EXCESSIVE` | mean WTL <-1.5 | reduce `wtl_penalty_multiplier` -0.20 |
| `RUNWAY_ACCURACY_LOW` | accuracy <0.35 | raise `runway_cal_alpha` +0.04 |
| `PNL_ALIGNMENT_WEAK` | PnL align <0.08, n≥8 | raise `pnl_alignment_multiplier` +0.10 |

Writes `data/performance_health.json` with `overall_health`, per-bot metrics, anomalies, and
corrections applied. Run manually: `python3 scripts/performance_analyzer.py --auto-heal --hours 24`.

## Paper Exploration Constants (src/constants.py)

All epsilon/exploration values are sourced from `constants.py` — never hardcode them:

```python
PAPER_EPSILON_START: float = 1.0    # 100% random at start of paper training
PAPER_EPSILON_END:   float = 0.25   # Floor: keep 25% exploration throughout (was 0.1)
PAPER_EPSILON_DECAY: float = 0.9998 # Slower decay than before (was 0.9995)
PAPER_FORCE_EXPLORATION: bool = True

LIVE_EPSILON_START: float = 0.05
LIVE_EPSILON_END:   float = 0.01
LIVE_EPSILON_DECAY: float = 0.9995
```

`DualPolicy.load_checkpoint()` clamps restored epsilon to the configured floor and **does not
override paper epsilon_decay with stale checkpoint metadata** — the paper decay is always the
slower configured rate, not whatever a stale checkpoint recorded.

## HUD Navigation (src/monitoring/hud_tabbed.py)

Global 5-level context hierarchy across all 7 tabs: Mode → Portfolio → Instrument → Instrument/TF → Period Detail.

| Key | Action |
|-----|--------|
| `Enter` | Drill down into highlighted row |
| `Esc` | Drill up one level |
| `↑`/`↓` or `j`/`k` | Move row selection |
| `d` | Toggle detail pane (ALL tabs) |
| `b` | Close detail pane |
| `p` | Cycle period focus |
| `s` | Jump-scope shortcut |

Periods (Lifetime | Epoch | Month | 7d | 24h) shown as columns side-by-side at summary levels.
See `docs/HUD_REDESIGN.md`.

## HUD Self-Healing Panel (src/monitoring/hud_tabbed.py)

The Overview tab's **🏥 SYSTEM HEALTH** block now includes a **🔄 SELF-HEAL** row reading
`data/performance_health.json`. Shows:

- Overall fleet health (`HEALTHY` / `DEGRADED` / `CRITICAL`) and report age
- Fleet win rate, profit factor, emergency rate (color-coded)
- Active anomaly codes per bot (up to 4)
- Last parameter corrections applied with `old→new` values

When the file does not exist yet: `no report yet — runs every 4 h`.

## Code Style

- Complete, production-ready implementations only — no stubs, no `pass`, no `# TODO`
- No comments unless the WHY is non-obvious
- No trailing summaries in responses
