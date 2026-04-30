# HUD Redesign — Finalized Design

The HUD answers three operator questions:

1. Is the fleet healthy and safe right now?
2. Is performance improving or degrading by period, symbol, and timeframe?
3. What changed, why did it change, and what should be inspected next?

Default view: summaries first. Detailed raw metrics behind drill-down only.

---

## Design Principles

- Every number declares its scope: `Portfolio`, `Symbol`, or `Symbol/TF`.
- Every performance number declares its period: `24h`, `7d`, `Month`, `Epoch`, or `Lifetime`.
- Every result row declares its mode: `Paper` or `Live`. Mode is per-instrument — XAUUSD can be
  Live while BTCUSD is still Paper. Never a global fleet mode gate.
- Paper and Live rows are stacked vertically within each scope (Paper above Live). Never
  side-by-side comparison columns for mode.
- Offline metrics (training, backtest, tournament, champion evaluation) are visually separated
  from paper/live execution results in every tab that shows both.
- Period columns (`Lifetime | Epoch | Month | 7d | 24h`) appear side-by-side at Levels 1–3 so the
  operator compares periods without cycling. The `p` key cycles period focus at Level 4 only.
- Tables use fixed column widths, clipped text with ellipsis, consistent units. No row wraps.
- Old or deprecated metric pathways must not remain visible as current signals.

---

## Scope Model

```
period:  Lifetime | Epoch | Month | 7d | 24h
scope:   Portfolio | Symbol | Symbol/TF
mode:    Paper | Live   (per instrument — not a global fleet setting)
metric:  PnL | Win% | PF | MaxDD | TQ | EQ | Capture | Runway Acc | Health
```

---

## Consistent Drill-Down Hierarchy

One global trading-context state shared by all seven tabs:
`(_ctx_level, _ctx_symbol, _ctx_tf, _ctx_mode, _ctx_cursor, _ctx_period, _ctx_detail)`

Seven tabs are analytical lenses over the same context — not independent state machines.

```
Level 1 — Portfolio       All (symbol, mode) pairs, period columns side-by-side
    │  Enter on a row
    ▼
Level 2 — Symbol          One symbol, all TFs, period columns side-by-side
    │  Enter on a row
    ▼
Level 3 — Symbol/TF       One bot (symbol+TF+mode), period columns + trade list
    │  Enter on a row (where applicable)
    ▼
Level 4 — Detail          Single item: trade card, decision detail, period breakdown
```

**No Level 0 mode gate.** Mode is captured from the L1 row the operator selects (the L1
row model already encodes mode). There is no separate mode-selection screen.

**L1 row model (mixed-mode fleet):**

```
┌─ LIVE ─────────────────────────────────────────────────────────────────────┐
│  XAUUSD  LIVE  │ Lifetime │  Epoch  │  Month  │   7d    │  24h    │
│  BTCUSD  LIVE  │    —     │    —    │    —    │    —    │    —    │  (if promoted)
├─ PAPER ────────────────────────────────────────────────────────────────────┤
│  XAUUSD  PAPER │ Lifetime │  Epoch  │  Month  │   7d    │  24h    │
│  BTCUSD  PAPER │ Lifetime │  Epoch  │  Month  │   7d    │  24h    │
└────────────────────────────────────────────────────────────────────────────┘
```

Rows are ordered: Live bots first (if any), then Paper. Within each section, alphabetical
by symbol. Cursor navigation crosses both sections. On `Enter`, `_ctx_symbol` AND
`_ctx_mode` are captured from the selected row — not inferred from a global mode variable.

**Implemented:** `_l1_rows()` builds a `list[tuple[str, str]]` of `(symbol, mode)` pairs in
render order (Live first, then Paper, alphabetical within each section). `_drill_down()` at L1
indexes into that list to set both `_ctx_symbol` and `_ctx_mode` before advancing to L2.

---

## Navigation Contract (same keys, every tab)

| Key | Action |
|-----|--------|
| `Enter` | Drill down into highlighted row |
| `Esc` | Drill up one level |
| `↑` / `↓` or `j`/`k` | Move row selection within current level |
| `s` | Jump-scope: Portfolio → Symbol → Symbol/TF → Portfolio |
| `p` | Cycle period focus: 24h → 7d → Month → Epoch → Lifetime (Level 4 only) |
| `d` | Toggle detail/diagnostics pane (ALL tabs, consistent meaning) |
| `b` | Back — close detail pane |
| `1`–`7` | Switch tab (preserves full context hierarchy) |
| `r` | Review & reset circuit breakers |
| `e` | Set/clear stats epoch |
| `h` | Help overlay |
| `Alt+K` | Emergency kill |
| `q` | Quit HUD |

---

## Breadcrumb (first rendered line after tab header)

```
[1] OVERVIEW    ›  Portfolio
[2] PERFORMANCE ›  🟡 PAPER › Portfolio                              [Esc] [Enter] [s]
[2] PERFORMANCE ›  🟡 PAPER › Portfolio › XAUUSD                    [Esc] [Enter] [s]
[2] PERFORMANCE ›  🟡 PAPER › Portfolio › XAUUSD › M5               [Esc] [s] [d]
[7] TRADES      ›  🟡 PAPER › Portfolio › XAUUSD › M5 › 7d          [Esc] [d]
```

Mode badge: 🔴 LIVE, 🟡 PAPER, ⚫ OFFLINE. Shown from L2 onward (L1 has both modes).

---

## Naming Contract

| Term | Definition |
|------|-----------|
| `TF` | Canonical timeframe label: M1, M5, M15, M30, M60, M240 |
| `Epoch` | Configured stats epoch from `data/stats_epoch.json` |
| `Lifetime` | All available trade-log records before any epoch filter |
| `TQ` | Trade Quality (capture ratio × win rate) |
| `EQ` | Edge Quality (profit factor normalized) |
| `Runway Acc` | Realized MFE vs predicted-runway accuracy |
| `MFE` | Maximum Favorable Excursion (account currency USD) |
| `MAE` | Maximum Adverse Excursion (account currency USD) |
| `MFE pts` | MFE in raw price points (not dollars) |
| `MAE pts` | MAE in raw price points (not dollars) |
| `ZΩ` | Z-weighted Omega ratio from offline training/promotion |
| `CB` | Circuit Breaker |
| `WTL` | Winner-to-Loser (profitable trade turned to a loss) |

**Unit rule:** Never display `mfe_points` or `mae_points` labeled as USD. They are price
movement. Account-currency MFE/MAE are separate fields (`mfe`, `mae`) scaled by
`quantity × contract_size`.

---

## Per-Tab Specification

### Tab 1 — Overview

Dashboard tab. Status and health readout, not a data explorer. Maximum Level 2.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Fleet status block (all bots: stale/up/error per symbol+TF), account balance/equity (Paper and Live stacked), open positions count, active circuit breakers, system health indicators, self-healing status (from `data/performance_health.json`), exception strip (worst 3 anomalies), latest offline improvement summary |
| **2** | Symbol | Per-symbol fleet card: all TFs with bot status, current position direction, session PnL, buffer fill %, ZΩ, epsilon |

No period columns at Level 1 (this tab shows NOW, not historical performance).
No Level 3 or 4.

**Audit data sources:**
- `transactions.jsonl` per bot — SESSION_START, COMPONENT_HEALTH events for connection state
- `data/performance_health.json` — self-healing summary
- `data/current_position_*.json` — live position state
- `data/training_stats_*.json` — epsilon, buffer fill

---

### Tab 2 — Performance

Primary analysis tab. Summary-first performance tables.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Period columns (Lifetime/Epoch/Month/7d/24h): Trades, PnL, Win%, PF, MaxDD. LIVE rows stacked above PAPER rows. Offline block below (separated by header, labeled OFFLINE). |
| **2** | Symbol | Same period columns, one symbol, all TFs aggregated. LIVE above PAPER. Offline below. |
| **3** | Symbol/TF | Period columns for that bot + TQ, EQ, prediction convergence (runway accuracy), capture efficiency. |
| **4** | Detail | Top/bottom 5 trades by capture and expectancy contribution, linked to trade cards. Period focus cycling with `p`. |

**Audit data source:** `data/trade_log.jsonl` — aggregated by period, filtered by symbol, TF, mode.

---

### Tab 3 — Training

Offline and runtime learning. Mode distinction: runtime (paper) learning metrics vs offline training results.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Offline training queue status (from `data/offline_training_status.json`), fleet runtime learning summary per bot: epsilon (ε), buffer fill %, loss trend, reward trend, explore_flag |
| **2** | Symbol | Per-symbol offline results: champion vs incumbent ZΩ, accepted/rejected, Optuna/tournament summary |
| **3** | Symbol/TF | Single-bot training detail: ε, β, buffer fill %, loss trend, reward trend, learned parameter change log, win_rate_ema, conf_calib_err |
| **4** | Detail | Single training run metrics, checkpoint comparison, parameter old→new values |

**Audit data sources:**
- `data/offline_training_status.json` — queue, supervisor metadata
- `data/training_stats_*.json` — runtime RL metrics
- `data/checkpoints/offline_champions.json` — champion ZΩ per symbol/TF
- Learned parameters from `LearnedParametersManager` — per-symbol/TF

**Note:** `explore_flag` from decision log reasoning must be surfaced here (currently invisible).
Offline champion ZΩ must be visually separated from paper/live win rates.

---

### Tab 4 — Risk

Account risk and per-bot risk. Mode-aware.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Account exposure (Paper and Live stacked), total open positions, margin used, portfolio drawdown, active circuit breakers across all bots |
| **2** | Symbol | Per-symbol risk: CB state, VaR 95%, regime ζ, reward weights (current learned), path geometry, kurtosis vs threshold |
| **3** | Symbol/TF | Single-bot risk detail: kurtosis current vs learned threshold (NOT legacy 3.0 alert), VaR breakdown, no-entry reason (last `gated_conditions`), spread, VPIN-z, win_rate_ema, entry floor |

No Level 4.

**Audit data sources:**
- `data/risk_metrics_*.json` — per-bot VaR, kurtosis, drawdown
- `circuit_breakers.json` — CB state
- `decisions.jsonl` — last NO_ENTRY `gated_conditions` per bot (meaningful entries only)
- `data/current_position_*.json` — open position state

**Important:** Circuit breaker threshold shown is the learned per-symbol/TF action threshold from
`LearnedParametersManager`, not a hardcoded 3.0 alert threshold.

---

### Tab 5 — Market

Market microstructure and L2 order book. **No mode dimension** — price feed is the same
regardless of Paper vs Live bot status.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Per-symbol compact summary: spread, VPIN-z, regime label (one row per symbol) |
| **2** | Symbol | Per-TF market table: spread, depth ratio, VPIN-z, imbalance, regime ζ, realized volatility |
| **3** | Symbol/TF | Full order book ladder, depth ratio, L2 snapshot, signal synthesis, alignment score |

No Level 4.

**Audit data sources:** On-tick telemetry from running bots only (no JSONL files).

---

### Tab 6 — Decision Log

Strategy reasoning and policy trace. Shows meaningful decisions only (CACHED flood fixed).
**Fully hierarchical** — level dispatch mirrors Tab 7 (Trades) reference implementation.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | 15 most-recent signal decisions across all bots: timestamp, bot (symbol/TF), mode badge, agent, decision, confidence, brief context. CACHED/WARMING_UP entries skipped. |
| **2** | Symbol | 25 entries filtered to selected symbol+mode, newest-first |
| **3** | Symbol/TF | 40 entries filtered to symbol+TF with cursor (`↑↓`/`jk`). NO_ENTRY rows inline-expand `gated_conditions`. `Enter`/`d` → L4. |
| **4** | Detail | Full decision card: timestamp, bot/mode badge, agent, decision, confidence, full `context` dict, full `reasoning` dict, all `gated_conditions` expanded (no cap), `trade_id`/`position_id` links. `[b]`/`Esc` back. |

**gated_conditions (Level 4 — implemented):**
For NO_ENTRY decisions, every rejection reason from `gated_conditions` is rendered:
```
  ├─ GATED CONDITIONS (3) ────────────────
  │    ✗ kurtosis=14.15 > 5.94
  │    ✗ vpin_z=3.21 > 2.50
  │    ✗ spread=4.2 > 3.0 max
```
Also shown as `[Ngates]` badge inline on L3 rows. At L3+ `gated_conditions` from NO_ENTRY
rows are expanded inline below the row (up to all entries, uncapped in L4 card).

**CRITICAL — CACHED flood fix:**
`decisions.jsonl` is ~90% `CACHED` entries (startup cache-load events). Reading the last 200
lines gives virtually no real decisions. Fix: scan backward skipping non-signal entries:

```python
_SKIP_DECISIONS = frozenset({'CACHED', 'WARMING_UP', 'FLAT_SKIP', 'NO_ENTRY_SKIP'})

def _tail_meaningful(path: Path, n: int = 50) -> list[dict]:
    lines = path.read_text().splitlines()
    results = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get('decision') in _SKIP_DECISIONS:
            continue
        results.append(e)
        if len(results) >= n:
            break
    return list(reversed(results))
```

**Implemented** as `_tail_meaningful()` static method on `HUDRenderer`. Used by
`_load_decision_entries()` which powers all Tab 6 level views.

**Audit data source:** `decisions.jsonl` per bot (meaningful entries only via `_tail_meaningful`).

---

### Tab 7 — Trades

Executed trades and trade lifecycle. **Reference implementation** for level dispatch — all other
tabs should follow this pattern.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Per-instrument summary with period columns (Trades, PnL, WR per period). LIVE above PAPER. |
| **2** | Symbol | Per-TF metrics table with period columns. LIVE above PAPER. |
| **3** | Symbol/TF | Individual trade list (paginated): #, mode badge, date/time, dir, entry, exit, PnL, cap%, MFE, MAE, bars, reason. `p` key narrows list to period. |
| **4** | Detail | **Trade card**: full detail with trigger_data, exit_data, reward breakdown |

**Trade card fields (Level 4):**
- Identity: trade_id, symbol, TF, mode, direction, quantity
- Timing: entry_time, exit_time, bars_held, ticks_held
- P&L: pnl (USD), pnl_points, spread_cost_points, balance_after
- Excursions: mfe (USD), mae (USD), mfe_points, mae_points, mfe_bar_offset, bars_from_mfe_to_exit, capture_ratio, winner_to_loser
- Entry conditions: entry_confidence, entry_dynamic_floor, entry_conf_margin, entry_vpin_z, win_rate_ema_at_entry, conf_calib_err_at_entry
- Runway: predicted_runway_net_points, runway_utilization, runway_accuracy_ema, trigger_quality
- Exit: close_reason, exit_regime, exit_vol, exit_depth_ratio
- Trigger snapshot (trigger_data dict): regime, geom_efficiency, geom_runway, hmm_probs, kurtosis, var_95, gated_conditions, alignment_score
- Exit snapshot (exit_data dict): exit_confidence, exit_floor, close_reason, trailing_stop state, breakeven state, capture_decay state
- Rewards: trigger_reward, reward_harvester_total + full breakdowns

**Unit validation:** MFE/MAE in USD are `mfe`/`mae` fields. `mfe_points`/`mae_points` are raw
price movement. The trade card must label these distinctly — never display points as dollars.

**Audit data sources:**
- `data/trade_log.jsonl` — trade records (primary)
- `transactions.jsonl` — POSITION_OPEN/CLOSE events linked by position_id to trade card

---

## Audit Log Routing Summary

| Log file | Tab 1 | Tab 2 | Tab 3 | Tab 4 | Tab 5 | Tab 6 | Tab 7 |
|----------|-------|-------|-------|-------|-------|-------|-------|
| `data/trade_log.jsonl` | — | ✅ primary | — | — | — | link | ✅ primary |
| `logs/audit/decisions.jsonl` | — | — | explore_flag | last gated | — | ✅ primary | linked |
| `logs/audit/transactions.jsonl` | ✅ health | — | — | — | — | — | ✅ linked |
| `data/training_stats_*.json` | ε/buf | — | ✅ primary | — | — | — | — |
| `data/risk_metrics_*.json` | alerts | — | — | ✅ primary | — | — | — |
| `data/performance_health.json` | ✅ self-heal | — | anomalies | — | — | — | — |
| `data/offline_training_status.json` | queue | — | ✅ primary | — | — | — | — |
| `data/current_position_*.json` | ✅ positions | — | — | exposure | — | — | — |
| `circuit_breakers.json` | ✅ CB alerts | — | — | ✅ primary | — | — | — |

**Currently broken:** `transactions.jsonl` is written per bot but read by zero HUD tabs.
Tab 1 should use it for session/connection health; Tab 7 should link POSITION_OPEN/CLOSE
events to trade cards.

---

## Known Bugs / Implementation Status

### ✅ FIXED — L1→L2 drill cursor/mode bug

`_l1_rows()` returns `list[tuple[str, str]]` of (symbol, mode) pairs in render order.
`_drill_down()` at L1 indexes into that list and captures both `_ctx_symbol` and `_ctx_mode`.

### ✅ FIXED — Decision log CACHED flood

`_tail_meaningful()` static method scans backward, skipping `CACHED`/`WARMING_UP` entries.
Used by `_load_decision_entries()` which powers all Tab 6 level views.

### ✅ FIXED — Performance tab period columns

`_render_perf_period_columns()` renders period columns side-by-side at L1/L2/L3.
`_render_performance()` dispatches by `_ctx_level`.

### ✅ FIXED — Level dispatch missing from Tabs 1–5

All tabs (Overview, Performance, Training, Risk, Market) now dispatch by `_ctx_level`.
Tab 6 (Decision Log) fully hierarchical as of commit `9f33dad`.
Tab 7 (Trades) remains the reference implementation.

### ✅ FIXED — `gated_conditions` invisible in HUD

- Tab 6 L3: inline `[Ngates]` badge on NO_ENTRY rows + per-gate expansion below row
- Tab 6 L4: full decision card with all gates expanded (uncapped)
- Tab 4 L3: last NO_ENTRY `gated_conditions` shown in risk detail

### ✅ FIXED — `transactions.jsonl` routed to Tab 1 and Tab 7

- Tab 1 System Health block now includes `🔌 SESSION LOG` via `_render_health_session_events()`:
  per-bot last SESSION_START age, recent COMPONENT_HEALTH and SESSION_EVENT entries.
- Tab 7 trade card L4 shows `BROKER EVENTS` section: POSITION_OPEN (entry price, direction,
  quantity, confidence, entry_gated_conditions) and POSITION_CLOSE (exit price, PnL, capture
  ratio, close reason) looked up by `position_id` via `_load_transactions_for_position()`.

---

## Regression / Omission Flags (from Cross-Project Review)

These are **not** in scope for the current HUD implementation sprint but must be tracked
as future RL/reward improvements.

### RL Architecture
| Flag | Current State | Reference | Priority |
|------|--------------|-----------|----------|
| Dueling DQN (Value + Advantage streams) | Single Conv1d head | Kinetra `drl_dueling_dqn.py:47-92` | High |
| Prioritized Experience Replay | Uniform sampling | Kinetra PER + constraint violation boost (+3.0) | High |
| Continual learning / drift detection | Missing | Kinetra ContinualLearningManager — retroactive labeling, >30% poor trades triggers policy update | High |
| MFE/MAE efficiency in experience priority | Missing | Kinetra `experience_replay.py:103-114` (-2 to +1 scoring) | Medium |
| Phase-aware reward shaping | Basic timing penalty | Supra `phase_action_shaping()` (pre_entry→entry→hold→exit→post_exit) | High |
| Time-in-trade penalty in reward | Missing | Kinetra `R_t = PnL/E_t + α·MFE/ATR − β·MAE/ATR − γ·Time` | Medium |
| Regime-adaptive reward weights | Static multipliers | Kinetra: dynamic α,β per regime | Medium |

### State Features
| Flag | Current State | Reference | Priority |
|------|--------------|-----------|----------|
| ATR-normalized features | 18 features, not ATR-normalized | Supra: 65-feature ATR-normalized vector | High |
| FFT dominant period detection | Missing | Kinetra `rl_gpu_trainer.py:120-199` — dominant_short/long_period, period_ratio | Medium |
| Physics features (Torque, Reynolds, etc.) | Missing | Kinetra `rl_gpu_trainer.py:275-337` — Torque, Market Reynolds, Phase Compression, Suppression Ratio, Spring Stiffness, Entropy Proxy | Low |
| Candle/Energy magnitude percentile | Missing | Kinetra `rl_gpu_trainer.py:378-392` | Low |
| VOL regime ratio (short/long vol baseline) | Missing | Kinetra `rl_gpu_trainer.py:413-422` | Medium |
| MFE/MAE in ATR units | Only USD and points | Supra ExcursionStats | Medium |

### Metrics / Analysis
| Flag | Current State | Reference | Priority |
|------|--------------|-----------|----------|
| Omega ratio | Missing | Supra `metrics.compute_omega()` | Low |
| Giveback-focused harvest analysis | Partial | Supra `harvest.py` (GOOD/BAD_HARVEST by hold bars + MFE deciles) | Medium |
| Hold risk lookup (harvest classifier) | Missing | Supra `build_hold_risk_lookup.py` | Low |
| Calmar ratio | Missing | Supra metrics | Low |
| Phase-specific accuracy tracking | Missing | Supra per-phase entry/hold/exit accuracy | Medium |

---

## HUD Sync Requirement For Code Changes

Any change to training, promotion, reward shaping, trade logging, risk, decision logging,
learned parameters, or self-healing telemetry must include a HUD impact check:

- Does the HUD still read the canonical source of truth?
- Did any field name, unit, scope, or period meaning change?
- Does a visible label now need renaming?
- Is any old metric pathway now deprecated or misleading?
- Are paper and live results still separated, with no blended metric rows?
- Are portfolio, symbol, and symbol/TF views still separate?
- Are tests or fixtures needed for the affected render helper?

---

## Implementation Order

1. ✅ **Fix `_tail_meaningful()`** — Decision log CACHED flood (commit `ce9bccf`)
2. ✅ **Fix `_l1_rows()` and `_drill_down()` at L1** — Mode capture on drill (commit `ce9bccf`)
3. ✅ **Add level dispatch to Tab 2 (Performance)** — Period columns at L1/L2/L3 (commit `ce9bccf`)
4. ✅ **Add level dispatch to Tab 4 (Risk)** — Fleet vs symbol vs bot views (commit `ce9bccf`)
5. ✅ **Add level dispatch to Tab 3 (Training)** — Queue vs symbol vs bot views (commit `ce9bccf`)
6. ✅ **Add level dispatch to Tab 1 (Overview)** — Dashboard vs symbol card (commit `ce9bccf`)
7. ✅ **Add level dispatch to Tab 5 (Market)** — Symbol summary vs TF detail (commit `ce9bccf`)
8. ✅ **Tab 6 full hierarchical dispatch** — L1→L2→L3(cursor)→L4(detail card) (commit `9f33dad`)
9. ✅ **Surface `gated_conditions`** — Tab 6 L3 inline + L4 card + Tab 4 L3 risk detail (commit `9f33dad`)
10. ✅ **Link `transactions.jsonl`** — Tab 1 session health block + Tab 7 broker events (commit `HEAD`)
11. ⬜ **Render period columns in Trades tab** — Side-by-side at L1/L2 (currently rows)
