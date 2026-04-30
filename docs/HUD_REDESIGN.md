# HUD Redesign Direction

The HUD should answer three operator questions quickly:

1. Is the fleet healthy and safe right now?
2. Is performance improving or degrading by period, symbol, and timeframe?
3. What changed, why did it change, and what should be inspected next?

The standard display should be summaries first. Detailed raw metrics belong behind
drill-down views, not in the default viewport.

## Design Principles

- Every number must declare its scope: `Portfolio`, `Symbol`, or
  `Symbol / Timeframe`.
- Every performance number must declare its period: `24h`, `7d`, `Month`,
  `Epoch`, or `Lifetime`.
- Every result row must declare its mode: `Paper`, `Live`, or `Offline`.
  Paper, live, and offline results must not be mixed into one metric row.
- Do not mix account-level exposure and per-bot learning metrics in the same row
  unless the row label says `Portfolio`.
- Prefer compact summary tables over long vertical metric lists.
- Default views show status, deltas, and exceptions. Drill-down views show raw
  values, source fields, and explanatory detail.
- Use one naming contract everywhere. For example:
  - `TF` means canonical timeframe label (`M1`, `M5`, `M15`, `M30`, `M60`,
    `M240`).
  - `Epoch` means the configured stats epoch from `data/stats_epoch.json`.
  - `Lifetime` means all available trade-log records before any epoch filter.
  - `TQ` means trade quality.
  - `EQ` means edge quality.
  - `Runway Acc` means realized MFE-vs-predicted-runway accuracy.
- Tables must use fixed column widths, clipped text with clear ellipses, and
  consistent units. No row should wrap in normal terminal widths.
- Old or deprecated metric pathways must not remain visible as if they are
  current. When source fields change, HUD renderers and labels must be updated
  in the same change set.

## Scope Model

The HUD should treat the data model as a cube:

```text
period:    24h | 7d | Month | Epoch | Lifetime
scope:     Portfolio | Symbol | Symbol/TF
mode:      Paper | Live | Offline
metric:    PnL | Win% | PF | MaxDD | TQ | EQ | Capture | Runway Acc | Health
```

The default screen should render a curated slice of this cube. Drill-down
screens should let the operator move from portfolio summary to symbol summary to
timeframe detail without duplicating the same long tables on every tab.

`Offline` is not an execution/account mode. It represents training,
backtest/validation, tournament, Optuna, champion, and incumbent-evaluation
results. Keep offline metrics visually separate from paper/live trading results
so an accepted or rejected candidate cannot be mistaken for current account PnL.

Do not blend paper, live, and offline into combined performance rows. They can
be compared side by side, but each row must keep one mode. Offline should never
be combined with paper/live account metrics; it is evidence for promotion and
improvement, not realized execution.

## Consistent Drill-Down Hierarchy

One global trading-context hierarchy. Seven tabs as analytical lenses over
the same context. The HUD always knows its current `(mode, level, symbol, tf, period)`
and renders accordingly.

```
Level 0 — Mode               Live / Paper / Offline trading mode selection
    │  Enter on a row
    ▼
Level 1 — Portfolio          All symbols, all TFs, period columns visible
    │  Enter on a row
    ▼
Level 2 — Symbol             One symbol, all TFs, period columns visible
    │  Enter on a row
    ▼
Level 3 — Symbol / TF        One symbol, one timeframe, period columns + trade list
    │  Enter on a row (where applicable)
    ▼
Level 4 — Detail             Single item: trade card, decision detail, period breakdown
```

**Period column rule:**

Periods (Lifetime | Epoch | Month | 7d | 24h) are shown as **columns side-by-side**
at Levels 1–3 so the operator can compare performance across periods without
cycling. The `p` key cycles the period focus at Level 4 only.

**Navigation contract (same keys, every tab):**

| Key | Action |
|-----|--------|
| `Enter` | Drill down into highlighted row |
| `Esc` | Drill up one level |
| `↑` / `↓` or `j`/`k` | Move row selection within current level |
| `s` | Jump-scope shortcut: Portfolio → Symbol → Symbol/TF → Portfolio |
| `p` | Cycle period focus: 24h → 7d → Month → Epoch → Lifetime |
| `d` | Toggle detail/diagnostics pane (consistent across ALL tabs) |
| `b` | Back — close detail pane |
| `1`–`7` | Switch tab (preserves context hierarchy) |

**Mode display rule:**
Paper and Live are **stacked vertically** within the same scope/level — never
side-by-side comparison columns. Paper rows appear above Live rows. Offline is
separated visually (not account PnL). Mode is shown at the top of every breadcrumb.

**Breadcrumb:**
Every rendered view shows a compact breadcrumb so the operator always knows
where they are:

```
[2] PERFORMANCE  ›  🟡 PAPER › Portfolio            [Esc back] [Enter drill] [s scope]
[2] PERFORMANCE  ›  🟡 PAPER › Portfolio › XAUUSD   [Esc back] [Enter drill] [s scope]
[2] PERFORMANCE  ›  🟡 PAPER › Portfolio › XAUUSD › M5   [Esc back] [s scope] [d detail]
[7] TRADES  ›  🟡 PAPER › Portfolio › XAUUSD › M5 › 7 days   [Esc back] [d detail]
```

The breadcrumb is the first line rendered after the tab header.

---

## Per-Tab Drill-Down Specification

### 1. Overview (Tab `1`)

The landing dashboard. Light on drill-down — mostly a status readout.

| Level | Scope | Renders |
|-------|-------|---------|
| **0** | Mode | Mode selection screen (Live/Paper/Offline) |
| **1** | Portfolio | Fleet status (all bots, stale/up), account balance/equity, open positions, active circuit breakers, system health block, self-healing status, exception strip (worst 3 anomalies), improvement summary (latest offline run) |
| **2** | Symbol | Per-symbol fleet cards: bot status, position, session PnL, buffer fill, ZΩ |

No Level 3/4 for this tab — it's a dashboard, not a data explorer.

---

### 2. Performance (Tab `2`)

Summary-first performance tables. The primary analysis tab.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Period columns (24h/7d/Month/Epoch/Lifetime) — Trades, PnL, Win%, PF, MaxDD. Paper rows stacked above Live rows. Offline rows in a separate block below. |
| **2** | Symbol | Same period columns, scoped to one symbol. All TFs aggregated. Paper above Live, Offline below. |
| **3** | Symbol/TF | Period columns for that bot + trade quality, edge quality, prediction convergence. |
| **4** | Detail | Top/bottom 5 recent trades by capture and expectancy contribution (linked trade cards). |

Period focus cycling with `p` at Level 4 only.

---

### 3. Training (Tab `3`)

Offline and runtime learning.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Training queue status, fleet runtime learning summary (epsilon, buffer fill, loss trend per bot). |
| **2** | Symbol | Per-symbol offline results: champion vs incumbent ZΩ, accepted/rejected. Optuna/tournament summary. |
| **3** | Symbol/TF | Single-bot training detail: runtime learning (epsilon, β, buffer fill %, loss trend, reward trend), learned parameter change log. |
| **4** | Detail | Single training run metrics, checkpoint comparison. |

---

### 4. Risk (Tab `4`)

Account risk and per-bot risk separated. Mode-aware.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Account exposure, total open positions, margin used, portfolio drawdown. Paper and Live exposure stacked. |
| **2** | Symbol | Per-symbol risk: CB state, VaR, regime ζ, reward weights, path geometry. |
| **3** | Symbol/TF | Single-bot risk detail: kurtosis threshold vs current, VaR breakdown, no-entry reason, spread, VPIN-z. |

No Level 4.

---

### 5. Market (Tab `5`)

Market microstructure and L2 order book.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Compact market summary: spread, VPIN, regime per symbol (one row per symbol). |
| **2** | Symbol | Per-TF market table: spread, depth, VPIN-z, imbalance, regime ζ, volatility. |
| **3** | Symbol/TF | Single-TF detail: full order book ladder, depth ratio, L2 snapshot, signal synthesis. |

No Level 4.

---

### 6. Decision Log (Tab `6`)

Strategy reasoning and policy trace.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Newest-first across all bots, compact row (timestamp, bot, decision, confidence). |
| **2** | Symbol | Filtered to symbol, newest-first. |
| **3** | Symbol/TF | Filtered to symbol+TF, newest-first with full context row. |
| **4** | Detail | Decision card: full context, reasoning, linked trade_id, position_id. |

---

### 7. Trades (Tab `7`)

Executed trades and trade lifecycle. Separate tab from Decision Log.

| Level | Scope | Renders |
|-------|-------|---------|
| **1** | Portfolio | Per-instrument summary with **period columns** (Trades, PnL, WR per period). Paper stacked above Live. |
| **2** | Symbol | Per-TF metrics table with period columns (Trades, PnL, WR per TF per period). Paper above Live. |
| **3** | Symbol/TF | Individual trade list (paginated) — columns: #, mode badge, date/time, dir, entry, exit, PnL, cap%, MFE, MAE, bars, reason. |
| **4** | Detail | **Trade card**: full detail with trigger_data, exit_data, reward breakdown. |

Period cycling with `p` at Level 3 narrows trade list to that period.

---

## Navigation Summary

| Key | Action |
|-----|--------|
| `1`–`7` | Switch tab (preserves context hierarchy) |
| `Enter` | Drill down (highlighted row → next level) |
| `Esc` | Drill up (back one level) |
| `↑` `↓` or `j`/`k` | Move row selection |
| `s` | Jump-scope: Portfolio → Symbol → Symbol/TF (cycles) |
| `p` | Cycle period: 24h → 7d → Month → Epoch → Lifetime |
| `d` | Toggle detail/diagnostics pane (ALL tabs, consistent meaning) |
| `b` | Back — close detail pane |
| `Tab` / `Shift+Tab` | Cycle focus region |
| `r` | Review & reset circuit breakers |
| `e` | Set/clear stats epoch |
| `h` | Help overlay |
| `Alt+K` | Emergency kill |
| `q` | Quit HUD |

## HUD Sync Requirement For Code Changes

Any change to training, promotion, reward shaping, trade logging, risk,
decision logging, learned parameters, runtime metrics, or self-healing telemetry
must include a HUD impact check in the same change set.

The check must answer:

- Does the HUD still read the canonical source of truth?
- Did any field name, unit, scope, or period meaning change?
- Does a visible label now need renaming?
- Is any old metric pathway now deprecated or misleading?
- Are paper, live, and offline results still separated, with no blended metric
  rows?
- Do portfolio, symbol, and symbol/timeframe views still remain separate?
- Are tests or fixtures needed for the affected render helper?

## Implementation Path

1. Add a small presentation model between raw telemetry and render methods:
   `HudMetricCube`, `HudScope`, `HudPeriod`, and table row DTOs.
2. Move all period/scope aggregation into that presentation model.
3. Convert the default Performance tab to summary-first tables.
4. Add row selection and drill-down state before adding more detail.
5. Split Health from Overview so operational health, data freshness, and
   self-healing status have a stable home.
6. Add render tests for row width, no wrapping, naming consistency, and source
   field freshness.
