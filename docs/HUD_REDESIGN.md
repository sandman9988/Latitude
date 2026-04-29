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

## Proposed Information Architecture

### 1. Command Center

Default landing tab. One screen should fit in an 80x24 terminal.

Sections:

- Fleet status: running bots, stale bots, FIX status, data freshness, open
  positions, active circuit breakers.
- Portfolio summary by period: `24h`, `7d`, `Month`, `Epoch`, `Lifetime`.
  Columns: trades, PnL, Win%, PF, MaxDD, TQ, EQ, Runway Acc.
- Improvement summary: latest offline run status, accepted/rejected champion
  changes, and self-healing correction count.
- Exception strip: worst 3 anomalies across health, risk, data freshness, and
  self-healing corrections.
- Active scope hint: selected symbol/timeframe and available drill-down keys.

### 2. Performance

Summary first, drill-down second.

Default view:

- Portfolio period table.
- Mode summary table: `Paper`, `Live`, and `Offline` shown separately. Offline
  rows should use validation/backtest fields, not account PnL fields.
- Symbol summary table sorted by risk-adjusted degradation first.
- Timeframe heatmap-style table with one row per `Symbol / TF`.

Drill-down view for a selected `Symbol / TF`:

- Period table for that bot only.
- Trade quality and edge quality for the same periods.
- Prediction convergence: runway bias, runway accuracy, confidence Brier error,
  sample count, and source-field freshness.
- Top/bottom recent trades by capture and expectancy contribution.

### 3. Health

Operational health and telemetry freshness.

Sections:

- Process/runtime: watcher, bot PIDs, uptime, restart counts, memory, errors.
- Data freshness: scoped stats files, decision logs, trade log, order book,
  performance health report.
- Self-healing analyzer: overall health, anomaly codes, last correction, next
  run estimate.
- HUD data integrity: unknown timeframe count, inferred mode count, missing
  field counts, deprecated field usage.

### 4. Risk

Account and bot risk separated.

Portfolio section:

- Account exposure, open positions, margin, drawdown, emergency close state.

Per-bot section:

- Kurtosis action threshold, current kurtosis, VaR, no-entry reason, spread,
  VPIN, runway friction, reward-shaping guards.

### 5. Training And Improvement

Continuous improvement results, not raw training noise.

Sections:

- Offline training queue/status by `Symbol / TF`.
- Champion/incumbent comparison: candidate ZOmega, incumbent ZOmega, accepted
  or rejected, acceptance guard reason.
- Optuna/tournament result summary.
- Offline-to-runtime promotion trace: source checkpoint, promoted universe path,
  runtime checkpoint sync status, and whether a bot restart is pending/done.
- Runtime learning: epsilon, buffer fill, loss trend, reward trend, checkpoint
  freshness.
- Learned parameter changes: last changed value, direction, source, and reason.

### 6. Market

Market conditions for the selected bot.

Default view should stay scoped to the active `Symbol / TF` and show spread,
depth, imbalance, VPIN, regime, volatility, and signal synthesis. Portfolio-wide
market tables belong in drill-down or summary strips only.

### 7. Logs And Trades

Operational trace and forensic detail.

- Decision log should be newest-first, scoped by current selection by default,
  with an explicit `Portfolio` mode for cross-bot inspection.
- Trades should default to recent closed trades for the active selection.
- Trade detail should expose linked `trigger_data`, `exit_data`, and reward
  breakdowns, but the row list should stay compact.

## Navigation

- `1` to `7`: switch primary view.
- `s`: select `Portfolio`, `Symbol`, or `Symbol / TF` scope.
- `Enter`: drill into the highlighted row.
- `Esc`: move one level up.
- `Tab` / `Shift+Tab`: cycle focus within the current view.
- Arrow keys: move row selection or switch tabs when no table is focused.
- `/`: filter current table.

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
