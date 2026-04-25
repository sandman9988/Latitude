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
- Tournament variants are evaluated per symbol/timeframe. Promote only the best
  accepted candidate for that exact pair.
- Acceptance must beat the evaluated runtime incumbent and the registered
  champion/live universe guard, with the configured acceptance margin.
- If a candidate does not beat both guards, leave the current champion in place
  and run the configured retrain rounds instead of degrading the live pipeline.
- Focused replay should use the 10 best and 10 worst recent capture records per
  symbol/timeframe by default, without contaminating another timeframe.

## Runtime And HUD Rules

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
- Multiple simultaneously running bots need a portfolio/account view before
  making account-level exposure decisions. Per-timeframe bots may learn
  independently, but order ownership and exposure should be reconciled through
  the broker/account source of truth.

## Operational Safety

- Do not commit `data/`, `logs/`, `trades/`, `store/`, `.env`, credentials,
  model artifacts, or live runtime outputs.
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
