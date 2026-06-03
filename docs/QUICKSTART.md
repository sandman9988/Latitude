# Quick Start Guide

**Last Updated:** April 25, 2026
**Status:** Active
**Audience:** All

---

## Prerequisites

- Python 3.12+, pip, git
- cTrader account with FIX API credentials
- `.env` file populated from `.env.example`
- Historical OHLCV CSV for at least one symbol (e.g. XAUUSD M5)

---

## 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Configure credentials

```bash
cp .env.example .env
# Edit .env: set CTRADER_ACCOUNT_ID, FIX_SENDER_COMP_ID, passwords, etc.
```

FIX session config lives in `config/ctrader_quote.cfg` and `config/ctrader_trade.cfg`.
For universe mode the supervisor generates per-bot copies automatically.

---

## 3. Train a model offline

```bash
python train_offline.py \
  --symbol XAUUSD \
  --timeframe 5 \
  --data data/XAUUSD_M5.csv \
  --epochs 3
```

Accepts candidates with ZΩ ≥ 1.0 and writes them to `data/checkpoints/offline_champions.json`.

Promote a champion to the live universe:

```bash
python train_offline.py --auto-promote --symbol XAUUSD --timeframe 5
```

This sets the entry's stage to `PAPER` in `data/universe.json` and records `weights_path`.

---

## 4. Run a single paper bot

```bash
./run.sh --symbol XAUUSD --timeframe 5 --paper
```

Runtime data for this bot lands in `data/paper_XAUUSD_M5/`.  
HUD data, decision logs, and checkpoints are all scoped to that directory.

---

## 5. Run the universe supervisor (recommended)

The supervisor launches and keeps alive a paper bot for every `PAPER`-stage entry in `data/universe.json`:

```bash
python run_universe.py --watch
```

Each bot is **fully isolated** per `(symbol, timeframe_minutes)`:

| Resource | Path |
| -------- | ---- |
| FIX config | `data/paper_XAUUSD_M5/fix/` |
| Checkpoint | `data/paper_XAUUSD_M5/checkpoints/XAUUSD_M5/` |
| HUD data | `data/paper_XAUUSD_M5/` |
| Log | `logs/paper_XAUUSD_M5.log` |

Promoted weights are synced from `universe.json → weights_path` into the bot's runtime checkpoint dir before each launch. If a running bot's weights go stale it is restarted automatically.

**Broker topology** (`UNIVERSE_BROKER_TOPOLOGY` env var):

| Mode | Behaviour |
| ---- | --------- |
| `isolated` (default) | Each bot owns its own QUOTE+TRADE FIX session pair |
| `shared-symbol` | One QUOTE session shared per symbol; TRADE sessions isolated per TF |
| `shared-account` | Single QUOTE+TRADE pair shared across all bots |

```bash
# Example: transitional guard — one direct-FIX owner per account
python run_universe.py --watch --broker-topology shared-account
```

---

## 6. Open the HUD

```bash
./run.sh --hud-only
```

The HUD auto-discovers all running bots by reading scoped JSON files
(`production_metrics_XAUUSD_M5.json`, `order_book_XAUUSD_M5.json`, etc.) from `data/`.

- **Tab / Shift-Tab** — switch bot view
- **Alt+K** — emergency kill-switch for the focused bot (independent of bar close)
- 7 tabs: Overview · Market · Performance · Trades · Training · Agents · System

---

## 7. Run the test suite

```bash
python -m pytest tests/ -q
```

---

## 8. Weekend offline training

```bash
./run.sh weekend-train-setup   # installs cron entry (market-closed guard included)
./run.sh weekend-train         # run manually (safe: exits if market is open)
```

See [TRAINING_TO_PRODUCTION_GUIDE.md](TRAINING_TO_PRODUCTION_GUIDE.md) for the full champion acceptance workflow.

Optional Optuna search for one scoped bot:

```bash
python train_offline.py data/ \
  --symbols XAUUSD --timeframes M5 \
  --optuna-trials 12 --accept-if-better
```

Optuna studies are stored per symbol/timeframe under `data/optuna/` and winners
still pass through the normal incumbent/champion guard.

---

## Key paths

| Path | Purpose |
| ---- | ------- |
| `data/universe.json` | Fleet registry — stage, z_omega, weights_path per (symbol, TF) |
| `data/checkpoints/offline_champions.json` | Offline training acceptance guard (source of truth) |
| `data/paper_XAUUSD_M5/` | Runtime data dir for this bot |
| `data/paper_XAUUSD_M5/logs/audit/decisions.jsonl` | Per-bot decision audit log |
| `config/learned_parameters.json` | Adaptive thresholds, keyed `XAUUSD_M5_default` |
| `data/reward_shaping_monitor_XAUUSD_M5.json` | Hourly quality-guard output for this bot |
| `logs/paper_XAUUSD_M5.log` | Per-bot process log |

---

## Scoping rule

> Every metric, parameter, checkpoint, decision log, cache, and reward monitor output
> is scoped by `(symbol, timeframe_minutes)`.  The canonical label for H4 is `M240` —
> never a separate H4 runtime path.

---

**Navigation:** [🏠 Root](../README.md) | [📖 Index](INDEX.md) | [📄 Current State](CURRENT_STATE.md) | [🔧 Training Guide](TRAINING_TO_PRODUCTION_GUIDE.md)
