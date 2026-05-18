# Training to Production Transition Guide

## Overview
This guide explains how to train your bot with exploration/online learning, then transition to production with learned parameters.

All training state is scoped by `(symbol, timeframe_minutes)`. A model,
parameter, metric, decision log, cache, reward-shaping recommendation, or
checkpoint for one timeframe must not be reused as the source of truth for
another timeframe.

---

## Weekend Offline Champion Workflow

Weekend retraining is the standard closed-market improvement loop. It is
designed to improve each active symbol/timeframe without degrading the current
pipeline.

### Install Cron

```bash
./run.sh weekend-train-setup
```

Default cron:

- Time zone: `Africa/Johannesburg`
- Schedule: Saturday 03:00 local time
- Guarded market window: Friday 22:00 UTC through Sunday 21:30 UTC

### Run Manually

```bash
./run.sh weekend-train
```

The script discovers all per-bot training caches under `data/`, stops the
universe watcher only when needed, runs `train_offline.py`, syncs accepted
champions into isolated runtime checkpoint directories, and restarts the watcher
if it was running before training.

Saturday scheduled retraining starts a fresh weekly queue by default. It archives
the previous `data/offline_training_status.json`, sets
`CTRADER_OFFLINE_RESUME_STATUS=0` for the new run, and replaces any stale
repo-local `train_offline.py` process that would otherwise block the full
symbol/timeframe retrain. Once the new run has started, normal watcher recovery
uses resume mode so interrupted unfinished rows continue instead of starting the
whole weekend queue again.

On GPU hosts the weekend script opts into parallel workers explicitly. Tune with
`WEEKEND_TRAIN_WORKERS=<N>` and disable that opt-in with
`WEEKEND_TRAIN_ALLOW_GPU_PARALLEL=0` if VRAM contention appears.

Accepted checkpoints are staged into the isolated paper runtime directories
incrementally while the weekend queue is still running. The wrapper watches
`data/checkpoints/offline_champions.json`, copies only changed accepted trigger
and harvester weights, and repeats every
`WEEKEND_TRAIN_INCREMENTAL_SYNC_SECS` seconds (default `300`). This keeps
completed timeframes ready for the final watcher restart instead of waiting for
the slowest timeframe before copying files.

M1 weekend jobs are capped by default with
`WEEKEND_TRAIN_MAX_BARS_BY_TF=M1=500000`. Higher timeframes still use the full
discovered history unless another cap is configured. This prevents deterministic
tournament variants from spending the entire weekend repeatedly replaying
multi-million-bar M1 datasets.

### Queue Recovery And Autorestart

Offline training is restartable while `data/offline_training_status.json` shows
queued or running jobs. New `train_offline.py` runs write supervisor metadata
into that file:

- process id and heartbeat
- Python executable
- original command arguments
- working directory
- restart count

`run_universe.py --watch` checks this status on startup and every supervisor
poll. If unfinished offline training has no live `train_offline.py` process, the
watcher restarts the saved command with `CTRADER_OFFLINE_RESUME_STATUS=1`. If the
process is live, it is left alone so duplicate offline trainers are not spawned.
If the status/progress files stop updating for longer than
`UNIVERSE_OFFLINE_STALL_SECS` (default `3600` seconds), the watcher terminates
and restarts the unfinished queue. Restarts are capped by
`UNIVERSE_OFFLINE_MAX_RESTARTS` (default `12`).

Resume keeps completed `(symbol, timeframe_minutes)` rows marked `done` and only
queues unfinished rows. Set `CTRADER_OFFLINE_RESUME_STATUS=0` only when you
intentionally want to ignore the previous status and rerun all discovered jobs.

For the scheduled weekend wrapper, `WEEKEND_TRAIN_FRESH_STATUS=1` is the default
so the Saturday 03:00 run retrains every discovered active symbol/timeframe. Set
`WEEKEND_TRAIN_FRESH_STATUS=0` only when manually continuing an existing weekend
queue outside watcher recovery.

Operational controls:

```bash
# Disable watcher-managed offline training recovery
UNIVERSE_OFFLINE_AUTORESTART=0 python3 run_universe.py --watch

# Use a shorter stall threshold while testing recovery
UNIVERSE_OFFLINE_STALL_SECS=900 python3 run_universe.py --watch

# Inspect recovery decisions and restarted training output
tail -f logs/train_offline_supervisor.log
```

For older interrupted runs that predate supervisor metadata, the watcher can
rebuild a recovery command from the unfinished status rows and the available
`data/training_cache_*_M*.jsonl`, `data/paper_*_M*/training_cache_*_M*.jsonl`,
and `data/history/*_M*.csv` inputs. New runs should rely on the explicit
metadata path.

### Acceptance Rules

For every `(symbol, timeframe_minutes)` pair:

1. Train deterministic tournament variants, or optional Optuna-generated
   candidates when `--optuna-trials N` is set.
2. Evaluate the current runtime incumbent.
3. Load the registered champion from `data/checkpoints/offline_champions.json`.
4. Fall back to the live `data/universe.json` score only when no registered
   champion exists.
5. Accept only if the candidate beats the evaluated incumbent and champion/live
   guard, plus any configured acceptance margin.
6. If the candidate does not beat both guards, leave the current champion in
   place and use the configured retrain rounds.

Historical logs such as `logs/train_offline.log` are diagnostics only and must
not be used as champion or acceptance sources.

### Optional Optuna Search

Optuna is available as an offline candidate generator for faster initial DDQN
convergence:

```bash
python3 train_offline.py data/ \
  --symbols XAUUSD --timeframes M5 \
  --optuna-trials 12 \
  --optuna-min-val-trades 5 \
  --accept-if-better --auto-promote
```

Each study is scoped per bot, for example
`data/optuna/offline_XAUUSD_M5.db`. Trials tune the same candidate fields used
by the tournament path: epochs, update cadence, epsilon schedule, penalty scale,
warm start, and focused capture replay passes. The Optuna objective combines
`ZOmega`, validation net PnL, average PnL, profit factor, and a low-validation
trade penalty so high capture with cents-only edge does not dominate.

Optuna does not promote weights directly. The selected trial still has to beat
the evaluated runtime incumbent and the registered offline champion.

### Focused Capture Replay

By default, weekend training includes focused replay from the 10 best and 10
worst recent capture records per symbol/timeframe. Keep this scoped: M1 replay
records must not contaminate M5, M15, or M240 training.

### Runtime Sync

Accepted champion weights are recorded in `data/universe.json`. On launch,
`run_universe.py --watch` compares those promoted weights with the isolated
runtime checkpoint directory, copies stale/missing trigger and harvester weights,
and restarts a running paper bot when the runtime weights differ.

---

## Phase 1: Training Mode (Learning & Exploration)

### Goal
Let the bot explore different strategies, learn from experiences, and adapt parameters through online learning.

### Configuration
```bash
# Load training environment
source .env.training

# Start bot with HUD
./run.sh --with-hud
```

### What Happens in Training Mode

**Exploration (PAPER_MODE=1)**
- **Epsilon-greedy exploration**: Bot randomly explores 100% initially, decaying to 10% over ~2000 decisions
- **Actions**: Random choice between LONG(1) and SHORT(2) - deliberately excludes NO_ENTRY(0) for faster learning
- **Forced exploration**: After 10 bars flat, bot is forced to take action to prevent staleness
- **No gating**: Confidence floors are disabled - bot learns through rewards, not restrictions

**Online Learning (DDQN)**
- **Experience buffer**: Every trade creates experiences (state, action, reward, next_state)
  - Trigger buffer: Entry decisions
  - Harvester buffer: Exit decisions
  - Capacity: 50,000 experiences each
  - Min before training: 100 experiences
- **DDQN training**: Batch updates (size 64) using prioritized experience replay
  - Learning rate: 0.0005
  - Discount factor (γ): 0.99
  - Soft update (τ): 0.005
- **Parameter adaptation**: Confidence floors, profit targets, etc. adapt based on performance

**Reward Functions**
- **Trigger (Entry)**: Runway utilization = actual_MFE / predicted_runway
  - Good entries → high MFE → positive reward → reinforce behavior
  - Bad entries → low MFE → negative reward → discourage behavior
- **Harvester (Exit)**: Capture ratio = realized_pnl / peak_pnl
  - Good exits → high capture → positive reward
  - Too early/late exits → low capture → negative reward

### Monitoring Training

**Real-time monitoring**:
```bash
./scripts/monitor_training.sh
```

**Watch logs for training activity**:
```bash
tail -f logs/*.log | grep -i "training step\|experience added\|epsilon"
```

**Check exploration logs**:
```bash
tail -f logs/*.log | grep -i "explore\|random action"
```

**Monitor learned parameters**:
```bash
watch -n 5 'cat data/learned_parameters.json | jq ".data.instruments.XAUUSD_M15_default.params | {confidence_floor, entry_confidence_threshold, exit_confidence_threshold, profit_target}"'
```

### Training Metrics to Track

1. **Experience Buffer Size**
   - Trigger buffer: Growing toward 50,000
   - Harvester buffer: Growing toward 50,000
   - Target: At least 5,000+ experiences before transitioning

2. **Training Loss**
   - Should decrease over time as network learns
   - Spikes are normal during exploration
   - Look for general downward trend

3. **Epsilon Decay**
   - Starts at 1.0 (100% exploration)
   - Decays to 0.1 (10% exploration)
   - ~2000 decisions to reach minimum
   - Check logs: `[TRIGGER] EXPLORE: random action=X (ε=Y.YYY)`

4. **Parameter Adaptation**
   - `confidence_floor`: Should stabilize around optimal trade selectivity
   - `entry_confidence_threshold`: Adjusts based on runway prediction accuracy
   - `exit_confidence_threshold`: Adjusts based on capture ratio
   - `profit_target`: Learns optimal exit level based on market behavior
   - Check for convergence: parameters stop changing drastically

5. **Trade Performance**
   - Win rate improving over time
   - Average P&L per trade stabilizing or improving
   - Reduced variance in outcomes (more consistent)

### When to Transition to Production

**Readiness Checklist**:
- ✅ **Experience**: At least 500-1000 trades completed
- ✅ **Buffer**: 5,000+ experiences in both buffers
- ✅ **Epsilon**: Decayed to ~0.1 or lower (minimal exploration)
- ✅ **Parameters converged**: No large swings in learned parameters for 100+ trades
- ✅ **Performance**: Positive or neutral P&L trajectory over last 200 trades
- ✅ **Stability**: No crashes, memory leaks, or errors in logs
- ✅ **Training loss**: Stabilized (not increasing consistently)

**Estimated Timeline**:
- Fast learning (15-minute timeframe): 3-7 days
- Moderate learning (1-hour timeframe): 1-2 weeks
- Conservative (4-hour timeframe): 2-4 weeks

### Backup Before Transition

```bash
# Backup learned parameters
cp data/learned_parameters.json data/learned_parameters.json.trained_$(date +%Y%m%d_%H%M%S)

# Backup experience buffers (if persisted)
cp -r data/checkpoints data/checkpoints_trained_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

# Backup trade history
cp data/trade_log.jsonl data/trade_log.jsonl.trained_$(date +%Y%m%d_%H%M%S)
```

---

## Phase 2: Production Mode (Exploitation)

### Goal
Use learned parameters and confidence gates to trade conservatively with minimal exploration.

### Transition Steps

**1. Stop training bot**:
```bash
pkill -f "python3 -m src.core.ctrader_ddqn_paper"
sleep 3
```

**2. Source production environment**:
```bash
source .env.production
```

**3. Verify learned parameters loaded**:
```bash
cat data/learned_parameters.json | jq ".data.instruments.XAUUSD_M15_default.params | {confidence_floor, entry_confidence_threshold, exit_confidence_threshold, profit_target}"
```

**4. Start production bot**:
```bash
./run.sh --with-hud
```

**5. Verify production mode active**:
```bash
./scripts/monitor_training.sh
```
Should show "Mode: PRODUCTION" with epsilon very low (0.01-0.05)

### What Changes in Production Mode

**Gating Enabled (PAPER_MODE=0)**:
- **Confidence floor**: Only takes trades above learned threshold (e.g., 0.60)
- **Feasibility gate**: Rejects trades with poor path geometry
- **Economics threshold**: Requires positive expected value after friction
- **No forced exploration**: Bot can stay flat indefinitely if no good setups

**Minimal Exploration (ε=0.01)**:
- Only 1% of decisions explore randomly
- 99% use learned policy
- Enables continuous adaptation to market regime changes
- Prevents overfitting to historical patterns

**Conservative Risk Management**:
- Uses learned profit targets and stop losses
- Position sizing based on learned parameters
- Circuit breakers and drawdown limits active
- Max daily loss enforced

### Monitoring Production

**Performance dashboard**:
```bash
./scripts/monitor_production.sh  # (create this similar to monitor_training.sh)
```

**Track key metrics**:
```bash
# Watch trades
tail -f data/trade_log.jsonl | jq -r '[.trade_id, .entry_type, .pnl, .capture_ratio] | @tsv'

# Check P&L
python3 -c "import json; trades = [json.loads(l) for l in open('data/trade_log.jsonl')]; print(f'Total P&L: {sum(t[\"pnl\"] for t in trades):.2f}')"

# Monitor adaptive parameters (should change slowly, not drastically)
watch -n 30 'cat data/learned_parameters.json | jq ".data.instruments.XAUUSD_M15_default.params | {confidence_floor, profit_target}"'
```

### Production Health Checks

**Daily**:
- [ ] P&L trajectory (cumulative chart)
- [ ] Drawdown vs limits
- [ ] Trade frequency (not too many, not too few)
- [ ] Win rate consistency
- [ ] Check logs for errors

**Weekly**:
- [ ] Compare learned parameters vs. last week
- [ ] Review worst trades for patterns
- [ ] Check if confidence floor needs manual adjustment
- [ ] Verify circuit breakers haven't triggered excessively

**Monthly**:
- [ ] Full performance report
- [ ] Consider re-training if market regime changed dramatically
- [ ] Update profit targets if instrument volatility changed
- [ ] Backup learned parameters

---

## Phase 3: Continuous Adaptation (Production with Learning)

### Optional: Keep Online Learning Enabled in Production

**Tradeoff**:
- ✅ **Pro**: Bot adapts to changing market conditions
- ✅ **Pro**: Parameters evolve with regime shifts
- ⚠️ **Con**: Risk of overfitting to recent poor performance
- ⚠️ **Con**: Can diverge from well-trained state

**How to enable**:
1. Keep `DDQN_ONLINE_LEARNING=1` (default)
2. Use production epsilon schedule (ε=0.01, very low exploration)
3. Enable PAPER_MODE=0 (gates active, conservative)
4. Monitor parameter drift closely
5. Set up parameter bounds to prevent wild divergence

**Recommended for**:
- High-frequency trading (fast adaptation needed)
- Markets with frequent regime changes
- After 6+ months of production (re-training)

**Not recommended for**:
- First production deployment (use static learned params)
- Unstable markets with extreme volatility
- When conservative capital preservation is priority

---

## Rollback Procedure (If Production Performs Poorly)

**1. Stop production bot**:
```bash
pkill -f "python3 -m src.core.ctrader_ddqn_paper"
```

**2. Restore previous learned parameters**:
```bash
cp data/learned_parameters.json data/learned_parameters.json.production_failed_$(date +%Y%m%d_%H%M%S)
cp data/learned_parameters.json.trained_YYYYMMDD_HHMMSS data/learned_parameters.json
```

**3. Return to training mode OR use conservative defaults**:
```bash
# Option A: Re-train
source .env.training && ./run.sh --with-hud

# Option B: Use defaults and monitor
# Edit data/learned_parameters.json manually to reset to conservative values:
# confidence_floor: 0.60
# entry_confidence_threshold: 0.60
# exit_confidence_threshold: 0.50
# profit_target: 1.01
./run.sh --with-hud
```

**4. Review logs for root cause**:
```bash
grep -i "error\|exception\|blocked by\|blocked.*gate" logs/*.log | tail -50
```

---

## Quick Reference

### Training Mode
```bash
source .env.training && ./run.sh --with-hud
./scripts/monitor_training.sh
```

### Production Mode
```bash
pkill -f ctrader_ddqn_paper && sleep 3
source .env.production && ./run.sh --with-hud
```

### Check Current Mode
```bash
./scripts/monitor_training.sh | grep "Mode:"
```

### Emergency Stop
```bash
pkill -f "python3 -m src.core.ctrader_ddqn_paper"
```

---

## Summary

| Phase | PAPER_MODE | Epsilon | Gates | Goal |
|-------|------------|---------|-------|------|
| Training | 1 | 1.0→0.1 | Disabled | Learn optimal policy |
| Production | 0 | 0.05→0.01 | Enabled | Execute learned policy |
| Continuous | 0 | 0.01 | Enabled | Adapt while trading |

**Key Insight**: 
- **Training = Explore** (high epsilon, no gates, pure learning)
- **Production = Exploit** (low epsilon, gates active, use learned knowledge)
- **Line 317 (`random.choice([1, 2])`** = Training-only exploration - NOT used in production!

---

## Next Steps

1. ✅ Configuration files created (`.env.training`, `.env.production`)
2. ✅ Monitoring script created (`scripts/monitor_training.sh`)
3. ⏳ **Start training**: `source .env.training && pkill -f ctrader_ddqn_paper && sleep 3 && ./run.sh --with-hud`
4. ⏳ Monitor for 500-1000 trades
5. ⏳ Transition to production when ready
6. ⏳ Enjoy profitable automated trading! 🚀
