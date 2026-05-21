# Running the Trading Bot with Live Monitoring

This guide explains how to run the trading bot with real-time log streaming for debugging connection issues.

## Quick Start (Two Terminal Setup)

### Terminal 1: Start the Trading Bot
```bash
cd ~/Documents/ctrader_trading_bot
./scripts/run.sh
```

### Terminal 2: Stream Live Logs
```bash
cd ~/Documents/ctrader_trading_bot
./scripts/stream_logs.sh
```

---

## What You'll See in Each Terminal

### Terminal 1 (Bot Process)
- Startup sequence
- Environment validation
- Initial connection attempts
- HUD display (if enabled)
- Process health

### Terminal 2 (Live Logs - Color Coded)
- 🔴 **RED**: ERRORS, CRITICAL issues, Circuit Breakers
- 🟡 **YELLOW**: WARNINGS
- 🟢 **GREEN**: INFO messages
- 🔵 **CYAN**: DEBUG details
- 🟣 **MAGENTA**: Bar closes, Logon events, Position updates, Trades
- 🔵 **BLUE**: Training/Learning events

---

## Connection Stability Monitoring

The live log stream will show you:

### FIX Connection Events
```
[LOGON] QUOTE qual=QUOTE
[LOGON] TRADE qual=TRADE
[MAIN] Connection healthy
```

### Connection Issues
```
⚠ Connection unhealthy (consecutive failures: 1/5)
[RECONNECT] Attempting reconnection...
```

### Bar Building
```
[BAR M1] 2026-01-10T... O=2023.45 H=2023.67 L=2023.12 C=2023.34
```

### Training Activity
```
[ONLINE_LEARNING] TriggerAgent training step complete
[TRAIN] Buffer size: 1234, Loss: 0.0023
[CHECKPOINT] Restored metadata: {'trigger_training_steps': 11437, ...}
[BUFFER] Loaded 22511 experiences from data/checkpoints/XAUUSD_M1/trigger_buffer.npz
```

---

## Configuration (.env file)

Make sure your `.env` file has these settings:

```bash
# Required Credentials
CTRADER_USERNAME=your_username
CTRADER_PASSWORD_QUOTE=your_quote_password
CTRADER_PASSWORD_TRADE=your_trade_password

# Symbol Configuration
SYMBOL=BTCUSD           # Or XAUUSD, EURUSD, etc.
SYMBOL_ID=1             # Symbol ID from cTrader
QTY=0.01                # Position size (lots)
TIMEFRAME_MINUTES=1     # Bar timeframe (1, 5, 15, etc.)

# Connection Settings (optional - for debugging)
CTRADER_HEARTBEAT=15    # Heartbeat interval in seconds
CTRADER_RECONNECT_MAX=100  # Max reconnection attempts

# Logging Level (optional)
LOG_LEVEL=INFO          # DEBUG for more verbose output
```

---

## Common Issues & Solutions

### 1. Connection Drops Frequently

**Symptoms in logs:**
```
Connection unhealthy
Heartbeat timeout
Session disconnected
```

**Solutions:**
- Increase `CTRADER_HEARTBEAT` to 30 seconds
- Check your internet connection stability
- Verify firewall isn't blocking FIX protocol ports
- Check if cTrader servers are experiencing issues

### 2. No Market Data Received

**Symptoms:**
```
No bars being built
MarketDataSnapshot timeout
```

**Solutions:**
- Verify `SYMBOL` and `SYMBOL_ID` are correct
- Check trading hours for the symbol
- Ensure quote session is connected (look for `[LOGON] QUOTE`)

### 3. Slow Bar Building

**Symptoms:**
```
Bars arriving late
Significant gaps in timestamps
```

**Solutions:**
- Lower `TIMEFRAME_MINUTES` (e.g., use M1 instead of M15)
- Check system resources (CPU/RAM)
- Verify network latency is acceptable

### 4. Training Not Happening

**Symptoms:**
```
Buffer size not growing
No training steps in logs
HUD/training_stats shows last_training_time: Never
```

**Solutions:**
- Check if trades are executing
- Verify `DDQN_ONLINE_LEARNING=1` in `.env`
- Look for circuit breaker trips preventing trading
- Check the scoped runtime truth before assuming a cold start:

```bash
python3 -m json.tool data/training_stats_XAUUSD_M5.json | sed -n '1,100p'
python3 -m json.tool data/checkpoints/XAUUSD_M5/training_metadata.json | sed -n '1,100p'
rg 'Restored metadata|Loaded [0-9]+ experiences|Saved training metadata' logs/hub_XAUUSD.log
```

If `training_stats_<SYMBOL>_M<TF>.json` shows `last_training_time: Never` while
checkpoint metadata has non-zero steps and hub logs show buffer loads, treat it
as telemetry drift rather than lost learning state.

### 5. Scoped Circuit Breaker Lockout

**Symptoms:**
```
CB_LOCKOUT in data/performance_health.json
circuit breaker tripped - skip entry
0 trades for one symbol/timeframe while bars and quotes are fresh
```

**Solutions:**
- Prefer scoped analyzer self-healing first:

```bash
python3 scripts/performance_analyzer.py --auto-heal --hours 4
find data/paper_* -name circuit_breaker_reset.json -o -name learned_parameters_reload.json
```

- The analyzer writes `data/paper_<SYMBOL>_<TF>/circuit_breaker_reset.json` with
  `target_timeframes`; the OpenAPI hub resets only matching in-process agents.
- If stale in-memory state is suspected or several bots are locked, use the
  integrated restart path:

```bash
UNIVERSE_FIX_CB_LOCKOUT_ON_RESTART=1 ./run.sh universe
```

---

## Advanced Debugging

### Save Logs to File
```bash
# Terminal 2 - stream AND save to file
./scripts/stream_logs.sh | tee debug_session_$(date +%Y%m%d_%H%M%S).log
```

### Filter Specific Events
```bash
# Only show connection events
tail -f bot_console.log | grep -i "logon\|connection\|heartbeat"

# Only show trading events
tail -f bot_console.log | grep -i "bar\|position\|trade"

# Only show errors
tail -f bot_console.log | grep -i "error\|critical\|warning"
```

### Check Current Connection Status
```bash
# In another terminal
tail -20 data/system_status.json
```

---

## Stopping the Bot

### Graceful Shutdown
```bash
# In Terminal 1 (where bot is running)
Ctrl+C
```

This will:
1. Stop accepting new signals
2. Save agent state
3. Close FIX sessions cleanly
4. Export final performance report

### Force Kill (if frozen)
```bash
pkill -f ctrader_ddqn_paper.py
```

---

## Log File Locations

The bot writes logs to multiple locations:

1. **Console Output**: `bot_console.log` (main log, streamed by `stream_logs.sh`)
2. **Python Logs**: `logs/python/bot.log`
3. **cTrader Logs**: `logs/ctrader/app.log`
4. **Startup Log**: `logs/startup.log` (from run.sh)
5. **Universe Watcher**: `logs/universe_watch.log` when the watcher is started
   with `nohup`/background redirection
6. **Offline Training Recovery**: `logs/train_offline_supervisor.log` for
   watcher-managed offline-training restarts and resumed training output

### Offline Training Recovery Checks

When weekend/offline training is interrupted, `run_universe.py --watch` checks
`data/offline_training_status.json` and restarts unfinished queued/running work
if no live `train_offline.py` process exists. Confirm recovery with:

```bash
pgrep -af 'run_universe.py --watch|train_offline.py'
python3 -m json.tool data/offline_training_status.json | sed -n '1,120p'
tail -f logs/train_offline_supervisor.log
```

Autorestart is on by default. Disable it for manual recovery with:

```bash
UNIVERSE_OFFLINE_AUTORESTART=0 python3 run_universe.py --watch
```

---

## Monitoring Checklist

Before starting a live/paper trading session, verify:

- [ ] `.env` file configured correctly
- [ ] Virtual environment activated
- [ ] Both FIX sessions connect (QUOTE + TRADE)
- [ ] Market data flowing (bars being built)
- [ ] Circuit breakers not tripped
- [ ] HUD data being exported (if using HUD)
- [ ] Training enabled (if using online learning)
- [ ] Performance tracker recording trades

---

## Example Session

**Terminal 1:**
```bash
$ ./scripts/run.sh
✓ Loading environment from .env
✓ All required environment variables are set
✓ Activating virtual environment
✓ Python environment ready
✓ Quote config: config/ctrader_quote.cfg
✓ Trade config: config/ctrader_trade.cfg
✓ Log directories ready
✓ Trading bot running (PID: 12345)
```

**Terminal 2:**
```bash
$ ./scripts/stream_logs.sh
═══════════════════════════════════════════════════════════════
  cTrader Trading Bot - Live Log Monitor
═══════════════════════════════════════════════════════════════

✓ Found log file: bot_console.log
  Streaming live logs (Ctrl+C to stop)...

───────────────────────────────────────────────────────────────

[MAIN] FIX initiators started (QUOTE + TRADE)
[LOGON] QUOTE qual=QUOTE
[LOGON] TRADE qual=TRADE
[MAIN] Connection healthy (uptime: 0h 0m 5s)
[BAR M1] 2026-01-10T10:30:00 O=2023.45 H=2023.67 L=2023.12 C=2023.34
[POLICY] TriggerAgent: NO_ENTRY (confidence=0.234)
...
```

---

**For more details, see:**
- [SYSTEM_FLOW.md](SYSTEM_FLOW.md) - Complete execution flow
- [GAP_ANALYSIS_AND_REMEDIATION_SCHEDULE.md](GAP_ANALYSIS_AND_REMEDIATION_SCHEDULE.md) - Known issues
- [MASTER_HANDBOOK.md](MASTER_HANDBOOK.md) - System architecture
