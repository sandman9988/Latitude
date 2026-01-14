# Disaster Recovery Runbook
**Project:** cTrader Adaptive Trading Bot  
**Version:** 1.0  
**Last Updated:** 2026-01-11  
**Owner:** Operations Team

---

## 🚨 EMERGENCY CONTACTS

**Primary On-Call:** [Your Name/Contact]  
**Backup On-Call:** [Backup Contact]  
**Broker Support:** cTrader Support Hotline  
**Critical Escalation:** [Escalation Contact]

---

## QUICK REFERENCE - EMERGENCY PROCEDURES

| Scenario | Immediate Action | Command |
|----------|------------------|---------|
| **Runaway Trading** | Kill bot immediately | `./scripts/emergency_shutdown.sh` |
| **Bot Crashed** | Check positions, restore state | `./scripts/check_positions.sh` |
| **Corrupt State** | Restore from backup | `./scripts/restore_backup.sh` |
| **Network Outage** | Wait for auto-reconnect (5 min) | Monitor logs |
| **Bad Performance** | Review logs, consider halt | `tail -1000 logs/python/app.log` |

---

## SCENARIO 1: BOT CRASHES MID-TRADE

### Symptoms
- Bot process terminated unexpectedly
- Logs show exception or crash
- Uncertain if open positions exist
- Last known state unclear

### IMMEDIATE ACTIONS (First 5 Minutes)

#### Step 1: Check if Positions Are Open
```bash
# Check current positions via cTrader
./scripts/check_positions.sh

# Or manually check cTrader platform
# Look for: Open positions, pending orders
```

**Decision Point:**
- **If positions open:** Proceed to Step 2
- **If no positions:** Proceed to Step 4 (State Recovery)

#### Step 2: Assess Position Risk
```bash
# Check position details
./scripts/get_position_summary.sh

# Look for:
# - Position size
# - Current P&L (unrealized)
# - Distance from stop loss
```

**Decision Point:**
- **If P&L acceptable and stop loss in place:** Leave positions (proceed to Step 4)
- **If large loss or no stop loss:** CLOSE POSITIONS IMMEDIATELY (Step 3)

#### Step 3: Emergency Position Close (If Required)
```bash
# Option A: Close via emergency script
./scripts/emergency_close_all.sh

# Option B: Manual close via cTrader platform
# 1. Log into cTrader
# 2. Go to Positions tab
# 3. Click "Close All" or close individually
```

**⚠️ WARNING:** Do NOT restart bot until positions are closed or protected!

#### Step 4: Restore Bot State from Journal
```bash
# Check journal file integrity
./scripts/verify_journal.sh

# Restore state from journal (if available)
python scripts/restore_from_journal.py --journal logs/journal/state.journal

# Verify restored state
python scripts/verify_state.py --state-dir state/
```

**Expected Output:**
```
✓ Journal replay complete: 1,247 operations
✓ State restored to bar 45,623
✓ Last trade: 2026-01-11 14:23:45 UTC
✓ Open positions: 0
✓ Checksum valid: CRC32 verified
```

#### Step 5: Verify State Consistency
```bash
# Run consistency checks
python scripts/check_state_consistency.py

# Check for:
# - Trade history completeness
# - Parameter file integrity
# - Model checkpoint validity
# - Experience buffer consistency
```

**Decision Point:**
- **If state valid:** Proceed to Step 6 (Restart)
- **If state corrupt:** Proceed to Scenario 2 (Corrupt State)

#### Step 6: Restart Bot with Verification
```bash
# Restart in monitoring mode (no trading for 10 bars)
./launch_micro_learning.sh --verify-mode

# Monitor logs in real-time
tail -f logs/python/app.log

# Look for:
# ✓ FIX connection established
# ✓ State loaded successfully
# ✓ Agents initialized
# ✓ First bar processed without errors
```

#### Step 7: Monitor First Hour
- Watch logs for 1 hour before leaving bot unattended
- Verify first trade executes correctly
- Check that metrics are being updated
- Ensure no repeated crashes

### POST-INCIDENT REVIEW
```bash
# Collect crash artifacts
./scripts/collect_crash_artifacts.sh crash_$(date +%Y%m%d_%H%M%S)

# Artifacts include:
# - Core dump (if available)
# - Last 10,000 log lines
# - State files (pre-crash)
# - Trade history
# - System metrics (memory, CPU)
```

**Root Cause Analysis:**
1. Review exception traceback in logs
2. Check for memory issues (OOM killer?)
3. Review recent code changes
4. Check system resource availability
5. Document findings in incident log

---

## SCENARIO 2: CORRUPT STATE FILES

### Symptoms
- Bot fails to load state files
- CRC32 checksum failures
- JSON parse errors
- State files missing or truncated

### IMMEDIATE ACTIONS

#### Step 1: Emergency Shutdown (If Not Already Stopped)
```bash
./scripts/emergency_shutdown.sh
```

#### Step 2: Assess Damage
```bash
# Check which state files are corrupt
./scripts/diagnose_state_files.sh

# Output shows:
# ✓ learned_parameters.json - OK
# ✗ experience_buffer.pkl - CORRUPT (CRC mismatch)
# ✓ agent_checkpoints/ - OK
# ✗ trade_history.json - CORRUPT (truncated)
```

#### Step 3: Restore from Backups
```bash
# List available backups
ls -lh backups/state/

# Example output:
# state_backup_20260111_000000.tar.gz  (automated daily)
# state_backup_20260111_120000.tar.gz  (automated 12-hour)
# state_backup_20260111_143000.tar.gz  (pre-crash snapshot)

# Restore from most recent valid backup
./scripts/restore_backup.sh backups/state/state_backup_20260111_143000.tar.gz
```

**⚠️ WARNING:** Restoring from backup will lose all state since backup time!

#### Step 4: Replay Journal to Recover Lost State
```bash
# Replay journal from backup point to crash
python scripts/replay_journal.py \
    --from-backup 20260111_143000 \
    --to-crash

# This reconstructs state by replaying all operations:
# - Trade executions
# - Parameter updates
# - Agent learning steps
# - Experience additions
```

**Expected Output:**
```
✓ Backup state loaded: bar 45,120
✓ Replaying journal from bar 45,120 to 45,623
✓ 503 operations replayed
✓ State reconstructed to bar 45,623
✓ Checksums verified
```

#### Step 5: Validate Restored State
```bash
# Run full validation suite
python scripts/validate_state.py --full

# Checks:
# ✓ Parameter files loadable
# ✓ Model checkpoints valid (can load into PyTorch)
# ✓ Experience buffer accessible
# ✓ Trade history complete (no gaps)
# ✓ Ring buffers initialized
# ✓ VaR estimates reasonable
```

**Decision Point:**
- **If validation passes:** Proceed to Step 6 (Restart)
- **If validation fails:** Proceed to Step 7 (Manual Recovery)

#### Step 6: Restart Bot
```bash
./launch_micro_learning.sh --verify-mode
```

#### Step 7: Manual Recovery (If Automated Fails)
```bash
# Last resort: Start fresh from checkpoint but keep trade history

# 1. Initialize new state
python scripts/initialize_fresh_state.py --keep-trade-history

# 2. Restore trade history from backup
cp backups/state/trade_history.json state/

# 3. Initialize agents with default parameters
# (This loses learned parameters but keeps trading safe)

# 4. Start in OBSERVATION phase to re-learn
./launch_micro_learning.sh --cold-start
```

**⚠️ WARNING:** Manual recovery loses learned parameters. Bot will need to re-learn!

---

## SCENARIO 3: RUNAWAY TRADING

### Symptoms
- Unusually high trade frequency (>5 trades/minute)
- Large positions opened without authorization
- Circuit breakers not triggering
- Losses mounting rapidly

### 🚨 IMMEDIATE ACTION - DO NOT DELAY

#### Step 1: EMERGENCY SHUTDOWN (30 Seconds)
```bash
# Kill bot immediately - don't wait for graceful shutdown
./scripts/emergency_shutdown.sh --force

# If script fails, manual kill:
pkill -9 -f ctrader_ddqn_paper
```

#### Step 2: Close All Positions Manually (2 Minutes)
```bash
# Option A: Via cTrader platform (FASTEST)
# 1. Log into cTrader
# 2. Click "Close All Positions"
# 3. Confirm closure

# Option B: Via emergency script
./scripts/emergency_close_all.sh --market-orders
```

**⚠️ DO NOT RESTART BOT UNTIL ROOT CAUSE IDENTIFIED!**

#### Step 3: Secure the System (5 Minutes)
```bash
# Disable auto-restart (if configured)
systemctl disable ctrader-bot.service

# Move bot executable to prevent accidental restart
mv ctrader_ddqn_paper.py ctrader_ddqn_paper.py.DISABLED

# Lock state directory
chmod 000 state/
```

#### Step 4: Review Logs - Identify Root Cause (15 Minutes)
```bash
# Get last 2000 log lines
tail -2000 logs/python/app.log > /tmp/runaway_incident.log

# Look for:
grep "TRADE EXECUTED" /tmp/runaway_incident.log | wc -l  # Count trades
grep "circuit_breaker" /tmp/runaway_incident.log          # Why didn't they trip?
grep "ERROR\|EXCEPTION" /tmp/runaway_incident.log         # Any errors?
grep "reward" /tmp/runaway_incident.log | tail -50        # Reward anomalies?

# Check decision logs
cat logs/decisions/decision_*.jsonl | jq '.confidence' | tail -50
```

**Common Root Causes:**
1. **Circuit breaker bug** - Not triggering when it should
2. **Reward gaming** - Agent found exploit in reward function
3. **Parameter corruption** - Learned parameters became extreme values
4. **Logic bug** - Recent code change introduced infinite loop
5. **External manipulation** - State files manually edited incorrectly

#### Step 5: Fix Root Cause
```bash
# Example: Circuit breaker bug
# 1. Review circuit_breakers.py for logic errors
# 2. Check recent commits: git log --oneline -20
# 3. Revert problematic change: git revert <commit>

# Example: Reward gaming
# 1. Review reward_integrity_monitor.py output
# 2. Check reward-P&L correlation: python scripts/check_reward_correlation.py
# 3. Adjust reward function to close exploit

# Example: Parameter corruption
# 1. Restore learned_parameters.json from backup
# 2. Add validation to prevent extreme values
```

#### Step 6: Implement Safeguards Before Restart
```bash
# Add additional circuit breaker
cat >> config/circuit_breakers_override.json << EOF
{
    "max_trades_per_hour": 20,
    "max_trades_per_day": 100,
    "max_position_size": 0.01,
    "emergency_mode": true
}
EOF

# Enable enhanced monitoring
cat >> config/monitoring_override.json << EOF
{
    "alert_on_high_frequency": true,
    "trade_frequency_threshold": 3,
    "require_human_approval_above": 0.05
}
EOF
```

#### Step 7: Controlled Restart in Safe Mode
```bash
# Unlock state directory
chmod 755 state/

# Restore executable
mv ctrader_ddqn_paper.py.DISABLED ctrader_ddqn_paper.py

# Start in OBSERVATION mode only (no trading)
./launch_micro_learning.sh --observe-only --max-runtime 3600

# Monitor closely for 1 hour
# If stable, gradually enable trading with micro positions
```

### POST-INCIDENT MANDATORY STEPS
1. **Document incident** in `incidents/runaway_$(date +%Y%m%d).md`
2. **Review all circuit breakers** - ensure comprehensive coverage
3. **Add regression test** for identified bug
4. **Consider code freeze** until stability confirmed
5. **Notify stakeholders** of incident and remediation

---

## SCENARIO 4: NETWORK OUTAGE / FIX DISCONNECT

### Symptoms
- FIX connection status: DISCONNECTED
- Logs show "Connection refused" or "Timeout"
- No market data updates
- Cannot execute trades

### ACTIONS

#### Step 1: Assess Duration (First 2 Minutes)
```bash
# Check FIX connection status
grep "FIX.*connection" logs/python/app.log | tail -20

# Look for:
# - Last successful heartbeat
# - Number of reconnection attempts
# - Error messages (auth failure vs. network)
```

**Decision Point:**
- **Network issue (transient):** Wait for auto-reconnect (Step 2)
- **Authentication failure:** Fix credentials (Step 3)
- **Extended outage:** Manual intervention (Step 4)

#### Step 2: Wait for Auto-Reconnect (5 Minutes)
Bot should automatically reconnect with exponential backoff:
- Attempt 1: 5 seconds
- Attempt 2: 10 seconds
- Attempt 3: 20 seconds
- Attempt 4: 40 seconds
- ...

```bash
# Monitor reconnection attempts
tail -f logs/python/app.log | grep "FIX.*connect"

# Should see:
# "FIX connection attempt 1..."
# "FIX connection attempt 2..."
# "FIX connection established ✓"
```

**If reconnected within 5 minutes:** No action needed, monitor Step 5

#### Step 3: Fix Authentication Issues
```bash
# Check credentials in config
cat config/ctrader_credentials.json

# Verify:
# - API key valid
# - Account ID correct
# - No typos in credentials

# Test connection manually
python scripts/test_fix_connection.py --config config/ctrader_credentials.json
```

#### Step 4: Manual Restart (If Auto-Reconnect Fails >5 Minutes)
```bash
# Graceful restart
./scripts/graceful_restart.sh

# If that fails, force restart
./scripts/emergency_shutdown.sh
./launch_micro_learning.sh
```

#### Step 5: Verify Position Synchronization Post-Reconnect
```bash
# Critical: Ensure bot and broker agree on positions
python scripts/verify_position_sync.py

# Check:
# ✓ Bot state matches broker positions
# ✓ No orphaned positions
# ✓ P&L calculations correct
```

**⚠️ WARNING:** If positions diverged during outage, manually reconcile before trading!

#### Step 6: Check for Missed Fills During Outage
```bash
# Query broker for all fills during outage
python scripts/query_missed_fills.py \
    --from "2026-01-11 14:00:00" \
    --to "2026-01-11 14:15:00"

# Reconcile with bot's trade history
python scripts/reconcile_trades.py
```

**If fills missed:** Update trade history and state files manually

---

## SCENARIO 5: PERFORMANCE DEGRADATION

### Symptoms
- Win rate dropped >10 percentage points
- Drawdown increasing
- Sharpe ratio declining
- Agent confidence low

### ACTIONS (Non-Emergency - Methodical Analysis)

#### Step 1: Collect Diagnostic Data
```bash
# Generate performance report
python scripts/generate_performance_report.py --last-7-days

# Outputs:
# - Win rate trend
# - Sharpe ratio evolution
# - Drawdown chart
# - Trade frequency
# - Agent confidence distribution
```

#### Step 2: Check for Parameter Staleness
```bash
# Run staleness detector
python scripts/check_parameter_staleness.py

# Look for:
# - Regime shift detected?
# - Parameters drifting?
# - Performance decay signal?
```

#### Step 3: Review Recent Trades
```bash
# Get last 100 trades
python scripts/export_trades.py --last 100 --format csv

# Analyze:
# - Are losses due to stop-outs or bad exits?
# - Is entry quality degrading?
# - Are circuit breakers triggering too often?
```

#### Step 4: Decision Tree

**If parameter staleness detected:**
```bash
# Reset parameters to defaults
python scripts/reset_parameters.py --confirm

# Restart in PAPER_TRADING mode to re-learn
./launch_micro_learning.sh --force-phase PAPER_TRADING
```

**If market regime changed:**
```bash
# Wait for regime detector to adapt (50-100 bars)
# Monitor but don't intervene immediately

# If degradation continues >200 bars:
# Reduce position size temporarily
python scripts/set_position_multiplier.py 0.5
```

**If logic bug suspected:**
```bash
# Review recent code changes
git log --oneline --since="1 week ago"

# Run regression tests
python -m pytest tests/

# If tests fail, revert to last stable version
git revert <commit>
```

**If external factors (news, volatility spike):**
```bash
# Temporarily halt trading during high uncertainty
./scripts/pause_trading.sh --resume-after 4h

# Bot will continue observation but not trade
```

---

## BACKUP & RESTORE PROCEDURES

### Automated Backups (Already Running)
```bash
# Backups run via cron:
# - Every 12 hours: Full state backup
# - Every 1 hour: Incremental journal backup
# - Daily: Trade history export

# Check backup status
./scripts/check_backup_status.sh

# Output:
# ✓ Last full backup: 2 hours ago
# ✓ Last incremental: 15 minutes ago
# ✓ Backup disk usage: 450 MB / 10 GB
# ✓ Oldest backup: 30 days ago
```

### Manual Backup (Before Risky Operations)
```bash
# Create snapshot before update
./scripts/create_backup_snapshot.sh "pre_update_$(date +%Y%m%d_%H%M%S)"

# Backup saved to: backups/state/pre_update_20260111_153045.tar.gz
```

### Restore from Backup
```bash
# List available backups
ls -lht backups/state/ | head -10

# Restore specific backup
./scripts/restore_backup.sh backups/state/state_backup_20260111_120000.tar.gz

# Verify restoration
python scripts/verify_state.py
```

---

## MONITORING & ALERTING

### Real-Time Monitoring Dashboard
```bash
# Start monitoring web interface
python production_monitor.py --port 8080

# Access at: http://localhost:8080
# Shows:
# - Current P&L
# - Open positions
# - Recent trades
# - Agent confidence
# - Circuit breaker status
# - System health
```

### Log Monitoring Commands
```bash
# Monitor errors in real-time
tail -f logs/python/app.log | grep ERROR

# Monitor trade executions
tail -f logs/python/app.log | grep "TRADE EXECUTED"

# Monitor circuit breakers
tail -f logs/python/app.log | grep "circuit_breaker"

# Monitor FIX connection
tail -f logs/python/app.log | grep "FIX"
```

### Performance Metrics Export
```bash
# Export metrics to CSV for analysis
python scripts/export_metrics.py \
    --start "2026-01-01" \
    --end "2026-01-11" \
    --output metrics_export.csv
```

---

## PREVENTIVE MAINTENANCE

### Daily Checks (5 Minutes)
```bash
# Run daily health check
./scripts/daily_health_check.sh

# Checks:
# ✓ Bot process running
# ✓ Disk space >10% free
# ✓ Log files not oversized
# ✓ Backup completed today
# ✓ No error spikes in logs
# ✓ FIX connection stable
```

### Weekly Reviews (30 Minutes)
```bash
# Generate weekly report
./scripts/weekly_review.sh

# Review:
# - Performance summary (P&L, Sharpe, drawdown)
# - Trade analysis (win rate, avg profit)
# - Circuit breaker activity
# - Error log summary
# - Backup verification
```

### Monthly Tasks
1. **Test disaster recovery:** Simulate crash and restore from backup
2. **Review & rotate logs:** Archive old logs to free space
3. **Update dependencies:** Check for security patches
4. **Performance audit:** Compare to baseline benchmarks
5. **Documentation update:** Update runbook with new learnings

---

## APPENDIX: SCRIPTS REFERENCE

| Script | Purpose | Usage |
|--------|---------|-------|
| `emergency_shutdown.sh` | Kill bot immediately | `./scripts/emergency_shutdown.sh` |
| `check_positions.sh` | Check current broker positions | `./scripts/check_positions.sh` |
| `emergency_close_all.sh` | Close all positions NOW | `./scripts/emergency_close_all.sh` |
| `restore_from_journal.py` | Replay journal to restore state | `python scripts/restore_from_journal.py` |
| `verify_state.py` | Validate state file integrity | `python scripts/verify_state.py` |
| `restore_backup.sh` | Restore from backup archive | `./scripts/restore_backup.sh <file>` |
| `replay_journal.py` | Replay journal from point | `python scripts/replay_journal.py` |
| `graceful_restart.sh` | Restart bot gracefully | `./scripts/graceful_restart.sh` |
| `daily_health_check.sh` | Run daily health checks | `./scripts/daily_health_check.sh` |

---

## CONTACT INFORMATION FOR ESCALATION

**L1 Support:** On-call engineer (responds within 15 minutes)  
**L2 Support:** Senior engineer (responds within 1 hour)  
**L3 Support:** System architect (responds within 4 hours)  

**Broker Support:** cTrader technical support hotline  
**Emergency After-Hours:** [Emergency contact number]

---

## REVISION HISTORY

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-01-11 | 1.0 | Initial runbook creation | System |

---

## IMPORTANT REMINDERS

1. **When in doubt, SHUTDOWN FIRST, investigate later**
2. **Always check positions before restarting**
3. **Never restart without understanding root cause**
4. **Document every incident for learning**
5. **Test disaster recovery procedures regularly**

**Emergency Mantra:**
> **STOP → ASSESS → PROTECT → RESTORE → VERIFY → RESUME**
