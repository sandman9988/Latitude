# Disaster Recovery Runbook

> ⚠️ **LEGACY VERSION** — The canonical, more comprehensive runbook is at  
> **[operations/DISASTER_RECOVERY_RUNBOOK.md](operations/DISASTER_RECOVERY_RUNBOOK.md)**  
> This file is kept for backward compatibility only.

---

**CRITICAL**: This document contains step-by-step procedures for recovering from catastrophic failures during production trading. Keep printed copy accessible.

**Emergency Contact**: [YOUR_CONTACT_INFO]  
**Broker Support**: [BROKER_PHONE_NUMBER]  
**Last Updated**: 2026-01-11

---

## Table of Contents

1. [Scenario 1: Bot Crashes Mid-Trade](#scenario-1-bot-crashes-mid-trade)
2. [Scenario 2: Corrupt State Files](#scenario-2-corrupt-state-files)
3. [Scenario 3: Runaway Trading](#scenario-3-runaway-trading)
4. [Scenario 4: Network/FIX Connection Outage](#scenario-4-networkfix-connection-outage)
5. [Scenario 5: Broker Rejection / Account Issues](#scenario-5-broker-rejection--account-issues)
6. [Scenario 6: Data Loss / Journal Corruption](#scenario-6-data-loss--journal-corruption)
7. [General Recovery Principles](#general-recovery-principles)
8. [Post-Incident Review](#post-incident-review)

---

## Scenario 1: Bot Crashes Mid-Trade

**Symptoms**: Process terminated unexpectedly, terminal shows crash/exception, bot not responding.

### Immediate Actions (< 5 minutes)

1. **Check Open Positions**
   ```bash
   # Login to broker web interface
   # Navigate to Positions → Open Positions
   # Record: Symbol, Direction, Quantity, Entry Price, Unrealized P&L
   ```

2. **Manual Position Management**
   - **If position is profitable and near target**: Close manually via broker interface
   - **If position is at loss near stop**: Let stop execute, or close manually if stop not set
   - **If position is neutral**: Leave open and proceed to restart

3. **Assess Crash Cause**
   ```bash
   # Check last 100 lines of log
   tail -n 100 ctrader_py_logs/ctrader_ddqn_paper_*.log
   
   # Check for common causes:
   # - "MemoryError" → Insufficient RAM
   # - "FIX connection" → Network issue
   # - "Assertion failed" → Logic bug
   # - "Killed" → OOM killer
   ```

### Recovery Procedure

4. **Replay Journal (if journal exists)**
   ```python
   from journaled_persistence import Journal
   
   journal = Journal(journal_file="data/store/state.journal")
   checkpoint_path = journal.get_latest_checkpoint()
   
   if checkpoint_path:
       # Replay from checkpoint
       operations = journal.replay_from_checkpoint()
       print(f"Replayed {len(operations)} operations")
   else:
       print("No checkpoint found, manual recovery required")
   ```

5. **Restart Bot**
   ```bash
   # Start bot with --recovery flag (if implemented)
   ./run.sh --recovery
   
   # OR standard restart
   ./run.sh
   ```

6. **Verify State Recovery**
   ```bash
   # Check current_position.json
   cat data/current_position.json
   
   # Verify matches broker position
   # If mismatch → STOP and reconcile manually
   ```

### If Recovery Fails

- **Do NOT restart bot blindly**
- Close all positions manually via broker
- Restore from backup state files:
  ```bash
  cd data/
  ls -lt *.json | head -5  # Find most recent backups
  # Manually restore last known good state
  ```

---

## Scenario 2: Corrupt State Files

**Symptoms**: Bot fails to load state, JSON parsing errors, `current_position.json` has invalid data.

### Immediate Actions

1. **Backup Corrupt Files**
   ```bash
   mkdir -p data/corrupt_$(date +%Y%m%d_%H%M%S)
   cp data/*.json data/corrupt_$(date +%Y%m%d_%H%M%S)/
   ```

2. **Attempt Automated Recovery**
   ```python
   import json
   from pathlib import Path
   
   # Try to salvage what we can
   for file in Path("data").glob("*.json"):
       try:
           with open(file) as f:
               data = json.load(f)
           print(f"✓ {file.name} is valid")
       except Exception as e:
           print(f"✗ {file.name} corrupt: {e}")
   ```

3. **Rebuild from Journal**
   ```python
   from journaled_persistence import Journal
   
   # Replay full journal to reconstruct state
   journal = Journal()
   operations = journal.replay_from_checkpoint()
   
   # Manually rebuild state files from operations
   # [See journal format in journaled_persistence.py]
   ```

4. **Manual State Reconstruction**
   ```bash
   # Check broker for ground truth
   # Manually create minimal valid state:
   
   echo '{"symbol": "", "direction": null, "entry_price": 0}' > data/current_position.json
   echo '{"phase": "OBSERVATION"}' > data/cold_start_manager.json
   echo '{"total_pnl": 0.0, "trades": 0}' > data/performance_snapshot.json
   ```

5. **Restart in Safe Mode**
   - Set `cold_start_manager` to OBSERVATION phase (no trading)
   - Let bot run for 100 bars to rebuild internal state
   - Manually graduate to PAPER_TRADING once stable

---

## Scenario 3: Runaway Trading

**Symptoms**: Excessive trade frequency (>10 trades/hour), rapid position flipping, unexplained losses.

### CRITICAL: STOP TRADING IMMEDIATELY

1. **Emergency Stop**
   ```bash
   # Option 1: Kill bot process
   pkill -9 python
   
   # Option 2: Create emergency stop file (if implemented)
   touch data/EMERGENCY_STOP
   ```

2. **Close All Positions**
   - Login to broker web interface
   - **Close all open positions manually**
   - DO NOT restart bot until root cause identified

3. **Capture Diagnostic Data**
   ```bash
   # Copy ALL state files
   mkdir -p incident_$(date +%Y%m%d_%H%M%S)
   cp -r data/ incident_$(date +%Y%m%d_%H%M%S)/
   cp -r ctrader_py_logs/ incident_$(date +%Y%m%d_%H%M%S)/
   ```

### Root Cause Analysis

4. **Check Circuit Breakers**
   ```bash
   # Verify circuit breakers were active
   grep "circuit_breaker" ctrader_py_logs/*.log | tail -50
   
   # Check for disabled safety systems
   grep -i "bypass\|disable\|override" *.py
   ```

5. **Check Reward Function**
   ```bash
   # Run reward integrity monitor
   python -c "
   from reward_integrity_monitor import RewardIntegrityMonitor
   monitor = RewardIntegrityMonitor()
   # Load recent trades and check for gaming
   "
   ```

6. **Check Feedback Loops**
   ```python
   from feedback_loop_breaker import FeedbackLoopBreaker
   
   # Check if loop breaker detected issues
   breaker = FeedbackLoopBreaker()
   breaker.load_state()
   print(f"Recent interventions: {breaker.interventions[-10:]}")
   ```

### Recovery

- **DO NOT resume trading until**:
  - Root cause identified and fixed
  - Circuit breakers verified functional
  - Reward calculations validated
  - Manual review of last 100 agent decisions
  
- **Restart in OBSERVATION mode**
- **Monitor for 24 hours before enabling trading**

---

## Scenario 4: Network/FIX Connection Outage

**Symptoms**: `FIX connection lost`, heartbeat timeouts, orders not executing.

### Immediate Actions

1. **Check Positions at Risk**
   ```bash
   # If connection lost, you cannot manage positions via bot
   # MUST use broker web interface or mobile app
   
   # Check:
   # - Are stop-losses set? (broker-side stops persist)
   # - Is position near liquidation?
   # - Is market moving against position?
   ```

2. **Assess Network**
   ```bash
   # Check internet connectivity
   ping 8.8.8.8 -c 5
   
   # Check broker FIX endpoint
   ping fix.youro broker.com -c 5
   
   # Check DNS
   nslookup fix.yourbroker.com
   ```

3. **Check FIX Logs**
   ```bash
   # Look for disconnect reason
   grep -i "disconnect\|timeout\|reject" ctrader_py_logs/FIX*.log | tail -20
   
   # Common causes:
   # - "Heartbeat timeout" → Network hiccup
   # - "Session rejected" → Authentication issue
   # - "Sequence number mismatch" → Need to reset sequence
   ```

### Recovery

4. **FIX Session Reset**
   ```bash
   # If sequence number mismatch:
   # Delete FIX session state (forces clean reconnect)
   rm data/store/FIX.4.4-*.body
   rm data/store/FIX.4.4-*.header
   rm data/store/FIX.4.4-*.seqnums
   ```

5. **Restart with Connection Monitoring**
   ```bash
   # Restart bot
   ./run.sh
   
   # In separate terminal, monitor FIX connection
   tail -f ctrader_py_logs/FIX*.log | grep -i "logon\|logout"
   ```

6. **Manual Reconnection**
   ```python
   # If bot fails to reconnect automatically:
   # Use broker API directly (if available)
   # OR close positions via web interface
   # OR call broker support for manual order entry
   ```

---

## Scenario 5: Broker Rejection / Account Issues

**Symptoms**: Orders rejected, "Insufficient margin", "Account suspended", "Trading not allowed".

### Immediate Actions

1. **Check Account Status**
   - Login to broker web portal
   - Navigate to Account → Status
   - Check for:
     - Insufficient margin
     - Account restrictions
     - Trading hours (market closed?)
     - Symbol trading restrictions

2. **Check Order Rejection Logs**
   ```bash
   grep -i "reject\|denied\|insufficient" ctrader_py_logs/*.log | tail -20
   ```

### Common Causes & Fixes

| Rejection Reason | Fix |
|-----------------|-----|
| Insufficient margin | Close positions OR deposit funds |
| Market closed | Wait for market open (check trading hours) |
| Symbol restricted | Check broker symbol permissions |
| Position limit exceeded | Close existing positions |
| Order size too small/large | Check min/max order size for symbol |
| Invalid price | Check tick size requirements |

### Recovery

- **If margin call**: Close losing positions immediately
- **If account suspended**: Contact broker support
- **If temporary restriction**: Wait and retry (bot will retry automatically)

---

## Scenario 6: Data Loss / Journal Corruption

**Symptoms**: Journal file missing, replay fails, `FileNotFoundError` on startup.

### Recovery from Checkpoint

1. **Find Latest Checkpoint**
   ```bash
   ls -lt data/store/*.checkpoint | head -5
   ```

2. **Manual State Reconstruction**
   ```bash
   # Use broker as source of truth
   # Manually create state files matching broker positions
   
   # Get positions from broker API or web interface
   # Create minimal valid state
   ```

3. **Disable Journal Temporarily**
   ```python
   # Edit ctrader_ddqn_paper.py to bypass journal
   # Comment out journal.log_*() calls
   # Restart in OBSERVATION mode
   ```

4. **Rebuild Journal from Logs**
   ```bash
   # Extract trade history from logs
   grep "TRADE_OPEN\|TRADE_CLOSE" ctrader_py_logs/*.log > reconstructed_trades.txt
   
   # Manually replay into new journal
   ```

---

## General Recovery Principles

### Priority Order

1. **Protect capital** (close risky positions)
2. **Stop trading** (prevent further damage)
3. **Gather diagnostics** (logs, state files)
4. **Identify root cause** (don't guess)
5. **Fix and verify** (test before resuming)
6. **Resume cautiously** (start in OBSERVATION mode)

### Data Integrity Checks

Before restarting bot after ANY incident:

```bash
# 1. Verify JSON files are valid
for f in data/*.json; do python -m json.tool "$f" > /dev/null && echo "✓ $f" || echo "✗ $f CORRUPT"; done

# 2. Verify broker position matches bot state
# [Manual check via web interface]

# 3. Verify journal integrity
python -c "from journaled_persistence import Journal; j=Journal(); j.replay_from_checkpoint(); print('✓ Journal OK')"

# 4. Verify no EMERGENCY_STOP flag
test ! -f data/EMERGENCY_STOP && echo "✓ No emergency stop" || echo "⚠️  Emergency stop active"

# 5. Check circuit breakers reset
grep "circuit_breaker" data/*.json
```

### Safe Restart Checklist

- [ ] All positions closed (or accounted for)
- [ ] Root cause identified
- [ ] Fix applied and tested
- [ ] State files validated
- [ ] Broker account healthy (margin, connectivity)
- [ ] Circuit breakers enabled
- [ ] Starting in OBSERVATION or PAPER mode
- [ ] Monitoring dashboard active
- [ ] Emergency stop procedure rehearsed

---

## Post-Incident Review

After EVERY incident, document:

1. **Timeline**
   - When did incident start?
   - When was it detected?
   - When was it resolved?

2. **Root Cause**
   - What failed?
   - Why did it fail?
   - What was the trigger?

3. **Impact**
   - Financial loss
   - Downtime
   - Data loss

4. **Response Effectiveness**
   - What went well?
   - What went poorly?
   - What was confusing?

5. **Prevention**
   - Code changes needed
   - Monitoring improvements
   - Documentation updates
   - Process changes

**Template**: Create `incident_reports/YYYY-MM-DD_<description>.md`

---

## Emergency Contacts

| Role | Contact | Phone | Email |
|------|---------|-------|-------|
| Primary Operator | [YOUR_NAME] | [PHONE] | [EMAIL] |
| Backup Operator | [NAME] | [PHONE] | [EMAIL] |
| Broker Support | [BROKER] | [PHONE] | [EMAIL] |
| Technical Support | [NAME] | [PHONE] | [EMAIL] |

---

## Quick Reference Commands

```bash
# EMERGENCY STOP
pkill -9 python
touch data/EMERGENCY_STOP

# CHECK POSITIONS (broker web interface)
# [BROKER_URL]/positions

# CHECK LOGS
tail -100 ctrader_py_logs/ctrader_ddqn_paper_*.log

# CHECK STATE
cat data/current_position.json
cat data/cold_start_manager.json

# REPLAY JOURNAL
python -c "from journaled_persistence import Journal; Journal().replay_from_checkpoint()"

# SAFE RESTART
./run.sh --mode=observation

# MONITOR METRICS
curl http://localhost:8765/metrics | python -m json.tool
```

---

**Last Updated**: 2026-01-11  
**Review Frequency**: Quarterly or after each incident  
**Next Review Date**: 2026-04-11

---

## Appendix: Known Issues & Workarounds

### Issue 1: Journal Replay Slow
- **Symptom**: Replay takes >5 minutes
- **Cause**: Large journal file (>100MB)
- **Workaround**: Delete old checkpoints, force new checkpoint

### Issue 2: FIX Sequence Number Mismatch
- **Symptom**: "MsgSeqNum too low" rejection
- **Cause**: Bot crashed during session
- **Workaround**: Delete FIX session files, force clean reconnect

### Issue 3: Memory Leak
- **Symptom**: Bot crashes after 24hrs with "MemoryError"
- **Cause**: Experience buffer unbounded growth
- **Workaround**: Restart daily until fix deployed

---
