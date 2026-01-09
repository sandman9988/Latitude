# Known Issues

## Current Version: 0.1.0

### Active Issues

1. **Multiple bot instances running**
   - PIDs: 36885, 37808
   - Started: 07:31-07:32 UTC
   - Impact: May cause duplicate orders
   - Fix: Kill duplicate before next run
   - Command: `pkill -f ctrader_ddqn_paper.py`

2. **Environment variables in run.sh**
   - run.sh doesn't export variables correctly
   - Error code 127 when running
   - Workaround: Export manually before running
   - Fix planned: Update run.sh in next patch

### Resolved Issues

None yet.

### Pending Validation

1. **First M15 bar close**
   - Bot has been collecting data since 07:31
   - Next bar closes at 07:45 (or :00, :15, :30, :45)
   - Need to verify trading logic triggers

2. **Position reporting**
   - Position request sent at 07:31:07
   - No response logged yet
   - May be normal (no positions)

---

## How to Report Issues

1. Check logs: `tail -100 logs/python/ctrader_*.log`
2. Note the error message
3. Note the timestamp
4. Add to this file with:
   - Description
   - How to reproduce
   - Workaround (if known)
   - Priority (Low/Medium/High/Critical)
