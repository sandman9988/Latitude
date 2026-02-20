# Log File Management & Cleanup Summary

## Summary

Reduced log file duplication and disk usage from **2.7GB to 0.9GB** (66% reduction).

## Changes Made

### 1. Consolidated Log Locations

**Before:**
- Root directory: `bot_console.log`, `hud_console.log`, `startup.log`
- logs/ directory: Same files duplicated
- Multiple logging directories: `logs/`, `log/`, `ctrader_py_logs/`

**After:**
- All operational logs → `logs/` directory
- Audit trails (immutable) → `log/` directory  
- Python application logs → `ctrader_py_logs/` directory

**Updated files:**
- [run.sh](../run.sh): Changed log paths from root to `logs/` directory
- `bot_console.log` → `logs/bot_console.log`
- `hud_console.log` → `logs/hud_console.log`
- `startup.log` → `logs/startup.log`

### 2. Created Cleanup Scripts

#### [`scripts/cleanup_old_logs.sh`](cleanup_old_logs.sh)
Removes old timestamped log files while keeping recent ones.

**Configuration:**
- Keep last 7 days OR latest 50 files (whichever keeps more)
- Cleans: ctrader_py_logs/, logs/, logs/python/, logs/archived/, logs/archive/
- Safe: Dry-run mode available with `--dry-run`

**Usage:**
```bash
# Preview what would be deleted
./scripts/cleanup_old_logs.sh --dry-run

# Actually delete old logs (default: 7 days, 50 files)
./scripts/cleanup_old_logs.sh

# Custom thresholds
./scripts/cleanup_old_logs.sh --days 3 --max-files 20
```

#### [`scripts/cleanup_logs_daily.sh`](cleanup_logs_daily.sh)
Automated daily cleanup (for cron/systemd timer).

**Features:**
- Runs cleanup script automatically
- Rotates large log files (>500MB)
- Truncates old audit logs (>30 days, >100MB)

**Setup automatic cleanup:**
```bash
# Add to crontab (runs daily at 2 AM)
crontab -e
# Add line:
0 2 * * * /home/renierdejager/Documents/ctrader_trading_bot/scripts/cleanup_logs_daily.sh
```

### 3. Removed Redundant Logging

Simplified initialization logging in 11+ files:
- `src/core/ctrader_ddqn_paper.py` - Consolidated bot initialization
- `src/agents/trigger_agent.py` - Removed duplicate "initialized" messages
- `src/agents/harvester_agent.py` - Consolidated agent setup logging
- `src/agents/dual_policy.py` - Reduced feature logging
- `src/monitoring/audit_logger.py` - Removed redundant logger init messages
- `src/monitoring/trade_audit_logger.py` - Removed redundant init
- `src/monitoring/activity_monitor.py` - Simplified init logging
- `src/monitoring/trade_exporter.py` - Removed verbose init
- `src/persistence/bot_persistence.py` - Removed redundant init
- `src/persistence/learned_parameters.py` - Consolidated param logging
- `src/persistence/atomic_persistence.py` - Simplified init

**Result:** ~60-70% reduction in initialization log noise.

### 4. Cleanup Results

**Initial cleanup run:**
```
Deleted 458 old log files from ctrader_py_logs/
  - 420 files older than 7 days
  - 38 excess files beyond 50 max
Kept: 50 most recent files
```

**Aggressive cleanup run:**
```
Deleted 249 old log files:
  - 168 from logs/archived/ (including 1.5GB hud_console.log!)
  - 33 from logs/python/
  - 18 from logs/archive/
  - 29 from ctrader_py_logs/
  - 1 old bot log
```

**Disk usage:**
```
Before cleanup:
  1.7GB   logs/
  668MB   log/
  326MB   ctrader_py_logs/
  Total: 2.7GB

After cleanup:
  31MB    logs/        (98% reduction!)
  669MB   log/         (audit trails - kept)
  192MB   ctrader_py_logs/  (41% reduction)
  Total: 0.9GB        (66% overall reduction)
```

## Log Directory Structure

```
logs/                       # Operational logs (cleaned regularly)
├── bot_console.log         # Main bot output (from run.sh)
├── hud_console.log         # HUD output
├── startup.log             # Startup script log
├── training_*.log          # Training session logs
├── fix/                    # FIX protocol logs
├── fix_quote/              # Quote session logs
├── fix_trade/              # Trade session logs
└── python/                 # Old Python logs (now cleaned)

log/                        # Audit trails (append-only, kept longer)
├── decisions.jsonl         # All agent decisions
├── trade_audit.jsonl       # All trade events
├── transactions.jsonl      # All system transactions
├── QUOTE/                  # FIX quote session logs
└── TRADE/                  # FIX trade session logs

ctrader_py_logs/            # Timestamped Python application logs
└── ctrader_YYYYMMDD_HHMMSS.log  # One per bot run (kept 50 most recent)
```

## Best Practices

1. **Run cleanup weekly:** `./scripts/cleanup_old_logs.sh`
2. **Monitor disk usage:** `du -sh logs/ log/ ctrader_py_logs/`
3. **Check large files:** `find . -name "*.log" -size +100M -ls`
4. **Set up automation:** Add cleanup script to crontab
5. **Keep audit trails:** Don't delete `log/*.jsonl` files (append-only audit)

## Maintenance

### Manual cleanup if needed:
```bash
# Remove all logs older than 3 days
./scripts/cleanup_old_logs.sh --days 3 --max-files 20

# Remove very old logs (1 month+)
find ctrader_py_logs/ -name "*.log" -mtime +30 -delete
find logs/python/ logs/archived/ logs/archive/ -name "*.log" -mtime +30 -delete
```

### Archive before deletion:
```bash
# Create compressed archive of old logs
tar -czf logs_archive_$(date +%Y%m%d).tar.gz \
    ctrader_py_logs/*.log logs/python/*.log logs/archived/*.log 2>/dev/null

# Then run cleanup
./scripts/cleanup_old_logs.sh
```

## Related Files

- [run.sh](../run.sh) - Updated log paths
- [cleanup_old_logs.sh](cleanup_old_logs.sh) - Main cleanup script
- [cleanup_logs_daily.sh](cleanup_logs_daily.sh) - Automated daily cleanup
- [MASTER_HANDBOOK.md](../MASTER_HANDBOOK.md) - Complete bot documentation
