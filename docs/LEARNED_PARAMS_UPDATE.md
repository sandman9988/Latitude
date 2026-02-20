# Learned Parameters Update Guide

## CRC32 Validation System

The bot uses CRC32 checksums to detect corrupted learned parameters files. Any manual edit will invalidate the checksum and trigger an automatic restore from backup.

## Safe Update Tool

Use the provided script to update parameters safely:

```bash
# List all parameters
python3 scripts/update_learned_params.py --list

# Update a single parameter (main file only)
python3 scripts/update_learned_params.py \
  --param harvester_profit_target_pct \
  --value 0.45

# Update parameter in ALL files (including backups)
python3 scripts/update_learned_params.py \
  --param harvester_min_soft_profit_pct \
  --value 0.40 \
  --all-files
```

## Key Parameters

### Exit Strategy (Harvester)
- `harvester_profit_target_pct`: Profit target threshold (default: 0.45%)
- `harvester_min_soft_profit_pct`: Breakeven trigger (default: 0.40%)
- `harvester_stop_loss_pct`: Stop loss threshold (default: 0.40%)
- `harvester_soft_time_bars`: Soft time stop (default: 200 bars)
- `harvester_hard_time_bars`: Hard time stop (default: 400 bars)

### Risk Management
- `base_position_size`: Position size in lots (default: 0.1)
- `max_drawdown_pct`: Max drawdown limit (default: 15%)
- `var_multiplier`: VaR position sizing multiplier (default: 1.0)
- `risk_budget_usd`: Daily risk budget (default: $100)

### Confidence Thresholds
- `entry_confidence_threshold`: Min confidence to enter (default: 0.6)
- `exit_confidence_threshold`: Min confidence to exit (default: 0.5)

## Manual CRC32 Calculation

If you need to manually edit the file:

```python
import json, zlib
from datetime import datetime, timezone

# Load file
with open('data/learned_parameters.json', 'r') as f:
    data = json.load(f)

# Make your changes
data['data']['instruments']['XAUUSD_M5_default']['params']['harvester_profit_target_pct']['value'] = 0.45

# Update timestamp
data['timestamp'] = datetime.now(timezone.utc).isoformat()

# Recalculate CRC32
data_str = json.dumps(data['data'], sort_keys=True)
data['crc32'] = zlib.crc32(data_str.encode()) & 0xffffffff

# Save
with open('data/learned_parameters.json', 'w') as f:
    json.dump(data, f, indent=2)
```

## Backup System

The bot maintains automatic backups:
- `learned_parameters.json` - Main file
- `learned_parameters.json.YYYYMMDD_HHMMSS.bak` - Timestamped backups
- `learned_parameters.json.backup_YYYYMMDD_HHMMSS` - Long-term backups

When CRC32 mismatch is detected, the bot automatically restores from the most recent backup.

## Important Notes

⚠️ **Always update ALL files** when changing critical parameters:
- Main file gets auto-restored from backups if CRC32 is invalid
- Must update backups too, or changes will be lost on next restart
- Use `--all-files` flag to update everything at once

✓ **The bot will overwrite learned_parameters.json** during normal operation as it learns from trades
- Manual edits should be made when bot is stopped
- Or use the provided script which handles checksums correctly
