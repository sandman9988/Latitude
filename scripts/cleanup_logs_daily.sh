#!/bin/bash
# Daily log cleanup - runs automatically to keep disk usage low
# Add to crontab: 0 2 * * * /path/to/cleanup_logs_daily.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Run cleanup script: keep 7 days or 50 files
bash scripts/cleanup_old_logs.sh --days 7 --max-files 50 >> logs/log_cleanup.log 2>&1

# Also clean up very old audit logs (older than 30 days)
find log/ -name "*.jsonl" -size +100M -mtime +30 -exec truncate -s 0 {} \; 2>/dev/null

# Rotate large current logs if > 500MB
for log in logs/bot_console.log logs/hud_output.log; do
    if [[ -f "$log" ]] && [[ $(stat -f%z "$log" 2>/dev/null || stat -c%s "$log" 2>/dev/null || echo "0") -gt 524288000 ]]; then
        timestamp=$(date +%Y%m%d_%H%M%S)
        mv "$log" "${log%.log}_${timestamp}.log.old"
        touch "$log"
    fi
done
