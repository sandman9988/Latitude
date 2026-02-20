#!/usr/bin/env bash
# rotate_logs.sh — User-space log rotation for cTrader bot
# Called on startup (via run.sh setup_logging) and hourly by cron.
# Uses logrotate with a project-local state file (no root required).
#
# Usage:  scripts/rotate_logs.sh [--force]
#   --force  Force rotation regardless of size (useful after manual inspection)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONF="$SCRIPT_DIR/logrotate.conf"
STATE="$PROJECT_DIR/data/logrotate.state"
FORCE="${1:-}"

# Ensure state dir exists
mkdir -p "$(dirname "$STATE")"

if ! command -v logrotate &>/dev/null; then
    echo "[rotate_logs] WARNING: logrotate not installed, skipping rotation" >&2
    exit 0
fi

if [[ "$FORCE" == "--force" ]]; then
    logrotate --force --state "$STATE" "$CONF"
else
    logrotate --state "$STATE" "$CONF"
fi

# Prune one-off timestamped startup logs older than 7 days
find "$PROJECT_DIR/logs" -maxdepth 1 -name "startup_*.log" -mtime +7 -delete 2>/dev/null || true

# Prune ctrader_py_logs by filename date — keep last 2 days
# Files are named ctrader_YYYYMMDD_HHMMSS.log; derive cutoff from system date
KEEP_DAYS=2
cutoff_date=$(date -d "$KEEP_DAYS days ago" '+%Y%m%d')
find "$PROJECT_DIR/ctrader_py_logs" -maxdepth 1 -name "ctrader_????????_*.log" | while read -r f; do
    basename=$(basename "$f")
    file_date="${basename:8:8}"   # extract YYYYMMDD
    if [[ "$file_date" < "$cutoff_date" ]]; then
        rm -f "$f"
    fi
done

echo "[rotate_logs] Done ($(date '+%Y-%m-%d %H:%M:%S'))"
