#!/usr/bin/env bash
# Install/update the cron entry for guarded weekend offline retraining.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/weekend_offline_training.sh"

CRON_TZ_VALUE="${WEEKEND_TRAIN_CRON_TZ:-Africa/Johannesburg}"
CRON_SCHEDULE="${WEEKEND_TRAIN_CRON_SCHEDULE:-0 3 * * 6}"
MARKER_BEGIN="# cTrader weekend offline training BEGIN"
MARKER_END="# cTrader weekend offline training END"

chmod +x "$TRAIN_SCRIPT"
mkdir -p "$PROJECT_ROOT/logs"

tmp_current="$(mktemp)"
tmp_next="$(mktemp)"
trap 'rm -f "$tmp_current" "$tmp_next"' EXIT

crontab -l > "$tmp_current" 2>/dev/null || true

awk -v begin="$MARKER_BEGIN" -v end="$MARKER_END" '
    $0 == begin {skip=1; next}
    $0 == end {skip=0; next}
    !skip {print}
' "$tmp_current" > "$tmp_next"

{
    printf '%s\n' "$MARKER_BEGIN"
    printf 'CRON_TZ=%s\n' "$CRON_TZ_VALUE"
    printf '%s cd %q && %q >> %q 2>&1\n' \
        "$CRON_SCHEDULE" \
        "$PROJECT_ROOT" \
        "$TRAIN_SCRIPT" \
        "$PROJECT_ROOT/logs/weekend_offline_training.cron.log"
    printf '%s\n' "$MARKER_END"
} >> "$tmp_next"

crontab "$tmp_next"

echo "Installed guarded weekend offline training cron entry:"
echo "  Time zone : $CRON_TZ_VALUE"
echo "  Schedule  : $CRON_SCHEDULE"
echo "  Script    : $TRAIN_SCRIPT"
echo "  Log       : $PROJECT_ROOT/logs/weekend_offline_training.log"
echo ""
echo "Override examples:"
echo "  WEEKEND_TRAIN_CRON_SCHEDULE='0 3 * * 0' ./run.sh weekend-train-setup"
echo "  WEEKEND_TRAIN_EPOCHS=3 WEEKEND_TRAIN_WORKERS=4 ./run.sh weekend-train"
