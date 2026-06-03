#!/bin/bash
# Market Monitor - Checks if markets are open and starts bot automatically
# Run every 30 minutes starting from 22:00 local time

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

LOG_FILE="$SCRIPT_DIR/logs/market_monitor.log"
PID_FILE="$SCRIPT_DIR/data/bot.pid"
LOCK_FILE="/tmp/ctrader_bot.lock"

# Ensure logs directory exists
mkdir -p "$SCRIPT_DIR/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

is_bot_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # Bot is running
        else
            # Stale PID file
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    
    # Check for process by name
    if pgrep -f "ctrader_ddqn_paper.py" > /dev/null 2>&1; then
        return 0
    fi
    
    return 1
}

is_market_hours() {
    # Forex markets are open:
    # Sunday 22:00 UTC - Friday 22:00 UTC
    # Gold (XAUUSD) follows forex hours
    
    local day_of_week=$(date +%u)  # 1=Monday, 7=Sunday
    local hour=$(date +%H)
    local minute=$(date +%M)
    local time_decimal=$((10#$hour * 100 + 10#$minute))
    
    # Friday after 22:00 - markets closed
    if [[ $day_of_week -eq 5 ]] && [[ $time_decimal -ge 2200 ]]; then
        return 1
    fi
    
    # Saturday - markets closed
    if [[ $day_of_week -eq 6 ]]; then
        return 1
    fi
    
    # Sunday before 22:00 - markets closed
    if [[ $day_of_week -eq 7 ]] && [[ $time_decimal -lt 2200 ]]; then
        return 1
    fi
    
    # Monday-Thursday or Sunday after 22:00 or Friday before 22:00 - markets open
    return 0
}

cleanup_orphaned_huds() {
    # Check for orphaned HUD processes (older than 1 day)
    local hud_count=$(pgrep -f hud_tabbed | wc -l)
    if [[ $hud_count -gt 3 ]]; then
        log "⚠️  Detected $hud_count HUD processes - cleaning up orphans"
        # Kill HUDs older than 1 day
        ps aux | grep hud_tabbed | grep -v grep | awk '$9 !~ /^[0-9]{2}:[0-9]{2}$/ {print $2}' | xargs -r kill 2>/dev/null
        log "✓ Orphaned HUD processes cleaned up"
    fi
}

start_bot() {
    log "🚀 Starting trading bot..."
    
    # Check if virtual environment exists
    if [[ ! -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
        log "❌ Virtual environment not found at $SCRIPT_DIR/.venv"
        return 1
    fi
    
    # Clean up orphaned HUD processes before starting
    cleanup_orphaned_huds
    
    # Use run.sh to start bot with HUD
    if [[ -f "$SCRIPT_DIR/run.sh" ]]; then
        nohup "$SCRIPT_DIR/run.sh" --with-hud >> "$SCRIPT_DIR/logs/bot_output.log" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        log "✓ Bot started (PID: $pid)"
        return 0
    else
        log "❌ run.sh not found"
        return 1
    fi
}

main() {
    log "=== Market Monitor Check ==="
    
    # Check if another instance is running (prevent duplicate checks)
    if [[ -f "$LOCK_FILE" ]]; then
        local lock_age=$(($(date +%s) - $(stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0)))
        if [[ $lock_age -lt 300 ]]; then
            log "⏭️  Another instance running (lock age: ${lock_age}s), skipping"
            exit 0
        else
            log "⚠️  Removing stale lock file (age: ${lock_age}s)"
            rm -f "$LOCK_FILE"
        fi
    fi
    
    # Create lock file
    touch "$LOCK_FILE"
    trap 'rm -f "$LOCK_FILE"' EXIT
    
    # Regular HUD cleanup (keep max 2 instances)
    local hud_count=$(pgrep -f hud_tabbed | wc -l)
    if [[ $hud_count -gt 2 ]]; then
        log "⚠️  Found $hud_count HUD instances - cleaning up"
        cleanup_orphaned_huds
    fi
    
    # Check market status
    if is_market_hours; then
        log "📈 Markets are OPEN"
        
        if is_bot_running; then
            log "✓ Bot is already running"
        else
            log "⚠️  Bot not running - attempting to start"
            if start_bot; then
                log "✓ Bot successfully started"
            else
                log "❌ Failed to start bot"
                exit 1
            fi
        fi
    else
        local day=$(date +%A)
        local time=$(date +%H:%M)
        log "📉 Markets are CLOSED ($day $time)"
        
        if is_bot_running; then
            log "ℹ️  Bot running during closed hours (training/catching up)"
        else
            log "ℹ️  Bot not running (expected during market close)"
        fi
    fi
    
    log "=== Check Complete ==="
}

main "$@"
