#!/bin/bash
# watchdog.sh - External process monitor with auto-restart capability
# Continuously monitors bot health and restarts if necessary

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Configuration
CHECK_INTERVAL=30  # seconds between health checks
RESTART_DELAY=10   # seconds to wait before restart
MAX_RESTART_HOUR=5 # max restarts per hour
WATCHDOG_LOG="${SCRIPT_DIR}/watchdog.log"
RESTART_COUNT_FILE="${SCRIPT_DIR}/.restart_count"
LAST_RESTART_FILE="${SCRIPT_DIR}/.last_restart"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo -e "$msg" | tee -a "$WATCHDOG_LOG"
}

log_error() {
    log "${RED}[ERROR]${NC} $*"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $*"
}

log_info() {
    log "${BLUE}[INFO]${NC} $*"
}

log_success() {
    log "${GREEN}[OK]${NC} $*"
}

get_restart_count() {
    if [[ -f "$RESTART_COUNT_FILE" ]]; then
        cat "$RESTART_COUNT_FILE"
    else
        echo "0"
    fi
}

increment_restart_count() {
    local count=$(get_restart_count)
    count=$((count + 1))
    echo "$count" > "$RESTART_COUNT_FILE"
    date +%s > "$LAST_RESTART_FILE"
    echo "$count"
}

reset_restart_count() {
    echo "0" > "$RESTART_COUNT_FILE"
}

check_restart_limit() {
    local count=$(get_restart_count)
    
    # Reset counter if last restart was > 1 hour ago
    if [[ -f "$LAST_RESTART_FILE" ]]; then
        local last_restart=$(cat "$LAST_RESTART_FILE")
        local now=$(date +%s)
        local age=$((now - last_restart))
        
        if [[ $age -gt 3600 ]]; then
            log_info "Resetting restart counter (last restart was ${age}s ago)"
            reset_restart_count
            count=0
        fi
    fi
    
    if [[ $count -ge $MAX_RESTART_HOUR ]]; then
        log_error "Restart limit reached ($count/$MAX_RESTART_HOUR per hour)"
        log_error "Manual intervention required. Watchdog stopping."
        return 1
    fi
    
    return 0
}

restart_bot() {
    log_warn "Initiating bot restart..."
    
    # Check restart limit
    if ! check_restart_limit; then
        return 1
    fi
    
    # Kill existing process
    log_info "Stopping existing bot process..."
    pkill -f "ctrader_ddqn_paper" || true
    sleep 2
    
    # Force kill if still running
    if pgrep -f "ctrader_ddqn_paper" > /dev/null; then
        log_warn "Process still running, force killing..."
        pkill -9 -f "ctrader_ddqn_paper" || true
        sleep 2
    fi
    
    # Clean up stale PID file
    rm -f "${PROJECT_ROOT}/.bot.pid"
    
    # Wait before restart
    log_info "Waiting ${RESTART_DELAY}s before restart..."
    sleep "$RESTART_DELAY"
    
    # Start bot
    log_info "Starting bot with run.sh..."
    if bash "${PROJECT_ROOT}/run.sh" --no-hud >> "$WATCHDOG_LOG" 2>&1; then
        local new_count=$(increment_restart_count)
        log_success "Bot restarted successfully (restart #${new_count} this hour)"
        return 0
    else
        log_error "Failed to restart bot via run.sh"
        return 1
    fi
}

# Main monitoring loop
log_info "=========================================="
log_info "Watchdog started (PID: $$)"
log_info "Check interval: ${CHECK_INTERVAL}s"
log_info "Max restarts/hour: ${MAX_RESTART_HOUR}"
log_info "=========================================="

CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE=3  # restart after 3 consecutive failures

while true; do
    sleep "$CHECK_INTERVAL"
    
    # Run health check
    if bash "${SCRIPT_DIR}/health_check.sh" > /dev/null 2>&1; then
        # Health check passed
        if [[ $CONSECUTIVE_FAILURES -gt 0 ]]; then
            log_success "Health check passed (recovered from $CONSECUTIVE_FAILURES failures)"
        fi
        CONSECUTIVE_FAILURES=0
    else
        EXIT_CODE=$?
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        
        if [[ $EXIT_CODE -eq 2 ]]; then
            # CRITICAL status
            log_error "Health check CRITICAL (consecutive failures: $CONSECUTIVE_FAILURES/$MAX_CONSECUTIVE)"
            
            if [[ $CONSECUTIVE_FAILURES -ge $MAX_CONSECUTIVE ]]; then
                log_warn "Max consecutive failures reached, triggering restart..."
                if restart_bot; then
                    CONSECUTIVE_FAILURES=0
                else
                    log_error "Restart failed, will retry on next check"
                fi
            fi
        elif [[ $EXIT_CODE -eq 1 ]]; then
            # WARNING status
            log_warn "Health check WARNING (consecutive: $CONSECUTIVE_FAILURES)"
        else
            # UNKNOWN
            log_warn "Health check UNKNOWN status (exit code: $EXIT_CODE)"
        fi
    fi
    
    # Log heartbeat every 10 checks (~5 minutes with 30s interval)
    if [[ $(($(date +%s) % 300)) -lt 30 ]]; then
        local restart_count=$(get_restart_count)
        log_info "Watchdog heartbeat - restarts this hour: $restart_count/$MAX_RESTART_HOUR"
    fi
done
