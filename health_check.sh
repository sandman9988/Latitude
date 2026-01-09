#!/bin/bash
# health_check.sh - Production health monitoring for cTrader FIX bot
# Checks connection status, market data flow, and critical errors

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Load environment
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Configuration
LOG_DIR="${PY_LOGDIR:-ctrader_py_logs}"
MAX_LOG_AGE=300  # 5 minutes
MAX_DATA_STALE=120  # 2 minutes without market data = problem
PYTHON_PID_FILE="${SCRIPT_DIR}/.bot.pid"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Exit codes
EXIT_OK=0
EXIT_WARNING=1
EXIT_CRITICAL=2
EXIT_UNKNOWN=3

# Global status
CRITICAL_COUNT=0
WARNING_COUNT=0

check_process() {
    echo -e "${BLUE}[CHECK]${NC} Verifying bot process..."
    
    if [[ -f "$PYTHON_PID_FILE" ]]; then
        PID=$(cat "$PYTHON_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} Bot running (PID: $PID)"
            return 0
        else
            echo -e "${RED}✗${NC} PID file exists but process not found (stale PID: $PID)"
            CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
            return 1
        fi
    else
        # Fallback: check by process name
        PID=$(pgrep -f "ctrader_ddqn_paper.py" | head -1)
        if [[ -n "$PID" ]]; then
            echo -e "${YELLOW}⚠${NC} Bot running but no PID file (PID: $PID)"
            WARNING_COUNT=$((WARNING_COUNT + 1))
            echo "$PID" > "$PYTHON_PID_FILE"
            return 0
        else
            echo -e "${RED}✗${NC} Bot process not running"
            CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
            return 1
        fi
    fi
}

check_log_files() {
    echo -e "${BLUE}[CHECK]${NC} Checking log files..."
    
    if [[ ! -d "$LOG_DIR" ]]; then
        echo -e "${RED}✗${NC} Log directory not found: $LOG_DIR"
        CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
        return 1
    fi
    
    # Find most recent log file
    LATEST_LOG=$(find "$LOG_DIR" -name "ctrader_*.log" -type f -printf '%T+ %p\n' | sort -r | head -1 | cut -d' ' -f2-)
    
    if [[ -z "$LATEST_LOG" ]]; then
        echo -e "${RED}✗${NC} No log files found in $LOG_DIR"
        CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
        return 1
    fi
    
    # Check log age
    LOG_AGE=$(( $(date +%s) - $(stat -c %Y "$LATEST_LOG") ))
    
    if [[ $LOG_AGE -gt $MAX_LOG_AGE ]]; then
        echo -e "${YELLOW}⚠${NC} Latest log is stale (${LOG_AGE}s old): $LATEST_LOG"
        WARNING_COUNT=$((WARNING_COUNT + 1))
    else
        echo -e "${GREEN}✓${NC} Log file active (${LOG_AGE}s old): $LATEST_LOG"
    fi
    
    echo "$LATEST_LOG"
}

check_fix_connections() {
    local log_file="$1"
    echo -e "${BLUE}[CHECK]${NC} Verifying FIX connections..."
    
    # Check for recent LOGONs (last 10 minutes)
    RECENT_LOGON=$(tail -10000 "$log_file" | grep -c "\[LOGON\]" || echo "0")
    
    if [[ $RECENT_LOGON -ge 2 ]]; then
        echo -e "${GREEN}✓${NC} QUOTE and TRADE sessions logged in"
    else
        echo -e "${YELLOW}⚠${NC} Missing LOGON messages (found: $RECENT_LOGON)"
        WARNING_COUNT=$((WARNING_COUNT + 1))
    fi
    
    # Check for recent LOGOUTs (critical if recent)
    RECENT_LOGOUT=$(tail -1000 "$log_file" | grep -c "\[LOGOUT\]" || echo "0")
    
    if [[ $RECENT_LOGOUT -gt 0 ]]; then
        LAST_LOGOUT=$(tail -1000 "$log_file" | grep "\[LOGOUT\]" | tail -1)
        echo -e "${RED}✗${NC} Recent LOGOUT detected: $LAST_LOGOUT"
        CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
    fi
    
    # Check connection health status
    UNHEALTHY=$(tail -500 "$log_file" | grep -c "Connection unhealthy" || echo "0")
    
    if [[ $UNHEALTHY -gt 0 ]]; then
        LAST_UNHEALTHY=$(tail -500 "$log_file" | grep "Connection unhealthy" | tail -1)
        echo -e "${RED}✗${NC} Unhealthy connection: $LAST_UNHEALTHY"
        CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
    fi
}

check_market_data() {
    local log_file="$1"
    echo -e "${BLUE}[CHECK]${NC} Verifying market data flow..."
    
    # Get timestamp of last quote
    LAST_QUOTE_LINE=$(grep "\[QUOTE\]" "$log_file" | tail -1)
    
    if [[ -z "$LAST_QUOTE_LINE" ]]; then
        echo -e "${RED}✗${NC} No market data quotes found in log"
        CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
        return 1
    fi
    
    # Extract timestamp (format: 2026-01-09 23:11:33.xxx)
    LAST_QUOTE_TS=$(echo "$LAST_QUOTE_LINE" | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
    
    if [[ -z "$LAST_QUOTE_TS" ]]; then
        echo -e "${YELLOW}⚠${NC} Could not parse quote timestamp"
        WARNING_COUNT=$((WARNING_COUNT + 1))
        return 0
    fi
    
    # Calculate age
    LAST_QUOTE_EPOCH=$(date -d "$LAST_QUOTE_TS" +%s 2>/dev/null || echo "0")
    NOW_EPOCH=$(date +%s)
    DATA_AGE=$((NOW_EPOCH - LAST_QUOTE_EPOCH))
    
    if [[ $DATA_AGE -gt $MAX_DATA_STALE ]]; then
        echo -e "${RED}✗${NC} Market data stale (${DATA_AGE}s since last quote)"
        echo -e "    Last quote: $LAST_QUOTE_TS"
        CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
    elif [[ $DATA_AGE -gt 60 ]]; then
        echo -e "${YELLOW}⚠${NC} Market data slow (${DATA_AGE}s since last quote)"
        WARNING_COUNT=$((WARNING_COUNT + 1))
    else
        echo -e "${GREEN}✓${NC} Market data flowing (${DATA_AGE}s since last quote)"
    fi
}

check_critical_errors() {
    local log_file="$1"
    echo -e "${BLUE}[CHECK]${NC} Scanning for critical errors..."
    
    # Check last 500 lines for errors
    ERROR_COUNT=$(tail -500 "$log_file" | grep -c "ERROR" || echo "0")
    
    if [[ $ERROR_COUNT -gt 10 ]]; then
        echo -e "${RED}✗${NC} Many errors detected ($ERROR_COUNT in last 500 lines)"
        tail -500 "$log_file" | grep "ERROR" | tail -3
        CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
    elif [[ $ERROR_COUNT -gt 0 ]]; then
        echo -e "${YELLOW}⚠${NC} Some errors detected ($ERROR_COUNT in last 500 lines)"
        tail -500 "$log_file" | grep "ERROR" | tail -1
        WARNING_COUNT=$((WARNING_COUNT + 1))
    else
        echo -e "${GREEN}✓${NC} No recent errors"
    fi
    
    # Check for specific critical patterns
    if tail -200 "$log_file" | grep -q "Max attempts reached"; then
        echo -e "${RED}✗${NC} Max reconnection attempts reached - manual intervention required"
        CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
    fi
    
    if tail -200 "$log_file" | grep -q "REJECT"; then
        REJECT_MSG=$(tail -200 "$log_file" | grep "REJECT" | tail -1)
        echo -e "${RED}✗${NC} Order rejection detected: $REJECT_MSG"
        CRITICAL_COUNT=$((CRITICAL_COUNT + 1))
    fi
}

check_bar_processing() {
    local log_file="$1"
    echo -e "${BLUE}[CHECK]${NC} Checking bar processing..."
    
    BAR_COUNT=$(tail -1000 "$log_file" | grep -c "BAR M" || echo "0")
    
    if [[ $BAR_COUNT -eq 0 ]]; then
        echo -e "${YELLOW}⚠${NC} No bars processed recently (may be waiting for first bar)"
        WARNING_COUNT=$((WARNING_COUNT + 1))
    else
        echo -e "${GREEN}✓${NC} Bars processing ($BAR_COUNT bars in last ~1000 lines)"
    fi
}

# Main execution
echo "=========================================="
echo "  cTrader Bot Health Check"
echo "  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "=========================================="
echo ""

# Run checks
check_process

LOG_FILE=$(check_log_files)

if [[ -n "$LOG_FILE" ]] && [[ -f "$LOG_FILE" ]]; then
    check_fix_connections "$LOG_FILE"
    check_market_data "$LOG_FILE"
    check_critical_errors "$LOG_FILE"
    check_bar_processing "$LOG_FILE"
fi

# Summary
echo ""
echo "=========================================="
if [[ $CRITICAL_COUNT -gt 0 ]]; then
    echo -e "${RED}Status: CRITICAL${NC} (${CRITICAL_COUNT} critical issues)"
    exit $EXIT_CRITICAL
elif [[ $WARNING_COUNT -gt 0 ]]; then
    echo -e "${YELLOW}Status: WARNING${NC} (${WARNING_COUNT} warnings)"
    exit $EXIT_WARNING
else
    echo -e "${GREEN}Status: OK${NC}"
    exit $EXIT_OK
fi
