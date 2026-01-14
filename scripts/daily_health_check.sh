#!/bin/bash
#
# Production Health Check Script
# Runs comprehensive health checks on trading bot
# Usage: ./daily_health_check.sh [--verbose] [--email alerts@example.com]
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BOT_DIR"

VERBOSE=false
EMAIL_ALERTS=""
FAILED_CHECKS=0
TOTAL_CHECKS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --email)
            EMAIL_ALERTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((TOTAL_CHECKS++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((TOTAL_CHECKS++))
    ((FAILED_CHECKS++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((TOTAL_CHECKS++))
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo "  $1"
    fi
}

echo "=========================================="
echo "Production Health Check"
echo "Date: $(date)"
echo "=========================================="
echo ""

# ============================================================================
# CHECK 1: Bot Process Running
# ============================================================================
echo "[1/12] Checking bot process..."

if pgrep -f "ctrader_ddqn_paper.py" > /dev/null; then
    PID=$(pgrep -f "ctrader_ddqn_paper.py")
    UPTIME=$(ps -p "$PID" -o etime= | tr -d ' ')
    check_pass "Bot process running (PID: $PID, uptime: $UPTIME)"
    log_verbose "Process command: $(ps -p $PID -o command=)"
else
    check_fail "Bot process NOT running"
fi

# ============================================================================
# CHECK 2: Disk Space
# ============================================================================
echo "[2/12] Checking disk space..."

DISK_USAGE=$(df -h "$BOT_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')

if [ "$DISK_USAGE" -lt 90 ]; then
    check_pass "Disk usage: ${DISK_USAGE}% (OK)"
else
    check_fail "Disk usage: ${DISK_USAGE}% (CRITICAL - >90%)"
fi

log_verbose "Disk details: $(df -h $BOT_DIR | awk 'NR==2')"

# ============================================================================
# CHECK 3: Log File Sizes
# ============================================================================
echo "[3/12] Checking log file sizes..."

if [ -d "logs/python" ]; then
    LOG_SIZE=$(du -sh logs/python | awk '{print $1}')
    LOG_SIZE_MB=$(du -sm logs/python | awk '{print $1}')
    
    if [ "$LOG_SIZE_MB" -lt 1000 ]; then
        check_pass "Log size: $LOG_SIZE (OK)"
    elif [ "$LOG_SIZE_MB" -lt 5000 ]; then
        check_warn "Log size: $LOG_SIZE (Consider rotation)"
    else
        check_fail "Log size: $LOG_SIZE (EXCESSIVE - >5GB)"
    fi
else
    check_warn "Log directory not found"
fi

# ============================================================================
# CHECK 4: Recent Backups
# ============================================================================
echo "[4/12] Checking recent backups..."

if [ -d "backups/state" ]; then
    LATEST_BACKUP=$(find backups/state -name "state_backup_*.tar.gz" -type f -mtime -1 2>/dev/null | head -1)
    
    if [ -n "$LATEST_BACKUP" ]; then
        BACKUP_AGE=$(stat -c %Y "$LATEST_BACKUP")
        CURRENT_TIME=$(date +%s)
        AGE_HOURS=$(( (CURRENT_TIME - BACKUP_AGE) / 3600 ))
        
        if [ "$AGE_HOURS" -lt 24 ]; then
            check_pass "Backup completed ${AGE_HOURS}h ago"
            log_verbose "Latest backup: $(basename $LATEST_BACKUP)"
        else
            check_fail "Latest backup is ${AGE_HOURS}h old (>24h)"
        fi
    else
        check_fail "No backups found in last 24 hours"
    fi
else
    check_warn "Backup directory not found"
fi

# ============================================================================
# CHECK 5: State File Integrity
# ============================================================================
echo "[5/12] Checking state file integrity..."

if python3 -c "
import json, sys
try:
    with open('state/learned_parameters.json') as f:
        json.load(f)
    print('OK')
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; then
    check_pass "State files loadable (JSON valid)"
else
    check_fail "State files CORRUPT or missing"
fi

# ============================================================================
# CHECK 6: Error Rate in Recent Logs
# ============================================================================
echo "[6/12] Checking error rate..."

if [ -f "logs/python/app.log" ]; then
    # Count errors in last 1000 lines (roughly last hour)
    ERROR_COUNT=$(tail -1000 logs/python/app.log 2>/dev/null | grep -ci "ERROR\|EXCEPTION\|CRITICAL" || echo "0")
    
    if [ "$ERROR_COUNT" -lt 5 ]; then
        check_pass "Error count (last 1000 lines): $ERROR_COUNT (OK)"
    elif [ "$ERROR_COUNT" -lt 20 ]; then
        check_warn "Error count (last 1000 lines): $ERROR_COUNT (Elevated)"
    else
        check_fail "Error count (last 1000 lines): $ERROR_COUNT (HIGH)"
    fi
    
    if [ "$VERBOSE" = true ] && [ "$ERROR_COUNT" -gt 0 ]; then
        echo "  Recent errors:"
        tail -1000 logs/python/app.log | grep -i "ERROR\|EXCEPTION" | tail -3 | sed 's/^/    /'
    fi
else
    check_warn "Application log not found"
fi

# ============================================================================
# CHECK 7: FIX Connection Status
# ============================================================================
echo "[7/12] Checking FIX connection..."

if [ -f "logs/python/app.log" ]; then
    # Check last 100 lines for FIX connection status
    if tail -100 logs/python/app.log 2>/dev/null | grep -qi "FIX.*connected\|FIX.*established"; then
        check_pass "FIX connection established"
    else
        LAST_FIX=$(tail -500 logs/python/app.log 2>/dev/null | grep -i "FIX" | tail -1)
        if echo "$LAST_FIX" | grep -qi "disconnect\|lost\|failed"; then
            check_fail "FIX connection LOST"
            log_verbose "Last FIX log: $LAST_FIX"
        else
            check_warn "FIX connection status unclear"
        fi
    fi
else
    check_warn "Cannot verify FIX connection (no log)"
fi

# ============================================================================
# CHECK 8: Recent Trading Activity
# ============================================================================
echo "[8/12] Checking recent trading activity..."

if [ -f "logs/python/app.log" ]; then
    # Find last trade execution
    LAST_TRADE=$(tail -2000 logs/python/app.log 2>/dev/null | grep -i "TRADE EXECUTED\|position opened\|position closed" | tail -1)
    
    if [ -n "$LAST_TRADE" ]; then
        check_pass "Recent trade activity detected"
        log_verbose "Last trade: $(echo $LAST_TRADE | head -c 100)..."
    else
        # Could be normal (no signals), but worth noting
        check_warn "No trades in last ~2000 log lines"
    fi
else
    check_warn "Cannot verify trading activity"
fi

# ============================================================================
# CHECK 9: Memory Usage
# ============================================================================
echo "[9/12] Checking memory usage..."

if command -v free &> /dev/null; then
    MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", ($3/$2) * 100}')
    
    if [ "$MEM_USAGE" -lt 80 ]; then
        check_pass "Memory usage: ${MEM_USAGE}% (OK)"
    elif [ "$MEM_USAGE" -lt 90 ]; then
        check_warn "Memory usage: ${MEM_USAGE}% (High)"
    else
        check_fail "Memory usage: ${MEM_USAGE}% (CRITICAL)"
    fi
else
    check_warn "Cannot check memory (free command not available)"
fi

# ============================================================================
# CHECK 10: Circuit Breaker Status
# ============================================================================
echo "[10/12] Checking circuit breakers..."

if [ -f "logs/python/app.log" ]; then
    # Check for circuit breaker trips in recent logs
    CB_TRIPS=$(tail -1000 logs/python/app.log 2>/dev/null | grep -ci "circuit.*breaker.*trip\|breaker.*active" || echo "0")
    
    if [ "$CB_TRIPS" -eq 0 ]; then
        check_pass "No circuit breakers tripped"
    elif [ "$CB_TRIPS" -lt 5 ]; then
        check_warn "Circuit breakers tripped ${CB_TRIPS} times recently"
    else
        check_fail "Circuit breakers tripped ${CB_TRIPS} times (EXCESSIVE)"
    fi
    
    if [ "$VERBOSE" = true ] && [ "$CB_TRIPS" -gt 0 ]; then
        echo "  Recent breaker trips:"
        tail -1000 logs/python/app.log | grep -i "circuit.*breaker" | tail -2 | sed 's/^/    /'
    fi
else
    check_warn "Cannot verify circuit breakers"
fi

# ============================================================================
# CHECK 11: Python Dependencies
# ============================================================================
echo "[11/12] Checking Python dependencies..."

if python3 -c "import torch, numpy, pandas; print('OK')" 2>/dev/null; then
    check_pass "Core Python dependencies loadable"
else
    check_fail "Python dependency import failed"
fi

# ============================================================================
# CHECK 12: Journal File Integrity
# ============================================================================
echo "[12/12] Checking journal integrity..."

if [ -f "logs/journal/state.journal" ]; then
    # Check if journal file is readable and not corrupt
    LINE_COUNT=$(wc -l < logs/journal/state.journal 2>/dev/null || echo "0")
    
    if [ "$LINE_COUNT" -gt 0 ]; then
        # Try to parse last line as JSON
        if tail -1 logs/journal/state.journal 2>/dev/null | python3 -c "import json, sys; json.load(sys.stdin)" 2>/dev/null; then
            check_pass "Journal file intact ($LINE_COUNT entries)"
        else
            check_warn "Journal file exists but last entry not valid JSON"
        fi
    else
        check_warn "Journal file empty"
    fi
else
    check_warn "Journal file not found (may not be initialized yet)"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=========================================="
echo "Health Check Summary"
echo "=========================================="
echo "Total checks: $TOTAL_CHECKS"
echo "Failed: $FAILED_CHECKS"
echo ""

if [ "$FAILED_CHECKS" -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED${NC}"
    EXIT_CODE=0
elif [ "$FAILED_CHECKS" -lt 3 ]; then
    echo -e "${YELLOW}⚠️  MINOR ISSUES DETECTED${NC}"
    EXIT_CODE=1
else
    echo -e "${RED}❌ CRITICAL ISSUES DETECTED${NC}"
    EXIT_CODE=2
fi

# Send email alert if configured and issues found
if [ -n "$EMAIL_ALERTS" ] && [ "$FAILED_CHECKS" -gt 0 ]; then
    SUBJECT="Trading Bot Health Check: $FAILED_CHECKS Failed"
    echo "Sending email alert to $EMAIL_ALERTS..."
    # Note: Requires mail command configured
    # mail -s "$SUBJECT" "$EMAIL_ALERTS" < health_check_summary.txt || true
fi

echo ""
echo "Check completed: $(date)"
echo "=========================================="

exit $EXIT_CODE
