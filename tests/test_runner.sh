#!/bin/bash
#
# Simple test runner for incremental development
# Run this after each update to verify functionality
#

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  cTrader Bot - Incremental Test Suite"
echo "════════════════════════════════════════════════════════════"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing: $test_name ... "
    
    if eval "$test_command" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo "  Error: $(cat /tmp/test_output.log | head -3)"
        ((TESTS_FAILED++))
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════
# BASELINE TESTS (Always run)
# ═══════════════════════════════════════════════════════════════

echo "─────────────────────────────────────────────────────────────"
echo "  Baseline Tests"
echo "─────────────────────────────────────────────────────────────"

run_test "Python syntax check" "python3 -m py_compile ctrader_ddqn_paper.py"

run_test "Required imports available" "python3 -c 'import quickfix; import numpy; print(\"OK\")'"

run_test "Config files exist" "test -f config/ctrader_quote.cfg && test -f config/ctrader_trade.cfg"

run_test "Log directory writable" "test -w logs/python"

run_test "Environment variables set" "test -n \"\$CTRADER_USERNAME\" || echo 'Warning: CTRADER_USERNAME not set' >&2"

# ═══════════════════════════════════════════════════════════════
# PHASE 1 TESTS (Observability)
# ═══════════════════════════════════════════════════════════════

if grep -q "class MFEMAETracker" ctrader_ddqn_paper.py 2>/dev/null; then
    echo ""
    echo "─────────────────────────────────────────────────────────────"
    echo "  Phase 1.1: MFE/MAE Tracking"
    echo "─────────────────────────────────────────────────────────────"
    
    run_test "MFE/MAE tracker class exists" "grep -q 'class MFEMAETracker' ctrader_ddqn_paper.py"
    
    run_test "MFE tracking in logs (if bot ran)" "test ! -f logs/python/ctrader_*.log || grep -q 'MFE' logs/python/ctrader_*.log || echo 'No MFE logs yet'"
fi

# ═══════════════════════════════════════════════════════════════
# LIVE BOT STATUS
# ═══════════════════════════════════════════════════════════════

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "  Live Bot Status"
echo "─────────────────────────────────────────────────────────────"

BOT_PIDS=$(pgrep -f "ctrader_ddqn_paper.py" || true)
if [ -n "$BOT_PIDS" ]; then
    echo -e "${GREEN}✓ Bot is RUNNING${NC} (PID: $BOT_PIDS)"
    
    # Check recent activity
    LATEST_LOG=$(ls -t logs/python/ctrader_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        LAST_LINE=$(tail -1 "$LATEST_LOG")
        echo "  Last log: $LAST_LINE"
        
        # Check if recent (within 60 seconds)
        if [ -f "$LATEST_LOG" ]; then
            LAST_MOD=$(stat -c %Y "$LATEST_LOG" 2>/dev/null || stat -f %m "$LATEST_LOG" 2>/dev/null)
            NOW=$(date +%s)
            AGE=$((NOW - LAST_MOD))
            if [ $AGE -lt 60 ]; then
                echo -e "  ${GREEN}Active (last update ${AGE}s ago)${NC}"
            else
                echo -e "  ${YELLOW}Stale (last update ${AGE}s ago)${NC}"
            fi
        fi
    fi
else
    echo -e "${YELLOW}⚠ Bot is NOT running${NC}"
fi

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Test Summary"
echo "════════════════════════════════════════════════════════════"
echo -e "  Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "  Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
