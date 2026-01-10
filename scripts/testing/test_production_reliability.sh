#!/bin/bash
# Quick production reliability test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Production Reliability Test"
echo "=========================================="
echo ""

# 1. Check scripts are executable
echo "[TEST] Checking script permissions..."
if [[ -x run.sh ]] && [[ -x health_check.sh ]] && [[ -x watchdog.sh ]]; then
    echo "✓ Scripts are executable"
else
    echo "✗ Making scripts executable..."
    chmod +x run.sh health_check.sh watchdog.sh
    echo "✓ Fixed"
fi

# 2. Test health check script
echo ""
echo "[TEST] Testing health_check.sh..."
if bash health_check.sh > /dev/null 2>&1; then
    echo "✓ Health check runs successfully"
else
    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 1 ]]; then
        echo "⚠ Health check WARNING (expected if bot not running)"
    elif [[ $EXIT_CODE -eq 2 ]]; then
        echo "⚠ Health check CRITICAL (expected if bot not running)"
    else
        echo "✗ Health check failed with unknown error"
        exit 1
    fi
fi

# 3. Check environment variables
echo ""
echo "[TEST] Checking environment configuration..."
source .env
required_vars=(
    "CTRADER_USERNAME"
    "CTRADER_PASSWORD_QUOTE"
    "CTRADER_PASSWORD_TRADE"
    "CTRADER_CFG_QUOTE"
    "CTRADER_CFG_TRADE"
    "SYMBOL"
    "SYMBOL_ID"
)

missing=0
for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo "✗ Missing: $var"
        missing=$((missing + 1))
    fi
done

if [[ $missing -eq 0 ]]; then
    echo "✓ All required environment variables set"
else
    echo "✗ $missing required variables missing"
    exit 1
fi

# 4. Check FIX config files
echo ""
echo "[TEST] Checking FIX configuration files..."
if [[ -f "$CTRADER_CFG_QUOTE" ]] && [[ -f "$CTRADER_CFG_TRADE" ]]; then
    echo "✓ FIX config files exist"
    echo "  - QUOTE: $CTRADER_CFG_QUOTE"
    echo "  - TRADE: $CTRADER_CFG_TRADE"
else
    echo "✗ FIX config files missing"
    exit 1
fi

# 5. Test signal handling (Python import check)
echo ""
echo "[TEST] Testing Python dependencies..."
source .venv/bin/activate 2>/dev/null || true
python3 -c "
import signal
import quickfix44
import sys
print('✓ Python dependencies OK')
print('  - Python:', sys.version.split()[0])
print('  - QuickFIX: Available')
print('  - Signal handling: Available')
" || {
    echo "✗ Python dependency check failed"
    exit 1
}

# 6. Test graceful shutdown (import check)
echo ""
echo "[TEST] Checking graceful shutdown implementation..."
if grep -q "def graceful_shutdown" ctrader_ddqn_paper.py; then
    echo "✓ Graceful shutdown implemented"
else
    echo "✗ Graceful shutdown not found"
    exit 1
fi

# 7. Test reconnection logic
echo ""
echo "[TEST] Checking reconnection logic..."
if grep -q "reconnect_base_delay" ctrader_ddqn_paper.py && \
   grep -q "math.pow" ctrader_ddqn_paper.py; then
    echo "✓ Exponential backoff reconnection implemented"
else
    echo "✗ Reconnection logic incomplete"
    exit 1
fi

# 8. Systemd service file
echo ""
echo "[TEST] Checking systemd service file..."
if [[ -f ctrader-bot@.service ]]; then
    echo "✓ Systemd service file exists"
    if grep -q "Restart=always" ctrader-bot@.service; then
        echo "✓ Auto-restart enabled"
    else
        echo "⚠ Auto-restart not configured"
    fi
else
    echo "⚠ Systemd service file not found (optional)"
fi

# Summary
echo ""
echo "=========================================="
echo "✓ Production Reliability Test PASSED"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review .env configuration (especially credentials)"
echo "2. Start bot: bash run.sh"
echo "3. Monitor health: bash health_check.sh"
echo "4. Optional: Start watchdog: nohup bash watchdog.sh &"
echo "5. Optional: Install systemd service (see PRODUCTION_DEPLOYMENT.md)"
echo ""
