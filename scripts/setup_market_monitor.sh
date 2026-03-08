#!/bin/bash
# Setup automatic market monitoring with cron
# Runs every 30 minutes starting from 22:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/market_monitor.sh"

echo "🔧 Setting up market monitor..."

# Make monitor script executable
chmod +x "$MONITOR_SCRIPT"
echo "✓ Made market_monitor.sh executable"

# Create crontab entry
CRON_ENTRY="0,30 22-23 * * * $MONITOR_SCRIPT
0,30 0-21 * * * $MONITOR_SCRIPT"

# Check if entry already exists
if crontab -l 2>/dev/null | grep -q "market_monitor.sh"; then
    echo "⚠️  Cron entry already exists. Remove it? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        # Remove existing entry
        crontab -l 2>/dev/null | grep -v "market_monitor.sh" | crontab -
        echo "✓ Removed old cron entry"
    else
        echo "ℹ️  Keeping existing entry"
        exit 0
    fi
fi

# Add new cron entry
(crontab -l 2>/dev/null; echo "# cTrader Bot Market Monitor - Run every 30 minutes") | crontab -
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "✓ Added cron entry:"
echo "  - Runs every 30 minutes (at :00 and :30)"
echo "  - Checks if markets are open"
echo "  - Auto-starts bot when markets open"

echo ""
echo "📋 Current crontab:"
crontab -l | grep -A 2 "cTrader"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To verify:"
echo "  - Check logs: tail -f logs/market_monitor.log"
echo "  - Manual test: ./market_monitor.sh"
echo "  - View cron: crontab -l"
echo ""
echo "To remove:"
echo "  crontab -l | grep -v market_monitor.sh | crontab -"
