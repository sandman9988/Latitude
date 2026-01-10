#!/bin/bash
# analyze_trades.sh
# Quick script to analyze all trades from Phase 1/2/3
#
# Usage: ./analyze_trades.sh [trades_csv_file]
# If no file specified, analyzes most recent in trades/ directory

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Trade Analysis Tool${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Find CSV file
if [ $# -eq 0 ]; then
    # Look for most recent trades CSV
    TRADES_CSV=$(find trades/ exports/ -name "trades_*.csv" -type f 2>/dev/null | sort -r | head -1)
    
    if [ -z "$TRADES_CSV" ]; then
        echo -e "${YELLOW}No trades CSV files found in trades/ or exports/${NC}"
        echo ""
        echo "Looking for any CSV in current directory..."
        TRADES_CSV=$(find . -maxdepth 1 -name "*.csv" -type f 2>/dev/null | head -1)
        
        if [ -z "$TRADES_CSV" ]; then
            echo -e "${YELLOW}No CSV files found.${NC}"
            echo ""
            echo "Please specify CSV file: ./analyze_trades.sh <file.csv>"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}Found: ${TRADES_CSV}${NC}"
else
    TRADES_CSV=$1
    if [ ! -f "$TRADES_CSV" ]; then
        echo "Error: File not found: $TRADES_CSV"
        exit 1
    fi
fi

echo ""

# Run analysis
echo -e "${BLUE}Running analysis...${NC}"
echo ""

python3 trade_analyzer.py "$TRADES_CSV"

echo ""
echo -e "${GREEN}✓ Analysis complete${NC}"
