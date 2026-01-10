#!/bin/bash
# monitor_phase1.sh
# Real-time monitoring dashboard for Phase 1 micro-position learning
#
# Usage: ./monitor_phase1.sh [log_file]
# If no log file specified, watches most recent in logs/live_micro/

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Find log file
if [ $# -eq 0 ]; then
    LOGFILE=$(ls -t logs/live_micro/learning_*.log 2>/dev/null | head -1)
    if [ -z "$LOGFILE" ]; then
        echo -e "${RED}No log files found in logs/live_micro/${NC}"
        echo -e "${YELLOW}Start Phase 1 with: ./launch_micro_learning.sh${NC}"
        exit 1
    fi
else
    LOGFILE=$1
fi

if [ ! -f "$LOGFILE" ]; then
    echo -e "${RED}Log file not found: $LOGFILE${NC}"
    exit 1
fi

echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Phase 1 Live Micro-Position Learning Monitor${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "Log File: ${CYAN}${LOGFILE}${NC}"
echo -e "Press ${BOLD}Ctrl+C${NC} to exit"
echo ""

# Function to extract metric from log
get_metric() {
    local pattern=$1
    local default=${2:-0}
    grep "$pattern" "$LOGFILE" | tail -1 | grep -oP '\d+(\.\d+)?' || echo "$default"
}

# Monitor loop
while true; do
    clear
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Phase 1 Micro-Position Learning - Live Dashboard${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo -e "Updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Extract key metrics from log
    TOTAL_TRADES=$(grep -c "POSITION CLOSED\|Order.*filled" "$LOGFILE" 2>/dev/null || echo "0")
    WINS=$(grep "POSITION CLOSED.*profit" "$LOGFILE" 2>/dev/null | grep -c "pnl.*[1-9]" || echo "0")
    LOSSES=$(grep "POSITION CLOSED.*loss" "$LOGFILE" 2>/dev/null | wc -l || echo "0")
    
    # Win rate
    if [ "$TOTAL_TRADES" -gt 0 ]; then
        WIN_RATE=$(awk "BEGIN {printf \"%.1f\", ($WINS / $TOTAL_TRADES) * 100}")
    else
        WIN_RATE="0.0"
    fi
    
    # Circuit breaker trips
    CB_TRIPS=$(grep -c "CIRCUIT BREAKER TRIP" "$LOGFILE" 2>/dev/null || echo "0")
    
    # Recent errors
    ERRORS=$(grep -c "ERROR\|Exception" "$LOGFILE" 2>/dev/null || echo "0")
    
    # Last 10 trades PnL
    LAST_10_PNL=$(grep "POSITION CLOSED" "$LOGFILE" | tail -10 | grep -oP 'pnl[=:]\s*[-+]?\d+\.\d+' | grep -oP '[-+]?\d+\.\d+' | awk '{s+=$1} END {printf "%.2f", s}' || echo "0.00")
    
    # Epsilon (exploration rate)
    EPSILON=$(grep "epsilon" "$LOGFILE" | tail -1 | grep -oP 'epsilon[=:]\s*\d+\.\d+' | grep -oP '\d+\.\d+' || echo "0.30")
    
    # Session status
    QUOTE_STATUS=$(grep "Quote session" "$LOGFILE" | tail -1 | grep -q "connected" && echo "CONNECTED" || echo "DISCONNECTED")
    TRADE_STATUS=$(grep "Trade session" "$LOGFILE" | tail -1 | grep -q "connected" && echo "CONNECTED" || echo "DISCONNECTED")
    
    # Display metrics
    echo -e "${CYAN}Trading Performance:${NC}"
    echo -e "  Total Trades:      ${BOLD}$TOTAL_TRADES${NC}"
    echo -e "  Wins:              ${GREEN}$WINS${NC}"
    echo -e "  Losses:            ${RED}$LOSSES${NC}"
    echo -e "  Win Rate:          ${BOLD}${WIN_RATE}%${NC}"
    echo -e "  Last 10 PnL:       ${BOLD}\$${LAST_10_PNL}${NC}"
    echo ""
    
    echo -e "${CYAN}System Health:${NC}"
    echo -e "  Circuit Breakers:  ${BOLD}$CB_TRIPS${NC} trips"
    echo -e "  Errors:            ${BOLD}$ERRORS${NC}"
    echo -e "  Epsilon:           ${BOLD}${EPSILON}${NC} (exploration rate)"
    echo ""
    
    echo -e "${CYAN}FIX Sessions:${NC}"
    if [ "$QUOTE_STATUS" = "CONNECTED" ]; then
        echo -e "  Quote:             ${GREEN}●${NC} ${QUOTE_STATUS}"
    else
        echo -e "  Quote:             ${RED}●${NC} ${QUOTE_STATUS}"
    fi
    
    if [ "$TRADE_STATUS" = "CONNECTED" ]; then
        echo -e "  Trade:             ${GREEN}●${NC} ${TRADE_STATUS}"
    else
        echo -e "  Trade:             ${RED}●${NC} ${TRADE_STATUS}"
    fi
    echo ""
    
    # Graduation progress
    echo -e "${CYAN}Graduation Progress (Phase 1 → Phase 2):${NC}"
    
    # Check 500+ trades
    if [ "$TOTAL_TRADES" -ge 500 ]; then
        echo -e "  ✅ 500+ trades ($TOTAL_TRADES completed)"
    else
        REMAINING=$((500 - TOTAL_TRADES))
        echo -e "  ⏳ Trade count: $TOTAL_TRADES / 500 (${REMAINING} remaining)"
    fi
    
    # Check win rate
    if [ "$TOTAL_TRADES" -gt 100 ]; then
        WIN_RATE_NUM=$(echo $WIN_RATE | tr -d '%')
        if (( $(echo "$WIN_RATE_NUM >= 45" | bc -l) )); then
            echo -e "  ✅ Win rate ≥ 45% ($WIN_RATE%)"
        else
            echo -e "  ⏳ Win rate: $WIN_RATE% / 45%"
        fi
    else
        echo -e "  ⏳ Win rate: Need 100+ trades for metric"
    fi
    
    # Check circuit breaker trips
    if [ "$TOTAL_TRADES" -gt 100 ]; then
        CB_PCT=$(awk "BEGIN {printf \"%.1f\", ($CB_TRIPS / $TOTAL_TRADES) * 100}")
        CB_PCT_NUM=$(echo $CB_PCT | tr -d '%')
        if (( $(echo "$CB_PCT_NUM < 5" | bc -l) )); then
            echo -e "  ✅ Circuit breakers < 5% (${CB_PCT}%)"
        else
            echo -e "  ⏳ Circuit breakers: ${CB_PCT}% / 5%"
        fi
    else
        echo -e "  ⏳ Circuit breakers: Need 100+ trades for metric"
    fi
    
    echo ""
    echo -e "${YELLOW}Recent Activity (last 5 lines):${NC}"
    tail -5 "$LOGFILE" | sed 's/^/  /'
    echo ""
    echo -e "${BLUE}Press Ctrl+C to exit | Refreshing in 5s...${NC}"
    
    sleep 5
done
