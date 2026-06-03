#!/bin/bash
# monitor_agents.sh
# Real-time monitoring of TriggerAgent and HarvesterAgent performance
#
# Shows:
# - Entry signals (TriggerAgent runway predictions)
# - Exit signals (HarvesterAgent capture decisions)
# - Quality assessments for both agents

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Find most recent log
LOGFILE=""
if [ $# -eq 0 ]; then
    # Try multiple locations
    for dir in logs/live_micro logs/phase0_validation logs/ctrader; do
        if [ -d "$dir" ]; then
            LOGFILE=$(ls -t "$dir"/*.log 2>/dev/null | head -1)
            if [ -n "$LOGFILE" ]; then
                break
            fi
        fi
    done
    
    if [ -z "$LOGFILE" ]; then
        echo -e "${RED}No log files found${NC}"
        echo "Start the bot first, or specify log file: ./monitor_agents.sh <logfile>"
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
echo -e "${BOLD}  Dual-Agent Performance Monitor${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "Log: ${CYAN}${LOGFILE}${NC}"
echo -e "Press ${BOLD}Ctrl+C${NC} to exit"
echo ""

# Track last position for tailing
LAST_LINE=0

while true; do
    clear
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Dual-Agent Monitor - $(date '+%H:%M:%S')${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
    echo ""
    
    # TriggerAgent entries (last 10)
    echo -e "${CYAN}━━━ TriggerAgent (Entry Signals) ━━━${NC}"
    TRIGGER_SIGNALS=$(grep -i "TRIGGER.*ENTRY\|predicted.*runway\|trigger.*quality" "$LOGFILE" 2>/dev/null | tail -10 || echo "No trigger signals yet")
    if [ "$TRIGGER_SIGNALS" = "No trigger signals yet" ]; then
        echo -e "${YELLOW}  Waiting for entry signals...${NC}"
    else
        echo "$TRIGGER_SIGNALS" | while IFS= read -r line; do
            if echo "$line" | grep -qi "good\|excellent"; then
                echo -e "  ${GREEN}✓${NC} $line"
            elif echo "$line" | grep -qi "poor\|bad"; then
                echo -e "  ${RED}✗${NC} $line"
            else
                echo -e "  ${BLUE}•${NC} $line"
            fi
        done
    fi
    echo ""
    
    # HarvesterAgent exits (last 10)
    echo -e "${MAGENTA}━━━ HarvesterAgent (Exit Signals) ━━━${NC}"
    HARVEST_SIGNALS=$(grep -i "HARVEST.*EXIT\|capture.*efficiency\|harvester.*quality" "$LOGFILE" 2>/dev/null | tail -10 || echo "No harvest signals yet")
    if [ "$HARVEST_SIGNALS" = "No harvest signals yet" ]; then
        echo -e "${YELLOW}  Waiting for exit signals...${NC}"
    else
        echo "$HARVEST_SIGNALS" | while IFS= read -r line; do
            if echo "$line" | grep -qi "good\|excellent"; then
                echo -e "  ${GREEN}✓${NC} $line"
            elif echo "$line" | grep -qi "poor\|bad"; then
                echo -e "  ${RED}✗${NC} $line"
            else
                echo -e "  ${MAGENTA}•${NC} $line"
            fi
        done
    fi
    echo ""
    
    # Recent trades with agent quality
    echo -e "${CYAN}━━━ Recent Trades (Agent Quality) ━━━${NC}"
    RECENT_TRADES=$(grep "POSITION CLOSED" "$LOGFILE" 2>/dev/null | tail -5 || echo "No closed trades yet")
    if [ "$RECENT_TRADES" = "No closed trades yet" ]; then
        echo -e "${YELLOW}  No trades closed yet${NC}"
    else
        echo "$RECENT_TRADES" | while IFS= read -r line; do
            # Extract PnL
            if echo "$line" | grep -q "pnl.*[0-9]"; then
                PNL=$(echo "$line" | grep -oP 'pnl[=:]\s*[-+]?\d+\.\d+' | grep -oP '[-+]?\d+\.\d+' || echo "0")
                if (( $(echo "$PNL > 0" | bc -l 2>/dev/null || echo 0) )); then
                    echo -e "  ${GREEN}WIN${NC} $line"
                else
                    echo -e "  ${RED}LOSS${NC} $line"
                fi
            else
                echo -e "  ${BLUE}•${NC} $line"
            fi
        done
    fi
    echo ""
    
    # Performance stats
    echo -e "${CYAN}━━━ Current Session Stats ━━━${NC}"
    TOTAL_TRADES=$(grep -c "POSITION CLOSED" "$LOGFILE" 2>/dev/null || echo "0")
    WINS=$(grep "POSITION CLOSED" "$LOGFILE" 2>/dev/null | grep -c "pnl.*[1-9]" || echo "0")
    
    if [ "$TOTAL_TRADES" -gt 0 ]; then
        WIN_RATE=$(awk "BEGIN {printf \"%.1f\", ($WINS / $TOTAL_TRADES) * 100}")
        echo -e "  Total Trades:  ${BOLD}$TOTAL_TRADES${NC}"
        echo -e "  Wins:          ${GREEN}$WINS${NC}"
        echo -e "  Win Rate:      ${BOLD}${WIN_RATE}%${NC}"
    else
        echo -e "  ${YELLOW}No trades completed yet${NC}"
    fi
    echo ""
    
    # Agent exploration status
    echo -e "${CYAN}━━━ Exploration Status ━━━${NC}"
    EPSILON=$(grep "epsilon" "$LOGFILE" | tail -1 | grep -oP 'epsilon[=:]\s*\d+\.\d+' | grep -oP '\d+\.\d+' || echo "unknown")
    echo -e "  Current Epsilon: ${BOLD}$EPSILON${NC} (exploration rate)"
    
    # Check for circuit breakers
    CB_TRIPS=$(grep -c "CIRCUIT.*BREAKER.*TRIP" "$LOGFILE" 2>/dev/null || echo "0")
    if [ "$CB_TRIPS" -gt 0 ]; then
        echo -e "  ${YELLOW}⚠${NC} Circuit Breaker Trips: ${BOLD}$CB_TRIPS${NC}"
    fi
    echo ""
    
    # Live log tail (last 8 lines)
    echo -e "${BLUE}━━━ Live Log (last 8 lines) ━━━${NC}"
    tail -8 "$LOGFILE" | sed 's/^/  /'
    echo ""
    
    echo -e "${YELLOW}Refreshing in 3s... (Ctrl+C to exit)${NC}"
    sleep 3
done
