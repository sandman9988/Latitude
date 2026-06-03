#!/bin/bash
# cTrader Trading Bot Monitor
# Real-time status display

clear

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check if bot is running
check_status() {
    if pgrep -f "ctrader_ddqn_paper" > /dev/null; then
        echo -e "${GREEN}●${NC} Bot is RUNNING"
        PID=$(pgrep -f "ctrader_ddqn_paper" | head -1)
        echo -e "  PID: $PID"
    else
        echo -e "${RED}●${NC} Bot is STOPPED"
        echo ""
        echo "To start: ./run.sh"
        exit 0
    fi
}

# Get latest log file
get_latest_log() {
    ls -1t logs/python/*.log 2>/dev/null | head -1
}

# Parse market data
show_market_data() {
    LOG=$(get_latest_log)
    if [ -z "$LOG" ]; then
        echo -e "${RED}No log file found${NC}"
        return
    fi
    
    # Get latest market data snapshot
    SYMBOL_LABEL="${SYMBOL:-XAUUSD}"
    UPDATES=$(tail -10 "$LOG" | grep "QUOTE.*35=W" | wc -l)
    LAST_TIME=$(tail -10 "$LOG" | grep "QUOTE" | tail -1 | awk '{print $1, $2}')
    
    if [ "$UPDATES" -gt 0 ]; then
        echo -e "${CYAN}${SYMBOL_LABEL} Market Data:${NC}"
        echo -e "  Status:     ${GREEN}● LIVE${NC}"
        echo -e "  Updates:    $UPDATES in last 10 lines"
        echo -e "  Last seen:  $LAST_TIME UTC"
        echo -e "  ${YELLOW}(Viewing detailed prices in full logs)${NC}"
    else
        echo -e "${YELLOW}Waiting for market data...${NC}"
    fi
}

# Show session status
show_sessions() {
    LOG=$(get_latest_log)
    if [ -z "$LOG" ]; then
        return
    fi
    
    # Check for recent logon messages
    QUOTE_LOGON=$(tail -200 "$LOG" | grep "qual=QUOTE" | grep -E "LOGON|CREATE" | tail -1)
    TRADE_LOGON=$(tail -200 "$LOG" | grep "qual=TRADE" | grep -E "LOGON|CREATE" | tail -1)
    
    # Check for recent activity (last 10 seconds worth)
    QUOTE_ACTIVE=$(tail -50 "$LOG" | grep -c "qual=QUOTE")
    TRADE_ACTIVE=$(tail -50 "$LOG" | grep -c "qual=TRADE")
    
    echo ""
    echo -e "${CYAN}FIX Sessions:${NC}"
    
    if [ $QUOTE_ACTIVE -gt 0 ]; then
        echo -e "  QUOTE: ${GREEN}●${NC} Connected (${QUOTE_ACTIVE} recent msgs)"
    else
        echo -e "  QUOTE: ${YELLOW}●${NC} Status unknown"
    fi
    
    if [ $TRADE_ACTIVE -gt 0 ]; then
        echo -e "  TRADE: ${GREEN}●${NC} Connected (${TRADE_ACTIVE} recent msgs)"
    else
        echo -e "  TRADE: ${YELLOW}●${NC} Status unknown"
    fi
}

# Show recent bars
show_recent_bars() {
    LOG=$(get_latest_log)
    if [ -z "$LOG" ]; then
        return
    fi
    
    # Look for bar close messages
    BARS=$(tail -100 "$LOG" | grep "\[BAR\]" | tail -3)
    TF="M${TIMEFRAME_MINUTES:-1}"
    
    if [ -n "$BARS" ]; then
        echo ""
        echo -e "${CYAN}Recent ${TF} Bars:${NC}"
        echo "$BARS" | while read -r line; do
            TIME=$(echo "$line" | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')
            CLOSE=$(echo "$line" | grep -oP 'C=\K[0-9.]+')
            DESIRED=$(echo "$line" | grep -oP 'desired=\K[-0-9]+')
            CUR=$(echo "$line" | grep -oP 'cur=\K[-0-9]+')
            
            if [ -n "$TIME" ]; then
                POS_TEXT="FLAT"
                [ "$DESIRED" = "1" ] && POS_TEXT="${GREEN}LONG${NC}"
                [ "$DESIRED" = "-1" ] && POS_TEXT="${RED}SHORT${NC}"
                
                echo -e "  $TIME | Close: \$$CLOSE | Signal: $POS_TEXT"
            fi
        done
    fi
}

# Show recent trades
show_recent_trades() {
    LOG=$(get_latest_log)
    if [ -z "$LOG" ]; then
        return
    fi
    
    # Look for trade messages
    TRADES=$(tail -100 "$LOG" | grep "\[TRADE\] Sent" | tail -5)
    
    if [ -n "$TRADES" ]; then
        echo ""
        echo -e "${CYAN}Recent Orders:${NC}"
        echo "$TRADES" | while read -r line; do
            TIME=$(echo "$line" | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
            SIDE=$(echo "$line" | grep -oP 'Sent MKT \K[A-Z]+')
            QTY=$(echo "$line" | grep -oP 'qty=\K[0-9.]+')
            
            SIDE_COLOR=$GREEN
            [ "$SIDE" = "SELL" ] && SIDE_COLOR=$RED
            
            echo -e "  $TIME | ${SIDE_COLOR}$SIDE${NC} $QTY ${SYMBOL:-XAUUSD}"
        done
    fi
}

# Show log tail
show_log_tail() {
    LOG=$(get_latest_log)
    if [ -z "$LOG" ]; then
        return
    fi
    
    echo ""
    echo -e "${CYAN}Recent Activity:${NC}"
    tail -5 "$LOG" | sed 's/^/  /'
}

# Main display
echo "════════════════════════════════════════════════"
echo "  cTrader DDQN Trading Bot - Live Monitor"
echo "════════════════════════════════════════════════"
echo ""

check_status
echo ""
show_market_data
show_sessions
show_recent_bars
show_recent_trades

echo ""
echo "════════════════════════════════════════════════"
echo ""
echo "Commands:"
echo "  ./monitor.sh          - Refresh this status"
echo "  ./monitor.sh watch    - Live monitoring mode"
echo "  tail -f logs/*.log    - View full logs"
echo "  pkill -f ctrader_ddqn_paper  - Stop bot"
echo ""

# Watch mode
if [ "$1" = "watch" ]; then
    echo "Starting live monitoring... Press Ctrl+C to exit"
    echo ""
    while true; do
        sleep 3
        clear
        $0
        echo "Refreshing every 3 seconds..."
    done
fi
