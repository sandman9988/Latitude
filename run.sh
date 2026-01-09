#!/bin/bash
# cTrader Trading Bot Launcher
# This script starts the dual FIX session trading bot with robust error handling

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Logging
LOG_FILE="startup.log"
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Load environment from .env file
load_dotenv() {
    if [ -f .env ]; then
        log "${GREEN}✓ Loading environment from .env${NC}"
        set -a
        # shellcheck disable=SC1091
        source .env
        set +a
    else
        log "${YELLOW}⚠ No .env file found${NC}"
        log "${BLUE}  Copy .env.example to .env and configure your credentials${NC}"
        if [ ! -f .env.example ]; then
            log "${RED}✗ .env.example not found!${NC}"
            exit 1
        fi
        return 1
    fi
}

# Check required environment variables
check_env() {
    local missing=0
    local required_vars=(
        "CTRADER_USERNAME"
        "CTRADER_PASSWORD_QUOTE"
        "CTRADER_PASSWORD_TRADE"
        "SYMBOL"
        "SYMBOL_ID"
    )
    
    log "${BLUE}Checking environment variables...${NC}"
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log "${RED}✗ Missing: $var${NC}"
            missing=1
        else
            # Mask passwords in output
            if [[ "$var" == *"PASSWORD"* ]]; then
                log "${GREEN}✓ $var: ********${NC}"
            else
                log "${GREEN}✓ $var: ${!var}${NC}"
            fi
        fi
    done
    
    if [ $missing -eq 1 ]; then
        log ""
        log "${RED}ERROR: Missing required environment variables${NC}"
        log "${YELLOW}Solution: Copy .env.example to .env and fill in your credentials${NC}"
        log "  cp .env.example .env"
        log "  nano .env  # or use your preferred editor"
        exit 1
    fi
    
    log "${GREEN}✓ All required environment variables are set${NC}"
}

# Activate virtual environment
activate_venv() {
    if [ -f ".venv/bin/activate" ]; then
        log "${GREEN}✓ Activating virtual environment${NC}"
        # shellcheck disable=SC1091
        source .venv/bin/activate
        
        # Verify Python version
        python_version=$(python3 --version 2>&1 | awk '{print $2}')
        log "${GREEN}  Python version: $python_version${NC}"
        
        # Check for required modules
        if ! python3 -c "import quickfix" 2>/dev/null; then
            log "${RED}✗ quickfix module not found in venv${NC}"
            log "${YELLOW}  Installing required packages...${NC}"
            pip install -q -r requirements.txt || {
                log "${RED}✗ Failed to install requirements${NC}"
                exit 1
            }
        fi
        log "${GREEN}✓ Python environment ready${NC}"
    else
        log "${RED}✗ Virtual environment not found at .venv${NC}"
        log "${YELLOW}  Creating virtual environment...${NC}"
        python3 -m venv .venv || {
            log "${RED}✗ Failed to create venv${NC}"
            exit 1
        }
        # shellcheck disable=SC1091
        source .venv/bin/activate
        
        log "${YELLOW}  Installing requirements...${NC}"
        pip install -q --upgrade pip
        pip install -q -r requirements.txt || {
            log "${RED}✗ Failed to install requirements${NC}"
            exit 1
        }
        log "${GREEN}✓ Virtual environment created and configured${NC}"
    fi
}

# Check configuration files exist
check_configs() {
    log "${BLUE}Checking configuration files...${NC}"
    
    if [ ! -f "${CTRADER_CFG_QUOTE:-config/ctrader_quote.cfg}" ]; then
        log "${RED}✗ Quote config not found: ${CTRADER_CFG_QUOTE}${NC}"
        exit 1
    fi
    log "${GREEN}✓ Quote config: ${CTRADER_CFG_QUOTE}${NC}"
    
    if [ ! -f "${CTRADER_CFG_TRADE:-config/ctrader_trade.cfg}" ]; then
        log "${RED}✗ Trade config not found: ${CTRADER_CFG_TRADE}${NC}"
        exit 1
    fi
    log "${GREEN}✓ Trade config: ${CTRADER_CFG_TRADE}${NC}"
}

# Create log directories
setup_logging() {
    mkdir -p logs/python ctrader_py_logs data store exports
    log "${GREEN}✓ Log directories ready${NC}"
}

# Kill any existing bot processes
cleanup_old_processes() {
    if pgrep -f "ctrader_ddqn_paper.py" > /dev/null; then
        log "${YELLOW}⚠ Found running bot process, stopping...${NC}"
        pkill -f "ctrader_ddqn_paper.py" || true
        sleep 2
    fi
}

# Main startup sequence
main() {
    log ""
    log "=========================================="
    log "  cTrader DDQN Trading Bot - Startup"
    log "=========================================="
    log ""
    
    # Load environment
    if ! load_dotenv; then
        exit 1
    fi
    
    # Validate environment
    check_env
    
    # Setup Python
    activate_venv
    
    # Check configs
    check_configs
    
    # Setup directories
    setup_logging
    
    # Cleanup
    cleanup_old_processes
    
    # Display configuration
    log ""
    log "${GREEN}Configuration:${NC}"
    log "  Symbol:        ${SYMBOL} (ID: ${SYMBOL_ID})"
    log "  Quantity:      ${QTY}"
    log "  Timeframe:     M${TIMEFRAME_MINUTES}"
    log "  Mode:          $([ "${PAPER_MODE:-0}" == "1" ] && echo "PAPER (Training)" || echo "LIVE (Production)")"
    log "  Epsilon:       ${EPSILON_START:-1.0} → ${EPSILON_END:-0.1}"
    log "  Quote Config:  ${CTRADER_CFG_QUOTE}"
    log "  Trade Config:  ${CTRADER_CFG_TRADE}"
    log ""
    log "${YELLOW}Starting trading bot...${NC}"
    log "  Log: ctrader_py_logs/ctrader_*.log"
    log "  Press Ctrl+C to stop"
    log ""
    log "=========================================="
    log ""
    
    # Run the bot
    exec python3 ctrader_ddqn_paper.py
}

# Run main
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
