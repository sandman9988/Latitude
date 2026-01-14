#!/bin/bash

# Monitoring Suite Launcher for cTrader DDQN Trading Bot
# Quick access to all monitoring tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        cTrader DDQN Bot - Monitoring Suite              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}✗ Virtual environment not found!${NC}"
    echo "  Creating .venv..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
MISSING_DEPS=()

python3 -c "import jupyter" 2>/dev/null || MISSING_DEPS+=("jupyter")
python3 -c "import plotly" 2>/dev/null || MISSING_DEPS+=("plotly")
python3 -c "import pandas" 2>/dev/null || MISSING_DEPS+=("pandas")

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing dependencies: ${MISSING_DEPS[*]}${NC}"
    pip install -q "${MISSING_DEPS[@]}"
fi

echo -e "${GREEN}✓ Dependencies ready${NC}"
echo

# Menu
echo "Select monitoring tool:"
echo
echo "  1) Terminal HUD        (Tabbed real-time dashboard)"
echo "  2) Jupyter Notebook    (Interactive analysis)"
echo "  3) Both                (HUD + Notebook)"
echo "  4) Test Suite          (Run online learning tests)"
echo "  5) Exit"
echo
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo -e "${GREEN}Launching Terminal HUD...${NC}"
        python3 -m src.monitoring.hud_tabbed
        ;;
    2)
        echo -e "${GREEN}Launching Jupyter Notebook...${NC}"
        echo "Opening analysis_notebook.ipynb in your browser..."
        jupyter notebook analysis_notebook.ipynb
        ;;
    3)
        echo -e "${GREEN}Launching HUD in background...${NC}"
        python3 -m src.monitoring.hud_tabbed &
        HUD_PID=$!
        echo "  PID: $HUD_PID"
        
        echo -e "${GREEN}Launching Jupyter Notebook...${NC}"
        jupyter notebook analysis_notebook.ipynb
        
        # Kill HUD when notebook exits
        kill $HUD_PID 2>/dev/null
        ;;
    4)
        echo -e "${GREEN}Running test suite...${NC}"
        python3 test_online_learning.py
        ;;
    5)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

deactivate
