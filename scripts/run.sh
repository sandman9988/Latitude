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

# CLI / launch mode state
HUD_MODE="auto"           # auto|on|off
HUD_ONLY=0                 # --hud-only flag
INTERNAL_BOT_DAEMON=0      # internal recursive call flag
declare -a FORWARDED_ARGS=()
BOT_LAUNCHER_PID=""
HUD_INTERRUPTED=0

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --with-hud)
                HUD_MODE="on"
                ;;
            --no-hud|--bot-only)
                HUD_MODE="off"
                ;;
            --hud-only)
                HUD_ONLY=1
                ;;
            --bot-daemon)
                INTERNAL_BOT_DAEMON=1
                ;;
            *)
                FORWARDED_ARGS+=("$1")
                ;;
        esac
        shift
    done
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
        "QTY"
        "TIMEFRAME_MINUTES"
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

# Apply defaults for optional environment variables
apply_defaults() {
    : "${CTRADER_CFG_QUOTE:=config/ctrader_quote.cfg}"
    : "${CTRADER_CFG_TRADE:=config/ctrader_trade.cfg}"
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
    
    if [ ! -f "${CTRADER_CFG_QUOTE}" ]; then
        log "${RED}✗ Quote config not found: ${CTRADER_CFG_QUOTE}${NC}"
        exit 1
    fi
    log "${GREEN}✓ Quote config: ${CTRADER_CFG_QUOTE}${NC}"
    
    if [ ! -f "${CTRADER_CFG_TRADE}" ]; then
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

should_enable_hud() {
    if [[ $HUD_ONLY -eq 1 ]]; then
        return 0
    fi
    case "$HUD_MODE" in
        on)
            return 0
            ;;
        off)
            return 1
            ;;
        auto)
            if [[ -t 1 ]]; then
                return 0
            fi
            return 1
            ;;
    esac
}

bot_process_running() {
    pgrep -f "ctrader_ddqn_paper.py" > /dev/null
}

wait_for_bot_process() {
    local retries=30
    local bot_pid
    for ((i=1; i<=retries; i++)); do
        if bot_pid=$(pgrep -f "ctrader_ddqn_paper.py" | head -n1); then
            echo "$bot_pid" > .bot.pid
            log "${GREEN}✓ Trading bot running (PID: ${bot_pid})${NC}"
            return 0
        fi
        sleep 1
    done
    log "${RED}✗ Trading bot process not detected after ${retries}s${NC}"
    log "${YELLOW}  Check bot_console.log for details${NC}"
    return 1
}

seed_hud_state() {
    if [[ -f "data/performance_snapshot.json" ]]; then
        return
    fi
    log "${YELLOW}⚠ HUD data not found. Initializing default telemetry...${NC}"
    python3 <<'PY'
import json
from pathlib import Path

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

payload = {
    'performance_snapshot.json': {
        'daily': {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0},
        'weekly': {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0},
        'monthly': {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0},
        'lifetime': {'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    },
    'current_position.json': {
        'direction': 'FLAT', 'entry_price': 0.0, 'current_price': 0.0,
        'mfe': 0.0, 'mae': 0.0, 'unrealized_pnl': 0.0, 'bars_held': 0
    },
    'training_stats.json': {
        'trigger_buffer_size': 0, 'harvester_buffer_size': 0,
        'trigger_loss': 0.0, 'harvester_loss': 0.0
    },
    'risk_metrics.json': {
        'var': 0.0, 'kurtosis': 0.0, 'circuit_breaker': 'INACTIVE',
        'vpin': 0.0, 'vpin_zscore': 0.0
    }
}

for filename, data in payload.items():
    with open(data_dir / filename, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)
PY
    log "${GREEN}✓ Default HUD state created${NC}"
}

apply_pending_profile() {
    local pending_file="data/pending_profile.json"
    if [[ ! -f "$pending_file" ]]; then
        return
    fi

    log "${YELLOW}⚠ Pending profile detected (data/pending_profile.json). Applying...${NC}"

    local py_result
    if ! py_result=$(
        python3 <<'PY' 2>&1
import json
import sys
from pathlib import Path

pending = Path("data/pending_profile.json")
env_path = Path(".env")

if not env_path.exists():
    print(".env file not found", file=sys.stderr)
    sys.exit(1)
if not pending.exists():
    sys.exit(0)

try:
    selection = json.loads(pending.read_text())
except Exception as exc:  # noqa: BLE001
    print(f"invalid pending_profile.json: {exc}", file=sys.stderr)
    sys.exit(2)

required = {"symbol", "symbol_id", "timeframe_minutes", "qty"}
if not required.issubset(selection):
    print("pending_profile.json missing required keys", file=sys.stderr)
    sys.exit(3)

updates = {
    "SYMBOL": str(selection["symbol"]),
    "SYMBOL_ID": str(selection["symbol_id"]),
    "TIMEFRAME_MINUTES": str(selection["timeframe_minutes"]),
    "QTY": str(selection["qty"]),
    "CTRADER_SYMBOL": str(selection["symbol"]),
    "CTRADER_SYMBOL_ID": str(selection["symbol_id"]),
    "CTRADER_TIMEFRAME_MIN": str(selection["timeframe_minutes"]),
    "CTRADER_QTY": str(selection["qty"]),
}

lines = env_path.read_text(encoding="utf-8").splitlines()
seen = set()
new_lines = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith('#') or '=' not in line:
        new_lines.append(line)
        continue
    key, _, value = line.partition('=')
    key = key.strip()
    if key in updates:
        new_lines.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        new_lines.append(line)

for key, value in updates.items():
    if key not in seen:
        new_lines.append(f"{key}={value}")

env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
pending.unlink(missing_ok=True)

print("|".join([
    updates["SYMBOL"],
    updates["SYMBOL_ID"],
    updates["TIMEFRAME_MINUTES"],
    updates["QTY"],
]))
PY
    ); then
        log "${RED}✗ Failed to apply pending profile:${NC}"
        echo "$py_result"
        return
    fi

    if [[ -z "$py_result" ]]; then
        log "${YELLOW}⚠ Pending profile cleared before update could be read${NC}"
        return
    fi

    local new_symbol=""
    local new_symbol_id=""
    local new_tf=""
    local new_qty=""
    IFS='|' read -r new_symbol new_symbol_id new_tf new_qty <<<"$py_result"
    log "${GREEN}✓ .env updated → ${new_symbol} (ID: ${new_symbol_id}) M${new_tf} qty ${new_qty}${NC}"

    # Re-source environment silently so the rest of the script uses new vars
    if [[ -f .env ]]; then
        set -a
        # shellcheck disable=SC1091
        source .env
        set +a
        log "${GREEN}✓ Environment refreshed with pending profile${NC}"
    fi
}

start_bot_daemon() {
    local launcher_log="bot_console.log"
    log "${BLUE}Launching trading bot (daemon mode)...${NC}"
    ("$0" --bot-daemon "${FORWARDED_ARGS[@]}") >> "$launcher_log" 2>&1 &
    BOT_LAUNCHER_PID=$!
    sleep 2
    if ! kill -0 "$BOT_LAUNCHER_PID" 2>/dev/null; then
        log "${RED}✗ Bot launcher exited early. See ${launcher_log}${NC}"
        tail -n 40 "$launcher_log" 2>/dev/null || true
        exit 1
    fi
    wait_for_bot_process || exit 1
    log "${BLUE}Bot console log: ${launcher_log}${NC}"
}

wait_for_bot_telemetry() {
    local target="data/current_position.json"
    log "${BLUE}Waiting for bot telemetry (${target})...${NC}"
    for ((i=1; i<=30; i++)); do
        if [[ -f "$target" ]]; then
            log "${GREEN}✓ Telemetry detected${NC}"
            return 0
        fi
        if ! bot_process_running; then
            log "${RED}✗ Bot stopped before telemetry became available${NC}"
            return 1
        fi
        sleep 1
    done
    log "${YELLOW}⚠ Telemetry file not found after 30s - HUD will start anyway${NC}"
}

handle_hud_sigint() {
    HUD_INTERRUPTED=1
}

launch_hud_foreground() {
    log ""
    log "=========================================="
    log "  Launching HUD"
    log "=========================================="
    log ""
    log "Press Ctrl+C to close the HUD. Trading bot continues running."
    HUD_INTERRUPTED=0
    trap handle_hud_sigint INT
    local hud_status=0
    set +e
    python3 hud_tabbed.py
    hud_status=$?
    set -e

    # Always operate from the script directory so relative paths stay valid
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    trap - INT
    if [[ $HUD_INTERRUPTED -eq 1 ]]; then
        log "${YELLOW}HUD interrupted via Ctrl+C. Bot is still running in background.${NC}"
    elif [[ $hud_status -ne 0 ]]; then
        log "${YELLOW}HUD exited with status ${hud_status}${NC}"
    else
        log "${GREEN}HUD closed normally. Bot remains online.${NC}"
    fi
    log "${BLUE}Stop bot: pkill -f ctrader_ddqn_paper.py${NC}"
}

hud_only_flow() {
    if ! load_dotenv; then
        exit 1
    fi
    apply_pending_profile
    apply_defaults
    activate_venv
    setup_logging
    seed_hud_state
    launch_hud_foreground
}

orchestrate_with_hud() {
    if ! load_dotenv; then
        exit 1
    fi
    apply_pending_profile
    apply_defaults
    check_env
    activate_venv
    check_configs
    setup_logging
    cleanup_old_processes
    start_bot_daemon
    wait_for_bot_telemetry || true
    seed_hud_state
    launch_hud_foreground
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
    apply_pending_profile
    apply_defaults
    
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
    exec python3 ctrader_ddqn_paper.py "$@"
}

# Run main / orchestrator
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"

    if [[ $HUD_ONLY -eq 1 ]]; then
        hud_only_flow
    elif [[ $INTERNAL_BOT_DAEMON -eq 1 ]]; then
        main "${FORWARDED_ARGS[@]}"
    elif should_enable_hud; then
        orchestrate_with_hud
    else
        main "${FORWARDED_ARGS[@]}"
    fi
fi
