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
if [ -d "$SCRIPT_DIR/.venv/bin" ]; then
    export PATH="$SCRIPT_DIR/.venv/bin:$PATH"
fi

# Logging
mkdir -p logs 2>/dev/null || true
LOG_FILE="logs/startup.log"
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# CLI / launch mode state
HUD_MODE="auto"           # auto|on|off
HUD_ONLY=0                 # --hud-only flag
INTERNAL_BOT_DAEMON=0      # internal recursive call flag
COMMAND=""                 # train|pipeline|universe|production|live-train|status|monitor|monitor-setup|weekend-train|weekend-train-setup|help
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
            pipeline|--pipeline)
                COMMAND="pipeline"
                ;;
            universe|--universe|paper|--paper)
                COMMAND="universe"
                ;;
            train|--train)
                COMMAND="train"
                ;;
            status|--status)
                COMMAND="status"
                ;;
            production|--production)
                COMMAND="production"
                ;;
            live-train|--live-train|explore|--explore)
                COMMAND="live-train"
                ;;
            monitor|--monitor)
                COMMAND="monitor"
                ;;
            monitor-setup|--monitor-setup)
                COMMAND="monitor-setup"
                ;;
            weekend-train|--weekend-train)
                COMMAND="weekend-train"
                ;;
            weekend-train-setup|--weekend-train-setup)
                COMMAND="weekend-train-setup"
                ;;
            select|--select)
                COMMAND="select"
                ;;
            help|--help|-h)
                COMMAND="help"
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

# Check for AMD GPU and load ROCm optimizations if applicable
# This should be called after load_dotenv()

# Load ROCm environment for AMD GPUs
load_rocm_env() {
    local rocm_env_file="${SCRIPT_DIR}/config/rocm_env.sh"

    # Check if AMD GPU is present
    local is_amd=0
    if command -v rocm-smi &>/dev/null; then
        if rocm-smi --showid 2>/dev/null | grep -qi "amd\|radeon\|gfx"; then
            is_amd=1
        fi
    fi

    # Also check via PyTorch if available
    if [ "$is_amd" -eq 0 ] && command -v python3 &>/dev/null; then
        is_amd=$(python3 -c "
import sys
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).upper()
        if any(x in name for x in ['AMD', 'RADEON', 'RX', 'NAVI', 'GFX']):
            print('1')
            sys.exit(0)
except: pass
print('0')
" 2>/dev/null || echo "0")
    fi

    if [ "$is_amd" -eq 1 ]; then
        log "${GREEN}✓ AMD GPU detected - loading ROCm optimizations${NC}"
        if [ -f "$rocm_env_file" ]; then
            # shellcheck disable=SC1091
            source "$rocm_env_file"
            log "${GREEN}✓ ROCm environment configured${NC}"
        else
            log "${YELLOW}⚠ config/rocm_env.sh not found - using defaults${NC}"
            # Set essential ROCm variables
            export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
            export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
            export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-1}"
            export USE_MIOPEN="${USE_MIOPEN:-1}"
            export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
        fi
    else
        log "${BLUE}ℹ NVIDIA/other GPU - skipping ROCm config${NC}"
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
    mkdir -p logs/fix/QUOTE logs/fix/TRADE logs/audit logs/ctrader logs/python data store exports
    log "${GREEN}✓ Log directories ready${NC}"
    # Rotate logs on every startup (size-based, no-op when under threshold)
    local rotate_script
    rotate_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/rotate_logs.sh"
    if [[ -x "$rotate_script" ]]; then
        "$rotate_script" 2>/dev/null || true
    fi
}

# Kill any existing bot processes
# NOTE: Only kills the single bot managed by run.sh (via .bot.pid).
# Universe-managed paper bots are intentionally left alone.
cleanup_old_processes() {
    if [[ -f .bot.pid ]]; then
        local old_pid
        old_pid=$(cat .bot.pid 2>/dev/null)
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            log "${YELLOW}⚠ Stopping previous bot (PID: ${old_pid})...${NC}"
            kill -TERM "$old_pid" 2>/dev/null || true
            sleep 2
            # Force kill if still alive
            kill -0 "$old_pid" 2>/dev/null && kill -KILL "$old_pid" 2>/dev/null || true
        fi
        rm -f .bot.pid
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
    [[ -n "$BOT_LAUNCHER_PID" ]] && kill -0 "$BOT_LAUNCHER_PID" 2>/dev/null
}

wait_for_bot_process() {
    local retries=30
    # BOT_LAUNCHER_PID becomes the bot process directly (setsid + exec in main()).
    # Don't use pgrep-by-name — that would match universe paper bots too.
    for ((i=1; i<=retries; i++)); do
        if kill -0 "$BOT_LAUNCHER_PID" 2>/dev/null; then
            echo "$BOT_LAUNCHER_PID" > .bot.pid
            log "${GREEN}✓ Trading bot running (PID: ${BOT_LAUNCHER_PID})${NC}"
            return 0
        fi
        sleep 1
    done
    log "${RED}✗ Trading bot process not detected after ${retries}s${NC}"
    log "${YELLOW}  Check logs/bot_console.log for details${NC}"
    return 1
}

seed_hud_state() {
    if compgen -G "data/performance_snapshot_*_M*.json" >/dev/null; then
        return
    fi
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

start_universe_watcher() {
    # Source Open API credentials so hubs can authenticate
    if [[ -f "${SCRIPT_DIR}/.env.openapi" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "${SCRIPT_DIR}/.env.openapi"
        set +a
    fi
    # Load ROCm env so hub subprocesses see GPU
    if [[ -f "${SCRIPT_DIR}/config/rocm_env.sh" ]]; then
        # shellcheck disable=SC1091
        source "${SCRIPT_DIR}/config/rocm_env.sh" 2>/dev/null || true
    fi
    if command -v setsid >/dev/null 2>&1; then
        setsid python3 run_universe.py --watch >> logs/run_universe.log 2>&1 &
    else
        nohup python3 run_universe.py --watch >> logs/run_universe.log 2>&1 &
    fi
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

apply_session_config() {
    local session_file="data/session.json"
    if [[ ! -f "$session_file" ]]; then
        return 0  # no session.json — fall through to existing .env values
    fi

    local py_result
    if ! py_result=$(python3 - <<'PY' 2>&1
import json, sys
from pathlib import Path

session = json.loads(Path("data/session.json").read_text())
live  = session.get("live",  [])
paper = session.get("paper", [])

# Use live[0] as the primary bot target; fall back to paper[0]
primary = live[0] if live else (paper[0] if paper else None)
if not primary:
    sys.exit(0)

tfs = sorted(primary.get("timeframes", [5]))
print("|".join([
    primary["symbol"],
    str(primary["symbol_id"]),
    str(tfs[0]),          # primary (lowest) timeframe for single-bot .env compat
    str(primary.get("qty", 0.1)),
]))
PY
    ); then
        log "${YELLOW}⚠ Could not read session.json: ${py_result}${NC}"
        return 0
    fi

    [[ -z "$py_result" ]] && return 0

    local new_symbol new_symbol_id new_tf new_qty
    IFS='|' read -r new_symbol new_symbol_id new_tf new_qty <<<"$py_result"

    # Patch .env with the session values
    python3 - "$new_symbol" "$new_symbol_id" "$new_tf" "$new_qty" <<'PY' 2>/dev/null || true
import sys
from pathlib import Path

symbol, symbol_id, tf, qty = sys.argv[1:]
env_path = Path(".env")
if not env_path.exists():
    sys.exit(0)

updates = {
    "SYMBOL":            symbol,
    "SYMBOL_ID":         symbol_id,
    "TIMEFRAME_MINUTES": tf,
    "QTY":               qty,
}
lines = env_path.read_text(encoding="utf-8").splitlines()
seen, new_lines = set(), []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in line:
        new_lines.append(line)
        continue
    key = line.partition("=")[0].strip()
    if key in updates:
        new_lines.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        new_lines.append(line)
for key, val in updates.items():
    if key not in seen:
        new_lines.append(f"{key}={val}")
env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
PY

    # Re-source so the rest of this run picks up the new values
    if [[ -f .env ]]; then
        set -a
        # shellcheck disable=SC1091
        source .env
        set +a
    fi
    log "${GREEN}✓ Session config applied → ${new_symbol} (ID: ${new_symbol_id}) M${new_tf} qty ${new_qty}${NC}"
}

sync_session_to_universe() {
    # Merge session.json paper entries into data/universe.json so the
    # universe watcher picks them up on next start.
    local session_file="data/session.json"
    [[ -f "$session_file" ]] || return 0

    python3 - <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path

session_path = Path("data/session.json")
universe_path = Path("data/universe.json")

session = json.loads(session_path.read_text())
paper_entries = session.get("paper", [])
if not paper_entries:
    sys.exit(0)

# Load or init universe
if universe_path.exists():
    try:
        universe = json.loads(universe_path.read_text())
    except Exception:
        universe = {"version": 1, "instruments": []}
else:
    universe = {"version": 1, "instruments": []}

instruments = universe.get("instruments", [])
if isinstance(instruments, dict):
    # Normalise legacy dict format to list
    instruments = list(instruments.values())

# Build a set of (symbol, tf) pairs already in universe
existing = {
    (e.get("symbol", "").upper(), int(e.get("timeframe_minutes", 0)))
    for e in instruments
}

added = 0
for entry in paper_entries:
    sym = str(entry["symbol"]).upper()
    for tf in sorted(entry.get("timeframes", [])):
        if (sym, tf) not in existing:
            instruments.append({
                "symbol":           sym,
                "timeframe_minutes": tf,
                "stage":            "PAPER",
                "z_omega":          0.0,
                "weights_path":     "",
                "promoted_at":      None,
                "updated_at":       None,
            })
            existing.add((sym, tf))
            added += 1

universe["instruments"] = instruments
universe_path.write_text(json.dumps(universe, indent=2) + "\n", encoding="utf-8")
if added:
    print(f"  Added {added} new paper instrument(s) to universe.json")
PY
}

start_bot_daemon() {
    mkdir -p logs
    local launcher_log="logs/bot_console.log"
    # Always use the absolute path to this script for recursion
    local script_path
    script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    log "${BLUE}Launching trading bot (daemon mode)...${NC}"
    # setsid gives the bot its own session so HUD terminal signals (SIGINT/SIGHUP
    # from Ctrl+C, terminal close, tab cycling, etc.) never reach the bot process.
    setsid "$script_path" --bot-daemon "${FORWARDED_ARGS[@]}" >> "$launcher_log" 2>&1 &
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
    # Kill any stale HUD before spawning a new one
    pkill -f hud_tabbed 2>/dev/null || true
    rm -f /tmp/ctrader_hud.pid
    sleep 0.3
    log ""
    log "=========================================="
    log "  Launching HUD"
    log "=========================================="
    log ""
    log "Press Ctrl+C to close the HUD. Trading bot continues running."
    log "Re-attach HUD at any time:  ./run.sh --hud-only"
    HUD_INTERRUPTED=0
    trap handle_hud_sigint INT
    local hud_status=0
    set +e
    python3 -m src.monitoring.hud_tabbed
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
    log "${BLUE}Stop bot: pkill -f ctrader_ddqn_paper${NC}"
}

# ── Pipeline commands ─────────────────────────────────────────────────────────

help_flow() {
    echo -e ""
    echo -e "${GREEN}cTrader Trading Bot${NC} — unified launcher"
    echo -e ""
    echo -e "${YELLOW}Usage:${NC}  ./run.sh [command] [options]"
    echo -e ""
    echo -e "${BLUE}Session commands:${NC}"
    echo -e "  select            interactive instrument / timeframe / mode selector"
    echo -e "                    (reads config/instruments.json, writes data/session.json)"
    echo -e ""
    echo -e "${BLUE}Bot commands:${NC}"
    echo -e "  (none)            launch bot; runs selector first if no session.json exists"
    echo -e "  production        production mode: minimal epsilon, all gates active"
    echo -e "  live-train        live training mode: full exploration, gates off"
    echo -e "  --hud-only        reattach HUD to already-running bot"
    echo -e "  --no-hud          run bot without HUD"
    echo -e ""
    echo -e "${BLUE}Pipeline commands:${NC}"
    echo -e "  train             offline training (uses session.json if present, else SYMBOLS/TIMEFRAMES)"
    echo -e "  pipeline          offline train → auto-promote → universe watcher"
    echo -e "  universe          (re)start paper trading supervisor in background"
    echo -e "  paper             alias for 'universe'"
    echo -e ""
    echo -e "${BLUE}Ops commands:${NC}"
    echo -e "  status            show running processes and universe.json summary"
    echo -e "  monitor           run one market open/close check (as cron does)"
    echo -e "  monitor-setup     install/update cron entry for market monitoring"
    echo -e "  weekend-train     guarded accept-if-better offline training, all cached TFs"
    echo -e "  weekend-train-setup install/update Saturday weekend training cron entry"
    echo -e ""
    echo -e "${BLUE}Override env vars for train/pipeline:${NC}"
    echo -e "  SYMBOLS=\"XAUUSD\"          single symbol (overrides session.json)"
    echo -e "  TIMEFRAMES=\"M60 M240\"     specific timeframes (canonical M* labels)"
    echo -e "  THRESHOLD=1.5             stricter Z-Omega gate"
    echo -e "  EPOCHS=5                  training passes per dataset"
    echo -e "  WORKERS=4                 parallel workers"
    echo -e "  WEEKEND_TRAIN_EPOCHS=3    weekend training passes per cache"
    echo -e "  WEEKEND_TRAIN_FORCE=1     bypass weekend market-close guard manually"
    echo -e ""
    echo -e "${BLUE}Examples:${NC}"
    echo -e "  ./run.sh select           pick instruments, timeframes and modes"
    echo -e "  ./run.sh                  launch with current session config"
    echo -e "  ./run.sh production"
    echo -e "  ./run.sh live-train"
    echo -e "  ./run.sh pipeline"
    echo -e "  ./run.sh weekend-train"
    echo -e "  ./run.sh weekend-train-setup"
    echo -e "  ./run.sh status"
    echo -e ""
}

status_flow() {
    log ""
    log "${BLUE}=== Process Status ===${NC}"
    log ""
    log "${YELLOW}Live/paper bot:${NC}"
    pgrep -af "ctrader_ddqn_paper" 2>/dev/null | grep -v grep || log "  (none)"
    log ""
    log "${YELLOW}Universe watcher:${NC}"
    pgrep -af "run_universe" 2>/dev/null | grep -v grep || log "  (none)"
    log ""
    log "${YELLOW}Offline training:${NC}"
    pgrep -af "train_offline" 2>/dev/null | grep -v grep || log "  (none)"
    log ""
    log "${YELLOW}HUD:${NC}"
    pgrep -af "hud_tabbed" 2>/dev/null | grep -v grep || log "  (none)"
    log ""
    log "${YELLOW}Universe (from data/universe.json):${NC}"
    python3 -c "
import json, sys
try:
    u = json.load(open('data/universe.json'))
    inst = u.get('instruments', u) if isinstance(u, dict) else u
    if isinstance(inst, list):
        for v in inst:
            zo = v.get('z_omega', 0) or 0
            print(f'  {v.get(\"symbol\",\"?\"):8s} M{v.get(\"timeframe_minutes\",\"?\"): <4} stage={v.get(\"stage\",\"?\"):6s} ZΩ={zo:.4f}')
    elif isinstance(inst, dict):
        for sym, v in inst.items():
            zo = v.get('z_omega', 0) or 0
            print(f'  {sym:8s} M{v.get(\"timeframe_minutes\",\"?\"): <4} stage={v.get(\"stage\",\"?\"):6s} ZΩ={zo:.4f}')
except Exception as e:
    print(f'  (could not read: {e})')
" 2>/dev/null || log "  (no universe.json)"
    log ""
}

select_flow() {
    # Run the interactive session selector TUI.
    # On success, data/session.json is written and .env is patched.
    # Optionally launch immediately afterwards.
    activate_venv
    log ""
    log "${BLUE}=== Session Selector ===${NC}"
    log ""
    python3 -m src.monitoring.session_selector
    local selector_exit=$?
    if [[ $selector_exit -ne 0 ]]; then
        log "${YELLOW}⚠ No session saved — no changes made.${NC}"
        return 0
    fi
    # Re-source the updated .env so subsequent commands see the new values
    if [[ -f .env ]]; then
        set -a
        # shellcheck disable=SC1091
        source .env
        set +a
    fi
    log ""
    if [[ -t 1 ]]; then
        read -r -p "Launch now with this session? (y/N): " _reply
        if [[ "$_reply" =~ ^[Yy]$ ]]; then
            orchestrate_with_hud
        fi
    fi
}

universe_flow() {
    activate_venv
    # Merge any session.json paper instruments into universe.json before starting
    sync_session_to_universe
    log ""
    log "${YELLOW}Stopping any existing universe watcher...${NC}"
    pkill -f "run_universe.py --watch" 2>/dev/null || true
    sleep 1
    mkdir -p logs
    log "${YELLOW}Stopping tracked/orphan paper bots for a clean restart...${NC}"
    python3 run_universe.py --stop-all >> logs/run_universe.log 2>&1 || true
    sleep 1
    log "${GREEN}Starting universe watcher (paper trading supervisor)...${NC}"
    start_universe_watcher
    UPID=$!
    sleep 3
    if kill -0 "$UPID" 2>/dev/null; then
        log "${GREEN}✓ Universe watcher started (PID: ${UPID})${NC}"
        log "  Log: logs/run_universe.log"
        log "  Stop: pkill -f 'run_universe.py --watch'"
        log ""
        log "${BLUE}Recent log:${NC}"
        tail -8 logs/run_universe.log 2>/dev/null || true
    else
        log "${RED}✗ Universe watcher failed to start — see logs/run_universe.log${NC}"
        tail -20 logs/run_universe.log 2>/dev/null || true
        exit 1
    fi

    # Autostart HUD once universe bots are up (skip in non-interactive shells)
    if should_enable_hud; then
        log ""
        log "${BLUE}Waiting for paper bots to come online...${NC}"
        local _i
        for ((_i=1; _i<=30; _i++)); do
            if pgrep -f "src\.core\.\(ctrader_ddqn_paper\|openapi_hub\)" >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        seed_hud_state
        launch_hud_foreground
    else
        log "${YELLOW}⚠ Non-interactive shell — skipping HUD autostart. Attach later with: ./run.sh --hud-only${NC}"
    fi
}

train_flow() {
    activate_venv
    load_rocm_env

    # Derive symbols / timeframes from session.json train block when not overridden
    local symbols timeframes
    if [[ -z "${SYMBOLS:-}" ]] && [[ -f "data/session.json" ]]; then
        symbols=$(python3 -c "
import json
s = json.load(open('data/session.json'))
t = s.get('train', {})
syms = t.get('symbols', [])
print(' '.join(syms) if syms else '')
" 2>/dev/null)
    fi
    symbols="${symbols:-${SYMBOLS:-XAUUSD BTCUSD}}"

    if [[ -z "${TIMEFRAMES:-}" ]] && [[ -f "data/session.json" ]]; then
        timeframes=$(python3 -c "
import json
s = json.load(open('data/session.json'))
t = s.get('train', {})
tfs = [f'M{tf}' for tf in t.get('timeframes', [])]
print(' '.join(tfs) if tfs else '')
" 2>/dev/null)
    fi
    timeframes="${timeframes:-${TIMEFRAMES:-M1 M5 M15 M30 M60 M240}}"

    # Build input list: downloaded CSVs + Kinetra master history + live JSONL caches
    local -a inputs=()
    if [[ -d "data/history" ]] && compgen -G "data/history/*.csv" >/dev/null 2>&1; then
        mapfile -t _csvs < <(find data/history -name '*.csv' | sort)
        inputs+=("${_csvs[@]}")
    fi
    local kinetra_dir="${HISTORY_DIR:-/home/renierdejager/Projects/Kinetra/data/master_standardized}"
    if [[ -d "$kinetra_dir" ]]; then
        inputs+=("$kinetra_dir")
    fi
    mapfile -t _caches < <(find data -maxdepth 2 -name 'training_cache_*_M*.jsonl' 2>/dev/null | sort)
    inputs+=("${_caches[@]}")

    if [[ ${#inputs[@]} -eq 0 ]]; then
        log "${RED}✗ No training data found. Run scripts/bootstrap_offline_training.sh first.${NC}"
        exit 1
    fi

    log ""
    log "${BLUE}=== Offline Training ===${NC}"
    log "  Inputs  : ${#inputs[@]} source(s)"
    log "  Symbols : $symbols"
    log "  TFs     : $timeframes"
    log "  GPU     : ${HSA_OVERRIDE_GFX_VERSION:-unset} workers=${WORKERS:-1}"
    log ""
    # shellcheck disable=SC2206
    SYM_ARR=($symbols)
    TF_ARR=($timeframes)
    python3 train_offline.py "${inputs[@]}" \
        --symbols "${SYM_ARR[@]}" \
        --timeframes "${TF_ARR[@]}" \
        --workers "${WORKERS:-1}" \
        --n-epochs "${EPOCHS:-5}" \
        --warm-start \
        --auto-promote \
        --paper-threshold "${THRESHOLD:-1.0}" \
        --retrain-rounds "${RETRAIN_ROUNDS:-3}" \
        --tournament-variants "${TOURNAMENT_VARIANTS:-6}" \
        "${FORWARDED_ARGS[@]}"
}

pipeline_flow() {
    # Full offline training → auto-promote → launch paper universe watcher
    train_flow
    log ""
    log "${GREEN}Training complete — starting universe watcher...${NC}"
    universe_flow
}

production_flow() {
    # Production mode: minimal exploration, all learned gates active
    export PAPER_MODE=0
    export DISABLE_GATES=0
    export EPSILON_START=0.05
    export EPSILON_END=0.01
    export EPSILON_DECAY=0.9995
    export EXPLORATION_BOOST=0.0
    export FORCE_EXPLORATION=0
    export MAX_BARS_INACTIVE=1000
    export DDQN_ONLINE_LEARNING=1
    log ""
    log "${BLUE}=== Production Mode ===${NC}"
    log "  PAPER_MODE=0 | gates ACTIVE | epsilon 0.05 → 0.01 | online learning ON"
    log ""
    if [[ ! -f "data/learned_parameters.json" ]]; then
        log "${YELLOW}⚠ No learned_parameters.json found.${NC}"
        log "  Consider running: ./run.sh train"
        read -r -p "Continue anyway with defaults? (y/N): " _reply
        [[ "$_reply" =~ ^[Yy]$ ]] || { log "Aborted."; exit 1; }
    fi
    orchestrate_with_hud
}

live_train_flow() {
    # Live/paper training mode: full exploration, no confidence gates
    export PAPER_MODE=1
    export DISABLE_GATES=1
    export EPSILON_START=1.0
    export EPSILON_END=0.25
    export EPSILON_DECAY=0.9998
    export EXPLORATION_BOOST=0.5
    export FORCE_EXPLORATION=1
    export MAX_BARS_INACTIVE=10
    export DDQN_ONLINE_LEARNING=1
    log ""
    log "${BLUE}=== Live Training Mode ===${NC}"
    log "  PAPER_MODE=1 | gates OFF | epsilon 1.0 → 0.25 | full exploration"
    log ""
    orchestrate_with_hud
}

monitor_flow() {
    # Run one market-monitor check (same logic called by cron every 30 min)
    if [[ ! -x "$SCRIPT_DIR/scripts/market_monitor.sh" ]]; then
        chmod +x "$SCRIPT_DIR/scripts/market_monitor.sh"
    fi
    exec bash "$SCRIPT_DIR/scripts/market_monitor.sh" "${FORWARDED_ARGS[@]}"
}

monitor_setup_flow() {
    # Install (or update) the cron entry for automatic market monitoring
    if [[ ! -x "$SCRIPT_DIR/scripts/setup_market_monitor.sh" ]]; then
        chmod +x "$SCRIPT_DIR/scripts/setup_market_monitor.sh"
    fi
    exec bash "$SCRIPT_DIR/scripts/setup_market_monitor.sh"
}

weekend_train_flow() {
    # Run guarded offline training only during the weekend market-close window.
    if [[ ! -x "$SCRIPT_DIR/scripts/weekend_offline_training.sh" ]]; then
        chmod +x "$SCRIPT_DIR/scripts/weekend_offline_training.sh"
    fi
    exec bash "$SCRIPT_DIR/scripts/weekend_offline_training.sh" "${FORWARDED_ARGS[@]}"
}

weekend_train_setup_flow() {
    # Install (or update) the cron entry for guarded weekend offline training.
    if [[ ! -x "$SCRIPT_DIR/scripts/setup_weekend_training.sh" ]]; then
        chmod +x "$SCRIPT_DIR/scripts/setup_weekend_training.sh"
    fi
    exec bash "$SCRIPT_DIR/scripts/setup_weekend_training.sh"
}

hud_only_flow() {
    if ! load_dotenv; then
        exit 1
    fi
    apply_pending_profile
    apply_session_config
    apply_defaults
    activate_venv
    setup_logging
    seed_hud_state
    launch_hud_foreground
}

prompt_select_if_needed() {
    # If no session.json exists yet and we are in an interactive terminal,
    # offer the user a chance to run the selector before launching.
    [[ -f "data/session.json" ]] && return 0
    [[ -t 1 ]] || return 0
    log ""
    log "${YELLOW}ℹ No session.json found.${NC}"
    log "${BLUE}  Run the session selector to choose instruments, timeframes and modes.${NC}"
    read -r -p "  Open selector now? (Y/n): " _reply
    if [[ ! "$_reply" =~ ^[Nn]$ ]]; then
        activate_venv
        python3 -m src.monitoring.session_selector || true
        # Re-source .env after selector may have patched it
        if [[ -f .env ]]; then
            set -a
            # shellcheck disable=SC1091
            source .env
            set +a
        fi
    fi
}

orchestrate_with_hud() {
    if ! load_dotenv; then
        exit 1
    fi
    apply_pending_profile
    apply_session_config
    apply_defaults

    # Guard: if universe paper bots (managed by run_universe.py) are already
    # running, don't spawn a duplicate standalone bot — the extra LOGON would
    # cause the broker to kick every session repeatedly.  Attach the HUD to
    # the existing bots instead.
    local universe_pids
    universe_pids=$(pgrep -f "src\.core\.ctrader_ddqn_paper" 2>/dev/null || true)
    local universe_count=0
    if [[ -n "$universe_pids" ]]; then
        universe_count=$(echo "$universe_pids" | wc -l)
    fi
    if (( universe_count > 0 )); then
        log "${YELLOW}⚠ Detected ${universe_count} paper bot(s) already running (likely universe-managed).${NC}"
        log "${YELLOW}  Skipping duplicate bot start to avoid FIX session conflicts.${NC}"
        log "${BLUE}  Attaching HUD to existing bots...${NC}"
        activate_venv
        setup_logging
        seed_hud_state
        launch_hud_foreground
        return
    fi

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
    apply_session_config
    apply_defaults

    # Validate environment
    check_env

    # Setup Python
    activate_venv

    # Load ROCm environment for AMD GPUs (after venv activation for Python detection)
    load_rocm_env

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
    log "  Epsilon:       ${EPSILON_START:-1.0} → ${EPSILON_END:-0.25}"
    log "  Quote Config:  ${CTRADER_CFG_QUOTE}"
    log "  Trade Config:  ${CTRADER_CFG_TRADE}"
    log ""
    log "${YELLOW}Starting trading bot...${NC}"
    log "  Log: logs/ctrader/ctrader_*.log"
    log "  Press Ctrl+C to stop"
    log ""
    log "=========================================="
    log ""

    # Run the bot
    exec python3 -m src.core.openapi_hub "$@"
}

# Run main / orchestrator
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"

    # Pipeline sub-commands take priority over HUD flags
    if [[ -n "$COMMAND" ]]; then
        case "$COMMAND" in
            pipeline)       pipeline_flow      ;;
            universe)       universe_flow      ;;
            train)          train_flow         ;;
            status)         status_flow        ;;
            production)     production_flow    ;;
            live-train)     live_train_flow    ;;
            monitor)        monitor_flow       ;;
            monitor-setup)  monitor_setup_flow ;;
            weekend-train)  weekend_train_flow ;;
            weekend-train-setup) weekend_train_setup_flow ;;
            select)         select_flow        ;;
            help)           help_flow          ;;
        esac
    elif [[ $HUD_ONLY -eq 1 ]]; then
        hud_only_flow
    elif [[ $INTERNAL_BOT_DAEMON -eq 1 ]]; then
        main "${FORWARDED_ARGS[@]}"
    elif should_enable_hud; then
        prompt_select_if_needed
        orchestrate_with_hud
    else
        main "${FORWARDED_ARGS[@]}"
    fi
fi
