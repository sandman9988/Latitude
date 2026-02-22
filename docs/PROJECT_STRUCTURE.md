# Project Structure

## Overview
The project has been reorganized for better maintainability and clarity.

## Directory Layout

```
ctrader_trading_bot/
├── src/                    # Source code
│   ├── agents/            # Agent implementations (dual-policy, trigger, harvester)
│   ├── core/              # Core trading system (main bot, reward, trade manager)
│   ├── risk/              # Risk management (VaR, circuit breakers, friction)
│   ├── features/          # Feature engineering (regime, event-time, tournament)
│   ├── persistence/       # State management (atomic, journaled, learned params)
│   ├── monitoring/        # HUD and monitoring (tabbed HUD, audit, exporter)
│   ├── training/          # Offline training (bar cache, offline trainer)
│   └── utils/             # Utilities (safe_math, ring_buffer, experience_buffer)
├── tests/                 # Test suite (124 files, 2506 passing)
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── validation/       # Validation tests
├── docs/                  # Documentation
│   ├── architecture/     # System design documents
│   ├── guides/           # User and operator guides
│   └── operations/       # Runbooks
├── config/               # FIX configuration files
├── data/                 # Runtime data (learned params, state, trade log)
├── logs/                 # Application and FIX logs
└── scripts/              # Utility scripts
    ├── monitoring/       # Monitoring dashboards
    └── testing/          # Validation scripts
```

## Running the Bot

```bash
# Standard launch (auto-detects HUD based on terminal)
./run.sh

# Force HUD mode
./run.sh --with-hud

# Bot only (no HUD)
./run.sh --bot-only

# HUD only (connects to running bot)
./run.sh --hud-only
```

## Running Tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Specific test
pytest tests/unit/test_calculation_safety.py
```

## Importing Modules

Since source code is now in `src/`, use:

```python
from src.core.ctrader_ddqn_paper import CTraderFixApp
from src.monitoring.hud_tabbed import TabbedHUD
from src.risk.risk_manager import RiskManager
```

Or run as module:
```bash
python3 -m src.core.ctrader_ddqn_paper
```

## Key Files

- `run.sh` - Main launch script
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Project configuration
- `.env` - Environment configuration (not in repo)

## Documentation

See `docs/` for comprehensive documentation:
- Architecture diagrams in `docs/architecture/`
- Operations guides in `docs/operations/`
- User guides in `docs/guides/`
- Navigation index: `docs/INDEX.md`
- Current status: `docs/CURRENT_STATE.md`
- Master handbook: `MASTER_HANDBOOK.md` (root)
