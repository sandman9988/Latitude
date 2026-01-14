# Project Structure

## Overview
The project has been reorganized for better maintainability and clarity.

## Directory Layout

```
ctrader_trading_bot/
├── src/                    # Source code
│   ├── agents/            # Agent implementations
│   ├── core/              # Core trading system
│   ├── risk/              # Risk management
│   ├── features/          # Feature engineering
│   ├── persistence/       # State management
│   ├── monitoring/        # HUD and monitoring
│   └── utils/             # Utilities
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── validation/       # Validation tests
├── docs/                  # Documentation
│   ├── architecture/     # System design
│   ├── operations/       # Runbooks and guides
│   └── gap_analysis/     # Gap tracking
├── config/               # Configuration files
├── data/                 # Data files
├── logs/                 # Log files
├── archive/              # Archived files
└── scripts/              # Utility scripts
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
- Gap analysis in `docs/gap_analysis/`
