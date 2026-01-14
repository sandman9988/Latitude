#!/usr/bin/env python3
"""
Project Reorganization Script
Safely reorganizes the ctrader_trading_bot project structure
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.resolve()

# New directory structure
DIRECTORIES = {
    "src/agents": "Agent implementations (trigger, harvester, arena)",
    "src/core": "Core trading system and training components",
    "src/risk": "Risk management modules",
    "src/features": "Feature engineering",
    "src/persistence": "State management and persistence",
    "src/monitoring": "HUD and monitoring tools",
    "src/utils": "Utility modules",
    "tests/unit": "Unit tests",
    "tests/integration": "Integration tests",
    "tests/validation": "Validation and flow tests",
    "docs/architecture": "System architecture documentation",
    "docs/operations": "Operations and runbooks",
    "docs/gap_analysis": "Gap analysis and tracking",
    "archive/old_docs": "Archived documentation",
    "archive/old_tests": "Archived tests",
    "archive/old_scripts": "Archived scripts",
    "logs/archived": "Archived logs",
}

# File mappings: source -> destination
FILE_MOVES = {
    # Agent files
    "trigger_agent.py": "src/agents/",
    "harvester_agent.py": "src/agents/",
    "agent_arena.py": "src/agents/",
    "dual_policy.py": "src/agents/",
    # Core system files
    "ctrader_ddqn_paper.py": "src/core/",
    "ddqn_network.py": "src/core/",
    "trade_manager.py": "src/core/",
    "paper_mode.py": "src/core/",
    "broker_execution_model.py": "src/core/",
    "order_book.py": "src/core/",
    # Training/Learning
    "adaptive_regularization.py": "src/core/",
    "early_stopping.py": "src/core/",
    "ensemble_tracker.py": "src/core/",
    "generalization_monitor.py": "src/core/",
    "reward_shaper.py": "src/core/",
    "reward_integrity_monitor.py": "src/core/",
    "cold_start_manager.py": "src/core/",
    "feedback_loop_breaker.py": "src/core/",
    # Risk management
    "risk_manager.py": "src/risk/",
    "risk_aware_sac_manager.py": "src/risk/",
    "circuit_breakers.py": "src/risk/",
    "path_geometry.py": "src/risk/",
    "friction_costs.py": "src/risk/",
    # Feature engineering
    "feature_engine.py": "src/features/",
    "feature_tournament.py": "src/features/",
    "time_features.py": "src/features/",
    "event_time_features.py": "src/features/",
    "regime_detector.py": "src/features/",
    # Persistence
    "atomic_persistence.py": "src/persistence/",
    "journaled_persistence.py": "src/persistence/",
    "bot_persistence.py": "src/persistence/",
    "learned_parameters.py": "src/persistence/",
    # Monitoring
    "hud_tabbed.py": "src/monitoring/",
    "production_monitor.py": "src/monitoring/",
    "performance_tracker.py": "src/monitoring/",
    "activity_monitor.py": "src/monitoring/",
    "trade_analyzer.py": "src/monitoring/",
    "trade_exporter.py": "src/monitoring/",
    "audit_logger.py": "src/monitoring/",
    # Utilities
    "safe_math.py": "src/utils/",
    "safe_utils.py": "src/utils/",
    "ring_buffer.py": "src/utils/",
    "sum_tree.py": "src/utils/",
    "experience_buffer.py": "src/utils/",
    "secure_random.py": "src/utils/",
    "non_repaint_guards.py": "src/utils/",
    # Unit tests
    "test_calculation_safety.py": "tests/unit/",
    "test_math_verification.py": "tests/unit/",
    "test_multi_position.py": "tests/unit/",
    "test_reward_calculations.py": "tests/unit/",
    # Integration tests
    "test_harvester_integration.py": "tests/integration/",
    "test_phase3_5_integration.py": "tests/integration/",
    "test_risk_aware_sac.py": "tests/integration/",
    "test_risk_manager_rl.py": "tests/integration/",
    # Validation tests
    "test_bar_building.py": "tests/validation/",
    "test_bar_closure_flow.py": "tests/validation/",
    "test_decision_flow.py": "tests/validation/",
    "test_decision_log_format.py": "tests/validation/",
    "test_harvester_exit.py": "tests/validation/",
    "test_harvester_flow.py": "tests/validation/",
    "test_on_bar_close_verification.py": "tests/validation/",
    "test_composite_predictor.py": "tests/validation/",
    "test_risk_manager.py": "tests/validation/",
    "test_trade_manager_safety.py": "tests/validation/",
    "test_hud_plumbing.py": "tests/validation/",
    "verify_runtime.py": "tests/validation/",
    # Architecture docs
    "SYSTEM_ARCHITECTURE.md": "docs/architecture/",
    "SYSTEM_FLOW.md": "docs/architecture/",
    "ORDER_EXECUTION_FLOW.md": "docs/architecture/",
    "DECISION_FLOW_VERIFICATION.md": "docs/architecture/",
    "MULTI_POSITION_IMPLEMENTATION.md": "docs/architecture/",
    "MULTI_POSITION_ANALYSIS.md": "docs/architecture/",
    "DOCS_INDEX.md": "docs/",
    "MASTER_HANDBOOK.md": "docs/",
    # Operations docs
    "RUNNING_WITH_LOGS.md": "docs/operations/",
    "HUD_QUICK_REFERENCE.md": "docs/operations/",
    "HUD_REVIEW_REPORT.md": "docs/operations/",
    # Gap analysis docs
    "GAP_ANALYSIS_AND_REMEDIATION_SCHEDULE.md": "docs/gap_analysis/",
    "GAP_FIXES_APPLIED.md": "docs/gap_analysis/",
    "PRODUCTION_DEPLOYMENT_GAPS.md": "docs/gap_analysis/",
    "FLOW_GAP_ANALYSIS.md": "docs/gap_analysis/",
    "AGENT_TRAINING_GAP_ANALYSIS.md": "docs/gap_analysis/",
    # Archive old documentation
    "CALCULATION_AUDIT.md": "archive/old_docs/",
    "CALCULATION_SAFETY_SUMMARY.md": "archive/old_docs/",
    "CONSOLIDATED_DOCUMENTATION_AND_GAPS.md": "archive/old_docs/",
    "MATH_REVIEW_SUMMARY.md": "archive/old_docs/",
    "MATH_VERIFICATION_AUDIT.md": "archive/old_docs/",
    "IMPLEMENTATION_INVENTORY.md": "archive/old_docs/",
    "RISK_MANAGER_IMPLEMENTATION.md": "archive/old_docs/",
    # Archive examples
    "trade_manager_example.py": "archive/",
    "trade_manager_safety.py": "archive/",
}


def create_directories():
    """Create new directory structure"""
    print("📁 Creating directory structure...")
    for dir_path, description in DIRECTORIES.items():
        full_path = ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path:30s} - {description}")


def create_init_files():
    """Create __init__.py files for Python packages"""
    print("\n🐍 Creating __init__.py files...")

    init_files = {
        "src/__init__.py": "# cTrader Trading Bot - Main Package\n",
        "src/agents/__init__.py": "# Agent implementations\n",
        "src/core/__init__.py": "# Core trading system\n",
        "src/risk/__init__.py": "# Risk management\n",
        "src/features/__init__.py": "# Feature engineering\n",
        "src/persistence/__init__.py": "# State persistence\n",
        "src/monitoring/__init__.py": "# Monitoring and HUD\n",
        "src/utils/__init__.py": "# Utility modules\n",
        "tests/__init__.py": "# Test suite\n",
        "tests/unit/__init__.py": "# Unit tests\n",
        "tests/integration/__init__.py": "# Integration tests\n",
        "tests/validation/__init__.py": "# Validation tests\n",
    }

    for file_path, content in init_files.items():
        full_path = ROOT / file_path
        with open(full_path, "w") as f:
            f.write(content)
        print(f"  ✓ {file_path}")


def move_files():
    """Move files to new locations"""
    print("\n📦 Moving files...")
    moved = 0
    skipped = 0

    for source, dest_dir in FILE_MOVES.items():
        src_path = ROOT / source
        dest_path = ROOT / dest_dir / source

        if src_path.exists():
            if dest_path.exists():
                print(f"  ⚠ Already exists: {dest_path}")
                skipped += 1
            else:
                shutil.move(str(src_path), str(dest_path))
                print(f"  ✓ {source:40s} → {dest_dir}")
                moved += 1
        else:
            print(f"  ⊘ Not found: {source}")
            skipped += 1

    print(f"\n  Moved: {moved}, Skipped: {skipped}")


def update_run_script():
    """Update run.sh to use new structure"""
    print("\n🔧 Updating run.sh...")

    run_sh = ROOT / "run.sh"
    content = run_sh.read_text()

    # Update main bot launch command
    updated = content.replace(
        'exec python3 ctrader_ddqn_paper.py "$@"', 'exec python3 -m src.core.ctrader_ddqn_paper "$@"'
    )

    run_sh.write_text(updated)
    print("  ✓ Updated bot launch command")


def update_systemd_service():
    """Update systemd service file"""
    print("\n🔧 Updating systemd service...")

    service_file = ROOT / "ctrader-bot@.service"
    if service_file.exists():
        content = service_file.read_text()

        # Update ExecStart paths
        updated = content.replace("hud_tabbed.py", "src/monitoring/hud_tabbed.py")

        service_file.write_text(updated)
        print("  ✓ Updated service file")


def create_project_readme():
    """Create a README explaining new structure"""
    print("\n📝 Creating PROJECT_STRUCTURE.md...")

    readme_content = """# Project Structure

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
"""

    (ROOT / "PROJECT_STRUCTURE.md").write_text(readme_content)
    print("  ✓ Created PROJECT_STRUCTURE.md")


def consolidate_logs():
    """Consolidate log directories"""
    print("\n📋 Consolidating logs...")

    # Ensure logs/archived exists
    archived = ROOT / "logs" / "archived"
    archived.mkdir(parents=True, exist_ok=True)

    # Move old log files
    old_log_dirs = ["log", "ctrader_py_logs"]
    for log_dir in old_log_dirs:
        dir_path = ROOT / log_dir
        if dir_path.exists() and dir_path.is_dir():
            for log_file in dir_path.glob("*.log"):
                dest = archived / log_file.name
                if not dest.exists():
                    shutil.move(str(log_file), str(dest))
                    print(f"  ✓ Archived {log_file.name}")

            # Remove empty directory
            if not any(dir_path.iterdir()):
                dir_path.rmdir()
                print(f"  ✓ Removed empty {log_dir}/")


def print_summary():
    """Print summary of reorganization"""
    print("\n" + "=" * 80)
    print("✅ Project Reorganization Complete!")
    print("=" * 80)
    print("\nNew Structure:")
    print("  src/          - All source code (organized by function)")
    print("  tests/        - All tests (organized by type)")
    print("  docs/         - All documentation (organized by category)")
    print("  archive/      - Old/unused files")
    print("  logs/         - Consolidated log directory")
    print("\nNext Steps:")
    print("  1. Review the new structure in your editor")
    print("  2. Test the bot: ./run.sh --with-hud")
    print("  3. Run tests: pytest tests/")
    print("  4. Read PROJECT_STRUCTURE.md for details")
    print("\n⚠️  Important:")
    print("  - Import paths have changed! Use 'from src.module...'")
    print("  - Update any scripts that import these modules")
    print("  - The run.sh script has been updated automatically")
    print("=" * 80)


def main():
    """Main reorganization process"""
    print("\n" + "=" * 80)
    print("  cTrader Trading Bot - Project Reorganization")
    print("=" * 80)

    try:
        create_directories()
        create_init_files()
        move_files()
        update_run_script()
        update_systemd_service()
        create_project_readme()
        consolidate_logs()
        print_summary()

        return 0

    except Exception as e:
        print(f"\n❌ Error during reorganization: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
