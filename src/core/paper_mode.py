#!/usr/bin/env python3
"""Paper Trading Mode Configuration.
================================
Training mode with NO GATING - pure exploration for learning.

Philosophy:
  - In training: NO gates, let agents explore freely
  - Reward shaping teaches what works
  - Natural selection through learning (not hard gates)
  - In live trading: Learned behavior gates entries naturally

Usage:
  1. Source this as environment variables before running the bot
  2. Or import and call setup_paper_mode() in your script

Settings:
  - NO feasibility gate (threshold = 0.0)
  - High epsilon for exploration (start 1.0, end 0.25)
  - Aggressive random exploration
  - All entries allowed - learning determines quality
"""

import logging
import os

from src.constants import (
    LIVE_EPSILON_DECAY,
    LIVE_EPSILON_END,
    LIVE_EPSILON_START,
    PAPER_EPSILON_DECAY,
    PAPER_EPSILON_END,
    PAPER_EPSILON_START,
    PAPER_FORCE_EXPLORATION,
)

LOG = logging.getLogger(__name__)


def setup_paper_mode():
    """Configure training mode with NO GATING for pure exploration.

    Philosophy: Let the agents explore freely. Reward shaping will
    teach what works. No hard gates - learning provides the filter.

    Call this before creating the trading bot components.
    """
    # Core paper mode flag
    os.environ["PAPER_MODE"] = "1"

    # DISABLE ALL GATING - let agents explore freely
    os.environ["FEAS_THRESHOLD"] = "0.0"  # NO feasibility gate
    os.environ["DISABLE_GATES"] = "1"  # Signal to disable all hard gates

    # HIGH EXPLORATION for learning
    os.environ["EPSILON_START"] = str(PAPER_EPSILON_START)  # Start with 100% random
    os.environ["EPSILON_END"] = str(PAPER_EPSILON_END)  # Keep 25% random floor
    os.environ["EPSILON_DECAY"] = str(PAPER_EPSILON_DECAY)  # Slower decay
    os.environ["EXPLORATION_BOOST"] = "0.5"  # 50% random exploration

    # Force trades for experience collection
    os.environ["MAX_BARS_INACTIVE"] = "10"  # Force trade after 10 bars flat
    os.environ["MIN_TRADES_PER_DAY"] = "50"  # Target many trades
    os.environ["FORCE_EXPLORATION"] = "1" if PAPER_FORCE_EXPLORATION else "0"

    LOG.info("[PAPER] ═══════════════════════════════════════════")
    LOG.info("[PAPER] TRAINING MODE - NO GATING - PURE EXPLORATION")
    LOG.info("[PAPER] ═══════════════════════════════════════════")
    LOG.info("[PAPER] Feasibility gate: DISABLED (threshold=0)")
    LOG.info("[PAPER] Epsilon: %.2f → %.2f (decay=%.4f)", PAPER_EPSILON_START, PAPER_EPSILON_END, PAPER_EPSILON_DECAY)
    LOG.info("[PAPER] Exploration boost: 50%%")
    LOG.info("[PAPER] Force trade after 10 bars flat")
    LOG.info("[PAPER] Philosophy: Learn through reward, not hard gates")

    return {
        "paper_mode": True,
        "disable_gates": True,
        "feasibility_threshold": 0.0,
        "epsilon_start": PAPER_EPSILON_START,
        "epsilon_end": PAPER_EPSILON_END,
        "epsilon_decay": PAPER_EPSILON_DECAY,
        "exploration_boost": 0.5,
        "max_bars_inactive": 10,
        "min_trades_per_day": 50,
        "force_exploration": PAPER_FORCE_EXPLORATION,
    }


def setup_live_mode():
    """Configure live trading mode with LEARNED GATING.

    Gates are based on what the agent learned works,
    not hard-coded thresholds. Confidence floor at 55%.
    """
    os.environ["PAPER_MODE"] = "0"
    os.environ["DISABLE_GATES"] = "0"

    # Low exploration - trust learned policy
    os.environ["EPSILON_START"] = str(LIVE_EPSILON_START)
    os.environ["EPSILON_END"] = str(LIVE_EPSILON_END)
    os.environ["EPSILON_DECAY"] = str(LIVE_EPSILON_DECAY)
    os.environ["EXPLORATION_BOOST"] = "0.0"

    # LEARNED GATING via confidence floor
    os.environ["CONFIDENCE_FLOOR"] = "0.55"  # Only take trades with >55% predicted probability
    os.environ["FEAS_THRESHOLD"] = "0.5"  # Use learned feasibility
    os.environ["MAX_BARS_INACTIVE"] = "1000"  # No forced trades
    os.environ["FORCE_EXPLORATION"] = "0"

    LOG.info("[LIVE] Live trading mode - using learned policy")
    LOG.info("[LIVE] Confidence floor: 55%% (model must predict >55%% win probability)")

    return {
        "paper_mode": False,
        "disable_gates": False,
        "confidence_floor": 0.55,
        "feasibility_threshold": 0.5,
        "epsilon_start": LIVE_EPSILON_START,
        "epsilon_end": LIVE_EPSILON_END,
        "epsilon_decay": LIVE_EPSILON_DECAY,
        "exploration_boost": 0.0,
        "force_exploration": False,
    }


def get_paper_settings() -> dict:
    """Get current paper trading settings from environment."""
    paper_mode = os.environ.get("PAPER_MODE", "0") == "1"

    return {
        "paper_mode": paper_mode,
        "disable_gates": os.environ.get("DISABLE_GATES", "0") == "1",
        "feasibility_threshold": float(os.environ.get("FEAS_THRESHOLD", "0.5")),
        "epsilon_start": float(
            os.environ.get("EPSILON_START", str(PAPER_EPSILON_START) if paper_mode else str(LIVE_EPSILON_START)),
        ),
        "epsilon_end": float(
            os.environ.get("EPSILON_END", str(PAPER_EPSILON_END) if paper_mode else str(LIVE_EPSILON_END)),
        ),
        "epsilon_decay": float(
            os.environ.get("EPSILON_DECAY", str(PAPER_EPSILON_DECAY) if paper_mode else str(LIVE_EPSILON_DECAY)),
        ),
        "exploration_boost": float(os.environ.get("EXPLORATION_BOOST", "0.5" if paper_mode else "0.0")),
        "max_bars_inactive": int(os.environ.get("MAX_BARS_INACTIVE", "10" if paper_mode else "1000")),
        "min_trades_per_day": float(os.environ.get("MIN_TRADES_PER_DAY", "50" if paper_mode else "2")),
        "force_exploration": os.environ.get(
            "FORCE_EXPLORATION",
            "1" if paper_mode and PAPER_FORCE_EXPLORATION else "0",
        )
        == "1",
    }


if __name__ == "__main__":
    # Print settings for shell sourcing
    pass
