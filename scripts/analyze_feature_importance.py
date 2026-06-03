#!/usr/bin/env python3
"""Analyze which observation features your trained agents actually use.
Based on lessons from feature engineering experiments.
"""

import json
from pathlib import Path

import numpy as np
import torch

# Constants
MIN_SAMPLE_SIZE_WARNING = 50
BOTTOM_PERCENTILE_CUTOFF = 0.3
HEADER_WIDTH = 70
BAR_MAX_WIDTH = 30


def print_section_header(title) -> None:
    """Print a formatted section header."""


def _convert_to_numpy(weights):
    """Convert weights tensor to numpy array if needed."""
    return weights.numpy() if hasattr(weights, "numpy") else weights


def _extract_from_state_dict(state_dict):
    """Extract weights from nested state_dict."""
    weights = next(
        (v for k, v in state_dict.items() if ("fc1" in k or "linear1" in k or "0" in k) and "weight" in k), None,
    )
    return _convert_to_numpy(weights) if weights is not None else None


def load_checkpoint(checkpoint_path):
    """Load checkpoint and extract weights from first layer.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        numpy array of weights or None if not found

    """
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    possible_keys = ["q_network.0.weight", "model_state_dict", "fc1.weight"]

    # Try direct keys first
    for key in possible_keys:
        if key in checkpoint:
            return _convert_to_numpy(checkpoint[key])

    # Try nested state_dict
    if "state_dict" in checkpoint:
        weights = _extract_from_state_dict(checkpoint["state_dict"])
        if weights is not None:
            return weights

    return None


def print_feature_importance(labels, importance, title):
    """Print feature importance analysis with formatted table.

    Args:
        labels: List of feature names
        importance: Array of importance values
        title: Analysis title

    """
    ranked = sorted(zip(labels, importance, strict=True), key=lambda x: -x[1])

    print_section_header(title)

    max_imp = max(importance) if len(importance) > 0 else 1.0
    for _label, imp in ranked:
        "#" * int((imp / max_imp) * BAR_MAX_WIDTH)

    print_section_header("REMOVAL CANDIDATES (Bottom 30%)")
    cutoff = int(len(ranked) * BOTTOM_PERCENTILE_CUTOFF)
    for _label, _imp in ranked[-cutoff:]:
        pass

    return ranked


def analyze_trigger_agent():
    """Analyze TriggerAgent feature usage from trained weights."""
    checkpoint_path = Path("data/checkpoints/trigger_online.pt")
    weights = load_checkpoint(checkpoint_path)
    if weights is None:
        return None

    # L1 importance per input feature
    importance = np.mean(np.abs(weights), axis=0)

    # TriggerAgent has 7 features (from MASTER_HANDBOOK.md)
    labels = [
        "0: distance_pct",
        "1: regime_score",
        "2: vol_norm",
        "3: momentum",
        "4: re_market",
        "5: flow_quality",
        "6: acceleration",
    ]

    if len(importance) != len(labels):
        labels = [f"{i}: feature_{i}" for i in range(len(importance))]

    return print_feature_importance(labels, importance, "TRIGGER AGENT FEATURE IMPORTANCE (L1 Weight Magnitude)")


def analyze_harvester_agent():
    """Analyze HarvesterAgent feature usage from trained weights."""
    checkpoint_path = Path("data/checkpoints/harvester_online.pt")
    weights = load_checkpoint(checkpoint_path)
    if weights is None:
        return None

    # L1 importance per input feature
    importance = np.mean(np.abs(weights), axis=0)

    # HarvesterAgent has 10 features (7 market + 3 position from MASTER_HANDBOOK.md)
    labels = [
        "0: distance_pct",
        "1: regime_score",
        "2: vol_norm",
        "3: momentum",
        "4: re_market",
        "5: flow_quality",
        "6: acceleration",
        "7: unrealized_pnl_pct",
        "8: mfe_pct",
        "9: bars_held",
    ]

    if len(importance) != len(labels):
        labels = [f"{i}: feature_{i}" for i in range(len(importance))]

    return print_feature_importance(
        labels,
        importance,
        "HARVESTER AGENT FEATURE IMPORTANCE (L1 Weight Magnitude)",
    )


def analyze_trade_discriminators() -> None:
    """Analyze what features discriminate winning vs losing trades.
    Requires decision_log.json with sufficient trade history.

    Note: Cohen's d analysis implementation pending - requires entry feature snapshots.
    See TODO in code for details.
    """
    decision_log_path = Path("data/decision_log.json")
    if not decision_log_path.exists():
        return

    try:
        with open(decision_log_path, encoding="utf-8") as f:
            trades = json.load(f)
    except (OSError, json.JSONDecodeError):
        return

    print_section_header("TRADE DISCRIMINATOR ANALYSIS")

    # Filter to closed positions only
    closed = [t for t in trades if t.get("exit_reason")]
    [t for t in closed if t.get("pnl_pct", 0) > 0]
    [t for t in closed if t.get("pnl_pct", 0) <= 0]


    if len(closed) < MIN_SAMPLE_SIZE_WARNING:
        pass

    # NOTE: Cohen's d implementation requires logging feature snapshots at entry time.
    # This is a future enhancement tracked separately.


if __name__ == "__main__":
    print_section_header("FEATURE IMPORTANCE ANALYSIS")

    trigger_results = analyze_trigger_agent()
    harvester_results = analyze_harvester_agent()
    analyze_trade_discriminators()

    print_section_header("KEY LESSONS FROM TREND_SNIPER EXPERIMENTS")

    print_section_header("RECOMMENDED NEXT STEPS")
