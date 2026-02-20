#!/usr/bin/env python3
"""
Analyze which observation features your trained agents actually use.
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


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * HEADER_WIDTH)
    print(f"  {title}")
    print("=" * HEADER_WIDTH)


def _convert_to_numpy(weights):
    """Convert weights tensor to numpy array if needed."""
    return weights.numpy() if hasattr(weights, "numpy") else weights


def _extract_from_state_dict(state_dict):
    """Extract weights from nested state_dict."""
    weights = next(
        (v for k, v in state_dict.items() if ("fc1" in k or "linear1" in k or "0" in k) and "weight" in k), None
    )
    return _convert_to_numpy(weights) if weights is not None else None


def load_checkpoint(checkpoint_path):
    """
    Load checkpoint and extract weights from first layer.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        numpy array of weights or None if not found
    """
    if not checkpoint_path.exists():
        print(f"[WARN] {checkpoint_path} not found")
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

    print("[ERROR] Could not find weights in checkpoint keys:", checkpoint.keys())
    return None


def print_feature_importance(labels, importance, title):
    """
    Print feature importance analysis with formatted table.

    Args:
        labels: List of feature names
        importance: Array of importance values
        title: Analysis title
    """
    ranked = sorted(zip(labels, importance, strict=True), key=lambda x: -x[1])

    print_section_header(title)
    print(f"{'Feature':<25s} {'Importance':>12s}  {'Bar':>30s}")
    print("-" * HEADER_WIDTH)

    max_imp = max(importance) if len(importance) > 0 else 1.0
    for label, imp in ranked:
        bar = "#" * int((imp / max_imp) * BAR_MAX_WIDTH)
        print(f"{label:<25s} {imp:>12.6f}  {bar}")

    print_section_header("REMOVAL CANDIDATES (Bottom 30%)")
    cutoff = int(len(ranked) * BOTTOM_PERCENTILE_CUTOFF)
    for label, imp in ranked[-cutoff:]:
        print(f"  {label}: {imp:.6f}")

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
        print(f"[WARN] Expected {len(labels)} features, got {len(importance)}")
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
        print(f"[WARN] Expected {len(labels)} features, got {len(importance)}")
        labels = [f"{i}: feature_{i}" for i in range(len(importance))]

    return print_feature_importance(
        labels,
        importance,
        "HARVESTER AGENT FEATURE IMPORTANCE (L1 Weight Magnitude)",
    )


def analyze_trade_discriminators():
    """
    Analyze what features discriminate winning vs losing trades.
    Requires decision_log.json with sufficient trade history.

    Note: Cohen's d analysis implementation pending - requires entry feature snapshots.
    See TODO in code for details.
    """
    decision_log_path = Path("data/decision_log.json")
    if not decision_log_path.exists():
        print(f"[WARN] {decision_log_path} not found")
        return

    try:
        with open(decision_log_path, encoding="utf-8") as f:
            trades = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[ERROR] Could not load decision log: {e}")
        return

    print_section_header("TRADE DISCRIMINATOR ANALYSIS")
    print(f"Total trades in log: {len(trades)}")

    # Filter to closed positions only
    closed = [t for t in trades if t.get("exit_reason")]
    winners = [t for t in closed if t.get("pnl_pct", 0) > 0]
    losers = [t for t in closed if t.get("pnl_pct", 0) <= 0]

    print(f"Closed trades: {len(closed)}")
    print(f"Winners: {len(winners)} ({len(winners)/max(len(closed),1)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/max(len(closed),1)*100:.1f}%)")

    if len(closed) < MIN_SAMPLE_SIZE_WARNING:
        print(f"\n[WARN] Less than {MIN_SAMPLE_SIZE_WARNING} closed trades - results may be unreliable")
        print("[LESSON] Small sample sizes can show misleading patterns!")
        print("         Wait for 500+ trades before trusting discriminator analysis")

    # NOTE: Cohen's d implementation requires logging feature snapshots at entry time.
    # This is a future enhancement tracked separately.
    print("\n[INFO] Cohen's d analysis requires entry feature snapshots")
    print("       Consider logging full observation vector at entry time")
    print("       to enable discriminator analysis (see FEATURE_ENGINEERING_LESSONS.md)")


if __name__ == "__main__":
    print_section_header("FEATURE IMPORTANCE ANALYSIS")
    print("  Based on: Feature engineering lessons from trend_sniper experiments")
    print("=" * HEADER_WIDTH)

    trigger_results = analyze_trigger_agent()
    harvester_results = analyze_harvester_agent()
    analyze_trade_discriminators()

    print_section_header("KEY LESSONS FROM TREND_SNIPER EXPERIMENTS")
    print(
        """
1. SUBTRACTION > ADDITION
   - Removing 3 noise features (+96 reward) beat adding best feature
   - Try removing low L1-weight features before adding new ones

2. SMALL SAMPLES LIE
   - 27 trades showed wrong pattern (d=0.506 accel_rate)
   - 173 trades showed truth (d=0.029 accel_rate ≈ zero)
   - WAIT FOR 500+ TRADES before trusting patterns

3. FEATURE SYNERGY
   - Some features useless alone but powerful combined
   - Test individual features first, then combinations

4. GREEDY ELIMINATION
   - Stepwise removal found optimal 17 features (from 20)
   - Every feature removed that didn't hurt performance

5. DIAGNOSTIC-FIRST
   - L1 weight analysis + Cohen's d on trade outcomes
   - Don't assume - measure what actually discriminates W vs L
    """
    )

    print_section_header("RECOMMENDED NEXT STEPS")
    print(
        """
1. Run this script regularly to track feature usage over training
2. After 500+ trades, implement Cohen's d analysis on entry features
3. Consider ablation study: remove bottom 2-3 features, retrain
4. Add observation logging at entry time for discriminator analysis
5. Cross-reference L1 weights with actual win/loss discrimination
    """
    )
