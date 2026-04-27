#!/usr/bin/env python3
"""Empirical backtest: compare OLD vs NEW runway prediction pipeline.

Tests three enhancements against historical data:
  B) Multi-horizon vol ratio (σ_short/σ_long blend)
  C) HMM regime blended multiplier vs discrete VR multiplier
  A) EWMA Q→Runway calibration (simulated online learning)

Data sources:
  - data/bars_cache.json: 500 continuous M5 bars for sliding-window analysis
  - data/training_cache_XAUUSD_M5.jsonl: 150 real trade episodes

Metrics:
  - Spearman rank correlation with actual forward MFE
  - Discrimination ratio (avg MFE when forecast above vs below median)
  - Mean absolute prediction error (normalised)
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hmm_regime import HMMRegimeDetector
from src.features.regime_detector import RegimeDetector
from src.risk.path_geometry import VOL_RATIO_BLEND_WEIGHT, VOL_RATIO_NEUTRAL, VOL_RATIO_RUNWAY_SCALE
from src.utils.safe_math import SafeMath

# ── Helpers ──────────────────────────────────────────────────────────────────


def realized_vol(closes: list[float], window: int) -> float:
    """Rogers-Satchell style realized vol from close prices."""
    if len(closes) < window + 1:
        return 0.0
    rets = [
        np.log(closes[i] / closes[i - 1])
        for i in range(len(closes) - window, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]
    if len(rets) < 2:
        return 0.0
    return float(np.std(rets, ddof=1))


def forward_mfe_fractional(bars: list, start_idx: int, horizon: int) -> float | None:
    """Compute fractional forward MFE from start_idx over next `horizon` bars.
    MFE = max(high) - entry_close, normalised by entry_close.
    """
    entry_close = bars[start_idx][4]  # close
    if entry_close <= 0:
        return None
    end_idx = min(start_idx + horizon, len(bars))
    if end_idx <= start_idx + 1:
        return None
    highs = [bars[i][2] for i in range(start_idx + 1, end_idx)]  # high
    lows = [bars[i][3] for i in range(start_idx + 1, end_idx)]  # low
    # MFE is max favourable excursion (both directions matter, take magnitude)
    mfe_up = max(highs) - entry_close
    mfe_down = entry_close - min(lows)
    mfe_abs = max(mfe_up, mfe_down)
    return mfe_abs / entry_close


def old_runway(sigma: float) -> float:
    """Original static runway: 1/(1+50σ)."""
    return 1.0 / (1.0 + VOL_RATIO_RUNWAY_SCALE * sigma)


def new_runway_volratio(sigma: float, sigma_long: float) -> float:
    """Enhanced runway with vol ratio blend."""
    base = old_runway(sigma)
    if sigma_long <= 0:
        return base
    ratio = SafeMath.safe_div(sigma, sigma_long, VOL_RATIO_NEUTRAL)
    vol_adj = max(0.5, min(1.5, 1.0 - VOL_RATIO_BLEND_WEIGHT * (ratio - VOL_RATIO_NEUTRAL)))
    return base * vol_adj


def print_section(title: str) -> None:
    pass


def print_comparison_table(labels: list[str], results: dict[str, dict]) -> None:
    """Print a formatted comparison table."""
    metrics = list(next(iter(results.values())).keys())
    # Header
    col_w = 16
    f"{'Metric':<25}" + "".join(f"{label:>{col_w}}" for label in labels)
    for m in metrics:
        row = f"{m:<25}"
        for label in labels:
            val = results[label].get(m)
            if val is None:
                row += f"{'N/A':>{col_w}}"
            elif isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{val!s:>{col_w}}"


def discrimination_ratio(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Ratio of mean actual MFE when forecast > median vs <= median."""
    median = np.median(forecasts)
    above = actuals[forecasts > median]
    below = actuals[forecasts <= median]
    if len(above) == 0 or len(below) == 0:
        return 1.0
    mean_above = np.mean(above)
    mean_below = np.mean(below)
    if mean_below == 0:
        return 1.0
    return float(mean_above / mean_below)


def compute_metrics(forecasts: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute comparison metrics for a forecast series vs actuals."""
    if len(forecasts) < 5:
        return {
            "spearman_r": None,
            "spearman_p": None,
            "disc_ratio": None,
            "mae_norm": None,
            "n_points": len(forecasts),
        }

    # Spearman rank correlation
    spear_r, spear_p = stats.spearmanr(forecasts, actuals)

    # Discrimination ratio
    disc = discrimination_ratio(forecasts, actuals)

    # Mean absolute error (normalised by mean actual)
    mean_actual = np.mean(actuals) if np.mean(actuals) > 0 else 1.0
    mae_norm = float(np.mean(np.abs(forecasts - actuals)) / mean_actual)

    return {
        "spearman_r": float(spear_r),
        "spearman_p": float(spear_p),
        "disc_ratio": disc,
        "mae_norm": mae_norm,
        "n_points": len(forecasts),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1: Sliding-window analysis on bars_cache (continuous bar stream)
# ══════════════════════════════════════════════════════════════════════════════


def test_bars_cache() -> None:
    print_section("TEST 1: Sliding-Window Forward MFE — bars_cache.json (500 bars)")

    with open(PROJECT_ROOT / "data" / "bars_cache.json") as f:
        cache = json.load(f)
    raw_bars = cache["bars"]  # list of [ts, o, h, l, c]

    WARMUP = 50  # need 50 bars for sigma_long
    HORIZON = 10  # forward MFE horizon (10 bars = 50 min)
    SHORT_VOL_WIN = 10
    LONG_VOL_WIN = 50

    # Build detectors
    vr_detector = RegimeDetector(window_size=30, update_interval=1, instrument_volatility=1.0)
    hmm_detector = HMMRegimeDetector(window_size=30, update_interval=1, instrument_volatility=1.0)

    # Warm up detectors with first WARMUP bars
    for i in range(WARMUP):
        close = raw_bars[i][4]
        vr_detector.add_price(close)
        hmm_detector.add_price(close)

    # Collect forecasts
    fc_old: list[float] = []  # runway_old * discrete_mult
    fc_B: list[float] = []  # runway_volratio * discrete_mult
    fc_C: list[float] = []  # runway_old * hmm_blended_mult
    fc_BC: list[float] = []  # runway_volratio * hmm_blended_mult
    actual_mfe: list[float] = []
    hmm_fitted_count = 0

    for i in range(WARMUP, len(raw_bars) - HORIZON):
        close = raw_bars[i][4]
        vr_detector.add_price(close)
        hmm_detector.add_price(close)

        # Collect close prices for vol computation
        closes = [raw_bars[j][4] for j in range(max(0, i - LONG_VOL_WIN), i + 1)]
        sigma_s = realized_vol(closes, SHORT_VOL_WIN)
        sigma_l = realized_vol(closes, LONG_VOL_WIN)

        if sigma_s <= 0:
            continue

        # Forward MFE
        fmfe = forward_mfe_fractional(raw_bars, i, HORIZON)
        if fmfe is None or fmfe <= 0:
            continue

        # OLD: static runway * discrete regime mult
        r_old = old_runway(sigma_s)
        disc_mult = vr_detector.get_regime_multiplier()

        # NEW B: vol-ratio runway * discrete
        r_B = new_runway_volratio(sigma_s, sigma_l)

        # NEW C: static runway * HMM blended mult
        hmm_mult = hmm_detector.get_blended_runway_multiplier()
        if hmm_detector._hmm_fitted:
            hmm_fitted_count += 1

        fc_old.append(r_old * disc_mult)
        fc_B.append(r_B * disc_mult)
        fc_C.append(r_old * hmm_mult)
        fc_BC.append(r_B * hmm_mult)
        actual_mfe.append(fmfe)

    fc_old = np.array(fc_old)
    fc_B = np.array(fc_B)
    fc_C = np.array(fc_C)
    fc_BC = np.array(fc_BC)
    actual_mfe_arr = np.array(actual_mfe)


    # Compute metrics
    results = {
        "OLD (static)": compute_metrics(fc_old, actual_mfe_arr),
        "B (vol ratio)": compute_metrics(fc_B, actual_mfe_arr),
        "C (HMM blend)": compute_metrics(fc_C, actual_mfe_arr),
        "B+C (both)": compute_metrics(fc_BC, actual_mfe_arr),
    }

    labels = list(results.keys())
    print_comparison_table(labels, results)

    # Interpretation
    max(results.items(), key=lambda x: x[1].get("spearman_r") or -1)
    max(results.items(), key=lambda x: x[1].get("disc_ratio") or 0)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2: Real trade episodes from training cache
# ══════════════════════════════════════════════════════════════════════════════


def test_training_cache() -> None:
    print_section("TEST 2: Real Trade Episodes — training_cache (150 episodes)")

    episodes: list[dict[str, Any]] = []
    with open(PROJECT_ROOT / "data" / "training_cache_XAUUSD_M5.jsonl") as f:
        for line in f:
            ep = json.loads(line.strip())
            episodes.append(ep)

    # We need episodes with exit_bars (have at least 30 bars for vol + HMM warmup)
    usable = [ep for ep in episodes if len(ep.get("exit_bars", [])) >= 30]

    SHORT_VOL_WIN = 10
    LONG_VOL_WIN = 30  # Use 30 instead of 50 since episode bar count varies

    # For each episode, compute forecasts at the entry point and compare to actual MFE
    fc_old: list[float] = []
    fc_B: list[float] = []
    fc_C: list[float] = []
    fc_BC: list[float] = []
    actual_mfe_list: list[float] = []
    actual_pnl: list[float] = []

    for ep in usable:
        bars = ep["exit_bars"]  # list of [ts, o, h, l, c]
        entry_price = ep.get("entry_price", 0)
        mfe_abs = ep.get("mfe", 0)

        if entry_price <= 0 or mfe_abs is None:
            continue

        # Fractional MFE
        mfe_frac = abs(mfe_abs) / entry_price

        # Use bars leading up to entry (first ~50% of bars as "pre-entry context"
        # since bars include the holding period)
        # Actually, exit_bars contain the full holding period
        # Use bar closes for vol computation at entry
        closes = [b[4] for b in bars]

        # Compute vol at the start of the trade (using first N bars as warm-up)
        warmup_end = min(LONG_VOL_WIN, len(closes) - 1)
        if warmup_end < SHORT_VOL_WIN + 1:
            continue

        sigma_s = realized_vol(closes[: warmup_end + 1], SHORT_VOL_WIN)
        sigma_l = realized_vol(closes[: warmup_end + 1], min(LONG_VOL_WIN, warmup_end))

        if sigma_s <= 0:
            continue

        # Build regime detectors from the bar sequence
        vr = RegimeDetector(window_size=20, update_interval=1, instrument_volatility=1.0)
        hmm = HMMRegimeDetector(window_size=20, update_interval=1, instrument_volatility=1.0)
        for j in range(warmup_end + 1):
            vr.add_price(closes[j])
            hmm.add_price(closes[j])

        r_old = old_runway(sigma_s)
        r_B = new_runway_volratio(sigma_s, sigma_l)
        disc_mult = vr.get_regime_multiplier()
        hmm_mult = hmm.get_blended_runway_multiplier()

        fc_old.append(r_old * disc_mult)
        fc_B.append(r_B * disc_mult)
        fc_C.append(r_old * hmm_mult)
        fc_BC.append(r_B * hmm_mult)
        actual_mfe_list.append(mfe_frac)
        actual_pnl.append(ep.get("pnl_pts", 0))

    fc_old = np.array(fc_old)
    fc_B = np.array(fc_B)
    fc_C = np.array(fc_C)
    fc_BC = np.array(fc_BC)
    actual_mfe_arr = np.array(actual_mfe_list)
    actual_pnl_arr = np.array(actual_pnl)


    # Metrics
    results = {
        "OLD (static)": compute_metrics(fc_old, actual_mfe_arr),
        "B (vol ratio)": compute_metrics(fc_B, actual_mfe_arr),
        "C (HMM blend)": compute_metrics(fc_C, actual_mfe_arr),
        "B+C (both)": compute_metrics(fc_BC, actual_mfe_arr),
    }

    labels = list(results.keys())
    print_comparison_table(labels, results)

    # PnL-weighted analysis: filter by forecast quartiles
    avg_all = float(np.mean(actual_pnl_arr))
    for _label, forecasts in [
        ("OLD (static)", fc_old),
        ("B (vol ratio)", fc_B),
        ("C (HMM blend)", fc_C),
        ("B+C (both)", fc_BC),
    ]:
        median_fc = np.median(forecasts)
        mask = forecasts > median_fc
        avg_filtered = float(np.mean(actual_pnl_arr[mask])) if mask.sum() > 0 else 0
        ((avg_filtered / avg_all) - 1) * 100 if avg_all != 0 else 0

    # Best method
    max(results.items(), key=lambda x: x[1].get("spearman_r") or -1)
    max(results.items(), key=lambda x: x[1].get("disc_ratio") or 0)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3: Enhancement A — EWMA Q→Runway calibration (online simulation)
# ══════════════════════════════════════════════════════════════════════════════


def test_ewma_calibration() -> None:
    print_section("TEST 3: EWMA Calibration — Online Learning Simulation")

    episodes: list[dict[str, Any]] = []
    with open(PROJECT_ROOT / "data" / "training_cache_XAUUSD_M5.jsonl") as f:
        episodes.extend(json.loads(line.strip()) for line in f)

    # Sort by timestamp to simulate chronological online learning
    episodes.sort(key=lambda e: e.get("ts_recorded", ""))

    # EWMA calibration state (mirrors trigger_agent.py)
    from src.agents.trigger_agent import (
        RUNWAY_CAL_ALPHA,
        RUNWAY_CAL_MIN_SAMPLES,
        RUNWAY_CAL_N_BUCKETS,
        RUNWAY_CAL_Q_EDGES,
    )

    ewma_values = [0.0] * RUNWAY_CAL_N_BUCKETS
    ewma_counts = [0] * RUNWAY_CAL_N_BUCKETS

    # Static Q→runway mapping (old)
    def static_q_to_runway(q: float) -> float:
        return 0.8 * max(0.0, q - 0.5)

    # Calibrated Q→runway with EWMA
    def calibrated_q_to_runway(q: float) -> float:
        bucket = 0
        for edge_idx in range(1, len(RUNWAY_CAL_Q_EDGES)):
            if q >= RUNWAY_CAL_Q_EDGES[edge_idx]:
                bucket = edge_idx - 1
            else:
                break
        bucket = min(bucket, RUNWAY_CAL_N_BUCKETS - 1)

        if ewma_counts[bucket] >= RUNWAY_CAL_MIN_SAMPLES:
            return ewma_values[bucket]
        return static_q_to_runway(q)

    # Update EWMA with actual outcome
    def update_ewma(q: float, actual_mfe_frac: float) -> None:
        bucket = 0
        for edge_idx in range(1, len(RUNWAY_CAL_Q_EDGES)):
            if q >= RUNWAY_CAL_Q_EDGES[edge_idx]:
                bucket = edge_idx - 1
            else:
                break
        bucket = min(bucket, RUNWAY_CAL_N_BUCKETS - 1)

        if ewma_counts[bucket] == 0:
            ewma_values[bucket] = actual_mfe_frac
        else:
            ewma_values[bucket] = RUNWAY_CAL_ALPHA * actual_mfe_frac + (1 - RUNWAY_CAL_ALPHA) * ewma_values[bucket]
        ewma_counts[bucket] += 1

    # Process episodes in order, tracking prediction error
    static_errors: list[float] = []
    calibrated_errors: list[float] = []
    warmup_static: list[float] = []
    warmup_calibrated: list[float] = []
    WARMUP_TRADES = 20

    for idx, ep in enumerate(episodes):
        entry_price = ep.get("entry_price", 0)
        mfe_abs = ep.get("mfe", 0)
        if entry_price <= 0 or mfe_abs is None:
            continue

        mfe_frac = abs(mfe_abs) / entry_price

        # Synthesise a Q-value from trade quality signals
        # Use capture_reward as a proxy for Q-value (it reflects
        # how much of the MFE was captured)
        capture_r = ep.get("capture_reward", 0) or 0
        trigger_r = ep.get("trigger_reward", 0) or 0
        # Scale to approximate Q-value range [0, 3]
        synthetic_q = max(0, min(3.0, 1.0 + capture_r + trigger_r * 0.5))

        # Predictions
        pred_static = static_q_to_runway(synthetic_q)
        pred_calibrated = calibrated_q_to_runway(synthetic_q)

        # Errors
        err_s = abs(pred_static - mfe_frac)
        err_c = abs(pred_calibrated - mfe_frac)

        if idx < WARMUP_TRADES:
            warmup_static.append(err_s)
            warmup_calibrated.append(err_c)
        else:
            static_errors.append(err_s)
            calibrated_errors.append(err_c)

        # Learn from outcome
        update_ewma(synthetic_q, mfe_frac)

    if len(static_errors) < 5:
        return

    static_errors = np.array(static_errors)
    calibrated_errors = np.array(calibrated_errors)


    mae_s = float(np.mean(static_errors))
    mae_c = float(np.mean(calibrated_errors))
    ((mae_c / mae_s) - 1) * 100 if mae_s > 0 else 0

    med_s = float(np.median(static_errors))
    med_c = float(np.median(calibrated_errors))
    ((med_c / med_s) - 1) * 100 if med_s > 0 else 0

    p90_s = float(np.percentile(static_errors, 90))
    p90_c = float(np.percentile(calibrated_errors, 90))
    ((p90_c / p90_s) - 1) * 100 if p90_s > 0 else 0

    # How many times calibrated was better?
    int(np.sum(calibrated_errors < static_errors))
    len(static_errors)

    if len(warmup_static) >= 3:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4: Combined Enhancement Impact — Entry/No-Entry Decision Quality
# ══════════════════════════════════════════════════════════════════════════════


def test_entry_decision_quality() -> None:
    print_section("TEST 4: Entry Decision Quality — Would Enhancements Filter Better?")

    episodes: list[dict[str, Any]] = []
    with open(PROJECT_ROOT / "data" / "training_cache_XAUUSD_M5.jsonl") as f:
        episodes.extend(json.loads(line.strip()) for line in f)

    usable = [ep for ep in episodes if len(ep.get("exit_bars", [])) >= 30]

    SHORT_VOL_WIN = 10
    LONG_VOL_WIN = 30

    forecasts_map: dict[str, list[float]] = {"OLD": [], "B": [], "C": [], "B+C": []}
    pnl_list: list[float] = []
    mfe_list: list[float] = []

    for ep in usable:
        bars = ep["exit_bars"]
        entry_price = ep.get("entry_price", 0)
        if entry_price <= 0:
            continue

        closes = [b[4] for b in bars]
        warmup_end = min(LONG_VOL_WIN, len(closes) - 1)
        if warmup_end < SHORT_VOL_WIN + 1:
            continue

        sigma_s = realized_vol(closes[: warmup_end + 1], SHORT_VOL_WIN)
        sigma_l = realized_vol(closes[: warmup_end + 1], min(LONG_VOL_WIN, warmup_end))
        if sigma_s <= 0:
            continue

        vr = RegimeDetector(window_size=20, update_interval=1, instrument_volatility=1.0)
        hmm = HMMRegimeDetector(window_size=20, update_interval=1, instrument_volatility=1.0)
        for j in range(warmup_end + 1):
            vr.add_price(closes[j])
            hmm.add_price(closes[j])

        r_old = old_runway(sigma_s)
        r_B = new_runway_volratio(sigma_s, sigma_l)
        disc_mult = vr.get_regime_multiplier()
        hmm_mult = hmm.get_blended_runway_multiplier()

        forecasts_map["OLD"].append(r_old * disc_mult)
        forecasts_map["B"].append(r_B * disc_mult)
        forecasts_map["C"].append(r_old * hmm_mult)
        forecasts_map["B+C"].append(r_B * hmm_mult)
        pnl_list.append(ep.get("pnl_pts", 0))
        mfe_list.append(abs(ep.get("mfe", 0)))

    if len(pnl_list) < 10:
        return

    pnl_arr = np.array(pnl_list)
    mfe_arr = np.array(mfe_list)


    # Test at different percentile thresholds
    for pct in [25, 50, 75]:
        for fc_list in forecasts_map.values():
            fc = np.array(fc_list)
            thresh = np.percentile(fc, pct)
            mask = fc >= thresh
            mask.sum()
            avg_pnl = float(np.mean(pnl_arr[mask]))
            float(np.mean(pnl_arr[mask] > 0)) * 100
            float(np.mean(mfe_arr[mask]))
            ((avg_pnl / np.mean(pnl_arr)) - 1) * 100 if np.mean(pnl_arr) != 0 else 0


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    # Suppress noisy regime detector INFO logging
    import logging

    logging.getLogger("src.features.regime_detector").setLevel(logging.WARNING)
    logging.getLogger("src.features.hmm_regime").setLevel(logging.WARNING)


    test_bars_cache()
    test_training_cache()
    test_ewma_calibration()
    test_entry_decision_quality()

    print_section("SUMMARY")


if __name__ == "__main__":
    main()
