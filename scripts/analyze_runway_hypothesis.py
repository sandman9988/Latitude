#!/usr/bin/env python3
"""Empirical runway hypothesis test on XAUUSD M5 data.

Tests three hypotheses:
  H1 — Zero-MFE trades always lose (user's observation)
  H2 — Swing-level distance (ATR-normalized) predicts MFE magnitude
  H3 — Volatility regime (σ_short / σ_long) predicts MFE magnitude

Uses:
  data/history/XAUUSD_M5.csv  — 163K bars, Jan 2024–Apr 2026
  data/trade_log.jsonl         — 3 704 XAUUSD M5 closed trades
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
HIST_CSV   = Path("data/history/XAUUSD_M5.csv")
TRADE_LOG  = Path("data/trade_log.jsonl")
SWING_LOOKBACK  = 30   # bars back to find swing highs/lows
SWING_CONFIRM   = 2    # bars each side to confirm a swing pivot
ATR_PERIOD      = 14   # bars for ATR
VOL_SHORT_WIN   = 10   # bars for short-vol σ
VOL_LONG_WIN    = 50   # bars for long-vol σ (regime)
MFE_ZERO_THR    = 0.05 # points — below this = "zero MFE" on gold M5


# ── Load bars ─────────────────────────────────────────────────────────────────
def load_bars() -> pd.DataFrame:
    df = pd.read_csv(HIST_CSV, parse_dates=["Date & Time"])
    df.rename(columns={"Date & Time": "ts", "Open": "o", "High": "h",
                        "Low": "l", "Close": "c", "Volume": "v"}, inplace=True)
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["ts"] = df["ts"].dt.tz_localize("UTC")
    return df


# ── ATR ───────────────────────────────────────────────────────────────────────
def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    tr = pd.concat([
        df["h"] - df["l"],
        (df["h"] - df["c"].shift(1)).abs(),
        (df["l"] - df["c"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=period, adjust=False).mean()
    return df


# ── Volatility regime ─────────────────────────────────────────────────────────
def add_vol_regime(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["c"].pct_change()
    df["sigma_short"] = ret.rolling(VOL_SHORT_WIN).std()
    df["sigma_long"]  = ret.rolling(VOL_LONG_WIN).std()
    df["vol_ratio"]   = df["sigma_short"] / df["sigma_long"].replace(0, np.nan)
    return df


# ── Swing levels ──────────────────────────────────────────────────────────────
def swing_distance_at(df: pd.DataFrame, idx: int, direction: str) -> float:
    """ATR-normalized distance from entry bar to nearest opposing swing level.

    For LONG: distance to nearest swing high above current close.
    For SHORT: distance to nearest swing low below current close.
    Returns np.nan if ATR is zero or no swing found in lookback window.
    """
    if idx < SWING_CONFIRM * 2 + 1:
        return np.nan
    atr = df.at[idx, "atr"]
    if not np.isfinite(atr) or atr <= 0:
        return np.nan
    entry_price = df.at[idx, "c"]
    start = max(0, idx - SWING_LOOKBACK)
    end   = idx - 1  # don't include the entry bar itself

    swings = []
    for i in range(start + SWING_CONFIRM, end - SWING_CONFIRM + 1):
        if direction == "LONG":
            # Swing high: bar[i].high > all neighbours within SWING_CONFIRM
            is_swing = all(
                df.at[i, "h"] >= df.at[i - k, "h"] and
                df.at[i, "h"] >= df.at[i + k, "h"]
                for k in range(1, SWING_CONFIRM + 1)
            )
            if is_swing:
                level = df.at[i, "h"]
                if level > entry_price:
                    swings.append(level)
        else:
            # Swing low: bar[i].low < all neighbours
            is_swing = all(
                df.at[i, "l"] <= df.at[i - k, "l"] and
                df.at[i, "l"] <= df.at[i + k, "l"]
                for k in range(1, SWING_CONFIRM + 1)
            )
            if is_swing:
                level = df.at[i, "l"]
                if level < entry_price:
                    swings.append(level)

    if not swings:
        return np.nan

    if direction == "LONG":
        nearest = min(swings)          # closest swing high above
        dist = nearest - entry_price
    else:
        nearest = max(swings)          # closest swing low below
        dist = entry_price - nearest

    return dist / atr


# ── Load trades ───────────────────────────────────────────────────────────────
def load_m5_trades() -> list[dict]:
    trades = []
    with open(TRADE_LOG) as f:
        for line in f:
            t = json.loads(line)
            if t.get("symbol") == "XAUUSD" and t.get("timeframe") == "M5":
                trades.append(t)
    return trades


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(timezone.utc)


# ── Match trade → bar index ───────────────────────────────────────────────────
def match_trades_to_bars(df: pd.DataFrame, trades: list[dict]) -> pd.DataFrame:
    """Binary-search each trade's entry_time to the nearest bar index."""
    bar_ts = df["ts"].values  # numpy datetime64[ns, UTC]

    rows = []
    for t in trades:
        mfe = t.get("mfe_points", t.get("mfe", 0.0)) or 0.0
        mae = t.get("mae_points", t.get("mae", 0.0)) or 0.0
        pnl = t.get("pnl_points", t.get("pnl", 0.0)) or 0.0
        direction = t.get("direction", "LONG")

        entry_dt = parse_ts(t["entry_time"])
        entry_np = np.datetime64(entry_dt.replace(tzinfo=None), "ns")
        idx = int(np.searchsorted(bar_ts, entry_np, side="left"))
        idx = max(0, min(idx, len(df) - 1))

        rows.append({
            "bar_idx":   idx,
            "direction": direction,
            "mfe":       float(mfe),
            "mae":       float(mae),
            "pnl":       float(pnl),
        })

    return pd.DataFrame(rows)


# ── Reporting helpers ─────────────────────────────────────────────────────────
def pct_str(n: int, total: int) -> str:
    return f"{n}/{total} ({100*n/total:.1f}%)" if total else "0/0"


def bucket_stats(values: list[float], label: str) -> None:
    arr = np.array(values)
    print(f"  {label}: n={len(arr)}  mean={arr.mean():.3f}  "
          f"median={np.median(arr):.3f}  p25={np.percentile(arr,25):.3f}  "
          f"p75={np.percentile(arr,75):.3f}  win%={100*(arr>0).mean():.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading bars …")
    df = load_bars()
    df = add_atr(df)
    df = add_vol_regime(df)

    print(f"Bars loaded: {len(df)}  ({df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()})")

    print("Loading trades …")
    trades = load_m5_trades()
    print(f"XAUUSD M5 trades: {len(trades)}")

    tm = match_trades_to_bars(df, trades)

    # ── H1: Zero-MFE confirmation ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("H1 — Do zero-MFE trades always lose?")
    print("="*70)

    zero_mfe = tm[tm["mfe"] < MFE_ZERO_THR]
    nonzero_mfe = tm[tm["mfe"] >= MFE_ZERO_THR]

    print(f"Zero-MFE  (<{MFE_ZERO_THR} pts): {pct_str(len(zero_mfe), len(tm))}")
    print(f"  Winners (pnl>0): {pct_str((zero_mfe['pnl']>0).sum(), len(zero_mfe))}")
    print(f"  Losers  (pnl<0): {pct_str((zero_mfe['pnl']<0).sum(), len(zero_mfe))}")
    print(f"  Mean PnL: {zero_mfe['pnl'].mean():.4f} pts")
    print()
    print(f"Non-zero MFE (>={MFE_ZERO_THR} pts): {pct_str(len(nonzero_mfe), len(tm))}")
    print(f"  Winners (pnl>0): {pct_str((nonzero_mfe['pnl']>0).sum(), len(nonzero_mfe))}")
    print(f"  Mean PnL: {nonzero_mfe['pnl'].mean():.4f} pts")

    # MFE distribution percentiles
    print(f"\nMFE distribution (all trades):")
    for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        print(f"  p{p:3d}: {np.percentile(tm['mfe'], p):.3f} pts")

    # ── H2: Swing distance vs MFE ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("H2 — Does swing-level distance (ATR-normalized) predict MFE?")
    print("="*70)
    print("Computing swing distances … (may take ~60s for 3K trades)")

    dists = []
    for _, row in tm.iterrows():
        d = swing_distance_at(df, int(row["bar_idx"]), row["direction"])
        dists.append(d)
    tm["swing_dist"] = dists

    valid = tm[tm["swing_dist"].notna() & np.isfinite(tm["swing_dist"])]
    print(f"Trades with valid swing distance: {len(valid)}/{len(tm)}")

    # Correlation
    corr = valid[["swing_dist", "mfe"]].corr().iloc[0, 1]
    print(f"Pearson correlation(swing_dist, MFE): {corr:.4f}")

    # Bucket by swing distance
    edges = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 999]
    labels = ["<0.5 ATR", "0.5–1.0", "1.0–1.5", "1.5–2.0", "2.0–3.0", ">3.0"]
    valid = valid.copy()
    valid["dist_bucket"] = pd.cut(valid["swing_dist"], bins=edges, labels=labels)

    print("\nMFE and win-rate by distance to nearest swing level:")
    print(f"  {'Bucket':<12}  {'N':>5}  {'MFE mean':>10}  {'MFE p50':>10}  {'Win%':>7}  {'PnL mean':>10}")
    for lbl in labels:
        grp = valid[valid["dist_bucket"] == lbl]
        if len(grp) == 0:
            continue
        win_pct = 100 * (grp["pnl"] > 0).mean()
        print(f"  {lbl:<12}  {len(grp):>5}  {grp['mfe'].mean():>10.3f}  "
              f"{grp['mfe'].median():>10.3f}  {win_pct:>6.1f}%  {grp['pnl'].mean():>10.4f}")

    # ── H3: Vol regime vs MFE ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("H3 — Does volatility regime (σ_short / σ_long) predict MFE?")
    print("="*70)

    tm2 = tm.copy()
    tm2["vol_ratio"] = df.loc[tm2["bar_idx"].values, "vol_ratio"].values

    valid2 = tm2[tm2["vol_ratio"].notna() & np.isfinite(tm2["vol_ratio"])]
    corr2 = valid2[["vol_ratio", "mfe"]].corr().iloc[0, 1]
    print(f"Pearson correlation(vol_ratio, MFE): {corr2:.4f}")

    # Expanding vol (ratio > 1.2) vs contracting (< 0.8) vs neutral
    expand  = valid2[valid2["vol_ratio"] > 1.2]
    neutral = valid2[(valid2["vol_ratio"] >= 0.8) & (valid2["vol_ratio"] <= 1.2)]
    contract = valid2[valid2["vol_ratio"] < 0.8]

    for name, grp in [("Expanding vol (>1.2)", expand),
                       ("Neutral   (0.8-1.2)", neutral),
                       ("Contracting (<0.8) ", contract)]:
        if len(grp) == 0:
            continue
        win_pct = 100 * (grp["pnl"] > 0).mean()
        print(f"  {name}: n={len(grp):>4}  MFE mean={grp['mfe'].mean():.3f}  "
              f"MFE p50={grp['mfe'].median():.3f}  win%={win_pct:.1f}%  "
              f"pnl={grp['pnl'].mean():.4f}")

    # ── Combined gate simulation ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("Combined gate simulation: swing_dist>1.0 ATR AND vol_ratio in [0.6, 1.4]")
    print("="*70)

    both_valid = tm2[tm2["vol_ratio"].notna() & tm2["swing_dist"].notna()
                     & np.isfinite(tm2["vol_ratio"]) & np.isfinite(tm2["swing_dist"])].copy()

    gated_in  = both_valid[(both_valid["swing_dist"] > 1.0) &
                            (both_valid["vol_ratio"] > 0.6) &
                            (both_valid["vol_ratio"] < 1.4)]
    gated_out = both_valid[~((both_valid["swing_dist"] > 1.0) &
                              (both_valid["vol_ratio"] > 0.6) &
                              (both_valid["vol_ratio"] < 1.4))]

    print(f"Would PASS gate: {pct_str(len(gated_in), len(both_valid))}")
    print(f"  MFE mean={gated_in['mfe'].mean():.3f}  p50={gated_in['mfe'].median():.3f}  "
          f"win%={100*(gated_in['pnl']>0).mean():.1f}%  pnl_mean={gated_in['pnl'].mean():.4f}")
    print(f"Would BLOCK gate: {pct_str(len(gated_out), len(both_valid))}")
    print(f"  MFE mean={gated_out['mfe'].mean():.3f}  p50={gated_out['mfe'].median():.3f}  "
          f"win%={100*(gated_out['pnl']>0).mean():.1f}%  pnl_mean={gated_out['pnl'].mean():.4f}")

    total_pnl_all   = both_valid["pnl"].sum()
    total_pnl_gated = gated_in["pnl"].sum()
    print(f"\nTotal PnL (all):   {total_pnl_all:.2f} pts")
    print(f"Total PnL (gated): {total_pnl_gated:.2f} pts  ({100*total_pnl_gated/total_pnl_all:.1f}% of all on {100*len(gated_in)/len(both_valid):.1f}% of trades)")


if __name__ == "__main__":
    main()
