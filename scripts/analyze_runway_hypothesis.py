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
from datetime import UTC, datetime
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
    df = df.rename(columns={"Date & Time": "ts", "Open": "o", "High": "h",
                        "Low": "l", "Close": "c", "Volume": "v"})
    df = df.sort_values("ts")
    df = df.reset_index(drop=True)
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
    return datetime.fromisoformat(s).astimezone(UTC)


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
    np.array(values)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    df = load_bars()
    df = add_atr(df)
    df = add_vol_regime(df)


    trades = load_m5_trades()

    tm = match_trades_to_bars(df, trades)

    # ── H1: Zero-MFE confirmation ─────────────────────────────────────────────

    tm[tm["mfe"] < MFE_ZERO_THR]
    tm[tm["mfe"] >= MFE_ZERO_THR]


    # MFE distribution percentiles
    for _p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        pass

    # ── H2: Swing distance vs MFE ─────────────────────────────────────────────

    dists = []
    for _, row in tm.iterrows():
        d = swing_distance_at(df, int(row["bar_idx"]), row["direction"])
        dists.append(d)
    tm["swing_dist"] = dists

    valid = tm[tm["swing_dist"].notna() & np.isfinite(tm["swing_dist"])]

    # Correlation
    valid[["swing_dist", "mfe"]].corr().iloc[0, 1]

    # Bucket by swing distance
    edges = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 999]
    labels = ["<0.5 ATR", "0.5–1.0", "1.0–1.5", "1.5–2.0", "2.0–3.0", ">3.0"]
    valid = valid.copy()
    valid["dist_bucket"] = pd.cut(valid["swing_dist"], bins=edges, labels=labels)

    for lbl in labels:
        grp = valid[valid["dist_bucket"] == lbl]
        if len(grp) == 0:
            continue
        100 * (grp["pnl"] > 0).mean()

    # ── H3: Vol regime vs MFE ─────────────────────────────────────────────────

    tm2 = tm.copy()
    tm2["vol_ratio"] = df.loc[tm2["bar_idx"].values, "vol_ratio"].values

    valid2 = tm2[tm2["vol_ratio"].notna() & np.isfinite(tm2["vol_ratio"])]
    valid2[["vol_ratio", "mfe"]].corr().iloc[0, 1]

    # Expanding vol (ratio > 1.2) vs contracting (< 0.8) vs neutral
    expand  = valid2[valid2["vol_ratio"] > 1.2]
    neutral = valid2[(valid2["vol_ratio"] >= 0.8) & (valid2["vol_ratio"] <= 1.2)]
    contract = valid2[valid2["vol_ratio"] < 0.8]

    for _name, grp in [("Expanding vol (>1.2)", expand),
                       ("Neutral   (0.8-1.2)", neutral),
                       ("Contracting (<0.8) ", contract)]:
        if len(grp) == 0:
            continue
        100 * (grp["pnl"] > 0).mean()

    # ── Combined gate simulation ──────────────────────────────────────────────

    both_valid = tm2[tm2["vol_ratio"].notna() & tm2["swing_dist"].notna()
                     & np.isfinite(tm2["vol_ratio"]) & np.isfinite(tm2["swing_dist"])].copy()

    gated_in  = both_valid[(both_valid["swing_dist"] > 1.0) &
                            (both_valid["vol_ratio"] > 0.6) &
                            (both_valid["vol_ratio"] < 1.4)]
    both_valid[~((both_valid["swing_dist"] > 1.0) &
                              (both_valid["vol_ratio"] > 0.6) &
                              (both_valid["vol_ratio"] < 1.4))]


    both_valid["pnl"].sum()
    gated_in["pnl"].sum()


if __name__ == "__main__":
    main()
