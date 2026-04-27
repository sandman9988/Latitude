#!/usr/bin/env python3
"""Reconstruct the complete trade lifecycle by stitching 5 data sources.

Sources (in priority order for each field):
  1. trade_log.jsonl          — final trade records (P&L, MFE/MAE, close_reason)
  2. training_cache_*.jsonl   — bar-level snapshots at entry/exit
  3. logs/audit/decisions.jsonl — per-bar agent decisions (trigger + harvester)
  4. logs/audit/transactions.jsonl — transaction events (ORDER_SUBMIT, POSITION_CLOSE)
  5. data/history/*.csv       — raw OHLCV price history for context

Usage:
    python3 scripts/reconstruct_trade_lifecycle.py [--symbol XAUUSD] [--tf 5]
                                                   [--month 2026-04]
                                                   [--out data/analysis_lifecycle.jsonl]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """A single trade with all available fields stitched from every source."""
    trade_id: int | None = None
    ticket: str | None = None
    position_id: str | None = None
    symbol: str = ""
    timeframe: str = ""
    timeframe_minutes: int = 0
    direction: str = ""
    quantity: float = 0.0

    # Prices & P&L (source: trade_log)
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_points: float = 0.0
    pnl_net_points: float = 0.0
    spread_cost_points: float = 0.0

    # Excursions (source: trade_log + cache)
    mfe: float = 0.0
    mae: float = 0.0
    mfe_points: float = 0.0
    mae_points: float = 0.0

    # Timing (source: trade_log + decisions)
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    hold_seconds: float = 0.0
    ticks_held: int = 0
    bars_held: int = 0

    # Entry conditions (source: trade_log + decisions)
    entry_confidence: float = 0.0
    entry_vpin_z: float = 0.0
    entry_var_95: float = 0.0
    entry_imbalance: float = 0.0
    entry_dynamic_floor: float = 0.0
    entry_conf_margin: float = 0.0
    regime: str = ""
    exit_regime: str = ""
    exit_vol: float = 0.0
    exit_depth_ratio: float = 0.0
    trigger_reward: float = 0.0
    capture_reward: float = 0.0

    # Runway (source: trade_log)
    predicted_runway_gross: float = 0.0
    predicted_runway_net: float = 0.0
    predicted_runway_gross_points: float = 0.0
    predicted_runway_net_points: float = 0.0
    runway_utilization: float = 0.0
    runway_delta_ema: float = 0.0
    runway_accuracy_ema: float = 0.0

    # Entry calibration (source: trade_log + decisions)
    win_rate_ema_at_entry: float = 0.0
    total_trades_at_entry: int = 0
    equity_at_entry: float = 0.0
    conf_calib_err_at_entry: float = 0.0
    runway_accuracy_at_entry: float = 0.0
    balance_after: float = 0.0

    # Quality flags (source: trade_log + cache)
    winner_to_loser: bool = False
    close_reason: str = ""
    capture_ratio: float = 0.0
    harvester_quality: str = ""

    # Diagnostics (source: trade_log)
    diag_cb_active: bool = False
    diag_cb_tripped: list[str] = field(default_factory=list)

    # Correlation keys (source: all)
    decision_trade_id: str | None = None
    session: str | None = None

    # Bar-level data from cache (source: training_cache)
    entry_bars: list[list] = field(default_factory=list)  # [[iso,o,h,l,c], ...]
    exit_bars: list[list] = field(default_factory=list)

    # Linked decisions (source: decisions.jsonl)
    trigger_entries: list[dict] = field(default_factory=list)  # LONG/SHORT/NO_ENTRY
    hold_decisions: list[dict] = field(default_factory=list)
    close_decision: dict | None = None

    # Price context from CSV (source: history/*.csv)
    csv_bars_around_entry: list[list] = field(default_factory=list)
    csv_bars_around_exit: list[list] = field(default_factory=list)

    # Data quality
    source_flags: dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def load_csv_bars(path: Path) -> list[list]:
    """Load CSV history bars as [timestamp, o, h, l, c, v]."""
    bars: list[list] = []
    if not path.exists():
        LOG.warning("CSV not found: %s", path)
        return bars
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("Date & Time", "")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                o = float(row["Open"])
                h = float(row["High"])
                l = float(row["Low"])
                c = float(row["Close"])
                v = float(row.get("Volume", 0))
                bars.append([ts, o, h, l, c, v])
            except (ValueError, KeyError):
                continue
    LOG.info("Loaded %d CSV bars from %s", len(bars), path)
    return bars


def load_trade_log(path: Path, symbol: str, timeframe_minutes: int, month: str) -> list[dict]:
    """Load trade_log entries for a specific symbol/TF/month."""
    trades: list[dict] = []
    if not path.exists():
        LOG.warning("trade_log not found: %s", path)
        return trades
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = r.get("exit_time", "") or r.get("entry_time", "") or ""
        if month in ts and r.get("symbol") == symbol and r.get("timeframe_minutes") == timeframe_minutes:
            trades.append(r)
    LOG.info("Loaded %d trade_log entries for %s M%d %s", len(trades), symbol, timeframe_minutes, month)
    return trades


def load_cache(path: Path, month: str) -> list[dict]:
    """Load training cache entries (streaming with binary month filter)."""
    records: list[dict] = []
    if not path.exists():
        LOG.warning("Cache not found: %s", path)
        return records
    month_bytes = month.encode()
    with open(path, "rb") as f:
        for raw_line in f:
            if month_bytes not in raw_line:
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(r)
    LOG.info("Loaded %d cache records from %s", len(records), path.name)
    return records


def load_decisions(path: Path, month: str) -> list[dict]:
    """Load decision log entries (streaming with binary month filter)."""
    entries: list[dict] = []
    if not path.exists():
        LOG.warning("Decisions not found: %s", path)
        return entries
    month_bytes = month.encode()
    with open(path, "rb") as f:
        for raw_line in f:
            if month_bytes not in raw_line:
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(r)
    LOG.info("Loaded %d decision entries", len(entries))
    return entries


def load_transactions(path: Path, month: str) -> list[dict]:
    """Load transaction log POSITION_CLOSE events (streaming, memory-efficient).

    Only loads the month's POSITION_CLOSE events from the 33MB file.
    """
    events: list[dict] = []
    if not path.exists():
        LOG.warning("Transactions not found: %s", path)
        return events
    month_bytes = month.encode()
    with open(path, "rb") as f:
        for raw_line in f:
            if month_bytes not in raw_line:
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event_type") == "POSITION_CLOSE":
                events.append(r)
    LOG.info("Loaded %d POSITION_CLOSE events from %s", len(events), path)
    return events


# ---------------------------------------------------------------------------
# Stitching helpers
# ---------------------------------------------------------------------------

def _ts_parse(s: str) -> datetime | None:
    """Parse an ISO timestamp string to aware datetime."""
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _ts_key(dt: datetime | None) -> float:
    """Return float seconds for sorting/grouping."""
    if dt is None:
        return 0.0
    return dt.timestamp()


def _price_close(prices: list[float], target: float, _tolerance: float = 5.0) -> float:
    """Return min absolute price diff or big number if no close match."""
    return min((abs(p - target) for p in prices), default=float("inf"))


def _match_by_timestamp(
    target_ts: datetime | None,
    candidates: list[dict],
    ts_field: str,
    max_seconds: float = 5.0,
) -> dict | None:
    """Find the candidate with closest timestamp within max_seconds."""
    if target_ts is None or not candidates:
        return None
    target_secs = _ts_key(target_ts)
    for c in sorted(candidates, key=lambda x: abs(_ts_key(_ts_parse(x.get(ts_field, ""))) - target_secs)):
        diff = abs(_ts_key(_ts_parse(c.get(ts_field, ""))) - target_secs)
        if diff <= max_seconds:
            return c
    return None


def _build_bollinger(bars: list[list], lookback: int = 20) -> dict:
    """Compute Bollinger band stats from a bar list [ts, o, h, l, c, ...]."""
    if len(bars) < lookback:
        return {"bb_upper": 0, "bb_lower": 0, "bb_width": 0, "zscore": 0}
    closes = [b[4] for b in bars[-lookback:]]
    mu = sum(closes) / len(closes)
    var = sum((c - mu) ** 2 for c in closes) / len(closes)
    std = math.sqrt(var) if var > 0 else 0.0
    latest = closes[-1]
    return {
        "bb_upper": mu + 2 * std,
        "bb_lower": mu - 2 * std,
        "bb_width": (mu + 2 * std - max(mu - 2 * std, 0)) / max(mu, 1),
        "zscore": (latest - mu) / std if std > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Main reconstruction
# ---------------------------------------------------------------------------

def reconstruct(
    symbol: str = "XAUUSD",
    timeframe_minutes: int = 5,
    month: str = "2026-04",
    data_root: str = "data",
    log_root: str = "logs/audit",
    history_root: str = "data/history",
) -> list[dict]:
    """Run the full multi-source reconstruction."""
    root = Path(data_root)
    lr = Path(log_root)
    hr = Path(history_root)

    # 1. Load all sources
    tf_label = f"M{timeframe_minutes}"
    csv_path = hr / f"{symbol}_{tf_label}.csv"
    cache_path = root / f"training_cache_{symbol}_{tf_label}.jsonl"
    trade_log_path = root / "trade_log.jsonl"
    decisions_path = lr / "decisions.jsonl"
    transactions_path = lr / "transactions.jsonl"

    csv_bars = load_csv_bars(csv_path)
    cache_records = load_cache(cache_path, month)
    trade_log_entries = load_trade_log(trade_log_path, symbol, timeframe_minutes, month)
    decisions = load_decisions(decisions_path, month)
    transactions = load_transactions(transactions_path, month)

    if not trade_log_entries:
        LOG.error("No trade_log entries found for %s M%d %s", symbol, timeframe_minutes, month)
        return []

    # 2. Build indices
    # Cache index: bucket by rounded entry_price for O(1) lookup
    # Each bucket key = round(entry_price * 100) → all records within ~0.01 price
    cache_by_bucket: dict[int, list[tuple[float, float, dict]]] = defaultdict(list)
    for c in cache_records:
        ep = c.get("entry_price", 0.0) or 0.0
        xp = c.get("exit_price", 0.0) or 0.0
        if ep > 0 and xp > 0:
            bucket = int(round(ep * 100))
            cache_by_bucket[bucket].append((ep, xp, c))

    # CSV time index for O(1) context lookup: bucket timestamp to nearest M5 boundary
    csv_by_bucket: dict[int, list[list]] = defaultdict(list)
    for b in csv_bars:
        ts_s = int(round(_ts_key(b[0])))
        bucket = (ts_s // 300) * 300  # snap to M5: 0, 300, 600, ...
        csv_by_bucket[bucket].append(b)

    # Decision indices
    trig_by_second: dict[int, list[dict]] = defaultdict(list)
    close_by_second: dict[int, list[dict]] = defaultdict(list)
    hold_by_tid: dict[str, list[dict]] = defaultdict(list)

    for d in decisions:
        ts_k = int(round(_ts_key(_ts_parse(d.get("timestamp", "")))))
        ag = d.get("agent", "")
        dec = d.get("decision", "")
        tid = d.get("trade_id")
        if ag == "TriggerAgent" and dec in ("LONG", "SHORT"):
            trig_by_second[ts_k].append(d)
        elif ag == "HarvesterAgent" and dec == "CLOSE":
            close_by_second[ts_k].append(d)
        elif ag == "HarvesterAgent" and dec == "HOLD" and tid:
            hold_by_tid[tid].append(d)

    # Transaction index by timestamp (rounded to nearest second for fast lookup)
    txn_by_second: dict[int, list[dict]] = defaultdict(list)
    for t in transactions:
        ts_k = int(round(_ts_key(_ts_parse(t.get("timestamp", "")))))
        txn_by_second[ts_k].append(t)

    # 3. Stitch each trade_log entry
    stitched: list[dict] = []
    matched_count = 0

    for t in trade_log_entries:
        ep = float(t.get("entry_price", 0.0) or 0.0)
        xp = float(t.get("exit_price", 0.0) or 0.0)
        et = _ts_parse(t.get("entry_time", "") or "")
        xt = _ts_parse(t.get("exit_time", "") or "")
        tl_pnl = float(t.get("pnl", 0) or 0)
        dtid = t.get("decision_trade_id")

        # --- Step A: Find closest CLOSE decision by timestamp (bucketed) ---
        close_decision = None
        if xt:
            xt_k = int(round(_ts_key(xt)))
            for offset in range(-10, 11):
                candidates = close_by_second.get(xt_k + offset, [])
                if candidates:
                    close_decision = candidates[0]
                    break

        # --- Step B: Find trade_id from close decision ---
        trade_id = dtid
        if trade_id is None and close_decision:
            trade_id = close_decision.get("trade_id")

        # --- Step C: Find matching cache record by price bucket ---
        cache_record = None
        if ep > 0:
            bucket = int(round(ep * 100))
            candidates = cache_by_bucket.get(bucket, [])
            # Also check adjacent buckets for price slip
            for adj in (bucket - 1, bucket, bucket + 1):
                candidates.extend(cache_by_bucket.get(adj, []))
            best_cache_dist = float("inf")
            for cep, cxp, cr in candidates:
                dist = abs(cep - ep) + abs(cxp - xp)
                if dist < best_cache_dist:
                    best_cache_dist = dist
                    cache_record = cr
            # Accept cache match only if price distance is reasonable (<= 3 * spread)
            if best_cache_dist > 1.0:
                cache_record = None

        # --- Step D: Find preceding trigger decisions (bucketed timestamp) ---
        trig_entries = []
        if et:
            et_k = int(round(_ts_key(et)))
            for offset in range(-120, 1):  # within 2 min before entry
                candidates = trig_by_second.get(et_k + offset, [])
                trig_entries.extend(candidates)

        # --- Step E: Find matching transaction (bucketed timestamp lookup) ---
        matching_txn = None
        if xt:
            xt_k = int(round(_ts_key(xt)))
            for offset in range(-5, 6):
                candidates = txn_by_second.get(xt_k + offset, [])
                if candidates:
                    matching_txn = candidates[0]
                    break

        # --- Step F: CSV context around entry/exit (M5-bucketed lookup) ---
        csv_entry_ctx: list[list] = []
        csv_exit_ctx: list[list] = []
        if csv_bars:
            if et:
                et_s = int(round(_ts_key(et)))
                et_bucket = (et_s // 300) * 300
                for offset_b in range(-2, 3):  # ±2 M5 bars = ±10 min
                    csv_entry_ctx.extend(csv_by_bucket.get(et_bucket + offset_b * 300, []))
            if xt:
                xt_s = int(round(_ts_key(xt)))
                xt_bucket = (xt_s // 300) * 300
                for offset_b in range(-2, 3):
                    csv_exit_ctx.extend(csv_by_bucket.get(xt_bucket + offset_b * 300, []))

        # --- Step G: Bollinger context at exit ---
        bb = {}
        if csv_exit_ctx:
            bb = _build_bollinger(csv_exit_ctx)

        # --- Step H: Harvester quality from legacy field ---
        cap_ratio = float(t.get("capture_ratio", 0) or 0)
        mfe_val = float(t.get("mfe_points", 0) or 0)
        pnl_pts = float(t.get("pnl_points", 0) or 0)
        if mfe_val > 0:
            derived_cap = pnl_pts / mfe_val if mfe_val > 0 else 0.0
        else:
            derived_cap = 0.0

        has_wtl = t.get("winner_to_loser", False)
        if has_wtl:
            harv_qual = "POOR_WTL"
        elif derived_cap >= 0.85:
            harv_qual = "EXCELLENT"
        elif derived_cap >= 0.65:
            harv_qual = "GOOD"
        elif derived_cap >= 0.4:
            harv_qual = "FAIR"
        elif mfe_val > 0 and pnl_pts <= 0:
            harv_qual = "LOSS"
        else:
            harv_qual = "UNKNOWN"

        # --- Assemble ---
        record = {
            # Identity
            "trade_id": t.get("trade_id"),
            "ticket": t.get("ticket"),
            "position_id": t.get("position_id"),
            "symbol": t.get("symbol", symbol),
            "timeframe": t.get("timeframe", tf_label),
            "timeframe_minutes": t.get("timeframe_minutes", timeframe_minutes),
            "direction": t.get("direction", ""),
            "quantity": float(t.get("quantity", 0.01) or 0.01),

            # P&L
            "entry_price": ep,
            "exit_price": xp,
            "pnl": tl_pnl,
            "pnl_points": float(t.get("pnl_points", 0) or 0),
            "pnl_net_points": float(t.get("pnl_net_points", 0) or 0),
            "spread_cost_points": float(t.get("spread_cost_points", 0) or 0),

            # Excursions
            "mfe_points": float(t.get("mfe_points", 0) or 0),
            "mae_points": float(t.get("mae_points", 0) or 0),
            "mfe": float(t.get("mfe", 0) or 0),
            "mae": float(t.get("mae", 0) or 0),

            # Timing
            "entry_time": t.get("entry_time"),
            "exit_time": t.get("exit_time"),
            "hold_seconds": float(t.get("hold_seconds", 0) or 0),
            "ticks_held": int(t.get("ticks_held", 0) or 0),
            "bars_held": int(t.get("bars_held", 0) or 0),

            # Entry conditions
            "entry_confidence": float(t.get("entry_confidence", 0) or 0),
            "entry_vpin_z": float(t.get("entry_vpin_z", 0) or 0),
            "entry_var_95": float(t.get("entry_var_95", 0) or 0),
            "entry_imbalance": float(t.get("entry_imbalance", 0) or 0),
            "entry_dynamic_floor": float(t.get("entry_dynamic_floor", 0) or 0),
            "entry_conf_margin": float(t.get("entry_conf_margin", 0) or 0),
            "regime": t.get("regime", ""),
            "exit_regime": t.get("exit_regime", ""),
            "exit_vol": float(t.get("exit_vol", 0) or 0),
            "exit_depth_ratio": float(t.get("exit_depth_ratio", 0) or 0),
            "trigger_reward": float(t.get("trigger_reward", 0) or 0),
            "capture_reward": float(t.get("capture_reward", 0) or 0),

            # Runway
            "predicted_runway_gross": float(t.get("predicted_runway_gross", 0) or 0),
            "predicted_runway_net": float(t.get("predicted_runway_net", 0) or 0),
            "predicted_runway_gross_points": float(t.get("predicted_runway_gross_points", 0) or 0),
            "predicted_runway_net_points": float(t.get("predicted_runway_net_points", 0) or 0),
            "runway_utilization": float(t.get("runway_utilization", 0) or 0),
            "runway_delta_ema": float(t.get("runway_delta_ema", 0) or 0),
            "runway_accuracy_ema": float(t.get("runway_accuracy_ema", 0) or 0),

            # Calibration
            "win_rate_ema_at_entry": float(t.get("win_rate_ema_at_entry", 0) or 0),
            "total_trades_at_entry": int(t.get("total_trades_at_entry", 0) or 0),
            "equity_at_entry": float(t.get("equity_at_entry", 0) or 0),
            "conf_calib_err_at_entry": float(t.get("conf_calib_err_at_entry", 0) or 0),
            "runway_accuracy_at_entry": float(t.get("runway_accuracy_at_entry", 0) or 0),
            "balance_after": float(t.get("balance_after", 0) or 0),

            # Quality
            "winner_to_loser": bool(t.get("winner_to_loser", False)),
            "close_reason": t.get("close_reason", ""),
            "capture_ratio": cap_ratio,
            "derived_capture_ratio": round(derived_cap, 4),
            "harvester_quality": harv_qual,

            # Diagnostics
            "diag_cb_active": bool(t.get("diag_circuit_breaker_active", False)),
            "diag_cb_tripped": t.get("diag_circuit_breakers_tripped", []),

            # Correlation
            "decision_trade_id": trade_id,
            "has_decision_link": trade_id is not None,
            "session": (close_decision.get("session") if close_decision else None)
                       or (t.get("session")),

            # Linked decisions
            "num_trigger_entries_before": len(trig_entries),
            "num_hold_decisions": len(hold_by_tid.get(trade_id or "", [])),
            "trigger_confidence": [d.get("confidence", 0) for d in trig_entries],
            "trigger_regime": [d.get("context", {}).get("regime", "")
                               for d in trig_entries if d.get("context")],
            "close_decision_conf": close_decision.get("confidence") if close_decision else None,
            "close_decision_reasoning": (close_decision.get("reasoning") if close_decision else None),

            # Transaction data
            "txn_pnl": (matching_txn.get("data", {}).get("pnl") if matching_txn else None),
            "txn_mfe": (matching_txn.get("data", {}).get("mfe") if matching_txn else None),
            "txn_mae": (matching_txn.get("data", {}).get("mae") if matching_txn else None),

            # Cache data
            "cache_mfe": (cache_record.get("mfe") if cache_record else None),
            "cache_mae": (cache_record.get("mae") if cache_record else None),
            "cache_capture_reward": (cache_record.get("capture_reward") if cache_record else None),
            "cache_trigger_action": (cache_record.get("trigger_action") if cache_record else None),
            "cache_regime": (cache_record.get("regime") if cache_record else None),
            "cache_num_entry_bars": (len(cache_record.get("entry_bars", [])) if cache_record else 0),
            "cache_num_exit_bars": (len(cache_record.get("exit_bars", [])) if cache_record else 0),

            # Bollinger context at exit
            "bb_upper": round(bb.get("bb_upper", 0), 2),
            "bb_lower": round(bb.get("bb_lower", 0), 2),
            "bb_width": round(bb.get("bb_width", 0), 4),
            "bb_zscore": round(bb.get("zscore", 0), 4),

            # CSV price context count
            "csv_bars_around_entry": len(csv_entry_ctx),
            "csv_bars_around_exit": len(csv_exit_ctx),

            # Data quality flags
            "source_trade_log": True,
            "source_cache": cache_record is not None,
            "source_decisions": close_decision is not None,
            "source_transactions": matching_txn is not None,
            "source_csv": len(csv_exit_ctx) > 0,
        }

        stitched.append(record)
        if close_decision:
            matched_count += 1

    # Stats
    total = len(stitched)
    matched_pct = matched_count / total * 100 if total > 0 else 0
    cache_pct = sum(1 for r in stitched if r["source_cache"]) / total * 100 if total > 0 else 0
    txn_pct = sum(1 for r in stitched if r["source_transactions"]) / total * 100 if total > 0 else 0
    csv_pct = sum(1 for r in stitched if r["source_csv"]) / total * 100 if total > 0 else 0

    qual_map = defaultdict(int)
    for r in stitched:
        qual_map[r["harvester_quality"]] += 1

    LOG.info("")
    LOG.info("=== RECONSTRUCTION RESULTS ===")
    LOG.info("Total trades: %d", total)
    LOG.info("Linked to decisions: %d (%.1f%%)", matched_count, matched_pct)
    LOG.info("Linked to cache: %d (%.1f%%)", sum(1 for r in stitched if r["source_cache"]), cache_pct)
    LOG.info("Linked to transactions: %d (%.1f%%)", sum(1 for r in stitched if r["source_transactions"]), txn_pct)
    LOG.info("Linked to CSV context: %d (%.1f%%)", sum(1 for r in stitched if r["source_csv"]), csv_pct)
    LOG.info("Harvester quality distribution:")
    for k, v in sorted(qual_map.items(), key=lambda x: -x[1]):
        LOG.info("  %s: %d (%.1f%%)", k, v, v / total * 100)

    return stitched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct trade lifecycle from all sources")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol (default: XAUUSD)")
    parser.add_argument("--tf", type=int, default=5, help="Timeframe minutes (default: 5)")
    parser.add_argument("--month", default="2026-04", help="Month filter (default: 2026-04)")
    parser.add_argument("--out", default="data/analysis_lifecycle.jsonl",
                        help="Output path (default: data/analysis_lifecycle.jsonl)")
    parser.add_argument("--data-root", default="data", help="Data directory root")
    parser.add_argument("--log-root", default="logs/audit", help="Audit log directory")
    parser.add_argument("--history-root", default="data/history", help="History CSV directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results = reconstruct(
        symbol=args.symbol,
        timeframe_minutes=args.tf,
        month=args.month,
        data_root=args.data_root,
        log_root=args.log_root,
        history_root=args.history_root,
    )

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    LOG.info("Written %d records to %s", len(results), out_path)


if __name__ == "__main__":
    main()
