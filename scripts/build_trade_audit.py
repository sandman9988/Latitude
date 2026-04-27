#!/usr/bin/env python3
"""Build auditable per-trade lifecycle records from existing logs.

Reads from (per paper_SYMBOL_TF directory):
  - trade_log.jsonl            — completed trade outcomes
  - logs/audit/decisions.jsonl — trigger + harvester decisions (trade_id correlated)
  - logs/audit/transactions.jsonl — order submit / position close events
  - decision_log_*.json        — rolling bar-by-bar state snapshots (recent only)
  - data/history/SYMBOL_M*.csv — downloaded OHLCV history for bar-by-bar gap-fill

Global reconstruction mode (--reconstruct):
  - Reads the global data/trade_log.jsonl (routes by symbol+TF)
  - Correlates data/training_cache_SYMBOL_M*.jsonl for trades lacking decision log coverage
    (training cache holds: regime, imbalance, vpin_z, depth_ratio, mfe, mae, entry/exit bars)
  - Falls back to data/logs/audit/decisions.jsonl for the overlap window
  - Outputs data/reconstructed_audit_SYMBOL.jsonl (one record per trade, all TFs merged)

Writes:
  - logs/audit/trade_lifecycle.jsonl — one record per completed trade, full lifecycle

Correlation:
  - trade_log entry  ↔  TriggerAgent decision: entry_time ≈ decision.timestamp (within 5 min)
  - harvester decisions: matched by trade_id (hex) from the trigger decision
  - training cache: matched by entry_price proximity (≤0.5 pts) + timestamp (≤tf_minutes*60s)
  - CSV bars: OHLCV bars during trade window for bar-by-bar price context

Usage:
  python3 scripts/build_trade_audit.py [--data-dir data] [--symbol XAUUSD] [--tf M15] [--all]
  python3 scripts/build_trade_audit.py --reconstruct --symbol XAUUSD [--from-date 2026-03-01]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRADE_LOG_FILENAME = "trade_log.jsonl"
_GLOBAL_TRADE_LOG = "trade_log.jsonl"
_TRAINING_CACHE_PRICE_TOL = 1.0   # pts — max entry_price delta for cache correlation
_TRAINING_CACHE_TIME_TOL_MULT = 3  # × tf_minutes seconds allowed between cache ts and entry_time

# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _abs_delta(a: datetime | None, b: datetime | None) -> float:
    if a is None or b is None:
        return float("inf")
    a = a.astimezone(timezone.utc) if a.tzinfo else a.replace(tzinfo=timezone.utc)
    b = b.astimezone(timezone.utc) if b.tzinfo else b.replace(tzinfo=timezone.utc)
    return abs((a - b).total_seconds())


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _nearest_txn(pool: list[dict], ref_ts: datetime | None, max_delta_s: float = 300.0) -> dict | None:
    if ref_ts is None:
        return None
    best, best_d = None, float("inf")
    for t in pool:
        d = _abs_delta(_parse_ts(t.get("timestamp")), ref_ts)
        if d < best_d and d <= max_delta_s:
            best, best_d = t, d
    return best


# ── CSV history loader ─────────────────────────────────────────────────────────

def _parse_csv_ts(ts_raw: str) -> datetime | None:
    if not ts_raw:
        return None
    try:
        if ts_raw.isdigit():
            return datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
        dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


def _parse_csv_row(row: dict) -> dict | None:
    ts_raw = (row.get("Date & Time") or row.get("timestamp") or row.get("time") or row.get("date") or "").strip()
    ts = _parse_csv_ts(ts_raw)
    if ts is None:
        return None
    try:
        return {
            "ts": ts,
            "open": float(row.get("open", 0) or 0),
            "high": float(row.get("high", 0) or 0),
            "low": float(row.get("low", 0) or 0),
            "close": float(row.get("close", 0) or 0),
            "volume": float(row.get("volume", 0) or 0),
        }
    except (ValueError, TypeError):
        return None


def _read_csv_bars(csv_path: Path) -> list[dict]:
    bars: list[dict] = []
    try:
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                bar = _parse_csv_row(row)
                if bar is not None:
                    bars.append(bar)
    except OSError:
        pass
    return bars


def load_history_csv(data_root: Path, symbol: str, tf_minutes: int) -> list[dict]:
    """Load OHLCV bars from downloaded CSV history for a symbol/timeframe."""
    history_dir = data_root / "history"
    if not history_dir.exists():
        return []
    tf_label = f"M{tf_minutes}"
    candidates = sorted(history_dir.glob(f"{symbol}_{tf_label}*.csv"))
    if not candidates:
        candidates = sorted(history_dir.glob(f"{symbol}*{tf_label}*.csv"))
    bars: list[dict] = []
    for csv_path in candidates:
        bars.extend(_read_csv_bars(csv_path))
    bars.sort(key=lambda b: b["ts"])
    return bars


def _bars_during_trade(
    bars: list[dict], entry_ts: datetime | None, exit_ts: datetime | None
) -> list[dict]:
    if not bars or entry_ts is None or exit_ts is None:
        return []
    entry_utc = entry_ts.astimezone(timezone.utc) if entry_ts.tzinfo else entry_ts.replace(tzinfo=timezone.utc)
    exit_utc = exit_ts.astimezone(timezone.utc) if exit_ts.tzinfo else exit_ts.replace(tzinfo=timezone.utc)
    return [
        {
            "timestamp": b["ts"].isoformat(),
            "open": b["open"],
            "high": b["high"],
            "low": b["low"],
            "close": b["close"],
            "volume": b["volume"],
        }
        for b in bars
        if entry_utc <= b["ts"] <= exit_utc
    ]


# ── Training cache ─────────────────────────────────────────────────────────────

def load_training_cache(data_root: Path, symbol: str, tf_minutes: int) -> list[dict]:
    """Load training cache records for a symbol/TF, sorted by ts_recorded."""
    path = data_root / f"training_cache_{symbol}_M{tf_minutes}.jsonl"
    records = _load_jsonl(path)
    records.sort(key=lambda r: r.get("ts_recorded", ""))
    return records


def _cache_candidate_score(rec: dict, entry_price: float, entry_ts: datetime, time_tol: float) -> float:
    """Return a match score for one cache record; inf if outside tolerance."""
    rec_ts = _parse_ts(rec.get("ts_recorded"))
    if rec_ts is None:
        return float("inf")
    try:
        rec_price = float(rec.get("entry_price") or 0)
    except (TypeError, ValueError):
        return float("inf")
    price_delta = abs(rec_price - entry_price)
    if price_delta > _TRAINING_CACHE_PRICE_TOL:
        return float("inf")
    time_delta = _abs_delta(rec_ts, entry_ts)
    if time_delta > time_tol:
        return float("inf")
    return price_delta * 10 + time_delta / 60.0


def _enrich_from_cache(trade: dict, cache: list[dict], tf_minutes: int) -> dict | None:
    """Find the best-matching training cache record for a trade by price+time proximity."""
    if not cache:
        return None
    entry_ts = _parse_ts(trade.get("entry_time"))
    try:
        entry_price = float(trade.get("entry_price") or 0)
    except (TypeError, ValueError):
        return None
    if not entry_price or entry_ts is None:
        return None

    time_tol = tf_minutes * 60.0 * _TRAINING_CACHE_TIME_TOL_MULT
    scored = (
        (rec, _cache_candidate_score(rec, entry_price, entry_ts, time_tol))
        for rec in cache
    )
    candidates = [(rec, s) for rec, s in scored if s < float("inf")]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])[0]


def _cache_to_entry_market_state(rec: dict) -> dict:
    """Build a market_state block from a training cache record (no bar-level data)."""
    return {
        "regime": rec.get("regime"),
        "imbalance": rec.get("imbalance"),
        "vpin_z": rec.get("vpin_z"),
        "depth_ratio": rec.get("depth_ratio"),
        "was_explore": rec.get("was_explore"),
        "trigger_action": rec.get("trigger_action"),
        "trigger_reward": rec.get("trigger_reward"),
        "capture_reward": rec.get("capture_reward"),
    }


def _cache_bars(rec: dict, field: str) -> list[dict]:
    """Convert raw bar tuples from training cache to timeline-keyed dicts."""
    raw = rec.get(field) or []
    out = []
    for b in raw:
        if isinstance(b, (list, tuple)) and len(b) >= 5:
            ts, o, h, l, c = b[0], b[1], b[2], b[3], b[4]
            out.append({"timestamp": str(ts), "open": o, "high": h, "low": l, "close": c})
    return out


# ── Decision indexing ──────────────────────────────────────────────────────────

def _index_decisions(decisions: list[dict]) -> tuple[dict, dict, list[dict], list[dict]]:
    """Return (trigger_by_id, harvester_by_id, all_triggers_sorted, all_harvesters_sorted)."""
    trigger_by_id: dict[str, dict] = {}
    harvester_by_id: dict[str, list[dict]] = {}
    all_triggers: list[dict] = []
    all_harvesters: list[dict] = []
    for dec in decisions:
        tid = dec.get("trade_id")
        agent = dec.get("agent", "")
        if agent == "TriggerAgent" and dec.get("decision") in ("LONG", "SHORT"):
            if tid:
                trigger_by_id[tid] = dec
            all_triggers.append(dec)
        elif agent == "HarvesterAgent":
            all_harvesters.append(dec)
            if tid:
                harvester_by_id.setdefault(tid, []).append(dec)
    all_triggers.sort(key=lambda d: d.get("timestamp", ""))
    all_harvesters.sort(key=lambda d: d.get("timestamp", ""))
    return trigger_by_id, harvester_by_id, all_triggers, all_harvesters


def _tf_match_window(tf_minutes: int) -> float:
    """Return timestamp correlation window in seconds, scaled to the timeframe.

    The trigger fires at bar close and the entry happens on the next bar open,
    so the natural gap is tf_minutes * 60 seconds.  We allow 3× that as slack
    to handle processing delays, re-entries, and session warm-up periods.
    """
    return max(300.0, tf_minutes * 60.0 * 3.0)


def _match_trigger(
    trade: dict,
    trigger_by_id: dict,
    all_triggers_sorted: list[dict],
    used_ids: set[str],
    entry_ts: datetime | None,
    max_delta_s: float = 300.0,
) -> tuple[str | None, dict | None]:
    """Return (hex_id, trigger_decision) for a trade."""
    hex_id: str | None = trade.get("decision_trade_id")
    if hex_id and hex_id in trigger_by_id:
        return hex_id, trigger_by_id[hex_id]

    target = trade.get("direction", "").upper()
    if target not in ("LONG", "SHORT"):
        return hex_id, None

    best_cand: dict | None = None
    best_delta = float("inf")
    for cand in all_triggers_sorted:
        cand_id = cand.get("trade_id")
        if cand_id in used_ids:
            continue
        if cand.get("decision") != target:
            continue
        d = _abs_delta(_parse_ts(cand.get("timestamp")), entry_ts)
        if d <= max_delta_s and d < best_delta:
            best_delta = d
            best_cand = cand

    if best_cand is not None:
        return best_cand.get("trade_id"), best_cand

    return hex_id, None


# ── Block builders ─────────────────────────────────────────────────────────────

def _build_trigger_decision_block(dec: dict) -> dict:
    ctx = dec.get("context", {})
    rsn = dec.get("reasoning", {})
    block: dict = {
        "timestamp": dec.get("timestamp"),
        "decision": dec.get("decision"),
        "confidence": dec.get("confidence"),
        "price": ctx.get("price"),
        "volatility": ctx.get("volatility"),
        "imbalance": ctx.get("imbalance"),
        "vpin_z": ctx.get("vpin_z"),
        "regime": ctx.get("regime"),
        "predicted_runway": rsn.get("predicted_runway"),
        "feasibility": rsn.get("feasibility"),
        "circuit_breakers_ok": rsn.get("circuit_breakers_ok"),
        "q_spread": rsn.get("q_spread"),
        "session": dec.get("session"),
    }
    ms: dict = {}
    for key in (
        # RS volatility (preferred over ATR; fractional log-price units)
        "rs_vol_short", "rs_vol_long", "rs_vol_ratio",
        # momentum and energy
        "er10", "ret1", "ret5", "ret20",
        "alignment_score", "bars_since_energy_bar", "hmm_probs",
        # gap context
        "gap_rs",
        # transaction cost at entry
        "half_spread",
        # model maturity and exploration
        "training_steps", "epsilon", "explore_flag",
        # entry bar OHLCV
        "entry_bar_ohlcv",
        # geometry and risk
        "geometry_efficiency", "geometry_runway",
        "depth_ratio", "var_95", "kurtosis",
        # soft gates that were active at entry (list of strings like "kurtosis=4.5>3.0")
        "gated_conditions",
        # runway prediction — gross from policy, net after spread cost
        "predicted_runway_gross", "predicted_runway_net",
        # runway prediction accuracy EMAs at time of entry
        "runway_accuracy", "runway_delta_ema", "conf_calib_err",
        # legacy — kept for backward compat with log entries written before RS migration
        "atr14",
    ):
        val = rsn.get(key)
        if val is not None:
            ms[key] = val
    if ms:
        block["market_state"] = ms
    return block


def _build_entry_block(trade: dict, trigger_dec: dict | None, order_open: dict | None) -> dict:
    block: dict = {
        "time": trade.get("entry_time"),
        "price": trade.get("entry_price"),
        "confidence": trade.get("entry_confidence"),
    }
    if trigger_dec:
        block["trigger_decision"] = _build_trigger_decision_block(trigger_dec)
    if order_open:
        d = order_open.get("data", {})
        block["order_submit"] = {
            "time": order_open.get("timestamp"),
            "order_id": d.get("order_id"),
            "side": d.get("side"),
            "quantity": d.get("quantity"),
            "session": order_open.get("session"),
        }
    return block


def _build_harvester_close_block(dec: dict) -> dict:
    ctx = dec.get("context", {})
    rsn = dec.get("reasoning", {})
    return {
        "timestamp": dec.get("timestamp"),
        "decision": dec.get("decision"),
        "confidence": dec.get("confidence"),
        "price": ctx.get("price"),
        "entry_price": ctx.get("entry_price"),
        "unrealized_pnl": ctx.get("unrealized_pnl"),
        "mfe": rsn.get("mfe"),
        "mae": rsn.get("mae"),
        "ticks_held": rsn.get("ticks_held"),
        "capture_ratio": rsn.get("capture_ratio"),
        "q_spread": rsn.get("q_spread"),
        "session": dec.get("session"),
    }


def _build_exit_block(
    trade: dict, harvester_close: dict | None, position_close_txn: dict | None, broker_exec: dict | None
) -> dict:
    block: dict = {
        "time": trade.get("exit_time"),
        "price": trade.get("exit_price"),
        "close_reason": trade.get("close_reason"),
    }
    if harvester_close:
        block["harvester_decision"] = _build_harvester_close_block(harvester_close)
    if position_close_txn:
        d = position_close_txn.get("data", {})
        block["position_close_event"] = {
            "time": position_close_txn.get("timestamp"),
            "position_id": d.get("position_id"),
            "pnl": d.get("pnl"),
            "mfe": d.get("mfe"),
            "mae": d.get("mae"),
            "session": position_close_txn.get("session"),
        }
    if broker_exec:
        d = broker_exec.get("data", {})
        block["broker_execution"] = {
            "time": broker_exec.get("timestamp"),
            "commission": d.get("commission"),
            "swap": d.get("swap"),
            "gross_profit": d.get("gross_profit"),
            "balance_after": d.get("balance_after"),
            "deal_id": d.get("deal_id"),
            "position_id": d.get("position_id"),
            "source": d.get("source"),
        }
    return block


def _build_costs_block(trade: dict) -> dict:
    """Extract cost fields added by new ctrader_ddqn_paper.py code."""
    return {
        "commission_rt": trade.get("commission_rt"),
        "swap": trade.get("swap"),
        "spread_cost_rt": trade.get("spread_cost_rt"),
        "pnl_net": trade.get("pnl_net"),
        "balance_after": trade.get("balance_after"),
        "commission_source": trade.get("commission_source", "model"),
        "diag_close_spread": trade.get("diag_close_spread"),
        "diag_close_spread_bps": trade.get("diag_close_spread_bps"),
    }


def _compute_capture(trade: dict) -> tuple[float | None, float | None]:
    """Derive (capture_ratio, capture_pct) from entry/exit prices and mfe_points.

    Always derives pnl_points from entry/exit prices — stored pnl_points is unreliable
    (normalize_trade_log_scale.py wrote pnl/qty/contract_size instead of price difference).
    """
    direction = str(trade.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return None, None
    try:
        mfe_pts = float(trade.get("mfe_points") or 0)
        entry = float(trade.get("entry_price") or 0)
        exit_ = float(trade.get("exit_price") or 0)
    except (TypeError, ValueError):
        return None, None
    if mfe_pts <= 0 or not entry or not exit_:
        return None, None
    pnl_pts = (exit_ - entry) if direction == "LONG" else (entry - exit_)
    ratio = pnl_pts / mfe_pts
    return ratio, ratio * 100.0


def _derive_winner_to_loser(trade: dict) -> bool | None:
    """True when trade moved in-the-money then closed at a loss."""
    stored = trade.get("winner_to_loser")
    if stored is not None:
        return stored
    try:
        pnl = float(trade.get("pnl") or 0)
        mfe_pts = float(trade.get("mfe_points") or 0)
    except (TypeError, ValueError):
        return None
    if mfe_pts > 1e-9:
        return pnl < 0
    return None


def _resolve_capture(trade: dict) -> tuple[float | None, float | None]:
    """Return (capture_ratio, capture_pct), preferring price-point derivation over stored values."""
    ratio, pct = _compute_capture(trade)
    if ratio is not None:
        return ratio, pct
    try:
        stored = trade.get("capture_ratio")
        if stored is not None:
            ratio = float(stored)
            return ratio, ratio * 100.0
        stored_pct = trade.get("capture_pct")
        if stored_pct is not None:
            pct = float(stored_pct)
            return pct / 100.0, pct
    except (TypeError, ValueError):
        pass
    return None, None


def _derive_pnl_points(trade: dict) -> float | None:
    """Return pnl_points from stored field or derived from entry/exit prices."""
    pts = trade.get("pnl_points")
    if pts is not None:
        return pts
    direction = str(trade.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return None
    try:
        entry = float(trade.get("entry_price") or 0)
        exit_ = float(trade.get("exit_price") or 0)
        return (exit_ - entry) if direction == "LONG" else (entry - exit_)
    except (TypeError, ValueError):
        return None


def _build_outcome_block(trade: dict) -> dict:
    cap_ratio, cap_pct = _resolve_capture(trade)
    pnl_pts = _derive_pnl_points(trade)

    return {
        "pnl": trade.get("pnl"),
        "pnl_points": pnl_pts,
        "mfe": trade.get("mfe"),
        "mfe_points": trade.get("mfe_points"),
        "mae": trade.get("mae"),
        "mae_points": trade.get("mae_points"),
        "hold_seconds": trade.get("hold_seconds"),
        "bars_held": trade.get("bars_held"),
        "ticks_held": trade.get("ticks_held"),
        "capture_ratio": cap_ratio,
        "capture_pct": cap_pct,
        "trigger_quality": trade.get("trigger_quality"),
        "harvester_quality": trade.get("harvester_quality"),
        "winner_to_loser": _derive_winner_to_loser(trade),
        "predicted_runway_net_points": trade.get("predicted_runway_net_points"),
        "runway_utilization": trade.get("runway_utilization"),
        "runway_error_pct": trade.get("runway_error_pct"),
        "diag_zero_mfe_loss": trade.get("diag_zero_mfe_loss"),
        "diag_circuit_breaker_active": trade.get("diag_circuit_breaker_active"),
        "diag_circuit_breakers_tripped": trade.get("diag_circuit_breakers_tripped"),
    }


# ── Transaction pool bundle ────────────────────────────────────────────────────

class _TxnPools:
    """Groups the three transaction event lists so _assemble_trade_record stays under 13 params."""

    __slots__ = ("order_submits", "position_closes", "broker_execs")

    def __init__(
        self,
        order_submits: list[dict],
        position_closes: list[dict],
        broker_execs: list[dict],
    ) -> None:
        self.order_submits = order_submits
        self.position_closes = position_closes
        self.broker_execs = broker_execs

    @classmethod
    def from_transactions(cls, transactions: list[dict]) -> "_TxnPools":
        return cls(
            order_submits=[t for t in transactions if t.get("event_type") == "ORDER_SUBMIT"],
            position_closes=[t for t in transactions if t.get("event_type") == "POSITION_CLOSE"],
            broker_execs=[t for t in transactions if t.get("event_type") == "BROKER_EXECUTION"],
        )

    @classmethod
    def empty(cls) -> "_TxnPools":
        return cls([], [], [])


# ── Main builder ───────────────────────────────────────────────────────────────

def _load_bar_states(paper_dir: Path) -> dict[str, list[dict]]:
    bar_states: list[dict] = []
    for dl_path in sorted(paper_dir.glob("decision_log_*.json")):
        raw = _load_json(dl_path)
        if isinstance(raw, list):
            bar_states.extend(raw)
    index: dict[str, list[dict]] = {}
    for bs in bar_states:
        tid = bs.get("trade_id")
        if tid:
            index.setdefault(tid, []).append(bs)
    return index


def _build_bar_snaps(bar_snaps: list[dict]) -> list[dict]:
    return [
        {
            "timestamp": b.get("timestamp"),
            "close": b.get("details", {}).get("close"),
            "action": b.get("details", {}).get("action"),
            "exit_action": b.get("details", {}).get("exit_action"),
            "confidence": b.get("details", {}).get("confidence"),
            "exit_conf": b.get("details", {}).get("exit_conf"),
            "mfe": b.get("details", {}).get("mfe"),
            "mae": b.get("details", {}).get("mae"),
            "bars_held": b.get("details", {}).get("bars_held"),
            "entry_price": b.get("details", {}).get("entry_price"),
            "circuit_breaker": b.get("details", {}).get("circuit_breaker"),
        }
        for b in sorted(bar_snaps, key=lambda b: b.get("timestamp", ""))
    ]


def _harvester_decs_for_trade(
    hex_id: str | None,
    harvester_by_id: dict,
    all_harvesters: list[dict],
    entry_ts: datetime | None,
    exit_ts: datetime | None,
    match_window: float,
) -> list[dict]:
    """Return harvester decisions for a trade, falling back to timestamp window."""
    if hex_id and hex_id in harvester_by_id:
        return sorted(harvester_by_id[hex_id], key=lambda d: d.get("timestamp", ""))

    if entry_ts is None or exit_ts is None:
        return []

    entry_utc = entry_ts.astimezone(timezone.utc) if entry_ts.tzinfo else entry_ts.replace(tzinfo=timezone.utc)
    exit_utc = exit_ts.astimezone(timezone.utc) if exit_ts.tzinfo else exit_ts.replace(tzinfo=timezone.utc)
    slack = max(60.0, match_window * 0.5)
    lo = entry_utc.timestamp() - slack
    hi = exit_utc.timestamp() + slack
    return [
        d for d in all_harvesters
        if lo <= (_parse_ts(d.get("timestamp")) or entry_utc).replace(tzinfo=timezone.utc).timestamp() <= hi
    ]


def _assemble_trade_record(
    trade: dict,
    trigger_dec: dict | None,
    hex_id: str | None,
    harvester_by_id: dict,
    all_harvesters: list[dict],
    bars_by_id: dict,
    txn_pools: _TxnPools,
    history_bars: list[dict],
    entry_ts: datetime | None,
    exit_ts: datetime | None,
    match_window: float = 300.0,
    cache_rec: dict | None = None,
) -> dict:
    harvester_decs = _harvester_decs_for_trade(
        hex_id, harvester_by_id, all_harvesters, entry_ts, exit_ts, match_window
    )
    harvester_close = next((d for d in harvester_decs if d.get("decision") == "CLOSE"), None)
    harvester_holds = [d for d in harvester_decs if d.get("decision") == "HOLD"]

    bars_summary = _build_bar_snaps(bars_by_id.get(hex_id, []) if hex_id else [])
    # Fall back: training cache bars, then CSV history
    cache_bars = _cache_bars(cache_rec, "exit_bars") if (cache_rec and not bars_summary) else []
    csv_bars = (
        _bars_during_trade(history_bars, entry_ts, exit_ts)
        if (not bars_summary and not cache_bars)
        else []
    )

    holds_summary = [
        {
            "timestamp": d.get("timestamp"),
            "price": d.get("context", {}).get("price"),
            "unrealized_pnl": d.get("context", {}).get("unrealized_pnl"),
            "mfe": d.get("reasoning", {}).get("mfe"),
            "mae": d.get("reasoning", {}).get("mae"),
            "ticks_held": d.get("reasoning", {}).get("ticks_held"),
            "capture_ratio": d.get("reasoning", {}).get("capture_ratio"),
            "confidence": d.get("confidence"),
            "regime": d.get("reasoning", {}).get("regime"),
            "realized_vol": d.get("reasoning", {}).get("realized_vol"),
            "depth_ratio": d.get("reasoning", {}).get("depth_ratio"),
            "exit_floor": d.get("reasoning", {}).get("exit_floor"),
            "trailing_stop_active": d.get("reasoning", {}).get("trailing_stop_active"),
            "capture_decay_armed": d.get("reasoning", {}).get("capture_decay_armed"),
        }
        for d in harvester_holds
    ]

    order_open = _nearest_txn(txn_pools.order_submits, entry_ts, max_delta_s=120.0)
    position_close_txn = _nearest_txn(txn_pools.position_closes, exit_ts, max_delta_s=120.0)
    broker_exec_txn = _nearest_txn(txn_pools.broker_execs, exit_ts, max_delta_s=120.0)

    # Build entry block; if no trigger_dec but we have a cache record, inject market state
    entry_block = _build_entry_block(trade, trigger_dec, order_open)
    if trigger_dec is None and cache_rec is not None:
        entry_block["market_state"] = _cache_to_entry_market_state(cache_rec)

    # Enrich outcome block with cache values where trade_log fields are absent
    outcome_block = _build_outcome_block(trade)
    if cache_rec is not None:
        if outcome_block.get("mfe") in (None, 0) and cache_rec.get("mfe"):
            outcome_block["mfe"] = cache_rec["mfe"]
            outcome_block["mfe_points"] = cache_rec["mfe"]
        if outcome_block.get("mae") in (None, 0) and cache_rec.get("mae"):
            outcome_block["mae"] = cache_rec["mae"]
            outcome_block["mae_points"] = cache_rec["mae"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_trade_id": hex_id,
        "trade_log_id": trade.get("trade_id"),
        "ticket": trade.get("ticket"),
        "position_id": trade.get("position_id"),
        "symbol": trade.get("symbol"),
        "timeframe": trade.get("timeframe"),
        "timeframe_minutes": trade.get("timeframe_minutes"),
        "direction": trade.get("direction"),
        "quantity": trade.get("quantity"),
        "entry": entry_block,
        "holds": holds_summary,
        "bars": bars_summary or cache_bars or csv_bars,
        "exit": _build_exit_block(trade, harvester_close, position_close_txn, broker_exec_txn),
        "costs": _build_costs_block(trade),
        "outcome": outcome_block,
        "data_sources": {
            "trade_log": "",
            "decisions": "",
            "transactions": "",
            "trigger_decision_matched": trigger_dec is not None,
            "harvester_close_matched": harvester_close is not None,
            "bar_snapshots_available": len(bars_summary),
            "cache_bars_used": len(cache_bars),
            "csv_bars_used": len(csv_bars),
            "harvester_holds": len(holds_summary),
            "broker_execution_matched": broker_exec_txn is not None,
            "training_cache_matched": cache_rec is not None,
        },
    }


def _load_trade_log_with_fallback(
    paper_dir: Path, symbol: str, tf_minutes: int
) -> tuple[list[dict], Path]:
    """Return (trade_records, source_path), falling back to global log when local is absent/empty."""
    local_path = paper_dir / _TRADE_LOG_FILENAME
    records = _load_jsonl(local_path)
    if records:
        return records, local_path

    global_path = paper_dir.parent / _TRADE_LOG_FILENAME
    if not global_path.exists():
        return [], local_path

    sym_up = symbol.upper()
    all_records = _load_jsonl(global_path)
    filtered = [
        r for r in all_records
        if str(r.get("symbol", "")).upper() == sym_up
        and int(r.get("timeframe_minutes", 0) or 0) == tf_minutes
    ]
    return filtered, global_path


def build_lifecycle_for_dir(
    paper_dir: Path,
    history_bars: list[dict] | None = None,
    training_cache: list[dict] | None = None,
) -> list[dict]:
    """Build trade lifecycle records for one paper_SYMBOL_TF directory."""
    sym, tf_minutes = _sym_tf_from_dir(paper_dir.name)
    trade_log, trade_log_path = _load_trade_log_with_fallback(paper_dir, sym, tf_minutes)
    if not trade_log:
        return []

    global_log_used = trade_log_path != (paper_dir / _TRADE_LOG_FILENAME)

    dec_path = paper_dir / "logs" / "audit" / "decisions.jsonl"
    txn_path = paper_dir / "logs" / "audit" / "transactions.jsonl"
    decisions = _load_jsonl(dec_path)
    transactions = _load_jsonl(txn_path)
    trigger_by_id, harvester_by_id, all_triggers_sorted, all_harvesters = _index_decisions(decisions)
    bars_by_id = _load_bar_states(paper_dir)

    txn_pools = _TxnPools.from_transactions(transactions)
    h_bars = history_bars or []
    cache = training_cache or []

    match_window = _tf_match_window(tf_minutes) if tf_minutes > 0 else 300.0
    results: list[dict] = []
    used_trigger_ids: set[str] = set()

    for trade in trade_log:
        entry_ts = _parse_ts(trade.get("entry_time"))
        exit_ts = _parse_ts(trade.get("exit_time"))
        hex_id, trigger_dec = _match_trigger(
            trade, trigger_by_id, all_triggers_sorted, used_trigger_ids, entry_ts,
            max_delta_s=match_window,
        )
        if hex_id:
            used_trigger_ids.add(hex_id)
        cache_rec = _enrich_from_cache(trade, cache, tf_minutes) if cache else None
        record = _assemble_trade_record(
            trade, trigger_dec, hex_id,
            harvester_by_id, all_harvesters, bars_by_id,
            txn_pools,
            h_bars, entry_ts, exit_ts, match_window,
            cache_rec=cache_rec,
        )
        record["data_sources"]["trade_log"] = str(trade_log_path)
        record["data_sources"]["decisions"] = str(dec_path)
        record["data_sources"]["transactions"] = str(txn_path)
        record["data_sources"]["trade_log_global"] = global_log_used
        results.append(record)

    return results


def build_reconstructed_audit(
    data_root: Path,
    symbol: str,
    from_date: str | None = None,
) -> tuple[list[dict], Path]:
    """Build a unified enriched lifecycle file for all TFs of one symbol.

    Reads the global trade_log.jsonl, routes each trade to its paper_SYMBOL_TF
    directory for decision-log correlation, and falls back to training cache
    enrichment for trades outside the decision-log window.  All TFs are merged
    into a single list sorted by entry_time, written to
    data/reconstructed_audit_{symbol}.jsonl.

    Returns (records, output_path).
    """
    global_log = data_root / _GLOBAL_TRADE_LOG
    all_trades = _load_jsonl(global_log)
    sym_up = symbol.upper()
    trades = [r for r in all_trades if str(r.get("symbol", "")).upper() == sym_up]
    if from_date:
        trades = [r for r in trades if (r.get("entry_time") or "") >= from_date]
    if not trades:
        return [], data_root / f"reconstructed_audit_{sym_up}.jsonl"

    # Collect all TFs present
    tf_minutes_set = sorted({int(r.get("timeframe_minutes", 0) or 0) for r in trades if r.get("timeframe_minutes")})

    all_records: list[dict] = []
    for tf_min in tf_minutes_set:
        paper_dir = data_root / f"paper_{sym_up}_M{tf_min}"
        h_bars = load_history_csv(data_root, sym_up, tf_min)
        cache = load_training_cache(data_root, sym_up, tf_min)

        # If paper dir exists, use full pipeline (it will fall back to global trade_log internally)
        if paper_dir.is_dir():
            records = build_lifecycle_for_dir(paper_dir, history_bars=h_bars, training_cache=cache)
        else:
            # Paper dir absent — build directly from global trade_log slice + cache only
            tf_trades = [r for r in trades if int(r.get("timeframe_minutes", 0) or 0) == tf_min]
            txn_pools = _TxnPools.empty()
            match_window = _tf_match_window(tf_min) if tf_min > 0 else 300.0
            records = []
            for trade in tf_trades:
                entry_ts = _parse_ts(trade.get("entry_time"))
                exit_ts = _parse_ts(trade.get("exit_time"))
                cache_rec = _enrich_from_cache(trade, cache, tf_min) if cache else None
                csv_bars = _bars_during_trade(h_bars, entry_ts, exit_ts)
                record = _assemble_trade_record(
                    trade, None, None,
                    {}, [], {},
                    txn_pools,
                    csv_bars, entry_ts, exit_ts, match_window,
                    cache_rec=cache_rec,
                )
                record["data_sources"]["trade_log"] = str(global_log)
                record["data_sources"]["trade_log_global"] = True
                records.append(record)

        for r in records:
            r["epoch"] = "openapi" if (r.get("entry", {}).get("time") or "") >= "2026-04-24" else "legacy"
        all_records.extend(records)

    all_records.sort(key=lambda r: r.get("entry", {}).get("time") or "")
    for i, r in enumerate(all_records):
        r["audit_seq"] = i + 1

    out_path = data_root / f"reconstructed_audit_{sym_up}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, default=str) + "\n")

    return all_records, out_path


# ── CLI helpers ────────────────────────────────────────────────────────────────

def _dir_matches(name: str, symbol: str, tf: str) -> bool:
    sym_up = symbol.upper()
    tf_up = tf.upper()
    if symbol and f"_{sym_up}_" not in name and not name.endswith(f"_{sym_up}"):
        return False
    if tf and not name.endswith(f"_{tf_up}"):
        return False
    return True


def _discover_candidates(data_root: Path, symbol: str, tf: str) -> list[Path]:
    return [
        d for d in sorted(data_root.iterdir())
        if d.is_dir() and d.name.startswith("paper_") and _dir_matches(d.name, symbol, tf)
    ]


def _sym_tf_from_dir(name: str) -> tuple[str, int]:
    parts = name.split("_")
    if len(parts) < 3:
        return "", 0
    sym = parts[1]
    tf_str = parts[2]
    tf_minutes = int(tf_str[1:]) if tf_str.startswith("M") and tf_str[1:].isdigit() else 0
    return sym, tf_minutes


def _emit_stdout(records: list[dict], pretty: bool) -> None:
    indent = 2 if pretty else None
    for r in records:
        print(json.dumps(r, indent=indent, default=str))


def _write_to_file(paper_dir: Path, records: list[dict]) -> int:
    out_path = paper_dir / "logs" / "audit" / "trade_lifecycle.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_count = sum(r["data_sources"]["csv_bars_used"] for r in records)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            r["audit_seq"] = i + 1
            f.write(json.dumps(r, default=str) + "\n")
    csv_note = f" ({csv_count} CSV bars)" if csv_count else ""
    print(f"[OK] {paper_dir.name}: {len(records)} trade(s) → {out_path}{csv_note}")
    return len(records)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-trade lifecycle audit records")
    ap.add_argument("--data-dir", default="data", help="Root data directory")
    ap.add_argument("--symbol", default="", help="Filter by symbol (e.g. XAUUSD)")
    ap.add_argument("--tf", default="", help="Filter by timeframe (e.g. M15)")
    ap.add_argument("--all", action="store_true", dest="all_dirs", help="Process all paper dirs")
    ap.add_argument("--stdout", action="store_true", help="Print to stdout instead of writing file")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = ap.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.exists():
        print(f"[ERROR] Data directory not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    candidates = _discover_candidates(data_root, args.symbol, args.tf)
    if not candidates:
        print("[INFO] No matching paper directories found.", file=sys.stderr)
        sys.exit(0)

    total_written = 0
    for paper_dir in candidates:
        sym, tf_minutes = _sym_tf_from_dir(paper_dir.name)
        h_bars = load_history_csv(data_root, sym, tf_minutes) if sym and tf_minutes else []
        records = build_lifecycle_for_dir(paper_dir, history_bars=h_bars)
        if not records:
            continue
        if args.stdout:
            _emit_stdout(records, args.pretty)
        else:
            total_written += _write_to_file(paper_dir, records)

    if not args.stdout:
        print(f"\n[DONE] {total_written} lifecycle records written across {len(candidates)} directory/ies.")


if __name__ == "__main__":
    main()
