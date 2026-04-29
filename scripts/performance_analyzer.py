#!/usr/bin/env python3
"""Performance Analyzer & Self-Healing System.

Reads trade_log.jsonl for recent trades, computes fleet and per-bot health
metrics, detects anomalies, and optionally applies corrective adjustments to
learned_parameters.json via LearnedParametersManager.

Writes data/performance_health.json on every run — the HUD and run_universe
supervisor read this file to surface health status.

Usage:
    python3 scripts/performance_analyzer.py --hours 24
    python3 scripts/performance_analyzer.py --hours 12 --auto-heal --verbose
    python3 scripts/performance_analyzer.py --hours 24 --auto-heal --min-trades 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

DATA_DIR = Path(os.environ.get("CTRADER_DATA_DIR", "data"))
TRADE_LOG = DATA_DIR / "trade_log.jsonl"
HEALTH_FILE = DATA_DIR / "performance_health.json"

LOG = logging.getLogger("perf_analyzer")

# ── anomaly thresholds ────────────────────────────────────────────────────────
# All thresholds are documented so a future reader knows the intent.
DDQN_WR_LOW_THRESHOLD = 0.30          # DDQN WR below this → harvester firing randomly
DDQN_MIN_TRADES = 5                   # minimum DDQN closes before we trust the WR
EMERGENCY_RATE_HIGH = 0.05            # emergency stops > 5% of trades → entries too risky
LOSER_WINNER_RATIO_SEVERE = 2.8       # avg_loser / avg_winner above this → R:R broken
CAPTURE_MEAN_LOW = 0.25               # mean capture ratio below this → exits too early
CAPTURE_MIN_TRADES = 8                # minimum closed trades before capture check fires
WTL_PENALTY_EXCESSIVE = -1.5         # mean WTL penalty below this → signal too noisy
RUNWAY_ACCURACY_LOW = 0.35            # runway accuracy mean below → predictor drifting
PNL_ALIGNMENT_WEAK = 0.08             # mean pnl_alignment in harvester below → signal weak
TRIGGER_SATURATION_PCT = 0.15         # fraction of trigger rewards at ±2.99 rail → regression
BOT_LOSING_STREAK_MULT = 4.0          # bot total PnL < -(fleet avg_winner * mult) → flag

# ── correction step sizes ─────────────────────────────────────────────────────
# Kept small; compound slowly rather than over-correct in one cycle.
DELTA_EXIT_CONF_DDQN_LOW    = +0.04   # raise exit_confidence_threshold
DELTA_CONF_FLOOR_EMERGENCY  = +0.02   # raise confidence_floor (more selective entries)
DELTA_CONF_FLOOR_BAD_RR     = +0.02   # raise confidence_floor when R:R is broken
DELTA_EXIT_CONF_CAPTURE     = +0.03   # raise exit_confidence_threshold on poor capture
DELTA_WTL_MULT_REDUCE       = -0.20   # reduce wtl_penalty_multiplier (absolute)
DELTA_RUNWAY_CAL_ALPHA       = +0.04   # speed up runway calibration when drifting
DELTA_PNL_ALIGN_MULT        = +0.10   # raise pnl_alignment_multiplier when weak


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class BotMetrics:
    symbol: str
    timeframe: str
    n_trades: int = 0
    n_winners: int = 0
    n_losers: int = 0
    total_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    # Close reason breakdown
    n_ddqn: int = 0
    n_ddqn_wins: int = 0
    n_runway_capture: int = 0
    n_runway_wins: int = 0
    n_emergency: int = 0
    n_other: int = 0
    # Signal quality
    mean_capture: float = 0.0
    mean_trigger_reward: float = 0.0
    mean_harvest_reward: float = 0.0
    mean_pnl_alignment: float = 0.0
    mean_wtl_penalty: float = 0.0
    trigger_at_rail: float = 0.0     # fraction at ±2.99
    runway_accuracy: float = 0.0
    # Anomalies detected for this bot
    anomalies: list[str] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.n_winners / self.n_trades if self.n_trades > 0 else 0.0

    @property
    def ddqn_win_rate(self) -> float:
        return self.n_ddqn_wins / self.n_ddqn if self.n_ddqn > 0 else 0.0

    @property
    def runway_win_rate(self) -> float:
        return self.n_runway_wins / self.n_runway_capture if self.n_runway_capture > 0 else 0.0

    @property
    def emergency_rate(self) -> float:
        return self.n_emergency / self.n_trades if self.n_trades > 0 else 0.0

    @property
    def loser_winner_ratio(self) -> float:
        return abs(self.avg_loser) / self.avg_winner if self.avg_winner > 0 else 0.0

    @property
    def expectancy(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return self.total_pnl / self.n_trades


@dataclass
class FleetMetrics:
    n_trades: int = 0
    total_pnl: float = 0.0
    fleet_wr: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    ddqn_wr: float = 0.0
    runway_wr: float = 0.0
    emergency_rate: float = 0.0
    mean_capture: float = 0.0
    mean_trigger_reward: float = 0.0
    mean_wtl_penalty: float = 0.0
    trigger_saturation_pct: float = 0.0
    runway_accuracy: float = 0.0
    analysis_window_hours: float = 24.0
    generated_at: str = ""


@dataclass
class Anomaly:
    code: str
    severity: str          # "CRITICAL", "WARNING", "INFO"
    symbol: str
    timeframe: str
    message: str
    metric_value: float
    threshold: float
    correction: str        # description of correction to apply
    param_name: str        # LearnedParametersManager param to adjust (empty = none)
    delta: float           # correction delta (0 = flag only)


# ── data loading ─────────────────────────────────────────────────────────────

def load_recent_trades(hours: float = 24.0) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    trades: list[dict[str, Any]] = []
    if not TRADE_LOG.exists():
        LOG.warning("trade_log.jsonl not found at %s", TRADE_LOG)
        return trades
    with TRADE_LOG.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                ts_str = t.get("exit_time") or t.get("entry_time") or ""
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts >= cutoff:
                        trades.append(t)
            except Exception:
                pass
    return trades


# ── metrics computation ────────────────────────────────────────────────────────

def compute_bot_metrics(trades: list[dict[str, Any]]) -> dict[str, BotMetrics]:
    by_bot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades:
        key = f"{t.get('symbol', '?')}_{t.get('timeframe', '?')}"
        by_bot[key].append(t)

    result: dict[str, BotMetrics] = {}
    for key, bot_trades in by_bot.items():
        sym, tf = key.rsplit("_", 1)
        m = BotMetrics(symbol=sym, timeframe=tf, n_trades=len(bot_trades))

        pnls = [t.get("pnl", 0.0) or 0.0 for t in bot_trades]
        winners = [p for p in pnls if p > 0]
        losers  = [p for p in pnls if p <= 0]
        m.n_winners = len(winners)
        m.n_losers  = len(losers)
        m.total_pnl = sum(pnls)
        m.avg_winner = sum(winners) / len(winners) if winners else 0.0
        m.avg_loser  = sum(losers)  / len(losers)  if losers  else 0.0

        for t in bot_trades:
            reason = (t.get("close_reason") or "").strip()
            pnl    = t.get("pnl", 0.0) or 0.0
            if reason == "ddqn_model":
                m.n_ddqn += 1
                if pnl > 0:
                    m.n_ddqn_wins += 1
            elif reason == "runway_capture":
                m.n_runway_capture += 1
                if pnl > 0:
                    m.n_runway_wins += 1
            elif reason == "emergency_stop":
                m.n_emergency += 1
            else:
                m.n_other += 1

        def _mean(key_fn):
            vals = [key_fn(t) for t in bot_trades if key_fn(t) is not None]
            return sum(vals) / len(vals) if vals else 0.0

        m.mean_capture        = _mean(lambda t: t.get("capture_ratio"))
        m.mean_trigger_reward = _mean(lambda t: t.get("trigger_reward"))
        m.mean_harvest_reward = _mean(lambda t: t.get("reward_harvester_total"))
        m.mean_wtl_penalty    = _mean(lambda t: t.get("reward_wtl_penalty"))
        m.runway_accuracy     = _mean(lambda t: t.get("runway_accuracy_at_entry"))

        pnl_aligns = []
        for t in bot_trades:
            bd = t.get("reward_harvester_breakdown") or {}
            pa = bd.get("pnl_alignment")
            if pa is not None:
                pnl_aligns.append(pa)
        m.mean_pnl_alignment = sum(pnl_aligns) / len(pnl_aligns) if pnl_aligns else 0.0

        trigger_rewards = [t.get("trigger_reward", 0.0) or 0.0 for t in bot_trades]
        at_rail = sum(1 for r in trigger_rewards if abs(r) >= 2.99)
        m.trigger_at_rail = at_rail / len(trigger_rewards) if trigger_rewards else 0.0

        result[key] = m
    return result


def compute_fleet_metrics(
    trades: list[dict[str, Any]],
    bot_metrics: dict[str, BotMetrics],
    hours: float,
) -> FleetMetrics:
    f = FleetMetrics(n_trades=len(trades), analysis_window_hours=hours,
                     generated_at=datetime.now(timezone.utc).isoformat())
    if not trades:
        return f

    pnls    = [t.get("pnl", 0.0) or 0.0 for t in trades]
    winners = [p for p in pnls if p > 0]
    losers  = [p for p in pnls if p <= 0]
    f.total_pnl  = sum(pnls)
    f.fleet_wr   = len(winners) / len(pnls) if pnls else 0.0
    f.avg_winner = sum(winners) / len(winners) if winners else 0.0
    f.avg_loser  = sum(losers)  / len(losers)  if losers  else 0.0

    n_ddqn  = sum(m.n_ddqn for m in bot_metrics.values())
    n_ddqn_w = sum(m.n_ddqn_wins for m in bot_metrics.values())
    n_run   = sum(m.n_runway_capture for m in bot_metrics.values())
    n_run_w = sum(m.n_runway_wins for m in bot_metrics.values())
    n_emerg = sum(m.n_emergency for m in bot_metrics.values())
    f.ddqn_wr       = n_ddqn_w / n_ddqn if n_ddqn > 0 else 0.0
    f.runway_wr     = n_run_w  / n_run  if n_run  > 0 else 0.0
    f.emergency_rate = n_emerg / len(trades) if trades else 0.0

    captures = [t.get("capture_ratio") for t in trades if t.get("capture_ratio") is not None]
    f.mean_capture = sum(captures) / len(captures) if captures else 0.0

    trigger_r = [t.get("trigger_reward", 0.0) or 0.0 for t in trades]
    f.mean_trigger_reward = sum(trigger_r) / len(trigger_r) if trigger_r else 0.0
    at_rail = sum(1 for r in trigger_r if abs(r) >= 2.99)
    f.trigger_saturation_pct = at_rail / len(trigger_r) if trigger_r else 0.0

    wtl = [t.get("reward_wtl_penalty", 0.0) or 0.0 for t in trades]
    f.mean_wtl_penalty = sum(wtl) / len(wtl) if wtl else 0.0

    acc = [t.get("runway_accuracy_at_entry") for t in trades if t.get("runway_accuracy_at_entry") is not None]
    f.runway_accuracy = sum(acc) / len(acc) if acc else 0.0

    return f


# ── anomaly detection ─────────────────────────────────────────────────────────

def detect_anomalies(
    bot_metrics: dict[str, BotMetrics],
    fleet: FleetMetrics,
    min_trades: int,
) -> list[Anomaly]:
    anomalies: list[Anomaly] = []

    # Fleet-level: trigger saturation regression check
    if fleet.trigger_saturation_pct > TRIGGER_SATURATION_PCT and fleet.n_trades >= 10:
        anomalies.append(Anomaly(
            code="TRIGGER_SATURATION",
            severity="CRITICAL",
            symbol="FLEET", timeframe="ALL",
            message=f"Trigger reward at ±3.0 rail: {fleet.trigger_saturation_pct:.0%} of trades (threshold {TRIGGER_SATURATION_PCT:.0%}). Runway predictor saturation bug may have re-appeared.",
            metric_value=fleet.trigger_saturation_pct,
            threshold=TRIGGER_SATURATION_PCT,
            correction="Check openapi_hub.py _close_position — ensure shaped_tr fallback is active (abs(shaped_tr) < 2.99 gate).",
            param_name="", delta=0.0,
        ))

    # Fleet-level: emergency stop rate
    if fleet.emergency_rate > EMERGENCY_RATE_HIGH and fleet.n_trades >= min_trades:
        anomalies.append(Anomaly(
            code="EMERGENCY_RATE_HIGH",
            severity="CRITICAL",
            symbol="FLEET", timeframe="ALL",
            message=f"Emergency stop rate {fleet.emergency_rate:.1%} > {EMERGENCY_RATE_HIGH:.0%}. Entries are too risky — raise confidence floor.",
            metric_value=fleet.emergency_rate,
            threshold=EMERGENCY_RATE_HIGH,
            correction=f"Raise confidence_floor by {DELTA_CONF_FLOOR_EMERGENCY:+.3f} for affected bots.",
            param_name="confidence_floor", delta=DELTA_CONF_FLOOR_EMERGENCY,
        ))

    for key, m in bot_metrics.items():
        if m.n_trades < min_trades:
            continue

        # DDQN harvester underperforming
        if m.n_ddqn >= DDQN_MIN_TRADES and m.ddqn_win_rate < DDQN_WR_LOW_THRESHOLD:
            anomalies.append(Anomaly(
                code="DDQN_WIN_RATE_LOW",
                severity="WARNING",
                symbol=m.symbol, timeframe=m.timeframe,
                message=(
                    f"{key}: DDQN closes WR={m.ddqn_win_rate:.0%} ({m.n_ddqn} closes) "
                    f"vs runway_capture WR={m.runway_win_rate:.0%}. "
                    "DDQN is exiting at random — raise exit confidence threshold."
                ),
                metric_value=m.ddqn_win_rate,
                threshold=DDQN_WR_LOW_THRESHOLD,
                correction=f"Raise exit_confidence_threshold by {DELTA_EXIT_CONF_DDQN_LOW:+.3f}.",
                param_name="exit_confidence_threshold", delta=DELTA_EXIT_CONF_DDQN_LOW,
            ))

        # Per-bot emergency stop rate
        if m.emergency_rate > EMERGENCY_RATE_HIGH:
            anomalies.append(Anomaly(
                code="EMERGENCY_RATE_HIGH",
                severity="WARNING",
                symbol=m.symbol, timeframe=m.timeframe,
                message=f"{key}: Emergency stop rate {m.emergency_rate:.1%}. Raise confidence floor.",
                metric_value=m.emergency_rate,
                threshold=EMERGENCY_RATE_HIGH,
                correction=f"Raise confidence_floor by {DELTA_CONF_FLOOR_EMERGENCY:+.3f}.",
                param_name="confidence_floor", delta=DELTA_CONF_FLOOR_EMERGENCY,
            ))

        # R:R broken — losers much larger than winners
        if m.loser_winner_ratio > LOSER_WINNER_RATIO_SEVERE and m.avg_winner > 0:
            anomalies.append(Anomaly(
                code="BAD_RISK_REWARD",
                severity="WARNING",
                symbol=m.symbol, timeframe=m.timeframe,
                message=(
                    f"{key}: avg_loser={m.avg_loser:.2f} is {m.loser_winner_ratio:.1f}× avg_winner={m.avg_winner:.2f}. "
                    "R:R is broken — raise entry selectivity."
                ),
                metric_value=m.loser_winner_ratio,
                threshold=LOSER_WINNER_RATIO_SEVERE,
                correction=f"Raise confidence_floor by {DELTA_CONF_FLOOR_BAD_RR:+.3f}.",
                param_name="confidence_floor", delta=DELTA_CONF_FLOOR_BAD_RR,
            ))

        # Poor capture efficiency
        if m.n_trades >= CAPTURE_MIN_TRADES and m.mean_capture < CAPTURE_MEAN_LOW:
            anomalies.append(Anomaly(
                code="CAPTURE_EFFICIENCY_LOW",
                severity="WARNING",
                symbol=m.symbol, timeframe=m.timeframe,
                message=f"{key}: mean_capture={m.mean_capture:.3f} < {CAPTURE_MEAN_LOW}. DDQN is exiting before protective stops fire.",
                metric_value=m.mean_capture,
                threshold=CAPTURE_MEAN_LOW,
                correction=f"Raise exit_confidence_threshold by {DELTA_EXIT_CONF_CAPTURE:+.3f}.",
                param_name="exit_confidence_threshold", delta=DELTA_EXIT_CONF_CAPTURE,
            ))

        # WTL penalty signal too noisy / large
        if m.mean_wtl_penalty < WTL_PENALTY_EXCESSIVE:
            anomalies.append(Anomaly(
                code="WTL_PENALTY_EXCESSIVE",
                severity="INFO",
                symbol=m.symbol, timeframe=m.timeframe,
                message=f"{key}: mean WTL penalty={m.mean_wtl_penalty:.3f} < {WTL_PENALTY_EXCESSIVE}. Reward dominated by penalty — reduce wtl_penalty_multiplier.",
                metric_value=m.mean_wtl_penalty,
                threshold=WTL_PENALTY_EXCESSIVE,
                correction=f"Reduce wtl_penalty_multiplier by {DELTA_WTL_MULT_REDUCE:+.3f}.",
                param_name="wtl_penalty_multiplier", delta=DELTA_WTL_MULT_REDUCE,
            ))

        # Runway predictor accuracy drifting
        if m.runway_accuracy < RUNWAY_ACCURACY_LOW and m.runway_accuracy > 0:
            anomalies.append(Anomaly(
                code="RUNWAY_ACCURACY_LOW",
                severity="INFO",
                symbol=m.symbol, timeframe=m.timeframe,
                message=f"{key}: runway_accuracy={m.runway_accuracy:.3f} < {RUNWAY_ACCURACY_LOW}. Predictor under-calibrated — speed up adaptation.",
                metric_value=m.runway_accuracy,
                threshold=RUNWAY_ACCURACY_LOW,
                correction=f"Raise runway_cal_alpha by {DELTA_RUNWAY_CAL_ALPHA:+.3f}.",
                param_name="runway_cal_alpha", delta=DELTA_RUNWAY_CAL_ALPHA,
            ))

        # PnL alignment signal weak
        if m.mean_pnl_alignment < PNL_ALIGNMENT_WEAK and m.n_trades >= CAPTURE_MIN_TRADES:
            anomalies.append(Anomaly(
                code="PNL_ALIGNMENT_WEAK",
                severity="INFO",
                symbol=m.symbol, timeframe=m.timeframe,
                message=f"{key}: mean pnl_alignment={m.mean_pnl_alignment:.4f} < {PNL_ALIGNMENT_WEAK}. PnL gradient too weak in reward signal.",
                metric_value=m.mean_pnl_alignment,
                threshold=PNL_ALIGNMENT_WEAK,
                correction=f"Raise pnl_alignment_multiplier by {DELTA_PNL_ALIGN_MULT:+.3f}.",
                param_name="pnl_alignment_multiplier", delta=DELTA_PNL_ALIGN_MULT,
            ))

    return anomalies


# ── corrections ───────────────────────────────────────────────────────────────

def apply_corrections(
    anomalies: list[Anomaly],
    auto_heal: bool,
    verbose: bool,
) -> list[str]:
    applied: list[str] = []
    if not auto_heal:
        return applied

    try:
        from src.persistence.learned_parameters import LearnedParametersManager
    except ImportError:
        LOG.error("Cannot import LearnedParametersManager — no corrections applied")
        return applied

    mgr = LearnedParametersManager(DATA_DIR / "learned_parameters.json")

    # De-duplicate: per (symbol, timeframe, param_name) apply only the largest delta
    dedup: dict[tuple[str, str, str], Anomaly] = {}
    for a in anomalies:
        if not a.param_name or a.delta == 0.0:
            continue
        key = (a.symbol, a.timeframe, a.param_name)
        if key not in dedup or abs(a.delta) > abs(dedup[key].delta):
            dedup[key] = a

    for (symbol, timeframe, param_name), a in dedup.items():
        if symbol == "FLEET":
            # Apply to all active bots for fleet-wide corrections
            try:
                targets = [
                    (s.split("_")[0], "_".join(s.split("_")[1:]))
                    for s in mgr.instruments
                ]
            except Exception:
                continue
        else:
            targets = [(symbol, timeframe)]

        for sym, tf in targets:
            try:
                current = mgr.get_param(sym, param_name, tf)
                new_val = current + a.delta
                # get_param may not exist — use set_value which clamps to bounds
                mgr.set_value(sym, param_name, new_val, timeframe=tf)
                final = mgr.get_param(sym, param_name, tf)
                msg = (f"[{a.code}] {sym} {tf}: {param_name} "
                       f"{current:.4f} → {final:.4f} (Δ{a.delta:+.4f})")
                applied.append(msg)
                if verbose:
                    LOG.info("CORRECTION: %s", msg)
            except Exception as e:
                LOG.warning("Could not apply correction for %s %s %s: %s", sym, tf, param_name, e)

    if applied:
        try:
            mgr.save()
            LOG.info("Saved %d correction(s) to learned_parameters.json", len(applied))
        except Exception as e:
            LOG.error("Failed to save learned_parameters.json: %s", e)

    return applied


# ── report ────────────────────────────────────────────────────────────────────

def generate_report(
    fleet: FleetMetrics,
    bot_metrics: dict[str, BotMetrics],
    anomalies: list[Anomaly],
    applied: list[str],
) -> dict[str, Any]:
    overall_health = "HEALTHY"
    if any(a.severity == "CRITICAL" for a in anomalies):
        overall_health = "CRITICAL"
    elif any(a.severity == "WARNING" for a in anomalies):
        overall_health = "WARNING"

    report: dict[str, Any] = {
        "generated_at": fleet.generated_at,
        "analysis_window_hours": fleet.analysis_window_hours,
        "overall_health": overall_health,
        "fleet": {
            "n_trades":               fleet.n_trades,
            "total_pnl":              round(fleet.total_pnl, 4),
            "win_rate":               round(fleet.fleet_wr, 4),
            "avg_winner":             round(fleet.avg_winner, 4),
            "avg_loser":              round(fleet.avg_loser, 4),
            "ddqn_win_rate":          round(fleet.ddqn_wr, 4),
            "runway_capture_win_rate": round(fleet.runway_wr, 4),
            "emergency_rate":         round(fleet.emergency_rate, 4),
            "mean_capture":           round(fleet.mean_capture, 4),
            "mean_trigger_reward":    round(fleet.mean_trigger_reward, 4),
            "mean_wtl_penalty":       round(fleet.mean_wtl_penalty, 4),
            "trigger_saturation_pct": round(fleet.trigger_saturation_pct, 4),
            "runway_accuracy":        round(fleet.runway_accuracy, 4),
        },
        "bots": {
            key: {
                "n_trades":         m.n_trades,
                "win_rate":         round(m.win_rate, 4),
                "total_pnl":        round(m.total_pnl, 4),
                "expectancy":       round(m.expectancy, 4),
                "avg_winner":       round(m.avg_winner, 4),
                "avg_loser":        round(m.avg_loser, 4),
                "ddqn_wr":          round(m.ddqn_win_rate, 4),
                "ddqn_n":           m.n_ddqn,
                "runway_wr":        round(m.runway_win_rate, 4),
                "runway_n":         m.n_runway_capture,
                "emergency_n":      m.n_emergency,
                "emergency_rate":   round(m.emergency_rate, 4),
                "mean_capture":     round(m.mean_capture, 4),
                "runway_accuracy":  round(m.runway_accuracy, 4),
                "pnl_alignment":    round(m.mean_pnl_alignment, 4),
                "wtl_penalty":      round(m.mean_wtl_penalty, 4),
                "trigger_at_rail":  round(m.trigger_at_rail, 4),
                "anomalies":        m.anomalies,
            }
            for key, m in sorted(bot_metrics.items())
        },
        "anomalies": [
            {
                "code":         a.code,
                "severity":     a.severity,
                "symbol":       a.symbol,
                "timeframe":    a.timeframe,
                "message":      a.message,
                "metric_value": round(a.metric_value, 6),
                "threshold":    round(a.threshold, 6),
                "correction":   a.correction,
                "param_name":   a.param_name,
                "delta":        a.delta,
            }
            for a in anomalies
        ],
        "corrections_applied": applied,
    }
    return report


def _write_health_report(report: dict[str, Any]) -> None:
    tmp = HEALTH_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        tmp.replace(HEALTH_FILE)
    except Exception as e:
        LOG.error("Failed to write health report: %s", e)


def _print_summary(fleet: FleetMetrics, bot_metrics: dict[str, BotMetrics],
                   anomalies: list[Anomaly], applied: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"PERFORMANCE HEALTH — last {fleet.analysis_window_hours:.0f}h")
    print(f"{'='*60}")
    print(f"Trades:   {fleet.n_trades}  WR={fleet.fleet_wr:.0%}  PnL=${fleet.total_pnl:.2f}")
    print(f"Avg W/L:  +${fleet.avg_winner:.2f} / -${abs(fleet.avg_loser):.2f}  R:R={abs(fleet.avg_loser)/max(fleet.avg_winner,0.01):.2f}×")
    print(f"DDQN WR:  {fleet.ddqn_wr:.0%}   Runway WR: {fleet.runway_wr:.0%}")
    print(f"Emergency:{fleet.emergency_rate:.1%}  Capture:{fleet.mean_capture:.3f}  RunwayAcc:{fleet.runway_accuracy:.3f}")
    print(f"Trigger@rail:{fleet.trigger_saturation_pct:.0%}")
    print()
    print("Per-bot:")
    for key, m in sorted(bot_metrics.items()):
        flag = " ⚠" if m.anomalies else ""
        print(f"  {key:20s} {m.n_trades:3d} trades  WR={m.win_rate:.0%}  PnL=${m.total_pnl:+.2f}"
              f"  DDQN={m.ddqn_win_rate:.0%}({m.n_ddqn})  RW={m.runway_win_rate:.0%}({m.n_runway_capture}){flag}")
    if anomalies:
        print(f"\nAnomalies ({len(anomalies)}):")
        for a in anomalies:
            print(f"  [{a.severity}] {a.code} — {a.symbol} {a.timeframe}: {a.message}")
    else:
        print("\nNo anomalies detected.")
    if applied:
        print(f"\nCorrections applied ({len(applied)}):")
        for c in applied:
            print(f"  {c}")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def run_analysis(
    hours: float = 24.0,
    auto_heal: bool = False,
    min_trades: int = 5,
    verbose: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    trades = load_recent_trades(hours)
    if not trades:
        LOG.warning("No trades found in the last %.0fh", hours)
        report = {"generated_at": datetime.now(timezone.utc).isoformat(),
                  "analysis_window_hours": hours, "overall_health": "NO_DATA",
                  "fleet": {}, "bots": {}, "anomalies": [], "corrections_applied": []}
        _write_health_report(report)
        return report

    bot_metrics = compute_bot_metrics(trades)
    fleet       = compute_fleet_metrics(trades, bot_metrics, hours)
    anomalies   = detect_anomalies(bot_metrics, fleet, min_trades)

    # Tag bots with their anomaly codes
    for a in anomalies:
        key = f"{a.symbol}_{a.timeframe}"
        if key in bot_metrics:
            bot_metrics[key].anomalies.append(a.code)

    applied = apply_corrections(anomalies, auto_heal=auto_heal, verbose=verbose)
    report  = generate_report(fleet, bot_metrics, anomalies, applied)
    _write_health_report(report)

    if not quiet:
        _print_summary(fleet, bot_metrics, anomalies, applied)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Performance analyzer & self-healing system")
    parser.add_argument("--hours",       type=float, default=24.0,  help="Analysis window in hours (default 24)")
    parser.add_argument("--auto-heal",   action="store_true",        help="Apply corrective adjustments to learned_parameters.json")
    parser.add_argument("--min-trades",  type=int,   default=5,      help="Min trades before anomaly fires (default 5)")
    parser.add_argument("--verbose",     action="store_true",        help="Log each correction detail")
    parser.add_argument("--quiet",       action="store_true",        help="Skip human-readable summary")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    report = run_analysis(
        hours=args.hours,
        auto_heal=args.auto_heal,
        min_trades=args.min_trades,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    health = report.get("overall_health", "UNKNOWN")
    print(f"Health: {health}  →  {HEALTH_FILE}")
    return 0 if health in ("HEALTHY", "NO_DATA") else 1


if __name__ == "__main__":
    sys.exit(main())
