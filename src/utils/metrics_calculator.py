"""metrics_calculator.
==================
Single source of truth for period performance metrics (Sharpe, Sortino,
profit factor, win rate, max drawdown, capture ratio, etc.).

Used by:
  - ctrader_ddqn_paper._build_performance_snapshot (via period_metrics)
  - hud_tabbed (via period_metrics)
"""

from __future__ import annotations

import math

from src.utils.safe_math import SAFE_DIV_MIN

_EMPTY: dict = {
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "win_rate": 0.0,
    "total_pnl": 0.0,
    "sharpe_ratio": 0.0,
    "sortino_ratio": 0.0,
    "max_drawdown": 0.0,
    "avg_win": 0.0,
    "avg_loss": 0.0,
    "profit_factor": 0.0,
    "expectancy": 0.0,
    "best_trade": 0.0,
    "worst_trade": 0.0,
    "avg_trade": 0.0,
    "max_consec_wins": 0,
    "max_consec_losses": 0,
    "winner_to_loser_count": 0,
    "winner_to_loser_pnl": 0.0,
    "avg_mfe": 0.0,
    "avg_mae": 0.0,
    "avg_capture_ratio": 0.0,
    "avg_conf_win": 0.0,
    "avg_conf_loss": 0.0,
    "avg_bars_held": 0.0,
    "recent_pnl_sequence": [],
}


def _drawdown_and_streaks(pnls: list[float], starting_equity: float) -> tuple[float, int, int]:
    """Return (max_drawdown_pct, max_consec_wins, max_consec_losses)."""
    cum = 0.0
    peak = starting_equity
    max_dd = 0.0
    max_cw = max_cl = cw = cl = 0
    for p in pnls:
        cum += p
        equity = starting_equity + cum
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)
        if p > 0:
            cw += 1
            cl = 0
        else:
            cl += 1
            cw = 0
        max_cw = max(max_cw, cw)
        max_cl = max(max_cl, cl)
    return max_dd, max_cw, max_cl


def _winner_to_loser(trades: list) -> tuple[int, float]:
    """Return (count, total_pnl) for winner-to-loser trades."""
    count = 0
    pnl_sum = 0.0
    for t in trades:
        if isinstance(t, dict) and t.get("winner_to_loser"):
            count += 1
            pnl_sum += float(t.get("pnl") or 0.0)
    return count, pnl_sum


def _trade_lot_value(t: dict) -> float:
    """Derive qty×contract_size from pnl / price_diff. Returns 0 when undeterminable."""
    pnl = float(t.get("pnl") or 0.0)
    ep = float(t.get("entry_price") or 0.0)
    ex = float(t.get("exit_price") or 0.0)
    direction = str(t.get("direction", "")).upper()
    diff = (ex - ep) if direction == "LONG" else (ep - ex)
    if not math.isfinite(pnl) or not math.isfinite(diff):
        return 0.0
    return abs(pnl / diff) if abs(diff) > SAFE_DIV_MIN else 0.0


def _trade_excursion_and_capture(t: dict) -> tuple[float, float, float | None]:
    """Return (mfe_usd, mae_usd, capture_ratio) for one trade. mfe_usd=0 when unavailable."""
    mfe_pts = float(t.get("mfe_points") or 0.0)
    if not math.isfinite(mfe_pts) or mfe_pts <= 0:
        return 0.0, 0.0, None
    mae_pts = float(t.get("mae_points") or 0.0)
    ep = float(t.get("entry_price") or 0.0)
    ex = float(t.get("exit_price") or 0.0)
    direction = str(t.get("direction", "")).upper()
    diff = (ex - ep) if direction == "LONG" else (ep - ex)
    lot_value = _trade_lot_value(t)
    if lot_value > 0:
        mfe_usd = mfe_pts * lot_value
        return mfe_usd, mae_pts * lot_value, float(t.get("pnl") or 0.0) / mfe_usd
    if abs(diff) > SAFE_DIV_MIN:
        return 0.0, 0.0, diff / mfe_pts
    return 0.0, 0.0, None


def _edge_quality(trades: list) -> tuple[float, float, float]:
    """Return (avg_mfe_usd, avg_mae_usd, avg_capture_ratio) derived from price-unit fields.

    Uses entry/exit prices and mfe_points to avoid relying on stored mfe_usd
    or stored capture_ratio (both may be wrong due to incorrect qty×contract_size).
    """
    mfe_vals: list[float] = []
    mae_vals: list[float] = []
    captures: list[float] = []
    for t in trades:
        if not isinstance(t, dict):
            continue
        mfe_usd, mae_usd, cap = _trade_excursion_and_capture(t)
        if mfe_usd > 0:
            mfe_vals.append(mfe_usd)
            mae_vals.append(mae_usd)
        if cap is not None:
            captures.append(cap)
    avg_mfe = sum(mfe_vals) / len(mfe_vals) if mfe_vals else 0.0
    avg_mae = sum(mae_vals) / len(mae_vals) if mae_vals else 0.0
    clipped = [max(-5.0, min(5.0, c)) for c in captures]
    return avg_mfe, avg_mae, (sum(clipped) / len(clipped) if clipped else 0.0)


def _close_reason_breakdown(trades: list) -> dict[str, int]:
    """Count close reasons across a list of trade dicts."""
    from collections import Counter  # noqa: PLC0415
    return dict(Counter(t.get("close_reason", "unknown") for t in trades if isinstance(t, dict)))


def _capture_quality(trades: list) -> dict:
    """Analyze capture quality across trades. Returns EXCELLENT/GOOD/FAIR/POOR counts."""
    from collections import Counter  # noqa: PLC0415
    qual = Counter()
    for t in trades:
        if not isinstance(t, dict):
            continue
        if t.get("winner_to_loser"):
            qual["POOR_WTL"] += 1
            continue
        cap = t.get("capture_ratio")
        if cap is None or cap <= 0:
            qual["UNKNOWN"] += 1
            continue
        if cap >= 0.85:
            qual["EXCELLENT"] += 1
        elif cap >= 0.65:
            qual["GOOD"] += 1
        elif cap >= 0.40:
            qual["FAIR"] += 1
        else:
            qual["POOR"] += 1
    return dict(qual)


def _regime_breakdown(trades: list) -> dict[str, dict]:
    """Return per-regime performance: {regime: {trades, wins, pnl, wtl, mfe, mae}}."""
    result: dict[str, dict] = {}
    for t in trades:
        if not isinstance(t, dict):
            continue
        regime = t.get("regime") or t.get("trigger_data", {}).get("entry_regime", "UNKNOWN")
        if regime not in result:
            result[regime] = {"trades": 0, "wins": 0, "pnl": 0.0, "wtl": 0, "mfe": 0.0, "mae": 0.0}
        result[regime]["trades"] += 1
        result[regime]["pnl"] += float(t.get("pnl", 0) or 0)
        if float(t.get("pnl", 0) or 0) > 0:
            result[regime]["wins"] += 1
        if t.get("winner_to_loser"):
            result[regime]["wtl"] += 1
        result[regime]["mfe"] += float(t.get("mfe_points", 0) or 0)
        result[regime]["mae"] += float(t.get("mae_points", 0) or 0)
    for stats in result.values():
        n = stats["trades"]
        if n > 0:
            stats["mfe"] /= n
            stats["mae"] /= n
    return result


def _confidence_split(trades: list) -> tuple[float, float]:
    """Return (avg_conf_win, avg_conf_loss) by trade outcome."""
    wins: list[float] = []
    losses: list[float] = []
    for t in trades:
        if not isinstance(t, dict):
            continue
        ec = t.get("entry_confidence")
        if ec is None:
            continue
        (wins if t.get("pnl", 0.0) > 0 else losses).append(float(ec))
    return (
        sum(wins) / len(wins) if wins else 0.0,
        sum(losses) / len(losses) if losses else 0.0,
    )


def _bars_held_from_trade(t: dict) -> int | None:
    """Return bars_held for a single trade, derived from timestamps when not stored."""
    bh = t.get("bars_held")
    if bh is not None and bh > 0:
        return int(bh)
    dur: float | None = None
    hs = t.get("hold_seconds")
    if hs is not None and float(hs) > 0:
        dur = float(hs)
    else:
        et, xt = t.get("entry_time"), t.get("exit_time")
        if et and xt:
            try:
                from datetime import datetime  # noqa: PLC0415
                dur = (datetime.fromisoformat(str(xt)) - datetime.fromisoformat(str(et))).total_seconds()
            except (ValueError, TypeError):
                pass
    if dur and dur > 0:
        return max(1, int(dur / (5 * 60)))
    return None


def _avg_bars_held(trades: list) -> float:
    bars = [b for t in trades if isinstance(t, dict) and (b := _bars_held_from_trade(t)) is not None]
    return sum(bars) / len(bars) if bars else 0.0


def period_metrics(pts: list, starting_equity: float = 10_000.0) -> dict:
    """Compute period performance metrics from a list of trade dicts.

    Args:
        pts: List of trade dicts, each with at least a ``pnl`` key.
        starting_equity: Starting equity for drawdown calculation.

    Returns:
        Dict of performance metrics.  Returns a zero-filled dict when
        *pts* is empty so callers never need to guard against missing keys.

    """
    if not pts:
        return dict(_EMPTY)
    pts_non_ghost = [t for t in pts if not (isinstance(t, dict) and t.get("close_reason") == "GHOST_RECONCILE")]
    if not pts_non_ghost:
        return dict(_EMPTY)

    pnls = [t.get("pnl", 0.0) for t in pts_non_ghost]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    decisive_n = len(wins) + len(losses)
    total_pnl = sum(pnls)
    win_rate = (len(wins) / decisive_n) if decisive_n > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
    loss_rate = (len(losses) / decisive_n) if decisive_n > 0 else 0.0
    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

    mean_p = total_pnl / n
    variance = sum((p - mean_p) ** 2 for p in pnls) / n
    std_p = math.sqrt(variance) if variance > 0 else 0.0
    sharpe = mean_p / std_p if std_p > 0 else 0.0

    n_losses = max(1, len(losses))
    down_var = sum(p**2 for p in pnls if p < 0) / n_losses
    sortino = mean_p / math.sqrt(down_var) if down_var > 0 else 0.0

    max_dd, max_cw, max_cl = _drawdown_and_streaks(pnls, starting_equity)
    w2l_count, w2l_pnl = _winner_to_loser(pts_non_ghost)
    avg_mfe, avg_mae, avg_capture_ratio = _edge_quality(pts_non_ghost)
    avg_conf_win, avg_conf_loss = _confidence_split(pts_non_ghost)
    avg_bars = _avg_bars_held(pts_non_ghost)

    return {
        "total_trades": n,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "best_trade": max(pnls),
        "worst_trade": min(pnls),
        "avg_trade": mean_p,
        "profit_factor": min(profit_factor, 99.0),
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_consec_wins": max_cw,
        "max_consec_losses": max_cl,
        "winner_to_loser_count": w2l_count,
        "winner_to_loser_pnl": w2l_pnl,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "avg_capture_ratio": avg_capture_ratio,
        "avg_conf_win": avg_conf_win,
        "avg_conf_loss": avg_conf_loss,
        "avg_bars_held": avg_bars,
        "recent_pnl_sequence": pnls,
    }


def period_comparison(short_pts: list, long_pts: list, starting_equity: float = 10_000.0) -> dict:
    """Compare short-window vs long-window performance.

    Args:
        short_pts: Trades in the short window (e.g. last 24h).
        long_pts: Trades in the long window (e.g. last 7d).
        starting_equity: Starting equity for drawdown.

    Returns:
        Dict with short/long metrics and absolute/relative deltas.

    """
    s = period_metrics(short_pts, starting_equity)
    L = period_metrics(long_pts, starting_equity)

    def _delta(key: str) -> float | None:
        sv = s.get(key)
        lv = L.get(key)
        if sv is None or lv is None or (isinstance(lv, (int, float)) and lv == 0):
            return None
        if isinstance(sv, (int, float)) and isinstance(lv, (int, float)):
            return float(sv) - float(lv)
        return None

    def _pct(key: str) -> float | None:
        d = _delta(key)
        lv = L.get(key)
        if d is not None and isinstance(lv, (int, float)) and lv != 0:
            return d / abs(float(lv)) * 100.0
        return None

    return {
        "short": s,
        "long": L,
        "delta_pnl": _delta("total_pnl"),
        "delta_wr": _delta("win_rate"),
        "delta_sharpe": _delta("sharpe_ratio"),
        "delta_sortino": _delta("sortino_ratio"),
        "delta_avg_trade": _delta("avg_trade"),
        "delta_wtl_pct": _pct("winner_to_loser_count"),
        "delta_mfe": _delta("avg_mfe"),
        "delta_mae": _delta("avg_mae"),
        "delta_capture": _delta("avg_capture_ratio"),
        "delta_avg_win": _delta("avg_win"),
        "delta_avg_loss": _delta("avg_loss"),
        "pnl_trend": ("improving" if (_delta("total_pnl") or 0) > 0 else "degrading"),
    }


def decision_quality(trades: list) -> dict:
    """Derive decision-quality metrics from trade fields.

    Uses entry_confidence, trigger_reward, capture_reward, and close_reason
    to assess how well the agents are performing.

    Returns:
        Dict with confidence calibration, reward distribution, exit quality.

    """
    confs_pos = [t.get("entry_confidence", 0) for t in trades if isinstance(t, dict) and t.get("pnl", 0) > 0]
    confs_neg = [t.get("entry_confidence", 0) for t in trades if isinstance(t, dict) and t.get("pnl", 0) <= 0]
    trig_rews = [t.get("trigger_reward", 0) for t in trades if isinstance(t, dict) and t.get("trigger_reward") is not None]
    cap_rews = [t.get("capture_reward", 0) for t in trades if isinstance(t, dict) and t.get("capture_reward") is not None]
    reasons = _close_reason_breakdown(trades)
    qual = _capture_quality(trades)
    wtl_penalties = [t.get("reward_wtl_penalty", 0) for t in trades if isinstance(t, dict) and t.get("reward_wtl_penalty") is not None]

    return {
        "avg_conf_win": sum(confs_pos) / len(confs_pos) if confs_pos else 0.0,
        "avg_conf_loss": sum(confs_neg) / len(confs_neg) if confs_neg else 0.0,
        "conf_separation": (sum(confs_pos) / len(confs_pos) - sum(confs_neg) / len(confs_neg))
                           if confs_pos and confs_neg else 0.0,
        "avg_trigger_reward": sum(trig_rews) / len(trig_rews) if trig_rews else 0.0,
        "avg_capture_reward": sum(cap_rews) / len(cap_rews) if cap_rews else 0.0,
        "avg_wtl_penalty": sum(wtl_penalties) / len(wtl_penalties) if wtl_penalties else 0.0,
        "exit_reasons": reasons,
        "capture_quality": qual,
    }


def _trend_label(delta: float, threshold: float, up_label: str, down_label: str) -> str:
    """Classify a delta as stable/up/down relative to a threshold."""
    if abs(delta) < threshold:
        return "stable"
    return up_label if delta > 0 else down_label


def cap_trend(trades: list, window: int = 50) -> dict:
    """Compute capture-ratio trend and micro-winner / trailing-stop efficacy.

    Splits last N trades into two halves and compares capture quality.
    Used to detect protective-stop degradation before it causes losses.

    Returns:
        Dict with first-half vs second-half capture metrics.

    """
    if len(trades) < window:
        return {"samples": len(trades), "status": "insufficient"}
    recent = trades[-window:]
    mid = len(recent) // 2
    first = recent[:mid]
    second = recent[mid:]

    def _half_metrics(half: list) -> dict:
        pnls = [t.get("pnl", 0) for t in half if isinstance(t, dict)]
        wins = [p for p in pnls if p > 0]
        caps = [t.get("capture_ratio", 0) for t in half if isinstance(t, dict) and t.get("capture_ratio") is not None]
        mfes = [t.get("mfe_points", 0) for t in half if isinstance(t, dict)]
        maes = [t.get("mae_points", 0) for t in half if isinstance(t, dict)]
        return {
            "trades": len(half),
            "wr": len(wins) / len(pnls) if pnls else 0.0,
            "avg_cap": sum(caps) / len(caps) if caps else 0.0,
            "avg_mfe": sum(mfes) / len(mfes) if mfes else 0.0,
            "avg_mae": sum(maes) / len(maes) if maes else 0.0,
        }

    f = _half_metrics(first)
    s = _half_metrics(second)
    cap_delta = s.get("avg_cap", 0) - f.get("avg_cap", 0)
    return {
        "samples": window,
        "cap_delta": cap_delta,
        "first_half": f,
        "second_half": s,
        "status": _trend_label(cap_delta, 0.1, "improving", "degrading"),
        "mfe_trend": _trend_label(
            s.get("avg_mfe", 0) - f.get("avg_mfe", 0), 1.0, "rising", "falling"),
        "mae_trend": _trend_label(
            s.get("avg_mae", 0) - f.get("avg_mae", 0), 0.5, "rising", "falling"),
    }


def self_healing_metrics(trades: list, starting_equity: float = 10_000.0) -> dict:
    """Produce self-healing recommendations from trade data.

    Analyzes capture quality trend, WTL rate, P&L trajectory, and exit-reason
    distribution to detect emerging issues before they compound.

    Returns:
        Dict with overall health, degradation flags, and actionable recommendations.

    """
    if not trades:
        return {"health": "UNKNOWN", "flags": [], "recommendations": []}
    pm = period_metrics(trades, starting_equity)
    ct = cap_trend(trades)
    reasons = _close_reason_breakdown(trades)

    # Use non-ghost count for all ratio calculations
    non_ghost = [t for t in trades if not (isinstance(t, dict) and t.get("close_reason") == "GHOST_RECONCILE")]
    total = len(non_ghost)
    if total == 0:
        return {"health": "UNKNOWN", "flags": [], "recommendations": [],
                "wtl_rate": 0.0, "cap_trend": "unknown", "mfe_trend": "unknown",
                "mae_trend": "unknown", "recent_wr": 0.0, "avg_trade_pnl": 0.0,
                "profit_factor": 0.0, "keeper": "cross-period comparison"}

    flags: list[str] = []
    recs: list[str] = []
    wr = pm.get("win_rate", 0)
    wtl_count = pm.get("winner_to_loser_count", 0)

    # WTL rate check
    if total > 10 and wtl_count / total > 0.15:
        flags.append("high_wtl_rate")
        recs.append(f"WTL rate {wtl_count/total*100:.0f}% > 15% — tighten trailing stops or raise exit confidence floor")

    # Capture degradation
    if ct.get("status") == "degrading":
        flags.append("capture_degrading")
        recs.append(f"Capture ratio trending down (Δ={ct.get('cap_delta',0):.2f}) — review protective stop distances")

    # MFE/MAE ratio
    avg_mfe = pm.get("avg_mfe", 0)
    avg_mae = pm.get("avg_mae", 0)
    if avg_mfe > 0 and avg_mae > 0 and avg_mae / avg_mfe > 0.8:
        flags.append("adverse_excursion_high")
        recs.append(f"MAE/MFE ratio {avg_mae/avg_mfe:.2f} > 0.8 — entries may be poorly timed or stops too wide")

    # Profit factor check
    pf = pm.get("profit_factor", 0)
    if total > 20 and pf < 1.0:
        flags.append("negative_expectancy")
        recs.append("Profit factor < 1.0 — consider reducing entry frequency or raising confidence floor")

    # P&L trajectory
    if total > 10:
        recent_pnls = trades[-min(10, total):]
        recent_win = sum(1 for t in recent_pnls if isinstance(t, dict) and t.get("pnl", 0) > 0)
        if recent_win / len(recent_pnls) < 0.4:
            flags.append("recent_loss_streak")
            recs.append(f"Last {len(recent_pnls)} trades: {recent_win}/{len(recent_pnls)} wins — check market regime or reduce position size")

    # Max-loss cap frequency
    max_loss_count = reasons.get("max_loss_cap", 0)
    if total > 20 and max_loss_count / total > 0.05:
        flags.append("frequent_max_loss")
        recs.append(f"max_loss_cap in {max_loss_count}/{total} trades ({max_loss_count/total*100:.0f}%) — entries too aggressive for current volatility")

    health = "HEALTHY"
    if len(flags) >= 2:
        health = "DEGRADING"
    elif len(flags) >= 1:
        health = "CAUTION"

    return {
        "health": health,
        "flags": flags,
        "recommendations": recs,
        "wtl_rate": wtl_count / total if total > 0 else 0.0,
        "cap_trend": ct.get("status", "unknown"),
        "mfe_trend": ct.get("mfe_trend", "unknown"),
        "mae_trend": ct.get("mae_trend", "unknown"),
        "recent_wr": wr,
        "avg_trade_pnl": pm.get("avg_trade", 0),
        "profit_factor": pf,
        "keeper": "cross-period comparison",
    }
