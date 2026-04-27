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
