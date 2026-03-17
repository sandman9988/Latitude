"""
metrics_calculator
==================
Single source of truth for period performance metrics (Sharpe, Sortino,
profit factor, win rate, max drawdown, capture ratio, etc.).

Used by:
  - ctrader_ddqn_paper._build_performance_snapshot (via period_metrics)
  - hud_tabbed (via period_metrics)
"""

from __future__ import annotations

import math

_EMPTY: dict = {
    "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
    "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown": 0.0,
    "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
    "expectancy": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
    "avg_trade": 0.0,
    "max_consec_wins": 0, "max_consec_losses": 0,
    "winner_to_loser_count": 0,
    "avg_mfe": 0.0, "avg_mae": 0.0, "avg_capture_ratio": 0.0,
    "avg_conf_win": 0.0, "avg_conf_loss": 0.0,
    "avg_bars_held": 0.0,
    "recent_pnl_sequence": [],
}


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

    pnls = [t.get("pnl", 0.0) for t in pts]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    total_pnl = sum(pnls)
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    mean_p = total_pnl / n
    variance = sum((p - mean_p) ** 2 for p in pnls) / n
    std_p = math.sqrt(variance) if variance > 0 else 0.0
    sharpe = mean_p / std_p if std_p > 0 else 0.0

    # Sortino: downside deviation uses losses-only count, not all-trade count.
    n_losses = max(1, len(losses))
    down_var = sum(p ** 2 for p in pnls if p < 0) / n_losses
    sortino = mean_p / math.sqrt(down_var) if down_var > 0 else 0.0

    # Max drawdown
    cum = 0.0
    peak_equity = starting_equity
    max_dd_pct = 0.0
    for p in pnls:
        cum += p
        equity = starting_equity + cum
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            dd_pct = (peak_equity - equity) / peak_equity * 100.0
            max_dd_pct = max(max_dd_pct, dd_pct)

    # Consecutive wins / losses
    max_cw = max_cl = cw = cl = 0
    for p in pnls:
        if p > 0:
            cw += 1
            cl = 0
        else:
            cl += 1
            cw = 0
        max_cw = max(max_cw, cw)
        max_cl = max(max_cl, cl)

    # Winner-to-loser count
    w2l_count = sum(1 for t in pts if isinstance(t, dict) and t.get("winner_to_loser"))

    # Edge quality: capture ratio, avg MFE, avg MAE
    # Convert legacy price-point MFE/MAE to dollar values when possible.
    _mfe_vals: list[float] = []
    _mae_vals: list[float] = []
    for t in pts:
        if not isinstance(t, dict):
            continue
        _m = t.get("mfe", 0.0)
        _a = t.get("mae", 0.0)
        _qty = t.get("quantity", 0)
        _ep = t.get("entry_price", 0)
        _ex = t.get("exit_price", 0)
        _pnl_t = t.get("pnl", 0.0)
        # Detect price-point units: if |pnl/mfe| >> 1, MFE is likely raw points
        _needs_convert = (
            _m > 0 and _qty > 0 and _ep > 0 and _ex > 0
            and abs(_pnl_t / _m) > 5.0
        )
        if _needs_convert:
            _pd = abs(_ex - _ep)
            if _pd > 0:
                _cs = abs(_pnl_t) / (_pd * _qty)
                _m = _m * _qty * _cs
                _a = _a * _qty * _cs
        _mfe_vals.append(_m)
        _mae_vals.append(_a)
    avg_mfe = sum(_mfe_vals) / len(_mfe_vals) if _mfe_vals else 0.0
    avg_mae = sum(_mae_vals) / len(_mae_vals) if _mae_vals else 0.0

    _captures: list[float] = []
    for t in pts:
        if not isinstance(t, dict):
            continue
        _mfe = t.get("mfe", 0.0)
        _pnl = t.get("pnl", 0.0)
        if _mfe <= 0:
            continue
        # Legacy trades stored MFE in price points while PnL is in dollars.
        # Detect and correct: if |pnl/mfe| >> 1 and quantity/entry_price
        # are available, convert MFE from price points to dollars.
        _ratio = _pnl / _mfe
        if abs(_ratio) > 5.0:
            _qty = t.get("quantity", 0)
            _ep = t.get("entry_price", 0)
            if _qty > 0 and _ep > 0:
                # Infer contract_size from pnl / (price_diff * qty)
                _exit_p = t.get("exit_price", 0)
                _price_diff = abs(_exit_p - _ep) if _exit_p else 0
                if _price_diff > 0:
                    _cs = abs(_pnl) / (_price_diff * _qty)
                    _mfe_dollar = _mfe * _qty * _cs
                    _ratio = _pnl / _mfe_dollar
        _captures.append(_ratio)
    # Clip per-trade ratios to [-5, 5] before averaging to prevent extreme
    # outliers (tiny-MFE trades that reversed hard) from dominating the mean.
    _clipped = [max(-5.0, min(5.0, c)) for c in _captures]
    avg_capture_ratio = sum(_clipped) / len(_clipped) if _clipped else 0.0

    # Confidence calibration
    _conf_wins: list[float] = []
    _conf_losses: list[float] = []
    for t in pts:
        if not isinstance(t, dict):
            continue
        _ec = t.get("entry_confidence")
        if _ec is None:
            continue
        if t.get("pnl", 0.0) > 0:
            _conf_wins.append(_ec)
        else:
            _conf_losses.append(_ec)
    avg_conf_win = sum(_conf_wins) / len(_conf_wins) if _conf_wins else 0.0
    avg_conf_loss = sum(_conf_losses) / len(_conf_losses) if _conf_losses else 0.0

    # Bars held — fall back to computing from entry/exit timestamps for legacy data
    _bars: list[int] = []
    for t in pts:
        if not isinstance(t, dict):
            continue
        bh = t.get("bars_held")
        if bh is not None and bh > 0:
            _bars.append(bh)
        else:
            # Compute from timestamps if available
            _et = t.get("entry_time")
            _xt = t.get("exit_time")
            if _et and _xt:
                try:
                    from datetime import datetime  # noqa: PLC0415

                    _entry_dt = datetime.fromisoformat(str(_et))
                    _exit_dt = datetime.fromisoformat(str(_xt))
                    _dur_secs = (_exit_dt - _entry_dt).total_seconds()
                    _tf_secs = 5 * 60  # default M5
                    _computed = max(1, int(_dur_secs / _tf_secs))
                    _bars.append(_computed)
                except (ValueError, TypeError):
                    pass
    avg_bars_held = sum(_bars) / len(_bars) if _bars else 0.0

    return {
        "total_trades": n, "win_rate": win_rate, "total_pnl": total_pnl,
        "sharpe_ratio": sharpe, "sortino_ratio": sortino,
        "max_drawdown": max_dd_pct,
        "best_trade": max(pnls), "worst_trade": min(pnls), "avg_trade": mean_p,
        "profit_factor": min(profit_factor, 99.0), "expectancy": expectancy,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "max_consec_wins": max_cw, "max_consec_losses": max_cl,
        "winner_to_loser_count": w2l_count,
        "avg_mfe": avg_mfe, "avg_mae": avg_mae,
        "avg_capture_ratio": avg_capture_ratio,
        "avg_conf_win": avg_conf_win, "avg_conf_loss": avg_conf_loss,
        "avg_bars_held": avg_bars_held,
        "recent_pnl_sequence": pnls,
    }
