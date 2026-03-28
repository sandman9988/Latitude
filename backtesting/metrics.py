"""
Performance metrics — win rate, Sharpe, drawdown, expectancy, profit factor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict
from core.math_utils import safe_div, safe_sqrt
from core.numeric import non_negative


@dataclass
class TradeMetrics:
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    avg_bars_held: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    mfe_mae_ratio: float = 0.0


def compute_metrics(trades, initial_balance: float, equity_curve: List[float]) -> Dict[str, float]:
    """Compute full performance metrics from completed trades + equity curve."""
    m = TradeMetrics()

    if not trades:
        return _to_dict(m)

    m.total_trades = len(trades)
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    m.win_rate = safe_div(len(wins), m.total_trades)
    m.avg_win = safe_div(sum(wins), len(wins)) if wins else 0.0
    m.avg_loss = safe_div(sum(losses), len(losses)) if losses else 0.0
    m.net_pnl = sum(pnls)
    m.return_pct = safe_div(m.net_pnl, initial_balance)

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    m.profit_factor = safe_div(gross_profit, gross_loss, fallback=0.0) if gross_loss > 0 else float("inf")

    m.expectancy = safe_div(m.net_pnl, m.total_trades)

    m.avg_bars_held = safe_div(sum(t.bars_held for t in trades), m.total_trades)
    m.avg_mfe = safe_div(sum(t.mfe for t in trades), m.total_trades)
    m.avg_mae = safe_div(sum(t.mae for t in trades), m.total_trades)
    m.mfe_mae_ratio = safe_div(m.avg_mfe, m.avg_mae)

    if equity_curve:
        m.max_drawdown, m.max_drawdown_pct = _max_drawdown(equity_curve)
        m.sharpe = _sharpe(equity_curve)
        m.sortino = _sortino(equity_curve)
        ann_return = m.return_pct
        m.calmar = safe_div(ann_return, m.max_drawdown_pct) if m.max_drawdown_pct > 0 else 0.0

    return _to_dict(m)


def _max_drawdown(equity: List[float]) -> tuple[float, float]:
    peak = equity[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        dd_pct = safe_div(dd, peak)
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    return max_dd, max_dd_pct


def _sharpe(equity: List[float], risk_free: float = 0.0) -> float:
    if len(equity) < 2:
        return 0.0
    rets = [safe_div(equity[i] - equity[i - 1], equity[i - 1]) for i in range(1, len(equity))]
    mean = sum(rets) / len(rets)
    std = safe_sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))
    if std <= 0:
        return 0.0
    return safe_div(mean - risk_free, std) * math.sqrt(252)


def _sortino(equity: List[float], risk_free: float = 0.0) -> float:
    if len(equity) < 2:
        return 0.0
    rets = [safe_div(equity[i] - equity[i - 1], equity[i - 1]) for i in range(1, len(equity))]
    mean = sum(rets) / len(rets)
    downside = [r for r in rets if r < risk_free]
    if not downside:
        return 0.0
    downside_std = safe_sqrt(sum(r ** 2 for r in downside) / len(downside))
    if downside_std <= 0:
        return 0.0
    return safe_div(mean - risk_free, downside_std) * math.sqrt(252)


def _to_dict(m: TradeMetrics) -> Dict[str, float]:
    return {
        "total_trades": float(m.total_trades),
        "win_rate": m.win_rate,
        "profit_factor": m.profit_factor,
        "expectancy": m.expectancy,
        "avg_win": m.avg_win,
        "avg_loss": m.avg_loss,
        "net_pnl": m.net_pnl,
        "return_pct": m.return_pct,
        "max_drawdown": m.max_drawdown,
        "max_drawdown_pct": m.max_drawdown_pct,
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "calmar": m.calmar,
        "avg_bars_held": m.avg_bars_held,
        "avg_mfe": m.avg_mfe,
        "avg_mae": m.avg_mae,
        "mfe_mae_ratio": m.mfe_mae_ratio,
    }
