#!/usr/bin/env python3
"""
Performance Tracker - Real-time trading performance metrics
Tracks Sharpe ratio, win rate, drawdown, and other key statistics.
"""

import datetime as dt
import math
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class AgentAttribution:
    """Phase 3.3 dual-agent attribution for a completed trade.

    Validation is applied in __post_init__:
    - Numeric fields clamped to valid ranges
    - Quality strings fall back to "N/A" if unrecognised
    - Bar-offset ints clamped to >= -1
    """

    _VALID_TRIGGER: ClassVar[frozenset] = frozenset(
        {"EXCELLENT", "GOOD", "OVERPREDICTED", "UNDERPREDICTED", "N/A"}
    )
    _VALID_HARVESTER: ClassVar[frozenset] = frozenset(
        {"EXCELLENT", "GOOD", "FAIR", "POOR", "POOR_WTL", "STOPPED_OUT", "N/A"}
    )

    predicted_runway: float = 0.0
    runway_utilization: float = 0.0
    runway_error_pct: float = 0.0
    trigger_quality: str = "N/A"
    harvester_quality: str = "N/A"
    mfe_bar_offset: int = -1
    mae_bar_offset: int = -1
    bars_from_mfe_to_exit: int = -1

    def __post_init__(self) -> None:
        self.predicted_runway = max(0.0, self.predicted_runway or 0.0)
        self.runway_utilization = max(0.0, min(10.0, self.runway_utilization or 0.0))
        self.runway_error_pct = max(0.0, min(1000.0, self.runway_error_pct or 0.0))
        if self.trigger_quality not in self._VALID_TRIGGER:
            self.trigger_quality = "N/A"
        if self.harvester_quality not in self._VALID_HARVESTER:
            self.harvester_quality = "N/A"
        self.mfe_bar_offset = max(-1, self.mfe_bar_offset if self.mfe_bar_offset is not None else -1)
        self.mae_bar_offset = max(-1, self.mae_bar_offset if self.mae_bar_offset is not None else -1)
        self.bars_from_mfe_to_exit = max(
            -1, self.bars_from_mfe_to_exit if self.bars_from_mfe_to_exit is not None else -1
        )


class PerformanceTracker:
    """Track real-time trading performance metrics."""

    # Minimum trades required for statistical metrics
    MIN_TRADES_FOR_METRICS: int = 2

    def __init__(self):
        self.trades: list[dict] = []
        self.equity_curve: list[tuple] = []  # (timestamp, equity)
        self.initial_equity = 10000.0  # Starting capital
        self.current_equity = self.initial_equity

        # Running statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_winner_pnl = 0.0
        self.total_loser_pnl = 0.0

        # Drawdown tracking
        self.peak_equity = self.initial_equity
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

        # Consecutive tracking
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0

    def add_trade(
        self,
        pnl: float,
        entry_time: dt.datetime,
        exit_time: dt.datetime,
        direction: str,
        entry_price: float,
        exit_price: float,
        mfe: float = 0.0,
        mae: float = 0.0,
        winner_to_loser: bool = False,
        attribution: AgentAttribution | None = None,
    ):
        """Record a completed trade.

        Phase 3.3: Pass an ``AgentAttribution`` instance for dual-agent metrics.
        All individual Phase 3.3 fields are bundled there to keep the signature
        within SonarQube's parameter-count limit.
        """
        pnl, mfe, mae = self._sanitize_trade_inputs(pnl, mfe, mae)
        attr = attribution or AgentAttribution()

        self.total_trades += 1
        self.total_pnl += pnl
        self.current_equity += pnl

        self._update_win_loss_streaks(pnl > 0, pnl)
        self._update_drawdown()

        # Store trade record (Phase 3.3: Extended with dual-agent metrics)
        self.trades.append(
            {
                "trade_num": self.total_trades,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "mfe": mfe,
                "mae": mae,
                "winner_to_loser": winner_to_loser,
                "equity_after": self.current_equity,
                # Phase 3.3: Dual-agent attribution
                "predicted_runway": attr.predicted_runway,
                "runway_utilization": attr.runway_utilization,
                "runway_error_pct": attr.runway_error_pct,
                "trigger_quality": attr.trigger_quality,
                "harvester_quality": attr.harvester_quality,
                "mfe_bar_offset": attr.mfe_bar_offset,
                "mae_bar_offset": attr.mae_bar_offset,
                "bars_from_mfe_to_exit": attr.bars_from_mfe_to_exit,
            }
        )
        self.equity_curve.append((exit_time, self.current_equity))

    @staticmethod
    def _sanitize_trade_inputs(pnl: object, mfe: object, mae: object) -> tuple[float, float, float]:
        """Defensive sanitization of core numeric trade inputs."""
        if pnl is None or not isinstance(pnl, (int, float)):
            pnl = 0.0
        if mfe is None or not isinstance(mfe, (int, float)) or mfe < 0:
            mfe = 0.0
        if mae is None or not isinstance(mae, (int, float)) or mae < 0:
            mae = 0.0
        return float(pnl), float(mfe), float(mae)

    def _update_win_loss_streaks(self, is_winner: bool, pnl: float) -> None:
        """Update win/loss tallies and consecutive-streak counters."""
        if is_winner:
            self.winning_trades += 1
            self.total_winner_pnl += pnl
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
        else:
            self.losing_trades += 1
            self.total_loser_pnl += pnl
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)

    def _update_drawdown(self) -> None:
        """Recompute current_drawdown against peak equity after an equity change."""
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            self.current_drawdown = 0.0
        elif self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        else:
            self.current_drawdown = 0.0

    def get_metrics(self) -> dict:
        """Calculate and return current performance metrics."""

        if self.total_trades == 0:
            return self._empty_metrics()

        # Win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

        # Average winner/loser
        avg_winner = self.total_winner_pnl / self.winning_trades if self.winning_trades > 0 else 0.0
        avg_loser = self.total_loser_pnl / self.losing_trades if self.losing_trades > 0 else 0.0

        # Profit factor
        gross_profit = self.total_winner_pnl
        gross_loss = abs(self.total_loser_pnl)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy
        expectancy = (win_rate * avg_winner) + ((1 - win_rate) * avg_loser)

        # Sharpe ratio (simplified - assuming returns are trade PnLs)
        sharpe_ratio = self._calculate_sharpe()

        # Sortino ratio (downside risk only - from handbook)
        sortino_ratio = self._calculate_sortino()

        # Omega ratio (probability-weighted gains/losses - from handbook)
        omega_ratio = self._calculate_omega()

        # Return on capital (defensive: protect against zero initial equity)
        total_return = (
            (self.current_equity - self.initial_equity) / self.initial_equity if self.initial_equity > 0 else 0.0
        )

        # Winner-to-loser count
        wtl_count = sum(1 for t in self.trades if t.get("winner_to_loser", False))

        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "sortino_ratio": sortino_ratio,
            "omega_ratio": omega_ratio,
            "total_pnl": self.total_pnl,
            "avg_winner": avg_winner,
            "avg_loser": avg_loser,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "sharpe_ratio": sharpe_ratio,
            "initial_equity": self.initial_equity,
            "current_equity": self.current_equity,
            "total_return": total_return,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "winner_to_loser_count": wtl_count,
        }

    def _calculate_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from trade returns."""
        if len(self.trades) < self.MIN_TRADES_FOR_METRICS:
            return 0.0

        # Calculate returns as % of equity before trade
        returns = []
        equity = self.initial_equity
        for trade in self.trades:
            ret = trade["pnl"] / equity if equity > 0 else 0.0
            returns.append(ret)
            equity = trade["equity_after"]

        # Mean and std of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        # Sharpe ratio (annualized factor not applied - depends on timeframe)
        sharpe = (mean_return - risk_free_rate) / std_dev if std_dev > 0 else 0.0
        return sharpe

    def _calculate_sortino(self, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(self.trades) < self.MIN_TRADES_FOR_METRICS:
            return 0.0

        # Calculate returns
        returns = []
        equity = self.initial_equity
        for trade in self.trades:
            ret = trade["pnl"] / equity if equity > 0 else 0.0
            returns.append(ret)
            equity = trade["equity_after"]

        # Mean return
        mean_return = sum(returns) / len(returns)

        # Downside deviation (only negative deviations from target)
        downside_diffs = [min(0, r - target_return) for r in returns]
        downside_variance = sum(d**2 for d in downside_diffs) / len(downside_diffs)
        downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0.0

        # Sortino ratio
        sortino = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0
        return sortino

    def _calculate_omega(self, threshold: float = 0.0) -> float:
        """Calculate Omega ratio (probability weighted ratio of gains vs losses)."""
        if len(self.trades) < self.MIN_TRADES_FOR_METRICS:
            return 0.0

        # Calculate returns
        returns = []
        equity = self.initial_equity
        for trade in self.trades:
            ret = trade["pnl"] / equity if equity > 0 else 0.0
            returns.append(ret)
            equity = trade["equity_after"]

        # Sum of gains above threshold
        gains = sum(max(0, r - threshold) for r in returns)

        # Sum of losses below threshold
        losses = sum(max(0, threshold - r) for r in returns)

        # Omega ratio
        omega = gains / losses if losses > 0 else float("inf")
        return omega

    def _empty_metrics(self) -> dict:
        """Return empty metrics structure."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "sortino_ratio": 0.0,
            "omega_ratio": 0.0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "sharpe_ratio": 0.0,
            "initial_equity": self.initial_equity,
            "current_equity": self.initial_equity,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "winner_to_loser_count": 0,
        }

    def print_dashboard(self) -> str:
        """Generate formatted dashboard string."""
        metrics = self.get_metrics()

        dashboard = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    PERFORMANCE DASHBOARD                         ║
╚══════════════════════════════════════════════════════════════════╝

📊 TRADING STATISTICS
   Total Trades:        {metrics['total_trades']:>6}
   Winners:             {metrics['winning_trades']:>6} ({metrics['win_rate']*100:>5.1f}%)
   Losers:              {metrics['losing_trades']:>6}

💰 P&L METRICS
   Total PnL:           ${metrics['total_pnl']:>10.2f}
   Avg Winner:          ${metrics['avg_winner']:>10.2f}
   Avg Loser:           ${metrics['avg_loser']:>10.2f}
   Profit Factor:       {metrics['profit_factor']:>10.2f}
   Expectancy:          ${metrics['expectancy']:>10.2f}

📈 EQUITY
   Initial:             ${metrics['initial_equity']:>10.2f}
   Current:             ${metrics['current_equity']:>10.2f}
   Return:              {metrics['total_return']*100:>9.2f}%

📉 RISK METRICS
   Max Drawdown:        {metrics['max_drawdown']*100:>9.2f}%
   Current Drawdown:    {metrics['current_drawdown']*100:>9.2f}%
   Sharpe Ratio:        {metrics['sharpe_ratio']:>10.3f}

🔥 STREAKS
   Max Consecutive Wins:   {metrics['max_consecutive_wins']:>3}
   Max Consecutive Losses: {metrics['max_consecutive_losses']:>3}
   Winner-to-Loser Count:  {metrics['winner_to_loser_count']:>3}

"""
        return dashboard

    def get_trade_history(self) -> list[dict]:
        """Return complete trade history."""
        return self.trades.copy()
