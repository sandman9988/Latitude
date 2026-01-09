#!/usr/bin/env python3
"""
Performance Tracker - Real-time trading performance metrics
Tracks Sharpe ratio, win rate, drawdown, and other key statistics.
"""

import datetime as dt
from typing import List, Dict, Optional
import math


class PerformanceTracker:
    """Track real-time trading performance metrics."""
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: List[tuple] = []  # (timestamp, equity)
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
        
    def add_trade(self, pnl: float, entry_time: dt.datetime, exit_time: dt.datetime,
                  direction: str, entry_price: float, exit_price: float,
                  mfe: float = 0.0, mae: float = 0.0, winner_to_loser: bool = False):
        """Record a completed trade."""
        
        self.total_trades += 1
        self.total_pnl += pnl
        self.current_equity += pnl
        
        # Track win/loss
        is_winner = pnl > 0
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
        
        # Update drawdown
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Store trade record
        trade_record = {
            'trade_num': self.total_trades,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'mfe': mfe,
            'mae': mae,
            'winner_to_loser': winner_to_loser,
            'equity_after': self.current_equity
        }
        self.trades.append(trade_record)
        
        # Update equity curve
        self.equity_curve.append((exit_time, self.current_equity))
    
    def get_metrics(self) -> Dict:
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
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_winner) + ((1 - win_rate) * avg_loser)
        
        # Sharpe ratio (simplified - assuming returns are trade PnLs)
        sharpe_ratio = self._calculate_sharpe()
        
        # Return on capital
        total_return = (self.current_equity - self.initial_equity) / self.initial_equity
        
        # Winner-to-loser count
        wtl_count = sum(1 for t in self.trades if t.get('winner_to_loser', False))
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'initial_equity': self.initial_equity,
            'current_equity': self.current_equity,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'winner_to_loser_count': wtl_count
        }
    
    def _calculate_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from trade returns."""
        if len(self.trades) < 2:
            return 0.0
        
        # Calculate returns as % of equity before trade
        returns = []
        equity = self.initial_equity
        for trade in self.trades:
            ret = trade['pnl'] / equity if equity > 0 else 0.0
            returns.append(ret)
            equity = trade['equity_after']
        
        # Mean and std of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        
        # Sharpe ratio (annualized factor not applied - depends on timeframe)
        sharpe = (mean_return - risk_free_rate) / std_dev if std_dev > 0 else 0.0
        return sharpe
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_winner': 0.0,
            'avg_loser': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'sharpe_ratio': 0.0,
            'initial_equity': self.initial_equity,
            'current_equity': self.initial_equity,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'winner_to_loser_count': 0
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
    
    def get_trade_history(self) -> List[Dict]:
        """Return complete trade history."""
        return self.trades.copy()
