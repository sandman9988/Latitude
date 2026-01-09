#!/usr/bin/env python3
"""
Trade Exporter - Export trading history to CSV format
Exports performance tracker data for offline analysis in Excel, pandas, etc.
"""

import csv
import datetime as dt
from pathlib import Path
from typing import List, Dict, Optional


class TradeExporter:
    """Export trade history to CSV format."""
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_trades(self, trades: List[Dict], filename: Optional[str] = None) -> str:
        """
        Export list of trades to CSV file.
        
        Args:
            trades: List of trade dictionaries from PerformanceTracker
            filename: Optional custom filename. If None, auto-generates with timestamp
            
        Returns:
            Path to created CSV file
        """
        if not trades:
            raise ValueError("No trades to export")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
            filename = f"trades_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Define CSV columns
        fieldnames = [
            'trade_num',
            'entry_time',
            'exit_time',
            'duration_seconds',
            'direction',
            'entry_price',
            'exit_price',
            'price_change',
            'pnl',
            'pnl_percent',
            'mfe',
            'mae',
            'mfe_percent',
            'mae_percent',
            'capture_efficiency',
            'winner_to_loser',
            'equity_after',
            'result'
        ]
        
        # Write CSV
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for trade in trades:
                # Calculate derived metrics
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds()
                price_change = trade['exit_price'] - trade['entry_price']
                pnl_percent = (trade['pnl'] / trade['entry_price']) * 100 if trade['entry_price'] > 0 else 0
                mfe_percent = (trade['mfe'] / trade['entry_price']) * 100 if trade['entry_price'] > 0 else 0
                mae_percent = (trade['mae'] / trade['entry_price']) * 100 if trade['entry_price'] > 0 else 0
                
                # Capture efficiency: how much of MFE was captured as PnL
                capture_efficiency = (trade['pnl'] / trade['mfe']) if trade['mfe'] > 0 else 0
                
                # Trade result
                result = 'WIN' if trade['pnl'] > 0 else 'LOSS' if trade['pnl'] < 0 else 'BREAKEVEN'
                
                # Format row
                row = {
                    'trade_num': trade['trade_num'],
                    'entry_time': trade['entry_time'].isoformat(),
                    'exit_time': trade['exit_time'].isoformat(),
                    'duration_seconds': f"{duration:.1f}",
                    'direction': trade['direction'],
                    'entry_price': f"{trade['entry_price']:.2f}",
                    'exit_price': f"{trade['exit_price']:.2f}",
                    'price_change': f"{price_change:.2f}",
                    'pnl': f"{trade['pnl']:.2f}",
                    'pnl_percent': f"{pnl_percent:.4f}",
                    'mfe': f"{trade['mfe']:.2f}",
                    'mae': f"{trade['mae']:.2f}",
                    'mfe_percent': f"{mfe_percent:.4f}",
                    'mae_percent': f"{mae_percent:.4f}",
                    'capture_efficiency': f"{capture_efficiency:.4f}",
                    'winner_to_loser': 'TRUE' if trade.get('winner_to_loser', False) else 'FALSE',
                    'equity_after': f"{trade['equity_after']:.2f}",
                    'result': result
                }
                
                writer.writerow(row)
        
        return str(filepath)
    
    def export_summary(self, metrics: Dict, filename: Optional[str] = None) -> str:
        """
        Export performance summary to CSV file.
        
        Args:
            metrics: Performance metrics dictionary from PerformanceTracker
            filename: Optional custom filename. If None, auto-generates with timestamp
            
        Returns:
            Path to created CSV file
        """
        # Generate filename if not provided
        if filename is None:
            timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
            filename = f"performance_summary_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Write summary as key-value pairs
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            
            # Format metrics
            summary_rows = [
                ['Total Trades', metrics['total_trades']],
                ['Winning Trades', metrics['winning_trades']],
                ['Losing Trades', metrics['losing_trades']],
                ['Win Rate', f"{metrics['win_rate']*100:.2f}%"],
                ['Total PnL', f"${metrics['total_pnl']:.2f}"],
                ['Average Winner', f"${metrics['avg_winner']:.2f}"],
                ['Average Loser', f"${metrics['avg_loser']:.2f}"],
                ['Profit Factor', f"{metrics['profit_factor']:.2f}"],
                ['Expectancy', f"${metrics['expectancy']:.2f}"],
                ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.3f}"],
                ['Initial Equity', f"${metrics['initial_equity']:.2f}"],
                ['Current Equity', f"${metrics['current_equity']:.2f}"],
                ['Total Return', f"{metrics['total_return']*100:.2f}%"],
                ['Max Drawdown', f"{metrics['max_drawdown']*100:.2f}%"],
                ['Current Drawdown', f"{metrics['current_drawdown']*100:.2f}%"],
                ['Max Consecutive Wins', metrics['max_consecutive_wins']],
                ['Max Consecutive Losses', metrics['max_consecutive_losses']],
                ['Winner-to-Loser Count', metrics['winner_to_loser_count']]
            ]
            
            writer.writerows(summary_rows)
        
        return str(filepath)
    
    def export_all(self, performance_tracker, prefix: Optional[str] = None) -> Dict[str, str]:
        """
        Export both trades and summary from a PerformanceTracker instance.
        
        Args:
            performance_tracker: PerformanceTracker instance
            prefix: Optional prefix for filenames
            
        Returns:
            Dictionary with paths to created files
        """
        timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        
        if prefix:
            trades_file = f"{prefix}_trades_{timestamp}.csv"
            summary_file = f"{prefix}_summary_{timestamp}.csv"
        else:
            trades_file = None
            summary_file = None
        
        results = {}
        
        # Export trades if any exist
        trades = performance_tracker.get_trade_history()
        if trades:
            results['trades'] = self.export_trades(trades, trades_file)
        
        # Export summary
        metrics = performance_tracker.get_metrics()
        results['summary'] = self.export_summary(metrics, summary_file)
        
        return results
