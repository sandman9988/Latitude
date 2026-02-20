#!/usr/bin/env python3
"""
Trade Exporter - Export trading history to CSV format
Exports performance tracker data for offline analysis in Excel, pandas, etc.
"""

import csv
import datetime as dt
import logging
from pathlib import Path

from src.utils.safe_math import SafeMath

LOG = logging.getLogger(__name__)


class TradeExporter:
    """Export trade history to CSV format."""

    def __init__(self, output_dir: str = "trades"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def export_trades(self, trades: list[dict], filename: str | None = None) -> str:
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

        # Define CSV columns (Phase 3.3: Added dual-agent attribution metrics)
        fieldnames = [
            "trade_num",
            "entry_time",
            "exit_time",
            "duration_seconds",
            "direction",
            "entry_price",
            "exit_price",
            "price_change",
            "pnl",
            "pnl_percent",
            "mfe",
            "mae",
            "mfe_percent",
            "mae_percent",
            "capture_efficiency",
            "winner_to_loser",
            "equity_after",
            "result",
            # Phase 3.3: Dual-agent attribution
            "predicted_runway",
            "runway_utilization",
            "runway_error_pct",
            "trigger_quality",
            "harvester_quality",
            "mfe_bar_offset",
            "mae_bar_offset",
            "bars_from_mfe_to_exit",
        ]

        # Write CSV
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for trade in trades:

                # Defensive: Validate required fields
                if "entry_time" not in trade or "exit_time" not in trade:
                    LOG.warning("Trade %d missing timestamps. Skipping.", trade.get("trade_num", -1))
                    continue

                # Defensive: Calculate duration with error handling
                try:
                    entry_time = trade["entry_time"]
                    exit_time = trade["exit_time"]
                    duration = (exit_time - entry_time).total_seconds()
                except (TypeError, AttributeError) as e:
                    LOG.warning("Trade %d invalid timestamp format: %s", trade.get("trade_num", -1), e)
                    duration = 0.0

                # Defensive: Validate numeric fields
                entry_price = trade.get("entry_price", 0.0)
                exit_price = trade.get("exit_price", 0.0)
                pnl = trade.get("pnl", 0.0)
                mfe = trade.get("mfe", 0.0)
                mae = trade.get("mae", 0.0)

                # Prevent division by zero
                if SafeMath.is_zero(entry_price):
                    LOG.warning(
                        "Trade %d has zero entry_price. Using exit_price.",
                        trade.get("trade_num", -1),
                    )
                    entry_price = exit_price if SafeMath.is_not_zero(exit_price) else 1.0  # Fallback

                # Calculate derived metrics
                price_change = exit_price - entry_price
                pnl_percent = (price_change / entry_price) * 100 if entry_price != 0 else 0.0
                mfe_percent = (mfe / entry_price) * 100 if entry_price != 0 else 0.0
                mae_percent = (mae / entry_price) * 100 if entry_price != 0 else 0.0
                capture_efficiency = (pnl / mfe) if mfe > 0 else 0.0

                # Defensive: Clamp extreme values for display
                pnl_percent = max(-1000, min(1000, pnl_percent))  # ±1000%
                capture_efficiency = max(-2.0, min(2.0, capture_efficiency))  # ±200%

                result = "WIN" if pnl > 0 else "LOSS"

                # Format row with safe conversions
                try:
                    row = {
                        "trade_num": trade.get("trade_num", -1),
                        "entry_time": (entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time)),
                        "exit_time": (exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time)),
                        "duration_seconds": f"{duration:.1f}",
                        "direction": trade.get("direction", "UNKNOWN"),
                        "entry_price": f"{entry_price:.2f}",
                        "exit_price": f"{exit_price:.2f}",
                        "price_change": f"{price_change:.2f}",
                        "pnl": f"{pnl:.2f}",
                        "pnl_percent": f"{pnl_percent:.4f}",
                        "mfe": f"{mfe:.2f}",
                        "mae": f"{mae:.2f}",
                        "mfe_percent": f"{mfe_percent:.4f}",
                        "mae_percent": f"{mae_percent:.4f}",
                        "capture_efficiency": f"{capture_efficiency:.4f}",
                        "winner_to_loser": ("TRUE" if trade.get("winner_to_loser", False) else "FALSE"),
                        "equity_after": f"{trade.get('equity_after', 0.0):.2f}",
                        "result": result,
                        # Phase 3.3: Dual-agent attribution (optional fields with safe defaults)
                        "predicted_runway": f"{trade.get('predicted_runway', 0.0):.6f}",
                        "runway_utilization": f"{trade.get('runway_utilization', 0.0):.4f}",
                        "runway_error_pct": f"{trade.get('runway_error_pct', 0.0):.2f}",
                        "trigger_quality": trade.get("trigger_quality", "N/A"),
                        "harvester_quality": trade.get("harvester_quality", "N/A"),
                        "mfe_bar_offset": trade.get("mfe_bar_offset", -1),
                        "mae_bar_offset": trade.get("mae_bar_offset", -1),
                        "bars_from_mfe_to_exit": trade.get("bars_from_mfe_to_exit", -1),
                    }

                    writer.writerow(row)
                except (ValueError, TypeError) as e:
                    LOG.error("Trade %d formatting error: %s", trade.get("trade_num", -1), e)

        return str(filepath)

    def export_summary(self, metrics: dict, filename: str | None = None) -> str:
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
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", "Value"])

            # Format metrics
            summary_rows = [
                ["Total Trades", metrics["total_trades"]],
                ["Winning Trades", metrics["winning_trades"]],
                ["Losing Trades", metrics["losing_trades"]],
                ["Win Rate", f"{metrics['win_rate']*100:.2f}%"],
                ["Total PnL", f"${metrics['total_pnl']:.2f}"],
                ["Average Winner", f"${metrics['avg_winner']:.2f}"],
                ["Average Loser", f"${metrics['avg_loser']:.2f}"],
                ["Profit Factor", f"{metrics['profit_factor']:.2f}"],
                ["Expectancy", f"${metrics['expectancy']:.2f}"],
                ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}"],
                ["Initial Equity", f"${metrics['initial_equity']:.2f}"],
                ["Current Equity", f"${metrics['current_equity']:.2f}"],
                ["Total Return", f"{metrics['total_return']*100:.2f}%"],
                ["Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%"],
                ["Current Drawdown", f"{metrics['current_drawdown']*100:.2f}%"],
                ["Max Consecutive Wins", metrics["max_consecutive_wins"]],
                ["Max Consecutive Losses", metrics["max_consecutive_losses"]],
                ["Winner-to-Loser Count", metrics["winner_to_loser_count"]],
            ]

            writer.writerows(summary_rows)

        return str(filepath)

    def export_all(self, performance_tracker, prefix: str | None = None) -> dict[str, str]:
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
            results["trades"] = self.export_trades(trades, trades_file)

        # Export summary
        metrics = performance_tracker.get_metrics()
        results["summary"] = self.export_summary(metrics, summary_file)

        return results
