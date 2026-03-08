#!/usr/bin/env python3
"""
Trade Analyzer - Comprehensive analysis of trading performance
Analyzes CSV exports from TradeExporter with detailed metrics and visualizations.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CAPTURE_EFFICIENCY_THRESHOLD = 0.5


class TradeAnalyzer:
    """Analyze trade history from CSV exports."""

    def __init__(self, csv_path: str):
        """Load trades from CSV file."""
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self._validate_data()
        self._prepare_data()

    def _validate_data(self):
        """Validate required columns exist."""
        required = ["trade_num", "entry_time", "exit_time", "pnl", "result"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _prepare_data(self):
        """Prepare data for analysis."""
        # Convert timestamps
        self.df["entry_time"] = pd.to_datetime(self.df["entry_time"])
        self.df["exit_time"] = pd.to_datetime(self.df["exit_time"])

        # Convert numeric columns
        numeric_cols = [
            "pnl",
            "mfe",
            "mae",
            "entry_price",
            "exit_price",
            "capture_efficiency",
            "equity_after",
            "duration_seconds",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Add derived columns
        self.df["is_win"] = self.df["pnl"] > 0
        self.df["hour"] = self.df["entry_time"].dt.hour
        self.df["day_of_week"] = self.df["entry_time"].dt.dayofweek
        self.df["cumulative_pnl"] = self.df["pnl"].cumsum()

        # Calculate running max and drawdown
        self.df["running_max"] = self.df["cumulative_pnl"].cummax()
        self.df["drawdown"] = self.df["cumulative_pnl"] - self.df["running_max"]
        self.df["drawdown_pct"] = (self.df["drawdown"] / (self.df["running_max"] + 1e-8)) * 100

    def get_summary_stats(self) -> dict:
        """Get comprehensive summary statistics."""
        total_trades, wins, losses = self._split_trades()
        pnl_stats = self._calc_pnl_stats(total_trades, wins, losses)
        risk_stats = self._calc_risk_metrics()
        drawdown_stats = self._calc_drawdown_stats()
        streak_stats = self._calc_streaks()
        mfe_stats = self._calc_mfe_mae_stats()
        duration_stats = self._calc_duration_stats()

        return {
            **pnl_stats,
            **risk_stats,
            **drawdown_stats,
            **streak_stats,
            **mfe_stats,
            **duration_stats,
            "first_trade": self.df["entry_time"].min(),
            "last_trade": self.df["entry_time"].max(),
        }

    def _split_trades(self) -> tuple[int, pd.DataFrame, pd.DataFrame]:
        total_trades = len(self.df)
        wins = self.df[self.df["is_win"]]
        losses = self.df[~self.df["is_win"]]
        return total_trades, wins, losses

    def _calc_pnl_stats(self, total_trades: int, wins: pd.DataFrame, losses: pd.DataFrame) -> dict:
        total_pnl = self.df["pnl"].sum()
        win_rate = len(wins) / total_trades if total_trades > 0 else 0

        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        returns = self.df["pnl"].values
        expectancy = returns.mean() if len(returns) > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
        }

    def _calc_risk_metrics(self) -> dict:
        returns = self.df["pnl"].values
        if len(returns) > 1:
            sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
        else:
            sharpe = 0

        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            sortino = (returns.mean() / (downside_returns.std() + 1e-8)) * np.sqrt(252)
        else:
            sortino = 0

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
        }

    def _calc_drawdown_stats(self) -> dict:
        return {
            "max_drawdown": self.df["drawdown"].min(),
            "max_drawdown_pct": self.df["drawdown_pct"].min(),
        }

    def _calc_streaks(self) -> dict:
        self.df["win_streak"] = (self.df["is_win"] != self.df["is_win"].shift()).cumsum()
        win_streaks = self.df[self.df["is_win"]].groupby("win_streak").size()
        loss_streaks = self.df[~self.df["is_win"]].groupby("win_streak").size()

        max_win_streak = win_streaks.max() if len(win_streaks) > 0 else 0
        max_loss_streak = loss_streaks.max() if len(loss_streaks) > 0 else 0

        return {
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
        }

    def _calc_mfe_mae_stats(self) -> dict:
        if "mfe" in self.df.columns and "mae" in self.df.columns:
            avg_mfe = self.df["mfe"].mean()
            avg_mae = self.df["mae"].mean()
            avg_capture = self.df["capture_efficiency"].mean() if "capture_efficiency" in self.df.columns else 0
        else:
            avg_mfe = avg_mae = avg_capture = None

        return {
            "avg_mfe": avg_mfe,
            "avg_mae": avg_mae,
            "avg_capture_efficiency": avg_capture,
        }

    def _calc_duration_stats(self) -> dict:
        if "duration_seconds" in self.df.columns:
            avg_duration = self.df["duration_seconds"].mean()
            median_duration = self.df["duration_seconds"].median()
        else:
            avg_duration = median_duration = None

        return {
            "avg_duration_seconds": avg_duration,
            "median_duration_seconds": median_duration,
        }

    def analyze_by_hour(self) -> pd.DataFrame:
        """Analyze performance by hour of day."""
        hourly = self.df.groupby("hour").agg({"pnl": ["sum", "mean", "count"], "is_win": "mean"}).round(4)
        hourly.columns = ["total_pnl", "avg_pnl", "num_trades", "win_rate"]
        return hourly.sort_values("total_pnl", ascending=False)

    def analyze_by_day(self) -> pd.DataFrame:
        """Analyze performance by day of week."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily = self.df.groupby("day_of_week").agg({"pnl": ["sum", "mean", "count"], "is_win": "mean"}).round(4)
        daily.columns = ["total_pnl", "avg_pnl", "num_trades", "win_rate"]
        daily.index = [days[i] for i in daily.index]
        return daily.sort_values("total_pnl", ascending=False)

    def analyze_dual_agents(self) -> dict:
        """Analyze dual-agent performance (TriggerAgent + HarvesterAgent)."""
        if "trigger_quality" not in self.df.columns or "harvester_quality" not in self.df.columns:
            return {"error": "Dual-agent metrics not available in this dataset"}

        trigger_counts = self.df["trigger_quality"].value_counts()
        harvester_counts = self.df["harvester_quality"].value_counts()

        # Analyze runway predictions
        if "predicted_runway" in self.df.columns and "mfe" in self.df.columns:
            df_valid = self.df[self.df["predicted_runway"] > 0].copy()
            if len(df_valid) > 0:
                df_valid["runway_error"] = abs(df_valid["predicted_runway"] - df_valid["mfe"])
                avg_runway_error = df_valid["runway_error"].mean()
                avg_runway_error_pct = (df_valid["runway_error"] / df_valid["mfe"]).mean() * 100
            else:
                avg_runway_error = avg_runway_error_pct = None
        else:
            avg_runway_error = avg_runway_error_pct = None

        # Analyze capture efficiency by harvester quality
        if "capture_efficiency" in self.df.columns:
            capture_by_harvester = self.df.groupby("harvester_quality")["capture_efficiency"].mean()
        else:
            capture_by_harvester = None

        return {
            "trigger_quality_distribution": trigger_counts.to_dict(),
            "harvester_quality_distribution": harvester_counts.to_dict(),
            "avg_runway_error": avg_runway_error,
            "avg_runway_error_pct": avg_runway_error_pct,
            "capture_by_harvester_quality": (
                capture_by_harvester.to_dict() if capture_by_harvester is not None else None
            ),
        }

    def find_best_trades(self, n: int = 10) -> pd.DataFrame:
        """Find top N most profitable trades."""
        return self.df.nlargest(n, "pnl")[["trade_num", "entry_time", "direction", "pnl", "mfe", "capture_efficiency"]]

    def find_worst_trades(self, n: int = 10) -> pd.DataFrame:
        """Find top N worst trades."""
        return self.df.nsmallest(n, "pnl")[["trade_num", "entry_time", "direction", "pnl", "mae"]]

    def analyze_capture_efficiency(self) -> dict:
        """Analyze MFE capture efficiency."""
        if "capture_efficiency" not in self.df.columns:
            return {"error": "Capture efficiency metrics not available"}

        wins = self.df[self.df["is_win"]]["capture_efficiency"]
        losses = self.df[~self.df["is_win"]]["capture_efficiency"]

        return {
            "overall_avg": self.df["capture_efficiency"].mean(),
            "overall_median": self.df["capture_efficiency"].median(),
            "wins_avg": wins.mean() if len(wins) > 0 else None,
            "wins_median": wins.median() if len(wins) > 0 else None,
            "losses_avg": losses.mean() if len(losses) > 0 else None,
            "losses_median": losses.median() if len(losses) > 0 else None,
            "pct_above_50": (self.df["capture_efficiency"] > CAPTURE_EFFICIENCY_THRESHOLD).sum() / len(self.df) * 100,
        }

    def export_analysis(self, output_path: str = None) -> str:
        """Export comprehensive analysis to JSON."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"analysis_{timestamp}.json"

        analysis = {
            "metadata": {
                "source_file": str(self.csv_path),
                "analysis_date": datetime.now().isoformat(),
                "total_trades": len(self.df),
            },
            "summary": self.get_summary_stats(),
            "by_hour": self.analyze_by_hour().to_dict(),
            "by_day": self.analyze_by_day().to_dict(),
            "dual_agents": self.analyze_dual_agents(),
            "capture_efficiency": self.analyze_capture_efficiency(),
            "best_trades": self.find_best_trades(5).to_dict("records"),
            "worst_trades": self.find_worst_trades(5).to_dict("records"),
        }

        # Convert numpy/pandas types to Python native for JSON serialization
        analysis = self._convert_types(analysis)

        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        return output_path

    def _convert_types(self, obj):  # noqa: PLR0911
        """Convert numpy/pandas types to Python native types for JSON."""
        if isinstance(obj, dict):
            return {k: self._convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def _print_mfe_mae_section(self, stats: dict) -> None:
        """Print MFE/MAE and duration sub-sections (only when data is available)."""
        if stats["avg_mfe"] is not None:
            print(f"\n{'MFE/MAE ANALYSIS':-^80}")
            print(f"Average MFE:         ${stats['avg_mfe']:>10.2f}")
            print(f"Average MAE:         ${stats['avg_mae']:>10.2f}")
            print(f"Avg Capture Eff:     {stats['avg_capture_efficiency']*100:>9.2f}%")

        if stats["avg_duration_seconds"] is not None:
            print(f"\n{'DURATION ANALYSIS':-^80}")
            print(f"Avg Duration:        {stats['avg_duration_seconds']/60:>10.1f} minutes")
            print(f"Median Duration:     {stats['median_duration_seconds']/60:>10.1f} minutes")

    def _print_dual_agent_section(self, dual: dict) -> None:
        """Print dual-agent analysis sub-section."""
        if "error" in dual:
            return
        print(f"\n{'DUAL-AGENT ANALYSIS':-^80}")
        if dual.get("trigger_quality_distribution"):
            print("\nTrigger Quality Distribution:")
            for quality, count in sorted(dual["trigger_quality_distribution"].items()):
                print(f"  {quality:<20} {count:>5} trades")
        if dual.get("harvester_quality_distribution"):
            print("\nHarvester Quality Distribution:")
            for quality, count in sorted(dual["harvester_quality_distribution"].items()):
                print(f"  {quality:<20} {count:>5} trades")
        if dual.get("avg_runway_error_pct") is not None:
            print(f"\nRunway Prediction Error: {dual['avg_runway_error_pct']:.2f}%")

    def print_report(self):  # noqa: PLR0915
        """Print comprehensive analysis report to console."""
        stats = self.get_summary_stats()

        print("=" * 80)
        print("TRADE ANALYSIS REPORT")
        print("=" * 80)
        print(f"\nSource: {self.csv_path}")
        print(f"Period: {stats['first_trade']} to {stats['last_trade']}")
        print(f"\n{'OVERALL PERFORMANCE':-^80}")
        print(f"Total Trades:        {stats['total_trades']:>10}")
        print(f"Winning Trades:      {stats['winning_trades']:>10} ({stats['win_rate']*100:>6.2f}%)")
        print(f"Losing Trades:       {stats['losing_trades']:>10}")
        print(f"\nTotal PnL:           ${stats['total_pnl']:>10.2f}")
        print(f"Average Win:         ${stats['avg_win']:>10.2f}")
        print(f"Average Loss:        ${stats['avg_loss']:>10.2f}")
        print(f"Profit Factor:       {stats['profit_factor']:>10.2f}")
        print(f"Expectancy:          ${stats['expectancy']:>10.2f}")

        print(f"\n{'RISK METRICS':-^80}")
        print(f"Sharpe Ratio:        {stats['sharpe_ratio']:>10.3f}")
        print(f"Sortino Ratio:       {stats['sortino_ratio']:>10.3f}")
        print(f"Max Drawdown:        ${stats['max_drawdown']:>10.2f} ({stats['max_drawdown_pct']:.2f}%)")
        print(f"Max Win Streak:      {stats['max_win_streak']:>10}")
        print(f"Max Loss Streak:     {stats['max_loss_streak']:>10}")

        self._print_mfe_mae_section(stats)

        # Hourly analysis
        print(f"\n{'BEST HOURS (Top 5)':-^80}")
        hourly = self.analyze_by_hour().head(5)
        print(f"{'Hour':<10} {'Total PnL':<15} {'Avg PnL':<15} {'Trades':<10} {'Win Rate'}")
        for hour, row in hourly.iterrows():
            print(
                f"{hour:02d}:00      ${row['total_pnl']:<13.2f} ${row['avg_pnl']:<13.2f} {int(row['num_trades']):<10} {row['win_rate']*100:.1f}%"
            )

        # Daily analysis
        print(f"\n{'BEST DAYS':-^80}")
        daily = self.analyze_by_day()
        print(f"{'Day':<15} {'Total PnL':<15} {'Avg PnL':<15} {'Trades':<10} {'Win Rate'}")
        for day, row in daily.iterrows():
            print(
                f"{day:<15} ${row['total_pnl']:<13.2f} ${row['avg_pnl']:<13.2f} {int(row['num_trades']):<10} {row['win_rate']*100:.1f}%"
            )

        self._print_dual_agent_section(self.analyze_dual_agents())

        # Best and worst trades
        print(f"\n{'BEST TRADES (Top 5)':-^80}")
        best = self.find_best_trades(5)
        print(f"{'Trade#':<10} {'Time':<20} {'Dir':<8} {'PnL':<15} {'MFE':<15} {'Capture'}")
        for _, trade in best.iterrows():
            print(
                f"{int(trade['trade_num']):<10} {str(trade['entry_time']):<20} {trade['direction']:<8} "
                f"${trade['pnl']:<13.2f} ${trade['mfe']:<13.2f} {trade['capture_efficiency']*100:.1f}%"
            )

        print(f"\n{'WORST TRADES (Top 5)':-^80}")
        worst = self.find_worst_trades(5)
        print(f"{'Trade#':<10} {'Time':<20} {'Dir':<8} {'PnL':<15} {'MAE'}")
        for _, trade in worst.iterrows():
            print(
                f"{int(trade['trade_num']):<10} {str(trade['entry_time']):<20} {trade['direction']:<8} "
                f"${trade['pnl']:<13.2f} ${trade['mae']:<13.2f}"
            )

        print("=" * 80)


def main():
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Analyze trading performance from CSV exports")
    parser.add_argument("csv_file", help="Path to trades CSV file")
    parser.add_argument("--export", "-e", help="Export analysis to JSON file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress console output")

    args = parser.parse_args()

    try:
        analyzer = TradeAnalyzer(args.csv_file)

        if not args.quiet:
            analyzer.print_report()

        if args.export:
            output_path = analyzer.export_analysis(args.export)
            print(f"\n✓ Analysis exported to: {output_path}")
        elif not args.quiet:
            # Auto-export if not suppressed
            output_path = analyzer.export_analysis()
            print(f"\n✓ Analysis exported to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
