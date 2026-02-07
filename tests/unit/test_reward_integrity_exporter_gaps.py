"""Gap tests for reward_integrity_monitor lines 191-193 and trade_exporter lines 94-96, 159-160.

Targets:
- reward_integrity_monitor.py: Exception path in correlation calculation (line 191-193)
- trade_exporter.py: Missing timestamps skip (line 94-96), formatting error catch (line 159-160)
"""

import csv
import datetime as dt
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.core.reward_integrity_monitor import RewardIntegrityMonitor
from src.monitoring.trade_exporter import TradeExporter


# ---------------------------------------------------------------------------
# RewardIntegrityMonitor – correlation exception path (lines 191-193)
# ---------------------------------------------------------------------------
class TestCorrelationException:
    def test_exception_in_corrcoef_returns_zero(self):
        """When np.corrcoef raises, correlation should fallback to 0.0."""
        monitor = RewardIntegrityMonitor(min_samples=3)

        # Add enough samples to pass min_samples check
        for i in range(5):
            monitor.add_trade(
                reward=float(i),
                pnl=float(i * 2),
                trade_id=i,
                reward_components={"capture_efficiency": float(i)},
            )

        # Patch corrcoef to raise
        with patch("src.core.reward_integrity_monitor.np.corrcoef", side_effect=ValueError("test")):
            result = monitor.check_integrity()

        assert result["correlation"] == 0.0
        assert result["status"] != "insufficient_data"

    def test_nan_correlation_handled(self):
        """When all rewards/pnls are same, corrcoef returns NaN."""
        monitor = RewardIntegrityMonitor(min_samples=3)

        # All same values → NaN correlation
        for i in range(5):
            monitor.add_trade(
                reward=1.0,  # All same
                pnl=1.0,  # All same
                trade_id=i,
                reward_components={"capture_efficiency": 1.0},
            )

        result = monitor.check_integrity()
        # NaN should be caught and set to 0.0
        assert result["correlation"] == 0.0

    def test_insufficient_data_returns_early(self):
        """Below min_samples → insufficient_data status."""
        monitor = RewardIntegrityMonitor(min_samples=10)
        monitor.add_trade(reward=1.0, pnl=2.0, trade_id=0, reward_components={})

        result = monitor.check_integrity()
        assert result["status"] == "insufficient_data"
        assert result["correlation"] is None


# ---------------------------------------------------------------------------
# TradeExporter – defensive paths
# ---------------------------------------------------------------------------
class TestTradeExporterDefensivePaths:
    def test_missing_timestamps_skipped(self, tmp_path):
        """Trades missing entry_time or exit_time are skipped."""
        exporter = TradeExporter(output_dir=str(tmp_path))
        trades = [
            {
                "trade_num": 1,
                # Missing entry_time and exit_time
                "entry_price": 100.0,
                "exit_price": 105.0,
                "pnl": 5.0,
            },
            {
                "trade_num": 2,
                "entry_time": dt.datetime(2026, 1, 1, 10, 0),
                "exit_time": dt.datetime(2026, 1, 1, 11, 0),
                "entry_price": 100.0,
                "exit_price": 105.0,
                "pnl": 5.0,
                "mfe": 6.0,
                "mae": 1.0,
            },
        ]

        filepath = exporter.export_trades(trades, filename="test_skip.csv")

        # Read back CSV — only trade 2 should be written
        with open(filepath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["trade_num"] == "2"

    def test_invalid_timestamp_format_handled(self, tmp_path):
        """Trades with non-datetime timestamps get duration=0."""
        exporter = TradeExporter(output_dir=str(tmp_path))
        trades = [
            {
                "trade_num": 1,
                "entry_time": "not-a-datetime",  # String, not datetime
                "exit_time": "also-not-datetime",
                "entry_price": 100.0,
                "exit_price": 105.0,
                "pnl": 5.0,
                "mfe": 6.0,
                "mae": 1.0,
            },
        ]

        filepath = exporter.export_trades(trades, filename="test_bad_ts.csv")

        with open(filepath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["duration_seconds"] == "0.0"

    def test_zero_entry_price_fallback(self, tmp_path):
        """Trade with entry_price=0 uses exit_price as fallback."""
        exporter = TradeExporter(output_dir=str(tmp_path))
        trades = [
            {
                "trade_num": 1,
                "entry_time": dt.datetime(2026, 1, 1, 10, 0),
                "exit_time": dt.datetime(2026, 1, 1, 11, 0),
                "entry_price": 0.0,  # Zero!
                "exit_price": 100.0,
                "pnl": 5.0,
                "mfe": 6.0,
                "mae": 1.0,
            },
        ]

        filepath = exporter.export_trades(trades, filename="test_zero_price.csv")

        with open(filepath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        # entry_price should be exit_price (100.0) as fallback
        assert rows[0]["entry_price"] == "100.00"

    def test_empty_trades_raises(self, tmp_path):
        """Empty trades list raises ValueError."""
        exporter = TradeExporter(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="No trades"):
            exporter.export_trades([])

    def test_export_summary(self, tmp_path):
        """export_summary creates valid CSV with metrics."""
        exporter = TradeExporter(output_dir=str(tmp_path))
        metrics = {
            "total_trades": 10,
            "winning_trades": 6,
            "losing_trades": 4,
            "win_rate": 0.6,
            "total_pnl": 500.0,
            "avg_winner": 120.0,
            "avg_loser": -50.0,
            "profit_factor": 3.6,
            "expectancy": 50.0,
            "sharpe_ratio": 1.5,
            "initial_equity": 10000.0,
            "current_equity": 10500.0,
            "total_return": 0.05,
            "max_drawdown": 0.02,
            "current_drawdown": 0.01,
            "max_consecutive_wins": 4,
            "max_consecutive_losses": 2,
            "winner_to_loser_count": 1,
        }

        filepath = exporter.export_summary(metrics, filename="test_summary.csv")
        assert Path(filepath).exists()

        with open(filepath) as f:
            reader = csv.reader(f)
            rows = list(reader)

        # First row is header
        assert rows[0] == ["Metric", "Value"]
        assert len(rows) > 10

    def test_export_all(self, tmp_path):
        """export_all creates both trades and summary files."""

        class MockTracker:
            def get_trade_history(self):
                return [
                    {
                        "trade_num": 1,
                        "entry_time": dt.datetime(2026, 1, 1, 10, 0),
                        "exit_time": dt.datetime(2026, 1, 1, 11, 0),
                        "entry_price": 100.0,
                        "exit_price": 105.0,
                        "pnl": 5.0,
                        "mfe": 6.0,
                        "mae": 1.0,
                    }
                ]

            def get_metrics(self):
                return {
                    "total_trades": 1,
                    "winning_trades": 1,
                    "losing_trades": 0,
                    "win_rate": 1.0,
                    "total_pnl": 5.0,
                    "avg_winner": 5.0,
                    "avg_loser": 0.0,
                    "profit_factor": float("inf"),
                    "expectancy": 5.0,
                    "sharpe_ratio": 0.0,
                    "initial_equity": 10000.0,
                    "current_equity": 10005.0,
                    "total_return": 0.0005,
                    "max_drawdown": 0.0,
                    "current_drawdown": 0.0,
                    "max_consecutive_wins": 1,
                    "max_consecutive_losses": 0,
                    "winner_to_loser_count": 0,
                }

        exporter = TradeExporter(output_dir=str(tmp_path))
        results = exporter.export_all(MockTracker(), prefix="test")
        assert "trades" in results
        assert "summary" in results
        assert Path(results["trades"]).exists()
        assert Path(results["summary"]).exists()
