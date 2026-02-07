"""Tests for src.monitoring.trade_exporter – TradeExporter (CSV export)."""

import csv
import datetime as dt
from pathlib import Path

import pytest

from src.monitoring.trade_exporter import TradeExporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_trade(num: int = 1, pnl: float = 50.0) -> dict:
    """Create a minimal valid trade dict."""
    entry_time = dt.datetime(2026, 1, 10, 8, 0, tzinfo=dt.UTC)
    exit_time = entry_time + dt.timedelta(minutes=30)
    return {
        "trade_num": num,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "direction": "LONG",
        "entry_price": 100_000.0,
        "exit_price": 100_000.0 + pnl,
        "pnl": pnl,
        "mfe": abs(pnl) + 10,
        "mae": 5.0,
        "winner_to_loser": False,
        "equity_after": 10_050.0,
    }


@pytest.fixture()
def exporter(tmp_path):
    return TradeExporter(output_dir=str(tmp_path / "trades"))


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestTradeExporterInit:
    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "exports"
        TradeExporter(output_dir=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# export_trades
# ---------------------------------------------------------------------------

class TestExportTrades:
    def test_empty_list_raises(self, exporter):
        with pytest.raises(ValueError, match="No trades"):
            exporter.export_trades([])

    def test_creates_csv(self, exporter):
        path = exporter.export_trades([_sample_trade()])
        assert Path(path).exists()

    def test_csv_has_header(self, exporter):
        path = exporter.export_trades([_sample_trade()])
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "trade_num" in header
        assert "pnl" in header

    def test_csv_row_count(self, exporter):
        trades = [_sample_trade(i, pnl=10.0 * i) for i in range(1, 6)]
        path = exporter.export_trades(trades)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

    def test_custom_filename(self, exporter):
        path = exporter.export_trades([_sample_trade()], filename="custom.csv")
        assert Path(path).name == "custom.csv"

    def test_skip_trade_missing_timestamps(self, exporter):
        trade = _sample_trade()
        del trade["entry_time"]
        path = exporter.export_trades([trade])
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0  # Skipped

    def test_zero_entry_price_handled(self, exporter):
        trade = _sample_trade()
        trade["entry_price"] = 0.0
        path = exporter.export_trades([trade])
        assert Path(path).exists()  # Should not crash

    def test_win_result(self, exporter):
        path = exporter.export_trades([_sample_trade(pnl=50.0)])
        with open(path) as f:
            row = next(csv.DictReader(f))
        assert row["result"] == "WIN"

    def test_loss_result(self, exporter):
        path = exporter.export_trades([_sample_trade(pnl=-20.0)])
        with open(path) as f:
            row = next(csv.DictReader(f))
        assert row["result"] == "LOSS"


# ---------------------------------------------------------------------------
# export_summary
# ---------------------------------------------------------------------------

class TestExportSummary:
    def test_creates_csv(self, exporter):
        metrics = {
            "total_trades": 10, "winning_trades": 7, "losing_trades": 3,
            "win_rate": 0.7, "total_pnl": 500.0, "avg_winner": 100.0,
            "avg_loser": -50.0, "profit_factor": 4.67, "expectancy": 35.0,
            "sharpe_ratio": 1.5, "initial_equity": 10000.0,
            "current_equity": 10500.0, "total_return": 0.05,
            "max_drawdown": 0.03, "current_drawdown": 0.01,
            "max_consecutive_wins": 5, "max_consecutive_losses": 2,
            "winner_to_loser_count": 1,
        }
        path = exporter.export_summary(metrics)
        assert Path(path).exists()

    def test_custom_filename(self, exporter):
        metrics = {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0, "total_pnl": 0, "avg_winner": 0, "avg_loser": 0,
            "profit_factor": 0, "expectancy": 0, "sharpe_ratio": 0,
            "initial_equity": 10000, "current_equity": 10000, "total_return": 0,
            "max_drawdown": 0, "current_drawdown": 0,
            "max_consecutive_wins": 0, "max_consecutive_losses": 0,
            "winner_to_loser_count": 0,
        }
        path = exporter.export_summary(metrics, filename="summary.csv")
        assert Path(path).name == "summary.csv"
