"""Tests for src.monitoring.trade_analyzer – TradeAnalyzer (CSV trade analysis)."""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.monitoring.trade_analyzer import TradeAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(rows: list[dict], path: str) -> str:
    """Write a list of row-dicts to *path* as CSV and return *path*."""
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _sample_rows(n: int = 20, *, seed: int = 42) -> list[dict]:
    """Return *n* plausible trade rows with all standard columns."""
    rng = np.random.default_rng(seed)
    base_time = datetime(2026, 1, 10, 8, 0, 0)
    rows = []
    for i in range(n):
        pnl = float(rng.normal(10, 50))
        mfe = abs(pnl) + float(rng.uniform(5, 30))
        mae = abs(pnl) * float(rng.uniform(0.1, 0.5)) if pnl < 0 else float(rng.uniform(1, 10))
        entry_price = 100_000 + float(rng.uniform(-500, 500))
        exit_price = entry_price + pnl
        duration = int(rng.integers(60, 7200))
        entry_time = base_time + timedelta(hours=i)
        exit_time = entry_time + timedelta(seconds=duration)
        rows.append(
            {
                "trade_num": i + 1,
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "pnl": round(pnl, 4),
                "result": "WIN" if pnl > 0 else "LOSS",
                "mfe": round(mfe, 4),
                "mae": round(mae, 4),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "capture_efficiency": round(pnl / mfe if mfe > 0 else 0, 4),
                "equity_after": round(10_000 + pnl * (i + 1), 2),
                "duration_seconds": duration,
                "direction": rng.choice(["LONG", "SHORT"]),
            }
        )
    return rows


@pytest.fixture()
def csv_path(tmp_path):
    """Create a temporary CSV with 20 sample trades."""
    path = str(tmp_path / "trades.csv")
    _make_csv(_sample_rows(20), path)
    return path


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestTradeAnalyzerInit:

    def test_load_valid_csv(self, csv_path):
        ta = TradeAnalyzer(csv_path)
        assert len(ta.df) == 20

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TradeAnalyzer(str(tmp_path / "nonexistent.csv"))

    def test_missing_columns_raises(self, tmp_path):
        bad = str(tmp_path / "bad.csv")
        pd.DataFrame({"foo": [1]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            TradeAnalyzer(bad)

    def test_derived_columns_created(self, csv_path):
        ta = TradeAnalyzer(csv_path)
        for col in ("is_win", "hour", "day_of_week", "cumulative_pnl", "running_max", "drawdown", "drawdown_pct"):
            assert col in ta.df.columns


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

class TestSummaryStats:

    def test_keys_present(self, csv_path):
        stats = TradeAnalyzer(csv_path).get_summary_stats()
        for key in ("total_trades", "win_rate", "profit_factor", "sharpe_ratio",
                     "sortino_ratio", "max_drawdown", "max_win_streak", "max_loss_streak",
                     "avg_mfe", "avg_mae", "avg_capture_efficiency",
                     "avg_duration_seconds", "median_duration_seconds"):
            assert key in stats

    def test_total_trades(self, csv_path):
        stats = TradeAnalyzer(csv_path).get_summary_stats()
        assert stats["total_trades"] == 20

    def test_win_rate_range(self, csv_path):
        stats = TradeAnalyzer(csv_path).get_summary_stats()
        assert 0.0 <= stats["win_rate"] <= 1.0

    def test_all_wins_profit_factor_inf(self, tmp_path):
        rows = _sample_rows(5)
        for r in rows:
            r["pnl"] = abs(r["pnl"]) + 1  # ensure positive
            r["result"] = "WIN"
        path = str(tmp_path / "wins.csv")
        _make_csv(rows, path)
        stats = TradeAnalyzer(path).get_summary_stats()
        assert stats["profit_factor"] == np.inf


# ---------------------------------------------------------------------------
# Hourly / daily analysis
# ---------------------------------------------------------------------------

class TestGroupedAnalysis:

    def test_analyze_by_hour_returns_df(self, csv_path):
        result = TradeAnalyzer(csv_path).analyze_by_hour()
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"total_pnl", "avg_pnl", "num_trades", "win_rate"}

    def test_analyze_by_day_returns_df(self, csv_path):
        result = TradeAnalyzer(csv_path).analyze_by_day()
        assert isinstance(result, pd.DataFrame)
        # Index labels should be day names
        for day_name in result.index:
            assert day_name in ("Monday", "Tuesday", "Wednesday", "Thursday",
                                "Friday", "Saturday", "Sunday")


# ---------------------------------------------------------------------------
# Best / worst trades
# ---------------------------------------------------------------------------

class TestBestWorstTrades:

    def test_find_best_trades_count(self, csv_path):
        best = TradeAnalyzer(csv_path).find_best_trades(n=5)
        assert len(best) == 5

    def test_find_worst_trades_count(self, csv_path):
        worst = TradeAnalyzer(csv_path).find_worst_trades(n=3)
        assert len(worst) == 3

    def test_best_trades_sorted_desc(self, csv_path):
        best = TradeAnalyzer(csv_path).find_best_trades(n=5)
        pnls = best["pnl"].tolist()
        assert pnls == sorted(pnls, reverse=True)


# ---------------------------------------------------------------------------
# Capture efficiency analysis
# ---------------------------------------------------------------------------

class TestCaptureEfficiency:

    def test_returns_dict_with_expected_keys(self, csv_path):
        result = TradeAnalyzer(csv_path).analyze_capture_efficiency()
        for key in ("overall_avg", "overall_median", "wins_avg", "losses_avg", "pct_above_50"):
            assert key in result

    def test_missing_column_returns_error(self, tmp_path):
        rows = _sample_rows(5)
        for r in rows:
            del r["capture_efficiency"]
        path = str(tmp_path / "no_ce.csv")
        _make_csv(rows, path)
        result = TradeAnalyzer(path).analyze_capture_efficiency()
        assert "error" in result


# ---------------------------------------------------------------------------
# Dual-agent analysis (graceful degradation)
# ---------------------------------------------------------------------------

class TestDualAgentAnalysis:

    def test_returns_error_when_columns_missing(self, csv_path):
        result = TradeAnalyzer(csv_path).analyze_dual_agents()
        assert "error" in result


# ---------------------------------------------------------------------------
# Export / convert types
# ---------------------------------------------------------------------------

class TestExportAnalysis:

    def test_export_creates_json(self, csv_path, tmp_path):
        out = str(tmp_path / "report.json")
        ta = TradeAnalyzer(csv_path)
        result_path = ta.export_analysis(out)
        assert os.path.exists(result_path)

    def test_convert_types_handles_numpy(self, csv_path):
        ta = TradeAnalyzer(csv_path)
        converted = ta._convert_types({"a": np.int64(5), "b": np.float64(3.14), "c": np.array([1, 2])})
        assert converted == {"a": 5, "b": 3.14, "c": [1, 2]}
