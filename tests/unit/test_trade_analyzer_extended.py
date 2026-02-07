"""Extended tests for src.monitoring.trade_analyzer.

Covers print_report(), main() CLI, dual-agent analysis with proper columns,
_convert_types edge cases, export auto-path, and capture efficiency details.
"""

import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.monitoring.trade_analyzer import TradeAnalyzer, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _sample_rows(n=20, *, seed=42, include_dual_agent=False):
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
        row = {
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
        if include_dual_agent:
            row["trigger_quality"] = rng.choice(["excellent", "good", "poor"])
            row["harvester_quality"] = rng.choice(["excellent", "good", "poor"])
            row["predicted_runway"] = round(float(rng.uniform(5, 40)), 2)
        rows.append(row)
    return rows


@pytest.fixture()
def csv_path(tmp_path):
    path = str(tmp_path / "trades.csv")
    _make_csv(_sample_rows(20), path)
    return path


@pytest.fixture()
def dual_csv_path(tmp_path):
    path = str(tmp_path / "dual_trades.csv")
    _make_csv(_sample_rows(20, include_dual_agent=True), path)
    return path


# ---------------------------------------------------------------------------
# print_report()
# ---------------------------------------------------------------------------
class TestPrintReport:
    def test_print_report_runs(self, csv_path, capsys):
        ta = TradeAnalyzer(csv_path)
        ta.print_report()
        captured = capsys.readouterr()
        assert "TRADE ANALYSIS REPORT" in captured.out
        assert "OVERALL PERFORMANCE" in captured.out
        assert "RISK METRICS" in captured.out

    def test_print_report_includes_mfe_section(self, csv_path, capsys):
        ta = TradeAnalyzer(csv_path)
        ta.print_report()
        captured = capsys.readouterr()
        assert "MFE/MAE ANALYSIS" in captured.out

    def test_print_report_includes_duration_section(self, csv_path, capsys):
        ta = TradeAnalyzer(csv_path)
        ta.print_report()
        captured = capsys.readouterr()
        assert "DURATION ANALYSIS" in captured.out

    def test_print_report_with_dual_agents(self, dual_csv_path, capsys):
        ta = TradeAnalyzer(dual_csv_path)
        ta.print_report()
        captured = capsys.readouterr()
        assert "DUAL-AGENT ANALYSIS" in captured.out

    def test_print_report_best_worst(self, csv_path, capsys):
        ta = TradeAnalyzer(csv_path)
        ta.print_report()
        captured = capsys.readouterr()
        assert "BEST TRADES" in captured.out
        assert "WORST TRADES" in captured.out

    def test_print_report_shows_hours(self, csv_path, capsys):
        ta = TradeAnalyzer(csv_path)
        ta.print_report()
        captured = capsys.readouterr()
        assert "BEST HOURS" in captured.out

    def test_print_report_shows_days(self, csv_path, capsys):
        ta = TradeAnalyzer(csv_path)
        ta.print_report()
        captured = capsys.readouterr()
        assert "BEST DAYS" in captured.out


# ---------------------------------------------------------------------------
# Dual-agent analysis with actual columns
# ---------------------------------------------------------------------------
class TestDualAgentWithData:
    def test_quality_distributions(self, dual_csv_path):
        ta = TradeAnalyzer(dual_csv_path)
        result = ta.analyze_dual_agents()
        assert "error" not in result
        assert "trigger_quality_distribution" in result
        assert "harvester_quality_distribution" in result
        assert isinstance(result["trigger_quality_distribution"], dict)

    def test_runway_error_computed(self, dual_csv_path):
        ta = TradeAnalyzer(dual_csv_path)
        result = ta.analyze_dual_agents()
        # Predicted runway > 0 in our data, so should compute error
        assert result["avg_runway_error"] is not None
        assert result["avg_runway_error_pct"] is not None

    def test_capture_by_harvester_quality(self, dual_csv_path):
        ta = TradeAnalyzer(dual_csv_path)
        result = ta.analyze_dual_agents()
        assert result["capture_by_harvester_quality"] is not None


# ---------------------------------------------------------------------------
# _convert_types() edge cases
# ---------------------------------------------------------------------------
class TestConvertTypes:
    def test_nested_dict(self, csv_path):
        ta = TradeAnalyzer(csv_path)
        result = ta._convert_types({"outer": {"inner": np.int64(42)}})
        assert result == {"outer": {"inner": 42}}

    def test_list_of_numpy(self, csv_path):
        ta = TradeAnalyzer(csv_path)
        result = ta._convert_types([np.float64(1.5), np.int64(3)])
        assert result == [1.5, 3]

    def test_pandas_nan(self, csv_path):
        ta = TradeAnalyzer(csv_path)
        result = ta._convert_types(pd.NA)
        assert result is None

    def test_regular_int_passthrough(self, csv_path):
        ta = TradeAnalyzer(csv_path)
        assert ta._convert_types(42) == 42

    def test_regular_string_passthrough(self, csv_path):
        ta = TradeAnalyzer(csv_path)
        assert ta._convert_types("hello") == "hello"

    def test_ndarray_to_list(self, csv_path):
        ta = TradeAnalyzer(csv_path)
        result = ta._convert_types(np.array([1, 2, 3]))
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# export_analysis()
# ---------------------------------------------------------------------------
class TestExportAnalysis:
    def test_auto_generated_path(self, csv_path):
        ta = TradeAnalyzer(csv_path)
        result_path = ta.export_analysis()
        assert os.path.exists(result_path)
        # Clean up
        os.unlink(result_path)

    def test_exported_json_valid(self, csv_path, tmp_path):
        out = str(tmp_path / "analysis.json")
        ta = TradeAnalyzer(csv_path)
        ta.export_analysis(out)
        with open(out) as f:
            data = json.load(f)
        assert "metadata" in data
        assert "summary" in data
        assert "by_hour" in data
        assert "by_day" in data

    def test_exported_metadata_correct(self, csv_path, tmp_path):
        out = str(tmp_path / "analysis.json")
        ta = TradeAnalyzer(csv_path)
        ta.export_analysis(out)
        with open(out) as f:
            data = json.load(f)
        assert data["metadata"]["total_trades"] == 20


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------
class TestMainCli:
    def test_main_with_report(self, csv_path, capsys):
        with patch("sys.argv", ["trade_analyzer", csv_path, "--quiet"]):
            ret = main()
        assert ret == 0

    def test_main_with_export(self, csv_path, tmp_path):
        out = str(tmp_path / "cli_export.json")
        with patch("sys.argv", ["trade_analyzer", csv_path, "--export", out, "--quiet"]):
            ret = main()
        assert ret == 0
        assert os.path.exists(out)

    def test_main_missing_file(self, capsys):
        with patch("sys.argv", ["trade_analyzer", "/nonexistent/path.csv"]):
            ret = main()
        assert ret == 1

    def test_main_prints_report(self, csv_path, capsys):
        # Auto-export happens when not quiet and no --export
        with patch("sys.argv", ["trade_analyzer", csv_path]):
            ret = main()
        captured = capsys.readouterr()
        assert "TRADE ANALYSIS REPORT" in captured.out
        assert ret == 0
        # Clean up auto-exported file
        import glob
        for f in glob.glob("analysis_*.json"):
            os.unlink(f)


# ---------------------------------------------------------------------------
# Summary stats edge cases
# ---------------------------------------------------------------------------
class TestSummaryStatsEdge:
    def test_no_mfe_mae_columns(self, tmp_path):
        rows = _sample_rows(5)
        for r in rows:
            del r["mfe"]
            del r["mae"]
            del r["capture_efficiency"]
        path = str(tmp_path / "no_mfe.csv")
        _make_csv(rows, path)
        stats = TradeAnalyzer(path).get_summary_stats()
        assert stats["avg_mfe"] is None
        assert stats["avg_mae"] is None

    def test_no_duration_column(self, tmp_path):
        rows = _sample_rows(5)
        for r in rows:
            del r["duration_seconds"]
        path = str(tmp_path / "no_dur.csv")
        _make_csv(rows, path)
        stats = TradeAnalyzer(path).get_summary_stats()
        assert stats["avg_duration_seconds"] is None

    def test_all_losses(self, tmp_path):
        rows = _sample_rows(10)
        for r in rows:
            r["pnl"] = -abs(r["pnl"]) - 1
            r["result"] = "LOSS"
        path = str(tmp_path / "losses.csv")
        _make_csv(rows, path)
        stats = TradeAnalyzer(path).get_summary_stats()
        assert stats["win_rate"] == pytest.approx(0.0)
        assert stats["total_pnl"] < 0
        assert stats["profit_factor"] == pytest.approx(0.0)  # no gross_profit


# ---------------------------------------------------------------------------
# Capture efficiency details
# ---------------------------------------------------------------------------
class TestCaptureEfficiencyExtended:
    def test_all_wins(self, tmp_path):
        rows = _sample_rows(10)
        for r in rows:
            r["pnl"] = abs(r["pnl"]) + 1
            r["result"] = "WIN"
            r["capture_efficiency"] = 0.8
        path = str(tmp_path / "all_wins.csv")
        _make_csv(rows, path)
        result = TradeAnalyzer(path).analyze_capture_efficiency()
        assert result["wins_avg"] is not None
        assert result["losses_avg"] is None  # no losses

    def test_all_losses(self, tmp_path):
        rows = _sample_rows(10)
        for r in rows:
            r["pnl"] = -abs(r["pnl"]) - 1
            r["result"] = "LOSS"
            r["capture_efficiency"] = -0.5
        path = str(tmp_path / "all_losses.csv")
        _make_csv(rows, path)
        result = TradeAnalyzer(path).analyze_capture_efficiency()
        assert result["losses_avg"] is not None
        assert result["wins_avg"] is None  # no wins
