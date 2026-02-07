"""Extended tests for src.monitoring.production_monitor.

Covers HTTP server, metrics JSON formatting, alert severity logging,
_save_metrics details, and edge cases not in the original test file.
"""

import json
import time
import pytest

from src.monitoring.production_monitor import (
    ProductionMonitor,
    TradingMetrics,
)


# ---------------------------------------------------------------------------
# HTTP server endpoint tests
# ---------------------------------------------------------------------------
class TestHTTPServer:
    def test_start_and_stop(self):
        mon = ProductionMonitor(http_enabled=True, http_port=0)
        mon.update_metrics(trades_today=3)
        # Port 0 will fail in HTTPServer if binding is used;
        # We test that start_http_server handles the error gracefully
        # (logs error, doesn't crash)
        mon.start_http_server()
        # If the server started, stop it
        if mon.http_server is not None:
            mon.stop_http_server()

    def test_start_http_server_disabled(self):
        mon = ProductionMonitor(http_enabled=False)
        mon.start_http_server()
        assert mon.http_server is None

    def test_stop_http_server_without_start(self):
        mon = ProductionMonitor(http_enabled=False)
        mon.stop_http_server()  # Should not raise


# ---------------------------------------------------------------------------
# get_metrics_json edge cases
# ---------------------------------------------------------------------------
class TestGetMetricsJson:
    def test_no_metrics_returns_error(self):
        mon = ProductionMonitor(http_enabled=False)
        result = json.loads(mon.get_metrics_json())
        assert "error" in result

    def test_with_metrics_and_alerts(self):
        mon = ProductionMonitor(http_enabled=False, alert_drawdown_pct=0.05)
        mon.update_metrics(drawdown_current=0.15)  # triggers alert
        result = json.loads(mon.get_metrics_json())
        assert result["status"] == "alerts"
        assert len(result["alerts"]) > 0

    def test_with_metrics_no_alerts(self):
        mon = ProductionMonitor(http_enabled=False)
        mon.update_metrics(trades_today=5)
        result = json.loads(mon.get_metrics_json())
        assert result["status"] == "ok"
        assert result["metrics"]["trades_today"] == 5

    def test_timestamp_present(self):
        mon = ProductionMonitor(http_enabled=False)
        mon.update_metrics()
        result = json.loads(mon.get_metrics_json())
        assert "timestamp" in result


# ---------------------------------------------------------------------------
# _save_metrics edge cases
# ---------------------------------------------------------------------------
class TestSaveMetrics:
    def test_save_creates_file(self, tmp_path):
        mf = tmp_path / "metrics.json"
        mon = ProductionMonitor(metrics_file=mf, http_enabled=False)
        mon.update_metrics(trades_total=42)
        assert mf.exists()
        data = json.loads(mf.read_text())
        assert data["metrics"]["trades_total"] == 42

    def test_save_no_metrics_noop(self, tmp_path):
        mf = tmp_path / "metrics.json"
        mon = ProductionMonitor(metrics_file=mf, http_enabled=False)
        mon._save_metrics()  # no metrics set
        assert not mf.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        mf = tmp_path / "sub" / "dir" / "metrics.json"
        mon = ProductionMonitor(metrics_file=mf, http_enabled=False)
        mon.update_metrics(trades_today=1)
        assert mf.exists()

    def test_saved_alerts_included(self, tmp_path):
        mf = tmp_path / "metrics.json"
        mon = ProductionMonitor(metrics_file=mf, http_enabled=False, alert_drawdown_pct=0.05)
        mon.update_metrics(drawdown_current=0.15)
        data = json.loads(mf.read_text())
        assert len(data["alerts"]) >= 1

    def test_updated_at_in_saved_data(self, tmp_path):
        mf = tmp_path / "metrics.json"
        mon = ProductionMonitor(metrics_file=mf, http_enabled=False)
        mon.update_metrics()
        data = json.loads(mf.read_text())
        assert "updated_at" in data


# ---------------------------------------------------------------------------
# Alert detail checks
# ---------------------------------------------------------------------------
class TestAlertDetails:
    def test_no_trade_alert_severity(self):
        mon = ProductionMonitor(alert_no_trade_hours=1.0, http_enabled=False)
        mon.update_metrics(last_trade_mins_ago=90)
        assert any(a.severity == "warning" and a.category == "trade" for a in mon.active_alerts)

    def test_drawdown_alert_severity(self):
        mon = ProductionMonitor(alert_drawdown_pct=0.05, http_enabled=False)
        mon.update_metrics(drawdown_current=0.12)
        assert any(a.severity == "error" and a.category == "pnl" for a in mon.active_alerts)

    def test_circuit_breaker_alert_message_includes_names(self):
        mon = ProductionMonitor(http_enabled=False)
        mon.update_metrics(circuit_breakers_tripped=2, circuit_breaker_names=["max_loss", "vol"])
        cb_alert = next(a for a in mon.active_alerts if "circuit" in a.message.lower())
        assert "max_loss" in cb_alert.message
        assert "vol" in cb_alert.message

    def test_fix_disconnect_is_critical(self):
        mon = ProductionMonitor(http_enabled=False)
        mon.update_metrics(fix_connected=False)
        assert any(a.severity == "critical" and a.category == "connection" for a in mon.active_alerts)

    def test_memory_alert_threshold(self):
        mon = ProductionMonitor(alert_memory_pct=0.80, http_enabled=False)
        mon.update_metrics(memory_usage_pct=0.90)
        assert any(a.category == "system" and "memory" in a.message.lower() for a in mon.active_alerts)

    def test_error_rate_alert(self):
        mon = ProductionMonitor(alert_error_rate_1h=10, http_enabled=False)
        mon.update_metrics(error_count_1h=25)
        assert any(a.severity == "error" and "error" in a.message.lower() for a in mon.active_alerts)

    def test_all_alerts_at_once(self):
        mon = ProductionMonitor(
            alert_no_trade_hours=0.5,
            alert_drawdown_pct=0.05,
            alert_memory_pct=0.70,
            alert_error_rate_1h=5,
            http_enabled=False,
        )
        mon.update_metrics(
            last_trade_mins_ago=60,
            drawdown_current=0.20,
            circuit_breakers_tripped=1,
            circuit_breaker_names=["x"],
            fix_connected=False,
            memory_usage_pct=0.85,
            error_count_1h=10,
        )
        # Should have at least 6 alerts (trade, pnl, circuit, fix, memory, error)
        assert len(mon.active_alerts) >= 6

    def test_alerts_cleared_when_conditions_resolve(self):
        mon = ProductionMonitor(alert_drawdown_pct=0.05, http_enabled=False)
        mon.update_metrics(drawdown_current=0.15)
        assert len(mon.active_alerts) > 0
        mon.update_metrics(drawdown_current=0.01)
        assert len(mon.active_alerts) == 0


# ---------------------------------------------------------------------------
# TradingMetrics dataclass
# ---------------------------------------------------------------------------
class TestTradingMetricsDataclass:
    def test_all_fields_set(self):
        m = TradingMetrics(
            realized_pnl_day=100, realized_pnl_total=500, unrealized_pnl=10,
            drawdown_current=0.05, drawdown_max=0.10, trades_today=5,
            trades_total=50, win_rate=0.6, avg_profit=10.0, avg_loss=-5.0,
            avg_trade_duration_mins=30, trigger_confidence_avg=0.7,
            harvester_confidence_avg=0.8, last_trade_mins_ago=15,
            circuit_breakers_tripped=0, circuit_breaker_names=[],
            uptime_hours=5.0, memory_usage_pct=0.5, error_count_1h=0,
            fix_connected=True, timestamp=time.time(),
        )
        assert m.realized_pnl_day == 100
        assert m.win_rate == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Uptime calculation
# ---------------------------------------------------------------------------
class TestUptimeCalculation:
    def test_uptime_increases(self):
        mon = ProductionMonitor(http_enabled=False)
        mon.start_time = time.time() - 7200  # 2 hours ago
        mon.update_metrics()
        assert mon.metrics.uptime_hours >= 1.9
