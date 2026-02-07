"""Tests for src.monitoring.production_monitor – ProductionMonitor, TradingMetrics, Alert."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from src.monitoring.production_monitor import Alert, ProductionMonitor, TradingMetrics


# ---------------------------------------------------------------------------
# TradingMetrics / Alert dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_trading_metrics_fields(self):
        m = TradingMetrics(
            realized_pnl_day=100.0,
            realized_pnl_total=500.0,
            unrealized_pnl=20.0,
            drawdown_current=0.05,
            drawdown_max=0.08,
            trades_today=5,
            trades_total=50,
            win_rate=0.55,
            avg_profit=10.0,
            avg_loss=-5.0,
            avg_trade_duration_mins=30.0,
            trigger_confidence_avg=0.7,
            harvester_confidence_avg=0.6,
            last_trade_mins_ago=15.0,
            circuit_breakers_tripped=0,
            circuit_breaker_names=[],
            uptime_hours=2.0,
            memory_usage_pct=0.4,
            error_count_1h=1,
            fix_connected=True,
            timestamp=time.time(),
        )
        assert m.realized_pnl_day == pytest.approx(100.0)
        assert m.fix_connected is True

    def test_alert_fields(self):
        a = Alert(
            severity="warning",
            category="trade",
            message="No trades",
            metric_value=5.0,
            threshold=4.0,
            timestamp=time.time(),
        )
        assert a.severity == "warning"
        assert a.category == "trade"


# ---------------------------------------------------------------------------
# ProductionMonitor
# ---------------------------------------------------------------------------

class TestProductionMonitor:
    @pytest.fixture()
    def monitor(self, tmp_path):
        return ProductionMonitor(
            http_enabled=False,
            metrics_file=tmp_path / "metrics.json",
        )

    # -- init --
    def test_init_defaults(self, monitor):
        assert monitor.metrics is None
        assert monitor.active_alerts == []
        assert monitor.alert_drawdown_pct == pytest.approx(0.10)

    # -- update_metrics --
    def test_update_creates_metrics(self, monitor):
        monitor.update_metrics(realized_pnl_day=50.0, trades_today=3)
        assert monitor.metrics is not None
        assert monitor.metrics.realized_pnl_day == pytest.approx(50.0)
        assert monitor.metrics.trades_today == 3

    def test_uptime_calculated(self, monitor):
        monitor.update_metrics()
        assert monitor.metrics.uptime_hours >= 0

    # -- alerts --
    def test_no_alerts_normal(self, monitor):
        monitor.update_metrics(
            last_trade_mins_ago=10,
            drawdown_current=0.01,
            fix_connected=True,
            memory_usage_pct=0.3,
            error_count_1h=0,
        )
        assert len(monitor.active_alerts) == 0

    def test_no_trade_alert(self):
        m = ProductionMonitor(alert_no_trade_hours=1.0, http_enabled=False)
        m.update_metrics(last_trade_mins_ago=90)  # 1.5 h > 1 h threshold
        alerts = [a for a in m.active_alerts if a.category == "trade"]
        assert len(alerts) == 1
        assert "No trades" in alerts[0].message

    def test_drawdown_alert(self):
        m = ProductionMonitor(alert_drawdown_pct=0.05, http_enabled=False)
        m.update_metrics(drawdown_current=0.12)
        alerts = [a for a in m.active_alerts if a.category == "pnl"]
        assert len(alerts) == 1
        assert alerts[0].severity == "error"

    def test_circuit_breaker_alert(self):
        m = ProductionMonitor(http_enabled=False)
        m.update_metrics(circuit_breakers_tripped=2, circuit_breaker_names=["loss", "vol"])
        alerts = [a for a in m.active_alerts if "circuit" in a.message.lower()]
        assert len(alerts) == 1

    def test_fix_disconnect_alert(self):
        m = ProductionMonitor(http_enabled=False)
        m.update_metrics(fix_connected=False)
        alerts = [a for a in m.active_alerts if a.severity == "critical"]
        assert len(alerts) == 1
        assert alerts[0].category == "connection"

    def test_memory_alert(self):
        m = ProductionMonitor(alert_memory_pct=0.5, http_enabled=False)
        m.update_metrics(memory_usage_pct=0.85)
        alerts = [a for a in m.active_alerts if "Memory" in a.message]
        assert len(alerts) == 1

    def test_error_rate_alert(self):
        m = ProductionMonitor(alert_error_rate_1h=5, http_enabled=False)
        m.update_metrics(error_count_1h=10)
        alerts = [a for a in m.active_alerts if "errors" in a.message]
        assert len(alerts) == 1

    def test_multiple_alerts_at_once(self):
        m = ProductionMonitor(
            alert_drawdown_pct=0.05,
            alert_memory_pct=0.5,
            http_enabled=False,
        )
        m.update_metrics(drawdown_current=0.10, memory_usage_pct=0.9, fix_connected=False)
        assert len(m.active_alerts) >= 3

    # -- metrics persistence --
    def test_save_metrics_creates_file(self, monitor, tmp_path):
        monitor.update_metrics(trades_total=42)
        path = tmp_path / "metrics.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["metrics"]["trades_total"] == 42

    # -- get_metrics_json --
    def test_get_metrics_json_no_metrics(self, monitor):
        result = json.loads(monitor.get_metrics_json())
        assert "error" in result

    def test_get_metrics_json_with_metrics(self, monitor):
        monitor.update_metrics(trades_today=5)
        result = json.loads(monitor.get_metrics_json())
        assert result["metrics"]["trades_today"] == 5
        assert "status" in result

    def test_status_ok_no_alerts(self, monitor):
        monitor.update_metrics()
        result = json.loads(monitor.get_metrics_json())
        assert result["status"] == "ok"

    def test_status_alerts_when_present(self):
        m = ProductionMonitor(http_enabled=False)
        m.update_metrics(fix_connected=False)
        result = json.loads(m.get_metrics_json())
        assert result["status"] == "alerts"

    # -- alerts cleared on next update --
    def test_alerts_cleared_when_resolved(self):
        m = ProductionMonitor(http_enabled=False)
        m.update_metrics(fix_connected=False)
        assert len(m.active_alerts) >= 1
        m.update_metrics(fix_connected=True)
        fix_alerts = [a for a in m.active_alerts if a.category == "connection"]
        assert len(fix_alerts) == 0
