"""Tests for uncovered production paths in ProductionMonitor.

Covers: HTTP server handler (do_GET for /metrics, /health, 404),
log_message suppression, _check_alerts with no metrics (line 190),
start_http_server with actual requests, stop_http_server.
"""

import json
import urllib.request
import pytest

from src.monitoring.production_monitor import ProductionMonitor


class TestCheckAlertsNoMetrics:
    """Line 190: _check_alerts returns early when self.metrics is None."""

    def test_check_alerts_without_metrics_noop(self):
        mon = ProductionMonitor(http_enabled=False)
        assert mon.metrics is None
        mon._check_alerts()  # Should not raise
        assert mon.active_alerts == []


class TestHTTPServerEndpoints:
    """Cover HTTP handler do_GET for /metrics, /health, and 404 paths."""

    @pytest.fixture()
    def monitor_with_server(self):
        """Start a monitor with HTTP server on a free port."""
        # Use port 0 to let OS assign a free port
        mon = ProductionMonitor(http_enabled=True, http_port=0)
        mon.update_metrics(trades_today=7, realized_pnl_day=42.5)
        mon.start_http_server()
        if mon.http_server is None:
            pytest.skip("HTTP server could not start")
        port = mon.http_server.server_address[1]
        yield mon, port
        mon.stop_http_server()

    def test_metrics_endpoint(self, monitor_with_server):
        """GET /metrics returns JSON with trading metrics."""
        mon, port = monitor_with_server
        url = f"http://localhost:{port}/metrics"
        response = urllib.request.urlopen(url, timeout=2)
        assert response.status == 200
        data = json.loads(response.read())
        assert "metrics" in data
        assert data["metrics"]["trades_today"] == 7

    def test_health_endpoint(self, monitor_with_server):
        """GET /health returns status ok with uptime."""
        mon, port = monitor_with_server
        url = f"http://localhost:{port}/health"
        response = urllib.request.urlopen(url, timeout=2)
        assert response.status == 200
        data = json.loads(response.read())
        assert data["status"] == "ok"
        assert "uptime_hours" in data

    def test_unknown_path_returns_404(self, monitor_with_server):
        """GET /unknown returns 404."""
        mon, port = monitor_with_server
        url = f"http://localhost:{port}/unknown"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(url, timeout=2)
        assert exc_info.value.code == 404

    def test_start_http_when_disabled(self):
        """start_http_server does nothing when http_enabled=False."""
        mon = ProductionMonitor(http_enabled=False)
        mon.start_http_server()
        assert mon.http_server is None

    def test_stop_without_server(self):
        """stop_http_server is safe when no server was started."""
        mon = ProductionMonitor(http_enabled=False)
        mon.stop_http_server()  # Should not raise

    def test_start_on_unavailable_port(self):
        """start_http_server handles bind error gracefully."""
        # Start a server on a port, then try to start another on the same port
        mon1 = ProductionMonitor(http_enabled=True, http_port=0)
        mon1.update_metrics()
        mon1.start_http_server()
        if mon1.http_server is None:
            pytest.skip("First server could not start")
        port = mon1.http_server.server_address[1]
        try:
            mon2 = ProductionMonitor(http_enabled=True, http_port=port)
            mon2.update_metrics()
            mon2.start_http_server()  # Should handle error gracefully
            # If it somehow succeeded, clean up
            if mon2.http_server is not None:
                mon2.stop_http_server()
        finally:
            mon1.stop_http_server()


class TestGetMetricsJsonPaths:
    """Cover get_metrics_json with and without alerts."""

    def test_no_metrics(self):
        mon = ProductionMonitor(http_enabled=False)
        result = json.loads(mon.get_metrics_json())
        assert result["error"] == "No metrics available"

    def test_with_metrics_no_alerts(self):
        mon = ProductionMonitor(http_enabled=False)
        mon.update_metrics(trades_today=5)
        result = json.loads(mon.get_metrics_json())
        assert result["status"] == "ok"
        assert "timestamp" in result

    def test_with_metrics_and_alerts(self):
        mon = ProductionMonitor(http_enabled=False, alert_drawdown_pct=0.05)
        mon.update_metrics(drawdown_current=0.10)
        result = json.loads(mon.get_metrics_json())
        assert result["status"] == "alerts"
        assert len(result["alerts"]) > 0


class TestSaveMetrics:
    """Cover _save_metrics method."""

    def test_save_no_metrics_noop(self, tmp_path):
        """_save_metrics does nothing when metrics is None."""
        metrics_file = tmp_path / "metrics.json"
        mon = ProductionMonitor(http_enabled=False, metrics_file=metrics_file)
        mon._save_metrics()
        assert not metrics_file.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        """_save_metrics creates parent directories."""
        metrics_file = tmp_path / "deep" / "nested" / "metrics.json"
        mon = ProductionMonitor(http_enabled=False, metrics_file=metrics_file)
        mon.update_metrics(trades_total=99)
        assert metrics_file.exists()
        data = json.loads(metrics_file.read_text())
        assert data["metrics"]["trades_total"] == 99
        assert "updated_at" in data

    def test_save_includes_alerts(self, tmp_path):
        """Saved JSON includes alert data."""
        metrics_file = tmp_path / "metrics.json"
        mon = ProductionMonitor(
            http_enabled=False,
            metrics_file=metrics_file,
            alert_drawdown_pct=0.01,
        )
        mon.update_metrics(drawdown_current=0.05)
        data = json.loads(metrics_file.read_text())
        assert len(data["alerts"]) > 0
        assert data["alerts"][0]["category"] == "pnl"
