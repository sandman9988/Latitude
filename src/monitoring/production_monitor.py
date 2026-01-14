"""
Production Monitoring - Metrics Collection & Alerting

Provides real-time metrics for production trading bot monitoring:
- P&L tracking (realized, unrealized, daily, cumulative)
- Trade statistics (frequency, win rate, average duration)
- Agent confidence and action distributions
- Circuit breaker status
- System health (uptime, memory, errors)

Exposes metrics via:
1. JSON endpoint (simple HTTP server)
2. JSON file (for external scrapers)
3. Prometheus format (optional)

Alerting for critical conditions:
- No trades for >4 hours despite opportunities
- Drawdown exceeds 10%
- Circuit breakers trip
- FIX connection lost
- Memory usage >80%
- Error rate spike
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradingMetrics:
    """Core trading metrics."""

    # P&L
    realized_pnl_day: float
    realized_pnl_total: float
    unrealized_pnl: float
    drawdown_current: float
    drawdown_max: float

    # Trade stats
    trades_today: int
    trades_total: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    avg_trade_duration_mins: float

    # Agent stats
    trigger_confidence_avg: float
    harvester_confidence_avg: float
    last_trade_mins_ago: float

    # Circuit breakers
    circuit_breakers_tripped: int
    circuit_breaker_names: List[str]

    # System health
    uptime_hours: float
    memory_usage_pct: float
    error_count_1h: int
    fix_connected: bool

    # Timestamp
    timestamp: float


@dataclass
class Alert:
    """Alert for critical condition."""

    severity: str  # "warning", "error", "critical"
    category: str  # "pnl", "trade", "system", "connection"
    message: str
    metric_value: float
    threshold: float
    timestamp: float


class ProductionMonitor:
    """
    Production monitoring for trading bot.

    Collects metrics, detects alert conditions, exposes data via HTTP/JSON.
    """

    def __init__(
        self,
        # Alert thresholds
        alert_no_trade_hours: float = 4.0,
        alert_drawdown_pct: float = 0.10,
        alert_memory_pct: float = 0.80,
        alert_error_rate_1h: int = 20,
        # Metrics file
        metrics_file: Optional[Path] = None,
        # HTTP server
        http_enabled: bool = True,
        http_port: int = 8765,
    ):
        self.alert_no_trade_hours = alert_no_trade_hours
        self.alert_drawdown_pct = alert_drawdown_pct
        self.alert_memory_pct = alert_memory_pct
        self.alert_error_rate_1h = alert_error_rate_1h

        self.metrics_file = metrics_file or Path("data/production_metrics.json")

        # Current metrics
        self.metrics: Optional[TradingMetrics] = None
        self.active_alerts: List[Alert] = []

        # HTTP server
        self.http_enabled = http_enabled
        self.http_port = http_port
        self.http_server: Optional[HTTPServer] = None
        self.http_thread: Optional[Thread] = None

        # Start time
        self.start_time = time.time()

    def update_metrics(
        self,
        # P&L
        realized_pnl_day: float = 0.0,
        realized_pnl_total: float = 0.0,
        unrealized_pnl: float = 0.0,
        drawdown_current: float = 0.0,
        drawdown_max: float = 0.0,
        # Trade stats
        trades_today: int = 0,
        trades_total: int = 0,
        win_rate: float = 0.0,
        avg_profit: float = 0.0,
        avg_loss: float = 0.0,
        avg_trade_duration_mins: float = 0.0,
        # Agent stats
        trigger_confidence_avg: float = 0.0,
        harvester_confidence_avg: float = 0.0,
        last_trade_mins_ago: float = 0.0,
        # Circuit breakers
        circuit_breakers_tripped: int = 0,
        circuit_breaker_names: Optional[List[str]] = None,
        # System health
        memory_usage_pct: float = 0.0,
        error_count_1h: int = 0,
        fix_connected: bool = True,
    ):
        """Update current metrics."""
        uptime_hours = (time.time() - self.start_time) / 3600

        self.metrics = TradingMetrics(
            realized_pnl_day=realized_pnl_day,
            realized_pnl_total=realized_pnl_total,
            unrealized_pnl=unrealized_pnl,
            drawdown_current=drawdown_current,
            drawdown_max=drawdown_max,
            trades_today=trades_today,
            trades_total=trades_total,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            avg_trade_duration_mins=avg_trade_duration_mins,
            trigger_confidence_avg=trigger_confidence_avg,
            harvester_confidence_avg=harvester_confidence_avg,
            last_trade_mins_ago=last_trade_mins_ago,
            circuit_breakers_tripped=circuit_breakers_tripped,
            circuit_breaker_names=circuit_breaker_names or [],
            uptime_hours=uptime_hours,
            memory_usage_pct=memory_usage_pct,
            error_count_1h=error_count_1h,
            fix_connected=fix_connected,
            timestamp=time.time(),
        )

        # Check for alerts
        self._check_alerts()

        # Save to file
        self._save_metrics()

    def _check_alerts(self):
        """Check for alert conditions."""
        if not self.metrics:
            return

        new_alerts = []

        # Alert: No trades for too long
        if self.metrics.last_trade_mins_ago > self.alert_no_trade_hours * 60:
            new_alerts.append(
                Alert(
                    severity="warning",
                    category="trade",
                    message=f"No trades for {self.metrics.last_trade_mins_ago/60:.1f} hours",
                    metric_value=self.metrics.last_trade_mins_ago / 60,
                    threshold=self.alert_no_trade_hours,
                    timestamp=time.time(),
                )
            )

        # Alert: Excessive drawdown
        if self.metrics.drawdown_current > self.alert_drawdown_pct:
            new_alerts.append(
                Alert(
                    severity="error",
                    category="pnl",
                    message=f"Drawdown {self.metrics.drawdown_current:.1%} exceeds threshold",
                    metric_value=self.metrics.drawdown_current,
                    threshold=self.alert_drawdown_pct,
                    timestamp=time.time(),
                )
            )

        # Alert: Circuit breakers tripped
        if self.metrics.circuit_breakers_tripped > 0:
            new_alerts.append(
                Alert(
                    severity="warning",
                    category="system",
                    message=f"{self.metrics.circuit_breakers_tripped} circuit breakers tripped: {', '.join(self.metrics.circuit_breaker_names)}",
                    metric_value=self.metrics.circuit_breakers_tripped,
                    threshold=0,
                    timestamp=time.time(),
                )
            )

        # Alert: FIX disconnected
        if not self.metrics.fix_connected:
            new_alerts.append(
                Alert(
                    severity="critical",
                    category="connection",
                    message="FIX connection lost",
                    metric_value=0.0,
                    threshold=1.0,
                    timestamp=time.time(),
                )
            )

        # Alert: High memory usage
        if self.metrics.memory_usage_pct > self.alert_memory_pct:
            new_alerts.append(
                Alert(
                    severity="warning",
                    category="system",
                    message=f"Memory usage {self.metrics.memory_usage_pct:.1%} exceeds threshold",
                    metric_value=self.metrics.memory_usage_pct,
                    threshold=self.alert_memory_pct,
                    timestamp=time.time(),
                )
            )

        # Alert: High error rate
        if self.metrics.error_count_1h > self.alert_error_rate_1h:
            new_alerts.append(
                Alert(
                    severity="error",
                    category="system",
                    message=f"{self.metrics.error_count_1h} errors in last hour",
                    metric_value=self.metrics.error_count_1h,
                    threshold=self.alert_error_rate_1h,
                    timestamp=time.time(),
                )
            )

        # Update active alerts
        self.active_alerts = new_alerts

        # Log critical/error alerts
        for alert in new_alerts:
            if alert.severity in ("critical", "error"):
                logger.error(f"🚨 ALERT [{alert.severity.upper()}] {alert.category}: {alert.message}")
            elif alert.severity == "warning":
                logger.warning(f"⚠️  ALERT [WARNING] {alert.category}: {alert.message}")

    def _save_metrics(self):
        """Save metrics to JSON file."""
        if not self.metrics:
            return

        data = {
            "metrics": asdict(self.metrics),
            "alerts": [asdict(a) for a in self.active_alerts],
            "updated_at": time.time(),
        }

        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_metrics_json(self) -> str:
        """Get metrics as JSON string."""
        if not self.metrics:
            return json.dumps({"error": "No metrics available"})

        data = {
            "metrics": asdict(self.metrics),
            "alerts": [asdict(a) for a in self.active_alerts],
            "status": "ok" if not self.active_alerts else "alerts",
            "timestamp": time.time(),
        }

        return json.dumps(data, indent=2)

    def start_http_server(self):
        """Start HTTP server for metrics endpoint."""
        if not self.http_enabled:
            return

        monitor = self

        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(monitor.get_metrics_json().encode())
                elif self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    health = {"status": "ok", "uptime_hours": (time.time() - monitor.start_time) / 3600}
                    self.wfile.write(json.dumps(health).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Suppress HTTP logs
                pass

        try:
            self.http_server = HTTPServer(("0.0.0.0", self.http_port), MetricsHandler)
            self.http_thread = Thread(target=self.http_server.serve_forever, daemon=True)
            self.http_thread.start()
            logger.info(f"📊 Metrics HTTP server started on port {self.http_port}")
            logger.info(f"   Endpoints: http://localhost:{self.http_port}/metrics")
            logger.info(f"              http://localhost:{self.http_port}/health")
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")

    def stop_http_server(self):
        """Stop HTTP server."""
        if self.http_server:
            self.http_server.shutdown()
            logger.info("HTTP server stopped")


# Self-test
if __name__ == "__main__":
    import psutil

    logging.basicConfig(level=logging.INFO)

    print("=== ProductionMonitor Self-Test ===\n")

    # Test 1: Basic metrics update
    print("Test 1: Metrics update and JSON export")
    monitor = ProductionMonitor(http_enabled=False)
    monitor.update_metrics(
        realized_pnl_day=150.50,
        realized_pnl_total=1500.00,
        unrealized_pnl=25.00,
        trades_today=10,
        trades_total=100,
        win_rate=0.55,
        last_trade_mins_ago=30,
    )

    metrics_json = json.loads(monitor.get_metrics_json())
    if metrics_json["metrics"]["realized_pnl_day"] == 150.50:
        print("  ✓ Metrics updated correctly")
    else:
        print("  ✗ Metrics update failed")

    # Test 2: No-trade alert
    print("\nTest 2: No-trade alert")
    monitor2 = ProductionMonitor(alert_no_trade_hours=1.0, http_enabled=False)
    monitor2.update_metrics(last_trade_mins_ago=90)  # 1.5 hours

    if len(monitor2.active_alerts) > 0 and monitor2.active_alerts[0].category == "trade":
        print(f"  ✓ Alert triggered: {monitor2.active_alerts[0].message}")
    else:
        print("  ✗ No alert triggered")

    # Test 3: Drawdown alert
    print("\nTest 3: Drawdown alert")
    monitor3 = ProductionMonitor(alert_drawdown_pct=0.05, http_enabled=False)
    monitor3.update_metrics(drawdown_current=0.12)  # 12% drawdown

    drawdown_alerts = [a for a in monitor3.active_alerts if a.category == "pnl"]
    if drawdown_alerts:
        print(f"  ✓ Drawdown alert: {drawdown_alerts[0].message}")
    else:
        print("  ✗ No drawdown alert")

    # Test 4: Circuit breaker alert
    print("\nTest 4: Circuit breaker alert")
    monitor4 = ProductionMonitor(http_enabled=False)
    monitor4.update_metrics(circuit_breakers_tripped=2, circuit_breaker_names=["max_loss", "volatility"])

    cb_alerts = [a for a in monitor4.active_alerts if "circuit" in a.message.lower()]
    if cb_alerts:
        print(f"  ✓ Circuit breaker alert: {cb_alerts[0].message}")
    else:
        print("  ✗ No circuit breaker alert")

    # Test 5: FIX connection alert
    print("\nTest 5: FIX connection alert")
    monitor5 = ProductionMonitor(http_enabled=False)
    monitor5.update_metrics(fix_connected=False)

    fix_alerts = [a for a in monitor5.active_alerts if a.severity == "critical"]
    if fix_alerts:
        print(f"  ✓ Critical FIX alert: {fix_alerts[0].message}")
    else:
        print("  ✗ No FIX alert")

    # Test 6: Metrics file persistence
    print("\nTest 6: Metrics file persistence")
    import tempfile

    temp_file = Path(tempfile.mktemp(suffix=".json"))
    monitor6 = ProductionMonitor(metrics_file=temp_file, http_enabled=False)
    monitor6.update_metrics(trades_total=42)

    if temp_file.exists():
        with open(temp_file) as f:
            saved = json.load(f)
        if saved["metrics"]["trades_total"] == 42:
            print("  ✓ Metrics persisted to file")
        else:
            print("  ✗ Metrics file mismatch")
    else:
        print("  ✗ Metrics file not created")

    temp_file.unlink(missing_ok=True)

    # Test 7: HTTP server (basic)
    print("\nTest 7: HTTP server startup")
    monitor7 = ProductionMonitor(http_enabled=True, http_port=8766)
    monitor7.update_metrics(trades_today=5)
    monitor7.start_http_server()

    time.sleep(0.5)  # Let server start

    # Try to fetch metrics
    try:
        import urllib.request

        response = urllib.request.urlopen("http://localhost:8766/metrics", timeout=2)
        data = json.loads(response.read())
        if data["metrics"]["trades_today"] == 5:
            print("  ✓ HTTP server working")
        else:
            print("  ✗ HTTP response mismatch")
    except Exception as e:
        print(f"  ✗ HTTP server failed: {e}")
    finally:
        monitor7.stop_http_server()

    print("\n=== Self-Test Complete ===")
