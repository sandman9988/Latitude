"""Regression tests for depth gating and HUD risk metrics export."""

import datetime as dt
import json
from unittest.mock import MagicMock, patch

import pytest

from src.core import ctrader_ddqn_paper


def _build_app():
    """Instantiate CTraderFixApp without launching background threads."""
    with patch("src.core.ctrader_ddqn_paper.threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        mock_thread.return_value.start.return_value = None
        app = ctrader_ddqn_paper.CTraderFixApp(symbol_id=10028, qty=0.1, timeframe_minutes=1, symbol="BTCUSD")
    app._health_monitor_running = False
    return app


def test_depth_gate_and_hud_export(tmp_path):
    app = _build_app()
    app.hud_data_dir = tmp_path
    app.start_time = dt.datetime.now(dt.UTC)
    app.bar_count = 1
    app.best_bid = 100.0
    app.best_ask = 100.5
    app.cur_pos = 0
    app.last_depth_floor = 0.5
    depth_bid = 0.25
    depth_ask = 0.40

    # Depth helper should flag thin books
    assert app._depth_is_too_thin(depth_bid, depth_ask, app.last_depth_floor) is True

    app.last_depth_metrics = {"bid": depth_bid, "ask": depth_ask, "ratio": 0.62, "levels": 3}
    app.last_depth_gate = True
    app.last_risk_cap_qty = 0.25
    app.last_base_qty = 0.50
    app.last_final_qty = 0.25
    app.last_vpin_stats = {"vpin": 0.6, "mean": 0.5, "std": 0.1, "zscore": 2.0}

    # Seed a few bars so RS volatility calculations succeed
    app.bars.clear()
    now = dt.datetime.now(dt.UTC)
    for i in range(5):
        app.bars.append((now, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i))

    app._export_hud_data()
    risk_metrics = json.loads((tmp_path / "risk_metrics.json").read_text())

    assert risk_metrics["depth_gate_active"] is True
    assert risk_metrics["depth_bid"] == pytest.approx(depth_bid)
    assert risk_metrics["vpin_zscore"] == pytest.approx(2.0)
    assert risk_metrics["vpin_threshold"] == pytest.approx(app.vpin_z_threshold)
