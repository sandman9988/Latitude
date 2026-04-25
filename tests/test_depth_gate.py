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
    app.shared_hud_dir = tmp_path
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
    scoped_risk_metrics = json.loads((tmp_path / "risk_metrics_BTCUSD_M1.json").read_text())
    training_stats = json.loads((tmp_path / "training_stats.json").read_text())

    assert risk_metrics["depth_gate_active"] is True
    assert scoped_risk_metrics["depth_gate_active"] is True
    assert training_stats["symbol"] == "BTCUSD"
    assert training_stats["timeframe"] == "M1"
    assert training_stats["timeframe_minutes"] == 1
    assert risk_metrics["depth_bid"] == pytest.approx(depth_bid)
    assert risk_metrics["vpin_zscore"] == pytest.approx(2.0)
    assert risk_metrics["vpin_threshold"] == pytest.approx(app.vpin_z_threshold)


def test_performance_snapshot_filters_shared_trade_log_by_bot_scope(tmp_path):
    app = ctrader_ddqn_paper.CTraderFixApp.__new__(ctrader_ddqn_paper.CTraderFixApp)
    app.shared_hud_dir = tmp_path
    app.symbol = "XAUUSD"
    app.timeframe_minutes = 5
    app.paper_mode = True
    app.starting_equity = 10_000.0

    now = dt.datetime.now(dt.UTC).isoformat()
    rows = [
        {"symbol": "XAUUSD", "timeframe_minutes": 5, "trading_mode": "paper", "entry_time": now, "pnl": 10.0},
        {"symbol": "XAUUSD", "timeframe_minutes": 15, "trading_mode": "paper", "entry_time": now, "pnl": 20.0},
        {"symbol": "EURUSD", "timeframe_minutes": 5, "trading_mode": "paper", "entry_time": now, "pnl": 30.0},
        {"symbol": "XAUUSD", "timeframe_minutes": 5, "trading_mode": "live", "entry_time": now, "pnl": 40.0},
    ]
    (tmp_path / "trade_log.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    snapshot = app._build_performance_snapshot({})

    assert snapshot["symbol"] == "XAUUSD"
    assert snapshot["timeframe"] == "M5"
    assert snapshot["timeframe_minutes"] == 5
    assert snapshot["lifetime"]["total_trades"] == 1
    assert snapshot["lifetime"]["total_pnl"] == pytest.approx(10.0)
