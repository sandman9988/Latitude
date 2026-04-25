from datetime import UTC, datetime, timedelta

from scripts.rebuild_trade_metrics import build_epoch_metrics, build_performance_snapshot


def _trade(entry_time, pnl, mfe=10.0, exit_time=None):
    return {
        "entry_time": entry_time.isoformat(),
        "exit_time": (exit_time or entry_time).isoformat(),
        "pnl": pnl,
        "mfe": mfe,
    }


def test_build_performance_snapshot_rebuilds_history_windows():
    now = datetime(2026, 4, 23, 12, 0, tzinfo=UTC)
    trades = [
        _trade(now - timedelta(hours=1), 10.0, mfe=20.0),
        _trade(now - timedelta(days=3), -5.0, mfe=10.0),
        _trade(now - timedelta(days=40), 7.0, mfe=14.0),
    ]

    snapshot = build_performance_snapshot(
        trades,
        trading_mode="paper",
        starting_equity=10_000.0,
        now=now,
    )

    assert snapshot["trading_mode"] == "paper"
    assert snapshot["daily"]["total_trades"] == 1
    assert snapshot["weekly"]["total_trades"] == 2
    assert snapshot["monthly"]["total_trades"] == 2
    assert snapshot["lifetime"]["total_trades"] == 3
    assert snapshot["lifetime"]["total_pnl"] == 12.0


def test_build_performance_snapshot_can_rebuild_one_bot_scope():
    now = datetime(2026, 4, 23, 12, 0, tzinfo=UTC)
    trades = [
        {**_trade(now - timedelta(hours=1), 10.0), "symbol": "XAUUSD", "timeframe_minutes": 5, "trading_mode": "paper"},
        {**_trade(now - timedelta(hours=1), 99.0), "symbol": "XAUUSD", "timeframe_minutes": 15, "trading_mode": "paper"},
        {**_trade(now - timedelta(hours=1), 88.0), "symbol": "EURUSD", "timeframe_minutes": 5, "trading_mode": "paper"},
        {**_trade(now - timedelta(hours=1), 77.0), "symbol": "XAUUSD", "timeframe_minutes": 5, "trading_mode": "live"},
    ]

    snapshot = build_performance_snapshot(
        trades,
        trading_mode="paper",
        starting_equity=10_000.0,
        symbol="XAUUSD",
        timeframe_minutes=5,
        now=now,
    )

    assert snapshot["symbol"] == "XAUUSD"
    assert snapshot["timeframe"] == "M5"
    assert snapshot["timeframe_minutes"] == 5
    assert snapshot["lifetime"]["total_trades"] == 1
    assert snapshot["lifetime"]["total_pnl"] == 10.0


def test_build_epoch_metrics_filters_by_close_or_entry_time(tmp_path):
    now = datetime(2026, 4, 23, 12, 0, tzinfo=UTC)
    epoch = now - timedelta(days=2)
    epoch_path = tmp_path / "stats_epoch.json"
    epoch_path.write_text(
        f'{{"epoch": "{epoch.isoformat()}", "set_at": "{now.isoformat()}"}}',
        encoding="utf-8",
    )
    trades = [
        _trade(now - timedelta(days=3), -10.0),
        _trade(now - timedelta(days=1), 15.0),
    ]

    metrics = build_epoch_metrics(
        trades,
        epoch_path=epoch_path,
        starting_equity=10_000.0,
        now=now,
    )

    assert metrics is not None
    assert metrics["included_trades"] == 1
    assert metrics["excluded_trades"] == 1
    assert metrics["excluded_pnl"] == -10.0
    assert metrics["metrics"]["total_pnl"] == 15.0
