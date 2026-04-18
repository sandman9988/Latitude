import pytest

from src.monitoring.hud_tabbed import TabbedHUD


def test_normalize_timeframe_label_prefers_timeframe_minutes():
    trade = {"timeframe_minutes": 15, "timeframe": "H1"}
    assert TabbedHUD._normalize_timeframe_label(trade) == "M15"


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [
        ("M15", "M15"),
        ("15m", "M15"),
        ("H1", "M60"),
        ("H4", "M240"),
        ("H12", "M720"),
        ("D1", "M1440"),
        ("legacy", "LEGACY"),
    ],
)
def test_normalize_timeframe_label_legacy_formats(timeframe, expected):
    trade = {"timeframe": timeframe}
    assert TabbedHUD._normalize_timeframe_label(trade) == expected


def test_normalize_timeframe_label_unknown_returns_mq():
    assert TabbedHUD._normalize_timeframe_label({}) == "M?"


def test_build_metrics_cube_groups_by_symbol_tf_mode():
    hud = TabbedHUD()
    trades = [
        {"symbol": "xauusd", "timeframe_minutes": 15, "trading_mode": "paper", "pnl": 10.0},
        {"symbol": "XAUUSD", "timeframe": "M15", "trading_mode": "paper", "pnl": -5.0},
        {"symbol": "XAUUSD", "timeframe": "H1", "ticket": "12345", "pnl": 15.0},
        {"symbol": "eurusd", "timeframe": "", "position_id": "abc", "pnl": 7.0},
    ]

    hud._build_metrics_cube(trades)

    assert ("XAUUSD", "M15", "paper") in hud.metrics_cube
    assert ("XAUUSD", "M60", "live") in hud.metrics_cube
    assert ("EURUSD", "M?", "live") in hud.metrics_cube
    assert len(hud.metrics_cube[("XAUUSD", "M15", "paper")]) == 2


def test_compute_metrics_sets_unknown_timeframe_counter(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    trade_log = data_dir / "trade_log.jsonl"
    trade_log.write_text(
        "\n".join(
            [
                '{"symbol":"XAUUSD","entry_time":"2026-01-01T00:00:00+00:00","exit_time":"2026-01-01T01:00:00+00:00","pnl":10.0,"timeframe_minutes":15,"trading_mode":"paper"}',
                '{"symbol":"EURUSD","entry_time":"2026-01-02T00:00:00+00:00","exit_time":"2026-01-02T01:00:00+00:00","pnl":-5.0,"trading_mode":"live"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    hud = TabbedHUD()
    hud.data_dir = data_dir
    hud._trade_log_reader = hud._trade_log_reader.__class__(trade_log)
    hud._stats_epoch = None

    hud._compute_metrics_from_trade_log()

    assert hud._trade_log_unknown_timeframe_count == 1
    assert ("EURUSD", "M?", "live") in hud.metrics_cube
