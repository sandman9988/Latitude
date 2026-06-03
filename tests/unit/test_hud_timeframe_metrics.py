import io
import json
import os
from contextlib import redirect_stdout

import pytest

import src.monitoring.hud_tabbed as hud_module
from src.monitoring.hud_tabbed import TabbedHUD, _strip_ansi


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
        '{"symbol":"XAUUSD","entry_time":"2026-01-01T00:00:00+00:00","exit_time":"2026-01-01T01:00:00+00:00","pnl":10.0,"timeframe_minutes":15,"trading_mode":"paper"}\n{"symbol":"EURUSD","entry_time":"2026-01-02T00:00:00+00:00","exit_time":"2026-01-02T01:00:00+00:00","pnl":-5.0,"trading_mode":"live"}'
         "\n",
        encoding="utf-8",
    )

    hud = TabbedHUD()
    hud.data_dir = data_dir
    hud._trade_log_reader = hud._trade_log_reader.__class__(trade_log)
    hud._stats_epoch = None

    hud._compute_metrics_from_trade_log()

    assert hud._trade_log_unknown_timeframe_count == 1
    assert ("EURUSD", "M?", "live") in hud.metrics_cube


def test_preferred_data_file_uses_active_scoped_file(tmp_path):
    hud = TabbedHUD()
    hud.data_dir = tmp_path
    hud.active_sym = "XAUUSD"
    hud.active_tf_min = 5
    generic = tmp_path / "risk_metrics.json"
    scoped = tmp_path / "risk_metrics_XAUUSD_M5.json"
    generic.write_text('{"symbol":"EURUSD"}', encoding="utf-8")
    scoped.write_text('{"symbol":"XAUUSD"}', encoding="utf-8")

    assert hud._preferred_data_file("risk_metrics.json") == scoped


def test_preferred_data_file_uses_active_paper_dir_before_freshest_other_bot(tmp_path):
    hud = TabbedHUD()
    hud.data_dir = tmp_path
    hud.active_sym = "XAUUSD"
    hud.active_tf_min = 30
    active_dir = tmp_path / "paper_XAUUSD_M30"
    other_dir = tmp_path / "paper_XAUUSD_M5"
    active_dir.mkdir()
    other_dir.mkdir()
    active = active_dir / "production_metrics.json"
    other = other_dir / "production_metrics.json"
    active.write_text('{"metrics":{"fix_connected":true}}', encoding="utf-8")
    other.write_text('{"metrics":{"fix_connected":false}}', encoding="utf-8")
    os.utime(active, (100, 100))
    os.utime(other, (200, 200))

    assert hud._preferred_data_file("production_metrics.json") == active


def test_health_connectivity_uses_active_paper_stats_not_freshest_other_tf(tmp_path, monkeypatch):
    hud = TabbedHUD()
    hud.data_dir = tmp_path
    hud.active_sym = "XAUUSD"
    hud.active_tf_min = 30
    hud.risk_stats = {
        "symbol": "XAUUSD",
        "timeframe_minutes": 30,
        "circuit_breaker": "INACTIVE",
        "feasibility": 0.8,
    }
    active = tmp_path / "paper_stats_XAUUSD_M30.json"
    other = tmp_path / "paper_stats_XAUUSD_M5.json"
    active.write_text('{"symbol":"XAUUSD","timeframe_minutes":30}', encoding="utf-8")
    other.write_text('{"symbol":"XAUUSD","timeframe_minutes":5}', encoding="utf-8")
    os.utime(active, (100, 100))
    os.utime(other, (119, 119))
    monkeypatch.setattr(hud_module.time, "time", lambda: 120)

    buf = io.StringIO()
    with redirect_stdout(buf):
        hud._render_health_connectivity()

    text = _strip_ansi(buf.getvalue())
    assert "Bot silent XAUUSD M30 20s" in text
    assert "XAUUSD M5" not in text
    assert "Data XAUUSD M30 1s" not in text


def test_refresh_uses_configured_timeframe_before_newest_training_fallback(tmp_path):
    hud = TabbedHUD()
    hud.data_dir = tmp_path
    (tmp_path / "bot_config.json").write_text(
        '{"symbol":"XAUUSD","timeframe_minutes":5,"starting_equity":10000}',
        encoding="utf-8",
    )
    (tmp_path / "bot_config_XAUUSD_M5.json").write_text(
        '{"symbol":"XAUUSD","timeframe_minutes":5,"starting_equity":10000,"qty":0.2}',
        encoding="utf-8",
    )
    (tmp_path / "training_stats_XAUUSD_M5.json").write_text(
        '{"symbol":"XAUUSD","timeframe_minutes":5,"trading_mode":"paper"}',
        encoding="utf-8",
    )
    newest = tmp_path / "training_stats_XAUUSD_M240.json"
    newest.write_text(
        '{"symbol":"XAUUSD","timeframe_minutes":240,"trading_mode":"paper"}',
        encoding="utf-8",
    )
    os.utime(newest, (200, 200))

    hud._refresh_data()

    assert hud.active_sym == "XAUUSD"
    assert hud.active_tf_min == 5
    assert hud.bot_config.get("qty") == 0.2


def test_performance_mode_breakdown_keeps_portfolio_view(tmp_path):
    hud = TabbedHUD()
    hud.data_dir = tmp_path
    hud.active_sym = "XAUUSD"
    hud.active_tf_min = 5
    trade_log = tmp_path / "trade_log.jsonl"
    trades = [
        {
            "symbol": "XAUUSD",
            "timeframe_minutes": 5,
            "trading_mode": "paper",
            "entry_time": "2026-04-24T08:00:00+00:00",
            "exit_time": "2026-04-24T08:05:00+00:00",
            "pnl": 100.0,
        },
        {
            "symbol": "XAUUSD",
            "timeframe_minutes": 30,
            "trading_mode": "paper",
            "entry_time": "2026-04-24T08:00:00+00:00",
            "exit_time": "2026-04-24T08:30:00+00:00",
            "pnl": -999.0,
        },
    ]
    trade_log.write_text("\n".join(json.dumps(t) for t in trades) + "\n", encoding="utf-8")
    hud._trade_log_reader = hud._trade_log_reader.__class__(trade_log)
    hud._compute_metrics_from_trade_log()

    buf = io.StringIO()
    with redirect_stdout(buf):
        hud._render_mode_breakdown()

    text = _strip_ansi(buf.getvalue())
    assert "MODE BREAKDOWN (PORTFOLIO)" in text
    assert "-899.00" in text


def test_decision_log_keeps_all_scoped_bot_entries(tmp_path):
    for tf, decision in [(5, "LONG"), (30, "SHORT")]:
        audit_dir = tmp_path / f"paper_XAUUSD_M{tf}" / "logs" / "audit"
        audit_dir.mkdir(parents=True)
        (audit_dir / "decisions.jsonl").write_text(
            json.dumps(
                {
                    "timestamp": f"2026-04-24T08:{tf:02d}:00+00:00",
                    "symbol": "XAUUSD",
                    "timeframe_minutes": tf,
                    "trading_mode": "paper",
                    "agent": "trigger",
                    "decision": decision,
                    "confidence": 0.8,
                    "context": {},
                    "reasoning": {},
                },
            )
            + "\n",
            encoding="utf-8",
        )

    hud = TabbedHUD()
    hud.data_dir = tmp_path
    hud.active_sym = "XAUUSD"
    hud.active_tf_min = 5

    buf = io.StringIO()
    with redirect_stdout(buf):
        hud._render_decision_log()

    text = _strip_ansi(buf.getvalue())
    assert "XAUUSD/M5" in text
    assert "XAUUSD/M30" in text
    assert "LONG" in text
    assert "SHORT" in text


def test_trades_tab_keeps_portfolio_trade_history(tmp_path):
    hud = TabbedHUD()
    hud.data_dir = tmp_path
    hud.active_sym = "XAUUSD"
    hud.active_tf_min = 5
    trade_log = tmp_path / "trade_log.jsonl"
    trades = [
        {
            "trade_id": "m5",
            "symbol": "XAUUSD",
            "timeframe_minutes": 5,
            "trading_mode": "paper",
            "direction": "LONG",
            "entry_time": "2026-04-24T08:00:00+00:00",
            "exit_time": "2026-04-24T08:05:00+00:00",
            "entry_price": 2400.0,
            "exit_price": 2401.0,
            "pnl": 10.0,
        },
        {
            "trade_id": "m30",
            "symbol": "XAUUSD",
            "timeframe_minutes": 30,
            "trading_mode": "paper",
            "direction": "SHORT",
            "entry_time": "2026-04-24T08:00:00+00:00",
            "exit_time": "2026-04-24T08:30:00+00:00",
            "entry_price": 2400.0,
            "exit_price": 2390.0,
            "pnl": 30.0,
        },
    ]
    trade_log.write_text("\n".join(json.dumps(t) for t in trades) + "\n", encoding="utf-8")
    hud._trade_log_reader = hud._trade_log_reader.__class__(trade_log)
    hud._compute_metrics_from_trade_log()
    hud._load_all_trades_cached()

    buf = io.StringIO()
    with redirect_stdout(buf):
        hud._render_trades()

    text = _strip_ansi(buf.getvalue())
    assert "TRADES" in text
    # Trade IDs only visible at Level 3 (instrument/TF trade list),
    # not at Level 1 (portfolio with period columns).
    # Verify the trades are present by drilling down.
    hud._ctx_level = 3
    hud._ctx_symbol = "XAUUSD"
    hud._ctx_tf = 5
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        hud._render_trades()
    text2 = _strip_ansi(buf2.getvalue())
    assert "m5" in text2
    assert hud._available_timeframes()  # M5 and M30 both exist


def test_position_block_lists_multiple_open_bot_positions():
    hud = TabbedHUD()
    hud.bot_config = {"trading_mode": "paper"}
    hud.position = {"direction": "LONG", "symbol": "XAUUSD", "timeframe_minutes": 60}
    hud.all_bots_stats = [
        {
            "symbol": "XAUUSD",
            "timeframe_minutes": 60,
            "_position": {
                "direction": "LONG",
                "entry_price": 4700.0,
                "current_price": 4701.0,
                "unrealized_pnl": 10.0,
                "ticks_held": 3,
            },
        },
        {
            "symbol": "XAUUSD",
            "timeframe_minutes": 240,
            "_position": {
                "direction": "SHORT",
                "entry_price": 4710.0,
                "current_price": 4708.0,
                "unrealized_pnl": 20.0,
                "ticks_held": 5,
            },
        },
    ]

    buf = io.StringIO()
    with redirect_stdout(buf):
        hud._render_position_block()

    text = _strip_ansi(buf.getvalue())
    assert "2 open positions" in text
    assert "Unrealized: +30.00" in text
    assert "XAUUSD/M60" in text
    assert "XAUUSD/M240" in text


def test_training_stats_filtered_to_universe_and_sorted_by_timeframe(tmp_path):
    hud = TabbedHUD()
    hud.data_dir = tmp_path
    hud.universe_stats = {
        "XAUUSD::M1": {"symbol": "XAUUSD", "timeframe_minutes": 1},
        "XAUUSD::M5": {"symbol": "XAUUSD", "timeframe_minutes": 5},
        "XAUUSD::M15": {"symbol": "XAUUSD", "timeframe_minutes": 15},
    }
    for sym, tf in [("XAUUSD", 15), ("XAUUSD", 1), ("BTCUSD", 1), ("XAUUSD", 5)]:
        (tmp_path / f"training_stats_{sym}_M{tf}.json").write_text(
            f'{{"trading_mode":"paper","trigger_training_steps":{tf}}}',
            encoding="utf-8",
        )

    hud._load_all_training_stats()

    assert [(i["symbol"], i["timeframe_minutes"]) for i in hud.training_stats_all] == [
        ("XAUUSD", 1),
        ("XAUUSD", 5),
        ("XAUUSD", 15),
    ]


def test_offline_status_normalized_accepts_completed_aliases():
    hud = TabbedHUD()

    assert hud._offline_status_normalized({"status": "completed"}) == "complete"
    assert hud._offline_status_normalized({"status": "done"}) == "complete"
    assert hud._offline_status_normalized({"status": "finished"}) == "complete"


def test_kurtosis_gate_display_uses_payload_threshold(tmp_path):
    hud = TabbedHUD()
    hud.data_dir = tmp_path
    (tmp_path / "circuit_breakers.json").write_text(
        '{"kurtosis":{"is_tripped":false,"threshold":0.0}}',
        encoding="utf-8",
    )
    risk_stats = {
        "symbol": "XAUUSD",
        "timeframe_minutes": 30,
        "kurtosis": 19.5,
        "kurtosis_threshold": 5.0,
        "kurtosis_gate_active": True,
        "circuit_breaker": "INACTIVE",
    }

    buf = io.StringIO()
    with redirect_stdout(buf):
        hud._render_risk_circuit_breaker(risk_stats)
        hud._render_risk_tail(risk_stats)

    text = _strip_ansi(buf.getvalue())
    assert "Kurtosis gate: ACTIVE [XAUUSD M30]" in text
    assert "Kurtosis gate active [XAUUSD M30]" in text
    assert "excess > 5.0" in text
    assert "thr=5.00" in text
    assert "threshold >5.0" in text
    assert "thr=3.00" not in text


def test_offline_total_jobs_falls_back_to_results_length():
    hud = TabbedHUD()
    results = [{"symbol": "XAUUSD"}, {"symbol": "EURUSD"}, {"symbol": "BTCUSD"}]

    assert hud._offline_total_jobs({"total_jobs": 0}, results) == 3
    assert hud._offline_total_jobs({}, results) == 3
    assert hud._offline_total_jobs({"total_jobs": 5}, results) == 5
