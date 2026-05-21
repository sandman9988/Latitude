import json
from datetime import UTC, datetime, timedelta

import scripts.performance_analyzer as pa
from src.persistence.learned_parameters import LearnedParametersManager


def test_capture_health_signal_is_bounded():
    trades = [
        {"symbol": "XAUUSD", "timeframe": "M5", "pnl": -10.0, "capture_ratio": -30.0},
        {"symbol": "XAUUSD", "timeframe": "M5", "pnl": 1.0, "capture_ratio": 0.5},
    ]

    bots = pa.compute_bot_metrics(trades)
    fleet = pa.compute_fleet_metrics(trades, bots, hours=24)

    assert bots["XAUUSD_M5"].mean_capture == -0.25
    assert fleet.mean_capture == -0.25


def test_apply_corrections_updates_learned_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pa, "HEALTH_FILE", tmp_path / "performance_health.json")
    mgr_path = tmp_path / "learned_parameters.json"
    scoped_path = tmp_path / "paper_XAUUSD_M5" / "learned_parameters.json"

    root_mgr = LearnedParametersManager(mgr_path)
    root_mgr.get("XAUUSD", "exit_confidence_threshold", timeframe="M5")
    root_mgr.save()

    scoped_path.parent.mkdir()
    scoped_mgr = LearnedParametersManager(scoped_path)
    scoped_mgr.set_value("XAUUSD", "exit_confidence_threshold", 0.60, timeframe="M5")
    scoped_mgr.save()

    anomaly = pa.Anomaly(
        code="CAPTURE_EFFICIENCY_LOW",
        severity="WARNING",
        symbol="XAUUSD",
        timeframe="M5",
        message="capture low",
        metric_value=0.0,
        threshold=0.25,
        correction="raise exit threshold",
        param_name="exit_confidence_threshold",
        delta=0.03,
    )

    applied = pa.apply_corrections([anomaly], auto_heal=True, verbose=False)

    assert len(applied) == 2
    assert any("learned_parameters.json: exit_confidence_threshold 0.5000" in msg for msg in applied)
    assert any("paper_XAUUSD_M5/learned_parameters.json: exit_confidence_threshold 0.6000" in msg for msg in applied)
    raw = json.loads(mgr_path.read_text())
    data = raw.get("data", raw)
    value = data["instruments"]["XAUUSD_M5_default"]["params"]["exit_confidence_threshold"]["value"]
    assert value == 0.53
    scoped_raw = json.loads(scoped_path.read_text())
    scoped_data = scoped_raw.get("data", scoped_raw)
    scoped_value = scoped_data["instruments"]["XAUUSD_M5_default"]["params"]["exit_confidence_threshold"]["value"]
    assert scoped_value == 0.63
    assert (tmp_path / pa.PARAM_RELOAD_FILE).exists()
    assert (tmp_path / "paper_XAUUSD_M5" / pa.PARAM_RELOAD_FILE).exists()


def test_cb_lockout_auto_heal_requests_targeted_reset_and_normalizes_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pa, "HEALTH_FILE", tmp_path / "performance_health.json")
    root_path = tmp_path / "learned_parameters.json"
    scoped_path = tmp_path / "paper_XAUUSD_M5" / "learned_parameters.json"
    scoped_path.parent.mkdir()

    for path in (root_path, scoped_path):
        mgr = LearnedParametersManager(path)
        mgr.set_value("XAUUSD", "entry_confidence_threshold", 0.90, timeframe="M5")
        mgr.set_value("XAUUSD", "feasibility_threshold", 0.80, timeframe="M5")
        mgr.save()

    anomaly = pa.Anomaly(
        code="CB_LOCKOUT",
        severity="CRITICAL",
        symbol="XAUUSD",
        timeframe="M5",
        message="locked out",
        metric_value=4.5,
        threshold=4.0,
        correction="reset",
        param_name="",
        delta=0.0,
    )

    applied = pa.apply_corrections([anomaly], auto_heal=True, verbose=False)

    reset_path = tmp_path / "paper_XAUUSD_M5" / pa.CB_RESET_FILE
    payload = json.loads(reset_path.read_text())
    assert payload["reason"] == "performance_analyzer_cb_lockout"
    assert payload["target_timeframes"] == [5]
    assert any("requested targeted circuit-breaker reset" in msg for msg in applied)

    for path in (root_path, scoped_path):
        raw = json.loads(path.read_text())
        data = raw.get("data", raw)
        params = data["instruments"]["XAUUSD_M5_default"]["params"]
        assert params["entry_confidence_threshold"]["value"] == 0.6
        assert params["feasibility_threshold"]["value"] == 0.5

    assert (tmp_path / pa.PARAM_RELOAD_FILE).exists()
    assert (tmp_path / "paper_XAUUSD_M5" / pa.PARAM_RELOAD_FILE).exists()


def test_run_analysis_detects_cb_lockout_even_with_no_recent_trades(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pa, "TRADE_LOG", tmp_path / "trade_log.jsonl")
    monkeypatch.setattr(pa, "HEALTH_FILE", tmp_path / "performance_health.json")
    bot_dir = tmp_path / "paper_BTCUSD_M1"
    bot_dir.mkdir(parents=True)
    trip_time = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
    (bot_dir / "circuit_breakers.json").write_text(json.dumps({
        "sortino": {
            "is_tripped": True,
            "trip_time": trip_time,
            "trip_reason": "Sortino ratio below threshold",
            "trip_value": -0.5,
            "threshold": 0.9,
            "returns": [-1.0],
        }
    }))
    pa.TRADE_LOG.write_text("")

    report = pa.run_analysis(hours=4, auto_heal=True, min_trades=3, quiet=True)

    assert report["overall_health"] == "CRITICAL"
    assert report["anomalies"][0]["code"] == "CB_LOCKOUT"
    assert (bot_dir / pa.CB_RESET_FILE).exists()
