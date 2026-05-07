import json

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
