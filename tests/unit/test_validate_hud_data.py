import json

from scripts.validate_hud_data import HUDDataValidator


def test_multi_bot_check_validates_scope_not_root_equality(tmp_path):
    (tmp_path / "trade_log.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "training_stats.json").write_text(
        json.dumps(
            {
                "symbol": "XAUUSD",
                "timeframe_minutes": 1,
                "trigger_training_steps": 1,
                "harvester_training_steps": 1,
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "training_stats_XAUUSD_M5.json").write_text(
        json.dumps(
            {
                "symbol": "XAUUSD",
                "timeframe_minutes": 5,
                "trigger_training_steps": 99,
                "harvester_training_steps": 42,
            },
        ),
        encoding="utf-8",
    )
    paper_dir = tmp_path / "paper_XAUUSD_M5"
    paper_dir.mkdir()
    (paper_dir / "production_metrics.json").write_text(
        json.dumps({"metrics": {"symbol": "XAUUSD", "timeframe_minutes": 5}}),
        encoding="utf-8",
    )

    validator = HUDDataValidator(data_dir=tmp_path)

    result = validator.check_multi_bot_sync()

    assert result["training_stats_XAUUSD_M5.json"] is True
    assert result["paper_XAUUSD_M5/production_metrics.json"] is True
    assert not any("differs from training_stats.json" in warning for warning in validator.warnings)


def test_scoped_filename_parser_preserves_symbols_with_underscores(tmp_path):
    (tmp_path / "trade_log.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "risk_metrics_US_500_M15.json").write_text(
        json.dumps({"symbol": "US_500", "timeframe_minutes": 15}),
        encoding="utf-8",
    )

    validator = HUDDataValidator(data_dir=tmp_path)

    assert validator.check_multi_bot_sync()["risk_metrics_US_500_M15.json"] is True
    assert validator.issues == []
