import importlib.util
from pathlib import Path

import pytest


def _load_normalizer():
    script = Path(__file__).resolve().parents[2] / "scripts" / "normalize_trade_log_scale.py"
    spec = importlib.util.spec_from_file_location("normalize_trade_log_scale", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_backfills_point_scale_from_dollar_mfe_for_legacy_xau_record():
    normalizer = _load_normalizer()
    trade = {
        "symbol": "XAUUSD",
        "quantity": 0.1,
        "pnl": 3.1,
        "mfe": 3.1,
        "mae": 0.56,
        "winner_to_loser": True,
    }

    rec, changed = normalizer.normalize_trade(
        trade,
        {},
        fallback_contract_size=100.0,
        timestamp="2026-04-23T00:00:00+00:00",
    )

    assert changed
    assert rec["mfe"] == pytest.approx(3.1)
    assert rec["mfe_points"] == pytest.approx(0.31)
    assert rec["mae_points"] == pytest.approx(0.056)
    assert rec["capture_ratio"] == pytest.approx(1.0)
    assert rec["winner_to_loser"] is False
    assert rec["harvester_quality"] == "EXCELLENT"


def test_existing_point_scale_recomputes_dollar_mfe_consistently():
    normalizer = _load_normalizer()
    trade = {
        "symbol": "XAUUSD",
        "quantity": 0.1,
        "pnl": -18.9,
        "mfe": 0.0,
        "mae": 18.9,
        "mfe_points": 0.0,
        "mae_points": 1.89,
    }

    rec, _ = normalizer.normalize_trade(
        trade,
        {"XAUUSD": 100.0},
        fallback_contract_size=100.0,
        timestamp="2026-04-23T00:00:00+00:00",
    )

    assert rec["mae"] == pytest.approx(18.9)
    assert rec["pnl_points"] == pytest.approx(-1.89)
    assert rec["capture_ratio"] == pytest.approx(0.0)
    assert rec["diag_zero_mfe_loss"] is True
    assert rec["harvester_quality"] == "STOPPED_OUT"
