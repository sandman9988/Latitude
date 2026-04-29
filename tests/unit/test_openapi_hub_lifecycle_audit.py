from __future__ import annotations

import datetime as dt
import json
import threading
from types import SimpleNamespace

import pytest

from src.core.openapi_hub import TFAgent


def _agent_stub() -> TFAgent:
    agent = TFAgent.__new__(TFAgent)
    agent._trade_sequence_lock = threading.Lock()
    agent._trade_sequence = 0
    agent._epoch_ts = 1777000000
    agent.symbol = "XAUUSD"
    agent.symbol_id = 10028
    agent.tf_label = "M5"
    agent.timeframe_minutes = 5
    agent.qty = 0.01
    agent.contract_size = 100.0
    agent.last_half_spread = 0.05
    agent._entry_conf = 0.77
    agent._entry_imbalance = 0.12
    agent._last_depth_bid = 2.5
    agent._last_depth_ask = 1.25
    agent._has_real_sizes = True
    agent._last_l2_snapshot = {
        "bids": [[4611.9, 1.5]],
        "asks": [[4612.1, 1.0]],
        "depth_bid": 2.5,
        "depth_ask": 1.25,
    }
    agent._vpin_z = 1.7
    agent._last_var_95 = 0.002
    agent._last_kurtosis = 4.2
    agent.last_mid = 4612.1
    agent.last_half_spread = 0.05
    agent._last_harvester_conf = 0.88
    agent._exit_conf_dynamic_floor = 0.55
    agent._runway_delta_ema = 1.2
    agent._runway_accuracy_ema = 0.64
    agent.equity = 10001.23
    agent.policy = SimpleNamespace(harvester=SimpleNamespace(last_close_reason="capture_decay"))
    return agent


def test_trade_log_persists_exit_lifecycle_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _agent_stub()
    entry = dt.datetime(2026, 4, 28, 10, 0, tzinfo=dt.UTC)
    exit_ = dt.datetime(2026, 4, 28, 10, 5, tzinfo=dt.UTC)

    agent._write_trade_log(
        direction=1,
        entry_price=4610.0,
        exit_price=4612.0,
        entry_time=entry,
        exit_time=exit_,
        pnl_usd=2.0,
        pnl_pts=2.0,
        mfe=3.0,
        mae=0.5,
        quantity=0.01,
        trigger_data={"entry_kurtosis": 1.4},
        exit_data={"exit_confidence": 0.88, "exit_capture_decay_armed": True},
        reward_trigger_breakdown={"accuracy": 0.5},
        reward_harvester_breakdown={"harvester_reward": 0.4},
    )

    records = [json.loads(line) for line in (tmp_path / "data" / "trade_log.jsonl").read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["trigger_data"]["entry_kurtosis"] == 1.4
    assert record["quantity"] == 0.01
    assert record["contract_size"] == 100.0
    assert record["mfe"] == 3.0
    assert record["mae"] == 0.5
    assert record["mfe_points"] == 3.0
    assert record["mae_points"] == 0.5
    assert record["exit_data"]["exit_confidence"] == 0.88
    assert record["exit_data"]["exit_capture_decay_armed"] is True
    assert record["reward_trigger_breakdown"]["accuracy"] == 0.5
    assert record["reward_harvester_breakdown"]["harvester_reward"] == 0.4
    assert record["predicted_runway_net_points_raw"] >= 0.0
    assert "runway_bias_ema_points" in record
    assert "runway_adjustment_scale" in record
    assert "runway_delta_points" in record
    assert "runway_error_pct" in record
    assert record["trigger_quality"] in {"N/A", "EXCELLENT", "UNDERPREDICTED", "GOOD", "OVERPREDICTED"}
    assert record["harvester_quality"] in {"N/A", "EXCELLENT", "GOOD", "FAIR", "POOR", "POOR_WTL", "STOPPED_OUT"}
    assert "mfe_bar_offset" in record
    assert "mae_bar_offset" in record
    assert "bars_from_mfe_to_exit" in record


def test_transaction_event_is_scoped_for_trade_lifecycle():
    agent = _agent_stub()
    calls = []
    agent._current_trade_id = "XAUUSD_M5_abc123"
    agent.transaction_log = SimpleNamespace(log_event=lambda *args, **kwargs: calls.append((args, kwargs)))

    agent._log_transaction_event("POSITION_CLOSE", {"pnl": 1.25, "mfe": 2.0})

    assert calls
    args, kwargs = calls[0]
    assert args[0] == "POSITION_CLOSE"
    payload = args[1]
    assert payload["symbol"] == "XAUUSD"
    assert payload["timeframe"] == "M5"
    assert payload["timeframe_minutes"] == 5
    assert payload["trade_id"] == "XAUUSD_M5_abc123"
    assert payload["pnl"] == 1.25
    assert kwargs["severity"] == "INFO"


def test_trade_log_converts_btc_excursions_from_points_to_usd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _agent_stub()
    agent.symbol = "BTCUSD"
    agent.contract_size = 1.0
    entry = dt.datetime(2026, 4, 28, 10, 0, tzinfo=dt.UTC)
    exit_ = dt.datetime(2026, 4, 28, 10, 5, tzinfo=dt.UTC)

    agent._write_trade_log(
        direction=-1,
        entry_price=76498.0,
        exit_price=76472.88,
        entry_time=entry,
        exit_time=exit_,
        pnl_usd=0.2512,
        pnl_pts=25.12,
        mfe=32.62,
        mae=25.62,
        quantity=0.01,
    )

    record = json.loads((tmp_path / "data" / "trade_log.jsonl").read_text().splitlines()[0])
    assert record["mfe"] == pytest.approx(0.3262)
    assert record["mae"] == pytest.approx(0.2562)
    assert record["mfe_points"] == 32.62
    assert record["mae_points"] == 25.62


def test_exit_lifecycle_data_includes_microstructure_inputs():
    agent = _agent_stub()
    agent._realized_vol = lambda: 0.003
    agent._depth_ratio = lambda: 2.0
    agent._active_kurtosis_threshold = lambda: 5.1
    agent.policy.current_regime = "TRANSITIONAL"
    agent.policy.current_zeta = 1.2

    data = agent._current_exit_lifecycle_data(
        entry_price=4610.0,
        fill_price=4612.0,
        pnl_pts=2.0,
        pnl_usd=2.0,
        mfe=3.0,
        mae=0.5,
        quantity=0.01,
        capture_ratio=0.67,
        ticks_held=12,
        close_reason="runway_capture",
        cb_tripped=["kurtosis"],
        close_drawdown_pct=0.01,
        close_cb_size_mult=0.5,
    )

    assert data["exit_depth_bid"] == 2.5
    assert data["exit_mfe_usd"] == 3.0
    assert data["exit_mae_usd"] == 0.5
    assert data["exit_depth_ask"] == 1.25
    assert data["exit_has_real_l2_sizes"] is True
    assert data["exit_l2_snapshot"]["bids"] == [[4611.9, 1.5]]
    assert data["exit_imbalance"] == 0.12
    assert data["exit_vpin_z"] == 1.7
    assert data["exit_var_95"] == 0.002
    assert data["exit_kurtosis"] == 4.2
    assert data["exit_kurtosis_threshold"] == 5.1
    assert data["exit_realized_vol"] == 0.003
