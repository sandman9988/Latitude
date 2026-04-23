from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.monitoring.reward_shaping_monitor import RewardShapingMonitor
from src.persistence.learned_parameters import LearnedParametersManager


def _write_json(path: Path, payload):
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")


def _trade(exit_dt: datetime, pnl: float, mfe: float, winner_to_loser: bool = False) -> dict:
    return {
        "exit_time": exit_dt.isoformat(),
        "pnl": pnl,
        "mfe": mfe,
        "winner_to_loser": winner_to_loser,
    }


def test_ranging_recommends_zero_trade_enforcement(tmp_path):
    now = datetime.now(UTC)
    trades = [_trade(now - timedelta(minutes=10), 5.0, 8.0)]
    decision_log = [
        {"timestamp": (now - timedelta(minutes=8)).isoformat(), "details": {"action": 1}},
        {"timestamp": (now - timedelta(minutes=6)).isoformat(), "details": {"action": 2}},
    ]

    trade_log_path = tmp_path / "trade_log.jsonl"
    trade_log_path.write_text("\n".join(__import__("json").dumps(t) for t in trades) + "\n", encoding="utf-8")

    decision_log_path = tmp_path / "decision_log.json"
    _write_json(decision_log_path, decision_log)

    pm = LearnedParametersManager(persistence_path=tmp_path / "learned_parameters.json")
    monitor = RewardShapingMonitor(
        symbol="XAUUSD",
        timeframe="M5",
        broker="default",
        param_manager=pm,
        trade_log_path=trade_log_path,
        decision_log_path=decision_log_path,
        output_path=tmp_path / "reward_shaping_monitor.json",
    )

    result = monitor.run(current_regime="TRANSITIONAL")
    reasons = {r["reason"] for r in result["recommendations"]}
    assert "enforce_zero_trades_ranging" in reasons


def test_trending_recommends_more_trades_when_underparticipating(tmp_path):
    now = datetime.now(UTC)
    trades = [_trade(now - timedelta(minutes=5), 2.0, 10.0)]
    decision_log = []
    for i in range(10):
        decision_log.append({"timestamp": (now - timedelta(minutes=55 - i)).isoformat(), "details": {"action": 1}})

    trade_log_path = tmp_path / "trade_log.jsonl"
    trade_log_path.write_text("\n".join(__import__("json").dumps(t) for t in trades) + "\n", encoding="utf-8")

    decision_log_path = tmp_path / "decision_log.json"
    _write_json(decision_log_path, decision_log)

    pm = LearnedParametersManager(persistence_path=tmp_path / "learned_parameters.json")
    monitor = RewardShapingMonitor(
        symbol="XAUUSD",
        timeframe="M5",
        broker="default",
        param_manager=pm,
        trade_log_path=trade_log_path,
        decision_log_path=decision_log_path,
        output_path=tmp_path / "reward_shaping_monitor.json",
    )

    result = monitor.run(current_regime="TRENDING")
    reasons = {r["reason"] for r in result["recommendations"]}
    assert "raise_trade_participation_trending" in reasons


def test_mean_reverting_recommends_selectivity_and_wtl_penalty(tmp_path):
    now = datetime.now(UTC)
    trades = [
        _trade(now - timedelta(minutes=30), 1.0, 10.0, winner_to_loser=True),
        _trade(now - timedelta(minutes=20), -1.0, 9.0),
        _trade(now - timedelta(minutes=10), 0.5, 8.0),
    ]

    trade_log_path = tmp_path / "trade_log.jsonl"
    trade_log_path.write_text("\n".join(__import__("json").dumps(t) for t in trades) + "\n", encoding="utf-8")

    pm = LearnedParametersManager(persistence_path=tmp_path / "learned_parameters.json")
    monitor = RewardShapingMonitor(
        symbol="XAUUSD",
        timeframe="M5",
        broker="default",
        param_manager=pm,
        trade_log_path=trade_log_path,
        decision_log_path=tmp_path / "missing_decision_log.json",
        output_path=tmp_path / "reward_shaping_monitor.json",
    )

    result = monitor.run(current_regime="MEAN_REVERTING")
    reasons = {r["reason"] for r in result["recommendations"]}
    assert "increase_selectivity_mean_reverting" in reasons
    assert "tighten_participation_mean_reverting" in reasons
    assert "penalize_winner_to_loser_paths" in reasons


def test_avg_capture_prefers_normalized_capture_ratio(tmp_path):
    monitor = RewardShapingMonitor(
        symbol="XAUUSD",
        timeframe="M5",
        broker="default",
        param_manager=LearnedParametersManager(persistence_path=tmp_path / "learned_parameters.json"),
        trade_log_path=tmp_path / "missing_trade_log.jsonl",
        decision_log_path=tmp_path / "missing_decision_log.json",
        output_path=tmp_path / "reward_shaping_monitor.json",
    )

    assert monitor._avg_capture([{"pnl": 100.0, "mfe": 1.0, "capture_ratio": 0.25}]) == 0.25
