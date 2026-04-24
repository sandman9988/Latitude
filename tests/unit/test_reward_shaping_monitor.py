from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.monitoring.reward_shaping_monitor import RewardShapingMonitor
from src.persistence.learned_parameters import LearnedParametersManager


def _write_json(path: Path, payload):
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")


def _trade(
    exit_dt: datetime,
    pnl: float,
    mfe: float,
    winner_to_loser: bool = False,
    symbol: str = "XAUUSD",
    timeframe_minutes: int = 5,
) -> dict:
    return {
        "exit_time": exit_dt.isoformat(),
        "symbol": symbol,
        "timeframe_minutes": timeframe_minutes,
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


def test_window_comparison_includes_24h_vs_7d_30d(tmp_path):
    now = datetime.now(UTC)
    trades = []
    for i in range(10):
        trades.append(_trade(now - timedelta(hours=2 * i), pnl=2.0, mfe=4.0))
    for i in range(15):
        trades.append(_trade(now - timedelta(days=2, hours=i), pnl=1.0, mfe=3.0))
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
        compare_short_window_hours=24,
        compare_baseline_7d_days=7,
        compare_baseline_30d_days=30,
    )

    result = monitor.run(current_regime="TRENDING")
    comp = result.get("window_comparison", {})
    assert comp.get("short_window_hours") == 24
    assert comp.get("baseline_7d_days") == 7
    assert comp.get("baseline_30d_days") == 30
    assert comp.get("short_window", {}).get("trades", 0) >= 1
    assert "delta_vs_7d" in comp
    assert "delta_vs_30d" in comp


def test_quality_guard_triggers_when_24h_metrics_collapse_vs_7d(tmp_path):
    now = datetime.now(UTC)
    trades = []
    # Baseline (older than 24h): strong winners
    for i in range(220):
        trades.append(_trade(now - timedelta(days=2, minutes=i), pnl=8.0, mfe=10.0))
    # Last 24h: poorer quality
    for i in range(40):
        pnl = -3.0 if i % 2 == 0 else 1.0
        trades.append(_trade(now - timedelta(hours=6, minutes=i), pnl=pnl, mfe=6.0))

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
        compare_short_window_hours=24,
        compare_baseline_7d_days=7,
    )

    result = monitor.run(current_regime="TRENDING")
    reasons = {r["reason"] for r in result["recommendations"]}
    assert "quality_guard_pf_pnl_drop" in reasons


def test_scope_filters_to_symbol_and_timeframe(tmp_path):
    now = datetime.now(UTC)
    trades = [
        _trade(now - timedelta(minutes=10), pnl=5.0, mfe=8.0, symbol="XAUUSD", timeframe_minutes=5),
        _trade(now - timedelta(minutes=9), pnl=4.0, mfe=7.0, symbol="XAUUSD", timeframe_minutes=15),
        _trade(now - timedelta(minutes=8), pnl=3.0, mfe=6.0, symbol="EURUSD", timeframe_minutes=5),
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
    result = monitor.run(current_regime="TRENDING")
    assert result["trade_count"] == 1
