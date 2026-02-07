"""Tests for src.monitoring.activity_monitor – ActivityMonitor & CounterfactualAnalyzer."""

from datetime import UTC, datetime

import pytest

from src.monitoring.activity_monitor import (
    ActivityMonitor,
    CounterfactualAnalyzer,
)


# ---------------------------------------------------------------------------
# ActivityMonitor
# ---------------------------------------------------------------------------


class TestActivityMonitorInit:
    def test_default_init(self):
        am = ActivityMonitor(max_bars_inactive=30, min_trades_per_day=10, exploration_boost=0.3)
        assert am.max_bars_inactive == 30
        assert am.bars_since_trade == 0
        assert am.activity_score == pytest.approx(1.0)

    def test_phase_maturity_blend(self):
        am = ActivityMonitor(phase_maturity=1.0, max_bars_inactive=50)
        assert am.phase_maturity == pytest.approx(1.0)


class TestOnBarClose:
    def test_increments_bars(self):
        am = ActivityMonitor(max_bars_inactive=100)
        am.on_bar_close()
        assert am.bars_since_trade == 1
        assert am.total_bars == 1

    def test_activity_decays(self):
        am = ActivityMonitor(max_bars_inactive=100)
        initial = am.activity_score
        am.on_bar_close()
        assert am.activity_score < initial

    def test_stagnation_triggers(self):
        am = ActivityMonitor(max_bars_inactive=5, exploration_boost=0.3)
        for _ in range(6):
            am.on_bar_close()
        assert am.is_stagnant is True
        assert am.should_explore is True


class TestOnTradeExecuted:
    def test_resets_bars_since_trade(self):
        am = ActivityMonitor(max_bars_inactive=100)
        for _ in range(10):
            am.on_bar_close()
        am.on_trade_executed()
        assert am.bars_since_trade == 0
        assert am.total_trades == 1

    def test_resets_stagnation(self):
        am = ActivityMonitor(max_bars_inactive=5)
        for _ in range(6):
            am.on_bar_close()
        assert am.is_stagnant is True
        am.on_trade_executed()
        assert am.is_stagnant is False
        assert am.should_explore is False

    def test_boosts_activity_score(self):
        am = ActivityMonitor(max_bars_inactive=100)
        # Decay score
        for _ in range(20):
            am.on_bar_close()
        low_score = am.activity_score
        am.on_trade_executed()
        assert am.activity_score > low_score

    def test_records_timestamp(self):
        am = ActivityMonitor(max_bars_inactive=100)
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
        am.on_trade_executed(timestamp=ts)
        assert am.last_trade_time == ts
        assert len(am.trade_timestamps) == 1


class TestExplorationBonus:
    def test_zero_when_not_stagnant(self):
        am = ActivityMonitor(max_bars_inactive=100, exploration_boost=0.3)
        assert am.get_exploration_bonus() == pytest.approx(0.0)

    def test_positive_when_stagnant(self):
        am = ActivityMonitor(max_bars_inactive=5, exploration_boost=0.3)
        for _ in range(6):
            am.on_bar_close()
        bonus = am.get_exploration_bonus()
        assert bonus > 0


class TestInactivityPenalty:
    def test_zero_when_active(self):
        am = ActivityMonitor(max_bars_inactive=100)
        assert am.get_inactivity_penalty() == pytest.approx(0.0)

    def test_negative_when_stagnant(self):
        am = ActivityMonitor(max_bars_inactive=5)
        for _ in range(10):
            am.on_bar_close()
        penalty = am.get_inactivity_penalty()
        assert penalty < 0

    def test_penalty_grows_with_inactivity(self):
        am = ActivityMonitor(max_bars_inactive=5)
        for _ in range(6):
            am.on_bar_close()
        p1 = am.get_inactivity_penalty()
        for _ in range(5):
            am.on_bar_close()
        p2 = am.get_inactivity_penalty()
        assert p2 < p1  # More negative


class TestTradeFrequency:
    def test_zero_without_trades(self):
        am = ActivityMonitor(max_bars_inactive=100)
        assert am.get_trade_frequency(24.0) == pytest.approx(0.0)

    def test_with_recent_trades(self):
        am = ActivityMonitor(max_bars_inactive=100)
        for _ in range(5):
            am.on_trade_executed()
        freq = am.get_trade_frequency(24.0)
        assert freq > 0

    def test_is_below_target_frequency(self):
        am = ActivityMonitor(max_bars_inactive=100, min_trades_per_day=10.0)
        assert am.is_below_target_frequency() is True


class TestGetMetrics:
    def test_returns_all_keys(self):
        am = ActivityMonitor(max_bars_inactive=100)
        am.on_bar_close()
        am.on_trade_executed()
        m = am.get_metrics()
        for key in ("bars_since_trade", "total_bars", "total_trades",
                     "activity_score", "is_stagnant", "exploration_active",
                     "trade_freq_24h", "trade_freq_1h",
                     "inactivity_penalty", "exploration_bonus"):
            assert key in m


# ---------------------------------------------------------------------------
# CounterfactualAnalyzer
# ---------------------------------------------------------------------------


class TestCounterfactualAnalyzer:
    def test_init(self):
        ca = CounterfactualAnalyzer(lookback_bars=20)
        assert ca.lookback_bars == 20

    def test_perfect_exit(self):
        """Exit exactly at MFE → small timing bonus, no penalty."""
        ca = CounterfactualAnalyzer()
        reward, metrics = ca.analyze_exit(
            entry_price=100.0,
            exit_price=105.0,
            mfe=5.0,
            mfe_bar_offset=1,
            direction=1,
        )
        assert metrics["efficiency"] == pytest.approx(1.0)
        assert metrics["early_exit_penalty"] == pytest.approx(0.0)
        assert metrics["timing_bonus"] > 0  # mfe_bar_offset ≤ 2

    def test_early_exit_from_winner(self):
        """Exit before MFE with profit → penalty for leaving money."""
        ca = CounterfactualAnalyzer()
        reward, metrics = ca.analyze_exit(
            entry_price=100.0,
            exit_price=103.0,  # Captured 3
            mfe=5.0,           # Could have captured 5
            mfe_bar_offset=10, # Exited far from MFE
            direction=1,
        )
        assert metrics["early_exit_penalty"] < 0
        assert metrics["missed_pnl"] > 0

    def test_short_direction(self):
        """Short trade: entry_price > exit_price is profitable."""
        ca = CounterfactualAnalyzer()
        reward, metrics = ca.analyze_exit(
            entry_price=100.0,
            exit_price=95.0,   # Short profit
            mfe=5.0,
            mfe_bar_offset=1,
            direction=-1,
        )
        assert metrics["actual_pnl"] > 0

    def test_losing_trade_no_early_exit_penalty(self):
        """Losing trade → no early-exit penalty (only applies to winners)."""
        ca = CounterfactualAnalyzer()
        reward, metrics = ca.analyze_exit(
            entry_price=100.0,
            exit_price=98.0,  # Loss
            mfe=2.0,
            mfe_bar_offset=15,
            direction=1,
        )
        assert metrics["early_exit_penalty"] == pytest.approx(0.0)
