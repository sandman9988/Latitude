"""Tests for src.monitoring.performance_tracker – PerformanceTracker."""

import datetime as dt

import pytest

from src.monitoring.performance_tracker import AgentAttribution, PerformanceTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = dt.datetime(2026, 1, 10, 8, 0, tzinfo=dt.UTC)
_T1 = dt.datetime(2026, 1, 10, 8, 30, tzinfo=dt.UTC)


def _add_trades(pt: PerformanceTracker, pnls: list[float]):
    """Add trades with the given PnL values."""
    for i, pnl in enumerate(pnls):
        pt.add_trade(
            pnl=pnl,
            entry_time=_T0 + dt.timedelta(hours=i),
            exit_time=_T1 + dt.timedelta(hours=i),
            direction="LONG",
            entry_price=100_000.0,
            exit_price=100_000.0 + pnl,
        )


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestPerformanceTrackerInit:
    def test_defaults(self):
        pt = PerformanceTracker()
        assert pt.total_trades == 0
        assert pt.current_equity == 10_000.0

    def test_empty_metrics(self):
        pt = PerformanceTracker()
        m = pt.get_metrics()
        assert m["total_trades"] == 0
        assert m["win_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# add_trade
# ---------------------------------------------------------------------------

class TestAddTrade:
    def test_increments_count(self):
        pt = PerformanceTracker()
        _add_trades(pt, [50.0])
        assert pt.total_trades == 1
        assert pt.winning_trades == 1

    def test_loss_tracked(self):
        pt = PerformanceTracker()
        _add_trades(pt, [-30.0])
        assert pt.losing_trades == 1

    def test_equity_updated(self):
        pt = PerformanceTracker()
        _add_trades(pt, [50.0, -20.0])
        assert pt.current_equity == pytest.approx(10_030.0)

    def test_pnl_none_treated_as_zero(self):
        pt = PerformanceTracker()
        pt.add_trade(pnl=None, entry_time=_T0, exit_time=_T1,
                     direction="LONG", entry_price=100_000, exit_price=100_000)
        assert pt.total_pnl == pytest.approx(0.0)

    def test_negative_mfe_clamped(self):
        pt = PerformanceTracker()
        pt.add_trade(pnl=10, entry_time=_T0, exit_time=_T1,
                     direction="LONG", entry_price=100_000, exit_price=100_010,
                     mfe=-5)
        assert pt.trades[0]["mfe"] == pytest.approx(0.0)

    def test_invalid_quality_string_replaced(self):
        pt = PerformanceTracker()
        pt.add_trade(
            pnl=10,
            entry_time=_T0,
            exit_time=_T1,
            direction="LONG",
            entry_price=100_000,
            exit_price=100_010,
            attribution=AgentAttribution(
                trigger_quality="GARBAGE",
                harvester_quality="INVALID",
            ),
        )
        assert pt.trades[0]["trigger_quality"] == "N/A"
        assert pt.trades[0]["harvester_quality"] == "N/A"


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_no_drawdown_on_wins(self):
        pt = PerformanceTracker()
        _add_trades(pt, [50.0, 50.0, 50.0])
        assert pt.max_drawdown == pytest.approx(0.0)

    def test_drawdown_after_loss(self):
        pt = PerformanceTracker()
        _add_trades(pt, [100.0, -50.0])
        assert pt.max_drawdown > 0

    def test_peak_equity_updates(self):
        pt = PerformanceTracker()
        _add_trades(pt, [200.0])
        assert pt.peak_equity == pytest.approx(10_200.0)


# ---------------------------------------------------------------------------
# Consecutive streaks
# ---------------------------------------------------------------------------

class TestStreaks:
    def test_consecutive_wins(self):
        pt = PerformanceTracker()
        _add_trades(pt, [10.0, 20.0, 30.0])
        assert pt.max_consecutive_wins == 3

    def test_consecutive_losses(self):
        pt = PerformanceTracker()
        _add_trades(pt, [-10.0, -20.0])
        assert pt.max_consecutive_losses == 2

    def test_streak_resets(self):
        pt = PerformanceTracker()
        _add_trades(pt, [10.0, 10.0, -5.0, 10.0])
        assert pt.max_consecutive_wins == 2
        assert pt.consecutive_wins == 1


# ---------------------------------------------------------------------------
# get_metrics
# ---------------------------------------------------------------------------

class TestGetMetrics:
    def test_win_rate(self):
        pt = PerformanceTracker()
        _add_trades(pt, [10, 20, -5])
        m = pt.get_metrics()
        assert m["win_rate"] == pytest.approx(2 / 3)

    def test_profit_factor(self):
        pt = PerformanceTracker()
        _add_trades(pt, [100, -50])
        m = pt.get_metrics()
        assert m["profit_factor"] == pytest.approx(2.0)

    def test_profit_factor_no_losses(self):
        pt = PerformanceTracker()
        _add_trades(pt, [100, 200])
        m = pt.get_metrics()
        assert m["profit_factor"] == float("inf")

    def test_sharpe_ratio_nonzero(self):
        pt = PerformanceTracker()
        _add_trades(pt, [10, 20, -5, 15, -3])
        m = pt.get_metrics()
        assert m["sharpe_ratio"] != 0.0

    def test_sortino_ratio(self):
        pt = PerformanceTracker()
        _add_trades(pt, [10, 20, -5, 15, -3])
        m = pt.get_metrics()
        assert "sortino_ratio" in m

    def test_omega_ratio(self):
        pt = PerformanceTracker()
        _add_trades(pt, [10, 20, -5, 15, -3])
        m = pt.get_metrics()
        assert m["omega_ratio"] > 0

    def test_total_return(self):
        pt = PerformanceTracker()
        _add_trades(pt, [1000])  # 10% return on 10000
        m = pt.get_metrics()
        assert m["total_return"] == pytest.approx(0.1)

    def test_wtl_count(self):
        pt = PerformanceTracker()
        pt.add_trade(pnl=-10, entry_time=_T0, exit_time=_T1,
                     direction="LONG", entry_price=100_000, exit_price=99_990,
                     winner_to_loser=True)
        m = pt.get_metrics()
        assert m["winner_to_loser_count"] == 1

    def test_all_expected_keys(self):
        pt = PerformanceTracker()
        _add_trades(pt, [10, -5])
        m = pt.get_metrics()
        for key in ("total_trades", "winning_trades", "losing_trades", "win_rate",
                     "total_pnl", "avg_winner", "avg_loser", "profit_factor",
                     "expectancy", "sharpe_ratio", "sortino_ratio", "omega_ratio",
                     "initial_equity", "current_equity", "total_return",
                     "max_drawdown", "current_drawdown",
                     "max_consecutive_wins", "max_consecutive_losses",
                     "winner_to_loser_count"):
            assert key in m


# ---------------------------------------------------------------------------
# Dashboard / history
# ---------------------------------------------------------------------------

class TestDashboardAndHistory:
    def test_print_dashboard_returns_str(self):
        pt = PerformanceTracker()
        _add_trades(pt, [10, -5])
        d = pt.print_dashboard()
        assert "PERFORMANCE DASHBOARD" in d

    def test_get_trade_history_copy(self):
        pt = PerformanceTracker()
        _add_trades(pt, [10])
        h = pt.get_trade_history()
        assert len(h) == 1
        # Should be a copy – modifications shouldn't affect original
        h.clear()
        assert len(pt.trades) == 1
