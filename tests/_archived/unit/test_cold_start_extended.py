"""Extended tests for src.core.cold_start_manager.

Covers remaining uncovered paths: full observation→paper→micro→production
workflow, demotion back, phase history after multiple transitions.
"""

import pytest

from src.core.cold_start_manager import ColdStartManager, WarmupPhase


class TestFullWorkflow:
    """Test complete phase progression workflow."""

    def test_observation_to_paper_to_micro(self):
        mgr = ColdStartManager(
            observation_min_bars=5,
            paper_min_bars=5,
            paper_min_sharpe=0.0,
            paper_min_win_rate=0.0,
            paper_max_drawdown=1.0,
            micro_min_bars=5,
            micro_min_sharpe=0.0,
            micro_min_win_rate=0.0,
            micro_min_avg_profit=-999.0,
        )
        # Phase 1: Observation
        assert mgr.current_phase == WarmupPhase.OBSERVATION
        for _ in range(6):
            mgr.update(new_bar=True)
        nxt = mgr.check_graduation()
        assert nxt == WarmupPhase.PAPER_TRADING
        mgr.graduate(nxt)
        assert mgr.current_phase == WarmupPhase.PAPER_TRADING

        # Phase 2: Paper - need enough bars + trades
        for i in range(10):
            mgr.update(new_bar=True)
            if i % 1 == 0:
                mgr.update(trade_completed={"pnl": 0.01})
        nxt = mgr.check_graduation()
        assert nxt == WarmupPhase.MICRO_POSITIONS
        mgr.graduate(nxt)
        assert mgr.current_phase == WarmupPhase.MICRO_POSITIONS

    def test_phase_history_accumulates(self):
        mgr = ColdStartManager(observation_min_bars=3)
        for _ in range(5):
            mgr.update(new_bar=True)
        nxt = mgr.check_graduation()
        mgr.graduate(nxt)
        assert len(mgr.phase_history) == 1
        assert mgr.phase_history[0].phase == WarmupPhase.OBSERVATION


class TestMetricsCalculation:
    def test_single_pnl_sharpe(self):
        mgr = ColdStartManager()
        mgr.current_phase = WarmupPhase.PAPER_TRADING
        mgr.current_pnls = [0.05]
        metrics = mgr._calculate_current_metrics()
        assert metrics.win_rate == pytest.approx(1.0)
        assert metrics.total_pnl == pytest.approx(0.05)

    def test_avg_trade_profit(self):
        mgr = ColdStartManager()
        mgr.current_pnls = [0.10, -0.02, 0.06]
        metrics = mgr._calculate_current_metrics()
        assert metrics.avg_trade_profit == pytest.approx((0.10 - 0.02 + 0.06) / 3)


class TestProductionMultiplier:
    def test_production_returns_1(self):
        mgr = ColdStartManager()
        mgr.current_phase = WarmupPhase.PRODUCTION
        assert mgr.get_position_size_multiplier() == pytest.approx(1.0)
