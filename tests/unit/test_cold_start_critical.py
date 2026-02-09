"""
Tests for cold_start_manager.py — Tier 2: production demotion dispatch.

Covers line 186: check_graduation() → _check_production_demotion() when in PRODUCTION phase.
"""

import pytest
import numpy as np

from src.core.cold_start_manager import ColdStartManager, DemotionThresholds, WarmupPhase


@pytest.fixture()
def production_csm():
    """ColdStartManager already in PRODUCTION phase with enough trades."""
    csm = ColdStartManager(
        demotion=DemotionThresholds(sharpe=0.2, win_rate=0.40, drawdown=0.30),
    )
    csm.current_phase = WarmupPhase.PRODUCTION
    csm.trades_in_current_phase = 60  # > 50 threshold
    return csm


class TestProductionDemotionDispatch:
    """Line 186: PRODUCTION phase dispatches to _check_production_demotion."""

    def test_bad_sharpe_triggers_demotion(self, production_csm):
        """Sharpe below threshold → demote to MICRO_POSITIONS."""
        # Generate pnls with very low Sharpe (random around 0 with high variance)
        rng = np.random.default_rng(42)
        production_csm.current_pnls = list(rng.normal(-0.5, 5.0, 60))
        result = production_csm.check_graduation()
        assert result == WarmupPhase.MICRO_POSITIONS

    def test_bad_win_rate_triggers_demotion(self, production_csm):
        """Win rate below 40% → demote."""
        # 20% wins (12 of 60)
        production_csm.current_pnls = [1.0] * 12 + [-1.0] * 48
        result = production_csm.check_graduation()
        assert result == WarmupPhase.MICRO_POSITIONS

    def test_bad_drawdown_triggers_demotion(self, production_csm):
        """Max drawdown > 30% → demote."""
        # Create a sequence with severe drawdown
        production_csm.current_pnls = [10.0] * 10 + [-20.0] * 10 + [0.5] * 40
        result = production_csm.check_graduation()
        assert result == WarmupPhase.MICRO_POSITIONS

    def test_good_performance_no_demotion(self, production_csm):
        """Healthy production metrics → no demotion (returns None)."""
        # Positive, consistent returns → good Sharpe, high win rate, low drawdown
        production_csm.current_pnls = [1.0] * 40 + [-0.3] * 20
        result = production_csm.check_graduation()
        assert result is None

    def test_insufficient_trades_no_demotion(self):
        """< 50 trades → too early to judge, return None."""
        csm = ColdStartManager()
        csm.current_phase = WarmupPhase.PRODUCTION
        csm.trades_in_current_phase = 30
        csm.current_pnls = [-10.0] * 30  # Terrible, but not enough trades
        result = csm.check_graduation()
        assert result is None

    def test_non_production_phases_dont_demote(self):
        """check_graduation on non-PRODUCTION phases should call their own graduation logic."""
        csm = ColdStartManager()
        csm.current_phase = WarmupPhase.OBSERVATION
        csm.bars_in_current_phase = 5  # Not enough to graduate
        result = csm.check_graduation()
        assert result is None  # Not ready yet, but no demotion

    def test_graduate_resets_counters(self, production_csm):
        """After demotion, graduate() resets phase counters."""
        production_csm.current_pnls = [-5.0] * 60
        next_phase = production_csm.check_graduation()
        assert next_phase == WarmupPhase.MICRO_POSITIONS

        production_csm.graduate(next_phase)
        assert production_csm.current_phase == WarmupPhase.MICRO_POSITIONS
        assert production_csm.bars_in_current_phase == 0
        assert production_csm.trades_in_current_phase == 0
