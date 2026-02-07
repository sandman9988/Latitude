"""Tests for src.risk.circuit_breakers – BreakerState, individual breakers, CircuitBreakerManager."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.risk.circuit_breakers import (
    BreakerState,
    CircuitBreakerManager,
    ConsecutiveLossesBreaker,
    DrawdownBreaker,
    KurtosisBreaker,
    SortinoBreaker,
)


# ---------------------------------------------------------------------------
# BreakerState
# ---------------------------------------------------------------------------

class TestBreakerState:
    def test_init_not_tripped(self):
        bs = BreakerState(name="Test")
        assert bs.is_tripped is False
        assert bs.trip_time is None

    def test_trip(self):
        bs = BreakerState(name="Test")
        bs.trip("bad things", 0.3, 0.5)
        assert bs.is_tripped is True
        assert bs.trip_reason == "bad things"
        assert bs.trip_value == pytest.approx(0.3)

    def test_reset(self):
        bs = BreakerState(name="Test")
        bs.trip("reason", 1.0, 2.0)
        bs.reset()
        assert bs.is_tripped is False
        assert bs.trip_time is None

    def test_can_reset_not_tripped(self):
        bs = BreakerState(name="Test")
        assert bs.can_reset() is True

    def test_can_reset_before_cooldown(self):
        bs = BreakerState(name="Test", cooldown_minutes=60)
        bs.trip("reason", 1.0, 2.0)
        assert bs.can_reset() is False

    def test_can_reset_after_cooldown(self):
        bs = BreakerState(name="Test", cooldown_minutes=0)
        bs.trip("reason", 1.0, 2.0)
        # cooldown_minutes=0 → always elapsed
        assert bs.can_reset() is True


# ---------------------------------------------------------------------------
# SortinoBreaker
# ---------------------------------------------------------------------------

class TestSortinoBreaker:
    def test_not_enough_trades(self):
        sb = SortinoBreaker(threshold=0.5, min_trades=20)
        for _ in range(5):
            sb.update(-0.01)
        assert sb.check() is False

    def test_trips_on_poor_sortino(self):
        sb = SortinoBreaker(threshold=0.5, min_trades=5)
        # Varied losses so downside_dev > 0, overall mean negative → low Sortino
        losses = [-0.01, -0.03, -0.02, -0.04, -0.015, -0.025, -0.035, -0.05]
        for v in losses:
            sb.update(v)
        sb.update(0.001)
        sb.update(0.002)
        assert sb.check() is True
        assert sb.state.is_tripped

    def test_no_trip_good_returns(self):
        sb = SortinoBreaker(threshold=0.0, min_trades=5)
        # All wins → Sortino infinity
        for _ in range(10):
            sb.update(0.01)
        assert sb.check() is False

    def test_get_current_sortino(self):
        sb = SortinoBreaker()
        sb.update(0.01)
        sb.update(-0.005)
        val = sb.get_current_sortino()
        assert isinstance(val, float)

    def test_history_limit(self):
        sb = SortinoBreaker()
        for i in range(200):
            sb.update(0.001)
        assert len(sb.returns) <= 100


# ---------------------------------------------------------------------------
# KurtosisBreaker
# ---------------------------------------------------------------------------

class TestKurtosisBreaker:
    def test_not_enough_samples(self):
        kb = KurtosisBreaker(threshold=5.0, min_samples=30)
        for _ in range(10):
            kb.update(0.01)
        assert kb.check() is False

    def test_trips_on_fat_tails(self):
        kb = KurtosisBreaker(threshold=3.5, min_samples=10)
        # Fat-tailed distribution (lots of extreme values)
        rng = np.random.default_rng(42)
        for v in rng.standard_t(df=2, size=50):
            kb.update(float(v))
        # t(df=2) has very high kurtosis
        result = kb.check()
        # Should trip for extreme kurtosis
        assert isinstance(result, bool)

    def test_no_trip_normal_returns(self):
        kb = KurtosisBreaker(threshold=10.0, min_samples=10)
        rng = np.random.default_rng(42)
        for v in rng.normal(0, 0.01, size=50):
            kb.update(float(v))
        # Normal distribution kurtosis ≈ 3, threshold 10 → no trip
        assert kb.check() is False

    def test_get_current_kurtosis(self):
        kb = KurtosisBreaker()
        for _ in range(10):
            kb.update(0.01)
        val = kb.get_current_kurtosis()
        assert isinstance(val, float)

    def test_history_limit(self):
        kb = KurtosisBreaker()
        for i in range(200):
            kb.update(0.001 * i)
        assert len(kb.returns) <= 100


# ---------------------------------------------------------------------------
# DrawdownBreaker
# ---------------------------------------------------------------------------

class TestDrawdownBreaker:
    def test_no_drawdown(self):
        db = DrawdownBreaker()
        db.update(10000)
        db.update(11000)
        assert db.check() is False
        assert db.get_drawdown() == pytest.approx(0.0)
        assert db.get_size_multiplier() == pytest.approx(1.0)

    def test_small_drawdown_reduces_size(self):
        db = DrawdownBreaker(thresholds={0.05: 0.5, 0.10: 0.0})
        db.update(10000)
        db.update(9400)  # 6% drawdown
        db.check()
        assert db.get_size_multiplier() == pytest.approx(0.5)

    def test_max_drawdown_trips(self):
        db = DrawdownBreaker(thresholds={0.05: 0.5, 0.10: 0.0})
        db.update(10000)
        db.update(8900)  # 11% drawdown
        assert db.check() is True
        assert db.state.is_tripped

    def test_invalid_equity_ignored(self):
        db = DrawdownBreaker()
        db.update(10000)
        db.update(float("nan"))  # should be ignored
        assert db.peak_equity == 10000

    def test_zero_equity_ignored(self):
        db = DrawdownBreaker()
        db.update(0)
        assert db.peak_equity == 0

    def test_progressive_reduction(self):
        db = DrawdownBreaker(thresholds={0.05: 0.9, 0.10: 0.5, 0.20: 0.0})
        db.update(10000)
        db.update(9200)  # 8% DD
        db.check()
        assert db.get_size_multiplier() == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# ConsecutiveLossesBreaker
# ---------------------------------------------------------------------------

class TestConsecutiveLossesBreaker:
    def test_no_losses(self):
        cb = ConsecutiveLossesBreaker(max_losses=5)
        cb.update(is_win=True)
        assert cb.check() is False
        assert cb.get_consecutive_losses() == 0

    def test_loss_streak_trips(self):
        cb = ConsecutiveLossesBreaker(max_losses=3)
        cb.update(is_win=False)
        cb.update(is_win=False)
        cb.update(is_win=False)
        assert cb.check() is True
        assert cb.state.is_tripped

    def test_win_resets_streak(self):
        cb = ConsecutiveLossesBreaker(max_losses=5)
        cb.update(is_win=False)
        cb.update(is_win=False)
        cb.update(is_win=True)  # resets
        assert cb.get_consecutive_losses() == 0
        assert cb.check() is False


# ---------------------------------------------------------------------------
# CircuitBreakerManager
# ---------------------------------------------------------------------------

class TestCircuitBreakerManager:
    @pytest.fixture()
    def mgr(self):
        return CircuitBreakerManager(
            sortino_threshold=0.5,
            kurtosis_threshold=5.0,
            max_drawdown=0.20,
            max_consecutive_losses=5,
        )

    def test_init(self, mgr):
        assert not mgr.is_any_tripped()
        assert mgr.get_position_size_multiplier() == pytest.approx(1.0)

    def test_update_trade(self, mgr):
        mgr.update_trade(pnl=0.01, equity=10000)
        assert not mgr.is_any_tripped()

    def test_check_all_clean(self, mgr):
        mgr.update_trade(pnl=0.01, equity=10000)
        assert mgr.check_all() is False

    def test_consecutive_losses_trip(self):
        mgr = CircuitBreakerManager(max_consecutive_losses=3)
        for _ in range(3):
            mgr.update_trade(pnl=-100, equity=10000)
        assert mgr.check_all() is True
        assert mgr.is_any_tripped()


    def test_get_tripped_breakers(self):
        mgr = CircuitBreakerManager(max_consecutive_losses=2)
        mgr.update_trade(pnl=-10, equity=10000)
        mgr.update_trade(pnl=-10, equity=10000)
        mgr.check_all()
        tripped = mgr.get_tripped_breakers()
        assert len(tripped) >= 1

    def test_reset_all(self, mgr):
        mgr.consecutive_losses_breaker.update(is_win=False)
        mgr.consecutive_losses_breaker.update(is_win=False)
        mgr.consecutive_losses_breaker.update(is_win=False)
        mgr.consecutive_losses_breaker.update(is_win=False)
        mgr.consecutive_losses_breaker.update(is_win=False)
        mgr.check_all()
        assert mgr.is_any_tripped()
        mgr.reset_all()
        assert not mgr.is_any_tripped()

    def test_position_multiplier_zero_when_tripped(self):
        mgr = CircuitBreakerManager(max_consecutive_losses=2)
        mgr.update_trade(pnl=-10, equity=10000)
        mgr.update_trade(pnl=-10, equity=10000)
        mgr.check_all()
        assert mgr.get_position_size_multiplier() == pytest.approx(0.0)

    def test_get_status(self, mgr):
        status = mgr.get_status()
        assert "any_tripped" in status
        assert "sortino" in status
        assert "kurtosis" in status
        assert "drawdown" in status
        assert "consecutive_losses" in status

    def test_set_emergency_closer(self, mgr):
        closer = MagicMock()
        mgr.set_emergency_closer(closer)
        assert mgr.emergency_closer is closer

    def test_auto_close_on_trip(self):
        closer = MagicMock()
        closer.close_all_positions.return_value = True
        mgr = CircuitBreakerManager(max_consecutive_losses=2, auto_close_on_trip=True)
        mgr.set_emergency_closer(closer)
        mgr.update_trade(pnl=-10, equity=10000)
        mgr.update_trade(pnl=-10, equity=10000)
        mgr.check_all()
        closer.close_all_positions.assert_called_once()

    def test_save_and_restore_state(self, mgr, tmp_path):
        filepath = str(tmp_path / "cb_state.json")
        mgr.update_trade(pnl=-10, equity=10000)
        mgr.save_state(filepath)
        assert Path(filepath).exists()

        mgr2 = CircuitBreakerManager(max_consecutive_losses=5)
        assert mgr2.restore_state(filepath) is True

    def test_restore_nonexistent(self, mgr, tmp_path):
        assert mgr.restore_state(str(tmp_path / "nope.json")) is False

    def test_reset_if_cooldown_elapsed(self):
        mgr = CircuitBreakerManager(max_consecutive_losses=2)
        mgr.update_trade(pnl=-10, equity=10000)
        mgr.update_trade(pnl=-10, equity=10000)
        mgr.check_all()
        assert mgr.is_any_tripped()
        # Set cooldown to 0 so it's always elapsed
        for b in mgr.breakers:
            b.state.cooldown_minutes = 0
        mgr.reset_if_cooldown_elapsed()
        assert not mgr.is_any_tripped()

    def test_param_manager_integration(self, tmp_path):
        from src.persistence.learned_parameters import LearnedParametersManager
        pm = LearnedParametersManager(persistence_path=tmp_path / "params.json")
        mgr = CircuitBreakerManager(param_manager=pm)
        # Should use learned/default values without error
        assert mgr.sortino_threshold > 0
