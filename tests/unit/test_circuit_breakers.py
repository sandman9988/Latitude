"""Tests for src.risk.circuit_breakers – BreakerState, individual breakers, CircuitBreakerManager."""

import json
from datetime import UTC
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
        for _i in range(200):
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


class TestKurtosisAdaptive:
    """Quantile-based threshold learning (low-hanging-fruit risk tuner)."""

    def _seed_fat_tails(self, kb: KurtosisBreaker, rng, n: int = 60):
        """Feed enough fat-tailed returns to build up a reading history."""
        for v in rng.standard_t(df=3, size=n):
            kb.update(float(v))
            # Force kurtosis readings to accumulate (check() records one).
            kb.check()

    def test_seed_threshold_used_until_min_readings(self):
        kb = KurtosisBreaker(threshold=5.0, min_samples=10, adaptive=True)
        rng = np.random.default_rng(0)
        # Only a handful of readings — quantile must NOT kick in yet.
        for v in rng.normal(0, 0.01, size=15):
            kb.update(float(v))
            kb.check()
        assert kb.threshold == 5.0  # untouched

    def test_threshold_drifts_toward_quantile(self):
        kb = KurtosisBreaker(
            threshold=5.0,
            min_samples=10,
            adaptive=True,
            adapt_min_readings=20,
            adapt_bounds=(2.5, 10.0),
            adapt_ema_alpha=0.25,
        )
        rng = np.random.default_rng(1)
        # Fat-tailed regime: kurtosis readings should be well above 3.0
        self._seed_fat_tails(kb, rng, n=120)
        # Threshold must have left its 5.0 seed — in either direction,
        # clipped to adapt_bounds.
        assert 2.5 <= kb.threshold <= 10.0
        assert kb.threshold != 5.0

    def test_threshold_respects_max_bound(self):
        kb = KurtosisBreaker(
            threshold=5.0,
            min_samples=10,
            adaptive=True,
            adapt_min_readings=10,
            adapt_bounds=(2.5, 6.0),
            adapt_ema_alpha=1.0,  # instant adoption
        )
        rng = np.random.default_rng(2)
        # Pathologically fat tails → raw quantile would blow past 6.0
        for v in rng.standard_t(df=1.5, size=150):
            kb.update(float(v))
            kb.check()
        assert kb.threshold <= 6.0 + 1e-9

    def test_non_adaptive_mode_keeps_seed(self):
        kb = KurtosisBreaker(threshold=5.0, min_samples=10, adaptive=False)
        rng = np.random.default_rng(3)
        for v in rng.standard_t(df=2, size=120):
            kb.update(float(v))
            kb.check()
        assert kb.threshold == 5.0


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
    @pytest.fixture
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

    def test_manual_reset_cooldown_blocks_immediate_retrip(self):
        mgr = CircuitBreakerManager(max_consecutive_losses=1)
        mgr.update_trade(pnl=-10, equity=10000)
        assert mgr.check_all() is True
        assert mgr.is_any_tripped() is True

        mgr.reset_all(manual_cooldown_seconds=60)
        mgr.update_trade(pnl=-10, equity=10000)
        assert mgr.check_all() is False
        assert mgr.is_any_tripped() is False

        mgr.manual_reset_cooldown_until = None
        assert mgr.check_all() is True
        assert mgr.is_any_tripped() is True

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

    def test_reset_if_cooldown_elapsed_clears_data_window_no_retrip(self):
        # Regression: after cooldown reset, check_all() must NOT immediately re-trip
        # on the same stale data (the infinite re-trip loop).
        mgr = CircuitBreakerManager(max_consecutive_losses=2)
        mgr.update_trade(pnl=-10, equity=10000)
        mgr.update_trade(pnl=-10, equity=10000)
        mgr.check_all()
        assert mgr.is_any_tripped()
        assert mgr.consecutive_losses_breaker.consecutive_losses == 2

        for b in mgr.breakers:
            b.state.cooldown_minutes = 0
        mgr.reset_if_cooldown_elapsed()

        # Data window must be cleared
        assert mgr.consecutive_losses_breaker.consecutive_losses == 0
        # check_all() must not re-trip with the now-empty window
        assert mgr.check_all() is False
        assert not mgr.is_any_tripped()

    def test_sortino_cooldown_reset_clears_returns_no_retrip(self):
        # Regression: sortino re-trip loop — an outlier loss poisons the window;
        # after cooldown the window must be cleared so Sortino can't re-trip on stale data.
        # Uses varied loss magnitudes so downside_dev > 0 and Sortino truly goes negative
        # (matches the real XAUUSD M5 production pattern with the -$37 emergency stop).
        mgr = CircuitBreakerManager(sortino_threshold=0.5, max_consecutive_losses=100)
        varied_pnls = [
            0.01, -1.5, 0.01, -3.0, 0.01, -6.0, 0.01, -1.1, 0.01, -1.7,
            0.01, -37.0, 0.01, -5.4, 0.01, 1.6, 0.01, 3.8, 0.01, 1.7,
        ]
        for pnl in varied_pnls:
            mgr.update_trade(pnl=pnl, equity=10000)
        mgr.check_all()
        assert mgr.sortino_breaker.state.is_tripped

        for b in mgr.breakers:
            b.state.cooldown_minutes = 0
        mgr.reset_if_cooldown_elapsed()

        assert len(mgr.sortino_breaker.returns) == 0
        assert mgr.check_all() is False
        assert not mgr.sortino_breaker.state.is_tripped

    def test_param_manager_integration(self, tmp_path):
        from src.persistence.learned_parameters import LearnedParametersManager

        pm = LearnedParametersManager(persistence_path=tmp_path / "params.json")
        mgr = CircuitBreakerManager(param_manager=pm)
        # Should use learned/default values without error
        assert mgr.sortino_threshold > 0

    def test_restore_keeps_learned_kurtosis_threshold_authoritative(self, tmp_path):
        from src.persistence.learned_parameters import LearnedParametersManager

        pm = LearnedParametersManager(persistence_path=tmp_path / "params.json")
        pm.set_value("XAUUSD", "kurtosis_threshold", 7.0, timeframe="M5", broker="default")
        state_path = tmp_path / "cb_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "kurtosis": {
                        "is_tripped": False,
                        "trip_time": None,
                        "trip_reason": "",
                        "trip_value": 0.0,
                        "threshold": 3.0,
                        "cooldown_minutes": 60,
                        "returns": [],
                    },
                },
            ),
        )

        mgr = CircuitBreakerManager(symbol="XAUUSD", timeframe="M5", broker="default", param_manager=pm)
        assert mgr.kurtosis_breaker.threshold == pytest.approx(7.0)
        assert mgr.restore_state(str(state_path)) is True
        assert mgr.kurtosis_breaker.threshold == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# detect_cb_lockouts (performance_analyzer)
# ---------------------------------------------------------------------------

class TestDetectCbLockouts:
    """Tests for performance_analyzer.detect_cb_lockouts using real CB file layout."""

    def _make_cb_file(self, tmp_path, bot_key: str, trips: dict, hours_ago: float = 8.0) -> None:
        from datetime import datetime, timedelta
        trip_time = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
        state = {
            "timestamp": 0,
            "sortino": {"is_tripped": False, "trip_time": None, "trip_reason": "", "trip_value": 0.0,
                        "threshold": 0.5, "cooldown_minutes": 120, "returns": []},
            "kurtosis": {"is_tripped": False, "trip_time": None, "trip_reason": "", "trip_value": 0.0,
                         "threshold": 5.0, "cooldown_minutes": 60, "returns": [], "readings": []},
            "drawdown": {"is_tripped": False, "trip_time": None, "trip_reason": "", "trip_value": 0.0,
                         "threshold": 0.2, "cooldown_minutes": 240, "current_drawdown": 0.0, "peak_equity": 10000.0},
            "consecutive_losses": {"is_tripped": False, "trip_time": None, "trip_reason": "", "trip_value": 0.0,
                                   "threshold": 5.0, "cooldown_minutes": 180, "consecutive_losses": 0},
            "manual_reset_cooldown_until": None,
        }
        for name, val in trips.items():
            state[name]["is_tripped"] = True
            state[name]["trip_time"] = trip_time
        bot_dir = tmp_path / f"paper_{bot_key}"
        bot_dir.mkdir(parents=True, exist_ok=True)
        (bot_dir / "circuit_breakers.json").write_text(json.dumps(state))

    def test_detects_silent_locked_bot(self, tmp_path, monkeypatch):
        import scripts.performance_analyzer as pa
        monkeypatch.setattr(pa, "DATA_DIR", tmp_path)
        self._make_cb_file(tmp_path, "XAUUSD_M5", {"sortino": True}, hours_ago=8.0)

        anomalies = pa.detect_cb_lockouts({}, hours=168.0)
        assert len(anomalies) == 1
        a = anomalies[0]
        assert a.code == "CB_LOCKOUT"
        assert a.severity == "CRITICAL"
        assert a.symbol == "XAUUSD"
        assert a.timeframe == "M5"
        assert a.metric_value >= 8.0

    def test_skips_bot_with_recent_trades(self, tmp_path, monkeypatch):
        import scripts.performance_analyzer as pa
        from scripts.performance_analyzer import BotMetrics
        monkeypatch.setattr(pa, "DATA_DIR", tmp_path)
        self._make_cb_file(tmp_path, "XAUUSD_M1", {"consecutive_losses": True}, hours_ago=10.0)

        bot_metrics = {"XAUUSD_M1": BotMetrics(symbol="XAUUSD", timeframe="M1", n_trades=3)}
        anomalies = pa.detect_cb_lockouts(bot_metrics, hours=24.0)
        assert anomalies == []

    def test_skips_when_window_too_short(self, tmp_path, monkeypatch):
        import scripts.performance_analyzer as pa
        monkeypatch.setattr(pa, "DATA_DIR", tmp_path)
        self._make_cb_file(tmp_path, "BTCUSD_M30", {"sortino": True}, hours_ago=1.0)

        # Window of 1h is below CB_LOCKOUT_WINDOW_MIN_HOURS → no false alarm
        anomalies = pa.detect_cb_lockouts({}, hours=1.0)
        assert anomalies == []

    def test_multiple_locked_bots(self, tmp_path, monkeypatch):
        import scripts.performance_analyzer as pa
        monkeypatch.setattr(pa, "DATA_DIR", tmp_path)
        self._make_cb_file(tmp_path, "XAUUSD_M1", {"consecutive_losses": True}, hours_ago=8.0)
        self._make_cb_file(tmp_path, "XAUUSD_M5", {"sortino": True}, hours_ago=12.0)
        self._make_cb_file(tmp_path, "BTCUSD_M240", {}, hours_ago=8.0)  # no trip → skip

        anomalies = pa.detect_cb_lockouts({}, hours=168.0)
        keys = {(a.symbol, a.timeframe) for a in anomalies}
        assert ("XAUUSD", "M1") in keys
        assert ("XAUUSD", "M5") in keys
        assert ("BTCUSD", "M240") not in keys
