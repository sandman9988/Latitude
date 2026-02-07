"""Tests for src.core.cold_start_manager."""

import json
import pytest
import numpy as np

from src.core.cold_start_manager import ColdStartManager, PhaseMetrics, WarmupPhase


# ---------------------------------------------------------------------------
# WarmupPhase enum
# ---------------------------------------------------------------------------
class TestWarmupPhase:
    def test_phase_values(self):
        assert WarmupPhase.OBSERVATION.value == 1
        assert WarmupPhase.PAPER_TRADING.value == 2
        assert WarmupPhase.MICRO_POSITIONS.value == 3
        assert WarmupPhase.PRODUCTION.value == 4

    def test_phase_names(self):
        assert WarmupPhase["OBSERVATION"] is WarmupPhase.OBSERVATION
        assert WarmupPhase["PRODUCTION"] is WarmupPhase.PRODUCTION


# ---------------------------------------------------------------------------
# PhaseMetrics dataclass
# ---------------------------------------------------------------------------
class TestPhaseMetrics:
    def test_fields(self):
        m = PhaseMetrics(
            phase=WarmupPhase.OBSERVATION,
            bars_completed=100,
            trades_completed=0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            max_drawdown=0.0,
            avg_trade_profit=0.0,
            total_pnl=0.0,
        )
        assert m.phase == WarmupPhase.OBSERVATION
        assert m.bars_completed == 100


# ---------------------------------------------------------------------------
# ColdStartManager: init & basic state
# ---------------------------------------------------------------------------
class TestColdStartManagerInit:
    def test_default_phase(self):
        mgr = ColdStartManager()
        assert mgr.current_phase == WarmupPhase.OBSERVATION

    def test_custom_thresholds(self):
        mgr = ColdStartManager(observation_min_bars=50, paper_min_sharpe=0.5)
        assert mgr.observation_min_bars == 50
        assert mgr.paper_min_sharpe == pytest.approx(0.5)

    def test_initial_counters_zero(self):
        mgr = ColdStartManager()
        assert mgr.bars_in_current_phase == 0
        assert mgr.trades_in_current_phase == 0
        assert mgr.current_pnls == []
        assert mgr.current_trades == []
        assert mgr.phase_history == []


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------
class TestUpdate:
    def test_new_bar_increments_counter(self):
        mgr = ColdStartManager()
        mgr.update(new_bar=True)
        assert mgr.bars_in_current_phase == 1

    def test_trade_completed_increments_counter(self):
        mgr = ColdStartManager()
        mgr.update(trade_completed={"pnl": 10.0, "is_paper": True})
        assert mgr.trades_in_current_phase == 1
        assert mgr.current_pnls == [10.0]

    def test_trade_missing_pnl_defaults_zero(self):
        mgr = ColdStartManager()
        mgr.update(trade_completed={"is_paper": True})
        assert mgr.current_pnls == [0.0]

    def test_both_bar_and_trade(self):
        mgr = ColdStartManager()
        mgr.update(new_bar=True, trade_completed={"pnl": 5.0})
        assert mgr.bars_in_current_phase == 1
        assert mgr.trades_in_current_phase == 1


# ---------------------------------------------------------------------------
# can_trade() / is_paper_only() / get_position_size_multiplier()
# ---------------------------------------------------------------------------
class TestPhaseProperties:
    @pytest.mark.parametrize(
        "phase, expected",
        [
            (WarmupPhase.OBSERVATION, False),
            (WarmupPhase.PAPER_TRADING, True),
            (WarmupPhase.MICRO_POSITIONS, True),
            (WarmupPhase.PRODUCTION, True),
        ],
    )
    def test_can_trade(self, phase, expected):
        mgr = ColdStartManager()
        mgr.current_phase = phase
        assert mgr.can_trade() is expected

    @pytest.mark.parametrize(
        "phase, expected",
        [
            (WarmupPhase.OBSERVATION, False),
            (WarmupPhase.PAPER_TRADING, True),
            (WarmupPhase.MICRO_POSITIONS, False),
            (WarmupPhase.PRODUCTION, False),
        ],
    )
    def test_is_paper_only(self, phase, expected):
        mgr = ColdStartManager()
        mgr.current_phase = phase
        assert mgr.is_paper_only() is expected

    @pytest.mark.parametrize(
        "phase, expected_mult",
        [
            (WarmupPhase.OBSERVATION, 0.0),
            (WarmupPhase.PAPER_TRADING, 0.0),
            (WarmupPhase.MICRO_POSITIONS, 0.001),
            (WarmupPhase.PRODUCTION, 1.0),
        ],
    )
    def test_position_size_multiplier(self, phase, expected_mult):
        mgr = ColdStartManager()
        mgr.current_phase = phase
        assert mgr.get_position_size_multiplier() == expected_mult

    def test_micro_custom_size(self):
        mgr = ColdStartManager(micro_position_size=0.01)
        mgr.current_phase = WarmupPhase.MICRO_POSITIONS
        assert mgr.get_position_size_multiplier() == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# check_graduation(): Observation → Paper
# ---------------------------------------------------------------------------
class TestObservationGraduation:
    def test_graduates_after_min_bars(self):
        mgr = ColdStartManager(observation_min_bars=10)
        for _ in range(10):
            mgr.update(new_bar=True)
        assert mgr.check_graduation() == WarmupPhase.PAPER_TRADING

    def test_no_graduation_before_min_bars(self):
        mgr = ColdStartManager(observation_min_bars=10)
        for _ in range(9):
            mgr.update(new_bar=True)
        assert mgr.check_graduation() is None


# ---------------------------------------------------------------------------
# check_graduation(): Paper → Micro
# ---------------------------------------------------------------------------
class TestPaperGraduation:
    def _make_paper_mgr(self, **kwargs):
        mgr = ColdStartManager(paper_min_bars=20, paper_min_sharpe=0.3,
                                paper_min_win_rate=0.45, paper_max_drawdown=0.20,
                                **kwargs)
        mgr.current_phase = WarmupPhase.PAPER_TRADING
        return mgr

    def test_not_enough_bars(self):
        mgr = self._make_paper_mgr()
        for _ in range(10):
            mgr.update(new_bar=True)
        assert mgr.check_graduation() is None

    def test_not_enough_trades(self):
        mgr = self._make_paper_mgr()
        for _ in range(25):
            mgr.update(new_bar=True)
        # Only 5 trades (< 10 required)
        for _ in range(5):
            mgr.update(trade_completed={"pnl": 5.0})
        assert mgr.check_graduation() is None

    def test_graduation_with_good_performance(self):
        mgr = self._make_paper_mgr()
        for _ in range(25):
            mgr.update(new_bar=True)
        # 15 winning trades → high win rate, positive Sharpe
        rng = np.random.default_rng(42)
        for _ in range(15):
            pnl = abs(rng.standard_normal()) + 1.0  # All positive
            mgr.update(trade_completed={"pnl": pnl})
        result = mgr.check_graduation()
        assert result == WarmupPhase.MICRO_POSITIONS

    def test_no_graduation_with_bad_performance(self):
        mgr = self._make_paper_mgr()
        for _ in range(25):
            mgr.update(new_bar=True)
        # Mostly losing trades
        for _ in range(15):
            mgr.update(trade_completed={"pnl": -5.0})
        assert mgr.check_graduation() is None


# ---------------------------------------------------------------------------
# check_graduation(): Micro → Production
# ---------------------------------------------------------------------------
class TestMicroGraduation:
    def _make_micro_mgr(self, **kwargs):
        mgr = ColdStartManager(micro_min_bars=20, micro_min_sharpe=0.3,
                                micro_min_win_rate=0.45, micro_min_avg_profit=0.0,
                                **kwargs)
        mgr.current_phase = WarmupPhase.MICRO_POSITIONS
        return mgr

    def test_not_enough_bars(self):
        mgr = self._make_micro_mgr()
        for _ in range(10):
            mgr.update(new_bar=True)
        assert mgr.check_graduation() is None

    def test_not_enough_trades(self):
        mgr = self._make_micro_mgr()
        for _ in range(25):
            mgr.update(new_bar=True)
        for _ in range(15):
            mgr.update(trade_completed={"pnl": 5.0})
        assert mgr.check_graduation() is None  # needs 20

    def test_graduation_with_good_performance(self):
        mgr = self._make_micro_mgr()
        for _ in range(25):
            mgr.update(new_bar=True)
        rng = np.random.default_rng(42)
        for _ in range(25):
            pnl = abs(rng.standard_normal()) + 1.0
            mgr.update(trade_completed={"pnl": pnl})
        assert mgr.check_graduation() == WarmupPhase.PRODUCTION

    def test_no_graduation_bad_performance(self):
        mgr = self._make_micro_mgr()
        for _ in range(25):
            mgr.update(new_bar=True)
        for _ in range(25):
            mgr.update(trade_completed={"pnl": -5.0})
        assert mgr.check_graduation() is None


# ---------------------------------------------------------------------------
# check_graduation(): Production → demotion
# ---------------------------------------------------------------------------
class TestProductionDemotion:
    def test_no_demotion_insufficient_trades(self):
        mgr = ColdStartManager()
        mgr.current_phase = WarmupPhase.PRODUCTION
        for _ in range(30):
            mgr.update(trade_completed={"pnl": -5.0})
        assert mgr.check_graduation() is None  # needs 50

    def test_demotion_on_bad_performance(self):
        mgr = ColdStartManager(
            production_demotion_sharpe=0.2,
            production_demotion_win_rate=0.40,
            production_demotion_drawdown=0.30,
        )
        mgr.current_phase = WarmupPhase.PRODUCTION
        # Many losing trades → low Sharpe, low win rate
        for _ in range(60):
            mgr.update(new_bar=True)
            mgr.update(trade_completed={"pnl": -10.0})
        result = mgr.check_graduation()
        assert result == WarmupPhase.MICRO_POSITIONS

    def test_no_demotion_with_good_performance(self):
        mgr = ColdStartManager(
            production_demotion_sharpe=0.2,
            production_demotion_win_rate=0.40,
        )
        mgr.current_phase = WarmupPhase.PRODUCTION
        rng = np.random.default_rng(42)
        for _ in range(60):
            mgr.update(new_bar=True)
            pnl = abs(rng.standard_normal()) + 1.0
            mgr.update(trade_completed={"pnl": pnl})
        assert mgr.check_graduation() is None


# ---------------------------------------------------------------------------
# _calculate_current_metrics()
# ---------------------------------------------------------------------------
class TestCalculateMetrics:
    def test_empty_pnls(self):
        mgr = ColdStartManager()
        m = mgr._calculate_current_metrics()
        assert m.sharpe_ratio == pytest.approx(0.0)
        assert m.win_rate == pytest.approx(0.0)
        assert m.total_pnl == pytest.approx(0.0)

    def test_all_wins(self):
        mgr = ColdStartManager()
        mgr.current_pnls = [5.0, 10.0, 3.0]
        mgr.bars_in_current_phase = 10
        mgr.trades_in_current_phase = 3
        m = mgr._calculate_current_metrics()
        assert m.win_rate == pytest.approx(1.0)
        assert m.total_pnl == pytest.approx(18.0)
        assert m.avg_trade_profit == pytest.approx(6.0)
        assert m.sharpe_ratio > 0

    def test_all_losses(self):
        mgr = ColdStartManager()
        mgr.current_pnls = [-5.0, -10.0, -3.0]
        m = mgr._calculate_current_metrics()
        assert m.win_rate == pytest.approx(0.0)
        assert m.total_pnl == pytest.approx(-18.0)
        assert m.sharpe_ratio < 0

    def test_mixed_pnls(self):
        mgr = ColdStartManager()
        mgr.current_pnls = [5.0, -3.0, 7.0, -1.0]
        m = mgr._calculate_current_metrics()
        assert m.win_rate == pytest.approx(0.5)
        assert m.total_pnl == pytest.approx(8.0)

    def test_max_drawdown_positive(self):
        mgr = ColdStartManager()
        mgr.current_pnls = [10.0, -5.0, -5.0, 3.0]
        m = mgr._calculate_current_metrics()
        assert m.max_drawdown >= 0


# ---------------------------------------------------------------------------
# graduate()
# ---------------------------------------------------------------------------
class TestGraduate:
    def test_graduate_transitions_phase(self):
        mgr = ColdStartManager()
        mgr.bars_in_current_phase = 100
        mgr.current_pnls = [1.0, 2.0]
        mgr.graduate(WarmupPhase.PAPER_TRADING)
        assert mgr.current_phase == WarmupPhase.PAPER_TRADING

    def test_graduate_resets_counters(self):
        mgr = ColdStartManager()
        mgr.bars_in_current_phase = 100
        mgr.trades_in_current_phase = 5
        mgr.current_pnls = [1.0]
        mgr.current_trades = [{"pnl": 1.0}]
        mgr.graduate(WarmupPhase.PAPER_TRADING)
        assert mgr.bars_in_current_phase == 0
        assert mgr.trades_in_current_phase == 0
        assert mgr.current_pnls == []
        assert mgr.current_trades == []

    def test_graduate_saves_history(self):
        mgr = ColdStartManager()
        mgr.current_pnls = [5.0, -2.0]
        mgr.graduate(WarmupPhase.PAPER_TRADING)
        assert len(mgr.phase_history) == 1
        assert mgr.phase_history[0].phase == WarmupPhase.OBSERVATION


# ---------------------------------------------------------------------------
# get_status()
# ---------------------------------------------------------------------------
class TestGetStatus:
    def test_status_keys(self):
        mgr = ColdStartManager()
        s = mgr.get_status()
        assert "current_phase" in s
        assert "can_trade" in s
        assert "is_paper_only" in s
        assert "position_size_multiplier" in s
        assert "current_metrics" in s

    def test_status_reflects_phase(self):
        mgr = ColdStartManager()
        mgr.current_phase = WarmupPhase.PRODUCTION
        s = mgr.get_status()
        assert s["current_phase"] == "PRODUCTION"
        assert s["can_trade"] is True
        assert s["position_size_multiplier"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# save_state() / load_state()
# ---------------------------------------------------------------------------
class TestPersistence:
    def test_save_and_load(self, tmp_path):
        sf = tmp_path / "cold_start.json"
        mgr = ColdStartManager(state_file=sf)
        mgr.current_phase = WarmupPhase.MICRO_POSITIONS
        mgr.bars_in_current_phase = 42
        mgr.trades_in_current_phase = 7
        mgr.current_pnls = [1.0, -0.5]
        mgr.current_trades = [{"pnl": 1.0}, {"pnl": -0.5}]
        mgr.save_state()

        mgr2 = ColdStartManager(state_file=sf)
        assert mgr2.load_state() is True
        assert mgr2.current_phase == WarmupPhase.MICRO_POSITIONS
        assert mgr2.bars_in_current_phase == 42
        assert mgr2.trades_in_current_phase == 7
        assert mgr2.current_pnls == [1.0, -0.5]

    def test_load_missing_file_returns_false(self, tmp_path):
        sf = tmp_path / "missing.json"
        mgr = ColdStartManager(state_file=sf)
        assert mgr.load_state() is False

    def test_load_corrupt_file_returns_false(self, tmp_path):
        sf = tmp_path / "corrupt.json"
        sf.write_text("not json!!!")
        mgr = ColdStartManager(state_file=sf)
        assert mgr.load_state() is False

    def test_save_includes_phase_history(self, tmp_path):
        sf = tmp_path / "state.json"
        mgr = ColdStartManager(state_file=sf)
        mgr.current_pnls = [5.0]
        mgr.graduate(WarmupPhase.PAPER_TRADING)
        mgr.save_state()

        data = json.loads(sf.read_text())
        assert len(data["phase_history"]) == 1
        assert data["phase_history"][0]["phase"] == "OBSERVATION"

    def test_load_restores_phase_history(self, tmp_path):
        sf = tmp_path / "state.json"
        mgr = ColdStartManager(state_file=sf)
        mgr.current_pnls = [5.0]
        mgr.graduate(WarmupPhase.PAPER_TRADING)
        mgr.save_state()

        mgr2 = ColdStartManager(state_file=sf)
        mgr2.load_state()
        assert len(mgr2.phase_history) == 1
        assert mgr2.phase_history[0].phase == WarmupPhase.OBSERVATION

    def test_save_creates_parent_dirs(self, tmp_path):
        sf = tmp_path / "sub" / "dir" / "state.json"
        mgr = ColdStartManager(state_file=sf)
        mgr.save_state()
        assert sf.exists()

    def test_save_truncates_long_history(self, tmp_path):
        sf = tmp_path / "state.json"
        mgr = ColdStartManager(state_file=sf)
        mgr.current_pnls = list(range(200))
        mgr.current_trades = [{"pnl": i} for i in range(200)]
        mgr.save_state()
        data = json.loads(sf.read_text())
        assert len(data["current_pnls"]) == 100
        assert len(data["current_trades"]) == 100


# ---------------------------------------------------------------------------
# Full graduation flow integration
# ---------------------------------------------------------------------------
class TestFullGraduationFlow:
    def test_observation_to_paper(self):
        mgr = ColdStartManager(observation_min_bars=5)
        for _ in range(5):
            mgr.update(new_bar=True)
        nxt = mgr.check_graduation()
        assert nxt == WarmupPhase.PAPER_TRADING
        mgr.graduate(nxt)
        assert mgr.current_phase == WarmupPhase.PAPER_TRADING
        assert mgr.bars_in_current_phase == 0
