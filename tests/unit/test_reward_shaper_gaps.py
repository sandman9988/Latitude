"""Gap tests for src.core.reward_shaper – RewardShaper class.

Targets:
- Lines 272-273: activity bonus path (r_activity > 0) in calculate_total_reward
- Lines 526-527: KeyError fallback for capture_multiplier in calculate_harvester_reward
- Lines 536-537: KeyError fallback for wtl_multiplier in calculate_harvester_reward
- Also target calculate_trigger_reward KeyError for runway_multiplier (lines 456-457)
"""

from unittest.mock import MagicMock

import pytest

from src.core.reward_shaper import RewardShaper
from src.monitoring.activity_monitor import ActivityMonitor
from src.persistence.learned_parameters import LearnedParametersManager


@pytest.fixture
def param_manager(tmp_path):
    """Create a LearnedParametersManager with temp file."""
    pm = LearnedParametersManager(persistence_path=tmp_path / "params.json")
    return pm


@pytest.fixture
def shaper(param_manager):
    """Create RewardShaper with mocked param_manager."""
    monitor = ActivityMonitor()
    return RewardShaper(
        symbol="BTCUSD",
        timeframe="M15",
        broker="default",
        param_manager=param_manager,
        activity_monitor=monitor,
    )


# ---------------------------------------------------------------------------
# Activity bonus path (lines 272-273)
# ---------------------------------------------------------------------------
class TestActivityBonusPath:
    def test_activity_bonus_positive(self, shaper):
        """When activity_monitor.get_exploration_bonus() > 0,
        we hit the activity stats tracking path."""
        # Mock the activity monitor to return a positive bonus
        shaper.activity_monitor.get_exploration_bonus = MagicMock(return_value=0.5)

        result = shaper.calculate_total_reward({
            "exit_pnl": 10.0,
            "mfe": 20.0,
            "mae": 5.0,
            "winner_to_loser": False,
        })

        assert result["activity_bonus"] == pytest.approx(0.5)
        assert shaper.component_stats["activity"]["sum"] == pytest.approx(0.5)
        assert shaper.component_stats["activity"]["count"] == 1

    def test_activity_bonus_zero_skips_stats(self, shaper):
        """When activity bonus is 0, stats are not updated."""
        shaper.activity_monitor.get_exploration_bonus = MagicMock(return_value=0.0)

        result = shaper.calculate_total_reward({
            "exit_pnl": 10.0,
            "mfe": 20.0,
        })

        assert result["activity_bonus"] == pytest.approx(0.0)
        assert shaper.component_stats["activity"]["count"] == 0

    def test_activity_bonus_accumulated(self, shaper):
        """Multiple calls with positive bonus accumulate stats."""
        shaper.activity_monitor.get_exploration_bonus = MagicMock(return_value=0.3)

        for _ in range(3):
            shaper.calculate_total_reward({
                "exit_pnl": 5.0,
                "mfe": 10.0,
            })

        assert shaper.component_stats["activity"]["count"] == 3
        assert shaper.component_stats["activity"]["sum"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# KeyError fallbacks in calculate_harvester_reward (lines 526-527, 536-537)
# ---------------------------------------------------------------------------
class TestHarvesterRewardKeyErrors:
    def test_capture_multiplier_keyerror_fallback(self, shaper):
        """When _get_param("capture_multiplier") raises KeyError,
        should use CAPTURE_MULT_FALLBACK=2.0."""
        # Make _get_param raise KeyError for capture_multiplier
        original_get_param = shaper._get_param

        def mock_get_param(name):
            if name == "capture_multiplier":
                raise KeyError(name)
            return original_get_param(name)

        shaper._get_param = mock_get_param

        result = shaper.calculate_harvester_reward(
            exit_pnl=80.0,
            mfe=100.0,
            mae=10.0,
            was_wtl=False,
            bars_held=10,
            bars_from_mfe_to_exit=2,
        )

        # With fallback capture_mult=2.0: (80/100 - 0.7) * 2.0 = 0.2
        assert result["capture_efficiency"] == pytest.approx(0.2)
        assert "harvester_reward" in result

    def test_wtl_multiplier_keyerror_fallback(self, shaper):
        """When _get_param("wtl_multiplier") raises KeyError,
        should use WTL_MULT_DEFAULT=3.0."""
        original_get_param = shaper._get_param

        def mock_get_param(name):
            if name == "wtl_multiplier":
                raise KeyError(name)
            return original_get_param(name)

        shaper._get_param = mock_get_param

        result = shaper.calculate_harvester_reward(
            exit_pnl=80.0,
            mfe=100.0,
            mae=10.0,
            was_wtl=True,
            bars_held=10,
        )

        # wtl_mult fallback = 3.0, was_wtl = True → r_wtl = -3.0
        assert result["wtl_penalty"] == pytest.approx(-3.0)

    def test_both_fallbacks_together(self, shaper):
        """Both capture_multiplier and wtl_multiplier KeyError at once."""
        def mock_get_param(name):
            raise KeyError(name)

        shaper._get_param = mock_get_param

        result = shaper.calculate_harvester_reward(
            exit_pnl=50.0,
            mfe=100.0,
            mae=20.0,
            was_wtl=True,
            bars_held=20,
            bars_from_mfe_to_exit=5,
        )

        # capture: (50/100 - 0.7) * 2.0 = -0.4
        assert result["capture_efficiency"] == pytest.approx(-0.4)
        # wtl: -3.0
        assert result["wtl_penalty"] == pytest.approx(-3.0)
        # timing: -0.5 * (5/20) = -0.125
        assert result["timing_penalty"] == pytest.approx(-0.125)


# ---------------------------------------------------------------------------
# KeyError fallback in calculate_trigger_reward (runway_multiplier)
# ---------------------------------------------------------------------------
class TestTriggerRewardKeyError:
    def test_runway_multiplier_keyerror_fallback(self, shaper):
        """When _get_param("runway_multiplier") raises KeyError,
        should use RUNWAY_MULT_DEFAULT=2.0."""
        def mock_get_param(name):
            raise KeyError(name)

        shaper._get_param = mock_get_param

        result = shaper.calculate_trigger_reward(
            actual_mfe=100.0,
            predicted_runway=100.0,
            direction=1,
            entry_price=50000.0,
        )

        # utilization = 100/100 = 1.0, log(1.0) = 0.0
        assert result["runway_reward"] == pytest.approx(0.0)
        assert result["utilization"] == pytest.approx(1.0)
        assert result["prediction_quality"] == "EXCELLENT"


# ---------------------------------------------------------------------------
# calculate_harvester_reward quality paths
# ---------------------------------------------------------------------------
class TestHarvesterRewardQuality:
    def test_excellent_capture(self, shaper):
        """capture_ratio >= 0.8 → quality = EXCELLENT."""
        shaper._get_param = MagicMock(return_value=2.0)
        result = shaper.calculate_harvester_reward(
            exit_pnl=90.0, mfe=100.0, mae=5.0,
            was_wtl=False, bars_held=10,
        )
        assert result["quality"] == "EXCELLENT"

    def test_good_capture(self, shaper):
        """0.6 <= capture_ratio < 0.8 → quality = GOOD."""
        shaper._get_param = MagicMock(return_value=2.0)
        result = shaper.calculate_harvester_reward(
            exit_pnl=70.0, mfe=100.0, mae=5.0,
            was_wtl=False, bars_held=10,
        )
        assert result["quality"] == "GOOD"

    def test_fair_capture(self, shaper):
        """0.4 <= capture_ratio < 0.6 → quality = FAIR."""
        shaper._get_param = MagicMock(return_value=2.0)
        result = shaper.calculate_harvester_reward(
            exit_pnl=50.0, mfe=100.0, mae=5.0,
            was_wtl=False, bars_held=10,
        )
        assert result["quality"] == "FAIR"

    def test_poor_capture(self, shaper):
        """capture_ratio < 0.4 → quality = POOR."""
        shaper._get_param = MagicMock(return_value=2.0)
        result = shaper.calculate_harvester_reward(
            exit_pnl=10.0, mfe=100.0, mae=5.0,
            was_wtl=False, bars_held=10,
        )
        assert result["quality"] == "POOR"

    def test_zero_mfe_neutral(self, shaper):
        """mfe=0 → capture_ratio=0, r_capture=0."""
        result = shaper.calculate_harvester_reward(
            exit_pnl=0.0, mfe=0.0, mae=5.0,
            was_wtl=False, bars_held=10,
        )
        assert result["capture_efficiency"] == pytest.approx(0.0)
        assert result["capture_ratio"] == pytest.approx(0.0)

    def test_timing_penalty_applied(self, shaper):
        """bars_from_mfe > 0 → timing penalty calculated."""
        shaper._get_param = MagicMock(return_value=2.0)
        result = shaper.calculate_harvester_reward(
            exit_pnl=80.0, mfe=100.0, mae=5.0,
            was_wtl=False, bars_held=20,
            bars_from_mfe_to_exit=10,
        )
        # timing = -0.5 * (10/20) = -0.25
        assert result["timing_penalty"] == pytest.approx(-0.25)

    def test_timing_zero_when_no_bars(self, shaper):
        """bars_held=0 or bars_from_mfe=0 → no timing penalty."""
        shaper._get_param = MagicMock(return_value=2.0)
        result = shaper.calculate_harvester_reward(
            exit_pnl=80.0, mfe=100.0, mae=5.0,
            was_wtl=False, bars_held=0,
        )
        assert result["timing_penalty"] == pytest.approx(0.0)
