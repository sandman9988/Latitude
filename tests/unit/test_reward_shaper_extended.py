"""Extended tests for src.core.reward_shaper.

Covers: calculate_total_reward with counterfactual/ensemble bonus,
calculate_harvester_reward WTL + timing, calculate_trigger_reward
zero actual_mfe/underpredicted/overpredicted, calculate_dual_agent_rewards,
adapt_weights noop, print_summary, get_statistics averages.
"""

import pytest

from src.core.reward_shaper import RewardShaper


@pytest.fixture()
def shaper():
    return RewardShaper(symbol="BTCUSD", timeframe="M5", broker="test")


# ---------------------------------------------------------------------------
# Counterfactual & ensemble in total reward
# ---------------------------------------------------------------------------
class TestTotalRewardExtended:
    def test_counterfactual_entry_triggers_component(self, shaper):
        """When entry/exit/mfe are set, counterfactual component fires."""
        result = shaper.calculate_total_reward({
            "exit_pnl": 50.0,
            "mfe": 100.0,
            "mae": 10.0,
            "winner_to_loser": False,
            "entry_price": 95000.0,
            "exit_price": 95050.0,
            "direction": 1,
            "mfe_bar_offset": 5,
        })
        # Counterfactual analyzer runs, may produce non-zero adjustment
        assert "counterfactual_adjustment" in result

    def test_ensemble_bonus_included(self, shaper):
        result = shaper.calculate_total_reward({
            "exit_pnl": 50.0,
            "mfe": 100.0,
            "mae": 10.0,
            "winner_to_loser": False,
            "ensemble_bonus": 0.3,
        })
        assert result["ensemble_bonus"] == pytest.approx(0.3)

    def test_all_components_zero_when_no_data(self, shaper):
        result = shaper.calculate_total_reward({
            "exit_pnl": 0.0,
            "mfe": 0.0,
            "mae": 0.0,
            "winner_to_loser": False,
        })
        assert result["capture_efficiency"] == pytest.approx(0.0)
        assert result["wtl_penalty"] == pytest.approx(0.0)

    def test_components_active_counts_nonzero(self, shaper):
        result = shaper.calculate_total_reward({
            "exit_pnl": 80.0,
            "mfe": 100.0,
            "mae": 20.0,
            "winner_to_loser": False,
        })
        assert result["components_active"] >= 1


# ---------------------------------------------------------------------------
# Harvester reward – capture quality labels
# ---------------------------------------------------------------------------
class TestHarvesterQuality:
    def test_excellent_quality(self, shaper):
        r = shaper.calculate_harvester_reward(
            exit_pnl=0.0035, mfe=0.004, mae=0.001,
            was_wtl=False, bars_held=10,
        )
        assert r["quality"] == "EXCELLENT"
        assert r["capture_ratio"] >= 0.8

    def test_good_quality(self, shaper):
        r = shaper.calculate_harvester_reward(
            exit_pnl=0.0028, mfe=0.004, mae=0.001,
            was_wtl=False, bars_held=10,
        )
        assert r["quality"] == "GOOD"

    def test_fair_quality(self, shaper):
        r = shaper.calculate_harvester_reward(
            exit_pnl=0.002, mfe=0.004, mae=0.001,
            was_wtl=False, bars_held=10,
        )
        assert r["quality"] == "FAIR"

    def test_poor_quality(self, shaper):
        r = shaper.calculate_harvester_reward(
            exit_pnl=0.001, mfe=0.004, mae=0.002,
            was_wtl=False, bars_held=10,
        )
        assert r["quality"] == "POOR"

    def test_wtl_produces_penalty(self, shaper):
        r = shaper.calculate_harvester_reward(
            exit_pnl=-0.001, mfe=0.004, mae=0.005,
            was_wtl=True, bars_held=20, bars_from_mfe_to_exit=15,
        )
        assert r["wtl_penalty"] < 0

    def test_timing_penalty(self, shaper):
        r = shaper.calculate_harvester_reward(
            exit_pnl=0.003, mfe=0.004, mae=0.001,
            was_wtl=False, bars_held=20, bars_from_mfe_to_exit=10,
        )
        assert r["timing_penalty"] < 0

    def test_zero_bars_held_no_timing_penalty(self, shaper):
        r = shaper.calculate_harvester_reward(
            exit_pnl=0.003, mfe=0.004, mae=0.001,
            was_wtl=False, bars_held=0, bars_from_mfe_to_exit=0,
        )
        assert r["timing_penalty"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Trigger reward – quality labels
# ---------------------------------------------------------------------------
class TestTriggerQuality:
    def test_excellent_prediction(self, shaper):
        r = shaper.calculate_trigger_reward(0.0025, 0.0025, 1, 95000.0)
        assert r["prediction_quality"] == "EXCELLENT"

    def test_underpredicted(self, shaper):
        r = shaper.calculate_trigger_reward(0.010, 0.002, 1, 95000.0)
        assert r["prediction_quality"] == "UNDERPREDICTED"

    def test_overpredicted(self, shaper):
        r = shaper.calculate_trigger_reward(0.0005, 0.005, 1, 95000.0)
        assert r["prediction_quality"] == "OVERPREDICTED"

    def test_good_prediction(self, shaper):
        # utilization ~0.6 → GOOD
        r = shaper.calculate_trigger_reward(0.003, 0.005, 1, 95000.0)
        assert r["prediction_quality"] == "GOOD"

    def test_zero_actual_mfe_severe_penalty(self, shaper):
        r = shaper.calculate_trigger_reward(0.0, 0.003, 1, 95000.0)
        assert r["runway_reward"] <= -3.0

    def test_reward_clamped(self, shaper):
        # Very extreme ratio to test clamping
        r = shaper.calculate_trigger_reward(100.0, 0.001, 1, 95000.0)
        assert r["runway_reward"] <= 3.0


# ---------------------------------------------------------------------------
# Dual-agent combined reward
# ---------------------------------------------------------------------------
class TestDualAgentRewards:
    def test_combined_reward_keys(self, shaper):
        r = shaper.calculate_dual_agent_rewards(
            actual_mfe=0.003, predicted_runway=0.0025,
            direction=1, entry_price=95000.0,
            exit_pnl=0.002, mae=0.001,
            was_wtl=False, bars_held=15,
        )
        for key in ("total_reward", "trigger_reward", "harvester_reward",
                     "trigger_breakdown", "harvester_breakdown"):
            assert key in r

    def test_weight_split_40_60(self, shaper):
        r = shaper.calculate_dual_agent_rewards(
            actual_mfe=0.003, predicted_runway=0.003,
            direction=1, entry_price=95000.0,
            exit_pnl=0.003, mae=0.001,
            was_wtl=False, bars_held=10,
        )
        expected = 0.4 * r["trigger_reward"] + 0.6 * r["harvester_reward"]
        assert r["total_reward"] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Statistics & summary after multiple rewards
# ---------------------------------------------------------------------------
class TestStatisticsExtended:
    def test_averages_after_multiple_rewards(self, shaper):
        for i in range(5):
            shaper.calculate_total_reward({
                "exit_pnl": 80.0 + i * 10,
                "mfe": 100.0,
                "mae": 20.0,
                "winner_to_loser": False,
            })
        stats = shaper.get_statistics()
        assert stats["total_rewards_calculated"] == 5
        assert stats["avg_capture_reward"] != 0.0

    def test_print_summary_contains_symbol(self, shaper):
        summary = shaper.print_summary()
        assert "BTCUSD" in summary
        assert "REWARD SHAPER" in summary
