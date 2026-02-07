"""Tests for src.core.reward_shaper – RewardShaper (6-component reward calculation)."""

import pytest

from src.core.reward_shaper import RewardShaper


# ---------------------------------------------------------------------------
# Fixture – fresh shaper with defaults
# ---------------------------------------------------------------------------

@pytest.fixture()
def shaper():
    return RewardShaper(symbol="BTCUSD", timeframe="M15")


# ---------------------------------------------------------------------------
# Capture efficiency
# ---------------------------------------------------------------------------

class TestCaptureEfficiency:
    def test_above_target(self, shaper):
        """80% capture of 100 MFE → positive reward (target=70%)."""
        r = shaper.calculate_capture_efficiency_reward(exit_pnl=80.0, mfe=100.0)
        assert r > 0

    def test_below_target(self, shaper):
        """40% capture → negative reward."""
        r = shaper.calculate_capture_efficiency_reward(exit_pnl=40.0, mfe=100.0)
        assert r < 0

    def test_zero_mfe(self, shaper):
        """No favorable movement → 0 reward."""
        assert shaper.calculate_capture_efficiency_reward(exit_pnl=10.0, mfe=0.0) == pytest.approx(0.0)

    def test_negative_mfe(self, shaper):
        assert shaper.calculate_capture_efficiency_reward(exit_pnl=10.0, mfe=-5.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# WTL penalty
# ---------------------------------------------------------------------------

class TestWTLPenalty:
    def test_not_wtl(self, shaper):
        assert shaper.calculate_wtl_penalty(was_wtl=False, mfe=200.0, exit_pnl=50.0) == pytest.approx(0.0)

    def test_below_threshold(self, shaper):
        """Small MFE → no penalty even if WTL."""
        assert shaper.calculate_wtl_penalty(was_wtl=True, mfe=5.0, exit_pnl=-2.0) == pytest.approx(0.0)

    def test_wtl_penalty_negative(self, shaper):
        p = shaper.calculate_wtl_penalty(was_wtl=True, mfe=200.0, exit_pnl=-30.0, bars_from_mfe_to_exit=10)
        assert p < 0

    def test_longer_hold_bigger_penalty(self, shaper):
        p1 = shaper.calculate_wtl_penalty(was_wtl=True, mfe=200.0, exit_pnl=-30.0, bars_from_mfe_to_exit=5)
        p2 = shaper.calculate_wtl_penalty(was_wtl=True, mfe=200.0, exit_pnl=-30.0, bars_from_mfe_to_exit=20)
        assert p2 < p1  # More negative


# ---------------------------------------------------------------------------
# Opportunity cost
# ---------------------------------------------------------------------------

class TestOpportunityCost:
    def test_below_threshold(self, shaper):
        assert shaper.calculate_opportunity_cost(potential_mfe=10.0, signal_strength=0.8) == pytest.approx(0.0)

    def test_weak_signal(self, shaper):
        assert shaper.calculate_opportunity_cost(potential_mfe=200.0, signal_strength=0.3) == pytest.approx(0.0)

    def test_opportunity_penalty_negative(self, shaper):
        p = shaper.calculate_opportunity_cost(potential_mfe=200.0, signal_strength=0.8)
        assert p < 0


# ---------------------------------------------------------------------------
# Total reward
# ---------------------------------------------------------------------------

class TestTotalReward:
    def test_returns_expected_keys(self, shaper):
        result = shaper.calculate_total_reward({"exit_pnl": 50.0, "mfe": 100.0, "mae": 10.0, "winner_to_loser": False})
        for key in ("capture_efficiency", "wtl_penalty", "opportunity_cost",
                     "activity_bonus", "counterfactual_adjustment", "ensemble_bonus",
                     "total_reward", "components_active"):
            assert key in result

    def test_components_active_count(self, shaper):
        result = shaper.calculate_total_reward({"exit_pnl": 80.0, "mfe": 100.0, "mae": 10.0, "winner_to_loser": False})
        assert result["components_active"] >= 1

    def test_zero_mfe_total(self, shaper):
        result = shaper.calculate_total_reward({"exit_pnl": 0.0, "mfe": 0.0, "mae": 0.0, "winner_to_loser": False})
        # No MFE → capture = 0, WTL = 0
        assert result["capture_efficiency"] == pytest.approx(0.0)
        assert result["wtl_penalty"] == pytest.approx(0.0)

    def test_increments_counter(self, shaper):
        assert shaper.total_rewards_calculated == 0
        shaper.calculate_total_reward({"exit_pnl": 10.0, "mfe": 20.0, "winner_to_loser": False})
        assert shaper.total_rewards_calculated == 1


# ---------------------------------------------------------------------------
# Trigger reward (dual-agent)
# ---------------------------------------------------------------------------

class TestTriggerReward:
    def test_perfect_prediction(self, shaper):
        r = shaper.calculate_trigger_reward(actual_mfe=100.0, predicted_runway=100.0, direction=1, entry_price=50000.0)
        assert r["runway_reward"] == pytest.approx(0.0, abs=0.01)
        assert r["prediction_quality"] == "EXCELLENT"

    def test_exceeded_prediction(self, shaper):
        r = shaper.calculate_trigger_reward(actual_mfe=200.0, predicted_runway=100.0, direction=1, entry_price=50000.0)
        assert r["runway_reward"] > 0
        assert r["utilization"] == pytest.approx(2.0)

    def test_fell_short(self, shaper):
        r = shaper.calculate_trigger_reward(actual_mfe=30.0, predicted_runway=100.0, direction=1, entry_price=50000.0)
        assert r["runway_reward"] < 0
        assert r["prediction_quality"] == "OVERPREDICTED"

    def test_zero_prediction_invalid(self, shaper):
        r = shaper.calculate_trigger_reward(actual_mfe=50.0, predicted_runway=0.0, direction=1, entry_price=50000.0)
        assert r["prediction_quality"] == "INVALID"
        assert r["runway_reward"] < 0

    def test_zero_actual_mfe(self, shaper):
        r = shaper.calculate_trigger_reward(actual_mfe=0.0, predicted_runway=100.0, direction=1, entry_price=50000.0)
        assert r["runway_reward"] < 0  # log penalty


# ---------------------------------------------------------------------------
# Harvester reward (dual-agent)
# ---------------------------------------------------------------------------

class TestHarvesterReward:
    def test_excellent_capture(self, shaper):
        r = shaper.calculate_harvester_reward(
            exit_pnl=85.0, mfe=100.0, mae=5.0, was_wtl=False, bars_held=20, bars_from_mfe_to_exit=2,
        )
        assert r["quality"] == "EXCELLENT"
        assert r["harvester_reward"] > 0

    def test_wtl_penalty(self, shaper):
        r = shaper.calculate_harvester_reward(
            exit_pnl=-10.0, mfe=100.0, mae=50.0, was_wtl=True, bars_held=30, bars_from_mfe_to_exit=25,
        )
        assert r["wtl_penalty"] < 0
        assert r["was_wtl"] is True

    def test_timing_penalty(self, shaper):
        r = shaper.calculate_harvester_reward(
            exit_pnl=50.0, mfe=100.0, mae=10.0, was_wtl=False, bars_held=20, bars_from_mfe_to_exit=15,
        )
        assert r["timing_penalty"] < 0

    def test_zero_mfe_neutral(self, shaper):
        r = shaper.calculate_harvester_reward(
            exit_pnl=0.0, mfe=0.0, mae=0.0, was_wtl=False, bars_held=10,
        )
        assert r["capture_efficiency"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dual-agent combined
# ---------------------------------------------------------------------------

class TestDualAgentRewards:
    def test_combined_reward(self, shaper):
        r = shaper.calculate_dual_agent_rewards(
            actual_mfe=100.0, predicted_runway=100.0, direction=1, entry_price=50000.0,
            exit_pnl=80.0, mae=10.0, was_wtl=False, bars_held=20, bars_from_mfe_to_exit=3,
        )
        assert "total_reward" in r
        assert "trigger_reward" in r
        assert "harvester_reward" in r
        assert "trigger_breakdown" in r
        assert "harvester_breakdown" in r

    def test_weight_split(self, shaper):
        """Total = 0.4*trigger + 0.6*harvester."""
        r = shaper.calculate_dual_agent_rewards(
            actual_mfe=100.0, predicted_runway=100.0, direction=1, entry_price=50000.0,
            exit_pnl=80.0, mae=10.0, was_wtl=False, bars_held=20,
        )
        expected = 0.4 * r["trigger_reward"] + 0.6 * r["harvester_reward"]
        assert r["total_reward"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Statistics / summary helpers
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_get_statistics(self, shaper):
        shaper.calculate_total_reward({"exit_pnl": 80.0, "mfe": 100.0, "winner_to_loser": False})
        stats = shaper.get_statistics()
        assert stats["total_rewards_calculated"] == 1
        assert "parameters" in stats

    def test_adapt_weights_noop(self, shaper):
        # Should not raise – weights are now fixed
        shaper.adapt_weights(0.1)

    def test_print_summary_returns_str(self, shaper):
        shaper.calculate_total_reward({"exit_pnl": 80.0, "mfe": 100.0, "winner_to_loser": False})
        summary = shaper.print_summary()
        assert isinstance(summary, str)
        assert "REWARD SHAPER SUMMARY" in summary
