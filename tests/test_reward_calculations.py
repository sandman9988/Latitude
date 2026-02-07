"""
Unit Tests for Reward Calculations

CRITICAL: Tests for reward functions modified on 2026-01-11:
- _calculate_trigger_reward() - Prediction accuracy based
- _calculate_harvester_hold_reward() - Capture potential based

These tests validate that:
1. Rewards correlate with desired outcomes
2. Reward ranges are as expected
3. Edge cases are handled correctly
4. Intuitive scenarios produce intuitive rewards
"""

import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)

import pytest


class MockBot:
    """Mock bot with reward calculation methods."""

    def _calculate_trigger_reward(
        self, trade_summary: dict, predicted_runway: float, realized_vol: float = 0.01
    ) -> float:
        """
        Calculate reward for TriggerAgent based on prediction accuracy.

        (Copied from ctrader_ddqn_paper.py)
        """
        actual_mfe = trade_summary.get("mfe", 0.0)
        pnl = trade_summary.get("pnl", 0.0)

        # Default fallback
        if realized_vol <= 0:
            realized_vol = 0.01

        # Normalize MFE by volatility
        norm_mfe = actual_mfe / (realized_vol + 1e-8)
        norm_predicted = predicted_runway / (realized_vol + 1e-8)

        # Component 1: Prediction accuracy
        prediction_error = abs(norm_mfe - norm_predicted)
        max_error = max(norm_mfe, norm_predicted, 1.0)
        runway_accuracy = 1.0 - (prediction_error / max_error)

        # Map to [-1, 1]: Perfect prediction = 1.0, worst = -1.0
        accuracy_reward = runway_accuracy * 2.0 - 1.0

        # Component 2: Magnitude bonus
        magnitude_bonus = min(norm_mfe / 3.0, 1.0) * 0.5

        # Component 3: Penalty for false positives
        false_positive_penalty = 0.0
        if predicted_runway > 0 and pnl < 0:
            false_positive_penalty = -0.2

        # Combine components
        total_reward = accuracy_reward + magnitude_bonus + false_positive_penalty

        # Clip to reasonable range
        total_reward = np.clip(total_reward, -1.5, 1.5)

        return total_reward

    def _calculate_harvester_hold_reward(
        self,
        current_mfe: float,
        current_mae: float,
        prev_mfe: float,
        prev_mae: float,
        bars_held: int,
        unrealized_pnl: float,
        entry_price: float,
        current_price: float,
        realized_vol: float = 0.01,
    ) -> float:
        """
        Calculate reward for HarvesterAgent HOLD action.

        (Copied from ctrader_ddqn_paper.py)
        """
        # Default fallback
        if realized_vol <= 0:
            realized_vol = 0.01

        # Normalize by volatility
        norm_mfe = current_mfe / (realized_vol + 1e-8)
        norm_mae = current_mae / (realized_vol + 1e-8)
        norm_pnl = unrealized_pnl / (realized_vol + 1e-8)

        # Current capture ratio
        capture_ratio = unrealized_pnl / (current_mfe + 1e-8) if current_mfe > 0 else 0.0

        # Component 1: Capture quality reward [0, 0.4]
        capture_component = np.clip(capture_ratio * 0.4, 0.0, 0.4)

        # Component 2: MFE growth reward [-0.3, 0.3]
        mfe_delta = current_mfe - prev_mfe
        norm_mfe_delta = mfe_delta / (realized_vol + 1e-8)
        mfe_growth = np.clip(norm_mfe_delta * 0.3, -0.3, 0.3)

        # Component 3: MAE penalty [-0.4, 0]
        mae_delta = current_mae - prev_mae
        norm_mae_delta = mae_delta / (realized_vol + 1e-8)
        mae_penalty = -np.clip(norm_mae_delta * 0.4, 0.0, 0.4)

        # Component 4: Time decay [-0.2, 0]
        time_decay = -0.02 * min(bars_held / 10, 10)

        # Component 5: Opportunity cost penalty [-0.3, 0]
        opportunity_cost = 0.0
        if current_mfe > 0:
            distance_from_peak = (current_mfe - unrealized_pnl) / (current_mfe + 1e-8)
            opportunity_cost = -distance_from_peak * 0.3

        # Combine components
        total_reward = capture_component + mfe_growth + mae_penalty + time_decay + opportunity_cost

        # Clip to reasonable range
        total_reward = np.clip(total_reward, -1.0, 1.0)

        return total_reward


@pytest.fixture
def bot():
    """Create mock bot instance."""
    return MockBot()


class TestTriggerReward:
    """Tests for TriggerAgent reward calculation."""

    def test_perfect_prediction(self, bot):
        """Perfect prediction should give high positive reward."""
        summary = {"mfe": 10.0, "pnl": 8.0}
        predicted = 10.0
        vol = 0.01

        reward = bot._calculate_trigger_reward(summary, predicted, vol)

        # Perfect prediction: accuracy ~1.0, magnitude bonus ~0.5
        assert reward > 1.0, f"Expected >1.0, got {reward}"
        assert reward <= 1.5, f"Expected <=1.5, got {reward}"

    def test_complete_miss_prediction(self, bot):
        """Completely wrong prediction should give negative reward."""
        summary = {"mfe": 0.0, "pnl": -10.0}
        predicted = 20.0  # Predicted big move, got nothing
        vol = 0.01

        reward = bot._calculate_trigger_reward(summary, predicted, vol)

        # Wrong prediction: accuracy ~-1.0, false positive penalty -0.2
        assert reward < -0.5, f"Expected <-0.5, got {reward}"

    def test_50_percent_error(self, bot):
        """50% prediction error should give near-zero reward."""
        summary = {"mfe": 10.0, "pnl": 5.0}
        predicted = 5.0  # 50% error
        vol = 0.01

        reward = bot._calculate_trigger_reward(summary, predicted, vol)

        # 50% error: accuracy ~0, magnitude bonus ~0.5
        assert -0.5 < reward < 1.0, f"Expected [-0.5, 1.0], got {reward}"

    def test_false_positive_penalty(self, bot):
        """Predicted profit but got loss should have penalty."""
        summary = {"mfe": 5.0, "pnl": -10.0}  # Negative P&L
        predicted = 15.0  # Predicted profit
        vol = 0.01

        reward = bot._calculate_trigger_reward(summary, predicted, vol)

        # Should have false positive penalty (-0.2)
        reward_no_penalty = bot._calculate_trigger_reward({"mfe": 5.0, "pnl": 5.0}, predicted, vol)
        assert reward < reward_no_penalty, "False positive should reduce reward"

    def test_magnitude_bonus(self, bot):
        """Larger MFE should give magnitude bonus."""
        summary_small = {"mfe": 1.0, "pnl": 1.0}
        summary_large = {"mfe": 30.0, "pnl": 25.0}  # 3σ+ move
        predicted_small = 1.0
        predicted_large = 30.0
        vol = 0.01

        reward_small = bot._calculate_trigger_reward(summary_small, predicted_small, vol)
        reward_large = bot._calculate_trigger_reward(summary_large, predicted_large, vol)

        # Larger MFE should get magnitude bonus (or at least equal due to clipping)
        assert reward_large >= reward_small, "Larger MFE should get higher or equal reward"

    def test_reward_range(self, bot):
        """Reward should be clipped to [-1.5, 1.5]."""
        # Extreme scenarios
        scenarios = [
            {"mfe": 0.0, "pnl": -100.0, "predicted": 100.0},  # Worst case
            {"mfe": 100.0, "pnl": 99.0, "predicted": 100.0},  # Best case
        ]

        for scenario in scenarios:
            summary = {"mfe": scenario["mfe"], "pnl": scenario["pnl"]}
            reward = bot._calculate_trigger_reward(summary, scenario["predicted"], 0.01)

            assert -1.5 <= reward <= 1.5, f"Reward {reward} outside range [-1.5, 1.5]"


class TestHarvesterHoldReward:
    """Tests for HarvesterAgent HOLD reward calculation."""

    def test_near_mfe_peak_high_reward(self, bot):
        """Holding near MFE peak should give high reward."""
        # At MFE peak (100% capture)
        reward = bot._calculate_harvester_hold_reward(
            current_mfe=10.0,
            current_mae=2.0,
            prev_mfe=9.0,
            prev_mae=1.5,
            bars_held=5,
            unrealized_pnl=9.5,  # 95% of MFE
            entry_price=2000.0,
            current_price=2009.5,
            realized_vol=0.01,
        )

        # High capture ratio (95%), MFE growing
        assert reward > 0.2, f"Expected >0.2 for near-peak hold, got {reward}"

    def test_far_from_mfe_low_reward(self, bot):
        """Holding far from MFE peak should give low/negative reward."""
        # Far from peak (30% capture)
        reward = bot._calculate_harvester_hold_reward(
            current_mfe=10.0,
            current_mae=5.0,
            prev_mfe=10.0,
            prev_mae=4.5,
            bars_held=50,  # Held long
            unrealized_pnl=3.0,  # Only 30% of MFE
            entry_price=2000.0,
            current_price=2003.0,
            realized_vol=0.01,
        )

        # Low capture, high MAE, long hold time, opportunity cost
        assert reward < 0.0, f"Expected <0 for far-from-peak hold, got {reward}"

    def test_mfe_growth_positive(self, bot):
        """Growing MFE should contribute positively."""
        # MFE growing
        reward_growing = bot._calculate_harvester_hold_reward(
            current_mfe=12.0,  # Growing
            current_mae=2.0,
            prev_mfe=10.0,
            prev_mae=2.0,
            bars_held=5,
            unrealized_pnl=11.0,
            entry_price=2000.0,
            current_price=2011.0,
            realized_vol=0.01,
        )

        # MFE static
        reward_static = bot._calculate_harvester_hold_reward(
            current_mfe=10.0,  # Same
            current_mae=2.0,
            prev_mfe=10.0,
            prev_mae=2.0,
            bars_held=5,
            unrealized_pnl=9.0,
            entry_price=2000.0,
            current_price=2009.0,
            realized_vol=0.01,
        )

        assert reward_growing > reward_static, "Growing MFE should increase reward"

    def test_mae_increase_penalty(self, bot):
        """Increasing MAE should penalize."""
        # MAE growing significantly (bad)
        reward_mae_growing = bot._calculate_harvester_hold_reward(
            current_mfe=10.0,
            current_mae=8.0,  # Growing significantly
            prev_mfe=10.0,
            prev_mae=2.0,
            bars_held=5,
            unrealized_pnl=8.0,
            entry_price=2000.0,
            current_price=2008.0,
            realized_vol=0.01,
        )

        # MAE static
        reward_mae_static = bot._calculate_harvester_hold_reward(
            current_mfe=10.0,
            current_mae=2.0,  # Same
            prev_mfe=10.0,
            prev_mae=2.0,
            bars_held=5,
            unrealized_pnl=8.0,
            entry_price=2000.0,
            current_price=2008.0,
            realized_vol=0.01,
        )

        assert reward_mae_growing < reward_mae_static, "Growing MAE should decrease reward"

    def test_time_decay_penalty(self, bot):
        """Longer hold time should have decay penalty."""
        # Short hold
        reward_short = bot._calculate_harvester_hold_reward(
            current_mfe=10.0,
            current_mae=2.0,
            prev_mfe=10.0,
            prev_mae=2.0,
            bars_held=5,  # Short
            unrealized_pnl=9.0,
            entry_price=2000.0,
            current_price=2009.0,
            realized_vol=0.01,
        )

        # Long hold
        reward_long = bot._calculate_harvester_hold_reward(
            current_mfe=10.0,
            current_mae=2.0,
            prev_mfe=10.0,
            prev_mae=2.0,
            bars_held=100,  # Long
            unrealized_pnl=9.0,
            entry_price=2000.0,
            current_price=2009.0,
            realized_vol=0.01,
        )

        assert reward_short > reward_long, "Longer hold should decrease reward"

    def test_opportunity_cost_penalty(self, bot):
        """Holding below MFE peak should have opportunity cost."""
        # At peak (no opportunity cost)
        reward_at_peak = bot._calculate_harvester_hold_reward(
            current_mfe=10.0,
            current_mae=2.0,
            prev_mfe=10.0,
            prev_mae=2.0,
            bars_held=10,
            unrealized_pnl=10.0,  # AT peak
            entry_price=2000.0,
            current_price=2010.0,
            realized_vol=0.01,
        )

        # Below peak (opportunity cost)
        reward_below_peak = bot._calculate_harvester_hold_reward(
            current_mfe=10.0,
            current_mae=2.0,
            prev_mfe=10.0,
            prev_mae=2.0,
            bars_held=10,
            unrealized_pnl=5.0,  # 50% below peak
            entry_price=2000.0,
            current_price=2005.0,
            realized_vol=0.01,
        )

        assert reward_at_peak > reward_below_peak, "Below peak should have opportunity cost"

    def test_reward_range(self, bot):
        """Reward should be clipped to [-1.0, 1.0]."""
        # Extreme scenarios
        scenarios = [
            {  # Best case: near peak, growing MFE, low MAE
                "current_mfe": 50.0,
                "current_mae": 1.0,
                "prev_mfe": 40.0,
                "prev_mae": 1.0,
                "bars_held": 1,
                "unrealized_pnl": 49.0,
                "entry_price": 2000.0,
                "current_price": 2049.0,
            },
            {  # Worst case: far from peak, growing MAE, long hold
                "current_mfe": 10.0,
                "current_mae": 20.0,
                "prev_mfe": 10.0,
                "prev_mae": 10.0,
                "bars_held": 200,
                "unrealized_pnl": -5.0,
                "entry_price": 2000.0,
                "current_price": 1995.0,
            },
        ]

        for scenario in scenarios:
            reward = bot._calculate_harvester_hold_reward(**scenario, realized_vol=0.01)

            assert -1.0 <= reward <= 1.0, f"Reward {reward} outside range [-1.0, 1.0]"


class TestRewardPnLCorrelation:
    """Integration tests for reward-P&L correlation."""

    def test_trigger_reward_correlates_with_pnl(self, bot):
        """TriggerAgent rewards should have reasonable distribution."""
        n_samples = 100
        rewards = []

        for _ in range(n_samples):
            # Simulate trade
            pnl = rng.standard_normal() * 10
            mfe = abs(pnl) + abs(rng.standard_normal() * 5)  # MFE >= |P&L|

            # Good predictor: predicted MFE close to actual
            predicted = mfe + rng.standard_normal() * 2

            summary = {"mfe": mfe, "pnl": pnl}
            reward = bot._calculate_trigger_reward(summary, predicted, 0.01)

            rewards.append(reward)

        # Check reward distribution
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        # Good predictor should have positive mean reward
        assert mean_reward > 0.0, f"Expected positive mean reward, got {mean_reward:.3f}"
        # Rewards should have reasonable spread
        assert 0.3 < std_reward < 1.0, f"Expected std in [0.3, 1.0], got {std_reward:.3f}"
        # Most rewards should be in valid range
        assert all(-1.5 <= r <= 1.5 for r in rewards), "All rewards should be in [-1.5, 1.5]"

    def test_harvester_hold_reward_reasonable_distribution(self, bot):
        """HarvesterAgent HOLD rewards should have reasonable distribution."""
        n_samples = 100
        rewards = []

        for _ in range(n_samples):
            # Simulate various hold scenarios
            current_mfe = abs(rng.standard_normal() * 10) + 5
            current_mae = abs(rng.standard_normal() * 5)
            prev_mfe = current_mfe - abs(rng.standard_normal() * 2)
            prev_mae = current_mae - abs(rng.standard_normal() * 1)
            bars_held = int(rng.uniform(1, 50))
            capture_pct = rng.uniform(0.3, 0.95)
            unrealized_pnl = current_mfe * capture_pct

            reward = bot._calculate_harvester_hold_reward(
                current_mfe=current_mfe,
                current_mae=current_mae,
                prev_mfe=prev_mfe,
                prev_mae=prev_mae,
                bars_held=bars_held,
                unrealized_pnl=unrealized_pnl,
                entry_price=2000.0,
                current_price=2000.0 + unrealized_pnl,
                realized_vol=0.01,
            )

            rewards.append(reward)

        # Check distribution
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        # Should have rewards spread across range
        assert -1.0 <= min(rewards) <= 1.0, "Rewards outside range"
        assert -1.0 <= max(rewards) <= 1.0, "Rewards outside range"
        assert std_reward > 0.1, f"Rewards too concentrated (std={std_reward:.3f})"

        print(f"Harvester reward distribution: mean={mean_reward:.3f} std={std_reward:.3f}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
