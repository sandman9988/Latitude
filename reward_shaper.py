#!/usr/bin/env python3
"""
Reward Shaper - Asymmetric component-based reward calculation
Python port of MASTER_HANDBOOK.md Section 4.6 - Reward Shaping

Implements three reward components:
1. Capture Efficiency: Rewards capturing high % of MFE
2. Winner-to-Loser Penalty: Punishes giving back profits
3. Opportunity Cost: Penalizes missing potential profits

All weights are adaptive per instrument (NO MAGIC NUMBERS principle).
"""

import datetime as dt
from typing import Dict, Optional
import math


class AdaptiveRewardParams:
    """
    Self-optimizing reward parameters per instrument.
    Implements momentum-based updates following handbook design.
    """
    
    def __init__(self, name: str, initial_value: float, learning_rate: float = 0.01):
        self.name = name
        self.value = initial_value
        self.learning_rate = learning_rate
        self.momentum = 0.0
        self.momentum_decay = 0.9
        self.update_count = 0
        
    def update(self, gradient: float):
        """Update parameter using momentum-based gradient descent."""
        self.momentum = self.momentum_decay * self.momentum + (1 - self.momentum_decay) * gradient
        self.value += self.learning_rate * self.momentum
        self.update_count += 1
        
    def soft_clamp(self, min_val: float, max_val: float):
        """Soft clamping using tanh to avoid hard boundaries."""
        # Map value to [0, 1] then scale to [min_val, max_val]
        normalized = (math.tanh(self.value) + 1) / 2
        self.value = min_val + normalized * (max_val - min_val)


class RewardShaper:
    """
    Asymmetric reward shaper for DDQN training.
    Implements component-based rewards with adaptive weights.
    """
    
    def __init__(self, instrument: str = "BTCUSD"):
        self.instrument = instrument
        
        # Baseline MFE for normalization (in price units, will adapt)
        self.baseline_mfe = AdaptiveRewardParams("baseline_mfe", initial_value=100.0, learning_rate=0.005)
        
        # Capture efficiency parameters
        self.target_capture_ratio = AdaptiveRewardParams("target_capture", initial_value=0.7, learning_rate=0.01)
        self.capture_multiplier = AdaptiveRewardParams("capture_mult", initial_value=2.0, learning_rate=0.01)
        
        # WTL penalty parameters
        self.wtl_penalty_mult = AdaptiveRewardParams("wtl_penalty", initial_value=3.0, learning_rate=0.01)
        self.wtl_threshold = AdaptiveRewardParams("wtl_threshold", initial_value=10.0, learning_rate=0.005)
        
        # Opportunity cost parameters
        self.opportunity_weight = AdaptiveRewardParams("opp_weight", initial_value=1.0, learning_rate=0.01)
        self.opportunity_threshold = AdaptiveRewardParams("opp_threshold", initial_value=50.0, learning_rate=0.005)
        
        # Component weights (how much each component contributes to total reward)
        self.weight_capture = AdaptiveRewardParams("weight_capture", initial_value=1.0, learning_rate=0.01)
        self.weight_wtl = AdaptiveRewardParams("weight_wtl", initial_value=1.0, learning_rate=0.01)
        self.weight_opportunity = AdaptiveRewardParams("weight_opp", initial_value=0.5, learning_rate=0.01)
        
        # Statistics for monitoring
        self.total_rewards_calculated = 0
        self.component_stats = {
            'capture': {'sum': 0.0, 'count': 0},
            'wtl': {'sum': 0.0, 'count': 0},
            'opportunity': {'sum': 0.0, 'count': 0}
        }
        
    def calculate_capture_efficiency_reward(self, exit_pnl: float, mfe: float) -> float:
        """
        Reward based on how much of MFE was captured at exit.
        
        Formula from handbook:
        capture_ratio = exit_pnl / mfe
        r_capture = (capture_ratio - target_capture) * multiplier
        
        Args:
            exit_pnl: Final P&L at trade exit
            mfe: Maximum favorable excursion during trade
            
        Returns:
            Capture efficiency reward (positive if above target, negative if below)
        """
        if mfe <= 0:
            return 0.0
        
        capture_ratio = exit_pnl / mfe
        
        # Reward = difference from target × multiplier
        reward = (capture_ratio - self.target_capture_ratio.value) * self.capture_multiplier.value
        
        # Track statistics
        self.component_stats['capture']['sum'] += reward
        self.component_stats['capture']['count'] += 1
        
        return reward
    
    def calculate_wtl_penalty(self, was_wtl: bool, mfe: float, exit_pnl: float, 
                             bars_from_mfe_to_exit: int = 0) -> float:
        """
        Penalty for Winner-to-Loser trades (had profit, ended in loss).
        
        Formula from handbook:
        if was_winner_to_loser AND mfe > threshold:
            mfe_normalized = mfe / baseline_mfe
            giveback_ratio = (mfe - exit_pnl) / mfe
            time_penalty = 1 + (bars_from_mfe_to_exit / 10)
            r_wtl = -mfe_normalized * giveback_ratio * penalty_mult * time_penalty
        
        Args:
            was_wtl: Boolean flag from MFEMAETracker
            mfe: Maximum favorable excursion
            exit_pnl: Final P&L (negative for WTL)
            bars_from_mfe_to_exit: Time elapsed from MFE peak to exit
            
        Returns:
            Penalty (negative reward) for WTL, 0 otherwise
        """
        if not was_wtl or mfe < self.wtl_threshold.value:
            return 0.0
        
        # Normalize MFE by baseline
        mfe_normalized = mfe / max(self.baseline_mfe.value, 1.0)
        
        # Calculate how much profit was given back
        giveback_ratio = (mfe - exit_pnl) / mfe
        
        # Time penalty: longer hold after MFE = worse
        time_penalty = 1.0 + (bars_from_mfe_to_exit / 10.0)
        
        # Final penalty (negative reward)
        penalty = -mfe_normalized * giveback_ratio * self.wtl_penalty_mult.value * time_penalty
        
        # Track statistics
        self.component_stats['wtl']['sum'] += penalty
        self.component_stats['wtl']['count'] += 1
        
        return penalty
    
    def calculate_opportunity_cost(self, potential_mfe: float, signal_strength: float = 1.0) -> float:
        """
        Penalty for missed opportunities (didn't enter when signal was strong).
        
        Formula from handbook:
        if potential_mfe > threshold AND signal_strength > 0.5:
            opportunity_normalized = potential_mfe / baseline_mfe
            r_opportunity = -opportunity_normalized * signal_strength * weight * 0.3
        
        Args:
            potential_mfe: Estimated profit if had entered (from backtesting)
            signal_strength: Confidence of entry signal (0 to 1)
            
        Returns:
            Opportunity cost penalty (negative reward)
        """
        if potential_mfe < self.opportunity_threshold.value or signal_strength < 0.5:
            return 0.0
        
        # Normalize opportunity by baseline
        opportunity_normalized = potential_mfe / max(self.baseline_mfe.value, 1.0)
        
        # Penalty scaled by signal strength and weight
        penalty = -opportunity_normalized * signal_strength * self.opportunity_weight.value * 0.3
        
        # Track statistics
        self.component_stats['opportunity']['sum'] += penalty
        self.component_stats['opportunity']['count'] += 1
        
        return penalty
    
    def calculate_total_reward(self, trade_data: Dict) -> Dict[str, float]:
        """
        Calculate total reward from all components.
        
        Args:
            trade_data: Dictionary with keys:
                - exit_pnl: Final P&L
                - mfe: Maximum favorable excursion
                - mae: Maximum adverse excursion
                - winner_to_loser: WTL flag
                - bars_from_mfe: Time from MFE to exit (optional)
                - potential_mfe: Missed opportunity (optional)
                - signal_strength: Entry signal confidence (optional)
        
        Returns:
            Dictionary with component rewards and total
        """
        # Extract trade data
        exit_pnl = trade_data.get('exit_pnl', 0.0)
        mfe = trade_data.get('mfe', 0.0)
        mae = trade_data.get('mae', 0.0)
        was_wtl = trade_data.get('winner_to_loser', False)
        bars_from_mfe = trade_data.get('bars_from_mfe', 0)
        potential_mfe = trade_data.get('potential_mfe', 0.0)
        signal_strength = trade_data.get('signal_strength', 1.0)
        
        # Calculate components
        r_capture = self.calculate_capture_efficiency_reward(exit_pnl, mfe)
        r_wtl = self.calculate_wtl_penalty(was_wtl, mfe, exit_pnl, bars_from_mfe)
        r_opportunity = self.calculate_opportunity_cost(potential_mfe, signal_strength)
        
        # Weighted total
        total_reward = (
            self.weight_capture.value * r_capture +
            self.weight_wtl.value * r_wtl +
            self.weight_opportunity.value * r_opportunity
        )
        
        self.total_rewards_calculated += 1
        
        return {
            'capture_efficiency': r_capture,
            'wtl_penalty': r_wtl,
            'opportunity_cost': r_opportunity,
            'total_reward': total_reward,
            'components_active': sum([
                1 if r_capture != 0 else 0,
                1 if r_wtl != 0 else 0,
                1 if r_opportunity != 0 else 0
            ])
        }
    
    def update_baseline_mfe(self, observed_mfe: float, learning_rate: float = 0.05):
        """
        Update baseline MFE using exponential moving average.
        
        Args:
            observed_mfe: MFE from recent trade
            learning_rate: How quickly to adapt baseline
        """
        if observed_mfe > 0:
            # EMA update
            self.baseline_mfe.value = (
                (1 - learning_rate) * self.baseline_mfe.value +
                learning_rate * observed_mfe
            )
    
    def adapt_weights(self, performance_delta: float):
        """
        Adjust reward component weights based on performance feedback.
        
        This is where self-optimization happens - weights that lead to better
        performance get increased, weights that lead to worse get decreased.
        
        Args:
            performance_delta: Change in performance metric (e.g., Sharpe ratio)
        """
        # Simplified gradient: increase weights if performance improved
        gradient = performance_delta
        
        self.weight_capture.update(gradient)
        self.weight_wtl.update(gradient)
        self.weight_opportunity.update(gradient * 0.5)  # Opportunity cost less aggressive
        
        # Ensure weights stay positive and reasonable
        self.weight_capture.soft_clamp(0.1, 5.0)
        self.weight_wtl.soft_clamp(0.1, 5.0)
        self.weight_opportunity.soft_clamp(0.0, 2.0)
    
    def get_statistics(self) -> Dict:
        """Return statistics about reward components."""
        stats = {
            'total_rewards_calculated': self.total_rewards_calculated,
            'baseline_mfe': self.baseline_mfe.value,
            'parameters': {
                'target_capture_ratio': self.target_capture_ratio.value,
                'wtl_penalty_mult': self.wtl_penalty_mult.value,
                'opportunity_weight': self.opportunity_weight.value,
            },
            'weights': {
                'capture': self.weight_capture.value,
                'wtl': self.weight_wtl.value,
                'opportunity': self.weight_opportunity.value,
            }
        }
        
        # Calculate averages
        for component in ['capture', 'wtl', 'opportunity']:
            count = self.component_stats[component]['count']
            if count > 0:
                avg = self.component_stats[component]['sum'] / count
                stats[f'avg_{component}_reward'] = avg
            else:
                stats[f'avg_{component}_reward'] = 0.0
        
        return stats
    
    def print_summary(self) -> str:
        """Generate human-readable summary of reward shaper state."""
        stats = self.get_statistics()
        
        summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║               REWARD SHAPER SUMMARY - {self.instrument:^20}          ║
╚══════════════════════════════════════════════════════════════════╝

📊 REWARD STATISTICS
   Total Rewards Calculated: {stats['total_rewards_calculated']}
   Baseline MFE:             ${stats['baseline_mfe']:.2f}

⚙️  ADAPTIVE PARAMETERS
   Target Capture Ratio:     {stats['parameters']['target_capture_ratio']:.2%}
   WTL Penalty Multiplier:   {stats['parameters']['wtl_penalty_mult']:.2f}
   Opportunity Weight:       {stats['parameters']['opportunity_weight']:.2f}

🎚️  COMPONENT WEIGHTS (Self-Optimizing)
   Capture Efficiency:       {stats['weights']['capture']:.3f}
   WTL Penalty:              {stats['weights']['wtl']:.3f}
   Opportunity Cost:         {stats['weights']['opportunity']:.3f}

📈 AVERAGE COMPONENT REWARDS
   Capture Efficiency:       {stats['avg_capture_reward']:+.4f}
   WTL Penalty:              {stats['avg_wtl_reward']:+.4f}
   Opportunity Cost:         {stats['avg_opportunity_reward']:+.4f}
"""
        return summary


# Example usage and testing
if __name__ == "__main__":
    print("Testing RewardShaper module...")
    
    shaper = RewardShaper(instrument="BTCUSD")
    
    # Test 1: Good capture efficiency
    print("\n=== Test 1: Good Capture (exit_pnl=80, MFE=100) ===")
    reward1 = shaper.calculate_total_reward({
        'exit_pnl': 80.0,
        'mfe': 100.0,
        'mae': 20.0,
        'winner_to_loser': False
    })
    print(f"Capture Efficiency: {reward1['capture_efficiency']:+.4f}")
    print(f"Total Reward: {reward1['total_reward']:+.4f}")
    shaper.update_baseline_mfe(100.0)
    
    # Test 2: Winner-to-Loser scenario
    print("\n=== Test 2: Winner-to-Loser (MFE=150, exit_pnl=-30) ===")
    reward2 = shaper.calculate_total_reward({
        'exit_pnl': -30.0,
        'mfe': 150.0,
        'mae': 50.0,
        'winner_to_loser': True,
        'bars_from_mfe': 20
    })
    print(f"Capture Efficiency: {reward2['capture_efficiency']:+.4f}")
    print(f"WTL Penalty: {reward2['wtl_penalty']:+.4f}")
    print(f"Total Reward: {reward2['total_reward']:+.4f}")
    shaper.update_baseline_mfe(150.0)
    
    # Test 3: Missed opportunity
    print("\n=== Test 3: Missed Opportunity (potential_mfe=200, signal=0.8) ===")
    reward3 = shaper.calculate_total_reward({
        'exit_pnl': 0.0,
        'mfe': 0.0,
        'mae': 0.0,
        'winner_to_loser': False,
        'potential_mfe': 200.0,
        'signal_strength': 0.8
    })
    print(f"Opportunity Cost: {reward3['opportunity_cost']:+.4f}")
    print(f"Total Reward: {reward3['total_reward']:+.4f}")
    
    # Show summary
    print(shaper.print_summary())
    
    print("\n✅ All tests complete")
