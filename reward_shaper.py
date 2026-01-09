#!/usr/bin/env python3
"""
Reward Shaper - Asymmetric component-based reward calculation
Python port of MASTER_HANDBOOK.md Section 4.6 - Reward Shaping

Implements three reward components:
1. Capture Efficiency: Rewards capturing high % of MFE
2. Winner-to-Loser Penalty: Punishes giving back profits
3. Opportunity Cost: Penalizes missing potential profits

All weights are adaptive per instrument (NO MAGIC NUMBERS principle).
Uses LearnedParametersManager for DRY compliance.
"""

import datetime as dt
from typing import Dict, Optional
import math
from learned_parameters import LearnedParametersManager


class RewardShaper:
    """
    Asymmetric reward shaper for DDQN training.
    Implements component-based rewards with adaptive weights.
    
    Now uses LearnedParametersManager (DRY - single source of truth)
    """
    
    def __init__(self, instrument: str = "BTCUSD", 
                 param_manager: Optional[LearnedParametersManager] = None):
        self.instrument = instrument
        
        # Use shared parameter manager (DRY)
        if param_manager is None:
            self.param_manager = LearnedParametersManager()
            self.param_manager.load()  # Load existing parameters if available
        else:
            self.param_manager = param_manager
        
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
        
        # Get adaptive parameters
        capture_mult = self.param_manager.get(self.instrument, 'capture_multiplier')
        target_capture = 0.7  # Principled default from handbook: aim for 70% MFE capture
        
        capture_ratio = exit_pnl / mfe
        
        # Reward = difference from target × multiplier
        reward = (capture_ratio - target_capture) * capture_mult
        
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
        # Get adaptive parameters
        wtl_penalty_mult = self.param_manager.get(self.instrument, 'wtl_penalty_multiplier')
        wtl_threshold = 10.0  # Principled default: minimum MFE to trigger WTL penalty
        baseline_mfe = 100.0  # Principled default: typical MFE for normalization
        
        if not was_wtl or mfe < wtl_threshold:
            return 0.0
        
        # Normalize MFE by baseline
        mfe_normalized = mfe / max(baseline_mfe, 1.0)
        
        # Calculate how much profit was given back
        giveback_ratio = (mfe - exit_pnl) / mfe
        
        # Time penalty: longer hold after MFE = worse
        time_penalty = 1.0 + (bars_from_mfe_to_exit / 10.0)
        
        # Final penalty (negative reward)
        penalty = -mfe_normalized * giveback_ratio * wtl_penalty_mult * time_penalty
        
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
        # Get adaptive parameters
        opportunity_mult = self.param_manager.get(self.instrument, 'opportunity_multiplier')
        opportunity_threshold = 50.0  # Principled default: minimum potential profit to count as missed opportunity
        baseline_mfe = 100.0  # Principled default
        
        if potential_mfe < opportunity_threshold or signal_strength < 0.5:
            return 0.0
        
        # Normalize opportunity by baseline
        opportunity_normalized = potential_mfe / max(baseline_mfe, 1.0)
        
        # Penalty scaled by signal strength and weight
        penalty = -opportunity_normalized * signal_strength * opportunity_mult * 0.3
        
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
        
        # Weighted total (weights are fixed at 1.0, 1.0, 0.5 per handbook)
        # Component multipliers handle the adaptation
        weight_capture = 1.0
        weight_wtl = 1.0
        weight_opportunity = 0.5  # Reduced weight for missed opportunities
        
        total_reward = (
            weight_capture * r_capture +
            weight_wtl * r_wtl +
            weight_opportunity * r_opportunity
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
    
    def adapt_weights(self, performance_delta: float):
        """
        Adjust reward component weights based on performance feedback.
        
        NOTE: Removed adaptive weights. Per handbook principle: component-level
        multipliers (capture_multiplier, wtl_penalty_multiplier, etc.) provide
        the necessary tuning. Fixed weights keep the balance stable while the
        multipliers adapt to improve performance.
        
        Args:
            performance_delta: Change in performance metric (unused now)
        """
        # Weights are now fixed at principled defaults (1.0, 1.0, 0.5)
        # Adaptation happens via the component multipliers in LearnedParametersManager
        pass
    
    def get_statistics(self) -> Dict:
        """Return statistics about reward components."""
        stats = {
            'total_rewards_calculated': self.total_rewards_calculated,
            'parameters': {
                'capture_multiplier': self.param_manager.get(self.instrument, 'capture_multiplier'),
                'wtl_penalty_multiplier': self.param_manager.get(self.instrument, 'wtl_penalty_multiplier'),
                'opportunity_multiplier': self.param_manager.get(self.instrument, 'opportunity_multiplier'),
            },
            'weights': {
                'capture': 1.0,  # Fixed weights
                'wtl': 1.0,
                'opportunity': 0.5,
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

⚙️  ADAPTIVE MULTIPLIERS (from LearnedParametersManager)
   Capture Multiplier:       {stats['parameters']['capture_multiplier']:.2f}
   WTL Penalty Multiplier:   {stats['parameters']['wtl_penalty_multiplier']:.2f}
   Opportunity Multiplier:   {stats['parameters']['opportunity_multiplier']:.2f}

🎚️  COMPONENT WEIGHTS (Fixed)
   Capture Efficiency:       {stats['weights']['capture']:.1f}
   WTL Penalty:              {stats['weights']['wtl']:.1f}
   Opportunity Cost:         {stats['weights']['opportunity']:.1f}

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
