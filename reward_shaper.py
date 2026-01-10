#!/usr/bin/env python3
"""
Reward Shaper - Asymmetric component-based reward calculation
Python port of MASTER_HANDBOOK.md Section 4.6 - Reward Shaping

Implements three reward components:
1. Capture Efficiency: Rewards capturing high % of MFE
2. Winner-to-Loser Penalty: Punishes giving back profits
3. Opportunity Cost: Penalizes missing potential profits
4. Activity Bonus: Rewards action when stagnant (NEW)
5. Counterfactual Adjustment: Penalty for early exits (NEW)
6. Ensemble Disagreement: Rewards exploration in uncertain states (NEW)

All weights are adaptive per instrument (NO MAGIC NUMBERS principle).
Uses LearnedParametersManager for DRY compliance.
"""

from typing import Dict, Optional
import math
from learned_parameters import LearnedParametersManager
from activity_monitor import ActivityMonitor, CounterfactualAnalyzer


class RewardShaper:
    """
    Asymmetric reward shaper for DDQN training.
    Implements component-based rewards with adaptive weights.
    
    Now uses LearnedParametersManager (DRY - single source of truth)
    Includes activity monitoring and counterfactual analysis
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSD",
        timeframe: str = "M15",
        broker: str = "default",
        param_manager: Optional[LearnedParametersManager] = None,
        activity_monitor: Optional[ActivityMonitor] = None
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.broker = broker
        
        # Use shared parameter manager (DRY)
        if param_manager is None:
            self.param_manager = LearnedParametersManager()
            self.param_manager.load()  # Load existing parameters if available
        else:
            self.param_manager = param_manager

        self._param_cache = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'broker': self.broker
        }
        
        # Activity monitoring (prevent learned helplessness)
        self.activity_monitor = activity_monitor or ActivityMonitor()
        
        # Counterfactual analysis (optimal vs actual exit)
        self.counterfactual = CounterfactualAnalyzer()
        
        # Statistics for monitoring
        self.total_rewards_calculated = 0
        self.component_stats = {
            'capture': {'sum': 0.0, 'count': 0},
            'wtl': {'sum': 0.0, 'count': 0},
            'opportunity': {'sum': 0.0, 'count': 0},
            'activity': {'sum': 0.0, 'count': 0},
            'counterfactual': {'sum': 0.0, 'count': 0},
            'ensemble': {'sum': 0.0, 'count': 0}  # NEW: Ensemble disagreement bonus
        }
    
    def _get_param(self, name: str, default: Optional[float] = None) -> float:
        return self.param_manager.get(
            self.symbol,
            name,
            timeframe=self.timeframe,
            broker=self.broker,
            default=default
        )
        
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
        capture_mult = self._get_param('capture_multiplier')
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
        wtl_penalty_mult = self._get_param('wtl_penalty_multiplier')
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
        opportunity_mult = self._get_param('opportunity_multiplier')
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
                - ensemble_bonus: Exploration bonus from disagreement (optional, NEW)
        
        Returns:
            Dictionary with component rewards and total (6 components)
        """
        # Extract trade data
        exit_pnl = trade_data.get('exit_pnl', 0.0)
        mfe = trade_data.get('mfe', 0.0)
        mae = trade_data.get('mae', 0.0)
        was_wtl = trade_data.get('winner_to_loser', False)
        bars_from_mfe = trade_data.get('bars_from_mfe', 0)
        potential_mfe = trade_data.get('potential_mfe', 0.0)
        signal_strength = trade_data.get('signal_strength', 1.0)
        
        # NEW: Counterfactual analysis (optimal vs actual exit)
        entry_price = trade_data.get('entry_price', 0.0)
        exit_price = trade_data.get('exit_price', 0.0)
        direction = trade_data.get('direction', 1)
        mfe_bar_offset = trade_data.get('mfe_bar_offset', 0)
        
        # Calculate components
        r_capture = self.calculate_capture_efficiency_reward(exit_pnl, mfe)
        r_wtl = self.calculate_wtl_penalty(was_wtl, mfe, exit_pnl, bars_from_mfe)
        r_opportunity = self.calculate_opportunity_cost(potential_mfe, signal_strength)
        
        # NEW: Activity bonus (exploration when stagnant)
        r_activity = self.activity_monitor.get_exploration_bonus()
        if r_activity > 0:
            self.component_stats['activity']['sum'] += r_activity
            self.component_stats['activity']['count'] += 1
        
        # NEW: Counterfactual reward (penalty for early exits)
        r_counterfactual = 0.0
        if entry_price > 0 and exit_price > 0 and mfe > 0:
            r_counterfactual, cf_metrics = self.counterfactual.analyze_exit(
                entry_price, exit_price, mfe, mfe_bar_offset, direction
            )
            self.component_stats['counterfactual']['sum'] += r_counterfactual
            self.component_stats['counterfactual']['count'] += 1
        
        # NEW: Ensemble disagreement bonus (epistemic uncertainty reward)
        r_ensemble = trade_data.get('ensemble_bonus', 0.0)
        if r_ensemble > 0:
            self.component_stats['ensemble']['sum'] += r_ensemble
            self.component_stats['ensemble']['count'] += 1
        
        # Weighted total (6 components now)
        # Component multipliers handle the adaptation
        weight_capture = 1.0
        weight_wtl = 1.0
        weight_opportunity = 0.5        # Reduced weight for missed opportunities
        weight_activity = 0.8            # Activity exploration bonus
        weight_counterfactual = 0.6      # Exit timing adjustment
        weight_ensemble = 0.4            # NEW: Ensemble disagreement bonus
        
        total_reward = (
            weight_capture * r_capture +
            weight_wtl * r_wtl +
            weight_opportunity * r_opportunity +
            weight_activity * r_activity +
            weight_counterfactual * r_counterfactual +
            weight_ensemble * r_ensemble
        )
        
        self.total_rewards_calculated += 1
        
        return {
            'capture_efficiency': r_capture,
            'wtl_penalty': r_wtl,
            'opportunity_cost': r_opportunity,
            'activity_bonus': r_activity,
            'counterfactual_adjustment': r_counterfactual,
            'ensemble_bonus': r_ensemble,  # NEW: 6th component
            'total_reward': total_reward,
            'components_active': sum([
                1 if r_capture != 0 else 0,
                1 if r_wtl != 0 else 0,
                1 if r_opportunity != 0 else 0,
                1 if r_activity != 0 else 0,
                1 if r_counterfactual != 0 else 0,
                1 if r_ensemble != 0 else 0
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
                'capture_multiplier': self._get_param('capture_multiplier'),
                'wtl_penalty_multiplier': self._get_param('wtl_penalty_multiplier'),
                'opportunity_multiplier': self._get_param('opportunity_multiplier'),
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
        context = f"{self.symbol}_{self.timeframe}_{self.broker}"
        
        summary = f"""
╔══════════════════════════════════════════════════════════════════╗
    ║               REWARD SHAPER SUMMARY - {context:^20}          ║
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
    
    # ========================================================================
    # Phase 3.2: Specialized Dual-Agent Rewards
    # ========================================================================
    
    def calculate_trigger_reward(
        self,
        actual_mfe: float,
        predicted_runway: float,
        direction: int,
        entry_price: float
    ) -> Dict[str, float]:
        """
        Calculate reward for TriggerAgent (entry specialist).
        
        Measures runway utilization: How well did we predict MFE?
        
        Formula:
            runway_utilization = actual_MFE / predicted_runway
            reward = log(runway_utilization) if runway > 0 else large_penalty
            
        Asymmetric:
            - Underprediction (actual > predicted): Small positive reward
            - Good prediction (actual ≈ predicted): Maximum reward
            - Overprediction (actual < predicted): Larger penalty
        
        Args:
            actual_mfe: Actual maximum favorable excursion achieved
            predicted_runway: Predicted MFE from TriggerAgent
            direction: Trade direction (+1 LONG, -1 SHORT)
            entry_price: Entry price for percentage calculation
        
        Returns:
            Dict with 'runway_reward', 'utilization', 'error_pct'
        """
        if predicted_runway <= 0:
            # Invalid prediction - large penalty
            return {
                'runway_reward': -2.0,
                'utilization': 0.0,
                'error_pct': 100.0,
                'prediction_quality': 'INVALID'
            }
        
        # Runway utilization ratio
        utilization = actual_mfe / predicted_runway
        
        # Logarithmic reward (symmetric around 1.0)
        # - utilization = 1.0 → reward = 0.0 (perfect)
        # - utilization > 1.0 → positive (exceeded prediction)
        # - utilization < 1.0 → negative (fell short)
        if utilization > 0:
            base_reward = math.log(utilization)  # Natural log
        else:
            base_reward = -5.0  # No favorable movement - bad entry
        
        # Scale reward
        try:
            runway_mult = self._get_param('runway_multiplier')
        except KeyError:
            runway_mult = 2.0  # Default multiplier for runway prediction
        runway_reward = base_reward * runway_mult
        
        # Clip extreme values
        runway_reward = max(min(runway_reward, 3.0), -3.0)
        
        # Calculate error percentage
        error_pct = abs(actual_mfe - predicted_runway) / predicted_runway * 100
        
        # Quality assessment
        if 0.8 <= utilization <= 1.2:
            quality = 'EXCELLENT'  # Within 20%
        elif 0.5 <= utilization <= 1.5:
            quality = 'GOOD'  # Within 50%
        elif utilization < 0.5:
            quality = 'OVERPREDICTED'  # Predicted too high
        else:
            quality = 'UNDERPREDICTED'  # Predicted too low
        
        return {
            'runway_reward': runway_reward,
            'utilization': utilization,
            'error_pct': error_pct,
            'prediction_quality': quality,
            'actual_mfe': actual_mfe,
            'predicted_runway': predicted_runway
        }
    
    def calculate_harvester_reward(
        self,
        exit_pnl: float,
        mfe: float,
        mae: float,
        was_wtl: bool,
        bars_held: int,
        bars_from_mfe_to_exit: int = 0
    ) -> Dict[str, float]:
        """
        Calculate reward for HarvesterAgent (exit specialist).
        
        Combines:
        1. Capture efficiency (how much of MFE captured)
        2. WTL penalty (winner-to-loser prevention)
        3. Timing penalty (exiting too late after MFE)
        
        Formula:
            r_capture = (exit_pnl / mfe - 0.7) * multiplier
            r_wtl = -2.0 if WTL else 0.0
            r_timing = -0.5 * (bars_from_mfe / bars_held)
            total = r_capture + r_wtl + r_timing
        
        Args:
            exit_pnl: Final P&L at exit
            mfe: Maximum favorable excursion
            mae: Maximum adverse excursion
            was_wtl: Winner-to-loser flag
            bars_held: Total bars position was held
            bars_from_mfe_to_exit: Bars between MFE and exit
        
        Returns:
            Dict with component rewards and total
        """
        # 1. Capture efficiency
        if mfe > 0:
            capture_ratio = exit_pnl / mfe
            target_capture = 0.7  # Aim for 70% of MFE
            try:
                capture_mult = self._get_param('capture_multiplier')
            except KeyError:
                capture_mult = 2.0  # Default
            r_capture = (capture_ratio - target_capture) * capture_mult
        else:
            # No favorable movement - neutral
            capture_ratio = 0.0
            r_capture = 0.0
        
        # 2. WTL penalty (strong negative signal)
        try:
            wtl_mult = self._get_param('wtl_multiplier')
        except KeyError:
            wtl_mult = 3.0  # Default WTL penalty multiplier
        r_wtl = -wtl_mult if was_wtl else 0.0
        
        # 3. Timing penalty (waited too long after MFE)
        if bars_held > 0 and bars_from_mfe_to_exit > 0:
            timing_ratio = bars_from_mfe_to_exit / bars_held
            r_timing = -0.5 * timing_ratio  # Max penalty: -0.5
        else:
            r_timing = 0.0
        
        # Total harvester reward
        total_reward = r_capture + r_wtl + r_timing
        
        # Quality assessment
        if capture_ratio >= 0.8:
            quality = 'EXCELLENT'
        elif capture_ratio >= 0.6:
            quality = 'GOOD'
        elif capture_ratio >= 0.4:
            quality = 'FAIR'
        else:
            quality = 'POOR'
        
        return {
            'harvester_reward': total_reward,
            'capture_efficiency': r_capture,
            'wtl_penalty': r_wtl,
            'timing_penalty': r_timing,
            'capture_ratio': capture_ratio,
            'quality': quality,
            'was_wtl': was_wtl
        }
    
    def calculate_dual_agent_rewards(
        self,
        # Trigger data
        actual_mfe: float,
        predicted_runway: float,
        direction: int,
        entry_price: float,
        # Harvester data
        exit_pnl: float,
        mae: float,
        was_wtl: bool,
        bars_held: int,
        bars_from_mfe_to_exit: int = 0
    ) -> Dict[str, float]:
        """
        Calculate rewards for both trigger and harvester agents.
        
        This is the main reward method for dual-agent mode.
        
        Returns:
            Dict with:
                - trigger_reward: TriggerAgent reward
                - harvester_reward: HarvesterAgent reward  
                - total_reward: Combined reward for overall performance
                - All component breakdowns
        """
        # Calculate individual agent rewards
        trigger_result = self.calculate_trigger_reward(
            actual_mfe, predicted_runway, direction, entry_price
        )
        
        harvester_result = self.calculate_harvester_reward(
            exit_pnl, actual_mfe, mae, was_wtl, bars_held, bars_from_mfe_to_exit
        )
        
        # Combined total (weighted average)
        # Trigger: 40% weight (entry quality)
        # Harvester: 60% weight (exit execution is harder)
        total_reward = (
            0.4 * trigger_result['runway_reward'] +
            0.6 * harvester_result['harvester_reward']
        )
        
        return {
            'total_reward': total_reward,
            'trigger_reward': trigger_result['runway_reward'],
            'harvester_reward': harvester_result['harvester_reward'],
            'trigger_breakdown': trigger_result,
            'harvester_breakdown': harvester_result
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing RewardShaper module...")
    
    shaper = RewardShaper(symbol="BTCUSD", timeframe="M15")
    
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
    
    # ===== Phase 3.2: Dual-Agent Reward Tests =====
    print("\n" + "=" * 70)
    print("Phase 3.2: Dual-Agent Reward Tests")
    print("=" * 70)
    
    # Test 4: TriggerAgent - Perfect prediction
    print("\n=== Test 4: TriggerAgent - Perfect Prediction ===")
    trigger_result = shaper.calculate_trigger_reward(
        actual_mfe=0.0025,      # 25 pips achieved
        predicted_runway=0.0025, # 25 pips predicted
        direction=1,
        entry_price=100000.0
    )
    print(f"Runway Reward: {trigger_result['runway_reward']:+.4f}")
    print(f"Utilization: {trigger_result['utilization']:.2f}x")
    print(f"Error: {trigger_result['error_pct']:.1f}%")
    print(f"Quality: {trigger_result['prediction_quality']}")
    
    # Test 5: TriggerAgent - Exceeded prediction
    print("\n=== Test 5: TriggerAgent - Exceeded Prediction ===")
    trigger_result2 = shaper.calculate_trigger_reward(
        actual_mfe=0.0040,      # 40 pips achieved
        predicted_runway=0.0025, # 25 pips predicted (underpredicted)
        direction=1,
        entry_price=100000.0
    )
    print(f"Runway Reward: {trigger_result2['runway_reward']:+.4f}")
    print(f"Utilization: {trigger_result2['utilization']:.2f}x")
    print(f"Error: {trigger_result2['error_pct']:.1f}%")
    print(f"Quality: {trigger_result2['prediction_quality']}")
    
    # Test 6: TriggerAgent - Fell short
    print("\n=== Test 6: TriggerAgent - Fell Short ===")
    trigger_result3 = shaper.calculate_trigger_reward(
        actual_mfe=0.0010,      # 10 pips achieved
        predicted_runway=0.0025, # 25 pips predicted (overpredicted)
        direction=1,
        entry_price=100000.0
    )
    print(f"Runway Reward: {trigger_result3['runway_reward']:+.4f}")
    print(f"Utilization: {trigger_result3['utilization']:.2f}x")
    print(f"Error: {trigger_result3['error_pct']:.1f}%")
    print(f"Quality: {trigger_result3['prediction_quality']}")
    
    # Test 7: HarvesterAgent - Excellent capture
    print("\n=== Test 7: HarvesterAgent - Excellent Capture (85%) ===")
    harvester_result = shaper.calculate_harvester_reward(
        exit_pnl=0.0034,        # 34 pips captured
        mfe=0.0040,             # 40 pips MFE
        mae=0.0005,             # 5 pips MAE
        was_wtl=False,
        bars_held=15,
        bars_from_mfe_to_exit=3
    )
    print(f"Harvester Reward: {harvester_result['harvester_reward']:+.4f}")
    print(f"Capture Ratio: {harvester_result['capture_ratio']:.1%}")
    print(f"Quality: {harvester_result['quality']}")
    print(f"Components: capture={harvester_result['capture_efficiency']:+.4f}, "
          f"wtl={harvester_result['wtl_penalty']:+.4f}, timing={harvester_result['timing_penalty']:+.4f}")
    
    # Test 8: HarvesterAgent - WTL scenario
    print("\n=== Test 8: HarvesterAgent - Winner-to-Loser ===")
    harvester_result2 = shaper.calculate_harvester_reward(
        exit_pnl=-0.0010,       # -10 pips (loss)
        mfe=0.0040,             # Had 40 pips profit
        mae=0.0050,             # 50 pips adverse
        was_wtl=True,
        bars_held=30,
        bars_from_mfe_to_exit=25  # Waited 25 bars after MFE
    )
    print(f"Harvester Reward: {harvester_result2['harvester_reward']:+.4f}")
    print(f"Capture Ratio: {harvester_result2['capture_ratio']:.1%}")
    print(f"Quality: {harvester_result2['quality']}")
    print(f"WTL Penalty: {harvester_result2['wtl_penalty']:+.4f} (severe)")
    
    # Test 9: Full dual-agent rewards
    print("\n=== Test 9: Full Dual-Agent Trade ===")
    dual_result = shaper.calculate_dual_agent_rewards(
        # Trigger: predicted 25 pips, got 30 pips
        actual_mfe=0.0030,
        predicted_runway=0.0025,
        direction=1,
        entry_price=100000.0,
        # Harvester: captured 75% of MFE
        exit_pnl=0.0022,
        mae=0.0008,
        was_wtl=False,
        bars_held=20,
        bars_from_mfe_to_exit=5
    )
    print(f"Total Reward: {dual_result['total_reward']:+.4f}")
    print(f"  Trigger Reward (40%): {dual_result['trigger_reward']:+.4f}")
    print(f"  Harvester Reward (60%): {dual_result['harvester_reward']:+.4f}")
    print(f"Trigger Quality: {dual_result['trigger_breakdown']['prediction_quality']}")
    print(f"Harvester Quality: {dual_result['harvester_breakdown']['quality']}")
    
    print("\n" + "=" * 70)
    print("✅ All dual-agent reward tests complete!")
    print("=" * 70)
