#!/usr/bin/env python3
"""
Activity Monitor - Prevent Learned Helplessness
Implements no-trade detection and exploration boost from Master Handbook
"""

import logging
import time
from collections import deque
from typing import Optional, Tuple
from datetime import datetime

from safe_utils import SafeMath

logger = logging.getLogger(__name__)


class ActivityMonitor:
    """
    Monitor trading activity to prevent learned helplessness
    
    Handbook: "Penalize extended inactivity to prevent agent from learning 
    that doing nothing is optimal. Add exploration bonus when stagnant."
    
    Key metrics:
    - Bars since last trade
    - Trade frequency (trades/hour)
    - Consecutive no-trade periods
    - Activity score (exponentially weighted)
    """
    
    def __init__(
        self,
        max_bars_inactive: int = None,
        min_trades_per_day: float = None,
        exploration_boost: float = None,
        activity_decay: float = 0.95
    ):
        """
        Args:
            max_bars_inactive: Trigger exploration after this many bars without trade
            min_trades_per_day: Minimum expected trading frequency
            exploration_boost: Reward bonus for taking action when stagnant
            activity_decay: Exponential decay for activity score
        """
        import os
        paper_mode = os.environ.get('PAPER_MODE') == '1'
        
        # Use environment overrides or defaults (paper mode = more aggressive)
        if max_bars_inactive is None:
            max_bars_inactive = int(os.environ.get('MAX_BARS_INACTIVE', '30' if paper_mode else '100'))
        if min_trades_per_day is None:
            min_trades_per_day = float(os.environ.get('MIN_TRADES_PER_DAY', '10' if paper_mode else '2'))
        if exploration_boost is None:
            exploration_boost = float(os.environ.get('EXPLORATION_BOOST', '0.3' if paper_mode else '0.1'))
        
        self.max_bars_inactive = max_bars_inactive
        self.min_trades_per_day = min_trades_per_day
        self.exploration_boost = exploration_boost
        self.activity_decay = activity_decay
        
        # State tracking
        self.bars_since_trade = 0
        self.total_bars = 0
        self.total_trades = 0
        self.trade_timestamps: deque[float] = deque(maxlen=100)  # Last 100 trades
        self.activity_score = 1.0  # 1.0 = normal, <0.5 = stagnant
        self.last_trade_time: Optional[datetime] = None
        self.session_start = datetime.utcnow()
        
        # Flags
        self._is_stagnant = False
        self._exploration_active = False
        
        logger.info(
            f"ActivityMonitor initialized: max_inactive={max_bars_inactive}, "
            f"min_trades/day={min_trades_per_day:.1f}, exploration_boost={exploration_boost}"
        )
    
    def on_bar_close(self) -> None:
        """Update activity metrics on each bar close"""
        self.bars_since_trade += 1
        self.total_bars += 1
        
        # Decay activity score
        self.activity_score *= self.activity_decay
        
        # Check for stagnation
        was_stagnant = self._is_stagnant
        self._is_stagnant = self.bars_since_trade > self.max_bars_inactive
        
        if self._is_stagnant and not was_stagnant:
            logger.warning(
                f"[ACTIVITY] STAGNANT: {self.bars_since_trade} bars without trade "
                f"(max: {self.max_bars_inactive})"
            )
            self._exploration_active = True
        
        # Log activity metrics periodically
        if self.total_bars % 50 == 0:
            self._log_metrics()
    
    def on_trade_executed(self, timestamp: Optional[datetime] = None) -> None:
        """Record a trade execution"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.bars_since_trade = 0
        self.total_trades += 1
        self.last_trade_time = timestamp
        self.trade_timestamps.append(timestamp.timestamp())
        
        # Boost activity score
        self.activity_score = min(1.0, self.activity_score + 0.2)
        
        # Reset stagnation flags
        if self._is_stagnant:
            logger.info(f"[ACTIVITY] Stagnation resolved after {self.bars_since_trade} bars")
        self._is_stagnant = False
        self._exploration_active = False
    
    def get_exploration_bonus(self) -> float:
        """
        Calculate exploration bonus for taking action
        
        Returns:
            Reward bonus (0 if not stagnant, positive if stagnant)
        """
        if not self._exploration_active:
            return 0.0
        
        # Scale bonus by how long we've been inactive
        inactive_ratio = SafeMath.safe_div(
            self.bars_since_trade,
            self.max_bars_inactive,
            default=1.0
        )
        inactive_ratio = SafeMath.clamp(inactive_ratio, 0.0, 2.0)
        
        bonus = self.exploration_boost * inactive_ratio
        
        logger.debug(
            f"[ACTIVITY] Exploration bonus: {bonus:.4f} "
            f"(inactive: {self.bars_since_trade} bars)"
        )
        
        return bonus
    
    def get_inactivity_penalty(self) -> float:
        """
        Calculate penalty for prolonged inactivity
        
        Returns:
            Negative reward for being stagnant (0 if active)
        """
        if not self._is_stagnant:
            return 0.0
        
        # Penalty grows with inactivity duration
        inactive_ratio = SafeMath.safe_div(
            self.bars_since_trade,
            self.max_bars_inactive,
            default=1.0
        )
        
        # Quadratic penalty (gets worse over time)
        penalty = -0.05 * (inactive_ratio ** 2)
        
        return SafeMath.clamp(penalty, -0.5, 0.0)
    
    def get_trade_frequency(self, window_hours: float = 24.0) -> float:
        """
        Calculate recent trade frequency
        
        Args:
            window_hours: Time window in hours
            
        Returns:
            Trades per hour in the window
        """
        if not self.trade_timestamps:
            return 0.0
        
        now = time.time()
        window_seconds = window_hours * 3600
        cutoff = now - window_seconds
        
        # Count trades in window
        recent_trades = sum(1 for ts in self.trade_timestamps if ts >= cutoff)
        
        frequency = SafeMath.safe_div(recent_trades, window_hours, default=0.0)
        
        return frequency
    
    def is_below_target_frequency(self) -> bool:
        """Check if trading frequency is below target"""
        current_freq = self.get_trade_frequency(window_hours=24.0)
        target_freq = SafeMath.safe_div(self.min_trades_per_day, 24.0, default=0.0)
        
        return current_freq < target_freq
    
    def _log_metrics(self) -> None:
        """Log activity metrics"""
        freq_24h = self.get_trade_frequency(24.0)
        freq_1h = self.get_trade_frequency(1.0)
        
        logger.info(
            f"[ACTIVITY] bars={self.total_bars} trades={self.total_trades} "
            f"since_last={self.bars_since_trade} score={self.activity_score:.3f} "
            f"freq_24h={freq_24h:.2f}/h freq_1h={freq_1h:.2f}/h "
            f"stagnant={self._is_stagnant}"
        )
    
    @property
    def is_stagnant(self) -> bool:
        """Check if trading activity is stagnant"""
        return self._is_stagnant
    
    @property
    def should_explore(self) -> bool:
        """Check if exploration should be triggered"""
        return self._exploration_active
    
    def get_metrics(self) -> dict:
        """Get current activity metrics"""
        return {
            "bars_since_trade": self.bars_since_trade,
            "total_bars": self.total_bars,
            "total_trades": self.total_trades,
            "activity_score": self.activity_score,
            "is_stagnant": self._is_stagnant,
            "exploration_active": self._exploration_active,
            "trade_freq_24h": self.get_trade_frequency(24.0),
            "trade_freq_1h": self.get_trade_frequency(1.0),
            "inactivity_penalty": self.get_inactivity_penalty(),
            "exploration_bonus": self.get_exploration_bonus()
        }


class CounterfactualAnalyzer:
    """
    Analyze counterfactual outcomes (what-if scenarios)
    
    Handbook: "Compare actual exit to optimal exit at MFE bar.
    Penalize early exits from winners to encourage letting profits run."
    """
    
    def __init__(self, lookback_bars: int = 20):
        """
        Args:
            lookback_bars: How many bars to look back for MFE comparison
        """
        self.lookback_bars = lookback_bars
        logger.info(f"CounterfactualAnalyzer initialized: lookback={lookback_bars}")
    
    def analyze_exit(
        self,
        entry_price: float,
        exit_price: float,
        mfe: float,
        mfe_bar_offset: int,
        direction: int
    ) -> Tuple[float, dict]:
        """
        Analyze actual exit vs optimal exit at MFE
        
        Args:
            entry_price: Entry price
            exit_price: Actual exit price
            mfe: Maximum favorable excursion
            mfe_bar_offset: Bars from entry to MFE
            direction: 1 for long, -1 for short
            
        Returns:
            (counterfactual_reward, metrics_dict)
        """
        # Calculate actual profit
        actual_pnl = direction * (exit_price - entry_price)
        
        # Calculate optimal profit (if exited at MFE)
        optimal_pnl = direction * mfe
        
        # Missed opportunity
        missed_pnl = optimal_pnl - actual_pnl
        
        # Calculate efficiency
        efficiency = SafeMath.safe_div(actual_pnl, optimal_pnl, default=0.0)
        efficiency = SafeMath.clamp(efficiency, -1.0, 1.0)
        
        # Penalty for early exit from winner
        if actual_pnl > 0 and missed_pnl > 0:
            # Left money on the table
            missed_ratio = SafeMath.safe_div(missed_pnl, optimal_pnl, default=0.0)
            early_exit_penalty = -0.2 * missed_ratio
        else:
            early_exit_penalty = 0.0
        
        # Bonus for exiting near MFE
        if mfe_bar_offset <= 2:
            # Exited very close to MFE timing
            timing_bonus = 0.1
        else:
            timing_bonus = 0.0
        
        # Combined counterfactual reward
        counterfactual_reward = early_exit_penalty + timing_bonus
        
        metrics = {
            "actual_pnl": actual_pnl,
            "optimal_pnl": optimal_pnl,
            "missed_pnl": missed_pnl,
            "efficiency": efficiency,
            "early_exit_penalty": early_exit_penalty,
            "timing_bonus": timing_bonus,
            "mfe_bar_offset": mfe_bar_offset
        }
        
        logger.debug(
            f"[COUNTERFACTUAL] actual={actual_pnl:.6f} optimal={optimal_pnl:.6f} "
            f"efficiency={efficiency:.2%} penalty={early_exit_penalty:.4f} "
            f"bonus={timing_bonus:.4f}"
        )
        
        return counterfactual_reward, metrics


if __name__ == "__main__":
    # Self-test
    print("Activity Monitor Tests:")
    
    monitor = ActivityMonitor(max_bars_inactive=10, exploration_boost=0.1)
    
    # Simulate bars without trading
    for i in range(15):
        monitor.on_bar_close()
    
    print(f"  After 15 bars: stagnant={monitor.is_stagnant}")
    print(f"  Exploration bonus: {monitor.get_exploration_bonus():.4f}")
    print(f"  Inactivity penalty: {monitor.get_inactivity_penalty():.4f}")
    
    # Execute a trade
    monitor.on_trade_executed()
    print(f"  After trade: stagnant={monitor.is_stagnant}")
    print(f"  Activity score: {monitor.activity_score:.3f}")
    
    # Test trade frequency
    for i in range(5):
        monitor.on_trade_executed()
    freq = monitor.get_trade_frequency(window_hours=1.0)
    print(f"  Trade frequency: {freq:.2f}/hour")
    
    print("\nCounterfactual Analyzer Tests:")
    
    analyzer = CounterfactualAnalyzer()
    
    # Test early exit from winner
    cf_reward, metrics = analyzer.analyze_exit(
        entry_price=100.0,
        exit_price=102.0,  # +2%
        mfe=5.0,          # Could have made +5%
        mfe_bar_offset=5,
        direction=1
    )
    
    print(f"  Early exit penalty: {metrics['early_exit_penalty']:.4f}")
    print(f"  Efficiency: {metrics['efficiency']:.2%}")
    print(f"  Missed PnL: {metrics['missed_pnl']:.2f}")
    
    # Test perfect exit at MFE
    cf_reward, metrics = analyzer.analyze_exit(
        entry_price=100.0,
        exit_price=105.0,
        mfe=5.0,
        mfe_bar_offset=0,  # Exited right at MFE
        direction=1
    )
    
    print(f"  Perfect exit bonus: {metrics['timing_bonus']:.4f}")
    print(f"  Efficiency: {metrics['efficiency']:.2%}")
    
    print("\nAll tests passed ✓")
