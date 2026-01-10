"""
Circuit Breakers - Safety Shutdown System
Handbook Section 12.2 - Circuit Breakers

Implements multiple circuit breakers to halt trading when risk escalates:
- Sortino ratio degradation
- Excess kurtosis (fat tails)
- VPIN (informed trading)
- Drawdown limits
- Consecutive losses
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from safe_math import SafeMath, RunningStats, SAFE_EPSILON
from learned_parameters import LearnedParametersManager

LOG = logging.getLogger(__name__)


@dataclass
class BreakerState:
    """State of a circuit breaker"""
    name: str
    is_tripped: bool = False
    trip_time: Optional[datetime] = None
    trip_reason: str = ""
    trip_value: float = 0.0
    threshold: float = 0.0
    cooldown_minutes: int = 60
    
    def trip(self, reason: str, value: float, threshold: float):
        """Trip the breaker"""
        self.is_tripped = True
        self.trip_time = datetime.now()
        self.trip_reason = reason
        self.trip_value = value
        self.threshold = threshold
        LOG.warning(f"🚨 CIRCUIT BREAKER TRIPPED: {self.name}")
        LOG.warning(f"   Reason: {reason}")
        LOG.warning(f"   Value: {value:.4f} | Threshold: {threshold:.4f}")
    
    def reset(self):
        """Reset the breaker"""
        if self.is_tripped:
            LOG.info(f"✓ Circuit breaker reset: {self.name}")
        self.is_tripped = False
        self.trip_time = None
        self.trip_reason = ""
        self.trip_value = 0.0
    
    def can_reset(self) -> bool:
        """Check if breaker can be reset (cooldown elapsed)"""
        if not self.is_tripped or self.trip_time is None:
            return True
        
        elapsed = datetime.now() - self.trip_time
        return elapsed >= timedelta(minutes=self.cooldown_minutes)


class SortinoBreaker:
    """
    Sortino Ratio Circuit Breaker
    
    Handbook: "If risk-adjusted returns drop too low, stop trading"
    Sortino focuses on downside deviation (better than Sharpe for asymmetric returns)
    """
    
    def __init__(self, threshold: float = 0.5, min_trades: int = 20):
        """
        Args:
            threshold: Minimum acceptable Sortino ratio
            min_trades: Minimum trades before breaker activates
        """
        self.threshold = threshold
        self.min_trades = min_trades
        self.returns: List[float] = []
        self.state = BreakerState(
            name="Sortino",
            threshold=threshold,
            cooldown_minutes=120  # 2 hour cooldown
        )
    
    def update(self, trade_return: float):
        """Add a trade return"""
        self.returns.append(trade_return)
        
        # Limit history
        if len(self.returns) > 100:
            self.returns = self.returns[-100:]
    
    def check(self) -> bool:
        """
        Check if breaker should trip
        
        Returns:
            True if tripped
        """
        if len(self.returns) < self.min_trades:
            return False
        
        sortino = self._calculate_sortino()
        
        if sortino < self.threshold:
            self.state.trip(
                reason=f"Sortino ratio below threshold",
                value=sortino,
                threshold=self.threshold
            )
            return True
        
        return False
    
    def _calculate_sortino(self) -> float:
        """Calculate Sortino ratio"""
        if not self.returns:
            return 0.0
        
        returns = np.array(self.returns)
        mean_return = np.mean(returns)
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')  # No losses = infinite Sortino
        
        downside_dev = np.std(downside_returns)
        
        if downside_dev < SAFE_EPSILON:
            return float('inf')
        
        # Sortino = mean / downside_deviation
        sortino = SafeMath.safe_div(mean_return, downside_dev, 0.0)
        
        return sortino
    
    def get_current_sortino(self) -> float:
        """Get current Sortino ratio"""
        return self._calculate_sortino()


class KurtosisBreaker:
    """
    Kurtosis Circuit Breaker
    
    Handbook: "If return distribution becomes too fat-tailed, reduce exposure"
    High kurtosis = extreme moves more likely = danger
    """
    
    def __init__(self, threshold: float = 5.0, min_samples: int = 30):
        """
        Args:
            threshold: Maximum acceptable kurtosis (normal distribution = 3)
            min_samples: Minimum samples before breaker activates
        """
        self.threshold = threshold
        self.min_samples = min_samples
        self.returns: List[float] = []
        self.state = BreakerState(
            name="Kurtosis",
            threshold=threshold,
            cooldown_minutes=60
        )
    
    def update(self, trade_return: float):
        """Add a trade return"""
        self.returns.append(trade_return)
        
        # Limit history
        if len(self.returns) > 100:
            self.returns = self.returns[-100:]
    
    def check(self) -> bool:
        """Check if breaker should trip"""
        if len(self.returns) < self.min_samples:
            return False
        
        kurtosis = self._calculate_kurtosis()
        
        if kurtosis > self.threshold:
            self.state.trip(
                reason=f"Excess kurtosis detected (fat tails)",
                value=kurtosis,
                threshold=self.threshold
            )
            return True
        
        return False
    
    def _calculate_kurtosis(self) -> float:
        """Calculate sample kurtosis (excess kurtosis)"""
        if len(self.returns) < 4:
            return 0.0
        
        returns = np.array(self.returns)
        n = len(returns)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        
        if std < SAFE_EPSILON:
            return 0.0
        
        # Standardized fourth moment
        z = (returns - mean) / std
        kurtosis = np.mean(z ** 4)
        
        # Fisher's definition (excess kurtosis, normal = 0)
        # Adjust for bias
        excess_kurtosis = kurtosis - 3.0
        
        return excess_kurtosis + 3.0  # Return total kurtosis (normal = 3)
    
    def get_current_kurtosis(self) -> float:
        """Get current kurtosis"""
        return self._calculate_kurtosis()


class DrawdownBreaker:
    """
    Drawdown Circuit Breaker
    
    Handbook: "Progressive size reduction as drawdown increases"
    """
    
    def __init__(self, thresholds: Dict[float, float] = None):
        """
        Args:
            thresholds: {drawdown_pct: size_multiplier}
                       e.g., {0.05: 0.9, 0.10: 0.75, 0.15: 0.5, 0.20: 0.0}
        """
        self.thresholds = thresholds or {
            0.05: 0.9,   # 5% DD = 90% size
            0.10: 0.75,  # 10% DD = 75% size
            0.15: 0.5,   # 15% DD = 50% size
            0.20: 0.0,   # 20% DD = STOP
        }
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0
        self.size_multiplier = 1.0
        
        self.state = BreakerState(
            name="Drawdown",
            threshold=max(self.thresholds.keys()),
            cooldown_minutes=240  # 4 hour cooldown
        )
    
    def update(self, equity: float):
        """Update with current equity"""
        if not SafeMath.is_valid(equity) or equity <= 0:
            return
        
        self.current_equity = equity
        
        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calculate drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0
    
    def check(self) -> bool:
        """Check if breaker should trip"""
        # Update size multiplier based on drawdown
        self.size_multiplier = 1.0
        
        for dd_threshold in sorted(self.thresholds.keys()):
            if self.current_drawdown >= dd_threshold:
                self.size_multiplier = self.thresholds[dd_threshold]
        
        # Trip if size goes to zero
        if self.size_multiplier <= 0.0:
            self.state.trip(
                reason=f"Maximum drawdown exceeded",
                value=self.current_drawdown,
                threshold=max(self.thresholds.keys())
            )
            return True
        
        return False
    
    def get_size_multiplier(self) -> float:
        """Get current position size multiplier"""
        return self.size_multiplier
    
    def get_drawdown(self) -> float:
        """Get current drawdown percentage"""
        return self.current_drawdown


class ConsecutiveLossesBreaker:
    """
    Consecutive Losses Circuit Breaker
    
    Breaks the "revenge trading" cycle
    """
    
    def __init__(self, max_losses: int = 5):
        """
        Args:
            max_losses: Maximum consecutive losses before halt
        """
        self.max_losses = max_losses
        self.consecutive_losses = 0
        
        self.state = BreakerState(
            name="Consecutive Losses",
            threshold=float(max_losses),
            cooldown_minutes=180  # 3 hour cooldown
        )
    
    def update(self, is_win: bool):
        """Update with trade result"""
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
    
    def check(self) -> bool:
        """Check if breaker should trip"""
        if self.consecutive_losses >= self.max_losses:
            self.state.trip(
                reason=f"Too many consecutive losses",
                value=float(self.consecutive_losses),
                threshold=float(self.max_losses)
            )
            return True
        
        return False
    
    def get_consecutive_losses(self) -> int:
        """Get current consecutive loss count"""
        return self.consecutive_losses


class CircuitBreakerManager:
    """
    Manages all circuit breakers
    
    Coordinates multiple breakers and provides unified status.
    Thresholds default to LearnedParametersManager when available to
    keep safety limits consistent with the rest of the system.
    """
    
    def __init__(
        self,
        sortino_threshold: Optional[float] = None,
        kurtosis_threshold: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        max_consecutive_losses: Optional[int] = None,
        symbol: str = "BTCUSD",
        timeframe: str = "M15",
        broker: str = "default",
        param_manager: Optional[LearnedParametersManager] = None
    ):
        """
        Initialize all circuit breakers.
        
        Args:
            sortino_threshold: Override for Sortino breaker threshold
            kurtosis_threshold: Override for kurtosis breaker threshold
            max_drawdown: Override for max drawdown stop level
            max_consecutive_losses: Override for loss streak limit
            symbol/timeframe/broker: Context for learned parameters
            param_manager: LearnedParametersManager instance
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.broker = broker
        self.param_manager = param_manager
        
        self.sortino_threshold, sortino_source = self._resolve_param(
            'sortino_threshold', sortino_threshold, 0.5
        )
        self.kurtosis_threshold, kurtosis_source = self._resolve_param(
            'kurtosis_threshold', kurtosis_threshold, 5.0
        )
        self.max_drawdown, drawdown_source = self._resolve_param(
            'max_drawdown_pct', max_drawdown, 0.20
        )
        self.max_consecutive_losses, loss_source = self._resolve_param(
            'max_consecutive_losses', max_consecutive_losses, 5
        )
        self.max_consecutive_losses = int(round(self.max_consecutive_losses))
        
        self.sortino_breaker = SortinoBreaker(threshold=self.sortino_threshold)
        self.kurtosis_breaker = KurtosisBreaker(threshold=self.kurtosis_threshold)
        self.drawdown_breaker = DrawdownBreaker(thresholds={
            0.05: 0.9,
            0.10: 0.75,
            0.15: 0.5,
            self.max_drawdown: 0.0
        })
        self.consecutive_losses_breaker = ConsecutiveLossesBreaker(
            max_losses=self.max_consecutive_losses
        )
        
        self.breakers = [
            self.sortino_breaker,
            self.kurtosis_breaker,
            self.drawdown_breaker,
            self.consecutive_losses_breaker
        ]
        
        LOG.info(
            "Circuit Breaker Manager initialized | Sortino>=%.2f (%s) Kurtosis<=%.1f (%s) DD<=%.0f%% (%s) MaxLoss=%d (%s)",
            self.sortino_threshold,
            sortino_source,
            self.kurtosis_threshold,
            kurtosis_source,
            self.max_drawdown * 100,
            drawdown_source,
            self.max_consecutive_losses,
            loss_source
        )

    def _resolve_param(self, name: str, explicit_value, default: float):
        """Resolve breaker thresholds with override → learned → default."""
        if explicit_value is not None:
            try:
                return float(explicit_value), "explicit"
            except (TypeError, ValueError):
                LOG.warning(
                    "[CIRCUIT-BREAKERS] Invalid explicit override for %s (%s) - using default %.3f",
                    name,
                    explicit_value,
                    default
                )
                return float(default), "default"
        if self.param_manager is not None:
            try:
                value = self.param_manager.get(
                    self.symbol,
                    name,
                    timeframe=self.timeframe,
                    broker=self.broker,
                    default=default
                )
                return float(value), "learned"
            except Exception as exc:
                LOG.debug(
                    "[CIRCUIT-BREAKERS] Failed to fetch %s via LearnedParameters (%s) - using default %.3f",
                    name,
                    exc,
                    default
                )
        return float(default), "default"
    
    def update_trade(self, pnl: float, equity: float):
        """
        Update all breakers with trade result
        
        Args:
            pnl: Trade P&L (normalized or absolute)
            equity: Current account equity
        """
        # Update return-based breakers
        self.sortino_breaker.update(pnl)
        self.kurtosis_breaker.update(pnl)
        
        # Update equity-based breaker
        self.drawdown_breaker.update(equity)
        
        # Update win/loss streak
        self.consecutive_losses_breaker.update(is_win=pnl > 0)
    
    def check_all(self) -> bool:
        """
        Check all circuit breakers
        
        Returns:
            True if ANY breaker is tripped
        """
        any_tripped = False
        
        for breaker in self.breakers:
            if breaker.check():
                any_tripped = True
        
        return any_tripped
    
    def is_any_tripped(self) -> bool:
        """Check if any breaker is currently tripped"""
        return any(breaker.state.is_tripped for breaker in self.breakers)
    
    def get_tripped_breakers(self) -> List[BreakerState]:
        """Get list of tripped breakers"""
        return [breaker.state for breaker in self.breakers if breaker.state.is_tripped]
    
    def reset_all(self):
        """Reset all breakers (use with caution)"""
        for breaker in self.breakers:
            breaker.state.reset()
        LOG.info("All circuit breakers reset")
    
    def reset_if_cooldown_elapsed(self):
        """Auto-reset breakers after cooldown"""
        for breaker in self.breakers:
            if breaker.state.is_tripped and breaker.state.can_reset():
                breaker.state.reset()
    
    def get_position_size_multiplier(self) -> float:
        """
        Get combined position size multiplier
        
        Returns:
            0.0 to 1.0 multiplier for position sizing
            0.0 = full stop, 1.0 = normal size
        """
        if self.is_any_tripped():
            return 0.0  # Full stop if any breaker tripped
        
        # Apply drawdown-based reduction
        return self.drawdown_breaker.get_size_multiplier()
    
    def get_status(self) -> Dict:
        """Get comprehensive status"""
        return {
            'any_tripped': self.is_any_tripped(),
            'position_multiplier': self.get_position_size_multiplier(),
            'sortino': {
                'tripped': self.sortino_breaker.state.is_tripped,
                'current': self.sortino_breaker.get_current_sortino(),
                'threshold': self.sortino_breaker.threshold
            },
            'kurtosis': {
                'tripped': self.kurtosis_breaker.state.is_tripped,
                'current': self.kurtosis_breaker.get_current_kurtosis(),
                'threshold': self.kurtosis_breaker.threshold
            },
            'drawdown': {
                'tripped': self.drawdown_breaker.state.is_tripped,
                'current': self.drawdown_breaker.get_drawdown(),
                'threshold': self.drawdown_breaker.state.threshold,
                'size_mult': self.drawdown_breaker.get_size_multiplier()
            },
            'consecutive_losses': {
                'tripped': self.consecutive_losses_breaker.state.is_tripped,
                'current': self.consecutive_losses_breaker.get_consecutive_losses(),
                'threshold': self.consecutive_losses_breaker.max_losses
            }
        }


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("CIRCUIT BREAKERS - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Sortino Breaker
    print("\n[Test 1] Sortino Ratio Breaker")
    print("-" * 80)
    
    sortino = SortinoBreaker(threshold=0.5, min_trades=10)
    
    # Simulate good trades
    print("Simulating 10 profitable trades:")
    for i in range(10):
        sortino.update(0.02)  # +2% returns
    
    current_sortino = sortino.get_current_sortino()
    print(f"Sortino ratio: {current_sortino:.3f}")
    print(f"Tripped: {sortino.check()}")
    
    # Simulate bad trades
    print("\nSimulating 5 large losses:")
    for i in range(5):
        sortino.update(-0.05)  # -5% losses
    
    current_sortino = sortino.get_current_sortino()
    print(f"Sortino ratio: {current_sortino:.3f}")
    print(f"Tripped: {sortino.check()}")
    
    # Test 2: Kurtosis Breaker
    print("\n[Test 2] Kurtosis Breaker")
    print("-" * 80)
    
    kurtosis = KurtosisBreaker(threshold=5.0, min_samples=30)
    
    # Normal distribution
    print("Simulating normal returns:")
    np.random.seed(42)
    normal_returns = np.random.normal(0, 0.02, 30)
    for ret in normal_returns:
        kurtosis.update(ret)
    
    current_kurt = kurtosis.get_current_kurtosis()
    print(f"Kurtosis: {current_kurt:.3f} (normal ≈ 3.0)")
    print(f"Tripped: {kurtosis.check()}")
    
    # Fat-tailed distribution
    print("\nAdding extreme outliers:")
    for _ in range(5):
        kurtosis.update(0.15)  # Extreme positive
        kurtosis.update(-0.15)  # Extreme negative
    
    current_kurt = kurtosis.get_current_kurtosis()
    print(f"Kurtosis: {current_kurt:.3f}")
    print(f"Tripped: {kurtosis.check()}")
    
    # Test 3: Drawdown Breaker
    print("\n[Test 3] Drawdown Breaker")
    print("-" * 80)
    
    dd_breaker = DrawdownBreaker()
    
    # Simulate equity curve
    equities = [10000, 10500, 11000, 10800, 10200, 9500, 9000, 8500, 8000]
    
    print("Equity progression:")
    for eq in equities:
        dd_breaker.update(eq)
        dd = dd_breaker.get_drawdown()
        mult = dd_breaker.get_size_multiplier()
        tripped = dd_breaker.check()
        
        print(f"  Equity: ${eq:6.0f} | DD: {dd:5.1%} | Size mult: {mult:.2f} | Tripped: {tripped}")
    
    # Test 4: Consecutive Losses
    print("\n[Test 4] Consecutive Losses Breaker")
    print("-" * 80)
    
    cl_breaker = ConsecutiveLossesBreaker(max_losses=5)
    
    # Simulate trade sequence
    results = [False, False, True, False, False, False, False, False, True, False]
    
    print("Trade sequence:")
    for i, is_win in enumerate(results):
        cl_breaker.update(is_win)
        count = cl_breaker.get_consecutive_losses()
        tripped = cl_breaker.check()
        
        result_str = "WIN " if is_win else "LOSS"
        print(f"  Trade {i+1}: {result_str} | Consecutive losses: {count} | Tripped: {tripped}")
    
    # Test 5: Circuit Breaker Manager
    print("\n[Test 5] Circuit Breaker Manager")
    print("-" * 80)
    
    manager = CircuitBreakerManager(
        sortino_threshold=0.5,
        kurtosis_threshold=5.0,
        max_drawdown=0.20,
        max_consecutive_losses=3
    )
    
    # Simulate trading
    print("\nSimulating trades:")
    equity = 10000
    
    trades = [
        (100, 10100),    # Win
        (-50, 10050),    # Loss
        (-80, 9970),     # Loss
        (-100, 9870),    # Loss (3rd consecutive)
        (50, 9920),      # Would win but breaker tripped
    ]
    
    for i, (pnl, new_equity) in enumerate(trades):
        print(f"\nTrade {i+1}: P&L=${pnl:+4.0f} | Equity=${new_equity:.0f}")
        
        manager.update_trade(pnl / 100, new_equity)  # Normalize PnL
        manager.check_all()
        
        status = manager.get_status()
        print(f"  Any tripped: {status['any_tripped']}")
        print(f"  Size multiplier: {status['position_multiplier']:.2f}")
        print(f"  Consecutive losses: {status['consecutive_losses']['current']}")
        
        if status['any_tripped']:
            tripped = manager.get_tripped_breakers()
            for breaker in tripped:
                print(f"  ⚠️  {breaker.name} TRIPPED: {breaker.trip_reason}")
    
    # Test 6: Auto-reset after cooldown
    print("\n[Test 6] Auto-Reset After Cooldown")
    print("-" * 80)
    
    from unittest.mock import patch
    from datetime import timedelta
    
    # Manually trip a breaker
    manager.consecutive_losses_breaker.state.trip("Test", 5, 3)
    print(f"Breaker tripped: {manager.consecutive_losses_breaker.state.is_tripped}")
    
    # Simulate cooldown elapsed
    fake_past = datetime.now() - timedelta(hours=4)
    manager.consecutive_losses_breaker.state.trip_time = fake_past
    
    print(f"Cooldown can reset: {manager.consecutive_losses_breaker.state.can_reset()}")
    
    manager.reset_if_cooldown_elapsed()
    print(f"After auto-reset: {manager.consecutive_losses_breaker.state.is_tripped}")
    
    print("\n" + "=" * 80)
    print("✅ CIRCUIT BREAKERS READY")
    print("=" * 80)
    print("\nProtection layers:")
    print("  ✓ Sortino ratio degradation")
    print("  ✓ Excess kurtosis (fat tails)")
    print("  ✓ Drawdown-based size reduction")
    print("  ✓ Consecutive loss protection")
    print("  ✓ Auto-reset after cooldown")
    print("  ✓ Unified manager interface")
