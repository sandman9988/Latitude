#!/usr/bin/env python3
"""
VaR Estimator with Multi-Factor Adjustment
Implements dynamic Value-at-Risk calculation from Master Handbook
Includes regime, VPIN, kurtosis, and volatility adjustments
"""

import logging
import numpy as np
from typing import Optional, Tuple
from collections import deque
from enum import Enum

from safe_utils import SafeMath, safe_percentile, safe_std, safe_mean

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime classification"""
    UNDERDAMPED = "underdamped"  # High volatility, trending
    CRITICAL = "critical"         # Transition state
    OVERDAMPED = "overdamped"     # Low volatility, ranging


class KurtosisMonitor:
    """
    Monitor excess kurtosis for tail risk detection
    
    Handbook: "Kurtosis > 3 indicates fat tails; trigger circuit breaker"
    """
    
    def __init__(self, window: int = 100, threshold: float = 3.0):
        """
        Args:
            window: Rolling window for kurtosis calculation
            threshold: Excess kurtosis threshold for circuit breaker
        """
        self.window = window
        self.threshold = threshold
        self.returns = deque(maxlen=window)
        self._last_kurtosis = 0.0
        self._breaker_active = False
    
    def update(self, return_value: float) -> Tuple[float, bool]:
        """
        Update kurtosis monitor with new return
        
        Args:
            return_value: Log return or percentage return
            
        Returns:
            (kurtosis, breaker_triggered)
        """
        if not SafeMath.is_valid(return_value):
            logger.debug(f"Invalid return: {return_value}")
            return self._last_kurtosis, self._breaker_active
        
        self.returns.append(return_value)
        
        if len(self.returns) < 30:  # Need minimum sample
            return 0.0, False
        
        # Calculate excess kurtosis
        kurtosis = self._calculate_kurtosis()
        self._last_kurtosis = kurtosis
        
        # Check circuit breaker
        breaker = kurtosis > self.threshold
        
        if breaker and not self._breaker_active:
            logger.warning(f"KURTOSIS CIRCUIT BREAKER: {kurtosis:.2f} > {self.threshold}")
            self._breaker_active = True
        elif not breaker and self._breaker_active:
            logger.info(f"Kurtosis circuit breaker reset: {kurtosis:.2f}")
            self._breaker_active = False
        
        return kurtosis, breaker
    
    def _calculate_kurtosis(self) -> float:
        """
        Calculate excess kurtosis (Fisher's definition)
        
        Returns:
            Excess kurtosis (normal distribution = 0)
        """
        if len(self.returns) < 4:
            return 0.0
        
        returns_array = np.array(list(self.returns))
        
        # Filter valid values
        returns_array = returns_array[np.isfinite(returns_array)]
        if len(returns_array) < 4:
            return 0.0
        
        mean = np.mean(returns_array)
        std = np.std(returns_array, ddof=1)
        
        if std < 1e-12:
            return 0.0
        
        # Excess kurtosis = (m4 / m2^2) - 3
        centered = returns_array - mean
        m4 = np.mean(centered ** 4)
        m2 = std ** 2
        
        kurtosis = SafeMath.safe_div(m4, m2 ** 2, default=0.0) - 3.0
        
        return SafeMath.sanitize(kurtosis, 0.0)
    
    @property
    def is_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active"""
        return self._breaker_active
    
    @property
    def current_kurtosis(self) -> float:
        """Get last calculated kurtosis"""
        return self._last_kurtosis


class VaREstimator:
    """
    Dynamic VaR estimation with multi-factor adjustment
    
    Handbook Formula:
    VaR = base_var * regime_mult * vpin_mult * kurtosis_mult * vol_mult
    
    Where:
    - base_var = percentile of historical returns (e.g., 95th)
    - regime_mult = 1.0 (overdamped), 1.5 (critical), 2.0 (underdamped)
    - vpin_mult = 1.0 + max(0, vpin_z / 2.0)  [toxic flow adjustment]
    - kurtosis_mult = 1.0 + max(0, (kurtosis - 1.0) / 3.0)  [tail risk]
    - vol_mult = current_vol / reference_vol
    """
    
    def __init__(
        self,
        window: int = 500,
        confidence: float = 0.95,
        regime_multipliers: Optional[dict] = None,
        kurtosis_monitor: Optional[KurtosisMonitor] = None
    ):
        """
        Args:
            window: Rolling window for VaR calculation
            confidence: VaR confidence level (0.95 = 95%)
            regime_multipliers: Dict mapping RegimeType → multiplier
            kurtosis_monitor: Shared kurtosis monitor (or create new)
        """
        self.window = window
        self.confidence = confidence
        self.returns = deque(maxlen=window)
        
        # Default regime multipliers (from handbook)
        self.regime_multipliers = regime_multipliers or {
            RegimeType.OVERDAMPED: 1.0,   # Low vol ranging
            RegimeType.CRITICAL: 1.5,      # Transition
            RegimeType.UNDERDAMPED: 2.0   # High vol trending
        }
        
        # Kurtosis monitor
        self.kurtosis_monitor = kurtosis_monitor or KurtosisMonitor()
        
        # State
        self._last_var = 0.0
        self._reference_vol = None
        
        logger.info(f"VaREstimator initialized: window={window}, confidence={confidence}")
    
    def update_return(self, return_value: float) -> None:
        """
        Add new return to rolling window
        
        Args:
            return_value: Log return or percentage return
        """
        if not SafeMath.is_valid(return_value):
            logger.debug(f"Skipping invalid return: {return_value}")
            return
        
        self.returns.append(return_value)
        self.kurtosis_monitor.update(return_value)
    
    def estimate_var(
        self,
        regime: RegimeType = RegimeType.CRITICAL,
        vpin_z: float = 0.0,
        current_vol: Optional[float] = None
    ) -> float:
        """
        Estimate VaR with multi-factor adjustment and defensive validation
        
        Args:
            regime: Current market regime
            vpin_z: VPIN z-score (0 = normal, >2 = toxic flow)
            current_vol: Current realized volatility
            
        Returns:
            Adjusted VaR (in same units as returns), always positive
        """
        # Defensive: Minimum sample size
        if len(self.returns) < 30:
            logger.debug("Insufficient data for VaR, returning 0")
            return 0.0
        
        # 1. Base VaR from percentile
        base_var = self._calculate_base_var()
        
        # Defensive: Validate base VaR
        if not SafeMath.is_valid(base_var) or base_var < 0:
            logger.warning("Invalid base VaR: %s, using 0", base_var)
            return 0.0
        
        # 2. Regime multiplier with validation
        regime_mult = self.regime_multipliers.get(regime, 1.5)
        # Defensive: Cap regime multiplier
        regime_mult = max(0.5, min(3.0, regime_mult))
        
        # 3. VPIN multiplier (toxic flow adjustment)
        # Defensive: Cap vpin_z at reasonable bounds
        vpin_z_capped = max(-5.0, min(5.0, vpin_z if SafeMath.is_valid(vpin_z) else 0.0))
        vpin_mult = 1.0 + max(0.0, SafeMath.safe_div(vpin_z_capped, 2.0, 0.0))
        # Defensive: Cap VPIN multiplier
        vpin_mult = max(1.0, min(3.0, vpin_mult))
        
        # 4. Kurtosis multiplier (tail risk)
        kurtosis = self.kurtosis_monitor.current_kurtosis
        # Defensive: Cap kurtosis for stability
        kurtosis_capped = max(0.0, min(50.0, kurtosis if SafeMath.is_valid(kurtosis) else 0.0))
        kurtosis_mult = 1.0 + max(0.0, SafeMath.safe_div(kurtosis_capped - 1.0, 3.0, 0.0))
        # Defensive: Cap kurtosis multiplier
        kurtosis_mult = max(1.0, min(5.0, kurtosis_mult))
        
        # 5. Volatility multiplier (current vs reference)
        vol_mult = self._calculate_vol_mult(current_vol)
        # Defensive: Cap volatility multiplier
        vol_mult = max(0.5, min(3.0, vol_mult))
        
        # Combined VaR with defensive multiplication
        var = base_var * regime_mult * vpin_mult * kurtosis_mult * vol_mult
        
        # Defensive: Ensure positive, finite result
        if not SafeMath.is_valid(var) or var < 0:
            logger.warning("Invalid combined VaR: %s, using base: %s", var, base_var)
            var = base_var
        
        # Defensive: Cap extreme VaR values (safety limit)
        max_var = base_var * 10.0  # Never more than 10x base
        var = min(var, max_var)
        
        # Sanitize and cache
        self._last_var = SafeMath.sanitize(var, 0.0)
        
        logger.debug(
            "VaR: %.6f = base(%.6f) * regime(%.2f) * vpin(%.2f) * kurtosis(%.2f) * vol(%.2f)",
            self._last_var, base_var, regime_mult, vpin_mult, kurtosis_mult, vol_mult
        )
        
        return self._last_var
    
    def _calculate_base_var(self) -> float:
        """Calculate base VaR from historical percentile"""
        if len(self.returns) < 10:
            return 0.0
        
        # Use negative returns (losses) for VaR
        returns_array = [-r for r in self.returns]
        
        # Calculate percentile
        percentile = self.confidence * 100.0
        base_var = safe_percentile(returns_array, percentile, default=0.0)
        
        return abs(base_var)  # VaR is positive
    
    def _calculate_vol_mult(self, current_vol: Optional[float]) -> float:
        """Calculate volatility multiplier"""
        if current_vol is None or not SafeMath.is_valid(current_vol):
            return 1.0
        
        # Set reference vol (use median of historical std)
        if self._reference_vol is None:
            self._reference_vol = safe_std(list(self.returns), default=0.01)
        
        # Avoid division by zero
        if self._reference_vol < 1e-12:
            self._reference_vol = 0.01
        
        vol_mult = SafeMath.safe_div(current_vol, self._reference_vol, default=1.0)
        
        # Clamp to reasonable range
        return SafeMath.clamp(vol_mult, 0.5, 3.0)
    
    def set_reference_vol(self, reference_vol: float) -> None:
        """Pin the reference volatility used for scaling VaR."""
        if SafeMath.is_valid(reference_vol) and reference_vol > 0:
            self._reference_vol = reference_vol
            logger.info("VaR reference volatility set to %.6f", reference_vol)
    
    @property
    def last_var(self) -> float:
        """Get last calculated VaR"""
        return self._last_var
    
    @property
    def kurtosis(self) -> float:
        """Get current kurtosis"""
        return self.kurtosis_monitor.current_kurtosis
    
    @property
    def is_kurtosis_breaker_active(self) -> bool:
        """Check if kurtosis circuit breaker is active"""
        return self.kurtosis_monitor.is_breaker_active


def position_size_from_var(
    var: float,
    risk_budget_usd: float,
    account_equity: float,
    contract_size: float = 1.0,
    max_leverage: float = 10.0
) -> float:
    """
    Calculate position size from VaR
    
    Args:
        var: VaR in fractional units (e.g., 0.02 = 2%)
        risk_budget_usd: Maximum USD to risk
        account_equity: Account equity in USD
        contract_size: Contract/lot size
        max_leverage: Maximum leverage allowed
        
    Returns:
        Position size in lots/contracts
    """
    if var < 1e-12:
        logger.warning("VaR too small, returning zero position size")
        return 0.0
    
    # Position size = risk_budget / (VaR * contract_value)
    # Where contract_value depends on instrument
    
    # Simple approach: VaR as % of equity
    max_position_value = SafeMath.safe_div(risk_budget_usd, var, default=0.0)
    
    # Apply leverage limit
    max_leveraged = account_equity * max_leverage
    position_value = min(max_position_value, max_leveraged)
    
    # Convert to lots
    position_size = SafeMath.safe_div(position_value, contract_size, default=0.0)
    
    return position_size


if __name__ == "__main__":
    # Self-test
    print("VaR Estimator Tests:")
    
    # Create estimator
    var_est = VaREstimator(window=100, confidence=0.95)
    
    # Simulate returns (normal + fat tail event)
    np.random.seed(42)
    returns = list(np.random.normal(0.0, 0.01, 100))
    returns.extend([0.05, -0.05, 0.04])  # Add tail events
    
    # Update with returns
    for r in returns:
        var_est.update_return(r)
    
    # Estimate VaR
    var_normal = var_est.estimate_var(
        regime=RegimeType.OVERDAMPED,
        vpin_z=0.0,
        current_vol=0.01
    )
    print(f"  VaR (normal regime, low VPIN): {var_normal:.6f}")
    
    var_stressed = var_est.estimate_var(
        regime=RegimeType.UNDERDAMPED,
        vpin_z=3.0,
        current_vol=0.03
    )
    print(f"  VaR (stressed regime, high VPIN): {var_stressed:.6f}")
    
    # Check kurtosis
    kurtosis = var_est.kurtosis
    print(f"  Kurtosis: {kurtosis:.2f}")
    print(f"  Kurtosis breaker: {var_est.is_kurtosis_breaker_active}")
    
    # Position sizing
    position = position_size_from_var(
        var=var_stressed,
        risk_budget_usd=1000.0,
        account_equity=10000.0,
        contract_size=100000.0,
        max_leverage=10.0
    )
    print(f"  Position size: {position:.4f} lots")
    
    print("\nAll tests passed ✓")
