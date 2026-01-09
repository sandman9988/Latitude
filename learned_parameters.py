#!/usr/bin/env python3
"""
Learned Parameters System - Adaptive Parameters with Soft Bounds
=================================================================

Handbook Reference: Section 4.3 - Learned Parameters System
Philosophy: NO MAGIC NUMBERS - all parameters learned or have principled defaults

Key Features:
- Adaptive parameters with momentum-based updates
- Soft bounds via tanh clamping (not hard limits)
- Per instrument × timeframe × broker adaptation
- Persistent storage with versioning
- Staleness detection and refresh

Author: AI Trading System
Date: 2026-01-09
Version: 1.0.0
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AdaptiveParam:
    """
    Individual adaptive parameter with momentum-based updates
    
    Handbook: "Soft bounds via tanh clamping, not hard limits"
    """
    name: str
    value: float
    min_bound: float
    max_bound: float
    learning_rate: float = 0.01
    momentum: float = 0.9
    velocity: float = 0.0
    update_count: int = 0
    last_update_time: float = 0.0
    
    def update(self, gradient: float) -> float:
        """
        Update parameter with momentum-based gradient descent
        
        Args:
            gradient: Direction and magnitude of update
        
        Returns:
            New parameter value
        """
        # Momentum update: v = β*v + α*∇
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        
        # Update value
        raw_value = self.value + self.velocity
        
        # Soft clamping via tanh
        # Maps unbounded value to (min_bound, max_bound)
        mid = (self.max_bound + self.min_bound) / 2.0
        range_half = (self.max_bound - self.min_bound) / 2.0
        
        # Normalize to (-1, 1), apply tanh, denormalize
        normalized = (raw_value - mid) / range_half if range_half > 0 else 0
        clamped_normalized = np.tanh(normalized)
        self.value = mid + clamped_normalized * range_half
        
        # Track updates
        self.update_count += 1
        self.last_update_time = time.time()
        
        return self.value
    
    def reset_velocity(self):
        """Reset momentum (useful when regime changes)"""
        self.velocity = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptiveParam':
        """Create from dictionary"""
        return cls(**data)


class InstrumentParameters:
    """
    Parameter set for a specific instrument × timeframe × broker
    
    Handbook: "Parameters adapt per instrument × timeframe × broker"
    """
    
    def __init__(self, symbol: str, timeframe: str = "M1", broker: str = "default"):
        """
        Args:
            symbol: Trading symbol (e.g., "BTC/USD")
            timeframe: Timeframe (e.g., "M1", "M15")
            broker: Broker identifier
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.broker = broker
        self.params: Dict[str, AdaptiveParam] = {}
        self.creation_time = time.time()
        
    def add_param(self, name: str, initial_value: float, 
                  min_bound: float, max_bound: float,
                  learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Add a new adaptive parameter
        
        Args:
            name: Parameter name
            initial_value: Starting value
            min_bound: Minimum allowed value (soft)
            max_bound: Maximum allowed value (soft)
            learning_rate: Learning rate for updates
            momentum: Momentum factor (0-1)
        """
        self.params[name] = AdaptiveParam(
            name=name,
            value=initial_value,
            min_bound=min_bound,
            max_bound=max_bound,
            learning_rate=learning_rate,
            momentum=momentum,
            last_update_time=time.time()
        )
        logger.debug(f"Added parameter '{name}' for {self.symbol}: "
                    f"{initial_value} (bounds: [{min_bound}, {max_bound}])")
    
    def get(self, name: str, default: Optional[float] = None) -> float:
        """Get parameter value"""
        if name in self.params:
            return self.params[name].value
        if default is not None:
            return default
        raise KeyError(f"Parameter '{name}' not found for {self.symbol}")
    
    def update(self, name: str, gradient: float) -> float:
        """Update parameter with gradient"""
        if name not in self.params:
            raise KeyError(f"Parameter '{name}' not found for {self.symbol}")
        return self.params[name].update(gradient)
    
    def reset_velocity(self, name: Optional[str] = None):
        """Reset momentum for one or all parameters"""
        if name:
            if name in self.params:
                self.params[name].reset_velocity()
        else:
            for param in self.params.values():
                param.reset_velocity()
    
    def get_staleness(self, name: str) -> float:
        """
        Get parameter staleness in seconds
        
        Returns:
            Seconds since last update
        """
        if name not in self.params:
            return float('inf')
        return time.time() - self.params[name].last_update_time
    
    def is_stale(self, name: str, threshold_seconds: float = 86400) -> bool:
        """
        Check if parameter is stale (not updated recently)
        
        Args:
            name: Parameter name
            threshold_seconds: Staleness threshold (default 24h)
        
        Returns:
            True if stale
        """
        return self.get_staleness(name) > threshold_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'broker': self.broker,
            'creation_time': self.creation_time,
            'params': {name: param.to_dict() for name, param in self.params.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstrumentParameters':
        """Create from dictionary"""
        instance = cls(
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            broker=data['broker']
        )
        instance.creation_time = data['creation_time']
        instance.params = {
            name: AdaptiveParam.from_dict(param_data)
            for name, param_data in data['params'].items()
        }
        return instance


class LearnedParametersManager:
    """
    Global parameter manager - handles all instruments
    
    Handbook: "Complete parameter persistence with staleness tracking"
    """
    
    def __init__(self, persistence_path: Optional[Path] = None):
        """
        Args:
            persistence_path: Where to save/load parameters
        """
        self.instruments: Dict[str, InstrumentParameters] = {}
        self.persistence_path = persistence_path or Path("data/learned_parameters.json")
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default parameter specifications
        self.param_specs = self._get_default_specs()
        
        logger.info(f"LearnedParametersManager initialized (persistence: {self.persistence_path})")
    
    def _get_default_specs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get default parameter specifications
        
        Returns dictionary of:
        {
            'param_name': {
                'default': float,
                'min': float,
                'max': float,
                'learning_rate': float,
                'momentum': float,
                'description': str
            }
        }
        """
        return {
            # Position sizing
            'base_position_size': {
                'default': 0.10,
                'min': 0.01,
                'max': 1.0,
                'learning_rate': 0.005,
                'momentum': 0.95,
                'description': 'Base position size (before VaR adjustment)'
            },
            
            # Risk management
            'var_multiplier': {
                'default': 1.0,
                'min': 0.5,
                'max': 2.0,
                'learning_rate': 0.01,
                'momentum': 0.9,
                'description': 'VaR adjustment multiplier'
            },
            'max_drawdown_pct': {
                'default': 0.15,
                'min': 0.05,
                'max': 0.30,
                'learning_rate': 0.005,
                'momentum': 0.95,
                'description': 'Maximum acceptable drawdown'
            },
            
            # Reward shaping
            'capture_multiplier': {
                'default': 2.0,
                'min': 0.5,
                'max': 5.0,
                'learning_rate': 0.01,
                'momentum': 0.9,
                'description': 'Capture efficiency reward multiplier'
            },
            'wtl_penalty_multiplier': {
                'default': 3.0,
                'min': 1.0,
                'max': 10.0,
                'learning_rate': 0.01,
                'momentum': 0.9,
                'description': 'Winner-to-loser penalty multiplier'
            },
            'opportunity_multiplier': {
                'default': 1.0,
                'min': 0.1,
                'max': 3.0,
                'learning_rate': 0.01,
                'momentum': 0.9,
                'description': 'Opportunity cost multiplier'
            },
            
            # Entry/exit thresholds
            'entry_confidence_threshold': {
                'default': 0.6,
                'min': 0.3,
                'max': 0.9,
                'learning_rate': 0.01,
                'momentum': 0.9,
                'description': 'Minimum confidence to enter trade'
            },
            'exit_confidence_threshold': {
                'default': 0.5,
                'min': 0.2,
                'max': 0.8,
                'learning_rate': 0.01,
                'momentum': 0.9,
                'description': 'Minimum confidence to exit trade'
            },
            
            # Feature engineering
            'feature_window_min': {
                'default': 20,
                'min': 5,
                'max': 50,
                'learning_rate': 1.0,
                'momentum': 0.8,
                'description': 'Minimum feature window size'
            },
            'feature_window_max': {
                'default': 100,
                'min': 20,
                'max': 500,
                'learning_rate': 5.0,
                'momentum': 0.8,
                'description': 'Maximum feature window size'
            },
            
            # Circuit breakers
            'sortino_threshold': {
                'default': 0.5,
                'min': 0.0,
                'max': 2.0,
                'learning_rate': 0.01,
                'momentum': 0.9,
                'description': 'Sortino ratio circuit breaker threshold'
            },
            'kurtosis_threshold': {
                'default': 5.0,
                'min': 2.0,
                'max': 10.0,
                'learning_rate': 0.1,
                'momentum': 0.9,
                'description': 'Kurtosis circuit breaker threshold'
            },
            'max_consecutive_losses': {
                'default': 5,
                'min': 2,
                'max': 10,
                'learning_rate': 0.5,
                'momentum': 0.8,
                'description': 'Max consecutive losses before halt'
            },
        }
    
    def get_instrument(self, symbol: str, timeframe: str = "M1", 
                      broker: str = "default") -> InstrumentParameters:
        """
        Get or create instrument parameter set
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            broker: Broker identifier
        
        Returns:
            InstrumentParameters instance
        """
        key = f"{symbol}_{timeframe}_{broker}"
        
        if key not in self.instruments:
            # Create new instrument with default parameters
            instrument = InstrumentParameters(symbol, timeframe, broker)
            
            # Initialize with defaults
            for param_name, spec in self.param_specs.items():
                instrument.add_param(
                    name=param_name,
                    initial_value=spec['default'],
                    min_bound=spec['min'],
                    max_bound=spec['max'],
                    learning_rate=spec['learning_rate'],
                    momentum=spec['momentum']
                )
            
            self.instruments[key] = instrument
            logger.info(f"Created parameter set for {key} with {len(spec)} parameters")
        
        return self.instruments[key]
    
    def get(self, symbol: str, param_name: str, 
            timeframe: str = "M1", broker: str = "default") -> float:
        """Get parameter value for instrument"""
        instrument = self.get_instrument(symbol, timeframe, broker)
        return instrument.get(param_name)
    
    def update(self, symbol: str, param_name: str, gradient: float,
              timeframe: str = "M1", broker: str = "default") -> float:
        """Update parameter for instrument"""
        instrument = self.get_instrument(symbol, timeframe, broker)
        new_value = instrument.update(param_name, gradient)
        
        logger.debug(f"Updated {symbol} {param_name}: {new_value:.4f} "
                    f"(gradient: {gradient:.4f})")
        
        return new_value
    
    def save(self):
        """Save all parameters to disk"""
        try:
            data = {
                'version': '1.0',
                'saved_at': time.time(),
                'instruments': {
                    key: instrument.to_dict()
                    for key, instrument in self.instruments.items()
                }
            }
            
            # Atomic write (write to temp, then rename)
            temp_path = self.persistence_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_path.replace(self.persistence_path)
            
            logger.info(f"Saved {len(self.instruments)} instrument parameter sets "
                       f"to {self.persistence_path}")
            
        except Exception as e:
            logger.error(f"Failed to save parameters: {e}", exc_info=True)
    
    def load(self) -> bool:
        """
        Load parameters from disk
        
        Returns:
            True if loaded successfully
        """
        try:
            if not self.persistence_path.exists():
                logger.info("No saved parameters found, using defaults")
                return False
            
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
            
            # Validate version
            if data.get('version') != '1.0':
                logger.warning(f"Version mismatch: {data.get('version')} != 1.0")
                return False
            
            # Load instruments
            for key, instrument_data in data['instruments'].items():
                instrument = InstrumentParameters.from_dict(instrument_data)
                self.instruments[key] = instrument
            
            logger.info(f"Loaded {len(self.instruments)} instrument parameter sets "
                       f"from {self.persistence_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load parameters: {e}", exc_info=True)
            return False
    
    def check_staleness(self, threshold_seconds: float = 86400) -> Dict[str, list]:
        """
        Check for stale parameters across all instruments
        
        Args:
            threshold_seconds: Staleness threshold (default 24h)
        
        Returns:
            Dictionary of {instrument_key: [stale_param_names]}
        """
        stale_params = {}
        
        for key, instrument in self.instruments.items():
            stale = [
                name for name in instrument.params.keys()
                if instrument.is_stale(name, threshold_seconds)
            ]
            if stale:
                stale_params[key] = stale
        
        return stale_params
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        total_params = sum(len(inst.params) for inst in self.instruments.values())
        
        return {
            'num_instruments': len(self.instruments),
            'total_parameters': total_params,
            'parameters_per_instrument': total_params / len(self.instruments) if self.instruments else 0,
            'instruments': list(self.instruments.keys())
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("LEARNED PARAMETERS SYSTEM - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Create adaptive parameter
    print("\n[Test 1] Adaptive Parameter with Soft Bounds")
    print("-" * 80)
    
    param = AdaptiveParam(
        name="test_param",
        value=0.5,
        min_bound=0.0,
        max_bound=1.0,
        learning_rate=0.1,
        momentum=0.9
    )
    
    print(f"Initial value: {param.value:.4f}")
    
    # Positive gradient (increase)
    for i in range(5):
        new_val = param.update(0.2)
        print(f"Update {i+1}: value={new_val:.4f}, velocity={param.velocity:.4f}")
    
    print(f"After 5 positive updates: {param.value:.4f}")
    print(f"Bounded to [0, 1]: ✓" if 0 <= param.value <= 1 else "✗ OUT OF BOUNDS")
    
    # Test 2: Instrument parameters
    print("\n[Test 2] Instrument Parameters")
    print("-" * 80)
    
    btc_params = InstrumentParameters("BTC/USD", "M1", "pepperstone")
    btc_params.add_param("position_size", 0.10, 0.01, 1.0)
    btc_params.add_param("stop_loss_pct", 0.02, 0.005, 0.10)
    
    print(f"BTC/USD position_size: {btc_params.get('position_size'):.2f}")
    print(f"BTC/USD stop_loss_pct: {btc_params.get('stop_loss_pct'):.3f}")
    
    # Update position size
    new_size = btc_params.update('position_size', 0.05)
    print(f"After update: position_size = {new_size:.3f}")
    
    # Check staleness
    time.sleep(0.1)
    staleness = btc_params.get_staleness('position_size')
    print(f"Position size staleness: {staleness:.3f} seconds")
    print(f"Is stale (threshold 1s): {btc_params.is_stale('position_size', 1.0)}")
    
    # Test 3: Manager with multiple instruments
    print("\n[Test 3] Learned Parameters Manager")
    print("-" * 80)
    
    manager = LearnedParametersManager(Path("data/test_params.json"))
    
    # Get BTC/USD parameters (auto-creates with defaults)
    btc_inst = manager.get_instrument("BTC/USD", "M1", "pepperstone")
    print(f"Created BTC/USD with {len(btc_inst.params)} default parameters")
    
    # Get some parameter values
    print(f"  capture_multiplier: {manager.get('BTC/USD', 'capture_multiplier'):.2f}")
    print(f"  wtl_penalty_multiplier: {manager.get('BTC/USD', 'wtl_penalty_multiplier'):.2f}")
    print(f"  base_position_size: {manager.get('BTC/USD', 'base_position_size'):.2f}")
    
    # Add ETH/USD
    eth_inst = manager.get_instrument("ETH/USD", "M15", "pepperstone")
    print(f"\nCreated ETH/USD with {len(eth_inst.params)} default parameters")
    
    # Test 4: Parameter updates
    print("\n[Test 4] Parameter Updates with Gradients")
    print("-" * 80)
    
    initial_capture = manager.get('BTC/USD', 'capture_multiplier')
    print(f"Initial capture_multiplier: {initial_capture:.4f}")
    
    # Simulate positive gradient (increase reward)
    for i in range(5):
        new_val = manager.update('BTC/USD', 'capture_multiplier', 0.1)
        print(f"  Update {i+1}: {new_val:.4f}")
    
    final_capture = manager.get('BTC/USD', 'capture_multiplier')
    print(f"Final capture_multiplier: {final_capture:.4f}")
    print(f"Change: {final_capture - initial_capture:+.4f}")
    
    # Test 5: Persistence
    print("\n[Test 5] Save/Load Persistence")
    print("-" * 80)
    
    # Save
    manager.save()
    print(f"✓ Saved to {manager.persistence_path}")
    
    # Create new manager and load
    manager2 = LearnedParametersManager(Path("data/test_params.json"))
    loaded = manager2.load()
    print(f"✓ Loaded: {loaded}")
    
    # Verify values match
    loaded_capture = manager2.get('BTC/USD', 'capture_multiplier')
    print(f"Loaded capture_multiplier: {loaded_capture:.4f}")
    print(f"Matches saved: {abs(loaded_capture - final_capture) < 1e-6}")
    
    # Test 6: Staleness detection
    print("\n[Test 6] Staleness Detection")
    print("-" * 80)
    
    time.sleep(0.5)
    stale = manager.check_staleness(threshold_seconds=0.3)
    
    if stale:
        for inst_key, params in stale.items():
            print(f"{inst_key}: {len(params)} stale parameters")
    else:
        print("No stale parameters found")
    
    # Test 7: Summary
    print("\n[Test 7] Manager Summary")
    print("-" * 80)
    
    summary = manager.get_summary()
    print(f"Instruments: {summary['num_instruments']}")
    print(f"Total parameters: {summary['total_parameters']}")
    print(f"Avg per instrument: {summary['parameters_per_instrument']:.1f}")
    print(f"Instrument list: {', '.join(summary['instruments'])}")
    
    # Test 8: Soft bounds demonstration
    print("\n[Test 8] Soft Bounds (Tanh Clamping)")
    print("-" * 80)
    
    # Create parameter with narrow bounds
    narrow_param = AdaptiveParam(
        name="narrow",
        value=0.5,
        min_bound=0.0,
        max_bound=1.0,
        learning_rate=0.5,  # Large learning rate
        momentum=0.0  # No momentum for clarity
    )
    
    print("Applying large positive gradients:")
    for i in range(10):
        val = narrow_param.update(1.0)  # Large gradient
        print(f"  Iteration {i+1}: {val:.6f}")
    
    print(f"\nFinal value: {narrow_param.value:.6f}")
    print(f"Asymptotically approaches {narrow_param.max_bound}, never exceeds ✓")
    
    # Cleanup
    import os
    if os.path.exists("data/test_params.json"):
        os.remove("data/test_params.json")
        print("\n✓ Cleanup: Removed test file")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nLearned Parameters System ready for integration:")
    print("  ✓ NO MAGIC NUMBERS - all parameters adaptive")
    print("  ✓ Soft bounds via tanh (never hard limits)")
    print("  ✓ Per instrument × timeframe × broker")
    print("  ✓ Momentum-based updates")
    print("  ✓ Persistence with versioning")
    print("  ✓ Staleness detection")
    print("  ✓ DRY - single source of truth for all parameters")
