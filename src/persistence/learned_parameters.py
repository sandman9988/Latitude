#!/usr/bin/env python3
"""Learned Parameters System - Adaptive Parameters with Soft Bounds.
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

import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.persistence.atomic_persistence import AtomicPersistence

# Test and validation constants
FLOATING_POINT_TOLERANCE: float = 1e-6

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AdaptiveParam:
    """Individual adaptive parameter with momentum-based updates.

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
        """Update parameter with momentum-based gradient descent.

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

    def reset_velocity(self) -> None:
        """Reset momentum (useful when regime changes)."""
        self.velocity = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdaptiveParam":
        """Create from dictionary."""
        return cls(**data)


class InstrumentParameters:
    """Parameter set for a specific instrument × timeframe × broker.

    Handbook: "Parameters adapt per instrument × timeframe × broker"
    """

    def __init__(self, symbol: str, timeframe: str = "M1", broker: str = "default") -> None:
        """Args:
        symbol: Trading symbol (e.g., "BTC/USD")
        timeframe: Timeframe (e.g., "M1", "M15")
        broker: Broker identifier.

        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.broker = broker
        self.params: dict[str, AdaptiveParam] = {}
        self.creation_time = time.time()

    def add_param(
        self,
        name: str,
        initial_value: float,
        min_bound: float,
        max_bound: float,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        """Add a new adaptive parameter.

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
            last_update_time=time.time(),
        )
        logger.debug(
            "Added parameter '%s' for %s: %s (bounds: [%s, %s])",
            name,
            self.symbol,
            initial_value,
            min_bound,
            max_bound,
        )

    def get(self, name: str, default: float | None = None) -> float:
        """Get parameter value."""
        if name in self.params:
            return self.params[name].value
        if default is not None:
            return default
        msg = f"Parameter '{name}' not found for {self.symbol}"
        raise KeyError(msg)

    def update(self, name: str, gradient: float) -> float:
        """Update parameter with gradient."""
        if name not in self.params:
            msg = f"Parameter '{name}' not found for {self.symbol}"
            raise KeyError(msg)
        return self.params[name].update(gradient)

    def reset_velocity(self, name: str | None = None) -> None:
        """Reset momentum for one or all parameters."""
        if name:
            if name in self.params:
                self.params[name].reset_velocity()
        else:
            for param in self.params.values():
                param.reset_velocity()

    def get_staleness(self, name: str) -> float:
        """Get parameter staleness in seconds.

        Returns:
            Seconds since last update

        """
        if name not in self.params:
            return float("inf")
        return time.time() - self.params[name].last_update_time

    def is_stale(self, name: str, threshold_seconds: float = 86400) -> bool:
        """Check if parameter is stale (not updated recently).

        Args:
            name: Parameter name
            threshold_seconds: Staleness threshold (default 24h)

        Returns:
            True if stale

        """
        return self.get_staleness(name) > threshold_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "broker": self.broker,
            "creation_time": self.creation_time,
            "params": {name: param.to_dict() for name, param in self.params.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstrumentParameters":
        """Create from dictionary."""
        instance = cls(symbol=data["symbol"], timeframe=data["timeframe"], broker=data["broker"])
        instance.creation_time = data["creation_time"]
        instance.params = {name: AdaptiveParam.from_dict(param_data) for name, param_data in data["params"].items()}
        return instance


class LearnedParametersManager:
    """Global parameter manager - handles all instruments.

    Handbook: "Complete parameter persistence with staleness tracking"
    """

    def __init__(self, persistence_path: Path | None = None) -> None:
        """Args:
        persistence_path: Where to save/load parameters.

        """
        self.instruments: dict[str, InstrumentParameters] = {}
        self.persistence_path = (
            persistence_path or Path(os.environ.get("CTRADER_DATA_DIR", "data")) / "learned_parameters.json"
        )
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic persistence (with CRC32 and backups)
        self.atomic_persist = AtomicPersistence(base_dir=str(self.persistence_path.parent))

        # Default parameter specifications
        self.param_specs = self._get_default_specs()

        # Load saved parameters if they exist
        self.load()

    def _get_default_specs(self) -> dict[str, dict[str, Any]]:
        """Get default parameter specifications.

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
            "base_position_size": {
                "default": 0.10,
                "min": 0.01,
                "max": 1.0,
                "learning_rate": 0.005,
                "momentum": 0.95,
                "description": "Base position size (before VaR adjustment)",
            },
            # Risk management
            "var_multiplier": {
                "default": 1.0,
                "min": 0.5,
                "max": 2.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "VaR adjustment multiplier",
            },
            "max_drawdown_pct": {
                "default": 0.15,
                "min": 0.05,
                "max": 0.30,
                "learning_rate": 0.005,
                "momentum": 0.95,
                "description": "Maximum acceptable drawdown",
            },
            # Reward shaping
            "capture_multiplier": {
                "default": 2.0,
                "min": 0.5,
                "max": 5.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Capture efficiency reward multiplier",
            },
            "wtl_penalty_multiplier": {
                "default": 3.0,
                "min": 1.0,
                "max": 10.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Winner-to-loser penalty multiplier",
            },
            "opportunity_multiplier": {
                "default": 1.0,
                "min": 0.1,
                "max": 3.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Opportunity cost multiplier",
            },
            "pnl_alignment_multiplier": {
                "default": 1.5,
                "min": 0.0,
                "max": 3.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Realized PnL alignment reward multiplier",
            },
            # Entry/exit thresholds
            "entry_confidence_threshold": {
                "default": 0.6,
                "min": 0.3,
                "max": 0.9,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Minimum confidence to enter trade",
            },
            "exit_confidence_threshold": {
                "default": 0.5,
                "min": 0.2,
                "max": 0.8,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Minimum confidence to exit trade",
            },
            "feasibility_threshold": {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Minimum feasibility score required for entry",
            },
            "confidence_floor": {
                "default": 0.55,
                "min": 0.3,
                "max": 0.9,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Confidence floor for trigger agent",
            },
            "entry_conf_deadzone_low": {
                "default": 0.45,
                "min": 0.0,
                "max": 0.8,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Lower bound for confidence dead-zone entry block",
            },
            "entry_conf_deadzone_high": {
                "default": 0.55,
                "min": 0.2,
                "max": 0.95,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Upper bound for confidence dead-zone entry block",
            },
            "high_conf_risk_low": {
                "default": 0.80,
                "min": 0.5,
                "max": 0.98,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Lower confidence bound for high-confidence risk pocket",
            },
            "high_conf_risk_high": {
                "default": 0.90,
                "min": 0.6,
                "max": 0.99,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Upper confidence bound for high-confidence risk pocket",
            },
            "high_conf_vol_z_gate": {
                "default": 1.0,
                "min": 0.0,
                "max": 3.0,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "vol_z threshold for high-confidence risk-pocket blocking",
            },
            "high_conf_vpin_z_gate": {
                "default": 2.0,
                "min": 0.5,
                "max": 5.0,
                "learning_rate": 0.05,
                "momentum": 0.9,
                "description": "Absolute VPIN z threshold for high-confidence risk-pocket blocking",
            },
            "runway_cal_alpha": {
                "default": 0.22,
                "min": 0.05,
                "max": 0.60,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Base EWMA alpha for runway calibration speed",
            },
            "runway_error_abs_alpha": {
                "default": 0.25,
                "min": 0.05,
                "max": 0.80,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "EWMA alpha for absolute runway error tracking",
            },
            "runway_adapt_gain": {
                "default": 0.80,
                "min": 0.0,
                "max": 3.0,
                "learning_rate": 0.05,
                "momentum": 0.9,
                "description": "Adaptive gain scaling runway calibration alpha by prediction error",
            },
            "runway_huber_k": {
                "default": 1.0,
                "min": 0.1,
                "max": 3.0,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Huber clipping threshold for runway residual updates",
            },
            "runway_gate_min_fraction": {
                "default": 0.40,
                "min": 0.0,
                "max": 1.0,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Minimum fraction of Q_RUNWAY_MIN required for live runway-length entry gating",
            },
            "entry_guard_min_trade_samples": {
                "default": 40,
                "min": 5,
                "max": 400,
                "learning_rate": 1.0,
                "momentum": 0.8,
                "description": "Minimum closed trades before calibration/runway uplift terms are applied to entry guard",
            },
            "entry_guard_calib_err_start": {
                "default": 0.30,
                "min": 0.05,
                "max": 0.80,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Calibration-error EMA threshold where dynamic entry-floor uplift starts",
            },
            "entry_guard_calib_uplift_cap": {
                "default": 0.08,
                "min": 0.0,
                "max": 0.30,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Maximum calibration-error uplift added to dynamic entry floor",
            },
            "entry_guard_runway_acc_target": {
                "default": 0.60,
                "min": 0.30,
                "max": 0.95,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Runway-accuracy target used to compute dynamic entry-floor penalty",
            },
            "entry_guard_runway_penalty_cap": {
                "default": 0.05,
                "min": 0.0,
                "max": 0.20,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Maximum runway-accuracy penalty added to dynamic entry floor",
            },
            "entry_guard_rl_floor_extra_cap": {
                "default": 0.10,
                "min": 0.0,
                "max": 0.30,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Maximum extra headroom above base confidence floor allowed from RL entry-floor recommendation",
            },
            "zero_mfe_floor_frac": {
                "default": 1e-7,
                "min": 0.0,
                "max": 5e-4,
                "learning_rate": 0.00001,
                "momentum": 0.9,
                "description": "Fractional MFE floor used to classify near-zero-MFE trades",
            },
            "zero_mfe_conf_boost": {
                "default": 0.08,
                "min": 0.0,
                "max": 0.30,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Additional confidence-floor tightening gradient on zero-MFE trades",
            },
            "zero_mfe_feasibility_step": {
                "default": 0.02,
                "min": 0.0,
                "max": 0.10,
                "learning_rate": 0.005,
                "momentum": 0.9,
                "description": "Feasibility-threshold tightening step applied on zero-MFE trades",
            },
            "max_loss_cap_usd": {
                "default": 85.0,
                "min": 40.0,
                "max": 120.0,
                "learning_rate": 1.0,
                "momentum": 0.8,
                "description": "Per-trade max-loss cap in USD (adaptive tail-risk guard)",
            },
            "harvester_min_hold_ticks": {
                "default": 6,
                "min": 1,
                "max": 30,
                "learning_rate": 0.5,
                "momentum": 0.8,
                "description": "Minimum ticks held before non-emergency close decisions",
            },
            "harvester_min_hold_ticks_trend": {
                "default": 8,
                "min": 1,
                "max": 40,
                "learning_rate": 0.5,
                "momentum": 0.8,
                "description": "Minimum ticks held before non-emergency close decisions in trending regimes",
            },
            "harvester_early_adverse_mae_pct": {
                "default": 0.22,
                "min": 0.05,
                "max": 1.50,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Early adverse MAE%% threshold for fast protective exits",
            },
            "harvester_early_adverse_mfe_ceiling_pct": {
                "default": 0.08,
                "min": 0.0,
                "max": 1.00,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Maximum MFE%% allowed when early-adverse protection triggers",
            },
            "harvester_early_adverse_ticks": {
                "default": 120,
                "min": 20,
                "max": 500,
                "learning_rate": 1.0,
                "momentum": 0.8,
                "description": "Tick window where early-adverse protection is active",
            },
            "harvester_chop_soft_mult": {
                "default": 0.80,
                "min": 0.50,
                "max": 1.00,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Soft time-stop multiplier in mean-reverting/choppy regimes",
            },
            "harvester_chop_hard_mult": {
                "default": 0.90,
                "min": 0.60,
                "max": 1.20,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Hard time-stop multiplier in mean-reverting/choppy regimes",
            },
            "harvester_profit_target_pct": {
                "default": 2.50,
                "min": 0.05,
                "max": 5.00,
                "learning_rate": 0.10,
                "momentum": 0.5,
                "description": "Target MFE percentage for harvester exits",
            },
            "harvester_stop_loss_pct": {
                "default": 0.80,
                "min": 0.05,
                "max": 2.00,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Maximum MAE percentage before forced exit",
            },
            "harvester_soft_time_bars": {
                "default": 200,
                "min": 10,
                "max": 500,
                "learning_rate": 1.0,
                "momentum": 0.8,
                "description": "Bars held before soft time-based exit check",
            },
            "harvester_hard_time_bars": {
                "default": 400,
                "min": 20,
                "max": 800,
                "learning_rate": 1.0,
                "momentum": 0.8,
                "description": "Bars held before mandatory exit",
            },
            "harvester_min_soft_profit_pct": {
                "default": 0.20,
                "min": 0.01,
                "max": 1.00,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Minimum profit required to allow soft time stop",
            },
            "harvester_micro_zeta_ref": {
                "default": 0.50,
                "min": 0.20,
                "max": 1.20,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Regime reference zeta for adaptive micro-winner protection",
            },
            "harvester_micro_trend_relief": {
                "default": 0.35,
                "min": 0.0,
                "max": 0.90,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Trend-regime relief factor that dampens micro-winner exits",
            },
            "harvester_micro_giveback_relief_max": {
                "default": 0.25,
                "min": 0.0,
                "max": 0.80,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Max additive giveback allowance for micro-winner exits in trend regimes",
            },
            "harvester_micro_mfe_baseline_frac": {
                "default": 0.30,
                "min": 0.05,
                "max": 0.90,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Fraction of trailing activation used as adaptive micro-winner MFE threshold",
            },
            "runway_capture_floor": {
                "default": 0.55,
                "min": 0.10,
                "max": 1.50,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Harvester runway-capture ratio floor before allowing protective de-risk exits",
            },
            "runway_capture_giveback_frac": {
                "default": 0.20,
                "min": 0.0,
                "max": 0.80,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Allowed giveback fraction of current MFE after runway capture floor is reached",
            },
            # Feature engineering
            "feature_window_min": {
                "default": 20,
                "min": 5,
                "max": 50,
                "learning_rate": 1.0,
                "momentum": 0.8,
                "description": "Minimum feature window size",
            },
            "feature_window_max": {
                "default": 100,
                "min": 20,
                "max": 500,
                "learning_rate": 5.0,
                "momentum": 0.8,
                "description": "Maximum feature window size",
            },
            # Circuit breakers
            "sortino_threshold": {
                "default": 0.5,
                "min": 0.0,
                "max": 2.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Sortino ratio circuit breaker threshold",
            },
            "kurtosis_threshold": {
                "default": 5.0,
                "min": 2.0,
                "max": 10.0,
                "learning_rate": 0.1,
                "momentum": 0.9,
                "description": "Kurtosis circuit breaker threshold",
            },
            "max_consecutive_losses": {
                "default": 5,
                "min": 2,
                "max": 10,
                "learning_rate": 0.5,
                "momentum": 0.8,
                "description": "Max consecutive losses before halt",
            },
            # Market data parameters
            "depth_levels": {
                "default": 5,
                "min": 1,
                "max": 10,
                "learning_rate": 0.5,
                "momentum": 0.8,
                "description": "Order book depth levels to analyze",
            },
            "depth_buffer": {
                "default": 0.10,
                "min": 0.0,
                "max": 1.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Depth buffer for market making",
            },
            "spread_relax": {
                "default": 2.0,
                "min": 0.0,
                "max": 2.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Spread relaxation factor",
            },
            "vpin_z_threshold": {
                "default": 2.5,
                "min": 0.5,
                "max": 5.0,
                "learning_rate": 0.1,
                "momentum": 0.9,
                "description": "VPIN z-score threshold",
            },
            "vpin_z_limit": {
                "default": 2.5,
                "min": 0.5,
                "max": 5.0,
                "learning_rate": 0.1,
                "momentum": 0.9,
                "description": "VPIN z-score limit (alias for compatibility)",
            },
            "vpin_bucket_volume": {
                "default": 25.0,
                "min": 1.0,
                "max": 1000.0,
                "learning_rate": 1.0,
                "momentum": 0.8,
                "description": "Volume bucket size for VPIN calculation",
            },
            "volatility_reference": {
                "default": 0.005,
                "min": 0.001,
                "max": 0.10,
                "learning_rate": 0.001,
                "momentum": 0.95,
                "description": "Reference volatility for normalization",
            },
            "vol_ref": {
                "default": 0.005,
                "min": 0.001,
                "max": 0.10,
                "learning_rate": 0.001,
                "momentum": 0.95,
                "description": "Reference volatility (alias for compatibility)",
            },
            "volatility_cap": {
                "default": 0.05,
                "min": 0.01,
                "max": 0.20,
                "learning_rate": 0.005,
                "momentum": 0.9,
                "description": "Maximum volatility cap",
            },
            "vol_cap": {
                "default": 0.05,
                "min": 0.01,
                "max": 0.20,
                "learning_rate": 0.005,
                "momentum": 0.9,
                "description": "Volatility cap (alias for compatibility)",
            },
            "risk_budget_usd": {
                "default": 100.0,
                "min": 10.0,
                "max": 10000.0,
                "learning_rate": 10.0,
                "momentum": 0.9,
                "description": "Risk budget in USD",
            },
            # Self-calibrating instrument baselines (replace magic numbers)
            # These update via EMA from real trade data so the system adapts
            # to any instrument scale (XAUUSD, EURUSD, BTC, etc.) automatically.
            "mfe_p50_baseline": {
                "default": 10.0,
                "min": 0.01,
                "max": 100000.0,
                "learning_rate": 0.05,
                "momentum": 0.95,
                "description": "EMA-tracked p50 of trade MFE — instrument-agnostic replacement for BASELINE_MFE magic number",
            },
            "opportunity_p75_baseline": {
                "default": 15.0,
                "min": 0.01,
                "max": 100000.0,
                "learning_rate": 0.05,
                "momentum": 0.95,
                "description": "EMA-tracked p75 of trade MFE — instrument-agnostic opportunity cost threshold",
            },
            # Regime adjustment: fractional scale, not absolute price delta.
            # ±scale% applied to whatever base threshold is in use.
            # Self-adjusts: if regime-aware gating improves profits, scale grows.
            "regime_adj_scale": {
                "default": 0.15,
                "min": 0.0,
                "max": 0.50,
                "learning_rate": 0.005,
                "momentum": 0.9,
                "description": "Fractional regime threshold adjustment magnitude (instrument-agnostic, replaces TRIGGER_ADJUST_* constants)",
            },
            # Runway prediction accuracy EMAs — updated per trade, persisted across restarts.
            # Managed by TFAgent._close_position via set_value (direct write, no momentum).
            "runway_delta_ema": {
                "default": 0.0,
                "min": -10000.0,
                "max": 10000.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Signed EMA of (predicted_runway_pts - actual_mfe); positive = over-predicted",
            },
            "runway_accuracy_ema": {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "EMA of 1 - |runway_delta|/max_err in [0,1]; 1.0 = perfect prediction",
            },
            "conf_calib_err_ema": {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Brier score EMA of (entry_confidence - trade_outcome)^2; 0.0 = perfectly calibrated",
            },
            # Fallback MA-diff threshold — learned per instrument instead of
            # hardcoded LIVE_BASE_THRESHOLD / PAPER_BASE_THRESHOLD constants.
            "fallback_base_threshold": {
                "default": 0.3,
                "min": 0.01,
                "max": 2.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "MA-diff fallback strategy base threshold (learned per instrument)",
            },
            # ── Reward-shaping parameters (replaces hardcoded constants) ────────
            "reward_clip_harvester": {
                "default": 2.0,
                "min": 0.5,
                "max": 5.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Capture-reward clip bound for harvester agent",
            },
            "reward_clip_trigger": {
                "default": 0.5,
                "min": 0.1,
                "max": 2.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "PnL-reward clip bound for trigger agent",
            },
            "capture_baseline": {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Capture-ratio baseline subtracted before clipping (offline trainer)",
            },
            "capture_norm_factor": {
                "default": 0.3,
                "min": 0.05,
                "max": 1.0,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "description": "Capture normalisation divisor for preseed harvester reward",
            },
            # ── Reward component weights (adapted by RewardShaper.adapt_weights) ──
            "reward_weight_capture": {
                "default": 1.0,
                "min": 0.2,
                "max": 2.0,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Capture efficiency reward weight",
            },
            "reward_weight_wtl": {
                "default": 1.0,
                "min": 0.2,
                "max": 2.0,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Winner-to-loser penalty weight",
            },
            "reward_weight_opportunity": {
                "default": 0.5,
                "min": 0.2,
                "max": 2.0,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Missed MFE opportunity cost weight",
            },
            "reward_weight_activity": {
                "default": 0.8,
                "min": 0.2,
                "max": 2.0,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Exploration bonus weight when stagnant",
            },
            "reward_weight_counterfactual": {
                "default": 0.6,
                "min": 0.2,
                "max": 2.0,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Penalty for early exits vs optimal weight",
            },
            "reward_weight_ensemble": {
                "default": 0.4,
                "min": 0.2,
                "max": 2.0,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Epistemic uncertainty bonus weight",
            },
            "reward_weight_pnl_alignment": {
                "default": 0.6,
                "min": 0.2,
                "max": 2.0,
                "learning_rate": 0.02,
                "momentum": 0.9,
                "description": "Realized PnL alignment reward weight",
            },
        }

    def get_instrument(self, symbol: str, timeframe: str = "M1", broker: str = "default") -> InstrumentParameters:
        """Get or create instrument parameter set.

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
                    initial_value=spec["default"],
                    min_bound=spec["min"],
                    max_bound=spec["max"],
                    learning_rate=spec["learning_rate"],
                    momentum=spec["momentum"],
                )

            self.instruments[key] = instrument
            logger.info("Created parameter set for %s with %s parameters", key, len(self.param_specs))
        else:
            # Backfill any params added to param_specs after initial creation.
            # This handles schema evolution: new learned params introduced in
            # later versions are seeded with their defaults on existing instruments.
            instrument = self.instruments[key]
            added = []
            for param_name, spec in self.param_specs.items():
                if param_name not in instrument.params:
                    instrument.add_param(
                        name=param_name,
                        initial_value=spec["default"],
                        min_bound=spec["min"],
                        max_bound=spec["max"],
                        learning_rate=spec["learning_rate"],
                        momentum=spec["momentum"],
                    )
                    added.append(param_name)
            if added:
                logger.info("Backfilled %s new param(s) for %s: %s", len(added), key, added)

        return self.instruments[key]

    def get(
        self,
        symbol: str,
        param_name: str,
        timeframe: str = "M1",
        broker: str = "default",
        default: float | None = None,
    ) -> float:
        """Get parameter value for instrument."""
        instrument = self.get_instrument(symbol, timeframe, broker)
        return instrument.get(param_name, default)

    def update(
        self,
        symbol: str,
        param_name: str,
        gradient: float,
        timeframe: str = "M1",
        broker: str = "default",
    ) -> float:
        """Update parameter for instrument."""
        instrument = self.get_instrument(symbol, timeframe, broker)
        new_value = instrument.update(param_name, gradient)

        logger.debug("Updated %s %s: %.4f (gradient: %.4f)", symbol, param_name, new_value, gradient)

        return new_value

    def set_value(
        self,
        symbol: str,
        param_name: str,
        value: float,
        timeframe: str = "M1",
        broker: str = "default",
    ) -> float:
        """Directly set a parameter value, bypassing the momentum/gradient mechanism.

        Use this for EMA-style self-calibrating baselines (e.g. mfe_p50_baseline)
        where the caller manages the smoothing and just wants to write the result.
        The value is soft-clamped to the parameter's (min, max) bounds but does
        NOT go through the tanh sigmoid — that sigmoid is designed for gradient
        descent and produces distorted results when used to set absolute values
        on params with wide bounds.

        Args:
            symbol: Trading symbol
            param_name: Parameter name
            value: New value to set (will be clamped to bounds)
            timeframe: Timeframe
            broker: Broker identifier

        Returns:
            Clamped value actually stored

        """
        instrument = self.get_instrument(symbol, timeframe, broker)
        if param_name not in instrument.params:
            msg = f"Parameter '{param_name}' not found for {symbol}"
            raise KeyError(msg)
        param = instrument.params[param_name]
        clamped = max(param.min_bound, min(param.max_bound, value))
        param.value = clamped
        import time

        param.last_update_time = time.time()
        param.update_count += 1
        # Reset velocity so momentum does not carry stale direction forward
        param.velocity = 0.0
        logger.debug("Set %s %s = %.4f (direct)", symbol, param_name, clamped)
        return clamped

    def save(self) -> None:
        """Save all parameters to disk using atomic persistence with CRC32."""
        try:
            data = {
                "version": "1.0",
                "saved_at": time.time(),
                "instruments": {key: instrument.to_dict() for key, instrument in self.instruments.items()},
            }

            # Use atomic persistence with CRC32 and backup
            success = self.atomic_persist.save_json(data, self.persistence_path.name, create_backup=True)

            if success:
                logger.info(
                    "Saved %s instrument parameter sets to %s (atomic + CRC32)",
                    len(self.instruments),
                    self.persistence_path,
                )
            else:
                logger.error("Failed to save parameters atomically")

        except Exception as e:
            logger.error("Failed to save parameters: %s", e, exc_info=True)

    def load(self) -> bool:
        """Load parameters from disk using atomic persistence with CRC32 verification.

        Returns:
            True if loaded successfully

        """
        try:
            if not self.persistence_path.exists():
                logger.info("No saved parameters found, using defaults")
                return False

            # Use atomic persistence with CRC32 verification
            data = self.atomic_persist.load_json(self.persistence_path.name, verify_crc=True)

            if data is None:
                logger.warning("Failed to load parameters (CRC error or corrupt), using defaults")
                return False

            # Validate version
            if data.get("version") != "1.0":
                logger.warning("Version mismatch: %s != 1.0", data.get("version"))
                return False

            # Load instruments
            for key, instrument_data in data.get("instruments", {}).items():
                instrument = InstrumentParameters.from_dict(instrument_data)
                self.instruments[key] = instrument

            logger.info(
                "Loaded %s instrument parameter sets from %s (CRC32 verified)",
                len(self.instruments),
                self.persistence_path,
            )

            return True

        except Exception as e:
            logger.error("Failed to load parameters: %s", e, exc_info=True)
            return False

    def check_staleness(self, threshold_seconds: float = 86400) -> dict[str, list]:
        """Check for stale parameters across all instruments.

        Args:
            threshold_seconds: Staleness threshold (default 24h)

        Returns:
            Dictionary of {instrument_key: [stale_param_names]}

        """
        stale_params = {}

        for key, instrument in self.instruments.items():
            stale = [name for name in instrument.params if instrument.is_stale(name, threshold_seconds)]
            if stale:
                stale_params[key] = stale

        return stale_params

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        total_params = sum(len(inst.params) for inst in self.instruments.values())

        return {
            "num_instruments": len(self.instruments),
            "total_parameters": total_params,
            "parameters_per_instrument": (total_params / len(self.instruments) if self.instruments else 0),
            "instruments": list(self.instruments.keys()),
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test-script wide constants — change once, consistent everywhere
    _TEST_SYMBOL = "BTC/USD"
    _TEST_PARAMS_FILE = Path("data/test_params.json")


    # Test 1: Create adaptive parameter

    param = AdaptiveParam(name="test_param", value=0.5, min_bound=0.0, max_bound=1.0, learning_rate=0.1, momentum=0.9)


    # Positive gradient (increase)
    for _i in range(5):
        new_val = param.update(0.2)


    # Test 2: Instrument parameters

    btc_params = InstrumentParameters(_TEST_SYMBOL, "M1", "pepperstone")
    btc_params.add_param("position_size", 0.10, 0.01, 1.0)
    btc_params.add_param("stop_loss_pct", 0.02, 0.005, 0.10)


    # Update position size
    new_size = btc_params.update("position_size", 0.05)

    # Check staleness
    time.sleep(0.1)
    staleness = btc_params.get_staleness("position_size")

    # Test 3: Manager with multiple instruments

    manager = LearnedParametersManager(_TEST_PARAMS_FILE)

    # Get BTC/USD parameters (auto-creates with defaults)
    btc_inst = manager.get_instrument(_TEST_SYMBOL, "M1", "pepperstone")

    # Get some parameter values

    # Add ETH/USD
    eth_inst = manager.get_instrument("ETH/USD", "M15", "pepperstone")

    # Test 4: Parameter updates

    initial_capture = manager.get(_TEST_SYMBOL, "capture_multiplier")

    # Simulate positive gradient (increase reward)
    for _i in range(5):
        new_val = manager.update(_TEST_SYMBOL, "capture_multiplier", 0.1)

    final_capture = manager.get(_TEST_SYMBOL, "capture_multiplier")

    # Test 5: Persistence

    # Save
    manager.save()

    # Create new manager and load
    manager2 = LearnedParametersManager(_TEST_PARAMS_FILE)
    loaded = manager2.load()

    # Verify values match
    loaded_capture = manager2.get(_TEST_SYMBOL, "capture_multiplier")

    # Test 6: Staleness detection

    time.sleep(0.5)
    stale = manager.check_staleness(threshold_seconds=0.3)

    if stale:
        for _inst_key, _params in stale.items():
            pass
    else:
        pass

    # Test 7: Summary

    summary = manager.get_summary()

    # Test 8: Soft bounds demonstration

    # Create parameter with narrow bounds
    narrow_param = AdaptiveParam(
        name="narrow",
        value=0.5,
        min_bound=0.0,
        max_bound=1.0,
        learning_rate=0.5,  # Large learning rate
        momentum=0.0,  # No momentum for clarity
    )

    for _i in range(10):
        val = narrow_param.update(1.0)  # Large gradient


    # Cleanup
    _test_path = Path(_TEST_PARAMS_FILE)
    if _test_path.exists():
        _test_path.unlink()
