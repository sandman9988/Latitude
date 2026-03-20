#!/usr/bin/env python3
"""
Path Geometry Module - Entry Trigger Features
==============================================
Implements PathGeometry from C# skeleton (Core/PathGeometry.cs)

Extracts path-based features for entry quality assessment:
- Efficiency: Displacement / path length
- Gamma (γ): Acceleration (second derivative of price)
- Jerk: Third derivative of price (rate of change of acceleration)
- Runway: Inverse volatility pressure
- Feasibility: Composite entry quality score

From MASTER_HANDBOOK.md: Path-Centric Experience Design
From C# Skeleton: AdaptiveRL_cTrader_Skeleton_v0_1/Core/PathGeometry.cs
"""

import logging
from collections import deque

from src.utils.safe_math import SafeMath

import numpy as np

# Path geometry calculation constants
MIN_BARS_FOR_GEOMETRY: int = 3  # Need at least 3 bars for derivatives

# Multi-horizon volatility ratio constants
VOL_RATIO_BLEND_WEIGHT: float = 0.30    # How much vol_ratio adjusts runway (0=ignore, 1=full)
VOL_RATIO_NEUTRAL: float = 1.0          # Vol ratio at which no adjustment is made
VOL_RATIO_RUNWAY_SCALE: float = 50.0    # Original C# skeleton constant for 1/(1+scale*sigma)

LOG = logging.getLogger(__name__)


class PathGeometry:
    """
    Calculate path geometry features for entry trigger evaluation.

    Features (all Rogers-Satchell normalized where applicable):
    - efficiency: Path efficiency (0..1) - displacement / total path length
    - gamma: Acceleration (2nd derivative) - normalized by volatility
    - jerk: Rate of change of acceleration (3rd derivative) - normalized by volatility
    - runway: Inverse volatility pressure (0..1)
    - feasibility: Composite entry quality (0..1)

    Philosophy:
    - High efficiency → Direct price movement (trending)
    - Low jerk → Smooth acceleration (predictable)
    - High runway → Low volatility headwind
    - High feasibility → Good entry opportunity
    """

    def __init__(self):
        """Initialize PathGeometry calculator."""
        self._prev_ret = 0.0
        self._prev_gamma = 0.0
        self._initialized = False

        # Last calculated snapshot
        self.last = {
            "efficiency": 0.0,
            "gamma": 0.0,
            "jerk": 0.0,
            "runway": 0.5,
            "feasibility": 0.5,
        }

    def update(
        self,
        bars: deque,
        sigma: float,
        sigma_long: float = 0.0,
    ) -> dict[str, float]:
        """
        Calculate path geometry from recent price bars.

        Args:
            bars: Deque of (t, o, h, l, c) tuples (at least 3 bars needed)
            sigma: Current short-term volatility (Rogers-Satchell or realized vol)
            sigma_long: Long-term volatility (50-bar). When > 0, the vol ratio
                        sigma/sigma_long modulates the runway estimate:
                        ratio > 1 → vol expanding → reduce runway
                        ratio < 1 → vol contracting → increase runway
                        When 0, falls back to pure inverse sigma.

        Returns:
            Dictionary with keys: efficiency, gamma, jerk, runway, feasibility
        """
        # Need at least 3 bars: c0, c1, c2
        if len(bars) < MIN_BARS_FOR_GEOMETRY or sigma <= 0:
            return self.last

        # Get last 3 close prices
        c0 = bars[-3][4]  # Close 2 bars ago
        c1 = bars[-2][4]  # Close 1 bar ago
        c2 = bars[-1][4]  # Latest close

        # Defensive: Validate prices
        if c0 <= 0 or c1 <= 0 or c2 <= 0:
            LOG.warning("[GEOM] Invalid prices: c0=%.4f, c1=%.4f, c2=%.4f", c0, c1, c2)
            return self.last

        # Calculate returns (velocity)
        r1 = (c1 - c0) / c0
        r2 = (c2 - c1) / c1

        # Gamma (acceleration): difference in returns
        gamma = r2 - r1

        # Jerk: rate of change of acceleration
        if self._initialized:
            jerk = gamma - self._prev_gamma
        elif len(bars) >= 4:
            # First call but enough bars — derive prev gamma from bars[-4:-1]
            c_m1 = bars[-4][4]
            if c_m1 > 0:
                r0 = (c0 - c_m1) / c_m1
                gamma_prev = r1 - r0
                jerk = gamma - gamma_prev
            else:
                jerk = 0.0
        else:
            jerk = 0.0

        # Efficiency: displacement / path length over 3 points
        displacement = abs(c2 - c0)
        path_length = abs(c1 - c0) + abs(c2 - c1)
        efficiency = min(1.0, displacement / path_length) if path_length > 0 else 0.0

        # Runway: inverse volatility pressure with optional multi-horizon blend
        # High sigma → low runway (harder to move through volatility)
        # Base formula from C# skeleton: 1.0 / (1.0 + 50.0 * sigma)
        base_runway = 1.0 / (1.0 + VOL_RATIO_RUNWAY_SCALE * sigma)

        if sigma_long > 0:
            # Vol ratio: short / long.  >1 = expanding vol, <1 = contracting
            vol_ratio = SafeMath.safe_div(sigma, sigma_long, VOL_RATIO_NEUTRAL)
            # Adjustment: ratio=1 → 1.0, ratio=2 → 0.7, ratio=0.5 → 1.15
            # Clamped to [0.5, 1.5] to prevent extreme adjustments.
            vol_adj = max(0.5, min(1.5,
                1.0 - VOL_RATIO_BLEND_WEIGHT * (vol_ratio - VOL_RATIO_NEUTRAL)))
            runway = base_runway * vol_adj
        else:
            runway = base_runway

        # Rogers-Satchell normalize gamma and jerk
        gamma_z = SafeMath.safe_div(gamma, sigma, 0.0)
        jerk_z = SafeMath.safe_div(jerk, sigma, 0.0)

        # Feasibility: composite score (from C# skeleton weights)
        # 40% efficiency, 30% smooth jerk, 20% runway, 10% base
        smooth_jerk = 1.0 - min(1.0, SafeMath.safe_div(abs(jerk), sigma, 0.0))
        feasibility = self._clamp01(0.40 * efficiency + 0.30 * smooth_jerk + 0.20 * runway + 0.10 * 0.5)  # Base score

        # Update state
        self.last = {
            "efficiency": efficiency,
            "gamma": gamma_z,
            "jerk": jerk_z,
            "runway": runway,
            "feasibility": feasibility,
        }

        # Debug: Log geometry calculations
        LOG.debug(
            "[GEOM] sigma=%.6f eff=%.4f gam_z=%.4f jerk_z=%.4f runway=%.4f feas=%.4f",
            sigma,
            efficiency,
            gamma_z,
            jerk_z,
            runway,
            feasibility,
        )

        self._prev_gamma = gamma
        self._prev_ret = r2
        self._initialized = True

        LOG.debug(
            "[GEOM] eff=%.3f γ=%.3f jerk=%.3f runway=%.3f feas=%.3f",
            efficiency,
            gamma_z,
            jerk_z,
            runway,
            feasibility,
        )

        return self.last

    def get_feature_vector(self) -> np.ndarray:
        """
        Get geometry features as numpy array for RL state.

        Returns:
            Array: [efficiency, gamma, jerk, runway, feasibility]
        """
        return np.array(
            [
                self.last["efficiency"],
                self.last["gamma"],
                self.last["jerk"],
                self.last["runway"],
                self.last["feasibility"],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _clamp01(x: float) -> float:
        """Clamp value to [0, 1] range."""
        return max(0.0, min(1.0, x))
