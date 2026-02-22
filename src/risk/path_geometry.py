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

import numpy as np

# Path geometry calculation constants
MIN_BARS_FOR_GEOMETRY: int = 3  # Need at least 3 bars for derivatives

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

    def update(self, bars: deque, sigma: float) -> dict[str, float]:
        """
        Calculate path geometry from recent price bars.

        Args:
            bars: Deque of (t, o, h, l, c) tuples (at least 3 bars needed)
            sigma: Current volatility (Rogers-Satchell or realized vol)

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
        jerk = gamma - self._prev_gamma

        # Efficiency: displacement / path length over 3 points
        displacement = abs(c2 - c0)
        path_length = abs(c1 - c0) + abs(c2 - c1)
        efficiency = min(1.0, displacement / path_length) if path_length > 0 else 0.0

        # Runway: inverse volatility pressure
        # High sigma → low runway (harder to move through volatility)
        # Formula from C# skeleton: 1.0 / (1.0 + 50.0 * sigma)
        runway = 1.0 / (1.0 + 50.0 * sigma)

        # Rogers-Satchell normalize gamma and jerk
        gamma_z = gamma / (sigma + 1e-12)
        jerk_z = jerk / (sigma + 1e-12)

        # Feasibility: composite score (from C# skeleton weights)
        # 40% efficiency, 30% smooth jerk, 20% runway, 10% base
        smooth_jerk = 1.0 - min(1.0, abs(jerk) / (sigma + 1e-9))
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
