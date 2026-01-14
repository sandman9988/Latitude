"""
Risk-Aware SAC Manager - Tail Risk Monitoring for Deep RL Trading

This module implements a lightweight pre-processor that wraps RL policies
with sophisticated tail-risk awareness:

1. **Market Diagnostics:**
   - Rolling excess kurtosis → tail heaviness detection
   - VPIN z-score → financial stress/liquidity monitoring

2. **Non-linear Exposure Scaling:**
   - Collapses position size when both signals spike
   - Smooth degradation as risk increases

3. **Tail-Risk Estimation:**
   - Truncated Generalized Pareto Distribution (GPD) hazard
   - Estimates instantaneous probability of ruin

4. **RL Integration:**
   - Returns (exposure, hazard) scalars
   - Exposure multiplies raw agent actions
   - Hazard can be used in reward shaping

References:
- "Risk-Aware Deep Reinforcement Learning in Finance"
- Extreme Value Theory (EVT) for tail modeling
- VPIN: Volume-Synchronized Probability of Informed Trading

Author: Generated from blueprint specification
Date: 2026-01-11
"""

import numpy as np
from typing import Tuple, Optional, Callable
from collections import deque
from scipy.stats import genpareto
import logging

LOG = logging.getLogger(__name__)


class RiskAwareSAC_Manager:
    """
    Monitors market tail-risk metrics and scales RL agent actions accordingly.

    Core Features:
    - Rolling excess kurtosis tracking
    - VPIN-style liquidity stress monitoring
    - Truncated GPD hazard estimation
    - Non-linear exposure collapse under extreme conditions

    Usage:
        manager = RiskAwareSAC_Manager(window=500)

        # Every market tick:
        exposure, hazard = manager.update(return_pct, vpin_metric)

        # Scale agent action:
        scaled_action = raw_action * exposure

        # Optionally use hazard in reward shaping:
        reward -= hazard_penalty_weight * hazard
    """

    def __init__(
        self,
        window: int = 500,
        kurt_max: float = 3.0,
        vpin_trigger: float = 2.0,
        collapse_fac: float = 0.1,
        hazard_shape: float = 0.8,
        hazard_scale: float = 0.5,
        tail_percentile: float = 0.95,
        enable_logging: bool = True,
    ):
        """
        Initialize the Risk-Aware Manager.

        Parameters
        ----------
        window : int
            Rolling window size for all metrics (default: 500 ticks)
        kurt_max : float
            Maximum tolerable excess kurtosis (default: 3.0)
            Normal markets: 0-2, stressed: 3-5, crisis: >5
        vpin_trigger : float
            Z-score threshold for VPIN stress signal (default: 2.0)
        collapse_fac : float
            Minimum exposure during extreme conditions (default: 0.1 = 10%)
        hazard_shape : float
            GPD shape parameter for tail modeling (default: 0.8)
        hazard_scale : float
            GPD scale parameter (default: 0.5)
        tail_percentile : float
            Percentile defining the "tail" (default: 0.95 = top 5%)
        enable_logging : bool
            Enable diagnostic logging (default: True)
        """
        # Configuration
        self.window = window
        self.kurt_max = kurt_max
        self.vpin_trigger = vpin_trigger
        self.collapse_fac = collapse_fac
        self.hazard_shape = hazard_shape
        self.hazard_scale = hazard_scale
        self.tail_percentile = tail_percentile
        self.enable_logging = enable_logging

        # Rolling buffers (using deque for O(1) append/pop)
        self.ret_buf = deque(maxlen=window)
        self.vpin_buf = deque(maxlen=window)

        # Latest computed values
        self.latest_exposure: float = 1.0
        self.latest_hazard: float = 0.0
        self.latest_kurtosis: float = 0.0
        self.latest_vpin_z: float = 0.0

        # Statistics
        self.total_updates: int = 0
        self.collapse_events: int = 0
        self.extreme_hazard_events: int = 0

        if self.enable_logging:
            LOG.info(
                "[RISK_AWARE_SAC] Initialized: window=%d, kurt_max=%.2f, " "vpin_trigger=%.2f, collapse=%.2f",
                window,
                kurt_max,
                vpin_trigger,
                collapse_fac,
            )

    def update(self, new_return: float, new_vpin: float) -> Tuple[float, float]:
        """
        Update all risk metrics with new market data.

        Call this ONCE per market tick after obtaining fresh price data.

        Parameters
        ----------
        new_return : float
            Latest return (price change / previous price)
            Can be log-return or simple return
        new_vpin : float
            Latest VPIN-like liquidity/stress metric
            Higher values = more stressed/illiquid conditions

        Returns
        -------
        exposure : float
            Scaling factor in [collapse_fac, 1.0]
            Multiply raw agent action by this
        hazard : float
            Instantaneous tail-risk probability in [0, 1]
            Can be used in reward shaping or logging
        """
        # Update buffers
        self.ret_buf.append(new_return)
        self.vpin_buf.append(new_vpin)
        self.total_updates += 1

        # Compute diagnostics
        self.latest_kurtosis = self._compute_rolling_kurtosis()
        self.latest_vpin_z = self._compute_vpin_zscore()
        self.latest_hazard = self._compute_gpd_hazard()

        # Compute exposure scaling
        self.latest_exposure = self._compute_exposure()

        # Track extreme events
        if self.latest_exposure <= self.collapse_fac * 1.01:
            self.collapse_events += 1
            if self.enable_logging:
                LOG.warning(
                    "[RISK_AWARE_SAC] EXPOSURE COLLAPSE: kurt=%.3f (max=%.2f), "
                    "vpin_z=%.3f (trigger=%.2f), exposure=%.3f, hazard=%.4f",
                    self.latest_kurtosis,
                    self.kurt_max,
                    self.latest_vpin_z,
                    self.vpin_trigger,
                    self.latest_exposure,
                    self.latest_hazard,
                )

        if self.latest_hazard > 0.1:
            self.extreme_hazard_events += 1

        return self.latest_exposure, self.latest_hazard

    def _compute_rolling_kurtosis(self) -> float:
        """
        Compute excess kurtosis of returns in rolling window.

        Excess kurtosis = kurtosis - 3
        - Normal distribution: 0
        - Fat tails (Laplace, t-dist): > 0
        - Thin tails: < 0

        Returns
        -------
        float
            Excess kurtosis value
        """
        if len(self.ret_buf) < 4:
            return 0.0

        arr = np.array(self.ret_buf, dtype=np.float64)

        # Handle edge cases
        if arr.std() < 1e-12:
            return 0.0

        n = len(arr)
        mean = arr.mean()
        std = arr.std(ddof=0)

        # Fourth moment
        m4 = ((arr - mean) ** 4).mean()

        # Kurtosis
        kurt = m4 / (std**4)

        # Excess kurtosis
        excess_kurt = kurt - 3.0

        return excess_kurt

    def _compute_vpin_zscore(self) -> float:
        """
        Compute z-score of VPIN metric.

        Measures how many standard deviations the current VPIN
        is from its historical mean.

        Returns
        -------
        float
            Z-score of VPIN metric
        """
        if len(self.vpin_buf) < 2:
            return 0.0

        arr = np.array(self.vpin_buf, dtype=np.float64)

        # Use recent window for "current" mean
        recent_window = min(50, len(arr) // 4)
        if recent_window < 2:
            return 0.0

        recent_mean = arr[-recent_window:].mean()

        # Historical statistics
        hist_mean = arr.mean()
        hist_std = arr.std(ddof=0)

        if hist_std < 1e-12:
            return 0.0

        z_score = (recent_mean - hist_mean) / hist_std

        return z_score

    def _compute_gpd_hazard(self) -> float:
        """
        Estimate instantaneous tail-risk probability using
        Truncated Generalized Pareto Distribution (GPD).

        Fits GPD to the top (1 - tail_percentile) of returns
        and computes P(X > threshold).

        Returns
        -------
        float
            Tail-risk probability in [0, 1]
        """
        if len(self.ret_buf) < 20:
            return 0.0

        arr = np.array(self.ret_buf, dtype=np.float64)

        # Define threshold at tail percentile
        threshold = np.percentile(arr, self.tail_percentile * 100)

        # Extract exceedances (values above threshold)
        exceedances = arr[arr > threshold] - threshold

        if len(exceedances) < 3:
            return 0.0

        try:
            # Fit GPD to exceedances
            # genpareto.fit returns (shape, loc, scale)
            shape, loc, scale = genpareto.fit(exceedances, self.hazard_shape)

            # Survival function: P(X > threshold)
            # For GPD fitted to exceedances, we want P(excess > 0)
            tail_prob = genpareto.sf(0, shape, loc, scale)

            # Clip to valid probability range
            hazard = np.clip(tail_prob, 0.0, 1.0)

            return hazard

        except Exception as e:
            if self.enable_logging:
                LOG.debug("[RISK_AWARE_SAC] GPD fit failed: %s", str(e))
            return 0.0

    def _compute_exposure(self) -> float:
        """
        Compute exposure scaling factor using non-linear collapse rule.

        Logic:
        - If BOTH kurtosis > max AND vpin_z > trigger: collapse to minimum
        - Otherwise: smooth degradation based on individual metrics

        Returns
        -------
        float
            Exposure factor in [collapse_fac, 1.0]
        """
        k = self.latest_kurtosis
        z = abs(self.latest_vpin_z)

        # RULE 1: Extreme collapse when BOTH conditions met
        if k > self.kurt_max and z > self.vpin_trigger:
            return self.collapse_fac

        # RULE 2: Smooth degradation
        # Start at 1.0, reduce as kurtosis increases
        kurt_factor = 1.0
        if k > 0:
            # Exponential decay as kurtosis grows
            alpha = 2.0  # Decay rate
            kurt_factor = np.exp(-alpha * max(0, k / self.kurt_max))

        # Additional penalty for VPIN stress
        vpin_penalty = 0.0
        if z > self.vpin_trigger:
            # Linear penalty up to 50% reduction
            excess_z = z - self.vpin_trigger
            vpin_penalty = min(0.5, excess_z * 0.15)

        # Combined factor
        exposure = kurt_factor * (1.0 - vpin_penalty)

        # Ensure minimum exposure
        exposure = max(self.collapse_fac, min(1.0, exposure))

        return exposure

    def scale_action(self, raw_action: float) -> Tuple[float, float]:
        """
        Scale a raw agent action by current exposure factor.

        Parameters
        ----------
        raw_action : float
            Raw action from RL agent (e.g., log position size)

        Returns
        -------
        scaled_action : float
            Action scaled by exposure factor
        hazard : float
            Current tail-risk hazard (for logging/reward shaping)
        """
        scaled = raw_action * self.latest_exposure
        return scaled, self.latest_hazard

    def get_diagnostics(self) -> dict:
        """
        Get current diagnostic values.

        Returns
        -------
        dict
            Current state of all metrics
        """
        return {
            "exposure": self.latest_exposure,
            "hazard": self.latest_hazard,
            "kurtosis": self.latest_kurtosis,
            "vpin_z": self.latest_vpin_z,
            "buffer_size": len(self.ret_buf),
            "total_updates": self.total_updates,
            "collapse_events": self.collapse_events,
            "extreme_hazard_events": self.extreme_hazard_events,
            "collapse_rate": self.collapse_events / max(1, self.total_updates),
        }

    def reset(self):
        """Reset all buffers and statistics."""
        self.ret_buf.clear()
        self.vpin_buf.clear()
        self.latest_exposure = 1.0
        self.latest_hazard = 0.0
        self.latest_kurtosis = 0.0
        self.latest_vpin_z = 0.0
        self.total_updates = 0
        self.collapse_events = 0
        self.extreme_hazard_events = 0

        if self.enable_logging:
            LOG.info("[RISK_AWARE_SAC] Reset all buffers and statistics")


# Standalone helper functions (for direct use without class)


def rolling_kurtosis(arr: np.ndarray, window: int) -> float:
    """
    Compute excess kurtosis of last `window` elements.

    Parameters
    ----------
    arr : np.ndarray
        Time series data
    window : int
        Rolling window size

    Returns
    -------
    float
        Excess kurtosis
    """
    x = arr[-window:]
    if len(x) < 4:
        return 0.0

    mean = x.mean()
    var = x.var(ddof=0)

    if var < 1e-12:
        return 0.0

    m4 = ((x - mean) ** 4).mean()
    kurt = m4 / (var**2)

    return kurt - 3.0


def vpin_zscore(vpin_arr: np.ndarray, window: int) -> float:
    """
    Compute z-score of VPIN metric.

    Parameters
    ----------
    vpin_arr : np.ndarray
        VPIN time series
    window : int
        Rolling window size

    Returns
    -------
    float
        Z-score
    """
    arr = vpin_arr[-window:]
    if len(arr) < 2:
        return 0.0

    recent_mean = arr[-min(50, len(arr) // 4) :].mean()
    hist_mean = arr.mean()
    hist_std = arr.std(ddof=0)

    if hist_std < 1e-12:
        return 0.0

    return (recent_mean - hist_mean) / hist_std


def truncated_gpd_hazard(arr: np.ndarray, shape_par: float = 0.8, tail_percentile: float = 0.95) -> float:
    """
    Estimate tail-risk hazard using GPD.

    Parameters
    ----------
    arr : np.ndarray
        Return time series
    shape_par : float
        GPD shape parameter
    tail_percentile : float
        Percentile defining tail (0.95 = top 5%)

    Returns
    -------
    float
        Tail probability in [0, 1]
    """
    if len(arr) < 10:
        return 0.0

    threshold = np.percentile(arr, tail_percentile * 100)
    exceedances = arr[arr > threshold] - threshold

    if len(exceedances) < 3:
        return 0.0

    try:
        shape, loc, scale = genpareto.fit(exceedances, shape_par)
        tail_prob = genpareto.sf(0, shape, loc, scale)
        return np.clip(tail_prob, 0.0, 1.0)
    except:
        return 0.0
