#!/usr/bin/env python3
"""
HMM-Based Regime Detector
==========================

Augments the DSP variance-ratio RegimeDetector with a 3-state Gaussian HMM
fitted on log-returns.  The key advantage over the discrete VR approach is
**smooth probabilistic transitions** — the runway multiplier blends
continuously between regime states instead of jumping discontinuously.

States
------
The HMM discovers 3 latent states from data.  After fitting, states are
labelled by their emission mean:
    - Highest |μ| or lowest σ → Trending (momentum persists)
    - Highest σ with near-zero μ → Mean-Reverting / Volatile
    - Lowest |μ| and moderate σ → Low-Activity / Neutral

Integration
-----------
``HMMRegimeDetector`` wraps the existing ``RegimeDetector`` and exposes:
    - ``get_regime_probabilities()`` → array of 3 posterior probabilities
    - ``get_blended_runway_multiplier()`` → continuous multiplier ∈ [0.7, 1.3]
    - Existing ``get_regime_multiplier()`` still works (from base class)

The caller (DualPolicy) can switch between discrete and blended multipliers
without changing the rest of the pipeline.

Dependencies: hmmlearn (pip install hmmlearn)
"""

import logging
from collections import deque

import numpy as np

from src.features.regime_detector import (
    DEFAULT_UPDATE_INTERVAL,
    DEFAULT_WINDOW_SIZE,
    RUNWAY_MULT_MEAN_REVERTING,
    RUNWAY_MULT_NEUTRAL,
    RUNWAY_MULT_TRENDING,
    RegimeDetector,
)

LOG = logging.getLogger(__name__)

# ── HMM configuration ────────────────────────────────────────────────────────
HMM_N_STATES: int = 3
HMM_MIN_OBSERVATIONS: int = 30          # Minimum returns before first HMM fit
HMM_REFIT_INTERVAL: int = 50            # Re-fit every N new observations
HMM_MAX_HISTORY: int = 500              # Cap rolling window to prevent slow fits
HMM_FIT_ITERATIONS: int = 20            # EM iterations (fast convergence is fine)
HMM_COVARIANCE_TYPE: str = "full"       # "full" works well with 1D observations


class HMMRegimeDetector(RegimeDetector):
    """
    Gaussian HMM regime detector with smooth probability blending.

    Inherits from RegimeDetector so the existing VR-based regime is always
    available as a fallback.  The HMM layer adds posterior state probabilities
    for continuous runway multiplier blending.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        update_interval: int = DEFAULT_UPDATE_INTERVAL,
        instrument_volatility: float = 1.0,
    ):
        super().__init__(
            window_size=window_size,
            update_interval=update_interval,
            instrument_volatility=instrument_volatility,
        )

        # HMM-specific state
        self._hmm_returns: deque[float] = deque(maxlen=HMM_MAX_HISTORY)
        self._hmm_model = None
        self._hmm_fitted: bool = False
        self._hmm_obs_since_fit: int = 0
        self._state_labels: np.ndarray = np.array([0, 1, 2])  # Will be remapped after fit
        self._state_probs: np.ndarray = np.array([0.0, 0.0, 1.0])  # Start as "unknown" → neutral

        # Multiplier map (indexed by labelled state: 0=trending, 1=mean_rev, 2=neutral)
        self._multiplier_map: np.ndarray = np.array([
            RUNWAY_MULT_TRENDING,       # State 0: Trending
            RUNWAY_MULT_MEAN_REVERTING,  # State 1: Mean-reverting
            RUNWAY_MULT_NEUTRAL,         # State 2: Neutral / Low-activity
        ], dtype=np.float64)

        self._prev_price: float | None = None
        LOG.info("[HMM_REGIME] Initialized: n_states=%d, min_obs=%d, refit_every=%d",
                 HMM_N_STATES, HMM_MIN_OBSERVATIONS, HMM_REFIT_INTERVAL)

    def add_price(self, price: float) -> tuple:
        """Add price, update both VR regime and HMM posteriors."""
        # Base class handles VR regime
        result = super().add_price(price)

        # Accumulate log-returns for HMM
        if self._prev_price is not None and self._prev_price > 0 and price > 0:
            log_ret = np.log(price / self._prev_price)
            if np.isfinite(log_ret):
                self._hmm_returns.append(log_ret)
                self._hmm_obs_since_fit += 1
                self._maybe_update_hmm()
        self._prev_price = price

        return result

    def _maybe_update_hmm(self) -> None:
        """Fit or update HMM when enough new observations have accumulated."""
        n = len(self._hmm_returns)
        if n < HMM_MIN_OBSERVATIONS:
            return

        # Only refit periodically (EM is O(n·k²) per iteration)
        if self._hmm_fitted and self._hmm_obs_since_fit < HMM_REFIT_INTERVAL:
            # Still update posteriors with existing model
            self._update_posteriors()
            return

        self._fit_hmm()
        self._hmm_obs_since_fit = 0

    def _fit_hmm(self) -> None:
        """Fit a GaussianHMM to the accumulated returns."""
        try:
            from hmmlearn.hmm import GaussianHMM  # noqa: PLC0415
        except ImportError:
            LOG.warning("[HMM_REGIME] hmmlearn not available — using VR regime only")
            return

        returns = np.array(self._hmm_returns, dtype=np.float64).reshape(-1, 1)

        try:
            model = GaussianHMM(
                n_components=HMM_N_STATES,
                covariance_type=HMM_COVARIANCE_TYPE,
                n_iter=HMM_FIT_ITERATIONS,
                random_state=42,
            )
            model.fit(returns)
            self._hmm_model = model
            self._hmm_fitted = True

            # Label states by behaviour
            self._label_states()
            self._update_posteriors()

            LOG.debug(
                "[HMM_REGIME] Fitted: means=%s, vars=%s, labels=%s",
                model.means_.flatten().round(6),
                model.covars_.flatten().round(8),
                self._state_labels,
            )
        except Exception as exc:
            LOG.warning("[HMM_REGIME] Fit failed: %s — keeping previous model", exc)

    def _label_states(self) -> None:
        """
        Assign semantic labels to HMM states based on emission parameters.

        Strategy:
        - State with highest abs(mean) → Trending (momentum signal)
        - State with highest variance and small mean → Mean-reverting (volatile, no direction)
        - Remaining → Neutral / Low-activity
        """
        model = self._hmm_model
        if model is None:
            return

        means = model.means_.flatten()
        variances = model.covars_.flatten()  # For "full" 1D, shape is (n, 1, 1)
        if variances.ndim > 1:
            variances = np.array([model.covars_[i][0, 0] for i in range(HMM_N_STATES)])

        abs_means = np.abs(means)

        # Labels: 0=trending, 1=mean_rev, 2=neutral
        labels = np.full(HMM_N_STATES, 2, dtype=int)  # Default all to neutral

        # Trending: highest |mean| — momentum persists
        trending_idx = int(np.argmax(abs_means))
        labels[trending_idx] = 0

        # Mean-reverting: highest variance among remaining states
        remaining = [i for i in range(HMM_N_STATES) if i != trending_idx]
        if len(remaining) >= 2:
            var_of_remaining = [variances[i] for i in remaining]
            mean_rev_idx = remaining[int(np.argmax(var_of_remaining))]
            labels[mean_rev_idx] = 1
            # Last one stays neutral (label=2)
        elif len(remaining) == 1:
            labels[remaining[0]] = 1  # Only one left — call it mean-reverting

        self._state_labels = labels

    def _update_posteriors(self) -> None:
        """Compute posterior state probabilities from the fitted model."""
        model = self._hmm_model
        if model is None or len(self._hmm_returns) < HMM_MIN_OBSERVATIONS:
            return

        returns = np.array(self._hmm_returns, dtype=np.float64).reshape(-1, 1)
        try:
            posteriors = model.predict_proba(returns)
            # Take the last row (current bar's posterior)
            raw_probs = posteriors[-1]

            # Remap to semantic order: [p_trending, p_mean_rev, p_neutral]
            semantic_probs = np.zeros(HMM_N_STATES)
            for raw_idx in range(HMM_N_STATES):
                semantic_idx = self._state_labels[raw_idx]
                semantic_probs[semantic_idx] += raw_probs[raw_idx]

            self._state_probs = semantic_probs
        except Exception as exc:
            LOG.debug("[HMM_REGIME] Posterior update failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_regime_probabilities(self) -> np.ndarray:
        """
        Get posterior probabilities for each regime state.

        Returns:
            Array of shape (3,): [p_trending, p_mean_reverting, p_neutral]
            Sums to 1.0.
        """
        return self._state_probs.copy()

    def get_blended_runway_multiplier(self) -> float:
        """
        Compute a continuous runway multiplier from posterior state probabilities.

        Instead of a hard switch between 0.7/1.0/1.3, this blends:
            multiplier = Σ(p_state × mult_state)

        Returns:
            float in range [RUNWAY_MULT_MEAN_REVERTING, RUNWAY_MULT_TRENDING]
                  i.e. typically [0.7, 1.3]
        """
        if not self._hmm_fitted:
            # Fall back to discrete VR multiplier until HMM is ready
            return self.get_regime_multiplier()

        return float(np.dot(self._state_probs, self._multiplier_map))

    def get_regime_info(self) -> dict:
        """Extended regime info including HMM posterior probabilities."""
        info = super().get_regime_info()
        info["hmm_fitted"] = self._hmm_fitted
        info["hmm_observations"] = len(self._hmm_returns)
        if self._hmm_fitted:
            info["hmm_probs"] = {
                "trending": float(self._state_probs[0]),
                "mean_reverting": float(self._state_probs[1]),
                "neutral": float(self._state_probs[2]),
            }
            info["hmm_blended_multiplier"] = self.get_blended_runway_multiplier()
        return info
