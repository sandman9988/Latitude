#!/usr/bin/env python3
"""
Phase 3.4: Regime Detection via DSP-Based Damping Ratio
=========================================================

Implements Section 6.2 of the handbook: Regime classification using
digital signal processing to detect trending vs mean-reverting markets.

Theory:
-------
Damping ratio (ζ) characterizes oscillator behavior:
- ζ < 0.7: Underdamped → Trending (momentum persists)
- ζ > 1.3: Overdamped → Mean-reverting (quick decay)
- 0.7 ≤ ζ ≤ 1.3: Critically damped → Transitional

We estimate ζ from price autocorrelation decay rate using a
second-order discrete-time system model.

Performance Optimization:
- Rolling window (50 bars) to avoid recalculating entire history
- Numpy vectorization for autocorrelation
- Cached regime state (only recalculate every N bars)
"""

import logging
from collections import deque
from typing import Literal

import numpy as np

DEFAULT_WINDOW_SIZE: int = 50
DEFAULT_UPDATE_INTERVAL: int = 5
TRENDING_THRESHOLD: float = 0.7
MEAN_REVERTING_THRESHOLD: float = 1.3
NEUTRAL_ZETA: float = 1.0
MIN_RETURNS_REQUIRED: int = 10
MIN_RETURNS_TWO_PERIOD: int = 2
VARIANCE_EPSILON: float = 1e-12
VR_CLAMP_MIN: float = 0.1
VR_CLAMP_MAX: float = 5.0
ZETA_MAP_MULTIPLIER: float = 2.0
ZETA_CLAMP_MIN: float = 0.1
ZETA_CLAMP_MAX: float = 2.0
RUNWAY_MULT_TRENDING: float = 1.3
RUNWAY_MULT_MEAN_REVERTING: float = 0.7
RUNWAY_MULT_NEUTRAL: float = 1.0
# Regime threshold adjustment — expressed as a FRACTIONAL scale, not an
# absolute price delta. The trigger agent applies:
#   adjusted_threshold = base_threshold * (1 + regime_adj)
# This makes the regime detector instrument-agnostic: the same ±15% signal
# works for XAUUSD ($5000), EURUSD (1.08), BTC ($90k), etc.
# Previously this was -0.0002 / +0.0003 (raw price) which only worked for
# EURUSD-scale instruments.
REGIME_ADJ_TRENDING: float = -0.15   # 15% easier to trigger in trending regime
REGIME_ADJ_MEAN_REVERTING: float = 0.15  # 15% harder to trigger in choppy regime
REGIME_ADJ_NEUTRAL: float = 0.0

LOG = logging.getLogger(__name__)

RegimeType = Literal["TRENDING", "MEAN_REVERTING", "TRANSITIONAL", "UNKNOWN"]


class RegimeDetector:
    """
    Detect market regime using DSP-based damping ratio analysis.

    Optimized for real-time trading:
    - Rolling window (50 bars)
    - Cached regime (update every 5 bars)
    - Vectorized computation with numpy
    - Instrument-adaptive thresholds
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        update_interval: int = DEFAULT_UPDATE_INTERVAL,
        instrument_volatility: float = 1.0,  # Relative vol multiplier for threshold adjustment
    ):
        """
        Args:
            window_size: Number of bars for regime calculation (default 50)
            update_interval: Recalculate regime every N bars (default 5)
        """
        self.window_size = window_size
        self.update_interval = update_interval

        # Price history (rolling window)
        self.price_buffer: deque[float] = deque(maxlen=window_size)

        # Cached regime state
        self.current_regime: RegimeType = "UNKNOWN"
        self.current_zeta = NEUTRAL_ZETA  # Damping ratio
        self.current_vr: float = 1.0  # Variance ratio (1.0 = random walk)
        self.bars_since_update = 0
        self.total_updates: int = 0  # How many times regime has been recalculated

        # Performance cache
        self._cache_invalidated: bool = True
        self._cached_var_1: float | None = None
        self._cached_returns: np.ndarray | None = None

        # Regime thresholds (from handbook Section 6.2.3)
        # Scale by instrument characteristics: high-vol assets need wider bands
        vol_scale = max(0.5, min(2.0, instrument_volatility))
        self.trending_threshold = TRENDING_THRESHOLD * vol_scale  # ζ < 0.7 → trending
        self.mean_reverting_threshold = MEAN_REVERTING_THRESHOLD * vol_scale  # ζ > 1.3 → mean-reverting
        self.instrument_volatility = vol_scale

        LOG.info(
            "[REGIME] Initialized: window=%d, update_every=%d bars",
            window_size,
            update_interval,
        )

    def add_price(self, price: float) -> tuple[RegimeType, float]:
        """
        Add new price and update regime detection.

        Args:
            price: Current close price

        Returns:
            (regime, damping_ratio) tuple
        """
        # Defensive: Validate price
        if price is None or price <= 0:
            LOG.warning("[REGIME] Invalid price: %s. Skipping.", price)
            return self.current_regime, self.current_zeta

        # Add to rolling buffer
        self.price_buffer.append(price)
        # deque(maxlen=...) auto-evicts oldest

        # Performance: Invalidate cache
        self._cache_invalidated = True  # attribute defined in __init__

        self.bars_since_update += 1

        # Performance optimization: Only recalculate every N bars
        if self.bars_since_update >= self.update_interval and len(self.price_buffer) >= self.window_size:
            self._update_regime()
            self.bars_since_update = 0

        return self.current_regime, self.current_zeta

    def _update_regime(self):
        """Calculate damping ratio and classify regime using variance ratio test."""
        try:
            returns = self._compute_validated_returns()
            if returns is None:
                return

            var_1 = self._get_or_compute_variance(returns)
            if var_1 is None or var_1 < VARIANCE_EPSILON:
                self.current_zeta = NEUTRAL_ZETA
                self.current_regime = "TRANSITIONAL"
                return

            vr = self._compute_variance_ratio(returns, var_1)
            if vr is None:
                return

            self.current_vr = vr
            self.total_updates += 1
            self._classify_regime(vr, var_1)

        except (ValueError, ArithmeticError, RuntimeError) as e:
            LOG.error("[REGIME] Calculation error: %s", e, exc_info=True)
            self.current_regime = "TRANSITIONAL"
            self.current_zeta = NEUTRAL_ZETA

    def _compute_validated_returns(self) -> np.ndarray | None:
        """Compute and validate log returns from price buffer."""
        prices = np.array(self.price_buffer, dtype=np.float64)

        if not np.all(np.isfinite(prices)):
            LOG.warning("[REGIME] Non-finite prices detected, skipping update")
            return None
        if not np.all(prices > 0):
            LOG.warning("[REGIME] Non-positive prices detected, skipping update")
            return None

        try:
            log_prices = np.log(prices)
            returns = np.diff(log_prices)
        except (ValueError, RuntimeWarning) as e:
            LOG.warning("[REGIME] Log calculation failed: %s", e)
            return None

        if not np.all(np.isfinite(returns)):
            LOG.warning("[REGIME] Non-finite returns detected, skipping update")
            return None

        if len(returns) < MIN_RETURNS_REQUIRED:
            LOG.warning("[REGIME] Insufficient data: %d returns", len(returns))
            return None

        return returns

    def _get_or_compute_variance(self, returns: np.ndarray) -> float | None:
        """Get cached variance or compute it."""
        if self._cache_invalidated or self._cached_var_1 is None:
            var_1 = float(np.var(returns))
            self._cached_var_1 = var_1
            self._cached_returns = returns
            self._cache_invalidated = False
        else:
            var_1 = self._cached_var_1
        return var_1

    def _compute_variance_ratio(self, returns: np.ndarray, var_1: float) -> float | None:
        """Compute and validate the variance ratio."""
        returns_2 = returns[:-1] + returns[1:]

        if not np.all(np.isfinite(returns_2)):
            LOG.warning("[REGIME] Non-finite 2-period returns, skipping")
            return None

        var_2 = np.var(returns_2)

        if not np.isfinite(var_2) or var_2 < 0:
            LOG.warning("[REGIME] Invalid var_2: %s", var_2)
            return None

        vr = var_2 / (2.0 * var_1)

        if not np.isfinite(vr):
            LOG.warning("[REGIME] Non-finite variance ratio")
            return None

        return float(max(VR_CLAMP_MIN, min(VR_CLAMP_MAX, vr)))

    def _classify_regime(self, vr: float, var_1: float):
        """Map variance ratio to damping ratio and classify regime."""
        self.current_zeta = NEUTRAL_ZETA + ZETA_MAP_MULTIPLIER * (1.0 - vr)
        self.current_zeta = max(ZETA_CLAMP_MIN, min(ZETA_CLAMP_MAX, self.current_zeta))

        if self.current_zeta < self.trending_threshold:
            self.current_regime = "TRENDING"
        elif self.current_zeta > self.mean_reverting_threshold:
            self.current_regime = "MEAN_REVERTING"
        else:
            self.current_regime = "TRANSITIONAL"

        var_2 = np.var(np.array(self.price_buffer, dtype=np.float64))
        LOG.info(
            "[REGIME] Updated: ζ=%.3f | %s | VR(2)=%.3f var_1=%.6f var_2=%.6f",
            self.current_zeta,
            self.current_regime,
            vr,
            var_1,
            var_2,
        )

    def _autocorrelation(self, x: np.ndarray, lag: int) -> float:
        """
        Calculate autocorrelation at given lag using proper normalization.

        Optimized: Uses numpy for vectorized operations.
        """
        # Defensive: Handle edge cases
        if len(x) <= lag:
            return 0.0

        # Demean the series
        x_mean = np.mean(x)
        x_centered = x - x_mean

        # Calculate variance of full series
        variance = np.var(x_centered)
        if variance < VARIANCE_EPSILON:
            return 0.0  # No variance → no correlation

        # Calculate covariance at lag
        # Cov(X_t, X_{t-lag}) = E[(X_t - μ)(X_{t-lag} - μ)]
        x1 = x_centered[lag:]  # t
        x2 = x_centered[:-lag]  # t-lag

        covariance = np.mean(x1 * x2)

        autocorr = covariance / variance

        # Defensive: Clamp to valid correlation range
        return float(max(-1.0, min(1.0, autocorr)))

    def get_regime_multiplier(self) -> float:
        """
        Get runway multiplier based on current regime.

        Returns:
            float: Multiplier for predicted_runway
            - TRENDING: 1.3x (expect larger moves)
            - MEAN_REVERTING: 0.7x (expect smaller moves)
            - TRANSITIONAL/UNKNOWN: 1.0x (neutral)
        """
        if self.current_regime == "TRENDING":
            return RUNWAY_MULT_TRENDING
        elif self.current_regime == "MEAN_REVERTING":
            return RUNWAY_MULT_MEAN_REVERTING
        else:
            return RUNWAY_MULT_NEUTRAL

    def get_trigger_threshold_adjustment(self) -> float:
        """
        Get trigger threshold adjustment based on regime.

        Returns a FRACTIONAL scale factor. The caller should apply it as:
            adjusted_threshold = base_threshold * (1 + regime_adj)

        This is instrument-agnostic: the same fraction works regardless of
        whether the base threshold is 0.0003 (EURUSD) or 3.0 (XAUUSD).

        Returns:
            float: Fractional adjustment in range [-0.15, +0.15]
            - TRENDING:       -0.15 (15% easier to trigger — ride momentum)
            - MEAN_REVERTING: +0.15 (15% harder to trigger — avoid whipsaws)
            - TRANSITIONAL/UNKNOWN: 0.0 (neutral)
        """
        if self.current_regime == "TRENDING":
            return REGIME_ADJ_TRENDING
        elif self.current_regime == "MEAN_REVERTING":
            return REGIME_ADJ_MEAN_REVERTING
        else:
            return REGIME_ADJ_NEUTRAL

    def update_adj_scale(self, scale: float):
        """
        Allow an external learner (e.g. the trigger agent) to widen or narrow
        the regime adjustment magnitude based on observed benefit. The scale
        is clamped to [0.0, 0.50] to prevent degenerate gating.

        In practice this is driven by LearnedParametersManager via the
        `regime_adj_scale` parameter which the trigger agent updates after
        each trade outcome. The RegimeDetector reads it on the next call to
        get_trigger_threshold_adjustment().
        """
        global REGIME_ADJ_TRENDING, REGIME_ADJ_MEAN_REVERTING  # noqa: PLW0603
        scale = max(0.0, min(0.50, abs(scale)))
        REGIME_ADJ_TRENDING = -scale
        REGIME_ADJ_MEAN_REVERTING = scale

    def get_regime_info(self) -> dict:
        """Get current regime information for logging/debugging."""
        return {
            "regime": self.current_regime,
            "damping_ratio": self.current_zeta,
            "runway_multiplier": self.get_regime_multiplier(),
            "trigger_adjustment": self.get_trigger_threshold_adjustment(),
            "trigger_adj_note": "fractional scale (multiply base_threshold by (1+adj))",
            "buffer_size": len(self.price_buffer),
            "bars_since_update": self.bars_since_update,
        }


# ----------------------------
# Self-test
# ----------------------------
def _test_regime_detector():  # noqa: PLR0915
    """Test regime detector with synthetic data."""
    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║     Phase 3.4: Regime Detector Self-Test                 ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    detector = RegimeDetector(window_size=50, update_interval=5)
    rng = np.random.default_rng(42)

    # Test 1: Trending market — returns with positive autocorrelation
    #   AR(1) returns with positive coefficient → VR > 1 → TRENDING
    print("Test 1: TRENDING market (momentum in returns)")
    print("-" * 60)
    price = 100000.0
    prices_list = []
    prev_ret = 0.0
    for _i in range(60):
        ret = 0.6 * prev_ret + rng.normal(0, 10.0)  # persistent returns
        price += ret
        price = max(price, 1.0)
        prices_list.append(price)
        _, _ = detector.add_price(price)
        prev_ret = ret

    info = detector.get_regime_info()
    print(f"Regime: {info['regime']}")
    print(f"Damping ratio (ζ): {info['damping_ratio']:.3f}")
    print(f"Runway multiplier: {info['runway_multiplier']:.2f}x")
    print(f"Trigger adjustment: {info['trigger_adjustment']:+.4f}")
    print()

    # Test 2: Mean-reverting market — returns with negative autocorrelation
    #   AR(1) returns with negative coefficient → VR < 1 → MEAN_REVERTING
    print("Test 2: MEAN-REVERTING market (choppy returns)")
    print("-" * 60)
    detector2 = RegimeDetector(window_size=50, update_interval=5)
    base_price = 100000.0
    prev_ret = 0.0
    for _i in range(60):
        ret = -0.6 * prev_ret + rng.normal(0, 10.0)  # reverting returns
        base_price += ret
        base_price = max(base_price, 1.0)
        _, _ = detector2.add_price(base_price)
        prev_ret = ret

    info2 = detector2.get_regime_info()
    print(f"Regime: {info2['regime']}")
    print(f"Damping ratio (ζ): {info2['damping_ratio']:.3f}")
    print(f"Runway multiplier: {info2['runway_multiplier']:.2f}x")
    print(f"Trigger adjustment: {info2['trigger_adjustment']:+.4f}")
    print()

    # Test 3: Transitional market (random walk)
    print("Test 3: TRANSITIONAL market (random walk)")
    print("-" * 60)
    detector3 = RegimeDetector(window_size=50, update_interval=5)
    base_price = 100000.0
    for _i in range(60):
        # Random walk (no autocorrelation)
        noise = rng.normal(0, 10.0)
        base_price += noise
        _, _ = detector3.add_price(base_price)

    info3 = detector3.get_regime_info()
    print(f"Regime: {info3['regime']}")
    print(f"Damping ratio (ζ): {info3['damping_ratio']:.3f}")
    print(f"Runway multiplier: {info3['runway_multiplier']:.2f}x")
    print(f"Trigger adjustment: {info3['trigger_adjustment']:+.4f}")
    print()

    # Summary
    print("=" * 60)
    print("✅ Regime detection tests complete!")
    print()
    print("Expected behavior:")
    print("  • TRENDING: ζ < 0.7, runway 1.3x, trigger adj -0.15 (fractional: 15% easier)")
    print("  • MEAN_REVERTING: ζ > 1.3, runway 0.7x, trigger adj +0.15 (fractional: 15% harder)")
    print("  • TRANSITIONAL: 0.7 ≤ ζ ≤ 1.3, runway 1.0x, trigger adj 0.0")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _test_regime_detector()
