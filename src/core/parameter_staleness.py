"""
Parameter Staleness Detector

GAP 8 FIX: Detects when learned parameters become stale in new market regimes.

The Problem:
- LearnedParameters adapts to current market conditions
- When market regime shifts, old parameters may no longer be valid
- Using stale parameters can lead to poor performance or losses
- Need automated detection to know when to reset/re-learn

Detection Mechanisms:
1. **Performance Degradation**: Win rate / Sharpe declining despite stable parameters
2. **Regime Change**: Sustained shift in market characteristics (volatility, trend)
3. **Parameter Drift**: Parameters changing rapidly (unstable learning)
4. **Confidence Collapse**: Agent confidence dropping without recovery

Interventions:
1. Alert operator of potential staleness
2. Flag parameters for manual review
3. Automatically reset to defaults (with operator approval)
4. Trigger re-learning phase with increased exploration

Usage:
    detector = ParameterStalenessDetector(
        performance_window=500,
        staleness_threshold=0.6
    )

    # Update each bar
    detector.update(
        bar_num=bar_num,
        parameters=learned_params.get_all(),
        performance_metrics={
            "win_rate": current_win_rate,
            "sharpe": current_sharpe,
            "avg_confidence": avg_agent_confidence
        },
        regime=current_regime
    )

    # Check for staleness
    status = detector.check_staleness()
    if status["is_stale"]:
        LOG.warning("PARAMETER STALENESS DETECTED: %s", status)
"""

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from src.utils.safe_math import SafeMath

LOG = logging.getLogger(__name__)

RegimeType = Literal["TRENDING", "MEAN_REVERTING", "TRANSITIONAL", "UNKNOWN"]


@dataclass
class StalenessSnapshot:
    """Snapshot of parameters and performance at a point in time."""

    bar_num: int
    timestamp: str
    parameters: dict[str, float]
    performance: dict[str, float]
    regime: str


@dataclass
class StalenessSignal:
    """Signal indicating potential parameter staleness."""

    signal_type: str  # "performance_decay", "regime_shift", "parameter_drift", "confidence_collapse"
    severity: float  # 0.0 to 1.0
    evidence: dict[str, any]
    recommendation: str


class ParameterStalenessDetector:
    """
    Detects when learned parameters become stale and need reset/re-learning.

    Key Indicators of Staleness:
    1. **Performance Decay**: Win rate drops >10pp, Sharpe drops >0.3
    2. **Regime Shift**: Market regime changes and stays changed for >50 bars
    3. **Parameter Drift**: Parameters changing >20% per 100 bars (unstable)
    4. **Confidence Collapse**: Agent confidence <0.5 for >50 bars

    Staleness Score Calculation:
        staleness = weighted_average([
            performance_decay_score * 0.4,
            regime_shift_score * 0.3,
            parameter_drift_score * 0.2,
            confidence_collapse_score * 0.1
        ])

        is_stale = staleness > staleness_threshold
    """

    def __init__(
        self,
        performance_window: int = 500,
        regime_stability_bars: int = 50,
        parameter_drift_window: int = 100,
        staleness_threshold: float = 0.6,
        persistence_path: Path | None = None,
    ):
        """
        Initialize staleness detector.

        Args:
            performance_window: Number of bars to track for performance metrics
            regime_stability_bars: Bars required to confirm regime shift
            parameter_drift_window: Window for measuring parameter drift
            staleness_threshold: Threshold for declaring parameters stale (0-1)
            persistence_path: Path to save/load detector state
        """
        self.performance_window = performance_window
        self.regime_stability_bars = regime_stability_bars
        self.parameter_drift_window = parameter_drift_window
        self.staleness_threshold = staleness_threshold
        self.persistence_path = persistence_path

        # History tracking
        self.snapshots: deque = deque(maxlen=performance_window)
        self.regime_history: deque = deque(maxlen=regime_stability_bars * 2)

        # Baseline performance (first 500 bars after initialization)
        self.baseline_win_rate: float | None = None
        self.baseline_sharpe: float | None = None
        self.baseline_confidence: float | None = None
        self.baseline_established: bool = False
        self.bars_for_baseline = min(500, performance_window)

        # Current staleness state
        self.staleness_score: float = 0.0
        self.is_stale: bool = False
        self.staleness_signals: list[StalenessSignal] = []

        # State tracking
        self.last_bar_num: int = 0
        self.staleness_start_bar: int | None = None

        LOG.info(
            "ParameterStalenessDetector initialized: window=%d, regime_bars=%d, threshold=%.2f",
            performance_window,
            regime_stability_bars,
            staleness_threshold,
        )

    def update(
        self,
        bar_num: int,
        parameters: dict[str, float],
        performance_metrics: dict[str, float],
        regime: RegimeType,
    ) -> None:
        """
        Update detector with current state.

        Args:
            bar_num: Current bar number
            parameters: Current learned parameters
            performance_metrics: Dict with keys: win_rate, sharpe, avg_confidence
            regime: Current market regime
        """
        # Validate inputs
        required_metrics = ["win_rate", "sharpe", "avg_confidence"]
        for key in required_metrics:
            if key not in performance_metrics:
                LOG.warning("Missing required metric: %s", key)
                return

        # Create snapshot
        snapshot = StalenessSnapshot(
            bar_num=bar_num,
            timestamp=datetime.now(UTC).isoformat(),
            parameters=parameters.copy(),
            performance=performance_metrics.copy(),
            regime=regime,
        )

        self.snapshots.append(snapshot)
        self.regime_history.append(regime)
        self.last_bar_num = bar_num

        # Establish baseline if needed
        if not self.baseline_established and len(self.snapshots) >= self.bars_for_baseline:
            self._establish_baseline()

        # Check for staleness (only after baseline established)
        if self.baseline_established:
            self._detect_staleness()

    def _establish_baseline(self) -> None:
        """
        Establish baseline performance from initial learning period.

        Takes median performance over first N bars as baseline.
        This represents "good" performance to compare against.
        """
        if len(self.snapshots) < self.bars_for_baseline:
            LOG.warning("Insufficient data for baseline: %d bars", len(self.snapshots))
            return

        # Get baseline window (first 500 bars)
        baseline_snapshots = list(self.snapshots)[: self.bars_for_baseline]

        win_rates = [s.performance["win_rate"] for s in baseline_snapshots]
        sharpes = [s.performance["sharpe"] for s in baseline_snapshots]
        confidences = [s.performance["avg_confidence"] for s in baseline_snapshots]

        # Use median to be robust to outliers
        self.baseline_win_rate = SafeMath.safe_percentile(win_rates, 50.0, default=0.5)
        self.baseline_sharpe = SafeMath.safe_percentile(sharpes, 50.0, default=0.0)
        self.baseline_confidence = SafeMath.safe_percentile(confidences, 50.0, default=0.5)

        self.baseline_established = True

        LOG.info(
            "Baseline established: WinRate=%.2f%%, Sharpe=%.2f, Confidence=%.2f",
            self.baseline_win_rate * 100,
            self.baseline_sharpe,
            self.baseline_confidence,
        )

    def _detect_staleness(self) -> None:
        """
        Detect staleness signals and compute overall staleness score.
        """
        if not self.baseline_established or len(self.snapshots) < 50:
            return

        signals = []

        # 1. Performance decay detection
        perf_signal = self._check_performance_decay()
        if perf_signal:
            signals.append(perf_signal)

        # 2. Regime shift detection
        regime_signal = self._check_regime_shift()
        if regime_signal:
            signals.append(regime_signal)

        # 3. Parameter drift detection
        drift_signal = self._check_parameter_drift()
        if drift_signal:
            signals.append(drift_signal)

        # 4. Confidence collapse detection
        confidence_signal = self._check_confidence_collapse()
        if confidence_signal:
            signals.append(confidence_signal)

        self.staleness_signals = signals

        # Compute weighted staleness score
        if signals:
            weights = {
                "performance_decay": 0.4,
                "regime_shift": 0.3,
                "parameter_drift": 0.2,
                "confidence_collapse": 0.1,
            }

            total_score = 0.0
            total_weight = 0.0

            for signal in signals:
                weight = weights.get(signal.signal_type, 0.0)
                total_score += signal.severity * weight
                total_weight += weight

            self.staleness_score = total_score / max(total_weight, 0.001)
        else:
            self.staleness_score = 0.0

        # Update staleness state
        was_stale = self.is_stale
        self.is_stale = self.staleness_score > self.staleness_threshold

        # Track when staleness started
        if self.is_stale and not was_stale:
            self.staleness_start_bar = self.last_bar_num
            LOG.warning(
                "PARAMETER STALENESS DETECTED at bar %d: score=%.2f, signals=%d",
                self.last_bar_num,
                self.staleness_score,
                len(signals),
            )
        elif not self.is_stale and was_stale:
            bars_stale = self.last_bar_num - self.staleness_start_bar if self.staleness_start_bar else 0
            LOG.info("Parameter staleness cleared after %d bars", bars_stale)
            self.staleness_start_bar = None

    def _check_performance_decay(self) -> StalenessSignal | None:
        """
        Check if performance has decayed significantly vs. baseline.

        Returns staleness signal if:
        - Win rate dropped >10 percentage points
        - Sharpe dropped >0.3
        - Both persisted for >50 bars
        """
        if len(self.snapshots) < 50:
            return None

        # Get recent performance (last 50 bars)
        recent = list(self.snapshots)[-50:]
        recent_win_rates = [s.performance["win_rate"] for s in recent]
        recent_sharpes = [s.performance["sharpe"] for s in recent]

        current_win_rate = SafeMath.safe_mean(recent_win_rates, default=0.5)
        current_sharpe = SafeMath.safe_mean(recent_sharpes, default=0.0)

        # Compare to baseline
        win_rate_drop = self.baseline_win_rate - current_win_rate
        sharpe_drop = self.baseline_sharpe - current_sharpe

        # Severity based on magnitude of drop
        severity = 0.0

        if win_rate_drop > 0.10:  # >10pp drop
            severity = max(severity, SafeMath.safe_min([win_rate_drop * 5.0, 1.0], default=1.0))

        if sharpe_drop > 0.3:  # Sharpe drop >0.3
            severity = max(severity, SafeMath.safe_min([sharpe_drop * 2.0, 1.0], default=1.0))

        if severity > 0.3:  # Significant decay
            return StalenessSignal(
                signal_type="performance_decay",
                severity=severity,
                evidence={
                    "baseline_win_rate": self.baseline_win_rate,
                    "current_win_rate": current_win_rate,
                    "win_rate_drop": win_rate_drop,
                    "baseline_sharpe": self.baseline_sharpe,
                    "current_sharpe": current_sharpe,
                    "sharpe_drop": sharpe_drop,
                },
                recommendation="Consider resetting parameters or increasing exploration",
            )

        return None

    def _check_regime_shift(self) -> StalenessSignal | None:
        """
        Check if market regime has shifted and stayed shifted.

        Returns staleness signal if regime has been different from baseline
        for >regime_stability_bars consecutively.
        """
        if len(self.regime_history) < self.regime_stability_bars:
            return None

        # Get baseline regime (most common regime in first 500 bars)
        if len(self.snapshots) < self.bars_for_baseline:
            return None

        baseline_regimes = [s.regime for s in list(self.snapshots)[: self.bars_for_baseline]]
        baseline_regime = max(set(baseline_regimes), key=baseline_regimes.count)

        # Check recent regime stability
        recent_regimes = list(self.regime_history)[-self.regime_stability_bars :]

        # Count how many bars have been in different regime
        different_count = sum(1 for r in recent_regimes if r != baseline_regime)
        shift_fraction = different_count / len(recent_regimes)

        if shift_fraction > 0.8:  # >80% of recent bars in different regime
            return StalenessSignal(
                signal_type="regime_shift",
                severity=shift_fraction,
                evidence={
                    "baseline_regime": baseline_regime,
                    "recent_regimes": recent_regimes[-10:],  # Last 10 for context
                    "shift_fraction": shift_fraction,
                    "bars_since_shift": self.regime_stability_bars,
                },
                recommendation="Parameters learned for different regime may be stale",
            )

        return None

    def _check_parameter_drift(self) -> StalenessSignal | None:
        """
        Check if parameters are changing rapidly (unstable learning).

        Returns staleness signal if parameters are drifting >20% per 100 bars.
        This indicates learning is unstable or chasing noise.
        """
        if len(self.snapshots) < self.parameter_drift_window:
            return None

        # Get parameter snapshots from drift window
        recent = list(self.snapshots)[-self.parameter_drift_window :]
        old_params = recent[0].parameters
        new_params = recent[-1].parameters

        # Calculate drift for each parameter
        drifts = {}
        for key in old_params:
            if key in new_params:
                old_val = old_params[key]
                new_val = new_params[key]

                # Relative change (avoid division by zero)
                if abs(old_val) > 1e-6:
                    drift = abs(new_val - old_val) / abs(old_val)
                else:
                    drift = abs(new_val - old_val)

                drifts[key] = drift

        if not drifts:
            return None

        # Maximum drift across all parameters
        max_drift = max(drifts.values())
        avg_drift = SafeMath.safe_mean(list(drifts.values()), default=0.0)

        # Severity based on drift magnitude
        if max_drift > 0.20:  # >20% change
            severity = SafeMath.safe_min([max_drift * 3.0, 1.0], default=1.0)

            return StalenessSignal(
                signal_type="parameter_drift",
                severity=severity,
                evidence={
                    "window_bars": self.parameter_drift_window,
                    "max_drift": max_drift,
                    "avg_drift": avg_drift,
                    "unstable_parameters": {k: v for k, v in drifts.items() if v > 0.20},
                },
                recommendation="Learning may be unstable or chasing noise",
            )

        return None

    def _check_confidence_collapse(self) -> StalenessSignal | None:
        """
        Check if agent confidence has collapsed and stayed low.

        Returns staleness signal if avg_confidence <0.5 for >50 bars.
        """
        if len(self.snapshots) < 50:
            return None

        recent = list(self.snapshots)[-50:]
        confidences = [s.performance["avg_confidence"] for s in recent]
        avg_confidence = SafeMath.safe_mean(confidences, default=0.5)

        if avg_confidence < 0.5:
            # Count how many bars below threshold
            low_confidence_count = sum(1 for c in confidences if c < 0.5)
            low_fraction = low_confidence_count / len(confidences)

            if low_fraction > 0.7:  # >70% of bars with low confidence
                severity = 1.0 - avg_confidence  # Lower confidence = higher severity

                return StalenessSignal(
                    signal_type="confidence_collapse",
                    severity=severity,
                    evidence={
                        "avg_confidence": avg_confidence,
                        "baseline_confidence": self.baseline_confidence,
                        "low_confidence_fraction": low_fraction,
                        "bars_analyzed": 50,
                    },
                    recommendation="Agent uncertain about decisions, may need re-training",
                )

        return None

    def check_staleness(self) -> dict[str, any]:
        """
        Get current staleness status.

        Returns:
            Dict with staleness information:
            - is_stale: bool
            - staleness_score: float (0-1)
            - signals: List of StalenessSignal
            - bars_stale: int (if stale)
            - recommendations: List of strings
        """
        result = {
            "is_stale": self.is_stale,
            "staleness_score": self.staleness_score,
            "baseline_established": self.baseline_established,
            "snapshots_collected": len(self.snapshots),
        }

        if self.is_stale and self.staleness_start_bar is not None:
            result["bars_stale"] = self.last_bar_num - self.staleness_start_bar

        if self.staleness_signals:
            result["signals"] = [
                {
                    "type": s.signal_type,
                    "severity": s.severity,
                    "evidence": s.evidence,
                    "recommendation": s.recommendation,
                }
                for s in self.staleness_signals
            ]
            result["recommendations"] = [s.recommendation for s in self.staleness_signals]

        return result

    def reset_baseline(self) -> None:
        """
        Reset baseline to current performance.

        Call this after manually resetting parameters or after
        confirming new regime is stable.
        """
        if len(self.snapshots) < 50:
            LOG.warning("Insufficient data to reset baseline: %d bars", len(self.snapshots))
            return

        # Use recent performance as new baseline
        recent = list(self.snapshots)[-50:]
        win_rates = [s.performance["win_rate"] for s in recent]
        sharpes = [s.performance["sharpe"] for s in recent]
        confidences = [s.performance["avg_confidence"] for s in recent]

        self.baseline_win_rate = SafeMath.safe_mean(win_rates, default=0.5)
        self.baseline_sharpe = SafeMath.safe_mean(sharpes, default=0.0)
        self.baseline_confidence = SafeMath.safe_mean(confidences, default=0.5)

        # Clear staleness state
        self.is_stale = False
        self.staleness_score = 0.0
        self.staleness_signals = []
        self.staleness_start_bar = None

        LOG.info(
            "Baseline reset: WinRate=%.2f%%, Sharpe=%.2f, Confidence=%.2f",
            self.baseline_win_rate * 100,
            self.baseline_sharpe,
            self.baseline_confidence,
        )

    def save_state(self, path: Path | None = None) -> None:
        """Save detector state to JSON."""
        path = path or self.persistence_path
        if not path:
            LOG.warning("No persistence path configured")
            return

        state = {
            "baseline_win_rate": self.baseline_win_rate,
            "baseline_sharpe": self.baseline_sharpe,
            "baseline_confidence": self.baseline_confidence,
            "baseline_established": self.baseline_established,
            "staleness_score": self.staleness_score,
            "is_stale": self.is_stale,
            "last_bar_num": self.last_bar_num,
            "staleness_start_bar": self.staleness_start_bar,
            # Don't save full history (too large), just recent stats
            "snapshots_count": len(self.snapshots),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        LOG.info("Staleness detector state saved to %s", path)

    def load_state(self, path: Path | None = None) -> bool:
        """Load detector state from JSON."""
        path = path or self.persistence_path
        if not path or not path.exists():
            LOG.warning("No state file to load: %s", path)
            return False

        try:
            with open(path) as f:
                state = json.load(f)

            self.baseline_win_rate = state.get("baseline_win_rate")
            self.baseline_sharpe = state.get("baseline_sharpe")
            self.baseline_confidence = state.get("baseline_confidence")
            self.baseline_established = state.get("baseline_established", False)
            self.staleness_score = state.get("staleness_score", 0.0)
            self.is_stale = state.get("is_stale", False)
            self.last_bar_num = state.get("last_bar_num", 0)
            self.staleness_start_bar = state.get("staleness_start_bar")

            LOG.info("Staleness detector state loaded from %s", path)
            return True

        except Exception as e:
            LOG.error("Failed to load staleness detector state: %s", e)
            return False
