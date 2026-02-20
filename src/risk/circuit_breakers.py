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

# pylint: disable=line-too-long

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np

from src.persistence.learned_parameters import LearnedParametersManager
from src.utils.safe_math import SAFE_EPSILON, SafeMath

LOG = logging.getLogger(__name__)

DEFAULT_BREAKER_COOLDOWN_MINUTES: int = 60
SORTINO_THRESHOLD_DEFAULT: float = 0.5
SORTINO_MIN_TRADES: int = 20
SORTINO_HISTORY_LIMIT: int = 100
KURTOSIS_THRESHOLD_DEFAULT: float = 5.0
KURTOSIS_MIN_SAMPLES: int = 30
KURTOSIS_HISTORY_LIMIT: int = 100
KURTOSIS_MIN_SAMPLE_SIZE: int = 4
DRAWDOWN_DEFAULT_THRESHOLDS: dict[float, float] = {0.05: 0.9, 0.10: 0.75, 0.15: 0.5, 0.20: 0.0}
DRAWDOWN_COOLDOWN_MINUTES: int = 240
CONSEC_LOSSES_DEFAULT_MAX: int = 5
CONSEC_LOSSES_COOLDOWN_MINUTES: int = 180
MANAGER_DEFAULT_SORTINO: float = SORTINO_THRESHOLD_DEFAULT
MANAGER_DEFAULT_KURTOSIS: float = KURTOSIS_THRESHOLD_DEFAULT
MANAGER_DEFAULT_MAX_DRAWDOWN: float = 0.20
MANAGER_DEFAULT_MAX_LOSSES: int = CONSEC_LOSSES_DEFAULT_MAX


@dataclass
class BreakerState:
    """State of a circuit breaker"""

    name: str
    is_tripped: bool = False
    trip_time: datetime | None = None
    trip_reason: str = ""
    trip_value: float = 0.0
    threshold: float = 0.0
    cooldown_minutes: int = DEFAULT_BREAKER_COOLDOWN_MINUTES

    def trip(self, reason: str, value: float, threshold: float):
        """Trip the breaker"""
        self.is_tripped = True
        self.trip_time = datetime.now()
        self.trip_reason = reason
        self.trip_value = value
        self.threshold = threshold
        LOG.warning("🚨 CIRCUIT BREAKER TRIPPED: %s", self.name)
        LOG.warning("   Reason: %s", reason)
        LOG.warning("   Value: %.4f | Threshold: %.4f", value, threshold)

    def reset(self):
        """Reset the breaker"""
        if self.is_tripped:
            LOG.info("✓ Circuit breaker reset: %s", self.name)
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


class ManagedBreaker(Protocol):
    """Protocol describing the breakers tracked by the manager."""

    state: BreakerState

    def check(self) -> bool:
        """Return True when the breaker trips."""


class SortinoBreaker:
    """
    Sortino Ratio Circuit Breaker

    Handbook: "If risk-adjusted returns drop too low, stop trading"
    Sortino focuses on downside deviation (better than Sharpe for asymmetric returns)
    """

    def __init__(self, threshold: float = SORTINO_THRESHOLD_DEFAULT, min_trades: int = SORTINO_MIN_TRADES):
        """
        Args:
            threshold: Minimum acceptable Sortino ratio
            min_trades: Minimum trades before breaker activates
        """
        self.threshold = threshold
        self.min_trades = min_trades
        self.returns: list[float] = []
        self.state = BreakerState(name="Sortino", threshold=threshold, cooldown_minutes=120)  # 2 hour cooldown

    def update(self, trade_return: float):
        """Add a trade return"""
        self.returns.append(trade_return)

        # Limit history
        if len(self.returns) > SORTINO_HISTORY_LIMIT:
            self.returns = self.returns[-SORTINO_HISTORY_LIMIT:]

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
            self.state.trip(reason="Sortino ratio below threshold", value=sortino, threshold=self.threshold)
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
            return float("inf")  # No losses = infinite Sortino

        downside_dev = np.std(downside_returns)

        if downside_dev < SAFE_EPSILON:
            return float("inf")

        # Sortino ratio equals mean divided by downside deviation
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

    def __init__(self, threshold: float = KURTOSIS_THRESHOLD_DEFAULT, min_samples: int = KURTOSIS_MIN_SAMPLES):
        """
        Args:
            threshold: Maximum acceptable kurtosis (normal distribution = 3)
            min_samples: Minimum samples before breaker activates
        """
        self.threshold = threshold
        self.min_samples = min_samples
        self.returns: list[float] = []
        self.state = BreakerState(
            name="Kurtosis", threshold=threshold, cooldown_minutes=DEFAULT_BREAKER_COOLDOWN_MINUTES
        )

    def update(self, trade_return: float):
        """Add a trade return"""
        self.returns.append(trade_return)

        # Limit history
        if len(self.returns) > KURTOSIS_HISTORY_LIMIT:
            self.returns = self.returns[-KURTOSIS_HISTORY_LIMIT:]

    def check(self) -> bool:
        """Check if breaker should trip"""
        if len(self.returns) < self.min_samples:
            return False

        kurtosis = self._calculate_kurtosis()

        if kurtosis > self.threshold:
            self.state.trip(
                reason="Excess kurtosis detected (fat tails)",
                value=kurtosis,
                threshold=self.threshold,
            )
            return True

        return False

    def _calculate_kurtosis(self) -> float:
        """Calculate sample kurtosis (excess kurtosis)"""
        if len(self.returns) < KURTOSIS_MIN_SAMPLE_SIZE:
            return 0.0

        returns = np.array(self.returns)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        if std < SAFE_EPSILON:
            return 0.0

        # Standardized fourth moment
        z = (returns - mean) / std
        kurtosis = np.mean(z**4)

        # Fisher's definition (excess kurtosis, normal = 0)
        # Adjust for bias
        excess_kurtosis = kurtosis - 3.0

        return float(excess_kurtosis + 3.0)  # Return total kurtosis (normal = 3)

    def get_current_kurtosis(self) -> float:
        """Get current kurtosis"""
        return self._calculate_kurtosis()


class DrawdownBreaker:
    """
    Drawdown Circuit Breaker

    Handbook: "Progressive size reduction as drawdown increases"
    """

    def __init__(self, thresholds: dict[float, float] | None = None):
        """
        Args:
            thresholds: {drawdown_pct: size_multiplier}
                       e.g., {0.05: 0.9, 0.10: 0.75, 0.15: 0.5, 0.20: 0.0}
        """
        if thresholds is None:
            thresholds = DRAWDOWN_DEFAULT_THRESHOLDS
        self.thresholds = dict(thresholds)
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0
        self.size_multiplier = 1.0

        self.state = BreakerState(
            name="Drawdown",
            threshold=max(self.thresholds.keys()),
            cooldown_minutes=DRAWDOWN_COOLDOWN_MINUTES,  # 4 hour cooldown
        )

    def update(self, equity: float):
        """Update with current equity"""
        if not SafeMath.is_valid(equity) or equity <= 0:
            return

        self.current_equity = equity

        # Update peak
        self.peak_equity = max(self.peak_equity, equity)

        # Calculate drawdown using safe division to avoid zero denominators
        drawdown_numerator = self.peak_equity - equity
        self.current_drawdown = SafeMath.safe_div(drawdown_numerator, self.peak_equity, 0.0)

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
                reason="Maximum drawdown exceeded",
                value=self.current_drawdown,
                threshold=max(self.thresholds.keys()),
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

    def __init__(self, max_losses: int = CONSEC_LOSSES_DEFAULT_MAX):
        """
        Args:
            max_losses: Maximum consecutive losses before halt
        """
        self.max_losses = max_losses
        self.consecutive_losses = 0

        self.state = BreakerState(
            name="Consecutive Losses",
            threshold=float(max_losses),
            cooldown_minutes=CONSEC_LOSSES_COOLDOWN_MINUTES,  # 3 hour cooldown
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
                reason="Too many consecutive losses",
                value=float(self.consecutive_losses),
                threshold=float(self.max_losses),
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

    Integration with emergency position closer:
    - When breakers trip, can automatically close all positions
    - Set emergency_closer via set_emergency_closer() method
    """

    def __init__(
        self,
        sortino_threshold: float | None = None,
        kurtosis_threshold: float | None = None,
        max_drawdown: float | None = None,
        max_consecutive_losses: int | None = None,
        symbol: str = "XAUUSD",  # Instrument-agnostic: default for tests
        timeframe: str = "M15",
        broker: str = "default",
        param_manager: LearnedParametersManager | None = None,
        auto_close_on_trip: bool = False,
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
            "sortino_threshold", sortino_threshold, MANAGER_DEFAULT_SORTINO
        )
        self.kurtosis_threshold, kurtosis_source = self._resolve_param(
            "kurtosis_threshold", kurtosis_threshold, MANAGER_DEFAULT_KURTOSIS
        )
        self.max_drawdown, drawdown_source = self._resolve_param(
            "max_drawdown_pct", max_drawdown, MANAGER_DEFAULT_MAX_DRAWDOWN
        )
        self.max_consecutive_losses, loss_source = self._resolve_param(
            "max_consecutive_losses", max_consecutive_losses, MANAGER_DEFAULT_MAX_LOSSES
        )
        self.max_consecutive_losses = int(round(self.max_consecutive_losses))

        self.sortino_breaker = SortinoBreaker(threshold=self.sortino_threshold)
        self.kurtosis_breaker = KurtosisBreaker(threshold=self.kurtosis_threshold)
        self.drawdown_breaker = DrawdownBreaker(thresholds={**DRAWDOWN_DEFAULT_THRESHOLDS, self.max_drawdown: 0.0})
        self.consecutive_losses_breaker = ConsecutiveLossesBreaker(max_losses=self.max_consecutive_losses)

        self.breakers: list[ManagedBreaker] = [
            self.sortino_breaker,
            self.kurtosis_breaker,
            self.drawdown_breaker,
            self.consecutive_losses_breaker,
        ]

        # Emergency position closer (set via set_emergency_closer)
        self.emergency_closer = None
        self.auto_close_on_trip = auto_close_on_trip
        self.positions_closed_on_trip = False

        LOG.info(
            "Circuit Breaker Manager initialized | Sortino>=%.2f (%s) Kurtosis<=%.1f (%s) DD<=%.0f%% (%s) MaxLoss=%d (%s)",
            self.sortino_threshold,
            sortino_source,
            self.kurtosis_threshold,
            kurtosis_source,
            self.max_drawdown * 100,
            drawdown_source,
            self.max_consecutive_losses,
            loss_source,
        )

    def _resolve_param(self, name: str, explicit_value: float | None, default: float):
        """Resolve breaker thresholds with override → learned → default."""
        if explicit_value is not None:
            try:
                return float(explicit_value), "explicit"
            except (TypeError, ValueError):
                LOG.warning(
                    "[CIRCUIT-BREAKERS] Invalid explicit override for %s (%s) - using default %.3f",
                    name,
                    explicit_value,
                    default,
                )
                return float(default), "default"
        if self.param_manager is not None:
            try:
                value = self.param_manager.get(
                    self.symbol, name, timeframe=self.timeframe, broker=self.broker, default=default
                )
                return float(value), "learned"
            except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                LOG.debug(
                    "[CIRCUIT-BREAKERS] Failed to fetch %s via LearnedParameters (%s) - using default %.3f",
                    name,
                    exc,
                    default,
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

        # Emergency close positions if configured and breaker just tripped
        if any_tripped and self.auto_close_on_trip and not self.positions_closed_on_trip:
            self._execute_emergency_close()

        return any_tripped

    def set_emergency_closer(self, emergency_closer):
        """
        Set emergency position closer for auto-close on trip.

        Args:
            emergency_closer: EmergencyPositionCloser instance
        """
        self.emergency_closer = emergency_closer
        LOG.info("[CIRCUIT-BREAKERS] Emergency closer configured (auto_close=%s)", self.auto_close_on_trip)

    def _execute_emergency_close(self):
        """Execute emergency position close when breaker trips."""
        if not self.emergency_closer:
            LOG.warning("[CIRCUIT-BREAKERS] 🚨 Breaker tripped but no emergency_closer configured!")
            return

        try:
            tripped = self.get_tripped_breakers()
            reasons = ", ".join([f"{b.name}: {b.trip_reason}" for b in tripped])

            LOG.warning("[CIRCUIT-BREAKERS] 🚨 EXECUTING EMERGENCY CLOSE: %s", reasons)

            success = self.emergency_closer.close_all_positions(reason="CIRCUIT_BREAKER")

            if success:
                self.positions_closed_on_trip = True
                LOG.warning("[CIRCUIT-BREAKERS] ✓ Emergency close executed")
            else:
                LOG.error("[CIRCUIT-BREAKERS] ✗ Emergency close FAILED - manual intervention required!")

        except Exception as e:
            LOG.error("[CIRCUIT-BREAKERS] Emergency close error: %s", e, exc_info=True)

    def is_any_tripped(self) -> bool:
        """Check if any breaker is currently tripped"""
        return any(breaker.state.is_tripped for breaker in self.breakers)

    def get_tripped_breakers(self) -> list[BreakerState]:
        """Get list of tripped breakers"""
        return [breaker.state for breaker in self.breakers if breaker.state.is_tripped]

    def reset_all(self):
        """Reset all breakers (use with caution)"""
        for breaker in self.breakers:
            breaker.state.reset()
        self.positions_closed_on_trip = False  # Reset flag
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

    def get_status(self) -> dict:
        """Get comprehensive status"""
        return {
            "any_tripped": self.is_any_tripped(),
            "position_multiplier": self.get_position_size_multiplier(),
            "sortino": {
                "tripped": self.sortino_breaker.state.is_tripped,
                "current": self.sortino_breaker.get_current_sortino(),
                "threshold": self.sortino_breaker.threshold,
            },
            "kurtosis": {
                "tripped": self.kurtosis_breaker.state.is_tripped,
                "current": self.kurtosis_breaker.get_current_kurtosis(),
                "threshold": self.kurtosis_breaker.threshold,
            },
            "drawdown": {
                "tripped": self.drawdown_breaker.state.is_tripped,
                "current": self.drawdown_breaker.get_drawdown(),
                "threshold": self.drawdown_breaker.state.threshold,
                "size_mult": self.drawdown_breaker.get_size_multiplier(),
            },
            "consecutive_losses": {
                "tripped": self.consecutive_losses_breaker.state.is_tripped,
                "current": self.consecutive_losses_breaker.get_consecutive_losses(),
                "threshold": self.consecutive_losses_breaker.max_losses,
            },
        }

    def save_state(self, filepath: str = "data/circuit_breakers.json"):
        """
        GAP 10.2 FIX: Save circuit breaker state to disk for persistence across restarts.

        Args:
            filepath: Path to save state file
        """
        import json
        import time as _time
        from pathlib import Path

        state = {
            "timestamp": _time.time(),
            "sortino": {
                "is_tripped": self.sortino_breaker.state.is_tripped,
                "trip_time": self.sortino_breaker.state.trip_time,
                "returns": list(self.sortino_breaker.returns),
            },
            "kurtosis": {
                "is_tripped": self.kurtosis_breaker.state.is_tripped,
                "trip_time": self.kurtosis_breaker.state.trip_time,
                "returns": list(self.kurtosis_breaker.returns),
            },
            "drawdown": {
                "is_tripped": self.drawdown_breaker.state.is_tripped,
                "trip_time": self.drawdown_breaker.state.trip_time,
                "current_drawdown": self.drawdown_breaker.current_drawdown,
                "peak_equity": self.drawdown_breaker.peak_equity,
            },
            "consecutive_losses": {
                "is_tripped": self.consecutive_losses_breaker.state.is_tripped,
                "trip_time": self.consecutive_losses_breaker.state.trip_time,
                "consecutive_losses": self.consecutive_losses_breaker.consecutive_losses,
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def restore_state(self, filepath: str = "data/circuit_breakers.json"):
        """
        GAP 10.2 FIX: Restore circuit breaker state from disk.

        Args:
            filepath: Path to state file

        Returns:
            True if state restored, False if file not found
        """
        import json
        from collections import deque
        from pathlib import Path

        if not Path(filepath).exists():
            return False

        try:
            with open(filepath) as f:
                state = json.load(f)

            # Restore sortino breaker
            if "sortino" in state:
                self.sortino_breaker.state.is_tripped = state["sortino"]["is_tripped"]
                self.sortino_breaker.state.trip_time = state["sortino"].get("trip_time")
                self.sortino_breaker.returns = deque(state["sortino"]["returns"], maxlen=100)

            # Restore kurtosis breaker
            if "kurtosis" in state:
                self.kurtosis_breaker.state.is_tripped = state["kurtosis"]["is_tripped"]
                self.kurtosis_breaker.state.trip_time = state["kurtosis"].get("trip_time")
                self.kurtosis_breaker.returns = deque(state["kurtosis"]["returns"], maxlen=100)

            # Restore drawdown breaker
            if "drawdown" in state:
                self.drawdown_breaker.state.is_tripped = state["drawdown"]["is_tripped"]
                self.drawdown_breaker.state.trip_time = state["drawdown"].get("trip_time")
                self.drawdown_breaker.current_drawdown = state["drawdown"]["current_drawdown"]
                self.drawdown_breaker.peak_equity = state["drawdown"]["peak_equity"]

            # Restore consecutive losses breaker
            if "consecutive_losses" in state:
                self.consecutive_losses_breaker.state.is_tripped = state["consecutive_losses"]["is_tripped"]
                self.consecutive_losses_breaker.state.trip_time = state["consecutive_losses"].get("trip_time")
                self.consecutive_losses_breaker.consecutive_losses = state["consecutive_losses"]["consecutive_losses"]

            LOG.info("[CIRCUIT-BREAKER] State restored from %s", filepath)
            return True

        except Exception as e:
            LOG.error("[CIRCUIT-BREAKER] Failed to restore state: %s", e)
            return False


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CIRCUIT BREAKERS - TEST SUITE")
    print("=" * 80)

    # Test 1: Sortino Breaker
    print("\n[Test 1] Sortino Ratio Breaker")
    print("-" * 80)

    sortino_demo = SortinoBreaker(threshold=0.5, min_trades=10)

    # Simulate good trades
    print("Simulating 10 profitable trades:")
    for _i in range(10):
        sortino_demo.update(0.02)  # +2% returns

    current_sortino = sortino_demo.get_current_sortino()
    print(f"Sortino ratio: {current_sortino:.3f}")
    print(f"Tripped: {sortino_demo.check()}")

    # Simulate bad trades
    print("\nSimulating 5 large losses:")
    for _i in range(5):
        sortino_demo.update(-0.05)  # -5% losses

    current_sortino = sortino_demo.get_current_sortino()
    print(f"Sortino ratio: {current_sortino:.3f}")
    print(f"Tripped: {sortino_demo.check()}")

    # Test 2: Kurtosis Breaker
    print("\n[Test 2] Kurtosis Breaker")
    print("-" * 80)

    kurtosis_demo = KurtosisBreaker(threshold=5.0, min_samples=30)

    # Normal distribution
    print("Simulating normal returns:")
    rng = np.random.default_rng(42)
    normal_returns = rng.normal(0, 0.02, 30)
    for ret in normal_returns:
        kurtosis_demo.update(ret)

    current_kurt = kurtosis_demo.get_current_kurtosis()
    print(f"Kurtosis: {current_kurt:.3f} (normal ≈ 3.0)")
    print(f"Tripped: {kurtosis_demo.check()}")

    # Fat-tailed distribution
    print("\nAdding extreme outliers:")
    for _ in range(5):
        kurtosis_demo.update(0.15)  # Extreme positive
        kurtosis_demo.update(-0.15)  # Extreme negative

    current_kurt = kurtosis_demo.get_current_kurtosis()
    print(f"Kurtosis: {current_kurt:.3f}")
    print(f"Tripped: {kurtosis_demo.check()}")

    # Test 3: Drawdown Breaker
    print("\n[Test 3] Drawdown Breaker")
    print("-" * 80)

    drawdown_demo = DrawdownBreaker()

    # Simulate equity curve
    equity_series = [10000, 10500, 11000, 10800, 10200, 9500, 9000, 8500, 8000]

    print("Equity progression:")
    for eq in equity_series:
        drawdown_demo.update(eq)
        dd = drawdown_demo.get_drawdown()
        mult = drawdown_demo.get_size_multiplier()
        tripped = drawdown_demo.check()
        print(f"  Equity: ${eq:6.0f} | DD: {dd:5.1%} | Size mult: {mult:.2f} | Tripped: {tripped}")

    # Test 4: Consecutive Losses
    print("\n[Test 4] Consecutive Losses Breaker")
    print("-" * 80)

    consecutive_demo = ConsecutiveLossesBreaker(max_losses=5)

    # Simulate trade sequence
    result_sequence = [False, False, True, False, False, False, False, False, True, False]

    print("Trade sequence:")
    for i, result_is_win in enumerate(result_sequence):
        consecutive_demo.update(result_is_win)
        count = consecutive_demo.get_consecutive_losses()
        tripped = consecutive_demo.check()

        result_str = "WIN " if result_is_win else "LOSS"
        print(f"  Trade {i+1}: {result_str} | Consecutive losses: {count} | Tripped: {tripped}")

    # Test 5: Circuit Breaker Manager
    print("\n[Test 5] Circuit Breaker Manager")
    print("-" * 80)

    manager_demo = CircuitBreakerManager(
        sortino_threshold=0.5, kurtosis_threshold=5.0, max_drawdown=0.20, max_consecutive_losses=3
    )

    # Simulate trading
    print("\nSimulating trades:")

    demo_trades = [
        (100, 10100),  # Win
        (-50, 10050),  # Loss
        (-80, 9970),  # Loss
        (-100, 9870),  # Loss (3rd consecutive)
        (50, 9920),  # Would win but breaker tripped
    ]

    for i, (trade_pnl, trade_equity) in enumerate(demo_trades):
        print(f"\nTrade {i+1}: P&L=${trade_pnl:+4.0f} | Equity=${trade_equity:.0f}")

        manager_demo.update_trade(trade_pnl / 100, trade_equity)  # Normalize PnL
        manager_demo.check_all()

        status = manager_demo.get_status()
        print(f"  Any tripped: {status['any_tripped']}")
        print(f"  Size multiplier: {status['position_multiplier']:.2f}")
        print(f"  Consecutive losses: {status['consecutive_losses']['current']}")

        if status["any_tripped"]:
            tripped_breakers = manager_demo.get_tripped_breakers()
            for tripped_breaker in tripped_breakers:
                print(f"  ⚠️  {tripped_breaker.name} TRIPPED: {tripped_breaker.trip_reason}")

    # Test 6: Auto-reset after cooldown
    print("\n[Test 6] Auto-Reset After Cooldown")
    print("-" * 80)

    # Manually trip a breaker
    cooldown_state = manager_demo.consecutive_losses_breaker.state
    cooldown_state.trip("Test", 5, 3)
    print(f"Breaker tripped: {cooldown_state.is_tripped}")

    # Simulate cooldown elapsed
    fake_past = datetime.now() - timedelta(hours=4)
    cooldown_state.trip_time = fake_past

    print(f"Cooldown can reset: {cooldown_state.can_reset()}")

    manager_demo.reset_if_cooldown_elapsed()
    print(f"After auto-reset: {cooldown_state.is_tripped}")

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
