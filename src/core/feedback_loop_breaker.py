"""
Feedback Loop Breaker

Detects and breaks circular dependencies and stuck states in RL trading systems:
- No trades despite opportunities (learned helplessness)
- Permanently tripped circuit breakers (stuck safety systems)
- Deteriorating performance without recovery (negative spiral)
- Exploration collapse (agent stuck in local minimum)

These feedback loops can emerge when:
1. Circuit breakers trip → No trades → No new experiences → No learning → Circuit breakers stay tripped
2. Bad trades → Negative rewards → Defensive policy → Missed opportunities → More negative rewards
3. Low confidence → No exploration → Stale policy → Even lower confidence

The loop breaker provides interventions to restart healthy learning dynamics.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeedbackLoopSignal:
    """Signal indicating a potential feedback loop."""

    loop_type: str  # "no_trades", "circuit_breaker", "performance_decay", "exploration_collapse"
    severity: float  # 0.0 to 1.0
    duration_bars: int
    evidence: dict[str, any]
    suggested_intervention: str


class FeedbackLoopBreaker:
    """
    Detects and breaks feedback loops in RL trading systems.

    Detection mechanisms:
    1. **No-Trade Loop**: Market volatility + opportunities present, but no trades for extended period
    2. **Circuit Breaker Loop**: Circuit breakers tripped for >N bars without reset
    3. **Performance Decay Loop**: Win rate / Sharpe declining over rolling window
    4. **Exploration Collapse**: Entropy / action diversity below threshold

    Interventions:
    1. **Gentle**: Increase exploration epsilon temporarily
    2. **Moderate**: Reset circuit breakers, inject synthetic experiences
    3. **Aggressive**: Reset policy to earlier checkpoint, force random trades
    4. **Nuclear**: Full reset to initialization (last resort)
    """

    def __init__(
        self,
        # No-trade detection
        no_trade_window_bars: int = 240,  # 4 hours at 1-min bars
        min_volatility_threshold: float = 0.005,  # 50 bps - market is moving
        # Circuit breaker detection
        circuit_breaker_stuck_bars: int = 120,  # 2 hours
        # Performance decay detection
        performance_window_bars: int = 1000,  # ~16 hours
        decay_threshold: float = 0.3,  # 30% decline in Sharpe
        # Exploration collapse detection
        entropy_threshold: float = 0.2,  # Bits of entropy in action distribution
        min_exploration_rate: float = 0.05,
        # Intervention settings
        intervention_cooldown_bars: int = 500,  # Don't intervene too frequently
        state_file: Path | None = None,
    ):
        self.no_trade_window_bars = no_trade_window_bars
        self.min_volatility_threshold = min_volatility_threshold
        self.circuit_breaker_stuck_bars = circuit_breaker_stuck_bars
        self.performance_window_bars = performance_window_bars
        self.decay_threshold = decay_threshold
        self.entropy_threshold = entropy_threshold
        self.min_exploration_rate = min_exploration_rate
        self.intervention_cooldown_bars = intervention_cooldown_bars

        # State tracking
        self.bars_since_trade = 0
        self.bars_since_circuit_breaker_trip = 0
        self.circuit_breaker_tripped = False
        self.recent_volatilities: list[float] = []
        self.recent_sharpes: list[float] = []
        self.recent_win_rates: list[float] = []
        self.recent_action_entropies: list[float] = []
        self.bars_since_intervention = 999999  # Large number initially

        # Intervention history
        self.interventions: list[dict] = []

        # Persistence
        self.state_file = state_file or Path("data/feedback_loop_breaker.json")

    def update(
        self,
        bars_since_last_trade: int,
        current_volatility: float,
        circuit_breakers_tripped: bool,
        recent_sharpe: float | None = None,
        recent_win_rate: float | None = None,
        action_entropy: float | None = None,
        exploration_rate: float | None = None,
    ) -> FeedbackLoopSignal | None:
        """
        Update loop detection with current state.

        Returns signal if loop detected, None otherwise.
        """
        self.bars_since_trade = bars_since_last_trade
        self.circuit_breaker_tripped = circuit_breakers_tripped
        self.bars_since_intervention += 1

        # Track volatility
        self.recent_volatilities.append(current_volatility)
        if len(self.recent_volatilities) > self.no_trade_window_bars:
            self.recent_volatilities.pop(0)

        # Track performance metrics
        if recent_sharpe is not None:
            self.recent_sharpes.append(recent_sharpe)
            if len(self.recent_sharpes) > 10:  # Keep last 10 snapshots
                self.recent_sharpes.pop(0)

        if recent_win_rate is not None:
            self.recent_win_rates.append(recent_win_rate)
            if len(self.recent_win_rates) > 10:
                self.recent_win_rates.pop(0)

        if action_entropy is not None:
            self.recent_action_entropies.append(action_entropy)
            if len(self.recent_action_entropies) > 100:
                self.recent_action_entropies.pop(0)

        # Track circuit breaker duration
        if circuit_breakers_tripped:
            self.bars_since_circuit_breaker_trip += 1
        else:
            self.bars_since_circuit_breaker_trip = 0

        # Check for feedback loops (in priority order)
        signal = None

        # 1. Circuit breaker stuck (highest priority)
        signal = self._detect_circuit_breaker_loop()
        if signal:
            return signal

        # 2. No-trade loop
        signal = self._detect_no_trade_loop()
        if signal:
            return signal

        # 3. Performance decay
        signal = self._detect_performance_decay_loop()
        if signal:
            return signal

        # 4. Exploration collapse
        signal = self._detect_exploration_collapse(exploration_rate)
        if signal:
            return signal

        return None

    def _detect_circuit_breaker_loop(self) -> FeedbackLoopSignal | None:
        """Detect stuck circuit breakers."""
        if not self.circuit_breaker_tripped:
            return None

        if self.bars_since_circuit_breaker_trip < self.circuit_breaker_stuck_bars:
            return None

        # Circuit breaker has been tripped for too long
        severity = min(1.0, self.bars_since_circuit_breaker_trip / (self.circuit_breaker_stuck_bars * 2))

        return FeedbackLoopSignal(
            loop_type="circuit_breaker",
            severity=severity,
            duration_bars=self.bars_since_circuit_breaker_trip,
            evidence={
                "bars_since_trip": self.bars_since_circuit_breaker_trip,
                "threshold": self.circuit_breaker_stuck_bars,
            },
            suggested_intervention="reset_circuit_breakers",
        )

    def _detect_no_trade_loop(self) -> FeedbackLoopSignal | None:
        """Detect no trades despite market opportunities."""
        if self.bars_since_trade < self.no_trade_window_bars:
            return None

        # Check if market is active (volatility present)
        if len(self.recent_volatilities) < self.no_trade_window_bars // 2:
            return None  # Not enough data

        avg_volatility = sum(self.recent_volatilities) / len(self.recent_volatilities)

        if avg_volatility < self.min_volatility_threshold:
            return None  # Market is dead, no trades is OK

        # We have volatility but no trades - potential loop
        severity = min(1.0, self.bars_since_trade / (self.no_trade_window_bars * 2))

        return FeedbackLoopSignal(
            loop_type="no_trades",
            severity=severity,
            duration_bars=self.bars_since_trade,
            evidence={
                "bars_since_trade": self.bars_since_trade,
                "avg_volatility": avg_volatility,
                "threshold_volatility": self.min_volatility_threshold,
            },
            suggested_intervention="increase_exploration" if severity < 0.7 else "inject_synthetic_experiences",
        )

    def _detect_performance_decay_loop(self) -> FeedbackLoopSignal | None:
        """Detect deteriorating performance spiral."""
        if len(self.recent_sharpes) < 5:
            return None  # Not enough history

        # Check for declining trend in Sharpe
        early_sharpe = sum(self.recent_sharpes[:3]) / 3
        recent_sharpe = sum(self.recent_sharpes[-3:]) / 3

        if early_sharpe <= 0:
            return None  # Was already bad

        decay_pct = (early_sharpe - recent_sharpe) / early_sharpe

        if decay_pct < self.decay_threshold:
            return None  # Not declining enough

        severity = min(1.0, decay_pct)

        return FeedbackLoopSignal(
            loop_type="performance_decay",
            severity=severity,
            duration_bars=len(self.recent_sharpes) * 100,  # Approximate
            evidence={
                "early_sharpe": early_sharpe,
                "recent_sharpe": recent_sharpe,
                "decay_pct": decay_pct,
            },
            suggested_intervention="restore_earlier_checkpoint" if severity > 0.5 else "increase_exploration",
        )

    def _detect_exploration_collapse(self, exploration_rate: float | None) -> FeedbackLoopSignal | None:
        """Detect collapsed exploration (agent stuck in local minimum)."""
        if len(self.recent_action_entropies) < 50:
            return None

        avg_entropy = sum(self.recent_action_entropies) / len(self.recent_action_entropies)

        # Check if entropy is too low
        if avg_entropy >= self.entropy_threshold:
            return None

        # Also check exploration rate if available
        low_exploration = False
        if exploration_rate is not None and exploration_rate < self.min_exploration_rate:
            low_exploration = True

        severity = 1.0 - (avg_entropy / self.entropy_threshold)
        if low_exploration:
            severity = min(1.0, severity * 1.5)

        return FeedbackLoopSignal(
            loop_type="exploration_collapse",
            severity=severity,
            duration_bars=len(self.recent_action_entropies),
            evidence={
                "avg_entropy": avg_entropy,
                "threshold_entropy": self.entropy_threshold,
                "exploration_rate": exploration_rate,
            },
            suggested_intervention="force_exploration",
        )

    def apply_intervention(self, signal: FeedbackLoopSignal) -> dict[str, any]:
        """
        Apply intervention to break feedback loop.

        Returns dict with intervention details for caller to execute.
        """
        # Check cooldown
        if self.bars_since_intervention < self.intervention_cooldown_bars:
            logger.warning(
                f"Intervention requested but still in cooldown ({self.bars_since_intervention}/{self.intervention_cooldown_bars} bars)"
            )
            return {"action": "none", "reason": "cooldown"}

        # Reset cooldown
        self.bars_since_intervention = 0

        # Map intervention type to action
        intervention = {
            "timestamp": time.time(),
            "signal": signal.loop_type,
            "severity": signal.severity,
            "action": signal.suggested_intervention,
            "params": {},
        }

        if signal.suggested_intervention == "reset_circuit_breakers":
            intervention["params"] = {"reset_all": True, "reason": "stuck_loop"}

        elif signal.suggested_intervention == "increase_exploration":
            # Temporarily boost epsilon
            intervention["params"] = {
                "epsilon_boost": 0.2,
                "duration_bars": 100,
            }

        elif signal.suggested_intervention == "inject_synthetic_experiences":
            # Add synthetic successful trades to experience buffer
            intervention["params"] = {
                "num_experiences": 50,
                "reward_range": [0.3, 0.8],  # Positive experiences
            }

        elif signal.suggested_intervention == "force_exploration":
            # Force random actions for N steps
            intervention["params"] = {
                "random_action_bars": 50,
            }

        elif signal.suggested_intervention == "restore_earlier_checkpoint":
            # Revert to earlier policy checkpoint
            intervention["params"] = {
                "checkpoint_age_bars": 5000,  # Go back ~3 days
            }

        # Log intervention
        self.interventions.append(intervention)
        logger.warning(
            f"FEEDBACK LOOP INTERVENTION: {signal.loop_type} (severity={signal.severity:.2f}) -> {signal.suggested_intervention}"
        )

        return intervention

    def save_state(self):
        """Persist state to disk."""
        state = {
            "bars_since_trade": self.bars_since_trade,
            "bars_since_circuit_breaker_trip": self.bars_since_circuit_breaker_trip,
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
            "bars_since_intervention": self.bars_since_intervention,
            "recent_volatilities": self.recent_volatilities,
            "recent_sharpes": self.recent_sharpes,
            "recent_win_rates": self.recent_win_rates,
            "recent_action_entropies": self.recent_action_entropies,
            "interventions": self.interventions[-100:],  # Keep last 100
        }

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> bool:
        """Load state from disk."""
        if not self.state_file.exists():
            return False

        try:
            with open(self.state_file) as f:
                state = json.load(f)

            self.bars_since_trade = state.get("bars_since_trade", 0)
            self.bars_since_circuit_breaker_trip = state.get("bars_since_circuit_breaker_trip", 0)
            self.circuit_breaker_tripped = state.get("circuit_breaker_tripped", False)
            self.bars_since_intervention = state.get("bars_since_intervention", 999999)
            self.recent_volatilities = state.get("recent_volatilities", [])
            self.recent_sharpes = state.get("recent_sharpes", [])
            self.recent_win_rates = state.get("recent_win_rates", [])
            self.recent_action_entropies = state.get("recent_action_entropies", [])
            self.interventions = state.get("interventions", [])

            logger.info(f"Loaded FeedbackLoopBreaker state from {self.state_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False


# Self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== FeedbackLoopBreaker Self-Test ===\n")

    breaker = FeedbackLoopBreaker()

    # Test 1: No-trade loop
    print("Test 1: No-trade loop detection")
    for i in range(300):
        signal = breaker.update(
            bars_since_last_trade=i,
            current_volatility=0.01,  # High volatility
            circuit_breakers_tripped=False,
        )
        if signal:
            print(f"  ✓ Detected {signal.loop_type} at bar {i}, severity={signal.severity:.2f}")
            print(f"  Suggested: {signal.suggested_intervention}")
            break
    else:
        print("  ✗ Failed to detect no-trade loop")

    # Test 2: Circuit breaker stuck
    print("\nTest 2: Circuit breaker stuck detection")
    breaker2 = FeedbackLoopBreaker()
    for i in range(150):
        signal = breaker2.update(
            bars_since_last_trade=i,
            current_volatility=0.005,
            circuit_breakers_tripped=True,  # Stuck
        )
        if signal and signal.loop_type == "circuit_breaker":
            print(f"  ✓ Detected {signal.loop_type} at bar {i}, severity={signal.severity:.2f}")
            print(f"  Suggested: {signal.suggested_intervention}")
            break
    else:
        print("  ✗ Failed to detect circuit breaker loop")

    # Test 3: Performance decay
    print("\nTest 3: Performance decay detection")
    breaker3 = FeedbackLoopBreaker()
    sharpes = [0.8, 0.75, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2]  # Declining
    for i, sharpe in enumerate(sharpes):
        signal = breaker3.update(
            bars_since_last_trade=10,
            current_volatility=0.005,
            circuit_breakers_tripped=False,
            recent_sharpe=sharpe,
        )
        if signal and signal.loop_type == "performance_decay":
            print(f"  ✓ Detected {signal.loop_type} at snapshot {i}, severity={signal.severity:.2f}")
            print(f"  Suggested: {signal.suggested_intervention}")
            break
    else:
        print("  ✗ Failed to detect performance decay")

    # Test 4: Exploration collapse
    print("\nTest 4: Exploration collapse detection")
    breaker4 = FeedbackLoopBreaker()
    for i in range(100):
        signal = breaker4.update(
            bars_since_last_trade=10,
            current_volatility=0.005,
            circuit_breakers_tripped=False,
            action_entropy=0.05,  # Very low
            exploration_rate=0.01,  # Very low
        )
        if signal and signal.loop_type == "exploration_collapse":
            print(f"  ✓ Detected {signal.loop_type} at bar {i}, severity={signal.severity:.2f}")
            print(f"  Suggested: {signal.suggested_intervention}")
            break
    else:
        print("  ✗ Failed to detect exploration collapse")

    # Test 5: Intervention cooldown
    print("\nTest 5: Intervention cooldown")
    breaker5 = FeedbackLoopBreaker(intervention_cooldown_bars=50)
    signal = FeedbackLoopSignal(
        loop_type="no_trades",
        severity=0.8,
        duration_bars=300,
        evidence={},
        suggested_intervention="increase_exploration",
    )
    result1 = breaker5.apply_intervention(signal)
    result2 = breaker5.apply_intervention(signal)  # Should be blocked by cooldown

    if result1["action"] != "none" and result2["action"] == "none":
        print("  ✓ Cooldown working correctly")
    else:
        print(f"  ✗ Cooldown failed: result1={result1}, result2={result2}")

    # Test 6: State persistence
    print("\nTest 6: State persistence")
    import tempfile

    temp_file = Path(tempfile.mktemp(suffix=".json"))
    breaker6 = FeedbackLoopBreaker(state_file=temp_file)
    breaker6.update(
        bars_since_last_trade=100,
        current_volatility=0.01,
        circuit_breakers_tripped=False,
    )
    breaker6.save_state()

    breaker6_reloaded = FeedbackLoopBreaker(state_file=temp_file)
    if breaker6_reloaded.load_state():
        if breaker6_reloaded.bars_since_trade == 100:
            print("  ✓ State persistence working")
        else:
            print(f"  ✗ State mismatch: expected 100, got {breaker6_reloaded.bars_since_trade}")
    else:
        print("  ✗ Failed to load state")

    temp_file.unlink(missing_ok=True)

    print("\n=== Self-Test Complete ===")
