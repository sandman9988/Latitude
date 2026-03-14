"""
Tests for src.core.feedback_loop_breaker

Coverage targets:
- FeedbackLoopSignal dataclass
- FeedbackLoopBreaker.__init__ / update / detectors / interventions / persistence
"""

import json

import pytest

from src.core.feedback_loop_breaker import FeedbackLoopBreaker, FeedbackLoopSignal


# ── FeedbackLoopSignal ──────────────────────────────────────────────────────

class TestFeedbackLoopSignal:
    def test_create_signal(self):
        sig = FeedbackLoopSignal(
            loop_type="no_trades",
            severity=0.5,
            duration_bars=100,
            evidence={"key": "val"},
            suggested_intervention="increase_exploration",
        )
        assert sig.loop_type == "no_trades"
        assert sig.severity == pytest.approx(0.5)
        assert sig.duration_bars == 100
        assert sig.evidence == {"key": "val"}
        assert sig.suggested_intervention == "increase_exploration"


# ── Constructor / defaults ──────────────────────────────────────────────────

class TestFeedbackLoopBreakerInit:
    def test_default_parameters(self):
        fb = FeedbackLoopBreaker()
        assert fb.no_trade_window_bars == 240
        assert fb.min_volatility_threshold == pytest.approx(0.005)
        assert fb.circuit_breaker_stuck_bars == 120
        assert fb.performance_window_bars == 1000
        assert fb.decay_threshold == pytest.approx(0.3)
        assert fb.entropy_threshold == pytest.approx(0.2)
        assert fb.min_exploration_rate == pytest.approx(0.05)
        assert fb.intervention_cooldown_bars == 500
        assert fb.bars_since_trade == 0
        assert fb.bars_since_circuit_breaker_trip == 0
        assert fb.circuit_breaker_tripped is False
        assert fb.interventions == []

    def test_custom_parameters(self):
        fb = FeedbackLoopBreaker(
            no_trade_window_bars=100,
            min_volatility_threshold=0.01,
            circuit_breaker_stuck_bars=50,
        )
        assert fb.no_trade_window_bars == 100
        assert fb.min_volatility_threshold == pytest.approx(0.01)
        assert fb.circuit_breaker_stuck_bars == 50


# ── Circuit breaker stuck detection ────────────────────────────────────────

class TestCircuitBreakerLoop:
    def test_not_detected_when_not_tripped(self):
        fb = FeedbackLoopBreaker(circuit_breaker_stuck_bars=10)
        for _ in range(20):
            sig = fb.update(
                bars_since_last_trade=0,
                current_volatility=0.01,
                circuit_breakers_tripped=False,
            )
        assert sig is None

    def test_not_detected_before_threshold(self):
        fb = FeedbackLoopBreaker(circuit_breaker_stuck_bars=20)
        sig = None
        for i in range(15):
            sig = fb.update(
                bars_since_last_trade=i,
                current_volatility=0.005,
                circuit_breakers_tripped=True,
            )
        assert sig is None

    def test_detected_after_threshold(self):
        fb = FeedbackLoopBreaker(circuit_breaker_stuck_bars=10)
        sig = None
        for i in range(15):
            sig = fb.update(
                bars_since_last_trade=i,
                current_volatility=0.005,
                circuit_breakers_tripped=True,
            )
        assert sig is not None
        assert sig.loop_type == "circuit_breaker"
        assert sig.suggested_intervention == "reset_circuit_breakers"

    def test_severity_scales_with_duration(self):
        fb = FeedbackLoopBreaker(circuit_breaker_stuck_bars=10)
        sigs = []
        for i in range(25):
            sig = fb.update(
                bars_since_last_trade=i,
                current_volatility=0.005,
                circuit_breakers_tripped=True,
            )
            if sig and sig.loop_type == "circuit_breaker":
                sigs.append(sig)
        # Severity should increase over time
        assert len(sigs) >= 2
        assert sigs[-1].severity >= sigs[0].severity

    def test_resets_when_breakers_clear(self):
        fb = FeedbackLoopBreaker(circuit_breaker_stuck_bars=10)
        # Trip for a while
        for i in range(12):
            fb.update(bars_since_last_trade=i, current_volatility=0.005, circuit_breakers_tripped=True)
        # Clear breakers
        _sig = fb.update(bars_since_last_trade=13, current_volatility=0.005, circuit_breakers_tripped=False)
        # Counter should reset
        assert fb.bars_since_circuit_breaker_trip == 0


# ── No-trade loop detection ────────────────────────────────────────────────

class TestNoTradeLoop:
    def test_not_detected_with_recent_trade(self):
        fb = FeedbackLoopBreaker(no_trade_window_bars=10)
        for _ in range(20):
            sig = fb.update(
                bars_since_last_trade=5,
                current_volatility=0.01,
                circuit_breakers_tripped=False,
            )
        assert sig is None

    def test_not_detected_with_low_volatility(self):
        """No-trade loop ignored when market is dead."""
        fb = FeedbackLoopBreaker(no_trade_window_bars=10, min_volatility_threshold=0.01)
        # Fill volatility buffer with low values
        for i in range(20):
            sig = fb.update(
                bars_since_last_trade=i + 10,
                current_volatility=0.001,  # Below threshold
                circuit_breakers_tripped=False,
            )
        assert sig is None

    def test_detected_with_high_volatility(self):
        """No-trade loop detected when market is active but no trades."""
        fb = FeedbackLoopBreaker(no_trade_window_bars=10, min_volatility_threshold=0.005)
        sig = None
        for i in range(20):
            sig = fb.update(
                bars_since_last_trade=i + 10,
                current_volatility=0.02,  # Well above threshold
                circuit_breakers_tripped=False,
            )
        assert sig is not None
        assert sig.loop_type == "no_trades"
        assert "bars_since_trade" in sig.evidence

    def test_not_enough_volatility_data(self):
        fb = FeedbackLoopBreaker(no_trade_window_bars=100)
        # Only a few updates, not enough data
        for i in range(5):
            sig = fb.update(
                bars_since_last_trade=i + 200,
                current_volatility=0.02,
                circuit_breakers_tripped=False,
            )
        # Not enough volatility data to conclude
        assert sig is None

    def test_gentle_vs_aggressive_intervention(self):
        """Low severity → increase_exploration, high severity → inject_synthetic_experiences."""
        fb = FeedbackLoopBreaker(no_trade_window_bars=10, min_volatility_threshold=0.001)
        # Fill volatility buffer
        for i in range(20):
            sig = fb.update(
                bars_since_last_trade=i + 10,
                current_volatility=0.01,
                circuit_breakers_tripped=False,
            )
        # At moderate severity
        assert sig is not None
        assert sig.suggested_intervention in ("increase_exploration", "inject_synthetic_experiences")


# ── Performance decay detection ─────────────────────────────────────────────

class TestPerformanceDecayLoop:
    def test_not_detected_without_enough_history(self):
        fb = FeedbackLoopBreaker()
        sig = fb.update(
            bars_since_last_trade=5,
            current_volatility=0.01,
            circuit_breakers_tripped=False,
            recent_sharpe=0.5,
        )
        assert sig is None  # Only 1 snapshot, need 5

    def test_not_detected_when_early_sharpes_negative(self):
        """If early Sharpe was already bad, no decay signal."""
        fb = FeedbackLoopBreaker()
        sharpes = [-0.2, -0.3, -0.1, -0.5, -0.4, -0.6, -0.7]
        for s in sharpes:
            sig = fb.update(
                bars_since_last_trade=5,
                current_volatility=0.005,
                circuit_breakers_tripped=False,
                recent_sharpe=s,
            )
        assert sig is None  # early_sharpe <= 0, so no signal

    def test_detected_on_significant_decline(self):
        fb = FeedbackLoopBreaker(decay_threshold=0.3)
        sharpes = [0.8, 0.75, 0.7, 0.4, 0.3, 0.2, 0.15]
        sig = None
        for s in sharpes:
            sig = fb.update(
                bars_since_last_trade=5,
                current_volatility=0.005,
                circuit_breakers_tripped=False,
                recent_sharpe=s,
            )
        assert sig is not None
        assert sig.loop_type == "performance_decay"
        assert sig.evidence["decay_pct"] >= 0.3

    def test_not_detected_with_mild_decline(self):
        fb = FeedbackLoopBreaker(decay_threshold=0.5)
        sharpes = [0.8, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68]
        for s in sharpes:
            sig = fb.update(
                bars_since_last_trade=5,
                current_volatility=0.005,
                circuit_breakers_tripped=False,
                recent_sharpe=s,
            )
        assert sig is None  # Decline is only ~10%, threshold is 50%


# ── Exploration collapse detection ──────────────────────────────────────────

class TestExplorationCollapseLoop:
    def test_not_detected_with_high_entropy(self):
        fb = FeedbackLoopBreaker(entropy_threshold=0.2)
        for _ in range(60):
            sig = fb.update(
                bars_since_last_trade=5,
                current_volatility=0.005,
                circuit_breakers_tripped=False,
                action_entropy=0.5,  # Well above threshold
                exploration_rate=0.1,
            )
        assert sig is None

    def test_not_detected_with_insufficient_data(self):
        fb = FeedbackLoopBreaker(entropy_threshold=0.2)
        for _ in range(10):
            sig = fb.update(
                bars_since_last_trade=5,
                current_volatility=0.005,
                circuit_breakers_tripped=False,
                action_entropy=0.01,
            )
        assert sig is None  # Need 50 datapoints

    def test_detected_with_low_entropy(self):
        fb = FeedbackLoopBreaker(entropy_threshold=0.2)
        sig = None
        for _ in range(60):
            sig = fb.update(
                bars_since_last_trade=5,
                current_volatility=0.005,
                circuit_breakers_tripped=False,
                action_entropy=0.05,
                exploration_rate=0.01,
            )
        assert sig is not None
        assert sig.loop_type == "exploration_collapse"
        assert sig.suggested_intervention == "force_exploration"

    def test_low_exploration_boosts_severity(self):
        fb1 = FeedbackLoopBreaker(entropy_threshold=0.2, min_exploration_rate=0.05)
        fb2 = FeedbackLoopBreaker(entropy_threshold=0.2, min_exploration_rate=0.05)
        for _ in range(60):
            sig1 = fb1.update(
                bars_since_last_trade=5,
                current_volatility=0.005,
                circuit_breakers_tripped=False,
                action_entropy=0.1,
                exploration_rate=0.1,  # Above min
            )
            sig2 = fb2.update(
                bars_since_last_trade=5,
                current_volatility=0.005,
                circuit_breakers_tripped=False,
                action_entropy=0.1,
                exploration_rate=0.01,  # Below min → boost severity
            )
        assert sig1 is not None
        assert sig2 is not None
        assert sig2.severity >= sig1.severity


# ── Update state tracking ──────────────────────────────────────────────────

class TestUpdateStateTracking:
    def test_volatility_window_capped(self):
        fb = FeedbackLoopBreaker(no_trade_window_bars=10)
        for _ in range(50):
            fb.update(0, 0.01, False)
        assert len(fb.recent_volatilities) == 10

    def test_sharpe_history_capped(self):
        fb = FeedbackLoopBreaker()
        for i in range(20):
            fb.update(0, 0.01, False, recent_sharpe=float(i))
        assert len(fb.recent_sharpes) == 10

    def test_win_rate_history_capped(self):
        fb = FeedbackLoopBreaker()
        for i in range(20):
            fb.update(0, 0.01, False, recent_win_rate=float(i) / 20)
        assert len(fb.recent_win_rates) == 10

    def test_entropy_history_capped(self):
        fb = FeedbackLoopBreaker()
        for i in range(200):
            fb.update(0, 0.01, False, action_entropy=0.5)
        assert len(fb.recent_action_entropies) == 100


# ── apply_intervention ──────────────────────────────────────────────────────

class TestApplyIntervention:
    def _make_signal(self, intervention="increase_exploration", severity=0.5):
        return FeedbackLoopSignal(
            loop_type="test",
            severity=severity,
            duration_bars=100,
            evidence={},
            suggested_intervention=intervention,
        )

    def test_increase_exploration(self):
        fb = FeedbackLoopBreaker()
        result = fb.apply_intervention(self._make_signal("increase_exploration"))
        assert result["action"] == "increase_exploration"
        assert "epsilon_boost" in result["params"]

    def test_reset_circuit_breakers(self):
        fb = FeedbackLoopBreaker()
        result = fb.apply_intervention(self._make_signal("reset_circuit_breakers"))
        assert result["action"] == "reset_circuit_breakers"
        assert result["params"]["reset_all"] is True

    def test_inject_synthetic_experiences(self):
        fb = FeedbackLoopBreaker()
        result = fb.apply_intervention(self._make_signal("inject_synthetic_experiences"))
        assert result["action"] == "inject_synthetic_experiences"
        assert "num_experiences" in result["params"]

    def test_force_exploration(self):
        fb = FeedbackLoopBreaker()
        result = fb.apply_intervention(self._make_signal("force_exploration"))
        assert result["action"] == "force_exploration"
        assert "random_action_bars" in result["params"]

    def test_restore_earlier_checkpoint(self):
        fb = FeedbackLoopBreaker()
        result = fb.apply_intervention(self._make_signal("restore_earlier_checkpoint"))
        assert result["action"] == "restore_earlier_checkpoint"
        assert "checkpoint_age_bars" in result["params"]

    def test_cooldown_blocks_second_intervention(self):
        fb = FeedbackLoopBreaker(intervention_cooldown_bars=100)
        sig = self._make_signal()
        result1 = fb.apply_intervention(sig)
        assert result1["action"] != "none"
        result2 = fb.apply_intervention(sig)
        assert result2["action"] == "none"
        assert result2["reason"] == "cooldown"

    def test_cooldown_expires_after_enough_bars(self):
        fb = FeedbackLoopBreaker(intervention_cooldown_bars=5)
        sig = self._make_signal()
        fb.apply_intervention(sig)
        # Advance bars past cooldown
        for _ in range(10):
            fb.update(0, 0.01, False)
        result = fb.apply_intervention(sig)
        assert result["action"] != "none"

    def test_intervention_logged(self):
        fb = FeedbackLoopBreaker()
        fb.apply_intervention(self._make_signal())
        assert len(fb.interventions) == 1
        assert fb.interventions[0]["signal"] == "test"


# ── Priority ordering ──────────────────────────────────────────────────────

class TestDetectionPriority:
    def test_circuit_breaker_takes_priority_over_no_trade(self):
        """Circuit breaker detection runs before no-trade detection."""
        fb = FeedbackLoopBreaker(
            circuit_breaker_stuck_bars=5,
            no_trade_window_bars=100,  # No-trade won't trigger early
            min_volatility_threshold=0.001,
        )
        sig = None
        for i in range(20):
            sig = fb.update(
                bars_since_last_trade=i,
                current_volatility=0.02,
                circuit_breakers_tripped=True,
            )
            if sig:
                break
        assert sig.loop_type == "circuit_breaker"


# ── State persistence ──────────────────────────────────────────────────────

class TestStatePersistence:
    def test_save_and_load(self, tmp_path):
        state_file = tmp_path / "fb_state.json"
        fb = FeedbackLoopBreaker(state_file=state_file)
        # Build up some state
        for i in range(10):
            fb.update(i + 50, 0.02, True, recent_sharpe=0.5)
        fb.save_state()

        # Reload
        fb2 = FeedbackLoopBreaker(state_file=state_file)
        assert fb2.load_state() is True
        assert fb2.bars_since_trade == fb.bars_since_trade
        assert fb2.circuit_breaker_tripped == fb.circuit_breaker_tripped
        assert len(fb2.recent_volatilities) == len(fb.recent_volatilities)
        assert len(fb2.recent_sharpes) == len(fb.recent_sharpes)

    def test_load_missing_file(self, tmp_path):
        state_file = tmp_path / "nonexistent.json"
        fb = FeedbackLoopBreaker(state_file=state_file)
        assert fb.load_state() is False

    def test_load_corrupt_file(self, tmp_path):
        state_file = tmp_path / "corrupt.json"
        state_file.write_text("not json at all {{{")
        fb = FeedbackLoopBreaker(state_file=state_file)
        assert fb.load_state() is False

    def test_save_creates_parent_dirs(self, tmp_path):
        state_file = tmp_path / "deep" / "nested" / "state.json"
        fb = FeedbackLoopBreaker(state_file=state_file)
        fb.save_state()
        assert state_file.exists()

    def test_interventions_capped_in_save(self, tmp_path):
        state_file = tmp_path / "fb_state.json"
        fb = FeedbackLoopBreaker(state_file=state_file)
        fb.interventions = [{"i": i} for i in range(200)]
        fb.save_state()
        data = json.loads(state_file.read_text())
        assert len(data["interventions"]) == 100  # Capped at 100
