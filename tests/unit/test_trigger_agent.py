"""
Tests for TriggerAgent - Entry specialist agent.

Covers:
- Initialization (fallback mode, paper mode, live mode)
- decide() method - epsilon-greedy, feasibility gate, confidence floor
- _fallback_strategy() - MA crossover with imbalance tilt + regime adj
- _softmax() / _platt_calibrate() / _q_to_runway()
- _decay_epsilon()
- update_from_trade()
- add_experience() / train_step() / get_training_stats()
- update_platt_params()
"""

import logging
import os
from unittest.mock import patch

import numpy as np

rng = np.random.default_rng(42)

import pytest

from src.agents.trigger_agent import (
    TriggerAgent,
    Q_RUNWAY_MIN,
    Q_RUNWAY_MAX,
    Q_RUNWAY_MAX_Q,
    RUNWAY_SAFETY_FLOOR,
    RUNWAY_SAFETY_CEILING,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestTriggerInit:

    def test_default_init(self):
        ta = TriggerAgent(window=64, n_features=7)
        assert ta.window == 64
        assert ta.n_features == 7
        assert ta.use_torch is False
        assert ta.model is None
        assert ta.ddqn is None  # DDQN only created when training enabled
        assert ta.training_steps == 0

    def test_init_with_training(self):
        ta = TriggerAgent(window=32, n_features=7, enable_training=True)
        assert ta.enable_training is True
        assert ta.buffer is not None

    def test_init_without_training(self):
        ta = TriggerAgent(window=32, n_features=7, enable_training=False)
        assert ta.enable_training is False
        assert ta.buffer is None

    @patch.dict(os.environ, {"PAPER_MODE": "1"})
    def test_init_paper_mode(self):
        ta = TriggerAgent(window=64, n_features=7)
        assert ta.paper_mode is True
        assert ta.epsilon == pytest.approx(1.0)  # Higher epsilon in paper mode

    def test_init_symbol_params(self):
        ta = TriggerAgent(symbol="XAUUSD", timeframe="M5", broker="pepperstone")
        assert ta.symbol == "XAUUSD"
        assert ta.timeframe == "M5"
        assert ta.broker == "pepperstone"


# ---------------------------------------------------------------------------
# decide() basics
# ---------------------------------------------------------------------------

class TestTriggerDecide:

    def test_decide_already_in_position_returns_no_entry(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        action, conf, runway = ta.decide(state, current_position=1)
        assert action == 0
        assert conf == pytest.approx(0.0)
        assert runway == pytest.approx(0.0)

    def test_decide_already_short_returns_no_entry(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        action, conf, runway = ta.decide(state, current_position=-1)
        assert action == 0

    def test_decide_flat_position_returns_valid_action(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.epsilon = 0.0  # No random exploration
        state = rng.standard_normal((64, 7)).astype(np.float32)
        action, conf, runway = ta.decide(state, current_position=0)
        assert action in [0, 1, 2]
        assert 0.0 <= conf <= 1.0
        assert runway >= 0.0

    def test_decide_feasibility_gate_blocks_low_feasibility(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.epsilon = 0.0
        ta.paper_mode = False
        ta.disable_gates = False
        state = rng.standard_normal((64, 7)).astype(np.float32)
        # Set a very low feasibility
        action, conf, runway = ta.decide(state, current_position=0, feasibility=0.0)
        assert action == 0  # Blocked by feasibility gate

    @patch.dict(os.environ, {"PAPER_MODE": "1"})
    def test_decide_paper_mode_epsilon_exploration(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.epsilon = 1.0  # Force exploration
        state = np.zeros((64, 7), dtype=np.float32)
        # Run many trials — exploration now includes NO_ENTRY (0) with 50% weight
        # so all three actions (0=NO_ENTRY, 1=LONG, 2=SHORT) are valid
        seen_actions = set()
        for _ in range(100):
            action, conf, runway = ta.decide(state, current_position=0)
            assert action in [0, 1, 2], f"Unexpected action {action}"
            seen_actions.add(action)
        # With 100 trials and weights [2,1,1] we should see all 3 actions
        assert seen_actions == {0, 1, 2}, f"Expected all actions explored, got {seen_actions}"

    def test_decide_resets_bars_since_trade_when_in_position(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.bars_since_trade = 100
        state = np.zeros((64, 7), dtype=np.float32)
        ta.decide(state, current_position=1)
        assert ta.bars_since_trade == 0


# ---------------------------------------------------------------------------
# Fallback strategy
# ---------------------------------------------------------------------------

class TestFallbackStrategy:

    def test_fallback_empty_state(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((0, 7), dtype=np.float32)
        action = ta._fallback_strategy(state)
        assert action == 0  # NO_ENTRY

    def test_fallback_too_few_features(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((10, 2), dtype=np.float32)
        action = ta._fallback_strategy(state)
        assert action == 0

    def test_fallback_strong_long_signal(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = 0.5  # Strong positive MA diff
        state[-1, 0] = 0.3  # ret1 confirms momentum (required by multi-factor)
        action = ta._fallback_strategy(state)
        assert action == 1  # LONG

    def test_fallback_strong_short_signal(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = -0.5  # Strong negative MA diff
        state[-1, 0] = -0.3  # ret1 confirms downward momentum (required)
        action = ta._fallback_strategy(state)
        assert action == 2  # SHORT

    def test_fallback_neutral_no_entry(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = 0.0  # Neutral
        action = ta._fallback_strategy(state)
        assert action == 0  # NO_ENTRY

    def test_fallback_with_regime_adjustment(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = 0.35  # Medium positive MA diff
        state[-1, 0] = 0.3   # ret1 confirms momentum

        # Without adjustment — should trigger LONG (live threshold=0.3, 0.35>0.3)
        _action_no_adj = ta._fallback_strategy(state, regime_threshold_adj=0.0)
        assert _action_no_adj == 1

        # With positive adjustment (mean-reverting → threshold=0.45, 0.35<0.45 → blocked)
        action_hard = ta._fallback_strategy(state, regime_threshold_adj=0.5)
        assert action_hard == 0

    def test_fallback_with_imbalance_tilt(self):
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = 0.25  # Near threshold
        state[-1, 4] = 0.9   # Strong positive imbalance → tilts LONG easier
        action = ta._fallback_strategy(state)
        # With tilt, the effective threshold for LONG is lower
        assert action in [0, 1]  # May or may not trigger depending on threshold

    def test_fallback_ma_alone_insufficient(self):
        """MA-diff alone (no momentum) returns NO_ENTRY under multi-factor rules."""
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = 0.5  # Strong MA diff — ret1=ret5=0
        action = ta._fallback_strategy(state)
        assert action == 0  # No momentum confirmation → NO_ENTRY

    def test_fallback_vpin_veto_blocks_long(self):
        """Strong opposing VPIN z-score (< −2σ) vetoes a LONG entry."""
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = 0.5   # MA diff → LONG
        state[-1, 0] = 0.3   # ret1 confirms
        state[-1, 5] = -3.0  # Strong sell-side flow: veto
        action = ta._fallback_strategy(state)
        assert action == 0  # Vetoed by VPIN

    def test_fallback_vpin_veto_blocks_short(self):
        """Strong opposing VPIN z-score (> +2σ) vetoes a SHORT entry."""
        ta = TriggerAgent(window=64, n_features=7)
        state = np.zeros((64, 7), dtype=np.float32)
        state[-1, 2] = -0.5   # MA diff → SHORT
        state[-1, 0] = -0.3   # ret1 confirms
        state[-1, 5] = 3.0    # Strong buy-side flow: veto SHORT
        action = ta._fallback_strategy(state)
        assert action == 0  # Vetoed by VPIN

    def test_fallback_decide_confidence_increases_with_factors(self):
        """Confidence increases as more momentum and VPIN factors confirm direction."""
        ta = TriggerAgent(window=64, n_features=7)
        # ret1 only (1 factor)
        s1 = np.zeros((64, 7), dtype=np.float32)
        s1[-1, 2] = 0.5
        s1[-1, 0] = 0.3
        # ret1 + ret5 (2 factors)
        s2 = np.zeros((64, 7), dtype=np.float32)
        s2[-1, 2] = 0.5
        s2[-1, 0] = 0.3
        s2[-1, 1] = 0.2
        # ret1 + ret5 + vpin agrees (3 factors)
        s3 = np.zeros((64, 7), dtype=np.float32)
        s3[-1, 2] = 0.5
        s3[-1, 0] = 0.3
        s3[-1, 1] = 0.2
        s3[-1, 5] = 1.0  # positive VPIN → agrees with LONG
        _, c1, _ = ta._fallback_decide(s1)
        _, c2, _ = ta._fallback_decide(s2)
        _, c3, _ = ta._fallback_decide(s3)
        assert c1 < c2 < c3
        assert 0.5 <= c1 <= 0.85
        assert 0.5 <= c3 <= 0.85

    def test_fallback_decide_runway_scales_with_vol(self):
        """Predicted runway is larger in high-volatility regimes."""
        ta = TriggerAgent(window=64, n_features=7)

        def _state(vol_z: float) -> np.ndarray:
            s = np.zeros((64, 7), dtype=np.float32)
            s[-1, 2] = 0.5
            s[-1, 0] = 0.3
            s[-1, 3] = vol_z  # z-scored rolling volatility
            return s

        _, _, runway_quiet = ta._fallback_decide(_state(-2.0))
        _, _, runway_hot   = ta._fallback_decide(_state(2.0))
        assert runway_hot > runway_quiet


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------

class TestTriggerUtils:

    def test_softmax_basic(self):
        ta = TriggerAgent(window=64, n_features=7)
        q = np.array([1.0, 2.0, 3.0])
        probs = ta._softmax(q)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert probs[2] > probs[1] > probs[0]

    def test_softmax_equal_values(self):
        ta = TriggerAgent(window=64, n_features=7)
        q = np.array([1.0, 1.0, 1.0])
        probs = ta._softmax(q)
        assert abs(probs.sum() - 1.0) < 1e-6
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3], atol=1e-6)

    def test_softmax_large_values(self):
        ta = TriggerAgent(window=64, n_features=7)
        q = np.array([1000.0, 1001.0, 1002.0])
        probs = ta._softmax(q)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert np.all(np.isfinite(probs))

    def test_platt_calibrate_midpoint(self):
        ta = TriggerAgent(window=64, n_features=7)
        result = ta._platt_calibrate(0.5)
        assert 0.0 < result < 1.0

    def test_platt_calibrate_extreme_low(self):
        ta = TriggerAgent(window=64, n_features=7)
        result = ta._platt_calibrate(0.0)
        assert 0.0 < result < 1.0  # Should handle edge case

    def test_platt_calibrate_extreme_high(self):
        ta = TriggerAgent(window=64, n_features=7)
        result = ta._platt_calibrate(1.0)
        assert 0.0 < result < 1.0

    def test_q_to_runway_minimum(self):
        ta = TriggerAgent(window=64, n_features=7)
        assert ta._q_to_runway(-1.0) == Q_RUNWAY_MIN
        assert ta._q_to_runway(0.0) == Q_RUNWAY_MIN

    def test_q_to_runway_maximum(self):
        ta = TriggerAgent(window=64, n_features=7)
        assert ta._q_to_runway(3.0) == Q_RUNWAY_MAX
        assert ta._q_to_runway(10.0) == Q_RUNWAY_MAX

    def test_q_to_runway_interpolation(self):
        ta = TriggerAgent(window=64, n_features=7)
        mid = ta._q_to_runway(1.5)
        expected = Q_RUNWAY_MIN + (1.5 / Q_RUNWAY_MAX_Q) * (Q_RUNWAY_MAX - Q_RUNWAY_MIN)
        assert abs(mid - expected) < 1e-9

    def test_decay_epsilon(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.epsilon = 0.5
        ta.epsilon_end = 0.01
        ta.epsilon_decay = 0.99
        ta._decay_epsilon()
        assert abs(ta.epsilon - 0.495) < 1e-6

    def test_decay_epsilon_minimum(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.epsilon = 0.001
        ta.epsilon_end = 0.01
        ta.epsilon_decay = 0.5
        ta._decay_epsilon()
        assert ta.epsilon == pytest.approx(0.01)  # Should not go below epsilon_end


# ---------------------------------------------------------------------------
# Update from trade
# ---------------------------------------------------------------------------

class TestTriggerUpdateFromTrade:

    def test_update_from_trade_positive_runway(self):
        ta = TriggerAgent(window=64, n_features=7)
        # Should not crash, just logs
        ta.update_from_trade(actual_mfe=0.003, predicted_runway=0.002)

    def test_update_from_trade_zero_runway(self):
        ta = TriggerAgent(window=64, n_features=7)
        ta.update_from_trade(actual_mfe=0.003, predicted_runway=0.0)
        # Zero predicted → skip logging


# ---------------------------------------------------------------------------
# Platt parameter updates
# ---------------------------------------------------------------------------

class TestPlattUpdate:

    def test_update_platt_params_training_enabled(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=True)
        old_a = ta.platt_a
        old_b = ta.platt_b
        ta.update_platt_params(predicted_prob=0.7, actual_outcome=1.0)
        # Parameters should have changed
        assert ta.platt_a != old_a or ta.platt_b != old_b

    def test_update_platt_params_training_disabled(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=False)
        old_a = ta.platt_a
        old_b = ta.platt_b
        ta.update_platt_params(predicted_prob=0.7, actual_outcome=1.0)
        # Should not change when training disabled
        assert ta.platt_a == old_a
        assert ta.platt_b == old_b


# ---------------------------------------------------------------------------
# Experience buffer and training
# ---------------------------------------------------------------------------

class TestTriggerTraining:

    def test_add_experience_no_buffer(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=False)
        state = np.zeros((64, 7), dtype=np.float32)
        # Should not crash
        ta.add_experience(state, action=1, reward=0.5, next_state=state, done=True)

    def test_train_step_no_training(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=False)
        assert ta.train_step() is None

    def test_train_step_insufficient_experiences(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=True)
        assert ta.train_step() is None  # No experiences added

    def test_get_training_stats_disabled(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=False)
        stats = ta.get_training_stats()
        assert stats == {"enabled": False}

    def test_get_training_stats_enabled(self):
        ta = TriggerAgent(window=64, n_features=7, enable_training=True)
        stats = ta.get_training_stats()
        assert stats["enabled"] is True
        assert stats["training_steps"] == 0
        assert stats["buffer_size"] >= 0


# ---------------------------------------------------------------------------
# EWMA Runway Calibration (Enhancement A)
# ---------------------------------------------------------------------------

class TestEWMARunwayCalibration:

    def test_initial_state_uses_static_mapping(self):
        """Before any trades, _q_to_runway should use static linear mapping."""
        ta = TriggerAgent(window=64, n_features=7)
        assert ta._q_to_runway(0.0) == Q_RUNWAY_MIN
        assert ta._q_to_runway(Q_RUNWAY_MAX_Q) == Q_RUNWAY_MAX
        mid_q = Q_RUNWAY_MAX_Q / 2.0
        expected = Q_RUNWAY_MIN + (mid_q / Q_RUNWAY_MAX_Q) * (Q_RUNWAY_MAX - Q_RUNWAY_MIN)
        assert ta._q_to_runway(mid_q) == pytest.approx(expected)

    def test_calibration_updates_after_trade(self):
        """After enough trades in a bucket, EWMA should be populated."""
        from src.agents.trigger_agent import RUNWAY_CAL_MIN_SAMPLES
        ta = TriggerAgent(window=64, n_features=7)
        # Simulate entries with Q=1.0 (bucket 1: [0.6, 1.2])
        for i in range(RUNWAY_CAL_MIN_SAMPLES + 1):
            ta._last_entry_q = 1.0  # Simulate Q at entry
            ta._q_to_runway(1.0)  # Sets _last_entry_q
            # Simulate trade close with actual MFE (fractional)
            ta._update_runway_calibration(0.003)  # 0.3% MFE
        # Now the calibrated value should be used
        calibrated = ta._calibrated_runway(1.0)
        assert calibrated is not None
        assert calibrated == pytest.approx(0.003, abs=0.001)

    def test_insufficient_samples_returns_none(self):
        """With fewer samples than threshold, calibration returns None."""
        ta = TriggerAgent(window=64, n_features=7)
        ta._last_entry_q = 1.0
        ta._update_runway_calibration(0.003)
        assert ta._calibrated_runway(1.0) is None  # Only 1 sample

    def test_ewma_adapts_over_time(self):
        """EWMA should track changing MFE values."""
        from src.agents.trigger_agent import RUNWAY_CAL_MIN_SAMPLES, RUNWAY_CAL_ALPHA
        ta = TriggerAgent(window=64, n_features=7)
        # Fill bucket with 0.002 first
        for _ in range(RUNWAY_CAL_MIN_SAMPLES + 1):
            ta._last_entry_q = 2.0
            ta._update_runway_calibration(0.002)
        val_before = ta._runway_cal_ewma[ta._q_bucket(2.0)]
        # Now feed higher MFE values
        for _ in range(5):
            ta._last_entry_q = 2.0
            ta._update_runway_calibration(0.004)
        val_after = ta._runway_cal_ewma[ta._q_bucket(2.0)]
        assert val_after > val_before  # Should have moved up

    def test_q_bucket_mapping(self):
        """Q-values should map to correct buckets."""
        assert TriggerAgent._q_bucket(0.0) == 0
        assert TriggerAgent._q_bucket(0.5) == 0
        assert TriggerAgent._q_bucket(0.7) == 1
        assert TriggerAgent._q_bucket(1.5) == 2
        assert TriggerAgent._q_bucket(2.1) == 3
        assert TriggerAgent._q_bucket(2.9) == 4
        assert TriggerAgent._q_bucket(3.0) == 4  # Edge: maps to last bucket

    def test_update_from_trade_with_entry_price(self):
        """update_from_trade should convert abs MFE to fractional for calibration."""
        ta = TriggerAgent(window=64, n_features=7)
        ta._last_entry_q = 1.5
        ta._q_to_runway(1.5)  # Sets _last_entry_q
        # actual_mfe=10.0 at entry_price=5000 → fractional = 0.002
        ta.update_from_trade(
            actual_mfe=10.0,
            predicted_runway=0.002,
            entry_price=5000.0,
        )
        bucket = TriggerAgent._q_bucket(1.5)
        assert ta._runway_cal_counts[bucket] == 1
        assert ta._runway_cal_ewma[bucket] == pytest.approx(0.002)

    def test_calibrated_value_clipped_to_safety_bounds(self):
        """Calibrated runway should be clipped to [RUNWAY_SAFETY_FLOOR, RUNWAY_SAFETY_CEILING]."""
        from src.agents.trigger_agent import RUNWAY_CAL_MIN_SAMPLES
        ta = TriggerAgent(window=64, n_features=7)
        # Feed extreme (very high) MFE — should be clipped to RUNWAY_SAFETY_CEILING
        for _ in range(RUNWAY_CAL_MIN_SAMPLES + 1):
            ta._last_entry_q = 0.3
            ta._update_runway_calibration(0.10)  # 10% — way above safety ceiling
        result = ta._q_to_runway(0.3)
        assert result <= RUNWAY_SAFETY_CEILING
        assert result >= RUNWAY_SAFETY_FLOOR
        # Also verify it can exceed old Q_RUNWAY_MAX when data supports it
        ta2 = TriggerAgent(window=64, n_features=7)
        for _ in range(RUNWAY_CAL_MIN_SAMPLES + 1):
            ta2._last_entry_q = 2.5
            ta2._update_runway_calibration(0.008)  # 0.8% > old Q_RUNWAY_MAX (0.5%)
        result2 = ta2._q_to_runway(2.5)
        assert result2 > Q_RUNWAY_MAX  # RL-learned value exceeds old static ceiling
        assert result2 == pytest.approx(0.008, abs=0.002)

    def test_calibrated_global_runway(self):
        """_calibrated_global_runway returns weighted average across calibrated buckets."""
        from src.agents.trigger_agent import RUNWAY_CAL_MIN_SAMPLES
        ta = TriggerAgent(window=64, n_features=7)
        # No data → None
        assert ta._calibrated_global_runway() is None
        # Fill bucket 0 with 0.003
        for _ in range(RUNWAY_CAL_MIN_SAMPLES + 1):
            ta._last_entry_q = 0.3
            ta._update_runway_calibration(0.003)
        # Fill bucket 3 with 0.006
        for _ in range(RUNWAY_CAL_MIN_SAMPLES + 1):
            ta._last_entry_q = 2.0
            ta._update_runway_calibration(0.006)
        global_avg = ta._calibrated_global_runway()
        assert global_avg is not None
        assert 0.003 < global_avg < 0.006  # Weighted average between the two

    def test_mae_ewma_tracked(self):
        """MAE EWMA should be updated alongside MFE."""
        from src.agents.trigger_agent import RUNWAY_CAL_MIN_SAMPLES
        ta = TriggerAgent(window=64, n_features=7)
        for _ in range(RUNWAY_CAL_MIN_SAMPLES + 1):
            ta._last_entry_q = 1.0
            ta._update_runway_calibration(0.003, actual_mae_frac=0.001)
        bucket = TriggerAgent._q_bucket(1.0)
        assert ta._runway_cal_mae_ewma[bucket] == pytest.approx(0.001, abs=0.0005)

    def test_fallback_runway_uses_global_calibration(self):
        """_fallback_runway should use calibrated global average when available."""
        from src.agents.trigger_agent import RUNWAY_CAL_MIN_SAMPLES, PREDICTED_RUNWAY_FALLBACK
        ta = TriggerAgent(window=64, n_features=7)
        # Before calibration: should use vol-scaled PREDICTED_RUNWAY_FALLBACK
        cold_val = ta._fallback_runway(0.0)
        assert cold_val == pytest.approx(PREDICTED_RUNWAY_FALLBACK * 1.1, abs=0.001)
        # Fill calibration with known values (much higher than fallback)
        for _ in range(RUNWAY_CAL_MIN_SAMPLES + 1):
            ta._last_entry_q = 1.0
            ta._update_runway_calibration(0.008)
        # Now fallback should use calibrated global average
        warm_val = ta._fallback_runway(0.0)
        assert warm_val > cold_val  # Calibrated average (0.008) >> old fallback (0.0015)
