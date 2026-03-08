#!/usr/bin/env python3
"""
Trigger Agent - Entry Specialist (Phase 3)
==========================================
Dual-agent architecture component for trade entry decisions.

Responsibilities:
- Identify high-quality entry opportunities
- Predict runway (expected MFE)
- Output: entry signal (LONG/SHORT/NONE) + confidence + predicted runway

Reward Function:
- Runway utilization: actual_MFE / predicted_runway
- Entry quality bonus: Did trade achieve meaningful MFE?
- False signal penalty: No MFE achieved after entry

From MASTER_HANDBOOK.md Section 2.2: Dual-Agent Architecture

Phase 3.5: Online Learning
- ExperienceBuffer integration for continuous improvement
- train_step() for DDQN updates
"""

import logging
import os
import random
from typing import NamedTuple

import numpy as np

from src.agents.agent_training_mixin import AgentTrainingMixin, softmax
from src.constants import (
    GAMMA,
    GRAD_CLIP_NORM,
    L2_WEIGHT,
    LEARNING_RATE,
    STATE_WINDOW_SIZE,
    TAU,
    TRIGGER_BUFFER_CAPACITY,
)
from src.core.ddqn_network import DDQNNetwork
from src.persistence.learned_parameters import LearnedParametersManager
from src.utils.experience_buffer import ExperienceBuffer

LOG = logging.getLogger(__name__)


MIN_FEATURE_COLS = 3
IMBALANCE_INDEX = 4
VOL_Z_INDEX = 3           # state array column index for vol_z
VPIN_Z_INDEX = 5          # state array column index for vpin_z
TILT_SCALE = 0.1
PAPER_EPSILON = 0.15
PAPER_BASE_THRESHOLD = 0.15
LIVE_BASE_THRESHOLD = 0.3
_UTILIZATION_BAD_THRESHOLD: float = 0.3   # utilization below this is a bad entry
_UTILIZATION_OUTLIER_LOW: float = 0.2     # below this is an outlier (too poor)
_UTILIZATION_OUTLIER_HIGH: float = 2.0    # above this is an outlier (excessive)
PREDICTED_RUNWAY_FALLBACK = 0.0015
Q_RUNWAY_MIN = 0.0010
Q_RUNWAY_MAX = 0.0050
Q_RUNWAY_MAX_Q = 3.0
FALLBACK_VOL_SCALE_QUIET: float = 0.70  # Runway scale during below-average volatility
FALLBACK_VOL_SCALE_HOT: float = 1.50    # Runway scale during above-average volatility


class _EconomicsGateParams(NamedTuple):
    expected_gain: float
    expected_loss: float
    friction_cost: float
    breakeven_prob: float


class TriggerAgent(AgentTrainingMixin):
    """
    Entry specialist agent - decides WHEN and WHICH DIRECTION to enter.

    Philosophy: "Find trades with runway to harvest"

    Action Space:
        0 = NO_ENTRY (stay flat)
        1 = LONG (buy)
        2 = SHORT (sell)

    State: Market features (7-dim by default)
        - ret1: 1-bar return
        - ret5: 5-bar return
        - ma_diff: MA fast/slow difference
        - vol: 20-bar volatility
        - imbalance: Order book imbalance [-1, 1]
        - vpin_z: VPIN z-score
        - depth_ratio: Bid+ask depth relative to median

    Output:
        - action: 0/1/2 (NO_ENTRY/LONG/SHORT)
        - confidence: [0, 1] from softmax probabilities
        - predicted_runway: Expected MFE in price units
    """

    _AGENT_TAG = "TRIGGER"

    def __init__(  # noqa: PLR0913
        self,
        window: int = STATE_WINDOW_SIZE,
        n_features: int = 7,
        enable_training: bool = False,
        symbol: str = "XAUUSD",  # Instrument-agnostic: default for tests/demos
        timeframe: str = "M15",
        broker: str = "default",
        param_manager: LearnedParametersManager | None = None,
        timeframe_minutes: int = 5,
        buffer_capacity: int = TRIGGER_BUFFER_CAPACITY,
    ):
        """
        Initialize Trigger Agent.

        Args:
            window: Lookback window for state
            n_features: Number of input features
            enable_training: Enable online learning (Phase 3.5)
        """
        self.window = window
        self.n_features = n_features
        self.use_torch = False
        self.model = None
        self.torch = None
        self.symbol = symbol
        self.timeframe = timeframe
        self.timeframe_minutes = timeframe_minutes
        self.broker = broker
        self.param_manager = param_manager

        # Paper mode settings - NO GATING in training
        self.paper_mode = os.environ.get("PAPER_MODE", "0") == "1"
        self.disable_gates = os.environ.get("DISABLE_GATES", "0") == "1"

        # Epsilon-greedy exploration for training
        self.epsilon = float(os.environ.get("EPSILON_START", "1.0" if self.paper_mode else "0.05"))
        self.epsilon_end = float(os.environ.get("EPSILON_END", "0.1" if self.paper_mode else "0.01"))
        self.epsilon_decay = float(os.environ.get("EPSILON_DECAY", "0.998"))
        self.exploration_boost = float(os.environ.get("EXPLORATION_BOOST", "0.5" if self.paper_mode else "0.0"))
        self.force_exploration = os.environ.get("FORCE_EXPLORATION", "0") == "1"
        self.bars_since_trade = 0
        self.max_bars_inactive = int(os.environ.get("MAX_BARS_INACTIVE", "10" if self.paper_mode else "1000"))

        # Phase 3.5: Experience replay buffer
        # Capacity sized to ~20 days at ~100 trades/day (staleness halflife = 1 day,
        # so >2,000 entries are already near-zero weight and waste memory/diversity).
        self.enable_training = enable_training
        self.buffer = ExperienceBuffer(capacity=buffer_capacity, timeframe_minutes=self.timeframe_minutes) if enable_training else None
        self.min_experiences = 32  # 1 batch – start training as soon as we have enough
        self.batch_size = 64
        self.training_steps = 0
        self.last_state = None  # Track state for experience creation
        self.last_action = None

        # Phase 3.5: DDQN network for online learning (numpy-based, no PyTorch required)
        # This is the actual trainable network - separate from the PyTorch model loaded from disk
        self.ddqn = (
            DDQNNetwork(
                state_dim=window * n_features,  # Flattened state vector
                n_actions=3,  # three actions: NO_ENTRY, LONG, SHORT
                learning_rate=LEARNING_RATE,
                gamma=GAMMA,
                tau=TAU,
                l2_weight=L2_WEIGHT,
                grad_clip_norm=GRAD_CLIP_NORM,
            )
            if enable_training
            else None
        )

        # Phase 2: Platt calibration for probability estimates (online learning)
        # Converts raw scores to calibrated probabilities: p = 1/(1 + exp(A*score + B))
        self.platt_a = 1.0  # Slope parameter (learned online)
        self.platt_b = 0.0  # Intercept parameter (learned online)
        self.platt_lr = 0.01  # Learning rate for Platt updates

        # Phase 2: Gating strategy
        # Training mode: NO GATES - pure exploration, learn through rewards
        # Live mode: LEARNED GATES - confidence floor from trained model
        if self.disable_gates or self.paper_mode:
            self.feasibility_threshold = 0.0
            self.confidence_floor = 0.0
        else:
            # Live trading: use learned gating
            self.feasibility_threshold, _ = self._resolve_gate_value(
                env_key="FEAS_THRESHOLD", param_name="feasibility_threshold", fallback=0.5
            )
            self.confidence_floor, _ = self._resolve_gate_value(
                env_key="CONFIDENCE_FLOOR", param_name="confidence_floor", fallback=0.55
            )

        # Log consolidated initialization
        mode_str = "TRAINING" if (self.disable_gates or self.paper_mode) else "LIVE"
        training_str = f"online_learn={enable_training}" if enable_training else "no_training"
        LOG.info("[TRIGGER] Init: %s ε=%.2f→%.2f decay=%.4f | %s",
            mode_str, self.epsilon, self.epsilon_end, self.epsilon_decay, training_str)

        # Try to load model if path specified
        model_path = os.environ.get("DDQN_TRIGGER_MODEL", "").strip()
        if model_path:
            self._load_model(model_path)

    def _resolve_gate_value(self, env_key: str, param_name: str, fallback: float) -> tuple[float, str]:
        """Resolve gate thresholds via env override → learned params → fallback."""
        env_val = os.environ.get(env_key)
        if env_val is not None and env_val != "":
            try:
                return float(env_val), "ENV"
            except ValueError:
                LOG.warning("[TRIGGER] Invalid %s env value '%s' - falling back", env_key, env_val)
        value = self._get_param(param_name, fallback)
        source = "LP" if self.param_manager else "DEFAULT"
        return value, source

    def _get_param(self, name: str, default: float) -> float:
        if not self.param_manager:
            return default
        try:
            value = self.param_manager.get(
                self.symbol, name, timeframe=self.timeframe, broker=self.broker, default=default
            )
            return float(value)
        except Exception as exc:
            LOG.debug(
                "[TRIGGER] Failed to load %s from LearnedParameters (%s) - using default %.2f",
                name,
                exc,
                default,
            )
            return default

    def _load_model(self, model_path: str):
        """Load PyTorch DDQN model for trigger agent."""
        try:
            import torch  # noqa: PLC0415

            from src.core.ddqn_network import Conv1dQNet  # noqa: PLC0415

            self.torch = torch
            # Saved weights were trained with pool-to-1 architecture; use
            # temporal_pool_size=1 here so load_state_dict succeeds.
            self.model = Conv1dQNet(n_features=self.n_features, n_actions=3, temporal_pool_size=1)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            self.model.eval()
            self.use_torch = True
            LOG.info("[TRIGGER] Loaded DDQN model: %s", model_path)
        except (OSError, ImportError, RuntimeError) as e:
            LOG.warning("[TRIGGER] Failed to load model: %s. Using fallback.", e)
            self.use_torch = False

    def _try_training_decision(self) -> tuple[int, float, float] | None:
        """
        Attempt an epsilon-greedy or forced-exploration decision (training / paper mode).

        Returns the action tuple when a training decision is made, or None when
        the caller should continue to the live-mode logic.
        """
        # Epsilon-greedy: randomly explore with probability ε
        # Include NO_ENTRY (action=0) with 50% weight so the agent learns when
        # NOT to enter — without this, exploration never discovers staying flat
        # and the replay buffer is 100% entry samples (severe class imbalance).
        if random.random() < self.epsilon:
            action = random.choices([0, 1, 2], weights=[2, 1, 1])[0]
            LOG.info(
                "[TRIGGER] EXPLORE: random action=%d (ε=%.3f, bars_flat=%d)",
                action, self.epsilon, self.bars_since_trade,
            )
            self._decay_epsilon()
            if action != 0:
                self.bars_since_trade = 0
            self.last_action = action
            if action == 0:
                return 0, 0.0, 0.0
            return action, 0.5, PREDICTED_RUNWAY_FALLBACK

        # Forced entry when idle for too many bars
        if self.force_exploration and self.bars_since_trade >= self.max_bars_inactive:
            action = random.choice([1, 2])
            LOG.info("[TRIGGER] FORCED ENTRY after %d bars flat: action=%d", self.bars_since_trade, action)
            self.bars_since_trade = 0
            self.last_action = action
            return action, 0.5, PREDICTED_RUNWAY_FALLBACK

        return None  # Carry on to normal model decision

    def _decide_numpy_path(
        self, state: np.ndarray, regime_threshold_adj: float, friction_cost: float
    ) -> tuple[int, float, float]:
        """決 Decision path for the numpy-based DDQN or MA-crossover fallback."""
        if self.ddqn is not None and self.enable_training and self.training_steps > 0:
            flat_state = state.reshape(1, -1).astype(np.float64)
            q_values = self.ddqn.predict(flat_state).flatten()
            action = int(np.argmax(q_values))
            probs = self._softmax(q_values)
            confidence = float(probs[action])
            gross_runway = self._q_to_runway(float(q_values[action]))
            predicted_runway = max(0.0, gross_runway - friction_cost)
            LOG.debug(
                "[TRIGGER] DDQN decision: Q=%s, action=%d, conf=%.3f, runway=%.4f",
                q_values, action, confidence, predicted_runway,
            )
        else:
            action, confidence, predicted_runway = self._fallback_decide(state, regime_threshold_adj)
            if action == 0:
                return 0, 0.0, 0.0

        self.last_action = action

        if not self.paper_mode and confidence < self.confidence_floor:
            LOG.debug("[TRIGGER] BLOCKED by confidence floor: %.3f < %.3f", confidence, self.confidence_floor)
            return 0, confidence, 0.0

        self._decay_epsilon()
        return action, confidence, predicted_runway

    def decide(  # noqa: PLR0911, PLR0913
        self,
        state: np.ndarray,
        current_position: int = 0,
        regime_threshold_adj: float = 0.0,  # Phase 3.4: Regime-based adjustment
        feasibility: float = 1.0,  # Phase 2: Path geometry feasibility [0, 1]
        expected_gain: float = 0.002,  # Phase 2: Expected gain (G)
        expected_loss: float = 0.001,  # Phase 2: Expected loss (L)
        friction_cost: float = 0.0002,  # Phase 2: Friction costs (K) - spread + slippage
    ) -> tuple[int, float, float]:
        """
        Decide entry action based on current market state.

        Args:
            state: Normalized state features (window, n_features)
            current_position: Current position (-1/0/+1 for SHORT/FLAT/LONG)
            regime_threshold_adj: Phase 3.4 adjustment from RegimeDetector
            feasibility: Phase 2 path geometry feasibility score [0, 1]
            expected_gain: Expected gain (G) for economics threshold
            expected_loss: Expected loss (L) for economics threshold
            friction_cost: Friction costs (K) - spread, slippage, commissions

        Returns:
            (action, confidence, predicted_runway)
            - action: 0=NO_ENTRY, 1=LONG, 2=SHORT
            - confidence: [0, 1] Platt-calibrated probability
            - predicted_runway: Expected MFE as percentage (e.g., 0.002 = 0.2% of entry price)
        """
        self.bars_since_trade += 1

        # Always record the state we see — needed for experience replay
        # regardless of which gate or decision path fires.
        self.last_state = state.copy()
        LOG.debug(
            "[TRIGGER-DIAG] decide() called: state_shape=%s, setting last_state for experience buffers",
            state.shape if hasattr(state, 'shape') else 'unknown',
        )

        if self._should_block_for_position(current_position):
            self.bars_since_trade = 0
            return 0, 0.0, 0.0

        result = self._maybe_training_decision()
        if result is not None:
            return result

        if self._feasibility_gate_blocked(feasibility):
            return 0, 0.0, 0.0

        if not self._use_torch_inference():
            return self._decide_numpy_path(state, regime_threshold_adj, friction_cost)

        return self._decide_torch_path(state, expected_gain, expected_loss, friction_cost)

    def _should_block_for_position(self, current_position: int) -> bool:
        """Return True if a position is already open."""
        return current_position != 0

    def _maybe_training_decision(self) -> tuple[int, float, float] | None:
        """Return a training-mode decision when applicable."""
        if self.paper_mode or self.disable_gates:
            return self._try_training_decision()
        return None

    def _feasibility_gate_blocked(self, feasibility: float) -> bool:
        """Return True if the feasibility gate blocks entry."""
        if self.disable_gates or feasibility >= self.feasibility_threshold:
            return False
        LOG.debug("[TRIGGER] BLOCKED by feasibility gate: %.3f < %.3f", feasibility, self.feasibility_threshold)
        return True

    def _use_torch_inference(self) -> bool:
        """Return True when torch inference is available and enabled."""
        return bool(self.use_torch and self.torch is not None and self.model is not None)

    def _decide_torch_path(
        self,
        state: np.ndarray,
        expected_gain: float,
        expected_loss: float,
        friction_cost: float,
    ) -> tuple[int, float, float]:
        """Decision path when torch inference is enabled."""
        torch = self.torch
        model = self.model
        if torch is None or model is None:
            return self._decide_numpy_path(state, 0.0, friction_cost)

        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).float()
            q_values = model(t).squeeze(0).numpy()
            action = int(q_values.argmax())
            probs = self._softmax(q_values)
            raw_prob = float(probs[action])
            calibrated_prob = self._platt_calibrate(raw_prob)

            if self._confidence_gate_blocked(calibrated_prob):
                return 0, calibrated_prob, 0.0

            breakeven_prob = self._calc_breakeven_prob(expected_gain, expected_loss, friction_cost)
            econ_params = _EconomicsGateParams(
                expected_gain=expected_gain,
                expected_loss=expected_loss,
                friction_cost=friction_cost,
                breakeven_prob=breakeven_prob,
            )
            if self._economics_gate_blocked(action, calibrated_prob, econ_params):
                return 0, 0.0, 0.0

            q_max = q_values[action]
            gross_runway = self._q_to_runway(q_max)
            predicted_runway = max(0.0, gross_runway - friction_cost)

            LOG.debug(
                "[TRIGGER] Q=%s action=%d raw_p=%.3f calib_p=%.3f be=%.3f gross=%.4f K=%.4f net=%.4f",
                q_values, action, raw_prob, calibrated_prob,
                breakeven_prob, gross_runway, friction_cost, predicted_runway,
            )

            self._decay_epsilon()
            return action, calibrated_prob, predicted_runway

    def _confidence_gate_blocked(self, calibrated_prob: float) -> bool:
        """Return True if confidence floor blocks entry."""
        if self.paper_mode or calibrated_prob >= self.confidence_floor:
            return False
        LOG.debug("[TRIGGER] BLOCKED by confidence floor: %.3f < %.3f", calibrated_prob, self.confidence_floor)
        return True

    def _calc_breakeven_prob(self, expected_gain: float, expected_loss: float, friction_cost: float) -> float:
        return (expected_loss + friction_cost) / (expected_gain + expected_loss + 1e-9)

    def _economics_gate_blocked(
        self,
        action: int,
        calibrated_prob: float,
        params: _EconomicsGateParams,
    ) -> bool:
        """Return True if economics gate blocks entry."""
        if self.paper_mode or action == 0 or calibrated_prob >= params.breakeven_prob:
            return False
        LOG.debug(
            "[TRIGGER] BLOCKED by economics: p=%.3f < breakeven=%.3f (G=%.4f, L=%.4f, K=%.4f)",
            calibrated_prob,
            params.breakeven_prob,
            params.expected_gain,
            params.expected_loss,
            params.friction_cost,
        )
        return True

    def _decay_epsilon(self):
        """Decay epsilon for exploration schedule."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _fallback_decide(
        self, state: np.ndarray, regime_threshold_adj: float = 0.0
    ) -> tuple[int, float, float]:
        """Multi-factor fallback entry decision with dynamic confidence and vol-scaled runway.

        Improvements over the legacy single-factor MA-diff crossover:
        1. CONFIRM  — at least one of ret1/ret5 must align with direction.
                      Prevents entries on brief MA-diff spikes with no momentum.
        2. FILTER   — strong opposing VPIN z-score (> ±2σ) vetoes the entry.
                      Protects against entering into toxic institutional flow.
        3. DYN CONF — 0.55 base + 0.10 per momentum factor + 0.05 if VPIN agrees.
                      Range [0.55, 0.85]; previously a flat 0.6.
        4. VOL RWWY — PREDICTED_RUNWAY_FALLBACK × vol-regime multiplier.
                      Quiet markets → tighter runway; volatile → wider.

        Returns:
            (action, confidence, predicted_runway)
        """
        state = self._normalize_fallback_state(state)
        if self._fallback_state_invalid(state):
            return 0, 0.0, 0.0

        ma_diff, ret1, ret5, vol_z, imbalance, vpin_z = self._extract_fallback_features(state)
        adjusted_threshold, tilt = self._resolve_fallback_threshold(imbalance, regime_threshold_adj)
        direction = self._resolve_fallback_direction(ma_diff, adjusted_threshold, tilt)
        if direction == 0:
            return 0, 0.0, 0.0

        ret1_ok, ret5_ok = self._momentum_flags(direction, ret1, ret5)
        if not (ret1_ok or ret5_ok):
            return 0, 0.0, 0.0

        if not self._vpin_allows(direction, vpin_z):
            return 0, 0.0, 0.0

        confidence = self._fallback_confidence(direction, ret1_ok, ret5_ok, vpin_z)
        predicted_runway = self._fallback_runway(vol_z)
        action = 1 if direction == 1 else 2
        return action, confidence, predicted_runway

    def _normalize_fallback_state(self, state: np.ndarray) -> np.ndarray:
        """Ensure fallback state is a numpy array."""
        if isinstance(state, np.ndarray):
            return state
        return np.array(state)

    def _fallback_state_invalid(self, state: np.ndarray) -> bool:
        """Return True if fallback state is missing required columns."""
        return state.shape[0] == 0 or state.shape[1] < MIN_FEATURE_COLS

    def _extract_fallback_features(self, state: np.ndarray) -> tuple[float, float, float, float, float, float]:
        """Extract normalized features from the latest bar."""
        ma_diff = float(state[-1, 2])
        ret1 = float(state[-1, 0])
        ret5 = float(state[-1, 1])
        vol_z = float(state[-1, VOL_Z_INDEX]) if state.shape[1] > VOL_Z_INDEX else 0.0
        imbalance = float(state[-1, IMBALANCE_INDEX]) if state.shape[1] > IMBALANCE_INDEX else 0.0
        vpin_z = float(state[-1, VPIN_Z_INDEX]) if state.shape[1] > VPIN_Z_INDEX else 0.0
        return ma_diff, ret1, ret5, vol_z, imbalance, vpin_z

    def _resolve_fallback_threshold(self, imbalance: float, regime_threshold_adj: float) -> tuple[float, float]:
        """Return (adjusted_threshold, tilt) for fallback decision."""
        paper_mode = os.environ.get("PAPER_MODE") == "1"
        seed_threshold = PAPER_BASE_THRESHOLD if paper_mode else LIVE_BASE_THRESHOLD
        base_threshold, _ = self._resolve_gate_value(
            env_key="FALLBACK_THRESHOLD",
            param_name="fallback_base_threshold",
            fallback=seed_threshold,
        )
        tilt = imbalance * TILT_SCALE
        adjusted_threshold = max(base_threshold * (1.0 + regime_threshold_adj), 0.0)
        return adjusted_threshold, tilt

    def _resolve_fallback_direction(self, ma_diff: float, threshold: float, tilt: float) -> int:
        """Return 1 for LONG, -1 for SHORT, or 0 for no signal."""
        ma_long = ma_diff > (threshold - tilt)
        ma_short = ma_diff < -(threshold + tilt)
        if not ma_long and not ma_short:
            return 0
        return 1 if ma_long else -1

    def _momentum_flags(self, direction: int, ret1: float, ret5: float) -> tuple[bool, bool]:
        """Return momentum alignment flags for ret1 and ret5."""
        momentum_min = 0.05
        ret1_ok = (direction == 1 and ret1 > momentum_min) or (direction == -1 and ret1 < -momentum_min)
        ret5_ok = (direction == 1 and ret5 > momentum_min) or (direction == -1 and ret5 < -momentum_min)
        return ret1_ok, ret5_ok

    def _vpin_allows(self, direction: int, vpin_z: float) -> bool:
        """Return True if VPIN does not veto the fallback entry."""
        vpin_veto = 2.0
        return not (
            (direction == 1 and vpin_z < -vpin_veto)
            or (direction == -1 and vpin_z > vpin_veto)
        )

    def _fallback_confidence(self, direction: int, ret1_ok: bool, ret5_ok: bool, vpin_z: float) -> float:
        """Compute fallback confidence based on confirmation factors."""
        vpin_agrees = (direction == 1 and vpin_z > 0.0) or (direction == -1 and vpin_z < 0.0)
        confidence = 0.55 + 0.10 * int(ret1_ok) + 0.10 * int(ret5_ok) + 0.05 * int(vpin_agrees)
        return min(confidence, 0.85)

    def _fallback_runway(self, vol_z: float) -> float:
        """Compute volatility-scaled fallback runway."""
        if vol_z < -1.0:
            vol_scale = FALLBACK_VOL_SCALE_QUIET
        elif vol_z > 1.0:
            vol_scale = FALLBACK_VOL_SCALE_HOT
        else:
            vol_scale = FALLBACK_VOL_SCALE_QUIET + (
                (FALLBACK_VOL_SCALE_HOT - FALLBACK_VOL_SCALE_QUIET) * (vol_z + 1.0) / 2.0
            )
        return float(np.clip(PREDICTED_RUNWAY_FALLBACK * vol_scale, Q_RUNWAY_MIN, Q_RUNWAY_MAX))

    def _fallback_strategy(self, state: np.ndarray, regime_threshold_adj: float = 0.0) -> int:
        """Action-only shim for backward compatibility. Delegates to _fallback_decide()."""
        action, _, _ = self._fallback_decide(state, regime_threshold_adj)
        return action

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax with temperature. Delegates to shared utility."""
        return softmax(x, temperature)

    def _platt_calibrate(self, raw_prob: float) -> float:
        """
        Apply Platt scaling to calibrate probability estimates.

        Phase 2: Converts raw model output to calibrated probability.
        Formula: p = 1 / (1 + exp(A*score + B))

        Args:
            raw_prob: Raw probability from model softmax

        Returns:
            Calibrated probability [0, 1]
        """
        # Convert probability to logit score
        if raw_prob <= 0:
            raw_prob = 1e-9
        if raw_prob >= 1:
            raw_prob = 1 - 1e-9

        logit = np.log(raw_prob / (1 - raw_prob))

        # Apply Platt transformation
        calibrated_logit = self.platt_a * logit + self.platt_b
        calibrated_prob = 1.0 / (1.0 + np.exp(-calibrated_logit))

        return float(calibrated_prob)

    def update_platt_params(self, predicted_prob: float, actual_outcome: float):
        """
        Online update of Platt calibration parameters.

        Phase 2: Gradient descent on log-loss to improve calibration.

        Args:
            predicted_prob: Predicted probability (calibrated)
            actual_outcome: Actual outcome (1.0 for success, 0.0 for failure)
        """
        if not self.enable_training:
            return

        # Clip probabilities for numerical stability
        p = np.clip(predicted_prob, 1e-9, 1 - 1e-9)

        # Gradient of log-loss with respect to Platt parameters
        # d(loss)/dA and d(loss)/dB
        error = p - actual_outcome

        # Update Platt parameters (simple gradient descent)
        self.platt_a -= self.platt_lr * error
        self.platt_b -= self.platt_lr * error * 0.1  # Smaller update for intercept

        LOG.debug(
            "[TRIGGER|PLATT] Updated: A=%.4f, B=%.4f (p_pred=%.3f, outcome=%.1f)",
            self.platt_a,
            self.platt_b,
            predicted_prob,
            actual_outcome,
        )

    def _q_to_runway(self, q_value: float) -> float:
        """
        Convert Q-value to predicted GROSS runway (expected MFE before friction).

        NOTE: This returns GROSS runway. Caller must subtract friction_cost to get NET runway.

        Heuristic mapping (as percentage of entry price):
        - Q ≤ 0: 0.0010 (0.10%, minimal runway)
        - Q = 1: 0.0020 (0.20%, moderate runway)
        - Q = 2: 0.0030 (0.30%, good runway)
        - Q ≥ 3: 0.0050 (0.50%, excellent runway)

        Example: For entry at $4600, 0.20% runway = $9.20 expected MFE
        """
        if q_value <= 0:
            return Q_RUNWAY_MIN  # 0.10% minimum
        if q_value >= Q_RUNWAY_MAX_Q:
            return Q_RUNWAY_MAX  # 0.50% maximum
        # Linear interpolation: Q in [0, 3] → Runway in [0.001, 0.005]
        return Q_RUNWAY_MIN + (q_value / Q_RUNWAY_MAX_Q) * (Q_RUNWAY_MAX - Q_RUNWAY_MIN)

    def update_from_trade(self, actual_mfe: float, predicted_runway: float, entry_confidence: float = 0.5):
        """
        Update trigger agent based on trade outcome.

        Phase 3.5: Online learning updates:
        1. Log prediction error
        2. Update Platt calibration with actual trade outcome
        3. Update entry confidence threshold based on prediction accuracy
        4. Update confidence_floor via LearnedParameters

        Args:
            actual_mfe: Actual MFE achieved during trade
            predicted_runway: What trigger predicted
        """
        if predicted_runway <= 0:
            return
        utilization = self._log_runway_error(actual_mfe, predicted_runway)
        outcome = self._trade_outcome(actual_mfe, predicted_runway)
        self._update_platt_from_trade(entry_confidence, outcome)
        self._update_confidence_from_trade(utilization)

    def _log_runway_error(self, actual_mfe: float, predicted_runway: float) -> float:
        """Log runway prediction error and return utilization."""
        error = actual_mfe - predicted_runway
        error_pct = (error / predicted_runway) * 100
        LOG.debug(
            "[TRIGGER] Runway prediction: %.4f vs actual: %.4f (error: %.1f%%)",
            predicted_runway,
            actual_mfe,
            error_pct,
        )
        return actual_mfe / predicted_runway if predicted_runway > 0 else 0.0

    def _trade_outcome(self, actual_mfe: float, predicted_runway: float) -> float:
        """Return 1.0 for success, 0.0 for failure based on runway utilization."""
        trade_success = actual_mfe >= (predicted_runway * 0.5)
        return 1.0 if trade_success else 0.0

    def _update_platt_from_trade(self, entry_confidence: float, outcome: float) -> None:
        """Update Platt calibration parameters from trade outcome."""
        if not (self.enable_training and hasattr(self, "platt_a")):
            return
        predicted_prob = float(entry_confidence)
        self.update_platt_params(predicted_prob, outcome)

    def _update_confidence_from_trade(self, utilization: float) -> None:
        """Update confidence_floor and related parameters using utilization."""
        if self.param_manager is None:
            return
        try:
            gradient = self._confidence_gradient_from_utilization(utilization)
            new_floor = self.param_manager.update(
                self.symbol,
                "confidence_floor",
                gradient,
                timeframe=self.timeframe,
                broker=self.broker,
            )
            self.confidence_floor = float(new_floor)

            self.param_manager.update(
                self.symbol,
                "fallback_base_threshold",
                gradient * 0.5,
                timeframe=self.timeframe,
                broker=self.broker,
            )

            if utilization < _UTILIZATION_OUTLIER_LOW or utilization > _UTILIZATION_OUTLIER_HIGH:
                self.param_manager.update(
                    self.symbol,
                    "regime_adj_scale",
                    -0.001,
                    timeframe=self.timeframe,
                    broker=self.broker,
                )

            self.param_manager.save()
            LOG.debug(
                "[TRIGGER] Updated confidence_floor: %.3f (gradient=%.3f, utilization=%.2f)",
                self.confidence_floor,
                gradient,
                utilization,
            )
        except Exception as exc:
            LOG.warning("[TRIGGER] Failed to update confidence_floor: %s", exc)

    def _confidence_gradient_from_utilization(self, utilization: float) -> float:
        """Return gradient adjustment from utilization metrics."""
        if utilization < _UTILIZATION_BAD_THRESHOLD:
            return 0.05
        if utilization > 1.0:
            return -0.02
        return (0.6 - utilization) * 0.03

    # add_experience, train_step, _train_step_torch, get_training_stats
    # are inherited from AgentTrainingMixin.

    def _extra_training_stats(self) -> dict:
        """Trigger-specific stats appended by the mixin."""
        return {"epsilon": self.epsilon}


# ============================================================================
# Self-Test
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("TriggerAgent Self-Test")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Test 1: Initialize without model (fallback)
    print("\n[TEST 1] Initialize without model")
    trigger = TriggerAgent(window=STATE_WINDOW_SIZE, n_features=7)
    assert not trigger.use_torch
    print("✓ Fallback mode initialized")

    # Test 2: Decide with synthetic state (flat position)
    print("\n[TEST 2] Entry decision (flat position)")
    state = rng.standard_normal((STATE_WINDOW_SIZE, 7)).astype(np.float32)
    state[:, 2] = 0.35  # Strong positive MA diff → should signal LONG
    action, conf, runway = trigger.decide(state, current_position=0)
    assert action in [0, 1, 2]
    assert 0 <= conf <= 1
    assert runway >= 0
    print(f"✓ Action: {action}, Confidence: {conf:.3f}, Runway: {runway:.4f}")

    # Test 3: Decide with existing position (should return NO_ENTRY)
    print("\n[TEST 3] Entry decision (already in position)")
    action, conf, runway = trigger.decide(state, current_position=1)
    assert action == 0  # NO_ENTRY
    from src.utils.safe_math import SafeMath

    assert SafeMath.is_zero(conf)
    assert SafeMath.is_zero(runway)
    print("✓ Correctly blocks entry when position exists")

    # Test 4: Update from trade (logging only)
    print("\n[TEST 4] Update from trade outcome")
    trigger.update_from_trade(actual_mfe=0.0025, predicted_runway=0.0020)
    print("✓ Trade outcome logged")

    # Test 5: Q-to-runway mapping
    print("\n[TEST 5] Q-value to runway conversion")
    assert trigger._q_to_runway(0.0) == Q_RUNWAY_MIN
    assert trigger._q_to_runway(1.5) == Q_RUNWAY_MIN + (1.5 / Q_RUNWAY_MAX_Q) * (Q_RUNWAY_MAX - Q_RUNWAY_MIN)
    assert trigger._q_to_runway(3.0) == Q_RUNWAY_MAX
    print("✓ Q-to-runway mapping correct")

    print("\n" + "=" * 70)
    print("✓ All TriggerAgent tests passed!")
    print("=" * 70)
