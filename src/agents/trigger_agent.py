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
import math
import os
import random

import numpy as np

from src.core.ddqn_network import DDQNNetwork
from src.utils.experience_buffer import ExperienceBuffer, RegimeSampling
from src.persistence.learned_parameters import LearnedParametersManager

LOG = logging.getLogger(__name__)


MIN_FEATURE_COLS = 3
IMBALANCE_INDEX = 4
TILT_SCALE = 0.1
PAPER_EPSILON = 0.15
PAPER_BASE_THRESHOLD = 0.15
LIVE_BASE_THRESHOLD = 0.3
PREDICTED_RUNWAY_FALLBACK = 0.0015
Q_RUNWAY_MIN = 0.0010
Q_RUNWAY_MAX = 0.0050
Q_RUNWAY_MAX_Q = 3.0
TD_ERROR_CAP = 10.0


class TriggerAgent:
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

    def __init__(  # noqa: PLR0913
        self,
        window: int = 64,
        n_features: int = 7,
        enable_training: bool = False,
        symbol: str = "BTCUSD",
        timeframe: str = "M15",
        broker: str = "default",
        param_manager: LearnedParametersManager | None = None,
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
        self.enable_training = enable_training
        self.buffer = ExperienceBuffer(capacity=50_000) if enable_training else None
        self.min_experiences = 100  # Minimum before training starts (lowered for faster training)
        self.batch_size = 64
        self.training_steps = 0
        self.last_state = None  # Track state for experience creation
        self.last_action = None

        # Phase 3.5: DDQN network for online learning (numpy-based, no PyTorch required)
        # This is the actual trainable network - separate from the PyTorch model loaded from disk
        self.ddqn = (
            DDQNNetwork(
                state_dim=window * n_features,  # Flattened state vector
                n_actions=3,  # NO_ENTRY=0, LONG=1, SHORT=2
                learning_rate=0.0005,
                gamma=0.99,
                tau=0.005,
                l2_weight=0.0001,
                grad_clip_norm=1.0,
            )
            if enable_training
            else None
        )
        if enable_training:
            LOG.info("[TRIGGER] DDQNNetwork initialized: state_dim=%d, actions=3", window * n_features)

        # Phase 2: Platt calibration for probability estimates (online learning)
        # Converts raw scores to calibrated probabilities: p = 1/(1 + exp(A*score + B))
        self.platt_a = 1.0  # Slope parameter (learned online)
        self.platt_b = 0.0  # Intercept parameter (learned online)
        self.platt_lr = 0.01  # Learning rate for Platt updates

        # Phase 2: Gating strategy
        # Training mode: NO GATES - pure exploration, learn through rewards
        # Live mode: LEARNED GATES - confidence floor from trained model
        if self.disable_gates or self.paper_mode:
            self.feasibility_threshold = 0.0  # No feasibility gate in training
            self.confidence_floor = 0.0  # No confidence floor in training
            LOG.info("[TRIGGER] TRAINING MODE - All gates DISABLED")
        else:
            # Live trading: use learned gating
            self.feasibility_threshold, feas_source = self._resolve_gate_value(
                env_key="FEAS_THRESHOLD", param_name="feasibility_threshold", fallback=0.5
            )
            self.confidence_floor, conf_source = self._resolve_gate_value(
                env_key="CONFIDENCE_FLOOR", param_name="confidence_floor", fallback=0.55
            )
            LOG.info(
                "[TRIGGER] LIVE MODE - Confidence floor: %.2f (%s) | Feasibility: %.2f (%s)",
                self.confidence_floor,
                conf_source,
                self.feasibility_threshold,
                feas_source,
            )

        LOG.info(
            "[TRIGGER] Epsilon: %.2f → %.2f (decay=%.4f)",
            self.epsilon,
            self.epsilon_end,
            self.epsilon_decay,
        )

        # Try to load model if path specified
        model_path = os.environ.get("DDQN_TRIGGER_MODEL", "").strip()
        if model_path:
            self._load_model(model_path)
        else:
            LOG.info("[TRIGGER] No model specified, using fallback strategy")

        if self.enable_training:
            LOG.info(
                "[TRIGGER] Online learning ENABLED (buffer capacity=50k, min=%d)",
                self.min_experiences,
            )

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
            from torch import nn  # noqa: PLC0415

            class TriggerQNet(nn.Module):
                """Q-Network for trigger agent (entry specialist)."""

                def __init__(self, window: int, n_features: int, n_actions: int = 3):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
                        nn.ReLU(),
                        nn.Conv1d(64, 64, kernel_size=5, padding=2),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool1d(1),
                        nn.Flatten(),
                        nn.Linear(64, 128),
                        nn.ReLU(),
                        nn.Linear(128, n_actions),
                    )

                def forward(self, x):
                    # x: (B,T,F) -> (B,F,T)
                    return self.net(x.transpose(1, 2))

            self.torch = torch
            self.model = TriggerQNet(window=self.window, n_features=self.n_features, n_actions=3)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            self.use_torch = True
            LOG.info("[TRIGGER] Loaded DDQN model: %s", model_path)
        except Exception as e:
            LOG.warning("[TRIGGER] Failed to load model: %s. Using fallback.", e)
            self.use_torch = False

    def decide(  # noqa: PLR0913, PLR0911
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
            - predicted_runway: Expected MFE (e.g., 0.002 = 20 pips for BTC)
        """
        # Track bars since last trade for forced exploration
        self.bars_since_trade += 1

        # Don't signal entry if already in position
        if current_position != 0:
            self.bars_since_trade = 0  # Reset when in position
            return 0, 0.0, 0.0  # NO_ENTRY

        # TRAINING MODE: Epsilon-greedy exploration
        # Philosophy: No gates in training - learn through rewards
        if self.paper_mode or self.disable_gates:
            # Random exploration with epsilon probability
            if random.random() < self.epsilon:
                # Random action: 1=LONG, 2=SHORT (exclude 0=HOLD for faster learning)
                action = random.choice([1, 2])
                LOG.info(
                    "[TRIGGER] EXPLORE: random action=%d (ε=%.3f, bars_flat=%d)",
                    action,
                    self.epsilon,
                    self.bars_since_trade,
                )
                self._decay_epsilon()
                self.bars_since_trade = 0  # Reset counter when taking action
                self.last_state = state.copy()  # FIX: Store state for experience creation
                self.last_action = action
                return action, 0.5, PREDICTED_RUNWAY_FALLBACK

            # Forced exploration after too many bars flat
            if self.force_exploration and self.bars_since_trade >= self.max_bars_inactive:
                action = random.choice([1, 2])  # Force LONG or SHORT
                LOG.info(
                    "[TRIGGER] FORCED ENTRY after %d bars flat: action=%d",
                    self.bars_since_trade,
                    action,
                )
                self.bars_since_trade = 0
                self.last_state = state.copy()  # FIX: Store state for experience creation
                self.last_action = action
                return action, 0.5, PREDICTED_RUNWAY_FALLBACK

        # LIVE MODE: Feasibility gate (only if not disabled)
        if not self.disable_gates and feasibility < self.feasibility_threshold:
            LOG.debug(
                "[TRIGGER] BLOCKED by feasibility gate: %.3f < %.3f",
                feasibility,
                self.feasibility_threshold,
            )
            return 0, 0.0, 0.0  # NO_ENTRY

        if not self.use_torch:
            # Use DDQN network if available (online learning), else fallback to MA crossover
            if self.ddqn is not None and self.enable_training and self.training_steps > 0:
                # DDQN-based decision (numpy network, trained online)
                flat_state = state.reshape(1, -1).astype(np.float64)
                q_values = self.ddqn.predict(flat_state).flatten()  # flatten (1,N) → (N,)
                action = int(np.argmax(q_values))

                # Confidence from softmax
                probs = self._softmax(q_values)
                confidence = float(probs[action])

                # Predicted runway from Q-value
                gross_runway = self._q_to_runway(float(q_values[action]))
                predicted_runway = max(0.0, gross_runway - friction_cost)

                # Store last state for experience creation
                self.last_state = state.copy()
                self.last_action = action

                LOG.debug(
                    "[TRIGGER] DDQN decision: Q=%s, action=%d, conf=%.3f, runway=%.4f",
                    q_values,
                    action,
                    confidence,
                    predicted_runway,
                )
            else:
                # Fallback: Simple MA crossover with microstructure tilt + regime adjustment
                action = self._fallback_strategy(state, regime_threshold_adj)
                confidence = 0.6  # Medium confidence for rule-based
                predicted_runway = PREDICTED_RUNWAY_FALLBACK  # Conservative estimate: 15 pips

            # Store last state for experience creation (always, for both paths)
            self.last_state = state.copy()
            self.last_action = action

            # In training, accept all signals. In live, require confidence floor
            if not self.paper_mode and confidence < self.confidence_floor:
                LOG.debug(
                    "[TRIGGER] BLOCKED by confidence floor: %.3f < %.3f",
                    confidence,
                    self.confidence_floor,
                )
                return 0, confidence, 0.0

            self._decay_epsilon()
            return action, confidence, predicted_runway

        # Model-based decision
        with self.torch.no_grad():
            t = self.torch.from_numpy(state).unsqueeze(0).float()
            q_values = self.model(t).squeeze(0).numpy()

            # Action selection (greedy)
            action = int(q_values.argmax())

            # Raw probability from softmax
            probs = self._softmax(q_values)
            raw_prob = float(probs[action])

            # Phase 2: Platt calibration for better probability estimates
            # Calibrated p = 1 / (1 + exp(A*raw + B))
            calibrated_prob = self._platt_calibrate(raw_prob)

            # LIVE MODE: Confidence floor gate
            # Only take trades where model is confident (e.g., >55%)
            if not self.paper_mode and calibrated_prob < self.confidence_floor:
                LOG.debug(
                    "[TRIGGER] BLOCKED by confidence floor: %.3f < %.3f",
                    calibrated_prob,
                    self.confidence_floor,
                )
                return 0, calibrated_prob, 0.0  # NO_ENTRY

            # Phase 2: Economics-derived probability threshold (secondary gate)
            # From handbook: p > (L + K) / (G + L)
            # Ensures expected value is positive after friction costs
            breakeven_prob = (expected_loss + friction_cost) / (expected_gain + expected_loss + 1e-9)

            # Gate on economics threshold (only in live mode)
            if not self.paper_mode and action != 0 and calibrated_prob < breakeven_prob:
                LOG.debug(
                    "[TRIGGER] BLOCKED by economics: p=%.3f < breakeven=%.3f (G=%.4f, L=%.4f, K=%.4f)",
                    calibrated_prob,
                    breakeven_prob,
                    expected_gain,
                    expected_loss,
                    friction_cost,
                )
                return 0, 0.0, 0.0  # NO_ENTRY

            # Predicted runway from Q-value magnitude (NET after friction)
            q_max = q_values[action]
            gross_runway = self._q_to_runway(q_max)
            predicted_runway = max(0.0, gross_runway - friction_cost)  # Net runway after friction

            LOG.debug(
                "[TRIGGER] Q-values: %s, Action: %d, Raw_p: %.3f, Calib_p: %.3f, Breakeven: %.3f, Gross_runway: %.4f, Friction: %.4f, Net_runway: %.4f",
                q_values,
                action,
                raw_prob,
                calibrated_prob,
                breakeven_prob,
                gross_runway,
                friction_cost,
                predicted_runway,
            )

            self._decay_epsilon()
            return action, calibrated_prob, predicted_runway

    def _decay_epsilon(self):
        """Decay epsilon for exploration schedule."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _fallback_strategy(self, state: np.ndarray, regime_threshold_adj: float = 0.0) -> int:
        """
        Fallback entry strategy when no model loaded.

        Uses MA crossover + microstructure tilt + Phase 3.4 regime adjustment.
        NOTE: Epsilon exploration is handled in decide() method, NOT here.

        Args:
            state: Normalized state features (can be deque or ndarray)
            regime_threshold_adj: Adjustment from regime detector
                - Negative (trending): easier to trigger
                - Positive (mean-reverting): harder to trigger
        """
        # Convert deque to ndarray if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        if state.shape[0] == 0 or state.shape[1] < MIN_FEATURE_COLS:
            return 0  # NO_ENTRY

        # Get latest features (already normalized)
        ma_diff = float(state[-1, 2])  # MA fast/slow difference
        imbalance = float(state[-1, IMBALANCE_INDEX]) if state.shape[1] > IMBALANCE_INDEX else 0.0

        # Tilt thresholds by order book imbalance
        tilt = imbalance * TILT_SCALE

        # Phase 3.4: Apply regime-based threshold adjustment
        # Paper mode: Lower base threshold for more entries
        paper_mode = os.environ.get("PAPER_MODE") == "1"
        base_threshold = PAPER_BASE_THRESHOLD if paper_mode else LIVE_BASE_THRESHOLD
        adjusted_threshold = base_threshold + regime_threshold_adj

        if ma_diff > (adjusted_threshold - tilt):
            return 1  # LONG
        if ma_diff < (-adjusted_threshold - tilt):
            return 2  # SHORT
        return 0  # NO_ENTRY

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax with temperature for confidence calculation."""
        exp_x = np.exp((x - np.max(x)) / temperature)
        return exp_x / exp_x.sum()

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

        Heuristic mapping:
        - Q ≤ 0: 0.0010 (10 pips, minimal runway)
        - Q = 1: 0.0020 (20 pips, moderate runway)
        - Q = 2: 0.0030 (30 pips, good runway)
        - Q ≥ 3: 0.0050 (50 pips, excellent runway)

        For BTC/USD at ~$100k, 1 pip = $10, so:
        - 10 pips = 0.0001 = $10
        - 20 pips = 0.0002 = $20
        - 50 pips = 0.0005 = $50
        """
        if q_value <= 0:
            return Q_RUNWAY_MIN  # 10 pips minimum
        if q_value >= Q_RUNWAY_MAX_Q:
            return Q_RUNWAY_MAX  # 50 pips maximum
        # Linear interpolation: Q in [0, 3] → Runway in [0.001, 0.005]
        return Q_RUNWAY_MIN + (q_value / Q_RUNWAY_MAX_Q) * (Q_RUNWAY_MAX - Q_RUNWAY_MIN)

    def update_from_trade(self, actual_mfe: float, predicted_runway: float):
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
        if predicted_runway > 0:
            error = actual_mfe - predicted_runway
            error_pct = (error / predicted_runway) * 100
            LOG.debug(
                "[TRIGGER] Runway prediction: %.4f vs actual: %.4f (error: %.1f%%)",
                predicted_runway,
                actual_mfe,
                error_pct,
            )

            # Determine if trade was a success (MFE exceeded prediction)
            # Use 50% of predicted runway as success threshold
            trade_success = actual_mfe >= (predicted_runway * 0.5)
            outcome = 1.0 if trade_success else 0.0

            # Update Platt calibration with actual outcome
            if self.enable_training and hasattr(self, "platt_a"):
                # Use the last known confidence as predicted probability
                # (Platt calibration improves over time)
                predicted_prob = 0.5  # Default if we don't track per-trade confidence
                self.update_platt_params(predicted_prob, outcome)

            # Update confidence_floor via LearnedParameters
            if self.param_manager is not None:
                try:
                    utilization = actual_mfe / predicted_runway if predicted_runway > 0 else 0.0
                    if utilization < 0.3:
                        # Bad entry - raise confidence floor to be more selective
                        gradient = 0.05
                    elif utilization > 1.0:
                        # Great entry - can slightly lower floor to allow more trades
                        gradient = -0.02
                    else:
                        # Moderate - small adjustment toward center
                        gradient = (0.6 - utilization) * 0.03

                    new_floor = self.param_manager.update(
                        self.symbol,
                        "confidence_floor",
                        gradient,
                        timeframe=self.timeframe,
                        broker=self.broker,
                    )
                    self.confidence_floor = float(new_floor)
                    self.param_manager.save()
                    LOG.debug(
                        "[TRIGGER] Updated confidence_floor: %.3f (gradient=%.3f, utilization=%.2f)",
                        self.confidence_floor,
                        gradient,
                        utilization,
                    )
                except Exception as exc:
                    LOG.warning("[TRIGGER] Failed to update confidence_floor: %s", exc)

    def add_experience(  # noqa: PLR0913
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: int = RegimeSampling.UNKNOWN,
    ):
        """Add experience to replay buffer.

        Args:
            state: Entry state vector
            action: Action taken (0=NO_ENTRY, 1=LONG, 2=SHORT)
            reward: Runway utilization reward
            next_state: Exit state vector
            done: Always True for trigger (trade completed)
            regime: Current market regime
        """
        if not self.enable_training or self.buffer is None:
            return

        self.buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            regime=regime,
        )

        LOG.debug(
            "[TRIGGER] Experience added: action=%d, reward=%.4f, buffer_size=%d",
            action,
            reward,
            self.buffer.tree.n_entries,
        )

    def train_step(self) -> dict | None:
        """Perform one training step using prioritized experience replay.

        Uses DDQNNetwork.train_batch() for actual backpropagation and weight updates.

        Returns:
            Dictionary with training metrics, or None if insufficient data
        """
        if not self.enable_training or self.buffer is None:
            return None

        # Check if we have enough experiences
        if self.buffer.tree.n_entries < self.min_experiences:
            return None

        # Sample batch
        batch = self.buffer.sample(batch_size=self.batch_size)
        if batch is None:
            return None

        rewards = batch["rewards"]  # (batch_size,)
        indices = batch["indices"]  # (batch_size,)

        # Defensive: Validate batch
        if not all(math.isfinite(r) for r in rewards):
            LOG.warning("[TRIGGER] Non-finite rewards in batch, skipping training")
            return None

        if self.ddqn is not None:
            # Real DDQN training with backpropagation
            try:
                # Flatten states: (batch, window, features) -> (batch, window*features)
                states = batch["states"].reshape(batch["states"].shape[0], -1).astype(np.float64)
                next_states = batch["next_states"].reshape(batch["next_states"].shape[0], -1).astype(np.float64)
                actions = batch["actions"].astype(np.intp)
                dones = batch["dones"].astype(np.float64)
                weights = batch["weights"].astype(np.float64)
                rewards_f = rewards.astype(np.float64)

                # Train batch: forward pass, TD targets, backward pass, weight update
                train_result = self.ddqn.train_batch(
                    states=states,
                    actions=actions,
                    rewards=rewards_f,
                    next_states=next_states,
                    dones=dones,
                    weights=weights,
                )

                # Update buffer priorities with actual TD errors
                td_errors = np.abs(train_result["td_errors"])
                td_errors = np.clip(td_errors, 0, TD_ERROR_CAP)
                self.buffer.update_priorities(indices, td_errors)

                metrics = {
                    "loss": train_result["loss"],
                    "mean_q": train_result["mean_q"],
                    "mean_td_error": train_result["mean_td_error"],
                    "max_td_error": train_result["max_td_error"],
                    "grad_norm": train_result["grad_norm"],
                    "mean_reward": float(np.mean(rewards)),
                }
            except Exception as e:
                LOG.error("[TRIGGER] DDQN train_batch failed: %s", e, exc_info=True)
                # Fallback to priority-only update
                td_errors = np.clip(np.abs(rewards), 0, TD_ERROR_CAP)
                self.buffer.update_priorities(indices, td_errors)
                metrics = {
                    "loss": 0.0,
                    "mean_q": 0.0,
                    "mean_td_error": float(np.mean(td_errors)),
                    "max_td_error": float(np.max(td_errors)),
                    "mean_reward": float(np.mean(rewards)),
                }
        elif self.use_torch:
            # PyTorch training path
            metrics = self._train_step_torch(batch)
        else:
            # No network available - priority updates only (degraded mode)
            td_errors = np.abs(rewards)
            td_errors = np.clip(td_errors, -TD_ERROR_CAP, TD_ERROR_CAP)
            self.buffer.update_priorities(indices, td_errors)
            metrics = {
                "loss": 0.0,
                "mean_q": 0.0,
                "mean_td_error": float(np.mean(td_errors)),
                "max_td_error": float(np.max(td_errors)),
                "mean_reward": float(np.mean(rewards)),
            }
            LOG.warning("[TRIGGER] No DDQN network - only updating priorities (no weight updates)")

        self.training_steps += 1

        # Log every 10 steps (more frequent during early training)
        log_interval = 10 if self.training_steps < 100 else 100
        if self.training_steps % log_interval == 0:
            LOG.info(
                "[TRIGGER] Training step %d: loss=%.4f, mean_q=%.3f, mean_reward=%.4f, mean_td=%.4f, buffer=%d",
                self.training_steps,
                metrics.get("loss", 0.0),
                metrics.get("mean_q", 0.0),
                metrics.get("mean_reward", 0.0),
                metrics.get("mean_td_error", 0.0),
                self.buffer.tree.n_entries,
            )

        return metrics

    def _train_step_torch(self, batch: dict) -> dict:
        """Training step when use_torch=True (PyTorch model loaded from disk).

        Delegates to the numpy DDQN network for actual weight updates.
        The PyTorch model is only used for inference (pre-trained).

        Args:
            batch: Sampled batch from buffer

        Returns:
            Training metrics
        """
        # Delegate to DDQN numpy network for online learning
        if self.ddqn is not None:
            states = batch["states"].reshape(batch["states"].shape[0], -1).astype(np.float64)
            next_states = batch["next_states"].reshape(batch["next_states"].shape[0], -1).astype(np.float64)
            train_result = self.ddqn.train_batch(
                states=states,
                actions=batch["actions"].astype(np.intp),
                rewards=batch["rewards"].astype(np.float64),
                next_states=next_states,
                dones=batch["dones"].astype(np.float64),
                weights=batch["weights"].astype(np.float64),
            )
            self.buffer.update_priorities(batch["indices"], np.abs(train_result["td_errors"]))
            return {
                "loss": train_result["loss"],
                "mean_q": train_result["mean_q"],
                "mean_td_error": train_result["mean_td_error"],
                "mean_reward": float(np.mean(batch["rewards"])),
            }

        # No DDQN network — priority-only update (degraded mode)
        LOG.warning("[TRIGGER] No DDQN network in torch path — priority-only update")
        td_errors = np.abs(batch["rewards"])
        td_errors = np.clip(td_errors, 0, 10.0)
        self.buffer.update_priorities(batch["indices"], td_errors)
        return {
            "loss": 0.0,
            "mean_q": 0.0,
            "mean_td_error": float(np.mean(td_errors)),
            "mean_reward": float(np.mean(batch["rewards"])),
        }

    def get_training_stats(self) -> dict:
        """Get training statistics for monitoring.

        Returns:
            Dictionary with training stats
        """
        if not self.enable_training or self.buffer is None:
            return {"enabled": False}

        buffer_stats = self.buffer.get_stats()

        return {
            "enabled": True,
            "training_steps": self.training_steps,
            "buffer_size": buffer_stats["size"],
            "buffer_utilization": buffer_stats["utilization"],
            "total_added": buffer_stats["total_added"],
            "total_sampled": buffer_stats["total_sampled"],
            "beta": buffer_stats["beta"],
            "ready_to_train": buffer_stats["size"] >= self.min_experiences,
        }


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
    trigger = TriggerAgent(window=64, n_features=7)
    assert not trigger.use_torch
    print("✓ Fallback mode initialized")

    # Test 2: Decide with synthetic state (flat position)
    print("\n[TEST 2] Entry decision (flat position)")
    state = rng.standard_normal((64, 7)).astype(np.float32)
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
