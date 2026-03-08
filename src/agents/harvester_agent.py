#!/usr/bin/env python3
"""
Harvester Agent - Exit Specialist (Phase 3)
==========================================
Dual-agent architecture component for trade exit decisions.

Responsibilities:
- Maximize capture ratio (exit_pnl / MFE)
- Avoid winner-to-loser trades
- Output: exit signal (HOLD/CLOSE) + confidence

Reward Function:
- Capture efficiency: How much of MFE was captured?
- WTL penalty: Did a winner become a loser?
- Timing bonus: Early exit with high capture = good

From MASTER_HANDBOOK.md Section 2.2: Dual-Agent Architecture

Phase 3.5: Online Learning
- ExperienceBuffer integration for continuous improvement
- train_step() for DDQN updates
"""

import logging
import os

import numpy as np

from src.agents.agent_training_mixin import AgentTrainingMixin, softmax
from src.constants import (
    BREAKEVEN_TRIGGER_PCT,
    CAPTURE_DECAY_MIN_MFE_PCT,
    CAPTURE_DECAY_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FRICTION_PCT,
    GAMMA,
    GRAD_CLIP_NORM,
    HARD_TIME_STOP_BARS,
    HARVESTER_BUFFER_CAPACITY,
    L2_WEIGHT,
    LEARNING_RATE,
    MICRO_WINNER_GIVEBACK_PCT,
    MICRO_WINNER_MFE_THRESHOLD_PCT,
    MIN_EXPERIENCES,
    MIN_SOFT_PROFIT_PCT,
    PROFIT_TARGET_PCT_DEFAULT,
    SOFT_TIME_STOP_BARS,
    STATE_WINDOW_SIZE,
    STOP_LOSS_PCT_DEFAULT,
    TAU,
    TRAILING_STOP_ACTIVATION_PCT,
    TRAILING_STOP_DISTANCE_PCT,
)
from src.core.ddqn_network import DDQNNetwork
from src.persistence.learned_parameters import LearnedParametersManager
from src.utils.experience_buffer import ExperienceBuffer

LOG = logging.getLogger(__name__)

# Internal implementation constants (harvester-only, not shared across modules)
CONFIDENCE_FALLBACK: float = 0.7
PCT_SCALE: float = 100.0
TICKS_HELD_NORM_DENOM: float = 100.0  # Normalize tick count to [0,1] range

# Magic number constants for code quality
MAX_MAE_PCT: float = 100.0  # Maximum MAE percentage (clip suspicious values)
SOFT_TIME_CAPTURE_THRESHOLD: float = 0.50  # Soft time stop capture threshold
HIGH_CAPTURE_RATIO: float = 0.7  # High capture ratio for profit target adjustment
LOW_CAPTURE_RATIO: float = 0.3  # Low capture ratio for profit target adjustment
EXCELLENT_CAPTURE_RATIO: float = 0.8  # Excellent capture ratio for trailing stop
FLOAT_EPSILON: float = 1e-9  # Floating point comparison tolerance
MFE_TO_SL_RATIO_HIGH: float = 2.0  # High MFE to SL ratio threshold
MFE_TO_SL_RATIO_LOW: float = 0.5  # Low MFE to SL ratio threshold


class HarvesterAgent(AgentTrainingMixin):
    """
    Exit specialist agent - decides WHEN to close position.

    Philosophy: "Capture profits before they evaporate"

    Action Space:
        0 = HOLD (keep position open)
        1 = CLOSE (exit position)

    State: Market features + Position state (10-dim)
        Market (7):
        - ret1: 1-bar return
        - ret5: 5-bar return
        - ma_diff: MA fast/slow difference
        - vol: 20-bar volatility
        - imbalance: Order book imbalance [-1, 1]
        - vpin_z: VPIN z-score
        - depth_ratio: Bid+ask depth relative to median

        Position (3):
        - mfe_norm: Current MFE / entry price (normalized)
        - mae_norm: Current MAE / entry price (normalized)
        - bars_held_norm: Bars held / 100 (normalized)

    Output:
        - action: 0/1 (HOLD/CLOSE)
        - confidence: [0, 1] from softmax probabilities
    """

    _AGENT_TAG = "HARVESTER"

    def __init__(  # noqa: PLR0913
        self,
        window: int = STATE_WINDOW_SIZE,
        n_features: int = 10,
        enable_training: bool = False,
        symbol: str = "XAUUSD",  # Instrument-agnostic: required param, default only for tests
        timeframe: str = "M15",
        broker: str = "default",
        param_manager: LearnedParametersManager | None = None,
        friction_calculator=None,
        timeframe_minutes: int = 5,
        buffer_capacity: int = HARVESTER_BUFFER_CAPACITY,
    ):
        """
        Initialize Harvester Agent.

        Args:
            window: Lookback window for state
            n_features: Number of input features (7 market + 3 position)
            enable_training: Enable online learning (Phase 3.5)
            friction_calculator: FrictionCalculator for cost-aware exit decisions
        """
        self.window = window
        self.n_features = n_features
        self.use_torch = False
        self.model = None
        self.torch = None
        self.symbol = symbol
        self.friction_calculator = friction_calculator
        self.timeframe = timeframe
        self.timeframe_minutes = timeframe_minutes
        self.broker = broker
        self.param_manager = param_manager

        # Phase 3.5: Experience replay buffer
        self.enable_training = enable_training
        self.buffer = ExperienceBuffer(capacity=buffer_capacity, timeframe_minutes=self.timeframe_minutes) if enable_training else None
        self.min_experiences = MIN_EXPERIENCES
        self.batch_size = DEFAULT_BATCH_SIZE
        self.training_steps = 0
        self.last_state = None  # Track state for experience creation

        # Phase 3.5: DDQN network for online learning (numpy-based, no PyTorch required)
        self.ddqn = (
            DDQNNetwork(
                state_dim=window * n_features,
                n_actions=2,
                learning_rate=LEARNING_RATE,
                gamma=GAMMA,
                tau=TAU,
                l2_weight=L2_WEIGHT,
                grad_clip_norm=GRAD_CLIP_NORM,
            )
            if enable_training
            else None
        )

        # Try to load model if path specified
        model_path = os.environ.get("DDQN_HARVESTER_MODEL", "").strip()
        if model_path:
            self._load_model(model_path)

        LOG.info("[HARVESTER] Init: training=%s | buffer=%dk min=%d",
            enable_training, buffer_capacity // 1000, self.min_experiences)

        self._init_exit_thresholds()

        # Minimum hold period: prevents DDQN (which may have stale learned Q-values)
        # from issuing a CLOSE on the very first tick after entry before any MFE develops.
        # Only the emergency stop loss is exempt from this guard.
        # Scale default to ~1 bar (timeframe ≤ 60 min) or 1 bar minimum for anything
        # longer, so at H4 we don't enforce a 40-hour minimum hold.
        # Default: ~1 hour expressed in bars (min 1).
        _default_hold = max(1, int(60 / max(1, timeframe_minutes)))
        self.min_hold_ticks = int(os.environ.get("MIN_HOLD_TICKS", str(_default_hold)))

    def _load_model(self, model_path: str):
        """Load PyTorch DDQN model for harvester agent."""
        try:
            import torch  # noqa: PLC0415

            from src.core.ddqn_network import (
                Conv1dQNet,  # noqa: PLC0415 - conditional import for optional torch dependency
            )

            self.torch = torch
            self.model = Conv1dQNet(n_features=self.n_features, n_actions=2, temporal_pool_size=1)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            self.model.eval()
            self.use_torch = True
            LOG.info("[HARVESTER] Loaded DDQN model: %s", model_path)
        except (OSError, ImportError, RuntimeError) as e:
            LOG.warning("[HARVESTER] Failed to load model: %s. Using fallback.", e)
            self.use_torch = False

    def _check_emergency_stop_loss(self, mae: float, entry_price: float) -> tuple[bool, tuple[int, float] | None]:
        """Check if emergency stop loss is triggered.

        Returns:
            (should_exit, exit_decision) where exit_decision is (action, confidence) if should exit
        """
        if entry_price is None or entry_price <= 0:
            return False, None

        try:
            mae_pct = (mae / entry_price) * PCT_SCALE

            # Defensive: Validate calculated percentage is reasonable
            if mae_pct < 0 or mae_pct > MAX_MAE_PCT:
                LOG.warning("[HARVESTER_EMERGENCY_SL] Suspicious MAE percentage %.2f%% - recalculating", mae_pct)
                mae_pct = min(max(mae_pct, 0), MAX_MAE_PCT)  # Clamp to [0, 100]

            if mae_pct >= self.stop_loss_pct:
                LOG.warning(
                    "[HARVESTER_EMERGENCY_SL] Stop loss TRIGGERED: MAE=%.2f%% >= SL=%.2f%% - CLOSING!",
                    mae_pct,
                    self.stop_loss_pct,
                )
                return True, (1, 1.0)  # CLOSE with full confidence
        except (ValueError, TypeError, ZeroDivisionError) as e:
            LOG.error("[HARVESTER_EMERGENCY_SL] Error calculating MAE percentage: %s", e)

        return False, None

    def _decide_with_ddqn(self, full_state: np.ndarray) -> tuple[int, float]:
        """Make decision using DDQN network.

        Args:
            full_state: Full state array (window, n_features)

        Returns:
            (action, confidence)
        """
        flat_state = full_state.reshape(1, -1).astype(np.float64)
        q_values = self.ddqn.predict(flat_state).flatten()  # flatten (1,N) → (N,)
        action = int(np.argmax(q_values))

        probs = self._softmax(q_values)
        confidence = float(probs[action])

        LOG.debug(
            "[HARVESTER] DDQN decision: Q=%s, action=%d (%s), conf=%.3f",
            q_values,
            action,
            "CLOSE" if action == 1 else "HOLD",
            confidence,
        )
        return action, confidence

    def _decide_with_torch(
        self, market_state: np.ndarray, mfe: float, mae: float, ticks_held: int, entry_price: float
    ) -> tuple[int, float]:
        """Make decision using PyTorch model.

        Args:
            market_state: Normalized market features (window, 7)
            mfe: Maximum favorable excursion
            mae: Maximum adverse excursion
            ticks_held: Ticks held in position
            entry_price: Position entry price

        Returns:
            (action, confidence)
        """
        # Delegate to the single source of truth for state construction
        full_state = self._build_full_state(market_state, mfe, mae, ticks_held, entry_price)

        # Model-based decision
        with self.torch.no_grad():
            t = self.torch.from_numpy(full_state).unsqueeze(0).float()
            q_values = self.model(t).squeeze(0).numpy()

            # Action selection (greedy)
            action = int(q_values.argmax())

            # Confidence from softmax probabilities
            probs = self._softmax(q_values)
            confidence = float(probs[action])

            LOG.debug(
                "[HARVESTER] Q-values: %s, Action: %d (%s), Conf: %.3f, MFE: %.4f, MAE: %.4f, Bars: %d",
                q_values,
                action,
                "CLOSE" if action == 1 else "HOLD",
                confidence,
                mfe,
                mae,
                ticks_held,
            )

            return action, confidence

    def decide(  # noqa: PLR0913
        self,
        market_state: np.ndarray,
        mfe: float,
        mae: float,
        ticks_held: int,
        entry_price: float,
        current_price: float = 0.0,
        direction: int = 1,
    ) -> tuple[int, float]:
        """Decide exit action based on market + position state.

        Args:
            market_state: Normalized market features (window, 7)
            mfe: Maximum favorable excursion (absolute price)
            mae: Maximum adverse excursion (absolute price)
            ticks_held: Number of market data ticks position has been open (~2-3/sec)
            entry_price: Entry price for normalization
            current_price: Current market price (for unrealized P&L calculation)
            direction: +1=LONG, -1=SHORT (used for correct sign of unrealized P&L)

        Returns:
            (action, confidence)
            - action: 0=HOLD, 1=CLOSE
            - confidence: [0, 1] probability from model
        """
        LOG.info(
            "[HARVESTER_DECIDE] use_torch=%s ddqn=%s enable_training=%s training_steps=%d",
            self.use_torch,
            self.ddqn is not None,
            self.enable_training,
            self.training_steps,
        )

        # Always build & record the state we see — needed for experience replay
        # regardless of which gate fires (emergency stop, min-hold, etc.).
        full_state = self._build_full_state(market_state, mfe, mae, ticks_held, entry_price)
        self.last_state = full_state.copy()

        # Emergency stop loss check (always executed, not subject to min-hold)
        should_exit, exit_decision = self._check_emergency_stop_loss(mae, entry_price)
        if should_exit:
            return exit_decision

        # Minimum hold period: only emergency stop may close before this threshold.
        # This prevents DDQN Q-values (potentially stale from prior training) from
        # issuing an immediate CLOSE before any price movement can develop.
        if ticks_held < self.min_hold_ticks:
            LOG.debug(
                "[HARVESTER] Min-hold: ticks=%d < min=%d → HOLD",
                ticks_held,
                self.min_hold_ticks,
            )
            return 0, 0.0  # HOLD

        # Hard time stop: safety valve that overrides DDQN/model decisions.
        # Prevents degenerate Q-functions from holding positions indefinitely.
        if ticks_held > self.hard_time_stop_bars:
            LOG.warning(
                "[HARVESTER] Hard time stop override: ticks=%d > hard_limit=%d → CLOSE",
                ticks_held, self.hard_time_stop_bars,
            )
            return 1, 1.0  # CLOSE with full confidence

        # Soft time stop: exit when holding too long with diminished or positive profits.
        if ticks_held > self.soft_time_stop_bars and entry_price > 0:
            mfe_pct = (mfe / entry_price) * PCT_SCALE
            mae_pct = (mae / entry_price) * PCT_SCALE
            # Use actual unrealized P&L (not MFE−MAE range) so the stop fires
            # against real current profit, not the total price range since entry.
            if current_price > 0 and direction != 0:
                current_profit_pct = max(0.0, direction * (current_price - entry_price) / entry_price * PCT_SCALE)
            else:
                current_profit_pct = max(0.0, mfe_pct - mae_pct)  # fallback if no price
            friction_pct = self.get_friction_cost_pct(entry_price) * PCT_SCALE
            net_profit_pct = mfe_pct - friction_pct
            if self._check_soft_time_stop(ticks_held, mfe_pct, current_profit_pct, net_profit_pct):
                LOG.info(
                    "[HARVESTER] Soft time stop override: ticks=%d > soft_limit=%d → CLOSE",
                    ticks_held, self.soft_time_stop_bars,
                )
                return 1, 0.9  # CLOSE with high confidence

        if not self.use_torch:
            # Use DDQN network if available and trained, else fallback
            if self.ddqn is not None and self.enable_training and self.training_steps > 0:
                return self._decide_with_ddqn(full_state)

            # Fallback: Simple profit target + stop loss
            action = self._fallback_strategy(mfe, mae, ticks_held, entry_price, current_price, direction)
            return action, CONFIDENCE_FALLBACK

        # Use PyTorch model
        return self._decide_with_torch(market_state, mfe, mae, ticks_held, entry_price)

    def _build_full_state(
        self, market_state: np.ndarray, mfe: float, mae: float, ticks_held: int, entry_price: float
    ) -> np.ndarray:
        """Build full state vector with market features + position stats.

        Args:
            market_state: Market features (window, n_market_features)
            mfe: Maximum favorable excursion (absolute price)
            mae: Maximum adverse excursion (absolute price)
            ticks_held: Number of ticks in position
            entry_price: Position entry price

        Returns:
            Full state array (window, n_features) including position stats
        """
        if entry_price <= 0:
            entry_price = 1.0  # Prevent division by zero
        mfe_norm = (mfe / entry_price) * PCT_SCALE
        mae_norm = (mae / entry_price) * PCT_SCALE
        ticks_held_norm = min(ticks_held / TICKS_HELD_NORM_DENOM, 1.0)
        position_features = np.full((market_state.shape[0], 3), [mfe_norm, mae_norm, ticks_held_norm], dtype=np.float32)
        return np.hstack([market_state, position_features])

    def _check_stop_loss(self, mae_pct: float) -> bool:
        """Check if stop loss is triggered."""
        if mae_pct >= self.stop_loss_pct:
            LOG.info("[HARVESTER] Stop loss TRIGGERED: MAE=%.2f%% >= SL=%.2f%%", mae_pct, self.stop_loss_pct)
            return True
        return False

    def _check_profit_target(self, net_profit_pct: float, mfe_pct: float, friction_pct: float) -> bool:
        """Check if profit target is hit."""
        if net_profit_pct >= self.profit_target_pct:
            LOG.debug(
                "[HARVESTER] Profit target hit: MFE=%.2f%%, Friction=%.2f%%, Net=%.2f%% (target=%.2f%%)",
                mfe_pct,
                friction_pct,
                net_profit_pct,
                self.profit_target_pct,
            )
            return True
        return False

    def _check_micro_winner_exit(self, mfe_pct: float, current_profit_pct: float) -> bool:
        """Check if micro-winner protection should trigger.

        Exits immediately when:
        - Trade has shown ANY profit (MFE > threshold)
        - But is NOW reversing back (giving back > 30% of MFE)
        - Prevents small winners from becoming massive losses (winner-to-loser trades)
        """
        mfe_threshold = getattr(self, "micro_winner_mfe_threshold_pct", MICRO_WINNER_MFE_THRESHOLD_PCT)
        giveback_threshold = getattr(self, "micro_winner_giveback_pct", MICRO_WINNER_GIVEBACK_PCT)

        # Only apply if MFE is very small (below normal trailing stop activation).
        # Use the instance attribute so this scales correctly with the timeframe.
        if mfe_pct >= getattr(self, "trailing_stop_activation_pct", TRAILING_STOP_ACTIVATION_PCT):
            return False  # Normal exits handle larger winners

        if mfe_pct > mfe_threshold:
            giveback_pct = mfe_pct - current_profit_pct
            if giveback_pct >= (mfe_pct * giveback_threshold):
                LOG.info(
                    "[HARVESTER] Micro-winner quick exit: MFE=%.4f%%, current=%.4f%%, giveback=%.1f%% of MFE",
                    mfe_pct,
                    current_profit_pct,
                    (giveback_pct / mfe_pct * 100),
                )
                return True
        return False

    def _check_trailing_stop(self, mfe_pct: float, current_profit_pct: float) -> bool:
        """Check if trailing stop is triggered."""
        trailing_activation = getattr(self, "trailing_stop_activation_pct", TRAILING_STOP_ACTIVATION_PCT)
        trailing_distance = getattr(self, "trailing_stop_distance_pct", TRAILING_STOP_DISTANCE_PCT)
        if mfe_pct >= trailing_activation:
            giveback_pct = mfe_pct - current_profit_pct
            if giveback_pct >= trailing_distance:
                LOG.info(
                    "[HARVESTER] Trailing stop hit: MFE=%.2f%%, giveback=%.2f%% >= trail=%.2f%%",
                    mfe_pct,
                    giveback_pct,
                    trailing_distance,
                )
                return True
        return False

    def _check_breakeven_stop(self, mfe_pct: float, current_profit_pct: float, friction_pct: float) -> bool:
        """Check if breakeven stop is triggered."""
        breakeven_trigger = getattr(self, "breakeven_trigger_pct", BREAKEVEN_TRIGGER_PCT)
        if mfe_pct >= breakeven_trigger and current_profit_pct <= friction_pct:
            LOG.info(
                "[HARVESTER] Breakeven stop: MFE=%.2f%% but current_profit=%.2f%% <= friction=%.2f%%",
                mfe_pct,
                current_profit_pct,
                friction_pct,
            )
            return True
        return False

    def _check_capture_decay(self, mfe_pct: float, current_profit_pct: float) -> bool:
        """Check if capture decay exit is triggered."""
        capture_decay_min = getattr(self, "capture_decay_min_mfe_pct", CAPTURE_DECAY_MIN_MFE_PCT)
        capture_decay_thresh = getattr(self, "capture_decay_threshold", CAPTURE_DECAY_THRESHOLD)
        if mfe_pct >= capture_decay_min:
            capture_ratio = current_profit_pct / mfe_pct if mfe_pct > 0 else 0.0
            if capture_ratio < capture_decay_thresh:
                LOG.info(
                    "[HARVESTER] Capture decay exit: capture=%.1f%% < threshold=%.1f%%, MFE=%.2f%%",
                    capture_ratio * 100,
                    capture_decay_thresh * 100,
                    mfe_pct,
                )
                return True
        return False

    def _check_soft_time_stop(
        self, ticks_held: int, mfe_pct: float, current_profit_pct: float, net_profit_pct: float
    ) -> bool:
        """Check if soft time stop is triggered."""
        if ticks_held > self.soft_time_stop_bars:
            if mfe_pct > 0:
                soft_capture = current_profit_pct / mfe_pct
                if soft_capture < SOFT_TIME_CAPTURE_THRESHOLD or net_profit_pct > 0:
                    LOG.debug(
                        "[HARVESTER] Soft time stop: %d ticks, MFE=%.2f%%, capture=%.1f%%, Net=%.2f%%",
                        ticks_held,
                        mfe_pct,
                        soft_capture * 100,
                        net_profit_pct,
                    )
                    return True
            elif net_profit_pct > 0:
                return True
        return False

    def _check_hard_time_stop(self, ticks_held: int) -> bool:
        """Check if hard time stop is triggered."""
        if ticks_held > self.hard_time_stop_bars:
            LOG.debug("[HARVESTER] Hard time stop: %d ticks", ticks_held)
            return True
        return False

    def _fallback_strategy(  # noqa: PLR0911
        self, mfe: float, mae: float, ticks_held: int, entry_price: float,
        current_price: float = 0.0, direction: int = 1,
    ) -> int:
        """Fallback exit strategy when no model loaded.

        Rules (designed for profit protection on M5):
        1. Stop loss at MAE threshold
        2. Profit target hit → CLOSE
        3. Trailing stop: once MFE exceeds activation, trail behind peak
        4. Breakeven stop: once MFE exceeds breakeven trigger, don't let winner become loser
        5. Capture decay: if current P&L drops below 40% of MFE, exit to protect profits
        6. Soft time stop: exit if holding too long with diminished profits
        7. Hard time stop: exit regardless
        """
        # Guard against division by zero
        if entry_price <= 0:
            LOG.warning("[HARVESTER] entry_price=0, cannot evaluate exit - holding")
            return 0  # HOLD

        # Calculate metrics
        friction_pct = self.get_friction_cost_pct(entry_price) * PCT_SCALE
        mfe_pct = (mfe / entry_price) * PCT_SCALE
        mae_pct = (mae / entry_price) * PCT_SCALE

        LOG.info(
            "[HARVESTER_FALLBACK] SL Check: mae=%.4f entry=%.2f mae_pct=%.4f stop_loss_pct=%.4f check=%s",
            mae,
            entry_price,
            mae_pct,
            self.stop_loss_pct,
            mae_pct >= self.stop_loss_pct,
        )

        # Compute actual unrealized P&L from current price.
        # direction * (current_price - entry_price) gives signed P&L regardless of side.
        # Fall back to MFE−MAE range only when current_price is unavailable.
        if current_price > 0 and direction != 0:
            current_profit_pct = max(0.0, direction * (current_price - entry_price) / entry_price * PCT_SCALE)
        else:
            current_profit_pct = max(0.0, mfe_pct - mae_pct)  # fallback: use range
        net_profit_pct = mfe_pct - friction_pct

        # Check exit conditions in priority order
        if self._check_stop_loss(mae_pct):
            return 1
        if self._check_profit_target(net_profit_pct, mfe_pct, friction_pct):
            return 1
        if self._check_micro_winner_exit(mfe_pct, current_profit_pct):
            return 1
        if self._check_trailing_stop(mfe_pct, current_profit_pct):
            return 1
        if self._check_breakeven_stop(mfe_pct, current_profit_pct, friction_pct):
            return 1
        if self._check_capture_decay(mfe_pct, current_profit_pct):
            return 1
        if self._check_soft_time_stop(ticks_held, mfe_pct, current_profit_pct, net_profit_pct):
            return 1
        if self._check_hard_time_stop(ticks_held):
            return 1

        return 0  # HOLD

    def _init_exit_thresholds(self):
        """Load exit thresholds from LearnedParametersManager (or defaults).

        Thresholds adapt to timeframe:
        - M1: Tighter stops, faster exits (scalping)
        - M15: Balanced (default)
        - H1+: Wider stops, patience (swing trading)
        """
        # Infer timeframe multiplier from symbol context (if available)
        timeframe_scale = self._get_timeframe_scale()

        # Base defaults scaled by timeframe
        self.profit_target_pct = self._get_param(
            "harvester_profit_target_pct", PROFIT_TARGET_PCT_DEFAULT * timeframe_scale
        )
        self.stop_loss_pct = self._get_param("harvester_stop_loss_pct", STOP_LOSS_PCT_DEFAULT * timeframe_scale)
        self.soft_time_stop_bars = int(
            round(self._get_param("harvester_soft_time_bars", SOFT_TIME_STOP_BARS / timeframe_scale))
        )
        self.hard_time_stop_bars = int(
            round(self._get_param("harvester_hard_time_bars", HARD_TIME_STOP_BARS / timeframe_scale))
        )
        self.min_soft_profit_pct = self._get_param(
            "harvester_min_soft_profit_pct", MIN_SOFT_PROFIT_PCT * timeframe_scale
        )
        # Trailing / breakeven thresholds — all scaled by timeframe so the same
        # constants work correctly across M5 (scale=0.6) through H4 (scale=3.5).
        self.trailing_stop_activation_pct = self._get_param(
            "harvester_trailing_stop_activation_pct", TRAILING_STOP_ACTIVATION_PCT * timeframe_scale
        )
        self.trailing_stop_distance_pct = self._get_param(
            "harvester_trailing_stop_distance_pct", TRAILING_STOP_DISTANCE_PCT * timeframe_scale
        )
        self.breakeven_trigger_pct = self._get_param(
            "harvester_breakeven_trigger_pct", BREAKEVEN_TRIGGER_PCT * timeframe_scale
        )
        self.micro_winner_mfe_threshold_pct = self._get_param(
            "harvester_micro_winner_mfe_threshold_pct", MICRO_WINNER_MFE_THRESHOLD_PCT * timeframe_scale
        )
        self.capture_decay_min_mfe_pct = self._get_param(
            "harvester_capture_decay_min_mfe_pct", CAPTURE_DECAY_MIN_MFE_PCT * timeframe_scale
        )
        LOG.info(
            "[HARVESTER] Exit plan: TP=%.2f%% SL=%.2f%% trail_act=%.2f%% trail_dist=%.2f%% "
            "be=%.2f%% micro=%.3f%% soft=%d bars hard=%d bars (timeframe=%s scale=%.2f)",
            self.profit_target_pct,
            self.stop_loss_pct,
            self.trailing_stop_activation_pct,
            self.trailing_stop_distance_pct,
            self.breakeven_trigger_pct,
            self.micro_winner_mfe_threshold_pct,
            self.soft_time_stop_bars,
            self.hard_time_stop_bars,
            self.timeframe,
            self._get_timeframe_scale(),
        )

    def _get_timeframe_scale(self) -> float:
        """Infer timeframe multiplier for threshold scaling.

        All percentage thresholds (TP, SL, trailing activation/distance, breakeven,
        micro-winner) are multiplied by this factor, making exit logic timeframe-agnostic.
        M15 (scale=1.0) is the calibration baseline; other timeframes are derived ratios.

        M1: 0.3x  M5: 0.6x  M15: 1.0x (base)  M30: 1.5x  H1: 2.0x  H4: 3.5x  D1: 5.0x
        """
        tf_map = {"M1": 0.3, "M5": 0.6, "M15": 1.0, "M30": 1.5, "H1": 2.0, "H4": 3.5, "D1": 5.0}
        return tf_map.get(self.timeframe, 1.0)

    def get_friction_cost_pct(self, entry_price: float, quantity: float = 0.10, side: str = "BUY") -> float:
        """Calculate estimated friction costs as percentage of entry price.

        Args:
            entry_price: Position entry price
            quantity: Position size in lots
            side: "BUY" or "SELL"

        Returns:
            Friction cost as percentage (e.g., 0.0005 = 0.05% of entry price)
        """
        if not self.friction_calculator or entry_price <= 0:
            # Conservative estimate: spread ~0.10% + commission ~0.02% + slippage ~0.03% = 0.15%
            # Example: 0.15% friction on $4600 entry = $6.90 cost
            return DEFAULT_FRICTION_PCT  # Default 0.15% friction estimate (instrument-agnostic)

        try:
            # Calculate total friction in USD
            # SAME CODE PATH for both paper trading and live trading
            # CRITICAL: M5 intraday trades don't cross rollover → swap = 0
            friction = self.friction_calculator.calculate_total_friction(
                quantity=quantity,
                side=side,
                price=entry_price,
                holding_days=0.1,  # Assume ~2-3 hours avg hold (M5 trades)
                volatility_factor=1.0,
                crosses_rollover=False,  # Intraday M5 trades don't cross rollover
            )

            # Convert USD friction to percentage of entry price
            # Example: $3.4 friction on $4600 entry (spread+comm+slip, NO swap) = 0.074%
            contract_size = getattr(self.friction_calculator.costs, "contract_size", 100.0)
            position_value = quantity * entry_price * contract_size
            if position_value > 0:
                friction_pct = friction["total"] / position_value
                LOG.debug(
                    "[HARVESTER-FRICTION] Entry=%.2f, Quantity=%.2f, Friction=$%.2f (%.3f%%) [swap=$%.2f intraday=0]",
                    entry_price,
                    quantity,
                    friction["total"],
                    friction_pct * 100,
                    friction.get("swap", 0),
                )
                return friction_pct
        except (AttributeError, TypeError, ZeroDivisionError) as exc:
            LOG.warning("[HARVESTER] Friction calculation failed: %s", exc)

        return DEFAULT_FRICTION_PCT  # Fallback to conservative 0.15%

    def _ensure_param_manager(self) -> LearnedParametersManager:
        if self.param_manager is None:
            self.param_manager = LearnedParametersManager()
            self.param_manager.load()
        return self.param_manager

    def _get_param(self, name: str, default: float) -> float:
        try:
            manager = self._ensure_param_manager()
            value = manager.get(self.symbol, name, timeframe=self.timeframe, broker=self.broker, default=default)
            return float(value)
        except (AttributeError, ValueError, TypeError) as exc:
            LOG.debug("[HARVESTER] Falling back to default %.3f for %s (%s)", default, name, exc)
            return float(default)

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax with temperature. Delegates to shared utility."""
        return softmax(x, temperature)

    # ── Parameter learning helpers ────────────────────────────────────────────

    def _profit_target_gradient(self, capture_ratio: float, was_wtl: bool) -> float:
        """Return the gradient for the profit-target parameter based on trade outcome."""
        if was_wtl:
            LOG.info("[HARVESTER] WTL trade - reducing profit target")
            return -0.15  # Should have exited earlier → reduce by 15%
        if capture_ratio > HIGH_CAPTURE_RATIO:
            LOG.debug("[HARVESTER] High capture %.2f%% - tightening target", capture_ratio * 100)
            return -0.06
        if capture_ratio < LOW_CAPTURE_RATIO:
            LOG.info("[HARVESTER] Low capture %.2f%% - widening profit target", capture_ratio * 100)
            return 0.10
        LOG.debug("[HARVESTER] Capture %.2f%% - minor adjustment", capture_ratio * 100)
        return (0.5 - capture_ratio) * 0.15  # Moderate capture → small nudge

    def _trail_stop_gradient(self, capture_ratio: float, was_wtl: bool) -> float:
        """Return the gradient for the trailing-stop distance."""
        if was_wtl:
            return -0.10  # Trailing stop too loose → tighten
        if capture_ratio > EXCELLENT_CAPTURE_RATIO:
            return 0.03  # Possibly too tight → loosen slightly
        return 0.0

    def _sl_gradient(self, was_wtl: bool) -> float:
        """Return the gradient for the stop-loss parameter (WTL trades only)."""
        if not was_wtl:
            return 0.0
        mfe_to_sl = getattr(self, "_last_mfe_pct", 0.0) / (self.stop_loss_pct + FLOAT_EPSILON)
        if mfe_to_sl > MFE_TO_SL_RATIO_HIGH:
            LOG.info("[HARVESTER] WTL with large MFE (%.2fx SL) → tightening stop loss", mfe_to_sl)
            return -0.08  # MFE > 2x stop loss → SL too wide
        if mfe_to_sl < MFE_TO_SL_RATIO_LOW:
            LOG.debug("[HARVESTER] WTL with small MFE (%.2fx SL) → SL unchanged", mfe_to_sl)
        return 0.0

    def update_from_trade(self, capture_ratio: float, was_wtl: bool):
        """
        Update harvester exit thresholds based on trade outcome.

        Uses gradient-based learning to adapt:
        - Profit targets and stop losses
        - Trailing stop distance
        - Breakeven trigger level

        Args:
            capture_ratio: exit_pnl / MFE (how much of MFE captured)
            was_wtl: Was this a winner-to-loser trade?
        """
        if not self.param_manager:
            return  # No parameter manager, skip updates

        try:
            profit_gradient = self._profit_target_gradient(capture_ratio, was_wtl)
            new_tp = self.param_manager.update(
                self.symbol, "harvester_profit_target_pct", profit_gradient,
                timeframe=self.timeframe, broker=self.broker,
            )
            self.profit_target_pct = new_tp * self._get_timeframe_scale()

            trail_gradient = self._trail_stop_gradient(capture_ratio, was_wtl)
            if abs(trail_gradient) > FLOAT_EPSILON:
                current_trail = getattr(self, "trailing_stop_distance_pct", TRAILING_STOP_DISTANCE_PCT)
                self.trailing_stop_distance_pct = max(0.05, min(0.40, current_trail + trail_gradient * current_trail))
                LOG.debug("[HARVESTER] Updated trailing distance: %.2f%%", self.trailing_stop_distance_pct)

            sl_gradient = self._sl_gradient(was_wtl)
            if abs(sl_gradient) > FLOAT_EPSILON:
                new_sl = self.param_manager.update(
                    self.symbol, "harvester_stop_loss_pct", sl_gradient,
                    timeframe=self.timeframe, broker=self.broker,
                )
                self.stop_loss_pct = new_sl * self._get_timeframe_scale()
                LOG.info("[HARVESTER] Updated stop loss: %.4f%% (gradient=%.3f)", self.stop_loss_pct, sl_gradient)

            self.param_manager.save()
            LOG.info(
                "[HARVESTER] Updated profit target: %.2f%% (gradient=%.3f, saved to disk)",
                self.profit_target_pct, profit_gradient,
            )

        except (AttributeError, ValueError, TypeError, OSError) as exc:
            LOG.warning("[HARVESTER] Failed to update parameters: %s", exc)

    # add_experience, train_step, _train_step_torch, get_training_stats
    # are inherited from AgentTrainingMixin.

    def _extra_training_stats(self) -> dict:
        """Harvester-specific stats appended by the mixin."""
        return {"min_hold_ticks": self.min_hold_ticks}


# ============================================================================
# Self-Test
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("HarvesterAgent Self-Test")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Test 1: Initialize without model (fallback)
    print("\n[TEST 1] Initialize without model")
    harvester = HarvesterAgent(window=STATE_WINDOW_SIZE, n_features=10)
    assert not harvester.use_torch
    print("✓ Fallback mode initialized")

    # Test 2: Decide with synthetic state (profit target)
    print("\n[TEST 2] Exit decision (profit target hit)")
    market_state = rng.standard_normal((STATE_WINDOW_SIZE, 7)).astype(np.float32)
    entry_price = 100000.0
    mfe = entry_price * 0.004  # 0.4% MFE (above 0.3% target)
    mae = entry_price * 0.001  # 0.1% MAE
    bars_held = 10

    action, conf = harvester.decide(market_state, mfe, mae, bars_held, entry_price)
    assert action == 1  # Should CLOSE (profit target)
    assert 0 <= conf <= 1
    print(f"✓ Action: {action} (CLOSE), Confidence: {conf:.3f}")

    # Test 3: Decide with stop loss
    print("\n[TEST 3] Exit decision (stop loss hit)")
    mfe = entry_price * 0.001  # 0.1% MFE
    mae = entry_price * 0.003  # 0.3% MAE (above 0.2% stop)

    action, conf = harvester.decide(market_state, mfe, mae, bars_held, entry_price)
    assert action == 1  # Should CLOSE (stop loss)
    print(f"✓ Action: {action} (CLOSE), Confidence: {conf:.3f}")

    # Test 4: Decide with HOLD (no exit conditions)
    print("\n[TEST 4] Exit decision (hold position)")
    mfe = entry_price * 0.002  # 0.2% MFE (below target)
    mae = entry_price * 0.0015  # 0.15% MAE (below stop)
    bars_held = 5

    action, conf = harvester.decide(market_state, mfe, mae, bars_held, entry_price)
    assert action == 0  # Should HOLD
    print(f"✓ Action: {action} (HOLD), Confidence: {conf:.3f}")

    # Test 5: Update from trade (logging only)
    print("\n[TEST 5] Update from trade outcome")
    harvester.update_from_trade(capture_ratio=0.75, was_wtl=False)
    harvester.update_from_trade(capture_ratio=0.0, was_wtl=True)
    print("✓ Trade outcomes logged")

    print("\n" + "=" * 70)
    print("✓ All HarvesterAgent tests passed!")
    print("=" * 70)
