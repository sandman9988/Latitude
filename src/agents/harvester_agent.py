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

from src.core.ddqn_network import DDQNNetwork
from src.utils.experience_buffer import ExperienceBuffer, RegimeSampling
from src.persistence.learned_parameters import LearnedParametersManager

LOG = logging.getLogger(__name__)

BUFFER_CAPACITY: int = 50_000
MIN_EXPERIENCES_DEFAULT: int = 100  # Lowered for faster training
BATCH_SIZE_DEFAULT: int = 64
CONFIDENCE_FALLBACK: float = 0.7
PCT_SCALE: float = 100.0
TICKS_HELD_NORM_DENOM: float = 100.0  # Normalize tick count to [0,1] range
SOFT_TIME_STOP_BARS: int = 200  # Let trades develop (200 ticks ≈ 10-15 min on M5)
HARD_TIME_STOP_BARS: int = 400  # Hard cap at 400 ticks
MIN_SOFT_PROFIT_PCT: float = 0.20  # Need at least 20 pips ($2) to soft-exit
PROFIT_TARGET_PCT_DEFAULT: float = 2.50  # Target 250 pips (~$25 on XAUUSD) - UPDATED_JAN13_2230
STOP_LOSS_PCT_DEFAULT: float = 0.80  # Allow 80 pips ($8) drawdown


class HarvesterAgent:
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

    def __init__(
        self,
        window: int = 64,
        n_features: int = 10,
        enable_training: bool = False,
        symbol: str = "BTCUSD",
        timeframe: str = "M15",
        broker: str = "default",
        param_manager: LearnedParametersManager | None = None,
        friction_calculator=None,
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
        self.broker = broker
        self.param_manager = param_manager

        # Phase 3.5: Experience replay buffer
        self.enable_training = enable_training
        self.buffer = ExperienceBuffer(capacity=BUFFER_CAPACITY) if enable_training else None
        self.min_experiences = MIN_EXPERIENCES_DEFAULT  # Minimum before training starts
        self.batch_size = BATCH_SIZE_DEFAULT
        self.training_steps = 0
        self.last_state = None  # Track state for experience creation

        # Phase 3.5: DDQN network for online learning (numpy-based, no PyTorch required)
        self.ddqn = (
            DDQNNetwork(
                state_dim=window * n_features,  # Flattened state vector
                n_actions=2,  # HOLD=0, CLOSE=1
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
            LOG.info("[HARVESTER] DDQNNetwork initialized: state_dim=%d, actions=2", window * n_features)

        # Try to load model if path specified
        model_path = os.environ.get("DDQN_HARVESTER_MODEL", "").strip()
        if model_path:
            self._load_model(model_path)
        else:
            LOG.info("[HARVESTER] No model specified, using fallback strategy")

        if self.enable_training:
            LOG.info(
                "[HARVESTER] Online learning ENABLED (buffer capacity=50k, min=%d)",
                self.min_experiences,
            )

        self._init_exit_thresholds()

    def _load_model(self, model_path: str):
        """Load PyTorch DDQN model for harvester agent."""
        try:
            import torch
            from torch import nn

            class HarvesterQNet(nn.Module):
                """Q-Network for harvester agent (exit specialist)."""

                def __init__(self, window: int, n_features: int, n_actions: int = 2):
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
            self.model = HarvesterQNet(window=self.window, n_features=self.n_features, n_actions=2)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
            self.use_torch = True
            LOG.info("[HARVESTER] Loaded DDQN model: %s", model_path)
        except Exception as e:
            LOG.warning("[HARVESTER] Failed to load model: %s. Using fallback.", e)
            self.use_torch = False

    def decide(
        self,
        market_state: np.ndarray,
        mfe: float,
        mae: float,
        ticks_held: int,
        entry_price: float,
        direction: int,
    ) -> tuple[int, float]:
        """
        Decide exit action based on market + position state.

        Args:
            market_state: Normalized market features (window, 7)
            mfe: Maximum favorable excursion (absolute price)
            mae: Maximum adverse excursion (absolute price)
            ticks_held: Number of market data ticks position has been open (~2-3/sec)
            entry_price: Entry price for normalization
            direction: +1 for LONG, -1 for SHORT

        Returns:
            (action, confidence)
            - action: 0=HOLD, 1=CLOSE
            - confidence: [0, 1] probability from model
        """
        if not self.use_torch:
            # Build full state for DDQN (market + position features)
            full_state = self._build_full_state(market_state, mfe, mae, ticks_held, entry_price)
            self.last_state = full_state.copy()  # Track for experience creation

            # Use DDQN network if available and trained, else fallback
            if self.ddqn is not None and self.enable_training and self.training_steps > 0:
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
            else:
                # Fallback: Simple profit target + stop loss
                action = self._fallback_strategy(mfe, mae, ticks_held, entry_price)
                confidence = CONFIDENCE_FALLBACK
                return action, confidence

        # Augment state with position information
        # Normalize MFE/MAE/ticks_held to [0, 1] range
        mfe_norm = (mfe / entry_price) * PCT_SCALE  # Convert to percentage
        mae_norm = (mae / entry_price) * PCT_SCALE
        ticks_held_norm = min(ticks_held / TICKS_HELD_NORM_DENOM, 1.0)  # Cap at normalization window

        # Broadcast position features across window
        position_features = np.full((market_state.shape[0], 3), [mfe_norm, mae_norm, ticks_held_norm], dtype=np.float32)

        # Combine: (window, 7 + 3) = (window, 10)
        full_state = np.hstack([market_state, position_features])

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
                "[HARVESTER] Q-values: %s, Action: %d (%s), Conf: %.3f, " "MFE: %.4f, MAE: %.4f, Bars: %d",
                q_values,
                action,
                "CLOSE" if action == 1 else "HOLD",
                confidence,
                mfe,
                mae,
                ticks_held,
            )

            return action, confidence

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

    def _fallback_strategy(self, mfe: float, mae: float, ticks_held: int, entry_price: float) -> int:
        """
        Fallback exit strategy when no model loaded.

        Rules (designed for serious MFE development):
        1. Take profit at target AFTER subtracting friction costs
        2. Stop loss at -0.8% MAE (-80 pips) - wider cushion for volatility
        3. Soft time stop: 200 ticks if NET profit > 0 (after friction)
        4. Hard time stop: 400 ticks (allow full runway for trend capture)

        Critical: Profit targets are friction-adjusted to ensure NET profitability.
        Raw target of 2.5% - 0.15% friction = 2.35% net target.
        """
        # Guard against division by zero (entry_price not set yet)
        if entry_price <= 0:
            LOG.warning("[HARVESTER] entry_price=0, cannot evaluate exit - holding")
            return 0  # HOLD

        # Calculate friction cost percentage
        friction_pct = self.get_friction_cost_pct(entry_price) * PCT_SCALE  # Convert to same scale as mfe_pct

        mfe_pct = (mfe / entry_price) * PCT_SCALE
        mae_pct = (mae / entry_price) * PCT_SCALE

        # Net profit after friction costs
        net_profit_pct = mfe_pct - friction_pct

        # Stop loss (priority)
        if mae_pct >= self.stop_loss_pct:
            LOG.debug("[HARVESTER] Stop loss hit: %.2f%%", mae_pct)
            return 1  # CLOSE

        # Profit target (NET profit must exceed target after friction)
        if net_profit_pct >= self.profit_target_pct:
            LOG.debug(
                "[HARVESTER] Profit target hit: MFE=%.2f%%, Friction=%.2f%%, Net=%.2f%% (target=%.2f%%)",
                mfe_pct,
                friction_pct,
                net_profit_pct,
                self.profit_target_pct,
            )
            return 1  # CLOSE

        # Soft time stop (only exit if NET profit is positive after friction)
        # TODO: Rename self.soft_time_stop_bars -> self.soft_time_stop_ticks in __init__
        if ticks_held > self.soft_time_stop_bars:
            if net_profit_pct > 0:  # Any net profit after friction
                LOG.debug(
                    "[HARVESTER] Soft time stop: %d ticks, MFE=%.2f%%, Net=%.2f%% (after %.2f%% friction)",
                    ticks_held,
                    mfe_pct,
                    net_profit_pct,
                    friction_pct,
                )
                return 1  # CLOSE

        # Hard time stop (exit regardless to free up capital)
        # TODO: Rename self.hard_time_stop_bars -> self.hard_time_stop_ticks in __init__
        if ticks_held > self.hard_time_stop_bars:
            LOG.debug("[HARVESTER] Hard time stop: %d ticks", ticks_held)
            return 1  # CLOSE

        return 0  # HOLD

    def quick_exit_check(
        self, mfe: float, mae: float, entry_price: float, current_price: float, direction: int
    ) -> bool:
        """
        Fast tick-based exit check for stop loss and profit targets only.
        Called on every price tick for responsive exit execution.

        Args:
            mfe: Maximum Favorable Excursion (absolute)
            mae: Maximum Adverse Excursion (absolute)
            entry_price: Position entry price
            current_price: Current market price
            direction: 1 for LONG, -1 for SHORT

        Returns:
            True if should exit immediately, False otherwise
        """
        if entry_price <= 0:
            return False

        friction_pct = self.get_friction_cost_pct(entry_price) * PCT_SCALE
        mfe_pct = (mfe / entry_price) * PCT_SCALE
        mae_pct = (mae / entry_price) * PCT_SCALE
        net_profit_pct = mfe_pct - friction_pct

        # Stop loss check (immediate exit)
        if mae_pct >= self.stop_loss_pct:
            LOG.info("[TICK_HARVESTER] Stop loss triggered: %.2f%% >= %.2f%%", mae_pct, self.stop_loss_pct)
            return True

        # Profit target check (NET profit after friction)
        if net_profit_pct >= self.profit_target_pct:
            LOG.info(
                "[TICK_HARVESTER] Profit target triggered: MFE=%.2f%%, Net=%.2f%% (friction=%.2f%%) >= %.2f%%",
                mfe_pct,
                net_profit_pct,
                friction_pct,
                self.profit_target_pct,
            )
            return True

        return False

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
        LOG.info(
            "[HARVESTER] Exit plan: TP=%.2f%% SL=%.2f%% soft=%d bars hard=%d bars min_profit=%.2f%% (timeframe=%s scale=%.2f)",
            self.profit_target_pct,
            self.stop_loss_pct,
            self.soft_time_stop_bars,
            self.hard_time_stop_bars,
            self.min_soft_profit_pct,
            self.timeframe,
            self._get_timeframe_scale(),
        )

    def _get_timeframe_scale(self) -> float:
        """Infer timeframe multiplier for threshold scaling.

        M1: 0.3x (tight), M5: 0.6x, M15: 1.0x (base), H1: 2.0x, D1: 5.0x (wide)
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
            Friction cost as percentage (e.g., 0.05 = 5 pips for 0.05% of price)
        """
        if not self.friction_calculator or entry_price <= 0:
            # Conservative estimate: spread ~0.10% + commission ~0.02% + slippage ~0.03% = 0.15%
            # For XAUUSD @ $4600: 0.15% = 6.9 pips ($6.90)
            return 0.0015  # Default 0.15% = 15 pips per 1% move

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
            # For XAUUSD @ $4600: friction ~$3.4 (spread+comm+slip, NO swap) → 0.074% = 7.4 pips
            contract_size = getattr(self.friction_calculator.costs, "contract_size", 100000)
            position_value = quantity * entry_price * contract_size
            if position_value > 0:
                friction_pct = friction["total"] / position_value
                LOG.debug(
                    "[HARVESTER-FRICTION] Entry=%.2f, Quantity=%.2f, Friction=$%.2f (%.3f%% = %.2f pips) [swap=$%.2f intraday=0]",
                    entry_price,
                    quantity,
                    friction["total"],
                    friction_pct * 100,
                    friction.get("total_pips", 0),
                    friction.get("swap", 0),
                )
                return friction_pct
        except Exception as exc:
            LOG.warning("[HARVESTER] Friction calculation failed: %s", exc)

        return 0.0015  # Fallback to conservative 0.15%

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
        except Exception as exc:
            LOG.debug("[HARVESTER] Falling back to default %.3f for %s (%s)", default, name, exc)
            return float(default)

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax with temperature for confidence calculation."""
        exp_x = np.exp((x - np.max(x)) / temperature)
        return exp_x / exp_x.sum()

    def update_from_trade(self, capture_ratio: float, was_wtl: bool):
        """
        Update harvester exit thresholds based on trade outcome.

        Uses gradient-based learning to adapt profit targets and stop losses:
        - High capture ratio (>0.7) → can afford tighter profit targets
        - Low capture ratio (<0.3) → need wider profit targets to let winners run
        - WTL trades → profit target was too loose, tighten it

        Args:
            capture_ratio: exit_pnl / MFE (how much of MFE captured)
            was_wtl: Was this a winner-to-loser trade?
        """
        if not self.param_manager:
            return  # No parameter manager, skip updates

        try:
            # Calculate gradients based on trade outcome
            if was_wtl:
                # WTL: Should have exited earlier → reduce profit target
                profit_gradient = -0.15  # Reduce by 15% (3x faster)
                LOG.info("[HARVESTER] WTL trade - reducing profit target")
            elif capture_ratio > 0.7:
                # Great capture → can tighten profit target
                profit_gradient = -0.06  # 3x faster
                LOG.debug("[HARVESTER] High capture %.2f%% - tightening target", capture_ratio * 100)
            elif capture_ratio < 0.3:
                # Poor capture → widen profit target to let winners run
                profit_gradient = 0.15  # 3x faster
                LOG.info("[HARVESTER] Low capture %.2f%% - widening profit target", capture_ratio * 100)
            else:
                # Moderate capture → small adjustment
                profit_gradient = (0.5 - capture_ratio) * 0.3  # 3x faster
                LOG.debug("[HARVESTER] Capture %.2f%% - minor adjustment", capture_ratio * 100)

            # Update profit target parameter
            new_tp = self.param_manager.update(
                self.symbol,
                "harvester_profit_target_pct",
                profit_gradient,
                timeframe=self.timeframe,
                broker=self.broker,
            )

            # Update local cached value
            self.profit_target_pct = new_tp * self._get_timeframe_scale()

            # Persist updated parameters to disk
            self.param_manager.save()

            LOG.info(
                "[HARVESTER] Updated profit target: %.2f%% (gradient=%.3f, saved to disk)",
                self.profit_target_pct,
                profit_gradient,
            )

        except Exception as exc:
            LOG.warning("[HARVESTER] Failed to update parameters: %s", exc)

    def add_experience(
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
            state: Position state vector (market + MFE/MAE/bars)
            action: Action taken (0=HOLD, 1=CLOSE)
            reward: Capture efficiency reward
            next_state: Next position state
            done: True if position closed
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
            "[HARVESTER] Experience added: action=%d, reward=%.4f, buffer_size=%d",
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

        # Extract batch components
        rewards = batch["rewards"]  # (batch_size,)
        indices = batch["indices"]  # (batch_size,)

        # Defensive: Validate batch
        import math

        if not all(math.isfinite(r) for r in rewards):
            LOG.warning("[HARVESTER] Non-finite rewards in batch, skipping training")
            return None

        TD_ERROR_CAP = 10.0

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
                LOG.error("[HARVESTER] DDQN train_batch failed: %s", e, exc_info=True)
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
            LOG.warning("[HARVESTER] No DDQN network - only updating priorities (no weight updates)")

        self.training_steps += 1

        # Log every 10 steps (more frequent during early training)
        log_interval = 10 if self.training_steps < 100 else 100
        if self.training_steps % log_interval == 0:
            LOG.info(
                "[HARVESTER] Training step %d: loss=%.4f, mean_q=%.3f, mean_reward=%.4f, mean_td=%.4f, buffer=%d",
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
        LOG.warning("[HARVESTER] No DDQN network in torch path — priority-only update")
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
    print("HarvesterAgent Self-Test")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Test 1: Initialize without model (fallback)
    print("\n[TEST 1] Initialize without model")
    harvester = HarvesterAgent(window=64, n_features=10)
    assert not harvester.use_torch
    print("✓ Fallback mode initialized")

    # Test 2: Decide with synthetic state (profit target)
    print("\n[TEST 2] Exit decision (profit target hit)")
    market_state = rng.standard_normal((64, 7)).astype(np.float32)
    entry_price = 100000.0
    mfe = entry_price * 0.004  # 0.4% MFE (above 0.3% target)
    mae = entry_price * 0.001  # 0.1% MAE
    bars_held = 10

    action, conf = harvester.decide(market_state, mfe, mae, bars_held, entry_price, direction=1)
    assert action == 1  # Should CLOSE (profit target)
    assert 0 <= conf <= 1
    print(f"✓ Action: {action} (CLOSE), Confidence: {conf:.3f}")

    # Test 3: Decide with stop loss
    print("\n[TEST 3] Exit decision (stop loss hit)")
    mfe = entry_price * 0.001  # 0.1% MFE
    mae = entry_price * 0.003  # 0.3% MAE (above 0.2% stop)

    action, conf = harvester.decide(market_state, mfe, mae, bars_held, entry_price, direction=1)
    assert action == 1  # Should CLOSE (stop loss)
    print(f"✓ Action: {action} (CLOSE), Confidence: {conf:.3f}")

    # Test 4: Decide with HOLD (no exit conditions)
    print("\n[TEST 4] Exit decision (hold position)")
    mfe = entry_price * 0.002  # 0.2% MFE (below target)
    mae = entry_price * 0.0015  # 0.15% MAE (below stop)
    bars_held = 5

    action, conf = harvester.decide(market_state, mfe, mae, bars_held, entry_price, direction=1)
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
