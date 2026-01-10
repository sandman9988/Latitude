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

from experience_buffer import ExperienceBuffer, RegimeSampling
from learned_parameters import LearnedParametersManager

LOG = logging.getLogger(__name__)

BUFFER_CAPACITY: int = 50_000
MIN_EXPERIENCES_DEFAULT: int = 1000
BATCH_SIZE_DEFAULT: int = 64
CONFIDENCE_FALLBACK: float = 0.7
PCT_SCALE: float = 100.0
BARS_HELD_NORM_DENOM: float = 100.0
SOFT_TIME_STOP_BARS: int = 50
HARD_TIME_STOP_BARS: int = 80
MIN_SOFT_PROFIT_PCT: float = 0.05
PROFIT_TARGET_PCT_DEFAULT: float = 0.30
STOP_LOSS_PCT_DEFAULT: float = 0.20


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
    ):
        """
        Initialize Harvester Agent.

        Args:
            window: Lookback window for state
            n_features: Number of input features (7 market + 3 position)
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

        # Phase 3.5: Experience replay buffer
        self.enable_training = enable_training
        self.buffer = ExperienceBuffer(capacity=BUFFER_CAPACITY) if enable_training else None
        self.min_experiences = MIN_EXPERIENCES_DEFAULT  # Minimum before training starts
        self.batch_size = BATCH_SIZE_DEFAULT
        self.training_steps = 0

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
        bars_held: int,
        entry_price: float,
        direction: int,
    ) -> tuple[int, float]:
        """
        Decide exit action based on market + position state.

        Args:
            market_state: Normalized market features (window, 7)
            mfe: Maximum favorable excursion (absolute price)
            mae: Maximum adverse excursion (absolute price)
            bars_held: Number of bars position has been open
            entry_price: Entry price for normalization
            direction: +1 for LONG, -1 for SHORT

        Returns:
            (action, confidence)
            - action: 0=HOLD, 1=CLOSE
            - confidence: [0, 1] probability from model
        """
        if not self.use_torch:
            # Fallback: Simple profit target + stop loss
            action = self._fallback_strategy(mfe, mae, bars_held, entry_price)
            confidence = CONFIDENCE_FALLBACK  # Medium-high confidence for rule-based
            return action, confidence

        # Augment state with position information
        # Normalize MFE/MAE/bars_held to [0, 1] range
        mfe_norm = (mfe / entry_price) * PCT_SCALE  # Convert to percentage
        mae_norm = (mae / entry_price) * PCT_SCALE
        bars_held_norm = min(bars_held / BARS_HELD_NORM_DENOM, 1.0)  # Cap at normalization window

        # Broadcast position features across window
        position_features = np.full((market_state.shape[0], 3), [mfe_norm, mae_norm, bars_held_norm], dtype=np.float32)

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
                bars_held,
            )

            return action, confidence

    def _fallback_strategy(self, mfe: float, mae: float, bars_held: int, entry_price: float) -> int:
        """
        Fallback exit strategy when no model loaded.

        Rules:
        1. Take profit at 0.3% MFE (30 pips) - only if capture ratio decent
        2. Stop loss at -0.2% MAE (-20 pips)
        3. Soft time stop: 50 bars if MFE > 0.05% (5 pips minimum profit)
        4. Hard time stop: 80 bars (prevent stagnation)

        Prevents closing on negligible MFE (0.00% case).
        """
        mfe_pct = (mfe / entry_price) * PCT_SCALE
        mae_pct = (mae / entry_price) * PCT_SCALE

        # Stop loss (priority)
        if mae_pct >= self.stop_loss_pct:
            LOG.debug("[HARVESTER] Stop loss hit: %.2f%%", mae_pct)
            return 1  # CLOSE

        # Profit target (only if meaningful MFE)
        if mfe_pct >= self.profit_target_pct:
            LOG.debug("[HARVESTER] Profit target hit: %.2f%%", mfe_pct)
            return 1  # CLOSE

        # Soft time stop (only exit if we have SOME profit - min 5 pips)
        if bars_held > self.soft_time_stop_bars:
            if mfe_pct > self.min_soft_profit_pct:  # At least threshold profit
                LOG.debug("[HARVESTER] Soft time stop: %d bars, MFE=%.2f%%", bars_held, mfe_pct)
                return 1  # CLOSE

        # Hard time stop (exit regardless to free up capital)
        if bars_held > self.hard_time_stop_bars:
            LOG.debug("[HARVESTER] Hard time stop: %d bars", bars_held)
            return 1  # CLOSE

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
        LOG.info(
            "[HARVESTER] Exit plan: TP=%.2f%% SL=%.2f%% soft=%d bars hard=%d bars min_profit=%.2f%%",
            self.profit_target_pct,
            self.stop_loss_pct,
            self.soft_time_stop_bars,
            self.hard_time_stop_bars,
            self.min_soft_profit_pct,
        )

    def _get_timeframe_scale(self) -> float:
        """Infer timeframe multiplier for threshold scaling.

        M1: 0.3x (tight), M5: 0.6x, M15: 1.0x (base), H1: 2.0x, D1: 5.0x (wide)
        """
        tf_map = {"M1": 0.3, "M5": 0.6, "M15": 1.0, "M30": 1.5, "H1": 2.0, "H4": 3.5, "D1": 5.0}
        return tf_map.get(self.timeframe, 1.0)

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
        Update harvester agent based on trade outcome.

        This is a hook for online learning (Phase 3.5).
        For now, just log the performance.

        Args:
            capture_ratio: exit_pnl / MFE (how much of MFE captured)
            was_wtl: Was this a winner-to-loser trade?
        """
        if was_wtl:
            LOG.info("[HARVESTER] WTL trade - should have exited earlier")
        else:
            LOG.debug("[HARVESTER] Capture ratio: %.2f%%", capture_ratio * 100)

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
            self.buffer.tree.size,
        )

    def train_step(self) -> dict | None:
        """Perform one training step using prioritized experience replay.

        Returns:
            Dictionary with training metrics, or None if insufficient data
        """
        if not self.enable_training or self.buffer is None:
            return None

        # Check if we have enough experiences
        if self.buffer.tree.size < self.min_experiences:
            return None

        # Sample batch
        batch = self.buffer.sample(batch_size=self.batch_size)
        if batch is None:
            return None

        # Extract batch components
        batch["states"]  # (batch_size, window, n_features)
        batch["actions"]  # (batch_size,)
        rewards = batch["rewards"]  # (batch_size,)
        batch["next_states"]  # (batch_size, window, n_features)
        batch["dones"]  # (batch_size,)
        indices = batch["indices"]  # (batch_size,)
        batch["weights"]  # (batch_size,) for importance sampling

        # Defensive: Validate batch
        import math

        if not all(math.isfinite(r) for r in rewards):
            LOG.warning("[HARVESTER] Non-finite rewards in batch, skipping training")
            return None

        if self.use_torch:
            # PyTorch training (full implementation)
            metrics = self._train_step_torch(batch)
        else:
            # Placeholder: Calculate TD-errors for priority updates (no actual training)
            # TD-error = |reward + γ * max Q(s') - Q(s,a)|
            # Simplified: Use reward directly as TD-error proxy
            td_errors = np.abs(rewards)

            # Defensive: Cap TD-errors
            td_errors = np.clip(td_errors, -10.0, 10.0)

            # Update priorities
            self.buffer.update_priorities(indices, td_errors)

            metrics = {
                "loss": 0.0,  # Placeholder
                "mean_q": 0.0,
                "mean_td_error": float(np.mean(td_errors)),
                "max_td_error": float(np.max(td_errors)),
                "mean_reward": float(np.mean(rewards)),
            }

        self.training_steps += 1

        # Log every 100 steps
        if self.training_steps % 100 == 0:
            LOG.info(
                "[HARVESTER] Training step %d: mean_reward=%.4f, mean_td_error=%.4f, buffer_size=%d",
                self.training_steps,
                metrics["mean_reward"],
                metrics["mean_td_error"],
                self.buffer.tree.size,
            )

        return metrics

    def _train_step_torch(self, batch: dict) -> dict:
        """Full PyTorch training step (placeholder for future implementation).

        Args:
            batch: Sampled batch from buffer

        Returns:
            Training metrics
        """
        # TODO: Implement full DDQN training
        # 1. Forward pass: Q(s,a)
        # 2. Target network: Q_target(s',a')
        # 3. TD-target: r + γ * max Q_target(s') * (1 - done)
        # 4. Loss: weighted MSE with importance sampling weights
        # 5. Backward pass with gradient clipping
        # 6. Update priorities based on TD-errors
        # 7. Sync target network every 1000 steps

        LOG.warning("[HARVESTER] PyTorch training not yet implemented, using placeholder")

        # Placeholder: Return dummy metrics
        return {
            "loss": 0.0,
            "mean_q": 0.0,
            "mean_td_error": 0.0,
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
