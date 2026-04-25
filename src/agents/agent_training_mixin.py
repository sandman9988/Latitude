"""
Agent Training Mixin — Shared DDQN online-learning logic
=========================================================
Eliminates ~250 lines of near-identical code that was duplicated
between TriggerAgent and HarvesterAgent.

Provides:
  - add_experience()
  - train_step()
  - _train_step_torch()
  - get_training_stats()
  - softmax()              (was _softmax in both agents)
"""

import logging
import math
from typing import Any

import numpy as np

from src.constants import TD_ERROR_CAP, TRAINING_LOG_INTERVAL_EARLY, TRAINING_LOG_INTERVAL_LATE, TRAINING_STEPS_EARLY

LOG = logging.getLogger(__name__)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature for confidence calculation."""
    exp_x = np.exp((x - np.max(x)) / temperature)
    return exp_x / exp_x.sum()


# ── Confidence computation constants ─────────────────────────────────────────
_TEMP_MAX: float = 1.0  # Starting softmax temperature (exploration)
_TEMP_MIN: float = 0.5  # Minimum temperature (exploitation)
_TEMP_DECAY_STEPS: int = 5000  # Steps to fully decay temperature
_ADV_SCALE: float = 3.0  # Sigmoid scaling for advantage-based confidence
_SOFTMAX_WEIGHT: float = 0.5  # Blend weight for softmax vs advantage confidence


def adaptive_temperature(training_steps: int) -> float:
    """Compute softmax temperature that decays with training progress.

    Early in training (few steps): higher temperature → uniform probabilities,
    encourages exploration and doesn't produce misleading high-confidence signals.

    Later in training (many steps): lower temperature → sharper probabilities,
    reflects genuine Q-value differentiation from learning.

    Returns temperature in [_TEMP_MIN, _TEMP_MAX].
    """
    progress = min(1.0, training_steps / _TEMP_DECAY_STEPS)
    return _TEMP_MAX - (_TEMP_MAX - _TEMP_MIN) * progress


def advantage_confidence(q_values: np.ndarray) -> float:
    """Compute confidence from Q-value advantage (best minus second-best).

    For a 3-action softmax, even modest Q-value advantages produce low softmax
    probabilities. The advantage measure directly captures how strongly the
    network prefers one action, mapped to [0, 1] via sigmoid.

    This is more discriminative than softmax for multi-action spaces:
    - Q=[0, 0.3, 0] → softmax max = 0.40, advantage_conf = 0.71
    - Q=[0, 0.5, 0] → softmax max = 0.47, advantage_conf = 0.82
    - Q=[0, 0, 0]   → softmax max = 0.33, advantage_conf = 0.50
    """
    if len(q_values) < 2:
        return 0.5
    sorted_q = np.sort(q_values)[::-1]
    advantage = sorted_q[0] - sorted_q[1]
    return float(1.0 / (1.0 + np.exp(-_ADV_SCALE * advantage)))


def compute_confidence(q_values: np.ndarray, training_steps: int) -> float:
    """Compute blended confidence from Q-values using adaptive temperature + advantage.

    Combines two complementary signals:
    1. Softmax probability (with adaptive temperature) — distribution-aware
    2. Advantage-based confidence — directly measures action preference strength

    The blend produces confidence > 0.5 when the network has a genuine preference,
    even with modest Q-value spreads typical of early-to-mid training.
    """
    temp = adaptive_temperature(training_steps)
    probs = softmax(q_values, temperature=temp)
    action = int(np.argmax(q_values))
    softmax_conf = float(probs[action])
    adv_conf = advantage_confidence(q_values)
    return _SOFTMAX_WEIGHT * softmax_conf + (1.0 - _SOFTMAX_WEIGHT) * adv_conf


class AgentTrainingMixin:
    """Mixin that supplies shared DDQN training helpers.

    The consuming class MUST provide:
      - self.enable_training  (bool)
      - self.buffer           (ExperienceBuffer | None)
      - self.min_experiences  (int)
      - self.batch_size       (int)
      - self.training_steps   (int)
      - self.ddqn             (DDQNNetwork | None)
      - self.use_torch        (bool)
      - _AGENT_TAG            (str, e.g. "TRIGGER" or "HARVESTER")
    """

    # Class constant overridden by each subclass
    _AGENT_TAG: str = "AGENT"

    # Instance attributes initialized by mixin methods or host class
    # (default values satisfy type checker while allowing mixin pattern)
    torch: Any = None  # torch module (lazy import in _load_torch_model)
    model: Any = None  # Conv1dQNet model
    use_torch: bool = False
    param_manager: Any = None  # LearnedParametersManager
    window: int = 0
    n_features: int = 0
    symbol: str = ""
    timeframe: str = ""
    timeframe_minutes: int = 0
    broker: str = ""
    enable_training: bool = False
    buffer: Any = None  # ExperienceBuffer
    min_experiences: int = 0
    batch_size: int = 0
    training_steps: int = 0
    last_state: np.ndarray | None = None
    ddqn: Any = None  # DDQNNetwork

    def _get_param(self, name: str, default: float) -> float:
        """Load a learned parameter if available; otherwise return default."""
        manager = getattr(self, "param_manager", None)
        if manager is None:
            return float(default)

        symbol = getattr(self, "symbol", "XAUUSD")
        timeframe = getattr(self, "timeframe", "M15")
        broker = getattr(self, "broker", "default")

        try:
            value = manager.get(symbol, name, timeframe=timeframe, broker=broker, default=default)
            return float(value)
        except (AttributeError, ValueError, TypeError) as exc:
            agent_tag = getattr(self, "_AGENT_TAG", "AGENT")
            LOG.debug("[%s] Falling back to default %.3f for %s (%s)", agent_tag, default, name, exc)
            return float(default)

    def _load_torch_model(self, model_path: str, n_actions: int, tag: str) -> bool:
        """Load optional PyTorch model for inference path."""
        try:
            import torch  # noqa: PLC0415

            from src.core.ddqn_network import Conv1dQNet  # noqa: PLC0415

            self.torch = torch
            self.model = Conv1dQNet(n_features=self.n_features, n_actions=n_actions, temporal_pool_size=1)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            self.model.eval()
            self.use_torch = True
            LOG.info("[%s] Loaded DDQN model: %s", tag, model_path)
            return True
        except (OSError, ImportError, RuntimeError) as exc:
            LOG.warning("[%s] Failed to load model: %s. Using fallback.", tag, exc)
            self.use_torch = False
            return False

    def _ensure_param_manager(self) -> Any:
        """Lazily initialize LearnedParametersManager when required."""
        manager = getattr(self, "param_manager", None)
        if manager is None:
            from src.persistence.learned_parameters import LearnedParametersManager  # noqa: PLC0415

            manager = LearnedParametersManager()
            manager.load()
            self.param_manager = manager
        return manager

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Backward-compatible instance wrapper over shared softmax utility."""
        return softmax(x, temperature)

    def _init_agent_state(
        self,
        *,
        window: int,
        n_features: int,
        symbol: str,
        timeframe: str,
        timeframe_minutes: int,
        broker: str,
        param_manager: Any,
    ) -> None:
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

    def _init_training_components(
        self,
        *,
        enable_training: bool,
        buffer_capacity: int,
        min_experiences: int,
        batch_size: int,
        state_dim: int,
        n_actions: int,
        learning_rate: float,
        gamma: float,
        tau: float,
        l2_weight: float,
        grad_clip_norm: float,
    ) -> None:
        from src.core.ddqn_network import DDQNNetwork  # noqa: PLC0415
        from src.utils.experience_buffer import ExperienceBuffer  # noqa: PLC0415

        self.enable_training = enable_training
        self.buffer = (
            ExperienceBuffer(capacity=buffer_capacity, timeframe_minutes=self.timeframe_minutes)
            if enable_training
            else None
        )
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.training_steps = 0
        self.last_state = None
        self.ddqn = (
            DDQNNetwork(
                state_dim=state_dim,
                n_actions=n_actions,
                learning_rate=learning_rate,
                gamma=gamma,
                tau=tau,
                l2_weight=l2_weight,
                grad_clip_norm=grad_clip_norm,
            )
            if enable_training
            else None
        )

    # ── add_experience ────────────────────────────────────────────────────────

    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: int | None = None,
    ) -> None:
        """Store a transition in the replay buffer."""
        if not self.enable_training or self.buffer is None:
            LOG.info(
                "[DIAG] %s.add_experience: SKIPPED — enable_training=%s, buffer=%s",
                self._AGENT_TAG,
                self.enable_training,
                self.buffer is not None,
            )
            return

        buf_before = self.buffer.tree.n_entries
        from src.utils.experience_buffer import RegimeSampling  # noqa: PLC0415

        # Convert None to UNKNOWN regime (default int value)
        regime_int = regime if regime is not None else RegimeSampling.UNKNOWN
        self.buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            regime=regime_int,
        )
        buf_after = self.buffer.tree.n_entries

        LOG.info(
            "[DIAG] %s.add_experience: action=%d, reward=%.4f, buffer_before=%d, buffer_after=%d, total_added=%d",
            self._AGENT_TAG,
            action,
            reward,
            buf_before,
            buf_after,
            self.buffer.total_added,
        )

    # ── train_step ────────────────────────────────────────────────────────────

    def train_step(self) -> dict[str, Any] | None:
        """Perform one training step using prioritised experience replay.

        Returns:
            Dictionary with training metrics, or None if insufficient data.
        """
        if not self.enable_training:
            return None

        buffer = self.buffer
        if buffer is None:
            return None

        if buffer.tree.n_entries < self.min_experiences:
            return None

        effective_batch = min(self.batch_size, buffer.tree.n_entries)
        batch = buffer.sample(batch_size=effective_batch)
        if batch is None:
            return None

        rewards = batch["rewards"]
        indices = batch["indices"]

        if not all(math.isfinite(r) for r in rewards):
            LOG.warning("[%s] Non-finite rewards in batch, skipping training", self._AGENT_TAG)
            return None

        if self.ddqn is not None:
            try:
                states = batch["states"].reshape(batch["states"].shape[0], -1).astype(np.float64)
                next_states = batch["next_states"].reshape(batch["next_states"].shape[0], -1).astype(np.float64)
                actions = batch["actions"].astype(np.intp)
                dones = batch["dones"].astype(np.float64)
                weights = batch["weights"].astype(np.float64)
                rewards_f = rewards.astype(np.float64)

                train_result = self.ddqn.train_batch(
                    states=states,
                    actions=actions,
                    rewards=rewards_f,
                    next_states=next_states,
                    dones=dones,
                    weights=weights,
                )

                td_errors = np.abs(train_result["td_errors"])
                td_errors = np.clip(td_errors, 0, TD_ERROR_CAP)
                buffer.update_priorities(indices, td_errors)

                metrics = {
                    "loss": train_result["loss"],
                    "mean_q": train_result["mean_q"],
                    "mean_td_error": train_result["mean_td_error"],
                    "max_td_error": train_result["max_td_error"],
                    "grad_norm": train_result["grad_norm"],
                    "mean_reward": float(np.mean(rewards)),
                }
            except Exception as e:
                LOG.error("[%s] DDQN train_batch failed: %s", self._AGENT_TAG, e, exc_info=True)
                td_errors = np.clip(np.abs(rewards), 0, TD_ERROR_CAP)
                buffer.update_priorities(indices, td_errors)
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
            td_errors = np.abs(rewards)
            td_errors = np.clip(td_errors, -TD_ERROR_CAP, TD_ERROR_CAP)
            buffer.update_priorities(indices, td_errors)
            metrics = {
                "loss": 0.0,
                "mean_q": 0.0,
                "mean_td_error": float(np.mean(td_errors)),
                "max_td_error": float(np.max(td_errors)),
                "mean_reward": float(np.mean(rewards)),
            }
            LOG.warning("[%s] No DDQN network - only updating priorities (no weight updates)", self._AGENT_TAG)

        self.training_steps += 1

        log_interval = (
            TRAINING_LOG_INTERVAL_EARLY if self.training_steps < TRAINING_STEPS_EARLY else TRAINING_LOG_INTERVAL_LATE
        )
        if self.training_steps % log_interval == 0:
            LOG.info(
                "[%s] Training step %d: loss=%.4f, mean_q=%.3f, mean_reward=%.4f, mean_td=%.4f, buffer=%d",
                self._AGENT_TAG,
                self.training_steps,
                metrics.get("loss", 0.0),
                metrics.get("mean_q", 0.0),
                metrics.get("mean_reward", 0.0),
                metrics.get("mean_td_error", 0.0),
                buffer.tree.n_entries,
            )

        return metrics

    # ── _train_step_torch ─────────────────────────────────────────────────────

    def _train_step_torch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Training step when use_torch=True (PyTorch model loaded from disk)."""
        buffer = self.buffer
        if self.ddqn is not None and buffer is not None:
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
            buffer.update_priorities(batch["indices"], np.abs(train_result["td_errors"]))
            return {
                "loss": train_result["loss"],
                "mean_q": train_result["mean_q"],
                "mean_td_error": train_result["mean_td_error"],
                "mean_reward": float(np.mean(batch["rewards"])),
            }

        LOG.warning("[%s] No DDQN network in torch path — priority-only update", self._AGENT_TAG)
        td_errors = np.abs(batch["rewards"])
        td_errors = np.clip(td_errors, 0, TD_ERROR_CAP)
        if buffer is not None:
            buffer.update_priorities(batch["indices"], td_errors)
        return {
            "loss": 0.0,
            "mean_q": 0.0,
            "mean_td_error": float(np.mean(td_errors)),
            "mean_reward": float(np.mean(batch["rewards"])),
        }

    # ── get_training_stats ────────────────────────────────────────────────────

    def get_training_stats(self) -> dict[str, Any]:
        """Get training statistics for monitoring."""
        if not self.enable_training or self.buffer is None:
            return {"enabled": False}

        buffer_stats = self.buffer.get_stats()

        stats = {
            "enabled": True,
            "training_steps": self.training_steps,
            "buffer_size": buffer_stats["size"],
            "buffer_utilization": buffer_stats["utilization"],
            "total_added": buffer_stats["total_added"],
            "total_sampled": buffer_stats["total_sampled"],
            "beta": buffer_stats["beta"],
            "ready_to_train": buffer_stats["size"] >= self.min_experiences,
        }
        # Let subclasses append agent-specific keys
        stats.update(self._extra_training_stats())
        return stats

    def _extra_training_stats(self) -> dict[str, Any]:
        """Override in subclass to add agent-specific stats keys."""
        return {}
