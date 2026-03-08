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

import numpy as np

from src.constants import TD_ERROR_CAP, TRAINING_LOG_INTERVAL_EARLY, TRAINING_LOG_INTERVAL_LATE, TRAINING_STEPS_EARLY

LOG = logging.getLogger(__name__)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature for confidence calculation."""
    exp_x = np.exp((x - np.max(x)) / temperature)
    return exp_x / exp_x.sum()


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

    _AGENT_TAG: str = "AGENT"        # overridden by each subclass

    # ── add_experience ────────────────────────────────────────────────────────

    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: str | None = None,
    ) -> None:
        """Store a transition in the replay buffer."""
        if not self.enable_training or self.buffer is None:
            LOG.info(
                "[DIAG] %s.add_experience: SKIPPED — enable_training=%s, buffer=%s",
                self._AGENT_TAG, self.enable_training, self.buffer is not None,
            )
            return

        buf_before = self.buffer.tree.n_entries
        self.buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            regime=regime,
        )
        buf_after = self.buffer.tree.n_entries

        LOG.info(
            "[DIAG] %s.add_experience: action=%d, reward=%.4f, "
            "buffer_before=%d, buffer_after=%d, total_added=%d",
            self._AGENT_TAG, action, reward, buf_before, buf_after, self.buffer.total_added,
        )

    # ── train_step ────────────────────────────────────────────────────────────

    def train_step(self) -> dict | None:
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

    def _train_step_torch(self, batch: dict) -> dict:
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

    def get_training_stats(self) -> dict:
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

    def _extra_training_stats(self) -> dict:
        """Override in subclass to add agent-specific stats keys."""
        return {}
