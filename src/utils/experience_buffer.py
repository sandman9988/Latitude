#!/usr/bin/env python3
"""Prioritized Experience Replay Buffer.
=====================================
Implements SumTree-based prioritized sampling for DDQN online learning.

Master Handbook alignment:
- Section: "PER Buffer + Online Learning" (Phase 3.5)
- Purpose: Continuous model adaptation via prioritized experience replay
- Architecture: SumTree for O(log n) sampling, TD-error based priorities

Key Features:
1. SumTree - Binary tree for efficient O(log n) priority sampling
2. ExperienceBuffer - Storage + sampling with staleness decay
3. Prioritized sampling - Sample high TD-error transitions more frequently
4. Staleness decay - Old experiences lose priority over time
5. Regime-aware weighting - Prioritize experiences from current regime

Usage:
    buffer = ExperienceBuffer(capacity=100_000)

    # Add experience (called after trade completes)
    buffer.add(
        state=state_vector,
        action=action,
        reward=shaped_reward,
        next_state=next_state_vector,
        done=True,
        regime=regime_type
    )

    # Sample batch for training
    batch = buffer.sample(batch_size=64)

    # Update priorities after training
    td_errors = calculate_td_errors(batch)
    buffer.update_priorities(batch['indices'], td_errors)

Performance:
- add(): O(log n)
- sample(): O(log n) per sample
- update_priorities(): O(log n) per update
- Memory: ~16 bytes per experience (64-bit floats)
"""

import contextlib
import logging
import math
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator, default_rng

from src.utils.sum_tree import SumTree

# Try to import AMD opts for float16 auto-detection (AMD GPU optimization)
try:
    from src.core.ddqn_network import AMD_OPTS

    _amd_opts_available = True
except ImportError:
    AMD_OPTS: dict = {}
    _amd_opts_available = False

LOG = logging.getLogger(__name__)
RNG: Generator = default_rng(42)

# ---------------------------------------------------------------------------
# Session-based staleness halflife
# ---------------------------------------------------------------------------
# Express staleness in trading-session units rather than raw seconds so the
# decay is instrument- and timeframe-agnostic.
#
# halflife_secs = n_sessions × session_bars × timeframe_minutes × 60
#              = n_sessions × (session_minutes / tf_min) × tf_min × 60
#              = n_sessions × session_minutes × 60
#
# timeframe_minutes cancels → the result is a pure wall-clock value that
# correctly represents "N trading sessions" regardless of bar size.
#
# Example (all yield 43 200 s = 12 h):
#   M1  → 1.5 × 480 bars × 1 min × 60  = 43 200 s
#   M5  → 1.5 × 96 bars  × 5 min × 60  = 43 200 s
#   H1  → 1.5 × 8 bars   × 60 min × 60 = 43 200 s
#   D1  → 1.5 × 1 bar    × 1440 min × 60 = 129 600 s (≈ 1.5 days, appropriate)
TRADING_SESSION_MINUTES: float = 480.0  # one FX intraday session (8 h)
HALFLIFE_SESSIONS: float = 1.5  # 50% decay after 1.5 sessions


def staleness_halflife_for_timeframe(
    timeframe_minutes: int,
    n_sessions: float = HALFLIFE_SESSIONS,
    session_minutes: float = TRADING_SESSION_MINUTES,
) -> float:
    """Return staleness halflife in seconds scaled to the trading timeframe.

    Args:
        timeframe_minutes: Bar duration in minutes (e.g. 5 for M5, 60 for H1).
        n_sessions: Number of trading sessions that span the halflife window.
            Default 1.5 → an experience from 1.5 sessions ago counts 50%.
        session_minutes: Duration of one trading session in minutes.
            Default 480 (8 h) covers standard FX and most equity sessions.

    Returns:
        Halflife in seconds.  The formula is:
            session_bars  = session_minutes / timeframe_minutes
            halflife_secs = n_sessions × session_bars × timeframe_minutes × 60
                          = n_sessions × session_minutes × 60
        timeframe_minutes cancels, making the result timeframe-agnostic in
        wall-clock units while remaining conceptually grounded in session units.

    """
    session_bars = session_minutes / max(1, timeframe_minutes)
    return n_sessions * session_bars * timeframe_minutes * 60.0


class RegimeSampling(IntEnum):
    """Regime types for prioritization weighting."""

    TRENDING = 0
    MEAN_REVERTING = 1
    TRANSITIONAL = 2
    UNKNOWN = 3


@dataclass
class Experience:
    """Single experience tuple for DDQN training."""

    state: np.ndarray  # State vector (7-10 dims)
    action: int  # Action taken: 0=SHORT, 1=FLAT, 2=LONG
    reward: float  # Shaped reward (from RewardShaper)
    next_state: np.ndarray  # Next state vector
    done: bool  # True if episode terminal
    timestamp: float  # Unix timestamp (for staleness)
    regime: int  # RegimeSampling enum value
    priority: float  # TD-error magnitude (updated during training)
    zeta: float = 1.0  # Regime damping ratio at time of experience (for continuous boost)


class ExperienceBuffer:
    """Prioritized Experience Replay buffer for DDQN online learning.

    Features:
    - TD-error based prioritization (high error = more important)
    - Staleness decay (old experiences lose priority)
    - Regime-aware weighting (prioritize current regime)
    - Efficient O(log n) sampling via SumTree
    - Float16 storage for memory efficiency (50% reduction)
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        staleness_halflife: float | None = None,
        regime_boost: float = 1.5,
        epsilon: float = 0.01,
        seed: int | None = None,
        timeframe_minutes: int = 5,
        use_float16: bool | None = None,  # None = auto-detect from AMD_OPTS
    ) -> None:
        """Initialize experience buffer.

        Args:
            capacity: Maximum experiences to store
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (0=no correction, 1=full correction)
            beta_increment: Beta increase per sample (annealing)
            staleness_halflife: Override halflife in seconds.  ``None`` (default)
                auto-computes from *timeframe_minutes* via
                :func:`staleness_halflife_for_timeframe`, producing a value
                grounded in trading-session units (instrument- and
                timeframe-agnostic).
            regime_boost: Priority multiplier for experiences from current regime
            epsilon: Small constant to ensure non-zero priorities
            seed: Random seed for reproducibility (default: None for non-deterministic)
            timeframe_minutes: Bar duration in minutes.  Used to auto-compute
                *staleness_halflife* when that argument is ``None``.

        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.staleness_halflife = (
            staleness_halflife
            if staleness_halflife is not None
            else staleness_halflife_for_timeframe(timeframe_minutes)
        )
        self.regime_boost = regime_boost
        self.epsilon = epsilon
        self.timeframe_minutes = timeframe_minutes

        # SumTree for efficient sampling (stores priorities only)
        self.tree = SumTree(capacity, seed=seed)

        # Data storage (circular buffer for experiences)
        self.data: list[Experience | None] = [None] * capacity
        self.write_idx = 0

        # Current regime (for regime-aware weighting)
        self.current_regime: RegimeSampling = RegimeSampling.UNKNOWN
        self.current_zeta: float = 1.0  # Current damping ratio for continuous boost

        # Float16 storage for memory efficiency (50% reduction on AMD GPUs)
        # Auto-detect from AMD opts if not specified
        if use_float16 is None:
            self._use_float16 = AMD_OPTS.get("is_amd", False) if _amd_opts_available else False
        else:
            self._use_float16 = use_float16

        # Stats
        self.total_added = 0
        self.total_sampled = 0

        LOG.info(
            "ExperienceBuffer initialized: capacity=%d, alpha=%.2f, beta=%.2f, "
            "staleness_halflife=%.0fs (tf=%dmin, %.1f sessions), regime_boost=%.2f, float16=%s",
            capacity,
            alpha,
            beta,
            self.staleness_halflife,
            timeframe_minutes,
            self.staleness_halflife / max(1.0, TRADING_SESSION_MINUTES * 60),
            regime_boost,
            self._use_float16,
        )

    def set_current_regime(self, regime: int) -> None:
        """Update current regime for prioritization weighting.

        Args:
            regime: RegimeSampling enum value

        """
        self.current_regime = RegimeSampling(regime)
        LOG.debug("Current regime updated: %s", self.current_regime.name)

    def set_current_zeta(self, zeta: float) -> None:
        """Update current damping ratio for continuous regime boost.

        Args:
            zeta: Damping ratio from RegimeDetector (lower = trending)

        """
        self.current_zeta = zeta

    def _calculate_staleness_weight(self, timestamp: float) -> float:
        """Calculate staleness decay weight.

        Args:
            timestamp: Experience timestamp (Unix time)

        Returns:
            Decay weight in [0, 1] (exponential decay)

        """
        age = time.time() - timestamp

        # Defensive: Handle negative age (clock skew)
        if age < 0:
            return 1.0

        # Exponential decay: weight = 0.5^(age / halflife)
        decay = float(np.exp(-age * np.log(2) / self.staleness_halflife))

        return float(max(0.0001, decay))  # Ensure non-zero

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: int = RegimeSampling.UNKNOWN,
        zeta: float | None = None,
    ) -> bool:
        """Add experience to buffer.

        Args:
            state: State vector (must be numpy array)
            action: Action taken (0=SHORT, 1=FLAT, 2=LONG)
            reward: Shaped reward from RewardShaper
            next_state: Next state vector
            done: True if terminal state
            regime: RegimeSampling enum value
            zeta: Damping ratio at time of experience (uses current_zeta if None)

        Returns:
            True if experience was added, False if validation failed

        """
        # Defensive: Validate inputs
        if not isinstance(state, np.ndarray) or not isinstance(next_state, np.ndarray):
            LOG.error(
                "Invalid state type: state=%s, next_state=%s (experience not added)", type(state), type(next_state),
            )
            return False

        if state.size == 0 or next_state.size == 0:
            LOG.error("Empty state vectors (experience not added)")
            return False

        if not math.isfinite(reward):
            LOG.error("Non-finite reward: %.4f (experience not added)", reward)
            return False

        if action not in (0, 1, 2):
            LOG.error("Invalid action: %d (experience not added)", action)
            return False

        # Validate regime is a valid enum value
        try:
            regime_enum = RegimeSampling(regime)
        except ValueError:
            LOG.exception("Invalid regime value: %d (must be 0-3, experience not added)", regime)
            return False

        # Use validated regime_enum (stored for regime-aware weighting)
        validated_regime = regime_enum

        # Convert to float16 for memory efficiency (50% reduction) if enabled
        # This is transparent to the caller - states are converted back to float32 during sampling
        # Validate precision loss for critical features
        if self._use_float16:
            state_f16 = state.astype(np.float16)
            next_state_f16 = next_state.astype(np.float16)

            # Check for precision loss (float16 has ~3 decimal digits of precision)
            # Round-trip conversion to detect catastrophic loss
            state_roundtrip = state_f16.astype(np.float32)
            next_state_roundtrip = next_state_f16.astype(np.float32)

            state_loss = np.max(np.abs(state - state_roundtrip))
            next_state_loss = np.max(np.abs(next_state - next_state_roundtrip))

            # Relative precision threshold: allow up to 0.1% of feature magnitude.
            state_mag = max(np.max(np.abs(state)), 1e-6)
            next_mag = max(np.max(np.abs(next_state)), 1e-6)
            if state_loss / state_mag > 0.001 or next_state_loss / next_mag > 0.001:
                LOG.warning(
                    "Float16 precision loss: state_rel=%.4f%% next_rel=%.4f%% — using float16 anyway",
                    state_loss / state_mag * 100, next_state_loss / next_mag * 100)


            state_stored = state_f16
            next_state_stored = next_state_f16
        else:
            state_stored = state.copy()
            next_state_stored = next_state.copy()

        # Create experience
        exp = Experience(
            state=state_stored,  # Stored in float16 if enabled
            action=action,
            reward=reward,
            next_state=next_state_stored,
            done=done,
            timestamp=time.time(),
            regime=validated_regime,  # Use validated enum
            priority=1.0,  # Will be updated during training
            zeta=zeta if zeta is not None else self.current_zeta,
        )

        # Initial priority: max existing priority (ensures new experiences sampled at least once)
        max_priority = float(
            np.max(self.tree.tree[self.tree.capacity - 1 : self.tree.capacity - 1 + self.tree.n_entries])
            if self.tree.n_entries > 0
            else 1.0,
        )

        # Defensive: Cap max priority
        max_priority = min(max_priority, 1000.0)

        # Store experience in data buffer
        self.data[self.write_idx] = exp

        # Add priority to tree
        self.tree.add(max_priority)

        # Update write pointer
        self.write_idx = (self.write_idx + 1) % self.capacity

        self.total_added += 1

        if self.total_added % 1000 == 0:
            LOG.info("ExperienceBuffer: added %d experiences (size=%d)", self.total_added, self.tree.n_entries)

        return True

    def sample(self, batch_size: int = 64) -> dict[str, Any] | None:
        """Sample batch of experiences with prioritized sampling.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Dictionary with:
                - states: (batch_size, state_dim) array
                - actions: (batch_size,) array
                - rewards: (batch_size,) array
                - next_states: (batch_size, state_dim) array
                - dones: (batch_size,) array
                - indices: (batch_size,) array (for priority updates)
                - weights: (batch_size,) array (importance sampling weights)

        """
        if self.tree.n_entries < batch_size:
            LOG.warning("Insufficient experiences: have %d, need %d", self.tree.n_entries, batch_size)
            return None

        # Sample storage
        states_list: list[np.ndarray] = []
        actions_list: list[int] = []
        rewards_list: list[float] = []
        next_states_list: list[np.ndarray] = []
        dones_list: list[bool] = []
        indices_list: list[int] = []
        raw_priorities_list: list[float] = []  # Raw tree priorities for correct IS weights
        priority_updates: list[tuple[int, float]] = []  # (tree_idx, adjusted_priority)

        # Divide total priority range into batch_size segments
        segment_size = self.tree.total() / batch_size

        for i in range(batch_size):
            # Sample uniformly within segment
            a = segment_size * i
            b = segment_size * (i + 1)
            sample_value = float(RNG.uniform(a, b))

            # Sample data index from tree
            data_idx = self.tree.sample(sample_value)

            # Get experience from data buffer
            exp = self.data[data_idx]

            # Defensive: Handle None experiences (shouldn't happen but be safe)
            if exp is None:
                LOG.warning("Sampled None experience at data_idx=%d", data_idx)
                continue

            # Get raw priority for this experience (actual sampling probability basis)
            priority = self.tree.get_priority(data_idx)

            # Compute adjusted priority (staleness-decay + regime-boost) and
            # queue it for a post-loop tree update.  We must NOT update the tree
            # inside this loop because doing so alters tree.total(), which
            # invalidates the stratified segment boundaries for later iterations.
            staleness_weight = self._calculate_staleness_weight(exp.timestamp)
            # Continuous regime boost: scale by ζ similarity between the
            # experience and the current market condition.  Experiences from
            # a similar regime get a stronger boost than those from a very
            # different regime.
            zeta_similarity = max(0.0, 1.0 - abs(getattr(exp, "zeta", self.current_zeta) - self.current_zeta))
            regime_weight = 1.0 + (self.regime_boost - 1.0) * zeta_similarity
            adjusted_priority = max(priority * staleness_weight * regime_weight, self.epsilon)
            tree_idx = data_idx + self.tree.capacity - 1
            priority_updates.append((tree_idx, adjusted_priority))

            # Store
            states_list.append(exp.state)
            actions_list.append(int(exp.action))
            rewards_list.append(float(exp.reward))
            next_states_list.append(exp.next_state)
            dones_list.append(bool(exp.done))
            indices_list.append(data_idx)  # Store data index for updates
            raw_priorities_list.append(float(priority))  # Raw priority for IS weights

        # Apply staleness/regime priority adjustments after sampling is complete
        for tree_idx, adj_p in priority_updates:
            self.tree.update(tree_idx, adj_p)

        # Defensive: Check we got enough samples
        if len(states_list) < batch_size // 2:
            LOG.warning("Sample failed: only got %d/%d experiences", len(states_list), batch_size)
            return None

        # Convert to numpy arrays
        # If states were stored in float16, convert back to float32 for training
        # (BF16 training will handle precision conversion via autocast)
        states = np.asarray(states_list, dtype=np.float32)
        actions = np.asarray(actions_list, dtype=np.int32)
        rewards = np.asarray(rewards_list, dtype=np.float32)
        next_states = np.asarray(next_states_list, dtype=np.float32)
        dones = np.asarray(dones_list, dtype=np.bool_)
        indices = np.asarray(indices_list, dtype=np.int32)
        raw_priorities = np.asarray(raw_priorities_list, dtype=np.float32)

        # Importance sampling weights: debias the prioritised distribution.
        # P(i) = raw_priority_i / total  (actual probability used by SumTree)
        # w_i = (1 / (N * P(i)))^β  — normalised by max for numerical stability.
        # We intentionally use raw (unadjusted) priorities here; staleness/regime
        # adjustments were already written back into the tree during sampling so
        # future draws reflect them without distorting the IS correction.
        probs = np.clip(raw_priorities / (self.tree.total() + 1e-8), 1e-10, 1.0)
        weights = (1.0 / (self.tree.n_entries * probs)) ** self.beta
        weights = weights / weights.max()  # Normalize to [0, 1]

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        self.total_sampled += len(states)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "indices": indices,
            "weights": weights,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for sampled experiences based on TD-errors.

        Args:
            indices: Tree indices from sample()
            td_errors: TD-error magnitudes (|target - prediction|)

        """
        # Defensive: Validate inputs
        if len(indices) != len(td_errors):
            LOG.warning("Mismatched lengths: indices=%d, td_errors=%d", len(indices), len(td_errors))
            return

        for data_idx, td_error in zip(indices, td_errors, strict=True):
            # Defensive: Validate TD-error
            if not math.isfinite(td_error):
                LOG.warning("Non-finite TD-error: %.4f, skipping", td_error)
                continue

            # Cap extreme TD-errors (prevent priority explosion)
            clamped_td_error = max(-10.0, min(10.0, td_error))

            # Priority = (|TD-error| + ε)^α
            priority = (abs(clamped_td_error) + self.epsilon) ** self.alpha

            # Update tree (convert data_idx to tree_idx)
            tree_idx = int(data_idx) + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)

    @property
    def size(self) -> int:
        """Get current buffer size.

        Returns:
            Number of experiences in buffer

        """
        return self.tree.n_entries

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics for monitoring.

        Returns:
            Dictionary with buffer stats

        """
        return {
            "size": self.tree.n_entries,
            "capacity": self.capacity,
            "utilization": self.tree.n_entries / self.capacity,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
            "beta": self.beta,
            "current_regime": RegimeSampling(self.current_regime).name,
            "current_zeta": self.current_zeta,
            "total_priority": self.tree.total(),
        }

    def save(self, filepath: str) -> bool:
        """Save buffer state to disk for persistence across restarts.

        Serializes all experiences, priorities, and metadata so the buffer
        can be restored exactly as it was.  Uses atomic write (temp file +
        rename) to prevent corruption from SIGKILL during save.

        Args:
            filepath: Path to save the buffer (without extension, .npz added)

        Returns:
            True if save succeeded

        """
        import tempfile

        try:
            n = self.tree.n_entries
            if n == 0:
                LOG.info("[BUFFER] Nothing to save (empty)")
                return True

            # Collect all valid experiences into arrays.
            # Use the most-common flat state size as canonical — this discards any
            # stale experiences left over from a prior offline-training run that
            # used a different feature set, preventing np.array() from raising
            # "inhomogeneous shape" when mixing old and new state dimensions.
            states, actions, rewards, next_states, dones = [], [], [], [], []
            timestamps, regimes, priorities_list = [], [], []

            canonical_state_size: int | None = None
            dropped_save = 0
            for i in range(n):
                exp = self.data[i]
                if exp is None:
                    continue
                state_flat = np.asarray(exp.state, dtype=np.float32).ravel()
                next_flat = np.asarray(exp.next_state, dtype=np.float32).ravel()
                if canonical_state_size is None:
                    canonical_state_size = state_flat.size
                if state_flat.size != canonical_state_size or next_flat.size != canonical_state_size:
                    dropped_save += 1
                    continue
                states.append(state_flat)
                actions.append(exp.action)
                rewards.append(exp.reward)
                next_states.append(next_flat)
                dones.append(exp.done)
                timestamps.append(exp.timestamp)
                regimes.append(exp.regime)
                # Get priority from tree leaf
                leaf_idx = i + self.tree.capacity - 1
                priorities_list.append(self.tree.tree[leaf_idx])

            if dropped_save:
                LOG.warning(
                    "[BUFFER] Dropped %d/%d experiences with mismatched state size (canonical=%s) during save",
                    dropped_save, n, canonical_state_size,
                )
            if not states:
                LOG.info("[BUFFER] Nothing to save after size filtering")
                return True

            dest = Path(filepath)
            if not dest.suffix:
                dest = dest.with_suffix(".npz")
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write: save to temp file in same directory, then rename.
            # os.rename is atomic on POSIX when src and dst are on the same
            # filesystem, so a SIGKILL during save can only leave behind a
            # stale temp file — the previous checkpoint stays intact.
            #
            # IMPORTANT: suffix must be ".npz" so numpy does NOT append another
            # ".npz" — otherwise the data lands in tmp_xxx.npz.tmp.npz while
            # os.replace renames the empty original, producing a 0-byte file.
            fd, tmp_path = tempfile.mkstemp(
                suffix=".npz",
                dir=str(dest.parent),
            )
            os.close(fd)
            try:
                np.savez_compressed(
                    tmp_path,
                    states=np.array(states),
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    next_states=np.array(next_states),
                    dones=np.array(dones),
                    timestamps=np.array(timestamps),
                    regimes=np.array(regimes),
                    priorities=np.array(priorities_list),
                    # Metadata
                    write_idx=self.write_idx,
                    total_added=self.total_added,
                    total_sampled=self.total_sampled,
                    beta=self.beta,
                    current_regime=int(self.current_regime),
                )
                os.replace(tmp_path, str(dest))
            except Exception:
                # Clean up temp file on failure
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()
                raise

            LOG.info("[BUFFER] Saved %d experiences to %s", n, dest)
            return True
        except Exception as e:
            LOG.error("[BUFFER] Failed to save: %s", e, exc_info=True)
            return False

    @staticmethod
    def _resolve_load_path(filepath: str) -> Path | None:
        # Handle both with and without .npz extension
        path = Path(filepath)
        if path.exists():
            return path
        npz_path = Path(f"{filepath}.npz")
        return npz_path if npz_path.exists() else None

    def _restore_loaded_experiences(self, data, filepath: str) -> int:
        states = data["states"]
        n = min(len(states), self.capacity)
        if len(states) > self.capacity:
            LOG.warning("[BUFFER] Saved buffer (%d) exceeds capacity (%d), truncating", len(states), self.capacity)
        self.tree = SumTree(self.capacity, seed=None)
        self.data = [None] * self.capacity
        canonical_state_size: int | None = None
        dropped = 0
        slot = 0
        for i in range(n):
            state_flat = states[i].astype(np.float32).ravel()
            next_flat = data["next_states"][i].astype(np.float32).ravel()
            if canonical_state_size is None:
                canonical_state_size = state_flat.size
            if state_flat.size != canonical_state_size or next_flat.size != canonical_state_size:
                dropped += 1
                continue
            self._restore_loaded_experience_slot(data, i, slot, state_flat, next_flat)
            slot += 1
        if dropped:
            LOG.warning(
                "[BUFFER] Dropped %d/%d experiences with mismatched state size (canonical=%s) from %s",
                dropped, n, canonical_state_size, filepath,
            )
        self._rebuild_tree_sums()
        return slot

    def _restore_loaded_experience_slot(
        self,
        data,
        index: int,
        slot: int,
        state_flat: np.ndarray,
        next_flat: np.ndarray,
    ) -> None:
        exp = Experience(
            state=state_flat,
            action=int(data["actions"][index]),
            reward=float(data["rewards"][index]),
            next_state=next_flat,
            done=bool(data["dones"][index]),
            timestamp=float(data["timestamps"][index]),
            regime=int(data["regimes"][index]),
            priority=float(data["priorities"][index]),
        )
        self.data[slot] = exp
        tree_idx = slot + self.tree.capacity - 1
        self.tree.tree[tree_idx] = float(data["priorities"][index])
        self.tree.n_entries = slot + 1
        self.tree.write_index = (slot + 1) % self.capacity

    def _rebuild_tree_sums(self) -> None:
        for i in range(self.tree.capacity - 2, -1, -1):
            self.tree.tree[i] = self.tree.tree[2 * i + 1] + self.tree.tree[2 * i + 2]

    def _restore_loaded_metadata(self, data) -> None:
        # Experiences are compacted into slots ``0..n-1`` during load, so the next
        # write must follow the loaded block. Do not blindly reuse saved write_idx.
        self.write_idx = self.tree.write_index
        self.total_added = int(data["total_added"])
        self.total_sampled = int(data["total_sampled"])
        self.beta = float(data["beta"])
        self.current_regime = RegimeSampling(int(data["current_regime"]))

    def load(self, filepath: str) -> bool:
        """Load buffer state from disk."""
        path = self._resolve_load_path(filepath)
        if path is None:
            LOG.warning("[BUFFER] No saved buffer found at %s", filepath)
            return False
        try:
            data = np.load(str(path), allow_pickle=False)
            if len(data["states"]) == 0:
                LOG.info("[BUFFER] Loaded empty buffer from %s", filepath)
                return True
            n = self._restore_loaded_experiences(data, filepath)
            self._restore_loaded_metadata(data)
            LOG.info("[BUFFER] Loaded %d experiences from %s", n, filepath)
            return True
        except Exception as e:
            LOG.error("[BUFFER] Failed to load: %s", e, exc_info=True)
            return False


# ============================================
# Module Testing
# ============================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    # Test 1: SumTree basic operations

    tree = SumTree(capacity=8)
    test_data: list[Experience | None] = [None] * 8

    # Add experiences
    for idx in range(5):
        test_exp = Experience(
            state=np.array([idx]),
            action=idx % 3,
            reward=float(idx),
            next_state=np.array([idx + 1]),
            done=False,
            timestamp=time.time(),
            regime=RegimeSampling.UNKNOWN,
            priority=1.0,
        )
        test_data[idx] = test_exp
        tree.add(priority=float(idx + 1))


    # Sample
    sample_data_idx = tree.sample(tree.total() * 0.5)
    sample_priority = tree.get_priority(sample_data_idx)
    sample_exp = test_data[sample_data_idx]
    if sample_exp is None:
        pass
    else:
        pass

    # Test 2: ExperienceBuffer sampling

    buffer = ExperienceBuffer(capacity=1000)

    # Add experiences
    for idx2 in range(200):
        buffer.add(
            state=RNG.standard_normal(7),
            action=idx2 % 3,
            reward=float(RNG.standard_normal()),
            next_state=RNG.standard_normal(7),
            done=(idx2 % 50 == 0),
            regime=RegimeSampling.TRENDING if idx2 % 2 == 0 else RegimeSampling.MEAN_REVERTING,
        )

    # Sample batch
    batch = buffer.sample(batch_size=32)

    if batch:
        min_weight = batch["weights"].min()
        max_weight = batch["weights"].max()

    # Test 3: Priority updates

    if batch:
        # Simulate TD-errors
        test_td_errors = RNG.uniform(0.0, 2.0, size=len(batch["indices"]))


        buffer.update_priorities(batch["indices"], test_td_errors)


    # Test 4: Staleness decay

    # Add old experience
    old_timestamp = time.time() - 86400  # 1 day ago
    weight_old = buffer._calculate_staleness_weight(old_timestamp)

    # Add new experience
    new_timestamp = time.time()
    weight_new = buffer._calculate_staleness_weight(new_timestamp)


    # Test 5: Regime-aware weighting

    buffer.set_current_regime(RegimeSampling.TRENDING)

    # Add experiences with different regimes
    for _ in range(50):
        buffer.add(
            state=RNG.standard_normal(7),
            action=0,
            reward=1.0,
            next_state=RNG.standard_normal(7),
            done=False,
            regime=RegimeSampling.TRENDING,
        )

    for _ in range(50):
        buffer.add(
            state=RNG.standard_normal(7),
            action=0,
            reward=1.0,
            next_state=RNG.standard_normal(7),
            done=False,
            regime=RegimeSampling.MEAN_REVERTING,
        )

    # Sample and check regime distribution
    batch = buffer.sample(batch_size=64)

    if batch:
        # Count regimes in batch (need to track in Experience, not currently stored in batch)
        pass

    # Test 6: Stats

    stats = buffer.get_stats()
    for value in stats.values():
        if isinstance(value, float):
            pass
        else:
            pass
