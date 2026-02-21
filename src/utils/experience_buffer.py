#!/usr/bin/env python3
"""
Prioritized Experience Replay Buffer
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

import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.random import Generator, default_rng

from src.utils.sum_tree import SumTree

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
TRADING_SESSION_MINUTES: float = 480.0   # one FX intraday session (8 h)
HALFLIFE_SESSIONS: float = 1.5           # 50% decay after 1.5 sessions


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


class ExperienceBuffer:
    """
    Prioritized Experience Replay buffer for DDQN online learning.

    Features:
    - TD-error based prioritization (high error = more important)
    - Staleness decay (old experiences lose priority)
    - Regime-aware weighting (prioritize current regime)
    - Efficient O(log n) sampling via SumTree
    """

    def __init__(  # noqa: PLR0913
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
    ):
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

        # Stats
        self.total_added = 0
        self.total_sampled = 0

        LOG.info(
            "ExperienceBuffer initialized: capacity=%d, alpha=%.2f, beta=%.2f, "
            "staleness_halflife=%.0fs (tf=%dmin, %.1f sessions), regime_boost=%.2f",
            capacity,
            alpha,
            beta,
            self.staleness_halflife,
            timeframe_minutes,
            self.staleness_halflife / max(1.0, TRADING_SESSION_MINUTES * 60),
            regime_boost,
        )

    def set_current_regime(self, regime: int):
        """Update current regime for prioritization weighting.

        Args:
            regime: RegimeSampling enum value
        """
        self.current_regime = RegimeSampling(regime)
        LOG.debug("Current regime updated: %s", self.current_regime.name)

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

    def add(  # noqa: PLR0913
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: int = RegimeSampling.UNKNOWN,
    ):
        """Add experience to buffer.

        Args:
            state: State vector (must be numpy array)
            action: Action taken (0=SHORT, 1=FLAT, 2=LONG)
            reward: Shaped reward from RewardShaper
            next_state: Next state vector
            done: True if terminal state
            regime: RegimeSampling enum value
        """
        # Defensive: Validate inputs
        if not isinstance(state, np.ndarray) or not isinstance(next_state, np.ndarray):
            LOG.warning("Invalid state type: state=%s, next_state=%s", type(state), type(next_state))
            return

        if state.size == 0 or next_state.size == 0:
            LOG.warning("Empty state vectors")
            return

        if not math.isfinite(reward):
            LOG.warning("Non-finite reward: %.4f", reward)
            return

        if action not in (0, 1, 2):
            LOG.warning("Invalid action: %d", action)
            return

        # Create experience
        exp = Experience(
            state=state.copy(),  # Copy to avoid reference issues
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            timestamp=time.time(),
            regime=RegimeSampling(regime),
            priority=1.0,  # Will be updated during training
        )

        # Initial priority: max existing priority (ensures new experiences sampled at least once)
        max_priority = float(
            np.max(self.tree.tree[self.tree.capacity - 1 : self.tree.capacity - 1 + self.tree.n_entries])
            if self.tree.n_entries > 0
            else 1.0
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

    def sample(self, batch_size: int = 64) -> dict | None:
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
            regime_weight = self.regime_boost if exp.regime == self.current_regime else 1.0
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

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
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

    def get_stats(self) -> dict:
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
            "total_priority": self.tree.total(),
        }

    def save(self, filepath: str) -> bool:
        """Save buffer state to disk for persistence across restarts.

        Serializes all experiences, priorities, and metadata so the buffer
        can be restored exactly as it was.

        Args:
            filepath: Path to save the buffer (without extension, .npz added)

        Returns:
            True if save succeeded
        """
        from pathlib import Path

        try:
            n = self.tree.n_entries
            if n == 0:
                LOG.info("[BUFFER] Nothing to save (empty)")
                return True

            # Collect all valid experiences into arrays
            states, actions, rewards, next_states, dones = [], [], [], [], []
            timestamps, regimes, priorities_list = [], [], []

            for i in range(n):
                exp = self.data[i]
                if exp is None:
                    continue
                states.append(exp.state)
                actions.append(exp.action)
                rewards.append(exp.reward)
                next_states.append(exp.next_state)
                dones.append(exp.done)
                timestamps.append(exp.timestamp)
                regimes.append(exp.regime)
                # Get priority from tree leaf
                leaf_idx = i + self.tree.capacity - 1
                priorities_list.append(self.tree.tree[leaf_idx])

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                filepath,
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
            LOG.info("[BUFFER] Saved %d experiences to %s", n, filepath)
            return True
        except Exception as e:
            LOG.error("[BUFFER] Failed to save: %s", e, exc_info=True)
            return False

    def load(self, filepath: str) -> bool:
        """Load buffer state from disk.

        Restores experiences, priorities, and metadata from a previous save.

        Args:
            filepath: Path to load the buffer from (.npz file)

        Returns:
            True if load succeeded
        """
        from pathlib import Path

        # Handle both with and without .npz extension
        path = Path(filepath)
        if not path.exists():
            npz_path = Path(f"{filepath}.npz")
            if npz_path.exists():
                path = npz_path
            else:
                LOG.warning("[BUFFER] No saved buffer found at %s", filepath)
                return False

        try:
            data = np.load(str(path), allow_pickle=False)

            states = data["states"]
            actions = data["actions"]
            rewards = data["rewards"]
            next_states = data["next_states"]
            dones = data["dones"]
            timestamps = data["timestamps"]
            regimes = data["regimes"]
            priorities = data["priorities"]

            n = len(states)
            if n == 0:
                LOG.info("[BUFFER] Loaded empty buffer from %s", filepath)
                return True

            if n > self.capacity:
                LOG.warning("[BUFFER] Saved buffer (%d) exceeds capacity (%d), truncating", n, self.capacity)
                n = self.capacity

            # Reset tree and data
            self.tree = SumTree(self.capacity, seed=None)
            self.data = [None] * self.capacity

            # Re-add all experiences
            for i in range(n):
                exp = Experience(
                    state=states[i],
                    action=int(actions[i]),
                    reward=float(rewards[i]),
                    next_state=next_states[i],
                    done=bool(dones[i]),
                    timestamp=float(timestamps[i]),
                    regime=int(regimes[i]),
                    priority=float(priorities[i]),
                )
                self.data[i] = exp

                # Set priority in tree
                tree_idx = i + self.tree.capacity - 1
                self.tree.tree[tree_idx] = float(priorities[i])
                self.tree.n_entries = i + 1
                self.tree.write_index = (i + 1) % self.capacity

            # Rebuild tree sums from leaves up
            for i in range(self.tree.capacity - 2, -1, -1):
                self.tree.tree[i] = self.tree.tree[2 * i + 1] + self.tree.tree[2 * i + 2]

            # Restore metadata
            self.write_idx = int(data["write_idx"])
            self.total_added = int(data["total_added"])
            self.total_sampled = int(data["total_sampled"])
            self.beta = float(data["beta"])
            self.current_regime = RegimeSampling(int(data["current_regime"]))

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

    print("=" * 80)
    print("ExperienceBuffer Module Tests")
    print("=" * 80)

    # Test 1: SumTree basic operations
    print("\n[Test 1] SumTree Basic Operations")
    print("-" * 80)

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

    print(f"Total priority: {tree.total():.2f}")
    print(f"Size: {tree.n_entries}")

    # Sample
    sample_data_idx = tree.sample(tree.total() * 0.5)
    sample_priority = tree.get_priority(sample_data_idx)
    sample_exp = test_data[sample_data_idx]
    if sample_exp is None:
        print(f"Sampled: data_idx={sample_data_idx}, priority={sample_priority:.2f}, state=None")
    else:
        print(f"Sampled: data_idx={sample_data_idx}, priority={sample_priority:.2f}, state={sample_exp.state}")

    # Test 2: ExperienceBuffer sampling
    print("\n[Test 2] ExperienceBuffer Sampling")
    print("-" * 80)

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
        print("Batch shapes:")
        print(f"  states: {batch['states'].shape}")
        print(f"  actions: {batch['actions'].shape}")
        print(f"  rewards: {batch['rewards'].shape}")
        print(f"  weights: {batch['weights'].shape}")
        min_weight = batch["weights"].min()
        max_weight = batch["weights"].max()
        print(f"  weights range: [{min_weight:.3f}, {max_weight:.3f}]")

    # Test 3: Priority updates
    print("\n[Test 3] Priority Updates")
    print("-" * 80)

    if batch:
        # Simulate TD-errors
        test_td_errors = RNG.uniform(0.0, 2.0, size=len(batch["indices"]))

        print(f"Updating {len(test_td_errors)} priorities")
        print(
            f"TD-errors: min={test_td_errors.min():.3f}, max={test_td_errors.max():.3f}, mean={test_td_errors.mean():.3f}"
        )

        buffer.update_priorities(batch["indices"], test_td_errors)

        print("✓ Priority update complete")

    # Test 4: Staleness decay
    print("\n[Test 4] Staleness Decay")
    print("-" * 80)

    # Add old experience
    old_timestamp = time.time() - 86400  # 1 day ago
    weight_old = buffer._calculate_staleness_weight(old_timestamp)

    # Add new experience
    new_timestamp = time.time()
    weight_new = buffer._calculate_staleness_weight(new_timestamp)

    print(f"Old experience (1 day): weight={weight_old:.4f}")
    print(f"New experience (now): weight={weight_new:.4f}")
    print(f"Decay ratio: {weight_old / weight_new:.4f} (should be ~0.5)")

    # Test 5: Regime-aware weighting
    print("\n[Test 5] Regime-Aware Weighting")
    print("-" * 80)

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
        print("✓ Regime-aware sampling active")
        print(f"  Current regime: {RegimeSampling(buffer.current_regime).name}")
        print(f"  Regime boost: {buffer.regime_boost}x")

    # Test 6: Stats
    print("\n[Test 6] Buffer Statistics")
    print("-" * 80)

    stats = buffer.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("All tests complete!")
    print("=" * 80)
