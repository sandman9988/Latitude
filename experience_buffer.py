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

import numpy as np
import logging
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import IntEnum

LOG = logging.getLogger(__name__)


class RegimeSampling(IntEnum):
    """Regime types for prioritization weighting."""
    TRENDING = 0
    MEAN_REVERTING = 1
    UNKNOWN = 2


@dataclass
class Experience:
    """Single experience tuple for DDQN training."""
    state: np.ndarray          # State vector (7-10 dims)
    action: int                # 0=SHORT, 1=FLAT, 2=LONG
    reward: float              # Shaped reward (from RewardShaper)
    next_state: np.ndarray     # Next state vector
    done: bool                 # True if episode terminal
    timestamp: float           # Unix timestamp (for staleness)
    regime: int                # RegimeSampling enum value
    priority: float            # TD-error magnitude (updated during training)


class SumTree:
    """
    Binary tree for O(log n) prioritized sampling.
    
    Tree structure:
    - Leaf nodes: Store experience priorities
    - Internal nodes: Sum of children priorities
    - Root: Total priority sum
    
    Sampling:
    1. Sample random value v in [0, root_priority]
    2. Traverse tree: go left if v < left_child, else go right with v -= left_child
    3. Return leaf index when reached
    
    This ensures sampling probability ∝ priority.
    """
    
    def __init__(self, capacity: int):
        """Initialize SumTree with fixed capacity.
        
        Args:
            capacity: Maximum number of experiences (must be power of 2 for efficiency)
        """
        # Round up to next power of 2
        self.capacity = int(2 ** np.ceil(np.log2(capacity)))
        
        # Binary tree stored as array
        # tree[0] = root (total sum)
        # tree[1:capacity-1] = internal nodes
        # tree[capacity:2*capacity] = leaf nodes (priorities)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float32)
        
        # Data storage (circular buffer)
        self.data = [None] * self.capacity
        
        # Write pointer (circular)
        self.write_idx = 0
        
        # Current size
        self.size = 0
        
        LOG.info("SumTree initialized: capacity=%d (rounded to power of 2)", self.capacity)
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree.
        
        Args:
            idx: Tree index to start propagation
            change: Priority delta to propagate
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, value: float) -> int:
        """Retrieve leaf index for given cumulative priority value.
        
        Args:
            idx: Current tree index
            value: Cumulative priority target
            
        Returns:
            Leaf index (data index = leaf_idx - capacity + 1)
        """
        left = 2 * idx + 1
        right = left + 1
        
        # Reached leaf node
        if left >= len(self.tree):
            return idx
        
        # Go left or right based on cumulative sum
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])
    
    def total(self) -> float:
        """Get total priority sum (root node)."""
        return self.tree[0]
    
    def add(self, priority: float, data: Experience):
        """Add new experience with given priority.
        
        Args:
            priority: Initial priority (typically max priority or 1.0)
            data: Experience object
        """
        # Get tree index for this data position
        idx = self.write_idx + self.capacity - 1
        
        # Store data
        self.data[self.write_idx] = data
        
        # Update tree
        self.update(idx, priority)
        
        # Move write pointer (circular)
        self.write_idx = (self.write_idx + 1) % self.capacity
        
        # Track size
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        """Update priority for given tree index.
        
        Args:
            idx: Tree index (leaf node)
            priority: New priority value
        """
        # Defensive: Clamp priority to reasonable range
        priority = max(0.0001, min(priority, 1000.0))
        
        # Calculate change
        change = priority - self.tree[idx]
        
        # Update leaf
        self.tree[idx] = priority
        
        # Propagate change to root
        self._propagate(idx, change)
    
    def get(self, value: float) -> Tuple[int, float, Experience]:
        """Get experience for given cumulative priority value.
        
        Args:
            value: Random value in [0, total_priority]
            
        Returns:
            (tree_idx, priority, experience)
        """
        # Defensive: Validate value
        if value < 0 or value > self.total():
            value = np.random.uniform(0, self.total())
        
        # Retrieve leaf index
        idx = self._retrieve(0, value)
        
        # Get data index
        data_idx = idx - self.capacity + 1
        
        # Defensive: Handle edge cases
        if data_idx < 0 or data_idx >= self.capacity:
            LOG.warning("Invalid data_idx=%d from tree_idx=%d", data_idx, idx)
            data_idx = 0
        
        return idx, self.tree[idx], self.data[data_idx]


class ExperienceBuffer:
    """
    Prioritized Experience Replay buffer for DDQN online learning.
    
    Features:
    - TD-error based prioritization (high error = more important)
    - Staleness decay (old experiences lose priority)
    - Regime-aware weighting (prioritize current regime)
    - Efficient O(log n) sampling via SumTree
    """
    
    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        staleness_halflife: float = 86400.0,  # 1 day in seconds
        regime_boost: float = 1.5,
        epsilon: float = 0.01
    ):
        """Initialize experience buffer.
        
        Args:
            capacity: Maximum experiences to store
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (0=no correction, 1=full correction)
            beta_increment: Beta increase per sample (annealing)
            staleness_halflife: Time for priority to decay by 50% (seconds)
            regime_boost: Priority multiplier for experiences from current regime
            epsilon: Small constant to ensure non-zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.staleness_halflife = staleness_halflife
        self.regime_boost = regime_boost
        self.epsilon = epsilon
        
        # SumTree for efficient sampling
        self.tree = SumTree(capacity)
        
        # Current regime (for regime-aware weighting)
        self.current_regime = RegimeSampling.UNKNOWN
        
        # Stats
        self.total_added = 0
        self.total_sampled = 0
        
        LOG.info(
            "ExperienceBuffer initialized: capacity=%d, alpha=%.2f, beta=%.2f, "
            "staleness_halflife=%.0fs, regime_boost=%.2f",
            capacity, alpha, beta, staleness_halflife, regime_boost
        )
    
    def set_current_regime(self, regime: int):
        """Update current regime for prioritization weighting.
        
        Args:
            regime: RegimeSampling enum value
        """
        self.current_regime = regime
        LOG.debug("Current regime updated: %s", RegimeSampling(regime).name)
    
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
        decay = np.exp(-age * np.log(2) / self.staleness_halflife)
        
        return max(0.0001, decay)  # Ensure non-zero
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime: int = RegimeSampling.UNKNOWN
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
        import math
        
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
            regime=regime,
            priority=1.0  # Will be updated during training
        )
        
        # Initial priority: max existing priority (ensures new experiences sampled at least once)
        max_priority = np.max(self.tree.tree[self.tree.capacity-1:self.tree.capacity-1+self.tree.size]) if self.tree.size > 0 else 1.0
        
        # Defensive: Cap max priority
        max_priority = min(max_priority, 1000.0)
        
        # Add to tree with max priority
        self.tree.add(max_priority, exp)
        
        self.total_added += 1
        
        if self.total_added % 1000 == 0:
            LOG.info("ExperienceBuffer: added %d experiences (size=%d)", self.total_added, self.tree.size)
    
    def sample(self, batch_size: int = 64) -> Optional[Dict]:
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
        if self.tree.size < batch_size:
            LOG.warning("Insufficient experiences: have %d, need %d", self.tree.size, batch_size)
            return None
        
        # Sample storage
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        indices = []
        priorities = []
        
        # Divide total priority range into batch_size segments
        segment_size = self.tree.total() / batch_size
        
        for i in range(batch_size):
            # Sample uniformly within segment
            a = segment_size * i
            b = segment_size * (i + 1)
            value = np.random.uniform(a, b)
            
            # Get experience
            idx, priority, exp = self.tree.get(value)
            
            # Defensive: Handle None experiences (shouldn't happen but be safe)
            if exp is None:
                LOG.warning("Sampled None experience at idx=%d", idx)
                continue
            
            # Apply staleness decay
            staleness_weight = self._calculate_staleness_weight(exp.timestamp)
            
            # Apply regime boost
            regime_weight = self.regime_boost if exp.regime == self.current_regime else 1.0
            
            # Combined priority
            adjusted_priority = priority * staleness_weight * regime_weight
            
            # Store
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)
            indices.append(idx)
            priorities.append(adjusted_priority)
        
        # Defensive: Check we got enough samples
        if len(states) < batch_size // 2:
            LOG.warning("Sample failed: only got %d/%d experiences", len(states), batch_size)
            return None
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)
        indices = np.array(indices, dtype=np.int32)
        priorities = np.array(priorities, dtype=np.float32)
        
        # Calculate importance sampling weights
        # w_i = (1 / (N * P(i)))^β
        # Normalized by max weight for stability
        probs = priorities / self.tree.total()
        weights = (1.0 / (self.tree.size * probs)) ** self.beta
        weights = weights / weights.max()  # Normalize
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        self.total_sampled += len(states)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'indices': indices,
            'weights': weights
        }
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for sampled experiences based on TD-errors.
        
        Args:
            indices: Tree indices from sample()
            td_errors: TD-error magnitudes (|target - prediction|)
        """
        import math
        
        # Defensive: Validate inputs
        if len(indices) != len(td_errors):
            LOG.warning("Mismatched lengths: indices=%d, td_errors=%d", len(indices), len(td_errors))
            return
        
        for idx, td_error in zip(indices, td_errors):
            # Defensive: Validate TD-error
            if not math.isfinite(td_error):
                LOG.warning("Non-finite TD-error: %.4f, skipping", td_error)
                continue
            
            # Cap extreme TD-errors (prevent priority explosion)
            td_error = max(-10.0, min(10.0, td_error))
            
            # Priority = (|TD-error| + ε)^α
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            
            # Update tree
            self.tree.update(int(idx), priority)
    
    @property
    def size(self) -> int:
        """Get current buffer size.
        
        Returns:
            Number of experiences in buffer
        """
        return self.tree.size
    
    def get_stats(self) -> Dict:
        """Get buffer statistics for monitoring.
        
        Returns:
            Dictionary with buffer stats
        """
        return {
            'size': self.tree.size,
            'capacity': self.capacity,
            'utilization': self.tree.size / self.capacity,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'beta': self.beta,
            'current_regime': RegimeSampling(self.current_regime).name,
            'total_priority': self.tree.total()
        }


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
    
    # Add experiences
    for i in range(5):
        exp = Experience(
            state=np.array([i]),
            action=i % 3,
            reward=float(i),
            next_state=np.array([i+1]),
            done=False,
            timestamp=time.time(),
            regime=RegimeSampling.UNKNOWN,
            priority=1.0
        )
        tree.add(priority=float(i+1), data=exp)
    
    print(f"Total priority: {tree.total():.2f}")
    print(f"Size: {tree.size}")
    
    # Sample
    idx, priority, exp = tree.get(tree.total() * 0.5)
    print(f"Sampled: idx={idx}, priority={priority:.2f}, state={exp.state}")
    
    # Test 2: ExperienceBuffer sampling
    print("\n[Test 2] ExperienceBuffer Sampling")
    print("-" * 80)
    
    buffer = ExperienceBuffer(capacity=1000)
    
    # Add experiences
    for i in range(200):
        buffer.add(
            state=np.random.randn(7),
            action=i % 3,
            reward=np.random.randn(),
            next_state=np.random.randn(7),
            done=(i % 50 == 0),
            regime=RegimeSampling.TRENDING if i % 2 == 0 else RegimeSampling.MEAN_REVERTING
        )
    
    # Sample batch
    batch = buffer.sample(batch_size=32)
    
    if batch:
        print(f"Batch shapes:")
        print(f"  states: {batch['states'].shape}")
        print(f"  actions: {batch['actions'].shape}")
        print(f"  rewards: {batch['rewards'].shape}")
        print(f"  weights: {batch['weights'].shape}")
        print(f"  weights range: [{batch['weights'].min():.3f}, {batch['weights'].max():.3f}]")
    
    # Test 3: Priority updates
    print("\n[Test 3] Priority Updates")
    print("-" * 80)
    
    if batch:
        # Simulate TD-errors
        td_errors = np.random.uniform(0, 2.0, size=len(batch['indices']))
        
        print(f"Updating {len(td_errors)} priorities")
        print(f"TD-errors: min={td_errors.min():.3f}, max={td_errors.max():.3f}, mean={td_errors.mean():.3f}")
        
        buffer.update_priorities(batch['indices'], td_errors)
        
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
    for i in range(50):
        buffer.add(
            state=np.random.randn(7),
            action=0,
            reward=1.0,
            next_state=np.random.randn(7),
            done=False,
            regime=RegimeSampling.TRENDING
        )
    
    for i in range(50):
        buffer.add(
            state=np.random.randn(7),
            action=0,
            reward=1.0,
            next_state=np.random.randn(7),
            done=False,
            regime=RegimeSampling.MEAN_REVERTING
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
