"""
Sum Tree for Prioritized Experience Replay

Binary tree where each node stores the sum of its children's priorities.
Enables O(log n) sampling and updates for experience replay.

References:
- Schaul et al. (2016) "Prioritized Experience Replay"
- OpenAI Baselines implementation
"""

import numpy as np


class SumTree:
    """
    Binary sum tree for O(log n) priority-based sampling.

    Structure:
        - Leaf nodes: Store priorities for each experience
        - Internal nodes: Sum of children
        - Root: Total sum of all priorities

    Operations:
        - update(idx, priority): O(log n) - Update leaf and propagate
        - sample(value): O(log n) - Sample by cumulative priority
        - get_priority(idx): O(1) - Retrieve leaf priority
    """

    def __init__(self, capacity: int, seed: int | None = None):
        """
        Initialize sum tree with fixed capacity.

        Args:
            capacity: Maximum number of leaf nodes (experiences)
            seed: Random seed for reproducibility (default: None for non-deterministic)
        """
        self.capacity = capacity
        self.write_index = 0
        self.n_entries = 0
        self.rng = np.random.default_rng(seed)

        # Tree storage: [internal_nodes | leaf_nodes]
        # Internal nodes: capacity - 1
        # Leaf nodes: capacity
        # Total: 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def _propagate(self, idx: int, change: float):
        """
        Propagate priority change up the tree.

        Args:
            idx: Tree index (leaf or internal node)
            change: Delta to add to ancestors
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """
        Traverse tree to find leaf with cumulative priority >= value.

        Args:
            idx: Current node index (start at root=0)
            value: Target cumulative priority

        Returns:
            Leaf index in tree array
        """
        left = 2 * idx + 1
        right = left + 1

        # Reached leaf node
        if left >= len(self.tree):
            return idx

        # Traverse left if value <= left sum
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            # Traverse right, subtract left sum from value
            return self._retrieve(right, value - self.tree[left])

    def total(self) -> float:
        """Get total sum of all priorities (root value)."""
        return self.tree[0]

    def add(self, priority: float):
        """
        Add new priority at current write position.

        Args:
            priority: Priority value (typically |TD error|^alpha)
        """
        idx = self.write_index + self.capacity - 1
        self.update(idx, priority)

        self.write_index = (self.write_index + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """
        Update priority at tree index and propagate change.

        Args:
            idx: Tree index (in range [capacity-1, 2*capacity-2])
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, idx: int) -> float:
        """
        Get priority at tree index.

        Args:
            idx: Tree index

        Returns:
            Priority value
        """
        return self.tree[idx]

    def sample(self, value: float) -> int:
        """
        Sample leaf index by cumulative priority.

        Args:
            value: Random value in [0, total()]

        Returns:
            Data index (in range [0, capacity-1])
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return data_idx

    def get_priority(self, data_idx: int) -> float:
        """
        Get priority for data index.

        Args:
            data_idx: Data index in [0, capacity-1]

        Returns:
            Priority value
        """
        tree_idx = data_idx + self.capacity - 1
        return self.tree[tree_idx]

    def batch_update(self, data_indices: np.ndarray, priorities: np.ndarray):
        """
        Batch update priorities (more efficient than individual updates).

        Args:
            data_indices: Array of data indices
            priorities: Array of new priority values
        """
        for data_idx, priority in zip(data_indices, priorities, strict=False):
            tree_idx = data_idx + self.capacity - 1
            self.update(tree_idx, priority)

    def get_stats(self) -> dict:
        """
        Get statistics about current tree state.

        Returns:
            Dictionary with min/max/mean priorities, total sum
        """
        if self.n_entries == 0:
            return {"total": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "n_entries": 0}

        # Get all leaf priorities
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        priorities = self.tree[leaf_start:leaf_end]

        return {
            "total": self.total(),
            "min": np.min(priorities),
            "max": np.max(priorities),
            "mean": np.mean(priorities),
            "n_entries": self.n_entries,
        }


class PrioritizedReplayBuffer:
    """
    Experience replay buffer with prioritized sampling.

    Key features:
    - Sum tree for O(log n) sampling by TD error
    - Importance sampling weights with beta annealing
    - Stratified batch sampling
    - Priority statistics tracking
    """

    def __init__(  # noqa: PLR0913
        self,
        capacity: int,
        state_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            state_dim: Dimension of state vector
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta_start: Initial importance sampling exponent
            beta_frames: Frames to anneal beta from beta_start to 1.0
            epsilon: Small constant added to priorities
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        # Sum tree for priorities
        self.tree = SumTree(capacity)

        # Experience storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        # Metadata
        self.regime_tags = np.zeros(capacity, dtype=np.int32)  # For regime-aware sampling
        self.timestamps = np.zeros(capacity, dtype=np.float64)

        # Tracking
        self.position = 0
        self.size = 0
        self.frame_count = 0
        self.max_priority = 1.0  # Initialize with 1.0 for new experiences

    def _get_beta(self) -> float:
        """
        Get current beta value (annealed from beta_start to 1.0).

        Returns:
            Current beta value
        """
        progress = min(self.frame_count / self.beta_frames, 1.0)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def add(  # noqa: PLR0913
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        regime_tag: int = 0,
        timestamp: float = 0.0,
    ):
        """
        Add experience with maximum priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
            regime_tag: Optional regime identifier
            timestamp: Optional timestamp
        """
        # Store experience
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.regime_tags[self.position] = regime_tag
        self.timestamps[self.position] = timestamp

        # Add to tree with max priority
        priority = self.max_priority**self.alpha
        self.tree.add(priority)

        # Update tracking
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """
        Sample batch using prioritized sampling with importance weights.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones,
                     indices, weights, regime_tags)
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, self.size)

        # Stratified sampling (divide priority range into segments)
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)

        priority_segment = self.tree.total() / batch_size

        for i in range(batch_size):
            # Sample uniformly within segment
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = self.rng.uniform(a, b)

            # Get data index from tree
            idx = self.tree.sample(value)
            indices[i] = idx
            priorities[i] = self.tree.get_priority(idx)

        # Calculate importance sampling weights
        beta = self._get_beta()

        # P(i) = p_i^alpha / sum(p_j^alpha)
        sampling_probabilities = priorities / self.tree.total()

        # IS weight = (N * P(i))^(-beta) / max_weight
        weights = (self.size * sampling_probabilities) ** (-beta)
        weights = weights / np.max(weights)  # Normalize by max weight

        # Gather experiences
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        regime_tags = self.regime_tags[indices]

        self.frame_count += 1

        return (states, actions, rewards, next_states, dones, indices, weights, regime_tags)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.

        Args:
            indices: Data indices to update
            td_errors: TD error values (used to compute priorities)
        """
        for idx, td_error in zip(indices, td_errors, strict=False):
            # Priority = |TD error|^alpha + epsilon
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx + self.capacity - 1, priority)

            # Track max priority for new experiences
            self.max_priority = max(self.max_priority, abs(td_error) + self.epsilon)

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def get_stats(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary with size, beta, priority stats
        """
        tree_stats = self.tree.get_stats()

        return {
            "size": self.size,
            "capacity": self.capacity,
            "beta": self._get_beta(),
            "max_priority": self.max_priority,
            "frame_count": self.frame_count,
            **tree_stats,
        }
