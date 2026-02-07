import pytest

"""Tests for SumTree and PrioritizedReplayBuffer."""

import numpy as np

from src.utils.sum_tree import SumTree, PrioritizedReplayBuffer


class TestSumTree:
    def test_init(self):
        tree = SumTree(capacity=8)
        assert tree.capacity == 8
        assert tree.n_entries == 0
        assert tree.total() == pytest.approx(0.0)

    def test_add_single(self):
        tree = SumTree(capacity=4)
        tree.add(1.0)
        assert tree.total() == pytest.approx(1.0)
        assert tree.n_entries == 1

    def test_add_multiple(self):
        tree = SumTree(capacity=4)
        tree.add(1.0)
        tree.add(2.0)
        tree.add(3.0)
        assert tree.total() == pytest.approx(6.0)
        assert tree.n_entries == 3

    def test_add_wraps_around(self):
        tree = SumTree(capacity=2)
        tree.add(1.0)
        tree.add(2.0)
        assert tree.n_entries == 2
        tree.add(5.0)  # Overwrites first entry
        assert tree.n_entries == 2
        assert tree.total() == pytest.approx(7.0)  # 2.0 + 5.0

    def test_update_propagates(self):
        tree = SumTree(capacity=4)
        tree.add(1.0)
        tree.add(2.0)
        assert tree.total() == pytest.approx(3.0)
        # Update first leaf (capacity-1 = 3 for capacity 4)
        tree.update(3, 5.0)
        assert tree.total() == pytest.approx(7.0)

    def test_sample_returns_valid_index(self):
        tree = SumTree(capacity=4)
        tree.add(1.0)
        tree.add(2.0)
        tree.add(3.0)
        idx = tree.sample(3.5)
        assert 0 <= idx < 4

    def test_sample_distribution(self):
        """Higher priority items should be sampled more often."""
        tree = SumTree(capacity=4, seed=42)
        tree.add(1.0)   # Low priority
        tree.add(100.0)  # High priority
        counts = {0: 0, 1: 0}
        for _ in range(1000):
            val = tree.rng.uniform(0, tree.total())
            idx = tree.sample(val)
            counts[idx] = counts.get(idx, 0) + 1
        # Item 1 (priority 100) should be sampled much more
        assert counts.get(1, 0) > counts.get(0, 0) * 5

    def test_get_priority(self):
        tree = SumTree(capacity=4)
        tree.add(3.14)
        assert tree.get_priority(0) == pytest.approx(3.14)

    def test_batch_update(self):
        tree = SumTree(capacity=4)
        tree.add(1.0)
        tree.add(1.0)
        tree.add(1.0)
        tree.batch_update(np.array([0, 1, 2]), np.array([10.0, 20.0, 30.0]))
        assert tree.total() == pytest.approx(60.0)

    def test_get_stats_empty(self):
        tree = SumTree(capacity=4)
        stats = tree.get_stats()
        assert stats["n_entries"] == 0
        assert stats["total"] == pytest.approx(0.0)

    def test_get_stats_with_data(self):
        tree = SumTree(capacity=4)
        tree.add(1.0)
        tree.add(2.0)
        tree.add(3.0)
        stats = tree.get_stats()
        assert stats["n_entries"] == 3
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(3.0)
        assert abs(stats["mean"] - 2.0) < 1e-10


class TestPrioritizedReplayBuffer:
    def test_init(self):
        buf = PrioritizedReplayBuffer(capacity=100, state_dim=4)
        assert len(buf) == 0
        assert buf.capacity == 100

    def test_add_experience(self):
        buf = PrioritizedReplayBuffer(capacity=100, state_dim=4)
        state = np.array([1.0, 2.0, 3.0, 4.0])
        next_state = np.array([2.0, 3.0, 4.0, 5.0])
        buf.add(state, action=1, reward=1.0, next_state=next_state, done=False)
        assert len(buf) == 1

    def test_add_multiple(self):
        buf = PrioritizedReplayBuffer(capacity=10, state_dim=2)
        for i in range(5):
            s = np.array([float(i), float(i + 1)])
            buf.add(s, action=0, reward=float(i), next_state=s, done=False)
        assert len(buf) == 5

    def test_add_with_regime_tag(self):
        buf = PrioritizedReplayBuffer(capacity=10, state_dim=2)
        s = np.array([1.0, 2.0])
        buf.add(s, action=0, reward=1.0, next_state=s, done=False, regime_tag=3)
        assert buf.regime_tags[0] == 3

    def test_beta_annealing(self):
        buf = PrioritizedReplayBuffer(
            capacity=10, state_dim=2, beta_start=0.4, beta_frames=100
        )
        assert buf._get_beta() == pytest.approx(0.4)
        buf.frame_count = 50
        beta_50 = buf._get_beta()
        assert 0.4 < beta_50 < 1.0
        buf.frame_count = 100
        assert buf._get_beta() == pytest.approx(1.0)

    def test_get_stats(self):
        buf = PrioritizedReplayBuffer(capacity=10, state_dim=2)
        s = np.array([1.0, 2.0])
        buf.add(s, action=0, reward=1.0, next_state=s, done=False)
        stats = buf.get_stats()
        assert stats["size"] == 1
        assert stats["capacity"] == 10
        assert "beta" in stats
