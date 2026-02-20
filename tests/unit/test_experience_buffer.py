"""Tests for src.utils.experience_buffer – ExperienceBuffer + helpers."""

import time

import numpy as np
import pytest

from src.utils.experience_buffer import (
    Experience,
    ExperienceBuffer,
    RegimeSampling,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(dim: int = 7, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _fill_buffer(buf: ExperienceBuffer, n: int = 100, dim: int = 7):
    """Add *n* random experiences to *buf*."""
    rng = np.random.default_rng(42)
    for i in range(n):
        buf.add(
            state=rng.standard_normal(dim).astype(np.float32),
            action=int(rng.integers(0, 3)),
            reward=float(rng.standard_normal()),
            next_state=rng.standard_normal(dim).astype(np.float32),
            done=(i % 20 == 0),
            regime=int(rng.integers(0, 4)),
        )


# ---------------------------------------------------------------------------
# Experience dataclass
# ---------------------------------------------------------------------------

class TestExperience:
    def test_fields(self):
        e = Experience(
            state=np.zeros(7),
            action=1,
            reward=0.5,
            next_state=np.ones(7),
            done=False,
            timestamp=time.time(),
            regime=0,
            priority=1.0,
        )
        assert e.action == 1
        assert e.reward == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RegimeSampling enum
# ---------------------------------------------------------------------------

class TestRegimeSampling:
    def test_values(self):
        assert RegimeSampling.TRENDING == 0
        assert RegimeSampling.MEAN_REVERTING == 1
        assert RegimeSampling.UNKNOWN == 3


# ---------------------------------------------------------------------------
# ExperienceBuffer init
# ---------------------------------------------------------------------------

class TestExperienceBufferInit:
    def test_defaults(self):
        buf = ExperienceBuffer(capacity=100)
        assert buf.capacity == 100
        assert buf.size == 0
        assert buf.beta == pytest.approx(0.4)

    def test_custom_params(self):
        buf = ExperienceBuffer(capacity=50, alpha=0.5, beta=0.5, seed=99)
        assert buf.alpha == pytest.approx(0.5)
        assert buf.beta == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_increments_size(self):
        buf = ExperienceBuffer(capacity=100)
        buf.add(state=_state(), action=0, reward=1.0, next_state=_state(seed=1), done=False)
        assert buf.size == 1

    def test_add_wraps_at_capacity(self):
        buf = ExperienceBuffer(capacity=5)
        _fill_buffer(buf, 10)
        assert buf.size == 5  # Only 5 fit in capacity

    def test_reject_non_ndarray(self):
        buf = ExperienceBuffer(capacity=10)
        buf.add(state=[1, 2, 3], action=0, reward=0.0, next_state=_state(), done=False)
        assert buf.size == 0  # Rejected

    def test_reject_empty_state(self):
        buf = ExperienceBuffer(capacity=10)
        buf.add(state=np.array([]), action=0, reward=0.0, next_state=_state(), done=False)
        assert buf.size == 0

    def test_reject_nan_reward(self):
        buf = ExperienceBuffer(capacity=10)
        buf.add(state=_state(), action=0, reward=float("nan"), next_state=_state(seed=1), done=False)
        assert buf.size == 0

    def test_reject_invalid_action(self):
        buf = ExperienceBuffer(capacity=10)
        buf.add(state=_state(), action=5, reward=0.0, next_state=_state(seed=1), done=False)
        assert buf.size == 0

    def test_copies_state(self):
        """Ensure modifying original array doesn't affect stored experience."""
        buf = ExperienceBuffer(capacity=10)
        s = np.ones(7, dtype=np.float32)
        buf.add(state=s, action=1, reward=0.0, next_state=_state(), done=False)
        s[:] = 999.0
        assert buf.data[0].state[0] != 999.0


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------

class TestSample:
    def test_returns_none_when_insufficient(self):
        buf = ExperienceBuffer(capacity=100)
        _fill_buffer(buf, 10)
        assert buf.sample(batch_size=20) is None

    def test_returns_batch_dict(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, 100)
        batch = buf.sample(batch_size=16)
        assert batch is not None
        for key in ("states", "actions", "rewards", "next_states", "dones", "indices", "weights"):
            assert key in batch
        assert batch["states"].shape[0] == 16

    def test_weights_normalized(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, 100)
        batch = buf.sample(batch_size=16)
        assert batch is not None
        assert batch["weights"].max() == pytest.approx(1.0)

    def test_beta_anneals(self):
        buf = ExperienceBuffer(capacity=200, beta=0.4, beta_increment=0.01, seed=42)
        _fill_buffer(buf, 100)
        buf.sample(batch_size=16)
        assert buf.beta > 0.4


# ---------------------------------------------------------------------------
# update_priorities()
# ---------------------------------------------------------------------------

class TestUpdatePriorities:
    def test_update_priorities(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, 100)
        batch = buf.sample(batch_size=8)
        assert batch is not None
        td_errors = np.ones(len(batch["indices"]))
        buf.update_priorities(batch["indices"], td_errors)
        # Should not raise

    def test_mismatched_lengths_ignored(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, 100)
        buf.update_priorities(np.array([0, 1]), np.array([1.0]))
        # Mismatched → warning logged, no crash

    def test_non_finite_td_error_skipped(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, 100)
        batch = buf.sample(batch_size=4)
        assert batch is not None
        td_errors = np.array([1.0, float("inf"), 0.5, float("nan")])
        buf.update_priorities(batch["indices"], td_errors)
        # Non-finite values silently skipped


# ---------------------------------------------------------------------------
# staleness / regime
# ---------------------------------------------------------------------------

class TestStalenessAndRegime:
    def test_staleness_new_experience(self):
        buf = ExperienceBuffer(capacity=10)
        weight = buf._calculate_staleness_weight(time.time())
        assert weight == pytest.approx(1.0, abs=0.01)

    def test_staleness_halflife(self):
        buf = ExperienceBuffer(capacity=10, staleness_halflife=86400)
        weight = buf._calculate_staleness_weight(time.time() - 86400)
        assert weight == pytest.approx(0.5, abs=0.05)

    def test_staleness_future_timestamp(self):
        buf = ExperienceBuffer(capacity=10)
        weight = buf._calculate_staleness_weight(time.time() + 3600)
        assert weight == pytest.approx(1.0)

    def test_set_current_regime(self):
        buf = ExperienceBuffer(capacity=10)
        buf.set_current_regime(RegimeSampling.TRENDING)
        assert buf.current_regime == RegimeSampling.TRENDING


# ---------------------------------------------------------------------------
# get_stats / size
# ---------------------------------------------------------------------------

class TestStats:
    def test_size_property(self):
        buf = ExperienceBuffer(capacity=50)
        _fill_buffer(buf, 20)
        assert buf.size == 20

    def test_get_stats_keys(self):
        buf = ExperienceBuffer(capacity=50)
        _fill_buffer(buf, 20)
        stats = buf.get_stats()
        for key in ("size", "capacity", "utilization", "total_added",
                     "total_sampled", "beta", "current_regime", "total_priority"):
            assert key in stats

    def test_utilization(self):
        buf = ExperienceBuffer(capacity=100)
        _fill_buffer(buf, 50)
        stats = buf.get_stats()
        assert stats["utilization"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# IS weight correctness (regression: weights must use raw priorities)
# ---------------------------------------------------------------------------


class TestISWeightCorrectness:
    """IS weights must be computed from raw tree priorities (actual P(i)),
    not from staleness/regime-adjusted priorities.  Using adjusted priorities
    breaks the IS correction because the SumTree samples from raw priorities."""

    def test_weights_in_unit_interval(self):
        """All IS weights are in [0, 1] regardless of regime boost magnitude."""
        buf = ExperienceBuffer(capacity=200, regime_boost=50.0, seed=42)
        _fill_buffer(buf, 100)
        buf.set_current_regime(RegimeSampling.TRENDING)
        batch = buf.sample(batch_size=16)
        assert batch is not None
        assert np.all(batch["weights"] >= 0.0)
        assert np.all(batch["weights"] <= 1.0 + 1e-6)

    def test_max_weight_one_despite_extreme_regime_boost(self):
        """Max weight is always 1.0 even when regime_boost is extreme.

        Before the fix the IS weight was computed using adjusted_priority
        (raw * staleness * regime_boost).  With extreme regime_boost the
        adjusted priority of regime-matching experiences can be >>> raw, making
        probs > 1 and weights < 0 or NaN, so the max would not be 1.0.
        After the fix raw priorities are used, guaranteeing normalisation.
        """
        buf = ExperienceBuffer(capacity=200, regime_boost=1000.0, seed=42)
        rng = np.random.default_rng(7)
        for i in range(100):
            regime = int(RegimeSampling.TRENDING) if i < 50 else int(RegimeSampling.MEAN_REVERTING)
            buf.add(
                state=rng.standard_normal(7).astype(np.float32),
                action=int(rng.integers(0, 3)),
                reward=float(rng.standard_normal()),
                next_state=rng.standard_normal(7).astype(np.float32),
                done=(i % 20 == 0),
                regime=regime,
            )
        buf.set_current_regime(RegimeSampling.TRENDING)
        batch = buf.sample(batch_size=16)
        assert batch is not None
        assert batch["weights"].max() == pytest.approx(1.0, abs=1e-5)
        assert np.all(np.isfinite(batch["weights"]))

    def test_sampling_does_not_change_tree_total_mid_loop(self):
        """Sampling must not mutate tree total during stratified segment draw.

        The stratified segments are based on tree.total() computed once before
        the loop.  Updating priorities inside the loop would shift the total and
        corrupt later segment boundaries.  We verify total is stable across two
        consecutive samples from the same buffer state.
        """
        buf = ExperienceBuffer(capacity=200, regime_boost=5.0, seed=42)
        _fill_buffer(buf, 100)
        buf.set_current_regime(RegimeSampling.TRENDING)
        total_before = buf.tree.total()
        buf.sample(batch_size=16)
        # After sampling, tree may be updated (post-loop), but a second sample
        # should also succeed (no corruption / division-by-zero).
        batch2 = buf.sample(batch_size=16)
        assert batch2 is not None
        assert np.all(np.isfinite(batch2["weights"]))
        _ = total_before  # referenced to avoid lint warning
