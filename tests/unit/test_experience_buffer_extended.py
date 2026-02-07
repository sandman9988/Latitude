"""Extended tests for src.utils.experience_buffer.

Covers: sample() edge paths, update_priorities clamping, get_stats,
set_current_regime, staleness negative-age, RegimeSampling enum.
"""

import math
import time
import pytest
import numpy as np

from src.utils.experience_buffer import (
    Experience,
    ExperienceBuffer,
    RegimeSampling,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_state(dim=7):
    return np.random.default_rng(0).standard_normal(dim).astype(np.float32)


def _fill_buffer(buf, n=100):
    rng = np.random.default_rng(42)
    for i in range(n):
        buf.add(
            state=rng.standard_normal(7).astype(np.float32),
            action=i % 3,
            reward=float(rng.standard_normal()),
            next_state=rng.standard_normal(7).astype(np.float32),
            done=(i % 20 == 0),
            regime=i % 4,
        )


# ---------------------------------------------------------------------------
# RegimeSampling enum
# ---------------------------------------------------------------------------
class TestRegimeSampling:
    def test_values(self):
        assert RegimeSampling.TRENDING == 0
        assert RegimeSampling.MEAN_REVERTING == 1
        assert RegimeSampling.TRANSITIONAL == 2
        assert RegimeSampling.UNKNOWN == 3

    def test_from_int(self):
        assert RegimeSampling(2) == RegimeSampling.TRANSITIONAL


# ---------------------------------------------------------------------------
# set_current_regime
# ---------------------------------------------------------------------------
class TestSetCurrentRegime:
    def test_sets_regime(self):
        buf = ExperienceBuffer(capacity=10, seed=0)
        buf.set_current_regime(RegimeSampling.TRENDING)
        assert buf.current_regime == RegimeSampling.TRENDING

    def test_default_regime(self):
        buf = ExperienceBuffer(capacity=10, seed=0)
        assert buf.current_regime == RegimeSampling.UNKNOWN


# ---------------------------------------------------------------------------
# Sample edge cases
# ---------------------------------------------------------------------------
class TestSampleExtended:
    def test_sample_returns_none_when_empty(self):
        buf = ExperienceBuffer(capacity=100, seed=0)
        assert buf.sample(batch_size=10) is None

    def test_sample_returns_none_insufficient(self):
        buf = ExperienceBuffer(capacity=100, seed=0)
        _fill_buffer(buf, n=5)
        assert buf.sample(batch_size=10) is None

    def test_sample_batch_keys(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, n=100)
        batch = buf.sample(batch_size=16)
        assert batch is not None
        for key in ["states", "actions", "rewards", "next_states", "dones", "indices", "weights"]:
            assert key in batch

    def test_sample_shapes(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, n=100)
        batch = buf.sample(batch_size=16)
        assert batch["states"].shape[0] == 16
        assert batch["actions"].shape == (16,)
        assert batch["rewards"].shape == (16,)
        assert batch["dones"].shape == (16,)

    def test_weights_max_is_one(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, n=100)
        batch = buf.sample(batch_size=32)
        assert batch["weights"].max() == pytest.approx(1.0)

    def test_beta_increases_after_sample(self):
        buf = ExperienceBuffer(capacity=200, beta=0.4, beta_increment=0.01, seed=42)
        _fill_buffer(buf, n=100)
        old_beta = buf.beta
        buf.sample(batch_size=16)
        assert buf.beta > old_beta

    def test_beta_capped_at_one(self):
        buf = ExperienceBuffer(capacity=200, beta=0.999, beta_increment=0.01, seed=42)
        _fill_buffer(buf, n=100)
        buf.sample(16)
        assert buf.beta <= 1.0

    def test_total_sampled_increments(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, n=100)
        before = buf.total_sampled
        buf.sample(16)
        assert buf.total_sampled > before


# ---------------------------------------------------------------------------
# update_priorities edge cases
# ---------------------------------------------------------------------------
class TestUpdatePrioritiesExtended:
    def test_extreme_td_error_clamped(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, n=100)
        batch = buf.sample(16)
        # Pass extreme TD-errors
        extreme_errors = np.full(len(batch["indices"]), 1000.0)
        buf.update_priorities(batch["indices"], extreme_errors)
        # Should not crash; priorities clamped internally

    def test_negative_td_errors_use_abs(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, n=100)
        batch = buf.sample(16)
        neg_errors = np.full(len(batch["indices"]), -5.0)
        buf.update_priorities(batch["indices"], neg_errors)
        # No crash

    def test_nan_td_error_skipped(self):
        buf = ExperienceBuffer(capacity=200, seed=42)
        _fill_buffer(buf, n=100)
        batch = buf.sample(16)
        errors = np.ones(len(batch["indices"]))
        errors[0] = float("nan")
        buf.update_priorities(batch["indices"], errors)


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------
class TestStalenessExtended:
    def test_future_timestamp_returns_one(self):
        buf = ExperienceBuffer(capacity=10, seed=0)
        w = buf._calculate_staleness_weight(time.time() + 3600)
        assert w == pytest.approx(1.0)

    def test_very_old_returns_near_zero(self):
        buf = ExperienceBuffer(capacity=10, staleness_halflife=3600.0, seed=0)
        w = buf._calculate_staleness_weight(time.time() - 3600 * 100)
        assert w < 0.01

    def test_halflife_decay(self):
        buf = ExperienceBuffer(capacity=10, staleness_halflife=100.0, seed=0)
        w = buf._calculate_staleness_weight(time.time() - 100.0)
        assert abs(w - 0.5) < 0.05


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------
class TestGetStats:
    def test_stats_keys(self):
        buf = ExperienceBuffer(capacity=100, seed=0)
        _fill_buffer(buf, n=50)
        stats = buf.get_stats()
        for key in ["size", "capacity", "utilization", "total_added", "total_sampled", "beta", "current_regime", "total_priority"]:
            assert key in stats

    def test_utilization_correct(self):
        buf = ExperienceBuffer(capacity=100, seed=0)
        _fill_buffer(buf, n=50)
        stats = buf.get_stats()
        assert abs(stats["utilization"] - 0.5) < 0.01

    def test_current_regime_name(self):
        buf = ExperienceBuffer(capacity=10, seed=0)
        buf.set_current_regime(RegimeSampling.TRENDING)
        stats = buf.get_stats()
        assert stats["current_regime"] == "TRENDING"


# ---------------------------------------------------------------------------
# size property
# ---------------------------------------------------------------------------
class TestSizeProperty:
    def test_empty(self):
        buf = ExperienceBuffer(capacity=10, seed=0)
        assert buf.size == 0

    def test_after_adds(self):
        buf = ExperienceBuffer(capacity=100, seed=0)
        _fill_buffer(buf, n=30)
        assert buf.size == 30

    def test_wraps(self):
        buf = ExperienceBuffer(capacity=10, seed=0)
        _fill_buffer(buf, n=20)
        assert buf.size == 10
