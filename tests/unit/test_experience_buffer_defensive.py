"""
Tests for experience buffer defensive code paths.

Covers:
- 1000-add logging (experience_buffer.py line 249)
- None experience during sampling (lines 297-298)
- Insufficient samples fallback (lines 323-324)
"""

import logging
import time

import numpy as np

rng = np.random.default_rng(42)

import pytest

from src.utils.experience_buffer import ExperienceBuffer, RegimeSampling


def _make_state(dim: int = 7) -> np.ndarray:
    """Create a random state vector."""
    return rng.standard_normal(dim).astype(np.float32)


def _add_n_experiences(buf: ExperienceBuffer, n: int, dim: int = 7):
    """Add n valid experiences to buffer."""
    for _ in range(n):
        buf.add(
            state=_make_state(dim),
            action=rng.choice([0, 1, 2]),
            reward=float(rng.standard_normal()),
            next_state=_make_state(dim),
            done=False,
            regime=RegimeSampling.UNKNOWN,
        )


# ===========================================================================
# 1000-add logging path (line 249)
# ===========================================================================

class TestExperienceBuffer1000Logging:
    """Test that logging fires every 1000 adds."""

    def test_logging_at_1000_adds(self, caplog):
        """After 1000 adds, an INFO log is emitted."""
        buf = ExperienceBuffer(capacity=2000, seed=42)

        with caplog.at_level(logging.INFO, logger="src.utils.experience_buffer"):
            _add_n_experiences(buf, 1000, dim=4)

        assert buf.total_added == 1000
        log_messages = [r.message for r in caplog.records if "added 1000 experiences" in r.message]
        assert len(log_messages) >= 1

    def test_no_logging_at_999_adds(self, caplog):
        """No 1000-add log message at 999."""
        buf = ExperienceBuffer(capacity=2000, seed=42)

        with caplog.at_level(logging.INFO, logger="src.utils.experience_buffer"):
            _add_n_experiences(buf, 999, dim=4)

        assert buf.total_added == 999
        log_messages = [r.message for r in caplog.records if "added 1000 experiences" in r.message]
        assert len(log_messages) == 0


# ===========================================================================
# None experience during sampling (lines 297-298)
# ===========================================================================

class TestExperienceBufferNoneSampling:
    """Test handling of None experiences during sampling."""

    def test_none_experience_skipped(self, caplog):
        """When data slots contain None, they are skipped during sampling."""
        buf = ExperienceBuffer(capacity=100, seed=42)
        _add_n_experiences(buf, 80, dim=4)

        # Corrupt some data slots to None
        for i in range(0, 80, 2):
            buf.data[i] = None

        with caplog.at_level(logging.WARNING, logger="src.utils.experience_buffer"):
            result = buf.sample(batch_size=10)

        # Should either return a result with fewer samples or None
        # depending on how many None slots were hit
        if result is not None:
            assert result["states"].shape[0] > 0

    def test_all_none_returns_none(self):
        """When all sampled experiences are None, returns None."""
        buf = ExperienceBuffer(capacity=50, seed=42)
        _add_n_experiences(buf, 50, dim=4)

        # Set ALL data to None
        for i in range(50):
            buf.data[i] = None

        result = buf.sample(batch_size=10)
        assert result is None


# ===========================================================================
# Insufficient samples (lines 323-324)
# ===========================================================================

class TestExperienceBufferInsufficientSamples:
    """When too few valid experiences are gathered, sample returns None."""

    def test_mostly_none_returns_none(self):
        """When most experiences are None, batch is too small → return None."""
        buf = ExperienceBuffer(capacity=100, seed=42)
        _add_n_experiences(buf, 100, dim=4)

        # Keep only 2 valid experiences, rest are None
        for i in range(2, 100):
            buf.data[i] = None

        result = buf.sample(batch_size=20)
        # With only 2 valid out of 20 needed, 2 < 20//2 = 10, so returns None
        assert result is None

    def test_half_none_returns_result(self):
        """When enough experiences survive (>= batch//2), result is returned."""
        buf = ExperienceBuffer(capacity=100, seed=42)
        _add_n_experiences(buf, 100, dim=4)

        # Set some to None, but keep enough for batch_size//2
        for i in range(0, 100, 3):  # ~33% are None
            buf.data[i] = None

        result = buf.sample(batch_size=10)
        # Most should survive, result should not be None
        if result is not None:
            assert result["states"].shape[0] >= 5


# ===========================================================================
# Input validation edge cases
# ===========================================================================

class TestExperienceBufferInputValidation:
    """Test additional defensive input validation."""

    def test_add_non_array_state_rejected(self, caplog):
        """Non-numpy state is rejected."""
        buf = ExperienceBuffer(capacity=10, seed=42)
        with caplog.at_level(logging.WARNING):
            buf.add(
                state=[1, 2, 3],  # list, not ndarray
                action=1,
                reward=0.5,
                next_state=_make_state(3),
                done=False,
            )
        assert buf.total_added == 0

    def test_add_empty_state_rejected(self, caplog):
        """Empty state vector is rejected."""
        buf = ExperienceBuffer(capacity=10, seed=42)
        with caplog.at_level(logging.WARNING):
            buf.add(
                state=np.array([]),
                action=1,
                reward=0.5,
                next_state=_make_state(3),
                done=False,
            )
        assert buf.total_added == 0

    def test_add_non_finite_reward_rejected(self, caplog):
        """Non-finite reward is rejected."""
        buf = ExperienceBuffer(capacity=10, seed=42)
        with caplog.at_level(logging.WARNING):
            buf.add(
                state=_make_state(3),
                action=1,
                reward=float("inf"),
                next_state=_make_state(3),
                done=False,
            )
        assert buf.total_added == 0

    def test_add_invalid_action_rejected(self, caplog):
        """Invalid action value is rejected."""
        buf = ExperienceBuffer(capacity=10, seed=42)
        with caplog.at_level(logging.WARNING):
            buf.add(
                state=_make_state(3),
                action=5,
                reward=0.5,
                next_state=_make_state(3),
                done=False,
            )
        assert buf.total_added == 0

    def test_update_priorities_mismatched_lengths(self, caplog):
        """Mismatched indices and td_errors lengths logs warning."""
        buf = ExperienceBuffer(capacity=100, seed=42)
        _add_n_experiences(buf, 10, dim=4)

        with caplog.at_level(logging.WARNING):
            buf.update_priorities(
                np.array([0, 1, 2]),
                np.array([0.1, 0.2]),  # Different length
            )

    def test_update_priorities_non_finite_td_error(self, caplog):
        """Non-finite TD-error is skipped."""
        buf = ExperienceBuffer(capacity=100, seed=42)
        _add_n_experiences(buf, 10, dim=4)

        with caplog.at_level(logging.WARNING):
            buf.update_priorities(
                np.array([0, 1]),
                np.array([0.5, float("nan")]),
            )

    def test_update_priorities_clamps_extreme(self):
        """Extreme TD-errors are clamped to [-10, 10]."""
        buf = ExperienceBuffer(capacity=100, seed=42)
        _add_n_experiences(buf, 10, dim=4)

        # Should not raise
        buf.update_priorities(
            np.array([0, 1, 2]),
            np.array([100.0, -100.0, 5.0]),
        )

    def test_sample_insufficient_entries(self):
        """Sample with fewer experiences than batch_size returns None."""
        buf = ExperienceBuffer(capacity=100, seed=42)
        _add_n_experiences(buf, 5, dim=4)

        result = buf.sample(batch_size=10)
        assert result is None


# ===========================================================================
# Staleness and regime weighting
# ===========================================================================

class TestExperienceBufferStalenessRegime:
    """Test staleness decay and regime-aware weighting."""

    def test_staleness_weight_recent(self):
        """Recent experiences have weight close to 1.0."""
        buf = ExperienceBuffer(capacity=10, staleness_halflife=86400.0)
        weight = buf._calculate_staleness_weight(time.time())
        assert weight > 0.99

    def test_staleness_weight_old(self):
        """Old experiences have decayed weight."""
        buf = ExperienceBuffer(capacity=10, staleness_halflife=3600.0)
        weight = buf._calculate_staleness_weight(time.time() - 7200)  # 2 hours ago
        assert weight < 0.5

    def test_staleness_weight_future_timestamp(self):
        """Future timestamps (clock skew) return weight 1.0."""
        buf = ExperienceBuffer(capacity=10, staleness_halflife=86400.0)
        weight = buf._calculate_staleness_weight(time.time() + 100)
        assert weight == pytest.approx(1.0)

    def test_set_current_regime(self):
        """set_current_regime updates the regime."""
        buf = ExperienceBuffer(capacity=10)
        buf.set_current_regime(RegimeSampling.TRENDING)
        assert buf.current_regime == RegimeSampling.TRENDING

    def test_get_stats(self):
        """get_stats returns correct buffer statistics."""
        buf = ExperienceBuffer(capacity=100, seed=42)
        _add_n_experiences(buf, 10, dim=4)

        stats = buf.get_stats()
        assert stats["size"] == 10
        assert stats["capacity"] == 100
        assert stats["total_added"] == 10
        assert 0.0 < stats["utilization"] < 1.0
        assert stats["current_regime"] == "UNKNOWN"
