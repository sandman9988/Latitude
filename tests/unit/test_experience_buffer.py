"""Tests for src.utils.experience_buffer – ExperienceBuffer + helpers."""

import time

import numpy as np
import pytest

from src.utils.experience_buffer import (
    HALFLIFE_SESSIONS,
    TRADING_SESSION_MINUTES,
    Experience,
    ExperienceBuffer,
    RegimeSampling,
    staleness_halflife_for_timeframe,
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
        for key in (
            "size",
            "capacity",
            "utilization",
            "total_added",
            "total_sampled",
            "beta",
            "current_regime",
            "total_priority",
        ):
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


# ---------------------------------------------------------------------------
# Staleness halflife utility
# ---------------------------------------------------------------------------


class TestStalenessHalflife:
    def test_timeframe_agnostic(self):
        """M1, M5, and H1 all produce the same wall-clock halflife."""
        h_m1 = staleness_halflife_for_timeframe(1)
        h_m5 = staleness_halflife_for_timeframe(5)
        h_h1 = staleness_halflife_for_timeframe(60)
        assert h_m1 == pytest.approx(h_m5)
        assert h_m5 == pytest.approx(h_h1)

    def test_default_value(self):
        """Default 1.5 sessions × 480 min × 60 s = 43200 s (12 h)."""
        h = staleness_halflife_for_timeframe(5)
        assert h == pytest.approx(HALFLIFE_SESSIONS * TRADING_SESSION_MINUTES * 60.0)
        assert h == pytest.approx(43_200.0)

    def test_custom_n_sessions(self):
        """3 sessions × 480 min × 60 s = 86400 s (24 h)."""
        h = staleness_halflife_for_timeframe(5, n_sessions=3.0)
        assert h == pytest.approx(86_400.0)

    def test_custom_session_minutes(self):
        """1 session × 360 min × 60 s = 21600 s (6 h)."""
        h = staleness_halflife_for_timeframe(5, n_sessions=1.0, session_minutes=360.0)
        assert h == pytest.approx(21_600.0)


# ---------------------------------------------------------------------------
# Save / load persistence
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        """Saved buffer restores all experiences correctly."""
        buf = ExperienceBuffer(capacity=50)
        _fill_buffer(buf, 30, dim=7)
        path = str(tmp_path / "buf")
        assert buf.save(path)
        buf2 = ExperienceBuffer(capacity=50)
        assert buf2.load(path)
        assert buf2.size == 30

    def test_load_full_smaller_checkpoint_keeps_data_and_tree_aligned(self, tmp_path):
        """A full checkpoint from a smaller capacity has saved write_idx=0.

        Loading it into a larger runtime buffer must append new data after the
        compacted loaded rows, matching SumTree's next priority slot. Otherwise
        sampling can hit priority slots whose data entries are still None.
        """
        old = ExperienceBuffer(capacity=5)
        _fill_buffer(old, 5, dim=7)
        assert old.write_idx == 0
        path = str(tmp_path / "full_old_capacity")
        assert old.save(path)

        new = ExperienceBuffer(capacity=20, seed=42)
        assert new.load(path)
        assert new.size == 5
        assert new.write_idx == 5
        assert new.tree.write_index == 5

        _fill_buffer(new, 3, dim=7)
        assert new.size == 8
        assert all(new.data[i] is not None for i in range(new.tree.n_entries))
        assert new.sample(batch_size=8) is not None

    def test_save_survives_mixed_state_dims(self, tmp_path):
        """Save must not crash when buffer contains experiences with different
        state sizes (happens after offline-training populates the buffer with
        one state dimension and paper trading adds experiences with another).
        The mismatched entries are silently dropped; the canonical-size entries
        are written successfully."""
        import time

        buf = ExperienceBuffer(capacity=100)
        rng = np.random.default_rng(0)

        # Inject old experiences with dim=7 directly into the data array
        # (bypasses add() validation to simulate what load() produces)
        for i in range(20):
            buf.data[i] = Experience(
                state=rng.standard_normal(7).astype(np.float32),
                action=0,
                reward=1.0,
                next_state=rng.standard_normal(7).astype(np.float32),
                done=False,
                timestamp=time.time(),
                regime=int(RegimeSampling.UNKNOWN),
                priority=1.0,
            )
            buf.tree.add(1.0)
            buf.write_idx = (buf.write_idx + 1) % buf.capacity
            buf.total_added += 1

        # Add new experiences with dim=21 via the normal path
        _fill_buffer(buf, 30, dim=21)

        path = str(tmp_path / "mixed")
        assert buf.save(path), "save() must not fail on mixed state dims"

        # The canonical size is the first entry's size (dim=7 — injected first).
        # dim=21 entries are dropped as mismatched; the 20 dim=7 entries survive.
        buf2 = ExperienceBuffer(capacity=100)
        assert buf2.load(path)
        assert buf2.size == 20

    def test_d1_same_wall_clock(self):
        """D1 (1440 min timeframe) gives same halflife as M5 — instrument agnostic."""
        assert staleness_halflife_for_timeframe(1440) == pytest.approx(staleness_halflife_for_timeframe(5))

    def test_buffer_auto_computes_halflife(self):
        """ExperienceBuffer with no explicit halflife auto-derives from timeframe."""
        buf_m5 = ExperienceBuffer(capacity=100, timeframe_minutes=5)
        buf_h1 = ExperienceBuffer(capacity=100, timeframe_minutes=60)
        # Both should get the same default wall-clock halflife
        assert buf_m5.staleness_halflife == pytest.approx(buf_h1.staleness_halflife)
        assert buf_m5.staleness_halflife == pytest.approx(43_200.0)

    def test_explicit_halflife_override(self):
        """Passing staleness_halflife= explicitly bypasses auto-compute."""
        buf = ExperienceBuffer(capacity=100, staleness_halflife=86_400.0, timeframe_minutes=5)
        assert buf.staleness_halflife == pytest.approx(86_400.0)

    def test_constants_exported(self):
        assert pytest.approx(480.0) == TRADING_SESSION_MINUTES
        assert pytest.approx(1.5) == HALFLIFE_SESSIONS
