"""Tests for _add_replay_experiences shape correctness in TFAgent.

Regression guard for the harvester train_batch shape mismatch:
  Trigger state: (window, 18)  — trigger DDQNNetwork state_dim = window*18
  Harvester state: (window, 21) — harvester DDQNNetwork state_dim = window*21

Before the fix, _exit_state (harvester-shaped) was passed as next_state to
the trigger buffer, and the trigger-shaped _entry_state was used as a fallback
state for the harvester buffer — causing shape errors in both DDQNNetworks.
"""

from __future__ import annotations

import numpy as np

from src.agents.dual_policy import DualPolicy, DualPolicyConfig


# ---------------------------------------------------------------------------
# Helpers — build a minimal DualPolicy and exercise experience paths
# ---------------------------------------------------------------------------

WINDOW = 8          # small for speed
TRIG_FEATURES = 18  # 7 base + 5 geometry + 6 event
HARV_FEATURES = 21  # trigger_features + 3 position stats


def _make_policy(enable_training: bool = True) -> DualPolicy:
    cfg = DualPolicyConfig(
        window=WINDOW,
        enable_training=enable_training,
        path_geometry=True,
        enable_event_features=True,
        trigger_buffer_capacity=200,
        harvester_buffer_capacity=200,
    )
    return DualPolicy(config=cfg)


def _trig_state() -> np.ndarray:
    return np.random.default_rng(0).random((WINDOW, TRIG_FEATURES), dtype=np.float32)


def _harv_state() -> np.ndarray:
    return np.random.default_rng(1).random((WINDOW, HARV_FEATURES), dtype=np.float32)


# ---------------------------------------------------------------------------
# Shape invariant tests
# ---------------------------------------------------------------------------

class TestTriggerExperienceShape:
    """Trigger buffer must only receive (window, 18)-shaped states."""

    def test_add_trigger_experience_correct_shape(self):
        policy = _make_policy()
        s = _trig_state()
        policy.add_trigger_experience(state=s, action=1, reward=0.5, next_state=s, done=True)
        buf = policy.trigger.buffer
        assert buf is not None and buf.tree.n_entries == 1

    def test_trigger_buffer_state_shape(self):
        policy = _make_policy()
        s = _trig_state()
        policy.add_trigger_experience(state=s, action=1, reward=0.5, next_state=s, done=True)
        batch = policy.trigger.buffer.sample(batch_size=1)
        assert batch is not None
        # States must be trigger-shaped before flatten
        assert batch["states"].shape == (1, WINDOW, TRIG_FEATURES)
        assert batch["next_states"].shape == (1, WINDOW, TRIG_FEATURES)

    def test_trigger_ddqn_state_dim_matches_buffer(self):
        policy = _make_policy()
        expected = WINDOW * TRIG_FEATURES
        assert policy.trigger.ddqn.state_dim == expected


class TestHarvesterExperienceShape:
    """Harvester buffer must only receive (window, 21)-shaped states."""

    def test_add_harvester_experience_correct_shape(self):
        policy = _make_policy()
        s = _harv_state()
        policy.add_harvester_experience(state=s, action=1, reward=0.5, next_state=s, done=True)
        buf = policy.harvester.buffer
        assert buf is not None and buf.tree.n_entries == 1

    def test_harvester_buffer_state_shape(self):
        policy = _make_policy()
        s = _harv_state()
        policy.add_harvester_experience(state=s, action=1, reward=0.5, next_state=s, done=True)
        batch = policy.harvester.buffer.sample(batch_size=1)
        assert batch is not None
        assert batch["states"].shape == (1, WINDOW, HARV_FEATURES)
        assert batch["next_states"].shape == (1, WINDOW, HARV_FEATURES)

    def test_harvester_ddqn_state_dim_matches_buffer(self):
        policy = _make_policy()
        expected = WINDOW * HARV_FEATURES
        assert policy.harvester.ddqn.state_dim == expected


class TestAddReplayExperiencesFallback:
    """_add_replay_experiences must not mix trigger and harvester state shapes."""

    def _simulate_forced_close(
        self, policy: DualPolicy, harv_last_state: np.ndarray | None
    ) -> None:
        """Simulate what _add_replay_experiences does when called from _close_position."""
        entry_state = _trig_state()

        harv_exit = None
        if harv_last_state is not None:
            harv_exit = harv_last_state.copy()

        # Trigger: next_state uses entry_state (same shape, done=True zeros Bellman target)
        policy.add_trigger_experience(
            state=entry_state, action=1, reward=0.3,
            next_state=entry_state, done=True,
        )
        # Harvester: only add if harvester state available
        if harv_exit is not None:
            policy.add_harvester_experience(
                state=harv_exit, action=1, reward=0.5,
                next_state=harv_exit, done=True,
            )

    def test_no_shape_mismatch_when_harv_state_available(self):
        policy = _make_policy()
        self._simulate_forced_close(policy, _harv_state())
        assert policy.trigger.buffer.tree.n_entries == 1
        assert policy.harvester.buffer.tree.n_entries == 1
        # Verify both buffers have correct shapes
        tb = policy.trigger.buffer.sample(batch_size=1)
        hb = policy.harvester.buffer.sample(batch_size=1)
        assert tb["states"].shape[-1] == TRIG_FEATURES
        assert hb["states"].shape[-1] == HARV_FEATURES

    def test_harvester_skipped_when_no_harv_state(self):
        """When harvester.last_state is None (very short trade), harvester buffer stays empty."""
        policy = _make_policy()
        self._simulate_forced_close(policy, harv_last_state=None)
        assert policy.trigger.buffer.tree.n_entries == 1
        assert policy.harvester.buffer.tree.n_entries == 0

    def test_trigger_ddqn_accepts_flattened_trigger_states(self):
        """Trigger DDQNNetwork forward pass must not raise with trigger-shaped states."""
        policy = _make_policy()
        s = _trig_state()
        policy.add_trigger_experience(state=s, action=1, reward=0.3, next_state=s, done=True)
        flat = s.reshape(1, -1).astype(np.float64)
        assert flat.shape == (1, WINDOW * TRIG_FEATURES)
        result = policy.trigger.ddqn.predict(flat)
        assert result.shape[1] == 3  # 3 actions (HOLD/LONG/SHORT)

    def test_harvester_ddqn_accepts_flattened_harvester_states(self):
        """Harvester DDQNNetwork forward pass must not raise with harvester-shaped states."""
        policy = _make_policy()
        s = _harv_state()
        policy.add_harvester_experience(state=s, action=1, reward=0.5, next_state=s, done=True)
        flat = s.reshape(1, -1).astype(np.float64)
        assert flat.shape == (1, WINDOW * HARV_FEATURES)
        result = policy.harvester.ddqn.predict(flat)
        assert result.shape[1] == 2  # 2 actions (HOLD/CLOSE)
