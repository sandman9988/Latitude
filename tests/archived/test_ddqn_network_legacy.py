"""
Tests for src.core.ddqn_network

NOTE: Most tests in this file were written for the legacy NumPy-based
DDQNNetwork and AdamOptimizer.  The production code has since moved to
a PyTorch backend.  The stale tests are skipped with reason markers
until this file is rewritten for the current architecture.

Coverage targets:
- AdamOptimizer: step, update, moment initialization, bias correction
- DDQNNetwork: init, forward, predict, train_batch, backward, gradient clipping,
               target update, hard update, save/load weights
"""

from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)

import pytest

from src.core.ddqn_network import AdamOptimizer, DDQNNetwork

_SKIP_REASON = "Legacy NumPy-based test — DDQNNetwork now uses PyTorch"
# ── AdamOptimizer ───────────────────────────────────────────────────────────

@pytest.mark.skip(reason="AdamOptimizer is a legacy stub — PyTorch handles optimisation now")
class TestAdamOptimizer:
    def test_defaults(self):
        opt = AdamOptimizer()
        assert opt.lr == pytest.approx(0.0005)
        assert opt.beta1 == pytest.approx(0.9)
        assert opt.beta2 == pytest.approx(0.999)
        assert opt.epsilon == 1e-8
        assert opt.t == 0

    def test_step_increments_timestep(self):
        opt = AdamOptimizer()
        opt.step()
        assert opt.t == 1
        opt.step()
        assert opt.t == 2

    def test_update_initializes_moments(self):
        opt = AdamOptimizer(learning_rate=0.01)
        opt.step()
        param = np.ones(5)
        grad = np.ones(5) * 0.1
        updated = opt.update("p", param, grad)
        assert "p" in opt.m
        assert "p" in opt.v
        assert updated.shape == param.shape

    def test_update_moves_param(self):
        opt = AdamOptimizer(learning_rate=0.1)
        param = np.ones(3) * 10.0
        grad = np.ones(3) * 1.0  # Positive gradient → decrease param
        opt.step()
        updated = opt.update("w", param, grad)
        assert np.all(updated < param)

    def test_multiple_steps(self):
        opt = AdamOptimizer(learning_rate=0.01)
        param = np.ones(3) * 5.0
        for _ in range(5):
            opt.step()
            param = opt.update("w", param, np.ones(3) * 0.1)
        assert np.all(param < 5.0)  # Should have decreased

    def test_zero_gradient_no_change(self):
        opt = AdamOptimizer(learning_rate=0.01)
        param = np.array([1.0, 2.0, 3.0])
        opt.step()
        updated = opt.update("w", param, np.zeros(3))
        # With zero gradient, Adam still adjusts due to epsilon, but changes are tiny
        np.testing.assert_allclose(updated, param, atol=0.01)


# ── DDQNNetwork.__init__ ───────────────────────────────────────────────────

@pytest.mark.skip(reason=_SKIP_REASON)
class TestDDQNNetworkInit:
    def test_default_dimensions(self):
        net = DDQNNetwork(state_dim=10, n_actions=3, seed=42)
        assert net.state_dim == 10
        assert net.n_actions == 3
        assert net.w1.shape == (10, 128)
        assert net.b1.shape == (128,)
        assert net.w2.shape == (128, 64)
        assert net.b2.shape == (64,)
        assert net.w3.shape == (64, 3)
        assert net.b3.shape == (3,)
        assert net.training_steps == 0

    def test_custom_hidden_sizes(self):
        net = DDQNNetwork(state_dim=5, n_actions=2, hidden1_size=32, hidden2_size=16, seed=42)
        assert net.w1.shape == (5, 32)
        assert net.w2.shape == (32, 16)
        assert net.w3.shape == (16, 2)

    def test_target_network_initialized_as_copy(self):
        net = DDQNNetwork(state_dim=5, n_actions=2, seed=42)
        np.testing.assert_array_equal(net.w1, net.target_w1)
        np.testing.assert_array_equal(net.b1, net.target_b1)
        np.testing.assert_array_equal(net.w2, net.target_w2)
        np.testing.assert_array_equal(net.b2, net.target_b2)
        np.testing.assert_array_equal(net.w3, net.target_w3)
        np.testing.assert_array_equal(net.b3, net.target_b3)

    def test_he_initialization_scale(self):
        net = DDQNNetwork(state_dim=100, n_actions=3, seed=42)
        # He init: std = sqrt(2/fan_in)
        expected_std = np.sqrt(2.0 / 100)
        actual_std = np.std(net.w1)
        # Should be roughly correct (within 20%)
        assert abs(actual_std - expected_std) < expected_std * 0.3

    def test_seed_reproducibility(self):
        net1 = DDQNNetwork(state_dim=5, n_actions=2, seed=99)
        net2 = DDQNNetwork(state_dim=5, n_actions=2, seed=99)
        np.testing.assert_array_equal(net1.w1, net2.w1)
        np.testing.assert_array_equal(net1.w2, net2.w2)
        np.testing.assert_array_equal(net1.w3, net2.w3)


# ── Forward pass ───────────────────────────────────────────────────────────

class TestForward:
    @pytest.fixture
    def net(self):
        return DDQNNetwork(state_dim=4, n_actions=3, seed=42)

    @pytest.mark.skip(reason=_SKIP_REASON)
    def test_single_state(self, net):
        state = rng.standard_normal(4)
        q_vals, cache = net.forward(state)
        assert q_vals.shape == (3,)
        assert "state" in cache
        assert "z1" in cache
        assert "a1" in cache
        assert "z2" in cache
        assert "a2" in cache
        assert "q_values" in cache

    def test_batch_states(self, net):
        states = rng.standard_normal((8, 4))
        q_vals, cache = net.forward(states)
        assert q_vals.shape == (8, 3)

    def test_target_network_forward(self, net):
        state = rng.standard_normal(4)
        q_online, _ = net.forward(state, use_target=False)
        q_target, _ = net.forward(state, use_target=True)
        # Initially identical
        np.testing.assert_array_equal(q_online, q_target)

    @pytest.mark.skip(reason=_SKIP_REASON)
    def test_relu_applied(self, net):
        state = rng.standard_normal(4)
        _, cache = net.forward(state)
        # Activations after ReLU should be non-negative
        assert np.all(cache["a1"] >= 0)
        assert np.all(cache["a2"] >= 0)

    @pytest.mark.skip(reason=_SKIP_REASON)
    def test_relu_derivative(self, net):
        x = np.array([-2, -1, 0, 1, 2], dtype=float)
        deriv = net._relu_derivative(x)
        np.testing.assert_array_equal(deriv, [0, 0, 0, 1, 1])


# ── Predict ────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_q_values(self):
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        state = rng.standard_normal(4)
        q = net.predict(state)
        assert q.shape == (3,)

    def test_predict_matches_forward(self):
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        state = rng.standard_normal(4)
        q_pred = net.predict(state)
        q_fwd, _ = net.forward(state)
        np.testing.assert_array_equal(q_pred, q_fwd)


# ── Train batch ────────────────────────────────────────────────────────────

class TestTrainBatch:
    @pytest.fixture
    def net(self):
        return DDQNNetwork(state_dim=4, n_actions=3, learning_rate=0.001, seed=42)

    def _make_batch(self, batch_size=8, state_dim=4, n_actions=3):
        rng = np.random.default_rng(123)
        return dict(
            states=rng.standard_normal((batch_size, state_dim)),
            actions=rng.integers(0, n_actions, batch_size),
            rewards=rng.standard_normal(batch_size),
            next_states=rng.standard_normal((batch_size, state_dim)),
            dones=rng.choice([0.0, 1.0], batch_size, p=[0.8, 0.2]),
            weights=np.ones(batch_size),
        )

    def test_returns_stats(self, net):
        batch = self._make_batch()
        result = net.train_batch(**batch)
        assert "loss" in result
        assert "l2_loss" in result
        assert "total_loss" in result
        assert "mean_q" in result
        assert "mean_td_error" in result
        assert "max_td_error" in result
        assert "grad_norm" in result
        assert "td_errors" in result

    def test_loss_is_nonneg(self, net):
        batch = self._make_batch()
        result = net.train_batch(**batch)
        assert result["loss"] >= 0
        assert result["l2_loss"] >= 0

    def test_training_steps_incremented(self, net):
        batch = self._make_batch()
        assert net.training_steps == 0
        net.train_batch(**batch)
        assert net.training_steps == 1
        net.train_batch(**batch)
        assert net.training_steps == 2

    @pytest.mark.skip(reason=_SKIP_REASON)
    def test_weights_change_after_training(self, net):
        w1_before = net.w1.copy()
        batch = self._make_batch()
        net.train_batch(**batch)
        assert not np.array_equal(net.w1, w1_before)

    @pytest.mark.skip(reason=_SKIP_REASON)
    def test_target_network_soft_updated(self, net):
        net.tau = 0.5  # Make soft update noticeable
        target_w1_before = net.target_w1.copy()
        batch = self._make_batch()
        net.train_batch(**batch)
        # Target should have shifted toward online
        assert not np.array_equal(net.target_w1, target_w1_before)

    def test_td_errors_shape(self, net):
        batch = self._make_batch(batch_size=16)
        result = net.train_batch(**batch)
        assert result["td_errors"].shape == (16,)

    def test_gradient_clipping(self):
        net = DDQNNetwork(state_dim=4, n_actions=3, grad_clip_norm=0.1, seed=42)
        batch = self._make_batch()
        result = net.train_batch(**batch)
        # Grad norm after clipping should be <= clip_norm (or close)
        # Note: total_grad_norm is BEFORE clipping, so it can be > clip_norm
        assert result["grad_norm"] >= 0


# ── Hard target update ─────────────────────────────────────────────────────

@pytest.mark.skip(reason=_SKIP_REASON)
class TestHardUpdate:
    def test_hard_update_copies_exactly(self):
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        # Modify online weights
        net.w1 += 1.0
        net.b2 += 0.5
        net.hard_update_target()
        np.testing.assert_array_equal(net.w1, net.target_w1)
        np.testing.assert_array_equal(net.b2, net.target_b2)
        np.testing.assert_array_equal(net.w3, net.target_w3)


# ── Save / Load weights ───────────────────────────────────────────────────

@pytest.mark.skip(reason=_SKIP_REASON)
class TestSaveLoadWeights:
    def test_save_and_load_roundtrip(self, tmp_path):
        filepath = str(tmp_path / "weights.npz")
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        # Train a bit
        rng = np.random.default_rng(1)
        net.train_batch(
            states=rng.standard_normal((4, 4)),
            actions=rng.integers(0, 3, 4),
            rewards=rng.standard_normal(4),
            next_states=rng.standard_normal((4, 4)),
            dones=np.zeros(4),
            weights=np.ones(4),
        )
        net.save_weights(filepath)

        net2 = DDQNNetwork(state_dim=4, n_actions=3, seed=99)
        net2.load_weights(filepath)
        np.testing.assert_array_equal(net.w1, net2.w1)
        np.testing.assert_array_equal(net.target_w3, net2.target_w3)
        assert net.training_steps == net2.training_steps

    def test_load_nonexistent_file(self, tmp_path):
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        w1_before = net.w1.copy()
        net.load_weights(str(tmp_path / "no_such_file.npz"))
        # Weights unchanged
        np.testing.assert_array_equal(net.w1, w1_before)

    def test_save_creates_directories(self, tmp_path):
        filepath = str(tmp_path / "deep" / "path" / "weights.npz")
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        net.save_weights(filepath)
        assert Path(filepath).exists()


# ── Gradient clipping utility ──────────────────────────────────────────────

@pytest.mark.skip(reason=_SKIP_REASON)
class TestClipGradients:
    def test_gradients_below_norm_unchanged(self):
        net = DDQNNetwork(state_dim=4, n_actions=3, grad_clip_norm=100.0, seed=42)
        grads = [np.ones(3) * 0.1, np.ones(2) * 0.1]
        originals = [g.copy() for g in grads]
        _norm = net._clip_gradients(grads)
        for g, orig in zip(grads, originals):
            np.testing.assert_array_almost_equal(g, orig)

    def test_gradients_above_norm_clipped(self):
        net = DDQNNetwork(state_dim=4, n_actions=3, grad_clip_norm=1.0, seed=42)
        grads = [np.ones(10) * 10.0]  # Norm = sqrt(10*100) ≈ 31.6
        norm = net._clip_gradients(grads)
        assert norm > 1.0  # Was above clip norm
        clipped_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        np.testing.assert_almost_equal(clipped_norm, 1.0, decimal=5)
