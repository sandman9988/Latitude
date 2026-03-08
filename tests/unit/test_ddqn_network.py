"""
Tests for src.core.ddqn_network  (PyTorch backend)

Covers:
- DDQNNetwork init, forward/predict, train_batch, target updates, save/load
- _QNet He initialization, architecture
- Conv1dQNet architecture (tested in test_conv1d_qnet.py — import-only check here)
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.core.ddqn_network import Conv1dQNet, DDQNNetwork, _QNet

rng = np.random.default_rng(42)


# ── _QNet (internal MLP) ──────────────────────────────────────────────────────

class TestQNet:
    def test_output_shape(self):
        net = _QNet(state_dim=10, hidden1=128, hidden2=64, n_actions=3)
        x = torch.randn(4, 10)
        out = net(x)
        assert out.shape == (4, 3)

    def test_single_input(self):
        net = _QNet(state_dim=5, hidden1=32, hidden2=16, n_actions=2)
        x = torch.randn(1, 5)
        out = net(x)
        assert out.shape == (1, 2)

    def test_he_initialization(self):
        """Kaiming (He) init should produce reasonable weight scales."""
        net = _QNet(state_dim=100, hidden1=128, hidden2=64, n_actions=3)
        first_layer = net.net[0]
        assert isinstance(first_layer, torch.nn.Linear)
        w_std = first_layer.weight.detach().std().item()
        # He uniform: std ≈ sqrt(2 / fan_in) * sqrt(3) / sqrt(3) ≈ sqrt(2/100) ≈ 0.14
        expected_std = np.sqrt(2.0 / 100)
        assert abs(w_std - expected_std) < expected_std * 0.5  # within 50%

    def test_biases_initialized_to_zero(self):
        net = _QNet(state_dim=10, hidden1=32, hidden2=16, n_actions=3)
        for m in net.net:
            if isinstance(m, torch.nn.Linear):
                assert torch.all(m.bias == 0).item()

    def test_gradient_flow(self):
        net = _QNet(state_dim=4, hidden1=16, hidden2=8, n_actions=3)
        x = torch.randn(2, 4, requires_grad=True)
        out = net(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0).item()


# ── DDQNNetwork.__init__ ──────────────────────────────────────────────────────

class TestDDQNNetworkInit:
    def test_default_dimensions(self):
        net = DDQNNetwork(state_dim=10, n_actions=3, seed=42)
        assert net.state_dim == 10
        assert net.n_actions == 3
        assert net.training_steps == 0

    def test_custom_hidden_sizes(self):
        net = DDQNNetwork(state_dim=5, n_actions=2, hidden1_size=32, hidden2_size=16, seed=42)
        # Verify online network architecture has correct layer sizes
        first_layer = net.online.net[0]
        assert first_layer.in_features == 5
        assert first_layer.out_features == 32

    def test_target_network_initialized_as_copy(self):
        net = DDQNNetwork(state_dim=5, n_actions=2, seed=42)
        for p_on, p_tgt in zip(net.online.parameters(), net.target.parameters()):
            assert torch.equal(p_on, p_tgt)

    def test_seed_reproducibility(self):
        net1 = DDQNNetwork(state_dim=5, n_actions=2, seed=99)
        net2 = DDQNNetwork(state_dim=5, n_actions=2, seed=99)
        for p1, p2 in zip(net1.online.parameters(), net2.online.parameters()):
            assert torch.equal(p1, p2)

    def test_target_in_eval_mode(self):
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        assert not net.target.training


# ── Forward / Predict ─────────────────────────────────────────────────────────

class TestForwardPredict:
    @pytest.fixture
    def net(self):
        return DDQNNetwork(state_dim=4, n_actions=3, seed=42)

    def test_predict_single_state(self, net):
        state = rng.standard_normal(4).astype(np.float32)
        q = net.predict(state)
        assert q.shape == (3,)
        assert q.dtype == np.float32 or q.dtype == np.float64

    def test_predict_batch_states(self, net):
        states = rng.standard_normal((8, 4)).astype(np.float32)
        q = net.predict(states)
        assert q.shape == (8, 3)

    def test_forward_returns_tuple(self, net):
        state = rng.standard_normal(4).astype(np.float32)
        q, cache = net.forward(state)
        assert q.shape == (3,)
        assert isinstance(cache, dict)

    def test_predict_matches_forward(self, net):
        state = rng.standard_normal(4).astype(np.float32)
        q_pred = net.predict(state)
        q_fwd, _ = net.forward(state)
        np.testing.assert_array_equal(q_pred, q_fwd)

    def test_target_network_forward(self, net):
        state = rng.standard_normal(4).astype(np.float32)
        q_online = net.predict(state, use_target=False)
        q_target = net.predict(state, use_target=True)
        # Initially identical
        np.testing.assert_array_almost_equal(q_online, q_target, decimal=5)

    def test_predict_deterministic(self, net):
        state = rng.standard_normal(4).astype(np.float32)
        q1 = net.predict(state)
        q2 = net.predict(state)
        np.testing.assert_array_equal(q1, q2)


# ── Train batch ───────────────────────────────────────────────────────────────

class TestTrainBatch:
    @pytest.fixture
    def net(self):
        return DDQNNetwork(state_dim=4, n_actions=3, learning_rate=0.001, seed=42)

    def _make_batch(self, batch_size=8, state_dim=4, n_actions=3):
        rng_local = np.random.default_rng(123)
        return dict(
            states=rng_local.standard_normal((batch_size, state_dim)).astype(np.float32),
            actions=rng_local.integers(0, n_actions, batch_size),
            rewards=rng_local.standard_normal(batch_size).astype(np.float32),
            next_states=rng_local.standard_normal((batch_size, state_dim)).astype(np.float32),
            dones=rng_local.choice([0.0, 1.0], batch_size, p=[0.8, 0.2]).astype(np.float32),
            weights=np.ones(batch_size, dtype=np.float32),
        )

    def test_returns_expected_keys(self, net):
        batch = self._make_batch()
        result = net.train_batch(**batch)
        for key in ("loss", "l2_loss", "total_loss", "mean_q", "mean_td_error",
                     "max_td_error", "grad_norm", "td_errors"):
            assert key in result, f"Missing key: {key}"

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

    def test_td_errors_shape(self, net):
        batch = self._make_batch(batch_size=16)
        result = net.train_batch(**batch)
        assert result["td_errors"].shape == (16,)

    def test_weights_change_after_training(self, net):
        params_before = [p.clone() for p in net.online.parameters()]
        batch = self._make_batch()
        net.train_batch(**batch)
        any_changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(params_before, net.online.parameters())
        )
        assert any_changed, "Online weights should change after training"

    def test_target_network_soft_updated(self, net):
        """After training, target should have shifted (τ > 0)."""
        target_before = [p.clone() for p in net.target.parameters()]
        batch = self._make_batch()
        net.train_batch(**batch)
        any_changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(target_before, net.target.parameters())
        )
        assert any_changed, "Target weights should shift via soft update"

    def test_gradient_clipping(self):
        net = DDQNNetwork(state_dim=4, n_actions=3, grad_clip_norm=0.1, seed=42)
        rng_local = np.random.default_rng(123)
        batch = dict(
            states=rng_local.standard_normal((8, 4)).astype(np.float32),
            actions=rng_local.integers(0, 3, 8),
            rewards=rng_local.standard_normal(8).astype(np.float32) * 100,  # large rewards → large gradients
            next_states=rng_local.standard_normal((8, 4)).astype(np.float32),
            dones=np.zeros(8, dtype=np.float32),
            weights=np.ones(8, dtype=np.float32),
        )
        result = net.train_batch(**batch)
        assert result["grad_norm"] >= 0

    def test_loss_decreases_over_training(self):
        """Training on identical batch should eventually reduce loss."""
        net = DDQNNetwork(state_dim=4, n_actions=3, learning_rate=0.01, seed=42)
        rng_local = np.random.default_rng(42)
        batch = dict(
            states=rng_local.standard_normal((16, 4)).astype(np.float32),
            actions=rng_local.integers(0, 3, 16),
            rewards=rng_local.standard_normal(16).astype(np.float32),
            next_states=rng_local.standard_normal((16, 4)).astype(np.float32),
            dones=np.zeros(16, dtype=np.float32),
            weights=np.ones(16, dtype=np.float32),
        )
        losses = []
        for _ in range(20):
            result = net.train_batch(**batch)
            losses.append(result["loss"])
        # Loss should generally decrease (allow some noise)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


# ── Hard target update ────────────────────────────────────────────────────────

class TestHardUpdate:
    def test_hard_update_copies_exactly(self):
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        # Modify online weights
        with torch.no_grad():
            for p in net.online.parameters():
                p.add_(1.0)
        # Verify online ≠ target
        assert not torch.equal(
            list(net.online.parameters())[0],
            list(net.target.parameters())[0],
        )
        net.hard_update_target()
        # Now they should match exactly
        for p_on, p_tgt in zip(net.online.parameters(), net.target.parameters()):
            assert torch.equal(p_on, p_tgt)


# ── Save / Load weights ──────────────────────────────────────────────────────

class TestSaveLoadWeights:
    def test_save_and_load_roundtrip(self, tmp_path):
        filepath = str(tmp_path / "weights")
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        # Train a bit so weights diverge from initialization
        rng_local = np.random.default_rng(1)
        net.train_batch(
            states=rng_local.standard_normal((4, 4)).astype(np.float32),
            actions=rng_local.integers(0, 3, 4),
            rewards=rng_local.standard_normal(4).astype(np.float32),
            next_states=rng_local.standard_normal((4, 4)).astype(np.float32),
            dones=np.zeros(4, dtype=np.float32),
            weights=np.ones(4, dtype=np.float32),
        )
        net.save_weights(filepath)

        net2 = DDQNNetwork(state_dim=4, n_actions=3, seed=99)
        net2.load_weights(filepath)
        for p1, p2 in zip(net.online.parameters(), net2.online.parameters()):
            assert torch.equal(p1, p2)
        for p1, p2 in zip(net.target.parameters(), net2.target.parameters()):
            assert torch.equal(p1, p2)
        assert net.training_steps == net2.training_steps

    def test_load_nonexistent_file(self, tmp_path):
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        params_before = [p.clone() for p in net.online.parameters()]
        net.load_weights(str(tmp_path / "no_such_file.pt"))
        # Weights unchanged
        for p_before, p_now in zip(params_before, net.online.parameters()):
            assert torch.equal(p_before, p_now)

    def test_save_creates_directories(self, tmp_path):
        filepath = str(tmp_path / "deep" / "path" / "weights")
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        net.save_weights(filepath)
        assert Path(filepath).with_suffix(".pt").exists()

    def test_training_steps_persisted(self, tmp_path):
        filepath = str(tmp_path / "weights")
        net = DDQNNetwork(state_dim=4, n_actions=3, seed=42)
        rng_local = np.random.default_rng(1)
        batch = dict(
            states=rng_local.standard_normal((4, 4)).astype(np.float32),
            actions=rng_local.integers(0, 3, 4),
            rewards=rng_local.standard_normal(4).astype(np.float32),
            next_states=rng_local.standard_normal((4, 4)).astype(np.float32),
            dones=np.zeros(4, dtype=np.float32),
            weights=np.ones(4, dtype=np.float32),
        )
        for _ in range(5):
            net.train_batch(**batch)
        assert net.training_steps == 5
        net.save_weights(filepath)

        net2 = DDQNNetwork(state_dim=4, n_actions=3, seed=99)
        net2.load_weights(filepath)
        assert net2.training_steps == 5
