"""
Tests for Conv1dQNet (shared network architecture).

Verifies:
- Forward pass shape correctness
- Parameterization (n_features, n_actions)
- Weight loading compatibility
- All three consumers use the shared class
"""

import numpy as np
import pytest
import torch

from src.core.ddqn_network import Conv1dQNet


class TestConv1dQNet:
    def test_output_shape_3_actions(self):
        net = Conv1dQNet(n_features=7, n_actions=3)
        x = torch.randn(1, 60, 7)  # (batch, window, features)
        out = net(x)
        assert out.shape == (1, 3)

    def test_output_shape_2_actions(self):
        net = Conv1dQNet(n_features=10, n_actions=2)
        x = torch.randn(4, 60, 10)
        out = net(x)
        assert out.shape == (4, 2)

    def test_output_shape_batch_size(self):
        net = Conv1dQNet(n_features=4, n_actions=3)
        x = torch.randn(16, 60, 4)
        out = net(x)
        assert out.shape == (16, 3)

    def test_different_window_sizes(self):
        net = Conv1dQNet(n_features=7, n_actions=3)
        for window in [10, 30, 60, 120]:
            x = torch.randn(1, window, 7)
            out = net(x)
            assert out.shape == (1, 3), f"Failed for window={window}"

    def test_save_and_load_state_dict(self, tmp_path):
        net = Conv1dQNet(n_features=7, n_actions=3)
        path = tmp_path / "test_model.pt"
        torch.save(net.state_dict(), path)

        net2 = Conv1dQNet(n_features=7, n_actions=3)
        net2.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

        x = torch.randn(1, 60, 7)
        with torch.no_grad():
            assert torch.allclose(net(x), net2(x))

    def test_gradient_flow(self):
        net = Conv1dQNet(n_features=7, n_actions=3)
        x = torch.randn(2, 60, 7)
        out = net(x)
        loss = out.sum()
        loss.backward()
        # Check that all parameters have gradients
        for name, p in net.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_eval_mode_deterministic(self):
        net = Conv1dQNet(n_features=7, n_actions=3)
        net.eval()
        x = torch.randn(1, 60, 7)
        with torch.no_grad():
            out1 = net(x)
            out2 = net(x)
        assert torch.allclose(out1, out2)


class TestConv1dQNetImports:
    """Verify all three consumers import from the shared module."""

    def test_trigger_agent_loads_conv1d_qnet(self):
        """TriggerAgent._load_model should use Conv1dQNet."""
        import inspect
        from src.agents.trigger_agent import TriggerAgent
        source = inspect.getsource(TriggerAgent._load_model)
        assert "Conv1dQNet" in source
        assert "class TriggerQNet" not in source

    def test_harvester_agent_loads_conv1d_qnet(self):
        """HarvesterAgent._load_model should use Conv1dQNet."""
        import inspect
        from src.agents.harvester_agent import HarvesterAgent
        source = inspect.getsource(HarvesterAgent._load_model)
        assert "Conv1dQNet" in source
        assert "class HarvesterQNet" not in source

    def test_policy_uses_conv1d_qnet(self):
        """Policy.__init__ should reference Conv1dQNet."""
        import inspect
        from src.core.ctrader_ddqn_paper import Policy
        source = inspect.getsource(Policy.__init__)
        assert "Conv1dQNet" in source
        assert "class QNet" not in source
