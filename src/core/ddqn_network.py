"""
Enhanced DDQN Neural Network with Prioritized Experience Replay

Backend: PyTorch with AMD ROCm / CUDA GPU acceleration (CPU fallback).

Features:
- GPU acceleration (ROCm gfx1100 / CUDA) with transparent CPU fallback
- Proper Adam optimizer with bias correction
- Gradient clipping by norm
- L2 weight decay
- He initialization for ReLU networks
- Soft target network updates (τ parameter)
- Double DQN target calculation
- Network persistence — saves torch .pt format;
  also loads legacy NumPy .npz checkpoints transparently
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.constants import GAMMA, GRAD_CLIP_NORM, L2_WEIGHT, LEARNING_RATE, TAU

LOG = logging.getLogger(__name__)


# ── Device selection ──────────────────────────────────────────────────────────
def _select_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        LOG.info("[DDQN] GPU: %s (%.1f GB VRAM)  — ROCm/CUDA backend", name, vram)
    else:
        dev = torch.device("cpu")
        LOG.info("[DDQN] GPU not available — falling back to CPU")
    return dev


DEVICE: torch.device = _select_device()


# ── Conv1d Q-Network (shared by Trigger / Harvester / Policy agents) ─────────
class Conv1dQNet(nn.Module):
    """Conv1d Q-Network for agents operating on (batch, window, features) inputs.

    Architecture (temporal_pool_size=4, the new default):
        Conv1d(n_features, 64, k=5, pad=2) → ReLU
        Conv1d(64, 64, k=5, pad=2) → ReLU
        AdaptiveAvgPool1d(temporal_pool_size) → Flatten
        Linear(64 * temporal_pool_size, 128) → ReLU
        Linear(128, n_actions)

    With temporal_pool_size=4 the pooled output retains 4 coarse time-steps
    (early / mid / late / final quarter of the window), giving the linear
    head temporal context that AdaptiveAvgPool1d(1) discarded entirely.

    Backward compatibility: pass temporal_pool_size=1 to reproduce the old
    architecture exactly (required when loading pre-trained weights that were
    saved with the pool-to-1 architecture).

    Input shape:  (B, T, F)  – batch, time/window, features
    Output shape: (B, n_actions)
    """

    def __init__(self, n_features: int, n_actions: int = 3, temporal_pool_size: int = 4):
        super().__init__()
        fc_in = 64 * temporal_pool_size
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(temporal_pool_size),
            nn.Flatten(),
            nn.Linear(fc_in, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T)
        return self.net(x.transpose(1, 2))


# ── Internal MLP module (used by DDQNNetwork) ────────────────────────────────
class _QNet(nn.Module):
    """3-layer MLP: state_dim → hidden1 → hidden2 → n_actions (linear out)."""

    def __init__(self, state_dim: int, hidden1: int, hidden2: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions),
        )
        self._he_init()

    def _he_init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



# ── Public API ────────────────────────────────────────────────────────────────
class DDQNNetwork:
    """
    Double Deep Q-Network with online and target networks.

    Architecture:
        Input (state_dim) → Hidden1 (128) → Hidden2 (64) → Output (n_actions)

    All public methods accept / return NumPy arrays for drop-in compatibility
    with the existing agent code. GPU transfers are handled internally.
    """

    def __init__(  # noqa: PLR0913
        self,
        state_dim: int,
        n_actions: int,
        hidden1_size: int = 128,
        hidden2_size: int = 64,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        tau: float = TAU,
        l2_weight: float = L2_WEIGHT,
        grad_clip_norm: float = GRAD_CLIP_NORM,
        seed: int | None = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.l2_weight = l2_weight
        self.grad_clip_norm = grad_clip_norm
        self.device = DEVICE

        if seed is not None:
            torch.manual_seed(seed)

        self.online = _QNet(state_dim, hidden1_size, hidden2_size, n_actions).to(self.device)
        self.target = _QNet(state_dim, hidden1_size, hidden2_size, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=learning_rate,
            weight_decay=l2_weight,
        )

        self.training_steps: int = 0
        self.total_grad_norm: float = 0.0

        LOG.info(
            "[DDQN] Initialized: state_dim=%d, actions=%d, hidden=[%d,%d],"
            " lr=%.4f, tau=%.4f, device=%s",
            state_dim, n_actions, hidden1_size, hidden2_size,
            learning_rate, tau, self.device,
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def _to_tensor(self, arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    # ── inference ──────────────────────────────────────────────────────────

    def predict(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Return Q-values for *state* as a NumPy array (no grad)."""
        net = self.target if use_target else self.online
        single = state.ndim == 1
        s = self._to_tensor(state)
        if single:
            s = s.unsqueeze(0)
        with torch.no_grad():
            q = net(s)
        out = q.cpu().numpy()
        return out[0] if single else out

    def forward(self, state: np.ndarray, use_target: bool = False):
        """Legacy shim — returns (q_values_numpy, {}).

        Cache is empty because backprop is handled by PyTorch autograd.
        """
        return self.predict(state, use_target=use_target), {}

    # ── training ───────────────────────────────────────────────────────────

    def train_batch(  # noqa: PLR0913
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        weights: np.ndarray,
    ) -> dict:
        """Train one batch; returns loss stats and per-sample td_errors."""
        s  = self._to_tensor(states)
        ns = self._to_tensor(next_states)
        a  = self._to_tensor(actions, dtype=torch.long)
        r  = self._to_tensor(rewards)
        d  = self._to_tensor(dones)
        w  = self._to_tensor(weights)

        # Online Q-values for taken actions
        self.online.train()
        q_online = self.online(s)                                          # (B, A)
        q_current = q_online.gather(1, a.unsqueeze(1)).squeeze(1)         # (B,)

        # Double DQN: online selects next action, target evaluates it
        with torch.no_grad():
            next_actions = self.online(ns).argmax(dim=1)                   # (B,)
            q_next_max   = self.target(ns).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)                                                   # (B,)

        td_targets = r + self.gamma * q_next_max * (1.0 - d)
        td_errors_t = (td_targets - q_current).detach()                   # no grad

        # Importance-sampling weighted MSE
        loss = (w * td_errors_t ** 2).mean()

        # Separate forward for the backward pass (avoid double-graph issue)
        q_bp = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss_bp = (w * (td_targets - q_bp) ** 2).mean()

        self.optimizer.zero_grad()
        loss_bp.backward()

        total_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), self.grad_clip_norm
        )
        self.total_grad_norm = float(total_norm)
        self.optimizer.step()

        # Soft update target
        self._soft_update_target()
        self.online.eval()
        self.training_steps += 1

        td_np = td_errors_t.cpu().numpy()
        return {
            "loss":          float(loss),
            "l2_loss":       0.0,          # absorbed into Adam weight_decay
            "total_loss":    float(loss),
            "mean_q":        float(q_online.detach().mean()),
            "mean_td_error": float(np.mean(np.abs(td_np))),
            "max_td_error":  float(np.max(np.abs(td_np))),
            "grad_norm":     self.total_grad_norm,
            "td_errors":     td_np,
        }

    # ── target network ─────────────────────────────────────────────────────

    def _soft_update_target(self):
        """θ_target ← τ·θ_online + (1−τ)·θ_target"""
        with torch.no_grad():
            for p_on, p_tgt in zip(
                self.online.parameters(), self.target.parameters(), strict=False
            ):
                p_tgt.data.mul_(1.0 - self.tau)
                p_tgt.data.add_(self.tau * p_on.data)

    def hard_update_target(self):
        """Copy online → target (τ = 1)."""
        self.target.load_state_dict(self.online.state_dict())
        LOG.info("[DDQN] Hard update: copied online → target")

    # ── persistence ────────────────────────────────────────────────────────

    def save_weights(self, filepath: str):
        """Save to *filepath* (torch .pt format, .pt suffix auto-added)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        pt_path = path.with_suffix(".pt") if path.suffix != ".pt" else path
        torch.save(
            {
                "online":         self.online.state_dict(),
                "target":         self.target.state_dict(),
                "optimizer":      self.optimizer.state_dict(),
                "training_steps": self.training_steps,
            },
            pt_path,
        )
        LOG.info("[DDQN] Saved weights → %s (step %d)", pt_path, self.training_steps)

    def load_weights(self, filepath: str):
        """Load weights from *filepath*.

        Supports:
        - New torch .pt checkpoint
        - Legacy NumPy .npz checkpoint (migrates weight layout automatically)
        """
        path = Path(filepath)
        pt_path = path.with_suffix(".pt") if path.suffix != ".pt" else path

        # Prefer .pt, fall back to the exact path, then .npz
        candidates = [pt_path, path, path.with_suffix(".npz")]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            LOG.warning("[DDQN] Weight file not found: %s", filepath)
            return

        if found.suffix in (".pt", ".pth"):
            ckpt = torch.load(found, map_location=self.device, weights_only=True)
            self.online.load_state_dict(ckpt["online"])
            self.target.load_state_dict(ckpt["target"])
            if "optimizer" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                except Exception as exc:
                    LOG.warning("[DDQN] Skipping optimizer state load (shape mismatch?): %s", exc)
            self.training_steps = int(ckpt.get("training_steps", 0))
            LOG.info("[DDQN] Loaded %s (step %d)", found, self.training_steps)
        else:
            self._load_npz(found)

    def _load_npz(self, path: Path):
        """Migrate a legacy NumPy .npz checkpoint into the torch model.

        NumPy layout: w1 (fan_in × fan_out) → torch expects (fan_out × fan_in).
        """
        data = np.load(path)
        # (torch_key, npz_key, transpose?)
        mapping = [
            ("net.0.weight", "w1",        True),
            ("net.0.bias",   "b1",        False),
            ("net.2.weight", "w2",        True),
            ("net.2.bias",   "b2",        False),
            ("net.4.weight", "w3",        True),
            ("net.4.bias",   "b3",        False),
        ]

        def _apply(module: _QNet, npz_prefix: str):
            sd = module.state_dict()
            for torch_key, npz_key, transpose in mapping:
                arr = data[npz_prefix + npz_key]
                t = torch.as_tensor(arr.T if transpose else arr, dtype=torch.float32)
                sd[torch_key] = t
            module.load_state_dict(sd)

        _apply(self.online, "")
        _apply(self.target, "target_")

        if "training_steps" in data:
            self.training_steps = int(data["training_steps"])

        LOG.info("[DDQN] Migrated NumPy checkpoint %s → torch (step %d)", path, self.training_steps)
