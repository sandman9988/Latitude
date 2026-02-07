"""
Enhanced DDQN Neural Network with Prioritized Experience Replay

Features:
- Proper Adam optimizer with bias correction
- Gradient clipping by norm
- L2 regularization
- He initialization for ReLU networks
- Soft target network updates (τ parameter)
- Double DQN target calculation
- Network persistence (save/load weights)
"""

import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)


class AdamOptimizer:
    """
    Adam optimizer with bias correction.

    Reference: Kingma & Ba (2015) "Adam: A Method for Stochastic Optimization"
    """

    def __init__(
        self,
        learning_rate: float = 0.0005,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Moment estimates (initialized lazily)
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
        self.t = 0  # Time step

    def step(self):
        """Advance the timestep counter. Call once per batch before updating all parameters."""
        self.t += 1

    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Update parameter using Adam.

        Args:
            param_name: Identifier for parameter (for moment storage)
            param: Current parameter value
            grad: Gradient

        Returns:
            Updated parameter
        """
        # Initialize moments if first time
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)

        # Update biased first moment estimate
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad

        # Update biased second moment estimate
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad**2)

        # Bias correction (t must be incremented via step() before calling update)
        m_hat = self.m[param_name] / (1 - self.beta1**self.t)
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)

        # Update parameter
        param_updated = param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return param_updated


class DDQNNetwork:
    """
    Double Deep Q-Network with online and target networks.

    Architecture:
        Input (state_dim) → Hidden1 (128) → Hidden2 (64) → Output (n_actions)

    Features:
        - He initialization for ReLU
        - Adam optimizer with bias correction
        - Gradient clipping
        - L2 regularization
        - Soft target updates
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden1_size: int = 128,
        hidden2_size: int = 64,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        tau: float = 0.005,
        l2_weight: float = 0.0001,
        grad_clip_norm: float = 1.0,
        seed: int | None = None,
    ):
        """
        Initialize DDQN network.

        Args:
            state_dim: Input state dimension
            n_actions: Number of actions
            hidden1_size: First hidden layer size
            hidden2_size: Second hidden layer size
            learning_rate: Learning rate for Adam
            gamma: Discount factor
            tau: Soft update parameter (0=no update, 1=hard update)
            l2_weight: L2 regularization weight
            grad_clip_norm: Maximum gradient norm
            seed: Random seed for reproducibility (default: None for non-deterministic)
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.l2_weight = l2_weight
        self.grad_clip_norm = grad_clip_norm

        self.rng = np.random.default_rng(seed)

        # Initialize online network (He initialization for ReLU)
        self.w1 = self._he_init((state_dim, hidden1_size))
        self.b1 = np.zeros(hidden1_size)

        self.w2 = self._he_init((hidden1_size, hidden2_size))
        self.b2 = np.zeros(hidden2_size)

        self.w3 = self._he_init((hidden2_size, n_actions))
        self.b3 = np.zeros(n_actions)

        # Initialize target network (copy from online)
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()
        self.target_w3 = self.w3.copy()
        self.target_b3 = self.b3.copy()

        # Optimizer
        self.optimizer = AdamOptimizer(learning_rate=learning_rate)

        # Statistics
        self.training_steps = 0
        self.total_grad_norm = 0.0

        LOG.info(
            "[DDQN] Initialized: state_dim=%d, actions=%d, hidden=[%d,%d], lr=%.4f, tau=%.4f",
            state_dim,
            n_actions,
            hidden1_size,
            hidden2_size,
            learning_rate,
            tau,
        )

    def _he_init(self, shape: tuple[int, int]) -> np.ndarray:
        """
        He initialization for ReLU networks.

        Variance = 2 / fan_in
        """
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return self.rng.standard_normal(shape) * std

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative (1 if x > 0, else 0)."""
        return (x > 0).astype(float)

    def forward(self, state: np.ndarray, use_target: bool = False) -> tuple[np.ndarray, dict]:
        """
        Forward pass through network.

        Args:
            state: Input state (batch_size, state_dim) or (state_dim,)
            use_target: Use target network instead of online

        Returns:
            Tuple of (q_values, cache) where cache stores activations for backprop
        """
        # Handle single state
        single_input = False
        if state.ndim == 1:
            state = state.reshape(1, -1)
            single_input = True

        # Select weights
        if use_target:
            w1, b1 = self.target_w1, self.target_b1
            w2, b2 = self.target_w2, self.target_b2
            w3, b3 = self.target_w3, self.target_b3
        else:
            w1, b1 = self.w1, self.b1
            w2, b2 = self.w2, self.b2
            w3, b3 = self.w3, self.b3

        # Layer 1
        z1 = state @ w1 + b1
        a1 = self._relu(z1)

        # Layer 2
        z2 = a1 @ w2 + b2
        a2 = self._relu(z2)

        # Output layer (linear activation for Q-values)
        q_values = a2 @ w3 + b3

        # Cache for backpropagation
        cache = {"state": state, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "q_values": q_values}

        if single_input:
            q_values = q_values[0]

        return q_values, cache

    def predict(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """
        Get Q-values for state (no cache).

        Args:
            state: Input state
            use_target: Use target network

        Returns:
            Q-values for all actions
        """
        q_values, _ = self.forward(state, use_target=use_target)
        return q_values

    def train_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        weights: np.ndarray,
    ) -> dict:
        """
        Train on batch using Double DQN with importance sampling.

        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions (batch_size,)
            rewards: Batch of rewards (batch_size,)
            next_states: Batch of next states (batch_size, state_dim)
            dones: Batch of terminal flags (batch_size,)
            weights: Importance sampling weights (batch_size,)

        Returns:
            Dictionary with loss, TD errors, and statistics
        """
        batch_size = states.shape[0]

        # Forward pass (online network)
        q_values, cache = self.forward(states, use_target=False)

        # Double DQN: Online network selects actions for next states
        q_next_online, _ = self.forward(next_states, use_target=False)
        best_actions = np.argmax(q_next_online, axis=1)

        # Target network evaluates those actions
        q_next_target, _ = self.forward(next_states, use_target=True)
        q_next_max = q_next_target[np.arange(batch_size), best_actions]

        # TD targets
        td_targets = rewards + self.gamma * q_next_max * (1 - dones)

        # Current Q-values for taken actions
        q_current = q_values[np.arange(batch_size), actions]

        # TD errors
        td_errors = td_targets - q_current

        # Weighted MSE loss (importance sampling)
        loss = np.mean(weights * (td_errors**2))

        # Add L2 regularization
        l2_loss = self.l2_weight * (np.sum(self.w1**2) + np.sum(self.w2**2) + np.sum(self.w3**2))
        total_loss = loss + l2_loss

        # Backward pass
        self._backward(cache, actions, td_errors, weights, batch_size)

        # Soft update target network
        self._update_target_network()

        self.training_steps += 1

        return {
            "loss": float(loss),
            "l2_loss": float(l2_loss),
            "total_loss": float(total_loss),
            "mean_q": float(np.mean(q_values)),
            "mean_td_error": float(np.mean(np.abs(td_errors))),
            "max_td_error": float(np.max(np.abs(td_errors))),
            "grad_norm": float(self.total_grad_norm),
            "td_errors": td_errors,  # For priority updates
        }

    def _backward(
        self,
        cache: dict,
        actions: np.ndarray,
        td_errors: np.ndarray,
        weights: np.ndarray,
        batch_size: int,
    ):
        """
        Backward pass with gradient clipping.

        Args:
            cache: Activations from forward pass
            actions: Actions taken (batch_size,)
            td_errors: TD errors (batch_size,)
            weights: Importance sampling weights (batch_size,)
            batch_size: Batch size
        """
        # Gradient of loss w.r.t. Q-values
        # Only update Q-values for actions actually taken
        dq = np.zeros_like(cache["q_values"])
        dq[np.arange(batch_size), actions] = -2 * weights * td_errors / batch_size

        # Layer 3 gradients
        dw3 = cache["a2"].T @ dq
        db3 = np.sum(dq, axis=0)

        # Add L2 regularization gradient
        dw3 += 2 * self.l2_weight * self.w3

        # Backprop to layer 2
        da2 = dq @ self.w3.T
        dz2 = da2 * self._relu_derivative(cache["z2"])

        dw2 = cache["a1"].T @ dz2
        db2 = np.sum(dz2, axis=0)
        dw2 += 2 * self.l2_weight * self.w2

        # Backprop to layer 1
        da1 = dz2 @ self.w2.T
        dz1 = da1 * self._relu_derivative(cache["z1"])

        dw1 = cache["state"].T @ dz1
        db1 = np.sum(dz1, axis=0)
        dw1 += 2 * self.l2_weight * self.w1

        # Gradient clipping by norm
        gradients = [dw1, db1, dw2, db2, dw3, db3]
        self.total_grad_norm = self._clip_gradients(gradients)

        # Advance Adam timestep once per batch (not per parameter)
        self.optimizer.step()

        # Update weights with Adam
        self.w3 = self.optimizer.update("w3", self.w3, dw3)
        self.b3 = self.optimizer.update("b3", self.b3, db3)
        self.w2 = self.optimizer.update("w2", self.w2, dw2)
        self.b2 = self.optimizer.update("b2", self.b2, db2)
        self.w1 = self.optimizer.update("w1", self.w1, dw1)
        self.b1 = self.optimizer.update("b1", self.b1, db1)

    def _clip_gradients(self, gradients: list) -> float:
        """
        Clip gradients by global norm.

        Args:
            gradients: List of gradient arrays

        Returns:
            Total gradient norm before clipping
        """
        # Calculate global norm
        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))

        # Clip if necessary
        if total_norm > self.grad_clip_norm:
            scale = self.grad_clip_norm / (total_norm + 1e-8)
            for g in gradients:
                g *= scale

        return total_norm

    def _update_target_network(self):
        """
        Soft update of target network: θ_target ← τ*θ_online + (1-τ)*θ_target
        """
        self.target_w1 = self.tau * self.w1 + (1 - self.tau) * self.target_w1
        self.target_b1 = self.tau * self.b1 + (1 - self.tau) * self.target_b1

        self.target_w2 = self.tau * self.w2 + (1 - self.tau) * self.target_w2
        self.target_b2 = self.tau * self.b2 + (1 - self.tau) * self.target_b2

        self.target_w3 = self.tau * self.w3 + (1 - self.tau) * self.target_w3
        self.target_b3 = self.tau * self.b3 + (1 - self.tau) * self.target_b3

    def hard_update_target(self):
        """Copy online network to target network (τ=1.0 update)."""
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()
        self.target_w3 = self.w3.copy()
        self.target_b3 = self.b3.copy()
        LOG.info("[DDQN] Hard update: Copied online → target network")

    def save_weights(self, filepath: str):
        """
        Save network weights to file.

        Args:
            filepath: Path to save weights
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            filepath,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
            target_w1=self.target_w1,
            target_b1=self.target_b1,
            target_w2=self.target_w2,
            target_b2=self.target_b2,
            target_w3=self.target_w3,
            target_b3=self.target_b3,
            training_steps=self.training_steps,
        )
        LOG.info("[DDQN] Saved weights to %s (step %d)", filepath, self.training_steps)

    def load_weights(self, filepath: str):
        """
        Load network weights from file.

        Args:
            filepath: Path to load weights from
        """
        if not Path(filepath).exists():
            LOG.warning("[DDQN] Weight file not found: %s", filepath)
            return

        data = np.load(filepath)

        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w3 = data["w3"]
        self.b3 = data["b3"]

        self.target_w1 = data["target_w1"]
        self.target_b1 = data["target_b1"]
        self.target_w2 = data["target_w2"]
        self.target_b2 = data["target_b2"]
        self.target_w3 = data["target_w3"]
        self.target_b3 = data["target_b3"]

        if "training_steps" in data:
            self.training_steps = int(data["training_steps"])

        LOG.info("[DDQN] Loaded weights from %s (step %d)", filepath, self.training_steps)
