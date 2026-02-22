"""
Agent Arena - Multi-Agent Ensemble with Consensus

Coordinates multiple DDQN agents for entry and exit decisions.
Implements consensus mechanisms and dynamic allocation.

Architecture:
- Multiple Trigger agents (entry specialists)
- Multiple Harvester agents (exit specialists)
- Consensus modes: weighted_average, voting, max_confidence, min_risk
- Performance-based dynamic allocation
"""

import logging
import traceback
from enum import Enum

import numpy as np

# Consensus constants
MIN_CONFIDENCE_THRESHOLD: float = 0.6
DEFAULT_CONFIDENCE: float = 0.5

LOG = logging.getLogger(__name__)


class ConsensusMode(Enum):
    """Consensus calculation methods."""

    WEIGHTED_AVERAGE = "weighted_average"  # Weight by recent performance
    VOTING = "voting"  # Majority vote
    MAX_CONFIDENCE = "max_confidence"  # Highest Q-value spread
    MIN_RISK = "min_risk"  # Most conservative (prefer no-action)


class AgentStats:
    """Track performance statistics for each agent."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.recent_rewards = []
        self.recent_accuracies = []
        self.total_decisions = 0
        self.correct_decisions = 0
        self.total_reward = 0.0

    def update(self, reward: float, was_correct: bool):
        """Update statistics with new outcome."""
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)

        self.recent_accuracies.append(1.0 if was_correct else 0.0)
        if len(self.recent_accuracies) > self.window_size:
            self.recent_accuracies.pop(0)

        self.total_decisions += 1
        if was_correct:
            self.correct_decisions += 1
        self.total_reward += reward

    def get_avg_reward(self) -> float:
        """Get recent average reward."""
        if not self.recent_rewards:
            return 0.0
        return np.mean(self.recent_rewards)

    def get_accuracy(self) -> float:
        """Get recent accuracy."""
        if not self.recent_accuracies:
            return 0.5
        return np.mean(self.recent_accuracies)

    def get_weight(self) -> float:
        """
        Calculate agent weight for consensus.

        Combines reward and accuracy with softmax normalization.
        """
        avg_reward = self.get_avg_reward()
        accuracy = self.get_accuracy()

        # Combine metrics (reward weighted more heavily)
        score = 0.7 * avg_reward + 0.3 * (accuracy - 0.5)  # Center accuracy at 0

        return score


class AgentArena:
    """
    Multi-agent ensemble coordinator.

    Manages multiple agents for entry and exit decisions,
    combining their outputs through consensus mechanisms.
    """

    def __init__(
        self,
        trigger_agents: list,  # List[TriggerAgent]
        harvester_agents: list,  # List[HarvesterAgent]
        consensus_mode: ConsensusMode = ConsensusMode.WEIGHTED_AVERAGE,
        min_agreement: float = 0.6,
        param_manager=None,  # LearnedParametersManager instance
    ):
        """
        Initialize agent arena.

        Args:
            trigger_agents: List of TriggerAgent instances
            harvester_agents: List of HarvesterAgent instances
            consensus_mode: Method for combining agent outputs
            min_agreement: Minimum agreement threshold (0-1)
            param_manager: LearnedParametersManager for adaptive thresholds (None = use default)
        """
        self.trigger_agents = trigger_agents
        self.harvester_agents = harvester_agents
        self.consensus_mode = consensus_mode
        self.min_agreement = min_agreement

        # Load confidence threshold from param_manager if available
        if param_manager is not None and len(trigger_agents) > 0:
            # Get symbol from first trigger agent (all agents should be for same symbol)
            agent_symbol = getattr(trigger_agents[0], "symbol", "XAUUSD")
            agent_timeframe = getattr(trigger_agents[0], "timeframe", "M15")
            agent_broker = getattr(trigger_agents[0], "broker", "default")
            self.min_confidence_threshold = param_manager.get(
                agent_symbol,
                "entry_confidence_threshold",
                timeframe=agent_timeframe,
                broker=agent_broker,
                default=MIN_CONFIDENCE_THRESHOLD,
            )
        else:
            self.min_confidence_threshold = MIN_CONFIDENCE_THRESHOLD

        # Performance tracking
        self.trigger_stats = [AgentStats() for _ in trigger_agents]
        self.harvester_stats = [AgentStats() for _ in harvester_agents]

        # Diversity tracking
        self.last_trigger_actions = []
        self.last_harvester_actions = []

        # Agreement tracking (for HUD export)
        self.last_entry_agreement = 0.0
        self.last_exit_agreement = 0.0

        LOG.info(
            "[ARENA] Initialized with %d trigger agents, %d harvester agents, mode=%s",
            len(trigger_agents),
            len(harvester_agents),
            consensus_mode.value,
        )

    def get_entry_signal(
        self, bars: list, imbalance: float, vpin_z: float, depth_ratio: float, realized_vol: float
    ) -> tuple[int, float, float, float]:
        """
        Get consensus entry signal from trigger agents.

        Args:
            bars: Recent OHLCV bars
            imbalance: Order book imbalance
            vpin_z: VPIN z-score
            depth_ratio: Order book depth ratio
            realized_vol: Rogers-Satchell volatility

        Returns:
            Tuple of (action, confidence, predicted_runway, agreement_score)
        """
        if len(self.trigger_agents) == 0:
            return 0, 0.0, 0.5, 0.0

        # Gather decisions from all agents
        actions = []
        confidences = []
        runways = []

        for i, agent in enumerate(self.trigger_agents):
            try:
                action, confidence, runway = agent.decide(bars, imbalance, vpin_z, depth_ratio, realized_vol)
                actions.append(action)
                confidences.append(confidence)
                runways.append(runway)
            except Exception as e:
                LOG.error(
                    "[ARENA-ENTRY] Trigger agent %d failed: %s\n%s",
                    i,
                    str(e),
                    traceback.format_exc(),
                )
                # Use default values for failed agent
                actions.append(0)
                confidences.append(0.0)
                runways.append(0.5)

        # Store for diversity analysis
        self.last_trigger_actions = actions

        # Calculate consensus
        consensus_action, consensus_confidence, agreement = self._calculate_consensus(
            actions, confidences, self.trigger_stats, is_entry=True
        )

        # Average runway (independent of consensus mechanism)
        avg_runway = np.mean(runways)

        # Store agreement for HUD export
        self.last_entry_agreement = agreement

        LOG.debug(
            "[ARENA-ENTRY] Actions=%s, Consensus=%d, Confidence=%.3f, Agreement=%.3f",
            actions,
            consensus_action,
            consensus_confidence,
            agreement,
        )

        return consensus_action, consensus_confidence, avg_runway, agreement

    def get_exit_signal(
        self, bars: list, current_price: float, imbalance: float, vpin_z: float, depth_ratio: float
    ) -> tuple[int, float, float]:
        """
        Get consensus exit signal from harvester agents.

        Args:
            bars: Recent OHLCV bars
            current_price: Current market price
            imbalance: Order book imbalance
            vpin_z: VPIN z-score
            depth_ratio: Order book depth ratio

        Returns:
            Tuple of (action, confidence, agreement_score)
        """
        if len(self.harvester_agents) == 0:
            return 0, 0.0, 0.0

        # Gather decisions from all agents
        actions = []
        confidences = []

        for i, agent in enumerate(self.harvester_agents):
            try:
                action, confidence = agent.decide(bars, current_price, imbalance, vpin_z, depth_ratio)
                actions.append(action)
                confidences.append(confidence)
            except Exception as e:
                LOG.error(
                    "[ARENA-EXIT] Harvester agent %d failed: %s\n%s",
                    i,
                    str(e),
                    traceback.format_exc(),
                )
                # Use default values for failed agent
                actions.append(0)
                confidences.append(0.0)

        # Store for diversity analysis
        self.last_harvester_actions = actions

        # Calculate consensus
        consensus_action, consensus_confidence, agreement = self._calculate_consensus(
            actions, confidences, self.harvester_stats, is_entry=False
        )

        # Store agreement for HUD export
        self.last_exit_agreement = agreement

        LOG.debug(
            "[ARENA-EXIT] Actions=%s, Consensus=%d, Confidence=%.3f, Agreement=%.3f",
            actions,
            consensus_action,
            consensus_confidence,
            agreement,
        )

        return consensus_action, consensus_confidence, agreement

    def _calculate_consensus(
        self, actions: list[int], confidences: list[float], stats: list[AgentStats], is_entry: bool
    ) -> tuple[int, float, float]:
        """
        Calculate consensus action from multiple agent decisions.

        Args:
            actions: List of actions from agents
            confidences: List of confidence scores
            stats: List of AgentStats for weighting
            is_entry: True if entry decision, False if exit

        Returns:
            Tuple of (consensus_action, consensus_confidence, agreement_score)
        """
        if len(actions) == 0:
            return 0, 0.0, 0.0

        if self.consensus_mode == ConsensusMode.WEIGHTED_AVERAGE:
            return self._weighted_average_consensus(actions, confidences, stats, is_entry)

        elif self.consensus_mode == ConsensusMode.VOTING:
            return self._voting_consensus(actions, confidences)

        elif self.consensus_mode == ConsensusMode.MAX_CONFIDENCE:
            return self._max_confidence_consensus(actions, confidences)

        elif self.consensus_mode == ConsensusMode.MIN_RISK:
            return self._min_risk_consensus(actions, confidences)

        else:
            # Fallback to voting
            return self._voting_consensus(actions, confidences)

    def _weighted_average_consensus(
        self, actions: list[int], confidences: list[float], stats: list[AgentStats], is_entry: bool
    ) -> tuple[int, float, float]:
        """
        Weight agents by recent performance.

        Uses softmax over agent weights to avoid negative values.
        """
        # Calculate weights from performance
        raw_weights = np.array([s.get_weight() for s in stats])

        # Softmax normalization
        exp_weights = np.exp(raw_weights - np.max(raw_weights))  # Numerical stability
        weights = exp_weights / np.sum(exp_weights)

        # For entry: 3 actions (0=NONE, 1=LONG, 2=SHORT)
        # For exit: 2 actions (0=HOLD, 1=CLOSE)
        n_actions = 3 if is_entry else 2

        # Calculate weighted vote for each action
        action_scores = np.zeros(n_actions)
        for action, confidence, weight in zip(actions, confidences, weights, strict=False):
            if 0 <= action < n_actions:
                action_scores[action] += weight * confidence

        # Select action with highest weighted score
        consensus_action = int(np.argmax(action_scores))
        consensus_confidence = action_scores[consensus_action]

        # Agreement: ratio of vote for winner vs total votes
        total_votes = np.sum(action_scores)
        agreement = action_scores[consensus_action] / total_votes if total_votes > 0 else 0.0

        return consensus_action, consensus_confidence, agreement

    def _voting_consensus(self, actions: list[int], confidences: list[float]) -> tuple[int, float, float]:
        """
        Simple majority voting (1 agent = 1 vote).
        """
        # Count votes for each action
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        # Find action with most votes
        consensus_action = max(action_counts, key=action_counts.get)
        votes_for_winner = action_counts[consensus_action]
        total_votes = len(actions)

        # Average confidence of agents who voted for winner
        winning_confidences = [conf for act, conf in zip(actions, confidences, strict=False) if act == consensus_action]
        consensus_confidence = np.mean(winning_confidences) if winning_confidences else 0.0

        # Agreement: percentage of agents agreeing
        agreement = votes_for_winner / total_votes

        return consensus_action, consensus_confidence, agreement

    def _max_confidence_consensus(self, actions: list[int], confidences: list[float]) -> tuple[int, float, float]:
        """
        Select action from most confident agent.
        """
        max_idx = int(np.argmax(confidences))
        consensus_action = actions[max_idx]
        consensus_confidence = confidences[max_idx]

        # Agreement: how many other agents chose same action
        agreements = sum(1 for a in actions if a == consensus_action)
        agreement = agreements / len(actions)

        return consensus_action, consensus_confidence, agreement

    def _min_risk_consensus(self, actions: list[int], confidences: list[float]) -> tuple[int, float, float]:
        """
        Conservative consensus: prefer no-action unless strong agreement.

        For entry: Default to 0 (NONE) unless agents strongly agree on LONG/SHORT
        For exit: Default to 0 (HOLD) unless agents strongly agree on CLOSE
        """
        # Default to no-action
        no_action = 0

        # Count votes for non-default actions
        action_votes = {}
        for action, confidence in zip(actions, confidences, strict=False):
            if action != no_action:
                if action not in action_votes:
                    action_votes[action] = {"count": 0, "total_conf": 0.0}
                action_votes[action]["count"] += 1
                action_votes[action]["total_conf"] += confidence

        # Check if any action has sufficient agreement
        total_agents = len(actions)
        for action, stats in action_votes.items():
            vote_ratio = stats["count"] / total_agents
            avg_confidence = stats["total_conf"] / stats["count"]

            # Require both majority vote AND high confidence
            if vote_ratio >= self.min_agreement and avg_confidence >= self.min_confidence_threshold:
                return action, avg_confidence, vote_ratio

        # No strong consensus - return no-action
        no_action_count = sum(1 for a in actions if a == no_action)
        agreement = no_action_count / total_agents

        return no_action, DEFAULT_CONFIDENCE, agreement

    def update_trigger_performance(self, agent_idx: int, reward: float, was_correct: bool):
        """
        Update trigger agent performance statistics.

        Args:
            agent_idx: Index of agent to update
            reward: Reward received
            was_correct: Whether decision was correct
        """
        if 0 <= agent_idx < len(self.trigger_stats):
            self.trigger_stats[agent_idx].update(reward, was_correct)

    def update_harvester_performance(self, agent_idx: int, reward: float, was_correct: bool):
        """
        Update harvester agent performance statistics.

        Args:
            agent_idx: Index of agent to update
            reward: Reward received
            was_correct: Whether decision was correct
        """
        if 0 <= agent_idx < len(self.harvester_stats):
            self.harvester_stats[agent_idx].update(reward, was_correct)

    def get_diversity_score(self) -> dict[str, float]:
        """
        Calculate agent diversity scores.

        High diversity = agents disagree (exploring different strategies)
        Low diversity = agents converge (potentially overfitting)

        Returns:
            Dictionary with trigger_diversity and harvester_diversity
        """
        trigger_diversity = 0.0
        if len(self.last_trigger_actions) > 1:
            unique_actions = len(set(self.last_trigger_actions))
            trigger_diversity = unique_actions / len(self.last_trigger_actions)

        harvester_diversity = 0.0
        if len(self.last_harvester_actions) > 1:
            unique_actions = len(set(self.last_harvester_actions))
            harvester_diversity = unique_actions / len(self.last_harvester_actions)

        return {"trigger_diversity": trigger_diversity, "harvester_diversity": harvester_diversity}

    def get_stats(self) -> dict:
        """
        Get arena statistics.

        Returns:
            Dictionary with performance metrics for all agents
        """
        trigger_stats = []
        for i, stats in enumerate(self.trigger_stats):
            trigger_stats.append(
                {
                    "agent_id": i,
                    "avg_reward": stats.get_avg_reward(),
                    "accuracy": stats.get_accuracy(),
                    "total_decisions": stats.total_decisions,
                    "weight": stats.get_weight(),
                }
            )

        harvester_stats = []
        for i, stats in enumerate(self.harvester_stats):
            harvester_stats.append(
                {
                    "agent_id": i,
                    "avg_reward": stats.get_avg_reward(),
                    "accuracy": stats.get_accuracy(),
                    "total_decisions": stats.total_decisions,
                    "weight": stats.get_weight(),
                }
            )

        diversity = self.get_diversity_score()

        return {
            "trigger_agents": trigger_stats,
            "harvester_agents": harvester_stats,
            "diversity": diversity,
            "consensus_mode": self.consensus_mode.value,
        }

    def train_all_agents(self, batch_size: int = 64) -> dict[str, list[float]]:
        """
        Train all agents on their experience buffers.

        Args:
            batch_size: Training batch size

        Returns:
            Dictionary with trigger_losses and harvester_losses
        """
        trigger_losses = []
        for i, agent in enumerate(self.trigger_agents):
            if hasattr(agent, "train") and len(agent.buffer) >= batch_size:
                loss = agent.train(batch_size)
                trigger_losses.append(loss)
                LOG.debug("[ARENA] Trained trigger agent %d, loss=%.4f", i, loss)

        harvester_losses = []
        for i, agent in enumerate(self.harvester_agents):
            if hasattr(agent, "train") and len(agent.buffer) >= batch_size:
                loss = agent.train(batch_size)
                harvester_losses.append(loss)
                LOG.debug("[ARENA] Trained harvester agent %d, loss=%.4f", i, loss)

        return {"trigger_losses": trigger_losses, "harvester_losses": harvester_losses}

    def save_weights(self, directory: str):
        """
        Save weights for all agents.

        Args:
            directory: Directory to save weights
        """
        from pathlib import Path  # noqa: PLC0415

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        for i, agent in enumerate(self.trigger_agents):
            if hasattr(agent, "save_weights"):
                agent.save_weights(str(path / f"trigger_agent_{i}.npz"))

        for i, agent in enumerate(self.harvester_agents):
            if hasattr(agent, "save_weights"):
                agent.save_weights(str(path / f"harvester_agent_{i}.npz"))

        LOG.info("[ARENA] Saved weights to %s", directory)

    def load_weights(self, directory: str):
        """
        Load weights for all agents.

        Args:
            directory: Directory containing saved weights
        """
        from pathlib import Path  # noqa: PLC0415

        path = Path(directory)

        if not path.exists():
            LOG.warning("[ARENA] Weight directory %s does not exist", directory)
            return

        for i, agent in enumerate(self.trigger_agents):
            weight_file = path / f"trigger_agent_{i}.npz"
            if weight_file.exists() and hasattr(agent, "load_weights"):
                agent.load_weights(str(weight_file))
                LOG.info("[ARENA] Loaded trigger agent %d weights", i)

        for i, agent in enumerate(self.harvester_agents):
            weight_file = path / f"harvester_agent_{i}.npz"
            if weight_file.exists() and hasattr(agent, "load_weights"):
                agent.load_weights(str(weight_file))
                LOG.info("[ARENA] Loaded harvester agent %d weights", i)
