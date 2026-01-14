"""
Reward Integrity Monitor - Anti-Gaming Detection

GAP 4 FIX: Monitors correlation between rewards and actual P&L to detect reward hacking.

CRITICAL: This is especially important after reward calculation changes made on 2026-01-11:
- TriggerAgent: Switched from exit-quality to prediction-accuracy rewards
- HarvesterAgent: Switched from ad-hoc to principled capture-based rewards

Key Features:
- Tracks reward-P&L correlation
- Detects abnormal reward distributions
- Identifies reward component imbalance
- Alerts on potential gaming behaviors

Usage:
    monitor = RewardIntegrityMonitor(correlation_threshold=0.7)

    # After each trade
    monitor.add_trade(
        reward=shaped_reward["total_reward"],
        pnl=actual_pnl,
        reward_components=shaped_reward
    )

    # Periodic check
    status = monitor.check_integrity()
    if status["is_gaming"]:
        LOG.error("REWARD GAMING DETECTED: %s", status)
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)


@dataclass
class RewardPnLPair:
    """Single reward-P&L observation."""

    ts: str
    reward: float
    pnl: float
    components: dict  # Reward component breakdown
    trade_id: int


class RewardIntegrityMonitor:
    """
    Monitors correlation between rewards and actual P&L.

    Detects:
    - Reward-P&L decorrelation (agent gaming rewards)
    - Abnormal reward distributions (outliers)
    - Reward component imbalance (one component dominates)
    - Negative reward with positive P&L (or vice versa)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.7,
        min_samples: int = 50,
        max_history: int = 1000,
        outlier_std_threshold: float = 3.0,
    ):
        """
        Initialize reward integrity monitor.

        Args:
            correlation_threshold: Minimum acceptable correlation (0-1)
            min_samples: Minimum trades before checking correlation
            max_history: Maximum trades to keep in history
            outlier_std_threshold: Standard deviations for outlier detection
        """
        self.correlation_threshold = correlation_threshold
        self.min_samples = min_samples
        self.max_history = max_history
        self.outlier_std_threshold = outlier_std_threshold

        # History
        self.history: deque[RewardPnLPair] = deque(maxlen=max_history)

        # Quick access arrays (for correlation)
        self.rewards: deque[float] = deque(maxlen=max_history)
        self.pnls: deque[float] = deque(maxlen=max_history)

        # Component tracking
        self.component_sums: dict[str, float] = {}
        self.component_counts: dict[str, int] = {}

        # Anomaly tracking
        self.gaming_alerts: list[dict] = []
        self.last_check_result: dict | None = None

        LOG.info(
            "[REWARD_INTEGRITY] Initialized: corr_threshold=%.2f min_samples=%d",
            correlation_threshold,
            min_samples,
        )

    def add_trade(
        self,
        reward: float,
        pnl: float,
        reward_components: dict | None = None,
        trade_id: int | None = None,
    ):
        """
        Record reward and actual P&L for correlation analysis.

        Args:
            reward: Total reward given to agent
            pnl: Actual P&L from trade
            reward_components: Breakdown of reward (e.g., capture, WTL, etc.)
            trade_id: Optional trade identifier
        """
        # Validate inputs
        if not np.isfinite(reward):
            LOG.warning("[REWARD_INTEGRITY] Invalid reward: %s", reward)
            return

        if not np.isfinite(pnl):
            LOG.warning("[REWARD_INTEGRITY] Invalid P&L: %s", pnl)
            return

        # Create pair
        pair = RewardPnLPair(
            ts=datetime.now(timezone.utc).isoformat(),
            reward=reward,
            pnl=pnl,
            components=reward_components or {},
            trade_id=trade_id or len(self.history) + 1,
        )

        # Add to history
        self.history.append(pair)
        self.rewards.append(reward)
        self.pnls.append(pnl)

        # Track components
        if reward_components:
            for component, value in reward_components.items():
                if component != "total_reward":  # Skip total
                    self.component_sums[component] = self.component_sums.get(component, 0.0) + abs(value)
                    self.component_counts[component] = self.component_counts.get(component, 0) + 1

        # Check for obvious anomalies
        self._check_sign_mismatch(reward, pnl, trade_id)

    def check_integrity(self) -> dict:
        """
        Check reward integrity.

        Returns:
            Dictionary with:
            - status: "ok", "warning", "critical", or "insufficient_data"
            - correlation: Pearson correlation between reward and P&L
            - is_gaming: True if correlation below threshold
            - outliers: List of trade IDs with abnormal reward/P&L ratios
            - component_balance: Analysis of reward component contributions
            - sign_mismatches: Count of reward-P&L sign mismatches
        """
        if len(self.rewards) < self.min_samples:
            return {
                "status": "insufficient_data",
                "samples": len(self.rewards),
                "min_required": self.min_samples,
                "correlation": None,
                "is_gaming": False,
            }

        # Calculate correlation
        try:
            rewards_array = np.array(self.rewards)
            pnls_array = np.array(self.pnls)

            # Pearson correlation
            corr_matrix = np.corrcoef(rewards_array, pnls_array)
            correlation = float(corr_matrix[0, 1])

            # Check for NaN (can happen if all rewards/pnls are same)
            if not np.isfinite(correlation):
                correlation = 0.0

        except Exception as e:
            LOG.error("[REWARD_INTEGRITY] Correlation calculation failed: %s", e)
            correlation = 0.0

        # Detect outliers
        outliers = self._detect_outliers()

        # Analyze component balance
        component_balance = self._analyze_component_balance()

        # Count sign mismatches
        sign_mismatches = sum(1 for r, p in zip(self.rewards, self.pnls) if (r > 0 and p < 0) or (r < 0 and p > 0))

        # Determine status
        is_gaming = correlation < self.correlation_threshold
        has_many_outliers = len(outliers) > len(self.rewards) * 0.1  # >10% outliers
        has_many_mismatches = sign_mismatches > len(self.rewards) * 0.15  # >15% mismatches

        if is_gaming or has_many_outliers or has_many_mismatches:
            status = "critical" if is_gaming else "warning"

            # Log gaming alert
            alert = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "correlation": correlation,
                "outliers_count": len(outliers),
                "sign_mismatches": sign_mismatches,
            }
            self.gaming_alerts.append(alert)

            LOG.warning(
                "[REWARD_INTEGRITY] %s: corr=%.3f outliers=%d mismatches=%d",
                status.upper(),
                correlation,
                len(outliers),
                sign_mismatches,
            )
        else:
            status = "ok"

        result = {
            "status": status,
            "correlation": correlation,
            "is_gaming": is_gaming,
            "samples": len(self.rewards),
            "outliers": outliers,
            "outliers_count": len(outliers),
            "component_balance": component_balance,
            "sign_mismatches": sign_mismatches,
            "threshold": self.correlation_threshold,
        }

        self.last_check_result = result
        return result

    def _check_sign_mismatch(self, reward: float, pnl: float, trade_id: int | None):
        """Check for reward-P&L sign mismatch (immediate red flag)."""
        if (reward > 0 and pnl < 0) or (reward < 0 and pnl > 0):
            LOG.warning(
                "[REWARD_INTEGRITY] Sign mismatch: trade_id=%s reward=%+.4f pnl=%+.4f",
                trade_id,
                reward,
                pnl,
            )

    def _detect_outliers(self) -> list[int]:
        """
        Detect trades with abnormal reward/P&L ratios.

        Returns:
            List of trade IDs that are outliers
        """
        if len(self.history) < 10:
            return []

        outliers = []

        # Calculate reward/P&L ratios
        ratios = []
        for pair in self.history:
            if abs(pair.pnl) > 1e-6:  # Avoid division by zero
                ratio = pair.reward / pair.pnl
                ratios.append((pair.trade_id, ratio))

        if not ratios:
            return []

        # Detect outliers using z-score
        ratio_values = [r[1] for r in ratios]
        mean_ratio = np.mean(ratio_values)
        std_ratio = np.std(ratio_values)

        if std_ratio < 1e-6:  # All ratios same
            return []

        for trade_id, ratio in ratios:
            z_score = abs((ratio - mean_ratio) / std_ratio)
            if z_score > self.outlier_std_threshold:
                outliers.append(trade_id)

        return outliers

    def _analyze_component_balance(self) -> dict:
        """
        Analyze reward component contributions.

        Returns:
            Dictionary with component analysis
        """
        if not self.component_sums:
            return {"status": "no_components"}

        total_abs_sum = sum(self.component_sums.values())

        if total_abs_sum < 1e-6:
            return {"status": "zero_components"}

        # Calculate percentage contribution
        percentages = {comp: (abs_sum / total_abs_sum) * 100 for comp, abs_sum in self.component_sums.items()}

        # Check for dominance (one component >80%)
        max_component = max(percentages, key=percentages.get)
        max_percentage = percentages[max_component]

        is_dominated = max_percentage > 80.0

        if is_dominated:
            LOG.warning(
                "[REWARD_INTEGRITY] Component imbalance: %s dominates (%.1f%%)",
                max_component,
                max_percentage,
            )

        return {
            "status": "dominated" if is_dominated else "balanced",
            "percentages": percentages,
            "dominant_component": max_component,
            "dominant_percentage": max_percentage,
        }

    def get_statistics(self) -> dict:
        """Get comprehensive statistics."""
        if not self.rewards:
            return {"status": "no_data"}

        return {
            "samples": len(self.rewards),
            "mean_reward": float(np.mean(self.rewards)),
            "std_reward": float(np.std(self.rewards)),
            "mean_pnl": float(np.mean(self.pnls)),
            "std_pnl": float(np.std(self.pnls)),
            "last_check": self.last_check_result,
            "gaming_alerts": len(self.gaming_alerts),
        }

    def reset(self):
        """Reset monitor (e.g., after major parameter changes)."""
        self.history.clear()
        self.rewards.clear()
        self.pnls.clear()
        self.component_sums.clear()
        self.component_counts.clear()
        self.gaming_alerts.clear()
        self.last_check_result = None

        LOG.info("[REWARD_INTEGRITY] Reset")


# Self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    monitor = RewardIntegrityMonitor(correlation_threshold=0.7, min_samples=20)

    # Test 1: Good correlation (reward follows P&L)
    print("\n--- Test 1: Good Correlation ---")
    for i in range(50):
        pnl = np.random.randn() * 10  # Random P&L
        reward = pnl + np.random.randn() * 2  # Reward correlated with P&L

        monitor.add_trade(
            reward=reward,
            pnl=pnl,
            reward_components={"capture": reward * 0.6, "wtl": reward * 0.3, "activity": reward * 0.1},
        )

    status = monitor.check_integrity()
    print(f"Status: {status['status']}")
    print(f"Correlation: {status['correlation']:.3f}")
    print(f"Is Gaming: {status['is_gaming']}")

    # Test 2: Poor correlation (reward gaming)
    print("\n--- Test 2: Poor Correlation (Gaming) ---")
    monitor.reset()

    for i in range(50):
        pnl = np.random.randn() * 10
        reward = np.random.randn() * 5  # Reward UNCORRELATED with P&L

        monitor.add_trade(reward=reward, pnl=pnl)

    status = monitor.check_integrity()
    print(f"Status: {status['status']}")
    print(f"Correlation: {status['correlation']:.3f}")
    print(f"Is Gaming: {status['is_gaming']}")

    # Test 3: Sign mismatches
    print("\n--- Test 3: Sign Mismatches ---")
    monitor.reset()

    for i in range(30):
        pnl = 10.0  # Positive P&L
        reward = -5.0  # Negative reward (mismatch!)

        monitor.add_trade(reward=reward, pnl=pnl)

    status = monitor.check_integrity()
    print(f"Status: {status['status']}")
    print(f"Sign Mismatches: {status['sign_mismatches']}")

    print("\n✓ RewardIntegrityMonitor self-test complete")
