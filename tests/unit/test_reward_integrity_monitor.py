"""Tests for src.core.reward_integrity_monitor."""

import pytest
import numpy as np

from src.core.reward_integrity_monitor import RewardIntegrityMonitor, RewardPnLPair


# ---------------------------------------------------------------------------
# RewardPnLPair dataclass
# ---------------------------------------------------------------------------
class TestRewardPnLPair:
    def test_fields(self):
        p = RewardPnLPair(ts="2025-01-01T00:00:00Z", reward=1.0, pnl=2.0,
                          components={"a": 0.5}, trade_id=1)
        assert p.reward == pytest.approx(1.0)
        assert p.pnl == pytest.approx(2.0)
        assert p.trade_id == 1


# ---------------------------------------------------------------------------
# RewardIntegrityMonitor: init
# ---------------------------------------------------------------------------
class TestInit:
    def test_default_params(self):
        m = RewardIntegrityMonitor()
        assert m.correlation_threshold == pytest.approx(0.7)
        assert m.min_samples == 50
        assert m.max_history == 1000
        assert m.outlier_std_threshold == pytest.approx(3.0)

    def test_custom_params(self):
        m = RewardIntegrityMonitor(
            correlation_threshold=0.5, min_samples=10,
            max_history=100, outlier_std_threshold=2.0,
        )
        assert m.correlation_threshold == pytest.approx(0.5)
        assert m.min_samples == 10

    def test_initial_state_empty(self):
        m = RewardIntegrityMonitor()
        assert len(m.history) == 0
        assert len(m.rewards) == 0
        assert len(m.pnls) == 0
        assert m.last_check_result is None
        assert m.gaming_alerts == []


# ---------------------------------------------------------------------------
# add_trade()
# ---------------------------------------------------------------------------
class TestAddTrade:
    def test_basic_add(self):
        m = RewardIntegrityMonitor()
        m.add_trade(reward=1.0, pnl=2.0)
        assert len(m.history) == 1
        assert len(m.rewards) == 1
        assert len(m.pnls) == 1

    def test_auto_trade_id(self):
        m = RewardIntegrityMonitor()
        m.add_trade(reward=1.0, pnl=2.0)
        assert m.history[0].trade_id == 1

    def test_explicit_trade_id(self):
        m = RewardIntegrityMonitor()
        m.add_trade(reward=1.0, pnl=2.0, trade_id=42)
        assert m.history[0].trade_id == 42

    def test_component_tracking(self):
        m = RewardIntegrityMonitor()
        m.add_trade(reward=1.0, pnl=2.0,
                     reward_components={"capture": 0.6, "wtl": 0.3, "total_reward": 1.0})
        # total_reward should be skipped
        assert "capture" in m.component_sums
        assert "wtl" in m.component_sums
        assert "total_reward" not in m.component_sums

    def test_invalid_reward_skipped(self):
        m = RewardIntegrityMonitor()
        m.add_trade(reward=float("nan"), pnl=2.0)
        assert len(m.history) == 0

    def test_invalid_pnl_skipped(self):
        m = RewardIntegrityMonitor()
        m.add_trade(reward=1.0, pnl=float("inf"))
        assert len(m.history) == 0

    def test_max_history_enforced(self):
        m = RewardIntegrityMonitor(max_history=10)
        for i in range(20):
            m.add_trade(reward=float(i), pnl=float(i))
        assert len(m.history) == 10
        assert len(m.rewards) == 10


# ---------------------------------------------------------------------------
# check_integrity()
# ---------------------------------------------------------------------------
class TestCheckIntegrity:
    def test_insufficient_data(self):
        m = RewardIntegrityMonitor(min_samples=50)
        for i in range(10):
            m.add_trade(reward=float(i), pnl=float(i))
        result = m.check_integrity()
        assert result["status"] == "insufficient_data"
        assert result["is_gaming"] is False

    def test_good_correlation(self):
        m = RewardIntegrityMonitor(min_samples=20, correlation_threshold=0.5)
        rng = np.random.default_rng(42)
        for _ in range(30):
            pnl = rng.standard_normal() * 10
            reward = pnl + rng.standard_normal() * 0.5  # highly correlated
            m.add_trade(reward=reward, pnl=pnl)
        result = m.check_integrity()
        assert result["status"] == "ok"
        assert result["is_gaming"] is False
        assert result["correlation"] > 0.5

    def test_poor_correlation_detected(self):
        m = RewardIntegrityMonitor(min_samples=20, correlation_threshold=0.7)
        rng = np.random.default_rng(42)
        for _ in range(30):
            pnl = rng.standard_normal() * 10
            reward = rng.standard_normal() * 5  # uncorrelated
            m.add_trade(reward=reward, pnl=pnl)
        result = m.check_integrity()
        assert result["is_gaming"] is True
        assert result["status"] in ("critical", "warning")

    def test_sign_mismatches_counted(self):
        m = RewardIntegrityMonitor(min_samples=10)
        # All sign mismatches
        for _ in range(15):
            m.add_trade(reward=5.0, pnl=-3.0)
        result = m.check_integrity()
        assert result["sign_mismatches"] == 15

    def test_constant_values_handled(self):
        """All same reward/pnl → std=0 → correlation should be 0."""
        m = RewardIntegrityMonitor(min_samples=10, correlation_threshold=0.5)
        for _ in range(15):
            m.add_trade(reward=1.0, pnl=1.0)
        result = m.check_integrity()
        # NaN correlation → set to 0.0, which is below threshold
        assert result["correlation"] == pytest.approx(0.0)

    def test_gaming_alert_logged(self):
        m = RewardIntegrityMonitor(min_samples=10, correlation_threshold=0.9)
        rng = np.random.default_rng(42)
        for _ in range(20):
            m.add_trade(reward=rng.standard_normal(), pnl=rng.standard_normal())
        m.check_integrity()
        assert len(m.gaming_alerts) >= 1

    def test_last_check_result_updated(self):
        m = RewardIntegrityMonitor(min_samples=5)
        for i in range(10):
            m.add_trade(reward=float(i), pnl=float(i))
        m.check_integrity()
        assert m.last_check_result is not None
        assert "correlation" in m.last_check_result


# ---------------------------------------------------------------------------
# _detect_outliers()
# ---------------------------------------------------------------------------
class TestDetectOutliers:
    def test_no_outliers_with_few_samples(self):
        m = RewardIntegrityMonitor()
        for i in range(5):
            m.add_trade(reward=float(i), pnl=float(i + 1))
        assert m._detect_outliers() == []

    def test_outlier_detected(self):
        m = RewardIntegrityMonitor(outlier_std_threshold=2.0)
        # Normal cluster
        for _ in range(20):
            m.add_trade(reward=10.0, pnl=10.0, trade_id=0)
        # One extreme outlier
        m.add_trade(reward=1000.0, pnl=1.0, trade_id=999)
        outliers = m._detect_outliers()
        assert 999 in outliers

    def test_no_outliers_uniform_ratios(self):
        m = RewardIntegrityMonitor(outlier_std_threshold=3.0)
        for i in range(20):
            m.add_trade(reward=float(i + 1) * 2, pnl=float(i + 1))
        # All ratios are 2.0, no outlier
        assert m._detect_outliers() == []

    def test_zero_pnl_trades_skipped_in_ratio(self):
        m = RewardIntegrityMonitor()
        for _ in range(15):
            m.add_trade(reward=1.0, pnl=0.0)  # pnl=0 → skipped
        assert m._detect_outliers() == []

    def test_all_same_ratio_no_outlier(self):
        m = RewardIntegrityMonitor()
        for i in range(15):
            m.add_trade(reward=5.0, pnl=1.0, trade_id=i)
        # std = 0 → early return
        assert m._detect_outliers() == []


# ---------------------------------------------------------------------------
# _analyze_component_balance()
# ---------------------------------------------------------------------------
class TestComponentBalance:
    def test_no_components(self):
        m = RewardIntegrityMonitor()
        result = m._analyze_component_balance()
        assert result["status"] == "no_components"

    def test_balanced_components(self):
        m = RewardIntegrityMonitor()
        for _ in range(10):
            m.add_trade(reward=1.0, pnl=1.0,
                         reward_components={"a": 0.4, "b": 0.3, "c": 0.3})
        result = m._analyze_component_balance()
        assert result["status"] == "balanced"
        assert "percentages" in result

    def test_dominated_component(self):
        m = RewardIntegrityMonitor()
        for _ in range(10):
            m.add_trade(reward=1.0, pnl=1.0,
                         reward_components={"big": 10.0, "small": 0.1})
        result = m._analyze_component_balance()
        assert result["status"] == "dominated"
        assert result["dominant_component"] == "big"
        assert result["dominant_percentage"] > 80

    def test_zero_components(self):
        m = RewardIntegrityMonitor()
        m.component_sums = {"a": 0.0, "b": 0.0}
        result = m._analyze_component_balance()
        assert result["status"] == "zero_components"


# ---------------------------------------------------------------------------
# get_statistics()
# ---------------------------------------------------------------------------
class TestGetStatistics:
    def test_no_data(self):
        m = RewardIntegrityMonitor()
        assert m.get_statistics() == {"status": "no_data"}

    def test_with_data(self):
        m = RewardIntegrityMonitor()
        for i in range(5):
            m.add_trade(reward=float(i), pnl=float(i + 1))
        stats = m.get_statistics()
        assert stats["samples"] == 5
        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert "mean_pnl" in stats
        assert "gaming_alerts" in stats


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------
class TestReset:
    def test_reset_clears_all(self):
        m = RewardIntegrityMonitor()
        for i in range(10):
            m.add_trade(reward=float(i), pnl=float(i),
                         reward_components={"a": float(i)})
        m.gaming_alerts.append({"test": True})
        m.last_check_result = {"status": "ok"}

        m.reset()

        assert len(m.history) == 0
        assert len(m.rewards) == 0
        assert len(m.pnls) == 0
        assert len(m.component_sums) == 0
        assert len(m.component_counts) == 0
        assert len(m.gaming_alerts) == 0
        assert m.last_check_result is None


# ---------------------------------------------------------------------------
# _check_sign_mismatch() (implicit via add_trade)
# ---------------------------------------------------------------------------
class TestSignMismatch:
    def test_positive_reward_negative_pnl(self):
        """Sign mismatch is logged but trade is still added."""
        m = RewardIntegrityMonitor()
        m.add_trade(reward=5.0, pnl=-3.0)
        assert len(m.history) == 1

    def test_negative_reward_positive_pnl(self):
        m = RewardIntegrityMonitor()
        m.add_trade(reward=-5.0, pnl=3.0)
        assert len(m.history) == 1

    def test_same_sign_no_mismatch_warning(self):
        m = RewardIntegrityMonitor()
        m.add_trade(reward=5.0, pnl=3.0)
        assert len(m.history) == 1

    def test_zero_values_no_mismatch(self):
        m = RewardIntegrityMonitor()
        m.add_trade(reward=0.0, pnl=0.0)
        assert len(m.history) == 1
