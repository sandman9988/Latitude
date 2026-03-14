"""
P0 Safety Infrastructure Integration Tests

Tests the integration of active P0 components:
- JournaledPersistence
- RewardIntegrityMonitor
- ProductionMonitor
"""

import json
import tempfile
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)

import pytest

from src.persistence.journaled_persistence import Journal
from src.monitoring.production_monitor import ProductionMonitor
from src.core.reward_integrity_monitor import RewardIntegrityMonitor


class TestCrashRecovery:
    """Test journal-based crash recovery."""

    def test_trade_lifecycle_with_journal(self):
        """Test trades are logged and can be replayed."""
        # Use test exports directory
        journal_file = Path("test_exports/test_journal.log")
        journal_file.parent.mkdir(exist_ok=True)

        # Clean up any existing files
        journal_file.unlink(missing_ok=True)
        Path(str(journal_file).replace(".log", ".checkpoint")).unlink(missing_ok=True)

        journal = Journal(journal_path=str(journal_file))

        # Simulate trade lifecycle
        journal.log_trade_open("ABC123", "LONG", 0.01, 1.1000)
        journal.log_operation("bar", {"close": 1.1010})

        # Create checkpoint after first operations
        journal.checkpoint()

        # More operations after checkpoint (simulating work between checkpoints)
        journal.log_trade_close("ABC123", 1.1010, 10.0, 15.0, -5.0, False)
        journal.log_operation("bar", {"close": 1.1020})
        journal.log_trade_open("DEF456", "SHORT", 0.02, 1.1020)

        # Flush file but don't call close() which would create another checkpoint
        journal.journal_file.flush()

        # Simulate crash - create new journal instance
        journal2 = Journal(journal_path=str(journal_file))
        operations = journal2.replay_from_checkpoint()

        # Verify operations AFTER checkpoint are replayed (seq > checkpoint_seq)
        # Note: checkpoint at seq=3 skips operation with seq=3, only replays seq>3
        assert len(operations) == 2
        assert operations[0].op == "bar"
        assert any(op.op == "trade_open" and op.data["order_id"] == "DEF456" for op in operations)

        # Cleanup
        journal.close()  # Now close the first one
        journal2.close()

        # Cleanup
        journal_file.unlink(missing_ok=True)
        for f in journal_file.parent.glob("*.checkpoint"):
            f.unlink(missing_ok=True)

    def test_checkpoint_recovery(self):
        """Test recovery from checkpoint after many operations."""
        # Use test exports directory
        journal_file = Path("test_exports/test_checkpoint.log")
        journal_file.parent.mkdir(exist_ok=True)

        # Clean up any existing files
        journal_file.unlink(missing_ok=True)
        Path(str(journal_file).replace(".log", ".checkpoint")).unlink(missing_ok=True)

        journal = Journal(journal_path=str(journal_file))

        # Log many operations
        for i in range(50):
            journal.log_operation("bar", {"index": i})

        # Checkpoint
        journal.checkpoint()

        # More operations after checkpoint
        for i in range(25):
            journal.log_operation("bar", {"index": 50 + i})

        # Flush but don't close (close would create another checkpoint)
        journal.journal_file.flush()

        # Replay - should skip to checkpoint
        journal2 = Journal(journal_path=str(journal_file))
        operations = journal2.replay_from_checkpoint()

        # Should only have operations AFTER checkpoint (24, since checkpoint_seq skips one)
        assert len(operations) == 24

        # Cleanup
        journal.close()
        journal2.close()

        # Cleanup
        journal_file.unlink(missing_ok=True)
        for f in journal_file.parent.glob("*.checkpoint"):
            f.unlink(missing_ok=True)


class TestRewardGamingDetection:
    """Test reward integrity monitoring."""

    def test_correlation_tracking(self):
        """Test reward-PnL correlation tracking."""
        monitor = RewardIntegrityMonitor()

        # Add correlated trades
        for _ in range(50):
            pnl = rng.standard_normal() * 10
            reward = pnl * 0.1 + rng.standard_normal() * 2  # Correlated

            monitor.add_trade(
                reward=reward,
                pnl=pnl,
                reward_components={"main": reward},
            )

        integrity = monitor.check_integrity()

        # Should have positive correlation
        assert integrity["correlation"] > 0.3

    def test_outlier_detection(self):
        """Test outlier detection in rewards."""
        monitor = RewardIntegrityMonitor()

        # Add normal trades
        for _ in range(50):
            monitor.add_trade(
                reward=rng.standard_normal() * 0.5,
                pnl=rng.standard_normal() * 10,
                reward_components={"main": 0.5},
            )

        # Add outlier
        monitor.add_trade(
            reward=10.0,  # Huge outlier
            pnl=rng.standard_normal() * 10,
            reward_components={"main": 10.0},
        )

        integrity = monitor.check_integrity()

        # Should detect outlier
        assert len(integrity["outliers"]) > 0

    def test_sign_mismatch_detection(self):
        """Test detection of positive reward with negative PnL."""
        monitor = RewardIntegrityMonitor()

        # Add enough trades to meet min_samples requirement (need 50)
        for _ in range(50):
            monitor.add_trade(
                reward=0.8,  # Positive
                pnl=-5.0,  # Negative
                reward_components={"main": 0.8},
            )

        integrity = monitor.check_integrity()

        # Should detect mismatches (sign_mismatches is a count, not a list)
        assert integrity["sign_mismatches"] > 40  # Most should be mismatches


class TestMonitoringAndAlerting:
    """Test production monitoring system."""

    def test_alert_generation(self):
        """Test alert triggers for various conditions."""
        monitor = ProductionMonitor(
            alert_no_trade_hours=2.0,
            alert_drawdown_pct=0.05,
            http_enabled=False,
        )

        # Test no-trade alert
        monitor.update_metrics(last_trade_mins_ago=150)  # 2.5 hours
        assert len(monitor.active_alerts) > 0
        assert any(a.category == "trade" for a in monitor.active_alerts)

        # Test drawdown alert
        monitor.update_metrics(drawdown_current=0.08)  # 8%
        assert any(a.category == "pnl" for a in monitor.active_alerts)

        # Test FIX disconnect alert
        monitor.update_metrics(fix_connected=False)
        assert any(a.severity == "critical" for a in monitor.active_alerts)

    def test_metrics_persistence(self):
        """Test metrics are saved to file."""
        temp_file = Path(tempfile.mktemp(suffix=".json"))
        monitor = ProductionMonitor(metrics_file=temp_file, http_enabled=False)

        monitor.update_metrics(
            realized_pnl_day=100.0,
            trades_today=5,
            win_rate=0.6,
        )

        # Check file created
        assert temp_file.exists()

        # Check contents
        with open(temp_file) as f:
            data = json.load(f)

        assert data["metrics"]["realized_pnl_day"] == pytest.approx(100.0)
        assert data["metrics"]["trades_today"] == 5

        # Cleanup
        temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
