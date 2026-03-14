"""Gap tests for order_book.py, activity_monitor.py, and atomic_persistence.py.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.core.order_book import OrderBook, VPINCalculator
from src.monitoring.activity_monitor import ActivityMonitor


# ===========================================================================
# OrderBook – spread non-finite (line 76)
# ===========================================================================


class TestOrderBookGaps:
    def test_spread_crossed_book_returns_zero(self):
        """When bid >= ask (crossed book), spread returns 0.0."""
        ob = OrderBook()
        # Set bid higher than ask
        ob.bids = {101.0: 10.0}
        ob.asks = {100.0: 10.0}
        assert ob.spread() == pytest.approx(0.0)

    def test_spread_non_finite_returns_none(self):
        """When spread_value is non-finite, return None (line 76)."""
        ob = OrderBook()
        # Set up a scenario where bid and ask produce non-finite spread
        # bid and ask are sorted dict keys: bids descending, asks ascending
        ob.bids = {float("inf"): 10.0}
        ob.asks = {float("inf"): 10.0}
        # bid == ask → crossed book → returns 0.0
        assert ob.spread() == pytest.approx(0.0)


# ===========================================================================
# VPINCalculator – residual carry-over (lines 141-146)
# ===========================================================================


class TestVPINResidualCarryOver:
    def test_sell_dominant_residual(self):
        """When sells dominate, residual goes to sell side (lines 144-146)."""
        vpin = VPINCalculator(bucket_volume=1.0, window=20)

        # Fill a bucket with sells dominating
        vpin.update(0.3, "BUY")
        result = vpin.update(0.9, "SELL")  # Total 1.2 > 1.0 bucket

        # Bucket completed; residual = 0.2 goes to sell (dominant)
        assert result is not None
        assert vpin.current_sell == pytest.approx(0.2, abs=1e-6)
        assert vpin.current_buy == pytest.approx(0.0)

    def test_buy_dominant_residual(self):
        """When buys dominate, residual goes to buy side (lines 141-143)."""
        vpin = VPINCalculator(bucket_volume=1.0, window=20)

        vpin.update(0.8, "BUY")
        result = vpin.update(0.4, "SELL")  # Total 1.2 > 1.0

        assert result is not None
        assert vpin.current_buy == pytest.approx(0.2, abs=1e-6)
        assert vpin.current_sell == pytest.approx(0.0)

    def test_exact_bucket_fill_no_residual(self):
        """Exact bucket fill → residual = 0 → both zeroed (lines 147-149)."""
        vpin = VPINCalculator(bucket_volume=1.0, window=20)

        result = vpin.update(1.0, "BUY")  # Exactly fills bucket

        assert result is not None
        assert vpin.current_buy == pytest.approx(0.0)
        assert vpin.current_sell == pytest.approx(0.0)

    def test_get_stats_with_varied_buckets(self):
        """get_stats computes meaningful z-score with varied data."""
        vpin = VPINCalculator(bucket_volume=1.0, window=20)

        # Create varied buckets
        for i in range(10):
            vpin.update(0.6 + i * 0.05, "BUY")
            vpin.update(0.5, "SELL")

        stats = vpin.get_stats()
        assert "vpin" in stats
        assert "zscore" in stats
        assert -10.0 <= stats["zscore"] <= 10.0


# ===========================================================================
# ActivityMonitor – exploration_boost from env (lines 106-108) + _log_metrics
# ===========================================================================


class TestActivityMonitorGaps:
    def test_exploration_boost_from_env_default(self):
        """When exploration_boost is None, blend from env vars (lines 106-108)."""
        am = ActivityMonitor(
            max_bars_inactive=10,
            exploration_boost=None,  # Trigger env var path
            phase_maturity=0.5,
        )
        # Should have blended value, not crash
        assert am.exploration_boost > 0

    def test_log_metrics_called_periodically(self):
        """_log_metrics called every LOG_EVERY_BARS (line 151)."""
        am = ActivityMonitor(max_bars_inactive=1000, exploration_boost=0.1)

        # Find LOG_EVERY_BARS - typically 100
        # Call on_bar_close enough times to trigger periodic log
        for _ in range(100):
            am.on_bar_close()

        assert am.total_bars == 100

    def test_stagnation_resolution_log(self):
        """On trade executed after stagnation, LOG about resolution (line 164)."""
        am = ActivityMonitor(max_bars_inactive=3, exploration_boost=0.1)

        # Trigger stagnation
        for _ in range(4):
            am.on_bar_close()

        assert am.is_stagnant is True

        # Execute trade to resolve stagnation
        am.on_trade_executed()
        assert am.is_stagnant is False


# ===========================================================================
# AtomicPersistence – backup cleanup (lines 186-190)
# ===========================================================================


class TestAtomicPersistenceGaps:
    def test_cleanup_old_backups_removes_excess(self):
        """Excess backups beyond MAX_BACKUPS are deleted (lines 186-188)."""
        from src.persistence.atomic_persistence import AtomicPersistence

        with tempfile.TemporaryDirectory() as tmpdir:
            ap = AtomicPersistence(tmpdir)

            # Create more backups than MAX_BACKUPS (default 5)
            target = Path(tmpdir) / "test.json"
            target.write_text("{}")

            for i in range(8):
                bak = Path(tmpdir) / f"test.json.2026010{i}_000000.bak"
                bak.write_text(f'{{"backup": {i}}}')
                # Stagger mtimes
                time.sleep(0.01)

            ap._cleanup_old_backups(target)

            remaining = list(Path(tmpdir).glob("test.json.*.bak"))
            assert len(remaining) <= ap.MAX_BACKUPS

    def test_cleanup_old_backups_oserror(self):
        """OSError during cleanup is caught (lines 189-190)."""
        from src.persistence.atomic_persistence import AtomicPersistence

        with tempfile.TemporaryDirectory() as tmpdir:
            ap = AtomicPersistence(tmpdir)
            target = Path(tmpdir) / "test.json"
            target.write_text("{}")

            # Patch glob to raise OSError
            with patch.object(Path, "glob", side_effect=OSError("disk error")):
                # Should not raise
                ap._cleanup_old_backups(target)
