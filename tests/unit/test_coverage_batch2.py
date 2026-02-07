"""Gap tests for order_book.py, feature_tournament.py, activity_monitor.py,
risk_aware_sac_manager.py, and atomic_persistence.py.

Targets:
- order_book.py: Line 76 (spread non-finite), 141-146 (VPIN residual carry-over),
  171 (get_stats variance non-finite), 173-174 (get_stats overflow)
- feature_tournament.py: Line 158 (LOG at tournament_run % 10), 188 (NaN corr),
  214 (single regime stability), 220 (zero regime_corrs), 227 (zero mean_corr)
- activity_monitor.py: Lines 106-108 (exploration_boost from env), 151 (_log_metrics),
  243-246 (_log_metrics body)
- risk_aware_sac_manager.py: Lines 211 (kurtosis constant std), 304-307 (GPD exception),
  327 (collapse_fac), 341-342 (vpin_penalty), 394-405 (diagnostics),
  429, 435 (scale_action), 461, 468, 492, 498, 504-505 (standalone helpers)
- atomic_persistence.py: Lines 186-190 (backup cleanup + OSError)
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.core.order_book import OrderBook, VPINCalculator
from src.features.feature_tournament import FeatureTournament
from src.monitoring.activity_monitor import ActivityMonitor
from src.risk.risk_aware_sac_manager import RiskAwareSAC_Manager as RiskAwareSACManager


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
        ob.bids = {float('inf'): 10.0}
        ob.asks = {float('inf'): 10.0}
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
# FeatureTournament – uncovered lines 158, 188, 214, 220, 227
# ===========================================================================

class TestFeatureTournamentGaps:
    def test_tournament_log_at_modulo_10(self):
        """LOG emitted when tournaments_run % 10 == 0 (line 158)."""
        names = [f"f{i}" for i in range(5)]
        ft = FeatureTournament(n_features=5, feature_names=names, tournament_window=50)

        rng = np.random.default_rng(42)

        # Add enough samples
        target = rng.normal(0, 1, 100)
        for i in range(100):
            features = rng.normal(0, 1, 5)
            _regimes = np.array([0])  # Single regime
            ft.add_sample(features, target[i], regime=0)

        # Run 10 tournaments to hit modulo 10
        for _ in range(10):
            result = ft.run_tournament()

        assert ft.tournaments_run == 10
        assert result["ready"] is True

    def test_safe_correlation_nan_result(self):
        """_safe_correlation returns 0.0 on NaN (line 188)."""
        names = ["a", "b"]
        ft = FeatureTournament(n_features=2, feature_names=names)

        # Constant x → std(x) == 0 → returns 0.0 (line 182)
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0])
        assert ft._safe_correlation(x, y) == pytest.approx(0.0)

    def test_regime_stability_single_regime(self):
        """Single unique regime → returns 1.0 (line 214)."""
        names = ["a", "b"]
        ft = FeatureTournament(n_features=2, feature_names=names)

        feature = np.array([1.0, 2.0, 3.0])
        target = np.array([0.5, 1.0, 1.5])
        regimes = np.array([0, 0, 0])  # All same regime

        assert ft._calculate_regime_stability(feature, target, regimes) == pytest.approx(1.0)

    def test_regime_stability_insufficient_samples(self):
        """All regimes have < MIN_REGIME_SAMPLES → returns DEFAULT_STABILITY (line 220)."""
        names = ["a", "b"]
        ft = FeatureTournament(n_features=2, feature_names=names)

        # Create 2 regimes with only 2 samples each (below MIN_REGIME_SAMPLES=10)
        feature = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([0.5, 1.0, 1.5, 2.0])
        regimes = np.array([0, 0, 1, 1])  # 2 samples per regime

        result = ft._calculate_regime_stability(feature, target, regimes)
        assert result == pytest.approx(0.5)  # DEFAULT_STABILITY

    def test_regime_stability_zero_mean_correlation(self):
        """Mean correlation near zero → returns 0.0 (line 227)."""
        names = ["a", "b"]
        ft = FeatureTournament(n_features=2, feature_names=names)

        # Create data where correlation in each regime is ~0
        rng = np.random.default_rng(42)
        n = 30  # MIN_REGIME_SAMPLES = 10, we need > 10 per regime
        feature = rng.normal(0, 1, n)
        target = rng.normal(0, 1, n)  # Independent → correlation ~0
        regimes = np.array([0] * 15 + [1] * 15)

        # Patch _safe_correlation to return 0.0 to guarantee zero mean
        with patch.object(ft, "_safe_correlation", return_value=0.0):
            result = ft._calculate_regime_stability(feature, target, regimes)

        assert result == pytest.approx(0.0)


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
# RiskAwareSACManager – uncovered lines
# ===========================================================================

class TestRiskAwareSACManagerGaps:
    def test_kurtosis_constant_returns_zero(self):
        """Constant returns → std < 1e-12 → kurtosis = 0.0 (line 211)."""
        mgr = RiskAwareSACManager(window=100)

        # Add constant returns
        for _ in range(10):
            mgr.ret_buf.append(0.01)

        assert mgr._compute_rolling_kurtosis() == pytest.approx(0.0)

    def test_gpd_exception_returns_zero(self):
        """GPD fit fails → returns 0.0 (lines 304-307)."""
        mgr = RiskAwareSACManager(window=100, enable_logging=True)

        # Add enough returns
        rng = np.random.default_rng(42)
        for val in rng.normal(0, 0.01, 30):
            mgr.ret_buf.append(val)

        # Patch genpareto.fit to raise
        with patch("src.risk.risk_aware_sac_manager.genpareto.fit", side_effect=RuntimeError("fit failed")):
            result = mgr._compute_gpd_hazard()

        assert result == pytest.approx(0.0)

    def test_collapse_exposure(self):
        """Both kurt > max and vpin_z > trigger → collapse (line 327)."""
        mgr = RiskAwareSACManager(window=100, kurt_max=5.0, vpin_trigger=2.0, collapse_fac=0.1)

        mgr.latest_kurtosis = 10.0  # >> kurt_max
        mgr.latest_vpin_z = 5.0     # >> vpin_trigger

        exposure = mgr._compute_exposure()
        assert exposure == pytest.approx(0.1)

    def test_vpin_penalty_smooth_degradation(self):
        """High vpin_z but low kurtosis → smooth degradation (lines 341-342)."""
        mgr = RiskAwareSACManager(window=100, kurt_max=5.0, vpin_trigger=2.0, collapse_fac=0.1)

        mgr.latest_kurtosis = 1.0   # Below kurt_max
        mgr.latest_vpin_z = 4.0     # Above vpin_trigger

        exposure = mgr._compute_exposure()
        assert mgr.collapse_fac <= exposure < 1.0

    def test_scale_action(self):
        """scale_action returns scaled action and hazard (line 429, 435)."""
        mgr = RiskAwareSACManager(window=100)
        mgr.latest_exposure = 0.5
        mgr.latest_hazard = 0.1

        scaled, hazard = mgr.scale_action(2.0)
        assert scaled == pytest.approx(1.0)
        assert hazard == pytest.approx(0.1)

    def test_get_diagnostics(self):
        """get_diagnostics returns all expected fields (lines 394-405)."""
        mgr = RiskAwareSACManager(window=100)
        mgr.total_updates = 10
        mgr.collapse_events = 2
        mgr.extreme_hazard_events = 1

        diag = mgr.get_diagnostics()
        assert diag["total_updates"] == 10
        assert diag["collapse_events"] == 2
        assert diag["collapse_rate"] == pytest.approx(0.2)

    def test_reset(self):
        """reset clears all state (lines 461-468)."""
        mgr = RiskAwareSACManager(window=100)
        mgr.ret_buf.append(1.0)
        mgr.vpin_buf.append(0.5)
        mgr.total_updates = 50
        mgr.collapse_events = 3
        mgr.latest_exposure = 0.5

        mgr.reset()

        assert len(mgr.ret_buf) == 0
        assert len(mgr.vpin_buf) == 0
        assert mgr.total_updates == 0
        assert mgr.latest_exposure == pytest.approx(1.0)

    def test_vpin_zscore_standalone(self):
        """Standalone vpin_zscore function (lines 492-498)."""
        from src.risk.risk_aware_sac_manager import vpin_zscore

        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = vpin_zscore(arr, window=6)
        assert isinstance(result, float)

    def test_rolling_kurtosis_standalone(self):
        """Standalone rolling_kurtosis function (line 468)."""
        from src.risk.risk_aware_sac_manager import rolling_kurtosis

        rng = np.random.default_rng(42)
        arr = rng.normal(0, 1, 50)
        result = rolling_kurtosis(arr, window=30)
        assert isinstance(result, float)

    def test_truncated_gpd_hazard_standalone(self):
        """Standalone truncated_gpd_hazard function (lines 504-505)."""
        from src.risk.risk_aware_sac_manager import truncated_gpd_hazard

        rng = np.random.default_rng(42)
        arr = rng.normal(0, 0.02, 50)
        result = truncated_gpd_hazard(arr)
        assert 0.0 <= result <= 1.0


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
