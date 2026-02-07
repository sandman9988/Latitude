"""Coverage batch 3: Edge-case tests for near-100% modules.

Targets:
  - performance_tracker.py  (98% → ~100%): lines 75, 121
  - parameter_staleness.py  (97% → ~100%): lines 239, 314, 362, 366, 401, 455
  - safe_math.py            (94% → ~98%):  lines 199-200, 212-213, 225-226, 238-239, 274-276, 357-358
  - risk_aware_sac_manager.py standalone  (92% → ~98%): lines 429, 435, 461, 468, 492, 498, 504-505
  - feature_tournament.py   (99% → 100%): line 188
  - journaled_persistence.py(84% → ~88%): lines 111, 128, 165
"""

import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ── performance_tracker ──────────────────────────────────────────────────

from src.monitoring.performance_tracker import PerformanceTracker


class TestPerformanceTrackerGaps:
    """Cover lines 75 (negative mae clamped) and 121 (zero peak_equity drawdown)."""

    _now = dt.datetime(2025, 1, 1, 12, 0, 0)

    def _trade_kwargs(self, **overrides):
        defaults = dict(
            pnl=-10.0,
            entry_time=self._now,
            exit_time=self._now,
            direction="BUY",
            entry_price=100.0,
            exit_price=99.0,
            mfe=1.0,
            mae=0.5,
        )
        defaults.update(overrides)
        return defaults

    def test_negative_mae_clamped_to_zero(self):
        """Line 75: mae < 0 → mae = 0.0."""
        pt = PerformanceTracker()
        pt.add_trade(**self._trade_kwargs(mae=-5.0))
        # Stored trade should have mae clamped to 0
        assert pt.trades[-1]["mae"] == pytest.approx(0.0)

    def test_zero_peak_equity_drawdown(self):
        """Line 121: peak_equity == 0 → current_drawdown stays 0."""
        pt = PerformanceTracker()
        # Manually set impossible-in-practice state where peak is 0
        pt.peak_equity = 0.0
        pt.current_equity = -100.0
        pt.add_trade(**self._trade_kwargs(pnl=-50.0))
        # Should NOT divide by zero; drawdown stays 0
        assert pt.current_drawdown == pytest.approx(0.0)


# ── parameter_staleness ──────────────────────────────────────────────────

from src.core.parameter_staleness import ParameterStalenessDetector


class TestParameterStalenessGaps:
    """Cover early-return guards in private detection methods."""

    def _make_detector(self, **kwargs):
        defaults = dict(performance_window=500, regime_stability_bars=50, parameter_drift_window=100)
        defaults.update(kwargs)
        return ParameterStalenessDetector(**defaults)

    # Line 239: _detect_staleness returns when snapshots < 50
    def test_detect_staleness_short_window(self):
        det = self._make_detector(performance_window=20)
        perf = {"win_rate": 0.5, "sharpe": 0.5, "avg_confidence": 0.6}
        params = {"lr": 0.01}
        # Fill 25 bars to establish baseline (bars_for_baseline = min(500,20) = 20)
        for i in range(25):
            det.update(bar_num=i, parameters=params, performance_metrics=perf, regime="TRENDING")
        # Baseline established at bar 20. But len(snapshots) == 20 < 50 → triggers line 239
        assert det.baseline_established
        assert det.staleness_score == pytest.approx(0.0)  # No detection ran

    # Line 314: _check_performance_decay returns None when < 50 snapshots
    def test_check_performance_decay_insufficient(self):
        det = self._make_detector()
        # Manually add 10 snapshots so < 50
        from src.core.parameter_staleness import StalenessSnapshot

        for i in range(10):
            det.snapshots.append(
                StalenessSnapshot(
                    bar_num=i,
                    timestamp="",
                    parameters={"lr": 0.01},
                    performance={"win_rate": 0.5, "sharpe": 0.5, "avg_confidence": 0.6},
                    regime="TRENDING",
                )
            )
        result = det._check_performance_decay()
        assert result is None

    # Line 362: _check_regime_shift returns None when regime_history too short
    def test_check_regime_shift_short_regime_history(self):
        det = self._make_detector(regime_stability_bars=100)
        # 10 regimes < 100
        for i in range(10):
            det.regime_history.append("TRENDING")
        result = det._check_regime_shift()
        assert result is None

    # Line 366: _check_regime_shift returns None when snapshots < bars_for_baseline
    def test_check_regime_shift_insufficient_snapshots(self):
        det = self._make_detector(performance_window=500, regime_stability_bars=10)
        # regime_history has enough, but snapshots < bars_for_baseline (500)
        for i in range(20):
            det.regime_history.append("TRENDING")
        result = det._check_regime_shift()
        assert result is None

    # Line 401: _check_parameter_drift returns None when < parameter_drift_window
    def test_check_parameter_drift_insufficient(self):
        det = self._make_detector(parameter_drift_window=200)
        from src.core.parameter_staleness import StalenessSnapshot

        for i in range(10):
            det.snapshots.append(
                StalenessSnapshot(
                    bar_num=i,
                    timestamp="",
                    parameters={"lr": 0.01},
                    performance={"win_rate": 0.5, "sharpe": 0.5, "avg_confidence": 0.6},
                    regime="TRENDING",
                )
            )
        result = det._check_parameter_drift()
        assert result is None

    # Line 455: _check_confidence_collapse returns None when < 50 snapshots
    def test_check_confidence_collapse_insufficient(self):
        det = self._make_detector()
        from src.core.parameter_staleness import StalenessSnapshot

        for i in range(10):
            det.snapshots.append(
                StalenessSnapshot(
                    bar_num=i,
                    timestamp="",
                    parameters={"lr": 0.01},
                    performance={"win_rate": 0.5, "sharpe": 0.5, "avg_confidence": 0.3},
                    regime="TRENDING",
                )
            )
        result = det._check_confidence_collapse()
        assert result is None


# ── safe_math ────────────────────────────────────────────────────────────

from src.utils.safe_math import SafeMath, safe_array_operation


class TestSafeMathExceptionPaths:
    """Cover except handlers in safe_mean, safe_percentile, safe_min, safe_max,
    running_variance_update, and safe_array_operation."""

    class _Unconvertible:
        """Object that raises when numpy tries to convert to array."""

        def __float__(self):
            raise TypeError("nope")

        def __iter__(self):
            raise TypeError("nope")

    def test_safe_mean_exception(self):
        """Lines 199-200: catch Exception → default."""
        result = SafeMath.safe_mean(self._Unconvertible(), default=42.0)
        assert result == pytest.approx(42.0)

    def test_safe_percentile_exception(self):
        """Lines 212-213: catch Exception → default."""
        result = SafeMath.safe_percentile(self._Unconvertible(), 50.0, default=42.0)
        assert result == pytest.approx(42.0)

    def test_safe_min_exception(self):
        """Lines 225-226: catch Exception → default."""
        result = SafeMath.safe_min(self._Unconvertible(), default=42.0)
        assert result == pytest.approx(42.0)

    def test_safe_max_exception(self):
        """Lines 238-239: catch Exception → default."""
        result = SafeMath.safe_max(self._Unconvertible(), default=42.0)
        assert result == pytest.approx(42.0)

    def test_running_variance_small_count(self):
        """Lines 274-276: count < MIN_SAMPLE_COUNT → return 0.0."""
        result = SafeMath.running_variance_update(
            old_variance=1.0, old_mean=5.0, new_mean=6.0, new_value=7.0, count=1
        )
        assert result == pytest.approx(0.0)

    def test_safe_array_operation_exception(self):
        """Lines 357-358: catch Exception → default."""
        # Pass array with object that makes operation fail
        bad_arr = np.array([1.0, 2.0, 3.0])
        # Mock the operation to raise
        with patch("numpy.mean", side_effect=RuntimeError("boom")):
            result = safe_array_operation(bad_arr, "mean", default=-99.0)
        assert result == pytest.approx(-99.0)


# ── risk_aware_sac_manager standalone functions ──────────────────────────

from src.risk.risk_aware_sac_manager import (
    rolling_kurtosis,
    truncated_gpd_hazard,
    vpin_zscore,
)


class TestStandaloneFunctionEdgeCases:
    """Cover edge-case returns in standalone helper functions."""

    def test_rolling_kurtosis_too_few_elements(self):
        """Line 429: len(x) < 4 → 0.0."""
        arr = np.array([1.0, 2.0, 3.0])
        assert rolling_kurtosis(arr, 10) == pytest.approx(0.0)

    def test_rolling_kurtosis_zero_variance(self):
        """Line 435: var < 1e-12 → 0.0."""
        arr = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        assert rolling_kurtosis(arr, 5) == pytest.approx(0.0)

    def test_vpin_zscore_single_element(self):
        """Line 461: len(arr) < 2 → 0.0."""
        arr = np.array([1.0])
        assert vpin_zscore(arr, 10) == pytest.approx(0.0)

    def test_vpin_zscore_zero_std(self):
        """Line 468: hist_std < 1e-12 → 0.0."""
        arr = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        assert vpin_zscore(arr, 5) == pytest.approx(0.0)

    def test_truncated_gpd_hazard_short_array(self):
        """Line 492: len(arr) < 10 → 0.0."""
        arr = np.array([1.0, 2.0, 3.0])
        assert truncated_gpd_hazard(arr) == pytest.approx(0.0)

    def test_truncated_gpd_hazard_few_exceedances(self):
        """Line 498: len(exceedances) < 3 → 0.0."""
        # Array of 10 identical values → percentile == value → no exceedances
        arr = np.array([5.0] * 10)
        assert truncated_gpd_hazard(arr) == pytest.approx(0.0)

    def test_truncated_gpd_hazard_fit_exception(self):
        """Lines 504-505: GPD fit failure → 0.0."""
        with patch("src.risk.risk_aware_sac_manager.genpareto") as mock_gp:
            mock_gp.fit.side_effect = RuntimeError("GPD boom")
            # Array that would have exceedances
            arr = np.concatenate([np.zeros(90), np.ones(10) * 100])
            result = truncated_gpd_hazard(arr)
        assert result == pytest.approx(0.0)


# ── feature_tournament NaN correlation ───────────────────────────────────

from src.features.feature_tournament import FeatureTournament


class TestFeatureTournamentNaN:
    """Cover line 188: np.isnan(corr) → 0.0."""

    def test_nan_correlation_from_nan_values(self):
        """When array has NaN, std != 0 but corrcoef returns NaN."""
        ft = FeatureTournament(n_features=4, feature_names=["a", "b", "c", "d"])
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = ft._safe_correlation(x, y)
        assert result == pytest.approx(0.0)


# ── journaled_persistence ────────────────────────────────────────────────

from src.persistence.journaled_persistence import Journal


class TestJournaledPersistenceGaps:
    """Cover lines 111 (no file), 128 (empty last line), 165 (auto-rotate)."""

    def test_get_last_sequence_no_file(self, tmp_path):
        """Line 111: journal file doesn't exist → return 0."""
        journal_path = tmp_path / "subdir" / "journal.log"
        j = Journal(journal_path=str(journal_path))
        # The constructor calls _get_last_sequence; since file didn't exist
        # before open(), sequence starts at 1 (0+1)
        assert j.sequence_num == 1
        j.journal_file.close()

    def test_get_last_sequence_empty_last_line(self, tmp_path):
        """Line 128: last line is empty after strip → return 0."""
        journal_path = tmp_path / "journal.log"
        # Write a journal with trailing empty lines
        with open(journal_path, "w", encoding="utf-8") as f:
            f.write('{"seq": 5, "ts": "", "op": "test", "data": {}}\n')
            f.write("\n")  # empty last line
        j = Journal(journal_path=str(journal_path))
        # _get_last_sequence reads last line, strips it → empty → falls through to return 0
        # BUT there are 2 lines: line[1] is empty, line[0] has seq=5
        # The code reads ALL lines, takes lines[-1].strip() → ""
        # Falls through if block to return 0 at line 128
        assert j.sequence_num == 1  # 0 + 1
        j.journal_file.close()

    def test_log_operation_triggers_rotation(self, tmp_path):
        """Line 165: _should_rotate=True triggers _rotate_journal."""
        journal_path = tmp_path / "journal.log"
        j = Journal(journal_path=str(journal_path), checkpoint_interval=9999)
        with patch.object(j, "_should_rotate", return_value=True), patch.object(
            j, "_rotate_journal"
        ) as mock_rotate:
            j.log_operation("test_op", {"key": "val"})
        mock_rotate.assert_called_once()
        j.journal_file.close()
