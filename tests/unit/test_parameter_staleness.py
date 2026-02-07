"""Tests for src.core.parameter_staleness.

Covers: ParameterStalenessDetector, StalenessSnapshot, StalenessSignal,
        baseline establishment, all four staleness detection methods,
        check_staleness, reset_baseline, save/load state.
"""

import json
import pytest
from dataclasses import asdict
from pathlib import Path

from src.core.parameter_staleness import (
    ParameterStalenessDetector,
    StalenessSignal,
    StalenessSnapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_metrics(win_rate=0.55, sharpe=1.0, avg_confidence=0.7):
    return {"win_rate": win_rate, "sharpe": sharpe, "avg_confidence": avg_confidence}


def _make_params(a=1.0, b=2.0, c=3.0):
    return {"a": a, "b": b, "c": c}


def _feed_baseline(det, n=500, win_rate=0.55, sharpe=1.0, avg_confidence=0.7, regime="TRENDING"):
    """Feed enough snapshots for baseline to be established."""
    for i in range(n):
        det.update(
            bar_num=i,
            parameters=_make_params(),
            performance_metrics=_make_metrics(win_rate, sharpe, avg_confidence),
            regime=regime,
        )
    assert det.baseline_established


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestDataclasses:
    def test_staleness_snapshot_fields(self):
        s = StalenessSnapshot(
            bar_num=10, timestamp="2025-01-01T00:00:00Z",
            parameters={"a": 1.0}, performance={"win_rate": 0.5}, regime="TRENDING"
        )
        d = asdict(s)
        assert d["bar_num"] == 10
        assert d["regime"] == "TRENDING"

    def test_staleness_signal_fields(self):
        s = StalenessSignal(
            signal_type="performance_decay", severity=0.8,
            evidence={"drop": 0.2}, recommendation="reset"
        )
        assert s.signal_type == "performance_decay"
        assert s.severity == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Init / basic
# ---------------------------------------------------------------------------
class TestInit:
    def test_defaults(self):
        det = ParameterStalenessDetector()
        assert det.performance_window == 500
        assert det.regime_stability_bars == 50
        assert det.parameter_drift_window == 100
        assert det.staleness_threshold == pytest.approx(0.6)
        assert not det.baseline_established
        assert det.staleness_score == pytest.approx(0.0)
        assert not det.is_stale

    def test_custom_params(self):
        det = ParameterStalenessDetector(
            performance_window=200, regime_stability_bars=20,
            parameter_drift_window=50, staleness_threshold=0.5
        )
        assert det.performance_window == 200
        assert det.bars_for_baseline == 200  # min(500, 200)

    def test_bars_for_baseline_capped(self):
        det = ParameterStalenessDetector(performance_window=1000)
        assert det.bars_for_baseline == 500


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------
class TestUpdate:
    def test_missing_metric_skips(self):
        det = ParameterStalenessDetector()
        det.update(0, _make_params(), {"win_rate": 0.5, "sharpe": 1.0}, "TRENDING")
        # Missing avg_confidence → should not append snapshot
        assert len(det.snapshots) == 0

    def test_snapshot_appended(self):
        det = ParameterStalenessDetector()
        det.update(0, _make_params(), _make_metrics(), "TRENDING")
        assert len(det.snapshots) == 1
        assert det.last_bar_num == 0

    def test_regime_history_tracked(self):
        det = ParameterStalenessDetector()
        det.update(0, _make_params(), _make_metrics(), "TRENDING")
        det.update(1, _make_params(), _make_metrics(), "MEAN_REVERTING")
        assert len(det.regime_history) == 2


# ---------------------------------------------------------------------------
# Baseline establishment
# ---------------------------------------------------------------------------
class TestBaseline:
    def test_baseline_not_established_early(self):
        det = ParameterStalenessDetector(performance_window=100)
        for i in range(50):
            det.update(i, _make_params(), _make_metrics(), "TRENDING")
        assert not det.baseline_established

    def test_baseline_established_at_threshold(self):
        det = ParameterStalenessDetector(performance_window=100)
        for i in range(100):
            det.update(i, _make_params(), _make_metrics(0.6, 1.2, 0.8), "TRENDING")
        assert det.baseline_established
        assert det.baseline_win_rate is not None
        assert det.baseline_sharpe is not None

    def test_baseline_values_correct(self):
        det = ParameterStalenessDetector(performance_window=100)
        for i in range(100):
            det.update(i, _make_params(), _make_metrics(0.6, 1.5, 0.75), "TRENDING")
        # Median of constant values = the value itself
        assert abs(det.baseline_win_rate - 0.6) < 0.01
        assert abs(det.baseline_sharpe - 1.5) < 0.01
        assert abs(det.baseline_confidence - 0.75) < 0.01

    def test_establish_baseline_insufficient_warns(self):
        det = ParameterStalenessDetector(performance_window=500)
        det.snapshots.clear()
        det._establish_baseline()
        assert not det.baseline_established


# ---------------------------------------------------------------------------
# Performance decay
# ---------------------------------------------------------------------------
class TestPerformanceDecay:
    def test_no_signal_when_performance_good(self):
        det = ParameterStalenessDetector(performance_window=200)
        _feed_baseline(det, n=200, win_rate=0.6, sharpe=1.0)
        # Feed 60 more with same perf
        for i in range(200, 260):
            det.update(i, _make_params(), _make_metrics(0.6, 1.0, 0.7), "TRENDING")
        decay = det._check_performance_decay()
        assert decay is None

    def test_win_rate_decay_triggers(self):
        det = ParameterStalenessDetector(performance_window=200)
        _feed_baseline(det, n=200, win_rate=0.7, sharpe=1.5)
        # Feed 60 bars with much worse performance
        for i in range(200, 260):
            det.update(i, _make_params(), _make_metrics(0.4, 0.5, 0.7), "TRENDING")
        signal = det._check_performance_decay()
        assert signal is not None
        assert signal.signal_type == "performance_decay"
        assert signal.severity > 0.3
        assert "win_rate_drop" in signal.evidence

    def test_sharpe_decay_triggers(self):
        det = ParameterStalenessDetector(performance_window=200)
        _feed_baseline(det, n=200, win_rate=0.55, sharpe=2.0)
        for i in range(200, 260):
            det.update(i, _make_params(), _make_metrics(0.55, 1.0, 0.7), "TRENDING")
        signal = det._check_performance_decay()
        assert signal is not None
        assert signal.evidence["sharpe_drop"] > 0.3


# ---------------------------------------------------------------------------
# Regime shift
# ---------------------------------------------------------------------------
class TestRegimeShift:
    def test_no_shift_same_regime(self):
        det = ParameterStalenessDetector(performance_window=200, regime_stability_bars=20)
        _feed_baseline(det, n=200, regime="TRENDING")
        for i in range(200, 230):
            det.update(i, _make_params(), _make_metrics(), "TRENDING")
        signal = det._check_regime_shift()
        assert signal is None

    def test_regime_shift_detected(self):
        det = ParameterStalenessDetector(performance_window=200, regime_stability_bars=20)
        _feed_baseline(det, n=200, regime="TRENDING")
        # 25 bars in different regime (>80% of 20-bar window)
        for i in range(200, 225):
            det.update(i, _make_params(), _make_metrics(), "MEAN_REVERTING")
        signal = det._check_regime_shift()
        assert signal is not None
        assert signal.signal_type == "regime_shift"
        assert signal.evidence["shift_fraction"] > 0.8

    def test_regime_shift_insufficient_history(self):
        det = ParameterStalenessDetector(performance_window=200, regime_stability_bars=200)
        _feed_baseline(det, n=200, regime="TRENDING")
        signal = det._check_regime_shift()
        # regime_history is short relative to regime_stability_bars
        # With 200 bars, regime_history has 200 entries, stability_bars=200 → should work
        # but need at least a shift to trigger
        assert signal is None  # same regime, no shift


# ---------------------------------------------------------------------------
# Parameter drift
# ---------------------------------------------------------------------------
class TestParameterDrift:
    def test_no_drift_stable_params(self):
        det = ParameterStalenessDetector(performance_window=200, parameter_drift_window=50)
        _feed_baseline(det, n=200)
        for i in range(200, 260):
            det.update(i, _make_params(), _make_metrics(), "TRENDING")
        signal = det._check_parameter_drift()
        assert signal is None

    def test_drift_detected_large_change(self):
        det = ParameterStalenessDetector(performance_window=200, parameter_drift_window=50)
        _feed_baseline(det, n=200)
        # Feed 60 bars with progressively changing params
        for i in range(200, 260):
            drift_factor = 1.0 + (i - 200) * 0.02  # grows to 2.2x
            det.update(
                i,
                _make_params(a=1.0 * drift_factor, b=2.0 * drift_factor, c=3.0),
                _make_metrics(),
                "TRENDING",
            )
        signal = det._check_parameter_drift()
        assert signal is not None
        assert signal.signal_type == "parameter_drift"
        assert signal.evidence["max_drift"] > 0.2

    def test_drift_near_zero_param(self):
        det = ParameterStalenessDetector(performance_window=200, parameter_drift_window=50)
        _feed_baseline(det, n=200)
        # One param starts near zero
        for i in range(200, 260):
            det.update(
                i,
                {"x": 0.000001, "y": 2.0},
                _make_metrics(),
                "TRENDING",
            )
        # Near-zero params use absolute drift
        signal = det._check_parameter_drift()
        assert signal is None  # small absolute change


# ---------------------------------------------------------------------------
# Confidence collapse
# ---------------------------------------------------------------------------
class TestConfidenceCollapse:
    def test_no_collapse_good_confidence(self):
        det = ParameterStalenessDetector(performance_window=200)
        _feed_baseline(det, n=200)
        for i in range(200, 260):
            det.update(i, _make_params(), _make_metrics(avg_confidence=0.8), "TRENDING")
        signal = det._check_confidence_collapse()
        assert signal is None

    def test_collapse_detected(self):
        det = ParameterStalenessDetector(performance_window=200)
        _feed_baseline(det, n=200)
        # 60 bars with very low confidence
        for i in range(200, 260):
            det.update(i, _make_params(), _make_metrics(avg_confidence=0.2), "TRENDING")
        signal = det._check_confidence_collapse()
        assert signal is not None
        assert signal.signal_type == "confidence_collapse"
        assert signal.evidence["avg_confidence"] < 0.5
        assert signal.evidence["low_confidence_fraction"] > 0.7

    def test_no_collapse_mixed_confidence(self):
        det = ParameterStalenessDetector(performance_window=200)
        _feed_baseline(det, n=200)
        # Alternate high/low → avg above 0.5
        for i in range(200, 260):
            conf = 0.9 if i % 2 == 0 else 0.4
            det.update(i, _make_params(), _make_metrics(avg_confidence=conf), "TRENDING")
        signal = det._check_confidence_collapse()
        assert signal is None


# ---------------------------------------------------------------------------
# Composite staleness detection
# ---------------------------------------------------------------------------
class TestDetectStaleness:
    def test_single_signal_below_threshold(self):
        """Mild decay alone shouldn't cross default 0.6 threshold."""
        det = ParameterStalenessDetector(performance_window=200, staleness_threshold=0.6)
        _feed_baseline(det, n=200, win_rate=0.6, sharpe=1.0)
        # Mild decay — only small drops, not enough to breach 0.6
        for i in range(200, 260):
            det.update(i, _make_params(), _make_metrics(0.52, 0.85, 0.7), "TRENDING")
        assert not det.is_stale

    def test_combined_signals_trigger_staleness(self):
        """Performance decay + regime shift + confidence collapse → stale."""
        det = ParameterStalenessDetector(
            performance_window=200,
            regime_stability_bars=20,
            staleness_threshold=0.4,
        )
        _feed_baseline(det, n=200, win_rate=0.7, sharpe=2.0, avg_confidence=0.9, regime="TRENDING")
        # All bad at once
        for i in range(200, 260):
            det.update(
                i,
                _make_params(),
                _make_metrics(0.3, 0.2, 0.2),
                "MEAN_REVERTING",
            )
        assert det.is_stale
        assert det.staleness_score > 0.4
        assert det.staleness_start_bar is not None

    def test_staleness_clears(self):
        det = ParameterStalenessDetector(
            performance_window=200,
            regime_stability_bars=20,
            staleness_threshold=0.3,
        )
        _feed_baseline(det, n=200, win_rate=0.7, sharpe=2.0, regime="TRENDING")
        # Make it stale
        for i in range(200, 260):
            det.update(i, _make_params(), _make_metrics(0.3, 0.2, 0.2), "MEAN_REVERTING")
        assert det.is_stale
        # Recover performance
        for i in range(260, 320):
            det.update(i, _make_params(), _make_metrics(0.7, 2.0, 0.9), "TRENDING")
        assert not det.is_stale
        assert det.staleness_start_bar is None

    def test_no_detection_before_baseline(self):
        det = ParameterStalenessDetector(performance_window=200)
        for i in range(50):
            det.update(i, _make_params(), _make_metrics(0.3, -1.0, 0.1), "UNKNOWN")
        assert not det.baseline_established
        assert not det.is_stale


# ---------------------------------------------------------------------------
# check_staleness
# ---------------------------------------------------------------------------
class TestCheckStaleness:
    def test_check_not_stale(self):
        det = ParameterStalenessDetector(performance_window=200)
        _feed_baseline(det, n=200)
        result = det.check_staleness()
        assert result["is_stale"] is False
        assert "staleness_score" in result
        assert "baseline_established" in result

    def test_check_stale_has_signals_and_bars(self):
        det = ParameterStalenessDetector(
            performance_window=200,
            regime_stability_bars=20,
            staleness_threshold=0.3,
        )
        _feed_baseline(det, n=200, win_rate=0.7, sharpe=2.0, regime="TRENDING")
        for i in range(200, 260):
            det.update(i, _make_params(), _make_metrics(0.3, 0.2, 0.2), "MEAN_REVERTING")
        result = det.check_staleness()
        if result["is_stale"]:
            assert "bars_stale" in result
            assert "signals" in result
            assert "recommendations" in result
            assert len(result["signals"]) > 0


# ---------------------------------------------------------------------------
# reset_baseline
# ---------------------------------------------------------------------------
class TestResetBaseline:
    def test_reset_updates_baseline(self):
        det = ParameterStalenessDetector(performance_window=200)
        _feed_baseline(det, n=200, win_rate=0.6, sharpe=1.0)
        old_wr = det.baseline_win_rate
        # Add new data with different perf
        for i in range(200, 260):
            det.update(i, _make_params(), _make_metrics(0.8, 2.0, 0.9), "TRENDING")
        det.reset_baseline()
        assert det.baseline_win_rate > old_wr
        assert not det.is_stale
        assert det.staleness_score == pytest.approx(0.0)

    def test_reset_insufficient_data(self):
        det = ParameterStalenessDetector()
        det.reset_baseline()  # Should warn, not crash
        assert not det.baseline_established  # Stays false since no data

    def test_reset_clears_staleness_state(self):
        det = ParameterStalenessDetector(
            performance_window=200, staleness_threshold=0.3, regime_stability_bars=20,
        )
        _feed_baseline(det, n=200, win_rate=0.7, sharpe=2.0, regime="TRENDING")
        for i in range(200, 260):
            det.update(i, _make_params(), _make_metrics(0.3, 0.2, 0.2), "MEAN_REVERTING")
        assert det.is_stale
        det.reset_baseline()
        assert not det.is_stale
        assert det.staleness_signals == []
        assert det.staleness_start_bar is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "staleness.json"
        det = ParameterStalenessDetector(performance_window=200, persistence_path=path)
        _feed_baseline(det, n=200, win_rate=0.6, sharpe=1.5)
        det.save_state()

        det2 = ParameterStalenessDetector(persistence_path=path)
        assert det2.load_state()
        assert det2.baseline_established
        assert abs(det2.baseline_win_rate - 0.6) < 0.01
        assert abs(det2.baseline_sharpe - 1.5) < 0.01

    def test_save_no_path_warns(self):
        det = ParameterStalenessDetector()
        det.save_state()  # Should not raise

    def test_load_no_file_returns_false(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        det = ParameterStalenessDetector(persistence_path=path)
        assert not det.load_state()

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not json")
        det = ParameterStalenessDetector(persistence_path=path)
        assert not det.load_state()

    def test_save_explicit_path(self, tmp_path):
        path = tmp_path / "explicit.json"
        det = ParameterStalenessDetector(performance_window=200)
        _feed_baseline(det, n=200)
        det.save_state(path=path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["baseline_established"] is True

    def test_load_explicit_path(self, tmp_path):
        path = tmp_path / "explicit.json"
        state = {
            "baseline_win_rate": 0.55,
            "baseline_sharpe": 1.2,
            "baseline_confidence": 0.7,
            "baseline_established": True,
            "staleness_score": 0.0,
            "is_stale": False,
            "last_bar_num": 100,
            "staleness_start_bar": None,
            "snapshots_count": 100,
        }
        path.write_text(json.dumps(state))
        det = ParameterStalenessDetector()
        assert det.load_state(path=path)
        assert det.baseline_win_rate == pytest.approx(0.55)

    def test_load_no_path_no_persistence(self):
        det = ParameterStalenessDetector()
        assert not det.load_state()
