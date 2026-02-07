"""Gap tests for generalization_monitor.py lines 267-271 and time_features.py line 74 + exception paths.

Targets:
- generalization_monitor.py:
  - Lines 267-268: UNDERFITTING when ks_statistic > threshold and both means < 0
  - Lines 270-271: REGIME_SHIFT when ks_statistic > threshold, gap small, not both negative
- time_features.py:
  - Line 74: normalize_angle with negative input (normalized += 360)
  - Lines 256-258: minutes_to_session_close exception path
  - Lines 298-299: minutes_to_rollover exception path
  - Lines 335-336: minutes_to_day_end exception path
  - Lines 358-359: day_of_week_encoded exception path
  - Lines 379-380: is_friday_close_approaching exception path
"""

import datetime as dt
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.core.generalization_monitor import GeneralizationMonitor, GeneralizationState
from src.features.time_features import TimeFeatures

# Import the local SafeMath from time_features for normalize_angle testing
from src.features.time_features import SafeMath as TFSafeMath


# ---------------------------------------------------------------------------
# GeneralizationMonitor – _classify_state edge cases (lines 267-271)
# ---------------------------------------------------------------------------

class TestClassifyStateEdgeCases:
    def test_underfitting_with_distribution_shift(self):
        """ks_statistic > threshold, both means negative → UNDERFITTING (lines 267-268)."""
        gm = GeneralizationMonitor(window_size=100, ks_threshold=0.3)

        # Set ks_statistic above threshold (simulating distribution shift)
        gm.ks_statistic = 0.5  # > 0.3 threshold

        # Call _classify_state with both means negative and small gap
        state = gm._classify_state(
            train_mean=-0.5,  # Negative
            live_mean=-0.3,   # Negative
            gap=0.05,         # Small gap, below OVERFITTING_GAP_THRESHOLD (0.1)
        )

        assert state == GeneralizationState.UNDERFITTING

    def test_regime_shift_with_distribution_shift(self):
        """ks_statistic > threshold, gap small, at least one mean >= 0 → REGIME_SHIFT (lines 270-271)."""
        gm = GeneralizationMonitor(window_size=100, ks_threshold=0.3)

        # Set ks_statistic above threshold
        gm.ks_statistic = 0.5

        # Call with one positive mean and small gap
        state = gm._classify_state(
            train_mean=0.2,    # Positive
            live_mean=-0.1,    # Negative (but not both)
            gap=0.05,          # Small gap
        )

        assert state == GeneralizationState.REGIME_SHIFT

    def test_regime_shift_both_means_positive(self):
        """ks_statistic > threshold, gap small, both means positive → REGIME_SHIFT."""
        gm = GeneralizationMonitor(window_size=100, ks_threshold=0.3)
        gm.ks_statistic = 0.4

        state = gm._classify_state(
            train_mean=0.3,
            live_mean=0.2,
            gap=0.05,
        )

        assert state == GeneralizationState.REGIME_SHIFT

    def test_overfitting_baseline(self):
        """ks_statistic > threshold, large gap → OVERFITTING (already covered, baseline)."""
        gm = GeneralizationMonitor(window_size=100, ks_threshold=0.3)
        gm.ks_statistic = 0.5

        state = gm._classify_state(
            train_mean=0.5,
            live_mean=0.1,
            gap=0.4,  # Above OVERFITTING_GAP_THRESHOLD (0.1)
        )

        assert state == GeneralizationState.OVERFITTING


# ---------------------------------------------------------------------------
# TimeFeatures – normalize_angle negative input (line 74)
# ---------------------------------------------------------------------------

class TestNormalizeAngleNegative:
    def test_negative_degrees_wraps_correctly(self):
        """Negative degrees → normalized += 360.0 (line 74)."""
        # normalize_angle is on the local SafeMath class in time_features.py
        # In Python, -90 % 360 = 270 (always positive), so line 74 is unreachable.
        # But we test the function works correctly with negative input.
        result = TFSafeMath.normalize_angle(-90.0)
        assert 0.0 <= result < 360.0
        assert result == pytest.approx(270.0)

    def test_large_negative_degrees(self):
        """Large negative value normalizes correctly."""
        result = TFSafeMath.normalize_angle(-720.0)
        assert result == pytest.approx(0.0)

    def test_nan_degrees_returns_zero(self):
        """NaN degrees → returns 0.0 via SafeMath.is_valid check."""
        result = TFSafeMath.normalize_angle(float('nan'))
        assert result == pytest.approx(0.0)

    def test_inf_degrees_returns_zero(self):
        """Inf degrees → returns 0.0."""
        result = TFSafeMath.normalize_angle(float('inf'))
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TimeFeatures – exception paths (lines 256-258, 298-299, 335-336, 358-359, 379-380)
# ---------------------------------------------------------------------------

class TestTimeFeatureExceptionPaths:
    def _get_valid_time(self):
        return dt.datetime(2026, 1, 14, 12, 0, 0, tzinfo=dt.UTC)

    def test_minutes_to_session_close_exception(self):
        """Exception in minutes_to_session_close → returns 0.0 (lines 256-258)."""
        tf = TimeFeatures()
        valid_time = self._get_valid_time()

        # Patch timedelta to raise during the calculation
        with patch.object(tf, "cache", side_effect=Exception("test")):
            # Direct cache access failing won't trigger the right path.
            # Instead, pass a malicious datetime that causes errors deep in the calculation.
            pass

        # A simpler approach: patch the internal to throw
        with patch("src.features.time_features.SafeMath.clamp", side_effect=Exception("boom")):
            result = tf.minutes_to_session_close(valid_time)
            assert result == pytest.approx(0.0)

    def test_minutes_to_rollover_exception(self):
        """Exception in minutes_to_rollover → returns 0.0 (lines 298-299)."""
        tf = TimeFeatures()
        valid_time = self._get_valid_time()

        with patch("src.features.time_features.SafeMath.clamp", side_effect=Exception("boom")):
            result = tf.minutes_to_rollover(valid_time)
            assert result == pytest.approx(0.0)

    def test_minutes_to_day_end_exception(self):
        """Exception in minutes_to_day_end → returns 0.0 (lines 335-336)."""
        tf = TimeFeatures()
        valid_time = self._get_valid_time()

        with patch("src.features.time_features.SafeMath.clamp", side_effect=Exception("boom")):
            result = tf.minutes_to_day_end(valid_time)
            assert result == pytest.approx(0.0)

    def test_day_of_week_encoded_exception(self):
        """Exception in day_of_week_encoded → returns 0.0 (lines 358-359)."""
        tf = TimeFeatures()
        valid_time = self._get_valid_time()

        # Patch weekday to raise
        mock_time = MagicMock(spec=dt.datetime)
        mock_time.tzinfo = dt.UTC
        mock_time.weekday.side_effect = Exception("boom")

        # We need _validate_datetime to pass, so patch it
        with patch.object(tf, "_validate_datetime", return_value=True):
            result = tf.day_of_week_encoded(mock_time)
            assert result == pytest.approx(0.0)

    def test_is_friday_close_approaching_exception(self):
        """Exception in is_friday_close_approaching → returns False (lines 379-380)."""
        tf = TimeFeatures()

        # Patch minutes_to_session_close to raise
        with patch.object(tf, "minutes_to_session_close", side_effect=Exception("boom")):
            with patch.object(tf, "_validate_datetime", return_value=True):
                result = tf.is_friday_close_approaching(self._get_valid_time())
                assert result is False
