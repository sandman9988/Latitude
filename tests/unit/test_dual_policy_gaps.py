"""Gap tests for dual_policy.py uncovered production lines.

Targets:
- Line 194: path_geometry.last.get("feasibility") during decide_entry
- Lines 205-219: friction_calculator path in decide_entry
- Lines 252-253: LOG.info for regime-adjusted LONG/SHORT entry
- Line 308: LOG.info for harvester CLOSE action
- Lines 386-387: _update_mfe_mae exception on float(current_price)
- Lines 390-391: _update_mfe_mae exception on float(entry_price)
- Line 590: buffer.set_current_regime with training enabled
- Lines 616, 646: add_trigger/harvester_experience with training enabled
- Line 683: train_step LOG when both agents produce metrics
"""

import datetime as dt
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.dual_policy import DualPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 100, base_price: float = 100_000.0, step: float = 10.0) -> deque:
    """Create synthetic bars as deque of (t, o, h, l, c) tuples."""
    bars = deque(maxlen=200)
    base = dt.datetime(2025, 1, 1)
    for i in range(n):
        t = base + dt.timedelta(minutes=i)
        c = base_price + i * step
        bars.append((t, c, c + 5, c - 5, c))
    return bars


# ---------------------------------------------------------------------------
# decide_entry with friction_calculator (lines 205-219)
# ---------------------------------------------------------------------------

class TestDecideEntryFriction:
    def test_friction_calculator_used_when_bars_present(self):
        """friction_calculator.calculate_total_friction is called during decide_entry."""
        mock_friction = MagicMock()
        mock_friction.calculate_total_friction.return_value = {
            "total": 5.0,
            "spread": 1.6,
            "commission": 0.8,
            "swap": 0.0,
            "slippage": 1.0,
        }

        dp = DualPolicy(
            window=64,
            friction_calculator=mock_friction,
            enable_event_features=False,
        )

        bars = _make_bars(n=100)
        action, conf, runway = dp.decide_entry(bars, imbalance=0.0)

        mock_friction.calculate_total_friction.assert_called_once()
        assert action in [0, 1, 2]

    def test_friction_fallback_when_no_bars(self):
        """Fallback friction when bars empty (lines 220-221)."""
        mock_friction = MagicMock()
        dp = DualPolicy(
            window=64,
            friction_calculator=mock_friction,
            enable_event_features=False,
        )

        bars = deque(maxlen=200)  # Empty
        action, conf, runway = dp.decide_entry(bars, imbalance=0.0)

        # Friction calculator should NOT be called with empty bars
        mock_friction.calculate_total_friction.assert_not_called()
        assert action in [0, 1, 2]


# ---------------------------------------------------------------------------
# decide_entry with path_geometry (line 194)
# ---------------------------------------------------------------------------

class TestDecideEntryPathGeometry:
    def test_path_geometry_feasibility_read(self):
        """path_geometry.last.get('feasibility') is used in decide_entry (line 194)."""
        mock_geom = MagicMock()
        mock_geom.update.return_value = {
            "efficiency": 0.8,
            "gamma": 0.1,
            "jerk": 0.05,
            "runway": 0.6,
            "feasibility": 0.7,
        }
        mock_geom.last = {"feasibility": 0.7}

        dp = DualPolicy(
            window=64,
            path_geometry=mock_geom,
            enable_event_features=False,
        )

        bars = _make_bars(n=100)
        action, conf, runway = dp.decide_entry(bars, imbalance=0.0)
        assert action in [0, 1, 2]


# ---------------------------------------------------------------------------
# decide_entry regime-adjusted entry LOG (lines 252-253)
# ---------------------------------------------------------------------------

class TestDecideEntryRegimeLog:
    def test_regime_log_on_long_entry(self):
        """When trigger returns LONG (action=1) and regime_detector is set, LOG is emitted."""
        dp = DualPolicy(window=64, enable_regime_detection=True, enable_event_features=False)

        bars = _make_bars(n=100)

        # Force trigger to return LONG
        dp.trigger.decide = MagicMock(return_value=(1, 0.85, 0.003))

        action, conf, runway = dp.decide_entry(bars, imbalance=0.0)
        assert action == 1
        assert conf == 0.85

    def test_regime_log_on_short_entry(self):
        """When trigger returns SHORT (action=2), regime adjustment is applied."""
        dp = DualPolicy(window=64, enable_regime_detection=True, enable_event_features=False)

        bars = _make_bars(n=100)

        dp.trigger.decide = MagicMock(return_value=(2, 0.75, 0.002))

        action, conf, runway = dp.decide_entry(bars, imbalance=0.0)
        assert action == 2

    def test_entry_without_regime_detector(self):
        """Entry LONG without regime detector → different LOG path (lines 258-262)."""
        dp = DualPolicy(window=64, enable_regime_detection=False, enable_event_features=False)

        bars = _make_bars(n=100)
        dp.trigger.decide = MagicMock(return_value=(1, 0.9, 0.005))

        action, conf, runway = dp.decide_entry(bars, imbalance=0.0)
        assert action == 1
        assert dp.predicted_runway == 0.005


# ---------------------------------------------------------------------------
# decide_exit CLOSE action LOG (line 308)
# ---------------------------------------------------------------------------

class TestDecideExitClose:
    def test_close_signal_logged(self):
        """When harvester returns CLOSE (action=1), the LOG is emitted."""
        dp = DualPolicy(window=64, enable_event_features=False)
        dp.on_entry(direction=1, entry_price=100000.0, entry_time=dt.datetime.now())

        bars = _make_bars(n=100)

        # Force harvester to return CLOSE
        dp.harvester.decide = MagicMock(return_value=(1, 0.95))

        action, conf = dp.decide_exit(bars, current_price=100050.0, imbalance=0.0)
        assert action == 1
        assert conf == 0.95


# ---------------------------------------------------------------------------
# _update_mfe_mae exception handling (lines 386-387, 390-391)
# ---------------------------------------------------------------------------

class TestUpdateMfeMaeExceptions:
    def test_non_float_current_price_handled(self):
        """Non-convertible current_price → cp = 0.0 (lines 386-387)."""
        dp = DualPolicy(window=64)
        dp.on_entry(direction=1, entry_price=100.0, entry_time=dt.datetime.now())

        # Pass an object that cannot be converted to float
        dp._update_mfe_mae(object())
        # Should not crash; mfe/mae should be based on cp=0.0 vs ep=100.0
        # For LONG: profit = 0.0 - 100.0 = -100.0 → mae = 100.0
        assert dp.mae == 100.0

    def test_non_float_entry_price_handled(self):
        """Non-convertible entry_price → ep = 0.0 (lines 390-391)."""
        dp = DualPolicy(window=64)
        dp.current_position = 1
        # Use a numeric-like object that passes SafeMath.is_zero (not zero)
        # but fails float() conversion for the ep variable.
        # Since entry_price goes through SafeMath.is_zero first, we need it to
        # be non-zero (pass the guard) but then fail float() conversion.
        # A custom class that has __abs__ and __lt__ but no __float__:
        class BadFloat:
            def __abs__(self):
                return 1.0  # Non-zero → passes is_zero check
            def __lt__(self, other):
                return False  # abs(x) < eps → False → not zero
            def __float__(self):
                raise ValueError("cannot convert")
        dp.entry_price = BadFloat()

        dp._update_mfe_mae(100.0)
        # cp=100.0, ep=0.0 → profit=100.0 → mfe=100.0
        assert dp.mfe == 100.0


# ---------------------------------------------------------------------------
# Training-enabled paths (lines 590, 616, 646, 683)
# ---------------------------------------------------------------------------

class TestTrainingEnabledPaths:
    def test_add_trigger_experience_training_enabled(self):
        """add_trigger_experience with training enabled → calls trigger.add_experience (line 616)."""
        dp = DualPolicy(window=64, enable_training=True, enable_event_features=False)

        state = np.zeros((64, 7), dtype=np.float32)
        dp.trigger.add_experience = MagicMock()

        dp.add_trigger_experience(state, action=1, reward=0.5, next_state=state, done=True)

        dp.trigger.add_experience.assert_called_once()

    def test_add_harvester_experience_training_enabled(self):
        """add_harvester_experience with training enabled → calls harvester.add_experience (line 646)."""
        dp = DualPolicy(window=64, enable_training=True, enable_event_features=False)

        state = np.zeros((64, 10), dtype=np.float32)
        dp.harvester.add_experience = MagicMock()

        dp.add_harvester_experience(state, action=0, reward=0.1, next_state=state, done=False)

        dp.harvester.add_experience.assert_called_once()

    def test_sync_replay_buffer_regime_with_training(self):
        """_sync_replay_buffer_regime with training-enabled buffer (line 590)."""
        dp = DualPolicy(window=64, enable_training=True, enable_event_features=False)

        # Setup mock buffer on trigger with set_current_regime
        mock_buffer = MagicMock()
        mock_buffer.set_current_regime = MagicMock()
        dp.trigger.buffer = mock_buffer
        dp.trigger.enable_training = True

        dp.current_regime = "TRENDING"
        dp._sync_replay_buffer_regime()

        mock_buffer.set_current_regime.assert_called()

    def test_train_step_with_metrics_logged(self):
        """train_step when both agents return metrics → LOG emitted (line 683)."""
        dp = DualPolicy(window=64, enable_training=True, enable_event_features=False)

        dp.trigger.train_step = MagicMock(return_value={"loss": 0.01, "mean_td_error": 0.005})
        dp.harvester.train_step = MagicMock(return_value={"loss": 0.02, "mean_td_error": 0.01})

        result = dp.train_step()
        assert result["trigger"]["loss"] == 0.01
        assert result["harvester"]["loss"] == 0.02
