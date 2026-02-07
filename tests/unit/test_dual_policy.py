"""
Tests for DualPolicy - Orchestrates Trigger and Harvester agents.

Covers:
- Initialization with various configurations
- decide_entry() state building and delegation
- decide_exit() with MFE/MAE tracking
- on_entry() / on_exit() position lifecycle
- _update_mfe_mae() for LONG and SHORT
- _build_state() feature engineering
- _ingest_price_for_regime() / _sync_replay_buffer_regime()
- Online learning methods (add_experience, train_step, get_training_stats)
- get_position_metrics()
"""

import datetime as dt
import logging
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.dual_policy import DualPolicy, MIN_BARS_FOR_FEATURES

LOG = logging.getLogger(__name__)


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
# Initialization
# ---------------------------------------------------------------------------

class TestDualPolicyInit:

    def test_default_init(self):
        dp = DualPolicy(window=64)
        assert dp.current_position == 0
        assert dp.entry_price == pytest.approx(0.0)
        assert dp.mfe == pytest.approx(0.0)
        assert dp.mae == pytest.approx(0.0)
        assert dp.ticks_held == 0
        assert dp.trigger is not None
        assert dp.harvester is not None
        assert dp.enable_training is False

    def test_init_with_training_enabled(self):
        dp = DualPolicy(window=32, enable_training=True)
        assert dp.enable_training is True
        assert dp.trigger.enable_training is True
        assert dp.harvester.enable_training is True

    def test_init_event_features_enabled(self):
        dp = DualPolicy(window=64, enable_event_features=True)
        assert dp.enable_event_features is True
        assert dp.event_feature_count == 6

    def test_init_event_features_disabled(self):
        dp = DualPolicy(window=64, enable_event_features=False)
        assert dp.enable_event_features is False
        assert dp.event_feature_count == 0

    def test_init_with_path_geometry(self):
        mock_geom = MagicMock()
        dp = DualPolicy(window=64, path_geometry=mock_geom)
        assert dp.path_geometry is mock_geom

    def test_init_symbol_timeframe(self):
        dp = DualPolicy(symbol="XAUUSD", timeframe="M5", broker="pepperstone")
        assert dp.symbol == "XAUUSD"
        assert dp.timeframe == "M5"
        assert dp.broker == "pepperstone"


# ---------------------------------------------------------------------------
# Position lifecycle
# ---------------------------------------------------------------------------

class TestPositionLifecycle:

    def test_on_entry_long(self):
        dp = DualPolicy(window=64)
        dp.on_entry(direction=1, entry_price=50000.0, entry_time=dt.datetime.now())
        assert dp.current_position == 1
        assert dp.entry_price == pytest.approx(50000.0)
        assert dp.mfe == pytest.approx(0.0)
        assert dp.mae == pytest.approx(0.0)
        assert dp.ticks_held == 0

    def test_on_entry_short(self):
        dp = DualPolicy(window=64)
        dp.on_entry(direction=-1, entry_price=50000.0, entry_time=dt.datetime.now())
        assert dp.current_position == -1

    def test_on_exit_resets_state(self):
        dp = DualPolicy(window=64)
        dp.on_entry(direction=1, entry_price=50000.0, entry_time=dt.datetime.now())
        dp.mfe = 100.0
        dp.mae = 50.0
        dp.ticks_held = 42
        dp.predicted_runway = 0.003

        # Mock agent methods
        dp.trigger.update_from_trade = MagicMock()
        dp.harvester.update_from_trade = MagicMock()

        dp.on_exit(exit_price=50100.0, capture_ratio=0.8, was_wtl=False)

        assert dp.current_position == 0
        assert dp.entry_price == pytest.approx(0.0)
        assert dp.mfe == pytest.approx(0.0)
        assert dp.mae == pytest.approx(0.0)
        assert dp.ticks_held == 0
        assert dp.predicted_runway == pytest.approx(0.0)
        dp.trigger.update_from_trade.assert_called_once()
        dp.harvester.update_from_trade.assert_called_once()

    def test_get_position_metrics(self):
        dp = DualPolicy(window=64)
        dp.on_entry(direction=1, entry_price=50000.0, entry_time=dt.datetime.now())
        dp.mfe = 200.0
        dp.mae = 50.0
        dp.ticks_held = 10

        metrics = dp.get_position_metrics()
        assert metrics["mfe"] == pytest.approx(200.0)
        assert metrics["mae"] == pytest.approx(50.0)
        assert metrics["ticks_held"] == 10
        assert metrics["entry_price"] == pytest.approx(50000.0)
        assert metrics["current_position"] == 1


# ---------------------------------------------------------------------------
# MFE / MAE tracking
# ---------------------------------------------------------------------------

class TestMfeMae:

    def test_update_mfe_mae_long_profit(self):
        dp = DualPolicy(window=64)
        dp.on_entry(direction=1, entry_price=100.0, entry_time=dt.datetime.now())
        dp._update_mfe_mae(110.0)  # +10 profit
        assert dp.mfe == pytest.approx(10.0)
        assert dp.mae == pytest.approx(0.0)

    def test_update_mfe_mae_long_loss(self):
        dp = DualPolicy(window=64)
        dp.on_entry(direction=1, entry_price=100.0, entry_time=dt.datetime.now())
        dp._update_mfe_mae(90.0)  # -10 loss
        assert dp.mfe == pytest.approx(0.0)
        assert dp.mae == pytest.approx(10.0)

    def test_update_mfe_mae_short_profit(self):
        dp = DualPolicy(window=64)
        dp.on_entry(direction=-1, entry_price=100.0, entry_time=dt.datetime.now())
        dp._update_mfe_mae(90.0)  # Short profit
        assert dp.mfe == pytest.approx(10.0)
        assert dp.mae == pytest.approx(0.0)

    def test_update_mfe_mae_short_loss(self):
        dp = DualPolicy(window=64)
        dp.on_entry(direction=-1, entry_price=100.0, entry_time=dt.datetime.now())
        dp._update_mfe_mae(110.0)  # Short loss
        assert dp.mfe == pytest.approx(0.0)
        assert dp.mae == pytest.approx(10.0)

    def test_update_mfe_mae_zero_entry_price(self):
        dp = DualPolicy(window=64)
        dp.entry_price = 0.0
        dp.current_position = 1
        dp._update_mfe_mae(100.0)
        # Should not crash, should not update
        assert dp.mfe == pytest.approx(0.0)
        assert dp.mae == pytest.approx(0.0)

    def test_mfe_mae_tracks_maximum(self):
        dp = DualPolicy(window=64)
        dp.on_entry(direction=1, entry_price=100.0, entry_time=dt.datetime.now())
        dp._update_mfe_mae(110.0)  # MFE=10
        dp._update_mfe_mae(105.0)  # MFE stays 10
        dp._update_mfe_mae(95.0)   # MAE now 5
        assert dp.mfe == pytest.approx(10.0)
        assert dp.mae == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# State building
# ---------------------------------------------------------------------------

class TestBuildState:

    def test_insufficient_bars_returns_zeros(self):
        dp = DualPolicy(window=64, enable_event_features=False)
        bars = _make_bars(n=30)  # Less than MIN_BARS_FOR_FEATURES
        state = dp._build_state(bars, imbalance=0.0, vpin_z=0.0, depth_ratio=1.0)
        assert state.shape == (64, 7)
        assert np.allclose(state, 0.0)

    def test_sufficient_bars_returns_features(self):
        dp = DualPolicy(window=64, enable_event_features=False)
        bars = _make_bars(n=100)
        state = dp._build_state(bars, imbalance=0.1, vpin_z=0.5, depth_ratio=1.2)
        assert state.shape == (64, 7)
        # Normalized data should have roughly zero mean
        assert not np.allclose(state, 0.0)

    def test_build_state_with_event_features(self):
        dp = DualPolicy(window=64, enable_event_features=True)
        bars = _make_bars(n=100)
        event_feats = {
            "london_active": 1.0,
            "ny_active": 0.0,
            "tokyo_active": 0.0,
            "london_ny_overlap": 0.0,
            "rollover_proximity_norm": 0.2,
            "week_progress": 0.3,
        }
        state = dp._build_state(bars, 0.0, 0.0, 1.0, event_features=event_feats)
        assert state.shape == (64, 13)  # 7 base + 6 event

    def test_build_state_with_event_features_none(self):
        """Event features enabled but no event data should use defaults."""
        dp = DualPolicy(window=64, enable_event_features=True)
        bars = _make_bars(n=100)
        state = dp._build_state(bars, 0.0, 0.0, 1.0, event_features=None)
        assert state.shape == (64, 13)

    def test_build_state_with_geometry(self):
        mock_geom = MagicMock()
        mock_geom.update.return_value = {
            "efficiency": 0.8,
            "gamma": 0.1,
            "jerk": 0.05,
            "runway": 0.6,
            "feasibility": 0.7,
        }
        dp = DualPolicy(window=64, path_geometry=mock_geom, enable_event_features=False)
        bars = _make_bars(n=100)
        state = dp._build_state(bars, 0.0, 0.0, 1.0)
        assert state.shape == (64, 12)  # 7 base + 5 geometry


# ---------------------------------------------------------------------------
# Entry decision
# ---------------------------------------------------------------------------

class TestDecideEntry:

    def test_decide_entry_returns_tuple_of_three(self):
        dp = DualPolicy(window=64, enable_event_features=False)
        bars = _make_bars(n=100)
        result = dp.decide_entry(bars, imbalance=0.0)
        assert len(result) == 3
        action, conf, runway = result
        assert action in [0, 1, 2]
        assert 0.0 <= conf <= 1.0
        assert runway >= 0.0

    def test_decide_entry_with_few_bars(self):
        dp = DualPolicy(window=64, enable_event_features=False)
        bars = _make_bars(n=10)
        action, conf, runway = dp.decide_entry(bars, imbalance=0.0)
        # Should still return valid results (zeros state → likely NO_ENTRY)
        assert action in [0, 1, 2]


# ---------------------------------------------------------------------------
# Exit decision
# ---------------------------------------------------------------------------

class TestDecideExit:

    def test_decide_exit_returns_tuple_of_two(self):
        dp = DualPolicy(window=64, enable_event_features=False)
        dp.on_entry(direction=1, entry_price=100000.0, entry_time=dt.datetime.now())
        bars = _make_bars(n=100)
        result = dp.decide_exit(bars, current_price=100050.0, imbalance=0.0)
        assert len(result) == 2
        action, conf = result
        assert action in [0, 1]
        assert 0.0 <= conf <= 1.0

    def test_decide_exit_increments_ticks_held(self):
        dp = DualPolicy(window=64, enable_event_features=False)
        dp.on_entry(direction=1, entry_price=100000.0, entry_time=dt.datetime.now())
        bars = _make_bars(n=100)
        dp.decide_exit(bars, current_price=100050.0, imbalance=0.0)
        assert dp.ticks_held == 1
        dp.decide_exit(bars, current_price=100060.0, imbalance=0.0)
        assert dp.ticks_held == 2

    def test_decide_exit_updates_mfe(self):
        dp = DualPolicy(window=64, enable_event_features=False)
        dp.on_entry(direction=1, entry_price=100000.0, entry_time=dt.datetime.now())
        bars = _make_bars(n=100)
        dp.decide_exit(bars, current_price=100200.0, imbalance=0.0)
        assert dp.mfe == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

class TestRegimeDetection:

    def test_ingest_price_no_regime_detector(self):
        dp = DualPolicy(window=64, enable_regime_detection=False)
        # Should not crash
        dp._ingest_price_for_regime(50000.0)

    def test_ingest_price_with_regime_detector(self):
        dp = DualPolicy(window=64, enable_regime_detection=True)
        if dp.regime_detector is not None:
            dp._ingest_price_for_regime(50000.0)
            # Just verify no crash

    def test_sync_replay_buffer_regime(self):
        dp = DualPolicy(window=64)
        dp.current_regime = "TRENDING"
        dp._sync_replay_buffer_regime()
        # Verify the enum was set
        assert dp.current_regime_enum is not None


# ---------------------------------------------------------------------------
# Online learning
# ---------------------------------------------------------------------------

class TestOnlineLearning:

    def test_add_trigger_experience_training_disabled(self):
        dp = DualPolicy(window=64, enable_training=False)
        state = np.zeros((64, 7), dtype=np.float32)
        # Should not crash
        dp.add_trigger_experience(state, action=1, reward=0.5, next_state=state, done=True)

    def test_add_harvester_experience_training_disabled(self):
        dp = DualPolicy(window=64, enable_training=False)
        state = np.zeros((64, 10), dtype=np.float32)
        dp.add_harvester_experience(state, action=0, reward=0.1, next_state=state, done=False)

    def test_train_step_training_disabled(self):
        dp = DualPolicy(window=64, enable_training=False)
        result = dp.train_step()
        assert result == {"trigger": None, "harvester": None}

    def test_get_training_stats(self):
        dp = DualPolicy(window=64, enable_training=False)
        stats = dp.get_training_stats()
        assert "trigger" in stats
        assert "harvester" in stats
        assert stats["enable_training"] is False

    def test_get_training_stats_enabled(self):
        dp = DualPolicy(window=64, enable_training=True)
        stats = dp.get_training_stats()
        assert stats["enable_training"] is True

    def test_train_step_training_enabled_insufficient_data(self):
        dp = DualPolicy(window=32, enable_training=True, enable_event_features=False)
        result = dp.train_step()
        # Should return None for both agents (no experiences yet)
        assert result["trigger"] is None
        assert result["harvester"] is None
