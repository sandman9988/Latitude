"""Tests for src.persistence.learned_parameters – AdaptiveParam, InstrumentParameters, LearnedParametersManager."""

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.persistence.learned_parameters import (
    AdaptiveParam,
    InstrumentParameters,
    LearnedParametersManager,
)


# ---------------------------------------------------------------------------
# AdaptiveParam
# ---------------------------------------------------------------------------

class TestAdaptiveParam:
    def test_init_defaults(self):
        p = AdaptiveParam(name="x", value=0.5, min_bound=0.0, max_bound=1.0)
        assert p.name == "x"
        assert p.value == pytest.approx(0.5)
        assert p.velocity == pytest.approx(0.0)
        assert p.update_count == 0

    def test_update_positive_gradient(self):
        p = AdaptiveParam(name="x", value=0.5, min_bound=0.0, max_bound=1.0, learning_rate=0.1, momentum=0.0)
        new_val = p.update(1.0)
        assert new_val > 0.5
        assert p.update_count == 1

    def test_update_negative_gradient(self):
        p = AdaptiveParam(name="x", value=0.5, min_bound=0.0, max_bound=1.0, learning_rate=0.1, momentum=0.0)
        new_val = p.update(-1.0)
        assert new_val < 0.5

    def test_soft_clamping_stays_in_bounds(self):
        p = AdaptiveParam(name="x", value=0.5, min_bound=0.0, max_bound=1.0, learning_rate=1.0, momentum=0.0)
        # Apply very large gradient
        for _ in range(50):
            p.update(100.0)
        assert p.value <= 1.0  # tanh saturates at bounds
        assert p.value >= 0.0

    def test_momentum_accumulates(self):
        p = AdaptiveParam(name="x", value=0.5, min_bound=0.0, max_bound=1.0, learning_rate=0.01, momentum=0.9)
        p.update(1.0)
        v1 = p.velocity
        p.update(1.0)
        v2 = p.velocity
        assert abs(v2) > abs(v1)  # momentum builds up

    def test_reset_velocity(self):
        p = AdaptiveParam(name="x", value=0.5, min_bound=0.0, max_bound=1.0, learning_rate=0.1, momentum=0.9)
        p.update(1.0)
        assert p.velocity != 0.0
        p.reset_velocity()
        assert p.velocity == pytest.approx(0.0)

    def test_to_dict_roundtrip(self):
        p = AdaptiveParam(name="y", value=0.3, min_bound=0.0, max_bound=1.0)
        d = p.to_dict()
        assert d["name"] == "y"
        p2 = AdaptiveParam.from_dict(d)
        assert p2.name == "y"
        assert p2.value == pytest.approx(0.3)

    def test_update_tracks_time(self):
        p = AdaptiveParam(name="x", value=0.5, min_bound=0.0, max_bound=1.0)
        before = time.time()
        p.update(0.1)
        assert p.last_update_time >= before

    def test_zero_range_no_crash(self):
        p = AdaptiveParam(name="x", value=5.0, min_bound=5.0, max_bound=5.0, learning_rate=0.1, momentum=0.0)
        v = p.update(1.0)
        assert v == pytest.approx(5.0)  # mid + tanh(0)*0 = 5


# ---------------------------------------------------------------------------
# InstrumentParameters
# ---------------------------------------------------------------------------

class TestInstrumentParameters:
    def test_add_and_get(self):
        ip = InstrumentParameters("BTCUSD")
        ip.add_param("spread", 1.0, 0.0, 5.0)
        assert ip.get("spread") == pytest.approx(1.0)

    def test_get_missing_with_default(self):
        ip = InstrumentParameters("BTCUSD")
        assert ip.get("missing", default=42.0) == pytest.approx(42.0)

    def test_get_missing_no_default_raises(self):
        ip = InstrumentParameters("BTCUSD")
        with pytest.raises(KeyError):
            ip.get("missing")

    def test_update_changes_value(self):
        ip = InstrumentParameters("BTCUSD")
        ip.add_param("spread", 1.0, 0.0, 5.0, learning_rate=0.5, momentum=0.0)
        new_val = ip.update("spread", 1.0)
        assert new_val != 1.0

    def test_update_missing_raises(self):
        ip = InstrumentParameters("BTCUSD")
        with pytest.raises(KeyError):
            ip.update("nope", 1.0)

    def test_reset_velocity_single(self):
        ip = InstrumentParameters("BTCUSD")
        ip.add_param("a", 1.0, 0.0, 5.0)
        ip.update("a", 1.0)
        ip.reset_velocity("a")
        assert ip.params["a"].velocity == pytest.approx(0.0)

    def test_reset_velocity_all(self):
        ip = InstrumentParameters("BTCUSD")
        ip.add_param("a", 1.0, 0.0, 5.0)
        ip.add_param("b", 2.0, 0.0, 5.0)
        ip.update("a", 1.0)
        ip.update("b", 1.0)
        ip.reset_velocity()
        assert ip.params["a"].velocity == pytest.approx(0.0)
        assert ip.params["b"].velocity == pytest.approx(0.0)

    def test_staleness(self):
        ip = InstrumentParameters("BTCUSD")
        ip.add_param("x", 1.0, 0.0, 5.0)
        # Just created → not stale with large threshold
        assert not ip.is_stale("x", threshold_seconds=3600)
        # Stale with zero threshold
        assert ip.is_stale("x", threshold_seconds=0)

    def test_staleness_missing_param(self):
        ip = InstrumentParameters("BTCUSD")
        assert ip.get_staleness("nope") == float("inf")

    def test_to_dict_from_dict_roundtrip(self):
        ip = InstrumentParameters("XAUUSD", "M15", "broker1")
        ip.add_param("a", 1.0, 0.0, 5.0)
        d = ip.to_dict()
        ip2 = InstrumentParameters.from_dict(d)
        assert ip2.symbol == "XAUUSD"
        assert ip2.timeframe == "M15"
        assert "a" in ip2.params


# ---------------------------------------------------------------------------
# LearnedParametersManager
# ---------------------------------------------------------------------------

class TestLearnedParametersManager:
    @pytest.fixture()
    def manager(self, tmp_path):
        path = tmp_path / "params.json"
        return LearnedParametersManager(persistence_path=path)

    def test_init_creates_dir(self, tmp_path):
        path = tmp_path / "sub" / "params.json"
        m = LearnedParametersManager(persistence_path=path)
        assert path.parent.exists()

    def test_get_instrument_creates_defaults(self, manager):
        inst = manager.get_instrument("BTCUSD")
        assert inst.symbol == "BTCUSD"
        assert len(inst.params) > 0

    def test_get_instrument_cached(self, manager):
        i1 = manager.get_instrument("BTCUSD")
        i2 = manager.get_instrument("BTCUSD")
        assert i1 is i2

    def test_get_value(self, manager):
        val = manager.get("BTCUSD", "base_position_size")
        assert val == pytest.approx(0.10)  # default spec

    def test_update_value(self, manager):
        old = manager.get("BTCUSD", "base_position_size")
        new = manager.update("BTCUSD", "base_position_size", 0.5)
        assert new != old

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "params.json"
        m1 = LearnedParametersManager(persistence_path=path)
        m1.update("BTCUSD", "base_position_size", 0.5)
        m1.save()
        assert path.exists()

        m2 = LearnedParametersManager(persistence_path=path)
        # Should load on init
        val = m2.get("BTCUSD", "base_position_size")
        # After update the value should differ from default
        assert val != 0.10

    def test_load_nonexistent(self, tmp_path):
        path = tmp_path / "nope.json"
        m = LearnedParametersManager(persistence_path=path)
        assert not m.load()

    def test_check_staleness(self, manager):
        # Freshly created → not stale with large threshold
        stale = manager.check_staleness(threshold_seconds=9999)
        assert stale == {}

    def test_check_staleness_zero_threshold(self, manager):
        _ = manager.get_instrument("BTCUSD")
        stale = manager.check_staleness(threshold_seconds=0)
        # All params should be "stale" with zero threshold
        assert len(stale) > 0

    def test_get_summary(self, manager):
        _ = manager.get_instrument("BTCUSD")
        summary = manager.get_summary()
        assert summary["num_instruments"] == 1
        assert summary["total_parameters"] > 0
