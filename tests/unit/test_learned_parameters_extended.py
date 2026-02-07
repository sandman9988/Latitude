"""Extended tests for src.persistence.learned_parameters.

Covers: save failure path, load version mismatch, load corrupt CRC,
check_staleness returns stale params, get_summary details.
"""

import json
import time

import pytest

from src.persistence.learned_parameters import (
    AdaptiveParam,
    InstrumentParameters,
    LearnedParametersManager,
)


# ---------------------------------------------------------------------------
# Save / load edge cases
# ---------------------------------------------------------------------------
class TestSaveLoadEdgeCases:
    def test_save_failure_path(self, tmp_path, monkeypatch):
        """If atomic_persist.save_json returns False, no crash."""
        mgr = LearnedParametersManager(persistence_path=tmp_path / "lp.json")
        mgr.get_instrument("BTCUSD")
        monkeypatch.setattr(mgr.atomic_persist, "save_json", lambda *a, **kw: False)
        mgr.save()  # Should log error but not raise

    def test_save_exception_handled(self, tmp_path, monkeypatch):
        """If atomic_persist.save_json raises, no crash."""
        mgr = LearnedParametersManager(persistence_path=tmp_path / "lp.json")
        monkeypatch.setattr(mgr.atomic_persist, "save_json", lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")))
        mgr.save()  # Should not raise

    def test_load_version_mismatch(self, tmp_path, monkeypatch):
        """Version != 1.0 should return False."""
        mgr = LearnedParametersManager(persistence_path=tmp_path / "lp.json")
        bad_data = {"version": "2.0", "instruments": {}}
        monkeypatch.setattr(mgr.atomic_persist, "load_json", lambda *a, **kw: bad_data)
        # Write a file so path.exists() is True
        (tmp_path / "lp.json").write_text("{}")
        assert mgr.load() is False

    def test_load_corrupt_crc_returns_false(self, tmp_path, monkeypatch):
        """load_json returns None on CRC fail → load should return False."""
        mgr = LearnedParametersManager(persistence_path=tmp_path / "lp.json")
        (tmp_path / "lp.json").write_text("{}")
        monkeypatch.setattr(mgr.atomic_persist, "load_json", lambda *a, **kw: None)
        assert mgr.load() is False

    def test_load_exception_returns_false(self, tmp_path, monkeypatch):
        """Exception during load returns False."""
        mgr = LearnedParametersManager(persistence_path=tmp_path / "lp.json")
        (tmp_path / "lp.json").write_text("{}")
        monkeypatch.setattr(mgr.atomic_persist, "load_json", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError))
        assert mgr.load() is False


# ---------------------------------------------------------------------------
# check_staleness
# ---------------------------------------------------------------------------
class TestCheckStaleness:
    def test_no_stale_params(self, tmp_path):
        mgr = LearnedParametersManager(persistence_path=tmp_path / "lp.json")
        mgr.get_instrument("BTCUSD")
        # With huge threshold nothing should be stale
        result = mgr.check_staleness(threshold_seconds=1e9)
        assert result == {}

    def test_stale_params_detected(self, tmp_path, monkeypatch):
        """With threshold 0 everything is stale."""
        mgr = LearnedParametersManager(persistence_path=tmp_path / "lp.json")
        inst = mgr.get_instrument("BTCUSD")
        # Force param last_update_time far in the past
        for p in inst.params.values():
            p.last_update_time = 0
        result = mgr.check_staleness(threshold_seconds=1.0)
        assert len(result) > 0
        # Should contain the instrument key
        assert any("BTCUSD" in k for k in result)


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------
class TestGetSummary:
    def test_empty_manager(self, tmp_path):
        mgr = LearnedParametersManager(persistence_path=tmp_path / "lp.json")
        # Don't create any instruments
        mgr.instruments.clear()
        summary = mgr.get_summary()
        assert summary["num_instruments"] == 0
        assert summary["total_parameters"] == 0
        assert summary["parameters_per_instrument"] == 0

    def test_multiple_instruments(self, tmp_path):
        mgr = LearnedParametersManager(persistence_path=tmp_path / "lp.json")
        mgr.get_instrument("BTCUSD")
        mgr.get_instrument("ETHUSD")
        summary = mgr.get_summary()
        assert summary["num_instruments"] == 2
        assert "BTCUSD_M1_default" in summary["instruments"]
        assert "ETHUSD_M1_default" in summary["instruments"]


# ---------------------------------------------------------------------------
# AdaptiveParam extended
# ---------------------------------------------------------------------------
class TestAdaptiveParamExtended:
    def test_large_positive_gradient_stays_in_bounds(self):
        p = AdaptiveParam(name="x", value=0.5, min_bound=0.0, max_bound=1.0, learning_rate=10.0, momentum=0.0)
        for _ in range(50):
            p.update(100.0)
        assert p.value <= 1.0
        assert p.value >= 0.0

    def test_large_negative_gradient_stays_in_bounds(self):
        p = AdaptiveParam(name="x", value=0.5, min_bound=0.0, max_bound=1.0, learning_rate=10.0, momentum=0.0)
        for _ in range(50):
            p.update(-100.0)
        assert p.value >= 0.0
        assert p.value <= 1.0

    def test_update_count_increments(self):
        p = AdaptiveParam(name="x", value=0.5, min_bound=0.0, max_bound=1.0)
        p.update(0.1)
        p.update(0.1)
        assert p.update_count == 2


# ---------------------------------------------------------------------------
# InstrumentParameters extended
# ---------------------------------------------------------------------------
class TestInstrumentParametersExtended:
    def test_is_stale_true_for_old_param(self):
        inst = InstrumentParameters("TEST")
        inst.add_param("p1", 1.0, 0.0, 2.0)
        inst.params["p1"].last_update_time = 0  # epoch
        assert inst.is_stale("p1", threshold_seconds=1.0) is True

    def test_is_stale_false_for_fresh_param(self):
        inst = InstrumentParameters("TEST")
        inst.add_param("p1", 1.0, 0.0, 2.0)
        assert inst.is_stale("p1", threshold_seconds=86400) is False

    def test_reset_velocity_missing_param_no_crash(self):
        inst = InstrumentParameters("TEST")
        inst.reset_velocity("nonexistent")  # Should not raise
