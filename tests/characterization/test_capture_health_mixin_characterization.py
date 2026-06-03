"""Characterization tests for the extracted TFAgentCaptureHealthMixin.

Pins the threshold tighten/relax/emergency behaviour and verifies TFAgent still
exposes the cluster via inheritance after the behaviour-preserving extraction.
"""

import types

import pytest

from src.core.openapi_hub import TFAgent
from src.core.tf_agent_capture_health import TFAgentCaptureHealthMixin


class _Harv:
    def __init__(self):
        self.trailing_stop_activation_pct = 0.25
        self.trailing_stop_distance_pct = 0.12
        self.capture_decay_threshold = 0.35
        self.capture_decay_min_mfe_pct = 0.10

    def _get_timeframe_scale(self):
        return 1.0


class _Bench(TFAgentCaptureHealthMixin):
    def __init__(self):
        self.symbol = "XAUUSD"
        self.tf_label = "M5"
        self.timeframe_minutes = 5
        self._capture_ema = 0.5
        self._capture_ema_n = 0
        self._capture_last_intervention = 0.0
        self.harv = _Harv()
        self.policy = types.SimpleNamespace(harvester=self.harv)


@pytest.fixture
def agent():
    return _Bench()


def test_tfagent_inherits_capture_health_mixin():
    assert issubclass(TFAgent, TFAgentCaptureHealthMixin)
    for name in (
        "_check_capture_health",
        "_apply_capture_tighten",
        "_apply_capture_emergency_reset",
        "_apply_capture_relax",
    ):
        assert hasattr(TFAgent, name)


def test_capture_min_samples_table_unchanged():
    assert TFAgent._CAPTURE_MIN_SAMPLES == {1: 15, 5: 10, 15: 7, 30: 5, 60: 4, 240: 3}


def test_apply_capture_tighten_reduces_trail_and_raises_decay(agent):
    agent._apply_capture_tighten(agent.harv, factor=0.75)
    assert agent.harv.trailing_stop_activation_pct == pytest.approx(0.25 * 0.75)
    assert agent.harv.trailing_stop_distance_pct == pytest.approx(0.12 * 0.75)
    assert agent.harv.capture_decay_threshold == pytest.approx(0.35 + 0.25 * 0.40)


def test_apply_capture_tighten_respects_floors(agent):
    agent.harv.trailing_stop_activation_pct = 0.05
    agent.harv.trailing_stop_distance_pct = 0.02
    agent._apply_capture_tighten(agent.harv, factor=0.10)
    assert agent.harv.trailing_stop_activation_pct == pytest.approx(max(0.03, 0.25 * 0.40))
    assert agent.harv.trailing_stop_distance_pct == pytest.approx(max(0.01, 0.12 * 0.30))


def test_apply_capture_emergency_reset_halves_defaults(agent):
    agent._apply_capture_emergency_reset(agent.harv)
    assert agent.harv.trailing_stop_activation_pct == pytest.approx(0.25 * 0.50)
    assert agent.harv.trailing_stop_distance_pct == pytest.approx(0.12 * 0.50)
    assert agent.harv.capture_decay_threshold == 0.50
    assert agent.harv.capture_decay_min_mfe_pct == pytest.approx(0.10)


def test_apply_capture_relax_loosens_but_floors_decay(agent):
    agent.harv.trailing_stop_activation_pct = 0.10
    agent.harv.trailing_stop_distance_pct = 0.05
    agent.harv.capture_decay_threshold = 0.35
    agent._apply_capture_relax(agent.harv, factor=0.97)
    assert agent.harv.trailing_stop_activation_pct == pytest.approx(0.10 / 0.97)
    assert agent.harv.capture_decay_threshold == pytest.approx(0.35)


def test_check_capture_health_large_delta_immediate_tighten(agent):
    before = agent.harv.trailing_stop_activation_pct
    agent._check_capture_health(capture_ratio=0.0, mfe=100.0, entry_price=100.0)
    assert agent.harv.trailing_stop_activation_pct < before
    assert agent._capture_last_intervention > 0.0


def test_check_capture_health_no_harvester_safe(agent):
    agent.policy = types.SimpleNamespace(harvester=None)
    agent._check_capture_health(capture_ratio=0.5, mfe=1.0, entry_price=100.0)
    assert agent._capture_ema_n == 1


def test_check_capture_health_waits_for_min_samples(agent):
    agent._check_capture_health(capture_ratio=-1.0, mfe=0.0, entry_price=100.0)
    assert agent.harv.trailing_stop_activation_pct == 0.25
    assert agent._capture_last_intervention == 0.0
