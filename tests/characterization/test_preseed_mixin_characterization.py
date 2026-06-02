"""Characterization tests for the extracted TFAgentPreseedMixin.

Pins the pure preseed math helpers and verifies TFAgent still exposes the
cluster via inheritance after the behaviour-preserving mixin extraction.
"""

import math

import pytest

from src.core.openapi_hub import TFAgent
from src.core.tf_agent_preseed import TFAgentPreseedMixin


class _Bench(TFAgentPreseedMixin):
    timeframe_minutes = 5


@pytest.fixture
def agent():
    return _Bench()


def _bar(close, high=None, low=None):
    high = close if high is None else high
    low = close if low is None else low
    return (0, close, high, low, close, 0)


def test_tfagent_inherits_preseed_mixin():
    assert issubclass(TFAgent, TFAgentPreseedMixin)
    for name in (
        "_compute_preseed_vol",
        "_preseed_harvester_from_bars",
        "_preseed_one_harvester_entry",
        "_preseed_mfe_mae",
        "_build_preseed_harvester_state",
        "_add_preseed_hold_experience",
        "_preseed_exit_pnl",
        "_add_preseed_close_experience",
        "_preseed_trigger_buffer",
    ):
        assert hasattr(TFAgent, name)


def test_preseed_constants_unchanged():
    assert TFAgent._PRESEED_STOP_PCT == 0.003
    assert TFAgent._PRESEED_TARGET_PCT == 0.002
    assert TFAgent._PRESEED_MAX_HOLD == 20


def test_compute_preseed_vol_short_window_default(agent):
    assert agent._compute_preseed_vol([_bar(1.0)], 0) == 0.005
    assert agent._compute_preseed_vol([_bar(1.0), _bar(1.0)], 1) == 0.005


def test_compute_preseed_vol_positive_for_varied_closes(agent):
    bars = [_bar(c) for c in (1.0, 1.01, 0.99, 1.02, 0.98, 1.03)]
    v = agent._compute_preseed_vol(bars, len(bars))
    assert v > 0.0
    assert math.isfinite(v)


def test_preseed_mfe_mae_long(agent):
    cur_mfe, cur_mae = agent._preseed_mfe_mae(1, 100.0, 103.0, 98.0, 0.0, 0.0)
    assert cur_mfe == 3.0
    assert cur_mae == 2.0


def test_preseed_mfe_mae_short(agent):
    cur_mfe, cur_mae = agent._preseed_mfe_mae(-1, 100.0, 103.0, 98.0, 0.0, 0.0)
    assert cur_mfe == 2.0
    assert cur_mae == 3.0


def test_preseed_mfe_mae_monotonic_with_prev(agent):
    cur_mfe, cur_mae = agent._preseed_mfe_mae(1, 100.0, 100.5, 99.5, 5.0, 4.0)
    assert cur_mfe == 5.0
    assert cur_mae == 4.0


def test_preseed_exit_pnl_target_hit_long(agent):
    pnl = agent._preseed_exit_pnl(1, 100.5, 99.5, 100.2, 100.0, 99.0, 100.2, 1.0, 0.2, 1)
    assert pnl == 0.2


def test_preseed_exit_pnl_stop_hit_long(agent):
    pnl = agent._preseed_exit_pnl(1, 100.1, 98.9, 99.0, 100.0, 99.0, 102.0, 1.0, 2.0, 1)
    assert pnl == -1.0


def test_preseed_exit_pnl_max_hold_close(agent):
    pnl = agent._preseed_exit_pnl(
        1, 100.1, 99.9, 100.05, 100.0, 95.0, 105.0, 5.0, 5.0, agent._PRESEED_MAX_HOLD,
    )
    assert pnl == pytest.approx(0.05)


def test_preseed_exit_pnl_no_exit_returns_none(agent):
    pnl = agent._preseed_exit_pnl(1, 100.1, 99.9, 100.05, 100.0, 95.0, 105.0, 5.0, 5.0, 1)
    assert pnl is None


def test_add_preseed_close_experience_none_state(agent):
    assert agent._add_preseed_close_experience(None, 1.0, 1.0) == 0
