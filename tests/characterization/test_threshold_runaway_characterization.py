"""Characterization tests for the adaptive entry-threshold feedback loop.

These tests pin the *current* behavior of ``TFAgent._update_risk_feedback_thresholds``
so that the Phase 1 runaway fix can be verified as an intentional behavior change.

Documented failure mode (AGENTS.md "Known Failure Modes"): the persisted
``entry_confidence_threshold`` is used as ``_base_floor`` on every call, and the
per-call cap is ``_base_floor + 0.10``. Because each save writes
``max(_base_floor, dynamic_floor)`` back to the same parameter, the baseline ratchets
upward every save interval during a losing streak, compounding well past the intended
single-step cap until it pins at the parameter's hard maximum (0.9).
"""

from __future__ import annotations

from types import SimpleNamespace

from src.core.openapi_hub import TFAgent
from src.persistence.learned_parameters import LearnedParametersManager

_SYMBOL = "XAUUSD"
_TF_LABEL = "M5"
_TF_MIN = 5
_PARAM = "entry_confidence_threshold"
_PARAM_DEFAULT = 0.6
_SINGLE_STEP_CAP = _PARAM_DEFAULT + 0.10
_PARAM_MAX = 0.9


def _agent(tmp_path) -> TFAgent:
    agent = TFAgent.__new__(TFAgent)
    agent.symbol = _SYMBOL
    agent.tf_label = _TF_LABEL
    agent.timeframe_minutes = _TF_MIN
    agent._win_rate_ema = 0.5
    agent._win_rate_ema_n = 0
    agent._entry_conf_dynamic_floor = 0.0
    agent._exit_conf_dynamic_floor = 0.0
    agent._ddqn_exit_win_ema = 0.5
    agent._ddqn_exit_n = 0
    agent.total_trades = 0
    agent.starting_equity = 1000.0
    agent.policy = SimpleNamespace(harvester=None)
    agent._param_manager = LearnedParametersManager(persistence_path=tmp_path / "lp.json")
    return agent


def _persisted_floor(agent: TFAgent) -> float:
    return float(
        agent._param_manager.get(_SYMBOL, _PARAM, timeframe=_TF_LABEL, broker="default", default=_PARAM_DEFAULT)
    )


def test_losing_streak_ratchets_floor_past_single_step_cap(tmp_path):
    agent = _agent(tmp_path)
    for _ in range(300):
        agent.total_trades += 1
        agent._update_risk_feedback_thresholds(pnl_usd=-1.0)

    final = _persisted_floor(agent)
    assert final > _SINGLE_STEP_CAP, (
        f"expected compounding past single-step cap {_SINGLE_STEP_CAP}, got {final}"
    )
    assert final == _PARAM_MAX, f"expected runaway to pin at param max {_PARAM_MAX}, got {final}"


def test_floor_is_monotonic_nondecreasing_during_losses(tmp_path):
    agent = _agent(tmp_path)
    prev = _persisted_floor(agent)
    for _ in range(300):
        agent.total_trades += 1
        agent._update_risk_feedback_thresholds(pnl_usd=-1.0)
        cur = _persisted_floor(agent)
        assert cur >= prev - 1e-9, f"floor decreased during losing streak: {prev} -> {cur}"
        prev = cur


def test_winning_streak_does_not_ratchet_above_base(tmp_path):
    agent = _agent(tmp_path)
    for _ in range(300):
        agent.total_trades += 1
        agent._update_risk_feedback_thresholds(pnl_usd=+1.0)

    final = _persisted_floor(agent)
    assert final <= _PARAM_DEFAULT + 1e-9, f"winning streak should not raise floor, got {final}"
