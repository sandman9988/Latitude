"""Characterization tests for shadow-gate observability (phase 1).

Shadow-gate design: paper mode executes ALL trades (preserving RL
exploration), but every *live* entry gate is still evaluated on each
decision and the ``would_block`` verdicts are recorded on
``TriggerAgent.last_shadow_gates`` so the paper/live execution gap is
observable for telemetry and learning.

These tests pin the contract:
- paper mode still executes the trade even when live gates would block it;
- the shadow verdicts use the *live* threshold values, not the relaxed
  (0.0) active thresholds that paper runs with;
- live enforcement behaviour is unchanged (gates still block).
"""

import os
from unittest.mock import patch

import numpy as np

import pytest

from src.agents.trigger_agent import Q_RUNWAY_MIN, TriggerAgent


def _paper_agent() -> TriggerAgent:
    with patch.dict(os.environ, {"PAPER_MODE": "1"}):
        return TriggerAgent(window=64, n_features=7)


class TestShadowGatePopulation:
    def test_decide_populates_shadow_gates(self):
        agent = _paper_agent()
        agent.epsilon = 0.0
        state = np.random.default_rng(0).standard_normal((64, 7)).astype(np.float32)
        agent.decide(state, current_position=0)
        sg = agent.last_shadow_gates
        assert isinstance(sg, dict)
        for key in (
            "feasibility",
            "confidence",
            "entry_risk",
            "runway_length",
            "economics",
            "would_block_any",
            "reasons",
            "action",
        ):
            assert key in sg

    def test_in_position_clears_shadow_gates(self):
        agent = _paper_agent()
        state = np.zeros((64, 7), dtype=np.float32)
        agent.decide(state, current_position=1)
        assert agent.last_shadow_gates == {}


class TestShadowUsesLiveThresholds:
    def test_confidence_shadow_uses_live_floor_not_active(self):
        agent = _paper_agent()
        # Paper relaxes the *active* floor to 0.0 but keeps the live floor.
        assert agent.confidence_floor == pytest.approx(0.0)
        assert agent._live_confidence_floor > 0.0
        agent._is_runway_predictor_reliable = lambda: True
        # A low probability is below the live floor -> shadow would block,
        # even though the active (paper) floor would not.
        assert agent._confidence_would_block(0.01) is True
        assert agent._confidence_would_block(0.999) is False

    def test_feasibility_shadow_uses_live_threshold_not_active(self):
        agent = _paper_agent()
        assert agent.feasibility_threshold == pytest.approx(0.0)
        assert agent._live_feasibility_threshold > 0.0
        assert agent._feasibility_would_block(0.0) is True
        assert agent._feasibility_would_block(1.0) is False


class TestPaperExecutesDespiteShadowBlock:
    def test_paper_takes_trade_that_live_would_block(self):
        agent = _paper_agent()
        agent.epsilon = 1.0  # force exploration path
        agent._is_runway_predictor_reliable = lambda: True
        # Pin the live confidence floor above the explore confidence (0.5).
        agent._live_confidence_floor = 0.9
        state = np.zeros((64, 7), dtype=np.float32)
        with patch("src.agents.trigger_agent.random.random", return_value=0.0), \
             patch("src.agents.trigger_agent.random.choices", return_value=[1]):
            action, conf, _runway = agent.decide(state, current_position=0)
        # Paper still executes the entry...
        assert action == 1
        # ...but the shadow gate records that live would have blocked it.
        sg = agent.last_shadow_gates
        assert sg["confidence"] is True
        assert sg["would_block_any"] is True
        assert "confidence" in sg["reasons"]


class TestLiveEnforcementUnchanged:
    def test_live_confidence_gate_still_blocks(self):
        agent = TriggerAgent(window=64, n_features=7)  # live (not paper)
        assert agent.paper_mode is False
        agent._is_runway_predictor_reliable = lambda: True
        agent.confidence_floor = 0.9
        agent._live_confidence_floor = 0.9
        assert agent._confidence_gate_blocked(0.1) is True
        assert agent._confidence_gate_blocked(0.95) is False

    def test_paper_confidence_gate_never_blocks(self):
        agent = _paper_agent()
        agent._is_runway_predictor_reliable = lambda: True
        # Active gate is bypassed in paper regardless of probability.
        assert agent._confidence_gate_blocked(0.0) is False
