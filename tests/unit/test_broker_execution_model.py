"""
Tests for src.core.broker_execution_model

Coverage targets:
- OrderSide enum
- ExecutionCosts dataclass
- BrokerExecutionModel: estimate_execution_costs, _get_regime_multiplier,
  adjust_position_size_for_costs, get_state_dict, load_state_dict
"""

import pytest

from src.core.broker_execution_model import (
    BrokerExecutionModel,
    ExecutionCosts,
    OrderSide,
)


# ── OrderSide ──────────────────────────────────────────────────────────────

class TestOrderSide:
    def test_buy_value(self):
        assert OrderSide.BUY.value == 1

    def test_sell_value(self):
        assert OrderSide.SELL.value == 2


# ── ExecutionCosts ─────────────────────────────────────────────────────────

class TestExecutionCosts:
    def test_create(self):
        ec = ExecutionCosts(
            base_slippage_bps=2.0,
            regime_adjustment_bps=1.0,
            size_impact_bps=0.5,
            spread_cost_bps=2.5,
            total_slippage_bps=6.0,
            expected_fill_price=50003.0,
            cost_adjusted_size=0.0997,
        )
        assert ec.total_slippage_bps == pytest.approx(6.0)


# ── BrokerExecutionModel defaults ─────────────────────────────────────────

class TestBrokerInit:
    def test_default_params(self):
        model = BrokerExecutionModel()
        assert model.typical_spread_bps == pytest.approx(5.0)
        assert model.base_slippage_bps == pytest.approx(2.0)
        assert model.volatile_multiplier == pytest.approx(2.0)
        assert model.trending_multiplier == pytest.approx(1.5)
        assert model.mean_reverting_multiplier == pytest.approx(0.8)
        assert model.size_impact_coefficient == pytest.approx(10.0)
        assert model.max_total_cost_bps == pytest.approx(50.0)

    def test_custom_params(self):
        model = BrokerExecutionModel(
            typical_spread_bps=10.0,
            base_slippage_bps=5.0,
            max_total_cost_bps=100.0,
        )
        assert model.typical_spread_bps == pytest.approx(10.0)
        assert model.base_slippage_bps == pytest.approx(5.0)


# ── estimate_execution_costs ──────────────────────────────────────────────

class TestEstimateExecutionCosts:
    @pytest.fixture
    def model(self):
        return BrokerExecutionModel(
            typical_spread_bps=5.0,
            base_slippage_bps=2.0,
            volatile_multiplier=2.0,
            trending_multiplier=1.5,
            mean_reverting_multiplier=0.8,
            size_impact_coefficient=10.0,
            max_total_cost_bps=50.0,
        )

    def test_buy_pays_above_mid(self, model):
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0
        )
        assert costs.expected_fill_price > 50000.0

    def test_sell_receives_below_mid(self, model):
        costs = model.estimate_execution_costs(
            side=OrderSide.SELL, quantity=0.10, mid_price=50000.0
        )
        assert costs.expected_fill_price < 50000.0

    def test_spread_cost_is_half_spread(self, model):
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, spread_bps=10.0
        )
        assert costs.spread_cost_bps == pytest.approx(5.0)

    def test_uses_typical_spread_when_none(self, model):
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, spread_bps=None
        )
        assert costs.spread_cost_bps == model.typical_spread_bps / 2.0

    def test_base_slippage_present(self, model):
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, regime="UNKNOWN"
        )
        assert costs.base_slippage_bps == pytest.approx(2.0)

    def test_no_regime_adjustment_for_unknown(self, model):
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, regime="UNKNOWN"
        )
        assert costs.regime_adjustment_bps == pytest.approx(0.0)

    def test_volatile_regime_increases_cost(self, model):
        normal = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, regime="UNKNOWN"
        )
        volatile = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, regime="TRANSITIONAL"
        )
        assert volatile.total_slippage_bps > normal.total_slippage_bps
        assert volatile.regime_adjustment_bps > 0

    def test_trending_regime_increases_cost(self, model):
        normal = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, regime="UNKNOWN"
        )
        trending = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, regime="TRENDING"
        )
        assert trending.total_slippage_bps > normal.total_slippage_bps

    def test_mean_reverting_decreases_cost(self, model):
        normal = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, regime="UNKNOWN"
        )
        mr = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, regime="MEAN_REVERTING"
        )
        assert mr.total_slippage_bps < normal.total_slippage_bps

    def test_size_impact_zero_for_normal_size(self, model):
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, typical_quantity=0.10
        )
        assert abs(costs.size_impact_bps) < 0.01

    def test_size_impact_grows_for_large_orders(self, model):
        small = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, typical_quantity=0.10
        )
        large = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=1.0, mid_price=50000.0, typical_quantity=0.10
        )
        assert large.size_impact_bps > small.size_impact_bps
        assert large.total_slippage_bps > small.total_slippage_bps

    def test_cost_capped(self):
        model = BrokerExecutionModel(
            base_slippage_bps=100.0,
            volatile_multiplier=10.0,
            max_total_cost_bps=50.0,
        )
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=10.0, mid_price=50000.0,
            regime="TRANSITIONAL", typical_quantity=0.10,
        )
        assert costs.total_slippage_bps <= 50.0

    def test_cost_adjusted_size_less_than_original(self, model):
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0
        )
        assert costs.cost_adjusted_size < 0.10


# ── _get_regime_multiplier ────────────────────────────────────────────────

class TestRegimeMultiplier:
    def test_transitional(self):
        model = BrokerExecutionModel(volatile_multiplier=3.0)
        assert model._get_regime_multiplier("TRANSITIONAL") == pytest.approx(3.0)

    def test_trending(self):
        model = BrokerExecutionModel(trending_multiplier=1.8)
        assert model._get_regime_multiplier("TRENDING") == pytest.approx(1.8)

    def test_mean_reverting(self):
        model = BrokerExecutionModel(mean_reverting_multiplier=0.6)
        assert model._get_regime_multiplier("MEAN_REVERTING") == pytest.approx(0.6)

    def test_unknown(self):
        model = BrokerExecutionModel()
        assert model._get_regime_multiplier("UNKNOWN") == pytest.approx(1.0)

    def test_unrecognized_string(self):
        model = BrokerExecutionModel()
        assert model._get_regime_multiplier("SOMETHING_ELSE") == pytest.approx(1.0)


# ── adjust_position_size_for_costs ────────────────────────────────────────

class TestAdjustPositionSize:
    def test_adjusted_less_than_target(self):
        model = BrokerExecutionModel()
        adjusted = model.adjust_position_size_for_costs(
            side=OrderSide.BUY,
            target_quantity=0.10,
            mid_price=50000.0,
            regime="TRANSITIONAL",
        )
        assert adjusted < 0.10

    def test_adjusted_not_dramatically_different(self):
        model = BrokerExecutionModel()
        adjusted = model.adjust_position_size_for_costs(
            side=OrderSide.BUY,
            target_quantity=0.10,
            mid_price=50000.0,
            regime="UNKNOWN",
        )
        assert adjusted > 0.10 * 0.9

    def test_large_cost_triggers_warning(self):
        """When adjusted < 50% of target, warning is logged."""
        model = BrokerExecutionModel(
            base_slippage_bps=5000.0,
            volatile_multiplier=10.0,
            max_total_cost_bps=100000.0,  # Very high cap
        )
        adjusted = model.adjust_position_size_for_costs(
            side=OrderSide.BUY,
            target_quantity=0.10,
            mid_price=50000.0,
            regime="TRANSITIONAL",
            spread_bps=5000.0,
        )
        assert adjusted < 0.10 * 0.5  # Should trigger the warning path


# ── State dict ─────────────────────────────────────────────────────────────

class TestStateDict:
    def test_get_state_dict(self):
        model = BrokerExecutionModel(
            typical_spread_bps=7.0,
            base_slippage_bps=3.0,
        )
        state = model.get_state_dict()
        assert state["typical_spread_bps"] == pytest.approx(7.0)
        assert state["base_slippage_bps"] == pytest.approx(3.0)
        assert "volatile_multiplier" in state
        assert "max_total_cost_bps" in state

    def test_load_state_dict(self):
        model = BrokerExecutionModel()
        state = {
            "typical_spread_bps": 12.0,
            "base_slippage_bps": 4.0,
            "volatile_multiplier": 3.0,
            "trending_multiplier": 2.0,
            "mean_reverting_multiplier": 0.5,
            "size_impact_coefficient": 20.0,
            "max_total_cost_bps": 80.0,
        }
        model.load_state_dict(state)
        assert model.typical_spread_bps == pytest.approx(12.0)
        assert model.base_slippage_bps == pytest.approx(4.0)
        assert model.volatile_multiplier == pytest.approx(3.0)
        assert model.max_total_cost_bps == pytest.approx(80.0)

    def test_load_state_dict_partial(self):
        """Load with missing keys uses defaults."""
        model = BrokerExecutionModel()
        model.load_state_dict({"typical_spread_bps": 99.0})
        assert model.typical_spread_bps == pytest.approx(99.0)
        assert model.base_slippage_bps == pytest.approx(2.0)  # Default

    def test_roundtrip(self):
        model = BrokerExecutionModel(
            typical_spread_bps=8.0,
            base_slippage_bps=3.5,
            volatile_multiplier=2.5,
        )
        state = model.get_state_dict()
        model2 = BrokerExecutionModel()
        model2.load_state_dict(state)
        assert model2.typical_spread_bps == model.typical_spread_bps
        assert model2.base_slippage_bps == model.base_slippage_bps
        assert model2.volatile_multiplier == model.volatile_multiplier
