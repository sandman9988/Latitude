"""Extended tests for src.core.broker_execution_model.

Covers: adjust_position_size_for_costs warning path (large cost),
estimate_execution_costs with very small typical qty, sell fill below mid,
load_state_dict with empty dict, full cost breakdown verification.
"""

import pytest

from src.core.broker_execution_model import (
    BrokerExecutionModel,
    OrderSide,
)


class TestAdjustPositionSizeExtended:
    def test_large_cost_triggers_warning(self):
        model = BrokerExecutionModel(
            base_slippage_bps=100.0,
            volatile_multiplier=10.0,
            max_total_cost_bps=5000.0,  # Allow extreme to test warning path
        )
        adjusted = model.adjust_position_size_for_costs(
            side=OrderSide.BUY,
            target_quantity=0.10,
            mid_price=50000.0,
            regime="TRANSITIONAL",
        )
        # Should still return a valid number
        assert 0 < adjusted < 0.10

    def test_normal_cost_no_dramatic_change(self):
        model = BrokerExecutionModel()
        adjusted = model.adjust_position_size_for_costs(
            side=OrderSide.SELL,
            target_quantity=0.10,
            mid_price=50000.0,
            regime="UNKNOWN",
        )
        assert adjusted < 0.10
        assert adjusted > 0.10 * 0.9


class TestExecutionCostsBreakdown:
    def test_sell_fill_below_mid(self):
        model = BrokerExecutionModel()
        costs = model.estimate_execution_costs(
            side=OrderSide.SELL, quantity=0.10, mid_price=50000.0
        )
        assert costs.expected_fill_price < 50000.0

    def test_zero_spread_only_base_slippage(self):
        model = BrokerExecutionModel(base_slippage_bps=5.0)
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0,
            spread_bps=0.0, regime="UNKNOWN", typical_quantity=0.10,
        )
        # Only base slippage + zero spread + zero size impact
        assert costs.total_slippage_bps == pytest.approx(5.0)

    def test_small_typical_qty_large_size_impact(self):
        model = BrokerExecutionModel(size_impact_coefficient=10.0)
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=1.0, mid_price=50000.0,
            typical_quantity=0.01,  # Very small typical → huge ratio
        )
        assert costs.size_impact_bps > 50.0

    def test_cost_adjusted_size_always_less(self):
        model = BrokerExecutionModel()
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY, quantity=0.10, mid_price=50000.0
        )
        assert costs.cost_adjusted_size < 0.10


class TestLoadStateDictExtended:
    def test_load_empty_dict_uses_defaults(self):
        model = BrokerExecutionModel(base_slippage_bps=99.0)
        model.load_state_dict({})
        assert model.base_slippage_bps == pytest.approx(2.0)  # Default
        assert model.typical_spread_bps == pytest.approx(5.0)

    def test_load_partial_dict(self):
        model = BrokerExecutionModel()
        model.load_state_dict({"base_slippage_bps": 7.5})
        assert model.base_slippage_bps == pytest.approx(7.5)
        assert model.volatile_multiplier == pytest.approx(2.0)  # Default kept
