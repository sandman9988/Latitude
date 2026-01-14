"""
Broker Execution Model - Realistic Slippage & Execution Cost Modeling

Models asymmetric slippage based on:
- Order side (buy/sell)
- Market regime (trending, volatile, mean-reverting)
- Order size relative to typical volume
- Spread dynamics

Critical for accurate position sizing and RL reward calculations.
Without this, agent learns on idealized fills and underperforms in production.

Reference: Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
           Obizhaeva & Wang (2013) "Optimal Trading Strategy and Supply/Demand Dynamics"
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

from src.utils.safe_math import SafeMath

LOG = logging.getLogger(__name__)

# RegimeType from regime_detector (Literal type)
RegimeType = Literal["TRENDING", "MEAN_REVERTING", "TRANSITIONAL", "UNKNOWN"]


class OrderSide(Enum):
    """Order side for slippage calculation"""

    BUY = 1
    SELL = 2


@dataclass
class ExecutionCosts:
    """Complete execution cost breakdown"""

    base_slippage_bps: float  # Base slippage in basis points
    regime_adjustment_bps: float  # Regime-based adjustment
    size_impact_bps: float  # Market impact from order size
    spread_cost_bps: float  # Bid-ask spread cost
    total_slippage_bps: float  # Total expected slippage
    expected_fill_price: float  # Expected fill price after slippage
    cost_adjusted_size: float  # Position size adjusted for costs


class BrokerExecutionModel:
    """
    Models realistic execution costs for position sizing and reward calculation.

    Asymmetric slippage model:
    - BUY orders: Pay offer + slippage (worse execution when buying)
    - SELL orders: Receive bid - slippage (worse execution when selling)
    - Slippage increases with:
      * Market volatility (volatile regime)
      * Trend strength (trending regime)
      * Order size (market impact)

    Usage:
        model = BrokerExecutionModel(typical_spread_bps=5.0)
        costs = model.estimate_execution_costs(
            side=OrderSide.BUY,
            quantity=0.10,
            mid_price=50000.0,
            spread_bps=6.0,
            regime=RegimeType.VOLATILE
        )
        adjusted_qty = costs.cost_adjusted_size
    """

    def __init__(
        self,
        typical_spread_bps: float = 5.0,
        base_slippage_bps: float = 2.0,
        volatile_multiplier: float = 2.0,
        trending_multiplier: float = 1.5,
        mean_reverting_multiplier: float = 0.8,
        size_impact_coefficient: float = 10.0,
        max_total_cost_bps: float = 50.0,
    ):
        """
        Initialize execution model with cost parameters.

        Args:
            typical_spread_bps: Typical bid-ask spread (basis points)
            base_slippage_bps: Base slippage in normal conditions
            volatile_multiplier: Slippage multiplier in volatile regime
            trending_multiplier: Slippage multiplier in trending regime
            mean_reverting_multiplier: Slippage multiplier in mean-reverting regime
            size_impact_coefficient: Market impact scaling (bps per lot)
            max_total_cost_bps: Maximum total execution cost (safety cap)
        """
        self.typical_spread_bps = typical_spread_bps
        self.base_slippage_bps = base_slippage_bps
        self.volatile_multiplier = volatile_multiplier
        self.trending_multiplier = trending_multiplier
        self.mean_reverting_multiplier = mean_reverting_multiplier
        self.size_impact_coefficient = size_impact_coefficient
        self.max_total_cost_bps = max_total_cost_bps

        LOG.info(
            "[EXECUTION_MODEL] Initialized: base_slip=%.1f bps, volatile_mult=%.1fx, "
            "trending_mult=%.1fx, mean_rev_mult=%.1fx, impact_coef=%.1f",
            base_slippage_bps,
            volatile_multiplier,
            trending_multiplier,
            mean_reverting_multiplier,
            size_impact_coefficient,
        )

    def estimate_execution_costs(
        self,
        side: OrderSide,
        quantity: float,
        mid_price: float,
        spread_bps: Optional[float] = None,
        regime: RegimeType = "UNKNOWN",
        typical_quantity: float = 0.10,
    ) -> ExecutionCosts:
        """
        Estimate total execution costs for an order.

        Args:
            side: BUY or SELL
            quantity: Order quantity (e.g., 0.10 BTC)
            mid_price: Current mid price
            spread_bps: Current spread in bps (None = use typical)
            regime: Market regime
            typical_quantity: Typical order size for impact calculation

        Returns:
            ExecutionCosts with complete breakdown
        """
        # Spread cost (half-spread for market orders)
        if spread_bps is None:
            spread_bps = self.typical_spread_bps
        spread_cost_bps = spread_bps / 2.0

        # Base slippage
        base_slip = self.base_slippage_bps

        # Regime adjustment
        regime_mult = self._get_regime_multiplier(regime)
        regime_adjustment_bps = base_slip * (regime_mult - 1.0)

        # Size impact (market impact scales with order size)
        size_ratio = SafeMath.safe_div(quantity, typical_quantity, default=1.0)
        size_impact_bps = self.size_impact_coefficient * max(0.0, size_ratio - 1.0)

        # Total slippage
        total_slippage_bps = base_slip + regime_adjustment_bps + size_impact_bps + spread_cost_bps

        # Cap at maximum
        total_slippage_bps = min(total_slippage_bps, self.max_total_cost_bps)

        # Calculate expected fill price
        slippage_fraction = total_slippage_bps / 10000.0  # bps to fraction
        if side == OrderSide.BUY:
            # Buy: pay mid + slippage (worse execution)
            expected_fill_price = mid_price * (1.0 + slippage_fraction)
        else:
            # Sell: receive mid - slippage (worse execution)
            expected_fill_price = mid_price * (1.0 - slippage_fraction)

        # Adjust position size to maintain same dollar exposure after costs
        # If we're paying more (BUY) or receiving less (SELL), reduce size proportionally
        cost_adjusted_size = quantity / (1.0 + slippage_fraction)

        LOG.debug(
            "[EXECUTION_MODEL] %s %.4f @ mid=%.2f: base=%.1f bps, regime=%.1f bps, "
            "size_impact=%.1f bps, spread=%.1f bps → total=%.1f bps, "
            "fill_price=%.2f, adjusted_size=%.4f",
            side.name,
            quantity,
            mid_price,
            base_slip,
            regime_adjustment_bps,
            size_impact_bps,
            spread_cost_bps,
            total_slippage_bps,
            expected_fill_price,
            cost_adjusted_size,
        )

        return ExecutionCosts(
            base_slippage_bps=base_slip,
            regime_adjustment_bps=regime_adjustment_bps,
            size_impact_bps=size_impact_bps,
            spread_cost_bps=spread_cost_bps,
            total_slippage_bps=total_slippage_bps,
            expected_fill_price=expected_fill_price,
            cost_adjusted_size=cost_adjusted_size,
        )

    def _get_regime_multiplier(self, regime: RegimeType) -> float:
        """Get slippage multiplier for given regime."""
        if regime == "TRANSITIONAL":  # High volatility/uncertainty
            return self.volatile_multiplier
        elif regime == "TRENDING":
            return self.trending_multiplier
        elif regime == "MEAN_REVERTING":
            return self.mean_reverting_multiplier
        else:
            return 1.0  # Unknown regime = no adjustment

    def adjust_position_size_for_costs(
        self,
        side: OrderSide,
        target_quantity: float,
        mid_price: float,
        spread_bps: Optional[float] = None,
        regime: RegimeType = "UNKNOWN",
    ) -> float:
        """
        Adjust position size downward to account for execution costs.

        This ensures that the agent's actual capital at risk matches the
        target after accounting for slippage.

        Args:
            side: BUY or SELL
            target_quantity: Desired quantity before cost adjustment
            mid_price: Current mid price
            spread_bps: Current spread
            regime: Market regime

        Returns:
            Cost-adjusted quantity (always <= target_quantity)
        """
        costs = self.estimate_execution_costs(
            side=side,
            quantity=target_quantity,
            mid_price=mid_price,
            spread_bps=spread_bps,
            regime=regime,
        )

        adjusted = costs.cost_adjusted_size

        if adjusted < target_quantity * 0.5:
            LOG.warning(
                "[EXECUTION_MODEL] Large cost adjustment: %.4f → %.4f (%.1f%% reduction)",
                target_quantity,
                adjusted,
                100.0 * (1.0 - adjusted / target_quantity),
            )

        return adjusted

    def get_state_dict(self) -> dict:
        """Get model parameters for persistence."""
        return {
            "typical_spread_bps": self.typical_spread_bps,
            "base_slippage_bps": self.base_slippage_bps,
            "volatile_multiplier": self.volatile_multiplier,
            "trending_multiplier": self.trending_multiplier,
            "mean_reverting_multiplier": self.mean_reverting_multiplier,
            "size_impact_coefficient": self.size_impact_coefficient,
            "max_total_cost_bps": self.max_total_cost_bps,
        }

    def load_state_dict(self, state: dict):
        """Load model parameters from persistence."""
        self.typical_spread_bps = state.get("typical_spread_bps", 5.0)
        self.base_slippage_bps = state.get("base_slippage_bps", 2.0)
        self.volatile_multiplier = state.get("volatile_multiplier", 2.0)
        self.trending_multiplier = state.get("trending_multiplier", 1.5)
        self.mean_reverting_multiplier = state.get("mean_reverting_multiplier", 0.8)
        self.size_impact_coefficient = state.get("size_impact_coefficient", 10.0)
        self.max_total_cost_bps = state.get("max_total_cost_bps", 50.0)
        LOG.info("[EXECUTION_MODEL] Loaded state from persistence")


# ============================================================================
# SELF-TESTS
# ============================================================================


def _test_basic_execution_costs():
    """Test basic execution cost calculation."""
    model = BrokerExecutionModel(
        typical_spread_bps=5.0,
        base_slippage_bps=2.0,
        volatile_multiplier=2.0,
    )

    # BUY order in normal regime
    costs = model.estimate_execution_costs(
        side=OrderSide.BUY, quantity=0.10, mid_price=50000.0, spread_bps=5.0, regime="UNKNOWN"
    )

    # Should have spread cost + base slippage
    expected_min = 2.0 + 2.5  # base + half-spread
    assert costs.total_slippage_bps >= expected_min, f"Expected >= {expected_min}, got {costs.total_slippage_bps}"

    # Buy should pay above mid
    assert costs.expected_fill_price > 50000.0, "BUY should pay above mid"

    # Adjusted size should be less
    assert costs.cost_adjusted_size < 0.10, "Cost-adjusted size should be less"

    LOG.info("✓ _test_basic_execution_costs PASSED")


def _test_asymmetric_slippage():
    """Test that BUY and SELL have asymmetric impact."""
    model = BrokerExecutionModel()

    mid = 50000.0
    qty = 0.10

    buy_costs = model.estimate_execution_costs(side=OrderSide.BUY, quantity=qty, mid_price=mid, regime="UNKNOWN")

    sell_costs = model.estimate_execution_costs(side=OrderSide.SELL, quantity=qty, mid_price=mid, regime="UNKNOWN")

    # Both should have positive slippage
    assert buy_costs.total_slippage_bps > 0, "BUY slippage should be positive"
    assert sell_costs.total_slippage_bps > 0, "SELL slippage should be positive"

    # BUY pays above mid
    assert buy_costs.expected_fill_price > mid, "BUY should pay above mid"

    # SELL receives below mid
    assert sell_costs.expected_fill_price < mid, "SELL should receive below mid"

    LOG.info("✓ _test_asymmetric_slippage PASSED")


def _test_regime_impact():
    """Test regime multipliers."""
    model = BrokerExecutionModel(
        base_slippage_bps=10.0, volatile_multiplier=2.0, trending_multiplier=1.5, mean_reverting_multiplier=0.8
    )

    qty = 0.10
    mid = 50000.0

    normal = model.estimate_execution_costs(side=OrderSide.BUY, quantity=qty, mid_price=mid, regime="UNKNOWN")

    volatile = model.estimate_execution_costs(side=OrderSide.BUY, quantity=qty, mid_price=mid, regime="TRANSITIONAL")

    trending = model.estimate_execution_costs(side=OrderSide.BUY, quantity=qty, mid_price=mid, regime="TRENDING")

    mean_rev = model.estimate_execution_costs(side=OrderSide.BUY, quantity=qty, mid_price=mid, regime="MEAN_REVERTING")

    # Volatile should cost most
    assert volatile.total_slippage_bps > trending.total_slippage_bps, "Volatile should have highest slippage"

    # Trending should cost more than normal
    assert trending.total_slippage_bps > normal.total_slippage_bps, "Trending should cost more than normal"

    # Mean-reverting should cost less than normal
    assert mean_rev.total_slippage_bps < normal.total_slippage_bps, "Mean-reverting should cost less than normal"

    LOG.info("✓ _test_regime_impact PASSED")


def _test_size_impact():
    """Test that larger orders have more impact."""
    model = BrokerExecutionModel(size_impact_coefficient=10.0)

    mid = 50000.0
    typical = 0.10

    small = model.estimate_execution_costs(side=OrderSide.BUY, quantity=0.05, mid_price=mid, typical_quantity=typical)

    normal = model.estimate_execution_costs(side=OrderSide.BUY, quantity=0.10, mid_price=mid, typical_quantity=typical)

    large = model.estimate_execution_costs(side=OrderSide.BUY, quantity=0.50, mid_price=mid, typical_quantity=typical)

    # Small order should have minimal size impact
    assert small.size_impact_bps < 1.0, "Small order should have minimal size impact"

    # Normal should have zero size impact
    assert abs(normal.size_impact_bps) < 0.01, "Normal order should have ~zero size impact"

    # Large order should have significant impact
    assert large.size_impact_bps > 10.0, "Large order should have significant size impact"

    LOG.info("✓ _test_size_impact PASSED")


def _test_cost_cap():
    """Test maximum cost cap."""
    model = BrokerExecutionModel(
        base_slippage_bps=100.0,  # Unrealistically high
        volatile_multiplier=10.0,
        max_total_cost_bps=50.0,
    )

    costs = model.estimate_execution_costs(
        side=OrderSide.BUY, quantity=10.0, mid_price=50000.0, regime="TRANSITIONAL", typical_quantity=0.10
    )

    # Should be capped
    assert costs.total_slippage_bps <= 50.0, f"Cost should be capped at 50 bps, got {costs.total_slippage_bps}"

    LOG.info("✓ _test_cost_cap PASSED")


def _test_position_size_adjustment():
    """Test position size adjustment."""
    model = BrokerExecutionModel()

    target = 0.10
    adjusted = model.adjust_position_size_for_costs(
        side=OrderSide.BUY, target_quantity=target, mid_price=50000.0, regime="TRANSITIONAL"
    )

    # Adjusted should be less than target
    assert adjusted < target, "Adjusted size should be less than target"

    # Should not be dramatically different
    assert adjusted > target * 0.9, "Adjustment should not be too large"

    LOG.info("✓ _test_position_size_adjustment PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    _test_basic_execution_costs()
    _test_asymmetric_slippage()
    _test_regime_impact()
    _test_size_impact()
    _test_cost_cap()
    _test_position_size_adjustment()

    LOG.info("=" * 60)
    LOG.info("ALL EXECUTION MODEL TESTS PASSED ✓")
    LOG.info("=" * 60)
