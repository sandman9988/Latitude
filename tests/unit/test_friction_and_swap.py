import pytest

from src.risk.friction_costs import FrictionCalculator
from src.agents.harvester_agent import HarvesterAgent


@pytest.fixture
def fc_xauusd():
    # Ensure config-based specs are available
    return FrictionCalculator(symbol="XAUUSD", symbol_id=41, timeframe="M5", broker="default")


def test_swap_intraday_zero(fc_xauusd):
    # Intraday trade not crossing rollover should have zero swap
    cost = fc_xauusd.calculate_swap(quantity=0.1, side="BUY", holding_days=0.1, crosses_rollover=False)
    assert cost == pytest.approx(0.0)


def test_swap_overnight_single_rollover(fc_xauusd):
    # Overnight crossing rollover must incur non-zero swap (can be positive or negative)
    cost = fc_xauusd.calculate_swap(quantity=0.1, side="BUY", holding_days=1.0, crosses_rollover=True)
    assert cost != 0.0


def test_friction_breakdown_includes_zero_swap_intraday(fc_xauusd):
    # Intraday total should exclude swap when not crossing rollover
    price = 4600.0
    out = fc_xauusd.calculate_total_friction(
        quantity=0.1, side="BUY", price=price, holding_days=0.1, volatility_factor=1.0, crosses_rollover=False
    )
    assert out["swap"] == pytest.approx(0.0)
    # Spread + commission + slippage must be positive
    assert out["spread"] > 0
    assert out["commission"] >= 0
    assert out["slippage"] >= 0
    assert out["total"] == pytest.approx(out["spread"] + out["commission"] + out["slippage"], rel=1e-6)


def test_friction_breakdown_includes_swap_overnight(fc_xauusd):
    price = 4600.0
    out = fc_xauusd.calculate_total_friction(
        quantity=0.1, side="BUY", price=price, holding_days=1.0, volatility_factor=1.0, crosses_rollover=True
    )
    assert out["swap"] != 0.0
    assert out["total"] == pytest.approx(out["spread"] + out["commission"] + out["slippage"] + out["swap"], rel=1e-6)


def test_friction_pct_uses_contract_size(monkeypatch):
    # Build a HarvesterAgent with a friction calculator whose contract_size is 100 (XAUUSD)
    fc = FrictionCalculator(symbol="XAUUSD", symbol_id=41, timeframe="M5", broker="default")
    harvester = HarvesterAgent(window=64, n_features=10, enable_training=False, symbol="XAUUSD", timeframe="M5", friction_calculator=fc)

    entry_price = 4600.0
    pct = harvester.get_friction_cost_pct(entry_price=entry_price, quantity=0.1, side="BUY")
    # Sanity: friction percent should be small but positive (<1%)
    assert 0.0 < pct < 0.01

    # Now force contract_size to a very large number to see percent shrink as notional increases
    fc.costs.contract_size = 1_000_000
    pct_large = harvester.get_friction_cost_pct(entry_price=entry_price, quantity=0.1, side="BUY")
    assert pct_large < pct
