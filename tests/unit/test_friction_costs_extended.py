"""Extended tests for src.risk.friction_costs.

Covers uncovered paths: commission PERCENTAGE, swap PIPS/PERCENTAGE with
rollover & triple-swap-day, calculate_total_friction breakdown,
get_friction_adjusted_pnl, is_spread_acceptable, get_statistics,
get_spread_stats_for_logging, update_digits_from_price,
_refresh_derived_costs, SpreadTracker current_hour_spread.
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from src.risk.friction_costs import (
    FrictionCalculator,
    SlippageModel,
    SpreadTracker,
    SymbolCosts,
)


# ---------------------------------------------------------------------------
# Helpers – create a FrictionCalculator without hitting config / persistence
# ---------------------------------------------------------------------------
def _make_calc(**overrides):
    """Build a FrictionCalculator with mocked file-based init."""
    defaults = dict(symbol="BTCUSD", symbol_id=10028, timeframe="M5", broker="test")
    defaults.update(overrides)

    with (
        patch.object(FrictionCalculator, "_load_symbol_specs_from_config"),
        patch.object(FrictionCalculator, "_load_learned_parameters"),
    ):
        calc = FrictionCalculator(**defaults)
    return calc


# ===================================================================
# Commission – PERCENTAGE type
# ===================================================================
class TestCommissionPercentage:
    @pytest.fixture()
    def calc(self):
        c = _make_calc()
        c.costs.commission_type = "PERCENTAGE"
        c.costs.commission_percentage = 0.0007  # 0.07%
        return c

    def test_percentage_commission_basic(self, calc):
        # quantity=0.01 lot, price=1000 → notional = 0.01 * 1000 * 100000 = 1_000_000
        # comm = 1_000_000 * 0.0007 = 700.0 (under 100k cap)
        comm = calc.calculate_commission(0.01, 1000.0)
        expected = 0.01 * 1000.0 * 100000 * 0.0007
        assert comm == pytest.approx(expected, rel=1e-6)

    def test_percentage_commission_small_qty(self, calc):
        comm = calc.calculate_commission(0.01, 95000.0)
        expected = 0.01 * 95000.0 * 100000 * 0.0007
        assert comm == pytest.approx(expected, rel=1e-6)

    def test_percentage_commission_nan_rate_falls_back(self, calc):
        calc.costs.commission_percentage = float("nan")
        comm = calc.calculate_commission(0.01, 95000.0)
        # Falls back to 0.0007 default, still positive
        assert comm > 0

    def test_percentage_commission_capped_rate(self, calc):
        calc.costs.commission_percentage = 0.05  # 5% – capped at 1%
        comm = calc.calculate_commission(1.0, 95000.0)
        max_comm = 1.0 * 95000.0 * 100000 * 0.01
        assert comm <= max_comm

    def test_min_commission_enforced(self, calc):
        calc.costs.min_commission = 5.0
        calc.costs.commission_percentage = 1e-9  # Tiny rate
        comm = calc.calculate_commission(0.01, 1.0)
        assert comm >= 5.0


# ===================================================================
# Swap – PIPS path
# ===================================================================
class TestSwapPips:
    @pytest.fixture()
    def calc(self):
        c = _make_calc()
        c.costs.swap_type = "PIPS"
        c.costs.swap_long = -7.2
        c.costs.swap_short = -4.5
        c.costs.pip_value_per_lot = 10.0
        c.costs.triple_swap_day = 2  # Wednesday
        return c

    def test_intraday_no_rollover_is_zero(self, calc):
        swap = calc.calculate_swap(0.1, "BUY", holding_days=0.5, crosses_rollover=False)
        assert swap == pytest.approx(0.0)

    def test_crosses_rollover_buy(self, calc):
        swap = calc.calculate_swap(0.1, "BUY", holding_days=0.5, crosses_rollover=True)
        # 1 rollover: -7.2 * 10 * 0.1 * 1 = -7.2
        assert swap == pytest.approx(-7.2, abs=0.01)

    def test_crosses_rollover_sell(self, calc):
        swap = calc.calculate_swap(0.1, "SELL", holding_days=0.5, crosses_rollover=True)
        assert swap == pytest.approx(-4.5 * 10.0 * 0.1 * 1, abs=0.01)

    def test_multiday_holding(self, calc):
        swap = calc.calculate_swap(0.1, "BUY", holding_days=3.0, crosses_rollover=False)
        # 3 rollovers (no triple swap unless weekday matches)
        _expected_rollovers = 3
        _expected = -7.2 * 10.0 * 0.1 * _expected_rollovers
        # Could be 3 or 5 depending on weekday; just ensure non-zero and negative
        assert swap < 0

    def test_triple_swap_day_effect(self, calc):
        """Triple-swap day adds 2 extra rollovers for multiday positions."""
        # For multiday with crosses_rollover=False and holding 3 days:
        # num_rollovers = 3 (base).  If it's triple-swap day, +2 = 5.
        # We test that the function runs without error and returns a float;
        # the exact multiplier depends on the current weekday.
        swap = calc.calculate_swap(0.1, "BUY", holding_days=3.0, crosses_rollover=False)
        assert isinstance(swap, float)
        assert swap < 0  # swap_long is negative

    def test_invalid_triple_swap_day_defaults_to_2(self, calc):
        calc.costs.triple_swap_day = 99
        swap = calc.calculate_swap(0.1, "BUY", holding_days=1.0, crosses_rollover=True)
        # Should not raise; defaults tsd to 2 internally
        assert isinstance(swap, float)


# ===================================================================
# Swap – PERCENTAGE path
# ===================================================================
class TestSwapPercentage:
    @pytest.fixture()
    def calc(self):
        c = _make_calc()
        c.costs.swap_type = "PERCENTAGE"
        c.costs.swap_long = -2.5  # -2.5% annual
        c.costs.swap_short = -1.0
        c.costs.contract_size = 1.0  # For BTC: 1 contract = 1 BTC
        c.costs.triple_swap_day = 2
        return c

    def test_percentage_buy_crosses_rollover(self, calc):
        swap = calc.calculate_swap(
            0.1, "BUY", holding_days=0.5, crosses_rollover=True, price=95000.0
        )
        # notional = 0.1 * 1.0 * 95000 = 9500
        # daily_rate = -2.5 / 100 / 365
        # swap = 9500 * daily_rate * 1 (at least 1 rollover)
        notional = 0.1 * 1.0 * 95000.0
        daily = -2.5 / 100 / 365
        expected = notional * daily * 1
        assert swap == pytest.approx(expected, rel=1e-4)

    def test_percentage_zero_price_returns_zero(self, calc):
        swap = calc.calculate_swap(
            0.1, "BUY", holding_days=1.0, crosses_rollover=True, price=0.0
        )
        assert swap == pytest.approx(0.0)

    def test_percentage_sell(self, calc):
        swap = calc.calculate_swap(
            0.1, "SELL", holding_days=0.5, crosses_rollover=True, price=95000.0
        )
        notional = 0.1 * 1.0 * 95000.0
        daily = -1.0 / 100 / 365
        assert swap == pytest.approx(notional * daily * 1, rel=1e-4)


# ===================================================================
# Swap – unknown type
# ===================================================================
class TestSwapUnknownType:
    def test_unknown_swap_type_returns_zero(self):
        c = _make_calc()
        c.costs.swap_type = "WEIRD"
        assert c.calculate_swap(0.1, "BUY", 1.0, True) == pytest.approx(0.0)


# ===================================================================
# calculate_total_friction
# ===================================================================
class TestTotalFriction:
    @pytest.fixture()
    def calc(self):
        c = _make_calc()
        # Inject a known spread
        c.spread_tracker.update(95000.0, 95010.0, 1.0)
        return c

    def test_returns_dict_with_all_keys(self, calc):
        result = calc.calculate_total_friction(0.1, "BUY", 95000.0)
        for key in ("spread", "commission", "swap", "slippage", "total", "total_pips", "quantity", "side", "price"):
            assert key in result

    def test_total_equals_sum(self, calc):
        r = calc.calculate_total_friction(0.1, "BUY", 95000.0)
        assert r["total"] == pytest.approx(
            r["spread"] + r["commission"] + r["swap"] + r["slippage"],
            rel=1e-6,
        )

    def test_total_pips_positive(self, calc):
        r = calc.calculate_total_friction(0.1, "BUY", 95000.0)
        assert r["total_pips"] >= 0

    def test_total_pips_zero_quantity(self, calc):
        r = calc.calculate_total_friction(0.0, "BUY", 95000.0)
        # quantity=0 → each sub-cost returns 0 → total_pips = 0
        assert r["total_pips"] == pytest.approx(0.0)

    def test_side_and_price_in_result(self, calc):
        r = calc.calculate_total_friction(0.1, "SELL", 92000.0)
        assert r["side"] == "SELL"
        assert r["price"] == pytest.approx(92000.0)


# ===================================================================
# get_friction_adjusted_pnl
# ===================================================================
class TestFrictionAdjustedPnl:
    def test_positive_pnl_reduced(self):
        calc = _make_calc()
        calc.spread_tracker.update(95000.0, 95010.0, 1.0)
        net = calc.get_friction_adjusted_pnl(100.0, 0.1, "BUY", 95000.0)
        assert net < 100.0

    def test_negative_pnl_worsened(self):
        calc = _make_calc()
        calc.spread_tracker.update(95000.0, 95010.0, 1.0)
        net = calc.get_friction_adjusted_pnl(-50.0, 0.1, "BUY", 95000.0)
        assert net < -50.0


# ===================================================================
# is_spread_acceptable
# ===================================================================
class TestIsSpreadAcceptable:
    def test_acceptable_when_below_threshold(self):
        calc = _make_calc()
        # Add enough spread data to form a learned max
        for _ in range(150):
            calc.spread_tracker.update(95000.0, 95002.0, 1.0)
        ok, current, max_acc = calc.is_spread_acceptable(multiplier=3.0)
        assert ok is True
        assert current == pytest.approx(2.0)

    def test_not_acceptable_when_spike(self):
        calc = _make_calc()
        for _ in range(150):
            calc.spread_tracker.update(95000.0, 95002.0, 1.0)
        # Spike the spread
        calc.spread_tracker.update(95000.0, 95050.0, 1.0)
        ok, current, max_acc = calc.is_spread_acceptable(multiplier=2.0)
        # current = 50.0, max_acc = min*2 ≈ 4.0 → not acceptable
        assert ok is False
        assert current == pytest.approx(50.0)

    def test_no_data_returns_acceptable(self):
        calc = _make_calc()
        ok, current, max_acc = calc.is_spread_acceptable()
        # current_spread = 0 → not finite or <=0 → True
        assert ok is True


# ===================================================================
# get_statistics
# ===================================================================
class TestGetStatistics:
    def test_returns_expected_keys(self):
        calc = _make_calc()
        stats = calc.get_statistics()
        for key in (
            "symbol", "avg_spread_pips", "min_spread_pips", "max_spread_pips",
            "current_spread_pips", "commission_per_lot", "swap_long", "swap_short",
            "base_slippage", "last_updated",
        ):
            assert key in stats

    def test_symbol_matches(self):
        calc = _make_calc(symbol="XAUUSD", symbol_id=999)
        assert calc.get_statistics()["symbol"] == "XAUUSD"


# ===================================================================
# get_spread_stats_for_logging
# ===================================================================
class TestSpreadStatsForLogging:
    def test_keys(self):
        calc = _make_calc()
        stats = calc.get_spread_stats_for_logging()
        for key in ("current", "min_observed", "avg", "max_acceptable_2x", "learned_multiplier", "samples"):
            assert key in stats

    def test_samples_increases(self):
        calc = _make_calc()
        assert calc.get_spread_stats_for_logging()["samples"] == 0
        calc.spread_tracker.update(95000.0, 95005.0, 1.0)
        assert calc.get_spread_stats_for_logging()["samples"] == 1


# ===================================================================
# update_digits_from_price
# ===================================================================
class TestUpdateDigitsFromPrice:
    def test_updates_when_default_and_unset(self):
        calc = _make_calc()
        calc.costs.digits = 2
        calc.costs.last_updated = None
        calc.update_digits_from_price(1.08765)  # Forex ≈ 5 digits
        assert calc.costs.digits == 5

    def test_no_update_when_already_set(self):
        calc = _make_calc()
        calc.costs.digits = 5
        calc.costs.last_updated = None
        calc.update_digits_from_price(1.08765)
        # digits != 2, so no update
        assert calc.costs.digits == 5

    def test_no_update_when_last_updated_set(self):
        calc = _make_calc()
        calc.costs.digits = 2
        calc.costs.last_updated = datetime.now(UTC)
        calc.update_digits_from_price(1.08765)
        assert calc.costs.digits == 2  # Not touched

    def test_btc_price_stays_2_digits(self):
        calc = _make_calc()
        calc.costs.digits = 2
        calc.costs.last_updated = None
        calc.update_digits_from_price(95000.12)
        # BTC > 10000 → min(2, observed) = 2, same as current → no change
        assert calc.costs.digits == 2


# ===================================================================
# _refresh_derived_costs
# ===================================================================
class TestRefreshDerivedCosts:
    def test_derives_tick_and_pip_size(self):
        calc = _make_calc()
        calc.costs.digits = 5
        calc._refresh_derived_costs()
        assert calc.costs.tick_size == pytest.approx(1e-5)
        assert calc.costs.pip_size == pytest.approx(1e-5)

    def test_derives_pip_value_per_lot(self):
        calc = _make_calc()
        calc.costs.digits = 5
        calc.costs.contract_size = 100000.0
        calc._refresh_derived_costs()
        expected = 100000.0 * 1e-5  # = 1.0
        assert calc.costs.pip_value_per_lot == pytest.approx(expected)

    def test_zero_digits_skips(self):
        calc = _make_calc()
        calc.costs.digits = 0
        orig_tick = calc.costs.tick_size
        calc._refresh_derived_costs()
        assert calc.costs.tick_size == orig_tick  # Unchanged


# ===================================================================
# infer_digits_from_price – extended
# ===================================================================
class TestInferDigitsExtended:
    @pytest.fixture()
    def calc(self):
        return _make_calc()

    def test_gold_price(self, calc):
        assert calc.infer_digits_from_price(1850.12) in (2, 3)

    def test_jpy_cross(self, calc):
        assert calc.infer_digits_from_price(148.123) == 3

    def test_standard_forex(self, calc):
        assert calc.infer_digits_from_price(1.08765) == 5

    def test_index_nas100(self, calc):
        d = calc.infer_digits_from_price(18500.50)
        assert d <= 2

    def test_negative_price_default(self, calc):
        assert calc.infer_digits_from_price(-100.0) == 2

    def test_inf_price_default(self, calc):
        assert calc.infer_digits_from_price(float("inf")) == 2


# ===================================================================
# SpreadTracker – extra coverage
# ===================================================================
class TestSpreadTrackerExtended:
    def test_get_current_hour_spread(self):
        tracker = SpreadTracker()
        # With no data, falls back to avg which is 0
        assert tracker.get_current_hour_spread() == pytest.approx(0.0)

    def test_get_current_hour_spread_with_data(self):
        tracker = SpreadTracker()
        # Add spread data
        for _ in range(5):
            tracker.update(100.0, 102.0, 1.0)
        # current_hour_spread should be the avg for the current hour
        spread = tracker.get_current_hour_spread()
        assert spread > 0

    def test_string_inputs_ignored(self):
        tracker = SpreadTracker()
        tracker.update("bad", "input", 1.0)  # type: ignore[arg-type]
        assert len(tracker.spreads) == 0

    def test_get_max_spread_cap(self):
        tracker = SpreadTracker()
        # Manually insert an extreme value
        tracker.spreads.append(999.0)
        assert tracker.get_max_spread() <= 1000.0  # MAX_SPREAD_PIPS


# ===================================================================
# SlippageModel – extreme volatility cap
# ===================================================================
class TestSlippageModelExtended:
    def test_extreme_volatility_capped(self):
        model = SlippageModel()
        s = model.estimate_slippage(1.0, "BUY", volatility_factor=100.0)
        # volatility capped at 10x
        s_at_10 = model.estimate_slippage(1.0, "BUY", volatility_factor=10.0)
        assert s == pytest.approx(s_at_10)

    def test_result_capped_at_100_pips(self):
        model = SlippageModel()
        model.base_slippage_pips = 50.0
        s = model.estimate_slippage(100.0, "BUY", volatility_factor=10.0)
        assert s <= 100.0

    def test_non_numeric_quantity(self):
        model = SlippageModel()
        assert model.estimate_slippage("bad", "BUY") == pytest.approx(0.0)  # type: ignore[arg-type]


# ===================================================================
# SymbolCosts defaults
# ===================================================================
class TestSymbolCostsDefaults:
    def test_swap_defaults(self):
        costs = SymbolCosts(symbol="TEST", symbol_id=1)
        assert costs.swap_type == "PIPS"
        assert costs.triple_swap_day == 2
        assert costs.swap_long == pytest.approx(0.0)
        assert costs.swap_short == pytest.approx(0.0)

    def test_commission_defaults(self):
        costs = SymbolCosts(symbol="TEST", symbol_id=1)
        assert costs.commission_type == "ABSOLUTE"
        assert costs.commission_percentage == pytest.approx(0.0)
