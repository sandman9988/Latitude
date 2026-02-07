"""
Tests for friction_costs defensive code paths.

Covers scattered uncovered production lines in:
- SpreadTracker (error handlers, edge cases)
- SlippageModel (invalid inputs, extreme volatility)
- FrictionCalculator (config loading, normalize methods, commission/swap modes)
"""

import math
import statistics
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.risk.friction_costs import (
    FrictionCalculator,
    SlippageModel,
    SpreadTracker,
    SymbolCosts,
    MAX_SPREAD_PIPS,
    MIN_SPREAD_SAMPLES,
)


# ===========================================================================
# SpreadTracker defensive paths
# ===========================================================================

class TestSpreadTrackerDefensive:
    """Test SpreadTracker error handlers and edge cases."""

    @pytest.fixture()
    def tracker(self):
        return SpreadTracker(window_size=200)

    def test_get_avg_spread_empty(self, tracker):
        """Empty spreads returns 0."""
        assert tracker.get_avg_spread() == pytest.approx(0.0)

    def test_get_min_spread_empty(self, tracker):
        """Empty spreads returns 0 for min."""
        assert tracker.get_min_spread() == pytest.approx(0.0)

    def test_get_max_spread_empty(self, tracker):
        """Empty spreads returns 0 for max."""
        assert tracker.get_max_spread() == pytest.approx(0.0)

    def test_get_current_spread_empty(self, tracker):
        """Empty spreads returns 0 for current."""
        assert tracker.get_current_spread() == pytest.approx(0.0)

    def test_get_avg_spread_non_finite_result(self, tracker):
        """If mean is non-finite, return 0."""
        # Manually inject spreads that could cause issues
        tracker.spreads.append(5.0)
        # Patch statistics.mean to return inf
        with patch("src.risk.friction_costs.statistics.mean", return_value=float("inf")):
            assert tracker.get_avg_spread() == pytest.approx(0.0)

    def test_get_avg_spread_negative_result(self, tracker):
        """If mean is negative (shouldn't happen), return 0."""
        tracker.spreads.append(5.0)
        with patch("src.risk.friction_costs.statistics.mean", return_value=-1.0):
            assert tracker.get_avg_spread() == pytest.approx(0.0)

    def test_get_avg_spread_statistics_error(self, tracker):
        """If statistics.mean raises, return 0."""
        tracker.spreads.append(5.0)
        with patch("src.risk.friction_costs.statistics.mean", side_effect=statistics.StatisticsError("no data")):
            assert tracker.get_avg_spread() == pytest.approx(0.0)

    def test_get_min_spread_value_error(self, tracker):
        """If min() raises ValueError, return 0."""
        tracker.spreads.append(5.0)
        with patch("builtins.min", side_effect=ValueError("bad")):
            assert tracker.get_min_spread() == pytest.approx(0.0)

    def test_get_max_spread_value_error(self, tracker):
        """If max() raises ValueError, return 0."""
        tracker.spreads.append(5.0)
        with patch("builtins.max", side_effect=ValueError("bad")):
            assert tracker.get_max_spread() == pytest.approx(0.0)

    def test_get_learned_max_insufficient_with_zero_current(self):
        """Insufficient data and zero current spread returns inf."""
        tracker = SpreadTracker()
        # Add some spreads but fewer than MIN_SPREAD_SAMPLES
        for i in range(10):
            tracker.update(100.0, 102.0, pip_size=1.0)
        assert len(tracker.spreads) < MIN_SPREAD_SAMPLES

        # Set current spread to 0 by manipulating the deque
        tracker.spreads.clear()
        tracker.spreads.append(0.0)  # Current spread is 0
        result = tracker.get_learned_max_spread()
        # When current <= 0, returns inf
        assert result == float("inf")

    def test_get_learned_max_with_enough_data(self):
        """With enough data, learned max is min_spread * multiplier."""
        tracker = SpreadTracker()
        for _ in range(MIN_SPREAD_SAMPLES + 10):
            tracker.update(100.0, 103.0, pip_size=1.0)  # spread = 3 pips

        result = tracker.get_learned_max_spread(multiplier=2.0)
        assert result == pytest.approx(6.0)

    def test_get_learned_max_with_zero_min_spread(self):
        """If min spread is 0, falls back to avg spread."""
        tracker = SpreadTracker()
        for _ in range(MIN_SPREAD_SAMPLES + 10):
            tracker.update(100.0, 105.0, pip_size=1.0)

        # Mock get_min_spread to return 0
        with patch.object(tracker, "get_min_spread", return_value=0.0):
            result = tracker.get_learned_max_spread(multiplier=2.0)
            # Falls back to avg spread * multiplier
            assert result > 0

    def test_hourly_avg_spread_fallback(self, tracker):
        """Unknown hour falls back to overall average."""
        tracker.update(100.0, 103.0, pip_size=1.0)
        # Hour 13 has data (current hour), but hour 5 might not
        result = tracker.get_hourly_avg_spread(5)
        # Should fall back to overall avg
        assert result == pytest.approx(tracker.get_avg_spread())

    def test_string_input_rejected(self, tracker):
        """Non-numeric input is rejected."""
        tracker.update("bad", 100.0, pip_size=1.0)
        assert len(tracker.spreads) == 0

    def test_extreme_spread_capped(self, tracker):
        """Spread exceeding MAX_SPREAD_PIPS is capped."""
        # bid=100, ask=2100, pip=1.0 => spread=2000 > 1000
        tracker.update(100.0, 2100.0, pip_size=1.0)
        assert tracker.get_current_spread() == pytest.approx(MAX_SPREAD_PIPS)


# ===========================================================================
# SlippageModel defensive paths
# ===========================================================================

class TestSlippageModelDefensive:
    """Test SlippageModel input validation and extreme cases."""

    @pytest.fixture()
    def model(self):
        return SlippageModel()

    def test_string_quantity_returns_zero(self, model):
        """Non-numeric quantity returns 0."""
        assert model.estimate_slippage("bad", "BUY") == pytest.approx(0.0)

    def test_none_quantity_returns_zero(self, model):
        """None quantity returns 0."""
        assert model.estimate_slippage(None, "BUY") == pytest.approx(0.0)

    def test_nan_quantity_returns_zero(self, model):
        """NaN quantity returns 0."""
        assert model.estimate_slippage(float("nan"), "BUY") == pytest.approx(0.0)

    def test_inf_quantity_returns_zero(self, model):
        """Inf quantity returns 0."""
        assert model.estimate_slippage(float("inf"), "BUY") == pytest.approx(0.0)

    def test_negative_quantity_returns_zero(self, model):
        """Negative quantity returns 0."""
        assert model.estimate_slippage(-1.0, "BUY") == pytest.approx(0.0)

    def test_zero_quantity_returns_zero(self, model):
        """Zero quantity returns 0."""
        assert model.estimate_slippage(0.0, "BUY") == pytest.approx(0.0)

    def test_extreme_volatility_capped(self, model):
        """Extreme volatility factor is capped at 10."""
        normal = model.estimate_slippage(1.0, "BUY", volatility_factor=1.0)
        extreme = model.estimate_slippage(1.0, "BUY", volatility_factor=100.0)
        capped = model.estimate_slippage(1.0, "BUY", volatility_factor=10.0)
        assert extreme == pytest.approx(capped)
        assert extreme > normal

    def test_negative_volatility_defaults_to_one(self, model):
        """Negative volatility factor defaults to 1.0."""
        default = model.estimate_slippage(1.0, "BUY", volatility_factor=1.0)
        neg_vol = model.estimate_slippage(1.0, "BUY", volatility_factor=-5.0)
        assert neg_vol == pytest.approx(default)

    def test_nan_volatility_defaults(self, model):
        """NaN volatility factor defaults to 1.0."""
        default = model.estimate_slippage(1.0, "BUY", volatility_factor=1.0)
        nan_vol = model.estimate_slippage(1.0, "BUY", volatility_factor=float("nan"))
        assert nan_vol == pytest.approx(default)

    def test_sell_side_multiplier(self, model):
        """SELL has different multiplier than BUY."""
        buy = model.estimate_slippage(1.0, "BUY")
        sell = model.estimate_slippage(1.0, "SELL")
        # BUY has 1.2x multiplier, SELL has 1.0x
        assert buy > sell

    def test_large_quantity_capped(self, model):
        """Quantity > 1000 lots is capped."""
        capped = model.estimate_slippage(5000.0, "BUY")
        at_cap = model.estimate_slippage(1000.0, "BUY")
        assert capped == pytest.approx(at_cap)

    def test_slippage_result_capped_at_100(self, model):
        """Result cannot exceed 100 pips."""
        # With extreme base_slippage
        model.base_slippage_pips = 500.0
        result = model.estimate_slippage(100.0, "BUY", volatility_factor=10.0)
        assert result <= 100.0


# ===========================================================================
# FrictionCalculator defensive paths
# ===========================================================================

class TestFrictionCalculatorDefensive:
    """Test FrictionCalculator config loading, normalization, and cost edge cases."""

    @pytest.fixture()
    def calc(self, tmp_path):
        """Create FrictionCalculator without relying on config file."""
        with patch("src.risk.friction_costs.FrictionCalculator._load_symbol_specs_from_config"):
            with patch("src.risk.friction_costs.FrictionCalculator._load_learned_parameters"):
                fc = FrictionCalculator(symbol="XAUUSD", symbol_id=10026)
        return fc

    def test_normalize_quantity_invalid_input(self, calc):
        """Invalid quantity normalizes to min_volume."""
        assert calc.normalize_quantity(float("nan")) == calc.costs.min_volume
        assert calc.normalize_quantity(float("inf")) == calc.costs.min_volume
        assert calc.normalize_quantity(-1.0) == calc.costs.min_volume
        assert calc.normalize_quantity(0.0) == calc.costs.min_volume

    def test_normalize_quantity_clamp_to_max(self, calc):
        """Quantity above max is clamped."""
        calc.costs.max_volume = 50.0
        assert calc.normalize_quantity(100.0) == pytest.approx(50.0)

    def test_normalize_quantity_snap_to_step(self, calc):
        """Quantity is snapped to volume_step."""
        calc.costs.volume_step = 0.1
        calc.costs.min_volume = 0.1
        # 0.35 / 0.1 = 3.5, round() = 4 on Python 3 banker's rounding? No, round(3.5)=4
        # Actually round(3.5)=4 in Python 3, so 4*0.1=0.4... but float precision
        # may give 0.3. Let's test with 0.36 which unambiguously rounds to 4.
        assert calc.normalize_quantity(0.36) == pytest.approx(0.4)

    def test_normalize_price_invalid(self, calc):
        """Invalid price returns 0."""
        assert calc.normalize_price(float("nan")) == pytest.approx(0.0)
        assert calc.normalize_price(float("inf")) == pytest.approx(0.0)
        assert calc.normalize_price(-100.0) == pytest.approx(0.0)
        assert calc.normalize_price(0.0) == pytest.approx(0.0)

    def test_normalize_price_valid(self, calc):
        """Valid price is rounded to tick size precision."""
        calc.costs.tick_size = 0.01
        result = calc.normalize_price(1850.1234)
        assert result == pytest.approx(1850.12)

    def test_normalize_price_zero_tick(self, calc):
        """If tick_size is 0, fall back to digits-based rounding."""
        calc.costs.tick_size = 0.0
        calc.costs.digits = 2
        result = calc.normalize_price(1850.1234)
        assert result == pytest.approx(1850.12)

    def test_calculate_commission_percentage(self, calc):
        """Test commission calculation with PERCENTAGE type."""
        calc.costs.commission_type = "PERCENTAGE"
        calc.costs.commission_percentage = 0.0007  # 0.07%
        comm = calc.calculate_commission(0.1, 2000.0)
        assert comm > 0

    def test_calculate_commission_invalid_inputs(self, calc):
        """Invalid inputs return 0."""
        assert calc.calculate_commission("bad", 100.0) == pytest.approx(0.0)
        assert calc.calculate_commission(0.1, float("nan")) == pytest.approx(0.0)
        assert calc.calculate_commission(-1.0, 100.0) == pytest.approx(0.0)
        assert calc.calculate_commission(0.0, 100.0) == pytest.approx(0.0)

    def test_calculate_commission_non_finite_pct(self, calc):
        """Non-finite commission percentage uses default."""
        calc.costs.commission_type = "PERCENTAGE"
        calc.costs.commission_percentage = float("nan")
        comm = calc.calculate_commission(0.1, 2000.0)
        assert comm >= 0

    def test_calculate_spread_cost_invalid(self, calc):
        """Invalid inputs return 0."""
        assert calc.calculate_spread_cost("bad") == pytest.approx(0.0)
        assert calc.calculate_spread_cost(float("nan")) == pytest.approx(0.0)
        assert calc.calculate_spread_cost(-1.0) == pytest.approx(0.0)
        assert calc.calculate_spread_cost(0.0) == pytest.approx(0.0)

    def test_calculate_swap_intraday_no_rollover(self, calc):
        """Intraday trade without rollover crossing: swap = 0."""
        swap = calc.calculate_swap(0.1, "BUY", holding_days=0.5, crosses_rollover=False)
        assert swap == pytest.approx(0.0)

    def test_calculate_swap_crosses_rollover(self, calc):
        """Overnight trade crossing rollover incurs swap."""
        calc.costs.swap_long = -7.2
        calc.costs.pip_value_per_lot = 1.0
        swap = calc.calculate_swap(0.1, "BUY", holding_days=1.0, crosses_rollover=True)
        assert swap != 0.0

    def test_calculate_swap_percentage_type(self, calc):
        """Test swap calculation with PERCENTAGE type."""
        calc.costs.swap_type = "PERCENTAGE"
        calc.costs.swap_long = -2.5  # -2.5% annual
        calc.costs.contract_size = 100.0
        swap = calc.calculate_swap(0.1, "BUY", holding_days=1.0, crosses_rollover=True, price=1850.0)
        assert swap != 0.0

    def test_calculate_swap_percentage_zero_price(self, calc):
        """PERCENTAGE swap with price=0 returns 0."""
        calc.costs.swap_type = "PERCENTAGE"
        swap = calc.calculate_swap(0.1, "BUY", holding_days=1.0, crosses_rollover=True, price=0.0)
        assert swap == pytest.approx(0.0)

    def test_calculate_swap_unknown_type(self, calc):
        """Unknown swap type returns 0."""
        calc.costs.swap_type = "POINTS"  # Not PIPS or PERCENTAGE
        swap = calc.calculate_swap(0.1, "BUY", holding_days=1.0, crosses_rollover=True)
        assert swap == pytest.approx(0.0)

    def test_calculate_swap_invalid_triple_swap_day(self, calc):
        """Invalid triple_swap_day defaults to Wednesday (2)."""
        calc.costs.triple_swap_day = "invalid"
        calc.costs.swap_long = -5.0
        calc.costs.pip_value_per_lot = 1.0
        # Should not raise
        swap = calc.calculate_swap(0.1, "BUY", holding_days=1.0, crosses_rollover=True)

    def test_calculate_total_friction(self, calc):
        """Total friction is sum of all components."""
        calc.spread_tracker.update(1850.0, 1851.0, pip_size=0.01)
        result = calc.calculate_total_friction(0.1, "BUY", 1850.0)
        assert "spread" in result
        assert "commission" in result
        assert "swap" in result
        assert "slippage" in result
        assert "total" in result
        assert result["total"] >= 0

    def test_get_friction_adjusted_pnl(self, calc):
        """Friction-adjusted PnL subtracts total friction."""
        net = calc.get_friction_adjusted_pnl(100.0, 0.1, "BUY", 1850.0)
        assert net <= 100.0

    def test_get_statistics(self, calc):
        """get_statistics returns expected keys."""
        stats = calc.get_statistics()
        assert "symbol" in stats
        assert "avg_spread_pips" in stats
        assert "swap_long" in stats

    def test_get_symbol_info(self, calc):
        """get_symbol_info returns expected keys."""
        info = calc.get_symbol_info()
        assert info["symbol"] == "XAUUSD"
        assert "digits" in info

    def test_update_symbol_costs(self, calc):
        """update_symbol_costs updates cost fields."""
        calc.update_symbol_costs(digits=5, min_volume=0.01, swap_long=-3.0)
        assert calc.costs.digits == 5
        assert calc.costs.swap_long == pytest.approx(-3.0)

    def test_is_spread_acceptable_no_data(self, calc):
        """With no spread data, trading is allowed."""
        ok, current, max_acc = calc.is_spread_acceptable()
        assert ok is True

    def test_is_spread_acceptable_non_finite_multiplier(self, calc):
        """Non-finite multiplier defaults to 2.0."""
        ok, _, _ = calc.is_spread_acceptable(multiplier=float("nan"))
        assert ok is True

    def test_infer_digits_from_price(self, calc):
        """Infer digits for various price magnitudes."""
        assert calc.infer_digits_from_price(95000.12) <= 2  # BTC
        assert calc.infer_digits_from_price(1850.12) <= 2  # Gold
        assert calc.infer_digits_from_price(148.123) <= 3  # JPY
        assert calc.infer_digits_from_price(1.08765) <= 5  # EUR
        assert calc.infer_digits_from_price(float("nan")) == 2  # invalid = default
        assert calc.infer_digits_from_price(-100) == 2

    def test_update_digits_from_price(self, calc):
        """update_digits_from_price only updates if default."""
        calc.costs.digits = 2
        calc.costs.last_updated = None
        calc.update_digits_from_price(1.08765)
        # Should infer 5 digits for forex
        assert calc.costs.digits == 5

    def test_update_digits_skipped_if_already_set(self, calc):
        """update_digits_from_price skips if already updated by broker."""
        calc.costs.digits = 2
        calc.costs.last_updated = datetime.now(UTC)
        calc.update_digits_from_price(1.08765)
        # Should NOT update because last_updated is set
        assert calc.costs.digits == 2

    def test_update_spread(self, calc):
        """update_spread tracks spread and updates cost stats."""
        calc.costs.pip_size = 0.01
        calc.update_spread(1850.00, 1850.50)
        assert calc.costs.avg_spread_pips > 0

    def test_get_spread_stats_for_logging(self, calc):
        """get_spread_stats_for_logging returns all expected keys."""
        stats = calc.get_spread_stats_for_logging()
        assert "current" in stats
        assert "min_observed" in stats
        assert "samples" in stats
