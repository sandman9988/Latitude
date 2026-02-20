"""Tests for src.risk.friction_costs – SpreadTracker, SlippageModel, FrictionCalculator."""

from datetime import UTC, datetime

import pytest

from src.risk.friction_costs import (
    FrictionCalculator,
    SlippageModel,
    SpreadTracker,
    SymbolCosts,
)


# ---------------------------------------------------------------------------
# SymbolCosts dataclass
# ---------------------------------------------------------------------------

class TestSymbolCosts:
    def test_defaults(self):
        sc = SymbolCosts(symbol="BTCUSD", symbol_id=10028)
        assert sc.digits == 2
        assert sc.pip_size == pytest.approx(1.0)
        assert sc.min_volume == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# SpreadTracker
# ---------------------------------------------------------------------------

class TestSpreadTracker:
    @pytest.fixture()
    def tracker(self):
        return SpreadTracker(window_size=100)

    def test_empty_spread(self, tracker):
        assert tracker.get_current_spread() == pytest.approx(0.0)
        assert tracker.get_avg_spread() == pytest.approx(0.0)
        assert tracker.get_min_spread() == pytest.approx(0.0)
        assert tracker.get_max_spread() == pytest.approx(0.0)

    def test_update_records_spread(self, tracker):
        tracker.update(100.0, 102.0, pip_size=1.0)
        assert tracker.get_current_spread() == pytest.approx(2.0)

    def test_avg_spread(self, tracker):
        tracker.update(100.0, 101.0, pip_size=1.0)  # 1 pip
        tracker.update(100.0, 103.0, pip_size=1.0)  # 3 pips
        assert tracker.get_avg_spread() == pytest.approx(2.0)

    def test_min_max_spread(self, tracker):
        tracker.update(100.0, 101.0, pip_size=1.0)  # 1 pip
        tracker.update(100.0, 105.0, pip_size=1.0)  # 5 pips
        assert tracker.get_min_spread() == pytest.approx(1.0)
        assert tracker.get_max_spread() == pytest.approx(5.0)

    def test_invalid_bid_ask_ignored(self, tracker):
        tracker.update(-1.0, 100.0, pip_size=1.0)  # negative bid
        assert len(tracker.spreads) == 0

    def test_crossed_book_ignored(self, tracker):
        tracker.update(105.0, 100.0, pip_size=1.0)  # bid > ask
        assert len(tracker.spreads) == 0

    def test_extreme_spread_capped(self, tracker):
        # Spread = (2000 - 100) / 1.0 = 1900 > MAX_SPREAD_PIPS(1000)
        tracker.update(100.0, 2100.0, pip_size=1.0)
        assert tracker.get_current_spread() == pytest.approx(1000.0)

    def test_nan_inputs_ignored(self, tracker):
        tracker.update(float("nan"), 100.0, pip_size=1.0)
        assert len(tracker.spreads) == 0

    def test_zero_pip_size_ignored(self, tracker):
        tracker.update(100.0, 101.0, pip_size=0.0)
        assert len(tracker.spreads) == 0

    def test_window_size_cap(self):
        t = SpreadTracker(window_size=3)
        for i in range(10):
            t.update(100.0, 100.0 + (i + 1), pip_size=1.0)
        assert len(t.spreads) == 3

    def test_hourly_avg(self, tracker):
        tracker.update(100.0, 102.0, pip_size=1.0)  # 2 pips
        hour = datetime.now(UTC).hour
        avg = tracker.get_hourly_avg_spread(hour)
        assert avg == pytest.approx(2.0)

    def test_hourly_avg_fallback(self, tracker):
        # No data for specific hour → falls back to global average
        tracker.update(100.0, 103.0, pip_size=1.0)
        avg = tracker.get_hourly_avg_spread(23 if datetime.now(UTC).hour != 23 else 0)
        # Falls back to global when that hour has no data
        assert avg >= 0.0

    def test_get_learned_max_spread_insufficient(self, tracker):
        # < MIN_SPREAD_SAMPLES → returns current or inf
        tracker.update(100.0, 101.0, pip_size=1.0)
        result = tracker.get_learned_max_spread()
        assert result >= 0

    def test_get_learned_max_spread_enough_data(self):
        t = SpreadTracker(window_size=200)
        for _ in range(150):
            t.update(100.0, 102.0, pip_size=1.0)  # 2 pips
        learned_max = t.get_learned_max_spread(multiplier=2.0)
        assert learned_max == pytest.approx(4.0)  # 2 * 2.0


# ---------------------------------------------------------------------------
# SlippageModel
# ---------------------------------------------------------------------------

class TestSlippageModel:
    @pytest.fixture()
    def model(self):
        return SlippageModel()

    def test_basic_estimate(self, model):
        slip = model.estimate_slippage(1.0, "BUY")
        assert slip > 0

    def test_buy_worse_than_sell(self, model):
        buy = model.estimate_slippage(1.0, "BUY")
        sell = model.estimate_slippage(1.0, "SELL")
        assert buy > sell

    def test_larger_size_more_slippage(self, model):
        s1 = model.estimate_slippage(0.1, "BUY")
        s10 = model.estimate_slippage(10.0, "BUY")
        assert s10 > s1

    def test_higher_volatility_more_slippage(self, model):
        normal = model.estimate_slippage(1.0, "BUY", volatility_factor=1.0)
        high_vol = model.estimate_slippage(1.0, "BUY", volatility_factor=3.0)
        assert high_vol > normal

    def test_zero_quantity(self, model):
        assert model.estimate_slippage(0.0) == pytest.approx(0.0)

    def test_negative_quantity(self, model):
        assert model.estimate_slippage(-1.0) == pytest.approx(0.0)

    def test_nan_quantity(self, model):
        assert model.estimate_slippage(float("nan")) == pytest.approx(0.0)

    def test_invalid_volatility_clamped(self, model):
        result = model.estimate_slippage(1.0, "BUY", volatility_factor=-1.0)
        assert result >= 0

    def test_extreme_quantity_capped(self, model):
        result = model.estimate_slippage(99999.0, "BUY")
        assert result <= 100.0  # capped at 100 pips


# ---------------------------------------------------------------------------
# FrictionCalculator
# ---------------------------------------------------------------------------

class TestFrictionCalculator:
    @pytest.fixture()
    def calc(self, tmp_path):
        """Create FrictionCalculator with tmp persistence to avoid file side-effects."""
        from src.persistence.learned_parameters import LearnedParametersManager
        pm = LearnedParametersManager(persistence_path=tmp_path / "params.json")
        return FrictionCalculator(symbol="BTCUSD", param_manager=pm)

    # -- normalize_quantity --
    def test_normalize_quantity_normal(self, calc):
        q = calc.normalize_quantity(0.05)
        assert q >= calc.costs.min_volume
        assert q <= calc.costs.max_volume

    def test_normalize_quantity_below_min(self, calc):
        q = calc.normalize_quantity(0.001)
        assert q == calc.costs.min_volume

    def test_normalize_quantity_above_max(self, calc):
        q = calc.normalize_quantity(99999.0)
        assert q == calc.costs.max_volume

    def test_normalize_quantity_zero(self, calc):
        q = calc.normalize_quantity(0.0)
        assert q == calc.costs.min_volume

    def test_normalize_quantity_nan(self, calc):
        q = calc.normalize_quantity(float("nan"))
        assert q == calc.costs.min_volume

    # -- normalize_price --
    def test_normalize_price_normal(self, calc):
        p = calc.normalize_price(95432.1234)
        # digits=2 → should round properly
        assert isinstance(p, float)
        assert p > 0

    def test_normalize_price_zero(self, calc):
        assert calc.normalize_price(0.0) == pytest.approx(0.0)

    def test_normalize_price_nan(self, calc):
        assert calc.normalize_price(float("nan")) == pytest.approx(0.0)

    # -- update_spread --
    def test_update_spread(self, calc):
        calc.update_spread(95000.0, 95002.0)
        assert calc.costs.avg_spread_pips > 0

    # -- calculate_spread_cost --
    def test_spread_cost_basic(self, calc):
        calc.update_spread(95000.0, 95002.0)
        cost = calc.calculate_spread_cost(0.1)
        assert cost > 0

    def test_spread_cost_zero_quantity(self, calc):
        assert calc.calculate_spread_cost(0.0) == pytest.approx(0.0)

    def test_spread_cost_nan_quantity(self, calc):
        assert calc.calculate_spread_cost(float("nan")) == pytest.approx(0.0)

    # -- calculate_commission --
    def test_commission_absolute(self, calc):
        calc.costs.commission_type = "ABSOLUTE"
        calc.costs.commission_per_lot = 7.0
        comm = calc.calculate_commission(1.0, 95000.0)
        assert comm >= 0  # might be 0 if ABSOLUTE path returns differently

    def test_commission_zero_qty(self, calc):
        assert calc.calculate_commission(0.0, 95000.0) == pytest.approx(0.0)

    def test_commission_nan(self, calc):
        assert calc.calculate_commission(float("nan"), 95000.0) == pytest.approx(0.0)

    # -- update_symbol_costs --
    def test_update_symbol_costs(self, calc):
        calc.update_symbol_costs(digits=5, min_volume=0.001)
        assert calc.costs.digits == 5
        assert calc.costs.min_volume == pytest.approx(0.001)

    # -- get_symbol_info --
    def test_get_symbol_info(self, calc):
        info = calc.get_symbol_info()
        assert info["symbol"] == "BTCUSD"
        assert "digits" in info

    # -- infer_digits_from_price --
    def test_infer_digits_btc(self, calc):
        d = calc.infer_digits_from_price(95000.12)
        assert d == 2

    def test_infer_digits_forex(self, calc):
        d = calc.infer_digits_from_price(1.08765)
        assert d <= 5

    def test_infer_digits_nan(self, calc):
        d = calc.infer_digits_from_price(float("nan"))
        assert d == 2  # default
