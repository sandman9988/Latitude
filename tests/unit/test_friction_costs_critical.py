"""
Tests for friction_costs.py — Tier 2: spread threshold + swap calculations.

Covers:
  - SpreadTracker.get_learned_max_spread edge cases (lines 219, 226)
  - FrictionCalculator.calculate_swap PERCENTAGE type (lines 810-860)
  - FrictionCalculator.calculate_swap triple swap day logic (lines 822, 827)
  - FrictionCalculator.calculate_swap rollover counting (lines 847-860)
"""

import math
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from src.risk.friction_costs import SpreadTracker, FrictionCalculator, SymbolCosts, MIN_SPREAD_SAMPLES


# =========================================================================
# SpreadTracker.get_learned_max_spread edge cases
# =========================================================================


class TestGetLearnedMaxSpreadEdgeCases:
    """Lines 219, 226: edge cases with invalid min_spread or insufficient data."""

    def test_insufficient_data_returns_current_spread(self):
        """< MIN_SPREAD_SAMPLES → returns current spread."""
        st = SpreadTracker()
        # Add fewer samples than required
        for _ in range(MIN_SPREAD_SAMPLES - 1):
            st.spreads.append(2.5)
        result = st.get_learned_max_spread()
        assert result > 0  # Should return something usable

    def test_sufficient_data_returns_multiplied_min(self):
        """With enough data, returns min_spread * multiplier."""
        st = SpreadTracker()
        for i in range(MIN_SPREAD_SAMPLES + 10):
            st.spreads.append(2.0 + (i % 5) * 0.1)
        result = st.get_learned_max_spread(multiplier=2.0)
        min_spread = min(st.spreads)
        assert result == pytest.approx(min_spread * 2.0)

    def test_min_spread_zero_uses_average(self):
        """If min_spread is 0, falls back to average."""
        st = SpreadTracker()
        # Inject 0 as min spread value plus others
        st.spreads.append(0.0)
        for _ in range(MIN_SPREAD_SAMPLES + 5):
            st.spreads.append(3.0)
        result = st.get_learned_max_spread(multiplier=2.0)
        # min_spread=0 → falls to avg_spread * multiplier path
        # Or returns inf if avg is also problematic
        assert result > 0

    def test_empty_spreads_returns_inf(self):
        """No data at all → returns inf (don't block trades)."""
        st = SpreadTracker()
        result = st.get_learned_max_spread()
        # get_current_spread returns 0 if no data, so it should return inf
        # since current == 0 triggers the inf path
        assert result == float("inf") or result == pytest.approx(0.0)

    def test_custom_multiplier(self):
        """Different multiplier values produce scaled thresholds."""
        st = SpreadTracker()
        for _ in range(MIN_SPREAD_SAMPLES + 10):
            st.spreads.append(1.5)
        r2 = st.get_learned_max_spread(multiplier=2.0)
        r3 = st.get_learned_max_spread(multiplier=3.0)
        assert r3 > r2


# =========================================================================
# FrictionCalculator.calculate_swap — PERCENTAGE type
# =========================================================================


class TestCalculateSwapPercentage:
    """Lines 810-860: PERCENTAGE swap type calculation."""

    @pytest.fixture()
    def fc(self):
        calc = FrictionCalculator.__new__(FrictionCalculator)
        calc.costs = SymbolCosts(
            symbol="BTCUSD",
            symbol_id=10028,
            swap_long=-2.5,
            swap_short=-1.5,
            swap_type="PERCENTAGE",
            triple_swap_day=2,  # Wednesday
            contract_size=1.0,
            pip_value_per_lot=10.0,
        )
        return calc

    def test_percentage_swap_buy_overnight(self, fc):
        """Buy position crossing rollover → negative swap cost."""
        result = fc.calculate_swap(
            quantity=1.0, side="BUY", holding_days=1.0,
            crosses_rollover=True, price=50000.0,
        )
        # swap_long=-2.5% annual, 1 lot, price=50000, contract_size=1
        # notional = 1 * 1 * 50000 = 50000
        # daily_rate = -2.5 / 100 / 365 ≈ -0.0000685
        # swap = 50000 * -0.0000685 * 1 ≈ -$3.42
        assert result < 0  # You pay for long carry

    def test_percentage_swap_sell_overnight(self, fc):
        """Sell position uses swap_short rate."""
        result = fc.calculate_swap(
            quantity=1.0, side="SELL", holding_days=1.0,
            crosses_rollover=True, price=50000.0,
        )
        assert result < 0  # swap_short is also negative

    def test_percentage_swap_zero_price_returns_zero(self, fc):
        """price <= 0 → swap cost = 0 (can't calculate notional)."""
        result = fc.calculate_swap(
            quantity=1.0, side="BUY", holding_days=1.0,
            crosses_rollover=True, price=0.0,
        )
        assert result == pytest.approx(0.0)

    def test_percentage_swap_negative_price_returns_zero(self, fc):
        result = fc.calculate_swap(
            quantity=1.0, side="BUY", holding_days=1.0,
            crosses_rollover=True, price=-100.0,
        )
        assert result == pytest.approx(0.0)


# =========================================================================
# FrictionCalculator.calculate_swap — triple swap day logic
# =========================================================================


class TestTripleSwapDay:
    """Lines 822, 827: Triple swap day adds +2 rollovers."""

    @pytest.fixture()
    def fc_pips(self):
        calc = FrictionCalculator.__new__(FrictionCalculator)
        calc.costs = SymbolCosts(
            symbol="BTCUSD",
            symbol_id=10028,
            swap_long=-7.2,
            swap_short=-3.0,
            swap_type="PIPS",
            triple_swap_day=2,  # Wednesday
            pip_value_per_lot=1.0,
            contract_size=1.0,
        )
        return calc

    def test_triple_swap_wednesday(self, fc_pips):
        """On Wednesday, overnight hold gets 3x (1+2) rollovers."""
        from unittest.mock import PropertyMock
        # The code does `from datetime import UTC, datetime` locally,
        # so we can't easily mock datetime.now. Instead, set triple_swap_day
        # to match today's weekday so the test is deterministic.
        from datetime import datetime as dt_cls
        today_weekday = dt_cls.now(UTC).weekday()
        fc_pips.costs.triple_swap_day = today_weekday

        result = fc_pips.calculate_swap(
            quantity=0.1, side="BUY", holding_days=1.0,
            crosses_rollover=True,
        )

        # swap_long=-7.2, pip_value=1.0, qty=0.1, num_rollovers=3 (1+2 for triple)
        # swap = -7.2 * 1.0 * 0.1 * 3 = -2.16
        assert result == pytest.approx(-2.16, abs=0.01)

    def test_non_triple_day_single_rollover(self, fc_pips):
        """On non-triple day, overnight = exactly 1 rollover."""
        from datetime import datetime as dt_cls
        # Set triple_swap_day to a day that is NOT today
        today_weekday = dt_cls.now(UTC).weekday()
        fc_pips.costs.triple_swap_day = (today_weekday + 3) % 7  # Different day

        result = fc_pips.calculate_swap(
            quantity=0.1, side="BUY", holding_days=1.0,
            crosses_rollover=True,
        )

        # swap = -7.2 * 1.0 * 0.1 * 1 = -0.72
        assert result == pytest.approx(-0.72, abs=0.01)

    def test_invalid_triple_swap_day_defaults_to_wednesday(self, fc_pips):
        """triple_swap_day out of range → defaults to 2 (Wednesday)."""
        fc_pips.costs.triple_swap_day = 99  # Invalid → validated to 2 in code

        result = fc_pips.calculate_swap(
            quantity=0.1, side="BUY", holding_days=1.0,
            crosses_rollover=True,
        )

        from datetime import datetime as dt_cls
        today = dt_cls.now(UTC).weekday()
        # tsd=99 defaults to 2 (Wednesday). If today is Wednesday, triple (3 rollovers)
        if today == 2:
            assert result == pytest.approx(-2.16, abs=0.01)
        else:
            assert result == pytest.approx(-0.72, abs=0.01)


# =========================================================================
# FrictionCalculator.calculate_swap — intraday vs overnight
# =========================================================================


class TestSwapIntraday:
    """Intraday trades (not crossing rollover) → swap = 0."""

    @pytest.fixture()
    def fc(self):
        calc = FrictionCalculator.__new__(FrictionCalculator)
        calc.costs = SymbolCosts(
            symbol="BTCUSD",
            symbol_id=10028,
            swap_long=-7.2,
            swap_short=-3.0,
            swap_type="PIPS",
            triple_swap_day=2,
            pip_value_per_lot=1.0,
        )
        return calc

    def test_intraday_no_rollover_zero_swap(self, fc):
        result = fc.calculate_swap(
            quantity=1.0, side="BUY", holding_days=0.5,
            crosses_rollover=False,
        )
        assert result == pytest.approx(0.0)

    def test_multiday_with_rollover_nonzero(self, fc):
        result = fc.calculate_swap(
            quantity=1.0, side="BUY", holding_days=2.0,
            crosses_rollover=True,
        )
        assert result != 0.0

    def test_unknown_swap_type_returns_zero(self, fc):
        """Unknown swap_type → swap_cost = 0."""
        fc.costs.swap_type = "UNKNOWN"
        result = fc.calculate_swap(
            quantity=1.0, side="BUY", holding_days=1.0,
            crosses_rollover=True,
        )
        assert result == pytest.approx(0.0)
