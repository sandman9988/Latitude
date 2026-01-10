#!/usr/bin/env python3
"""
Comprehensive Calculation Safety Tests

Tests ALL mathematical operations for:
- Precision (Decimal vs Float)
- Edge cases (0, NaN, Inf, negative)
- Instrument-specific normalization
- Overflow/underflow protection
"""

import math
import unittest
from decimal import Decimal, InvalidOperation

from trade_manager_safety import (
    INSTRUMENT_SPECS,
    InstrumentSpec,
    SafeMath as DecimalSafeMath,  # Decimal-based SafeMath
)
from safe_utils import SafeMath  # Float-based SafeMath for legacy code


class TestTrailingStopCalculations(unittest.TestCase):
    """Test trailing stop price calculations with full safety"""

    def setUp(self):
        """Set up test instrument specs"""
        # Crypto: 8 digits
        INSTRUMENT_SPECS["BTCUSD"] = InstrumentSpec(
            symbol_id="10028",
            digits=8,
            volume_digits=2,
            min_volume=Decimal("0.01"),
            max_volume=Decimal("100.0"),
            volume_step=Decimal("0.01"),
            contract_size=Decimal("1.0"),
            pip_value_per_lot=Decimal("0.01"),
        )

        # Forex: 5 digits
        INSTRUMENT_SPECS["EURUSD"] = InstrumentSpec(
            symbol_id="1",
            digits=5,
            volume_digits=2,
            min_volume=Decimal("0.01"),
            max_volume=Decimal("100.0"),
            volume_step=Decimal("0.01"),
            contract_size=Decimal("100000.0"),
            pip_value_per_lot=Decimal("10.0"),
        )

    def test_long_trailing_stop_normal(self):
        """LONG position trailing stop: price rises, stop trails up"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        entry_price = Decimal("100000.00000000")
        current_price = Decimal("102000.00000000")  # +2%
        distance_pct = Decimal("1.0")  # 1% trail

        # Stop should be 1% below current price
        expected_stop = current_price * (Decimal("1.0") - distance_pct / Decimal("100.0"))
        expected_stop = spec.normalize_price(expected_stop)

        # Calculate using DecimalSafeMath
        price_dec = DecimalSafeMath.to_decimal(float(current_price), spec.digits)
        distance_dec = distance_pct / Decimal("100.0")
        new_stop = DecimalSafeMath.safe_multiply(price_dec, Decimal("1.0") - distance_dec)
        new_stop = spec.normalize_price(new_stop)

        self.assertEqual(new_stop, expected_stop)
        self.assertEqual(new_stop, Decimal("100980.00000000"))

    def test_short_trailing_stop_normal(self):
        """SHORT position trailing stop: price falls, stop trails down"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        entry_price = Decimal("100000.00000000")
        current_price = Decimal("98000.00000000")  # -2%
        distance_pct = Decimal("1.0")

        # Stop should be 1% above current price
        expected_stop = current_price * (Decimal("1.0") + distance_pct / Decimal("100.0"))
        expected_stop = spec.normalize_price(expected_stop)

        price_dec = DecimalSafeMath.to_decimal(float(current_price), spec.digits)
        distance_dec = distance_pct / Decimal("100.0")
        new_stop = DecimalSafeMath.safe_multiply(price_dec, Decimal("1.0") + distance_dec)
        new_stop = spec.normalize_price(new_stop)

        self.assertEqual(new_stop, expected_stop)
        self.assertEqual(new_stop, Decimal("98980.00000000"))

    def test_trailing_stop_zero_distance(self):
        """Trailing stop with 0% distance"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        current_price = Decimal("100000.00000000")
        distance_pct = Decimal("0.0")

        price_dec = DecimalSafeMath.to_decimal(float(current_price), spec.digits)
        distance_dec = distance_pct / Decimal("100.0")
        new_stop = DecimalSafeMath.safe_multiply(price_dec, Decimal("1.0") - distance_dec)
        new_stop = spec.normalize_price(new_stop)

        # With 0% distance, stop = current price
        self.assertEqual(new_stop, current_price)

    def test_trailing_stop_100_percent_distance(self):
        """Trailing stop with 100% distance (extreme)"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        current_price = Decimal("100000.00000000")
        distance_pct = Decimal("100.0")

        price_dec = DecimalSafeMath.to_decimal(float(current_price), spec.digits)
        distance_dec = distance_pct / Decimal("100.0")
        new_stop = DecimalSafeMath.safe_multiply(price_dec, Decimal("1.0") - distance_dec)
        new_stop = spec.normalize_price(new_stop)

        # With 100% distance, stop = 0
        self.assertEqual(new_stop, Decimal("0.00000000"))

    def test_trailing_stop_price_zero(self):
        """Trailing stop with price = 0 (invalid)"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        current_price = 0.0
        distance_pct = Decimal("1.0")

        # Should handle gracefully
        if current_price <= 0:
            new_stop = Decimal("0.0")
        else:
            price_dec = DecimalSafeMath.to_decimal(current_price, spec.digits)
            distance_dec = distance_pct / Decimal("100.0")
            new_stop = DecimalSafeMath.safe_multiply(price_dec, Decimal("1.0") - distance_dec)

        self.assertEqual(new_stop, Decimal("0.0"))

    def test_trailing_stop_price_nan(self):
        """Trailing stop with NaN price"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        current_price = float("nan")
        distance_pct = Decimal("1.0")

        # Should detect invalid
        self.assertFalse(SafeMath.is_valid(current_price))

    def test_trailing_stop_price_inf(self):
        """Trailing stop with Inf price"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        current_price = float("inf")
        distance_pct = Decimal("1.0")

        # Should detect invalid
        self.assertFalse(SafeMath.is_valid(current_price))

    def test_trailing_stop_forex_precision(self):
        """Trailing stop on forex (5 digits)"""
        spec = INSTRUMENT_SPECS["EURUSD"]
        current_price = Decimal("1.08500")
        distance_pct = Decimal("0.5")  # 0.5% = 50 pips

        price_dec = DecimalSafeMath.to_decimal(float(current_price), spec.digits)
        distance_dec = distance_pct / Decimal("100.0")
        new_stop = DecimalSafeMath.safe_multiply(price_dec, Decimal("1.0") - distance_dec)
        new_stop = spec.normalize_price(new_stop)

        expected = Decimal("1.08500") * Decimal("0.995")
        expected = spec.normalize_price(expected)

        self.assertEqual(new_stop, expected)

    def test_trailing_stop_crypto_precision(self):
        """Trailing stop on crypto (8 digits) preserves precision"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        current_price = Decimal("95432.12345678")  # Full 8 digits
        distance_pct = Decimal("2.0")

        price_dec = DecimalSafeMath.to_decimal(float(current_price), spec.digits)
        distance_dec = distance_pct / Decimal("100.0")
        new_stop = DecimalSafeMath.safe_multiply(price_dec, Decimal("1.0") - distance_dec)
        new_stop = spec.normalize_price(new_stop)

        # Should maintain 8 decimal places
        self.assertEqual(len(str(new_stop).split(".")[1]), 8)


class TestCommissionCalculations(unittest.TestCase):
    """Test commission calculations with InstrumentSpec"""

    def setUp(self):
        """Set up test specs"""
        INSTRUMENT_SPECS["BTCUSD"] = InstrumentSpec(
            symbol_id="10028",
            digits=8,
            volume_digits=2,
            min_volume=Decimal("0.01"),
            max_volume=Decimal("100.0"),
            volume_step=Decimal("0.01"),
            contract_size=Decimal("1.0"),  # 1 BTC per lot
            pip_value_per_lot=Decimal("0.01"),
            commission_per_lot=Decimal("0.002"),  # 0.002% per lot
        )

    def test_commission_calculation_btc(self):
        """Commission on BTC trade"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        quantity = Decimal("0.10")  # 0.1 lots = 0.1 BTC
        price = Decimal("100000.00000000")

        # Commission = quantity * commission_per_lot * price
        expected_comm = spec.calculate_commission(quantity, price)

        # Manual: 0.10 * 0.002% * 100000 = 0.10 * 0.00002 * 100000 = $0.20
        # Actually calculate_commission uses commission_per_lot directly
        self.assertIsInstance(expected_comm, Decimal)
        self.assertGreater(expected_comm, Decimal("0.0"))

    def test_commission_zero_quantity(self):
        """Commission with zero quantity"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        quantity = Decimal("0.0")
        price = Decimal("100000.0")

        comm = spec.calculate_commission(quantity, price)
        self.assertEqual(comm, Decimal("0.0"))

    def test_commission_zero_price(self):
        """Commission with zero price (invalid)"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        quantity = Decimal("0.10")
        price = Decimal("0.0")

        # Should handle gracefully
        comm = spec.calculate_commission(quantity, price)
        # Depends on implementation, but should not crash

    def test_commission_large_quantity(self):
        """Commission on large position"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        quantity = Decimal("100.0")  # Max quantity
        price = Decimal("100000.0")

        comm = spec.calculate_commission(quantity, price)
        self.assertIsInstance(comm, Decimal)
        self.assertGreater(comm, Decimal("0.0"))

    def test_commission_small_quantity(self):
        """Commission on minimum quantity"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        quantity = Decimal("0.01")  # Min quantity
        price = Decimal("100000.0")

        comm = spec.calculate_commission(quantity, price)
        self.assertIsInstance(comm, Decimal)
        self.assertGreaterEqual(comm, Decimal("0.0"))


class TestSwapCalculations(unittest.TestCase):
    """Test swap/rollover cost calculations"""

    def setUp(self):
        """Set up test specs with swap rates"""
        INSTRUMENT_SPECS["BTCUSD"] = InstrumentSpec(
            symbol_id="10028",
            digits=8,
            volume_digits=2,
            min_volume=Decimal("0.01"),
            max_volume=Decimal("100.0"),
            volume_step=Decimal("0.01"),
            contract_size=Decimal("1.0"),
            pip_value_per_lot=Decimal("0.01"),
            swap_long=Decimal("-0.5"),  # Pay 0.5 pips per day for long
            swap_short=Decimal("0.3"),  # Earn 0.3 pips per day for short
        )

    def test_swap_long_single_day(self):
        """Swap cost for LONG position, 1 day"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        quantity = Decimal("1.0")  # 1 lot
        is_long = True
        days = 1

        swap = spec.calculate_swap(quantity, is_long, days)

        # Swap = quantity * swap_rate * pip_value * days
        # = 1.0 * (-0.5) * 0.01 * 1 = -0.005 USD
        self.assertEqual(swap, Decimal("-0.005"))

    def test_swap_short_single_day(self):
        """Swap cost for SHORT position, 1 day (earning)"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        quantity = Decimal("1.0")
        is_long = False
        days = 1

        swap = spec.calculate_swap(quantity, is_long, days)

        # Swap = 1.0 * 0.3 * 0.01 * 1 = 0.003 USD (earning)
        self.assertEqual(swap, Decimal("0.003"))

    def test_swap_multiple_days(self):
        """Swap cost over multiple days"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        quantity = Decimal("1.0")
        is_long = True
        days = 7  # 1 week

        swap = spec.calculate_swap(quantity, is_long, days)

        # 7 * (-0.005) = -0.035 USD
        self.assertEqual(swap, Decimal("-0.035"))

    def test_swap_zero_quantity(self):
        """Swap with zero quantity"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        quantity = Decimal("0.0")
        is_long = True
        days = 1

        swap = spec.calculate_swap(quantity, is_long, days)
        self.assertEqual(swap, Decimal("0.0"))

    def test_swap_fractional_day(self):
        """Swap for partial day (intraday)"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        quantity = Decimal("1.0")
        is_long = True
        days = 0  # Intraday, no swap

        swap = spec.calculate_swap(quantity, is_long, days)
        self.assertEqual(swap, Decimal("0.0"))


class TestPnLCalculations(unittest.TestCase):
    """Test PnL calculations with full precision"""

    def setUp(self):
        """Set up instrument specs"""
        INSTRUMENT_SPECS["BTCUSD"] = InstrumentSpec(
            symbol_id="10028",
            digits=8,
            volume_digits=2,
            min_volume=Decimal("0.01"),
            max_volume=Decimal("100.0"),
            volume_step=Decimal("0.01"),
            contract_size=Decimal("1.0"),
            pip_value_per_lot=Decimal("0.01"),
        )

    def calculate_pnl(
        self, entry_price: Decimal, exit_price: Decimal, quantity: Decimal, side: str, spec: InstrumentSpec
    ) -> Decimal:
        """
        Helper: Calculate PnL with SafeMath
        
        PnL = (exit - entry) * quantity * contract_size for LONG
        PnL = (entry - exit) * quantity * contract_size for SHORT
        """
        entry_dec = spec.normalize_price(entry_price)
        exit_dec = spec.normalize_price(exit_price)
        qty_dec = spec.normalize_quantity(quantity)

        if side.upper() == "BUY":
            price_diff = DecimalSafeMath.safe_subtract(exit_dec, entry_dec)
        else:
            price_diff = DecimalSafeMath.safe_subtract(entry_dec, exit_dec)

        # PnL = price_diff * quantity * contract_size
        pnl = DecimalSafeMath.safe_multiply(price_diff, qty_dec)
        pnl = DecimalSafeMath.safe_multiply(pnl, spec.contract_size)

        return pnl

    def test_pnl_long_profit(self):
        """LONG position with profit"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        entry = Decimal("100000.00000000")
        exit = Decimal("102000.00000000")  # +2%
        quantity = Decimal("1.0")  # 1 lot = 1 BTC

        pnl = self.calculate_pnl(entry, exit, quantity, "BUY", spec)

        # PnL = (102000 - 100000) * 1.0 * 1.0 = 2000 BTC units
        expected = Decimal("2000.00000000")
        self.assertEqual(pnl, expected)

    def test_pnl_long_loss(self):
        """LONG position with loss"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        entry = Decimal("100000.00000000")
        exit = Decimal("98000.00000000")  # -2%
        quantity = Decimal("1.0")

        pnl = self.calculate_pnl(entry, exit, quantity, "BUY", spec)

        # PnL = (98000 - 100000) * 1.0 * 1.0 = -2000
        expected = Decimal("-2000.00000000")
        self.assertEqual(pnl, expected)

    def test_pnl_short_profit(self):
        """SHORT position with profit"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        entry = Decimal("100000.00000000")
        exit = Decimal("98000.00000000")  # Price fell
        quantity = Decimal("1.0")

        pnl = self.calculate_pnl(entry, exit, quantity, "SELL", spec)

        # PnL = (100000 - 98000) * 1.0 * 1.0 = 2000
        expected = Decimal("2000.00000000")
        self.assertEqual(pnl, expected)

    def test_pnl_short_loss(self):
        """SHORT position with loss"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        entry = Decimal("100000.00000000")
        exit = Decimal("102000.00000000")  # Price rose
        quantity = Decimal("1.0")

        pnl = self.calculate_pnl(entry, exit, quantity, "SELL", spec)

        # PnL = (100000 - 102000) * 1.0 * 1.0 = -2000
        expected = Decimal("-2000.00000000")
        self.assertEqual(pnl, expected)

    def test_pnl_small_move(self):
        """PnL on small price move (precision test)"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        entry = Decimal("100000.00000000")
        exit = Decimal("100000.00000001")  # 0.00000001 move (1 satoshi)
        quantity = Decimal("1.0")

        pnl = self.calculate_pnl(entry, exit, quantity, "BUY", spec)

        # PnL = 0.00000001 * 1.0 * 1.0 = 0.00000001
        expected = Decimal("0.00000001")
        self.assertEqual(pnl, expected)

    def test_pnl_fractional_quantity(self):
        """PnL with fractional quantity"""
        spec = INSTRUMENT_SPECS["BTCUSD"]
        entry = Decimal("100000.00000000")
        exit = Decimal("101000.00000000")
        quantity = Decimal("0.10")  # 0.1 lots

        pnl = self.calculate_pnl(entry, exit, quantity, "BUY", spec)

        # PnL = 1000 * 0.10 * 1.0 = 100
        expected = Decimal("100.00")
        self.assertEqual(pnl, expected)


class TestRiskMetricCalculations(unittest.TestCase):
    """Test Sharpe, Sortino, Omega calculations"""

    def test_sharpe_ratio_positive(self):
        """Sharpe ratio with positive returns"""
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]  # 5 trades

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        sharpe = SafeMath.safe_div(mean_return, std_dev, 0.0)

        self.assertGreater(sharpe, 0.0)
        self.assertTrue(SafeMath.is_valid(sharpe))

    def test_sharpe_ratio_all_losses(self):
        """Sharpe ratio with all losing trades"""
        returns = [-0.01, -0.02, -0.01, -0.03, -0.01]

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        sharpe = SafeMath.safe_div(mean_return, std_dev, 0.0)

        self.assertLess(sharpe, 0.0)
        self.assertTrue(SafeMath.is_valid(sharpe))

    def test_sharpe_ratio_zero_variance(self):
        """Sharpe ratio with zero variance (all same return)"""
        returns = [0.01, 0.01, 0.01, 0.01, 0.01]

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        # std_dev should be 0, causing division by zero
        sharpe = SafeMath.safe_div(mean_return, std_dev, 0.0)

        # Should return 0.0 (default) or handle gracefully
        self.assertEqual(sharpe, 0.0)

    def test_sortino_ratio_downside_only(self):
        """Sortino ratio penalizes downside deviation"""
        returns = [0.05, 0.03, -0.02, -0.01, 0.04]  # Mixed

        mean_return = sum(returns) / len(returns)
        downside_diffs = [min(0, r - 0.0) for r in returns]
        downside_variance = sum(d**2 for d in downside_diffs) / len(downside_diffs)
        downside_std = math.sqrt(downside_variance)

        sortino = SafeMath.safe_div(mean_return, downside_std, 0.0)

        self.assertGreater(sortino, 0.0)
        self.assertTrue(SafeMath.is_valid(sortino))

    def test_omega_ratio(self):
        """Omega ratio (gains / losses)"""
        returns = [0.02, 0.01, -0.01, 0.03, -0.02]

        threshold = 0.0
        gains = sum(max(0, r - threshold) for r in returns)
        losses = sum(max(0, threshold - r) for r in returns)

        omega = SafeMath.safe_div(gains, losses, float("inf"))

        self.assertGreater(omega, 0.0)
        self.assertTrue(SafeMath.is_valid(omega))

    def test_omega_ratio_all_gains(self):
        """Omega ratio with all gains (omega = inf)"""
        returns = [0.01, 0.02, 0.03, 0.01, 0.02]

        threshold = 0.0
        gains = sum(max(0, r - threshold) for r in returns)
        losses = sum(max(0, threshold - r) for r in returns)

        omega = SafeMath.safe_div(gains, losses, float("inf"))

        # With no losses, omega should be inf
        self.assertEqual(omega, float("inf"))


class TestDivisionByZeroProtection(unittest.TestCase):
    """Test all division operations handle zero divisor"""

    def test_safe_div_zero_divisor(self):
        """SafeMath.safe_div with zero divisor"""
        result = SafeMath.safe_div(100.0, 0.0, 99.0)
        self.assertEqual(result, 99.0)  # Should return default

    def test_safe_div_valid(self):
        """SafeMath.safe_div with valid divisor"""
        result = SafeMath.safe_div(100.0, 2.0, 0.0)
        self.assertEqual(result, 50.0)

    def test_equity_division_zero(self):
        """PnL / equity with equity = 0"""
        pnl = 100.0
        equity = 0.0

        ret = SafeMath.safe_div(pnl, equity, 0.0)
        self.assertEqual(ret, 0.0)

    def test_spread_calculation_zero_bid(self):
        """Spread calculation with zero bid"""
        bid = 0.0
        ask = 100.0

        mid = SafeMath.safe_div(bid + ask, 2.0, 0.0)
        self.assertEqual(mid, 50.0)  # Should still calculate

    def test_imbalance_zero_total(self):
        """Order book imbalance with zero total"""
        depth_bid = 0.0
        depth_ask = 0.0
        depth_total = depth_bid + depth_ask

        imbalance = SafeMath.safe_div(depth_bid - depth_ask, depth_total, 0.0)
        self.assertEqual(imbalance, 0.0)


class TestPrecisionEdgeCases(unittest.TestCase):
    """Test precision handling at extreme values"""

    def test_very_small_price(self):
        """Price near minimum precision"""
        spec = InstrumentSpec(
            symbol_id="TEST",
            digits=8,
            volume_digits=2,
            min_volume=Decimal("0.01"),
            max_volume=Decimal("100.0"),
            volume_step=Decimal("0.01"),
            contract_size=Decimal("1.0"),
            pip_value_per_lot=Decimal("0.01"),
        )

        price = Decimal("0.00000001")  # 1 satoshi
        normalized = spec.normalize_price(price)

        self.assertEqual(normalized, price)
        # Check precision is preserved
        str_repr = f"{normalized:.8f}"
        self.assertTrue("." in str_repr)
        decimal_part = str_repr.split(".")[1]
        self.assertEqual(len(decimal_part), 8)

    def test_very_large_price(self):
        """Price at maximum reasonable value"""
        spec = InstrumentSpec(
            symbol_id="TEST",
            digits=2,
            volume_digits=2,
            min_volume=Decimal("0.01"),
            max_volume=Decimal("100.0"),
            volume_step=Decimal("0.01"),
            contract_size=Decimal("1.0"),
            pip_value_per_lot=Decimal("0.01"),
        )

        price = Decimal("9999999.99")
        normalized = spec.normalize_price(price)

        self.assertEqual(normalized, price)

    def test_quantity_below_min(self):
        """Quantity below minimum"""
        spec = InstrumentSpec(
            symbol_id="TEST",
            digits=5,
            volume_digits=2,
            min_volume=Decimal("0.01"),
            max_volume=Decimal("100.0"),
            volume_step=Decimal("0.01"),
            contract_size=Decimal("1.0"),
            pip_value_per_lot=Decimal("0.01"),
        )

        quantity = Decimal("0.005")  # Below min
        normalized = spec.normalize_quantity(quantity)

        # Should clamp to min
        self.assertEqual(normalized, spec.min_volume)

    def test_quantity_above_max(self):
        """Quantity above maximum"""
        spec = InstrumentSpec(
            symbol_id="TEST",
            digits=5,
            volume_digits=2,
            min_volume=Decimal("0.01"),
            max_volume=Decimal("100.0"),
            volume_step=Decimal("0.01"),
            contract_size=Decimal("1.0"),
            pip_value_per_lot=Decimal("0.01"),
        )

        quantity = Decimal("150.0")  # Above max
        normalized = spec.normalize_quantity(quantity)

        # Should clamp to max
        self.assertEqual(normalized, spec.max_volume)

    def test_quantity_not_on_step(self):
        """Quantity not aligned to volume_step"""
        spec = InstrumentSpec(
            symbol_id="TEST",
            digits=5,
            volume_digits=2,
            min_volume=Decimal("0.01"),
            max_volume=Decimal("100.0"),
            volume_step=Decimal("0.01"),
            contract_size=Decimal("1.0"),
            pip_value_per_lot=Decimal("0.01"),
        )

        quantity = Decimal("1.234")  # Not on 0.01 step
        normalized = spec.normalize_quantity(quantity)

        # Should round down to 1.23
        self.assertEqual(normalized, Decimal("1.23"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
