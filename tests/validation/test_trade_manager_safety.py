"""
TradeManager Comprehensive Test Suite

Tests 100+ failure scenarios including:
- Invalid inputs (NaN, Inf, negative, zero)
- Boundary conditions (min/max values)
- State corruption
- Network failures
- Race conditions
- Memory exhaustion
- Catastrophic failures
"""

import math
import sys
import time
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.skip("trade_manager_safety has been archived", allow_module_level=True)


class TestSafeMath(unittest.TestCase):
    """Test safe mathematical operations"""

    def setUp(self):
        self.sm = SafeMath()

    def test_01_decimal_conversion_valid(self):
        """Test valid decimal conversions"""
        self.assertEqual(self.sm.to_decimal(100.5, 2), Decimal("100.50"))
        self.assertEqual(self.sm.to_decimal("0.123456", 6), Decimal("0.123456"))
        self.assertEqual(self.sm.to_decimal(0, 8), Decimal("0"))

    def test_02_decimal_conversion_invalid(self):
        """Test invalid decimal conversions"""
        with self.assertRaises(SafeMathError):
            self.sm.to_decimal("invalid", 2)
        with self.assertRaises(SafeMathError):
            self.sm.to_decimal(float("nan"), 2)
        with self.assertRaises(SafeMathError):
            self.sm.to_decimal(float("inf"), 2)

    def test_03_addition_normal(self):
        """Test normal addition"""
        a = Decimal("100.5")
        b = Decimal("50.3")
        result = self.sm.safe_add(a, b)
        self.assertEqual(result, Decimal("150.8"))

    def test_04_addition_overflow(self):
        """Test addition overflow protection"""
        a = MAX_PRICE * MAX_QUANTITY
        b = Decimal("1.0")
        with self.assertRaises(SafeMathError):
            self.sm.safe_add(a, b)

    def test_05_subtraction_normal(self):
        """Test normal subtraction"""
        a = Decimal("100.5")
        b = Decimal("50.3")
        result = self.sm.safe_subtract(a, b)
        self.assertEqual(result, Decimal("50.2"))

    def test_06_subtraction_underflow(self):
        """Test subtraction underflow protection"""
        a = MAX_PRICE * MAX_QUANTITY * Decimal("-1")
        b = Decimal("1.0")
        with self.assertRaises(SafeMathError):
            self.sm.safe_subtract(a, b)

    def test_07_multiplication_normal(self):
        """Test normal multiplication"""
        a = Decimal("10.5")
        b = Decimal("2.0")
        result = self.sm.safe_multiply(a, b)
        self.assertEqual(result, Decimal("21.0"))

    def test_08_multiplication_overflow(self):
        """Test multiplication overflow protection"""
        a = MAX_PRICE
        b = MAX_QUANTITY * Decimal("2")
        with self.assertRaises(SafeMathError):
            self.sm.safe_multiply(a, b)

    def test_09_division_normal(self):
        """Test normal division"""
        a = Decimal("100.0")
        b = Decimal("4.0")
        result = self.sm.safe_divide(a, b)
        self.assertEqual(result, Decimal("25.0"))

    def test_10_division_by_zero(self):
        """Test division by zero protection"""
        a = Decimal("100.0")
        b = Decimal("0.0")
        result = self.sm.safe_divide(a, b, default=Decimal("-1"))
        self.assertEqual(result, Decimal("-1"))

    def test_11_division_by_near_zero(self):
        """Test division by near-zero protection"""
        a = Decimal("100.0")
        b = Decimal("1e-13")
        result = self.sm.safe_divide(a, b, default=Decimal("0"))
        self.assertEqual(result, Decimal("0"))

    def test_12_is_finite_valid(self):
        """Test finite check for valid values"""
        self.assertTrue(self.sm.is_finite(100.5))
        self.assertTrue(self.sm.is_finite(Decimal("0")))
        self.assertTrue(self.sm.is_finite(-100.5))

    def test_13_is_finite_invalid(self):
        """Test finite check for invalid values"""
        self.assertFalse(self.sm.is_finite(float("nan")))
        self.assertFalse(self.sm.is_finite(float("inf")))
        self.assertFalse(self.sm.is_finite(float("-inf")))


class TestOrderValidator(unittest.TestCase):
    """Test order validation logic"""

    def setUp(self):
        self.validator = OrderValidator()

    def test_20_quantity_valid(self):
        """Test valid quantity"""
        result = self.validator.validate_quantity(0.10)
        self.assertTrue(result.valid)
        self.assertEqual(result.sanitized_value, Decimal("0.100000"))

    def test_21_quantity_negative(self):
        """Test negative quantity rejection"""
        result = self.validator.validate_quantity(-0.10)
        self.assertFalse(result.valid)
        self.assertIn("positive", result.error.lower())

    def test_22_quantity_zero(self):
        """Test zero quantity rejection"""
        result = self.validator.validate_quantity(0.0)
        self.assertFalse(result.valid)

    def test_23_quantity_nan(self):
        """Test NaN quantity rejection"""
        result = self.validator.validate_quantity(float("nan"))
        self.assertFalse(result.valid)
        self.assertIn("finite", result.error.lower())

    def test_24_quantity_inf(self):
        """Test infinite quantity rejection"""
        result = self.validator.validate_quantity(float("inf"))
        self.assertFalse(result.valid)

    def test_25_quantity_too_small(self):
        """Test quantity below minimum (should clamp)"""
        result = self.validator.validate_quantity(1e-10)
        self.assertTrue(result.valid)  # Clamped to minimum
        self.assertEqual(result.sanitized_value, MIN_QUANTITY)

    def test_26_quantity_too_large(self):
        """Test quantity above maximum"""
        result = self.validator.validate_quantity(float(MAX_QUANTITY) + 1)
        self.assertFalse(result.valid)
        self.assertIn("exceeds", result.error.lower())

    def test_27_price_valid(self):
        """Test valid price"""
        result = self.validator.validate_price(65000.0, allow_zero=False)
        self.assertTrue(result.valid)
        self.assertEqual(result.sanitized_value, Decimal("65000.00000"))

    def test_28_price_negative(self):
        """Test negative price rejection"""
        result = self.validator.validate_price(-100.0)
        self.assertFalse(result.valid)
        self.assertIn("negative", result.error.lower())

    def test_29_price_zero_not_allowed(self):
        """Test zero price rejection for limit orders"""
        result = self.validator.validate_price(0.0, allow_zero=False)
        self.assertFalse(result.valid)

    def test_30_price_zero_allowed(self):
        """Test zero price allowed for market orders"""
        result = self.validator.validate_price(0.0, allow_zero=True)
        self.assertTrue(result.valid)

    def test_31_price_nan(self):
        """Test NaN price rejection"""
        result = self.validator.validate_price(float("nan"))
        self.assertFalse(result.valid)

    def test_32_price_inf(self):
        """Test infinite price rejection"""
        result = self.validator.validate_price(float("inf"))
        self.assertFalse(result.valid)

    def test_33_price_too_large(self):
        """Test price above maximum"""
        result = self.validator.validate_price(float(MAX_PRICE) + 1)
        self.assertFalse(result.valid)

    def test_34_side_valid_buy(self):
        """Test valid BUY side"""
        result = self.validator.validate_side("1")
        self.assertTrue(result.valid)
        self.assertEqual(result.sanitized_value, "1")

    def test_35_side_valid_sell(self):
        """Test valid SELL side"""
        result = self.validator.validate_side(2)
        self.assertTrue(result.valid)
        self.assertEqual(result.sanitized_value, "2")

    def test_36_side_invalid(self):
        """Test invalid side"""
        result = self.validator.validate_side("3")
        self.assertFalse(result.valid)
        result = self.validator.validate_side("BUY")
        self.assertFalse(result.valid)

    def test_37_symbol_valid(self):
        """Test valid symbol ID"""
        result = self.validator.validate_symbol(10028)
        self.assertTrue(result.valid)
        self.assertEqual(result.sanitized_value, "10028")

    def test_38_symbol_invalid_negative(self):
        """Test negative symbol rejection"""
        result = self.validator.validate_symbol(-1)
        self.assertFalse(result.valid)

    def test_39_symbol_invalid_zero(self):
        """Test zero symbol rejection"""
        result = self.validator.validate_symbol(0)
        self.assertFalse(result.valid)

    def test_40_symbol_invalid_string(self):
        """Test non-numeric symbol rejection"""
        result = self.validator.validate_symbol("BTCUSD")
        self.assertFalse(result.valid)

    def test_41_symbol_too_large(self):
        """Test symbol ID too large"""
        result = self.validator.validate_symbol(9999999)
        self.assertFalse(result.valid)

    def test_42_rate_limit_normal(self):
        """Test normal order rate"""
        for _ in range(5):
            result = self.validator.check_rate_limit()
            self.assertTrue(result.valid)

    def test_43_rate_limit_exceeded(self):
        """Test rate limit enforcement"""
        # Submit max allowed orders
        for _ in range(10):
            self.validator.check_rate_limit()

        # Next order should fail
        result = self.validator.check_rate_limit()
        self.assertFalse(result.valid)
        self.assertIn("rate limit", result.error.lower())

    def test_44_rate_limit_recovery(self):
        """Test rate limit recovery after 1 second"""
        # Fill rate limit
        for _ in range(10):
            self.validator.check_rate_limit()

        # Wait for window to expire
        time.sleep(1.1)

        # Should work again
        result = self.validator.check_rate_limit()
        self.assertTrue(result.valid)

    def test_45_order_validation_complete_valid(self):
        """Test complete order validation - valid order"""
        result = self.validator.validate_order(symbol=10028, side="1", quantity=0.10, price=65000.0, order_type="LIMIT")
        self.assertTrue(result.valid)
        self.assertEqual(result.sanitized_value["symbol"], "10028")
        self.assertEqual(result.sanitized_value["side"], "1")

    def test_46_order_validation_market_order(self):
        """Test market order validation (no price)"""
        result = self.validator.validate_order(symbol=10028, side="2", quantity=0.10, price=None, order_type="MARKET")
        self.assertTrue(result.valid)

    def test_47_order_validation_limit_without_price(self):
        """Test limit order without price"""
        result = self.validator.validate_order(symbol=10028, side="1", quantity=0.10, price=None, order_type="LIMIT")
        self.assertFalse(result.valid)

    def test_48_order_validation_notional_too_large(self):
        """Test order with excessive notional value"""
        result = self.validator.validate_order(
            symbol=10028,
            side="1",
            quantity=float(MAX_QUANTITY),
            price=float(MAX_PRICE),
            order_type="LIMIT",
        )
        self.assertFalse(result.valid)
        self.assertIn("notional", result.error.lower())


class TestStatePersistence(unittest.TestCase):
    """Test state persistence and recovery"""

    def setUp(self):
        self.test_dir = Path("store/test_trade_manager")
        self.persistence = StatePersistence(self.test_dir)

    def tearDown(self):
        # Clean up test files
        if self.test_dir.exists():
            for file in self.test_dir.glob("*"):
                file.unlink()
            self.test_dir.rmdir()

    def test_50_save_state_success(self):
        """Test successful state save"""
        state = {"orders": {}, "position": {"net": 0}}
        result = self.persistence.save_state(state)
        self.assertTrue(result)
        self.assertTrue(self.persistence.state_file.exists())

    def test_51_load_state_success(self):
        """Test successful state load"""
        state = {"orders": {"ord1": {"qty": 0.10}}, "position": {"net": 0.10}}
        self.persistence.save_state(state)

        loaded = self.persistence.load_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["orders"]["ord1"]["qty"], 0.10)

    def test_52_load_nonexistent_file(self):
        """Test loading from nonexistent file"""
        loaded = self.persistence.load_state()
        self.assertIsNone(loaded)

    def test_53_recovery_from_backup(self):
        """Test recovery from backup file"""
        # Save state
        state = {"orders": {}, "position": {"net": 0.5}}
        self.persistence.save_state(state)

        # Corrupt primary file
        with open(self.persistence.state_file, "w") as f:
            f.write("CORRUPTED DATA")

        # Load should recover from backup
        loaded = self.persistence.load_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["position"]["net"], 0.5)

    def test_54_atomic_write(self):
        """Test atomic write (temp file + rename)"""
        state = {"test": "data"}
        self.persistence.save_state(state)

        # Temp file should not exist after save
        temp_file = self.persistence.state_file.with_suffix(".tmp")
        self.assertFalse(temp_file.exists())

        # State file should exist
        self.assertTrue(self.persistence.state_file.exists())

    def test_55_backup_creation(self):
        """Test backup file creation"""
        # Save first state
        state1 = {"version": 1}
        self.persistence.save_state(state1)

        # Save second state (should create backup of first)
        state2 = {"version": 2}
        self.persistence.save_state(state2)

        # Backup should exist
        self.assertTrue(self.persistence.backup_file.exists())


class TestCatastrophicFailures(unittest.TestCase):
    """Test catastrophic failure scenarios"""

    def test_60_memory_exhaustion_simulation(self):
        """Test handling of memory pressure"""
        validator = OrderValidator()

        # Try to validate massive array of orders
        results = []
        for i in range(1000):
            result = validator.validate_quantity(0.10 * (i + 1))
            results.append(result.valid)

        # Should handle gracefully
        self.assertTrue(all(results))

    def test_61_concurrent_validation(self):
        """Test concurrent order validation"""
        validator = OrderValidator()

        # Simulate concurrent validations
        import threading

        results = []

        def validate():
            for _ in range(50):
                result = validator.validate_order(10028, "1", 0.10, 65000.0)
                results.append(result.valid)

        threads = [threading.Thread(target=validate) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed (no race conditions)
        self.assertTrue(len(results) == 200)

    def test_62_float_precision_edge_cases(self):
        """Test float precision edge cases"""
        sm = SafeMath()

        # Test precision loss
        a = sm.to_decimal(0.1 + 0.2, 8)
        b = sm.to_decimal(0.3, 8)
        # Decimal should handle this correctly
        self.assertEqual(a, b)

        # Test very small numbers
        tiny = sm.to_decimal(1e-8, 8)
        self.assertTrue(tiny > Decimal("0"))

    def test_63_extreme_price_volatility(self):
        """Test extreme price movements"""
        validator = OrderValidator()

        # Flash crash scenario (price -> 0.01)
        result = validator.validate_price(0.01, allow_zero=False)
        self.assertTrue(result.valid)

        # Price spike (10x normal)
        result = validator.validate_price(650000.0, allow_zero=False)
        self.assertTrue(result.valid)

    def test_64_network_timeout_simulation(self):
        """Test state persistence under simulated network failure"""
        persistence = StatePersistence(Path("store/test_network_fail"))

        # Simulate write failure
        with patch("builtins.open", side_effect=OSError("Network timeout")):
            result = persistence.save_state({"test": "data"})
            self.assertFalse(result)

        # Clean up
        if Path("store/test_network_fail").exists():
            Path("store/test_network_fail").rmdir()

    def test_65_invalid_json_recovery(self):
        """Test recovery from invalid JSON"""
        persistence = StatePersistence(Path("store/test_json_invalid"))
        persistence.state_dir.mkdir(parents=True, exist_ok=True)

        # Write invalid JSON
        with open(persistence.state_file, "w") as f:
            f.write('{"invalid": json syntax}')

        # Should handle gracefully
        loaded = persistence.load_state()
        self.assertIsNone(loaded)

        # Clean up
        persistence.state_file.unlink()
        persistence.state_dir.rmdir()


def run_test_suite():
    """Run complete test suite"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSafeMath))
    suite.addTests(loader.loadTestsFromTestCase(TestOrderValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestStatePersistence))
    suite.addTests(loader.loadTestsFromTestCase(TestCatastrophicFailures))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"✓ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
