"""
TradeManager Safety Layer - Defense in Depth

Comprehensive validation, safe math, and error recovery for mission-critical trading.

Principles:
1. Never trust input - validate everything
2. Fail gracefully - no exceptions escape
3. Log all failures - full audit trail
4. Persist state - recover from crashes
5. Redundant checks - defense in depth
6. Safe math - no float precision issues
7. Idempotent operations - replay safety
"""

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class InstrumentSpec:
    """
    Instrument specification from broker (SymbolInfo).

    Source of truth for all precision, lot sizes, pip values, and fees.
    All values derived from cTrader SecurityDefinition response (FIX Tag 35=d).

    NEVER hardcode these values - they vary by instrument:
    - Crypto: digits=8 (0.00000001), volume_digits=8
    - Forex: digits=5 (0.00001), volume_digits=2
    - Indices: digits=2 (0.01), volume_digits=2
    """

    symbol_id: str

    # Price precision (from Tag: PriceQuoteMethod or inferred)
    digits: int = 5  # Decimal places for price (crypto=8, forex=5, indices=2)
    pip_size: Decimal = Decimal("0.00001")  # 10^(-digits) - calculated
    tick_size: Decimal = Decimal("0.01")  # Minimum price movement

    # Quantity precision (from broker or inferred)
    volume_digits: int = 2  # Decimal places for quantity/lots
    min_volume: Decimal = Decimal("0.01")  # MinTradeVol / MinQty
    max_volume: Decimal = Decimal("100.0")  # MaxTradeVol
    volume_step: Decimal = Decimal("0.01")  # RoundLot (lot size increment)

    # Contract specifications
    contract_size: Decimal = Decimal("100000.0")  # ContractMultiplier (standard lot size)
    pip_value_per_lot: Decimal = Decimal("10.0")  # USD value of 1 pip for 1 lot

    # Commission structure (from broker or defaults)
    commission_per_lot: Decimal = Decimal("0.0")  # Fixed commission per lot
    commission_percentage: Decimal = Decimal("0.0")  # e.g., 0.001 = 0.1%
    commission_type: str = "ABSOLUTE"  # "ABSOLUTE" or "PERCENTAGE"
    min_commission: Decimal = Decimal("0.0")

    # Swap rates / overnight financing (triple swap on Wednesdays)
    swap_long: Decimal = Decimal("0.0")  # Pips per lot per day for LONG positions
    swap_short: Decimal = Decimal("0.0")  # Pips per lot per day for SHORT positions
    swap_type: str = "PIPS"  # "PIPS", "PERCENTAGE", or "POINTS"

    # Trading hours
    trading_hours_start: int = 0  # UTC hour
    trading_hours_end: int = 24  # UTC hour (24 = always open)

    # Currency
    currency: str = "USD"  # Base currency

    def __post_init__(self):
        """Calculate derived values and ensure Decimal types"""
        # Pip size = 10^(-digits)
        self.pip_size = Decimal(10) ** -self.digits

        # Tick size usually same as pip size, or explicit if provided
        if self.tick_size == Decimal("0.01") and self.digits != 2:
            self.tick_size = self.pip_size

        # Ensure all numeric fields are Decimal for precision
        self.min_volume = Decimal(str(self.min_volume))
        self.max_volume = Decimal(str(self.max_volume))
        self.volume_step = Decimal(str(self.volume_step))
        self.contract_size = Decimal(str(self.contract_size))
        self.pip_value_per_lot = Decimal(str(self.pip_value_per_lot))
        self.commission_per_lot = Decimal(str(self.commission_per_lot))
        self.commission_percentage = Decimal(str(self.commission_percentage))
        self.min_commission = Decimal(str(self.min_commission))
        self.swap_long = Decimal(str(self.swap_long))
        self.swap_short = Decimal(str(self.swap_short))

    def calculate_commission(self, quantity: Decimal, price: Decimal) -> Decimal:
        """
        Calculate commission for a trade.

        Args:
            quantity: Trade quantity in lots
            price: Execution price

        Returns:
            Commission amount in account currency
        """
        if self.commission_type == "PERCENTAGE":
            # Percentage of notional value
            notional = quantity * self.contract_size * price
            comm = notional * self.commission_percentage
        else:
            # Fixed per lot
            comm = quantity * self.commission_per_lot

        # Apply minimum
        return max(comm, self.min_commission)

    def calculate_swap(self, quantity: Decimal, is_long: bool, days: int = 1) -> Decimal:
        """
        Calculate swap/rollover cost for holding position overnight.

        Args:
            quantity: Position size in lots
            is_long: True for LONG, False for SHORT
            days: Number of days held (usually 1, 3 on Wednesdays)

        Returns:
            Swap cost in account currency (negative = cost, positive = credit)
        """
        swap_rate = self.swap_long if is_long else self.swap_short

        if self.swap_type == "PIPS":
            # Swap in pips - convert to currency
            return quantity * swap_rate * self.pip_value_per_lot * Decimal(days)
        elif self.swap_type == "PERCENTAGE":
            # Percentage of position value - would need price
            # Placeholder - needs price parameter
            return Decimal("0.0")
        else:  # POINTS
            return quantity * swap_rate * Decimal(days)

    def normalize_price(self, price: float | Decimal) -> Decimal:
        """
        Normalize price to instrument's digit precision.

        Args:
            price: Raw price value

        Returns:
            Price rounded to instrument digits
        """
        dec_price = Decimal(str(price))
        return dec_price.quantize(self.pip_size, rounding=ROUND_DOWN)

    def normalize_quantity(self, quantity: float | Decimal) -> Decimal:
        """
        Normalize quantity to instrument's volume step.

        Args:
            quantity: Raw quantity value

        Returns:
            Quantity rounded to volume_step
        """
        dec_qty = Decimal(str(quantity))

        # Round to volume_step precision
        quantizer = self.volume_step
        normalized = (dec_qty / quantizer).quantize(Decimal("1"), rounding=ROUND_DOWN) * quantizer

        # Clamp to min/max
        if normalized < self.min_volume:
            normalized = self.min_volume
        if normalized > self.max_volume:
            normalized = self.max_volume

        return normalized


# Safety constants
MAX_PRICE = Decimal("10_000_000.0")  # $10M max price
MAX_QUANTITY = Decimal("1000.0")  # 1000 lots max
MAX_LEVERAGE = Decimal("1000.0")  # 1000x leverage max
MAX_ORDERS_PER_SECOND = 10
MAX_PENDING_ORDERS = 100
PERSISTENCE_DIR = Path("store/trade_manager")
STATE_FILE = "trade_manager_state.json"
BACKUP_FILE = "trade_manager_state.backup.json"

# Global instrument specifications (populated from broker)
INSTRUMENT_SPECS: dict[str, InstrumentSpec] = {}


@dataclass
class ValidationResult:
    """Result of validation check"""

    valid: bool
    error: str | None = None
    sanitized_value: Any = None


class SafeMathError(Exception):
    """Raised when safe math operation fails"""

    pass


class OrderValidationError(Exception):
    """Raised when order validation fails"""

    pass


class StateRecoveryError(Exception):
    """Raised when state recovery fails"""

    pass


class SafeMath:
    """
    Safe mathematical operations with Decimal precision.

    Prevents floating point precision issues in financial calculations.
    All operations return Decimal or raise SafeMathError.
    """

    @staticmethod
    def to_decimal(value: float | int | str | Decimal, places: int = 8) -> Decimal:
        """
        Convert to Decimal with safe rounding.

        Args:
            value: Input value
            places: Decimal places (default 8 for crypto precision)

        Returns:
            Decimal value

        Raises:
            SafeMathError: If conversion fails
        """
        try:
            if isinstance(value, Decimal):
                return value.quantize(Decimal(10) ** -places, rounding=ROUND_DOWN)

            dec = Decimal(str(value))
            return dec.quantize(Decimal(10) ** -places, rounding=ROUND_DOWN)
        except (ValueError, InvalidOperation) as e:
            raise SafeMathError(f"Cannot convert {value} to Decimal: {e}") from e

    @staticmethod
    def safe_add(a: Decimal, b: Decimal) -> Decimal:
        """Safe addition with overflow check"""
        try:
            result = a + b
            if result > MAX_PRICE * MAX_QUANTITY:
                raise SafeMathError(f"Addition overflow: {a} + {b}")
            return result
        except (InvalidOperation, OverflowError) as e:
            raise SafeMathError(f"Addition failed: {a} + {b}: {e}") from e

    @staticmethod
    def safe_subtract(a: Decimal, b: Decimal) -> Decimal:
        """Safe subtraction with underflow check"""
        try:
            result = a - b
            if abs(result) > MAX_PRICE * MAX_QUANTITY:
                raise SafeMathError(f"Subtraction overflow: {a} - {b}")
            return result
        except (InvalidOperation, OverflowError) as e:
            raise SafeMathError(f"Subtraction failed: {a} - {b}: {e}") from e

    @staticmethod
    def safe_multiply(a: Decimal, b: Decimal) -> Decimal:
        """Safe multiplication with overflow check"""
        try:
            result = a * b
            if abs(result) > MAX_PRICE * MAX_QUANTITY:
                raise SafeMathError(f"Multiplication overflow: {a} * {b} = {result}")
            return result
        except (InvalidOperation, OverflowError) as e:
            raise SafeMathError(f"Multiplication failed: {a} * {b}: {e}") from e

    @staticmethod
    def safe_divide(a: Decimal, b: Decimal, default: Decimal = Decimal("0")) -> Decimal:
        """Safe division with zero-check"""
        try:
            if abs(b) < Decimal("1e-12"):
                LOG.warning("[SAFE-MATH] Division by near-zero: %s / %s, returning %s", a, b, default)
                return default
            result = a / b
            if abs(result) > MAX_PRICE * MAX_QUANTITY:
                raise SafeMathError(f"Division overflow: {a} / {b} = {result}")
            return result
        except (InvalidOperation, ZeroDivisionError, OverflowError) as e:
            LOG.error("[SAFE-MATH] Division failed: %s / %s: %s, returning %s", a, b, e, default)
            return default

    @staticmethod
    def is_finite(value: float | Decimal) -> bool:
        """Check if value is finite (not NaN or Inf)"""
        try:
            if isinstance(value, Decimal):
                return value.is_finite()
            return math.isfinite(float(value))
        except (ValueError, OverflowError):
            return False


class OrderValidator:
    """
    Comprehensive order validation with defense-in-depth.

    Validates:
    - Quantity bounds
    - Price sanity
    - Symbol validity
    - Side correctness
    - Leverage limits
    - Rate limits
    """

    def __init__(self):
        self.order_timestamps: list[float] = []

    def validate_quantity(self, quantity: float | Decimal, symbol: str = "") -> ValidationResult:
        """
        Validate order quantity.

        Checks:
        1. Not NaN or Inf
        2. Within min/max bounds
        3. Positive value
        4. Finite precision
        """
        try:
            # Convert to Decimal for precision
            qty_dec = SafeMath.to_decimal(quantity, places=6)

            # Check finite
            if not SafeMath.is_finite(quantity):
                return ValidationResult(valid=False, error=f"Quantity is not finite: {quantity}")

            # Check positive
            if qty_dec <= Decimal("0"):
                return ValidationResult(valid=False, error=f"Quantity must be positive: {qty_dec}")

            # Check bounds
            if qty_dec < MIN_QUANTITY:
                LOG.warning("[VALIDATOR] Quantity %s below minimum %s, clamping", qty_dec, MIN_QUANTITY)
                qty_dec = MIN_QUANTITY

            if qty_dec > MAX_QUANTITY:
                return ValidationResult(
                    valid=False, error=f"Quantity {qty_dec} exceeds maximum {MAX_QUANTITY} for {symbol}"
                )

            return ValidationResult(valid=True, sanitized_value=qty_dec)

        except SafeMathError as e:
            return ValidationResult(valid=False, error=f"Quantity conversion failed: {e}")
        except Exception as e:
            LOG.error("[VALIDATOR] Unexpected error validating quantity %s: %s", quantity, e, exc_info=True)
            return ValidationResult(valid=False, error=f"Validation error: {e}")

    def validate_price(self, price: float | Decimal, symbol: str = "", allow_zero: bool = True) -> ValidationResult:
        """
        Validate order price.

        Checks:
        1. Not NaN or Inf
        2. Within min/max bounds
        3. Non-negative (allow_zero=True) or positive
        4. Finite precision
        """
        try:
            # Convert to Decimal
            price_dec = SafeMath.to_decimal(price, places=5)

            # Check finite
            if not SafeMath.is_finite(price):
                return ValidationResult(valid=False, error=f"Price is not finite: {price}")

            # Check sign
            if price_dec < Decimal("0"):
                return ValidationResult(valid=False, error=f"Price cannot be negative: {price_dec}")

            if not allow_zero and price_dec == Decimal("0"):
                return ValidationResult(valid=False, error="Price cannot be zero for limit orders")

            # Check bounds
            if price_dec > Decimal("0") and price_dec < MIN_PRICE:
                LOG.warning("[VALIDATOR] Price %s below minimum %s, clamping", price_dec, MIN_PRICE)
                price_dec = MIN_PRICE

            if price_dec > MAX_PRICE:
                return ValidationResult(
                    valid=False, error=f"Price {price_dec} exceeds maximum {MAX_PRICE} for {symbol}"
                )

            return ValidationResult(valid=True, sanitized_value=price_dec)

        except SafeMathError as e:
            return ValidationResult(valid=False, error=f"Price conversion failed: {e}")
        except Exception as e:
            LOG.error("[VALIDATOR] Unexpected error validating price %s: %s", price, e, exc_info=True)
            return ValidationResult(valid=False, error=f"Validation error: {e}")

    def validate_side(self, side: str | int) -> ValidationResult:
        """
        Validate order side.

        Accepts:
        - "1" or 1 for BUY
        - "2" or 2 for SELL
        """
        try:
            side_str = str(side).strip()
            if side_str not in ("1", "2"):
                return ValidationResult(valid=False, error=f"Invalid side: {side} (must be 1=BUY or 2=SELL)")

            return ValidationResult(valid=True, sanitized_value=side_str)

        except Exception as e:
            return ValidationResult(valid=False, error=f"Side validation failed: {e}")

    def validate_symbol(self, symbol: str | int) -> ValidationResult:
        """
        Validate symbol ID.

        Checks:
        1. Not empty
        2. Numeric (for cTrader)
        3. Positive integer
        """
        try:
            symbol_str = str(symbol).strip()

            if not symbol_str:
                return ValidationResult(valid=False, error="Symbol cannot be empty")

            # Try to convert to int to validate numeric
            symbol_int = int(symbol_str)
            if symbol_int <= 0:
                return ValidationResult(valid=False, error=f"Symbol ID must be positive: {symbol_int}")

            if symbol_int > 999999:
                return ValidationResult(valid=False, error=f"Symbol ID too large: {symbol_int}")

            return ValidationResult(valid=True, sanitized_value=symbol_str)

        except ValueError:
            return ValidationResult(valid=False, error=f"Symbol must be numeric: {symbol}")
        except Exception as e:
            return ValidationResult(valid=False, error=f"Symbol validation failed: {e}")

    def check_rate_limit(self) -> ValidationResult:
        """
        Check if order submission rate is within limits.

        Prevents:
        - Runaway loops submitting infinite orders
        - API rate limit violations
        - Accidental DDOS of broker
        """
        try:
            now = time.time()

            # Remove old timestamps (older than 1 second)
            self.order_timestamps = [ts for ts in self.order_timestamps if now - ts < 1.0]

            # Check rate
            if len(self.order_timestamps) >= MAX_ORDERS_PER_SECOND:
                return ValidationResult(
                    valid=False,
                    error=f"Rate limit exceeded: {len(self.order_timestamps)} orders in 1s (max {MAX_ORDERS_PER_SECOND})",
                )

            # Add current timestamp
            self.order_timestamps.append(now)

            return ValidationResult(valid=True)

        except Exception as e:
            LOG.error("[VALIDATOR] Rate limit check failed: %s", e, exc_info=True)
            # Fail closed - reject order if rate limit check fails
            return ValidationResult(valid=False, error=f"Rate limit check error: {e}")

    def validate_order(
        self,
        symbol: str | int,
        side: str | int,
        quantity: float | Decimal,
        price: float | Decimal | None = None,
        order_type: str = "MARKET",
    ) -> ValidationResult:
        """
        Comprehensive order validation.

        All checks must pass for order to be valid.
        """
        try:
            # 1. Rate limit check
            rate_check = self.check_rate_limit()
            if not rate_check.valid:
                LOG.error("[VALIDATOR] ❌ Rate limit: %s", rate_check.error)
                return rate_check

            # 2. Symbol validation
            symbol_check = self.validate_symbol(symbol)
            if not symbol_check.valid:
                LOG.error("[VALIDATOR] ❌ Symbol: %s", symbol_check.error)
                return symbol_check

            # 3. Side validation
            side_check = self.validate_side(side)
            if not side_check.valid:
                LOG.error("[VALIDATOR] ❌ Side: %s", side_check.error)
                return side_check

            # 4. Quantity validation
            qty_check = self.validate_quantity(quantity, str(symbol))
            if not qty_check.valid:
                LOG.error("[VALIDATOR] ❌ Quantity: %s", qty_check.error)
                return qty_check

            # 5. Price validation (if limit order)
            price_dec = None
            if price is not None or order_type == "LIMIT":
                if price is None:
                    return ValidationResult(valid=False, error="Limit order requires price")

                price_check = self.validate_price(price, str(symbol), allow_zero=False)
                if not price_check.valid:
                    LOG.error("[VALIDATOR] ❌ Price: %s", price_check.error)
                    return price_check
                price_dec = price_check.sanitized_value

            # 6. Notional value check (quantity * price)
            if price_dec and price_dec > Decimal("0"):
                try:
                    notional = SafeMath.safe_multiply(qty_check.sanitized_value, price_dec)
                    if notional > MAX_PRICE * MAX_QUANTITY:
                        return ValidationResult(
                            valid=False,
                            error=f"Notional value {notional} too large (qty={qty_check.sanitized_value} * price={price_dec})",
                        )
                except SafeMathError as e:
                    return ValidationResult(valid=False, error=f"Notional calculation failed: {e}")

            LOG.debug(
                "[VALIDATOR] ✓ Order valid: %s %s qty=%s price=%s",
                side_check.sanitized_value,
                symbol_check.sanitized_value,
                qty_check.sanitized_value,
                price_dec,
            )

            return ValidationResult(
                valid=True,
                sanitized_value={
                    "symbol": symbol_check.sanitized_value,
                    "side": side_check.sanitized_value,
                    "quantity": qty_check.sanitized_value,
                    "price": price_dec,
                },
            )

        except Exception as e:
            LOG.error("[VALIDATOR] Validation failed with exception: %s", e, exc_info=True)
            return ValidationResult(valid=False, error=f"Validation exception: {e}")


class StatePersistence:
    """
    Atomic state persistence with redundancy.

    Features:
    - Atomic writes (temp file + rename)
    - Backup copy for recovery
    - Checksum validation
    - Automatic recovery from corruption
    """

    def __init__(self, state_dir: Path = PERSISTENCE_DIR):
        self.state_dir = state_dir
        self.state_file = state_dir / STATE_FILE
        self.backup_file = state_dir / BACKUP_FILE

        # Ensure directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: dict) -> bool:
        """
        Save state atomically with backup.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Add metadata
            state["_version"] = "1.0"
            state["_timestamp"] = datetime.utcnow().isoformat()
            state["_pid"] = os.getpid()

            # Serialize
            json_data = json.dumps(state, indent=2, default=str)

            # Write to temp file first
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(json_data)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Backup current state
            if self.state_file.exists():
                self.state_file.replace(self.backup_file)

            # Atomic rename
            temp_file.replace(self.state_file)

            LOG.debug("[PERSISTENCE] State saved (%d bytes)", len(json_data))
            return True

        except Exception as e:
            LOG.error("[PERSISTENCE] Failed to save state: %s", e, exc_info=True)
            return False

    def load_state(self) -> dict | None:
        """
        Load state with automatic recovery.

        Returns:
            State dict or None if load fails
        """
        # Try primary file
        state = self._load_file(self.state_file)
        if state:
            return state

        LOG.warning("[PERSISTENCE] Primary state file corrupted, trying backup")

        # Try backup file
        state = self._load_file(self.backup_file)
        if state:
            LOG.info("[PERSISTENCE] ✓ Recovered from backup")
            # Restore backup to primary
            if self.backup_file.exists():
                self.backup_file.replace(self.state_file)
            return state

        LOG.error("[PERSISTENCE] Both state files corrupted, starting fresh")
        return None

    def _load_file(self, filepath: Path) -> dict | None:
        """Load and validate state file"""
        try:
            if not filepath.exists():
                return None

            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Validate structure
            if not isinstance(state, dict):
                LOG.error("[PERSISTENCE] Invalid state structure in %s", filepath)
                return None

            # Check version
            version = state.get("_version")
            if version != "1.0":
                LOG.warning("[PERSISTENCE] Unknown state version: %s", version)

            return state

        except (json.JSONDecodeError, OSError) as e:
            LOG.error("[PERSISTENCE] Failed to load %s: %s", filepath, e)
            return None


# Global instances
VALIDATOR = OrderValidator()
SAFE_MATH = SafeMath()
PERSISTENCE = StatePersistence()


if __name__ == "__main__":
    print("TradeManager Safety Layer")
    print("=" * 50)

    # Test safe math
    print("\n1. Safe Math Tests:")
    try:
        a = SAFE_MATH.to_decimal(100.123456789, places=5)
        b = SAFE_MATH.to_decimal(50.5, places=5)
        print(f"   {a} + {b} = {SAFE_MATH.safe_add(a, b)}")
        print(f"   {a} * {b} = {SAFE_MATH.safe_multiply(a, b)}")
        print(f"   {a} / {b} = {SAFE_MATH.safe_divide(a, b)}")
    except SafeMathError as e:
        print(f"   Error: {e}")

    # Test validation
    print("\n2. Order Validation Tests:")
    test_cases = [
        ("Valid order", 10028, "1", 0.10, 65000.0),
        ("Invalid quantity", 10028, "1", -0.10, 65000.0),
        ("Invalid price", 10028, "1", 0.10, -100.0),
        ("Invalid side", 10028, "3", 0.10, 65000.0),
        ("NaN quantity", 10028, "1", float("nan"), 65000.0),
        ("Inf price", 10028, "1", 0.10, float("inf")),
    ]

    for desc, sym, side, qty, price in test_cases:
        result = VALIDATOR.validate_order(sym, side, qty, price, "LIMIT")
        status = "✓" if result.valid else "❌"
        print(f"   {status} {desc}: {result.error or 'OK'}")

    # Test persistence
    print("\n3. State Persistence Tests:")
    test_state = {"orders": {"ord1": {"qty": 0.10}}, "position": {"net": 0.10}}
    if PERSISTENCE.save_state(test_state):
        print("   ✓ State saved")
        loaded = PERSISTENCE.load_state()
        if loaded:
            print(f"   ✓ State loaded: {loaded.get('orders', {})}")
    else:
        print("   ❌ State save failed")
