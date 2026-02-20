#!/usr/bin/env python3
"""
friction_costs.py

Friction Cost Calculator for cTrader
Tracks and models all trading costs: spread, commission, swap, slippage

Per handbook: Friction is asymmetric (buy vs sell different) and varies by:
- Time of day (wider spreads during illiquid hours)
- Volatility regime (wider spreads in volatile markets)
- Position size (larger orders have worse slippage)

This module extracts costs from cTrader's FIX messages and provides
real-time friction estimates for position sizing and reward calculations.
"""

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from src.persistence.learned_parameters import LearnedParametersManager

LOG = logging.getLogger(__name__)

MAX_SPREAD_PIPS: Final[float] = 1000.0
MIN_SPREAD_SAMPLES: Final[int] = 100


@dataclass
class SymbolCosts:
    """Trading costs for a specific symbol from cTrader."""

    symbol: str
    symbol_id: int

    # From cTrader symbol specification
    digits: int = 2  # Price precision (BTCUSD usually 2 decimals: 50000.00)
    pip_size: float = 1.0  # Size of 1 pip in price units (e.g., $1 for BTCUSD)
    tick_size: float = 0.01  # Minimum price movement
    pip_value_per_lot: float = 10.0  # USD value of 1 pip for 1 standard lot
    contract_size: float = 100000.0  # Contract size (100k for standard forex lot, 1 for index CFD)

    # Commission structure
    commission_per_lot: float = 0.0  # Fixed commission per lot
    commission_percentage: float = 0.0  # Percentage commission (e.g., 0.001 = 0.1%)
    commission_type: str = "ABSOLUTE"  # "ABSOLUTE" or "PERCENTAGE"
    min_commission: float = 0.0

    # Swap rates (overnight financing)
    swap_long: float = 0.0  # Swap for long positions (pips per lot per day)
    swap_short: float = 0.0  # Swap for short positions (pips per lot per day)
    swap_type: str = "PIPS"  # "PIPS", "PERCENTAGE", or "POINTS"
    triple_swap_day: int = 2  # Day of week for triple swap (0=Mon..6=Sun). Commonly Wednesday=2.

    # Position limits
    min_volume: float = 0.01  # Minimum position size
    max_volume: float = 100.0  # Maximum position size
    volume_step: float = 0.01  # Lot size increment

    # Market hours (for spread modeling)
    trading_hours_start: int = 0  # UTC hour when market opens
    trading_hours_end: int = 24  # UTC hour when market closes

    # Observed characteristics
    avg_spread_pips: float = 0.0  # Average observed spread
    min_spread_pips: float = 0.0  # Tightest observed spread
    max_spread_pips: float = 0.0  # Widest observed spread

    # Update tracking
    last_updated: datetime | None = None


class SpreadTracker:
    """
    Track real-time spreads to model spread patterns.

    Spreads vary by:
    - Time of day (wider during Asian session for BTC)
    - Volatility (wider during news events)
    - Liquidity (wider during holidays)
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.spreads = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        # Hourly spread buckets (0-23 UTC)
        self.hourly_spreads: dict[int, deque] = {h: deque(maxlen=100) for h in range(24)}

    def update(self, bid: float, ask: float, pip_size: float = 1.0):
        """Record a new bid/ask spread with defensive validation."""
        import math

        # Defensive: Validate inputs
        if not all(isinstance(x, (int, float)) for x in (bid, ask, pip_size)):
            return
        if not all(math.isfinite(x) for x in (bid, ask, pip_size)):
            return
        if bid <= 0 or ask <= 0 or pip_size <= 0:
            LOG.warning("Invalid values: bid=%.2f ask=%.2f pip_size=%.6f", bid, ask, pip_size)
            return

        # Defensive: Check for crossed book
        if ask <= bid:
            LOG.warning("Crossed book: bid=%.2f >= ask=%.2f", bid, ask)
            return

        # Calculate spread in pips
        spread_pips = (ask - bid) / pip_size

        # Defensive: Cap extreme spreads (likely data error)
        # For BTC ~$100k, spread should be $1-$50 typically
        # Cap at MAX_SPREAD_PIPS (0.1% for BTC) as safety limit
        if spread_pips > MAX_SPREAD_PIPS:
            LOG.warning("Extreme spread detected: %.2f pips, capping to %.0f", spread_pips, MAX_SPREAD_PIPS)
            spread_pips = MAX_SPREAD_PIPS

        # Defensive: Validate final value
        if not math.isfinite(spread_pips) or spread_pips < 0:
            LOG.warning("Invalid spread_pips: %.2f", spread_pips)
            return

        now = datetime.now(UTC)
        hour = now.hour

        self.spreads.append(spread_pips)
        self.timestamps.append(now)
        self.hourly_spreads[hour].append(spread_pips)

    def get_current_spread(self) -> float:
        """Get most recent spread in pips."""
        if not self.spreads:
            return 0.0
        return self.spreads[-1]

    def get_avg_spread(self) -> float:
        """Get average spread over window with defensive validation."""
        if not self.spreads:
            return 0.0
        try:
            avg = statistics.mean(self.spreads)
            # Defensive: Validate result
            import math

            if not math.isfinite(avg) or avg < 0:
                return 0.0
            return avg
        except (statistics.StatisticsError, ValueError):
            return 0.0

    def get_min_spread(self) -> float:
        """Get tightest spread observed with defensive validation."""
        if not self.spreads:
            return 0.0
        try:
            min_spread = min(self.spreads)
            # Defensive: Ensure non-negative
            return max(0.0, min_spread)
        except (ValueError, TypeError):
            return 0.0

    def get_max_spread(self) -> float:
        """Get widest spread observed with defensive validation."""
        if not self.spreads:
            return 0.0
        try:
            max_spread = max(self.spreads)
            # Defensive: Cap at reasonable limit
            return min(MAX_SPREAD_PIPS, max_spread)
        except (ValueError, TypeError):
            return 0.0

    def get_hourly_avg_spread(self, hour: int) -> float:
        """Get average spread for specific hour (0-23 UTC)."""
        if hour not in self.hourly_spreads or not self.hourly_spreads[hour]:
            return self.get_avg_spread()  # Fallback to overall average
        return statistics.mean(self.hourly_spreads[hour])

    def get_current_hour_spread(self) -> float:
        """Get average spread for current hour."""
        now = datetime.now(UTC)
        return self.get_hourly_avg_spread(now.hour)

    def get_learned_max_spread(self, multiplier: float = 2.0) -> float:
        """
        Calculate learned maximum acceptable spread.

        Uses historical minimum spread * multiplier to determine
        what spread should be considered "acceptable" for trading.
        This adapts to market conditions rather than using hardcoded thresholds.

        Args:
            multiplier: How many times minimum spread is acceptable
                       2.0 = allow up to 2x the tightest observed spread
                       3.0 = allow up to 3x the tightest observed spread

        Returns:
            Maximum acceptable spread in pips based on learned behavior
        """
        import math

        min_spread = self.get_min_spread()

        # Defensive: Need enough data points
        if len(self.spreads) < MIN_SPREAD_SAMPLES:
            # Insufficient data - return current spread as acceptable
            # (allows trading during warmup phase)
            current = self.get_current_spread()
            return current if current > 0 else float("inf")

        # Defensive: Validate minimum spread
        if min_spread <= 0 or not math.isfinite(min_spread):
            # No valid minimum - use average as baseline
            min_spread = self.get_avg_spread()

        # Calculate learned threshold
        max_acceptable = min_spread * multiplier

        # Defensive: Ensure reasonable bounds
        if not math.isfinite(max_acceptable) or max_acceptable <= 0:
            return float("inf")  # Don't block trades if calculation fails

        return max_acceptable


class SlippageModel:
    """
    Model slippage as a function of position size and market conditions.

    Per handbook: Slippage is ASYMMETRIC:
    - Market buy (aggressive): worse fill than ask
    - Market sell (aggressive): worse fill than bid
    - Limit orders: no slippage (but may not fill)

    Slippage increases with:
    - Position size (market impact)
    - Volatility (wider bid-ask, faster price movement)
    - Low liquidity (thinner order book)
    """

    def __init__(self):
        # Base slippage (pips) for 1 standard lot
        self.base_slippage_pips = 0.5

        # Slippage scaling factor (how much slippage increases with size)
        # slippage = base * (1 + scale * (size - 1))
        self.size_scale = 0.2  # 20% increase per additional lot

        # Volatility adjustment (higher vol = more slippage)
        self.volatility_multiplier = 1.0

        # Asymmetry: buys typically have worse slippage than sells
        self.buy_multiplier = 1.2  # Buys pay 20% more slippage
        self.sell_multiplier = 1.0

    def estimate_slippage(self, quantity: float, side: str = "BUY", volatility_factor: float = 1.0) -> float:
        """
        Estimate slippage in pips with defensive validation.

        Args:
            quantity: Position size in lots
            side: "BUY" or "SELL"
            volatility_factor: Multiplier for volatility (1.0 = normal, 2.0 = 2x vol)

        Returns:
            Expected slippage in pips
        """
        import math

        # Defensive: Validate inputs
        if not isinstance(quantity, (int, float)):
            return 0.0
        if not math.isfinite(quantity) or quantity <= 0:
            return 0.0

        # Cap extreme quantities
        quantity = min(quantity, 1000.0)

        # Defensive: Validate volatility_factor
        if not math.isfinite(volatility_factor) or volatility_factor < 0:
            volatility_factor = 1.0

        # Cap extreme volatility (10x = extreme market stress)
        volatility_factor = min(volatility_factor, 10.0)

        # Base slippage increases with size (square root to avoid extreme scaling)
        size_factor = 1.0 + self.size_scale * (quantity**0.5 - 1.0)

        # Defensive: Validate size_factor
        if not math.isfinite(size_factor):
            size_factor = 1.0

        # Apply asymmetry
        side_mult = self.buy_multiplier if side.upper() == "BUY" else self.sell_multiplier

        # Total slippage
        slippage = self.base_slippage_pips * size_factor * side_mult * volatility_factor

        # Defensive: Validate result
        if not math.isfinite(slippage):
            return 0.0

        # Cap at reasonable maximum (100 pips = extreme slippage)
        return max(0.0, min(slippage, 100.0))


class FrictionCalculator:
    """
    Calculate total friction costs for a trade.

    Total friction = spread + commission + swap + slippage

    All costs are converted to USD (or account currency) for consistency.
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",  # Instrument-agnostic: default for tests
        symbol_id: int = 10028,
        timeframe: str = "M15",
        broker: str = "default",
        param_manager: LearnedParametersManager | None = None,
    ):
        self.symbol = symbol
        self.symbol_id = symbol_id
        self.timeframe = timeframe
        self.broker = broker
        self.param_manager = param_manager
        self._param_refresh_interval = 300  # seconds
        self._last_param_refresh = 0.0
        self.spread_multiplier = 2.0
        self.depth_buffer = 0.10
        self.depth_levels = 5

        # Symbol cost specification
        self.costs = SymbolCosts(symbol=symbol, symbol_id=symbol_id)

        # Load symbol specs from config file (fallback for SecurityDefinition)
        self._load_symbol_specs_from_config()

        # Real-time spread tracking
        self.spread_tracker = SpreadTracker()

        # Slippage modeling
        self.slippage_model = SlippageModel()

        self._load_learned_parameters(force=True)
        LOG.info(
            "FrictionCalculator initialized for %s (id=%d) [tf=%s broker=%s] digits=%d min=%.4f max=%.2f step=%.4f",
            symbol,
            symbol_id,
            timeframe,
            broker,
            self.costs.digits,
            self.costs.min_volume,
            self.costs.max_volume,
            self.costs.volume_step,
        )

    def _load_symbol_specs_from_config(self) -> None:
        """
        Load symbol specifications from config/symbol_specs.json as fallback.

        SecurityDefinition from broker takes precedence when received.
        This provides reasonable defaults for trading before SecurityDef arrives.
        """
        import json
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / "config" / "symbol_specs.json"
        if not config_path.exists():
            LOG.debug("[FRICTION] No symbol_specs.json found, using defaults")
            return

        try:
            with open(config_path) as f:
                specs = json.load(f)

            # Look up by symbol name, fallback to _default
            symbol_spec = specs.get(self.symbol, specs.get("_default", {}))

            if symbol_spec:
                # Apply each field if present
                for key in [
                    "digits",
                    "pip_size",
                    "min_volume",
                    "max_volume",
                    "volume_step",
                    "contract_size",
                    "pip_value_per_lot",
                    "swap_long",
                    "swap_short",
                    "triple_swap_day",
                ]:
                    if key in symbol_spec:
                        setattr(self.costs, key, symbol_spec[key])

                self._refresh_derived_costs()
                LOG.info(
                    "[FRICTION] Loaded symbol specs from config for %s: digits=%d min=%.4f max=%.2f step=%.4f contract_size=%.2f swap_long=%.2f swap_short=%.2f",
                    self.symbol,
                    self.costs.digits,
                    self.costs.min_volume,
                    self.costs.max_volume,
                    self.costs.volume_step,
                    self.costs.contract_size,
                    self.costs.swap_long,
                    self.costs.swap_short,
                )
        except Exception as e:
            LOG.warning("[FRICTION] Failed to load symbol_specs.json: %s", e)

    def _ensure_param_manager(self) -> LearnedParametersManager:
        if self.param_manager is None:
            self.param_manager = LearnedParametersManager()
        return self.param_manager

    def _get_param(self, name: str, default: float) -> float:
        try:
            manager = self._ensure_param_manager()
            value = manager.get(self.symbol, name, timeframe=self.timeframe, broker=self.broker, default=default)
            return float(value)
        except Exception as exc:
            LOG.debug("[FRICTION] Falling back to default %.3f for %s (%s)", default, name, exc)
            return float(default)

    def _load_learned_parameters(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_param_refresh) < self._param_refresh_interval:
            return
        self.spread_multiplier = self._get_param("spread_relax", 2.0)
        self.depth_buffer = self._get_param("depth_buffer", 0.10)
        self.depth_levels = int(round(self._get_param("depth_levels", 5.0)))
        self._last_param_refresh = now
        LOG.info(
            "[FRICTION] Learned thresholds loaded: spread<=%.2fx min | depth_levels=%d | depth_buffer=%.2f",
            self.spread_multiplier,
            self.depth_levels,
            self.depth_buffer,
        )

    def refresh_learned_thresholds(self) -> None:
        """Force refresh of learned friction thresholds."""
        self._load_learned_parameters(force=True)

    def normalize_quantity(self, quantity: float) -> float:
        """
        Normalize quantity to broker's volume constraints.

        Symbol-agnostic: Uses min_volume, max_volume, volume_step from SecurityDefinition.
        Examples:
            - BTCUSD: min=0.01, max=100, step=0.01
            - XAUUSD: min=0.01, max=50, step=0.01
            - EURUSD: min=0.01, max=100, step=0.01

        Args:
            quantity: Raw quantity to normalize

        Returns:
            Normalized quantity conforming to min/max/step constraints
        """
        import math

        # Defensive: Handle invalid input
        if not math.isfinite(quantity) or quantity <= 0:
            return self.costs.min_volume

        # Clamp to min/max
        normalized = max(self.costs.min_volume, min(quantity, self.costs.max_volume))

        # Round to volume_step (snap to nearest valid lot size)
        if self.costs.volume_step > 0:
            steps = round(normalized / self.costs.volume_step)
            normalized = steps * self.costs.volume_step

        # Final clamp after rounding (edge case: step rounds above max)
        normalized = max(self.costs.min_volume, min(normalized, self.costs.max_volume))

        return normalized

    def normalize_price(self, price: float) -> float:
        """
        Normalize price to broker's digit precision.

        Symbol-agnostic: Uses digits from SecurityDefinition.
        Examples:
            - XAUUSD (Gold): digits=2 → 1850.12
            - USDJPY: digits=3 → 148.123
            - EURUSD: digits=5 → 1.08765
            - BTCUSD: digits=2 → 95432.12

        Args:
            price: Raw price to normalize

        Returns:
            Price rounded to instrument's tick_size/digits precision
        """
        import math

        # Defensive: Handle invalid input
        if not math.isfinite(price) or price <= 0:
            return 0.0

        # Round to tick_size precision
        if self.costs.tick_size > 0:
            # Snap to nearest tick
            normalized = round(price / self.costs.tick_size) * self.costs.tick_size
        else:
            # Fallback: use digits for precision
            precision = 10**self.costs.digits
            normalized = round(price * precision) / precision

        return normalized

    def get_symbol_info(self) -> dict:
        """
        Get current symbol specification for logging/debugging.

        Returns:
            Dict with all broker-provided symbol constraints
        """
        return {
            "symbol": self.symbol,
            "symbol_id": self.symbol_id,
            "digits": self.costs.digits,
            "tick_size": self.costs.tick_size,
            "pip_size": self.costs.pip_size,
            "min_volume": self.costs.min_volume,
            "max_volume": self.costs.max_volume,
            "volume_step": self.costs.volume_step,
            "contract_size": self.costs.contract_size,
            "pip_value_per_lot": self.costs.pip_value_per_lot,
            "last_updated": str(self.costs.last_updated) if self.costs.last_updated else None,
        }

    def update_symbol_costs(self, **kwargs):
        """
        Update symbol cost parameters from broker SecurityDefinition or config.

        Broker-provided values always take precedence over config fallbacks.
        This is the symbol-agnostic approach - all costs come from broker.

        Args:
            **kwargs: Fields from FIX SecurityDefinition message or config
                - digits: Price precision
                - pip_value_per_lot: Value of 1 pip per lot (USD)
                - commission_per_lot: Fixed commission per lot
                - commission_percentage: Percentage commission
                - swap_long: Long swap rate (pips/lot/day) - BROKER-SPECIFIC
                - swap_short: Short swap rate (pips/lot/day) - BROKER-SPECIFIC
                - triple_swap_day: Day of week for triple swap (0=Mon, 2=Wed, etc.)
                - min_volume: Minimum lot size
                - max_volume: Maximum lot size
                - volume_step: Lot size increment
                - contract_size: Units per lot (e.g., 100 for XAUUSD)
        """
        for key, value in kwargs.items():
            if hasattr(self.costs, key):
                setattr(self.costs, key, value)

        self.costs.last_updated = datetime.now(UTC)
        self._refresh_derived_costs()

        # Log full symbol info after update (including swap rates)
        info = self.get_symbol_info()
        source = "broker" if kwargs else "config"
        LOG.info(
            "[FRICTION] Symbol costs updated for %s: digits=%d tick=%.6f min=%.4f max=%.2f step=%.4f contract_size=%.2f swap_long=%.2f swap_short=%.2f (source: %s)",
            self.symbol,
            info["digits"],
            info["tick_size"],
            info["min_volume"],
            info["max_volume"],
            info["volume_step"],
            self.costs.contract_size,
            self.costs.swap_long,
            self.costs.swap_short,
            source,
        )

    def _refresh_derived_costs(self) -> None:
        """Derive tick/pip relationships from provided symbol info to avoid hardcoded values."""
        # Derive tick size from digits if provided
        if self.costs.digits is not None and self.costs.digits > 0:
            try:
                self.costs.tick_size = 10 ** (-int(self.costs.digits))
                # For FX/CFD style quoting, pip is often the minimum tick
                self.costs.pip_size = self.costs.tick_size
            except Exception:
                pass

        # Derive pip value from contract size when available
        contract_size = getattr(self.costs, "contract_size", 0)
        if contract_size and contract_size > 0 and self.costs.pip_size > 0:
            self.costs.pip_value_per_lot = contract_size * self.costs.pip_size

    def infer_digits_from_price(self, price: float) -> int:
        """
        Infer price digits from observed price when SecurityDefinition doesn't provide it.

        Symbol-agnostic heuristics:
            - XAUUSD (Gold): ~1850.12 → 2 digits
            - USDJPY: ~148.123 → 3 digits
            - EURUSD: ~1.08765 → 5 digits
            - BTCUSD: ~95432.12 → 2 digits

        Args:
            price: Observed market price

        Returns:
            Inferred number of decimal places
        """
        import math

        if not math.isfinite(price) or price <= 0:
            return 2  # Default

        # Count significant decimal places in price
        price_str = f"{price:.10f}".rstrip("0")
        if "." in price_str:
            decimal_part = price_str.split(".")[1]
            observed_digits = len(decimal_part)
        else:
            observed_digits = 0

        # Symbol-agnostic heuristics based on price magnitude
        if price > 10000:  # BTC, indices like NAS100
            return min(2, observed_digits)
        elif price > 100:  # Gold, JPY crosses, indices
            return min(2, observed_digits) if price > 1000 else min(3, observed_digits)
        elif price > 10:  # JPY pairs
            return min(3, observed_digits)
        else:  # Standard forex pairs (EURUSD, GBPUSD)
            return min(5, observed_digits)

    def update_digits_from_price(self, price: float) -> None:
        """
        Update digits if not set from SecurityDefinition.

        Only updates if current digits seems to be default/unset.
        """
        if self.costs.digits == 2 and self.costs.last_updated is None:
            inferred = self.infer_digits_from_price(price)
            if inferred != self.costs.digits:
                LOG.info(
                    "[FRICTION] Inferred digits=%d from price=%.6f (was default=%d)",
                    inferred,
                    price,
                    self.costs.digits,
                )
                self.costs.digits = inferred
                self._refresh_derived_costs()

    def update_spread(self, bid: float, ask: float):
        """Update current spread observation."""
        self.spread_tracker.update(bid, ask, self.costs.pip_size)

        # Update observed spread statistics
        self.costs.avg_spread_pips = self.spread_tracker.get_avg_spread()
        self.costs.min_spread_pips = self.spread_tracker.get_min_spread()
        self.costs.max_spread_pips = self.spread_tracker.get_max_spread()

    def calculate_spread_cost(self, quantity: float) -> float:
        """
        Calculate spread cost in USD with defensive validation.

        Args:
            quantity: Position size in lots

        Returns:
            Spread cost in USD
        """
        import math

        # Defensive: Validate inputs
        if not isinstance(quantity, (int, float)):
            return 0.0
        if not math.isfinite(quantity) or quantity <= 0:
            return 0.0

        # Cap extreme positions (safety limit)
        quantity = min(quantity, 1000.0)  # Max 1000 lots

        current_spread = self.spread_tracker.get_current_spread()
        if current_spread <= 0:
            current_spread = self.costs.avg_spread_pips or 2.0  # Fallback to 2 pips

        # Defensive: Validate spread value
        if not math.isfinite(current_spread) or current_spread < 0:
            current_spread = 2.0

        # Defensive: Validate pip_value_per_lot
        pip_value = self.costs.pip_value_per_lot
        if not math.isfinite(pip_value) or pip_value <= 0:
            pip_value = 10.0  # Default for BTCUSD

        # For BTCUSD: 1 pip per standard lot = $10
        # spread_cost = spread_pips * pip_value_per_lot * quantity
        spread_cost = current_spread * pip_value * quantity

        # Defensive: Validate result
        if not math.isfinite(spread_cost) or spread_cost < 0:
            return 0.0

        # Cap at reasonable limit (prevent extreme costs)
        return min(spread_cost, 1_000_000.0)

    def calculate_commission(self, quantity: float, price: float) -> float:
        """
        Calculate commission in USD with defensive validation.

        Args:
            quantity: Position size in lots
            price: Entry price

        Returns:
            Commission in USD
        """
        import math

        # Defensive: Validate inputs
        if not all(isinstance(x, (int, float)) for x in (quantity, price)):
            return 0.0
        if not all(math.isfinite(x) for x in (quantity, price)):
            return 0.0
        if quantity <= 0 or price <= 0:
            return 0.0

        # Cap extreme values
        quantity = min(quantity, 1000.0)
        price = min(price, 10_000_000.0)  # $10M max price

        if self.costs.commission_type == "PERCENTAGE":
            # Percentage of notional
            notional = quantity * price * 100000  # 1 lot = 100,000 units

            # Defensive: Validate commission_percentage
            comm_pct = self.costs.commission_percentage
            if not math.isfinite(comm_pct) or comm_pct < 0:
                comm_pct = 0.0007  # Default 0.07%

            # Cap commission percentage (prevent extreme rates)
            comm_pct = min(comm_pct, 0.01)  # Max 1%

            commission = notional * comm_pct
        else:
            # Fixed per lot
            comm_per_lot = self.costs.commission_per_lot

            # Defensive: Validate commission_per_lot
            if not math.isfinite(comm_per_lot) or comm_per_lot < 0:
                comm_per_lot = 7.0  # Default $7 per lot

            commission = quantity * comm_per_lot

        # Apply min commission
        min_comm = self.costs.min_commission
        if not math.isfinite(min_comm) or min_comm < 0:
            min_comm = 0.0
        commission = max(commission, min_comm)

        # Defensive: Validate result
        if not math.isfinite(commission) or commission < 0:
            return 0.0

        # Cap at reasonable limit
        return min(commission, 100_000.0)

    def calculate_swap(
        self, quantity: float, side: str, holding_days: float = 1.0, crosses_rollover: bool = False, price: float = 0.0
    ) -> float:
        """
        Calculate swap (overnight financing) cost in USD.

        CRITICAL: Swap only charged at daily rollover time (typically 5pm EST/10pm UTC).
        For intraday M5 trades that close before rollover: swap = 0
        For overnight trades crossing rollover: swap = full day rate (or 3x on Wednesday)

        Args:
            quantity: Position size in lots
            side: "BUY" (long) or "SELL" (short)
            holding_days: Expected holding period in days (for multi-day positions)
            crosses_rollover: Whether position will cross daily rollover time (default: False for intraday)

        Returns:
            Swap cost in USD (negative = you pay, positive = you earn)
            Returns 0 for intraday trades that don't cross rollover
        """
        from datetime import UTC, datetime

        # INTRADAY TRADES: No swap if not crossing rollover
        # Most M5 trades (~2.4hrs) close before rollover → swap = 0
        if not crosses_rollover and holding_days < 1.0:
            return 0.0

        swap_rate = self.costs.swap_long if side.upper() == "BUY" else self.costs.swap_short

        if self.costs.swap_type == "PIPS":
            # Swap charged at rollover, in full day increments
            now = datetime.now(UTC)
            # Validate and normalize triple_swap_day to [0..6]
            try:
                tsd = int(self.costs.triple_swap_day)
            except (TypeError, ValueError):
                tsd = 2
            if tsd < 0 or tsd > 6:
                tsd = 2  # Default to Wednesday
            is_triple_swap_day = now.weekday() == tsd

            # Calculate number of rollovers
            if crosses_rollover:
                # At least 1 rollover if crossing rollover time
                num_rollovers = max(1, int(holding_days))
                # Triple swap on Wednesday (accounts for weekend)
                if is_triple_swap_day and num_rollovers > 0:
                    num_rollovers += 2  # +2 extra days for weekend
            else:
                # Multi-day position: count full days
                num_rollovers = int(holding_days)
                if is_triple_swap_day and num_rollovers > 0:
                    num_rollovers += 2

            # For XAUUSD: swap_long=-7.2 pips, pip_value=1.0, qty=0.1
            # Intraday (crosses_rollover=False): swap = $0
            # Overnight (crosses_rollover=True): swap = -$0.72 (1 rollover)
            # Wednesday overnight: swap = -$2.16 (3 rollovers for weekend)
            swap_cost = swap_rate * self.costs.pip_value_per_lot * quantity * num_rollovers
        elif self.costs.swap_type == "PERCENTAGE":
            # Swap as annual percentage of notional value, charged per rollover day
            # swap_rate is expressed as annual % (e.g., -2.5 means -2.5% per year)
            # Formula: notional * (swap_rate / 100) / 365 * num_rollover_days
            if price <= 0:
                swap_cost = 0.0
            else:
                notional = quantity * self.costs.contract_size * price
                daily_rate = swap_rate / 100.0 / 365.0

                now = datetime.now(UTC)
                try:
                    tsd = int(self.costs.triple_swap_day)
                except (TypeError, ValueError):
                    tsd = 2
                if tsd < 0 or tsd > 6:
                    tsd = 2
                is_triple_swap_day = now.weekday() == tsd

                if crosses_rollover:
                    num_rollovers = max(1, int(holding_days))
                    if is_triple_swap_day and num_rollovers > 0:
                        num_rollovers += 2
                else:
                    num_rollovers = int(holding_days)
                    if is_triple_swap_day and num_rollovers > 0:
                        num_rollovers += 2

                swap_cost = notional * daily_rate * num_rollovers
        else:
            swap_cost = 0.0

        return swap_cost

    def calculate_slippage_cost(self, quantity: float, side: str, volatility_factor: float = 1.0) -> float:
        """
        Calculate expected slippage cost in USD.

        Args:
            quantity: Position size in lots
            side: "BUY" or "SELL"
            volatility_factor: Volatility multiplier

        Returns:
            Expected slippage cost in USD
        """
        slippage_pips = self.slippage_model.estimate_slippage(quantity, side, volatility_factor)

        # Convert pips to USD
        # For BTCUSD: 1 pip per lot = $10 per standard lot
        slippage_cost = slippage_pips * self.costs.pip_value_per_lot * quantity

        return slippage_cost

    def calculate_total_friction(
        self,
        quantity: float,
        side: str,
        price: float,
        holding_days: float = 1.0,
        volatility_factor: float = 1.0,
        crosses_rollover: bool = False,
    ) -> dict[str, float]:
        """
        Calculate all friction costs for a trade.

        Same logic for BOTH paper trading and live trading.

        Args:
            quantity: Position size in lots
            side: "BUY" or "SELL"
            price: Entry price
            holding_days: Expected holding period in days
            volatility_factor: Volatility multiplier for slippage
            crosses_rollover: Whether position crosses daily rollover (default: False for M5 intraday)

        Returns:
            Dictionary with breakdown of costs:
                - spread: Spread cost (USD)
                - commission: Commission cost (USD)
                - swap: Swap cost (USD, 0 for intraday trades, can be negative)
                - slippage: Expected slippage (USD)
                - total: Total friction (USD)
                - total_pips: Total friction in pips
        """
        spread = self.calculate_spread_cost(quantity)
        commission = self.calculate_commission(quantity, price)
        swap = self.calculate_swap(quantity, side, holding_days, crosses_rollover, price=price)
        slippage = self.calculate_slippage_cost(quantity, side, volatility_factor)

        total = spread + commission + swap + slippage

        # Convert total back to pips for reference
        if quantity > 0 and self.costs.pip_value_per_lot > 0:
            total_pips = total / (self.costs.pip_value_per_lot * quantity)
        else:
            total_pips = 0.0

        return {
            "spread": spread,
            "commission": commission,
            "swap": swap,
            "slippage": slippage,
            "total": total,
            "total_pips": total_pips,
            "quantity": quantity,
            "side": side,
            "price": price,
        }

    def get_friction_adjusted_pnl(
        self,
        raw_pnl: float,
        quantity: float,
        side: str,
        entry_price: float,
        holding_days: float = 1.0,
    ) -> float:
        """
        Adjust raw P&L for friction costs.

        Args:
            raw_pnl: Raw P&L before costs
            quantity: Position size
            side: "BUY" or "SELL"
            entry_price: Entry price
            holding_days: How long position was held

        Returns:
            Net P&L after friction
        """
        friction = self.calculate_total_friction(quantity, side, entry_price, holding_days)

        net_pnl = raw_pnl - friction["total"]

        LOG.debug(
            "Friction-adjusted P&L: raw=%.2f friction=%.2f net=%.2f",
            raw_pnl,
            friction["total"],
            net_pnl,
        )

        return net_pnl

    def get_statistics(self) -> dict:
        """Get friction cost statistics."""
        return {
            "symbol": self.symbol,
            "avg_spread_pips": self.costs.avg_spread_pips,
            "min_spread_pips": self.costs.min_spread_pips,
            "max_spread_pips": self.costs.max_spread_pips,
            "current_spread_pips": self.spread_tracker.get_current_spread(),
            "commission_per_lot": self.costs.commission_per_lot,
            "swap_long": self.costs.swap_long,
            "swap_short": self.costs.swap_short,
            "base_slippage": self.slippage_model.base_slippage_pips,
            "last_updated": self.costs.last_updated,
        }

    def is_spread_acceptable(self, multiplier: float | None = None) -> tuple[bool, float, float]:
        """
        Check if current spread is acceptable for trading.

        Uses learned minimum spread * multiplier as the threshold.
        This adapts to market conditions rather than hardcoded values.

        Args:
            multiplier: How many times minimum spread is acceptable (default 2x)

        Returns:
            Tuple of (is_acceptable, current_spread, max_acceptable)
        """
        import math

        self._load_learned_parameters()
        effective_multiplier = multiplier if multiplier is not None else self.spread_multiplier
        if not math.isfinite(effective_multiplier) or effective_multiplier <= 0:
            effective_multiplier = 2.0
        current_spread = self.spread_tracker.get_current_spread()
        max_acceptable = self.spread_tracker.get_learned_max_spread(effective_multiplier)

        # Defensive: If either is invalid, allow trading (don't block on calculation error)
        if not math.isfinite(current_spread) or current_spread <= 0:
            return (True, 0.0, max_acceptable)

        if not math.isfinite(max_acceptable) or max_acceptable <= 0:
            return (True, current_spread, float("inf"))

        is_acceptable = current_spread <= max_acceptable

        return (is_acceptable, current_spread, max_acceptable)

    def get_spread_stats_for_logging(self) -> dict:
        """Get spread statistics for logging/monitoring."""
        current = self.spread_tracker.get_current_spread()
        min_spread = self.spread_tracker.get_min_spread()
        avg_spread = self.spread_tracker.get_avg_spread()
        max_acceptable = self.spread_tracker.get_learned_max_spread(multiplier=2.0)
        sample_count = len(self.spread_tracker.spreads)

        return {
            "current": current,
            "min_observed": min_spread,
            "avg": avg_spread,
            "max_acceptable_2x": max_acceptable,
            "learned_multiplier": self.spread_multiplier,
            "samples": sample_count,
        }


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing FrictionCalculator module...")

    calc = FrictionCalculator(symbol="BTCUSD", symbol_id=10028)

    # Simulate cTrader symbol info
    print("\n=== Test 1: Update symbol costs from cTrader ===")
    calc.update_symbol_costs(
        digits=2,
        pip_size=1.0,  # For BTCUSD, $1 = 1 pip
        tick_size=0.01,  # Minimum price movement
        pip_value_per_lot=10.0,  # $10 per pip per standard lot
        commission_per_lot=7.0,  # $7 per lot round-trip
        commission_type="ABSOLUTE",
        swap_long=-0.5,  # Pay 0.5 pips per day
        swap_short=-0.3,
        min_volume=0.01,
        max_volume=100.0,
    )
    print(f"✓ Symbol costs updated: commission=${calc.costs.commission_per_lot}/lot")

    # Simulate spread updates (BTCUSD typical spreads are $1-5)
    print("\n=== Test 2: Track spreads ===")
    spreads_btc = [
        (50000.00, 50001.00),  # $1 spread
        (50000.00, 50001.50),  # $1.50 spread
        (50000.00, 50002.00),  # $2 spread
        (50000.00, 50002.50),  # $2.50 spread (wider)
    ]

    for bid, ask in spreads_btc:
        calc.update_spread(bid, ask)

    print(f"✓ Spreads tracked: avg={calc.costs.avg_spread_pips:.2f} pips")
    print(f"  Min: {calc.costs.min_spread_pips:.2f} pips")
    print(f"  Max: {calc.costs.max_spread_pips:.2f} pips")

    # Calculate friction for a trade
    print("\n=== Test 3: Calculate friction for BUY 0.10 lot @ $50,000 ===")
    friction = calc.calculate_total_friction(
        quantity=0.10, side="BUY", price=50000.0, holding_days=1.0, volatility_factor=1.0
    )

    print(f"Spread cost:     ${friction['spread']:.2f}")
    print(f"Commission:      ${friction['commission']:.2f}")
    print(f"Swap (1 day):    ${friction['swap']:.2f}")
    print(f"Slippage:        ${friction['slippage']:.2f}")
    print("─────────────────────────────")
    print(f"TOTAL FRICTION:  ${friction['total']:.2f}")
    print(f"Total in pips:   {friction['total_pips']:.2f} pips")

    # Test friction-adjusted P&L
    print("\n=== Test 4: Friction-adjusted P&L ===")
    raw_pnl = 100.0  # Made $100 gross
    net_pnl = calc.get_friction_adjusted_pnl(
        raw_pnl=raw_pnl, quantity=0.10, side="BUY", entry_price=50000.0, holding_days=1.0
    )
    print(f"Raw P&L:         ${raw_pnl:.2f}")
    print(f"Friction cost:   ${friction['total']:.2f}")
    print(f"Net P&L:         ${net_pnl:.2f}")
    print(f"Friction ratio:  {(friction['total']/raw_pnl)*100:.1f}% of gross profit")

    # Show statistics
    print("\n=== Test 5: Friction statistics ===")
    stats = calc.get_statistics()
    print(f"Symbol: {stats['symbol']}")
    print(f"Average spread: {stats['avg_spread_pips']:.2f} pips")
    print(f"Commission: ${stats['commission_per_lot']}/lot")
    print(f"Swap long: {stats['swap_long']:.2f} pips/day")
    print(f"Base slippage: {stats['base_slippage']:.2f} pips")

    print("\n✅ All tests complete")
