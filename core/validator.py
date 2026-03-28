"""
Broker specification and order validation.
Symbol-agnostic — BrokerSpec is injected, not hardcoded.
All fields sourced from ProtoOASymbol + ProtoOATrader via ctrader/spec_fetcher.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from .numeric import round_to_step, clamp
from .math_utils import safe_div


@dataclass
class TradingInterval:
    """One trading session window. start/end in seconds from Sunday 00:00 UTC."""
    start_second: int
    end_second: int


@dataclass
class Holiday:
    name: str
    date_timestamp: int       # unix ms
    is_recurring: bool
    start_second: int = 0
    end_second: int = 86400


@dataclass
class LeverageTier:
    """Dynamic leverage tier — volume in lots, leverage as ratio (e.g. 30.0 = 1:30)."""
    volume_threshold: float   # minimum volume in lots for this tier
    leverage: float           # e.g. 30.0 means 1:30


@dataclass
class BrokerSpec:
    """
    Complete broker/symbol contract specification.
    Populated from ProtoOASymbol + ProtoOATrader by ctrader/spec_fetcher.py.
    Strategy never hardcodes any of these values.
    """
    # Identity
    symbol: str
    symbol_id: int = 0
    description: str = ""
    base_asset: str = ""
    quote_asset: str = ""
    asset_class: str = ""
    category: str = ""

    # Price precision
    digits: int = 5
    pip_size: float = 0.0001       # e.g. 0.0001 FX, 1.0 indices, 0.01 gold
    tick_size: float = 0.00001     # minimum price movement
    tick_value: float = 1.0        # monetary value of one tick per 1 lot

    # Volume constraints (in lots)
    lot_size: float = 100000.0     # contract size per 1 lot (100000 FX, varies CFD)
    lot_step: float = 0.01
    min_volume: float = 0.01
    max_volume: float = 100.0

    # Margin
    margin_rate: float = 0.01      # fraction e.g. 0.01 = 1% = 1:100 leverage
    leverage_tiers: List[LeverageTier] = field(default_factory=list)

    # Commission — from ProtoOACommissionType
    # Type: 1=USD_per_million_USD, 2=USD_per_lot, 3=pct_of_value, 4=quote_ccy_per_lot
    commission_type: int = 2
    commission_rate: float = 0.0   # raw rate per commissionType unit
    min_commission: float = 0.0
    min_commission_type: int = 1   # 1=currency, 2=quote_currency

    # Swap (overnight cost)
    # swap_type: 0=pips, 1=percentage_annual, 2=points
    swap_type: int = 0
    swap_long: float = 0.0         # positive = credit, negative = debit
    swap_short: float = 0.0
    swap_rollover_day: int = 3     # day triple swap applies (3=Wed)
    swap_period: int = 1           # days between swap applications

    # Spread (typical, sourced from tick history or broker spec)
    spread_pips: float = 1.0

    # Distance constraints (for SL/TP placement)
    # distance_type: 1=points, 2=percentage
    sl_min_distance: float = 0.0
    tp_min_distance: float = 0.0
    distance_type: int = 1

    # P&L conversion fee (fraction, e.g. 0.003 = 0.3%)
    pnl_conversion_fee_rate: float = 0.0

    # Trading hours
    schedule: List[TradingInterval] = field(default_factory=list)
    schedule_timezone: str = "UTC"
    holidays: List[Holiday] = field(default_factory=list)

    # Flags
    trading_mode: int = 0          # 0=enabled, 1=disabled, 2=disabled_with_pending, 3=close_only
    short_selling_enabled: bool = True
    guaranteed_sl_available: bool = False
    swap_free: bool = False        # Islamic account

    # Account currency
    currency: str = "USD"
    money_digits: int = 2          # decimal places for monetary values

    # Conversion chain (symbol names needed to convert P&L to account currency)
    conversion_symbols: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.tick_size > 0, "tick_size must be positive"
        assert self.lot_step > 0, "lot_step must be positive"
        assert self.min_volume >= 0, "min_volume must be non-negative"
        assert self.max_volume > self.min_volume, "max_volume must exceed min_volume"
        assert 0.0 < self.margin_rate <= 1.0, "margin_rate must be in (0, 1]"

    def round_price(self, price: float) -> float:
        return round_to_step(price, self.tick_size)

    def round_volume(self, volume: float) -> float:
        rounded = round_to_step(volume, self.lot_step)
        return clamp(rounded, self.min_volume, self.max_volume)

    def effective_margin_rate(self, volume: float) -> float:
        """Return margin rate considering dynamic leverage tiers."""
        if not self.leverage_tiers:
            return self.margin_rate
        applicable = self.margin_rate
        for tier in sorted(self.leverage_tiers, key=lambda t: t.volume_threshold):
            if volume >= tier.volume_threshold:
                applicable = safe_div(1.0, tier.leverage, fallback=self.margin_rate)
        return applicable

    def commission_cost(self, volume: float, price: float = 0.0) -> float:
        """One-way commission in account currency."""
        if self.commission_type == 1:  # USD per million USD
            notional = price * volume * self.lot_size
            return safe_div(notional * self.commission_rate, 1_000_000.0)
        elif self.commission_type == 2:  # USD per lot
            return self.commission_rate * volume
        elif self.commission_type == 3:  # percentage of value
            notional = price * volume * self.lot_size
            return notional * safe_div(self.commission_rate, 100.0)
        elif self.commission_type == 4:  # quote currency per lot
            return self.commission_rate * volume
        return 0.0

    def friction_cost(self, volume: float, price: float = 0.0) -> float:
        """
        Total round-trip friction in account currency:
        spread + round-trip commission + P&L conversion fee.
        """
        spread_cost = self.spread_pips * self.pip_size * self.tick_value * volume
        commission = self.commission_cost(volume, price) * 2.0  # round-trip
        commission = max(commission, self.min_commission)
        notional = price * volume * self.lot_size if price > 0 else 0.0
        conversion_fee = notional * self.pnl_conversion_fee_rate
        return spread_cost + commission + conversion_fee

    def swap_cost_per_day(self, volume: float, price: float, is_long: bool) -> float:
        """
        Estimated daily swap cost in account currency.
        swap_type 0=pips, 1=percentage_annual, 2=points
        """
        rate = self.swap_long if is_long else self.swap_short
        if self.swap_type == 0:  # pips
            return rate * self.pip_size * self.tick_value * volume
        elif self.swap_type == 1:  # annual percentage
            notional = price * volume * self.lot_size
            return safe_div(notional * rate, 36500.0)  # daily
        elif self.swap_type == 2:  # points
            return rate * self.tick_size * self.tick_value * volume
        return 0.0

    def margin_required(self, price: float, volume: float) -> float:
        """Margin required considering dynamic leverage tiers."""
        rate = self.effective_margin_rate(volume)
        notional = price * volume * self.lot_size
        return notional * rate

    def is_trading_open(self, utc_timestamp: float) -> bool:
        """Check if market is open at given UTC timestamp (approximate — use for gap filtering)."""
        if self.trading_mode != 0:
            return False
        if not self.schedule:
            return True
        import datetime
        dt = datetime.datetime.fromtimestamp(utc_timestamp, tz=datetime.timezone.utc)
        # Seconds since Sunday 00:00 UTC
        weekday = dt.weekday()  # 0=Mon
        sunday_offset = (weekday + 1) % 7
        sec_since_sunday = sunday_offset * 86400 + dt.hour * 3600 + dt.minute * 60 + dt.second
        for interval in self.schedule:
            if interval.start_second <= sec_since_sunday <= interval.end_second:
                return True
        return False


@dataclass
class OrderRequest:
    symbol: str
    volume: float
    price: float          # 0.0 for market orders
    stop_loss: float
    take_profit: float
    is_buy: bool
    comment: str = ""


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)


def validate_order(order: OrderRequest, spec: BrokerSpec, account_balance: float) -> ValidationResult:
    """
    Validate an order against broker constraints.
    Returns ValidationResult — strategy must check .valid before submitting.
    """
    result = ValidationResult(valid=True)

    if not math.isfinite(order.volume) or order.volume <= 0:
        result.add_error(f"Invalid volume: {order.volume}")

    if order.volume < spec.min_volume:
        result.add_error(f"Volume {order.volume} below minimum {spec.min_volume}")

    if order.volume > spec.max_volume:
        result.add_error(f"Volume {order.volume} exceeds maximum {spec.max_volume}")

    rounded_vol = spec.round_volume(order.volume)
    if abs(rounded_vol - order.volume) > spec.lot_step * 0.5:
        result.add_error(f"Volume {order.volume} not aligned to lot step {spec.lot_step}")

    if order.price < 0:
        result.add_error(f"Invalid price: {order.price}")

    if order.stop_loss <= 0:
        result.add_error("Stop loss must be set and positive")

    if order.is_buy and order.stop_loss >= order.price and order.price > 0:
        result.add_error("Buy stop loss must be below entry price")

    if not order.is_buy and order.stop_loss <= order.price and order.price > 0:
        result.add_error("Sell stop loss must be above entry price")

    margin = spec.margin_required(
        order.price if order.price > 0 else order.stop_loss,
        order.volume
    )
    if margin > account_balance:
        result.add_error(f"Insufficient margin: need {margin:.2f}, have {account_balance:.2f}")

    return result
