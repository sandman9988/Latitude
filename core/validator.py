"""
Broker specification and order validation.
Symbol-agnostic — BrokerSpec is injected, not hardcoded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
from .numeric import round_to_step, clamp
from .math_utils import safe_div


@dataclass
class BrokerSpec:
    """
    All broker/symbol constraints in one place.
    Inject one of these per symbol — strategy never hardcodes values.
    """
    symbol: str
    tick_size: float          # minimum price movement e.g. 0.01, 0.25
    tick_value: float         # monetary value of one tick per 1 lot
    lot_step: float           # minimum lot increment e.g. 0.01
    min_volume: float         # minimum trade size in lots
    max_volume: float         # maximum trade size in lots
    margin_rate: float        # margin as fraction of notional e.g. 0.01 = 1%
    commission_per_lot: float # round-trip commission per lot in account currency
    spread_pips: float        # typical spread in pips (used for friction cost)
    pip_size: float           # size of one pip e.g. 0.0001 for FX, 1.0 for indices
    currency: str = "USD"
    digits: int = 5           # price decimal places

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

    def friction_cost(self, volume: float) -> float:
        """Total round-trip friction: spread + commission in account currency."""
        spread_cost = self.spread_pips * self.pip_size * self.tick_value * volume
        commission = self.commission_per_lot * volume
        return spread_cost + commission

    def margin_required(self, price: float, volume: float) -> float:
        """Approximate margin required for a position."""
        notional = price * volume * (1.0 / self.pip_size if self.pip_size < 1 else 1.0)
        return notional * self.margin_rate


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
