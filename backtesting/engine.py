"""
Backtesting engine — bar-by-bar simulation.
Symbol-agnostic, broker spec injected, no lookahead.
Supports pyramiding on strong trends.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Dict, Any, Callable
from core.math_utils import safe_div
from core.numeric import non_negative, clamp, round_to_step, is_valid_number
from core.validator import BrokerSpec
from core.logger import get_logger
from pipeline.cleaner import Bar

logger = get_logger("backtest")


class TradeDirection(IntEnum):
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    trade_id: str
    direction: TradeDirection
    entry_price: float
    entry_bar: int
    entry_time: float
    volume: float
    stop_loss: float
    take_profit: float
    symbol: str
    tf: str
    exit_price: float = 0.0
    exit_bar: int = 0
    exit_time: float = 0.0
    pnl: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    bars_held: int = 0
    is_open: bool = True
    pyramid_level: int = 1   # 1 = initial, 2+ = pyramid add

    def update_excursions(self, high: float, low: float) -> None:
        if self.direction == TradeDirection.LONG:
            self.mfe = max(self.mfe, non_negative(high - self.entry_price))
            self.mae = max(self.mae, non_negative(self.entry_price - low))
        else:
            self.mfe = max(self.mfe, non_negative(self.entry_price - low))
            self.mae = max(self.mae, non_negative(high - self.entry_price))


@dataclass
class BacktestConfig:
    initial_balance: float = 10_000.0
    risk_per_trade: float = 100.0          # fixed $ risk per trade
    lots_per_1000: float = 0.01            # lot size per $1000 balance
    use_dynamic_sizing: bool = False       # if True, use lots_per_1000 * balance
    max_open_trades: int = 3
    max_pyramid_levels: int = 3            # max add-ons per trend
    pyramid_atr_step: float = 1.0         # add pyramid position every N ATR
    commission_model: str = "spec"         # 'spec' uses BrokerSpec friction
    slippage_pips: float = 0.5
    allow_short: bool = True


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    balance_curve: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    final_balance: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """
    Bar-by-bar backtesting engine.
    Strategy logic is injected via on_bar callback:

        def on_bar(bar, bar_index, engine) -> Optional[Signal]:
            ...
            return Signal(direction=1, stop_loss=..., take_profit=..., volume=...)

    Signal returning None = no trade this bar.
    """

    def __init__(self, config: BacktestConfig, spec: BrokerSpec) -> None:
        self._config = config
        self._spec = spec
        self._balance = config.initial_balance
        self._equity = config.initial_balance
        self._open_trades: List[Trade] = []
        self._closed_trades: List[Trade] = []
        self._equity_curve: List[float] = []
        self._balance_curve: List[float] = []
        self._timestamps: List[float] = []
        self._trade_counter = 0
        self._bar_index = 0

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def open_trades(self) -> List[Trade]:
        return list(self._open_trades)

    @property
    def closed_trades(self) -> List[Trade]:
        return list(self._closed_trades)

    def run(self, bars: List[Bar], on_bar: Callable) -> BacktestResult:
        """
        Run backtest over bars list.
        on_bar(bar, bar_index, engine) -> Signal | None
        """
        self._reset()

        for i, bar in enumerate(bars):
            self._bar_index = i

            # 1. Update open trade excursions and check SL/TP
            self._process_bar(bar, i)

            # 2. Record equity
            self._update_equity(bar.close)
            self._equity_curve.append(self._equity)
            self._balance_curve.append(self._balance)
            self._timestamps.append(bar.timestamp)

            # 3. Call strategy
            signal = on_bar(bar, i, self)
            if signal is not None and self._can_open_trade():
                self._open_trade(signal, bar, i)

        # Close any remaining open trades at last price
        if bars:
            last_bar = bars[-1]
            for trade in list(self._open_trades):
                self._close_trade(trade, last_bar.close, self._bar_index, last_bar.timestamp, "end_of_data")

        return self._build_result()

    def _process_bar(self, bar: Bar, bar_index: int) -> None:
        for trade in list(self._open_trades):
            trade.update_excursions(bar.high, bar.low)
            trade.bars_held += 1

            hit_sl, hit_tp = False, False
            exit_price = 0.0

            if trade.direction == TradeDirection.LONG:
                if bar.low <= trade.stop_loss:
                    hit_sl = True
                    exit_price = trade.stop_loss - self._spec.pip_size * self._config.slippage_pips
                elif bar.high >= trade.take_profit:
                    hit_tp = True
                    exit_price = trade.take_profit
            else:
                if bar.high >= trade.stop_loss:
                    hit_sl = True
                    exit_price = trade.stop_loss + self._spec.pip_size * self._config.slippage_pips
                elif bar.low <= trade.take_profit:
                    hit_tp = True
                    exit_price = trade.take_profit

            if hit_sl or hit_tp:
                reason = "tp" if hit_tp else "sl"
                self._close_trade(trade, exit_price, bar_index, bar.timestamp, reason)

    def _open_trade(self, signal: "Signal", bar: Bar, bar_index: int) -> None:
        volume = self._calculate_volume(signal, bar.close)
        if volume < self._spec.min_volume:
            return

        self._trade_counter += 1
        trade = Trade(
            trade_id=f"T{self._trade_counter:05d}",
            direction=TradeDirection(signal.direction),
            entry_price=bar.close,
            entry_bar=bar_index,
            entry_time=bar.timestamp,
            volume=volume,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            symbol=self._spec.symbol,
            tf=bar.timeframe,
            pyramid_level=signal.pyramid_level,
        )

        friction = self._spec.friction_cost(volume, bar.close)
        self._balance -= friction

        self._open_trades.append(trade)
        logger.trade(
            "open", self._spec.symbol, trade.trade_id,
            direction=signal.direction, price=bar.close,
            volume=volume, sl=signal.stop_loss, tp=signal.take_profit,
        )

    def _close_trade(self, trade: Trade, exit_price: float, bar_index: int, timestamp: float, reason: str) -> None:
        if not is_valid_number(exit_price) or exit_price <= 0:
            exit_price = trade.entry_price

        # P&L: price_diff / tick_size * tick_value * volume
        # tick_value = monetary value of one tick_size move per lot
        if trade.direction == TradeDirection.LONG:
            pnl = (exit_price - trade.entry_price) / self._spec.tick_size * self._spec.tick_value * trade.volume
        else:
            pnl = (trade.entry_price - exit_price) / self._spec.tick_size * self._spec.tick_value * trade.volume

        trade.exit_price = exit_price
        trade.exit_bar = bar_index
        trade.exit_time = timestamp
        trade.pnl = pnl
        trade.is_open = False

        self._balance += pnl
        self._balance = max(0.0, self._balance)
        self._open_trades.remove(trade)
        self._closed_trades.append(trade)

        logger.trade(
            f"close_{reason}", self._spec.symbol, trade.trade_id,
            pnl=round(pnl, 2), exit=exit_price, bars=trade.bars_held,
        )

    def _update_equity(self, current_price: float) -> None:
        unrealised = 0.0
        for trade in self._open_trades:
            if trade.direction == TradeDirection.LONG:
                upnl = (current_price - trade.entry_price) * trade.volume / self._spec.pip_size * self._spec.tick_value
            else:
                upnl = (trade.entry_price - current_price) * trade.volume / self._spec.pip_size * self._spec.tick_value
            unrealised += upnl
        self._equity = self._balance + unrealised

    def _calculate_volume(self, signal: "Signal", price: float) -> float:
        if signal.volume is not None and signal.volume > 0:
            return self._spec.round_volume(signal.volume)

        if self._config.use_dynamic_sizing:
            lots = self._config.lots_per_1000 * safe_div(self._balance, 1000.0)
        else:
            lots = safe_div(self._config.risk_per_trade, self._spec.tick_value * 100)

        return self._spec.round_volume(non_negative(lots))

    def _can_open_trade(self) -> bool:
        return len(self._open_trades) < self._config.max_open_trades

    def _build_result(self) -> BacktestResult:
        from .metrics import compute_metrics
        result = BacktestResult(
            trades=self._closed_trades,
            equity_curve=self._equity_curve,
            balance_curve=self._balance_curve,
            timestamps=self._timestamps,
            final_balance=self._balance,
        )
        result.metrics = compute_metrics(self._closed_trades, self._config.initial_balance, self._equity_curve)
        return result

    def _reset(self) -> None:
        self._balance = self._config.initial_balance
        self._equity = self._config.initial_balance
        self._open_trades = []
        self._closed_trades = []
        self._equity_curve = []
        self._balance_curve = []
        self._timestamps = []
        self._trade_counter = 0
        self._bar_index = 0


@dataclass
class Signal:
    direction: int          # 1 = long, -1 = short
    stop_loss: float
    take_profit: float
    volume: Optional[float] = None   # None = engine calculates
    pyramid_level: int = 1
