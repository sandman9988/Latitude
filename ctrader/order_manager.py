"""
cTrader order management — place, cancel, and close orders via the connector.

All order placement goes through this module so there is one place to add
risk checks, logging, and error handling before any message hits the wire.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.logger import get_logger

logger = get_logger("order_manager")

try:
    from ctrader_open_api.messages import OpenApiMessages_pb2 as _api_msgs
    _CTRADER_AVAILABLE = True
except ImportError:
    _CTRADER_AVAILABLE = False
    _api_msgs = None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Position:
    position_id: int
    symbol_id: int
    direction: int        # 1 = long, -1 = short
    volume: float         # lots
    entry_price: float
    current_price: float = 0.0
    unrealised_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.direction == 1


@dataclass
class AccountState:
    balance: float = 0.0
    equity: float = 0.0
    margin_used: float = 0.0
    free_margin: float = 0.0
    positions: List[Position] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class OrderManager:
    """
    Thin wrapper around CTraderConnector for order operations.
    Translates Signal objects and strategy intents into protobuf messages.
    """

    # cTrader order type constants
    _MARKET_ORDER = 1        # ProtoOAOrderType.MARKET
    _ORDER_BUY    = 1        # ProtoOATradeSide.BUY
    _ORDER_SELL   = 2        # ProtoOATradeSide.SELL

    def __init__(self, connector: Any, account_id: int) -> None:
        self._conn = connector
        self._account_id = account_id

    # -----------------------------------------------------------------------
    # Orders
    # -----------------------------------------------------------------------

    def place_market_order(
        self,
        symbol_id: int,
        direction: int,     # 1 = long, -1 = short
        volume_lots: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        label: str = "latitude",
        comment: str = "",
    ) -> Optional[int]:
        """
        Send a market order. Returns position_id on success, None on failure.
        volume_lots is in standard lots (1.0 = 100,000 units for FX).
        SL/TP are absolute price levels (0.0 = not set).
        """
        if not _CTRADER_AVAILABLE:
            raise RuntimeError("ctrader_open_api not installed")

        # cTrader volumes are in units (lots * 100 for 0.01 lot granularity)
        volume_units = int(round(volume_lots * 100))
        if volume_units <= 0:
            logger.warning("place_market_order: volume_units=0 — skipping")
            return None

        req = _api_msgs.ProtoOANewOrderReq()
        req.ctidTraderAccountId = self._account_id
        req.symbolId = symbol_id
        req.orderType = self._MARKET_ORDER
        req.tradeSide = self._ORDER_BUY if direction == 1 else self._ORDER_SELL
        req.volume = volume_units
        req.label = label[:32]
        if comment:
            req.comment = comment[:64]

        if stop_loss > 0:
            req.stopLoss = stop_loss
            req.trailingStopLoss = False
        if take_profit > 0:
            req.takeProfit = take_profit

        resp = self._conn.send_and_wait(req)
        if resp is None:
            logger.error("place_market_order: no response (timeout)")
            return None

        descriptor_name = getattr(getattr(resp, "DESCRIPTOR", None), "name", "")
        if descriptor_name == "ProtoOAErrorRes":
            err_code = getattr(resp, "errorCode", "?")
            desc = getattr(resp, "description", "")
            logger.error(f"place_market_order: API error {err_code}: {desc}")
            return None

        # ProtoOAExecutionEvent contains the deal/position
        deal = getattr(resp, "deal", None)
        if deal is not None:
            position_id = int(getattr(deal, "positionId", 0))
            logger.info(
                f"Order filled: symbol={symbol_id} dir={'long' if direction==1 else 'short'} "
                f"lots={volume_lots} pos_id={position_id}"
            )
            return position_id or None

        logger.warning("place_market_order: execution response had no deal field")
        return None

    def close_position(
        self,
        position_id: int,
        volume_lots: Optional[float] = None,
    ) -> bool:
        """
        Close a position fully (or partially if volume_lots is specified).
        Returns True on success.
        """
        if not _CTRADER_AVAILABLE:
            raise RuntimeError("ctrader_open_api not installed")

        req = _api_msgs.ProtoOAClosePositionReq()
        req.ctidTraderAccountId = self._account_id
        req.positionId = position_id

        if volume_lots is not None:
            req.volume = int(round(volume_lots * 100))

        resp = self._conn.send_and_wait(req)
        if resp is None:
            logger.error(f"close_position {position_id}: timeout")
            return False

        descriptor_name = getattr(getattr(resp, "DESCRIPTOR", None), "name", "")
        if descriptor_name == "ProtoOAErrorRes":
            err_code = getattr(resp, "errorCode", "?")
            logger.error(f"close_position {position_id}: API error {err_code}")
            return False

        logger.info(f"Position {position_id} closed")
        return True

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending (limit/stop) order."""
        if not _CTRADER_AVAILABLE:
            raise RuntimeError("ctrader_open_api not installed")

        req = _api_msgs.ProtoOACancelOrderReq()
        req.ctidTraderAccountId = self._account_id
        req.orderId = order_id

        resp = self._conn.send_and_wait(req)
        if resp is None:
            logger.error(f"cancel_order {order_id}: timeout")
            return False

        descriptor_name = getattr(getattr(resp, "DESCRIPTOR", None), "name", "")
        return descriptor_name != "ProtoOAErrorRes"

    # -----------------------------------------------------------------------
    # Account / position queries
    # -----------------------------------------------------------------------

    def get_account_state(self) -> AccountState:
        """Fetch current balance, equity, and open positions."""
        state = AccountState()

        # Balance / equity from ProtoOATraderReq
        trader = self._fetch_trader()
        if trader is not None:
            state.balance    = float(getattr(trader, "balance",    0)) / 100.0
            state.equity     = float(getattr(trader, "equity",     0)) / 100.0  if hasattr(trader, "equity")     else state.balance
            state.margin_used  = float(getattr(trader, "usedMargin",  0)) / 100.0 if hasattr(trader, "usedMargin")  else 0.0
            state.free_margin  = float(getattr(trader, "freeMargin",  0)) / 100.0 if hasattr(trader, "freeMargin")  else state.equity

        # Open positions from ProtoOAReconcileReq
        positions = self._fetch_positions()
        state.positions = positions

        return state

    def get_open_positions(self) -> List[Position]:
        return self._fetch_positions()

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _fetch_trader(self) -> Optional[Any]:
        req = _api_msgs.ProtoOATraderReq()
        req.ctidTraderAccountId = self._account_id
        resp = self._conn.send_and_wait(req)
        if resp is None or hasattr(resp, "errorCode"):
            return None
        return getattr(resp, "trader", None)

    def _fetch_positions(self) -> List[Position]:
        req = _api_msgs.ProtoOAReconcileReq()
        req.ctidTraderAccountId = self._account_id
        resp = self._conn.send_and_wait(req)
        if resp is None or hasattr(resp, "errorCode"):
            return []

        result: List[Position] = []
        for p in getattr(resp, "position", []):
            details = getattr(p, "tradeData", None)
            if details is None:
                continue
            side = getattr(details, "tradeSide", 1)
            direction = 1 if side == 1 else -1
            result.append(Position(
                position_id  = int(getattr(p, "positionId", 0)),
                symbol_id    = int(getattr(details, "symbolId", 0)),
                direction    = direction,
                volume       = float(getattr(details, "volume", 0)) / 100.0,
                entry_price  = float(getattr(p, "price", 0)),
                stop_loss    = float(getattr(p, "stopLoss",   0)) if getattr(p, "stopLoss",   0) else 0.0,
                take_profit  = float(getattr(p, "takeProfit", 0)) if getattr(p, "takeProfit", 0) else 0.0,
            ))
        return result
