"""
TradeManager - Centralized FIX Protocol Order & Position Management

Handles complete order lifecycle using FIX 4.4 Protocol:
- Order submission via NewOrderSingle (35=D)
- ExecutionReport processing (35=8)
- Order state tracking (Tag 39: OrdStatus)
- Position reconciliation (35=AN/AP)
- Order modification via OrderCancelReplaceRequest (35=G)
- Order cancellation via OrderCancelRequest (35=F)

FIX Tag Reference:
    Tag 11 (ClOrdID): Client order identifier
    Tag 37 (OrderID): Broker's order ID
    Tag 39 (OrdStatus): 0=New, 1=PartialFill, 2=Filled, 4=Canceled, 8=Rejected
    Tag 150 (ExecType): 0=New, F=Fill, 4=Canceled, 8=Rejected, I=OrderStatus
    Tag 54 (Side): 1=Buy, 2=Sell
    Tag 55 (Symbol): Numeric instrument ID
    Tag 40 (OrdType): 1=Market, 2=Limit
    Tag 38 (OrderQty): Order quantity
    Tag 721 (PosMaintRptID): Position ID for hedged accounts
"""

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

import quickfix as fix
import quickfix44 as fix44

from safe_utils import utc_now, utc_ts_ms

LOG = logging.getLogger(__name__)


class OrderStatus(Enum):
    """FIX OrdStatus (Tag 39) values"""

    NEW = "0"
    PARTIALLY_FILLED = "1"
    FILLED = "2"
    DONE_FOR_DAY = "3"
    CANCELED = "4"
    REPLACED = "5"
    PENDING_CANCEL = "6"
    STOPPED = "7"
    REJECTED = "8"
    SUSPENDED = "9"
    PENDING_NEW = "A"
    CALCULATED = "B"
    EXPIRED = "C"
    ACCEPTED_FOR_BIDDING = "D"
    PENDING_REPLACE = "E"


class ExecType(Enum):
    """FIX ExecType (Tag 150) values"""

    NEW = "0"
    DONE_FOR_DAY = "3"
    CANCELED = "4"
    REPLACED = "5"
    PENDING_CANCEL = "6"
    STOPPED = "7"
    REJECTED = "8"
    SUSPENDED = "9"
    PENDING_NEW = "A"
    CALCULATED = "B"
    EXPIRED = "C"
    RESTATED = "D"
    PENDING_REPLACE = "E"
    TRADE = "F"  # Fill
    TRADE_CORRECT = "G"
    TRADE_CANCEL = "H"
    ORDER_STATUS = "I"


class Side(Enum):
    """FIX Side (Tag 54) values"""

    BUY = "1"
    SELL = "2"


class OrdType(Enum):
    """FIX OrdType (Tag 40) values"""

    MARKET = "1"
    LIMIT = "2"
    STOP = "3"
    STOP_LIMIT = "4"


@dataclass
class Order:
    """Represents a FIX order with full lifecycle tracking"""

    clord_id: str  # Tag 11: Client Order ID
    symbol: str  # Tag 55: Symbol ID
    side: Side  # Tag 54: Buy/Sell
    ord_type: OrdType  # Tag 40: Market/Limit
    quantity: float  # Tag 38: Order quantity
    price: float | None = None  # Tag 44: Limit price (if applicable)

    # Broker-assigned fields (from ExecutionReport)
    order_id: str | None = None  # Tag 37: Broker order ID
    status: OrderStatus = OrderStatus.PENDING_NEW  # Tag 39: Order status
    filled_qty: float = 0.0  # Tag 14: Cumulative filled quantity
    avg_price: float = 0.0  # Tag 6: Average fill price
    last_qty: float = 0.0  # Tag 32: Last fill quantity
    last_px: float = 0.0  # Tag 31: Last fill price

    # Timestamps
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    filled_at: datetime | None = None

    # Error tracking
    reject_reason: str | None = None  # Tag 103/58: Reject reason

    def is_terminal(self) -> bool:
        """Check if order reached terminal state"""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    def is_active(self) -> bool:
        """Check if order is actively working"""
        return self.status in (
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_NEW,
        )

    def remaining_qty(self) -> float:
        """Calculate unfilled quantity"""
        return max(0.0, self.quantity - self.filled_qty)


@dataclass
class Position:
    """Represents current position for a symbol"""

    symbol: str
    long_qty: float = 0.0  # Tag 704: Long position quantity
    short_qty: float = 0.0  # Tag 705: Short position quantity
    net_qty: float = 0.0  # Calculated: long - short

    # Position tracking (for hedged accounts)
    pos_maint_rpt_id: str | None = None  # Tag 721: Position maintenance report ID

    # Timestamps
    updated_at: datetime = field(default_factory=utc_now)

    def update_from_report(self, long_qty: float, short_qty: float, pos_id: str | None = None):
        """Update position from PositionReport (35=AP)"""
        self.long_qty = long_qty
        self.short_qty = short_qty
        self.net_qty = long_qty - short_qty
        self.pos_maint_rpt_id = pos_id
        self.updated_at = utc_now()


class TradeManager:
    """
    Centralized FIX Protocol order and position management.

    Responsibilities:
    1. Order submission via NewOrderSingle (35=D)
    2. ExecutionReport processing (35=8) with ExecType routing
    3. Order state tracking and reconciliation
    4. Position tracking via PositionReport (35=AP)
    5. Order modification (35=G) and cancellation (35=F)

    Usage:
        manager = TradeManager(
            session_id=trade_session_id,
            symbol_id=1,
            on_fill_callback=handle_fill
        )

        # Submit market order
        order = manager.submit_market_order(Side.BUY, quantity=0.01)

        # Process ExecutionReport
        manager.on_execution_report(fix_message)

        # Get current position
        pos = manager.get_position()
    """

    def __init__(
        self,
        session_id: fix.SessionID,
        symbol_id: int,
        on_fill_callback: Callable[[Order], None] | None = None,
        on_reject_callback: Callable[[Order], None] | None = None,
        max_pending_orders: int = 10,
    ):
        """
        Initialize TradeManager.

        Args:
            session_id: FIX session ID for TRADE session
            symbol_id: Numeric symbol ID (Tag 55)
            on_fill_callback: Called when order fills (ExecType=F)
            on_reject_callback: Called when order rejected (ExecType=8)
            max_pending_orders: Maximum concurrent pending orders
        """
        self.session_id = session_id
        self.symbol_id = str(symbol_id)
        self.on_fill_callback = on_fill_callback
        self.on_reject_callback = on_reject_callback
        self.max_pending_orders = max_pending_orders

        # Order tracking
        self.orders: dict[str, Order] = {}  # clord_id -> Order
        self.broker_orders: dict[str, str] = {}  # order_id -> clord_id
        self.clord_counter = 0

        # Position tracking
        self.position = Position(symbol=self.symbol_id)

        # FIX P1-8: Position request tracking with retry logic
        self.pos_req_id: str | None = None
        self.pending_position_requests: dict[str, dict] = {}  # req_id -> metadata
        self.position_request_timeout = 5.0  # seconds
        self.position_request_max_retries = 3

        # Execution history (for debugging)
        self.exec_reports: deque[dict] = deque(maxlen=100)

        LOG.info(
            "[TRADEMGR] Initialized for symbol=%s session=%s",
            self.symbol_id,
            session_id,
        )

    def _generate_clord_id(self) -> str:
        """Generate unique client order ID"""
        self.clord_counter += 1
        timestamp = int(time.time())
        return f"cl_{timestamp}_{self.clord_counter}"

    def submit_market_order(
        self,
        side: Side,
        quantity: float,
        tag_prefix: str | None = None,
    ) -> Order | None:
        """
        Submit market order via NewOrderSingle (35=D).

        Args:
            side: Side.BUY or Side.SELL
            quantity: Order quantity
            tag_prefix: Optional prefix for ClOrdID (for tracking)

        Returns:
            Order object if submitted, None if failed
        """
        if len([o for o in self.orders.values() if o.is_active()]) >= self.max_pending_orders:
            LOG.warning("[TRADEMGR] Max pending orders reached (%d)", self.max_pending_orders)
            return None

        # Create order object
        clord_id = f"{tag_prefix}_{self._generate_clord_id()}" if tag_prefix else self._generate_clord_id()
        order = Order(
            clord_id=clord_id,
            symbol=self.symbol_id,
            side=side,
            ord_type=OrdType.MARKET,
            quantity=quantity,
        )

        # Build FIX message
        msg = fix44.NewOrderSingle()
        msg.setField(fix.ClOrdID(clord_id))
        msg.setField(fix.Symbol(self.symbol_id))
        msg.setField(fix.Side(side.value))
        msg.setField(fix.TransactTime(utc_ts_ms()))
        msg.setField(fix.OrdType(OrdType.MARKET.value))
        msg.setField(fix.OrderQty(quantity))

        try:
            fix.Session.sendToTarget(msg, self.session_id)
            self.orders[clord_id] = order
            LOG.info(
                "[TRADEMGR] ✓ Submitted MKT order: %s %s qty=%.6f clOrdID=%s",
                side.name,
                self.symbol_id,
                quantity,
                clord_id,
            )
            return order
        except Exception as e:
            LOG.error("[TRADEMGR] ✗ Failed to submit order: %s", e)
            return None

    def submit_limit_order(
        self,
        side: Side,
        quantity: float,
        price: float,
        tag_prefix: str | None = None,
    ) -> Order | None:
        """
        Submit limit order via NewOrderSingle (35=D).

        Args:
            side: Side.BUY or Side.SELL
            quantity: Order quantity
            price: Limit price
            tag_prefix: Optional prefix for ClOrdID

        Returns:
            Order object if submitted, None if failed
        """
        if len([o for o in self.orders.values() if o.is_active()]) >= self.max_pending_orders:
            LOG.warning("[TRADEMGR] Max pending orders reached (%d)", self.max_pending_orders)
            return None

        clord_id = f"{tag_prefix}_{self._generate_clord_id()}" if tag_prefix else self._generate_clord_id()
        order = Order(
            clord_id=clord_id,
            symbol=self.symbol_id,
            side=side,
            ord_type=OrdType.LIMIT,
            quantity=quantity,
            price=price,
        )

        msg = fix44.NewOrderSingle()
        msg.setField(fix.ClOrdID(clord_id))
        msg.setField(fix.Symbol(self.symbol_id))
        msg.setField(fix.Side(side.value))
        msg.setField(fix.TransactTime(utc_ts_ms()))
        msg.setField(fix.OrdType(OrdType.LIMIT.value))
        msg.setField(fix.OrderQty(quantity))
        msg.setField(fix.Price(price))

        try:
            fix.Session.sendToTarget(msg, self.session_id)
            self.orders[clord_id] = order
            LOG.info(
                "[TRADEMGR] ✓ Submitted LMT order: %s %s qty=%.6f @%.5f clOrdID=%s",
                side.name,
                self.symbol_id,
                quantity,
                price,
                clord_id,
            )
            return order
        except Exception as e:
            LOG.error("[TRADEMGR] ✗ Failed to submit limit order: %s", e)
            return None

    def cancel_order(self, clord_id: str) -> bool:
        """
        Cancel order via OrderCancelRequest (35=F).

        Args:
            clord_id: Client order ID to cancel

        Returns:
            True if cancel request sent, False otherwise
        """
        order = self.orders.get(clord_id)
        if not order:
            LOG.warning("[TRADEMGR] Cannot cancel - order not found: %s", clord_id)
            return False

        if not order.is_active():
            LOG.warning("[TRADEMGR] Cannot cancel - order not active: %s (status=%s)", clord_id, order.status)
            return False

        # Generate new ClOrdID for cancel request
        orig_clord_id = clord_id
        cancel_clord_id = f"cancel_{self._generate_clord_id()}"

        msg = fix44.OrderCancelRequest()
        msg.setField(fix.OrigClOrdID(orig_clord_id))
        msg.setField(fix.ClOrdID(cancel_clord_id))
        msg.setField(fix.Symbol(self.symbol_id))
        msg.setField(fix.Side(order.side.value))
        msg.setField(fix.TransactTime(utc_ts_ms()))

        if order.order_id:
            msg.setField(fix.OrderID(order.order_id))

        try:
            fix.Session.sendToTarget(msg, self.session_id)
            order.status = OrderStatus.PENDING_CANCEL
            order.updated_at = utc_now()
            LOG.info("[TRADEMGR] ✓ Sent cancel request: %s", orig_clord_id)
            return True
        except Exception as e:
            LOG.error("[TRADEMGR] ✗ Failed to cancel order: %s", e)
            return False

    def modify_order(self, clord_id: str, new_price: float, new_qty: float | None = None) -> bool:
        """
        Modify order via OrderCancelReplaceRequest (35=G).

        Args:
            clord_id: Original client order ID
            new_price: New limit price
            new_qty: New quantity (optional, keeps original if None)

        Returns:
            True if modify request sent, False otherwise
        """
        order = self.orders.get(clord_id)
        if not order:
            LOG.warning("[TRADEMGR] Cannot modify - order not found: %s", clord_id)
            return False

        if not order.is_active():
            LOG.warning("[TRADEMGR] Cannot modify - order not active: %s", clord_id)
            return False

        if order.ord_type != OrdType.LIMIT:
            LOG.warning("[TRADEMGR] Cannot modify - not a limit order: %s", clord_id)
            return False

        orig_clord_id = clord_id
        replace_clord_id = f"replace_{self._generate_clord_id()}"
        quantity = new_qty if new_qty is not None else order.quantity

        msg = fix44.OrderCancelReplaceRequest()
        msg.setField(fix.OrigClOrdID(orig_clord_id))
        msg.setField(fix.ClOrdID(replace_clord_id))
        msg.setField(fix.Symbol(self.symbol_id))
        msg.setField(fix.Side(order.side.value))
        msg.setField(fix.TransactTime(utc_ts_ms()))
        msg.setField(fix.OrdType(OrdType.LIMIT.value))
        msg.setField(fix.OrderQty(quantity))
        msg.setField(fix.Price(new_price))

        if order.order_id:
            msg.setField(fix.OrderID(order.order_id))

        try:
            fix.Session.sendToTarget(msg, self.session_id)
            order.status = OrderStatus.PENDING_REPLACE
            order.updated_at = utc_now()
            LOG.info(
                "[TRADEMGR] ✓ Sent replace request: %s -> price=%.5f qty=%.6f",
                orig_clord_id,
                new_price,
                quantity,
            )
            return True
        except Exception as e:
            LOG.error("[TRADEMGR] ✗ Failed to modify order: %s", e)
            return False

    def on_execution_report(self, msg: fix.Message):
        """
        Process ExecutionReport (35=8) from FIX session.

        Routes to appropriate handler based on ExecType (Tag 150):
        - ExecType=0 (New): Order accepted
        - ExecType=F (Fill/Trade): Order filled
        - ExecType=4 (Canceled): Order canceled
        - ExecType=5 (Replaced): Order modified
        - ExecType=8 (Rejected): Order rejected
        - ExecType=I (OrderStatus): Status update
        """
        try:
            # Extract ClOrdID
            clord_field = fix.ClOrdID()
            if not msg.isSetField(clord_field):
                LOG.warning("[TRADEMGR] ExecutionReport missing ClOrdID")
                return
            msg.getField(clord_field)
            clord_id = clord_field.getValue()

            # Extract ExecType
            exec_type_field = fix.ExecType()
            if not msg.isSetField(exec_type_field):
                LOG.warning("[TRADEMGR] ExecutionReport missing ExecType for %s", clord_id)
                return
            msg.getField(exec_type_field)
            exec_type_str = exec_type_field.getValue()

            # Extract OrdStatus
            ord_status_field = fix.OrdStatus()
            if not msg.isSetField(ord_status_field):
                LOG.warning("[TRADEMGR] ExecutionReport missing OrdStatus for %s", clord_id)
                return
            msg.getField(ord_status_field)
            ord_status_str = ord_status_field.getValue()

            # Get or create order
            order = self.orders.get(clord_id)
            if not order:
                # Handle unsolicited ExecutionReport (from order recovery)
                LOG.warning("[TRADEMGR] Received ExecutionReport for unknown order: %s", clord_id)
                # Could create order from ExecutionReport fields if needed
                return

            # Update order fields
            order.status = OrderStatus(ord_status_str)
            order.updated_at = utc_now()

            # Extract OrderID (broker-assigned)
            order_id_field = fix.OrderID()
            if msg.isSetField(order_id_field):
                msg.getField(order_id_field)
                order.order_id = order_id_field.getValue()
                self.broker_orders[order.order_id] = clord_id

            # Extract fill quantities
            cum_qty_field = fix.CumQty()
            if msg.isSetField(cum_qty_field):
                msg.getField(cum_qty_field)
                order.filled_qty = float(cum_qty_field.getValue())

            avg_px_field = fix.AvgPx()
            if msg.isSetField(avg_px_field):
                msg.getField(avg_px_field)
                order.avg_price = float(avg_px_field.getValue())

            last_qty_field = fix.LastQty()
            if msg.isSetField(last_qty_field):
                msg.getField(last_qty_field)
                order.last_qty = float(last_qty_field.getValue())

            last_px_field = fix.LastPx()
            if msg.isSetField(last_px_field):
                msg.getField(last_px_field)
                order.last_px = float(last_px_field.getValue())

            # Store execution report for debugging
            self.exec_reports.append(
                {
                    "timestamp": utc_now(),
                    "clord_id": clord_id,
                    "exec_type": exec_type_str,
                    "ord_status": ord_status_str,
                    "filled_qty": order.filled_qty,
                    "avg_price": order.avg_price,
                }
            )

            # Route based on ExecType
            if exec_type_str == ExecType.NEW.value:
                self._handle_new(order)
            elif exec_type_str == ExecType.TRADE.value:
                self._handle_fill(order)
            elif exec_type_str == ExecType.CANCELED.value:
                self._handle_canceled(order)
            elif exec_type_str == ExecType.REJECTED.value:
                self._handle_rejected(order, msg)
            elif exec_type_str == ExecType.REPLACED.value:
                self._handle_replaced(order)
            elif exec_type_str == ExecType.ORDER_STATUS.value:
                self._handle_status_update(order)
            else:
                LOG.info(
                    "[TRADEMGR] ExecutionReport: %s ExecType=%s OrdStatus=%s",
                    clord_id,
                    exec_type_str,
                    ord_status_str,
                )

        except Exception as e:
            LOG.error("[TRADEMGR] Error processing ExecutionReport: %s", e, exc_info=True)

    def _handle_new(self, order: Order):
        """Handle ExecType=0 (New) - Order accepted by broker"""
        LOG.info(
            "[TRADEMGR] ✓ Order accepted: %s %s qty=%.6f OrderID=%s",
            order.side.name,
            order.symbol,
            order.quantity,
            order.order_id or "N/A",
        )

    def _handle_fill(self, order: Order):
        """Handle ExecType=F (Fill) - Order filled (full or partial)"""
        if order.status == OrderStatus.FILLED:
            order.filled_at = utc_now()
            LOG.info(
                "[TRADEMGR] ✓✓ Order FILLED: %s qty=%.6f @%.5f (ClOrdID=%s)",
                order.side.name,
                order.filled_qty,
                order.avg_price,
                order.clord_id,
            )
        else:
            LOG.info(
                "[TRADEMGR] ◐ Order PARTIAL FILL: %s filled=%.6f/%.6f @%.5f",
                order.side.name,
                order.filled_qty,
                order.quantity,
                order.avg_price,
            )

        # Trigger fill callback
        if self.on_fill_callback:
            try:
                self.on_fill_callback(order)
            except Exception as e:
                LOG.error("[TRADEMGR] Error in fill callback: %s", e, exc_info=True)

    def _handle_canceled(self, order: Order):
        """Handle ExecType=4 (Canceled) - Order canceled"""
        LOG.info("[TRADEMGR] Order canceled: %s", order.clord_id)

    def _handle_rejected(self, order: Order, msg: fix.Message):
        """Handle ExecType=8 (Rejected) - Order rejected"""
        # Extract reject reason
        text_field = fix.Text()
        if msg.isSetField(text_field):
            msg.getField(text_field)
            order.reject_reason = text_field.getValue()

        LOG.warning(
            "[TRADEMGR] ✗ Order REJECTED: %s - %s",
            order.clord_id,
            order.reject_reason or "No reason provided",
        )

        # Trigger reject callback
        if self.on_reject_callback:
            try:
                self.on_reject_callback(order)
            except Exception as e:
                LOG.error("[TRADEMGR] Error in reject callback: %s", e, exc_info=True)

    def _handle_replaced(self, order: Order):
        """Handle ExecType=5 (Replaced) - Order modified successfully"""
        LOG.info("[TRADEMGR] ✓ Order replaced: %s", order.clord_id)

    def _handle_status_update(self, order: Order):
        """Handle ExecType=I (OrderStatus) - Status update from OrderStatusRequest"""
        LOG.debug(
            "[TRADEMGR] Status update: %s status=%s filled=%.6f/%.6f",
            order.clord_id,
            order.status.value,
            order.filled_qty,
            order.quantity,
        )

    def request_positions(self, retry_count: int = 0):
        """
        Request current positions via RequestForPositions (35=AN).

        FIX P1-8: Added retry logic with timeout tracking

        Response will be PositionReport (35=AP) handled by on_position_report().

        Args:
            retry_count: Current retry attempt number
        """
        if not self.session_id:
            LOG.warning("[TRADEMGR] Cannot request positions - no session")
            return

        req_id = f"pos_{uuid.uuid4().hex[:10]}"
        self.pos_req_id = req_id

        # Track request for timeout/retry
        self.pending_position_requests[req_id] = {
            "sent_at": utc_now(),
            "retry_count": retry_count,
            "timeout": self.position_request_timeout,
        }

        msg = fix44.RequestForPositions()
        msg.setField(fix.PosReqID(req_id))

        try:
            fix.Session.sendToTarget(msg, self.session_id)
            LOG.info("[TRADEMGR] ✓ Requested positions (PosReqID=%s, retry=%d)", req_id, retry_count)

            # Schedule timeout check
            import threading

            def check_timeout():
                time.sleep(self.position_request_timeout)
                self._check_position_request_timeout(req_id)

            threading.Thread(target=check_timeout, daemon=True).start()

        except Exception as e:
            LOG.error("[TRADEMGR] ✗ Failed to request positions: %s", e)
            # Remove from pending
            self.pending_position_requests.pop(req_id, None)

    def _check_position_request_timeout(self, req_id: str):
        """
        FIX P1-8: Check if position request timed out and retry if needed.

        Args:
            req_id: Position request ID to check
        """
        request_info = self.pending_position_requests.get(req_id)
        if not request_info:
            # Already received response
            return

        elapsed = (utc_now() - request_info["sent_at"]).total_seconds()
        if elapsed < request_info["timeout"]:
            # Not timed out yet
            return

        retry_count = request_info["retry_count"]
        if retry_count < self.position_request_max_retries:
            LOG.warning(
                "[TRADEMGR] Position request timeout (%.1fs) - retrying (%d/%d)",
                elapsed,
                retry_count + 1,
                self.position_request_max_retries,
            )
            # Remove old request
            self.pending_position_requests.pop(req_id, None)
            # Retry
            self.request_positions(retry_count=retry_count + 1)
        else:
            LOG.error(
                "[TRADEMGR] Position request failed after %d retries - giving up",
                self.position_request_max_retries,
            )
            self.pending_position_requests.pop(req_id, None)

    def on_position_report(self, msg: fix.Message):
        """
        Process PositionReport (35=AP) from FIX session.

        Updates internal position tracking with:
        - Tag 704 (LongQty): Long position quantity
        - Tag 705 (ShortQty): Short position quantity
        - Tag 721 (PosMaintRptID): Position ID for hedged accounts
        """
        try:
            # FIX P1-8: Clear pending position request on successful response
            pos_req_id_field = fix.PosReqID()
            if msg.isSetField(pos_req_id_field):
                msg.getField(pos_req_id_field)
                req_id = pos_req_id_field.getValue()
                if req_id in self.pending_position_requests:
                    elapsed = (utc_now() - self.pending_position_requests[req_id]["sent_at"]).total_seconds()
                    LOG.debug("[TRADEMGR] Position response received (%.3fs latency)", elapsed)
                    self.pending_position_requests.pop(req_id, None)

            # Verify symbol
            symbol_field = fix.Symbol()
            if msg.isSetField(symbol_field):
                msg.getField(symbol_field)
                if symbol_field.getValue() != self.symbol_id:
                    return  # Not our symbol

            # Extract position quantities
            long_qty = 0.0
            short_qty = 0.0

            long_qty_field = fix.StringField(704)  # LongQty
            if msg.isSetField(long_qty_field):
                msg.getField(long_qty_field)
                long_qty = float(long_qty_field.getValue())

            short_qty_field = fix.StringField(705)  # ShortQty
            if msg.isSetField(short_qty_field):
                msg.getField(short_qty_field)
                short_qty = float(short_qty_field.getValue())

            # Extract position ID (hedged accounts)
            pos_id = None
            pos_id_field = fix.StringField(721)  # PosMaintRptID
            if msg.isSetField(pos_id_field):
                msg.getField(pos_id_field)
                pos_id = pos_id_field.getValue()

            # Update position
            self.position.update_from_report(long_qty, short_qty, pos_id)

            LOG.info(
                "[TRADEMGR] Position updated: long=%.6f short=%.6f net=%.6f",
                long_qty,
                short_qty,
                self.position.net_qty,
            )

        except Exception as e:
            LOG.error("[TRADEMGR] Error processing PositionReport: %s", e, exc_info=True)

    def get_position(self) -> Position:
        """Get current position"""
        return self.position

    def get_position_direction(self, min_qty: float = 0.0) -> int:
        """
        Get position direction as integer.

        Args:
            min_qty: Minimum quantity threshold to consider non-flat

        Returns:
            1 for long, -1 for short, 0 for flat
        """
        if abs(self.position.net_qty) < min_qty:
            return 0
        return 1 if self.position.net_qty > 0 else -1

    def get_order(self, clord_id: str) -> Order | None:
        """Get order by client order ID"""
        return self.orders.get(clord_id)

    def get_active_orders(self) -> list[Order]:
        """Get all active orders"""
        return [o for o in self.orders.values() if o.is_active()]

    def get_filled_orders(self) -> list[Order]:
        """Get all filled orders"""
        return [o for o in self.orders.values() if o.status == OrderStatus.FILLED]

    def get_statistics(self) -> dict:
        """Get order statistics"""
        total_orders = len(self.orders)
        filled = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        rejected = len([o for o in self.orders.values() if o.status == OrderStatus.REJECTED])
        canceled = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELED])
        active = len([o for o in self.orders.values() if o.is_active()])

        return {
            "total_orders": total_orders,
            "filled": filled,
            "rejected": rejected,
            "canceled": canceled,
            "active": active,
            "fill_rate": filled / total_orders if total_orders > 0 else 0.0,
            "reject_rate": rejected / total_orders if total_orders > 0 else 0.0,
            "position_net_qty": self.position.net_qty,
            "position_long_qty": self.position.long_qty,
            "position_short_qty": self.position.short_qty,
        }

    def cleanup_old_orders(self, max_age_hours: int = 24):
        """
        Clean up old terminal orders from memory.

        Args:
            max_age_hours: Remove terminal orders older than this
        """
        cutoff = utc_now().timestamp() - (max_age_hours * 3600)
        to_remove = [
            clord_id
            for clord_id, order in self.orders.items()
            if order.is_terminal() and order.updated_at.timestamp() < cutoff
        ]

        for clord_id in to_remove:
            order = self.orders.pop(clord_id)
            if order.order_id and order.order_id in self.broker_orders:
                del self.broker_orders[order.order_id]

        if to_remove:
            LOG.info("[TRADEMGR] Cleaned up %d old orders", len(to_remove))


if __name__ == "__main__":
    # Example usage (requires active FIX session)
    print("TradeManager - FIX Protocol Order Management")
    print("\nFeatures:")
    print("  ✓ Order submission (Market/Limit)")
    print("  ✓ ExecutionReport processing with ExecType routing")
    print("  ✓ Order state tracking (NEW/FILLED/CANCELED/REJECTED)")
    print("  ✓ Position reconciliation via PositionReport")
    print("  ✓ Order modification and cancellation")
    print("\nOrder Lifecycle:")
    print("  1. submit_market_order() -> NewOrderSingle (35=D)")
    print("  2. on_execution_report() <- ExecutionReport (35=8) ExecType=0 (NEW)")
    print("  3. on_execution_report() <- ExecutionReport (35=8) ExecType=F (FILL)")
    print("  4. on_fill_callback() triggered")
    print("  5. request_positions() -> RequestForPositions (35=AN)")
    print("  6. on_position_report() <- PositionReport (35=AP)")
