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
from collections.abc import Callable
from enum import Enum

try:
    import quickfix as fix
    import quickfix44 as fix44
except ImportError:
    def _make_stub(name: str):
        """Return a lightweight stub class for FIX field/message types."""
        return type(name, (), {"__init__": lambda self, *a, **kw: None,
                               "__repr__": lambda self: f"<{name}>",
                               "setField": lambda self, *a: None,
                               "sendToTarget": staticmethod(lambda *a: None)})

    class fix:  # type: ignore[no-redef]
        """Stub namespace – quickfix C-extension not installed."""
        def __getattr__(self, name):  # noqa: D105 – instance
            return _make_stub(name)
        __class_getitem__ = classmethod(lambda cls, item: None)

        Application = _make_stub("Application")  # NOSONAR
        Session = _make_stub("Session")  # NOSONAR
        SessionID = _make_stub("SessionID")  # NOSONAR
        Message = _make_stub("Message")  # NOSONAR

    for _attr in ("ClOrdID", "Symbol", "Side", "TransactTime", "OrdType",
                  "OrderQty", "Price", "OrigClOrdID", "Text", "ExecType",
                  "OrdStatus", "CumQty", "AvgPx", "LeavesQty"):
        setattr(fix, _attr, _make_stub(_attr))

    class fix44:  # type: ignore[no-redef]
        NewOrderSingle = _make_stub("NewOrderSingle")  # NOSONAR
        OrderCancelRequest = _make_stub("OrderCancelRequest")  # NOSONAR

from src.monitoring.trade_audit_logger import get_trade_audit_logger
from src.utils.safe_math import SafeMath
from src.utils.safe_utils import utc_now

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


class Order:
    """Represents a FIX order with full lifecycle tracking, using SafeMath for precision"""

    def __init__(self, clord_id, symbol, side, ord_type, quantity, price=None, instrument_digits=2):  # noqa: PLR0913
        from src.utils.safe_math import SafeMath  # noqa: PLC0415

        self.clord_id = clord_id
        self.symbol = symbol
        self.side = side
        self.ord_type = ord_type
        self.instrument_digits = instrument_digits
        self.quantity = SafeMath.to_decimal(quantity, instrument_digits)
        self.price = SafeMath.to_decimal(price, instrument_digits) if price is not None else None
        self.order_id = None
        self.position_ticket = None
        self.status = OrderStatus.PENDING_NEW
        self.filled_qty = SafeMath.to_decimal(0.0, instrument_digits)
        self.avg_price = SafeMath.to_decimal(0.0, instrument_digits)
        self.last_qty = SafeMath.to_decimal(0.0, instrument_digits)
        self.last_px = SafeMath.to_decimal(0.0, instrument_digits)
        self.created_at = utc_now()
        self.updated_at = utc_now()
        self.filled_at = None
        self.reject_reason = None

    def is_terminal(self):
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    def is_active(self):
        return self.status in (
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_NEW,
        )

    def remaining_qty(self):
        from src.utils.safe_math import SafeMath  # noqa: PLC0415

        return SafeMath.quantize(
            max(SafeMath.to_decimal(0.0, self.instrument_digits), self.quantity - self.filled_qty),
            self.instrument_digits,
        )

    def to_dict(self):
        from src.utils.safe_math import SafeMath  # noqa: PLC0415

        return {
            "clord_id": self.clord_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "ord_type": self.ord_type.value,
            "quantity": str(SafeMath.quantize(self.quantity, self.instrument_digits)),
            "price": str(SafeMath.quantize(self.price, self.instrument_digits)) if self.price is not None else None,
            "order_id": self.order_id,
            "position_ticket": self.position_ticket,
            "status": self.status.value,
            "filled_qty": str(SafeMath.quantize(self.filled_qty, self.instrument_digits)),
            "avg_price": str(SafeMath.quantize(self.avg_price, self.instrument_digits)),
            "last_qty": str(SafeMath.quantize(self.last_qty, self.instrument_digits)),
            "last_px": str(SafeMath.quantize(self.last_px, self.instrument_digits)),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "reject_reason": self.reject_reason,
        }

    @classmethod
    def from_dict(cls, data, instrument_digits=2):
        from src.utils.safe_math import SafeMath  # noqa: PLC0415

        order = cls(
            clord_id=data.get("clord_id", ""),
            symbol=data.get("symbol", ""),
            side=Side(data.get("side")),
            ord_type=OrdType(data.get("ord_type")),
            quantity=SafeMath.to_decimal(data.get("quantity", 0.0), instrument_digits),
            price=(
                SafeMath.to_decimal(data.get("price", 0.0), instrument_digits)
                if data.get("price") is not None
                else None
            ),
            instrument_digits=instrument_digits,
        )
        order.order_id = data.get("order_id")
        order.position_ticket = data.get("position_ticket")
        order.status = OrderStatus(data.get("status", OrderStatus.PENDING_NEW.value))
        order.filled_qty = SafeMath.to_decimal(str(data.get("filled_qty", 0.0)), instrument_digits)
        order.avg_price = SafeMath.to_decimal(str(data.get("avg_price", 0.0)), instrument_digits)
        order.last_qty = SafeMath.to_decimal(str(data.get("last_qty", 0.0)), instrument_digits)
        order.last_px = SafeMath.to_decimal(str(data.get("last_px", 0.0)), instrument_digits)
        from datetime import datetime  # noqa: PLC0415

        order.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else utc_now()
        order.updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else utc_now()
        order.filled_at = datetime.fromisoformat(data["filled_at"]) if data.get("filled_at") else None
        order.reject_reason = data.get("reject_reason")
        return order


class Position:
    """Represents current position for a symbol, using SafeMath for precision"""

    def __init__(  # noqa: PLR0913
        self,
        symbol,
        long_qty=0.0,
        short_qty=0.0,
        net_qty=0.0,
        pos_maint_rpt_id=None,
        updated_at=None,
        instrument_digits=2,
    ):
        from src.utils.safe_math import SafeMath  # noqa: PLC0415

        self.symbol = symbol
        self.instrument_digits = instrument_digits
        self.long_qty = SafeMath.to_decimal(long_qty, instrument_digits)
        self.short_qty = SafeMath.to_decimal(short_qty, instrument_digits)
        self.net_qty = SafeMath.to_decimal(net_qty, instrument_digits)
        self.pos_maint_rpt_id = pos_maint_rpt_id
        self.updated_at = updated_at if updated_at else utc_now()

    def update_from_report(self, long_qty, short_qty, pos_id=None):
        from src.utils.safe_math import SafeMath  # noqa: PLC0415

        self.long_qty = SafeMath.to_decimal(long_qty, self.instrument_digits)
        self.short_qty = SafeMath.to_decimal(short_qty, self.instrument_digits)
        self.net_qty = SafeMath.quantize(self.long_qty - self.short_qty, self.instrument_digits)
        self.pos_maint_rpt_id = pos_id
        self.updated_at = utc_now()

    def update_from_fill(self, side, filled_qty, _avg_price):
        from src.utils.safe_math import SafeMath  # noqa: PLC0415

        filled_qty = SafeMath.to_decimal(filled_qty, self.instrument_digits)
        if side == Side.BUY:
            self.long_qty += filled_qty
        else:
            self.short_qty += filled_qty
        self.net_qty = SafeMath.quantize(self.long_qty - self.short_qty, self.instrument_digits)
        self.updated_at = utc_now()
        LOG.info(
            "[POSITION] Updated from fill: %s %s → net=%s (long=%s, short=%s)",
            side.name,
            str(filled_qty),
            str(self.net_qty),
            str(self.long_qty),
            str(self.short_qty),
        )

    def seed(self, net_qty, _entry_price=0.0):
        from src.utils.safe_math import SafeMath  # noqa: PLC0415

        net_qty = SafeMath.to_decimal(net_qty, self.instrument_digits)
        if net_qty > 0:
            self.long_qty = net_qty
            self.short_qty = SafeMath.to_decimal(0.0, self.instrument_digits)
        elif net_qty < 0:
            self.long_qty = SafeMath.to_decimal(0.0, self.instrument_digits)
            self.short_qty = abs(net_qty)
        else:
            self.long_qty = SafeMath.to_decimal(0.0, self.instrument_digits)
            self.short_qty = SafeMath.to_decimal(0.0, self.instrument_digits)
        self.net_qty = net_qty
        self.updated_at = utc_now()
        LOG.info(
            "[POSITION] 🌱 Seeded position: net=%s (long=%s, short=%s)",
            str(self.net_qty),
            str(self.long_qty),
            str(self.short_qty),
        )

    def to_dict(self):
        from src.utils.safe_math import SafeMath  # noqa: PLC0415

        return {
            "symbol": self.symbol,
            "long_qty": str(SafeMath.quantize(self.long_qty, self.instrument_digits)),
            "short_qty": str(SafeMath.quantize(self.short_qty, self.instrument_digits)),
            "net_qty": str(SafeMath.quantize(self.net_qty, self.instrument_digits)),
            "pos_maint_rpt_id": self.pos_maint_rpt_id,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data, instrument_digits=2):
        from src.utils.safe_math import SafeMath  # noqa: PLC0415

        updated_at = None
        if data.get("updated_at"):
            from datetime import datetime  # noqa: PLC0415

            try:
                updated_at = datetime.fromisoformat(data["updated_at"])
            except (ValueError, TypeError):
                updated_at = utc_now()
        return cls(
            symbol=data.get("symbol", ""),
            long_qty=SafeMath.to_decimal(data.get("long_qty", 0.0), instrument_digits),
            short_qty=SafeMath.to_decimal(data.get("short_qty", 0.0), instrument_digits),
            net_qty=SafeMath.to_decimal(data.get("net_qty", 0.0), instrument_digits),
            pos_maint_rpt_id=data.get("pos_maint_rpt_id"),
            updated_at=updated_at,
            instrument_digits=instrument_digits,
        )


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

    def __init__(  # noqa: PLR0913
        self,
        session_id: fix.SessionID,
        symbol_id: int,
        on_fill_callback: Callable[[Order], None] | None = None,
        on_reject_callback: Callable[[Order], None] | None = None,
        max_pending_orders: int = 10,
        paper_mode: bool = False,
        get_price_callback: Callable[[], tuple[float, float]] | None = None,
    ):
        """
        Initialize TradeManager.

        Args:
            session_id: FIX session ID for TRADE session
            symbol_id: Numeric symbol ID (Tag 55)
            on_fill_callback: Called when order fills (ExecType=F)
            on_reject_callback: Called when order rejected (ExecType=8)
            max_pending_orders: Maximum concurrent pending orders
            paper_mode: If True, simulate fills locally when broker doesn't respond
            get_price_callback: Callback returning (bid, ask) for paper fill price
        """
        self.session_id = session_id
        self.symbol_id = str(symbol_id)
        self.on_fill_callback = on_fill_callback
        self.on_reject_callback = on_reject_callback
        self.max_pending_orders = max_pending_orders

        # PAPER MODE: Simulate fills when broker doesn't respond
        self.paper_mode = paper_mode
        self.get_price_callback = get_price_callback
        self.paper_fill_timeout = 2.0  # seconds before simulating fill
        self.paper_fill_counter = 0  # For unique paper ticket IDs

        # Order tracking
        self.orders: dict[str, Order] = {}  # clord_id -> Order
        self.broker_orders: dict[str, str] = {}  # order_id -> clord_id
        self.clord_counter = 0

        # Position tracking
        self.position = Position(symbol=self.symbol_id)

        # Audit logging
        self.audit = get_trade_audit_logger()

        # FIX P1-8: Position request tracking with retry logic
        self.pos_req_id: str | None = None
        self.pending_position_requests: dict[str, dict] = {}  # req_id -> metadata
        self.position_request_timeout = 5.0  # seconds
        self.position_request_max_retries = 3

        # P0 FIX: Order timeout tracking (prevent "lost in flight" orders)
        self.pending_orders: dict[str, dict] = {}  # clord_id -> {submitted_at, retries}
        self.order_ack_timeout = 10.0  # seconds before querying status
        self.order_ack_max_retries = 3  # max status queries

        # Execution history (for debugging)
        self.exec_reports: deque[dict] = deque(maxlen=100)

        LOG.info(
            "[TRADEMGR] Initialized for symbol=%s session=%s paper_mode=%s",
            self.symbol_id,
            session_id,
            paper_mode,
        )

    def seed_position(self, net_qty: float, _entry_price: float = 0.0):
        """
        Seed TradeManager with externally-known position.

        Use this when bot starts with positions already open at broker.

        Args:
            net_qty: Net position (positive=LONG, negative=SHORT)
            entry_price: Optional entry price for tracking
        """
        self.position.seed(net_qty, _entry_price)
        if net_qty > 0:
            _direction = "LONG"
        elif net_qty < 0:
            _direction = "SHORT"
        else:
            _direction = "FLAT"
        LOG.info("[TRADEMGR] 🌱 Position seeded: net=%.6f direction=%s", net_qty, _direction)

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
        position_ticket: str | None = None,  # Added back for hedging mode
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
        msg.setField(fix.TransactTime())
        msg.setField(fix.OrdType(OrdType.MARKET.value))
        msg.setField(fix.OrderQty(round(float(quantity), 2)))

        # HEDGING MODE FIX: Specify position to close via PosMaintRptID (tag 721)
        if position_ticket:
            msg.setField(721, position_ticket)  # PosMaintRptID
            LOG.debug("[TRADEMGR] Set PosMaintRptID=%s to close specific position", position_ticket)

        try:
            fix.Session.sendToTarget(msg, self.session_id)
            self.orders[clord_id] = order

            # P0 FIX: Track order submission time for timeout detection
            self.pending_orders[clord_id] = {
                "submitted_at": utc_now(),
                "retries": 0,
            }

            LOG.info(
                "[TRADEMGR] ✓ Submitted MKT order: %s %s qty=%.6f clOrdID=%s",
                side.name,
                self.symbol_id,
                quantity,
                clord_id,
            )

            # Audit log: Order submission
            self.audit.log_order_submit(
                order_id=clord_id,
                side="BUY" if side == Side.BUY else "SELL",
                quantity=quantity,  # FIXED: was qty=
                price=None,  # Market order
                ticket=None,  # Not yet assigned
            )

            # PAPER MODE: Schedule paper fill timeout
            if self.paper_mode and self.get_price_callback:
                self._schedule_paper_fill_timeout(clord_id, side, quantity)

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
        msg.setField(fix.TransactTime())
        msg.setField(fix.OrdType(OrdType.LIMIT.value))
        msg.setField(fix.OrderQty(round(float(quantity), 2)))
        msg.setField(fix.Price(float(price)))

        try:
            fix.Session.sendToTarget(msg, self.session_id)
            self.orders[clord_id] = order

            # P0 FIX: Track order submission time for timeout detection
            self.pending_orders[clord_id] = {
                "submitted_at": utc_now(),
                "retries": 0,
            }

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
        msg.setField(fix.TransactTime())

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
        msg.setField(fix.TransactTime())
        msg.setField(fix.OrdType(OrdType.LIMIT.value))
        msg.setField(fix.OrderQty(round(float(quantity), 2)))
        msg.setField(fix.Price(float(new_price)))

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

    # ----------------------------
    # PAPER MODE: Simulated Fills
    # ----------------------------
    def _schedule_paper_fill_timeout(self, clord_id: str, side: Side, quantity: float):
        """
        Schedule a paper fill timeout check.

        In PAPER_MODE, if broker doesn't respond within paper_fill_timeout seconds,
        simulate a fill locally using current bid/ask prices.

        This allows training to continue even when:
        - Market is closed
        - Broker connection is slow
        - Demo account has issues

        Args:
            clord_id: Client order ID to check
            side: Order side (BUY/SELL)
            quantity: Order quantity
        """
        import threading  # noqa: PLC0415

        def check_paper_fill():
            time.sleep(self.paper_fill_timeout)
            self._check_paper_fill_timeout(clord_id, side, quantity)

        thread = threading.Thread(
            target=check_paper_fill,
            daemon=True,
            name=f"PaperFill-{clord_id[:12]}",
        )
        thread.start()
        LOG.debug("[PAPER] Scheduled paper fill timeout for %s in %.1fs", clord_id, self.paper_fill_timeout)

    def _check_paper_fill_timeout(self, clord_id: str, side: Side, quantity: float):
        """
        Check if order needs paper fill simulation.

        Called after paper_fill_timeout seconds. If order still pending (broker
        hasn't responded), simulate the fill locally.

        Args:
            clord_id: Client order ID to check
            side: Order side
            quantity: Order quantity
        """
        # Check if order is still pending (broker hasn't responded)
        if clord_id not in self.pending_orders:
            # Already received broker response, no paper fill needed
            return

        order = self.orders.get(clord_id)
        if not order:
            LOG.warning("[PAPER] Order not found for paper fill: %s", clord_id)
            return

        # Order is still pending - simulate fill
        LOG.warning(
            "[PAPER] ⚠ No broker response after %.1fs - simulating paper fill for %s",
            self.paper_fill_timeout,
            clord_id,
        )
        self._simulate_paper_fill(order, side, quantity)

    def _simulate_paper_fill(self, order: Order, side: Side, quantity: float):
        """
        Simulate a paper fill using current market prices.

        This creates a fake ExecutionReport that updates position and triggers
        all the same callbacks as a real broker fill.

        Args:
            order: Order to fill
            side: Order side
            quantity: Order quantity
        """
        if not self.get_price_callback:
            LOG.error("[PAPER] Cannot simulate fill - no price callback")
            return

        try:
            bid, ask = self.get_price_callback()
        except Exception as e:
            LOG.error("[PAPER] Failed to get price for paper fill: %s", e)
            return

        if bid <= 0 or ask <= 0:
            LOG.error("[PAPER] Invalid prices for paper fill: bid=%.5f ask=%.5f", bid, ask)
            return

        # BUY fills at ASK, SELL fills at BID
        fill_price = ask if side == Side.BUY else bid

        # Generate paper ticket ID
        self.paper_fill_counter += 1
        paper_ticket = f"PAPER_{int(time.time())}_{self.paper_fill_counter}"
        paper_order_id = f"PAPER_ORD_{self.paper_fill_counter}"

        # Update order state
        order.status = OrderStatus.FILLED
        order.filled_qty = quantity
        order.avg_price = fill_price
        order.last_qty = quantity
        order.last_px = fill_price
        order.order_id = paper_order_id
        order.position_ticket = paper_ticket
        order.filled_at = utc_now()

        # Remove from pending orders
        if order.clord_id in self.pending_orders:
            del self.pending_orders[order.clord_id]

        # Store in broker_orders mapping
        self.broker_orders[paper_order_id] = order.clord_id

        # Store execution report for debugging
        self.exec_reports.append(
            {
                "timestamp": utc_now(),
                "clord_id": order.clord_id,
                "exec_type": "F",
                "ord_status": "2",
                "filled_qty": quantity,
                "avg_price": fill_price,
                "paper_fill": True,
            }
        )

        LOG.info(
            "[PAPER] ✓ Paper fill executed: %s qty=%.6f @%.5f ticket=%s",
            side.name,
            quantity,
            fill_price,
            paper_ticket,
        )

        # Update position from fill
        self.position.update_from_fill(side, quantity, fill_price)

        # Audit log: Paper fill
        self.audit.log_order_fill(
            order_id=order.clord_id,
            fill_price=fill_price,
            fill_qty=quantity,
            ticket=paper_ticket,
            fill_id=paper_order_id,
        )

        # Trigger fill callback (same as real fill)
        if self.on_fill_callback:
            try:
                self.on_fill_callback(order)
            except Exception as e:
                LOG.error("[PAPER] Error in fill callback: %s", e, exc_info=True)

    def _resolve_exec_report(self, msg: fix.Message) -> tuple | None:
        """Extract and validate required ExecutionReport fields.

        Returns (clord_id, exec_type_str, ord_status_str, order) or None if invalid.
        """
        clord_field = fix.ClOrdID()
        if not msg.isSetField(clord_field):
            LOG.warning("[TRADEMGR] ExecutionReport missing ClOrdID"); return None
        msg.getField(clord_field)
        clord_id = clord_field.getValue()

        exec_type_field = fix.ExecType()
        if not msg.isSetField(exec_type_field):
            LOG.warning("[TRADEMGR] ExecutionReport missing ExecType for %s", clord_id); return None
        msg.getField(exec_type_field)

        ord_status_field = fix.OrdStatus()
        if not msg.isSetField(ord_status_field):
            LOG.warning("[TRADEMGR] ExecutionReport missing OrdStatus for %s", clord_id); return None
        msg.getField(ord_status_field)

        order = self.orders.get(clord_id)
        if not order:
            LOG.warning("[TRADEMGR] Received ExecutionReport for unknown order: %s", clord_id); return None

        return clord_id, exec_type_field.getValue(), ord_status_field.getValue(), order

    def _populate_order_from_execution(self, msg: fix.Message, order, clord_id: str) -> None:
        """Update order with optional fields from an ExecutionReport message."""
        order_id_field = fix.OrderID()
        if msg.isSetField(order_id_field):
            msg.getField(order_id_field)
            order.order_id = order_id_field.getValue()
            self.broker_orders[order.order_id] = clord_id

        pos_ticket_field = fix.StringField(721)
        if msg.isSetField(pos_ticket_field):
            msg.getField(pos_ticket_field)
            order.position_ticket = pos_ticket_field.getValue()
            LOG.debug("[TRADEMGR] Position ticket: %s", order.position_ticket)
            self.audit.log_ticket_assigned(ticket=order.position_ticket, position_id=None, order_id=clord_id)

        cum_qty_field = fix.CumQty()
        if msg.isSetField(cum_qty_field):
            msg.getField(cum_qty_field)
            order.filled_qty = SafeMath.to_decimal(cum_qty_field.getValue(), self.position.instrument_digits)

        avg_px_field = fix.AvgPx()
        if msg.isSetField(avg_px_field):
            msg.getField(avg_px_field)
            order.avg_price = SafeMath.to_decimal(avg_px_field.getValue(), self.position.instrument_digits)

        last_qty_field = fix.LastQty()
        if msg.isSetField(last_qty_field):
            msg.getField(last_qty_field)
            order.last_qty = SafeMath.to_decimal(last_qty_field.getValue(), self.position.instrument_digits)

        last_px_field = fix.LastPx()
        if msg.isSetField(last_px_field):
            msg.getField(last_px_field)
            order.last_px = SafeMath.to_decimal(last_px_field.getValue(), self.position.instrument_digits)

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
            resolved = self._resolve_exec_report(msg)
            if resolved is None:
                return
            clord_id, exec_type_str, ord_status_str, order = resolved

            # Update order fields
            order.status = OrderStatus(ord_status_str)
            order.updated_at = utc_now()

            # Update optional fields from the report
            self._populate_order_from_execution(msg, order, clord_id)

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
                    clord_id, exec_type_str, ord_status_str,
                )

            # P0 FIX: Remove from pending orders once acknowledged
            if clord_id in self.pending_orders:
                del self.pending_orders[clord_id]

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

        # FIX: Update position from fill (cTrader doesn't respond to position requests)
        # Use last_qty for partial fills, filled_qty for full fills
        fill_qty = order.last_qty if order.last_qty > 0 else order.filled_qty
        self.position.update_from_fill(order.side, fill_qty, order.avg_price)

        # Audit log: Order fill
        self.audit.log_order_fill(
            order_id=order.clord_id,
            fill_price=order.avg_price,
            fill_qty=order.filled_qty,
            ticket=order.position_ticket,
            fill_id=order.order_id,
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

        # Audit log: Order cancellation
        self.audit.log_order_cancel(order_id=order.clord_id, reason="User/system cancellation")

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

        # Audit log: Order rejection
        self.audit.log_order_reject(order_id=order.clord_id, reason=order.reject_reason or "No reason provided")

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

            # P0 FIX: Schedule timeout check (improved threading-based timeout)
            import threading  # noqa: PLC0415

            def check_timeout():
                time.sleep(self.position_request_timeout)
                self._check_position_request_timeout(req_id)

            threading.Thread(target=check_timeout, daemon=True, name=f"PosReqTimeout-{req_id[:8]}").start()

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

    def check_all_position_request_timeouts(self):
        """
        P0 FIX: Manually check all pending position requests for timeouts.

        Call this periodically (e.g., every bar) as a fallback to threading-based checks.
        """
        for req_id in list(self.pending_position_requests.keys()):  # NOSONAR – list() needed: loop body pops from dict
            self._check_position_request_timeout(req_id)

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

    def check_pending_order_timeouts(self):
        """
        P0 FIX: Check for orders that haven't received acknowledgment within timeout.

        This prevents "lost in flight" orders where order status is unknown.
        Query order status via FIX OrderStatusRequest (35=H) after timeout.
        """
        now = utc_now()

        for clord_id in list(self.pending_orders.keys()):  # NOSONAR – list() needed: loop body deletes from dict
            pending_info = self.pending_orders[clord_id]
            elapsed = (now - pending_info["submitted_at"]).total_seconds()

            if elapsed > self.order_ack_timeout:
                retries = pending_info["retries"]

                if retries >= self.order_ack_max_retries:
                    LOG.error(
                        "[TRADEMGR] ✗ Order timeout - max retries reached: ClOrdID=%s (%.1fs elapsed)",
                        clord_id,
                        elapsed,
                    )
                    # Remove from pending (order is lost)
                    del self.pending_orders[clord_id]

                    # Mark order as failed
                    if clord_id in self.orders:
                        order = self.orders[clord_id]
                        order.status = OrderStatus.REJECTED
                        order.reject_reason = f"Timeout after {retries} status queries ({elapsed:.1f}s)"

                        # Trigger reject callback
                        if self.on_reject_callback:
                            try:
                                self.on_reject_callback(order)
                            except Exception as e:
                                LOG.error("[TRADEMGR] Error in reject callback: %s", e)
                else:
                    LOG.warning(
                        "[TRADEMGR] ⚠ Order acknowledgment timeout: ClOrdID=%s (%.1fs elapsed, retry %d/%d)",
                        clord_id,
                        elapsed,
                        retries + 1,
                        self.order_ack_max_retries,
                    )

                    # Query order status
                    self._query_order_status(clord_id)

                    # Increment retry counter
                    pending_info["retries"] += 1

    def _query_order_status(self, clord_id: str):
        """
        Query order status via FIX OrderStatusRequest (35=H).

        Args:
            clord_id: Client order ID to query
        """
        order = self.orders.get(clord_id)
        if not order:
            LOG.warning("[TRADEMGR] Cannot query status - order not found: %s", clord_id)
            return

        # Build OrderStatusRequest
        msg = fix44.OrderStatusRequest()
        msg.setField(fix.ClOrdID(clord_id))
        msg.setField(fix.Symbol(self.symbol_id))
        msg.setField(fix.Side(order.side.value))

        try:
            fix.Session.sendToTarget(msg, self.session_id)
            LOG.info("[TRADEMGR] → Sent OrderStatusRequest: ClOrdID=%s", clord_id)
        except Exception as e:
            LOG.error("[TRADEMGR] ✗ Failed to send OrderStatusRequest: %s", e)

    def on_logon(self):
        """
        P0 FIX: Handle session logon/reconnect.

        Critical recovery logic:
        1. Query status of all pending orders (might have been filled during disconnect)
        2. Force position reconciliation
        3. Clear stale pending requests

        Call this when FIX session logs on (especially after reconnect).
        """
        LOG.info("[TRADEMGR] ✓ Session logon - initiating recovery procedures")

        # 1. Query all pending orders
        pending_clord_ids = list(self.pending_orders.keys())
        if pending_clord_ids:
            LOG.warning(
                "[TRADEMGR] Found %d pending orders during reconnect - querying status",
                len(pending_clord_ids),
            )
            for clord_id in pending_clord_ids:
                self._query_order_status(clord_id)

        # 2. Force position reconciliation
        LOG.info("[TRADEMGR] Forcing position reconciliation after reconnect")
        self.request_positions(retry_count=0)

        # 3. Clear stale position requests (they're pre-disconnect)
        stale_count = len(self.pending_position_requests)
        if stale_count > 0:
            LOG.warning("[TRADEMGR] Clearing %d stale position requests", stale_count)
            self.pending_position_requests.clear()


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
