# Hedging Mode Position Tracking - Design Document

## Problem Statement
Current system tracks positions by internally-generated ClOrdIDs, but in hedging mode:
- Each trade creates a separate position with broker-assigned ticket number
- Multiple positions can be open simultaneously (e.g., 10 positions on H4 timeframe)
- To close a specific position, must reference its broker ticket number
- Positions persist across crashes/restarts - need to reconcile with broker state

## Broker Position Model (cTrader Hedging Mode)
```
When BUY 0.1 BTCUSD fills:
  → Broker creates position ticket: #12345678
  → ExecutionReport contains: Tag 721 (PosMaintRptID) = "12345678"
  
When SELL 0.1 BTCUSD fills:
  → Broker creates SEPARATE position ticket: #12345679
  → Does NOT offset existing LONG position
  → Two positions now exist independently

To close position #12345678:
  → Must submit order with PositionID reference
  → OR use OrderCancelRequest with specific ticket
```

## Required Changes

### 1. Position Ticket Tracking
**Order dataclass** (trade_manager.py):
```python
@dataclass
class Order:
    # ... existing fields ...
    
    # CRITICAL: Broker position ticket created by this order
    position_ticket: str | None = None  # Tag 721: PosMaintRptID from ExecutionReport
```

**Extract from ExecutionReport**:
```python
def on_execution_report(self, msg: fix.Message):
    # ... existing parsing ...
    
    # Extract position ticket (Tag 721)
    pos_ticket_field = fix.StringField(721)  # PosMaintRptID
    if msg.isSetField(pos_ticket_field):
        msg.getField(pos_ticket_field)
        order.position_ticket = pos_ticket_field.getValue()
```

### 2. Position Tracker Mapping
**Trade Manager Integration**:
```python
class TradeManagerIntegration:
    def __init__(self):
        # OLD: Position ID = f"{symbol}_{ClOrdID}"
        # NEW: Position ID = broker ticket number
        
        # Map broker ticket → tracker
        self.position_tickets: dict[str, PositionTracker] = {}
        
        # Map exit order ClOrdID → position ticket being closed
        self.exit_orders: dict[str, str] = {}  # clOrdID → ticket
```

**On order fill** (entry):
```python
def on_order_filled(self, order: Order):
    # Check if this is an exit order
    if order.clord_id in self.exit_orders:
        ticket_to_close = self.exit_orders.pop(order.clord_id)
        # Remove tracker for closed position
        if ticket_to_close in self.position_tickets:
            del self.position_tickets[ticket_to_close]
            LOG.info(f"Position {ticket_to_close} closed")
        return
    
    # This is an entry order - create tracker
    if order.position_ticket:
        tracker = PositionTracker(
            ticket=order.position_ticket,
            entry_price=order.avg_price,
            direction=1 if order.side == Side.BUY else -1,
            entry_time=order.filled_at,
            quantity=order.filled_qty
        )
        self.position_tickets[order.position_ticket] = tracker
        LOG.info(f"Tracking position {order.position_ticket}")
```

### 3. Position Closing by Ticket
**Close specific position**:
```python
def close_position(self, ticket: str, reason: str = "MANUAL") -> bool:
    """Close position by broker ticket number."""
    tracker = self.position_tickets.get(ticket)
    if not tracker:
        LOG.error(f"Position {ticket} not found")
        return False
    
    # Submit opposite order
    exit_side = Side.SELL if tracker.direction > 0 else Side.BUY
    order = self.trade_manager.submit_market_order(
        side=exit_side,
        quantity=tracker.quantity,
        tag_prefix=f"EXIT_{reason}"
    )
    
    if order:
        # Map this exit order to the position it's closing
        self.exit_orders[order.clord_id] = ticket
        LOG.info(f"Submitted exit order {order.clord_id} to close position {ticket}")
        return True
    return False
```

### 4. Crash Recovery / State Persistence
**Persist ALL position tickets**:
```python
def _persist_state(self):
    state = {
        "positions": {
            ticket: {
                "entry_price": tracker.entry_price,
                "direction": tracker.direction,
                "quantity": tracker.quantity,
                "entry_time": tracker.entry_time.isoformat(),
                "mfe": tracker.mfe,
                "mae": tracker.mae
            }
            for ticket, tracker in self.position_tickets.items()
        }
    }
    self.persistence.save_json(state, self.state_filename)
```

**Recover on restart**:
```python
def _recover_state(self):
    state = self.persistence.load_json(self.state_filename)
    if not state:
        return False
    
    # Restore all position trackers
    for ticket, pos_data in state.get("positions", {}).items():
        tracker = PositionTracker(
            ticket=ticket,
            entry_price=pos_data["entry_price"],
            direction=pos_data["direction"],
            quantity=pos_data["quantity"],
            entry_time=datetime.fromisoformat(pos_data["entry_time"])
        )
        tracker.mfe = pos_data.get("mfe", 0.0)
        tracker.mae = pos_data.get("mae", 0.0)
        self.position_tickets[ticket] = tracker
        LOG.info(f"Recovered position {ticket}")
```

### 5. Harvester Tick Evaluation
**Evaluate each position by ticket**:
```python
def _evaluate_harvester_on_tick(self):
    for ticket, tracker in list(self.position_tickets.items()):
        tracker.update(current_price)
        
        should_exit = harvester.decide_exit(tracker)
        if should_exit:
            LOG.info(f"Harvester: Closing position {ticket}")
            self.trade_integration.close_position(ticket, reason="HARVESTER")
```

## Implementation Priority
1. ✅ Extract Tag 721 from ExecutionReport → Order.position_ticket
2. ✅ Change position_id from ClOrdID to broker ticket
3. ✅ Map exit orders → tickets being closed
4. ✅ Persist/recover by ticket numbers
5. ✅ Update Harvester to evaluate by ticket

## Testing Plan
1. Open 2 LONG positions → verify 2 separate tickets tracked
2. Close 1 position → verify only that tracker removed
3. Restart bot → verify both positions recovered
4. Let Harvester close 1 position → verify correct ticket closed
5. Check state file → verify all tickets persisted

## Migration Notes
- Old state files use ClOrdID-based position IDs → need migration
- Can detect by checking if position_id contains "cl_" prefix
- Graceful degradation: if no Tag 721, fall back to ClOrdID tracking
