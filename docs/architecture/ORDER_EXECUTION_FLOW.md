# Order Execution Flow: TradeManager → HarvesterAgent → RiskManager → Broker

Complete end-to-end flow showing how orders move from agent decision to broker execution.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CTraderFixApp (Main Bot)                             │
│                                                                             │
│  ┌────────────────────┐         ┌─────────────────────┐                   │
│  │  TriggerAgent      │         │  HarvesterAgent     │                   │
│  │                    │         │                     │                   │
│  │  decide_entry()    │         │  decide_exit()      │                   │
│  │  → action 0/1/2    │         │  → exit 0/1         │                   │
│  └────────┬───────────┘         └──────────┬──────────┘                   │
│           │                                 │                              │
│           │ (Entry decision)                │ (Exit decision)              │
│           │                                 │                              │
│           ▼                                 ▼                              │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    on_bar_close() Decision Point                     │ │
│  │                                                                      │ │
│  │  if cur_pos == 0:                                                   │ │
│  │      # TriggerAgent decides entry                                   │ │
│  │      action, confidence, runway = policy.decide_entry(...)          │ │
│  │      desired = action  # 0=FLAT, 1=LONG, 2=SHORT                    │ │
│  │  else:                                                               │ │
│  │      # HarvesterAgent decides exit                                  │ │
│  │      exit_action, exit_conf = policy.decide_exit(...)               │ │
│  │      desired = 0 if exit_action == 1 else cur_pos                   │ │
│  └──────────────────────────┬───────────────────────────────────────────┘ │
│                             │                                             │
│                             │ if desired != cur_pos                       │
│                             ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                  CIRCUIT BREAKER CHECKS (Step 13)                    │ │
│  │                                                                      │ │
│  │  • VaR circuit breaker (kurtosis monitor)                           │ │
│  │  • VaR threshold check (current VaR vs max threshold)               │ │
│  │  • VPIN z-score filter                                              │ │
│  │  • Spread filter (2x learned minimum spread)                        │ │
│  │  • Circuit breaker position size multiplier                         │ │
│  │                                                                      │ │
│  │  if ANY breaker tripped → HALT, return early                        │ │
│  └──────────────────────────┬───────────────────────────────────────────┘ │
│                             │                                             │
│                             │ All safety checks passed                    │
│                             ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │               TradeManagerIntegration                                │ │
│  │                                                                      │ │
│  │  Methods:                                                            │ │
│  │  • enter_position(side, quantity) ← Entry                           │ │
│  │  • exit_position(quantity)         ← Exit                           │ │
│  │  • handle_execution_report(msg)    ← From broker                    │ │
│  │  • handle_position_report(msg)     ← From broker                    │ │
│  │                                                                      │ │
│  │  Callbacks:                                                          │ │
│  │  • on_order_filled(order)   → Update MFE/MAE, path tracking         │ │
│  │  • on_order_rejected(order) → Log rejection, update stats           │ │
│  └──────────────────────────┬───────────────────────────────────────────┘ │
│                             │                                             │
│                             ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                     TradeManager (Core)                              │ │
│  │                                                                      │ │
│  │  Order Lifecycle Management:                                         │ │
│  │  • submit_market_order(side, quantity)                              │ │
│  │  • submit_limit_order(side, quantity, price)                        │ │
│  │  • cancel_order(clord_id)                                           │ │
│  │  • modify_order(clord_id, new_price, new_qty)                       │ │
│  │                                                                      │ │
│  │  State Tracking:                                                     │ │
│  │  • orders: dict[clord_id → Order]                                   │ │
│  │  • pending_orders: dict[clord_id → metadata]                        │ │
│  │  • position: Position object (net_qty, avg_price)                   │ │
│  └──────────────────────────┬───────────────────────────────────────────┘ │
│                             │                                             │
└─────────────────────────────┼─────────────────────────────────────────────┘
                              │
                              │ FIX Protocol Messages
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          cTrader Broker (FIX Gateway)                       │
│                                                                             │
│  Incoming (from bot):                                                      │
│  • NewOrderSingle (35=D)      → Submit order                               │
│  • OrderCancelRequest (35=F)  → Cancel order                               │
│  • OrderCancelReplaceRequest (35=G) → Modify order                         │
│  • RequestForPositions (35=AN) → Query positions                           │
│                                                                             │
│  Outgoing (to bot):                                                        │
│  • ExecutionReport (35=8)     → Order status updates                       │
│      - ExecType=0 (New): Order accepted                                    │
│      - ExecType=F (Fill): Order filled                                     │
│      - ExecType=4 (Canceled): Order canceled                               │
│      - ExecType=8 (Rejected): Order rejected                               │
│  • PositionReport (35=AP)     → Position reconciliation                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Step-by-Step Flow

### Phase 1: Agent Decision (HarvesterAgent Example)

**Context**: Bar closure event, bot has open position

```python
# Line ~2567 in ctrader_ddqn_paper.py :: on_bar_close()

# HarvesterAgent decides exit
exit_action, exit_conf = self.policy.decide_exit(
    self.bars,
    current_price=c,
    imbalance=imbalance,
    vpin_z=vpin_zscore,
    depth_ratio=depth_ratio,
    event_features=event_features,
)

# exit_action: 0=HOLD, 1=CLOSE
desired = 0 if exit_action == 1 else self.cur_pos
```

**Outputs**:
- `exit_action`: 0 (HOLD) or 1 (CLOSE)
- `exit_conf`: Confidence level (0-1)
- `desired`: Desired position (0 if closing, cur_pos if holding)

---

### Phase 2: Risk Manager Validation (Circuit Breakers)

**Context**: Before any order submission, check all safety gates

```python
# Lines ~2750-2810 in ctrader_ddqn_paper.py

# Step 1: VaR circuit breaker check
if self.cur_pos == 0 and desired != 0:  # New entry only
    if self.kurtosis_monitor.is_breaker_active:
        LOG.warning("[CIRCUIT_BREAKER] Kurtosis breaker ACTIVE - skipping entry")
        return  # HALT

# Step 2: VaR threshold check
current_var = self.var_estimator.estimate_var(
    regime=self._current_var_regime(),
    vpin_z=vpin_zscore,
    current_vol=realized_vol
)
if current_var > max_var_threshold:
    LOG.warning("[CIRCUIT_BREAKER] VaR exceeds threshold - skipping entry")
    return  # HALT

# Step 3: VPIN z-score filter
if vpin_zscore > self.vpin_z_threshold:
    LOG.warning("[VPIN] z-score exceeds threshold - skipping entry")
    return  # HALT

# Step 4: Spread filter (2x learned minimum spread)
is_acceptable, current_spread, max_spread = self.friction_calculator.is_spread_acceptable()
if not is_acceptable:
    LOG.warning("[SPREAD_FILTER] Spread too wide - skipping entry")
    return  # HALT

# Step 5: Circuit breaker position size multiplier
size_multiplier = self.circuit_breakers.get_position_size_multiplier()
order_qty = self._compute_order_qty(abs(delta), size_multiplier, is_new_entry)

if size_multiplier < 1.0:
    LOG.warning("[CIRCUIT-BREAKER] Position size reduced: %.2f%%", size_multiplier * 100)

if order_qty <= 0:
    LOG.warning("[RISK] Order blocked - zero qty after constraints")
    return  # HALT
```

**Risk Manager Checks** (Conceptual - would be centralized in RiskManager):
- ✅ VaR within acceptable range
- ✅ VPIN toxicity acceptable
- ✅ Spread cost acceptable
- ✅ Position size approved (may be reduced)
- ✅ Circuit breakers not tripped

**Outputs**:
- `order_qty`: Approved order quantity (may be reduced)
- `size_multiplier`: Position size adjustment (0.0-1.0)

---

### Phase 3: Order Submission via TradeManager

**Context**: All risk checks passed, submit order to broker

```python
# Lines ~2840-2920 in ctrader_ddqn_paper.py :: send_market_order()

# Step 1: Safety checks (stale price, connection health)
if not self.connection_healthy:
    LOG.warning("[SAFETY] Order blocked - connection unhealthy")
    return None

quote_age = (utc_now() - self.last_quote_heartbeat).total_seconds()
if quote_age > self.max_quote_age_for_trading:
    LOG.warning("[SAFETY] Order blocked - quote data stale")
    return None

if not self.trade_integration.trade_manager:
    LOG.error("[SAFETY] Order blocked - TradeManager not initialized")
    return None

# Step 2: Check for pending order timeouts
self.trade_integration.trade_manager.check_pending_order_timeouts()

# Step 3: Adjust quantity for execution costs
mid_price = (self.best_bid + self.best_ask) / 2.0
spread_bps = ((self.best_ask - self.best_bid) / mid_price) * 10000.0

adjusted_qty = self.execution_model.adjust_position_size_for_costs(
    side=exec_side,
    target_quantity=qty,
    mid_price=mid_price,
    spread_bps=spread_bps,
    regime=self.regime_detector.current_regime
)

# Step 4: Submit order via TradeManager
order = self.trade_integration.trade_manager.submit_market_order(
    side=tm_side,
    quantity=adjusted_qty,
    tag_prefix="DDQN"
)
```

**TradeManager Submission Process**:

```python
# Lines ~259-322 in trade_manager.py :: submit_market_order()

# Step 1: Check max pending orders limit
if len(active_orders) >= self.max_pending_orders:
    LOG.warning("[TRADEMGR] Max pending orders reached")
    return None

# Step 2: Generate unique client order ID
clord_id = f"{tag_prefix}_{timestamp}_{counter}"

# Step 3: Create Order object
order = Order(
    clord_id=clord_id,
    symbol=self.symbol_id,
    side=side,
    ord_type=OrdType.MARKET,
    quantity=quantity,
)

# Step 4: Build FIX NewOrderSingle message (35=D)
msg = fix44.NewOrderSingle()
msg.setField(fix.ClOrdID(clord_id))
msg.setField(fix.Symbol(self.symbol_id))
msg.setField(fix.Side(side.value))
msg.setField(fix.TransactTime(utc_ts_ms()))
msg.setField(fix.OrdType(OrdType.MARKET.value))
msg.setField(fix.OrderQty(quantity))

# Step 5: Send to broker via FIX session
fix.Session.sendToTarget(msg, self.session_id)

# Step 6: Track order locally
self.orders[clord_id] = order
self.pending_orders[clord_id] = {
    "submitted_at": utc_now(),
    "retries": 0,
}

LOG.info("[TRADEMGR] ✓ Submitted MKT order: %s qty=%.6f clOrdID=%s", 
         side.name, quantity, clord_id)
return order
```

**FIX Message Sent to Broker**:
```
35=D              # NewOrderSingle
11=DDQN_1736616000_123  # ClOrdID (unique client ID)
55=1              # Symbol (BTCUSD)
54=1              # Side (1=BUY, 2=SELL)
60=20260111-14:30:00.000  # TransactTime (UTC)
40=1              # OrdType (1=MARKET)
38=0.10           # OrderQty
```

---

### Phase 4: Broker Processing

**Broker Side** (cTrader FIX Gateway):

1. **Receive NewOrderSingle** (35=D)
   - Validate order (margin, limits, market hours)
   - Assign broker OrderID
   - Queue for execution

2. **Send ExecutionReport - NEW** (35=8, ExecType=0)
   ```
   35=8              # ExecutionReport
   11=DDQN_1736616000_123  # ClOrdID (echoed)
   37=BROKER_987654  # OrderID (broker-assigned)
   39=0              # OrdStatus (0=NEW)
   150=0             # ExecType (0=NEW - order accepted)
   ```

3. **Execute Order** (match against order book)
   - Market order fills at best available price
   - Slippage may occur

4. **Send ExecutionReport - FILL** (35=8, ExecType=F)
   ```
   35=8              # ExecutionReport
   11=DDQN_1736616000_123  # ClOrdID
   37=BROKER_987654  # OrderID
   39=2              # OrdStatus (2=FILLED)
   150=F             # ExecType (F=FILL/Trade)
   14=0.10           # CumQty (cumulative filled)
   6=65123.45        # AvgPx (average fill price)
   32=0.10           # LastQty (last fill qty)
   31=65123.45       # LastPx (last fill price)
   ```

5. **Send PositionReport** (35=AP)
   ```
   35=AP             # PositionReport
   55=1              # Symbol
   721=BROKER_POS_ID # PosMaintRptID
   715=20260111      # ClearingBusinessDate
   730=0.10          # SettlPrice
   702=LONG          # PosType (LONG/SHORT)
   703=QTY           # PosQtyStatus
   704=0.10          # LongQty
   705=0.00          # ShortQty
   ```

---

### Phase 5: Broker Response Handling

**ExecutionReport Processing** (TradeManager):

```python
# Lines ~486-600 in trade_manager.py :: on_execution_report()

def on_execution_report(self, msg: fix.Message):
    # Step 1: Extract ClOrdID
    clord_field = fix.ClOrdID()
    msg.getField(clord_field)
    clord_id = clord_field.getValue()
    
    # Step 2: Find order
    order = self.orders.get(clord_id)
    if not order:
        LOG.warning("[TRADEMGR] Unknown order: %s", clord_id)
        return
    
    # Step 3: Extract ExecType
    exec_type_field = fix.ExecType()
    msg.getField(exec_type_field)
    exec_type = exec_type_field.getValue()
    
    # Step 4: Route to handler based on ExecType
    if exec_type == '0':      # NEW
        self._handle_new(order)
    elif exec_type == 'F':    # FILL
        self._handle_fill(order)
    elif exec_type == '4':    # CANCELED
        self._handle_canceled(order)
    elif exec_type == '8':    # REJECTED
        self._handle_rejected(order)
    elif exec_type == 'I':    # OrderStatus
        self._handle_status(order)
```

**FILL Handler** (TradeManager):

```python
# Lines ~613-648 in trade_manager.py :: _handle_fill()

def _handle_fill(self, order: Order):
    # Step 1: Update order status
    order.status = OrderStatus.FILLED
    
    # Step 2: Remove from pending orders
    self.pending_orders.pop(order.clord_id, None)
    
    # Step 3: Update position
    if order.side == Side.BUY:
        self.position.long_qty += order.filled_qty
    else:
        self.position.short_qty += order.filled_qty
    
    self.position.net_qty = self.position.long_qty - self.position.short_qty
    
    # Weighted average entry price
    total_value = self.position.avg_price * prev_qty + order.avg_price * order.filled_qty
    self.position.avg_price = total_value / self.position.net_qty
    
    # Step 4: Log fill
    LOG.info(
        "[TRADEMGR] ✓ Order FILLED: %s qty=%.6f @%.5f net_pos=%.6f",
        order.side.name,
        order.filled_qty,
        order.avg_price,
        self.position.net_qty
    )
    
    # Step 5: Trigger callback
    if self.on_fill_callback:
        self.on_fill_callback(order)
```

**Fill Callback** (TradeManagerIntegration → HarvesterAgent):

```python
# Lines ~88-166 in trade_manager_example.py :: on_order_filled()

def on_order_filled(self, order: Order):
    LOG.info(
        "[INTEGRATION] Order filled: %s qty=%.6f @%.5f",
        order.side.name,
        order.filled_qty,
        order.avg_price,
    )
    
    # Step 1: Determine position ID
    position_id = self._get_position_id_for_order(order)
    
    # Step 2: Start MFE/MAE tracking (GAP 4.3 fix)
    if hasattr(self.app, "mfe_mae_trackers") and order.avg_price > 0:
        direction = 1 if order.side == Side.BUY else -1
        
        # Create tracker for this position
        if position_id not in self.app.mfe_mae_trackers:
            from mfe_mae_tracker import MFEMAETracker
            self.app.mfe_mae_trackers[position_id] = MFEMAETracker(
                entry_price=order.avg_price,
                direction=direction,
                symbol=order.symbol
            )
        
        # Start path recording
        if hasattr(self.app, "path_recorders"):
            from path_geometry import PathRecorder
            self.app.path_recorders[position_id] = PathRecorder(
                entry_price=order.avg_price,
                direction=direction
            )
        
        LOG.info(
            "[INTEGRATION] ✓ Started MFE/MAE tracking for position %s: entry=%.5f dir=%d",
            position_id,
            order.avg_price,
            direction
        )
    
    # Step 3: Update activity monitor
    if hasattr(self.app, "activity_monitor"):
        self.app.activity_monitor.record_action("order_fill")
    
    # Step 4: Update internal position state
    self.app.cur_pos = self.trade_manager.get_position_direction(
        min_qty=self.app.qty * 0.5
    )
    
    LOG.info("[INTEGRATION] ✓ Position synced: cur_pos=%d", self.app.cur_pos)
```

**HarvesterAgent Receives Fill**:
- MFE/MAE tracking initiated
- Path geometry recording started
- Position state updated (`self.cur_pos`)
- Activity monitor notified

---

### Phase 6: Position Reconciliation

**Position Report Processing**:

```python
# Lines ~455-484 in trade_manager_example.py :: handle_position_report()

def handle_position_report(self, msg: fix.Message):
    if self.trade_manager:
        # Route to TradeManager
        self.trade_manager.on_position_report(msg)
        
        # Update app's cur_pos for backward compatibility
        self.app.cur_pos = self.trade_manager.get_position_direction(
            min_qty=self.app.qty * 0.5
        )
        
        LOG.info(
            "[INTEGRATION] Position synced: cur_pos=%d net_qty=%.6f",
            self.app.cur_pos,
            self.trade_manager.position.net_qty,
        )
```

**TradeManager Position Reconciliation**:

```python
# Lines ~769-843 in trade_manager.py :: on_position_report()

def on_position_report(self, msg: fix.Message):
    # Extract position details from FIX message
    symbol = msg.getField(fix.Symbol()).getValue()
    long_qty = float(msg.getField(fix.LongQty()).getValue())
    short_qty = float(msg.getField(fix.ShortQty()).getValue())
    net_qty = long_qty - short_qty
    
    # Update position
    self.position.long_qty = long_qty
    self.position.short_qty = short_qty
    self.position.net_qty = net_qty
    
    # Clear pending position request
    if self.pos_req_id:
        self.pending_position_requests.pop(self.pos_req_id, None)
        self.pos_req_id = None
    
    LOG.info(
        "[TRADEMGR] Position update: symbol=%s long=%.6f short=%.6f net=%.6f",
        symbol,
        long_qty,
        short_qty,
        net_qty
    )
```

---

## Complete Flow Summary

### Entry Flow (TriggerAgent → Broker)

```
TriggerAgent.decide_entry()
  → action=1 (LONG), confidence=0.85, runway=0.0035
  → desired=1 (from FLAT to LONG)
  
Circuit Breaker Checks
  ✓ VaR acceptable
  ✓ VPIN acceptable
  ✓ Spread acceptable
  ✓ Position size approved (0.10 lots)
  
TradeManagerIntegration.enter_position(side=1, quantity=0.10)
  → Validation via VALIDATOR
  → Adjusted quantity for costs
  
TradeManager.submit_market_order(Side.BUY, 0.10)
  → Generate ClOrdID: "DDQN_1736616000_123"
  → Build FIX NewOrderSingle (35=D)
  → Send to broker via FIX session
  → Track in pending_orders
  
Broker Receives NewOrderSingle
  → Validate order
  → Assign OrderID: "BROKER_987654"
  → Send ExecutionReport ExecType=0 (NEW)
  
TradeManager.on_execution_report() [ExecType=0]
  → Order accepted, status = NEW
  
Broker Executes Order
  → Fill at 65123.45
  → Send ExecutionReport ExecType=F (FILL)
  
TradeManager.on_execution_report() [ExecType=F]
  → Order filled, status = FILLED
  → Update position: net_qty=0.10, avg_price=65123.45
  → Trigger on_fill_callback()
  
TradeManagerIntegration.on_order_filled(order)
  → Start MFE/MAE tracking
  → Start path recording
  → Update activity monitor
  → Sync cur_pos=1 (LONG)
  
Broker Sends PositionReport (35=AP)
  
TradeManager.on_position_report()
  → Reconcile position: long_qty=0.10, short_qty=0.00
  → Update position state
```

### Exit Flow (HarvesterAgent → Broker)

```
HarvesterAgent.decide_exit()
  → exit_action=1 (CLOSE), exit_conf=0.92
  → desired=0 (from LONG to FLAT)
  → Current MFE=250 pips, MAE=80 pips, bars_held=15
  
Risk Checks
  ✓ No entry checks (exiting position)
  ✓ Position size approved
  
TradeManagerIntegration.exit_position(quantity=0.10)
  → Determine exit side: SELL (opposite of LONG)
  
TradeManager.submit_market_order(Side.SELL, 0.10, tag_prefix="EXIT")
  → Generate ClOrdID: "EXIT_1736617000_124"
  → Build FIX NewOrderSingle (35=D)
  → Send to broker
  
Broker Executes
  → Fill at 65373.45 (250 pip profit)
  → Send ExecutionReport ExecType=F (FILL)
  
TradeManager.on_execution_report() [ExecType=F]
  → Order filled
  → Update position: net_qty=0.00 (FLAT)
  → Trigger on_fill_callback()
  
TradeManagerIntegration.on_order_filled(order)
  → Finalize MFE/MAE metrics
  → Complete path recording
  → Calculate realized PnL: +250 pips
  → Update performance tracker
  → Sync cur_pos=0 (FLAT)
  
HarvesterAgent Learning
  → Calculate capture reward: (250 / 250) = 1.0 (100% capture)
  → Store experience: (state, action=CLOSE, reward=1.0, next_state, done=True)
  → Add to harvester experience buffer
  → Periodic training updates network weights
```

---

## Key Components Interaction Matrix

| Component | Inputs | Outputs | Interactions |
|-----------|--------|---------|--------------|
| **TriggerAgent** | Bars, features, volatility | action, confidence, runway | → on_bar_close() |
| **HarvesterAgent** | Current position, MFE, MAE, bars_held | exit_action, exit_conf | → on_bar_close() |
| **on_bar_close()** | Agent decisions | desired position | → Circuit breakers<br>→ TradeManagerIntegration |
| **Circuit Breakers** | VaR, VPIN, spread, position | approve/deny, size_multiplier | → Risk validation gate |
| **TradeManagerIntegration** | desired position, quantity | enter/exit methods | → TradeManager<br>→ Callbacks (fill/reject) |
| **TradeManager** | Order requests | FIX messages, order tracking | → Broker (FIX)<br>→ Position reconciliation |
| **Broker (cTrader)** | NewOrderSingle, requests | ExecutionReports, PositionReports | → TradeManager |
| **on_order_filled()** | Order object (fill details) | MFE/MAE tracking, position updates | → HarvesterAgent state<br>→ Path recording |

---

## Safety Layers

### Layer 1: Pre-Decision (Circuit Breakers)
- **VaR Monitoring**: Kurtosis circuit breaker, regime-aware VaR estimation
- **Market Quality**: VPIN toxicity filter, spread cost filter
- **Purpose**: Prevent bad decisions before they reach TradeManager

### Layer 2: Order Validation (TradeManager)
- **Connection Health**: Quote staleness check, heartbeat monitoring
- **Order Limits**: Max pending orders, timeout detection
- **Execution Costs**: Spread-adjusted quantity, regime-aware sizing
- **Purpose**: Ensure safe order submission

### Layer 3: Broker-Side (cTrader FIX)
- **Margin Checks**: Sufficient account balance
- **Order Validation**: Symbol, price, quantity limits
- **Market Hours**: Trading session validation
- **Purpose**: Final broker-side validation

### Layer 4: Post-Execution (Position Reconciliation)
- **PositionReport Sync**: Verify broker position matches internal state
- **Timeout Detection**: Pending order timeouts, position request retries
- **State Recovery**: Persistence-based state recovery on restart
- **Purpose**: Maintain accurate position state

---

## Error Handling

### Order Rejection Flow

```
Broker → ExecutionReport ExecType=8 (REJECTED)
  ↓
TradeManager._handle_rejected(order)
  → order.status = REJECTED
  → Extract reject_reason from Tag 103
  → Remove from pending_orders
  → Trigger on_reject_callback()
  ↓
TradeManagerIntegration.on_order_rejected(order)
  → Log rejection reason
  → Update rejection statistics
  → NO position change (order never filled)
  → Bot continues normal operation
```

### Order Timeout Detection

```
TradeManager.check_pending_order_timeouts()
  → For each pending order:
      if (now - submitted_at) > order_ack_timeout:
          if retries < max_retries:
              → Send OrderStatusRequest (35=H)
              → Increment retry count
          else:
              → LOG.error("Order timeout exceeded, manual intervention needed")
              → Mark order as TIMEOUT_ERROR
```

### Position Request Timeout

```
TradeManager.check_all_position_request_timeouts()
  → For each pending position request:
      if (now - requested_at) > position_request_timeout:
          if retries < max_retries:
              → Resend RequestForPositions (35=AN)
              → Increment retry count
          else:
              → LOG.error("Position request timeout, cannot reconcile")
```

---

## Data Flow: Decision → Execution → Learning

```
Bar Close Event
  ↓
Feature Calculation (VaR, volatility, imbalance, event-time)
  ↓
Agent Decision (TriggerAgent OR HarvesterAgent)
  ↓
Risk Validation (Circuit breakers, VaR, VPIN, spread)
  ↓
Order Submission (TradeManager → Broker)
  ↓
Broker Execution (FIX ExecutionReport)
  ↓
Position Update (PositionReport reconciliation)
  ↓
MFE/MAE Tracking (on_order_filled callback)
  ↓
Path Recording (Geometry features for next decision)
  ↓
Reward Calculation (HarvesterAgent hold/close rewards)
  ↓
Experience Storage (Replay buffer)
  ↓
Online Learning (Periodic training step)
  ↓
Updated Network Weights → Next Bar Decision
```

---

## Performance Impact

### Latency Breakdown

| Stage | Typical Latency | Notes |
|-------|----------------|-------|
| Agent Decision | <1ms | Neural network forward pass |
| Risk Validation | <1ms | Circuit breaker checks |
| Order Submission | 5-20ms | FIX message send to broker |
| Broker Processing | 10-50ms | Order matching, validation |
| ExecutionReport Return | 5-20ms | FIX message from broker |
| Position Update | <1ms | Internal state sync |
| **Total (Entry)** | **~25-100ms** | Bar-driven (not time-critical) |

**Note**: Bot operates on bar-close events (M1/M5/M15), so sub-second latency is acceptable.

---

## Files Reference

| File | Role | Key Methods |
|------|------|-------------|
| **ctrader_ddqn_paper.py** | Main bot, agents | on_bar_close(), send_market_order() |
| **trade_manager_example.py** | TradeManager integration | enter_position(), exit_position(), on_order_filled() |
| **trade_manager.py** | Order lifecycle | submit_market_order(), on_execution_report() |
| **trade_manager_safety.py** | Validation layer | VALIDATOR.validate_order() |
| **dual_policy.py** | TriggerAgent + HarvesterAgent | decide_entry(), decide_exit() |
| **harvester_agent.py** | Exit decision agent | decide_exit(), get_position_metrics() |
| **trigger_agent.py** | Entry decision agent | decide_entry() |
| **circuit_breakers.py** | Safety gates | is_any_tripped(), get_position_size_multiplier() |
| **var_estimator.py** | VaR calculation | estimate_var() |
| **friction_costs.py** | Spread filter | is_spread_acceptable() |

---

**Status**: ✅ Complete Integration - Order flow from agent decision to broker execution fully documented  
**Date**: January 11, 2026  
**System**: CTrader DDQN Trading Bot with Dual-Agent RL Architecture
