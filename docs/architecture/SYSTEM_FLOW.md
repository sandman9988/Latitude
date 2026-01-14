# System Flow - Complete Execution Sequence
**Date:** January 10, 2026  
**Project:** Adaptive Trading System (Python/cTrader)

---

## Overview

This document describes the complete logical flow of the trading system from startup through execution, learning, and shutdown. Each component's role and interaction points are detailed to guide implementation and debugging.

---

## 1. STARTUP SEQUENCE

### 1.1 Application Initialization (`main()`)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Signal Handlers Setup                                    │
│    - Register SIGTERM, SIGINT for graceful shutdown         │
│    - Set shutdown flags                                     │
├─────────────────────────────────────────────────────────────┤
│ 2. Environment Validation                                   │
│    - Check CTRADER_USERNAME                                 │
│    - Check CTRADER_PASSWORD_QUOTE                           │
│    - Check CTRADER_PASSWORD_TRADE                           │
│    - Parse symbol_id, symbol, qty, timeframe                │
├─────────────────────────────────────────────────────────────┤
│ 3. Component Initialization Order (CRITICAL)                │
│    a. SafeMath initialization (defensive programming)       │
│    b. Logging setup                                         │
│    c. BotPersistenceManager (state management)              │
│    d. LearnedParametersManager (adaptive parameters)        │
│    e. CTraderFixApp (main application)                      │
│       - FrictionCalculator                                  │
│       - PerformanceTracker                                  │
│       - CircuitBreakers                                     │
│       - EventTimeFeatureEngine                              │
│       - PathGeometry                                        │
│       - RewardShaper                                        │
│       - DualPolicy (TriggerAgent + HarvesterAgent)          │
│       - ActivityMonitor                                     │
│       - AdaptiveRegularization                              │
│       - GeneralizationMonitor                               │
│       - EarlyStopping                                       │
│       - OrderBook & VPINCalculator                          │
├─────────────────────────────────────────────────────────────┤
│ 4. FIX Session Configuration                                │
│    - Load quote session config (ctrader_quote.cfg)          │
│    - Load trade session config (ctrader_trade.cfg)          │
│    - Create StoreFactory, LogFactory                        │
│    - Create SocketInitiator for each session               │
├─────────────────────────────────────────────────────────────┤
│ 5. Start Sessions                                           │
│    - Start quote session initiator                          │
│    - Start trade session initiator                          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Critical Initialization Dependencies

```python
# Initialization order MUST be:
1. SafeMath          # No dependencies
2. Logging           # No dependencies
3. RingBuffer        # Depends on SafeMath
4. AtomicPersistence # Depends on Logging
5. BotPersistence    # Depends on AtomicPersistence
6. LearnedParams     # Depends on BotPersistence
7. All other components (can initialize in parallel)
```

**❌ Current Gap:** No `InitGate` to enforce initialization order

---

## 2. CONNECTION SEQUENCE

### 2.1 FIX Session Establishment

**FIX Session Protocol:**
- **Session-based communication**: Initiator (client) → Acceptor (cTrader server)
- **Sequence numbers**: Every message gets unique MsgSeqNum starting from 1
- **Bi-directional**: Both client and server track sequence numbers independently
- **Message recovery**: Missing messages re-transmitted via ResendRequest (35=2)
- **Heartbeat interval**: Default 30 seconds (configurable via HeartBtInt tag 108)

**FIX Message Structure:**
```
Header (BeginString, BodyLength, MsgType, MsgSeqNum, SendingTime, etc.)
  ↓
Body (message-specific tags)
  ↓
Footer (CheckSum)
```

**Key FIX Tags:**
- **Tag 8 (BeginString)**: FIX.4.4
- **Tag 35 (MsgType)**: A=Logon, 5=Logout, 0=Heartbeat, D=NewOrderSingle, 8=ExecutionReport, etc.
- **Tag 49 (SenderCompID)**: `{broker}.{accountId}` (e.g., "theBroker.12345")
- **Tag 56 (TargetCompID)**: CSERVER (always)
- **Tag 57 (TargetSubID)**: QUOTE (market data) or TRADE (order execution)
- **Tag 34 (MsgSeqNum)**: Message sequence number
- **Tag 108 (HeartBtInt)**: Heartbeat interval in seconds

```
┌─────────────────────────────────────────────────────────────┐
│ QUOTE SESSION (Market Data)                                │
├─────────────────────────────────────────────────────────────┤
│ onCreate(sessionID)                                         │
│   └─> Session created, not yet logged in                   │
│                                                             │
│ Client sends Logon (35=A):                                 │
│   ├─> Tag 49: {broker}.{accountId}                         │
│   ├─> Tag 56: CSERVER                                      │
│   ├─> Tag 57: QUOTE                                        │
│   ├─> Tag 553: Username (accountId)                        │
│   ├─> Tag 554: Password                                    │
│   └─> Tag 108: HeartBtInt (30 seconds)                     │
│                                                             │
│ onLogon(sessionID) [QUOTE]                                  │
│   ├─> Mark quote_sid = sessionID                           │
│   ├─> Reset MsgSeqNum to 1                                 │
│   ├─> Mark connection_healthy = True                       │
│   ├─> Start heartbeat timer (send Heartbeat if no msgs)    │
│   └─> Send MarketDataRequest (35=V) to subscribe           │
│       └─> Request bid/ask/last price updates               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TRADE SESSION (Order Execution)                            │
├─────────────────────────────────────────────────────────────┤
│ onCreate(sessionID)                                         │
│   └─> Session created, not yet logged in                   │
│                                                             │
│ Client sends Logon (35=A):                                 │
│   ├─> Tag 49: {broker}.{accountId}                         │
│   ├─> Tag 56: CSERVER                                      │
│   ├─> Tag 57: TRADE                                        │
│   ├─> Tag 553: Username (accountId)                        │
│   ├─> Tag 554: Password                                    │
│   └─> Tag 108: HeartBtInt (30 seconds)                     │
│                                                             │
│ onLogon(sessionID) [TRADE]                                  │
│   ├─> Mark trade_sid = sessionID                           │
│   ├─> Reset MsgSeqNum to 1                                 │
│   ├─> Mark connection_healthy = True                       │
│   ├─> Start heartbeat timer                                │
│   └─> Request current positions (PositionReport)           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Heartbeat & Sequence Number Management

**Heartbeat Mechanism (Tag 108 = 30 seconds):**
```python
# Client responsibilities:
1. If no message sent within HeartBtInt seconds:
   └─> Send Heartbeat (35=0) to server

2. If no message received from server within HeartBtInt + buffer:
   └─> Send TestRequest (35=1) to test link health
   └─> If no response: disconnect and reconnect

3. On receiving Heartbeat (35=0):
   └─> Update last_received_time
   └─> No action required

4. On receiving TestRequest (35=1):
   └─> Respond with Heartbeat (35=0) immediately
```

**Sequence Number Tracking:**
```python
# Both client and server maintain:
- outgoing_seqnum: MsgSeqNum for next outgoing message (starts at 1)
- incoming_seqnum: Expected MsgSeqNum for next incoming message (starts at 1)

# On send:
outgoing_seqnum += 1
msg.setField(fix.MsgSeqNum(outgoing_seqnum))

# On receive:
if msg_seqnum < incoming_seqnum:
   # Duplicate - ignore or log warning
elif msg_seqnum > incoming_seqnum:
   # Gap detected - send ResendRequest (35=2)
   resend_req = fix44.ResendRequest()
   resend_req.setField(fix.BeginSeqNo(incoming_seqnum))
   resend_req.setField(fix.EndSeqNo(msg_seqnum - 1))
   session.send(resend_req)
else:
   # Expected sequence - process normally
   incoming_seqnum += 1
```

**Session Recovery:**
```
┌─────────────────────────────────────────────────────────────┐
│ Gap Detection & Recovery                                    │
├─────────────────────────────────────────────────────────────┤
│ 1. Client detects gap (received SeqNum > expected)         │
│    └─> Send ResendRequest (35=2)                           │
│        ├─> BeginSeqNo = expected sequence                  │
│        └─> EndSeqNo = 0 (all missing messages)             │
│                                                             │
│ 2. Server sends missing messages                           │
│    └─> With PossDupFlag (43=Y) set                         │
│                                                             │
│ 3. Client processes resent messages                        │
│    └─> Update incoming_seqnum                              │
│                                                             │
│ 4. If too many gaps: SequenceReset (35=4)                  │
│    └─> Reset sequence to specific number                   │
│    └─> GapFillFlag determines if reset or gap fill         │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Health Monitoring Loop (Runs every 30 seconds)

```python
while True:
    time.sleep(1)
    
    # Every 30 seconds:
    if now - last_health_log >= 30:
        status = app.get_connection_status()
        
        if not status["connection_healthy"]:
            consecutive_failures += 1
            LOG.warning("Connection unhealthy: %s", status)
            
            if consecutive_failures >= max_consecutive_failures:
                LOG.critical("Too many failures, initiating shutdown")
                break
        else:
            consecutive_failures = 0
            LOG.info("Connection healthy: %s", status)
```

**❌ Current Gap:** No automatic reconnection with exponential backoff

---

## 3. MARKET DATA FLOW

### 3.1 Price Updates (Real-time)

```
┌─────────────────────────────────────────────────────────────┐
│ fromApp(message, sessionID)                                 │
├─────────────────────────────────────────────────────────────┤
│ IF message.header.getField(35) == "W" (MarketDataSnapshot): │
│                                                             │
│   1. Parse bid/ask/last from message                       │
│   2. Update self.last_bid, self.last_ask                   │
│   3. Calculate mid = (bid + ask) / 2                       │
│   4. Update OrderBook (for VPIN calculation)               │
│      └─> orderbook.update(timestamp, bid, ask, volume)     │
│   5. Calculate VPIN metrics                                │
│      └─> vpin_calc.update(orderbook)                       │
│   6. Store last_vpin_stats for features                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Bar Building (Time-based Aggregation)

```
┌─────────────────────────────────────────────────────────────┐
│ BAR CONSTRUCTION THREAD (background)                        │
├─────────────────────────────────────────────────────────────┤
│ Every 1 second:                                             │
│   1. Get current timestamp                                 │
│   2. Determine if new bar should start                     │
│      └─> Based on timeframe_minutes                        │
│                                                             │
│   IF new bar detected:                                     │
│      1. Finalize previous bar                              │
│         ├─> Set high, low, close                           │
│         └─> Set volume, timestamp                          │
│      2. Call on_bar_close(previous_bar)                    │
│      3. Start new bar                                      │
│         └─> Initialize with current price as OHLC          │
│                                                             │
│   ELSE (continuing current bar):                           │
│      1. Update high if price > current_high                │
│      2. Update low if price < current_low                  │
│      3. Update close to latest price                       │
│      4. Accumulate volume                                  │
└─────────────────────────────────────────────────────────────┘
```

**❌ Current Gap:** No `NonRepaintBarAccess` - may be using incomplete bars

---

## 4. ON BAR CLOSE - DECISION PIPELINE

### 4.1 Complete Bar Close Sequence

```
┌─────────────────────────────────────────────────────────────┐
│ on_bar_close(bar)                                           │
├─────────────────────────────────────────────────────────────┤
│ STEP 1: Bar Validation & Storage                           │
│   ├─> Append bar to self.bars (deque)                      │
│   ├─> Update NonRepaintBarAccess (confirmed bar only)      │
│   └─> Check minimum history requirement                    │
│       └─> If len(bars) < MIN_BARS: return (skip trading)   │
│                                                             │
│ STEP 2: Feature Extraction                                 │
│   ├─> Calculate event-relative time features               │
│   │   ├─> Minutes to rollover                              │
│   │   ├─> Session phase (pre-market, active, post)         │
│   │   ├─> High liquidity period flag                       │
│   │   └─> Economic calendar proximity                      │
│   ├─> Calculate path geometry features                     │
│   │   ├─> Price acceleration (gamma)                       │
│   │   ├─> Jerk (rate of acceleration change)               │
│   │   ├─> Runway (predicted price path)                    │
│   │   └─> Feasibility score                                │
│   ├─> Calculate realized volatility                        │
│   │   └─> Rolling standard deviation of returns            │
│   └─> Get VPIN z-score (order flow imbalance)              │
│                                                             │
│ STEP 3: Risk Manager Assessment                            │
│   ├─> Get circuit breaker state                           │
│   │   ├─> Sortino ratio threshold                          │
│   │   ├─> Kurtosis (tail risk)                             │
│   │   ├─> Maximum drawdown                                 │
│   │   └─> Consecutive losses                               │
│   ├─> Make risk decision                                   │
│   │   ├─> Allow normal trading                             │
│   │   ├─> Force position close (emergency exit)            │
│   │   └─> Block new entries                                │
│   └─> Pass breaker state to agents as features             │
│                                                             │
│ STEP 4A: ENTRY DECISION (if no position)                   │
│   ├─> Call policy.decide_entry(bars, features)             │
│   │   └─> TriggerAgent forward pass                        │
│   │       ├─> Prepare state tensor (window × features)     │
│   │       ├─> Include circuit breaker state as features    │
│   │       ├─> For each agent in arena:                     │
│   │       │   └─> Q-values = network(state)                │
│   │       ├─> Aggregate Q-values (weighted average)        │
│   │       ├─> Select action: argmax(aggregated_Q)          │
│   │       └─> Return (action, confidence)                  │
│   │                                                         │
│   ├─> Check action != NO_ENTRY                             │
│   ├─> Check confidence > threshold                         │
│   ├─> Pass to RiskManager for validation                   │
│   ├─> RiskManager checks entry allowed                     │
│   ├─> Store entry_state for experience replay              │
│   └─> If approved: TradeManager executes entry             │
│                                                             │
│ STEP 4B: EXIT DECISION (if has position)                   │
│   ├─> Update position metrics                              │
│   │   ├─> Current unrealized P&L                           │
│   │   ├─> MFE (Maximum Favorable Excursion)                │
│   │   ├─> MAE (Maximum Adverse Excursion)                  │
│   │   └─> Bars held                                        │
│   │                                                         │
│   ├─> Call policy.decide_exit(bars, position_state)        │
│   │   └─> HarvesterAgent forward pass                      │
│   │       ├─> State = market_features + position_features  │
│   │       ├─> For each agent in arena:                     │
│   │       │   └─> Q-values = network(state)                │
│   │       ├─> Aggregate Q-values                           │
│   │       ├─> Select action: argmax(Q)                     │
│   │       └─> Return (HOLD or CLOSE, confidence)           │
│   │                                                         │
│   ├─> If action == CLOSE:                                  │
│   │   └─> Execute exit order                               │
│   └─> Store exit metrics for reward calculation            │
│                                                             │
│ STEP 5: Periodic Training                                  │
│   ├─> Increment bars_since_training                        │
│   └─> If bars_since_training >= training_interval:         │
│       ├─> Call policy.trigger.train_step()                 │
│       ├─> Call policy.harvester.train_step()               │
│       ├─> Update adaptive regularization                   │
│       ├─> Check generalization monitor                     │
│       ├─> Update early stopping                            │
│       └─> Reset bars_since_training = 0                    │
│                                                             │
│ STEP 6: Export Monitoring Data                             │
│   └─> _export_hud_data() (for visualization)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. ORDER EXECUTION FLOW

### 5.1 FIX Protocol Order Lifecycle

The bot uses **FIX Protocol** (Financial Information eXchange) to communicate with cTrader via QuickFIX library.

**Connection Details:**
- **Quote Session**: Market data subscription (MarketDataRequest)
- **Trade Session**: Order execution (NewOrderSingle, ExecutionReport)
- **Protocol**: FIX 4.4
- **Config Files**: `ctrader_quote.cfg`, `ctrader_trade.cfg`

**Key FIX Message Types:**
- **NewOrderSingle (35=D)**: Submit new order
- **ExecutionReport (35=8)**: Order status updates from broker
- **PositionReport (35=AP)**: Position snapshot
- **MarketDataSnapshotFullRefresh (35=W)**: Bid/ask updates

**Order State Machine (FIX OrdStatus):**
```
PENDING_NEW ──> NEW (0) ──> PARTIALLY_FILLED (1) ──> FILLED (2)
     │           │                                        │
     │           └──> CANCELED (4)                       │
     └──> REJECTED (8)                              Position Opened
```

**FIX Message Tags (Common):**
- **Tag 11 (ClOrdID)**: Client order ID (our tracking ID)
- **Tag 37 (OrderID)**: Broker's order ID
- **Tag 39 (OrdStatus)**: Order status (0=NEW, 1=PARTIAL, 2=FILLED, 4=CANCELED, 8=REJECTED)
- **Tag 150 (ExecType)**: Execution type (0=NEW, F=FILL, 4=CANCELED, 8=REJECTED)
- **Tag 54 (Side)**: 1=BUY, 2=SELL
- **Tag 55 (Symbol)**: Trading symbol
- **Tag 38 (OrderQty)**: Order quantity
- **Tag 40 (OrdType)**: 1=MARKET, 2=LIMIT, 3=STOP

### 5.2 TradeManager Architecture (FIX-based)

```
┌─────────────────────────────────────────────────────────────┐
│ TRADE MANAGER (FIX Order & Position Tracker)               │
├─────────────────────────────────────────────────────────────┤
│ Components:                                                 │
│  - Order tracker: Dict[ClOrdID → OrderState]               │
│  - Position tracker: Dict[Symbol → PositionState]          │
│  - FIX session: QuickFIX session for trade messages        │
│  - Message router: Routes ExecutionReports by ClOrdID      │
│                                                             │
│ Order State Tracking (FIX OrdStatus):                      │
│  - PENDING_NEW: Order created, not yet sent                │
│  - NEW (0): Broker acknowledged, active in market          │
│  - PARTIALLY_FILLED (1): Some quantity executed            │
│  - FILLED (2): Fully executed                              │
│  - CANCELED (4): Order canceled                            │
│  - REJECTED (8): Broker rejected order                     │
│                                                             │
│ Position State (from PositionReport):                      │
│  - Symbol: Trading symbol                                  │
│  - LongQty / ShortQty: Position quantities                 │
│  - AvgPx: Average entry price                              │
│  - UnrealizedPnL: Current floating P&L                     │
│  - MFE/MAE: Tracked locally                                │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Entry Order Flow (Agent → RiskManager → TradeManager → Broker)

```
┌─────────────────────────────────────────────────────────────┐
│ TriggerAgent.decide_entry() → (action, confidence, runway) │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ RiskManager.validate_entry(action, confidence)             │
├─────────────────────────────────────────────────────────────┤
│ 1. Check circuit breakers                                  │
│    └─> If critical breaker: REJECT                         │
│ 2. Calculate position size (VaR-based)                     │
│    ├─> Account for current exposure                        │
│    └─> Apply drawdown limits                               │
│ 3. Validate confidence threshold                           │
│ 4. Check maximum position limits                           │
│ 5. Return: (approved: bool, qty: float, reason: str)       │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ TradeManager.place_entry_order(direction, qty)             │
├─────────────────────────────────────────────────────────────┤
│ 1. Generate ClOrdID                                        │
│    └─> clord_id = f"cl_{timestamp}_{counter}"             │
│                                                             │
│ 2. Create OrderState                                       │
│    ├─> symbol, side, qty                                   │
│    ├─> state = PENDING_NEW                                 │
│    └─> created_at = now()                                  │
│                                                             │
│ 3. Store in pending_orders dict                            │
│    └─> pending_orders[clord_id] = order_state             │
│                                                             │
│ 4. Create FIX NewOrderSingle (35=D)                        │
│    ├─> Tag 11 (ClOrdID) = clord_id                         │
│    ├─> Tag 55 (Symbol) = symbol_id                         │
│    ├─> Tag 54 (Side) = 1 (BUY) or 2 (SELL)                │
│    ├─> Tag 38 (OrderQty) = qty                             │
│    ├─> Tag 40 (OrdType) = 1 (MARKET)                       │
│    └─> Tag 60 (TransactTime) = UTC timestamp               │
│                                                             │
│ 5. Send to broker via FIX session                          │
│    └─> Session.sendToTarget(order, trade_sid)              │
│                                                             │
│ 6. Return clord_id for tracking                            │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ BROKER (cTrader via FIX)                                   │
│  - Validates order                                         │
│  - Places in order book (if limit) or executes (if market) │
│  - Sends ExecutionReport (35=8) back                       │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 ExecutionReport Processing (FIX 35=8)

```
┌─────────────────────────────────────────────────────────────┐
│ fromApp() - ExecutionReport (MsgType 35=8)                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Parse FIX tags                                          │
│    ├─> Tag 11 (ClOrdID) - find order in tracker           │
│    ├─> Tag 37 (OrderID) - broker's order ID               │
│    ├─> Tag 39 (OrdStatus) - order status                  │
│    ├─> Tag 150 (ExecType) - execution type                │
│    ├─> Tag 31 (LastPx) - execution price                  │
│    ├─> Tag 32 (LastQty) - executed quantity               │
│    └─> Tag 58 (Text) - rejection reason (if any)          │
│                                                             │
│ 2. Find order by ClOrdID                                   │
│    └─> order = pending_orders.get(clord_id)               │
│                                                             │
│ 3. Route by ExecType (Tag 150)                             │
│    IF ExecType == '0' (NEW):                               │
│       ├─> order_state.state = NEW                          │
│       ├─> order_state.broker_order_id = OrderID            │
│       └─> LOG: "Order accepted by exchange"                │
│                                                             │
│    IF ExecType == 'F' (FILL):                              │
│       ├─> order_state.state = FILLED                       │
│       ├─> order_state.fill_price = LastPx                  │
│       ├─> order_state.fill_qty = CumQty (Tag 14)           │
│       ├─> order_state.filled_at = now()                    │
│       ├─> Create/update Position                           │
│       │   ├─> entry_price = AvgPx                          │
│       │   ├─> quantity = LongQty or ShortQty               │
│       │   ├─> Initialize MFE = 0, MAE = 0                  │
│       │   └─> bars_held = 0                                │
│       ├─> Notify HarvesterAgent (position opened)          │
│       ├─> Export trade to CSV                              │
│       └─> Remove from pending_orders                       │
│                                                             │
│    IF OrdStatus == '1' (PARTIALLY_FILLED):                 │
│       ├─> order_state.state = PARTIALLY_FILLED             │
│       ├─> order_state.filled_qty = CumQty (Tag 14)         │
│       ├─> order_state.remaining_qty = LeavesQty (Tag 151)  │
│       └─> Update position with partial fill                │
│                                                             │
│    IF ExecType == '8' (REJECTED):                          │
│       ├─> order_state.state = REJECTED                     │
│       ├─> order_state.reject_reason = Text (Tag 58)        │
│       ├─> LOG ERROR with reason                            │
│       ├─> Notify RiskManager (potential issue)             │
│       └─> Remove from pending_orders                       │
│                                                             │
│    IF ExecType == '4' (CANCELED):                          │
│       ├─> order_state.state = CANCELED                     │
│       └─> Remove from pending_orders                       │
└─────────────────────────────────────────────────────────────┘
```

### 5.5 Exit Order Flow (Harvester → RiskManager → TradeManager → Broker)

```
┌─────────────────────────────────────────────────────────────┐
│ HarvesterAgent.decide_exit() → (action, confidence, type)  │
│  - action: 0=HOLD, 1=CLOSE                                 │
│  - type: "FULL" | "PARTIAL" | "TRAILING"                   │
│  - fraction: 0.5 (if partial)                              │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ RiskManager.validate_exit(action, type)                    │
├─────────────────────────────────────────────────────────────┤
│ 1. Check circuit breakers                                  │
│    └─> If emergency: override to FULL close                │
│ 2. Validate partial close fraction                         │
│ 3. Check minimum position size after partial               │
│ 4. Return: (approved: bool, volume: int, urgency: str)     │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ TradeManager.execute_exit(qty, urgency, type)              │
├─────────────────────────────────────────────────────────────┤
│ 1. Generate ClOrdID                                        │
│    └─> clord_id = f"exit_{timestamp}_{counter}"           │
│                                                             │
│ 2. Determine close strategy                                │
│    IF type == "FULL":                                      │
│       └─> qty = abs(current_position)                      │
│    IF type == "PARTIAL":                                   │
│       └─> qty = abs(current_position) * fraction           │
│    IF type == "TRAILING":                                  │
│       ├─> Create trailing stop state                       │
│       ├─> peak_price = current MFE                         │
│       ├─> trail_distance = X bps                           │
│       └─> Monitor each bar, trigger FULL close when breach │
│                                                             │
│ 3. Create FIX NewOrderSingle (closing order)               │
│    ├─> Tag 11 (ClOrdID) = clord_id                         │
│    ├─> Tag 55 (Symbol) = symbol_id                         │
│    ├─> Tag 54 (Side) = OPPOSITE of current position        │
│    │   └─> If LONG: Side=2 (SELL), if SHORT: Side=1 (BUY)│
│    ├─> Tag 38 (OrderQty) = qty                             │
│    ├─> Tag 40 (OrdType) = 1 (MARKET)                       │
│    └─> Tag 60 (TransactTime) = UTC timestamp               │
│                                                             │
│ 4. Send to broker via FIX session                          │
│    └─> Session.sendToTarget(order, trade_sid)              │
│                                                             │
│ 5. Position cleanup on ExecutionReport (FILL)              │
│    ├─> Parse PositionReport to confirm close               │
│    ├─> Calculate final P&L                                 │
│    ├─> Record final MFE, MAE, bars_held                    │
│    ├─> Add harvester CLOSE experience with capture ratio   │
│    ├─> Export trade record                                 │
│    └─> Clear position tracker (or update if partial)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. POSITION MONITORING & EXIT

### 6.1 Position Report Processing

```
┌─────────────────────────────────────────────────────────────┐
│ on_position_report(message)                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. Parse position details                                  │
│    ├─> PositionID                                          │
│    ├─> Symbol                                              │
│    ├─> LongQty / ShortQty                                  │
│    ├─> AvgPx (average entry price)                         │
│    └─> UnrealizedPnL                                       │
│                                                             │
│ 2. Update internal position tracking                       │
│    ├─> self.cur_pos = net_position                         │
│    ├─> self.entry_price = AvgPx                            │
│    └─> Update MFE/MAE if needed                            │
│                                                             │
│ 3. Handle position close (LongQty = 0, ShortQty = 0)       │
│    IF position just closed:                                │
│       ├─> Calculate final P&L                              │
│       ├─> Record exit price, time                          │
│       ├─> Compute path metrics (MFE, MAE, bars held)       │
│       ├─> Calculate shaped rewards                         │
│       │   ├─> Runway utilization (TriggerAgent)            │
│       │   ├─> Capture efficiency (HarvesterAgent)          │
│       │   ├─> WTL penalty (if applicable)                  │
│       │   └─> Opportunity cost                             │
│       ├─> Add experience to replay buffers                 │
│       │   ├─> TriggerAgent.buffer.add(entry_experience)    │
│       │   └─> HarvesterAgent.buffer.add(exit_experience)   │
│       ├─> Update PerformanceTracker                        │
│       ├─> Update CircuitBreakers                           │
│       ├─> Export trade to CSV                              │
│       └─> Clear position state                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. EXPERIENCE REPLAY & LEARNING

### 7.1 Experience Buffer Addition

```
┌─────────────────────────────────────────────────────────────┐
│ TriggerAgent.add_experience(...)                           │
├─────────────────────────────────────────────────────────────┤
│ 1. Validate experience                                     │
│    ├─> state.shape == (window, n_features)                 │
│    ├─> action in valid_actions                             │
│    ├─> reward is finite (not NaN/Inf)                      │
│    └─> next_state.shape == state.shape                     │
│                                                             │
│ 2. Create experience tuple                                 │
│    └─> (state, action, reward, next_state, done, regime)   │
│                                                             │
│ 3. Add to prioritized buffer                               │
│    ├─> Initial priority = max_priority                     │
│    ├─> Update sum_tree                                     │
│    └─> Increment total_added counter                       │
│                                                             │
│ 4. Apply regime boost                                      │
│    └─> If regime matches current: priority *= 1.5          │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Training Step (Every N bars)

```
┌─────────────────────────────────────────────────────────────┐
│ Agent.train_step()                                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Check readiness                                         │
│    └─> If buffer_size < min_experiences: return None       │
│                                                             │
│ 2. Sample batch from prioritized buffer                    │
│    ├─> Select batch_size indices (proportional to priority)│
│    ├─> Retrieve experiences                                │
│    └─> Convert to tensors                                  │
│                                                             │
│ 3. Forward pass (online network)                           │
│    ├─> Q_online = online_net(states)                       │
│    └─> Q_current = Q_online[range(batch), actions]         │
│                                                             │
│ 4. Forward pass (target network) - DDQN                    │
│    ├─> Q_online_next = online_net(next_states)             │
│    ├─> best_actions = argmax(Q_online_next, dim=1)         │
│    ├─> Q_target_next = target_net(next_states)             │
│    └─> Q_next = Q_target_next[range(batch), best_actions]  │
│                                                             │
│ 5. Compute target Q-values                                 │
│    └─> Q_target = rewards + gamma * Q_next * (1 - dones)   │
│                                                             │
│ 6. Compute TD-errors                                       │
│    └─> td_errors = Q_target - Q_current                    │
│                                                             │
│ 7. Update priorities in buffer                             │
│    └─> buffer.update_priorities(indices, |td_errors|)      │
│                                                             │
│ 8. Compute loss (with adaptive regularization)             │
│    ├─> loss_td = Huber(Q_current, Q_target)                │
│    ├─> loss_l2 = l2_lambda * sum(w^2)                      │
│    └─> total_loss = loss_td + loss_l2                      │
│                                                             │
│ 9. Backpropagation                                         │
│    ├─> optimizer.zero_grad()                               │
│    ├─> total_loss.backward()                               │
│    ├─> Clip gradients (prevent explosion)                  │
│    └─> optimizer.step()                                    │
│                                                             │
│ 10. Soft update target network (every N steps)             │
│     └─> target_params = tau*online_params + (1-tau)*target │
│                                                             │
│ 11. Update statistics                                      │
│     ├─> Increment training_steps                           │
│     ├─> Update total_sampled                               │
│     ├─> Track mean/max TD-error                            │
│     └─> Return metrics dict                                │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Overfitting Detection (After Training)

```
┌─────────────────────────────────────────────────────────────┐
│ GeneralizationMonitor.check_overfitting()                  │
├─────────────────────────────────────────────────────────────┤
│ 1. Compare training vs. live performance                   │
│    ├─> train_sharpe (from experience buffer)               │
│    ├─> live_sharpe (from actual trades)                    │
│    └─> gap = train_sharpe - live_sharpe                    │
│                                                             │
│ 2. Detect distribution shift                               │
│    ├─> Compare feature distributions (KL divergence)       │
│    └─> Flag if shift > threshold                           │
│                                                             │
│ 3. Check ensemble disagreement                             │
│    ├─> Variance of agent predictions                       │
│    └─> High variance = overfitting signal                  │
│                                                             │
│ IF overfitting detected:                                   │
│    ├─> AdaptiveRegularization.increase_regularization()    │
│    ├─> Increase L2 lambda                                  │
│    ├─> Increase dropout rate                               │
│    ├─> Decrease learning rate                              │
│    └─> EarlyStopping.check() (restore checkpoint if worse) │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. PERSISTENCE & STATE MANAGEMENT

### 8.1 Periodic State Saves (Every N bars or trades)

```
┌─────────────────────────────────────────────────────────────┐
│ BotPersistenceManager.save_agent_state()                   │
├─────────────────────────────────────────────────────────────┤
│ 1. Collect state from all components                       │
│    ├─> TriggerAgent network weights                        │
│    ├─> HarvesterAgent network weights                      │
│    ├─> Experience buffer (most recent N experiences)       │
│    ├─> LearnedParameters (all adaptive params)             │
│    ├─> PerformanceTracker (trade history)                  │
│    └─> Training statistics (steps, epsilon, etc.)          │
│                                                             │
│ 2. Serialize to dict                                       │
│    └─> Convert tensors to CPU, then numpy                  │
│                                                             │
│ 3. Atomic write with checksum                              │
│    ├─> Write to temp file                                  │
│    ├─> Compute CRC32 checksum                              │
│    ├─> Fsync to disk                                       │
│    ├─> Rename to target (atomic operation)                 │
│    └─> Delete old backup                                   │
└─────────────────────────────────────────────────────────────┘
```

**❌ Current Gap:** No `JournaledPersistence` - crash during save could corrupt state

### 8.2 State Recovery on Startup

```
┌─────────────────────────────────────────────────────────────┐
│ BotPersistenceManager.load_agent_state()                   │
├─────────────────────────────────────────────────────────────┤
│ 1. Check if state file exists                              │
│                                                             │
│ 2. Validate checksum                                       │
│    ├─> Load data                                           │
│    ├─> Compute checksum                                    │
│    ├─> Compare with stored checksum                        │
│    └─> If mismatch: try backup, else fail                  │
│                                                             │
│ 3. Deserialize and restore                                 │
│    ├─> Load network weights into agents                    │
│    ├─> Restore experience buffers                          │
│    ├─> Restore learned parameters                          │
│    └─> Restore performance history                         │
│                                                             │
│ 4. Validate restored state                                 │
│    ├─> Check parameter ranges                              │
│    ├─> Verify buffer integrity                             │
│    └─> Confirm network dimensions                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. MONITORING & HUD DATA EXPORT

### 9.1 HUD Data Export (Every bar)

```
┌─────────────────────────────────────────────────────────────┐
│ _export_hud_data()                                          │
├─────────────────────────────────────────────────────────────┤
│ 1. System Status                                           │
│    ├─> Connection health                                   │
│    ├─> Uptime                                              │
│    └─> Last update timestamp                               │
│                                                             │
│ 2. Position Information                                    │
│    ├─> Current position (LONG/SHORT/FLAT)                  │
│    ├─> Entry price, quantity                               │
│    ├─> Unrealized P&L                                      │
│    ├─> MFE, MAE                                            │
│    └─> Bars held                                           │
│                                                             │
│ 3. Performance Snapshot                                    │
│    ├─> Total P&L                                           │
│    ├─> Win rate                                            │
│    ├─> Sharpe ratio                                        │
│    ├─> Sortino ratio                                       │
│    ├─> Max drawdown                                        │
│    └─> Number of trades                                    │
│                                                             │
│ 4. Training Statistics                                     │
│    ├─> Trigger buffer size                                 │
│    ├─> Harvester buffer size                               │
│    ├─> Total training steps                                │
│    ├─> Last loss values                                    │
│    ├─> Epsilon (exploration rate)                          │
│    └─> Arena diversity scores                              │
│                                                             │
│ 5. Risk Metrics                                            │
│    ├─> VaR estimate                                        │
│    ├─> VPIN z-score                                        │
│    ├─> Kurtosis                                            │
│    └─> Circuit breaker status                              │
│                                                             │
│ 6. Market Features                                         │
│    ├─> Current price, bid, ask                             │
│    ├─> Realized volatility                                 │
│    ├─> Session phase                                       │
│    └─> High liquidity flag                                 │
│                                                             │
│ 7. Write JSON files to hud_data/                           │
│    ├─> system_status.json                                  │
│    ├─> position_info.json                                  │
│    ├─> performance_snapshot.json                           │
│    ├─> training_stats.json                                 │
│    ├─> risk_metrics.json                                   │
│    └─> market_features.json                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. GRACEFUL SHUTDOWN

### 10.1 Shutdown Sequence

```
┌─────────────────────────────────────────────────────────────┐
│ signal_handler(SIGTERM/SIGINT)                             │
├─────────────────────────────────────────────────────────────┤
│ 1. Set shutdown flag                                       │
│    └─> app._shutdown_requested = True                      │
│                                                             │
│ 2. Stop accepting new signals                              │
│    └─> Close any open positions (optional)                 │
│                                                             │
│ 3. Save state                                              │
│    ├─> BotPersistence.save_agent_state()                   │
│    ├─> Save learned parameters                             │
│    └─> Export final performance report                     │
│                                                             │
│ 4. Stop FIX sessions                                       │
│    ├─> Send logout messages                                │
│    ├─> initiator_quote.stop()                              │
│    └─> initiator_trade.stop()                              │
│                                                             │
│ 5. Wait for graceful disconnect                            │
│    └─> Timeout: 10 seconds                                 │
│                                                             │
│ 6. Log final statistics                                    │
│    ├─> Total runtime                                       │
│    ├─> Total trades                                        │
│    ├─> Final P&L                                           │
│    └─> Connection statistics                               │
│                                                             │
│ 7. Exit cleanly                                            │
│    └─> sys.exit(0)                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. CRITICAL GAPS IN CURRENT FLOW

### 11.1 Missing Safety Components

| # | Component | Impact | Current Status |
|---|-----------|--------|----------------|
| 1 | **ColdStartManager** | No graduated warm-up | ❌ MISSING |
| 2 | **FeedbackLoopBreaker** | Can get stuck in bad states | ❌ MISSING |
| 3 | **RewardIntegrityMonitor** | Agents could game rewards | ❌ MISSING |
| 4 | **JournaledPersistence** | Crash = state corruption | ❌ MISSING |
| 5 | **ParameterStaleness** | Stale params not detected | ❌ MISSING |
| 6 | **BrokerExecutionModel** | Slippage not realistic | ❌ MISSING |
| 7 | **InitGate** | No enforced init order | ❌ MISSING |
| 8 | **SafeArray** | Array bounds not checked | ❌ MISSING |

### 11.2 Recommended Fix Sequence

```
Priority 0 (Week 1): Foundation Safety
├─> 1. ColdStartManager (graduated phases)
├─> 2. InitGate (proper initialization order)
└─> 3. SafeArray (defensive array access)

Priority 1 (Week 1-2): State Integrity
├─> 4. JournaledPersistence (crash recovery)
├─> 5. FeedbackLoopBreaker (escape bad states)
└─> 6. RewardIntegrityMonitor (detect reward gaming)

Priority 2 (Week 2-3): Parameter Management
├─> 7. ParameterStaleness (detect stale params)
└─> 8. BrokerExecutionModel (realistic slippage)

Priority 3 (Week 3-4): Integration & Testing
├─> 9. Comprehensive unit tests
├─> 10. Integration test suite
└─> 11. End-to-end flow validation
```

---

## 12. SYSTEM FLOW DIAGRAM

```
┌──────────────┐
│   STARTUP    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│ Initialize Components (InitGate enforced)    │
│ ├─> SafeMath, Logging                       │
│ ├─> Persistence (Journaled)                 │
│ ├─> LearnedParameters                       │
│ ├─> ColdStartManager ← NEW                  │
│ ├─> Agents (Trigger + Harvester)            │
│ ├─> Risk (CircuitBreakers, VaR)             │
│ └─> Monitoring (Performance, HUD)           │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ Connect to cTrader (FIX Protocol)            │
│ ├─> Quote session (market data)             │
│ └─> Trade session (order execution)         │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ Market Data Stream (continuous)              │
│ ├─> Update OrderBook                        │
│ ├─> Calculate VPIN                          │
│ └─> Build bars (time-based aggregation)     │
└──────────────┬───────────────────────────────┘
               │
               ▼
       ┌───────────────┐
       │  ON BAR CLOSE │ ← Main Decision Loop
       └───────┬───────┘
               │
       ┌───────▼────────────────────────────────┐
       │ Check ColdStartManager phase          │
       │ └─> Observation? Skip trading         │
       └───────┬───────────────────────────────┘
               │
       ┌───────▼────────────────────────────────┐
       │ Extract Features                      │
       │ ├─> Event-relative time               │
       │ ├─> Path geometry                     │
       │ ├─> VPIN                              │
       │ └─> Technical indicators              │
       └───────┬───────────────────────────────┘
               │
       ┌───────▼────────────────────────────────┐
       │ Check CircuitBreakers                 │
       │ └─> If tripped: halt trading          │
       └───────┬───────────────────────────────┘
               │
               ├─────────────┬─────────────────┐
               ▼             ▼                 ▼
        ┌───────────┐  ┌──────────┐   ┌──────────────┐
        │ NO        │  │ LONG     │   │ SHORT        │
        │ POSITION  │  │ POSITION │   │ POSITION     │
        └─────┬─────┘  └────┬─────┘   └────┬─────────┘
              │             │              │
              │             └──────┬───────┘
              │                    │
              ▼                    ▼
    ┌──────────────────┐  ┌────────────────────┐
    │ TriggerAgent     │  │ HarvesterAgent     │
    │ (Entry decision) │  │ (Exit decision)    │
    └────────┬─────────┘  └────────┬───────────┘
             │                     │
             ▼                     ▼
    ┌──────────────────┐  ┌────────────────────┐
    │ Execute Entry    │  │ Execute Exit       │
    │ (if signal)      │  │ (if signal)        │
    └────────┬─────────┘  └────────┬───────────┘
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
             ┌────────────────────────┐
             │ On Trade Close:        │
             │ ├─> Calculate rewards  │
             │ ├─> Add experience     │
             │ ├─> Update performance │
             │ └─> Check overfitting  │
             └────────┬───────────────┘
                      │
                      ▼
             ┌────────────────────────┐
             │ Periodic Training      │
             │ (every N bars)         │
             │ ├─> Sample batch (PER) │
             │ ├─> DDQN forward/back  │
             │ ├─> Update priorities  │
             │ └─> Soft update target │
             └────────┬───────────────┘
                      │
                      ▼
             ┌────────────────────────┐
             │ Overfitting Check      │
             │ ├─> Train/live gap     │
             │ ├─> Distribution shift │
             │ ├─> Ensemble disagree  │
             │ └─> Adapt reg if needed│
             └────────┬───────────────┘
                      │
                      ▼
             ┌────────────────────────┐
             │ FeedbackLoopBreaker    │
             │ └─> Reset if degraded  │
             └────────┬───────────────┘
                      │
                      ▼
             ┌────────────────────────┐
             │ Export HUD Data        │
             └────────┬───────────────┘
                      │
                      └─────► Loop continues...
```

---

## 13. NEXT STEPS

1. **Implement ColdStartManager** - Prevents trading before sufficient data
2. **Add InitGate** - Enforces proper initialization order
3. **Implement JournaledPersistence** - Prevents state corruption
4. **Add FeedbackLoopBreaker** - Escapes degraded performance
5. **Expand test coverage** - Unit + integration tests
6. **Add RewardIntegrityMonitor** - Detect reward gaming
7. **Implement ParameterStaleness** - Refresh stale parameters
8. **Create comprehensive logs** - Audit trail for debugging

---

**END OF SYSTEM FLOW DOCUMENT**

*This document should be referenced for understanding execution sequence and debugging flow issues.*
