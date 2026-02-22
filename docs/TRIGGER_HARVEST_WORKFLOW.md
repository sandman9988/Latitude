# Trigger/Harvest Trading Workflow

## Architecture Overview

The bot uses a **Dual-Agent Architecture**:

1. **TriggerAgent** - Opens positions (epsilon-greedy exploration)
2. **HarvesterAgent** - Closes positions (MFE/MAE-based exits)

## Trade Lifecycle

### Phase 1: FLAT State - Trigger Agent Active

**Condition**: `has_any_open_positions() == False`

**Flow** ([ctrader_ddqn_paper.py](src/core/ctrader_ddqn_paper.py#L2615-2720)):

```
on_bar_close() →
├── Check: has_any_open_positions()? NO
├── Step 1: Position check ✓
├── Step 2: Circuit breakers check
│   └── If any tripped → ABORT (no entry)
├── Step 3: Depth check (order book liquidity)
│   └── If too thin → ABORT (no entry)
├── Step 4: policy.decide_entry()
│   ├── TriggerAgent.decide_entry()
│   ├── Returns: action, confidence, runway
│   │   action=0: NO_ENTRY (HOLD)
│   │   action=1: LONG
│   │   action=2: SHORT
│   ├── Store entry_state for online learning
│   └── Log trigger decision
└── Step 5: Execute entry order
    └── If action != 0 → submit_market_order()
```

**Key Methods**:
- [ctrader_ddqn_paper.py#L2609](src/core/ctrader_ddqn_paper.py#L2609): `has_positions = self.trade_integration.has_any_open_positions()`
- [ctrader_ddqn_paper.py#L2656](src/core/ctrader_ddqn_paper.py#L2656): `action, confidence, runway = self.policy.decide_entry(...)`

### Phase 2: IN POSITION - Harvester Agent Active

**Condition**: `has_any_open_positions() == True`

**Flow** ([ctrader_ddqn_paper.py](src/core/ctrader_ddqn_paper.py#L2720-2770)):

```
on_bar_close() →
├── Check: has_any_open_positions()? YES
├── Step 1: policy.decide_exit()
│   ├── HarvesterAgent.decide_exit()
│   ├── Tracks MFE/MAE continuously
│   ├── Returns: exit_action, exit_conf
│   │   exit_action=0: HOLD (keep position)
│   │   exit_action=1: CLOSE
│   └── Log harvester decision
└── Step 2: Execute exit order
    └── If exit_action == 1 → close_position()
```

**Key Methods**:
- [ctrader_ddqn_paper.py#L2725](src/core/ctrader_ddqn_paper.py#L2725): `exit_action, exit_conf = self.policy.decide_exit(...)`
- [trade_manager_integration.py#L590](src/core/trade_manager_integration.py#L590): `close_position(broker_ticket)`

### Phase 3: Position Tracking (Hedging Mode)

**Key Data Structures**:

1. **`mfe_mae_trackers`** - Dict[position_id → MAETracker]
   - Tracks entry_price, direction, mfe, mae per position
   - Initialized EMPTY `{}` (not with default tracker)
   - Updated on every market data quote

2. **`position_tickets`** - Dict[broker_ticket → position_id]
   - Maps broker ticket numbers to internal position IDs
   - Required for hedging mode (close by ticket)
   - Cleaned up when position closes

**Position State Check** ([trade_manager_integration.py](src/core/trade_manager_integration.py#L187-203)):

```python
def has_any_open_positions(self) -> bool:
    """Check if ANY positions exist (hedging-aware)."""
    # Check 1: Tracker dict count
    tracker_count = len(self.app.mfe_mae_trackers)
    LOG.info("[POSITION-CHECK] mfe_mae_trackers count=%d", tracker_count)
    
    # Check 2: Position tickets
    ticket_count = len(self.position_tickets)
    LOG.info("[POSITION-CHECK] position_tickets=%s", self.position_tickets)
    
    # Check 3: Net quantity (fallback)
    position = self.trade_manager.get_position()
    net_qty = abs(position.net_qty) if position else 0.0
    
    # Any non-zero count = has positions
    has_positions = tracker_count > 0 or ticket_count > 0 or net_qty > 0.0001
    LOG.info("[INTEGRATION] %s - returning %s", 
             "Positions found" if has_positions else "No positions found",
             has_positions)
    return has_positions
```

---

## State Persistence & Recovery

### Files

1. **`data/state/trade_integration_BTCUSD.json`** - Position/tracker state
   - Position quantities (long_qty, short_qty, net_qty)
   - Active MAE/MFE trackers
   - Position ticket mappings
   - Trailing stop state
   - ✅ **Float precision now normalized to 8 decimals**

2. **`logs/audit/trade_audit.jsonl`** - Trade execution audit trail
   - Every order submission
   - Every execution report
   - Position updates

3. **`logs/audit/decisions.jsonl`** - DDQN decision log
   - Trigger decisions (entry)
   - Harvester decisions (exit)
   - State vectors, Q-values, epsilon

### Persistence Flow

**Every Position Change** ([trade_manager_integration.py#L713](src/core/trade_manager_integration.py#L713)):

```python
def _persist_state(self):
    """Save position and tracker state to disk."""
    position_data = self.trade_manager.position.to_dict()
    
    # Build active_trackers dict
    active_trackers = {}
    for pos_id, tracker in self.app.mfe_mae_trackers.items():
        if entry_price > 0 and direction != 0:
            active_trackers[pos_id] = {
                "entry_price": round(entry_price, 8),  # ✅ Fixed
                "direction": direction,
                "mfe": round(tracker.mfe, 8),           # ✅ Fixed
                "mae": round(tracker.mae, 8),           # ✅ Fixed
            }
    
    # Build position_tickets dict
    position_tickets = {}
    for ticket, pos_id in self.position_tickets.items():
        # Serialize ticket → position mapping
        
    state = {
        "trailing_stop_active": self.trailing_stop_active,
        "trailing_stop_distance_pct": round(self.trailing_stop_distance_pct, 8),
        "highest_price_since_entry": round(self.highest_price_since_entry, 8),
        "entry_price": round(self.entry_price, 8),
        "position_direction": self.position_direction,
        "position": position_data,          # Position.to_dict()
        "active_trackers": active_trackers,
        "position_tickets": position_tickets,
        "persisted_at": utc_now().isoformat(),
    }
    
    self.persistence.save_json(state, self.state_filename, create_backup=True)
```

### Recovery Flow

**On Bot Restart** ([trade_manager_integration.py#L777](src/core/trade_manager_integration.py#L777)):

```python
def _recover_state(self) -> bool:
    """Recover position and trailing stop state after crash/restart."""
    state = self.persistence.load_json(self.state_filename)
    if not state:
        return False
        
    # Restore position
    position_data = state.get("position")
    if position_data:
        self.trade_manager.position = Position.from_dict(position_data)
    
    # Restore MAE/MFE trackers
    active_trackers = state.get("active_trackers", {})
    for pos_id, tracker_data in active_trackers.items():
        tracker = MAETracker(
            entry_price=tracker_data["entry_price"],
            direction=tracker_data["direction"]
        )
        tracker.mfe = tracker_data.get("mfe", 0.0)
        tracker.mae = tracker_data.get("mae", 0.0)
        self.app.mfe_mae_trackers[pos_id] = tracker
    
    # Restore position tickets
    position_tickets = state.get("position_tickets", {})
    for ticket, pos_id in position_tickets.items():
        self.position_tickets[ticket] = pos_id
    
    # Restore trailing stop
    self.trailing_stop_active = state.get("trailing_stop_active", False)
    self.entry_price = state.get("entry_price")
    
    LOG.info("[RECOVERY] ✓ Restored: trackers=%d tickets=%d",
             len(active_trackers), len(position_tickets))
    return True
```

---

## Hedging Mode Position Closing

**FIX Protocol**: Must specify **PosMaintRptID (tag 721)** to close specific position by ticket.

**Close Order Submission** ([trade_manager.py#L368-415](src/core/trade_manager.py#L368-415)):

```python
def submit_market_order(self, side, quantity, position_ticket=None):
    """Submit FIX order with optional position_ticket for hedging mode."""
    order = fix.Message()
    order.setField(fix.ClOrdID(order_id))
    order.setField(fix.Side(fix_side))
    order.setField(fix.OrderQty(quantity))
    
    # HEDGING MODE: Add position ticket to close specific position
    if position_ticket:
        order.setField(fix.StringField(721, str(position_ticket)))
        LOG.info("[HEDGING] Closing position ticket=%s", position_ticket)
    
    fix.Session.sendToTarget(order, self.session_id)
```

**Integration Layer** ([trade_manager_integration.py#L590-615](src/core/trade_manager_integration.py#L590-615)):

```python
def close_position(self, broker_ticket: str):
    """Close specific position by broker ticket (hedging mode)."""
    # Get position details from tracker
    pos_id = self.position_tickets.get(broker_ticket)
    tracker = self.app.mfe_mae_trackers.get(pos_id)
    
    # Determine close side
    close_side = "SELL" if tracker.direction == 1 else "BUY"
    
    # Submit close order WITH position_ticket
    order_id = self.trade_manager.submit_market_order(
        side=close_side,
        quantity=self.app.qty,
        position_ticket=broker_ticket  # ✅ Critical for hedging
    )
```

---

## Current Status (Jan 13, 2026 14:47 UTC)

### Bot State
- **Running**: PID 1929407
- **FLAT**: No open positions (trackers=0, tickets=0)
- **Trigger Agent**: Active, making decisions every bar
- **Action**: Currently choosing action=0 (NO_ENTRY/HOLD)
- **Confidence**: 0.00 (DDQN not confident in entry)
- **Bars**: 13 bars in memory
- **Health**: QUOTE=OK, TRADE=OK, uptime=10.1min

### Recent Activity
```
2026-01-13 14:47:00 [FLAT: Check for entry] has_positions=False
2026-01-13 14:47:00 [FLOW-TRACE] Step 2 PASSED: No circuit breakers tripped
2026-01-13 14:47:00 [FLOW-TRACE] Step 3 PASSED: Depth check OK
2026-01-13 14:47:00 [TRIGGER: action=0 conf=0.00 runway=0.0000 feas=0.50]
2026-01-13 14:47:00 [FLOW-ABORT] No action needed (desired=0 equals cur_pos=0)
```

### Why No Trades?
1. **DDQN Model**: Still learning, outputting low confidence (0.00)
2. **Epsilon-Greedy**: Need to see EXPLORE logs to verify random action selection
3. **Action=0**: HOLD decision is valid - no entry signal yet
4. **Circuit Breakers**: All OK (not blocking trades)
5. **Depth Check**: Passing (order book has liquidity)

### Next Steps to Get Trading
1. **Increase Epsilon**: Force more exploration to build experience buffer
2. **Check Epsilon Value**: Verify epsilon-greedy is active
3. **Wait for Market Conditions**: Bot may be waiting for better entry signals
4. **Monitor Decision Log**: Check if actions 1/2 (LONG/SHORT) appear

---

## Verification Commands

```bash
# Check if bot is trading
tail -f logs/ctrader/ctrader_*.log | grep -E "TRIGGER|HARVEST|EXPLORE|EXPLOIT"

# Check position state
cat data/state/trade_integration_BTCUSD.json | jq '.data | {net_qty: .position.net_qty, trackers: .active_trackers, tickets: .position_tickets}'

# Check decision distribution
tail -100 logs/audit/decisions.jsonl | jq -r '.action' | sort | uniq -c

# Check epsilon value
tail -200 logs/ctrader/ctrader_*.log | grep -i epsilon | tail -5

# Monitor order flow
tail -f logs/audit/trade_audit.jsonl | jq -r '[.timestamp, .event_type, .side, .quantity] | @tsv'
```

---

## Summary

✅ **WORKING**:
- MAE/MFE tracker initialization (empty dict `{}`)
- Position state detection (hedging-aware)
- Float precision in JSON (8 decimals)
- State persistence and recovery
- Trigger/Harvest agent switching
- Circuit breaker checks
- Depth checks
- Emergency close infrastructure

🔧 **CURRENT BEHAVIOR**:
- Bot is FLAT (no positions)
- TriggerAgent active and deciding
- Currently choosing HOLD (action=0)
- No entry signal from DDQN yet
- Waiting for confidence > 0 or epsilon exploration

⏳ **PENDING**:
- Close 6 orphaned positions (9663, 9727, 9728, 9782, 9783, 9786)
- Verify epsilon-greedy exploration is active
- Build experience buffer for training
- Test emergency close script with real positions
