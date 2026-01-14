# Multi-Position Support Analysis & Implementation Plan
**Date**: 2026-01-10  
**Scope**: Enable simultaneous multiple trades with proper tracking

---

## CURRENT LIMITATIONS

### ❌ Single Position Design Flaws

#### 1. **Single MFE/MAE Tracker**
```python
# Current (BROKEN for multiple positions):
self.mfe_mae_tracker = MFEMAETracker()  # Only ONE tracker
self.mfe_mae_tracker.start_tracking(entry_price, direction)
# New trade overwrites previous trade's tracking!
```

#### 2. **Single Path Recorder**
```python
# Current (BROKEN for multiple positions):
self.path_recorder = PathRecorder()  # Only ONE recorder
self.path_recorder.start_recording(entry_time, entry_price, direction)
# New path overwrites previous path!
```

#### 3. **Single Entry State**
```python
# Current (BROKEN for multiple positions):
self.entry_state = None  # Only ONE entry state
self.entry_action = None  # Only ONE action
# Multiple orders can't track their individual states!
```

#### 4. **Scalar Position Tracking**
```python
# Current (LIMITED):
self.cur_pos = 1  # +1, 0, or -1
# Can't track 2 LONGs, 1 LONG + 1 SHORT, etc.
```

#### 5. **Single Trade Entry Time**
```python
# Current (BROKEN for multiple positions):
self.trade_entry_time = None  # Only ONE timestamp
# Can't track multiple entry times!
```

---

## IMPACT ASSESSMENT

### Scenario 1: Multiple Sequential Orders
```
T1: Buy 0.10 @ $60,000 → Tracker starts
T2: Buy 0.10 @ $61,000 → Tracker RESETS (T1 data LOST!)
T3: Sell 0.10 → Which position closed? MFE/MAE wrong!
```
**Result**: First trade's MFE/MAE completely lost, path recording broken

### Scenario 2: Hedge Positions (LONG + SHORT)
```
T1: Buy 0.10 @ $60,000 → cur_pos = 1
T2: Sell 0.10 @ $61,000 → cur_pos = 0 (WRONG! Should be hedged)
```
**Result**: System thinks position closed when actually hedged

### Scenario 3: Scale In/Out
```
T1: Buy 0.05 @ $60,000 → Track MFE/MAE
T2: Buy 0.05 @ $61,000 → MFE/MAE reset (avg price wrong)
T3: Sell 0.05 → Partial exit (which 0.05 closed?)
```
**Result**: Can't attribute P&L correctly to individual entries

---

## REQUIRED CHANGES

### 1. Position-Keyed Tracking

```python
# Replace single trackers with dictionaries keyed by position ID

# Before:
self.mfe_mae_tracker = MFEMAETracker()

# After:
self.mfe_mae_trackers: dict[str, MFEMAETracker] = {}  # position_id -> tracker

# Before:
self.path_recorder = PathRecorder()

# After:
self.path_recorders: dict[str, PathRecorder] = {}  # position_id -> recorder

# Before:
self.entry_state = None
self.entry_action = None

# After:
self.entry_states: dict[str, Any] = {}  # order_id -> state
self.entry_actions: dict[str, int] = {}  # order_id -> action
```

### 2. Position ID Generation

```python
def _generate_position_id(self, order: Order) -> str:
    """
    Generate unique position ID from order.
    
    For non-hedged accounts: Use symbol (only one position per symbol)
    For hedged accounts: Use PosMaintRptID from broker
    """
    if order.pos_maint_rpt_id:
        return order.pos_maint_rpt_id  # Broker-assigned hedge ID
    else:
        return f"{self.symbol_id}"  # Symbol-level tracking
```

### 3. Enhanced TradeManager Position Tracking

```python
# TradeManager needs to track individual positions
class Position:
    symbol: str
    position_id: str  # NEW: Unique ID
    long_qty: float
    short_qty: float
    net_qty: float
    avg_price: float  # NEW: Average entry price
    realized_pnl: float  # NEW: Closed P&L
    unrealized_pnl: float  # NEW: Open P&L
    
    # Multiple entries
    entries: list[dict]  # [{qty, price, time, order_id}, ...]
```

### 4. Order → Position Mapping

```python
class TradeManager:
    def __init__(self, ...):
        # Track which orders contributed to which positions
        self.order_to_position: dict[str, str] = {}  # order_id -> position_id
        self.positions: dict[str, Position] = {}  # position_id -> Position
```

### 5. Callback Enhancement

```python
def on_order_filled(self, order: Order):
    """Enhanced with position ID"""
    position_id = self._get_or_create_position_id(order)
    
    # Create dedicated tracker for this position
    if position_id not in self.app.mfe_mae_trackers:
        self.app.mfe_mae_trackers[position_id] = MFEMAETracker()
    
    # Start tracking this specific position
    self.app.mfe_mae_trackers[position_id].start_tracking(
        order.avg_price,
        direction=1 if order.side == Side.BUY else -1
    )
    
    # Similar for path recorder
    if position_id not in self.app.path_recorders:
        self.app.path_recorders[position_id] = PathRecorder()
    
    self.app.path_recorders[position_id].start_recording(
        order.filled_at,
        order.avg_price,
        direction=1 if order.side == Side.BUY else -1
    )
```

### 6. Update Loop Changes

```python
# Before (single position):
if self.cur_pos != 0:
    self.mfe_mae_tracker.update(mid)
    self.path_recorder.add_bar(bar)

# After (multiple positions):
for position_id, position in self.trade_integration.trade_manager.positions.items():
    if position.net_qty != 0:
        if position_id in self.mfe_mae_trackers:
            self.mfe_mae_trackers[position_id].update(mid)
        if position_id in self.path_recorders:
            self.path_recorders[position_id].add_bar(bar)
```

### 7. Position Close Detection

```python
def on_position_report(self, msg: fix.Message):
    """Enhanced to detect partial/full closes"""
    position_id = self._extract_position_id(msg)
    old_qty = self.positions[position_id].net_qty if position_id in self.positions else 0
    new_qty = new_net_qty
    
    # Full close
    if old_qty != 0 and abs(new_qty) < min_qty:
        self._handle_position_close(position_id, full_close=True)
    
    # Partial close
    elif abs(new_qty) < abs(old_qty):
        self._handle_position_close(position_id, full_close=False, qty_closed=abs(old_qty) - abs(new_qty))
```

---

## IMPLEMENTATION STRATEGY

### Phase 1: Core Infrastructure (P0)
1. ✅ Convert single trackers to dictionaries
2. ✅ Add position_id to TradeManager
3. ✅ Update callbacks with position_id
4. ✅ Modify update loops for multiple positions

### Phase 2: Position Management (P0)
5. ✅ Enhanced Position class with entries tracking
6. ✅ Order → Position mapping
7. ✅ Partial close handling
8. ✅ FIFO/LIFO accounting options

### Phase 3: Integration (P1)
9. ⏳ Update experience replay with position_id
10. ⏳ Update performance tracker for multiple positions
11. ⏳ HUD updates for multiple positions
12. ⏳ Risk management per position

### Phase 4: Testing (P1)
13. ⏳ Test: Open 2 LONGs, close 1
14. ⏳ Test: Hedge (LONG + SHORT simultaneously)
15. ⏳ Test: Scale in (3 entries) then full close
16. ⏳ Test: Scale in then partial close

---

## BACKWARD COMPATIBILITY

### Simple Mode (Current Behavior)
```python
# For users who want single position at a time:
self.max_simultaneous_positions = 1  # Config option

def before_order_submission(self, order):
    if len(self.active_positions) >= self.max_simultaneous_positions:
        LOG.warning("Max positions reached - order blocked")
        return False
```

### Multi-Position Mode
```python
# For advanced users:
self.max_simultaneous_positions = 10  # Or unlimited
self.position_accounting = "FIFO"  # or "LIFO", "AVG"
```

---

## RISK CONSIDERATIONS

### Increased Complexity
- More state to track and persist
- Higher memory usage (N trackers vs 1)
- More potential for bugs in accounting

### Position Sizing
```python
# Before: Simple
qty = calculate_size_for_position()

# After: Consider existing exposure
existing_exposure = sum(pos.net_qty for pos in positions.values())
max_total_exposure = self.max_position_size
available = max_total_exposure - abs(existing_exposure)
qty = min(calculate_size_for_position(), available)
```

### Margin Requirements
```python
# Must track total margin used
total_margin = sum(pos.margin_required for pos in positions.values())
if total_margin > self.max_margin:
    LOG.error("Insufficient margin for new position")
    return False
```

---

## FILE CHANGES REQUIRED

### ctrader_ddqn_paper.py
- Line 611-612: Convert to dictionaries
- Line 718-719: Convert entry state to dict
- Line 1573-1574: Update loop for multiple positions
- Line 1642-1819: Handle multiple position closes
- Line 2073+: Strategy decision with position awareness

### trade_manager.py
- Position class enhancement (avg_price, entries)
- order_to_position mapping
- Partial fill handling
- Position close detection

### trade_manager_example.py
- on_order_filled: Add position_id logic
- New method: _get_or_create_position_id()
- Enhanced validation for multiple positions

---

## TESTING PLAN

### Unit Tests
```python
def test_multiple_long_positions():
    # Open 2 LONGs
    # Verify 2 trackers created
    # Close 1
    # Verify 1 tracker remains
    
def test_hedge_positions():
    # Open LONG
    # Open SHORT
    # Verify both trackers active
    # Close LONG
    # Verify SHORT tracker still active
```

### Integration Tests
```python
def test_scale_in_out():
    # Buy 0.05
    # Buy 0.05 (avg price updated)
    # Sell 0.05 (FIFO - first entry closed)
    # Verify P&L correct
```

---

## MIGRATION PATH

### Step 1: Add Multi-Position Support (Non-Breaking)
```python
# Old code still works:
self.mfe_mae_tracker.start_tracking(...)  # Uses default position

# New code:
self.mfe_mae_trackers[position_id].start_tracking(...)  # Explicit position
```

### Step 2: Deprecation Warning
```python
@property
def mfe_mae_tracker(self):
    """DEPRECATED: Use mfe_mae_trackers[position_id] instead."""
    LOG.warning("mfe_mae_tracker is deprecated - use mfe_mae_trackers dict")
    return self._default_tracker
```

### Step 3: Remove Old API (Future)
After 3 months, remove single-position API entirely.

---

## CONCLUSION

**Current system is fundamentally broken for multiple simultaneous positions.**

All tracking (MFE/MAE, paths, entry states) assumes single position. This must be fixed before any strategy that:
- Scales in/out
- Hedges
- Runs multiple instruments
- Uses bracket orders

**Estimated Effort**: 1-2 days for Phase 1+2 (core multi-position support)

**Priority**: P0 if planning any multi-position strategy, P2 otherwise
