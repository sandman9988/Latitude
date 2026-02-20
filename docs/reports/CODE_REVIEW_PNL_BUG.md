# Code Review: P&L Bug Analysis & Improvements

**Date:** 2026-02-17  
**Issue:** P&L calculation showing $0.00 for all trades  
**Root Cause:** Variable shadowing/overwriting in `_process_trade_completion()`

---

## 🐛 THE BUG

### Location
`src/core/ctrader_ddqn_paper.py`, line 2434 (before fix)

### What Happened
```python
# Line 2346: P&L correctly calculated
pnl = (exit_price - entry_price) * direction_sign * self.qty * self.contract_size
# Result: pnl = 7.9000 ✓

# ... 88 lines later ...

# Line 2434: P&L OVERWRITTEN (BUG!)
pnl = summary.get("pnl", 0.0)  # ← Returns 0.0 because summary has no "pnl" key
# Result: pnl = 0.0000 ✗

# Line 2509: Trade saved with wrong P&L
trade_record = {
    "pnl": pnl,  # ← Now 0.0 instead of 7.9
    ...
}
```

### Debug Output That Revealed It
```
[P&L_DEBUG] exit=4879.75 entry=4878.96 dir_sign=1 qty=0.1000 contract_size=100.00 → pnl=7.9000
[TRADE_COMPLETION] ✓ Processed: LONG entry=4878.96 exit=4879.75 pnl=0.0000 mfe=0.1250
```

---

## 🔍 CODE INCONSISTENCIES FOUND

### 1. **Variable Shadowing Anti-Pattern**
**Problem:** Reusing critical variable names for different purposes in long functions

**Location:** Line 2434
```python
# pnl already exists with correct calculation
pnl = (exit_price - entry_price) * ...  # Line 2346

# Then later, pnl reused for a different purpose
pnl = summary.get("pnl", 0.0)  # Line 2434 - WRONG!
```

**Fix Applied:**
```python
pnl_for_reward = pnl  # Use descriptive name, don't overwrite
```

**Recommendation:** Use distinct variable names:
- `calculated_pnl` - for the actual P&L calculation
- `pnl_for_reward` - for reward calculation input
- `summary_pnl` - if extracting from summary dict

---

### 2. **Inconsistent Data Flow**
**Problem:** The `summary` dict comes from MFE/MAE tracker but doesn't contain P&L

**Current State:**
```python
# TradeManagerIntegration calls this:
self.app._process_trade_completion(tracker_summary, order.avg_price)

# tracker_summary contains: {
#   "direction": "LONG",
#   "entry_price": 4878.96,
#   "mfe": 0.125,
#   "mae": 0.0,
#   "winner_to_loser": False
# }
# Note: NO "pnl" field!
```

**The function then:**
1. Calculates P&L (✓ correct)
2. Tries to read P&L from summary (✗ wrong assumption)

**Issue:** Mismatch between what tracker provides vs. what function expects

---

### 3. **P&L Calculated in Multiple Places**
**Problem:** P&L is calculated in at least 3 different locations with different formulas

**Location 1:** `ctrader_ddqn_paper.py` line 2346 (main calculation)
```python
pnl = (exit_price - entry_price) * direction_sign * self.qty * self.contract_size
```

**Location 2:** `trade_manager_integration.py` line 225
```python
pnl = (order.avg_price - entry_price) if tracker.direction > 0 else (entry_price - order.avg_price)
# Missing: * qty * contract_size ← INCOMPLETE!
```

**Location 3:** `path_recorder.py` (passed as parameter, not calculated)

**Issue:** No single source of truth for P&L calculation

---

### 4. **Contract Size Confusion**
**Problem:** Contract size set in multiple places with different values

```python
# Line 618: Default value
self.contract_size = float(os.environ.get("CTRADER_CONTRACT_SIZE", "100000"))  # 100k

# Line 658: Overwritten from friction calculator
if os.environ.get("CTRADER_CONTRACT_SIZE") is None:
    self.contract_size = max(self.friction_calculator.costs.contract_size or 1.0, 1.0)
    # Returns 1.0 or 100.0 depending on symbol_specs.json
```

**Result:** For XAUUSD:
- Default: 100,000 (wrong!)
- symbol_specs.json: 100.0 (correct)
- Actual: Depends on whether env var is set

**Missing:** Clear logging of which value is actually used

---

## 📝 LOGGING IMPROVEMENTS NEEDED

### 1. **Missing P&L Validation Checkpoints**
Current logging only shows the FINAL P&L value. We need checkpoints:

```python
# ✓ ADDED (temp debug)
LOG.info("[P&L_DEBUG] exit=%.2f entry=%.2f dir_sign=%d qty=%.4f contract_size=%.2f → pnl=%.4f", ...)

# ✗ MISSING: Log before any variable reuse
LOG.info("[P&L_CHECKPOINT] Calculated PnL: %.4f (before reward processing)", pnl)

# ✗ MISSING: Log trade record before save
LOG.debug("[TRADE_RECORD] Saving: %s", json.dumps(trade_record, indent=2))
```

### 2. **Function Entry/Exit Logging**
```python
def _process_trade_completion(self, summary: dict, exit_price: float):
    """Process a completed trade..."""
    # ✗ MISSING:
    LOG.debug("[TRADE_COMPLETION] Entry: summary=%s exit_price=%.2f", summary, exit_price)
    
    try:
        # ... processing ...
        
        # ✗ MISSING:
        LOG.debug("[TRADE_COMPLETION] Exit: pnl=%.4f trade_id=%d", pnl, trade_id)
```

### 3. **Contract Size Initialization Logging**
```python
# Current: Silent override
if os.environ.get("CTRADER_CONTRACT_SIZE") is None:
    self.contract_size = max(self.friction_calculator.costs.contract_size or 1.0, 1.0)

# Should be:
if os.environ.get("CTRADER_CONTRACT_SIZE") is None:
    old_value = self.contract_size
    self.contract_size = max(self.friction_calculator.costs.contract_size or 1.0, 1.0)
    LOG.info("[CONTRACT_SIZE] Overridden: %.2f → %.2f (from symbol_specs)", old_value, self.contract_size)
else:
    LOG.info("[CONTRACT_SIZE] Using env override: %.2f", self.contract_size)
```

### 4. **Summary Dict Structure Logging**
```python
# When receiving summary from tracker:
LOG.debug("[TRADE_COMPLETION] Summary keys: %s", list(summary.keys()))
LOG.debug("[TRADE_COMPLETION] Summary contents: %s", summary)
```

### 5. **Variable Shadowing Warnings**
Consider adding runtime assertions:
```python
# After calculating pnl
_initial_pnl = pnl

# ... long processing ...

# Before using pnl again, verify it hasn't changed
assert pnl == _initial_pnl, f"PnL was modified! Initial={_initial_pnl}, Current={pnl}"
```

---

## 🧪 WHY TESTS DIDN'T CATCH THIS

### Root Causes

#### 1. **No Integration Tests for _process_trade_completion()**
```bash
$ grep -r "_process_trade_completion" tests/
# Returns: 0 results
```

**Gap:** The function that had the bug is never tested directly.

#### 2. **Tests Use Mocked/Stubbed P&L Values**
Example from `test_calculation_safety.py`:
```python
def test_pnl_long_profit(self):
    pnl = self.calculate_pnl(entry, exit, quantity, "BUY", spec)
    # This tests the FORMULA, not the actual code path used in production
```

**Gap:** Tests calculate P&L separately, not through the actual bot's flow.

#### 3. **No End-to-End Trade Lifecycle Tests**
Existing tests mock individual components:
- ✓ Tests P&L formula in isolation
- ✓ Tests performance tracker with hardcoded P&L
- ✗ Never tests: Open position → Close position → Verify P&L in trade_log.jsonl

#### 4. **No Assertion on trade_log.jsonl Contents**
Tests don't read the actual output file and verify P&L values.

```python
# What's missing:
def test_trade_saves_correct_pnl(self):
    # 1. Open LONG @ 4878.96
    # 2. Close @ 4879.75 
    # 3. Read trade_log.jsonl
    # 4. Assert: pnl == 7.9 (not 0.0)
```

#### 5. **Summary Dict Structure Not Validated**
No test verifies:
```python
# What keys does tracker.get_summary() actually return?
summary = tracker.get_summary()
assert "entry_price" in summary  # ✓ Would pass
assert "direction" in summary     # ✓ Would pass
assert "pnl" in summary           # ✗ Would FAIL - but no test checks this!
```

---

## 🎯 RECOMMENDED IMPROVEMENTS

### Immediate (Critical)

1. **Add P&L Calculation Tests**
```python
# tests/test_pnl_calculation.py
def test_process_trade_completion_calculates_pnl_correctly():
    """Test that P&L is calculated and saved correctly."""
    bot = create_test_bot()
    
    summary = {
        "direction": "LONG",
        "entry_price": 4878.96,
        "mfe": 0.125,
        "mae": 0.0,
    }
    
    bot._process_trade_completion(summary, exit_price=4879.75)
    
    # Read the saved trade record
    with open("data/trade_log.jsonl") as f:
        last_trade = json.loads(f.readlines()[-1])
    
    # Verify P&L is correct (not 0.0)
    expected_pnl = (4879.75 - 4878.96) * 1 * 0.1 * 100.0  # 7.9
    assert abs(last_trade["pnl"] - expected_pnl) < 0.01
    assert last_trade["pnl"] != 0.0  # Critical: not zero!
```

2. **Refactor P&L Calculation into Single Function**
```python
def calculate_position_pnl(
    self, 
    entry_price: float, 
    exit_price: float, 
    direction: str, 
    quantity: float = None,
    contract_size: float = None
) -> float:
    """
    Single source of truth for P&L calculation.
    
    Args:
        entry_price: Entry execution price
        exit_price: Exit execution price
        direction: "LONG" or "SHORT"
        quantity: Position size in lots (default: self.qty)
        contract_size: Contract size (default: self.contract_size)
    
    Returns:
        P&L in USD
    """
    qty = quantity if quantity is not None else self.qty
    contract = contract_size if contract_size is not None else self.contract_size
    
    direction_sign = 1 if direction == "LONG" else -1
    pnl = (exit_price - entry_price) * direction_sign * qty * contract
    
    LOG.debug(
        "[PNL_CALC] %s: (%.2f - %.2f) * %d * %.4f * %.2f = %.4f",
        direction, exit_price, entry_price, direction_sign, qty, contract, pnl
    )
    
    return pnl
```

3. **Add Assertions to Catch Variable Corruption**
```python
def _process_trade_completion(self, summary: dict, exit_price: float):
    # Calculate P&L
    pnl = self.calculate_position_pnl(entry_price, exit_price, direction)
    
    # Guard against accidental overwrite
    _pnl_checkpoint = pnl
    
    # ... processing ...
    
    # Before saving, verify P&L wasn't corrupted
    if abs(pnl - _pnl_checkpoint) > 0.001:
        LOG.error(
            "[BUG_DETECTION] P&L changed during processing! "
            "Initial=%.4f Final=%.4f", 
            _pnl_checkpoint, pnl
        )
        pnl = _pnl_checkpoint  # Restore
```

### Short Term (Important)

4. **Standardize Summary Dict Structure**
```python
@dataclass
class TradeCompletionSummary:
    """Standardized structure for trade completion."""
    direction: str
    entry_price: float
    exit_price: float  # Add this!
    mfe: float
    mae: float
    winner_to_loser: bool
    # Do NOT include pnl - it should be calculated, not passed

def _process_trade_completion(self, summary: TradeCompletionSummary):
    """Now type-safe - can't access summary.pnl by accident!"""
```

5. **Add Integration Test Suite**
```python
# tests/integration/test_trade_lifecycle.py
class TestTradeLifecycle:
    """End-to-end tests for complete trade execution."""
    
    def test_long_profit_recorded_correctly(self):
        """LONG position with profit saves correct P&L."""
        
    def test_long_loss_recorded_correctly(self):
        """LONG position with loss saves correct P&L."""
        
    def test_short_profit_recorded_correctly(self):
        """SHORT position with profit saves correct P&L."""
        
    def test_exploration_entry_preserves_pnl(self):
        """P&L not corrupted during exploration entry processing."""
```

6. **Add Pre-Commit P&L Validation**
```python
# scripts/validate_pnl.py
def validate_recent_trades():
    """Check last 10 trades have non-zero P&L when prices differ."""
    with open("data/trade_log.jsonl") as f:
        trades = [json.loads(line) for line in f.readlines()[-10:]]
    
    for trade in trades:
        if trade["entry_price"] != trade["exit_price"]:
            assert trade["pnl"] != 0.0, f"Trade {trade['trade_id']} has zero P&L!"
```

### Long Term (Architecture)

7. **Separate Concerns: Calculation vs. Persistence**
```python
class TradeProcessor:
    def calculate_pnl(self, ...):
        """Pure calculation - no side effects."""
    
    def calculate_rewards(self, pnl, ...):
        """Pure calculation - takes pnl as input."""
    
    def save_trade(self, pnl, ...):
        """Persistence - uses pre-calculated values."""
```

8. **Add Type Hints Throughout**
```python
def _process_trade_completion(
    self, 
    summary: Dict[str, Any],  # ← Currently untyped
    exit_price: float
) -> None:
```

9. **Use Immutable Data Structures**
```python
from typing import NamedTuple

class TradePnL(NamedTuple):
    """Immutable P&L calculation result."""
    pnl: float
    entry_price: float
    exit_price: float
    # Can't be accidentally modified!
```

---

## 📊 IMPACT ASSESSMENT

### Scope
- **Affected:** ALL trades since bot inception
- **Duration:** Unknown (bug existed before 2026-02-17)
- **Data:** 722 trades in `trade_log.jsonl` have pnl=0.0

### Consequences
1. **Performance Metrics Invalid**
   - Total P&L: Wrong
   - Win rate: Wrong (can't identify winners)
   - Sharpe ratio: Wrong
   
2. **Learning Corrupted**
   - Reward calculations used pnl=0.0
   - RL agent learned from incorrect signals
   
3. **Circuit Breakers Ineffective**
   - Drawdown tracking based on wrong P&L
   
4. **Cannot Analyze Historical Performance**
   - Need to recalculate P&L for all past trades

### Recovery
```python
# scripts/recalculate_historical_pnl.py
def fix_trade_log():
    """Recalculate P&L for all historical trades."""
    trades = []
    with open("data/trade_log.jsonl") as f:
        for line in f:
            trade = json.loads(line)
            
            # Recalculate correct P&L
            direction_sign = 1 if trade["direction"] == "LONG" else -1
            pnl = (
                (trade["exit_price"] - trade["entry_price"]) 
                * direction_sign 
                * 0.1  # qty
                * 100.0  # contract_size for XAUUSD
            )
            
            trade["pnl"] = pnl
            trade["pnl_recalculated"] = True
            trades.append(trade)
    
    # Save corrected log
    with open("data/trade_log_corrected.jsonl", "w") as f:
        for trade in trades:
            f.write(json.dumps(trade) + "\n")
```

---

## ✅ VERIFICATION CHECKLIST

After applying fixes, verify:

- [ ] New trades show non-zero P&L in logs
- [ ] P&L matches expected: (exit - entry) × direction × qty × contract_size
- [ ] trade_log.jsonl contains correct P&L values
- [ ] Performance metrics update correctly
- [ ] Circuit breakers trigger on actual P&L
- [ ] No "P&L was modified" assertion failures
- [ ] Debug logging removed after confirmation
- [ ] Integration tests added and passing
- [ ] Historical trades recalculated

---

## 🎓 LESSONS LEARNED

1. **Variable Naming Matters:** Reusing critical variable names = recipe for bugs
2. **Long Functions Are Dangerous:** 240-line function with 88 lines between calculation and save
3. **Test What You Run:** Tests must cover actual production code paths
4. **Validate Assumptions:** Don't assume dicts contain keys without checking
5. **Single Responsibility:** P&L calculation should be one function, called once
6. **Immutability Helps:** Const/final variables prevent accidental modification
7. **Logging Is Critical:** Debug logging helped us find the bug in minutes

---

## 📚 REFERENCES

- Bug Location: `src/core/ctrader_ddqn_paper.py:2434` (before fix)
- Fix Commit: Line 2437 - use `pnl_for_reward` instead of reusing `pnl`
- Test Gap Analysis: No integration tests for `_process_trade_completion()`
- Related Issue: contract_size initialization confusion (lines 618, 658)

