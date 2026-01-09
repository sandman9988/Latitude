# Phase 3: Dual-Agent Quick Reference

**Quick lookup for dual-agent architecture (TriggerAgent + HarvesterAgent)**

---

## Environment Variables

```bash
# Enable dual-agent mode
export DDQN_DUAL_AGENT=1

# Optional: Provide trained models (falls back to rules if omitted)
export DDQN_TRIGGER_MODEL="models/trigger_q_net.pth"
export DDQN_HARVESTER_MODEL="models/harvester_q_net.pth"
```

**Disable dual-agent mode (use single Policy):**
```bash
export DDQN_DUAL_AGENT=0  # or omit variable entirely
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                       DualPolicy                         │
├────────────────────────┬─────────────────────────────────┤
│     TriggerAgent       │      HarvesterAgent             │
│   (Entry Specialist)   │     (Exit Specialist)           │
├────────────────────────┼─────────────────────────────────┤
│ State: 7 market        │ State: 7 market + 3 position    │
│ Actions: NONE/LONG/    │ Actions: HOLD/CLOSE             │
│          SHORT         │                                 │
│ Output: action, conf,  │ Output: action, conf            │
│         runway         │                                 │
└────────────────────────┴─────────────────────────────────┘
```

---

## Usage Patterns

### 1. Entry Decision (when flat)

```python
from dual_policy import DualPolicy

dual_policy = DualPolicy(window=64)

# On bar close (no position)
action, confidence, predicted_runway = dual_policy.decide_entry(
    bars=bars_deque,
    imbalance=0.1,      # Order book imbalance [-1, 1]
    vpin_z=0.5,         # VPIN z-score
    depth_ratio=1.2     # Depth ratio
)

# action: 0=NO_ENTRY, 1=LONG, 2=SHORT
# confidence: [0, 1]
# predicted_runway: Expected MFE (e.g., 0.0025 = 25 pips)
```

### 2. Exit Decision (when in position)

```python
# On bar close (in position)
action, confidence = dual_policy.decide_exit(
    bars=bars_deque,
    current_price=100050.0,
    imbalance=0.1,
    vpin_z=0.5,
    depth_ratio=1.2
)

# action: 0=HOLD, 1=CLOSE
# confidence: [0, 1]
```

### 3. Position Lifecycle Tracking

```python
# When entering position
dual_policy.on_entry(
    direction=1,           # 1=LONG, -1=SHORT
    entry_price=100000.0,
    entry_time=datetime.now()
)

# When closing position
dual_policy.on_exit(
    exit_price=100050.0,
    capture_ratio=0.85,    # exit_pnl / MFE
    was_wtl=False          # Winner-to-loser flag
)
```

---

## State Spaces

### TriggerAgent (7 features)
1. `ret1` - 1-bar log return
2. `ret5` - 5-bar log return
3. `ma_diff` - (MA10 - MA30) / MA30
4. `vol` - 20-bar volatility
5. `imbalance` - Order book imbalance [-1, 1]
6. `vpin_z` - VPIN z-score
7. `depth_ratio` - Normalized depth [0, 2]

### HarvesterAgent (10 features)
All 7 TriggerAgent features **PLUS**:

8. `mfe_norm` - Maximum favorable excursion (normalized)
9. `mae_norm` - Maximum adverse excursion (normalized)
10. `bars_held_norm` - Bars since entry (normalized)

---

## Fallback Strategies

### TriggerAgent (when no model)

**MA Crossover + Microstructure Tilt:**
```python
ma_diff = (MA10 - MA30) / MA30
tilt = imbalance * 0.1

if ma_diff > (0.3 - tilt):
    return LONG, confidence=0.6, runway=0.0015
elif ma_diff < (-0.3 - tilt):
    return SHORT, confidence=0.6, runway=0.0015
else:
    return NO_ENTRY, confidence=0.5, runway=0.0
```

**Conservative thresholds:** 0.3 MA diff (vs 0.2 in single-agent Policy)

### HarvesterAgent (when no model)

**Profit/Stop/Time Targets:**
```python
# Close if:
if MAE > 0.002 (0.2%):      # Stop loss (priority)
    return CLOSE, confidence=0.9
if MFE > 0.003 (0.3%):      # Profit target
    return CLOSE, confidence=0.8
if bars_held > 50 and MFE > 0.0005 (0.05%):  # Soft time stop (min 5 pips profit)
    return CLOSE, confidence=0.7
if bars_held > 80:          # Hard time stop (prevent stagnation)
    return CLOSE, confidence=0.75

# Otherwise:
return HOLD, confidence=0.7
```

**Key Improvement:** Prevents closing on negligible MFE (e.g., 75% of 0.00% = bad exit)

---

## Runway Prediction

**Q-to-runway mapping (TriggerAgent):**
```python
Q_max = 3.0              # Maximum expected Q-value
runway_min = 0.001       # 10 pips
runway_max = 0.005       # 50 pips

# Linear interpolation
q_value = max(Q_values)  # Best action Q-value
runway = runway_min + (q_value / Q_max) * (runway_max - runway_min)
```

**Example:**
- Q = 0.0 → runway = 0.001 (10 pips)
- Q = 1.5 → runway = 0.003 (30 pips)
- Q = 3.0 → runway = 0.005 (50 pips)

---

## Reward Functions (Phase 3.2 - Planned)

### TriggerAgent Reward
```python
# Runway utilization
reward = actual_MFE / predicted_runway

# Example:
# predicted_runway = 0.0025 (25 pips)
# actual_MFE = 0.0030 (30 pips)
# reward = 1.2 (exceeded prediction)
```

### HarvesterAgent Reward
```python
# Capture efficiency + WTL penalty
capture_ratio = exit_pnl / MFE
wtl_penalty = -1.0 if winner_to_loser else 0.0
reward = capture_ratio + wtl_penalty

# Example:
# MFE = 0.0040 (40 pips)
# exit_pnl = 0.0032 (32 pips)
# capture_ratio = 0.8 (80% efficiency)
# reward = 0.8 (no WTL penalty)
```

---

## Integration with Main Bot

**File:** `ctrader_ddqn_paper.py`

### Initialization
```python
self.use_dual_agent = os.environ.get("DDQN_DUAL_AGENT", "0") == "1"
if self.use_dual_agent:
    self.dual_policy = DualPolicy(window=64)
else:
    self.policy = Policy()  # Single-agent fallback
```

### On Bar Close
```python
if self.use_dual_agent:
    if self.cur_pos == 0:  # Flat
        action, conf, runway = self.dual_policy.decide_entry(...)
    else:  # In position
        action, conf = self.dual_policy.decide_exit(...)
else:
    action = self.policy.decide(...)  # Single-agent
```

---

## Testing

**Run all Phase 3 tests:**
```bash
python3 tests/test_phase3_dual_agent.py
```

**Expected output:**
```
✓ All 10 tests passing
- Import validation
- DualPolicy initialization
- Entry/exit workflows
- Position tracking
- Multiple trade cycles
- Backward compatibility
- State consistency
- Agent updates
```

---

## Logging

### Entry Logs
```
[DUAL_POLICY] TRIGGER: LONG entry, conf=0.75, predicted_runway=0.0025
```

### Exit Logs
```
[DUAL_POLICY] HARVESTER: CLOSE signal, conf=0.85, MFE=0.0040, MAE=0.0010, bars=12
```

### Position Lifecycle
```
[DUAL_POLICY] Position entered: LONG @ 100000.00
[DUAL_POLICY] Position closed @ 100050.00, MFE=50.0000, Capture=80.00%
```

---

## Common Workflows

### 1. Test dual-agent locally
```bash
export DDQN_DUAL_AGENT=1
python3 dual_policy.py  # Self-test
python3 tests/test_phase3_dual_agent.py  # Integration test
```

### 2. Run bot in dual-agent mode (no models)
```bash
export DDQN_DUAL_AGENT=1
./run.sh
# Uses fallback strategies
```

### 3. Run bot with trained models
```bash
export DDQN_DUAL_AGENT=1
export DDQN_TRIGGER_MODEL="models/trigger_checkpoint_500.pth"
export DDQN_HARVESTER_MODEL="models/harvester_checkpoint_500.pth"
./run.sh
```

### 4. Switch back to single-agent mode
```bash
export DDQN_DUAL_AGENT=0
export DDQN_MODEL_PATH="models/single_agent.pth"
./run.sh
```

---

## Performance Metrics

| Metric | Single-Agent | Dual-Agent | Change |
|--------|-------------|-----------|--------|
| Handbook Alignment | 35% | 60% | **+25%** |
| Entry Quality | Generic | Runway-aware | ✅ |
| Exit Quality | Generic | Capture-optimized | ✅ |
| State Dimensions | 7 (flat) | 7 entry + 10 exit | ✅ |
| Specialization | None | Entry/Exit split | ✅ |

---

## File Locations

- **TriggerAgent**: `trigger_agent.py` (300 lines)
- **HarvesterAgent**: `harvester_agent.py` (280 lines)
- **DualPolicy**: `dual_policy.py` (370 lines)
- **Integration**: `ctrader_ddqn_paper.py` (modified on_bar_close)
- **Tests**: `tests/test_phase3_dual_agent.py` (170 lines)
- **Documentation**: `docs/PHASE3_SUMMARY.md`

---

## Next Steps

**Phase 3.2: Specialized Rewards (1 day)**
- [ ] `calculate_trigger_reward()` - runway utilization
- [ ] `calculate_harvester_reward()` - capture efficiency + WTL

**Phase 3.3: PathRecorder Enhancement (0.5 days)**
- [ ] Track MFE/MAE bar offsets
- [ ] Store predicted runway
- [ ] Enable attribution (trigger → MFE, harvester → capture)

**Phase 3.4: Regime Detection (2 days)**
- [ ] DSP-based damping ratio
- [ ] Regime-aware runway prediction

**Phase 3.5: Experience Replay (future)**
- [ ] Online learning buffer
- [ ] Continuous improvement loop

---

## Troubleshooting

### Dual-agent not activating
```bash
# Check environment variable
echo $DDQN_DUAL_AGENT  # Should be "1"

# Verify log message
grep "DUAL_AGENT" logs/python/*.log
# Should see: "[DUAL_AGENT] Enabled: TriggerAgent + HarvesterAgent"
```

### Models not loading
```bash
# Check model paths exist
ls -la $DDQN_TRIGGER_MODEL
ls -la $DDQN_HARVESTER_MODEL

# Verify log messages
grep "TRIGGER.*model" logs/python/*.log
grep "HARVESTER.*model" logs/python/*.log
# Should see: "Loaded model from ..." or "No model specified, using fallback"
```

### Unexpected behavior
```bash
# Run self-tests
python3 trigger_agent.py  # 5/5 tests
python3 harvester_agent.py  # 5/5 tests
python3 dual_policy.py  # 5/5 tests
python3 tests/test_phase3_dual_agent.py  # 10/10 tests
```

---

## References

- **Full Documentation**: [docs/PHASE3_SUMMARY.md](PHASE3_SUMMARY.md)
- **Design Reference**: [docs/MASTER_HANDBOOK.md](MASTER_HANDBOOK.md) Section 2.2
- **Phase 1**: [docs/PHASE1_SUMMARY.md](PHASE1_SUMMARY.md)
- **Phase 2**: [docs/PHASE2_SUMMARY.md](PHASE2_SUMMARY.md)
