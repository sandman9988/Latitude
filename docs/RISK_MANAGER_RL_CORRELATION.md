# RiskManager RL Learning & Correlation Features

## Overview

The **RiskManager** now includes advanced self-learning and correlation monitoring capabilities:

1. **RL Q-Learning** - Learns optimal risk thresholds from outcomes
2. **Probability Calibration** - Tracks and improves prediction accuracy
3. **Correlation Breakdown Detection** - Flash crash early warning
4. **Correlation-Based Capital Allocation** - Diversification optimization

## ✅ Complete Test Results

```bash
$ python3 test_risk_manager_rl.py

╔====================================================================╗
║               RISK MANAGER RL & CORRELATION TEST SUITE             ║
╚====================================================================╝

✓ PASS: Probability Calibration
✓ PASS: RL Q-Learning
✓ PASS: Correlation Breakdown Detection
✓ PASS: Capital Allocation by Correlation
✓ PASS: Integrated Risk Assessment

✓ ALL RL & CORRELATION TESTS PASSED (5/5)
```

---

## 1. Probability Calibration (Self-Adjusting Predictions)

### Purpose
**Track how well agent confidence predictions match actual outcomes.**

A well-calibrated model:
- 70% confidence → 70% win rate ✓
- 90% confidence → 90% win rate ✓

A poorly calibrated model:
- 90% confidence → 50% win rate ✗ (overconfident)
- 60% confidence → 85% win rate ✗ (underconfident)

### Usage

```python
# 1. Feed decision outcomes back
risk_manager.update_decision_outcome(
    decision_type="entry",
    confidence=0.75,  # Agent's confidence
    approved=True,
    actual_outcome=True  # True = win, False = loss
)

# 2. Get calibration report
calibration = risk_manager.get_probability_calibration()

for bucket, calib in calibration.items():
    print(f"Confidence {bucket:.0%}:")
    print(f"  Predicted: {calib.predicted_success_rate:.1%}")
    print(f"  Actual: {calib.actual_success_rate:.1%}")
    print(f"  Error: {calib.calibration_error:.1%}")
    print(f"  Well calibrated: {calib.is_well_calibrated}")
```

### Auto-Calibration in `assess_risk()`

The risk assessment now automatically detects miscalibration:

```python
assessment = risk_manager.assess_risk()

# Recommendations include calibration warnings:
# "Miscalibrated at 80% confidence: predicted 75% vs actual 55%"
```

### Benefits

1. **Detect Agent Overconfidence** - Adjust thresholds when agent is too optimistic
2. **Detect Agent Underconfidence** - Lower thresholds when agent is too conservative  
3. **Continuous Improvement** - System learns true accuracy over time
4. **Threshold Adaptation** - Automatically suggest threshold adjustments

---

## 2. RL Q-Learning (Threshold Optimization)

### Purpose
**Learn optimal confidence thresholds through reinforcement learning.**

The system learns:
- When to be more aggressive (lower thresholds) → During winning streaks, low drawdown
- When to be more defensive (raise thresholds) → During losing streaks, high drawdown

### Q-Learning State Space

```python
State = (drawdown_level, win_rate_bucket, confidence_bucket)
- drawdown_level: 0-5% = 0, 5-10% = 1, etc.
- win_rate_bucket: 0-10% = 0, 10-20% = 1, etc.
- confidence_bucket: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

Actions = {lower_threshold, keep_threshold, raise_threshold}

Reward:
+1.0  if approved and won
+0.5  if rejected and would have lost
-1.0  if approved and lost
-0.5  if rejected but would have won (missed opportunity)
```

### Usage

```python
# 1. Feed outcomes (happens automatically in on_trade_complete)
risk_manager.update_decision_outcome(
    decision_type="entry",
    confidence=0.75,
    approved=True,
    actual_outcome=True  # Win
)

# 2. Get RL recommendations
rl_rec = risk_manager.get_rl_recommended_thresholds()

print(f"Entry threshold: {rl_rec['entry_threshold']:.2f}")
print(f"Exit threshold: {rl_rec['exit_threshold']:.2f}")
print(f"Confidence: {rl_rec['confidence']:.2f}")
print(f"Reason: {rl_rec['reason']}")

# 3. Apply recommendations (optional)
if rl_rec['confidence'] > 0.7:  # High confidence in recommendation
    risk_manager.min_confidence_entry = rl_rec['entry_threshold']
    risk_manager.min_confidence_exit = rl_rec['exit_threshold']
```

### Learning Process

```python
# Initial: Default thresholds (entry=0.6, exit=0.5)

# After 20 winning trades with 0.75 confidence:
# Q-table learns: "lower_threshold" action has higher Q-value
# Recommendation: entry=0.55 (more aggressive)

# After 15 losing trades with 0.65 confidence:
# Q-table learns: "raise_threshold" action has higher Q-value  
# Recommendation: entry=0.70 (more defensive)
```

### Benefits

1. **Adaptive Thresholds** - Automatically tune to current market conditions
2. **Performance-Based** - Learn from actual P&L, not just win rate
3. **State-Aware** - Different thresholds for different market states
4. **Exploration vs Exploitation** - Epsilon-greedy (15%) exploration

---

## 3. Correlation Breakdown Detection (Flash Crash Warning)

### Purpose
**Detect when all asset correlations suddenly approach 1.0 - a flash crash indicator.**

Normal markets:
- Assets have varying correlations (-1.0 to +1.0)
- Diversification works (negatively correlated assets hedge)

Flash crash:
- ALL correlations → +1.0 (everything drops together)
- Diversification fails
- Systemic risk event

### Usage

```python
# 1. Update returns for all symbols you're trading
risk_manager.update_returns("BTCUSD", 0.01)   # +1% return
risk_manager.update_returns("ETHUSD", -0.005) # -0.5% return
risk_manager.update_returns("XRPUSD", 0.002)  # +0.2% return

# 2. Check for correlation breakdown
breakdown = risk_manager.check_correlation_breakdown(
    current_time=time.time()
)

if breakdown:
    print(f"Avg Correlation: {breakdown.avg_correlation:.3f}")
    print(f"Flash Crash Risk: {breakdown.flash_crash_risk}")
    print(f"Recommended Action: {breakdown.recommended_action}")
    
    if breakdown.breakdown_detected:
        print("🚨 CORRELATION BREAKDOWN - SYSTEMIC RISK")
```

### Risk Levels

| Avg Correlation | Risk Level | Recommended Action |
|----------------|------------|-------------------|
| < 0.85         | LOW        | MONITOR           |
| 0.85 - 0.90    | MODERATE   | REDUCE_EXPOSURE   |
| 0.90 - 0.95    | HIGH       | REDUCE_EXPOSURE   |
| > 0.95         | CRITICAL   | CLOSE_ALL         |

### Example: Flash Crash Detection

```python
# Normal market (low correlation)
# BTCUSD: +1%, -0.5%, +2%, -1%...
# ETHUSD: -0.3%, +1.5%, -0.8%, +1%...
# XRPUSD: +0.5%, -1%, +0.2%, -0.5%...
# Avg correlation: ~0.15 (independent movements)

# Flash crash scenario (high correlation)
# BTCUSD: -5%, -7%, -6%, -8%...
# ETHUSD: -4.8%, -6.9%, -5.9%, -7.8%...
# XRPUSD: -5.1%, -7.2%, -6.1%, -8.1%...
# Avg correlation: ~0.98 (everything drops together)
# 🚨 BREAKDOWN DETECTED
```

### Auto-Integration

Risk assessment automatically checks correlation:

```python
assessment = risk_manager.assess_risk()

# If breakdown detected, recommendations include:
# "⚠️  CORRELATION BREAKDOWN: REDUCE_EXPOSURE"
```

### Benefits

1. **Early Warning** - Detect flash crashes before they fully develop
2. **Systemic Risk Protection** - Stop trading when diversification fails
3. **Multi-Symbol Awareness** - Portfolio-level risk monitoring
4. **Automated Response** - Clear action recommendations

---

## 4. Correlation-Based Capital Allocation

### Purpose
**Allocate more capital to negatively correlated assets for better diversification.**

Strategy:
- **Negative correlation** (e.g., -0.7) → Allocate MORE capital (hedge)
- **Low correlation** (e.g., 0.1) → Allocate moderate capital
- **High correlation** (e.g., +0.9) → Allocate LESS capital (concentration risk)

### Usage

```python
# 1. Update returns (build correlation matrix)
for i in range(50):  # Need minimum history
    risk_manager.update_returns("BTCUSD", btc_returns[i])
    risk_manager.update_returns("ETHUSD", eth_returns[i])
    risk_manager.update_returns("XRPUSD", xrp_returns[i])

# 2. Allocate capital
allocation = risk_manager.allocate_capital_by_correlation(
    symbols=["BTCUSD", "ETHUSD", "XRPUSD"],
    total_capital=10000.0
)

# 3. Results
for symbol, amount in allocation.items():
    print(f"{symbol}: ${amount:.2f}")

# Example output:
# XRPUSD: $4911.59 (49.1%)  ← Negatively correlated with BTC
# BTCUSD: $2552.06 (25.5%)  ← Reference asset
# ETHUSD: $2536.35 (25.4%)  ← Positively correlated with BTC
```

### Diversification Score

For each asset, calculate:
```
div_score = avg(1 - |correlation_with_others|)

Higher score = less correlated = better diversification
```

Example:
```
BTCUSD correlations: [0.93 with ETH, -0.83 with XRP]
BTCUSD div_score = avg([1-0.93, 1-(-0.83)]) = avg([0.07, 1.83]) = 0.95

ETHUSD correlations: [0.93 with BTC, -0.82 with XRP]
ETHUSD div_score = avg([1-0.93, 1-(-0.82)]) = avg([0.07, 1.82]) = 0.945

XRPUSD correlations: [-0.83 with BTC, -0.82 with ETH]
XRPUSD div_score = avg([1-(-0.83), 1-(-0.82)]) = avg([1.83, 1.82]) = 1.825 ← BEST
```

XRPUSD gets the most capital (49.1%) because it's negatively correlated with both other assets.

### Benefits

1. **Automatic Diversification** - No manual correlation analysis needed
2. **Hedge Identification** - Finds negatively correlated pairs automatically
3. **Risk Reduction** - Lower portfolio volatility through smart allocation
4. **Concentration Avoidance** - Penalizes highly correlated assets

---

## Integration with Main Bot

### 1. Initialize with RL Enabled

```python
risk_manager = RiskManager(
    circuit_breakers=circuit_breakers,
    var_estimator=var_estimator,
    risk_budget_usd=100.0,
    symbol="BTCUSD",
)

# RL is enabled by default
assert risk_manager.rl_enabled == True
```

### 2. Feed Returns for Correlation Tracking

```python
# In main bot loop, after each bar close
current_return = (current_price - prev_price) / prev_price
risk_manager.update_returns(symbol="BTCUSD", price_return=current_return)

# For multi-symbol:
for symbol, (current, prev) in prices.items():
    ret = (current - prev) / prev
    risk_manager.update_returns(symbol, ret)
```

### 3. Store Decision Metadata

```python
# In validate_entry/exit, metadata is automatically stored
entry_val = risk_manager.validate_entry(
    action=agent_action,
    confidence=agent_confidence,
    # ... other params
)

# Decision metadata saved internally:
# risk_manager._last_decision_type = "entry"
# risk_manager._last_decision_confidence = agent_confidence
```

### 4. Trade Completion Triggers Learning

```python
# In on_trade_complete, RL feedback happens automatically
risk_manager.on_trade_complete(
    pnl=trade_pnl,
    is_win=(trade_pnl > 0),
    equity=current_equity,
    symbol="BTCUSD"
)

# Internally calls:
# - update_decision_outcome() → RL learning
# - _update_q_learning() → Q-table update
# - Probability calibration update
```

### 5. Periodic Risk Assessment

```python
# Every N bars or on demand
assessment = risk_manager.assess_risk(
    current_regime=regime_detector.regime,
    current_vol=current_volatility
)

# Check for warnings
if assessment.correlation_status and assessment.correlation_status.breakdown_detected:
    logger.warning("🚨 CORRELATION BREAKDOWN - CLOSING POSITIONS")
    close_all_positions()

# Check calibration
if assessment.probability_calibration:
    for bucket, calib in assessment.probability_calibration.items():
        if not calib.is_well_calibrated and calib.sample_size > 20:
            logger.warning(f"Miscalibrated at {bucket:.0%}: {calib.calibration_error:.1%} error")

# Apply RL recommendations
if assessment.rl_recommended_thresholds:
    rl_rec = assessment.rl_recommended_thresholds
    if rl_rec['confidence'] > 0.75:
        logger.info(f"RL recommends: entry={rl_rec['entry_threshold']:.2f}")
        # Optionally apply:
        # risk_manager.min_confidence_entry = rl_rec['entry_threshold']
```

### 6. Multi-Symbol Capital Allocation

```python
# Rebalance portfolio based on correlation
symbols = ["BTCUSD", "ETHUSD", "XRPUSD", "SOLUSD"]
total_capital = account_balance * 0.8  # 80% of capital

allocation = risk_manager.allocate_capital_by_correlation(
    symbols=symbols,
    total_capital=total_capital
)

# Set position sizes based on allocation
for symbol, allocated_capital in allocation.items():
    max_position_size = calculate_position_size(allocated_capital, current_prices[symbol])
    # Update trader configuration for this symbol
```

---

## Configuration

### RL Parameters

```python
# In __init__, these are set automatically:
risk_manager.learning_rate = 0.1          # Alpha (Q-learning)
risk_manager.discount_factor = 0.95       # Gamma (future reward weight)
risk_manager.exploration_rate = 0.15      # Epsilon (exploration vs exploitation)
risk_manager.rl_enabled = True            # Enable/disable RL

# Adjust if needed:
risk_manager.learning_rate = 0.05         # Slower learning
risk_manager.exploration_rate = 0.10      # Less exploration
```

### Probability Calibration Parameters

```python
risk_manager.calibration_window = 100     # Keep last 100 trades per bucket

# Calibration buckets: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
# Automatically map confidence to nearest bucket
```

### Correlation Parameters

```python
risk_manager.correlation_window = 50      # Use last 50 bars for correlation
risk_manager.flash_crash_threshold = 0.85 # Avg correlation > 0.85 = warning

# Adjust thresholds:
risk_manager.flash_crash_threshold = 0.90 # Less sensitive
risk_manager.correlation_window = 100     # Longer history (more stable)
```

---

## Data Structures

### ProbabilityCalibration

```python
@dataclass
class ProbabilityCalibration:
    confidence_bucket: float           # 0.5, 0.6, ..., 1.0
    predicted_success_rate: float      # Agent's claimed probability
    actual_success_rate: float         # Observed win rate
    sample_size: int                   # Number of trades
    calibration_error: float           # |predicted - actual|
    is_well_calibrated: bool           # error < 10%
```

### CorrelationBreakdown

```python
@dataclass
class CorrelationBreakdown:
    timestamp: float
    avg_correlation: float             # Average pairwise correlation
    max_correlation: float             # Highest observed
    breakdown_detected: bool           # True if systemic risk
    flash_crash_risk: str              # LOW/MODERATE/HIGH/CRITICAL
    recommended_action: str            # MONITOR/REDUCE_EXPOSURE/CLOSE_ALL
```

### RiskAssessment (Extended)

```python
@dataclass
class RiskAssessment:
    # ... original fields ...
    
    # NEW: RL & Calibration extensions
    probability_calibration: Optional[Dict[float, ProbabilityCalibration]]
    correlation_status: Optional[CorrelationBreakdown]
    rl_recommended_thresholds: Optional[Dict[str, float]]
```

---

## Performance Characteristics

### Memory Footprint

- **Calibration buckets**: 6 buckets × 100 trades = ~600 outcomes stored
- **Returns history**: N symbols × 50 returns × 8 bytes = 400N bytes
- **Q-table**: Sparse storage, typically <1000 states × 3 actions × 8 bytes = ~24KB
- **RL state history**: 1000 states × 200 bytes = ~200KB

**Total**: <1MB for typical multi-symbol setup

### Computational Cost

- **update_decision_outcome()**: O(1) - append to bucket
- **get_probability_calibration()**: O(B×W) where B=buckets(6), W=window(100) = O(600)
- **check_correlation_breakdown()**: O(S²×W) where S=symbols, W=window = O(S²×50)
- **allocate_capital_by_correlation()**: O(S²) for S symbols
- **Q-learning update**: O(1) - single state update

**Bottleneck**: Correlation calculation for many symbols (>10 symbols may slow down)

**Optimization**: Run correlation checks less frequently (every 10 bars instead of every bar)

---

## Summary

The RiskManager now has **three self-improving feedback loops**:

1. **RL Q-Learning Loop**
   - Decisions → Outcomes → Q-table updates → Better thresholds

2. **Probability Calibration Loop**
   - Predictions → Actual results → Calibration analysis → Confidence adjustments

3. **Correlation Monitoring Loop**
   - Returns → Correlation matrix → Breakdown detection → Capital reallocation

All three work together to create a **self-adjusting, self-calibrating risk management system** that learns from experience and adapts to changing market conditions.

---

## Status

✅ **COMPLETE** - All features implemented and tested (5/5 tests passing)
✅ **RL Q-Learning** - Threshold optimization working
✅ **Probability Calibration** - Prediction tracking working
✅ **Correlation Breakdown** - Flash crash detection working
✅ **Capital Allocation** - Diversification optimization working
✅ **Integration** - Ready for main bot

**Next Steps**: Integrate with main bot and HUD display
