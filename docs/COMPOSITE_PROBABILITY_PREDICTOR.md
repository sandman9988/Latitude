# Composite Probability Predictor - Multi-Agent Calibration

## Overview

The **RiskManager** now includes a **Composite Probability Predictor** that tracks predictions separately for each agent:

- **TriggerAgent** - Entry decision predictions vs actuals
- **HarvesterAgent** - Exit decision predictions vs actuals  
- **Composite** - Combined view of both agents

This creates the **MAIN RISK MANAGEMENT TOOL** for probability predictions, measuring each agent's predicted vs actual outcomes.

## ✅ Test Results

```bash
$ python3 test_composite_predictor.py

╔====================================================================╗
║              COMPOSITE PROBABILITY PREDICTOR TEST SUITE            ║
║                      (Multi-Agent Calibration)                     ║
╚====================================================================╝

✓ PASS: Per-Agent Calibration
✓ PASS: Composite Probability Predictor
✓ PASS: Composite in Risk Assessment
✓ PASS: Adaptive Trust Weighting

✓ ALL TESTS PASSED (4/4)

The RiskManager now tracks:
  • TriggerAgent predictions vs actuals
  • HarvesterAgent predictions vs actuals
  • Composite view of both agents
  • Automatic identification of which agent to trust
```

**Total Tests**: 16/16 passing (100%)
- Original: 7/7 ✅
- RL & Correlation: 5/5 ✅
- Composite Predictor: 4/4 ✅

---

## Architecture

### Before: Single Probability Calibration

```
┌─────────────────────────────────────┐
│         Decision Outcome            │
│      confidence=0.75, win=True      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Single Calibration Bucket        │
│    All agents mixed together        │
└─────────────────────────────────────┘
```

**Problem**: Can't tell which agent is more accurate!

### After: Composite Probability Predictor

```
┌─────────────────────────────────────────────────────────┐
│                  Decision Outcome                        │
│   agent_id="trigger", confidence=0.75, win=True         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────┴───────────────┐
        │                              │
        ▼                              ▼
┌───────────────────┐      ┌────────────────────┐
│ TriggerAgent      │      │ HarvesterAgent     │
│  Calibration      │      │   Calibration      │
│                   │      │                    │
│ 70% conf → 72%    │      │ 70% conf → 50%     │
│ WELL CALIBRATED   │      │ OVERCONFIDENT      │
└──────┬────────────┘      └────────┬───────────┘
       │                            │
       └────────────┬───────────────┘
                    ▼
        ┌─────────────────────────┐
        │ Composite Predictor     │
        │                         │
        │ Best: TriggerAgent      │
        │ Recommendation:         │
        │ "Trust Trigger more"    │
        └─────────────────────────┘
```

**Solution**: Separate tracking reveals which agent is more accurate!

---

## Usage

### 1. Feed Outcomes Per Agent

```python
# TriggerAgent makes entry decision
entry_val = risk_manager.validate_entry(
    action=1, confidence=0.75, ...
)

if entry_val.approved:
    execute_trade()

# Later: Trade completes
risk_manager.update_decision_outcome(
    decision_type="entry",
    confidence=0.75,
    approved=True,
    actual_outcome=True,  # Win
    agent_id="trigger"  # ← NEW: Specify which agent
)

# HarvesterAgent makes exit decision  
exit_val = risk_manager.validate_exit(
    action=1, confidence=0.85, ...
)

if exit_val.approved:
    close_position()

# Later: Exit completes
risk_manager.update_decision_outcome(
    decision_type="exit",
    confidence=0.85,
    approved=True,
    actual_outcome=False,  # Loss
    agent_id="harvester"  # ← Harvester's prediction
)
```

### 2. Get Composite Predictor

```python
composite = risk_manager.get_composite_probability_predictor()

print(f"TriggerAgent accuracy: {composite.trigger_overall_accuracy:.1%}")
print(f"HarvesterAgent accuracy: {composite.harvester_overall_accuracy:.1%}")
print(f"Best calibrated: {composite.best_calibrated_agent}")
print(f"Recommendation: {composite.recommendation}")

# Example output:
# TriggerAgent accuracy: 72.4%
# HarvesterAgent accuracy: 50.0%
# Best calibrated: trigger
# Recommendation: Trust TriggerAgent more (error: 3.0% vs 30.0%)
```

### 3. Per-Agent Calibration Details

```python
# Get TriggerAgent calibration
trigger_calib = risk_manager.get_probability_calibration("trigger")

for bucket, calib in trigger_calib.items():
    print(f"{bucket:.0%} confidence:")
    print(f"  Predicted: {calib.predicted_success_rate:.1%}")
    print(f"  Actual: {calib.actual_success_rate:.1%}")
    print(f"  Error: {calib.calibration_error:.1%}")
    print(f"  Well calibrated: {calib.is_well_calibrated}")

# Example output:
# 70% confidence:
#   Predicted: 70.0%
#   Actual: 72.5%
#   Error: 2.5%
#   Well calibrated: True
```

### 4. Composite in Risk Assessment

```python
assessment = risk_manager.assess_risk()

# Composite predictor included automatically
if assessment.composite_predictor:
    print(f"Best agent: {assessment.composite_predictor.best_calibrated_agent}")
    print(f"Recommendation: {assessment.composite_predictor.recommendation}")

# Recommendations include per-agent warnings
for rec in assessment.recommendations:
    print(f"• {rec}")

# Example:
# • [TRIGGER] Miscalibrated at 80%: 80% vs 65%
# • [HARVESTER] Miscalibrated at 90%: 90% vs 50%
# • 📊 Trust TriggerAgent more (error: 5.0% vs 25.0%)
```

---

## Data Structures

### CompositeProbabilityPredictor

```python
@dataclass
class CompositeProbabilityPredictor:
    """Composite probability prediction combining all agents"""
    
    trigger_calibration: Dict[float, ProbabilityCalibration]
    # TriggerAgent's calibration by confidence bucket
    # Example: {0.7: ProbabilityCalibration(agent_id="trigger", ...)}
    
    harvester_calibration: Dict[float, ProbabilityCalibration]
    # HarvesterAgent's calibration by confidence bucket
    # Example: {0.8: ProbabilityCalibration(agent_id="harvester", ...)}
    
    composite_calibration: Dict[float, ProbabilityCalibration]
    # Combined view (for legacy compatibility)
    
    trigger_overall_accuracy: float
    # Overall win rate for TriggerAgent (e.g., 0.724 = 72.4%)
    
    harvester_overall_accuracy: float
    # Overall win rate for HarvesterAgent (e.g., 0.500 = 50%)
    
    best_calibrated_agent: str
    # "trigger" or "harvester" - which has lower calibration error
    
    recommendation: str
    # Human-readable recommendation
    # Examples:
    # - "Trust TriggerAgent more (error: 3.0% vs 30.0%)"
    # - "Both agents well-calibrated - trust both equally"
    # - "Insufficient data - need more trades"
```

### Enhanced ProbabilityCalibration

```python
@dataclass
class ProbabilityCalibration:
    """Tracks prediction accuracy for self-calibration (per-agent)"""
    
    agent_id: str  # NEW: "trigger", "harvester", or "composite"
    confidence_bucket: float  # 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    predicted_success_rate: float  # Agent's claimed probability
    actual_success_rate: float  # Observed win rate
    sample_size: int  # Number of trades in this bucket
    calibration_error: float  # |predicted - actual|
    is_well_calibrated: bool  # error < 10%
```

---

## Real-World Example

### Scenario: TriggerAgent Overconfident, HarvesterAgent Well-Calibrated

```python
# Over 100 trades:
# TriggerAgent says 80% confidence → actually wins 55% (overconfident)
# HarvesterAgent says 70% confidence → actually wins 68% (well-calibrated)

composite = risk_manager.get_composite_probability_predictor()

print(composite.recommendation)
# "Trust HarvesterAgent more (error: 2.0% vs 25.0%)"

# Risk assessment includes this in recommendations
assessment = risk_manager.assess_risk()
# Recommendations:
# • [TRIGGER] Miscalibrated at 80%: 80% vs 55%
# • 📊 Trust HarvesterAgent more (error: 2.0% vs 25.0%)
```

### Action: Adjust Confidence Thresholds Per Agent

```python
# Option 1: Require higher confidence from overconfident agent
if composite.best_calibrated_agent == "harvester":
    # TriggerAgent is overconfident, so require higher threshold
    risk_manager.min_confidence_entry = 0.75  # Was 0.6
    # HarvesterAgent is accurate, so keep normal threshold
    risk_manager.min_confidence_exit = 0.5

# Option 2: Weight predictions by calibration quality
trigger_weight = 1.0 / (1.0 + composite.trigger_avg_error)
harvester_weight = 1.0 / (1.0 + composite.harvester_avg_error)

# Use weights when combining agent outputs
combined_confidence = (
    trigger_confidence * trigger_weight +
    harvester_confidence * harvester_weight
) / (trigger_weight + harvester_weight)
```

---

## Integration with Main Bot

### Storing Agent ID with Decisions

```python
# In main bot, track which agent made each decision

# TriggerAgent evaluates entry
trigger_confidence = trigger_agent.get_confidence()
entry_val = risk_manager.validate_entry(
    action=trigger_action,
    confidence=trigger_confidence,
    ...
)

if entry_val.approved:
    # Store metadata with trade
    trade_metadata = {
        "agent_id": "trigger",
        "confidence": trigger_confidence,
        "decision_type": "entry"
    }
    execute_trade(metadata=trade_metadata)

# Later: On trade complete
risk_manager.update_decision_outcome(
    decision_type=trade_metadata["decision_type"],
    confidence=trade_metadata["confidence"],
    approved=True,
    actual_outcome=(pnl > 0),
    agent_id=trade_metadata["agent_id"]  # ← Feed back to correct agent
)
```

### Periodic Composite Analysis

```python
# Every N trades or every hour
if trade_count % 50 == 0:
    composite = risk_manager.get_composite_probability_predictor()
    
    logger.info("=" * 60)
    logger.info("COMPOSITE PROBABILITY PREDICTOR ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"TriggerAgent accuracy: {composite.trigger_overall_accuracy:.1%}")
    logger.info(f"HarvesterAgent accuracy: {composite.harvester_overall_accuracy:.1%}")
    logger.info(f"Best calibrated: {composite.best_calibrated_agent.upper()}")
    logger.info(f"Recommendation: {composite.recommendation}")
    
    # Detailed calibration
    logger.info("\nTriggerAgent calibration:")
    for bucket, calib in sorted(composite.trigger_calibration.items()):
        logger.info(f"  {bucket:.0%}: {calib.actual_success_rate:.1%} actual "
                   f"(predicted {calib.predicted_success_rate:.1%}, "
                   f"error {calib.calibration_error:.1%})")
    
    logger.info("\nHarvesterAgent calibration:")
    for bucket, calib in sorted(composite.harvester_calibration.items()):
        logger.info(f"  {bucket:.0%}: {calib.actual_success_rate:.1%} actual "
                   f"(predicted {calib.predicted_success_rate:.1%}, "
                   f"error {calib.calibration_error:.1%})")
```

### HUD Display

```python
# In hud_tabbed.py, add Composite Predictor tab

def render_composite_predictor_tab():
    composite = risk_manager.get_composite_probability_predictor()
    
    content = f"""
╔══════════════════════════════════════════════════════════╗
║         COMPOSITE PROBABILITY PREDICTOR                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║ TriggerAgent (Entry Decisions):                         ║
║   Overall Accuracy:     {composite.trigger_overall_accuracy:>26.1%} ║
║   Calibration Buckets:  {len(composite.trigger_calibration):>26} ║
║                                                          ║
║ HarvesterAgent (Exit Decisions):                        ║
║   Overall Accuracy:     {composite.harvester_overall_accuracy:>26.1%} ║
║   Calibration Buckets:  {len(composite.harvester_calibration):>26} ║
║                                                          ║
║ Best Calibrated:        {composite.best_calibrated_agent.upper():>30} ║
║                                                          ║
║ Recommendation:                                          ║
║   {composite.recommendation:<56} ║
╚══════════════════════════════════════════════════════════╝
"""
    
    # Detailed calibration tables
    content += "\n\nTriggerAgent Calibration:\n"
    content += "Bucket | Predicted | Actual | Error  | Status\n"
    content += "-------|-----------|--------|--------|---------\n"
    for bucket, calib in sorted(composite.trigger_calibration.items()):
        status = "✓" if calib.is_well_calibrated else "✗"
        content += f"{bucket:>5.0%} | {calib.predicted_success_rate:>8.1%} | " \
                  f"{calib.actual_success_rate:>6.1%} | {calib.calibration_error:>5.1%} | {status}\n"
    
    # Similar for HarvesterAgent...
    
    return content
```

---

## Benefits

### 1. Agent-Specific Feedback

**Before**: Mixed calibration data - can't tell which agent is problematic  
**After**: Clear visibility into each agent's prediction accuracy

### 2. Adaptive Trust

**Before**: Treat both agents equally  
**After**: Trust the better-calibrated agent more

Example:
```
TriggerAgent: 75% accuracy, 5% error → Trust more
HarvesterAgent: 55% accuracy, 25% error → Reduce confidence threshold
```

### 3. Targeted Improvement

**Before**: Generic "model is overconfident" warning  
**After**: Specific guidance per agent

Example:
```
[TRIGGER] Well-calibrated - no changes needed
[HARVESTER] Overconfident at 80-90% buckets - retrain exit logic
```

### 4. Portfolio Optimization

Use calibration quality to weight agent decisions:

```python
# Weight by inverse calibration error
trigger_weight = 1.0 / (1.0 + trigger_error)
harvester_weight = 1.0 / (1.0 + harvester_error)

# Combined decision
combined_signal = (
    trigger_signal * trigger_weight +
    harvester_signal * harvester_weight
) / (trigger_weight + harvester_weight)
```

---

## Calibration Examples

### Well-Calibrated Agent

```
TriggerAgent Calibration:
  60%: predicted=60% actual=62% error=2% ✓
  70%: predicted=70% actual=68% error=2% ✓
  80%: predicted=80% actual=78% error=2% ✓
  
Average error: 2% → WELL CALIBRATED
```

### Overconfident Agent

```
HarvesterAgent Calibration:
  70%: predicted=70% actual=50% error=20% ✗
  80%: predicted=80% actual=55% error=25% ✗
  90%: predicted=90% actual=60% error=30% ✗
  
Average error: 25% → OVERCONFIDENT
```

### Underconfident Agent

```
TriggerAgent Calibration:
  60%: predicted=60% actual=80% error=20% ✗
  70%: predicted=70% actual=85% error=15% ✗
  
Average error: 17.5% → UNDERCONFIDENT
(Agent should be MORE confident!)
```

---

## Configuration

### Per-Agent Calibration Windows

```python
# In RiskManager.__init__
self.calibration_window = 100  # Keep last 100 trades per agent

# Adjust if needed:
risk_manager.calibration_window = 200  # More stable, slower adaptation
risk_manager.calibration_window = 50   # Faster adaptation, more noise
```

### Minimum Sample Size

```python
# In get_probability_calibration
if len(outcomes) < 5:  # Need minimum 5 samples per bucket
    continue

# Adjust threshold:
if len(outcomes) < 10:  # More conservative
    continue
```

---

## Summary

The **Composite Probability Predictor** is now the **MAIN RISK MANAGEMENT TOOL** for probability predictions.

**Key Features**:
1. ✅ **Per-Agent Tracking** - TriggerAgent and HarvesterAgent tracked separately
2. ✅ **Automatic Best Agent** - System identifies which agent is more accurate
3. ✅ **Calibration Quality** - Know exactly how well each agent's predictions match reality
4. ✅ **Actionable Recommendations** - Clear guidance on which agent to trust
5. ✅ **Integrated Assessment** - Composite predictor included in risk assessment

**Test Coverage**: 16/16 tests passing (100%)

**Next Steps**:
- Integrate with main bot to track agent_id with each decision
- Add HUD display for composite predictor
- Use calibration quality to weight agent decisions
- Implement per-agent confidence threshold adjustments

---

**Date**: 2026-01-11  
**Component**: Composite Probability Predictor (Multi-Agent Calibration)  
**Status**: ✅ COMPLETE - Production ready
