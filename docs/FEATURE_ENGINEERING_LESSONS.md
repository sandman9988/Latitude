# Lessons from Feature Engineering Experiments

**Date:** February 14, 2026  
**Source:** trend_sniper v3.5-v3.9 feature optimization experiments  
**Application:** cTrader trading bot feature engineering

---

## 🎯 Top 5 Critical Lessons

### 1. **Subtraction Beats Addition** ⭐⭐⭐

**Finding:**
```
Config                          Dim    Reward    Strategy
Remove 3 noise features         17     +474      Subtraction only
Add best feature (roc_50)       21     +462      Addition  
Full v3.5 baseline              20     +379      Baseline
```

**Why it Works:**
- Removing noise = less for network to learn through
- Tighter gradient signal on remaining features
- Pruned networks use weight budget more efficiently

**Apply to Your Bot:**
1. Run `scripts/analyze_feature_importance.py` 
2. Identify features with L1 weight < 0.08 (bottom 20%)
3. Remove 2-3 lowest, retrain, compare performance
4. If performance improves → keep pruning
5. If performance degrades → restore

**Candidates to Consider Removing:**
- Any feature with training steps > 800 but still low L1 weight
- Features that are redundant (e.g., if you have both `momentum` and `roc_5`)
- Features added early that seemed good but aren't actually used

---

### 2. **Small Sample Size = Wrong Conclusions** ⭐⭐⭐

**Finding:**
```
Sample Size    Pattern Found           Cohen's d
27 trades      "Buy pullback"          accel_rate: 0.506 ★★★
173 trades     "Ride trend"            accel_rate: 0.029 (zero!)
               TRUE SIGNAL →           roc_50: 0.453 ★★★
```

**Why it Matters:**
- First 27 trades showed complete opposite of truth
- Pattern flipped when sample grew 6.4x
- Early conclusions would have optimized for wrong signal!

**Apply to Your Bot:**

**Current Status:**
```python
# From your training_metadata.json:
trigger_steps: 831
harvester_steps: 869
# Estimated trades: ~100-200 (based on typical RL frequency)
```

**ACTION: DO NOT TRUST CURRENT DISCRIMINATOR ANALYSIS YET**

Wait for:
- ✅ 500+ closed trades minimum
- ✅ At least 200 winners AND 200 losers for reliable Cohen's d
- ✅ Multiple market regimes (trending + ranging + volatile)

**How to Check Sample Size:**
```python
python -c "
import json
with open('data/decision_log.json') as f:
    trades = json.load(f)
closed = [t for t in trades if t.get('exit_reason')]
winners = [t for t in closed if t.get('pnl_pct', 0) > 0]
print(f'Closed: {len(closed)}, Winners: {len(winners)}, Losers: {len(closed)-len(winners)}')
if len(closed) < 500:
    print('⚠️  SAMPLE TOO SMALL - patterns unreliable')
"
```

---

### 3. **Feature Synergy Can Be Deceptive** ⭐⭐

**Finding:**
```
Feature         Solo Performance    Combined (A+C+E)
C: vol_roc      -4.9 (dead)        +478 (best!)
E: dist_sup     +46 (weak)         +478 (best!)
A: roc_50       +462 (strong)      +478 (synergy)
```

**Why it Happens:**
- Some features are useless individually but create compound signals when paired
- `roc_50 + vol_roc + dist_sup` = "momentum breakout detector"
- Each feature alone doesn't discriminate, but together they triangulate a pattern

**Apply to Your Bot:**

Your dual-agent architecture already shows this:
- **TriggerAgent (7 features):** Entry specialist
- **HarvesterAgent (10 features):** Exit specialist + 3 position features

The position features (`unrealized_pnl_pct`, `mfe_pct`, `bars_held`) are useless for the trigger agent but critical for harvester. **This is correct architecture - keep specialized feature sets.**

**WARNING:** Don't remove position features from Harvester even if they have low L1 weights in isolation. They create synergy with market features for exit decisions.

**Test Approach:**
1. Test individual features first to find strong solo performers
2. Test pairs/triplets for synergy
3. Don't keep features ONLY because they work in combo - check if removing them from combo helps

---

### 4. **L1 Weight Analysis Reveals Hidden Noise** ⭐⭐

**Finding:**
```
Feature         L1 Weight    Diagnostic d    Verdict
momentum        0.072        0.010           REMOVE (noise)
reg_delta       0.074        0.200           REMOVE (noise)  
trend_align     0.074        —               REMOVE (noise)
Re_norm         0.093        0.102           KEEP (used in exits)
flow_qual       0.087        0.015           KEEP (trade mgmt)
```

**Why This Works:**
- L1 weight = how much the network actually uses each feature
- Low L1 + low discriminatory power (Cohen's d) = confirmed noise
- High L1 even with low d = feature serves a different role (e.g., exit timing vs entry quality)

**Apply to Your Bot:**

Run the analysis script:
```bash
python scripts/analyze_feature_importance.py
```

Expected output:
```
Feature Importance (L1 weight magnitude):
Feature              Importance  Bar
--------------------------------------------------
mfe_pct              0.1181      ##############################
regime_score         0.1080      ###########################
unrealized_pnl       0.0992      #########################
...
momentum             0.0718      ##################  ← SUSPECT
```

**Decision Tree:**
```
L1 < 0.08 AND d < 0.15?  → Strong removal candidate
L1 < 0.08 AND d > 0.30?  → Feature might be valuable, test removal carefully
L1 > 0.09 AND d < 0.15?  → Keep (used for other decisions)
```

---

### 5. **Greedy Elimination Finds the Optimum** ⭐⭐

**Finding:**
```
Round 0: Start with 17 features (removed 3 noise)  → +474 reward
Round 1: Try removing each of 17, one at a time
  - All 17 removals hurt performance (-35 to -531)
  - STOP: 17 features is optimal
```

**Why Greedy Works:**
- Tests every feature in isolation against current best
- Doesn't assume feature importance is static
- Finds local optimum efficiently (vs 2^20 = 1M combinations)

**Apply to Your Bot:**

After you have 500+ trades and L1 analysis:

**Step 1: Remove obvious noise (L1 < 0.08, d < 0.15)**
```python
# Example removal candidates based on typical patterns:
REMOVE_CANDIDATES = [
    'momentum',      # if redundant with other ROC features
    'regime_delta',  # if v3.4-style addition not helping
]
```

**Step 2: Greedy elimination from pruned base**
```bash
# Pseudo-code for greedy elimination
base_features = current_features - REMOVE_CANDIDATES
best_reward = train(base_features)

for feature in base_features:
    test_features = base_features - {feature}
    reward = train(test_features, steps=200k)  # screening run
    if reward > best_reward:
        best_reward = reward
        base_features = test_features
        print(f" Removing {feature} improved reward!")
```

**Step 3: Validate final set with full 500k training**

---

## 🔧 Practical Implementation Roadmap

### Phase 1: Measurement (Current - Before Making Changes)
```bash
# 1. Analyze current feature usage
python scripts/analyze_feature_importance.py

# 2. Check sample size
python -c "
import json
with open('data/decision_log.json') as f:
    closed = [t for t in json.load(f) if t.get('exit_reason')]
print(f'Closed trades: {len(closed)}')
print('Ready for analysis' if len(closed) >= 500 else '⚠️ Need more data')
"

# 3. Let bot accumulate more trades if < 500
# DO NOT make feature changes yet if sample is small!
```

### Phase 2: Analysis (After 500+ Trades)
```bash
# 1. Identify removal candidates
# - L1 weight < 0.08
# - Cohen's d < 0.15  
# - Not critical for position management

# 2. Cross-reference with actual win/loss patterns
# - Implement Cohen's d analysis on entry features
# - Identify true discriminators (d > 0.30)

# 3. Shortlist 2-3 features to remove
```

### Phase 3: Ablation Testing
```bash
# 1. Remove candidate features one at a time
# 2. Train for 200k steps (screening run)
# 3. Compare against baseline

# Expected timeline:
# - 3 removal candidates × 200k steps × ~2 hours = ~6 hours
```

### Phase 4: Production Implementation
```bash
# 1. If ablation shows improvement:
#    - Update observation space in dual_policy.py
#    - Full 500k training run
#    - Validate on paper trading

# 2. If ablation shows degradation:
#    - Keep current feature set
#    - Consider ADDING strong discriminators instead (if found)
```

---

## ⚠️ Critical Warnings

### 1. Don't Remove Position Features from Harvester
```python
# THESE ARE SACRED for HarvesterAgent:
sacred_features = [
    'unrealized_pnl_pct',  # Current PnL
    'mfe_pct',             # Maximum Favorable Excursion
    'bars_held',           # Time in position
]
```

Even if L1 weight is low, these create synergy for exit decisions. The experiments showed removing position-related features collapsed performance (-513 to -531).

### 2. Physics Features (Re_market, flow_quality) May Be Trade Management Tools
```python
# Low discriminatory power for ENTRY quality (d < 0.15)
# But high L1 weight (>0.09) suggests used for EXITS
# Don't remove based on entry discrimination alone!
```

### 3. Replacement Strategy Has Mixed Results
```python
# Adding roc_50 to full base:     +462 reward
# Adding roc_50 to pruned base:   +393 reward (WORSE)
# Pruned base alone:              +474 reward (BEST)

# LESSON: After pruning, the network is already optimized.
#         Adding features back forces it to re-split attention.
#         Only add features to FULL (unpruned) base, not pruned.
```

---

## 📊 Success Metrics

Track these to measure if changes help:

### Training Metrics
```
Metric                  Current    Target     
----------------------------------------------
Epsilon decay          0.9995     Monitor (should stabilize ~0.15-0.25)
Trigger steps          831        Track (growth rate matters)
Harvester steps        869        Track
Exploration ratio      85%        Should decrease over time
```

### Performance Metrics  
```
Metric                  Baseline   After Pruning
----------------------------------------------
Win rate               TBD        Should stay ≥ baseline
PnL per trade          TBD        Should increase (better selectivity)
Trade count            TBD        May decrease (more selective = good)
Sharpe ratio           TBD        Should increase
```

### Feature Health Metrics
```
Metric                  Good       Bad
----------------------------------------------
L1 weight variance     >2x range  < 1.5x (all features used equally)
Bottom 20% L1 weight   >0.08      <0.07 (ignored features)
Cohen's d top 5        >0.30      <0.15 (no discriminators found)
```

---

## 🚀 Quick Wins You Can Try Now

### 1. Log Entry Feature Snapshots
```python
# In dual_policy.py, when entering position:
def _store_entry_snapshot(self, obs):
    """Log full observation vector at entry for later analysis."""
    self._entry_snapshot = {
        'timestamp': time.time(),
        'distance_pct': obs[0],
        'regime_score': obs[1],
        'vol_norm': obs[2],
        # ... all features
    }
    # Store in decision log for Cohen's d analysis later
```

### 2. Monitor Feature Usage Over Time
```bash
# Add to your monitoring:
every 10k steps:
    python scripts/analyze_feature_importance.py >> logs/feature_importance.log
    
# This creates historical record to see if certain features become more/less important as training progresses
```

### 3. Validate Your Current Features Aren't Redundant
```python
# Check for correlation between features:
import numpy as np
from data import decision_log

# If correlation > 0.9 between two features, one is redundant
# Example: momentum and roc_5 often highly correlated
```

---

## 📚 References

**Source Experiments:**
- trend_sniper v3.5 (baseline, 20 features)
- trend_sniper v3.8 (failed - added 5 features including noise)
- trend_sniper v3.9 (best - removed 3 noise features, 17 total)

**Key Results:**
```
Version    Features    Strategy              Reward    PnL
v3.5       20         Baseline              +379      +0.176
v3.8       25         Add 5 (1 signal + 4 noise)  +245  +0.149
v3.9       17         Remove 3 noise        +474      +0.264
```

**The Lesson:** Removing 3 features (+95 reward, +50% PnL) beat adding 5 features (-134 reward, -15% PnL).

---

## 🎓 Summary: The Feature Engineering Commandments

1. **Measure first, optimize second** - Run L1 analysis before making changes
2. **More data > clever analysis** - Wait for 500+ trades before trusting patterns  
3. **Subtraction > addition** - Remove noise before adding signal
4. **Test individually first** - Feature synergy is real but test alone first
5. **Greedy elimination works** - Stepwise removal finds local optimum efficiently
6. **Small samples lie** - 27 trades showed opposite of 173-trade truth
7. **Position features are sacred** - Never remove features used for trade management
8. **Weight != importance** - Low L1 weight with high role (exits) still matters
9. **Validate assumptions** - Don't assume features help - measure on actual trades
10. **Less can be more** - 17 features beat 20 beat 25

---

**Next Step:** Run `scripts/analyze_feature_importance.py` and review results before making any changes.
