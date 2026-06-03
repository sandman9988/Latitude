# Adaptive Parameters Architecture
## Context-Aware Defaults with Fuzzy Phase Transitions

### Design Philosophy
**Constants are DEFAULT values only** - they adapt via:
1. **LearnedParametersManager** (instrument/timeframe-specific)
2. **Environment variables** (deployment/phase-specific)  
3. **Constructor arguments** (explicit overrides)
4. **Phase maturity blending** (smooth exploration→exploitation)
5. **Instrument characteristics** (volatility, liquidity scaling)
6. **Timeframe scaling** (M1 scalping vs H1 swing)

---

## ✅ Implemented Adaptive Layers

### 1. **Activity Monitor** - Phase-Aware Exploration
```python
# Fuzzy phase blending: 0.0 (full exploration) → 1.0 (full exploitation)
ActivityMonitor(phase_maturity=0.3)  # 30% toward exploitation

# Automatically blends between:
# - Paper mode: max_bars_inactive=30, exploration_boost=0.3
# - Live mode: max_bars_inactive=100, exploration_boost=0.1
```

**Effect**: Smooth decay from aggressive exploration (early training) to conservative exploitation (live trading).

---

### 2. **Regime Detector** - Instrument Volatility Scaling
```python
# Adapt thresholds to instrument characteristics
RegimeDetector(instrument_volatility=1.5)  # 50% more volatile than baseline

# Trending threshold: 0.7 * 1.5 = 1.05
# Mean-reverting threshold: 1.3 * 1.5 = 1.95
```

**Effect**: High-volatility assets (crypto) get wider regime bands; low-vol assets (forex majors) get tighter bands.

---

### 3. **Harvester Agent** - Timeframe-Aware Exits
```python
# M1 scalping: 0.3x multiplier (tight stops, fast exits)
HarvesterAgent(timeframe="M1")
# → profit_target: 0.30 * 0.3 = 0.09%
# → time_stop: 50 / 0.3 = 167 bars

# H1 swing: 2.0x multiplier (wide stops, patience)
HarvesterAgent(timeframe="H1")
# → profit_target: 0.30 * 2.0 = 0.60%
# → time_stop: 50 / 2.0 = 25 bars
```

**Timeframe Scale Map**:
- M1: 0.3x (scalp)
- M5: 0.6x
- M15: 1.0x (baseline)
- M30: 1.5x
- H1: 2.0x
- H4: 3.5x
- D1: 5.0x (swing)

---

### 4. **Circuit Breakers** - Learned Threshold Overrides
```python
# Resolution hierarchy:
CircuitBreakerManager(
    sortino_threshold=0.6,  # 1. Explicit override
    param_manager=lpm       # 2. Learned params (per instrument/timeframe)
)
# 3. Falls back to SORTINO_THRESHOLD_DEFAULT = 0.5
```

**Sources logged**:
- `(explicit)` - Constructor argument
- `(learned)` - LearnedParametersManager
- `(ENV)` - Environment variable
- `(default)` - Module constant

---

## Parameter Resolution Flow

```
User Request
    ↓
┌─────────────────────────────────────┐
│ 1. Explicit Constructor Argument   │ ← Highest priority
└─────────────────────────────────────┘
    ↓ (if None)
┌─────────────────────────────────────┐
│ 2. Environment Variable             │ ← Deployment control
└─────────────────────────────────────┘
    ↓ (if not set)
┌─────────────────────────────────────┐
│ 3. LearnedParametersManager         │ ← Instrument/timeframe-specific
│    - symbol: BTCUSD vs EURUSD       │
│    - timeframe: M1 vs H1            │
│    - broker: default vs IC Markets  │
└─────────────────────────────────────┘
    ↓ (if not found)
┌─────────────────────────────────────┐
│ 4. Context-Aware Default            │ ← Scaled by phase/instrument/TF
│    - phase_maturity blending        │
│    - instrument_volatility scaling  │
│    - timeframe_scale multiplier     │
└─────────────────────────────────────┘
    ↓ (ultimate fallback)
┌─────────────────────────────────────┐
│ 5. Module Constant                  │ ← Universal baseline
│    TARGET_CAPTURE_RATIO = 0.7       │
└─────────────────────────────────────┘
```

---

## Fuzzy Phase Transitions

### Problem: Binary Switches Create Discontinuities
❌ **Before**: `if paper_mode: x=30 else: x=100`  
→ Sudden jump from 30→100 when switching to live

✅ **After**: Smooth interpolation
```python
phase_maturity = 0.4  # 40% toward exploitation
early_value = 30
late_value = 100
blended = early_value * (1 - 0.4) + late_value * 0.4
# = 30 * 0.6 + 100 * 0.4 = 18 + 40 = 58
```

### Usage Scenarios

**Early Training** (phase_maturity=0.0):
- Max exploration boost
- Short inactivity tolerance
- Aggressive entry signals
- Tight regime bands

**Mid Training** (phase_maturity=0.5):
- Balanced exploration/exploitation
- Moderate parameters
- Adaptive regime detection

**Live Trading** (phase_maturity=1.0):
- Conservative exploration
- Long inactivity tolerance  
- Strict entry filters
- Wide regime bands

---

## Instrument Adaptation Examples

### BTCUSD (High Volatility)
```python
regime = RegimeDetector(instrument_volatility=2.0)
# Trending threshold: 0.7 * 2.0 = 1.4
# Wider bands prevent false regime switches
```

### EURUSD (Low Volatility)
```python
regime = RegimeDetector(instrument_volatility=0.6)
# Trending threshold: 0.7 * 0.6 = 0.42
# Tighter bands catch subtle regime changes
```

---

## Best Practices

### ✅ DO:
1. **Pass context** (symbol, timeframe, broker) to all components
2. **Use LearnedParametersManager** for instrument-specific tuning
3. **Blend phases** gradually via `phase_maturity` parameter
4. **Log parameter sources** for debugging
5. **Scale thresholds** by instrument/timeframe characteristics

### ❌ DON'T:
1. Hardcode values in business logic
2. Use binary mode switches (`if paper_mode:`)
3. Assume one-size-fits-all thresholds
4. Ignore timeframe differences (M1 ≠ H1)
5. Skip parameter source logging

---

## Migration Guide

### Old Pattern (Hardcoded)
```python
if mae_pct >= 0.20:  # Magic number!
    return 1  # CLOSE
```

### New Pattern (Adaptive)
```python
# Module-level default (fallback)
STOP_LOSS_PCT_DEFAULT: float = 0.20

# In class __init__:
timeframe_scale = self._get_timeframe_scale()
self.stop_loss_pct = self._get_param(
    "harvester_stop_loss_pct",
    STOP_LOSS_PCT_DEFAULT * timeframe_scale  # Context-aware default
)

# In method:
if mae_pct >= self.stop_loss_pct:  # Adaptive threshold
    return 1  # CLOSE
```

---

## Testing Phase Transitions

```python
# Test smooth exploration decay
for maturity in [0.0, 0.25, 0.5, 0.75, 1.0]:
    monitor = ActivityMonitor(phase_maturity=maturity)
    print(f"Maturity {maturity:.0%}: boost={monitor.exploration_boost:.2f}")

# Output:
# Maturity   0%: boost=0.30
# Maturity  25%: boost=0.25
# Maturity  50%: boost=0.20
# Maturity  75%: boost=0.15
# Maturity 100%: boost=0.10
```

---

## Summary

**Constants provide sensible defaults, but the system adapts per:**
- 📊 **Trading context** (instrument, timeframe, broker)
- 🎯 **Phase maturity** (exploration → exploitation)
- 🌍 **Market regime** (trending vs mean-reverting)
- ⚙️ **Deployment mode** (paper vs live)

**Result**: Same codebase behaves optimally across BTCUSD/M1/paper, EURUSD/H1/live, etc., without hardcoded switches.
