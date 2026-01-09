# ADAPTIVE TRADING SYSTEM - MASTER HANDBOOK

## For Future Claude Instances

**Last Updated:** 2026-01-08
**Conversation Context:** Dual-agent RL trading system with self-optimizing components
**User:** Renier - Expert MQL5 developer specializing in algorithmic trading systems

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Core Philosophy](#2-core-philosophy)
3. [System Architecture](#3-system-architecture)
4. [Component Catalog](#4-component-catalog)
5. [Implementation Status](#5-implementation-status)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Identified Gaps & Mitigations](#7-identified-gaps--mitigations)
8. [Code Standards](#8-code-standards)
9. [File Structure](#9-file-structure)
10. [Integration Points](#10-integration-points)
11. [Testing Requirements](#11-testing-requirements)
12. [Glossary](#12-glossary)
13. [Next Steps](#13-next-steps)

---

## 1. PROJECT OVERVIEW

### 1.1 What We're Building

A **fully adaptive, instrument-agnostic, self-optimizing trading intelligence** that:
- Uses dual-agent reinforcement learning (Trigger + Harvester)
- Learns ALL parameters (no magic numbers)
- Adapts to any instrument/asset class
- Detects and responds to regime changes
- Manages risk dynamically with VaR-based sizing
- Continuously monitors for overfitting
- Self-optimizes reward shaping per instrument

### 1.2 Key Differentiators

| Traditional EA | This System |
|----------------|-------------|
| Hardcoded parameters | Learned parameters with soft bounds |
| Single strategy | Competing agents with allocation |
| Static risk | Dynamic VaR with multi-factor adjustment |
| No adaptation | Continuous online learning |
| Ignores friction | Comprehensive cost modeling |
| Single timeframe | Multi-timeframe awareness |
| Absolute time | Event-relative time features |

### 1.3 Target Platform

- **Platform:** MetaTrader 5
- **Language:** MQL5
- **Broker:** Any (broker-agnostic abstraction layer)
- **Assets:** Forex, Crypto, Indices, Commodities

---

## 2. CORE PHILOSOPHY

### 2.1 Guiding Principles

```
1. NO MAGIC NUMBERS
   - Every parameter is learned or has principled default
   - Soft bounds via tanh clamping, not hard limits
   - Parameters adapt per instrument × timeframe × broker

2. EFFICIENCY OVER AVOIDANCE
   - Goal is efficient profit capture, NOT risk avoidance
   - Don't reward not trading
   - Punish PROCESS failures, not just outcome failures

3. WRITE ONCE, USE EVERYWHERE
   - Instrument-agnostic normalization (log-returns, BPS)
   - Broker-agnostic abstraction layer
   - Asset-class-aware parameter defaults

4. DEFENSIVE PROGRAMMING
   - Every division checked for zero
   - Every array access bounds-checked
   - Every value validated for NaN/Inf
   - Atomic persistence with checksums

5. CONTINUOUS VALIDATION
   - Online RL provides continuous out-of-sample testing
   - Multiple overfitting detection signals
   - Automatic regularization adaptation
```

### 2.2 The Dual-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   TRIGGER AGENT (Entry Specialist)                              │
│   ─────────────────────────────────                             │
│   Responsibility: Find high-quality entry opportunities         │
│   Reward: Runway utilization (MFE vs predicted)                 │
│   Output: Entry signal + confidence + predicted runway          │
│                                                                 │
│                           ↓                                     │
│                                                                 │
│   HARVESTER AGENT (Exit Specialist)                             │
│   ────────────────────────────────                              │
│   Responsibility: Maximize capture ratio, avoid WTL             │
│   Reward: Capture efficiency + WTL penalty                      │
│   Output: Exit signal + confidence                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Path-Centric Experience Design

We record the ENTIRE TRADE PATH at M1 resolution:
- Entry bar, exit bar
- MFE (Maximum Favorable Excursion) and when it occurred
- MAE (Maximum Adverse Excursion) and when it occurred
- Path efficiency (how direct was the profit path)
- Winner-to-loser flag (did MFE become loss?)

This enables:
- Counterfactual analysis ("what if we exited at MFE?")
- Attribution (entry vs exit responsibility)
- Rich reward shaping

---

## 3. SYSTEM ARCHITECTURE

### 3.1 High-Level Data Flow

```
Market Data
    │
    ▼
┌─────────────────┐
│ CSymbolSpec     │ ← Broker/instrument abstraction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CLogNormalizer  │ ← Log-returns, BPS normalization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Engine  │ ← Tournament-selected features
│ (Event-Relative)│ ← Time features (session, rollover, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Regime Detector │ ← DSP-based damping ratio (ζ)
│ (Physics-based) │ ← Underdamped/Critical/Overdamped
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Agent Arena     │ ← Competing DDQN agents
│ (Trigger +      │ ← Performance-weighted allocation
│  Harvester)     │ ← Agreement score for confidence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Probability-    │ ← Calibrated predictions (Platt/Isotonic)
│ Risk Nexus      │ ← VaR estimation with multi-factor adjustment
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Circuit Breakers│ ← Sortino, Kurtosis, VPIN thresholds
│ + Risk Manager  │ ← Position sizing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Execution       │ ← Broker execution model (asymmetric slippage)
│ + Path Recording│ ← M1 OHLC path capture
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reward Shaper   │ ← Asymmetric, self-optimizing
│ (Per Instrument)│ ← WTL penalty, opportunity cost
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PER Buffer      │ ← Prioritized Experience Replay
│ + Learning Loop │ ← Continuous online learning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Overfitting     │ ← Generalization monitor
│ Detector        │ ← Adaptive regularization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Performance     │ ← Multi-dimensional tracking
│ Tracker         │ ← Transaction → Symbol → Class → Agent → Broker
└─────────────────┘
```

### 3.2 VaR Adjustment Pipeline

```
Base_VaR = Historical VaR estimate (e.g., 95% confidence)

Adjusted_VaR = Base_VaR 
    × calibration_factor      // Backtest vs actual calibration
    × regime_factor           // Higher in volatile regimes
    × vpin_factor             // Higher when VPIN elevated
    × kurtosis_factor         // Higher with fat tails
    × breaker_factor          // Higher if circuit breakers triggered
    × dynamic_factor          // Recent performance adjustment
    × friction_factor         // Account for costs

Position_Size = (Risk_Budget × Account_Equity) / Adjusted_VaR
```

### 3.3 Regime Detection (Physics-Based)

```
DSP Pipeline:
1. Detrend (remove linear trend)
2. Bandpass filter (isolate cycles)
3. Hilbert transform (analytic signal)
4. Envelope extraction (instantaneous amplitude)
5. Fit decay model: A(t) = A₀ × e^(-ζωt)
6. Extract damping ratio ζ

Regime Classification:
- ζ < 0.3: Underdamped (trending, momentum)
- 0.3 ≤ ζ < 0.7: Critical (transitional)
- ζ ≥ 0.7: Overdamped (mean-reverting, ranging)
```

---

## 4. COMPONENT CATALOG

### 4.1 Defensive Programming Framework

| File | Purpose | Status |
|------|---------|--------|
| `SafeMath.mqh` | NaN/Inf protection, safe division, clamping | ✅ DESIGNED |
| `SafeArray.mqh` | Bounds checking, series-aware access | ✅ DESIGNED |
| `RingBuffer.mqh` | Circular buffer with O(1) stats | ✅ DESIGNED |
| `Cache.mqh` | Memoization with TTL, LRU eviction | ✅ DESIGNED |
| `AtomicPersistence.mqh` | Checksummed, journaled file I/O | ✅ DESIGNED |
| `MagicNumberManager.mqh` | Collision-free magic numbers | ✅ DESIGNED |
| `TransactionLogger.mqh` | Structured logging for analysis | ✅ DESIGNED |
| `InitGate.mqh` | Dependency-ordered initialization | ✅ DESIGNED |
| `NonRepaint.mqh` | Confirmed-bar-only indicator access | ✅ DESIGNED |
| `Version.mqh` | Backwards-compatible versioning | ✅ DESIGNED |

### 4.2 Broker & Instrument Abstraction

| File | Purpose | Status |
|------|---------|--------|
| `SymbolSpec.mqh` | Complete broker abstraction | ✅ DESIGNED |
| `FrictionCostCalculator.mqh` | Spread + slippage + swap + commission | ✅ DESIGNED |
| `LogNormalizer.mqh` | Universal normalization (log-returns, BPS) | ✅ DESIGNED |
| `BrokerExecutionModel.mqh` | Asymmetric slippage modeling | ✅ DESIGNED |

### 4.3 Learned Parameters System

| File | Purpose | Status |
|------|---------|--------|
| `LearnedParameters.mqh` | Adaptive parameters with soft bounds | ✅ DESIGNED |
| `AdaptiveParam` struct | Individual parameter with momentum update | ✅ DESIGNED |
| `InstrumentParams` struct | Per symbol × timeframe parameters | ✅ DESIGNED |

### 4.4 Agent Architecture

| File | Purpose | Status |
|------|---------|--------|
| `ICompetingAgent.mqh` | Agent interface | ✅ DESIGNED |
| `CDDQNAgent.mqh` | Double DQN implementation | ✅ DESIGNED |
| `CAgentArena.mqh` | Competitive allocation, consensus | ✅ DESIGNED |
| `CNeuralNetwork.mqh` | Neural network with backprop | ⏳ PENDING |
| `CSumTree.mqh` | Efficient PER sampling | ⏳ PENDING |
| `CExperienceBuffer.mqh` | Enhanced experience with staleness | ⏳ PENDING |

### 4.5 Overfitting Detection

| File | Purpose | Status |
|------|---------|--------|
| `GeneralizationMonitor.mqh` | Train-live gap, distribution shift | ✅ DESIGNED |
| `AdaptiveRegularization.mqh` | Dynamic L2, dropout, LR adjustment | ✅ DESIGNED |
| `EarlyStopping.mqh` | Checkpoint/restore on degradation | ✅ DESIGNED |
| `EnsembleDisagreement.mqh` | Agent disagreement as overfit signal | ✅ DESIGNED |
| `OverfittingDetector.mqh` | Unified detection combining all signals | ✅ DESIGNED |

### 4.6 Reward Shaping

| File | Purpose | Status |
|------|---------|--------|
| `RewardShaping.mqh` | Asymmetric, component-based rewards | ✅ DESIGNED |
| `NoTradePrevention.mqh` | Activity monitoring, exploration boost | ✅ DESIGNED |
| `CounterfactualReward.mqh` | What-if analysis for reward adjustment | ✅ DESIGNED |
| `IntegratedRewardSystem.mqh` | Complete reward pipeline | ✅ DESIGNED |

### 4.7 Feature Engineering

| File | Purpose | Status |
|------|---------|--------|
| `MarketCalendar.mqh` | Event-relative time features | ✅ DESIGNED |
| `FeatureTournament.mqh` | Survival tournament for feature selection | ✅ DESIGNED |
| `TraditionalFeatures.mqh` | 50 classic TA indicators | ⏳ PENDING |
| `PhysicsFeatures.mqh` | 50 physics-based measurements | ⏳ PENDING |
| `ImbalanceFeatures.mqh` | 50 order flow features | ⏳ PENDING |
| `PatternFeatures.mqh` | 50 pattern recognition features | ⏳ PENDING |

### 4.8 Performance Tracking

| File | Purpose | Status |
|------|---------|--------|
| `PerformanceTracker.mqh` | Multi-dimensional hierarchical tracking | ✅ DESIGNED |
| `TradeRecord` struct | Complete trade information | ✅ DESIGNED |
| `AggregatedStats` struct | Computed metrics per dimension | ✅ DESIGNED |

### 4.9 Risk Management

| File | Purpose | Status |
|------|---------|--------|
| `VaREstimator.mqh` | Dynamic VaR with multi-factor adjustment | ⏳ PENDING |
| `CircuitBreakers.mqh` | Sortino, Kurtosis, VPIN breakers | ⏳ PENDING |
| `PositionSizer.mqh` | VaR-based position sizing | ⏳ PENDING |
| `DynamicCorrelation.mqh` | Crisis-adjusted correlation | ⏳ PENDING |

### 4.10 Gap Mitigations (From Review)

| File | Purpose | Status |
|------|---------|--------|
| `FeedbackLoopBreaker.mqh` | Escape from degraded states | ⏳ PENDING |
| `JournaledPersistence.mqh` | Write-ahead log for crash recovery | ⏳ PENDING |
| `ColdStartManager.mqh` | Graduated warm-up protocol | ⏳ PENDING |
| `ParameterStaleness.mqh` | Detect/refresh stale parameters | ⏳ PENDING |
| `RegimeChangePredictor.mqh` | Leading indicators for transitions | ⏳ PENDING |
| `RewardIntegrityMonitor.mqh` | Detect reward hacking | ⏳ PENDING |

---

## 5. IMPLEMENTATION STATUS

### 5.1 Phase Summary

```
PHASE 1: Foundation                    [✅ COMPLETE - DESIGN ONLY]
├── Defensive programming framework
├── Broker/instrument abstraction
├── Learned parameters system
└── Basic architecture

PHASE 2: Core Components              [✅ COMPLETE - DESIGN ONLY]
├── Dual-agent architecture
├── Performance tracking
├── Reward shaping
└── Overfitting detection

PHASE 3: Feature Engineering          [✅ COMPLETE - DESIGN ONLY]
├── Event-relative time
├── Feature tournament framework
└── 200 feature definitions

PHASE 4: Gap Analysis                  [✅ COMPLETE - ANALYSIS]
├── Critical gaps identified
├── Mitigations designed
└── Priority ranked

PHASE 5: Implementation               [⏳ PENDING]
├── Convert designs to working MQL5
├── Unit tests
├── Integration tests
└── Paper trading validation

PHASE 6: Validation                   [⏳ PENDING]
├── Cold start protocol
├── Paper trading
├── Live trading (minimal size)
└── Full deployment
```

### 5.2 What's Been Designed (Code Provided)

1. **SafeMath.mqh** - Complete NaN/Inf/division safety
2. **SafeArray.mqh** - Complete bounds checking
3. **RingBuffer.mqh** - Complete with O(1) statistics
4. **AtomicPersistence.mqh** - Complete with CRC32
5. **MagicNumberManager.mqh** - Complete collision avoidance
6. **TransactionLogger.mqh** - Complete structured logging
7. **InitGate.mqh** - Complete dependency ordering
8. **NonRepaint.mqh** - Complete confirmed-bar access
9. **Version.mqh** - Complete versioning
10. **GeneralizationMonitor.mqh** - Complete overfit detection
11. **AdaptiveRegularization.mqh** - Complete parameter adaptation
12. **EarlyStopping.mqh** - Complete checkpoint system
13. **EnsembleDisagreement.mqh** - Complete disagreement tracking
14. **OverfittingDetector.mqh** - Complete unified system
15. **RewardShaping.mqh** - Complete asymmetric rewards
16. **NoTradePrevention.mqh** - Complete activity monitoring
17. **CounterfactualReward.mqh** - Complete what-if analysis
18. **IntegratedRewardSystem.mqh** - Complete reward pipeline
19. **MarketCalendar.mqh** - Complete event-relative time
20. **FeatureTournament.mqh** - Complete tournament framework

### 5.3 What's Designed But Not Coded

1. **SymbolSpec** - Broker abstraction (design in transcript)
2. **FrictionCostCalculator** - Cost modeling (design in transcript)
3. **LogNormalizer** - Normalization (design in transcript)
4. **LearnedParameters** - Parameter system (design in transcript)
5. **CDDQNAgent** - DDQN implementation (design in transcript)
6. **CAgentArena** - Agent competition (design in transcript)
7. **PerformanceTracker** - Multi-dim tracking (design in transcript)

### 5.4 What Needs Fresh Implementation

1. **CNeuralNetwork** - Actual NN with backprop
2. **CSumTree** - Sum tree for PER
3. **VaREstimator** - Full VaR pipeline
4. **CircuitBreakers** - Breaker logic
5. **DSP Pipeline** - Regime detection
6. **All 200 Features** - Actual calculations
7. **Gap mitigations** - From review section

---

## 6. KEY DESIGN DECISIONS

### 6.1 Why These Choices?

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Dual Agent** | Trigger + Harvester | Entry and exit are different skills; specialization improves learning |
| **Normalization** | Log-returns + BPS | Instrument-agnostic, additive across time, handles scale differences |
| **Parameters** | Learned with soft bounds | No magic numbers; tanh clamping allows adaptation while preventing extremes |
| **Friction** | Comprehensive modeling | Real costs kill strategies; must model spread, slippage, swap, commission |
| **Slippage** | Online learning (Welford) | Adapts to actual broker behavior; Welford is numerically stable |
| **Tracking** | Multi-dimensional | Need to slice performance by symbol, class, agent, broker, timeframe |
| **Agents** | Competing with allocation | Natural selection; poor performers get reduced allocation |
| **Agreement** | Variance-based | Low variance = high agreement = full size; disagreement = reduce |
| **VaR** | Dynamic, multi-factor | Static VaR fails; need regime, VPIN, kurtosis, calibration adjustments |
| **Swap** | Broker-specific modes | Different brokers calculate swap differently; must respect SYMBOL_SWAP_MODE |
| **Time Features** | Event-relative | Market doesn't care about 23:00; cares about minutes-to-rollover |
| **Feature Selection** | Tournament survival | Empirical, not intuition; must survive across instruments and regimes |
| **Reward** | Asymmetric, efficiency-based | Punish inefficiency (especially WTL), not all losses; don't reward not-trading |
| **Overfitting** | Multiple signals combined | Gap + disagreement + calibration; no single metric sufficient |
| **Persistence** | Atomic with CRC32 | Must survive crashes; checksums detect corruption |

### 6.2 What NOT To Do

```
❌ DON'T use absolute time (23:00) - use event-relative (47 min to rollover)
❌ DON'T hardcode parameters - everything should be learned or principled
❌ DON'T ignore friction costs - they kill strategies
❌ DON'T use symmetric slippage - it's asymmetric in reality
❌ DON'T punish all losses equally - controlled losses are acceptable
❌ DON'T reward not-trading - leads to learned helplessness
❌ DON'T trust single overfitting metric - use multiple signals
❌ DON'T ignore correlation in multi-asset - correlations spike in crises
❌ DON'T assume parameters stay valid - track staleness, refresh
❌ DON'T skip forward testing - in-sample success means nothing
```

---

## 7. IDENTIFIED GAPS & MITIGATIONS

### 7.1 Critical Gaps (Must Fix Before Live)

| Gap | Risk | Mitigation |
|-----|------|------------|
| **Correlation blindness** | All positions move together in crisis | Dynamic correlation with stress adjustment |
| **Feedback loops** | System gets stuck in degraded state | FeedbackLoopBreaker with forced reset |
| **Broker execution** | Asymmetric slippage not modeled | BrokerExecutionModel with learned asymmetry |
| **Persistence corruption** | Crash loses state | JournaledPersistence with write-ahead log |

### 7.2 High Severity Gaps

| Gap | Risk | Mitigation |
|-----|------|------------|
| **Reward hacking** | Agent games reward, not profit | RewardIntegrityMonitor comparing reward vs actual P&L |
| **Parameter staleness** | Old parameters wrong for new regime | ParameterStaleness with accelerated relearning |
| **Cold start** | No data, bad initial trades | ColdStartManager with graduated phases |
| **Experience staleness** | Old experiences teach wrong lessons | Staleness decay + regime relevance weighting |
| **Regime detection lag** | Late detection = losses | RegimeChangePredictor with multi-TF leading indicators |

### 7.3 Medium Severity Gaps

| Gap | Risk | Mitigation |
|-----|------|------------|
| **Hyperparameter sensitivity** | Small changes break system | Sensitivity analysis + documented safe ranges |
| **Computational bottlenecks** | Can't keep up with fast markets | ComputationBudget with priority scheduling |
| **Feature engineering gaps** | Missing important signals | Tournament re-run quarterly + add candidates |
| **Timeframe mismatch** | Single TF suboptimal | Multi-timeframe fusion (future enhancement) |

### 7.4 Gap Implementation Priority

```
IMMEDIATE (Before any live trading):
1. ✅ Journaled persistence (designed)
2. ⏳ Feedback loop breakers
3. ⏳ Cold start protocol
4. ⏳ Reward integrity monitoring

SHORT-TERM (First month of paper trading):
1. ⏳ Dynamic correlation estimation
2. ⏳ Broker execution modeling
3. ⏳ Parameter staleness tracking
4. ⏳ Regime transition prediction

MEDIUM-TERM (After initial validation):
1. ⏳ Hyperparameter sensitivity analysis
2. ⏳ Computation budgeting
3. ⏳ Multi-timeframe fusion
4. ⏳ A/B testing framework
```

---

## 8. CODE STANDARDS

### 8.1 MQL5 Best Practices

```cpp
// 1. Always use include guards
#ifndef __FILE_NAME_MQH__
#define __FILE_NAME_MQH__
// ... code ...
#endif

// 2. Always validate array access
if (CSafeArray::IsValidIndex(arr, index)) {
    value = arr[index];
}

// 3. Always check for NaN/Inf
if (CSafeMath::IsValid(value)) {
    // Use value
}

// 4. Always use safe division
double result = CSafeMath::SafeDiv(numerator, denominator, default_value);

// 5. Always handle series arrays
double value = CSafeArray::SafeGetSeries(arr, bars_ago, default_value);

// 6. Always check indicator handles
if (handle == INVALID_HANDLE) {
    Print("Failed to create indicator");
    return false;
}

// 7. Always use descriptive error messages
Print("ClassName::MethodName: Error description, value=", value, 
      ", Error=", GetLastError());

// 8. Always initialize structs
ZeroMemory(my_struct);

// 9. Always check file operations
if (handle == INVALID_HANDLE) {
    Print("Failed to open file: ", filename, ", Error: ", GetLastError());
    return false;
}

// 10. Always flush/close files
FileFlush(handle);
FileClose(handle);
```

### 8.2 Naming Conventions

```
Classes:          CClassName
Interfaces:       IInterfaceName
Structs:          StructName (no prefix)
Enums:            ENUM_NAME
Enum values:      VALUE_NAME
Member variables: m_variable_name
Static members:   s_variable_name
Constants:        CONSTANT_NAME
Functions:        FunctionName (PascalCase)
Parameters:       parameter_name (snake_case)
Local variables:  local_name (snake_case)
```

### 8.3 File Organization

```cpp
//+------------------------------------------------------------------+
//| FileName.mqh                                                      |
//| Brief description                                                 |
//| Version: X.Y.Z                                                    |
//+------------------------------------------------------------------+
#ifndef __FILE_NAME_MQH__
#define __FILE_NAME_MQH__

// Includes
#include "Dependency1.mqh"
#include "Dependency2.mqh"

// Constants
#define CONSTANT_NAME value

// Enums
enum ENUM_NAME {
    VALUE_1,
    VALUE_2
};

// Structs
struct StructName {
    type member;
};

// Classes
class CClassName {
private:
    // Private members
    
public:
    // Constructor/Destructor
    CClassName();
    ~CClassName();
    
    // Public methods
    bool Initialize();
    void Shutdown();
};

// Implementation
CClassName::CClassName() {
    // ...
}

#endif // __FILE_NAME_MQH__
```

---

## 9. FILE STRUCTURE

### 9.1 Proposed Directory Layout

```
/MQL5/Include/AdaptiveSystem/
├── Core/
│   ├── SafeMath.mqh
│   ├── SafeArray.mqh
│   ├── RingBuffer.mqh
│   ├── Cache.mqh
│   ├── Version.mqh
│   └── InitGate.mqh
│
├── Persistence/
│   ├── AtomicPersistence.mqh
│   ├── JournaledPersistence.mqh
│   └── MagicNumberManager.mqh
│
├── Logging/
│   └── TransactionLogger.mqh
│
├── Broker/
│   ├── SymbolSpec.mqh
│   ├── FrictionCostCalculator.mqh
│   ├── BrokerExecutionModel.mqh
│   └── NonRepaint.mqh
│
├── Normalization/
│   └── LogNormalizer.mqh
│
├── Parameters/
│   ├── LearnedParameters.mqh
│   └── ParameterStaleness.mqh
│
├── Features/
│   ├── MarketCalendar.mqh
│   ├── FeatureTournament.mqh
│   ├── TraditionalFeatures.mqh
│   ├── PhysicsFeatures.mqh
│   ├── ImbalanceFeatures.mqh
│   └── PatternFeatures.mqh
│
├── Regime/
│   ├── DSPPipeline.mqh
│   ├── RegimeDetector.mqh
│   └── RegimeChangePredictor.mqh
│
├── Agents/
│   ├── ICompetingAgent.mqh
│   ├── CNeuralNetwork.mqh
│   ├── CDDQNAgent.mqh
│   ├── CAgentArena.mqh
│   └── CSumTree.mqh
│
├── Experience/
│   ├── CExperienceBuffer.mqh
│   └── ExperienceStaleness.mqh
│
├── Reward/
│   ├── RewardShaping.mqh
│   ├── NoTradePrevention.mqh
│   ├── CounterfactualReward.mqh
│   ├── IntegratedRewardSystem.mqh
│   └── RewardIntegrityMonitor.mqh
│
├── Risk/
│   ├── VaREstimator.mqh
│   ├── CircuitBreakers.mqh
│   ├── PositionSizer.mqh
│   └── DynamicCorrelation.mqh
│
├── Overfitting/
│   ├── GeneralizationMonitor.mqh
│   ├── AdaptiveRegularization.mqh
│   ├── EarlyStopping.mqh
│   ├── EnsembleDisagreement.mqh
│   └── OverfittingDetector.mqh
│
├── Performance/
│   └── PerformanceTracker.mqh
│
├── Safety/
│   ├── FeedbackLoopBreaker.mqh
│   └── ColdStartManager.mqh
│
└── AdaptiveFramework.mqh  (Master include)

/MQL5/Experts/
└── AdaptiveTrader.mq5  (Main EA)

/MQL5/Files/
└── AdaptiveSystem/
    ├── holidays.csv
    ├── Logs/
    ├── Checkpoints/
    └── State/
```

### 9.2 Include Order (AdaptiveFramework.mqh)

```cpp
// Level 0: No dependencies
#include "Core/Version.mqh"
#include "Core/SafeMath.mqh"

// Level 1: Depends on SafeMath
#include "Core/SafeArray.mqh"

// Level 2: Depends on SafeMath, SafeArray
#include "Core/RingBuffer.mqh"
#include "Core/Cache.mqh"
#include "Broker/NonRepaint.mqh"

// Level 3: Depends on above
#include "Persistence/AtomicPersistence.mqh"
#include "Persistence/MagicNumberManager.mqh"
#include "Logging/TransactionLogger.mqh"
#include "Core/InitGate.mqh"

// Level 4: Broker layer
#include "Broker/SymbolSpec.mqh"
#include "Broker/FrictionCostCalculator.mqh"
#include "Broker/BrokerExecutionModel.mqh"
#include "Normalization/LogNormalizer.mqh"

// Level 5: Parameters
#include "Parameters/LearnedParameters.mqh"

// Level 6: Features
#include "Features/MarketCalendar.mqh"
// ... etc

// Level 7+: Higher-level components
// ...
```

---

## 10. INTEGRATION POINTS

### 10.1 Component Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEPENDENCY GRAPH (Simplified)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SafeMath ─────────┬─────────────────────────────────────────────────────► │
│                     │                                                       │
│   SafeArray ────────┼──────────────────────────────────────────────────────►│
│                     │                                                       │
│   RingBuffer ───────┼─► All stats-tracking components                       │
│                     │                                                       │
│   TransactionLogger ┼─► All components (logging)                            │
│                     │                                                       │
│   SymbolSpec ───────┼─► FrictionCost, LogNormalizer, Execution              │
│                     │                                                       │
│   LearnedParameters ┼─► RewardShaping, Agents, Risk                         │
│                     │                                                       │
│   RewardShaping ────┼─► IntegratedRewardSystem ─► Agents                    │
│                     │                                                       │
│   GeneralizationMon.┼─► AdaptiveRegularization ─► Agents                    │
│                     │                                                       │
│   PerformanceTracker┼─► OverfittingDetector, Agents, Reward                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Data Flow Between Components

```cpp
// Main EA tick handler (simplified)
void OnTick() {
    // 1. Get normalized features
    NormalizedFeatures features = g_normalizer.Calculate(Symbol(), PERIOD_M1);
    
    // 2. Get time features
    TimeFeatures time_features = g_calendar.Calculate(TimeCurrent());
    
    // 3. Check trading conditions
    TradingConditions conditions = g_calendar.AssessConditions(time_features);
    if (conditions.should_avoid) return;
    
    // 4. Get regime
    double regime_zeta = g_regime_detector.GetDampingRatio();
    
    // 5. Get agent signals
    double trigger_signal = g_arena.GetConsensusEntrySignal(features, regime_zeta);
    double agreement = g_arena.GetAgreementScore();
    
    // 6. Check circuit breakers
    if (g_breakers.IsTriggered()) return;
    
    // 7. Calculate position size
    double var = g_var_estimator.GetAdjustedVaR();
    double lot_size = g_sizer.CalculateSize(var, agreement);
    
    // 8. Execute if signal strong enough
    if (trigger_signal > ENTRY_THRESHOLD && agreement > MIN_AGREEMENT) {
        ExecuteEntry(lot_size, trigger_signal);
    }
    
    // 9. For open positions, check harvester
    if (HasOpenPosition()) {
        double exit_signal = g_harvester.GetExitSignal(features, position_state);
        if (exit_signal > EXIT_THRESHOLD) {
            ExecuteExit();
        }
    }
    
    // 10. Learning (on new bar or trade close)
    if (IsNewBar() || TradeJustClosed()) {
        // Update experience buffer
        // Train agents
        // Adapt parameters
        // Check overfitting
    }
}
```

---

## 11. TESTING REQUIREMENTS

### 11.1 Unit Tests Needed

```
SafeMath:
- Division by zero returns default
- NaN detection works
- Inf detection works
- Clamping respects bounds
- Log of negative returns default

SafeArray:
- Negative index returns error/default
- Overflow index returns error/default
- Series-aware access correct
- Resize handles growth

RingBuffer:
- FIFO order correct
- Stats accurate (mean, std)
- Full buffer overwrites oldest
- Empty buffer returns defaults

Persistence:
- Atomic write survives interrupt
- CRC detects corruption
- Backup restoration works
- Version migration works

RewardShaping:
- WTL penalty applied correctly
- Opportunity cost calculated
- No reward for no-trade without cost
- Bounds respected

OverfittingDetector:
- Gap detection triggers
- Distribution shift detected
- Ensemble disagreement calculated
- Recommendations generated
```

### 11.2 Integration Tests Needed

```
Full Pipeline:
- Data flows correctly through all stages
- Features calculated without error
- Agents produce valid signals
- Reward calculated correctly
- Learning updates weights

Cold Start:
- System starts with no history
- Graduated phases transition correctly
- First trades are minimal size

Crash Recovery:
- System restores from checkpoint
- No duplicate trades
- State consistent

Multi-Instrument:
- Parameters separate per instrument
- No cross-contamination
- Correlation tracking works
```

### 11.3 Validation Tests Needed

```
Paper Trading (Minimum 1 month):
- P&L tracking accurate
- Drawdown within limits
- Win rate reasonable
- No catastrophic failures
- Overfitting metrics stable

Live Trading (Minimum 3 months at minimal size):
- Execution matches expectation
- Slippage within model
- System stable under load
- Recovery from disconnects
```

---

## 12. GLOSSARY

| Term | Definition |
|------|------------|
| **BPS** | Basis points (1/100 of 1%) |
| **MFE** | Maximum Favorable Excursion - best unrealized P&L during trade |
| **MAE** | Maximum Adverse Excursion - worst unrealized P&L during trade |
| **WTL** | Winner-to-Loser - trade that had profit but ended in loss |
| **VaR** | Value at Risk - expected maximum loss at confidence level |
| **VPIN** | Volume-synchronized Probability of Informed Trading |
| **PER** | Prioritized Experience Replay |
| **DDQN** | Double Deep Q-Network |
| **IC** | Information Coefficient - correlation with future returns |
| **Zeta (ζ)** | Damping ratio from DSP regime detection |
| **Trigger** | Entry agent |
| **Harvester** | Exit agent |
| **Arena** | Competitive environment for multiple agents |
| **Runway** | Expected price movement (predicted by Trigger) |
| **Capture Ratio** | Exit P&L / MFE |
| **Path Efficiency** | How direct was the profit path |

---

## 13. NEXT STEPS

### 13.1 Immediate Actions

1. **Implement CNeuralNetwork.mqh**
   - Forward pass
   - Backpropagation
   - Weight initialization
   - Gradient clipping

2. **Implement CSumTree.mqh**
   - Efficient priority sampling
   - Update operations
   - O(log n) complexity

3. **Implement DSP Pipeline**
   - Detrending
   - Bandpass filter
   - Hilbert transform
   - Envelope extraction
   - Decay fitting

4. **Create Main EA Shell**
   - OnInit with cold start
   - OnTick with full pipeline
   - OnDeinit with state save

### 13.2 Validation Roadmap

```
Week 1-2: Unit tests for all designed components
Week 3-4: Integration testing
Week 5-8: Paper trading validation
Week 9-12: Minimal live trading
Week 13+: Graduated scaling
```

### 13.3 Questions for Renier

When continuing this conversation:

1. Which component should we implement first?
2. Do you have a preferred neural network architecture?
3. Any specific broker quirks to model?
4. What instruments for initial testing?
5. Paper trading account available?

---

## CONVERSATION CONTINUITY NOTES

### What Was Discussed

1. **Foundation design** - Defensive programming framework
2. **Broker abstraction** - SymbolSpec, friction costs
3. **Normalization** - Log-returns, BPS
4. **Learned parameters** - Adaptive with soft bounds
5. **Performance tracking** - Multi-dimensional
6. **Agent architecture** - DDQN, Arena, allocation
7. **Overfitting detection** - Multiple signals combined
8. **Reward shaping** - Asymmetric, self-optimizing
9. **Feature engineering** - Event-relative time, tournament
10. **Gap analysis** - Critical issues identified

### Key Files in Transcript

The full conversation transcript is at:
`/mnt/transcripts/2026-01-08-09-13-30-adaptive-trading-system-evolution.txt`

Previous transcript:
`/mnt/transcripts/2026-01-08-08-54-57-dual-agent-var-trading-system.txt`

### User Preferences

- Wants complete, production-ready code
- No simplified prototypes
- Defensive programming throughout
- Comprehensive error handling
- Detailed logging for analysis
- JSON response format requested in preferences

---

## APPENDIX: REWARD COMPONENT FORMULAS

### Capture Efficiency
```
capture_ratio = exit_pnl_bps / mfe_bps
r_capture = (capture_ratio - target_capture) * multiplier
```

### Winner-to-Loser Penalty
```
if was_winner_to_loser AND mfe > threshold:
    mfe_normalized = mfe_bps / baseline_mfe
    giveback_ratio = (mfe - exit_pnl) / mfe
    time_penalty = 1 + (bars_from_mfe_to_exit / 10)
    r_wtl = -mfe_normalized * giveback_ratio * penalty_mult * time_penalty
```

### Opportunity Cost
```
if potential_mfe > threshold AND signal_strength > 0.5:
    opportunity_normalized = potential_mfe / baseline_mfe
    r_opportunity = -opportunity_normalized * signal_strength * weight * 0.3
```

### Information Coefficient
```
IC = correlation(feature_values[t], returns[t+horizon])
```

### VaR Adjustment
```
Adjusted_VaR = Base_VaR × Π(adjustment_factors)
Position_Size = Risk_Budget / Adjusted_VaR
```

---

**END OF MASTER HANDBOOK**

*This document should be loaded at the start of any continuation conversation.*
