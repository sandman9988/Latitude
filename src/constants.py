"""Central Constants Module — Single Source of Truth.
==================================================
All magic numbers and shared thresholds live here.
Every consuming module MUST import from this file instead of
hard-coding duplicate values.

Domain groups:
  - TRADING:   Feature computation, action IDs, volatility defaults
  - TRAINING:  DDQN hyper-parameters, buffer sizes, batch sizes
  - RISK:      Circuit-breaker thresholds, kurtosis, drawdown
"""

# ── TRADING ───────────────────────────────────────────────────────────────────

# Feature calculation
MIN_BARS_FOR_FEATURES: int = 70
RETURN_LAG_SHORT: int = 2
RETURN_LAG_MEDIUM: int = 6
STATE_WINDOW_SIZE: int = 64

# Volatility / friction defaults
DEFAULT_VOLATILITY: float = 0.005
DEFAULT_FRICTION_PCT: float = 0.0015

# ── TRAINING ──────────────────────────────────────────────────────────────────

# DDQN hyper-parameters (shared by TriggerAgent, HarvesterAgent, DDQNNetwork)
LEARNING_RATE: float = 0.0005
GAMMA: float = 0.99
TAU: float = 0.005
L2_WEIGHT: float = 0.0001
GRAD_CLIP_NORM: float = 1.0
TD_ERROR_CAP: float = 10.0

# Experience replay buffer capacities
TRIGGER_BUFFER_CAPACITY: int = 2_000
HARVESTER_BUFFER_CAPACITY: int = 10_000

# Batch / training thresholds
MIN_EXPERIENCES: int = 32
DEFAULT_BATCH_SIZE: int = 64

# Training log cadence
TRAINING_LOG_INTERVAL_EARLY: int = 10
TRAINING_LOG_INTERVAL_LATE: int = 100
TRAINING_STEPS_EARLY: int = 100

# Paper-training exploration defaults. Keep these deliberately high so paper
# fleet learning keeps collecting varied reward data across all timeframes.
PAPER_EPSILON_START: float = 1.0
PAPER_EPSILON_END: float = 0.25
PAPER_EPSILON_DECAY: float = 0.9998
PAPER_FORCE_EXPLORATION: bool = True

# Live/production exploration defaults.
LIVE_EPSILON_START: float = 0.05
LIVE_EPSILON_END: float = 0.01
LIVE_EPSILON_DECAY: float = 0.9995

# ── AMD GPU DETECTION ────────────────────────────────────────────────────────
# Auto-detect AMD GPU and apply optimizations
# This is done at import time in ddqn_network.py via AMD_OPTS


def get_amd_optimized_batch_size(state_dim: int, default: int = DEFAULT_BATCH_SIZE) -> int:
    """Get optimal batch size for AMD GPUs based on state dimension.

    Args:
        state_dim: Dimension of state vector
        default: Default batch size if not AMD GPU

    Returns:
        Optimal batch size for the hardware

    """
    try:
        from src.constants_amd import get_optimal_batch_size, is_amd_gpu

        if is_amd_gpu():
            return get_optimal_batch_size(state_dim)
    except ImportError:
        pass
    return default


def get_amd_optimized_buffer_capacity(default: int) -> int:
    """Get optimal buffer capacity for AMD GPUs.

    Args:
        default: Default buffer capacity

    Returns:
        Optimal buffer capacity for the hardware (50K for AMD, default otherwise)

    """
    try:
        from src.constants_amd import AMD_TRIGGER_BUFFER_CAPACITY, is_amd_gpu

        if is_amd_gpu():
            return AMD_TRIGGER_BUFFER_CAPACITY
    except ImportError:
        pass
    return default


# ── RISK ──────────────────────────────────────────────────────────────────────

# Kurtosis thresholds
# NOTE: The HUD / VaR monitor uses 3.0 (display alert level — "fat tails present").
#       The circuit breaker uses 5.0 (action level — "halt trading").
#       This two-tier design is intentional: warn early, act on extremes.
KURTOSIS_ALERT_THRESHOLD: float = 3.0  # HUD / VaR display warning
KURTOSIS_BREAKER_THRESHOLD: float = 5.0  # Circuit breaker trip

KURTOSIS_MIN_SAMPLES: int = 30
DEFAULT_COOLDOWN_MINUTES: int = 60
SORTINO_THRESHOLD: float = 0.5
CONSEC_LOSSES_MAX: int = 5

# ── HARVESTER EXIT THRESHOLDS ─────────────────────────────────────────────────
# Cold-start defaults only — overridden at runtime by LearnedParametersManager.
# harvester_agent._get_param() pulls from learned_parameters.json first;
# these values are used only if the parameter has never been learned.
# All values are percentages (0.35 = 0.35% of entry price).
# Adapted at runtime by _init_exit_thresholds() using a timeframe scale factor.

SOFT_TIME_STOP_BARS: int = 200  # Soft time stop threshold (bars held)
HARD_TIME_STOP_BARS: int = 400  # Hard time stop limit (bars held)
MIN_HOLD_TICKS_DEFAULT: int = 10  # Min ticks before DDQN close is allowed
MIN_SOFT_PROFIT_PCT: float = 0.20  # Min unrealized profit % for soft-time-stop exit

PROFIT_TARGET_PCT_DEFAULT: float = 0.45  # Target profit as % of entry price
STOP_LOSS_PCT_DEFAULT: float = 0.40  # Max adverse excursion % before forced exit

BREAKEVEN_TRIGGER_PCT: float = 0.30  # MFE % to move stop to breakeven
TRAILING_STOP_ACTIVATION_PCT: float = 0.25  # MFE % required to activate trailing stop
TRAILING_STOP_DISTANCE_PCT: float = 0.12  # Distance to trail behind peak MFE

CAPTURE_DECAY_THRESHOLD: float = 0.35  # Exit if current_profit/MFE ratio < this
CAPTURE_DECAY_MIN_MFE_PCT: float = 0.10  # Apply capture-decay only above this MFE %

MICRO_WINNER_MFE_THRESHOLD_PCT: float = 0.10  # Min MFE to activate micro-winner protection
MICRO_WINNER_GIVEBACK_PCT: float = 0.40  # Exit if giving back > this fraction of MFE

# Hard per-trade max-loss cap expressed as a multiple of 1R (the position's
# own expected stop-loss in USD).  Instrument- and size-agnostic: scales
# automatically with qty, contract_size, and entry price.
# cap_usd = entry_price × (STOP_LOSS_PCT_DEFAULT/100) × qty × contract_size × MAX_LOSS_MULT_PER_TRADE
# Clamped to [MIN_CAP_USD, MAX_CAP_USD] so edge-case tiny or huge positions
# don't produce absurd thresholds.
#
# XAUUSD (qty=0.01, cs=100, ep≈3300): 1R≈$13.2  → cap≈$66  (was fixed $100)
# BTCUSD (qty=0.01, cs=1,   ep≈114k): 1R≈$4.6   → cap≈$23  (was fixed $100, never fired)
MAX_LOSS_MULT_PER_TRADE: float = 5.0   # Hard cap = 5× the position's expected stop loss
MIN_CAP_USD: float = 2.0               # Absolute minimum cap (prevents near-zero on tiny lots)
MAX_CAP_USD: float = 200.0             # Absolute maximum cap (prevents runaway on large lots)

# ── CAPTURE HEALTH MONITORING ─────────────────────────────────────────────────
# Two-tier reactive system:
#   Tier 1 — IMMEDIATE: single large-delta trade (big MFE, tiny capture) → act at once.
#   Tier 2 — ROLLING EMA: persistent low capture over N TF-adaptive trades → act.
# Stable recovery: small relax requiring 2× min samples to prevent whipsawing.

CAPTURE_EMA_ALPHA: float = 0.35          # Fast EMA — half-life ≈ 2 trades
CAPTURE_LARGE_DELTA_MFE_MULT: float = 1.5  # "Large" = MFE > 1.5× trailing activation pct
CAPTURE_LARGE_DELTA_CAP_MAX: float = 0.20  # Immediate trigger if capture < 20% of large MFE
CAPTURE_ALERT_THRESHOLD: float = 0.25    # Rolling EMA alert level → tighten
CAPTURE_CRITICAL_THRESHOLD: float = 0.10  # Rolling EMA critical → emergency reset
CAPTURE_STABLE_THRESHOLD: float = 0.55   # Above this = healthy, relax very slowly
CAPTURE_TIGHTEN_IMMEDIATE: float = 0.75  # Factor on large delta: scale to 75% of current
CAPTURE_TIGHTEN_ALERT: float = 0.82      # Factor on rolling alert: scale to 82% of current
CAPTURE_RELAX_FACTOR: float = 0.97       # Relax 3% per trade when stably healthy
