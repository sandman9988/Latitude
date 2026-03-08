"""
Central Constants Module — Single Source of Truth
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

# Action IDs — Trigger
ACTION_NO_ENTRY: int = 0
ACTION_LONG: int = 1
ACTION_SHORT: int = 2

# Action IDs — Harvester
ACTION_HOLD: int = 0
ACTION_CLOSE: int = 1

# Volatility / friction defaults
DEFAULT_VOLATILITY: float = 0.005
DEFAULT_FRICTION_PCT: float = 0.0015
DEFAULT_QUANTITY: float = 0.10

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

# ── RISK ──────────────────────────────────────────────────────────────────────

# Kurtosis thresholds
# NOTE: The HUD / VaR monitor uses 3.0 (display alert level — "fat tails present").
#       The circuit breaker uses 5.0 (action level — "halt trading").
#       This two-tier design is intentional: warn early, act on extremes.
KURTOSIS_ALERT_THRESHOLD: float = 3.0      # HUD / VaR display warning
KURTOSIS_BREAKER_THRESHOLD: float = 5.0    # Circuit breaker trip

KURTOSIS_MIN_SAMPLES: int = 30
DEFAULT_COOLDOWN_MINUTES: int = 60
SORTINO_THRESHOLD: float = 0.5
CONSEC_LOSSES_MAX: int = 5

# Timer-based HUD export (Change A)
HUD_EXPORT_INTERVAL_CYCLES: int = 3     # Health-monitor cycles between HUD exports (3 × 10s = 30s)
HUD_EXPORT_MIN_INTERVAL_S: float = 2.0  # Global rate-limit: skip export if called within this window

# Tick-level drawdown circuit breaker (Change B)
TICK_DRAWDOWN_CHECK_INTERVAL_S: float = 1.0  # Min seconds between tick-level drawdown checks

# ── HARVESTER EXIT THRESHOLDS ─────────────────────────────────────────────────
# All values are percentages (0.35 = 0.35% of entry price).
# Adapted at runtime by _init_exit_thresholds() using a timeframe scale factor.

SOFT_TIME_STOP_BARS: int = 200          # Soft time stop threshold (bars held)
HARD_TIME_STOP_BARS: int = 400          # Hard time stop limit (bars held)
MIN_HOLD_TICKS_DEFAULT: int = 10        # Min ticks before DDQN close is allowed
MIN_SOFT_PROFIT_PCT: float = 0.20       # Min unrealized profit % for soft-time-stop exit

PROFIT_TARGET_PCT_DEFAULT: float = 0.45 # Target profit as % of entry price
STOP_LOSS_PCT_DEFAULT: float = 0.40     # Max adverse excursion % before forced exit

BREAKEVEN_TRIGGER_PCT: float = 0.40     # MFE % to move stop to breakeven
TRAILING_STOP_ACTIVATION_PCT: float = 0.35  # MFE % required to activate trailing stop
TRAILING_STOP_DISTANCE_PCT: float = 0.15    # Distance to trail behind peak MFE

CAPTURE_DECAY_THRESHOLD: float = 0.35   # Exit if current_profit/MFE ratio < this
CAPTURE_DECAY_MIN_MFE_PCT: float = 0.10 # Apply capture-decay only above this MFE %

MICRO_WINNER_MFE_THRESHOLD_PCT: float = 0.05  # Min MFE to activate micro-winner protection
MICRO_WINNER_GIVEBACK_PCT: float = 0.30       # Exit if giving back > this fraction of MFE
