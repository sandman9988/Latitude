#!/usr/bin/env python3
"""Tabbed Trading HUD (Heads-Up Display).
=====================================
Terminal-based live dashboard with multiple tabs for organized data display.

Tabs:
  1 - Overview (compact summary)
  2 - Performance (detailed metrics)
  3 - Training (agent stats)
  4 - Risk (risk management)
  5 - Market (microstructure)
  6 - Decision Log (last 20 entries)
  7 - Trades (closed-trade ledger)

Press 1-7 or ←/→ to switch tabs, Tab/Shift+Tab to cycle, s for presets, q or Ctrl+X to quit.
Note: Ctrl+C is ignored to prevent accidental termination when copying text.
"""

import contextlib
import io
import json
import logging
import os
import re
import select
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
from collections import deque
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from src.constants import (
    HARVESTER_BUFFER_CAPACITY,
    KURTOSIS_ALERT_THRESHOLD,
    TRIGGER_BUFFER_CAPACITY,
    get_amd_optimized_buffer_capacity,
)
from src.persistence.trade_log_reader import CachedTradeLogReader
from src.utils.metrics_calculator import period_metrics as _period_metrics_calc

LOG = logging.getLogger(__name__)

# Display threshold constants
FEASIBILITY_HIGH_THRESHOLD: float = 0.7
FEASIBILITY_MEDIUM_THRESHOLD: float = 0.5
BUFFER_HIGH_THRESHOLD: int = 1000
BUFFER_MEDIUM_THRESHOLD: int = 100
KURTOSIS_FAT_TAIL_THRESHOLD: float = KURTOSIS_ALERT_THRESHOLD
VPIN_HIGH_TOXICITY_THRESHOLD: float = 2.0
VPIN_ELEVATED_TOXICITY_THRESHOLD: float = 1.0
IMBALANCE_BUY_THRESHOLD: float = 0.3
IMBALANCE_SELL_THRESHOLD: float = -0.3
RUNWAY_SHORT_THRESHOLD: float = 0.5  # runway < 0.5 ≈ sigma > 0.02 (high vol headwind)

# Price-decimals heuristic fallback thresholds
_PRICE_REF_HIGH: float = 1000.0  # ref_price ≥ 1000 → 2 dp (BTC / Gold)
_PRICE_REF_MED: float = 10.0  # ref_price ≥ 10 → 3 dp; else 5 dp

# Z-Omega quality display bands (training/trading pipeline)
Z_OMEGA_OFFLINE_WARM_MIN: float = 0.8  # zo ≥ 0.8 → yellow in offline results
Z_OMEGA_POSITIVE_MIN: float = 0.5  # zo > 0.5 → green in trading pipeline

# Agent confidence display bands (°healthy°: 0.55 – 0.85)
CONF_HEALTHY_LOW: float = 0.55  # lower bound of healthy confidence range
CONF_HEALTHY_HIGH: float = 0.85  # upper bound of healthy confidence range
CONF_WARM_LOW: float = 0.50  # lower bound of warm confidence range

# Beta (Importance Sampling) display bands
BETA_HOT_MIN: float = 0.8  # beta > 0.8 → fully corrected (green)
BETA_WARM_MIN: float = 0.6  # beta > 0.6 → warm (yellow)

# Epsilon (exploration rate) display bands
EPS_HOT_MAX: float = 0.05  # eps < 0.05 → HOT (green)
EPS_WARM_MAX: float = 0.2  # eps < 0.2 → WARM (yellow); else COLD

# Buffer fill thresholds (fraction 0–1, used inside _pct_bar helper)
_BUF_FILL_HIGH: float = 0.5  # fraction > 0.5 → green
_BUF_FILL_WARN: float = 0.1  # fraction > 0.1 → yellow

# Buffer %-age thresholds (percent-scale 0–100, used in overview row logic)
BUF_PCT_HIGH: float = 50.0  # >50 % fill → green
BUF_PCT_WARN: float = 10.0  # >10 % fill → yellow

# Trend sparkline helper thresholds
TREND_MIN_SAMPLES: int = 6  # minimum history values before trend calc
TREND_DELTA_POS: float = 3.0  # delta_pct > +3 → DEGRADING (red)
TREND_DELTA_NEG: float = -3.0  # delta_pct < -3 → IMPROVING (green)
TREND_SPARK_TAIL: int = 20  # last N values for sparkline
TREND_STEP_PAIRS_MIN: int = 2  # history pairs needed for velocity calc
TREND_RATE_HIGH: float = 5.0  # > 5 steps/min → active training (green)
TREND_RATE_WARN: float = 1.0  # > 1 step/min → slow training (yellow)

# VPIN Z thresholds reused in overview and decision-log rows
_VPIN_OV_HIGH: float = 2.0  # same value as VPIN_HIGH_TOXICITY_THRESHOLD
_VPIN_OV_ELEVATED: float = 1.5  # elevated threshold used in overview row

# Spread / VaR / Vol colour bands — relative to mid price (basis points)
SPREAD_OK_BPS: float = 1.0  # spread < 1.0 bps → green
SPREAD_WARN_BPS: float = 3.0  # spread < 3.0 bps → yellow; else red
VAR_WARN_PCT: float = 1.5  # VaR % > 1.5 → yellow
VAR_HIGH_PCT: float = 3.0  # VaR % > 3 → red
VOL_HIGH_PCT: float = 2.0  # vol % > 2 → red
VOL_WARN_PCT: float = 1.0  # vol % > 1 → yellow
RUNWAY_WARN_BARS: float = 0.5  # runway < 0.5 → yellow (high vol headwind)
RUNWAY_OK_BARS: float = 0.7  # runway > 0.7 → green (smooth conditions)
EFF_HIGH_THRESHOLD: float = 0.6  # path efficiency > 0.6 → green
EFF_WARN_THRESHOLD: float = 0.3  # path efficiency > 0.3 → yellow
_BUDGET_OK_MIN: float = 10.0  # risk_budget > 10 USD → green

# Payoff / profit-factor colour bands
_PAYOFF_FLOOR: float = 1e-9  # zero guard: avg_win / avg_loss
PAYOFF_GOOD_MIN: float = 1.5  # payoff ≥ 1.5 → green
PROFIT_FACTOR_GOOD_MIN: float = 1.2  # PF ≥ 1.2 → green
DD_HIGH_PCT: float = 5.0  # drawdown > 5 % → red
DD_WARN_PCT: float = 2.0  # drawdown > 2 % → yellow

# _fmt_dur conversion constants
_DURATION_HOUR_MINS: float = 90.0  # < 90 min → show as minutes
_DURATION_DAY_MINS: float = 1440.0  # < 1440 min (24 h) → show as hours

# Prediction convergence colour bands
RUNWAY_DELTA_OK_MAX: float = 1.0  # |delta| < 1 pt → perfect (green)
RUNWAY_DELTA_WARN_MAX: float = 3.0  # |delta| > 3 pts → bad (red)
RUNWAY_ACCURACY_GOOD: float = 0.70  # accuracy > 0.70 → green
RUNWAY_ACCURACY_WARN: float = 0.40  # accuracy > 0.40 → yellow
# Brier score bands: 0=perfect, 0.25=no-skill (at p=0.5), 1.0=worst
CONF_CALIB_OK_MAX: float = 0.20  # Brier < 0.20 → green (better than no-skill)
CONF_CALIB_WARN_MAX: float = 0.30  # Brier < 0.30 → yellow
PLATT_ADAPTED_DELTA: float = 0.05  # |platt_a − 1.0| or |platt_b| > 0.05 → adapted
CONV_EMA_ALPHA: float = 0.1  # EMA alpha (matches bot-side production monitor)
CONV_MIN_SAMPLES: int = 10  # minimum trade samples to trust trade-log convergence

# Decision log display
_DEC_LOG_TS_MIN_LEN: int = 19  # timestamp ≥ 19 chars has full HH:MM:SS
_DEC_LOG_VPIN_WARN: float = 2.0  # |vpin_z| > 2 → flag warning icon

# Data-freshness thresholds (footer)
DATA_STALE_SECS: float = 15.0  # bot age > 15 s → "silent" warning
DATA_AGING_SECS: float = 5.0  # bot age > 5 s → "aging" warning
INPUT_POLL_SECS: float = 0.02  # UI input poll cadence (50 Hz)
INPUT_DRAIN_MAX: int = 64  # max queued key events handled per poll

# Position sizing zero-guard
_QTY_FLOOR: float = 1e-9  # guard division in qty-usage ratio

# Signal synthesis imbalance direction hint
_IMBALANCE_DIRECTION_HINT: float = 0.1  # |imbalance| > 0.1 used for directional hint

# Mouse event button codes (xterm SGR encoding)
_MOUSE_WHEEL_UP: int = 64
_MOUSE_WHEEL_DOWN: int = 65
_TAB_BAR_ROW: int = 10

# Terminal width breakpoints for label set selection
_TERM_WIDTH_FULL: int = 104
_TERM_WIDTH_MEDIUM: int = 85
_SCROLLBAR_MIN_WIDTH: int = 20

# Time unit constants
_SECS_PER_MIN: int = 60
_SECS_PER_HOUR: int = 3600
_SECS_PER_DAY: int = 86400
_MINS_PER_DAY: int = 1440

# Progress bar colour thresholds (fill fraction 0–1)
_PP_BAR_GREEN_FRAC: float = 0.5
_PP_BAR_YELLOW_FRAC: float = 0.2

# Reconnect / error count thresholds
_RECONNECT_WARN_COUNT: int = 5
_ERR_COUNT_WARN: int = 5

# Offline job: minimum validation trades before ZOmega is meaningful
_OFFLINE_MIN_VAL_TRADES: int = 5

# Adaptive tau (target network update rate) colour bands
_TAU_HIGH: float = 0.003
_TAU_LOW: float = 0.001

# Regime epsilon-factor colour bands
_REGIME_FACTOR_HEALTHY: float = 0.9
_REGIME_FACTOR_WARN: float = 0.7

# Dynamic entry floor colour bands (trigger agent)
_ENTRY_FLOOR_GOOD: float = 0.75
_ENTRY_FLOOR_WARN: float = 0.85

# Dynamic exit floor colour bands (harvester agent)
_EXIT_FLOOR_GOOD: float = 0.65
_EXIT_FLOOR_WARN: float = 0.80

# Regime hold-multiplier warning threshold (harvester)
_HOLD_MULT_WARN: float = 0.9

# Memory and error display thresholds
_MEM_HIGH_PCT: float = 80.0
_MEM_WARN_PCT: float = 60.0

# Open-position display cap (shows first N, then ellipsis)
_MAX_DISPLAY_POSITIONS: int = 8

# Bot label column width (matches :<13 format spec throughout)
_BOT_LABEL_MAX: int = 13

# Win / loss streak colour bands
_WIN_STREAK_GOOD: int = 3
_LOSS_STREAK_WARN: int = 5
_LOSS_STREAK_GOOD: int = 3

# Winner-to-loser percentage thresholds
_W2L_PCT_WARN: float = 15.0
_W2L_PCT_GOOD: float = 5.0

# Capture ratio colour bands (excursion efficiency)
_CAP_RATIO_GOOD: float = 0.60
_CAP_RATIO_WARN: float = 0.40

# Confidence calibration gap threshold (win − loss confidence)
_CAL_GAP_GOOD: float = 0.05

# Runway-utilisation warning threshold (prediction convergence)
_RUNWAY_UTIL_WARN: float = 0.7

# Prediction error percentage colour bands
_ERR_PCT_GOOD: float = 25.0
_ERR_PCT_WARN: float = 50.0

# Decision-log feasibility colour thresholds (compact display)
_DEC_FEAS_GOOD: float = 0.6
_DEC_FEAS_WARN: float = 0.3

# Reward weight visual range (adaptive weights tab)
_WEIGHT_NORMAL_LOW: float = 0.8
_WEIGHT_NORMAL_HIGH: float = 1.2
_WEIGHT_TIGHT_LOW: float = 0.5
_WEIGHT_TIGHT_HIGH: float = 1.5

# Jerk (dγ/dt) warning threshold
_JERK_WARN: float = 0.1

# Order-book depth ratio colour bands
_DEPTH_RATIO_GOOD: float = 0.8
_DEPTH_RATIO_WARN: float = 0.5

# Vol-ratio deviation threshold
_VOL_RATIO_DRIFT: float = 0.5

# Trade cache refresh interval (seconds)
_TRADE_CACHE_TTL: float = 5.0

# Trades tab display thresholds
_COMPACT_TRADES_THRESHOLD: int = 6
_TRADES_SEP_THRESHOLD: int = 8
_WIN_RATE_NEUTRAL: int = 50


def _hud_period_metrics(pts: list, starting_equity: float = 10_000.0) -> dict:
    """Compute period performance metrics from a list of trade dicts."""
    return _period_metrics_calc(pts, starting_equity=starting_equity)


def _hud_parse_dt(s: str):
    """Parse an ISO-format datetime string, returning None on failure."""
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _classify_trades_by_period(trades: list) -> tuple[list, list, list]:
    """Partition trade records into (daily, weekly, monthly) buckets by entry_time.

    Uses rolling windows so data is not lost at calendar boundaries:
      daily   = last 24 hours
      weekly  = last 7 days  (avoids 0-trade Monday morning)
      monthly = calendar month (1st of month to now)
    """
    now = datetime.now(UTC)
    cutoff_daily = now - timedelta(hours=24)
    cutoff_weekly = now - timedelta(days=7)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    daily: list = []
    weekly: list = []
    monthly: list = []
    for t in trades:
        dt = _hud_parse_dt(t.get("entry_time", ""))
        if dt is None:
            continue
        if dt >= cutoff_daily:
            daily.append(t)
        if dt >= cutoff_weekly:
            weekly.append(t)
        if dt >= month_start:
            monthly.append(t)
    return daily, weekly, monthly


def _filter_trades_by_period_single(period: str, trades: list) -> list:
    """Return trades within a single period window: 24h / 7 days / Month / Epoch / Lifetime."""
    if period == "Lifetime":
        return list(trades)
    now = datetime.now(UTC)
    if period == "24h":
        cutoff = now - timedelta(hours=24)
    elif period in ("7 days", "7d"):
        cutoff = now - timedelta(days=7)
    elif period == "Month":
        cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period == "Epoch":
        cutoff = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return list(trades)  # Epoch filtering already done at load time
    else:
        cutoff = now - timedelta(days=30)
    return [
        t for t in trades
        if _hud_parse_dt(t.get("entry_time", "")) and _hud_parse_dt(t.get("entry_time", "")) >= cutoff
    ]


# ANSI colour codes shared across HUD render methods
_ANSI_G = "\033[92m"  # green
_ANSI_Y = "\033[93m"  # yellow
_ANSI_R = "\033[91m"  # red
_ANSI_DIM = "\033[90m"  # dim
_ANSI_B = "\033[94m"  # blue
_ANSI_RST = "\033[0m"  # reset

# Regex for stripping ANSI escape sequences (SGR/CSI) — used by tests and
# any render helper that needs to know a string's *visible* width.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _fmt_compact(v: float, width: int) -> str:
    """Format a number to fit in exactly `width` visible chars with sign.

    Uses two decimal places for values < 10 000; switches to K/M/B suffix
    above that so the result never exceeds `width` chars.
    Always right-justified in the returned string.
    """
    av = abs(v)
    sign = "+" if v >= 0 else "-"
    if av >= 1e9:
        s = f"{av / 1e9:.1f}B" if av < 10e9 else f"{av / 1e9:.0f}B"
    elif av >= 1e6:
        s = f"{av / 1e6:.1f}M" if av < 10e6 else f"{av / 1e6:.0f}M"
    elif av >= 10_000:
        s = f"{av / 1e3:.0f}K"
    else:
        s = f"{av:.2f}"          # "+9999.99" = 8 chars — fits in width ≥ 8
    candidate = f"{sign}{s}"
    if len(candidate) > width:
        candidate = candidate[:width]
    return f"{candidate:>{width}}"


def _fmt_compact_pos(v: float, width: int) -> str:
    """Like _fmt_compact but treats the value as a positive magnitude (MFE/MAE)."""
    return _fmt_compact(abs(v), width)


def _fmt_count(n: int, width: int = 3) -> str:
    """Format a trade count right-justified in `width` chars; never overflows.

    Uses K suffix for 1 000–99 999, M for 100 000+. For 1 000–9 999 tries
    the one-decimal form first ("3.6K") and falls back to integer K ("3K")
    when that would exceed `width`.
    """
    if n >= 100_000:
        s = f"{n / 1e6:.1f}M" if n < 10_000_000 else f"{n / 1e6:.0f}M"
    elif n >= 10_000:
        s = f"{n // 1000}K"
    elif n >= 1_000:
        s = f"{n / 1000:.1f}K"
        if len(s) > width:
            s = f"{n // 1000}K"
    else:
        s = str(n)
    return f"{s:>{width}}"


def _ansi_cell(text: str, color: str, width: int, align: str = ">") -> str:
    """Pad outside ANSI codes so colored cells keep fixed visible width."""
    if align == "<":
        padded = f"{text:<{width}}"
    elif align == "^":
        padded = f"{text:^{width}}"
    else:
        padded = f"{text:>{width}}"
    return f"{color}{padded}{_ANSI_RST}"


def _fmt_trade_period_cell(trade_count: int, win_rate_pct: float, pnl: float, width: int = 20) -> str:
    """Format a Tab 7 period-summary cell with stable visible columns.

    Visible layout (20 chars): ` #NNN WR%% PnL$$$$ `
      1 + 1 + 3 + 1 + 4 + 1 + 1 + 7 + 1 = 20
    """
    pnl_color = _ANSI_G if pnl >= 0 else _ANSI_R
    wr_color = _ANSI_G if win_rate_pct >= 50 else _ANSI_R
    count_s = _fmt_count(trade_count, 3)
    pnl_s = _fmt_compact(pnl, 7)
    cell = f" #{count_s} {wr_color}{win_rate_pct:4.0f}%{_ANSI_RST} {pnl_color}{pnl_s}{_ANSI_RST} "
    return cell if _visible_width(cell) >= width else cell + (" " * (width - _visible_width(cell)))


def _strip_ansi(s: str) -> str:
    """Return *s* with all ANSI CSI escapes removed."""
    return _ANSI_RE.sub("", s)


# Single-codepoint glyphs that render as 2 terminal cells (wide emoji etc.).
# Anything we insert into a padded column must be counted against this set so
# that header/row/separator widths line up.  Kept intentionally small — the
# HUD does not use arbitrary emoji inside padded cells.
_WIDE_GLYPHS = frozenset("📄💰📊📈📉🤖🎯💲⚠️🔬🧠🏥🔌📐🌐🧩🔧🔥📝✅❌🔁⏱")


def _visible_width(s: str) -> int:
    """Return the terminal cell width of *s* ignoring ANSI escapes.

    Handles the small set of wide glyphs the HUD embeds in padded columns.
    For anything else we fall back to ``len`` which is correct for ASCII and
    common BMP box-drawing characters used throughout the HUD.
    """
    bare = _strip_ansi(s)
    extra = sum(1 for ch in bare if ch in _WIDE_GLYPHS)
    return len(bare) + extra


def _truncate_visible(s: str, width: int) -> str:
    """Truncate an ANSI-coloured string to a visible terminal width."""
    if width <= 0 or _visible_width(s) <= width:
        return s
    out: list[str] = []
    visible = 0
    i = 0
    truncated = False
    while i < len(s):
        if s[i] == "\x1b":
            match = _ANSI_RE.match(s, i)
            if match:
                out.append(match.group(0))
                i = match.end()
                continue
        ch = s[i]
        ch_width = 2 if ch in _WIDE_GLYPHS else 1
        if visible + ch_width > width:
            truncated = True
            break
        out.append(ch)
        visible += ch_width
        i += 1
    if truncated:
        out.append(_ANSI_RST)
    return "".join(out)


# Common data file names
_ORDER_BOOK_FILE = "order_book.json"
_BOT_CONFIG_FILE = "bot_config.json"
_HUD_PRODUCTION_METRICS_FILE = "production_metrics.json"
_HUD_HELP_PROMPT = "Press Enter to continue..."
_HUD_HELP_RETURN_PROMPT = "\nPress Enter to return to HUD..."

# Live training section layout constants
_RT_BAR_LEN: int = 26  # fill-bar character width
_RT_TRIG_CAP: int = get_amd_optimized_buffer_capacity(TRIGGER_BUFFER_CAPACITY)
_RT_HARV_CAP: int = get_amd_optimized_buffer_capacity(HARVESTER_BUFFER_CAPACITY)


class TabbedHUD:
    """Real-time tabbed HUD for trading bot monitoring."""

    TABS: ClassVar[dict[str, str]] = {
        "1": "overview",
        "2": "performance",
        "3": "training",
        "4": "risk",
        "5": "market",
        "6": "log",
        "7": "trades",
    }

    TAB_ORDER: ClassVar[list[str]] = ["overview", "performance", "training", "risk", "market", "log", "trades"]

    TAB_DISPLAY: ClassVar[dict[str, str]] = {
        "overview": "📊 Overview",
        "performance": "📈 Performance",
        "training": "🧠 Training",
        "risk": "⚠️  Risk",
        "market": "🔬 Market",
        "log": "📝 Decision Log",
        "trades": "📋 Trades",
    }

    # No-emoji variant for terminals 85–103 cols wide
    TAB_DISPLAY_MEDIUM: ClassVar[dict[str, str]] = {
        "overview": "Overview",
        "performance": "Performance",
        "training": "Training",
        "risk": "Risk",
        "market": "Market",
        "log": "Decision Log",
        "trades": "Trades",
    }

    # Abbreviated variant for terminals < 85 cols wide
    TAB_DISPLAY_SHORT: ClassVar[dict[str, str]] = {
        "overview": "Overview",
        "performance": "Perf",
        "training": "Train",
        "risk": "Risk",
        "market": "Market",
        "log": "Log",
        "trades": "Trades",
    }

    def __init__(self, refresh_rate: float = 1.0) -> None:
        self.refresh_rate = refresh_rate
        self.running = False
        self.thread = None
        self.current_tab = "overview"
        self.raw_mode_enabled = False
        self.data_dir = Path("data")
        self._init_data_state()
        self._init_trade_metric_state()
        self._init_ui_state()
        self._init_context_state()

    def _init_data_state(self) -> None:
        """Initialize source-backed HUD data containers."""
        self.position = {}
        self.metrics = {}
        self.training_stats = {}
        self.risk_stats = {}
        self.market_stats = {}
        self.bot_config = {}
        self.production_metrics = {}
        self.offline_stats: dict = {}
        self.offline_job_progress: dict = {}
        self.universe_stats: dict = {}
        self.all_bots_stats: list = []
        self.training_stats_all: list = []
        self.active_sym: str = ""
        self.active_tf_min: int = 0
        self._active_pos_file: str = ""
        self.self_test_results: list = []
        self._health_report: dict = {}
        self._metrics_from_trade_log = False

    def _init_trade_metric_state(self) -> None:
        """Initialize trade-log metric caches."""
        # Loss history for trend / sparkline (non-zero samples only)
        self._trig_loss_hist: deque = deque(maxlen=40)
        self._harv_loss_hist: deque = deque(maxlen=40)
        # Step-time pairs for training velocity: (wall_time, steps)
        self._trig_step_hist: deque = deque(maxlen=12)
        self._harv_step_hist: deque = deque(maxlen=12)
        self.last_update = None
        self.notification = ""
        self.notification_expiry = datetime.min.replace(tzinfo=UTC)
        self.profile_options = self._load_profile_options()
        self._last_frame: str = ""
        self._last_frame_key: str = ""
        self._force_redraw: bool = True
        self._body_scroll_offsets: dict[str, int] = {}
        self._body_scroll_max: int = 0
        self._tab_click_ranges: list[tuple[int, int, str]] = []
        self._performance_detail = False
        self._training_detail = False

        # Time-based metrics
        self.daily_metrics = {}
        self.weekly_metrics = {}
        self.monthly_metrics = {}
        self.lifetime_metrics = {}  # epoch-filtered lifetime view used by legacy callers
        self.all_time_metrics = {}
        self.daily_metrics_by_mode: dict[str, dict] = {}
        self.weekly_metrics_by_mode: dict[str, dict] = {}
        self.monthly_metrics_by_mode: dict[str, dict] = {}
        self.lifetime_metrics_by_mode: dict[str, dict] = {}  # epoch-filtered by mode
        self.all_time_metrics_by_mode: dict[str, dict] = {}
        self.per_symbol_metrics: dict[str, dict] = {}
        self.metrics_cube: dict[tuple[str, str, str], list[dict]] = {}
        self.metrics_cube_keys: list[tuple[str, str, str]] = []
        self.metrics_by_symbol_tf: dict[tuple[str, str], dict] = {}
        self._trade_log_metrics_trades: list[dict] = []
        self._trade_log_metrics_trades_by_mode: dict[str, list[dict]] = {}
        self._trade_log_all_trades: list[dict] = []
        self._trade_log_all_trades_by_mode: dict[str, list[dict]] = {}
        self._trade_log_unlabeled_count: int = 0
        self._trade_log_inferred_count: int = 0
        self._trade_log_unknown_timeframe_count: int = 0

    def _init_ui_state(self) -> None:
        """Initialize terminal and per-tab UI state."""
        self.heartbeat_idx = 0
        self.heartbeat_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.old_settings = None
        self._trades_page: int = 0
        self._trades_per_page: int = 22
        self._trades_cursor: int = 0
        self._trades_detail: bool = False
        self._trades_detail_trade: dict = {}
        self._all_trades: list = []  # newest-first sorted
        self._trades_view: list = []  # current filtered view for trades tab
        self._all_trades_loaded_at: float = 0.0
        self._trade_log_reader = CachedTradeLogReader(self.data_dir / "trade_log.jsonl")

        # Decision log tab state (mirrors trades pattern)
        self._dec_log_cursor: int = 0
        self._dec_log_detail: bool = False
        self._dec_log_detail_entry: dict = {}
        self._dec_log_view: list[dict] = []  # entries rendered at L3

        # Stats epoch: trades before this timestamp are excluded from metrics
        self._stats_epoch: datetime | None = None
        self._stats_epoch_excluded: int = 0  # count of excluded trades
        self._stats_epoch_excluded_pnl: float = 0.0  # PnL of excluded trades
        self._load_stats_epoch()

    def _init_context_state(self) -> None:
        """Initialize global trading-context hierarchy and legacy aliases."""
        # ── Global trading-context hierarchy ─────────────────────────────
        # One hierarchy, seven tabs as analytical lenses over the same context.
        # Level 0: Mode       (Live / Paper / Offline)
        # Level 1: Portfolio  (all instruments, all TFs)
        # Level 2: Instrument (one symbol, all TFs)
        # Level 3: Inst/TF    (one symbol, one TF)
        # Level 4: Detail     (single trade / decision card)
        self._ctx_mode: str = "paper"  # live | paper | offline
        self._ctx_level: int = 1       # start at Portfolio (skip Mode for now)
        self._ctx_symbol: str = ""     # active symbol at level ≥ 2
        self._ctx_tf: int = 0          # active TF minutes at level ≥ 3
        self._ctx_cursor: int = 0      # highlighted row index at current level
        self._ctx_period: str = "Month"  # active period for detail views
        self._ctx_periods: list[str] = ["24h", "7d", "Month", "Epoch", "Lifetime"]
        self._ctx_detail: bool = False   # detail/diagnostics pane toggle (d key)

        # ── Legacy aliases (kept for gradual migration) ─────────────────
        self._drill_level: int = 1       # mirrors _ctx_level
        self._drill_symbol: str = ""     # mirrors _ctx_symbol
        self._drill_tf_minutes: int = 0  # mirrors _ctx_tf
        self._drill_cursor: int = 0      # mirrors _ctx_cursor
        self._drill_period: str = "Month"  # mirrors _ctx_period
        self._drill_periods: list[str] = ["24h", "7d", "Month", "Epoch", "Lifetime"]

    # ── Stats epoch persistence ─────────────────────────────────────────

    def _load_stats_epoch(self) -> None:
        """Load stats_epoch from data/stats_epoch.json."""
        _path = self.data_dir / "stats_epoch.json"
        if _path.exists():
            try:
                with open(_path, encoding="utf-8") as f:
                    _data = json.load(f)
                _raw = _data.get("epoch")
                if _raw:
                    self._stats_epoch = _hud_parse_dt(_raw)
                else:
                    self._stats_epoch = None
            except Exception:
                LOG.debug("[HUD] Failed to load stats_epoch.json", exc_info=True)

    def _save_stats_epoch(self, epoch: datetime | None) -> None:
        """Atomically write stats_epoch to data/stats_epoch.json."""
        self._stats_epoch = epoch
        _path = self.data_dir / "stats_epoch.json"
        _data = {
            "epoch": epoch.isoformat() if epoch else None,
            "set_at": datetime.now(UTC).isoformat(),
        }
        _tmp_fd, _tmp_path = tempfile.mkstemp(dir=str(_path.parent), prefix=".stats_epoch_", suffix=".tmp")
        try:
            with os.fdopen(_tmp_fd, "w", encoding="utf-8") as f:
                json.dump(_data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(_tmp_path, _path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(_tmp_path)
            raise

    def _filter_trades_by_epoch(self, trades: list) -> list:
        """Return trades on or after the stats epoch. Updates excluded counters."""
        if not self._stats_epoch:
            self._stats_epoch_excluded = 0
            self._stats_epoch_excluded_pnl = 0.0
            return trades
        included: list = []
        excluded_pnl = 0.0
        for t in trades:
            dt = _hud_parse_dt(t.get("exit_time") or t.get("entry_time") or "")
            if dt and dt < self._stats_epoch:
                excluded_pnl += t.get("pnl", 0.0)
            else:
                included.append(t)
        self._stats_epoch_excluded = len(trades) - len(included)
        self._stats_epoch_excluded_pnl = excluded_pnl
        return included

    def _epoch_scope_label(self) -> str:
        """Human-readable scope for epoch-filtered closed-trade metrics."""
        if not self._stats_epoch:
            return "Lifetime"
        return f"Epoch since {self._stats_epoch.strftime('%Y-%m-%d')}"

    def _has_active_scope(self) -> bool:
        return bool(
            str(getattr(self, "active_sym", "") or "").strip() and int(getattr(self, "active_tf_min", 0) or 0) > 0
        )

    def _active_scope_label(self) -> str:
        if not self._has_active_scope():
            return "all symbols/timeframes"
        return f"{str(self.active_sym).upper()} {self._format_timeframe_minutes_label(int(self.active_tf_min))}"

    def _active_scope_trades(self, trades: list[dict]) -> list[dict]:
        """Return trades matching the active symbol/timeframe, or all trades without active scope."""
        if not self._has_active_scope():
            return list(trades)
        _sym = str(self.active_sym).upper()
        _tf = self._format_timeframe_minutes_label(int(self.active_tf_min))
        return [
            t
            for t in trades
            if self._normalize_symbol(t.get("symbol")) == _sym and self._normalize_timeframe_label(t) == _tf
        ]

    def _period_rows_for_trades(
        self, trades: list[dict], all_trades: list[dict] | None = None
    ) -> list[tuple[str, dict]]:
        """Build period metric rows for an already-scoped trade list."""
        _starting = self._universe_starting_equity()
        _daily, _weekly, _monthly = _classify_trades_by_period(trades)
        _rows = [
            ("24h", _hud_period_metrics(_daily, _starting)),
            ("7d", _hud_period_metrics(_weekly, _starting)),
            ("Month", _hud_period_metrics(_monthly, _starting)),
            ("Epoch" if self._stats_epoch else "Lifetime", _hud_period_metrics(trades, _starting)),
        ]
        if self._stats_epoch and all_trades is not None:
            _rows.append(("Lifetime", _hud_period_metrics(all_trades, _starting)))
        return _rows

    def _term_width(self) -> int:
        """Return current terminal column count (fallback 80)."""
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80

    def _term_height(self) -> int:
        """Return current terminal row count (fallback 40)."""
        try:
            return os.get_terminal_size().lines
        except OSError:
            return 40

    def _scroll_current_body(self, delta: int | None = None, *, absolute: int | None = None) -> None:
        """Move the current tab body viewport without affecting tab selection."""
        current = self._body_scroll_offsets.get(self.current_tab, 0)
        target = absolute if absolute is not None else current + int(delta or 0)
        target = max(0, min(target, max(0, self._body_scroll_max)))
        if target != current:
            self._body_scroll_offsets[self.current_tab] = target
            self._force_redraw = True

    def _activate_tab(self, tab_id: str) -> None:
        """Switch tabs and reset per-tab list/detail affordances safely."""
        if tab_id not in self.TAB_ORDER:
            return
        if self.current_tab != tab_id:
            self.current_tab = tab_id
            # Reset detail pane on tab switch; keep context hierarchy
            self._ctx_detail = False
            self._trades_detail = False
            self._trades_page = 0
            self._dec_log_detail = False
            self._force_redraw = True

    def _sync_ctx_to_legacy(self) -> None:
        """Keep legacy drill_* aliases in sync with ctx_* variables."""
        self._drill_level = self._ctx_level
        self._drill_symbol = self._ctx_symbol
        self._drill_tf_minutes = self._ctx_tf
        self._drill_cursor = self._ctx_cursor
        self._drill_period = self._ctx_period

    def _handle_mouse_event(self, seq: str) -> None:
        """Handle SGR mouse events: clicks on tabs, wheel scroll in body."""
        match = re.match(r"<?(\d+)[;,](\d+)[;,](\d+)([mM])", seq)
        if not match:
            return
        button = int(match.group(1))
        x = int(match.group(2))
        y = int(match.group(3))
        event_type = match.group(4)
        if button == 64:  # wheel up
            self._scroll_current_body(-3)
            return
        if button == 65:  # wheel down
            self._scroll_current_body(3)
            return
        if event_type != "M":
            return
        for start, end, tab_id in self._tab_click_ranges:
            if y == 10 and start <= x <= end:
                self._activate_tab(tab_id)
                return

    def _drain_csi_sequence(self, first: str = "", *, timeout: float = 0.20) -> str:
        """Read the rest of a CSI sequence so partial mouse bytes never leak as keys."""
        payload = first
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and select.select([sys.stdin.fileno()], [], [], 0.01)[0]:
            ch = self._read_raw()
            payload += ch
            if "@" <= ch <= "~":
                break
        return payload

    def _handle_x10_mouse_event(self) -> None:
        """Handle legacy X10 mouse packets: ESC [ M Cb Cx Cy."""
        raw = ""
        deadline = time.monotonic() + 0.20
        while len(raw) < 3 and time.monotonic() < deadline:
            if not select.select([sys.stdin.fileno()], [], [], 0.01)[0]:
                continue
            raw += self._read_raw()
        if len(raw) != 3:
            return
        button = max(0, ord(raw[0]) - 32)
        x = max(0, ord(raw[1]) - 32)
        y = max(0, ord(raw[2]) - 32)
        event = "M"
        if button & 3 == 3:
            event = "m"
        self._handle_mouse_event(f"{button};{x};{y}{event}")

    def start(self) -> None:
        """Start HUD."""
        self.running = True
        # Set terminal to raw mode for key input
        try:
            if sys.stdin.isatty():
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                self.raw_mode_enabled = True
        except Exception:
            pass

        # Switch to the alternate screen buffer (same technique used by less,
        # htop, vim).  On the alt buffer, \033[2J actually erases cells rather
        # than scrolling prior content into scrollback, which is the symptom
        # the user was seeing on the overview tab ("old renders accumulate").
        sys.stdout.write("\033[?1049h\033[?1000h\033[?1006h\033[?25l\033[2J\033[H")
        sys.stdout.flush()
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop HUD."""
        self.running = False
        # Restore terminal settings
        self._disable_raw_mode()
        # Leave alternate screen buffer and restore cursor visibility so the
        # user's shell scrollback is intact on exit.
        sys.stdout.write("\033[?1006l\033[?1000l\033[?25h\033[?1049l")
        sys.stdout.flush()
        if self.thread:
            self.thread.join(timeout=2)

    def _read_raw(self) -> str:
        r"""Read exactly one byte directly from the stdin fd, bypassing Python's BufferedReader.

        Using os.read() instead of sys.stdin.read(1) prevents Python's internal buffer from
        consuming multiple bytes (e.g. \x1bk) in a single OS read and hiding the second byte
        from subsequent select.select() calls on the underlying fd.
        """
        return os.read(sys.stdin.fileno(), 1).decode("latin-1")

    def _activate_relative_tab(self, step: int) -> None:
        idx = self.TAB_ORDER.index(self.current_tab)
        self._activate_tab(self.TAB_ORDER[(idx + step) % len(self.TAB_ORDER)])

    def _current_trade_page_count(self) -> int:
        page_start = self._trades_page * self._trades_per_page
        return min(self._trades_per_page, len(self._trades_view) - page_start)

    def _move_trades_cursor(self, step: int) -> None:
        page_cnt = self._current_trade_page_count()
        self._trades_cursor = max(0, min(self._trades_cursor + step, max(0, page_cnt - 1)))
        self._force_redraw = True

    def _move_dec_log_cursor(self, step: int) -> None:
        self._dec_log_cursor = max(0, min(self._dec_log_cursor + step, max(0, len(self._dec_log_view) - 1)))
        self._force_redraw = True

    def _move_ctx_cursor(self, step: int) -> None:
        self._ctx_cursor = max(0, self._ctx_cursor + step)
        self._sync_ctx_to_legacy()
        self._force_redraw = True

    def _handle_vertical_navigation(self, step: int) -> None:
        if self.current_tab == "trades" and self._ctx_level >= 3 and not self._trades_detail:
            self._move_trades_cursor(step)
            return
        if self.current_tab == "log" and self._ctx_level >= 3 and not self._dec_log_detail:
            self._move_dec_log_cursor(step)
            return
        if self._ctx_level <= 3:
            self._move_ctx_cursor(step)
            return
        self._scroll_current_body(step)

    def _handle_csi_arrow(self, seq2: str) -> bool:
        if seq2 in {"C", "D"}:
            self._activate_relative_tab(1 if seq2 == "C" else -1)
            return True
        if seq2 in {"A", "B"}:
            self._handle_vertical_navigation(-1 if seq2 == "A" else 1)
            return True
        return False

    def _handle_csi_page_key(self, seq2: str) -> bool:
        if seq2 in {"H", "F"}:
            absolute = 0 if seq2 == "H" else self._body_scroll_max
            self._scroll_current_body(absolute=absolute)
            return True
        if seq2 not in {"5", "6"}:
            return False
        if select.select([sys.stdin.fileno()], [], [], 0.02)[0] and self._read_raw() == "~":
            self._scroll_current_body(-10 if seq2 == "5" else 10)
        return True

    def _handle_csi_mouse_or_unknown(self, seq2: str) -> None:
        if seq2 == "<":
            self._handle_mouse_event("<" + self._drain_csi_sequence(timeout=0.25))
        elif seq2 == "M":
            self._handle_x10_mouse_event()
        elif seq2.isdigit() or seq2 in {";", ","}:
            self._handle_mouse_event(self._drain_csi_sequence(seq2, timeout=0.25))
        else:
            self._drain_csi_sequence(seq2, timeout=0.05)

    def _handle_csi_sequence(self) -> None:
        if not select.select([sys.stdin.fileno()], [], [], 0.05)[0]:
            return
        seq2 = self._read_raw()
        if seq2 == "Z":
            self._activate_relative_tab(-1)
            return
        if self._handle_csi_arrow(seq2) or self._handle_csi_page_key(seq2):
            return
        self._handle_csi_mouse_or_unknown(seq2)

    def _handle_escape_sequence(self, seq1: str) -> None:
        """Handle CSI / Alt-key escape sequences following the ESC byte."""
        if seq1 == "[":
            self._handle_csi_sequence()
        elif seq1.lower() == "k":  # Alt+K - emergency kill switch
            self._handle_kill_switch()

    def _page_trades_forward(self) -> None:
        if self.current_tab != "trades" or self._ctx_level < 3:
            return
        self._trades_detail = False
        max_page = max(0, (len(self._trades_view) - 1) // self._trades_per_page)
        self._trades_page = min(self._trades_page + 1, max_page)
        self._trades_cursor = 0
        self._force_redraw = True

    def _toggle_trade_detail(self) -> None:
        if not self._ctx_detail or self._ctx_level < 3:
            self._trades_detail = False
            return
        idx = self._trades_page * self._trades_per_page + self._trades_cursor
        if idx < len(self._trades_view):
            self._trades_detail_trade = self._trades_view[idx]
            self._trades_detail = True

    def _toggle_log_detail(self) -> None:
        if not self._ctx_detail or self._ctx_level < 3 or not self._dec_log_view:
            self._dec_log_detail = False
            return
        idx = min(self._dec_log_cursor, len(self._dec_log_view) - 1)
        self._dec_log_detail_entry = self._dec_log_view[idx]
        self._dec_log_detail = True

    def _toggle_detail_pane(self) -> None:
        self._ctx_detail = not self._ctx_detail
        if self.current_tab == "performance":
            self._performance_detail = self._ctx_detail
        elif self.current_tab == "training":
            self._training_detail = self._ctx_detail
        elif self.current_tab == "trades":
            self._toggle_trade_detail()
        elif self.current_tab == "log":
            self._toggle_log_detail()
        self._force_redraw = True

    def _close_detail_pane(self) -> None:
        detail_flags = {
            "trades": "_trades_detail",
            "log": "_dec_log_detail",
            "performance": "_performance_detail",
            "training": "_training_detail",
        }
        attr = detail_flags.get(self.current_tab)
        if attr and getattr(self, attr):
            setattr(self, attr, False)
            self._ctx_detail = False
            self._force_redraw = True

    def _handle_escape_key(self) -> None:
        if select.select([sys.stdin.fileno()], [], [], 0.05)[0]:
            self._handle_escape_sequence(self._read_raw())
            return
        self._drill_up()

    def _handle_command_key(self, key: str) -> bool:
        command_handlers = {
            "s": self._cycle_scope,
            "p": self._cycle_period,
            "n": self._page_trades_forward,
            "d": self._toggle_detail_pane,
            "b": self._close_detail_pane,
        }
        redraw_handlers = {
            "r": self._handle_cb_reset,
            "e": self._handle_stats_epoch,
            "h": self._show_help,
        }
        if key in command_handlers:
            command_handlers[key]()
            return True
        if key in redraw_handlers:
            redraw_handlers[key]()
            self._force_redraw = True
            return True
        return False

    def _handle_input_key(self, key: str) -> None:
        key_lower = key.lower()
        if key in self.TABS:
            self._activate_tab(self.TABS[key])
        elif key == "\t":
            self._activate_relative_tab(1)
        elif key == "\x1b":
            self._handle_escape_key()
        elif key in ("\r", "\n"):
            self._drill_down()
        elif key_lower == "q" or key in {"\x18", "\x11"}:
            self.running = False
        elif key_lower in {"j", "k"}:
            self._handle_vertical_navigation(1 if key_lower == "j" else -1)
        else:
            self._handle_command_key(key_lower)

    def _check_input(self):
        """Check for keyboard input (non-blocking).

        Drains queued bytes so rapid tab-cycling never lags behind by multiple
        refresh cycles.
        """
        handled = False
        try:
            fd = sys.stdin.fileno()
            drained = 0
            while drained < INPUT_DRAIN_MAX and select.select([fd], [], [], 0)[0]:
                drained += 1
                handled = True
                self._handle_input_key(self._read_raw())
        except Exception:
            pass
        return handled

    def _update_loop(self) -> None:
        """Main update loop."""
        _next_refresh = time.monotonic()
        while self.running:
            try:
                _input_seen = self._check_input()
                _now = time.monotonic()

                # Immediate visual response for tab/input actions; no need to
                # wait for the next data refresh tick.
                if _input_seen and self._force_redraw:
                    self._render()

                if _now >= _next_refresh:
                    self._refresh_data()
                    self._render()
                    _next_refresh = _now + self.refresh_rate

                _sleep_for = max(0.0, min(INPUT_POLL_SECS, _next_refresh - time.monotonic()))
                if _sleep_for > 0.0:
                    time.sleep(_sleep_for)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\033[0mHUD Error: {e}")
                time.sleep(2)

    def _load_performance_snapshot(self) -> None:
        """Load trading_mode from performance_snapshot.json.

        NOTE: period metrics (daily/weekly/monthly/lifetime) are NOT loaded from
        the snapshot because the bot writes session-only counters that reset on
        every restart.  _compute_metrics_from_trade_log() is the single source of
        truth for all period metrics and always runs after this call.
        """
        perf_file = self.data_dir / "performance_snapshot.json"
        if not perf_file.exists():
            return
        try:
            with open(perf_file) as f:
                data = json.load(f)
            # Only read trading_mode — metrics come from trade_log.jsonl
            self._perf_snapshot_mode = data.get("trading_mode", "")
        except Exception as e:
            if not hasattr(self, "_perf_error_shown"):
                self._set_notification(f"⚠️  Error loading performance data: {e}", ttl=10)
                self._perf_error_shown = True

    def _load_health_report(self) -> None:
        """Load self-healing analyzer report from data/performance_health.json."""
        _path = self.data_dir / "performance_health.json"
        if not _path.exists():
            return
        try:
            _payload = json.loads(_path.read_text())
            self._health_report = _payload if isinstance(_payload, dict) else {}
        except Exception:
            LOG.debug("[HUD] Failed to load performance_health.json", exc_info=True)

    def _accumulate_loss_history(self) -> None:
        """Append current training losses/steps to rolling history deques."""
        _tl = self.training_stats.get("trigger_loss", 0.0)
        _hl = self.training_stats.get("harvester_loss", 0.0)
        _now = time.time()
        if _tl > 0 and (not self._trig_loss_hist or self._trig_loss_hist[-1] != _tl):
            self._trig_loss_hist.append(_tl)
        if _hl > 0 and (not self._harv_loss_hist or self._harv_loss_hist[-1] != _hl):
            self._harv_loss_hist.append(_hl)
        _ts = self.training_stats.get("trigger_training_steps", 0)
        _hs = self.training_stats.get("harvester_training_steps", 0)
        if _ts > 0 and (not self._trig_step_hist or self._trig_step_hist[-1][1] != _ts):
            self._trig_step_hist.append((_now, _ts))
        if _hs > 0 and (not self._harv_step_hist or self._harv_step_hist[-1][1] != _hs):
            self._harv_step_hist.append((_now, _hs))

    @staticmethod
    def _resolve_trade_mode_from_record(trade: dict) -> str:
        """Resolve trade mode using explicit field, then legacy heuristics."""
        mode = str(trade.get("trading_mode", "") or "").strip().lower()
        if mode in ("paper", "live"):
            return mode

        ticket = trade.get("ticket")
        if ticket not in (None, "", "UNKNOWN"):
            return "live"

        position_id = trade.get("position_id")
        if position_id not in (None, "", "UNKNOWN"):
            return "live"

        return "paper"

    def _mixed_mode_view_filter(self) -> str:
        """Return the single mode to display when trade history contains paper and live rows."""
        if getattr(self, "_trade_log_mode", "") != "mixed":
            return ""
        _candidate = (
            str(getattr(self, "_perf_snapshot_mode", "") or self.bot_config.get("trading_mode", "") or "")
            .strip()
            .lower()
        )
        if _candidate in ("paper", "live"):
            return _candidate
        if self._trade_log_metrics_trades_by_mode.get("paper") or self._trade_log_all_trades_by_mode.get("paper"):
            return "paper"
        if self._trade_log_metrics_trades_by_mode.get("live") or self._trade_log_all_trades_by_mode.get("live"):
            return "live"
        return ""

    @staticmethod
    def _normalize_symbol(value: Any) -> str:
        _sym = str(value or "").strip().upper()
        return _sym or "UNKNOWN"

    @staticmethod
    def _normalize_timeframe_label(trade: dict) -> str:
        _tf_raw = trade.get("timeframe_minutes", 0)
        _tf = 0
        try:
            _tf = int(_tf_raw or 0)
        except (TypeError, ValueError):
            _tf = 0
        if _tf > 0:
            return f"M{_tf}"
        _label = str(trade.get("timeframe", "") or "").strip().upper()
        if _label:
            _m = re.fullmatch(r"M(\d+)", _label)
            if _m:
                return f"M{int(_m.group(1))}"
            _m = re.fullmatch(r"(\d+)M", _label)
            if _m:
                return f"M{int(_m.group(1))}"
            if _label in {"H1", "H4", "H12", "D1"}:
                _mins = {"H1": 60, "H4": 240, "H12": 720, "D1": 1440}[_label]
                return f"M{_mins}"
            return _label
        return "M?"

    @staticmethod
    def _timeframe_label_from_path(path: Path) -> str:
        """Infer a timeframe label from scoped bot/log paths."""
        for _part in [path.stem, *[p.name for p in path.parents]]:
            _m = re.search(r"(?:^|_)(M\d+|H\d+|D1|W1)(?:_|$)", str(_part).upper())
            if _m:
                return TabbedHUD._normalize_timeframe_label({"timeframe": _m.group(1)})
        return "M?"

    @staticmethod
    def _decision_entry_timeframe_label(entry: dict) -> str:
        """Return canonical timeframe label for a decision-log row."""
        _tf = TabbedHUD._normalize_timeframe_label(entry)
        if _tf != "M?":
            return _tf
        _details = entry.get("details", {})
        if isinstance(_details, dict):
            _tf = TabbedHUD._normalize_timeframe_label(_details)
            if _tf != "M?":
                return _tf
        _source_path = entry.get("_source_path")
        if _source_path:
            return TabbedHUD._timeframe_label_from_path(Path(str(_source_path)))
        return "M?"

    @staticmethod
    def _decision_entry_has_scope(entry: dict) -> bool:
        """Return True when a decision row carries explicit or inferred bot scope."""
        if TabbedHUD._decision_entry_timeframe_label(entry) != "M?":
            return True
        _sym = str(entry.get("symbol") or "").strip()
        if _sym:
            return True
        _ctx = entry.get("context", {})
        return isinstance(_ctx, dict) and bool(str(_ctx.get("symbol") or "").strip())

    def _decision_entry_symbol(self, entry: dict) -> str:
        _sym = str(entry.get("symbol") or "").strip().upper()
        if _sym:
            return _sym
        _ctx = entry.get("context", {})
        if isinstance(_ctx, dict):
            _sym = str(_ctx.get("symbol") or "").strip().upper()
            if _sym:
                return _sym
        _source_path = entry.get("_source_path")
        if _source_path:
            for _part in [Path(str(_source_path)).stem, *[p.name for p in Path(str(_source_path)).parents]]:
                _m = re.search(r"(?:^|paper_)([A-Z0-9]+)_M\d+(?:_|$)", str(_part).upper())
                if _m:
                    return _m.group(1)
        if self._has_active_scope():
            _tf = self._decision_entry_timeframe_label(entry)
            if _tf in ("M?", self._format_timeframe_minutes_label(int(self.active_tf_min))):
                return str(self.active_sym).upper()
        return ""

    def _decision_entry_matches_active_scope(self, entry: dict) -> bool:
        if not self._has_active_scope():
            return True
        _sym = self._decision_entry_symbol(entry)
        _tf = self._decision_entry_timeframe_label(entry)
        _active_tf = self._format_timeframe_minutes_label(int(self.active_tf_min))
        _sym_ok = not _sym or _sym == str(self.active_sym).upper()
        _tf_ok = _tf in ("M?", _active_tf)
        return _sym_ok and _tf_ok

    def _decision_entry_bot_label(self, entry: dict) -> str:
        _sym = self._decision_entry_symbol(entry) or "?"
        _tf = self._decision_entry_timeframe_label(entry)
        if _tf == "M?":
            return _sym
        if _sym == "?":
            return _tf
        return f"{_sym}/{_tf}"

    def _build_metrics_cube(self, trades: list[dict]) -> None:
        _cube: dict[tuple[str, str, str], list[dict]] = {}
        for _t in trades:
            _sym = self._normalize_symbol(_t.get("symbol"))
            _tf = self._normalize_timeframe_label(_t)
            _mode = self._resolve_trade_mode_from_record(_t)
            _t["trading_mode"] = _mode
            _cube.setdefault((_sym, _tf, _mode), []).append(_t)
        self.metrics_cube = _cube
        self.metrics_cube_keys = sorted(_cube.keys(), key=lambda k: (k[0], self._timeframe_sort_key(k[1]), k[2]))

    @staticmethod
    def _timeframe_sort_key(tf_label: str) -> int:
        _m = re.fullmatch(r"M(\d+)", str(tf_label).upper())
        if _m:
            return int(_m.group(1))
        return 999999

    @staticmethod
    def _format_timeframe_minutes_label(tf_minutes: Any) -> str:
        """Return canonical timeframe label for minute-based bot stats."""
        try:
            _tf = int(tf_minutes or 0)
        except (TypeError, ValueError):
            _tf = 0
        return f"M{_tf}" if _tf > 0 else "M?"

    @staticmethod
    def _entry_key(entry: dict, fallback_idx: int) -> str:
        """Stable key for entry dedup/grouping across legacy universe layouts."""
        _sym = str(entry.get("symbol") or entry.get("_symbol") or "")
        _tf = int(entry.get("timeframe_minutes", 0) or 0)
        if _sym and _tf:
            return f"{_sym}::M{_tf}"
        if _sym:
            return f"{_sym}::idx{fallback_idx}"
        return f"idx{fallback_idx}"

    @staticmethod
    def _normalize_universe_entry(item: dict, symbol: str = "") -> dict:
        entry = dict(item)
        if symbol:
            entry.setdefault("symbol", str(symbol).upper())
        elif not entry.get("symbol"):
            entry["symbol"] = str(entry.get("_symbol", "") or "")
        if entry.get("symbol"):
            entry["symbol"] = str(entry["symbol"]).upper()
        return entry

    def _iter_universe_collection(self, items: Any, symbol: str = "") -> list[dict]:
        if isinstance(items, list):
            return [self._normalize_universe_entry(item, symbol) for item in items if isinstance(item, dict)]
        if isinstance(items, dict):
            return [self._normalize_universe_entry(items, symbol)]
        return []

    def _iter_universe_mapping(self, mapping: dict, *, skip_reserved: bool = False) -> list[dict]:
        entries: list[dict] = []
        for sym, item in mapping.items():
            if skip_reserved and sym in {"version", "instruments"}:
                continue
            entries.extend(self._iter_universe_collection(item, str(sym)))
        return entries

    def _iter_universe_entries(self, uni_raw: dict) -> list[dict]:
        """Return normalized universe entries supporting old and new schemas."""
        instruments = uni_raw.get("instruments", {})
        if isinstance(instruments, list):
            return self._iter_universe_collection(instruments)
        if isinstance(instruments, dict):
            return self._iter_universe_mapping(instruments)
        return self._iter_universe_mapping(uni_raw, skip_reserved=True)

    def _universe_starting_equity(self) -> float:
        """Resolve starting equity for active bot, falling back safely."""
        if self.active_sym and self.active_tf_min:
            _k = f"{self.active_sym}::M{self.active_tf_min}"
            _e = self.universe_stats.get(_k, {})
            if isinstance(_e, dict) and _e.get("starting_equity") is not None:
                return float(_e.get("starting_equity"))

        if self.active_sym:
            for _entry in self.universe_stats.values():
                if not isinstance(_entry, dict):
                    continue
                if _entry.get("symbol") == self.active_sym and _entry.get("starting_equity") is not None:
                    return float(_entry.get("starting_equity"))

        for _entry in self.universe_stats.values():
            if isinstance(_entry, dict) and _entry.get("starting_equity") is not None:
                return float(_entry.get("starting_equity"))

        return float(self.bot_config.get("starting_equity", 10_000.0))

    def _load_bot_stats(self, symbol: str, timeframe_minutes: int) -> dict:
        _sym = str(symbol or "").upper()
        try:
            _tf = int(timeframe_minutes or 0)
        except (TypeError, ValueError):
            _tf = 0
        if not _sym or _tf <= 0:
            return {}
        _path = self.data_dir / f"paper_stats_{_sym}_M{_tf}.json"
        if not _path.exists():
            return {}
        try:
            return json.loads(_path.read_text())
        except Exception:
            LOG.debug("[HUD] Failed to load bot stats for %s M%s", _sym, _tf, exc_info=True)
            return {}

    def _load_bot_production_metrics(self, symbol: str, timeframe_minutes: int) -> dict:
        _sym = str(symbol or "").upper()
        try:
            _tf = int(timeframe_minutes or 0)
        except (TypeError, ValueError):
            _tf = 0
        if not _sym or _tf <= 0:
            return {}
        _path = self.data_dir / f"paper_{_sym}_M{_tf}" / "production_metrics.json"
        if not _path.exists():
            _path = self.data_dir / f"production_metrics_{_sym}_M{_tf}.json"
        if not _path.exists():
            return {}
        try:
            return json.loads(_path.read_text())
        except Exception:
            LOG.debug("[HUD] Failed to load production metrics for %s M%s", _sym, _tf, exc_info=True)
            return {}

    def _active_bot_stats_path(self) -> Path | None:
        _sym = str(getattr(self, "active_sym", "") or "").upper()
        try:
            _tf = int(getattr(self, "active_tf_min", 0) or 0)
        except (TypeError, ValueError):
            _tf = 0
        if not _sym or _tf <= 0:
            return None
        _path = self.data_dir / f"paper_stats_{_sym}_M{_tf}.json"
        return _path if _path.exists() else None

    def _load_bots_stats(self, symbol: str, timeframe_minutes: int) -> dict:
        return self._load_bot_stats(symbol, timeframe_minutes)

    def _parse_training_stats_identity(self, path: Path) -> tuple[str, int]:
        _stem = path.stem.removeprefix("training_stats_")
        if "_M" not in _stem:
            return "", 0
        _sym_part, _, _tf_str = _stem.rpartition("_M")
        try:
            _tf = int(_tf_str)
        except ValueError:
            return "", 0
        return _sym_part.upper(), _tf

    def _load_all_training_stats(self) -> None:
        _by_key: dict[tuple[str, int, str], dict] = {}
        _universe_keys: set[tuple[str, int]] = set()
        if self.universe_stats:
            for _entry in self.universe_stats.values():
                if not isinstance(_entry, dict):
                    continue
                _sym = str(_entry.get("symbol", "") or "").upper()
                try:
                    _tf = int(_entry.get("timeframe_minutes", 0) or 0)
                except (TypeError, ValueError):
                    _tf = 0
                if _sym and _tf > 0:
                    _universe_keys.add((_sym, _tf))

        _candidates = sorted(self.data_dir.glob("training_stats_*_M*.json"))
        for _path in _candidates:
            try:
                _payload = json.loads(_path.read_text())
            except Exception:
                LOG.debug("[HUD] Failed to load %s", _path, exc_info=True)
                continue
            if not isinstance(_payload, dict):
                continue
            _sym, _tf = self._parse_training_stats_identity(_path)
            if not _sym or _tf <= 0:
                continue
            if _universe_keys and (_sym, _tf) not in _universe_keys:
                continue
            _mode = str(_payload.get("trading_mode", "paper") or "paper").strip().lower()
            if _mode not in ("paper", "live"):
                _mode = "paper"
            _k = (_sym, _tf, _mode)
            _mtime = _path.stat().st_mtime if _path.exists() else 0
            if _k in _by_key and _by_key[_k].get("mtime", 0) >= _mtime:
                continue
            _by_key[_k] = {
                "symbol": _sym,
                "timeframe_minutes": _tf,
                "trading_mode": _mode,
                "stats": _payload,
                "path": _path,
                "mtime": _mtime,
            }
        self.training_stats_all = sorted(
            _by_key.values(),
            key=lambda _item: (
                int(_item.get("timeframe_minutes", 0) or 0),
                str(_item.get("symbol", "")),
                str(_item.get("trading_mode", "")),
            ),
        )

    def _enrich_offline_stats_from_champions(self) -> None:
        """Back-fill missing metadata in 'done' status entries from offline_champions.json.

        The misplaced break bug in train_offline._execute_pool caused z_omega,
        val_trades, train_trades, total_train_steps, accepted, and accept_reason
        to never be written to offline_training_status.json for any completed job
        in a running process. This enriches those entries from the champions registry
        so the HUD shows correct data without waiting for a new training run.
        """
        results = self.offline_stats.get("results")
        if not isinstance(results, list):
            return
        # Load champions registry
        _champ_path = self.data_dir / "checkpoints" / "offline_champions.json"
        if not _champ_path.exists():
            return
        try:
            _raw = json.loads(_champ_path.read_text())
            champions: dict = _raw.get("champions", {})
        except Exception:
            return
        for entry in results:
            if not isinstance(entry, dict) or entry.get("status") != "done":
                continue
            if entry.get("z_omega") is not None:
                continue
            sym = str(entry.get("symbol", "")).upper()
            tf = int(entry.get("timeframe_minutes", 0) or 0)
            if not sym or not tf:
                continue
            key = f"{sym}_M{tf}"
            champ = champions.get(key)
            if not isinstance(champ, dict):
                continue
            zo = champ.get("z_omega")
            if zo is None:
                continue
            entry["z_omega"] = float(zo)
            entry["val_trades"] = int(champ.get("val_trades", 0) or 0)
            entry.setdefault("train_trades", 0)
            entry.setdefault("total_train_steps", 0)
            entry.setdefault("accepted", True)
            entry.setdefault("accept_reason", "champion_registry")

    def _load_universe_stats(self) -> None:
        """Load universe.json, annotate liveness, auto-prune dead entries."""
        _uni_path = self.data_dir / "universe.json"
        if not _uni_path.exists():
            self.universe_stats = {}
            return
        try:
            _uni_raw: dict = json.loads(_uni_path.read_text())
            _entries = self._iter_universe_entries(_uni_raw)
            _dead_keys: list[str] = []
            _normalized: dict[str, dict] = {}
            for idx, _entry in enumerate(_entries):
                if not isinstance(_entry, dict):
                    continue
                _sym = str(_entry.get("symbol", "") or "").upper()
                if not _sym:
                    continue
                _entry["symbol"] = _sym
                _pid = _entry.get("paper_pid")
                _alive = False
                if _pid:
                    try:
                        os.kill(int(_pid), 0)
                        _r = subprocess.run(
                            ["ps", "-p", str(_pid), "-o", "stat", "--no-headers"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        _alive = bool(_r.stdout.strip()) and "Z" not in _r.stdout
                    except OSError:
                        pass
                _entry["_pid_alive"] = _alive
                _k = self._entry_key(_entry, idx)
                _entry["_entry_key"] = _k
                if _pid and not _alive:
                    _dead_keys.append(_k)
                    continue
                _tf = _entry.get("timeframe_minutes", 0)
                _entry["_bot_stats"] = self._load_bot_stats(_sym, _tf) if _tf else {}
                _normalized[_k] = _entry

            if _dead_keys:
                LOG.debug("[HUD] Ignoring %d dead universe entries in-memory", len(_dead_keys))
            self.universe_stats = _normalized
        except Exception:
            LOG.debug("[HUD] Failed to load universe.json", exc_info=True)

    def _load_risk_and_orderbook(self) -> None:
        """Load risk_metrics.json then overwrite with fresher order_book.json fields."""
        risk_file = self._preferred_data_file("risk_metrics.json")
        if risk_file.exists():
            try:
                with open(risk_file) as f:
                    data = json.load(f)
                self.risk_stats = data
                self.market_stats = {
                    "vpin": data.get("vpin", 0.0),
                    "vpin_z": data.get("vpin_zscore", 0.0),
                    "spread": data.get("spread", 0.0),
                    "imbalance": data.get("imbalance", 0.0),
                    "depth_bid": data.get("depth_bid", 0.0),
                    "depth_ask": data.get("depth_ask", 0.0),
                    "order_book_bids": data.get("order_book_bids", []),
                    "order_book_asks": data.get("order_book_asks", []),
                    "rs_vol_short": data.get("rs_vol_short", 0.0),
                    "rs_vol_long": data.get("rs_vol_long", 0.0),
                    "rs_vol_ratio": data.get("rs_vol_ratio", 1.0),
                    "kurtosis": data.get("kurtosis", 0.0),
                    "kurtosis_threshold": data.get("kurtosis_threshold", 0.0),
                    "kurtosis_gate_active": data.get("kurtosis_gate_active", False),
                    "depth_gate_active": data.get("depth_gate_active", False),
                    "depth_floor": data.get("depth_floor", 0.0),
                    "has_real_sizes": data.get("has_real_sizes", False),
                }
            except Exception as e:
                if not hasattr(self, "_risk_error_shown"):
                    self._set_notification(f"⚠️  Error loading risk metrics: {e}", ttl=10)
                    self._risk_error_shown = True

        self._apply_orderbook_overlay(debug_label="risk/orderbook data")

    def _load_account_balance_fallback(self) -> None:
        if self.bot_config.get("real_account_balance"):
            return
        bal_file = self.data_dir / "account_balance.json"
        if not bal_file.exists():
            return
        try:
            with open(bal_file, encoding="utf-8") as handle:
                bal_data = json.load(handle)
            bal_val = bal_data.get("balance")
            if bal_val is not None:
                self.bot_config["real_account_balance"] = float(bal_val)
        except Exception:
            LOG.debug("[HUD] Failed to load account_balance.json", exc_info=True)

    def _load_active_position(self) -> None:
        pos_files = sorted(
            [str(p) for p in self.data_dir.glob("current_position_*.json")],
            key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0,
            reverse=True,
        )
        self.position = {}
        self._active_pos_file = ""
        for pos_file in pos_files:
            try:
                with open(pos_file, encoding="utf-8") as handle:
                    pos_data = json.load(handle)
                if pos_data.get("direction", "FLAT") != "FLAT":
                    self.position = pos_data
                    self._active_pos_file = pos_file
                    break
            except Exception:
                LOG.debug("[HUD] Failed to load position file %s", pos_file, exc_info=True)
        if not self.position:
            self._load_json("current_position.json", "position")

    def _set_active_scope_from_position(self) -> None:
        self.active_sym = self.position.get("symbol", "")
        self.active_tf_min = int(self.position.get("timeframe_minutes", 0) or 0)
        if not self.active_sym and self._active_pos_file:
            self._set_active_scope_from_position_filename()
        if not self.active_sym or not self.active_tf_min:
            self._set_active_scope_from_config()

    def _set_active_scope_from_position_filename(self) -> None:
        stem = Path(self._active_pos_file).stem
        tail = stem.removeprefix("current_position_")
        if "_M" not in tail:
            return
        sym_part, _, tf_str = tail.rpartition("_M")
        self.active_sym = sym_part
        with contextlib.suppress(ValueError):
            self.active_tf_min = int(tf_str)

    def _set_active_scope_from_config(self) -> None:
        cfg_sym = str(self.bot_config.get("symbol", "") or "").upper()
        try:
            cfg_tf = int(self.bot_config.get("timeframe_minutes", 0) or 0)
        except (TypeError, ValueError):
            cfg_tf = 0
        if cfg_sym and cfg_tf > 0:
            self.active_sym = cfg_sym
            self.active_tf_min = cfg_tf

    def _load_offline_job_progress(self) -> None:
        progress: dict = {}
        for progress_file in self.data_dir.glob("offline_progress_*.json"):
            try:
                data = json.loads(progress_file.read_text())
                progress[(data["symbol"], data["timeframe_minutes"])] = data
            except Exception:
                LOG.debug("[HUD] Failed to load offline progress %s", progress_file, exc_info=True)
        self.offline_job_progress = progress

    def _load_active_scoped_stats(self) -> None:
        if self.active_sym and self.active_tf_min:
            sym, tf = self.active_sym, self.active_tf_min
            per_train = f"training_stats_{sym}_M{tf}.json"
            if (self.data_dir / per_train).exists():
                self._load_json(per_train, "training_stats")
                self._accumulate_loss_history()
            per_risk = f"risk_metrics_{sym}_M{tf}.json"
            if (self.data_dir / per_risk).exists():
                self._load_json(per_risk, "risk_stats")
                self._apply_risk_stats_to_market_stats()
            return
        self._fallback_to_freshest_scoped_stats()

    def _fallback_to_freshest_scoped_stats(self) -> None:
        if self.training_stats_all:
            active_item = max(self.training_stats_all, key=lambda item: item.get("mtime", 0))
            self.training_stats = active_item.get("stats", {})
            self._accumulate_loss_history()
            self.active_sym = active_item.get("symbol", "")
            self.active_tf_min = int(active_item.get("timeframe_minutes", 0) or 0)
        risk_files = sorted(
            self.data_dir.glob("risk_metrics_*_M*.json"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        risk_file = self._matched_or_freshest_risk_file(risk_files)
        if risk_file:
            self._load_json(risk_file.name, "risk_stats")
            self._apply_risk_stats_to_market_stats()

    def _matched_or_freshest_risk_file(self, risk_files: list[Path]) -> Path | None:
        if self.active_sym and self.active_tf_min:
            matched = self.data_dir / f"risk_metrics_{self.active_sym}_M{self.active_tf_min}.json"
            if matched.exists():
                return matched
        return risk_files[0] if risk_files else None

    def _reload_scoped_production_metrics(self) -> None:
        if not (self.active_sym and self.active_tf_min):
            return
        self._load_json(_BOT_CONFIG_FILE, "bot_config")
        self._load_json("production_metrics.json", "production_metrics")

    def _load_self_test_results(self) -> None:
        st_file = self.data_dir / "self_test.json"
        if not st_file.exists():
            return
        try:
            with open(st_file) as handle:
                self.self_test_results = json.load(handle).get("results", [])
        except Exception:
            LOG.debug("[HUD] Failed to load self_test.json", exc_info=True)

    def _apply_orderbook_overlay(self, *, debug_label: str = "order_book.json overlay") -> None:
        ob_file = self._preferred_data_file(_ORDER_BOOK_FILE)
        if not ob_file.exists():
            return
        try:
            with open(ob_file) as handle:
                ob = json.load(handle)
            ms = self.market_stats
            ms["spread"] = ob.get("spread", ms.get("spread", 0.0))
            ms["depth_bid"] = ob.get("depth_bid", ms.get("depth_bid", 0.0))
            ms["depth_ask"] = ob.get("depth_ask", ms.get("depth_ask", 0.0))
            ms["vpin"] = ob.get("vpin", ms.get("vpin", 0.0))
            ms["vpin_z"] = ob.get("vpin_zscore", ms.get("vpin_z", 0.0))
            ob_imb = ob.get("imbalance")
            if ob_imb is not None:
                ms["imbalance"] = float(ob_imb)
            ms["order_book_bids"] = ob.get("order_book_bids", ms.get("order_book_bids", []))
            ms["order_book_asks"] = ob.get("order_book_asks", ms.get("order_book_asks", []))
            ms["has_real_sizes"] = ob.get("has_real_sizes", False)
            ms["qfi_update_count"] = ob.get("qfi_update_count", 0)
            ms["next_bar_close_utc"] = ob.get("next_bar_close_utc")
            ms["timeframe_minutes"] = ob.get("timeframe_minutes")
        except Exception:
            LOG.debug("[HUD] Failed to load %s", debug_label, exc_info=True)

    def _load_all_bots_panel_stats(self) -> None:
        all_bots_by_key: dict[tuple[str, int], dict] = {}
        for paper_stats_file in sorted(self.data_dir.glob("paper_stats_*.json")):
            try:
                paper_stats = json.loads(paper_stats_file.read_text())
                sym = str(paper_stats.get("symbol", "") or "").upper()
                tf = int(paper_stats.get("timeframe_minutes", 0) or 0)
                if not sym or tf <= 0:
                    continue
                paper_stats["trading_mode"] = self._resolve_paper_stats_mode(paper_stats)
                pos_file = self.data_dir / f"current_position_{sym}_M{tf}.json"
                paper_stats["_position"] = json.loads(pos_file.read_text()) if pos_file.exists() else {}
                all_bots_by_key[(sym, tf)] = paper_stats
            except Exception:
                LOG.debug("[HUD] Failed to load paper_stats %s", paper_stats_file, exc_info=True)
        self.all_bots_stats = list(all_bots_by_key.values())

    @staticmethod
    def _resolve_paper_stats_mode(paper_stats: dict) -> str:
        mode = str(paper_stats.get("trading_mode", "") or "").strip().lower()
        if mode in ("paper", "live"):
            return mode
        return "paper" if bool(paper_stats.get("paper_mode", True)) else "live"

    def _refresh_data(self) -> None:
        """Refresh all data from bot exports."""
        self.last_update = datetime.now(UTC)
        self.heartbeat_idx = (self.heartbeat_idx + 1) % len(self.heartbeat_chars)

        if not self.data_dir.exists():
            self._set_notification(f"⚠️  Data directory not found: {self.data_dir}", ttl=30)
            return

        self._load_json(_BOT_CONFIG_FILE, "bot_config")
        self._load_account_balance_fallback()
        self._load_active_position()
        self._set_active_scope_from_position()
        self._load_performance_snapshot()
        self._load_health_report()
        self._load_json("training_stats.json", "training_stats")
        self._accumulate_loss_history()
        self._load_json("production_metrics.json", "production_metrics")
        self._load_json("offline_training_status.json", "offline_stats")
        self._enrich_offline_stats_from_champions()
        self._load_universe_stats()
        self._load_all_training_stats()
        self._load_offline_job_progress()
        self._compute_metrics_from_trade_log()
        self._metrics_from_trade_log = bool(self.lifetime_metrics.get("total_trades"))
        self._load_risk_and_orderbook()
        self._load_active_scoped_stats()
        self._reload_scoped_production_metrics()
        self._load_self_test_results()
        self._apply_orderbook_overlay()
        self._load_all_bots_panel_stats()
        self._load_all_trades_cached()

    def _load_profile_options(self):
        """Load preset symbol/timeframe profiles for selection UI."""
        presets_path = Path("config/profile_presets.json")
        if not presets_path.exists():
            return []
        try:
            with open(presets_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            self._set_notification(f"Failed to load profile presets: {exc}", ttl=6)
            return []

        profiles = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                required = {"symbol", "symbol_id", "timeframe_minutes", "qty"}
                if not required.issubset(item.keys()):
                    continue
                label = item.get("label") or f"{item['symbol']} M{item['timeframe_minutes']}"
                profiles.append(
                    {
                        "label": label,
                        "symbol": item["symbol"],
                        "symbol_id": item["symbol_id"],
                        "timeframe_minutes": item["timeframe_minutes"],
                        "qty": item["qty"],
                    }
                )
        return profiles

    def _reset_trade_log_metrics(self) -> None:
        self.daily_metrics = {}
        self.weekly_metrics = {}
        self.monthly_metrics = {}
        self.lifetime_metrics = {}
        self.all_time_metrics = {}
        self.daily_metrics_by_mode = {}
        self.weekly_metrics_by_mode = {}
        self.monthly_metrics_by_mode = {}
        self.lifetime_metrics_by_mode = {}
        self.all_time_metrics_by_mode = {}
        self.per_symbol_metrics = {}
        self.metrics_cube = {}
        self.metrics_cube_keys = []
        self.metrics_by_symbol_tf = {}
        self._trade_log_metrics_trades = []
        self._trade_log_metrics_trades_by_mode = {}
        self._trade_log_all_trades = []
        self._trade_log_all_trades_by_mode = {}

    @staticmethod
    def _log_trade_data_quality(trades: list[dict]) -> None:
        null_entry_time = sum(1 for trade in trades if trade.get("entry_time") is None)
        missing_quantity = sum(1 for trade in trades if "quantity" not in trade or trade.get("quantity") is None)
        recalc_trades = sum(1 for trade in trades if trade.get("pnl_recalculated"))
        if null_entry_time > 0:
            LOG.debug(
                "[DATA-QUALITY] %d/%d trades have NULL entry_time (will be excluded from duration calc)",
                null_entry_time,
                len(trades),
            )
        if missing_quantity > 0:
            LOG.debug(
                "[DATA-QUALITY] %d/%d trades missing 'quantity' field (HUD cannot display position sizing)",
                missing_quantity,
                len(trades),
            )
        if recalc_trades <= 0:
            return
        original_pnl = sum(trade.get("pnl_original", 0) for trade in trades if "pnl_original" in trade)
        current_pnl = sum(trade.get("pnl", 0) for trade in trades)
        variance = abs(current_pnl - original_pnl)
        LOG.debug(
            "[DATA-QUALITY] %d/%d trades recalculated. Original PnL: $%.2f, Current: $%.2f, Variance: $%.2f",
            recalc_trades,
            len(trades),
            original_pnl,
            current_pnl,
            variance,
        )

    def _resolve_trade_log_modes(self, trades: list[dict]) -> None:
        modes: set[str] = set()
        unlabeled = 0
        inferred = 0
        for trade in trades:
            raw_mode = str(trade.get("trading_mode", "") or "").strip().lower()
            if raw_mode in ("paper", "live"):
                resolved = raw_mode
            else:
                unlabeled += 1
                inferred += 1
                resolved = self._resolve_trade_mode_from_record(trade)
                trade["trading_mode"] = resolved
            modes.add(resolved)
        self._trade_log_unlabeled_count = unlabeled
        self._trade_log_inferred_count = inferred
        self._trade_log_unknown_timeframe_count = sum(
            1 for trade in trades if self._normalize_timeframe_label(trade) == "M?"
        )
        self._trade_log_mode = next(iter(modes)) if len(modes) == 1 else ("mixed" if modes else "")

    def _set_all_time_trade_metrics(self, trades: list[dict], starting_equity: float) -> None:
        self.all_time_metrics = _hud_period_metrics(trades, starting_equity)
        self.all_time_metrics_by_mode = {
            mode: _hud_period_metrics([trade for trade in trades if trade.get("trading_mode") == mode], starting_equity)
            for mode in ("paper", "live")
        }
        self._trade_log_all_trades = list(trades)
        self._trade_log_all_trades_by_mode = {
            mode: [trade for trade in trades if trade.get("trading_mode") == mode] for mode in ("paper", "live")
        }

    @staticmethod
    def _period_start_equity(all_trades: list[dict], period_trades: list[dict], starting_equity: float) -> float:
        period_ids = set(map(id, period_trades))
        return starting_equity + sum(trade.get("pnl", 0) for trade in all_trades if id(trade) not in period_ids)

    def _set_epoch_trade_sets(self, trades: list[dict]) -> dict[str, list[dict]]:
        self._trade_log_metrics_trades = list(trades)
        self._build_metrics_cube(trades)
        trades_by_mode = {
            "paper": [trade for trade in trades if trade.get("trading_mode") == "paper"],
            "live": [trade for trade in trades if trade.get("trading_mode") == "live"],
        }
        self._trade_log_metrics_trades_by_mode = {key: list(value) for key, value in trades_by_mode.items()}
        return trades_by_mode

    def _set_period_trade_metrics(self, trades: list[dict], starting_equity: float) -> None:
        daily, weekly, monthly = _classify_trades_by_period(trades)
        pre_daily = self._period_start_equity(trades, daily, starting_equity)
        pre_weekly = self._period_start_equity(trades, weekly, starting_equity)
        pre_monthly = self._period_start_equity(trades, monthly, starting_equity)
        self.daily_metrics = _hud_period_metrics(daily, pre_daily)
        self.weekly_metrics = _hud_period_metrics(weekly, pre_weekly)
        self.monthly_metrics = _hud_period_metrics(monthly, pre_monthly)
        self.lifetime_metrics = _hud_period_metrics(trades, starting_equity)

    def _set_mode_period_metrics(self, trades_by_mode: dict[str, list[dict]], starting_equity: float) -> None:
        self.daily_metrics_by_mode = {}
        self.weekly_metrics_by_mode = {}
        self.monthly_metrics_by_mode = {}
        self.lifetime_metrics_by_mode = {}
        for mode_name, mode_trades in trades_by_mode.items():
            daily, weekly, monthly = _classify_trades_by_period(mode_trades)
            self.daily_metrics_by_mode[mode_name] = _hud_period_metrics(
                daily, self._period_start_equity(mode_trades, daily, starting_equity)
            )
            self.weekly_metrics_by_mode[mode_name] = _hud_period_metrics(
                weekly, self._period_start_equity(mode_trades, weekly, starting_equity)
            )
            self.monthly_metrics_by_mode[mode_name] = _hud_period_metrics(
                monthly, self._period_start_equity(mode_trades, monthly, starting_equity)
            )
            self.lifetime_metrics_by_mode[mode_name] = _hud_period_metrics(mode_trades, starting_equity)

    def _augment_lifetime_timing_metrics(self, trades: list[dict]) -> None:
        durations: list[float] = []
        trades_with_complete_times = 0
        last_exit_dt = None
        for trade in trades:
            entry_dt = _hud_parse_dt(trade.get("entry_time", ""))
            exit_dt = _hud_parse_dt(trade.get("exit_time", ""))
            if entry_dt and exit_dt:
                durations.append((exit_dt - entry_dt).total_seconds() / 60.0)
                trades_with_complete_times += 1
            if exit_dt and (last_exit_dt is None or exit_dt > last_exit_dt):
                last_exit_dt = exit_dt
        now = datetime.now(UTC)
        self.lifetime_metrics["avg_trade_duration_mins"] = sum(durations) / len(durations) if durations else 0.0
        self.lifetime_metrics["last_trade_mins_ago"] = (
            (now - last_exit_dt).total_seconds() / 60.0 if last_exit_dt else 0.0
        )
        self.lifetime_metrics["_data_quality_trades_with_complete_times"] = trades_with_complete_times
        self.lifetime_metrics["_data_quality_total_trades"] = len(trades)

    def _set_trade_symbol_breakdowns(self, trades: list[dict], starting_equity: float) -> None:
        by_symbol: dict[str, list] = {}
        for trade in trades:
            by_symbol.setdefault(trade.get("symbol", "UNKNOWN"), []).append(trade)
        self.per_symbol_metrics = {
            symbol: _hud_period_metrics(symbol_trades, starting_equity)
            for symbol, symbol_trades in by_symbol.items()
        }
        by_symbol_tf: dict[tuple[str, str], list[dict]] = {}
        for (sym, tf, _mode), cube_trades in self.metrics_cube.items():
            by_symbol_tf.setdefault((sym, tf), []).extend(cube_trades)
        self.metrics_by_symbol_tf = {
            key: _hud_period_metrics(value, starting_equity) for key, value in by_symbol_tf.items()
        }

    def _compute_metrics_from_trade_log(self) -> None:
        """Compute performance metrics directly from trade_log.jsonl.

        Always re-classifies trades by the rolling time windows (daily/weekly/
        monthly) because those windows advance with wall-clock time even when
        the file itself hasn't changed.  File is only re-parsed when mtime
        changes, avoiding redundant I/O on every 1 Hz refresh cycle.
        """
        trades = self._trade_log_reader.trades
        if not trades:
            self._reset_trade_log_metrics()
            return

        starting_equity = float(self._universe_starting_equity())
        self._log_trade_data_quality(trades)
        self._resolve_trade_log_modes(trades)
        self._set_all_time_trade_metrics(trades, starting_equity)
        trades = self._filter_trades_by_epoch(trades)
        trades_by_mode = self._set_epoch_trade_sets(trades)
        self._set_period_trade_metrics(trades, starting_equity)
        self._set_mode_period_metrics(trades_by_mode, starting_equity)
        self._augment_lifetime_timing_metrics(trades)
        self._set_trade_symbol_breakdowns(trades, starting_equity)

    def _price_decimals(self, ref_price: float = 0.0) -> int:
        """Return the correct number of decimal places for the active symbol.

        Reads ``digits`` from config/symbol_specs.json for the symbol stored in
        bot_config.  Falls back to a heuristic based on price magnitude so the
        HUD works even without a specs file.
        """
        symbol = self.bot_config.get("symbol", "")
        try:
            specs_path = Path("config") / "symbol_specs.json"
            specs = json.loads(specs_path.read_text(encoding="utf-8"))
            if symbol in specs:
                return int(specs[symbol].get("digits", 5))
        except Exception:
            pass
        # Heuristic fallback: gold/BTC ≥1000 → 2dp; majors → 5dp
        if ref_price >= _PRICE_REF_HIGH:
            return 2
        if ref_price >= _PRICE_REF_MED:
            return 3
        return 5

    def _preferred_data_file(self, filename: str) -> Path:
        """Return the freshest path for *filename* across top-level data_dir and
        any per-bot ``paper_*_M*`` subdirectory.

        Universe-managed paper bots write their own copy of files like
        ``order_book.json`` and ``production_metrics.json`` into
        ``data/paper_{SYM}_M{TF}/``.  When an active symbol/timeframe is known,
        prefer that exact scope before any freshness fallback so one timeframe
        cannot overwrite another timeframe's Overview/System Health values.
        """
        active_sym = str(getattr(self, "active_sym", "") or "").upper()
        active_tf = int(getattr(self, "active_tf_min", 0) or 0)
        if active_sym and active_tf > 0:
            base = Path(filename)
            scoped = self.data_dir / f"{base.stem}_{active_sym}_M{active_tf}{base.suffix}"
            if scoped.exists():
                return scoped
            active_dir_file = self.data_dir / f"paper_{active_sym}_M{active_tf}" / filename
            if active_dir_file.exists():
                return active_dir_file

        top = self.data_dir / filename
        best = top if top.exists() else None
        best_mtime = best.stat().st_mtime if best else -1.0
        try:
            for bot_dir in self.data_dir.glob("paper_*_M*"):
                if not bot_dir.is_dir():
                    continue
                cand = bot_dir / filename
                if not cand.exists():
                    continue
                try:
                    m = cand.stat().st_mtime
                except OSError:
                    continue
                if m > best_mtime:
                    best_mtime = m
                    best = cand
        except OSError:
            pass
        return best if best is not None else top

    def _load_json(self, filename: str, attr: str) -> None:
        """Load JSON file into attribute."""
        filepath = self._preferred_data_file(filename)
        if filepath.exists():
            try:
                with open(filepath) as f:
                    setattr(self, attr, json.load(f))
            except Exception:
                pass

    def _apply_risk_stats_to_market_stats(self) -> None:
        """Merge risk_stats fields into market_stats, preserving existing values as fallback."""
        _rm = self.risk_stats
        self.market_stats.update(
            {
                "vpin": _rm.get("vpin", self.market_stats.get("vpin", 0.0)),
                "vpin_z": _rm.get("vpin_zscore", self.market_stats.get("vpin_z", 0.0)),
                "spread": _rm.get("spread", self.market_stats.get("spread", 0.0)),
                "imbalance": _rm.get("imbalance", self.market_stats.get("imbalance", 0.0)),
                "depth_bid": _rm.get("depth_bid", self.market_stats.get("depth_bid", 0.0)),
                "depth_ask": _rm.get("depth_ask", self.market_stats.get("depth_ask", 0.0)),
                "order_book_bids": _rm.get("order_book_bids", self.market_stats.get("order_book_bids", [])),
                "order_book_asks": _rm.get("order_book_asks", self.market_stats.get("order_book_asks", [])),
                "kurtosis": _rm.get("kurtosis", self.market_stats.get("kurtosis", 0.0)),
                "kurtosis_threshold": _rm.get(
                    "kurtosis_threshold", self.market_stats.get("kurtosis_threshold", KURTOSIS_FAT_TAIL_THRESHOLD)
                ),
                "kurtosis_gate_active": _rm.get(
                    "kurtosis_gate_active", self.market_stats.get("kurtosis_gate_active", False)
                ),
            }
        )

    def _disable_raw_mode(self) -> None:
        """Return terminal to original mode for blocking input prompts."""
        if self.raw_mode_enabled and self.old_settings and sys.stdin.isatty():
            with contextlib.suppress(BaseException):
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            self.raw_mode_enabled = False

    def _enable_raw_mode(self) -> None:
        """Re-enter raw mode after prompt interactions."""
        if (not self.raw_mode_enabled) and self.old_settings and sys.stdin.isatty():
            try:
                tty.setcbreak(sys.stdin.fileno())
                self.raw_mode_enabled = True
            except Exception:
                pass
        # Always hide cursor when re-entering HUD raw mode — menus show it.
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    def _menu_enter(self) -> None:
        """Clear screen and show cursor before an interactive menu prompt.

        Using direct ANSI sequences instead of os.system("clear") avoids the
        subprocess-stdout race and ensures the sequences operate on the active
        screen buffer (alternate or main) without spawning a shell that may
        emit smcup/rmcup and switch buffers unexpectedly.
        """
        sys.stdout.write("\033[2J\033[H\033[?25h")
        sys.stdout.flush()

    @staticmethod
    def _print_help_section(title: str, rows: list[str], *, gap_before: bool = True) -> None:
        prefix = "\n" if gap_before else ""
        print(f"{prefix}\033[1m{title}\033[0m\n")
        for row in rows:
            print(row)

    def _help_sections(self) -> list[tuple[str, list[str]]]:
        return [
            (
                "📋 KEYBOARD SHORTCUTS",
                [
                    "  [1]           - Overview tab (compact summary)",
                    "  [2]           - Performance tab (detailed metrics)",
                    "  [3]           - Training tab (agent statistics)",
                    "  [4]           - Risk tab (risk management)",
                    "  [5]           - Market tab (microstructure)",
                    "  [6]           - Decision Log tab (last 20 decisions)",
                    "  [7]           - Trade History tab (all closed trades with drill-down)",
                    "  [←] / [→]     - Cycle tabs without reaching for Tab",
                    "  [Tab]         - Cycle to next tab",
                    "  [Shift+Tab]   - Cycle to previous tab",
                    "  [s]           - Select symbol/timeframe preset",
                    "  [h]           - Show this help screen",
                    "  [q] / Ctrl+Q / Ctrl+X  - Quit HUD",
                    "  [Alt+K]       - Emergency kill switch (close all positions + halt trading)",
                    "  [r]           - Review tripped circuit breakers and reset if OK",
                    "  [e]           - Set/clear stats epoch (exclude old trades from metrics)",
                ],
            ),
            (
                "📈 PERFORMANCE TAB KEYS",
                [
                    "  [d]           - Toggle detailed quality / prediction drill-down",
                    "  [b]           - Back to summary view",
                ],
            ),
            (
                "🧠 TRAINING TAB KEYS",
                ["  [d]           - Toggle full per-agent training detail", "  [b]           - Back to summary view"],
            ),
            (
                "📋 TRADE HISTORY TAB KEYS",
                [
                    "  [↓]/[↑], [j]/[k] - Move selection down / up",
                    "  [n] / [p]     - Next / previous page",
                    "  [d]           - Drill into selected trade (full detail view)",
                    "  [b] / [d]     - Back from detail view to trade list",
                ],
            ),
            (
                "📊 TAB DESCRIPTIONS",
                [
                    "  Overview      - Quick snapshot of position, daily stats, risk, and health",
                    "  Performance   - Trade-log metrics for 24h/7d/month, epoch, lifetime, and current sessions",
                    "  Training      - Agent training status, buffer sizes, loss metrics",
                    "  Risk          - Circuit breaker (with trip reasons + reset), VaR, vol, regime",
                    "  Market        - Spread, VPIN toxicity, order imbalance, depth",
                    "  Decision Log  - Last 20 trading decisions with color-coded events",
                    "  Trades        - Full trade history, paginated, with per-trade drill-down",
                ],
            ),
            (
                "🎨 COLOR CODING",
                [
                    f"  {_ANSI_G}✓ Green{_ANSI_RST}       - Positive values, good status, active longs",
                    f"  {_ANSI_R}✗ Red{_ANSI_RST}         - Negative values, alerts, active shorts",
                    f"  {_ANSI_Y}⚡ Yellow{_ANSI_RST}      - Neutral/warning, hold actions",
                    f"  {_ANSI_B}ℹ Blue{_ANSI_RST}        - Informational messages",
                ],
            ),
            (
                "📁 DATA SOURCES",
                [
                    "  All data is read from JSON/JSONL files in the 'data/' directory:",
                    f"    • {_BOT_CONFIG_FILE:<27} - Bot configuration and status",
                    "    • current_position_SYM_MTF.json - Active position (per symbol/timeframe)",
                    "    • trade_log.jsonl           - All closed trades (primary source for performance)",
                    "    • training_stats.json        - Agent training statistics",
                    "    • training_stats_SYM_MTF.json- Per-bot training stats (overrides shared file)",
                    "    • risk_metrics.json          - Risk metrics (drawdown, VaR, circuit breakers)",
                    "    • order_book.json            - Live market data (spread, depth, VPIN, imbalance)",
                    "    • performance_snapshot.json  - Trading mode identifier only",
                    "    • logs/audit/decisions.jsonl - Decision history (primary, rich JSONL format)",
                    "    • decision_log.json          - Decision history (legacy fallback only)",
                ],
            ),
            (
                "⚙️  SYSTEM REQUIREMENTS",
                [
                    "  • Terminal with UTF-8 support",
                    "  • ANSI color support",
                    "  • Minimum 80x24 terminal size recommended",
                    "  • Bot must be running and exporting data files",
                ],
            ),
            (
                "🔧 TROUBLESHOOTING",
                [
                    "  Data stale warning    - Bot may be paused or crashed",
                    "  Missing files         - Check that bot is running and exporting",
                    "  Garbled display       - Ensure terminal supports UTF-8 and ANSI colors",
                    "  Keyboard not working  - Try running in a different terminal emulator",
                ],
            ),
        ]

    def _show_help(self) -> None:
        """Display help screen with keyboard shortcuts and information."""
        self._disable_raw_mode()
        try:
            self._menu_enter()
            print("╔" + "═" * 78 + "╗")
            print("║" + " " * 25 + "HUD HELP & REFERENCE" + " " * 32 + "║")
            print("╚" + "═" * 78 + "╝\n")
            for idx, (title, rows) in enumerate(self._help_sections()):
                self._print_help_section(title, rows, gap_before=idx > 0)
            print("\n" + "─" * 80)
            input("Press Enter to return to HUD...")
        except Exception as e:
            print(f"Error displaying help: {e}")
            input("Press Enter to continue...")
        finally:
            self._enable_raw_mode()

    def _control_file_targets(self, filename: str) -> list[Path]:
        """Return root and per-bot control-file paths for HUD broadcasts."""
        targets: list[Path] = [self.data_dir / filename]
        try:
            targets.extend(
                bot_dir / filename
                for bot_dir in sorted(self.data_dir.glob("paper_*_M*"))
                if bot_dir.is_dir()
            )
        except OSError:
            pass

        seen: set[Path] = set()
        unique: list[Path] = []
        for path in targets:
            try:
                key = path.resolve()
            except OSError:
                key = path
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    @staticmethod
    def _atomic_write_control_file(path: Path, payload: dict, prefix: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=prefix,
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _broadcast_control_file(self, filename: str, payload: dict, prefix: str) -> None:
        """Write a control request to every isolated runtime plus root fallback."""
        errors: list[str] = []
        for path in self._control_file_targets(filename):
            try:
                self._atomic_write_control_file(path, payload, prefix)
            except OSError as exc:
                errors.append(f"{path}: {exc}")
        if errors:
            raise RuntimeError("Failed to write control request to " + "; ".join(errors))

    def _handle_kill_switch(self) -> None:
        """Alt+K: confirm and write kill_switch.json — bot background thread acts within 5 seconds."""
        self._disable_raw_mode()
        try:
            self._menu_enter()
            RED = _ANSI_R
            YLW = _ANSI_Y
            RST = _ANSI_RST
            print(RED + "╔" + "═" * 60 + "╗")
            print("║" + " " * 16 + "⚠️  EMERGENCY KILL SWITCH" + " " * 17 + "║")
            print("╚" + "═" * 60 + "╝" + RST + "\n")
            print("This will (within ~5 seconds, regardless of bar interval):")
            print("  1. Trip ALL circuit breakers immediately")
            print("  2. Close ALL open positions via emergency close")
            print("  3. Halt all new entries until circuit breakers are manually reset\n")
            print(YLW + "Type KILL and press Enter to confirm, or press Enter to abort:" + RST)
            confirm = input("> ").strip()
            if confirm == "KILL":
                _ks_data = {
                    "active": True,
                    "reason": "MANUAL_HUD_KILL",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                self._broadcast_control_file("kill_switch.json", _ks_data, prefix=".kill_switch_")
                print("\n" + RED + "✓ KILL SWITCH ACTIVATED — bot will close all positions within 5 seconds" + RST)
                self._set_notification("🚨 KILL SWITCH ACTIVATED — closing all positions", ttl=120)
                input("\nPress Enter to return to HUD...")
            else:
                print("\n" + YLW + "Aborted — no action taken." + RST)
                time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            with contextlib.suppress(Exception):
                input("Press Enter to continue...")
        finally:
            self._enable_raw_mode()

    def _load_circuit_breaker_state(self) -> dict:
        cb_path = self.data_dir / "circuit_breakers.json"
        if not cb_path.exists():
            return {}
        try:
            with open(cb_path, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    @staticmethod
    def _breaker_labels() -> dict[str, tuple[str, str]]:
        return {
            "sortino": ("Sortino Ratio", "Risk-adjusted returns too low"),
            "kurtosis": ("Kurtosis", "Return distribution has fat tails"),
            "drawdown": ("Drawdown", "Equity drawdown exceeded limit"),
            "consecutive_losses": ("Consecutive Losses", "Too many losses in a row"),
        }

    def _kurtosis_gate_review(self) -> tuple[bool, float, float, str]:
        gate_active = bool(self.risk_stats.get("kurtosis_gate_active", False))
        kurtosis_now = float(self.risk_stats.get("kurtosis", 0.0) or 0.0)
        threshold = float(
            self.risk_stats.get("kurtosis_threshold", KURTOSIS_FAT_TAIL_THRESHOLD) or KURTOSIS_FAT_TAIL_THRESHOLD
        )
        return gate_active, kurtosis_now, threshold, self._risk_scope_label(self.risk_stats)

    def _print_breaker_cooldown(self, breaker: dict) -> None:
        trip_ts = breaker.get("trip_time", "")
        if not trip_ts:
            return
        print(f"    Tripped:   {trip_ts[:19]}")
        try:
            trip_dt = datetime.fromisoformat(trip_ts)
            cooldown_mins = breaker.get("cooldown_minutes", 60)
            elapsed = (datetime.now(UTC) - trip_dt).total_seconds() / 60.0
            remaining = max(0, cooldown_mins - elapsed)
            if remaining > 0:
                print(f"    Cooldown:  {remaining:.0f}m remaining (auto-reset after {cooldown_mins}m)")
            else:
                print(f"    Cooldown:  {_ANSI_G}Elapsed — safe to reset{_ANSI_RST}")
        except (ValueError, TypeError):
            pass

    def _print_tripped_breaker(
        self,
        label: str,
        reason: str,
        value: float,
        threshold: float,
        breaker: dict,
    ) -> None:
        print(f"  {_ANSI_R}✗ {label}: TRIPPED{_ANSI_RST}")
        print(f"    Reason:    {_ANSI_Y}{reason}{_ANSI_RST}")
        print(f"    Value:     {value:.4f}  (threshold: {threshold:.4f})")
        self._print_breaker_cooldown(breaker)
        print()

    def _print_circuit_breaker_rows(self, cb_data: dict) -> bool:
        any_tripped = False
        kurt_gate_active, kurtosis_now, kurtosis_threshold, risk_scope = self._kurtosis_gate_review()
        for key, (label, explain) in self._breaker_labels().items():
            breaker = cb_data.get(key, {})
            breaker = breaker if isinstance(breaker, dict) else {}
            tripped = bool(breaker.get("is_tripped", False))
            gate_only = key == "kurtosis" and kurt_gate_active and not tripped
            if not tripped and not gate_only:
                print(f"  {_ANSI_G}✓ {label}: OK{_ANSI_RST}")
                continue
            any_tripped = True
            reason = (
                f"Kurtosis gate active [{risk_scope}] (entry gate)"
                if gate_only
                else breaker.get("trip_reason", explain)
            )
            value = kurtosis_now if gate_only else breaker.get("trip_value", 0.0)
            threshold = kurtosis_threshold if gate_only else breaker.get("threshold", 0.0)
            self._print_tripped_breaker(label, reason, float(value), float(threshold), breaker)
        return any_tripped

    def _confirm_cb_reset(self) -> bool:
        print(f"\n{_ANSI_DIM}Resetting will allow the bot to resume trading immediately.{_ANSI_RST}")
        print(f"{_ANSI_DIM}Only reset if you understand why the breaker tripped and the condition is resolved.")
        print(f"{_ANSI_RST}")
        print(
            _ANSI_Y
            + "Type 'reset' and press Enter to reset all tripped breakers, or press Enter to abort:"
            + _ANSI_RST
        )
        return input("> ").strip().upper() == "RESET"

    def _send_cb_reset_request(self) -> None:
        reset_data = {
            "reset": True,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._broadcast_control_file("circuit_breaker_reset.json", reset_data, prefix=".cb_reset_")
        print(f"\n{_ANSI_G}✓ Reset request sent — bot will reset breakers within ~5 seconds{_ANSI_RST}")
        self._set_notification("🔄 Circuit breaker reset requested", ttl=30)
        input("\nPress Enter to return to HUD...")

    def _handle_cb_reset(self) -> None:
        """R key: Show tripped circuit breakers with reasons and offer reset."""
        self._disable_raw_mode()
        try:
            self._menu_enter()
            print(_ANSI_Y + "╔" + "═" * 60 + "╗")
            print("║" + " " * 14 + "🔌 CIRCUIT BREAKER REVIEW" + " " * 19 + "║")
            print("╚" + "═" * 60 + "╝" + _ANSI_RST + "\n")
            if not self._print_circuit_breaker_rows(self._load_circuit_breaker_state()):
                print(f"\n  {_ANSI_G}All circuit breakers are OK — nothing to reset.{_ANSI_RST}")
                input("\nPress Enter to return to HUD...")
                return

            if self._confirm_cb_reset():
                self._send_cb_reset_request()
            else:
                print(f"\n{_ANSI_Y}Aborted — no action taken.{_ANSI_RST}")
                time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            with contextlib.suppress(Exception):
                input("Press Enter to continue...")
        finally:
            self._enable_raw_mode()

    def _print_stats_epoch_state(self) -> None:
        if self._stats_epoch:
            epoch_str = self._stats_epoch.strftime("%Y-%m-%d %H:%M UTC")
            print(f"  Current epoch: {_ANSI_G}{epoch_str}{_ANSI_RST}")
            print(
                f"  Excluded:      {self._stats_epoch_excluded} trades, "
                f"${self._stats_epoch_excluded_pnl:+.2f} PnL\n"
            )
            return
        print(f"  Current epoch: {_ANSI_DIM}None (all trades included){_ANSI_RST}\n")

    def _print_stats_epoch_options(self) -> None:
        print("  Options:")
        print(f"    {_ANSI_Y}1{_ANSI_RST}  Set epoch to NOW (fresh start from this moment)")
        print(f"    {_ANSI_Y}2{_ANSI_RST}  Set epoch to start of today")
        print(f"    {_ANSI_Y}3{_ANSI_RST}  Set epoch to 7 days ago")
        print(f"    {_ANSI_Y}4{_ANSI_RST}  Set epoch to 30 days ago")
        print(f"    {_ANSI_Y}5{_ANSI_RST}  Enter a custom date (YYYY-MM-DD)")
        print(f"    {_ANSI_Y}c{_ANSI_RST}  Clear epoch (show all trades)")
        print(f"    {_ANSI_DIM}Enter{_ANSI_RST}  Cancel\n")

    def _resolve_stats_epoch_choice(self, choice: str) -> tuple[datetime | None, bool]:
        now = datetime.now(UTC)
        options = {
            "1": now,
            "2": now.replace(hour=0, minute=0, second=0, microsecond=0),
            "3": now - timedelta(days=7),
            "4": now - timedelta(days=30),
        }
        if choice in options:
            return options[choice], True
        if choice != "5":
            return None, False
        parsed = _hud_parse_dt(input("Enter date (YYYY-MM-DD): ").strip() + "T00:00:00+00:00")
        if parsed:
            return parsed, True
        print(f"\n  {_ANSI_R}Invalid date format.{_ANSI_RST}")
        input("Press Enter to return to HUD...")
        return None, True

    def _clear_stats_epoch_from_menu(self) -> None:
        self._save_stats_epoch(None)
        self._set_notification("Stats epoch cleared — all trades included", ttl=6)
        self._trade_log_reader.invalidate()

    def _handle_stats_epoch(self) -> None:
        """[e] key: Set or clear the stats epoch to exclude old trades from metrics."""
        self._disable_raw_mode()
        try:
            self._menu_enter()
            print(_ANSI_Y + "╔" + "═" * 60 + "╗")
            print("║" + " " * 14 + "📅 STATS EPOCH MANAGER" + " " * 22 + "║")
            print("╚" + "═" * 60 + "╝" + _ANSI_RST + "\n")
            self._print_stats_epoch_state()
            self._print_stats_epoch_options()
            choice = input("Selection: ").strip().lower()

            if choice == "c":
                self._clear_stats_epoch_from_menu()
                return
            new_epoch, handled = self._resolve_stats_epoch_choice(choice)
            if not handled:
                return

            self._save_stats_epoch(new_epoch)
            label = new_epoch.strftime("%Y-%m-%d %H:%M UTC") if new_epoch else "cleared"
            self._set_notification(f"Stats epoch set to {label}", ttl=6)
            self._trade_log_reader.invalidate()
        except Exception as exc:
            print(f"\nError: {exc}")
            with contextlib.suppress(Exception):
                input("Press Enter to continue...")
        finally:
            self._enable_raw_mode()

    def _handle_session_selector(self) -> None:
        """Launch the interactive session selector TUI (replaces legacy preset picker)."""
        import subprocess
        self._disable_raw_mode()
        try:
            self._menu_enter()
            result = subprocess.run(
                [sys.executable, "-m", "src.monitoring.session_selector"],
                check=False,
            )
            if result.returncode == 0:
                self._set_notification(
                    "Session saved → restart run.sh to apply new instrument/mode config", ttl=12
                )
            else:
                self._set_notification("Session selector cancelled — no changes made", ttl=6)
        except Exception as exc:
            self._set_notification(f"Selector error: {exc}", ttl=8)
        finally:
            self._enable_raw_mode()
            self._force_redraw = True

    def _handle_symbol_selection(self) -> None:
        """Deprecated: kept for any external callers — delegates to new selector."""
        self._handle_session_selector()

    def _apply_profile_selection(self, selection: dict[str, Any]) -> bool:
        """Write selection to .env and pending profile file."""
        updates = {
            "SYMBOL": selection["symbol"],
            "SYMBOL_ID": str(selection["symbol_id"]),
            "TIMEFRAME_MINUTES": str(selection["timeframe_minutes"]),
            "QTY": str(selection["qty"]),
        }
        env_updated = self._update_env_file(Path(".env"), updates)
        self._write_pending_profile(selection)
        label = selection.get("label") or f"{selection['symbol']} M{selection['timeframe_minutes']}"
        if env_updated:
            self._set_notification(f"Queued {label}. Restart run.sh to apply.", ttl=10)
        else:
            self._set_notification("Preset saved (pending_profile.json) but .env missing", ttl=10)
        return env_updated

    def _update_env_file(self, env_path: Path, updates: dict[str, str]) -> bool:
        """Update target keys inside .env while preserving other settings."""
        if not env_path.exists():
            return False
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return False
        seen = set()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in line:
                new_lines.append(line)
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                seen.add(key)
            else:
                new_lines.append(line)
        for key, value in updates.items():
            if key not in seen:
                new_lines.append(f"{key}={value}")
        try:
            env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            return True
        except Exception:
            return False

    def _write_pending_profile(self, selection: dict[str, Any]) -> None:
        """Persist requested profile for other tools/dashboard consumers."""
        payload = {
            "symbol": selection["symbol"],
            "symbol_id": selection["symbol_id"],
            "timeframe_minutes": selection["timeframe_minutes"],
            "qty": selection["qty"],
            "label": selection.get("label"),
            "requested_at": datetime.now(UTC).isoformat() + "Z",
            "status": "pending_restart",
        }
        self.data_dir.mkdir(exist_ok=True)
        try:
            with open(self.data_dir / "pending_profile.json", "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            pass

    def _set_notification(self, message: str, ttl: int = 5) -> None:
        """Display a temporary status message in the footer."""
        self.notification = message
        self.notification_expiry = datetime.now(UTC) + timedelta(seconds=ttl)

    def _current_notification(self) -> str:
        if self.notification and datetime.now(UTC) < self.notification_expiry:
            return self.notification
        return ""

    def _render(self) -> None:
        """Render current tab with fixed header/footer and scrollable body."""
        header_buf = io.StringIO()
        with redirect_stdout(header_buf):
            self._render_header()
            self._render_tab_bar()

        body_buf = io.StringIO()
        with redirect_stdout(body_buf):
            if self._ctx_level == 0:
                self._render_mode_selector()
            elif self.current_tab == "overview":
                self._render_overview()
            elif self.current_tab == "performance":
                self._render_performance()
            elif self.current_tab == "training":
                self._render_training()
            elif self.current_tab == "risk":
                self._render_risk()
            elif self.current_tab == "market":
                self._render_market()
            elif self.current_tab == "log":
                self._render_decision_log()
            elif self.current_tab == "trades":
                self._render_trades()

        footer_buf = io.StringIO()
        with redirect_stdout(footer_buf):
            self._render_footer()

        frame = self._compose_viewport_frame(
            header_buf.getvalue(),
            body_buf.getvalue(),
            footer_buf.getvalue(),
        )
        frame_key = frame
        frame_key = re.sub(
            r"^[^\n]*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$",
            "<HEARTBEAT>",
            frame_key,
            flags=re.MULTILINE,
        )
        frame_key = re.sub(r"\((\d+(?:\.\d+)?)s\)", "(<AGE>s)", frame_key)
        if self._force_redraw or frame_key != self._last_frame_key:
            # Root-cause of the post-flicker-fix 'duplicate / tab bleed'
            # symptoms: the previous in-place ESC[H + per-line ESC[K repaint
            # assumed every logical line fits in a single terminal row and
            # that the cursor position after writing was deterministic.  In
            # practice, long header/footer lines wrap, the terminal scrolls,
            # and ESC[H lands at the top of the *current viewport* rather
            # than the top of the scrolled content, leaving fragments of the
            # previous tab on-screen.
            #
            # We now render on the alternate screen buffer (see start()), so
            # ESC[2J truly erases cells instead of pushing them into
            # scrollback.  Erase first, then home the cursor, then write the
            # new frame.
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(frame)
            sys.stdout.write("\033[J")
            sys.stdout.flush()
            self._last_frame = frame
            self._last_frame_key = frame_key
            self._force_redraw = False

    @staticmethod
    def _frame_lines(text: str) -> list[str]:
        """Split rendered text into display lines without counting a final blank."""
        if not text:
            return []
        lines = text.splitlines()
        if text.endswith("\n"):
            return lines
        return lines

    def _scroll_status_line(self, offset: int, max_scroll: int) -> str:
        W = self._term_width()
        page = min(offset + 1, max_scroll + 1)
        label = f"body scroll {page}/{max_scroll + 1}  mouse wheel / ↑↓ / j/k / PgUp/PgDn"
        return f"  {_ANSI_DIM}{label[: max(0, W - 4)]}{_ANSI_RST}"

    @staticmethod
    def _with_right_scrollbar(
        lines: list[str], *, offset: int, max_scroll: int, body_rows: int, width: int
    ) -> list[str]:
        """Append a visual right-edge scrollbar to body lines."""
        if width < 20 or body_rows <= 0 or max_scroll <= 0:
            return lines
        track_rows = len(lines)
        if track_rows <= 0:
            return lines
        thumb_rows = max(1, int(track_rows * track_rows / (track_rows + max_scroll)))
        thumb_start = 0 if max_scroll <= 0 else round((track_rows - thumb_rows) * offset / max_scroll)
        out: list[str] = []
        for idx, line in enumerate(lines):
            bar = "█" if thumb_start <= idx < thumb_start + thumb_rows else "│"
            content = _truncate_visible(line, max(0, width - 1))
            pad = " " * max(0, width - 1 - _visible_width(content))
            out.append(f"{content}{pad}{_ANSI_DIM}{bar}{_ANSI_RST}")
        return out

    def _compose_viewport_frame(self, header: str, body: str, footer: str) -> str:
        """Keep header/footer visible and clip the current tab body to terminal height."""
        height = max(10, self._term_height())
        width = self._term_width()
        header_lines = [_truncate_visible(line, width) for line in self._frame_lines(header)]
        body_lines = [_truncate_visible(line, width) for line in self._frame_lines(body)]
        footer_lines = [_truncate_visible(line, width) for line in self._frame_lines(footer)]

        body_rows = max(1, height - len(header_lines) - len(footer_lines))
        max_scroll = max(0, len(body_lines) - body_rows)
        offset = max(0, min(self._body_scroll_offsets.get(self.current_tab, 0), max_scroll))
        self._body_scroll_offsets[self.current_tab] = offset
        self._body_scroll_max = max_scroll

        if self.current_tab == "trades":
            visible_body = body_lines[:body_rows]
        elif max_scroll > 0 and body_rows > 1:
            visible_body = body_lines[offset : offset + body_rows - 1]
            visible_body.append(self._scroll_status_line(offset, max_scroll))
        else:
            visible_body = body_lines[:body_rows]
        visible_body = self._with_right_scrollbar(
            visible_body,
            offset=offset,
            max_scroll=max_scroll,
            body_rows=body_rows,
            width=width,
        )

        frame_lines = header_lines + visible_body + footer_lines
        return "\n".join(frame_lines) + "\n"

    # ── _render_training helpers ──────────────────────────────────────────

    def _rt_pct_bar(self, val: int, cap: int) -> str:
        """Render a fill-percentage bar for buffer occupancy."""
        pct = min(val / cap, 1.0) if cap > 0 else 0.0
        filled = int(_RT_BAR_LEN * pct)
        if pct > _BUF_FILL_HIGH:
            col = _ANSI_G
        elif pct > _BUF_FILL_WARN:
            col = _ANSI_Y
        else:
            col = _ANSI_R
        return f"{col}[{'█' * filled}{'░' * (_RT_BAR_LEN - filled)}]{_ANSI_RST} {pct * 100:5.1f}%"

    def _rt_eps_bar(self, eps: float) -> str:
        """Render an epsilon-exploration fill bar with hot/warm/cold label."""
        pct = max(0.0, min(eps, 1.0))
        filled = int(_RT_BAR_LEN * pct)
        if eps > EPS_WARM_MAX:
            bracket = f"{_ANSI_R}COLD{_ANSI_RST}"
            col = _ANSI_R
        elif eps > EPS_HOT_MAX:
            bracket = f"{_ANSI_Y}WARM{_ANSI_RST}"
            col = _ANSI_Y
        else:
            bracket = f"{_ANSI_G}HOT{_ANSI_RST}"
            col = _ANSI_G
        return f"{col}[{'█' * filled}{'░' * (_RT_BAR_LEN - filled)}] {eps:.4f}{_ANSI_RST} {bracket}"

    def _rt_beta_bar(self, beta: float) -> str:
        """Render an IS-beta fill bar (0.4 cold → 1.0 fully corrected)."""
        pct = max(0.0, min((beta - 0.4) / 0.6, 1.0))
        filled = int(_RT_BAR_LEN * pct)
        if beta > BETA_HOT_MIN:
            col = _ANSI_G
        elif beta > BETA_WARM_MIN:
            col = _ANSI_Y
        else:
            col = _ANSI_DIM
        return (
            f"{col}[{'█' * filled}{'░' * (_RT_BAR_LEN - filled)}] {beta:.4f}{_ANSI_RST}  (0.4 cold→1.0 fully corrected)"
        )

    def _rt_trend(self, hist: deque) -> str:
        """Return a coloured ↓/↑/→ trend string from a loss-history deque."""
        vals = [v for v in hist if v > 0]
        if len(vals) < TREND_MIN_SAMPLES:
            return f"{_ANSI_DIM}→ — (need more samples){_ANSI_RST}"
        half = len(vals) // 2
        old_mean = sum(vals[:half]) / half
        new_mean = sum(vals[half:]) / (len(vals) - half)
        delta_pct = (new_mean - old_mean) / old_mean * 100 if old_mean > 0 else 0
        if delta_pct < TREND_DELTA_NEG:
            return f"{_ANSI_G}↓ {abs(delta_pct):.1f}% IMPROVING{_ANSI_RST}"
        if delta_pct > TREND_DELTA_POS:
            return f"{_ANSI_R}↑ +{delta_pct:.1f}% DEGRADING{_ANSI_RST}"
        return f"{_ANSI_Y}→ {delta_pct:+.1f}% STABLE{_ANSI_RST}"

    def _rt_spark(self, hist: deque) -> str:
        """Return a sparkline string from the tail of a loss-history deque."""
        vals = [v for v in hist if v > 0]
        if len(vals) < TREND_STEP_PAIRS_MIN:
            return ""
        return self._create_sparkline(vals[-TREND_SPARK_TAIL:])

    def _rt_velocity(self, step_hist: deque) -> str:
        """Return a coloured steps/min rate string from a step-history deque."""
        pairs = list(step_hist)
        if len(pairs) < TREND_STEP_PAIRS_MIN:
            return f"{_ANSI_DIM}—{_ANSI_RST}"
        dt = pairs[-1][0] - pairs[0][0]
        ds = pairs[-1][1] - pairs[0][1]
        if dt <= 0 or ds <= 0:
            return f"{_ANSI_DIM}0 steps/min{_ANSI_RST}"
        rate = ds / dt * 60
        if rate > TREND_RATE_HIGH:
            col = _ANSI_G
        elif rate > TREND_RATE_WARN:
            col = _ANSI_Y
        else:
            col = _ANSI_DIM
        return f"{col}{rate:.1f} steps/min{_ANSI_RST}"

    def _offline_status_normalized(self, ofs: dict) -> str:
        """Return normalized offline-training state label."""
        _raw = str(ofs.get("status", "idle") or "idle").strip().lower()
        if _raw in {"complete", "completed", "done", "finished", "success"}:
            return "complete"
        if _raw in {"running", "in_progress", "in-progress", "active", "started"}:
            return "running"
        return _raw

    def _offline_total_jobs(self, ofs: dict, results: list) -> int:
        """Return robust total_jobs value from status payload."""
        _total = int(ofs.get("total_jobs", 0) or 0)
        if _total > 0:
            return _total
        return len(results)

    def _render_offline_training(self, ofs: dict, *, detail: bool = True) -> None:
        """Render the offline training status block."""
        _results = ofs.get("results", [])
        ofs_status = self._offline_status_normalized(ofs)
        ofs_total = self._offline_total_jobs(ofs, _results)
        ofs_done = sum(1 for r in _results if r.get("status") in ("done", "error"))
        # Compute elapsed: from started_at for running jobs, else from stale elapsed_s.
        ofs_elapsed = 0.0
        ofs_start_str = ofs.get("started_at", "")
        ofs_end_str = ofs.get("completed_at", "")
        if ofs_status == "running" and ofs_start_str:
            try:
                _start = datetime.fromisoformat(ofs_start_str.replace("Z", "+00:00"))
                ofs_elapsed = (datetime.now(UTC) - _start).total_seconds()
            except (ValueError, TypeError):
                ofs_elapsed = ofs.get("elapsed_s", 0.0)
        else:
            ofs_elapsed = ofs.get("elapsed_s", 0.0)
        ofs_start = ofs_start_str[:19].replace("T", " ") if ofs_start_str else "—"
        ofs_end = ofs_end_str[:19].replace("T", " ") if ofs_end_str else None
        status_badge = self._offline_status_badge(ofs_status, ofs_done, ofs_total)
        prog_bar = self._offline_progress_bar(ofs_done, ofs_total)
        _elapsed_h = int(ofs_elapsed // 3600)
        _elapsed_m = int((ofs_elapsed % 3600) // 60)
        _elapsed_s = int(ofs_elapsed % 60)
        if _elapsed_h:
            _elapsed_str = f"{_elapsed_h}h {_elapsed_m}m"
        else:
            _elapsed_str = f"{_elapsed_m}m {_elapsed_s}s"
        print(f"\n  \033[1m🏋 OFFLINE TRAINING\033[0m  {status_badge}")
        print(f"    Progress:  {prog_bar}   Elapsed: {_elapsed_str}")
        print(f"    Started:   {ofs_start}" + (f"   Finished: {ofs_end}" if ofs_end else ""))
        if _results and detail:
            self._render_offline_jobs_table(_results)
        elif _results:
            _counts: dict[str, int] = {}
            for _row in _results:
                _key = str(_row.get("status", "unknown") or "unknown").lower()
                _counts[_key] = _counts.get(_key, 0) + 1
            _summary = " ".join(f"{k}:{v}" for k, v in sorted(_counts.items()))
            _active = next((r for r in _results if r.get("status") == "running"), None)
            print(f"    Jobs:      {_summary or 'none'}")
            if _active:
                _sym = str(_active.get("symbol", "?"))
                _tf = _active.get("label", f"M{_active.get('timeframe_minutes', '?')}")
                _detail = self._offline_job_detail("running", _active).strip()
                print(f"    Active:    {_sym}/{_tf}  {_detail}")
        print()

    def _offline_status_badge(self, status: str, done: int, total: int) -> str:
        """Build a colorized offline-training status badge."""
        if status == "running":
            return f"{_ANSI_Y}⚙  RUNNING ({done}/{total} done){_ANSI_RST}"
        if status == "complete":
            return f"{_ANSI_G}✓ COMPLETE  ({done}/{total} jobs){_ANSI_RST}"
        if status in {"idle", ""}:
            return f"{_ANSI_DIM}idle{_ANSI_RST}"
        return f"{_ANSI_DIM}{status}{_ANSI_RST}"

    def _offline_progress_bar(self, done: int, total: int) -> str:
        """Render the offline training progress bar string."""
        pct = done / total if total else 0.0
        filled = int(26 * pct)
        if pct >= 1.0:
            prog_col = _ANSI_G
        elif pct > 0:
            prog_col = _ANSI_Y
        else:
            prog_col = _ANSI_DIM
        return f"{prog_col}[{'█' * filled}{'░' * (26 - filled)}]{_ANSI_RST} {pct * 100:.0f}%"

    def _render_offline_jobs_table(self, results: list) -> None:
        """Render the symbol/TF results table for offline training."""
        sym_w = max(6, *(len(r.get("symbol", "")) for r in results))
        print()
        print(f"    {'Symbol':<{sym_w}}  {'TF':>5}  {'Status':<9}  {'Detail':<38}  {'ZOmega':>8}  {'Comment':<14}")
        print(f"    {'─' * sym_w}  {'─' * 5}  {'─' * 9}  {'─' * 38}  {'─' * 8}  {'─' * 14}")
        for r in results:
            self._render_offline_job_row(r, sym_w)

    def _render_offline_job_row(self, r: dict, sym_w: int) -> None:
        """Render a single job result row in the offline training table."""
        sym = r.get("symbol", "")
        tf_label = r.get("label", f"M{r.get('timeframe_minutes', '?')}")
        jstatus = r.get("status", "queued")
        jcol, jbadge = self._offline_job_status(jstatus)
        zo_str = self._offline_job_zo_str(r.get("z_omega"), jstatus, r.get("val_trades"))
        detail = self._offline_job_detail(jstatus, r)
        comment = self._offline_job_comment(jstatus, r)
        row = f"    {sym:<{sym_w}}  {tf_label:>5}  {jcol}{jbadge}{_ANSI_RST}  {detail}  {zo_str}  {comment}"
        if jstatus == "error" and r.get("error"):
            row += f"  {_ANSI_R}{r['error'][:30]}{_ANSI_RST}"
        print(row)

    def _offline_job_comment(self, status: str, r: dict) -> str:
        """Return the Comment column text for an offline job row.

        - Running jobs: show "run X/Y" from progress file epoch/n_epochs.
        - Done jobs: show an operator-facing acceptance outcome.
        - Otherwise: "—".
        """
        if status == "running":
            prog = self.offline_job_progress.get((r.get("symbol"), r.get("timeframe_minutes")), {})
            epoch = int(prog.get("epoch", 0) or 0)
            n_epochs = int(prog.get("n_epochs", 0) or 0)
            if epoch and n_epochs:
                return f"{_ANSI_Y}run {epoch}/{n_epochs}{_ANSI_RST}"
            return f"{'—':<14}"
        if status == "done":
            reason = r.get("accept_reason", "")
            if reason:
                return self._offline_acceptance_comment(str(reason), r)
            accepted = r.get("accepted", False)
            return f"{_ANSI_G}accepted{_ANSI_RST}" if accepted else f"{_ANSI_DIM}not accepted{_ANSI_RST}"
        return f"{'—':<14}"

    @staticmethod
    def _offline_acceptance_comment(reason: str, r: dict) -> str:
        """Compress acceptance reasons into readable HUD comments."""
        accepted = r.get("accepted", False)
        if reason.startswith("candidate_not_better_than_"):
            source = reason.removeprefix("candidate_not_better_than_")
            if source == "champion":
                return f"{_ANSI_DIM}kept champion{_ANSI_RST}"
            if source == "incumbent":
                return f"{_ANSI_DIM}kept runtime{_ANSI_RST}"
            return f"{_ANSI_DIM}not promoted{_ANSI_RST}"
        if "weights_missing" in reason:
            return f"{_ANSI_R}weights miss{_ANSI_RST}"
        if reason.startswith("candidate_better_than_"):
            source = reason.removeprefix("candidate_better_than_").removesuffix("_deferred")
            if source == "champion":
                return f"{_ANSI_G}beat champion{_ANSI_RST}"
            if source == "incumbent":
                return f"{_ANSI_G}beat runtime{_ANSI_RST}"
            return f"{_ANSI_G}promoted{_ANSI_RST}"
        if accepted:
            return f"{_ANSI_G}{reason[:14]}{_ANSI_RST}"
        return f"{_ANSI_DIM}{reason[:14]}{_ANSI_RST}"

    def _offline_job_status(self, status: str) -> tuple[str, str]:
        """Return (color, badge) for offline job status."""
        if status == "done":
            return _ANSI_G, "done     "
        if status == "error":
            return _ANSI_R, "ERROR    "
        if status == "running":
            return _ANSI_Y, "running  "
        return _ANSI_DIM, "queued   "

    def _offline_job_zo_str(self, zo: float | None, status: str, val_trades: int | None = None) -> str:
        """Render ZOmega column for offline job row."""
        if zo is None or status != "done":
            return f"{_ANSI_DIM}{'—':>8}{_ANSI_RST}"
        # ZOmega is intentionally forced to 0.0 by offline trainer when
        # validation trades are insufficient (<5); show this explicitly.
        if int(val_trades or 0) < 5:
            return f"{_ANSI_DIM}{'n/a<5':>8}{_ANSI_RST}"
        if zo >= 1.0:
            zo_col = _ANSI_G
        elif zo >= Z_OMEGA_OFFLINE_WARM_MIN:
            zo_col = _ANSI_Y
        else:
            zo_col = _ANSI_R
        return f"{zo_col}{zo:8.4f}{_ANSI_RST}"

    def _offline_job_detail(self, status: str, r: dict) -> str:
        """Render detail column for offline job row."""
        if status in ("done", "error"):
            ttrades = r.get("train_trades", 0)
            vtrades = r.get("val_trades", 0)
            steps = r.get("total_train_steps", 0)
            return f"tr={ttrades:,}  val={vtrades:,}  steps={steps:,}".ljust(38)
        if status == "running":
            prog = self.offline_job_progress.get((r.get("symbol"), r.get("timeframe_minutes")), {})
            if not prog:
                return f"{'—':<38}"
            pb = prog.get("pct", 0.0)
            pb_fill = int(14 * pb / 100)
            pb_bar = f"[{'█' * pb_fill}{'░' * (14 - pb_fill)}] {pb:4.1f}%"
            return f"{_ANSI_Y}{pb_bar}{_ANSI_RST}  ε={prog.get('epsilon', 0):.3f}  β={prog.get('beta', 0.4):.3f}"
        return f"{'—':<38}"

    def _render_trading_pipeline(self) -> None:
        """Render the trading pipeline status block — one card per bot.

        Skips rendering entirely when there are no live entries to avoid
        showing stale information from dead pipeline processes.
        """
        uni = self.universe_stats
        if not uni:
            return  # nothing to show — all entries were pruned or none exist
        running_count = sum(1 for e in uni.values() if e.get("_pid_alive"))
        total_count = len(uni)
        hdr_badge = (
            f"{_ANSI_G}{running_count}/{total_count} running{_ANSI_RST}"
            if running_count
            else f"{_ANSI_R}0/{total_count} running{_ANSI_RST}"
        )
        print(f"  \033[1m📈 TRADING PIPELINE\033[0m  {hdr_badge}")
        print()
        for _, entry in sorted(
            uni.items(), key=lambda kv: (str(kv[1].get("symbol", "")), int(kv[1].get("timeframe_minutes", 0) or 0))
        ):
            self._render_pipeline_card(str(entry.get("symbol", "?")), entry)
        print()

    @staticmethod
    def _pp_bar(filled_frac: float, width: int = 8) -> str:
        """Tiny inline progress bar."""
        filled = round(max(0.0, min(1.0, filled_frac)) * width)
        col = _ANSI_G if filled_frac >= 0.5 else (_ANSI_Y if filled_frac >= 0.2 else _ANSI_DIM)
        return f"{col}[{'█' * filled}{'░' * (width - filled)}]{_ANSI_RST}"

    @staticmethod
    def _pipeline_stage_color(stage: str) -> str:
        return {
            "PAPER": _ANSI_Y,
            "LIVE": _ANSI_G,
            "UNTRAINED": _ANSI_DIM,
            "DEMOTED": _ANSI_R,
        }.get(stage, _ANSI_DIM)

    @staticmethod
    def _pipeline_zo_str(entry: dict) -> str:
        zo = entry.get("z_omega")
        no_weights = not entry.get("weights_path")
        if zo is None or (zo == 0.0 and no_weights):
            return f"{_ANSI_DIM}ZΩ —{_ANSI_RST}"
        zo_c = _ANSI_G if zo > 1.0 else (_ANSI_Y if zo > 0 else _ANSI_R)
        return f"{zo_c}ZΩ {zo:.4f}{_ANSI_RST}"

    @staticmethod
    def _pipeline_pid_str(entry: dict) -> str:
        pid = entry.get("paper_pid")
        if entry.get("_pid_alive", False):
            return f"{_ANSI_G}▶ PID {pid}{_ANSI_RST}"
        return f"{_ANSI_R}✗ dead ({pid}){_ANSI_RST}" if pid else f"{_ANSI_DIM}not started{_ANSI_RST}"

    @staticmethod
    def _pipeline_uptime_str(ps: dict) -> str:
        uptime_s = int(ps.get("uptime_seconds", 0))
        if uptime_s >= 3600:
            return f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m"
        if uptime_s:
            return f"{uptime_s // 60}m {uptime_s % 60}s"
        return "—"

    def _print_pipeline_header(self, sym: str, entry: dict, ps: dict) -> None:
        stage = entry.get("stage", "?")
        tf_min = entry.get("timeframe_minutes", 0)
        tf_lbl = f"M{tf_min}" if tf_min else "?"
        print(
            f"  {_ANSI_B}◼ {sym} {tf_lbl}{_ANSI_RST}  "
            f"{self._pipeline_stage_color(stage)}{stage}{_ANSI_RST}  "
            f"{self._pipeline_zo_str(entry)}  {self._pipeline_pid_str(entry)}  "
            f"uptime {self._pipeline_uptime_str(ps)}"
        )

    @staticmethod
    def _print_pipeline_connection(ps: dict) -> None:
        q_str = f"{_ANSI_G}QUOTE ✓{_ANSI_RST}" if ps.get("quote_ok", False) else f"{_ANSI_R}QUOTE ✗{_ANSI_RST}"
        t_str = f"{_ANSI_G}TRADE ✓{_ANSI_RST}" if ps.get("trade_ok", False) else f"{_ANSI_R}TRADE ✗{_ANSI_RST}"
        h_str = (
            f"{_ANSI_G}healthy{_ANSI_RST}"
            if ps.get("connection_healthy", False)
            else f"{_ANSI_Y}unhealthy{_ANSI_RST}"
        )
        recon = ps.get("total_reconnects", 0)
        r_col = _ANSI_G if recon == 0 else (_ANSI_Y if recon < 5 else _ANSI_R)
        print(f"    FIX: {q_str}  {t_str}  {h_str}  │  reconnects: {r_col}{recon}{_ANSI_RST}")

    @staticmethod
    def _print_pipeline_activity(ps: dict) -> None:
        bars = ps.get("bar_count", 0)
        trades = ps.get("total_trades", 0)
        pnl = ps.get("total_pnl", 0.0)
        wr = ps.get("win_rate", 0.0)
        pnl_c = _ANSI_G if pnl >= 0 else _ANSI_R
        wr_str = f"{wr * 100:.1f}%" if trades > 0 else "—"
        print(f"    Bars: {bars}  │  Trades: {trades}  │  PnL: {pnl_c}{pnl:+.2f}{_ANSI_RST}  │  Win: {wr_str}")

    @staticmethod
    def _print_pipeline_account(ps: dict) -> None:
        real_balance = ps.get("real_account_balance")
        if real_balance is None:
            return
        pnl = ps.get("total_pnl", 0.0)
        real_equity = ps.get("real_account_equity")
        real_margin = ps.get("real_margin_free")
        bal_col = _ANSI_G if pnl >= 0 else _ANSI_R
        equity_str = f"  │  Equity: {_ANSI_B}{float(real_equity):,.2f}{_ANSI_RST}" if real_equity is not None else ""
        margin_str = (
            f"  │  Free margin: {_ANSI_B}{float(real_margin):,.2f}{_ANSI_RST}"
            if real_margin is not None
            else ""
        )
        print(
            f"    Balance: {bal_col}{float(real_balance):,.2f}{_ANSI_RST}  "
            f"{_ANSI_G}✓ live{_ANSI_RST}{equity_str}{margin_str}"
        )

    def _print_pipeline_training(self, ps: dict) -> None:
        trigger_buffer = ps.get("trigger_buffer", 0)
        harvester_buffer = ps.get("harvester_buffer", 0)
        t_bar = self._pp_bar(trigger_buffer / _RT_TRIG_CAP if _RT_TRIG_CAP else 0)
        h_bar = self._pp_bar(harvester_buffer / _RT_HARV_CAP if _RT_HARV_CAP else 0)
        t_pct = f"{100 * trigger_buffer / _RT_TRIG_CAP:4.0f}%" if _RT_TRIG_CAP else ""
        h_pct = f"{100 * harvester_buffer / _RT_HARV_CAP:4.0f}%" if _RT_HARV_CAP else ""
        t_rd = f"{_ANSI_G}ready{_ANSI_RST}" if ps.get("trigger_ready", False) else f"{_ANSI_Y}filling{_ANSI_RST}"
        h_rd = f"{_ANSI_G}ready{_ANSI_RST}" if ps.get("harvester_ready", False) else f"{_ANSI_Y}filling{_ANSI_RST}"
        t_loss = ps.get("trigger_loss", 0.0)
        h_loss = ps.get("harvester_loss", 0.0)
        t_ls = f"{t_loss:.4f}" if t_loss > 0 else f"{_ANSI_DIM}—{_ANSI_RST}"
        h_ls = f"{h_loss:.4f}" if h_loss > 0 else f"{_ANSI_DIM}—{_ANSI_RST}"
        print(
            f"    Trig:  {ps.get('trigger_steps', 0):>6,} steps  ε={ps.get('trigger_epsilon', 0.0):.3f}  "
            f"buf {t_bar}{t_pct}  loss {t_ls}  {t_rd}"
        )
        print(
            f"    Harv:  {ps.get('harvester_steps', 0):>6,} steps  β={ps.get('harvester_beta', 0.4):.3f}  "
            f"buf {h_bar}{h_pct}  loss {h_ls}  {h_rd}"
        )

    def _render_pipeline_card(self, sym: str, entry: dict) -> None:
        """Render one bot card with connection + training + activity stats."""
        ps = entry.get("_bot_stats", {})  # per-bot stats JSON from bot
        self._print_pipeline_header(sym, entry, ps)
        if ps:
            self._print_pipeline_connection(ps)
            self._print_pipeline_activity(ps)
            self._print_pipeline_account(ps)
            self._print_pipeline_training(ps)
        else:
            started = entry.get("paper_started_at", "")
            started_str = started[:19].replace("T", " ") if started else "—"
            print(f"    {_ANSI_DIM}Stats not yet available  (started {started_str}){_ANSI_RST}")
        print()

    def _render_training_summary(self, training_items: list[dict]) -> None:
        """Render the standard Training tab view without per-agent blocks."""
        if self.universe_stats:
            running_count = sum(1 for e in self.universe_stats.values() if e.get("_pid_alive"))
            total_count = len(self.universe_stats)
            _fleet_col = _ANSI_G if running_count == total_count else (_ANSI_Y if running_count else _ANSI_R)
            print(f"  \033[1m📈 FLEET\033[0m  {_fleet_col}{running_count}/{total_count} bots running{_ANSI_RST}")
            _hdr = f"  {'Bot':<13} {'Stage':<9} {'PID':>7} {'Trades':>7} {'PnL $':>11} {'Trig':>9} {'Harv':>9}"
            print(_hdr)
            print("  " + "─" * (_visible_width(_hdr) - 2))
            for _, entry in sorted(
                self.universe_stats.items(),
                key=lambda kv: (str(kv[1].get("symbol", "")), int(kv[1].get("timeframe_minutes", 0) or 0)),
            ):
                _sym = str(entry.get("symbol", "?")).upper()
                _tf = self._format_timeframe_minutes_label(entry.get("timeframe_minutes", 0))
                _bot = f"{_sym}/{_tf}"[:13]
                _stage = str(entry.get("stage", "?")).upper()[:9]
                _pid = entry.get("paper_pid") or "—"
                _pid_str = str(_pid)
                _alive_col = _ANSI_G if entry.get("_pid_alive") else _ANSI_R
                _ps = entry.get("_bot_stats", {}) if isinstance(entry.get("_bot_stats", {}), dict) else {}
                _trades = int(_ps.get("total_trades", 0) or 0)
                _pnl = float(_ps.get("total_pnl", 0.0) or 0.0)
                _trig = int(_ps.get("trigger_steps", 0) or 0)
                _harv = int(_ps.get("harvester_steps", 0) or 0)
                print(
                    f"  {_bot:<13} {_stage:<9} {_alive_col}{_pid_str:>7}{_ANSI_RST} "
                    f"{_trades:>7} {self._pnl_color(_pnl)}{_pnl:>+11.2f}{_ANSI_RST} "
                    f"{_trig:>9,} {_harv:>9,}"
                )
            print()

        if training_items:
            print(f"  \033[1m🧠 LEARNING BUFFERS\033[0m  {_ANSI_DIM}per symbol/timeframe; session counters{_ANSI_RST}")
            _hdr = f"  {'Bot':<13} {'TrigBuf':>9} {'Trigε':>7} {'HarvBuf':>9} {'Harvβ':>7} {'Position':>9}"
            print(_hdr)
            print("  " + "─" * (_visible_width(_hdr) - 2))
            for _item in training_items:
                _its = _item.get("stats", {})
                if not isinstance(_its, dict):
                    continue
                _tf = int(_item.get("timeframe_minutes") or self.active_tf_min or 0)
                _sym = str(_item.get("symbol") or self.active_sym or "?").upper()
                _bot = f"{_sym}/{self._format_timeframe_minutes_label(_tf)}"[:13]
                _tb = int(_its.get("trigger_buffer_size", 0) or 0)
                _hb = int(_its.get("harvester_buffer_size", 0) or 0)
                _eps = float(_its.get("trigger_epsilon", 0.0) or 0.0)
                _beta = float(_its.get("harvester_beta", 0.0) or 0.0)
                _pos_raw = _its.get("is_in_position")
                _pos = "IN_POS" if _pos_raw is True else ("FLAT" if _pos_raw is False else "UNKNOWN")
                _pos_col = _ANSI_Y if _pos == "IN_POS" else (_ANSI_G if _pos == "FLAT" else _ANSI_DIM)
                print(
                    f"  {_bot:<13} {_tb:>9,} {_eps:>7.3f} {_hb:>9,} {_beta:>7.3f} "
                    f"{_pos_col}{_pos:>9}{_ANSI_RST}"
                )
            print()

        print(f"  {_ANSI_DIM}Detail collapsed: [d] full trigger/harvester blocks and pipeline cards.{_ANSI_RST}")

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _agent_conf_color(confidence: float) -> str:
        if CONF_HEALTHY_LOW < confidence < CONF_HEALTHY_HIGH:
            return _ANSI_G
        if CONF_WARM_LOW < confidence <= CONF_HEALTHY_LOW:
            return _ANSI_Y
        return _ANSI_R

    @staticmethod
    def _agent_fill_note(is_in_pos: Any, *, fills_when_in_position: bool) -> str:
        if is_in_pos is None:
            return ""
        filling = is_in_pos if fills_when_in_position else not is_in_pos
        if filling:
            return f"  {_ANSI_G}⬆ filling — {'in position' if is_in_pos else 'bot flat'}{_ANSI_RST}"
        return f"  {_ANSI_Y}⏸ paused — {'bot in position' if is_in_pos else 'bot flat'}{_ANSI_RST}"

    def _print_agent_loss_history(self, loss: float, hist: deque[float]) -> None:
        loss_str = f"{loss:.6f}" if loss > 0 else f"{_ANSI_DIM}0.000000 (idle/no training event){_ANSI_RST}"
        print(f"    Loss:   {loss_str}")
        print(f"    Trend:  {self._rt_trend(hist)}")
        spark = self._rt_spark(hist)
        if spark:
            print(f"    Hist:   {spark}")

    def _print_agent_confidence(self, confidence: float) -> None:
        col = self._agent_conf_color(confidence)
        print(f"    Conf:   {col}{confidence:.3f}{_ANSI_RST}  {_ANSI_DIM}(healthy 0.55–0.85){_ANSI_RST}")

    @staticmethod
    def _print_agent_floor(label: str, floor: float | None, confidence: float, *, warn: float, alert: float) -> None:
        if floor is None:
            print(f"    RL min: {_ANSI_DIM}—{_ANSI_RST}  {_ANSI_DIM}(risk tuner pending){_ANSI_RST}")
            return
        floor_col = _ANSI_G if floor <= warn else (_ANSI_Y if floor <= alert else _ANSI_R)
        gap = float(confidence) - floor
        gap_col = _ANSI_G if gap >= 0 else _ANSI_R
        print(
            f"    RL min: {floor_col}{floor:.3f}{_ANSI_RST}  "
            f"{_ANSI_DIM}(dynamic {label} floor){_ANSI_RST}  Δnow {gap_col}{gap:+.3f}{_ANSI_RST}"
        )

    def _render_live_trigger_agent(self, ts: dict, pm: dict) -> None:
        """Render the Trigger Agent training block."""
        trig_buf = ts.get("trigger_buffer_size", 0)
        trig_added = ts.get("trigger_total_added", 0)
        trig_eps = ts.get("trigger_epsilon", 0.0)
        trig_ready = ts.get("trigger_ready", False)
        trig_steps = ts.get("trigger_training_steps", 0)
        trig_conf = ts.get("trigger_confidence", pm.get("trigger_confidence_avg", 0.5))
        ready = f"{_ANSI_G}✓ Ready{_ANSI_RST}" if trig_ready else f"{_ANSI_Y}⏳ Filling…{_ANSI_RST}"
        note = self._agent_fill_note(ts.get("is_in_position"), fills_when_in_position=False)
        print(f"  \033[1m🎯 TRIGGER AGENT  (Entry)\033[0m  {ready}  {_ANSI_DIM}fills when flat{_ANSI_RST}")
        print(f"    Steps:  {trig_steps:>10,}   Velocity: {self._rt_velocity(self._trig_step_hist)}")
        print(f"    Buffer: {self._rt_pct_bar(trig_buf, _RT_TRIG_CAP)}  {trig_buf:,}/{_RT_TRIG_CAP:,}{note}")
        if trig_added > 0:
            print(f"    Added:  {trig_added:,} total experiences")
        print(f"    ε:      {self._rt_eps_bar(trig_eps)}")
        self._print_trigger_exploration_controls(ts)
        self._print_agent_loss_history(ts.get("trigger_loss", 0.0), self._trig_loss_hist)
        self._print_agent_confidence(float(trig_conf))
        floor = self._float_or_none(ts.get("entry_conf_dynamic_floor"))
        self._print_agent_floor("entry", floor, float(trig_conf), warn=0.75, alert=0.85)
        self._print_trigger_runway_status(ts)
        print()

    def _print_trigger_exploration_controls(self, ts: dict) -> None:
        regime_f = ts.get("trigger_epsilon_regime_factor", 1.0)
        rf_col = _ANSI_G if regime_f >= 0.9 else (_ANSI_Y if regime_f >= 0.7 else _ANSI_B)
        print(
            f"    ε ζ:    {rf_col}{regime_f:.2f}{_ANSI_RST}  "
            f"{_ANSI_DIM}(decay factor — 1.0 normal, <1 slower){_ANSI_RST}"
        )
        trig_tau = ts.get("trigger_tau", 0.005)
        tau_col = _ANSI_G if trig_tau > 0.003 else (_ANSI_Y if trig_tau > 0.001 else _ANSI_R)
        print(f"    τ:      {tau_col}{trig_tau:.5f}{_ANSI_RST}  {_ANSI_DIM}(adaptive target sync){_ANSI_RST}")

    @staticmethod
    def _print_trigger_runway_status(ts: dict) -> None:
        total = int(ts.get("trigger_runway_cal_total_samples", 0) or 0)
        active = int(ts.get("trigger_runway_cal_active_buckets", 0) or 0)
        reliable = bool(ts.get("trigger_runway_predictor_reliable", False))
        col = _ANSI_G if reliable else _ANSI_Y
        label = "RELIABLE (gate active)" if reliable else "LEARNING (gate bypass)"
        print(f"    Runway: {col}{label}{_ANSI_RST}  {_ANSI_DIM}samples={total} active_buckets={active}{_ANSI_RST}")

    def _render_live_harvester_agent(self, ts: dict, pm: dict) -> None:
        """Render the Harvester Agent training block."""
        harv_buf = ts.get("harvester_buffer_size", 0)
        harv_added = ts.get("harvester_total_added", 0)
        harv_beta = ts.get("harvester_beta", 0.4)
        harv_ready = ts.get("harvester_ready", False)
        harv_steps = ts.get("harvester_training_steps", 0)
        harv_conf = ts.get("harvester_confidence", pm.get("harvester_confidence_avg", 0.5))
        ready = f"{_ANSI_G}✓ Ready{_ANSI_RST}" if harv_ready else f"{_ANSI_Y}⏳ Filling…{_ANSI_RST}"
        note = self._agent_fill_note(ts.get("is_in_position"), fills_when_in_position=True)
        print(f"  \033[1m🌾 HARVESTER AGENT  (Exit)\033[0m  {ready}  {_ANSI_DIM}fills in position{_ANSI_RST}")
        print(f"    Steps:  {harv_steps:>10,}   Velocity: {self._rt_velocity(self._harv_step_hist)}")
        print(f"    Buffer: {self._rt_pct_bar(harv_buf, _RT_HARV_CAP)}  {harv_buf:,}/{_RT_HARV_CAP:,}{note}")
        if harv_added > 0:
            print(f"    Added:  {harv_added:,} total experiences")
        print(f"    β IS:   {self._rt_beta_bar(harv_beta)}")
        self._print_harvester_tau(ts)
        self._print_agent_loss_history(ts.get("harvester_loss", 0.0), self._harv_loss_hist)
        self._print_agent_confidence(float(harv_conf))
        floor = self._float_or_none(ts.get("exit_conf_dynamic_floor"))
        self._print_agent_floor("exit", floor, float(harv_conf), warn=0.65, alert=0.80)
        self._print_harvester_exit_controls(ts)
        print()

    @staticmethod
    def _print_harvester_tau(ts: dict) -> None:
        harv_tau = ts.get("harvester_tau", 0.005)
        col = _ANSI_G if harv_tau > 0.003 else (_ANSI_Y if harv_tau > 0.001 else _ANSI_R)
        print(f"    τ:      {col}{harv_tau:.5f}{_ANSI_RST}  {_ANSI_DIM}(adaptive target sync){_ANSI_RST}")

    @staticmethod
    def _print_harvester_exit_controls(ts: dict) -> None:
        hold_mult = ts.get("harvester_regime_hold_mult", 1.0)
        hold_col = _ANSI_G if hold_mult > 1.0 else (_ANSI_Y if hold_mult >= 0.9 else _ANSI_R)
        min_hold = ts.get("harvester_min_hold_ticks", 10)
        print(
            f"    Hold:   {min_hold} ticks min  {hold_col}×{hold_mult:.2f}{_ANSI_RST}  "
            f"{_ANSI_DIM}(regime mult — >1 trend run, <1 quick exit){_ANSI_RST}"
        )
        cd = float(ts.get("harvester_capture_decay_threshold", 0.0) or 0.0)
        mw = float(ts.get("harvester_micro_winner_giveback_pct", 0.0) or 0.0)
        print(f"    WTL:    {_ANSI_Y}capture_decay<{cd:.2f}  micro_giveback>{mw:.2f}×MFE{_ANSI_RST}")

    def _render_live_arena_and_health(
        self,
        ts: dict,
    ) -> None:
        """Render Arena + Learning Health blocks."""
        trig_ready = ts.get("trigger_ready", False)
        harv_ready = ts.get("harvester_ready", False)
        trig_steps = ts.get("trigger_training_steps", 0)
        harv_steps = ts.get("harvester_training_steps", 0)
        total_agents = ts.get("total_agents", 0)
        if total_agents > 0:
            diversity = ts.get("arena_diversity", {})
            trig_div = diversity.get("trigger_diversity", 0) if isinstance(diversity, dict) else 0
            harv_div = diversity.get("harvester_diversity", 0) if isinstance(diversity, dict) else 0
            agreement = ts.get("last_agreement_score", 0)
            consensus = ts.get("consensus_mode", "unknown")
            print("  \033[1m🤖 ARENA\033[0m")
            print(f"    Agents: {total_agents}   Consensus: {consensus}   Agreement: {agreement:.3f}")
            print(f"    Diversity — Trig: {trig_div:.3f}   Harv: {harv_div:.3f}")
            print()
        last_train = ts.get("last_training_time", "Never")
        train_on = self.bot_config.get("training_enabled", False)
        train_str = f"{_ANSI_G}ON{_ANSI_RST}" if train_on else f"{_ANSI_R}OFF{_ANSI_RST}"
        trig_ok = f"{_ANSI_G}✓{_ANSI_RST}" if trig_ready else f"{_ANSI_Y}⏳{_ANSI_RST}"
        harv_ok = f"{_ANSI_G}✓{_ANSI_RST}" if harv_ready else f"{_ANSI_Y}⏳{_ANSI_RST}"
        print("  \033[1m📊 LEARNING HEALTH\033[0m")
        print(f"    Training: {train_str}   Last event: {last_train}")
        print(f"    Trigger {trig_ok}  Harvester {harv_ok}   Total steps: {trig_steps + harv_steps:,}")
        print()

    @staticmethod
    def _offline_stats_stale(ofs: dict) -> bool:
        if not ofs.get("completed_at"):
            return False
        try:
            completed_at = datetime.fromisoformat(ofs["completed_at"])
            if completed_at.tzinfo is None:
                completed_at = completed_at.replace(tzinfo=UTC)
            return (datetime.now(UTC) - completed_at).total_seconds() > 86400
        except Exception:
            return False

    def _render_training_offline_status(self, *, detail: bool) -> None:
        ofs = self.offline_stats
        if not ofs:
            return
        if self._offline_status_normalized(ofs) == "complete" and self._offline_stats_stale(ofs):
            self.offline_stats = {}
            with contextlib.suppress(Exception):
                (self.data_dir / "offline_training_status.json").unlink(missing_ok=True)
            return
        self._render_offline_training(ofs, detail=detail)

    def _training_items_for_context(self) -> list[dict]:
        items = [item for item in self.training_stats_all if isinstance(item.get("stats"), dict)]
        if self._ctx_level < 2 or not self._ctx_symbol:
            return items
        sym_filter = self._ctx_symbol.upper()
        tf_filter = self._ctx_tf if self._ctx_level >= 3 and self._ctx_tf else 0
        filtered = [
            item
            for item in items
            if str(item.get("symbol", "")).upper() == sym_filter
            and (not tf_filter or int(item.get("timeframe_minutes", 0) or 0) == tf_filter)
        ]
        return filtered or items

    def _training_fallback_items(self, ts: dict, mode_label: str) -> list[dict] | None:
        if any(value for value in ts.values() if value):
            return [
                {
                    "symbol": self.active_sym,
                    "timeframe_minutes": self.active_tf_min,
                    "stats": ts,
                }
            ]
        print(f"\033[1m🤖 {mode_label} BOT TRAINING\033[0m  {_ANSI_DIM}(no live bot running){_ANSI_RST}\n")
        return None

    @staticmethod
    def _training_item_scope(item: dict, active_sym: str, active_tf_min: int) -> tuple[str, str, int]:
        tf_value = int(item.get("timeframe_minutes") or active_tf_min or 0)
        sym_value = str(item.get("symbol") or active_sym or "").upper()
        if sym_value and tf_value:
            return f"{sym_value} M{tf_value}", sym_value, tf_value
        if tf_value:
            return f"M{tf_value}", sym_value, tf_value
        return "BOT", sym_value, tf_value

    def _render_training_detail_items(self, items: list[dict], mode_label: str, pm: dict) -> None:
        if self.universe_stats:
            self._render_trading_pipeline()
        for idx, item in enumerate(items):
            item_stats = item.get("stats", {})
            scope, sym, tf_value = self._training_item_scope(item, self.active_sym, self.active_tf_min)
            item_pm_payload = self._load_bot_production_metrics(sym, tf_value)
            item_pm = item_pm_payload.get("metrics", pm) if isinstance(item_pm_payload, dict) else pm
            train_label = f"{mode_label} {scope} TRAINING" if scope != "BOT" else f"{mode_label} BOT TRAINING"
            print(f"\033[1m🤖 {train_label}\033[0m\n")
            self._render_live_trigger_agent(item_stats, item_pm)
            self._render_live_harvester_agent(item_stats, item_pm)
            self._render_live_arena_and_health(item_stats)
            if idx < len(items) - 1:
                print("  " + "─" * (self._term_width() - 4))
                print()

    @staticmethod
    def _format_duration_hint(seconds: float) -> str:
        if seconds < 0:
            return "bar closing…"
        if seconds < 60:
            return f"{int(seconds)}s"
        if seconds < 3600:
            mins, secs = divmod(int(seconds), 60)
            return f"{mins}m {secs:02d}s"
        if seconds < 86400:
            hours, rem = divmod(int(seconds), 3600)
            return f"{hours}h {rem // 60:02d}m"
        days = int(seconds) // 86400
        hours = (int(seconds) % 86400) // 3600
        return f"{days}d {hours}h"

    @staticmethod
    def _format_timeframe_hint(tf_min: int) -> str:
        if tf_min >= 1440:
            return f"{tf_min // 1440}d"
        if tf_min >= 60:
            return f"{tf_min // 60}h"
        return f"{tf_min}m"

    def _next_training_bar_close_raw(self) -> Any:
        nbc_raw = self.market_stats.get("next_bar_close_utc")
        if not nbc_raw and self.active_sym and self.active_tf_min:
            nbc_raw = self._load_bot_stats(self.active_sym, self.active_tf_min).get("next_bar_close_utc")
        return nbc_raw or self.bot_config.get("next_bar_close_utc")

    def _render_training_next_update_hint(self) -> None:
        nbc_raw = self._next_training_bar_close_raw()
        tf_min = (
            self.market_stats.get("timeframe_minutes")
            or self.bot_config.get("timeframe_minutes")
            or self.active_tf_min
        )
        if nbc_raw:
            try:
                nbc_dt = datetime.fromisoformat(nbc_raw)
                if nbc_dt.tzinfo is None:
                    nbc_dt = nbc_dt.replace(tzinfo=UTC)
                hint = self._format_duration_hint((nbc_dt - datetime.now(UTC)).total_seconds())
                print(f"  {_ANSI_DIM}ℹ️  Training stats update on bar close — next in {hint}{_ANSI_RST}")
            except Exception:
                pass
        elif tf_min:
            label = self._format_timeframe_hint(int(tf_min))
            print(f"  {_ANSI_DIM}ℹ️  Training stats update every {label} bar close (awaiting first tick){_ANSI_RST}")

    def _render_training(self) -> None:
        """Render agent training status — dispatches by drill level."""
        self._render_breadcrumb("TRAINING", 3)
        level_detail = self._ctx_level >= 3 or self._training_detail
        ts = self.training_stats
        print(f"  {_ANSI_DIM}(canonical source: training_stats_*.json; per-bot file preferred){_ANSI_RST}\n")
        pm = self.production_metrics.get("metrics", {})
        self._render_training_offline_status(detail=level_detail)
        mode = self.bot_config.get("trading_mode", "paper")
        mode_label = "PAPER" if mode == "paper" else ("LIVE" if mode == "live" else "OFFLINE")

        training_items = self._training_items_for_context()
        if not training_items:
            training_items = self._training_fallback_items(ts, mode_label)
            if training_items is None:
                return

        if not level_detail:
            self._render_training_summary(training_items)
            return

        self._render_training_detail_items(training_items, mode_label, pm)
        self._render_training_next_update_hint()

    @staticmethod
    def _print_centered_box(text: str, inner: int, *, prefix: str = "", suffix: str = "") -> None:
        pad_total = max(0, inner - len(text))
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        print(f"{prefix}╔" + "═" * inner + f"╗{suffix}")
        print(f"{prefix}║" + " " * pad_left + text + " " * pad_right + f"║{suffix}")
        print(f"{prefix}╚" + "═" * inner + f"╝{suffix}")

    def _render_header_banner(self, inner: int) -> None:
        cb_active = self.risk_stats.get("circuit_breaker", "INACTIVE") == "ACTIVE"
        if cb_active:
            self._print_centered_box(
                "⚠️  CIRCUIT BREAKER ACTIVE - TRADING HALTED ⚠️",
                inner,
                prefix="\033[41;97m",
                suffix="\033[0m",
            )
            return
        if self.risk_stats.get("kurtosis_gate_active", False):
            is_live_mode = getattr(self, "_perf_snapshot_mode", "") == "live"
            kurt_note = "entries BLOCKED" if is_live_mode else "entries bypassed in paper mode"
            self._print_centered_box(
                f"⚡ KURTOSIS GATE ACTIVE — {kurt_note}",
                inner,
                prefix="\033[43;30m",
                suffix="\033[0m",
            )
            return
        self._print_centered_box("ADAPTIVE RL TRADING BOT - TABBED HUD", inner)

    def _active_header_stats(self) -> tuple[str, str, int]:
        active_stats = (
            self._load_bot_stats(self.active_sym, self.active_tf_min)
            if self.active_sym and self.active_tf_min
            else {}
        )
        symbol = active_stats.get("symbol") or self.bot_config.get("symbol", "UNKNOWN")
        tf_min = active_stats.get("timeframe_minutes") or self.bot_config.get("timeframe_minutes")
        uptime = active_stats.get("uptime_seconds") or self.bot_config.get("uptime_seconds", 0)
        return str(symbol), self._format_timeframe_minutes_label(tf_min), int(uptime)

    def _header_price_str(self) -> str:
        price = self.position.get("current_price", 0)
        if not price or self.position.get("direction", "FLAT") == "FLAT":
            return "—"
        return f"{price:.{self._price_decimals(price)}f}"

    def _header_next_bar_str(self) -> str:
        nbc_raw = self.market_stats.get("next_bar_close_utc") or self.bot_config.get("next_bar_close_utc")
        if not nbc_raw:
            return ""
        try:
            nbc_dt = datetime.fromisoformat(nbc_raw)
            if nbc_dt.tzinfo is None:
                nbc_dt = nbc_dt.replace(tzinfo=UTC)
            return f"  📊 next bar {self._format_duration_hint((nbc_dt - datetime.now(UTC)).total_seconds())}"
        except Exception:
            return ""

    def _header_mode_badge(self) -> str:
        mode = self.bot_config.get("trading_mode", "paper")
        if mode == "live":
            return f"{_ANSI_G}● LIVE{_ANSI_RST}"
        if mode == "paper":
            return f"{_ANSI_Y}● PAPER{_ANSI_RST}"
        return f"{_ANSI_DIM}● OFFLINE{_ANSI_RST}"

    def _header_fleet_chunks(self) -> list[str]:
        fleet: dict[str, list[int]] = {}
        for bot in self.all_bots_stats:
            sym = str(bot.get("symbol", "") or "").upper()
            try:
                tfm = int(bot.get("timeframe_minutes", 0) or 0)
            except (TypeError, ValueError):
                tfm = 0
            if sym and tfm > 0:
                fleet.setdefault(sym, []).append(tfm)
        chunks: list[str] = []
        for sym in sorted(fleet):
            tf_labels = ",".join(self._format_timeframe_minutes_label(value) for value in sorted(set(fleet[sym])))
            chunks.append(f"{sym}[{tf_labels}]")
        return chunks

    def _render_header(self) -> None:
        """Render header."""
        heartbeat = self.heartbeat_chars[self.heartbeat_idx]
        self._render_header_banner(self._term_width() - 2)
        symbol, tf, uptime = self._active_header_stats()
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        now = self.last_update or datetime.now(UTC)
        print(
            f"\n🎯 {symbol} @ {tf}  {self._header_mode_badge()}    💲 {self._header_price_str()}    "
            f"⏱  {hours:02d}h {minutes:02d}m{self._header_next_bar_str()}"
        )
        fleet_chunks = self._header_fleet_chunks()
        if fleet_chunks:
            print(f"{_ANSI_DIM}🧭 Active paper TFs: {' | '.join(fleet_chunks)}{_ANSI_RST}")
        print(f"{heartbeat} {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    def _render_tab_bar(self) -> None:
        """Render tab navigation bar — adapts to terminal width."""
        W = self._term_width()
        print("\n" + "─" * W)

        # Pick label set based on available width.
        # Full (emoji) ~104 cols, medium (no emoji, full names) ~85 cols,
        # short (abbreviated) ~66 cols.
        if W >= 104:
            labels = self.TAB_DISPLAY
        elif W >= 85:
            labels = self.TAB_DISPLAY_MEDIUM
        else:
            labels = self.TAB_DISPLAY_SHORT

        tabs = []
        visible_col = 1
        self._tab_click_ranges = []
        for key, tab_id in self.TABS.items():
            name = labels.get(tab_id, tab_id.title())
            visible = f" [{key}] {name} "
            start = visible_col
            end = visible_col + _visible_width(visible) - 1
            self._tab_click_ranges.append((start, end, tab_id))
            if tab_id == self.current_tab:
                tabs.append(f"\033[7m{visible}\033[0m")  # Inverted
            else:
                tabs.append(visible)
            visible_col = end + 1

        print("".join(tabs))
        print("─" * W)

    # ── Drill-down navigation ──────────────────────────────────────────

    def _drill_scope_label(self) -> str:
        """Compact scope label for breadcrumb: 'Paper' / 'XAUUSD' / 'XAUUSD/M5'."""
        if self._ctx_level == 0:
            return self._ctx_mode.upper()
        if self._ctx_level == 1:
            return "Portfolio"
        if self._ctx_level == 2:
            return (self._ctx_symbol or "?").upper()
        _sym = (self._ctx_symbol or "?").upper()
        _tf = self._format_timeframe_minutes_label(self._ctx_tf)
        return f"{_sym}/{_tf}"

    def _render_breadcrumb(self, tab_label: str, tab_num: int) -> None:
        """Render compact breadcrumb line showing scope + period + available keys."""
        _mode_map = {"live": "🔴 LIVE", "paper": "🟡 PAPER", "offline": "🔵 OFFLINE"}
        _parts: list[str] = []
        if self._ctx_level >= 0:
            _parts.append(_mode_map.get(self._ctx_mode, self._ctx_mode.upper()))
        if self._ctx_level >= 1:
            _parts.append("Portfolio")
        if self._ctx_level >= 2 and self._ctx_symbol:
            _parts.append(self._ctx_symbol.upper())
        if self._ctx_level >= 3 and self._ctx_tf > 0:
            _parts.append(self._format_timeframe_minutes_label(self._ctx_tf))
        if self._ctx_level >= 4:
            _parts.append(self._ctx_period)
        _path = f"[{tab_num}] {tab_label}  ›  " + " › ".join(_parts)

        _keys = []
        if self._ctx_level > 0:
            _keys.append("Esc back")
        if self._ctx_level < 4:
            _keys.append("Enter drill")
        if self._ctx_level >= 1:
            _keys.append("s scope")
        if self._ctx_level >= 3:
            _keys.append("p period")
        if self._ctx_level >= 2:
            _keys.append("d detail")
        _key_hints = "  ".join(f"[{k}]" for k in _keys)

        W = self._term_width()
        _left = f"{_path}"
        _right = _key_hints
        _pad = max(1, W - _visible_width(_left) - _visible_width(_right) - 2)
        print(f"\n\033[1m{_left}\033[0m{_ANSI_DIM}{' ' * _pad}{_right}{_ANSI_RST}")

    def _drill_up(self) -> None:
        """Move up one drill level, clearing child scope. Works for ALL tabs."""
        if self._ctx_level <= 0:
            return
        # Clear tab-detail state if at Level 4
        if self._ctx_level == 4:
            self._trades_detail = False
            self._trades_detail_trade = {}
            self._dec_log_detail = False
            self._dec_log_detail_entry = {}
        self._ctx_level -= 1
        self._ctx_cursor = 0
        if self._ctx_level < 3:
            self._ctx_tf = 0
        if self._ctx_level < 2:
            self._ctx_symbol = ""
        if self._ctx_level < 1:
            self._ctx_mode = "paper"
        self._sync_ctx_to_legacy()
        self._force_redraw = True

    def _complete_drill_down(self, level: int) -> None:
        self._ctx_level = level
        self._ctx_cursor = 0
        self._sync_ctx_to_legacy()
        self._force_redraw = True

    def _drill_to_portfolio(self) -> None:
        modes = ["live", "paper", "offline"]
        self._ctx_mode = modes[min(self._ctx_cursor, len(modes) - 1)]
        self._complete_drill_down(1)

    def _drill_to_symbol(self) -> None:
        rows = self._l1_rows()
        if not rows:
            return
        self._ctx_symbol, self._ctx_mode = rows[min(self._ctx_cursor, len(rows) - 1)]
        self._complete_drill_down(2)

    def _drill_to_timeframe(self) -> None:
        timeframes = self._available_timeframes()
        if not timeframes:
            return
        self._ctx_tf = timeframes[min(self._ctx_cursor, len(timeframes) - 1)]
        self._trades_page = 0
        self._complete_drill_down(3)

    def _drill_to_trade_detail(self) -> None:
        idx = self._trades_page * self._trades_per_page + self._trades_cursor
        if 0 <= idx < len(self._trades_view):
            self._trades_detail_trade = self._trades_view[idx]
            self._trades_detail = True
            self._ctx_level = 4
            self._sync_ctx_to_legacy()
            self._force_redraw = True

    def _drill_to_decision_detail(self) -> None:
        if not self._dec_log_view:
            return
        idx = min(self._dec_log_cursor, len(self._dec_log_view) - 1)
        self._dec_log_detail_entry = self._dec_log_view[idx]
        self._dec_log_detail = True
        self._ctx_level = 4
        self._sync_ctx_to_legacy()
        self._force_redraw = True

    def _drill_to_period_detail(self) -> None:
        self._ctx_period = self._ctx_periods[min(self._ctx_cursor, len(self._ctx_periods) - 1)]
        self._ctx_level = 4
        self._sync_ctx_to_legacy()
        self._force_redraw = True

    def _drill_from_symbol_timeframe(self) -> None:
        if self.current_tab == "trades":
            self._drill_to_trade_detail()
        elif self.current_tab == "log":
            self._drill_to_decision_detail()
        else:
            self._drill_to_period_detail()

    def _drill_down(self) -> None:
        """Move down one drill level. Uses cursor to select row. Works for ALL tabs."""
        if self._ctx_level >= 4:
            return
        if self._ctx_level == 0:
            self._drill_to_portfolio()
            return
        if self._ctx_level == 1:
            self._drill_to_symbol()
            return
        if self._ctx_level == 2:
            self._drill_to_timeframe()
            return
        if self._ctx_level == 3:
            self._drill_from_symbol_timeframe()

    def _available_symbols(self) -> list[str]:
        """Return sorted list of unique symbols across all trade_log trades."""
        return sorted({
            str(t.get("symbol", "")).upper()
            for t in self._trade_log_all_trades if t.get("symbol")
        })

    def _l1_rows(self) -> list[tuple[str, str]]:
        """Return (symbol, mode) pairs in L1 render order: Live first, then Paper."""
        _syms = self._available_symbols()
        _live_syms = [s for s in _syms if any(
            str(t.get("symbol", "")).upper() == s and t.get("trading_mode") == "live"
            for t in self._trade_log_all_trades
        )]
        _paper_syms = [s for s in _syms if any(
            str(t.get("symbol", "")).upper() == s and t.get("trading_mode") == "paper"
            for t in self._trade_log_all_trades
        )]
        rows: list[tuple[str, str]] = []
        for s in _live_syms:
            rows.append((s, "live"))
        for s in _paper_syms:
            rows.append((s, "paper"))
        return rows

    @staticmethod
    def _tail_meaningful(path: "Path", n: int = 50) -> list[dict]:
        """Read last N meaningful (non-CACHED) JSONL entries from a decisions file.

        Scans backward so 90%+ CACHED startup entries don't crowd out real decisions.
        """
        _SKIP = frozenset({"CACHED", "WARMING_UP", "FLAT_SKIP", "NO_ENTRY_SKIP"})
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        results: list[dict] = []
        for raw in reversed(lines):
            line = raw.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("decision") in _SKIP:
                continue
            results.append(e)
            if len(results) >= n:
                break
        return list(reversed(results))

    def _available_timeframes(self) -> list[int]:
        """Return sorted list of unique TFs for the current symbol."""
        _sym = self._ctx_symbol.upper()
        return sorted({
            int(t.get("timeframe_minutes", 0))
            for t in self._trade_log_all_trades
            if str(t.get("symbol", "")).upper() == _sym and t.get("timeframe_minutes")
        })

    def _cycle_scope(self) -> None:
        """Jump-scope: Portfolio → Instrument → Instrument/TF → Portfolio."""
        if self._ctx_level <= 1:
            _syms = self._available_symbols()
            if _syms:
                self._ctx_symbol = _syms[0]
                self._ctx_level = 2
                self._ctx_cursor = 0
                self._sync_ctx_to_legacy()
                self._force_redraw = True
        elif self._ctx_level == 2:
            _tfs = self._available_timeframes()
            if _tfs:
                self._ctx_tf = _tfs[0]
                self._ctx_level = 3
                self._ctx_cursor = 0
                self._sync_ctx_to_legacy()
                self._force_redraw = True
        else:
            self._ctx_level = 1
            self._ctx_symbol = ""
            self._ctx_tf = 0
            self._ctx_cursor = 0
            self._sync_ctx_to_legacy()
            self._force_redraw = True

    def _cycle_period(self) -> None:
        """Cycle through period filters: 24h → 7d → Month → Epoch → Lifetime."""
        if self._ctx_period not in self._ctx_periods:
            self._ctx_period = self._ctx_periods[0]
        else:
            _idx = self._ctx_periods.index(self._ctx_period)
            self._ctx_period = self._ctx_periods[(_idx + 1) % len(self._ctx_periods)]
        self._sync_ctx_to_legacy()
        self._force_redraw = True

    def _position_mode_tag(self) -> str:
        mode = self.bot_config.get("trading_mode", "paper")
        if mode == "paper":
            return f"  {_ANSI_Y}(paper){_ANSI_RST}"
        if mode == "live":
            return f"  {_ANSI_G}(live){_ANSI_RST}"
        return ""

    def _open_position_rows(self) -> list[tuple[str, dict]]:
        rows: list[tuple[str, dict]] = []
        for bot in self.all_bots_stats or []:
            pos = bot.get("_position", {}) if isinstance(bot, dict) else {}
            if not isinstance(pos, dict) or str(pos.get("direction", "FLAT")).upper() == "FLAT":
                continue
            sym = str(bot.get("symbol") or pos.get("symbol") or "?").upper()
            try:
                tf = int(bot.get("timeframe_minutes") or pos.get("timeframe_minutes") or 0)
            except (TypeError, ValueError):
                tf = 0
            label = f"{sym}/{self._format_timeframe_minutes_label(tf)}" if tf > 0 else sym
            rows.append((label, pos))
        if not rows and str(self.position.get("direction", "FLAT")).upper() != "FLAT":
            sym = str(self.position.get("symbol") or self.active_sym or "?").upper()
            tf = int(self.position.get("timeframe_minutes") or self.active_tf_min or 0)
            label = f"{sym}/{self._format_timeframe_minutes_label(tf)}" if tf > 0 else sym
            rows.append((label, self.position))
        return rows

    def _render_multi_position_block(self, open_positions: list[tuple[str, dict]]) -> bool:
        if len(open_positions) <= 1:
            return False
        total_unreal = sum(float(pos.get("unrealized_pnl", 0.0) or 0.0) for _, pos in open_positions)
        print(
            f"  {_ANSI_B}{len(open_positions)} open positions{_ANSI_RST}  |  "
            f"Unrealized: {self._pnl_color(total_unreal)}{total_unreal:+.2f}{_ANSI_RST}"
        )
        for label, pos in sorted(open_positions, key=lambda kv: (kv[0], str(kv[1].get("position_id", ""))))[:8]:
            self._render_open_position_row(label, pos)
        if len(open_positions) > 8:
            print(f"  {_ANSI_DIM}… {len(open_positions) - 8} more open positions{_ANSI_RST}")
        return True

    def _render_open_position_row(self, label: str, pos: dict) -> None:
        direction = str(pos.get("direction", "FLAT")).upper()
        direction_col = _ANSI_G if direction == "LONG" else (_ANSI_R if direction == "SHORT" else _ANSI_DIM)
        entry = float(pos.get("entry_price", 0.0) or 0.0)
        current = float(pos.get("current_price", 0.0) or 0.0)
        pnl = float(pos.get("unrealized_pnl", 0.0) or 0.0)
        ticks = pos.get("ticks_held", pos.get("bars_held", 0))
        decimals = self._price_decimals(max(entry, current, 0.0))
        print(
            f"  {label:<13} {direction_col}{direction:<5}{_ANSI_RST} "
            f"{entry:.{decimals}f} → {current:.{decimals}f}  "
            f"{self._pnl_color(pnl)}{pnl:+.2f}{_ANSI_RST}  ticks:{ticks}"
        )

    def _position_direction_color(self, direction: str) -> str:
        if direction == "LONG":
            return _ANSI_G
        if direction == "SHORT":
            return _ANSI_R
        return _ANSI_Y

    def _render_single_position_line(self) -> str:
        direction = self.position.get("direction", "FLAT")
        entry = self.position.get("entry_price", 0)
        current = self.position.get("current_price", 0)
        pnl = self.position.get("unrealized_pnl", 0)
        ticks = self.position.get("ticks_held", self.position.get("bars_held", 0))
        dir_color = self._position_direction_color(direction)
        if direction == "FLAT":
            return f"  {dir_color}FLAT{_ANSI_RST}  (no open position)"
        decimals = self._price_decimals(max(entry, current, 0.0))
        return (
            f"  {dir_color}{direction}{_ANSI_RST} @ {entry:.{decimals}f} → {current:.{decimals}f}  |  "
            f"PnL: {self._pnl_color(pnl)}{pnl:+.2f}{_ANSI_RST}  |  Ticks: {ticks}"
        )

    def _render_position_excursions(self, direction: str) -> None:
        if direction == "FLAT":
            print()
            return
        mfe = self.position.get("mfe", 0.0)
        mae = self.position.get("mae", 0.0)
        mfe_color = _ANSI_G if mfe > 0 else _ANSI_Y
        mae_color = _ANSI_R if mae > 0 else _ANSI_Y
        print(f"  MFE: {mfe_color}+{mfe:.2f}{_ANSI_RST}  |  MAE: {mae_color}-{mae:.2f}{_ANSI_RST}  (USD, excl. spread)")

    def _render_position_tracker(self, direction: str) -> None:
        pid = self.position.get("position_id", "") if direction != "FLAT" else ""
        tracker_key = self.position.get("tracker_key", "") if direction != "FLAT" else ""
        if pid:
            print(f"  {_ANSI_DIM}PID: {pid}  tracker: {tracker_key}{_ANSI_RST}")
        else:
            print()

    def _render_position_block(self) -> None:
        """Render the position header block (always fixed height to avoid layout jumps)."""
        print(f"\n\033[1m📊 POSITION\033[0m{self._position_mode_tag()}")
        if self._render_multi_position_block(self._open_position_rows()):
            return
        direction = self.position.get("direction", "FLAT")
        print(self._render_single_position_line())
        self._render_position_excursions(direction)
        self._render_position_tracker(direction)

    @staticmethod
    def _sorted_all_bots(bots: list[dict]) -> list[dict]:
        preferred_tf = {1: 0, 5: 1, 15: 2, 30: 3, 60: 4, 240: 5}
        return sorted(
            bots,
            key=lambda b: (
                str(b.get("symbol", "")).upper(),
                preferred_tf.get(int(b.get("timeframe_minutes", 0) or 0), 999),
                int(b.get("timeframe_minutes", 0) or 0),
            ),
        )

    @staticmethod
    def _all_bots_header() -> str:
        return (
            f"  {'Bot':<13}  {'Runtime':<7}  {'Bars':>4}  {'Position':<22}"
            f"  {'T-buf':>5}  {'H-buf':>5}  {'SessTrd':>7}  {'SessPnL':>11}  {'SessWin':>7}"
        )

    def _bot_universe_entry(self, bot: dict) -> dict | None:
        sym = str(bot.get("symbol", "")).upper()
        tf = int(bot.get("timeframe_minutes", 0) or 0)
        return next(
            (
                entry
                for entry in self.universe_stats.values()
                if isinstance(entry, dict)
                and str(entry.get("symbol", "")).upper() == sym
                and int(entry.get("timeframe_minutes", 0) or 0) == tf
            ),
            None,
        )

    @staticmethod
    def _bot_status_thresholds(tf_min: int) -> tuple[float, float]:
        live_age = max(120.0, min(float(tf_min * 30), 3600.0)) if tf_min > 0 else 120.0
        slow_age = max(180.0, min(float(tf_min * 120), 7200.0)) if tf_min > 0 else 180.0
        return live_age, slow_age

    @staticmethod
    def _bot_awaiting_bar(bot: dict, tf_min: int, now: datetime) -> bool:
        nbc_raw = bot.get("next_bar_close_utc")
        if not nbc_raw or tf_min <= 0:
            return False
        try:
            nbc_dt = datetime.fromisoformat(nbc_raw)
            if nbc_dt.tzinfo is None:
                nbc_dt = nbc_dt.replace(tzinfo=UTC)
            return 0.0 <= (nbc_dt - now).total_seconds() <= (tf_min * 60 + 180)
        except Exception:
            return False

    def _bot_runtime_status(self, bot: dict, now: datetime) -> str:
        try:
            updated_at = datetime.fromisoformat(bot.get("updated_at", ""))
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            age = (now - updated_at).total_seconds()
        except Exception:
            age = 9999.0
        conn = bot.get("connection_healthy", False) and bot.get("quote_ok", False)
        entry = self._bot_universe_entry(bot)
        pid_alive = bool(entry.get("_pid_alive", True)) if isinstance(entry, dict) else True
        tf_min = int(bot.get("timeframe_minutes", 0) or 0)
        live_age, slow_age = self._bot_status_thresholds(tf_min)
        awaiting_bar = self._bot_awaiting_bar(bot, tf_min, now)
        if not pid_alive or not conn:
            status_vis, status_col = "● DOWN ", _ANSI_R
        elif age <= live_age:
            status_vis, status_col = "● RUN  ", _ANSI_G
        elif age <= slow_age or awaiting_bar:
            status_vis, status_col = "● SLOW ", _ANSI_Y
        else:
            status_vis, status_col = "● STALE", _ANSI_R
        return f"{status_col}{status_vis:<7}{_ANSI_RST}"

    def _bot_position_cell(self, bot: dict) -> tuple[str, str]:
        pos = bot.get("_position", {})
        direction = (pos.get("direction") or "FLAT").upper()
        entry_px = pos.get("entry_price", 0.0)
        unreal = pos.get("unrealized_pnl", 0.0)
        if direction == "FLAT":
            return "FLAT", f"{_ANSI_DIM}FLAT{_ANSI_RST}"
        visible = f"{direction:<5}@{entry_px:.0f}({unreal:+.0f})"
        dir_color = _ANSI_G if direction == "LONG" else _ANSI_R
        colored = (
            f"{dir_color}{direction:<5}{_ANSI_RST}"
            f"@{entry_px:.0f}"
            f"({self._pnl_color(unreal)}{unreal:+.0f}{_ANSI_RST})"
        )
        return visible, colored

    def _render_all_bots_row(self, bot: dict, now: datetime) -> None:
        sym = bot.get("symbol", "?")
        tf = bot.get("timeframe_minutes", 0)
        label = f"{sym}/{self._format_timeframe_minutes_label(tf)}"
        label_cell = label if len(label) <= 13 else label[:12] + "…"
        visible_pos, colored_pos = self._bot_position_cell(bot)
        pos_pad = " " * max(0, 22 - len(visible_pos))
        trades = int(bot.get("total_trades", 0) or 0)
        pnl = float(bot.get("total_pnl", 0.0) or 0.0)
        wr = float(bot.get("win_rate", 0.0) or 0.0) * 100
        wr_str = f"{wr:>5.1f}%" if trades > 0 else "      -"
        print(
            f"  {label_cell:<13}  {self._bot_runtime_status(bot, now)}  {bot.get('bar_count', 0):>4}  "
            f"{colored_pos}{pos_pad}  {bot.get('trigger_buffer', 0):>5}  {bot.get('harvester_buffer', 0):>5}  "
            f"{trades:>7}  {self._pnl_color(pnl)}{pnl:>+11.2f}{_ANSI_RST}  {wr_str:>7}"
        )

    def _render_all_bots_panel(self) -> None:
        """Render a compact one-row-per-bot fleet summary."""
        bots = self.all_bots_stats
        if not bots:
            print(f"\n\033[1m🤖 ALL BOTS\033[0m  {_ANSI_DIM}No bots currently running{_ANSI_RST}")
            return
        bots = self._sorted_all_bots(bots)
        print(
            f"\n\033[1m🤖 ALL BOTS\033[0m  {_ANSI_DIM}session counters; runtime state, not trading mode{_ANSI_RST}"
        )
        header = self._all_bots_header()
        print(f"\033[2m{header}\033[0m")
        print("  " + "─" * (_visible_width(header) - 2))
        now = datetime.now(UTC)
        for bot in bots:
            self._render_all_bots_row(bot, now)

    def _render_mode_selector(self) -> None:
        """Level 0: Trading mode selection screen — shown when ctx_level == 0."""
        print("\n\033[1m🎯 SELECT TRADING MODE\033[0m\n")
        print(f"  {_ANSI_DIM}Choose the trading context for all tabs. All views will filter to this mode.{_ANSI_RST}\n")
        _modes = [
            ("live", "🔴 LIVE TRADING", "Real-money execution. Production account.", _ANSI_G),
            ("paper", "🟡 PAPER TRADING", "Simulated execution. Practice & validation.", _ANSI_Y),
            ("offline", "🔵 OFFLINE", "Backtest, training & validation. NOT account PnL.", _ANSI_B),
        ]
        _starting = self._universe_starting_equity()
        for _idx, (_mk, _label, _desc, _color) in enumerate(_modes):
            _sel = "\033[7m > " if _idx == self._ctx_cursor else "   "
            _end = "\033[0m" if _idx == self._ctx_cursor else ""
            _trades = self._trades_for_mode(_mk) if _mk != "offline" else []
            _n = len(_trades)
            _pnl = sum(float(t.get("pnl", 0) or 0) for t in _trades) if _trades else 0.0
            _pc = self._pnl_color(_pnl) if _n > 0 else _ANSI_DIM
            print(
                f"{_sel}{_color}{_label:<30}{_end}  "
                f"{_ANSI_DIM}{_desc:<48}{_ANSI_RST}  "
                f"{_n:>6} trades  {_pc}{_pnl:>+10.2f}{_ANSI_RST}"
            )
        print(f"\n  {_ANSI_DIM}↑/↓ select mode  |  Enter confirm  |  1-7 switch tabs after selection{_ANSI_RST}")

    def _render_symbol_overview(self, symbol: str) -> None:
        print(f"\n\033[1m🔍 {symbol} — SYMBOL OVERVIEW\033[0m  {_ANSI_DIM}[Esc] back to portfolio{_ANSI_RST}\n")
        sym_bots = [bot for bot in (self.all_bots_stats or []) if str(bot.get("symbol", "")).upper() == symbol]
        if sym_bots:
            self._render_symbol_overview_bots(sym_bots)
        else:
            print(f"  {_ANSI_DIM}No running bots for {symbol}{_ANSI_RST}")
        self._render_system_health_block()

    def _render_symbol_overview_bots(self, bots: list[dict]) -> None:
        header = (
            f"  {'TF':<6} {'Mode':<5} {'Pos':<6} {'ε':>6} {'Buf%':>5} "
            f"{'ZΩ':>6} {'24h Trades':>10} {'24h PnL':>10}"
        )
        print(header)
        print("  " + "─" * (_visible_width(header) - 2))
        for bot in sorted(bots, key=lambda item: int(item.get("timeframe_minutes", 0) or 0)):
            self._render_symbol_overview_bot_row(bot)

    def _render_symbol_overview_bot_row(self, bot: dict) -> None:
        tfm = int(bot.get("timeframe_minutes", 0) or 0)
        mode = str(bot.get("trading_mode", "paper") or "paper").upper()[:5]
        day_stats = bot.get("daily_stats", {}) or {}
        day_pnl = float(day_stats.get("total_pnl", 0.0) or 0.0)
        mode_col = _ANSI_Y if mode == "PAPER" else _ANSI_G
        print(
            f"  {self._format_timeframe_minutes_label(tfm):<6} {mode_col}{mode:<5}{_ANSI_RST} "
            f"{str(bot.get('position_direction', 'FLAT') or 'FLAT').upper()[:6]:<6} "
            f"{float(bot.get('epsilon', 0.0) or 0.0):>6.3f} "
            f"{float(bot.get('buffer_fill_pct', 0.0) or 0.0):>4.0f}% "
            f"{float(bot.get('z_omega', 0.0) or 0.0):>6.3f} "
            f"{int(day_stats.get('total_trades', 0) or 0):>10} "
            f"{self._pnl_color(day_pnl)}{day_pnl:>+10.2f}{_ANSI_RST}"
        )

    def _overview_mode(self) -> str:
        return (
            getattr(self, "_trade_log_mode", "")
            or getattr(self, "_perf_snapshot_mode", "")
            or self.bot_config.get("trading_mode", "paper")
        )

    @staticmethod
    def _overview_account_tag(mode: str) -> str:
        if mode == "paper":
            return f"  {_ANSI_Y}(paper){_ANSI_RST}"
        if mode == "live":
            return f"  {_ANSI_G}(live){_ANSI_RST}"
        if mode == "mixed":
            return f"  {_ANSI_Y}(mixed source; no blended estimate){_ANSI_RST}"
        return ""

    def _overview_balance_values(
        self,
        mode: str,
        starting: float,
        lifetime_pnl: float,
    ) -> tuple[float, float, str, str]:
        unreal = float(self.position.get("unrealized_pnl", 0.0))
        real_bal = self.bot_config.get("real_account_balance")
        real_eq = self.bot_config.get("real_account_equity")
        real_margin = self.bot_config.get("real_margin_free")
        if real_bal is not None:
            balance = float(real_bal)
            live_tag = "  \033[32m✓ live\033[0m"
        elif mode == "mixed":
            balance = starting
            live_tag = f"  {_ANSI_Y}~ est withheld: mixed modes{_ANSI_RST}"
        else:
            balance = starting + lifetime_pnl
            live_tag = "  \033[33m~ est.\033[0m"
        equity = float(real_eq) if real_eq is not None else balance + unreal
        margin_str = f"  |  Free margin: \033[36m{float(real_margin):>10.2f}\033[0m" if real_margin is not None else ""
        return balance, equity, live_tag, margin_str

    def _render_overview_account(self) -> None:
        mode = self._overview_mode()
        print(f"\n\033[1m💰 ACCOUNT\033[0m{self._overview_account_tag(mode)}")
        starting = self._universe_starting_equity()
        lifetime_pnl = float(self.all_time_metrics.get("total_pnl", self.lifetime_metrics.get("total_pnl", 0.0)))
        balance, equity, live_tag, margin_str = self._overview_balance_values(mode, starting, lifetime_pnl)
        unreal = float(self.position.get("unrealized_pnl", 0.0))
        direction = (self.position.get("direction") or "FLAT").upper()
        unreal_str = (
            f"{_ANSI_DIM}—{_ANSI_RST}"
            if direction == "FLAT"
            else f"{self._pnl_color(unreal)}{unreal:+.2f}\033[0m"
        )
        print(
            f"  Balance: {self._pnl_color(balance - starting)}{balance:>10.2f}\033[0m{live_tag}  |  "
            f"Equity:  {self._pnl_color(equity - starting)}{equity:>10.2f}\033[0m  |  "
            f"Unrealized: {unreal_str}{margin_str}"
        )

    def _render_overview_24h(self) -> None:
        print(f"\n\033[1m📈 LAST 24H\033[0m  {_ANSI_DIM}(rolling 24-hour window from trade_log.jsonl){_ANSI_RST}")
        if self._overview_mode() == "mixed":
            self._render_overview_24h_mode("Paper", self.daily_metrics_by_mode.get("paper", {}))
            self._render_overview_24h_mode("Live ", self.daily_metrics_by_mode.get("live", {}))
            return
        metrics = self.daily_metrics
        trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0) * 100
        day_pnl = metrics.get("total_pnl", 0)
        print(
            f"  Trades: {trades}  |  Win Rate: {win_rate:.1f}%  |  "
            f"PnL: {self._pnl_color(day_pnl)}{day_pnl:+.2f}\033[0m"
        )
        recent_pnl = metrics.get("recent_pnl_sequence", [])
        if recent_pnl and len(recent_pnl) > 1:
            print(f"  Recent: {self._create_sparkline(recent_pnl[-20:])}")

    def _render_overview_24h_mode(self, label: str, metrics: dict) -> None:
        trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0) * 100
        pnl = metrics.get("total_pnl", 0)
        print(f"  {label}: {trades} trades | WR {win_rate:.1f}% | PnL {self._pnl_color(pnl)}{pnl:+.2f}\033[0m")

    def _overview_symbol_tf_keys(self) -> set:
        tf_keys: set = set(self.metrics_by_symbol_tf.keys())
        for bot in self.all_bots_stats or []:
            symbol = self._normalize_symbol(bot.get("symbol"))
            tfm = int(bot.get("timeframe_minutes", 0) or 0)
            if symbol and tfm > 0:
                tf_keys.add((symbol, f"M{tfm}"))
        return tf_keys

    def _render_overview_symbol_tf_snapshot(self) -> None:
        tf_keys = self._overview_symbol_tf_keys()
        if not tf_keys:
            return
        scope = self._epoch_scope_label().upper()
        print(
            f"\n\033[1m🧩 SYMBOL / TF SNAPSHOT\033[0m  "
            f"{_ANSI_DIM}— {scope} closed trades from trade_log.jsonl{_ANSI_RST}"
        )
        header = f"  {'Symbol':<9} {'TF':<6} {'Trades':>7} {'Win%':>7} {'PnL $':>11}"
        print(header)
        print("  " + "─" * (_visible_width(header) - 2))
        for symbol, tf in sorted(tf_keys, key=lambda kv: (kv[0], self._timeframe_sort_key(kv[1]))):
            metrics = self.metrics_by_symbol_tf.get((symbol, tf), {})
            trades = metrics.get("total_trades", 0)
            pnl = metrics.get("total_pnl", 0.0)
            pnl_col = self._pnl_color(pnl) if trades > 0 else _ANSI_DIM
            print(
                f"  {symbol:<9} {tf:<6} {trades:>7} "
                f"{metrics.get('win_rate', 0) * 100:>6.1f}% {pnl_col}{pnl:>+11.2f}{_ANSI_RST}"
            )

    def _render_overview_risk(self) -> None:
        print("\n\033[1m⚠️  RISK STATUS\033[0m")
        cb = self.risk_stats.get("circuit_breaker", "INACTIVE")
        regime = self.risk_stats.get("regime", "UNKNOWN")
        zeta = self.risk_stats.get("regime_zeta", 1.0)
        vol = self.risk_stats.get("realized_vol", 0) * 100
        feas = self.risk_stats.get("feasibility", 0.5)
        cb_status = self._overview_cb_status(cb, bool(self.risk_stats.get("kurtosis_gate_active", False)))
        feas_color = (
            _ANSI_G
            if feas > FEASIBILITY_HIGH_THRESHOLD
            else (_ANSI_Y if feas > FEASIBILITY_MEDIUM_THRESHOLD else _ANSI_R)
        )
        regime_color = {
            "TRENDING": _ANSI_G,
            "MEAN_REVERTING": _ANSI_Y,
            "TRANSITIONAL": _ANSI_B,
            "UNKNOWN": _ANSI_DIM,
        }.get(regime, _ANSI_DIM)
        print(
            f"  Circuit: {cb_status}  |  Regime: {regime_color}{regime}\033[0m (ζ={zeta:.2f})  |  "
            f"Vol: {vol:.2f}%  |  Feasibility: {feas_color}{feas:.2f}\033[0m"
        )

    @staticmethod
    def _overview_cb_status(cb: str, kurt_gate: bool) -> str:
        if cb == "ACTIVE":
            return f"{_ANSI_R}● ACTIVE{_ANSI_RST}"
        if kurt_gate:
            return f"{_ANSI_Y}● κ-gate{_ANSI_RST}"
        return f"{_ANSI_G}● OK{_ANSI_RST}"

    def _render_overview_market(self) -> None:
        market_scope = self._risk_scope_label(self.risk_stats)
        print(f"\n\033[1m🔬 MARKET [{market_scope}]\033[0m")
        spread = self.market_stats.get("spread", 0)
        vpin = self.market_stats.get("vpin", 0)
        vpin_z = self.market_stats.get("vpin_z", 0)
        imb = self.market_stats.get("imbalance", 0)
        vpin_status = (
            f"{_ANSI_R}⚠️ HIGH{_ANSI_RST}"
            if abs(vpin_z) > VPIN_HIGH_TOXICITY_THRESHOLD
            else f"{_ANSI_G}✓{_ANSI_RST}"
        )
        imb_label = "Imb" if self.market_stats.get("has_real_sizes", False) else "QFI"
        print(
            f"  Spread: {self._spread_color()}{spread:.5f} ({self._spread_bps():.1f}bp){_ANSI_RST}  |  "
            f"VPIN: {vpin:.3f} (z={vpin_z:+.1f}) {vpin_status}  |  {imb_label}: {imb:+.3f}"
        )

    def _render_overview_alerts(self) -> None:
        alerts = self.production_metrics.get("alerts", [])
        if not alerts:
            return
        print("\n\033[1m🚨 ALERTS\033[0m")
        for alert in alerts:
            print(f"  {_ANSI_Y}⚠ {alert}{_ANSI_RST}")

    def _render_overview(self) -> None:
        """Render overview tab — dispatches by drill level (max L2)."""
        self._render_breadcrumb("OVERVIEW", 1)

        if self._ctx_level >= 2 and self._ctx_symbol:
            self._render_symbol_overview(self._ctx_symbol.upper())
            return

        self._render_all_bots_panel()
        self._render_position_block()
        print(f"  {_ANSI_DIM}(performance source: trade_log.jsonl; paper/live rows stay mode-separated){_ANSI_RST}")
        self._render_overview_account()
        self._render_overview_24h()
        self._render_overview_symbol_tf_snapshot()
        self._render_overview_risk()
        self._render_agent_status_block()
        self._render_overview_market()
        self._render_system_health_block()
        self._render_overview_alerts()

    def _render_agent_status_block(self) -> None:
        """Render the agent status (training snapshot) block."""
        print("\n\033[1m🧠 AGENT STATUS\033[0m")
        trig_buf = self.training_stats.get("trigger_buffer_size", 0)
        harv_buf = self.training_stats.get("harvester_buffer_size", 0)
        trig_steps = self.training_stats.get("trigger_training_steps", 0)
        harv_steps = self.training_stats.get("harvester_training_steps", 0)
        trig_eps = self.training_stats.get("trigger_epsilon", 0.0)
        harv_beta = self.training_stats.get("harvester_beta", 0.4)
        total_agents = self.training_stats.get("total_agents", 0)
        if total_agents > 0:
            print(f"  Arena: {total_agents} agents  |  Trigger: {trig_buf:,} exp  |  Harvester: {harv_buf:,} exp")
        else:
            print(f"  Trigger:   {trig_buf:,} exp  |  {trig_steps:,} steps  |  ε={trig_eps:.4f}")
            print(f"  Harvester: {harv_buf:,} exp  |  {harv_steps:,} steps  |  β={harv_beta:.4f}")

    def _render_system_health_block(self) -> None:
        """Render the expanded system health rows and startup self-test."""
        _scope = self._risk_scope_label(self.risk_stats)
        print(f"\n\033[1m🏥 SYSTEM HEALTH [{_scope}]\033[0m")
        self._render_health_connectivity()
        self._render_health_session_events()
        self._render_health_risk()
        self._render_health_buffers()
        self._render_health_model()
        self._render_health_microstructure()
        self._render_health_system_metrics()
        self._render_health_self_test()
        self._render_health_analyzer()

    def _render_health_connectivity(self) -> None:
        """Render data freshness and breaker status row."""

        def _ok(s: str) -> str:
            return f"{_ANSI_G}✓ {s}{_ANSI_RST}"

        def _warn(s: str) -> str:
            return f"{_ANSI_Y}⚡ {s}{_ANSI_RST}"

        def _bad(s: str) -> str:
            return f"{_ANSI_R}✗ {s}{_ANSI_RST}"

        _scope = self._risk_scope_label(self.risk_stats)
        _ob_path = self._preferred_data_file(_ORDER_BOOK_FILE)
        _bc_path = self._preferred_data_file(_BOT_CONFIG_FILE)
        _ref_path = self._active_bot_stats_path()
        if _ref_path is None and _ob_path.exists():
            _ref_path = _ob_path
        elif _ref_path is None and _bc_path.exists():
            _ref_path = _bc_path
        elif _ref_path is None and not (self.active_sym and self.active_tf_min):
            # Legacy no-active-bot startup: use newest paper stats only when
            # there is no active scope to protect from cross-timeframe bleed.
            _ps_paths = sorted(
                self.data_dir.glob("paper_stats_*.json"),
                key=lambda _p: _p.stat().st_mtime if _p.exists() else 0,
                reverse=True,
            )
            if _ps_paths:
                _ref_path = _ps_paths[0]
        if _ref_path is not None:
            _file_age = time.time() - Path(_ref_path).stat().st_mtime
            _age_str = f"{_file_age:.0f}s"
            if _file_age < DATA_AGING_SECS:
                _data_item = _ok(f"Data {_scope} {_age_str}")
            elif _file_age < DATA_STALE_SECS:
                _data_item = _warn(f"Data {_scope} {_age_str}")
            else:
                _data_item = _bad(f"Bot silent {_scope} {_age_str}")
        else:
            _data_item = _bad("No data")

        _cb = self.risk_stats.get("circuit_breaker", "INACTIVE")
        _cb_item = _bad("CB ACTIVE") if _cb == "ACTIVE" else _ok("CB OK")

        _depth_gate = self.risk_stats.get("depth_gate_active", False)
        _gate_item = _warn("Depth gate") if _depth_gate else _ok("Gate open")

        _feas = float(self.risk_stats.get("feasibility", 0.5))
        if _feas > FEASIBILITY_HIGH_THRESHOLD:
            _feas_col = _ANSI_G
        elif _feas > FEASIBILITY_MEDIUM_THRESHOLD:
            _feas_col = _ANSI_Y
        else:
            _feas_col = _ANSI_R
        _feas_item = f"Feas: {_feas_col}{_feas:.2f}{_ANSI_RST}"

        print(f"  {_data_item}  │  {_cb_item}  │  {_gate_item}  │  {_feas_item}")

    def _render_health_risk(self) -> None:
        """Render VaR/vol/budget/efficiency row."""
        _vol = float(self.risk_stats.get("realized_vol", 0)) * 100
        if _vol < VOL_WARN_PCT:
            _vol_col = _ANSI_G
        elif _vol < VOL_HIGH_PCT:
            _vol_col = _ANSI_Y
        else:
            _vol_col = _ANSI_R
        _vol_item = f"Vol: {_vol_col}{_vol:.2f}%{_ANSI_RST}"

        _var = float(self.risk_stats.get("var", 0)) * 100
        if _var < VAR_WARN_PCT:
            _var_col = _ANSI_G
        elif _var < VAR_HIGH_PCT:
            _var_col = _ANSI_Y
        else:
            _var_col = _ANSI_R
        _var_item = f"VaR: {_var_col}{_var:.2f}%{_ANSI_RST}"

        _budget = float(self.risk_stats.get("risk_budget_usd", 0))
        if _budget > _BUDGET_OK_MIN:
            _budget_col = _ANSI_G
        elif _budget > 0:
            _budget_col = _ANSI_Y
        else:
            _budget_col = _ANSI_R
        _budget_item = f"Budget: {_budget_col}${_budget:.2f}{_ANSI_RST}"

        _eff = float(self.risk_stats.get("efficiency", 0))
        if _eff > EFF_HIGH_THRESHOLD:
            _eff_col = _ANSI_G
        elif _eff > EFF_WARN_THRESHOLD:
            _eff_col = _ANSI_Y
        else:
            _eff_col = _ANSI_R
        _eff_item = f"Eff: {_eff_col}{_eff:.2f}{_ANSI_RST}"

        print(f"  {_vol_item}  │  {_var_item}  │  {_budget_item}  │  {_eff_item}")

    def _render_health_buffers(self) -> None:
        """Render replay buffer occupancy row."""
        _trig_buf = self.training_stats.get("trigger_buffer_size", 0)
        _harv_buf = self.training_stats.get("harvester_buffer_size", 0)
        _trig_pct = _trig_buf / _RT_TRIG_CAP * 100
        _harv_pct = _harv_buf / _RT_HARV_CAP * 100
        _trig_rdy = self.training_stats.get("trigger_ready", False)
        _harv_rdy = self.training_stats.get("harvester_ready", False)
        if _trig_pct > BUF_PCT_HIGH:
            _trig_col = _ANSI_G
        elif _trig_pct > BUF_PCT_WARN:
            _trig_col = _ANSI_Y
        else:
            _trig_col = _ANSI_R
        if _harv_pct > BUF_PCT_HIGH:
            _harv_col = _ANSI_G
        elif _harv_pct > BUF_PCT_WARN:
            _harv_col = _ANSI_Y
        else:
            _harv_col = _ANSI_R

        def _rdy_icon(r: bool) -> str:
            return f"{_ANSI_G}✓{_ANSI_RST}" if r else f"{_ANSI_Y}…{_ANSI_RST}"

        print(
            f"  Trig buf: {_trig_col}{_trig_buf:,}/{_RT_TRIG_CAP:,} "
            f"({_trig_pct:.0f}%){_ANSI_RST} {_rdy_icon(_trig_rdy)}  │  "
            f"Harv buf: {_harv_col}{_harv_buf:,}/{_RT_HARV_CAP:,} "
            f"({_harv_pct:.0f}%){_ANSI_RST} {_rdy_icon(_harv_rdy)}"
        )

    def _render_health_model(self) -> None:
        """Render epsilon/beta/steps/loss row."""
        _eps = float(self.training_stats.get("trigger_epsilon", 1.0))
        _beta = float(self.training_stats.get("harvester_beta", 0.4))
        _trig_steps = self.training_stats.get("trigger_training_steps", 0)
        _harv_steps = self.training_stats.get("harvester_training_steps", 0)
        _trig_loss = self.training_stats.get("trigger_loss", None)
        _harv_loss = self.training_stats.get("harvester_loss", None)
        if _eps < EPS_HOT_MAX:
            _eps_col = _ANSI_G
            _eps_lbl = "HOT"
        elif _eps < EPS_WARM_MAX:
            _eps_col = _ANSI_Y
            _eps_lbl = "WARM"
        else:
            _eps_col = _ANSI_R
            _eps_lbl = "COLD"

        def _loss_str(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "n/a"

        # Beta label
        if _beta >= BETA_HOT_MIN:
            _beta_col = _ANSI_G
            _beta_lbl = "HOT"
        elif _beta >= BETA_WARM_MIN:
            _beta_col = _ANSI_Y
            _beta_lbl = "WARM"
        else:
            _beta_col = _ANSI_R
            _beta_lbl = "COLD"

        print(
            f"  ε={_eps_col}{_eps:.4f} {_eps_lbl}{_ANSI_RST}  steps={_trig_steps:,}  loss={_loss_str(_trig_loss)}"
            f"  τ={self.training_stats.get('trigger_tau', 0.005):.5f}"
        )
        print(
            f"  β={_beta_col}{_beta:.4f} {_beta_lbl}{_ANSI_RST}  steps={_harv_steps:,}  loss={_loss_str(_harv_loss)}"
            f"  τ={self.training_stats.get('harvester_tau', 0.005):.5f}"
        )

    def _spread_bps(self) -> float:
        """Return current spread in basis points relative to mid price."""
        _spread = float(self.market_stats.get("spread", 0))
        _bids = self.market_stats.get("order_book_bids", [])
        _asks = self.market_stats.get("order_book_asks", [])
        _mid = 0.0
        if _bids and _asks:
            _mid = (_bids[0][0] + _asks[0][0]) / 2.0
        elif _bids:
            _mid = _bids[0][0]
        elif _asks:
            _mid = _asks[0][0]
        if _mid > 0:
            return (_spread / _mid) * 10_000.0
        return 0.0

    def _spread_color(self) -> str:
        """Return ANSI colour code for spread based on basis-point bands."""
        bps = self._spread_bps()
        if bps < SPREAD_OK_BPS:
            return _ANSI_G
        if bps < SPREAD_WARN_BPS:
            return _ANSI_Y
        return _ANSI_R

    def _render_health_microstructure(self) -> None:
        """Render spread/VPIN/runway row."""
        _spread = float(self.market_stats.get("spread", 0))
        _vpin = float(self.market_stats.get("vpin", 0))
        _vpin_z = float(self.market_stats.get("vpin_z", 0))
        _runway = float(self.risk_stats.get("runway", 0))
        _spread_col = self._spread_color()
        _bps = self._spread_bps()
        if abs(_vpin_z) > _VPIN_OV_HIGH:
            _vpin_col = _ANSI_R
        elif abs(_vpin_z) > _VPIN_OV_ELEVATED:
            _vpin_col = _ANSI_Y
        else:
            _vpin_col = _ANSI_G
        if _runway > RUNWAY_OK_BARS:
            _runway_col = _ANSI_G
        elif _runway > RUNWAY_WARN_BARS:
            _runway_col = _ANSI_Y
        else:
            _runway_col = _ANSI_R
        print(
            f"  Spread: {_spread_col}{_spread:.5f} ({_bps:.1f}bp){_ANSI_RST}  │  "
            f"VPIN: {_vpin_col}{_vpin:.3f} (z={_vpin_z:+.1f}){_ANSI_RST}  │  "
            f"Runway: {_runway_col}{_runway:.2f}{_ANSI_RST}"
        )

    def _render_health_system_metrics(self) -> None:
        """Render memory, error count, uptime, FIX connectivity from production_metrics."""
        _pm = self.production_metrics.get("metrics", {})
        if not _pm:
            return
        _items: list[str] = []
        _mem = _pm.get("memory_usage_pct")
        if _mem is not None:
            _mem_f = float(_mem)
            _mem_col = _ANSI_R if _mem_f > 80 else (_ANSI_Y if _mem_f > 60 else _ANSI_G)
            _items.append(f"Mem: {_mem_col}{_mem_f:.0f}%{_ANSI_RST}")
        _err = _pm.get("error_count_1h")
        if _err is not None:
            _err_i = int(_err)
            _err_col = _ANSI_R if _err_i > 5 else (_ANSI_Y if _err_i > 0 else _ANSI_G)
            _items.append(f"Err/1h: {_err_col}{_err_i}{_ANSI_RST}")
        _up = _pm.get("uptime_hours")
        if _up is not None:
            _items.append(f"Up: {float(_up):.1f}h")
        _fix = _pm.get("fix_connected")
        if _fix is not None:
            _fix_c = _ANSI_G if _fix else _ANSI_R
            _fix_s = "✓" if _fix else "✗"
            _items.append(f"FIX: {_fix_c}{_fix_s}{_ANSI_RST}")
        if _items:
            print(f"  {'  │  '.join(_items)}")

    def _render_health_self_test(self) -> None:
        """Render startup self-test entries when available."""
        if not self.self_test_results:
            return
        _sev_col = {
            "PASS": _ANSI_G,
            "INFO": _ANSI_B,
            "WARNING": _ANSI_Y,
            "CRITICAL": _ANSI_R,
        }
        _sev_icon = {"PASS": "✓", "INFO": "ℹ", "WARNING": "⚠", "CRITICAL": "✗"}
        n_crit = sum(1 for r in self.self_test_results if r["sev"] == "CRITICAL")
        n_warn = sum(1 for r in self.self_test_results if r["sev"] == "WARNING")
        if n_crit:
            status = f"{_ANSI_R}🔴 FAILED{_ANSI_RST}"
        elif n_warn:
            status = f"{_ANSI_Y}🟡 DEGRADED{_ANSI_RST}"
        else:
            status = f"{_ANSI_G}🟢 CLEAR{_ANSI_RST}"
        n_pass = sum(1 for r in self.self_test_results if r["sev"] in ("PASS", "INFO"))
        print(
            f"\n\033[1m🔍 STARTUP SELF-TEST\033[0m  {status}  "
            f"{_ANSI_DIM}({n_pass} OK, {n_warn} warn, {n_crit} crit){_ANSI_RST}"
        )
        only_fails = n_crit > 0 or n_warn > 0
        for r in self.self_test_results:
            sev = r["sev"]
            if only_fails and sev in ("PASS", "INFO"):
                continue  # show only problems when there are any
            col = _sev_col.get(sev, "")
            icon = _sev_icon.get(sev, "?")
            detail = f"  {_ANSI_DIM}{r['detail']}{_ANSI_RST}" if r.get("detail") else ""
            print(f"  {col}{icon} {r['name']}{_ANSI_RST}{detail}")

    @staticmethod
    def _health_status_style(overall: str) -> tuple[str, str]:
        if overall == "HEALTHY":
            return _ANSI_G, "🟢"
        if overall in ("DEGRADED", "WARNING"):
            return _ANSI_Y, "🟡"
        if overall == "NO_DATA":
            return _ANSI_DIM, "⬜"
        return _ANSI_R, "🔴"

    @staticmethod
    def _health_report_age(report: dict) -> str:
        generated = report.get("generated_at", "")
        if not generated:
            return ""
        try:
            generated_at = (
                datetime.fromisoformat(generated).replace(tzinfo=UTC)
                if generated.endswith("Z")
                else datetime.fromisoformat(generated)
            )
            age_s = (datetime.now(UTC) - generated_at).total_seconds()
            return f"{age_s / 60:.0f}m ago" if age_s < 3600 else f"{age_s / 3600:.1f}h ago"
        except Exception:
            return ""

    @staticmethod
    def _health_list(value: Any) -> list:
        return value if isinstance(value, list) else []

    @staticmethod
    def _health_dict(value: Any) -> dict:
        return value if isinstance(value, dict) else {}

    def _render_health_fleet_summary(self, fleet: dict) -> None:
        n_trades = int(fleet.get("total_trades", fleet.get("n_trades", 0)) or 0)
        win_rate = float(fleet.get("win_rate", 0)) * 100
        profit_factor = float(fleet.get("profit_factor", 0))
        emergency_rate = float(fleet.get("emergency_rate", 0)) * 100
        wr_col = _ANSI_G if win_rate >= 50 else (_ANSI_Y if win_rate >= 35 else _ANSI_R)
        pf_col = _ANSI_G if profit_factor >= 1.2 else (_ANSI_Y if profit_factor >= 1.0 else _ANSI_R)
        emg_col = _ANSI_R if emergency_rate > 5 else (_ANSI_Y if emergency_rate > 2 else _ANSI_G)
        print(
            f"  Fleet: {n_trades} trades │ "
            f"WR {wr_col}{win_rate:.0f}%{_ANSI_RST} │ "
            f"PF {pf_col}{profit_factor:.2f}{_ANSI_RST} │ "
            f"Emg {emg_col}{emergency_rate:.1f}%{_ANSI_RST}"
        )

    @staticmethod
    def _health_anomaly_label(anomaly: Any) -> str:
        if isinstance(anomaly, dict):
            bot = f"{anomaly.get('symbol', '?')} {anomaly.get('timeframe', '?')}"
            code = anomaly.get("code", "?")
        else:
            bot = "fleet"
            code = str(anomaly)
        return f"{_ANSI_Y}⚡ {bot} {code}{_ANSI_RST}"

    def _render_health_anomalies(self, anomalies: list) -> None:
        if not anomalies:
            print(f"  {_ANSI_G}✓ No anomalies detected{_ANSI_RST}")
            return
        labels = [self._health_anomaly_label(anomaly) for anomaly in anomalies[:4]]
        print(f"  Anomalies: {'  '.join(labels)}")
        if len(anomalies) > 4:
            print(f"  {_ANSI_DIM}  … and {len(anomalies) - 4} more{_ANSI_RST}")

    @staticmethod
    def _health_correction_label(correction: Any) -> str:
        if not isinstance(correction, dict):
            return _truncate_visible(str(correction), 72)
        bot = f"{correction.get('symbol', '?')} {correction.get('timeframe', '?')}"
        param = correction.get("parameter", "?").replace("_", " ")
        old = correction.get("old_value")
        new = correction.get("new_value")
        if old is not None and new is not None:
            return f"{bot} {param} {old:.3f}→{new:.3f}"
        return f"{bot} {param}"

    def _render_health_corrections(self, corrections: list) -> None:
        if not corrections:
            return
        parts = [self._health_correction_label(correction) for correction in corrections[:3]]
        print(f"  Applied: {_ANSI_G}{', '.join(parts)}{_ANSI_RST}")

    def _render_health_analyzer(self) -> None:
        """Render self-healing performance analyzer status row."""
        hr = self._health_report
        if not hr:
            print(f"\n\033[1m🔄 SELF-HEAL\033[0m  {_ANSI_DIM}no report yet — runs every 4 h{_ANSI_RST}")
            return

        overall = hr.get("overall_health", "UNKNOWN")
        health_col, health_icon = self._health_status_style(overall)
        age_str = self._health_report_age(hr)
        window = hr.get("analysis_window_hours", 4)
        header_age = f"  {_ANSI_DIM}({window:.0f}h window{', ' + age_str if age_str else ''}){_ANSI_RST}"
        print(f"\n\033[1m🔄 SELF-HEAL\033[0m  {health_icon} {health_col}{overall}{_ANSI_RST}{header_age}")
        self._render_health_fleet_summary(self._health_dict(hr.get("fleet", {})))
        self._render_health_anomalies(self._health_list(hr.get("anomalies", [])))
        self._render_health_corrections(self._health_list(hr.get("corrections_applied", [])))

    def _render_health_session_events(self) -> None:
        """Render per-bot last session start and recent connection events from transactions.jsonl."""
        _SESSION_TYPES = frozenset({"SESSION_START", "SESSION_EVENT", "COMPONENT_HEALTH"})
        _events = self._load_transaction_events(event_types=_SESSION_TYPES, n=80)
        if not _events:
            print(f"\n\033[1m🔌 SESSION LOG\033[0m  {_ANSI_DIM}no transactions.jsonl data yet{_ANSI_RST}")
            return

        # Latest SESSION_START per source file → one row per bot
        _last_start: dict[str, dict] = {}
        _non_start: list[dict] = []
        for e in _events:
            _src = e.get("_source_path", "")
            _et = e.get("event_type", "")
            if _et == "SESSION_START" and _src not in _last_start:
                _last_start[_src] = e
            elif _et in ("SESSION_EVENT", "COMPONENT_HEALTH"):
                _non_start.append(e)

        print(f"\n\033[1m🔌 SESSION LOG\033[0m  {_ANSI_DIM}(from transactions.jsonl){_ANSI_RST}")

        for _src, _e in sorted(_last_start.items()):
            _ts = str(_e.get("timestamp") or "?")
            try:
                _dt = datetime.fromisoformat(_ts)
                _age_s = (datetime.now(UTC) - _dt).total_seconds()
                _age_str = f"{_age_s/3600:.1f}h ago" if _age_s >= 3600 else f"{_age_s/60:.0f}m ago"
            except Exception:
                _age_str = _ts[:16]
            # Bot label from source path
            from pathlib import Path as _Path
            _bot_lbl = _Path(_src).parent.parent.parent.name
            if _bot_lbl in (".", "audit", "logs"):
                _bot_lbl = "root"
            _sess = str((_e.get("data") or {}).get("session_id") or _e.get("session") or "?")[:20]
            print(f"  {_ANSI_G}●{_ANSI_RST} {_bot_lbl:<22} last start {_age_str}  {_ANSI_DIM}({_sess}){_ANSI_RST}")

        # Recent non-start events (last 5)
        for _e in _non_start[:5]:
            _et = _e.get("event_type", "")
            _ts = str(_e.get("timestamp") or "?")[:16]
            _d = _e.get("data") or {}
            if _et == "COMPONENT_HEALTH":
                _healthy = bool(_d.get("healthy", True))
                _comp = str(_d.get("component") or "?")
                _errs = int(_d.get("error_count") or 0)
                _col = _ANSI_G if _healthy else _ANSI_R
                _icon = "✓" if _healthy else "✗"
                print(f"  {_col}{_icon} {_ts} COMPONENT_HEALTH {_comp} err={_errs}{_ANSI_RST}")
            elif _et == "SESSION_EVENT":
                _ev = str(_d.get("event") or "?")
                _st = str(_d.get("session_type") or "?")
                _col = _ANSI_R if "disconnect" in _ev.lower() or "logout" in _ev.lower() else _ANSI_Y
                print(f"  {_col}⚡ {_ts} SESSION_EVENT {_st} {_ev}{_ANSI_RST}")

    def _render_perf_period_columns(
        self,
        label_rows: list[tuple[str, str, list[dict], list[dict]]],
    ) -> None:
        """Render a period-column summary table.

        label_rows: list of (scope_label, mode_key, epoch_trades, all_trades)
        Columns: Lifetime | Epoch | Month | 7d | 24h
        Each cell: #N  WR%  PnL$
        """
        _starting = self._universe_starting_equity()
        _COL_W = 20
        _has_epoch = bool(self._stats_epoch)
        _periods_hdr = ["Lifetime", "Epoch", "Month", "7d", "24h"] if _has_epoch else ["Lifetime", "Month", "7d", "24h"]
        _scope_w = 18

        _hdr = f"  {'Scope':<{_scope_w}}"
        for _p in _periods_hdr:
            _hdr += f"  {_p:^{_COL_W}}"
        print(_hdr)
        print("  " + "─" * (_visible_width(_hdr) - 2))

        for _lbl, _mode_key, _ep_trades, _all_trades in label_rows:
            _daily, _weekly, _monthly = _classify_trades_by_period(_ep_trades)
            _mode_c = _ANSI_Y if _mode_key == "paper" else (_ANSI_G if _mode_key == "live" else _ANSI_DIM)

            def _cell(trades: list[dict]) -> str:
                m = _hud_period_metrics(trades, _starting)
                n = int(m.get("total_trades", 0) or 0)
                if n == 0:
                    return f"{'—':^{_COL_W}}"
                wr = m.get("win_rate", 0.0) * 100
                pnl = m.get("total_pnl", 0.0)
                return _fmt_trade_period_cell(n, wr, pnl, _COL_W)

            _cells: list[str] = []
            if _has_epoch:
                _cells.append(_cell(list(_all_trades)))
                _cells.append(_cell(list(_ep_trades)))
            else:
                _cells.append(_cell(list(_all_trades)))
            _cells.append(_cell(list(_monthly)))
            _cells.append(_cell(list(_weekly)))
            _cells.append(_cell(list(_daily)))

            _row = f"  {_mode_c}{_lbl:<{_scope_w}}{_ANSI_RST}"
            for _c in _cells:
                _row += f"  {_c}"
            print(_row)

    def _performance_mode(self) -> str:
        return (
            getattr(self, "_trade_log_mode", "")
            or getattr(self, "_perf_snapshot_mode", "")
            or self.bot_config.get("trading_mode", "paper")
        )

    def _render_performance_intro(self) -> None:
        src = (
            f"  {_ANSI_DIM}(source: trade_log.jsonl; modes never blended){_ANSI_RST}"
            if self._metrics_from_trade_log
            else ""
        )
        print(f"\n\033[1m📈 PERFORMANCE\033[0m{src}\n")
        if self._trade_log_unlabeled_count > 0:
            print(
                f"  {_ANSI_Y}⚠ {self._trade_log_unlabeled_count} legacy trades were missing trading_mode; "
                f"{self._trade_log_inferred_count} inferred via ticket/position heuristics.{_ANSI_RST}"
            )
        if self._trade_log_unknown_timeframe_count > 0:
            print(
                f"  {_ANSI_Y}ℹ M? = legacy trades missing timeframe metadata "
                f"({self._trade_log_unknown_timeframe_count} trades).{_ANSI_RST}"
            )
        if self._stats_epoch:
            epoch_str = self._stats_epoch.strftime("%Y-%m-%d %H:%M")
            excluded_pnl = self._stats_epoch_excluded_pnl
            print(
                f"  {_ANSI_DIM}📅 Epoch {epoch_str} UTC; excluded {self._stats_epoch_excluded} trades "
                f"({self._pnl_color(excluded_pnl)}{excluded_pnl:+.2f}{_ANSI_RST}{_ANSI_DIM}); [e] edit{_ANSI_RST}"
            )

    def _render_perf_period_rows(self, rows: list[tuple[str, dict]], *, indent: str = "  ") -> None:
        header = (
            f"{indent}{'Period':<9} {'Trades':>7} {'Win%':>7} {'PnL $':>12} "
            f"{'TQ':>7} {'PF':>7} {'MaxDD':>7}"
        )
        print(header)
        print(f"{indent}" + "─" * (_visible_width(header) - len(indent)))
        for label, metrics in rows:
            self._render_perf_period_row(label, metrics, indent=indent)

    def _render_perf_period_row(self, label: str, metrics: dict, *, indent: str) -> None:
        pnl = metrics.get("total_pnl", 0)
        maxdd = metrics.get("max_drawdown", 0.0)
        if maxdd > DD_HIGH_PCT:
            dd_color = _ANSI_R
        elif maxdd > DD_WARN_PCT:
            dd_color = _ANSI_Y
        else:
            dd_color = _ANSI_G
        print(
            f"{indent}{label:<9} {metrics.get('total_trades', 0):>7} "
            f"{metrics.get('win_rate', 0) * 100:>6.1f}% "
            f"{self._pnl_color(pnl)}{pnl:>+12.2f}{_ANSI_RST} "
            f"{metrics.get('sharpe_ratio', 0):>7.2f} {metrics.get('profit_factor', 0):>7.2f} "
            f"{dd_color}{maxdd:>6.2f}%{_ANSI_RST}"
        )

    @staticmethod
    def _symbol_mode_tf_trades(trades: list[dict], symbol: str, mode: str, tfm: int) -> list[dict]:
        return [
            trade for trade in trades
            if str(trade.get("symbol", "")).upper() == symbol
            and int(trade.get("timeframe_minutes", 0) or 0) == tfm
            and trade.get("trading_mode") == mode
        ]

    def _perf_portfolio_period_rows(self) -> list[tuple[str, str, list[dict], list[dict]]]:
        rows: list[tuple[str, str, list[dict], list[dict]]] = []
        for symbol, mode in self._l1_rows():
            epoch_trades = [
                trade for trade in self._trade_log_metrics_trades
                if str(trade.get("symbol", "")).upper() == symbol and trade.get("trading_mode") == mode
            ]
            all_trades = [
                trade for trade in self._trade_log_all_trades
                if str(trade.get("symbol", "")).upper() == symbol and trade.get("trading_mode") == mode
            ]
            if all_trades:
                mode_badge = "PPR" if mode == "paper" else "LIV"
                rows.append((f"{symbol} {mode_badge}", mode, epoch_trades, all_trades))
        return rows

    def _perf_symbol_period_rows(self) -> list[tuple[str, str, list[dict], list[dict]]]:
        symbol = self._ctx_symbol.upper()
        tf_filter = self._ctx_tf if self._ctx_level >= 3 else 0
        rows: list[tuple[str, str, list[dict], list[dict]]] = []
        for mode in ("live", "paper"):
            for tfm in self._perf_symbol_timeframes(symbol):
                if tf_filter and tfm != tf_filter:
                    continue
                epoch_trades = self._symbol_mode_tf_trades(self._trade_log_metrics_trades, symbol, mode, tfm)
                all_trades = self._symbol_mode_tf_trades(self._trade_log_all_trades, symbol, mode, tfm)
                if all_trades:
                    tf_label = self._format_timeframe_minutes_label(tfm)
                    mode_badge = "LIV" if mode == "live" else "PPR"
                    rows.append((f"{symbol}/{tf_label} {mode_badge}", mode, epoch_trades, all_trades))
        return rows

    def _perf_symbol_timeframes(self, symbol: str) -> list[int]:
        return sorted({
            int(trade.get("timeframe_minutes", 0) or 0)
            for trade in self._trade_log_all_trades
            if str(trade.get("symbol", "")).upper() == symbol and trade.get("timeframe_minutes")
        })

    def _render_performance_period_summary(self) -> None:
        if self._ctx_level == 1:
            self._render_portfolio_period_summary()
            return
        if self._ctx_level >= 2:
            self._render_symbol_period_summary()

    def _render_portfolio_period_summary(self) -> None:
        rows = self._perf_portfolio_period_rows()
        if not rows:
            return
        print("  \033[1mPORTFOLIO SUMMARY\033[0m  " + _ANSI_DIM + "period columns; [Enter] to drill" + _ANSI_RST)
        self._render_perf_period_columns(rows)
        print()

    def _render_symbol_period_summary(self) -> None:
        rows = self._perf_symbol_period_rows()
        if not rows:
            return
        tf_filter = self._ctx_tf if self._ctx_level >= 3 else 0
        scope = (
            f"{self._ctx_symbol.upper()}/{self._format_timeframe_minutes_label(tf_filter)}"
            if tf_filter
            else self._ctx_symbol.upper()
        )
        print(f"  \033[1m{scope} SUMMARY\033[0m  " + _ANSI_DIM + "period columns" + _ANSI_RST)
        self._render_perf_period_columns(rows)
        print()

    def _performance_mode_blocks(
        self,
        by_mode: dict[str, list[dict]],
        all_by_mode: dict[str, list[dict]],
    ) -> list[tuple[str, str, str, list[dict], list[dict]]]:
        return [
            ("paper", "PAPER", _ANSI_Y, by_mode["paper"], all_by_mode["paper"]),
            ("live", "LIVE", _ANSI_G, by_mode["live"], all_by_mode["live"]),
        ]

    def _render_performance_mode_results(self, mode: str, mode_blocks: list[tuple]) -> None:
        print("  \033[1mTRADING RESULTS BY MODE\033[0m  " + _ANSI_DIM + "Portfolio scope; no blended rows" + _ANSI_RST)
        printed_mode = False
        for mode_key, label, color, trades, all_trades in mode_blocks:
            if not trades and mode != mode_key:
                continue
            print(f"\n  {color}{label}{_ANSI_RST}")
            self._render_perf_period_rows(self._period_rows_for_trades(trades, all_trades), indent="    ")
            printed_mode = True
        if not printed_mode:
            print(f"    {_ANSI_DIM}No paper/live trade rows available.{_ANSI_RST}")

    def _render_per_symbol_mode_performance(self) -> None:
        if len(self.per_symbol_metrics) <= 1:
            return
        print("\n  \033[1mPER SYMBOL / MODE\033[0m")
        header = f"  {'Symbol':<10} {'Mode':<6} {'Trades':>7} {'Win%':>7} {'PnL $':>11} {'PF':>7} {'MaxDD':>7}"
        print(header)
        print("  " + "─" * (_visible_width(header) - 2))
        by_sym_mode: dict[tuple[str, str], list[dict]] = {}
        for (symbol, _tf, mode), trades in self.metrics_cube.items():
            by_sym_mode.setdefault((symbol, mode), []).extend(trades)
        for (symbol, mode), trades in sorted(by_sym_mode.items(), key=lambda item: (item[0][0], item[0][1])):
            self._render_per_symbol_mode_row(symbol, mode, trades)

    def _render_per_symbol_mode_row(self, symbol: str, mode: str, trades: list[dict]) -> None:
        metrics = _hud_period_metrics(trades, self._universe_starting_equity())
        trade_count = int(metrics.get("total_trades", 0) or 0)
        if trade_count <= 0:
            return
        pnl = metrics.get("total_pnl", 0)
        maxdd = metrics.get("max_drawdown", 0)
        dd_color = _ANSI_R if maxdd > DD_HIGH_PCT else (_ANSI_Y if maxdd > DD_WARN_PCT else _ANSI_G)
        mode_color = _ANSI_Y if mode == "paper" else _ANSI_G
        print(
            f"  {symbol:<10} {mode_color}{mode.upper():<6}{_ANSI_RST} {trade_count:>7} "
            f"{metrics.get('win_rate', 0) * 100:>6.1f}% "
            f"{self._pnl_color(pnl)}{pnl:>+11.2f}{_ANSI_RST} "
            f"{metrics.get('profit_factor', 0):>7.2f} {dd_color}{maxdd:>6.2f}%{_ANSI_RST}"
        )

    def _render_performance_quality_detail(self, mode: str, mode_blocks: list[tuple]) -> None:
        print(
            f"\n  \033[1mDETAIL DRILL-DOWN\033[0m  "
            f"{_ANSI_DIM}(same live trade source; still separated by mode){_ANSI_RST}"
        )
        for mode_key, label, color, trades, all_trades in mode_blocks:
            if not trades and mode != mode_key:
                continue
            print(f"\n  {color}{label} DETAIL{_ANSI_RST}")
            self._render_trade_quality(self._period_rows_for_trades(trades, all_trades))

    def _active_performance_quality_metrics(
        self,
        mode: str,
        active_by_mode: dict[str, list[dict]],
        active_all_by_mode: dict[str, list[dict]],
        portfolio_by_mode: dict[str, list[dict]],
    ) -> dict:
        if not self._has_active_scope():
            return _hud_period_metrics(portfolio_by_mode.get("paper", []), self._universe_starting_equity())
        print(
            f"\n  \033[1mACTIVE BOT DETAIL [{self._active_scope_label()}]\033[0m  "
            f"{_ANSI_DIM}(per-bot decision-learning scope){_ANSI_RST}"
        )
        active_mode_trades = []
        for mode_key, label, color, trades, all_trades in self._performance_mode_blocks(
            active_by_mode,
            active_all_by_mode,
        ):
            if not trades and mode != mode_key:
                continue
            print(f"  {color}{label}{_ANSI_RST}")
            rows = self._period_rows_for_trades(trades, all_trades)
            self._render_perf_period_rows(rows, indent="    ")
            self._render_trade_quality(rows)
            if not active_mode_trades:
                active_mode_trades = trades
        return _hud_period_metrics(active_mode_trades, self._universe_starting_equity())

    def _render_performance_detail(
        self,
        mode: str,
        mode_blocks: list[tuple],
        active_by_mode: dict[str, list[dict]],
        active_all_by_mode: dict[str, list[dict]],
        portfolio_by_mode: dict[str, list[dict]],
    ) -> None:
        if not self._performance_detail:
            print(f"\n  {_ANSI_DIM}Detail collapsed: [d] quality, edge, active bot, prediction.{_ANSI_RST}")
            return
        self._render_performance_quality_detail(mode, mode_blocks)
        quality_metrics = self._active_performance_quality_metrics(
            mode,
            active_by_mode,
            active_all_by_mode,
            portfolio_by_mode,
        )
        self._render_trade_timing(quality_metrics, self.production_metrics.get("metrics", {}))

    def _render_performance(self) -> None:
        """Render summary-first performance metrics — dispatches by drill level."""
        self._render_breadcrumb("PERFORMANCE", 2)
        mode = self._performance_mode()
        self._render_performance_intro()
        portfolio_by_mode = {
            mode_key: list(self._trade_log_metrics_trades_by_mode.get(mode_key, []))
            for mode_key in ("paper", "live")
        }
        portfolio_all_by_mode = {
            mode_key: list(self._trade_log_all_trades_by_mode.get(mode_key, []))
            for mode_key in ("paper", "live")
        }
        active_by_mode = {
            mode_key: self._active_scope_trades(self._trade_log_metrics_trades_by_mode.get(mode_key, []))
            for mode_key in ("paper", "live")
        }
        active_all_by_mode = {
            mode_key: self._active_scope_trades(self._trade_log_all_trades_by_mode.get(mode_key, []))
            for mode_key in ("paper", "live")
        }
        mode_blocks = self._performance_mode_blocks(portfolio_by_mode, portfolio_all_by_mode)
        self._render_performance_period_summary()
        self._render_performance_mode_results(mode, mode_blocks)
        self._render_offline_improvement_summary()
        self._render_current_session_performance()
        self._render_per_symbol_mode_performance()
        self._render_mode_breakdown()
        self._render_timeframe_mode_breakdown()
        self._render_performance_detail(
            mode,
            mode_blocks,
            active_by_mode,
            active_all_by_mode,
            portfolio_by_mode,
        )

    def _render_offline_improvement_summary(self) -> None:
        """Render offline training/champion evidence without mixing it into account PnL."""
        _results = self.offline_stats.get("results", []) if isinstance(self.offline_stats, dict) else []
        _status = (
            self._offline_status_normalized(self.offline_stats)
            if isinstance(self.offline_stats, dict)
            else "idle"
        )
        _counts: dict[str, int] = {}
        for _row in _results if isinstance(_results, list) else []:
            _key = str(_row.get("status", "unknown") or "unknown").lower()
            _counts[_key] = _counts.get(_key, 0) + 1
        _champions = []
        _champ_path = self.data_dir / "checkpoints" / "offline_champions.json"
        if _champ_path.exists():
            try:
                _champions = list((json.loads(_champ_path.read_text()).get("champions") or {}).values())
            except Exception:
                LOG.debug("[HUD] Failed to read offline champions for performance summary", exc_info=True)
        _accepted = sum(1 for _c in _champions if float(_c.get("z_omega", 0.0) or 0.0) > 0.0)
        _best = max(_champions, key=lambda _c: float(_c.get("z_omega", 0.0) or 0.0), default={})
        _best_sym = str(_best.get("symbol", "—") or "—").upper()
        _best_tf = self._format_timeframe_minutes_label(_best.get("timeframe_minutes", 0))
        _best_zo = float(_best.get("z_omega", 0.0) or 0.0)
        _summary = " ".join(f"{k}:{v}" for k, v in sorted(_counts.items())) or "no active jobs"
        print(f"\n  \033[1mOFFLINE RESULTS\033[0m  {_ANSI_DIM}Training/validation evidence; not account PnL{_ANSI_RST}")
        print(
            f"    Status: {_status:<9} Jobs: {_summary:<28} "
            f"Champions: {_accepted:>2}  Best: {_best_sym}/{_best_tf} ZΩ {_best_zo:.3f}"
        )

    def _render_current_session_performance(self) -> None:
        """Render current process-session stats from paper_stats_*.json."""
        if not self.all_bots_stats:
            return
        print(
            "\n  \033[1mCURRENT BOT SESSIONS\033[0m  " + _ANSI_DIM + "(runtime counters; reset on restart)" + _ANSI_RST
        )
        _hdr = f"  {'Bot':<13} {'Mode':<6} {'Trades':>7} {'Win%':>7} {'PnL $':>13} {'CapReward':>10}"
        print(_hdr)
        print("  " + "─" * (_visible_width(_hdr) - 2))
        for bot in sorted(
            self.all_bots_stats,
            key=lambda b: (str(b.get("symbol", "")).upper(), int(b.get("timeframe_minutes", 0) or 0)),
        ):
            sym = str(bot.get("symbol", "?")).upper()
            tf = int(bot.get("timeframe_minutes", 0) or 0)
            label = f"{sym}/M{tf}"
            if len(label) > 13:
                label = label[:12] + "…"
            mode = str(bot.get("trading_mode", "") or "paper").upper()[:6]
            trades = int(bot.get("total_trades", 0) or 0)
            win_pct = float(bot.get("win_rate", 0.0) or 0.0) * 100.0
            pnl = float(bot.get("total_pnl", 0.0) or 0.0)
            cap_reward = bot.get("reward_shaping", {}).get("components", {}).get("capture", {}).get("avg", 0.0)
            print(
                f"  {label:<13} {mode:<6} {trades:>7} {win_pct:>6.1f}% "
                f"{self._pnl_color(pnl)}{pnl:>+13.2f}{_ANSI_RST} {float(cap_reward or 0.0):>10.3f}"
            )

    def _render_mode_breakdown(self) -> None:
        """Show paper vs live trade breakdown using canonical trade-log grouping."""
        trades = self._trade_log_metrics_trades
        if not trades:
            return
        paper_trades = [t for t in trades if t.get("trading_mode") == "paper"]
        live_trades = [t for t in trades if t.get("trading_mode") == "live"]

        if not paper_trades and not live_trades:
            return

        print("\n  \033[1mMODE BREAKDOWN (PORTFOLIO)\033[0m")
        _mb_hdr = f"  {'Mode':<8} {'Trades':>7} {'Win%':>7} {'PnL $':>11}"
        _mb_sep = "  " + "\u2500" * (_visible_width(_mb_hdr) - 2)
        print(_mb_hdr)
        print(_mb_sep)
        for label, trades, color in [
            ("Paper", paper_trades, _ANSI_Y),
            ("Live", live_trades, _ANSI_G),
        ]:
            m = _hud_period_metrics(trades, self._universe_starting_equity())
            n = m.get("total_trades", 0)
            if n == 0:
                continue
            wr = m.get("win_rate", 0.0) * 100
            total_pnl = m.get("total_pnl", 0.0)
            pnl_c = self._pnl_color(total_pnl)
            print(f"  {color}{label:<8}{_ANSI_RST} {n:>7} {wr:>6.1f}% {pnl_c}{total_pnl:>+11.2f}{_ANSI_RST}")
        print(_mb_sep)

    def _render_timeframe_mode_breakdown(self) -> None:
        """Render canonical symbol/timeframe/mode table from metrics cube."""
        # Build set of (sym, tf_label, mode) present in cube
        _keys = set(self.metrics_cube_keys)

        # Supplement with all running bots (so zero-trade TFs still appear)
        for _bot in self.all_bots_stats or []:
            _sym = self._normalize_symbol(_bot.get("symbol"))
            _tfm = int(_bot.get("timeframe_minutes", 0) or 0)
            if not _sym or _tfm <= 0:
                continue
            _tf_label = f"M{_tfm}"
            _mode = str(_bot.get("trading_mode", "") or "").strip().lower()
            if _mode not in ("paper", "live"):
                _mode = "paper" if bool(_bot.get("paper_mode", True)) else "live"
            _keys.add((_sym, _tf_label, _mode))

        if not _keys:
            return

        print("\n  \033[1mPER SYMBOL / TIMEFRAME / MODE\033[0m")
        _tfm_hdr = f"  {'Symbol':<9} {'TF':<6} {'Mode':<6} {'Trades':>7} {'Win%':>7} {'PnL $':>11} {'PF':>7}"
        _tfm_sep = "  " + "\u2500" * (_visible_width(_tfm_hdr) - 2)
        print(_tfm_hdr)
        print(_tfm_sep)
        for _sym, _tf, _mode in sorted(_keys, key=lambda k: (k[0], self._timeframe_sort_key(k[1]), k[2])):
            _trades = self.metrics_cube.get((_sym, _tf, _mode), [])
            if _trades:
                _m = _hud_period_metrics(_trades, self._universe_starting_equity())
                _tr = _m.get("total_trades", 0)
                _wr = _m.get("win_rate", 0) * 100
                _pnl = _m.get("total_pnl", 0.0)
                _pf = _m.get("profit_factor", 0.0)
            else:
                _tr, _wr, _pnl, _pf = 0, 0.0, 0.0, 0.0
            _mc = _ANSI_Y if _mode == "paper" else _ANSI_G
            _pc = self._pnl_color(_pnl) if _tr > 0 else _ANSI_DIM
            print(
                f"  {_sym:<9} {_tf:<6} {_mc}{_mode.upper():<6}{_ANSI_RST} {_tr:>7} {_wr:>6.1f}% "
                f"{_pc}{_pnl:>+11.2f}{_ANSI_RST} {_pf:>7.2f}"
            )
        print(_tfm_sep)

    def _render_trade_quality(self, rows: list[tuple[str, dict]]) -> None:
        """Render trade quality & edge quality tables per period."""
        if not rows:
            return
        self._render_trade_quality_table(rows)
        self._render_edge_quality_table(rows)

    def _render_trade_quality_table(self, rows: list[tuple[str, dict]]) -> None:
        print(f"\n\033[1m📊 TRADE QUALITY\033[0m  {_ANSI_DIM}(per period){_ANSI_RST}\n")
        header = (
            f"  {'Period':<9} {'Trades':>7} {'Payoff':>8} {'PF':>7} {'Expect':>10} "
            f"{'Sortino':>8} {'Streak':>7} {'W→L':>8}"
        )
        print(header)
        print("  " + "─" * (_visible_width(header) - 2))
        for label, lt in rows:
            self._render_trade_quality_row(label, lt)
        print(
            f"  {_ANSI_DIM}(Payoff target ≥1.5; PF target ≥1.2; Streak shows max wins/losses){_ANSI_RST}"
        )

    def _render_trade_quality_row(self, label: str, metrics: dict) -> None:
        total = metrics.get("total_trades", 0)
        profit_f = metrics.get("profit_factor", 0.0)
        expect = metrics.get("expectancy", 0.0)
        payoff = self._quality_payoff(metrics)
        w2l_color, w2l_str = self._winner_loser_signal(total, metrics.get("winner_to_loser_count", 0))
        max_wins = metrics.get("max_consec_wins", 0)
        max_losses = metrics.get("max_consec_losses", 0)
        streak_color = _ANSI_R if max_losses >= 5 else (_ANSI_G if max_wins >= 3 else _ANSI_Y)
        print(
            f"  {label:<9} {total:>7} "
            f"{self._payoff_color(payoff)}{payoff:>7.2f}x{_ANSI_RST} "
            f"{self._profit_factor_color(profit_f)}{profit_f:>7.2f}{_ANSI_RST} "
            f"{_ANSI_G if expect > 0 else _ANSI_R}{expect:>+10.3f}{_ANSI_RST} "
            f"{metrics.get('sortino_ratio', 0.0):>8.3f} "
            f"{streak_color}{f'{max_wins}/{max_losses}':>7}{_ANSI_RST} "
            f"{w2l_color}{w2l_str:>8}{_ANSI_RST}"
        )

    @staticmethod
    def _quality_payoff(metrics: dict) -> float:
        abs_loss = abs(metrics.get("avg_loss", 0.0))
        if abs_loss <= _PAYOFF_FLOOR:
            return 0.0
        return metrics.get("avg_win", 0.0) / abs_loss

    @staticmethod
    def _payoff_color(payoff: float) -> str:
        if payoff >= PAYOFF_GOOD_MIN:
            return _ANSI_G
        if payoff >= 1.0:
            return _ANSI_Y
        return _ANSI_R

    @staticmethod
    def _profit_factor_color(profit_factor: float) -> str:
        if profit_factor >= PROFIT_FACTOR_GOOD_MIN:
            return _ANSI_G
        if profit_factor >= 1.0:
            return _ANSI_Y
        return _ANSI_R

    @staticmethod
    def _winner_loser_signal(total: int, w2l: int) -> tuple[str, str]:
        if total <= 0 or w2l <= 0:
            return _ANSI_DIM, "-"
        w2l_pct = w2l / total * 100
        color = _ANSI_R if w2l_pct > 15 else (_ANSI_Y if w2l_pct > 5 else _ANSI_G)
        return color, f"{w2l}/{w2l_pct:.0f}%"

    def _render_edge_quality_table(self, rows: list[tuple[str, dict]]) -> None:
        print(f"\n\033[1m🔬 EDGE QUALITY\033[0m  {_ANSI_DIM}(model tuning signals; per period){_ANSI_RST}\n")
        header = (
            f"  {'Period':<10} {'Capture':>8} {'AvgMFE':>10} {'AvgMAE':>10} {'Edge':>10} "
            f"{'Bars':>6} {'ConfW':>7} {'ConfL':>7} {'Gap':>8}"
        )
        print(header)
        print("  " + "─" * (_visible_width(header) - 2))
        for label, lt in rows:
            self._render_edge_quality_row(label, lt)
        print(
            f"  {_ANSI_DIM}(Capture exit_pnl/MFE target ≥60%; Edge MFE-MAE; "
            f"Gap conf_win-conf_loss +ve=calibrated){_ANSI_RST}"
        )

    def _render_edge_quality_row(self, label: str, metrics: dict) -> None:
        avg_mfe = metrics.get("avg_mfe", 0.0)
        avg_mae = metrics.get("avg_mae", 0.0)
        cap_ratio = metrics.get("avg_capture_ratio", 0.0)
        conf_w = metrics.get("avg_conf_win", 0.0)
        conf_l = metrics.get("avg_conf_loss", 0.0)
        edge = avg_mfe - avg_mae if avg_mfe > 0 else 0.0
        cal_gap = conf_w - conf_l
        print(
            f"  {label:<10} "
            f"{self._capture_color(cap_ratio)}{cap_ratio:>7.1%}{_ANSI_RST} "
            f"${avg_mfe:>+9.2f} ${avg_mae:>9.2f} "
            f"{_ANSI_G if edge > 0 else _ANSI_R}${edge:>+9.2f}{_ANSI_RST} "
            f"{metrics.get('avg_bars_held', 0.0):>6.1f} "
            f"{conf_w:>7.3f} {conf_l:>7.3f} "
            f"{self._calibration_gap_color(conf_w, conf_l, cal_gap)}{cal_gap:>+8.3f}{_ANSI_RST}"
        )

    @staticmethod
    def _capture_color(capture_ratio: float) -> str:
        if capture_ratio >= 0.60:
            return _ANSI_G
        if capture_ratio >= 0.40:
            return _ANSI_Y
        return _ANSI_R

    @staticmethod
    def _calibration_gap_color(conf_w: float, conf_l: float, gap: float) -> str:
        if conf_w <= 0 and conf_l <= 0:
            return _ANSI_DIM
        if gap > 0.05:
            return _ANSI_G
        if gap > 0:
            return _ANSI_Y
        return _ANSI_R

    def _compute_trade_log_convergence_metrics(self, trades: list[dict] | None = None) -> dict:
        trades = list(trades) if trades is not None else self._trade_log_metrics_trades
        rw_delta = 0.0
        rw_acc = 0.5
        cc_err = 0.5
        rw_n = 0
        cc_n = 0
        util_sum = 0.0
        err_pct_sum = 0.0
        delta_sum = 0.0
        acc_sum = 0.0
        brier_sum = 0.0
        for t in trades:
            _ec = t.get("entry_confidence")
            if _ec is not None:
                _outcome = 1.0 if float(t.get("pnl", 0.0)) > 0.0 else 0.0
                _brier = (float(_ec) - _outcome) ** 2
                cc_err = (1 - CONV_EMA_ALPHA) * cc_err + CONV_EMA_ALPHA * _brier
                brier_sum += _brier
                cc_n += 1
            _pred_frac = float(t.get("predicted_runway", 0.0) or 0.0)
            _entry_price = float(t.get("entry_price", 0.0) or 0.0)
            _actual_mfe = t.get("mfe")
            _pred_pts_raw = t.get("predicted_runway_net_points_raw")
            _pred_pts_adj = t.get("predicted_runway_net_points")
            if _actual_mfe is None:
                continue
            if _pred_pts_adj is not None:
                _pred_pts = float(_pred_pts_adj or 0.0)
            elif _pred_pts_raw is not None:
                _pred_pts = float(_pred_pts_raw or 0.0)
            elif _pred_frac > 0.0 and _entry_price > 0.0:
                _pred_pts = _pred_frac * _entry_price
            else:
                continue
            if _pred_pts <= 0.0:
                continue
            _actual_mfe_f = float(_actual_mfe)
            _delta = _pred_pts - _actual_mfe_f
            _max_err = max(abs(_actual_mfe_f), abs(_pred_pts), 1.0)
            _acc = 1.0 - min(abs(_delta) / _max_err, 1.0)
            rw_delta = (1 - CONV_EMA_ALPHA) * rw_delta + CONV_EMA_ALPHA * _delta
            rw_acc = (1 - CONV_EMA_ALPHA) * rw_acc + CONV_EMA_ALPHA * _acc
            rw_n += 1
            delta_sum += _delta
            acc_sum += _acc
            util_sum += _actual_mfe_f / _pred_pts
            err_pct_sum += (abs(_delta) / _pred_pts) * 100.0
        return {
            "runway_delta_ema": rw_delta,
            "runway_accuracy_ema": rw_acc,
            "conf_calib_err_ema": cc_err,
            "avg_runway_delta": (delta_sum / rw_n) if rw_n else 0.0,
            "avg_runway_accuracy": (acc_sum / rw_n) if rw_n else 0.0,
            "avg_conf_brier": (brier_sum / cc_n) if cc_n else 0.0,
            "runway_samples": rw_n,
            "conf_samples": cc_n,
            "avg_runway_utilization": (util_sum / rw_n) if rw_n else 0.0,
            "avg_runway_error_pct": (err_pct_sum / rw_n) if rw_n else 0.0,
            "trade_samples": len(trades),
        }

    def _resolve_prediction_convergence_metrics(self, pm: dict) -> dict:
        _mode = getattr(self, "_perf_snapshot_mode", "")
        _trades = self._active_scope_trades(self._trade_log_metrics_trades)
        if _mode in ("paper", "live"):
            _trades = [t for t in _trades if str(t.get("trading_mode", "") or "").lower() == _mode]

        _active_trades = _trades
        _active_tf = int(getattr(self, "active_tf_min", 0) or 0)
        _active_sym = str(getattr(self, "active_sym", "") or "").upper()
        if _active_tf > 0:
            _active_label = f"M{_active_tf}"
            _active_trades = [
                t
                for t in _trades
                if self._normalize_timeframe_label(t) == _active_label
                and (_active_sym == "" or self._normalize_symbol(t.get("symbol")) == _active_sym)
            ]

        tl_active = self._compute_trade_log_convergence_metrics(_active_trades)
        use_active_runway = tl_active["runway_samples"] >= CONV_MIN_SAMPLES
        use_active_conf = tl_active["conf_samples"] >= CONV_MIN_SAMPLES

        if use_active_runway:
            runway_val = tl_active["runway_delta_ema"]
            acc_val = tl_active["runway_accuracy_ema"]
            runway_source = "trade_log.jsonl(active_tf)"
            runway_samples = tl_active["runway_samples"]
            avg_util = tl_active["avg_runway_utilization"]
            avg_err = tl_active["avg_runway_error_pct"]
        else:
            runway_val = float(pm.get("runway_delta_ema", 0.0))
            acc_val = float(pm.get("runway_accuracy_ema", 0.5))
            runway_source = "production_metrics.json(active_scope)"
            runway_samples = tl_active["runway_samples"]
            avg_util = tl_active["avg_runway_utilization"]
            avg_err = tl_active["avg_runway_error_pct"]

        if use_active_conf:
            conf_val = tl_active["conf_calib_err_ema"]
            conf_source = "trade_log.jsonl(active_tf)"
            conf_samples = tl_active["conf_samples"]
        else:
            conf_val = float(pm.get("conf_calib_err_ema", 0.5))
            conf_source = "production_metrics.json(active_scope)"
            conf_samples = tl_active["conf_samples"]

        return {
            "runway_delta_ema": runway_val,
            "runway_accuracy_ema": acc_val,
            "conf_calib_err_ema": conf_val,
            "runway_source": runway_source,
            "conf_source": conf_source,
            "runway_samples": runway_samples,
            "conf_samples": conf_samples,
            "avg_runway_utilization": avg_util,
            "avg_runway_error_pct": avg_err,
            "trade_samples": tl_active["trade_samples"],
        }

    def _render_trade_timing(self, lt: dict, pm: dict) -> None:
        """Render trade timing and prediction convergence."""
        avg_dur = lt.get("avg_trade_duration_mins") or pm.get("avg_trade_duration_mins", 0.0)
        last_trade = lt.get("last_trade_mins_ago") or pm.get("last_trade_mins_ago", 0.0)
        _timing_source = (
            "trade_log.jsonl"
            if lt.get("avg_trade_duration_mins") or lt.get("last_trade_mins_ago")
            else "production_metrics.json"
        )
        print(
            f"\n  Avg hold time:    {self._format_duration(avg_dur):>8}  "
            f"{_ANSI_DIM}(source: {_timing_source}){_ANSI_RST}"
        )
        print(f"  Last trade:       {self._format_duration(last_trade):>8} ago")
        self._render_prediction_convergence(pm)

    def _format_duration(self, mins: float) -> str:
        """Format minutes into a compact duration label."""
        if mins <= 0:
            return "—"
        if mins < _DURATION_HOUR_MINS:
            return f"{mins:.0f}m"
        if mins < _DURATION_DAY_MINS:
            return f"{mins / 60:.1f}h"
        return f"{mins / _DURATION_DAY_MINS:.1f}d"

    def _prediction_convergence_rows(self, trades: list[dict], all_trades: list[dict]) -> list[tuple[str, dict]]:
        """Compute per-period convergence metrics for a given trades scope."""
        daily, weekly, monthly = _classify_trades_by_period(trades)
        rows: list[tuple[str, dict]] = [
            ("24h", self._compute_trade_log_convergence_metrics(daily)),
            ("7d", self._compute_trade_log_convergence_metrics(weekly)),
            ("Month", self._compute_trade_log_convergence_metrics(monthly)),
            (
                "Epoch" if self._stats_epoch else "Lifetime",
                self._compute_trade_log_convergence_metrics(trades),
            ),
        ]
        if self._stats_epoch and all_trades:
            rows.append(("Lifetime", self._compute_trade_log_convergence_metrics(all_trades)))
        return rows

    def _render_prediction_convergence_table(self, rows: list[tuple[str, dict]]) -> None:
        """Render a per-period convergence table for the provided rows."""
        header = f"  {'Period':<10} {'n':>5} {'rwΔ pts':>10} {'Accuracy':>9} {'Brier':>8} {'Util':>8} {'Err %':>8}"
        print(header)
        print("  " + "─" * (_visible_width(header) - 2))
        for label, conv in rows:
            self._render_prediction_convergence_row(label, conv)

    def _render_prediction_convergence_row(self, label: str, conv: dict) -> None:
        rw_n = conv["runway_samples"]
        cc_n = conv["conf_samples"]
        if rw_n == 0 and cc_n == 0:
            print(f"  {label:<10} {_ANSI_DIM}{0:>5}      —         —        —        —        —{_ANSI_RST}")
            return
        rw_delta = conv["avg_runway_delta"]
        rw_acc = conv["avg_runway_accuracy"]
        cc_err = conv["avg_conf_brier"]
        util = conv["avg_runway_utilization"]
        err_pct = conv["avg_runway_error_pct"]
        util_color, err_color = self._runway_util_error_colors(rw_n, util, err_pct)
        print(
            f"  {label:<10} {max(rw_n, cc_n):>5} "
            f"{self._runway_delta_color(rw_delta)}{rw_delta:>+9.2f}{_ANSI_RST} "
            f"{self._runway_accuracy_color(rw_acc)}{rw_acc:>9.3f}{_ANSI_RST} "
            f"{self._confidence_brier_color(cc_err)}{cc_err:>8.3f}{_ANSI_RST} "
            f"{util_color}{util:>7.3f}x{_ANSI_RST} "
            f"{err_color}{err_pct:>7.1f}%{_ANSI_RST}"
        )

    @staticmethod
    def _runway_delta_color(delta: float) -> str:
        if abs(delta) < RUNWAY_DELTA_OK_MAX:
            return _ANSI_G
        if abs(delta) < RUNWAY_DELTA_WARN_MAX:
            return _ANSI_Y
        return _ANSI_R

    @staticmethod
    def _runway_accuracy_color(accuracy: float) -> str:
        if accuracy > RUNWAY_ACCURACY_GOOD:
            return _ANSI_G
        if accuracy > RUNWAY_ACCURACY_WARN:
            return _ANSI_Y
        return _ANSI_R

    @staticmethod
    def _confidence_brier_color(brier: float) -> str:
        if brier < CONF_CALIB_OK_MAX:
            return _ANSI_G
        if brier < CONF_CALIB_WARN_MAX:
            return _ANSI_Y
        return _ANSI_R

    @staticmethod
    def _runway_util_error_colors(rw_n: int, util: float, err_pct: float) -> tuple[str, str]:
        if rw_n == 0:
            return _ANSI_DIM, _ANSI_DIM
        util_color = _ANSI_G if util >= 1.0 else (_ANSI_Y if util >= 0.7 else _ANSI_R)
        err_color = _ANSI_G if err_pct <= 25.0 else (_ANSI_Y if err_pct <= 50.0 else _ANSI_R)
        return util_color, err_color

    def _render_prediction_convergence(self, pm: dict) -> None:
        """Render prediction convergence metrics as per-period tables."""
        _mode = (
            getattr(self, "_trade_log_mode", "")
            or getattr(self, "_perf_snapshot_mode", "")
            or self.bot_config.get("trading_mode", "paper")
        )
        platt_a = pm.get("platt_a", 1.0)
        platt_b = pm.get("platt_b", 0.0)
        if abs(platt_a - 1.0) > PLATT_ADAPTED_DELTA or abs(platt_b) > PLATT_ADAPTED_DELTA:
            pa_col = _ANSI_B
        else:
            pa_col = _ANSI_DIM

        print(
            f"\n\033[1m🎯 PREDICTION CONVERGENCE"
            f"{' [' + self._active_scope_label() + ']' if self._has_active_scope() else ''}\033[0m  "
            f"{_ANSI_DIM}(per-period means; src=trade_log.jsonl){_ANSI_RST}\n"
        )

        if _mode == "mixed":
            print(f"  {_ANSI_Y}📄 PAPER{_ANSI_RST}")
            self._render_prediction_convergence_table(
                self._prediction_convergence_rows(
                    self._active_scope_trades(self._trade_log_metrics_trades_by_mode.get("paper", [])),
                    self._active_scope_trades(self._trade_log_all_trades_by_mode.get("paper", [])),
                )
            )
            print(f"\n  {_ANSI_G}💰 LIVE{_ANSI_RST}")
            self._render_prediction_convergence_table(
                self._prediction_convergence_rows(
                    self._active_scope_trades(self._trade_log_metrics_trades_by_mode.get("live", [])),
                    self._active_scope_trades(self._trade_log_all_trades_by_mode.get("live", [])),
                )
            )
        else:
            _mk = _mode if _mode in ("paper", "live") else None
            _scoped = self._trade_log_metrics_trades_by_mode.get(_mk, []) if _mk else self._trade_log_metrics_trades
            _all = self._trade_log_all_trades_by_mode.get(_mk, []) if _mk else self._trade_log_all_trades
            self._render_prediction_convergence_table(
                self._prediction_convergence_rows(self._active_scope_trades(_scoped), self._active_scope_trades(_all))
            )

        print(
            f"\n  {_ANSI_DIM}(rwΔ pred-actual pts; Acc 1=best; Brier 0=best; "
            f"Util actual/pred; Err mean abs %){_ANSI_RST}"
        )
        print(
            f"  Platt  a={pa_col}{platt_a:.4f}{_ANSI_RST}  "
            f"b={pa_col}{platt_b:+.4f}{_ANSI_RST}  "
            f"{_ANSI_DIM}(source: production_metrics.json; grey=default, blue=adapted){_ANSI_RST}"
        )

    def _render_jsonl_decision_entries(self, entries: list, mode_filter: str = "", cursor_idx: int = -1) -> None:
        """Render the rich JSONL decision log entries.

        Collapses consecutive CLOSE_PENDING rows into a single summary line
        and surfaces reward-relevant context (regime, runway, capture, PnL).

        Columns:
          Date+Time | Bot | Mode | Agent | Decision | Conf | Detail (varies by decision type)
        """
        # ── Pre-filter: keep only selected mode when mixed ───────────────────
        if mode_filter in ("paper", "live"):
            entries = [e for e in entries if e.get("trading_mode") == mode_filter]

        def _f(v: object, d: float = 0.0) -> float:
            try:
                return float(v) if v is not None else d
            except (TypeError, ValueError):
                return d

        _target_width = max(96, self._term_width())

        # ── Pre-pass: collapse CLOSE_PENDING runs ─────────────────────────
        collapsed: list[dict | str] = []  # dict = normal entry, str = summary line
        i = 0
        while i < len(entries):
            e = entries[i]
            if e.get("decision", "").upper() == "CLOSE_PENDING":
                run_start = i
                tid = e.get("trade_id", "?")
                while i < len(entries) and entries[i].get("decision", "").upper() == "CLOSE_PENDING":
                    i += 1
                run_len = i - run_start
                last = entries[i - 1]
                _rsn = last.get("reasoning", {})
                _ctx = last.get("context", {})
                _mfe = _f(_rsn.get("mfe"))
                _mae = _f(_rsn.get("mae"))
                _upnl = _f(_ctx.get("unrealized_pnl"))
                _pnl_c = _ANSI_G if _upnl >= 0 else _ANSI_R
                collapsed.append(
                    f"  {_ANSI_DIM}   ... Harvester held {run_len} bars (TrdID:{tid[:8]})  "
                    f"MFE:{_ANSI_G}+{_mfe:.2f}{_ANSI_DIM}  MAE:{_ANSI_R}-{_mae:.2f}{_ANSI_DIM}  "
                    f"uPnL:{_pnl_c}{_upnl:+.2f}{_ANSI_DIM}{_ANSI_RST}"
                )
            else:
                collapsed.append(e)
                i += 1

        # ── Distribution (only non-CLOSE_PENDING decisions) ───────────────
        _counts: dict[str, int] = {}
        _total_entries = 0
        for item in collapsed:
            if isinstance(item, dict):
                _d = item.get("decision", "?").upper()
                _counts[_d] = _counts.get(_d, 0) + 1
                _total_entries += 1
        _dist = "  ".join(f"{k}:{v}" for k, v in sorted(_counts.items()))
        print(f"  Decisions: {_dist}  ({len(entries) - _total_entries} held bars collapsed)\n")

        # ── Header ────────────────────────────────────────────────────────
        header = f"  {'Time':<12} {'Bot':<13} {'Mode':<5} {'Agent':<10} {'Decision':<10} {'Conf':>5}  {'Detail'}"
        print(header)
        print("  " + "─" * (_visible_width(header) - 2))

        _seen_sessions: set[str] = set()
        _row_idx = 0  # tracks only dict entries for cursor matching
        for item in collapsed:
            # Collapsed CLOSE_PENDING summary line
            if isinstance(item, str):
                print(item)
                continue

            entry = item
            _is_cursor = cursor_idx >= 0 and _row_idx == cursor_idx
            _row_idx += 1

            # ── Session break header ───────────────────────────────────────
            # Only emit the separator the first time we encounter a session in
            # this render (entries from multiple bot files may interleave by
            # timestamp; without this guard the same session banner repeats).
            _sess = entry.get("session", "")
            if _sess and _sess not in _seen_sessions:
                _seen_sessions.add(_sess)
                print(f"  {_ANSI_DIM}── session {_sess} ──{_ANSI_RST}")

            # ── Timestamp ──────────────────────────────────────────────────
            ts_raw = entry.get("timestamp", "?")
            try:
                ts_str = ts_raw[5:16] if len(ts_raw) >= _DEC_LOG_TS_MIN_LEN else ts_raw[:11]
            except Exception:
                ts_str = str(ts_raw)[:11]

            # ── Mode badge ─────────────────────────────────────────────────
            # Fixed 5-cell tag (no emoji) so the column stays aligned regardless
            # of the terminal's wide-character handling.
            _mode = entry.get("trading_mode", "")
            if _mode == "paper":
                mode_str = f"{_ANSI_Y}{'PPR':<5}{_ANSI_RST}"
            elif _mode == "live":
                mode_str = f"{_ANSI_G}{'LIV':<5}{_ANSI_RST}"
            else:
                mode_str = f"{_ANSI_DIM}{'?':<5}{_ANSI_RST}"

            agent = entry.get("agent", "?")[:9]
            decision = entry.get("decision", "?")
            conf = _f(entry.get("confidence"))
            bot_str = self._decision_entry_bot_label(entry)[:13]
            ctx = entry.get("context", {})
            if not isinstance(ctx, dict):
                ctx = {}
            reasoning = entry.get("reasoning", {})
            if not isinstance(reasoning, dict):
                reasoning = {}

            # ── Build decision-specific detail string ──────────────────────
            dec_upper = decision.upper()
            if dec_upper in ("LONG", "SHORT"):
                # Entry: show regime, feasibility, predicted_runway, VPIN-z, Q-spread
                regime = (ctx.get("regime") or "?")[:5]
                feas = _f(reasoning.get("feasibility"))
                runway = _f(reasoning.get("predicted_runway"))
                vpin_z = _f(ctx.get("vpin_z"))
                qs = _f(reasoning.get("q_spread"))
                tid = entry.get("trade_id", "")[:8]
                vpin_flag = f"{_ANSI_R}!{_ANSI_RST}" if abs(vpin_z) > _DEC_LOG_VPIN_WARN else " "
                feas_c = _ANSI_G if feas >= 0.6 else (_ANSI_Y if feas >= 0.3 else _ANSI_R)
                detail = (
                    f"ζ:{regime} F:{feas_c}{feas:.2f}{_ANSI_RST} "
                    f"rwy:{runway:.4f} vz:{vpin_z:+.1f}{vpin_flag} "
                    f"QΔ:{qs:.3f} [{tid}]"
                )
            elif dec_upper == "NO_ENTRY":
                # Rejected entry: show WHY (feasibility, regime, VPIN-z, gated_conditions)
                regime = (ctx.get("regime") or "?")[:5]
                feas = _f(reasoning.get("feasibility"))
                vpin_z = _f(ctx.get("vpin_z"))
                cb_ok = reasoning.get("circuit_breakers_ok", True)
                feas_c = _ANSI_R if feas < 0.3 else (_ANSI_Y if feas < 0.6 else _ANSI_G)
                cb_str = f" {_ANSI_R}CB!{_ANSI_RST}" if not cb_ok else ""
                gated = reasoning.get("gated_conditions") or []
                if isinstance(gated, list) and gated:
                    _gate_summary = f" [{_ANSI_R}{len(gated)}gate{'s' if len(gated) != 1 else ''}{_ANSI_RST}]"
                else:
                    _gate_summary = ""
                detail = f"ζ:{regime} F:{feas_c}{feas:.2f}{_ANSI_RST} vz:{vpin_z:+.1f}{cb_str}{_gate_summary}"
            elif dec_upper == "CLOSE":
                # Exit: show capture_ratio, MFE, MAE, unrealized PnL, Q-spread
                cap = _f(reasoning.get("capture_ratio"))
                mfe = _f(reasoning.get("mfe"))
                mae = _f(reasoning.get("mae"))
                upnl = _f(ctx.get("unrealized_pnl"))
                qs = _f(reasoning.get("q_spread"))
                tid = entry.get("trade_id", "")[:8]
                cap_c = _ANSI_G if cap >= 0.6 else (_ANSI_Y if cap >= 0.3 else _ANSI_R)
                pnl_c = _ANSI_G if upnl >= 0 else _ANSI_R
                detail = (
                    f"cap:{cap_c}{cap:+.2f}{_ANSI_RST} "
                    f"MFE:{_ANSI_G}+{mfe:.2f}{_ANSI_RST} "
                    f"MAE:{_ANSI_R}-{mae:.2f}{_ANSI_RST} "
                    f"uPnL:{pnl_c}{upnl:+.1f}{_ANSI_RST} Q\u0394:{qs:.3f} [{tid}]"
                )
            elif dec_upper == "HOLD":
                # Hold: show ticks_held, capture_ratio trajectory
                ticks = int(_f(reasoning.get("ticks_held")))
                cap = _f(reasoning.get("capture_ratio"))
                upnl = _f(ctx.get("unrealized_pnl"))
                pnl_c = _ANSI_G if upnl >= 0 else _ANSI_R
                detail = f"bars:{ticks} cap:{cap:+.2f} uPnL:{pnl_c}{upnl:+.1f}{_ANSI_RST}"
            else:
                price = _f(ctx.get("price"))
                detail = f"@ {price:.2f}"

            # ── Color by decision type ─────────────────────────────────────
            if dec_upper in ("BUY", "LONG", "ENTER"):
                color = _ANSI_G
            elif dec_upper in ("SELL", "SHORT", "EXIT", "CLOSE"):
                color = _ANSI_R
            elif dec_upper == "HOLD":
                color = _ANSI_Y
            elif dec_upper == "NO_ENTRY":
                color = _ANSI_DIM
            else:
                color = _ANSI_RST

            _cursor_pfx = f"{_ANSI_G}►{_ANSI_RST}" if _is_cursor else " "
            _prefix = (
                f"{_cursor_pfx} {ts_str:<12} {bot_str:<13} {mode_str} {agent:<10} "
                f"{color}{dec_upper:<10}{_ANSI_RST} {conf:>5.3f}  "
            )
            detail = _truncate_visible(detail, max(12, _target_width - _visible_width(_prefix)))
            print(f"{_prefix}{detail}")
            # Expand gated_conditions inline for NO_ENTRY at L3+ (symbol/TF scope)
            if dec_upper == "NO_ENTRY" and getattr(self, "_ctx_level", 1) >= 3:
                _gated = reasoning.get("gated_conditions") or []
                if isinstance(_gated, list) and _gated:
                    for _g in _gated[:6]:
                        print(f"  {'':38}{_ANSI_R}  ✗ {_g}{_ANSI_RST}")
        print("  " + "─" * (_visible_width(header) - 2))

    def _render_legacy_decision_entries(self, entries: list, mode_filter: str = "") -> None:
        """Render legacy JSON-format decision log entries — newest first."""
        if mode_filter in ("paper", "live"):
            entries = [e for e in entries if e.get("trading_mode") == mode_filter]
        recent = list(reversed(entries[-20:]))
        print(f"  Showing {len(recent)} most recent decisions (legacy format):\n")
        print("  " + "─" * 76)
        for entry in recent:
            ts = entry.get("timestamp", "?")
            event = entry.get("event", "?")
            # Trading mode badge
            _mode = entry.get("trading_mode", "")
            if _mode == "paper":
                mode_badge = f"{_ANSI_Y}[PAPER]{_ANSI_RST}"
            elif _mode == "live":
                mode_badge = f"{_ANSI_G}[LIVE]{_ANSI_RST}"
            else:
                mode_badge = ""
            details = entry.get("details", {})
            bot_str = self._decision_entry_bot_label(entry)
            if isinstance(details, dict):
                pos = details.get("cur_pos", "?")
                action = details.get("action", "?")
                conf = details.get("confidence", "?")
                # Real schema keys: exit_action / exit_conf — no bare "pnl" field
                exit_conf = details.get("exit_conf")
                exit_act = details.get("exit_action")
                if exit_conf is not None:
                    conf_str = f"{float(conf):.3f}" if conf not in ("?", None) else "?"
                    exit_str = f" ExAct:{exit_act} ExConf:{exit_conf:.3f}"
                else:
                    conf_str = str(conf)
                    exit_str = ""
                # Show broker position_ids when present (added by newer bot builds)
                pids = entry.get("position_ids")
                pid_str = f" PIDs:{pids}" if pids else ""
                details_str = f"Pos:{pos} Act:{action} Conf:{conf_str}{exit_str}{pid_str}"
            else:
                details_str = str(details)
            if "OPEN" in event.upper() or "entry" in event.lower():
                color = _ANSI_G
            elif "CLOSE" in event.upper() or "exit" in event.lower():
                color = _ANSI_R
            elif "HOLD" in event.upper():
                color = _ANSI_Y
            else:
                color = _ANSI_RST
            print(f"  [{ts}] [{bot_str}] {mode_badge} {color}{event}{_ANSI_RST}: {details_str}")
        print("  " + "─" * 76)
        print(f"\n  Total decisions logged: {len(entries)}")

    def _load_transaction_events(
        self,
        event_types: frozenset | None = None,
        n: int = 50,
        sym_filter: str = "",
        tf_filter: int = 0,
    ) -> list[dict]:
        """Load recent transaction events from all bots' transactions.jsonl files."""
        _tx_files: list[Path] = sorted(self.data_dir.glob("paper_*_M*/logs/audit/transactions.jsonl"))
        _primary = self.data_dir / "logs" / "audit" / "transactions.jsonl"
        if _primary.exists():
            _tx_files.append(_primary)

        entries: list[dict] = []
        try:
            for _jf in _tx_files:
                try:
                    lines = _jf.read_text(encoding="utf-8").splitlines()
                except OSError:
                    continue
                for raw in reversed(lines):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event_types and e.get("event_type") not in event_types:
                        continue
                    _d = e.get("data") or {}
                    if sym_filter:
                        _esym = str(_d.get("symbol") or "").upper()
                        if _esym and _esym != sym_filter:
                            continue
                    if tf_filter:
                        _etf = int(_d.get("timeframe_minutes") or 0)
                        if _etf and _etf != tf_filter:
                            continue
                    e.setdefault("_source_path", str(_jf))
                    entries.append(e)
                    if len(entries) >= n * len(_tx_files):
                        break
        except Exception:
            pass

        entries.sort(key=lambda e: str(e.get("timestamp") or ""), reverse=True)
        return entries[:n]

    def _load_transactions_for_position(self, position_id: str, sym: str = "", tf_m: int = 0) -> list[dict]:
        """Load POSITION_OPEN/CLOSE events matching a given position_id."""
        _tx_files: list[Path] = []
        if sym and tf_m:
            _scoped = (
                self.data_dir / f"paper_{sym.upper()}_M{tf_m}" / "logs" / "audit" / "transactions.jsonl"
            )
            if _scoped.exists():
                _tx_files.append(_scoped)
        if not _tx_files:
            _tx_files.extend(sorted(self.data_dir.glob("paper_*_M*/logs/audit/transactions.jsonl")))
            _primary = self.data_dir / "logs" / "audit" / "transactions.jsonl"
            if _primary.exists():
                _tx_files.append(_primary)

        _target = frozenset({"POSITION_OPEN", "POSITION_CLOSE"})
        results: list[dict] = []
        try:
            for _jf in _tx_files:
                try:
                    lines = _jf.read_text(encoding="utf-8").splitlines()
                except OSError:
                    continue
                for raw in lines:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("event_type") not in _target:
                        continue
                    _d = e.get("data") or {}
                    if str(_d.get("position_id") or "") == position_id:
                        results.append(e)
        except Exception:
            pass

        return sorted(results, key=lambda e: str(e.get("timestamp") or ""))

    def _load_decision_entries(
        self,
        sym_filter: str = "",
        tf_filter: int = 0,
        n_per_file: int = 100,
    ) -> list[dict]:
        """Load meaningful (non-CACHED) decision entries, optionally scoped."""
        _primary = self.data_dir / "logs" / "audit" / "decisions.jsonl"
        _decision_files: list[Path] = sorted(self.data_dir.glob("paper_*_M*/logs/audit/decisions.jsonl"))
        if _primary.exists():
            _decision_files.append(_primary)

        entries: list[dict] = []
        _seen: set[tuple] = set()
        try:
            for _jf in _decision_files:
                for _entry in self._tail_meaningful(_jf, n=n_per_file):
                    _entry.setdefault("_source_path", str(_jf))
                    if not _entry.get("trading_mode") and "/paper_" in str(_jf):
                        _entry["trading_mode"] = "paper"
                    if sym_filter:
                        _esym = str(_entry.get("symbol") or "").upper()
                        if _esym and _esym != sym_filter:
                            continue
                    if tf_filter:
                        _etf = int(_entry.get("timeframe_minutes") or 0)
                        if _etf and _etf != tf_filter:
                            continue
                    _tf_label = self._decision_entry_timeframe_label(_entry)
                    if (
                        _jf == _primary
                        and not self._decision_entry_has_scope(_entry)
                        and len(_decision_files) > 1
                    ):
                        continue
                    _ctx_k = _entry.get("context", {}) or {}
                    _key = (
                        str(_entry.get("timestamp") or _entry.get("ts") or ""),
                        str(_entry.get("symbol") or _ctx_k.get("symbol") or ""),
                        _tf_label,
                        str(_entry.get("trade_id") or ""),
                        str(_entry.get("decision") or _entry.get("event") or ""),
                    )
                    if _key in _seen:
                        continue
                    _seen.add(_key)
                    entries.append(_entry)
        except Exception:
            pass

        entries.sort(key=lambda e: str(e.get("timestamp") or e.get("ts") or ""), reverse=True)
        return entries

    def _render_dec_log_detail(self, entry: dict) -> None:
        """L4 decision detail card — full context, reasoning, gated_conditions."""
        def _f(v: object, d: float = 0.0) -> float:
            try:
                return float(v) if v is not None else d
            except (TypeError, ValueError):
                return d

        ctx = entry.get("context") or {}
        reasoning = entry.get("reasoning") or {}
        decision = str(entry.get("decision") or "?").upper()
        agent = str(entry.get("agent") or "?")
        confidence = _f(entry.get("confidence"))
        ts = str(entry.get("timestamp") or "?")
        mode = str(entry.get("trading_mode") or "?")
        sym = str(entry.get("symbol") or "?").upper()
        tf_m = entry.get("timeframe_minutes")
        tf_lbl = self._format_timeframe_minutes_label(int(tf_m)) if tf_m else "?"
        trade_id = str(entry.get("trade_id") or "—")
        position_id = entry.get("position_id") or []

        if decision in ("BUY", "LONG", "ENTER"):
            dec_color = _ANSI_G
        elif decision in ("SELL", "SHORT", "EXIT", "CLOSE"):
            dec_color = _ANSI_R
        elif decision == "HOLD":
            dec_color = _ANSI_Y
        elif decision == "NO_ENTRY":
            dec_color = _ANSI_DIM
        else:
            dec_color = _ANSI_RST

        mode_str = (
            f"{_ANSI_Y}PAPER{_ANSI_RST}"
            if mode == "paper"
            else (f"{_ANSI_G}LIVE{_ANSI_RST}" if mode == "live" else mode)
        )

        print("\n  ┌─ DECISION DETAIL ──────────────────────────────────────────────────")
        print(f"  │  {ts}")
        print(f"  │  Bot: {_ANSI_G}{sym}/{tf_lbl}{_ANSI_RST}   Mode: {mode_str}   Agent: {agent}")
        print(f"  │  Decision: {dec_color}{decision}{_ANSI_RST}   Confidence: {confidence:.4f}")
        print(f"  │  TradeID:  {_ANSI_DIM}{trade_id}{_ANSI_RST}")
        if position_id:
            _pids = ", ".join(str(p) for p in (position_id if isinstance(position_id, list) else [position_id]))
            print(f"  │  PositionIDs: {_ANSI_DIM}{_pids}{_ANSI_RST}")
        print("  ├─ CONTEXT ─────────────────────────────────────────────────────────")
        if ctx:
            for _k, _v in ctx.items():
                if isinstance(_v, float):
                    print(f"  │    {_k:<28} {_v:.6f}")
                else:
                    print(f"  │    {_k:<28} {_v}")
        else:
            print(f"  │    {_ANSI_DIM}(no context){_ANSI_RST}")

        print("  ├─ REASONING ───────────────────────────────────────────────────────")
        _gated = reasoning.get("gated_conditions") or []
        for _k, _v in reasoning.items():
            if _k == "gated_conditions":
                continue
            if isinstance(_v, float):
                print(f"  │    {_k:<28} {_v:.6f}")
            elif isinstance(_v, bool):
                _vc = _ANSI_G if _v else _ANSI_R
                print(f"  │    {_k:<28} {_vc}{_v}{_ANSI_RST}")
            else:
                print(f"  │    {_k:<28} {_v}")

        if _gated and isinstance(_gated, list):
            print(f"  ├─ GATED CONDITIONS ({len(_gated)}) ──────────────────────────────────────")
            for _g in _gated:
                print(f"  │    {_ANSI_R}✗ {_g}{_ANSI_RST}")

        print("  └───────────────────────────────────────────────────────────────────")
        print(f"\n  {_ANSI_DIM}[b] or [Esc] back to list{_ANSI_RST}")

    def _render_decision_log(self) -> None:
        """Render the Decision Log tab (Tab 6) — hierarchical level dispatch."""
        self._render_breadcrumb("DECISION LOG", 6)

        # L4: decision detail card for selected entry
        if self._ctx_level >= 4 and self._dec_log_detail and self._dec_log_detail_entry:
            self._render_dec_log_detail(self._dec_log_detail_entry)
            return

        _mode_filter = self._mixed_mode_view_filter()

        # Determine scope from drill level
        _sym_f = self._ctx_symbol.upper() if self._ctx_level >= 2 and self._ctx_symbol else ""
        _tf_f = self._ctx_tf if self._ctx_level >= 3 and self._ctx_tf else 0

        # Row count: more entries as we drill deeper (less noise, more context)
        _n_rows = {1: 15, 2: 25, 3: 40}.get(self._ctx_level, 15)
        _n_read = max(150, _n_rows * 4)

        entries = self._load_decision_entries(sym_filter=_sym_f, tf_filter=_tf_f, n_per_file=_n_read)
        if _mode_filter in ("paper", "live"):
            entries = [e for e in entries if e.get("trading_mode") == _mode_filter]

        # L3: update view list for cursor navigation + Enter → L4
        if self._ctx_level >= 3:
            self._dec_log_view = entries[:_n_rows]
            self._dec_log_cursor = min(self._dec_log_cursor, max(0, len(self._dec_log_view) - 1))

        if not entries:
            # Try legacy fallback
            legacy_files = sorted(self.data_dir.glob("decision_log_*_M*.json"))
            legacy_files.extend(sorted(self.data_dir.glob("paper_*_M*/decision_log_*_M*.json")))
            root_legacy = self.data_dir / "decision_log.json"
            if root_legacy.exists():
                legacy_files.append(root_legacy)
            if not legacy_files:
                print("  ⚠️  No decision log found.")
                print(f"\n  {_ANSI_DIM}Expected: paper_<SYM>_M<TF>/logs/audit/decisions.jsonl{_ANSI_RST}")
                return
            try:
                _dec = json.JSONDecoder()
                leg_entries: list = []
                for log_file in legacy_files:
                    raw_text = log_file.read_text(encoding="utf-8")
                    _pos = 0
                    while _pos < len(raw_text):
                        _stripped = raw_text[_pos:].lstrip()
                        if not _stripped:
                            break
                        _skip = len(raw_text[_pos:]) - len(_stripped)
                        try:
                            _obj, _idx = _dec.raw_decode(raw_text, _pos + _skip)
                            _pos = _pos + _skip + _idx
                            if isinstance(_obj, list):
                                for _e in _obj:
                                    if isinstance(_e, dict):
                                        _e.setdefault("_source_path", str(log_file))
                                    leg_entries.append(_e)
                            elif isinstance(_obj, dict):
                                _obj.setdefault("_source_path", str(log_file))
                                leg_entries.append(_obj)
                        except json.JSONDecodeError:
                            break
            except Exception as e:
                print(f"  ❌ Error reading decision log: {e}")
                return
            if not leg_entries:
                print("  No entries yet. Waiting for bot decisions...")
                return
            self._render_legacy_decision_entries(leg_entries, _mode_filter)
            return

        # ── Level-specific header ─────────────────────────────────────────
        if self._ctx_level == 1:
            print(f"\n  {_ANSI_DIM}Portfolio scope — all bots, {_n_rows} most recent signal decisions{_ANSI_RST}")
        elif self._ctx_level == 2:
            print(
                f"\n  {_ANSI_DIM}Symbol scope — {_sym_f} all TFs, {_n_rows} most recent decisions  "
                f"[Enter] drill to TF{_ANSI_RST}"
            )
        elif self._ctx_level >= 3:
            _tf_lbl = self._format_timeframe_minutes_label(_tf_f) if _tf_f else "?"
            _nav = "[↑↓/jk] cursor  [Enter/d] detail card  [Esc] back"
            print(
                f"\n  {_ANSI_DIM}Bot scope — {_sym_f}/{_tf_lbl}, "
                f"{len(self._dec_log_view)} decisions  {_nav}{_ANSI_RST}"
            )

        # ── Render entries list with cursor highlight at L3 ───────────────
        self._render_jsonl_decision_entries(
            entries[:_n_rows],
            _mode_filter,
            cursor_idx=self._dec_log_cursor if self._ctx_level >= 3 else -1,
        )

    def _render_risk(self) -> None:
        """Render risk management — dispatches by drill level."""
        self._render_breadcrumb("RISK", 4)
        rs = self.risk_stats

        # L1: fleet-level circuit breaker summary across all bots
        if self._ctx_level == 1 and self.all_bots_stats:
            print(f"\n\033[1m⚠️  RISK — FLEET OVERVIEW\033[0m  {_ANSI_DIM}(all bots; [Enter] to drill){_ANSI_RST}\n")
            _hdr = f"  {'Bot':<16} {'Mode':<5} {'CB':<8} {'VaR95':>7} {'Kurtosis':>9} {'DrawPct':>8} {'Position':<12}"
            print(_hdr)
            print("  " + "─" * (_visible_width(_hdr) - 2))
            for _bot in sorted(
                self.all_bots_stats,
                key=lambda b: (str(b.get("symbol", "")).upper(), int(b.get("timeframe_minutes", 0) or 0)),
            ):
                _sym = str(_bot.get("symbol", "?")).upper()
                _tfm = int(_bot.get("timeframe_minutes", 0) or 0)
                _lbl = f"{_sym}/M{_tfm}"[:15]
                _mode = str(_bot.get("trading_mode", "paper") or "paper").upper()[:5]
                _rk = _bot.get("risk", {}) or {}
                _cb = "ACTIVE" if _rk.get("circuit_breaker_active") else "ok"
                _cb_c = _ANSI_R if _cb == "ACTIVE" else _ANSI_G
                _var = float(_rk.get("var_95", 0.0) or 0.0)
                _kurt = float(_rk.get("kurtosis", 0.0) or 0.0)
                _dd = float(_rk.get("drawdown_pct", 0.0) or 0.0)
                _pos_dir = str(_bot.get("position_direction", "FLAT") or "FLAT").upper()[:12]
                _mc = _ANSI_Y if _mode == "PAPER" else _ANSI_G
                _dd_c = _ANSI_R if _dd > DD_HIGH_PCT else (_ANSI_Y if _dd > DD_WARN_PCT else _ANSI_G)
                print(
                    f"  {_lbl:<16} {_mc}{_mode:<5}{_ANSI_RST} "
                    f"{_cb_c}{_cb:<8}{_ANSI_RST} {_var:>7.4f} {_kurt:>9.2f} "
                    f"{_dd_c}{_dd:>7.2f}%{_ANSI_RST} {_pos_dir:<12}"
                )
            print()
            # Also show last NO_ENTRY gated_conditions for context
            print(f"  {_ANSI_DIM}[d] detail pane for active bot  [Enter] to drill to symbol{_ANSI_RST}")
            if not self._ctx_detail:
                return

        _scope = self._risk_scope_label(rs)
        print(f"\n\033[1m⚠️  RISK MANAGEMENT [{_scope}]\033[0m\n")
        print(f"  {_ANSI_DIM}(source: scoped risk_metrics + circuit_breakers.json){_ANSI_RST}")
        self._render_risk_circuit_breaker(rs)
        self._render_risk_tail(rs)
        self._render_risk_regime(rs)
        self._render_risk_reward_weights(rs)
        self._render_risk_path_geometry(rs)
        self._render_risk_position_sizing(rs)

        # L3: show last gated_conditions for this bot
        if self._ctx_level >= 3 and self._ctx_symbol and self._ctx_tf:
            _sym_f = self._ctx_symbol.upper()
            _tf_f = self._ctx_tf
            _jf_candidates = list(self.data_dir.glob(
                f"paper_{_sym_f}_M{_tf_f}/logs/audit/decisions.jsonl"
            ))
            for _jf in _jf_candidates:
                _recent = self._tail_meaningful(_jf, n=20)
                _no_entries = [e for e in _recent if e.get("decision", "").upper() == "NO_ENTRY"]
                if _no_entries:
                    _last_no = _no_entries[-1]
                    _gates = (_last_no.get("reasoning") or {}).get("gated_conditions") or []
                    if _gates:
                        print(f"\n  \033[1mLAST NO_ENTRY GATES [{_sym_f}/M{_tf_f}]\033[0m")
                        _ts = str(_last_no.get("timestamp", ""))[:16]
                        print(f"  {_ANSI_DIM}{_ts}{_ANSI_RST}")
                        for _g in _gates[:8]:
                            print(f"  {_ANSI_R}  ✗ {_g}{_ANSI_RST}")
                    break

    def _render_risk_circuit_breaker(self, rs: dict) -> None:
        """Render circuit breaker status block with individual breaker details."""
        cb = rs.get("circuit_breaker", "INACTIVE")
        kurt_gate = rs.get("kurtosis_gate_active", False)
        _scope = self._risk_scope_label(rs)
        if cb == "ACTIVE":
            print(f"  {_ANSI_R}╔════════════════════════════════════════╗")
            print("  ║     ⚠️  CIRCUIT BREAKER ACTIVE ⚠️       ║")
            print(f"  ╚════════════════════════════════════════╝{_ANSI_RST}")
            print(f"  {_ANSI_Y}Press [r] to review and reset circuit breakers{_ANSI_RST}\n")
        elif kurt_gate:
            _is_live_mode = getattr(self, "_perf_snapshot_mode", "") == "live"
            _kurt_note = "entries BLOCKED" if _is_live_mode else "bypassed in paper mode"
            _kurt_threshold = float(
                rs.get("kurtosis_threshold")
                or self.market_stats.get("kurtosis_threshold")
                or KURTOSIS_FAT_TAIL_THRESHOLD
            )
            print(
                f"  {_ANSI_Y}⚡ Kurtosis gate: ACTIVE [{_scope}] "
                f"(κ={rs.get('kurtosis', 0):.1f} excess > {_kurt_threshold:.1f}){_ANSI_RST}  "
                f"{_ANSI_DIM}{_kurt_note}{_ANSI_RST}\n"
            )
        else:
            print(f"  {_ANSI_G}✓ Circuit Breaker: INACTIVE{_ANSI_RST}\n")

        # Load individual breaker statuses from circuit_breakers.json
        _cb_path = self._preferred_data_file("circuit_breakers.json")
        _cb_data: dict = {}
        if _cb_path.exists():
            try:
                with open(_cb_path, encoding="utf-8") as _f:
                    _cb_data = json.load(_f)
            except Exception:
                pass
        if not _cb_data:
            return

        _breaker_labels = {
            "sortino": "Sortino",
            "kurtosis": "Kurtosis",
            "drawdown": "Drawdown",
            "consecutive_losses": "Consec Losses",
        }
        print("  \033[1m🔌 INDIVIDUAL BREAKERS\033[0m")
        _kurt_gate_active = bool(rs.get("kurtosis_gate_active", False))
        _kurtosis_now = float(rs.get("kurtosis", 0.0) or 0.0)
        _kurtosis_threshold = float(
            rs.get("kurtosis_threshold")
            or self.market_stats.get("kurtosis_threshold")
            or KURTOSIS_FAT_TAIL_THRESHOLD
        )
        for _key, _label in _breaker_labels.items():
            _b = _cb_data.get(_key)
            if not isinstance(_b, dict):
                _b = {}
            _tripped = bool(_b.get("is_tripped", False))
            _gate_only = False
            if _key == "kurtosis" and _kurt_gate_active and not _tripped:
                _tripped = True
                _gate_only = True
            if _tripped:
                _trip_ts = _b.get("trip_time", "")
                _ts_short = _trip_ts[11:19] if _trip_ts and len(_trip_ts) >= 19 else (_trip_ts or "?")
                _icon = f"{_ANSI_R}✗ TRIPPED{_ANSI_RST}"
                _detail = f"  {_ANSI_DIM}@ {_ts_short}{_ANSI_RST}" if _trip_ts else ""

                _reason = _b.get("trip_reason", "")
                if _gate_only:
                    _reason = f"Kurtosis gate active [{_scope}] (entry gate)"
                if _reason:
                    _detail += f"  {_ANSI_Y}→ {_reason}{_ANSI_RST}"

                _tv = _b.get("trip_value", 0.0)
                _th = _b.get("threshold", 0.0)
                if _gate_only:
                    _tv = _kurtosis_now
                    _th = _kurtosis_threshold
                if _tv or _th:
                    _detail += f"  {_ANSI_DIM}(val={_tv:.2f} thr={_th:.2f}){_ANSI_RST}"

                _cd_mins = _b.get("cooldown_minutes", 60)
                if _trip_ts:
                    try:
                        _trip_dt = datetime.fromisoformat(_trip_ts)
                        _elapsed = (datetime.now(UTC) - _trip_dt).total_seconds() / 60.0
                        _remaining = max(0, _cd_mins - _elapsed)
                        if _remaining > 0:
                            _detail += f"  {_ANSI_DIM}cooldown: {_remaining:.0f}m left{_ANSI_RST}"
                        else:
                            _detail += f"  {_ANSI_G}cooldown elapsed{_ANSI_RST}"
                    except (ValueError, TypeError):
                        pass
            else:
                _icon = f"{_ANSI_G}✓ OK{_ANSI_RST}"
                _detail = ""
            # Append breaker-specific live values
            _extra = ""
            if _key == "drawdown":
                _dd = _b.get("current_drawdown", 0.0)
                _peak = _b.get("peak_equity", 0.0)
                _dd_pct = _dd * 100 if _dd < 1 else _dd  # handle both fraction and %
                _dd_col = _ANSI_R if _dd_pct > DD_HIGH_PCT else (_ANSI_Y if _dd_pct > DD_WARN_PCT else _ANSI_G)
                _extra = f"  {_dd_col}DD={_dd_pct:.2f}%{_ANSI_RST}  peak={_peak:.0f}"
            elif _key == "consecutive_losses":
                _streak = _b.get("consecutive_losses", 0)
                _s_col = _ANSI_R if _streak >= 5 else (_ANSI_Y if _streak >= 3 else _ANSI_G)
                _extra = f"  {_s_col}streak={_streak}{_ANSI_RST}"
            print(f"    {_label:<15} {_icon}{_detail}{_extra}")
        print()

    def _risk_scope_label(self, rs: dict) -> str:
        """Return display scope for risk metrics."""
        _sym = str(rs.get("symbol") or self.active_sym or "").strip().upper()
        _tf = self._normalize_timeframe_label(rs)
        if _tf == "M?" and self.active_tf_min:
            _tf = self._format_timeframe_minutes_label(self.active_tf_min)
        if _sym and _tf != "M?":
            return f"{_sym} {_tf}"
        if _tf != "M?":
            return _tf
        return _sym or "unknown timeframe"

    def _render_risk_tail(self, rs: dict) -> None:
        """Render tail risk values."""
        print("  \033[1m📉 TAIL RISK\033[0m")
        var = rs.get("var", 0) * 100
        kurtosis = rs.get("kurtosis", 0)
        _threshold_raw = rs.get("kurtosis_threshold") or self.market_stats.get("kurtosis_threshold")
        kurtosis_threshold = float(_threshold_raw or KURTOSIS_FAT_TAIL_THRESHOLD)
        _threshold_label = "scoped entry gate" if _threshold_raw is not None else "fallback alert"
        vol = rs.get("realized_vol", 0) * 100

        kurt_col = _ANSI_R if kurtosis > kurtosis_threshold else _ANSI_G
        if var > VAR_HIGH_PCT:
            var_col = _ANSI_R
        elif var > VAR_WARN_PCT:
            var_col = _ANSI_Y
        else:
            var_col = _ANSI_G
        if vol > VOL_HIGH_PCT:
            vol_col = _ANSI_R
        elif vol > VOL_WARN_PCT:
            vol_col = _ANSI_Y
        else:
            vol_col = _ANSI_G

        print(
            f"    VaR 95%:           {var_col}{var:>9.3f}%{_ANSI_RST}  "
            f"{_ANSI_DIM}(position loss at 95th pct){_ANSI_RST}"
        )
        print(f"    Realized vol:      {vol_col}{vol:>9.3f}%{_ANSI_RST}")
        print(
            f"    Kurtosis:          {kurt_col}{kurtosis:>9.2f}{_ANSI_RST}  "
            f"{_ANSI_DIM}(excess; {_threshold_label} threshold >{kurtosis_threshold:.1f}){_ANSI_RST}"
        )
        print()

    def _render_risk_regime(self, rs: dict) -> None:
        """Render regime classification with ζ gauge and update status."""
        print("  \033[1m🌐 REGIME\033[0m")
        regime = rs.get("regime", "UNKNOWN")
        zeta = rs.get("regime_zeta", 1.0)
        vr = rs.get("regime_vr", 1.0)
        updates = rs.get("regime_updates", 0)
        next_in = rs.get("regime_next_in", 0)
        regime_colors = {
            "TRENDING": _ANSI_G,
            "MEAN_REVERTING": _ANSI_Y,
            "TRANSITIONAL": _ANSI_B,
            "UNKNOWN": _ANSI_DIM,
        }
        _regime_tips = {
            "TRENDING": "trend-follow; let winners run",
            "MEAN_REVERTING": "fade extremes; tighten target",
            "TRANSITIONAL": "reduce size; wait for clarity",
            "UNKNOWN": "cold-start; use fallback rules",
        }
        regime_color = regime_colors.get(regime, _ANSI_DIM)
        _next_tag = f"  next recalc in {next_in} bar{'s' if next_in != 1 else ''}" if next_in > 0 else "  recalc now"
        print(
            f"    Regime:            {regime_color}{regime}{_ANSI_RST}  "
            f"{_ANSI_DIM}({_regime_tips.get(regime, '')}){_ANSI_RST}"
        )
        print(f"    Damping (ζ):       {zeta:>10.3f}  {_ANSI_DIM}(< 0.7 trending | > 1.3 mean-rev){_ANSI_RST}")
        print(f"    Variance Ratio:    {vr:>10.3f}  {_ANSI_DIM}(1.0 = random walk){_ANSI_RST}")
        # ζ gauge: visual bar showing position within [0.1 .. 2.0] range
        # Markers: 0.7 (trending threshold) and 1.3 (mean-revert threshold)
        _gauge_w = 40
        _zeta_min, _zeta_max = 0.1, 2.0
        _pos = int(max(0, min(_gauge_w - 1, (zeta - _zeta_min) / (_zeta_max - _zeta_min) * _gauge_w)))
        _t_mark = int((0.7 - _zeta_min) / (_zeta_max - _zeta_min) * _gauge_w)  # trending threshold
        _m_mark = int((1.3 - _zeta_min) / (_zeta_max - _zeta_min) * _gauge_w)  # mean-revert threshold
        _bar_chars = list("─" * _gauge_w)
        _bar_chars[_t_mark] = "│"
        _bar_chars[_m_mark] = "│"
        _bar_chars[_pos] = "●"
        _gauge = "".join(_bar_chars)
        print(f"    ζ: [{_ANSI_G}TREND{_ANSI_RST}│{_ANSI_B}TRANS{_ANSI_RST}│{_ANSI_Y}M-REV{_ANSI_RST}]  {_gauge}")
        print(f"       {_ANSI_DIM}0.1{'':>10}0.7{'':>15}1.3{'':>12}2.0{_ANSI_RST}")
        print(f"    Updates:           {updates:>10d}{_ANSI_DIM}{_next_tag}{_ANSI_RST}")
        print()

    def _render_risk_reward_weights(self, rs: dict) -> None:
        """Render adaptive reward component weights."""
        weights = rs.get("reward_weights", {})
        if not weights:
            return
        print("  \033[1m🎚️  REWARD WEIGHTS\033[0m  " + f"{_ANSI_DIM}(adaptive, bounded 0.2–2.0){_ANSI_RST}")
        _gauge_w = 20
        # Pull per-bot reward-shaping telemetry (avg component rewards + total)
        # from the active bot's paper_stats entry, if available.
        _rshape: dict = {}
        try:
            for _b in getattr(self, "all_bots_stats", []) or []:
                if _b.get("symbol") == self.active_sym and int(_b.get("timeframe_minutes", 0) or 0) == int(
                    self.active_tf_min or 0
                ):
                    _rshape = _b.get("reward_shaping", {}) or {}
                    break
        except Exception:
            _rshape = {}
        _comps = _rshape.get("components", {}) if isinstance(_rshape, dict) else {}
        for name, val in weights.items():
            # Visual bar: 0.2 (min) to 2.0 (max), default 1.0
            frac = max(0.0, min(1.0, (val - 0.2) / 1.8))
            filled = int(_gauge_w * frac)
            bar = "█" * filled + "░" * (_gauge_w - filled)
            # Color: green near 1.0, yellow when drifted, red when at bounds
            if 0.8 <= val <= 1.2:
                col = _ANSI_G
            elif 0.5 <= val <= 1.5:
                col = _ANSI_Y
            else:
                col = _ANSI_R
            _c = _comps.get(name, {}) if isinstance(_comps, dict) else {}
            _cnt = int(_c.get("count", 0) or 0) if isinstance(_c, dict) else 0
            _avg = float(_c.get("avg", 0.0) or 0.0) if isinstance(_c, dict) else 0.0
            if _cnt > 0:
                _avg_col = _ANSI_G if _avg > 0 else (_ANSI_R if _avg < 0 else _ANSI_DIM)
                _tail = f"  {_ANSI_DIM}avg={_ANSI_RST}{_avg_col}{_avg:+.3f}{_ANSI_RST}  {_ANSI_DIM}n={_cnt}{_ANSI_RST}"
            else:
                _tail = f"  {_ANSI_DIM}avg= n/a   n=0{_ANSI_RST}"
            print(f"    {name:<16s} {col}{val:>5.2f}{_ANSI_RST}  [{bar}]{_tail}")
        _total = int(_rshape.get("total_rewards_calculated", 0) or 0) if isinstance(_rshape, dict) else 0
        if _total:
            print(f"    {_ANSI_DIM}total rewards calculated: {_total}{_ANSI_RST}")
        print()

    def _render_risk_path_geometry(self, rs: dict) -> None:
        """Render path geometry features."""
        print(f"  \033[1m📐 PATH GEOMETRY  {_ANSI_DIM}(RL feature inputs){_ANSI_RST}")
        eff = rs.get("efficiency", 0)
        gamma = rs.get("gamma", 0)
        runway = rs.get("runway", 0.5)
        feas = rs.get("feasibility", 0.5)

        if feas > FEASIBILITY_HIGH_THRESHOLD:
            feas_color = _ANSI_G
        elif feas > FEASIBILITY_MEDIUM_THRESHOLD:
            feas_color = _ANSI_Y
        else:
            feas_color = _ANSI_R
        if eff > EFF_HIGH_THRESHOLD:
            eff_col = _ANSI_G
        elif eff > EFF_WARN_THRESHOLD:
            eff_col = _ANSI_Y
        else:
            eff_col = _ANSI_R
        gam_col = _ANSI_G if gamma > 0 else _ANSI_R
        if runway > RUNWAY_OK_BARS:
            rwy_col = _ANSI_G
        elif runway > RUNWAY_WARN_BARS:
            rwy_col = _ANSI_Y
        else:
            rwy_col = _ANSI_R

        print(
            f"    Efficiency:        {eff_col}{eff:>10.3f}{_ANSI_RST}  "
            f"{_ANSI_DIM}(path directness; 1=straight trend){_ANSI_RST}"
        )
        print(
            f"    Gamma (γ):         {gam_col}{gamma:>+10.3f}{_ANSI_RST}  "
            f"{_ANSI_DIM}(price acceleration; +ve favours longs){_ANSI_RST}"
        )
        jerk = rs.get("jerk", 0.0)
        jerk_col = _ANSI_Y if abs(jerk) > 0.1 else _ANSI_DIM
        print(
            f"    Jerk (dγ/dt):      {jerk_col}{jerk:>+10.4f}{_ANSI_RST}  "
            f"{_ANSI_DIM}(rate of change of gamma){_ANSI_RST}"
        )
        print(
            f"    Runway:            {rwy_col}{runway:>10.3f}{_ANSI_RST}  "
            f"{_ANSI_DIM}(vol headwind score; 1=smooth, 0=heavy){_ANSI_RST}"
        )
        print(f"    Entry Feasibility: {feas_color}{feas:>10.3f}{_ANSI_RST}")

        # Depth metrics
        _depth_ratio = rs.get("depth_ratio", 0.0)
        _depth_levels = rs.get("depth_levels", 0)
        _depth_buffer = rs.get("depth_buffer", 0.0)
        _depth_gate = rs.get("depth_gate_active", False)
        _has_l2 = _depth_levels > 0
        if _has_l2 or _depth_ratio > 0:
            print()
            _gate_str = f"  {_ANSI_R}[GATE ACTIVE]{_ANSI_RST}" if _depth_gate else ""
            if _has_l2:
                _dr_col = _ANSI_G if _depth_ratio > 0.8 else (_ANSI_Y if _depth_ratio > 0.5 else _ANSI_R)
                print(
                    f"    Depth ratio:       {_dr_col}{_depth_ratio:>10.3f}{_ANSI_RST}  "
                    f"{_ANSI_DIM}(bid_depth/ask_depth; 1=balanced){_ANSI_RST}{_gate_str}"
                )
            else:
                # depth_ratio defaults to 1.0 when there is no real L2 feed.
                # Display N/A so it does not look like a balanced live order book.
                print(
                    f"    Depth ratio:       {_ANSI_DIM}       N/A{_ANSI_RST}  "
                    f"{_ANSI_DIM}(no L2 data){_ANSI_RST}{_gate_str}"
                )
            print(f"    Depth levels:      {_depth_levels:>10}    buffer: {_depth_buffer:.2f}")

        bar_len = 40
        feas_pct = max(0, min(1, feas))
        filled = int(bar_len * feas_pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n    [{bar}]")
        print("     LOW                                HIGH")

    def _render_risk_position_sizing(self, rs: dict) -> None:
        """Render position sizing block."""
        print()
        print("  \033[1m💰 POSITION SIZING\033[0m")
        risk_budget = rs.get("risk_budget_usd", 0.0)
        risk_req_qty = rs.get("risk_requested_qty", 0.0)
        risk_final_qty = rs.get("risk_final_qty", 0.0)
        vol_cap = rs.get("vol_cap", 0.0)
        vol_ref = rs.get("vol_reference", 0.0)
        realized_vol = rs.get("realized_vol", 0.0)

        qty_color = _ANSI_G if risk_final_qty == risk_req_qty else _ANSI_Y

        budget_used_pct = (risk_final_qty / risk_req_qty * 100) if risk_req_qty > _QTY_FLOOR else 100.0
        capped = risk_req_qty > _QTY_FLOOR and risk_final_qty < risk_req_qty * 0.999

        print(f"    Risk budget:       {risk_budget:>10.2f} USD")
        print(f"    Requested qty:     {risk_req_qty:>10.4f}")
        print(
            f"    Final qty:         {qty_color}{risk_final_qty:>10.4f}{_ANSI_RST}  ({budget_used_pct:.0f}% of request)"
        )
        if capped:
            print(f"    {_ANSI_Y}⚡ Qty capped — vol or depth constraint active{_ANSI_RST}")
        print(f"    Vol cap:           {vol_cap * 100:>9.2f}%  {_ANSI_DIM}(max position vol allowed){_ANSI_RST}")
        # vol_reference is a fixed config value used as the baseline for the
        # VaR vol multiplier (vol_mult = realized/reference).  Show the ratio
        # so it's clear how far the current market vol is from the reference.
        if vol_ref > 0 and realized_vol > 0:
            _vol_ratio = realized_vol / vol_ref
            _vr_col = _ANSI_Y if abs(_vol_ratio - 1.0) > 0.5 else _ANSI_G
            print(
                f"    Vol reference:     {vol_ref * 100:>9.3f}%  "
                f"{_ANSI_DIM}(fixed baseline; {_vr_col}×{_vol_ratio:.2f} vs realized{_ANSI_DIM}){_ANSI_RST}"
            )
        else:
            print(f"    Vol reference:     {vol_ref * 100:>9.3f}%  {_ANSI_DIM}(fixed baseline for vol cap){_ANSI_RST}")

    def _render_order_book_ladder(self, bids: list, asks: list, depth_bid: float, depth_ask: float, dec: int) -> None:
        """Render the L2 order-book price ladder (5 rows, aligned columns).

        Best bid and best ask share the top row; deeper levels descend.
        Bid bars grow rightward (toward spread), ask bars grow leftward
        (toward spread), making both visually converge at the center.
        """
        N = 5
        padded_asks = (asks + [[0.0, 0.0]] * N)[:N]
        padded_bids = (bids + [[0.0, 0.0]] * N)[:N]
        all_sizes = [s for _, s in bids + asks if s > 0]
        max_sz = max(all_sizes) if all_sizes else 1.0
        BAR = 12
        print(f"    {'BID depth':<20}   {'':<{BAR * 2 + 2}}   {'ASK depth':>20}")
        print(f"    {'─' * 20}   {'─' * (BAR * 2 + 2)}   {'─' * 20}")
        for bid_row, ask_row in zip(padded_bids, padded_asks, strict=False):
            b_px, b_sz = bid_row
            a_px, a_sz = ask_row
            b_bar = int(BAR * b_sz / max_sz) if b_sz > 0 else 0
            a_bar = int(BAR * a_sz / max_sz) if a_sz > 0 else 0
            b_side = f"{b_sz:>6.2f} {_ANSI_G}{'░' * (BAR - b_bar)}{'▓' * b_bar}{_ANSI_RST}" if b_px else " " * (BAR + 8)
            a_side = f"{_ANSI_R}{'▓' * a_bar}{'░' * (BAR - a_bar)}{_ANSI_RST} {a_sz:<6.2f}" if a_px else " " * (BAR + 8)
            b_px_str = f"{b_px:.{dec}f}" if b_px else "—"
            a_px_str = f"{a_px:.{dec}f}" if a_px else "—"
            print(f"    {b_side}  {b_px_str:<10}  {a_px_str:>10}  {a_side}")
        print(f"    Total depth — bid: {depth_bid:<8.2f}  │  ask: {depth_ask:.2f}")

    def _render_signal_synthesis(self, vpin_z: float, imbalance: float) -> None:
        """Render the signal synthesis advisory block."""
        print()
        print("  \033[1m🧭 SIGNAL SYNTHESIS\033[0m")
        rs_regime = self.risk_stats.get("regime", "UNKNOWN")
        rs_feas = float(self.risk_stats.get("feasibility", 0.5))
        rs_runway = float(self.risk_stats.get("runway", 0.0))
        toxic = abs(vpin_z) > VPIN_HIGH_TOXICITY_THRESHOLD
        gate = self.risk_stats.get("depth_gate_active", False)
        signals = []
        if rs_feas < FEASIBILITY_MEDIUM_THRESHOLD:
            signals.append(f"{_ANSI_R}✗ Low feasibility — no new entries{_ANSI_RST}")
        if toxic:
            signals.append(f"{_ANSI_R}✗ Toxic flow (VPIN) — stop widening advised{_ANSI_RST}")
        if gate:
            signals.append(f"{_ANSI_R}✗ Depth gate active — no new entries{_ANSI_RST}")
        if rs_regime == "TRENDING" and not toxic and rs_feas > FEASIBILITY_HIGH_THRESHOLD:
            if imbalance > _IMBALANCE_DIRECTION_HINT:
                dir_hint = "LONG"
            elif imbalance < -_IMBALANCE_DIRECTION_HINT:
                dir_hint = "SHORT"
            else:
                dir_hint = "either direction"
            signals.append(f"{_ANSI_G}✓ Trending + clean flow → favours {dir_hint}{_ANSI_RST}")
        if rs_regime == "MEAN_REVERTING" and not toxic:
            signals.append(f"{_ANSI_Y}⚡ Mean-reverting — shorter hold, tighter target{_ANSI_RST}")
        if rs_runway < RUNWAY_SHORT_THRESHOLD:
            # rs_runway is the path geometry vol-headwind score (1/(1+50σ)).
            # Low value = high realized vol, not a harvester-specific runway prediction.
            signals.append(
                f"{_ANSI_Y}⚡ Rough path conditions (runway={rs_runway:.2f}) — high vol, widen stops{_ANSI_RST}"
            )
        if not signals:
            signals.append(f"{_ANSI_DIM}— No strong signals; model discretion applies{_ANSI_RST}")
        for s in signals:
            print(f"    {s}")

    def _render_market(self) -> None:
        """Render market microstructure — dispatches by drill level."""
        self._render_breadcrumb("MARKET", 5)

        # L1: compact per-symbol summary (no mode dimension — same feed regardless)
        if self._ctx_level == 1 and self.all_bots_stats:
            print(f"\n\033[1m🔬 MARKET SUMMARY — ALL SYMBOLS\033[0m  {_ANSI_DIM}[Enter] to drill{_ANSI_RST}\n")
            _hdr = f"  {'Symbol':<10} {'Spread':>7} {'VPIN-z':>7} {'Depth':>7} {'Imbal':>7} {'Regime':<8} {'VaR95':>7}"
            print(_hdr)
            print("  " + "─" * (_visible_width(_hdr) - 2))
            _seen_syms: set[str] = set()
            for _bot in sorted(
                self.all_bots_stats,
                key=lambda b: str(b.get("symbol", "")).upper(),
            ):
                _sym = str(_bot.get("symbol", "?")).upper()
                if _sym in _seen_syms:
                    continue
                _seen_syms.add(_sym)
                _mk = _bot.get("market", {}) or {}
                _rk = _bot.get("risk", {}) or {}
                _spread = float(_mk.get("spread", 0.0) or 0.0)
                _vpin_z = float(_mk.get("vpin_z", 0.0) or 0.0)
                _depth = float(_mk.get("depth_ratio", 0.0) or 0.0)
                _imbal = float(_mk.get("imbalance", 0.0) or 0.0)
                _regime = str(_rk.get("regime", "?") or "?")[:8]
                _var = float(_rk.get("var_95", 0.0) or 0.0)
                _vz_c = _ANSI_R if abs(_vpin_z) > 2.5 else (_ANSI_Y if abs(_vpin_z) > 1.5 else _ANSI_G)
                print(
                    f"  {_sym:<10} {_spread:>7.2f} {_vz_c}{_vpin_z:>+7.2f}{_ANSI_RST} "
                    f"{_depth:>7.3f} {_imbal:>+7.3f} {_regime:<8} {_var:>7.4f}"
                )
            print(f"\n  {_ANSI_DIM}[d] detail for active bot  [Enter] to drill to symbol{_ANSI_RST}")
            if not self._ctx_detail:
                return

        _scope = self._risk_scope_label(self.risk_stats)
        print(f"\n\033[1m🔬 MARKET MICROSTRUCTURE [{_scope}]\033[0m\n")
        _market_age: float | None = None
        _sources = (
            self._preferred_data_file(_ORDER_BOOK_FILE),
            self._preferred_data_file("risk_metrics.json"),
        )
        _existing = [p for p in _sources if p.exists()]
        if _existing:
            _latest = max(p.stat().st_mtime for p in _existing)
            _market_age = time.time() - _latest
        if _market_age is None:
            _feed = f"{_ANSI_DIM}⏳ waiting for market feed{_ANSI_RST}"
        elif _market_age > DATA_STALE_SECS:
            _feed = f"{_ANSI_R}⚠ STALE ({_market_age:.0f}s){_ANSI_RST}"
        elif _market_age > DATA_AGING_SECS:
            _feed = f"{_ANSI_Y}⚡ AGING ({_market_age:.0f}s){_ANSI_RST}"
        else:
            _feed = f"{_ANSI_G}✓ LIVE ({_market_age:.1f}s){_ANSI_RST}"
        print(f"  Feed freshness: {_feed}\n")
        ms = self.market_stats
        self._render_market_spread(ms)
        self._render_market_vpin(ms)
        self._render_market_imbalance(ms)
        self._render_market_rs_vol(ms)
        self._render_market_kurtosis(ms)

    def _render_market_spread(self, ms: dict) -> None:
        """Render spread and order book ladder."""
        print("  \033[1m💹 SPREAD & LIQUIDITY\033[0m")
        spread = ms.get("spread", 0)
        depth_bid = ms.get("depth_bid", 0)
        depth_ask = ms.get("depth_ask", 0)
        bids = ms.get("order_book_bids", [])
        asks = ms.get("order_book_asks", [])
        if bids:
            base_price = bids[0][0]
        elif asks:
            base_price = asks[0][0]
        else:
            base_price = 0.0
        dec = self._price_decimals(base_price)
        print(f"    Bid-Ask Spread:    {spread:>12.{dec}f}")
        print()
        self._render_order_book_ladder(bids, asks, depth_bid, depth_ask, dec)
        print()

    def _render_market_vpin(self, ms: dict) -> None:
        """Render VPIN toxicity block."""
        print("  \033[1m☢️  ORDER FLOW TOXICITY (VPIN)\033[0m")
        vpin_z = ms.get("vpin_z", 0)
        if abs(vpin_z) > VPIN_HIGH_TOXICITY_THRESHOLD:
            vpin_status = f"{_ANSI_R}⚠️  HIGH TOXICITY{_ANSI_RST}"
        elif abs(vpin_z) > VPIN_ELEVATED_TOXICITY_THRESHOLD:
            vpin_status = f"{_ANSI_Y}⚡ ELEVATED{_ANSI_RST}"
        else:
            vpin_status = f"{_ANSI_G}✓ NORMAL{_ANSI_RST}"
        print(f"    VPIN Z-Score:      {vpin_z:>+12.2f}")
        print(f"    Status:            {vpin_status}")
        print(
            f"                       {_ANSI_DIM}High +z = informed sellers active "
            f"→ widen stops / reduce size{_ANSI_RST}"
        )
        bar_len = 40
        z_norm = max(0.0, min(1.0, (vpin_z + 3) / 6))
        pos = max(0, min(bar_len - 1, int(bar_len * z_norm)))
        center = bar_len // 2
        cells = ["─"] * bar_len
        if 0 <= center < bar_len:
            cells[center] = "┼"
        # Marker colour reflects toxicity band
        if abs(vpin_z) > VPIN_HIGH_TOXICITY_THRESHOLD:
            _mk_c = _ANSI_R
        elif abs(vpin_z) > VPIN_ELEVATED_TOXICITY_THRESHOLD:
            _mk_c = _ANSI_Y
        else:
            _mk_c = _ANSI_G
        cells[pos] = f"{_mk_c}●{_ANSI_RST}"
        gauge = "".join(cells)
        print(f"\n    Z: [{gauge}]")
        print("       -3              0              +3")
        print()

    def _render_market_imbalance(self, ms: dict) -> None:
        """Render order imbalance and signal synthesis."""
        print("  \033[1m⚖️  ORDER IMBALANCE (QFI)\033[0m")
        has_real_sizes = ms.get("has_real_sizes", False)
        qfi_updates = int(ms.get("qfi_update_count", 0))
        imbalance = ms.get("imbalance", 0.0)
        if has_real_sizes and qfi_updates > 0:
            signal_source = f"{_ANSI_G}size+QFI blend{_ANSI_RST}"
        elif has_real_sizes:
            signal_source = f"{_ANSI_G}size-weighted{_ANSI_RST}"
        else:
            signal_source = f"{_ANSI_Y}quote-flow (QFI){_ANSI_RST}"
        if imbalance > IMBALANCE_BUY_THRESHOLD:
            imb_status = f"{_ANSI_G}🔺 BUY PRESSURE{_ANSI_RST}"
        elif imbalance < IMBALANCE_SELL_THRESHOLD:
            imb_status = f"{_ANSI_R}🔻 SELL PRESSURE{_ANSI_RST}"
        else:
            imb_status = f"{_ANSI_Y}⚖️  BALANCED{_ANSI_RST}"
        print(f"    Imbalance:         {imbalance:>+12.4f}")
        print(f"    Signal source:     {signal_source}  (updates: {qfi_updates})")
        print(f"    Status:            {imb_status}")
        bar_len = 40
        mid = bar_len // 2
        # Clamp imbalance to [-1, 1] for display
        imb_clamped = max(-1.0, min(1.0, float(imbalance)))
        # Compute marker position; ensure a visible cursor even for tiny values
        pos = mid + round(mid * imb_clamped)
        pos = max(0, min(bar_len - 1, pos))
        cells = ["─"] * bar_len
        cells[mid] = "┼"
        if imb_clamped > IMBALANCE_BUY_THRESHOLD:
            _mk_c = _ANSI_G
        elif imb_clamped < IMBALANCE_SELL_THRESHOLD:
            _mk_c = _ANSI_R
        else:
            _mk_c = _ANSI_Y
        cells[pos] = f"{_mk_c}●{_ANSI_RST}"
        bar = "".join(cells)
        print(f"\n    [{bar}]")
        print("     SELL              ↕              BUY")
        self._render_signal_synthesis(ms.get("vpin_z", 0), imbalance)

    def _render_market_rs_vol(self, ms: dict) -> None:
        """Render Rogers-Satchell volatility regime block."""
        print("  \033[1m📊 VOLATILITY REGIME (Rogers-Satchell)\033[0m")
        rs_s = float(ms.get("rs_vol_short", 0.0) or 0.0)
        rs_l = float(ms.get("rs_vol_long", 0.0) or 0.0)
        ratio = float(ms.get("rs_vol_ratio", 1.0) or 1.0)
        if ratio > 1.4:
            regime_label = f"{_ANSI_R}⚡ EXPANDING{_ANSI_RST}"
        elif ratio < 0.7:
            regime_label = f"{_ANSI_G}🔵 CONTRACTING{_ANSI_RST}"
        else:
            regime_label = f"{_ANSI_Y}⚖  NEUTRAL{_ANSI_RST}"
        print(f"    RS Vol Short (10):  {rs_s * 100:>9.4f}%")
        print(f"    RS Vol Long  (50):  {rs_l * 100:>9.4f}%")
        print(f"    RS Ratio:           {ratio:>9.3f}   {regime_label}")
        bar_len = 40
        ratio_norm = max(0.0, min(1.0, (ratio - 0.4) / 1.6))
        pos = max(0, min(bar_len - 1, int(bar_len * ratio_norm)))
        center = int(bar_len * (1.0 - 0.4) / 1.6)
        cells = ["─"] * bar_len
        if 0 <= center < bar_len:
            cells[center] = "┼"
        _mk_c = _ANSI_R if ratio > 1.4 else (_ANSI_G if ratio < 0.7 else _ANSI_Y)
        cells[pos] = f"{_mk_c}●{_ANSI_RST}"
        print(f"\n    [{('').join(cells)}]")
        print("     0.4        1.0(neutral)        2.0+")
        print()

    def _render_market_kurtosis(self, ms: dict) -> None:
        """Render return-distribution kurtosis, fat-tail gate, and depth gate status."""
        print("  \033[1m📐 ENTRY GATES\033[0m")
        kurt = float(ms.get("kurtosis", 0.0) or 0.0)
        threshold = float(ms.get("kurtosis_threshold", KURTOSIS_FAT_TAIL_THRESHOLD) or KURTOSIS_FAT_TAIL_THRESHOLD)
        kurt_gate = bool(ms.get("kurtosis_gate_active", False))
        depth_gate = bool(ms.get("depth_gate_active", False))
        depth_floor = float(ms.get("depth_floor", 0.0) or 0.0)
        depth_bid = float(ms.get("depth_bid", 0.0) or 0.0)
        depth_ask = float(ms.get("depth_ask", 0.0) or 0.0)
        kurt_str = (f"{_ANSI_R}⛔ BLOCKED (fat tails){_ANSI_RST}" if kurt_gate
                    else f"{_ANSI_G}✓ clear{_ANSI_RST}")
        depth_str = (f"{_ANSI_R}⛔ BLOCKED (thin book){_ANSI_RST}" if depth_gate
                     else f"{_ANSI_G}✓ clear{_ANSI_RST}")
        excess_color = _ANSI_R if kurt > threshold else (_ANSI_Y if kurt > 0 else _ANSI_G)
        print(
            f"    Kurtosis:           {excess_color}{kurt:>9.3f}{_ANSI_RST}  "
            f"(excess; scoped threshold={threshold:.2f})"
        )
        print(f"    Kurtosis gate:      {kurt_str}")
        print(f"    Depth (bid/ask):    {depth_bid:>7.3f} / {depth_ask:<7.3f}  floor={depth_floor:.3f}")
        print(f"    Depth gate:         {depth_str}")
        print()

    def _render_footer(self) -> None:
        """Render footer with controls and data freshness."""
        W = self._term_width()
        print("\n" + "─" * W)
        self._render_footer_tab_nav()
        self._render_footer_controls(W)
        note = self._current_notification() or "Press 'h' for help and keyboard shortcuts."
        print(f"  {note}  |  {self._footer_data_freshness()}")
        print("─" * W)

    def _render_footer_tab_nav(self) -> None:
        idx = self.TAB_ORDER.index(self.current_tab)
        prev_tab = self.TAB_DISPLAY_SHORT[self.TAB_ORDER[(idx - 1) % len(self.TAB_ORDER)]]
        next_tab = self.TAB_DISPLAY_SHORT[self.TAB_ORDER[(idx + 1) % len(self.TAB_ORDER)]]
        tab_pos = f"Tab {idx + 1}/{len(self.TAB_ORDER)}: {self.TAB_DISPLAY_SHORT[self.current_tab]}"
        print(f"  {_ANSI_DIM}← {prev_tab}  |  {tab_pos}  |  {next_tab} →{_ANSI_RST}")

    def _render_footer_controls(self, width: int) -> None:
        # Controls — two lines when narrow, one line when wide
        _trades_hint = "  [↑/↓ or j/k] Select  [n/p] Page  [d] Detail  [b] Back" if self.current_tab == "trades" else ""
        _scroll_hint = "" if self.current_tab == "trades" else "  [Wheel/↑↓/j/k/Pg] Scroll"
        ctrl_wide = (
            "  [1-7/←/→] Tabs  |  [Tab/S+Tab] Cycle  |  [s] Presets  |  [r] Review CB  |  "
            f"[e] Epoch  |  [h] Help  |  [Alt+K] Kill  |  [q/^Q/^X] Quit{_scroll_hint}{_trades_hint}"
        )
        ctrl_line1 = "  [1-7/←/→] Tabs  |  [Tab/S+Tab] Cycle  |  [s] Presets  |  [r] Review CB"
        ctrl_line2 = f"  [h] Help  |  [Alt+K] Kill  |  [q/^Q/^X] Quit{_scroll_hint}{_trades_hint}"
        if len(ctrl_wide) <= width:
            print(ctrl_wide)
        else:
            print(ctrl_line1)
            print(ctrl_line2)

    def _footer_data_candidates(self) -> list[Path]:
        # Data freshness — use freshest file across all bots so one stale file
        # cannot trigger false "Bot silent" while others are active.
        candidates: list[Path] = []
        for _b in self.all_bots_stats or []:
            _sym = _b.get("symbol", "")
            _tf = int(_b.get("timeframe_minutes", 0) or 0)
            if _sym and _tf > 0:
                for candidate in (
                    self.data_dir / f"order_book_{_sym}_M{_tf}.json",
                    self.data_dir / f"bot_config_{_sym}_M{_tf}.json",
                    self.data_dir / f"paper_stats_{_sym}_M{_tf}.json",
                ):
                    if candidate.exists():
                        candidates.append(candidate)
        if not candidates:
            _ob = self.data_dir / _ORDER_BOOK_FILE
            _bc = self.data_dir / _BOT_CONFIG_FILE
            if _ob.exists():
                candidates.append(_ob)
            if _bc.exists():
                candidates.append(_bc)
        return candidates

    def _footer_data_freshness(self) -> str:
        candidates = self._footer_data_candidates()
        if not candidates:
            return f"{_ANSI_DIM}⏳ Waiting for data...{_ANSI_RST}"
        latest_mtime = max(path.stat().st_mtime for path in candidates)
        file_age = time.time() - latest_mtime
        if file_age > DATA_STALE_SECS:
            return f"{_ANSI_R}⚠️  Bot silent ({file_age:.0f}s){_ANSI_RST}"
        if file_age > DATA_AGING_SECS:
            return f"{_ANSI_Y}⚡ Data aging ({file_age:.0f}s){_ANSI_RST}"
        return f"{_ANSI_G}✓ Data fresh ({file_age:.1f}s){_ANSI_RST}"

    def _load_all_trades_cached(self) -> None:
        """Load trade_log.jsonl into self._all_trades (newest first), re-read at most once per 5 s."""
        _now = time.time()
        if _now - self._all_trades_loaded_at < 5.0:
            return
        trades = list(self._trade_log_reader.trades)
        for trade in trades:
            raw_mode = str(trade.get("trading_mode", "") or "").strip().lower()
            if raw_mode in ("paper", "live"):
                trade["trading_mode"] = raw_mode
            else:
                trade["trading_mode"] = self._resolve_trade_mode_from_record(trade)

        # Sort newest → oldest by exit_time, fallback to entry_time then trade_id
        def _skey(t: dict) -> tuple:
            return (t.get("exit_time") or t.get("entry_time") or "", t.get("trade_id", 0))

        trades.sort(key=_skey, reverse=True)
        self._all_trades = trades
        self._all_trades_loaded_at = _now

    @staticmethod
    def _capture_ratio_for_trade(trade: dict) -> float | None:
        """Return normalized capture ratio for a trade.

        Preference order:
        1) derived ``pnl_points / mfe_points`` — price units, accurate for all symbols/lot sizes;
        2) stored ``capture_ratio`` / ``capture_pct`` — only used when points fields absent
           (stored values are unreliable: computed from pnl_usd/mfe_usd where mfe_usd may use
           wrong qty×contract_size, causing 100x errors on BTCUSD and 357%+ errors on XAUUSD).

        pnl/mfe (USD ratio) is never used because mfe is stored in price points for BTCUSD
        and with an inconsistent multiplier for XAUUSD.
        """

        def _to_float(v: Any) -> float | None:
            if isinstance(v, bool):
                return None
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return None
                try:
                    return float(s)
                except ValueError:
                    return None
            return None

        pnl_f = float(_to_float(trade.get("pnl")) or 0.0)

        # Primary: derive from entry/exit prices — stored pnl_points is unreliable
        # (normalize_trade_log_scale.py wrote pnl/qty/contract_size instead of price diff).
        derived: float | None = None
        mfe_pts = _to_float(trade.get("mfe_points"))
        direction = str(trade.get("direction", "")).upper()
        entry = _to_float(trade.get("entry_price"))
        exit_ = _to_float(trade.get("exit_price"))
        if (mfe_pts is not None and abs(mfe_pts) > 1e-9
                and entry is not None and exit_ is not None
                and direction in ("LONG", "SHORT")):
            pnl_pts = (exit_ - entry) if direction == "LONG" else (entry - exit_)
            derived = pnl_pts / mfe_pts

        if derived is not None:
            return derived

        # Fallback: stored ratio when price-point fields are unavailable.
        stored_ratio = _to_float(trade.get("capture_ratio"))
        if stored_ratio is None:
            pct = _to_float(trade.get("capture_pct"))
            if pct is not None:
                stored_ratio = pct / 100.0

        if stored_ratio is not None:
            return stored_ratio

        # Last-resort: pnl/mfe — only valid when both happen to share the same units.
        mfe_f = float(_to_float(trade.get("mfe")) or 0.0)
        if abs(mfe_f) > 1e-9:
            return pnl_f / mfe_f

        return None

    @staticmethod
    def _excursion_usd_for_trade(trade: dict, pts_field: str, raw_field: str) -> float:
        """Return MFE or MAE in USD, converting from price points via derived qty.

        For BTCUSD the stored mfe/mae fields are price points, not USD.
        qty is derived from pnl / pnl_points when not explicitly stored.
        """
        def _f(v: Any) -> float | None:
            if v is None or isinstance(v, bool):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        pts = _f(trade.get(pts_field)) or _f(trade.get(raw_field))
        if not pts:
            return 0.0

        qty = _f(trade.get("quantity"))
        if qty is None or qty <= 0.0:
            qty = _f(trade.get("qty"))
        if qty is None or qty <= 0.0:
            direction = str(trade.get("direction", "")).upper()
            pnl = _f(trade.get("pnl"))
            entry = _f(trade.get("entry_price"))
            exit_ = _f(trade.get("exit_price"))
            if pnl is not None and entry is not None and exit_ is not None:
                pnl_pts = (exit_ - entry) if direction == "LONG" else (entry - exit_)
                if abs(pnl_pts) > 1e-9:
                    qty = pnl / pnl_pts

        if qty is None or qty <= 0.0:
            return pts

        contract_size = _f(trade.get("contract_size")) or 1.0
        return pts * qty * contract_size

    @staticmethod
    def _page_scrollbar(page: int, max_page: int, width: int = 18) -> str:
        """Compact page-position indicator for paged tabs."""
        if max_page <= 0:
            return "█" * width
        pos = min(width - 1, max(0, round((page / max_page) * (width - 1))))
        return "".join("█" if i == pos else "─" for i in range(width))

    def _trade_rows_per_page(self) -> int:
        """Return trade rows that fit in the fixed body viewport."""
        # Fixed HUD chrome:
        # header+tab bar is 11 rows in normal renders; footer is 6 rows.
        # Trades is intentionally compact: title, optional epoch line, table
        # header, and separator. It uses pagination rather than body scrolling.
        available_body = max(1, self._term_height() - 11 - 6)
        chrome_rows = 3
        row_budget = available_body - chrome_rows
        return max(3, min(80, row_budget))

    # ── Trades Tab — drill-down levels ─────────────────────────────────

    def _trades_for_mode(self, mode: str) -> list[dict]:
        """Return trades filtered to a single trading mode."""
        return [t for t in self._trade_log_all_trades if t.get("trading_mode") == mode]

    def _trades_for_symbol(self, mode: str, symbol: str) -> list[dict]:
        """Return trades for a given mode + symbol."""
        _sym = symbol.upper()
        return [t for t in self._trades_for_mode(mode) if str(t.get("symbol", "")).upper() == _sym]

    def _trades_for_symbol_tf(self, mode: str, symbol: str, tf_minutes: int) -> list[dict]:
        """Return trades for a given mode + symbol + timeframe."""
        return [
            t for t in self._trades_for_symbol(mode, symbol)
            if t.get("timeframe_minutes") == tf_minutes
        ]

    @staticmethod
    def _cursor_marker(is_selected: bool) -> str:
        """Return a visible cursor marker for the currently selected row."""
        return f"\033[7m {'>' if is_selected else ' '}\033[0m"

    def _render_trades_mode(self) -> None:
        """Level 0: Trading mode selection — Live / Paper / Offline."""
        print("\n\033[1m📋 TRADES — Select Trading Mode\033[0m\n")
        _modes = [
            ("live", "🔴 LIVE TRADING", "Real-money execution", _ANSI_G),
            ("paper", "🟡 PAPER TRADING", "Simulation / practice", _ANSI_Y),
            ("offline", "🔵 OFFLINE", "Backtest / validation (not account PnL)", _ANSI_B),
        ]
        _starting = self._universe_starting_equity()
        for _idx, (_mk, _label, _desc, _color) in enumerate(_modes):
            _trades = self._trades_for_mode(_mk) if _mk != "offline" else []
            _sel = " \033[7m" if _idx == self._ctx_cursor else "  "
            _end = "\033[0m" if _idx == self._ctx_cursor else ""
            _n = len(_trades)
            _pnl = sum(float(t.get("pnl", 0) or 0) for t in _trades) if _trades else 0.0
            _pc = self._pnl_color(_pnl) if _n > 0 else _ANSI_DIM
            print(
                f"{_sel}{_color}{_label:<30}{_end}  "
                f"{_ANSI_DIM}{_desc:<40}{_ANSI_RST}  "
                f"{_n:>6} trades  {_pc}{_pnl:>+10.2f}{_ANSI_RST}"
            )
        print(f"\n  {_ANSI_DIM}Use ↑/↓ to select mode, Enter to drill into portfolio{_ANSI_RST}")

    def _render_trades_portfolio(self) -> None:
        """Level 1: Per-instrument summary with period COLUMNS + mode stacked vertically."""
        self._render_breadcrumb("TRADES", 7)

        _COL_W = 20
        _SYM_W = 10
        _modes: list[tuple[str, str, str]] = [
            ("paper", f"{_ANSI_Y}📄 PAPER{_ANSI_RST}", _ANSI_Y),
            ("live", f"{_ANSI_G}💰 LIVE{_ANSI_RST}", _ANSI_G),
        ]
        _symbols = self._available_symbols()
        _starting = self._universe_starting_equity()
        _has_epoch = bool(self._stats_epoch)
        # Broad → narrow (left → right); Epoch only when a stats epoch is set
        _all_periods = ["Lifetime", "Epoch", "Month", "7d", "24h"] if _has_epoch else ["Lifetime", "Month", "7d", "24h"]
        _tw = self._term_width()
        _max_cols = max(2, (_tw - _SYM_W - 4) // (_COL_W + 2))
        _periods = _all_periods[:_max_cols]

        _row_idx = 0
        for _mode, _mode_label, _mode_color in _modes:
            _mode_trades = self._trades_for_mode(_mode)
            if not _mode_trades:
                continue

            print(f"\n  {_mode_label}")
            _hdr = f"  {'Symbol':<{_SYM_W}}"
            for _p in _periods:
                _hdr += f"  {_p:^{_COL_W}}"
            print(_hdr)
            print("  " + "─" * (_visible_width(_hdr) - 2))

            for _sym in _symbols:
                _sym_trades = [t for t in _mode_trades if str(t.get("symbol", "")).upper() == _sym]
                _cursor = self._cursor_marker(_row_idx == self._ctx_cursor)
                _row = f"{_cursor}{_sym:<{_SYM_W}}"
                _has_data = False
                for _p in _periods:
                    _filtered = _filter_trades_by_period_single(_p, _sym_trades)
                    if _filtered:
                        _m = _hud_period_metrics(_filtered, _starting)
                        _n = int(_m.get("total_trades", 0) or 0)
                        _wr = _m.get("win_rate", 0.0) * 100
                        _pnl = _m.get("total_pnl", 0.0)
                        _cell = _fmt_trade_period_cell(_n, _wr, _pnl, _COL_W)
                        _row += f"  {_cell}"
                        _has_data = True
                    else:
                        _row += f"  {'—':^{_COL_W}}"
                if _has_data:
                    print(_row)
                _row_idx += 1

        if not _symbols:
            print(f"\n  {_ANSI_DIM}No trades loaded yet.{_ANSI_RST}")
        _hint = " | ".join(_all_periods)
        print(f"\n  {_ANSI_DIM}{_hint}  —  ↑/↓ select, Enter drill{_ANSI_RST}")

    def _render_trades_symbol(self) -> None:
        """Level 2: Per-TF metrics for selected symbol with period COLUMNS."""
        _sym = self._ctx_symbol.upper()
        self._render_breadcrumb("TRADES", 7)

        _COL_W = 20
        _TF_W = 6
        _modes: list[tuple[str, str, str]] = [
            ("paper", f"{_ANSI_Y}📄 PAPER{_ANSI_RST}", _ANSI_Y),
            ("live", f"{_ANSI_G}💰 LIVE{_ANSI_RST}", _ANSI_G),
        ]
        _starting = self._universe_starting_equity()
        _has_epoch = bool(self._stats_epoch)
        _all_periods = ["Lifetime", "Epoch", "Month", "7d", "24h"] if _has_epoch else ["Lifetime", "Month", "7d", "24h"]
        _tw = self._term_width()
        _max_cols = max(2, (_tw - _TF_W - 4) // (_COL_W + 2))
        _periods = _all_periods[:_max_cols]
        _tf_order = [1, 5, 15, 30, 60, 240]

        _row_idx = 0
        _any_data = False
        for _mode, _mode_label, _mode_color in _modes:
            _trades = self._trades_for_symbol(_mode, _sym)
            if not _trades:
                continue
            _any_data = True

            print(f"\n  {_mode_label} › {_sym}")
            _hdr = f"  {'TF':<{_TF_W}}"
            for _p in _periods:
                _hdr += f"  {_p:^{_COL_W}}"
            print(_hdr)
            print("  " + "─" * (_visible_width(_hdr) - 2))

            for _tf in _tf_order:
                _tt = [t for t in _trades if t.get("timeframe_minutes") == _tf]
                _tf_label = self._format_timeframe_minutes_label(_tf)
                _cursor = self._cursor_marker(_row_idx == self._ctx_cursor)
                _row = f"{_cursor}{_tf_label:<{_TF_W}}"
                _has_tf_data = False
                for _p in _periods:
                    _filtered = _filter_trades_by_period_single(_p, _tt)
                    if _filtered:
                        _m = _hud_period_metrics(_filtered, _starting)
                        _n = int(_m.get("total_trades", 0) or 0)
                        _wr = _m.get("win_rate", 0.0) * 100
                        _pnl = _m.get("total_pnl", 0.0)
                        _cell = _fmt_trade_period_cell(_n, _wr, _pnl, _COL_W)
                        _row += f"  {_cell}"
                        _has_tf_data = True
                    else:
                        _row += f"  {'—':^{_COL_W}}"
                if _has_tf_data:
                    print(_row)
                _row_idx += 1

        if not _any_data:
            print(f"\n  {_ANSI_DIM}No trades for {_sym}{_ANSI_RST}")
        print(f"\n  {_ANSI_DIM}↑/↓ select timeframe, Enter drill into trade list{_ANSI_RST}")

    @staticmethod
    def _avg_capture_for_trades(trades: list[dict]) -> float | None:
        """Average capture ratio across trades that have positive MFE."""
        _caps = []
        for t in trades:
            _mfe = float(t.get("mfe", 0) or 0)
            _pnl = float(t.get("pnl", 0) or 0)
            if _mfe > 0:
                _caps.append(_pnl / _mfe)
        return sum(_caps) / len(_caps) if _caps else None

    def _render_trades_list(self) -> None:
        """Level 3: Individual trade list scoped to symbol + timeframe."""
        _sym = self._ctx_symbol.upper()
        _tf = self._ctx_tf
        self._render_breadcrumb("TRADES", 7)

        _mode_filter = self._mixed_mode_view_filter()
        trades_view = self._all_trades
        if _mode_filter:
            trades_view = [t for t in trades_view if t.get("trading_mode") == _mode_filter]
        # Scope to symbol + TF
        trades_view = [
            t for t in trades_view
            if str(t.get("symbol", "")).upper() == _sym and t.get("timeframe_minutes") == _tf
        ]
        self._trades_view = trades_view
        if self._trades_detail and self._trades_detail_trade not in trades_view:
            self._trades_detail = False
            self._trades_detail_trade = {}

        W = self._term_width()
        total = len(trades_view)
        _scope_label = f"{_sym}/{self._format_timeframe_minutes_label(_tf)}"
        if total == 0:
            _empty_label = f" ({_mode_filter})" if _mode_filter else ""
            print(f"\n\033[1m[T] TRADES › {_scope_label}\033[0m{_empty_label}  No trades recorded yet.")
            return
        self._trades_per_page = self._trade_rows_per_page()
        max_page = max(0, (total - 1) // self._trades_per_page)
        self._trades_page = min(self._trades_page, max_page)
        page_start = self._trades_page * self._trades_per_page
        page_trades = trades_view[page_start : page_start + self._trades_per_page]
        self._trades_cursor = min(self._trades_cursor, max(0, len(page_trades) - 1))

        # Header
        _all_for_header = self._trade_log_all_trades
        _epoch_for_header = self._trade_log_metrics_trades
        if _mode_filter:
            _all_for_header = [t for t in _all_for_header if t.get("trading_mode") == _mode_filter]
            _epoch_for_header = [t for t in _epoch_for_header if t.get("trading_mode") == _mode_filter]
        lm = _hud_period_metrics(_all_for_header, self._universe_starting_equity())
        epoch_lm = _hud_period_metrics(_epoch_for_header, self._universe_starting_equity())
        total_pnl = lm.get("total_pnl", 0.0)
        wins = lm.get("winning_trades", 0)
        losses = lm.get("losing_trades", 0)
        wr = lm.get("win_rate", 0.0) * 100
        pg_str = f"Pg {self._trades_page + 1}/{max_page + 1}"
        _tl_mode = getattr(self, "_trade_log_mode", "")
        if _tl_mode == "mixed":
            _mode_suffix = ""
            if _mode_filter == "paper":
                _mode_suffix = f"  {_ANSI_Y}📄 PAPER VIEW{_ANSI_RST}"
            elif _mode_filter == "live":
                _mode_suffix = f"  {_ANSI_G}💰 LIVE VIEW{_ANSI_RST}"
            _mode_hdr = f"  {_ANSI_Y}⚠ MIXED{_ANSI_RST}{_mode_suffix}"
        elif _tl_mode == "live":
            _mode_hdr = f"  {_ANSI_G}💰 LIVE{_ANSI_RST}"
        else:
            _mode_hdr = f"  {_ANSI_Y}📄 PAPER{_ANSI_RST}"
        _bar = self._page_scrollbar(self._trades_page, max_page)
        _compact = self._trades_per_page <= 6
        if _compact:
            print(
                f"\033[1m[T] TRADES › {_scope_label}\033[0m "
                f"[{total}] {pg_str} {_ANSI_DIM}{_bar}{_ANSI_RST} "
                f"PnL {self._pnl_color(total_pnl)}{total_pnl:+.2f}{_ANSI_RST}{_mode_hdr}"
            )
        else:
            print(
                f"\033[1m[T] TRADES › {_scope_label}\033[0m  "
                f"[{total} trades]  {pg_str}  {_ANSI_DIM}{_bar}{_ANSI_RST}{_mode_hdr}"
            )

        if self._trades_detail:
            self._render_trade_detail(self._trades_detail_trade)
            return

        # Mixed-mode banner — operator must know metrics are contaminated
        if _tl_mode == "mixed":
            if _mode_filter:
                _mode_lbl = "paper" if _mode_filter == "paper" else "live"
                print(
                    f"  {_ANSI_Y}⚠  MIXED MODE SOURCE — showing {_mode_lbl} trades only. "
                    f"Use bot mode switch to view the other mode.{_ANSI_RST}"
                )

        if not _compact:
            # Summary bar
            _pnl_c = _ANSI_G if total_pnl >= 0 else _ANSI_R
            print(
                f"  Lifetime PnL: {_pnl_c}{total_pnl:+.2f}{_ANSI_RST}  |  "
                f"W/L: {_ANSI_G}{wins}{_ANSI_RST}/{_ANSI_R}{losses}{_ANSI_RST}  "
                f"({_ANSI_G if wr >= 50 else _ANSI_R}{wr:.1f}%{_ANSI_RST} win rate)"
            )
        if self._stats_epoch and not _compact:
            _ep_pnl = epoch_lm.get("total_pnl", 0.0)
            _ep_tr = epoch_lm.get("total_trades", 0)
            print(
                f"  Epoch: {self._pnl_color(_ep_pnl)}{_ep_pnl:+.2f}{_ANSI_RST} / {_ep_tr} trades"
                f"  {_ANSI_DIM}(excluded {self._stats_epoch_excluded}; source: trade_log.jsonl){_ANSI_RST}"
            )

        # Column header — M = mode badge (P=paper / L=live)
        _C_ID = 8
        _C_DATE = 14
        _C_DIR = 5
        _C_SYM = 9
        _C_TF = 4
        _C_ENT = 9
        _C_EXT = 9
        _C_PNL = 12
        _C_CAP = 7
        _C_MFE = 9
        _C_MAE = 9
        _C_BRS = 5
        _C_RSN = 18
        _compact_table = W < 132
        if _compact_table:
            _C_ID = 5
            _C_DATE = 11
            _C_BOT = 12
            _C_DIR = 1
            _C_PNL = 9
            _C_CAP = 6
            _C_MFE = 8
            _C_MAE = 8
            _C_BRS = 4
            _C_RSN = 12
            _hdr_row = (
                f"  {'#':<{_C_ID}} M {'Time':<{_C_DATE}} {'Bot':<{_C_BOT}} {'D':<{_C_DIR}} "
                f"{'PnL $':>{_C_PNL}} {'Cap%':>{_C_CAP}} {'MFE $':>{_C_MFE}} {'MAE $':>{_C_MAE}} "
                f"{'Bars':>{_C_BRS}}  {'Reason':<{_C_RSN}}"
            )
        else:
            _hdr_row = (
                f"  {'#':<{_C_ID}} M {'Date/Time':<{_C_DATE}} {'Dir':<{_C_DIR}} "
                f"{'Sym':<{_C_SYM}} {'TF':<{_C_TF}} {'Entry':>{_C_ENT}} {'Exit':>{_C_EXT}} "
                f"{'PnL $':>{_C_PNL}} {'Cap%':>{_C_CAP}} {'MFE $':>{_C_MFE}} {'MAE $':>{_C_MAE}} "
                f"{'Bars':>{_C_BRS}}  {'Reason':<{_C_RSN}}"
            )
        # Plain-text header width (no ANSI) → matches the rendered row width.
        _hdr_plain_len = len(_hdr_row)
        _sep = "  " + "-" * max(10, min(W - 4, _hdr_plain_len - 2))
        print(f"{_ANSI_DIM}{_hdr_row}{_ANSI_RST}")
        print(_sep)

        # Trade rows — derive decimal places from first trade's price so we
        # never fall back to 5dp when bot_config is empty.
        _sample_price = 0.0
        if page_trades:
            _sample_price = float(page_trades[0].get("entry_price") or 0.0)
        _dec = self._price_decimals(_sample_price)
        for _row_idx, _t in enumerate(page_trades):
            _tid = _t.get("trade_id", page_start + _row_idx + 1)
            _dir = (_t.get("direction") or "").upper()
            _sym = _t.get("symbol", "?")
            _tf = self._normalize_timeframe_label(_t)
            _entry = float(_t.get("entry_price") or 0.0)
            _exit = float(_t.get("exit_price") or 0.0)
            _pnl = float(_t.get("pnl") or 0.0)
            _mfe = self._excursion_usd_for_trade(_t, "mfe_points", "mfe")
            _mae = self._excursion_usd_for_trade(_t, "mae_points", "mae")
            _bars = int(_t.get("bars_held") or _t.get("ticks_held") or 0)
            _rsn_raw = _t.get("close_reason") or _t.get("exit_reason") or ""
            _rsn = ("-" if _rsn_raw in ("", "unknown") else _rsn_raw)[:_C_RSN]
            _tid_s = str(_tid)
            if len(_tid_s) > _C_ID:
                _tid_s = _tid_s[: max(0, _C_ID - 1)] + "…"
            _sym_s = str(_sym)
            if len(_sym_s) > _C_SYM:
                _sym_s = _sym_s[: max(0, _C_SYM - 1)] + "…"
            _bot_s = f"{_sym}/{_tf}"
            if len(_bot_s) > 12:
                _bot_s = _bot_s[:11] + "…"

            _ts = _t.get("exit_time") or _t.get("entry_time") or ""
            try:
                _dt = datetime.fromisoformat(_ts)
                _date_str = _dt.strftime("%m-%d %H:%M") if _compact_table else _dt.strftime("%b-%d %H:%M")
            except Exception:
                _date_str = _ts[:13]
            _date_s = _date_str[:_C_DATE] if _compact_table else _date_str

            _dc = _ANSI_G if _dir == "LONG" else (_ANSI_R if _dir == "SHORT" else _ANSI_DIM)
            _pc = self._pnl_color(_pnl)
            _ep_s = f"{_entry:.{_dec}f}"[-_C_ENT:]
            _xp_s = f"{_exit:.{_dec}f}"[-_C_EXT:]

            _cap_ratio = self._capture_ratio_for_trade(_t)
            if _cap_ratio is not None:
                _cap_val = max(-999.0, min(999.0, _cap_ratio * 100.0))
                _cap_s = f"{_cap_val:+.0f}%"
                _cap_c = self._pnl_color(_cap_val)
            elif float(_pnl or 0.0) < 0.0:
                # No positive excursion => capture is undefined; flag losses red.
                _cap_s = "n/a"
                _cap_c = _ANSI_R
            else:
                _cap_s = "—"
                _cap_c = _ANSI_DIM

            # Mode badge — single char, always renders 1 column wide
            _tmode = _t.get("trading_mode", "")
            if _tmode == "paper":
                _mb = f"{_ANSI_Y}P{_ANSI_RST}"
            elif _tmode == "live":
                _mb = f"{_ANSI_G}L{_ANSI_RST}"
            else:
                _mb = f"{_ANSI_DIM}?{_ANSI_RST}"

            _pnl_s = _fmt_compact(_pnl, _C_PNL)
            _mfe_s = _fmt_compact(_mfe, _C_MFE)
            _mae_s = _fmt_compact(-abs(_mae), _C_MAE)
            if _compact_table:
                _dir_s = "L" if _dir == "LONG" else ("S" if _dir == "SHORT" else "?")
                _row = (
                    f"  {_tid_s:<{_C_ID}} {_mb} {_date_s:<{_C_DATE}} "
                    f"{_bot_s:<{_C_BOT}} {_dc}{_dir_s:<{_C_DIR}}{_ANSI_RST} "
                    f"{_ansi_cell(_pnl_s, _pc, _C_PNL)} "
                    f"{_ansi_cell(_cap_s, _cap_c, _C_CAP)} "
                    f"{_ansi_cell(_mfe_s, _ANSI_G, _C_MFE)} "
                    f"{_ansi_cell(_mae_s, _ANSI_R, _C_MAE)} "
                    f"{_bars:>{_C_BRS}}  {_ANSI_DIM}{_rsn:<{_C_RSN}}{_ANSI_RST}"
                )
            else:
                _row = (
                    f"  {_tid_s:<{_C_ID}} {_mb} {_date_s:<{_C_DATE}} "
                    f"{_dc}{_dir:<{_C_DIR}}{_ANSI_RST} "
                    f"{_sym_s:<{_C_SYM}} {_tf:<{_C_TF}} {_ep_s:>{_C_ENT}} {_xp_s:>{_C_EXT}} "
                    f"{_ansi_cell(_pnl_s, _pc, _C_PNL)} "
                    f"{_ansi_cell(_cap_s, _cap_c, _C_CAP)} "
                    f"{_ansi_cell(_mfe_s, _ANSI_G, _C_MFE)} "
                    f"{_ansi_cell(_mae_s, _ANSI_R, _C_MAE)} "
                    f"{_bars:>{_C_BRS}}  {_ANSI_DIM}{_rsn:<{_C_RSN}}{_ANSI_RST}"
                )
            if _row_idx == self._trades_cursor:
                print(f"\033[7m{_strip_ansi(_row)}\033[0m")
            else:
                print(_row)

        if self._trades_per_page >= 8:
            print(_sep)

    def _render_trades(self) -> None:
        """Trades tab — dispatches to the correct drill-down level renderer."""
        if self._ctx_level == 0:
            self._render_trades_mode()
        elif self._ctx_level == 1:
            self._render_trades_portfolio()
        elif self._ctx_level == 2:
            self._render_trades_symbol()
        elif self._ctx_level == 3:
            self._render_trades_list()
        elif self._ctx_level == 4:
            if self._trades_detail and self._trades_detail_trade:
                self._render_trade_detail(self._trades_detail_trade)
            else:
                self._ctx_level = 3
                self._render_trades_list()

    def _render_trade_detail(self, t: dict) -> None:
        """Render full detail card for a single trade."""
        W = self._term_width()
        _sep = "  " + "-" * min(W - 4, 74)
        _tid = t.get("trade_id", "?")
        _tick = t.get("ticket", "-")
        _pid = t.get("position_id", "-")
        _sym = t.get("symbol", "?")
        _mode = t.get("trading_mode", "?")
        _dir = (t.get("direction") or "").upper()
        _entry = float(t.get("entry_price") or 0.0)
        _exit = float(t.get("exit_price") or 0.0)
        _pnl = float(t.get("pnl") or 0.0)
        _mfe = self._excursion_usd_for_trade(t, "mfe_points", "mfe")
        _mae = self._excursion_usd_for_trade(t, "mae_points", "mae")
        _bars = int(t.get("bars_held") or t.get("ticks_held") or 0)
        _rsn_raw = t.get("close_reason") or t.get("exit_reason") or ""
        _rsn = "-" if _rsn_raw in ("", "unknown") else _rsn_raw
        _w2l = t.get("winner_to_loser", False)

        _entry_ts = t.get("entry_time", "")
        _exit_ts = t.get("exit_time", "")
        try:
            _edt = datetime.fromisoformat(_entry_ts)
            _entry_str = _edt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            _entry_str = _entry_ts[:19]
        try:
            _xdt = datetime.fromisoformat(_exit_ts)
            _exit_str = _xdt.strftime("%Y-%m-%d %H:%M:%S UTC")
            _edt2 = datetime.fromisoformat(_entry_ts)
            _dur = (_xdt - _edt2).total_seconds()
            _dm, _ds = divmod(int(_dur), 60)
            _dh, _dm = divmod(_dm, 60)
            _dur_str = (f"{_dh}h " if _dh else "") + f"{_dm}m {_ds}s"
        except Exception:
            _exit_str = _exit_ts[:19]
            _dur_str = "-"

        _dc = _ANSI_G if _dir == "LONG" else (_ANSI_R if _dir == "SHORT" else _ANSI_DIM)
        _pc = self._pnl_color(_pnl)
        _dec = self._price_decimals(max(_entry, _exit, 0.0))
        _ratio_str = f"  (MFE/MAE: {_mfe / _mae:.2f}x)" if _mae > 0 else ""

        print(_sep)
        print(f"\n  \033[1mTRADE #{_tid}\033[0m  {_dc}{_dir}{_ANSI_RST}  {_sym}  ({_mode})")
        print(_sep)
        print(f"  {'Ticket:':<16} {_ANSI_DIM}{_tick}{_ANSI_RST}")
        print(f"  {'Position ID:':<16} {_ANSI_DIM}{_pid}{_ANSI_RST}")
        print(f"  {'Entry:':<16} {_entry:.{_dec}f}  @  {_entry_str}")
        print(f"  {'Exit:':<16} {_exit:.{_dec}f}  @  {_exit_str}")
        print(f"  {'Duration:':<16} {_dur_str}  ({_bars} bars)")
        print()
        _result = f"  {_ANSI_G}[+] WIN{_ANSI_RST}" if _pnl > 0 else f"  {_ANSI_R}[-] LOSS{_ANSI_RST}"
        print(f"  {'PnL:':<16} {_pc}{_pnl:+.4f} USD{_ANSI_RST}{_result}")
        _cap_ratio = self._capture_ratio_for_trade(t)
        if _cap_ratio is not None:
            _cap_pct = max(-999.0, min(999.0, _cap_ratio * 100.0))
            _cc = self._pnl_color(_cap_pct)
            print(f"  {'%MFE captured:':<16} {_cc}{_cap_pct:+.1f}%{_ANSI_RST}  (normalized capture_ratio)")
        elif float(_pnl or 0.0) < 0.0:
            print(f"  {'%MFE captured:':<16} {_ANSI_R}n/a{_ANSI_RST}  (loss with zero MFE)")
        else:
            print(f"  {'%MFE captured:':<16} {_ANSI_DIM}—{_ANSI_RST}")
        print(f"  {'MFE:':<16} {_ANSI_G}+{_mfe:.4f} USD{_ANSI_RST}  (max favorable excursion, account currency)")
        print(
            f"  {'MAE:':<16} {_ANSI_R}-{_mae:.4f} USD{_ANSI_RST}  (max adverse excursion, account currency){_ratio_str}"
        )
        print(f"  {'Close reason:':<16} {_rsn}")
        _ts = self.training_stats if isinstance(self.training_stats, dict) else {}
        _rw_total = int(_ts.get("trigger_runway_cal_total_samples", 0) or 0)
        _rw_reliable = bool(_ts.get("trigger_runway_predictor_reliable", False))
        _rw_col = _ANSI_G if _rw_reliable else _ANSI_Y
        _rw_lbl = "RELIABLE (gate active)" if _rw_reliable else "LEARNING (gate bypass)"
        _cd = float(_ts.get("harvester_capture_decay_threshold", 0.0) or 0.0)
        _mw = float(_ts.get("harvester_micro_winner_giveback_pct", 0.0) or 0.0)
        print(f"  {'Runway model:':<16} {_rw_col}{_rw_lbl}{_ANSI_RST}  (samples={_rw_total})")
        print(f"  {'WTL protection:':<16} capture_decay<{_cd:.2f}  micro_giveback>{_mw:.2f}×MFE")
        if _w2l:
            print(f"  {_ANSI_Y}[!] Winner-to-Loser: trade reversed into a loss after reaching MFE{_ANSI_RST}")

        # Broker transaction events linked by position_id
        _sym_t = str(t.get("symbol") or "")
        _tf_t = int(t.get("timeframe_minutes") or 0)
        _tx_events = self._load_transactions_for_position(_pid, sym=_sym_t, tf_m=_tf_t)
        if _tx_events:
            print(f"\n  {'BROKER EVENTS':─<68}")
            for _tx in _tx_events:
                _tx_ts = str(_tx.get("timestamp") or "?")[:19]
                _tx_et = _tx.get("event_type", "?")
                _tx_d = _tx.get("data") or {}
                if _tx_et == "POSITION_OPEN":
                    _ep = float(_tx_d.get("entry_price") or 0.0)
                    _dr = str(_tx_d.get("direction") or "?").upper()
                    _qty = float(_tx_d.get("quantity") or 0.0)
                    _conf = float(_tx_d.get("entry_confidence") or 0.0)
                    _dc2 = _ANSI_G if _dr == "LONG" else _ANSI_R
                    print(
                        f"  {_ANSI_G}OPEN {_ANSI_RST} {_tx_ts}  {_dc2}{_dr}{_ANSI_RST}"
                        f"  {_ep:.{_dec}f}  qty={_qty}  conf={_conf:.4f}"
                    )
                    _eg = _tx_d.get("entry_trigger_data") or {}
                    _eg_gates = _eg.get("entry_gated_conditions") or []
                    if _eg_gates:
                        for _g in _eg_gates:
                            print(f"  {'':6}{_ANSI_R}✗ {_g}{_ANSI_RST}")
                elif _tx_et == "POSITION_CLOSE":
                    _xp = float(_tx_d.get("exit_price") or 0.0)
                    _xpnl = float(_tx_d.get("pnl") or 0.0)
                    _xrsn = str(_tx_d.get("close_reason") or "?")
                    _xcap = _tx_d.get("capture_ratio")
                    _xcap_str = f"  cap={_xcap:.3f}" if _xcap is not None else ""
                    _pc2 = self._pnl_color(_xpnl)
                    print(
                        f"  {_ANSI_R}CLOSE{_ANSI_RST} {_tx_ts}  {_xp:.{_dec}f}"
                        f"  {_pc2}PnL={_xpnl:+.4f}{_ANSI_RST}{_xcap_str}  [{_xrsn}]"
                    )

        print()
        print(f"  {_ANSI_DIM}[d] or [b] - return to trade list{_ANSI_RST}")

    def _pnl_color(self, pnl: float) -> str:
        """Return color code for PnL."""
        if pnl > 0:
            return _ANSI_G
        if pnl < 0:
            return _ANSI_R
        return _ANSI_Y

    def _create_sparkline(self, values: list) -> str:
        """Create a sparkline (mini-chart) from a list of values.

        For PnL sequences the midpoint of the bar scale is pinned at zero so
        a sequence of purely negative values correctly shows descending bars
        instead of misleadingly ascending ones.
        """
        if not values:
            return ""

        # Sparkline characters from lowest to highest
        chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        mid_idx = len(chars) // 2  # index that represents zero

        min_val = min(values)
        max_val = max(values)

        # Zero-pinned scale: use symmetric range so zero always maps to mid_idx.
        # Fall back to value-range normalisation for non-PnL (all-positive) data.
        has_mixed_signs = min_val < 0 < max_val or (min_val == 0) or (max_val == 0)
        half_range = max(abs(min_val), abs(max_val)) if has_mixed_signs else 0.0

        sparkline = ""
        for val in values:
            if has_mixed_signs and half_range > 0:
                # Map [-half_range, +half_range] → [0, len(chars)-1]
                normalized = (val + half_range) / (2 * half_range)
            elif max_val != min_val:
                normalized = (val - min_val) / (max_val - min_val)
            else:
                normalized = 0.5
            idx = max(0, min(len(chars) - 1, int(normalized * (len(chars) - 1))))

            # Color positive values green, negative red
            if val > 0:
                sparkline += _ANSI_G + chars[idx] + _ANSI_RST
            elif val < 0:
                sparkline += _ANSI_R + chars[idx] + _ANSI_RST
            else:
                sparkline += _ANSI_Y + chars[mid_idx] + _ANSI_RST
        return sparkline


_HUD_PIDFILE = Path("/tmp/ctrader_hud.pid")


def _acquire_pidfile() -> bool:
    """Write our PID to the pidfile, killing any stale predecessor first.

    Returns True if we acquired the lock, False if another live instance
    is already running (caller should exit cleanly).
    """
    if _HUD_PIDFILE.exists():
        try:
            old_pid = int(_HUD_PIDFILE.read_text().strip())
            os.kill(old_pid, 0)  # raises OSError if process is dead
            # Process is alive — refuse to start a second instance
            print(f"HUD already running (PID {old_pid}). Use 'kill {old_pid}' to stop it first.")
            return False
        except (OSError, ValueError):
            pass  # stale pidfile — safe to overwrite
    _HUD_PIDFILE.write_text(str(os.getpid()))
    return True


def _release_pidfile() -> None:
    """Remove the pidfile if it still contains our PID."""
    try:
        if _HUD_PIDFILE.exists() and int(_HUD_PIDFILE.read_text().strip()) == os.getpid():
            _HUD_PIDFILE.unlink()
    except Exception:
        pass


def main() -> None:
    """Run tabbed HUD."""
    if not _acquire_pidfile():
        sys.exit(1)

    # Redirect all logging to file so warnings don't flash on the TUI
    logging.basicConfig(
        filename="logs/hud.log",
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Starting Tabbed HUD...")
    print("Reading from: data/*.json")
    print()

    hud = TabbedHUD(refresh_rate=1.0)

    # Ignore Ctrl+C in HUD to prevent accidental termination when copying
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        hud.start()
        while hud.running:
            time.sleep(0.1)
    finally:
        hud.stop()
        _release_pidfile()
        print("\n\nHUD stopped.")


if __name__ == "__main__":
    main()
