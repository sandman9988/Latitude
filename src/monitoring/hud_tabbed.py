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
import glob as _glob
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

    TABS: ClassVar[dict[str, str]] = {"1": "overview", "2": "performance", "3": "training", "4": "risk", "5": "market", "6": "log", "7": "trades"}

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

        # Data sources
        self.data_dir = Path("data")

        # State
        self.position = {}
        self.metrics = {}
        self.training_stats = {}
        self.risk_stats = {}
        self.market_stats = {}
        self.bot_config = {}
        self.production_metrics = {}
        self.offline_stats: dict = {}  # offline_training_status.json
        self.offline_job_progress: dict = {}  # keyed by (symbol, tf_minutes)
        self.universe_stats: dict = {}  # universe.json + PID liveness
        self.all_bots_stats: list = []  # one entry per paper_stats_*.json
        self.training_stats_all: list = []  # one entry per training_stats_*_M*.json
        # Active-position bot identity (populated each refresh from position file metadata)
        self.active_sym: str = ""
        self.active_tf_min: int = 0
        self._active_pos_file: str = ""  # path of the file that provided self.position
        self.self_test_results: list = []
        self._health_report: dict = {}  # data/performance_health.json — self-healing analyzer
        self._metrics_from_trade_log = False
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

        # Heartbeat
        self.heartbeat_idx = 0
        self.heartbeat_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        # Terminal settings for non-blocking input
        self.old_settings = None

        # Trade history tab state
        self._trades_page: int = 0
        self._trades_per_page: int = 22
        self._trades_cursor: int = 0
        self._trades_detail: bool = False
        self._trades_detail_trade: dict = {}
        self._all_trades: list = []  # newest-first sorted
        self._trades_view: list = []  # current filtered view for trades tab
        self._all_trades_loaded_at: float = 0.0
        self._trade_log_reader = CachedTradeLogReader(self.data_dir / "trade_log.jsonl")

        # Stats epoch: trades before this timestamp are excluded from metrics
        self._stats_epoch: datetime | None = None
        self._stats_epoch_excluded: int = 0  # count of excluded trades
        self._stats_epoch_excluded_pnl: float = 0.0  # PnL of excluded trades
        self._load_stats_epoch()

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
            ("7 days", _hud_period_metrics(_weekly, _starting)),
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
            self._force_redraw = True

    def _handle_mouse_event(self, seq: str) -> None:
        """Handle SGR mouse events: clicks on tabs, wheel scroll in body."""
        match = re.match(r"<(\d+);(\d+);(\d+)([mM])", seq)
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

    def _handle_escape_sequence(self, seq1: str) -> None:
        """Handle CSI / Alt-key escape sequences following the ESC byte."""
        if seq1 == "[":  # CSI sequence (e.g. Shift+Tab = \x1b[Z)
            if select.select([sys.stdin.fileno()], [], [], 0.05)[0]:
                seq2 = self._read_raw()
                if seq2 == "Z":  # Shift+Tab
                    idx = self.TAB_ORDER.index(self.current_tab)
                    self.current_tab = self.TAB_ORDER[(idx - 1) % len(self.TAB_ORDER)]
                    self._force_redraw = True
                elif seq2 in {"C", "D"}:  # Right/Left arrows cycle tabs
                    idx = self.TAB_ORDER.index(self.current_tab)
                    step = 1 if seq2 == "C" else -1
                    self.current_tab = self.TAB_ORDER[(idx + step) % len(self.TAB_ORDER)]
                    self._force_redraw = True
                elif seq2 in {"A", "B"} and self.current_tab == "trades" and not self._trades_detail:
                    page_cnt = min(
                        self._trades_per_page,
                        len(self._trades_view) - self._trades_page * self._trades_per_page,
                    )
                    step = -1 if seq2 == "A" else 1
                    self._trades_cursor = max(0, min(self._trades_cursor + step, max(0, page_cnt - 1)))
                    self._force_redraw = True
                elif seq2 in {"A", "B"}:
                    self._scroll_current_body(-1 if seq2 == "A" else 1)
                elif seq2 in {"H", "F"}:
                    self._scroll_current_body(absolute=0 if seq2 == "H" else self._body_scroll_max)
                elif seq2 in {"5", "6"} and select.select([sys.stdin.fileno()], [], [], 0.02)[0]:
                    if self._read_raw() == "~":  # PageUp/PageDown
                        self._scroll_current_body(-10 if seq2 == "5" else 10)
                elif seq2 == "<":
                    mouse_seq = ""
                    deadline = time.monotonic() + 0.05
                    while time.monotonic() < deadline and select.select([sys.stdin.fileno()], [], [], 0.005)[0]:
                        ch = self._read_raw()
                        mouse_seq += ch
                        if ch in {"m", "M"}:
                            break
                    self._handle_mouse_event("<" + mouse_seq)
        elif seq1.lower() == "k":  # Alt+K — emergency kill switch (handle both 'k' and 'K')
            self._handle_kill_switch()

    def _check_input(self):
        """Check for keyboard input (non-blocking).

        Drains queued bytes so rapid tab-cycling never lags behind by multiple
        refresh cycles.
        """
        _handled = False
        try:
            _fd = sys.stdin.fileno()
            _drained = 0
            while _drained < INPUT_DRAIN_MAX and select.select([_fd], [], [], 0)[0]:
                _drained += 1
                _handled = True
                key = self._read_raw()
                if key in self.TABS:
                    self.current_tab = self.TABS[key]
                    self._force_redraw = True
                elif key == "\t":  # Tab key to cycle forward
                    idx = self.TAB_ORDER.index(self.current_tab)
                    self.current_tab = self.TAB_ORDER[(idx + 1) % len(self.TAB_ORDER)]
                    self._force_redraw = True
                elif key == "\x1b":  # Escape sequence: Shift+Tab or Alt+<key>
                    if select.select([sys.stdin.fileno()], [], [], 0.05)[0]:
                        seq1 = self._read_raw()
                        self._handle_escape_sequence(seq1)
                elif key.lower() == "q" or key in {"\x18", "\x11"}:  # 'q' or Ctrl+X (\x18) or Ctrl+Q (\x11)
                    self.running = False
                elif key.lower() == "s":
                    self._handle_session_selector()
                elif key.lower() == "r":
                    self._handle_cb_reset()
                elif key.lower() == "e":
                    self._handle_stats_epoch()
                elif key.lower() == "h":
                    self._show_help()
                    self._force_redraw = True
                elif key.lower() == "n":
                    if self.current_tab == "trades":
                        self._trades_detail = False
                        _max_pg = max(0, (len(self._trades_view) - 1) // self._trades_per_page)
                        self._trades_page = min(self._trades_page + 1, _max_pg)
                        self._trades_cursor = 0
                        self._force_redraw = True
                elif key.lower() == "p":
                    if self.current_tab == "trades":
                        self._trades_detail = False
                        self._trades_page = max(0, self._trades_page - 1)
                        self._trades_cursor = 0
                        self._force_redraw = True
                elif key.lower() == "j":
                    if self.current_tab == "trades" and not self._trades_detail:
                        _page_cnt = min(
                            self._trades_per_page, len(self._trades_view) - self._trades_page * self._trades_per_page
                        )
                        self._trades_cursor = min(self._trades_cursor + 1, max(0, _page_cnt - 1))
                        self._force_redraw = True
                    else:
                        self._scroll_current_body(1)
                elif key.lower() == "k":
                    if self.current_tab == "trades" and not self._trades_detail:
                        self._trades_cursor = max(0, self._trades_cursor - 1)
                        self._force_redraw = True
                    else:
                        self._scroll_current_body(-1)
                elif key.lower() == "d":
                    if self.current_tab == "trades":
                        if self._trades_detail:
                            self._trades_detail = False
                            self._force_redraw = True
                        else:
                            _idx = self._trades_page * self._trades_per_page + self._trades_cursor
                            if _idx < len(self._trades_view):
                                self._trades_detail_trade = self._trades_view[_idx]
                                self._trades_detail = True
                                self._force_redraw = True

                elif key.lower() == "b" and self.current_tab == "trades" and self._trades_detail:
                    self._trades_detail = False
                    self._force_redraw = True
        except Exception:
            pass
        return _handled

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
            self._health_report = json.loads(_path.read_text())
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

    def _iter_universe_entries(self, uni_raw: dict) -> list[dict]:
        """Return normalized universe entries supporting old and new schemas."""
        entries: list[dict] = []
        instruments = uni_raw.get("instruments", {})
        if isinstance(instruments, list):
            for item in instruments:
                if not isinstance(item, dict):
                    continue
                _entry = dict(item)
                if not _entry.get("symbol"):
                    _entry["symbol"] = str(_entry.get("_symbol", "") or "")
                if _entry.get("symbol"):
                    _entry["symbol"] = str(_entry["symbol"]).upper()
                entries.append(_entry)
            return entries
        if isinstance(instruments, dict):
            for sym, item in instruments.items():
                if isinstance(item, list):
                    for sub in item:
                        if not isinstance(sub, dict):
                            continue
                        _entry = dict(sub)
                        _entry.setdefault("symbol", str(sym).upper())
                        entries.append(_entry)
                elif isinstance(item, dict):
                    _entry = dict(item)
                    _entry.setdefault("symbol", str(sym).upper())
                    entries.append(_entry)
            return entries
        if isinstance(uni_raw, dict):
            for sym, item in uni_raw.items():
                if sym in ("version", "instruments"):
                    continue
                if isinstance(item, list):
                    for sub in item:
                        if not isinstance(sub, dict):
                            continue
                        _entry = dict(sub)
                        _entry.setdefault("symbol", str(sym).upper())
                        entries.append(_entry)
                elif isinstance(item, dict):
                    _entry = dict(item)
                    _entry.setdefault("symbol", str(sym).upper())
                    entries.append(_entry)
        return entries

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

        # order_book.json is fresher — overwrite book-specific fields
        ob_file = self._preferred_data_file(_ORDER_BOOK_FILE)
        if not ob_file.exists():
            return
        try:
            with open(ob_file) as f:
                ob = json.load(f)
            ms = self.market_stats
            ms["spread"] = ob.get("spread", ms.get("spread", 0.0))
            ms["depth_bid"] = ob.get("depth_bid", ms.get("depth_bid", 0.0))
            ms["depth_ask"] = ob.get("depth_ask", ms.get("depth_ask", 0.0))
            ms["order_book_bids"] = ob.get("order_book_bids", ms.get("order_book_bids", []))
            ms["order_book_asks"] = ob.get("order_book_asks", ms.get("order_book_asks", []))
            ms["vpin"] = ob.get("vpin", ms.get("vpin", 0.0))
            ms["vpin_z"] = ob.get("vpin_zscore", ms.get("vpin_z", 0.0))
            _ob_imb = ob.get("imbalance")
            if _ob_imb is not None:
                ms["imbalance"] = float(_ob_imb)
            ms["has_real_sizes"] = ob.get("has_real_sizes", False)
            ms["qfi_update_count"] = ob.get("qfi_update_count", 0)
            ms["next_bar_close_utc"] = ob.get("next_bar_close_utc")
            ms["timeframe_minutes"] = ob.get("timeframe_minutes")
        except Exception:
            LOG.debug("[HUD] Failed to load risk/orderbook data", exc_info=True)

    def _refresh_data(self) -> None:
        """Refresh all data from bot exports."""
        self.last_update = datetime.now(UTC)
        self.heartbeat_idx = (self.heartbeat_idx + 1) % len(self.heartbeat_chars)

        if not self.data_dir.exists():
            self._set_notification(f"⚠️  Data directory not found: {self.data_dir}", ttl=30)
            return

        self._load_json(_BOT_CONFIG_FILE, "bot_config")

        # ── Fallback: read live balance from Open API file if bot hasn't set it ──
        if not self.bot_config.get("real_account_balance"):
            _bal_file = self.data_dir / "account_balance.json"
            if _bal_file.exists():
                try:
                    with open(_bal_file, encoding="utf-8") as _bf:
                        _bal_data = json.load(_bf)
                    _bal_val = _bal_data.get("balance")
                    if _bal_val is not None:
                        self.bot_config["real_account_balance"] = float(_bal_val)
                except Exception:
                    LOG.debug("[HUD] Failed to load account_balance.json", exc_info=True)

        # Aggregate position files from all running bots (each writes a per-symbol file).
        # Display the first non-FLAT position found; fall back to singleton file if none.
        _pos_files = sorted(
            [str(p) for p in self.data_dir.glob("current_position_*.json")],
            key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0,
            reverse=True,
        )
        self.position = {}
        self._active_pos_file = ""
        for _pf in _pos_files:
            try:
                with open(_pf, encoding="utf-8") as _fh:
                    _pd = json.load(_fh)
                if _pd.get("direction", "FLAT") != "FLAT":
                    self.position = _pd
                    self._active_pos_file = _pf
                    break
            except Exception:
                LOG.debug("[HUD] Failed to load position file %s", _pf, exc_info=True)
        if not self.position:
            self._load_json("current_position.json", "position")  # legacy fallback
        # Identify which bot owns the active (non-FLAT) position for per-bot file loading
        self.active_sym = self.position.get("symbol", "")
        self.active_tf_min = int(self.position.get("timeframe_minutes", 0) or 0)
        # Fallback: parse sym/tf from the filename when bots haven't yet written metadata
        if not self.active_sym and self._active_pos_file:
            _stem = Path(self._active_pos_file).stem  # e.g. "current_position_BTCUSD_M60"
            _tail = _stem.removeprefix("current_position_")
            if "_M" in _tail:
                _sym_part, _, _tf_str = _tail.rpartition("_M")
                self.active_sym = _sym_part
                with contextlib.suppress(ValueError):
                    self.active_tf_min = int(_tf_str)
        if not self.active_sym or not self.active_tf_min:
            _cfg_sym = str(self.bot_config.get("symbol", "") or "").upper()
            try:
                _cfg_tf = int(self.bot_config.get("timeframe_minutes", 0) or 0)
            except (TypeError, ValueError):
                _cfg_tf = 0
            if _cfg_sym and _cfg_tf > 0:
                self.active_sym = _cfg_sym
                self.active_tf_min = _cfg_tf
        self._load_performance_snapshot()
        self._load_health_report()

        self._load_json("training_stats.json", "training_stats")
        self._accumulate_loss_history()

        self._load_json("production_metrics.json", "production_metrics")
        self._load_json("offline_training_status.json", "offline_stats")
        self._enrich_offline_stats_from_champions()
        self._load_universe_stats()
        self._load_all_training_stats()

        # Per-job live progress files (written by OfflineTrainer worker processes)
        _prog: dict = {}
        for _pf in self.data_dir.glob("offline_progress_*.json"):
            try:
                _d = json.loads(_pf.read_text())
                _prog[(_d["symbol"], _d["timeframe_minutes"])] = _d
            except Exception:
                LOG.debug("[HUD] Failed to load offline progress %s", _pf, exc_info=True)
        self.offline_job_progress = _prog

        # trade_log is the authoritative source — always recompute
        self._compute_metrics_from_trade_log()
        self._metrics_from_trade_log = bool(self.lifetime_metrics.get("total_trades"))

        self._load_risk_and_orderbook()

        # Per-bot overrides: replace shared data with the active position's bot-specific files.
        # In multi-bot setups each bot writes risk_metrics_SYM_MTF.json and
        # training_stats_SYM_MTF.json so the HUD always shows the correct bot's data.
        if self.active_sym and self.active_tf_min:
            _sym, _tf = self.active_sym, self.active_tf_min
            _per_train = f"training_stats_{_sym}_M{_tf}.json"
            if (self.data_dir / _per_train).exists():
                self._load_json(_per_train, "training_stats")
                self._accumulate_loss_history()
            _per_risk = f"risk_metrics_{_sym}_M{_tf}.json"
            if (self.data_dir / _per_risk).exists():
                self._load_json(_per_risk, "risk_stats")
                self._apply_risk_stats_to_market_stats()
        else:
            if self.training_stats_all:
                _active_item = max(self.training_stats_all, key=lambda _item: _item.get("mtime", 0))
                self.training_stats = _active_item.get("stats", {})
                self._accumulate_loss_history()
                self.active_sym = _active_item.get("symbol", "")
                self.active_tf_min = int(_active_item.get("timeframe_minutes", 0) or 0)
            _rm_candidates = sorted(
                self.data_dir.glob("risk_metrics_*_M*.json"),
                key=lambda _p: _p.stat().st_mtime if _p.exists() else 0,
                reverse=True,
            )
            if self.active_sym and self.active_tf_min:
                _matched_risk = self.data_dir / f"risk_metrics_{self.active_sym}_M{self.active_tf_min}.json"
                if _matched_risk.exists():
                    self._load_json(_matched_risk.name, "risk_stats")
                    self._apply_risk_stats_to_market_stats()
                elif _rm_candidates:
                    self._load_json(_rm_candidates[0].name, "risk_stats")
                    self._apply_risk_stats_to_market_stats()
            elif _rm_candidates:
                self._load_json(_rm_candidates[0].name, "risk_stats")
                self._apply_risk_stats_to_market_stats()

        # production_metrics.json is written inside each paper_<SYMBOL>_M<TF>
        # directory. Reload it after active_sym/active_tf_min are resolved so
        # Tab 1 system-health counters do not borrow the freshest other bot.
        if self.active_sym and self.active_tf_min:
            self._load_json(_BOT_CONFIG_FILE, "bot_config")
            self._load_json("production_metrics.json", "production_metrics")

        # Self-test results (written at startup by run_self_test())
        st_file = self.data_dir / "self_test.json"
        if st_file.exists():
            try:
                with open(st_file) as f:
                    self.self_test_results = json.load(f).get("results", [])
            except Exception:
                LOG.debug("[HUD] Failed to load self_test.json", exc_info=True)

        # Re-apply order_book.json on top of any per-bot risk-file override.
        # The per-bot risk_metrics_SYM_MTF.json is written at bar-close (every N minutes)
        # while order_book.json is written per-tick, so it always has the freshest
        # spread / bids / asks / vpin / imbalance.  Without this the market-structure
        # tab shows stale bar-close values.
        _ob_final = self._preferred_data_file(_ORDER_BOOK_FILE)
        if _ob_final.exists():
            try:
                with open(_ob_final) as _f:
                    _ob = json.load(_f)
                _ms = self.market_stats
                _ms["spread"] = _ob.get("spread", _ms.get("spread", 0.0))
                _ms["depth_bid"] = _ob.get("depth_bid", _ms.get("depth_bid", 0.0))
                _ms["depth_ask"] = _ob.get("depth_ask", _ms.get("depth_ask", 0.0))
                _ms["vpin"] = _ob.get("vpin", _ms.get("vpin", 0.0))
                _ms["vpin_z"] = _ob.get("vpin_zscore", _ms.get("vpin_z", 0.0))
                _ob_imb = _ob.get("imbalance")
                if _ob_imb is not None:
                    _ms["imbalance"] = float(_ob_imb)
                _ms["order_book_bids"] = _ob.get("order_book_bids", _ms.get("order_book_bids", []))
                _ms["order_book_asks"] = _ob.get("order_book_asks", _ms.get("order_book_asks", []))
                _ms["has_real_sizes"] = _ob.get("has_real_sizes", False)
                _ms["qfi_update_count"] = _ob.get("qfi_update_count", 0)
            except Exception:
                LOG.debug("[HUD] Failed to load order_book.json overlay", exc_info=True)

        # All-bots fleet panel: load every paper_stats_*.json + matching position file
        _all_bots: list[dict] = []
        _all_bots_by_key: dict[tuple[str, int], dict] = {}
        for _psf in sorted(self.data_dir.glob("paper_stats_*.json")):
            try:
                _ps = json.loads(_psf.read_text())
                _sym = str(_ps.get("symbol", "") or "").upper()
                _tf = int(_ps.get("timeframe_minutes", 0) or 0)
                if not _sym or _tf <= 0:
                    continue
                _mode = str(_ps.get("trading_mode", "") or "").strip().lower()
                if _mode not in ("paper", "live"):
                    _mode = "paper" if bool(_ps.get("paper_mode", True)) else "live"
                _ps["trading_mode"] = _mode
                _pos_f = self.data_dir / f"current_position_{_sym}_M{_tf}.json"
                _ps["_position"] = json.loads(_pos_f.read_text()) if _pos_f.exists() else {}
                _key = (_sym, _tf)
                _all_bots_by_key[_key] = _ps
            except Exception:
                LOG.debug("[HUD] Failed to load paper_stats %s", _psf, exc_info=True)
        _all_bots.extend(_all_bots_by_key.values())
        self.all_bots_stats = _all_bots
        # Trade history — cache-loaded (5s) for trades tab
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

    def _compute_metrics_from_trade_log(self) -> None:
        """Compute performance metrics directly from trade_log.jsonl.

        Always re-classifies trades by the rolling time windows (daily/weekly/
        monthly) because those windows advance with wall-clock time even when
        the file itself hasn't changed.  File is only re-parsed when mtime
        changes, avoiding redundant I/O on every 1 Hz refresh cycle.
        """
        trades = self._trade_log_reader.trades
        if not trades:
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
            return

        # Resolve starting_equity: universe.json entries are authoritative (they
        # reflect the real account size); bot_config.json is shared across bots
        # and may carry a stale or default value from whichever bot wrote last.
        _uni_eq = self._universe_starting_equity()
        starting_equity = float(_uni_eq)

        # DATA QUALITY CHECK: Log warnings for data integrity issues
        _null_entry_time = sum(1 for t in trades if t.get("entry_time") is None)
        _missing_quantity = sum(1 for t in trades if "quantity" not in t or t.get("quantity") is None)
        _recalc_trades = sum(1 for t in trades if t.get("pnl_recalculated"))

        if _null_entry_time > 0:
            LOG.debug(
                "[DATA-QUALITY] %d/%d trades have NULL entry_time (will be excluded from duration calc)",
                _null_entry_time,
                len(trades),
            )
        if _missing_quantity > 0:
            LOG.debug(
                "[DATA-QUALITY] %d/%d trades missing 'quantity' field (HUD cannot display position sizing)",
                _missing_quantity,
                len(trades),
            )
        if _recalc_trades > 0:
            _original_pnl = sum(t.get("pnl_original", 0) for t in trades if "pnl_original" in t)
            _current_pnl = sum(t.get("pnl", 0) for t in trades)
            _variance = abs(_current_pnl - _original_pnl)
            LOG.debug(
                "[DATA-QUALITY] %d/%d trades recalculated. Original PnL: $%.2f, Current: $%.2f, Variance: $%.2f",
                _recalc_trades,
                len(trades),
                _original_pnl,
                _current_pnl,
                _variance,
            )

        # Determine active trading mode; infer missing legacy labels.
        _modes: set[str] = set()
        _unlabeled = 0
        _inferred = 0
        for _t in trades:
            _raw_mode = str(_t.get("trading_mode", "") or "").strip().lower()
            if _raw_mode in ("paper", "live"):
                _resolved = _raw_mode
            else:
                _unlabeled += 1
                _resolved = self._resolve_trade_mode_from_record(_t)
                _inferred += 1
                _t["trading_mode"] = _resolved
            _modes.add(_resolved)
        self._trade_log_unlabeled_count = _unlabeled
        self._trade_log_inferred_count = _inferred
        self._trade_log_unknown_timeframe_count = sum(1 for _t in trades if self._normalize_timeframe_label(_t) == "M?")
        if len(_modes) == 1:
            self._trade_log_mode = next(iter(_modes))
        elif _modes:
            self._trade_log_mode = "mixed"
        else:
            self._trade_log_mode = ""

        self.all_time_metrics = _hud_period_metrics(trades, starting_equity)
        self.all_time_metrics_by_mode = {
            _mode_name: _hud_period_metrics([t for t in trades if t.get("trading_mode") == _mode_name], starting_equity)
            for _mode_name in ("paper", "live")
        }
        self._trade_log_all_trades = list(trades)
        self._trade_log_all_trades_by_mode = {
            _mode_name: [t for t in trades if t.get("trading_mode") == _mode_name] for _mode_name in ("paper", "live")
        }

        # Apply stats epoch filter — exclude old trades from all metrics
        trades = self._filter_trades_by_epoch(trades)
        self._trade_log_metrics_trades = list(trades)
        self._build_metrics_cube(trades)

        trades_by_mode = {
            "paper": [t for t in trades if t.get("trading_mode") == "paper"],
            "live": [t for t in trades if t.get("trading_mode") == "live"],
        }
        self._trade_log_metrics_trades_by_mode = {k: list(v) for k, v in trades_by_mode.items()}

        daily, weekly, monthly = _classify_trades_by_period(trades)

        # For period MaxDD to be meaningful it must be anchored to the account
        # equity at the START of each period, not at bot launch.  Trades
        # classified into a period are a subset of the global trade list; the
        # equity at period-start equals launch_equity + PnL of all trades that
        # completed BEFORE that period window.
        # id() is used to match the exact dict objects returned by
        # _classify_trades_by_period (same objects as in `trades`).
        _daily_ids = set(map(id, daily))
        _weekly_ids = set(map(id, weekly))
        _monthly_ids = set(map(id, monthly))
        _pre_daily_equity = starting_equity + sum(t.get("pnl", 0) for t in trades if id(t) not in _daily_ids)
        _pre_weekly_equity = starting_equity + sum(t.get("pnl", 0) for t in trades if id(t) not in _weekly_ids)
        _pre_monthly_equity = starting_equity + sum(t.get("pnl", 0) for t in trades if id(t) not in _monthly_ids)

        self.daily_metrics = _hud_period_metrics(daily, _pre_daily_equity)
        self.weekly_metrics = _hud_period_metrics(weekly, _pre_weekly_equity)
        self.monthly_metrics = _hud_period_metrics(monthly, _pre_monthly_equity)
        self.lifetime_metrics = _hud_period_metrics(trades, starting_equity)

        self.daily_metrics_by_mode = {}
        self.weekly_metrics_by_mode = {}
        self.monthly_metrics_by_mode = {}
        self.lifetime_metrics_by_mode = {}
        for _mode_name, _mode_trades in trades_by_mode.items():
            _d_m, _w_m, _m_m = _classify_trades_by_period(_mode_trades)
            _d_ids = set(map(id, _d_m))
            _w_ids = set(map(id, _w_m))
            _m_ids = set(map(id, _m_m))
            _pre_d = starting_equity + sum(t.get("pnl", 0) for t in _mode_trades if id(t) not in _d_ids)
            _pre_w = starting_equity + sum(t.get("pnl", 0) for t in _mode_trades if id(t) not in _w_ids)
            _pre_m = starting_equity + sum(t.get("pnl", 0) for t in _mode_trades if id(t) not in _m_ids)
            self.daily_metrics_by_mode[_mode_name] = _hud_period_metrics(_d_m, _pre_d)
            self.weekly_metrics_by_mode[_mode_name] = _hud_period_metrics(_w_m, _pre_w)
            self.monthly_metrics_by_mode[_mode_name] = _hud_period_metrics(_m_m, _pre_m)
            self.lifetime_metrics_by_mode[_mode_name] = _hud_period_metrics(_mode_trades, starting_equity)

        # Augment lifetime_metrics with timing data derived from trade timestamps.
        # These are more accurate than the runtime-counter values in production_metrics.json
        # which reset on each bot session and only reflect the current session.
        _durations: list[float] = []
        _trades_with_complete_times = 0
        _last_exit_dt = None
        for _t in trades:
            _entry_dt = _hud_parse_dt(_t.get("entry_time", ""))
            _exit_dt = _hud_parse_dt(_t.get("exit_time", ""))
            if _entry_dt and _exit_dt:
                _durations.append((_exit_dt - _entry_dt).total_seconds() / 60.0)
                _trades_with_complete_times += 1
            if _exit_dt and (_last_exit_dt is None or _exit_dt > _last_exit_dt):
                _last_exit_dt = _exit_dt
        _now = datetime.now(UTC)
        self.lifetime_metrics["avg_trade_duration_mins"] = sum(_durations) / len(_durations) if _durations else 0.0
        self.lifetime_metrics["last_trade_mins_ago"] = (
            (_now - _last_exit_dt).total_seconds() / 60.0 if _last_exit_dt else 0.0
        )
        # Track data quality for metrics
        self.lifetime_metrics["_data_quality_trades_with_complete_times"] = _trades_with_complete_times
        self.lifetime_metrics["_data_quality_total_trades"] = len(trades)

        # Per-symbol breakdown
        _by_sym: dict[str, list] = {}
        for _t in trades:
            _s = _t.get("symbol", "UNKNOWN")
            _by_sym.setdefault(_s, []).append(_t)
        self.per_symbol_metrics: dict[str, dict] = {}
        for _s, _st in _by_sym.items():
            self.per_symbol_metrics[_s] = _hud_period_metrics(_st, starting_equity)

        _by_symbol_tf: dict[tuple[str, str], list[dict]] = {}
        for (_sym, _tf, _mode), _trades in self.metrics_cube.items():
            _by_symbol_tf.setdefault((_sym, _tf), []).extend(_trades)
        self.metrics_by_symbol_tf = {_k: _hud_period_metrics(_v, starting_equity) for _k, _v in _by_symbol_tf.items()}

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

    def _show_help(self) -> None:
        """Display help screen with keyboard shortcuts and information."""
        self._disable_raw_mode()
        try:
            os.system("clear" if os.name != "nt" else "cls")
            print("╔" + "═" * 78 + "╗")
            print("║" + " " * 25 + "HUD HELP & REFERENCE" + " " * 32 + "║")
            print("╚" + "═" * 78 + "╝\n")

            print("\033[1m📋 KEYBOARD SHORTCUTS\033[0m\n")
            print("  [1]           - Overview tab (compact summary)")
            print("  [2]           - Performance tab (detailed metrics)")
            print("  [3]           - Training tab (agent statistics)")
            print("  [4]           - Risk tab (risk management)")
            print("  [5]           - Market tab (microstructure)")
            print("  [6]           - Decision Log tab (last 20 decisions)")
            print("  [7]           - Trade History tab (all closed trades with drill-down)")
            print("  [←] / [→]     - Cycle tabs without reaching for Tab")
            print("  [Tab]         - Cycle to next tab")
            print("  [Shift+Tab]   - Cycle to previous tab")
            print("  [s]           - Select symbol/timeframe preset")
            print("  [h]           - Show this help screen")
            print("  [q] / Ctrl+Q / Ctrl+X  - Quit HUD")
            print("  [Alt+K]       - Emergency kill switch (close all positions + halt trading)")
            print("  [r]           - Review tripped circuit breakers and reset if OK")
            print("  [e]           - Set/clear stats epoch (exclude old trades from metrics)")

            print("\n\033[1m📋 TRADE HISTORY TAB KEYS\033[0m\n")
            print("  [↓]/[↑], [j]/[k] - Move selection down / up")
            print("  [n] / [p]     - Next / previous page")
            print("  [d]           - Drill into selected trade (full detail view)")
            print("  [b] / [d]     - Back from detail view to trade list")

            print("\n\033[1m📊 TAB DESCRIPTIONS\033[0m\n")
            print("  Overview      - Quick snapshot of position, daily stats, risk, and health")
            print("  Performance   - Trade-log metrics for 24h/7d/month, epoch, lifetime, and current sessions")
            print("  Training      - Agent training status, buffer sizes, loss metrics")
            print("  Risk          - Circuit breaker (with trip reasons + reset), VaR, vol, regime")
            print("  Market        - Spread, VPIN toxicity, order imbalance, depth")
            print("  Decision Log  - Last 20 trading decisions with color-coded events")
            print("  Trades        - Full trade history, paginated, with per-trade drill-down")

            print("\n\033[1m🎨 COLOR CODING\033[0m\n")
            print(f"  {_ANSI_G}✓ Green{_ANSI_RST}       - Positive values, good status, active longs")
            print(f"  {_ANSI_R}✗ Red{_ANSI_RST}         - Negative values, alerts, active shorts")
            print(f"  {_ANSI_Y}⚡ Yellow{_ANSI_RST}      - Neutral/warning, hold actions")
            print(f"  {_ANSI_B}ℹ Blue{_ANSI_RST}        - Informational messages")

            print("\n\033[1m📁 DATA SOURCES\033[0m\n")
            print("  All data is read from JSON/JSONL files in the 'data/' directory:")
            print(f"    • {_BOT_CONFIG_FILE:<27} - Bot configuration and status")
            print("    • current_position_SYM_MTF.json - Active position (per symbol/timeframe)")
            print("    • trade_log.jsonl           - All closed trades (primary source for performance)")
            print("    • training_stats.json        - Agent training statistics")
            print("    • training_stats_SYM_MTF.json- Per-bot training stats (overrides shared file)")
            print("    • risk_metrics.json          - Risk metrics (drawdown, VaR, circuit breakers)")
            print("    • order_book.json            - Live market data (spread, depth, VPIN, imbalance)")
            print("    • performance_snapshot.json  - Trading mode identifier only")
            print("    • logs/audit/decisions.jsonl - Decision history (primary, rich JSONL format)")
            print("    • decision_log.json          - Decision history (legacy fallback only)")

            print("\n\033[1m⚙️  SYSTEM REQUIREMENTS\033[0m\n")
            print("  • Terminal with UTF-8 support")
            print("  • ANSI color support")
            print("  • Minimum 80x24 terminal size recommended")
            print("  • Bot must be running and exporting data files")

            print("\n\033[1m🔧 TROUBLESHOOTING\033[0m\n")
            print("  Data stale warning    - Bot may be paused or crashed")
            print("  Missing files         - Check that bot is running and exporting")
            print("  Garbled display       - Ensure terminal supports UTF-8 and ANSI colors")
            print("  Keyboard not working  - Try running in a different terminal emulator")

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
            os.system("clear" if os.name != "nt" else "cls")
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

    def _handle_cb_reset(self) -> None:
        """R key: Show tripped circuit breakers with reasons and offer reset."""
        self._disable_raw_mode()
        try:
            os.system("clear" if os.name != "nt" else "cls")
            YLW = _ANSI_Y
            GRN = _ANSI_G
            RED = _ANSI_R
            DIM = _ANSI_DIM
            RST = _ANSI_RST

            print(YLW + "╔" + "═" * 60 + "╗")
            print("║" + " " * 14 + "🔌 CIRCUIT BREAKER REVIEW" + " " * 19 + "║")
            print("╚" + "═" * 60 + "╝" + RST + "\n")

            # Load circuit_breakers.json
            _cb_path = self.data_dir / "circuit_breakers.json"
            _cb_data: dict = {}
            if _cb_path.exists():
                try:
                    with open(_cb_path, encoding="utf-8") as _f:
                        _cb_data = json.load(_f)
                except Exception:
                    pass

            _breaker_labels = {
                "sortino": ("Sortino Ratio", "Risk-adjusted returns too low"),
                "kurtosis": ("Kurtosis", "Return distribution has fat tails"),
                "drawdown": ("Drawdown", "Equity drawdown exceeded limit"),
                "consecutive_losses": ("Consecutive Losses", "Too many losses in a row"),
            }

            _any_tripped = False
            _kurt_gate_active = bool(self.risk_stats.get("kurtosis_gate_active", False))
            _kurtosis_now = float(self.risk_stats.get("kurtosis", 0.0) or 0.0)
            _kurtosis_threshold = float(
                self.risk_stats.get("kurtosis_threshold", KURTOSIS_FAT_TAIL_THRESHOLD) or KURTOSIS_FAT_TAIL_THRESHOLD
            )
            _risk_scope = self._risk_scope_label(self.risk_stats)
            for _key, (_label, _explain) in _breaker_labels.items():
                _b = _cb_data.get(_key, {})
                if not isinstance(_b, dict):
                    _b = {}
                _tripped = bool(_b.get("is_tripped", False))
                _gate_only = False
                if _key == "kurtosis" and _kurt_gate_active and not _tripped:
                    _tripped = True
                    _gate_only = True
                if _tripped:
                    _any_tripped = True
                    _reason = _b.get("trip_reason", _explain)
                    _tv = _b.get("trip_value", 0.0)
                    _th = _b.get("threshold", 0.0)
                    _trip_ts = _b.get("trip_time", "")
                    _cd_mins = _b.get("cooldown_minutes", 60)
                    if _gate_only:
                        _reason = f"Kurtosis gate active [{_risk_scope}] (entry gate)"
                        _tv = _kurtosis_now
                        _th = _kurtosis_threshold
                    print(f"  {RED}✗ {_label}: TRIPPED{RST}")
                    print(f"    Reason:    {YLW}{_reason}{RST}")
                    print(f"    Value:     {_tv:.4f}  (threshold: {_th:.4f})")
                    if _trip_ts:
                        print(f"    Tripped:   {_trip_ts[:19]}")
                        try:
                            _trip_dt = datetime.fromisoformat(_trip_ts)
                            _elapsed = (datetime.now(UTC) - _trip_dt).total_seconds() / 60.0
                            _remaining = max(0, _cd_mins - _elapsed)
                            if _remaining > 0:
                                print(f"    Cooldown:  {_remaining:.0f}m remaining (auto-reset after {_cd_mins}m)")
                            else:
                                print(f"    Cooldown:  {GRN}Elapsed — safe to reset{RST}")
                        except (ValueError, TypeError):
                            pass
                    print()
                else:
                    print(f"  {GRN}✓ {_label}: OK{RST}")

            if not _any_tripped:
                print(f"\n  {GRN}All circuit breakers are OK — nothing to reset.{RST}")
                input("\nPress Enter to return to HUD...")
                return

            print(f"\n{DIM}Resetting will allow the bot to resume trading immediately.{RST}")
            print(f"{DIM}Only reset if you understand why the breaker tripped and the condition is resolved.{RST}\n")
            print(YLW + "Type 'reset' and press Enter to reset all tripped breakers, or press Enter to abort:" + RST)
            confirm = input("> ").strip()
            if confirm.upper() == "RESET":
                _reset_data = {
                    "reset": True,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                self._broadcast_control_file(
                    "circuit_breaker_reset.json",
                    _reset_data,
                    prefix=".cb_reset_",
                )
                print(f"\n{GRN}✓ Reset request sent — bot will reset breakers within ~5 seconds{RST}")
                self._set_notification("🔄 Circuit breaker reset requested", ttl=30)
                input("\nPress Enter to return to HUD...")
            else:
                print(f"\n{YLW}Aborted — no action taken.{RST}")
                time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            with contextlib.suppress(Exception):
                input("Press Enter to continue...")
        finally:
            self._enable_raw_mode()

    def _handle_stats_epoch(self) -> None:
        """[e] key: Set or clear the stats epoch to exclude old trades from metrics."""
        self._disable_raw_mode()
        try:
            os.system("clear" if os.name != "nt" else "cls")
            YLW = _ANSI_Y
            GRN = _ANSI_G
            DIM = _ANSI_DIM
            RST = _ANSI_RST

            print(YLW + "╔" + "═" * 60 + "╗")
            print("║" + " " * 14 + "📅 STATS EPOCH MANAGER" + " " * 22 + "║")
            print("╚" + "═" * 60 + "╝" + RST + "\n")

            if self._stats_epoch:
                _epoch_str = self._stats_epoch.strftime("%Y-%m-%d %H:%M UTC")
                print(f"  Current epoch: {GRN}{_epoch_str}{RST}")
                print(
                    f"  Excluded:      {self._stats_epoch_excluded} trades, "
                    f"${self._stats_epoch_excluded_pnl:+.2f} PnL\n"
                )
            else:
                print(f"  Current epoch: {DIM}None (all trades included){RST}\n")

            print("  Options:")
            print(f"    {YLW}1{RST}  Set epoch to NOW (fresh start from this moment)")
            print(f"    {YLW}2{RST}  Set epoch to start of today")
            print(f"    {YLW}3{RST}  Set epoch to 7 days ago")
            print(f"    {YLW}4{RST}  Set epoch to 30 days ago")
            print(f"    {YLW}5{RST}  Enter a custom date (YYYY-MM-DD)")
            print(f"    {YLW}c{RST}  Clear epoch (show all trades)")
            print(f"    {DIM}Enter{RST}  Cancel\n")
            choice = input("Selection: ").strip().lower()

            _now = datetime.now(UTC)
            _new_epoch: datetime | None = None
            if choice == "1":
                _new_epoch = _now
            elif choice == "2":
                _new_epoch = _now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif choice == "3":
                _new_epoch = _now - timedelta(days=7)
            elif choice == "4":
                _new_epoch = _now - timedelta(days=30)
            elif choice == "5":
                _date_str = input("Enter date (YYYY-MM-DD): ").strip()
                _parsed = _hud_parse_dt(_date_str + "T00:00:00+00:00")
                if _parsed:
                    _new_epoch = _parsed
                else:
                    print(f"\n  {_ANSI_R}Invalid date format.{RST}")
                    input("Press Enter to return to HUD...")
                    return
            elif choice == "c":
                self._save_stats_epoch(None)
                self._set_notification("Stats epoch cleared — all trades included", ttl=6)
                # Force metrics recompute
                self._trade_log_reader.invalidate()
                return
            else:
                return

            self._save_stats_epoch(_new_epoch)
            _label = _new_epoch.strftime("%Y-%m-%d %H:%M UTC") if _new_epoch else "cleared"
            self._set_notification(f"Stats epoch set to {_label}", ttl=6)
            # Force metrics recompute on next refresh
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
            os.system("clear" if os.name != "nt" else "cls")
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
            if self.current_tab == "overview":
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
        frame_key = re.sub(r"^[^\n]*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$", "<HEARTBEAT>", frame_key, flags=re.MULTILINE)
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

    def _render_offline_training(self, ofs: dict) -> None:
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
        if _results:
            self._render_offline_jobs_table(_results)
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

    def _render_pipeline_card(self, sym: str, entry: dict) -> None:
        """Render one bot card with connection + training + activity stats."""
        stage = entry.get("stage", "?")
        tf_min = entry.get("timeframe_minutes", 0)
        tf_lbl = f"M{tf_min}" if tf_min else "?"
        zo = entry.get("z_omega")
        pid = entry.get("paper_pid")
        alive = entry.get("_pid_alive", False)
        ps = entry.get("_bot_stats", {})  # per-bot stats JSON from bot

        # ── title line ────────────────────────────────────────────────────────
        stage_col = {
            "PAPER": _ANSI_Y,
            "LIVE": _ANSI_G,
            "UNTRAINED": _ANSI_DIM,
            "DEMOTED": _ANSI_R,
        }.get(stage, _ANSI_DIM)
        _no_weights = not entry.get("weights_path")
        if zo is not None and not (zo == 0.0 and _no_weights):
            zo_c = _ANSI_G if zo > 1.0 else (_ANSI_Y if zo > 0 else _ANSI_R)
            zo_str = f"{zo_c}ZΩ {zo:.4f}{_ANSI_RST}"
        else:
            zo_str = f"{_ANSI_DIM}ZΩ —{_ANSI_RST}"
        pid_str = (
            f"{_ANSI_G}▶ PID {pid}{_ANSI_RST}"
            if alive
            else (f"{_ANSI_R}✗ dead ({pid}){_ANSI_RST}" if pid else f"{_ANSI_DIM}not started{_ANSI_RST}")
        )
        uptime_s = int(ps.get("uptime_seconds", 0))
        if uptime_s >= 3600:
            uptime_str = f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m"
        elif uptime_s:
            uptime_str = f"{uptime_s // 60}m {uptime_s % 60}s"
        else:
            uptime_str = "—"
        print(
            f"  {_ANSI_B}◼ {sym} {tf_lbl}{_ANSI_RST}  "
            f"{stage_col}{stage}{_ANSI_RST}  {zo_str}  {pid_str}  uptime {uptime_str}"
        )

        if ps:
            # ── FIX connection ────────────────────────────────────────────────
            q_ok = ps.get("quote_ok", False)
            t_ok = ps.get("trade_ok", False)
            healthy = ps.get("connection_healthy", False)
            recon = ps.get("total_reconnects", 0)
            q_str = f"{_ANSI_G}QUOTE ✓{_ANSI_RST}" if q_ok else f"{_ANSI_R}QUOTE ✗{_ANSI_RST}"
            t_str = f"{_ANSI_G}TRADE ✓{_ANSI_RST}" if t_ok else f"{_ANSI_R}TRADE ✗{_ANSI_RST}"
            h_str = f"{_ANSI_G}healthy{_ANSI_RST}" if healthy else f"{_ANSI_Y}unhealthy{_ANSI_RST}"
            r_col = _ANSI_G if recon == 0 else (_ANSI_Y if recon < 5 else _ANSI_R)
            print(f"    FIX: {q_str}  {t_str}  {h_str}  │  reconnects: {r_col}{recon}{_ANSI_RST}")

            # ── activity ─────────────────────────────────────────────────────
            bars = ps.get("bar_count", 0)
            trades = ps.get("total_trades", 0)
            pnl = ps.get("total_pnl", 0.0)
            wr = ps.get("win_rate", 0.0)
            pnl_c = _ANSI_G if pnl >= 0 else _ANSI_R
            wr_str = f"{wr * 100:.1f}%" if trades > 0 else "—"
            print(f"    Bars: {bars}  │  Trades: {trades}  │  PnL: {pnl_c}{pnl:+.2f}{_ANSI_RST}  │  Win: {wr_str}")

            # ── account balance (real from broker if CollateralReport arrived) ─
            _rb = ps.get("real_account_balance")
            _re = ps.get("real_account_equity")
            _rm = ps.get("real_margin_free")
            if _rb is not None:
                _rb_pnl = pnl  # compare relative to starting point
                _rb_c = _ANSI_G if _rb_pnl >= 0 else _ANSI_R
                _re_str = f"  │  Equity: {_ANSI_B}{float(_re):,.2f}{_ANSI_RST}" if _re is not None else ""
                _rm_str = f"  │  Free margin: {_ANSI_B}{float(_rm):,.2f}{_ANSI_RST}" if _rm is not None else ""
                print(f"    Balance: {_rb_c}{float(_rb):,.2f}{_ANSI_RST}  {_ANSI_G}✓ live{_ANSI_RST}{_re_str}{_rm_str}")

            # ── training stats ────────────────────────────────────────────────
            t_steps = ps.get("trigger_steps", 0)
            t_eps = ps.get("trigger_epsilon", 0.0)
            t_buf = ps.get("trigger_buffer", 0)
            t_loss = ps.get("trigger_loss", 0.0)
            t_ready = ps.get("trigger_ready", False)
            h_steps = ps.get("harvester_steps", 0)
            h_beta = ps.get("harvester_beta", 0.4)
            h_buf = ps.get("harvester_buffer", 0)
            h_loss = ps.get("harvester_loss", 0.0)
            h_ready = ps.get("harvester_ready", False)

            t_bar = self._pp_bar(t_buf / _RT_TRIG_CAP if _RT_TRIG_CAP else 0)
            h_bar = self._pp_bar(h_buf / _RT_HARV_CAP if _RT_HARV_CAP else 0)
            t_pct = f"{100 * t_buf / _RT_TRIG_CAP:4.0f}%" if _RT_TRIG_CAP else ""
            h_pct = f"{100 * h_buf / _RT_HARV_CAP:4.0f}%" if _RT_HARV_CAP else ""
            t_rd = f"{_ANSI_G}ready{_ANSI_RST}" if t_ready else f"{_ANSI_Y}filling{_ANSI_RST}"
            h_rd = f"{_ANSI_G}ready{_ANSI_RST}" if h_ready else f"{_ANSI_Y}filling{_ANSI_RST}"
            t_ls = f"{t_loss:.4f}" if t_loss > 0 else f"{_ANSI_DIM}—{_ANSI_RST}"
            h_ls = f"{h_loss:.4f}" if h_loss > 0 else f"{_ANSI_DIM}—{_ANSI_RST}"
            print(f"    Trig:  {t_steps:>6,} steps  ε={t_eps:.3f}  buf {t_bar}{t_pct}  loss {t_ls}  {t_rd}")
            print(f"    Harv:  {h_steps:>6,} steps  β={h_beta:.3f}  buf {h_bar}{h_pct}  loss {h_ls}  {h_rd}")
        else:
            started = entry.get("paper_started_at", "")
            started_str = started[:19].replace("T", " ") if started else "—"
            print(f"    {_ANSI_DIM}Stats not yet available  (started {started_str}){_ANSI_RST}")
        print()

    def _render_live_trigger_agent(self, ts: dict, pm: dict) -> None:
        """Render the Trigger Agent training block."""

        def _to_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        trig_buf = ts.get("trigger_buffer_size", 0)
        trig_added = ts.get("trigger_total_added", 0)
        trig_loss = ts.get("trigger_loss", 0.0)
        trig_eps = ts.get("trigger_epsilon", 0.0)
        is_in_pos = ts.get("is_in_position")  # None = unknown (old data)
        # Prefer training_stats confidence (single source of truth); fall back to production_metrics
        trig_ready = ts.get("trigger_ready", False)
        trig_steps = ts.get("trigger_training_steps", 0)
        trig_conf = ts.get("trigger_confidence", pm.get("trigger_confidence_avg", 0.5))
        ready_t = f"{_ANSI_G}✓ Ready{_ANSI_RST}" if trig_ready else f"{_ANSI_Y}⏳ Filling…{_ANSI_RST}"
        # Buffer fill-status annotation: trigger only fills when bot is FLAT
        if is_in_pos is True:
            _trig_buf_note = f"  {_ANSI_Y}⏸ paused — bot in position{_ANSI_RST}"
        elif is_in_pos is False:
            _trig_buf_note = f"  {_ANSI_G}⬆ filling — bot flat{_ANSI_RST}"
        else:
            _trig_buf_note = ""
        print(f"  \033[1m🎯 TRIGGER AGENT  (Entry)\033[0m  {ready_t}  {_ANSI_DIM}fills when flat{_ANSI_RST}")
        print(f"    Steps:  {trig_steps:>10,}   Velocity: {self._rt_velocity(self._trig_step_hist)}")
        print(f"    Buffer: {self._rt_pct_bar(trig_buf, _RT_TRIG_CAP)}  {trig_buf:,}/{_RT_TRIG_CAP:,}{_trig_buf_note}")
        if trig_added > 0:
            print(f"    Added:  {trig_added:,} total experiences")
        print(f"    ε:      {self._rt_eps_bar(trig_eps)}")
        # Regime-aware epsilon factor: 1.0 = normal decay, <1.0 = slower (exploring more)
        _regime_f = ts.get("trigger_epsilon_regime_factor", 1.0)
        _rf_col = _ANSI_G if _regime_f >= 0.9 else (_ANSI_Y if _regime_f >= 0.7 else _ANSI_B)
        print(
            f"    ε ζ:    {_rf_col}{_regime_f:.2f}{_ANSI_RST}  {_ANSI_DIM}(decay factor — 1.0 normal, <1 slower){_ANSI_RST}"
        )
        # Adaptive tau (target network update rate)
        _trig_tau = ts.get("trigger_tau", 0.005)
        _tau_col = _ANSI_G if _trig_tau > 0.003 else (_ANSI_Y if _trig_tau > 0.001 else _ANSI_R)
        print(f"    τ:      {_tau_col}{_trig_tau:.5f}{_ANSI_RST}  {_ANSI_DIM}(adaptive target sync){_ANSI_RST}")
        _tl_str = f"{trig_loss:.6f}" if trig_loss > 0 else f"{_ANSI_DIM}0.000000 (idle/no training event){_ANSI_RST}"
        print(f"    Loss:   {_tl_str}")
        print(f"    Trend:  {self._rt_trend(self._trig_loss_hist)}")
        _sp = self._rt_spark(self._trig_loss_hist)
        if _sp:
            print(f"    Hist:   {_sp}")
        if CONF_HEALTHY_LOW < trig_conf < CONF_HEALTHY_HIGH:
            _cc = _ANSI_G
        elif CONF_WARM_LOW < trig_conf <= CONF_HEALTHY_LOW:
            _cc = _ANSI_Y
        else:
            _cc = _ANSI_R
        print(f"    Conf:   {_cc}{trig_conf:.3f}{_ANSI_RST}  {_ANSI_DIM}(healthy 0.55–0.85){_ANSI_RST}")
        _entry_floor = _to_float(ts.get("entry_conf_dynamic_floor"))
        if _entry_floor is None:
            print(f"    RL min: {_ANSI_DIM}—{_ANSI_RST}  {_ANSI_DIM}(risk tuner pending){_ANSI_RST}")
        else:
            _floor_col = _ANSI_G if _entry_floor <= 0.75 else (_ANSI_Y if _entry_floor <= 0.85 else _ANSI_R)
            _gap = float(trig_conf) - _entry_floor
            _gap_col = _ANSI_G if _gap >= 0 else _ANSI_R
            print(
                f"    RL min: {_floor_col}{_entry_floor:.3f}{_ANSI_RST}  "
                f"{_ANSI_DIM}(dynamic entry floor){_ANSI_RST}  Δnow {_gap_col}{_gap:+.3f}{_ANSI_RST}"
            )
        _rw_total = int(ts.get("trigger_runway_cal_total_samples", 0) or 0)
        _rw_active = int(ts.get("trigger_runway_cal_active_buckets", 0) or 0)
        _rw_reliable = bool(ts.get("trigger_runway_predictor_reliable", False))
        _rw_col = _ANSI_G if _rw_reliable else _ANSI_Y
        _rw_lbl = "RELIABLE (gate active)" if _rw_reliable else "LEARNING (gate bypass)"
        print(
            f"    Runway: {_rw_col}{_rw_lbl}{_ANSI_RST}  {_ANSI_DIM}samples={_rw_total} active_buckets={_rw_active}{_ANSI_RST}"
        )
        print()

    def _render_live_harvester_agent(self, ts: dict, pm: dict) -> None:
        """Render the Harvester Agent training block."""

        def _to_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        harv_buf = ts.get("harvester_buffer_size", 0)
        harv_added = ts.get("harvester_total_added", 0)
        harv_loss = ts.get("harvester_loss", 0.0)
        harv_beta = ts.get("harvester_beta", 0.4)
        harv_min_hold = ts.get("harvester_min_hold_ticks", 10)
        is_in_pos = ts.get("is_in_position")
        harv_ready = ts.get("harvester_ready", False)
        harv_steps = ts.get("harvester_training_steps", 0)
        harv_conf = ts.get("harvester_confidence", pm.get("harvester_confidence_avg", 0.5))
        ready_h = f"{_ANSI_G}✓ Ready{_ANSI_RST}" if harv_ready else f"{_ANSI_Y}⏳ Filling…{_ANSI_RST}"
        # Buffer fill-status annotation: harvester only fills when bot is IN POSITION
        if is_in_pos is True:
            _harv_buf_note = f"  {_ANSI_G}⬆ filling — in position{_ANSI_RST}"
        elif is_in_pos is False:
            _harv_buf_note = f"  {_ANSI_Y}⏸ paused — bot flat{_ANSI_RST}"
        else:
            _harv_buf_note = ""
        print(f"  \033[1m🌾 HARVESTER AGENT  (Exit)\033[0m  {ready_h}  {_ANSI_DIM}fills in position{_ANSI_RST}")
        print(f"    Steps:  {harv_steps:>10,}   Velocity: {self._rt_velocity(self._harv_step_hist)}")
        print(f"    Buffer: {self._rt_pct_bar(harv_buf, _RT_HARV_CAP)}  {harv_buf:,}/{_RT_HARV_CAP:,}{_harv_buf_note}")
        if harv_added > 0:
            print(f"    Added:  {harv_added:,} total experiences")
        print(f"    β IS:   {self._rt_beta_bar(harv_beta)}")
        # Adaptive tau (target network update rate)
        _harv_tau = ts.get("harvester_tau", 0.005)
        _htau_col = _ANSI_G if _harv_tau > 0.003 else (_ANSI_Y if _harv_tau > 0.001 else _ANSI_R)
        print(f"    τ:      {_htau_col}{_harv_tau:.5f}{_ANSI_RST}  {_ANSI_DIM}(adaptive target sync){_ANSI_RST}")
        _hl_str = f"{harv_loss:.6f}" if harv_loss > 0 else f"{_ANSI_DIM}0.000000 (idle/no training event){_ANSI_RST}"
        print(f"    Loss:   {_hl_str}")
        print(f"    Trend:  {self._rt_trend(self._harv_loss_hist)}")
        _sp = self._rt_spark(self._harv_loss_hist)
        if _sp:
            print(f"    Hist:   {_sp}")
        if CONF_HEALTHY_LOW < harv_conf < CONF_HEALTHY_HIGH:
            _cc = _ANSI_G
        elif CONF_WARM_LOW < harv_conf <= CONF_HEALTHY_LOW:
            _cc = _ANSI_Y
        else:
            _cc = _ANSI_R
        print(f"    Conf:   {_cc}{harv_conf:.3f}{_ANSI_RST}  {_ANSI_DIM}(healthy 0.55–0.85){_ANSI_RST}")
        _exit_floor = _to_float(ts.get("exit_conf_dynamic_floor"))
        if _exit_floor is None:
            print(f"    RL min: {_ANSI_DIM}—{_ANSI_RST}  {_ANSI_DIM}(risk tuner pending){_ANSI_RST}")
        else:
            _floor_col = _ANSI_G if _exit_floor <= 0.65 else (_ANSI_Y if _exit_floor <= 0.80 else _ANSI_R)
            _gap = float(harv_conf) - _exit_floor
            _gap_col = _ANSI_G if _gap >= 0 else _ANSI_R
            print(
                f"    RL min: {_floor_col}{_exit_floor:.3f}{_ANSI_RST}  "
                f"{_ANSI_DIM}(dynamic exit floor){_ANSI_RST}  Δnow {_gap_col}{_gap:+.3f}{_ANSI_RST}"
            )
        # Regime-aware hold duration
        _hold_mult = ts.get("harvester_regime_hold_mult", 1.0)
        _hm_col = _ANSI_G if _hold_mult > 1.0 else (_ANSI_Y if _hold_mult >= 0.9 else _ANSI_R)
        print(
            f"    Hold:   {harv_min_hold} ticks min  {_hm_col}×{_hold_mult:.2f}{_ANSI_RST}  {_ANSI_DIM}(regime mult — >1 trend run, <1 quick exit){_ANSI_RST}"
        )
        _cd = float(ts.get("harvester_capture_decay_threshold", 0.0) or 0.0)
        _mw = float(ts.get("harvester_micro_winner_giveback_pct", 0.0) or 0.0)
        print(f"    WTL:    {_ANSI_Y}capture_decay<{_cd:.2f}  micro_giveback>{_mw:.2f}×MFE{_ANSI_RST}")
        print()

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

    def _render_training(self) -> None:
        """Render agent training status."""
        ts = self.training_stats
        print(f"  {_ANSI_DIM}(canonical source: training_stats_*.json; per-bot file preferred){_ANSI_RST}\n")
        pm = self.production_metrics.get("metrics", {})
        ofs = self.offline_stats
        if ofs:
            _ofs_status = self._offline_status_normalized(ofs)
            # Auto-prune completed offline training older than 24 h
            _ofs_stale = False
            if _ofs_status == "complete" and ofs.get("completed_at"):
                try:
                    _comp = datetime.fromisoformat(ofs["completed_at"])
                    if _comp.tzinfo is None:
                        _comp = _comp.replace(tzinfo=UTC)
                    _ofs_stale = (datetime.now(UTC) - _comp).total_seconds() > 86400
                except Exception:
                    pass
            if _ofs_stale:
                # Silently discard stale offline training display + remove file
                self.offline_stats = {}
                with contextlib.suppress(Exception):
                    (self.data_dir / "offline_training_status.json").unlink(missing_ok=True)
            else:
                self._render_offline_training(ofs)
        if self.universe_stats:
            self._render_trading_pipeline()
        _mode = self.bot_config.get("trading_mode", "paper")
        _mode_label = "PAPER" if _mode == "paper" else ("LIVE" if _mode == "live" else "OFFLINE")

        _training_items = [_item for _item in self.training_stats_all if isinstance(_item.get("stats"), dict)]
        if not _training_items:
            _ts_nonempty = any(v for v in ts.values() if v)
            if not _ts_nonempty:
                print(f"\033[1m🤖 {_mode_label} BOT TRAINING\033[0m  {_ANSI_DIM}(no live bot running){_ANSI_RST}\n")
                return
            _training_items = [
                {
                    "symbol": self.active_sym,
                    "timeframe_minutes": self.active_tf_min,
                    "stats": ts,
                }
            ]

        for _idx, _item in enumerate(_training_items):
            _its = _item.get("stats", {})
            _tf_for_label = _item.get("timeframe_minutes") or self.active_tf_min
            _sym_for_label = str(_item.get("symbol") or self.active_sym or "").upper()
            _scope = (
                f"{_sym_for_label} M{int(_tf_for_label)}"
                if _sym_for_label and _tf_for_label
                else (f"M{int(_tf_for_label)}" if _tf_for_label else "BOT")
            )
            _item_pm_payload = self._load_bot_production_metrics(_sym_for_label, int(_tf_for_label or 0))
            _item_pm = _item_pm_payload.get("metrics", pm) if isinstance(_item_pm_payload, dict) else pm
            _train_label = f"{_mode_label} {_scope} TRAINING" if _scope != "BOT" else f"{_mode_label} BOT TRAINING"
            print(f"\033[1m🤖 {_train_label}\033[0m\n")
            trig_ready = _its.get("trigger_ready", False)
            harv_ready = _its.get("harvester_ready", False)
            trig_steps = _its.get("trigger_training_steps", 0)
            harv_steps = _its.get("harvester_training_steps", 0)
            self._render_live_trigger_agent(_its, _item_pm)
            self._render_live_harvester_agent(_its, _item_pm)
            self._render_live_arena_and_health(_its)
            if _idx < len(_training_items) - 1:
                print("  " + "─" * (self._term_width() - 4))
                print()

        # Next-update hint — training stats only refresh on bar close
        _nbc_raw = self.market_stats.get("next_bar_close_utc")
        if not _nbc_raw and self.active_sym and self.active_tf_min:
            _aps = self._load_bot_stats(self.active_sym, self.active_tf_min)
            _nbc_raw = _aps.get("next_bar_close_utc")
        if not _nbc_raw:
            _nbc_raw = self.bot_config.get("next_bar_close_utc")
        _tf_min = (
            self.market_stats.get("timeframe_minutes") or self.bot_config.get("timeframe_minutes") or self.active_tf_min
        )
        if _nbc_raw:
            try:
                _nbc_dt = datetime.fromisoformat(_nbc_raw)
                if _nbc_dt.tzinfo is None:
                    _nbc_dt = _nbc_dt.replace(tzinfo=UTC)
                _rem = (_nbc_dt - datetime.now(UTC)).total_seconds()
                if _rem < 0:
                    _hint = "bar closing…"
                elif _rem < 60:
                    _hint = f"{int(_rem)}s"
                elif _rem < 3600:
                    _m, _s = divmod(int(_rem), 60)
                    _hint = f"{_m}m {_s:02d}s"
                elif _rem < 86400:
                    _h, _r = divmod(int(_rem), 3600)
                    _m = _r // 60
                    _hint = f"{_h}h {_m:02d}m"
                else:
                    _d = int(_rem) // 86400
                    _h = (int(_rem) % 86400) // 3600
                    _hint = f"{_d}d {_h}h"
                print(f"  {_ANSI_DIM}ℹ️  Training stats update on bar close — next in {_hint}{_ANSI_RST}")
            except Exception:
                pass
        elif _tf_min:
            # No bar building yet — just show the timeframe so user knows the cadence
            if _tf_min >= 1440:
                _lbl = f"{_tf_min // 1440}d"
            elif _tf_min >= 60:
                _lbl = f"{_tf_min // 60}h"
            else:
                _lbl = f"{_tf_min}m"
            print(f"  {_ANSI_DIM}ℹ️  Training stats update every {_lbl} bar close (awaiting first tick){_ANSI_RST}")

    def _render_header(self) -> None:
        """Render header."""
        heartbeat = self.heartbeat_chars[self.heartbeat_idx]
        W = self._term_width()
        inner = W - 2  # space inside the box borders

        # Check for circuit breaker alert
        cb_active = self.risk_stats.get("circuit_breaker", "INACTIVE") == "ACTIVE"
        kurt_gate = self.risk_stats.get("kurtosis_gate_active", False)
        title = "ADAPTIVE RL TRADING BOT - TABBED HUD"
        pad_total = inner - len(title)
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l

        if cb_active:
            alert = "⚠️  CIRCUIT BREAKER ACTIVE - TRADING HALTED ⚠️"
            alert_pad = inner - len(alert)
            al = alert_pad // 2
            ar = alert_pad - al
            print("\033[41;97m╔" + "═" * inner + "╗\033[0m")
            print("\033[41;97m║" + " " * al + alert + " " * ar + "║\033[0m")
            print("\033[41;97m╚" + "═" * inner + "╝\033[0m")
        elif kurt_gate:
            _is_live_mode = getattr(self, "_perf_snapshot_mode", "") == "live"
            _kurt_note = "entries BLOCKED" if _is_live_mode else "entries bypassed in paper mode"
            alert = f"⚡ KURTOSIS GATE ACTIVE — {_kurt_note}"
            alert_pad = inner - len(alert)
            al = alert_pad // 2
            ar = alert_pad - al
            print("\033[43;30m╔" + "═" * inner + "╗\033[0m")
            print("\033[43;30m║" + " " * al + alert + " " * ar + "║\033[0m")
            print("\033[43;30m╚" + "═" * inner + "╝\033[0m")
        else:
            print("╔" + "═" * inner + "╗")
            print("║" + " " * pad_l + title + " " * pad_r + "║")
            print("╚" + "═" * inner + "╝")

        # Bot info
        # Use the active-position bot's paper_stats for correct symbol/tf/uptime
        # in multi-bot setups where bot_config.json is shared (last writer wins).
        _aps = (
            self._load_bot_stats(self.active_sym, self.active_tf_min) if self.active_sym and self.active_tf_min else {}
        )
        symbol = _aps.get("symbol") or self.bot_config.get("symbol", "UNKNOWN")
        _tf_min = _aps.get("timeframe_minutes") or self.bot_config.get("timeframe_minutes")
        tf = self._format_timeframe_minutes_label(_tf_min)
        uptime = _aps.get("uptime_seconds") or self.bot_config.get("uptime_seconds", 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)

        price = self.position.get("current_price", 0)
        now = self.last_update or datetime.now(UTC)
        # When FLAT the bot writes current_price=0.0 — show "—" instead of 0.00000
        _direction = self.position.get("direction", "FLAT")
        _pdec = self._price_decimals(price)
        price_str = f"{price:.{_pdec}f}" if (price and _direction != "FLAT") else "—"

        # Next-bar countdown — computed from next_bar_close_utc (updated every tick)
        _nbc_str = ""
        _nbc_raw = self.market_stats.get("next_bar_close_utc")
        if not _nbc_raw:
            _nbc_raw = self.bot_config.get("next_bar_close_utc")
        if _nbc_raw:
            try:
                _nbc_dt = datetime.fromisoformat(_nbc_raw)
                if _nbc_dt.tzinfo is None:
                    _nbc_dt = _nbc_dt.replace(tzinfo=UTC)
                _rem = (_nbc_dt - datetime.now(UTC)).total_seconds()
                if _rem < 0:
                    _nbc_str = "  📊 bar closing…"
                elif _rem < 60:
                    _nbc_str = f"  📊 next bar {int(_rem)}s"
                elif _rem < 3600:
                    _m, _s = divmod(int(_rem), 60)
                    _nbc_str = f"  📊 next bar {_m}m {_s:02d}s"
                elif _rem < 86400:
                    _h, _r = divmod(int(_rem), 3600)
                    _m = _r // 60
                    _nbc_str = f"  📊 next bar {_h}h {_m:02d}m"
                else:
                    _d = int(_rem) // 86400
                    _h = (int(_rem) % 86400) // 3600
                    _nbc_str = f"  📊 next bar {_d}d {_h}h"
            except Exception:
                pass

        # Phase badge — OFFLINE / PAPER / LIVE
        _mode = self.bot_config.get("trading_mode", "paper")
        if _mode == "live":
            _mode_badge = f"{_ANSI_G}● LIVE{_ANSI_RST}"
        elif _mode == "paper":
            _mode_badge = f"{_ANSI_Y}● PAPER{_ANSI_RST}"
        else:
            _mode_badge = f"{_ANSI_DIM}● OFFLINE{_ANSI_RST}"

        print(f"\n🎯 {symbol} @ {tf}  {_mode_badge}    💲 {price_str}    ⏱  {hours:02d}h {minutes:02d}m{_nbc_str}")
        _fleet: dict[str, list[int]] = {}
        for _bot in self.all_bots_stats:
            _sym = str(_bot.get("symbol", "") or "").upper()
            try:
                _tfm = int(_bot.get("timeframe_minutes", 0) or 0)
            except (TypeError, ValueError):
                _tfm = 0
            if _sym and _tfm > 0:
                _fleet.setdefault(_sym, []).append(_tfm)
        if _fleet:
            _chunks: list[str] = []
            for _sym in sorted(_fleet.keys()):
                _tfs = sorted(set(_fleet[_sym]))
                _tf_labels = ",".join([self._format_timeframe_minutes_label(_v) for _v in _tfs])
                _chunks.append(f"{_sym}[{_tf_labels}]")
            print(f"{_ANSI_DIM}🧭 Active paper TFs: {' | '.join(_chunks)}{_ANSI_RST}")
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

    def _render_position_block(self) -> None:
        """Render the position header block (always fixed height to avoid layout jumps)."""
        _mode = self.bot_config.get("trading_mode", "paper")
        _mode_tag = (
            f"  {_ANSI_Y}(paper){_ANSI_RST}"
            if _mode == "paper"
            else (f"  {_ANSI_G}(live){_ANSI_RST}" if _mode == "live" else "")
        )
        print(f"\n\033[1m📊 POSITION\033[0m{_mode_tag}")
        _open_positions: list[tuple[str, dict]] = []
        for _bot in self.all_bots_stats or []:
            _pos = _bot.get("_position", {}) if isinstance(_bot, dict) else {}
            if not isinstance(_pos, dict) or str(_pos.get("direction", "FLAT")).upper() == "FLAT":
                continue
            _sym = str(_bot.get("symbol") or _pos.get("symbol") or "?").upper()
            try:
                _tf = int(_bot.get("timeframe_minutes") or _pos.get("timeframe_minutes") or 0)
            except (TypeError, ValueError):
                _tf = 0
            _label = f"{_sym}/{self._format_timeframe_minutes_label(_tf)}" if _tf > 0 else _sym
            _open_positions.append((_label, _pos))
        if not _open_positions and str(self.position.get("direction", "FLAT")).upper() != "FLAT":
            _sym = str(self.position.get("symbol") or self.active_sym or "?").upper()
            _tf = int(self.position.get("timeframe_minutes") or self.active_tf_min or 0)
            _label = f"{_sym}/{self._format_timeframe_minutes_label(_tf)}" if _tf > 0 else _sym
            _open_positions.append((_label, self.position))
        if len(_open_positions) > 1:
            _total_unreal = sum(float(_p.get("unrealized_pnl", 0.0) or 0.0) for _, _p in _open_positions)
            print(
                f"  {_ANSI_B}{len(_open_positions)} open positions{_ANSI_RST}  |  "
                f"Unrealized: {self._pnl_color(_total_unreal)}{_total_unreal:+.2f}{_ANSI_RST}"
            )
            for _label, _p in sorted(_open_positions, key=lambda kv: (kv[0], str(kv[1].get("position_id", ""))))[:8]:
                _direction = str(_p.get("direction", "FLAT")).upper()
                _dc = _ANSI_G if _direction == "LONG" else (_ANSI_R if _direction == "SHORT" else _ANSI_DIM)
                _entry = float(_p.get("entry_price", 0.0) or 0.0)
                _current = float(_p.get("current_price", 0.0) or 0.0)
                _pnl = float(_p.get("unrealized_pnl", 0.0) or 0.0)
                _ticks = _p.get("ticks_held", _p.get("bars_held", 0))
                _dec = self._price_decimals(max(_entry, _current, 0.0))
                print(
                    f"  {_label:<13} {_dc}{_direction:<5}{_ANSI_RST} "
                    f"{_entry:.{_dec}f} → {_current:.{_dec}f}  "
                    f"{self._pnl_color(_pnl)}{_pnl:+.2f}{_ANSI_RST}  ticks:{_ticks}"
                )
            if len(_open_positions) > 8:
                print(f"  {_ANSI_DIM}… {len(_open_positions) - 8} more open positions{_ANSI_RST}")
            return
        direction = self.position.get("direction", "FLAT")
        entry = self.position.get("entry_price", 0)
        current = self.position.get("current_price", 0)
        pnl = self.position.get("unrealized_pnl", 0)
        bars = self.position.get("bars_held", 0)
        ticks = self.position.get("ticks_held", bars)
        if direction == "LONG":
            dir_color = _ANSI_G
        elif direction == "SHORT":
            dir_color = _ANSI_R
        else:
            dir_color = _ANSI_Y
        pnl_color = self._pnl_color(pnl)
        _dec = self._price_decimals(max(entry, current, 0.0))
        # Line 1: direction / entry / price
        if direction == "FLAT":
            print(f"  {dir_color}FLAT{_ANSI_RST}  (no open position)")
        else:
            print(
                f"  {dir_color}{direction}{_ANSI_RST} @ {entry:.{_dec}f} → {current:.{_dec}f}  |  "
                f"PnL: {pnl_color}{pnl:+.2f}{_ANSI_RST}  |  Ticks: {ticks}"
            )
        # Line 2: MFE/MAE (always printed — blank spacer when FLAT for stable layout)
        if direction != "FLAT":
            mfe = self.position.get("mfe", 0.0)
            mae = self.position.get("mae", 0.0)
            mfe_color = _ANSI_G if mfe > 0 else _ANSI_Y
            mae_color = _ANSI_R if mae > 0 else _ANSI_Y
            print(
                f"  MFE: {mfe_color}+{mfe:.2f}{_ANSI_RST}  |  MAE: {mae_color}-{mae:.2f}{_ANSI_RST}  (USD, excl. spread)"
            )
        else:
            print()  # stable height spacer
        # Line 3: PID / tracker (always printed — blank spacer when not available)
        _pid = self.position.get("position_id", "") if direction != "FLAT" else ""
        _tkey = self.position.get("tracker_key", "") if direction != "FLAT" else ""
        if _pid:
            print(f"  {_ANSI_DIM}PID: {_pid}  tracker: {_tkey}{_ANSI_RST}")
        else:
            print()  # stable height spacer

    def _render_all_bots_panel(self) -> None:
        """Render a compact one-row-per-bot fleet summary."""
        bots = self.all_bots_stats
        if not bots:
            print(f"\n\033[1m🤖 ALL BOTS\033[0m  {_ANSI_DIM}No bots currently running{_ANSI_RST}")
            return
        _preferred_tf = {1: 0, 5: 1, 15: 2, 30: 3, 60: 4, 240: 5}
        bots = sorted(
            bots,
            key=lambda b: (
                str(b.get("symbol", "")).upper(),
                _preferred_tf.get(int(b.get("timeframe_minutes", 0) or 0), 999),
                int(b.get("timeframe_minutes", 0) or 0),
            ),
        )
        print(
            f"\n\033[1m🤖 ALL BOTS\033[0m  {_ANSI_DIM}— SESSION metrics (reset on each bot restart; see SYMBOL/TF SNAPSHOT below for lifetime){_ANSI_RST}"
        )
        # Column widths — keep header, row, and separator in lock-step.
        # Widths:  Bot=13  Status=7  Bars=4  Position=22  T-buf=5  H-buf=5
        #          SessTrd=7  SessPnL=11  SessWin=7
        _hdr = (
            f"  {'Bot':<13}  {'Status':<7}  {'Bars':>4}  {'Position':<22}"
            f"  {'T-buf':>5}  {'H-buf':>5}  {'SessTrd':>7}  {'SessPnL':>11}  {'SessWin':>7}"
        )
        print(f"\033[2m{_hdr}\033[0m")
        print("  " + "─" * (_visible_width(_hdr) - 2))
        _now = datetime.now(UTC)
        for bot in bots:
            sym = bot.get("symbol", "?")
            tf = bot.get("timeframe_minutes", 0)
            label = f"{sym}/{self._format_timeframe_minutes_label(tf)}"
            try:
                _updated_at = datetime.fromisoformat(bot.get("updated_at", ""))
                if _updated_at.tzinfo is None:
                    _updated_at = _updated_at.replace(tzinfo=UTC)
                _age = (_now - _updated_at).total_seconds()
            except Exception:
                _age = 9999.0
            conn = bot.get("connection_healthy", False) and bot.get("quote_ok", False)
            _entry = next(
                (
                    _e
                    for _e in self.universe_stats.values()
                    if isinstance(_e, dict)
                    and str(_e.get("symbol", "")).upper() == str(sym).upper()
                    and int(_e.get("timeframe_minutes", 0) or 0) == int(tf or 0)
                ),
                None,
            )
            _pid_alive = bool(_entry.get("_pid_alive", True)) if isinstance(_entry, dict) else True
            _tf_min = int(tf or 0)
            _live_age = max(120.0, min(float(_tf_min * 30), 3600.0)) if _tf_min > 0 else 120.0
            _slow_age = max(180.0, min(float(_tf_min * 120), 7200.0)) if _tf_min > 0 else 180.0
            _awaiting_bar = False
            _nbc_raw = bot.get("next_bar_close_utc")
            if _nbc_raw and _tf_min > 0:
                try:
                    _nbc_dt = datetime.fromisoformat(_nbc_raw)
                    if _nbc_dt.tzinfo is None:
                        _nbc_dt = _nbc_dt.replace(tzinfo=UTC)
                    _secs_to_bar = (_nbc_dt - _now).total_seconds()
                    _awaiting_bar = 0.0 <= _secs_to_bar <= (_tf_min * 60 + 180)
                except Exception:
                    _awaiting_bar = False
            # Status column — emit a fixed 7-visible-cell token regardless of
            # colour codes so the padding below stays aligned.
            if not _pid_alive or not conn:
                status_vis, status_col = "● STALE", _ANSI_R
            elif _age <= _live_age:
                status_vis, status_col = "● LIVE ", _ANSI_G
            elif _age <= _slow_age or _awaiting_bar:
                status_vis, status_col = "● SLOW ", _ANSI_Y
            else:
                status_vis, status_col = "● STALE", _ANSI_R
            status = f"{status_col}{status_vis:<7}{_ANSI_RST}"
            bars = bot.get("bar_count", 0)
            # Position — build visible and colored strings separately to keep columns aligned
            pos = bot.get("_position", {})
            direction = (pos.get("direction") or "FLAT").upper()
            entry_px = pos.get("entry_price", 0.0)
            unreal = pos.get("unrealized_pnl", 0.0)
            if direction != "FLAT":
                visible_pos = f"{direction:<5}@{entry_px:.0f}({unreal:+.0f})"
                dir_c = _ANSI_G if direction == "LONG" else _ANSI_R
                colored_pos = (
                    f"{dir_c}{direction:<5}{_ANSI_RST}"
                    f"@{entry_px:.0f}"
                    f"({self._pnl_color(unreal)}{unreal:+.0f}{_ANSI_RST})"
                )
            else:
                visible_pos = "FLAT"
                colored_pos = f"{_ANSI_DIM}FLAT{_ANSI_RST}"
            pos_pad = " " * max(0, 22 - len(visible_pos))
            trig_buf = bot.get("trigger_buffer", 0)
            harv_buf = bot.get("harvester_buffer", 0)
            trades = int(bot.get("total_trades", 0) or 0)
            pnl = float(bot.get("total_pnl", 0.0) or 0.0)
            wr = float(bot.get("win_rate", 0.0) or 0.0) * 100
            # Truncate bot label so it never spills beyond the 13-cell column.
            label_cell = label if len(label) <= 13 else label[:12] + "…"
            wr_str = f"{wr:>5.1f}%" if trades > 0 else "      -"
            print(
                f"  {label_cell:<13}  {status}  {bars:>4}  {colored_pos}{pos_pad}"
                f"  {trig_buf:>5}  {harv_buf:>5}  {trades:>7}  "
                f"{self._pnl_color(pnl)}{pnl:>+11.2f}{_ANSI_RST}  {wr_str:>7}"
            )

    def _render_overview(self) -> None:
        """Render overview tab - compact summary."""
        self._render_all_bots_panel()
        self._render_position_block()
        print(
            f"  {_ANSI_DIM}(canonical performance source: trade_log.jsonl; mode from performance_snapshot.json){_ANSI_RST}"
        )

        # Account balance / equity
        _mode = (
            getattr(self, "_trade_log_mode", "")
            or getattr(self, "_perf_snapshot_mode", "")
            or self.bot_config.get("trading_mode", "paper")
        )
        _acct_tag = (
            f"  {_ANSI_Y}(paper){_ANSI_RST}"
            if _mode == "paper"
            else (
                f"  {_ANSI_G}(live){_ANSI_RST}"
                if _mode == "live"
                else (f"  {_ANSI_Y}(paper){_ANSI_RST} + {_ANSI_G}(live){_ANSI_RST}" if _mode == "mixed" else "")
            )
        )
        print(f"\n\033[1m💰 ACCOUNT\033[0m{_acct_tag}")
        # Prefer starting_equity from universe.json for the active symbol.
        # bot_config.json is shared across bots; the last writer may reflect a
        # different instrument's equity baseline.
        _starting = self._universe_starting_equity()
        _lifetime_pnl = float(self.all_time_metrics.get("total_pnl", self.lifetime_metrics.get("total_pnl", 0.0)))
        _unreal = float(self.position.get("unrealized_pnl", 0.0))
        # Prefer real broker values (from CollateralReport BA) when available
        _real_bal = self.bot_config.get("real_account_balance")
        _real_eq = self.bot_config.get("real_account_equity")
        _real_mfr = self.bot_config.get("real_margin_free")
        if _real_bal is not None:
            _balance = float(_real_bal)
            _live_tag = "  \033[32m✓ live\033[0m"
        else:
            _balance = _starting + _lifetime_pnl
            _live_tag = "  \033[33m~ est.\033[0m"
        _equity = float(_real_eq) if _real_eq is not None else _balance + _unreal
        _margin_str = f"  |  Free margin: \033[36m{float(_real_mfr):>10.2f}\033[0m" if _real_mfr is not None else ""
        _direction = (self.position.get("direction") or "FLAT").upper()
        if _direction == "FLAT":
            _unreal_str = f"{_ANSI_DIM}—{_ANSI_RST}"
        else:
            _unreal_str = f"{self._pnl_color(_unreal)}{_unreal:+.2f}\033[0m"
        print(
            f"  Balance: {self._pnl_color(_balance - _starting)}{_balance:>10.2f}\033[0m{_live_tag}  |  "
            f"Equity:  {self._pnl_color(_equity - _starting)}{_equity:>10.2f}\033[0m  |  "
            f"Unrealized: {_unreal_str}"
            f"{_margin_str}"
        )

        # Quick metrics
        print(f"\n\033[1m📈 LAST 24H\033[0m  {_ANSI_DIM}(rolling 24-hour window from trade_log.jsonl){_ANSI_RST}")
        _mode = (
            getattr(self, "_trade_log_mode", "")
            or getattr(self, "_perf_snapshot_mode", "")
            or self.bot_config.get("trading_mode", "paper")
        )
        if _mode == "mixed":
            _paper = self.daily_metrics_by_mode.get("paper", {})
            _live = self.daily_metrics_by_mode.get("live", {})
            _p_trades = _paper.get("total_trades", 0)
            _p_wr = _paper.get("win_rate", 0) * 100
            _p_pnl = _paper.get("total_pnl", 0)
            _l_trades = _live.get("total_trades", 0)
            _l_wr = _live.get("win_rate", 0) * 100
            _l_pnl = _live.get("total_pnl", 0)
            print(f"  Paper: {_p_trades} trades | WR {_p_wr:.1f}% | PnL {self._pnl_color(_p_pnl)}{_p_pnl:+.2f}\033[0m")
            print(f"  Live:  {_l_trades} trades | WR {_l_wr:.1f}% | PnL {self._pnl_color(_l_pnl)}{_l_pnl:+.2f}\033[0m")
        else:
            d = self.daily_metrics
            trades = d.get("total_trades", 0)
            wr = d.get("win_rate", 0) * 100
            day_pnl = d.get("total_pnl", 0)
            print(
                f"  Trades: {trades}  |  Win Rate: {wr:.1f}%  |  PnL: {self._pnl_color(day_pnl)}{day_pnl:+.2f}\033[0m"
            )

            recent_pnl = d.get("recent_pnl_sequence", [])
            if recent_pnl and len(recent_pnl) > 1:
                sparkline = self._create_sparkline(recent_pnl[-20:])
                print(f"  Recent: {sparkline}")

        # Build the TF snapshot using union of (a) trade_log metrics and
        # (b) running bots — so zero-trade bots still appear rather than
        # silently dropping TFs.
        _tf_keys: set = set(self.metrics_by_symbol_tf.keys())
        for _bot in self.all_bots_stats or []:
            _s = self._normalize_symbol(_bot.get("symbol"))
            _tfm = int(_bot.get("timeframe_minutes", 0) or 0)
            if _s and _tfm > 0:
                _tf_keys.add((_s, f"M{_tfm}"))
        if _tf_keys:
            _scope = self._epoch_scope_label().upper()
            print(
                f"\n\033[1m🧩 SYMBOL / TF SNAPSHOT\033[0m  {_ANSI_DIM}— {_scope} closed trades from trade_log.jsonl{_ANSI_RST}"
            )
            _sn_hdr = f"  {'Symbol':<9} {'TF':<6} {'Trades':>7} {'Win%':>7} {'PnL $':>11}"
            print(_sn_hdr)
            print("  " + "─" * (_visible_width(_sn_hdr) - 2))
            for _sym, _tf in sorted(_tf_keys, key=lambda kv: (kv[0], self._timeframe_sort_key(kv[1]))):
                _sm = self.metrics_by_symbol_tf.get((_sym, _tf), {})
                _tr = _sm.get("total_trades", 0)
                _wr = _sm.get("win_rate", 0) * 100
                _pnl = _sm.get("total_pnl", 0.0)
                _pc = self._pnl_color(_pnl) if _tr > 0 else _ANSI_DIM
                print(f"  {_sym:<9} {_tf:<6} {_tr:>7} {_wr:>6.1f}% {_pc}{_pnl:>+11.2f}{_ANSI_RST}")

        # Risk snapshot
        print("\n\033[1m⚠️  RISK STATUS\033[0m")
        cb = self.risk_stats.get("circuit_breaker", "INACTIVE")
        regime = self.risk_stats.get("regime", "UNKNOWN")
        zeta = self.risk_stats.get("regime_zeta", 1.0)
        vol = self.risk_stats.get("realized_vol", 0) * 100
        feas = self.risk_stats.get("feasibility", 0.5)

        kurt_gate = self.risk_stats.get("kurtosis_gate_active", False)
        if cb == "ACTIVE":
            cb_status = f"{_ANSI_R}● ACTIVE{_ANSI_RST}"
        elif kurt_gate:
            cb_status = f"{_ANSI_Y}● κ-gate{_ANSI_RST}"
        else:
            cb_status = f"{_ANSI_G}● OK{_ANSI_RST}"
        if feas > FEASIBILITY_HIGH_THRESHOLD:
            feas_color = _ANSI_G
        elif feas > FEASIBILITY_MEDIUM_THRESHOLD:
            feas_color = _ANSI_Y
        else:
            feas_color = _ANSI_R
        _regime_colors = {
            "TRENDING": _ANSI_G,
            "MEAN_REVERTING": _ANSI_Y,
            "TRANSITIONAL": _ANSI_B,
            "UNKNOWN": _ANSI_DIM,
        }
        regime_color = _regime_colors.get(regime, _ANSI_DIM)

        print(
            f"  Circuit: {cb_status}  |  Regime: {regime_color}{regime}\033[0m (ζ={zeta:.2f})  |  Vol: {vol:.2f}%  |  "
            f"Feasibility: {feas_color}{feas:.2f}\033[0m"
        )

        self._render_agent_status_block()

        # Market snapshot
        _market_scope = self._risk_scope_label(self.risk_stats)
        print(f"\n\033[1m🔬 MARKET [{_market_scope}]\033[0m")
        spread = self.market_stats.get("spread", 0)
        vpin = self.market_stats.get("vpin", 0)
        vpin_z = self.market_stats.get("vpin_z", 0)
        imb = self.market_stats.get("imbalance", 0)

        vpin_status = (
            f"{_ANSI_R}⚠️ HIGH{_ANSI_RST}" if abs(vpin_z) > VPIN_HIGH_TOXICITY_THRESHOLD else f"{_ANSI_G}✓{_ANSI_RST}"
        )
        _has_real = self.market_stats.get("has_real_sizes", False)
        _imb_label = "Imb" if _has_real else "QFI"
        _sp_bps = self._spread_bps()
        _sp_col = self._spread_color()
        print(
            f"  Spread: {_sp_col}{spread:.5f} ({_sp_bps:.1f}bp){_ANSI_RST}  |  VPIN: {vpin:.3f} (z={vpin_z:+.1f}) {vpin_status}  |  {_imb_label}: {imb:+.3f}"
        )

        self._render_system_health_block()

        # Alerts from production_metrics.json (e.g. "No trades for 77.4 hours")
        _pm = self.production_metrics.get("metrics", {})
        _alerts = self.production_metrics.get("alerts", [])
        if _alerts:
            print("\n\033[1m🚨 ALERTS\033[0m")
            for _a in _alerts:
                print(f"  {_ANSI_Y}⚠ {_a}{_ANSI_RST}")

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
            f"  Trig buf: {_trig_col}{_trig_buf:,}/{_RT_TRIG_CAP:,} ({_trig_pct:.0f}%){_ANSI_RST} {_rdy_icon(_trig_rdy)}  │  "
            f"Harv buf: {_harv_col}{_harv_buf:,}/{_RT_HARV_CAP:,} ({_harv_pct:.0f}%){_ANSI_RST} {_rdy_icon(_harv_rdy)}"
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

    def _render_health_analyzer(self) -> None:
        """Render self-healing performance analyzer status row."""
        hr = self._health_report
        if not hr:
            print(f"\n\033[1m🔄 SELF-HEAL\033[0m  {_ANSI_DIM}no report yet — runs every 4 h{_ANSI_RST}")
            return

        overall = hr.get("overall_health", "UNKNOWN")
        if overall == "HEALTHY":
            _h_col, _h_icon = _ANSI_G, "🟢"
        elif overall in ("DEGRADED", "WARNING"):
            _h_col, _h_icon = _ANSI_Y, "🟡"
        elif overall == "NO_DATA":
            _h_col, _h_icon = _ANSI_DIM, "⬜"
        else:
            _h_col, _h_icon = _ANSI_R, "🔴"

        # Age of last run
        _gen = hr.get("generated_at", "")
        _age_str = ""
        if _gen:
            try:
                from datetime import timezone as _tz  # noqa: PLC0415
                _dt = datetime.fromisoformat(_gen).replace(tzinfo=_tz.utc) if _gen.endswith("Z") else datetime.fromisoformat(_gen)
                _age_s = (datetime.now(UTC) - _dt).total_seconds()
                if _age_s < 3600:
                    _age_str = f"{_age_s/60:.0f}m ago"
                else:
                    _age_str = f"{_age_s/3600:.1f}h ago"
            except Exception:
                _age_str = ""

        _window = hr.get("analysis_window_hours", 4)
        _header_age = f"  {_ANSI_DIM}({_window:.0f}h window{', ' + _age_str if _age_str else ''}){_ANSI_RST}"
        print(f"\n\033[1m🔄 SELF-HEAL\033[0m  {_h_icon} {_h_col}{overall}{_ANSI_RST}{_header_age}")

        anomalies: list = hr.get("anomalies", [])
        corrections: list = hr.get("corrections_applied", [])
        fleet: dict = hr.get("fleet", {})

        # Fleet summary row
        _n_trades = int(fleet.get("total_trades", 0))
        _wr = float(fleet.get("win_rate", 0)) * 100
        _pf = float(fleet.get("profit_factor", 0))
        _emg = float(fleet.get("emergency_rate", 0)) * 100
        _wr_col = _ANSI_G if _wr >= 50 else (_ANSI_Y if _wr >= 35 else _ANSI_R)
        _pf_col = _ANSI_G if _pf >= 1.2 else (_ANSI_Y if _pf >= 1.0 else _ANSI_R)
        _emg_col = _ANSI_R if _emg > 5 else (_ANSI_Y if _emg > 2 else _ANSI_G)
        print(
            f"  Fleet: {_n_trades} trades │ "
            f"WR {_wr_col}{_wr:.0f}%{_ANSI_RST} │ "
            f"PF {_pf_col}{_pf:.2f}{_ANSI_RST} │ "
            f"Emg {_emg_col}{_emg:.1f}%{_ANSI_RST}"
        )

        # Anomalies
        if anomalies:
            _a_strs = []
            for _a in anomalies[:4]:
                _bot = f"{_a.get('symbol','?')} {_a.get('timeframe','?')}"
                _code = _a.get("code", "?")
                _a_strs.append(f"{_ANSI_Y}⚡ {_bot} {_code}{_ANSI_RST}")
            print(f"  Anomalies: {'  '.join(_a_strs)}")
            if len(anomalies) > 4:
                print(f"  {_ANSI_DIM}  … and {len(anomalies) - 4} more{_ANSI_RST}")
        else:
            print(f"  {_ANSI_G}✓ No anomalies detected{_ANSI_RST}")

        # Last corrections
        if corrections:
            _c_parts = []
            for _c in corrections[:3]:
                _bot = f"{_c.get('symbol','?')} {_c.get('timeframe','?')}"
                _param = _c.get("parameter", "?").replace("_", " ")
                _old = _c.get("old_value")
                _new = _c.get("new_value")
                if _old is not None and _new is not None:
                    _c_parts.append(f"{_bot} {_param} {_old:.3f}→{_new:.3f}")
                else:
                    _c_parts.append(f"{_bot} {_param}")
            print(f"  Applied: {_ANSI_G}{', '.join(_c_parts)}{_ANSI_RST}")

    def _render_performance(self) -> None:
        """Render detailed performance metrics."""
        # Resolve trading mode: prefer trade_log-derived mode (covers all trades),
        # fall back to snapshot mode, then bot_config.
        _mode = (
            getattr(self, "_trade_log_mode", "")
            or getattr(self, "_perf_snapshot_mode", "")
            or self.bot_config.get("trading_mode", "paper")
        )
        if _mode == "paper":
            _mode_tag = f"  {_ANSI_Y}📄 PAPER{_ANSI_RST}"
        elif _mode == "live":
            _mode_tag = f"  {_ANSI_G}💰 LIVE{_ANSI_RST}"
        elif _mode == "mixed":
            _mode_tag = f"  {_ANSI_Y}📄 PAPER{_ANSI_RST} + {_ANSI_G}💰 LIVE{_ANSI_RST}"
        else:
            _mode_tag = ""
        src = (
            f"  {_ANSI_DIM}(source: trade_log.jsonl; portfolio all symbols/timeframes){_ANSI_RST}"
            if self._metrics_from_trade_log
            else ""
        )
        print(f"\n\033[1m📈 PERFORMANCE METRICS (PORTFOLIO)\033[0m{_mode_tag}{src}\n")
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

        # Stats epoch banner
        if self._stats_epoch:
            _epoch_str = self._stats_epoch.strftime("%Y-%m-%d %H:%M")
            _exc_n = self._stats_epoch_excluded
            _exc_pnl = self._stats_epoch_excluded_pnl
            _pnl_c = self._pnl_color(_exc_pnl)
            print(
                f"  {_ANSI_DIM}📅 Stats epoch: {_epoch_str} UTC  "
                f"({_exc_n} older trades excluded, {_pnl_c}{_exc_pnl:+.2f}{_ANSI_RST}{_ANSI_DIM} PnL)  "
                f"[e] to change{_ANSI_RST}\n"
            )

        def _render_period_rows(rows: list[tuple[str, dict]]) -> None:
            _per_hdr = f"  {'Period':<10} {'Trades':>8} {'Win%':>7} {'PnL $':>13} {'TQR':>8} {'PF':>8} {'MaxDD%':>8}"
            print(_per_hdr)
            print("  " + "─" * (_visible_width(_per_hdr) - 2))
            for label, metrics in rows:
                trades = metrics.get("total_trades", 0)
                wr = metrics.get("win_rate", 0) * 100
                pnl = metrics.get("total_pnl", 0)
                sharpe = metrics.get("sharpe_ratio", 0)
                pf = metrics.get("profit_factor", 0)
                maxdd = metrics.get("max_drawdown", 0.0)
                pnl_color = self._pnl_color(pnl)
                if maxdd > DD_HIGH_PCT:
                    dd_color = _ANSI_R
                elif maxdd > DD_WARN_PCT:
                    dd_color = _ANSI_Y
                else:
                    dd_color = _ANSI_G
                print(
                    f"  {label:<10} {trades:>8} {wr:>6.1f}% {pnl_color}{pnl:>+13.2f}{_ANSI_RST} "
                    f"{sharpe:>8.2f} {pf:>8.2f} {dd_color}{maxdd:>7.2f}%{_ANSI_RST}"
                )

        _portfolio_trades = list(self._trade_log_metrics_trades)
        _portfolio_all_trades = list(self._trade_log_all_trades)
        _portfolio_by_mode = {
            _mk: list(self._trade_log_metrics_trades_by_mode.get(_mk, [])) for _mk in ("paper", "live")
        }
        _portfolio_all_by_mode = {
            _mk: list(self._trade_log_all_trades_by_mode.get(_mk, [])) for _mk in ("paper", "live")
        }
        _active_trades = self._active_scope_trades(self._trade_log_metrics_trades)
        _active_all_trades = self._active_scope_trades(self._trade_log_all_trades)
        _active_by_mode = {
            _mk: self._active_scope_trades(self._trade_log_metrics_trades_by_mode.get(_mk, []))
            for _mk in ("paper", "live")
        }
        _active_all_by_mode = {
            _mk: self._active_scope_trades(self._trade_log_all_trades_by_mode.get(_mk, [])) for _mk in ("paper", "live")
        }

        # Column headers — 'TQR' = Trade Quality Ratio (mean/σ of trade PnL in USD).
        # This is NOT an annualised return-based Sharpe ratio.
        if _mode == "mixed":
            print(f"  {_ANSI_Y}📄 PAPER{_ANSI_RST}")
            _render_period_rows(
                self._period_rows_for_trades(_portfolio_by_mode["paper"], _portfolio_all_by_mode["paper"])
            )
            print(f"\n  {_ANSI_G}💰 LIVE{_ANSI_RST}")
            _render_period_rows(
                self._period_rows_for_trades(_portfolio_by_mode["live"], _portfolio_all_by_mode["live"])
            )
            print(f"\n  {_ANSI_DIM}Combined (paper+live){_ANSI_RST}")
        _render_period_rows(self._period_rows_for_trades(_portfolio_trades, _portfolio_all_trades))

        self._render_current_session_performance()

        # Per-symbol breakdown (only when multiple symbols exist)
        if len(self.per_symbol_metrics) > 1:
            print("\n  \033[1mPER SYMBOL\033[0m")
            _ps_hdr = f"  {'Symbol':<10} {'Trades':>7} {'Win%':>7} {'PnL $':>11} {'PF':>7} {'MaxDD%':>8}"
            print(_ps_hdr)
            print("  " + "─" * (_visible_width(_ps_hdr) - 2))
            for _sym in sorted(self.per_symbol_metrics):
                _sm = self.per_symbol_metrics[_sym]
                _tr = _sm.get("total_trades", 0)
                _wr = _sm.get("win_rate", 0) * 100
                _pnl = _sm.get("total_pnl", 0)
                _pf = _sm.get("profit_factor", 0)
                _mdd = _sm.get("max_drawdown", 0)
                _pc = self._pnl_color(_pnl)
                _dc = _ANSI_R if _mdd > DD_HIGH_PCT else (_ANSI_Y if _mdd > DD_WARN_PCT else _ANSI_G)
                print(
                    f"  {_sym:<10} {_tr:>7} {_wr:>6.1f}% {_pc}{_pnl:>+11.2f}{_ANSI_RST} "
                    f"{_pf:>7.2f} {_dc}{_mdd:>7.2f}%{_ANSI_RST}"
                )

        self._render_mode_breakdown()
        self._render_timeframe_mode_breakdown()

        def _q_rows(mode_key: str | None) -> list[tuple[str, dict]]:
            if mode_key:
                return self._period_rows_for_trades(
                    _portfolio_by_mode.get(mode_key, []), _portfolio_all_by_mode.get(mode_key, [])
                )
            return self._period_rows_for_trades(_portfolio_trades, _portfolio_all_trades)

        if _mode == "mixed":
            print(f"\n  {_ANSI_Y}📄 PAPER{_ANSI_RST}")
            self._render_trade_quality(_q_rows("paper"))
            print(f"\n  {_ANSI_G}💰 LIVE{_ANSI_RST}")
            self._render_trade_quality(_q_rows("live"))
            print(f"\n  {_ANSI_DIM}Combined (paper+live){_ANSI_RST}")
            self._render_trade_quality(_q_rows(None))
        else:
            _mk = _mode if _mode in ("paper", "live") else None
            self._render_trade_quality(_q_rows(_mk))

        pm = self.production_metrics.get("metrics", {})
        if self._has_active_scope():
            _scope = self._active_scope_label()
            print(
                f"\n  \033[1mACTIVE BOT DETAIL [{_scope}]\033[0m  {_ANSI_DIM}(per-bot decision-learning scope){_ANSI_RST}"
            )
            if _mode == "mixed":
                print(f"  {_ANSI_Y}📄 PAPER{_ANSI_RST}")
                _render_period_rows(
                    self._period_rows_for_trades(_active_by_mode["paper"], _active_all_by_mode["paper"])
                )
                print(f"\n  {_ANSI_G}💰 LIVE{_ANSI_RST}")
                _render_period_rows(self._period_rows_for_trades(_active_by_mode["live"], _active_all_by_mode["live"]))
                print(f"\n  {_ANSI_DIM}Combined (paper+live){_ANSI_RST}")
            _render_period_rows(self._period_rows_for_trades(_active_trades, _active_all_trades))
            _active_q = _active_trades
            if _mode == "paper":
                _active_q = _active_by_mode.get("paper", [])
            elif _mode == "live":
                _active_q = _active_by_mode.get("live", [])
            self._render_trade_quality(self._period_rows_for_trades(_active_q, _active_all_trades))
            _quality_metrics = _hud_period_metrics(_active_q, self._universe_starting_equity())
        else:
            _quality_metrics = _hud_period_metrics(_portfolio_trades, self._universe_starting_equity())
        # Prediction convergence is scoped to the active bot when available.
        self._render_trade_timing(_quality_metrics, pm)

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
        print(f"\n\033[1m📊 TRADE QUALITY\033[0m  {_ANSI_DIM}(per period){_ANSI_RST}\n")
        _tq_hdr = (
            f"  {'Period':<10} {'Trades':>7} {'Payoff':>8} {'PF':>8} {'Expect':>11} "
            f"{'Sortino':>8} {'Best':>10} {'Worst':>10} {'CW':>4} {'CL':>4} {'W→L':>10}"
        )
        print(_tq_hdr)
        print("  " + "─" * (_visible_width(_tq_hdr) - 2))
        for label, lt in rows:
            total = lt.get("total_trades", 0)
            avg_win = lt.get("avg_win", 0.0)
            avg_loss = lt.get("avg_loss", 0.0)
            profit_f = lt.get("profit_factor", 0.0)
            expect = lt.get("expectancy", 0.0)
            sortino = lt.get("sortino_ratio", 0.0)
            best = lt.get("best_trade", 0.0)
            worst = lt.get("worst_trade", 0.0)
            max_cw = lt.get("max_consec_wins", 0)
            max_cl = lt.get("max_consec_losses", 0)
            w2l = lt.get("winner_to_loser_count", 0)

            abs_loss = abs(avg_loss)
            payoff = avg_win / abs_loss if abs_loss > _PAYOFF_FLOOR else 0.0
            if payoff >= PAYOFF_GOOD_MIN:
                pay_col = _ANSI_G
            elif payoff >= 1.0:
                pay_col = _ANSI_Y
            else:
                pay_col = _ANSI_R
            if profit_f >= PROFIT_FACTOR_GOOD_MIN:
                pf_col = _ANSI_G
            elif profit_f >= 1.0:
                pf_col = _ANSI_Y
            else:
                pf_col = _ANSI_R
            exp_col = _ANSI_G if expect > 0 else _ANSI_R
            cw_col = _ANSI_G if max_cw >= 3 else _ANSI_Y
            cl_col = _ANSI_R if max_cl >= 5 else (_ANSI_Y if max_cl >= 3 else _ANSI_G)
            w2l_pct = (w2l / total * 100) if total > 0 else 0.0
            if total > 0 and w2l > 0:
                w2l_col = _ANSI_R if w2l_pct > 15 else (_ANSI_Y if w2l_pct > 5 else _ANSI_G)
                w2l_str = f"{w2l} ({w2l_pct:.0f}%)"
            else:
                w2l_col = _ANSI_DIM
                w2l_str = "-"
            print(
                f"  {label:<10} {total:>7} "
                f"{pay_col}{payoff:>7.2f}x{_ANSI_RST} "
                f"{pf_col}{profit_f:>8.2f}{_ANSI_RST} "
                f"{exp_col}${expect:>+9.4f}{_ANSI_RST} "
                f"{sortino:>8.3f} "
                f"${best:>+9.2f} ${worst:>+9.2f} "
                f"{cw_col}{max_cw:>4}{_ANSI_RST} {cl_col}{max_cl:>4}{_ANSI_RST} "
                f"{w2l_col}{w2l_str:>10}{_ANSI_RST}"
            )
        print(
            f"  {_ANSI_DIM}(Payoff avg_win/|avg_loss| target ≥1.5; PF gross_profit/gross_loss target ≥1.2; "
            f"Sortino mean/downside-σ){_ANSI_RST}"
        )

        print(f"\n\033[1m🔬 EDGE QUALITY\033[0m  {_ANSI_DIM}(model tuning signals; per period){_ANSI_RST}\n")
        _eq_hdr = (
            f"  {'Period':<10} {'Capture':>8} {'AvgMFE':>10} {'AvgMAE':>10} {'Edge':>10} "
            f"{'Bars':>6} {'ConfW':>7} {'ConfL':>7} {'Gap':>8}"
        )
        print(_eq_hdr)
        print("  " + "─" * (_visible_width(_eq_hdr) - 2))
        for label, lt in rows:
            _avg_mfe = lt.get("avg_mfe", 0.0)
            _avg_mae = lt.get("avg_mae", 0.0)
            _cap_ratio = lt.get("avg_capture_ratio", 0.0)
            _avg_bars = lt.get("avg_bars_held", 0.0)
            _conf_w = lt.get("avg_conf_win", 0.0)
            _conf_l = lt.get("avg_conf_loss", 0.0)
            _edge = _avg_mfe - _avg_mae if _avg_mfe > 0 else 0.0
            _cal_gap = _conf_w - _conf_l
            if _cap_ratio >= 0.60:
                _cap_col = _ANSI_G
            elif _cap_ratio >= 0.40:
                _cap_col = _ANSI_Y
            else:
                _cap_col = _ANSI_R
            _edge_col = _ANSI_G if _edge > 0 else _ANSI_R
            if _conf_w > 0 or _conf_l > 0:
                _cal_col = _ANSI_G if _cal_gap > 0.05 else (_ANSI_Y if _cal_gap > 0 else _ANSI_R)
            else:
                _cal_col = _ANSI_DIM
            print(
                f"  {label:<10} "
                f"{_cap_col}{_cap_ratio:>7.1%}{_ANSI_RST} "
                f"${_avg_mfe:>+9.2f} ${_avg_mae:>9.2f} "
                f"{_edge_col}${_edge:>+9.2f}{_ANSI_RST} "
                f"{_avg_bars:>6.1f} "
                f"{_conf_w:>7.3f} {_conf_l:>7.3f} "
                f"{_cal_col}{_cal_gap:>+8.3f}{_ANSI_RST}"
            )
        print(
            f"  {_ANSI_DIM}(Capture exit_pnl/MFE target ≥60%; Edge MFE-MAE; Gap conf_win-conf_loss +ve=calibrated){_ANSI_RST}"
        )

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
            f"\n  Avg hold time:    {self._format_duration(avg_dur):>8}  {_ANSI_DIM}(source: {_timing_source}){_ANSI_RST}"
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
            ("7 days", self._compute_trade_log_convergence_metrics(weekly)),
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
        _hdr = f"  {'Period':<10} {'n':>5} {'rwΔ pts':>10} {'Accuracy':>9} {'Brier':>8} {'Util':>8} {'Err %':>8}"
        print(_hdr)
        print("  " + "─" * (_visible_width(_hdr) - 2))
        for label, conv in rows:
            rw_delta = conv["avg_runway_delta"]
            rw_acc = conv["avg_runway_accuracy"]
            cc_err = conv["avg_conf_brier"]
            rw_n = conv["runway_samples"]
            cc_n = conv["conf_samples"]
            util = conv["avg_runway_utilization"]
            err_pct = conv["avg_runway_error_pct"]
            if rw_n == 0 and cc_n == 0:
                print(f"  {label:<10} {_ANSI_DIM}{0:>5}      —         —        —        —        —{_ANSI_RST}")
                continue
            if abs(rw_delta) < RUNWAY_DELTA_OK_MAX:
                d_col = _ANSI_G
            elif abs(rw_delta) < RUNWAY_DELTA_WARN_MAX:
                d_col = _ANSI_Y
            else:
                d_col = _ANSI_R
            if rw_acc > RUNWAY_ACCURACY_GOOD:
                a_col = _ANSI_G
            elif rw_acc > RUNWAY_ACCURACY_WARN:
                a_col = _ANSI_Y
            else:
                a_col = _ANSI_R
            if cc_err < CONF_CALIB_OK_MAX:
                b_col = _ANSI_G
            elif cc_err < CONF_CALIB_WARN_MAX:
                b_col = _ANSI_Y
            else:
                b_col = _ANSI_R
            if rw_n == 0:
                u_col = _ANSI_DIM
                e_col = _ANSI_DIM
            else:
                u_col = _ANSI_G if util >= 1.0 else (_ANSI_Y if util >= 0.7 else _ANSI_R)
                e_col = _ANSI_G if err_pct <= 25.0 else (_ANSI_Y if err_pct <= 50.0 else _ANSI_R)
            _n_cell = f"{max(rw_n, cc_n)}"
            print(
                f"  {label:<10} {_n_cell:>5} "
                f"{d_col}{rw_delta:>+9.2f}{_ANSI_RST} "
                f"{a_col}{rw_acc:>9.3f}{_ANSI_RST} "
                f"{b_col}{cc_err:>8.3f}{_ANSI_RST} "
                f"{u_col}{util:>7.3f}x{_ANSI_RST} "
                f"{e_col}{err_pct:>7.1f}%{_ANSI_RST}"
            )

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
            print(f"\n  {_ANSI_DIM}Combined (paper+live){_ANSI_RST}")
            self._render_prediction_convergence_table(
                self._prediction_convergence_rows(
                    self._active_scope_trades(self._trade_log_metrics_trades),
                    self._active_scope_trades(self._trade_log_all_trades),
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
            f"\n  {_ANSI_DIM}(rwΔ pred−actual pts, 0=perfect; Accuracy 1=perfect; Brier 0=perfect, "
            f"0.25=no-skill; Util actual_MFE/predicted_runway_pts_adj; Err % mean abs pred err %){_ANSI_RST}"
        )
        print(
            f"  Platt  a={pa_col}{platt_a:.4f}{_ANSI_RST}  "
            f"b={pa_col}{platt_b:+.4f}{_ANSI_RST}  "
            f"{_ANSI_DIM}(source: production_metrics.json; grey=default, blue=adapted){_ANSI_RST}"
        )

    def _render_jsonl_decision_entries(self, entries: list, mode_filter: str = "") -> None:
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
        for item in collapsed:
            # Collapsed CLOSE_PENDING summary line
            if isinstance(item, str):
                print(item)
                continue

            entry = item

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
                # Rejected entry: show WHY (feasibility, regime, VPIN-z)
                regime = (ctx.get("regime") or "?")[:5]
                feas = _f(reasoning.get("feasibility"))
                vpin_z = _f(ctx.get("vpin_z"))
                cb_ok = reasoning.get("circuit_breakers_ok", True)
                feas_c = _ANSI_R if feas < 0.3 else (_ANSI_Y if feas < 0.6 else _ANSI_G)
                cb_str = f" {_ANSI_R}CB!{_ANSI_RST}" if not cb_ok else ""
                detail = f"ζ:{regime} F:{feas_c}{feas:.2f}{_ANSI_RST} vz:{vpin_z:+.1f}{cb_str}"
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

            print(
                f"  {ts_str:<12} {bot_str:<13} {mode_str} {agent:<10} "
                f"{color}{dec_upper:<10}{_ANSI_RST} "
                f"{conf:>5.3f}  {detail}"
            )
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

    def _render_decision_log(self) -> None:
        """Render the Decision Log tab (Tab 6) — newest entries first."""
        print("\n\033[1m📝 DECISION LOG (ALL BOTS)\033[0m (last 20 entries)\n")
        print(
            f"  {_ANSI_DIM}(canonical source: per-bot logs/audit/decisions.jsonl; Bot column is symbol/timeframe scope){_ANSI_RST}"
        )

        _mode_filter = ""
        if getattr(self, "_trade_log_mode", "") == "mixed":
            _mode_filter = getattr(self, "_perf_snapshot_mode", "") or self.bot_config.get("trading_mode", "")
            if _mode_filter not in ("paper", "live"):
                _mode_filter = ""
            if _mode_filter:
                _mode_lbl = "PAPER" if _mode_filter == "paper" else "LIVE"
                print(f"  {_ANSI_DIM}Showing {_mode_lbl} entries only while trade history is mixed-mode.{_ANSI_RST}")

        _decision_files: list[Path] = []
        _primary = self.data_dir / "logs" / "audit" / "decisions.jsonl"
        _decision_files.extend(sorted(self.data_dir.glob("paper_*_M*/logs/audit/decisions.jsonl")))
        if _primary.exists():
            _decision_files.append(_primary)

        entries_jsonl: list[dict] = []
        if _decision_files:
            _seen: set[tuple[str, str, str, str, str]] = set()
            try:
                for _jf in _decision_files:
                    with open(_jf, encoding="utf-8") as f:
                        for raw_line in f.readlines()[-200:]:
                            stripped = raw_line.strip()
                            if not stripped:
                                continue
                            _entry = json.loads(stripped)
                            _entry.setdefault("_source_path", str(_jf))
                            _tf_label = self._decision_entry_timeframe_label(_entry)
                            if (
                                _jf == _primary
                                and not self._decision_entry_has_scope(_entry)
                                and len(_decision_files) > 1
                            ):
                                continue
                            _ctx_for_key = _entry.get("context", {})
                            if not isinstance(_ctx_for_key, dict):
                                _ctx_for_key = {}
                            _dedupe_key = (
                                str(_entry.get("timestamp") or _entry.get("ts") or _entry.get("time") or ""),
                                str(_entry.get("symbol") or _ctx_for_key.get("symbol") or ""),
                                _tf_label,
                                str(_entry.get("trade_id") or ""),
                                str(_entry.get("event") or _entry.get("decision") or ""),
                            )
                            if _dedupe_key in _seen:
                                continue
                            _seen.add(_dedupe_key)
                            entries_jsonl.append(_entry)
            except Exception:
                entries_jsonl = []

        if entries_jsonl:

            def _ts_key(e: dict) -> str:
                return str(e.get("timestamp") or e.get("ts") or e.get("time") or "")

            entries_jsonl.sort(key=_ts_key, reverse=True)
            self._render_jsonl_decision_entries(entries_jsonl[:20], _mode_filter)
            return

        legacy_files = sorted(self.data_dir.glob("decision_log_*_M*.json"))
        legacy_files.extend(sorted(self.data_dir.glob("paper_*_M*/decision_log_*_M*.json")))
        root_legacy = self.data_dir / "decision_log.json"
        if root_legacy.exists():
            legacy_files.append(root_legacy)
        if not legacy_files:
            print("  ⚠️  No decision log found.")
            print("\n  Expected files:")
            print("    paper_<SYMBOL>_M<TF>/logs/audit/decisions.jsonl  (rich — primary)")
            print("    paper_<SYMBOL>_M<TF>/decision_log_<SYMBOL>_M<TF>.json  (legacy — fallback)")
            return

        try:
            _dec = json.JSONDecoder()
            entries: list = []
            for log_file in legacy_files:
                raw_text = log_file.read_text(encoding="utf-8")
                # Legacy decision logs can end up as multiple appended JSON arrays
                # after restarts. Walk all top-level objects/arrays.
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
                            for _entry in _obj:
                                if isinstance(_entry, dict):
                                    _entry.setdefault("_source_path", str(log_file))
                                entries.append(_entry)
                        elif isinstance(_obj, dict):
                            _obj.setdefault("_source_path", str(log_file))
                            entries.append(_obj)
                    except json.JSONDecodeError:
                        break
        except Exception as e:
            print(f"  ❌ Error reading decision log: {e}")
            return

        if not entries:
            print("  No entries yet. Waiting for bot decisions...")
            return

        self._render_legacy_decision_entries(entries, _mode_filter)

    def _render_risk(self) -> None:
        """Render risk management details."""
        rs = self.risk_stats
        _scope = self._risk_scope_label(rs)
        print(f"\n\033[1m⚠️  RISK MANAGEMENT [{_scope}]\033[0m\n")
        print(f"  {_ANSI_DIM}(source: scoped risk_metrics + circuit_breakers.json){_ANSI_RST}")
        self._render_risk_circuit_breaker(rs)
        self._render_risk_tail(rs)
        self._render_risk_regime(rs)
        self._render_risk_reward_weights(rs)
        self._render_risk_path_geometry(rs)
        self._render_risk_position_sizing(rs)

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
                rs.get("kurtosis_threshold", KURTOSIS_FAT_TAIL_THRESHOLD) or KURTOSIS_FAT_TAIL_THRESHOLD
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
            rs.get("kurtosis_threshold", KURTOSIS_FAT_TAIL_THRESHOLD) or KURTOSIS_FAT_TAIL_THRESHOLD
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
        kurtosis_threshold = float(
            rs.get("kurtosis_threshold", KURTOSIS_FAT_TAIL_THRESHOLD) or KURTOSIS_FAT_TAIL_THRESHOLD
        )
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
            f"{_ANSI_DIM}(excess; >0 = fat tails; gate fires at >{kurtosis_threshold:.1f}){_ANSI_RST}"
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
        """Render market microstructure."""
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
            f"                       {_ANSI_DIM}High +z = informed sellers active → widen stops / reduce size{_ANSI_RST}"
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
        threshold = float(ms.get("kurtosis_threshold", 3.0) or 3.0)
        kurt_gate = bool(ms.get("kurtosis_gate_active", False))
        depth_gate = bool(ms.get("depth_gate_active", False))
        depth_floor = float(ms.get("depth_floor", 0.0) or 0.0)
        depth_bid = float(ms.get("depth_bid", 0.0) or 0.0)
        depth_ask = float(ms.get("depth_ask", 0.0) or 0.0)
        kurt_str = (f"{_ANSI_R}⛔ BLOCKED (fat tails){_ANSI_RST}" if kurt_gate
                    else f"{_ANSI_G}✓ clear{_ANSI_RST}")
        depth_str = (f"{_ANSI_R}⛔ BLOCKED (thin book){_ANSI_RST}" if depth_gate
                     else f"{_ANSI_G}✓ clear{_ANSI_RST}")
        excess = kurt - 3.0
        excess_color = _ANSI_R if excess > 4.0 else (_ANSI_Y if excess > 1.5 else _ANSI_G)
        print(f"    Kurtosis:           {excess_color}{kurt:>9.3f}{_ANSI_RST}  (excess={excess:+.3f}, threshold={threshold:.2f})")
        print(f"    Kurtosis gate:      {kurt_str}")
        print(f"    Depth (bid/ask):    {depth_bid:>7.3f} / {depth_ask:<7.3f}  floor={depth_floor:.3f}")
        print(f"    Depth gate:         {depth_str}")
        print()

    def _render_footer(self) -> None:
        """Render footer with controls and data freshness."""
        W = self._term_width()
        print("\n" + "─" * W)

        idx = self.TAB_ORDER.index(self.current_tab)
        prev_tab = self.TAB_DISPLAY_SHORT[self.TAB_ORDER[(idx - 1) % len(self.TAB_ORDER)]]
        next_tab = self.TAB_DISPLAY_SHORT[self.TAB_ORDER[(idx + 1) % len(self.TAB_ORDER)]]
        tab_pos = f"Tab {idx + 1}/{len(self.TAB_ORDER)}: {self.TAB_DISPLAY_SHORT[self.current_tab]}"
        print(f"  {_ANSI_DIM}← {prev_tab}  |  {tab_pos}  |  {next_tab} →{_ANSI_RST}")

        # Controls — two lines when narrow, one line when wide
        _trades_hint = "  [↑/↓ or j/k] Select  [n/p] Page  [d] Detail  [b] Back" if self.current_tab == "trades" else ""
        _scroll_hint = "" if self.current_tab == "trades" else "  [Wheel/↑↓/j/k/Pg] Scroll"
        ctrl_wide = f"  [1-7/←/→] Tabs  |  [Tab/S+Tab] Cycle  |  [s] Presets  |  [r] Review CB  |  [e] Epoch  |  [h] Help  |  [Alt+K] Kill  |  [q/^Q/^X] Quit{_scroll_hint}{_trades_hint}"
        ctrl_line1 = "  [1-7/←/→] Tabs  |  [Tab/S+Tab] Cycle  |  [s] Presets  |  [r] Review CB"
        ctrl_line2 = f"  [h] Help  |  [Alt+K] Kill  |  [q/^Q/^X] Quit{_scroll_hint}{_trades_hint}"
        if len(ctrl_wide) <= W:
            print(ctrl_wide)
        else:
            print(ctrl_line1)
            print(ctrl_line2)

        # Data freshness — use freshest file across all bots so one stale file
        # cannot trigger false "Bot silent" while others are active.
        _candidates: list[Path] = []
        for _b in self.all_bots_stats or []:
            _sym = _b.get("symbol", "")
            _tf = int(_b.get("timeframe_minutes", 0) or 0)
            if _sym and _tf > 0:
                _ob = self.data_dir / f"order_book_{_sym}_M{_tf}.json"
                _bc = self.data_dir / f"bot_config_{_sym}_M{_tf}.json"
                _ps = self.data_dir / f"paper_stats_{_sym}_M{_tf}.json"
                if _ob.exists():
                    _candidates.append(_ob)
                if _bc.exists():
                    _candidates.append(_bc)
                if _ps.exists():
                    _candidates.append(_ps)
        if not _candidates:
            _ob = self.data_dir / _ORDER_BOOK_FILE
            _bc = self.data_dir / _BOT_CONFIG_FILE
            if _ob.exists():
                _candidates.append(_ob)
            if _bc.exists():
                _candidates.append(_bc)

        if _candidates:
            _latest_mtime = max(_p.stat().st_mtime for _p in _candidates)
            _fage = time.time() - _latest_mtime
            if _fage > DATA_STALE_SECS:
                freshness = f"{_ANSI_R}⚠️  Bot silent ({_fage:.0f}s){_ANSI_RST}"
            elif _fage > DATA_AGING_SECS:
                freshness = f"{_ANSI_Y}⚡ Data aging ({_fage:.0f}s){_ANSI_RST}"
            else:
                freshness = f"{_ANSI_G}✓ Data fresh ({_fage:.1f}s){_ANSI_RST}"
        else:
            freshness = f"{_ANSI_DIM}⏳ Waiting for data...{_ANSI_RST}"

        note = self._current_notification() or "Press 'h' for help and keyboard shortcuts."
        print(f"  {note}  |  {freshness}")
        print("─" * W)

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

    def _render_trades(self) -> None:
        """Render the trade history tab with pagination and optional drill-down."""
        _mode_filter = ""
        if getattr(self, "_trade_log_mode", "") == "mixed":
            _mode_filter = getattr(self, "_perf_snapshot_mode", "") or self.bot_config.get("trading_mode", "")
            if _mode_filter not in ("paper", "live"):
                _mode_filter = ""
        trades_view = self._all_trades
        if _mode_filter:
            trades_view = [t for t in trades_view if t.get("trading_mode") == _mode_filter]
        self._trades_view = trades_view
        if self._trades_detail and self._trades_detail_trade not in trades_view:
            self._trades_detail = False
            self._trades_detail_trade = {}

        W = self._term_width()
        total = len(trades_view)
        if total == 0:
            _empty_label = f" ({_mode_filter})" if _mode_filter else ""
            print(f"\n\033[1m[T] TRADE HISTORY (PORTFOLIO)\033[0m{_empty_label}  No trades recorded yet.")
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
                f"\033[1m[T] TRADES (PORTFOLIO)\033[0m "
                f"[{total}] {pg_str} {_ANSI_DIM}{_bar}{_ANSI_RST} "
                f"PnL {self._pnl_color(total_pnl)}{total_pnl:+.2f}{_ANSI_RST}{_mode_hdr}"
            )
        else:
            print(
                f"\033[1m[T] TRADE HISTORY (PORTFOLIO)\033[0m  "
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
            else:
                print(
                    f"  {_ANSI_Y}⚠  MIXED MODE — paper and live trades combined. "
                    f"Metrics span both modes. See [P] Performance tab for breakdown.{_ANSI_RST}"
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
        _hdr_row = (
            f"  {'#':<{_C_ID}} M {'Date/Time':<{_C_DATE}} {'Dir':<{_C_DIR}} "
            f"{'Sym':<{_C_SYM}} {'TF':<{_C_TF}} {'Entry':>{_C_ENT}} {'Exit':>{_C_EXT}} "
            f"{'PnL $':>{_C_PNL}} {'Cap%':>{_C_CAP}} {'MFE $':>{_C_MFE}} {'MAE $':>{_C_MAE}} "
            f"{'Tks':>{_C_BRS}}  {'Reason':<{_C_RSN}}"
        )
        # Plain-text header width (no ANSI) → matches the rendered row width.
        _hdr_plain_len = len(_hdr_row)
        _sep = "  " + "-" * max(10, min(W - 4, _hdr_plain_len - 2))
        print(f"{_ANSI_DIM}{_hdr_row}{_ANSI_RST}")
        print(_sep)

        # Trade rows
        _dec = self._price_decimals()
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
            _bars = int(_t.get("ticks_held") or _t.get("bars_held") or 0)
            _rsn_raw = _t.get("close_reason") or _t.get("exit_reason") or ""
            _rsn = ("-" if _rsn_raw in ("", "unknown") else _rsn_raw)[:_C_RSN]
            _tid_s = str(_tid)
            if len(_tid_s) > _C_ID:
                _tid_s = _tid_s[: max(0, _C_ID - 1)] + "…"
            _sym_s = str(_sym)
            if len(_sym_s) > _C_SYM:
                _sym_s = _sym_s[: max(0, _C_SYM - 1)] + "…"

            _ts = _t.get("exit_time") or _t.get("entry_time") or ""
            try:
                _date_str = datetime.fromisoformat(_ts).strftime("%b-%d %H:%M")
            except Exception:
                _date_str = _ts[:13]

            _dc = _ANSI_G if _dir == "LONG" else (_ANSI_R if _dir == "SHORT" else _ANSI_DIM)
            _pc = self._pnl_color(_pnl)
            _ep_s = f"{_entry:.{_dec}f}"[-_C_ENT:]
            _xp_s = f"{_exit:.{_dec}f}"[-_C_EXT:]

            _cap_ratio = self._capture_ratio_for_trade(_t)
            if _cap_ratio is not None:
                _cap_val = max(-999.0, min(999.0, _cap_ratio * 100.0))
                _cap_s = f"{_cap_val:>+{_C_CAP - 1}.0f}%"
                _cap_c = self._pnl_color(_cap_val)
            elif float(_pnl or 0.0) < 0.0:
                # No positive excursion => capture is undefined; flag losses red.
                _cap_s = f"{'n/a':>{_C_CAP}}"
                _cap_c = _ANSI_R
            else:
                _cap_s = f"{'—':>{_C_CAP}}"
                _cap_c = _ANSI_DIM

            # Mode badge — single char, always renders 1 column wide
            _tmode = _t.get("trading_mode", "")
            if _tmode == "paper":
                _mb = f"{_ANSI_Y}P{_ANSI_RST}"
            elif _tmode == "live":
                _mb = f"{_ANSI_G}L{_ANSI_RST}"
            else:
                _mb = f"{_ANSI_DIM}?{_ANSI_RST}"

            _row = (
                f"  {_tid_s:<{_C_ID}} {_mb} {_date_str:<{_C_DATE}} "
                f"{_dc}{_dir:<{_C_DIR}}{_ANSI_RST} "
                f"{_sym_s:<{_C_SYM}} {_tf:<{_C_TF}} {_ep_s:>{_C_ENT}} {_xp_s:>{_C_EXT}} "
                f"{_pc}{_pnl:>+{_C_PNL}.2f}{_ANSI_RST} "
                f"{_cap_c}{_cap_s}{_ANSI_RST} "
                f"{_ANSI_G}+{abs(_mfe):>{_C_MFE - 1}.2f}{_ANSI_RST} "
                f"{_ANSI_R}-{abs(_mae):>{_C_MAE - 1}.2f}{_ANSI_RST} "
                f"{_bars:>{_C_BRS}}  {_ANSI_DIM}{_rsn:<{_C_RSN}}{_ANSI_RST}"
            )
            if _row_idx == self._trades_cursor:
                print(f"\033[7m{_row}\033[0m")  # inverted highlight
            else:
                print(_row)

        if self._trades_per_page >= 8:
            print(_sep)

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
        _bars = int(t.get("ticks_held") or t.get("bars_held") or 0)
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
        print(f"  {'MFE:':<16} {_ANSI_G}+{_mfe:.4f} USD{_ANSI_RST}  (max favourable account-currency excursion)")
        print(
            f"  {'MAE:':<16} {_ANSI_R}-{_mae:.4f} USD{_ANSI_RST}  (max adverse account-currency excursion){_ratio_str}"
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
    import signal  # noqa: PLC0415

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
