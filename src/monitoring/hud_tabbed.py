#!/usr/bin/env python3
"""
Tabbed Trading HUD (Heads-Up Display)
=====================================
Terminal-based live dashboard with multiple tabs for organized data display.

Tabs:
  1 - Overview (compact summary)
  2 - Performance (detailed metrics)
  3 - Training (agent stats)
  4 - Risk (risk management)
  5 - Market (microstructure)
  6 - Decision Log (last 20 entries)

Press 1-6 to switch tabs, Tab/Shift+Tab to cycle, s for presets, q or Ctrl+X to quit.
Note: Ctrl+C is ignored to prevent accidental termination when copying text.
"""

import contextlib
import json
import logging
import math
import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.constants import (
    HARVESTER_BUFFER_CAPACITY,
    KURTOSIS_ALERT_THRESHOLD,
    TRIGGER_BUFFER_CAPACITY,
)

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
_PRICE_REF_HIGH: float = 1000.0        # ref_price ≥ 1000 → 2 dp (BTC / Gold)
_PRICE_REF_MED: float = 10.0           # ref_price ≥ 10 → 3 dp; else 5 dp

# Z-Omega quality display bands (training/paper pipeline)
Z_OMEGA_OFFLINE_WARM_MIN: float = 0.8   # zo ≥ 0.8 → yellow in offline results
Z_OMEGA_POSITIVE_MIN: float = 0.5       # zo > 0.5 → green in paper pipeline

# Agent confidence display bands (°healthy°: 0.55 – 0.85)
CONF_HEALTHY_LOW: float = 0.55          # lower bound of healthy confidence range
CONF_HEALTHY_HIGH: float = 0.85         # upper bound of healthy confidence range
CONF_WARM_LOW: float = 0.50             # lower bound of warm confidence range

# Beta (Importance Sampling) display bands
BETA_HOT_MIN: float = 0.8               # beta > 0.8 → fully corrected (green)
BETA_WARM_MIN: float = 0.6              # beta > 0.6 → warm (yellow)

# Epsilon (exploration rate) display bands
EPS_HOT_MAX: float = 0.05              # eps < 0.05 → HOT (green)
EPS_WARM_MAX: float = 0.2              # eps < 0.2 → WARM (yellow); else COLD

# Buffer fill thresholds (fraction 0–1, used inside _pct_bar helper)
_BUF_FILL_HIGH: float = 0.5            # fraction > 0.5 → green
_BUF_FILL_WARN: float = 0.1            # fraction > 0.1 → yellow

# Buffer %-age thresholds (percent-scale 0–100, used in overview row logic)
BUF_PCT_HIGH: float = 50.0             # >50 % fill → green
BUF_PCT_WARN: float = 10.0             # >10 % fill → yellow

# Trend sparkline helper thresholds
TREND_MIN_SAMPLES: int = 6             # minimum history values before trend calc
TREND_DELTA_POS: float = 3.0           # delta_pct > +3 → DEGRADING (red)
TREND_DELTA_NEG: float = -3.0          # delta_pct < -3 → IMPROVING (green)
TREND_SPARK_TAIL: int = 20             # last N values for sparkline
TREND_STEP_PAIRS_MIN: int = 2          # history pairs needed for velocity calc
TREND_RATE_HIGH: float = 5.0           # > 5 steps/min → active training (green)
TREND_RATE_WARN: float = 1.0           # > 1 step/min → slow training (yellow)

# VPIN Z thresholds reused in overview and decision-log rows
_VPIN_OV_HIGH: float = 2.0             # same value as VPIN_HIGH_TOXICITY_THRESHOLD
_VPIN_OV_ELEVATED: float = 1.5         # elevated threshold used in overview row

# Spread / VaR / Vol colour bands — relative to mid price (basis points)
SPREAD_OK_BPS: float = 1.0             # spread < 1.0 bps → green
SPREAD_WARN_BPS: float = 3.0           # spread < 3.0 bps → yellow; else red
VAR_WARN_PCT: float = 1.5              # VaR % > 1.5 → yellow
VAR_HIGH_PCT: float = 3.0              # VaR % > 3 → red
VOL_HIGH_PCT: float = 2.0              # vol % > 2 → red
VOL_WARN_PCT: float = 1.0              # vol % > 1 → yellow
RUNWAY_WARN_BARS: float = 0.5           # runway < 0.5 → yellow (high vol headwind)
RUNWAY_OK_BARS: float = 0.7             # runway > 0.7 → green (smooth conditions)
EFF_HIGH_THRESHOLD: float = 0.6        # path efficiency > 0.6 → green
EFF_WARN_THRESHOLD: float = 0.3        # path efficiency > 0.3 → yellow
_BUDGET_OK_MIN: float = 10.0           # risk_budget > 10 USD → green

# Payoff / profit-factor colour bands
_PAYOFF_FLOOR: float = 1e-9            # zero guard: avg_win / avg_loss
PAYOFF_GOOD_MIN: float = 1.5           # payoff ≥ 1.5 → green
PROFIT_FACTOR_GOOD_MIN: float = 1.2    # PF ≥ 1.2 → green
DD_HIGH_PCT: float = 5.0               # drawdown > 5 % → red
DD_WARN_PCT: float = 2.0               # drawdown > 2 % → yellow

# _fmt_dur conversion constants
_DURATION_HOUR_MINS: float = 90.0      # < 90 min → show as minutes
_DURATION_DAY_MINS: float = 1440.0     # < 1440 min (24 h) → show as hours

# Prediction convergence colour bands
RUNWAY_DELTA_OK_MAX: float = 1.0       # |delta| < 1 pt → perfect (green)
RUNWAY_DELTA_WARN_MAX: float = 3.0     # |delta| > 3 pts → bad (red)
RUNWAY_ACCURACY_GOOD: float = 0.70     # accuracy > 0.70 → green
RUNWAY_ACCURACY_WARN: float = 0.40     # accuracy > 0.40 → yellow
CONF_CALIB_OK_MAX: float = 0.15        # calibration error < 0.15 → green
CONF_CALIB_WARN_MAX: float = 0.30      # calibration error < 0.30 → yellow
PLATT_ADAPTED_DELTA: float = 0.05      # |platt_a − 1.0| or |platt_b| > 0.05 → adapted

# Decision log display
_DEC_LOG_TS_MIN_LEN: int = 19          # timestamp ≥ 19 chars has full HH:MM:SS
_DEC_LOG_VPIN_WARN: float = 2.0        # |vpin_z| > 2 → flag warning icon

# Data-freshness thresholds (footer)
DATA_STALE_SECS: float = 15.0          # bot age > 15 s → "silent" warning
DATA_AGING_SECS: float = 5.0           # bot age > 5 s → "aging" warning

# Position sizing zero-guard
_QTY_FLOOR: float = 1e-9               # guard division in qty-usage ratio

# Signal synthesis imbalance direction hint
_IMBALANCE_DIRECTION_HINT: float = 0.1  # |imbalance| > 0.1 used for directional hint


def _hud_period_metrics(pts: list, starting_equity: float = 10_000.0) -> dict:
    """Compute period performance metrics from a list of trade dicts."""
    if not pts:
        return {}
    pnls = [t.get("pnl", 0.0) for t in pts]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n = len(pnls)
    total_pnl = sum(pnls)
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    mean_p = total_pnl / n
    variance = sum((p - mean_p) ** 2 for p in pnls) / n
    std_p = math.sqrt(variance) if variance > 0 else 0.0
    sharpe = mean_p / std_p if std_p > 0 else 0.0
    # Sortino: downside deviation uses losses-only count, not all-trade count.
    # Dividing by n (all trades) instead of len(losses) inflates Sortino by
    # roughly sqrt(1/win_rate) — e.g. 1.58× at 60% win rate.
    n_losses = max(1, len(losses))
    down_var = sum(p ** 2 for p in pnls if p < 0) / n_losses
    sortino = mean_p / math.sqrt(down_var) if down_var > 0 else 0.0
    cum = 0.0
    peak_equity = starting_equity
    max_dd_pct = 0.0
    for p in pnls:
        cum += p
        equity = starting_equity + cum
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            dd_pct = (peak_equity - equity) / peak_equity * 100.0
            max_dd_pct = max(max_dd_pct, dd_pct)
    max_cw = max_cl = cw = cl = 0
    w2l_count = sum(1 for t in pts if isinstance(t, dict) and t.get("winner_to_loser"))
    for p in pnls:
        if p > 0:
            cw += 1
            cl = 0
        else:
            cl += 1
            cw = 0
        max_cw = max(max_cw, cw)
        max_cl = max(max_cl, cl)
    return {
        "total_trades": n, "win_rate": win_rate, "total_pnl": total_pnl,
        "sharpe_ratio": sharpe, "sortino_ratio": sortino,
        "omega_ratio": min(profit_factor, 99.0), "max_drawdown": max_dd_pct,
        "best_trade": max(pnls), "worst_trade": min(pnls), "avg_trade": mean_p,
        "profit_factor": min(profit_factor, 99.0), "expectancy": expectancy,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "max_consec_wins": max_cw, "max_consec_losses": max_cl,
        "winner_to_loser_count": w2l_count,
        "recent_pnl_sequence": pnls,
    }


def _hud_parse_dt(s: str):
    """Parse an ISO-format datetime string, returning None on failure."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
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
    cutoff_daily  = now - timedelta(hours=24)
    cutoff_weekly = now - timedelta(days=7)
    month_start   = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
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
_ANSI_G = "\033[92m"    # green
_ANSI_Y = "\033[93m"    # yellow
_ANSI_R = "\033[91m"    # red
_ANSI_DIM = "\033[90m"  # dim
_ANSI_B = "\033[94m"    # blue
_ANSI_RST = "\033[0m"   # reset

# Common data file names
_ORDER_BOOK_FILE = "order_book.json"
_BOT_CONFIG_FILE = "bot_config.json"

# Live training section layout constants
_RT_BAR_LEN: int = 26        # fill-bar character width
_RT_TRIG_CAP: int = TRIGGER_BUFFER_CAPACITY    # trigger replay-buffer capacity
_RT_HARV_CAP: int = HARVESTER_BUFFER_CAPACITY   # harvester replay-buffer capacity


class TabbedHUD:
    """Real-time tabbed HUD for trading bot monitoring"""

    TABS = {"1": "overview", "2": "performance", "3": "training", "4": "risk", "5": "market", "6": "log", "7": "trades"}

    TAB_ORDER = ["overview", "performance", "training", "risk", "market", "log", "trades"]

    TAB_DISPLAY = {
        "overview":    "📊 Overview",
        "performance": "📈 Performance",
        "training":    "🧠 Training",
        "risk":        "⚠️  Risk",
        "market":      "🔬 Market",
        "log":         "📝 Decision Log",
        "trades":      "📋 Trades",
    }

    # No-emoji variant for terminals 85–103 cols wide
    TAB_DISPLAY_MEDIUM = {
        "overview":    "Overview",
        "performance": "Performance",
        "training":    "Training",
        "risk":        "Risk",
        "market":      "Market",
        "log":         "Decision Log",
        "trades":      "Trades",
    }

    # Abbreviated variant for terminals < 85 cols wide
    TAB_DISPLAY_SHORT = {
        "overview":    "Overview",
        "performance": "Perf",
        "training":    "Train",
        "risk":        "Risk",
        "market":      "Market",
        "log":         "Log",
        "trades":      "Trades",
    }

    def __init__(self, refresh_rate: float = 1.0):
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
        self.training_stats_h4: dict = {}   # H4 shadow policy stats
        self.risk_stats = {}
        self.market_stats = {}
        self.bot_config = {}
        self.production_metrics = {}
        self.offline_stats: dict = {}       # offline_training_status.json
        self.offline_job_progress: dict = {}  # keyed by (symbol, tf_minutes)
        self.universe_stats: dict = {}        # universe.json + PID liveness
        self.all_bots_stats: list = []        # one entry per paper_stats_*.json
        # Active-position bot identity (populated each refresh from position file metadata)
        self.active_sym: str = ""
        self.active_tf_min: int = 0
        self._active_pos_file: str = ""   # path of the file that provided self.position
        self.self_test_results: list = []
        self._metrics_from_trade_log = False
        # Loss history for trend / sparkline (non-zero samples only)
        self._trig_loss_hist: deque = deque(maxlen=40)
        self._harv_loss_hist: deque = deque(maxlen=40)
        # Step-time pairs for training velocity: (wall_time, steps)
        self._trig_step_hist: deque = deque(maxlen=12)
        self._harv_step_hist: deque = deque(maxlen=12)
        self.last_update = None
        self.notification = ""
        self.notification_expiry = datetime.min
        self.profile_options = self._load_profile_options()

        # Time-based metrics
        self.daily_metrics = {}
        self.weekly_metrics = {}
        self.monthly_metrics = {}
        self.lifetime_metrics = {}
        self.per_symbol_metrics: dict[str, dict] = {}

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
        self._all_trades: list = []          # newest-first sorted
        self._all_trades_loaded_at: float = 0.0

    def _term_width(self) -> int:
        """Return current terminal column count (fallback 80)."""
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80

    def start(self):
        """Start HUD"""
        self.running = True
        # Set terminal to raw mode for key input
        try:
            if sys.stdin.isatty():
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                self.raw_mode_enabled = True
        except Exception:  # noqa: BLE001 — terminal may not support raw mode
            pass

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop HUD"""
        self.running = False
        # Restore terminal settings
        self._disable_raw_mode()
        if self.thread:
            self.thread.join(timeout=2)

    def _handle_escape_sequence(self, seq1: str) -> None:
        """Handle CSI / Alt-key escape sequences following the ESC byte."""
        if seq1 == "[":  # CSI sequence (e.g. Shift+Tab = \x1b[Z)
            if select.select([sys.stdin], [], [], 0.01)[0]:
                seq2 = sys.stdin.read(1)
                if seq2 == "Z":  # Shift+Tab
                    idx = self.TAB_ORDER.index(self.current_tab)
                    self.current_tab = self.TAB_ORDER[(idx - 1) % len(self.TAB_ORDER)]
        elif seq1.lower() == "k":  # Alt+K — emergency kill switch (handle both 'k' and 'K')
            self._handle_kill_switch()

    def _check_input(self):
        """Check for keyboard input (non-blocking)"""
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key in self.TABS:
                    self.current_tab = self.TABS[key]
                elif key == "\t":  # Tab key to cycle forward
                    idx = self.TAB_ORDER.index(self.current_tab)
                    self.current_tab = self.TAB_ORDER[(idx + 1) % len(self.TAB_ORDER)]
                elif key == "\x1b":  # Escape sequence: Shift+Tab or Alt+<key>
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        seq1 = sys.stdin.read(1)
                        self._handle_escape_sequence(seq1)
                elif key.lower() == "q" or key in {"\x18", "\x11"}:  # 'q' or Ctrl+X (\x18) or Ctrl+Q (\x11)
                    self.running = False
                elif key.lower() == "s":
                    self._handle_symbol_selection()
                elif key.lower() == "r":
                    self._handle_cb_reset()
                elif key.lower() == "h":
                    self._show_help()
                elif key.lower() == "n":
                    if self.current_tab == "trades":
                        self._trades_detail = False
                        _max_pg = max(0, (len(self._all_trades) - 1) // self._trades_per_page)
                        self._trades_page = min(self._trades_page + 1, _max_pg)
                        self._trades_cursor = 0
                elif key.lower() == "p":
                    if self.current_tab == "trades":
                        self._trades_detail = False
                        self._trades_page = max(0, self._trades_page - 1)
                        self._trades_cursor = 0
                elif key.lower() == "j":
                    if self.current_tab == "trades" and not self._trades_detail:
                        _page_cnt = min(self._trades_per_page, len(self._all_trades) - self._trades_page * self._trades_per_page)
                        self._trades_cursor = min(self._trades_cursor + 1, max(0, _page_cnt - 1))
                elif key.lower() == "k":
                    if self.current_tab == "trades" and not self._trades_detail:
                        self._trades_cursor = max(0, self._trades_cursor - 1)
                elif key.lower() == "d":
                    if self.current_tab == "trades":
                        if self._trades_detail:
                            self._trades_detail = False
                        else:
                            _idx = self._trades_page * self._trades_per_page + self._trades_cursor
                            if _idx < len(self._all_trades):
                                self._trades_detail_trade = self._all_trades[_idx]
                                self._trades_detail = True
                elif key.lower() == "b" and self.current_tab == "trades" and self._trades_detail:
                    self._trades_detail = False
        except Exception:  # noqa: BLE001 — ignore terminal read errors silently
            pass

    def _update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                deadline = time.monotonic() + self.refresh_rate
                self._check_input()
                self._refresh_data()
                self._render()
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
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

    def _accumulate_loss_history(self) -> None:
        """Append current training losses/steps to rolling history deques."""
        _tl = self.training_stats.get("trigger_loss", 0.0)
        _hl = self.training_stats.get("harvester_loss", 0.0)
        _now = time.time()
        if _tl > 0:
            self._trig_loss_hist.append(_tl)
        if _hl > 0:
            self._harv_loss_hist.append(_hl)
        _ts = self.training_stats.get("trigger_training_steps", 0)
        _hs = self.training_stats.get("harvester_training_steps", 0)
        if _ts > 0:
            self._trig_step_hist.append((_now, _ts))
        if _hs > 0:
            self._harv_step_hist.append((_now, _hs))

    @staticmethod
    def _load_paper_bot_stats(symbol: str, tf: int) -> dict:
        """Load per-bot paper stats file written by ctrader_ddqn_paper."""
        path = Path("data") / f"paper_stats_{symbol}_M{tf}.json"
        try:
            with open(path, encoding="utf-8") as _f:
                return json.load(_f)
        except Exception:
            return {}

    def _load_universe_stats(self) -> None:
        """Load universe.json, annotate liveness, auto-prune dead entries."""
        _uni_path = self.data_dir / "universe.json"
        if not _uni_path.exists():
            self.universe_stats = {}
            return
        try:
            _uni_raw: dict = json.loads(_uni_path.read_text())
            # Support both flat {"XAUUSD": {...}} and nested {"instruments": {...}}
            _instruments: dict = _uni_raw.get("instruments", _uni_raw)
            _dead_keys: list[str] = []
            for _sym, _entry in list(_instruments.items()):
                if not isinstance(_entry, dict):
                    continue
                _pid = _entry.get("paper_pid")
                _alive = False
                if _pid:
                    try:
                        os.kill(int(_pid), 0)
                        _r = subprocess.run(
                            ["ps", "-p", str(_pid), "-o", "stat", "--no-headers"],
                            capture_output=True, text=True, check=False,
                        )
                        _alive = bool(_r.stdout.strip()) and "Z" not in _r.stdout
                    except OSError:
                        pass
                _entry["_pid_alive"] = _alive
                # Auto-prune: remove entries whose PID is dead — they are stale
                # from a previous pipeline run and should not clutter the HUD.
                if _pid and not _alive:
                    _dead_keys.append(_sym)
                    continue
                _tf = _entry.get("timeframe_minutes", 0)
                _entry["_paper_stats"] = self._load_paper_bot_stats(_sym, _tf) if _tf else {}
            # Remove dead entries from the file itself so they don't reappear
            if _dead_keys:
                for _dk in _dead_keys:
                    _instruments.pop(_dk, None)
                try:
                    _uni_raw["instruments"] = _instruments
                    _uni_path.write_text(json.dumps(_uni_raw, indent=2))
                except Exception:
                    pass
            self.universe_stats = _instruments
        except Exception:
            pass

    def _load_risk_and_orderbook(self) -> None:
        """Load risk_metrics.json then overwrite with fresher order_book.json fields."""
        risk_file = self.data_dir / "risk_metrics.json"
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
                }
            except Exception as e:
                if not hasattr(self, "_risk_error_shown"):
                    self._set_notification(f"⚠️  Error loading risk metrics: {e}", ttl=10)
                    self._risk_error_shown = True

        # order_book.json is fresher — overwrite book-specific fields
        ob_file = self.data_dir / _ORDER_BOOK_FILE
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
            pass

    def _refresh_data(self):
        """Refresh all data from bot exports"""
        # Capture heartbeat timestamp in UTC so header labelling stays accurate
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
                    pass

        # Aggregate position files from all running bots (each writes a per-symbol file).
        # Display the first non-FLAT position found; fall back to singleton file if none.
        import glob as _glob
        _pos_files = sorted(
            _glob.glob(str(self.data_dir / "current_position_*.json")),
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
                pass
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
                try:
                    self.active_tf_min = int(_tf_str)
                except ValueError:
                    pass
        self._load_performance_snapshot()

        self._load_json("training_stats.json", "training_stats")
        self._accumulate_loss_history()

        self._load_json("production_metrics.json", "production_metrics")
        self._load_json("offline_training_status.json", "offline_stats")
        self._load_universe_stats()

        # Per-job live progress files (written by OfflineTrainer worker processes)
        _prog: dict = {}
        for _pf in self.data_dir.glob("offline_progress_*.json"):
            try:
                _d = json.loads(_pf.read_text())
                _prog[(_d["symbol"], _d["timeframe_minutes"])] = _d
            except Exception:
                pass
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
            # FLAT with no active position: load from the most recently written
            # per-bot training_stats_*.json so the Training tab stays live.
            _ts_candidates = sorted(
                [_p for _p in self.data_dir.glob("training_stats_*_M*.json") if "_M240" not in _p.name],
                key=lambda _p: _p.stat().st_mtime if _p.exists() else 0,
                reverse=True,
            )
            if _ts_candidates:
                self._load_json(_ts_candidates[0].name, "training_stats")
                self._accumulate_loss_history()
            # Same for risk/market stats: pick the freshest per-bot risk file
            _rm_candidates = sorted(
                self.data_dir.glob("risk_metrics_*_M*.json"),
                key=lambda _p: _p.stat().st_mtime if _p.exists() else 0,
                reverse=True,
            )
            if _rm_candidates:
                self._load_json(_rm_candidates[0].name, "risk_stats")
                self._apply_risk_stats_to_market_stats()

        # Self-test results (written at startup by run_self_test())
        st_file = self.data_dir / "self_test.json"
        if st_file.exists():
            try:
                with open(st_file) as f:
                    self.self_test_results = json.load(f).get("results", [])
            except Exception:
                pass

        # H4 shadow training stats (written by _export_h4_training_stats)
        _h4_sym = self.active_sym or self.bot_config.get("symbol", "")
        self.training_stats_h4 = {}
        if _h4_sym:
            _h4_ts_path = self.data_dir / f"training_stats_{_h4_sym}_M240.json"
            if _h4_ts_path.exists():
                try:
                    self.training_stats_h4 = json.loads(_h4_ts_path.read_text())
                except Exception:
                    pass

        # Re-apply order_book.json on top of any per-bot risk-file override.
        # The per-bot risk_metrics_SYM_MTF.json is written at bar-close (every N minutes)
        # while order_book.json is written per-tick, so it always has the freshest
        # spread / bids / asks / vpin / imbalance.  Without this the market-structure
        # tab shows stale bar-close values.
        _ob_final = self.data_dir / _ORDER_BOOK_FILE
        if _ob_final.exists():
            try:
                with open(_ob_final) as _f:
                    _ob = json.load(_f)
                _ms = self.market_stats
                _ms["spread"]           = _ob.get("spread",           _ms.get("spread", 0.0))
                _ms["depth_bid"]        = _ob.get("depth_bid",        _ms.get("depth_bid", 0.0))
                _ms["depth_ask"]        = _ob.get("depth_ask",        _ms.get("depth_ask", 0.0))
                _ms["vpin"]             = _ob.get("vpin",             _ms.get("vpin", 0.0))
                _ms["vpin_z"]           = _ob.get("vpin_zscore",      _ms.get("vpin_z", 0.0))
                _ob_imb = _ob.get("imbalance")
                if _ob_imb is not None:
                    _ms["imbalance"] = float(_ob_imb)
                _ms["order_book_bids"]  = _ob.get("order_book_bids",  _ms.get("order_book_bids", []))
                _ms["order_book_asks"]  = _ob.get("order_book_asks",  _ms.get("order_book_asks", []))
                _ms["has_real_sizes"]   = _ob.get("has_real_sizes",   False)
                _ms["qfi_update_count"] = _ob.get("qfi_update_count", 0)
            except Exception:
                pass

        # All-bots fleet panel: load every paper_stats_*.json + matching position file
        _all_bots: list[dict] = []
        for _psf in sorted(self.data_dir.glob("paper_stats_*.json")):
            try:
                _ps = json.loads(_psf.read_text())
                _sym = _ps.get("symbol", "")
                _tf  = _ps.get("timeframe_minutes", 0)
                _pos_f = self.data_dir / f"current_position_{_sym}_M{_tf}.json"
                _ps["_position"] = json.loads(_pos_f.read_text()) if _pos_f.exists() else {}
                _all_bots.append(_ps)
            except Exception:
                pass
        self.all_bots_stats = _all_bots
        # Trade history — cache-loaded (5s) for trades tab
        self._load_all_trades_cached()

    def _load_profile_options(self):
        """Load preset symbol/timeframe profiles for selection UI"""
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

    def _compute_metrics_from_trade_log(self):
        """Compute performance metrics directly from trade_log.jsonl.

        Always re-classifies trades by the rolling time windows (daily/weekly/
        monthly) because those windows advance with wall-clock time even when
        the file itself hasn't changed.  Parsing 999 JSONL lines is <1 ms.
        """
        trade_file = Path("data/trade_log.jsonl")
        if not trade_file.exists():
            return
        # Resolve starting_equity: universe.json entries are authoritative (they
        # reflect the real account size); bot_config.json is shared across bots
        # and may carry a stale or default value from whichever bot wrote last.
        _uni_eq = (
            self.universe_stats.get(self.active_sym, {}).get("starting_equity")
            or next(
                (e.get("starting_equity") for e in self.universe_stats.values()
                 if isinstance(e, dict) and e.get("starting_equity")),
                None,
            )
            or self.bot_config.get("starting_equity", 10_000.0)
        )
        starting_equity = float(_uni_eq)
        try:
            trades = []
            with open(trade_file, encoding="utf-8") as f:
                for raw_line in f:
                    stripped = raw_line.strip()
                    if stripped:
                        trades.append(json.loads(stripped))
        except Exception:
            return
        if not trades:
            return

        # DATA QUALITY CHECK: Log warnings for data integrity issues
        _null_entry_time = sum(1 for t in trades if t.get("entry_time") is None)
        _missing_quantity = sum(1 for t in trades if "quantity" not in t or t.get("quantity") is None)
        _recalc_trades = sum(1 for t in trades if t.get("pnl_recalculated"))

        if _null_entry_time > 0:
            LOG.warning(
                "[DATA-QUALITY] %d/%d trades have NULL entry_time (will be excluded from duration calc)",
                _null_entry_time, len(trades)
            )
        if _missing_quantity > 0:
            LOG.warning(
                "[DATA-QUALITY] %d/%d trades missing 'quantity' field (HUD cannot display position sizing)",
                _missing_quantity, len(trades)
            )
        if _recalc_trades > 0:
            _original_pnl = sum(t.get("pnl_original", 0) for t in trades if "pnl_original" in t)
            _current_pnl = sum(t.get("pnl", 0) for t in trades)
            _variance = abs(_current_pnl - _original_pnl)
            LOG.warning(
                "[DATA-QUALITY] %d/%d trades recalculated. Original PnL: $%.2f, Current: $%.2f, Variance: $%.2f",
                _recalc_trades, len(trades), _original_pnl, _current_pnl, _variance
            )

        # Determine active trading mode; if all trades share one mode, use it.
        _modes = {t.get("trading_mode", "") for t in trades}
        _modes.discard("")
        self._trade_log_mode = next(iter(_modes)) if len(_modes) == 1 else "mixed" if _modes else ""

        daily, weekly, monthly = _classify_trades_by_period(trades)
        self.daily_metrics = _hud_period_metrics(daily, starting_equity)
        self.weekly_metrics = _hud_period_metrics(weekly, starting_equity)
        self.monthly_metrics = _hud_period_metrics(monthly, starting_equity)
        self.lifetime_metrics = _hud_period_metrics(trades, starting_equity)

        # Augment lifetime_metrics with timing data derived from trade timestamps.
        # These are more accurate than the runtime-counter values in production_metrics.json
        # which reset on each bot session and only reflect the current session.
        _durations: list[float] = []
        _trades_with_complete_times = 0
        _last_exit_dt = None
        for _t in trades:
            _entry_dt = _hud_parse_dt(_t.get("entry_time", ""))
            _exit_dt  = _hud_parse_dt(_t.get("exit_time", ""))
            if _entry_dt and _exit_dt:
                _durations.append((_exit_dt - _entry_dt).total_seconds() / 60.0)
                _trades_with_complete_times += 1
            if _exit_dt and (_last_exit_dt is None or _exit_dt > _last_exit_dt):
                _last_exit_dt = _exit_dt
        _now = datetime.now(UTC)
        self.lifetime_metrics["avg_trade_duration_mins"] = (
            sum(_durations) / len(_durations) if _durations else 0.0
        )
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

    def _load_json(self, filename: str, attr: str):
        """Load JSON file into attribute"""
        filepath = self.data_dir / filename
        if filepath.exists():
            try:
                with open(filepath) as f:
                    setattr(self, attr, json.load(f))
            except Exception:  # noqa: BLE001 — stale/partial JSON; keep last good value
                pass

    def _apply_risk_stats_to_market_stats(self) -> None:
        """Merge risk_stats fields into market_stats, preserving existing values as fallback."""
        _rm = self.risk_stats
        self.market_stats.update({
            "vpin":            _rm.get("vpin",            self.market_stats.get("vpin", 0.0)),
            "vpin_z":          _rm.get("vpin_zscore",    self.market_stats.get("vpin_z", 0.0)),
            "spread":          _rm.get("spread",         self.market_stats.get("spread", 0.0)),
            "imbalance":       _rm.get("imbalance",      self.market_stats.get("imbalance", 0.0)),
            "depth_bid":       _rm.get("depth_bid",      self.market_stats.get("depth_bid", 0.0)),
            "depth_ask":       _rm.get("depth_ask",      self.market_stats.get("depth_ask", 0.0)),
            "order_book_bids": _rm.get("order_book_bids", self.market_stats.get("order_book_bids", [])),
            "order_book_asks": _rm.get("order_book_asks", self.market_stats.get("order_book_asks", [])),
        })

    def _disable_raw_mode(self):
        """Return terminal to original mode for blocking input prompts"""
        if self.raw_mode_enabled and self.old_settings and sys.stdin.isatty():
            with contextlib.suppress(BaseException):
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            self.raw_mode_enabled = False

    def _enable_raw_mode(self):
        """Re-enter raw mode after prompt interactions"""
        if (not self.raw_mode_enabled) and self.old_settings and sys.stdin.isatty():
            try:
                tty.setcbreak(sys.stdin.fileno())
                self.raw_mode_enabled = True
            except Exception:  # noqa: BLE001 — terminal may lose raw-mode capability
                pass

    def _show_help(self):  # noqa: PLR0915
        """Display help screen with keyboard shortcuts and information"""
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
            print("  [Tab]         - Cycle to next tab")
            print("  [Shift+Tab]   - Cycle to previous tab")
            print("  [s]           - Select symbol/timeframe preset")
            print("  [h]           - Show this help screen")
            print("  [q] / Ctrl+Q / Ctrl+X  - Quit HUD")
            print("  [Alt+K]       - Emergency kill switch (close all positions + halt trading)")
            print("  [r]           - Review tripped circuit breakers and reset if OK")

            print("\n\033[1m📋 TRADE HISTORY TAB KEYS\033[0m\n")
            print("  [j] / [k]     - Move selection down / up")
            print("  [n] / [p]     - Next / previous page")
            print("  [d]           - Drill into selected trade (full detail view)")
            print("  [b] / [d]     - Back from detail view to trade list")

            print("\n\033[1m📊 TAB DESCRIPTIONS\033[0m\n")
            print("  Overview      - Quick snapshot of position, daily stats, risk, and health")
            print("  Performance   - Detailed performance metrics (daily/weekly/monthly/lifetime)")
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

    def _handle_kill_switch(self):
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
                ks_path = self.data_dir / "kill_switch.json"
                with open(ks_path, "w") as f:
                    json.dump(
                        {
                            "active": True,
                            "reason": "MANUAL_HUD_KILL",
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                        f,
                    )
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

    def _handle_cb_reset(self):
        """r key: Show tripped circuit breakers with reasons and offer reset."""
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
            for _key, (_label, _explain) in _breaker_labels.items():
                _b = _cb_data.get(_key, {})
                if not isinstance(_b, dict):
                    continue
                _tripped = _b.get("is_tripped", False)
                if _tripped:
                    _any_tripped = True
                    _reason = _b.get("trip_reason", _explain)
                    _tv = _b.get("trip_value", 0.0)
                    _th = _b.get("threshold", 0.0)
                    _trip_ts = _b.get("trip_time", "")
                    _cd_mins = _b.get("cooldown_minutes", 60)
                    print(f"  {RED}✗ {_label}: TRIPPED{RST}")
                    print(f"    Reason:    {YLW}{_reason}{RST}")
                    print(f"    Value:     {_tv:.4f}  (threshold: {_th:.4f})")
                    if _trip_ts:
                        print(f"    Tripped:   {_trip_ts[:19]}")
                        try:
                            _trip_dt = datetime.fromisoformat(_trip_ts)
                            _elapsed = (datetime.now() - _trip_dt).total_seconds() / 60.0
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
            print(YLW + "Type RESET and press Enter to reset all tripped breakers, or press Enter to abort:" + RST)
            confirm = input("> ").strip()
            if confirm == "RESET":
                _reset_path = self.data_dir / "circuit_breaker_reset.json"
                with open(_reset_path, "w") as _f:
                    json.dump(
                        {
                            "reset": True,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                        _f,
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

    def _handle_symbol_selection(self):
        """Interactive prompt for choosing symbol/timeframe presets"""
        if not self.profile_options:
            self._set_notification("No profile presets defined (config/profile_presets.json)", ttl=6)
            return

        self._disable_raw_mode()
        try:
            os.system("clear" if os.name != "nt" else "cls")
            print("Preset Trading Profiles\n")
            for idx, option in enumerate(self.profile_options, start=1):
                print(
                    f"  {idx}. {option['label']}  |  Symbol: {option['symbol']}  |  ID: {option['symbol_id']}  |  M{option['timeframe_minutes']}  |  Qty: {option['qty']}"
                )
            print("\nSelect the profile number to queue it (bot restart required). Type 'c' to cancel.\n")
            choice = input("Selection: ").strip()
            if not choice or choice.lower().startswith("c"):
                return
            try:
                selection_idx = int(choice) - 1
            except ValueError:
                input("Invalid entry. Press Enter to continue...")
                return
            if selection_idx < 0 or selection_idx >= len(self.profile_options):
                input("Option out of range. Press Enter to continue...")
                return
            selected = self.profile_options[selection_idx]
            env_updated = self._apply_profile_selection(selected)
            status = "✓ Profile queued and .env updated." if env_updated else "⚠ Could not update .env (file missing?)."
            print(f"\n{status}")
            print("Restart the bot launcher (run.sh) to apply the new market selection.")
            input("Press Enter to return to the HUD...")
        except Exception as exc:
            input(f"Unexpected error: {exc}. Press Enter to return...")
        finally:
            self._enable_raw_mode()

    def _apply_profile_selection(self, selection: dict[str, Any]) -> bool:
        """Write selection to .env and pending profile file"""
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
        """Update target keys inside .env while preserving other settings"""
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

    def _write_pending_profile(self, selection: dict[str, Any]):
        """Persist requested profile for other tools/dashboard consumers"""
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

    def _set_notification(self, message: str, ttl: int = 5):
        """Display a temporary status message in the footer"""
        self.notification = message
        self.notification_expiry = datetime.now() + timedelta(seconds=ttl)

    def _current_notification(self) -> str:
        if self.notification and datetime.now() < self.notification_expiry:
            return self.notification
        return ""

    def _render(self):
        """Render current tab and always show footer"""
        os.system("clear")

        # Header
        self._render_header()

        # Tab bar
        self._render_tab_bar()

        # Tab content
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

        # Always render footer (bottom menu)
        self._render_footer()

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
            f"{col}[{'█' * filled}{'░' * (_RT_BAR_LEN - filled)}] {beta:.4f}{_ANSI_RST}"
            "  (0.4 cold→1.0 fully corrected)"
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

    def _render_offline_training(self, ofs: dict) -> None:
        """Render the offline training status block."""
        ofs_status = ofs.get("status", "idle")
        ofs_total = ofs.get("total_jobs", 0)
        ofs_done = sum(1 for r in ofs.get("results", []) if r.get("status") in ("done", "error"))
        ofs_elapsed = ofs.get("elapsed_s", 0.0)
        ofs_start = ofs.get("started_at", "")[:19].replace("T", " ") if ofs.get("started_at") else "—"
        ofs_end = ofs.get("completed_at", "")[:19].replace("T", " ") if ofs.get("completed_at") else None
        status_badge = self._offline_status_badge(ofs_status, ofs_done, ofs_total)
        prog_bar = self._offline_progress_bar(ofs_done, ofs_total)
        print(f"\n  \033[1m🏋 OFFLINE TRAINING\033[0m  {status_badge}")
        print(f"    Progress:  {prog_bar}   Elapsed: {ofs_elapsed:.0f}s")
        print(f"    Started:   {ofs_start}" + (f"   Finished: {ofs_end}" if ofs_end else ""))
        _results = ofs.get("results", [])
        if _results:
            self._render_offline_jobs_table(_results)
        print()

    def _offline_status_badge(self, status: str, done: int, total: int) -> str:
        """Build a colorized offline-training status badge."""
        if status == "running":
            return f"{_ANSI_Y}⚙  RUNNING ({done}/{total} done){_ANSI_RST}"
        if status == "complete":
            return f"{_ANSI_G}✓ COMPLETE  ({done}/{total} jobs){_ANSI_RST}"
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
        print(f"    {'Symbol':<{sym_w}}  {'TF':>5}  {'Status':<9}  {'Detail':<38}  ZOmega")
        print(f"    {'─'*sym_w}  {'─'*5}  {'─'*9}  {'─'*38}  {'─'*8}")
        for r in results:
            self._render_offline_job_row(r, sym_w)

    def _render_offline_job_row(self, r: dict, sym_w: int) -> None:
        """Render a single job result row in the offline training table."""
        sym = r.get("symbol", "")
        tf_label = r.get("label", f"M{r.get('timeframe_minutes', '?')}")
        jstatus = r.get("status", "queued")
        jcol, jbadge = self._offline_job_status(jstatus)
        zo_str = self._offline_job_zo_str(r.get("z_omega"), jstatus)
        detail = self._offline_job_detail(jstatus, r)
        row = f"    {sym:<{sym_w}}  {tf_label:>5}  {jcol}{jbadge}{_ANSI_RST}  {detail}  {zo_str}"
        if jstatus == "error" and r.get("error"):
            row += f"  {_ANSI_R}{r['error'][:30]}{_ANSI_RST}"
        print(row)

    def _offline_job_status(self, status: str) -> tuple[str, str]:
        """Return (color, badge) for offline job status."""
        if status == "done":
            return _ANSI_G, "done     "
        if status == "error":
            return _ANSI_R, "ERROR    "
        if status == "running":
            return _ANSI_Y, "running  "
        return _ANSI_DIM, "queued   "

    def _offline_job_zo_str(self, zo: float | None, status: str) -> str:
        """Render ZOmega column for offline job row."""
        if zo is None or status != "done":
            return f"{_ANSI_DIM}{'—':>8}{_ANSI_RST}"
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
            prog = self.offline_job_progress.get(
                (r.get("symbol"), r.get("timeframe_minutes")), {}
            )
            if not prog:
                return f"{'—':<38}"
            pb = prog.get("pct", 0.0)
            pb_fill = int(14 * pb / 100)
            pb_bar = f"[{'█'*pb_fill}{'░'*(14-pb_fill)}] {pb:4.1f}%"
            return (
                f"{_ANSI_Y}{pb_bar}{_ANSI_RST}"
                f"  ε={prog.get('epsilon', 0):.3f}  β={prog.get('beta', 0.4):.3f}"
            )
        return f"{'—':<38}"

    def _render_paper_pipeline(self) -> None:
        """Render the paper trading pipeline status block — one card per bot.

        Skips rendering entirely when there are no live entries to avoid
        showing stale information from dead pipeline processes.
        """
        uni = self.universe_stats
        if not uni:
            return  # nothing to show — all entries were pruned or none exist
        running_count = sum(1 for e in uni.values() if e.get("_pid_alive"))
        total_count   = len(uni)
        hdr_badge = (
            f"{_ANSI_G}{running_count}/{total_count} running{_ANSI_RST}"
            if running_count else
            f"{_ANSI_R}0/{total_count} running{_ANSI_RST}"
        )
        print(f"  \033[1m📈 PAPER TRADING PIPELINE\033[0m  {hdr_badge}")
        print()
        for sym, entry in sorted(uni.items()):
            self._render_paper_pipeline_card(sym, entry)
        print()

    @staticmethod
    def _pp_bar(filled_frac: float, width: int = 8) -> str:
        """Tiny inline progress bar."""
        filled = round(max(0.0, min(1.0, filled_frac)) * width)
        col = _ANSI_G if filled_frac >= 0.5 else (_ANSI_Y if filled_frac >= 0.2 else _ANSI_DIM)
        return f"{col}[{'█' * filled}{'░' * (width - filled)}]{_ANSI_RST}"

    def _render_paper_pipeline_card(self, sym: str, entry: dict) -> None:
        """Render one paper-bot card with connection + training + activity stats."""
        stage    = entry.get("stage", "?")
        tf_min   = entry.get("timeframe_minutes", 0)
        tf_lbl   = f"M{tf_min}" if tf_min else "?"
        zo       = entry.get("z_omega")
        pid      = entry.get("paper_pid")
        alive    = entry.get("_pid_alive", False)
        ps       = entry.get("_paper_stats", {})   # per-bot stats JSON from bot

        # ── title line ────────────────────────────────────────────────────────
        stage_col = {
            "PAPER": _ANSI_Y, "LIVE": _ANSI_G,
            "UNTRAINED": _ANSI_DIM, "DEMOTED": _ANSI_R,
        }.get(stage, _ANSI_DIM)
        if zo is not None:
            zo_c = _ANSI_G if zo > 1.0 else (_ANSI_Y if zo > 0 else _ANSI_R)
            zo_str = f"{zo_c}ZΩ {zo:.4f}{_ANSI_RST}"
        else:
            zo_str = f"{_ANSI_DIM}ZΩ —{_ANSI_RST}"
        pid_str = (
            f"{_ANSI_G}▶ PID {pid}{_ANSI_RST}" if alive
            else (f"{_ANSI_R}✗ dead ({pid}){_ANSI_RST}" if pid else f"{_ANSI_DIM}not started{_ANSI_RST}")
        )
        uptime_s = ps.get("uptime_seconds", 0)
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
            q_ok    = ps.get("quote_ok", False)
            t_ok    = ps.get("trade_ok", False)
            healthy = ps.get("connection_healthy", False)
            recon   = ps.get("total_reconnects", 0)
            q_str = f"{_ANSI_G}QUOTE ✓{_ANSI_RST}" if q_ok else f"{_ANSI_R}QUOTE ✗{_ANSI_RST}"
            t_str = f"{_ANSI_G}TRADE ✓{_ANSI_RST}" if t_ok else f"{_ANSI_R}TRADE ✗{_ANSI_RST}"
            h_str = f"{_ANSI_G}healthy{_ANSI_RST}" if healthy else f"{_ANSI_Y}unhealthy{_ANSI_RST}"
            r_col = _ANSI_G if recon == 0 else (_ANSI_Y if recon < 5 else _ANSI_R)
            print(f"    FIX: {q_str}  {t_str}  {h_str}  │  "
                  f"reconnects: {r_col}{recon}{_ANSI_RST}")

            # ── activity ─────────────────────────────────────────────────────
            bars   = ps.get("bar_count", 0)
            trades = ps.get("total_trades", 0)
            pnl    = ps.get("total_pnl", 0.0)
            wr     = ps.get("win_rate", 0.0)
            pnl_c  = _ANSI_G if pnl >= 0 else _ANSI_R
            wr_str = f"{wr * 100:.1f}%" if trades > 0 else "—"
            print(
                f"    Bars: {bars}  │  Trades: {trades}  │  "
                f"PnL: {pnl_c}{pnl:+.2f}{_ANSI_RST}  │  Win: {wr_str}"
            )

            # ── account balance (real from broker if CollateralReport arrived) ─
            _rb = ps.get("real_account_balance")
            _re = ps.get("real_account_equity")
            _rm = ps.get("real_margin_free")
            if _rb is not None:
                _rb_pnl = pnl  # compare relative to starting point
                _rb_c = _ANSI_G if _rb_pnl >= 0 else _ANSI_R
                _re_str = f"  │  Equity: {_ANSI_B}{float(_re):,.2f}{_ANSI_RST}" if _re is not None else ""
                _rm_str = f"  │  Free margin: {_ANSI_B}{float(_rm):,.2f}{_ANSI_RST}" if _rm is not None else ""
                print(
                    f"    Balance: {_rb_c}{float(_rb):,.2f}{_ANSI_RST}  "
                    f"{_ANSI_G}✓ live{_ANSI_RST}"
                    f"{_re_str}{_rm_str}"
                )

            # ── training stats ────────────────────────────────────────────────
            t_steps = ps.get("trigger_steps", 0)
            t_eps   = ps.get("trigger_epsilon", 0.0)
            t_buf   = ps.get("trigger_buffer", 0)
            t_loss  = ps.get("trigger_loss", 0.0)
            t_ready = ps.get("trigger_ready", False)
            h_steps = ps.get("harvester_steps", 0)
            h_beta  = ps.get("harvester_beta", 0.4)
            h_buf   = ps.get("harvester_buffer", 0)
            h_loss  = ps.get("harvester_loss", 0.0)
            h_ready = ps.get("harvester_ready", False)

            t_bar = self._pp_bar(t_buf / _RT_TRIG_CAP if _RT_TRIG_CAP else 0)
            h_bar = self._pp_bar(h_buf / _RT_HARV_CAP if _RT_HARV_CAP else 0)
            t_pct = f"{100 * t_buf / _RT_TRIG_CAP:4.0f}%" if _RT_TRIG_CAP else ""
            h_pct = f"{100 * h_buf / _RT_HARV_CAP:4.0f}%" if _RT_HARV_CAP else ""
            t_rd  = f"{_ANSI_G}ready{_ANSI_RST}" if t_ready else f"{_ANSI_Y}filling{_ANSI_RST}"
            h_rd  = f"{_ANSI_G}ready{_ANSI_RST}" if h_ready else f"{_ANSI_Y}filling{_ANSI_RST}"
            t_ls  = f"{t_loss:.4f}" if t_loss > 0 else f"{_ANSI_DIM}—{_ANSI_RST}"
            h_ls  = f"{h_loss:.4f}" if h_loss > 0 else f"{_ANSI_DIM}—{_ANSI_RST}"
            print(
                f"    Trig:  {t_steps:>6,} steps  "
                f"ε={t_eps:.3f}  buf {t_bar}{t_pct}  loss {t_ls}  {t_rd}"
            )
            print(
                f"    Harv:  {h_steps:>6,} steps  "
                f"β={h_beta:.3f}  buf {h_bar}{h_pct}  loss {h_ls}  {h_rd}"
            )
        else:
            started = entry.get("paper_started_at", "")
            started_str = started[:19].replace("T", " ") if started else "—"
            print(f"    {_ANSI_DIM}Stats not yet available  (started {started_str}){_ANSI_RST}")
        print()

    def _render_live_trigger_agent(
        self, ts: dict, pm: dict, trig_ready: bool, trig_steps: int
    ) -> None:
        """Render the Trigger Agent training block."""
        trig_buf   = ts.get("trigger_buffer_size", 0)
        trig_added = ts.get("trigger_total_added", 0)
        trig_loss  = ts.get("trigger_loss", 0.0)
        trig_eps   = ts.get("trigger_epsilon", 0.0)
        is_in_pos  = ts.get("is_in_position")  # None = unknown (old data)
        # Prefer training_stats confidence (single source of truth); fall back to production_metrics
        trig_conf = ts.get("trigger_confidence", pm.get("trigger_confidence_avg", 0.5))
        ready_t = (
            f"{_ANSI_G}✓ Ready{_ANSI_RST}" if trig_ready
            else f"{_ANSI_Y}⏳ Filling…{_ANSI_RST}"
        )
        # Buffer fill-status annotation: trigger only fills when bot is FLAT
        if is_in_pos is True:
            _trig_buf_note = f"  {_ANSI_Y}⏸ paused — bot in position{_ANSI_RST}"
        elif is_in_pos is False:
            _trig_buf_note = f"  {_ANSI_G}⬆ filling — bot flat{_ANSI_RST}"
        else:
            _trig_buf_note = ""
        print(f"  \033[1m🎯 TRIGGER AGENT  (Entry)\033[0m  {ready_t}  {_ANSI_DIM}fills when flat{_ANSI_RST}")
        print(f"    Steps:  {trig_steps:>10,}   Velocity: {self._rt_velocity(self._trig_step_hist)}")
        print(
            f"    Buffer: {self._rt_pct_bar(trig_buf, _RT_TRIG_CAP)}"
            f"  {trig_buf:,}/{_RT_TRIG_CAP:,}{_trig_buf_note}"
        )
        if trig_added > 0:
            print(f"    Added:  {trig_added:,} total experiences")
        print(f"    ε:      {self._rt_eps_bar(trig_eps)}")
        _tl_str = (
            f"{trig_loss:.6f}" if trig_loss > 0
            else f"{_ANSI_DIM}0.000000 (idle/no training event){_ANSI_RST}"
        )
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
        print()

    def _render_live_harvester_agent(
        self, ts: dict, pm: dict, harv_ready: bool, harv_steps: int
    ) -> None:
        """Render the Harvester Agent training block."""
        harv_buf      = ts.get("harvester_buffer_size", 0)
        harv_added    = ts.get("harvester_total_added", 0)
        harv_loss     = ts.get("harvester_loss", 0.0)
        harv_beta     = ts.get("harvester_beta", 0.4)
        harv_min_hold = ts.get("harvester_min_hold_ticks", 10)
        is_in_pos     = ts.get("is_in_position")
        harv_conf     = ts.get("harvester_confidence", pm.get("harvester_confidence_avg", 0.5))
        ready_h = (
            f"{_ANSI_G}✓ Ready{_ANSI_RST}" if harv_ready
            else f"{_ANSI_Y}⏳ Filling…{_ANSI_RST}"
        )
        # Buffer fill-status annotation: harvester only fills when bot is IN POSITION
        if is_in_pos is True:
            _harv_buf_note = f"  {_ANSI_G}⬆ filling — in position{_ANSI_RST}"
        elif is_in_pos is False:
            _harv_buf_note = f"  {_ANSI_Y}⏸ paused — bot flat{_ANSI_RST}"
        else:
            _harv_buf_note = ""
        print(f"  \033[1m🌾 HARVESTER AGENT  (Exit)\033[0m  {ready_h}  {_ANSI_DIM}fills in position{_ANSI_RST}")
        print(
            f"    Steps:  {harv_steps:>10,}   Velocity: {self._rt_velocity(self._harv_step_hist)}"
        )
        print(
            f"    Buffer: {self._rt_pct_bar(harv_buf, _RT_HARV_CAP)}"
            f"  {harv_buf:,}/{_RT_HARV_CAP:,}{_harv_buf_note}"
        )
        if harv_added > 0:
            print(f"    Added:  {harv_added:,} total experiences")
        print(f"    β IS:   {self._rt_beta_bar(harv_beta)}")
        _hl_str = (
            f"{harv_loss:.6f}" if harv_loss > 0
            else f"{_ANSI_DIM}0.000000 (idle/no training event){_ANSI_RST}"
        )
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
        print(f"    Hold:   {harv_min_hold} ticks min")
        print()

    def _render_live_arena_and_health(
        self,
        ts: dict,
        trig_ready: bool,
        harv_ready: bool,
        trig_steps: int,
        harv_steps: int,
    ) -> None:
        """Render Arena + Learning Health blocks."""
        total_agents = ts.get("total_agents", 0)
        if total_agents > 0:
            diversity = ts.get("arena_diversity", {})
            trig_div  = diversity.get("trigger_diversity", 0) if isinstance(diversity, dict) else 0
            harv_div  = diversity.get("harvester_diversity", 0) if isinstance(diversity, dict) else 0
            agreement = ts.get("last_agreement_score", 0)
            consensus = ts.get("consensus_mode", "unknown")
            print("  \033[1m🤖 ARENA\033[0m")
            print(f"    Agents: {total_agents}   Consensus: {consensus}   Agreement: {agreement:.3f}")
            print(f"    Diversity — Trig: {trig_div:.3f}   Harv: {harv_div:.3f}")
            print()
        last_train = ts.get("last_training_time", "Never")
        train_on   = self.bot_config.get("training_enabled", False)
        train_str  = f"{_ANSI_G}ON{_ANSI_RST}" if train_on else f"{_ANSI_R}OFF{_ANSI_RST}"
        trig_ok = f"{_ANSI_G}✓{_ANSI_RST}" if trig_ready else f"{_ANSI_Y}⏳{_ANSI_RST}"
        harv_ok = f"{_ANSI_G}✓{_ANSI_RST}" if harv_ready else f"{_ANSI_Y}⏳{_ANSI_RST}"
        print("  \033[1m📊 LEARNING HEALTH\033[0m")
        print(f"    Training: {train_str}   Last event: {last_train}")
        print(
            f"    Trigger {trig_ok}  Harvester {harv_ok}"
            f"   Total steps: {trig_steps + harv_steps:,}"
        )
        print()


    def _render_h4_shadow_training(self) -> None:
        """Render H4 shadow DualPolicy training block (Training tab)."""
        ts = self.training_stats_h4
        tr = ts.get("trigger", {})
        hr = ts.get("harvester", {})

        t_steps = tr.get("training_steps", 0)
        t_buf   = tr.get("buffer_size", 0)
        t_eps   = tr.get("epsilon", 0.0)
        t_loss  = tr.get("loss", 0.0) if "loss" in tr else ts.get("trigger_loss", 0.0)
        t_ready = tr.get("ready_to_train", False)

        h_steps = hr.get("training_steps", 0)
        h_buf   = hr.get("buffer_size", 0)
        h_beta  = hr.get("beta", 0.4)
        h_loss  = hr.get("loss", 0.0) if "loss" in hr else ts.get("harvester_loss", 0.0)
        h_ready = hr.get("ready_to_train", False)

        shadow_pos = ts.get("h4_shadow_pos", 0)
        h4_bars    = ts.get("h4_bars", 0)
        pos_str    = {1: f"{_ANSI_G}LONG{_ANSI_RST}", -1: f"{_ANSI_R}SHORT{_ANSI_RST}"}.get(shadow_pos, f"{_ANSI_DIM}FLAT{_ANSI_RST}")

        W = self._term_width()
        print("  " + "─" * (W - 4))
        print("  \033[1m📡 H4 SHADOW TRAINING  (M240 · no real orders)\033[0m")
        print(f"    Shadow position: {pos_str}   Bars accumulated: {h4_bars}")
        print()

        ready_t = f"{_ANSI_G}✓ Ready{_ANSI_RST}" if t_ready else f"{_ANSI_Y}⏳ Filling…{_ANSI_RST}"
        print(f"  \033[1m🎯 TRIGGER AGENT  (H4 Entry)\033[0m  {ready_t}")
        print(f"    Steps:  {t_steps:>10,}")
        print(f"    Buffer: {self._rt_pct_bar(t_buf, _RT_TRIG_CAP)}  {t_buf:,}/{_RT_TRIG_CAP:,}")
        print(f"    ε:      {self._rt_eps_bar(t_eps)}")
        _tl = f"{t_loss:.6f}" if t_loss > 0 else f"{_ANSI_DIM}0.000000 (idle){_ANSI_RST}"
        print(f"    Loss:   {_tl}")
        print()

        ready_h = f"{_ANSI_G}✓ Ready{_ANSI_RST}" if h_ready else f"{_ANSI_Y}⏳ Filling…{_ANSI_RST}"
        print(f"  \033[1m🌾 HARVESTER AGENT  (H4 Exit)\033[0m  {ready_h}")
        print(f"    Steps:  {h_steps:>10,}")
        print(f"    Buffer: {self._rt_pct_bar(h_buf, _RT_HARV_CAP)}  {h_buf:,}/{_RT_HARV_CAP:,}")
        print(f"    β IS:   {self._rt_beta_bar(h_beta)}")
        _hl = f"{h_loss:.6f}" if h_loss > 0 else f"{_ANSI_DIM}0.000000 (idle){_ANSI_RST}"
        print(f"    Loss:   {_hl}")
        print("    ℹ️  H4 bar closes every ~4 h — next training on H4 bar close")
        print()

    def _render_training(self):
        """Render agent training status."""
        ts = self.training_stats
        pm = self.production_metrics.get("metrics", {})
        ofs = self.offline_stats
        if ofs:
            # Auto-prune completed offline training older than 24 h
            _ofs_stale = False
            if ofs.get("status") == "complete" and ofs.get("completed_at"):
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
                try:
                    (self.data_dir / "offline_training_status.json").unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                self._render_offline_training(ofs)
        if self.universe_stats:
            self._render_paper_pipeline()
        _ts_nonempty = any(v for v in ts.values() if v)
        _mode = self.bot_config.get("trading_mode", "paper")
        _mode_label = "PAPER" if _mode == "paper" else ("LIVE" if _mode == "live" else "OFFLINE")
        if not _ts_nonempty:
            print(f"\033[1m🤖 {_mode_label} BOT TRAINING\033[0m  {_ANSI_DIM}(no live bot running){_ANSI_RST}\n")
            return
        _has_h4 = bool(self.training_stats_h4)
        _m5_label = f"{_mode_label} M5 TRAINING" if _has_h4 else f"{_mode_label} BOT TRAINING"
        print(f"\033[1m🤖 {_m5_label}\033[0m\n")
        trig_ready = ts.get("trigger_ready", False)
        harv_ready = ts.get("harvester_ready", False)
        trig_steps = ts.get("trigger_training_steps", 0)
        harv_steps = ts.get("harvester_training_steps", 0)
        self._render_live_trigger_agent(ts, pm, trig_ready, trig_steps)
        self._render_live_harvester_agent(ts, pm, harv_ready, harv_steps)
        self._render_live_arena_and_health(ts, trig_ready, harv_ready, trig_steps, harv_steps)
        if _has_h4:
            self._render_h4_shadow_training()

        # Next-update hint — training stats only refresh on bar close
        _nbc_raw = self.market_stats.get("next_bar_close_utc")
        if not _nbc_raw:
            _nbc_raw = self.bot_config.get("next_bar_close_utc")
        _tf_min = (
            self.market_stats.get("timeframe_minutes")
            or self.bot_config.get("timeframe_minutes")
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
                print(
                    f"  {_ANSI_DIM}ℹ️  Training stats update on bar close — "
                    f"next in {_hint}{_ANSI_RST}"
                )
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
            print(
                f"  {_ANSI_DIM}ℹ️  Training stats update every {_lbl} bar close "
                f"(awaiting first tick){_ANSI_RST}"
            )

    def _render_header(self):
        """Render header"""
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
            alert = "⚡ KURTOSIS GATE ACTIVE — entries bypassed in paper mode"
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
            self._load_paper_bot_stats(self.active_sym, self.active_tf_min)
            if self.active_sym and self.active_tf_min else {}
        )
        symbol = _aps.get("symbol") or self.bot_config.get("symbol", "UNKNOWN")
        _tf_min = _aps.get("timeframe_minutes") or self.bot_config.get("timeframe_minutes")
        tf = f"{_tf_min}m" if _tf_min else self.bot_config.get("timeframe", "1m")
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

        print(f"\n🎯 {symbol} @ {tf}  {_mode_badge}    💰 {price_str}    ⏱  {hours:02d}h {minutes:02d}m{_nbc_str}")
        print(f"{heartbeat} {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    def _render_tab_bar(self):
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
        for key, tab_id in self.TABS.items():
            name = labels.get(tab_id, tab_id.title())
            if tab_id == self.current_tab:
                tabs.append(f"\033[7m [{key}] {name} \033[0m")  # Inverted
            else:
                tabs.append(f" [{key}] {name} ")

        print("".join(tabs))
        print("─" * W)

    def _render_position_block(self) -> None:
        """Render the position header block (always fixed height to avoid layout jumps)."""
        _mode = self.bot_config.get("trading_mode", "paper")
        _mode_tag = f"  {_ANSI_Y}(paper){_ANSI_RST}" if _mode == "paper" else (
            f"  {_ANSI_G}(live){_ANSI_RST}" if _mode == "live" else ""
        )
        print(f"\n\033[1m📊 POSITION\033[0m{_mode_tag}")
        direction = self.position.get("direction", "FLAT")
        entry = self.position.get("entry_price", 0)
        current = self.position.get("current_price", 0)
        pnl = self.position.get("unrealized_pnl", 0)
        bars = self.position.get("bars_held", 0)
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
                f"PnL: {pnl_color}{pnl:+.2f}{_ANSI_RST}  |  Bars: {bars}"
            )
        # Line 2: MFE/MAE (always printed — blank spacer when FLAT for stable layout)
        if direction != "FLAT":
            mfe = self.position.get("mfe", 0.0)
            mae = self.position.get("mae", 0.0)
            mfe_color = _ANSI_G if mfe > 0 else _ANSI_Y
            mae_color = _ANSI_R if mae > 0 else _ANSI_Y
            print(f"  MFE: {mfe_color}+{mfe:.2f}{_ANSI_RST}  |  MAE: {mae_color}-{mae:.2f}{_ANSI_RST}  (USD, excl. spread)")
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
        print(f"\n\033[1m🤖 ALL BOTS\033[0m  {_ANSI_DIM}(Trd/PnL/Win% = session counters, reset on restart){_ANSI_RST}")
        _hdr = ("  " + "Bot".ljust(13) + "  Status   " + "Bars".rjust(4)
                + "  Position               "
                + "  T-buf  H-buf  Trd    PnL     Win%")
        print(f"\033[2m{_hdr}\033[0m")
        print("  " + "─" * 82)
        _now = datetime.now(UTC)
        for bot in bots:
            sym = bot.get("symbol", "?")
            tf  = bot.get("timeframe_minutes", 0)
            label = f"{sym}/M{tf}"
            # Freshness
            try:
                _age = (_now - datetime.fromisoformat(bot.get("updated_at", ""))).total_seconds()
            except Exception:
                _age = 9999.0
            conn = bot.get("connection_healthy", False) and bot.get("quote_ok", False)
            if not conn or _age > 180:
                status = f"{_ANSI_R}● STALE{_ANSI_RST}"
            elif _age < 70:
                status = f"{_ANSI_G}● LIVE {_ANSI_RST}"
            else:
                status = f"{_ANSI_Y}● SLOW {_ANSI_RST}"
            bars = bot.get("bar_count", 0)
            # Position — build visible and colored strings separately to keep columns aligned
            pos = bot.get("_position", {})
            direction = (pos.get("direction") or "FLAT").upper()
            entry_px = pos.get("entry_price", 0.0)
            unreal   = pos.get("unrealized_pnl", 0.0)
            if direction != "FLAT":
                visible_pos = f"{direction:<5}@{entry_px:.0f}({unreal:+.0f})"
                dir_c = _ANSI_G if direction == "LONG" else _ANSI_R
                colored_pos = (f"{dir_c}{direction:<5}{_ANSI_RST}"
                               f"@{entry_px:.0f}"
                               f"({self._pnl_color(unreal)}{unreal:+.0f}{_ANSI_RST})")
            else:
                visible_pos = "FLAT"
                colored_pos = f"{_ANSI_DIM}FLAT{_ANSI_RST}"
            pos_pad = " " * max(0, 22 - len(visible_pos))
            trig_buf = bot.get("trigger_buffer", 0)
            harv_buf = bot.get("harvester_buffer", 0)
            trades   = bot.get("total_trades", 0)
            pnl      = bot.get("total_pnl", 0.0)
            wr       = bot.get("win_rate", 0.0) * 100
            wr_str   = f"{wr:.0f}%" if trades > 0 else "  -"
            print(
                f"  {label:<13}  {status}  {bars:>4}  {colored_pos}{pos_pad}  "
                f"{trig_buf:>5}  {harv_buf:>5}  {trades:>3}  "
                f"{self._pnl_color(pnl)}{pnl:>+8.2f}{_ANSI_RST}  {wr_str:>4}"
            )

    def _render_overview(self):
        """Render overview tab - compact summary"""
        self._render_all_bots_panel()
        self._render_position_block()

        # Account balance / equity
        _mode = self.bot_config.get("trading_mode", "paper")
        _acct_tag = f"  {_ANSI_Y}(paper){_ANSI_RST}" if _mode == "paper" else (
            f"  {_ANSI_G}(live){_ANSI_RST}" if _mode == "live" else ""
        )
        print(f"\n\033[1m💰 ACCOUNT\033[0m{_acct_tag}")
        # Prefer starting_equity from universe.json for the active symbol.
        # bot_config.json is shared across bots; the last writer may reflect a
        # different instrument's equity baseline.
        _ue = self.universe_stats.get(
            self.active_sym or self.bot_config.get("symbol", ""), {}
        )
        _starting = float(
            _ue.get("starting_equity")
            or self.bot_config.get("starting_equity", 10_000.0)
        )
        _lifetime_pnl = float(self.lifetime_metrics.get("total_pnl", 0.0))
        _unreal = float(self.position.get("unrealized_pnl", 0.0))
        # Prefer real broker values (from CollateralReport BA) when available
        _real_bal = self.bot_config.get("real_account_balance")
        _real_eq  = self.bot_config.get("real_account_equity")
        _real_mfr = self.bot_config.get("real_margin_free")
        if _real_bal is not None:
            _balance = float(_real_bal)
            _live_tag = "  \033[32m✓ live\033[0m"
        else:
            _balance = _starting + _lifetime_pnl
            _live_tag = "  \033[33m~ est.\033[0m"
        if _real_eq is not None:
            _equity = float(_real_eq)
        else:
            _equity = _balance + _unreal
        _margin_str = (
            f"  |  Free margin: \033[36m{float(_real_mfr):>10.2f}\033[0m"
            if _real_mfr is not None
            else ""
        )
        _direction = (self.position.get("direction") or "FLAT").upper()
        if _direction == "FLAT":
            _unreal_str = f"{_ANSI_DIM}—{_ANSI_RST}"
        else:
            _unreal_str = f"{self._pnl_color(_unreal)}{_unreal:+.2f}\033[0m"
        print(
            f"  Balance: {self._pnl_color(_balance - _starting)}{_balance:>10.2f}\033[0m{_live_tag}  |  "
            f"Equity:  {self._pnl_color(_equity  - _starting)}{_equity:>10.2f}\033[0m  |  "
            f"Unrealized: {_unreal_str}"
            f"{_margin_str}"
        )

        # Quick metrics
        print("\n\033[1m📈 TODAY'S STATS\033[0m")
        d = self.daily_metrics
        trades = d.get("total_trades", 0)
        wr = d.get("win_rate", 0) * 100
        day_pnl = d.get("total_pnl", 0)
        print(f"  Trades: {trades}  |  Win Rate: {wr:.1f}%  |  PnL: {self._pnl_color(day_pnl)}{day_pnl:+.2f}\033[0m")

        # Add sparkline for recent performance if available
        recent_pnl = d.get("recent_pnl_sequence", [])
        if recent_pnl and len(recent_pnl) > 1:
            sparkline = self._create_sparkline(recent_pnl[-20:])  # Last 20 trades
            print(f"  Recent: {sparkline}")

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
        print("\n\033[1m🔬 MARKET\033[0m")
        spread = self.market_stats.get("spread", 0)
        vpin = self.market_stats.get("vpin", 0)
        vpin_z = self.market_stats.get("vpin_z", 0)
        imb = self.market_stats.get("imbalance", 0)

        vpin_status = (
            f"{_ANSI_R}⚠️ HIGH{_ANSI_RST}"
            if abs(vpin_z) > VPIN_HIGH_TOXICITY_THRESHOLD
            else f"{_ANSI_G}✓{_ANSI_RST}"
        )
        _has_real = self.market_stats.get("has_real_sizes", False)
        _imb_label = "Imb" if _has_real else "QFI"
        _sp_bps = self._spread_bps()
        _sp_col = self._spread_color()
        print(f"  Spread: {_sp_col}{spread:.5f} ({_sp_bps:.1f}bp){_ANSI_RST}  |  VPIN: {vpin:.3f} (z={vpin_z:+.1f}) {vpin_status}  |  {_imb_label}: {imb:+.3f}")

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
        print("\n\033[1m🏥 SYSTEM HEALTH\033[0m")
        self._render_health_connectivity()
        self._render_health_risk()
        self._render_health_buffers()
        self._render_health_model()
        self._render_health_microstructure()
        self._render_health_system_metrics()
        self._render_health_self_test()

    def _render_health_connectivity(self) -> None:
        """Render data freshness and breaker status row."""
        def _ok(s: str) -> str:
            return f"{_ANSI_G}✓ {s}{_ANSI_RST}"

        def _warn(s: str) -> str:
            return f"{_ANSI_Y}⚡ {s}{_ANSI_RST}"

        def _bad(s: str) -> str:
            return f"{_ANSI_R}✗ {s}{_ANSI_RST}"

        _ob_path = self.data_dir / _ORDER_BOOK_FILE
        _bc_path = self.data_dir / _BOT_CONFIG_FILE
        # Prefer the freshest per-bot paper_stats_*.json — written every HUD cycle.
        # Fall back to order_book.json then bot_config.json for legacy setups.
        _ps_paths = sorted(
            self.data_dir.glob("paper_stats_*.json"),
            key=lambda _p: _p.stat().st_mtime if _p.exists() else 0,
            reverse=True,
        )
        if _ps_paths:
            _ref_path = _ps_paths[0]
        elif _ob_path.exists():
            _ref_path = _ob_path
        elif _bc_path.exists():
            _ref_path = _bc_path
        else:
            _ref_path = None
        if _ref_path is not None:
            _file_age = time.time() - os.path.getmtime(_ref_path)
            _age_str = f"{_file_age:.0f}s"
            if _file_age < DATA_AGING_SECS:
                _data_item = _ok(f"Data {_age_str}")
            elif _file_age < DATA_STALE_SECS:
                _data_item = _warn(f"Data {_age_str}")
            else:
                _data_item = _bad(f"Bot silent {_age_str}")
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
        )
        print(
            f"  β={_beta_col}{_beta:.4f} {_beta_lbl}{_ANSI_RST}  steps={_harv_steps:,}  loss={_loss_str(_harv_loss)}"
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
        elif bps < SPREAD_WARN_BPS:
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
        src = f"  {_ANSI_DIM}(source: trade_log.jsonl){_ANSI_RST}" if self._metrics_from_trade_log else ""
        print(f"\n\033[1m📈 PERFORMANCE METRICS\033[0m{_mode_tag}{src}\n")

        # Column headers — 'TQR' = Trade Quality Ratio (mean/σ of trade PnL in USD).
        # This is NOT an annualised return-based Sharpe ratio.
        print(
            f"  {'Period':<9} {'Trades':>7} {'Win%':>7} {'PnL':>11} {'TQR':>7} {'PF':>7} {'MaxDD%':>8}"
        )
        print("  " + "─" * 62)

        for label, metrics in [
            ("24h",    self.daily_metrics),
            ("7 days", self.weekly_metrics),
            ("Month",  self.monthly_metrics),
            ("All",    self.lifetime_metrics),
        ]:
            trades = metrics.get("total_trades", 0)
            wr = metrics.get("win_rate", 0) * 100
            pnl = metrics.get("total_pnl", 0)
            sharpe = metrics.get("sharpe_ratio", 0)
            pf = metrics.get("profit_factor", metrics.get("omega_ratio", 0))  # profit factor (capped at 99)
            maxdd = metrics.get("max_drawdown", 0.0)  # already a % of peak equity

            pnl_color = self._pnl_color(pnl)
            if maxdd > DD_HIGH_PCT:
                dd_color = _ANSI_R
            elif maxdd > DD_WARN_PCT:
                dd_color = _ANSI_Y
            else:
                dd_color = _ANSI_G

            print(
                f"  {label:<9} {trades:>7} {wr:>6.1f}% {pnl_color}{pnl:>+10.2f}{_ANSI_RST} "
                f"{sharpe:>7.2f} {pf:>7.2f} {dd_color}{maxdd:>7.2f}%{_ANSI_RST}"
            )

        # Per-symbol breakdown (only when multiple symbols exist)
        if len(self.per_symbol_metrics) > 1:
            print("\n  \033[1mPER SYMBOL\033[0m")
            print(
                f"  {'Symbol':<10} {'Trades':>7} {'Win%':>7} {'PnL':>11} {'PF':>7} {'MaxDD%':>8}"
            )
            print("  " + "─" * 55)
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
                    f"  {_sym:<10} {_tr:>7} {_wr:>6.1f}% {_pc}{_pnl:>+10.2f}{_ANSI_RST} "
                    f"{_pf:>7.2f} {_dc}{_mdd:>7.2f}%{_ANSI_RST}"
                )

        self._render_mode_breakdown()

        self._render_trade_quality(self.lifetime_metrics)

        pm = self.production_metrics.get("metrics", {})
        # Always render timing and prediction convergence; use trade_log-derived
        # lifetime_metrics for authoritative timing, pm for runtime convergence stats.
        self._render_trade_timing(self.lifetime_metrics, pm)

    def _render_mode_breakdown(self) -> None:
        """Show paper vs live trade breakdown when trade_log.jsonl has both modes."""
        trade_file = Path("data/trade_log.jsonl")
        if not trade_file.exists():
            return
        try:
            paper_trades: list[dict] = []
            live_trades: list[dict] = []
            with open(trade_file, encoding="utf-8") as f:
                for raw_line in f:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    t = json.loads(stripped)
                    mode = t.get("trading_mode", "")
                    if mode == "paper":
                        paper_trades.append(t)
                    elif mode == "live":
                        live_trades.append(t)
        except Exception:
            return

        # Only show breakdown when there is something to display
        if not paper_trades and not live_trades:
            return

        print("\n  \033[1mMODE BREAKDOWN\033[0m")
        print(
            f"  {'Mode':<8} {'Trades':>7} {'Win%':>7} {'PnL':>11}"
        )
        print("  " + "\u2500" * 37)
        for label, trades, color in [
            ("\U0001f4c4 Paper", paper_trades, _ANSI_Y),
            ("\U0001f4b0 Live", live_trades, _ANSI_G),
        ]:
            n = len(trades)
            if n == 0:
                continue
            pnls = [t.get("pnl", 0.0) for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / n * 100
            total_pnl = sum(pnls)
            pnl_c = self._pnl_color(total_pnl)
            print(
                f"  {color}{label:<8}{_ANSI_RST} {n:>7} {wr:>6.1f}% "
                f"{pnl_c}{total_pnl:>+10.2f}{_ANSI_RST}"
            )
        print("  " + "\u2500" * 37)

    def _render_trade_quality(self, lt: dict) -> None:
        """Render trade quality metrics block."""
        print("\n\033[1m📊 TRADE QUALITY\033[0m\n")

        avg_win = lt.get("avg_win", 0.0)
        avg_loss = lt.get("avg_loss", 0.0)   # stored as negative
        profit_f = lt.get("profit_factor", 0.0)
        expect = lt.get("expectancy", 0.0)
        sortino = lt.get("sortino_ratio", 0.0)
        best = lt.get("best_trade", 0.0)
        worst = lt.get("worst_trade", 0.0)

        abs_loss = abs(avg_loss)
        payoff = avg_win / abs_loss if abs_loss > _PAYOFF_FLOOR else 0.0
        if payoff >= PAYOFF_GOOD_MIN:
            pay_col = _ANSI_G
        elif payoff >= 1.0:
            pay_col = _ANSI_Y
        else:
            pay_col = _ANSI_R
        exp_col = _ANSI_G if expect > 0 else _ANSI_R
        if profit_f >= PROFIT_FACTOR_GOOD_MIN:
            pf_col = _ANSI_G
        elif profit_f >= 1.0:
            pf_col = _ANSI_Y
        else:
            pf_col = _ANSI_R

        print(
            f"  Payoff ratio:     {pay_col}{payoff:>7.2f}x{_ANSI_RST}  "
            f"{_ANSI_DIM}(avg_win/|avg_loss|  target ≥1.5){_ANSI_RST}"
        )
        print(f"  Avg W / Avg L:    {avg_win:>+8.2f} / {avg_loss:>+8.2f}")
        print(
            f"  Profit factor:    {pf_col}{profit_f:>7.2f} {_ANSI_RST} "
            f"{_ANSI_DIM}(gross_profit/gross_loss  target ≥1.2){_ANSI_RST}"
        )
        print(f"  Expectancy/trade: {exp_col}{expect:>+8.4f}{_ANSI_RST}")
        print(
            f"  Sortino ratio:    {sortino:>7.3f}  "
            f"{_ANSI_DIM}(mean/downside-\u03c3 of trade PnL; losses-only denom){_ANSI_RST}"
        )
        print(f"  Best / Worst:     {best:>+8.2f} / {worst:>+8.2f}")
        max_cw = lt.get("max_consec_wins", 0)
        max_cl = lt.get("max_consec_losses", 0)
        cw_col = _ANSI_G if max_cw >= 3 else _ANSI_Y
        cl_col = _ANSI_R if max_cl >= 5 else (_ANSI_Y if max_cl >= 3 else _ANSI_G)
        print(
            f"  Consec W / L:     {cw_col}{max_cw:>4} wins{_ANSI_RST} / "
            f"{cl_col}{max_cl:>4} losses{_ANSI_RST}"
        )
        # Winner-to-loser: trades where MFE exceeded entry but reversed to a loss
        w2l = lt.get("winner_to_loser_count", 0)
        total = lt.get("total_trades", 0)
        if total > 0 and w2l > 0:
            w2l_pct = w2l / total * 100
            w2l_col = _ANSI_R if w2l_pct > 15 else (_ANSI_Y if w2l_pct > 5 else _ANSI_G)
            print(
                f"  Winner→Loser:     {w2l_col}{w2l:>4} ({w2l_pct:.1f}%){_ANSI_RST}  "
                f"{_ANSI_DIM}(had profit but reversed to loss){_ANSI_RST}"
            )

    def _render_trade_timing(self, lt: dict, pm: dict) -> None:
        """Render trade timing (from trade_log) and prediction convergence (from runtime)."""
        # avg_trade_duration and last_trade are added by _compute_metrics_from_trade_log;
        # fall back to production_metrics for legacy / warm-up period.
        avg_dur = lt.get("avg_trade_duration_mins") or pm.get("avg_trade_duration_mins", 0.0)
        last_trade = lt.get("last_trade_mins_ago") or pm.get("last_trade_mins_ago", 0.0)
        print(f"\n  Avg hold time:    {self._format_duration(avg_dur):>8}")
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

    def _render_prediction_convergence(self, pm: dict) -> None:
        """Render prediction convergence metrics from production stats."""
        rw_delta = pm.get("runway_delta_ema", 0.0)
        rw_acc = pm.get("runway_accuracy_ema", 0.5)
        cc_err = pm.get("conf_calib_err_ema", 0.5)
        platt_a = pm.get("platt_a", 1.0)
        platt_b = pm.get("platt_b", 0.0)

        if abs(rw_delta) < 1.0:
            delta_col = _ANSI_G
        elif abs(rw_delta) < RUNWAY_DELTA_WARN_MAX:
            delta_col = _ANSI_Y
        else:
            delta_col = _ANSI_R
        if rw_acc > RUNWAY_ACCURACY_GOOD:
            acc_col = _ANSI_G
        elif rw_acc > RUNWAY_ACCURACY_WARN:
            acc_col = _ANSI_Y
        else:
            acc_col = _ANSI_R
        if cc_err < CONF_CALIB_OK_MAX:
            cc_col = _ANSI_G
        elif cc_err < CONF_CALIB_WARN_MAX:
            cc_col = _ANSI_Y
        else:
            cc_col = _ANSI_R
        if abs(platt_a - 1.0) > PLATT_ADAPTED_DELTA or abs(platt_b) > PLATT_ADAPTED_DELTA:
            pa_col = _ANSI_B
        else:
            pa_col = _ANSI_DIM

        print(
            f"\n\033[1m🎯 PREDICTION CONVERGENCE\033[0m  {_ANSI_DIM}(EMA-10 trades){_ANSI_RST}\n"
        )
        print(
            f"  Runway Δ (pred−actual):   "
            f"{delta_col}{rw_delta:>+7.2f} pts{_ANSI_RST}  "
            f"{_ANSI_DIM}→ 0 = perfect{_ANSI_RST}"
        )
        print(
            f"  Runway Accuracy:          "
            f"{acc_col}{rw_acc:>7.3f}{_ANSI_RST}      "
            f"{_ANSI_DIM}→ 1 = perfect{_ANSI_RST}"
        )
        print(
            f"  Conf Calibration Error:   "
            f"{cc_col}{cc_err:>7.3f}{_ANSI_RST}      "
            f"{_ANSI_DIM}→ 0 = calibrated{_ANSI_RST}"
        )
        print(
            f"  Platt  a={pa_col}{platt_a:.4f}{_ANSI_RST}  "
            f"b={pa_col}{platt_b:+.4f}{_ANSI_RST}  "
            f"{_ANSI_DIM}(grey=default, blue=adapted){_ANSI_RST}"
        )

    def _render_jsonl_decision_entries(self, entries: list) -> None:
        """Render the rich JSONL decision log entries.

        Columns: Date+Time | Mode | Agent | Decision | Conf | Rnwy$ | VPIN-z | Price | TradeID
        Session breaks are inserted when the session_id changes.
        """
        header = (
            f"  {'Date/Time':<12} {'Mode':<6} {'Agent':<10} {'Decision':<14} "
            f"{'Conf':>6} {'Rnwy$':>6} {'VPIN-z':>7} {'Price':>10}  {'TrdID':<9}"
        )
        _counts: dict[str, int] = {}
        for _e in entries:
            _d = _e.get("decision", "?").upper()
            _counts[_d] = _counts.get(_d, 0) + 1
        _dist = "  ".join(f"{k}:{v}" for k, v in sorted(_counts.items()))
        print(f"  Distribution (last {len(entries)}): {_dist}\n")
        print(header)
        print("  " + "─" * 80)
        _prev_session: str | None = None
        for entry in entries:
            # ── Session break header ───────────────────────────────────────
            _sess = entry.get("session", "")
            if _sess and _sess != _prev_session:
                _prev_session = _sess
                print(f"  {_ANSI_DIM}── session {_sess} ──{_ANSI_RST}")

            # ── Timestamp: MM-DD HH:MM (date preserved across multi-day logs) ──
            ts_raw = entry.get("timestamp", "?")
            try:
                # ISO 2026-03-08T15:30:00+00:00 → slice [5:16] = 03-08 15:30
                ts_str = ts_raw[5:16] if len(ts_raw) >= _DEC_LOG_TS_MIN_LEN else ts_raw[:11]
            except Exception:
                ts_str = str(ts_raw)[:11]

            # ── Mode badge ─────────────────────────────────────────────────
            _mode = entry.get("trading_mode", "")
            if _mode == "paper":
                mode_str = f"{_ANSI_Y}📄PPR{_ANSI_RST}"
            elif _mode == "live":
                mode_str = f"{_ANSI_G}💰LIV{_ANSI_RST}"
            else:
                mode_str = f"{_ANSI_DIM}  ?  {_ANSI_RST}"

            agent = entry.get("agent", "?")[:9]
            decision = entry.get("decision", "?")[:13]
            conf = entry.get("confidence", 0.0)
            ctx = entry.get("context", {})
            reasoning = entry.get("reasoning", {})
            price = ctx.get("price", 0.0)
            vpin_z = ctx.get("vpin_z", 0.0)
            runway_pct = reasoning.get("predicted_runway", 0.0)
            runway_usd = runway_pct * price if price > 0 else runway_pct

            # ── Trade correlation ID (links entry→HOLDs→close for one trade) ──
            trade_id = entry.get("trade_id", "")
            tid_str = trade_id[:8] if trade_id else f"{_ANSI_DIM}--------{_ANSI_RST}"

            if decision.upper() in ("BUY", "LONG", "ENTER"):
                color = _ANSI_G
            elif decision.upper() in ("SELL", "SHORT", "EXIT", "CLOSE", "CLOSE_PENDING"):
                color = _ANSI_R
            elif decision.upper() == "HOLD":
                color = _ANSI_Y
            else:
                color = _ANSI_RST
            vpin_flag = (
                f"{_ANSI_R}⚠{_ANSI_RST}"
                if abs(vpin_z) > _DEC_LOG_VPIN_WARN
                else " "
            )
            dec = self._price_decimals(price)
            print(
                f"  {ts_str:<12} {mode_str} {agent:<10} "
                f"{color}{decision:<14}{_ANSI_RST} "
                f"{conf:>6.3f} {runway_usd:>6.2f} {vpin_z:>+6.2f}{vpin_flag} "
                f"{price:>10.{dec}f}  {tid_str}"
            )
        print("  " + "─" * 80)

    def _render_legacy_decision_entries(self, entries: list) -> None:
        """Render legacy JSON-format decision log entries."""
        print(f"  Showing {len(entries[-20:])} most recent decisions (legacy format):\n")
        print("  " + "─" * 76)
        for entry in entries[-20:]:
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
            if isinstance(details, dict):
                pos = details.get("cur_pos", "?")
                action = details.get("action", "?")
                conf = details.get("confidence", "?")
                # Real schema keys: exit_action / exit_conf — no bare "pnl" field
                exit_conf = details.get("exit_conf")
                exit_act  = details.get("exit_action")
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
            print(f"  [{ts}] {mode_badge} {color}{event}{_ANSI_RST}: {details_str}")
        print("  " + "─" * 76)
        print(f"\n  Total decisions logged: {len(entries)}")

    def _render_decision_log(self):
        """Render the Decision Log tab (Tab 6)"""
        print("\n\033[1m📝 DECISION LOG\033[0m (last 20 entries)\n")

        jsonl_file = Path("logs/audit/decisions.jsonl")
        entries_jsonl: list[dict] = []
        if jsonl_file.exists():
            try:
                with open(jsonl_file, encoding="utf-8") as f:
                    lines = f.readlines()
                for raw_line in lines[-20:]:
                    stripped = raw_line.strip()
                    if stripped:
                        entries_jsonl.append(json.loads(stripped))
            except Exception:
                entries_jsonl = []

        if entries_jsonl:
            self._render_jsonl_decision_entries(entries_jsonl)
            return

        log_file = self.data_dir / "decision_log.json"
        if not log_file.exists():
            print("  ⚠️  No decision log found.")
            print("\n  Expected files:")
            print("    logs/audit/decisions.jsonl  (rich — primary)")
            print("    data/decision_log.json  (legacy — fallback)")
            return

        try:
            raw_text = log_file.read_text(encoding="utf-8")
            # decision_log.json can end up as multiple appended JSON arrays
            # (e.g. [...][\n...]) due to successive bot restarts writing the
            # same file.  Walk through all top-level objects/arrays to collect
            # every entry regardless of how many times the file was rewritten.
            _dec = json.JSONDecoder()
            _pos = 0
            entries: list = []
            while _pos < len(raw_text):
                _stripped = raw_text[_pos:].lstrip()
                if not _stripped:
                    break
                _skip = len(raw_text[_pos:]) - len(_stripped)
                try:
                    _obj, _idx = _dec.raw_decode(raw_text, _pos + _skip)
                    _pos = _pos + _skip + _idx
                    if isinstance(_obj, list):
                        entries.extend(_obj)
                    elif isinstance(_obj, dict):
                        entries.append(_obj)
                except json.JSONDecodeError:
                    break
        except Exception as e:
            print(f"  ❌ Error reading decision log: {e}")
            return

        if not entries:
            print("  No entries yet. Waiting for bot decisions...")
            return

        self._render_legacy_decision_entries(entries)

    def _render_risk(self):
        """Render risk management details."""
        print("\n\033[1m⚠️  RISK MANAGEMENT\033[0m\n")
        rs = self.risk_stats
        self._render_risk_circuit_breaker(rs)
        self._render_risk_tail(rs)
        self._render_risk_regime(rs)
        self._render_risk_path_geometry(rs)
        self._render_risk_position_sizing(rs)

    def _render_risk_circuit_breaker(self, rs: dict) -> None:
        """Render circuit breaker status block with individual breaker details."""
        cb = rs.get("circuit_breaker", "INACTIVE")
        kurt_gate = rs.get("kurtosis_gate_active", False)
        if cb == "ACTIVE":
            print(f"  {_ANSI_R}╔════════════════════════════════════════╗")
            print("  ║     ⚠️  CIRCUIT BREAKER ACTIVE ⚠️       ║")
            print(f"  ╚════════════════════════════════════════╝{_ANSI_RST}")
            print(f"  {_ANSI_Y}Press [r] to review and reset circuit breakers{_ANSI_RST}\n")
        elif kurt_gate:
            print(
                f"  {_ANSI_Y}⚡ Kurtosis gate: ACTIVE "
                f"(κ={rs.get('kurtosis', 0):.1f} excess > 3.0){_ANSI_RST}  "
                f"{_ANSI_DIM}bypassed in paper mode{_ANSI_RST}\n"
            )
        else:
            print(f"  {_ANSI_G}✓ Circuit Breaker: INACTIVE{_ANSI_RST}\n")

        # Load individual breaker statuses from circuit_breakers.json
        _cb_path = self.data_dir / "circuit_breakers.json"
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
        for _key, _label in _breaker_labels.items():
            _b = _cb_data.get(_key)
            if not isinstance(_b, dict):
                continue
            _tripped = _b.get("is_tripped", False)
            if _tripped:
                _trip_ts = _b.get("trip_time", "")
                _ts_short = _trip_ts[11:19] if _trip_ts and len(_trip_ts) >= 19 else (_trip_ts or "?")
                _icon = f"{_ANSI_R}✗ TRIPPED{_ANSI_RST}"
                _detail = f"  {_ANSI_DIM}@ {_ts_short}{_ANSI_RST}"

                # Show reason
                _reason = _b.get("trip_reason", "")
                if _reason:
                    _detail += f"  {_ANSI_Y}→ {_reason}{_ANSI_RST}"

                # Show value vs threshold
                _tv = _b.get("trip_value", 0.0)
                _th = _b.get("threshold", 0.0)
                if _tv or _th:
                    _detail += f"  {_ANSI_DIM}(val={_tv:.2f} thr={_th:.2f}){_ANSI_RST}"

                # Show cooldown remaining
                _cd_mins = _b.get("cooldown_minutes", 60)
                if _trip_ts:
                    try:
                        _trip_dt = datetime.fromisoformat(_trip_ts)
                        _elapsed = (datetime.now() - _trip_dt).total_seconds() / 60.0
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

    def _render_risk_tail(self, rs: dict) -> None:
        """Render tail risk values."""
        print("  \033[1m📉 TAIL RISK\033[0m")
        var = rs.get("var", 0) * 100
        kurtosis = rs.get("kurtosis", 0)
        vol = rs.get("realized_vol", 0) * 100

        kurt_col = _ANSI_R if kurtosis > KURTOSIS_FAT_TAIL_THRESHOLD else _ANSI_G
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
            f"{_ANSI_DIM}(excess; >0 = fat tails; gate fires at >3){_ANSI_RST}"
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

        qty_color = _ANSI_G if risk_final_qty == risk_req_qty else _ANSI_Y

        budget_used_pct = (risk_final_qty / risk_req_qty * 100) if risk_req_qty > _QTY_FLOOR else 100.0
        capped = risk_req_qty > _QTY_FLOOR and risk_final_qty < risk_req_qty * 0.999

        print(f"    Risk budget:       {risk_budget:>10.2f} USD")
        print(f"    Requested qty:     {risk_req_qty:>10.4f}")
        print(
            f"    Final qty:         {qty_color}{risk_final_qty:>10.4f}{_ANSI_RST}  "
            f"({budget_used_pct:.0f}% of request)"
        )
        if capped:
            print(f"    {_ANSI_Y}⚡ Qty capped — vol or depth constraint active{_ANSI_RST}")
        print(f"    Vol cap:           {vol_cap * 100:>9.2f}%  {_ANSI_DIM}(max position vol allowed){_ANSI_RST}")
        print(f"    Vol reference:     {vol_ref * 100:>9.3f}%  {_ANSI_DIM}(baseline for cap calc){_ANSI_RST}")

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
        print(f"    {'SIZE':>8}  {'BID':>{dec+6}}  {'ASK':<{dec+6}}  {'SIZE':<8}")
        print("    " + "─" * (BAR * 2 + dec * 2 + 18))
        for bid_row, ask_row in zip(padded_bids, padded_asks, strict=False):
            b_px, b_sz = bid_row
            a_px, a_sz = ask_row
            b_bar = int(BAR * b_sz / max_sz) if b_sz > 0 else 0
            a_bar = int(BAR * a_sz / max_sz) if a_sz > 0 else 0
            # Bid bar: filled part on the right (near price), empty on the left
            b_str = (
                f"{b_sz:>6.2f} "
                f"{_ANSI_G}{'░' * (BAR - b_bar)}{'▓' * b_bar}{_ANSI_RST}"
                if b_px
                else " " * (BAR + 8)
            )
            # Ask bar: filled part on the left (near price), empty on the right
            a_str = (
                f"{_ANSI_R}{'▓' * a_bar}{'░' * (BAR - a_bar)}{_ANSI_RST}"
                f" {a_sz:<6.2f}"
                if a_px
                else " " * (BAR + 8)
            )
            b_px_str = f"{b_px:.{dec}f}" if b_px else "   —  "
            a_px_str = f"{a_px:.{dec}f}" if a_px else "   —  "
            print(f"    {b_str}  {b_px_str}  {a_px_str}  {a_str}")
        print("    " + "─" * (BAR * 2 + dec * 2 + 18))
        print(
            f"    Total: {_ANSI_G}{depth_bid:>8.2f}{_ANSI_RST}              "
            f"{_ANSI_R}{depth_ask:>8.2f}{_ANSI_RST}"
        )

    def _render_signal_synthesis(self, vpin_z: float, imbalance: float) -> None:
        """Render the signal synthesis advisory block."""
        print()
        print("  \033[1m🧭 SIGNAL SYNTHESIS\033[0m")
        rs_regime = self.risk_stats.get("regime", "UNKNOWN")
        rs_feas   = float(self.risk_stats.get("feasibility", 0.5))
        rs_runway = float(self.risk_stats.get("runway", 0.0))
        toxic     = abs(vpin_z) > VPIN_HIGH_TOXICITY_THRESHOLD
        gate      = self.risk_stats.get("depth_gate_active", False)
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
            signals.append(f"{_ANSI_Y}⚡ Short runway ({rs_runway:.2f}) — harvester may exit early{_ANSI_RST}")
        if not signals:
            signals.append(f"{_ANSI_DIM}— No strong signals; model discretion applies{_ANSI_RST}")
        for s in signals:
            print(f"    {s}")

    def _render_market(self):
        """Render market microstructure."""
        print("\n\033[1m🔬 MARKET MICROSTRUCTURE\033[0m\n")
        ms = self.market_stats
        self._render_market_spread(ms)
        self._render_market_vpin(ms)
        self._render_market_imbalance(ms)

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
        print(f"                       {_ANSI_DIM}High +z = informed sellers active → widen stops / reduce size{_ANSI_RST}")
        bar_len = 40
        z_norm = max(0, min(1, (vpin_z + 3) / 6))
        pos = int(bar_len * z_norm)
        gauge = "░" * pos + "│" + "░" * (bar_len - pos - 1)
        print(f"\n    Z: [{gauge}]")
        print("       -3              0              +3")
        print()

    def _render_market_imbalance(self, ms: dict) -> None:
        """Render order imbalance and signal synthesis."""
        print("  \033[1m⚖️  ORDER IMBALANCE (QFI)\033[0m")
        has_real_sizes = ms.get("has_real_sizes", False)
        qfi_updates = int(ms.get("qfi_update_count", 0))
        imbalance = ms.get("imbalance", 0.0)
        signal_source = "size-weighted" if has_real_sizes else "quote-flow (QFI)"
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
        imb_scaled = int(mid * imbalance)
        if imbalance >= 0:
            bar = " " * mid + _ANSI_G + "█" * imb_scaled + _ANSI_RST + " " * (mid - imb_scaled)
        else:
            bar = " " * (mid + imb_scaled) + _ANSI_R + "█" * (-imb_scaled) + _ANSI_RST + " " * mid
        print(f"\n    [{bar}]")
        print("     SELL              ↕              BUY")
        self._render_signal_synthesis(ms.get("vpin_z", 0), imbalance)

    def _render_footer(self):
        """Render footer with controls and data freshness"""
        W = self._term_width()
        print("\n" + "─" * W)

        # Controls — two lines when narrow, one line when wide
        _trades_hint = "  [j/k] Select  [n/p] Page  [d] Detail  [b] Back" if self.current_tab == "trades" else ""
        ctrl_wide  = f"  [1-7] Tabs  |  [Tab/S+Tab] Cycle  |  [s] Presets  |  [r] Review CB  |  [h] Help  |  [Alt+K] Kill  |  [q/^Q/^X] Quit{_trades_hint}"
        ctrl_line1 = "  [1-7] Tabs  |  [Tab/S+Tab] Cycle  |  [s] Presets  |  [r] Review CB"
        ctrl_line2 = f"  [h] Help  |  [Alt+K] Kill  |  [q/^Q/^X] Quit{_trades_hint}"
        if len(ctrl_wide) <= W:
            print(ctrl_wide)
        else:
            print(ctrl_line1)
            print(ctrl_line2)

        # Data freshness — use order_book.json mtime (written ~1s by FIX handler)
        _ob = self.data_dir / _ORDER_BOOK_FILE
        _bc = self.data_dir / _BOT_CONFIG_FILE
        if _ob.exists():
            _fp = _ob
        elif _bc.exists():
            _fp = _bc
        else:
            _fp = None
        if _fp is not None:
            _fage = time.time() - os.path.getmtime(_fp)
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
        trade_file = Path("data/trade_log.jsonl")
        if not trade_file.exists():
            self._all_trades = []
            self._all_trades_loaded_at = _now
            return
        trades: list = []
        try:
            with open(trade_file, encoding="utf-8") as _f:
                for _raw in _f:
                    _s = _raw.strip()
                    if _s:
                        try:
                            trades.append(json.loads(_s))
                        except Exception:
                            pass
        except Exception:
            pass
        # Sort newest → oldest by exit_time, fallback to entry_time then trade_id
        def _skey(t: dict) -> tuple:
            return (t.get("exit_time") or t.get("entry_time") or "", t.get("trade_id", 0))
        trades.sort(key=_skey, reverse=True)
        self._all_trades = trades
        self._all_trades_loaded_at = _now

    def _render_trades(self) -> None:
        """Render the trade history tab with pagination and optional drill-down."""
        W = self._term_width()
        total = len(self._all_trades)
        if total == 0:
            print("\n\033[1m[T] TRADE HISTORY\033[0m  No trades recorded yet.")
            return
        max_page = max(0, (total - 1) // self._trades_per_page)
        self._trades_page = min(self._trades_page, max_page)
        page_start = self._trades_page * self._trades_per_page
        page_trades = self._all_trades[page_start: page_start + self._trades_per_page]
        self._trades_cursor = min(self._trades_cursor, max(0, len(page_trades) - 1))

        # Header
        lm = self.lifetime_metrics
        total_pnl = lm.get("total_pnl", 0.0)
        wins      = lm.get("winning_trades", 0)
        losses    = lm.get("losing_trades", 0)
        wr        = lm.get("win_rate", 0.0) * 100
        pg_str    = f"Pg {self._trades_page + 1}/{max_page + 1}"
        _tl_mode  = getattr(self, "_trade_log_mode", "")
        if _tl_mode == "mixed":
            _mode_hdr = f"  {_ANSI_Y}⚠ MIXED{_ANSI_RST}"
        elif _tl_mode == "live":
            _mode_hdr = f"  {_ANSI_G}💰 LIVE{_ANSI_RST}"
        else:
            _mode_hdr = f"  {_ANSI_Y}📄 PAPER{_ANSI_RST}"
        print(f"\n\033[1m[T] TRADE HISTORY\033[0m  [{total} trades]  {pg_str}{_mode_hdr}")

        if self._trades_detail:
            self._render_trade_detail(self._trades_detail_trade)
            return

        # Mixed-mode banner — operator must know metrics are contaminated
        if _tl_mode == "mixed":
            print(
                f"  {_ANSI_Y}⚠  MIXED MODE — paper and live trades combined. "
                f"Metrics span both modes. See [P] Performance tab for breakdown.{_ANSI_RST}"
            )

        # Summary bar
        _pnl_c = _ANSI_G if total_pnl >= 0 else _ANSI_R
        print(
            f"  Total PnL: {_pnl_c}{total_pnl:+.2f}{_ANSI_RST}  |  "
            f"W/L: {_ANSI_G}{wins}{_ANSI_RST}/{_ANSI_R}{losses}{_ANSI_RST}  "
            f"({_ANSI_G if wr >= 50 else _ANSI_R}{wr:.1f}%{_ANSI_RST} win rate)"
        )

        # Column header — M = mode badge (P=paper / L=live)
        _C_ID   = 5
        _C_DATE = 14
        _C_DIR  = 5
        _C_SYM  = 6
        _C_ENT  = 9
        _C_EXT  = 9
        _C_PNL  = 9
        _C_MFE  = 7
        _C_MAE  = 7
        _C_BRS  = 4
        _C_RSN  = 16
        _hdr_row = (
            f"  {'#':<{_C_ID}} M {'Date/Time':<{_C_DATE}} {'Dir':<{_C_DIR}} "
            f"{'Sym':<{_C_SYM}} {'Entry':>{_C_ENT}} {'Exit':>{_C_EXT}} "
            f"{'PnL':>{_C_PNL}} {'MFE':>{_C_MFE}} {'MAE':>{_C_MAE}} "
            f"{'Brs':>{_C_BRS}}  {'Reason':<{_C_RSN}}"
        )
        _sep = "  " + "-" * min(W - 4, max(len(_hdr_row) - 2, 50))
        print(f"\n{_ANSI_DIM}{_hdr_row}{_ANSI_RST}")
        print(_sep)

        # Trade rows
        _dec = self._price_decimals()
        for _row_idx, _t in enumerate(page_trades):
            _tid   = _t.get("trade_id", page_start + _row_idx + 1)
            _dir   = (_t.get("direction") or "").upper()
            _sym   = _t.get("symbol", "?")
            _entry = _t.get("entry_price", 0.0)
            _exit  = _t.get("exit_price", 0.0)
            _pnl   = _t.get("pnl", 0.0)
            _mfe   = _t.get("mfe", 0.0)
            _mae   = _t.get("mae", 0.0)
            _bars  = _t.get("bars_held", 0)
            _rsn   = (_t.get("close_reason") or _t.get("exit_reason") or "-")[:_C_RSN]

            _ts = _t.get("exit_time") or _t.get("entry_time") or ""
            try:
                _date_str = datetime.fromisoformat(_ts).strftime("%b-%d %H:%M")
            except Exception:
                _date_str = _ts[:13]

            _dc   = _ANSI_G if _dir == "LONG" else (_ANSI_R if _dir == "SHORT" else _ANSI_DIM)
            _pc   = self._pnl_color(_pnl)
            _ep_s = f"{_entry:.{_dec}f}"[-_C_ENT:]
            _xp_s = f"{_exit:.{_dec}f}"[-_C_EXT:]

            # Mode badge — single char, always renders 1 column wide
            _tmode = _t.get("trading_mode", "")
            if _tmode == "paper":
                _mb = f"{_ANSI_Y}P{_ANSI_RST}"
            elif _tmode == "live":
                _mb = f"{_ANSI_G}L{_ANSI_RST}"
            else:
                _mb = f"{_ANSI_DIM}?{_ANSI_RST}"

            _row = (
                f"  {str(_tid):<{_C_ID}} {_mb} {_date_str:<{_C_DATE}} "
                f"{_dc}{_dir:<{_C_DIR}}{_ANSI_RST} "
                f"{_sym:<{_C_SYM}} {_ep_s:>{_C_ENT}} {_xp_s:>{_C_EXT}} "
                f"{_pc}{_pnl:>+{_C_PNL}.2f}{_ANSI_RST} "
                f"{_ANSI_G}+{abs(_mfe):>{_C_MFE - 1}.2f}{_ANSI_RST} "
                f"{_ANSI_R}-{abs(_mae):>{_C_MAE - 1}.2f}{_ANSI_RST} "
                f"{_bars:>{_C_BRS}}  {_ANSI_DIM}{_rsn:<{_C_RSN}}{_ANSI_RST}"
            )
            if _row_idx == self._trades_cursor:
                print(f"\033[7m{_row}\033[0m")  # inverted highlight
            else:
                print(_row)

        print(_sep)
        print(
            f"  {_ANSI_DIM}[j/k] select row  "
            f"[n] next pg  [p] prev pg  "
            f"[d] drill into selected trade{_ANSI_RST}"
        )

    def _render_trade_detail(self, t: dict) -> None:
        """Render full detail card for a single trade."""
        W = self._term_width()
        _sep = "  " + "-" * min(W - 4, 74)
        _tid  = t.get("trade_id", "?")
        _tick = t.get("ticket", "-")
        _pid  = t.get("position_id", "-")
        _sym  = t.get("symbol", "?")
        _mode = t.get("trading_mode", "?")
        _dir  = (t.get("direction") or "").upper()
        _entry = t.get("entry_price", 0.0)
        _exit  = t.get("exit_price",  0.0)
        _pnl   = t.get("pnl",         0.0)
        _mfe   = t.get("mfe",         0.0)
        _mae   = t.get("mae",         0.0)
        _bars  = t.get("bars_held",   0)
        _rsn   = t.get("close_reason") or t.get("exit_reason") or "-"
        _w2l   = t.get("winner_to_loser", False)

        _entry_ts = t.get("entry_time", "")
        _exit_ts  = t.get("exit_time",  "")
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

        _dc  = _ANSI_G if _dir == "LONG" else (_ANSI_R if _dir == "SHORT" else _ANSI_DIM)
        _pc  = self._pnl_color(_pnl)
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
        print(f"  {'MFE:':<16} {_ANSI_G}+{_mfe:.4f}{_ANSI_RST}  (max favourable excursion)")
        print(f"  {'MAE:':<16} {_ANSI_R}-{_mae:.4f}{_ANSI_RST}  (max adverse excursion){_ratio_str}")
        print(f"  {'Close reason:':<16} {_rsn}")
        if _w2l:
            print(f"  {_ANSI_Y}[!] Winner-to-Loser: trade reversed into a loss after reaching MFE{_ANSI_RST}")
        print()
        print(f"  {_ANSI_DIM}[d] or [b] - return to trade list{_ANSI_RST}")

    def _pnl_color(self, pnl: float) -> str:
        """Return color code for PnL"""
        if pnl > 0:
            return _ANSI_G
        elif pnl < 0:
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
            os.kill(old_pid, 0)          # raises OSError if process is dead
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
    except Exception:  # noqa: BLE001
        pass


def main():
    """Run tabbed HUD"""
    if not _acquire_pidfile():
        sys.exit(1)

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
