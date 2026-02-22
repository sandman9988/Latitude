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
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Display threshold constants
FEASIBILITY_HIGH_THRESHOLD: float = 0.7
FEASIBILITY_MEDIUM_THRESHOLD: float = 0.5
BUFFER_HIGH_THRESHOLD: int = 1000
BUFFER_MEDIUM_THRESHOLD: int = 100
KURTOSIS_FAT_TAIL_THRESHOLD: float = 3.0
VPIN_HIGH_TOXICITY_THRESHOLD: float = 2.0
VPIN_ELEVATED_TOXICITY_THRESHOLD: float = 1.0
IMBALANCE_BUY_THRESHOLD: float = 0.3
IMBALANCE_SELL_THRESHOLD: float = -0.3

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

# Spread / VaR / Vol colour bands
SPREAD_OK_MAX: float = 0.0002          # spread < 0.0002 → green
SPREAD_WARN_MAX: float = 0.0005        # spread < 0.0005 → yellow
VAR_WARN_PCT: float = 1.5              # VaR % > 1.5 → yellow
VAR_HIGH_PCT: float = 3.0              # VaR % > 3 → red
VOL_HIGH_PCT: float = 2.0              # vol % > 2 → red
VOL_WARN_PCT: float = 1.0              # vol % > 1 → yellow
RUNWAY_WARN_BARS: float = 1.0          # runway < 1 bar → yellow
RUNWAY_OK_BARS: float = 3.0            # runway > 3 bars → green
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
    down_var = sum(p ** 2 for p in pnls if p < 0) / n
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
        "recent_pnl_sequence": pnls,
    }


def _hud_parse_dt(s: str):
    """Parse an ISO-format datetime string, returning None on failure."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _classify_trades_by_period(trades: list) -> tuple[list, list, list]:
    """Partition trade records into (daily, weekly, monthly) buckets by entry_time."""
    now = datetime.now(UTC)
    today = now.date()
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    daily: list = []
    weekly: list = []
    monthly: list = []
    for t in trades:
        dt = _hud_parse_dt(t.get("entry_time", ""))
        if dt is None:
            continue
        d = dt.date()
        if d == today:
            daily.append(t)
        if d >= week_start:
            weekly.append(t)
        if d >= month_start:
            monthly.append(t)
    return daily, weekly, monthly


class TabbedHUD:
    """Real-time tabbed HUD for trading bot monitoring"""

    TABS = {"1": "overview", "2": "performance", "3": "training", "4": "risk", "5": "market", "6": "log"}

    TAB_ORDER = ["overview", "performance", "training", "risk", "market", "log"]

    TAB_NAMES = {
        "overview": "📊 Overview",
        "performance": "📈 Performance",
        "training": "🧠 Training",
        "risk": "⚠️  Risk",
        "market": "🔬 Market",
        "log": "📝 Decision Log",
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
        self.risk_stats = {}
        self.market_stats = {}
        self.bot_config = {}
        self.production_metrics = {}
        self.offline_stats: dict = {}       # offline_training_status.json
        self.offline_job_progress: dict = {}  # keyed by (symbol, tf_minutes)
        self.universe_stats: dict = {}        # universe.json + PID liveness
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

        # Heartbeat
        self.heartbeat_idx = 0
        self.heartbeat_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        # Terminal settings for non-blocking input
        self.old_settings = None

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
        elif seq1 == "k":  # Alt+K — emergency kill switch
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
                elif key.lower() == "q" or key == "\x18":
                    self.running = False
                elif key.lower() == "s":
                    self._handle_symbol_selection()
                elif key.lower() == "h":
                    self._show_help()
        except Exception:  # noqa: BLE001 — ignore terminal read errors silently
            pass

    def _update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                self._check_input()
                self._refresh_data()
                self._render()
                time.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\033[0mHUD Error: {e}")
                time.sleep(2)

    def _load_performance_snapshot(self) -> None:
        """Load performance_snapshot.json into daily/weekly/monthly/lifetime metrics."""
        perf_file = self.data_dir / "performance_snapshot.json"
        if not perf_file.exists():
            return
        try:
            with open(perf_file) as f:
                data = json.load(f)
            self.daily_metrics = data.get("daily", {})
            self.weekly_metrics = data.get("weekly", {})
            self.monthly_metrics = data.get("monthly", {})
            self.lifetime_metrics = data.get("lifetime", {})
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

    def _load_universe_stats(self) -> None:
        """Load universe.json and annotate each entry with live PID status."""
        import os as _os  # noqa: PLC0415
        _uni_path = self.data_dir / "universe.json"
        if not _uni_path.exists():
            self.universe_stats = {}
            return
        try:
            _uni_raw: dict = json.loads(_uni_path.read_text())
            for _entry in _uni_raw.values():
                _pid = _entry.get("paper_pid")
                _alive = False
                if _pid:
                    try:
                        _os.kill(int(_pid), 0)
                        _alive = True
                    except (OSError, ProcessLookupError):
                        pass
                _entry["_pid_alive"] = _alive
            self.universe_stats = _uni_raw
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
        ob_file = self.data_dir / "order_book.json"
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

        self._load_json("bot_config.json", "bot_config")
        self._load_json("current_position.json", "position")
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

        # Self-test results (written at startup by run_self_test())
        st_file = self.data_dir / "self_test.json"
        if st_file.exists():
            try:
                with open(st_file) as f:
                    self.self_test_results = json.load(f).get("results", [])
            except Exception:
                pass

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
        """Compute performance metrics directly from trade_log.jsonl when snapshot is empty"""
        trade_file = Path("data/trade_log.jsonl")
        if not trade_file.exists():
            return
        starting_equity = float(self.bot_config.get("starting_equity", 10_000.0))
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

        daily, weekly, monthly = _classify_trades_by_period(trades)
        self.daily_metrics = _hud_period_metrics(daily, starting_equity)
        self.weekly_metrics = _hud_period_metrics(weekly, starting_equity)
        self.monthly_metrics = _hud_period_metrics(monthly, starting_equity)
        self.lifetime_metrics = _hud_period_metrics(trades, starting_equity)

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
            print("  [Tab]         - Cycle to next tab")
            print("  [Shift+Tab]   - Cycle to previous tab")
            print("  [s]           - Select symbol/timeframe preset")
            print("  [h]           - Show this help screen")
            print("  [q] / Ctrl+X  - Quit HUD")
            print("  [Alt+K]       - Emergency kill switch (close all positions + halt trading)")

            print("\n\033[1m📊 TAB DESCRIPTIONS\033[0m\n")
            print("  Overview      - Quick snapshot of position, daily stats, risk, and health")
            print("  Performance   - Detailed performance metrics (daily/weekly/monthly/lifetime)")
            print("  Training      - Agent training status, buffer sizes, loss metrics")
            print("  Risk          - Circuit breaker, VaR, volatility, regime, path geometry")
            print("  Market        - Spread, VPIN toxicity, order imbalance, depth")
            print("  Decision Log  - Last 20 trading decisions with color-coded events")

            print("\n\033[1m🎨 COLOR CODING\033[0m\n")
            print("  \033[92m✓ Green\033[0m       - Positive values, good status, active longs")
            print("  \033[91m✗ Red\033[0m         - Negative values, alerts, active shorts")
            print("  \033[93m⚡ Yellow\033[0m      - Neutral/warning, hold actions")
            print("  \033[94mℹ Blue\033[0m        - Informational messages")

            print("\n\033[1m📁 DATA SOURCES\033[0m\n")
            print("  All data is read from JSON files in the 'data/' directory:")
            print("    • bot_config.json          - Bot configuration and status")
            print("    • current_position.json    - Current position details")
            print("    • performance_snapshot.json - Performance metrics")
            print("    • training_stats.json      - Agent training statistics")
            print("    • risk_metrics.json        - Risk and market data")
            print("    • decision_log.json        - Trading decision history")

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
            RED = "\033[91m"
            YLW = "\033[93m"
            RST = "\033[0m"
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
        os.system("clear" if os.name != "nt" else "cls")

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

        # Always render footer (bottom menu)
        self._render_footer()

    def _render_training(self):  # noqa: PLR0912, PLR0915
        """Render agent training status"""
        ts = self.training_stats
        pm = self.production_metrics.get("metrics", {})
        TRIG_CAP = 2_000
        HARV_CAP = 10_000
        BAR_LEN = 26

        _G = "\033[92m"
        _Y = "\033[93m"
        _R = "\033[91m"
        _B = "\033[94m"
        _DIM = "\033[90m"
        _RST = "\033[0m"

        # ── Offline Training (rendered FIRST so it's always visible) ──────────
        ofs = self.offline_stats
        if ofs:
            ofs_status = ofs.get("status", "idle")
            ofs_total  = ofs.get("total_jobs", 0)
            ofs_done   = sum(1 for r in ofs.get("results", []) if r.get("status") in ("done", "error"))
            ofs_elapsed = ofs.get("elapsed_s", 0.0)
            ofs_start  = ofs.get("started_at", "")[:19].replace("T", " ") if ofs.get("started_at") else "—"
            ofs_end    = ofs.get("completed_at", "")[:19].replace("T", " ") if ofs.get("completed_at") else None

            if ofs_status == "running":
                status_badge = f"\033[93m⚙  RUNNING ({ofs_done}/{ofs_total} done)\033[0m"
            elif ofs_status == "complete":
                status_badge = f"{_G}✓ COMPLETE  ({ofs_done}/{ofs_total} jobs)\033[0m"
            else:
                status_badge = f"{_DIM}{ofs_status}\033[0m"

            pct = ofs_done / ofs_total if ofs_total else 0.0
            filled = int(BAR_LEN * pct)
            prog_col = _G if pct >= 1.0 else (_Y if pct > 0 else _DIM)
            prog_bar = f"{prog_col}[{'█' * filled}{'░' * (BAR_LEN - filled)}]\033[0m {pct*100:.0f}%"

            print(f"\n  \033[1m🏋 OFFLINE TRAINING\033[0m  {status_badge}")
            print(f"    Progress:  {prog_bar}   Elapsed: {ofs_elapsed:.0f}s")
            print(f"    Started:   {ofs_start}" + (f"   Finished: {ofs_end}" if ofs_end else ""))

            _results = ofs.get("results", [])
            if _results:
                sym_w = max(6, *(len(r.get("symbol", "")) for r in _results))
                print()
                print(f"    {'Symbol':<{sym_w}}  {'TF':>5}  {'Status':<9}  {'Detail':<38}  ZOmega")
                print(f"    {'─'*sym_w}  {'─'*5}  {'─'*9}  {'─'*38}  {'─'*8}")
                for r in _results:
                    sym      = r.get("symbol", "")
                    tf_label = r.get("label", f"M{r.get('timeframe_minutes', '?')}")
                    jstatus  = r.get("status", "queued")
                    if jstatus == "done":
                        jcol, jbadge = _G, "done     "
                    elif jstatus == "error":
                        jcol, jbadge = _R, "ERROR    "
                    elif jstatus == "running":
                        jcol, jbadge = _Y, "running  "
                    else:
                        jcol, jbadge = _DIM, "queued   "
                    zo      = r.get("z_omega")
                    ttrades = r.get("train_trades", 0)
                    vtrades = r.get("val_trades", 0)
                    steps   = r.get("total_train_steps", 0)
                    if zo is not None and jstatus == "done":
                        zo_col = _G if zo >= 1.0 else (_Y if zo >= Z_OMEGA_OFFLINE_WARM_MIN else _R)
                        zo_str = f"{zo_col}{zo:8.4f}\033[0m"
                    else:
                        zo_str = f"{_DIM}{'—':>8}\033[0m"

                    if jstatus in ("done", "error"):
                        detail = f"tr={ttrades:,}  val={vtrades:,}  steps={steps:,}"
                        row = f"    {sym:<{sym_w}}  {tf_label:>5}  {jcol}{jbadge}\033[0m  {detail:<38}  {zo_str}"
                    elif jstatus == "running":
                        prog = self.offline_job_progress.get(
                            (r.get("symbol"), r.get("timeframe_minutes")), {}
                        )
                        if prog:
                            pb = prog.get("pct", 0.0)
                            pb_fill = int(14 * pb / 100)
                            pb_bar = f"[{'█'*pb_fill}{'░'*(14-pb_fill)}] {pb:4.1f}%"
                            detail = f"{_Y}{pb_bar}{_RST}  ε={prog.get('epsilon',0):.3f}  β={prog.get('beta',0.4):.3f}"
                            row = f"    {sym:<{sym_w}}  {tf_label:>5}  {jcol}{jbadge}\033[0m  {detail}  {zo_str}"
                        else:
                            row = f"    {sym:<{sym_w}}  {tf_label:>5}  {jcol}{jbadge}\033[0m  {'—':<38}  {zo_str}"
                    else:
                        row = f"    {sym:<{sym_w}}  {tf_label:>5}  {jcol}{jbadge}\033[0m  {'—':<38}  {zo_str}"
                    if jstatus == "error" and r.get("error"):
                        row += f"  {_R}{r['error'][:30]}\033[0m"
                    print(row)
            print()

        # ── Paper Trading Pipeline ────────────────────────────────────────────
        if self.universe_stats:
            uni = self.universe_stats
            _STAGE_COL = {
                "PAPER":     _Y,
                "LIVE":      _G,
                "UNTRAINED": _DIM,
                "DEMOTED":   _R,
            }
            running_count = sum(1 for e in uni.values() if e.get("_pid_alive"))
            total_count   = len(uni)
            hdr_badge = (
                f"{_G}{running_count}/{total_count} running{_RST}"
                if running_count else
                f"{_DIM}0/{total_count} running{_RST}"
            )
            print(f"  \033[1m📈 PAPER TRADING PIPELINE\033[0m  {hdr_badge}")
            print()
            uni_sym_w = max(8, max((len(s) for s in uni), default=0))
            print(f"    {'Symbol':<{uni_sym_w}}  {'TF':>5}  {'Stage':<10}  {'ZΩ':>8}  {'Bot':>6}  Started")
            print(f"    {'─'*uni_sym_w}  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*19}")
            for sym, entry in sorted(uni.items()):
                stage   = entry.get("stage", "?")[:10]
                tf_min  = entry.get("timeframe_minutes", 0)
                tf_lbl  = f"M{tf_min}" if tf_min else "?"
                zo      = entry.get("z_omega")
                zo_str  = f"{zo:8.4f}" if zo is not None else f"{'—':>8}"
                if zo is not None:
                    zo_col = _G if zo > Z_OMEGA_POSITIVE_MIN else (_Y if zo > 0 else _R)
                    zo_str = f"{zo_col}{zo_str}{_RST}"
                pid_alive = entry.get("_pid_alive", False)
                pid       = entry.get("paper_pid")
                if pid_alive:
                    bot_str = f"{_G}  ▶ {pid}{_RST}"
                elif pid:
                    bot_str = f"{_R}dead {_DIM}({pid}){_RST}"
                else:
                    bot_str = f"{_DIM}  —{_RST}"
                started = entry.get("paper_started_at", "")[:19].replace("T", " ") if entry.get("paper_started_at") else "—"
                stage_col = _STAGE_COL.get(stage, _DIM)
                print(f"    {sym:<{uni_sym_w}}  {tf_lbl:>5}  {stage_col}{stage:<10}{_RST}  {zo_str}  {bot_str}  {started}")
            print()

        # ── Live Bot Training ─────────────────────────────────────────────────
        _ts_nonempty = any(v for v in self.training_stats.values() if v)
        if not _ts_nonempty:
            print(f"\033[1m🤖 LIVE BOT TRAINING\033[0m  {_DIM}(no live bot running){_RST}\n")
            return   # offline + paper already printed above; nothing more to show
        print("\033[1m🤖 LIVE BOT TRAINING\033[0m\n")

        def _pct_bar(val: int, cap: int) -> str:
            pct = min(val / cap, 1.0) if cap > 0 else 0.0
            filled = int(BAR_LEN * pct)
            col = _G if pct > _BUF_FILL_HIGH else (_Y if pct > _BUF_FILL_WARN else _R)
            return f"{col}[{'█' * filled}{'░' * (BAR_LEN - filled)}]{_RST} {pct * 100:5.1f}%"

        def _eps_bar(eps: float) -> str:
            pct = max(0.0, min(eps, 1.0))
            filled = int(BAR_LEN * pct)
            bracket = f"{_R}COLD{_RST}" if eps > EPS_WARM_MAX else (f"{_Y}WARM{_RST}" if eps > EPS_HOT_MAX else f"{_G}HOT{_RST}")
            col = _R if eps > EPS_WARM_MAX else (_Y if eps > EPS_HOT_MAX else _G)
            return f"{col}[{'█' * filled}{'░' * (BAR_LEN - filled)}] {eps:.4f}{_RST} {bracket}"

        def _beta_bar(beta: float) -> str:
            pct = max(0.0, min((beta - 0.4) / 0.6, 1.0))
            filled = int(BAR_LEN * pct)
            col = _G if beta > BETA_HOT_MIN else (_Y if beta > BETA_WARM_MIN else _DIM)
            return f"{col}[{'█' * filled}{'░' * (BAR_LEN - filled)}] {beta:.4f}{_RST}  (0.4 cold→1.0 fully corrected)"

        def _trend(hist: deque) -> str:
            vals = [v for v in list(hist) if v > 0]
            if len(vals) < TREND_MIN_SAMPLES:
                return f"{_DIM}→ — (need more samples){_RST}"
            half = len(vals) // 2
            old_mean = sum(vals[:half]) / half
            new_mean = sum(vals[half:]) / (len(vals) - half)
            delta_pct = (new_mean - old_mean) / old_mean * 100 if old_mean > 0 else 0
            if delta_pct < TREND_DELTA_NEG:
                return f"{_G}↓ {abs(delta_pct):.1f}% IMPROVING{_RST}"
            elif delta_pct > TREND_DELTA_POS:
                return f"{_R}↑ +{delta_pct:.1f}% DEGRADING{_RST}"
            else:
                return f"{_Y}→ {delta_pct:+.1f}% STABLE{_RST}"

        def _spark(hist: deque) -> str:
            vals = [v for v in list(hist) if v > 0]
            if len(vals) < TREND_STEP_PAIRS_MIN:
                return ""
            return self._create_sparkline(vals[-TREND_SPARK_TAIL:])

        def _velocity(step_hist: deque) -> str:
            pairs = list(step_hist)
            if len(pairs) < TREND_STEP_PAIRS_MIN:
                return f"{_DIM}—{_RST}"
            dt = pairs[-1][0] - pairs[0][0]
            ds = pairs[-1][1] - pairs[0][1]
            if dt <= 0 or ds <= 0:
                return f"{_DIM}0 steps/min{_RST}"
            rate = ds / dt * 60
            col = _G if rate > TREND_RATE_HIGH else (_Y if rate > TREND_RATE_WARN else _DIM)
            return f"{col}{rate:.1f} steps/min{_RST}"

        # ── Trigger Agent ─────────────────────────────────────────────────────
        trig_buf   = ts.get("trigger_buffer_size", 0)
        trig_steps = ts.get("trigger_training_steps", 0)
        trig_loss  = ts.get("trigger_loss", 0.0)
        trig_eps   = ts.get("trigger_epsilon", 0.0)
        trig_ready = ts.get("trigger_ready", False)
        ts.get("trigger_td_error", 0.0)
        trig_conf  = pm.get("trigger_confidence_avg", 0.5)

        ready_t = f"{_G}✓ Ready{_RST}" if trig_ready else f"{_Y}⏳ Filling…{_RST}"
        print(f"  \033[1m🎯 TRIGGER AGENT  (Entry)\033[0m  {ready_t}")
        print(f"    Steps:  {trig_steps:>10,}   Velocity: {_velocity(self._trig_step_hist)}")
        print(f"    Buffer: {_pct_bar(trig_buf, TRIG_CAP)}  {trig_buf:,}/{TRIG_CAP:,}")
        print(f"    ε:      {_eps_bar(trig_eps)}")
        _tl_str = f"{trig_loss:.6f}" if trig_loss > 0 else f"{_DIM}0.000000 (idle/no training event){_RST}"
        print(f"    Loss:   {_tl_str}")
        print(f"    Trend:  {_trend(self._trig_loss_hist)}")
        _sp = _spark(self._trig_loss_hist)
        if _sp:
            print(f"    Hist:   {_sp}")
        _cc = _G if CONF_HEALTHY_LOW < trig_conf < CONF_HEALTHY_HIGH else (_Y if CONF_WARM_LOW < trig_conf <= CONF_HEALTHY_LOW else _R)
        print(f"    Conf:   {_cc}{trig_conf:.3f}{_RST}  {_DIM}(healthy 0.55–0.85){_RST}")
        print()

        # ── Harvester Agent ───────────────────────────────────────────────────
        harv_buf      = ts.get("harvester_buffer_size", 0)
        harv_steps    = ts.get("harvester_training_steps", 0)
        harv_loss     = ts.get("harvester_loss", 0.0)
        harv_beta     = ts.get("harvester_beta", 0.4)
        harv_ready    = ts.get("harvester_ready", False)
        ts.get("harvester_td_error", 0.0)
        harv_min_hold = ts.get("harvester_min_hold_ticks", 10)
        harv_conf     = pm.get("harvester_confidence_avg", 0.5)

        ready_h = f"{_G}✓ Ready{_RST}" if harv_ready else f"{_Y}⏳ Filling…{_RST}"
        print(f"  \033[1m🌾 HARVESTER AGENT  (Exit)\033[0m  {ready_h}")
        print(f"    Steps:  {harv_steps:>10,}   Velocity: {_velocity(self._harv_step_hist)}")
        print(f"    Buffer: {_pct_bar(harv_buf, HARV_CAP)}  {harv_buf:,}/{HARV_CAP:,}")
        print(f"    β IS:   {_beta_bar(harv_beta)}")
        _hl_str = f"{harv_loss:.6f}" if harv_loss > 0 else f"{_DIM}0.000000 (idle/no training event){_RST}"
        print(f"    Loss:   {_hl_str}")
        print(f"    Trend:  {_trend(self._harv_loss_hist)}")
        _sp = _spark(self._harv_loss_hist)
        if _sp:
            print(f"    Hist:   {_sp}")
        _cc = _G if CONF_HEALTHY_LOW < harv_conf < CONF_HEALTHY_HIGH else (_Y if CONF_WARM_LOW < harv_conf <= CONF_HEALTHY_LOW else _R)
        print(f"    Conf:   {_cc}{harv_conf:.3f}{_RST}  {_DIM}(healthy 0.55–0.85){_RST}")
        print(f"    Min-hold: {harv_min_hold} ticks")
        print()

        # ── Arena ─────────────────────────────────────────────────────────────
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

        # ── Learning health summary ───────────────────────────────────────────
        last_train = ts.get("last_training_time", "Never")
        train_on   = self.bot_config.get("training_enabled", False)
        train_str  = f"{_G}ON{_RST}" if train_on else f"{_R}OFF{_RST}"
        trig_ok = f"{_G}✓{_RST}" if trig_ready else f"{_Y}⏳{_RST}"
        harv_ok = f"{_G}✓{_RST}" if harv_ready else f"{_Y}⏳{_RST}"
        print("  \033[1m📊 LEARNING HEALTH\033[0m")
        print(f"    Training: {train_str}   Last event: {last_train}")
        print(f"    Trigger {trig_ok}  Harvester {harv_ok}   Total steps: {trig_steps + harv_steps:,}")
        print()

    def _render_header(self):
        """Render header"""
        heartbeat = self.heartbeat_chars[self.heartbeat_idx]

        # Check for circuit breaker alert
        cb_active = self.risk_stats.get("circuit_breaker", "INACTIVE") == "ACTIVE"

        if cb_active:
            print("\033[41;97m╔" + "═" * 78 + "╗\033[0m")
            print("\033[41;97m║" + " " * 15 + "⚠️  CIRCUIT BREAKER ACTIVE - TRADING HALTED ⚠️" + " " * 16 + "║\033[0m")
            print("\033[41;97m╚" + "═" * 78 + "╝\033[0m")
        else:
            print("╔" + "═" * 78 + "╗")
            print("║" + " " * 18 + "ADAPTIVE RL TRADING BOT - TABBED HUD" + " " * 22 + "║")
            print("╚" + "═" * 78 + "╝")

        # Bot info
        symbol = self.bot_config.get("symbol", "UNKNOWN")
        tf = self.bot_config.get("timeframe", "1m")
        uptime = self.bot_config.get("uptime_seconds", 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)

        price = self.position.get("current_price", 0)
        now = self.last_update or datetime.now(UTC)

        print(f"\n🎯 {symbol} @ {tf}    💰 {price:.2f}    ⏱  {hours:02d}h {minutes:02d}m")
        print(f"{heartbeat} {now.strftime('%Y-%m-%d %H:%M:%S')} UTC (Live)")

    def _render_tab_bar(self):
        """Render tab navigation bar, including Tab 6 (Decision Log)"""
        print("\n" + "─" * 80)

        tabs = []
        tab_defs = [
            ("1", "Overview"),
            ("2", "Performance"),
            ("3", "Training"),
            ("4", "Risk"),
            ("5", "Market"),
            ("6", "Decision Log"),
        ]
        for key, name in tab_defs:
            tab_id = self.TABS.get(key, None)
            if tab_id == self.current_tab:
                tabs.append(f"\033[7m [{key}] {name} \033[0m")  # Inverted
            else:
                tabs.append(f" [{key}] {name} ")

        print("".join(tabs))
        print("─" * 80)

    def _render_position_block(self) -> None:
        """Render the position header block."""
        print("\n\033[1m📊 POSITION\033[0m")
        direction = self.position.get("direction", "FLAT")
        entry = self.position.get("entry_price", 0)
        current = self.position.get("current_price", 0)
        pnl = self.position.get("unrealized_pnl", 0)
        bars = self.position.get("bars_held", 0)
        dir_color = "\033[92m" if direction == "LONG" else ("\033[91m" if direction == "SHORT" else "\033[93m")
        pnl_color = self._pnl_color(pnl)
        print(
            f"  {dir_color}{direction}\033[0m @ {entry:.2f} → {current:.2f}  |  "
            f"PnL: {pnl_color}{pnl:+.2f}\033[0m  |  Bars: {bars}"
        )
        if direction != "FLAT":
            mfe = self.position.get("mfe", 0.0)
            mae = self.position.get("mae", 0.0)
            mfe_color = "\033[92m" if mfe > 0 else "\033[93m"
            mae_color = "\033[91m" if mae > 0 else "\033[93m"
            # MFE is max favourable excursion (positive = profit)
            # MAE is max adverse excursion stored as magnitude — display negated
            print(f"  MFE: {mfe_color}+{mfe:.2f}\033[0m  |  MAE: {mae_color}-{mae:.2f}\033[0m  (USD, excl. spread)")

    def _render_overview(self):
        """Render overview tab - compact summary"""
        self._render_position_block()

        # Account balance / equity
        print("\n\033[1m💰 ACCOUNT\033[0m")
        _starting = float(self.bot_config.get("starting_equity", 10_000.0))
        _lifetime_pnl = float(self.lifetime_metrics.get("total_pnl", 0.0))
        _unreal = float(self.position.get("unrealized_pnl", 0.0))
        _balance = _starting + _lifetime_pnl
        _equity  = _balance + _unreal
        print(
            f"  Balance: {self._pnl_color(_balance - _starting)}{_balance:>10.2f}\033[0m  |  "
            f"Equity:  {self._pnl_color(_equity  - _starting)}{_equity:>10.2f}\033[0m  |  "
            f"Unrealized: {self._pnl_color(_unreal)}{_unreal:+.2f}\033[0m"
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
        vol = self.risk_stats.get("realized_vol", 0) * 100
        feas = self.risk_stats.get("feasibility", 0.5)

        cb_status = "\033[91m● ACTIVE\033[0m" if cb == "ACTIVE" else "\033[92m● OK\033[0m"
        feas_color = (
            "\033[92m"
            if feas > FEASIBILITY_HIGH_THRESHOLD
            else ("\033[93m" if feas > FEASIBILITY_MEDIUM_THRESHOLD else "\033[91m")
        )
        _regime_colors = {
            "TRENDING": "\033[92m",
            "MEAN_REVERTING": "\033[93m",
            "TRANSITIONAL": "\033[94m",
            "UNKNOWN": "\033[90m",
        }
        regime_color = _regime_colors.get(regime, "\033[90m")

        print(
            f"  Circuit: {cb_status}  |  Regime: {regime_color}{regime}\033[0m  |  Vol: {vol:.2f}%  |  "
            f"Feasibility: {feas_color}{feas:.2f}\033[0m"
        )

        self._render_agent_status_block()

        # Market snapshot
        print("\n\033[1m🔬 MARKET\033[0m")
        spread = self.market_stats.get("spread", 0)
        vpin = self.market_stats.get("vpin", 0)
        vpin_z = self.market_stats.get("vpin_z", 0)
        imb = self.market_stats.get("imbalance", 0)

        vpin_status = "\033[91m⚠️ HIGH\033[0m" if abs(vpin_z) > VPIN_HIGH_TOXICITY_THRESHOLD else "\033[92m✓\033[0m"
        _has_real = self.market_stats.get("has_real_sizes", False)
        _imb_label = "Imb" if _has_real else "QFI"
        print(f"  Spread: {spread:.5f}  |  VPIN: {vpin:.3f} {vpin_status}  |  {_imb_label}: {imb:+.3f}")

        self._render_system_health_block()

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

    def _render_system_health_block(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Render the expanded system health rows and startup self-test."""
        _G = "\033[92m"
        _Y = "\033[93m"
        _R = "\033[91m"
        _RST = "\033[0m"

        def _ok(s: str) -> str: return f"{_G}✓ {s}{_RST}"
        def _warn(s: str) -> str: return f"{_Y}⚡ {s}{_RST}"
        def _bad(s: str) -> str: return f"{_R}✗ {s}{_RST}"

        print("\n\033[1m🏥 SYSTEM HEALTH\033[0m")

        # ── Row 1: Connectivity ───────────────────────────────────────────────
        # Freshness of the bot's most recent data write — use order_book.json mtime
        # (written ~1s by FIX handler). Falls back to bot_config.json mtime.
        _age_str = "n/a"
        _ob_path = self.data_dir / "order_book.json"
        _bc_path = self.data_dir / "bot_config.json"
        _ref_path = _ob_path if _ob_path.exists() else (_bc_path if _bc_path.exists() else None)
        if _ref_path is not None:
            import os as _os  # noqa: PLC0415
            _file_age = time.time() - _os.path.getmtime(_ref_path)
            _age_str = f"{_file_age:.0f}s"
            _data_item = _ok(f"Data {_age_str}") if _file_age < DATA_AGING_SECS else (_warn(f"Data {_age_str}") if _file_age < DATA_STALE_SECS else _bad(f"Bot silent {_age_str}"))
        else:
            _data_item = _bad("No data")

        _cb = self.risk_stats.get("circuit_breaker", "INACTIVE")
        _cb_item = _bad("CB ACTIVE") if _cb == "ACTIVE" else _ok("CB OK")

        _depth_gate = self.risk_stats.get("depth_gate_active", False)
        _gate_item = _warn("Depth gate") if _depth_gate else _ok("Gate open")

        _feas = float(self.risk_stats.get("feasibility", 0.5))
        _feas_col = _G if _feas > FEASIBILITY_HIGH_THRESHOLD else (_Y if _feas > FEASIBILITY_MEDIUM_THRESHOLD else _R)
        _feas_item = f"Feas: {_feas_col}{_feas:.2f}{_RST}"

        print(f"  {_data_item}  │  {_cb_item}  │  {_gate_item}  │  {_feas_item}")

        # ── Row 2: Risk ───────────────────────────────────────────────────────
        _vol = float(self.risk_stats.get("realized_vol", 0)) * 100
        _vol_col = _G if _vol < VOL_WARN_PCT else (_Y if _vol < VOL_HIGH_PCT else _R)
        _vol_item = f"Vol: {_vol_col}{_vol:.2f}%{_RST}"

        _var = float(self.risk_stats.get("var", 0)) * 100
        _var_col = _G if _var < VAR_WARN_PCT else (_Y if _var < VAR_HIGH_PCT else _R)
        _var_item = f"VaR: {_var_col}{_var:.2f}%{_RST}"

        _budget = float(self.risk_stats.get("risk_budget_usd", 0))
        _budget_col = _G if _budget > _BUDGET_OK_MIN else (_Y if _budget > 0 else _R)
        _budget_item = f"Budget: {_budget_col}${_budget:.2f}{_RST}"

        _eff = float(self.risk_stats.get("efficiency", 0))
        _eff_col = _G if _eff > EFF_HIGH_THRESHOLD else (_Y if _eff > EFF_WARN_THRESHOLD else _R)
        _eff_item = f"Eff: {_eff_col}{_eff:.2f}{_RST}"

        print(f"  {_vol_item}  │  {_var_item}  │  {_budget_item}  │  {_eff_item}")

        # ── Row 3: Buffers ────────────────────────────────────────────────────
        _TRIG_CAP = 2_000
        _HARV_CAP = 10_000
        _trig_buf = self.training_stats.get("trigger_buffer_size", 0)
        _harv_buf = self.training_stats.get("harvester_buffer_size", 0)
        _trig_pct = _trig_buf / _TRIG_CAP * 100
        _harv_pct = _harv_buf / _HARV_CAP * 100
        _trig_rdy = self.training_stats.get("trigger_ready", False)
        _harv_rdy = self.training_stats.get("harvester_ready", False)
        _trig_col = _G if _trig_pct > BUF_PCT_HIGH else (_Y if _trig_pct > BUF_PCT_WARN else _R)
        _harv_col = _G if _harv_pct > BUF_PCT_HIGH else (_Y if _harv_pct > BUF_PCT_WARN else _R)
        def _rdy_icon(r):
            return f"{_G}✓{_RST}" if r else f"{_Y}…{_RST}"
        print(
            f"  Trig buf: {_trig_col}{_trig_buf:,}/{_TRIG_CAP:,} ({_trig_pct:.0f}%){_RST} {_rdy_icon(_trig_rdy)}  │  "
            f"Harv buf: {_harv_col}{_harv_buf:,}/{_HARV_CAP:,} ({_harv_pct:.0f}%){_RST} {_rdy_icon(_harv_rdy)}"
        )

        # ── Row 4: Model ──────────────────────────────────────────────────────
        _eps = float(self.training_stats.get("trigger_epsilon", 1.0))
        _beta = float(self.training_stats.get("harvester_beta", 0.4))
        _trig_steps = self.training_stats.get("trigger_training_steps", 0)
        _harv_steps = self.training_stats.get("harvester_training_steps", 0)
        _trig_loss = self.training_stats.get("trigger_loss", None)
        _harv_loss = self.training_stats.get("harvester_loss", None)
        _eps_col = _G if _eps < EPS_HOT_MAX else (_Y if _eps < EPS_WARM_MAX else _R)
        _eps_lbl = "HOT" if _eps < EPS_HOT_MAX else ("WARM" if _eps < EPS_WARM_MAX else "COLD")
        def _loss_str(v):
            return (f"{v:.4f}" if v is not None else "n/a")
        print(
            f"  ε={_eps_col}{_eps:.4f} {_eps_lbl}{_RST}  steps={_trig_steps:,}  loss={_loss_str(_trig_loss)}  │  "
            f"β={_beta:.4f}  steps={_harv_steps:,}  loss={_loss_str(_harv_loss)}"
        )

        # ── Row 5: Market microstructure ──────────────────────────────────────
        _spread = float(self.market_stats.get("spread", 0))
        _vpin = float(self.market_stats.get("vpin", 0))
        _vpin_z = float(self.market_stats.get("vpin_z", 0))
        _runway = float(self.risk_stats.get("runway", 0))
        _spread_col = _G if _spread < SPREAD_OK_MAX else (_Y if _spread < SPREAD_WARN_MAX else _R)
        _vpin_col = _R if abs(_vpin_z) > _VPIN_OV_HIGH else (_Y if abs(_vpin_z) > _VPIN_OV_ELEVATED else _G)
        _runway_col = _G if _runway > RUNWAY_OK_BARS else (_Y if _runway > RUNWAY_WARN_BARS else _R)
        print(
            f"  Spread: {_spread_col}{_spread:.5f}{_RST}  │  "
            f"VPIN: {_vpin_col}{_vpin:.3f} (z={_vpin_z:+.1f}){_RST}  │  "
            f"Runway: {_runway_col}{_runway:.1f} bars{_RST}"
        )

        # Self-test results
        if self.self_test_results:
            _sev_col  = {"PASS": "\033[92m", "INFO": "\033[94m", "WARNING": "\033[93m", "CRITICAL": "\033[91m"}
            _sev_icon = {"PASS": "✓", "INFO": "ℹ", "WARNING": "⚠", "CRITICAL": "✗"}
            n_crit = sum(1 for r in self.self_test_results if r["sev"] == "CRITICAL")
            n_warn = sum(1 for r in self.self_test_results if r["sev"] == "WARNING")
            status = "\033[91m🔴 FAILED\033[0m" if n_crit else ("\033[93m🟡 DEGRADED\033[0m" if n_warn else "\033[92m🟢 CLEAR\033[0m")
            n_pass = sum(1 for r in self.self_test_results if r["sev"] in ("PASS", "INFO"))
            print(f"\n\033[1m🔍 STARTUP SELF-TEST\033[0m  {status}  "
                  f"\033[90m({n_pass} OK, {n_warn} warn, {n_crit} crit)\033[0m")
            only_fails = n_crit > 0 or n_warn > 0
            for r in self.self_test_results:
                sev = r["sev"]
                if only_fails and sev in ("PASS", "INFO"):
                    continue  # show only problems when there are any
                col  = _sev_col.get(sev, "")
                icon = _sev_icon.get(sev, "?")
                detail = f"  \033[90m{r['detail']}\033[0m" if r.get("detail") else ""
                print(f"  {col}{icon} {r['name']}\033[0m{detail}")

    def _render_performance(self):  # noqa: PLR0915
        """Render detailed performance metrics"""
        src = "  \033[90m(source: trade_log.jsonl)\033[0m" if self._metrics_from_trade_log else ""
        print(f"\n\033[1m📈 PERFORMANCE METRICS\033[0m{src}\n")

        # Column headers
        print(
            f"  {'Period':<9} {'Trades':>7} {'Win%':>7} {'PnL':>11} {'Sharpe':>7} {'Omega':>7} {'MaxDD%':>8}"
        )
        print("  " + "─" * 61)

        for label, metrics in [
            ("Daily", self.daily_metrics),
            ("Weekly", self.weekly_metrics),
            ("Monthly", self.monthly_metrics),
            ("Lifetime", self.lifetime_metrics),
        ]:
            trades = metrics.get("total_trades", 0)
            wr = metrics.get("win_rate", 0) * 100
            pnl = metrics.get("total_pnl", 0)
            sharpe = metrics.get("sharpe_ratio", 0)
            omega = metrics.get("omega_ratio", 0)
            maxdd = metrics.get("max_drawdown", 0.0)  # already a % of peak equity

            pnl_color = self._pnl_color(pnl)
            dd_color = "\033[91m" if maxdd > DD_HIGH_PCT else ("\033[93m" if maxdd > DD_WARN_PCT else "\033[92m")

            print(
                f"  {label:<9} {trades:>7} {wr:>6.1f}% {pnl_color}{pnl:>+10.2f}\033[0m "
                f"{sharpe:>7.2f} {omega:>7.2f} {dd_color}{maxdd:>7.2f}%\033[0m"
            )

        # ── Trade quality ─────────────────────────────────────────────────
        print("\n\033[1m📊 TRADE QUALITY\033[0m\n")

        lt = self.lifetime_metrics
        avg_win  = lt.get("avg_win", 0.0)
        avg_loss = lt.get("avg_loss", 0.0)   # stored as negative
        profit_f = lt.get("profit_factor", 0.0)
        expect   = lt.get("expectancy", 0.0)
        sortino  = lt.get("sortino_ratio", 0.0)
        best     = lt.get("best_trade", 0.0)
        worst    = lt.get("worst_trade", 0.0)

        # Payoff ratio: avg_win / |avg_loss| — the core RL reward sanity check.
        # >1 means wins are larger than losses; target ≥ 1.5 for positive EV.
        abs_loss = abs(avg_loss)
        payoff   = avg_win / abs_loss if abs_loss > _PAYOFF_FLOOR else 0.0
        pay_col  = "\033[92m" if payoff >= PAYOFF_GOOD_MIN else ("\033[93m" if payoff >= 1.0 else "\033[91m")
        exp_col  = "\033[92m" if expect > 0 else "\033[91m"
        pf_col   = "\033[92m" if profit_f >= PROFIT_FACTOR_GOOD_MIN else ("\033[93m" if profit_f >= 1.0 else "\033[91m")

        print(f"  Payoff ratio:     {pay_col}{payoff:>7.2f}x\033[0m  \033[90m(avg_win/|avg_loss|  target ≥1.5)\033[0m")
        print(f"  Avg W / Avg L:    {avg_win:>+8.2f} / {avg_loss:>+8.2f}")
        print(f"  Profit factor:    {pf_col}{profit_f:>7.2f}\033[0m  \033[90m(gross_profit/gross_loss  target ≥1.2)\033[0m")
        print(f"  Expectancy/trade: {exp_col}{expect:>+8.4f}\033[0m")
        print(f"  Sortino ratio:    {sortino:>7.3f}")
        print(f"  Best / Worst:     {best:>+8.2f} / {worst:>+8.2f}")

        # ── Trade timing ──────────────────────────────────────────────────
        pm = self.production_metrics.get("metrics", {})
        if pm:
            def _fmt_dur(mins: float) -> str:
                if mins <= 0:
                    return "—"
                if mins < _DURATION_HOUR_MINS:
                    return f"{mins:.0f}m"
                if mins < _DURATION_DAY_MINS:
                    return f"{mins / 60:.1f}h"
                return f"{mins / _DURATION_DAY_MINS:.1f}d"
            avg_dur    = pm.get("avg_trade_duration_mins", 0.0)
            last_trade = pm.get("last_trade_mins_ago", 0.0)
            print(f"\n  Avg hold time:    {_fmt_dur(avg_dur):>8}")
            print(f"  Last trade:       {_fmt_dur(last_trade):>8} ago")

            # ── Prediction convergence block ───────────────────────────────
            rw_delta  = pm.get("runway_delta_ema", 0.0)
            rw_acc    = pm.get("runway_accuracy_ema", 0.5)
            cc_err    = pm.get("conf_calib_err_ema", 0.5)
            platt_a   = pm.get("platt_a", 1.0)
            platt_b   = pm.get("platt_b", 0.0)

            # Runway delta: green when |delta| < 1pt, yellow < 3pt, red otherwise
            delta_col = ("\033[92m" if abs(rw_delta) < 1.0
                         else ("\033[93m" if abs(rw_delta) < RUNWAY_DELTA_WARN_MAX else "\033[91m"))
            # Runway accuracy: green > 0.7, yellow > 0.4, red otherwise
            acc_col   = ("\033[92m" if rw_acc > RUNWAY_ACCURACY_GOOD
                         else ("\033[93m" if rw_acc > RUNWAY_ACCURACY_WARN else "\033[91m"))
            # Calibration error: green < 0.15, yellow < 0.30, red otherwise
            cc_col    = ("\033[92m" if cc_err < CONF_CALIB_OK_MAX
                         else ("\033[93m" if cc_err < CONF_CALIB_WARN_MAX else "\033[91m"))
            # Platt params: grey = default (1.0/0.0), blue = adapted
            pa_col = "\033[94m" if abs(platt_a - 1.0) > PLATT_ADAPTED_DELTA or abs(platt_b) > PLATT_ADAPTED_DELTA else "\033[90m"

            print("\n\033[1m🎯 PREDICTION CONVERGENCE\033[0m  \033[90m(EMA-10 trades)\033[0m\n")
            print(
                f"  Runway Δ (pred−actual):   "
                f"{delta_col}{rw_delta:>+7.2f} pts\033[0m  "
                f"\033[90m→ 0 = perfect\033[0m"
            )
            print(
                f"  Runway Accuracy:          "
                f"{acc_col}{rw_acc:>7.3f}\033[0m      "
                f"\033[90m→ 1 = perfect\033[0m"
            )
            print(
                f"  Conf Calibration Error:   "
                f"{cc_col}{cc_err:>7.3f}\033[0m      "
                f"\033[90m→ 0 = calibrated\033[0m"
            )
            print(
                f"  Platt  a={pa_col}{platt_a:.4f}\033[0m  "
                f"b={pa_col}{platt_b:+.4f}\033[0m  "
                f"\033[90m(grey=default, blue=adapted)\033[0m"
            )

    def _render_jsonl_decision_entries(self, entries: list) -> None:
        """Render the rich JSONL decision log entries."""
        header = (
            f"  {'Time':<8} {'Agent':<10} {'Decision':<10} {'Conf':>6} {'Rnwy$':>6} {'VPIN-z':>7} {'Price':>10}"
        )
        _counts: dict[str, int] = {}
        for _e in entries:
            _d = _e.get("decision", "?").upper()
            _counts[_d] = _counts.get(_d, 0) + 1
        _dist = "  ".join(f"{k}:{v}" for k, v in sorted(_counts.items()))
        print(f"  Distribution (last {len(entries)}): {_dist}\n")
        print(header)
        print("  " + "─" * 60)
        for entry in entries:
            ts_raw = entry.get("timestamp", "?")
            try:
                ts_str = ts_raw[11:19] if len(ts_raw) >= _DEC_LOG_TS_MIN_LEN else ts_raw
            except Exception:
                ts_str = str(ts_raw)[:8]
            agent = entry.get("agent", "?")[:9]
            decision = entry.get("decision", "?")[:9]
            conf = entry.get("confidence", 0.0)
            ctx = entry.get("context", {})
            reasoning = entry.get("reasoning", {})
            price = ctx.get("price", 0.0)
            vpin_z = ctx.get("vpin_z", 0.0)
            runway_pct = reasoning.get("predicted_runway", 0.0)
            runway_usd = runway_pct * price if price > 0 else runway_pct
            if decision.upper() in ("BUY", "LONG", "ENTER"):
                color = "\033[92m"
            elif decision.upper() in ("SELL", "SHORT", "EXIT", "CLOSE"):
                color = "\033[91m"
            elif decision.upper() == "HOLD":
                color = "\033[93m"
            else:
                color = "\033[0m"
            vpin_flag = "\033[91m⚠\033[0m" if abs(vpin_z) > _DEC_LOG_VPIN_WARN else " "
            dec = self._price_decimals(price)
            print(
                f"  {ts_str:<8} {agent:<10} {color}{decision:<10}\033[0m "
                f"{conf:>6.3f} {runway_usd:>6.2f} {vpin_z:>+6.2f}{vpin_flag} {price:>10.{dec}f}"
            )
        print("  " + "─" * 60)

    def _render_legacy_decision_entries(self, entries: list) -> None:
        """Render legacy JSON-format decision log entries."""
        print(f"  Showing {len(entries[-20:])} most recent decisions (legacy format):\n")
        print("  " + "─" * 76)
        for entry in entries[-20:]:
            ts = entry.get("timestamp", "?")
            event = entry.get("event", "?")
            details = entry.get("details", {})
            if isinstance(details, dict):
                pos = details.get("cur_pos", "?")
                action = details.get("action", "?")
                conf = details.get("confidence", "?")
                pnl = details.get("pnl", "?")
                details_str = f"Pos:{pos} Act:{action} Conf:{conf} PnL:{pnl}"
            else:
                details_str = str(details)
            if "OPEN" in event.upper() or "entry" in event.lower():
                color = "\033[92m"
            elif "CLOSE" in event.upper() or "exit" in event.lower():
                color = "\033[91m"
            elif "HOLD" in event.upper():
                color = "\033[93m"
            else:
                color = "\033[0m"
            print(f"  [{ts}] {color}{event}\033[0m: {details_str}")
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
            with open(log_file, encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as e:
            print(f"  ❌ Error reading decision log: {e}")
            return

        if not entries:
            print("  No entries yet. Waiting for bot decisions...")
            return

        self._render_legacy_decision_entries(entries)

    def _render_risk(self):  # noqa: PLR0915
        """Render risk management details"""
        print("\n\033[1m⚠️  RISK MANAGEMENT\033[0m\n")

        rs = self.risk_stats

        # Circuit breaker status
        cb = rs.get("circuit_breaker", "INACTIVE")
        if cb == "ACTIVE":
            print("  \033[91m╔════════════════════════════════════════╗")
            print("  ║     ⚠️  CIRCUIT BREAKER ACTIVE ⚠️       ║")
            print("  ╚════════════════════════════════════════╝\033[0m\n")
        else:
            print("  \033[92m✓ Circuit Breaker: INACTIVE\033[0m\n")

        # Tail risk
        print("  \033[1m📉 TAIL RISK\033[0m")
        var      = rs.get("var", 0) * 100   # express as %
        kurtosis = rs.get("kurtosis", 0)
        vol      = rs.get("realized_vol", 0) * 100

        kurt_col = "\033[91m" if kurtosis > KURTOSIS_FAT_TAIL_THRESHOLD else "\033[92m"
        var_col  = "\033[91m" if var > VAR_HIGH_PCT else ("\033[93m" if var > VAR_WARN_PCT else "\033[92m")
        vol_col  = "\033[91m" if vol > VOL_HIGH_PCT else ("\033[93m" if vol > VOL_WARN_PCT else "\033[92m")

        print(f"    VaR 95%:           {var_col}{var:>9.3f}%\033[0m  \033[90m(position loss at 95th pct)\033[0m")
        print(f"    Realized vol:      {vol_col}{vol:>9.3f}%\033[0m")
        print(f"    Kurtosis:          {kurt_col}{kurtosis:>9.2f}\033[0m  \033[90m(>3 = fat tails → wider stops)\033[0m")
        print()

        # Regime
        print("  \033[1m🌐 REGIME\033[0m")
        regime = rs.get("regime", "UNKNOWN")
        zeta   = rs.get("regime_zeta", 1.0)

        regime_colors = {
            "TRENDING": "\033[92m",
            "MEAN_REVERTING": "\033[93m",
            "TRANSITIONAL": "\033[94m",
            "UNKNOWN": "\033[90m",
        }
        _regime_tips = {
            "TRENDING":      "trend-follow; let winners run",
            "MEAN_REVERTING":"fade extremes; tighten target",
            "TRANSITIONAL":  "reduce size; wait for clarity",
            "UNKNOWN":       "cold-start; use fallback rules",
        }
        regime_color = regime_colors.get(regime, "\033[90m")
        print(f"    Regime:            {regime_color}{regime}\033[0m  \033[90m({_regime_tips.get(regime, '')})\033[0m")
        print(f"    Confidence (ζ):    {zeta:>10.2f}  \033[90m(1.0 = full confidence)\033[0m")

        print()

        # Path geometry — these are features fed directly to the RL agents
        print("  \033[1m📐 PATH GEOMETRY  \033[90m(RL feature inputs)\033[0m")
        eff    = rs.get("efficiency", 0)
        gamma  = rs.get("gamma", 0)
        runway = rs.get("runway", 0.5)
        feas   = rs.get("feasibility", 0.5)

        feas_color = (
            "\033[92m"
            if feas > FEASIBILITY_HIGH_THRESHOLD
            else ("\033[93m" if feas > FEASIBILITY_MEDIUM_THRESHOLD else "\033[91m")
        )
        eff_col = "\033[92m" if eff > EFF_HIGH_THRESHOLD else ("\033[93m" if eff > EFF_WARN_THRESHOLD else "\033[91m")
        gam_col = "\033[92m" if gamma > 0 else "\033[91m"
        rwy_col = "\033[92m" if runway > RUNWAY_OK_BARS else ("\033[93m" if runway > RUNWAY_WARN_BARS else "\033[91m")

        print(f"    Efficiency:        {eff_col}{eff:>10.3f}\033[0m  \033[90m(path directness; 1=straight trend)\033[0m")
        print(f"    Gamma (γ):         {gam_col}{gamma:>+10.3f}\033[0m  \033[90m(price acceleration; +ve favours longs)\033[0m")
        print(f"    Runway:            {rwy_col}{runway:>10.3f}\033[0m  \033[90m(bars before vol kills the move)\033[0m")
        print(f"    Entry Feasibility: {feas_color}{feas:>10.3f}\033[0m")

        # Feasibility gauge
        bar_len = 40
        feas_pct = max(0, min(1, feas))
        filled = int(bar_len * feas_pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n    [{bar}]")
        print("     LOW                                HIGH")

        # ── Position Sizing ───────────────────────────────────────────────
        print()
        print("  \033[1m💰 POSITION SIZING\033[0m")
        risk_budget = rs.get("risk_budget_usd", 0.0)
        risk_req_qty = rs.get("risk_requested_qty", 0.0)
        risk_final_qty = rs.get("risk_final_qty", 0.0)
        vol_cap = rs.get("vol_cap", 0.0)
        vol_ref = rs.get("vol_reference", 0.0)

        qty_color = "\033[92m" if risk_final_qty == risk_req_qty else "\033[93m"

        # % of budget consumed by this sizing decision
        budget_used_pct = (risk_final_qty / risk_req_qty * 100) if risk_req_qty > _QTY_FLOOR else 100.0
        capped = risk_req_qty > _QTY_FLOOR and risk_final_qty < risk_req_qty * 0.999

        print(f"    Risk budget:       {risk_budget:>10.2f} USD")
        print(f"    Requested qty:     {risk_req_qty:>10.4f}")
        print(f"    Final qty:         {qty_color}{risk_final_qty:>10.4f}\033[0m  ({budget_used_pct:.0f}% of request)")
        if capped:
            print("    \033[93m⚡ Qty capped — vol or depth constraint active\033[0m")
        print(f"    Vol cap:           {vol_cap * 100:>9.2f}%  \033[90m(max position vol allowed)\033[0m")
        print(f"    Vol reference:     {vol_ref * 100:>9.3f}%  \033[90m(baseline for cap calc)\033[0m")

    def _render_order_book_ladder(self, bids: list, asks: list, depth_bid: float, depth_ask: float, dec: int) -> None:
        """Render the L2 order-book price ladder (5 rows, aligned columns)."""
        N = 5
        padded_asks = (asks + [[0.0, 0.0]] * N)[:N]
        padded_bids = (bids + [[0.0, 0.0]] * N)[:N]
        all_sizes = [s for _, s in bids + asks if s > 0]
        max_sz = max(all_sizes) if all_sizes else 1.0
        BAR = 12
        print(f"    {'SIZE':>8}  {'BID':>{dec+6}}  {'ASK':<{dec+6}}  {'SIZE':<8}")
        print("    " + "─" * (BAR * 2 + dec * 2 + 18))
        for ask_row, bid_row in zip(reversed(padded_asks), padded_bids, strict=False):
            a_px, a_sz = ask_row
            b_px, b_sz = bid_row
            b_bar = int(BAR * b_sz / max_sz) if b_sz > 0 else 0
            a_bar = int(BAR * a_sz / max_sz) if a_sz > 0 else 0
            b_str = (f"{b_sz:>6.2f} " + "\033[92m" + "▓" * b_bar + "░" * (BAR - b_bar) + "\033[0m") if b_px else " " * (BAR + 8)
            a_str = ("\033[91m" + "▓" * a_bar + "░" * (BAR - a_bar) + "\033[0m" + f" {a_sz:<6.2f}") if a_px else " " * (BAR + 8)
            b_px_str = f"{b_px:.{dec}f}" if b_px else "   —  "
            a_px_str = f"{a_px:.{dec}f}" if a_px else "   —  "
            print(f"    {b_str}  {b_px_str}  {a_px_str}  {a_str}")
        print("    " + "─" * (BAR * 2 + dec * 2 + 18))
        print(f"    Total: \033[92m{depth_bid:>8.2f}\033[0m              \033[91m{depth_ask:>8.2f}\033[0m")

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
            signals.append("\033[91m✗ Low feasibility — no new entries\033[0m")
        if toxic:
            signals.append("\033[91m✗ Toxic flow (VPIN) — stop widening advised\033[0m")
        if gate:
            signals.append("\033[91m✗ Depth gate active — no new entries\033[0m")
        if rs_regime == "TRENDING" and not toxic and rs_feas > FEASIBILITY_HIGH_THRESHOLD:
            dir_hint = "LONG" if imbalance > _IMBALANCE_DIRECTION_HINT else ("SHORT" if imbalance < -_IMBALANCE_DIRECTION_HINT else "either direction")
            signals.append(f"\033[92m✓ Trending + clean flow → favours {dir_hint}\033[0m")
        if rs_regime == "MEAN_REVERTING" and not toxic:
            signals.append("\033[93m⚡ Mean-reverting — shorter hold, tighter target\033[0m")
        if rs_runway < 1.0:
            signals.append("\033[93m⚡ Short runway — harvester may exit early\033[0m")
        if not signals:
            signals.append("\033[90m— No strong signals; model discretion applies\033[0m")
        for s in signals:
            print(f"    {s}")

    def _render_market(self):
        """Render market microstructure"""
        print("\n\033[1m🔬 MARKET MICROSTRUCTURE\033[0m\n")

        ms = self.market_stats

        # Spread analysis
        print("  \033[1m💹 SPREAD & LIQUIDITY\033[0m")
        spread = ms.get("spread", 0)
        depth_bid = ms.get("depth_bid", 0)
        depth_ask = ms.get("depth_ask", 0)
        bids = ms.get("order_book_bids", [])  # [[price, size], ...]
        asks = ms.get("order_book_asks", [])  # [[price, size], ...]
        dec = self._price_decimals(bids[0][0] if bids else asks[0][0] if asks else 0.0)

        print(f"    Bid-Ask Spread:    {spread:>12.{dec}f}")
        print()
        self._render_order_book_ladder(bids, asks, depth_bid, depth_ask, dec)
        print()

        # Order flow toxicity
        print("  \033[1m☢️  ORDER FLOW TOXICITY (VPIN)\033[0m")
        vpin_z = ms.get("vpin_z", 0)

        if abs(vpin_z) > VPIN_HIGH_TOXICITY_THRESHOLD:
            vpin_status = "\033[91m⚠️  HIGH TOXICITY\033[0m"
        elif abs(vpin_z) > VPIN_ELEVATED_TOXICITY_THRESHOLD:
            vpin_status = "\033[93m⚡ ELEVATED\033[0m"
        else:
            vpin_status = "\033[92m✓ NORMAL\033[0m"

        print(f"    VPIN Z-Score:      {vpin_z:>+12.2f}")
        print(f"    Status:            {vpin_status}")
        print("                       \033[90mHigh +z = informed sellers active → widen stops / reduce size\033[0m")

        # VPIN gauge
        bar_len = 40
        z_norm = max(0, min(1, (vpin_z + 3) / 6))
        pos = int(bar_len * z_norm)
        gauge = "░" * pos + "│" + "░" * (bar_len - pos - 1)
        print(f"\n    Z: [{gauge}]")
        print("       -3              0              +3")
        print()

        # Order imbalance
        print("  \033[1m⚖️  ORDER IMBALANCE (QFI)\033[0m")
        has_real_sizes = ms.get("has_real_sizes", False)
        qfi_updates = int(ms.get("qfi_update_count", 0))
        imbalance = ms.get("imbalance", 0.0)
        signal_source = "size-weighted" if has_real_sizes else "quote-flow (QFI)"

        if imbalance > IMBALANCE_BUY_THRESHOLD:
            imb_status = "\033[92m🔺 BUY PRESSURE\033[0m"
        elif imbalance < IMBALANCE_SELL_THRESHOLD:
            imb_status = "\033[91m🔻 SELL PRESSURE\033[0m"
        else:
            imb_status = "\033[93m⚖️  BALANCED\033[0m"

        print(f"    Imbalance:         {imbalance:>+12.4f}")
        print(f"    Signal source:     {signal_source}  (updates: {qfi_updates})")
        print(f"    Status:            {imb_status}")

        # Imbalance visualization
        mid = bar_len // 2
        imb_scaled = int(mid * imbalance)
        if imbalance >= 0:
            bar = " " * mid + "\033[92m" + "█" * imb_scaled + "\033[0m" + " " * (mid - imb_scaled)
        else:
            bar = " " * (mid + imb_scaled) + "\033[91m" + "█" * (-imb_scaled) + "\033[0m" + " " * mid
        print(f"\n    [{bar}]")
        print("     SELL              ↕              BUY")

        self._render_signal_synthesis(vpin_z, imbalance)

    def _render_footer(self):
        """Render footer with controls and data freshness"""
        print("\n" + "─" * 80)
        print("  [1-6] Tabs  |  [Tab] Next  |  [Shift+Tab] Prev  |  [h] Help  |  [s] Presets  |  [q/^X] Quit")

        # Data freshness — use order_book.json mtime (written ~1s by FIX handler)
        _ob = self.data_dir / "order_book.json"
        _bc = self.data_dir / "bot_config.json"
        _fp = _ob if _ob.exists() else (_bc if _bc.exists() else None)
        if _fp is not None:
            import os as _os2  # noqa: PLC0415
            _fage = time.time() - _os2.path.getmtime(_fp)
            if _fage > DATA_STALE_SECS:
                freshness = f"\033[91m⚠️  Bot silent ({_fage:.0f}s)\033[0m"
            elif _fage > DATA_AGING_SECS:
                freshness = f"\033[93m⚡ Data aging ({_fage:.0f}s)\033[0m"
            else:
                freshness = f"\033[92m✓ Data fresh ({_fage:.1f}s)\033[0m"
        else:
            freshness = "\033[90m⏳ Waiting for data...\033[0m"

        note = self._current_notification() or "Press 'h' for help and keyboard shortcuts."
        print(f"  {note}  |  {freshness}")
        print("─" * 80)

    def _pnl_color(self, pnl: float) -> str:
        """Return color code for PnL"""
        if pnl > 0:
            return "\033[92m"
        elif pnl < 0:
            return "\033[91m"
        return "\033[93m"

    def _create_sparkline(self, values: list) -> str:
        """Create a sparkline (mini-chart) from a list of values"""
        if not values:
            return ""

        # Sparkline characters from lowest to highest
        chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return chars[4] * len(values)  # Middle bar if all values same

        # Normalize and map to characters
        sparkline = ""
        for val in values:
            normalized = (val - min_val) / (max_val - min_val)
            idx = int(normalized * (len(chars) - 1))

            # Color positive values green, negative red
            if val > 0:
                sparkline += "\033[92m" + chars[idx] + "\033[0m"
            elif val < 0:
                sparkline += "\033[91m" + chars[idx] + "\033[0m"
            else:
                sparkline += "\033[93m" + chars[idx] + "\033[0m"

        return sparkline


def main():
    """Run tabbed HUD"""
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
        print("\n\nHUD stopped.")


if __name__ == "__main__":
    main()
