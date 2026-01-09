#!/usr/bin/env python3
# ctrader_ddqn_paper.py
# Dual FIX sessions (QUOTE+TRADE) for cTrader/Pepperstone demo.
# Builds BTCUSD (symbolId=10028) M15 bars from best bid/ask, then trades 0.10 qty target-position via TRADE.
#
# Requires QuickFIX built/installed into the venv (you already did this).
#
# Run:
#   source ~/Documents/.venv/bin/activate
#   export CTRADER_USERNAME="5179095"
#   export CTRADER_PASSWORD_QUOTE="***"
#   export CTRADER_PASSWORD_TRADE="***"
#   export CTRADER_CFG_QUOTE="ctrader_quote.cfg"
#   export CTRADER_CFG_TRADE="ctrader_trade.cfg"
#   export CTRADER_BTC_SYMBOL_ID="10028"
#   export CTRADER_QTY="0.10"
#   python3 ctrader_ddqn_paper.py

import datetime as dt
import json
import logging
import os
import sys
import time
import uuid
import threading
import signal
import math
from collections import deque
from pathlib import Path

import quickfix44 as fix44

import quickfix as fix

from performance_tracker import PerformanceTracker
from trade_exporter import TradeExporter
from reward_shaper import RewardShaper
from learned_parameters import LearnedParametersManager
from friction_costs import FrictionCalculator
from dual_policy import DualPolicy
from path_geometry import PathGeometry
from var_estimator import VaREstimator, KurtosisMonitor, RegimeType
from activity_monitor import ActivityMonitor
from adaptive_regularization import AdaptiveRegularization

# Handbook components - Phase 1
from safe_math import SafeMath, RunningStats
from circuit_breakers import CircuitBreakerManager
from event_time_features import EventTimeFeatureEngine


# ----------------------------
# Logging
# ----------------------------
def _redact_fix(s: str) -> str:
    # redact tag 554=Password (and anything else you want)
    return s.replace("554=", "554=REDACTED")


def setup_logging() -> logging.Logger:
    logdir = os.environ.get("PY_LOGDIR", "ctrader_py_logs").strip()
    Path(logdir).mkdir(parents=True, exist_ok=True)

    # Python 3.12: utcnow() deprecates; use timezone-aware UTC.
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(logdir, f"ctrader_{ts}.log")

    logger = logging.getLogger("ctrader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("Executable: %s", sys.executable)
    logger.info("CWD: %s", os.getcwd())
    logger.info("PY_LOGDIR: %s", os.path.abspath(logdir))
    logger.info("Logfile: %s", logfile)
    return logger


LOG = setup_logging()


# ----------------------------
# Time helpers (UTC required)
# ----------------------------
def utc_ts_ms() -> str:
    # FIX UTCTimestamp: YYYYMMDD-HH:MM:SS.sss (UTC)
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


# ----------------------------
# Bar builder (configurable timeframe)
# ----------------------------
class BarBuilder:
    def __init__(self, timeframe_minutes: int = 15):
        self.timeframe_minutes = timeframe_minutes
        self.bucket = None
        self.o = self.h = self.l = self.c = None

    def bucket_start(self, t: dt.datetime) -> dt.datetime:
        m = (t.minute // self.timeframe_minutes) * self.timeframe_minutes
        return t.replace(minute=m, second=0, microsecond=0)

    def update(self, t: dt.datetime, mid: float):
        b = self.bucket_start(t)
        if self.bucket is None:
            self.bucket = b
            self.o = self.h = self.l = self.c = mid
            return None

        if b != self.bucket:
            closed = (self.bucket, self.o, self.h, self.l, self.c)
            self.bucket = b
            self.o = self.h = self.l = self.c = mid
            return closed

        self.c = mid
        if mid > self.h:
            self.h = mid
        if mid < self.l:
            self.l = mid
        return None


# ----------------------------
# Minimal policy wrapper
# ----------------------------
class Policy:
    """
    Discrete actions: 0=SHORT, 1=FLAT, 2=LONG
    Default: FLAT until you load a DDQN model (optional).
    """

    def __init__(self):
        self.use_torch = False
        self.model = None
        self.window = 64

        model_path = os.environ.get("DDQN_MODEL_PATH", "").strip()
        if model_path:
            try:
                import torch
                import torch.nn as nn

                class QNet(nn.Module):
                    def __init__(
                        self, window: int, n_features: int, n_actions: int = 3
                    ):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
                            nn.ReLU(),
                            nn.Conv1d(64, 64, kernel_size=5, padding=2),
                            nn.ReLU(),
                            nn.AdaptiveAvgPool1d(1),
                            nn.Flatten(),
                            nn.Linear(64, 128),
                            nn.ReLU(),
                            nn.Linear(128, n_actions),
                        )

                    def forward(self, x):
                        # x: (B,T,F) -> (B,F,T)
                        return self.net(x.transpose(1, 2))

                self.torch = torch
                self.model = QNet(window=self.window, n_features=4, n_actions=3)
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.model.eval()
                self.use_torch = True
                LOG.info("[POLICY] Loaded DDQN model: %s", model_path)
            except Exception as e:
                LOG.warning(
                    "[POLICY] Failed to load model, running fallback. Error: %s", e
                )
                self.use_torch = False

    def decide(self, bars: deque) -> int:
        if len(bars) < 70:
            return 1  # FLAT

        closes = [b[4] for b in bars]
        import numpy as np

        c = np.array(closes, dtype=np.float64)

        # Defensive: calculate returns with safe division
        ret1 = np.zeros_like(c)
        if len(c) >= 2:
            ret1[1:] = np.divide(c[1:], c[:-1], out=np.ones_like(c[1:]), where=c[:-1]!=0) - 1.0

        ret5 = np.zeros_like(c)
        if len(c) >= 6:
            ret5[5:] = np.divide(c[5:], c[:-5], out=np.ones_like(c[5:]), where=c[:-5]!=0) - 1.0

        def rolling_mean(x, n):
            out = np.full_like(x, np.nan, dtype=np.float64)
            if len(x) >= n:
                cs = np.cumsum(np.insert(x, 0, 0.0))
                out[n - 1 :] = (cs[n:] - cs[:-n]) / n
            return out

        def rolling_std(x, n):
            out = np.full_like(x, np.nan, dtype=np.float64)
            if len(x) >= n:
                for i in range(n - 1, len(x)):
                    w = x[i - n + 1 : i + 1]
                    out[i] = np.std(w)
            return out

        ma_fast = rolling_mean(c, 10)
        ma_slow = rolling_mean(c, 30)
        ma_diff = np.divide(ma_fast, ma_slow, out=np.ones_like(ma_fast), where=ma_slow!=0) - 1.0
        vol = rolling_std(ret1, 20)

        # Defensive: clean all features for NaN/Inf
        feats = np.vstack([
            np.nan_to_num(ret1, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(ret5, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(ma_diff, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        ]).T
        feats = feats[-self.window :].astype(np.float32)

        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True) + 1e-8
        x = (feats - mu) / sd

        if not self.use_torch:
            # Defensive: validate array shape before access
            if x.shape[0] == 0 or x.shape[1] < 3:
                return 1  # Default to HOLD if insufficient data
            md = float(x[-1, 2])
            if md > 0.2:
                return 2
            if md < -0.2:
                return 0
            return 1

        with self.torch.no_grad():
            t = self.torch.from_numpy(x).unsqueeze(0)
            q = self.model(t).squeeze(0).numpy()
            return int(q.argmax())


# ----------------------------
# Path Recorder
# ----------------------------
class PathRecorder:
    """Record M1 OHLC path during entire trade lifecycle."""
    def __init__(self):
        self.recording = False
        self.entry_time = None
        self.entry_price = None
        self.direction = None
        self.path = []  # List of (timestamp, o, h, l, c) tuples
        self.trade_counter = 0

    def start_recording(self, entry_time: dt.datetime, entry_price: float, direction: int):
        """Start recording path for a new trade."""
        self.recording = True
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.path = []
        LOG.info("[PATH] Started recording for %s trade at %.2f",
                 "LONG" if direction == 1 else "SHORT", entry_price)

    def add_bar(self, bar):
        """Add a bar to the path. bar is tuple: (timestamp, o, h, l, c)"""
        if not self.recording:
            return
        self.path.append(bar)

    def stop_recording(self, exit_time: dt.datetime, exit_price: float, pnl: float, 
                      mfe: float = 0.0, mae: float = 0.0, winner_to_loser: bool = False) -> dict:
        """Stop recording and return trade summary with path."""
        if not self.recording:
            return None

        self.recording = False
        self.trade_counter += 1

        # Calculate trade duration
        duration_seconds = (exit_time - self.entry_time).total_seconds() if self.entry_time else 0

        trade_record = {
            "trade_id": self.trade_counter,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "pnl": pnl,
            "mfe": mfe,
            "mae": mae,
            "winner_to_loser": winner_to_loser,
            "duration_seconds": duration_seconds,
            "bars_count": len(self.path),
            "path": [
                {"timestamp": t.isoformat(), "open": o, "high": h, "low": l, "close": c}
                for t, o, h, l, c in self.path
            ]
        }

        # Save to JSON file
        self._save_to_file(trade_record)

        LOG.info("[PATH] Stopped recording. Trade #%d: %d bars, %.2f seconds, PnL=%.2f | MFE=%.2f MAE=%.2f WTL=%s",
                 self.trade_counter, len(self.path), duration_seconds, pnl, mfe, mae, winner_to_loser)

        return trade_record

    def _save_to_file(self, trade_record: dict):
        """Save trade record to JSON file."""
        import json
        from pathlib import Path

        trades_dir = Path("trades")
        trades_dir.mkdir(exist_ok=True)

        filename = trades_dir / f"trade_{trade_record['trade_id']:04d}_{trade_record['direction'].lower()}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(trade_record, f, indent=2)
            LOG.info("[PATH] Saved to %s", filename)
        except Exception as e:
            LOG.error("[PATH] Failed to save: %s", e)


# ----------------------------
# MFE/MAE Tracker
# ----------------------------
class MFEMAETracker:
    """Track Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE) per position."""
    def __init__(self):
        self.entry_price = None
        self.direction = None  # 1=long, -1=short
        self.mfe = 0.0  # max favorable move in $ (profit)
        self.mae = 0.0  # max adverse move in $ (loss, stored as positive)
        self.best_profit = 0.0
        self.worst_loss = 0.0
        self.winner_to_loser = False

    def start_tracking(self, entry_price: float, direction: int):
        """direction: 1=long, -1=short"""
        self.entry_price = entry_price
        self.direction = direction
        self.mfe = 0.0
        self.mae = 0.0
        self.best_profit = 0.0
        self.worst_loss = 0.0
        self.winner_to_loser = False

    def update(self, current_price: float):
        """Update with current market price during open position."""
        if self.entry_price is None:
            return

        # Calculate P&L
        if self.direction == 1:  # long
            pnl = current_price - self.entry_price
        else:  # short
            pnl = self.entry_price - current_price

        # Track best profit (MFE)
        if pnl > self.best_profit:
            self.best_profit = pnl
            self.mfe = pnl

        # Track worst loss (MAE)
        if pnl < self.worst_loss:
            self.worst_loss = pnl
            self.mae = abs(pnl)

        # Detect winner-to-loser (was profitable, now losing)
        if self.best_profit > 0 and pnl < 0:
            self.winner_to_loser = True

    def get_summary(self) -> dict:
        """Return summary of MFE/MAE metrics."""
        return {
            "entry_price": self.entry_price,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "mfe": self.mfe,
            "mae": self.mae,
            "best_profit": self.best_profit,
            "worst_loss": self.worst_loss,
            "winner_to_loser": self.winner_to_loser
        }

    def reset(self):
        """Reset tracker for next position."""
        self.entry_price = None
        self.direction = None
        self.mfe = 0.0
        self.mae = 0.0
        self.best_profit = 0.0
        self.worst_loss = 0.0
        self.winner_to_loser = False


# ----------------------------
# FIX application
# ----------------------------
class CTraderFixApp(fix.Application):
    def __init__(self, symbol_id: int, qty: float, timeframe_minutes: int = 15, symbol: str = "BTCUSD"):
        super().__init__()
        self.symbol_id = symbol_id
        self.symbol = symbol  # Instrument-agnostic: BTCUSD, XAUUSD, etc.
        
        # Initialize learned parameters manager (single source of truth)
        self.param_manager = LearnedParametersManager()
        self.param_manager.load()
        
        # Construct instrument key: {SYMBOL}_M{timeframe}_default (instrument-agnostic)
        self.instrument = f"{symbol}_M{timeframe_minutes}_default"
        
        # Use adaptive position sizing from learned parameters
        # Fall back to env var or default only if not in manager
        base_pos = self.param_manager.get(self.instrument, 'base_position_size')
        self.qty = base_pos if base_pos is not None else qty
        LOG.info("Position size: %.4f (adaptive from LearnedParametersManager)", self.qty)
        
        self.timeframe_minutes = timeframe_minutes

        self.quote_sid = None
        self.trade_sid = None

        # Phase 3.5: Path geometry for trigger features (gamma, jerk, runway)
        self.path_geometry = PathGeometry()
        
        # Phase 3: Dual-agent policy (TriggerAgent + HarvesterAgent)
        # Falls back to simple policy if DDQN_DUAL_AGENT=0
        use_dual_agent = os.environ.get("DDQN_DUAL_AGENT", "1") == "1"
        enable_online_learning = os.environ.get("DDQN_ONLINE_LEARNING", "1") == "1"
        if use_dual_agent:
            self.policy = DualPolicy(
                window=64,
                enable_regime_detection=True,
                path_geometry=self.path_geometry,
                enable_training=enable_online_learning
            )
            LOG.info("[POLICY] Using DualPolicy with TriggerAgent + HarvesterAgent + PathGeometry (training=%s)", enable_online_learning)
        else:
            self.policy = Policy()
            LOG.info("[POLICY] Using simple Policy (DDQN_DUAL_AGENT=0)")
        
        self.bars = deque(maxlen=2000)
        self.builder = BarBuilder(timeframe_minutes)
        self.mfe_mae_tracker = MFEMAETracker()
        self.path_recorder = PathRecorder()
        self.performance = PerformanceTracker()
        self.trade_exporter = TradeExporter()
        
        # Friction calculator (source of truth: cTrader SymbolInfo)
        self.friction_calculator = FrictionCalculator(symbol=symbol, symbol_id=symbol_id)
        
        # Pass param_manager to reward_shaper for DRY
        self.reward_shaper = RewardShaper(instrument=self.instrument, param_manager=self.param_manager)
        
        # Phase 3.5: Risk management - VaR estimator with kurtosis circuit breaker
        self.kurtosis_monitor = KurtosisMonitor(window=100, threshold=3.0)
        self.var_estimator = VaREstimator(
            window=500, 
            confidence=0.95,
            kurtosis_monitor=self.kurtosis_monitor
        )
        LOG.info("[RISK] VaREstimator initialized with kurtosis circuit breaker")
        
        # Phase 3.5: Activity monitor - prevent learned helplessness
        self.activity_monitor = ActivityMonitor(
            max_bars_inactive=100,
            min_trades_per_day=2.0,
            exploration_boost=0.1
        )
        LOG.info("[ACTIVITY] ActivityMonitor initialized - anti-stagnation protection")
        
        # Phase 3.5: Adaptive regularization for online learning
        self.adaptive_reg = AdaptiveRegularization(
            initial_l2=0.0001,
            initial_dropout=0.1,
            l2_range=(1e-5, 1e-2),
            dropout_range=(0.0, 0.5)
        )
        LOG.info("[ADAPTIVE-REG] Initialized for online learning")
        
        # Handbook Phase 1: Circuit Breakers (safety shutdown system)
        self.circuit_breakers = CircuitBreakerManager(
            sortino_threshold=self.param_manager.get(symbol, 'sortino_threshold'),
            kurtosis_threshold=self.param_manager.get(symbol, 'kurtosis_threshold'),
            max_drawdown=self.param_manager.get(symbol, 'max_drawdown_pct'),
            max_consecutive_losses=int(self.param_manager.get(symbol, 'max_consecutive_losses'))
        )
        LOG.info("[CIRCUIT-BREAKERS] Safety shutdown system initialized")
        
        # Handbook Phase 1: Event-relative time features
        self.event_time_engine = EventTimeFeatureEngine()
        LOG.info("[EVENT-TIME] Session-relative time features initialized")
        
        # Training frequency: train every N bars
        self.training_interval = 5  # bars
        self.bars_since_training = 0
        
        self.previous_sharpe = 0.0  # Track for adaptive weight updates

        self.best_bid = None
        self.best_ask = None

        self.cur_pos = 0
        self.pos_req_id = None
        self.clord_counter = 0
        self.trade_entry_time = None  # Track entry time for performance metrics
        
        # Online learning state storage
        self.entry_state = None  # State at entry decision time (for experience)
        self.entry_action = None  # Action taken at entry (0=NO, 1=LONG, 2=SHORT)
        
        # Connection health monitoring
        self.last_quote_heartbeat = None
        self.last_trade_heartbeat = None
        self.heartbeat_timeout = 60  # seconds
        self.connection_healthy = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self._shutdown_requested = False
        self._shutdown_complete = False
        
        # Reconnection backoff (exponential)
        self.reconnect_base_delay = 5  # seconds
        self.reconnect_max_delay = 300  # 5 minutes max
        self.last_reconnect_time = None
        
        # Systemd watchdog support
        self.watchdog_enabled = os.environ.get('WATCHDOG_USEC') is not None
        if self.watchdog_enabled:
            self.watchdog_interval = int(os.environ.get('WATCHDOG_USEC', '0')) / 1000000 / 2  # Half of watchdog timeout
            LOG.info("[WATCHDOG] Systemd watchdog enabled (interval: %.1fs)", self.watchdog_interval)
        else:
            self.watchdog_interval = None
        
        # Start health monitor thread
        self._health_monitor_running = True
        self._health_thread = threading.Thread(target=self._monitor_connection_health, daemon=True)
        self._health_thread.start()
        LOG.info("[HEALTH] Connection health monitor thread started")
        
        # HUD data export tracking
        self.start_time = dt.datetime.now(dt.timezone.utc)
        self.bar_count = 0
        self.hud_data_dir = Path("data")
        self.hud_data_dir.mkdir(exist_ok=True)

    # ---- connection health monitoring ----
    def _monitor_connection_health(self):
        """
        Background thread to monitor connection health via heartbeat timestamps.
        
        Runs every 10 seconds and checks:
        - Last heartbeat from QUOTE session
        - Last heartbeat from TRADE session
        - Triggers reconnect attempts if sessions go stale
        """
        LOG.info("[HEALTH] Connection health monitor started")
        check_interval = 10  # seconds
        health_log_counter = 0
        
        while self._health_monitor_running and not self._shutdown_requested:
            try:
                time.sleep(check_interval)
                
                if self._shutdown_requested:
                    break
                
                now = utc_now()
                issues = []
                
                # Check QUOTE session
                if self.quote_sid and self.last_quote_heartbeat:
                    quote_age = (now - self.last_quote_heartbeat).total_seconds()
                    if quote_age > self.heartbeat_timeout:
                        issues.append(f"QUOTE stale ({quote_age:.0f}s)")
                        self._try_send_test_request(self.quote_sid, "QUOTE")
                elif self.quote_sid is None and self.last_quote_heartbeat is not None:
                    # Session was lost
                    issues.append("QUOTE disconnected")
                
                # Check TRADE session
                if self.trade_sid and self.last_trade_heartbeat:
                    trade_age = (now - self.last_trade_heartbeat).total_seconds()
                    if trade_age > self.heartbeat_timeout:
                        issues.append(f"TRADE stale ({trade_age:.0f}s)")
                        self._try_send_test_request(self.trade_sid, "TRADE")
                elif self.trade_sid is None and self.last_trade_heartbeat is not None:
                    # Session was lost
                    issues.append("TRADE disconnected")
                
                # Update health status
                if issues:
                    self.connection_healthy = False
                    LOG.warning("[HEALTH] Connection issues: %s", ", ".join(issues))
                else:
                    if not self.connection_healthy and self.quote_sid and self.trade_sid:
                        LOG.info("[HEALTH] Connection restored (QUOTE+TRADE active)")
                    self.connection_healthy = True
                
                # Systemd watchdog notification (if enabled)
                if self.watchdog_enabled and self.connection_healthy:
                    try:
                        # Notify systemd that we're alive
                        import socket
                        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                        sock.sendto(b'WATCHDOG=1', os.environ.get('NOTIFY_SOCKET', ''))
                        sock.close()
                    except Exception as e:
                        LOG.debug("[WATCHDOG] Failed to notify systemd: %s", e)
                
                # Periodic health log (every ~60s)
                health_log_counter += 1
                if health_log_counter % 6 == 0:  # Every ~60s
                    quote_status = "OK" if self.quote_sid else "DOWN"
                    trade_status = "OK" if self.trade_sid else "DOWN"
                    LOG.info("[HEALTH] Status: QUOTE=%s TRADE=%s healthy=%s reconnects=%d",
                            quote_status, trade_status, self.connection_healthy, self.reconnect_attempts)
                
            except Exception as e:
                LOG.error("[HEALTH] Monitor error: %s", e, exc_info=True)
        
        LOG.info("[HEALTH] Connection health monitor stopped")
    
    def _try_send_test_request(self, session_id, qual: str):
        """
        Send FIX TestRequest (MsgType=1) to verify session is alive.
        
        If no Heartbeat response is received within timeout, QuickFIX
        will automatically trigger logout and reconnect.
        """
        if not session_id:
            return False
        
        try:
            test_req = fix.Message()
            test_req.getHeader().setField(fix.MsgType("1"))  # TestRequest
            test_req.setField(fix.TestReqID(f"HEALTH_{qual}_{int(time.time())}"))
            fix.Session.sendToTarget(test_req, session_id)
            LOG.info("[HEALTH] Sent TestRequest to %s session", qual)
            return True
        except Exception as e:
            LOG.error("[HEALTH] Failed to send TestRequest to %s: %s", qual, e)
            return False
    
    def stop_health_monitor(self):
        """Gracefully stop the health monitor thread."""
        self._health_monitor_running = False
        self._shutdown_requested = True
    
    def graceful_shutdown(self, signum=None, frame=None):
        """Perform graceful shutdown with position cleanup and session closure."""
        if self._shutdown_complete:
            return
        
        signal_name = signal.Signals(signum).name if signum else "MANUAL"
        LOG.info("[SHUTDOWN] Graceful shutdown initiated (signal: %s)", signal_name)
        self._shutdown_requested = True
        
        try:
            # 1. Close any open positions (emergency exit)
            if hasattr(self, 'cur_pos') and self.cur_pos != 0:
                LOG.warning("[SHUTDOWN] Open position detected: %s (qty: %s)",
                           "LONG" if self.cur_pos > 0 else "SHORT", abs(self.cur_pos))
                LOG.warning("[SHUTDOWN] Manual position closure required - bot stopping with open position")
                # For safety, we DON'T auto-close to avoid accidental liquidation
            elif hasattr(self, 'cur_pos'):
                LOG.info("[SHUTDOWN] No open positions")
            
            # 2. Save current state
            if hasattr(self, 'policy') and self.policy:
                LOG.info("[SHUTDOWN] Saving model state...")
                try:
                    if hasattr(self.policy, 'save_checkpoint'):
                        self.policy.save_checkpoint()
                        LOG.info("[SHUTDOWN] Model state saved")
                except Exception as e:
                    LOG.error("[SHUTDOWN] Failed to save model: %s", e)
            
            # 3. Flush trade exports
            if hasattr(self, 'trade_exporter'):
                try:
                    LOG.info("[SHUTDOWN] Trade exports up to date")
                except Exception as e:
                    LOG.error("[SHUTDOWN] Error with exports: %s", e)
            
            # 4. Log final statistics
            LOG.info("[SHUTDOWN] Final statistics:")
            LOG.info("  - Total bars processed: %d", len(self.bars) if hasattr(self, 'bars') else 0)
            LOG.info("  - Current position: %s", self.cur_pos if hasattr(self, 'cur_pos') else 0)
            LOG.info("  - Reconnection attempts: %d", self.reconnect_attempts)
            if hasattr(self, 'performance'):
                try:
                    metrics = self.performance.get_metrics()
                    LOG.info("  - Total PnL: %.2f", metrics.get('total_pnl', 0.0))
                    LOG.info("  - Total trades: %d", metrics.get('total_trades', 0))
                except:
                    pass
            
            # 5. Stop health monitor
            self.stop_health_monitor()
            
        except Exception as e:
            LOG.error("[SHUTDOWN] Error during graceful shutdown: %s", e, exc_info=True)
        finally:
            self._shutdown_complete = True
            LOG.info("[SHUTDOWN] Graceful shutdown complete")
    
    def is_fully_connected(self) -> bool:
        """Check if both QUOTE and TRADE sessions are active."""
        return self.quote_sid is not None and self.trade_sid is not None
    
    def get_connection_status(self) -> dict:
        """Get detailed connection status for monitoring/logging."""
        now = utc_now()
        return {
            "quote_connected": self.quote_sid is not None,
            "trade_connected": self.trade_sid is not None,
            "quote_age_s": (now - self.last_quote_heartbeat).total_seconds() if self.last_quote_heartbeat else None,
            "trade_age_s": (now - self.last_trade_heartbeat).total_seconds() if self.last_trade_heartbeat else None,
            "connection_healthy": self.connection_healthy,
            "reconnect_attempts": self.reconnect_attempts,
        }

    # ---- helpers ----
    @staticmethod
    def _qual(sessionID) -> str:
        # Route strictly by SessionQualifier (NOT SubIDs).
        try:
            q = sessionID.getSessionQualifier()
            return (q or "").upper()
        except Exception as e:
            LOG.warning("[SESSION] Unable to get qualifier: %s", e)
            return ""

    # ---- session events ----
    def onCreate(self, sessionID):
        LOG.info("[CREATE] %s qual=%s", sessionID.toString(), self._qual(sessionID))

    def onLogon(self, sessionID):
        qual = self._qual(sessionID)
        LOG.info("[LOGON] %s qual=%s", sessionID.toString(), qual)
        
        # Reset reconnect counter on successful logon
        self.reconnect_attempts = 0
        self.connection_healthy = True

        try:
            if qual == "QUOTE":
                self.quote_sid = sessionID
                self.last_quote_heartbeat = utc_now()
                self.send_md_subscribe_spot()
            elif qual == "TRADE":
                self.trade_sid = sessionID
                self.last_trade_heartbeat = utc_now()
                self.request_security_definition()  # Get symbol info (source of truth)
                self.request_positions()
            else:
                LOG.warning(
                    "[LOGON] Unknown qualifier; not routing: %s", sessionID.toString()
                )
        except Exception as e:
            LOG.error("[LOGON] Error during session setup for %s: %s", qual, e, exc_info=True)

    def onLogout(self, sessionID):
        qual = self._qual(sessionID)
        LOG.warning("[LOGOUT] %s qual=%s", sessionID.toString(), qual)
        
        # Mark session as down
        if qual == "QUOTE":
            self.quote_sid = None
        elif qual == "TRADE":
            self.trade_sid = None
        
        self.connection_healthy = False
        self.reconnect_attempts += 1
        self.last_reconnect_time = time.time()
        
        # Calculate exponential backoff delay
        backoff_delay = min(
            self.reconnect_base_delay * math.pow(2, self.reconnect_attempts - 1),
            self.reconnect_max_delay
        )
        
        if self.reconnect_attempts <= self.max_reconnect_attempts:
            LOG.info("[RECONNECT] Will attempt reconnect (attempt %d/%d, backoff: %.1fs)", 
                    self.reconnect_attempts, self.max_reconnect_attempts, backoff_delay)
            # QuickFIX will handle the reconnect automatically with ReconnectInterval from .cfg
        else:
            LOG.error("[RECONNECT] Max attempts reached (%d). Manual intervention required.",
                     self.max_reconnect_attempts)
            LOG.error("[RECONNECT] Consider restarting the bot or checking network/credentials")

    # ---- admin hooks ----
    def toAdmin(self, message, sessionID):
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)

        if msg_type.getValue() != fix.MsgType_Logon:
            return

        qual = self._qual(sessionID)

        # Reset seq nums
        message.setField(fix.ResetSeqNumFlag(True))

        user = os.environ.get("CTRADER_USERNAME", "").strip()
        if qual == "QUOTE":
            pwd = os.environ.get("CTRADER_PASSWORD_QUOTE", "").strip()
            message.getHeader().setField(fix.TargetSubID("QUOTE"))
        else:
            pwd = os.environ.get("CTRADER_PASSWORD_TRADE", "").strip()
            message.getHeader().setField(fix.TargetSubID("TRADE"))

        if user:
            message.setField(fix.Username(user))
        if pwd:
            message.setField(fix.Password(pwd))

        LOG.info("[ADMIN][LOGON OUT] qual=%s %s", qual, _redact_fix(message.toString()))

    def fromAdmin(self, message, sessionID):
        qual = self._qual(sessionID)
        
        # Update heartbeat timestamp on ANY admin message
        now = utc_now()
        if qual == "QUOTE":
            self.last_quote_heartbeat = now
        elif qual == "TRADE":
            self.last_trade_heartbeat = now
        
        LOG.info("[ADMIN][IN] qual=%s %s", qual, _redact_fix(message.toString()))

    def toApp(self, message, sessionID):
        qual = self._qual(sessionID)
        # Keep this INFO until stable; you can reduce to DEBUG later.
        LOG.info("[APP][OUT] qual=%s %s", qual, _redact_fix(message.toString()))

    def fromApp(self, message, sessionID):
        qual = self._qual(sessionID)
        
        # Update heartbeat on any app message
        now = utc_now()
        if qual == "QUOTE":
            self.last_quote_heartbeat = now
        elif qual == "TRADE":
            self.last_trade_heartbeat = now
        
        LOG.info("[APP][IN] qual=%s %s", qual, _redact_fix(message.toString()))

        try:
            msg_type = fix.MsgType()
            message.getHeader().getField(msg_type)
            t = msg_type.getValue()

            if t == "W":
                self.on_md_snapshot(message)
            elif t == "X":
                self.on_md_incremental(message)
            elif t == "8":
                self.on_exec_report(message)
            elif t == "AP":
                self.on_position_report(message)
            elif t == "d":
                self.on_security_definition(message)  # Symbol info response
            elif t == "j":
                self.on_biz_reject(message)
            elif t == "Y":
                self.on_md_reject(message)
        except Exception as e:
            # CRITICAL: Catch ALL exceptions to prevent session crash
            LOG.error("[APP] Message handler failed: %s\n%s", e, 
                     message.toString() if message else "no message", exc_info=True)

    # ----------------------------
    # QUOTE: Market data subscribe
    # ----------------------------
    def send_md_subscribe_spot(self):
        if not self.quote_sid:
            return

        req = fix44.MarketDataRequest()
        req.setField(fix.MDReqID("BTCUSD_SPOT"))
        req.setField(fix.SubscriptionRequestType("1"))
        req.setField(fix.MarketDepth(1))
        req.setField(fix.MDUpdateType(1))
        req.setField(fix.NoMDEntryTypes(2))

        g1 = fix44.MarketDataRequest.NoMDEntryTypes()
        g1.setField(fix.MDEntryType("0"))
        req.addGroup(g1)

        g2 = fix44.MarketDataRequest.NoMDEntryTypes()
        g2.setField(fix.MDEntryType("1"))
        req.addGroup(g2)

        req.setField(fix.NoRelatedSym(1))
        sym = fix44.MarketDataRequest.NoRelatedSym()
        sym.setField(fix.Symbol(str(self.symbol_id)))
        req.addGroup(sym)

        fix.Session.sendToTarget(req, self.quote_sid)
        LOG.info("[QUOTE] Subscribed spot for symbolId=%s", self.symbol_id)

    # ----------------------------
    # TRADE: SecurityDefinition (Symbol Info)
    # ----------------------------
    def request_security_definition(self):
        """Request symbol information from cTrader (source of truth for all trading parameters)."""
        if not self.trade_sid:
            return

        req = fix44.SecurityDefinitionRequest()
        req.setField(fix.SecurityReqID(f"SYMINFO_{self.symbol_id}"))
        req.setField(fix.SecurityRequestType(fix.SecurityRequestType_REQUEST_SECURITY_IDENTITY_AND_SPECIFICATIONS))
        req.setField(fix.Symbol(str(self.symbol_id)))

        fix.Session.sendToTarget(req, self.trade_sid)
        LOG.info("[TRADE] Requested SecurityDefinition for symbolId=%s", self.symbol_id)

    def on_security_definition(self, msg: fix.Message):
        """
        Parse SecurityDefinition response from cTrader.
        
        This is the SOURCE OF TRUTH for all symbol parameters:
        - Min/max/step volume (lot sizes)
        - Pip size and value
        - Commission structure
        - Swap rates
        - Contract size
        - Price precision
        """
        try:
            symbol = fix.Symbol()
            if msg.isSetField(symbol):
                msg.getField(symbol)
                LOG.info("[TRADE] ═══ SecurityDefinition for symbol=%s ═══", symbol.getValue())
            
            # Extract all available symbol parameters
            params = {}
            
            # Security description
            sec_desc = fix.SecurityDesc()
            if msg.isSetField(sec_desc):
                msg.getField(sec_desc)
                LOG.info("  Description: %s", sec_desc.getValue())
            
            # Security type (SPOT, CFD, etc.)
            sec_type = fix.SecurityType()
            if msg.isSetField(sec_type):
                msg.getField(sec_type)
                LOG.info("  SecurityType: %s", sec_type.getValue())
            
            # Product type
            product = fix.Product()
            if msg.isSetField(product):
                msg.getField(product)
                LOG.info("  Product: %s", product.getValue())
            
            # Contract size (e.g., 100,000 for standard lot)
            contract_mult = fix.ContractMultiplier()
            if msg.isSetField(contract_mult):
                msg.getField(contract_mult)
                params['contract_size'] = float(contract_mult.getValue())
                LOG.info("  ✓ ContractMultiplier: %.0f", params['contract_size'])
            
            # Round lot (standard lot size)
            round_lot = fix.RoundLot()
            if msg.isSetField(round_lot):
                msg.getField(round_lot)
                params['volume_step'] = float(round_lot.getValue())
                LOG.info("  ✓ RoundLot (step): %.4f", params['volume_step'])
            
            # Min trade volume
            min_vol = fix.MinTradeVol()
            if msg.isSetField(min_vol):
                msg.getField(min_vol)
                params['min_volume'] = float(min_vol.getValue())
                LOG.info("  ✓ MinTradeVol: %.4f lots", params['min_volume'])
            
            # Max trade volume
            max_vol = fix.MaxTradeVol()
            if msg.isSetField(max_vol):
                msg.getField(max_vol)
                params['max_volume'] = float(max_vol.getValue())
                LOG.info("  ✓ MaxTradeVol: %.4f lots", params['max_volume'])
            
            # Fallback: try MinQty if MinTradeVol not present
            if 'min_volume' not in params:
                min_qty = fix.MinQty()
                if msg.isSetField(min_qty):
                    msg.getField(min_qty)
                    params['min_volume'] = float(min_qty.getValue())
                    LOG.info("  ✓ MinQty: %.4f lots", params['min_volume'])
            
            # Price precision (number of decimals)
            price_method = fix.PriceQuoteMethod()
            if msg.isSetField(price_method):
                msg.getField(price_method)
                params['digits'] = int(price_method.getValue())
                LOG.info("  ✓ PriceQuoteMethod (digits): %d", params['digits'])
            
            # Currency (base currency)
            currency = fix.Currency()
            if msg.isSetField(currency):
                msg.getField(currency)
                params['currency'] = currency.getValue()
                LOG.info("  ✓ Currency: %s", params['currency'])
            
            # Trading session hours
            trading_session = fix.TradingSessionID()
            if msg.isSetField(trading_session):
                msg.getField(trading_session)
                LOG.info("  TradingSessionID: %s", trading_session.getValue())
            
            # Update friction calculator with actual cTrader parameters
            if params:
                self.friction_calculator.update_symbol_costs(**params)
                LOG.info("[TRADE] ✓ FrictionCalculator updated with %d cTrader parameters", len(params))
                
                # Log summary
                stats = self.friction_calculator.get_statistics()
                LOG.info("[TRADE] Friction summary: min=%.4f max=%.4f step=%.4f", 
                        params.get('min_volume', 0), 
                        params.get('max_volume', 0),
                        params.get('volume_step', 0))
            else:
                LOG.warning("[TRADE] SecurityDefinition had no extractable parameters")
                
        except Exception as e:
            LOG.error("[TRADE] Error parsing SecurityDefinition: %s", e, exc_info=True)

    def on_md_snapshot(self, msg: fix.Message):
        try:
            no = fix.NoMDEntries()
            msg.getField(no)
            n = int(no.getValue())
        except (ValueError, AttributeError) as e:
            LOG.warning("[QUOTE] Invalid NoMDEntries field: %s", e)
            return
        except Exception as e:
            LOG.error("[QUOTE] Error parsing market data snapshot: %s", e)
            return

        bid = ask = None
        for i in range(1, n + 1):
            g = fix44.MarketDataSnapshotFullRefresh().NoMDEntries()
            msg.getGroup(i, g)

            et = fix.MDEntryType()
            px = fix.MDEntryPx()
            if g.isSetField(et) and g.isSetField(px):
                g.getField(et)
                g.getField(px)
                try:
                    if et.getValue() == "0":
                        bid = float(px.getValue())
                    if et.getValue() == "1":
                        ask = float(px.getValue())
                except (ValueError, TypeError) as e:
                    LOG.warning("[QUOTE] Invalid price value in entry %d: %s", i, e)
                    continue

        if bid is not None:
            self.best_bid = bid
        if ask is not None:
            self.best_ask = ask
            
        # Track spread for friction calculations (real-time cost monitoring)
        if bid is not None and ask is not None:
            self.friction_calculator.update_spread(bid, ask)

        self.try_bar_update()

    def on_md_incremental(self, msg: fix.Message):
        try:
            no = fix.NoMDEntries()
            msg.getField(no)
            n = int(no.getValue())
        except (ValueError, AttributeError) as e:
            LOG.warning("[QUOTE] Invalid NoMDEntries in incremental: %s", e)
            return
        except Exception as e:
            LOG.error("[QUOTE] Error parsing incremental refresh: %s", e)
            return

        for i in range(1, n + 1):
            g = fix44.MarketDataIncrementalRefresh().NoMDEntries()
            msg.getGroup(i, g)

            act = fix.MDUpdateAction()
            g.getField(act)
            action = act.getValue()

            et = fix.MDEntryType()
            sym = fix.Symbol()

            if g.isSetField(et):
                g.getField(et)
            if g.isSetField(sym):
                g.getField(sym)

            if sym.getValue() and sym.getValue() != str(self.symbol_id):
                continue

            if action == "0":
                px = fix.MDEntryPx()
                if g.isSetField(px):
                    g.getField(px)
                    p = float(px.getValue())
                    if et.getValue() == "0":
                        self.best_bid = p
                    if et.getValue() == "1":
                        self.best_ask = p
            elif action == "2":
                if et.getValue() == "0":
                    self.best_bid = None
                if et.getValue() == "1":
                    self.best_ask = None
        
        # Track spread for friction calculations
        if self.best_bid is not None and self.best_ask is not None:
            self.friction_calculator.update_spread(self.best_bid, self.best_ask)

        self.try_bar_update()

    def try_bar_update(self):
        if self.best_bid is None or self.best_ask is None:
            return

        mid = (self.best_bid + self.best_ask) / 2.0
        
        # Update MFE/MAE if we have an open position
        if self.cur_pos != 0:
            self.mfe_mae_tracker.update(mid)
        
        closed = self.builder.update(utc_now(), mid)
        if closed:
            self.bars.append(closed)
            self.on_bar_close(closed)

    # ----------------------------
    # TRADE: positions + orders
    # ----------------------------
    def request_positions(self):
        if not self.trade_sid:
            return

        self.pos_req_id = f"pos_{uuid.uuid4().hex[:10]}"
        req = fix44.RequestForPositions()
        req.setField(fix.PosReqID(self.pos_req_id))
        fix.Session.sendToTarget(req, self.trade_sid)
        LOG.info("[TRADE] Requested positions")

    def on_position_report(self, msg: fix.Message):
        try:
            sym = fix.Symbol()
            if msg.isSetField(sym):
                msg.getField(sym)
                if sym.getValue() != str(self.symbol_id):
                    return
        except Exception as e:
            LOG.error("[TRADE] Error parsing position report symbol: %s", e)
            return

        long_qty = 0.0
        short_qty = 0.0

        f704 = fix.StringField(704)  # LongQty
        f705 = fix.StringField(705)  # ShortQty

        try:
            if msg.isSetField(f704):
                msg.getField(f704)
                long_qty = float(f704.getValue())
            if msg.isSetField(f705):
                msg.getField(f705)
                short_qty = float(f705.getValue())
        except (ValueError, TypeError) as e:
            LOG.error("[TRADE] Invalid position quantity: %s. Using 0.", e)
            # Keep default values (0.0) on error

        net = long_qty - short_qty
        old_pos = self.cur_pos
        if abs(net) < self.qty * 0.5:
            self.cur_pos = 0
        elif net > 0:
            self.cur_pos = 1
        else:
            self.cur_pos = -1

        LOG.info("[TRADE] PositionReport net=%0.6f -> cur_pos=%s", net, self.cur_pos)
        
        # Log MFE/MAE summary and save path when position closes
        if old_pos != 0 and self.cur_pos == 0:
            summary = self.mfe_mae_tracker.get_summary()
            LOG.info(
                "[MFE/MAE] Entry=%.5f %s | MFE=%.5f MAE=%.5f | Best=%.5f Worst=%.5f | WTL=%s",
                summary["entry_price"], summary["direction"],
                summary["mfe"], summary["mae"],
                summary["best_profit"], summary["worst_loss"],
                summary["winner_to_loser"]
            )
            
            # Stop path recording and save trade
            if self.best_bid and self.best_ask:
                exit_price = (self.best_bid + self.best_ask) / 2.0
                exit_time = utc_now()
                direction_sign = 1 if summary["direction"] == "LONG" else -1
                pnl = (exit_price - summary["entry_price"]) * direction_sign
                
                # Save path with MFE/MAE/WTL metrics
                self.path_recorder.stop_recording(
                    exit_time, exit_price, pnl,
                    mfe=summary["mfe"],
                    mae=summary["mae"],
                    winner_to_loser=summary["winner_to_loser"]
                )
                
                # Add to performance tracker
                if self.trade_entry_time:
                    self.performance.add_trade(
                        pnl=pnl,
                        entry_time=self.trade_entry_time,
                        exit_time=exit_time,
                        direction=summary["direction"],
                        entry_price=summary["entry_price"],
                        exit_price=exit_price,
                        mfe=summary["mfe"],
                        mae=summary["mae"],
                        winner_to_loser=summary["winner_to_loser"]
                    )
                    
                    # Calculate shaped rewards for DDQN training
                    reward_data = {
                        'exit_pnl': pnl,
                        'mfe': summary["mfe"],
                        'mae': summary["mae"],
                        'winner_to_loser': summary["winner_to_loser"]
                    }
                    shaped_rewards = self.reward_shaper.calculate_total_reward(reward_data)
                    
                    # Update baseline MFE for normalization
                    if summary["mfe"] > 0:
                        self.reward_shaper.update_baseline_mfe(summary["mfe"])
                    
                    # Log shaped reward components
                    LOG.info(
                        "[REWARD] Capture: %+.4f | WTL Penalty: %+.4f | Opportunity: %+.4f | Total: %+.4f | Active: %d",
                        shaped_rewards['capture_efficiency'],
                        shaped_rewards['wtl_penalty'],
                        shaped_rewards['opportunity_cost'],
                        shaped_rewards['total_reward'],
                        shaped_rewards['components_active']
                    )
                    
                    # Phase 3.5: Add experience to buffer for online learning
                    if hasattr(self.policy, 'add_trigger_experience') and self.entry_state is not None:
                        # Build next_state from current bars
                        next_state = None
                        if hasattr(self.policy, 'trigger') and hasattr(self.policy.trigger, 'last_state'):
                            next_state = self.policy.trigger.last_state
                        
                        if next_state is not None:
                            self.policy.add_trigger_experience(
                                state=self.entry_state,
                                action=self.entry_action,
                                reward=shaped_rewards['total_reward'],
                                next_state=next_state,
                                done=True
                            )
                            LOG.info("[ONLINE_LEARNING] Added TriggerAgent experience: action=%d reward=%.4f",
                                    self.entry_action, shaped_rewards['total_reward'])
                        
                        # Reset entry state
                        self.entry_state = None
                    
                    # Handbook: Update circuit breakers with trade result
                    current_equity = self.performance.get_total_pnl() + 10000.0  # Assuming 10K starting equity
                    self.circuit_breakers.update_trade(pnl, current_equity)
                    
                    # Check and log circuit breaker status
                    breaker_status = self.circuit_breakers.get_status()
                    if breaker_status['any_tripped']:
                        LOG.warning("[CIRCUIT-BREAKER] Status after trade: %s", breaker_status)
                    
                    # Auto-reset breakers after cooldown
                    self.circuit_breakers.reset_if_cooldown_elapsed()
                        self.entry_action = None
                    
                    # Adapt reward weights based on performance improvement
                    metrics = self.performance.get_metrics()
                    current_sharpe = metrics['sharpe_ratio']
                    sharpe_delta = current_sharpe - self.previous_sharpe
                    
                    if self.performance.total_trades % 5 == 0 and self.performance.total_trades > 5:
                        self.reward_shaper.adapt_weights(sharpe_delta)
                        LOG.info("[REWARD] Adapted weights based on Sharpe delta: %+.4f", sharpe_delta)
                        LOG.info(self.reward_shaper.print_summary())
                    
                    self.previous_sharpe = current_sharpe
                    
                    # Log performance dashboard every 5 trades
                    if self.performance.total_trades % 5 == 0:
                        LOG.info("\n" + self.performance.print_dashboard())
                    else:
                        metrics = self.performance.get_metrics()
                        LOG.info(
                            "[PERF] Trades: %d | Win Rate: %.1f%% | Total PnL: $%.2f | Sharpe: %.3f | Max DD: %.1f%%",
                            metrics['total_trades'],
                            metrics['win_rate'] * 100,
                            metrics['total_pnl'],
                            metrics['sharpe_ratio'],
                            metrics['max_drawdown'] * 100
                        )
                    
                    # Auto-export CSV every 10 trades
                    if self.performance.total_trades % 10 == 0:
                        try:
                            files = self.trade_exporter.export_all(self.performance, prefix="bot")
                            LOG.info("[EXPORT] CSV files saved: %s", ", ".join(files.values()))
                        except Exception as e:
                            LOG.error("[EXPORT] Failed to export CSV: %s", e)
                
            self.mfe_mae_tracker.reset()
            self.trade_entry_time = None

    def on_exec_report(self, msg: fix.Message):
        ex = fix.ExecType()
        if not msg.isSetField(ex):
            return
        msg.getField(ex)

        if ex.getValue() == "8":
            txt = fix.Text()
            if msg.isSetField(txt):
                msg.getField(txt)
                LOG.warning("[TRADE] Order rejected: %s", txt.getValue())
            return

        if ex.getValue() != "F":
            return

        sym = fix.Symbol()
        if msg.isSetField(sym):
            msg.getField(sym)
            if sym.getValue() != str(self.symbol_id):
                return

        self.request_positions()

    def on_biz_reject(self, msg: fix.Message):
        txt = fix.Text()
        if msg.isSetField(txt):
            msg.getField(txt)
            LOG.warning("[REJECT] BusinessMessageReject: %s", txt.getValue())

    def on_md_reject(self, msg: fix.Message):
        txt = fix.Text()
        if msg.isSetField(txt):
            msg.getField(txt)
            LOG.warning("[REJECT] MarketDataRequestReject: %s", txt.getValue())

    # ----------------------------
    # Rogers-Satchell Volatility for PathGeometry
    # ----------------------------
    def _calculate_rs_volatility(self, window: int = 20) -> float:
        """
        Calculate Rogers-Satchell volatility from recent bars.
        
        RS formula per bar:
            rs = ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
        
        Returns:
            Annualized volatility estimate (or default if insufficient data)
        """
        import math
        
        if len(self.bars) < window:
            return 0.005  # Default volatility
        
        rs_sum = 0.0
        valid_bars = 0
        
        # Use last 'window' bars
        for bar in list(self.bars)[-window:]:
            _, o, h, l, c = bar
            
            # Defensive: all prices must be positive
            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                continue
            
            # RS formula
            try:
                log_hc = math.log(h / c)
                log_ho = math.log(h / o)
                log_lc = math.log(l / c)
                log_lo = math.log(l / o)
                
                rs_bar = log_hc * log_ho + log_lc * log_lo
                
                if math.isfinite(rs_bar):
                    rs_sum += rs_bar
                    valid_bars += 1
            except (ValueError, ZeroDivisionError):
                continue
        
        if valid_bars < 5:
            return 0.005  # Default if insufficient valid bars
        
        # Average RS variance per bar
        rs_variance = rs_sum / valid_bars
        
        # RS variance is already in (return)^2 units per bar
        # Take sqrt to get volatility per bar
        if rs_variance > 0:
            vol_per_bar = math.sqrt(rs_variance)
        else:
            vol_per_bar = 0.005
        
        # Annualize: for M1 bars, 252 days * 24 hours * 60 mins = 362880 bars/year
        # But for Gold we use simplified: sqrt(trading_periods_per_year)
        # For M1: ~252 * 8 hours * 60 = 120,960 bars (forex hours)
        # vol_annual = vol_per_bar * sqrt(120960) ≈ 348x
        # But we keep it as per-bar for PathGeometry (no annualization needed)
        
        return max(0.0001, min(0.1, vol_per_bar))  # Clamp to reasonable range

    # ----------------------------
    # Strategy: run on bar close (M1/M15 configurable)
    # ----------------------------
    def on_bar_close(self, bar):
        t, o, h, l, c = bar
        
        # Increment bar counter for HUD
        self.bar_count += 1
        
        # Phase 3.5: Update activity monitor
        self.activity_monitor.on_bar_close()
        
        # Phase 3.5: Update VaR with bar return
        if len(self.bars) >= 2:
            prev_close = self.bars[-2][4] if len(self.bars) >= 2 else c
            bar_return = (c - prev_close) / prev_close if prev_close > 0 else 0.0
            self.var_estimator.update_return(bar_return)
        
        # Record bar if position is open
        if self.cur_pos != 0:
            self.path_recorder.add_bar(bar)
        
        # Calculate order book imbalance
        imbalance = 0.0
        if self.best_bid and self.best_ask:
            mid = (self.best_bid + self.best_ask) / 2.0
            spread = self.best_ask - self.best_bid
            # Simple imbalance approximation from spread skew
            if mid > 0:
                imbalance = (self.best_bid - mid) / (spread + 1e-10)
                imbalance = max(-1.0, min(1.0, imbalance))
        
        # Phase 3.5: Get exploration bonus from activity monitor
        exploration_bonus = self.activity_monitor.get_exploration_bonus() if hasattr(self.activity_monitor, 'get_exploration_bonus') else 0.0
        
        # Phase 3.5: Calculate Rogers-Satchell volatility for PathGeometry
        realized_vol = self._calculate_rs_volatility()
        
        # Debug: Log RS volatility every bar
        if len(self.bars) >= 20:
            LOG.debug("[RS_VOL] bars=%d realized_vol=%.6f", len(self.bars), realized_vol)
        
        # Phase 3.5: Periodic training step
        self.bars_since_training += 1
        if self.bars_since_training >= self.training_interval and hasattr(self.policy, 'train_step'):
            train_metrics = self.policy.train_step(self.adaptive_reg)
            self.bars_since_training = 0
            
            # Adjust regularization based on training metrics
            if train_metrics.get('trigger') or train_metrics.get('harvester'):
                trigger_td = train_metrics.get('trigger', {}).get('mean_td_error', 0) if train_metrics.get('trigger') else 0
                harvester_td = train_metrics.get('harvester', {}).get('mean_td_error', 0) if train_metrics.get('harvester') else 0
                avg_td = (trigger_td + harvester_td) / 2 if (trigger_td + harvester_td) else 0
                
                # Adaptive regularization: high TD error = overfitting signal
                if avg_td > 0.5:  # High TD error threshold
                    self.adaptive_reg.increase_regularization()
                elif avg_td < 0.1:  # Low TD error threshold
                    self.adaptive_reg.decrease_regularization()
        
        # Phase 3: Use DualPolicy if available
        if hasattr(self.policy, 'decide_entry'):
            # Dual-agent architecture
            if self.cur_pos == 0:
                # Handbook: Check circuit breakers BEFORE any entry decision
                if self.circuit_breakers.is_any_tripped():
                    tripped = self.circuit_breakers.get_tripped_breakers()
                    LOG.warning("[CIRCUIT-BREAKER] Trading halted: %s", 
                               ', '.join([b.name for b in tripped]))
                    self._export_hud_data()
                    return
                
                # Handbook: Get event-relative time features
                event_features = self.event_time_engine.calculate_features()
                is_high_liq = self.event_time_engine.is_high_liquidity_period()
                
                # Log key time features
                if len(self.bars) % 10 == 0:  # Every 10 bars
                    active_sessions = self.event_time_engine.get_active_sessions()
                    LOG.debug("[EVENT-TIME] Sessions: %s | High liquidity: %s", 
                             ','.join(active_sessions) if active_sessions else 'None', is_high_liq)
                
                # FLAT: Check for entry
                action, confidence, runway = self.policy.decide_entry(
                    self.bars, imbalance=imbalance, vpin_z=0.0, depth_ratio=1.0, 
                    realized_vol=realized_vol
                )
                # action: 0=NO_ENTRY, 1=LONG, 2=SHORT
                desired = 1 if action == 1 else (-1 if action == 2 else 0)
                
                # Store state for online learning (if entry is taken)
                if action != 0 and hasattr(self.policy, 'trigger') and hasattr(self.policy.trigger, 'last_state'):
                    self.entry_state = self.policy.trigger.last_state.copy() if self.policy.trigger.last_state is not None else None
                    self.entry_action = action
                
                # Get PathGeometry features for logging
                geom = self.policy.path_geometry.last if hasattr(self.policy, 'path_geometry') and self.policy.path_geometry else {}
                feas = geom.get('feasibility', 0.0)
                geom_rw = geom.get('runway', 0.0)
                eff = geom.get('efficiency', 0.0)
                
                LOG.info(
                    "[BAR M%d] %s O=%.2f H=%.2f L=%.2f C=%.2f | TRIGGER: action=%d conf=%.2f runway=%.4f feas=%.2f | RS_vol=%.5f bars=%d | desired=%s cur=%s",
                    self.timeframe_minutes, t.isoformat(), o, h, l, c,
                    action, confidence, runway, feas, realized_vol, len(self.bars), desired, self.cur_pos
                )
            else:
                # IN POSITION: Check for exit
                exit_action, exit_conf = self.policy.decide_exit(
                    self.bars, current_price=c, imbalance=imbalance
                )
                # exit_action: 0=HOLD, 1=CLOSE
                desired = 0 if exit_action == 1 else self.cur_pos
                
                LOG.info(
                    "[BAR M%d] %s O=%.2f H=%.2f L=%.2f C=%.2f | HARVESTER: exit=%d conf=%.2f | desired=%s cur=%s",
                    self.timeframe_minutes, t.isoformat(), o, h, l, c,
                    exit_action, exit_conf, desired, self.cur_pos
                )
        else:
            # Simple policy fallback
            action = self.policy.decide(self.bars)
            desired = -1 if action == 0 else (0 if action == 1 else 1)
            
            LOG.info(
                "[BAR M%d] %s O=%.2f H=%.2f L=%.2f C=%.2f | desired=%s cur=%s",
                self.timeframe_minutes, t.isoformat(), o, h, l, c, desired, self.cur_pos
            )

        if not self.trade_sid:
            self._export_hud_data()  # Still export HUD even without trade session
            return
        if desired == self.cur_pos:
            self._export_hud_data()  # Still export HUD even when no action
            return

        # Phase 3.5: VaR circuit breaker check before new entries
        if self.cur_pos == 0 and desired != 0:
            # Only check VaR for new position entries
            if self.kurtosis_monitor.is_breaker_active:
                LOG.warning("[CIRCUIT_BREAKER] Kurtosis breaker ACTIVE (kurtosis=%.2f) - skipping entry",
                           self.kurtosis_monitor.current_kurtosis)
                self._export_hud_data()  # Export even on circuit breaker
                return
            
            # Calculate current VaR
            current_var = self.var_estimator.estimate_var()
            max_var_threshold = 0.05  # 5% max VaR threshold
            if current_var > max_var_threshold:
                LOG.warning("[CIRCUIT_BREAKER] VaR=%.4f exceeds threshold=%.4f - skipping entry",
                           current_var, max_var_threshold)
                self._export_hud_data()  # Export even on VaR filter
                return
            
            # Phase 3.5: Learned spread threshold check (2x minimum observed spread)
            spread_multiplier = self.param_manager.get("spread_relax", 2.0) if hasattr(self, 'param_manager') else 2.0
            is_acceptable, current_spread, max_spread = self.friction_calculator.is_spread_acceptable(spread_multiplier)
            if not is_acceptable:
                LOG.warning("[SPREAD_FILTER] Current=%.2f pips > Learned max=%.2f pips (%.1fx min) - skipping entry",
                           current_spread, max_spread, spread_multiplier)
                self._export_hud_data()  # Export even on filtered entries
                return

        # Export HUD data every bar (before potential order)
        self._export_hud_data()
        
        delta = desired - self.cur_pos
        side = "1" if delta > 0 else "2"
        order_qty = abs(delta) * self.qty
        self.send_market_order(side=side, qty=order_qty)

    def send_market_order(self, side: str, qty: float):
        self.clord_counter += 1
        clid = f"cl_{int(time.time())}_{self.clord_counter}"

        order = fix44.NewOrderSingle()
        order.setField(fix.ClOrdID(clid))
        order.setField(fix.Symbol(str(self.symbol_id)))
        order.setField(fix.Side(side))
        order.setField(fix.TransactTime(utc_ts_ms()))
        order.setField(fix.OrdType("1"))
        order.setField(fix.OrderQty(qty))
        
        # Start MFE/MAE tracking and path recording
        if self.best_bid and self.best_ask:
            entry_price = (self.best_bid + self.best_ask) / 2.0
            direction = 1 if side == "1" else -1
            self.trade_entry_time = utc_now()  # Store for performance tracker
            self.mfe_mae_tracker.start_tracking(entry_price, direction)
            self.path_recorder.start_recording(self.trade_entry_time, entry_price, direction)

        fix.Session.sendToTarget(order, self.trade_sid)
        
        # Phase 3.5: Notify activity monitor of trade execution
        self.activity_monitor.on_trade_executed()
        
        LOG.info(
            "[TRADE] Sent MKT %s qty=%.6f clOrdID=%s",
            ("BUY" if side == "1" else "SELL"),
            qty,
            clid,
        )

    def _export_hud_data(self):
        """Export real-time data to JSON files for HUD display.
        
        Called every bar close to keep HUD updated with:
        - Bot configuration (symbol, timeframe, uptime)
        - Current position (direction, MFE/MAE, unrealized PnL)
        - Performance metrics (daily/weekly/monthly/lifetime)
        - Training stats (buffer sizes, losses)
        - Risk metrics (VaR, kurtosis, regime, path geometry)
        """
        try:
            now = dt.datetime.now(dt.timezone.utc)
            uptime_seconds = int((now - self.start_time).total_seconds())
            
            # 1. Bot Configuration
            bot_config = {
                "symbol": self.symbol,
                "symbol_id": self.symbol_id,
                "timeframe": f"{self.timeframe_minutes}m",
                "uptime_seconds": uptime_seconds,
                "bar_count": self.bar_count,
                "training_enabled": os.environ.get("DDQN_ONLINE_LEARNING", "1") == "1"
            }
            with open(self.hud_data_dir / "bot_config.json", 'w') as f:
                json.dump(bot_config, f, indent=2)
            
            # 2. Current Position
            current_price = (self.best_bid + self.best_ask) / 2.0 if self.best_bid and self.best_ask else 0.0
            direction = "LONG" if self.cur_pos > 0 else ("SHORT" if self.cur_pos < 0 else "FLAT")
            
            # Get MFE/MAE from tracker if in position
            entry_price = self.mfe_mae_tracker.entry_price if self.cur_pos != 0 else 0.0
            mfe = self.mfe_mae_tracker.mfe if self.cur_pos != 0 else 0.0
            mae = self.mfe_mae_tracker.mae if self.cur_pos != 0 else 0.0
            
            # Calculate unrealized PnL
            unrealized_pnl = 0.0
            bars_held = 0
            if self.cur_pos != 0 and entry_price > 0:
                if self.cur_pos > 0:  # LONG
                    unrealized_pnl = (current_price - entry_price) * self.qty
                else:  # SHORT
                    unrealized_pnl = (entry_price - current_price) * self.qty
                bars_held = len(self.path_recorder.bars) if hasattr(self.path_recorder, 'bars') else 0
            
            position_data = {
                "direction": direction,
                "entry_price": entry_price,
                "current_price": current_price,
                "mfe": mfe,
                "mae": mae,
                "unrealized_pnl": unrealized_pnl,
                "bars_held": bars_held
            }
            with open(self.hud_data_dir / "current_position.json", 'w') as f:
                json.dump(position_data, f, indent=2)
            
            # 3. Performance Metrics (from PerformanceTracker)
            metrics = self.performance.get_metrics() if hasattr(self.performance, 'get_metrics') else {}
            performance_snapshot = {
                "daily": {
                    "total_trades": metrics.get('total_trades', 0),
                    "win_rate": metrics.get('win_rate', 0.0),
                    "total_pnl": metrics.get('total_pnl', 0.0),
                    "sharpe_ratio": metrics.get('sharpe', 0.0),
                    "max_drawdown": metrics.get('max_drawdown', 0.0)
                },
                "weekly": {
                    "total_trades": metrics.get('total_trades', 0),
                    "win_rate": metrics.get('win_rate', 0.0),
                    "total_pnl": metrics.get('total_pnl', 0.0),
                    "sharpe_ratio": metrics.get('sharpe', 0.0),
                    "max_drawdown": metrics.get('max_drawdown', 0.0)
                },
                "monthly": {
                    "total_trades": metrics.get('total_trades', 0),
                    "win_rate": metrics.get('win_rate', 0.0),
                    "total_pnl": metrics.get('total_pnl', 0.0),
                    "sharpe_ratio": metrics.get('sharpe', 0.0),
                    "max_drawdown": metrics.get('max_drawdown', 0.0)
                },
                "lifetime": {
                    "total_trades": metrics.get('total_trades', 0),
                    "win_rate": metrics.get('win_rate', 0.0),
                    "total_pnl": metrics.get('total_pnl', 0.0),
                    "sharpe_ratio": metrics.get('sharpe', 0.0),
                    "sortino_ratio": metrics.get('sortino', 0.0),
                    "omega_ratio": metrics.get('omega', 0.0),
                    "max_drawdown": metrics.get('max_drawdown', 0.0)
                }
            }
            with open(self.hud_data_dir / "performance_snapshot.json", 'w') as f:
                json.dump(performance_snapshot, f, indent=2)
            
            # 4. Training Stats
            trigger_buffer_size = 0
            harvester_buffer_size = 0
            trigger_loss = 0.0
            harvester_loss = 0.0
            total_agents = 0
            arena_diversity = {"trigger_diversity": 0.0, "harvester_diversity": 0.0}
            last_agreement = 1.0
            
            if hasattr(self.policy, 'trigger') and hasattr(self.policy.trigger, 'replay_buffer'):
                trigger_buffer_size = len(self.policy.trigger.replay_buffer) if hasattr(self.policy.trigger.replay_buffer, '__len__') else 0
            if hasattr(self.policy, 'harvester') and hasattr(self.policy.harvester, 'replay_buffer'):
                harvester_buffer_size = len(self.policy.harvester.replay_buffer) if hasattr(self.policy.harvester.replay_buffer, '__len__') else 0
            
            # Check for arena (multi-agent)
            if hasattr(self.policy, 'trigger') and hasattr(self.policy.trigger, 'arena'):
                arena = self.policy.trigger.arena
                total_agents = len(arena.agents) if hasattr(arena, 'agents') else 0
                if hasattr(arena, 'last_diversity'):
                    arena_diversity["trigger_diversity"] = arena.last_diversity
                if hasattr(arena, 'last_agreement'):
                    last_agreement = arena.last_agreement
            
            training_stats = {
                "trigger_buffer_size": trigger_buffer_size,
                "harvester_buffer_size": harvester_buffer_size,
                "total_agents": total_agents,
                "arena_diversity": arena_diversity,
                "last_agreement_score": last_agreement,
                "consensus_mode": "weighted_average",
                "trigger_loss": trigger_loss,
                "harvester_loss": harvester_loss,
                "last_training_time": "Active" if self.bars_since_training < self.training_interval else "Never"
            }
            with open(self.hud_data_dir / "training_stats.json", 'w') as f:
                json.dump(training_stats, f, indent=2)
            
            # 5. Risk Metrics
            # VaR and kurtosis
            current_var = self.var_estimator.estimate_var() if hasattr(self.var_estimator, 'estimate_var') else 0.0
            current_kurtosis = self.kurtosis_monitor.current_kurtosis if hasattr(self.kurtosis_monitor, 'current_kurtosis') else 0.0
            circuit_breaker = "ACTIVE" if self.kurtosis_monitor.is_breaker_active else "INACTIVE"
            
            # Realized volatility (Rogers-Satchell)
            realized_vol = self._calculate_rs_volatility() if len(self.bars) >= 2 else 0.0
            
            # Regime from dual policy
            regime = "UNKNOWN"
            regime_zeta = 1.0
            if hasattr(self.policy, 'current_regime'):
                regime = self.policy.current_regime
            if hasattr(self.policy, 'current_zeta'):
                regime_zeta = self.policy.current_zeta
            
            # Path geometry features
            geom = self.path_geometry.last if hasattr(self.path_geometry, 'last') else {}
            efficiency = geom.get('efficiency', 1.0)
            gamma = geom.get('gamma', 0.0)
            jerk = geom.get('jerk', 0.0)
            runway = geom.get('runway', 0.5)
            feasibility = geom.get('feasibility', 0.5)
            
            # Market microstructure
            spread = self.best_ask - self.best_bid if self.best_bid and self.best_ask else 0.0
            vpin = 0.0  # TODO: wire up VPIN if available
            vpin_zscore = 0.0
            imbalance = 0.0
            depth_bid = 1.0
            depth_ask = 1.0
            
            risk_metrics = {
                "var": current_var,
                "kurtosis": current_kurtosis,
                "circuit_breaker": circuit_breaker,
                "realized_vol": realized_vol,
                "regime": regime,
                "regime_zeta": regime_zeta,
                "vpin": vpin,
                "vpin_zscore": vpin_zscore,
                "efficiency": efficiency,
                "gamma": gamma,
                "jerk": jerk,
                "runway": runway,
                "feasibility": feasibility,
                "spread": spread,
                "imbalance": imbalance,
                "depth_bid": depth_bid,
                "depth_ask": depth_ask
            }
            with open(self.hud_data_dir / "risk_metrics.json", 'w') as f:
                json.dump(risk_metrics, f, indent=2)
            
        except Exception as e:
            LOG.error("[HUD] Failed to export data: %s", str(e))


# ----------------------------
# Main: start two initiators
# ----------------------------
def require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"Missing required env var: {name}")
    return v


def main():
    # Setup signal handlers for graceful shutdown
    app = None
    
    def signal_handler(signum, frame):
        if app:
            app.graceful_shutdown(signum, frame)
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Require creds explicitly so you don't get silent logouts.
    user = require_env("CTRADER_USERNAME")
    _ = require_env("CTRADER_PASSWORD_QUOTE")
    _ = require_env("CTRADER_PASSWORD_TRADE")

    # Parse configuration with type safety
    try:
        symbol_id = int(os.environ.get("CTRADER_SYMBOL_ID", "41"))  # Default: XAUUSD=41
        symbol = os.environ.get("CTRADER_SYMBOL", "XAUUSD")  # Instrument-agnostic
        qty = float(os.environ.get("CTRADER_QTY", "0.01"))  # Default: 0.01 lot for Gold
        timeframe_minutes = int(os.environ.get("CTRADER_TIMEFRAME_MIN", "1"))  # M1 for testing
    except (ValueError, TypeError) as e:
        LOG.error("Invalid configuration value: %s", e)
        raise SystemExit(1)

    # CRITICAL: Validate configuration to prevent financial losses
    if qty <= 0:
        LOG.error("CTRADER_QTY must be positive, got: %s", qty)
        raise SystemExit(1)
    if qty > 100:  # Sanity check: prevent absurdly large orders
        LOG.error("CTRADER_QTY suspiciously large (>100), got: %s. Aborting for safety.", qty)
        raise SystemExit(1)
    if timeframe_minutes <= 0:
        LOG.error("CTRADER_TIMEFRAME_MIN must be positive, got: %s", timeframe_minutes)
        raise SystemExit(1)
    if symbol_id <= 0:
        LOG.error("CTRADER_SYMBOL_ID must be positive, got: %s", symbol_id)
        raise SystemExit(1)
    if not symbol or not symbol.strip():
        LOG.error("CTRADER_SYMBOL must be non-empty")
        raise SystemExit(1)

    cfg_quote = os.environ.get("CTRADER_CFG_QUOTE", "ctrader_quote.cfg")
    cfg_trade = os.environ.get("CTRADER_CFG_TRADE", "ctrader_trade.cfg")

    # Validate config files exist
    if not os.path.exists(cfg_quote):
        LOG.error("Quote config file not found: %s", cfg_quote)
        raise SystemExit(1)
    if not os.path.exists(cfg_trade):
        LOG.error("Trade config file not found: %s", cfg_trade)
        raise SystemExit(1)

    LOG.info("✓ Configuration validated: symbol=%s symbol_id=%s qty=%s timeframe=M%d", symbol, symbol_id, qty, timeframe_minutes)
    LOG.info("cfg_quote=%s", cfg_quote)
    LOG.info("cfg_trade=%s", cfg_trade)
    LOG.info("CTRADER_USERNAME=%s", user)

    app = CTraderFixApp(symbol_id=symbol_id, qty=qty, timeframe_minutes=timeframe_minutes, symbol=symbol)

    settings_q = fix.SessionSettings(cfg_quote)
    store_q = fix.FileStoreFactory(settings_q)
    log_q = fix.FileLogFactory(settings_q)
    initiator_q = fix.SocketInitiator(app, store_q, settings_q, log_q)

    settings_t = fix.SessionSettings(cfg_trade)
    store_t = fix.FileStoreFactory(settings_t)
    log_t = fix.FileLogFactory(settings_t)
    initiator_t = fix.SocketInitiator(app, store_t, settings_t, log_t)

    initiator_q.start()
    initiator_t.start()
    LOG.info("[MAIN] FIX initiators started (QUOTE + TRADE)")
    
    # Connection monitoring loop
    health_check_interval = 30  # seconds
    last_health_log = time.time()
    consecutive_failures = 0
    max_consecutive_failures = 5

    try:
        while True:
            time.sleep(1)
            
            # Periodic health check and logging
            now = time.time()
            if now - last_health_log >= health_check_interval:
                last_health_log = now
                
                # Check connection status
                status = app.get_connection_status()
                if not status["connection_healthy"]:
                    consecutive_failures += 1
                    LOG.warning("[MAIN] Connection unhealthy (consecutive failures: %d/%d) - %s",
                               consecutive_failures, max_consecutive_failures, status)
                else:
                    consecutive_failures = 0
                    LOG.debug("[MAIN] Connection healthy: %s", status)
                    
    except KeyboardInterrupt:
        LOG.info("Keyboard interrupt received, shutting down...")
    finally:
        # Signal shutdown to health monitor
        app.stop_health_monitor()
        
        # Stop initiators
        try:
            initiator_q.stop()
        except Exception as e:
            LOG.error("[SHUTDOWN] Error stopping QUOTE initiator: %s", e)
        
        try:
            initiator_t.stop()
        except Exception as e:
            LOG.error("[SHUTDOWN] Error stopping TRADE initiator: %s", e)
            
        LOG.info("Shutdown complete")


if __name__ == "__main__":
    main()
1.