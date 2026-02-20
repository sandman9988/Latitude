#!/usr/bin/env python3
# ctrader_ddqn_paper.py
# Dual FIX sessions (QUOTE+TRADE) for cTrader/Pepperstone demo.
# Builds BTCUSD (symbolId=10028) M15 bars from best bid/ask, then trades 0.10 qty target-position via TRADE.
#
# Requires QuickFIX built/installed into the venv (you already did this).
#
# Run:
#   source ~/Documents/.venv/bin/activate
#   export CTRADER_USERNAME="your_username_here"
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
import math
import os
import random
import signal
import socket
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

try:
    import quickfix as fix
    import quickfix44 as fix44
    QUICKFIX_AVAILABLE = True
except ImportError:
    QUICKFIX_AVAILABLE = False

    def _qfix_stub(name: str):
        """Return a lightweight stub for any FIX field/message type."""
        return type(name, (), {
            "__init__": lambda self, *a, **kw: None,
            "__repr__": lambda self: f"<{name}>",
            "setField": lambda self, *a: None,
            "getHeader": lambda self: type("Header", (), {"setField": lambda s, *a: None, "getField": lambda s, *a: ""})(),
            "getString": lambda self: "",
            "getValue": lambda self: "",
        })

    class _FixStubMeta(type):
        def __getattr__(cls, name):
            stub = _qfix_stub(name)
            setattr(cls, name, stub)
            return stub

    class _ApplicationStub:
        """Stub base class when quickfix C-extension is not installed."""

        def onCreate(self, session_id):  # NOSONAR
            """QuickFIX Application callback stub – no-op."""

        def onLogon(self, session_id):  # NOSONAR
            """QuickFIX Application callback stub – no-op."""

        def onLogout(self, session_id):  # NOSONAR
            """QuickFIX Application callback stub – no-op."""

        def toAdmin(self, message, session_id):  # NOSONAR
            """QuickFIX Application callback stub – no-op."""

        def toApp(self, message, session_id):  # NOSONAR
            """QuickFIX Application callback stub – no-op."""

        def fromAdmin(self, message, session_id):  # NOSONAR
            """QuickFIX Application callback stub – no-op."""

        def fromApp(self, message, session_id):  # NOSONAR
            """QuickFIX Application callback stub – no-op."""

        def send(self, message, session_id=None):  # noqa: stub
            """QuickFIX Application callback stub – no-op."""

    class fix(metaclass=_FixStubMeta):  # type: ignore[no-redef]
        """Stub namespace – quickfix C-extension not installed."""
        Application = _ApplicationStub
        Session = _qfix_stub("Session")
        SessionID = _qfix_stub("SessionID")
        Message = _qfix_stub("Message")
        SocketInitiator = _qfix_stub("SocketInitiator")
        FileStoreFactory = _qfix_stub("FileStoreFactory")
        FileLogFactory = _qfix_stub("FileLogFactory")
        ScreenLogFactory = _qfix_stub("ScreenLogFactory")
        SessionSettings = _qfix_stub("SessionSettings")

    class fix44(metaclass=_FixStubMeta):  # type: ignore[no-redef]
        """Stub namespace for quickfix44 message types."""
        pass

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from src.monitoring.activity_monitor import ActivityMonitor
from src.monitoring.production_monitor import ProductionMonitor
from src.core.adaptive_regularization import AdaptiveRegularization
from src.monitoring.audit_logger import DecisionLogger, TransactionLogger
from src.core.broker_execution_model import BrokerExecutionModel, OrderSide
from src.risk.circuit_breakers import CircuitBreakerManager
from src.agents.dual_policy import DualPolicy
from src.features.event_time_features import EventTimeFeatureEngine
from src.risk.friction_costs import FrictionCalculator
from src.persistence.learned_parameters import LearnedParametersManager
from src.utils.non_repaint_guards import NonRepaintBarAccess
from src.core.order_book import OrderBook, VPINCalculator
from src.risk.path_geometry import PathGeometry
from src.monitoring.performance_tracker import PerformanceTracker
from src.core.reward_shaper import RewardShaper
from src.core.trade_manager_integration import TradeManagerIntegration
from src.core.trade_manager import Side
from src.utils.ring_buffer import RollingStats
from src.risk.emergency_close import create_emergency_closer

# Handbook components - Phase 1
from src.utils.safe_math import SafeMath
from src.monitoring.trade_exporter import TradeExporter
from src.risk.var_estimator import KurtosisMonitor, RegimeType, VaREstimator, position_size_from_var
from src.core.self_test import run_self_test

# Feature calculation constants
MIN_BARS_FOR_FEATURES: int = 70
RETURN_LAG_SHORT: int = 2
RETURN_LAG_MEDIUM: int = 6
FEATURE_VALIDATION_MIN_COLS: int = 3
MOMENTUM_BULLISH_THRESHOLD: float = 0.2
MOMENTUM_BEARISH_THRESHOLD: float = -0.2

# Training and performance constants
ENSEMBLE_WEIGHT_CAPACITY: int = 500
PERFORMANCE_INTERVAL_TRADES: int = 5
MIN_TRADES_FOR_ADAPTATION: int = 5
AUTOSAVE_INTERVAL_BARS: int = 50
TRAINING_TD_HIGH_THRESHOLD: float = 0.5
TRAINING_TD_LOW_THRESHOLD: float = 0.1

# Volatility calculation constants
MIN_VALID_BARS_FOR_VOL: int = 5
DEFAULT_VOLATILITY: float = 0.005

# Safety validation constants
MAX_QTY_SANITY_CHECK: float = 100.0
MAX_PRICE_SANITY: float = 1e9         # Reject suspiciously large prices
MIN_POSITION_QTY: float = 0.0001      # Minimum qty to treat position as active
MIN_POSITION_THRESHOLD: float = 0.001  # Minimum qty for long/short side checks
MAX_POSITION_SANITY: float = 1000.0   # Sanity upper bound for position quantities
HARVESTER_DEBUG_INTERVAL: float = 60.0  # Seconds between harvester debug log lines

# Threshold / limit constants (also reduce magic-value violations)
MIN_BARS_FOR_VOL_CALC: int = 20          # Minimum bars before RS-volatility is reliable
EPSILON_HIGH_THRESHOLD: float = 0.5      # epsilon above this ⇒ still in random-exploration phase
RUNWAY_FALLBACK_THRESHOLD: float = 0.002 # Predicted-runway below this ⇒ treat as exploration entry
EXPLORATION_SAMPLE_RATE: float = 0.10    # Fraction of NO_ENTRY bars logged to experience buffer
MAX_LOG_ENTRIES: int = 1000              # Maximum decision-log entries kept in the JSON file

# Position and trading constants
MIN_BARS_FOR_VAR_UPDATE: int = 2
MIN_BARS_FOR_PREV_CLOSE: int = 2
MIN_TRADE_HISTORY_EXPLORATION: int = 20

# Action constants for policy decisions
ACTION_NO_ENTRY: int = 0
ACTION_LONG: int = 1
ACTION_SHORT: int = 2


# ----------------------------
# Lightweight grouping types (reduce per-function param count)
# ----------------------------
class TradeOutcome(NamedTuple):
    """Scalar result values for a completed trade."""

    pnl: float
    mfe: float = 0.0
    mae: float = 0.0
    winner_to_loser: bool = False


class MfeMaeSnapshot(NamedTuple):
    """Current and previous MFE/MAE readings for harvester reward calc."""

    current_mfe: float
    current_mae: float
    prev_mfe: float
    prev_mae: float


class HarvesterPositionState(NamedTuple):
    """Live position state inputs for harvester hold-reward calculation."""

    bars_held: int
    unrealized_pnl: float
    entry_price: float
    current_price: float
    realized_vol: float = 0.01


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

    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)

    # Configure root logger to capture ALL module logs (src.core.trade_manager, etc.)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(sh)
    root_logger.addHandler(fh)

    # Also configure named logger for backward compatibility
    logger = logging.getLogger("ctrader")
    logger.setLevel(logging.INFO)

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
        self.bucket: dt.datetime | None = None
        self.o: float | None = None
        self.h: float | None = None
        self.l: float | None = None
        self.c: float | None = None

    def bucket_start(self, t: dt.datetime) -> dt.datetime:
        m = (t.minute // self.timeframe_minutes) * self.timeframe_minutes
        return t.replace(minute=m, second=0, microsecond=0)

    def update(self, t: dt.datetime, mid: float):
        LOG.debug("[BARBUILDER] update called: t=%s, mid=%.5f", t, mid)

        # Defensive: Validate inputs
        if mid is None or mid <= 0:
            LOG.warning("[BARBUILDER] Invalid mid price: %.5f - skipping update", mid or 0)
            return None

        if not isinstance(t, dt.datetime):
            LOG.error("[BARBUILDER] Invalid datetime type: %s", type(t))
            return None

        b = self.bucket_start(t)
        if self.bucket is None:
            LOG.debug("[BARBUILDER] Initializing new bucket: %s mid=%.5f", b, mid)
            self.bucket = b
            self.o = self.h = self.l = self.c = mid
            return None

        if b != self.bucket:
            # Defensive: Validate bar data before closing
            if self.o is None or self.h is None or self.l is None or self.c is None:
                LOG.error(
                    "[BARBUILDER] Incomplete bar data - o=%.2f h=%.2f l=%.2f c=%.2f",
                    self.o or 0,
                    self.h or 0,
                    self.l or 0,
                    self.c or 0,
                )
                # Initialize new bucket but don't return invalid bar
                self.bucket = b
                self.o = self.h = self.l = self.c = mid
                return None

            closed = (self.bucket, self.o, self.h, self.l, self.c)
            LOG.debug("[BARBUILDER] Closing bar: %s starting new bucket: %s mid=%.5f", closed, b, mid)
            self.bucket = b
            self.o = self.h = self.l = self.c = mid
            return closed

        self.c = mid
        self._update_hl(mid)
        return None

    def _update_hl(self, mid: float) -> None:
        """Update the running high and low for the current bar."""
        if self.h is None or mid > self.h:
            self.h = mid
        if self.l is None or mid < self.l:
            self.l = mid


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Compute a simple rolling mean of window *n*, returning NaN for early entries."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) >= n:
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1 :] = (cs[n:] - cs[:-n]) / n
    return out


def _rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    """Compute a simple rolling std of window *n*, returning NaN for early entries."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) >= n:
        for i in range(n - 1, len(x)):
            w = x[i - n + 1 : i + 1]
            out[i] = np.std(w)
    return out


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
        self.ensemble_enabled = os.environ.get("DDQN_MODEL_ENSEMBLE", "0") == "1"
        self.ensemble_weights = []
        self._ensemble_stats = {
            "updates": 0,
            "last_weight": 0.0,
            "max_disagreement": 0.0,
        }

        model_path = os.environ.get("DDQN_MODEL_PATH", "").strip()
        if model_path and TORCH_AVAILABLE:
            try:
                # Type guards: nn is not None when TORCH_AVAILABLE is True
                if nn is None:
                    raise RuntimeError("nn module unavailable despite TORCH_AVAILABLE=True")
                if torch is None:
                    raise RuntimeError("torch module unavailable despite TORCH_AVAILABLE=True")

                class QNet(nn.Module):
                    def __init__(self, _window: int, n_features: int, n_actions: int = 3):
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
                self.model = QNet(_window=self.window, n_features=4, n_actions=3)
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.model.eval()
                self.use_torch = True
                LOG.info("[POLICY] Loaded DDQN model: %s", model_path)
            except Exception as e:
                LOG.warning("[POLICY] Failed to load model, running fallback. Error: %s", e)
                self.use_torch = False

    def decide(self, bars: deque, **_kwargs) -> int:
        if len(bars) < MIN_BARS_FOR_FEATURES:
            return 1  # FLAT

        closes = [b[4] for b in bars]
        c = np.array(closes, dtype=np.float64)

        # Defensive: calculate returns with safe division
        ret1 = np.zeros_like(c)
        if len(c) >= RETURN_LAG_SHORT:
            ret1[1:] = np.divide(c[1:], c[:-1], out=np.ones_like(c[1:]), where=c[:-1] != 0) - 1.0

        ret5 = np.zeros_like(c)
        if len(c) >= RETURN_LAG_MEDIUM:
            ret5[5:] = np.divide(c[5:], c[:-5], out=np.ones_like(c[5:]), where=c[:-5] != 0) - 1.0

        def rolling_mean(x, n):
            return _rolling_mean(x, n)

        def rolling_std(x, n):
            return _rolling_std(x, n)

        ma_fast = rolling_mean(c, 10)
        ma_slow = rolling_mean(c, 30)
        ma_diff = np.divide(ma_fast, ma_slow, out=np.ones_like(ma_fast), where=ma_slow != 0) - 1.0
        vol = rolling_std(ret1, 20)

        # Defensive: clean all features for NaN/Inf
        feats = np.vstack(
            [
                np.nan_to_num(ret1, nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(ret5, nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(ma_diff, nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0),
            ]
        ).T
        feats = feats[-self.window :].astype(np.float32)

        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True) + 1e-8
        x = (feats - mu) / sd

        if not self.use_torch:
            # Defensive: validate array shape before access
            if x.shape[0] == 0 or x.shape[1] < FEATURE_VALIDATION_MIN_COLS:
                return 1  # Default to HOLD if insufficient data
            md = float(x[-1, 2])
            if md > MOMENTUM_BULLISH_THRESHOLD:
                return 2
            if md < MOMENTUM_BEARISH_THRESHOLD:
                return 0
            return 1

        # Type guard: ensure torch is available
        if self.torch is None:
            raise RuntimeError("torch is None but use_torch=True — model load must have failed")
        if self.model is None:
            raise RuntimeError("model is None but use_torch=True — model load must have failed")
        with self.torch.no_grad():
            t = self.torch.from_numpy(x).unsqueeze(0)
            q = self.model(t).squeeze(0).numpy()
            return int(q.argmax())

    def update_ensemble_weights(self, disagreement: float, reward: float = 0.0) -> float:
        """Update ensemble weight tracking based on model disagreement and reward feedback."""
        disagreement = max(0.0, float(disagreement))
        normalized = min(disagreement, 1.0)
        weight = 1.0 - normalized
        if reward < 0:
            weight *= 0.9  # penalty for losses when disagreement was high
        self.ensemble_weights.append(weight)
        if len(self.ensemble_weights) > ENSEMBLE_WEIGHT_CAPACITY:
            self.ensemble_weights.pop(0)
        self._ensemble_stats["updates"] += 1
        self._ensemble_stats["last_weight"] = weight
        self._ensemble_stats["max_disagreement"] = max(self._ensemble_stats["max_disagreement"], disagreement)
        return weight

    def get_ensemble_stats(self) -> dict:
        """Return diagnostic metrics for ensemble hooks."""
        return {
            "enabled": self.ensemble_enabled,
            "updates": self._ensemble_stats["updates"],
            "last_weight": self._ensemble_stats["last_weight"],
            "max_disagreement": self._ensemble_stats["max_disagreement"],
            "weights_tracked": len(self.ensemble_weights),
        }


# ----------------------------
# Path Recorder
# ----------------------------
class PathRecorder:
    """Record M1 OHLC path during entire trade lifecycle."""

    def __init__(self, position_id: str | None = None):
        self.position_id = position_id  # NEW: Track which position this is for
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
        LOG.info(
            "[PATH] Started recording for %s trade at %.2f",
            "LONG" if direction == 1 else "SHORT",
            entry_price,
        )

    def add_bar(self, bar):
        """Add a bar to the path. bar is tuple: (timestamp, o, h, l, c)"""
        if not self.recording:
            return
        self.path.append(bar)

    def stop_recording(
        self,
        exit_time: dt.datetime,
        exit_price: float,
        outcome: TradeOutcome,
    ) -> dict:
        """Stop recording and return trade summary with path.

        Args:
            exit_time: Timestamp of the trade exit.
            exit_price: Execution price at exit.
            outcome: A :class:`TradeOutcome` carrying pnl, mfe, mae, winner_to_loser.
        """
        if not self.recording:
            return {}

        self.recording = False
        self.trade_counter += 1

        pnl = outcome.pnl
        mfe = outcome.mfe
        mae = outcome.mae
        winner_to_loser = outcome.winner_to_loser

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
                {"timestamp": t.isoformat(), "open": o, "high": h, "low": low_price, "close": c}
                for t, o, h, low_price, c in self.path
            ],
        }

        # Save to JSON file
        self._save_to_file(trade_record)

        LOG.info(
            "[PATH] Stopped recording. Trade #%d: %d bars, %.2f seconds, PnL=%.2f | MFE=%.2f MAE=%.2f WTL=%s",
            self.trade_counter,
            len(self.path),
            duration_seconds,
            pnl,
            mfe,
            mae,
            winner_to_loser,
        )

        return trade_record

    def _save_to_file(self, trade_record: dict):
        """Save trade record to JSON file."""
        trades_dir = Path("trades")
        trades_dir.mkdir(exist_ok=True)

        filename = trades_dir / f"trade_{trade_record['trade_id']:04d}_{trade_record['direction'].lower()}.json"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(trade_record, f, indent=2)
            LOG.info("[PATH] Saved to %s", filename)
        except Exception as e:
            LOG.error("[PATH] Failed to save: %s", e)


# ----------------------------
# MFE/MAE Tracker
# ----------------------------
class MFEMAETracker:
    """Track Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE) per position."""

    def __init__(self, position_id: str | None = None):
        self.position_id = position_id  # NEW: Track which position this is for
        self.entry_price = None
        self.direction = None
        self.mfe = 0.0  # max favorable move in $ (profit)
        self.mae = 0.0  # max adverse move in $ (loss, stored as positive)
        self.best_profit = 0.0
        self.worst_loss = 0.0
        self.winner_to_loser = False

    def start_tracking(self, entry_price: float, direction: int):
        """direction: 1=long, -1=short"""
        # Defensive: Validate entry_price
        if entry_price is None or entry_price <= 0:
            LOG.error(
                "[MFE_MAE] Invalid entry_price=%.5f for position %s - cannot track", entry_price or 0, self.position_id
            )
            return

        # Defensive: Validate direction
        if direction not in (1, -1):
            LOG.warning(
                "[MFE_MAE] Invalid direction=%d for position %s - defaulting to LONG", direction, self.position_id
            )
            direction = 1

        self.entry_price = float(entry_price)
        self.direction = int(direction)
        self.mfe = 0.0
        self.mae = 0.0
        self.best_profit = 0.0
        self.worst_loss = 0.0
        self.winner_to_loser = False

    def update(self, current_price: float):
        """Update with current market price during open position."""
        # Defensive: Check entry_price validity
        if self.entry_price is None or self.entry_price <= 0:
            LOG.debug("[MFE_MAE] Cannot update position %s - invalid entry_price", self.position_id)
            return

        # Defensive: Validate current_price
        if current_price is None or current_price <= 0:
            LOG.debug(
                "[MFE_MAE] Cannot update position %s - invalid current_price=%.5f", self.position_id, current_price or 0
            )
            return

        # Ensure both prices are float for safe arithmetic
        try:
            cp = float(current_price)
            ep = float(self.entry_price)
        except (ValueError, TypeError) as e:
            LOG.warning("[MFE_MAE] Price conversion error for position %s: %s", self.position_id, e)
            return

        # Calculate P&L
        pnl = (cp - ep) if self.direction == 1 else (ep - cp)  # long: cp-ep, short: ep-cp

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
            "winner_to_loser": self.winner_to_loser,
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
    def __init__(self, symbol_id: int, qty: float, timeframe_minutes: int = 15, symbol: str = "XAUUSD"):
        super().__init__()

        self.symbol = symbol  # Instrument-agnostic: BTCUSD, XAUUSD, etc.
        self.symbol_id = symbol_id  # Numeric symbol identifier for FIX messages

        # Symbol ID lookup cache: symbol name -> broker ID (populated from SecurityList)
        self.symbol_id_cache: dict[str, int] = {}
        self.symbol_id_pending = symbol_id == 0  # True if we need to look up symbol ID
        self.security_list_received = False  # True after SecurityList response received
        self.security_list_request_time: float | None = None  # Timestamp of SecurityList request
        self.security_list_timeout = 5.0  # Seconds to wait for SecurityList before fallback
        self.quote_subscription_deferred = False  # True if we deferred MD subscription

        # Initialize learned parameters manager (single source of truth)
        self.param_manager = LearnedParametersManager()
        self.param_manager.load()

        self.timeframe_minutes = timeframe_minutes
        self.timeframe_label = f"M{timeframe_minutes}"
        self.broker = "default"

        # Timeframe-aware scaling constants.
        # All "bar count" thresholds are derived from wall-clock durations so
        # H4/D1 bots behave correctly without manual env-var tuning.
        self._bars_per_day = max(1, int(24 * 60 / timeframe_minutes))
        # Minimum bars before state vector is meaningful (fast MA=10, slow MA=30).
        # Keep 70 for M5 (6h warmup); scale down for higher timeframes so the
        # agent isn't blind for 11+ days at H4.
        self._min_bars_for_features = max(35, min(70, self._bars_per_day * 2))

        # Risk scaffolding defaults + env overrides
        self.starting_equity = float(os.environ.get("CTRADER_STARTING_EQUITY", "10000"))
        self.max_leverage = float(os.environ.get("CTRADER_MAX_LEVERAGE", "10"))
        self.contract_size = float(os.environ.get("CTRADER_CONTRACT_SIZE", "100000"))
        self.last_estimated_var = 0.0
        self.last_risk_cap_qty = 0.0
        self.last_base_qty = 0.0
        self.last_final_qty = 0.0

        # Learned parameters (primary control plane)
        self.qty = self._resolve_param("CTRADER_BASE_POSITION_SIZE", "base_position_size", qty)
        self.risk_budget_usd = self._resolve_param("CTRADER_RISK_BUDGET_USD", "risk_budget_usd", 100.0)
        self.vol_ref = self._resolve_param("CTRADER_VOL_REF", "volatility_reference", 0.005)
        self.vol_cap = self._resolve_param("CTRADER_VOL_CAP", "volatility_cap", 0.05)
        self.vpin_z_threshold = self._resolve_param("CTRADER_VPIN_Z", "vpin_z_threshold", 2.5)
        self.vpin_bucket_volume = self._resolve_param("CTRADER_VPIN_BUCKET", "vpin_bucket_volume", 25.0)
        LOG.info("[INIT] Position size=%.4f | Risk budget=$%.0f | Vol ref=%.4f cap=%.4f | VPIN z=%.1f bucket=%.0f",
            self.qty, self.risk_budget_usd, self.vol_ref, self.vol_cap, self.vpin_z_threshold, self.vpin_bucket_volume)

        self.quote_sid = None
        self.trade_sid = None

        # Phase 3.5: Path geometry for trigger features (gamma, jerk, runway)
        self.path_geometry = PathGeometry()

        # Friction calculator (source of truth: cTrader SymbolInfo)
        # MUST be created before DualPolicy so it can be passed to HarvesterAgent
        self.friction_calculator = FrictionCalculator(
            symbol=symbol,
            symbol_id=symbol_id,
            timeframe=self.timeframe_label,
            broker="default",
            param_manager=self.param_manager,
        )
        if os.environ.get("CTRADER_CONTRACT_SIZE") is None:
            self.contract_size = max(self.friction_calculator.costs.contract_size or 1.0, 1.0)

        # Phase 3: Dual-agent policy (TriggerAgent + HarvesterAgent)
        # Falls back to simple policy if DDQN_DUAL_AGENT=0
        use_dual_agent = os.environ.get("DDQN_DUAL_AGENT", "1") == "1"
        enable_online_learning = os.environ.get("DDQN_ONLINE_LEARNING", "1") == "1"
        if use_dual_agent:
            self.policy = DualPolicy(
                window=64,
                enable_regime_detection=True,
                path_geometry=self.path_geometry,
                enable_training=enable_online_learning,
                param_manager=self.param_manager,
                symbol=symbol,
                timeframe=self.timeframe_label,
                broker="default",
                friction_calculator=self.friction_calculator,
                timeframe_minutes=timeframe_minutes,
                min_bars_for_features=self._min_bars_for_features,
            )
            LOG.info("[POLICY] DualPolicy: TriggerAgent + HarvesterAgent | training=%s", enable_online_learning)

            # Load checkpoint from previous session (if exists)
            if enable_online_learning and hasattr(self.policy, "load_checkpoint"):
                self.policy.load_checkpoint()
        else:
            self.policy = Policy()  # type: ignore[assignment]

        self.bars: deque = deque(maxlen=2000)
        self.builder = BarBuilder(timeframe_minutes)

        # Phase 2: Non-repaint discipline + O(1) rolling stats
        self.close_series = NonRepaintBarAccess("close", max_lookback=2000)
        self.high_series = NonRepaintBarAccess("high", max_lookback=2000)
        self.low_series = NonRepaintBarAccess("low", max_lookback=2000)
        self.volume_series = NonRepaintBarAccess("volume", max_lookback=2000)
        self.non_repaint_series = [
            self.close_series,
            self.high_series,
            self.low_series,
            self.volume_series,
        ]
        self.close_stats = RollingStats(period=100)
        self.current_bar_tick_count = 0
        self.mfe_mae_tracker = MFEMAETracker()
        self.path_recorder = PathRecorder()
        # Multi-position tracking (keyed by position_id)
        # FIX: Initialize as EMPTY dict - don't create default tracker that prevents epsilon-greedy exploration
        self.default_position_id = "default"
        self.mfe_mae_trackers: dict[str, MFEMAETracker] = (
            {}
        )  # FIXED: was {self.default_position_id: self.mfe_mae_tracker}
        self.path_recorders: dict[str, PathRecorder] = {}  # FIXED: was {self.default_position_id: self.path_recorder}
        self._tracker_lock = threading.Lock()  # Protects mfe_mae_trackers & path_recorders across FIX callbacks
        self.performance = PerformanceTracker()
        self.prod_monitor = ProductionMonitor(
            metrics_file=Path("data/production_metrics.json"),
            alert_drawdown_pct=0.10,
            http_enabled=False,
        )
        self.trade_exporter = TradeExporter(output_dir="trades")  # Save to trades/ directory
        self.last_export_count = 0  # Track last export to avoid duplicates
        self.bar_count = 0  # Track bars for periodic auto-save
        self.last_autosave_bar = 0  # Last bar when auto-save occurred

        # Activity monitor - prevent learned helplessness
        self.activity_monitor = ActivityMonitor(max_bars_inactive=100, min_trades_per_day=2.0, exploration_boost=0.1)

        # Pass param_manager to reward_shaper for DRY and align instrumentation context
        self.reward_shaper = RewardShaper(
            symbol=symbol,
            timeframe=self.timeframe_label,
            broker="default",
            param_manager=self.param_manager,
            activity_monitor=self.activity_monitor,
        )

        # Risk management - VaR estimator with kurtosis circuit breaker
        self.kurtosis_monitor = KurtosisMonitor(window=100, threshold=3.0)
        self.var_estimator = VaREstimator(window=500, confidence=0.95, kurtosis_monitor=self.kurtosis_monitor)
        self.var_estimator.set_reference_vol(self.vol_ref)

        # Broker execution model for realistic slippage and execution costs
        self.execution_model = BrokerExecutionModel(
            typical_spread_bps=5.0,
            base_slippage_bps=2.0,
            volatile_multiplier=2.0,
            trending_multiplier=1.5,
            mean_reverting_multiplier=0.8,
        )

        # Adaptive regularization for online learning
        self.adaptive_reg = AdaptiveRegularization(
            initial_l2=0.0001, initial_dropout=0.1, l2_range=(1e-5, 1e-2), dropout_range=(0.0, 0.5)
        )

        # Handbook Phase 1: Circuit Breakers (safety shutdown system)
        self.circuit_breakers = CircuitBreakerManager(
            symbol=symbol,
            timeframe=self.timeframe_label,
            broker="default",
            param_manager=self.param_manager,
        )

        # Restore circuit breaker state from previous session
        self.circuit_breakers.restore_state()
        LOG.info("[INIT] Circuit breakers: Sortino>=%.1f Kurtosis<=%.0f DD<=%.0f%% MaxLoss=%d",
            self.circuit_breakers.sortino_breaker.threshold,
            self.circuit_breakers.kurtosis_breaker.threshold,
            self.circuit_breakers.max_drawdown * 100,
            self.circuit_breakers.max_consecutive_losses)

        # Event-relative time features
        self.event_time_engine = EventTimeFeatureEngine()

        # Training frequency: train every N bars
        self.training_interval = 5  # bars
        self.bars_since_training = 0
        self.last_trigger_loss = 0.0  # Track last reported training loss for HUD
        self.last_harvester_loss = 0.0

        self.previous_sharpe = 0.0  # Track for adaptive weight updates

        depth_env = os.environ.get("CTRADER_ORDERBOOK_DEPTH", "").strip()
        try:
            learned_depth = max(1, int(round(self.friction_calculator.depth_levels)))
            order_book_depth = int(depth_env) if depth_env else learned_depth
        except ValueError:
            order_book_depth = 10
        self.order_book = OrderBook(depth=order_book_depth)
        # Maps MDEntryID (tag 278) → (side, price) so delete-by-ID can resolve price.
        self._md_entry_id_map: dict[str, tuple[str, float]] = {}
        self._last_ob_export_time: float = 0.0  # Rate-limit for order_book.json writes
        self.vpin_calculator = VPINCalculator(bucket_volume=max(self.vpin_bucket_volume, 1e-6), window=50)
        self.last_vpin_stats = {"vpin": 0.0, "mean": 0.0, "std": 0.0, "zscore": 0.0}
        self.last_vpin_mid: float | None = None
        self.last_depth_metrics = {"bid": 0.0, "ask": 0.0, "ratio": 1.0, "levels": order_book_depth}
        self.last_depth_gate = False
        self.last_depth_floor = self.friction_calculator.depth_buffer

        self.best_bid: float | None = None
        self.best_ask: float | None = None

        # Position tracking - will delegate to TradeManager after initialization
        self._cur_pos_fallback = 0  # Only used before TradeManager initialized
        self.pos_req_id = None
        self.clord_counter = 0
        self.trade_entry_time = None  # Track entry time for performance metrics

        # Online learning state storage (multi-position support)
        self.entry_states: dict[str, Any] = {}  # order_id -> state at entry
        self.entry_actions: dict[str, int] = {}  # order_id -> action (0=NO, 1=LONG, 2=SHORT)
        self.predicted_runways: dict[str, float] = {}  # FIX 1: order_id -> predicted MFE for TriggerAgent reward

        # Backward compatibility - default entry for single-position mode
        self.entry_state = None  # Will be deprecated
        self.entry_action = None  # Will be deprecated
        self.predicted_runway = 0.0  # FIX 1: Track predicted MFE for backward compatibility
        self.entry_confidence = 0.5  # Calibrated confidence at entry (for Platt update at close)
        self.entry_var = 0.0          # VaR at entry time (for regime-conditioned reward)
        # EMA-averaged confidence for HUD production_metrics export (α=0.1 ≈ 10-trade window)
        self._last_trigger_conf = 0.5
        self._last_harvester_conf = 0.5
        self.entry_vpin_z = 0.0       # VPIN z-score at entry time (for regime-conditioned reward)

        # Prediction-vs-actual convergence tracking (EMA α=0.1 ≈ 10-trade window)
        # runway_delta: (predicted_runway_pts - actual_mfe_pts) → 0 = perfect prediction
        # runway_accuracy: [0,1] → 1 = perfect prediction
        # conf_calib_err: |confidence - actual_win| → 0 = well-calibrated probability
        self._runway_delta_ema: float = 0.0
        self._runway_accuracy_ema: float = 0.5
        self._conf_calib_err_ema: float = 0.5

        # Trade timing — persistent across sessions via trade_log.jsonl
        # Seeded from trade_log at startup (see _seed_trade_timing_from_log).
        self._last_trade_close_ts: float | None = None  # epoch seconds of last trade close
        self._avg_trade_duration_mins: float = 0.0      # EMA of trade durations (α=0.1)

        # Harvester experience tracking (dense feedback every bar)
        self.prev_harvester_state = None
        self.prev_exit_action = None
        self.prev_mfe = 0.0
        self.prev_mae = 0.0

        # Connection health monitoring - ROCK SOLID for financial trading
        self.last_quote_heartbeat = None
        self.last_trade_heartbeat = None
        self.heartbeat_timeout = 45  # seconds (3x heartbeat interval)
        self.connection_healthy = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 100  # Increased - keep trying with failover hosts
        self._shutdown_requested = False
        self._shutdown_complete = False

        # Stale market data watchdog - auto-resubscribe after rollover/gap
        self.last_market_data_time = None  # Updated on every md_snapshot / md_incremental
        self.market_data_stale_threshold = 300  # seconds (5 min) with no data triggers resubscribe
        self._last_md_resubscribe_time = 0.0  # cooldown to avoid spamming

        # Connection statistics for monitoring
        self.total_reconnects = 0
        self.successful_reconnects = 0
        self.connection_uptime_start = None
        self.longest_uptime = 0
        self.last_disconnect_reason = None

        # Reconnection backoff (exponential with jitter)
        self.reconnect_base_delay = 2  # seconds (faster initial retry)
        self.reconnect_max_delay = 60  # 1 minute max (failover hosts handle longer outages)
        self.last_reconnect_time = None
        restart_cooldown = float(os.environ.get("CTRADER_FORCE_RESTART_COOLDOWN", "120"))
        self.force_restart_cooldown = max(restart_cooldown, 30.0)
        self.last_forced_restart: dict[str, float | None] = {"QUOTE": None, "TRADE": None}

        # Stale data protection - block trading if data too old
        self.max_quote_age_for_trading = 30  # seconds - don't trade on stale prices

        # Systemd watchdog support
        self.watchdog_enabled = os.environ.get("WATCHDOG_USEC") is not None
        self.watchdog_interval: float | None
        if self.watchdog_enabled:
            self.watchdog_interval = int(os.environ.get("WATCHDOG_USEC", "0")) / 1000000 / 2
        else:
            self.watchdog_interval = None

        # Start health monitor thread
        self._health_monitor_running = True
        self._health_thread = threading.Thread(target=self._monitor_connection_health, daemon=True)
        self._health_thread.start()

        # TradeManager integration - centralized order & position management
        self.trade_integration = TradeManagerIntegration(self)

        # Emergency position closer for circuit breaker integration
        emergency_closer = create_emergency_closer(self.trade_integration)
        self.circuit_breakers.set_emergency_closer(emergency_closer)

        # Component health tracking for graceful degradation
        self.components_healthy = {
            "quote_feed": True,
            "trade_session": True,
            "trademanager": True,
            "mfe_tracker": True,
            "performance": True,
            "policy": True,
            "circuit_breakers": True,
            "path_recorder": True,
        }
        self.component_error_counts = dict.fromkeys(self.components_healthy, 0)
        self.max_component_errors = 5

        # Audit logging for transaction trail and decision debugging
        self.transaction_log = TransactionLogger(log_dir="log", filename="transactions.jsonl")
        self.decision_log = DecisionLogger(log_dir="log", filename="decisions.jsonl")
        
        LOG.info("[INIT] ✓ Bot initialized: %s (ID:%d) M%d | Contract=%.0f | Online learning=%s",
            symbol, symbol_id, timeframe_minutes, self.contract_size, enable_online_learning)

        # HUD data export tracking - initialize early to avoid AttributeError
        self.start_time = dt.datetime.now(dt.UTC)
        self.hud_data_dir = Path("data")
        self.hud_data_dir.mkdir(exist_ok=True)

    @property
    def cur_pos(self) -> int:
        """Current position: +1=LONG, 0=FLAT, -1=SHORT (delegates to TradeManager)"""
        if self.trade_integration.trade_manager:
            return self.trade_integration.trade_manager.get_position_direction(min_qty=self.qty * 0.5)
        return self._cur_pos_fallback

    @cur_pos.setter
    def cur_pos(self, value: int):
        """Set position (only used before TradeManager initialized)"""
        if self.trade_integration.trade_manager:
            LOG.warning("[POSITION] Direct cur_pos assignment ignored - TradeManager is source of truth")
        else:
            self._cur_pos_fallback = value

    # ---- connection health monitoring ----
    def _flush_production_metrics(self):
        """Gather live metrics from all subsystems and write to data/production_metrics.json."""
        try:
            m = self.performance.get_metrics() if hasattr(self, "performance") else {}
            cb = self.circuit_breakers.get_status() if hasattr(self, "circuit_breakers") else {}
            tripped_names = [
                name
                for name in ("sortino", "kurtosis", "drawdown", "consecutive_losses")
                if cb.get(name, {}).get("tripped")
            ]
            self.prod_monitor.update_metrics(
                realized_pnl_day=m.get("total_pnl", 0.0),
                realized_pnl_total=m.get("total_pnl", 0.0),
                unrealized_pnl=0.0,
                drawdown_current=m.get("current_drawdown", 0.0),
                drawdown_max=m.get("max_drawdown", 0.0),
                trades_today=m.get("total_trades", 0),
                trades_total=m.get("total_trades", 0),
                win_rate=m.get("win_rate", 0.0),
                avg_profit=m.get("avg_winner", 0.0),
                avg_loss=abs(m.get("avg_loser", 0.0)),
                avg_trade_duration_mins=self._avg_trade_duration_mins,
                last_trade_mins_ago=(
                    (time.time() - self._last_trade_close_ts) / 60.0
                    if self._last_trade_close_ts is not None
                    else 0.0
                ),
                trigger_confidence_avg=getattr(self, "_last_trigger_conf", 0.0),
                harvester_confidence_avg=getattr(self, "_last_harvester_conf", 0.0),
                runway_delta_ema=getattr(self, "_runway_delta_ema", 0.0),
                runway_accuracy_ema=getattr(self, "_runway_accuracy_ema", 0.5),
                conf_calib_err_ema=getattr(self, "_conf_calib_err_ema", 0.5),
                platt_a=getattr(getattr(getattr(self, "policy", None), "trigger", None), "platt_a", 1.0),
                platt_b=getattr(getattr(getattr(self, "policy", None), "trigger", None), "platt_b", 0.0),
                circuit_breakers_tripped=len(tripped_names),
                circuit_breaker_names=tripped_names,
                fix_connected=self.connection_healthy,
            )
        except Exception as e:
            LOG.warning("[METRICS] Failed to flush production metrics: %s", e)

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
                        if quote_age > self.heartbeat_timeout * 2:
                            self._force_session_restart(self.quote_sid, "QUOTE", f"heartbeat stale {quote_age:.0f}s")
                elif self.quote_sid is None and self.last_quote_heartbeat is not None:
                    # Session was lost
                    issues.append("QUOTE disconnected")

                # Check TRADE session
                if self.trade_sid and self.last_trade_heartbeat:
                    trade_age = (now - self.last_trade_heartbeat).total_seconds()
                    if trade_age > self.heartbeat_timeout:
                        issues.append(f"TRADE stale ({trade_age:.0f}s)")
                        self._try_send_test_request(self.trade_sid, "TRADE")
                        if trade_age > self.heartbeat_timeout * 2:
                            self._force_session_restart(self.trade_sid, "TRADE", f"heartbeat stale {trade_age:.0f}s")
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

                # ---- Stale market data watchdog ----
                # If QUOTE session is alive but no market data for >threshold,
                # re-subscribe (handles rollover subscription drops)
                if self.quote_sid and self.last_market_data_time is not None and self.connection_healthy:
                    md_age = time.time() - self.last_market_data_time
                    resubscribe_cooldown = 120  # seconds between resubscribe attempts
                    if (
                        md_age > self.market_data_stale_threshold
                        and time.time() - self._last_md_resubscribe_time > resubscribe_cooldown
                    ):
                        LOG.warning(
                            "[HEALTH] Market data stale for %.0fs (threshold=%ds) — re-subscribing spot",
                            md_age,
                            self.market_data_stale_threshold,
                        )
                        try:
                            self.send_md_subscribe_spot()
                            self._last_md_resubscribe_time = time.time()
                        except Exception as e:
                            LOG.error("[HEALTH] Failed to re-subscribe market data: %s", e)

                # Systemd watchdog notification (if enabled)
                if self.watchdog_enabled and self.connection_healthy:
                    try:
                        # Notify systemd that we're alive
                        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                        sock.sendto(b"WATCHDOG=1", os.environ.get("NOTIFY_SOCKET", ""))
                        sock.close()
                    except Exception as e:
                        LOG.debug("[WATCHDOG] Failed to notify systemd: %s", e)

                # Periodic health log (every ~60s)
                health_log_counter += 1
                if health_log_counter % 6 == 0:  # Every ~60s
                    quote_status = "OK" if self.quote_sid else "DOWN"
                    trade_status = "OK" if self.trade_sid else "DOWN"

                    # Calculate current uptime
                    if self.connection_uptime_start:
                        current_uptime = time.time() - self.connection_uptime_start
                        uptime_str = f"{current_uptime/60:.1f}min"
                    else:
                        uptime_str = "N/A"

                    LOG.info(
                        "[HEALTH] Status: QUOTE=%s TRADE=%s healthy=%s uptime=%s reconnects=%d/%d",
                        quote_status,
                        trade_status,
                        self.connection_healthy,
                        uptime_str,
                        self.successful_reconnects,
                        self.total_reconnects,
                    )

                    # Flush production metrics on every 60s health tick
                    self._flush_production_metrics()

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

    def _force_session_restart(self, session_id, qual: str, reason: str = "manual") -> bool:
        """Force a FIX session reconnect when heartbeats stall."""
        if not session_id:
            return False

        last_restart = self.last_forced_restart.get(qual)
        now_ts = time.time()
        if last_restart and now_ts - last_restart < self.force_restart_cooldown:
            remaining = self.force_restart_cooldown - (now_ts - last_restart)
            LOG.debug("[RECONNECT] %s restart skipped (cooldown %.1fs left)", qual, remaining)
            return False

        try:
            session = fix.Session.lookupSession(session_id)
            if session:
                LOG.warning("[RECONNECT] Forcing %s session restart (%s)", qual, reason)
                session.disconnect(reason, False)
                self.last_forced_restart[qual] = now_ts
                self.last_disconnect_reason = str(reason)
                return True
            LOG.error("[RECONNECT] lookupSession failed for %s", qual)
        except Exception as e:
            LOG.error("[RECONNECT] Error forcing %s restart: %s", qual, e, exc_info=True)
        return False

    def stop_health_monitor(self):
        """Gracefully stop the health monitor thread."""
        self._health_monitor_running = False
        self._shutdown_requested = True

    def graceful_shutdown(self, signum=None, frame=None):  # noqa: ARG002
        """Perform graceful shutdown with position cleanup and session closure."""
        if self._shutdown_complete:
            return

        signal_name = signal.Signals(signum).name if signum else "MANUAL"
        LOG.info("[SHUTDOWN] Initiated (%s)", signal_name)
        self._shutdown_requested = True

        try:
            # Check for open positions
            if hasattr(self, "cur_pos") and self.cur_pos != 0:
                LOG.warning("[SHUTDOWN] ⚠ Open %s position (qty:%s) - manual closure required",
                    "LONG" if self.cur_pos > 0 else "SHORT", abs(self.cur_pos))

            # Export trades
            if hasattr(self, "trade_exporter") and hasattr(self, "performance"):
                try:
                    trades = self.performance.get_trade_history()
                    if trades:
                        timestamp = datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
                        self.trade_exporter.export_all(self.performance, prefix=f"shutdown_{timestamp}")
                        LOG.info("[SHUTDOWN] ✓ Exported %d trades", len(trades))
                except Exception as e:
                    LOG.error("[SHUTDOWN] Export failed: %s", e)

            # Save training checkpoint
            if hasattr(self, "policy") and self.policy and hasattr(self.policy, "save_checkpoint"):
                try:
                    self.policy.save_checkpoint()
                    LOG.info("[SHUTDOWN] ✓ Training checkpoint saved")
                except Exception as e:
                    LOG.error("[SHUTDOWN] Checkpoint failed: %s", e)

            # Final stats
            bars = len(self.bars) if hasattr(self, "bars") else 0
            pos = self.cur_pos if hasattr(self, "cur_pos") else 0
            metrics = self.performance.get_metrics() if hasattr(self, "performance") else {}
            LOG.info("[SHUTDOWN] Stats: bars=%d pos=%s pnl=%.2f trades=%d reconnects=%d",
                bars, pos, metrics.get("total_pnl", 0.0), metrics.get("total_trades", 0), self.reconnect_attempts)

            self.stop_health_monitor()

        except Exception as e:
            LOG.error("[SHUTDOWN] Error: %s", e, exc_info=True)
        finally:
            self._shutdown_complete = True
            LOG.info("[SHUTDOWN] Complete")

    def is_fully_connected(self) -> bool:
        """Check if both QUOTE and TRADE sessions are active."""
        return self.quote_sid is not None and self.trade_sid is not None

    def get_connection_status(self) -> dict:
        """Get detailed connection status for monitoring/logging."""
        now = utc_now()
        return {
            "quote_connected": self.quote_sid is not None,
            "trade_connected": self.trade_sid is not None,
            "quote_age_s": ((now - self.last_quote_heartbeat).total_seconds() if self.last_quote_heartbeat else None),
            "trade_age_s": ((now - self.last_trade_heartbeat).total_seconds() if self.last_trade_heartbeat else None),
            "connection_healthy": self.connection_healthy,
            "reconnect_attempts": self.reconnect_attempts,
        }

    # ---- helpers ----
    @staticmethod
    def _qual(session_id) -> str:
        # Route strictly by SessionQualifier (NOT SubIDs).
        try:
            q = session_id.getSessionQualifier()
            return (q or "").upper()
        except Exception as e:
            LOG.warning("[SESSION] Unable to get qualifier: %s", e)
            return ""

    # ---- session events ----
    def onCreate(self, session_id):
        LOG.info("[CREATE] %s qual=%s", session_id.toString(), self._qual(session_id))

    def onLogon(self, session_id):
        qual = self._qual(session_id)
        LOG.info("[LOGON] %s qual=%s", session_id.toString(), qual)

        # GAP 10.1: Log session event
        self.transaction_log.log_session_event(qual, "LOGON", {"session_id": session_id.toString()})

        # Track reconnection success
        if self.reconnect_attempts > 0:
            self.successful_reconnects += 1
            LOG.info(
                "[RECONNECT] ✓ Successfully reconnected %s after %d attempts",
                qual,
                self.reconnect_attempts,
            )

        # Reset reconnect counter on successful logon
        self.reconnect_attempts = 0
        self.connection_healthy = True

        # Track uptime
        if self.connection_uptime_start is None:
            self.connection_uptime_start = time.time()

        try:
            if qual == "QUOTE":
                self.quote_sid = session_id
                self.last_quote_heartbeat = utc_now()
                # Only subscribe if symbol ID is resolved
                if not self.symbol_id_pending:
                    self.send_md_subscribe_spot()
                else:
                    self.quote_subscription_deferred = True
                    LOG.info("[QUOTE] Deferring market data subscription until symbol ID resolved")
            elif qual == "TRADE":
                self.trade_sid = session_id
                self.last_trade_heartbeat = utc_now()

                # If symbol_id is 0 (AUTO mode), request SecurityList first
                if self.symbol_id_pending:
                    LOG.info("[TRADE] Symbol ID pending - requesting SecurityList for %s lookup", self.symbol)
                    self.request_security_list()
                    return  # Will continue in on_security_list -> _complete_trade_session_startup

                # Normal flow: proceed with trade session startup
                self._complete_trade_session_startup()
            else:
                LOG.warning("[LOGON] Unknown qualifier; not routing: %s", session_id.toString())
        except Exception as e:
            LOG.error("[LOGON] Error during session setup for %s: %s", qual, e, exc_info=True)

    def _seed_trade_timing_from_log(self) -> None:
        """Seed _last_trade_close_ts and _avg_trade_duration_mins from trade_log.jsonl.

        Called once at startup so HUD stats survive a bot restart.
        Reads the last 50 entries (most recent trades) and computes:
        - _last_trade_close_ts: unix timestamp of the most recent exit
        - _avg_trade_duration_mins: EMA (α=0.1) over trade durations
        """
        try:
            import pathlib
            trade_log_path = pathlib.Path(self.hud_data_dir) / "trade_log.jsonl"
            if not trade_log_path.exists():
                return

            # Efficiently read last 50 lines without loading the whole file
            lines: list[str] = []
            with open(trade_log_path, "rb") as fh:
                # Seek backward to collect up to 50 newline-terminated records
                fh.seek(0, 2)
                file_size = fh.tell()
                if file_size == 0:
                    return
                buf_size = min(file_size, 64 * 1024)
                fh.seek(-buf_size, 2)
                raw = fh.read(buf_size).decode("utf-8", errors="replace")
                lines = [l for l in raw.splitlines() if l.strip()][-50:]

            trades: list[dict] = []
            for line in lines:
                try:
                    rec = json.loads(line)
                    if rec.get("exit_time") and rec.get("entry_time"):
                        trades.append(rec)
                except json.JSONDecodeError:
                    continue

            if not trades:
                return

            # Sort by exit_time ascending
            import datetime as dt_mod
            def _parse_ts(s: str) -> float:
                try:
                    return dt_mod.datetime.fromisoformat(s).timestamp()
                except Exception:
                    return 0.0

            trades.sort(key=lambda t: _parse_ts(t["exit_time"]))

            # Seed last_trade_close_ts from most recent exit
            most_recent_exit = _parse_ts(trades[-1]["exit_time"])
            if most_recent_exit > 0:
                self._last_trade_close_ts = most_recent_exit

            # Build duration EMA over all retrieved trades
            avg = 0.0
            for t in trades:
                entry_ts = _parse_ts(t["entry_time"])
                exit_ts = _parse_ts(t["exit_time"])
                if entry_ts > 0 and exit_ts > entry_ts:
                    dur = (exit_ts - entry_ts) / 60.0
                    avg = 0.9 * avg + 0.1 * dur
            if avg > 0:
                self._avg_trade_duration_mins = avg

            LOG.info(
                "[STARTUP] Seeded trade timing from %d log entries: "
                "last_trade=%.1f min ago, avg_dur=%.1f min",
                len(trades),
                (time.time() - self._last_trade_close_ts) / 60.0
                if self._last_trade_close_ts
                else 0.0,
                self._avg_trade_duration_mins,
            )
        except Exception as e:
            LOG.warning("[STARTUP] Could not seed trade timing from log: %s", e)

    def _complete_trade_session_startup(self):
        """Complete TRADE session startup after symbol ID is resolved.

        Called either:
        1. Directly from onLogon if symbol_id was provided
        2. From on_security_list after symbol_id lookup succeeds
        """
        LOG.info("[TRADE] Completing trade session startup for symbol_id=%d", self.symbol_id)
        self._seed_trade_timing_from_log()  # Restore last_trade / avg duration from log

        # If we deferred quote subscription, do it now that symbol ID is resolved
        if self.quote_subscription_deferred and self.quote_sid:
            LOG.info("[QUOTE] Subscribing to market data for resolved symbol_id=%d", self.symbol_id)
            self.send_md_subscribe_spot()
            self.quote_subscription_deferred = False

        self.request_security_definition()  # Get symbol info (source of truth)

        # Initialize TradeManager now that TRADE session is connected
        # TradeManager will handle position request
        if not self.trade_integration.initialize_trade_manager():
            LOG.error("[INTEGRATION] TradeManager initialization failed - trading disabled")
            return

        # Seed position from env var ONLY if not recovered from persistence
        # This handles the case where broker doesn't respond to position requests
        # Set INITIAL_POSITION=0.4 for 4 LONG positions of 0.1 each, or -0.4 for SHORT
        if not getattr(self.trade_integration, "position_recovered", False):
            initial_pos_str = os.environ.get("INITIAL_POSITION", "0")
            try:
                initial_pos = float(initial_pos_str)
                if initial_pos != 0 and self.trade_integration.trade_manager:
                    self.trade_integration.trade_manager.seed_position(initial_pos)
                    LOG.info("[STARTUP] 🌱 Seeded position from INITIAL_POSITION=%.4f", initial_pos)
            except ValueError:
                LOG.warning("[STARTUP] Invalid INITIAL_POSITION=%r - ignoring", initial_pos_str)
        else:
            LOG.info("[STARTUP] ✓ Position recovered from persistence - skipping INITIAL_POSITION")

        # P0 FIX: Handle reconnect recovery (query pending orders, reconcile positions)
        if self.trade_integration.trade_manager:
            self.trade_integration.trade_manager.on_logon()

        # Restore bar history so realized_vol / path geometry work immediately.
        self._load_bars_cache()

        # Write initial training stats immediately at startup so HUD shows
        # correct buffer sizes before the first bar closes.
        try:
            self._export_hud_data()
        except Exception as e_hud:
            LOG.debug("[STARTUP] Initial HUD export deferred: %s", e_hud)

    def onLogout(self, session_id):
        qual = self._qual(session_id)

        # GAP 10.1: Log session event
        self.transaction_log.log_session_event(qual, "LOGOUT", {"session_id": session_id.toString()})

        # Track uptime before disconnect
        if self.connection_uptime_start:
            uptime = time.time() - self.connection_uptime_start
            self.longest_uptime = max(self.longest_uptime, uptime)
            LOG.warning("[LOGOUT] %s qual=%s (uptime was %.1f min)", session_id.toString(), qual, uptime / 60)
        else:
            LOG.warning("[LOGOUT] %s qual=%s", session_id.toString(), qual)

        # Mark session as down
        if qual == "QUOTE":
            self.quote_sid = None
        elif qual == "TRADE":
            self.trade_sid = None

        self.connection_healthy = False
        self.total_reconnects += 1
        self.reconnect_attempts += 1
        self.last_reconnect_time = time.time()
        self.connection_uptime_start = None  # Reset uptime tracking
        self.last_disconnect_reason = f"{qual} session logout"

        # Calculate exponential backoff with jitter for distributed retries
        jitter = random.uniform(0.5, 1.5)  # 50-150% of base delay
        backoff_delay = min(
            self.reconnect_base_delay * math.pow(2, min(self.reconnect_attempts - 1, 6)) * jitter,
            self.reconnect_max_delay,
        )

        if self.reconnect_attempts <= self.max_reconnect_attempts:
            LOG.info(
                "[RECONNECT] Will attempt reconnect via failover hosts (attempt %d/%d, backoff: %.1fs)",
                self.reconnect_attempts,
                self.max_reconnect_attempts,
                backoff_delay,
            )
            LOG.info(
                "[RECONNECT] Total reconnects this session: %d (successful: %d)",
                self.total_reconnects,
                self.successful_reconnects,
            )
            # QuickFIX will handle the reconnect automatically with ReconnectInterval from .cfg
            # Failover hosts will be tried in order: SocketConnectHost, SocketConnectHost1, SocketConnectHost2
        else:
            LOG.error(
                "[RECONNECT] Max attempts reached (%d). Manual intervention required.",
                self.max_reconnect_attempts,
            )
            LOG.error("[RECONNECT] Consider restarting the bot or checking network/credentials")

    # ---- admin hooks ----

    def toAdmin(self, message, session_id):
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)

        if msg_type.getValue() != fix.MsgType_Logon:
            return

        qual = self._qual(session_id)
        LOG.debug(f"[DEBUG] toAdmin called for session: {session_id}, qual: {qual}")

        # Reset seq nums
        message.setField(fix.ResetSeqNumFlag(True))

        user = os.environ.get("CTRADER_USERNAME", "").strip()
        if qual == "QUOTE":
            pwd = os.environ.get("CTRADER_PASSWORD_QUOTE", "").strip()
            message.getHeader().setField(fix.TargetSubID("QUOTE"))
            LOG.debug("[DEBUG] Set TargetSubID=QUOTE in logon header")
        else:
            pwd = os.environ.get("CTRADER_PASSWORD_TRADE", "").strip()
            message.getHeader().setField(fix.TargetSubID("TRADE"))
            LOG.debug("[DEBUG] Set TargetSubID=TRADE in logon header")

        if user:
            message.setField(fix.Username(user))
        if pwd:
            message.setField(fix.Password(pwd))

        LOG.info("[ADMIN][LOGON OUT] qual=%s %s", qual, _redact_fix(message.toString()))

    def fromAdmin(self, message, session_id):
        qual = self._qual(session_id)

        # Update heartbeat timestamp on ANY admin message
        now = utc_now()
        if qual == "QUOTE":
            self.last_quote_heartbeat = now
        elif qual == "TRADE":
            self.last_trade_heartbeat = now

        LOG.info("[ADMIN][IN] qual=%s %s", qual, _redact_fix(message.toString()))

    def toApp(self, message, session_id):
        qual = self._qual(session_id)
        # Keep this INFO until stable; you can reduce to DEBUG later.
        LOG.info("[APP][OUT] qual=%s %s", qual, _redact_fix(message.toString()))

    def fromApp(self, message, session_id):
        qual = self._qual(session_id)

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
            elif t == "y":
                self.on_security_list(message)  # Symbol list response (for ID lookup)
            elif t == "j":
                self.on_biz_reject(message)
            elif t == "Y":
                self.on_md_reject(message)
        except Exception as e:
            # CRITICAL: Catch ALL exceptions to prevent session crash
            LOG.error(
                "[APP] Message handler failed: %s\n%s",
                e,
                message.toString() if message else "no message",
                exc_info=True,
            )

    # ----------------------------
    # QUOTE: Market data subscribe
    # ----------------------------
    def send_md_subscribe_spot(self):
        if not self.quote_sid:
            return

        # Pepperstone cTrader FIX only accepts MarketDepth 0 (full book) or 1
        # (top-of-book). Use 0 to receive all available levels; the OrderBook
        # object (depth=5 by default) will keep only the top-N we care about.
        # CTRADER_MD_DEPTH_MAX is still honoured at the OrderBook level.
        req = fix44.MarketDataRequest()
        req.setField(fix.MDReqID(f"{self.symbol}_L2_FULL"))
        req.setField(fix.SubscriptionRequestType("1"))
        req.setField(fix.MarketDepth(0))   # 0 = full book
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
        LOG.info(
            "[QUOTE] Subscribed full L2 market data for symbolId=%s (OrderBook depth=%d)",
            self.symbol_id,
            getattr(self.order_book, "depth", 5),
        )

    # ----------------------------
    # TRADE: SecurityList (Symbol ID Lookup)
    # ----------------------------
    def request_security_list(self):
        """Request list of all available symbols from cTrader.

        Used to automatically look up symbol ID from symbol name.
        Sends SecurityListRequest (35=x) with SecurityListRequestType=4 (ALL_SECURITIES).

        NOTE: cTrader may not support SecurityListRequest. If no response within
        security_list_timeout seconds, falls back to config/symbol_specs.json lookup.
        """
        if not self.trade_sid:
            return

        req = fix44.SecurityListRequest()
        req.setField(fix.SecurityReqID(f"SECLIST_{uuid.uuid4().hex[:8]}"))
        _SECURITY_LIST_ALL_SYMBOLS = 4  # SecurityListRequestType tag 559
        req.setField(fix.IntField(559, _SECURITY_LIST_ALL_SYMBOLS))

        self.security_list_request_time = time.time()
        fix.Session.sendToTarget(req, self.trade_sid)
        LOG.info("[TRADE] Requested SecurityList (all symbols) for symbol lookup")

        # Schedule fallback check after timeout
        def check_security_list_timeout():
            time.sleep(self.security_list_timeout)
            if self.symbol_id_pending and not self.security_list_received:
                LOG.warning(
                    "[TRADE] SecurityList timeout after %.1fs - using config fallback", self.security_list_timeout
                )
                self._resolve_symbol_id_from_config()

        timeout_thread = threading.Thread(target=check_security_list_timeout, daemon=True)
        timeout_thread.start()

    def _resolve_symbol_id_from_config(self):
        """Fallback: Resolve symbol ID from config/symbol_specs.json when SecurityList fails."""
        try:
            config_path = Path("config/symbol_specs.json")
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    specs = json.load(f)

                symbol_upper = self.symbol.upper()
                if symbol_upper in specs and "symbol_id" in specs[symbol_upper]:
                    resolved_id = specs[symbol_upper]["symbol_id"]
                    self.symbol_id = resolved_id
                    self.symbol_id_pending = False
                    LOG.info("[TRADE] ✓ Resolved %s -> ID %d from symbol_specs.json", self.symbol, self.symbol_id)

                    # Proceed with startup
                    self._complete_trade_session_startup()
                    return

            LOG.error("[TRADE] ✗ Cannot resolve symbol ID for %s - not in symbol_specs.json", self.symbol)
            LOG.error("[TRADE] Please add '%s' with 'symbol_id' to config/symbol_specs.json", self.symbol)
        except Exception as e:
            LOG.error("[TRADE] Error loading symbol_specs.json: %s", e)

    def on_security_list(self, msg: fix.Message):
        """Parse SecurityList (35=y) response and populate symbol_id_cache.

        Maps symbol names to broker IDs for dynamic symbol resolution.
        """
        try:
            # Check for errors
            sec_req_result = fix.IntField(560)  # SecurityRequestResult
            if msg.isSetField(sec_req_result):
                msg.getField(sec_req_result)
                result = sec_req_result.getValue()
                _VALID_SECURITY_REQUEST = 0
                if result != _VALID_SECURITY_REQUEST:
                    LOG.error("[TRADE] SecurityList request failed with result=%d", result)
                    return

            # Get total count
            tot_related_sym = fix.TotNoRelatedSym()
            total = 0
            if msg.isSetField(tot_related_sym):
                msg.getField(tot_related_sym)
                total = tot_related_sym.getValue()
                LOG.info("[TRADE] SecurityList received with %d symbols", total)

            # Parse each symbol in the group
            no_related_sym = fix.NoRelatedSym()
            if msg.isSetField(no_related_sym):
                msg.getField(no_related_sym)
                count = int(no_related_sym.getValue())

                for i in range(count):
                    try:
                        # Extract symbol group
                        grp = fix44.SecurityList.NoRelatedSym()
                        msg.getGroup(i + 1, grp)  # Groups are 1-indexed

                        # Get Symbol name
                        symbol = fix.Symbol()
                        symbol_name = None
                        if grp.isSetField(symbol):
                            grp.getField(symbol)
                            symbol_name = symbol.getValue()

                        # Get SecurityID (broker's numeric ID)
                        security_id = fix.SecurityID()
                        sec_id = None
                        if grp.isSetField(security_id):
                            grp.getField(security_id)
                            try:
                                sec_id = int(security_id.getValue())
                            except ValueError:
                                sec_id = None

                        # Store in cache
                        if symbol_name and sec_id:
                            self.symbol_id_cache[symbol_name.upper()] = sec_id

                    except Exception as e:
                        LOG.debug("[TRADE] Failed to parse symbol group %d: %s", i, e)
                        continue

            self.security_list_received = True
            LOG.info("[TRADE] ✓ Symbol cache populated with %d symbols", len(self.symbol_id_cache))

            # Log a few examples
            examples = list(self.symbol_id_cache.items())[:5]
            for sym, sid in examples:
                LOG.debug("[TRADE]   %s -> %d", sym, sid)

            # If we were waiting for symbol ID, resolve it now
            if self.symbol_id_pending:
                resolved = self.lookup_symbol_id(self.symbol)
                if resolved:
                    self.symbol_id = resolved
                    self.symbol_id_pending = False
                    LOG.info("[TRADE] ✓ Resolved %s -> ID %d from SecurityList", self.symbol, self.symbol_id)
                    # Now proceed with the rest of the startup sequence
                    self._complete_trade_session_startup()
                else:
                    LOG.error("[TRADE] ✗ Failed to resolve symbol ID for %s - check symbol name", self.symbol)

        except Exception as e:
            LOG.error("[TRADE] Error parsing SecurityList: %s", e, exc_info=True)

    def lookup_symbol_id(self, symbol_name: str) -> int | None:
        """Look up symbol ID from cache by symbol name.

        Returns None if symbol not found in cache.
        """
        normalized = symbol_name.upper().strip()
        return self.symbol_id_cache.get(normalized)

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
                params["contract_size"] = float(contract_mult.getValue())
                LOG.info("  ✓ ContractMultiplier: %.0f", params["contract_size"])

            # Round lot (standard lot size)
            round_lot = fix.RoundLot()
            if msg.isSetField(round_lot):
                msg.getField(round_lot)
                params["volume_step"] = float(round_lot.getValue())
                LOG.info("  ✓ RoundLot (step): %.4f", params["volume_step"])

            # Min trade volume
            min_vol = fix.MinTradeVol()
            if msg.isSetField(min_vol):
                msg.getField(min_vol)
                params["min_volume"] = float(min_vol.getValue())
                LOG.info("  ✓ MinTradeVol: %.4f lots", params["min_volume"])

            # Max trade volume
            max_vol = fix.MaxTradeVol()
            if msg.isSetField(max_vol):
                msg.getField(max_vol)
                params["max_volume"] = float(max_vol.getValue())
                LOG.info("  ✓ MaxTradeVol: %.4f lots", params["max_volume"])

            # Fallback: try MinQty if MinTradeVol not present
            if "min_volume" not in params:
                min_qty = fix.MinQty()
                if msg.isSetField(min_qty):
                    msg.getField(min_qty)
                    params["min_volume"] = float(min_qty.getValue())
                    LOG.info("  ✓ MinQty: %.4f lots", params["min_volume"])

            # Price precision (number of decimals)
            price_method = fix.PriceQuoteMethod()
            if msg.isSetField(price_method):
                msg.getField(price_method)
                params["digits"] = int(price_method.getValue())
                LOG.info("  ✓ PriceQuoteMethod (digits): %d", params["digits"])

            # Currency (base currency)
            currency = fix.Currency()
            if msg.isSetField(currency):
                msg.getField(currency)
                params["currency"] = currency.getValue()
                LOG.info("  ✓ Currency: %s", params["currency"])

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
                self.friction_calculator.get_statistics()
                LOG.info(
                    "[TRADE] Friction summary: min=%.4f max=%.4f step=%.4f",
                    params.get("min_volume", 0),
                    params.get("max_volume", 0),
                    params.get("volume_step", 0),
                )
            else:
                LOG.warning("[TRADE] SecurityDefinition had no extractable parameters")

        except Exception as e:
            LOG.error("[TRADE] Error parsing SecurityDefinition: %s", e, exc_info=True)

    def on_md_snapshot(self, msg: fix.Message):
        self.last_market_data_time = time.time()
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

        self.order_book.reset()
        self._md_entry_id_map.clear()  # ID map is stale after a full snapshot
        for i in range(1, n + 1):
            try:
                g = fix44.MarketDataSnapshotFullRefresh().NoMDEntries()
                msg.getGroup(i, g)

                et = fix.MDEntryType()
                px = fix.MDEntryPx()
                qty = fix.MDEntrySize()
                if not (g.isSetField(et) and g.isSetField(px)):
                    continue

                g.getField(et)
                g.getField(px)

                price_f = float(px.getValue())

                # Defensive: Validate price is positive and reasonable
                if price_f <= 0 or price_f > MAX_PRICE_SANITY:  # Sanity check
                    LOG.warning("[QUOTE] Suspicious price %.2f in entry %d - skipping", price_f, i)
                    continue

                size = 0.0
                if g.isSetField(qty):
                    g.getField(qty)
                    size = float(qty.getValue())
                if size <= 0:
                    size = 1.0  # cTrader omits size at top-of-book

            except (ValueError, TypeError) as e:
                LOG.warning("[QUOTE] Invalid entry %d: %s", i, e)
                continue
            except Exception as e:
                LOG.error("[QUOTE] Unexpected error parsing entry %d: %s", i, e)
                continue

            entry_type = et.getValue()
            if entry_type == "0":
                self.order_book.update_level("BID", price_f, size)
            elif entry_type == "1":
                self.order_book.update_level("ASK", price_f, size)

        best_bid, best_ask = self.order_book.best_bid_ask()
        if best_bid is not None:
            self.best_bid = float(best_bid)
        if best_ask is not None:
            self.best_ask = float(best_ask)

        if best_bid is not None and best_ask is not None:
            self.friction_calculator.update_spread(self.best_bid, self.best_ask)

        # Evaluate Harvester on tick for faster exit execution
        self._evaluate_harvester_on_tick()

        self._export_order_book()
        self.try_bar_update()

    def on_md_incremental(self, msg: fix.Message):
        self.last_market_data_time = time.time()
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
            px = fix.MDEntryPx()
            qty = fix.MDEntrySize()

            et_set = g.isSetField(et)
            if et_set:
                g.getField(et)
            if g.isSetField(sym):
                g.getField(sym)

            if sym.getValue() and sym.getValue() != str(self.symbol_id):
                continue

            # Read MDEntryID (tag 278) — used for delete-by-ID resolution.
            md_id_field = fix.MDEntryID()
            md_id = None
            if g.isSetField(md_id_field):
                g.getField(md_id_field)
                md_id = md_id_field.getValue()

            if not et_set:
                # Delete-by-ID: broker omits MDEntryType — look up price from map.
                if action == "2" and md_id is not None:
                    entry = self._md_entry_id_map.pop(md_id, None)
                    if entry is not None:
                        side, price_f = entry
                        self.order_book.update_level(side, price_f, 0.0)
                continue

            entry_type = et.getValue()
            if action in ("0", "1"):  # new or change
                if not g.isSetField(px):
                    continue
                g.getField(px)
                try:
                    price_f = float(px.getValue())
                    size = 0.0
                    if g.isSetField(qty):
                        g.getField(qty)
                        size = float(qty.getValue())
                    if size <= 0:
                        size = 1.0
                except (ValueError, TypeError):
                    continue

                side = "BID" if entry_type == "0" else "ASK" if entry_type == "1" else None
                if side is None:
                    continue
                # If this ID was previously mapped to a different price, remove old level.
                if md_id is not None and action == "1":
                    old = self._md_entry_id_map.get(md_id)
                    if old is not None and old[1] != price_f:
                        self.order_book.update_level(old[0], old[1], 0.0)
                self.order_book.update_level(side, price_f, size)
                if md_id is not None:
                    self._md_entry_id_map[md_id] = (side, price_f)
            elif action == "2":  # delete with MDEntryType — price-keyed delete
                if not g.isSetField(px):
                    # No price either — try ID map as fallback.
                    if md_id is not None:
                        entry = self._md_entry_id_map.pop(md_id, None)
                        if entry is not None:
                            self.order_book.update_level(entry[0], entry[1], 0.0)
                    continue
                g.getField(px)
                try:
                    price_f = float(px.getValue())
                except (ValueError, TypeError):
                    continue
                side = "BID" if entry_type == "0" else "ASK" if entry_type == "1" else None
                if side is not None:
                    self.order_book.update_level(side, price_f, 0.0)
                if md_id is not None:
                    self._md_entry_id_map.pop(md_id, None)

        best_bid, best_ask = self.order_book.best_bid_ask()
        if best_bid is not None and best_ask is not None:
            self.best_bid, self.best_ask = float(best_bid), float(best_ask)
            self.friction_calculator.update_spread(self.best_bid, self.best_ask)

        self._export_order_book()
        self.try_bar_update()

    def _update_non_repaint_series(self, bar, tick_count: int):
        """Append closed bar data into the non-repaint guard series."""
        _, _, h, low_price, c = bar
        self.close_series.append(c)
        self.high_series.append(h)
        self.low_series.append(low_price)
        # Approximate volume by tick count within the bar (better than zero volume)
        volume_value = float(max(tick_count, 1))
        self.volume_series.append(volume_value)

    def _mark_non_repaint_closed(self):
        """Allow bar[0] access after bar close."""
        for series in self.non_repaint_series:
            series.mark_bar_closed()

    def _update_vpin(self, mid_price: float) -> None:
        """Update VPIN calculator using mid-price changes as trade proxies."""
        calc = getattr(self, "vpin_calculator", None)
        if calc is None or mid_price is None or mid_price <= 0:
            return
        if self.last_vpin_mid is None:
            self.last_vpin_mid = mid_price
            return
        price_delta = mid_price - self.last_vpin_mid
        costs = getattr(self, "friction_calculator", None)
        tick_size = getattr(costs.costs, "tick_size", 0.01) if costs and getattr(costs, "costs", None) else 0.01
        tick_size = max(tick_size, 1e-6)
        if abs(price_delta) < tick_size * 0.1:
            return
        side = "BUY" if price_delta > 0 else "SELL"
        volume = max(1.0, SafeMath.safe_div(abs(price_delta), tick_size, 1.0))
        calc.update(volume=volume, side=side)
        self.last_vpin_stats = calc.get_stats()
        self.last_vpin_mid = mid_price

    @staticmethod
    def _depth_is_too_thin(depth_bid: float, depth_ask: float, depth_floor: float) -> bool:
        """Return True when either side of the book breaches the learned depth floor."""
        if depth_floor <= 0:
            return False
        if depth_bid <= 0 or depth_ask <= 0:
            return True
        return min(depth_bid, depth_ask) < depth_floor

    def _mark_non_repaint_opened(self):
        """Reset non-repaint guards for the next bar."""
        for series in self.non_repaint_series:
            series.mark_bar_opened()

    def try_bar_update(self):
        LOG.debug(f"[DIAG] try_bar_update called. best_bid={self.best_bid}, best_ask={self.best_ask}")
        if self.best_bid is None or self.best_ask is None:
            LOG.debug("[BAR] Skipping bar update: best_bid or best_ask is None")
            return

        mid = (self.best_bid + self.best_ask) / 2.0
        LOG.debug(f"[DIAG] Calculated mid={mid}")
        self.current_bar_tick_count += 1
        self._update_vpin(mid)

        # Update MFE/MAE if we have an open position
        # FIX: trackers are keyed by "{symbol_id}_ticket_{ticket}", never by
        # "default".  Iterate over ALL active trackers so every open position
        # receives tick-by-tick MFE/MAE updates.
        with self._tracker_lock:
            has_active_position = False
            if self.trade_integration.trade_manager:
                position = self.trade_integration.trade_manager.get_position()
                has_active_position = abs(position.net_qty) > MIN_POSITION_QTY
            else:
                has_active_position = self.cur_pos != 0
            if has_active_position and self.mfe_mae_trackers:
                for tracker in self.mfe_mae_trackers.values():
                    tracker.update(mid)

        closed = self.builder.update(utc_now(), mid)
        if closed:
            LOG.debug(f"[BAR] Closed bar: {closed}")
            tick_count = self.current_bar_tick_count
            self.current_bar_tick_count = 0
            self._update_non_repaint_series(closed, tick_count)
            self._mark_non_repaint_closed()
            self.close_stats.update(closed[4])
            self.bars.append(closed)
            LOG.debug(f"[BAR] Appended to self.bars (len now {len(self.bars)})")
            self._save_bars_cache()
            self.on_bar_close(closed)
            self._mark_non_repaint_opened()

    # ── Order book live export ─────────────────────────────────────────────

    def _export_order_book(self, min_interval: float = 1.0) -> None:
        """Write data/order_book.json at most once per *min_interval* seconds.

        The HUD market tab reads this file so it always has the live L2 ladder
        without waiting for a bar close (which could be hours on H4).
        """
        now = time.time()
        if now - self._last_ob_export_time < min_interval:
            return
        try:
            spread = (
                float(self.best_ask) - float(self.best_bid)
                if self.best_bid and self.best_ask
                else 0.0
            )
            vpin_s = getattr(self, "last_vpin_stats", {"vpin": 0.0, "zscore": 0.0})
            ob = {
                "symbol": self.symbol,
                "spread": spread,
                "order_book_bids": [[p, s] for p, s in list(self.order_book.bids.items())[:5]],
                "order_book_asks": [[p, s] for p, s in list(self.order_book.asks.items())[:5]],
                "depth_bid": sum(s for _, s in self.order_book.bids.items()),
                "depth_ask": sum(s for _, s in self.order_book.asks.items()),
                "vpin": vpin_s.get("vpin", 0.0),
                "vpin_zscore": vpin_s.get("zscore", 0.0),
            }
            with open(self.hud_data_dir / "order_book.json", "w", encoding="utf-8") as fh:
                json.dump(ob, fh)

            # Also write a fresh current_position.json so the HUD overview always
            # shows the correct current_price, unrealized PnL, MFE and MAE
            # without waiting for the next bar close (which can be hours on H4).
            if self.cur_pos != 0 and self.best_bid and self.best_ask:
                live_price = (float(self.best_bid) + float(self.best_ask)) / 2.0
                with self._tracker_lock:
                    live_tracker = (
                        next(reversed(self.mfe_mae_trackers.values()), None)
                        if self.mfe_mae_trackers
                        else self.mfe_mae_tracker
                    )
                if live_tracker and live_tracker.entry_price:
                    ep = float(live_tracker.entry_price)
                    live_qty = self.qty
                    try:
                        if self.trade_integration and self.trade_integration.trade_manager:
                            _lp = self.trade_integration.trade_manager.get_position()
                            if _lp and abs(_lp.net_qty) > 0:
                                live_qty = float(abs(_lp.net_qty))
                    except Exception:
                        pass
                    live_pnl = (
                        (live_price - ep) * live_qty if self.cur_pos > 0
                        else (ep - live_price) * live_qty
                    )
                    with self._tracker_lock:
                        _prec = (
                            next(iter(self.path_recorders.values()), self.path_recorder)
                            if self.path_recorders
                            else self.path_recorder
                        )
                    live_bars_held = len(_prec.path) if hasattr(_prec, "path") else 0
                    pos_data = {
                        "direction": "LONG" if self.cur_pos > 0 else "SHORT",
                        "entry_price": ep,
                        "current_price": live_price,
                        "mfe": float(live_tracker.mfe) * live_qty,
                        "mae": float(live_tracker.mae) * live_qty,
                        "unrealized_pnl": live_pnl,
                        "bars_held": live_bars_held,
                    }
                    with open(self.hud_data_dir / "current_position.json", "w", encoding="utf-8") as fh:
                        json.dump(pos_data, fh)
            elif self.cur_pos == 0:
                # No open position — write a FLAT snapshot so the HUD clears
                # immediately after a close without waiting for bar-close flush.
                flat_data = {
                    "direction": "FLAT",
                    "entry_price": 0.0,
                    "current_price": 0.0,
                    "mfe": 0.0,
                    "mae": 0.0,
                    "unrealized_pnl": 0.0,
                    "bars_held": 0,
                }
                with open(self.hud_data_dir / "current_position.json", "w", encoding="utf-8") as fh:
                    json.dump(flat_data, fh)

            self._last_ob_export_time = now
        except Exception as e:
            LOG.debug("[OB] order_book.json export failed: %s", e)

    # ── Bar cache persistence ──────────────────────────────────────────────

    def _save_bars_cache(self, max_bars: int = 500) -> None:
        """Persist the last *max_bars* closed bars to data/bars_cache.json.

        Bars are stored as [iso_timestamp, o, h, l, c] rows so they can be
        reloaded after a restart without waiting for real bars to accumulate.
        """
        try:
            cache_path = self.hud_data_dir / "bars_cache.json"
            rows = [
                [b[0].isoformat(), b[1], b[2], b[3], b[4]]
                for b in list(self.bars)[-max_bars:]
            ]
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump({"timeframe_minutes": self.timeframe_minutes, "bars": rows}, fh)
        except Exception as e:
            LOG.warning("[BARS] Could not save bars cache: %s", e)

    def _load_bars_cache(self) -> None:
        """Re-seed self.bars from data/bars_cache.json on startup.

        Also replays bar returns into var_estimator, close_stats, and
        path_geometry so all derived signals are live from the first tick.
        """
        cache_path = self.hud_data_dir / "bars_cache.json"
        if not cache_path.exists():
            LOG.info("[BARS] No bars cache found — waiting for live bars")
            return
        try:
            with open(cache_path, encoding="utf-8") as fh:
                payload = json.load(fh)

            # Reject cache if it was written for a different timeframe
            cached_tf = payload.get("timeframe_minutes")
            if cached_tf is not None and int(cached_tf) != self.timeframe_minutes:
                LOG.warning(
                    "[BARS] Cache timeframe %sm ≠ current %sm — ignoring",
                    cached_tf, self.timeframe_minutes,
                )
                return

            rows = payload.get("bars", [])
            if not rows:
                return

            loaded = 0
            for row in rows:
                try:
                    ts = dt.datetime.fromisoformat(row[0])
                    bar = (ts, float(row[1]), float(row[2]), float(row[3]), float(row[4]))
                except (ValueError, IndexError, TypeError):
                    continue
                self.bars.append(bar)
                self.close_stats.update(bar[4])
                loaded += 1

            # Replay returns into VaR estimator
            bar_list = list(self.bars)
            for i in range(1, len(bar_list)):
                prev_c = bar_list[i - 1][4]
                curr_c = bar_list[i][4]
                if prev_c > 0:
                    self.var_estimator.update_return(SafeMath.safe_div(curr_c - prev_c, prev_c, 0.0))

            # Seed regime detector and path geometry
            if hasattr(self.policy, "seed_regime_from_bars") and len(self.bars) >= 10:
                self.policy.seed_regime_from_bars(self.bars)

            realized_vol = self._calculate_rs_volatility() if len(self.bars) >= MIN_BARS_FOR_VAR_UPDATE else 0.0
            if len(self.bars) >= 3 and realized_vol > 0:
                self.path_geometry.update(self.bars, realized_vol)

            LOG.info(
                "[BARS] ✓ Seeded %d bars from cache (tf=%sm, vol=%.6f)",
                loaded, self.timeframe_minutes, realized_vol,
            )
        except Exception as e:
            LOG.warning("[BARS] Failed to load bars cache: %s", e)

    # ── Harvester tick evaluation ─────────────────────────────────────────

    def _evaluate_harvester_on_tick(self):
        """
        Full Harvester evaluation on every tick for responsive exit decisions.
        Evaluates EACH individual position independently (not net position).
        Includes ML inference, time stops, and SL/TP checks.

        Hedging mode compatible: tracks which positions have pending close orders.
        """
        # DEBUG: Log entry to this function once per minute
        if not hasattr(self, "_last_harvester_debug_log"):
            self._last_harvester_debug_log = 0
        now = time.time()
        if now - self._last_harvester_debug_log > HARVESTER_DEBUG_INTERVAL:
            tracker_count = len(self.mfe_mae_trackers) if hasattr(self, "mfe_mae_trackers") else 0
            LOG.debug(
                "[HARVESTER_DEBUG] Eval tick: bid=%s ask=%s bars=%d trackers=%d",
                self.best_bid,
                self.best_ask,
                len(self.bars) if hasattr(self, "bars") else 0,
                tracker_count,
            )
            if tracker_count > 0:
                for pos_id, tracker in self.mfe_mae_trackers.items():
                    LOG.debug(
                        "[HARVESTER_DEBUG] Tracker %s: entry=%.2f dir=%d mfe=%.4f mae=%.4f",
                        pos_id,
                        getattr(tracker, "entry_price", 0),
                        getattr(tracker, "direction", 0),
                        getattr(tracker, "mfe", 0),
                        getattr(tracker, "mae", 0),
                    )
            self._last_harvester_debug_log = now

        # Need valid prices and bars
        if self.best_bid is None or self.best_ask is None:
            return
        if len(self.bars) == 0:
            return

        mid_price = (self.best_bid + self.best_ask) / 2.0

        # Evaluate ALL individual trackers (each position managed independently)
        if not hasattr(self, "mfe_mae_trackers") or len(self.mfe_mae_trackers) == 0:
            return

        # Track positions with pending close orders to prevent duplicates
        if not hasattr(self, "_pending_closes"):
            self._pending_closes = set()

        # Snapshot tracker items under lock to avoid dict-changed-during-iteration
        with self._tracker_lock:
            tracker_snapshot = list(self.mfe_mae_trackers.items())

        # Check each position independently
        for position_id, tracker in tracker_snapshot:
            # Skip if already closing this position
            if position_id in self._pending_closes:
                continue

            # Skip if tracker not initialized
            entry_price = getattr(tracker, "entry_price", None)
            if entry_price is None or entry_price <= 0:
                continue

            # Skip if no position direction set
            direction = getattr(tracker, "direction", None)
            if direction is None or direction == 0:
                continue

            # Update tracker with current price
            tracker.update(mid_price)

            # Full Harvester decision with ML for this position
            try:
                # Get current market features
                imbalance = self.order_book.imbalance() if hasattr(self.order_book, "imbalance") else 0.0
                vpin_zscore = self.last_vpin_stats.get("zscore", 0.0) if hasattr(self, "last_vpin_stats") else 0.0
                b_depth, a_depth = self.order_book.depth_sum() if hasattr(self.order_book, "depth_sum") else (0.0, 0.0)
                depth_ratio = b_depth / a_depth if a_depth > 0 else 1.0
                event_features = (
                    self.event_time_engine.calculate_features()
                    if hasattr(self, "event_time_engine") and self.event_time_engine
                    else {}
                )

                # Harvester decision (ML-based) for this specific position
                exit_action, exit_conf = self.policy.decide_exit(
                    self.bars,
                    current_price=mid_price,
                    imbalance=imbalance,
                    vpin_z=vpin_zscore,
                    depth_ratio=depth_ratio,
                    event_features=event_features,
                )
                # exit_action: 0=HOLD, 1=CLOSE

                if exit_action == 1:
                    # Mark as pending to prevent duplicate close orders
                    self._pending_closes.add(position_id)

                    LOG.info(
                        "[TICK_EXIT] Harvester CLOSE %s @ %.2f conf=%.2f | entry=%.2f MFE=%.4f MAE=%.4f dir=%d",
                        position_id,
                        mid_price,
                        exit_conf,
                        tracker.entry_price,
                        tracker.mfe,
                        tracker.mae,
                        tracker.direction,
                    )
                    # Close this specific position (not net)
                    if hasattr(self, "trade_integration") and hasattr(self.trade_integration, "close_position"):
                        # HEDGING MODE: Get broker ticket from tracker
                        ticket = getattr(tracker, "position_ticket", None)
                        if not ticket:
                            LOG.warning("[TICK_EXIT] No broker ticket for position %s, using legacy close", position_id)
                            # Legacy fallback: close by position_id
                            success = self.trade_integration.close_position(
                                position_id=position_id, reason="TICK_HARVESTER"
                            )
                        else:
                            # HEDGING MODE: Close by broker ticket (not position_id)
                            # This ensures correct position is closed even after crashes
                            success = self.trade_integration.close_position(
                                position_id=position_id, reason="TICK_HARVESTER"
                            )

                        if not success:
                            # Failed to submit, allow retry next tick
                            self._pending_closes.discard(position_id)

            except Exception as e:
                LOG.error("[TICK_EXIT] Error evaluating position %s: %s", position_id, e, exc_info=True)

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

    def _close_foreign_position(self, foreign_symbol_id: str, long_qty: float, short_qty: float, ticket: str | None):
        """Close a position that belongs to a non-target symbol at startup."""
        if not self.trade_sid:
            LOG.warning("[CLEANUP] Cannot close foreign position — no TRADE session")
            return

        # Close long side
        if long_qty > MIN_POSITION_THRESHOLD:
            clord_id = f"CLEANUP_SELL_{int(time.time() * 1000)}"
            msg = fix44.NewOrderSingle()
            msg.setField(fix.ClOrdID(clord_id))
            msg.setField(fix.Symbol(foreign_symbol_id))
            msg.setField(fix.Side("2"))  # SELL to close long
            msg.setField(fix.TransactTime())
            msg.setField(fix.OrdType("1"))  # Market
            msg.setField(fix.OrderQty(round(long_qty, 2)))
            if ticket:
                msg.setField(721, ticket)  # PosMaintRptID for hedging
            fix.Session.sendToTarget(msg, self.trade_sid)
            LOG.warning(
                "[CLEANUP] 🧹 Closing foreign LONG: symbol=%s qty=%.4f ticket=%s clOrdID=%s",
                foreign_symbol_id,
                long_qty,
                ticket,
                clord_id,
            )

        # Close short side
        if short_qty > MIN_POSITION_THRESHOLD:
            clord_id = f"CLEANUP_BUY_{int(time.time() * 1000)}"
            msg = fix44.NewOrderSingle()
            msg.setField(fix.ClOrdID(clord_id))
            msg.setField(fix.Symbol(foreign_symbol_id))
            msg.setField(fix.Side("1"))  # BUY to close short
            msg.setField(fix.TransactTime())
            msg.setField(fix.OrdType("1"))  # Market
            msg.setField(fix.OrderQty(round(short_qty, 2)))
            if ticket:
                msg.setField(721, ticket)  # PosMaintRptID for hedging
            fix.Session.sendToTarget(msg, self.trade_sid)
            LOG.warning(
                "[CLEANUP] 🧹 Closing foreign SHORT: symbol=%s qty=%.4f ticket=%s clOrdID=%s",
                foreign_symbol_id,
                short_qty,
                ticket,
                clord_id,
            )

    def on_position_report(self, msg: fix.Message):
        # Route to TradeManager first
        self.trade_integration.handle_position_report(msg)

        try:
            sym = fix.Symbol()
            if msg.isSetField(sym):
                msg.getField(sym)
                foreign_id = sym.getValue()

                # Defensive: Validate symbol_id is set
                if not hasattr(self, "symbol_id") or self.symbol_id is None:
                    LOG.warning("[CLEANUP] symbol_id not set, cannot check foreign positions")
                    return

                if foreign_id != str(self.symbol_id):
                    # --- AUTO-CLOSE foreign symbol positions at startup ---
                    f704 = fix.StringField(704)
                    f705 = fix.StringField(705)
                    f_long = 0.0
                    f_short = 0.0
                    f_ticket = None
                    try:
                        if msg.isSetField(f704):
                            msg.getField(f704)
                            f_long = float(f704.getValue())
                        if msg.isSetField(f705):
                            msg.getField(f705)
                            f_short = float(f705.getValue())
                        f721 = fix.StringField(721)
                        if msg.isSetField(f721):
                            msg.getField(f721)
                            f_ticket = f721.getValue()
                    except (ValueError, TypeError) as e:
                        LOG.warning("[CLEANUP] Error parsing foreign position quantities: %s", e)

                    # Defensive: Validate quantities are non-negative and reasonable
                    if f_long < 0 or f_short < 0:
                        LOG.error("[CLEANUP] Invalid negative quantities - long=%.4f short=%.4f", f_long, f_short)
                        return

                    if f_long > MAX_POSITION_SANITY or f_short > MAX_POSITION_SANITY:  # Sanity check
                        LOG.error("[CLEANUP] Suspicious large quantities - long=%.4f short=%.4f", f_long, f_short)
                        return

                    if f_long > MIN_POSITION_THRESHOLD or f_short > MIN_POSITION_THRESHOLD:
                        LOG.warning(
                            "[CLEANUP] Foreign position detected: symbol=%s long=%.4f short=%.4f — auto-closing",
                            foreign_id,
                            f_long,
                            f_short,
                        )
                        self._close_foreign_position(foreign_id, f_long, f_short, f_ticket)
                    return
        except Exception as e:
            LOG.error("[TRADE] Error parsing position report symbol: %s", e, exc_info=True)
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

        # FIX P0-2: Position is now tracked by TradeManager (single source of truth)
        # This code path kept for backward compatibility logging
        old_pos = self.cur_pos  # Gets from TradeManager via property

        LOG.info("[TRADE] PositionReport net=%0.6f -> cur_pos=%s (via TradeManager)", net, self.cur_pos)

        # GAP 10.1: Log position update
        avg_price = 0.0
        # Look up the active tracker from the multi-position dict (fall back to legacy singleton)
        _active_tracker = None
        with self._tracker_lock:
            if self.mfe_mae_trackers:
                # Use the most recent tracker (last entry)
                _active_tracker = next(reversed(self.mfe_mae_trackers.values()), None)
        if _active_tracker is None:
            _active_tracker = self.mfe_mae_tracker  # Legacy fallback
        if _active_tracker and _active_tracker.entry_price:
            avg_price = _active_tracker.entry_price

        self.transaction_log.log_position_update(position_id=f"{self.symbol_id}_net", net_qty=net, avg_price=avg_price)

        # Log MFE/MAE summary and save path when position closes
        if old_pos != 0 and self.cur_pos == 0:
            summary = (
                _active_tracker.get_summary()
                if _active_tracker and _active_tracker.entry_price
                else self.mfe_mae_tracker.get_summary()
            )
            LOG.info(
                "[MFE/MAE] Entry=%.5f %s | MFE=%.5f MAE=%.5f | Best=%.5f Worst=%.5f | WTL=%s",
                summary["entry_price"],
                summary["direction"],
                summary["mfe"],
                summary["mae"],
                summary["best_profit"],
                summary["worst_loss"],
                summary["winner_to_loser"],
            )

            # Stop path recording and save trade
            if self.best_bid and self.best_ask:
                exit_price = (float(self.best_bid) + float(self.best_ask)) / 2.0
                self._process_trade_completion(summary, exit_price)

            # GAP 7.1 FIX: Complete state reset on position close
            self.mfe_mae_tracker.reset()
            self.trade_entry_time = None
            self.entry_state = None
            self.entry_action = None
            self.prev_harvester_state = None
            self.prev_exit_action = None
            self.prev_mfe = 0.0
            self.prev_mae = 0.0
            LOG.debug("[CLEANUP] All position state reset after close")

    def _calculate_position_pnl(
        self,
        entry_price: float,
        exit_price: float,
        direction: str,
        quantity: float = None,
        contract_size: float = None,
    ) -> float:
        """
        Calculate position P&L (single source of truth).

        Formula: (exit - entry) * direction_sign * quantity * contract_size

        Args:
            entry_price: Entry execution price
            exit_price: Exit execution price
            direction: "LONG" or "SHORT"
            quantity: Position size in lots (default: self.qty)
            contract_size: Contract size (default: self.contract_size)

        Returns:
            P&L in USD

        Examples:
            LONG: (4879.75 - 4878.96) * 1 * 0.1 * 100.0 = +7.90
            SHORT: (4878.96 - 4879.75) * -1 * 0.1 * 100.0 = +7.90
        """
        qty = quantity if quantity is not None else self.qty
        contract = contract_size if contract_size is not None else self.contract_size
        direction_sign = 1 if direction == "LONG" else -1

        pnl = (exit_price - entry_price) * direction_sign * qty * contract

        LOG.debug(
            "[PNL_CALC] %s: (%.2f - %.2f) * %d * %.4f * %.2f = %.4f",
            direction,
            exit_price,
            entry_price,
            direction_sign,
            qty,
            contract,
            pnl,
        )

        return pnl

    def _process_trade_completion(self, summary: dict, exit_price: float):
        """Process a completed trade round-trip for experience collection and performance tracking.

        Called from:
        1. on_position_report — when FIX PositionReport arrives (live trading)
        2. TradeManagerIntegration — after closing a position (paper trading)

        Args:
            summary: MFE/MAE tracker summary dict with entry_price, direction, mfe, mae, etc.
            exit_price: The exit fill price
        """
        # Function entry logging
        LOG.debug(
            "[TRADE_COMPLETION] Entry: direction=%s entry=%.2f exit=%.2f mfe=%.4f mae=%.4f",
            summary.get("direction", "?"),
            summary.get("entry_price", 0.0),
            exit_price,
            summary.get("mfe", 0.0),
            summary.get("mae", 0.0),
        )

        if not summary or exit_price <= 0:
            LOG.warning("[TRADE_COMPLETION] Skipped: invalid summary or exit_price=%.5f", exit_price)
            return

        try:
            exit_time = utc_now()
            direction = summary.get("direction", "UNKNOWN")
            entry_price = summary.get("entry_price", 0.0)
            if entry_price <= 0:
                LOG.warning("[TRADE_COMPLETION] Skipped: entry_price=%.5f", entry_price)
                return

            # ── Trade timing tracking (single source of truth for HUD) ──────
            # Update duration EMA and record close timestamp so the HUD can
            # show "avg_trade_duration" and "last_trade_mins_ago" accurately.
            if self.trade_entry_time:
                _dur_mins = (exit_time - self.trade_entry_time).total_seconds() / 60.0
                self._avg_trade_duration_mins = (
                    0.9 * self._avg_trade_duration_mins + 0.1 * _dur_mins
                )
                LOG.debug(
                    "[TRADE_TIMING] dur=%.1f min → avg=%.1f min",
                    _dur_mins,
                    self._avg_trade_duration_mins,
                )
            self._last_trade_close_ts = time.time()
            # ─────────────────────────────────────────────────────────────────

            # Calculate P&L using dedicated method (single source of truth)
            pnl = self._calculate_position_pnl(entry_price, exit_price, direction)

            # Checkpoint: Store initial P&L to detect corruption
            _pnl_checkpoint = pnl
            LOG.debug("[PNL_CHECKPOINT] Initial P&L calculated: %.4f", _pnl_checkpoint)

            # Log position close
            self.transaction_log.log_position_close(
                position_id=f"{self.symbol_id}_net",
                pnl=pnl,
                mfe=summary.get("mfe", 0.0),
                mae=summary.get("mae", 0.0),
            )

            # Add to performance tracker
            if self.trade_entry_time:
                self.performance.add_trade(
                    pnl=pnl,
                    entry_time=self.trade_entry_time,
                    exit_time=exit_time,
                    direction=summary.get("direction", "UNKNOWN"),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    mfe=summary.get("mfe", 0.0),
                    mae=summary.get("mae", 0.0),
                    winner_to_loser=summary.get("winner_to_loser", False),
                )

            # --- Set prev_harvester_state for experience addition ---
            if hasattr(self.policy, "harvester") and hasattr(self.policy.harvester, "last_state"):
                self.prev_harvester_state = (
                    self.policy.harvester.last_state.copy() if self.policy.harvester.last_state is not None else None
                )
                self.prev_exit_action = 1  # CLOSE
            if self.entry_action is None:
                self.entry_action = 1 if summary.get("direction") == "LONG" else 2

            # Normalise PnL to price-point units so dimensions match MFE/MAE.
            # PnL is in $ (quantity × contract_size × price_move); MFE/MAE are in raw
            # price points. Dividing by the lot-value factor makes them comparable:
            #   pnl_pts = price_move  (e.g. -20.57 pts for a 20-pt adverse move)
            lot_value = max(self.qty * self.contract_size, 1.0)
            pnl_pts = pnl / lot_value

            # Calculate shaped rewards for DDQN training
            reward_data = {
                "exit_pnl": pnl_pts,   # price-point units — same scale as mfe/mae
                "mfe": summary.get("mfe", 0.0),
                "mae": summary.get("mae", 0.0),
                "winner_to_loser": summary.get("winner_to_loser", False),
            }
            shaped_rewards = self.reward_shaper.calculate_total_reward(reward_data)

            if summary.get("mfe", 0.0) > 0 and hasattr(self.reward_shaper, "update_baseline_mfe"):
                self.reward_shaper.update_baseline_mfe(summary["mfe"])

            LOG.info(
                "[REWARD] Capture: %+.4f | WTL Penalty: %+.4f | Opportunity: %+.4f | Total: %+.4f | Active: %d",
                shaped_rewards["capture_efficiency"],
                shaped_rewards["wtl_penalty"],
                shaped_rewards["opportunity_cost"],
                shaped_rewards["total_reward"],
                shaped_rewards["components_active"],
            )

            # === EXPERIENCE ADDITION FOR ONLINE LEARNING ===
            # TriggerAgent experience
            if hasattr(self.policy, "add_trigger_experience") and self.entry_state is not None:
                next_state = None
                if hasattr(self.policy, "trigger") and hasattr(self.policy.trigger, "last_state"):
                    next_state = self.policy.trigger.last_state

                if next_state is not None:
                    realized_vol = self._calculate_rs_volatility() if len(self.bars) >= MIN_BARS_FOR_VOL_CALC else 0.01
                    # FIX 7: Use neutral reward for exploration entries to avoid
                    # training on meaningless prediction-accuracy signals
                    was_explore = getattr(self, "was_exploration_entry", False)
                    if was_explore:
                        # Still add the experience (state→outcome), but with a
                        # simple outcome-based reward instead of prediction-accuracy.
                        # Normalise pnl to σ-of-price-movement for instrument-agnostic scale:
                        #   pnl_pts = pnl / lot_value            (USD → raw price-point move)
                        #   vol_pts = realized_vol × entry_price (fractional σ → price-pt σ)
                        #   reward  = pnl_pts / vol_pts / 3      (σ-normalised, clipped ±0.5)
                        pnl_for_reward = pnl  # Use already-calculated PnL (don't overwrite it!)
                        _ep_lot = max(self.qty * self.contract_size, 1.0)
                        _ep_price = max(float(summary.get("entry_price", 0.0) or entry_price), 1.0)
                        _ep_vol_pts = max(realized_vol * _ep_price, 1e-6)
                        trigger_reward = float(np.clip(
                            (pnl_for_reward / _ep_lot) / _ep_vol_pts / 3.0, -0.5, 0.5
                        ))
                        LOG.info(
                            "[ONLINE_LEARNING] Exploration entry: using outcome reward=%.4f "
                            "(skipping prediction-accuracy reward)",
                            trigger_reward,
                        )
                    else:
                        trigger_reward = self._calculate_trigger_reward(
                            trade_summary=summary,
                            predicted_runway=self.predicted_runway,
                            realized_vol=realized_vol,
                        )
                    self.policy.add_trigger_experience(
                        state=self.entry_state,
                        action=self.entry_action,
                        reward=trigger_reward,
                        next_state=next_state,
                        done=True,
                    )
                    if hasattr(self.policy.trigger, "buffer") and self.policy.trigger.buffer:
                        LOG.info("[BUFFER] TriggerAgent buffer size: %d", self.policy.trigger.buffer.size)
                    LOG.info(
                        "[ONLINE_LEARNING] Added TriggerAgent experience: action=%d reward=%.4f "
                        "(predicted_mfe=%.4f actual_mfe=%.4f explore=%s)",
                        self.entry_action,
                        trigger_reward,
                        self.predicted_runway,
                        summary.get("mfe", 0.0),
                        was_explore,
                    )

                # ── Prediction-vs-actual convergence tracking ──────────────────
                # Update EMAs regardless of exploration flag (outcomes are real).
                _ep_price = max(float(summary.get("entry_price", 0.0) or entry_price), 1.0)
                _pred_pts = self.predicted_runway * _ep_price  # fractional → price-pts
                _actual_mfe = float(summary.get("mfe", 0.0))
                _runway_delta = _pred_pts - _actual_mfe
                # Accuracy: 1 - normalised abs error (same formula as _calculate_trigger_reward)
                _max_err = max(abs(_actual_mfe), abs(_pred_pts), 1.0)
                _runway_acc = 1.0 - min(abs(_runway_delta) / _max_err, 1.0)
                _actual_win = 1.0 if pnl > 0 else 0.0
                _conf_err = abs(self.entry_confidence - _actual_win)
                _alpha = 0.1
                self._runway_delta_ema = (1 - _alpha) * self._runway_delta_ema + _alpha * _runway_delta
                self._runway_accuracy_ema = (1 - _alpha) * self._runway_accuracy_ema + _alpha * _runway_acc
                self._conf_calib_err_ema = (1 - _alpha) * self._conf_calib_err_ema + _alpha * _conf_err
                # ─────────────────────────────────────────────────────────────────

                self.entry_state = None
                self.predicted_runway = 0.0

            # HarvesterAgent experience (CLOSE decision)
            if hasattr(self.policy, "add_harvester_experience") and self.prev_harvester_state is not None:
                # Use total shaped reward (already normalised by baseline_mfe in reward_shaper).
                # Clamp to [-2, 2] for DDQN stability.
                raw_close_reward = shaped_rewards.get("total_reward", 0.0)

                # Regime conditioning: extra penalty when the entry was made in the
                # face of elevated risk signals that the agent should learn to avoid.
                _entry_var   = getattr(self, "entry_var", 0.0)
                _entry_vpin  = getattr(self, "entry_vpin_z", 0.0)
                regime_adj   = 0.0
                if _entry_var > self.vol_cap:
                    # Penalise entries in high-VaR regimes proportional to how far over the cap
                    regime_adj -= 0.3 * min(_entry_var / (self.vol_cap + 1e-8), 2.0)
                if self.vpin_z_threshold > 0 and _entry_vpin > self.vpin_z_threshold:
                    # Penalise entries in turbulent order-flow regimes
                    regime_adj -= 0.2 * min(_entry_vpin / (self.vpin_z_threshold + 1e-8), 2.0)
                LOG.debug(
                    "[CLOSE_REWARD] raw=%.4f regime_adj=%.4f (entry_var=%.4f entry_vpin=%.2f)",
                    raw_close_reward, regime_adj, _entry_var, _entry_vpin,
                )
                capture_reward = float(np.clip(raw_close_reward + regime_adj, -2.0, 2.0))

                next_state = (
                    self.policy.harvester.last_state
                    if hasattr(self.policy, "harvester") and hasattr(self.policy.harvester, "last_state")
                    else None
                )
                if next_state is not None:
                    self.policy.add_harvester_experience(
                        state=self.prev_harvester_state,
                        action=1,  # CLOSE
                        reward=capture_reward,
                        next_state=next_state,
                        done=True,
                    )
                    if hasattr(self.policy.harvester, "buffer") and self.policy.harvester.buffer:
                        LOG.info("[BUFFER] HarvesterAgent buffer size: %d", self.policy.harvester.buffer.size)
                    LOG.info("[ONLINE_LEARNING] Added HarvesterAgent CLOSE: reward=%.4f", capture_reward)

                self.prev_harvester_state = None
                self.prev_exit_action = None
                self.prev_mfe = 0.0
                self.prev_mae = 0.0

            # Update circuit breakers
            current_equity = self.performance.total_pnl + 10000.0
            self.circuit_breakers.update_trade(pnl, current_equity)
            breaker_status = self.circuit_breakers.get_status()
            if breaker_status["any_tripped"]:
                LOG.warning("[CIRCUIT-BREAKER] Status after trade: %s", breaker_status)
            self.circuit_breakers.reset_if_cooldown_elapsed()
            try:
                self.circuit_breakers.save_state()
            except Exception as cb_save_err:
                LOG.warning("[CIRCUIT-BREAKER] Failed to save state: %s", cb_save_err)

            self.entry_action = None

            # Verify P&L hasn't been corrupted during processing
            if abs(pnl - _pnl_checkpoint) > MIN_POSITION_THRESHOLD:
                LOG.error(
                    "[BUG_DETECTION] P&L changed during processing! Initial=%.4f Current=%.4f (diff=%.4f)",
                    _pnl_checkpoint,
                    pnl,
                    pnl - _pnl_checkpoint,
                )
                # Restore original value
                pnl = _pnl_checkpoint
                LOG.warning("[BUG_DETECTION] Restored P&L to checkpoint value: %.4f", pnl)

            # Save trade record
            trade_record = {
                "trade_id": self.performance.total_trades if hasattr(self, "performance") else int(time.time()),
                "symbol": self.symbol,
                "direction": summary.get("direction", "UNKNOWN"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_time": self.trade_entry_time.isoformat() if self.trade_entry_time else None,
                "exit_time": exit_time.isoformat(),
                "pnl": pnl,
                "mfe": summary.get("mfe", 0.0),
                "mae": summary.get("mae", 0.0),
                "winner_to_loser": summary.get("winner_to_loser", False),
            }
            LOG.debug("[TRADE_RECORD] Prepared for save: trade_id=%s pnl=%.4f", trade_record["trade_id"], pnl)
            try:
                self._atomic_save_trade(trade_record)
            except Exception as save_err:
                LOG.error("[SAVE] Trade save failed: %s", save_err, exc_info=True)

            # Performance dashboard
            if self.trade_entry_time:
                metrics = self.performance.get_metrics()
                current_sharpe = metrics.get("sharpe_ratio", 0.0)
                sharpe_delta = current_sharpe - self.previous_sharpe

                if (
                    self.performance.total_trades % PERFORMANCE_INTERVAL_TRADES == 0
                    and self.performance.total_trades > MIN_TRADES_FOR_ADAPTATION
                ):
                    self.reward_shaper.adapt_weights(sharpe_delta)
                    LOG.info("[REWARD] Adapted weights based on Sharpe delta: %+.4f", sharpe_delta)
                    LOG.info(self.reward_shaper.print_summary())

                self.previous_sharpe = current_sharpe

                if self.performance.total_trades % PERFORMANCE_INTERVAL_TRADES == 0:
                    LOG.info("\n" + self.performance.print_dashboard())
                else:
                    LOG.info(
                        "[PERF] Trades: %d | Win Rate: %.1f%% | Total PnL: $%.2f | Sharpe: %.3f | Max DD: %.1f%%",
                        metrics["total_trades"],
                        metrics.get("win_rate", 0.0) * 100,
                        metrics.get("total_pnl", 0.0),
                        metrics.get("sharpe_ratio", 0.0),
                        metrics.get("max_drawdown", 0.0) * 100,
                    )

            LOG.info(
                "[TRADE_COMPLETION] ✓ Processed: %s entry=%.2f exit=%.2f pnl=%.4f mfe=%.4f",
                summary.get("direction", "?"),
                entry_price,
                exit_price,
                pnl,
                summary.get("mfe", 0.0),
            )

            # Flush live metrics after every trade
            self._flush_production_metrics()

            # Function exit logging
            LOG.debug(
                "[TRADE_COMPLETION] Exit: trade_id=%s pnl=%.4f recorded=%s",
                trade_record.get("trade_id", "?"),
                pnl,
                "success",
            )

        except Exception as e:
            LOG.error("[TRADE_COMPLETION] Error: %s", e, exc_info=True)
            LOG.debug("[TRADE_COMPLETION] Exit: recorded=failed")

    def on_exec_report(self, msg: fix.Message):
        # Route to TradeManager first (callbacks will handle state updates)
        self.trade_integration.handle_execution_report(msg)

        ex = fix.ExecType()
        if not msg.isSetField(ex):
            return
        msg.getField(ex)

        # FIX P0-4: Clear entry state on rejection to prevent corrupted experiences
        if ex.getValue() == "8":
            txt = fix.Text()
            clid = fix.ClOrdID()
            reason = ""
            order_id = "UNKNOWN"

            if msg.isSetField(txt):
                msg.getField(txt)
                reason = txt.getValue()
                LOG.warning("[TRADE] Order rejected: %s", reason)

            if msg.isSetField(clid):
                msg.getField(clid)
                order_id = clid.getValue()

            # GAP 10.1: Log order rejection
            self.transaction_log.log_order_reject(order_id=order_id, reason=reason)

            # Clear stale state that shouldn't be used for learning
            self.entry_state = None
            self.entry_action = None
            self.trade_entry_time = None
            LOG.debug("[TRADE] Cleared entry state after rejection")
            return

        # TradeManager handles fill processing via callbacks
        # No need for explicit position request - callback updates state
        if ex.getValue() != "F":
            return

        sym = fix.Symbol()
        if msg.isSetField(sym):
            msg.getField(sym)
            if sym.getValue() != str(self.symbol_id):
                return

        # Position will be updated via on_order_filled callback

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
    # ----------------------------
    # HarvesterAgent HOLD Reward Calculation (FIX 2)
    # ----------------------------
    def _calculate_harvester_hold_reward(
        self,
        excursions: MfeMaeSnapshot,
        position: HarvesterPositionState,
    ) -> float:
        """Calculate reward for HarvesterAgent HOLD action based on capture potential.

        Principles:
        - Reward staying in when capturing more MFE
        - Penalty for accumulating MAE
        - Opportunity cost for holding past MFE peak
        - Time decay (don't hold forever)

        Args:
            excursions: Current and previous MFE/MAE readings.
            position: Live position state (bars_held, unrealized_pnl, prices, vol).

        Returns:
            Hold reward in range [-1.0, 1.0]
        """
        current_mfe = excursions.current_mfe
        current_mae = excursions.current_mae
        prev_mfe = excursions.prev_mfe
        prev_mae = excursions.prev_mae
        bars_held = position.bars_held
        unrealized_pnl = position.unrealized_pnl
        realized_vol = position.realized_vol if position.realized_vol > 0 else 0.01

        # Current capture ratio: how much of MFE are we capturing?
        capture_ratio = unrealized_pnl / (current_mfe + 1e-8) if current_mfe > 0 else 0.0

        # Component 1: Capture quality reward [0, 0.4]
        # Higher is better - capturing more of the MFE
        capture_component = np.clip(capture_ratio * 0.4, 0.0, 0.4)

        # Component 2: MFE growth reward [-0.3, 0.3]
        # Reward growing profit potential.
        # Convert price-point deltas to fractional return units so they are on the
        # same scale as realized_vol (which is a per-bar fractional-return std-dev).
        ref_price = max(position.entry_price, 1.0)
        mfe_delta = current_mfe - prev_mfe
        mfe_delta_frac = mfe_delta / ref_price          # price-pt → fractional return
        norm_mfe_delta = mfe_delta_frac / (realized_vol + 1e-8)   # in σ units
        mfe_growth = np.clip(norm_mfe_delta * 0.3, -0.3, 0.3)

        # Component 3: MAE penalty [-0.4, 0]
        # Penalize adverse moves
        mae_delta = current_mae - prev_mae
        mae_delta_frac = mae_delta / ref_price
        norm_mae_delta = mae_delta_frac / (realized_vol + 1e-8)   # in σ units
        mae_penalty = -np.clip(norm_mae_delta * 0.4, 0.0, 0.4)

        # Component 4: Time decay [-0.2, 0]
        # Small penalty for holding too long (encourages taking profits).
        # Scaled to the timeframe so max decay triggers after ~1 trading day
        # regardless of bar size: M5→288 bars, H1→24, H4→6, D1→1.
        _target_hold_bars = max(10, self._bars_per_day)
        time_decay = -0.02 * min(bars_held / max(1, _target_hold_bars / 10), 10)

        # Component 5: Opportunity cost penalty [-0.3, 0]
        # Penalty for holding past MFE peak (missed the best exit).
        # Clip distance_from_peak to [0, 2] so a large draw-down doesn't
        # dominate the total (the stop-loss logic exits well before that).
        opportunity_cost = 0.0
        if current_mfe > 0:
            distance_from_peak = np.clip(
                (current_mfe - unrealized_pnl) / (current_mfe + 1e-8),
                0.0, 2.0,
            )
            opportunity_cost = -distance_from_peak * 0.3  # Max -0.6, softened by later clip

        # Combine components
        total_reward = capture_component + mfe_growth + mae_penalty + time_decay + opportunity_cost

        # Clip to reasonable range
        total_reward = np.clip(total_reward, -1.0, 1.0)

        LOG.debug(
            "[HARVESTER_HOLD_REWARD] Capture: %.3f | MFE: %.3f | MAE: %.3f | Time: %.3f | OppCost: %.3f | Total: %.3f",
            capture_component,
            mfe_growth,
            mae_penalty,
            time_decay,
            opportunity_cost,
            total_reward,
        )

        return total_reward

    # ----------------------------
    # TriggerAgent Reward Calculation (FIX 1)
    # ----------------------------
    def _calculate_trigger_reward(
        self, trade_summary: dict, predicted_runway: float, realized_vol: float = 0.01
    ) -> float:
        """
        Calculate reward for TriggerAgent based on prediction accuracy.

        TriggerAgent should be rewarded for:
        1. Accurately predicting MFE (runway)
        2. Identifying larger opportunities

        NOT for exit quality (that's HarvesterAgent's job).

        Args:
            trade_summary: Trade summary with 'mfe', 'mae', 'pnl', etc.
            predicted_runway: Predicted MFE at entry
            realized_vol: Realized volatility for normalization

        Returns:
            Trigger reward in range [-1.5, 1.5]
        """
        actual_mfe = trade_summary.get("mfe", 0.0)
        pnl = trade_summary.get("pnl", 0.0)

        # Default fallback
        if realized_vol <= 0:
            realized_vol = 0.01

        # ----------------------------------------------------------------
        # Instrument-agnostic dimensional alignment
        # -  actual_mfe is in price-point units (absolute, e.g. 3.42 pts)
        # -  predicted_runway is in fractional-price units (e.g. 0.0015)
        # -  realized_vol is per-bar fractional-return std-dev
        # -  pnl is in USD  ($)
        # We normalise everything to σ-of-price-movement (σ_pts) so the
        # reward signal is instrument-agnostic.
        # ----------------------------------------------------------------
        entry_price_val = max(float(trade_summary.get("entry_price", 0.0) or 1.0), 1.0)
        lot_value = max(self.qty * self.contract_size, 1.0)
        # vol in price-point units: RS-std × price level  (e.g. 0.0005 × 5000 = 2.5 pts/bar σ)
        vol_pts = max(realized_vol * entry_price_val, 1e-6)
        # Convert predicted_runway (fractional) → price-point units
        predicted_runway_pts = predicted_runway * entry_price_val

        # Normalize MFE and predicted runway to σ-of-price-movement
        norm_mfe = actual_mfe / vol_pts
        norm_predicted = predicted_runway_pts / vol_pts

        # Component 1: Prediction accuracy
        # How close was predicted MFE to actual MFE?
        prediction_error = abs(norm_mfe - norm_predicted)
        max_error = max(norm_mfe, norm_predicted, 1.0)  # Normalize error
        runway_accuracy = 1.0 - (prediction_error / max_error)

        # Map to [-1, 1]: Perfect prediction = 1.0, worst = -1.0
        accuracy_reward = runway_accuracy * 2.0 - 1.0

        # Component 2: Magnitude bonus
        # Prefer identifying larger opportunities (but cap to prevent outliers)
        magnitude_bonus = min(norm_mfe / 3.0, 1.0) * 0.5  # Max +0.5 for 3σ+ moves

        # Component 3: Penalty for false positives
        # If trade resulted in loss despite positive prediction, penalize proportionally.
        # Use price-point pnl so the severity is instrument-agnostic.
        false_positive_penalty = 0.0
        if predicted_runway > 0 and pnl < 0:
            pnl_pts = pnl / lot_value                             # $ → price-point units
            loss_severity = min(abs(pnl_pts) / vol_pts / 3.0, 1.0)
            false_positive_penalty = -0.2 - 0.5 * loss_severity  # -0.2 to -0.7

        # Component 4: Toxic flow entry penalty
        # Entering when VPIN z-score is elevated means trading against informed
        # order flow.  Penalise the trigger so it learns to wait for cleaner
        # conditions.  Linear from 0.75× threshold up to -0.30 at 2× threshold.
        entry_vpin_z = abs(getattr(self, "entry_vpin_z", 0.0))
        _vpin_thresh = float(getattr(self, "vpin_z_threshold", 2.5))
        toxic_penalty = 0.0
        if entry_vpin_z > _vpin_thresh * 0.75:
            _excess = (entry_vpin_z - _vpin_thresh * 0.75) / max(_vpin_thresh, 1.0)
            toxic_penalty = -0.15 * min(_excess, 2.0)  # max -0.30

        # Combine components
        total_reward = accuracy_reward + magnitude_bonus + false_positive_penalty + toxic_penalty

        # Clip to reasonable range
        total_reward = np.clip(total_reward, -1.5, 1.5)

        LOG.debug(
            "[TRIGGER_REWARD] Accuracy: %.3f | Magnitude: %.3f | FP Penalty: %.3f | Toxic: %.3f | Total: %.3f",
            accuracy_reward,
            magnitude_bonus,
            false_positive_penalty,
            toxic_penalty,
            total_reward,
        )

        return total_reward

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

        if len(self.bars) < window:
            return 0.005  # Default volatility

        rs_sum = 0.0
        valid_bars = 0

        # Use last 'window' bars
        for bar in list(self.bars)[-window:]:
            _, o, h, low_price, c = bar

            # Defensive: all prices must be positive
            if o <= 0 or h <= 0 or low_price <= 0 or c <= 0:
                continue

            # RS formula with safe math operations
            try:
                log_hc = SafeMath.safe_log(SafeMath.safe_div(h, c, 1.0))
                log_ho = SafeMath.safe_log(SafeMath.safe_div(h, o, 1.0))
                log_lc = SafeMath.safe_log(SafeMath.safe_div(low_price, c, 1.0))
                log_lo = SafeMath.safe_log(SafeMath.safe_div(low_price, o, 1.0))

                rs_bar = log_hc * log_ho + log_lc * log_lo

                if SafeMath.is_valid(rs_bar):
                    rs_sum += rs_bar
                    valid_bars += 1
            except (ValueError, ZeroDivisionError):
                continue

        if valid_bars < MIN_VALID_BARS_FOR_VOL:
            return DEFAULT_VOLATILITY  # Default if insufficient valid bars

        # Average RS variance per bar
        rs_variance = rs_sum / valid_bars

        # RS variance is already in (return)^2 units per bar
        # Take sqrt to get volatility per bar
        vol_per_bar = SafeMath.safe_sqrt(rs_variance) if rs_variance > 0 else 0.005

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
        t, o, h, low_price, c = bar

        LOG.debug(f"[BAR] on_bar_close called: t={t}, o={o}, h={h}, l={low_price}, c={c}")

        # Initialize decision variables for logging (will be populated later)
        action = None
        confidence = None
        runway = None
        feas = None
        exit_action = None
        exit_conf = None
        desired = None
        imbalance = None
        depth_bid = None
        depth_ask = None
        depth_ratio = None

        # Increment bar counter for HUD
        self.bar_count += 1

        if self.bar_count % 10 == 0 and self.close_stats.is_ready():
            LOG.info(
                "[ROLLING] period=%d mean=%.2f std=%.4f min=%.2f max=%.2f",
                self.close_stats.period,
                self.close_stats.mean,
                self.close_stats.std,
                self.close_stats.min,
                self.close_stats.max,
            )

        # PERIODIC AUTO-SAVE: Save every 50 bars to prevent data loss
        if self.bar_count - self.last_autosave_bar >= AUTOSAVE_INTERVAL_BARS and self.performance.total_trades > 0:
            try:
                datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
                files = self.trade_exporter.export_all(self.performance, prefix=f"autosave_b{self.bar_count}")
                self.last_autosave_bar = self.bar_count
                LOG.info(
                    "[AUTOSAVE] ✓ Bar %d: Saved %d trades to: %s",
                    self.bar_count,
                    self.performance.total_trades,
                    ", ".join(files.values()),
                )
            except Exception as e:
                LOG.error("[AUTOSAVE] ✗ Failed at bar %d: %s", self.bar_count, e)

        # Phase 3.5: Update activity monitor
        self.activity_monitor.on_bar_close()

        # P0 FIX: Check for position reconciliation timeouts every bar
        if self.trade_integration and self.trade_integration.trade_manager:
            self.trade_integration.trade_manager.check_all_position_request_timeouts()

        # Phase 3.5: Update VaR with bar return
        if len(self.bars) >= MIN_BARS_FOR_VAR_UPDATE:
            prev_close = self.bars[-2][4] if len(self.bars) >= MIN_BARS_FOR_PREV_CLOSE else c
            bar_return = SafeMath.safe_div(c - prev_close, prev_close, 0.0)
            self.var_estimator.update_return(bar_return)

        # Record bar if position is open
        # Support multiple positions via TradeManager
        if self.trade_integration.trade_manager:
            position = self.trade_integration.trade_manager.get_position()
            if position and abs(position.net_qty) > MIN_POSITION_QTY:  # Active position
                position_id = position.position_id if hasattr(position, "position_id") else self.default_position_id
                if position_id in self.path_recorders:
                    self.path_recorders[position_id].add_bar(bar)
        elif self.cur_pos != 0:  # Fallback for legacy single-position mode
            if self.default_position_id in self.path_recorders:
                self.path_recorders[self.default_position_id].add_bar(bar)

        # Calculate depth-aware imbalance metrics
        ob_depth = max(1, getattr(self.order_book, "depth", 1))
        learned_levels = getattr(self.friction_calculator, "depth_levels", ob_depth)
        depth_levels = max(1, min(ob_depth, learned_levels))
        depth_bid, depth_ask = self.order_book.depth_sum(levels=depth_levels)
        depth_total = depth_bid + depth_ask
        depth_ratio = 1.0
        if depth_ask > 0:
            depth_ratio = SafeMath.safe_div(depth_bid, depth_ask, 1.0)
        imbalance = 0.0
        if depth_total > 0:
            imbalance = SafeMath.safe_div(depth_bid - depth_ask, depth_total, 0.0)
        elif self.best_bid and self.best_ask:
            mid = (self.best_bid + self.best_ask) / 2.0
            spread = self.best_ask - self.best_bid
            if mid > 0:
                imbalance = (self.best_bid - mid) / (spread + 1e-10)
                imbalance = max(-1.0, min(1.0, imbalance))
        self.last_depth_metrics = {
            "bid": depth_bid,
            "ask": depth_ask,
            "ratio": depth_ratio,
            "levels": depth_levels,
        }
        self.last_depth_floor = getattr(self.friction_calculator, "depth_buffer", 0.0)
        self.last_depth_gate = False

        # Phase 3.5: Get exploration bonus from activity monitor
        exploration_bonus = (
            self.activity_monitor.get_exploration_bonus()
            if hasattr(self.activity_monitor, "get_exploration_bonus")
            else 0.0
        )

        # Phase 3.5: Calculate Rogers-Satchell volatility for PathGeometry
        realized_vol = self._calculate_rs_volatility()

        # Debug: Log RS volatility every bar
        if len(self.bars) >= MIN_TRADE_HISTORY_EXPLORATION:
            LOG.debug("[RS_VOL] bars=%d realized_vol=%.6f", len(self.bars), realized_vol)

        # Phase 3: Event-relative time features every bar for both agents
        event_features = {}
        is_high_liq = False
        if hasattr(self, "event_time_engine") and self.event_time_engine:
            event_features = self.event_time_engine.calculate_features()
            is_high_liq = self.event_time_engine.is_high_liquidity_period()
        vpin_zscore = self.last_vpin_stats.get("zscore", 0.0) if hasattr(self, "last_vpin_stats") else 0.0

        # Phase 3.5: Periodic training step
        self.bars_since_training += 1
        if self.bars_since_training >= self.training_interval and hasattr(self.policy, "train_step"):
            # FIX 4: Check minimum experience threshold before training
            MIN_EXPERIENCES_FOR_TRAINING = 32  # At least one batch worth

            # Check buffer sizes
            trigger_size = 0
            harvester_size = 0

            if hasattr(self.policy, "trigger") and hasattr(self.policy.trigger, "buffer"):
                trigger_buffer = self.policy.trigger.buffer
                if hasattr(trigger_buffer, "tree") and trigger_buffer.tree:
                    trigger_size = trigger_buffer.tree.n_entries
                elif hasattr(trigger_buffer, "size"):
                    trigger_size = trigger_buffer.size

            if hasattr(self.policy, "harvester") and hasattr(self.policy.harvester, "buffer"):
                harvester_buffer = self.policy.harvester.buffer
                if hasattr(harvester_buffer, "tree") and harvester_buffer.tree:
                    harvester_size = harvester_buffer.tree.n_entries
                elif hasattr(harvester_buffer, "size"):
                    harvester_size = harvester_buffer.size

            # Only train if at least one agent has sufficient experiences
            if trigger_size >= MIN_EXPERIENCES_FOR_TRAINING or harvester_size >= MIN_EXPERIENCES_FOR_TRAINING:
                try:
                    train_metrics = self.policy.train_step(self.adaptive_reg)
                    # Store latest losses for HUD reporting
                    if train_metrics.get("trigger"):
                        self.last_trigger_loss = train_metrics["trigger"].get("loss", self.last_trigger_loss)
                    if train_metrics.get("harvester"):
                        self.last_harvester_loss = train_metrics["harvester"].get("loss", self.last_harvester_loss)
                    # Log training stats after each training step
                    stats = self.policy.get_training_stats() if hasattr(self.policy, "get_training_stats") else {}
                    LOG.info("[TRAINING] TriggerAgent stats: %s", stats.get("trigger", {}))
                    LOG.info("[TRAINING] HarvesterAgent stats: %s", stats.get("harvester", {}))
                    self.bars_since_training = 0

                    LOG.debug(
                        "[TRAINING] Completed: T_buffer=%d H_buffer=%d",
                        trigger_size,
                        harvester_size,
                    )

                    # Adjust regularization based on training metrics
                    if train_metrics.get("trigger") or train_metrics.get("harvester"):
                        trigger_td = (
                            train_metrics.get("trigger", {}).get("mean_td_error", 0)
                            if train_metrics.get("trigger")
                            else 0
                        )
                        harvester_td = (
                            train_metrics.get("harvester", {}).get("mean_td_error", 0)
                            if train_metrics.get("harvester")
                            else 0
                        )
                        avg_td = (trigger_td + harvester_td) / 2 if (trigger_td + harvester_td) else 0

                        # Adaptive regularization: high TD error = overfitting signal
                        if avg_td > TRAINING_TD_HIGH_THRESHOLD:  # High TD error threshold
                            self.adaptive_reg.increase_regularization()
                        elif avg_td < TRAINING_TD_LOW_THRESHOLD:  # Low TD error threshold
                            self.adaptive_reg.decrease_regularization()

                    # Periodic auto-save every 50 training steps
                    total_steps = getattr(self.policy.trigger, "training_steps", 0) + getattr(
                        self.policy.harvester, "training_steps", 0
                    )
                    if total_steps > 0 and total_steps % 50 == 0:
                        try:
                            self.policy.save_checkpoint()
                        except Exception as e_save:
                            LOG.warning("[CHECKPOINT] Auto-save failed: %s", e_save)
                except Exception as e:
                    LOG.error(f"[TRAINING] Training step failed: {e}", exc_info=True)
                    self.bars_since_training = 0  # Reset to prevent repeated attempts
            else:
                LOG.debug(
                    "[TRAINING] Skipping - insufficient experiences (T=%d H=%d, need %d)",
                    trigger_size,
                    harvester_size,
                    MIN_EXPERIENCES_FOR_TRAINING,
                )

        # Phase 3: Use DualPolicy if available
        has_decide_entry = hasattr(self.policy, "decide_entry")
        LOG.debug("[POLICY-CHECK] policy type=%s, has_decide_entry=%s", type(self.policy).__name__, has_decide_entry)
        if has_decide_entry:
            # Dual-agent architecture
            # HEDGING MODE: Check if ANY positions exist (not just net=0)
            has_positions = self.trade_integration.has_any_open_positions()
            LOG.debug("[FLAT: Check for entry] has_positions=%s, cur_pos=%d", has_positions, self.cur_pos)
            LOG.debug("[FLOW-TRACE] Step 1: Position check complete, proceeding to entry decision")
            if not has_positions:
                # Handbook: Check circuit breakers BEFORE any entry decision
                LOG.debug("[FLOW-TRACE] Step 2: Checking circuit breakers")
                if self.circuit_breakers.is_any_tripped():
                    tripped = self.circuit_breakers.get_tripped_breakers()
                    LOG.warning("[CIRCUIT-BREAKER] Trading halted: %s", ", ".join([b.name for b in tripped]))
                    LOG.error("[FLOW-ABORT] Stopped at circuit breaker check")
                    self._export_hud_data()
                    return
                LOG.debug("[FLOW-TRACE] Step 2 PASSED: No circuit breakers tripped")

                # Log key time features
                if len(self.bars) % 10 == 0:  # Every 10 bars
                    active_sessions = self.event_time_engine.get_active_sessions()
                    LOG.debug(
                        "[EVENT-TIME] Sessions: %s | High liquidity: %s",
                        ",".join(active_sessions) if active_sessions else "None",
                        is_high_liq,
                    )
                depth_floor = max(self.last_depth_floor, 0.0)
                LOG.info(
                    "[FLOW-TRACE] Step 3: Checking order book depth (bid=%.3f, ask=%.3f, floor=%.3f)",
                    depth_bid,
                    depth_ask,
                    depth_floor,
                )
                if self._depth_is_too_thin(depth_bid, depth_ask, depth_floor):
                    self.last_depth_gate = True
                    min_depth = min(depth_bid, depth_ask) if depth_bid > 0 and depth_ask > 0 else 0.0
                    LOG.warning(
                        "[DEPTH] Order book thin: bid=%.3f ask=%.3f min=%.3f < buffer=%.3f (levels=%d)",
                        depth_bid,
                        depth_ask,
                        min_depth,
                        depth_floor,
                        depth_levels,
                    )
                    LOG.error("[FLOW-ABORT] Stopped at depth check")
                    self._export_hud_data()
                    return
                self.last_depth_gate = False
                LOG.debug("[FLOW-TRACE] Step 3 PASSED: Depth check OK")

                # FLAT: Check for entry (pass event features if policy accepts them)
                LOG.debug("[FLOW-TRACE] Step 4: Calling policy.decide_entry()")
                action, confidence, runway = self.policy.decide_entry(
                    self.bars,
                    imbalance=imbalance,
                    vpin_z=vpin_zscore,
                    depth_ratio=depth_ratio,
                    realized_vol=realized_vol,
                    event_features=event_features,
                )
                # action: 0=NO_ENTRY, 1=LONG, 2=SHORT
                if action == ACTION_LONG:
                    desired = 1
                elif action == ACTION_SHORT:
                    desired = -1
                else:
                    desired = 0
                LOG.info(
                    "[FLOW-TRACE] Step 4 COMPLETE: action=%d, desired=%d, confidence=%.3f", action, desired, confidence
                )

                # GAP 10.1: Log trigger decision
                if action == 0:
                    decision_str = "NO_ENTRY"
                elif action == 1:
                    decision_str = "LONG"
                else:
                    decision_str = "SHORT"
                regime = (
                    getattr(self.policy, "current_regime", "UNKNOWN") if hasattr(self.policy, "policy") else "UNKNOWN"
                )
                geom_temp = (
                    self.policy.path_geometry.last
                    if hasattr(self.policy, "path_geometry") and self.policy.path_geometry
                    else {}
                )
                self.decision_log.log_trigger_decision(
                    decision=decision_str,
                    confidence=confidence,
                    price=c,
                    volatility=realized_vol,
                    imbalance=imbalance,
                    vpin_z=vpin_zscore,
                    regime=regime,
                    predicted_runway=runway,
                    feasibility=geom_temp.get("feasibility", 1.0),
                    circuit_breakers_ok=not self.circuit_breakers.is_any_tripped(),
                )

                # Store state for online learning (if entry is taken)
                if action != 0 and hasattr(self.policy, "trigger") and hasattr(self.policy.trigger, "last_state"):
                    self.entry_state = (
                        self.policy.trigger.last_state.copy() if self.policy.trigger.last_state is not None else None
                    )
                    self.entry_action = action
                    self.entry_confidence = confidence  # Store for Platt calibration at trade close
                    self._last_trigger_conf = 0.9 * self._last_trigger_conf + 0.1 * confidence
                    self.predicted_runway = runway  # FIX 1: Store predicted MFE for trigger reward
                    # Store market-condition snapshot at entry for regime-conditioned reward shaping
                    self.entry_vpin_z = vpin_zscore
                    # entry_var is set in the execution block where current_var is computed
                    # FIX 7: Track if entry was exploration (random) so we can use
                    # a neutral reward instead of prediction-accuracy reward
                    self.was_exploration_entry = (
                        hasattr(self.policy, "trigger")
                        and self.policy.trigger.epsilon > EPSILON_HIGH_THRESHOLD
                        and runway <= RUNWAY_FALLBACK_THRESHOLD
                    )

                # FIX 5: Add NO_ENTRY experiences so trigger learns when NOT to trade
                # This provides negative/neutral examples critical for balanced learning
                # RATE-LIMITED: Only add 10% of NO_ENTRY bars to prevent buffer flooding
                # (~288 bars/day would dominate vs ~5-10 actual trades/day)
                if (
                    action == 0
                    and hasattr(self.policy, "add_trigger_experience")
                    and random.random() < EXPLORATION_SAMPLE_RATE  # Sample a fraction of NO_ENTRY bars
                ):
                        trigger_state = (
                            self.policy.trigger.last_state.copy()
                            if hasattr(self.policy, "trigger")
                            and hasattr(self.policy.trigger, "last_state")
                            and self.policy.trigger.last_state is not None
                            else None
                        )
                        if trigger_state is not None:
                            # Reward of 0.0 for NO_ENTRY: neutral - neither penalize nor reward inaction
                            self.policy.add_trigger_experience(
                                state=trigger_state,
                                action=0,  # NO_ENTRY
                                reward=0.0,
                                next_state=trigger_state,  # Same state (no position change)
                                done=True,
                            )
                            LOG.debug("[ONLINE_LEARNING] Added NO_ENTRY trigger experience (sampled)")

                # Get PathGeometry features for logging
                geom = (
                    self.policy.path_geometry.last
                    if hasattr(self.policy, "path_geometry") and self.policy.path_geometry
                    else {}
                )
                feas = geom.get("feasibility", 0.0)
                geom.get("runway", 0.0)
                geom.get("efficiency", 0.0)

                LOG.info(
                    "[BAR M%d] %s O=%.2f H=%.2f L=%.2f C=%.2f | TRIGGER: action=%d conf=%.2f runway=%.4f feas=%.2f | RS_vol=%.5f bars=%d | desired=%s cur=%s",
                    self.timeframe_minutes,
                    t.isoformat(),
                    o,
                    h,
                    low_price,
                    c,
                    action,
                    confidence,
                    runway,
                    feas,
                    realized_vol,
                    len(self.bars),
                    desired,
                    self.cur_pos,
                )
            else:
                # IN POSITION: Check for exit
                # Skip if tick-level harvester already issued a close for this position
                _already_closing = hasattr(self, "_pending_closes") and len(self._pending_closes) > 0
                if _already_closing:
                    LOG.debug("[BAR] Tick harvester already issued close — skipping bar-level exit check")
                    exit_action = 0
                    exit_conf = 0.0
                else:
                    exit_action, exit_conf = self.policy.decide_exit(
                        self.bars,
                        current_price=c,
                        imbalance=imbalance,
                        vpin_z=vpin_zscore,
                        depth_ratio=depth_ratio,
                        event_features=event_features,
                    )
                self._last_harvester_conf = 0.9 * self._last_harvester_conf + 0.1 * exit_conf
                ticks_held = pos_metrics.get("ticks_held", 0)
                unrealized_pnl = (c - entry_price) * self.cur_pos if entry_price > 0 else 0.0
                capture_ratio = (unrealized_pnl / mfe) if mfe > 0 else 0.0

                self.decision_log.log_harvester_decision(
                    decision="CLOSE" if exit_action == 1 else "HOLD",
                    confidence=exit_conf,
                    price=c,
                    entry_price=entry_price,
                    mfe=mfe,
                    mae=mae,
                    ticks_held=ticks_held,
                    unrealized_pnl=unrealized_pnl,
                    capture_ratio=capture_ratio,
                )

                # Update trailing stop (HarvesterAgent controls, TradeManager communicates)
                if hasattr(self, "trade_integration") and self.trade_integration.trailing_stop_active:
                    mid_price = (
                        (float(self.best_bid) + float(self.best_ask)) / 2.0 if self.best_bid and self.best_ask else c
                    )
                    self.trade_integration.update_trailing_stop(mid_price)

                LOG.info(
                    "[BAR M%d] %s O=%.2f H=%.2f L=%.2f C=%.2f | HARVESTER: exit=%d conf=%.2f | desired=%s cur=%s",
                    self.timeframe_minutes,
                    t.isoformat(),
                    o,
                    h,
                    low_price,
                    c,
                    exit_action,
                    exit_conf,
                    desired,
                    self.cur_pos,
                )

                # Add harvester experience (dense feedback every bar)
                if hasattr(self.policy, "add_harvester_experience") and self.prev_harvester_state is not None:
                    pos_metrics = (
                        self.policy.get_position_metrics() if hasattr(self.policy, "get_position_metrics") else {}
                    )
                    current_mfe = pos_metrics.get("mfe", 0.0)
                    current_mae = pos_metrics.get("mae", 0.0)
                    bars_held_count = pos_metrics.get("bars_held", 0)
                    entry_price_val = pos_metrics.get("entry_price", 0.0)
                    current_price_val = c
                    unrealized_pnl_val = (
                        (current_price_val - entry_price_val) * self.cur_pos if entry_price_val > 0 else 0.0
                    )

                    # FIX 2: Calculate principled HOLD reward based on capture potential
                    realized_vol = self._calculate_rs_volatility() if len(self.bars) >= MIN_BARS_FOR_VOL_CALC else 0.01  # Default fallback

                    reward = self._calculate_harvester_hold_reward(
                        excursions=MfeMaeSnapshot(
                            current_mfe=current_mfe,
                            current_mae=current_mae,
                            prev_mfe=self.prev_mfe,
                            prev_mae=self.prev_mae,
                        ),
                        position=HarvesterPositionState(
                            bars_held=bars_held_count,
                            unrealized_pnl=unrealized_pnl_val,
                            entry_price=entry_price_val,
                            current_price=current_price_val,
                            realized_vol=realized_vol,
                        ),
                    )

                    next_state = (
                        self.policy.harvester.last_state
                        if hasattr(self.policy, "harvester") and hasattr(self.policy.harvester, "last_state")
                        else None
                    )
                    if next_state is not None:
                        self.policy.add_harvester_experience(
                            state=self.prev_harvester_state,
                            action=self.prev_exit_action,
                            reward=reward,
                            next_state=next_state,
                            done=False,
                        )
                        LOG.debug("[HARVESTER_EXP] HOLD reward=%.4f", reward)

                # Store current state for next bar
                if hasattr(self.policy, "harvester") and hasattr(self.policy.harvester, "last_state"):
                    self.prev_harvester_state = self.policy.harvester.last_state
                    self.prev_exit_action = exit_action
                    pos_metrics = (
                        self.policy.get_position_metrics() if hasattr(self.policy, "get_position_metrics") else {}
                    )
                    self.prev_mfe = pos_metrics.get("mfe", 0.0)
                    self.prev_mae = pos_metrics.get("mae", 0.0)
        else:
            # Simple policy fallback
            action = self.policy.decide(self.bars)
            if action == 0:
                desired = -1
            elif action == 1:
                desired = 0
            else:
                desired = 1

            LOG.info(
                "[BAR M%d] %s O=%.2f H=%.2f L=%.2f C=%.2f | desired=%s cur=%s",
                self.timeframe_minutes,
                t.isoformat(),
                o,
                h,
                low_price,
                c,
                desired,
                self.cur_pos,
            )

        # --- Decision Log Export (Tab 6 HUD) - AFTER ALL decision logic, BEFORE early returns ---
        try:
            log_path = Path("data/decision_log.json")
            log_path.parent.mkdir(exist_ok=True)
            # Read existing log (if any)
            if log_path.exists():
                with open(log_path, encoding="utf-8") as f:
                    log_entries = json.load(f)
            else:
                log_entries = []

            # Get position metrics from policy (if dual-agent)
            pos_metrics = {}
            if hasattr(self.policy, "get_position_metrics"):
                pos_metrics = self.policy.get_position_metrics()

            # Compose log entry with all decision variables
            log_entry = {
                "timestamp": t.isoformat() if hasattr(t, "isoformat") else str(t),
                "event": "bar_close",
                "details": {
                    "open": o,
                    "high": h,
                    "low": low_price,
                    "close": c,
                    "cur_pos": self.cur_pos,
                    "desired": desired if "desired" in locals() else None,
                    "depth_bid": depth_bid if "depth_bid" in locals() else None,
                    "depth_ask": depth_ask if "depth_ask" in locals() else None,
                    "depth_ratio": depth_ratio if "depth_ratio" in locals() else None,
                    "imbalance": imbalance if "imbalance" in locals() else None,
                    "runway": runway if "runway" in locals() else None,
                    "feasibility": feas if "feas" in locals() else None,
                    "action": action if "action" in locals() else None,
                    "confidence": confidence if "confidence" in locals() else None,
                    "exit_action": exit_action if "exit_action" in locals() else None,
                    "exit_conf": exit_conf if "exit_conf" in locals() else None,
                    # Harvester position metrics
                    "mfe": pos_metrics.get("mfe", None),
                    "mae": pos_metrics.get("mae", None),
                    "bars_held": pos_metrics.get("bars_held", None),
                    "entry_price": pos_metrics.get("entry_price", None),
                    "circuit_breaker": (
                        self.circuit_breakers.is_any_tripped() if hasattr(self, "circuit_breakers") else None
                    ),
                },
            }
            # Only keep last 1000 entries
            log_entries.append(log_entry)
            if len(log_entries) > MAX_LOG_ENTRIES:
                log_entries = log_entries[-MAX_LOG_ENTRIES:]
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entries, f, indent=2)
            LOG.info(f"[DECISION_LOG] Wrote entry for {t} (total: {len(log_entries)})")
        except Exception as e:
            LOG.error(f"[DECISION_LOG] Failed to write: {e}", exc_info=True)

        LOG.info(
            "[FLOW-TRACE] Step 5: Checking execution preconditions (trade_sid=%s, desired=%s, cur_pos=%s)",
            self.trade_sid,
            desired,
            self.cur_pos,
        )
        if not self.trade_sid:
            LOG.error("[FLOW-ABORT] No TRADE session - cannot execute")
            self._export_hud_data()  # Still export HUD even without trade session
            return
        if desired == self.cur_pos:
            LOG.debug("[FLOW-ABORT] No action needed (desired=%s equals cur_pos=%s)", desired, self.cur_pos)
            self._export_hud_data()  # Still export HUD even when no action
            return
        LOG.debug("[FLOW-TRACE] Step 5 PASSED: Execution preconditions met")

        # Phase 3.5: VaR circuit breaker check before new entries
        LOG.debug("[FLOW-TRACE] Step 6: Entry validation (cur_pos=%s, desired=%s)", self.cur_pos, desired)
        if self.cur_pos == 0 and desired != 0:
            # Only check VaR for new position entries
            LOG.debug("[FLOW-TRACE] Step 6a: Checking kurtosis breaker")
            if self.kurtosis_monitor.is_breaker_active:
                LOG.warning(
                    "[CIRCUIT_BREAKER] Kurtosis breaker ACTIVE (kurtosis=%.2f) - skipping entry",
                    self.kurtosis_monitor.current_kurtosis,
                )
                LOG.error("[FLOW-ABORT] Stopped at kurtosis check")
                self._export_hud_data()  # Export even on circuit breaker
                return
            LOG.debug("[FLOW-TRACE] Step 6a PASSED: Kurtosis OK")

            # Calculate current VaR with regime-aware multiplier
            current_var = self.var_estimator.estimate_var(
                regime=self._current_var_regime(), vpin_z=vpin_zscore, current_vol=realized_vol
            )
            self.last_estimated_var = current_var
            # Persist the VaR at entry so the close reward can apply regime conditioning.
            # (Only stored when we're about to enter; overwritten each time.)
            self.entry_var = current_var
            max_var_threshold = self.vol_cap
            if current_var > max_var_threshold:
                if not self.paper_mode:
                    LOG.warning(
                        "[CIRCUIT_BREAKER] VaR=%.4f exceeds threshold=%.4f - skipping entry",
                        current_var,
                        max_var_threshold,
                    )
                    self._export_hud_data()  # Export even on VaR filter
                    return
                else:
                    LOG.debug(
                        "[PAPER-GATE] VaR=%.4f > cap=%.4f — allowing entry so RL learns via regime reward",
                        current_var,
                        max_var_threshold,
                    )
            if self.vpin_z_threshold > 0 and vpin_zscore > self.vpin_z_threshold:
                if not self.paper_mode:
                    LOG.warning(
                        "[VPIN] z-score %.2f exceeds threshold %.2f - skipping entry",
                        vpin_zscore,
                        self.vpin_z_threshold,
                    )
                    self._export_hud_data()
                    return
                else:
                    LOG.debug(
                        "[PAPER-GATE] VPIN z=%.2f > thresh=%.2f — allowing entry so RL learns via regime reward",
                        vpin_zscore,
                        self.vpin_z_threshold,
                    )

            # Phase 3.5: Learned spread threshold check (2x minimum observed spread)
            env_multiplier = None
            spread_override = os.environ.get("SPREAD_RELAX_MULTIPLIER")
            if spread_override:
                try:
                    env_multiplier = float(spread_override)
                except ValueError:
                    LOG.warning(
                        "[SPREAD_FILTER] Invalid SPREAD_RELAX_MULTIPLIER=%s - ignoring",
                        spread_override,
                    )
            is_acceptable, current_spread, max_spread = self.friction_calculator.is_spread_acceptable(env_multiplier)
            effective_multiplier = (
                env_multiplier if env_multiplier is not None else self.friction_calculator.spread_multiplier
            )
            if not is_acceptable:
                if not self.paper_mode:
                    LOG.warning(
                        "[SPREAD_FILTER] Current=%.2f pips > Learned max=%.2f pips (%.1fx min) - skipping entry",
                        current_spread,
                        max_spread,
                        effective_multiplier,
                    )
                    self._export_hud_data()  # Export even on filtered entries
                    return
                else:
                    LOG.debug(
                        "[PAPER-GATE] Spread %.2f > max %.2f — allowing entry so RL learns friction cost",
                        current_spread,
                        max_spread,
                    )

        # Export HUD data every bar (before potential order)
        self._export_hud_data()

        delta = desired - self.cur_pos
        side = "1" if delta > 0 else "2"

        LOG.debug("[FLOW-TRACE] Step 7: Computing order (delta=%s, side=%s)", delta, side)
        # Handbook: Apply circuit breaker position size multiplier (reduces size during high risk)
        size_multiplier = self.circuit_breakers.get_position_size_multiplier()
        order_qty = self._compute_order_qty(abs(delta), size_multiplier, self.cur_pos == 0 and desired != 0)

        if size_multiplier < 1.0:
            LOG.warning(
                "[CIRCUIT-BREAKER] Position size reduced: %.2f%% (multiplier=%.2f)",
                size_multiplier * 100,
                size_multiplier,
            )
        if order_qty <= 0:
            LOG.warning("[RISK] Order blocked - zero qty after constraints")
            LOG.error("[FLOW-ABORT] Stopped at order quantity check (qty=%.6f)", order_qty)
            self._export_hud_data()
            return

        LOG.debug("[FLOW-TRACE] Step 8: EXECUTING ORDER side=%s qty=%.6f", side, order_qty)
        self.send_market_order(side=side, qty=order_qty)

    def send_market_order(self, side: str, qty: float):
        """DEPRECATED: Use TradeManager API via trade_integration.enter_position() instead."""
        # FIX P0-3: Migrate to TradeManager API for centralized order tracking
        # ROCK SOLID: Block trading on stale prices or unhealthy connection
        if not self.connection_healthy:
            LOG.warning("[SAFETY] ✗ Order blocked - connection unhealthy")
            return None

        if self.last_quote_heartbeat:
            quote_age = (utc_now() - self.last_quote_heartbeat).total_seconds()
            if quote_age > self.max_quote_age_for_trading:
                LOG.warning(
                    "[SAFETY] ✗ Order blocked - quote data stale (%.1fs old, max=%ds)",
                    quote_age,
                    self.max_quote_age_for_trading,
                )
                return None

        if not self.trade_sid:
            LOG.warning("[SAFETY] ✗ Order blocked - TRADE session not connected")
            return None

        if not self.trade_integration.trade_manager:
            LOG.error("[SAFETY] ✗ Order blocked - TradeManager not initialized")
            return None

        # P0 FIX: Check for order acknowledgment timeouts
        self.trade_integration.trade_manager.check_pending_order_timeouts()

        # P0 FIX: Adjust quantity for realistic execution costs
        last_close = self.bars[-1][4] if self.bars else 0.0
        mid_price = (
            (float(self.best_bid) + float(self.best_ask)) / 2.0 if self.best_bid and self.best_ask else last_close
        )
        spread_bps = 0.0
        if self.best_bid and self.best_ask and mid_price > 0:
            spread_bps = ((float(self.best_ask) - float(self.best_bid)) / mid_price) * 10000.0

        exec_side = OrderSide.BUY if side == "1" else OrderSide.SELL
        adjusted_qty = self.execution_model.adjust_position_size_for_costs(
            side=exec_side,
            target_quantity=qty,
            mid_price=mid_price,
            spread_bps=spread_bps,
            regime=getattr(self.policy, "current_regime", "UNKNOWN"),
        )

        if adjusted_qty < qty:
            LOG.info(
                "[EXECUTION] Cost-adjusted quantity: %.6f → %.6f (%.1f%% reduction, spread=%.1f bps)",
                qty,
                adjusted_qty,
                100.0 * (1.0 - adjusted_qty / qty),
                spread_bps,
            )

        # Use adjusted quantity
        qty = adjusted_qty

        # Normalize to broker's volume constraints (min/max/step)
        normalized_qty = self.friction_calculator.normalize_quantity(qty)
        if abs(normalized_qty - qty) > MIN_POSITION_QTY:
            LOG.info(
                "[EXECUTION] Volume normalized: %.6f → %.6f (step=%.4f)",
                qty,
                normalized_qty,
                self.friction_calculator.costs.volume_step,
            )
        qty = normalized_qty

        # Use TradeManager API for proper order lifecycle tracking
        tm_side = Side.BUY if side == "1" else Side.SELL

        order = self.trade_integration.trade_manager.submit_market_order(side=tm_side, quantity=qty, tag_prefix="DDQN")

        if order:
            # GAP 10.1: Log order submission
            self.transaction_log.log_order_submit(
                order_id=order.clord_id, side=tm_side.name, quantity=qty, price=None  # Market order - no limit price
            )

            # Phase 3.5: Notify activity monitor of trade execution
            self.activity_monitor.on_trade_executed()
            LOG.info(
                "[TRADE] Submitted MKT %s qty=%.6f via TradeManager (ClOrdID=%s)",
                tm_side.name,
                qty,
                order.clord_id,
            )
        else:
            LOG.error("[TRADE] Failed to submit order via TradeManager")

        return order

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
            now = dt.datetime.now(dt.UTC)
            uptime_seconds = int((now - self.start_time).total_seconds())

            # 1. Bot Configuration
            bot_config = {
                "symbol": self.symbol,
                "symbol_id": self.symbol_id,
                "timeframe": f"{self.timeframe_minutes}m",
                "uptime_seconds": uptime_seconds,
                "bar_count": self.bar_count,
                "training_enabled": os.environ.get("DDQN_ONLINE_LEARNING", "1") == "1",
                "starting_equity": self.starting_equity,
                "qty": self.qty,
            }
            with open(self.hud_data_dir / "bot_config.json", "w", encoding="utf-8") as f:
                json.dump(bot_config, f, indent=2)

            # 2. Current Position
            current_price = (
                (float(self.best_bid) + float(self.best_ask)) / 2.0
                if self.best_bid and self.best_ask
                else (float(self.bars[-1][4]) if self.bars else 0.0)
            )
            if self.cur_pos > 0:
                direction = "LONG"
            elif self.cur_pos < 0:
                direction = "SHORT"
            else:
                direction = "FLAT"

            # Get MFE/MAE from multi-position tracker (look up any active tracker)
            tracker = None
            with self._tracker_lock:
                if self.mfe_mae_trackers:
                    # Use the most recent active tracker
                    tracker = next(reversed(self.mfe_mae_trackers.values()), None)
            if tracker is None:
                tracker = self.mfe_mae_tracker  # Legacy fallback
            entry_price = float(tracker.entry_price) if self.cur_pos != 0 and tracker.entry_price else 0.0

            # Debug: Log tracker state
            if self.cur_pos != 0 and entry_price == 0:
                LOG.warning(
                    "[HUD_DEBUG] Tracker has entry_price=%s (cur_pos=%d, tracker=%s, trackers_count=%d)",
                    tracker.entry_price,
                    self.cur_pos,
                    type(tracker).__name__,
                    len(self.mfe_mae_trackers),
                )

            mfe = float(tracker.mfe) if self.cur_pos != 0 and tracker.mfe else 0.0
            mae = float(tracker.mae) if self.cur_pos != 0 and tracker.mae else 0.0

            # Calculate unrealized PnL
            unrealized_pnl = 0.0
            bars_held = 0
            if self.cur_pos != 0 and entry_price > 0:
                # Get actual position qty from TradeManager if available
                actual_qty = self.qty  # Default
                if self.trade_integration and self.trade_integration.trade_manager:
                    pos = self.trade_integration.trade_manager.get_position()
                    actual_qty = float(abs(pos.net_qty)) if abs(pos.net_qty) > 0 else self.qty
                # Convert MFE / MAE from raw price-points → USD (same unit as unrealized_pnl)
                mfe *= actual_qty
                mae *= actual_qty
                if self.cur_pos > 0:  # LONG
                    unrealized_pnl = (current_price - entry_price) * actual_qty
                else:  # SHORT
                    unrealized_pnl = (entry_price - current_price) * actual_qty
                # Use any active path recorder (look up like tracker)
                with self._tracker_lock:
                    position_path_recorder = (
                        next(iter(self.path_recorders.values()), self.path_recorder)
                        if self.path_recorders
                        else self.path_recorder
                    )
                bars_held = len(position_path_recorder.path) if hasattr(position_path_recorder, "path") else 0

            position_data = {
                "direction": direction,
                "entry_price": entry_price,
                "current_price": current_price,
                "mfe": mfe,
                "mae": mae,
                "unrealized_pnl": unrealized_pnl,
                "bars_held": bars_held,
            }
            with open(self.hud_data_dir / "current_position.json", "w", encoding="utf-8") as f:
                json.dump(position_data, f, indent=2)

            # 3. Performance Metrics (from PerformanceTracker)
            metrics = self.performance.get_metrics() if hasattr(self.performance, "get_metrics") else {}
            performance_snapshot = {
                "daily": {
                    "total_trades": metrics.get("total_trades", 0),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "total_pnl": metrics.get("total_pnl", 0.0),
                    "sharpe_ratio": metrics.get("sharpe", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                },
                "weekly": {
                    "total_trades": metrics.get("total_trades", 0),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "total_pnl": metrics.get("total_pnl", 0.0),
                    "sharpe_ratio": metrics.get("sharpe", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                },
                "monthly": {
                    "total_trades": metrics.get("total_trades", 0),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "total_pnl": metrics.get("total_pnl", 0.0),
                    "sharpe_ratio": metrics.get("sharpe", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                },
                "lifetime": {
                    "total_trades": metrics.get("total_trades", 0),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "total_pnl": metrics.get("total_pnl", 0.0),
                    "sharpe_ratio": metrics.get("sharpe", 0.0),
                    "sortino_ratio": metrics.get("sortino", 0.0),
                    "omega_ratio": metrics.get("omega", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                },
            }
            with open(self.hud_data_dir / "performance_snapshot.json", "w", encoding="utf-8") as f:
                json.dump(performance_snapshot, f, indent=2)

            # 4. Training Stats
            trigger_buffer_size = 0
            harvester_buffer_size = 0
            trigger_loss = getattr(self, "last_trigger_loss", 0.0)
            harvester_loss = getattr(self, "last_harvester_loss", 0.0)
            trigger_training_steps = 0
            harvester_training_steps = 0
            trigger_epsilon = 0.0
            harvester_beta = 0.4
            trigger_ready = False
            harvester_ready = False
            harvester_min_hold = 10
            total_agents = 0
            arena_diversity = {"trigger_diversity": 0.0, "harvester_diversity": 0.0}
            last_agreement = 1.0

            # Use get_training_stats() which correctly reads from .buffer attribute
            if hasattr(self.policy, "get_training_stats"):
                policy_stats = self.policy.get_training_stats()
                t_stats = policy_stats.get("trigger", {})
                h_stats = policy_stats.get("harvester", {})
                trigger_buffer_size = t_stats.get("buffer_size", 0)
                harvester_buffer_size = h_stats.get("buffer_size", 0)
                trigger_training_steps = t_stats.get("training_steps", 0)
                harvester_training_steps = h_stats.get("training_steps", 0)
                trigger_epsilon = t_stats.get("epsilon", 0.0)
                harvester_beta = h_stats.get("beta", 0.4)
                trigger_ready = t_stats.get("ready_to_train", False)
                harvester_ready = h_stats.get("ready_to_train", False)
                harvester_min_hold = h_stats.get("min_hold_ticks", 10)
            else:
                # Fallback: direct attribute access using correct name (.buffer not .replay_buffer)
                if hasattr(self.policy, "trigger") and hasattr(self.policy.trigger, "buffer"):
                    buf = self.policy.trigger.buffer
                    if hasattr(buf, "tree") and buf.tree:
                        trigger_buffer_size = buf.tree.n_entries
                    elif hasattr(buf, "size"):
                        trigger_buffer_size = buf.size
                if hasattr(self.policy, "harvester") and hasattr(self.policy.harvester, "buffer"):
                    buf = self.policy.harvester.buffer
                    if hasattr(buf, "tree") and buf.tree:
                        harvester_buffer_size = buf.tree.n_entries
                    elif hasattr(buf, "size"):
                        harvester_buffer_size = buf.size

            # Check for arena (multi-agent)
            if hasattr(self.policy, "trigger") and hasattr(self.policy.trigger, "arena"):
                arena = self.policy.trigger.arena
                total_agents = len(arena.agents) if hasattr(arena, "agents") else 0
                if hasattr(arena, "last_diversity"):
                    arena_diversity["trigger_diversity"] = arena.last_diversity
                if hasattr(arena, "last_agreement"):
                    last_agreement = arena.last_agreement

            training_stats = {
                "trigger_buffer_size": trigger_buffer_size,
                "harvester_buffer_size": harvester_buffer_size,
                "trigger_training_steps": trigger_training_steps,
                "harvester_training_steps": harvester_training_steps,
                "trigger_epsilon": trigger_epsilon,
                "harvester_beta": harvester_beta,
                "trigger_ready": trigger_ready,
                "harvester_ready": harvester_ready,
                "harvester_min_hold_ticks": harvester_min_hold,
                "total_agents": total_agents,
                "arena_diversity": arena_diversity,
                "last_agreement_score": last_agreement,
                "consensus_mode": "weighted_average",
                "trigger_loss": trigger_loss,
                "harvester_loss": harvester_loss,
                "last_training_time": ("Active" if self.bars_since_training < self.training_interval else "Never"),
            }
            with open(self.hud_data_dir / "training_stats.json", "w", encoding="utf-8") as f:
                json.dump(training_stats, f, indent=2)

            # 5. Risk Metrics
            # VaR and kurtosis
            realized_vol = self._calculate_rs_volatility() if len(self.bars) >= MIN_BARS_FOR_VAR_UPDATE else 0.0
            vpin_stats = self.last_vpin_stats if hasattr(self, "last_vpin_stats") else {"vpin": 0.0, "zscore": 0.0}
            vpin_value = vpin_stats.get("vpin", 0.0)
            vpin_zscore = vpin_stats.get("zscore", 0.0)
            current_var = (
                self.var_estimator.estimate_var(
                    regime=self._current_var_regime(), vpin_z=vpin_zscore, current_vol=realized_vol
                )
                if hasattr(self.var_estimator, "estimate_var")
                else 0.0
            )
            self.last_estimated_var = current_var
            current_kurtosis = (
                self.kurtosis_monitor.current_kurtosis if hasattr(self.kurtosis_monitor, "current_kurtosis") else 0.0
            )
            circuit_breaker = "ACTIVE" if self.kurtosis_monitor.is_breaker_active else "INACTIVE"

            # Seed regime detector from historical bars if still UNKNOWN after restart
            if (
                hasattr(self.policy, "seed_regime_from_bars")
                and getattr(self.policy, "current_regime", "UNKNOWN") == "UNKNOWN"
                and len(self.bars) >= 10
            ):
                self.policy.seed_regime_from_bars(self.bars)

            # Regime from dual policy
            regime = "UNKNOWN"
            regime_zeta = 1.0
            if hasattr(self.policy, "current_regime"):
                regime = self.policy.current_regime
            if hasattr(self.policy, "current_zeta"):
                regime_zeta = self.policy.current_zeta

            # Path geometry features — recompute fresh from current bars so HUD
            # always reflects live market state, not just the last entry/exit call.
            if len(self.bars) >= 3 and realized_vol > 0:
                geom = self.path_geometry.update(self.bars, realized_vol)
            else:
                geom = self.path_geometry.last if hasattr(self.path_geometry, "last") else {}
            efficiency = geom.get("efficiency", 1.0)
            gamma = geom.get("gamma", 0.0)
            jerk = geom.get("jerk", 0.0)
            runway = geom.get("runway", 0.5)
            feasibility = geom.get("feasibility", 0.5)

            # Market microstructure
            spread = float(self.best_ask) - float(self.best_bid) if self.best_bid and self.best_ask else 0.0
            depth_metrics = getattr(self, "last_depth_metrics", {}) or {}
            depth_bid = depth_metrics.get("bid", 0.0)
            depth_ask = depth_metrics.get("ask", 0.0)
            depth_ratio = depth_metrics.get("ratio", 1.0)
            depth_levels = depth_metrics.get("levels", 0)
            imbalance = (
                SafeMath.safe_div(depth_bid - depth_ask, depth_bid + depth_ask, 0.0)
                if (depth_bid + depth_ask) > 0
                else 0.0
            )
            depth_buffer = getattr(getattr(self, "friction_calculator", None), "depth_buffer", 0.0)
            depth_gate_active = getattr(self, "last_depth_gate", False)

            risk_metrics = {
                "var": current_var,
                "kurtosis": current_kurtosis,
                "circuit_breaker": circuit_breaker,
                "realized_vol": realized_vol,
                "regime": regime,
                "regime_zeta": regime_zeta,
                "vpin": vpin_value,
                "vpin_zscore": vpin_zscore,
                "vpin_threshold": self.vpin_z_threshold,
                "efficiency": efficiency,
                "gamma": gamma,
                "jerk": jerk,
                "runway": runway,
                "feasibility": feasibility,
                "spread": spread,
                "imbalance": imbalance,
                "depth_bid": depth_bid,
                "depth_ask": depth_ask,
                "depth_ratio": depth_ratio,
                "depth_levels": depth_levels,
                "depth_buffer": depth_buffer,
                "depth_gate_active": depth_gate_active,
                # Per-level L2 book snapshots (list of [price, size] pairs)
                "order_book_bids": [
                    [p, s] for p, s in list(self.order_book.bids.items())[:5]
                ],
                "order_book_asks": [
                    [p, s] for p, s in list(self.order_book.asks.items())[:5]
                ],
                "risk_budget_usd": self.risk_budget_usd,
                "risk_cap_qty": self.last_risk_cap_qty,
                "risk_requested_qty": self.last_base_qty,
                "risk_final_qty": self.last_final_qty,
                "vol_cap": self.vol_cap,
                "vol_reference": self.vol_ref,
            }
            with open(self.hud_data_dir / "risk_metrics.json", "w", encoding="utf-8") as f:
                json.dump(risk_metrics, f, indent=2)

        except Exception as e:
            LOG.error("[HUD] Failed to export data: %s", str(e))

    def _compute_order_qty(self, abs_delta: int, size_multiplier: float, is_new_entry: bool) -> float:
        base_qty = abs_delta * self.qty * size_multiplier
        self.last_base_qty = base_qty
        self.last_final_qty = base_qty
        self.last_risk_cap_qty = 0.0
        if abs_delta == 0 or base_qty <= 0:
            self.last_final_qty = 0.0
            return 0.0
        if not is_new_entry or self.risk_budget_usd <= 0:
            return base_qty
        var_value = max(self.last_estimated_var, 0.0)
        if var_value <= 0:
            return base_qty
        equity = self._estimate_account_equity()
        risk_cap = position_size_from_var(
            var=max(var_value, 1e-6),
            risk_budget_usd=self.risk_budget_usd,
            account_equity=equity,
            contract_size=max(self.contract_size, 1.0),
            max_leverage=max(self.max_leverage, 1.0),
        )
        if risk_cap <= 0:
            return base_qty
        self.last_risk_cap_qty = risk_cap
        adjusted = min(base_qty, risk_cap)
        if adjusted < base_qty:
            LOG.warning(
                "[RISK] Order size capped by risk budget: requested=%.4f capped=%.4f (VaR=%.4f, budget=$%.2f)",
                base_qty,
                adjusted,
                var_value,
                self.risk_budget_usd,
            )
        self.last_final_qty = adjusted
        return adjusted

    def _estimate_account_equity(self) -> float:
        try:
            pnl = self.performance.total_pnl
        except AttributeError:
            pnl = 0.0
        return float(max(self.starting_equity + pnl, 1.0))

    def _resolve_param(self, env_key: str | None, param_name: str, default: float) -> float:
        if env_key:
            env_val = os.environ.get(env_key)
            if env_val not in (None, ""):
                try:
                    return float(env_val or default)
                except ValueError:
                    LOG.warning("[PARAM] Invalid %s=%s - using fallback", env_key, env_val)
        return self._lp_get(param_name, default)

    def _lp_get(self, name: str, default: float) -> float:
        try:
            value = self.param_manager.get(
                self.symbol,
                name,
                timeframe=self.timeframe_label,
                broker=self.broker,
                default=default,
            )
            return float(value)
        except Exception as exc:
            LOG.debug("[PARAM] Falling back to %.4f for %s (%s)", default, name, exc)
            return float(default)

    def _atomic_save_trade(self, trade_record: dict):
        """
        Atomically save trade data with transactional guarantees.

        GAP 7.2 FIX: Implements transactional trade save with backup fallback.
        If any part of the save fails, data is preserved in backup location.

        Args:
            trade_record: Dict containing all trade data (path, performance, experience)
        """
        backup_dir = Path("data") / "trade_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        trade_id = trade_record.get("trade_id", int(time.time()))
        backup_file = backup_dir / f"trade_{trade_id}_backup.json"
        primary_file = Path("data") / "trade_log.jsonl"

        try:
            # First write to backup location
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(trade_record, f, indent=2, default=str)

            # Append to primary trade log (JSON Lines format)
            with open(primary_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade_record, default=str) + "\n")

            # Primary save succeeded - can remove backup
            if backup_file.exists():
                backup_file.unlink()

            LOG.debug("[SAVE] Trade %d saved successfully", trade_id)

        except Exception as e:
            LOG.error("[SAVE] Failed to save trade %d: %s", trade_id, e)
            LOG.warning("[SAVE] Trade data preserved in backup: %s", backup_file)
            # Backup file persists for manual recovery
            raise

    def _mark_component_error(self, component: str, error: Exception):
        """
        GAP 9.3 FIX: Track component errors for graceful degradation.

        Args:
            component: Component name (e.g., "policy", "mfe_tracker")
            error: Exception that occurred
        """
        if component not in self.component_error_counts:
            return

        self.component_error_counts[component] += 1

        if self.component_error_counts[component] >= self.max_component_errors:
            self.components_healthy[component] = False

            # GAP 10.1: Log component health change
            self.transaction_log.log_component_health(
                component=component, healthy=False, error_count=self.component_error_counts[component]
            )

            LOG.error(
                "[DEGRADED] Component '%s' marked unhealthy after %d errors: %s",
                component,
                self.component_error_counts[component],
                error,
            )
        else:
            LOG.warning(
                "[HEALTH] Component '%s' error %d/%d: %s",
                component,
                self.component_error_counts[component],
                self.max_component_errors,
                error,
            )

    def _mark_component_healthy(self, component: str):
        """Reset component to healthy state after successful operation."""
        if component in self.component_error_counts:
            self.component_error_counts[component] = 0
            if not self.components_healthy.get(component, True):
                self.components_healthy[component] = True

                # GAP 10.1: Log component health restoration
                self.transaction_log.log_component_health(component=component, healthy=True, error_count=0)

                LOG.info("[HEALTH] Component '%s' restored to healthy", component)

    def _is_component_healthy(self, component: str) -> bool:
        """Check if component is healthy."""
        return self.components_healthy.get(component, True)

    def _can_trade(self) -> bool:
        """
        GAP 9.3 FIX: Determine if bot can trade based on component health.

        Returns:
            True if critical components are healthy, False otherwise
        """
        critical_components = ["quote_feed", "trade_session", "trademanager", "circuit_breakers"]

        for comp in critical_components:
            if not self.components_healthy.get(comp, True):
                LOG.warning("[DEGRADED] Trading disabled - critical component unhealthy: %s", comp)
                return False

        return True

    def _current_var_regime(self) -> RegimeType:
        """Map DualPolicy regime states to VaR estimator enums."""
        regime_name = getattr(self.policy, "current_regime", "UNKNOWN") if hasattr(self, "policy") else "UNKNOWN"
        if regime_name == "TRENDING":
            return RegimeType.UNDERDAMPED
        if regime_name == "MEAN_REVERTING":
            return RegimeType.OVERDAMPED
        return RegimeType.CRITICAL


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
        # SYMBOL_ID: Can be 0 or "AUTO" for automatic lookup via SecurityListRequest
        symbol_id_raw = os.environ.get("CTRADER_SYMBOL_ID", "41").strip().upper()
        if symbol_id_raw in ("AUTO", "0", ""):
            symbol_id = 0  # Triggers automatic lookup via SecurityList
            LOG.info("[CONFIG] SYMBOL_ID=AUTO - will resolve from SecurityList")
        else:
            symbol_id = int(symbol_id_raw)  # Default: XAUUSD=41
        symbol = os.environ.get("CTRADER_SYMBOL", "XAUUSD")  # Instrument-agnostic
        qty = float(os.environ.get("CTRADER_QTY", "0.01"))  # Default: 0.01 lot for Gold
        timeframe_minutes = int(os.environ.get("CTRADER_TIMEFRAME_MIN", "1"))  # M1 for testing
    except (ValueError, TypeError) as e:
        LOG.error("Invalid configuration value: %s", e)
        raise SystemExit(1) from e
    # CRITICAL: Validate configuration to prevent financial losses
    if qty <= 0:
        LOG.error("CTRADER_QTY must be positive, got: %s", qty)
        raise SystemExit(1)
    if qty > MAX_QTY_SANITY_CHECK:  # Sanity check: prevent absurdly large orders
        LOG.error("CTRADER_QTY suspiciously large (>100), got: %s. Aborting for safety.", qty)
        raise SystemExit(1)
    if timeframe_minutes <= 0:
        LOG.error("CTRADER_TIMEFRAME_MIN must be positive, got: %s", timeframe_minutes)
        raise SystemExit(1)
    if symbol_id < 0:
        LOG.error("CTRADER_SYMBOL_ID must be non-negative (0=AUTO), got: %s", symbol_id)
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

    LOG.info(
        "✓ Configuration validated: symbol=%s symbol_id=%s qty=%s timeframe=M%d",
        symbol,
        "AUTO" if symbol_id == 0 else symbol_id,
        qty,
        timeframe_minutes,
    )
    LOG.info("cfg_quote=%s", cfg_quote)
    LOG.info("cfg_trade=%s", cfg_trade)
    LOG.info("CTRADER_USERNAME=%s", user)

    # ── Startup self-test ─────────────────────────────────────────────────────
    # Runs before any FIX session is created.  CRITICAL failures abort cleanly;
    # WARNINGs are logged and the bot continues in degraded/cold-start mode.
    run_self_test()
    # ────────────────────────────────────────────────────────────────────────────

    # Persist lightweight runtime profile for control panel / HUD
    try:
        status_dir = Path("data")
        status_dir.mkdir(exist_ok=True)
        profile = {
            "symbol": symbol,
            "symbol_id": symbol_id,
            "timeframe_minutes": timeframe_minutes,
            "quantity": qty,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        with open(status_dir / "current_profile.json", "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2)
        LOG.info("[PROFILE] Wrote current_profile.json for dashboard consumers")
    except Exception as e:
        LOG.warning("[PROFILE] Failed to write status file: %s", e)

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
                    LOG.warning(
                        "[MAIN] Connection unhealthy (consecutive failures: %d/%d) - %s",
                        consecutive_failures,
                        max_consecutive_failures,
                        status,
                    )
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
