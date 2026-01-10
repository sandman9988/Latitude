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

import numpy as np
import quickfix as fix
import quickfix44 as fix44

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from activity_monitor import ActivityMonitor
from adaptive_regularization import AdaptiveRegularization
from circuit_breakers import CircuitBreakerManager
from dual_policy import DualPolicy
from event_time_features import EventTimeFeatureEngine
from friction_costs import FrictionCalculator
from learned_parameters import LearnedParametersManager
from non_repaint_guards import NonRepaintBarAccess
from order_book import OrderBook, VPINCalculator
from path_geometry import PathGeometry
from performance_tracker import PerformanceTracker
from reward_shaper import RewardShaper
from trade_manager_example import TradeManagerIntegration
from ring_buffer import RollingStats

# Handbook components - Phase 1
from safe_math import SafeMath
from trade_exporter import TradeExporter
from var_estimator import KurtosisMonitor, RegimeType, VaREstimator, position_size_from_var

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

# Position and trading constants
MIN_BARS_FOR_VAR_UPDATE: int = 2
MIN_BARS_FOR_PREV_CLOSE: int = 2
MIN_TRADE_HISTORY_EXPLORATION: int = 20

# Action constants for policy decisions
ACTION_NO_ENTRY: int = 0
ACTION_LONG: int = 1
ACTION_SHORT: int = 2


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

    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

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
        self.bucket: dt.datetime | None = None
        self.o: float | None = None
        self.h: float | None = None
        self.l: float | None = None
        self.c: float | None = None

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
        if self.h is None or mid > self.h:
            self.h = mid
        if self.l is None or mid < self.l:
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
                assert nn is not None, "nn module should be available"
                assert torch is not None, "torch module should be available"

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
        assert self.torch is not None, "torch should be available when use_torch=True"
        assert self.model is not None, "model should be loaded when use_torch=True"
        with self.torch.no_grad():
            t = self.torch.from_numpy(x).unsqueeze(0)
            q = self.model(t).squeeze(0).numpy()
            return int(q.argmax())

    def update_ensemble_weights(self, disagreement: float, reward: float = 0.0) -> float:
        """Lightweight placeholder for Phase 2 ensemble tracking hooks."""
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
        pnl: float,
        mfe: float = 0.0,
        mae: float = 0.0,
        winner_to_loser: bool = False,
    ) -> dict:
        """Stop recording and return trade summary with path."""
        if not self.recording:
            return {}

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
    def __init__(self, symbol_id: int, qty: float, timeframe_minutes: int = 15, symbol: str = "BTCUSD"):
        super().__init__()

        self.symbol = symbol  # Instrument-agnostic: BTCUSD, XAUUSD, etc.
        self.symbol_id = symbol_id  # Numeric symbol identifier for FIX messages

        # Initialize learned parameters manager (single source of truth)
        self.param_manager = LearnedParametersManager()
        self.param_manager.load()

        self.timeframe_minutes = timeframe_minutes
        self.timeframe_label = f"M{timeframe_minutes}"
        self.broker = "default"

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
        LOG.info("Position size: %.4f (adaptive from LearnedParametersManager)", self.qty)
        LOG.info(
            "[PARAM] base=%.4f risk_budget=$%.2f vol_ref=%.4f vol_cap=%.4f vpin_z=%.2f",
            self.qty,
            self.risk_budget_usd,
            self.vol_ref,
            self.vol_cap,
            self.vpin_z_threshold,
        )
        LOG.info("[VPIN] bucket_volume=%.2f", self.vpin_bucket_volume)

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
                enable_training=enable_online_learning,
                param_manager=self.param_manager,
                symbol=symbol,
                timeframe=self.timeframe_label,
                broker="default",
            )
            LOG.info(
                "[POLICY] Using DualPolicy with TriggerAgent + HarvesterAgent + PathGeometry (training=%s)",
                enable_online_learning,
            )
        else:
            self.policy = Policy()  # type: ignore[assignment]
            LOG.info("[POLICY] Using simple Policy (DDQN_DUAL_AGENT=0)")

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
        self.performance = PerformanceTracker()
        self.trade_exporter = TradeExporter(output_dir="trades")  # Save to trades/ directory
        self.last_export_count = 0  # Track last export to avoid duplicates
        self.bar_count = 0  # Track total bars processed
        self.last_autosave_bar = 0  # Track last auto-save bar count
        self.bar_count = 0  # Track bars for periodic auto-save
        self.last_autosave_bar = 0  # Last bar when auto-save occurred

        # Friction calculator (source of truth: cTrader SymbolInfo)
        self.friction_calculator = FrictionCalculator(
            symbol=symbol,
            symbol_id=symbol_id,
            timeframe=self.timeframe_label,
            broker="default",
            param_manager=self.param_manager,
        )
        if os.environ.get("CTRADER_CONTRACT_SIZE") is None:
            self.contract_size = max(self.friction_calculator.costs.contract_size or 1.0, 1.0)
        LOG.info(
            "[PARAM] depth_levels=%d depth_buffer=%.2f spread_relax=%.2f vpin_z=%.2f vol_ref=%.4f vol_cap=%.4f risk_budget_usd=%.2f",
            self.friction_calculator.depth_levels,
            self.friction_calculator.depth_buffer,
            self.friction_calculator.spread_multiplier,
            self.vpin_z_threshold,
            self.vol_ref,
            self.vol_cap,
            self.risk_budget_usd,
        )

        # Phase 3.5: Activity monitor - prevent learned helplessness
        self.activity_monitor = ActivityMonitor(max_bars_inactive=100, min_trades_per_day=2.0, exploration_boost=0.1)
        LOG.info("[ACTIVITY] ActivityMonitor initialized - anti-stagnation protection")

        # Pass param_manager to reward_shaper for DRY and align instrumentation context
        self.reward_shaper = RewardShaper(
            symbol=symbol,
            timeframe=self.timeframe_label,
            broker="default",
            param_manager=self.param_manager,
            activity_monitor=self.activity_monitor,
        )

        # Phase 3.5: Risk management - VaR estimator with kurtosis circuit breaker
        self.kurtosis_monitor = KurtosisMonitor(window=100, threshold=3.0)
        self.var_estimator = VaREstimator(window=500, confidence=0.95, kurtosis_monitor=self.kurtosis_monitor)
        self.var_estimator.set_reference_vol(self.vol_ref)
        LOG.info("[RISK] VaREstimator initialized with kurtosis circuit breaker")

        # Phase 3.5: Adaptive regularization for online learning
        self.adaptive_reg = AdaptiveRegularization(
            initial_l2=0.0001, initial_dropout=0.1, l2_range=(1e-5, 1e-2), dropout_range=(0.0, 0.5)
        )
        LOG.info("[ADAPTIVE-REG] Initialized for online learning")

        # Handbook Phase 1: Circuit Breakers (safety shutdown system)
        self.circuit_breakers = CircuitBreakerManager(
            symbol=symbol,
            timeframe=self.timeframe_label,
            broker="default",
            param_manager=self.param_manager,
        )
        LOG.info(
            "[CIRCUIT-BREAKERS] Safety shutdown system initialized (Sortino>=%.2f, Kurtosis<=%.1f, DD<=%.0f%%, MaxLoss=%d)",
            self.circuit_breakers.sortino_breaker.threshold,
            self.circuit_breakers.kurtosis_breaker.threshold,
            self.circuit_breakers.max_drawdown * 100,
            self.circuit_breakers.max_consecutive_losses,
        )

        # Handbook Phase 1: Event-relative time features
        self.event_time_engine = EventTimeFeatureEngine()
        LOG.info("[EVENT-TIME] Session-relative time features initialized")

        # Training frequency: train every N bars
        self.training_interval = 5  # bars
        self.bars_since_training = 0

        self.previous_sharpe = 0.0  # Track for adaptive weight updates

        depth_env = os.environ.get("CTRADER_ORDERBOOK_DEPTH", "").strip()
        try:
            learned_depth = max(1, int(round(self.friction_calculator.depth_levels)))
            order_book_depth = int(depth_env) if depth_env else learned_depth
        except ValueError:
            LOG.warning("[ORDERBOOK] Invalid CTRADER_ORDERBOOK_DEPTH=%s, defaulting to 10", depth_env)
            order_book_depth = 10
        self.order_book = OrderBook(depth=order_book_depth)
        LOG.info("[ORDERBOOK] L2 book initialized (depth=%d)", order_book_depth)
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

        # Online learning state storage
        self.entry_state = None  # State at entry decision time (for experience)
        self.entry_action = None  # Action taken at entry (0=NO, 1=LONG, 2=SHORT)

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
            self.watchdog_interval = int(os.environ.get("WATCHDOG_USEC", "0")) / 1000000 / 2  # Half of watchdog timeout
            LOG.info("[WATCHDOG] Systemd watchdog enabled (interval: %.1fs)", self.watchdog_interval)
        else:
            self.watchdog_interval = None

        # Start health monitor thread
        self._health_monitor_running = True
        self._health_thread = threading.Thread(target=self._monitor_connection_health, daemon=True)
        self._health_thread.start()
        LOG.info("[HEALTH] Connection health monitor thread started")

        # TradeManager integration - centralized order & position management
        self.trade_integration = TradeManagerIntegration(self)
        LOG.info("[INTEGRATION] TradeManager integration initialized")

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

        # HUD data export tracking
        self.start_time = dt.datetime.now(dt.UTC)
        self.bar_count = 0
        self.hud_data_dir = Path("data")
        self.hud_data_dir.mkdir(exist_ok=True)
        # Seed HUD files immediately so the dashboard reflects the active symbol/timeframe
        try:
            self._export_hud_data()
        except Exception as hud_init_err:
            LOG.warning("[HUD] Initial export failed: %s", hud_init_err)

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
        LOG.info("[SHUTDOWN] Graceful shutdown initiated (signal: %s)", signal_name)
        self._shutdown_requested = True

        try:
            # 1. Close any open positions (emergency exit)
            if hasattr(self, "cur_pos") and self.cur_pos != 0:
                LOG.warning(
                    "[SHUTDOWN] Open position detected: %s (qty: %s)",
                    "LONG" if self.cur_pos > 0 else "SHORT",
                    abs(self.cur_pos),
                )
                LOG.warning("[SHUTDOWN] Manual position closure required - bot stopping with open position")
                # For safety, we DON'T auto-close to avoid accidental liquidation
            elif hasattr(self, "cur_pos"):
                LOG.info("[SHUTDOWN] No open positions")

            # 2. EMERGENCY EXPORT ALL TRADES
            if hasattr(self, "trade_exporter") and hasattr(self, "performance"):
                LOG.info("[SHUTDOWN] Exporting all trades (emergency save)...")
                try:
                    trades = self.performance.get_trade_history()
                    if trades:
                        timestamp = datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
                        files = self.trade_exporter.export_all(self.performance, prefix=f"shutdown_{timestamp}")
                        LOG.info(
                            "[SHUTDOWN] ✓ Exported %d trades to: %s",
                            len(trades),
                            ", ".join(files.values()),
                        )
                    else:
                        LOG.info("[SHUTDOWN] No trades to export")
                except Exception as e:
                    LOG.error("[SHUTDOWN] ✗ Failed to export trades: %s", e, exc_info=True)

            # 3. Save model state
            if hasattr(self, "policy") and self.policy:
                LOG.info("[SHUTDOWN] Saving model state...")
                try:
                    if hasattr(self.policy, "save_checkpoint"):
                        self.policy.save_checkpoint()
                        LOG.info("[SHUTDOWN] Model state saved")
                except Exception as e:
                    LOG.error("[SHUTDOWN] Failed to save model: %s", e)

            # 3. Flush trade exports
            if hasattr(self, "trade_exporter"):
                try:
                    LOG.info("[SHUTDOWN] Trade exports up to date")
                except Exception as e:
                    LOG.error("[SHUTDOWN] Error with exports: %s", e)

            # 4. Log final statistics
            LOG.info("[SHUTDOWN] Final statistics:")
            LOG.info("  - Total bars processed: %d", len(self.bars) if hasattr(self, "bars") else 0)
            LOG.info("  - Current position: %s", self.cur_pos if hasattr(self, "cur_pos") else 0)
            LOG.info("  - Reconnection attempts: %d", self.reconnect_attempts)
            if hasattr(self, "performance"):
                try:
                    metrics = self.performance.get_metrics()
                    LOG.info("  - Total PnL: %.2f", metrics.get("total_pnl", 0.0))
                    LOG.info("  - Total trades: %d", metrics.get("total_trades", 0))
                except Exception:
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
                self.send_md_subscribe_spot()
            elif qual == "TRADE":
                self.trade_sid = session_id
                self.last_trade_heartbeat = utc_now()
                self.request_security_definition()  # Get symbol info (source of truth)
                # Initialize TradeManager now that TRADE session is connected
                # TradeManager will handle position request
                if not self.trade_integration.initialize_trade_manager():
                    LOG.error("[INTEGRATION] TradeManager initialization failed - trading disabled")
                    return
            else:
                LOG.warning("[LOGON] Unknown qualifier; not routing: %s", session_id.toString())
        except Exception as e:
            LOG.error("[LOGON] Error during session setup for %s: %s", qual, e, exc_info=True)

    def onLogout(self, session_id):
        qual = self._qual(session_id)

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
        for i in range(1, n + 1):
            g = fix44.MarketDataSnapshotFullRefresh().NoMDEntries()
            msg.getGroup(i, g)

            et = fix.MDEntryType()
            px = fix.MDEntryPx()
            qty = fix.MDEntrySize()
            if not (g.isSetField(et) and g.isSetField(px)):
                continue

            g.getField(et)
            g.getField(px)

            try:
                price_f = float(px.getValue())
                size = 0.0
                if g.isSetField(qty):
                    g.getField(qty)
                    size = float(qty.getValue())
                if size <= 0:
                    size = 1.0  # cTrader omits size at top-of-book
            except (ValueError, TypeError) as e:
                LOG.warning("[QUOTE] Invalid entry %d: %s", i, e)
                continue

            entry_type = et.getValue()
            if entry_type == "0":
                self.order_book.update_level("BID", price_f, size)
            elif entry_type == "1":
                self.order_book.update_level("ASK", price_f, size)

        best_bid, best_ask = self.order_book.best_bid_ask()
        if best_bid is not None:
            self.best_bid = best_bid
        if best_ask is not None:
            self.best_ask = best_ask

        if best_bid is not None and best_ask is not None:
            self.friction_calculator.update_spread(best_bid, best_ask)

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
            px = fix.MDEntryPx()
            qty = fix.MDEntrySize()

            if g.isSetField(et):
                g.getField(et)
            if g.isSetField(sym):
                g.getField(sym)

            if sym.getValue() and sym.getValue() != str(self.symbol_id):
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

                if entry_type == "0":
                    self.order_book.update_level("BID", price_f, size)
                elif entry_type == "1":
                    self.order_book.update_level("ASK", price_f, size)
            elif action == "2":  # delete
                if entry_type == "0":
                    self.order_book.update_level("BID", 0.0, 0.0)
                elif entry_type == "1":
                    self.order_book.update_level("ASK", 0.0, 0.0)

        best_bid, best_ask = self.order_book.best_bid_ask()
        if best_bid is not None and best_ask is not None:
            self.best_bid, self.best_ask = best_bid, best_ask
            self.friction_calculator.update_spread(best_bid, best_ask)

        self.try_bar_update()

    def _update_non_repaint_series(self, bar, tick_count: int):
        """Append closed bar data into the non-repaint guard series."""
        _, o, h, low_price, c = bar
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
        if self.best_bid is None or self.best_ask is None:
            LOG.debug("[BAR] Skipping bar update: best_bid or best_ask is None")
            return

        mid = (self.best_bid + self.best_ask) / 2.0
        self.current_bar_tick_count += 1
        self._update_vpin(mid)

        # Update MFE/MAE if we have an open position
        if self.cur_pos != 0:
            self.mfe_mae_tracker.update(mid)

        closed = self.builder.update(utc_now(), mid)
        if closed:
            LOG.info(f"[BAR] Closed bar: {closed}")
            tick_count = self.current_bar_tick_count
            self.current_bar_tick_count = 0
            self._update_non_repaint_series(closed, tick_count)
            self._mark_non_repaint_closed()
            self.close_stats.update(closed[4])
            self.bars.append(closed)
            LOG.info(f"[BAR] Appended to self.bars (len now {len(self.bars)})")
            self.on_bar_close(closed)
            self._mark_non_repaint_opened()

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
        # Route to TradeManager first
        self.trade_integration.handle_position_report(msg)

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

        # FIX P0-2: Position is now tracked by TradeManager (single source of truth)
        # This code path kept for backward compatibility logging
        old_pos = self.cur_pos  # Gets from TradeManager via property

        LOG.info("[TRADE] PositionReport net=%0.6f -> cur_pos=%s (via TradeManager)", net, self.cur_pos)

        # Log MFE/MAE summary and save path when position closes
        if old_pos != 0 and self.cur_pos == 0:
            summary = self.mfe_mae_tracker.get_summary()
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
                exit_price = (self.best_bid + self.best_ask) / 2.0
                exit_time = utc_now()
                direction_sign = 1 if summary["direction"] == "LONG" else -1
                pnl = (exit_price - summary["entry_price"]) * direction_sign

                # Save path with MFE/MAE/WTL metrics
                self.path_recorder.stop_recording(
                    exit_time,
                    exit_price,
                    pnl,
                    mfe=summary["mfe"],
                    mae=summary["mae"],
                    winner_to_loser=summary["winner_to_loser"],
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
                        winner_to_loser=summary["winner_to_loser"],
                    )

                    # Calculate shaped rewards for DDQN training
                    reward_data = {
                        "exit_pnl": pnl,
                        "mfe": summary["mfe"],
                        "mae": summary["mae"],
                        "winner_to_loser": summary["winner_to_loser"],
                    }
                    shaped_rewards = self.reward_shaper.calculate_total_reward(reward_data)

                    # Update baseline MFE for normalization
                    if summary["mfe"] > 0:
                        self.reward_shaper.update_baseline_mfe(summary["mfe"])

                    # Log shaped reward components
                    LOG.info(
                        "[REWARD] Capture: %+.4f | WTL Penalty: %+.4f | Opportunity: %+.4f | Total: %+.4f | Active: %d",
                        shaped_rewards["capture_efficiency"],
                        shaped_rewards["wtl_penalty"],
                        shaped_rewards["opportunity_cost"],
                        shaped_rewards["total_reward"],
                        shaped_rewards["components_active"],
                    )

                    # Phase 3.5: Add experience to buffer for online learning
                    if hasattr(self.policy, "add_trigger_experience") and self.entry_state is not None:
                        # Build next_state from current bars
                        next_state = None
                        if hasattr(self.policy, "trigger") and hasattr(self.policy.trigger, "last_state"):
                            next_state = self.policy.trigger.last_state

                        if next_state is not None:
                            self.policy.add_trigger_experience(
                                state=self.entry_state,
                                action=self.entry_action,
                                reward=shaped_rewards["total_reward"],
                                next_state=next_state,
                                done=True,
                            )
                            LOG.info(
                                "[ONLINE_LEARNING] Added TriggerAgent experience: action=%d reward=%.4f",
                                self.entry_action,
                                shaped_rewards["total_reward"],
                            )

                        # Reset entry state
                        self.entry_state = None

                    # Add final harvester experience (CLOSE decision with capture reward)
                    if hasattr(self.policy, "add_harvester_experience") and self.prev_harvester_state is not None:
                        capture_reward = shaped_rewards.get("capture_efficiency", 0.0)
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
                            LOG.info("[ONLINE_LEARNING] Added HarvesterAgent CLOSE: reward=%.4f", capture_reward)

                        # Reset harvester tracking
                        self.prev_harvester_state = None
                        self.prev_exit_action = None
                        self.prev_mfe = 0.0
                        self.prev_mae = 0.0

                    # Handbook: Update circuit breakers with trade result
                    current_equity = self.performance.get_total_pnl() + 10000.0  # Assuming 10K starting equity
                    self.circuit_breakers.update_trade(pnl, current_equity)

                    # Check and log circuit breaker status
                    breaker_status = self.circuit_breakers.get_status()
                    if breaker_status["any_tripped"]:
                        LOG.warning("[CIRCUIT-BREAKER] Status after trade: %s", breaker_status)

                    # Auto-reset breakers after cooldown
                    self.circuit_breakers.reset_if_cooldown_elapsed()
                    self.entry_action = None

                    # Adapt reward weights based on performance improvement
                    metrics = self.performance.get_metrics()
                    current_sharpe = metrics["sharpe_ratio"]
                    sharpe_delta = current_sharpe - self.previous_sharpe

                    if (
                        self.performance.total_trades % PERFORMANCE_INTERVAL_TRADES == 0
                        and self.performance.total_trades > MIN_TRADES_FOR_ADAPTATION
                    ):
                        self.reward_shaper.adapt_weights(sharpe_delta)
                        LOG.info("[REWARD] Adapted weights based on Sharpe delta: %+.4f", sharpe_delta)
                        LOG.info(self.reward_shaper.print_summary())

                    self.previous_sharpe = current_sharpe

                    # Log performance dashboard every 5 trades
                    if self.performance.total_trades % PERFORMANCE_INTERVAL_TRADES == 0:
                        LOG.info("\n" + self.performance.print_dashboard())
                    else:
                        metrics = self.performance.get_metrics()
                        LOG.info(
                            "[PERF] Trades: %d | Win Rate: %.1f%% | Total PnL: $%.2f | Sharpe: %.3f | Max DD: %.1f%%",
                            metrics["total_trades"],
                            metrics["win_rate"] * 100,
                            metrics["total_pnl"],
                            metrics["sharpe_ratio"],
                            metrics["max_drawdown"] * 100,
                        )

                    # Auto-export CSV every 5 trades (more frequent for safety)
                    if (
                        self.performance.total_trades % PERFORMANCE_INTERVAL_TRADES == 0
                        and self.performance.total_trades != self.last_export_count
                    ):
                        try:
                            datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
                            files = self.trade_exporter.export_all(
                                self.performance, prefix=f"live_t{self.performance.total_trades}"
                            )
                            self.last_export_count = self.performance.total_trades
                            LOG.info(
                                "[EXPORT] ✓ Saved %d trades to: %s",
                                self.performance.total_trades,
                                ", ".join(files.values()),
                            )
                        except Exception as e:
                            LOG.error("[EXPORT] ✗ Failed to export CSV: %s", e, exc_info=True)

            self.mfe_mae_tracker.reset()
            self.trade_entry_time = None

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
            if msg.isSetField(txt):
                msg.getField(txt)
                LOG.warning("[TRADE] Order rejected: %s", txt.getValue())

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

        LOG.info(f"[BAR] on_bar_close called: t={t}, o={o}, h={h}, l={low_price}, c={c}")

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

        # Phase 3.5: Update VaR with bar return
        if len(self.bars) >= MIN_BARS_FOR_VAR_UPDATE:
            prev_close = self.bars[-2][4] if len(self.bars) >= MIN_BARS_FOR_PREV_CLOSE else c
            bar_return = SafeMath.safe_div(c - prev_close, prev_close, 0.0)
            self.var_estimator.update_return(bar_return)

        # Record bar if position is open
        if self.cur_pos != 0:
            self.path_recorder.add_bar(bar)

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
        (
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
            try:
                train_metrics = self.policy.train_step(self.adaptive_reg)
                self.bars_since_training = 0

                # Adjust regularization based on training metrics
                if train_metrics.get("trigger") or train_metrics.get("harvester"):
                    trigger_td = (
                        train_metrics.get("trigger", {}).get("mean_td_error", 0) if train_metrics.get("trigger") else 0
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
            except Exception as e:
                LOG.error(f"[TRAINING] Training step failed: {e}", exc_info=True)
                self.bars_since_training = 0  # Reset to prevent repeated attempts

        # Phase 3: Use DualPolicy if available
        if hasattr(self.policy, "decide_entry"):
            # Dual-agent architecture
            if self.cur_pos == 0:
                # Handbook: Check circuit breakers BEFORE any entry decision
                if self.circuit_breakers.is_any_tripped():
                    tripped = self.circuit_breakers.get_tripped_breakers()
                    LOG.warning("[CIRCUIT-BREAKER] Trading halted: %s", ", ".join([b.name for b in tripped]))
                    self._export_hud_data()
                    return

                # Log key time features
                if len(self.bars) % 10 == 0:  # Every 10 bars
                    active_sessions = self.event_time_engine.get_active_sessions()
                    LOG.debug(
                        "[EVENT-TIME] Sessions: %s | High liquidity: %s",
                        ",".join(active_sessions) if active_sessions else "None",
                        is_high_liq,
                    )
                depth_floor = max(self.last_depth_floor, 0.0)
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
                    self._export_hud_data()
                    return
                self.last_depth_gate = False

                # FLAT: Check for entry (pass event features if policy accepts them)
                action, confidence, runway = self.policy.decide_entry(
                    self.bars,
                    imbalance=imbalance,
                    vpin_z=vpin_zscore,
                    depth_ratio=depth_ratio,
                    realized_vol=realized_vol,
                    event_features=event_features,
                )
                # action: 0=NO_ENTRY, 1=LONG, 2=SHORT
                desired = 1 if action == ACTION_LONG else (-1 if action == ACTION_SHORT else 0)

                # Store state for online learning (if entry is taken)
                if action != 0 and hasattr(self.policy, "trigger") and hasattr(self.policy.trigger, "last_state"):
                    self.entry_state = (
                        self.policy.trigger.last_state.copy() if self.policy.trigger.last_state is not None else None
                    )
                    self.entry_action = action

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
                exit_action, exit_conf = self.policy.decide_exit(
                    self.bars,
                    current_price=c,
                    imbalance=imbalance,
                    vpin_z=vpin_zscore,
                    depth_ratio=depth_ratio,
                    event_features=event_features,
                )
                # exit_action: 0=HOLD, 1=CLOSE
                desired = 0 if exit_action == 1 else self.cur_pos

                # Update trailing stop (HarvesterAgent controls, TradeManager communicates)
                if hasattr(self, "trade_integration") and self.trade_integration.trailing_stop_active:
                    mid_price = (self.best_bid + self.best_ask) / 2.0 if self.best_bid and self.best_ask else c
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

                    # Incremental reward for previous HOLD
                    mfe_delta = current_mfe - self.prev_mfe
                    mae_delta = current_mae - self.prev_mae
                    reward = (0.1 if mfe_delta > 0 else 0.0) - (0.2 if mae_delta > 0 else 0.0) - 0.01

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
                        LOG.debug(
                            "[HARVESTER_EXP] HOLD reward=%.4f (MFE_Δ=%.2f MAE_Δ=%.2f)", reward, mfe_delta, mae_delta
                        )

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
            desired = -1 if action == 0 else (0 if action == 1 else 1)

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
                with open(log_path, "r", encoding="utf-8") as f:
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
            if len(log_entries) > 1000:
                log_entries = log_entries[-1000:]
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entries, f, indent=2)
            LOG.info(f"[DECISION_LOG] Wrote entry for {t} (total: {len(log_entries)})")
        except Exception as e:
            LOG.error(f"[DECISION_LOG] Failed to write: {e}", exc_info=True)

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
                LOG.warning(
                    "[CIRCUIT_BREAKER] Kurtosis breaker ACTIVE (kurtosis=%.2f) - skipping entry",
                    self.kurtosis_monitor.current_kurtosis,
                )
                self._export_hud_data()  # Export even on circuit breaker
                return

            # Calculate current VaR with regime-aware multiplier
            current_var = self.var_estimator.estimate_var(
                regime=self._current_var_regime(), vpin_z=vpin_zscore, current_vol=realized_vol
            )
            self.last_estimated_var = current_var
            max_var_threshold = self.vol_cap
            if current_var > max_var_threshold:
                LOG.warning(
                    "[CIRCUIT_BREAKER] VaR=%.4f exceeds threshold=%.4f - skipping entry",
                    current_var,
                    max_var_threshold,
                )
                self._export_hud_data()  # Export even on VaR filter
                return
            if self.vpin_z_threshold > 0 and vpin_zscore > self.vpin_z_threshold:
                LOG.warning(
                    "[VPIN] z-score %.2f exceeds threshold %.2f - skipping entry",
                    vpin_zscore,
                    self.vpin_z_threshold,
                )
                self._export_hud_data()
                return

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
                LOG.warning(
                    "[SPREAD_FILTER] Current=%.2f pips > Learned max=%.2f pips (%.1fx min) - skipping entry",
                    current_spread,
                    max_spread,
                    effective_multiplier,
                )
                self._export_hud_data()  # Export even on filtered entries
                return

        # Export HUD data every bar (before potential order)
        self._export_hud_data()

        delta = desired - self.cur_pos
        side = "1" if delta > 0 else "2"

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
            self._export_hud_data()
            return

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

        # Use TradeManager API for proper order lifecycle tracking
        from trade_manager import Side

        tm_side = Side.BUY if side == "1" else Side.SELL

        order = self.trade_integration.trade_manager.submit_market_order(side=tm_side, quantity=qty, tag_prefix="DDQN")

        if order:
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
            }
            with open(self.hud_data_dir / "bot_config.json", "w", encoding="utf-8") as f:
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
                bars_held = len(self.path_recorder.bars) if hasattr(self.path_recorder, "bars") else 0

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
            trigger_loss = 0.0
            harvester_loss = 0.0
            total_agents = 0
            arena_diversity = {"trigger_diversity": 0.0, "harvester_diversity": 0.0}
            last_agreement = 1.0

            if hasattr(self.policy, "trigger") and hasattr(self.policy.trigger, "replay_buffer"):
                trigger_buffer_size = (
                    len(self.policy.trigger.replay_buffer)
                    if hasattr(self.policy.trigger.replay_buffer, "__len__")
                    else 0
                )
            if hasattr(self.policy, "harvester") and hasattr(self.policy.harvester, "replay_buffer"):
                harvester_buffer_size = (
                    len(self.policy.harvester.replay_buffer)
                    if hasattr(self.policy.harvester.replay_buffer, "__len__")
                    else 0
                )

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

            # Regime from dual policy
            regime = "UNKNOWN"
            regime_zeta = 1.0
            if hasattr(self.policy, "current_regime"):
                regime = self.policy.current_regime
            if hasattr(self.policy, "current_zeta"):
                regime_zeta = self.policy.current_zeta

            # Path geometry features
            geom = self.path_geometry.last if hasattr(self.path_geometry, "last") else {}
            efficiency = geom.get("efficiency", 1.0)
            gamma = geom.get("gamma", 0.0)
            jerk = geom.get("jerk", 0.0)
            runway = geom.get("runway", 0.5)
            feasibility = geom.get("feasibility", 0.5)

            # Market microstructure
            spread = self.best_ask - self.best_bid if self.best_bid and self.best_ask else 0.0
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
        symbol_id = int(os.environ.get("CTRADER_SYMBOL_ID", "41"))  # Default: XAUUSD=41
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

    LOG.info(
        "✓ Configuration validated: symbol=%s symbol_id=%s qty=%s timeframe=M%d",
        symbol,
        symbol_id,
        qty,
        timeframe_minutes,
    )
    LOG.info("cfg_quote=%s", cfg_quote)
    LOG.info("cfg_trade=%s", cfg_trade)
    LOG.info("CTRADER_USERNAME=%s", user)

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
