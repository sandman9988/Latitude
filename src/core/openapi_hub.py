#!/usr/bin/env python3
"""openapi_hub.py - Multi-TF cTrader Open API paper trading hub.

One process per symbol. Single TCP connection to cTrader Open API.
N TFAgent instances (one per timeframe) share the connection and each
run an independent BarBuilder + DualPolicy paper-trading loop.

Compared to the FIX bot (ctrader_ddqn_paper.py) this hub:
- Uses one TCP connection instead of two FIX sessions per TF
- Handles any number of TFs without additional broker connections
- Simulates paper fills locally (no real orders sent)
- Writes the same telemetry files so the HUD works unchanged

Credentials (priority: env > config/cTraderAppTokens):
    CTRADER_CLIENT_ID       OAuth2 client ID
    CTRADER_CLIENT_SECRET   OAuth2 client secret
    CTRADER_ACCESS_TOKEN    OAuth2 access token
    CTRADER_ACCOUNT_ID      cTrader numeric account ID (numeric login)

Runtime config (env vars):
    OPENAPI_SYMBOL          Symbol name (e.g. XAUUSD)
    OPENAPI_SYMBOL_ID       Symbol ID (e.g. 41)
    OPENAPI_TIMEFRAMES      Comma-separated TF list (e.g. "5,15,60")
    OPENAPI_DIGITS          Price decimal digits override (default 5)
    OPENAPI_LIVE            "1" to use live server (default: demo)
    CTRADER_QTY             Position size per TF (default 0.5)
    CTRADER_DATA_DIR        Base data root (default "data")
    CTRADER_STARTING_EQUITY Starting paper equity (default 10000.0)
    CTRADER_CONTRACT_SIZE   Contract size override (default from symbol_specs.json)
    DDQN_ONLINE_LEARNING    Enable online DualPolicy training (default "1")

Usage:
    python3 -m src.core.openapi_hub
    # or via run_universe.py (launched automatically per symbol)
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import math
import os
import signal
import socket
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.safe_math import SAFE_DIV_MIN, SAFE_EPSILON, SAFE_SMALL, SafeMath
from src.core.tf_agent_preseed import TFAgentPreseedMixin
from src.core.tf_agent_capture_health import TFAgentCaptureHealthMixin
from src.core.tf_agent_trade_log import TFAgentTradeLogMixin
from src.core.tf_agent_telemetry import TFAgentTelemetryMixin

LOG = logging.getLogger("openapi_hub")

# ---------------------------------------------------------------------------
# Auth state labels
# ---------------------------------------------------------------------------
_S_CONNECTING = "connecting"
_S_APP_AUTH = "app_auth"
_S_ACC_AUTH = "acc_auth"
_S_SYM_INFO = "sym_info"
_S_SUBSCRIBING = "subscribing"
_S_READY = "ready"

_DEMO_HOST = "demo.ctraderapi.com"
_LIVE_HOST = "live.ctraderapi.com"
_PORT = 5035

_TRAIN_INTERVAL_BARS = 2
_TELEMETRY_INTERVAL_BARS = 5
_CHECKPOINT_INTERVAL_BARS = 100
_DEFAULT_DIGITS = 5
_DEFAULT_CONTRACT_SIZE = 100.0
_DEFAULT_STARTING_EQUITY = 10_000.0
_MIN_BARS_BEFORE_TRADE = 30
_ENTRY_STALE_DRIFT_VOL_MULT = 1.5
_ENTRY_STALE_MAX_AGE_TF_MULT = 3.0
_GAP_FILL_MIN_SECONDS = 60.0  # only fetch history if gap > this
_INITIAL_BACKFILL_BARS = 100  # bars to pre-fetch on cold start per TF
_CTRL_KILL_SWITCH = "kill_switch.json"
_CTRL_CB_RESET = "circuit_breaker_reset.json"
_CTRL_KG_RESET = "kurtosis_gate_reset.json"
_CTRL_EPSILON_OVERRIDE = "epsilon_override.json"
_CTRL_PARAM_RELOAD = "learned_parameters_reload.json"

# cTrader error codes that signal an expired or invalid access token.
# On any of these the hub sends ProtoOARefreshTokenReq (payload 2173) and
# replays the full auth sequence when the response (2174) arrives.
_AUTH_ERROR_CODES = frozenset({
    "OA_AUTH_TOKEN_EXPIRED",
    "CH_CLIENT_AUTH_FAILURE",
    "CH_ACCESS_TOKEN_INVALID",
    "ACCOUNT_NOT_AUTHORIZED",
    "ACCESS_TOKEN_INVALID",
    "INVALID_ACCESS_TOKEN",
})

# cTrader PERIOD enum → timeframe minutes
_PERIOD_TO_TF: dict[int, int] = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 10, 7: 15, 8: 30, 9: 60, 10: 240, 11: 720, 12: 1440,
}
_TF_TO_PERIOD: dict[int, int] = {v: k for k, v in _PERIOD_TO_TF.items()}


# ---------------------------------------------------------------------------
# Credentials loader
# ---------------------------------------------------------------------------

def _parse_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, _, v = s.partition("=")
            # strip optional `export ` prefix and surrounding quotes
            k = k.strip().removeprefix("export").strip()
            v = v.strip().strip('"').strip("'")
            if k:
                result[k] = v
    except OSError:
        pass
    return result


def _load_creds() -> dict[str, str]:
    """Load Open API credentials from env vars, local config, then Kinetra fallback.

    Resolution order (later wins):
    1. config/cTraderAppTokens (project-local)
    2. .env.openapi (project-local, cTrader-specific overrides)
    3. ../Kinetra/.env.openapi (shared cTrader credentials across projects)
    4. os.environ (highest priority)
    """
    merged: dict[str, str] = {}
    _root = Path(__file__).resolve().parent.parent.parent

    for p in [
        _root / "config" / "cTraderAppTokens",
        _root / ".env.openapi",
        _root.parent / "Kinetra" / ".env.openapi",
    ]:
        if p.exists():
            merged.update(_parse_env_file(p))

    def _get(key: str) -> str:
        return os.environ.get(key) or merged.get(key) or ""

    return {
        "client_id":     _get("CTRADER_CLIENT_ID"),
        "client_secret": _get("CTRADER_CLIENT_SECRET"),
        "access_token":  _get("CTRADER_ACCESS_TOKEN"),
        "refresh_token": _get("CTRADER_REFRESH_TOKEN"),
        "account_id":    _get("CTRADER_ACCOUNT_ID") or _get("CTRADER_USERNAME"),
    }


# ---------------------------------------------------------------------------
# Symbol specs loader (for contract size)
# ---------------------------------------------------------------------------

def _load_symbol_spec(symbol: str) -> dict:
    try:
        with open(Path("config/symbol_specs.json")) as f:
            specs = json.load(f)
        for k, v in specs.items():
            if k.upper() == symbol.upper() and isinstance(v, dict):
                return v
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Endpoint probe — picks first responsive host from primary + alt list
# ---------------------------------------------------------------------------

def _probe_endpoints(primary: str, port: int, alt_raw: str = "", timeout: float = 2.0) -> str:
    """TCP-probe primary then comma-separated alt endpoints; return first that connects."""
    candidates: list[str] = [primary]
    for raw_h in alt_raw.split(","):
        h = raw_h.strip()
        if h and h not in candidates:
            candidates.append(h)
    for host in candidates:
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            sock.close()
            if host != primary:
                LOG.info("[HUB] Alt endpoint selected: %s", host)
            return host
        except OSError:
            LOG.debug("[HUB] Endpoint unreachable: %s:%d", host, port)
    LOG.warning("[HUB] All endpoints unreachable — falling back to %s", primary)
    return primary


# ---------------------------------------------------------------------------
# JSON writers (shared with TFAgent mixins via persistence layer)
# ---------------------------------------------------------------------------

from src.persistence.json_io import (
    json_default as _json_default,
    save_json_atomic as _save_json_atomic,
    write_json_async as _write_json_async,
    write_json_atomic as _write_json_atomic,
)


# ---------------------------------------------------------------------------
# BarBuilder (copy from ctrader_ddqn_paper to avoid circular import)
# ---------------------------------------------------------------------------

class BarBuilder:
    def __init__(self, timeframe_minutes: int) -> None:
        self.timeframe_minutes = timeframe_minutes
        self.bucket: dt.datetime | None = None
        self.o = self.h = self.l = self.c = None

    def bucket_start(self, t: dt.datetime) -> dt.datetime:
        if self.timeframe_minutes < 60:
            m = (t.minute // self.timeframe_minutes) * self.timeframe_minutes
            return t.replace(minute=m, second=0, microsecond=0)
        # For TFs >= 60 min use total minutes-from-midnight so M240 gives
        # 00:00/04:00/08:00/12:00/16:00/20:00 buckets, not one-per-hour.
        total = t.hour * 60 + t.minute
        snapped = (total // self.timeframe_minutes) * self.timeframe_minutes
        return t.replace(hour=snapped // 60, minute=snapped % 60, second=0, microsecond=0)

    def update(self, t: dt.datetime, mid: float):
        if not mid or mid <= 0:
            return None
        b = self.bucket_start(t)
        if self.bucket is None:
            self.bucket, self.o, self.h, self.l, self.c = b, mid, mid, mid, mid
            return None
        if b != self.bucket:
            if None in (self.o, self.h, self.l, self.c):
                self.bucket, self.o, self.h, self.l, self.c = b, mid, mid, mid, mid
                return None
            closed = (self.bucket, self.o, self.h, self.l, self.c)
            self.bucket, self.o, self.h, self.l, self.c = b, mid, mid, mid, mid
            return closed
        self.c = mid
        self.h = max(self.h, mid)
        self.l = min(self.l, mid)
        return None

    def next_bar_close_utc(self) -> str | None:
        if self.bucket is None:
            return None
        return (self.bucket + dt.timedelta(minutes=self.timeframe_minutes)).isoformat()


# ---------------------------------------------------------------------------
# TFAgent – per-timeframe paper trading loop
# ---------------------------------------------------------------------------

class TFAgent(TFAgentPreseedMixin, TFAgentCaptureHealthMixin, TFAgentTradeLogMixin, TFAgentTelemetryMixin):
    """Manages one timeframe for a single symbol."""

    def __init__(
        self,
        symbol: str,
        symbol_id: int,
        timeframe_minutes: int,
        qty: float,
        contract_size: float,
        data_dir: Path,
        starting_equity: float,
        online_learning: bool,
    ) -> None:
        self._init_core_state(symbol, symbol_id, timeframe_minutes, qty, contract_size, data_dir, starting_equity)
        self._init_policy_stack(online_learning)
        self._init_loggers_and_cache(online_learning)
        self._init_risk_reward_stack(online_learning)
        self._init_runtime_tracking()

        # Warm-start: load persisted bars and preseed both replay buffers.
        # Runs after all state vars and policy are fully initialised.
        if online_learning:
            self._load_bars_cache()

        LOG.info("[%s %s] TFAgent initialized | qty=%.2f contract=%.0f equity=%.0f training=%s",
                 symbol, self.tf_label, qty, contract_size, starting_equity,
                 "ENABLED" if online_learning else "DISABLED")

    def _init_core_state(
        self,
        symbol: str,
        symbol_id: int,
        timeframe_minutes: int,
        qty: float,
        contract_size: float,
        data_dir: Path,
        starting_equity: float,
    ) -> None:
        self.symbol = symbol
        self.symbol_id = symbol_id
        self.timeframe_minutes = timeframe_minutes
        self.tf_label = f"M{timeframe_minutes}"
        self.qty = qty
        self.contract_size = contract_size
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.bar_builder = BarBuilder(timeframe_minutes)
        self.bars: deque = deque(maxlen=2000)
        self.bar_count = 0

        self.paper_mode = os.environ.get("PAPER_MODE", "1") == "1"

        # Paper position state
        self.equity = starting_equity
        self.starting_equity = starting_equity
        self.position: dict | None = None
        self.trades_pnl: list[float] = []
        self.total_trades = 0

        # Broker account state — populated from ProtoOATraderRes / ProtoOATraderUpdatedEvent
        self._broker_balance: float | None = None
        self._broker_equity: float | None = None
        self._broker_margin_free: float | None = None
        self._broker_money_digits: int = 2  # cTrader reports amounts × 10^moneyDigits

        # Last tick data for telemetry
        self.last_mid: float = 0.0
        self.last_half_spread: float = 0.0
        self.last_ts: dt.datetime | None = None

    def _init_policy_stack(self, online_learning: bool) -> None:
        # DualPolicy
        from src.agents.dual_policy import DualPolicy, DualPolicyConfig
        from src.constants import (
            HARVESTER_BUFFER_CAPACITY,
            TRIGGER_BUFFER_CAPACITY,
            get_amd_optimized_buffer_capacity,
        )
        from src.persistence.learned_parameters import LearnedParametersManager

        param_manager = LearnedParametersManager(
            persistence_path=self.data_dir / "learned_parameters.json",
        )
        param_manager.load()
        self._param_manager = param_manager

        from src.risk.path_geometry import PathGeometry
        self.path_geometry = PathGeometry()

        from src.risk.var_estimator import KurtosisMonitor, RegimeType, VaREstimator
        self._regime_type = RegimeType
        self.var_estimator = VaREstimator(
            window=500,
            confidence=0.95,
            kurtosis_monitor=KurtosisMonitor(window=100),
        )
        self.var_estimator.set_reference_vol(0.005)

        cfg = DualPolicyConfig(
            enable_regime_detection=True,
            enable_training=online_learning,
            symbol=self.symbol,
            timeframe=self.tf_label,
            broker="default",
            timeframe_minutes=self.timeframe_minutes,
            param_manager=param_manager,
            path_geometry=self.path_geometry,
            trigger_buffer_capacity=get_amd_optimized_buffer_capacity(TRIGGER_BUFFER_CAPACITY),
            harvester_buffer_capacity=get_amd_optimized_buffer_capacity(HARVESTER_BUFFER_CAPACITY),
        )
        self.policy = DualPolicy(**vars(cfg))
        if online_learning and hasattr(self.policy, "load_checkpoint"):
            try:
                self.policy.load_checkpoint()
            except Exception as e:
                LOG.warning("[%s %s] load_checkpoint failed: %s", self.symbol, self.tf_label, e)

    def _init_loggers_and_cache(self, online_learning: bool) -> None:
        # Decision audit log (HUD Tab 6 reads from logs/audit/decisions.jsonl)
        from src.monitoring.audit_logger import DecisionLogger, TransactionLogger
        self.decision_log = DecisionLogger(
            log_dir=str(self.data_dir / "logs" / "audit"),
            filename="decisions.jsonl",
            trading_mode="paper",
            symbol=self.symbol,
            timeframe=self.tf_label,
            timeframe_minutes=self.timeframe_minutes,
        )
        self.transaction_log = TransactionLogger(
            log_dir=str(self.data_dir / "logs" / "audit"),
            filename="transactions.jsonl",
        )

        # Training experience cache (writes training_cache_SYM_MTF.jsonl)
        from src.training.bar_experience_cache import BarExperienceCache
        self.bar_cache = BarExperienceCache(
            symbol=self.symbol,
            timeframe_minutes=self.timeframe_minutes,
            enabled=online_learning,
        )

        from src.features.event_time_features import EventTimeFeatureEngine
        self.event_time_engine = EventTimeFeatureEngine()

    def _init_risk_reward_stack(self, online_learning: bool) -> None:
        from src.risk.circuit_breakers import CircuitBreakerManager
        self.circuit_breakers = CircuitBreakerManager(
            symbol=self.symbol,
            timeframe=self.tf_label,
            broker="default",
            param_manager=self._param_manager,
            kurtosis_adaptive=True,
            auto_close_on_trip=True,
        ) if online_learning else None
        if self.circuit_breakers is not None:
            self.circuit_breakers.set_emergency_closer(_PaperEmergencyCloser(self))
            with contextlib.suppress(Exception):
                self.circuit_breakers.restore_state(str(self.data_dir / "circuit_breakers.json"))

        from src.risk.friction_costs import FrictionCalculator
        self.friction_calc = FrictionCalculator(
            symbol=self.symbol,
            symbol_id=self.symbol_id,
            timeframe=self.tf_label,
            broker="default",
            param_manager=self._param_manager,
        )

        # Shaped reward computation
        from src.core.reward_shaper import RewardShaper
        self.reward_shaper = RewardShaper(
            symbol=self.symbol,
            timeframe=self.tf_label,
            param_manager=self._param_manager,
        )

        # Adaptive regularization for DDQN training
        from src.core.adaptive_regularization import AdaptiveRegularization
        self.adaptive_reg = AdaptiveRegularization() if online_learning else None

        # Reward shaping monitor — runs every hour, updates learned_parameters.json
        from src.monitoring.reward_shaping_monitor import RewardShapingMonitor
        self.reward_shaping_monitor = RewardShapingMonitor(
            symbol=self.symbol,
            param_manager=self._param_manager,
            timeframe=self.tf_label,
            broker="default",
            decision_log_path=str(self.data_dir / "logs" / "audit" / "decisions.jsonl"),
        ) if online_learning else None

        from src.monitoring.production_monitor import ProductionMonitor
        self.prod_monitor = ProductionMonitor(
            metrics_file=self.data_dir / "production_metrics.json",
            http_enabled=False,
        )
        self._last_trade_close_ts: float | None = None

    def _init_runtime_tracking(self) -> None:
        self._init_entry_exit_snapshots()
        self._init_trade_sequence_state()
        self._init_latest_metric_state()
        self._init_adaptive_runtime_state()
        self._init_lifecycle_snapshots()
        self._init_harvester_hold_state()
        self.start_time = dt.datetime.now(dt.UTC)
        self._last_telemetry_time: float = 0.0  # wall-clock of last telemetry write
        self._bars_since_cache_save: int = 0

    def _init_entry_exit_snapshots(self) -> None:
        # Entry/exit state snapshots for experience replay
        self._entry_state: Any = None
        self._exit_state: Any = None
        self._entry_action: int = 0
        self._entry_conf: float = 0.5
        self._entry_raw_conf: float = 0.5
        self._entry_imbalance: float = 0.0
        self._last_depth_bid: float = 0.0
        self._last_depth_ask: float = 0.0
        self._vpin_z: float = 0.0
        self._has_real_sizes: bool = False
        self._last_l2_snapshot: dict = {}
        self._bars_since_train: int = 0

    def _init_trade_sequence_state(self) -> None:
        self._trade_sequence: int = 0  # local trade counter for trade_log ticket IDs
        self._trade_sequence_lock = threading.Lock()  # guard for concurrent TF bar closes
        self._epoch_ts: int = int(time.time())  # epoch at startup for ticket generation
        self._current_trade_id: str | None = None  # links entry → hold(s) → close in audit log

    def _init_latest_metric_state(self) -> None:
        # Last computed geometry and event features — updated each bar close
        self._last_event_feats: dict = {}
        self._last_var_95: float = 0.0
        self._last_kurtosis: float = 0.0

        from src.persistence.trade_log_reader import CachedTradeLogReader
        self._trade_log_reader = CachedTradeLogReader()

        # Confidence tracking (updated from decide_entry/decide_exit results)
        self._last_trigger_conf: float = 0.5
        self._last_harvester_conf: float = 0.5
        # Rolling MFE/MAE — timeframe-adaptive window (fewer bars needed for slow TFs)
        _mfe_window = max(10, 200 // max(1, self.timeframe_minutes))  # M1→200, M240→10
        self._rolling_mfe: deque = deque(maxlen=_mfe_window)
        self._rolling_mae: deque = deque(maxlen=_mfe_window)

        # Capture health monitoring — two-tier reactive system
        self._capture_ema: float = 0.5           # Rolling capture EMA (starts neutral)
        self._capture_ema_n: int = 0             # Trade count for sample gate
        self._capture_last_intervention: float = 0.0  # Wall-clock of last tighten/relax

    def _init_adaptive_runtime_state(self) -> None:
        # Runway prediction accuracy EMAs — persisted across restarts via param_manager.
        # delta_ema: signed EMA of (predicted_pts - actual_mfe); positive = over-predicted.
        # accuracy_ema: EMA of (1 - |delta|/max_err) in [0,1]; 1.0 = perfect.
        # conf_calib_err_ema: Brier score EMA of (confidence - outcome)^2.
        self._init_adaptive_emas()

        self._harvester_preseeded: bool = False

        # Cached training metrics — updated each train_step for HUD / telemetry
        self._last_trigger_grad_norm: float = 0.0
        self._last_harvester_grad_norm: float = 0.0
        self._last_trigger_tau: float = 0.005
        self._last_harvester_tau: float = 0.005

        # Dynamic entry confidence floor — updated by calibration and runway accuracy.
        # RL-adjusted floor converges separately and is capped relative to the base floor.
        self._entry_conf_dynamic_floor: float = 0.0
        # Stable base floor for the risk tuner — cached once from persisted params and
        # never re-read from the mutated value, so the adaptive cap cannot ratchet upward.
        self._risk_tuner_base_floor: float | None = None
        # Exit floor is persisted to param_manager and loaded here so it survives restarts.
        self._exit_conf_dynamic_floor = self._get_learned_float("exit_confidence_threshold", 0.0)
        # Rolling win-rate EMA for adaptive floor nudging (simplified risk tuner)
        self._win_rate_ema: float = 0.5
        self._win_rate_ema_n: int = 0
        # DDQN-exit-specific win-rate EMA for exit confidence floor adaptation
        self._ddqn_exit_win_ema: float = 0.5
        self._ddqn_exit_n: int = 0

    def _init_lifecycle_snapshots(self) -> None:
        # Entry-time vol/vpin snapshot — recorded at position open for close-time reward adj.
        self._entry_var: float = 0.0
        self._entry_vpin_z: float = 0.0

        # Entry-time lifecycle snapshots — persisted at open so trade_log has entry-vs-close diff.
        self._entry_dynamic_floor_applied: float = 0.0
        self._entry_conf_margin: float = 0.0
        self._entry_win_rate_ema: float = 0.5

        self._entry_total_trades: int = 0
        self._entry_equity: float = 0.0
        self._entry_conf_calib_err: float = 0.0
        self._entry_runway_accuracy: float = 0.0

        # Previous state tracking for transition events
        self._prev_cb_tripped: list[str] = []
        self._prev_regime: str = "UNKNOWN"

        # Entry trigger reasoning snapshot — populated at entry, consumed at close
        # by _write_trade_log so all trigger decision context is logged per trade.
        self._entry_trigger_data: dict = {}
        self._exit_lifecycle_data: dict = {}

    def _init_harvester_hold_state(self) -> None:
        # Dense harvester experience tracking — per-bar HOLD experiences while in position.
        # Mirrors the legacy ctrader_ddqn_paper.py pattern that kept the buffer full.
        # prev_harvester_state: harvester.last_state from the previous bar close.
        # Reset to None on position open; set each bar; consumed at position close.
        self._prev_harvester_state: Any = None
        self._prev_mfe: float = 0.0
        self._prev_mae: float = 0.0

    def _get_learned_float(self, param_name: str, default: float) -> float:
        return float(
            self._param_manager.get(
                self.symbol,
                param_name,
                timeframe=self.tf_label,
                broker="default",
                default=default,
            )
            or default
        )

    def _init_adaptive_emas(self) -> None:
        self._runway_delta_ema = self._get_learned_float("runway_delta_ema", 0.0)
        self._runway_accuracy_ema = self._get_learned_float("runway_accuracy_ema", 0.5)
        self._conf_calib_err_ema = self._get_learned_float("conf_calib_err_ema", 0.5)

    def reload_learned_parameters(self) -> None:
        """Reload this bot's scoped learned parameters and refresh cached thresholds."""
        self._param_manager.load()
        self._exit_conf_dynamic_floor = float(
            self._param_manager.get(
                self.symbol,
                "exit_confidence_threshold",
                timeframe=self.tf_label,
                broker="default",
                default=self._exit_conf_dynamic_floor,
            )
            or self._exit_conf_dynamic_floor
        )
        trigger = getattr(self.policy, "trigger", None)
        if trigger is not None and not getattr(trigger, "disable_gates", False):
            trigger.confidence_floor = float(
                self._param_manager.get(
                    self.symbol,
                    "confidence_floor",
                    timeframe=self.tf_label,
                    broker="default",
                    default=getattr(trigger, "confidence_floor", 0.55),
                )
                or getattr(trigger, "confidence_floor", 0.55)
            )
            if not getattr(trigger, "paper_mode", False):
                trigger.feasibility_threshold = float(
                    self._param_manager.get(
                        self.symbol,
                        "feasibility_threshold",
                        timeframe=self.tf_label,
                        broker="default",
                        default=getattr(trigger, "feasibility_threshold", 0.5),
                    )
                    or getattr(trigger, "feasibility_threshold", 0.5)
                )
            for attr, param_name, default in (
                ("entry_conf_deadzone_low", "entry_conf_deadzone_low", 0.45),
                ("entry_conf_deadzone_high", "entry_conf_deadzone_high", 0.55),
                ("high_conf_risk_low", "high_conf_risk_low", 0.80),
                ("high_conf_risk_high", "high_conf_risk_high", 0.90),
                ("high_conf_vol_z_gate", "high_conf_vol_z_gate", 1.0),
                ("high_conf_vpin_z_gate", "high_conf_vpin_z_gate", 2.0),
            ):
                setattr(
                    trigger,
                    attr,
                    float(
                        self._param_manager.get(
                            self.symbol,
                            param_name,
                            timeframe=self.tf_label,
                            broker="default",
                            default=getattr(trigger, attr, default),
                        )
                        or getattr(trigger, attr, default)
                    ),
                )
        LOG.info(
            "[%s %s] Reloaded learned parameters: confidence_floor=%.3f exit_floor=%.3f",
            self.symbol,
            self.tf_label,
            float(getattr(trigger, "confidence_floor", 0.0) if trigger is not None else 0.0),
            self._exit_conf_dynamic_floor,
        )

    # ---- tick ingestion ---------------------------------------------------

    def on_tick(
        self,
        ts: dt.datetime,
        mid: float,
        half_spread: float,
        imbalance: float = 0.0,
        depth_bid: float = 0.0,
        depth_ask: float = 0.0,
        vpin_z: float = 0.0,
        has_real_sizes: bool = False,
        l2_snapshot: dict | None = None,
    ) -> None:
        self.last_mid = mid
        self.last_half_spread = half_spread
        self.last_ts = ts
        self._entry_imbalance = imbalance
        self._last_depth_bid = depth_bid
        self._last_depth_ask = depth_ask
        self._vpin_z = vpin_z
        self._has_real_sizes = has_real_sizes
        self._last_l2_snapshot = l2_snapshot or {}
        if half_spread > 0:
            self.friction_calc.update_spread(mid - half_spread, mid + half_spread)

        # EXIT is tick-level: evaluate on every price update so trailing stops
        # and capture-decay fire at the actual peak, not at bar close.
        if self.position is not None and len(self.bars) >= _MIN_BARS_BEFORE_TRADE:
            self._handle_exit_on_tick(ts, mid, half_spread)
            if self.position is None:
                # Position was closed this tick — skip bar close entry logic.
                bar = self.bar_builder.update(ts, mid)
                if bar is not None:
                    self._on_bar_close_no_exit(bar, half_spread)
                else:
                    now = time.time()
                    if now - self._last_telemetry_time >= 30.0:
                        self._last_telemetry_time = now
                        self._write_telemetry()
                return

        # ENTRY is bar-close-level: wait for a completed bar before signalling.
        bar = self.bar_builder.update(ts, mid)
        if bar is not None:
            self._on_bar_close(bar, half_spread)
        else:
            now = time.time()
            if now - self._last_telemetry_time >= 30.0:
                self._last_telemetry_time = now
                self._write_telemetry()

    # ---- bar processing --------------------------------------------------

    def _on_bar_close(self, bar: tuple, half_spread: float) -> None:
        _ts, _o, _h, _l, _c = bar
        self.bars.append(bar)
        self.bar_count += 1
        self._bars_since_train += 1

        # Update path geometry, event-time features, and cached risk values on every bar
        _vol = self._realized_vol()
        self.path_geometry.update(self.bars, sigma=_vol)
        try:
            _bar_dt = _ts if isinstance(_ts, dt.datetime) else dt.datetime.fromtimestamp(float(_ts), tz=dt.UTC)
            self._last_event_feats = self.event_time_engine.calculate_features(_bar_dt)
        except Exception:
            pass
        if len(self.bars) >= 2:
            _prev_c = self.bars[-2][4]
            if _prev_c > 0 and _c > 0:
                try:
                    self.var_estimator.update_return(math.log(_c / _prev_c))
                    self.var_estimator.set_reference_vol(_vol)
                except Exception:
                    pass
        self._last_var_95, self._last_kurtosis = self._get_var_kurtosis()

        if len(self.bars) < _MIN_BARS_BEFORE_TRADE:
            if self.bar_count % 10 == 0:
                self._write_telemetry()
            with contextlib.suppress(Exception):
                self.decision_log.log_decision(
                    agent="TriggerAgent",
                    decision="WARMING_UP",
                    confidence=0.0,
                    context={"price": _c, "bars": self.bar_count, "required": _MIN_BARS_BEFORE_TRADE},
                )
            return

        if not self._harvester_preseeded:
            self._preseed_harvester_from_bars()

        # Auto-reset expired circuit breakers every bar (not only in _handle_flat)
        if self.circuit_breakers is not None:
            self.circuit_breakers.reset_if_cooldown_elapsed()

        # Notify activity monitor so inactivity counter stays accurate; also trigger
        # get_exploration_bonus() to keep internal stagnation state fresh.
        try:
            _am = self.reward_shaper.activity_monitor
            _am.on_bar_close()
            if hasattr(_am, "get_exploration_bonus"):
                _am.get_exploration_bonus()
        except Exception:
            pass

        # Exit is now tick-level (_handle_exit_on_tick). Bar close only handles entry.
        if self.position is None:
            self._handle_flat(bar, half_spread)
        else:
            # Dense harvester feedback: one HOLD experience per bar held in position.
            # This keeps the replay buffer full (same pattern as legacy ctrader_ddqn_paper.py).
            self._add_harvester_hold_experience()

        self._maybe_train()
        self._maybe_checkpoint_and_telemetry()

    def _on_bar_close_no_exit(self, bar: tuple, _half_spread: float) -> None:
        """Bar close processing when position was already closed this tick.

        Runs the same bar-state updates and training as _on_bar_close but skips
        entry logic since we just exited — no immediate re-entry on same tick.
        """
        _ts, _o, _h, _l, _c = bar
        self.bars.append(bar)
        self.bar_count += 1
        self._bars_since_train += 1
        _vol = self._realized_vol()
        self.path_geometry.update(self.bars, sigma=_vol)
        try:
            _bar_dt = _ts if isinstance(_ts, dt.datetime) else dt.datetime.fromtimestamp(float(_ts), tz=dt.UTC)
            self._last_event_feats = self.event_time_engine.calculate_features(_bar_dt)
        except Exception:
            pass
        if len(self.bars) >= 2:
            _prev_c = self.bars[-2][4]
            if _prev_c > 0 and _c > 0:
                try:
                    self.var_estimator.update_return(math.log(_c / _prev_c))
                    self.var_estimator.set_reference_vol(_vol)
                except Exception:
                    pass
        self._last_var_95, self._last_kurtosis = self._get_var_kurtosis()
        self._maybe_train()
        self._maybe_checkpoint_and_telemetry()

    def _maybe_train(self) -> None:
        if self._bars_since_train >= _TRAIN_INTERVAL_BARS and hasattr(self.policy, "train_step"):
            try:
                _train_metrics = self.policy.train_step(adaptive_reg=self.adaptive_reg)
                if _train_metrics:
                    _tm_t = _train_metrics.get("trigger") or {}
                    _tm_h = _train_metrics.get("harvester") or {}
                    self._last_trigger_grad_norm = float(
                        _tm_t.get("grad_norm", self._last_trigger_grad_norm) or self._last_trigger_grad_norm,
                    )
                    self._last_harvester_grad_norm = float(
                        _tm_h.get("grad_norm", self._last_harvester_grad_norm) or self._last_harvester_grad_norm,
                    )
                    self._last_trigger_tau = float(
                        _tm_t.get("adaptive_tau", self._last_trigger_tau) or self._last_trigger_tau,
                    )
                    self._last_harvester_tau = float(
                        _tm_h.get("adaptive_tau", self._last_harvester_tau) or self._last_harvester_tau,
                    )
                    if self.adaptive_reg is not None:
                        _t_td = float(_tm_t.get("mean_td_error", 0.0) or 0.0)
                        _h_td = float(_tm_h.get("mean_td_error", 0.0) or 0.0)
                        _avg_td = (_t_td + _h_td) / 2.0
                        if _avg_td > 0.5:
                            self.adaptive_reg.increase_regularization()
                        elif 0.0 < _avg_td < 0.1:
                            self.adaptive_reg.decrease_regularization()
            except Exception as e:
                LOG.debug("[%s %s] train_step error: %s", self.symbol, self.tf_label, e)
            self._bars_since_train = 0
        if self.reward_shaping_monitor is not None:
            try:
                regime = str(getattr(self.policy, "current_regime", "UNKNOWN") or "UNKNOWN")
                self.reward_shaping_monitor.run_if_due(current_regime=regime)
            except Exception as e:
                LOG.debug("[%s %s] reward_shaping_monitor error: %s", self.symbol, self.tf_label, e)

    def _maybe_checkpoint_and_telemetry(self) -> None:
        if self.bar_count % _CHECKPOINT_INTERVAL_BARS == 0:
            self._save_checkpoint()
        if self.bar_count % _TELEMETRY_INTERVAL_BARS == 0:
            self._write_telemetry()
        self._bars_since_cache_save += 1
        if self._bars_since_cache_save >= self._BARS_CACHE_SAVE_EVERY:
            self._save_bars_cache()
            self._bars_since_cache_save = 0

    # ---- market state helpers --------------------------------------------

    def _realized_vol(self) -> float:
        """Rolling std of log-returns over last 20 bars; falls back to 0.005."""
        if len(self.bars) < 5:
            return 0.005
        closes = [b[4] for b in list(self.bars)[-20:]]
        try:
            rets = np.diff(np.log(np.array(closes, dtype=float)))
            v = float(np.std(rets))
            return v if v > 0 else 0.005
        except Exception:
            return 0.005

    def _depth_ratio(self) -> float:
        return (self._last_depth_bid / self._last_depth_ask
                if self._last_depth_ask > 0 else 1.0)

    def _l2_book_crossed(self) -> tuple[bool, float, float]:
        """Return whether the latest L2 snapshot has bid >= ask."""
        snapshot = getattr(self, "_last_l2_snapshot", {}) or {}
        try:
            bids = snapshot.get("bids") or []
            asks = snapshot.get("asks") or []
            if not bids or not asks:
                return False, 0.0, 0.0
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            if best_bid <= 0 or best_ask <= 0:
                return False, best_bid, best_ask
            return best_bid >= best_ask, best_bid, best_ask
        except (TypeError, ValueError, IndexError):
            return False, 0.0, 0.0

    def _paper_entry_guard_blocks(self, gated: list[str]) -> bool:
        """Block paper execution on structurally unsafe market snapshots.

        Paper mode can still explore weak model setups, but it should not open
        account-level paper trades from crossed books or rejected spreads. Those
        states remain visible through gated_conditions for HUD/replay analysis.
        """
        if not self.paper_mode:
            return False
        enabled = float(self._param_manager.get(
            self.symbol,
            "paper_entry_guard_enabled",
            timeframe=self.tf_label,
            broker="default",
            default=1.0,
        ) or 0.0)
        if enabled <= 0.0:
            return False
        unsafe_prefixes = ("crossed_l2", "spread=")
        return any(str(g).startswith(unsafe_prefixes) for g in gated)

    # ---- entry / exit logic ----------------------------------------------

    def _active_kurtosis_threshold(self) -> float:
        """Return the canonical kurtosis gate threshold — CB learned value or default 5.0.

        Also keeps the VaREstimator's KurtosisMonitor threshold aligned so both
        use the same learned threshold.
        """
        threshold = 5.0
        if self.circuit_breakers is not None:
            with contextlib.suppress(Exception):
                threshold = float(self.circuit_breakers.kurtosis_threshold)
        with contextlib.suppress(Exception):
            self.var_estimator.kurtosis_monitor.threshold = threshold
        return threshold

    def _get_var_kurtosis(self) -> tuple[float, float]:
        """Return (var_95, excess_kurtosis) from rolling VaREstimator."""
        try:
            regime_str = str(getattr(self.policy, "current_regime", "CRITICAL") or "CRITICAL").upper()
            _rm = {"OVERDAMPED": self._regime_type.OVERDAMPED, "UNDERDAMPED": self._regime_type.UNDERDAMPED}
            regime = _rm.get(regime_str, self._regime_type.CRITICAL)
            var = float(
                self.var_estimator.estimate_var(
                    regime=regime,
                    vpin_z=self._vpin_z,
                    current_vol=self._realized_vol(),
                ),
            )
            kurtosis = float(self.var_estimator.kurtosis)
            return var, kurtosis
        except Exception:
            return self._compute_var_kurtosis()

    def _compute_rs_vol(self, n: int) -> float:
        """Rogers-Satchell volatility over last n bars (drift-independent, uses OHLC).

        RS² = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O); return sqrt(mean(RS²)).
        Result is in fractional (log-price) units, comparable to pct_change std.
        """
        bars_list = list(self.bars)
        window = bars_list[-n:] if len(bars_list) >= n else bars_list
        if len(window) < 2:
            return 0.0
        rs2_vals: list[float] = []
        for _, o, h, low, c in window:
            if not (SafeMath.is_valid(o) and SafeMath.is_valid(h)
                    and SafeMath.is_valid(low) and SafeMath.is_valid(c)):
                continue
            if o <= 0 or h <= 0 or low <= 0 or c <= 0:
                continue
            try:
                rs2 = math.log(h / c) * math.log(h / o) + math.log(low / c) * math.log(low / o)
            except (ValueError, ZeroDivisionError):
                continue
            rs2_vals.append(rs2)
        if not rs2_vals:
            return 0.0
        return float(math.sqrt(max(sum(rs2_vals) / len(rs2_vals), 0.0)))

    def _compute_er(self, n: int = 10) -> float:
        """Kaufman Efficiency Ratio: |net move| / sum(|bar moves|) over n bars."""
        bars_list = list(self.bars)
        if len(bars_list) < n + 1:
            return 0.0
        closes = [b[4] for b in bars_list[-(n + 1):]]
        net = abs(closes[-1] - closes[0])
        path = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
        return float(net / path) if path > 0 else 0.0

    def _compute_returns(self) -> tuple[float, float, float]:
        """(ret1, ret5, ret20) fractional price changes from bar buffer."""
        bars_list = list(self.bars)
        n = len(bars_list)
        c = bars_list[-1][4] if n >= 1 else 0.0
        c1 = bars_list[-2][4] if n >= 2 else c
        c5 = bars_list[-6][4] if n >= 6 else bars_list[0][4]
        c20 = bars_list[-21][4] if n >= 21 else bars_list[0][4]
        r1 = (c - c1) / c1 if c1 > 0 else 0.0
        r5 = (c - c5) / c5 if c5 > 0 else 0.0
        r20 = (c - c20) / c20 if c20 > 0 else 0.0
        return float(r1), float(r5), float(r20)

    def _alignment_score(self, action: int, ret1: float, ret5: float, ret20: float) -> int:
        """Count how many of ret1/ret5/ret20 align with the entry direction (1=LONG, 2=SHORT)."""
        if action not in (1, 2):
            return 0
        sign = 1 if action == 1 else -1
        return sum(1 for r in (ret1, ret5, ret20) if abs(r) > SAFE_EPSILON and (1 if r > 0 else -1) == sign)

    def _bars_since_energy_bar(self, rs_vol: float) -> int:
        """Bars since the last bar whose fractional range exceeded 1.5× RS volatility."""
        if rs_vol <= 0:
            return 0
        bars_list = list(self.bars)
        threshold = 1.5 * rs_vol
        for i in range(len(bars_list) - 1, -1, -1):
            _, _, h, low, c = bars_list[i]
            if c > 0 and (h - low) / c > threshold:
                return len(bars_list) - 1 - i
        return len(bars_list)

    def _compute_dynamic_entry_floor(self, base_floor: float) -> tuple[float, dict]:
        """Raise minimum entry confidence based on calibration error and runway accuracy.

        Ports the legacy dynamic_entry_floor logic: two additive uplifts on top of the
        base floor — one from Brier-score calibration error, one from runway accuracy gap.
        An RL-adjusted floor is also maintained but capped relative to base_floor.
        """
        min_samples = max(1, round(float(self._param_manager.get(
            self.symbol, "entry_guard_min_trade_samples",
            timeframe=self.tf_label, broker="default", default=40.0) or 40.0)))
        cal_err = float(self._conf_calib_err_ema or 0.0)
        runway_acc = float(self._runway_accuracy_ema or 0.5)
        cal_start = float(self._param_manager.get(
            self.symbol, "entry_guard_calib_err_start",
            timeframe=self.tf_label, broker="default", default=0.30) or 0.30)
        uplift_cap = float(self._param_manager.get(
            self.symbol, "entry_guard_calib_uplift_cap",
            timeframe=self.tf_label, broker="default", default=0.08) or 0.08)
        runway_target = float(self._param_manager.get(
            self.symbol, "entry_guard_runway_acc_target",
            timeframe=self.tf_label, broker="default", default=0.60) or 0.60)
        runway_cap = float(self._param_manager.get(
            self.symbol, "entry_guard_runway_penalty_cap",
            timeframe=self.tf_label, broker="default", default=0.05) or 0.05)
        rl_extra_cap = float(self._param_manager.get(
            self.symbol, "entry_guard_rl_floor_extra_cap",
            timeframe=self.tf_label, broker="default", default=0.10) or 0.10)

        if self.total_trades < min_samples:
            uplift = 0.0
            runway_penalty = 0.0
        else:
            uplift = min(max(cal_err - cal_start, 0.0), max(0.0, uplift_cap))
            runway_penalty = min(max(runway_target - runway_acc, 0.0), max(0.0, runway_cap))

        rl_floor_capped = min(
            self._entry_conf_dynamic_floor,
            float(base_floor) + max(0.0, rl_extra_cap),
        )
        dyn_floor = max(float(base_floor) + uplift + runway_penalty, rl_floor_capped)
        return float(dyn_floor), {
            "cal_err": cal_err, "uplift": uplift,
            "runway_penalty": runway_penalty, "rl_floor": rl_floor_capped,
        }

    def _get_hmm_probs(self) -> dict[str, float]:
        """Return HMM regime probabilities {trending, mean_reverting, neutral} if available."""
        rd = getattr(self.policy, "regime_detector", None)
        if rd is None:
            return {}
        get_probs = getattr(rd, "get_regime_probabilities", None)
        if get_probs is None:
            return {}
        try:
            probs = get_probs()
            return {
                "trending": float(probs[0]),
                "mean_reverting": float(probs[1]),
                "neutral": float(probs[2]),
            }
        except Exception:
            return {}

    def _log_entry_decision(
        self,
        action: int,
        conf: float,
        runway: float,
        price: float,
        vol: float,
        depth_ratio: float,
        half_spread: float,
        entry_bar: tuple,
        gated_conditions: list[str] | None = None,
    ) -> None:
        _LABELS = {1: "LONG", 2: "SHORT"}
        label = _LABELS.get(action, "NO_ENTRY")
        regime = str(getattr(self.policy, "current_regime", "UNKNOWN") or "UNKNOWN")
        feasibility = float(getattr(self.policy, "current_zeta", 1.0) or 1.0)
        geom = self.path_geometry.last
        cb_ok = self.circuit_breakers is None or not self.circuit_breakers.is_any_tripped()
        active_sessions = [
            k for k, v in self._last_event_feats.items()
            if k.endswith("_is_active") and v > 0
        ] if self._last_event_feats else []

        rs_vol_s = self._compute_rs_vol(10)
        rs_vol_l = self._compute_rs_vol(50)
        rs_vol_ratio = (rs_vol_s / rs_vol_l) if rs_vol_l > 0 else 1.0
        er10 = self._compute_er(10)
        ret1, ret5, ret20 = self._compute_returns()

        _ts_b, _o_b, _h_b, _l_b, _c_b = entry_bar
        bars_list = list(self.bars)
        prev_c = bars_list[-2][4] if len(bars_list) >= 2 else _o_b
        gap_pts = float(_o_b - prev_c)
        gap_rs = (gap_pts / prev_c / rs_vol_s) if rs_vol_s > 0 and prev_c > 0 else 0.0

        trig_stats: dict = {}
        if hasattr(self.policy, "get_training_stats"):
            trig_stats = (self.policy.get_training_stats() or {}).get("trigger") or {}
        training_steps = int(trig_stats.get("training_steps", 0))
        epsilon = float(trig_stats.get("epsilon", 1.0))

        try:
            self.decision_log.log_decision(
                agent="TriggerAgent",
                decision=label,
                confidence=float(conf),
                context={
                    "price": price,
                    "volatility": vol,
                    "imbalance": self._entry_imbalance,
                    "vpin_z": self._vpin_z,
                    "regime": regime,
                    "session": active_sessions,
                },
                reasoning={
                    "predicted_runway": float(runway),
                    "predicted_runway_gross": float(runway),
                    "predicted_runway_net": float(
                        max(0.0, runway - (2.0 * half_spread / max(abs(price), 1.0))),
                    ),
                    "feasibility": feasibility,
                    "geometry_efficiency": geom.get("efficiency", 0.0),
                    "geometry_runway": geom.get("runway", 0.5),
                    "circuit_breakers_ok": cb_ok,
                    "gated_conditions": gated_conditions or [],
                    "depth_ratio": depth_ratio,
                    "depth_bid": self._last_depth_bid,
                    "depth_ask": self._last_depth_ask,
                    "has_real_l2_sizes": self._has_real_sizes,
                    "l2_snapshot": self._last_l2_snapshot,
                    "var_95": self._last_var_95,
                    "kurtosis": self._last_kurtosis,
                    "kurtosis_threshold": self._active_kurtosis_threshold(),
                    "rs_vol_short": rs_vol_s,
                    "rs_vol_long": rs_vol_l,
                    "rs_vol_ratio": rs_vol_ratio,
                    "gap_rs": gap_rs,
                    "half_spread": half_spread,
                    "training_steps": training_steps,
                    "epsilon": epsilon,
                    "explore_flag": epsilon > 0.05,
                    "er10": er10,
                    "ret1": ret1,
                    "ret5": ret5,
                    "ret20": ret20,
                    "alignment_score": self._alignment_score(action, ret1, ret5, ret20),
                    "bars_since_energy_bar": self._bars_since_energy_bar(rs_vol_s),
                    "hmm_probs": self._get_hmm_probs(),
                    "runway_accuracy": self._runway_accuracy_ema,
                    "runway_delta_ema": self._runway_delta_ema,
                    "conf_calib_err": self._conf_calib_err_ema,
                    "entry_bar_ohlcv": {
                        "open": float(_o_b), "high": float(_h_b),
                        "low": float(_l_b), "close": float(_c_b),
                    },
                    "entry_dynamic_floor": self._entry_dynamic_floor_applied,
                    "entry_conf_margin": self._entry_conf_margin,
                    "win_rate_ema": self._win_rate_ema,
                    "equity": self.equity,
                    "total_trades": self.total_trades,
                },
                trade_id=self._current_trade_id,
            )
        except Exception as e:
            LOG.debug("[%s %s] decision_log error: %s", self.symbol, self.tf_label, e)

    def _calc_hold_reward(
        self, cur_mfe: float, cur_mae: float, entry_price: float, bars_held: int,
    ) -> float:
        """Incremental HOLD reward for one bar held in a position.

        Ported from ctrader_ddqn_paper._calculate_harvester_hold_reward.
        Five components: capture quality, MFE growth, MAE penalty, time decay, opportunity cost.
        All in [-1, 1].
        """
        ref_price = max(abs(entry_price), 1.0)
        vol = max(self._realized_vol(), SAFE_SMALL)
        pos = self.position
        direction = pos["direction"] if pos else 1
        unrealized = (self.last_mid - entry_price) * direction

        capture_ratio = unrealized / cur_mfe if cur_mfe > 0 else 0.0
        capture_component = float(np.clip(capture_ratio * 0.4, 0.0, 0.4))

        mfe_delta_frac = (cur_mfe - self._prev_mfe) / ref_price
        mfe_growth = float(np.clip(mfe_delta_frac / vol * 0.3, -0.3, 0.3))

        mae_delta_frac = (cur_mae - self._prev_mae) / ref_price
        mae_penalty = float(-np.clip(mae_delta_frac / vol * 0.4, 0.0, 0.4))

        bars_per_day = max(10, 1440 // max(1, self.timeframe_minutes))
        time_decay = -0.02 * min(bars_held / max(1, bars_per_day // 10), 10.0)

        dist_from_peak = float(np.clip(
            (cur_mfe - unrealized) / cur_mfe if cur_mfe > 0 else 0.0, 0.0, 2.0))
        opportunity_cost = -dist_from_peak * 0.3

        return float(np.clip(
            capture_component + mfe_growth + mae_penalty + time_decay + opportunity_cost,
            -1.0, 1.0,
        ))

    def _add_harvester_hold_experience(self) -> None:
        """Add a dense HOLD experience to the harvester buffer at each bar close while in position.

        This is the key mechanism that keeps the harvester replay buffer full — one
        experience per bar, not just one per trade. Mirrors the legacy paper DDQN pattern.
        """
        harv_state = getattr(getattr(self.policy, "harvester", None), "last_state", None)
        if harv_state is None:
            return

        if self._prev_harvester_state is None:
            # First bar after entry — save state for next bar's (state, action, reward, next) tuple
            self._prev_harvester_state = harv_state.copy()
            pos_metrics = self.policy.get_position_metrics() if hasattr(self.policy, "get_position_metrics") else {}
            self._prev_mfe = float(pos_metrics.get("mfe", 0.0))
            self._prev_mae = float(pos_metrics.get("mae", 0.0))
            return

        pos_metrics = self.policy.get_position_metrics() if hasattr(self.policy, "get_position_metrics") else {}
        cur_mfe = float(pos_metrics.get("mfe", 0.0))
        cur_mae = float(pos_metrics.get("mae", 0.0))
        entry_price = float(pos_metrics.get("entry_price", self.last_mid))
        bars_held = int(pos_metrics.get("ticks_held", 0)) // max(1, self.timeframe_minutes)

        reward = self._calc_hold_reward(cur_mfe, cur_mae, entry_price, bars_held)
        try:
            self.policy.add_harvester_experience(
                state=self._prev_harvester_state,
                action=0,       # HOLD
                reward=reward,
                next_state=harv_state,
                done=False,
            )
        except Exception as e:
            LOG.debug("[%s %s] hold_experience error: %s", self.symbol, self.tf_label, e)

        self._prev_harvester_state = harv_state.copy()
        self._prev_mfe = cur_mfe
        self._prev_mae = cur_mae

        # Log HOLD decision to audit trail so position history is traceable
        try:
            unrealized = (self.last_mid - entry_price) * (self.position["direction"] if self.position else 1)
            capture_r = unrealized / cur_mfe if cur_mfe > SAFE_EPSILON else 0.0
            _harv = getattr(self.policy, "harvester", None)
            _mfe_pct = (cur_mfe / max(abs(entry_price), 1.0)) * 100.0
            _trail_act = getattr(_harv, "trailing_stop_activation_pct", 0.25)
            _trail_dist = getattr(_harv, "trailing_stop_distance_pct", 0.12)
            _be_trig = getattr(_harv, "breakeven_trigger_pct", 0.30)
            _cd_min = getattr(_harv, "capture_decay_min_mfe_pct", 0.10)
            _cd_thr = float(getattr(_harv, "capture_decay_threshold", 0.35))
            self.decision_log.log_harvester_decision(
                decision="HOLD",
                confidence=self._last_harvester_conf,
                price=self.last_mid,
                entry_price=entry_price,
                mfe=cur_mfe,
                mae=cur_mae,
                ticks_held=bars_held,
                unrealized_pnl=unrealized,
                capture_ratio=float(capture_r),
                trade_id=self._current_trade_id,
                in_position=True,
                regime=str(getattr(self.policy, "current_regime", "UNKNOWN") or "UNKNOWN"),
                realized_vol=self._realized_vol(),
                depth_ratio=self._depth_ratio(),
                exit_floor=self._exit_conf_dynamic_floor,
                trailing_stop_active=_mfe_pct >= _trail_act,
                trailing_stop_activation_pct=_trail_act,
                trailing_stop_distance_pct=_trail_dist,
                breakeven_active=_mfe_pct >= _be_trig,
                breakeven_trigger_pct=_be_trig,
                capture_decay_armed=_mfe_pct >= _cd_min,
                capture_decay_threshold=_cd_thr,
            )
        except Exception as e:
            LOG.debug("[%s %s] hold_decision_log error: %s", self.symbol, self.tf_label, e)

    _BARS_CACHE_SIZE: int = 500   # bars to persist across restarts (matches legacy)
    _BARS_CACHE_SAVE_EVERY: int = 10  # save every N bar closes

    def _save_bars_cache(self) -> None:
        """Persist the last _BARS_CACHE_SIZE bars to disk for warm restart preseeding."""
        try:
            bars_list = list(self.bars)[-self._BARS_CACHE_SIZE:]
            serialized = []
            for b in bars_list:
                ts_val = b[0]
                if isinstance(ts_val, dt.datetime):
                    ts_val = ts_val.isoformat()
                serialized.append([ts_val, float(b[1]), float(b[2]), float(b[3]), float(b[4])])
            _write_json_atomic(
                self.data_dir / "bars_cache.json",
                {"symbol": self.symbol, "tf": self.tf_label, "bars": serialized},
            )
        except Exception as e:
            LOG.debug("[%s %s] bars_cache save error: %s", self.symbol, self.tf_label, e)

    def _load_bars_cache(self) -> None:
        """Load bars cache from disk and replay into self.bars + VaR estimator."""
        cache_path = self.data_dir / "bars_cache.json"
        if not cache_path.exists():
            return
        try:
            with open(cache_path, encoding="utf-8") as f:
                payload = json.load(f)
            raw_bars = payload.get("bars", [])
            if not raw_bars:
                return
            loaded = 0
            prev_close: float = 0.0
            for row in raw_bars:
                if len(row) < 5:
                    continue
                try:
                    ts_val = row[0]
                    if isinstance(ts_val, str):
                        ts_val = dt.datetime.fromisoformat(ts_val)
                    cur_close = float(row[4])
                    b = (ts_val, float(row[1]), float(row[2]), float(row[3]), cur_close)
                    self.bars.append(b)
                    if prev_close > 0 and cur_close > 0:
                        with contextlib.suppress(Exception):
                            self.var_estimator.update_return(math.log(cur_close / prev_close))
                    prev_close = cur_close if cur_close > 0 else prev_close
                    loaded += 1
                except Exception:
                    continue
            self.bar_count = max(self.bar_count, loaded)
            LOG.info("[%s %s] Bars cache loaded: %d bars → warm-start preseeding",
                     self.symbol, self.tf_label, loaded)
            # Log a decision for each cached bar so the audit trail shows the
            # full bar history, not just live ticks.  This matches legacy FIX
            # behaviour where every bar processed generated a trigger decision.
            if loaded >= _MIN_BARS_BEFORE_TRADE:
                try:
                    for b in list(self.bars)[_MIN_BARS_BEFORE_TRADE:]:
                        _bt, _bo, _bh, _bl, _bc = b
                        self.decision_log.log_decision(
                            agent="TriggerAgent",
                            decision="CACHED",
                            confidence=0.5,
                            context={"price": float(_bc), "bars": self.bar_count, "source": "cache_load"},
                            trade_id=None,
                        )
                except Exception:
                    pass
            # Immediately preseed both buffers from the loaded bars
            if loaded >= _MIN_BARS_BEFORE_TRADE:
                self._last_var_95, self._last_kurtosis = self._get_var_kurtosis()
                self._preseed_harvester_from_bars()
                self._preseed_trigger_buffer()
        except Exception as e:
            LOG.warning("[%s %s] bars_cache load error: %s", self.symbol, self.tf_label, e)

    def _entry_circuit_breaker_blocks(self) -> bool:
        if self.circuit_breakers is not None and self.circuit_breakers.check_all():
            LOG.info("[%s %s] Circuit breaker tripped — skip entry", self.symbol, self.tf_label)
            return True
        return False

    def _entry_depth_gate_blocks(self, depth_floor: float) -> bool:
        # Paper mode: soft-gate depth instead of hard-blocking so the RL agent
        # trains on thin-book conditions and learns to avoid them naturally.
        # Live mode keeps the hard block for execution safety.
        depth_too_thin = (
            depth_floor > 0 and self._last_depth_bid > 0 and self._last_depth_ask > 0
            and min(self._last_depth_bid, self._last_depth_ask) < depth_floor
        )
        if not depth_too_thin:
            return False
        LOG.debug(
            "[%s %s] Depth gate: book too thin (bid=%.3f ask=%.3f < floor=%.3f)%s",
            self.symbol,
            self.tf_label,
            self._last_depth_bid,
            self._last_depth_ask,
            depth_floor,
            " — skip entry" if not self.paper_mode else " — paper: allowing for RL training",
        )
        return not self.paper_mode

    def _entry_soft_gates(self) -> list[str]:
        # Soft gates — paper mode: log and allow entry so RL agent trains on all conditions.
        gated: list[str] = []

        kurtosis_threshold = self._active_kurtosis_threshold()
        if self._last_kurtosis > kurtosis_threshold:
            gated.append(f"kurtosis={self._last_kurtosis:.2f}>{kurtosis_threshold:.2f}")
            LOG.debug(
                "[%s %s] [SOFT-GATE] κ=%.2f > %.2f — allowing for RL training",
                self.symbol,
                self.tf_label,
                self._last_kurtosis,
                kurtosis_threshold,
            )

        vol_cap = float(
            self._param_manager.get(
                self.symbol, "vol_cap", timeframe=self.tf_label, broker="default", default=0.05,
            ) or 0.05,
        )
        if self._last_var_95 > vol_cap:
            gated.append(f"var={self._last_var_95:.4f}>{vol_cap:.4f}")
            LOG.debug(
                "[%s %s] [SOFT-GATE] VaR=%.4f > cap=%.4f — allowing for RL training",
                self.symbol,
                self.tf_label,
                self._last_var_95,
                vol_cap,
            )

        vpin_threshold = float(
            self._param_manager.get(
                self.symbol,
                "vpin_z_threshold",
                timeframe=self.tf_label,
                broker="default",
                default=2.5,
            ) or 2.5,
        )
        if vpin_threshold > 0 and abs(self._vpin_z) > vpin_threshold:
            gated.append(f"vpin_z={self._vpin_z:.2f}")
            LOG.debug(
                "[%s %s] [SOFT-GATE] VPIN z=%.2f > %.2f — allowing for RL training",
                self.symbol,
                self.tf_label,
                self._vpin_z,
                vpin_threshold,
            )

        crossed_l2, best_bid, best_ask = self._l2_book_crossed()
        if crossed_l2:
            gated.append(f"crossed_l2={best_bid:.5f}>={best_ask:.5f}")
            LOG.warning(
                "[%s %s] Crossed L2 book: bid=%.5f >= ask=%.5f — unsafe for paper execution",
                self.symbol,
                self.tf_label,
                best_bid,
                best_ask,
            )

        try:
            spread_ok, cur_spread, max_spread = self.friction_calc.is_spread_acceptable()
            if not spread_ok:
                gated.append(f"spread={cur_spread:.3f}>{max_spread:.3f}")
                LOG.debug(
                    "[%s %s] [SOFT-GATE] spread=%.3f > max=%.3f — allowing for RL training",
                    self.symbol,
                    self.tf_label,
                    cur_spread,
                    max_spread,
                )
        except Exception:
            pass
        return gated

    def _decide_flat_entry(self, vol: float, depth_ratio: float) -> tuple[int, float, float, Any] | None:
        try:
            action, conf, runway = self.policy.decide_entry(
                self.bars,
                imbalance=self._entry_imbalance,
                vpin_z=self._vpin_z,
                depth_ratio=depth_ratio,
                realized_vol=vol,
                event_features=self._last_event_feats or None,
            )
        except Exception as e:
            LOG.debug("[%s %s] decide_entry error: %s", self.symbol, self.tf_label, e)
            return None

        self._last_trigger_conf = float(conf)
        trig_state = getattr(self.policy.trigger, "last_state", None)
        self._entry_state = trig_state.copy() if trig_state is not None else None
        return int(action), float(conf), float(runway), trig_state

    def _apply_entry_dynamic_floor(self, action: int, conf: float, gated: list[str]) -> tuple[int, float]:
        # Dynamic entry floor: raises minimum confidence when calibration or runway accuracy is poor.
        # Paper mode: log the floor breach but allow entry — RL needs to train on all setups.
        # Live mode: block entry to protect capital from poorly calibrated decisions.
        dyn_floor = 0.0
        if action == 0:
            return action, dyn_floor
        base_floor = float(min(
            max(
                float(
                    self._param_manager.get(
                        self.symbol,
                        "entry_confidence_threshold",
                        timeframe=self.tf_label,
                        broker="default",
                        default=0.55,
                    ) or 0.55,
                ),
                self._RISK_TUNER_BASE_MIN,
            ),
            self._RISK_TUNER_BASE_MAX,
        ))
        dyn_floor, floor_dbg = self._compute_dynamic_entry_floor(base_floor)
        if conf >= dyn_floor:
            return action, dyn_floor
        if self.paper_mode:
            LOG.debug(
                "[%s %s] DynFloor LOG (paper): conf=%.3f < floor=%.3f "
                "(base=%.3f uplift=%.3f runway=%.3f) — allowing for RL training",
                self.symbol,
                self.tf_label,
                conf,
                dyn_floor,
                base_floor,
                floor_dbg["uplift"],
                floor_dbg["runway_penalty"],
            )
            gated.append(f"conf={conf:.3f}<floor={dyn_floor:.3f}")
            return action, dyn_floor
        LOG.debug(
            "[%s %s] DynFloor block: conf=%.3f < floor=%.3f (base=%.3f uplift=%.3f runway=%.3f)",
            self.symbol,
            self.tf_label,
            conf,
            dyn_floor,
            base_floor,
            floor_dbg["uplift"],
            floor_dbg["runway_penalty"],
        )
        return 0, dyn_floor

    def _apply_paper_entry_guard(self, action: int, conf: float, gated: list[str]) -> int:
        if action != 0 and self._paper_entry_guard_blocks(gated):
            LOG.warning(
                "[%s %s] Paper entry guard blocked execution: action=%d conf=%.3f gates=%s",
                self.symbol,
                self.tf_label,
                action,
                conf,
                ",".join(gated),
            )
            return 0
        return action

    def _snapshot_entry_lifecycle(
        self,
        *,
        action: int,
        conf: float,
        dyn_floor: float,
        gated: list[str],
        depth_ratio: float,
        half_spread: float,
        bar: tuple,
    ) -> None:
        _ts, bar_open, bar_high, bar_low, bar_close = bar
        # Snapshot entry-time trade_id and lifecycle metrics BEFORE logging the decision
        # so _log_entry_decision captures them with the correct trade_id for LONG/SHORT.
        del _ts
        self._current_trade_id = f"{self.symbol}_{self.tf_label}_{uuid.uuid4().hex[:8]}"
        self._entry_dynamic_floor_applied = dyn_floor
        self._entry_conf_margin = conf - dyn_floor
        self._entry_win_rate_ema = self._win_rate_ema
        self._entry_total_trades = self.total_trades
        self._entry_equity = self.equity
        self._entry_conf_calib_err = self._conf_calib_err_ema
        self._entry_runway_accuracy = self._runway_accuracy_ema

        self._entry_trigger_data = self._build_entry_trigger_data(
            action=action,
            conf=conf,
            gated=gated,
            depth_ratio=depth_ratio,
            half_spread=half_spread,
            bar_open=bar_open,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_close=bar_close,
        )

    def _build_entry_trigger_data(
        self,
        *,
        action: int,
        conf: float,
        gated: list[str],
        depth_ratio: float,
        half_spread: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
    ) -> dict[str, Any]:
        # Mirrors _log_entry_decision reasoning fields for trade_log correlation.
        geom = self.path_geometry.last
        cb_ok = self.circuit_breakers is None or not self.circuit_breakers.is_any_tripped()
        rs_vol_s = self._compute_rs_vol(10)
        rs_vol_l = self._compute_rs_vol(50)
        er10 = self._compute_er(10)
        ret1, ret5, ret20 = self._compute_returns()
        bars_list = list(self.bars)
        prev_close = bars_list[-2][4] if len(bars_list) >= 2 else bar_close
        gap_pts = float(bar_open - prev_close)
        gap_rs = gap_pts / prev_close / rs_vol_s if rs_vol_s > 0 and prev_close > 0 else 0.0
        trig_stats = (
            (self.policy.get_training_stats() or {}).get("trigger") or {}
            if hasattr(self.policy, "get_training_stats")
            else {}
        )
        cb_mult = (
            self.circuit_breakers.get_position_size_multiplier()
            if self.circuit_breakers is not None
            else 1.0
        )
        drawdown_pct = max(0.0, (self.starting_equity - self.equity) / max(abs(self.starting_equity), 1.0))
        regime: str = getattr(self.policy, "current_regime", "UNKNOWN") or "UNKNOWN"
        return {
            "entry_regime": regime,
            "entry_feasibility": float(geom.get("feasibility", 0.0) or 0.0),
            "entry_zeta": float(getattr(self.policy, "current_zeta", 1.0) or 1.0),
            "entry_geom_efficiency": geom.get("efficiency", 0.0),
            "entry_geom_runway": geom.get("runway", 0.5),
            "entry_geom_gamma": geom.get("gamma", 0.0),
            "entry_geom_jerk": geom.get("jerk", 0.0),
            "entry_cb_ok": cb_ok,
            "entry_gated_conditions": gated,
            "entry_depth_ratio": depth_ratio,
            "entry_depth_bid": self._last_depth_bid,
            "entry_depth_ask": self._last_depth_ask,
            "entry_has_real_l2_sizes": self._has_real_sizes,
            "entry_l2_snapshot": self._last_l2_snapshot,
            "entry_kurtosis": self._last_kurtosis,
            "entry_kurtosis_threshold": self._active_kurtosis_threshold(),
            "entry_rs_vol_short": rs_vol_s,
            "entry_rs_vol_long": rs_vol_l,
            "entry_rs_vol_ratio": rs_vol_s / rs_vol_l if rs_vol_l > 0 else 1.0,
            "entry_gap_rs": gap_rs,
            "entry_half_spread": half_spread,
            "entry_training_steps": int(trig_stats.get("training_steps", 0)),
            "entry_epsilon": float(trig_stats.get("epsilon", 1.0)),
            "entry_er10": er10,
            "entry_ret1": ret1,
            "entry_ret5": ret5,
            "entry_ret20": ret20,
            "entry_alignment_score": self._alignment_score(action, ret1, ret5, ret20),
            "entry_bars_since_energy_bar": self._bars_since_energy_bar(rs_vol_s),
            "entry_hmm_probs": self._get_hmm_probs(),
            "entry_bar_open": float(bar_open),
            "entry_bar_high": float(bar_high),
            "entry_bar_low": float(bar_low),
            "entry_bar_close": float(bar_close),
            "entry_cb_size_mult": cb_mult,
            "entry_drawdown_pct": drawdown_pct,
            "entry_confidence": conf,
            "entry_vpin_z": self._vpin_z,
            "entry_shadow_gates": dict(getattr(getattr(self.policy, "trigger", None), "last_shadow_gates", {}) or {}),
        }

    def _handle_flat(self, bar: tuple, half_spread: float) -> None:
        ts, _bar_open, _bar_high, _bar_low, bar_close = bar

        if self._entry_circuit_breaker_blocks():
            return

        depth_floor = getattr(self.friction_calc, "depth_buffer", 0.0)
        if self._entry_depth_gate_blocks(depth_floor):
            return

        vol = self._realized_vol()
        gated = self._entry_soft_gates()
        depth_ratio = self._depth_ratio()
        decision = self._decide_flat_entry(vol, depth_ratio)
        if decision is None:
            return
        action, conf, runway, trig_state = decision

        action, dyn_floor = self._apply_entry_dynamic_floor(action, conf, gated)
        action = self._apply_paper_entry_guard(action, conf, gated)

        live_mid, live_ts, drift_frac, age_s = self._entry_execution_context(ts, bar_close)
        self._entry_exec_drift_frac = drift_frac
        self._entry_exec_age_s = age_s
        if action != 0 and self._entry_staleness_blocks(drift_frac, age_s, vol):
            LOG.info(
                "[%s %s] Entry blocked (stale): drift=%.4f%% age=%.1fs bar_close=%.5f live_mid=%.5f",
                self.symbol, self.tf_label, drift_frac * 100.0, age_s, bar_close, live_mid,
            )
            gated = [*gated, "entry_stale"]
            action = 0

        if action != 0:
            self._snapshot_entry_lifecycle(
                action=action,
                conf=conf,
                dyn_floor=dyn_floor,
                gated=gated,
                depth_ratio=depth_ratio,
                half_spread=half_spread,
                bar=bar,
            )

        self._log_entry_decision(action, conf, runway, bar_close, vol, depth_ratio, half_spread, bar, gated)

        if action == 0:
            self._maybe_add_no_entry_experience(trig_state)
            return

        direction = 1 if action == 1 else -1
        fill_price = live_mid + direction * half_spread
        self._open_position(live_ts, direction, fill_price, conf, conf, action)

    def _entry_execution_context(
        self, bar_ts: dt.datetime, bar_close: float,
    ) -> tuple[float, dt.datetime, float, float]:
        """Resolve the live execution price/time for a flat entry.

        Entries are decided on completed-bar features but a real fill happens at
        the current market, so execution uses the live mid/timestamp rather than
        the (possibly stale) just-closed bar close. Returns
        (live_mid, live_ts, drift_frac, age_seconds).
        """
        live_mid = float(getattr(self, "last_mid", 0.0) or 0.0)
        if live_mid <= 0:
            live_mid = bar_close
        live_ts = getattr(self, "last_ts", None) or bar_ts
        drift_frac = abs(live_mid - bar_close) / bar_close if bar_close > 0 else 0.0
        try:
            age_s = max(0.0, (live_ts - bar_ts).total_seconds())
        except Exception:
            age_s = 0.0
        return live_mid, live_ts, drift_frac, age_s

    def _entry_staleness_blocks(self, drift_frac: float, age_s: float, vol: float) -> bool:
        """Return True when live execution context is too stale for a flat entry."""
        max_drift = _ENTRY_STALE_DRIFT_VOL_MULT * max(float(vol), 0.0)
        if max_drift > 0 and drift_frac > max_drift:
            return True
        max_age_s = _ENTRY_STALE_MAX_AGE_TF_MULT * self.timeframe_minutes * 60.0
        return age_s > max_age_s

    def _force_close_price(self, mid: float, half_spread: float) -> float:
        if self.position is None:
            return mid
        return mid - self.position["direction"] * half_spread

    def _set_harvester_close_reason(self, reason: str) -> None:
        harvester = getattr(getattr(self, "policy", None), "harvester", None)
        if harvester is not None:
            harvester.last_close_reason = reason

    def _unrealized_pnl_usd(self, mid: float, pos: dict[str, Any]) -> float:
        return (mid - pos["entry_price"]) * pos["direction"] * pos["qty"] * self.contract_size

    def _exit_non_finite_guard(self, ts: dt.datetime, mid: float, half_spread: float, pos: dict[str, Any]) -> bool:
        unrealized = self._unrealized_pnl_usd(mid, pos)
        if math.isfinite(unrealized):
            return False
        LOG.error(
            "[%s %s] Non-finite unrealized P&L: %.4f — force close",
            self.symbol,
            self.tf_label,
            unrealized,
        )
        self._set_harvester_close_reason("non_finite_pnl")
        self._close_position(ts, self._force_close_price(mid, half_spread))
        return True

    def _exit_risk_metrics(self, pos: dict[str, Any]) -> tuple[float, float, float]:
        from src.constants import MAX_CAP_USD, MAX_LOSS_MULT_PER_TRADE, MIN_CAP_USD, STOP_LOSS_PCT_DEFAULT

        lot_value = pos["qty"] * self.contract_size
        rr_risk_usd = pos["entry_price"] * (STOP_LOSS_PCT_DEFAULT / 100.0) * lot_value
        max_loss_usd = max(min(rr_risk_usd * MAX_LOSS_MULT_PER_TRADE, MAX_CAP_USD), MIN_CAP_USD)
        return lot_value, rr_risk_usd, max_loss_usd

    def _exit_max_loss_guard(
        self,
        ts: dt.datetime,
        mid: float,
        half_spread: float,
        pos: dict[str, Any],
        rr_risk_usd: float,
        max_loss_usd: float,
    ) -> bool:
        from src.constants import MAX_LOSS_MULT_PER_TRADE

        unrealized = self._unrealized_pnl_usd(mid, pos)
        if unrealized >= -max_loss_usd:
            return False
        LOG.warning(
            "[%s %s] Max-loss cap: unrealized=%.2f < -%.2f (%.1fR, 1R=%.2f) — force close",
            self.symbol,
            self.tf_label,
            unrealized,
            max_loss_usd,
            MAX_LOSS_MULT_PER_TRADE,
            rr_risk_usd,
        )
        self._set_harvester_close_reason("max_loss_cap")
        self._close_position(ts, self._force_close_price(mid, half_spread))
        return True

    def _exit_rr_floor_guard(
        self,
        ts: dt.datetime,
        mid: float,
        half_spread: float,
        pos: dict[str, Any],
        lot_value: float,
        rr_risk_usd: float,
    ) -> bool:
        from src.constants import MIN_HOLD_TICKS_DEFAULT

        ticks_held = int(getattr(self.policy, "ticks_held", 0))
        if ticks_held <= MIN_HOLD_TICKS_DEFAULT:
            return False
        mfe_usd = float(getattr(self.policy, "mfe", 0.0)) * lot_value
        mfe_usd = mfe_usd if math.isfinite(mfe_usd) else 0.0
        if mfe_usd < rr_risk_usd:
            return False
        pnl_floor = mfe_usd - rr_risk_usd
        unrealized = self._unrealized_pnl_usd(mid, pos)
        if unrealized >= pnl_floor:
            return False
        LOG.warning(
            "[%s %s] R:R floor: MFE=%.2f (%.1fR) pnl=%.2f < floor=%.2f — close",
            self.symbol,
            self.tf_label,
            mfe_usd,
            mfe_usd / max(rr_risk_usd, 0.01),
            unrealized,
            pnl_floor,
        )
        self._set_harvester_close_reason("rr_floor")
        self._close_position(ts, self._force_close_price(mid, half_spread))
        return True

    def _exit_guard_blocks(self, ts: dt.datetime, mid: float, half_spread: float, pos: dict[str, Any]) -> bool:
        if self._exit_non_finite_guard(ts, mid, half_spread, pos):
            return True
        lot_value, rr_risk_usd, max_loss_usd = self._exit_risk_metrics(pos)
        if self._exit_max_loss_guard(ts, mid, half_spread, pos, rr_risk_usd, max_loss_usd):
            return True
        return self._exit_rr_floor_guard(ts, mid, half_spread, pos, lot_value, rr_risk_usd)

    def _decide_exit_on_tick(self, mid: float, depth_ratio: float) -> tuple[int, float] | None:
        depth_ratio = self._depth_ratio()
        try:
            exit_action, exit_conf = self.policy.decide_exit(
                self.bars,
                mid,
                imbalance=self._entry_imbalance,
                vpin_z=self._vpin_z,
                depth_ratio=depth_ratio,
                event_features=self._last_event_feats or None,
            )
        except Exception as e:
            LOG.debug("[%s %s] decide_exit error: %s", self.symbol, self.tf_label, e)
            return None

        self._last_harvester_conf = float(exit_conf)
        return int(exit_action), float(exit_conf)

    def _apply_exit_conf_floor(self, exit_action: int, exit_conf: float) -> int:
        # Exit confidence floor — suppress low-confidence exit signals when the agent
        # is poorly calibrated, matching legacy _obc_get_exit_action() floor gate.
        if exit_action != 1 or self._exit_conf_dynamic_floor <= 0.0:
            return exit_action
        if exit_conf >= self._exit_conf_dynamic_floor:
            return exit_action
        LOG.debug(
            "[%s %s] ExitFloor block: exit_conf=%.3f < floor=%.3f",
            self.symbol,
            self.tf_label,
            exit_conf,
            self._exit_conf_dynamic_floor,
        )
        return 0

    def _log_tick_close_decision(self, exit_conf: float, mid: float, pos: dict[str, Any]) -> None:
        harv_state = getattr(self.policy.harvester, "last_state", None)
        pos_metrics = self.policy.get_position_metrics() if hasattr(self.policy, "get_position_metrics") else {}
        mfe = float(pos_metrics.get("mfe", 0.0))
        mae = float(pos_metrics.get("mae", 0.0))
        ticks = int(pos_metrics.get("ticks_held", 0))
        entry_price = float(pos_metrics.get("entry_price", pos["entry_price"]))
        unrealized = (mid - entry_price) * pos["direction"]
        capture_ratio = unrealized / mfe if mfe > SAFE_EPSILON else 0.0
        harvester = getattr(self.policy, "harvester", None)
        mfe_pct = (mfe / max(abs(entry_price), 1.0)) * 100.0
        trail_activation = getattr(harvester, "trailing_stop_activation_pct", 0.25)
        breakeven_trigger = getattr(harvester, "breakeven_trigger_pct", 0.30)
        capture_decay_min = getattr(harvester, "capture_decay_min_mfe_pct", 0.10)
        try:
            self.decision_log.log_harvester_decision(
                decision="CLOSE",
                confidence=float(exit_conf),
                price=mid,
                entry_price=entry_price,
                mfe=mfe,
                mae=mae,
                ticks_held=ticks,
                unrealized_pnl=unrealized,
                capture_ratio=float(capture_ratio),
                trade_id=self._current_trade_id,
                in_position=True,
                regime=str(getattr(self.policy, "current_regime", "UNKNOWN") or "UNKNOWN"),
                realized_vol=self._realized_vol(),
                depth_ratio=self._depth_ratio(),
                exit_floor=self._exit_conf_dynamic_floor,
                trailing_stop_active=mfe_pct >= trail_activation,
                trailing_stop_activation_pct=trail_activation,
                trailing_stop_distance_pct=getattr(harvester, "trailing_stop_distance_pct", 0.12),
                breakeven_active=mfe_pct >= breakeven_trigger,
                breakeven_trigger_pct=breakeven_trigger,
                capture_decay_armed=mfe_pct >= capture_decay_min,
                capture_decay_threshold=float(getattr(harvester, "capture_decay_threshold", 0.35)),
                close_reason=str(getattr(harvester, "last_close_reason", "") or ""),
            )
        except Exception as e:
            LOG.debug("[%s %s] decision_log error: %s", self.symbol, self.tf_label, e)
        self._exit_state = harv_state.copy() if harv_state is not None else None

    def _handle_exit_on_tick(self, ts: dt.datetime, mid: float, half_spread: float) -> None:
        """Full harvester exit pipeline, called on every price tick while in position."""
        if self.position is None:
            return

        pos = self.position
        if self._exit_guard_blocks(ts, mid, half_spread, pos):
            return

        decision = self._decide_exit_on_tick(mid, self._depth_ratio())
        if decision is None:
            return
        exit_action, exit_conf = decision
        exit_action = self._apply_exit_conf_floor(exit_action, exit_conf)
        if exit_action != 1:
            return

        self._log_tick_close_decision(exit_conf, mid, pos)
        self._close_position(ts, self._force_close_price(mid, half_spread))

    # ---- reward helpers --------------------------------------------------

    def _calculate_trigger_reward(
        self,
        mfe: float,
        pnl_pts: float,
        entry_price: float,
        predicted_runway_net: float,
        realized_vol: float,
    ) -> float:
        """Four-component trigger reward: accuracy + magnitude - false_positive - toxic_flow.

        Ported verbatim from legacy ctrader_ddqn_paper._calculate_trigger_reward().
        Returns reward in [-1.5, 1.5]; instrument-agnostic via σ-normalisation.
        """
        if realized_vol <= 0:
            realized_vol = 0.005
        entry_price_val = max(abs(entry_price), 1.0)
        lot_value = max(self.qty * self.contract_size, 1.0)
        vol_pts = max(realized_vol * entry_price_val, SAFE_SMALL)
        predicted_net_pts = max(0.0, float(predicted_runway_net or 0.0)) * entry_price_val
        norm_mfe = mfe / vol_pts
        norm_predicted = predicted_net_pts / vol_pts

        # Component 1: prediction accuracy  [-1, +1]
        prediction_error = abs(norm_mfe - norm_predicted)
        max_error = max(norm_mfe, norm_predicted, 1.0)
        accuracy_reward = (1.0 - prediction_error / max_error) * 2.0 - 1.0

        # Component 2: magnitude bonus  [0, +0.5]
        magnitude_bonus = min(norm_mfe / 3.0, 1.0) * 0.5

        # Component 3: false positive penalty  [0, -0.7]
        false_positive_penalty = 0.0
        if float(predicted_runway_net or 0.0) > 0 and pnl_pts < 0:
            pnl_pts_scaled = pnl_pts / lot_value if lot_value > 0 else pnl_pts
            loss_severity = min(abs(pnl_pts_scaled) / vol_pts / 3.0, 1.0)
            false_positive_penalty = -0.2 - 0.5 * loss_severity

        # Component 4: toxic flow penalty  [0, -0.30]
        _vpin_thr = float(self._param_manager.get(
            self.symbol, "vpin_z_threshold",
            timeframe=self.tf_label, broker="default", default=2.5) or 2.5)
        entry_vpin = abs(self._entry_vpin_z)
        toxic_penalty = 0.0
        if entry_vpin > _vpin_thr * 0.75 and _vpin_thr > 0:
            _excess = max(entry_vpin - _vpin_thr * 0.75, 0.0) / _vpin_thr
            toxic_penalty = -0.15 * min(_excess, 2.0)

        return float(np.clip(
            accuracy_reward + magnitude_bonus + false_positive_penalty + toxic_penalty,
            -1.5, 1.5,
        ))

    # ---- paper fill simulation ------------------------------------------

    def _log_transaction_event(
        self,
        event_type: str,
        data: dict,
        severity: str = "INFO",
    ) -> None:
        """Write scoped OpenAPI paper lifecycle events for later reconstruction."""
        tx = getattr(self, "transaction_log", None)
        if tx is None:
            return
        payload = {
            "symbol": self.symbol,
            "timeframe": self.tf_label,
            "timeframe_minutes": self.timeframe_minutes,
            "trading_mode": "paper",
            "trade_id": self._current_trade_id,
            **data,
        }
        try:
            tx.log_event(event_type, payload, severity=severity)
        except Exception as exc:
            LOG.debug("[%s %s] transaction_log error: %s", self.symbol, self.tf_label, exc)

    def _current_exit_lifecycle_data(
        self,
        *,
        entry_price: float,
        fill_price: float,
        pnl_pts: float,
        pnl_usd: float,
        mfe: float,
        mae: float,
        quantity: float,
        capture_ratio: float,
        ticks_held: int,
        close_reason: str,
        cb_tripped: list[str],
        close_drawdown_pct: float,
        close_cb_size_mult: float,
    ) -> dict:
        """Snapshot exit-side state that explains close quality and self-healing context."""
        harv = getattr(getattr(self, "policy", None), "harvester", None)
        entry_price = float(entry_price or fill_price)
        lot_value = float(quantity or 0.0) * float(getattr(self, "contract_size", 1.0) or 1.0)
        mfe_usd = float(mfe) * lot_value
        mae_usd = float(mae) * lot_value
        mfe_pct = (mfe / max(abs(entry_price), 1.0)) * 100.0
        return {
            "exit_confidence": float(getattr(self, "_last_harvester_conf", 0.0) or 0.0),
            "exit_dynamic_floor": float(getattr(self, "_exit_conf_dynamic_floor", 0.0) or 0.0),
            "exit_conf_margin": float(getattr(self, "_last_harvester_conf", 0.0) or 0.0)
            - float(getattr(self, "_exit_conf_dynamic_floor", 0.0) or 0.0),
            "exit_price": float(fill_price),
            "exit_mid": float(getattr(self, "last_mid", fill_price) or fill_price),
            "exit_half_spread": float(getattr(self, "last_half_spread", 0.0) or 0.0),
            "exit_pnl_points": float(pnl_pts),
            "exit_pnl_usd": float(pnl_usd),
            "exit_mfe_usd": float(mfe_usd),
            "exit_mae_usd": float(mae_usd),
            "exit_mfe_points": float(mfe),
            "exit_mae_points": float(mae),
            "exit_mfe_pct": float(mfe_pct),
            "exit_capture_ratio": float(capture_ratio),
            "exit_ticks_held": int(ticks_held),
            "exit_close_reason": close_reason,
            "exit_regime": str(getattr(self.policy, "current_regime", "UNKNOWN") or "UNKNOWN"),
            "exit_zeta": float(getattr(self.policy, "current_zeta", 1.0) or 1.0),
            "exit_realized_vol": float(self._realized_vol()),
            "exit_depth_ratio": float(self._depth_ratio()),
            "exit_depth_bid": float(getattr(self, "_last_depth_bid", 0.0) or 0.0),
            "exit_depth_ask": float(getattr(self, "_last_depth_ask", 0.0) or 0.0),
            "exit_has_real_l2_sizes": bool(getattr(self, "_has_real_sizes", False)),
            "exit_l2_snapshot": getattr(self, "_last_l2_snapshot", {}) or {},
            "exit_imbalance": float(getattr(self, "_entry_imbalance", 0.0) or 0.0),
            "exit_vpin_z": float(getattr(self, "_vpin_z", 0.0) or 0.0),
            "exit_var_95": float(getattr(self, "_last_var_95", 0.0) or 0.0),
            "exit_kurtosis": float(getattr(self, "_last_kurtosis", 0.0) or 0.0),
            "exit_kurtosis_threshold": float(self._active_kurtosis_threshold()),
            "exit_trailing_stop_active": bool(
                mfe_pct >= float(getattr(harv, "trailing_stop_activation_pct", 0.25) or 0.25),
            ),
            "exit_trailing_stop_activation_pct": float(
                getattr(harv, "trailing_stop_activation_pct", 0.25) or 0.25,
            ),
            "exit_trailing_stop_distance_pct": float(
                getattr(harv, "trailing_stop_distance_pct", 0.12) or 0.12,
            ),
            "exit_breakeven_active": bool(
                mfe_pct >= float(getattr(harv, "breakeven_trigger_pct", 0.30) or 0.30),
            ),
            "exit_breakeven_trigger_pct": float(getattr(harv, "breakeven_trigger_pct", 0.30) or 0.30),
            "exit_capture_decay_armed": bool(
                mfe_pct >= float(getattr(harv, "capture_decay_min_mfe_pct", 0.10) or 0.10),
            ),
            "exit_capture_decay_threshold": float(getattr(harv, "capture_decay_threshold", 0.35) or 0.35),
            "exit_cb_tripped": list(cb_tripped or []),
            "exit_cb_size_mult": float(close_cb_size_mult),
            "exit_drawdown_pct": float(close_drawdown_pct),
        }

    def _open_position(
        self,
        ts: dt.datetime,
        direction: int,
        fill_price: float,
        conf: float,
        raw_conf: float,
        action: int,
    ) -> None:
        size_mult = (
            self.circuit_breakers.get_position_size_multiplier()
            if self.circuit_breakers is not None else 1.0
        )
        effective_qty = max(self.qty * size_mult, 0.0)
        if effective_qty <= 0:
            LOG.warning("[%s %s] Position blocked — CB size multiplier=0", self.symbol, self.tf_label)
            return
        if size_mult < 1.0:
            LOG.warning("[%s %s] Position size reduced to %.0f%% (CB drawdown mult)",
                        self.symbol, self.tf_label, size_mult * 100)
        self.position = {
            "direction": direction,
            "entry_price": fill_price,
            "entry_time": ts,
            "qty": effective_qty,
        }
        self._entry_conf = conf
        self._entry_raw_conf = raw_conf
        self._entry_action = action

        # Snapshot bars for training cache
        with contextlib.suppress(Exception):
            self.bar_cache.snapshot_entry(self.bars)

        # Notify DualPolicy
        try:
            self.policy.on_entry(direction, fill_price, ts)
        except Exception as e:
            LOG.debug("[%s %s] policy.on_entry error: %s", self.symbol, self.tf_label, e)

        # Snapshot vol/vpin at entry for close-time regime-adjusted reward
        self._entry_var = self._last_var_95
        self._entry_vpin_z = self._vpin_z

        # Trade ID was set in _handle_flat before _log_entry_decision.
        # Reset dense experience state for this new position
        self._prev_harvester_state = None
        self._prev_mfe = 0.0
        self._prev_mae = 0.0
        self._exit_lifecycle_data = {}

        dir_label = "LONG" if direction == 1 else "SHORT"
        self._log_transaction_event(
            "POSITION_OPEN",
            {
                "position_id": self._current_trade_id,
                "direction": dir_label,
                "quantity": effective_qty,
                "entry_price": fill_price,
                "entry_time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "entry_confidence": conf,
                "entry_raw_confidence": raw_conf,
                "entry_action": action,
                "entry_trigger_data": self._entry_trigger_data,
            },
        )

        LOG.info("[%s %s] OPEN %s @ %.5f | conf=%.2f",
                 self.symbol, self.tf_label, dir_label, fill_price, conf)

    # Fraction of NO_ENTRY bars sampled into the trigger buffer once seeded.
    # Matches legacy ctrader_ddqn_paper.py EXPLORATION_SAMPLE_RATE.
    _NO_ENTRY_SAMPLE_RATE: float = 0.05
    # Cap NO_ENTRY share of the buffer to preserve action-class balance.
    _NO_ENTRY_MAX_RATIO: float = 0.45
    _NO_ENTRY_MIN_ENTRY_SAMPLES: int = 20

    def _trigger_no_entry_saturated(self) -> bool:
        """True when NO_ENTRY samples already occupy ≥45% of the trigger replay buffer."""
        trig_buf = getattr(getattr(self.policy, "trigger", None), "buffer", None)
        if trig_buf is None or getattr(trig_buf, "size", 0) <= 0:
            return False
        data = getattr(trig_buf, "data", None)
        if not data:
            return False
        counts = {0: 0, 1: 0, 2: 0}
        for exp in data:
            if exp is None:
                continue
            a = getattr(exp, "action", None)
            if a in counts:
                counts[int(a)] += 1
        total = sum(counts.values())
        entry_count = counts[1] + counts[2]
        if total <= 0 or entry_count < self._NO_ENTRY_MIN_ENTRY_SAMPLES:
            return False
        return (counts[0] / total) >= self._NO_ENTRY_MAX_RATIO

    def _maybe_add_no_entry_experience(self, trig_state: Any) -> None:
        """Sampled NO_ENTRY experience so the trigger buffer stays class-balanced.

        100% capture during warm-up (buffer below min_experiences), then throttled
        to _NO_ENTRY_SAMPLE_RATE (5%) once seeded. Mirrors legacy paper DDQN behaviour.

        Reward is the activity monitor's inactivity penalty when the bot has been flat
        past max_bars_inactive — a small negative signal that discourages learned
        helplessness (the bot learning that doing nothing is optimal).
        """
        if trig_state is None:
            return
        import random
        trig = getattr(self.policy, "trigger", None)
        if trig is None:
            return
        trig_buf = getattr(trig, "buffer", None)
        buf_size = getattr(trig_buf, "size", 0) if trig_buf is not None else 0
        min_exp = getattr(trig, "min_experiences", 32)
        sample_rate = 1.0 if buf_size < min_exp else self._NO_ENTRY_SAMPLE_RATE
        if self._trigger_no_entry_saturated():
            return
        if random.random() >= sample_rate:
            return
        _am = getattr(self.reward_shaper, "activity_monitor", None)
        no_entry_reward = _am.get_inactivity_penalty() if _am is not None else 0.0
        try:
            self.policy.add_trigger_experience(
                state=trig_state,
                action=0,
                reward=no_entry_reward,
                next_state=trig_state,
                done=True,
            )
        except Exception as e:
            LOG.debug("[%s %s] no_entry_experience error: %s", self.symbol, self.tf_label, e)

    def _add_replay_experiences(self, trigger_reward: float, capture_reward: float) -> None:
        if self._entry_state is None:
            return
        # Resolve harvester exit state (window, harvester_features) separately from
        # _entry_state which is trigger-shaped (window, trigger_features).  Mixing
        # them causes a shape mismatch in the respective DDQNNetwork linear layers.
        harv_exit = self._exit_state
        if harv_exit is None:
            _harv_last = getattr(getattr(self.policy, "harvester", None), "last_state", None)
            if _harv_last is not None:
                harv_exit = _harv_last.copy()
        try:
            # Trigger experience: both state and next_state are trigger-shaped.
            # done=True → next_state is zeroed out in the Bellman target, but the
            # network still forward-passes it so shapes must match state_dim.
            self.policy.add_trigger_experience(
                state=self._entry_state, action=self._entry_action,
                reward=trigger_reward, next_state=self._entry_state, done=True,
            )
            # Harvester experience: skip when no harvester state exists (very short
            # trades where the harvester never computed a state) to avoid injecting
            # trigger-shaped (window, 18) data into a buffer that expects (window, 21).
            if harv_exit is not None:
                self.policy.add_harvester_experience(
                    state=harv_exit, action=1,
                    reward=capture_reward, next_state=harv_exit, done=True,
                )
        except Exception as e:
            LOG.debug("[%s %s] add_experience error: %s", self.symbol, self.tf_label, e)
        self._entry_state = None
        self._exit_state = None
        self._prev_harvester_state = None
        self._prev_mfe = 0.0
        self._prev_mae = 0.0

    # Minimum trades before adaptive floor nudging activates.
    _RISK_TUNER_MIN_TRADES: dict = {1: 30, 5: 20, 15: 15, 30: 10, 60: 8, 240: 6}
    # Save interval for learned params (every N trades)
    _RISK_TUNER_SAVE_INTERVAL: int = 10
    # Stable-base clamp range — the cached base floor is bounded to this band so a
    # previously-runaway persisted value cannot seed an ever-rising adaptive cap.
    _RISK_TUNER_BASE_MIN: float = 0.30
    _RISK_TUNER_BASE_MAX: float = 0.60
    # Hard absolute cap on the adaptive entry floor regardless of base — the tuner can
    # raise the floor by at most +0.10 above base and never beyond this ceiling.
    _RISK_TUNER_FLOOR_ABS_CAP: float = 0.70

    def _update_risk_feedback_thresholds(self, pnl_usd: float) -> None:
        """Simplified adaptive floor tuner — no full RiskManager required.

        Tracks a per-TF win-rate EMA and nudges _entry_conf_dynamic_floor up/down:
        - Win rate consistently < 40% → raise floor by 0.02 (cap at min(base + 0.10, abs cap))
        - Win rate consistently > 65% → lower floor by 0.01 (floor at 0.0 → reverts to base)
        The base floor is cached once (stable) so the adaptive cap cannot ratchet upward
        across restarts/persistence. Persists both floors every _RISK_TUNER_SAVE_INTERVAL trades.
        """
        try:
            win = pnl_usd > 0.0
            _alpha_wr = 0.15
            self._win_rate_ema = (1.0 - _alpha_wr) * self._win_rate_ema + _alpha_wr * (1.0 if win else 0.0)
            self._win_rate_ema_n += 1

            min_trades = self._RISK_TUNER_MIN_TRADES.get(self.timeframe_minutes, 10)
            if self._win_rate_ema_n < min_trades:
                return

            if self._risk_tuner_base_floor is None:
                _persisted = float(self._param_manager.get(
                    self.symbol, "entry_confidence_threshold",
                    timeframe=self.tf_label, broker="default", default=0.55) or 0.55)
                self._risk_tuner_base_floor = float(min(
                    max(_persisted, self._RISK_TUNER_BASE_MIN),
                    self._RISK_TUNER_BASE_MAX,
                ))
            _base_floor = self._risk_tuner_base_floor
            _floor_max = min(_base_floor + 0.10, self._RISK_TUNER_FLOOR_ABS_CAP)
            _floor_min = 0.0

            if self._win_rate_ema < 0.40:
                self._entry_conf_dynamic_floor = min(
                    self._entry_conf_dynamic_floor + 0.02, _floor_max,
                )
                LOG.info(
                    "[%s %s] RiskTuner: win_rate=%.1f%% < 40%% → raise floor %.3f → %.3f",
                    self.symbol, self.tf_label, self._win_rate_ema * 100,
                    self._entry_conf_dynamic_floor - 0.02, self._entry_conf_dynamic_floor,
                )
            elif self._win_rate_ema > 0.65:
                self._entry_conf_dynamic_floor = max(
                    self._entry_conf_dynamic_floor - 0.01, _floor_min,
                )

            # DDQN exit floor: track win rate specifically for ddqn_model exits and
            # adapt the exit confidence floor independently of overall trade win rate.
            _close_reason = getattr(getattr(self.policy, "harvester", None), "last_close_reason", "") or ""
            if _close_reason == "ddqn_model":
                _alpha_ex = 0.15
                self._ddqn_exit_win_ema = (1.0 - _alpha_ex) * self._ddqn_exit_win_ema + _alpha_ex * (
                    1.0 if win else 0.0
                )
                self._ddqn_exit_n += 1
                if self._ddqn_exit_n >= 5:
                    if self._ddqn_exit_win_ema < 0.30:
                        _old_floor = self._exit_conf_dynamic_floor
                        self._exit_conf_dynamic_floor = min(self._exit_conf_dynamic_floor + 0.02, 0.80)
                        if self._exit_conf_dynamic_floor != _old_floor:
                            LOG.info(
                                "[%s %s] ExitTuner: ddqn_exit_wr=%.1f%% < 30%% → raise exit floor %.3f → %.3f",
                                self.symbol, self.tf_label, self._ddqn_exit_win_ema * 100,
                                _old_floor, self._exit_conf_dynamic_floor,
                            )
                    elif self._ddqn_exit_win_ema > 0.55:
                        self._exit_conf_dynamic_floor = max(self._exit_conf_dynamic_floor - 0.01, 0.0)

            # Persist floors periodically
            if self.total_trades % self._RISK_TUNER_SAVE_INTERVAL == 0:
                try:
                    self._param_manager.set_value(
                        self.symbol, "entry_confidence_threshold",
                        float(max(_base_floor, self._entry_conf_dynamic_floor)),
                        timeframe=self.tf_label, broker="default")
                    self._param_manager.set_value(
                        self.symbol, "exit_confidence_threshold",
                        float(max(0.0, self._exit_conf_dynamic_floor)),
                        timeframe=self.tf_label, broker="default")
                    self._param_manager.save()
                except Exception:
                    pass
        except Exception as e:
            LOG.debug("[%s %s] risk_feedback error: %s", self.symbol, self.tf_label, e)

    def _update_learned_params(self, pnl_usd: float) -> None:
        try:
            grad = float(np.clip(pnl_usd / max(abs(self.starting_equity) * 0.01, 1.0), -1.0, 1.0))
            for _pname in ("sortino_threshold", "kurtosis_threshold", "max_drawdown_pct"):
                self._param_manager.update(self.symbol, _pname, grad, timeframe=self.tf_label, broker="default")
            self._param_manager.save()
        except Exception as e:
            LOG.debug("[%s %s] param_manager.update error: %s", self.symbol, self.tf_label, e)

    # ── Position close ────────────────────────────────────────────────────────

    def _detach_position_for_close(self) -> tuple[dict[str, Any] | None, str | None]:
        pos = self.position
        if pos is None:
            return None, None
        self.position = None
        return pos, self._current_trade_id

    def _close_pnl(self, fill_price: float, entry_price: float, direction: int, qty: float) -> tuple[float, float]:
        pnl_pts = (fill_price - entry_price) * direction
        if not math.isfinite(pnl_pts):
            LOG.error(
                "[%s %s] Non-finite pnl_pts: fill=%.2f entry=%.2f dir=%d — using 0",
                self.symbol,
                self.tf_label,
                fill_price,
                entry_price,
                direction,
            )
            pnl_pts = 0.0
        pnl_usd = pnl_pts * qty * self.contract_size
        if not math.isfinite(pnl_usd):
            LOG.error(
                "[%s %s] Non-finite pnl_usd: pts=%.4f qty=%.4f cs=%.2f — using 0",
                self.symbol,
                self.tf_label,
                pnl_pts,
                qty,
                self.contract_size,
            )
            pnl_usd = 0.0
        return pnl_pts, pnl_usd

    def _apply_close_accounting(self, pnl_usd: float) -> None:
        self.equity += pnl_usd
        self.total_trades += 1
        self.trades_pnl.append(pnl_usd)
        self._last_trade_close_ts = time.time()

        if self.circuit_breakers is not None:
            self.circuit_breakers.update_trade(pnl_usd, self.equity)
        with contextlib.suppress(Exception):
            self.reward_shaper.activity_monitor.on_trade_executed()

    def _close_mfe_mae(self) -> tuple[float, float]:
        mfe = getattr(self.policy, "mfe", 0.0)
        mae = getattr(self.policy, "mae", 0.0)
        mfe = mfe if math.isfinite(mfe) else 0.0
        mae = mae if math.isfinite(mae) else 0.0
        self._rolling_mfe.append(mfe)
        self._rolling_mae.append(mae)
        return float(mfe), float(mae)

    def _close_reward_context(
        self,
        *,
        pnl_pts: float,
        mfe: float,
        mae: float,
        entry_price: float,
    ) -> dict[str, Any]:
        entry_half_spread = float(
            (self._entry_trigger_data or {}).get("entry_half_spread", self.last_half_spread) or 0.0,
        )
        reward_spread_cost_pts = entry_half_spread + float(self.last_half_spread or 0.0)
        reward_net_pnl_pts = pnl_pts - reward_spread_cost_pts
        capture_ratio = min(1.0, pnl_pts / mfe) if mfe > SAFE_EPSILON else 0.0
        was_wtl = (pnl_pts < 0) and (mfe > abs(mae) * 0.5)
        reward_wtl = (reward_net_pnl_pts < 0) and (mfe > abs(mae) * 0.5)
        try:
            self._check_capture_health(capture_ratio, mfe, entry_price)
        except Exception as e:
            LOG.debug("[%s %s] capture_health error: %s", self.symbol, self.tf_label, e)
        return {
            "capture_ratio": capture_ratio,
            "reward_net_pnl_pts": reward_net_pnl_pts,
            "reward_spread_cost_pts": reward_spread_cost_pts,
            "reward_wtl": reward_wtl,
            "was_wtl": was_wtl,
        }

    def _update_close_runway_emas(
        self,
        entry_price: float,
        mfe: float,
        reward_net_pnl_pts: float,
    ) -> tuple[float, float]:
        runway_gross = float(getattr(self.policy, "predicted_runway", 0.0) or 0.0)
        price_ref = max(abs(entry_price), 1.0)
        predicted_pts = runway_gross * price_ref
        runway_delta = predicted_pts - mfe
        max_err = max(abs(mfe), abs(predicted_pts), 1.0)
        alpha = 0.2
        self._runway_delta_ema = (1 - alpha) * self._runway_delta_ema + alpha * runway_delta
        self._runway_accuracy_ema = (1 - alpha) * self._runway_accuracy_ema + alpha * (
            1.0 - min(abs(runway_delta) / max_err, 1.0)
        )
        brier = (self._entry_conf - (1.0 if reward_net_pnl_pts > 0 else 0.0)) ** 2
        self._conf_calib_err_ema = (1 - alpha) * self._conf_calib_err_ema + alpha * brier
        try:
            self._param_manager.set_value(
                self.symbol, "runway_delta_ema", self._runway_delta_ema,
                timeframe=self.tf_label, broker="default")
            self._param_manager.set_value(
                self.symbol, "runway_accuracy_ema", self._runway_accuracy_ema,
                timeframe=self.tf_label, broker="default")
            self._param_manager.set_value(
                self.symbol, "conf_calib_err_ema", self._conf_calib_err_ema,
                timeframe=self.tf_label, broker="default")
            self._param_manager.save()
        except Exception:
            pass
        entry_half_spread = float(
            (self._entry_trigger_data or {}).get("entry_half_spread", self.last_half_spread) or self.last_half_spread,
        )
        runway_net = max(0.0, runway_gross - (2.0 * entry_half_spread / price_ref))
        return runway_gross, runway_net

    def _initial_close_rewards(
        self,
        *,
        mfe: float,
        pnl_pts: float,
        entry_price: float,
        runway_net: float,
        capture_ratio: float,
    ) -> tuple[float, float]:
        trigger_reward = self._calculate_trigger_reward(
            mfe=mfe,
            pnl_pts=pnl_pts,
            entry_price=entry_price,
            predicted_runway_net=runway_net,
            realized_vol=self._realized_vol(),
        )
        raw_capture = float(np.clip(capture_ratio, -1.0, 1.0))
        vol_cap = float(
            self._param_manager.get(
                self.symbol, "vol_cap", timeframe=self.tf_label, broker="default", default=0.05,
            ) or 0.05,
        )
        vpin_threshold = float(
            self._param_manager.get(
                self.symbol,
                "vpin_z_threshold",
                timeframe=self.tf_label,
                broker="default",
                default=2.5,
            ) or 2.5,
        )
        regime_adj = 0.0
        if self._entry_var > vol_cap:
            regime_adj -= 0.3 * min(self._entry_var / max(vol_cap, SAFE_DIV_MIN), 2.0)
        if abs(self._entry_vpin_z) > vpin_threshold:
            regime_adj -= 0.2 * min(abs(self._entry_vpin_z) / max(vpin_threshold, SAFE_DIV_MIN), 2.0)
        return trigger_reward, float(np.clip(raw_capture + regime_adj, -2.0, 2.0))

    def _notify_policy_exit(self, fill_price: float, capture_ratio: float, was_wtl: bool) -> None:
        try:
            self.policy.on_exit(
                fill_price,
                capture_ratio,
                was_wtl,
                entry_confidence=self._entry_conf,
                raw_confidence=self._entry_raw_conf,
            )
        except Exception as e:
            LOG.debug("[%s %s] policy.on_exit error: %s", self.symbol, self.tf_label, e)

    def _shape_close_rewards(
        self,
        *,
        ts: dt.datetime,
        mfe: float,
        mae: float,
        direction: int,
        entry_price: float,
        pnl_pts: float,
        reward_net_pnl_pts: float,
        reward_wtl: bool,
        bars_held: int,
        runway_net: float,
        trigger_reward: float,
        capture_reward: float,
    ) -> tuple[float, float, dict[str, Any], dict[str, Any]]:
        try:
            shaped = self.reward_shaper.calculate_dual_agent_rewards(
                actual_mfe=mfe,
                predicted_runway=runway_net,
                direction=direction,
                entry_price=entry_price,
                exit_pnl=pnl_pts,
                net_exit_pnl=reward_net_pnl_pts,
                mae=mae,
                was_wtl=reward_wtl,
                bars_held=bars_held,
                exit_time=ts.isoformat() if hasattr(ts, "isoformat") else "",
            )
            shaped_trigger = float(shaped.get("trigger_reward", trigger_reward))
            # Only override the 4-component trigger reward when the log-based runway
            # signal is not saturated at the clamp (±3.0).  A saturated log gives a
            # constant gradient — useless for learning — even when pnl_alignment has
            # shifted the combined value back inside ±2.99.  Use the explicit
            # log_saturated flag from the shaper to detect the pre-pnl_alignment state.
            trigger_breakdown = shaped.get("trigger_breakdown", {})
            log_saturated = bool(trigger_breakdown.get("log_saturated", False))
            if not log_saturated and abs(shaped_trigger) < 2.99:
                trigger_reward = shaped_trigger
            capture_reward = float(shaped.get("harvester_reward", capture_reward))
            return (
                trigger_reward,
                capture_reward,
                shaped.get("trigger_breakdown", {}),
                shaped.get("harvester_breakdown", {}),
            )
        except Exception as e:
            LOG.debug("[%s %s] reward_shaper error: %s", self.symbol, self.tf_label, e)
            return trigger_reward, capture_reward, {}, {}

    def _record_close_training_cache(
        self,
        *,
        direction: int,
        entry_price: float,
        fill_price: float,
        pnl_pts: float,
        mfe: float,
        mae: float,
        trigger_reward: float,
        capture_reward: float,
        regime: str,
    ) -> None:
        del direction
        try:
            self.bar_cache.record_trade(
                bars=self.bars,
                trigger_action=self._entry_action,
                trigger_reward=trigger_reward,
                capture_reward=capture_reward,
                entry_price=entry_price,
                exit_price=fill_price,
                pnl_pts=pnl_pts,
                mfe=mfe,
                mae=mae,
                regime=regime,
                imbalance=self._entry_imbalance,
                vpin_z=self._vpin_z,
                depth_ratio=self._depth_ratio(),
            )
        except Exception as e:
            LOG.debug("[%s %s] bar_cache.record_trade error: %s", self.symbol, self.tf_label, e)

    def _post_close_learning(self, pnl_usd: float, trigger_reward: float, capture_reward: float) -> str:
        regime = str(getattr(self.policy, "current_regime", "UNKNOWN"))
        self._add_replay_experiences(trigger_reward, capture_reward)
        self._update_learned_params(pnl_usd)
        self._update_risk_feedback_thresholds(pnl_usd)
        return regime

    def _close_cb_snapshot(self) -> tuple[list[str], float, float, str]:
        cb_tripped = []
        if self.circuit_breakers is not None:
            with contextlib.suppress(Exception):
                cb_tripped = [
                    k for k, v in self.circuit_breakers.get_status().items()
                    if isinstance(v, dict) and v.get("tripped")
                ]
        close_drawdown = max(0.0, (self.starting_equity - self.equity) / max(abs(self.starting_equity), 1.0))
        close_cb_mult = (
            self.circuit_breakers.get_position_size_multiplier()
            if self.circuit_breakers is not None
            else 1.0
        )
        close_reason = getattr(getattr(self.policy, "harvester", None), "last_close_reason", "") or ""
        return cb_tripped, close_drawdown, close_cb_mult, close_reason

    def _emit_close_transaction(self, ctx: dict[str, Any]) -> None:
        pos = ctx["pos"]
        self._exit_lifecycle_data = self._current_exit_lifecycle_data(
            entry_price=ctx["entry_price"],
            fill_price=ctx["fill_price"],
            pnl_pts=ctx["pnl_pts"],
            pnl_usd=ctx["pnl_usd"],
            mfe=ctx["mfe"],
            mae=ctx["mae"],
            quantity=ctx["qty"],
            capture_ratio=ctx["capture_ratio"],
            ticks_held=ctx["ticks_held"],
            close_reason=ctx["close_reason"],
            cb_tripped=ctx["cb_tripped"],
            close_drawdown_pct=ctx["close_drawdown"],
            close_cb_size_mult=ctx["close_cb_mult"],
        )
        self._log_transaction_event(
            "POSITION_CLOSE",
            {
                "position_id": ctx["trade_id"],
                "direction": ctx["dir_label"],
                "quantity": ctx["qty"],
                "entry_price": ctx["entry_price"],
                "exit_price": ctx["fill_price"],
                "entry_time": pos["entry_time"].isoformat()
                if hasattr(pos.get("entry_time"), "isoformat") else str(pos.get("entry_time")),
                "exit_time": ctx["ts"].isoformat() if hasattr(ctx["ts"], "isoformat") else str(ctx["ts"]),
                "pnl": ctx["pnl_usd"],
                "pnl_points": ctx["pnl_pts"],
                "pnl_net_points": ctx["reward_net_pnl_pts"],
                "reward_spread_cost_points": ctx["reward_spread_cost_pts"],
                "mfe": ctx["mfe"] * ctx["qty"] * float(getattr(self, "contract_size", 1.0) or 1.0),
                "mae": ctx["mae"] * ctx["qty"] * float(getattr(self, "contract_size", 1.0) or 1.0),
                "mfe_points": ctx["mfe"],
                "mae_points": ctx["mae"],
                "capture_ratio": ctx["capture_ratio"],
                "winner_to_loser": ctx["was_wtl"],
                "close_reason": ctx["close_reason"],
                "trigger_reward": ctx["trigger_reward"],
                "capture_reward": ctx["capture_reward"],
                "reward_trigger_breakdown": ctx["reward_breakdown"],
                "reward_harvester_breakdown": ctx["harv_breakdown"],
                "entry_trigger_data": self._entry_trigger_data,
                "exit_data": self._exit_lifecycle_data,
            },
        )

    def _write_close_trade_log_event(self, ctx: dict[str, Any]) -> bool:
        pos = ctx["pos"]
        return self._write_trade_log(
            direction=ctx["direction"],
            entry_price=ctx["entry_price"],
            exit_price=ctx["fill_price"],
            entry_time=pos["entry_time"],
            exit_time=ctx["ts"],
            pnl_usd=ctx["pnl_usd"],
            pnl_pts=ctx["pnl_pts"],
            mfe=ctx["mfe"],
            mae=ctx["mae"],
            quantity=ctx["qty"],
            trigger_reward=ctx["trigger_reward"],
            capture_reward=ctx["capture_reward"],
            regime=ctx["regime"],
            predicted_runway_gross=ctx["runway_gross"],
            predicted_runway_net=ctx["runway_net"],
            was_winner_to_loser=ctx["was_wtl"],
            reward_wtl_net_flag=ctx["reward_wtl"],
            entry_vpin_z=self._entry_vpin_z,
            entry_var_95=self._entry_var,
            capture_ratio=ctx["capture_ratio"],
            diag_cb_active=len(ctx["cb_tripped"]) > 0,
            diag_cb_tripped=ctx["cb_tripped"],
            trade_id=ctx["trade_id"],
            ticks_held=ctx["ticks_held"],
            exit_regime=ctx["regime"],
            exit_vol=self._realized_vol(),
            exit_depth_ratio=self._depth_ratio(),
            entry_dynamic_floor=self._entry_dynamic_floor_applied,
            entry_conf_margin=self._entry_conf_margin,
            win_rate_ema_at_entry=self._entry_win_rate_ema,
            total_trades_at_entry=self._entry_total_trades,
            equity_at_entry=self._entry_equity,
            conf_calib_err_at_entry=self._entry_conf_calib_err,
            runway_accuracy_at_entry=self._entry_runway_accuracy,
            # Reward component breakdown
            reward_capture_efficiency=ctx["reward_breakdown"].get(
                "runway_reward", ctx["reward_breakdown"].get("accuracy", 0.0),
            ),
            reward_wtl_penalty=ctx["harv_breakdown"].get("wtl_penalty", 0.0),
            reward_opportunity_cost=ctx["harv_breakdown"].get("undeveloped_mfe_penalty", 0.0),
            reward_session_quality=ctx["harv_breakdown"].get("session_quality", 1.0),
            reward_harvester_total=ctx["harv_breakdown"].get("harvester_reward", ctx["capture_reward"]),
            reward_trigger_breakdown=ctx["reward_breakdown"],
            reward_harvester_breakdown=ctx["harv_breakdown"],
            # Trigger entry reasoning snapshot
            trigger_data=self._entry_trigger_data,
            # Risk state at close
            close_drawdown_pct=ctx["close_drawdown"],
            close_cb_size_mult=ctx["close_cb_mult"],
            # Exit-side lifecycle reasoning snapshot
            exit_data=self._exit_lifecycle_data,
            close_reason=ctx["close_reason"],
        )

    def _emit_close_lifecycle(self, ctx: dict[str, Any]) -> None:
        self._emit_close_transaction(ctx)
        if self._write_close_trade_log_event(ctx):
            self._current_trade_id = None
        else:
            LOG.error(
                "[%s %s] close lifecycle retained trade_id=%s after trade_log write failure",
                self.symbol,
                self.tf_label,
                ctx["trade_id"],
            )

    def _close_position(self, _ts: dt.datetime, fill_price: float) -> None:
        pos, closed_trade_id = self._detach_position_for_close()
        if pos is None:
            return

        direction = int(pos["direction"])
        entry_price = float(pos["entry_price"])
        qty = float(pos["qty"])
        pnl_pts, pnl_usd = self._close_pnl(fill_price, entry_price, direction, qty)
        self._apply_close_accounting(pnl_usd)

        mfe, mae = self._close_mfe_mae()
        reward_ctx = self._close_reward_context(
            pnl_pts=pnl_pts,
            mfe=mfe,
            mae=mae,
            entry_price=entry_price,
        )
        runway_gross, runway_net = self._update_close_runway_emas(
            entry_price,
            mfe,
            reward_ctx["reward_net_pnl_pts"],
        )
        trigger_reward, capture_reward = self._initial_close_rewards(
            mfe=mfe,
            pnl_pts=reward_ctx["reward_net_pnl_pts"],
            entry_price=entry_price,
            runway_net=runway_net,
            capture_ratio=reward_ctx["capture_ratio"],
        )
        ticks_held = int(getattr(self.policy, "ticks_held", 0))
        self._notify_policy_exit(fill_price, reward_ctx["capture_ratio"], reward_ctx["was_wtl"])
        trigger_reward, capture_reward, reward_breakdown, harv_breakdown = self._shape_close_rewards(
            ts=_ts,
            mfe=mfe,
            mae=mae,
            direction=direction,
            entry_price=entry_price,
            pnl_pts=pnl_pts,
            reward_net_pnl_pts=reward_ctx["reward_net_pnl_pts"],
            reward_wtl=reward_ctx["reward_wtl"],
            bars_held=ticks_held,
            runway_net=runway_net,
            trigger_reward=trigger_reward,
            capture_reward=capture_reward,
        )
        regime = self._post_close_learning(pnl_usd, trigger_reward, capture_reward)
        self._record_close_training_cache(
            direction=direction,
            entry_price=entry_price,
            fill_price=fill_price,
            pnl_pts=pnl_pts,
            mfe=mfe,
            mae=mae,
            trigger_reward=trigger_reward,
            capture_reward=capture_reward,
            regime=regime,
        )

        dir_label = "LONG" if direction == 1 else "SHORT"
        LOG.info(
            "[%s %s] CLOSE %s | pnl=%.2f pts | pnl_usd=%.2f | MFE=%.5f MAE=%.5f | equity=%.2f",
            self.symbol,
            self.tf_label,
            dir_label,
            pnl_pts,
            pnl_usd,
            mfe,
            mae,
            self.equity,
        )
        cb_tripped, close_drawdown, close_cb_mult, close_reason = self._close_cb_snapshot()
        self._emit_close_lifecycle({
            **reward_ctx,
            "cb_tripped": cb_tripped,
            "capture_reward": capture_reward,
            "close_cb_mult": close_cb_mult,
            "close_drawdown": close_drawdown,
            "close_reason": close_reason,
            "dir_label": dir_label,
            "direction": direction,
            "entry_price": entry_price,
            "fill_price": fill_price,
            "harv_breakdown": harv_breakdown,
            "mae": mae,
            "mfe": mfe,
            "pnl_pts": pnl_pts,
            "pnl_usd": pnl_usd,
            "pos": pos,
            "qty": qty,
            "regime": regime,
            "reward_breakdown": reward_breakdown,
            "runway_gross": runway_gross,
            "runway_net": runway_net,
            "ticks_held": ticks_held,
            "trade_id": closed_trade_id,
            "trigger_reward": trigger_reward,
            "ts": _ts,
        })

    # ---- persistence -----------------------------------------------------

    def _save_checkpoint(self) -> None:
        if not hasattr(self.policy, "save_checkpoint"):
            return
        try:
            self.policy.save_checkpoint()
        except Exception as e:
            LOG.warning("[%s %s] save_checkpoint failed: %s", self.symbol, self.tf_label, e)

    def shutdown(self) -> None:
        """Close open position at last mid, save checkpoint."""
        if self.position is not None and self.last_mid > 0:
            LOG.info("[%s %s] Shutdown: closing open position at %.5f",
                     self.symbol, self.tf_label, self.last_mid)
            _harv = getattr(self.policy, "harvester", None)
            if _harv is not None:
                _harv.last_close_reason = "shutdown"
            direction = self.position["direction"]
            fill = self.last_mid - direction * self.last_half_spread
            ts = self.last_ts or dt.datetime.now(dt.UTC)
            self._close_position(ts, fill)
        self._save_checkpoint()
        self._write_telemetry()
        LOG.info("[%s %s] Shutdown complete | trades=%d equity=%.2f",
                 self.symbol, self.tf_label, self.total_trades, self.equity)


# ---------------------------------------------------------------------------
# Paper-mode emergency position closer (used by CircuitBreakerManager)
# ---------------------------------------------------------------------------

class _PaperEmergencyCloser:
    """Fulfils the EmergencyPositionCloser interface for paper trading.

    CircuitBreakerManager calls close_all_positions() when auto_close_on_trip=True
    and a breaker trips. In paper mode this just closes the open simulated position.
    """

    def __init__(self, agent: TFAgent) -> None:
        self._agent = agent

    def close_all_positions(self, reason: str = "CIRCUIT_BREAKER") -> bool:
        agent = self._agent
        if agent.position is not None and agent.last_mid > 0:
            ts = dt.datetime.now(dt.UTC)
            fill = agent.last_mid - agent.position["direction"] * agent.last_half_spread
            try:
                _harv = getattr(getattr(agent, "policy", None), "harvester", None)
                if _harv is not None:
                    _harv.last_close_reason = "circuit_breaker"
                agent._close_position(ts, fill)
                LOG.warning("[EMERGENCY] Paper position closed: %s %s reason=%s",
                            agent.symbol, agent.tf_label, reason)
            except Exception as e:
                LOG.exception("[EMERGENCY] Close failed %s %s: %s", agent.symbol, agent.tf_label, e)
                return False
        return True


# ---------------------------------------------------------------------------
# OpenAPIHub – network + dispatch layer
# ---------------------------------------------------------------------------

class OpenAPIHub:
    """Connects once to cTrader Open API and routes ticks to all TFAgents."""

    def __init__(
        self,
        symbol: str,
        symbol_id: int,
        timeframes: list[int],
        creds: dict[str, str],
        host: str,
        qty: float,
        contract_size: float,
        data_dir_root: Path,
        starting_equity: float,
        online_learning: bool,
    ) -> None:
        self.symbol = symbol
        self.symbol_id = symbol_id
        self.creds = creds
        self.host = host
        self.account_id = int(creds.get("account_id") or 0)

        # Price scaling (digits resolved from SymbolByIdRes; fallback from env)
        self._digits = int(os.environ.get("OPENAPI_DIGITS", str(_DEFAULT_DIGITS)))
        self._scale = 10 ** self._digits

        # Auth state machine
        self._state = _S_CONNECTING
        self._client: Any = None

        # L2 order book (shared across all TFAgents — same underlying market)
        from src.core.order_book import OrderBook
        self._order_book = OrderBook(depth=10)
        self._quote_id_map: dict[int, tuple[str, float]] = {}  # id → (side, price)

        # VPIN: VPINCalculator from order_book.py, fed by mid-price direction.
        # volume proxy = 1.0 per tick; bucket_volume=20; window=50 buckets.
        from src.core.order_book import VPINCalculator
        self._vpin_calc = VPINCalculator(bucket_volume=20.0, window=50)
        self._vpin_last_mid: float = 0.0
        self._vpin_stats: dict = {"vpin": 0.0, "zscore": 0.0}
        self._vpin_z: float = 0.0

        # QFI (Quote Flow Imbalance): bid vs ask refresh counts since last OB write.
        # True if broker ever sends non-zero L2 size (size-weighted imbalance vs QFI fallback).
        self._has_real_sizes: bool = False
        self._bid_refresh_count: int = 0
        self._ask_refresh_count: int = 0

        # Per-TF agents
        self.agents: dict[int, TFAgent] = {}
        for tf in timeframes:
            agent_dir = data_dir_root / f"paper_{symbol}_M{tf}"
            self.agents[tf] = TFAgent(
                symbol=symbol,
                symbol_id=symbol_id,
                timeframe_minutes=tf,
                qty=qty,
                contract_size=contract_size,
                data_dir=agent_dir,
                starting_equity=starting_equity,
                online_learning=online_learning,
            )

        self._tick_count = 0
        self._last_bid_raw: int = 0
        self._last_ask_raw: int = 0
        self._start_time = time.time()

        # Heartbeat watchdog
        self._last_heartbeat: float = time.time()
        self._shutdown_flag: bool = False
        self._reactor: Any = None

        # Gap-fill: record when we lost the connection so we can fetch missed bars on reconnect
        self._disconnect_time: float | None = None
        # Set during token auto-refresh so _handle_subscribe_res skips initial backfill
        self._token_refresh_in_progress: bool = False

        # Order-book write rate limit (write at most once per second)
        self._last_ob_write: float = 0.0
        self._last_l2_prune_log: float = 0.0
        self._l2_prune_suppressed: int = 0

        # Control-file poll thread (kill-switch, CB reset, kurtosis gate reset)
        self._control_poll_thread = threading.Thread(
            target=self._control_poll_fn, daemon=True, name=f"ctrl-{symbol}",
        )
        self._control_poll_thread.start()

        LOG.info("[HUB] %s (id=%d) | TFs=%s | host=%s", symbol, symbol_id, timeframes, host)

    # ---- Twisted callbacks -----------------------------------------------

    def _on_connected(self, client: Any) -> None:
        LOG.info("[HUB] Connected to %s", self.host)
        self._client = client
        self._state = _S_APP_AUTH
        self._last_heartbeat = time.time()
        self._send_app_auth()

    def _on_disconnected(self, client: Any, reason: Any) -> None:
        LOG.warning("[HUB] Disconnected: %s", reason)
        self._state = _S_CONNECTING
        self._disconnect_time = time.time()

    # ---- control-file poll thread ----------------------------------------

    def _ctrl_paths(self, filename: str) -> list[Path]:
        _data = Path("data")
        return [_data / filename] + [
            _data / f"paper_{self.symbol}_M{tf}" / filename for tf in self.agents
        ]

    def _poll_kill_switch(self) -> None:
        for _p in self._ctrl_paths(_CTRL_KILL_SWITCH):
            if not _p.exists():
                continue
            try:
                with open(_p) as _f:
                    _payload = json.load(_f)
            except Exception:
                _payload = {"active": True, "reason": "parse_error"}
            _p.unlink(missing_ok=True)
            if _payload.get("active"):
                self._execute_kill_switch(_payload)
            break

    def _poll_cb_reset(self) -> None:
        for _p in self._ctrl_paths(_CTRL_CB_RESET):
            if not _p.exists():
                continue
            try:
                try:
                    with open(_p, encoding="utf-8") as _f:
                        _payload = json.load(_f)
                except Exception:
                    _payload = {"reset": True, "reason": "parse_error"}
                _p.unlink(missing_ok=True)
                _target_tfs = {
                    int(_tf)
                    for _tf in (_payload.get("target_timeframes") or [])
                    if str(_tf).strip().isdigit()
                }
                _single_tf = _payload.get("timeframe_minutes")
                if _single_tf is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        _target_tfs.add(int(_single_tf))
                _reset_count = 0
                for _tf, agent in self.agents.items():
                    if _target_tfs and int(_tf) not in _target_tfs:
                        continue
                    if agent.circuit_breakers is not None:
                        agent.circuit_breakers.reset_all()
                        agent._save_cb_state()
                        _reset_count += 1
                LOG.info(
                    "[HUB] Circuit breakers reset via control file (%s, targets=%s, reset=%d, reason=%s)",
                    self.symbol,
                    sorted(_target_tfs) if _target_tfs else "ALL",
                    _reset_count,
                    _payload.get("reason", "unspecified"),
                )
            except Exception as _e:
                LOG.debug("[HUB] CB reset error: %s", _e)
            break

    def _poll_epsilon_override(self) -> None:
        for _p in self._ctrl_paths(_CTRL_EPSILON_OVERRIDE):
            if not _p.exists():
                continue
            try:
                with open(_p) as _f:
                    _payload = json.load(_f)
                _p.unlink(missing_ok=True)
                epsilon = float(_payload.get("epsilon", 1.0))
                epsilon = max(0.0, min(1.0, epsilon))
                for agent in self.agents.values():
                    with contextlib.suppress(Exception):
                        agent.policy.trigger.epsilon = epsilon
                LOG.info("[HUB] Epsilon override applied: %.3f (%s)", epsilon, self.symbol)
            except Exception as _e:
                LOG.debug("[HUB] Epsilon override error: %s", _e)
            break

    def _poll_param_reload(self) -> None:
        for _p in self._ctrl_paths(_CTRL_PARAM_RELOAD):
            if not _p.exists():
                continue
            try:
                _p.unlink(missing_ok=True)
                for agent in self.agents.values():
                    agent.reload_learned_parameters()
                LOG.info("[HUB] Learned parameters reloaded via control file (%s)", self.symbol)
            except Exception as _e:
                LOG.exception("[HUB] Learned-parameter reload error: %s", _e)
            break

    def _poll_kg_reset(self) -> None:
        for _p in self._ctrl_paths(_CTRL_KG_RESET):
            if not _p.exists():
                continue
            try:
                _p.unlink(missing_ok=True)
                for agent in self.agents.values():
                    with contextlib.suppress(Exception):
                        agent.var_estimator.kurtosis_monitor.reset()
                LOG.info("[HUB] Kurtosis gate reset via HUD (%s)", self.symbol)
            except Exception as _e:
                LOG.debug("[HUB] Kurtosis gate reset error: %s", _e)
            break

    def _control_poll_fn(self) -> None:
        """Daemon thread: polls control files every 5 s independent of bar closes."""
        LOG.info("[HUB] Control-file poll thread started (%s)", self.symbol)
        while not self._shutdown_flag:
            try:
                self._poll_kill_switch()
                self._poll_cb_reset()
                self._poll_kg_reset()
                self._poll_epsilon_override()
                self._poll_param_reload()
            except Exception as _e:
                LOG.exception("[HUB] Control poll error: %s", _e)
            time.sleep(5.0)

    def _execute_kill_switch(self, payload: dict) -> None:
        reason = payload.get("reason", "MANUAL_KILL_SWITCH")
        LOG.critical("[KILL-SWITCH] 🚨 Activated: %s (%s)", reason, self.symbol)
        ts_now = dt.datetime.now(dt.UTC)
        for agent in self.agents.values():
            self._kill_agent(agent, reason, ts_now)
        still_open = [a.tf_label for a in self.agents.values() if a.position is not None]
        if still_open:
            LOG.critical("[KILL-SWITCH] WARNING: positions still open after kill: %s (%s)", still_open, self.symbol)
        else:
            LOG.critical("[KILL-SWITCH] All breakers tripped, positions confirmed closed (%s)", self.symbol)

    def _kill_agent(self, agent: TFAgent, reason: str, ts_now: dt.datetime) -> None:
        if agent.circuit_breakers is not None:
            for _b in agent.circuit_breakers.breakers:
                if not _b.state.is_tripped:
                    _b.state.trip(reason=reason, value=999.0, threshold=0.0)
        if agent.position is not None and agent.last_mid > 0:
            fill = agent.last_mid - agent.position["direction"] * agent.last_half_spread
            try:
                agent._close_position(ts_now, fill)
            except Exception as _e:
                LOG.exception("[KILL-SWITCH] Close error %s %s: %s", agent.symbol, agent.tf_label, _e)

    def _on_message(self, client: Any, message: Any) -> None:
        self._last_heartbeat = time.time()  # any server message resets the watchdog
        pt = getattr(message, "payloadType", None)
        handlers = {
            51:   self._handle_heartbeat,
            2101: self._handle_app_auth_res,
            2103: self._handle_acc_auth_res,
            2117: self._handle_symbol_info_res,
            2122: self._handle_trader_res,
            2123: self._handle_trader_updated,
            2126: self._handle_execution_event,
            2128: self._handle_subscribe_res,
            2131: self._handle_spot_event,
            2138: self._handle_trendbars_res,
            2142: self._handle_error,
            2155: self._handle_depth_event,
            2157: self._handle_depth_sub_res,
            2174: self._handle_refresh_token_res,
        }
        handler = handlers.get(pt)
        if handler:
            try:
                handler(message)
            except Exception as e:
                LOG.exception("[HUB] Handler %d failed: %s", pt, e)
        else:
            LOG.info("[HUB] Unhandled payloadType=%s (state=%s)", pt, self._state)

    # ---- auth state machine ---------------------------------------------

    def _send(self, req: Any, timeout: int = 15) -> None:
        """Send a protobuf message and silence any Deferred timeout errors."""
        d = self._client.send(req, responseTimeoutInSeconds=timeout)
        d.addErrback(lambda _failure: None)

    def _send_app_auth(self) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAApplicationAuthReq
        req = ProtoOAApplicationAuthReq()
        req.clientId = self.creds["client_id"]
        req.clientSecret = self.creds["client_secret"]
        self._send(req)
        LOG.debug("[HUB] → ProtoOAApplicationAuthReq")

    def _handle_app_auth_res(self, message: Any) -> None:
        if self._state != _S_APP_AUTH:
            return
        LOG.info("[HUB] App auth OK")
        self._state = _S_ACC_AUTH
        self._send_acc_auth()

    def _send_acc_auth(self) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAAccountAuthReq
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.account_id
        req.accessToken = self.creds["access_token"]
        self._send(req)
        LOG.debug("[HUB] → ProtoOAAccountAuthReq account=%d", self.account_id)

    def _handle_acc_auth_res(self, message: Any) -> None:
        if self._state != _S_ACC_AUTH:
            return
        LOG.info("[HUB] Account auth OK (account=%d)", self.account_id)
        self._state = _S_SYM_INFO
        self._fetch_trader_info()
        self._fetch_symbol_info()

    def _fetch_trader_info(self) -> None:
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOATraderReq
            req = ProtoOATraderReq()
            req.ctidTraderAccountId = self.account_id
            self._send(req)
            LOG.debug("[HUB] → ProtoOATraderReq account=%d", self.account_id)
        except Exception as e:
            LOG.warning("[HUB] ProtoOATraderReq failed: %s", e)

    def _handle_trader_res(self, message: Any) -> None:
        try:
            from ctrader_open_api import Protobuf
            res = Protobuf.extract(message)
            trader = getattr(res, "trader", None)
            if trader is None:
                return
            md = int(getattr(trader, "moneyDigits", 2) or 2)
            self._broker_money_digits = md
            divisor = 10 ** md
            raw_balance = getattr(trader, "balance", None)
            if raw_balance is not None:
                self._broker_balance = round(int(raw_balance) / divisor, 2)
            LOG.info(
                "[HUB] Broker account balance=%.2f (moneyDigits=%d) account=%d",
                self._broker_balance or 0.0,
                md,
                self.account_id,
            )
        except Exception as e:
            LOG.warning("[HUB] ProtoOATraderRes parse failed: %s", e)

    def _handle_trader_updated(self, message: Any) -> None:
        try:
            from ctrader_open_api import Protobuf
            event = Protobuf.extract(message)
            trader = getattr(event, "trader", None)
            if trader is None:
                return
            md = int(getattr(trader, "moneyDigits", self._broker_money_digits) or self._broker_money_digits)
            self._broker_money_digits = md
            divisor = 10 ** md
            raw_balance = getattr(trader, "balance", None)
            if raw_balance is not None:
                self._broker_balance = round(int(raw_balance) / divisor, 2)
                LOG.debug("[HUB] Broker balance updated: %.2f", self._broker_balance)
        except Exception as e:
            LOG.warning("[HUB] ProtoOATraderUpdatedEvent parse failed: %s", e)

    def _handle_execution_event(self, message: Any) -> None:
        """Handle ProtoOAExecutionEvent — capture commission/swap/balance from live fills."""
        try:
            from ctrader_open_api import Protobuf
            event = Protobuf.extract(message)
            deal = getattr(event, "deal", None)
            if deal is None:
                return
            md = self._broker_money_digits
            divisor = 10 ** md
            cpd = getattr(deal, "closePositionDetail", None)
            if cpd is None:
                return
            # closePositionDetail gives us commission, swap, grossProfit, balance after close
            commission_raw = int(getattr(cpd, "commission", 0) or 0)
            swap_raw = int(getattr(cpd, "swap", 0) or 0)
            gross_profit_raw = int(getattr(cpd, "grossProfit", 0) or 0)
            balance_raw = int(getattr(cpd, "balance", 0) or 0)
            commission = round(commission_raw / divisor, 6)
            swap = round(swap_raw / divisor, 6)
            gross_profit = round(gross_profit_raw / divisor, 6)
            balance_after = round(balance_raw / divisor, 2)
            self._broker_balance = balance_after
            LOG.info(
                "[HUB] ExecutionEvent close: commission=%.4f swap=%.4f gross_pnl=%.4f balance=%.2f",
                commission, swap, gross_profit, balance_after,
            )
            # Emit to transaction log so the audit trail has real broker numbers
            try:
                if hasattr(self, "_agent") and hasattr(self._agent, "transaction_log"):
                    self._agent.transaction_log.log_event(
                        "BROKER_EXECUTION",
                        {
                            "commission": commission,
                            "swap": swap,
                            "gross_profit": gross_profit,
                            "balance_after": balance_after,
                            "deal_id": str(getattr(deal, "dealId", "")),
                            "position_id": str(getattr(deal, "positionId", "")),
                            "source": "ProtoOAExecutionEvent",
                        },
                    )
            except Exception:
                pass
        except Exception as e:
            LOG.warning("[HUB] ProtoOAExecutionEvent parse failed: %s", e)

    def _fetch_symbol_info(self) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASymbolByIdReq
        req = ProtoOASymbolByIdReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId.append(self.symbol_id)
        self._send(req)
        LOG.debug("[HUB] → ProtoOASymbolByIdReq symbolId=%d", self.symbol_id)

    def _handle_symbol_info_res(self, message: Any) -> None:
        if self._state != _S_SYM_INFO:
            return
        try:
            from ctrader_open_api import Protobuf
            res = Protobuf.extract(message)
            symbols = list(getattr(res, "symbol", []))
            if symbols:
                sym = symbols[0]
                # `digits` is the display precision only.
                # cTrader SpotEvent bid/ask are always encoded at 10^5 regardless of digits.
                display_digits = int(getattr(sym, "digits", self._digits))
                self._display_digits = display_digits
                # Keep _scale at 10^5 (SpotEvent wire format); don't override from display digits.
                LOG.info("[HUB] Symbol info: %s display_digits=%d wire_scale=%d",
                         self.symbol, display_digits, self._scale)
        except Exception as e:
            LOG.warning("[HUB] Symbol info parse failed (%s) — using scale=%d", e, self._scale)

        self._state = _S_SUBSCRIBING
        self._subscribe_spots()

    def _subscribe_spots(self) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASubscribeSpotsReq
        req = ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId.append(self.symbol_id)
        req.subscribeToSpotTimestamp = True
        self._send(req)
        LOG.debug("[HUB] → ProtoOASubscribeSpotsReq symbolId=%d", self.symbol_id)

    def _handle_subscribe_res(self, message: Any) -> None:
        if self._state != _S_SUBSCRIBING:
            return
        LOG.info("[HUB] Spot subscription active for %s (id=%d)", self.symbol, self.symbol_id)
        self._state = _S_READY
        self._write_bot_config()
        self._subscribe_depth()
        if self._token_refresh_in_progress:
            # Token refresh: the TCP stream was never interrupted, bars are current.
            # Skip backfill entirely to avoid injecting duplicates into bar deques.
            self._token_refresh_in_progress = False
        elif self._disconnect_time is not None:
            gap = time.time() - self._disconnect_time
            if gap > _GAP_FILL_MIN_SECONDS:
                self._request_gap_fill(self._disconnect_time)
            self._disconnect_time = None
        else:
            self._request_initial_backfill()

    def _write_bot_config(self) -> None:
        first_agent = next(iter(self.agents.values()), None)
        config = {
            "symbol": self.symbol,
            "symbol_id": self.symbol_id,
            "timeframes": sorted(self.agents.keys()),
            "trading_mode": "paper",
            "training_enabled": os.environ.get("DDQN_ONLINE_LEARNING", "1") == "1",
            "starting_equity": first_agent.starting_equity if first_agent else _DEFAULT_STARTING_EQUITY,
            "qty": first_agent.qty if first_agent else 0.5,
            "real_account_balance": None,
            "real_account_equity": None,
            "real_margin_free": None,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        try:
            _write_json_atomic(Path("data") / "bot_config.json", config, indent=2)
        except Exception as e:
            LOG.debug("[HUB] bot_config write error: %s", e)

    def _subscribe_depth(self) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASubscribeDepthQuotesReq
        req = ProtoOASubscribeDepthQuotesReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId.append(self.symbol_id)
        self._send(req)
        LOG.debug("[HUB] → ProtoOASubscribeDepthQuotesReq symbolId=%d", self.symbol_id)

    def _handle_depth_sub_res(self, _message: Any) -> None:
        LOG.info("[HUB] Depth subscription active for %s (id=%d)", self.symbol, self.symbol_id)

    # ---- reconnect gap-fill ---------------------------------------------

    def _request_gap_fill(self, from_time: float) -> None:
        """Fetch missed closed bars for each TF after a reconnect gap."""
        from_ms = int(from_time * 1000)
        to_ms = int(time.time() * 1000)
        gap_secs = (to_ms - from_ms) / 1000.0
        LOG.info("[HUB] Gap-fill: %.0fs gap detected — fetching missed bars for %s", gap_secs, self.symbol)
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAGetTrendbarsReq
        except ImportError:
            LOG.warning("[HUB] Gap-fill: ProtoOAGetTrendbarsReq unavailable — skipping")
            return
        for tf in self.agents:
            period = _TF_TO_PERIOD.get(tf)
            if period is None:
                LOG.warning("[HUB] Gap-fill: no period mapping for TF=%d, skipping", tf)
                continue
            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = self.account_id
            req.symbolId = self.symbol_id
            req.period = period
            req.fromTimestamp = from_ms
            req.toTimestamp = to_ms
            self._send(req)
            LOG.debug("[HUB] → ProtoOAGetTrendbarsReq TF=%d period=%d", tf, period)

    def _request_initial_backfill(self) -> None:
        """Pre-fill bar deques on cold start so agents skip WARMING_UP immediately."""
        cold_tfs = [tf for tf, a in self.agents.items() if len(a.bars) < _MIN_BARS_BEFORE_TRADE]
        if not cold_tfs:
            return
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAGetTrendbarsReq
        except ImportError:
            LOG.warning("[HUB] Initial backfill: ProtoOAGetTrendbarsReq unavailable — skipping")
            return
        now_ms = int(time.time() * 1000)
        for tf in cold_tfs:
            period = _TF_TO_PERIOD.get(tf)
            if period is None:
                continue
            from_ms = now_ms - _INITIAL_BACKFILL_BARS * tf * 60 * 1000
            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = self.account_id
            req.symbolId = self.symbol_id
            req.period = period
            req.fromTimestamp = from_ms
            req.toTimestamp = now_ms
            self._send(req)
            LOG.info("[HUB] Initial backfill: requesting %d bars for %s M%d", _INITIAL_BACKFILL_BARS, self.symbol, tf)

    def _handle_trendbars_res(self, message: Any) -> None:
        """Inject gap-fill bars into the appropriate TFAgent."""
        try:
            from ctrader_open_api import Protobuf
            res = Protobuf.extract(message)
        except Exception as e:
            LOG.debug("[HUB] trendbars extract failed: %s", e)
            return
        period = int(getattr(res, "period", 0))
        tf = _PERIOD_TO_TF.get(period)
        if tf is None:
            LOG.warning("[HUB] Gap-fill: unrecognised period=%d", period)
            return
        agent = self.agents.get(tf)
        if agent is None:
            return
        bars = list(getattr(res, "trendbar", []))
        if not bars:
            LOG.info("[HUB] Gap-fill: no bars returned for %s M%d", self.symbol, tf)
            return
        half_spread = agent.last_half_spread or 0.0
        injected = 0
        for tb in bars:
            try:
                low_raw = int(getattr(tb, "low", 0))
                delta_open = int(getattr(tb, "deltaOpen", 0))
                delta_high = int(getattr(tb, "deltaHigh", 0))
                delta_close = int(getattr(tb, "deltaClose", 0))
                ts_min = int(getattr(tb, "utcTimestampInMinutes", 0))
                if not ts_min or not low_raw:
                    continue
                low = low_raw / self._scale
                bar = (
                    dt.datetime.fromtimestamp(ts_min * 60, tz=dt.UTC),
                    (low_raw + delta_open) / self._scale,
                    (low_raw + delta_high) / self._scale,
                    low,
                    (low_raw + delta_close) / self._scale,
                )
                agent._on_bar_close(bar, half_spread)
                injected += 1
            except Exception as e:
                LOG.debug("[HUB] Gap-fill bar error: %s", e)
        LOG.info("[HUB] Gap-fill: injected %d bars into %s %s", injected, self.symbol, agent.tf_label)

    def _apply_new_quote(self, quote: Any) -> None:
        bid_raw = int(getattr(quote, "bid", 0) or 0)
        ask_raw = int(getattr(quote, "ask", 0) or 0)
        size = float(getattr(quote, "size", 0) or 0)
        qid = int(getattr(quote, "id", 0) or 0)
        if bid_raw:
            price, side = bid_raw / self._scale, "BID"
            self._bid_refresh_count += 1
        elif ask_raw:
            price, side = ask_raw / self._scale, "ASK"
            self._ask_refresh_count += 1
        else:
            return
        if size > 0:
            self._has_real_sizes = True
        if qid:
            self._quote_id_map[qid] = (side, price)
        self._order_book.update_level(side, price, size)

    def _apply_deleted_quote(self, quote: Any) -> None:
        qid = int(getattr(quote, "id", 0) or 0)
        entry = self._quote_id_map.pop(qid, None)
        if entry:
            self._order_book.update_level(entry[0], entry[1], 0.0)

    def _handle_depth_event(self, message: Any) -> None:
        if self._state != _S_READY:
            return
        try:
            from ctrader_open_api import Protobuf
            payload = Protobuf.extract(message)
        except Exception as e:
            LOG.debug("[HUB] depth extract failed: %s", e)
            return
        for quote in getattr(payload, "newQuotes", []):
            self._apply_new_quote(quote)
        for quote in getattr(payload, "deletedQuotes", []):
            self._apply_deleted_quote(quote)
        now = time.time()
        if now - self._last_ob_write >= 1.0:
            self._last_ob_write = now
            self._write_order_book()

    def _update_vpin(self, mid: float) -> None:
        """Feed mid-price change into VPINCalculator; update self._vpin_z."""
        if self._vpin_last_mid <= 0:
            self._vpin_last_mid = mid
            return
        delta = mid - self._vpin_last_mid
        if SafeMath.is_zero(delta):
            return
        side = "BUY" if delta > 0 else "SELL"
        self._vpin_calc.update(volume=1.0, side=side)
        self._vpin_stats = self._vpin_calc.get_stats()
        self._vpin_z = float(self._vpin_stats.get("zscore", 0.0))
        self._vpin_last_mid = mid

    def _prune_l2_against_spot(self, mid: float, half_spread: float) -> int:
        """Drop stale depth levels that cross the current spot bid/ask."""
        if half_spread <= 0:
            return 0
        spot_bid = mid - half_spread
        spot_ask = mid + half_spread
        if spot_bid <= 0 or spot_ask <= 0 or spot_bid >= spot_ask:
            return 0

        bad_bids = [price for price in self._order_book.bids if price >= spot_ask]
        bad_asks = [price for price in self._order_book.asks if price <= spot_bid]
        for price in bad_bids:
            self._order_book.bids.pop(price, None)
        for price in bad_asks:
            self._order_book.asks.pop(price, None)

        removed = len(bad_bids) + len(bad_asks)
        if removed:
            now = time.time()
            suppressed = int(getattr(self, "_l2_prune_suppressed", 0))
            if now - float(getattr(self, "_last_l2_prune_log", 0.0)) >= 30.0:
                LOG.warning(
                    "[HUB] Dropped %d stale L2 level(s) crossing spot bid/ask %.5f/%.5f"
                    " (suppressed=%d)",
                    removed,
                    spot_bid,
                    spot_ask,
                    suppressed,
                )
                self._last_l2_prune_log = now
                self._l2_prune_suppressed = 0
            else:
                self._l2_prune_suppressed = suppressed + removed
        return removed

    def _write_order_book(self) -> None:
        depth_bid, depth_ask = self._order_book.depth_sum()
        bids = [[p, s] for p, s in sorted(self._order_book.bids.items(), reverse=True)[:10]]
        asks = [[p, s] for p, s in sorted(self._order_book.asks.items())[:10]]

        # QFI: quote-flow imbalance from bid/ask refresh counts since last write.
        # Blended with size-weighted imbalance when real sizes are available.
        _qfi_total = self._bid_refresh_count + self._ask_refresh_count
        _qfi = (self._bid_refresh_count - self._ask_refresh_count) / _qfi_total if _qfi_total > 0 else 0.0
        raw_imbalance = self._order_book.imbalance()
        if self._has_real_sizes and abs(raw_imbalance) > 1e-6:
            imbalance = 0.5 * raw_imbalance + 0.5 * _qfi
        else:
            imbalance = _qfi if _qfi_total > 0 else raw_imbalance
        self._bid_refresh_count = 0
        self._ask_refresh_count = 0

        data = {
            "symbol": self.symbol,
            "imbalance": imbalance,
            "depth_bid": depth_bid,
            "depth_ask": depth_ask,
            "vpin_zscore": self._vpin_z,
            "has_real_sizes": self._has_real_sizes,
            "qfi_update_count": _qfi_total,
            "order_book_bids": bids,
            "order_book_asks": asks,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        base = Path("data")
        _write_json_async(base / "order_book.json", data)
        _write_json_async(base / f"order_book_{self.symbol}.json", data)
        # Write per-TF scoped files so _preferred_data_file finds fresh data
        # regardless of which TF is active in the HUD.
        for tf in self.agents:
            _write_json_async(base / f"order_book_{self.symbol}_M{tf}.json", data)

    def _l2_snapshot(self) -> dict:
        depth_bid, depth_ask = self._order_book.depth_sum()
        return {
            "depth_bid": depth_bid,
            "depth_ask": depth_ask,
            "bids": [[p, s] for p, s in sorted(self._order_book.bids.items(), reverse=True)[:10]],
            "asks": [[p, s] for p, s in sorted(self._order_book.asks.items())[:10]],
            "imbalance": self._order_book.imbalance(),
            "vpin_zscore": self._vpin_z,
            "has_real_sizes": self._has_real_sizes,
        }

    def _handle_heartbeat(self, _message: Any) -> None:
        """Echo heartbeat back to keep the connection alive (required by cTrader protocol)."""
        self._last_heartbeat = time.time()
        try:
            from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoHeartbeatEvent
            d = self._client.send(ProtoHeartbeatEvent())
            d.addErrback(lambda _: None)
        except Exception:
            pass

    def _force_reconnect(self, reason: str) -> None:
        """Sever the current TCP connection; ClientService will reconnect automatically."""
        LOG.warning("[HUB] %s — forcing reconnect", reason)
        self._state = _S_CONNECTING
        self._last_heartbeat = time.time()  # reset so watchdog doesn't loop
        if self._client is None:
            return
        try:
            # Preferred: get the live protocol via whenConnected and drop its transport
            d = self._client.whenConnected(failAfterFailures=1)
            d.addCallback(lambda proto: proto.transport.loseConnection())
            d.addErrback(lambda _: None)
        except Exception:
            pass

    def _check_stale_connection(self, now: float) -> None:
        if not (self._client and self._client.isConnected):
            return
        age = now - self._last_heartbeat
        if self._state == _S_READY and age > 60.0:
            self._force_reconnect(f"no heartbeat for {age:.0f}s")
        elif self._state not in (_S_CONNECTING, _S_READY) and age > 45.0:
            self._force_reconnect(f"stuck in auth state '{self._state}' for {age:.0f}s")

    def _watchdog_check(self) -> None:
        if self._shutdown_flag or self._reactor is None:
            return
        self._check_stale_connection(time.time())
        self._reactor.callLater(30, self._watchdog_check)

    def _handle_error(self, message: Any) -> None:
        try:
            from ctrader_open_api import Protobuf
            res = Protobuf.extract(message)
            code = str(getattr(res, "errorCode", "") or "")
            desc = str(getattr(res, "description", "") or "")
            LOG.error("[HUB] Error from API: code=%s desc=%s", code, desc)
            if code in _AUTH_ERROR_CODES:
                LOG.warning("[HUB] Auth error '%s' — attempting token refresh", code)
                self._refresh_access_token()
        except Exception:
            LOG.exception("[HUB] Received error message (payloadType=2142)")

    def _refresh_access_token(self) -> None:
        """Send ProtoOARefreshTokenReq over the existing TCP connection.

        The response (payload 2174) is handled by _handle_refresh_token_res,
        which updates self.creds and replays the full auth sequence.
        Refresh tokens have no expiry per the cTrader docs.
        """
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOARefreshTokenReq

        refresh_token = self.creds.get("refresh_token", "")
        if not refresh_token:
            LOG.error(
                "[HUB] Token refresh impossible — CTRADER_REFRESH_TOKEN not set in credentials. "
                "Obtain a new access_token and refresh_token via the OAuth flow and restart."
            )
            return
        req = ProtoOARefreshTokenReq()
        req.refreshToken = refresh_token
        self._send(req)
        LOG.info("[HUB] → ProtoOARefreshTokenReq")

    def _handle_refresh_token_res(self, message: Any) -> None:
        """Process ProtoOARefreshTokenRes (payload 2174).

        The response carries accessToken and refreshToken (camelCase per cTrader API).
        Access tokens last 2,628,000 s (~30 days); refresh tokens do not expire.
        """
        try:
            from ctrader_open_api import Protobuf
            res = Protobuf.extract(message)
        except Exception as exc:
            LOG.error("[HUB] Failed to parse ProtoOARefreshTokenRes: %s", exc)
            return

        new_access = str(getattr(res, "accessToken", "") or "")
        new_refresh = str(getattr(res, "refreshToken", "") or "") or self.creds.get("refresh_token", "")
        expires_in = int(getattr(res, "expiresIn", 0) or 0)
        if not new_access:
            LOG.error("[HUB] ProtoOARefreshTokenRes missing accessToken — cannot re-auth")
            return

        self.creds["access_token"] = new_access
        self.creds["refresh_token"] = new_refresh
        LOG.info("[HUB] Access token refreshed (expires in %dd) — replaying account auth",
                 expires_in // 86400 if expires_in else 30)
        self._persist_refreshed_tokens(new_access, new_refresh)
        self._token_refresh_in_progress = True
        self._state = _S_ACC_AUTH
        self._send_acc_auth()

    def _persist_refreshed_tokens(self, access_token: str, refresh_token: str) -> None:
        """Overwrite CTRADER_ACCESS_TOKEN and CTRADER_REFRESH_TOKEN in the credentials file."""
        _root = Path(__file__).resolve().parent.parent.parent
        for cred_path in (_root / ".env.openapi",):
            if not cred_path.exists():
                continue
            try:
                lines = cred_path.read_text(encoding="utf-8").splitlines()
                updated: list[str] = []
                found_access = found_refresh = False
                for line in lines:
                    key = line.partition("=")[0].strip().removeprefix("export").strip()
                    if key == "CTRADER_ACCESS_TOKEN":
                        updated.append(f"CTRADER_ACCESS_TOKEN={access_token}")
                        found_access = True
                    elif key == "CTRADER_REFRESH_TOKEN":
                        updated.append(f"CTRADER_REFRESH_TOKEN={refresh_token}")
                        found_refresh = True
                    else:
                        updated.append(line)
                if not found_access:
                    updated.append(f"CTRADER_ACCESS_TOKEN={access_token}")
                if not found_refresh:
                    updated.append(f"CTRADER_REFRESH_TOKEN={refresh_token}")
                cred_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
                LOG.info("[HUB] Persisted refreshed tokens to %s", cred_path)
            except Exception as exc:
                LOG.warning("[HUB] Could not persist tokens to %s: %s", cred_path, exc)
            break

    # ---- spot event handler ---------------------------------------------

    def _parse_spot_payload(self, payload: Any) -> tuple[float, float, dt.datetime] | None:
        """Extract (mid, half_spread, ts) from a ProtoOASpotEvent payload, or None."""
        if int(getattr(payload, "symbolId", 0)) != self.symbol_id:
            return None
        bid_raw = int(getattr(payload, "bid", 0)) or self._last_bid_raw
        ask_raw = int(getattr(payload, "ask", 0)) or self._last_ask_raw
        if bid_raw:
            self._last_bid_raw = bid_raw
        if ask_raw:
            self._last_ask_raw = ask_raw
        if not (bid_raw and ask_raw):
            return None
        bid, ask = bid_raw / self._scale, ask_raw / self._scale
        ts_ms = int(getattr(payload, "timestamp", 0))
        ts = dt.datetime.fromtimestamp(ts_ms / 1000.0, tz=dt.UTC) if ts_ms else dt.datetime.now(dt.UTC)
        return (bid + ask) / 2.0, (ask - bid) / 2.0, ts

    def _log_tick(self, mid: float, half_spread: float) -> None:
        n = self._tick_count
        if n == 1 or n % 1000 == 0:
            LOG.info("[HUB] %s tick #%d | mid=%.5f spread=%.5f",
                     self.symbol, n, mid, half_spread * 2)

    def _handle_spot_event(self, message: Any) -> None:
        if self._state != _S_READY:
            return
        try:
            from ctrader_open_api import Protobuf
            payload = Protobuf.extract(message)
        except Exception as e:
            LOG.debug("[HUB] spot extract failed: %s", e)
            return
        parsed = self._parse_spot_payload(payload)
        if parsed is None:
            return
        mid, half_spread, ts = parsed
        self._tick_count += 1
        self._log_tick(mid, half_spread)
        self._update_vpin(mid)
        self._prune_l2_against_spot(mid, half_spread)
        imbalance = self._order_book.imbalance()
        depth_bid, depth_ask = self._order_book.depth_sum()
        l2_snapshot = self._l2_snapshot()
        for agent in self.agents.values():
            try:
                agent.on_tick(ts, mid, half_spread, imbalance=imbalance,
                              depth_bid=depth_bid, depth_ask=depth_ask,
                              vpin_z=self._vpin_z,
                              has_real_sizes=self._has_real_sizes,
                              l2_snapshot=l2_snapshot)
            except Exception as e:
                LOG.exception("[HUB] agent %s error: %s", agent.tf_label, e)

    # ---- shutdown -------------------------------------------------------

    def shutdown(self) -> None:
        LOG.info("[HUB] Shutting down %s hub", self.symbol)
        for agent in self.agents.values():
            try:
                agent.shutdown()
            except Exception as e:
                LOG.warning("[HUB] agent %s shutdown error: %s", agent.tf_label, e)
        if self._client and hasattr(self._client, "stopService"):
            with contextlib.suppress(Exception):
                self._client.stopService()
        global _ASYNC_WRITER
        if _ASYNC_WRITER is not None:
            with contextlib.suppress(Exception):
                _ASYNC_WRITER.stop()

    # ---- start -----------------------------------------------------------

    def start(self) -> None:
        """Create Twisted client and run reactor (blocking)."""
        try:
            from ctrader_open_api import Client, TcpProtocol
            from twisted.internet import reactor
        except ImportError as e:
            LOG.exception("[HUB] ctrader-open-api or Twisted not installed: %s", e)
            sys.exit(1)

        client = Client(self.host, _PORT, TcpProtocol)
        client.setConnectedCallback(self._on_connected)
        client.setDisconnectedCallback(self._on_disconnected)
        client.setMessageReceivedCallback(self._on_message)
        self._client = client

        def _sigterm(*_) -> None:
            LOG.info("[HUB] SIGTERM received — shutting down")
            reactor.callFromThread(self._graceful_stop, reactor)

        signal.signal(signal.SIGTERM, _sigterm)
        signal.signal(signal.SIGINT, _sigterm)

        self._reactor = reactor
        client.startService()
        reactor.callLater(30, self._watchdog_check)
        LOG.info("[HUB] Reactor starting")
        try:
            reactor.run(installSignalHandlers=False)
        finally:
            self.shutdown()

    def _graceful_stop(self, reactor: Any) -> None:
        self._shutdown_flag = True
        self.shutdown()
        if reactor.running:
            reactor.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load credentials
    creds = _load_creds()
    missing = [k for k in ("client_id", "client_secret", "access_token", "account_id") if not creds.get(k)]
    if missing:
        LOG.error("[HUB] Missing credentials: %s", missing)
        LOG.error("[HUB] Set CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, CTRADER_ACCOUNT_ID")
        sys.exit(1)

    # Symbol config
    symbol = os.environ.get("OPENAPI_SYMBOL") or os.environ.get("SYMBOL", "XAUUSD")
    symbol_id_str = os.environ.get("OPENAPI_SYMBOL_ID") or os.environ.get("SYMBOL_ID", "41")
    try:
        symbol_id = int(symbol_id_str)
    except ValueError:
        LOG.exception("[HUB] Invalid OPENAPI_SYMBOL_ID=%r", symbol_id_str)
        sys.exit(1)

    # Timeframes
    tf_raw = os.environ.get("OPENAPI_TIMEFRAMES") or os.environ.get("TIMEFRAME_MINUTES", "5")
    try:
        timeframes = [int(x.strip()) for x in tf_raw.split(",") if x.strip()]
    except ValueError:
        LOG.exception("[HUB] Invalid OPENAPI_TIMEFRAMES=%r", tf_raw)
        sys.exit(1)

    if not timeframes:
        LOG.error("[HUB] No timeframes specified")
        sys.exit(1)

    # Position sizing
    qty = float(os.environ.get("CTRADER_QTY", "0.5"))
    starting_equity = float(os.environ.get("CTRADER_STARTING_EQUITY", str(_DEFAULT_STARTING_EQUITY)))

    # Contract size
    spec = _load_symbol_spec(symbol)
    default_cs = float(spec.get("contract_size", _DEFAULT_CONTRACT_SIZE))
    contract_size = float(os.environ.get("CTRADER_CONTRACT_SIZE", str(default_cs)))

    # Data dir
    data_dir_root = Path(os.environ.get("CTRADER_DATA_DIR", "data"))

    # Server — probe alt endpoints first
    live = os.environ.get("OPENAPI_LIVE", "0").strip() == "1"
    host = _LIVE_HOST if live else _DEMO_HOST
    alt_raw = os.environ.get("CTRADER_ALT_ENDPOINTS", "")
    if alt_raw:
        host = _probe_endpoints(host, _PORT, alt_raw)

    # Online learning
    online_learning = os.environ.get("DDQN_ONLINE_LEARNING", "1") == "1"

    LOG.info("[HUB] Starting | symbol=%s id=%d TFs=%s qty=%.2f contract=%.0f server=%s",
             symbol, symbol_id, timeframes, qty, contract_size, host)

    hub = OpenAPIHub(
        symbol=symbol,
        symbol_id=symbol_id,
        timeframes=timeframes,
        creds=creds,
        host=host,
        qty=qty,
        contract_size=contract_size,
        data_dir_root=data_dir_root,
        starting_equity=starting_equity,
        online_learning=online_learning,
    )
    hub.start()


if __name__ == "__main__":
    main()
