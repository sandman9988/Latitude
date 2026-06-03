#!/usr/bin/env python3
"""Dual Policy - Orchestrates Trigger and Harvester Agents (Phase 3).
=================================================================
Coordinates entry and exit specialists for dual-agent trading.

Architecture:
- TriggerAgent: Entry specialist (when to enter, which direction)
- HarvesterAgent: Exit specialist (when to close position)

From MASTER_HANDBOOK.md Section 2.2: Dual-Agent Architecture
"""

import datetime as dt
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.agents.harvester_agent import HarvesterAgent
from src.agents.trigger_agent import TriggerAgent
from src.constants import (
    DEFAULT_VOLATILITY,
    HARVESTER_BUFFER_CAPACITY,
    MIN_BARS_FOR_FEATURES,
    RETURN_LAG_MEDIUM,
    RETURN_LAG_SHORT,
    STATE_WINDOW_SIZE,
    TRIGGER_BUFFER_CAPACITY,
)
from src.features.regime_detector import TRENDING_THRESHOLD, RegimeDetector  # Phase 3.4
from src.persistence.learned_parameters import LearnedParametersManager
from src.utils.experience_buffer import RegimeSampling
from src.utils.mfe_mae import MFEMAECalculator
from src.utils.safe_math import SafeMath, rolling_mean, rolling_std
from src.utils.safe_utils import save_json_atomic

LOG = logging.getLogger(__name__)

TEST_ENTRY_PRICE: float = 100000.0
_FEATURE_VARIANCE_FLOOR: float = 1e-6  # minimum std to treat a feature column as variable


def _safe_path_token(value: str) -> str:
    return str(value or "UNKNOWN").replace("/", "_").replace("\\", "_")


_MIN_SEED_BARS: int = 3  # minimum bars required to seed the regime detector


@dataclass
class DualPolicyConfig:
    window: int = STATE_WINDOW_SIZE
    enable_regime_detection: bool = True
    path_geometry: Any = None
    enable_training: bool = False
    enable_event_features: bool = True
    param_manager: LearnedParametersManager | None = None
    symbol: str = "XAUUSD"
    timeframe: str = "M15"
    broker: str = "default"
    timeframe_minutes: int = 5
    min_bars_for_features: int = MIN_BARS_FOR_FEATURES
    friction_calculator: Any = None
    trigger_buffer_capacity: int = TRIGGER_BUFFER_CAPACITY
    harvester_buffer_capacity: int = HARVESTER_BUFFER_CAPACITY


# Delegate to shared rolling helpers (single source of truth in safe_math)
_dp_rolling_mean = rolling_mean
_dp_rolling_std = rolling_std


def _build_event_feature_columns(event_features: dict[str, float] | None, n_c: int) -> list[np.ndarray]:
    """Return 6 broadcast arrays for session-time event features."""
    ef = event_features or {}
    vals = [
        ef.get("london_active", 0.0),
        ef.get("ny_active", 0.0),
        ef.get("tokyo_active", 0.0),
        ef.get("london_ny_overlap", 0.0),
        ef.get("rollover_proximity_norm", 0.0),
        ef.get("week_progress", 0.5),
    ]
    return [np.full(n_c, v, dtype=np.float64) for v in vals]


class DualPolicy:
    """Orchestrates TriggerAgent and HarvesterAgent for specialized trading.

    Workflow:
    1. On bar close (flat): trigger.decide_entry() → LONG/SHORT/NONE
    2. On bar close (in position): harvester.decide_exit() → HOLD/CLOSE
    3. Track position state (MFE, MAE, ticks_held) for harvester

    Backward Compatibility:
    - If DDQN_DUAL_AGENT=0: Falls back to single Policy
    - If DDQN_DUAL_AGENT=1: Uses dual-agent architecture
    """

    def __init__(
        self,
        *args: int,
        config: DualPolicyConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DualPolicy with trigger and harvester agents.

        Args:
            config: Optional DualPolicyConfig instance
            **kwargs: Field overrides for DualPolicyConfig

        """
        config = self._coerce_config(args, config, kwargs)
        self._apply_config(config)
        trigger_features, harvester_total_features = self._feature_dimensions(config)
        self._init_agents(config, trigger_features, harvester_total_features)
        self._init_regime_detector(config)
        self._init_runtime_state()

        LOG.info("[DUAL_POLICY] Initialized with TriggerAgent + HarvesterAgent")

    @staticmethod
    def _coerce_config(
        args: tuple[int, ...],
        config: DualPolicyConfig | None,
        kwargs: dict[str, Any],
    ) -> DualPolicyConfig:
        if args:
            if len(args) > 1:
                msg = "DualPolicy accepts at most one positional argument (window)"
                raise TypeError(msg)
            if "window" in kwargs:
                msg = "DualPolicy received both positional window and keyword window"
                raise TypeError(msg)
            assert len(args) >= 1  # Guaranteed by the if args: check
            kwargs["window"] = args[0]

        if config is None:
            config = DualPolicyConfig()

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                msg = f"Unexpected argument: {key}"
                raise TypeError(msg)
        return config

    def _apply_config(self, config: DualPolicyConfig) -> None:
        self.window = config.window
        self.enable_training = config.enable_training
        self.enable_event_features = config.enable_event_features
        self.param_manager = config.param_manager
        self.symbol = config.symbol
        self.timeframe = config.timeframe
        self.timeframe_minutes = config.timeframe_minutes
        self.broker = config.broker
        self.friction_calculator = config.friction_calculator
        self.path_geometry = config.path_geometry
        # Scale minimum-bars threshold to wall-clock time so higher timeframes
        # don't produce zero-state for absurd durations (H4 would need 11 days!).
        self.min_bars_for_features = config.min_bars_for_features

    def _feature_dimensions(self, config: DualPolicyConfig) -> tuple[int, int]:
        # Calculate feature dimensions (base=7, geometry=5, event=6)
        base_features = 7
        geometry_features = 5 if config.path_geometry else 0
        self.event_feature_count = 6 if config.enable_event_features else 0

        trigger_features = base_features + geometry_features + self.event_feature_count
        harvester_market_features = trigger_features
        harvester_total_features = harvester_market_features + 3  # +3 position stats (MFE/MAE/bars)
        return trigger_features, harvester_total_features

    def _init_agents(
        self,
        config: DualPolicyConfig,
        trigger_features: int,
        harvester_total_features: int,
    ) -> None:
        self.trigger = TriggerAgent(
            window=config.window,
            n_features=trigger_features,
            enable_training=config.enable_training,
            symbol=self.symbol,
            timeframe=self.timeframe,
            broker=self.broker,
            param_manager=self.param_manager,
            timeframe_minutes=config.timeframe_minutes,
            buffer_capacity=config.trigger_buffer_capacity,
        )
        self.harvester = HarvesterAgent(
            window=config.window,
            n_features=harvester_total_features,
            enable_training=config.enable_training,
            symbol=self.symbol,
            timeframe=self.timeframe,
            broker=self.broker,
            param_manager=self.param_manager,
            friction_calculator=self.friction_calculator,
            timeframe_minutes=config.timeframe_minutes,
            buffer_capacity=config.harvester_buffer_capacity,
        )

        LOG.info("[DUAL_POLICY] TriggerAgent: %d features (7 base + 5 geometry + 6 event)", trigger_features)
        LOG.info("[DUAL_POLICY] HarvesterAgent: %d features (market + position)", harvester_total_features)

    def _init_regime_detector(self, config: DualPolicyConfig) -> None:
        # Regime detection
        self.enable_regime_detection = config.enable_regime_detection
        if self.enable_regime_detection:
            # Scale update_interval inversely with timeframe so regime reacts
            # at roughly the same wall-clock frequency regardless of bar size.
            # M5 → every 5 bars; H1 → every 1 bar; H4+ → every 1 bar
            update_interval = max(1, min(5, int(5 * 5 / max(1, config.timeframe_minutes))))
            self.regime_detector = RegimeDetector(window_size=50, update_interval=update_interval)
        else:
            self.regime_detector = None

    def _init_runtime_state(self) -> None:
        # Position tracking for harvester
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_bar_time = None
        self._mfe_calc = MFEMAECalculator()  # single source of truth
        self.ticks_held = 0  # Number of market data ticks (not bars!)
        self.predicted_runway = 0.0  # From trigger agent (net, friction-adjusted)
        self.predicted_runway_gross = 0.0

        # Phase 3.4: Regime state
        self.current_regime = "UNKNOWN"
        self.current_zeta = 1.0
        self.current_regime_enum = RegimeSampling.UNKNOWN

    # ── MFE / MAE properties (delegate to _mfe_calc) ──────────────────────

    @property
    def mfe(self) -> float:
        return self._mfe_calc.mfe

    @mfe.setter
    def mfe(self, value: float) -> None:
        self._mfe_calc.mfe = value

    @property
    def mae(self) -> float:
        return self._mfe_calc.mae

    @mae.setter
    def mae(self, value: float) -> None:
        self._mfe_calc.mae = value

    def decide_entry(
        self,
        bars: deque[Any],
        imbalance: float = 0.0,
        vpin_z: float = 0.0,
        depth_ratio: float = 1.0,
        realized_vol: float = DEFAULT_VOLATILITY,  # For economics calculations
        event_features: dict[str, float] | None = None,  # Phase 3: Event-relative time features
    ) -> tuple[int, float, float]:
        """Decide entry action using TriggerAgent.

        Args:
            bars: Deque of (t, o, h, l, c) tuples (closed bars)
            imbalance: Order book imbalance [-1, 1]
            vpin_z: VPIN z-score
            depth_ratio: Depth ratio
            realized_vol: Rogers-Satchell volatility for economics calculations
            event_features: Dict of event-relative time features (30+ features)

        Returns:
            (action, confidence, predicted_runway)
            - action: 0=NO_ENTRY, 1=LONG, 2=SHORT
            - confidence: [0, 1] Platt-calibrated probability
            - predicted_runway: Expected MFE

        """
        # Defensive sync: decide_entry is ONLY called when the bot confirms FLAT.
        # If DualPolicy.current_position disagrees, force-reset to prevent the
        # trigger from being permanently blocked by _should_block_for_position().
        if self.current_position != 0:
            LOG.warning(
                "[DUAL_POLICY] Position desync: current_position=%d but entry decision requested (syncing to FLAT)",
                self.current_position,
            )
            self.current_position = 0
            self.entry_price = 0.0
            self.entry_bar_time = None
            self.mfe = 0.0
            self.mae = 0.0
            self.ticks_held = 0
        self.predicted_runway = 0.0
        self.predicted_runway_gross = 0.0

        self._update_regime_from_bars(bars)

        # Build state (includes path geometry and event features if available)
        state = self._build_state(bars, imbalance, vpin_z, depth_ratio, realized_vol, event_features)

        # Phase 3.4: Get regime threshold adjustment for trigger
        regime_threshold_adj = 0.0
        if self.regime_detector:
            regime_threshold_adj = self.regime_detector.get_trigger_threshold_adjustment()

        feasibility = self._resolve_feasibility()
        expected_gain, expected_loss = self._estimate_economics(realized_vol)
        friction_cost = self._estimate_friction_cost(bars)

        # Call TriggerAgent with friction costs and economics parameters
        action, confidence, predicted_runway = self.trigger.decide(
            state=state,
            current_position=self.current_position,
            regime_threshold_adj=regime_threshold_adj,  # Phase 3.4
            feasibility=feasibility,  # Phase 2: Hard gate
            expected_gain=expected_gain,  # Phase 2: Economics
            expected_loss=expected_loss,
            friction_cost=friction_cost,  # Phase 2: Actual broker friction (commission + swap + spread + slippage)
            zeta=self.current_zeta,  # Regime ζ for adaptive epsilon scheduling
            bars=bars,  # Raw closed bars for the runway forecaster
        )

        self._record_predicted_runway(action, confidence, predicted_runway)

        return action, confidence, predicted_runway

    def _update_regime_from_bars(self, bars: deque[Any]) -> None:
        if len(bars) > 0:
            self._ingest_price_for_regime(bars[-1][4])

    def _resolve_feasibility(self) -> float:
        feasibility = 1.0
        if self.path_geometry:
            feasibility = self.path_geometry.last.get("feasibility", 1.0)  # type: ignore[union-attr]

        _zeta = self.current_zeta
        # Apply a regime uncertainty penalty only for NON-TRENDING regimes.
        # TRENDING (ζ < TRENDING_THRESHOLD=0.7) is the ideal regime for this bot
        # and must NOT be penalised.  As ζ rises above the trending boundary the
        # market becomes increasingly transitional/mean-reverting, so feasibility
        # is reduced linearly from 1.0 at ζ=0.7 down to 0.5 at ζ≥1.7.
        if _zeta >= TRENDING_THRESHOLD:
            _zeta_scale = max(0.5, 1.0 - 0.5 * min(1.0, _zeta - TRENDING_THRESHOLD))
            _raw_feas = feasibility
            feasibility = feasibility * _zeta_scale
            LOG.debug(
                "[DUAL_POLICY] ζ=%.2f → feasibility %.3f → %.3f (regime uncertainty gate)",
                _zeta,
                _raw_feas,
                feasibility,
            )
        return feasibility

    def _estimate_economics(self, realized_vol: float) -> tuple[float, float]:
        expected_gain = realized_vol * 2.0
        expected_loss = realized_vol * 1.0
        return expected_gain, expected_loss

    def _estimate_friction_cost(self, bars: deque[Any]) -> float:
        if self.friction_calculator and len(bars) > 0:
            current_price = bars[-1][4]
            friction_data = self.friction_calculator.calculate_total_friction(  # type: ignore[union-attr]
                quantity=0.10,
                side="BUY",
                price=current_price,
                holding_days=0.1,
                volatility_factor=1.0,
                crosses_rollover=False,
            )
            return friction_data["total"] / current_price if current_price > 0 else 0.0002
        return 3.0 * 0.0001

    def _record_predicted_runway(self, action: int, confidence: float, predicted_runway: float) -> None:
        if action in [1, 2] and self.regime_detector:
            regime_multiplier = self.regime_detector.get_regime_multiplier()
            predicted_runway_adjusted = predicted_runway * regime_multiplier
            predicted_runway_gross = float(getattr(self.trigger, "last_predicted_runway_gross", predicted_runway))
            predicted_runway_gross_adjusted = predicted_runway_gross * regime_multiplier

            LOG.info(
                "[DUAL_POLICY] TRIGGER: %s entry, conf=%.2f, runway_net=%.4f runway_gross=%.4f "
                "(base_net=%.4f base_gross=%.4f regime=%s mult=%.2fx)",
                "LONG" if action == 1 else "SHORT",
                confidence,
                predicted_runway_adjusted,
                predicted_runway_gross_adjusted,
                predicted_runway,
                predicted_runway_gross,
                self.current_regime,
                regime_multiplier,
            )

            self.predicted_runway = predicted_runway_adjusted
            self.predicted_runway_gross = predicted_runway_gross_adjusted
        elif action in [1, 2]:
            self.predicted_runway = predicted_runway
            self.predicted_runway_gross = float(getattr(self.trigger, "last_predicted_runway_gross", predicted_runway))
            LOG.info(
                "[DUAL_POLICY] TRIGGER: %s entry, conf=%.2f, predicted_runway_net=%.4f predicted_runway_gross=%.4f",
                "LONG" if action == 1 else "SHORT",
                confidence,
                predicted_runway,
                self.predicted_runway_gross,
            )

    def decide_exit(
        self,
        bars: deque[Any],
        current_price: float,
        imbalance: float = 0.0,
        vpin_z: float = 0.0,
        depth_ratio: float = 1.0,
        event_features: dict[str, float] | None = None,
    ) -> tuple[int, float]:
        """Decide exit action using HarvesterAgent.

        Args:
            bars: Deque of (t, o, h, l, c) tuples (closed bars)
            current_price: Current close price
            imbalance: Order book imbalance
            vpin_z: VPIN z-score
            depth_ratio: Depth ratio
            event_features: Event-relative time features (optional)

        Returns:
            (action, confidence)
            - action: 0=HOLD, 1=CLOSE
            - confidence: [0, 1]

        """
        if len(bars) > 0:
            self._ingest_price_for_regime(bars[-1][4])

        # Update MFE/MAE
        self._update_mfe_mae(current_price)
        self.ticks_held += 1  # Increment on every tick (not bar close!)

        # Build market state
        market_state = self._build_state(bars, imbalance, vpin_z, depth_ratio, event_features=event_features)

        # Harvester decides exit
        action, confidence = self.harvester.decide(
            market_state=market_state,
            mfe=self.mfe,
            mae=self.mae,
            ticks_held=self.ticks_held,
            entry_price=self.entry_price,
            current_price=current_price,
            direction=self.current_position,
            zeta=self.current_zeta,  # Regime ζ for adaptive hold duration
            predicted_runway=self.predicted_runway,
        )

        if action == 1:  # CLOSE
            LOG.info(
                "[DUAL_POLICY] HARVESTER: CLOSE signal, conf=%.2f, MFE=%.4f, MAE=%.4f, ticks=%d",
                confidence,
                self.mfe,
                self.mae,
                self.ticks_held,
            )

        return action, confidence

    def get_position_metrics(self) -> dict[str, float | int]:
        """Get current position tracking metrics for logging/debugging."""
        return {
            "mfe": self.mfe,
            "mae": self.mae,
            "ticks_held": self.ticks_held,
            "entry_price": self.entry_price,
            "current_position": self.current_position,
        }

    def on_entry(self, direction: int, entry_price: float, entry_time: Any) -> None:
        """Called when position is entered.

        Args:
            direction: +1 for LONG, -1 for SHORT
            entry_price: Entry price
            entry_time: Entry bar timestamp

        """
        # Defensive: Validate inputs
        if entry_price is None or entry_price <= 0:
            LOG.error("[DUAL_POLICY] Invalid entry_price=%.5f - cannot open position", entry_price or 0)
            return

        if direction not in (-1, 1):
            LOG.error("[DUAL_POLICY] Invalid direction=%d - expected 1 (LONG) or -1 (SHORT)", direction)
            return

        # Defensive: Check for orphaned position state
        if self.current_position != 0:
            LOG.warning(
                "[DUAL_POLICY] Position state inconsistency - current=%d but opening new position dir=%d @ %.2f",
                self.current_position,
                direction,
                entry_price,
            )
            # Reset state before opening new position
            self._mfe_calc.reset()
            self.ticks_held = 0

        self.current_position = direction
        self.entry_price = float(entry_price)
        self.entry_bar_time = entry_time
        self._mfe_calc.start(entry_price, direction)
        self.ticks_held = 0
        LOG.info(
            "[DUAL_POLICY] Position entered: %s @ %.2f",
            "LONG" if direction == 1 else "SHORT",
            entry_price,
        )

    def on_recovery(
        self,
        direction: int,
        entry_price: float,
        entry_time: Any,
        mfe: float = 0.0,
        mae: float = 0.0,
        ticks_held: int = 0,
    ) -> None:
        """Called when position is recovered from persistence.
        Unlike on_entry(), this preserves MFE/MAE from the persisted state.

        Args:
            direction: +1 for LONG, -1 for SHORT
            entry_price: Entry price
            entry_time: Entry bar timestamp
            mfe: Maximum favorable excursion (preserved from persistence)
            mae: Maximum adverse excursion (preserved from persistence)
            ticks_held: Number of ticks held (preserved from persistence)

        """
        self.current_position = direction
        self.entry_price = float(entry_price)
        self.entry_bar_time = entry_time
        self._mfe_calc.start(entry_price, direction)
        # Restore persisted MFE/MAE into the calculator
        self._mfe_calc.mfe = float(mfe)
        self._mfe_calc.best_profit = float(mfe)
        self._mfe_calc.mae = float(mae)
        self._mfe_calc.worst_loss = -float(mae) if mae > 0 else 0.0
        self.ticks_held = int(ticks_held)
        LOG.info(
            "[DUAL_POLICY] Position recovered: %s @ %.2f (MFE=%.4f MAE=%.4f ticks=%d)",
            "LONG" if direction == 1 else "SHORT",
            entry_price,
            self.mfe,
            self.mae,
            self.ticks_held,
        )

    def on_exit(
        self,
        exit_price: float,
        capture_ratio: float,
        was_wtl: bool,
        entry_confidence: float = 0.5,
        raw_confidence: float | None = None,
    ) -> None:
        """Called when position is closed.

        Args:
            exit_price: Exit price
            capture_ratio: exit_pnl / MFE
            was_wtl: Was this a winner-to-loser trade?
            entry_confidence: Calibrated trigger confidence recorded at entry (for Platt update)
            raw_confidence: Pre-Platt probability (for correct Platt gradient)

        """
        # Store MFE percentage for harvester's SL learning
        if self.entry_price > 0:
            self.harvester._last_mfe_pct = self.mfe / self.entry_price * 100.0

        # Update agents  with trade outcome
        self.trigger.update_from_trade(
            actual_mfe=self.mfe,
            predicted_runway=self.predicted_runway,
            entry_confidence=entry_confidence,
            entry_price=self.entry_price,
            raw_confidence=raw_confidence,
            predicted_runway_gross=self.predicted_runway_gross,
        )
        self.harvester.update_from_trade(capture_ratio=capture_ratio, was_wtl=was_wtl)

        LOG.info(
            "[DUAL_POLICY] Position closed @ %.2f, MFE=%.4f, Capture=%.2f%%",
            exit_price,
            self.mfe,
            capture_ratio * 100,
        )

        # Reset position state
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_bar_time = None
        self._mfe_calc.reset()
        self.ticks_held = 0
        self.predicted_runway = 0.0
        self.predicted_runway_gross = 0.0

    def _update_mfe_mae(self, current_price: float) -> None:
        """Update MFE and MAE based on current price.

        Delegates to the shared MFEMAECalculator.
        """
        # Ensure calculator is initialized for this position
        if self._mfe_calc.entry_price is None and not SafeMath.is_zero(self.entry_price):
            self._mfe_calc.start(self.entry_price, self.current_position)
        self._mfe_calc.update(current_price)

    def _build_state(
        self,
        bars: deque[Any],
        imbalance: float,
        vpin_z: float,
        depth_ratio: float,
        realized_vol: float = DEFAULT_VOLATILITY,  # Provide RS volatility for geometry calculation
        event_features: dict[str, float] | None = None,  # Phase 3: Event-relative time features
    ) -> np.ndarray:
        """Build normalized state features.

        Features (expandable based on enabled modules):
        Base (7):
            - ret1: 1-bar return
            - ret5: 5-bar return
            - ma_diff: MA fast/slow difference
            - vol: 20-bar volatility
            - imbalance: Order book imbalance
            - vpin_z: VPIN z-score
            - depth_ratio: Bid+ask depth ratio

        Geometry (5) - from handbook:
            - efficiency: Path displacement / path length
            - gamma: Acceleration (2nd derivative)
            - jerk: Rate of change of acceleration (3rd derivative)
            - runway: Inverse volatility pressure
            - feasibility: Composite entry quality score

        Event Time (6) - key session features:
            - london_active: London session active [0, 1]
            - ny_active: New York session active [0, 1]
            - tokyo_active: Tokyo session active [0, 1]
            - london_ny_overlap: High liquidity overlap [0, 1]
            - rollover_proximity: Proximity to 22:00 UTC rollover [-1, 1]
            - week_progress: Week progress [0, 1]

        Returns:
            State array (window, n_features) with features normalized

        """
        # Calculate expected feature dimension (MUST match __init__ dimensions)
        n_features = 7  # Base
        if self.path_geometry:
            n_features += 5  # Geometry features: efficiency, gamma, jerk, runway, feasibility
        if self.enable_event_features:
            n_features += self.event_feature_count  # Event time (always counted when enabled)

        if len(bars) < self.min_bars_for_features:
            return np.zeros((self.window, n_features), dtype=np.float32)

        closes = [b[4] for b in bars]
        c = np.array(closes, dtype=np.float64)

        # Calculate returns
        ret1 = np.zeros_like(c)
        if len(c) >= RETURN_LAG_SHORT:
            ret1[1:] = np.divide(c[1:], c[:-1], out=np.ones_like(c[1:]), where=c[:-1] != 0) - 1.0

        ret5 = np.zeros_like(c)
        if len(c) >= RETURN_LAG_MEDIUM:
            ret5[5:] = np.divide(c[5:], c[:-5], out=np.ones_like(c[5:]), where=c[:-5] != 0) - 1.0

        ma_fast = _dp_rolling_mean(c, 10)
        ma_slow = _dp_rolling_mean(c, 30)
        ma_diff = np.divide(ma_fast, ma_slow, out=np.ones_like(ma_fast), where=ma_slow != 0) - 1.0
        vol = _dp_rolling_std(ret1, 20)

        # Microstructure features (broadcast to window).
        # Clip to instrument-agnostic bounds before broadcasting so the DDQN
        # never sees extreme outliers in these scalar context signals.
        imb = np.full(len(c), np.clip(imbalance, -1.0, 1.0), dtype=np.float64)
        vpz = np.full(len(c), np.clip(vpin_z, -4.0, 4.0), dtype=np.float64)
        dpr = np.full(len(c), np.clip(depth_ratio, 0.1, 10.0), dtype=np.float64)

        # Base features (7-dim)
        base_feats = [
            np.nan_to_num(ret1, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(ret5, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(ma_diff, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(imb, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(vpz, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(dpr, nan=1.0, posinf=1.0, neginf=1.0),
        ]

        # Add path geometry features if available (5-dim)
        if self.path_geometry:
            # Compute long-term vol for multi-horizon ratio (50-bar std of returns)
            sigma_long = float(_dp_rolling_std(ret1, 50)[-1]) if len(ret1) >= 50 else 0.0
            # Update geometry with current bars and volatility
            geom = self.path_geometry.update(bars, realized_vol, sigma_long=sigma_long)  # type: ignore[union-attr]

            # Broadcast geometry features to window length
            eff = np.full(len(c), geom["efficiency"], dtype=np.float64)
            gamma = np.full(len(c), geom["gamma"], dtype=np.float64)
            jerk = np.full(len(c), geom["jerk"], dtype=np.float64)
            runway = np.full(len(c), geom["runway"], dtype=np.float64)
            feasibility = np.full(len(c), geom["feasibility"], dtype=np.float64)

            base_feats.extend(
                [
                    np.nan_to_num(eff, nan=0.0, posinf=0.0, neginf=0.0),
                    np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0),
                    np.nan_to_num(jerk, nan=0.0, posinf=0.0, neginf=0.0),
                    np.nan_to_num(runway, nan=0.5, posinf=0.5, neginf=0.5),
                    np.nan_to_num(feasibility, nan=0.5, posinf=0.5, neginf=0.5),
                ],
            )

        # Add event time features if enabled (6 key features) — always include
        # when self.enable_event_features is True, defaulting to zeros so the
        # feature count stays consistent with the DDQN's fixed state_dim.
        if self.enable_event_features:
            base_feats.extend(_build_event_feature_columns(event_features, len(c)))

        # Stack features (7, 12, 13, or 18-dim depending on modules enabled)
        feats = np.vstack(base_feats).T

        # Take last window bars
        feats = feats[-self.window :].astype(np.float32)

        # Normalize: z-score per feature, but SKIP constant columns (broadcast features)
        # Constant columns (std=0) like imbalance, vpin_z, geometry, event features
        # would get zeroed out by (x-mean)/0 = 0, destroying their signal.
        # Instead, preserve their raw values for the DDQN to learn from.
        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True)
        variable_mask = sd.flatten() > _FEATURE_VARIANCE_FLOOR  # True for columns with actual variance
        # Only normalize variable columns; leave constant columns as-is.
        # Clip to ±5σ after z-scoring to contain market-shock spikes without
        # discarding the signal (features beyond ±5σ carry no extra gradient signal).
        # Note: variable_mask guarantees sd > _FEATURE_VARIANCE_FLOOR, so
        # division is safe.  SafeMath.safe_div is scalar-only; use numpy ops.
        feats[:, variable_mask] = np.clip(
            (feats[:, variable_mask] - mu[:, variable_mask]) / sd[:, variable_mask],
            -5.0,
            5.0,
        )

        return feats

    def _ingest_price_for_regime(self, close_price: float) -> None:
        """Update regime detector with latest close and sync replay buffers."""
        if not self.regime_detector or close_price is None:
            return

        self.current_regime, self.current_zeta = self.regime_detector.add_price(close_price)
        self._sync_replay_buffer_regime()

    def seed_regime_from_bars(self, bars: deque[Any]) -> None:
        """Pre-seed regime detector from historical bar close prices.

        Called once when the regime is still UNKNOWN but historical bars are
        available (e.g. after a restart). Feeds up to ``window_size`` close
        prices so the regime is immediately classified rather than waiting
        50+ bars of live data.
        """
        if not self.regime_detector or self.current_regime != "UNKNOWN":
            return
        window = getattr(self.regime_detector, "window_size", 50)
        seed_bars = list(bars)[-window:]
        if len(seed_bars) < _MIN_SEED_BARS:
            return
        for bar in seed_bars:
            try:
                close = bar[4]  # OHLCV index 4 = close
                if close and close > 0:
                    self.current_regime, self.current_zeta = self.regime_detector.add_price(close)
            except (IndexError, TypeError):
                continue
        self._sync_replay_buffer_regime()
        LOG.info(
            "[REGIME] Seeded from %d historical bars → regime=%s zeta=%.3f",
            len(seed_bars),
            self.current_regime,
            self.current_zeta,
        )

    def _sync_replay_buffer_regime(self) -> None:
        """Align replay buffer sampling with current regime classification."""
        regime_map = {
            "TRENDING": RegimeSampling.TRENDING,
            "MEAN_REVERTING": RegimeSampling.MEAN_REVERTING,
            "TRANSITIONAL": RegimeSampling.TRANSITIONAL,
            "UNKNOWN": RegimeSampling.UNKNOWN,
        }
        new_enum = regime_map.get(self.current_regime, RegimeSampling.UNKNOWN)
        self.current_regime_enum = new_enum

        for agent in (self.trigger, self.harvester):
            buffer = getattr(agent, "buffer", None)
            if getattr(agent, "enable_training", False) and buffer and hasattr(buffer, "set_current_regime"):
                buffer.set_current_regime(self.current_regime_enum)
                if hasattr(buffer, "set_current_zeta"):
                    buffer.set_current_zeta(self.current_zeta)

    # -------------------------------------------------------------------------
    # Online Learning Methods
    # -------------------------------------------------------------------------
    def add_trigger_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = True,
    ) -> None:
        """Add experience to TriggerAgent buffer for online learning.

        Args:
            state: State at entry decision time (12-dim)
            action: 0=NO_ENTRY, 1=LONG, 2=SHORT
            reward: Shaped reward from RewardShaper
            next_state: State after trade closed
            done: Episode terminal (True for completed trade)

        """
        LOG.debug(
            "[TRIGGER-EXPERIENCE-DIAG] add_trigger_experience called: enable=%s, buffer=%s, action=%d, reward=%.4f",
            self.enable_training,
            self.trigger.buffer is not None if self.trigger else None,
            action,
            reward,
        )
        if not self.enable_training:
            LOG.warning("[TRIGGER-EXPERIENCE-DIAG] SKIPPED — enable_training=%s", self.enable_training)
            return

        LOG.debug(
            "[TRIGGER-EXPERIENCE-DIAG] Adding experience: "
            "(state_shape=%s, action=%d, reward=%.4f, enable_training=%s, regime=%s)",
            state.shape,
            action,
            reward,
            self.enable_training,
            self.current_regime_enum,
        )
        self.trigger.add_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            regime=self.current_regime_enum,
        )
        LOG.debug(
            "[TRIGGER-EXPERIENCE-DIAG] DONE — buffer_size=%d",
            self.trigger.buffer.size if self.trigger.buffer else -1,
        )

    def add_harvester_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = True,
    ) -> None:
        """Add experience to HarvesterAgent buffer for online learning.

        Args:
            state: State at exit decision time (10-dim)
            action: 0=HOLD, 1=CLOSE
            reward: Shaped reward from RewardShaper
            next_state: State after action
            done: Episode terminal (True for position closed)

        """
        if not self.enable_training:
            LOG.debug("[DIAG] add_harvester_experience: SKIPPED — enable_training=%s", self.enable_training)
            return

        LOG.debug(
            "[DIAG] add_harvester_experience: CALLING harvester.add_experience "
            "(state_shape=%s, action=%d, reward=%.4f, regime=%s)",
            state.shape,
            action,
            reward,
            self.current_regime_enum,
        )
        self.harvester.add_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            regime=self.current_regime_enum,
        )
        LOG.debug(
            "[DIAG] add_harvester_experience: DONE — buffer_size=%d",
            self.harvester.buffer.size if self.harvester.buffer else -1,
        )

    # Maximum gradient steps per training call (multi-step accelerates convergence)
    _MAX_STEPS_PER_TRAIN: int = 4

    def _agent_multi_step(self, agent: Any) -> dict[str, Any] | None:
        """Run up to _MAX_STEPS_PER_TRAIN gradient steps on one agent.

        Returns the metrics from the last successful step, or None.
        """
        buf = getattr(agent, "buffer", None)
        if buf is None:
            return agent.train_step()

        batch_size = getattr(agent, "batch_size", 64)
        n_entries = buf.tree.n_entries
        # Number of non-overlapping batches available, capped at _MAX_STEPS_PER_TRAIN
        n_steps = min(self._MAX_STEPS_PER_TRAIN, max(1, n_entries // batch_size))

        last_metrics = None
        for _ in range(n_steps):
            m = agent.train_step()
            if m is not None:
                last_metrics = m
        return last_metrics

    def train_step(self, adaptive_reg: Any = None) -> dict[str, Any]:
        """Execute multi-step training on both agents.

        Runs up to _MAX_STEPS_PER_TRAIN gradient steps per agent per call,
        accelerating Q-value convergence and improving confidence scores.

        Args:
            adaptive_reg: Optional AdaptiveRegularization instance for L2/dropout adjustment

        Returns:
            Dictionary with training metrics from both agents

        """
        if not self.enable_training:
            return {"trigger": None, "harvester": None}

        metrics = {}

        # Get current regularization if provided (for logging / future per-step tuning)
        reg_params = adaptive_reg.get_current() if adaptive_reg else {}
        if reg_params:
            metrics["adaptive_reg"] = reg_params

        # Multi-step train TriggerAgent
        trigger_metrics = self._agent_multi_step(self.trigger)
        metrics["trigger"] = trigger_metrics

        # Multi-step train HarvesterAgent
        harvester_metrics = self._agent_multi_step(self.harvester)
        metrics["harvester"] = harvester_metrics

        # Log training summary
        if trigger_metrics or harvester_metrics:
            LOG.info(
                "[TRAIN] Trigger: loss=%.4f td=%.4f | Harvester: loss=%.4f td=%.4f",
                trigger_metrics.get("loss", 0.0) if trigger_metrics else 0.0,
                trigger_metrics.get("mean_td_error", 0.0) if trigger_metrics else 0.0,
                harvester_metrics.get("loss", 0.0) if harvester_metrics else 0.0,
                harvester_metrics.get("mean_td_error", 0.0) if harvester_metrics else 0.0,
            )

        return metrics

    def get_training_stats(self) -> dict[str, Any]:
        """Get training statistics from both agents."""
        return {
            "trigger": (self.trigger.get_training_stats() if hasattr(self.trigger, "get_training_stats") else {}),
            "harvester": (self.harvester.get_training_stats() if hasattr(self.harvester, "get_training_stats") else {}),
            "enable_training": self.enable_training,
        }

    # ------------------------------------------------------------------
    # Persistence: save / load training state across restarts
    # ------------------------------------------------------------------

    def _save_agent_weights(self, agent: Any, label: str, path: str) -> bool:
        """Save DDQN weights for one agent. Returns False on failure."""
        if agent.ddqn is None:
            return True
        try:
            agent.ddqn.save_weights(path)
            return True
        except Exception as e:
            LOG.exception("[CHECKPOINT] Failed to save %s weights: %s", label, e)
            return False

    def _default_checkpoint_dir(self) -> Path:
        """Return this policy's symbol/timeframe-specific checkpoint directory."""
        data_dir = Path(os.environ.get("CTRADER_DATA_DIR", "data"))
        sym = _safe_path_token(self.symbol)
        tf = _safe_path_token(self.timeframe)
        return data_dir / "checkpoints" / f"{sym}_{tf}"

    def save_checkpoint(self, checkpoint_dir: str | Path | None = None) -> bool:
        """Save full training state: DDQN weights, buffers, epsilon, training_steps.

        Called during graceful shutdown to preserve training progress.

        Args:
            checkpoint_dir: Directory to store checkpoint files

        Returns:
            True if all saves succeeded

        """
        checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else self._default_checkpoint_dir()
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        success = True

        # 1. Save DDQN weights
        success &= self._save_agent_weights(self.trigger, "trigger", str(checkpoint_path / "trigger_ddqn_weights.pt"))
        success &= self._save_agent_weights(
            self.harvester, "harvester", str(checkpoint_path / "harvester_ddqn_weights.pt"),
        )

        # 2. Save experience buffers
        if self.trigger.buffer is not None and not self.trigger.buffer.save(str(checkpoint_path / "trigger_buffer")):
            success = False

        if self.harvester.buffer is not None and not self.harvester.buffer.save(
            str(checkpoint_path / "harvester_buffer"),
        ):
            success = False

        # 3. Save training metadata (epsilon, steps, calibration, etc.)
        metadata = {
            "saved_at": dt.datetime.now(dt.UTC).isoformat(),
            "trigger_training_steps": self.trigger.training_steps,
            "trigger_epsilon": self.trigger.epsilon,
            "trigger_epsilon_decay": self.trigger.epsilon_decay,
            "harvester_training_steps": self.harvester.training_steps,
            "trigger_platt_a": getattr(self.trigger, "platt_a", 1.0),
            "trigger_platt_b": getattr(self.trigger, "platt_b", 0.0),
        }
        # Persist runway calibration buckets (survive restarts)
        if hasattr(self.trigger, "get_calibration_state"):
            metadata["trigger_calibration"] = self.trigger.get_calibration_state()
        try:
            meta_path = checkpoint_path / "training_metadata.json"
            save_json_atomic(meta_path, metadata)
            LOG.info("[CHECKPOINT] Saved training metadata: %s", metadata)
        except Exception as e:
            LOG.exception("[CHECKPOINT] Failed to save metadata: %s", e)
            success = False

        # 4. Save regime detector state (price buffer) so regime survives restarts
        if self.regime_detector:
            try:
                regime_state = {
                    "price_buffer": list(self.regime_detector.price_buffer),
                    "current_regime": self.current_regime,
                    "current_zeta": self.current_zeta,
                }
                regime_path = checkpoint_path / "regime_state.json"
                save_json_atomic(regime_path, regime_state)
                LOG.debug(
                    "[CHECKPOINT] Saved regime state: regime=%s, %d prices",
                    self.current_regime,
                    len(self.regime_detector.price_buffer),
                )
            except Exception as e:
                LOG.warning("[CHECKPOINT] Failed to save regime state: %s", e)

        if success:
            LOG.info("[CHECKPOINT] ✓ Full checkpoint saved to %s", checkpoint_path)
        else:
            LOG.warning("[CHECKPOINT] Checkpoint saved with some failures to %s", checkpoint_path)

        return success

    def _ckpt_load_weights(self, cp: Path) -> bool:
        """Load DDQN weights for trigger and harvester. Returns True if any loaded."""
        loaded = False
        for agent_name, agent in [("trigger", self.trigger), ("harvester", self.harvester)]:
            if agent.ddqn is None:
                continue
            # Try .pt first (current format), fall back to .npz (legacy)
            pt_path = cp / f"{agent_name}_ddqn_weights.pt"
            npz_path = cp / f"{agent_name}_ddqn_weights.npz"
            weight_path = pt_path if pt_path.exists() else (npz_path if npz_path.exists() else None)
            if weight_path is None:
                continue
            try:
                agent.ddqn.load_weights(str(weight_path))
                loaded = True
            except Exception as e:
                LOG.exception("[CHECKPOINT] Failed to load %s weights: %s", agent_name, e)
        return loaded

    def _ckpt_load_buffers(self, cp: Path) -> bool:
        """Load experience replay buffers. Returns True if any loaded."""
        loaded = False
        trigger_buf = cp / "trigger_buffer.npz"
        if trigger_buf.exists() and self.trigger.buffer is not None and self.trigger.buffer.load(str(trigger_buf)):
            loaded = True
        harvester_buf = cp / "harvester_buffer.npz"
        if (
            harvester_buf.exists()
            and self.harvester.buffer is not None
            and self.harvester.buffer.load(str(harvester_buf))
        ):
            loaded = True
        return loaded

    def _ckpt_load_metadata(self, cp: Path) -> bool:
        """Load training metadata (steps, epsilon, Platt params). Returns True if loaded."""
        meta_path = cp / "training_metadata.json"
        if not meta_path.exists():
            return False
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
            self.trigger.training_steps = metadata.get("trigger_training_steps", 0)
            self.trigger.epsilon = metadata.get("trigger_epsilon", self.trigger.epsilon)
            # Clamp epsilon to the floor configured in .env so stale checkpoints
            # cannot lock the bot below the minimum exploration rate. Default floor
            # is the configured paper floor so stale checkpoints cannot lock
            # the bot below the minimum exploration rate.
            _env_floor = float(os.environ.get("EPSILON_END", str(getattr(self.trigger, "epsilon_end", 0.1))))
            self.trigger.epsilon = max(_env_floor, self.trigger.epsilon)
            if "trigger_epsilon_decay" in metadata:
                _metadata_decay = float(metadata["trigger_epsilon_decay"])
                _configured_decay = float(
                    os.environ.get("EPSILON_DECAY", str(getattr(self.trigger, "epsilon_decay", 0.998))),
                )
                if getattr(self.trigger, "paper_mode", False):
                    self.trigger.epsilon_decay = max(_configured_decay, _metadata_decay)
                else:
                    self.trigger.epsilon_decay = _metadata_decay
            self.harvester.training_steps = metadata.get("harvester_training_steps", 0)
            if self.trigger.ddqn is not None:
                self.trigger.ddqn.training_steps = self.trigger.training_steps
            if self.harvester.ddqn is not None:
                self.harvester.ddqn.training_steps = self.harvester.training_steps
            restored_at = metadata.get("last_training_time") or metadata.get("saved_at")
            if not restored_at:
                restored_at = dt.datetime.fromtimestamp(meta_path.stat().st_mtime, dt.UTC).isoformat()
            if self.trigger.training_steps > 0 and not getattr(self.trigger, "_last_training_time", ""):
                self.trigger._last_training_time = restored_at
            if self.harvester.training_steps > 0 and not getattr(self.harvester, "_last_training_time", ""):
                self.harvester._last_training_time = restored_at
            if hasattr(self.trigger, "platt_a"):
                self.trigger.platt_a = metadata.get("trigger_platt_a", 1.0)
                self.trigger.platt_b = metadata.get("trigger_platt_b", 0.0)
            # Restore runway calibration buckets
            cal_state = metadata.get("trigger_calibration")
            if cal_state and hasattr(self.trigger, "load_calibration_state"):
                self.trigger.load_calibration_state(cal_state)
            LOG.info("[CHECKPOINT] Restored metadata: %s", metadata)
            return True
        except Exception as e:
            LOG.exception("[CHECKPOINT] Failed to load metadata: %s", e)
            return False

    def _ckpt_restore_regime(self, cp: Path) -> bool:
        """Restore regime detector state from checkpoint. Returns True if loaded."""
        regime_path = cp / "regime_state.json"
        if not (regime_path.exists() and self.regime_detector):
            return False
        try:
            with open(regime_path) as f:
                regime_state = json.load(f)
            prices = regime_state.get("price_buffer", [])
            if prices:
                self.regime_detector.price_buffer = deque(prices, maxlen=self.regime_detector.window_size)
                self.regime_detector._update_regime()
                self.current_regime = self.regime_detector.current_regime
                self.current_zeta = self.regime_detector.current_zeta
                self._sync_replay_buffer_regime()
                LOG.info(
                    "[CHECKPOINT] Restored regime state: regime=%s zeta=%.3f (%d prices)",
                    self.current_regime,
                    self.current_zeta,
                    len(prices),
                )
                return True
        except Exception as e:
            LOG.warning("[CHECKPOINT] Failed to restore regime state: %s", e)
        return False

    def load_checkpoint(self, checkpoint_dir: str | Path | None = None) -> bool:
        """Load training state from a previous checkpoint.

        Called during startup to resume training from where it left off.

        Args:
            checkpoint_dir: Directory containing checkpoint files

        Returns:
            True if checkpoint was found and loaded (at least partially)

        """
        cp = Path(checkpoint_dir) if checkpoint_dir is not None else self._default_checkpoint_dir()
        if not cp.exists():
            LOG.info("[CHECKPOINT] No checkpoint directory found at %s", cp)
            return False

        loaded_anything = (
            self._ckpt_load_weights(cp)
            | self._ckpt_load_buffers(cp)
            | self._ckpt_load_metadata(cp)
            | self._ckpt_restore_regime(cp)
        )

        if loaded_anything:
            LOG.info(
                "[CHECKPOINT] ✓ Checkpoint loaded — Trigger: steps=%d eps=%.4f buf=%d | Harvester: steps=%d buf=%d",
                self.trigger.training_steps,
                self.trigger.epsilon,
                self.trigger.buffer.size if self.trigger.buffer else 0,
                self.harvester.training_steps,
                self.harvester.buffer.size if self.harvester.buffer else 0,
            )
        else:
            LOG.info("[CHECKPOINT] No checkpoint data found in %s", cp)

        return loaded_anything


# ============================================================================
# Self-Test
# ============================================================================
if __name__ == "__main__":
    import datetime as dt

    logging.basicConfig(level=logging.INFO)

    # Test 1: Initialize
    policy = DualPolicy(window=64)
    assert policy.current_position == 0
    assert policy.trigger is not None
    assert policy.harvester is not None

    # Test 2: Entry decision (flat)
    bars = deque(maxlen=100)
    for i in range(100):
        t = dt.datetime.now(tz=dt.UTC)
        o = h = lo = c = 100000.0 + i * 10
        bars.append((t, o, h, lo, c))

    action, conf, runway = policy.decide_entry(bars, imbalance=0.1)
    assert action in [0, 1, 2]
    assert 0 <= conf <= 1
    assert runway >= 0

    # Test 3: Enter position
    policy.on_entry(direction=1, entry_price=TEST_ENTRY_PRICE, entry_time=dt.datetime.now(tz=dt.UTC))
    assert policy.current_position == 1
    assert abs(policy.entry_price - TEST_ENTRY_PRICE) < 1e-6

    # Test 4: Exit decision (in position)
    current_price = 100050.0  # Small profit
    action, conf = policy.decide_exit(bars, current_price, imbalance=0.1)
    assert action in [0, 1]  # HOLD or CLOSE
    assert 0 <= conf <= 1
    assert policy.mfe > 0  # Should have tracked MFE

    # Test 5: Exit position
    policy.on_exit(exit_price=100050.0, capture_ratio=0.8, was_wtl=False)
    assert policy.current_position == 0
    assert SafeMath.is_zero(policy.mfe)
