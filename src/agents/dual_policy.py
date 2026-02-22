#!/usr/bin/env python3
"""
Dual Policy - Orchestrates Trigger and Harvester Agents (Phase 3)
=================================================================
Coordinates entry and exit specialists for dual-agent trading.

Architecture:
- TriggerAgent: Entry specialist (when to enter, which direction)
- HarvesterAgent: Exit specialist (when to close position)

From MASTER_HANDBOOK.md Section 2.2: Dual-Agent Architecture
"""

import logging
from collections import deque

import numpy as np

from src.agents.harvester_agent import HarvesterAgent
from src.agents.trigger_agent import TriggerAgent
from src.features.regime_detector import RegimeDetector  # Phase 3.4
from src.persistence.learned_parameters import LearnedParametersManager
from src.utils.experience_buffer import RegimeSampling
from src.utils.safe_math import SafeMath

LOG = logging.getLogger(__name__)

# Feature calculation constants
MIN_BARS_FOR_FEATURES: int = 70
RETURN_LAG_SHORT: int = 2
RETURN_LAG_MEDIUM: int = 6
TEST_ENTRY_PRICE: float = 100000.0
_FEATURE_VARIANCE_FLOOR: float = 1e-6  # minimum std to treat a feature column as variable
_MIN_SEED_BARS: int = 3                # minimum bars required to seed the regime detector


class DualPolicy:
    """
    Orchestrates TriggerAgent and HarvesterAgent for specialized trading.

    Workflow:
    1. On bar close (flat): trigger.decide_entry() → LONG/SHORT/NONE
    2. On bar close (in position): harvester.decide_exit() → HOLD/CLOSE
    3. Track position state (MFE, MAE, ticks_held) for harvester

    Backward Compatibility:
    - If DDQN_DUAL_AGENT=0: Falls back to single Policy
    - If DDQN_DUAL_AGENT=1: Uses dual-agent architecture
    """

    def __init__(  # noqa: PLR0913
        self,
        window: int = 64,
        enable_regime_detection: bool = True,
        path_geometry=None,
        enable_training: bool = False,
        enable_event_features: bool = True,
        param_manager: LearnedParametersManager | None = None,
        symbol: str = "XAUUSD",  # Instrument-agnostic: default for tests/demos
        timeframe: str = "M15",
        broker: str = "default",
        timeframe_minutes: int = 5,
        min_bars_for_features: int = 70,
        friction_calculator=None,
        trigger_buffer_capacity: int = 2_000,
        harvester_buffer_capacity: int = 10_000,
    ):
        """
        Initialize DualPolicy with trigger and harvester agents.

        Args:
            window: Lookback window for state
            enable_regime_detection: Enable Phase 3.4 regime detection (default True)
            path_geometry: PathGeometry instance for entry features (optional)
            enable_training: Enable online learning with PER buffer (default False)
            enable_event_features: Enable Phase 3 event-relative time features (default True)
            friction_calculator: FrictionCalculator for cost-aware exit decisions
        """
        self.window = window
        self.enable_training = enable_training
        self.enable_event_features = enable_event_features
        self.param_manager = param_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.timeframe_minutes = timeframe_minutes
        self.broker = broker
        self.friction_calculator = friction_calculator
        # Scale minimum-bars threshold to wall-clock time so higher timeframes
        # don't produce zero-state for absurd durations (H4 would need 11 days!).
        self.min_bars_for_features = min_bars_for_features

        # Calculate feature dimensions (base=7, geometry=5, event=6)
        base_features = 7
        geometry_features = 5 if path_geometry else 0
        self.event_feature_count = 6 if enable_event_features else 0

        trigger_features = base_features + geometry_features + self.event_feature_count
        harvester_market_features = trigger_features
        harvester_total_features = harvester_market_features + 3  # +3 position stats (MFE/MAE/bars)

        self.trigger = TriggerAgent(
            window=window,
            n_features=trigger_features,
            enable_training=enable_training,
            symbol=self.symbol,
            timeframe=self.timeframe,
            broker=self.broker,
            param_manager=self.param_manager,
            timeframe_minutes=timeframe_minutes,
            buffer_capacity=trigger_buffer_capacity,
        )
        self.harvester = HarvesterAgent(
            window=window,
            n_features=harvester_total_features,
            enable_training=enable_training,
            symbol=self.symbol,
            timeframe=self.timeframe,
            broker=self.broker,
            param_manager=self.param_manager,
            friction_calculator=self.friction_calculator,
            timeframe_minutes=timeframe_minutes,
            buffer_capacity=harvester_buffer_capacity,
        )

        LOG.info("[DUAL_POLICY] TriggerAgent: %d features (7 base + 5 geometry + 6 event)", trigger_features)
        LOG.info("[DUAL_POLICY] HarvesterAgent: %d features (market + position)", harvester_total_features)

        # Path geometry for entry features
        self.path_geometry = path_geometry

        # Regime detection
        self.enable_regime_detection = enable_regime_detection
        if self.enable_regime_detection:
            # Scale update_interval inversely with timeframe so regime reacts
            # at roughly the same wall-clock frequency regardless of bar size.
            # M5 → every 5 bars; H1 → every 1 bar; H4+ → every 1 bar
            update_interval = max(1, min(5, int(5 * 5 / max(1, timeframe_minutes))))
            self.regime_detector = RegimeDetector(window_size=50, update_interval=update_interval)
        else:
            self.regime_detector = None

        # Position tracking for harvester
        self.current_position = 0  # -1=SHORT, 0=FLAT, +1=LONG
        self.entry_price = 0.0
        self.entry_bar_time = None
        self.mfe = 0.0  # Maximum favorable excursion
        self.mae = 0.0  # Maximum adverse excursion
        self.ticks_held = 0  # Number of market data ticks (not bars!)
        self.predicted_runway = 0.0  # From trigger agent

        # Phase 3.4: Regime state
        self.current_regime = "UNKNOWN"
        self.current_zeta = 1.0
        self.current_regime_enum = RegimeSampling.UNKNOWN

        LOG.info("[DUAL_POLICY] Initialized with TriggerAgent + HarvesterAgent")

    def decide_entry(  # noqa: PLR0913
        self,
        bars: deque,
        imbalance: float = 0.0,
        vpin_z: float = 0.0,
        depth_ratio: float = 1.0,
        realized_vol: float = 0.005,  # For economics calculations
        event_features: dict = None,  # Phase 3: Event-relative time features
    ) -> tuple[int, float, float]:
        """
        Decide entry action using TriggerAgent.

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
        # Phase 3.4: Update regime detection with latest price
        if len(bars) > 0:
            self._ingest_price_for_regime(bars[-1][4])

        # Build state (includes path geometry and event features if available)
        state = self._build_state(bars, imbalance, vpin_z, depth_ratio, realized_vol, event_features)

        # Phase 3.4: Get regime threshold adjustment for trigger
        regime_threshold_adj = 0.0
        if self.regime_detector:
            regime_threshold_adj = self.regime_detector.get_trigger_threshold_adjustment()

        # Phase 2: Get path geometry feasibility
        feasibility = 1.0
        if self.path_geometry:
            feasibility = self.path_geometry.last.get("feasibility", 1.0)

        # Regime-confidence gate: low ζ → scale down effective feasibility so
        # the trigger's hard gate demands a cleaner setup in uncertain regimes.
        # ζ=1.0: no change.  ζ=0.5: feasibility × 0.75.  ζ=0: feasibility × 0.5.
        _zeta = max(0.0, min(1.0, self.current_zeta))
        if _zeta < 1.0:
            _zeta_scale = 0.5 + 0.5 * _zeta   # maps [0,1] → [0.5, 1.0]
            _raw_feas = feasibility
            feasibility = feasibility * _zeta_scale
            LOG.debug(
                "[DUAL_POLICY] ζ=%.2f → feasibility %.3f → %.3f (regime uncertainty gate)",
                _zeta, _raw_feas, feasibility,
            )

        # Phase 2: Calculate economics parameters
        # Expected gain/loss based on realized volatility and typical move sizes
        expected_gain = realized_vol * 2.0  # Expect 2σ move on winning trades
        expected_loss = realized_vol * 1.0  # Risk 1σ on losing trades

        # Calculate actual friction from broker data (spread, commission, swap, slippage)
        # Uses actual broker commission rates and swap fees (varies by instrument)
        # SAME CODE PATH for both paper trading and live trading
        if self.friction_calculator and len(bars) > 0:
            current_price = bars[-1][4]
            # Calculate friction for typical M5 intraday trade
            # CRITICAL: M5 trades (~2-3 hours) typically DON'T cross rollover → swap = 0
            # Swap only charged if position held through daily rollover (5pm EST/10pm UTC)
            friction_data = self.friction_calculator.calculate_total_friction(
                quantity=0.10,
                side="BUY",  # Use BUY as reference (SELL has similar costs)
                price=current_price,
                holding_days=0.1,  # ~2.4 hours for M5 trades
                volatility_factor=1.0,
                crosses_rollover=False,  # Intraday trades don't cross rollover → swap = 0
            )
            # Convert USD total to price units (for XAUUSD @ $4600: $6-7 friction = 0.0015 price units)
            # Breakdown: spread (~$1.6) + commission (~$0.8) + swap ($0) + slippage (~$1.0) = ~$3.4
            friction_cost = friction_data["total"] / current_price if current_price > 0 else 0.0002
        else:
            # Conservative fallback: 0.03% (spread + commission + slippage, no swap)
            friction_cost = 3.0 * 0.0001

        # Call TriggerAgent with friction costs and economics parameters
        action, confidence, predicted_runway = self.trigger.decide(
            state=state,
            current_position=self.current_position,
            regime_threshold_adj=regime_threshold_adj,  # Phase 3.4
            feasibility=feasibility,  # Phase 2: Hard gate
            expected_gain=expected_gain,  # Phase 2: Economics
            expected_loss=expected_loss,
            friction_cost=friction_cost,  # Phase 2: Actual broker friction (commission + swap + spread + slippage)
        )

        # Phase 3.4: Apply regime-aware runway adjustment
        if action in [1, 2] and self.regime_detector:  # LONG or SHORT
            regime_multiplier = self.regime_detector.get_regime_multiplier()
            predicted_runway_adjusted = predicted_runway * regime_multiplier

            LOG.info(
                "[DUAL_POLICY] TRIGGER: %s entry, conf=%.2f, runway=%.4f (base=%.4f, regime=%s, mult=%.2fx)",
                "LONG" if action == 1 else "SHORT",
                confidence,
                predicted_runway_adjusted,
                predicted_runway,
                self.current_regime,
                regime_multiplier,
            )

            self.predicted_runway = predicted_runway_adjusted
        elif action in [1, 2]:  # No regime detection
            self.predicted_runway = predicted_runway
            LOG.info(
                "[DUAL_POLICY] TRIGGER: %s entry, conf=%.2f, predicted_runway=%.4f",
                "LONG" if action == 1 else "SHORT",
                confidence,
                predicted_runway,
            )

        return action, confidence, predicted_runway

    def decide_exit(  # noqa: PLR0913
        self,
        bars: deque,
        current_price: float,
        imbalance: float = 0.0,
        vpin_z: float = 0.0,
        depth_ratio: float = 1.0,
        event_features: dict = None,
    ) -> tuple[int, float]:
        """
        Decide exit action using HarvesterAgent.

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

    def get_position_metrics(self) -> dict:
        """Get current position tracking metrics for logging/debugging."""
        return {
            "mfe": self.mfe,
            "mae": self.mae,
            "ticks_held": self.ticks_held,
            "entry_price": self.entry_price,
            "current_position": self.current_position,
        }

    def on_entry(self, direction: int, entry_price: float, entry_time):
        """
        Called when position is entered.

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
            self.mfe = 0.0
            self.mae = 0.0
            self.ticks_held = 0

        self.current_position = direction
        self.entry_price = float(entry_price)
        self.entry_bar_time = entry_time
        self.mfe = 0.0
        self.mae = 0.0
        self.ticks_held = 0
        LOG.info(
            "[DUAL_POLICY] Position entered: %s @ %.2f",
            "LONG" if direction == 1 else "SHORT",
            entry_price,
        )

    def on_recovery(  # noqa: PLR0913
        self, direction: int, entry_price: float, entry_time, mfe: float = 0.0, mae: float = 0.0, ticks_held: int = 0
    ):
        """
        Called when position is recovered from persistence.
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
        self.mfe = float(mfe)
        self.mae = float(mae)
        self.ticks_held = int(ticks_held)
        LOG.info(
            "[DUAL_POLICY] Position recovered: %s @ %.2f (MFE=%.4f MAE=%.4f ticks=%d)",
            "LONG" if direction == 1 else "SHORT",
            entry_price,
            self.mfe,
            self.mae,
            self.ticks_held,
        )

    def on_exit(self, exit_price: float, capture_ratio: float, was_wtl: bool, entry_confidence: float = 0.5):
        """
        Called when position is closed.

        Args:
            exit_price: Exit price
            capture_ratio: exit_pnl / MFE
            was_wtl: Was this a winner-to-loser trade?
            entry_confidence: Calibrated trigger confidence recorded at entry (for Platt update)
        """
        # Store MFE percentage for harvester's SL learning
        if self.entry_price > 0:
            self.harvester._last_mfe_pct = (self.mfe / self.entry_price) * 100.0

        # Update agents  with trade outcome
        self.trigger.update_from_trade(
            actual_mfe=self.mfe,
            predicted_runway=self.predicted_runway,
            entry_confidence=entry_confidence,
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
        self.mfe = 0.0
        self.mae = 0.0
        self.ticks_held = 0
        self.predicted_runway = 0.0

    def _update_mfe_mae(self, current_price: float):
        """Update MFE and MAE based on current price."""
        # Defensive: Validate entry_price (guard against zero — no position open)
        if SafeMath.is_zero(self.entry_price):
            return

        # Defensive: Validate position direction
        if self.current_position not in (1, -1):
            LOG.debug("[DUAL_POLICY] No position to update MFE/MAE (direction=%d)", self.current_position)
            return

        # Convert prices to float, falling back to 0.0 for non-numeric types.
        # A 0.0 fallback lets the calculation proceed so callers can observe the
        # resulting MFE/MAE (e.g. LONG at ep=100 with cp=0 → mae=100, correct
        # worst-case adverse excursion).
        try:
            cp = float(current_price) if current_price is not None else 0.0
        except (TypeError, ValueError) as e:
            LOG.error("[DUAL_POLICY] Price conversion error for current_price: %s", e)
            cp = 0.0

        try:
            ep = float(self.entry_price)
        except (TypeError, ValueError) as e:
            LOG.error("[DUAL_POLICY] Entry price conversion error: %s", e)
            ep = 0.0

        if self.current_position == 1:  # LONG
            profit = cp - ep
            self.mfe = max(self.mfe, profit)
            self.mae = max(self.mae, -profit)
        elif self.current_position == -1:  # SHORT
            profit = ep - cp
            self.mfe = max(self.mfe, profit)
            self.mae = max(self.mae, -profit)


    def _build_state(  # noqa: PLR0913, PLR0915
        self,
        bars: deque,
        imbalance: float,
        vpin_z: float,
        depth_ratio: float,
        realized_vol: float = 0.005,  # Provide RS volatility for geometry calculation
        event_features: dict = None,  # Phase 3: Event-relative time features
    ) -> np.ndarray:
        """
        Build normalized state features.

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

        # Moving averages
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
            # Update geometry with current bars and volatility
            geom = self.path_geometry.update(bars, realized_vol)

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
                ]
            )

        # Add event time features if enabled (6 key features) — always include
        # when self.enable_event_features is True, defaulting to zeros so the
        # feature count stays consistent with the DDQN's fixed state_dim.
        if self.enable_event_features:
            if event_features:
                london_active = event_features.get("london_active", 0.0)
                ny_active = event_features.get("ny_active", 0.0)
                tokyo_active = event_features.get("tokyo_active", 0.0)
                london_ny_overlap = event_features.get("london_ny_overlap", 0.0)
                rollover_proximity = event_features.get("rollover_proximity_norm", 0.0)
                week_progress = event_features.get("week_progress", 0.5)
            else:
                # No event data available — use neutral defaults
                london_active = 0.0
                ny_active = 0.0
                tokyo_active = 0.0
                london_ny_overlap = 0.0
                rollover_proximity = 0.0
                week_progress = 0.5

            # Broadcast event features to window length
            base_feats.extend(
                [
                    np.full(len(c), london_active, dtype=np.float64),
                    np.full(len(c), ny_active, dtype=np.float64),
                    np.full(len(c), tokyo_active, dtype=np.float64),
                    np.full(len(c), london_ny_overlap, dtype=np.float64),
                    np.full(len(c), rollover_proximity, dtype=np.float64),
                    np.full(len(c), week_progress, dtype=np.float64),
                ]
            )

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
        feats[:, variable_mask] = np.clip(
            (feats[:, variable_mask] - mu[:, variable_mask]) / (sd[:, variable_mask] + 1e-8),
            -5.0, 5.0,
        )

        return feats

    def _ingest_price_for_regime(self, close_price: float):
        """Update regime detector with latest close and sync replay buffers."""
        if not self.regime_detector or close_price is None:
            return

        self.current_regime, self.current_zeta = self.regime_detector.add_price(close_price)
        self._sync_replay_buffer_regime()

    def seed_regime_from_bars(self, bars) -> None:
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
            len(seed_bars), self.current_regime, self.current_zeta,
        )

    def _sync_replay_buffer_regime(self):
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
    ):
        """
        Add experience to TriggerAgent buffer for online learning.

        Args:
            state: State at entry decision time (12-dim)
            action: 0=NO_ENTRY, 1=LONG, 2=SHORT
            reward: Shaped reward from RewardShaper
            next_state: State after trade closed
            done: Episode terminal (True for completed trade)
        """
        if not self.enable_training:
            return

        self.trigger.add_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            regime=self.current_regime_enum,
        )

    def add_harvester_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = True,
    ):
        """
        Add experience to HarvesterAgent buffer for online learning.

        Args:
            state: State at exit decision time (10-dim)
            action: 0=HOLD, 1=CLOSE
            reward: Shaped reward from RewardShaper
            next_state: State after action
            done: Episode terminal (True for position closed)
        """
        if not self.enable_training:
            return

        self.harvester.add_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            regime=self.current_regime_enum,
        )

    def train_step(self, adaptive_reg=None) -> dict:
        """
        Execute one training step on both agents.

        Args:
            adaptive_reg: Optional AdaptiveRegularization instance for L2/dropout adjustment

        Returns:
            Dictionary with training metrics from both agents
        """
        if not self.enable_training:
            return {"trigger": None, "harvester": None}

        metrics = {}

        # Get current regularization if provided
        adaptive_reg.get_current() if adaptive_reg else None

        # Train TriggerAgent
        trigger_metrics = self.trigger.train_step()
        metrics["trigger"] = trigger_metrics

        # Train HarvesterAgent
        harvester_metrics = self.harvester.train_step()
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

    def get_training_stats(self) -> dict:
        """Get training statistics from both agents."""
        return {
            "trigger": (self.trigger.get_training_stats() if hasattr(self.trigger, "get_training_stats") else {}),
            "harvester": (self.harvester.get_training_stats() if hasattr(self.harvester, "get_training_stats") else {}),
            "enable_training": self.enable_training,
        }

    # ------------------------------------------------------------------
    # Persistence: save / load training state across restarts
    # ------------------------------------------------------------------

    def _save_agent_weights(self, agent, label: str, path: str) -> bool:
        """Save DDQN weights for one agent. Returns False on failure."""
        if agent.ddqn is None:
            return True
        try:
            agent.ddqn.save_weights(path)
            return True
        except Exception as e:
            LOG.error("[CHECKPOINT] Failed to save %s weights: %s", label, e)
            return False

    def save_checkpoint(self, checkpoint_dir: str = "data/checkpoints") -> bool:
        """Save full training state: DDQN weights, buffers, epsilon, training_steps.

        Called during graceful shutdown to preserve training progress.

        Args:
            checkpoint_dir: Directory to store checkpoint files

        Returns:
            True if all saves succeeded
        """
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        success = True

        # 1. Save DDQN weights
        success &= self._save_agent_weights(self.trigger, "trigger", f"{checkpoint_dir}/trigger_ddqn_weights.npz")
        success &= self._save_agent_weights(self.harvester, "harvester", f"{checkpoint_dir}/harvester_ddqn_weights.npz")

        # 2. Save experience buffers
        if self.trigger.buffer is not None and not self.trigger.buffer.save(f"{checkpoint_dir}/trigger_buffer"):
            success = False

        if self.harvester.buffer is not None and not self.harvester.buffer.save(f"{checkpoint_dir}/harvester_buffer"):
            success = False

        # 3. Save training metadata (epsilon, steps, etc.)
        metadata = {
            "trigger_training_steps": self.trigger.training_steps,
            "trigger_epsilon": self.trigger.epsilon,
            "harvester_training_steps": self.harvester.training_steps,
            "trigger_platt_a": getattr(self.trigger, "platt_a", 1.0),
            "trigger_platt_b": getattr(self.trigger, "platt_b", 0.0),
        }
        try:
            meta_path = Path(checkpoint_dir) / "training_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            LOG.info("[CHECKPOINT] Saved training metadata: %s", metadata)
        except Exception as e:
            LOG.error("[CHECKPOINT] Failed to save metadata: %s", e)
            success = False

        # 4. Save regime detector state (price buffer) so regime survives restarts
        if self.regime_detector:
            try:
                regime_state = {
                    "price_buffer": list(self.regime_detector.price_buffer),
                    "current_regime": self.current_regime,
                    "current_zeta": self.current_zeta,
                }
                regime_path = Path(checkpoint_dir) / "regime_state.json"
                with open(regime_path, "w") as f:
                    json.dump(regime_state, f)
                LOG.debug("[CHECKPOINT] Saved regime state: regime=%s, %d prices",
                          self.current_regime, len(self.regime_detector.price_buffer))
            except Exception as e:
                LOG.warning("[CHECKPOINT] Failed to save regime state: %s", e)

        if success:
            LOG.info("[CHECKPOINT] ✓ Full checkpoint saved to %s", checkpoint_dir)
        else:
            LOG.warning("[CHECKPOINT] Checkpoint saved with some failures to %s", checkpoint_dir)

        return success

    def load_checkpoint(self, checkpoint_dir: str = "data/checkpoints") -> bool:  # noqa: PLR0912, PLR0915
        """Load training state from a previous checkpoint.

        Called during startup to resume training from where it left off.

        Args:
            checkpoint_dir: Directory containing checkpoint files

        Returns:
            True if checkpoint was found and loaded (at least partially)
        """
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        cp = Path(checkpoint_dir)
        if not cp.exists():
            LOG.info("[CHECKPOINT] No checkpoint directory found at %s", checkpoint_dir)
            return False

        loaded_anything = False

        # 1. Load DDQN weights
        trigger_weights = cp / "trigger_ddqn_weights.npz"
        if trigger_weights.exists() and self.trigger.ddqn is not None:
            try:
                self.trigger.ddqn.load_weights(str(trigger_weights))
                loaded_anything = True
            except Exception as e:
                LOG.error("[CHECKPOINT] Failed to load trigger weights: %s", e)

        harvester_weights = cp / "harvester_ddqn_weights.npz"
        if harvester_weights.exists() and self.harvester.ddqn is not None:
            try:
                self.harvester.ddqn.load_weights(str(harvester_weights))
                loaded_anything = True
            except Exception as e:
                LOG.error("[CHECKPOINT] Failed to load harvester weights: %s", e)

        # 2. Load experience buffers
        trigger_buf = cp / "trigger_buffer.npz"
        if trigger_buf.exists() and self.trigger.buffer is not None and self.trigger.buffer.load(str(trigger_buf)):
            loaded_anything = True

        harvester_buf = cp / "harvester_buffer.npz"
        if harvester_buf.exists() and self.harvester.buffer is not None and self.harvester.buffer.load(str(harvester_buf)):
            loaded_anything = True

        # 3. Load training metadata
        meta_path = cp / "training_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)

                self.trigger.training_steps = metadata.get("trigger_training_steps", 0)
                self.trigger.epsilon = metadata.get("trigger_epsilon", self.trigger.epsilon)
                self.harvester.training_steps = metadata.get("harvester_training_steps", 0)

                # Sync DDQN network training_steps
                if self.trigger.ddqn is not None:
                    self.trigger.ddqn.training_steps = self.trigger.training_steps
                if self.harvester.ddqn is not None:
                    self.harvester.ddqn.training_steps = self.harvester.training_steps

                # Restore Platt calibration
                if hasattr(self.trigger, "platt_a"):
                    self.trigger.platt_a = metadata.get("trigger_platt_a", 1.0)
                    self.trigger.platt_b = metadata.get("trigger_platt_b", 0.0)

                LOG.info("[CHECKPOINT] Restored metadata: %s", metadata)
                loaded_anything = True
            except Exception as e:
                LOG.error("[CHECKPOINT] Failed to load metadata: %s", e)

        # 4. Restore regime detector price buffer so regime is immediately available
        regime_path = cp / "regime_state.json"
        if regime_path.exists() and self.regime_detector:
            try:
                with open(regime_path) as f:
                    regime_state = json.load(f)
                prices = regime_state.get("price_buffer", [])
                if prices:
                    self.regime_detector.price_buffer = list(prices)
                    # Force a regime recalculation from restored buffer
                    self.regime_detector._update_regime()
                    self.current_regime = self.regime_detector.current_regime
                    self.current_zeta = self.regime_detector.current_zeta
                    self._sync_replay_buffer_regime()
                    LOG.info("[CHECKPOINT] Restored regime state: regime=%s zeta=%.3f (%d prices)",
                             self.current_regime, self.current_zeta, len(prices))
                    loaded_anything = True
            except Exception as e:
                LOG.warning("[CHECKPOINT] Failed to restore regime state: %s", e)

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
            LOG.info("[CHECKPOINT] No checkpoint data found in %s", checkpoint_dir)

        return loaded_anything


# ============================================================================
# Self-Test
# ============================================================================
if __name__ == "__main__":
    import datetime as dt

    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("DualPolicy Self-Test")
    print("=" * 70)

    # Test 1: Initialize
    print("\n[TEST 1] Initialize DualPolicy")
    policy = DualPolicy(window=64)
    assert policy.current_position == 0
    assert policy.trigger is not None
    assert policy.harvester is not None
    print("✓ DualPolicy initialized with trigger + harvester")

    # Test 2: Entry decision (flat)
    print("\n[TEST 2] Entry decision (flat position)")
    bars = deque(maxlen=100)
    for i in range(100):
        t = dt.datetime.now()
        o = h = lo = c = 100000.0 + i * 10
        bars.append((t, o, h, lo, c))

    action, conf, runway = policy.decide_entry(bars, imbalance=0.1)
    assert action in [0, 1, 2]
    assert 0 <= conf <= 1
    assert runway >= 0
    print(f"✓ Entry decision: action={action}, conf={conf:.2f}, runway={runway:.4f}")

    # Test 3: Enter position
    print("\n[TEST 3] Enter LONG position")
    policy.on_entry(direction=1, entry_price=TEST_ENTRY_PRICE, entry_time=dt.datetime.now())
    assert policy.current_position == 1
    assert policy.entry_price == TEST_ENTRY_PRICE
    print(f"✓ Position entered: LONG @ {TEST_ENTRY_PRICE}")

    # Test 4: Exit decision (in position)
    print("\n[TEST 4] Exit decision (in position)")
    current_price = 100050.0  # Small profit
    action, conf = policy.decide_exit(bars, current_price, imbalance=0.1)
    assert action in [0, 1]  # HOLD or CLOSE
    assert 0 <= conf <= 1
    assert policy.mfe > 0  # Should have tracked MFE
    print(
        f"✓ Exit decision: action={action} ({'CLOSE' if action == 1 else 'HOLD'}), "
        f"conf={conf:.2f}, MFE={policy.mfe:.2f}"
    )

    # Test 5: Exit position
    print("\n[TEST 5] Exit position")
    policy.on_exit(exit_price=100050.0, capture_ratio=0.8, was_wtl=False)
    assert policy.current_position == 0
    assert SafeMath.is_zero(policy.mfe)
    print("✓ Position closed, state reset")

    print("\n" + "=" * 70)
    print("✓ All DualPolicy tests passed!")
    print("=" * 70)
