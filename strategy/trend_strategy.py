"""
Trend strategy — wires all pipeline components into a bar-by-bar signal generator.

Flow per bar:
  1. Update all indicators (smoothers, KER, VHF, Laguerre RSI, ATR, DTW)
  2. Regime gate — skip unless TRENDING
  3. Direction bias from Laguerre RSI + HTF KER alignment
  4. ATR-based SL/TP
  5. ML entry filter gate (LightGBM score >= threshold)
  6. Runway predictor gate (predicted floor >= floor_mult * friction)
  7. Emit Signal, record features for later labelling

Pyramiding:
  After initial entry, add positions while trend continues:
    - KER still above threshold
    - Price moved >= pyramid_atr_step ATR from last add
    - Pyramid level < max_pyramid_levels

Usage:
    strategy = TrendStrategy(config, spec, htf_bars_h4)
    on_bar = build_on_bar(strategy)
    result = engine.run(m30_bars, on_bar)

After backtest:
    labels = labeller.label_bars(...)
    strategy.train(feature_rows, labels)  # fits entry_filter + runway
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from core.math_utils import safe_div, safe_sqrt
from core.numeric import clamp, non_negative, is_valid_number
from core.validator import BrokerSpec
from core.logger import get_logger
from core.normaliser import ZScoreNormaliser

from pipeline.cleaner import Bar
from pipeline.features.trend import KER, VHF, MarketStructure
from pipeline.features.momentum import LaguerreRSI
from pipeline.features.volatility import ATR
from pipeline.features.regime import RegimeDetector, RegimeState
from pipeline.features.dtw_features import (
    TrendTemplateMatcher, make_impulse_template, make_breakout_template, make_pullback_template
)
from pipeline.features.orderflow import OrderBookImbalance, CumulativeDelta

from models.runway import RunwayPredictor
from models.entry_filter import EntryFilter
from backtesting.types import Signal

logger = get_logger("strategy")

# Feature vector slots — order is fixed, must match across entry filter + runway
FEATURE_NAMES = [
    "ker",          # Kaufman Efficiency Ratio at primary TF
    "vhf",          # Vertical Horizontal Filter
    "laguerre",     # Laguerre RSI normalised to [-1, 1]
    "atr_norm",     # ATR / close price (normalised volatility)
    "obi",          # Order Book Imbalance [-1, 1]
    "cum_delta",    # Cumulative delta direction [-1, 1]
    "htf_ker",      # Higher-TF KER (H4 or H1 context)
    "alignment",    # primary × HTF direction product [-1, 1]
    "dtw_sim",      # DTW template similarity [0, 1]
    "spread_ratio", # spread_pips * pip_size / ATR (friction weight)
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class StrategyConfig:
    """All tunable parameters — full search space exposed to Optuna."""
    # Regime gate
    ker_period: int = 14
    ker_trend_threshold: float = 0.45       # KER >= this to consider trending
    vhf_period: int = 28
    vhf_trend_threshold: float = 0.35
    regime_period: int = 40                  # HMM + combined detector window

    # Entry timing
    laguerre_gamma: float = 0.7             # Laguerre RSI smoothing (0=fast, 1=slow)
    laguerre_buy_threshold: float = 0.2     # RSI below this → look for long
    laguerre_sell_threshold: float = 0.8    # RSI above this → look for short

    # ATR-based risk
    atr_period: int = 14
    sl_atr_mult: float = 1.5               # SL = entry ± sl_atr_mult * ATR
    tp_atr_mult: float = 3.0               # TP = entry ± tp_atr_mult * ATR

    # Pyramid
    pyramid_atr_step: float = 1.0          # add position every N ATR from last add
    max_pyramid_levels: int = 2

    # Volume sizing
    lots_per_1000: float = 0.01

    # Gate thresholds
    entry_threshold: float = 0.55          # ML entry filter minimum score
    floor_multiplier: float = 2.0          # runway: floor / friction minimum

    # DTW template
    dtw_template: str = "impulse"          # "impulse", "breakout", "pullback"
    dtw_window: int = 5

    # HTF context — index into resampled bars dict key
    htf_key: str = "H4"

    # Warmup bars required before any signal
    warmup_bars: int = 50


class TrendStrategy:
    """
    Full trend-following strategy instance.
    One instance per symbol/primary-TF combination.
    """

    def __init__(
        self,
        config: StrategyConfig,
        spec: BrokerSpec,
        htf_bars: Optional[List[Bar]] = None,
    ) -> None:
        self._cfg = config
        self._spec = spec

        # --- Indicators ---
        self._ker = KER(period=config.ker_period)
        self._vhf = VHF(period=config.vhf_period)
        self._laguerre = LaguerreRSI(gamma=config.laguerre_gamma)
        self._atr = ATR(period=config.atr_period)
        self._regime = RegimeDetector(hmm_period=config.regime_period)
        self._structure = MarketStructure()

        self._dtw_matcher = self._make_dtw_matcher(config)

        # HTF KER (computed from pre-supplied HTF bars or updated inline)
        self._htf_ker = KER(period=config.ker_period)
        self._htf_bias = 0.0         # +1 uptrend, -1 downtrend, 0 neutral
        self._htf_ker_value = 0.0
        if htf_bars:
            self._preload_htf(htf_bars)

        # Live order flow (optional — injected by live session, not available in backtest)
        self._obi_calculator: Optional[OrderBookImbalance] = None
        self._cum_delta: Optional[CumulativeDelta] = None
        self._obi_value = 0.0
        self._cum_delta_value = 0.0

        # --- ML models (untrained until train() called) ---
        self._entry_filter = EntryFilter(
            model_type="lgbm",
            threshold=config.entry_threshold,
            symbol=spec.symbol,
        )
        self._runway = RunwayPredictor(
            n_features=N_FEATURES,
            floor_multiplier=config.floor_multiplier,
            symbol=spec.symbol,
        )

        # --- State ---
        self._bar_count = 0
        self._last_pyramid_price: Dict[int, float] = {}  # pyramid_level → price
        self._pending_features: List[List[float]] = []   # features at entry, pending outcome
        self._pending_mfe_pred: List[float] = []
        self._pending_friction: List[float] = []

        # Feature rows for offline training
        self._feature_log: List[Dict[str, Any]] = []

    def inject_orderflow(
        self,
        obi_calculator: OrderBookImbalance,
        cum_delta: CumulativeDelta,
    ) -> None:
        """Attach live order flow calculators (live session only)."""
        self._obi_calculator = obi_calculator
        self._cum_delta = cum_delta

    def on_bar(self, bar: Bar, bar_index: int, engine) -> Optional[Signal]:
        """
        Main strategy callback — called by BacktestEngine for each bar.
        Returns Signal to open a trade, or None.
        """
        self._bar_count += 1

        # --- Update all indicators ---
        self._ker.update(bar.close)
        self._vhf.update(bar.close)
        self._laguerre.update(bar.close)
        self._atr.update(bar.high, bar.low, bar.close)
        self._regime.update(bar.high, bar.low, bar.close)
        self._structure.update(bar.high, bar.low)
        self._dtw_matcher.update(bar.close)

        # Live order flow if available
        if self._obi_calculator is not None:
            self._obi_value = self._obi_calculator.value
        if self._cum_delta is not None:
            self._cum_delta_value = clamp(
                safe_div(self._cum_delta.cumulative, max(self._atr.value * 1000, 1.0)),
                -1.0, 1.0,
            )

        # Warmup gate — need enough bars for indicators to be meaningful
        if self._bar_count < self._cfg.warmup_bars:
            return None

        if not (self._ker.ready and self._laguerre.ready and self._atr.ready):
            return None

        ker = self._ker.value
        vhf = self._vhf.value
        laguerre = self._laguerre.value
        atr = self._atr.value
        regime = self._regime.state

        # --- Regime gate ---
        if regime not in (RegimeState.TRENDING, RegimeState.UNKNOWN):
            return None
        if ker < self._cfg.ker_trend_threshold:
            return None
        if vhf < self._cfg.vhf_trend_threshold:
            return None

        # --- Determine direction from Laguerre RSI + price structure ---
        direction = self._determine_direction(laguerre)
        if direction == 0:
            return None

        # --- Check short selling allowed ---
        if direction == -1 and not self._spec.short_selling_enabled:
            return None

        # --- HTF alignment filter ---
        if self._htf_ker_value > 0.3 and self._htf_bias != 0:
            if direction != self._htf_bias:
                return None   # trading against the HTF trend

        # --- Pyramid check: already in a trade at this level? ---
        pyramid_level = self._next_pyramid_level(engine, direction, bar.close, atr)
        if pyramid_level is None:
            return None

        # --- SL / TP ---
        sl, tp = self._compute_sl_tp(bar.close, direction, atr)
        if sl <= 0 or tp <= 0:
            return None

        # --- Build feature vector ---
        features = self._build_features(ker, vhf, laguerre, atr, bar.close)

        # --- ML gate ---
        if self._entry_filter.fitted:
            filter_result = self._entry_filter.predict_proba(features)
            if filter_result < self._cfg.entry_threshold:
                return None

        # --- Runway gate ---
        friction = self._spec.friction_cost(
            volume=self._cfg.lots_per_1000 * safe_div(engine.balance, 1000.0, fallback=0.01),
            price=bar.close,
        )
        runway = self._runway.evaluate(features, friction_cost=friction, atr=atr)
        if not runway.allow_entry:
            logger.signal(
                "blocked_runway", self._spec.symbol,
                floor_ratio=round(runway.floor_ratio, 3),
                reason=runway.reason,
            )
            return None

        # --- Volume sizing ---
        volume = self._spec.round_volume(
            non_negative(self._cfg.lots_per_1000 * safe_div(engine.balance, 1000.0, fallback=0.01))
        )
        if volume < self._spec.min_volume:
            return None

        # --- Log features for offline training ---
        self._feature_log.append({
            "timestamp": bar.timestamp,
            "direction": direction,
            "features": features,
            "ker": ker,
            "vhf": vhf,
            "laguerre": laguerre,
            "atr": atr,
            "close": bar.close,
            "sl": sl,
            "tp": tp,
        })

        # --- Stash entry context for runway calibration ---
        self._pending_features.append(features)
        self._pending_mfe_pred.append(runway.predicted_mfe)
        self._pending_friction.append(friction)

        # Update pyramid price tracking
        self._last_pyramid_price[pyramid_level] = bar.close

        logger.signal(
            self._spec.symbol, "M30", "long" if direction == 1 else "short",
            round(ker, 3),
            close=bar.close, sl=sl, tp=tp, pyramid_level=pyramid_level,
        )

        return Signal(
            direction=direction,
            stop_loss=sl,
            take_profit=tp,
            volume=volume,
            pyramid_level=pyramid_level,
        )

    def record_trade_outcome(self, actual_mfe: float) -> None:
        """
        Call after a trade closes to feed the runway predictor.
        Should be wired to engine.on_trade_close if available,
        or called in batch after walk-forward fold.
        """
        if not self._pending_features:
            return
        features = self._pending_features.pop(0)
        pred_mfe = self._pending_mfe_pred.pop(0)
        friction = self._pending_friction.pop(0)
        self._runway.record_outcome(features, pred_mfe, actual_mfe, friction)

    def train(
        self,
        feature_matrix: List[List[float]],
        labels: List[int],
    ) -> None:
        """
        Fit the ML entry filter from labelled feature rows.
        labels: 1 = good trade (hit TP), 0 = bad trade (hit SL or low MFE)
        """
        if len(feature_matrix) < 20 or len(labels) != len(feature_matrix):
            logger.warning("Not enough data to train entry filter", component="strategy")
            return
        self._entry_filter.fit(feature_matrix, labels, feature_names=FEATURE_NAMES)
        logger.info(
            "Entry filter trained",
            symbol=self._spec.symbol,
            n_samples=len(labels),
            component="strategy",
        )

    def get_feature_log(self) -> List[Dict[str, Any]]:
        return list(self._feature_log)

    def reset_for_new_fold(self) -> None:
        """Reset live state before starting a new walk-forward fold."""
        self._bar_count = 0
        self._last_pyramid_price = {}
        self._pending_features = []
        self._pending_mfe_pred = []
        self._pending_friction = []
        self._feature_log = []
        self._ker = KER(period=self._cfg.ker_period)
        self._vhf = VHF(period=self._cfg.vhf_period)
        self._laguerre = LaguerreRSI(gamma=self._cfg.laguerre_gamma)
        self._atr = ATR(period=self._cfg.atr_period)
        self._regime = RegimeDetector(period=self._cfg.regime_period)
        self._structure = MarketStructure()
        self._dtw_matcher = self._make_dtw_matcher(self._cfg)
        self._obi_value = 0.0
        self._cum_delta_value = 0.0

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _determine_direction(self, laguerre: float) -> int:
        """
        +1 long, -1 short, 0 no signal.
        Laguerre RSI oversold → look for longs; overbought → look for shorts.
        Confirmed by MarketStructure HH/HL (long) or LH/LL (short).
        """
        structure = self._structure.trend  # 1=uptrend, -1=downtrend, 0=neutral

        if laguerre <= self._cfg.laguerre_buy_threshold and structure >= 0:
            return 1
        if laguerre >= self._cfg.laguerre_sell_threshold and structure <= 0:
            return -1
        return 0

    def _compute_sl_tp(
        self, price: float, direction: int, atr: float
    ) -> Tuple[float, float]:
        if not is_valid_number(atr) or atr <= 0 or price <= 0:
            return 0.0, 0.0

        sl_dist = self._cfg.sl_atr_mult * atr
        tp_dist = self._cfg.tp_atr_mult * atr

        # Enforce minimum distance from spec
        min_dist = max(
            self._spec.sl_min_distance * self._spec.pip_size,
            self._spec.tick_size * 2,
        )
        sl_dist = max(sl_dist, min_dist)
        tp_dist = max(tp_dist, min_dist)

        if direction == 1:
            sl = self._spec.round_price(price - sl_dist)
            tp = self._spec.round_price(price + tp_dist)
        else:
            sl = self._spec.round_price(price + sl_dist)
            tp = self._spec.round_price(price - tp_dist)

        return sl, tp

    def _next_pyramid_level(
        self, engine, direction: int, price: float, atr: float
    ) -> Optional[int]:
        """
        Returns the pyramid level for the next entry, or None if cannot pyramid.
        Level 1 = initial entry.  Levels 2+ = add-ons.
        """
        open_trades = engine.open_trades
        if not open_trades:
            # No open trades — fresh entry at level 1
            self._last_pyramid_price = {}
            return 1

        # Filter trades matching our symbol + direction
        our_trades = [
            t for t in open_trades
            if t.symbol == self._spec.symbol and int(t.direction) == direction
        ]
        if not our_trades:
            return 1   # Different direction open — still allow fresh entry

        current_level = max(t.pyramid_level for t in our_trades)
        if current_level >= self._cfg.max_pyramid_levels:
            return None

        # Check price has moved far enough from last pyramid add
        last_price = self._last_pyramid_price.get(current_level, 0.0)
        if last_price > 0 and atr > 0:
            distance = abs(price - last_price)
            if distance < self._cfg.pyramid_atr_step * atr:
                return None

        return current_level + 1

    def _build_features(
        self,
        ker: float,
        vhf: float,
        laguerre: float,
        atr: float,
        price: float,
    ) -> List[float]:
        """Build the fixed-length feature vector matching FEATURE_NAMES."""
        laguerre_norm = clamp(laguerre * 2.0 - 1.0, -1.0, 1.0)  # [0,1] → [-1,1]
        atr_norm = clamp(safe_div(atr, price, fallback=0.0) * 100.0, 0.0, 5.0)
        spread_ratio = clamp(
            safe_div(self._spec.spread_pips * self._spec.pip_size, max(atr, 1e-10)),
            0.0, 1.0,
        )

        # HTF alignment: product of direction signals [-1, 1]
        primary_dir = 1.0 if laguerre_norm < 0 else -1.0  # oversold=long, overbought=short
        alignment = clamp(primary_dir * self._htf_bias, -1.0, 1.0)

        dtw_sim = self._dtw_matcher.best_match[1] if self._dtw_matcher.ready else 0.5

        return [
            clamp(ker, 0.0, 1.0),
            clamp(vhf, 0.0, 1.0),
            laguerre_norm,
            atr_norm / 5.0,          # normalise to [0, 1]
            clamp(self._obi_value, -1.0, 1.0),
            clamp(self._cum_delta_value, -1.0, 1.0),
            clamp(self._htf_ker_value, 0.0, 1.0),
            alignment,
            clamp(dtw_sim, 0.0, 1.0),
            spread_ratio,
        ]

    @staticmethod
    def _make_dtw_matcher(config: StrategyConfig) -> TrendTemplateMatcher:
        matcher = TrendTemplateMatcher(window=config.dtw_window)
        length = config.dtw_window
        if config.dtw_template == "impulse":
            matcher.add_template("impulse", make_impulse_template(length))
        elif config.dtw_template == "breakout":
            matcher.add_template("breakout", make_breakout_template(length))
        elif config.dtw_template == "pullback":
            matcher.add_template("pullback", make_pullback_template(length))
        else:
            matcher.add_template("impulse", make_impulse_template(length))
        return matcher

    def _preload_htf(self, htf_bars: List[Bar]) -> None:
        """Pre-compute HTF KER bias from historical bars."""
        htf_ker = KER(period=self._cfg.ker_period)
        for bar in htf_bars:
            htf_ker.update(bar.close)
        self._htf_ker_value = htf_ker.value
        # Estimate bias: last N closes trending up or down?
        n = min(self._cfg.ker_period, len(htf_bars))
        if n >= 2:
            slope = htf_bars[-1].close - htf_bars[-n].close
            self._htf_bias = 1.0 if slope > 0 else (-1.0 if slope < 0 else 0.0)

    def update_htf_bar(self, bar: Bar) -> None:
        """
        Call with each new H4 bar during live trading or multi-TF backtest
        to keep the HTF context current.
        """
        self._htf_ker.update(bar.close)
        self._htf_ker_value = self._htf_ker.value
        # Rolling slope on last ker_period bars
        self._htf_bias = 1.0 if bar.close > bar.open else -1.0


def build_on_bar(strategy: TrendStrategy):
    """
    Returns a closure suitable for BacktestEngine.run(bars, on_bar).
    Captures strategy state cleanly — safe for multiple backtest runs
    if strategy.reset_for_new_fold() is called between runs.
    """
    def on_bar(bar: Bar, bar_index: int, engine) -> Optional[Signal]:
        return strategy.on_bar(bar, bar_index, engine)
    return on_bar
