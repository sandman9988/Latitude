"""
Runway Predictor — self-calibrating MFE gate.

Entry is only allowed when:
    predicted_floor  >= floor_multiplier  * friction_cost
    predicted_mfe    >= min_mfe_threshold

Self-calibrating: recalibrates on every `recalibrate_every` completed trades
using recent actual MFE outcomes vs predictions. Adjusts conservatism factor
automatically based on prediction error.

Architecture:
  - Uses a quantile regression model (ridge-regularised) to predict
    the lower bound (floor) and expected MFE from feature vectors.
  - No external ML dependencies — pure numpy-style math.
  - Falls back to ATR-based heuristic until enough trades are observed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from core.math_utils import safe_div, safe_sqrt
from core.numeric import non_negative, clamp, is_valid_number
from core.logger import get_logger

logger = get_logger("runway")

_MIN_TRAINING_SAMPLES = 30
_LEARNING_RATE = 0.01
_L2_LAMBDA = 0.01


@dataclass
class RunwayDecision:
    allow_entry: bool
    predicted_mfe: float
    predicted_floor: float
    friction_cost: float
    floor_ratio: float      # predicted_floor / friction_cost
    confidence: float       # 0.0-1.0, based on sample count
    reason: str             # human-readable gate reason


@dataclass
class TradeOutcome:
    """Observed outcome for a completed trade — feeds calibration."""
    features: List[float]
    predicted_mfe: float
    actual_mfe: float
    friction_cost: float


class RunwayPredictor:
    """
    Self-calibrating runway predictor.

    Parameters:
        floor_multiplier: minimum ratio of predicted_floor / friction_cost to allow entry
        min_mfe_threshold: minimum predicted MFE in absolute price units
        recalibrate_every: recalibrate after this many completed trades
        atr_fallback_mult: MFE estimate multiplier on ATR when not yet calibrated
    """

    def __init__(
        self,
        n_features: int = 10,
        floor_multiplier: float = 2.0,
        min_mfe_threshold: float = 0.0,
        recalibrate_every: int = 20,
        atr_fallback_mult: float = 2.0,
        symbol: str = "",
        tf: str = "",
    ) -> None:
        self._n_features = n_features
        self._floor_mult = floor_multiplier
        self._min_mfe = min_mfe_threshold
        self._recal_every = recalibrate_every
        self._atr_fallback = atr_fallback_mult
        self._symbol = symbol
        self._tf = tf

        # Linear model weights — two heads: mfe and floor
        self._w_mfe = [0.0] * n_features
        self._b_mfe = 0.0
        self._w_floor = [0.0] * n_features
        self._b_floor = 0.0

        self._outcomes: List[TradeOutcome] = []
        self._trade_count = 0
        self._calibrated = False
        self._conservatism = 1.0    # scales down predictions if over-optimistic
        self._prediction_errors: List[float] = []

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def sample_count(self) -> int:
        return len(self._outcomes)

    def evaluate(
        self,
        features: List[float],
        friction_cost: float,
        atr: float = 0.0,
    ) -> RunwayDecision:
        """
        Evaluate whether to allow an entry.
        features: normalised feature vector (length must match n_features)
        friction_cost: total round-trip cost in price units
        atr: current ATR, used as fallback before calibration
        """
        if len(features) != self._n_features:
            features = _pad_or_trim(features, self._n_features)

        confidence = clamp(safe_div(len(self._outcomes), _MIN_TRAINING_SAMPLES * 2), 0.0, 1.0)

        if not self._calibrated:
            # ATR-based heuristic fallback
            pred_mfe = non_negative(atr * self._atr_fallback) if atr > 0 else friction_cost * 3.0
            pred_floor = pred_mfe * 0.3
            return self._decide(pred_mfe, pred_floor, friction_cost, confidence=0.0,
                                reason="heuristic_atr_fallback")

        pred_mfe = non_negative(self._predict(features, self._w_mfe, self._b_mfe) * self._conservatism)
        pred_floor = non_negative(self._predict(features, self._w_floor, self._b_floor) * self._conservatism)

        return self._decide(pred_mfe, pred_floor, friction_cost, confidence, reason="model")

    def _decide(
        self,
        pred_mfe: float,
        pred_floor: float,
        friction_cost: float,
        confidence: float,
        reason: str,
    ) -> RunwayDecision:
        floor_ratio = safe_div(pred_floor, friction_cost, fallback=0.0) if friction_cost > 0 else 0.0
        allow = (
            floor_ratio >= self._floor_mult
            and pred_mfe >= self._min_mfe
        )
        return RunwayDecision(
            allow_entry=allow,
            predicted_mfe=pred_mfe,
            predicted_floor=pred_floor,
            friction_cost=friction_cost,
            floor_ratio=floor_ratio,
            confidence=confidence,
            reason=reason,
        )

    def record_outcome(self, features: List[float], predicted_mfe: float, actual_mfe: float, friction_cost: float) -> None:
        """Record a completed trade outcome for calibration."""
        if len(features) != self._n_features:
            features = _pad_or_trim(features, self._n_features)

        self._outcomes.append(TradeOutcome(
            features=features,
            predicted_mfe=predicted_mfe,
            actual_mfe=actual_mfe,
            friction_cost=friction_cost,
        ))
        self._trade_count += 1

        # Track prediction error for conservatism adjustment
        if predicted_mfe > 0:
            error = safe_div(actual_mfe, predicted_mfe, fallback=1.0)
            self._prediction_errors.append(error)
            if len(self._prediction_errors) > 50:
                self._prediction_errors.pop(0)

        if self._trade_count % self._recal_every == 0 or (
            not self._calibrated and len(self._outcomes) >= _MIN_TRAINING_SAMPLES
        ):
            self._recalibrate()

    def _recalibrate(self) -> None:
        """
        Retrain linear model on accumulated outcomes.
        Uses SGD with L2 regularisation. Adjusts conservatism factor.
        """
        outcomes = self._outcomes[-200:]  # cap at 200 most recent
        n = len(outcomes)
        if n < _MIN_TRAINING_SAMPLES:
            return

        # SGD training — multiple passes
        for _ in range(10):
            for outcome in outcomes:
                x = outcome.features
                y_mfe = outcome.actual_mfe
                y_floor = outcome.actual_mfe * 0.25  # floor = 25th percentile proxy

                # MFE head
                pred_mfe = self._predict(x, self._w_mfe, self._b_mfe)
                err_mfe = pred_mfe - y_mfe
                self._w_mfe = [
                    w - _LEARNING_RATE * (err_mfe * x[i] + _L2_LAMBDA * w)
                    for i, w in enumerate(self._w_mfe)
                ]
                self._b_mfe -= _LEARNING_RATE * err_mfe

                # Floor head
                pred_floor = self._predict(x, self._w_floor, self._b_floor)
                err_floor = pred_floor - y_floor
                self._w_floor = [
                    w - _LEARNING_RATE * (err_floor * x[i] + _L2_LAMBDA * w)
                    for i, w in enumerate(self._w_floor)
                ]
                self._b_floor -= _LEARNING_RATE * err_floor

        # Adjust conservatism: if predictions are systematically over-optimistic, pull back
        if self._prediction_errors:
            mean_error_ratio = sum(self._prediction_errors) / len(self._prediction_errors)
            # Blend toward actual ratio
            self._conservatism = clamp(
                0.8 * self._conservatism + 0.2 * mean_error_ratio,
                0.3, 1.5
            )

        self._calibrated = True
        logger.info(
            "Runway predictor recalibrated",
            symbol=self._symbol, tf=self._tf,
            component="runway",
            samples=n,
            conservatism=round(self._conservatism, 3),
        )

    def _predict(self, x: List[float], w: List[float], b: float) -> float:
        raw = sum(xi * wi for xi, wi in zip(x, w)) + b
        return non_negative(raw)

    def reset(self) -> None:
        self.__init__(
            n_features=self._n_features,
            floor_multiplier=self._floor_mult,
            min_mfe_threshold=self._min_mfe,
            recalibrate_every=self._recal_every,
            atr_fallback_mult=self._atr_fallback,
            symbol=self._symbol,
            tf=self._tf,
        )


def _pad_or_trim(features: List[float], n: int) -> List[float]:
    if len(features) >= n:
        return features[:n]
    return features + [0.0] * (n - len(features))
