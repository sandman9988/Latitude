#!/usr/bin/env python3
"""Runway forecaster.
==================
Distributional forecaster for ATR-normalized forward favorable excursion,
fully decoupled from the entry DDQN Q-value.

Design goals
------------
* **Decoupled** — trained on direct forward-excursion labels (see
  ``src.features.runway_labels``), never on the entry agent's Q-value, breaking
  the circular dependency that made the legacy runway useless.
* **Distributional** — predicts quantiles (p10/p50/p90) via pinball loss so the
  entry gate can reason probabilistically instead of comparing noisy point
  estimates.
* **Cheap hot-loop inference** — pure numpy linear quantile heads on a small
  hand-crafted feature vector; a single ``predict`` is a couple of matmuls.
* **ATR-relative** — predicts excursion in ATR units; callers multiply by the
  current ATR to recover price-unit runway, which keeps the model stationary
  across symbols, timeframes and volatility regimes.

One forecaster is trained per (symbol, timeframe_minutes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "side",
    "atr_ratio",
    "ret_1",
    "ret_k",
    "vol_ratio",
    "range_pos",
    "body_ratio",
    "trend_slope",
)
_LOOKBACK = 20
_SHORT_VOL = 5
_EPS = 1e-12


def extract_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    idx: int,
    side: int = 1,
    lookback: int = _LOOKBACK,
) -> np.ndarray | None:
    """Build the feature vector observable at the close of bar ``idx``.

    ``side`` is +1 for a long entry and -1 for a short entry.  Uses only
    information up to and including bar ``idx`` (no lookahead).  Returns
    ``None`` when there is insufficient history.
    """
    if idx < lookback:
        return None
    c = closes[idx]
    if c <= _EPS:
        return None

    win_lo = lows[idx - lookback + 1 : idx + 1]
    win_hi = highs[idx - lookback + 1 : idx + 1]
    rng = float(np.max(win_hi) - np.min(win_lo))

    rets = np.diff(closes[idx - lookback : idx + 1]) / np.maximum(
        closes[idx - lookback : idx], _EPS
    )
    ret_1 = float(rets[-1]) if rets.size else 0.0
    ret_k = float(np.sum(rets))
    short_std = float(np.std(rets[-_SHORT_VOL:])) if rets.size >= _SHORT_VOL else 0.0
    long_std = float(np.std(rets)) if rets.size else 0.0
    vol_ratio = short_std / (long_std + _EPS)

    range_pos = (
        float((closes[idx] - np.min(win_lo)) / rng) if rng > _EPS else 0.5
    )
    bar_rng = float(highs[idx] - lows[idx])
    body_ratio = (
        float((closes[idx] - opens[idx]) / bar_rng) if bar_rng > _EPS else 0.0
    )

    xs = np.arange(lookback, dtype=np.float64)
    win_c = closes[idx - lookback + 1 : idx + 1]
    slope = float(np.polyfit(xs, win_c, 1)[0]) if win_c.size == lookback else 0.0
    trend_slope = slope / (c + _EPS)

    atr_ratio = float(atr[idx] / c) if atr[idx] > _EPS else 0.0

    return np.array(
        [
            1.0,
            float(1 if side >= 0 else -1),
            atr_ratio,
            ret_1,
            ret_k,
            vol_ratio,
            range_pos,
            body_ratio,
            trend_slope,
        ],
        dtype=np.float64,
    )


def build_feature_matrix(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    side: int = 1,
    lookback: int = _LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix for every bar with sufficient history, for one side.

    Returns ``(features, indices)`` where ``features`` has shape
    ``(m, n_features)`` and ``indices`` maps each row back to its bar index.
    """
    n = len(closes)
    feats: list[np.ndarray] = []
    idxs: list[int] = []
    for i in range(lookback, n):
        f = extract_features(opens, highs, lows, closes, atr, i, side, lookback)
        if f is not None:
            feats.append(f)
            idxs.append(i)
    if not feats:
        return np.empty((0, len(_FEATURE_NAMES))), np.empty(0, dtype=np.int64)
    return np.asarray(feats), np.asarray(idxs, dtype=np.int64)


class RunwayForecaster:
    """ATR-anchored distributional forecaster of forward favorable excursion.

    The dominant, regime-stable driver of favorable excursion is current
    volatility: on live history corr(ATR, realized MFE) ~= 0.55, versus 0.19
    for the legacy Q-derived predictor.  This model therefore anchors on ATR:

        runway_price[q] = multiple[side][q] * ATR

    where ``multiple[side][q]`` is the empirical q-quantile of the
    ATR-normalized forward excursion in the training window, estimated per side
    (+1 long / -1 short).  An optional linear *residual* conditioned on
    market-state features is layered on top, but only retained when it improves
    pinball loss on an internal time-ordered holdout, so noisy/regime-unstable
    conditioning can never degrade the robust ATR anchor.
    """

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        lookback: int = _LOOKBACK,
    ) -> None:
        self.quantiles = tuple(quantiles)
        self.lookback = lookback
        self.n_features = len(_FEATURE_NAMES)
        self.weights = np.zeros((len(self.quantiles), self.n_features))
        self.feat_mean = np.zeros(self.n_features)
        self.feat_std = np.ones(self.n_features)
        self.base_q = {1: np.zeros(len(self.quantiles)), -1: np.zeros(len(self.quantiles))}
        self.use_residual = False
        self.fitted = False

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.feat_mean) / self.feat_std
        z[..., 0] = 1.0
        z[..., 1] = x[..., 1]
        return z

    def _side_base(self, side: float) -> np.ndarray:
        return self.base_q[1] if side >= 0 else self.base_q[-1]

    @staticmethod
    def _pinball(err: np.ndarray, q: float) -> float:
        return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))

    def _fit_residual(
        self,
        x: np.ndarray,
        resid: np.ndarray,
        lr: float,
        epochs: int,
        l2: float,
        seed: int,
    ) -> np.ndarray:
        m = x.shape[0]
        rng = np.random.default_rng(seed)
        weights = rng.normal(0.0, 0.01, size=(len(self.quantiles), self.n_features))
        for qi, q in enumerate(self.quantiles):
            w = weights[qi]
            for _ in range(epochs):
                err = resid - x @ w
                grad_coef = np.where(err >= 0, -q, (1.0 - q))
                grad = (x.T @ grad_coef) / m + l2 * w
                w = w - lr * grad
            weights[qi] = w
        return weights

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.05,
        epochs: int = 300,
        l2: float = 1e-3,
        seed: int = 0,
    ) -> dict[str, float]:
        """Fit the ATR anchor and (if it validates) a feature residual."""
        if features.shape[0] == 0:
            msg = "cannot fit RunwayForecaster on empty feature matrix"
            raise ValueError(msg)

        y = labels.astype(np.float64)
        sides = features[:, 1]
        for s in (1, -1):
            mask = sides >= 0 if s == 1 else sides < 0
            ys = y[mask] if np.any(mask) else y
            self.base_q[s] = np.quantile(ys, self.quantiles)

        self.feat_mean = features.mean(axis=0)
        self.feat_std = features.std(axis=0)
        self.feat_std[self.feat_std < _EPS] = 1.0

        base_med = np.array(
            [self._side_base(sd)[self._p50_idx()] for sd in sides]
        )
        resid = y - base_med

        n = features.shape[0]
        cut = int(n * 0.75)
        x_all = self._standardize(features)
        self.weights = np.zeros((len(self.quantiles), self.n_features))
        self.use_residual = False
        if cut > 50 and n - cut > 50:
            w_try = self._fit_residual(
                x_all[:cut], resid[:cut], lr, epochs, l2, seed
            )
            base_hold = np.stack([self._side_base(sd) for sd in sides[cut:]])
            anchor_pin = self._total_pinball(base_hold, y[cut:])
            res_pred = base_hold + (x_all[cut:] @ w_try.T)
            res_pin = self._total_pinball(res_pred, y[cut:])
            if res_pin < anchor_pin:
                self.weights = self._fit_residual(x_all, resid, lr, epochs, l2, seed)
                self.use_residual = True

        self.fitted = True
        preds = self.predict_matrix(features)
        losses = {}
        for qi, q in enumerate(self.quantiles):
            losses[f"pinball_q{int(q * 100)}"] = self._pinball(y - preds[:, qi], q)
        losses["use_residual"] = float(self.use_residual)
        return losses

    def _p50_idx(self) -> int:
        return int(np.argmin(np.abs(np.asarray(self.quantiles) - 0.5)))

    def _total_pinball(self, pred_q: np.ndarray, y: np.ndarray) -> float:
        total = 0.0
        for qi, q in enumerate(self.quantiles):
            total += self._pinball(y - pred_q[:, qi], q)
        return total

    def predict_quantiles(self, feature_vec: np.ndarray) -> np.ndarray:
        """Predict ATR-normalized runway quantiles for one feature vector."""
        base = self._side_base(feature_vec[1]).astype(np.float64)
        if self.use_residual:
            z = self._standardize(feature_vec.reshape(1, -1))
            base = base + (z @ self.weights.T).ravel()
        return np.maximum(0.0, np.sort(base))

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        """Predict quantiles for a feature matrix -> shape (m, n_quantiles)."""
        base = np.stack([self._side_base(sd) for sd in features[:, 1]])
        if self.use_residual:
            z = self._standardize(features)
            base = base + (z @ self.weights.T)
        return np.sort(np.maximum(0.0, base), axis=1)

    def predict_runway(
        self,
        feature_vec: np.ndarray,
        atr: float,
        quantile: float = 0.5,
    ) -> float:
        """Predict runway in price units = atr * normalized-quantile."""
        qs = self.predict_quantiles(feature_vec)
        idx = int(np.argmin(np.abs(np.asarray(self.quantiles) - quantile)))
        return float(max(0.0, qs[idx]) * max(0.0, atr))

    def prob_exceed(self, feature_vec: np.ndarray, atr: float, threshold: float) -> float:
        """Approximate P(runway > threshold) from the predicted quantiles.

        Linear interpolation across the predicted quantile points; values beyond
        the outer quantiles saturate at 0/1.
        """
        if atr <= _EPS:
            return 0.0
        norm_thr = threshold / atr
        qs = self.predict_quantiles(feature_vec)
        probs = np.asarray(self.quantiles)
        if norm_thr <= qs[0]:
            return 1.0
        if norm_thr >= qs[-1]:
            return float(1.0 - probs[-1])
        cdf = np.interp(norm_thr, qs, probs)
        return float(max(0.0, min(1.0, 1.0 - cdf)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "quantiles": list(self.quantiles),
            "lookback": self.lookback,
            "n_features": self.n_features,
            "weights": self.weights.tolist(),
            "feat_mean": self.feat_mean.tolist(),
            "feat_std": self.feat_std.tolist(),
            "base_q_long": self.base_q[1].tolist(),
            "base_q_short": self.base_q[-1].tolist(),
            "use_residual": self.use_residual,
            "fitted": self.fitted,
            "feature_names": list(_FEATURE_NAMES),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str | Path) -> RunwayForecaster:
        with open(Path(path), encoding="utf-8") as fh:
            payload = json.load(fh)
        obj = cls(
            quantiles=tuple(payload["quantiles"]),
            lookback=int(payload["lookback"]),
        )
        obj.weights = np.asarray(payload["weights"], dtype=np.float64)
        obj.feat_mean = np.asarray(payload["feat_mean"], dtype=np.float64)
        obj.feat_std = np.asarray(payload["feat_std"], dtype=np.float64)
        obj.base_q = {
            1: np.asarray(payload.get("base_q_long", np.zeros(len(obj.quantiles))), dtype=np.float64),
            -1: np.asarray(payload.get("base_q_short", np.zeros(len(obj.quantiles))), dtype=np.float64),
        }
        obj.n_features = int(payload.get("n_features", obj.n_features))
        obj.use_residual = bool(payload.get("use_residual", False))
        obj.fitted = bool(payload["fitted"])
        return obj
