#!/usr/bin/env python3
"""Runway labels.
================
Pure-numpy construction of *forward favorable-excursion* labels used to train
the redesigned runway forecaster.

The legacy runway predictor re-mapped the entry DDQN Q-value into a maximum
favorable excursion (MFE) estimate.  That coupling produced no usable signal
(overall corr(predicted, realized) ~= 0.19 on live trade history, negative for
several high-volume bots).  This module instead derives a *direct, unbiased*
supervised target computed on every bar of a continuous OHLC series:

    long  MFE[i] = max(high[i+1 .. i+H]) - close[i]
    short MFE[i] = close[i] - min(low[i+1 .. i+H])

where H is the per-timeframe forward horizon (in bars).  Raw price MFE is
non-stationary across regimes and symbols, so labels are normalized by ATR:

    label_long[i]  = long_mfe[i]  / atr[i]
    label_short[i] = short_mfe[i] / atr[i]

ATR-normalized excursion is comparable across symbols and volatility regimes,
which is what makes a single forecaster trainable per (symbol, timeframe).

Identity throughout the project is (symbol, timeframe_minutes); this module is
keyed on timeframe_minutes for horizon selection only and is symbol-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datetime import datetime

    Bar = tuple[datetime, float, float, float, float, float]

# Forward horizon (number of bars to look ahead for favorable excursion),
# keyed by timeframe_minutes for the canonical TFs M1/M5/M15/M30/M60/M240.
DEFAULT_HORIZON_BARS: dict[int, int] = {
    1: 60,
    5: 36,
    15: 24,
    30: 16,
    60: 12,
    240: 6,
}

DEFAULT_ATR_PERIOD = 14
_FALLBACK_HORIZON = 24
_EPS = 1e-12


def horizon_for_timeframe(timeframe_minutes: int) -> int:
    """Return the forward horizon (in bars) for a given timeframe."""
    return DEFAULT_HORIZON_BARS.get(int(timeframe_minutes), _FALLBACK_HORIZON)


def bars_to_arrays(
    bars: list[Bar],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a list of (t, o, h, l, c, spread) bars into o/h/l/c arrays."""
    if not bars:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    arr = np.asarray([(b[1], b[2], b[3], b[4]) for b in bars], dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def wilder_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = DEFAULT_ATR_PERIOD,
) -> np.ndarray:
    """Wilder's ATR aligned per bar (atr[i] uses information up to bar i).

    True range = max(H-L, |H-C_prev|, |L-C_prev|).  The first bar has no prior
    close, so its true range is simply H-L.  The returned array is the same
    length as the inputs; early bars before the period is filled use the
    running mean of available true ranges.
    """
    n = len(closes)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    tr = np.empty(n, dtype=np.float64)
    tr[0] = abs(highs[0] - lows[0])
    if n > 1:
        hl = highs[1:] - lows[1:]
        hc = np.abs(highs[1:] - closes[:-1])
        lc = np.abs(lows[1:] - closes[:-1])
        tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    atr = np.empty(n, dtype=np.float64)
    period = max(1, int(period))
    if n <= period:
        running = np.cumsum(tr) / np.arange(1, n + 1)
        return running

    atr[:period] = np.cumsum(tr[:period]) / np.arange(1, period + 1)
    prev = float(np.mean(tr[:period]))
    atr[period - 1] = prev
    for i in range(period, n):
        prev = (prev * (period - 1) + tr[i]) / period
        atr[i] = prev
    return atr


def forward_favorable_excursion(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    horizon: int,
    side: str = "long",
) -> np.ndarray:
    """Raw favorable excursion over the next *horizon* bars for each bar.

    For ``side='long'`` this is ``max(high[i+1..i+H]) - close[i]`` (clipped at
    zero); for ``side='short'`` it is ``close[i] - min(low[i+1..i+H])``.  Bars
    whose full forward window is not available are set to NaN so callers can
    mask them out.
    """
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or horizon < 1:
        return out

    for i in range(n):
        end = i + horizon
        if end >= n:
            break
        window_hi = highs[i + 1 : end + 1]
        window_lo = lows[i + 1 : end + 1]
        if side == "long":
            out[i] = max(0.0, float(np.max(window_hi)) - closes[i])
        else:
            out[i] = max(0.0, closes[i] - float(np.min(window_lo)))
    return out


def build_labels(
    bars: list[Bar],
    timeframe_minutes: int,
    horizon: int | None = None,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> dict[str, np.ndarray]:
    """Build ATR-normalized forward favorable-excursion labels for a bar series.

    Returns a dict of equal-length arrays:
        ``close``       close price per bar
        ``atr``         Wilder ATR per bar
        ``mfe_long``    raw long favorable excursion (price units)
        ``mfe_short``   raw short favorable excursion (price units)
        ``label_long``  mfe_long / atr  (ATR-normalized, the training target)
        ``label_short`` mfe_short / atr
        ``valid``       boolean mask: forward window complete and atr > 0
    """
    _, h, l, c = bars_to_arrays(bars)
    if horizon is None:
        horizon = horizon_for_timeframe(timeframe_minutes)

    atr = wilder_atr(h, l, c, period=atr_period)
    mfe_long = forward_favorable_excursion(h, l, c, horizon, side="long")
    mfe_short = forward_favorable_excursion(h, l, c, horizon, side="short")

    safe_atr = np.where(atr > _EPS, atr, np.nan)
    label_long = mfe_long / safe_atr
    label_short = mfe_short / safe_atr

    valid = np.isfinite(label_long) & np.isfinite(label_short) & (atr > _EPS)

    return {
        "close": c,
        "atr": atr,
        "mfe_long": mfe_long,
        "mfe_short": mfe_short,
        "label_long": label_long,
        "label_short": label_short,
        "valid": valid,
    }
