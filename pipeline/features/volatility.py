"""
Volatility estimators.
Yang-Zhang for HTF/daily, Rogers-Satchell for intraday.
ATR smoothed via injectable smoother (default JMA).
"""
from __future__ import annotations

import math
from core.math_utils import safe_div, safe_log, safe_sqrt
from core.numeric import non_negative, is_valid_number, clamp
from core.memory import FloatCircularBuffer
from .smoothing import make_smoother, Smoother


# ---------------------------------------------------------------------------
# ATR — Average True Range (JMA-smoothed by default)
# ---------------------------------------------------------------------------
class ATR:
    """
    Average True Range with injectable smoother.
    smoother: 'jma' (default), 'kama', 'zlema', 't3', 'alma'
    """

    def __init__(self, period: int = 14, smoother: str = "jma") -> None:
        self._period = max(2, period)
        self._smoother_name = smoother
        self._smoother: Smoother = make_smoother(smoother, period=period)
        self._prev_close = 0.0
        self._value = 0.0
        self._count = 0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._smoother.ready

    def update(self, high: float, low: float, close: float) -> float:
        if not all(is_valid_number(v) for v in [high, low, close]):
            return self._value

        if self._count == 0:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )

        self._prev_close = close
        self._count += 1
        self._value = non_negative(self._smoother.update(tr))
        return self._value

    def reset(self) -> None:
        self.__init__(self._period, self._smoother_name)


# ---------------------------------------------------------------------------
# Rogers-Satchell Volatility
# Better than Parkinson for trending markets — no drift assumption.
# Uses O/H/L/C. Best for intraday (M30, H1, H2, H4).
# ---------------------------------------------------------------------------
class RogersSatchell:
    """
    Rogers-Satchell realized volatility estimator.
    period: rolling window (default 20)
    Returns annualised volatility (fraction, e.g. 0.15 = 15%).
    bars_per_year: used for annualisation (252 daily, 1460 for H4, etc.)
    """

    def __init__(self, period: int = 20, bars_per_year: float = 1460.0) -> None:
        self._period = max(2, period)
        self._bpy = bars_per_year
        self._buf = FloatCircularBuffer(self._period)
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._buf.full

    def update(self, open_: float, high: float, low: float, close: float) -> float:
        if not all(is_valid_number(v) and v > 0 for v in [open_, high, low, close]):
            return self._value

        rs = (
            safe_log(safe_div(high, close)) * safe_log(safe_div(high, open_))
            + safe_log(safe_div(low, close)) * safe_log(safe_div(low, open_))
        )
        self._buf.push(non_negative(rs))

        if not self._buf.full:
            return self._value

        mean_rs = self._buf.mean()
        self._value = safe_sqrt(mean_rs * self._bpy)
        return self._value

    def reset(self) -> None:
        self.__init__(self._period, self._bpy)


# ---------------------------------------------------------------------------
# Yang-Zhang Volatility
# Best all-round OHLC estimator — handles overnight gaps.
# Use for daily / higher timeframes.
# ---------------------------------------------------------------------------
class YangZhang:
    """
    Yang-Zhang volatility estimator.
    period: rolling window (default 20)
    k: weighting parameter (default 0.34, from Yang-Zhang paper)
    bars_per_year: annualisation factor
    """

    def __init__(self, period: int = 20, k: float = 0.34, bars_per_year: float = 252.0) -> None:
        self._period = max(2, period)
        self._k = clamp(k, 0.0, 1.0)
        self._bpy = bars_per_year
        self._opens: FloatCircularBuffer = FloatCircularBuffer(self._period)
        self._closes: FloatCircularBuffer = FloatCircularBuffer(self._period)
        self._rs_buf: FloatCircularBuffer = FloatCircularBuffer(self._period)
        self._prev_close = 0.0
        self._value = 0.0
        self._count = 0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._opens.full

    def update(self, open_: float, high: float, low: float, close: float) -> float:
        if not all(is_valid_number(v) and v > 0 for v in [open_, high, low, close]):
            return self._value

        if self._count > 0:
            # Overnight return component
            log_oc = safe_log(safe_div(open_, self._prev_close))
            # Open-to-close return
            log_co = safe_log(safe_div(close, open_))
            # Rogers-Satchell component
            rs = (
                safe_log(safe_div(high, close)) * safe_log(safe_div(high, open_))
                + safe_log(safe_div(low, close)) * safe_log(safe_div(low, open_))
            )
            self._opens.push(log_oc)
            self._closes.push(log_co)
            self._rs_buf.push(non_negative(rs))

        self._prev_close = close
        self._count += 1

        if not self._opens.full:
            return self._value

        opens = self._opens.to_list()
        closes = self._closes.to_list()
        n = len(opens)
        mean_o = sum(opens) / n
        mean_c = sum(closes) / n

        var_o = sum((v - mean_o) ** 2 for v in opens) / (n - 1)
        var_c = sum((v - mean_c) ** 2 for v in closes) / (n - 1)
        var_rs = self._rs_buf.mean()

        vol_sq = var_o + self._k * var_c + (1 - self._k) * var_rs
        self._value = safe_sqrt(non_negative(vol_sq) * self._bpy)
        return self._value

    def reset(self) -> None:
        self.__init__(self._period, self._k, self._bpy)


# ---------------------------------------------------------------------------
# Garman-Klass Volatility (kept for reference / Optuna comparison)
# Assumes no drift — less ideal for trending markets.
# ---------------------------------------------------------------------------
class GarmanKlass:
    """
    Garman-Klass volatility estimator.
    Assumes no drift — prefer Rogers-Satchell or Yang-Zhang for trends.
    """

    def __init__(self, period: int = 20, bars_per_year: float = 252.0) -> None:
        self._period = max(2, period)
        self._bpy = bars_per_year
        self._buf = FloatCircularBuffer(self._period)
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._buf.full

    def update(self, open_: float, high: float, low: float, close: float) -> float:
        if not all(is_valid_number(v) and v > 0 for v in [open_, high, low, close]):
            return self._value

        log_hl = safe_log(safe_div(high, low))
        log_co = safe_log(safe_div(close, open_))
        gk = 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2
        self._buf.push(non_negative(gk))

        if not self._buf.full:
            return self._value

        self._value = safe_sqrt(self._buf.mean() * self._bpy)
        return self._value

    def reset(self) -> None:
        self.__init__(self._period, self._bpy)
