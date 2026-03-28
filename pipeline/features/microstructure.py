"""
Market microstructure features.
VPIN requires tick data — separate pipeline branch, merged at bar close.
Elevated VPIN = informed order flow = potential sharp move or toxicity.
Block entries when VPIN > threshold.
"""
from __future__ import annotations

import math
from typing import List
from core.math_utils import safe_div, safe_sqrt
from core.numeric import non_negative, clamp, is_valid_number
from core.memory import FloatCircularBuffer


# ---------------------------------------------------------------------------
# VPIN — Volume-Synchronized Probability of Informed Trading
# Measures order flow imbalance using volume buckets.
# Values near 0 = uninformed flow. Values near 1 = high toxicity risk.
# ---------------------------------------------------------------------------
class VPIN:
    """
    VPIN estimator using bulk volume classification (BVC).
    bucket_size: volume per bucket (tune to ~1/50 of ADV)
    n_buckets: number of buckets in rolling window (default 50)
    """

    def __init__(self, bucket_size: float = 1000.0, n_buckets: int = 50) -> None:
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        self._bucket_size = bucket_size
        self._n_buckets = max(10, n_buckets)
        self._buy_volumes = FloatCircularBuffer(self._n_buckets)
        self._sell_volumes = FloatCircularBuffer(self._n_buckets)
        self._current_buy = 0.0
        self._current_sell = 0.0
        self._current_vol = 0.0
        self._prev_close = 0.0
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._buy_volumes.full

    def update_tick(self, price: float, volume: float, prev_close: float) -> float:
        """
        Update with a single tick.
        Uses bulk volume classification: proportion of buy/sell estimated
        from price change relative to sigma.
        """
        if not all(is_valid_number(v) for v in [price, volume, prev_close]):
            return self._value
        if volume <= 0 or prev_close <= 0:
            return self._value

        z = safe_div(price - prev_close, prev_close * 0.001, fallback=0.0)
        z_clamped = clamp(z, -4.0, 4.0)
        buy_prob = _standard_normal_cdf(z_clamped)
        buy_vol = volume * buy_prob
        sell_vol = volume * (1.0 - buy_prob)

        self._current_buy += buy_vol
        self._current_sell += sell_vol
        self._current_vol += volume

        while self._current_vol >= self._bucket_size:
            excess = self._current_vol - self._bucket_size
            scale = safe_div(self._bucket_size, self._current_vol, fallback=1.0)
            bucket_buy = self._current_buy * scale
            bucket_sell = self._current_sell * scale

            self._buy_volumes.push(bucket_buy)
            self._sell_volumes.push(bucket_sell)

            self._current_buy = self._current_buy * (1 - scale)
            self._current_sell = self._current_sell * (1 - scale)
            self._current_vol = excess

            self._recalculate()

        return self._value

    def _recalculate(self) -> None:
        if not self._buy_volumes.full:
            return
        buys = self._buy_volumes.to_list()
        sells = self._sell_volumes.to_list()
        imbalances = [abs(b - s) for b, s in zip(buys, sells)]
        total_vol = sum(b + s for b, s in zip(buys, sells))
        self._value = clamp(safe_div(sum(imbalances), total_vol, fallback=0.0), 0.0, 1.0)

    def reset(self) -> None:
        self.__init__(self._bucket_size, self._n_buckets)


# ---------------------------------------------------------------------------
# Tick Imbalance — bar-level buy/sell pressure from tick data
# ---------------------------------------------------------------------------
class TickImbalance:
    """
    Rolling tick imbalance: (buy_ticks - sell_ticks) / total_ticks
    Returns -1.0 to +1.0. Positive = buy pressure, negative = sell pressure.
    """

    def __init__(self, period: int = 20) -> None:
        self._period = max(2, period)
        self._imbalances = FloatCircularBuffer(self._period)
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._imbalances.full

    def update_bar(self, buy_ticks: int, sell_ticks: int) -> float:
        """Update with aggregated tick counts for a completed bar."""
        total = buy_ticks + sell_ticks
        imbalance = safe_div(float(buy_ticks - sell_ticks), float(total), fallback=0.0)
        self._imbalances.push(clamp(imbalance, -1.0, 1.0))
        self._value = self._imbalances.mean()
        return self._value

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# Spread Cost Monitor — tracks realised spread cost per trade
# ---------------------------------------------------------------------------
class SpreadMonitor:
    """
    Tracks rolling average spread and flags elevated spread conditions.
    Feeds into friction cost calculation for the runway predictor.
    """

    def __init__(self, period: int = 50, spike_multiplier: float = 2.5) -> None:
        self._period = max(10, period)
        self._spreads = FloatCircularBuffer(self._period)
        self._spike_mult = spike_multiplier
        self._current_spread = 0.0

    @property
    def average_spread(self) -> float:
        return self._spreads.mean()

    @property
    def current_spread(self) -> float:
        return self._current_spread

    @property
    def is_elevated(self) -> bool:
        """True if current spread is significantly above rolling average."""
        avg = self.average_spread
        if avg <= 0:
            return False
        return self._current_spread > avg * self._spike_mult

    @property
    def ready(self) -> bool:
        return self._spreads.full

    def update(self, bid: float, ask: float) -> float:
        if not all(is_valid_number(v) and v > 0 for v in [bid, ask]):
            return self._current_spread
        spread = non_negative(ask - bid)
        self._current_spread = spread
        self._spreads.push(spread)
        return spread

    def reset(self) -> None:
        self.__init__(self._period, self._spike_multiplier)


# ---------------------------------------------------------------------------
# Kurtosis monitor — excess kurtosis flags fat-tail risk
# Block or reduce size when kurtosis is elevated.
# ---------------------------------------------------------------------------
class KurtosisMonitor:
    """
    Rolling excess kurtosis of returns.
    Normal distribution = 0 excess kurtosis.
    High positive values = fat tails / crash risk.
    Threshold ~3.0 warrants caution.
    """

    def __init__(self, period: int = 100) -> None:
        self._period = max(10, period)
        self._returns: list[float] = []
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    @property
    def is_elevated(self) -> bool:
        return self._value > 3.0

    @property
    def ready(self) -> bool:
        return len(self._returns) >= self._period

    def update(self, ret: float) -> float:
        if not is_valid_number(ret):
            return self._value

        self._returns.append(ret)
        if len(self._returns) > self._period:
            self._returns.pop(0)

        if len(self._returns) < 4:
            return self._value

        n = len(self._returns)
        mean = sum(self._returns) / n
        diffs = [r - mean for r in self._returns]
        m2 = sum(d ** 2 for d in diffs) / n
        m4 = sum(d ** 4 for d in diffs) / n

        if m2 <= 0:
            return self._value

        excess_kurt = safe_div(m4, m2 ** 2, fallback=0.0) - 3.0
        self._value = excess_kurt
        return self._value

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# Standard normal CDF approximation (no scipy dependency in core)
# ---------------------------------------------------------------------------
def _standard_normal_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation. Error < 7.5e-8."""
    x = clamp(x, -6.0, 6.0)
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = safe_div(1.0, 1.0 + 0.2316419 * x, fallback=1.0)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return 0.5 + sign * (cdf - 0.5)
