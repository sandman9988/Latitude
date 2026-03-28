"""
Momentum indicators — low lag, tuned for trend-following entries.
Laguerre RSI is the primary entry timing signal.
"""
from __future__ import annotations

import math
from core.math_utils import safe_div, safe_sqrt
from core.numeric import clamp, is_valid_number
from core.memory import FloatCircularBuffer
from .smoothing import make_smoother, Smoother


# ---------------------------------------------------------------------------
# Laguerre RSI
# Tunable gamma: 0.0 = fastest/noisiest, 1.0 = slowest/smoothest
# 0.7-0.8 recommended — tune via Optuna
# ---------------------------------------------------------------------------
class LaguerreRSI:
    """
    Laguerre RSI — dramatically reduced lag vs standard RSI.
    gamma: damping factor 0.0-1.0 (default 0.7)
    Returns 0.0-1.0.
    """

    def __init__(self, gamma: float = 0.7) -> None:
        self._gamma = clamp(gamma, 0.0, 0.99)
        self._l0 = self._l1 = self._l2 = self._l3 = 0.0
        self._value = 0.0
        self._count = 0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._count >= 4

    def update(self, price: float) -> float:
        if not is_valid_number(price):
            return self._value

        g = self._gamma
        l0_prev = self._l0
        l1_prev = self._l1
        l2_prev = self._l2
        l3_prev = self._l3

        self._l0 = (1 - g) * price + g * l0_prev
        self._l1 = -g * self._l0 + l0_prev + g * l1_prev
        self._l2 = -g * self._l1 + l1_prev + g * l2_prev
        self._l3 = -g * self._l2 + l2_prev + g * l3_prev

        cu = 0.0
        cd = 0.0
        if self._l0 >= self._l1:
            cu += self._l0 - self._l1
        else:
            cd += self._l1 - self._l0
        if self._l1 >= self._l2:
            cu += self._l1 - self._l2
        else:
            cd += self._l2 - self._l1
        if self._l2 >= self._l3:
            cu += self._l2 - self._l3
        else:
            cd += self._l3 - self._l2

        self._value = safe_div(cu, cu + cd, fallback=0.5)
        self._count += 1
        return self._value

    def reset(self) -> None:
        self.__init__(self._gamma)


# ---------------------------------------------------------------------------
# Fisher Transform
# Normalises price to Gaussian — sharp reversals pop clearly.
# Useful as divergence signal and overbought/oversold gate.
# ---------------------------------------------------------------------------
class FisherTransform:
    """
    Fisher Transform.
    period: highest high / lowest low lookback (default 10)
    Returns typically -3.0 to +3.0.
    """

    def __init__(self, period: int = 10) -> None:
        self._period = max(2, period)
        self._highs = FloatCircularBuffer(self._period)
        self._lows = FloatCircularBuffer(self._period)
        self._fish = 0.0
        self._fish_prev = 0.0
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    @property
    def signal(self) -> float:
        """Previous bar's Fisher value — use for crossovers."""
        return self._fish_prev

    @property
    def ready(self) -> bool:
        return self._highs.full

    def update(self, high: float, low: float) -> float:
        if not is_valid_number(high) or not is_valid_number(low):
            return self._value

        self._highs.push(high)
        self._lows.push(low)

        if not self._highs.full:
            return self._value

        highest = max(self._highs.to_list())
        lowest = min(self._lows.to_list())

        hl_range = highest - lowest
        mid = 0.5 * (high + low)
        value = clamp(safe_div(2.0 * (mid - lowest), hl_range, fallback=0.0) - 1.0, -0.999, 0.999)

        self._fish_prev = self._fish
        self._fish = 0.5 * math.log(safe_div(1.0 + value, 1.0 - value, fallback=1.0)) + 0.5 * self._fish_prev
        self._value = self._fish

        return self._value

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# Connors RSI — 3-component composite
# RSI(3) + UpDown streak RSI + Percentile rank
# Good for trend pullback entry timing.
# ---------------------------------------------------------------------------
class ConnorsRSI:
    """
    Connors RSI.
    rsi_period: fast RSI period (default 3)
    streak_period: streak RSI period (default 2)
    rank_period: percentile rank lookback (default 100)
    Returns 0-100.
    """

    def __init__(self, rsi_period: int = 3, streak_period: int = 2, rank_period: int = 100) -> None:
        self._rsi_period = rsi_period
        self._streak_period = streak_period
        self._rsi = _RSI(rsi_period)
        self._streak_rsi = _RSI(streak_period)
        self._rank_period = max(10, rank_period)
        self._returns: list[float] = []
        self._prev_price = 0.0
        self._streak = 0.0
        self._value = 50.0
        self._count = 0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._count >= self._rank_period

    def update(self, price: float) -> float:
        if not is_valid_number(price):
            return self._value

        if self._count == 0:
            self._prev_price = price
            self._count += 1
            return self._value

        ret = price - self._prev_price

        # Streak: consecutive up/down bars
        if ret > 0:
            self._streak = self._streak + 1 if self._streak > 0 else 1
        elif ret < 0:
            self._streak = self._streak - 1 if self._streak < 0 else -1
        else:
            self._streak = 0

        rsi_val = self._rsi.update(price)
        streak_rsi_val = self._streak_rsi.update(self._streak)

        self._returns.append(ret)
        if len(self._returns) > self._rank_period:
            self._returns.pop(0)

        rank = sum(1 for r in self._returns if r < ret)
        pct_rank = safe_div(rank * 100.0, len(self._returns), fallback=50.0)

        self._value = safe_div(rsi_val + streak_rsi_val + pct_rank, 3.0)
        self._prev_price = price
        self._count += 1
        return self._value

    def reset(self) -> None:
        self.__init__(self._rsi_period, self._streak_period, self._rank_period)


# ---------------------------------------------------------------------------
# RVGI — Relative Vigor Index
# Close vs range ratio — less lag than RSI, trend-aligned.
# ---------------------------------------------------------------------------
class RVGI:
    """
    Relative Vigor Index.
    period: smoothing period (default 10)
    Returns -1.0 to +1.0.
    """

    def __init__(self, period: int = 10) -> None:
        self._period = max(4, period)
        self._numerators = FloatCircularBuffer(self._period)
        self._denominators = FloatCircularBuffer(self._period)
        self._value = 0.0
        self._signal = 0.0
        self._open_buf = FloatCircularBuffer(4)
        self._high_buf = FloatCircularBuffer(4)
        self._low_buf = FloatCircularBuffer(4)
        self._close_buf = FloatCircularBuffer(4)

    @property
    def value(self) -> float:
        return self._value

    @property
    def signal(self) -> float:
        return self._signal

    @property
    def ready(self) -> bool:
        return self._numerators.full

    def update(self, open_: float, high: float, low: float, close: float) -> float:
        if not all(is_valid_number(v) for v in [open_, high, low, close]):
            return self._value

        self._open_buf.push(open_)
        self._high_buf.push(high)
        self._low_buf.push(low)
        self._close_buf.push(close)

        if not self._open_buf.full:
            return self._value

        o = self._open_buf.to_list()
        h = self._high_buf.to_list()
        l = self._low_buf.to_list()
        c = self._close_buf.to_list()

        # Symmetrical weighted sum (triangular weights 1,2,2,1)
        num = safe_div(
            (c[3] - o[3]) + 2 * (c[2] - o[2]) + 2 * (c[1] - o[1]) + (c[0] - o[0]),
            6.0
        )
        den = safe_div(
            (h[3] - l[3]) + 2 * (h[2] - l[2]) + 2 * (h[1] - l[1]) + (h[0] - l[0]),
            6.0
        )

        self._numerators.push(num)
        self._denominators.push(den)

        if not self._numerators.full:
            return self._value

        num_sum = self._numerators.total()
        den_sum = self._denominators.total()

        self._value = safe_div(num_sum, den_sum, fallback=0.0)

        # 4-bar signal line
        vals = self._numerators.to_list()
        if len(vals) >= 4:
            self._signal = safe_div(
                vals[-1] + 2 * vals[-2] + 2 * vals[-3] + vals[-4],
                6.0
            )

        return self._value

    def reset(self) -> None:
        self.__init__(self._period)


# ---------------------------------------------------------------------------
# Internal RSI helper (Wilder smoothing)
# ---------------------------------------------------------------------------
class _RSI:
    def __init__(self, period: int = 14) -> None:
        self._period = max(2, period)
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._prev = 0.0
        self._count = 0
        self._value = 50.0

    @property
    def value(self) -> float:
        return self._value

    def update(self, price: float) -> float:
        if not is_valid_number(price):
            return self._value

        if self._count == 0:
            self._prev = price
            self._count += 1
            return self._value

        change = price - self._prev
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if self._count <= self._period:
            self._avg_gain += gain
            self._avg_loss += loss
            if self._count == self._period:
                self._avg_gain /= self._period
                self._avg_loss /= self._period
        else:
            self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
            self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period

        rs = safe_div(self._avg_gain, self._avg_loss, fallback=100.0)
        self._value = 100.0 - safe_div(100.0, 1.0 + rs, fallback=50.0)
        self._prev = price
        self._count += 1
        return self._value

    def reset(self) -> None:
        self.__init__(self._period)
