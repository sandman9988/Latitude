"""
Regime detector — classifies market state as TRENDING, RANGING, or REVERSAL.
Combines Markov HMM state probabilities + KER + VHF + Chande CMO.
Only TRENDING regime allows entries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional
from core.math_utils import safe_div, safe_sqrt
from core.numeric import clamp, is_valid_number
from core.logger import get_logger
from .trend import KER, VHF, ChandeCMO

logger = get_logger("regime")


class RegimeState(IntEnum):
    UNKNOWN = 0
    TRENDING = 1
    RANGING = 2
    REVERSAL = 3


@dataclass
class RegimeReading:
    state: RegimeState
    confidence: float       # 0.0-1.0
    ker: float
    vhf: float
    cmo: float
    hmm_trend_prob: float   # P(trending) from HMM
    direction: int          # 1 = bullish trend, -1 = bearish trend, 0 = unclear


class RegimeDetector:
    """
    Multi-signal regime classifier.
    Combines rule-based scoring (KER + VHF + CMO) with a lightweight
    2-state Gaussian HMM updated online via Viterbi-style smoothing.

    All entries must pass regime.state == TRENDING before proceeding.
    """

    def __init__(
        self,
        ker_period: int = 10,
        vhf_period: int = 28,
        cmo_period: int = 14,
        ker_threshold: float = 0.4,
        vhf_threshold: float = 0.5,
        cmo_trend_min: float = 20.0,
        hmm_period: int = 50,
        symbol: str = "",
        tf: str = "",
    ) -> None:
        self._ker = KER(ker_period)
        self._vhf = VHF(vhf_period)
        self._cmo = ChandeCMO(cmo_period)
        self._ker_threshold = ker_threshold
        self._vhf_threshold = vhf_threshold
        self._cmo_trend_min = cmo_trend_min
        self._symbol = symbol
        self._tf = tf

        # Lightweight online HMM: 2 states (trending=0, ranging=1)
        self._hmm = _OnlineGaussianHMM(n_states=2, window=hmm_period)

        self._last: Optional[RegimeReading] = None
        self._count = 0

    @property
    def state(self) -> RegimeState:
        if self._last is None:
            return RegimeState.UNKNOWN
        return self._last.state

    @property
    def is_trending(self) -> bool:
        return self.state == RegimeState.TRENDING

    @property
    def last(self) -> Optional[RegimeReading]:
        return self._last

    @property
    def ready(self) -> bool:
        return self._ker.ready and self._vhf.ready and self._cmo.ready

    def update(self, high: float, low: float, close: float, ret: float = 0.0) -> RegimeReading:
        """
        Update with latest bar. ret = bar return (close/prev_close - 1) for HMM.
        """
        if not all(is_valid_number(v) for v in [high, low, close]):
            return self._last or RegimeReading(
                state=RegimeState.UNKNOWN, confidence=0.0,
                ker=0.0, vhf=0.0, cmo=0.0, hmm_trend_prob=0.5, direction=0
            )

        self._count += 1
        ker = self._ker.update(close)
        vhf = self._vhf.update(close)
        cmo = self._cmo.update(close)
        hmm_prob = self._hmm.update(ret)

        reading = self._classify(ker, vhf, cmo, hmm_prob)
        self._last = reading

        if self._count % 100 == 0:
            logger.regime(
                self._symbol, self._tf,
                state=reading.state.name,
                confidence=reading.confidence,
                ker=ker, vhf=vhf, cmo=cmo, hmm=hmm_prob
            )

        return reading

    def _classify(self, ker: float, vhf: float, cmo: float, hmm_prob: float) -> RegimeReading:
        # Score 0-3: how many trend signals agree
        trend_votes = 0
        if ker >= self._ker_threshold:
            trend_votes += 1
        if vhf >= self._vhf_threshold:
            trend_votes += 1
        if abs(cmo) >= self._cmo_trend_min:
            trend_votes += 1

        # HMM adds weighted vote
        hmm_vote = hmm_prob  # probability of trending state

        # Combined score
        rule_score = safe_div(trend_votes, 3.0)
        combined = 0.6 * rule_score + 0.4 * hmm_vote

        # Direction from CMO sign
        if cmo > self._cmo_trend_min:
            direction = 1
        elif cmo < -self._cmo_trend_min:
            direction = -1
        else:
            direction = 0

        if combined >= 0.6:
            state = RegimeState.TRENDING
        elif combined <= 0.3:
            # Check for reversal: trend was strong, now rapidly weakening
            if self._last and self._last.state == RegimeState.TRENDING and combined < 0.2:
                state = RegimeState.REVERSAL
            else:
                state = RegimeState.RANGING
        else:
            state = RegimeState.RANGING

        return RegimeReading(
            state=state,
            confidence=clamp(combined, 0.0, 1.0),
            ker=ker,
            vhf=vhf,
            cmo=cmo,
            hmm_trend_prob=hmm_prob,
            direction=direction,
        )

    def reset(self) -> None:
        self.__init__(
            symbol=self._symbol,
            tf=self._tf,
        )


# ---------------------------------------------------------------------------
# Lightweight Online Gaussian HMM
# 2-state: state 0 = trending (low vol returns), state 1 = ranging (noisy)
# Uses simple Baum-Welch sufficient statistics updated online.
# ---------------------------------------------------------------------------
class _OnlineGaussianHMM:
    """
    Minimal 2-state HMM with Gaussian emissions, updated online.
    Returns P(state=0 | observations) — probability of trending state.
    """

    def __init__(self, n_states: int = 2, window: int = 50) -> None:
        self._n = n_states
        self._window = window
        self._obs: list[float] = []

        # Initial parameters: state 0 = low vol trend, state 1 = high vol range
        self._mu = [0.001, 0.0]       # mean return per state
        self._sigma = [0.005, 0.015]  # std dev per state
        self._pi = [0.5, 0.5]         # initial state probs
        # Transition matrix rows: from state i
        self._A = [[0.9, 0.1], [0.1, 0.9]]

        self._alpha = [0.5, 0.5]      # filtered state probs
        self._value = 0.5

    def update(self, obs: float) -> float:
        """Update with new observation (return). Returns P(state=0)."""
        if not is_valid_number(obs):
            return self._value

        self._obs.append(obs)
        if len(self._obs) > self._window:
            self._obs.pop(0)

        # Re-estimate parameters every 20 observations
        if len(self._obs) >= 10 and len(self._obs) % 20 == 0:
            self._estimate_params()

        # Forward step
        new_alpha = []
        total = 0.0
        for j in range(self._n):
            pred = sum(self._alpha[i] * self._A[i][j] for i in range(self._n))
            emission = _gaussian_pdf(obs, self._mu[j], self._sigma[j])
            a = pred * emission
            new_alpha.append(a)
            total += a

        if total > 0:
            self._alpha = [safe_div(a, total, fallback=safe_div(1.0, self._n)) for a in new_alpha]
        else:
            self._alpha = [safe_div(1.0, self._n)] * self._n

        self._value = self._alpha[0]
        return self._value

    def _estimate_params(self) -> None:
        """
        Simple k-means style re-estimation.
        Split observations into two groups by magnitude (trending = small moves).
        """
        if len(self._obs) < 4:
            return
        sorted_abs = sorted(abs(o) for o in self._obs)
        median_abs = sorted_abs[len(sorted_abs) // 2]

        low_vol = [o for o in self._obs if abs(o) <= median_abs]
        high_vol = [o for o in self._obs if abs(o) > median_abs]

        if low_vol:
            self._mu[0] = sum(low_vol) / len(low_vol)
            self._sigma[0] = max(
                safe_sqrt(sum((v - self._mu[0]) ** 2 for v in low_vol) / len(low_vol)),
                1e-6
            )
        if high_vol:
            self._mu[1] = sum(high_vol) / len(high_vol)
            self._sigma[1] = max(
                safe_sqrt(sum((v - self._mu[1]) ** 2 for v in high_vol) / len(high_vol)),
                1e-6
            )


def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1e-10
    z = safe_div(x - mu, sigma, fallback=0.0)
    return max(math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi)), 1e-10)
