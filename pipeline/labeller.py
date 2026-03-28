"""
Forward labelling for supervised learning.
Computes MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion)
for each bar given a hypothetical entry. Feeds the runway predictor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from .cleaner import Bar
from core.math_utils import safe_div
from core.numeric import non_negative


@dataclass
class Label:
    """Forward outcome for a single bar."""
    bar_index: int
    timestamp: float
    direction: int        # 1 = long, -1 = short
    entry_price: float
    mfe: float            # max favorable excursion in price units
    mae: float            # max adverse excursion in price units
    mfe_pct: float        # mfe as fraction of entry price
    mae_pct: float        # mae as fraction of entry price
    outcome_bars: int     # how many bars until SL or TP hit, or horizon
    hit_tp: bool
    hit_sl: bool


def label_bars(
    bars: List[Bar],
    direction: int,
    horizon: int = 20,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 3.0,
    atr_values: Optional[List[float]] = None,
) -> List[Label]:
    """
    Label each bar with forward MFE/MAE.

    direction: 1 = long entries, -1 = short entries
    horizon: max bars to look forward
    sl_atr_mult: stop loss distance in ATR multiples
    tp_atr_mult: take profit distance in ATR multiples
    atr_values: ATR per bar (same length as bars). If None, uses fixed % of price.
    """
    labels = []
    n = len(bars)

    for i in range(n - 1):
        bar = bars[i]
        entry = bar.close

        if atr_values and i < len(atr_values) and atr_values[i] > 0:
            atr = atr_values[i]
        else:
            atr = entry * 0.001  # fallback: 0.1% of price

        sl_dist = atr * sl_atr_mult
        tp_dist = atr * tp_atr_mult

        if direction == 1:
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        mfe = 0.0
        mae = 0.0
        hit_tp = False
        hit_sl = False
        outcome_bars = 0

        for j in range(i + 1, min(i + 1 + horizon, n)):
            fwd = bars[j]
            outcome_bars = j - i

            if direction == 1:
                favorable = fwd.high - entry
                adverse = entry - fwd.low
            else:
                favorable = entry - fwd.low
                adverse = fwd.high - entry

            mfe = max(mfe, non_negative(favorable))
            mae = max(mae, non_negative(adverse))

            # Check SL/TP hit
            if direction == 1:
                if fwd.low <= sl_price:
                    hit_sl = True
                    break
                if fwd.high >= tp_price:
                    hit_tp = True
                    break
            else:
                if fwd.high >= sl_price:
                    hit_sl = True
                    break
                if fwd.low <= tp_price:
                    hit_tp = True
                    break

        labels.append(Label(
            bar_index=i,
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry,
            mfe=mfe,
            mae=mae,
            mfe_pct=safe_div(mfe, entry),
            mae_pct=safe_div(mae, entry),
            outcome_bars=outcome_bars,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
        ))

    return labels
