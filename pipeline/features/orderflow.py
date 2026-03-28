"""
Order flow features from Depth of Market (DOM) and tick data.

DOM is live-only (no historical download possible from cTrader).
Record DOM snapshots during live session; use for:
  - Order Book Imbalance (OBI): real bid/ask size at each level
  - Cumulative Delta: running buy vol - sell vol
  - DOM depth ratio: total bid size / total ask size
  - Price level absorption: large orders sitting at key levels

Tick-based features (from historical bid/ask tick download):
  - Tick imbalance per bar
  - Aggressive buy/sell classification
  - Volume-weighted price pressure
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from core.math_utils import safe_div, safe_sqrt
from core.numeric import clamp, non_negative, is_valid_number
from core.memory import FloatCircularBuffer


# ---------------------------------------------------------------------------
# DOM Snapshot — one point-in-time order book state
# ---------------------------------------------------------------------------

@dataclass
class DOMLevel:
    price: float
    bid_size: float   # volume on the bid at this price
    ask_size: float   # volume on the ask at this price


@dataclass
class DOMSnapshot:
    timestamp: float
    symbol: str
    levels: List[DOMLevel] = field(default_factory=list)
    mid_price: float = 0.0

    @property
    def total_bid_size(self) -> float:
        return sum(l.bid_size for l in self.levels)

    @property
    def total_ask_size(self) -> float:
        return sum(l.ask_size for l in self.levels)

    @property
    def order_book_imbalance(self) -> float:
        """
        OBI = (bid_size - ask_size) / (bid_size + ask_size)
        +1.0 = all bids, -1.0 = all asks, 0 = balanced
        """
        b = self.total_bid_size
        a = self.total_ask_size
        return clamp(safe_div(b - a, b + a, fallback=0.0), -1.0, 1.0)

    def weighted_mid_price(self) -> float:
        """Volume-weighted mid price across DOM levels."""
        total = 0.0
        weighted = 0.0
        for l in self.levels:
            size = l.bid_size + l.ask_size
            weighted += l.price * size
            total += size
        return safe_div(weighted, total, fallback=self.mid_price)

    def depth_at_n_levels(self, n: int) -> Tuple[float, float]:
        """Total bid and ask size in top N levels."""
        levels = sorted(self.levels, key=lambda l: l.price)
        bid_levels = [l for l in levels if l.bid_size > 0][-n:]
        ask_levels = [l for l in levels if l.ask_size > 0][:n]
        return (
            sum(l.bid_size for l in bid_levels),
            sum(l.ask_size for l in ask_levels),
        )


# ---------------------------------------------------------------------------
# DOM Order Book Imbalance — rolling feature from live DOM events
# ---------------------------------------------------------------------------

class OrderBookImbalance:
    """
    Computes rolling OBI from a stream of DOM snapshots.
    Updated on each ProtoOADepthEvent.
    Returns smoothed OBI ready for feature vector.
    """

    def __init__(self, smoothing_period: int = 10, depth_levels: int = 5) -> None:
        self._period = max(2, smoothing_period)
        self._depth = depth_levels
        self._obi_buf = FloatCircularBuffer(self._period)
        self._dom: Optional[DOMSnapshot] = None
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    @property
    def ready(self) -> bool:
        return self._obi_buf.full

    def update_from_event(self, event_quotes: List[dict], timestamp: float, symbol: str) -> float:
        """
        Update from ProtoOADepthEvent fields.
        event_quotes: list of {"id": int, "bid": float|None, "ask": float|None, "size": float}
        """
        if self._dom is None:
            self._dom = DOMSnapshot(timestamp=timestamp, symbol=symbol)

        # Apply updates: new quotes overwrite by ID, deleted quotes (size=0) are removed
        existing = {int(l.price * 100000): l for l in self._dom.levels}

        for q in event_quotes:
            price_key = int(q.get("price", 0) * 100000)
            size = float(q.get("size", 0) or 0)
            bid = float(q.get("bid") or 0)
            ask = float(q.get("ask") or 0)

            if size <= 0:
                existing.pop(price_key, None)
            else:
                price = q.get("price", bid if bid > 0 else ask)
                existing[price_key] = DOMLevel(
                    price=float(price),
                    bid_size=bid * size if bid > 0 else 0.0,
                    ask_size=ask * size if ask > 0 else 0.0,
                )

        self._dom.levels = list(existing.values())
        self._dom.timestamp = timestamp

        obi = self._dom.order_book_imbalance
        self._obi_buf.push(obi)
        self._value = self._obi_buf.mean()
        return self._value


# ---------------------------------------------------------------------------
# Cumulative Delta — running buy volume minus sell volume
# Resets at configurable interval (e.g. per session or per bar)
# ---------------------------------------------------------------------------

class CumulativeDelta:
    """
    Running cumulative delta from classified ticks.
    buy_volume - sell_volume. Positive = net buying pressure.
    """

    def __init__(self, period: int = 50) -> None:
        self._period = max(2, period)
        self._delta_buf = FloatCircularBuffer(self._period)
        self._cumulative = 0.0
        self._bar_buy = 0.0
        self._bar_sell = 0.0

    @property
    def cumulative(self) -> float:
        return self._cumulative

    @property
    def bar_delta(self) -> float:
        """Delta for current incomplete bar."""
        return self._bar_buy - self._bar_sell

    @property
    def rolling_mean_delta(self) -> float:
        return self._delta_buf.mean()

    def update_tick(self, volume: float, is_buy: bool) -> None:
        """Update with a classified tick."""
        if not is_valid_number(volume) or volume <= 0:
            return
        if is_buy:
            self._bar_buy += volume
            self._cumulative += volume
        else:
            self._bar_sell += volume
            self._cumulative -= volume

    def close_bar(self) -> float:
        """Call at bar close. Returns bar delta and resets bar accumulators."""
        delta = self._bar_buy - self._bar_sell
        self._delta_buf.push(delta)
        self._bar_buy = 0.0
        self._bar_sell = 0.0
        return delta

    def reset_session(self) -> None:
        """Reset cumulative delta (e.g. start of new session)."""
        self._cumulative = 0.0
        self._bar_buy = 0.0
        self._bar_sell = 0.0


# ---------------------------------------------------------------------------
# Tick Classifier — classify each tick as buy or sell
# Uses Lee-Ready rule: tick above prev ask = buy, below prev bid = sell
# ---------------------------------------------------------------------------

class TickClassifier:
    """
    Classifies ticks as buy (aggressive buyer) or sell (aggressive seller).
    Uses Lee-Ready tick rule when bid/ask available,
    falls back to uptick/downtick rule on price movement.
    """

    def __init__(self) -> None:
        self._prev_bid = 0.0
        self._prev_ask = 0.0
        self._prev_price = 0.0
        self._count = 0

    def classify(self, price: float, bid: Optional[float] = None, ask: Optional[float] = None) -> Optional[bool]:
        """
        Returns True = buy, False = sell, None = indeterminate.
        """
        result: Optional[bool] = None

        if bid is not None and ask is not None and bid > 0 and ask > 0:
            # Lee-Ready: trade at or above ask = buy, at or below bid = sell
            if price >= ask:
                result = True
            elif price <= bid:
                result = False
            else:
                # Inside spread: use tick direction
                if self._prev_price > 0:
                    result = price > self._prev_price if price != self._prev_price else None
        elif self._prev_price > 0:
            if price > self._prev_price:
                result = True
            elif price < self._prev_price:
                result = False

        if bid is not None:
            self._prev_bid = bid
        if ask is not None:
            self._prev_ask = ask
        self._prev_price = price
        self._count += 1
        return result


# ---------------------------------------------------------------------------
# Bar-level order flow aggregator
# Accumulates ticks within a bar and produces a feature vector at close
# ---------------------------------------------------------------------------

@dataclass
class BarOrderFlow:
    """Aggregated order flow metrics for one completed bar."""
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_ticks: int = 0
    sell_ticks: int = 0
    unknown_ticks: int = 0
    delta: float = 0.0           # buy_volume - sell_volume
    delta_pct: float = 0.0       # delta / total_volume
    tick_imbalance: float = 0.0  # (buy_ticks - sell_ticks) / total_ticks
    absorption: float = 0.0      # large order sitting detection (vs price move)

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def cvd(self) -> float:
        """Cumulative volume delta ratio."""
        return clamp(safe_div(self.delta, self.total_volume, fallback=0.0), -1.0, 1.0)


class BarOrderFlowAggregator:
    """
    Aggregates tick-classified data into per-bar order flow features.
    Call update_tick() for each tick, close_bar() at bar close.
    """

    def __init__(self, period: int = 20) -> None:
        self._classifier = TickClassifier()
        self._cum_delta = CumulativeDelta(period)
        self._history: List[BarOrderFlow] = []
        self._max_history = period
        self._current = BarOrderFlow()

    def update_tick(self, price: float, volume: float, bid: Optional[float] = None, ask: Optional[float] = None) -> None:
        if not is_valid_number(price) or not is_valid_number(volume):
            return
        is_buy = self._classifier.classify(price, bid, ask)
        if is_buy is True:
            self._current.buy_volume += volume
            self._current.buy_ticks += 1
            self._cum_delta.update_tick(volume, True)
        elif is_buy is False:
            self._current.sell_volume += volume
            self._current.sell_ticks += 1
            self._cum_delta.update_tick(volume, False)
        else:
            self._current.unknown_ticks += 1

    def close_bar(self) -> BarOrderFlow:
        """Finalise the current bar and return its order flow metrics."""
        bar = self._current
        bar.delta = bar.buy_volume - bar.sell_volume
        total = bar.total_volume
        bar.delta_pct = safe_div(bar.delta, total, fallback=0.0)
        total_ticks = bar.buy_ticks + bar.sell_ticks
        bar.tick_imbalance = safe_div(
            float(bar.buy_ticks - bar.sell_ticks), float(total_ticks), fallback=0.0
        )
        self._cum_delta.close_bar()
        self._history.append(bar)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._current = BarOrderFlow()
        return bar

    @property
    def history(self) -> List[BarOrderFlow]:
        return list(self._history)

    @property
    def cumulative_delta(self) -> float:
        return self._cum_delta.cumulative

    def rolling_delta_mean(self) -> float:
        if not self._history:
            return 0.0
        return sum(b.delta_pct for b in self._history) / len(self._history)


# ---------------------------------------------------------------------------
# Historical tick-based order flow (from downloaded bid/ask tick data)
# ---------------------------------------------------------------------------

def compute_bar_orderflow_from_ticks(
    bid_ticks: List[Tuple[float, float]],   # [(timestamp, price), ...]
    ask_ticks: List[Tuple[float, float]],
    bar_open_ts: float,
    bar_close_ts: float,
) -> BarOrderFlow:
    """
    Compute BarOrderFlow from downloaded tick data for a historical bar.
    bid_ticks, ask_ticks: sorted by timestamp.
    Uses Lee-Ready rule with aligned bid/ask stream.
    """
    classifier = TickClassifier()
    bar = BarOrderFlow()

    # Merge bid+ask into unified stream, aligned by timestamp
    merged = _merge_tick_streams(bid_ticks, ask_ticks, bar_open_ts, bar_close_ts)

    for ts, price, bid, ask, vol in merged:
        is_buy = classifier.classify(price, bid, ask)
        if is_buy is True:
            bar.buy_volume += vol
            bar.buy_ticks += 1
        elif is_buy is False:
            bar.sell_volume += vol
            bar.sell_ticks += 1
        else:
            bar.unknown_ticks += 1

    bar.delta = bar.buy_volume - bar.sell_volume
    total = bar.total_volume
    bar.delta_pct = safe_div(bar.delta, total, fallback=0.0)
    total_ticks = bar.buy_ticks + bar.sell_ticks
    bar.tick_imbalance = safe_div(
        float(bar.buy_ticks - bar.sell_ticks), float(total_ticks), fallback=0.0
    )
    return bar


def _merge_tick_streams(
    bid_ticks: List[Tuple[float, float]],
    ask_ticks: List[Tuple[float, float]],
    start_ts: float,
    end_ts: float,
) -> List[Tuple[float, float, float, float, float]]:
    """
    Merge bid and ask tick streams into (timestamp, price, bid, ask, volume=1) tuples.
    Filters to [start_ts, end_ts]. Uses last known bid/ask for each tick.
    """
    result = []
    bi, ai = 0, 0
    last_bid = 0.0
    last_ask = 0.0

    # Filter to window
    bids = [(ts, p) for ts, p in bid_ticks if start_ts <= ts <= end_ts]
    asks = [(ts, p) for ts, p in ask_ticks if start_ts <= ts <= end_ts]

    all_ticks = sorted(
        [(ts, p, "bid") for ts, p in bids] + [(ts, p, "ask") for ts, p in asks],
        key=lambda x: x[0]
    )

    for ts, price, side in all_ticks:
        if side == "bid":
            last_bid = price
        else:
            last_ask = price
        mid = (last_bid + last_ask) / 2.0 if last_bid > 0 and last_ask > 0 else price
        result.append((ts, mid, last_bid, last_ask, 1.0))

    return result
