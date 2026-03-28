"""
Live bar builder — accumulates spot ticks into completed OHLCV bars.

Two modes:
  1. Tick-based: receives (timestamp_ms, bid, ask) from ProtoOASpotEvent
     and emits a Bar when the current period boundary rolls over.
  2. Trendbar-based: receives ProtoOATrendbar messages directly from
     ProtoOASubscribeLiveTrendbarReq (preferred when tick order flow is not needed).

Both modes call on_bar(bar) when a bar completes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from pipeline.cleaner import Bar
from core.logger import get_logger

logger = get_logger("bar_builder")

# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------

# Period length in seconds for each supported timeframe
_TF_SECONDS: dict = {
    "M1":  60,
    "M5":  300,
    "M15": 900,
    "M30": 1800,
    "H1":  3600,
    "H4":  14400,
    "D1":  86400,
}


def _bar_open_ts(ts_s: float, period_s: int) -> float:
    """Floor timestamp to the nearest period boundary."""
    return float(int(ts_s) // period_s * period_s)


# ---------------------------------------------------------------------------
# Tick-based bar builder
# ---------------------------------------------------------------------------

@dataclass
class _PartialBar:
    open_ts: float
    open:    float
    high:    float
    low:     float
    close:   float
    volume:  float = 0.0
    tick_count: int = 0


class TickBarBuilder:
    """
    Accumulates (timestamp_ms, bid, ask) ticks into OHLCV bars.

    Usage:
        builder = TickBarBuilder("M30", on_bar=my_callback)
        builder.on_tick(timestamp_ms, bid, ask)
    """

    def __init__(
        self,
        timeframe: str = "M30",
        on_bar: Optional[Callable[[Bar], None]] = None,
        lot_size: float = 1.0,
    ) -> None:
        period_s = _TF_SECONDS.get(timeframe.upper())
        if period_s is None:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}. Choose from {list(_TF_SECONDS)}")
        self._period_s = period_s
        self._tf = timeframe.upper()
        self._on_bar = on_bar
        self._lot_size = lot_size
        self._current: Optional[_PartialBar] = None

    def on_tick(self, timestamp_ms: int, bid: float, ask: float) -> Optional[Bar]:
        """
        Feed one tick. Returns the completed Bar if a period just rolled over,
        otherwise returns None.
        """
        mid = (bid + ask) * 0.5
        ts_s = timestamp_ms / 1000.0
        bar_ts = _bar_open_ts(ts_s, self._period_s)

        if self._current is None:
            self._current = _PartialBar(open_ts=bar_ts, open=mid, high=mid, low=mid, close=mid)
        elif bar_ts > self._current.open_ts:
            # Period rolled — emit the completed bar
            completed = self._emit()
            # Start new partial bar at new boundary
            self._current = _PartialBar(open_ts=bar_ts, open=mid, high=mid, low=mid, close=mid)
            self._current.tick_count = 1
            return completed
        else:
            # Update current bar
            self._current.high  = max(self._current.high,  mid)
            self._current.low   = min(self._current.low,   mid)
            self._current.close = mid

        self._current.volume += self._lot_size
        self._current.tick_count += 1
        return None

    def _emit(self) -> Bar:
        p = self._current
        bar = Bar(
            timestamp=p.open_ts,
            open=p.open,
            high=p.high,
            low=p.low,
            close=p.close,
            volume=p.volume,
        )
        logger.debug(f"[{self._tf}] bar close={bar.close:.5f} ts={bar.timestamp}")
        if self._on_bar is not None:
            self._on_bar(bar)
        return bar

    def flush(self) -> Optional[Bar]:
        """Emit whatever partial bar is in progress (e.g. on shutdown)."""
        if self._current is not None and self._current.tick_count > 0:
            return self._emit()
        return None


# ---------------------------------------------------------------------------
# Trendbar-based builder (receives completed bars from cTrader directly)
# ---------------------------------------------------------------------------

class TrendBarBuilder:
    """
    Wraps ProtoOATrendbar events from ProtoOASubscribeLiveTrendbarReq.
    Converts the protobuf trendbar into a pipeline Bar and calls on_bar.

    Usage:
        builder = TrendBarBuilder("M30", on_bar=my_callback)
        # In event loop:
        builder.on_trendbar_event(payload)  # payload = ProtoOASpotEvent or ProtoOALiveTrendbarEvent
    """

    def __init__(
        self,
        timeframe: str = "M30",
        on_bar: Optional[Callable[[Bar], None]] = None,
        digits: int = 5,
    ) -> None:
        self._tf = timeframe.upper()
        self._on_bar = on_bar
        self._scale = 10 ** digits      # cTrader prices are integer * 10^-digits

    def on_trendbar_event(self, payload: object) -> Optional[Bar]:
        """
        Parse a ProtoOATrendbar (nested inside ProtoOAGetTrendbarsRes or
        ProtoOASpotEvent trendbar sub-message) and emit a Bar.
        """
        trendbar = getattr(payload, "trendbar", None)
        if trendbar is None:
            # payload IS the trendbar
            trendbar = payload

        # cTrader stores price as integer scaled by 10^digits
        scale = self._scale
        try:
            low_price  = getattr(trendbar, "low",   None)
            delta_open = getattr(trendbar, "deltaOpen",  None)
            delta_high = getattr(trendbar, "deltaHigh",  None)
            delta_close= getattr(trendbar, "deltaClose", None)
            volume     = float(getattr(trendbar, "volume", 0))
            ts_utc_ms  = int(getattr(trendbar, "utcTimestampInMinutes", 0)) * 60 * 1000

            if low_price is None:
                return None

            low   = low_price  / scale
            open_ = (low_price + (delta_open  or 0)) / scale
            high  = (low_price + (delta_high  or 0)) / scale
            close = (low_price + (delta_close or 0)) / scale

            bar = Bar(
                timestamp=ts_utc_ms / 1000.0,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
            logger.debug(f"[{self._tf}] trendbar close={bar.close:.5f}")
            if self._on_bar is not None:
                self._on_bar(bar)
            return bar

        except Exception as exc:
            logger.warning(f"TrendBarBuilder.on_trendbar_event error: {exc}")
            return None
