"""
LiveSession — connects strategy to cTrader live execution.

Architecture:
  CTraderConnector (Twisted reactor thread)
      │
      ▼ event_queue (thread-safe)
  LiveSession.run_loop()
      │  on spot/trendbar events
      ▼
  TickBarBuilder / TrendBarBuilder
      │  on bar complete
      ▼
  TrendStrategy.on_bar()
      │  returns Signal | None
      ▼
  OrderManager.place_market_order()

Position tracking:
  - Open positions are mirrored from cTrader via reconcile on start and on
    each execution event.
  - SL/TP are set on the order itself (cTrader-side), so they fire even if
    the session disconnects.

The session never sends orders in a bar that is still forming — it only acts
on the closed bar passed to on_bar().
"""
from __future__ import annotations

import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from core.logger import get_logger
from core.validator import BrokerSpec
from pipeline.cleaner import Bar, clean_bars
from ctrader.order_manager import OrderManager, Position, AccountState
from live.bar_builder import TickBarBuilder, TrendBarBuilder

logger = get_logger("live_session")

try:
    from ctrader_open_api.messages import OpenApiMessages_pb2 as _api_msgs
    _CTRADER_AVAILABLE = True
except ImportError:
    _CTRADER_AVAILABLE = False
    _api_msgs = None


# ---------------------------------------------------------------------------
# Engine-shim helpers
# ---------------------------------------------------------------------------

class _TradeshimFromPosition:
    """Minimal Trade-compatible object for strategy._next_pyramid_level()."""
    __slots__ = ("symbol", "direction", "entry_price", "volume", "stop_loss", "take_profit")

    def __init__(self, pos: "LivePosition", symbol: str) -> None:
        self.symbol      = symbol
        self.direction   = pos.direction
        self.entry_price = pos.entry_price
        self.volume      = pos.volume
        self.stop_loss   = pos.stop_loss
        self.take_profit = pos.take_profit


# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    symbol: str
    timeframe: str = "M30"

    # Bar source: "trendbar" (preferred) or "tick" (needed for live order flow)
    bar_source: str = "trendbar"

    # Warm-up bars to pre-load from history before going live
    warmup_bars: int = 200

    # Position sizing
    lots_per_1000: float = 0.01     # lots per $1000 account equity
    max_open_trades: int = 3

    # Risk controls
    max_daily_loss_pct: float = 0.05    # stop trading if daily drawdown > 5%
    max_position_pct: float = 0.10      # max single position as % of equity

    # Reconnect
    reconnect_delay_s: float = 5.0
    max_reconnect_attempts: int = 10

    # Paper trade mode — log signals but don't send orders
    paper_mode: bool = False


# ---------------------------------------------------------------------------
# Open position tracker
# ---------------------------------------------------------------------------

@dataclass
class LivePosition:
    position_id: int
    symbol: str
    direction: int         # 1 = long, -1 = short
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    bar_opened: int = 0    # bar index at open
    is_open: bool = True


# ---------------------------------------------------------------------------
# LiveSession
# ---------------------------------------------------------------------------

class LiveSession:
    """
    Wires TrendStrategy to live cTrader execution.

    Usage:
        session = LiveSession(strategy, spec, connector, config)
        session.start()   # blocks; Ctrl-C or SIGTERM to stop
    """

    def __init__(
        self,
        strategy: Any,                  # TrendStrategy (not typed to avoid circular import)
        spec: BrokerSpec,
        connector: Any,                 # CTraderConnector
        config: SessionConfig,
        warmup_bars: Optional[List[Bar]] = None,
    ) -> None:
        self._strategy = strategy
        self._spec = spec
        self._conn = connector
        self._cfg = config
        self._order_manager = OrderManager(connector, connector.credentials.account_id)

        # Bar builder — switches mode based on config
        if config.bar_source == "trendbar":
            self._bar_builder: Any = TrendBarBuilder(
                timeframe=config.timeframe,
                on_bar=self._on_bar,
                digits=spec.digits,
            )
        else:
            self._bar_builder = TickBarBuilder(
                timeframe=config.timeframe,
                on_bar=self._on_bar,
                lot_size=spec.lot_size,
            )

        # State
        self._bar_index = 0
        self._open_positions: Dict[int, LivePosition] = {}
        self._daily_start_balance: float = 0.0
        self._running = False
        self._stop_event = threading.Event()
        self._event_queue: queue.Queue = queue.Queue(maxsize=10_000)

        # Pre-warm strategy with historical bars if provided
        if warmup_bars:
            self._warmup(warmup_bars)

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """
        Subscribe to live data and run the event loop (blocks until stopped).
        Call stop() or send SIGTERM/SIGINT to exit.
        """
        self._setup_signal_handlers()
        self._conn.set_event_queue(self._event_queue)
        self._running = True

        # Record daily-start balance for drawdown guard
        state = self._order_manager.get_account_state()
        self._daily_start_balance = state.balance or 10_000.0
        logger.info(f"[LiveSession] Starting — balance={state.balance:.2f} "
                    f"open_positions={len(state.positions)} "
                    f"paper_mode={self._cfg.paper_mode}")

        # Reconcile any pre-existing positions
        for pos in state.positions:
            if self._spec.symbol_id and pos.symbol_id == self._spec.symbol_id:
                self._register_position(pos)

        # Subscribe to live data
        symbol_id = self._conn.find_symbol_id(self._cfg.symbol)
        if symbol_id is None:
            raise RuntimeError(f"Symbol {self._cfg.symbol!r} not found on this account")

        self._subscribe_live_data(symbol_id)
        logger.info(f"[LiveSession] Subscribed to {self._cfg.symbol} ({self._cfg.bar_source})")

        # Main event loop
        try:
            self._run_loop()
        finally:
            self._cleanup(symbol_id)

    def stop(self) -> None:
        """Signal the session to stop cleanly after the current bar."""
        self._running = False
        self._stop_event.set()
        logger.info("[LiveSession] Stop requested")

    # -----------------------------------------------------------------------
    # Event loop
    # -----------------------------------------------------------------------

    def _run_loop(self) -> None:
        logger.info("[LiveSession] Event loop started")
        while self._running and not self._stop_event.is_set():
            try:
                event_name, payload = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._dispatch(event_name, payload)
            except Exception as exc:
                logger.error(f"[LiveSession] Error dispatching {event_name}: {exc}")

        logger.info("[LiveSession] Event loop stopped")

    def _dispatch(self, event_name: str, payload: Any) -> None:
        if event_name in ("ProtoOASpotEvent", "ProtoOALiveTrendbarEvent"):
            self._on_market_event(event_name, payload)
        elif event_name == "ProtoOAExecutionEvent":
            self._on_execution_event(payload)
        elif event_name == "ProtoOAOrderErrorEvent":
            err = getattr(payload, "errorCode", "?")
            desc = getattr(payload, "description", "")
            logger.error(f"[LiveSession] Order error {err}: {desc}")
        # Heartbeat, subscriptions, etc. are ignored

    # -----------------------------------------------------------------------
    # Market data
    # -----------------------------------------------------------------------

    def _on_market_event(self, event_name: str, payload: Any) -> None:
        if self._cfg.bar_source == "trendbar" or event_name == "ProtoOALiveTrendbarEvent":
            self._bar_builder.on_trendbar_event(payload)
        else:
            # Tick-based: extract bid/ask from ProtoOASpotEvent
            bid_raw = getattr(payload, "bid", 0)
            ask_raw = getattr(payload, "ask", 0)
            ts_ms   = int(getattr(payload, "timestamp", int(time.time() * 1000)))
            scale   = 10 ** self._spec.digits
            if bid_raw > 0 and ask_raw > 0:
                self._bar_builder.on_tick(ts_ms, bid_raw / scale, ask_raw / scale)

    def _on_bar(self, bar: Bar) -> None:
        """Called by bar_builder when a bar completes."""
        self._bar_index += 1

        # Daily drawdown guard
        if self._is_daily_loss_exceeded():
            logger.warning("[LiveSession] Daily loss limit reached — no new entries")
            return

        # Run strategy
        signal = self._strategy.on_bar(bar, self._bar_index, self)
        if signal is None:
            return

        # Execute signal
        self._execute_signal(signal, bar)

    # -----------------------------------------------------------------------
    # Order execution
    # -----------------------------------------------------------------------

    def _execute_signal(self, signal: Any, bar: Bar) -> None:
        """Translate a Signal into a cTrader market order."""
        # Hard limit on concurrent positions
        open_count = sum(1 for p in self._open_positions.values() if p.is_open)
        if open_count >= self._cfg.max_open_trades:
            logger.info(f"[LiveSession] Max open trades ({self._cfg.max_open_trades}) reached — skipping signal")
            return

        # Compute volume
        state = self._order_manager.get_account_state()
        equity = state.equity or self._daily_start_balance
        volume = self._spec.round_volume(
            self._cfg.lots_per_1000 * (equity / 1000.0)
        )
        if volume <= 0:
            return

        # Override volume from signal if provided
        if getattr(signal, "volume", None) and signal.volume > 0:
            volume = self._spec.round_volume(signal.volume)

        symbol_id = self._conn.find_symbol_id(self._cfg.symbol)
        if symbol_id is None:
            logger.error(f"[LiveSession] Cannot find symbol_id for {self._cfg.symbol}")
            return

        log_line = (
            f"[LiveSession] Signal: dir={'long' if signal.direction==1 else 'short'} "
            f"lots={volume} sl={signal.stop_loss:.5f} tp={signal.take_profit:.5f}"
        )

        if self._cfg.paper_mode:
            logger.info(f"[PAPER] {log_line}")
            return

        logger.info(log_line)
        position_id = self._order_manager.place_market_order(
            symbol_id=symbol_id,
            direction=signal.direction,
            volume_lots=volume,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            label="latitude",
        )

        if position_id is not None:
            self._open_positions[position_id] = LivePosition(
                position_id=position_id,
                symbol=self._cfg.symbol,
                direction=signal.direction,
                volume=volume,
                entry_price=bar.close,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                bar_opened=self._bar_index,
            )

    # -----------------------------------------------------------------------
    # Execution events (fills, SL/TP hits)
    # -----------------------------------------------------------------------

    def _on_execution_event(self, payload: Any) -> None:
        execution_type = getattr(payload, "executionType", None)
        deal = getattr(payload, "deal", None)
        if deal is None:
            return

        position_id = int(getattr(deal, "positionId", 0))
        deal_status = getattr(deal, "dealStatus", None)

        # Position closed (SL hit, TP hit, or manually closed)
        # executionType 4 = SWAP, 2 = CLOSE
        exec_name = {1: "FILL", 2: "CLOSE", 3: "CANCEL", 4: "SWAP"}.get(
            int(execution_type or 0), f"type={execution_type}"
        )

        if int(execution_type or 0) == 2 and position_id in self._open_positions:
            pos = self._open_positions.pop(position_id)
            close_price = float(getattr(deal, "executionPrice", 0))
            pnl_raw = float(getattr(deal, "grossProfit", 0)) / 100.0
            logger.info(
                f"[LiveSession] Position {position_id} closed ({exec_name}): "
                f"pnl={pnl_raw:.2f} close_price={close_price}"
            )
            # Feed runway predictor with actual MFE (approximated from close price)
            mfe = abs(close_price - pos.entry_price) / max(self._spec.tick_size, 1e-10)
            self._strategy.record_trade_outcome(mfe)

    # -----------------------------------------------------------------------
    # Subscriptions
    # -----------------------------------------------------------------------

    def _subscribe_live_data(self, symbol_id: int) -> None:
        if not _CTRADER_AVAILABLE:
            return

        if self._cfg.bar_source == "trendbar":
            # Subscribe to live trendbars
            from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs
            period_map = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
            period = period_map.get(self._cfg.timeframe, 30)

            req = msgs.ProtoOASubscribeLiveTrendbarReq()
            req.ctidTraderAccountId = self._conn.credentials.account_id
            req.symbolId = symbol_id
            req.period = period
            self._conn.send_and_wait(req)
        else:
            # Subscribe to spot ticks
            from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs
            req = msgs.ProtoOASubscribeSpotsReq()
            req.ctidTraderAccountId = self._conn.credentials.account_id
            req.symbolId.append(symbol_id)
            self._conn.send_and_wait(req)

    def _unsubscribe_live_data(self, symbol_id: int) -> None:
        if not _CTRADER_AVAILABLE:
            return
        try:
            from ctrader_open_api.messages import OpenApiMessages_pb2 as msgs
            if self._cfg.bar_source == "tick":
                req = msgs.ProtoOAUnsubscribeSpotsReq()
                req.ctidTraderAccountId = self._conn.credentials.account_id
                req.symbolId.append(symbol_id)
                self._conn.send_and_wait(req)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Warm-up
    # -----------------------------------------------------------------------

    def _warmup(self, bars: List[Bar]) -> None:
        """Pre-feed historical bars to warm up indicators without trading."""
        bars = clean_bars(bars)
        logger.info(f"[LiveSession] Warming up strategy on {len(bars)} historical bars")
        for i, bar in enumerate(bars):
            self._strategy.on_bar(bar, i, self)
        self._bar_index = len(bars)
        logger.info("[LiveSession] Warm-up complete")

    # -----------------------------------------------------------------------
    # Guards
    # -----------------------------------------------------------------------

    def _is_daily_loss_exceeded(self) -> bool:
        if self._daily_start_balance <= 0:
            return False
        try:
            state = self._order_manager.get_account_state()
            drawdown = (self._daily_start_balance - state.equity) / self._daily_start_balance
            return drawdown > self._cfg.max_daily_loss_pct
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _register_position(self, pos: Position) -> None:
        self._open_positions[pos.position_id] = LivePosition(
            position_id=pos.position_id,
            symbol=self._cfg.symbol,
            direction=pos.direction,
            volume=pos.volume,
            entry_price=pos.entry_price,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
        )

    def _cleanup(self, symbol_id: int) -> None:
        logger.info("[LiveSession] Cleaning up subscriptions")
        try:
            self._bar_builder.flush()
        except Exception:
            pass
        self._unsubscribe_live_data(symbol_id)

    def _setup_signal_handlers(self) -> None:
        def _handler(signum, frame):
            logger.info(f"[LiveSession] Signal {signum} received — stopping")
            self.stop()

        try:
            signal.signal(signal.SIGINT,  _handler)
            signal.signal(signal.SIGTERM, _handler)
        except (OSError, ValueError):
            # Non-main thread or restricted environment
            pass

    # -----------------------------------------------------------------------
    # Engine-compatibility shim
    # (strategy.on_bar receives 'engine' — it calls engine.balance / engine.open_trades)
    # -----------------------------------------------------------------------

    @property
    def balance(self) -> float:
        try:
            return self._order_manager.get_account_state().balance
        except Exception:
            return self._daily_start_balance

    @property
    def equity(self) -> float:
        try:
            return self._order_manager.get_account_state().equity
        except Exception:
            return self._daily_start_balance

    @property
    def open_trades(self) -> list:
        """
        Returns a list of shim objects compatible with the Trade interface
        that strategy._next_pyramid_level() inspects (.symbol, .direction).
        """
        return [
            _TradeshimFromPosition(pos, self._cfg.symbol)
            for pos in self._open_positions.values()
            if pos.is_open
        ]
