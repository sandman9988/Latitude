"""
Live DOM recorder — subscribes to ProtoOADepthEvent and records snapshots.
DOM is live-only in cTrader; no historical download exists.

Usage:
    recorder = DOMRecorder(conn, ["XAUUSD", "DE40"])
    recorder.start()
    # ... during live session, snapshots accumulate
    snapshots = recorder.get_snapshots("XAUUSD")
    recorder.stop()

Snapshots are also written to output_dir/dom/{symbol}_{date}.csv for
later feature computation.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from core.logger import get_logger
from core.numeric import is_valid_number
from pipeline.features.orderflow import (
    DOMSnapshot, DOMLevel, OrderBookImbalance
)

logger = get_logger("dom_recorder")


class DOMRecorder:
    """
    Subscribes to Depth of Market events for given symbols and records them.
    Runs in background thread consuming the connector's event queue.
    """

    def __init__(
        self,
        conn,
        symbols: List[str],
        output_dir: Optional[Path] = None,
        max_snapshots_per_symbol: int = 10000,
    ) -> None:
        self._conn = conn
        self._symbols = [s.strip().upper() for s in symbols]
        self._output_dir = output_dir
        self._max_snapshots = max_snapshots_per_symbol

        self._symbol_ids: Dict[str, int] = {}
        self._id_to_symbol: Dict[int, str] = {}
        self._snapshots: Dict[str, List[DOMSnapshot]] = {s: [] for s in self._symbols}
        self._obi_calculators: Dict[str, OrderBookImbalance] = {
            s: OrderBookImbalance(smoothing_period=10) for s in self._symbols
        }
        self._event_queue: queue.Queue = queue.Queue(maxsize=5000)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Live order book state per symbol
        self._book: Dict[str, Dict[int, DOMLevel]] = {s: {} for s in self._symbols}

    def start(self, timeout_s: float = 10.0) -> bool:
        """Subscribe to DOM events and start recording."""
        try:
            from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
        except ImportError:
            logger.error("ctrader_open_api not installed")
            return False

        account_id = self._conn.credentials.account_id

        # Resolve symbol IDs
        for sym in self._symbols:
            sid = self._conn.find_symbol_id(sym)
            if sid is None:
                logger.warning(f"Symbol not found: {sym}")
                continue
            self._symbol_ids[sym] = sid
            self._id_to_symbol[sid] = sym
            time.sleep(0.1)

        if not self._symbol_ids:
            logger.error("No symbols resolved for DOM recording")
            return False

        # Attach our queue to the connector
        self._conn.set_event_queue(self._event_queue)

        # Subscribe to depth quotes
        req = api_msgs.ProtoOASubscribeDepthQuotesReq()
        req.ctidTraderAccountId = account_id
        for sid in self._symbol_ids.values():
            req.symbolId.append(sid)
        resp = self._conn.send_and_wait(req, timeout_s=timeout_s)
        if resp is None or hasattr(resp, "errorCode"):
            logger.error("Failed to subscribe to depth quotes")
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._process_events,
            daemon=True,
            name="dom-recorder",
        )
        self._thread.start()
        logger.info(f"DOM recording started for: {list(self._symbol_ids.keys())}")
        return True

    def stop(self) -> None:
        """Unsubscribe from DOM and stop recording."""
        self._running = False
        try:
            from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
            req = api_msgs.ProtoOAUnsubscribeDepthQuotesReq()
            req.ctidTraderAccountId = self._conn.credentials.account_id
            for sid in self._symbol_ids.values():
                req.symbolId.append(sid)
            self._conn.send_and_wait(req, timeout_s=5.0)
        except Exception:
            pass
        self._conn.set_event_queue(None)
        if self._output_dir:
            self._flush_all_to_csv()
        logger.info("DOM recording stopped")

    def get_snapshots(self, symbol: str) -> List[DOMSnapshot]:
        return list(self._snapshots.get(symbol.upper(), []))

    def get_latest_obi(self, symbol: str) -> float:
        """Current smoothed Order Book Imbalance for symbol."""
        return self._obi_calculators.get(symbol.upper(), OrderBookImbalance()).value

    def _process_events(self) -> None:
        while self._running:
            try:
                event_type, payload = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event_type == "ProtoOADepthEvent":
                self._handle_depth_event(payload)

    def _handle_depth_event(self, payload) -> None:
        symbol_id = int(getattr(payload, "symbolId", 0) or 0)
        symbol = self._id_to_symbol.get(symbol_id)
        if symbol is None:
            return

        ts = time.time()
        new_quotes = getattr(payload, "newQuotes", [])
        deleted_ids = getattr(payload, "deletedQuotes", [])

        # Build scale from digits
        digits = self._conn.get_digits(symbol_id)
        scale = float(10 ** max(digits, 0))

        # Update book
        book = self._book[symbol]

        # Remove deleted
        for deleted in deleted_ids:
            q_id = int(getattr(deleted, "id", 0) or 0)
            book.pop(q_id, None)

        # Apply new/updated quotes
        for q in new_quotes:
            q_id = int(getattr(q, "id", 0) or 0)
            size = int(getattr(q, "size", 0) or 0)
            bid_raw = int(getattr(q, "bid", 0) or 0)
            ask_raw = int(getattr(q, "ask", 0) or 0)

            if size <= 0:
                book.pop(q_id, None)
                continue

            bid = float(bid_raw) / scale if bid_raw > 0 else 0.0
            ask = float(ask_raw) / scale if ask_raw > 0 else 0.0
            price = bid if bid > 0 else ask

            book[q_id] = DOMLevel(
                price=price,
                bid_size=float(size) if bid > 0 else 0.0,
                ask_size=float(size) if ask > 0 else 0.0,
            )

        # Build snapshot from current book state
        snapshot = DOMSnapshot(
            timestamp=ts,
            symbol=symbol,
            levels=list(book.values()),
        )

        # Compute mid from best bid/ask
        bids = [l for l in snapshot.levels if l.bid_size > 0]
        asks = [l for l in snapshot.levels if l.ask_size > 0]
        best_bid = max((l.price for l in bids), default=0.0)
        best_ask = min((l.price for l in asks), default=0.0)
        snapshot.mid_price = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0

        # Store snapshot
        snapshots = self._snapshots[symbol]
        snapshots.append(snapshot)
        if len(snapshots) > self._max_snapshots:
            snapshots.pop(0)

        # Update OBI calculator — bid/ask are sizes at each price level
        self._obi_calculators[symbol].update_from_event(
            [{"price": l.price, "bid": l.bid_size, "ask": l.ask_size}
             for l in snapshot.levels],
            timestamp=ts,
            symbol=symbol,
        )

    def _flush_all_to_csv(self) -> None:
        if not self._output_dir:
            return
        for symbol, snapshots in self._snapshots.items():
            if not snapshots:
                continue
            safe_sym = Path(symbol).name or symbol.replace("/", "_").replace("\\", "_")
            dom_dir = self._output_dir / safe_sym / "dom"
            dom_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
            path = dom_dir / f"{safe_sym}_dom_{date_str}.csv"
            with open(path, "w") as f:
                f.write("timestamp,mid_price,obi,total_bid_size,total_ask_size,n_levels\n")
                for snap in snapshots:
                    f.write(
                        f"{snap.timestamp:.3f},{snap.mid_price:.5f},"
                        f"{snap.order_book_imbalance:.4f},"
                        f"{snap.total_bid_size:.0f},{snap.total_ask_size:.0f},"
                        f"{len(snap.levels)}\n"
                    )
            logger.info(f"Flushed {len(snapshots)} DOM snapshots to {path}")
