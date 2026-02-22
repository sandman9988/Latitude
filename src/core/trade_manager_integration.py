"""
TradeManager Integration

Production-ready integration of TradeManager with CTraderFixApp
to centralize all order and position management.

Enhanced with defense-in-depth safety layer and state persistence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

try:
    import quickfix as fix
except ImportError:
    class fix:  # type: ignore[no-redef]
        class Message:
            pass

from src.core.trade_manager import Order, Side, TradeManager
from src.monitoring.trade_audit_logger import get_trade_audit_logger
from src.persistence.atomic_persistence import AtomicPersistence
from src.utils.safe_utils import utc_now

if TYPE_CHECKING:
    from src.core.ctrader_ddqn_paper import CTraderFixApp

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (replaces magic literals in comparisons)
# ---------------------------------------------------------------------------
_PRICE_NORMALIZE_TOLERANCE: float = 1e-10   # float tolerance for price-normalization diff check
_MIN_OPEN_POSITION_QTY: float = 0.0001      # minimum qty considered an open position

_MSG_RECOVERED_ENTRY = "[INTEGRATION] ✓ Notified DualPolicy of recovered entry @ %.5f dir=%d (MFE=%.4f MAE=%.4f)"


class TradeManagerIntegration:
    """
    Production TradeManager integration with CTraderFixApp.

    Centralizes order management with state persistence and recovery.
    """

    def __init__(self, app: CTraderFixApp):
        """
        Initialize TradeManager integration.

        Args:
            app: CTraderFixApp instance with active FIX sessions
        """
        self.app = app
        self.trade_manager: TradeManager | None = None

        # State persistence
        self.persistence = AtomicPersistence(base_dir="data/state")
        self.state_filename = f"trade_integration_{app.symbol}.json"

        # Trailing stop state
        self.trailing_stop_active = False
        self.trailing_stop_order: Order | None = None
        self.trailing_stop_distance_pct = 0.20  # 20 pips default (0.20%)
        self.highest_price_since_entry: float | None = None  # For LONG
        self.lowest_price_since_entry: float | None = None  # For SHORT
        self.entry_price: float | None = None
        self.position_direction: int = 0

        # Position recovery tracking
        self.position_recovered: bool = False

        # HEDGING MODE: Track positions by broker ticket numbers
        self.position_tickets: dict[str, str] = {}  # ticket → position_id (for backward compat)
        self.exit_order_to_ticket: dict[str, str] = {}  # clOrdID → ticket being closed

        # Central immutable audit log
        self.audit = get_trade_audit_logger()

        LOG.info("[INTEGRATION] TradeManager integration initialized")

    def has_any_open_positions(self) -> bool:
        """
        Check if there are ANY open positions (hedging mode aware).

        In hedging mode, net position can be 0 even with open positions
        (e.g., 0.1 LONG + 0.1 SHORT = 0 net). This checks for actual positions.

        Returns:
            bool: True if any positions are open (by ticket or MFE/MAE tracker)
        """
        # Check ticket-tracked positions
        LOG.info("[POSITION-CHECK] position_tickets=%s", self.position_tickets)
        if self.position_tickets:
            LOG.info("[POSITION-CHECK] Found %d ticket(s) - returning True", len(self.position_tickets))
            return True

        # Check MFE/MAE trackers
        tracker_count = len(self.app.mfe_mae_trackers) if hasattr(self.app, "mfe_mae_trackers") else 0
        LOG.info("[POSITION-CHECK] mfe_mae_trackers count=%d", tracker_count)
        if hasattr(self.app, "mfe_mae_trackers") and self.app.mfe_mae_trackers:
            LOG.info("[POSITION-CHECK] Found %d tracker(s) - returning True", len(self.app.mfe_mae_trackers))
            return True

        LOG.info("[INTEGRATION] No positions found - returning False")
        return False

    def cleanup_stale_trackers(self):
        """
        Remove all MFE/MAE trackers when no positions exist.

        CRITICAL: Prevents epsilon-greedy learning from being blocked by orphaned trackers.
        Called when position count is confirmed to be 0.
        """
        if not hasattr(self.app, "mfe_mae_trackers"):
            return

        if self.app.mfe_mae_trackers:
            tracker_count = len(self.app.mfe_mae_trackers)
            with self.app._tracker_lock:
                self.app.mfe_mae_trackers.clear()
            LOG.warning("[CLEANUP] Removed %d stale MFE/MAE trackers", tracker_count)

        if hasattr(self.app, "path_recorders") and self.app.path_recorders:
            recorder_count = len(self.app.path_recorders)
            with self.app._tracker_lock:
                self.app.path_recorders.clear()
            LOG.warning("[CLEANUP] Removed %d stale path recorders", recorder_count)

    def initialize_trade_manager(self) -> bool:
        """
        Initialize TradeManager once TRADE session is connected.

        Call this from CTraderFixApp.onCreate() after TRADE session login.

        Returns:
            True if initialization successful, False otherwise
        """
        if not self.app.trade_sid:
            LOG.warning("[INTEGRATION] Cannot initialize - TRADE session not connected")
            return False

        try:
            # Check if paper mode is enabled (check app attribute first, then env var)
            import os  # noqa: PLC0415

            paper_mode = getattr(self.app, "paper_mode", None)
            if paper_mode is None:
                paper_mode = os.environ.get("PAPER_MODE", "0") == "1"

            # Price callback for paper fills
            def get_current_price() -> tuple[float, float]:
                """Return (bid, ask) for paper fill simulation."""
                if hasattr(self.app, "order_book") and self.app.order_book:
                    bid, ask = self.app.order_book.best_bid_ask()
                    return bid or 0.0, ask or 0.0
                return 0.0, 0.0

            self.trade_manager = TradeManager(
                session_id=self.app.trade_sid,
                symbol_id=self.app.symbol_id,
                on_fill_callback=self.on_order_filled,
                on_reject_callback=self.on_order_rejected,
                max_pending_orders=10,
                paper_mode=paper_mode,
                get_price_callback=get_current_price if paper_mode else None,
            )

            if paper_mode:
                LOG.info("[INTEGRATION] PAPER MODE enabled - will simulate fills if broker doesn't respond")

            # Request initial positions
            self.trade_manager.request_positions()

            # Attempt to recover previous state (position + trailing stop)
            self.position_recovered = self._recover_state()

            # CRITICAL FIX: Clean up stale trackers only if NO positions exist (not hedged)
            # Check both net position AND position tickets to be hedge-aware
            if self.app.cur_pos == 0 and not self.position_tickets:
                LOG.info("[INTEGRATION] Starting with FLAT position (no tickets) - cleaning up stale trackers")
                self.cleanup_stale_trackers()
            elif self.position_tickets:
                LOG.info(
                    "[INTEGRATION] Starting with %d position ticket(s) - keeping trackers", len(self.position_tickets)
                )

            LOG.info(
                "[INTEGRATION] ✓ TradeManager initialized for symbol=%d (position_recovered=%s, paper_mode=%s)",
                self.app.symbol_id,
                self.position_recovered,
                paper_mode,
            )
            return True

        except Exception as e:
            LOG.error("[INTEGRATION] Failed to initialize TradeManager: %s", e, exc_info=True)
            return False

    def _notify_policy_on_exit(self, order: "Order", tracker_summary: dict, pnl: float, mfe: float, mae: float) -> dict:
        """Sync policy MFE/MAE values and call policy.on_exit. Returns updated tracker_summary."""
        if not (hasattr(self.app, "policy") and hasattr(self.app.policy, "on_exit")):
            return tracker_summary
        policy_mfe = getattr(self.app.policy, "mfe", 0.0) or 0.0
        policy_mae = getattr(self.app.policy, "mae", 0.0) or 0.0
        if policy_mfe > mfe:
            tracker_summary["mfe"] = policy_mfe
            mfe = policy_mfe
        if policy_mae > mae:
            tracker_summary["mae"] = policy_mae
        was_wtl = mfe > 0 and pnl < 0
        tracker_summary["winner_to_loser"] = was_wtl
        capture_ratio = (pnl / mfe) if mfe > 0 else 0.0
        self.app.policy.on_exit(
            exit_price=order.avg_price, capture_ratio=capture_ratio,
            was_wtl=was_wtl, entry_confidence=getattr(self.app, "entry_confidence", 0.5),
        )
        LOG.info(
            "[INTEGRATION] ✓ Notified DualPolicy of exit @ %.5f capture=%.2f%% wtl=%s mfe=%.4f mae=%.4f",
            order.avg_price, capture_ratio * 100, was_wtl, mfe, mae,
        )
        return tracker_summary

    def _find_and_process_exit_tracker(self, order: "Order", closed_ticket: str) -> tuple:
        """Scan mfe_mae_trackers for the closing ticket, calc P&L, notify policy, clean up.

        Returns (position_id_to_remove, tracker_summary) or (None, None).
        """
        if not hasattr(self.app, "mfe_mae_trackers"):
            return None, None

        position_id_to_remove = None
        with self.app._tracker_lock:
            for pos_id, tracker in self.app.mfe_mae_trackers.items():
                if getattr(tracker, "position_ticket", None) == closed_ticket:
                    position_id_to_remove = pos_id
                    break

        if not position_id_to_remove:
            return None, None

        tracker_summary = self.app.mfe_mae_trackers[position_id_to_remove].get_summary()
        mfe = tracker_summary.get("mfe", 0.0)
        mae = tracker_summary.get("mae", 0.0)
        entry_price = tracker_summary.get("entry_price", 0.0)
        direction = "LONG" if tracker.direction > 0 else "SHORT"

        if hasattr(self.app, "_calculate_position_pnl"):
            pnl = self.app._calculate_position_pnl(
                entry_price=entry_price, exit_price=order.avg_price,
                direction=direction, quantity=order.filled_qty,
            )
        else:
            direction_sign = 1 if tracker.direction > 0 else -1
            pnl = (order.avg_price - entry_price) * direction_sign * order.filled_qty * self.app.contract_size

        self.audit.log_position_close(
            position_id=position_id_to_remove, exit_price=order.avg_price,
            pnl=pnl, mfe=mfe, mae=mae, ticket=closed_ticket,
        )

        with self.app._tracker_lock:
            del self.app.mfe_mae_trackers[position_id_to_remove]

        if hasattr(self.app, "_pending_closes"):
            self.app._pending_closes.discard(position_id_to_remove)

        if hasattr(self.app, "path_recorders") and position_id_to_remove in self.app.path_recorders:
            with self.app._tracker_lock:
                del self.app.path_recorders[position_id_to_remove]

        LOG.info("[HEDGING] ✓ Closed position ticket %s (tracker=%s)", closed_ticket, position_id_to_remove)
        tracker_summary = self._notify_policy_on_exit(order, tracker_summary, pnl, mfe, mae)
        return position_id_to_remove, tracker_summary

    def _restore_policy_entry_after_hedge(self, position_id_removed: str) -> None:
        """Re-notify DualPolicy of the surviving position after a hedging close."""
        if not (self.app.cur_pos != 0 and hasattr(self.app, "mfe_mae_trackers")):
            return
        with self.app._tracker_lock:
            for pid, tracker in self.app.mfe_mae_trackers.items():
                if pid == position_id_removed:
                    continue
                if not (tracker.entry_price is not None and tracker.entry_price > 0):
                    continue
                if hasattr(self.app, "policy") and hasattr(self.app.policy, "on_entry"):
                    self.app.policy.on_entry(
                        direction=tracker.direction,
                        entry_price=tracker.entry_price,
                        entry_time=getattr(self.app, "trade_entry_time", None),
                    )
                    LOG.info(
                        "[INTEGRATION] ✓ Restored DualPolicy entry after hedging close: "
                        "price=%.5f dir=%d (tracker=%s)",
                        tracker.entry_price, tracker.direction, pid,
                    )
                break

    def _close_hedged_position(self, order: "Order", closed_ticket: str) -> None:
        """Handle fill for an exit (close) order in hedging mode."""
        position_id_to_remove, tracker_summary = self._find_and_process_exit_tracker(order, closed_ticket)

        if closed_ticket in self.position_tickets:
            del self.position_tickets[closed_ticket]

        self.app.cur_pos = self.trade_manager.get_position_direction(min_qty=self.app.qty * 0.5)
        self.position_direction = self.app.cur_pos
        LOG.info("[INTEGRATION] Position synced after exit: cur_pos=%d", self.app.cur_pos)

        if position_id_to_remove:
            self._restore_policy_entry_after_hedge(position_id_to_remove)

        if hasattr(self.app, "_process_trade_completion") and tracker_summary:
            self.app._process_trade_completion(tracker_summary, order.avg_price)

        self._persist_state()

    def _start_entry_tracker_and_path(self, order: "Order", position_id: str) -> None:
        """Create MFE/MAE tracker, path recorder, and notify DualPolicy for a new entry fill."""
        if hasattr(self.app, "mfe_mae_trackers") and order.avg_price > 0:
            direction = 1 if order.side == Side.BUY else -1
            with self.app._tracker_lock:
                if position_id not in self.app.mfe_mae_trackers:
                    from src.core.ctrader_ddqn_paper import MFEMAETracker  # noqa: PLC0415
                    self.app.mfe_mae_trackers[position_id] = MFEMAETracker(position_id)
                    self.app.mfe_mae_trackers[position_id].position_ticket = order.position_ticket
                self.app.mfe_mae_trackers[position_id].start_tracking(order.avg_price, direction)
            self.app.trade_entry_time = order.filled_at
            self.audit.log_position_open(
                position_id=position_id,
                direction="LONG" if direction > 0 else "SHORT",
                quantity=order.filled_qty, entry_price=order.avg_price,
                ticket=order.position_ticket or "UNKNOWN",
            )
            if hasattr(self.app, "policy") and hasattr(self.app.policy, "on_entry"):
                self.app.policy.on_entry(direction=direction, entry_price=order.avg_price, entry_time=order.filled_at)

        if hasattr(self.app, "path_recorders") and order.filled_at:
            direction = 1 if order.side == Side.BUY else -1
            if position_id not in self.app.path_recorders:
                from src.core.ctrader_ddqn_paper import PathRecorder  # noqa: PLC0415
                self.app.path_recorders[position_id] = PathRecorder(position_id)
            self.app.path_recorders[position_id].start_recording(order.filled_at, order.avg_price, direction)

    def _open_hedged_position(self, order: "Order", position_id: str) -> None:
        """Handle fill for an entry (open) order in hedging mode."""
        if order.avg_price > 0:
            self.entry_price = order.avg_price

        self._start_entry_tracker_and_path(order, position_id)

        if hasattr(self.app, "activity_monitor"):
            self.app.activity_monitor.on_trade_executed(order.filled_at)

        if hasattr(self.app, "policy") and hasattr(self.app.policy, "trigger"):
            self.app.entry_state = getattr(self.app.policy.trigger, "last_state", None)
            self.app.entry_action = 1 if order.side == Side.BUY else 2

        self.app.cur_pos = self.trade_manager.get_position_direction(min_qty=self.app.qty * 0.5)
        self.position_direction = self.app.cur_pos
        LOG.info(
            "[INTEGRATION] Position synced from fill: cur_pos=%d net_qty=%.6f",
            self.app.cur_pos, self.trade_manager.position.net_qty,
        )
        self._persist_state()

        pending_closes = getattr(self.app, "_pending_closes", set())
        if pending_closes:
            LOG.debug("[VALIDATION] Skipped during hedging transition (pending_closes=%d)", len(pending_closes))
        else:
            self._validate_position_after_fill(self.app.cur_pos, order)

        self.trade_manager.request_positions()

        if hasattr(self.app, "param_manager"):
            distance = self.app.param_manager.get(
                self.app.symbol, "trailing_stop_distance_pct",
                timeframe=self.app.timeframe_label, broker="default", default=0.20,
            )
            self.enable_trailing_stop(distance_pct=float(distance))

    def on_order_filled(self, order: Order):
        """
        Callback when order fills.

        FIX P0-2: Proper callback-based state updates (no race conditions)
        FIX P0-6: Position validation after order execution
        HEDGING MODE: Track by broker tickets, not ClOrdIDs
        """
        LOG.info(
            "[INTEGRATION] Order filled: %s qty=%.6f @%.5f clOrdID=%s ticket=%s",
            order.side.name, order.filled_qty, order.avg_price,
            order.clord_id, order.position_ticket or "N/A",
        )

        if order.clord_id in self.exit_order_to_ticket:
            closed_ticket = self.exit_order_to_ticket.pop(order.clord_id)
            self._close_hedged_position(order, closed_ticket)
            return

        if not order.position_ticket:
            LOG.warning("[HEDGING] No position ticket for order %s - using ClOrdID fallback", order.clord_id)
            position_id = f"{self.app.symbol_id}_{order.clord_id}"
        else:
            position_id = f"{self.app.symbol_id}_ticket_{order.position_ticket}"
            self.position_tickets[order.position_ticket] = position_id

        self._open_hedged_position(order, position_id)

    def on_order_rejected(self, order: Order):
        """
        Callback when order is rejected.

        FIX P0-4: State already cleared in main bot's on_exec_report
        """
        LOG.warning(
            "[INTEGRATION] Order rejected: %s - %s",
            order.clord_id,
            order.reject_reason or "Unknown reason",
        )

        # State cleanup already handled in ctrader_ddqn_paper.py on_exec_report
        # Log for monitoring
        if hasattr(self.app, "performance"):
            # Could track rejection statistics here
            pass

    def enter_position(self, side: int, quantity: float) -> bool:
        """
        Enter position using TradeManager with validation.

        Args:
            side: 1 for LONG, 2 for SHORT (agent action values)
            quantity: Order quantity

        Returns:
            True if order submitted, False otherwise
        """
        if not self.trade_manager:
            LOG.error("[INTEGRATION] TradeManager not initialized")
            return False

        # Basic quantity sanity check
        if quantity is None or quantity <= 0:
            LOG.error("[INTEGRATION] Invalid quantity=%.5f — rejecting order", quantity or 0)
            return False

        safe_qty = float(quantity)

        fix_side = Side.BUY if side == 1 else Side.SELL

        try:
            order = self.trade_manager.submit_market_order(
                side=fix_side,
                quantity=safe_qty,
                tag_prefix="ENTRY",
            )

            if order:
                LOG.info("[INTEGRATION] ✓ Order submitted: %s %.6f lots", fix_side.name, safe_qty)
                return True
            else:
                LOG.error("[INTEGRATION] ❌ Order submission returned None")
                return False

        except Exception as e:
            LOG.error("[INTEGRATION] ❌ Order submission exception: %s", e, exc_info=True)
            return False

    def enable_trailing_stop(self, distance_pct: float = 0.20) -> bool:
        """
        Enable trailing stop for current position.

        Args:
            distance_pct: Stop distance as % of price (e.g., 0.20 = 20 pips)

        Returns:
            True if trailing stop enabled, False otherwise
        """
        if not self.trade_manager:
            LOG.error("[TRAILING-STOP] TradeManager not initialized")
            return False

        pos_dir = self.get_current_position()
        if pos_dir == 0:
            LOG.warning("[TRAILING-STOP] No position to protect")
            return False

        self.trailing_stop_distance_pct = distance_pct
        self.position_direction = pos_dir
        self.trailing_stop_active = True

        # Initialize price tracking
        if hasattr(self.app, "best_bid") and hasattr(self.app, "best_ask"):
            mid = (self.app.best_bid + self.app.best_ask) / 2.0 if self.app.best_bid and self.app.best_ask else None
            if mid:
                self.highest_price_since_entry = mid if pos_dir == 1 else None
                self.lowest_price_since_entry = mid if pos_dir == -1 else None

        if hasattr(self.app, "mfe_mae_tracker"):
            summary = self.app.mfe_mae_tracker.get_summary()
            self.entry_price = summary.get("entry_price", mid)

        LOG.info(
            "[TRAILING-STOP] ✓ Enabled for %s position (distance=%.2f%%, entry=%.5f)",
            "LONG" if pos_dir == 1 else "SHORT",
            distance_pct * 100,
            self.entry_price or 0,
        )
        return True

    def update_trailing_stop(self, current_price: float):
        """
        Update trailing stop based on current price.
        Called by HarvesterAgent on each bar/tick.

        Logic:
        - LONG: Stop trails up as price rises, never moves down
        - SHORT: Stop trails down as price falls, never moves up

        Args:
            current_price: Current market price (mid/bid/ask)
        """
        if not self.trailing_stop_active or not self.trade_manager:
            return

        if self.position_direction == 0:
            LOG.debug("[TRAILING-STOP] Position closed, disabling")
            self.trailing_stop_active = False
            self._persist_state()
            return

        # Defensive: Validate price
        import math  # noqa: PLC0415

        if not (math.isfinite(current_price) and current_price > 0):
            LOG.error("[TRAILING-STOP] Invalid price: %s", current_price)
            return

        # Track highest/lowest price
        if self.position_direction == 1:  # LONG position
            if self.highest_price_since_entry is None:
                self.highest_price_since_entry = current_price
                self._persist_state()
            elif current_price > self.highest_price_since_entry:
                old_high = self.highest_price_since_entry
                self.highest_price_since_entry = current_price

                # Calculate new stop level (trails upward)
                new_stop = current_price * (1.0 - self.trailing_stop_distance_pct / 100.0)

                LOG.info(
                    "[TRAILING-STOP] LONG: New high %.5f (was %.5f) → Stop moved to %.5f",
                    current_price,
                    old_high,
                    new_stop,
                )

                # Submit/modify stop order via TradeManager
                self._submit_stop_order(new_stop)
                self._persist_state()

        elif self.position_direction == -1:  # SHORT position
            if self.lowest_price_since_entry is None:
                self.lowest_price_since_entry = current_price
                self._persist_state()
            elif current_price < self.lowest_price_since_entry:
                old_low = self.lowest_price_since_entry
                self.lowest_price_since_entry = current_price

                # Calculate new stop level (trails downward)
                new_stop = current_price * (1.0 + self.trailing_stop_distance_pct / 100.0)

                LOG.info(
                    "[TRAILING-STOP] SHORT: New low %.5f (was %.5f) → Stop moved to %.5f",
                    current_price,
                    old_low,
                    new_stop,
                )

                # Submit/modify stop order via TradeManager
                self._submit_stop_order(new_stop)

    def _submit_stop_order(self, stop_price: float):
        """
        Submit or modify stop loss order at specified price.

        Args:
            stop_price: Stop loss trigger price (will be normalized to broker digits)
        """
        if not self.trade_manager:
            return

        # Normalize price to broker's digit precision (symbol-agnostic)
        if hasattr(self.app, "friction_calculator"):
            original_price = stop_price
            stop_price = self.app.friction_calculator.normalize_price(stop_price)
            if abs(original_price - stop_price) > _PRICE_NORMALIZE_TOLERANCE:
                LOG.debug(
                    "[TRAILING-STOP] Price normalized: %.8f → %.8f (digits=%d)",
                    original_price,
                    stop_price,
                    self.app.friction_calculator.costs.digits,
                )

        # If stop order exists, modify it
        if self.trailing_stop_order and self.trailing_stop_order.clord_id:
            try:
                self.trade_manager.modify_order(
                    clord_id=self.trailing_stop_order.clord_id,
                    new_price=stop_price,
                )
                LOG.debug("[TRAILING-STOP] Modified stop order to %.5f", stop_price)
            except Exception as e:
                LOG.error("[TRAILING-STOP] Failed to modify stop: %s", e)
                # Retry by submitting new stop
                self.trailing_stop_order = None
                self._submit_stop_order(stop_price)
        else:
            # Submit new stop order
            try:
                side = Side.SELL if self.position_direction == 1 else Side.BUY
                self.trailing_stop_order = self.trade_manager.submit_limit_order(
                    side=side,
                    quantity=self.app.qty,
                    price=stop_price,
                    tag_prefix="STOP",
                )
                LOG.info("[TRAILING-STOP] Submitted stop order @ %.5f", stop_price)
            except Exception as e:
                LOG.error("[TRAILING-STOP] Failed to submit stop: %s", e)

    def disable_trailing_stop(self):
        """Disable trailing stop and cancel stop order."""
        if self.trailing_stop_order and self.trade_manager:
            try:
                self.trade_manager.cancel_order(self.trailing_stop_order.clord_id)
                LOG.info("[TRAILING-STOP] Cancelled stop order")
            except Exception as e:
                LOG.warning("[TRAILING-STOP] Failed to cancel stop: %s", e)

        self.trailing_stop_active = False
        self.trailing_stop_order = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        self.entry_price = None
        self.position_direction = 0

    def close_position(self, position_id: str | None = None, reason: str = "MANUAL") -> bool:  # noqa: PLR0911
        """
        Close a specific position by ID or ticket.

        Args:
            position_id: Position tracker ID (e.g., "10028_ticket_12345678")
                        If None, closes net position.
            reason: Reason for closing (for logging)

        Returns:
            True if order submitted, False otherwise
        """
        if not self.trade_manager:
            LOG.error("[INTEGRATION] TradeManager not initialized")
            return False

        # If position_id specified, close that specific position
        if position_id and hasattr(self.app, "mfe_mae_trackers"):
            tracker = self.app.mfe_mae_trackers.get(position_id)
            if not tracker:
                LOG.warning("[INTEGRATION] Position %s not found in trackers", position_id)
                return False

            direction = getattr(tracker, "direction", None)
            if direction is None or direction == 0:
                LOG.warning("[INTEGRATION] Position %s has invalid direction", position_id)
                return False

            # Get broker ticket for this position
            ticket = getattr(tracker, "position_ticket", None)
            if not ticket:
                LOG.warning("[INTEGRATION] Position %s has no broker ticket", position_id)
                return False

            # Submit opposite order to close
            exit_side = Side.SELL if direction > 0 else Side.BUY
            order = self.trade_manager.submit_market_order(
                side=exit_side,
                quantity=self.app.qty,
                tag_prefix=f"EXIT_{reason}",
                position_ticket=ticket,  # HEDGING MODE FIX: Close specific position by ticket
            )

            if order:
                # HEDGING MODE: Map exit order to broker ticket being closed
                self.exit_order_to_ticket[order.clord_id] = ticket
                LOG.info(
                    "[INTEGRATION] Closing position ticket=%s: %s %.6f @ market (clOrdID=%s reason=%s)",
                    ticket,
                    exit_side.name,
                    self.app.qty,
                    order.clord_id,
                    reason,
                )
            return order is not None

        # Otherwise close net position (legacy behavior)
        pos_dir = self.trade_manager.get_position_direction(min_qty=self.app.qty * 0.5)
        if pos_dir == 0:
            LOG.warning("[INTEGRATION] No net position to exit")
            return False

        exit_side = Side.SELL if pos_dir > 0 else Side.BUY
        order = self.trade_manager.submit_market_order(
            side=exit_side,
            quantity=self.app.qty,
            tag_prefix=f"EXIT_{reason}",
        )

        if order:
            LOG.info("[INTEGRATION] Closing net position: %s %.6f (reason=%s)", exit_side.name, self.app.qty, reason)
        return order is not None

    def exit_position(self) -> bool:
        """Exit position using TradeManager (legacy method).

        Returns:
            True if order submitted, False otherwise
        """
        return self.close_position(position_id=None, reason="LEGACY_EXIT")

    def handle_execution_report(self, msg: fix.Message):
        """
        Route ExecutionReport to TradeManager.

        Replace CTraderFixApp.on_exec_report() with this.
        """
        if self.trade_manager:
            self.trade_manager.on_execution_report(msg)
        else:
            LOG.warning("[INTEGRATION] ExecutionReport received but TradeManager not initialized")

    def handle_position_report(self, msg: fix.Message):
        """
        Route PositionReport to TradeManager.

        Replace CTraderFixApp.on_position_report() with this.
        """
        if self.trade_manager:
            self.trade_manager.on_position_report(msg)

            # Update app's cur_pos for backward compatibility
            self.app.cur_pos = self.trade_manager.get_position_direction(min_qty=self.app.qty * 0.5)

            LOG.info(
                "[INTEGRATION] Position synced: cur_pos=%d net_qty=%.6f",
                self.app.cur_pos,
                self.trade_manager.position.net_qty,
            )

            # CRITICAL FIX: Clean up stale trackers when position is FLAT
            if self.app.cur_pos == 0 and abs(self.trade_manager.position.net_qty) < _MIN_OPEN_POSITION_QTY:
                self.cleanup_stale_trackers()
        else:
            LOG.warning("[INTEGRATION] PositionReport received but TradeManager not initialized")

    def get_current_position(self) -> int:
        """
        Get current position direction.

        Returns:
            1 for LONG, -1 for SHORT, 0 for FLAT
        """
        if not self.trade_manager:
            return 0
        return self.trade_manager.get_position_direction(min_qty=self.app.qty * 0.5)

    def get_statistics(self) -> dict:
        """Get TradeManager statistics"""
        if self.trade_manager:
            return self.trade_manager.get_statistics()
        return {}

    def _persist_state(self):
        """Persist current position and trailing stop state for crash recovery"""
        try:
            # Include position data for recovery
            position_data = None
            if self.trade_manager and self.trade_manager.position:
                position_data = self.trade_manager.position.to_dict()

            # MULTI-POSITION: Persist all active trackers
            active_trackers = {}
            if hasattr(self.app, "mfe_mae_trackers"):
                for pos_id, tracker in self.app.mfe_mae_trackers.items():
                    entry_price = getattr(tracker, "entry_price", None)
                    direction = getattr(tracker, "direction", None)
                    # Only persist initialized trackers
                    if entry_price and entry_price > 0 and direction and direction != 0:
                        active_trackers[pos_id] = {
                            "entry_price": round(entry_price, 8),
                            "direction": direction,
                            "mfe": round(getattr(tracker, "mfe", 0.0), 8),
                            "mae": round(getattr(tracker, "mae", 0.0), 8),
                        }

            # HEDGING MODE: Persist broker ticket mappings
            position_tickets = {}
            if hasattr(self.app, "mfe_mae_trackers"):
                for ticket, pos_id in self.position_tickets.items():
                    tracker = self.app.mfe_mae_trackers.get(pos_id)
                    if tracker:
                        entry_price = getattr(tracker, "entry_price", None)
                        direction = getattr(tracker, "direction", None)
                        if entry_price and entry_price > 0 and direction and direction != 0:
                            position_tickets[ticket] = {
                                "position_id": pos_id,
                                "entry_price": round(entry_price, 8),
                                "direction": direction,
                                "quantity": round(self.app.qty, 8),
                                "entry_time": getattr(tracker, "entry_time", utc_now()).isoformat(),
                            }

            state = {
                "trailing_stop_active": self.trailing_stop_active,
                "trailing_stop_distance_pct": round(self.trailing_stop_distance_pct, 8),
                "highest_price_since_entry": (
                    round(self.highest_price_since_entry, 8) if self.highest_price_since_entry else None
                ),
                "lowest_price_since_entry": (
                    round(self.lowest_price_since_entry, 8) if self.lowest_price_since_entry else None
                ),
                "entry_price": round(self.entry_price, 8) if self.entry_price else None,
                "position_direction": self.position_direction,
                "symbol_id": self.app.symbol_id,
                "position": position_data,
                "active_trackers": active_trackers,  # Legacy tracker data
                "position_tickets": position_tickets,  # NEW: Broker ticket → position mapping
                "persisted_at": utc_now().isoformat(),
            }
            self.persistence.save_json(state, self.state_filename, create_backup=True)
            LOG.info(
                "[INTEGRATION] 💾 Position persisted: net=%.6f trackers=%d tickets=%d",
                float(position_data.get("net_qty", 0)) if position_data else 0.0,
                len(active_trackers),
                len(position_tickets),
            )
        except Exception as e:
            LOG.error("[INTEGRATION] Failed to persist state: %s", e, exc_info=True)

    def _recover_state(self) -> bool:  # noqa: PLR0912, PLR0915
        """Recover position and trailing stop state after crash/restart.

        Returns:
            True if position was recovered, False otherwise
        """
        try:
            LOG.info("[INTEGRATION] Attempting to load state from: %s", self.state_filename)
            state = self.persistence.load_json(self.state_filename)

            if not state:
                LOG.info("[INTEGRATION] No previous state to recover")
                return False

            LOG.info(
                "[INTEGRATION] State loaded: has_position=%s trackers=%d",
                bool(state.get("position")),
                len(state.get("active_trackers", {})),
            )
            if not state:
                LOG.info("[INTEGRATION] No previous state to recover")
                return False

            # Validate symbol matches
            if state.get("symbol_id") != self.app.symbol_id:
                LOG.warning(
                    "[INTEGRATION] State symbol mismatch: %s vs %s, skipping recovery",
                    state.get("symbol_id"),
                    self.app.symbol_id,
                )
                return False

            # CRITICAL: Restore position from persisted state
            position_data = state.get("position")
            position_recovered = False
            if position_data and self.trade_manager:
                from src.core.trade_manager import Position  # noqa: PLC0415

                recovered_pos = Position.from_dict(position_data)
                # HEDGING MODE: Check if ANY real positions exist (not just net_qty)
                # BUT: For recovery purposes, only restore if we have active trackers or tickets
                # (net_qty=0 with no trackers/tickets means positions were closed)
                has_position = abs(recovered_pos.long_qty) > _MIN_OPEN_POSITION_QTY or abs(recovered_pos.short_qty) > _MIN_OPEN_POSITION_QTY

                # Check if we have active trackers or tickets (indicates real open positions)
                has_trackers = len(state.get("active_trackers", {})) > 0
                has_tickets = len(state.get("position_tickets", {})) > 0

                # Only treat as position if we have trackers/tickets (not just non-zero qty)
                if has_position and (has_trackers or has_tickets):
                    self.trade_manager.position = recovered_pos
                    if recovered_pos.net_qty > 0:
                        self.app.cur_pos = 1
                    elif recovered_pos.net_qty < 0:
                        self.app.cur_pos = -1
                    else:
                        self.app.cur_pos = 0
                    position_recovered = True
                    if self.app.cur_pos == 1:
                        _recovered_dir = "LONG"
                    elif self.app.cur_pos == -1:
                        _recovered_dir = "SHORT"
                    elif has_position:
                        _recovered_dir = "HEDGED"
                    else:
                        _recovered_dir = "FLAT"
                    LOG.info(
                        "[INTEGRATION] 🔄 POSITION RECOVERED: long=%.6f short=%.6f net=%.6f direction=%s trackers=%d tickets=%d (persisted_at=%s)",
                        recovered_pos.long_qty,
                        recovered_pos.short_qty,
                        recovered_pos.net_qty,
                        _recovered_dir,
                        len(state.get("active_trackers", {})),
                        len(state.get("position_tickets", {})),
                        state.get("persisted_at", "unknown"),
                    )
                elif has_position:
                    # Position quantities exist but no trackers/tickets = stale data, positions were closed
                    LOG.warning(
                        "[INTEGRATION] ⚠️ Ignoring stale position data (long=%.6f short=%.6f net=%.6f) - no active trackers/tickets",
                        recovered_pos.long_qty,
                        recovered_pos.short_qty,
                        recovered_pos.net_qty,
                    )
                    position_recovered = False

            # Restore trailing stop state
            self.trailing_stop_active = state.get("trailing_stop_active", False)
            self.trailing_stop_distance_pct = state.get("trailing_stop_distance_pct", 0.20)
            self.highest_price_since_entry = state.get("highest_price_since_entry")
            self.lowest_price_since_entry = state.get("lowest_price_since_entry")
            self.entry_price = state.get("entry_price")
            self.position_direction = state.get("position_direction", 0)

            # MULTI-POSITION: Restore all active trackers from state
            active_trackers = state.get("active_trackers", {})
            if active_trackers and hasattr(self.app, "mfe_mae_trackers"):
                from src.core.ctrader_ddqn_paper import MFEMAETracker  # noqa: PLC0415

                for pos_id, tracker_data in active_trackers.items():
                    try:
                        # Defensive: Validate tracker_data structure
                        if not isinstance(tracker_data, dict):
                            LOG.error("[RECOVERY] Invalid tracker_data type for %s - skipping", pos_id)
                            continue

                        # Defensive: Validate required fields
                        entry_price = tracker_data.get("entry_price")
                        direction = tracker_data.get("direction")

                        if entry_price is None or entry_price <= 0:
                            LOG.error("[RECOVERY] Invalid entry_price for %s - skipping tracker", pos_id)
                            continue

                        if direction not in (1, -1):
                            LOG.warning(
                                "[RECOVERY] Invalid direction=%s for %s - defaulting to LONG", direction, pos_id
                            )
                            direction = 1

                        # Create tracker if doesn't exist
                        if pos_id not in self.app.mfe_mae_trackers:
                            self.app.mfe_mae_trackers[pos_id] = MFEMAETracker(pos_id)

                        # Restore tracker state
                        self.app.mfe_mae_trackers[pos_id].start_tracking(entry_price, direction)

                        # Restore MFE/MAE and best/worst if available
                        mfe_val = float(tracker_data.get("mfe", 0.0))
                        mae_val = float(tracker_data.get("mae", 0.0))
                        self.app.mfe_mae_trackers[pos_id].mfe = mfe_val
                        self.app.mfe_mae_trackers[pos_id].mae = mae_val
                        # Critical: Also restore best_profit/worst_loss for update() to work correctly
                        self.app.mfe_mae_trackers[pos_id].best_profit = mfe_val  # MFE is the best profit
                        self.app.mfe_mae_trackers[pos_id].worst_loss = (
                            -mae_val
                        )  # MAE is stored as positive, worst_loss as negative
                    except Exception as e:
                        LOG.error("[RECOVERY] Error restoring tracker %s: %s", pos_id, e, exc_info=True)
                        continue

                    LOG.info(
                        "[MULTI-POS] ✓ Recovered tracker %s: entry=%.5f dir=%d MFE=%.4f MAE=%.4f",
                        pos_id,
                        tracker_data["entry_price"],
                        tracker_data["direction"],
                        tracker_data.get("mfe", 0.0),
                        tracker_data.get("mae", 0.0),
                    )

            # HEDGING MODE: Restore broker ticket mappings (new format)
            position_tickets = state.get("position_tickets", {})
            if position_tickets and hasattr(self.app, "mfe_mae_trackers"):
                from src.core.ctrader_ddqn_paper import MFEMAETracker  # noqa: PLC0415

                for ticket, ticket_data in position_tickets.items():
                    position_id = ticket_data["position_id"]

                    # Create tracker if doesn't exist
                    tracker_is_new = position_id not in self.app.mfe_mae_trackers
                    if tracker_is_new:
                        self.app.mfe_mae_trackers[position_id] = MFEMAETracker(position_id)

                    # Restore tracker state (only start_tracking if newly created)
                    tracker = self.app.mfe_mae_trackers[position_id]
                    if tracker_is_new:
                        # New tracker - initialize with entry data
                        tracker.start_tracking(ticket_data["entry_price"], ticket_data["direction"])
                    # Always restore ticket reference
                    tracker.position_ticket = ticket  # Critical: restore broker ticket reference

                    # Restore ticket mapping
                    self.position_tickets[ticket] = position_id

                    LOG.info(
                        "[HEDGING] ✓ Recovered position ticket=%s: pos_id=%s entry=%.5f dir=%d qty=%.6f",
                        ticket,
                        position_id,
                        ticket_data["entry_price"],
                        ticket_data["direction"],
                        ticket_data["quantity"],
                    )

                # CRITICAL: Notify DualPolicy of the LAST recovered position
                # (DualPolicy still manages one position at a time, but all are monitored)
                if position_tickets and hasattr(self.app, "policy"):
                    # Use the last ticket (most recent position)
                    last_ticket_data = list(position_tickets.values())[-1]
                    import datetime as dt  # noqa: PLC0415

                    # Get MFE/MAE from tracker if available
                    position_id = last_ticket_data.get("position_id")
                    mfe, mae, ticks_held = 0.0, 0.0, 0
                    if position_id and hasattr(self.app, "mfe_mae_trackers"):
                        tracker = self.app.mfe_mae_trackers.get(position_id)
                        if tracker:
                            mfe = getattr(tracker, "mfe", 0.0)
                            mae = getattr(tracker, "mae", 0.0)
                            # Estimate ticks_held from time if not stored
                            # (For now, leave at 0 - will build up from current time)

                    # Use on_recovery() to preserve MFE/MAE
                    if hasattr(self.app.policy, "on_recovery"):
                        self.app.policy.on_recovery(
                            direction=last_ticket_data["direction"],
                            entry_price=last_ticket_data["entry_price"],
                            entry_time=dt.datetime.now(dt.UTC),
                            mfe=mfe,
                            mae=mae,
                            ticks_held=ticks_held,
                        )
                    else:
                        # Fallback for older code
                        self.app.policy.on_entry(
                            direction=last_ticket_data["direction"],
                            entry_price=last_ticket_data["entry_price"],
                            entry_time=dt.datetime.now(dt.UTC),
                        )
                    LOG.info(
                        _MSG_RECOVERED_ENTRY,
                        last_ticket_data["entry_price"],
                        last_ticket_data["direction"],
                        mfe,
                        mae,
                    )

            # Legacy fallback: Restore from active_trackers if position_tickets not present
            elif active_trackers:
                # CRITICAL: Notify DualPolicy of the LAST recovered position
                # (DualPolicy still manages one position at a time, but all are monitored)
                if hasattr(self.app, "policy"):
                    # Use the last tracker (most recent position)
                    last_tracker = list(active_trackers.values())[-1]
                    import datetime as dt  # noqa: PLC0415

                    mfe = last_tracker.get("mfe", 0.0)
                    mae = last_tracker.get("mae", 0.0)

                    # Use on_recovery() to preserve MFE/MAE
                    if hasattr(self.app.policy, "on_recovery"):
                        self.app.policy.on_recovery(
                            direction=last_tracker["direction"],
                            entry_price=last_tracker["entry_price"],
                            entry_time=dt.datetime.now(dt.UTC),
                            mfe=mfe,
                            mae=mae,
                            ticks_held=0,
                        )
                    else:
                        # Fallback for older code
                        self.app.policy.on_entry(
                            direction=last_tracker["direction"],
                            entry_price=last_tracker["entry_price"],
                            entry_time=dt.datetime.now(dt.UTC),
                        )
                    LOG.info(
                        _MSG_RECOVERED_ENTRY,
                        last_tracker["entry_price"],
                        last_tracker["direction"],
                        mfe,
                        mae,
                    )

            # Legacy: Keep backward compatibility for old state files without active_trackers
            elif position_recovered and self.entry_price and self.position_direction != 0:
                position_id = f"{self.app.symbol_id}_net"
                if hasattr(self.app, "mfe_mae_trackers"):
                    from src.core.ctrader_ddqn_paper import MFEMAETracker  # noqa: PLC0415

                    if position_id not in self.app.mfe_mae_trackers:
                        self.app.mfe_mae_trackers[position_id] = MFEMAETracker(position_id)

                    self.app.mfe_mae_trackers[position_id].start_tracking(self.entry_price, self.position_direction)
                    LOG.info(
                        "[INTEGRATION] ✓ MFE/MAE tracker reinitialized with entry=%.5f direction=%d",
                        self.entry_price,
                        self.position_direction,
                    )

                    # Notify DualPolicy
                    if hasattr(self.app, "policy"):
                        import datetime as dt  # noqa: PLC0415

                        # Get MFE/MAE from tracker if available
                        mfe, mae = 0.0, 0.0
                        tracker = self.app.mfe_mae_trackers.get(position_id)
                        if tracker:
                            mfe = getattr(tracker, "mfe", 0.0)
                            mae = getattr(tracker, "mae", 0.0)

                        # Use on_recovery() to preserve MFE/MAE
                        if hasattr(self.app.policy, "on_recovery"):
                            self.app.policy.on_recovery(
                                direction=self.position_direction,
                                entry_price=self.entry_price,
                                entry_time=dt.datetime.now(dt.UTC),
                                mfe=mfe,
                                mae=mae,
                                ticks_held=0,
                            )
                        else:
                            # Fallback for older code
                            self.app.policy.on_entry(
                                direction=self.position_direction,
                                entry_price=self.entry_price,
                                entry_time=dt.datetime.now(dt.UTC),
                            )
                    LOG.info(
                        _MSG_RECOVERED_ENTRY,
                        self.entry_price,
                        self.position_direction,
                        mfe,
                        mae,
                    )

            # CRITICAL: Force immediate MFE/MAE update after recovery
            # This ensures stale persisted MAE values get recalculated with current price
            if hasattr(self.app, "policy") and hasattr(self.app.policy, "_update_mfe_mae"):
                # Calculate mid_price from best bid/ask if available
                mid_price = None
                if (
                    hasattr(self.app, "best_bid")
                    and hasattr(self.app, "best_ask")
                    and self.app.best_bid
                    and self.app.best_ask
                ):
                    mid_price = (self.app.best_bid + self.app.best_ask) / 2.0

                if mid_price and mid_price > 0:
                    try:
                        self.app.policy._update_mfe_mae(mid_price)
                        LOG.info(
                            "[INTEGRATION] ✓ Forced MFE/MAE update after recovery: MFE=%.4f MAE=%.4f",
                            self.app.policy.mfe,
                            self.app.policy.mae,
                        )
                    except Exception as e:
                        LOG.warning("[INTEGRATION] Failed to force MFE/MAE update: %s", e)
                else:
                    LOG.debug("[INTEGRATION] No market price available yet for forced MFE/MAE update")

            if self.trailing_stop_active:
                LOG.info(
                    "[INTEGRATION] ✓ Recovered trailing stop state: %s pos, distance=%.2f%%",
                    "LONG" if self.position_direction == 1 else "SHORT",
                    self.trailing_stop_distance_pct * 100,
                )

            return position_recovered
        except Exception as e:
            LOG.error("[INTEGRATION] State recovery failed: %s", e, exc_info=True)
            return False

    def _validate_position_after_fill(self, expected_direction: int, order: Order):
        """
        FIX P0-6: Validate position matches expected state after order fill.

        Args:
            expected_direction: Expected position direction (1=LONG, -1=SHORT)
            order: The filled order
        """
        try:
            # Give TradeManager a moment to process PositionReport
            import time  # noqa: PLC0415

            time.sleep(0.1)

            actual_direction = self.trade_manager.get_position_direction(min_qty=self.app.qty * 0.5)

            if actual_direction != expected_direction:
                LOG.error(
                    "[VALIDATION] \u2717 Position mismatch after fill! Expected=%d Actual=%d (Order: %s %.6f)",
                    expected_direction,
                    actual_direction,
                    order.side.name,
                    order.filled_qty,
                )

                # Request fresh position report for reconciliation
                self.trade_manager.request_positions()

                # Alert could be sent here
                if hasattr(self.app, "alert_manager"):
                    self.app.alert_manager.send_alert(
                        "CRITICAL", f"Position validation failed: Expected {expected_direction}, got {actual_direction}"
                    )
            else:
                direction_str = "LONG" if actual_direction > 0 else ("SHORT" if actual_direction < 0 else "FLAT")
                LOG.debug(
                    "[VALIDATION] \u2713 Position confirmed: %s=%d (%.6f lots)",
                    direction_str,
                    actual_direction,
                    order.filled_qty,
                )

        except Exception as e:
            LOG.error("[VALIDATION] Position validation failed: %s", e, exc_info=True)

    def _get_position_id_for_order(self, order: Order) -> str:
        """
        Determine position ID for an order.

        MULTI-POSITION: Generate unique ID for each position to support
        multiple concurrent positions. Uses order ID or timestamp to ensure uniqueness.

        Args:
            order: The filled order

        Returns:
            Position ID string
        """
        # Try to get hedge position ID from order/TradeManager
        if hasattr(order, "pos_maint_rpt_id") and order.pos_maint_rpt_id:
            return order.pos_maint_rpt_id

        # Check TradeManager for mapped position
        if self.trade_manager and hasattr(self.trade_manager, "order_to_position"):
            position_id = self.trade_manager.order_to_position.get(order.clord_id)
            if position_id:
                return position_id

        # MULTI-POSITION: Create unique ID for each order fill
        # This allows tracking multiple independent positions instead of just net position
        # Use order ClOrdID as unique identifier
        if order.clord_id:
            return f"{self.app.symbol_id}_{order.clord_id}"

        # Fallback: use timestamp-based ID
        import time  # noqa: PLC0415

        return f"{self.app.symbol_id}_{int(time.time() * 1000)}"

    def _cleanup_position_trackers(self, position_id: str):
        """
        Clean up trackers for closed position.

        MULTI-POSITION: Remove trackers to free memory.

        Args:
            position_id: Position that was closed
        """
        if hasattr(self.app, "mfe_mae_trackers"):
            self.app.mfe_mae_trackers.pop(position_id, None)
            LOG.debug("[MULTI-POS] Removed MFE/MAE tracker for: %s", position_id)

        if hasattr(self.app, "path_recorders"):
            self.app.path_recorders.pop(position_id, None)
            LOG.debug("[MULTI-POS] Removed path recorder for: %s", position_id)


if __name__ == "__main__":
    print("TradeManager Integration Example")
    print("\nIntegration Steps:")
    print("1. Add TradeManagerIntegration to CTraderFixApp.__init__()")
    print("2. Call initialize_trade_manager() in onCreate() for TRADE session")
    print("3. Route ExecutionReport and PositionReport to TradeManager")
    print("4. Replace send_market_order() with enter_position()/exit_position()")
    print("\nBenefits:")
    print("  ✓ Centralized order management")
    print("  ✓ Complete order lifecycle tracking")
    print("  ✓ Automatic position reconciliation")
    print("  ✓ Clean separation of concerns")
    print("  ✓ Easy to test and debug")
