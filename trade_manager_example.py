"""
TradeManager Integration Example

Shows how to integrate TradeManager with existing CTraderFixApp
to centralize all order and position management.

Enhanced with defense-in-depth safety layer.
"""

import logging
from typing import TYPE_CHECKING

import quickfix as fix

from trade_manager import Order, Side, TradeManager
from trade_manager_safety import PERSISTENCE, VALIDATOR, SafeMath

if TYPE_CHECKING:
    from ctrader_ddqn_paper import CTraderFixApp

LOG = logging.getLogger(__name__)


class TradeManagerIntegration:
    """
    Example integration of TradeManager with CTraderFixApp.

    Replaces scattered order management code with centralized TradeManager.
    """

    def __init__(self, app: "CTraderFixApp"):
        """
        Initialize TradeManager integration.

        Args:
            app: CTraderFixApp instance with active FIX sessions
        """
        self.app = app
        self.trade_manager: TradeManager | None = None

        # Trailing stop state
        self.trailing_stop_active = False
        self.trailing_stop_order: Order | None = None
        self.trailing_stop_distance_pct = 0.20  # 20 pips default (0.20%)
        self.highest_price_since_entry: float | None = None  # For LONG
        self.lowest_price_since_entry: float | None = None  # For SHORT
        self.entry_price: float | None = None
        self.position_direction: int = 0  # 1=LONG, -1=SHORT, 0=FLAT

        LOG.info("[INTEGRATION] TradeManager integration initialized")

    def initialize_trade_manager(self):
        """
        Initialize TradeManager once TRADE session is connected.

        Call this from CTraderFixApp.onCreate() after TRADE session login.
        """
        if not self.app.trade_sid:
            LOG.warning("[INTEGRATION] Cannot initialize - TRADE session not connected")
            return

        self.trade_manager = TradeManager(
            session_id=self.app.trade_sid,
            symbol_id=self.app.symbol_id,
            on_fill_callback=self.on_order_filled,
            on_reject_callback=self.on_order_rejected,
            max_pending_orders=10,
        )

        # Request initial positions
        self.trade_manager.request_positions()

        # Attempt to recover previous state
        self._recover_state()

        LOG.info("[INTEGRATION] ✓ TradeManager initialized for symbol=%d", self.app.symbol_id)

    def on_order_filled(self, order: Order):
        """
        Callback when order fills.

        Replaces logic from CTraderFixApp.on_exec_report() for fill handling.
        """
        LOG.info(
            "[INTEGRATION] Order filled: %s qty=%.6f @%.5f",
            order.side.name,
            order.filled_qty,
            order.avg_price,
        )

        # Start MFE/MAE tracking
        if hasattr(self.app, "mfe_mae_tracker") and order.avg_price > 0:
            direction = 1 if order.side == Side.BUY else -1
            self.app.mfe_mae_tracker.start_tracking(order.avg_price, direction)
            self.app.trade_entry_time = order.filled_at

        # Start path recording
        if hasattr(self.app, "path_recorder") and order.filled_at:
            direction = 1 if order.side == Side.BUY else -1
            self.app.path_recorder.start_recording(
                order.filled_at,
                order.avg_price,
                direction,
            )

        # Notify activity monitor
        if hasattr(self.app, "activity_monitor"):
            self.app.activity_monitor.on_trade_executed(order.filled_at)

        # Store entry state for online learning
        if hasattr(self.app, "policy") and hasattr(self.app.policy, "trigger"):
            self.app.entry_state = getattr(self.app.policy.trigger, "last_state", None)
            self.app.entry_action = 1 if order.side == Side.BUY else 2

        # Request position update
        self.trade_manager.request_positions()

        # Enable trailing stop after entry
        if hasattr(self.app, "param_manager"):
            distance = self.app.param_manager.get(
                self.app.symbol,
                "trailing_stop_distance_pct",
                timeframe=self.app.timeframe_label,
                broker="default",
                default=0.20,
            )
            self.enable_trailing_stop(distance_pct=float(distance))

    def on_order_rejected(self, order: Order):
        """
        Callback when order is rejected.

        Replaces logic from CTraderFixApp.on_exec_report() for reject handling.
        """
        LOG.warning(
            "[INTEGRATION] Order rejected: %s - %s",
            order.clord_id,
            order.reject_reason or "Unknown reason",
        )

        # Could add rejection tracking/alerting here

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

        # Defensive: Validate order before submission
        fix_side_str = "1" if side == 1 else "2"
        validation = VALIDATOR.validate_order(
            symbol=self.app.symbol_id,
            side=fix_side_str,
            quantity=quantity,
            price=None,
            order_type="MARKET",
        )

        if not validation.valid:
            LOG.error("[INTEGRATION] ❌ Order validation failed: %s", validation.error)
            return False

        # Use validated quantity (safe Decimal)
        safe_qty = float(validation.sanitized_value["quantity"])

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
        if not SafeMath.is_finite(current_price) or current_price <= 0:
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
            stop_price: Stop loss trigger price
        """
        if not self.trade_manager:
            return

        # If stop order exists, modify it
        if self.trailing_stop_order and self.trailing_stop_order.clord_id:
            try:
                self.trade_manager.modify_order(
                    original_clord_id=self.trailing_stop_order.clord_id,
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

    def exit_position(self, quantity: float) -> bool:
        """
        Exit position using TradeManager.

        Args:
            quantity: Quantity to close

        Returns:
            True if order submitted, False otherwise
        """
        if not self.trade_manager:
            LOG.error("[INTEGRATION] TradeManager not initialized")
            return False

        # Determine exit side (opposite of current position)
        pos_dir = self.trade_manager.get_position_direction(min_qty=quantity * 0.5)
        if pos_dir == 0:
            LOG.warning("[INTEGRATION] No position to exit")
            return False

        exit_side = Side.SELL if pos_dir > 0 else Side.BUY
        order = self.trade_manager.submit_market_order(
            side=exit_side,
            quantity=quantity,
            tag_prefix="EXIT",
        )

        return order is not None

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
        if not self.trade_manager:
            return {}
        return self.trade_manager.get_statistics()

    def _persist_state(self):
        """Persist current trailing stop state for crash recovery"""
        try:
            state = {
                "trailing_stop_active": self.trailing_stop_active,
                "trailing_stop_distance_pct": self.trailing_stop_distance_pct,
                "highest_price_since_entry": self.highest_price_since_entry,
                "lowest_price_since_entry": self.lowest_price_since_entry,
                "entry_price": self.entry_price,
                "position_direction": self.position_direction,
                "symbol_id": self.app.symbol_id,
            }
            PERSISTENCE.save_state(state)
            LOG.debug("[INTEGRATION] State persisted")
        except Exception as e:
            LOG.error("[INTEGRATION] Failed to persist state: %s", e, exc_info=True)

    def _recover_state(self):
        """Recover trailing stop state after crash/restart"""
        try:
            state = PERSISTENCE.load_state()
            if not state:
                LOG.info("[INTEGRATION] No previous state to recover")
                return

            # Validate symbol matches
            if state.get("symbol_id") != self.app.symbol_id:
                LOG.warning(
                    "[INTEGRATION] State symbol mismatch: %s vs %s, skipping recovery",
                    state.get("symbol_id"),
                    self.app.symbol_id,
                )
                return

            # Restore trailing stop state
            self.trailing_stop_active = state.get("trailing_stop_active", False)
            self.trailing_stop_distance_pct = state.get("trailing_stop_distance_pct", 0.20)
            self.highest_price_since_entry = state.get("highest_price_since_entry")
            self.lowest_price_since_entry = state.get("lowest_price_since_entry")
            self.entry_price = state.get("entry_price")
            self.position_direction = state.get("position_direction", 0)

            if self.trailing_stop_active:
                LOG.info(
                    "[INTEGRATION] ✓ Recovered trailing stop state: %s pos, distance=%.2f%%",
                    "LONG" if self.position_direction == 1 else "SHORT",
                    self.trailing_stop_distance_pct * 100,
                )
        except Exception as e:
            LOG.error("[INTEGRATION] State recovery failed: %s", e, exc_info=True)


# Example: How to integrate into CTraderFixApp
"""
# In ctrader_ddqn_paper.py:

from trade_manager_example import TradeManagerIntegration

class CTraderFixApp(fix.Application):
    def __init__(self, symbol_id, qty, timeframe_minutes=15, symbol="BTCUSD"):
        # ... existing init ...
        
        # Add TradeManager integration
        self.trade_integration = TradeManagerIntegration(self)
    
    def onCreate(self, session_id):
        # ... existing onCreate ...
        
        qual = self._qual(session_id)
        if qual == "TRADE":
            # Initialize TradeManager after TRADE session connected
            self.trade_integration.initialize_trade_manager()
    
    def fromApp(self, message, session_id):
        # ... existing fromApp ...
        
        if t == "8":  # ExecutionReport
            self.trade_integration.handle_execution_report(message)
        elif t == "AP":  # PositionReport
            self.trade_integration.handle_position_report(message)
    
    def on_bar_close(self, bar):
        # ... existing logic ...
        
        # Replace send_market_order() calls:
        # OLD:
        # self.send_market_order(side="1", qty=order_qty)
        
        # NEW:
        side = 1 if action == 1 else 2
        self.trade_integration.enter_position(side=side, quantity=order_qty)
        
        # For exits:
        # OLD:
        # exit_side = "2" if self.cur_pos > 0 else "1"
        # self.send_market_order(side=exit_side, qty=self.qty)
        
        # NEW:
        self.trade_integration.exit_position(quantity=self.qty)
"""

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
