#!/usr/bin/env python3
"""
Standalone Emergency Close - Manually close all positions

Can be run while bot is running or standalone.
Handles both netting and hedging modes.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import quickfix as fix  # noqa: E402
import quickfix44 as fix44  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)


def close_via_running_bot():
    """Close positions via running bot's TradeManagerIntegration.

    Not supported from external scripts — logs guidance and returns False.
    """
    LOG.error("Cannot access running bot instance from external script")
    LOG.info("Use one of these methods instead:")
    LOG.info("  1. Set CTRADER_AUTO_CLOSE_ON_BREAKER=1 and trigger circuit breaker")
    LOG.info("  2. Kill bot and use standalone FIX session method below")
    return False


class EmergencyCloseApp(fix.Application):  # pylint: disable=arguments-renamed
    """FIX application that requests positions and submits market close orders."""

    def __init__(self) -> None:
        super().__init__()
        self.session_id = None
        self.positions = {}
        self.orders_submitted = 0
        self.orders_filled = 0

    def onCreate(self, session_id):
        LOG.info("Session created: %s", session_id)
        self.session_id = session_id

    def onLogon(self, session_id):
        LOG.info("Logged on: %s", session_id)
        self._request_positions(session_id)

    def onLogout(self, session_id):
        LOG.info("Logged out: %s", session_id)

    def toAdmin(self, message, _session_id):
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)
        if msg_type.getValue() == "A":  # Logon
            username = os.environ.get("CTRADER_USERNAME")
            password = os.environ.get("CTRADER_PASSWORD_TRADE")

            if not username or not password:
                LOG.error("Missing CTRADER_USERNAME or" " CTRADER_PASSWORD_TRADE environment variables")
                sys.exit(1)

            message.setField(fix.Username(username))
            message.setField(fix.Password(password))

    def fromAdmin(self, _message, _session_id):
        """No admin message handling needed for emergency close."""

    def toApp(self, _message, _session_id):
        """No outgoing message processing needed."""

    def fromApp(self, message, _session_id):
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)

        if msg_type.getValue() == "AP":  # Position Report
            self._handle_position_report(message)
        elif msg_type.getValue() == "8":  # Execution Report
            self._handle_execution_report(message)

    def _request_positions(self, session_id):
        """Send RequestForPositions to broker."""
        msg = fix44.RequestForPositions()
        msg.setField(fix.PosReqID(f"emergency_{int(time.time())}"))
        msg.setField(fix.PosReqType(0))  # Positions request
        account = os.environ.get("CTRADER_USERNAME", "5179095")
        msg.setField(fix.Account(account))
        msg.setField(fix.AccountType(1))
        msg.setField(fix.TransactTime())

        fix.Session.sendToTarget(msg, session_id)
        LOG.info("✓ Requested positions from broker")

    def _handle_position_report(self, message):
        """Parse position report and submit close orders."""
        symbol = fix.Symbol()
        message.getField(symbol)

        if symbol.getValue() != "10028":  # Only BTCUSD
            return

        long_qty, short_qty = self._parse_quantities(message)

        LOG.info(
            "Positions: LONG=%s SHORT=%s NET=%s",
            long_qty,
            short_qty,
            long_qty - short_qty,
        )

        if long_qty > 0:
            self._submit_close("SELL", long_qty)
        if short_qty > 0:
            self._submit_close("BUY", short_qty)
        if long_qty == 0 and short_qty == 0:
            LOG.info("✓ No positions to close")

    @staticmethod
    def _parse_quantities(message) -> tuple[float, float]:
        """Extract long/short quantities from a PositionReport."""
        long_qty = 0.0
        short_qty = 0.0

        num_positions = fix.NoPositions()
        if not message.isSetField(num_positions):
            return long_qty, short_qty

        message.getField(num_positions)
        for i in range(1, num_positions.getValue() + 1):
            group = fix44.PositionReport().NoPositions()
            message.getGroup(i, group)

            pos_type = fix.PosType()
            group.getField(pos_type)

            if pos_type.getValue() != "TQ":  # Total quantity
                continue

            long_qty_field = fix.LongQty()
            short_qty_field = fix.ShortQty()

            if group.isSetField(long_qty_field):
                group.getField(long_qty_field)
                long_qty = long_qty_field.getValue()
            if group.isSetField(short_qty_field):
                group.getField(short_qty_field)
                short_qty = short_qty_field.getValue()

        return long_qty, short_qty

    def _submit_close(self, side_str, qty):
        """Submit market order to close position."""
        clord_id = f"EMERGENCY_{int(time.time() * 1000)}_{self.orders_submitted}"

        msg = fix44.NewOrderSingle()
        msg.setField(fix.ClOrdID(clord_id))
        msg.setField(fix.Symbol("10028"))
        msg.setField(fix.Side("1" if side_str == "BUY" else "2"))
        msg.setField(fix.TransactTime())
        msg.setField(fix.OrdType("1"))  # Market
        msg.setField(fix.OrderQty(round(qty, 2)))

        fix.Session.sendToTarget(msg, self.session_id)
        self.orders_submitted += 1
        LOG.warning(
            "CLOSE ORDER: %s %s units (ClOrdID=%s)",
            side_str,
            qty,
            clord_id,
        )

    def _handle_execution_report(self, message):
        """Handle order fill confirmation."""
        exec_type = fix.ExecType()
        message.getField(exec_type)

        if exec_type.getValue() == "F":  # Filled
            cl_ord_id = fix.ClOrdID()
            message.getField(cl_ord_id)

            avg_px = fix.AvgPx()
            message.getField(avg_px)

            self.orders_filled += 1
            LOG.info(
                "Order filled: %s @ %.2f (%d/%d)",
                cl_ord_id.getValue(),
                avg_px.getValue(),
                self.orders_filled,
                self.orders_submitted,
            )


def close_via_fix_session():
    """Close positions using standalone FIX session."""
    try:
        settings = fix.SessionSettings("config/ctrader_trade.cfg")
        app = EmergencyCloseApp()
        store_factory = fix.FileStoreFactory(settings)
        log_factory = fix.FileLogFactory(settings)
        initiator = fix.SocketInitiator(app, store_factory, settings, log_factory)

        initiator.start()
        LOG.info("FIX session started, waiting for positions...")

        time.sleep(15)  # Wait for positions and closes to execute

        initiator.stop()

        LOG.info("=" * 80)
        LOG.info(
            "Emergency close complete: %d/%d orders filled",
            app.orders_filled,
            app.orders_submitted,
        )
        LOG.info("=" * 80)
        return True

    except (RuntimeError, OSError) as e:
        LOG.error("Error: %s", e, exc_info=True)
        return False


def main():
    """Parse arguments and execute emergency position close."""
    parser = argparse.ArgumentParser(description="Emergency position closer")
    parser.add_argument(
        "--method",
        choices=["fix", "bot"],
        default="fix",
        help="Method: 'fix' for standalone FIX session, 'bot' for running bot",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚨 EMERGENCY POSITION CLOSER 🚨")
    print("=" * 80)
    print(f"\nMethod: {args.method.upper()}")
    print("\nThis will IMMEDIATELY close ALL BTCUSD positions!")
    print()

    confirm = input("Type 'CLOSE ALL' to confirm: ")
    if confirm != "CLOSE ALL":
        print("❌ Aborted")
        return 1

    print()
    LOG.info("Starting emergency close...")

    if args.method == "fix":
        success = close_via_fix_session()
    else:
        success = close_via_running_bot()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
