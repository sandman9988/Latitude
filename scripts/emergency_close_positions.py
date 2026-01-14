#!/usr/bin/env python3
"""
Standalone Emergency Close - Manually close all positions

Can be run while bot is running or standalone.
Handles both netting and hedging modes.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)


def close_via_running_bot():
    """Close positions via running bot's TradeManagerIntegration"""
    try:
        # Try to import bot module
        from src.core.ctrader_ddqn_paper import CTraderFixApp
        from src.risk.emergency_close import create_emergency_closer

        LOG.error("Cannot access running bot instance from external script")
        LOG.info("Use one of these methods instead:")
        LOG.info("  1. Set CTRADER_AUTO_CLOSE_ON_BREAKER=1 and trigger circuit breaker")
        LOG.info("  2. Kill bot and use standalone FIX session method below")
        return False

    except Exception as e:
        LOG.error("Error: %s", e)
        return False


def close_via_fix_session():
    """Close positions using standalone FIX session"""
    import time
    import quickfix as fix
    import quickfix44 as fix44
    from src.core.trade_manager import Side

    class EmergencyCloseApp(fix.Application):
        def __init__(self):
            super().__init__()
            self.session_id = None
            self.positions = {}
            self.orders_submitted = 0
            self.orders_filled = 0

        def onCreate(self, session_id):
            LOG.info(f"Session created: {session_id}")
            self.session_id = session_id

        def onLogon(self, session_id):
            LOG.info(f"✓ Logged on: {session_id}")

            # Request positions
            msg = fix44.RequestForPositions()
            msg.setField(fix.PosReqID(f"emergency_{int(time.time())}"))
            msg.setField(fix.PosReqType(0))  # 0 = Positions
            msg.setField(fix.Account(os.environ.get("CTRADER_USERNAME", "5179095")))
            msg.setField(fix.AccountType(1))
            msg.setField(fix.TransactTime())

            fix.Session.sendToTarget(msg, session_id)
            LOG.info("✓ Requested positions from broker")

        def onLogout(self, session_id):
            LOG.info(f"Logged out: {session_id}")

        def toAdmin(self, message, session_id):
            msgType = fix.MsgType()
            message.getHeader().getField(msgType)
            if msgType.getValue() == "A":  # Logon
                username = os.environ.get("CTRADER_USERNAME")
                password = os.environ.get("CTRADER_PASSWORD_TRADE")

                if not username or not password:
                    LOG.error("Missing CTRADER_USERNAME or CTRADER_PASSWORD_TRADE environment variables")
                    sys.exit(1)

                message.setField(fix.Username(username))
                message.setField(fix.Password(password))

        def fromAdmin(self, message, session_id):
            pass

        def toApp(self, message, session_id):
            pass

        def fromApp(self, message, session_id):
            msgType = fix.MsgType()
            message.getHeader().getField(msgType)

            if msgType.getValue() == "AP":  # Position Report
                self._handle_position_report(message)
            elif msgType.getValue() == "8":  # Execution Report
                self._handle_execution_report(message)

        def _handle_position_report(self, message):
            """Parse position report and submit close orders"""
            symbol = fix.Symbol()
            message.getField(symbol)

            if symbol.getValue() != "10028":  # Only BTCUSD
                return

            # Parse positions
            long_qty = 0.0
            short_qty = 0.0

            numPosAmt = fix.NoPositions()
            if message.isSetField(numPosAmt):
                message.getField(numPosAmt)
                for i in range(1, numPosAmt.getValue() + 1):
                    group = fix44.PositionReport().NoPositions()
                    message.getGroup(i, group)

                    posType = fix.PosType()
                    group.getField(posType)

                    if posType.getValue() == "TQ":  # Total quantity
                        longQty = fix.LongQty()
                        shortQty = fix.ShortQty()

                        if group.isSetField(longQty):
                            group.getField(longQty)
                            long_qty = longQty.getValue()
                        if group.isSetField(shortQty):
                            group.getField(shortQty)
                            short_qty = shortQty.getValue()

            LOG.info(f"📊 Positions: LONG={long_qty} SHORT={short_qty} NET={long_qty - short_qty}")

            # Submit close orders
            if long_qty > 0:
                self._submit_close("SELL", long_qty)

            if short_qty > 0:
                self._submit_close("BUY", short_qty)

            if long_qty == 0 and short_qty == 0:
                LOG.info("✓ No positions to close")

        def _submit_close(self, side_str, qty):
            """Submit market order to close position"""
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
            LOG.warning(f"🚨 CLOSE ORDER: {side_str} {qty} units (ClOrdID={clord_id})")

        def _handle_execution_report(self, message):
            """Handle order fill confirmation"""
            execType = fix.ExecType()
            message.getField(execType)

            if execType.getValue() == "F":  # Filled
                clOrdID = fix.ClOrdID()
                message.getField(clOrdID)

                avgPx = fix.AvgPx()
                message.getField(avgPx)

                self.orders_filled += 1
                LOG.info(
                    f"✓ Order filled: {clOrdID.getValue()} @ {avgPx.getValue():.2f} ({self.orders_filled}/{self.orders_submitted})"
                )

    # Run FIX session
    try:
        settings = fix.SessionSettings("config/ctrader_trade.cfg")
        app = EmergencyCloseApp()
        store_factory = fix.FileStoreFactory(settings)
        log_factory = fix.FileLogFactory(settings)
        initiator = fix.SocketInitiator(app, store_factory, settings, log_factory)

        initiator.start()
        LOG.info("🔌 FIX session started, waiting for positions...")

        time.sleep(15)  # Wait for positions and closes to execute

        initiator.stop()

        LOG.info("=" * 80)
        LOG.info(f"✓ Emergency close complete: {app.orders_filled}/{app.orders_submitted} orders filled")
        LOG.info("=" * 80)
        return True

    except Exception as e:
        LOG.error(f"Error: {e}", exc_info=True)
        return False


def main():
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
