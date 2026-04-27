#!/usr/bin/env python3
"""Emergency position closer - submits market orders to close all positions.
Run this to clean up orphaned positions after testing.

Usage:
    python scripts/close_all_positions.py --symbol 10028 --qty 1.0 --side LONG
    python scripts/close_all_positions.py --symbol 10028 --qty 1.0 --side SHORT
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import quickfix as fix
import quickfix44 as fix44


def close_positions(symbol_id: int, quantity: float, side_to_close: str) -> None:
    """Submit market order to close positions.

    Args:
        symbol_id: Symbol ID (e.g., 10028 for BTCUSD)
        quantity: Total quantity to close
        side_to_close: "LONG" or "SHORT" - which side to close

    """
    # Determine order side (opposite of position)
    order_side = fix.Side_SELL if side_to_close.upper() == "LONG" else fix.Side_BUY


    # Load settings
    settings = fix.SessionSettings("config/ctrader_trade.cfg")

    # Create session
    class SimpleApplication(fix.Application):
        def onCreate(self, sessionID) -> None:
            pass

        def onLogon(self, sessionID) -> None:

            # Submit close order
            clord_id = f"CLEANUP_{int(time.time())}"
            msg = fix44.NewOrderSingle()
            msg.setField(fix.ClOrdID(clord_id))
            msg.setField(fix.Symbol(str(symbol_id)))
            msg.setField(fix.Side(order_side))
            msg.setField(fix.TransactTime())
            msg.setField(fix.OrdType(fix.OrdType_MARKET))
            msg.setField(fix.OrderQty(round(quantity, 2)))

            fix.Session.sendToTarget(msg, sessionID)

        def onLogout(self, sessionID) -> None:
            pass

        def toAdmin(self, message, sessionID) -> None:
            pass

        def fromAdmin(self, message, sessionID) -> None:
            pass

        def toApp(self, message, sessionID) -> None:
            pass

        def fromApp(self, message, _sessionID) -> None:
            msg_type = fix.MsgType()
            message.getHeader().getField(msg_type)

            if msg_type.getValue() == "8":  # ExecutionReport
                exec_type = fix.ExecType()
                if message.isSetField(exec_type):
                    message.getField(exec_type)
                    if exec_type.getValue() == "F":  # Fill
                        clord_id = fix.ClOrdID()
                        message.getField(clord_id)
                        avg_price = fix.AvgPx()
                        message.getField(avg_price)

    app = SimpleApplication()
    store_factory = fix.FileStoreFactory(settings)
    log_factory = fix.FileLogFactory(settings)
    initiator = fix.SocketInitiator(app, store_factory, settings, log_factory)

    initiator.start()
    time.sleep(5)
    initiator.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Close all positions for a symbol")
    parser.add_argument("--symbol", type=int, required=True, help="Symbol ID (e.g., 10028)")
    parser.add_argument("--qty", type=float, required=True, help="Quantity to close")
    parser.add_argument("--side", type=str, required=True, choices=["LONG", "SHORT"], help="Side to close")

    args = parser.parse_args()

    close_positions(args.symbol, args.qty, args.side)
