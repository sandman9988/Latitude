#!/usr/bin/env python3
"""
Emergency position closer - submits market orders to close all positions.
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

from src.core.trade_manager import Side


def close_positions(symbol_id: int, quantity: float, side_to_close: str):
    """
    Submit market order to close positions.

    Args:
        symbol_id: Symbol ID (e.g., 10028 for BTCUSD)
        quantity: Total quantity to close
        side_to_close: "LONG" or "SHORT" - which side to close
    """
    print(f"Closing {quantity} {side_to_close} positions for symbol {symbol_id}")

    # Determine order side (opposite of position)
    if side_to_close.upper() == "LONG":
        order_side = fix.Side_SELL
        side_name = "SELL"
    else:
        order_side = fix.Side_BUY
        side_name = "BUY"

    print(f"Will submit {side_name} order for {quantity} units")

    # Load settings
    settings = fix.SessionSettings("config/ctrader_trade.cfg")

    # Create session
    class SimpleApplication(fix.Application):
        def onCreate(self, sessionID):
            print(f"Session created: {sessionID}")

        def onLogon(self, sessionID):
            print(f"Logged on: {sessionID}")

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
            print(f"✓ Submitted {side_name} order: {clord_id} for {quantity}")

        def onLogout(self, sessionID):
            print(f"Logged out: {sessionID}")

        def toAdmin(self, message, sessionID):
            pass

        def fromAdmin(self, message, sessionID):
            pass

        def toApp(self, message, sessionID):
            pass

        def fromApp(self, message, sessionID):
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
                        print(f"✓ Order filled: {clord_id.getValue()} @ {avg_price.getValue()}")

    app = SimpleApplication()
    store_factory = fix.FileStoreFactory(settings)
    log_factory = fix.FileLogFactory(settings)
    initiator = fix.SocketInitiator(app, store_factory, settings, log_factory)

    initiator.start()
    print("Waiting 5 seconds for order to fill...")
    time.sleep(5)
    initiator.stop()
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Close all positions for a symbol")
    parser.add_argument("--symbol", type=int, required=True, help="Symbol ID (e.g., 10028)")
    parser.add_argument("--qty", type=float, required=True, help="Quantity to close")
    parser.add_argument("--side", type=str, required=True, choices=["LONG", "SHORT"], help="Side to close")

    args = parser.parse_args()

    close_positions(args.symbol, args.qty, args.side)
