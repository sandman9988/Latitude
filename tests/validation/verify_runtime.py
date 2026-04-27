#!/usr/bin/env python3
"""Runtime verification test for complete bot."""

from src.core.ctrader_ddqn_paper import CTraderFixApp

# Initialize bot
app = CTraderFixApp(symbol_id=41, symbol_name="EURUSD", qty=1000.0, timeframe_minutes=15)






stats = app.arena.get_stats()


