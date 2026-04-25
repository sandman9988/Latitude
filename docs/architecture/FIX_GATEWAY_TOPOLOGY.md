# FIX Gateway Topology

## Problem

The old paper universe starts one `src.core.ctrader_ddqn_paper` process per
`(symbol, timeframe)`. Each process creates its own QUOTE and TRADE FIX
initiators using the same broker account identity from `ctrader_quote.cfg` and
`ctrader_trade.cfg`.

That does not scale safely. Adding M1, M5, M15, and later more symbols multiplies
logons, heartbeat loops, reconnect logic, order state, and position recovery for
the same broker account. Even when the processes eventually reconnect, they are
not a single portfolio-aware trading system.

## Target Shape

Use one broker gateway per broker account/environment.

The gateway owns:

- one QUOTE FIX session;
- one TRADE FIX session;
- symbol subscriptions;
- broker position and pending-order reconciliation;
- account-level kill switches and exposure limits;
- a durable event log of ticks, bars, decisions, order intents, acknowledgements,
  fills, rejections, and recovery actions.

Strategy workers own:

- one `(symbol, timeframe)` learning context;
- timeframe-specific bars and feature state;
- timeframe-specific learned parameters, reward shaping, runway prediction,
  decision logs, and checkpoints;
- order intents, not direct FIX order submission.

The gateway arbitrates:

- whether multiple timeframes may stack, hedge, flatten, or veto each other;
- max net exposure by symbol and account;
- stale quote and stale decision guards;
- one canonical broker position view;
- one path for execution report handling.

## Current Migration Guard

`run_universe.py` now supports `UNIVERSE_BROKER_TOPOLOGY`:

- `isolated`: legacy behavior; one direct-FIX process per PAPER entry.
- `shared-symbol`: one direct-FIX owner per symbol; other timeframes wait for the
  shared gateway path.
- `shared-account`: one direct-FIX owner for the broker account; all other
  PAPER entries wait for the shared gateway path.

The guard is intentionally conservative. It prevents silent FIX session
multiplication before the shared gateway exists. It does not pretend that a
single owner process can already trade every timeframe and symbol.

Example:

```bash
UNIVERSE_BROKER_TOPOLOGY=shared-account python3 run_universe.py --watch
```

## Migration Plan

1. Keep `isolated` as the compatibility default while current live paper bots are
   running.
2. Use `shared-symbol` or `shared-account` during controlled restarts when FIX
   session stability is more important than keeping every legacy direct-FIX bot
   active.
3. Extract market-data fanout from `CTraderFixApp`: one gateway subscription per
   symbol, many timeframe bar builders behind it.
4. Extract order intent submission: strategy workers publish desired position
   changes, and the gateway submits broker orders.
5. Move position ownership into a portfolio arbiter so M1/M5/M15 cannot
   independently fight over the same broker position.
6. Switch the universe default from `isolated` to `shared-account` only after
   strategy workers no longer need direct FIX sessions.
