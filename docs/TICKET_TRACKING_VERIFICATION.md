# Ticket-Based Tracking Verification Checklist

## Quick Test Procedure

Use this checklist to verify the broker ticket tracking implementation works correctly.

---

## Pre-Test Setup

- [ ] Bot connected to demo account (Pepperstone cTrader)
- [ ] Clean state: `rm data/state/trade_integration_BTCUSD.json`
- [ ] Logs enabled: `tail -f logs/python/ctrader_app.log | grep -E "HEDGING|ticket"`

---

## Test 1: Single Position Entry ✓

**Objective**: Verify Tag 721 extraction and ticket-based tracker creation

**Steps**:
1. Start bot
2. Wait for TriggerAgent to open 1 position (BUY or SELL)
3. Check logs for ticket extraction

**Expected Logs**:
```
[TRADEMGR] Position ticket: 12345678 for order DDQN_cl_...
[INTEGRATION] Order filled: ... ticket=12345678
[HEDGING] Created tracker for position: 10028_ticket_12345678
```

**Verification**:
```bash
# Check state file
cat data/state/trade_integration_BTCUSD.json | jq '.position_tickets'
# Should show:
# {
#   "12345678": {
#     "position_id": "10028_ticket_12345678",
#     "entry_price": 95432.50,
#     ...
#   }
# }
```

**Pass Criteria**:
- [ ] Log shows `Position ticket: <number>`
- [ ] State file contains `position_tickets` dict with 1 entry
- [ ] Tracker ID matches format: `{symbol}_ticket_{ticket}`

---

## Test 2: Multi-Position Entry ✓

**Objective**: Verify multiple positions tracked independently by tickets

**Steps**:
1. Wait for bot to open 2nd position
2. Verify both positions have unique tickets

**Expected Logs**:
```
[HEDGING] Created tracker for position: 10028_ticket_12345678
[HEDGING] Created tracker for position: 10028_ticket_12345679
```

**Verification**:
```bash
cat data/state/trade_integration_BTCUSD.json | jq '.position_tickets | length'
# Should show: 2
```

**Pass Criteria**:
- [ ] Two different tickets in state file
- [ ] Two different position_ids in trackers
- [ ] Both positions tracked in `mfe_mae_trackers`

---

## Test 3: Position Exit ✓

**Objective**: Verify exit orders mapped to correct ticket

**Steps**:
1. Wait for Harvester to close 1 position (or manual close)
2. Check logs for ticket-based close

**Expected Logs**:
```
[INTEGRATION] Closing position ticket=12345678: SELL 0.1 (reason=TICK_HARVESTER)
[INTEGRATION] Order filled: ... ticket=12345678
[HEDGING] ✓ Closed position ticket 12345678 (tracker=10028_ticket_12345678)
```

**Verification**:
```bash
cat data/state/trade_integration_BTCUSD.json | jq '.position_tickets | length'
# Should show: 1 (one closed, one remains)
```

**Pass Criteria**:
- [ ] Exit order logs show `ticket=<number>`
- [ ] Correct tracker removed from `mfe_mae_trackers`
- [ ] State file updated (1 position remains)
- [ ] Other position(s) unaffected

---

## Test 4: Crash Recovery ⚠️ CRITICAL

**Objective**: Verify positions recovered from state file using tickets

**Steps**:
1. With 2+ positions open, check state file
2. Kill bot: `pkill -f ctrader_ddqn_paper`
3. Restart bot: `./run.sh`
4. Check logs for recovery

**Expected Logs**:
```
[INTEGRATION] Attempting to load state from: ...
[HEDGING] ✓ Recovered position ticket=12345678: pos_id=10028_ticket_12345678 entry=95432.50 dir=1
[HEDGING] ✓ Recovered position ticket=12345679: pos_id=10028_ticket_12345679 entry=95450.00 dir=-1
```

**Verification**:
```bash
# Before restart
cat data/state/trade_integration_BTCUSD.json | jq '.position_tickets | keys'
# Should show: ["12345678", "12345679"]

# After restart - check logs
grep "Recovered position ticket" logs/python/ctrader_app.log | tail -n 5
```

**Pass Criteria**:
- [ ] All positions recovered (count matches pre-crash)
- [ ] Tickets match between state file and logs
- [ ] Trackers recreated with same position_ids
- [ ] `tracker.position_ticket` set correctly
- [ ] Bot continues to track positions normally

---

## Test 5: Harvester Close After Recovery

**Objective**: Verify Harvester can close positions using recovered tickets

**Steps**:
1. After recovery test, let bot run
2. Wait for Harvester to close a recovered position
3. Check logs for ticket-based close

**Expected Logs**:
```
[TICK_EXIT] Harvester CLOSE 10028_ticket_12345678 @ 95500.00
[INTEGRATION] Closing position ticket=12345678: SELL 0.1 (reason=TICK_HARVESTER)
[HEDGING] ✓ Closed position ticket 12345678
```

**Pass Criteria**:
- [ ] Harvester finds recovered position
- [ ] Close operation uses correct ticket
- [ ] Position closed successfully
- [ ] No "No broker ticket" warnings

---

## Test 6: State Persistence Cycle

**Objective**: Verify tickets persist across multiple save/load cycles

**Steps**:
1. Open 3 positions
2. Note tickets from state file
3. Restart bot
4. Verify same tickets loaded
5. Close 1 position
6. Restart bot
7. Verify remaining 2 tickets loaded

**Verification**:
```bash
# Before first restart
jq '.position_tickets | keys' data/state/trade_integration_BTCUSD.json
# Should show: ["12345678", "12345679", "12345680"]

# After close + restart
jq '.position_tickets | keys' data/state/trade_integration_BTCUSD.json
# Should show: ["12345678", "12345680"] (one removed)
```

**Pass Criteria**:
- [ ] Tickets consistent across restarts
- [ ] Closed position ticket removed from state
- [ ] Remaining tickets preserved exactly

---

## Test 7: H4 Stress Test (10 Positions)

**Objective**: Simulate production scenario with 10 concurrent H4 positions

**Setup**:
```bash
# Temporarily reduce entry threshold to open many positions quickly
# Edit bot_config.json or use aggressive profile
```

**Steps**:
1. Let bot open 10 positions
2. Check state file shows 10 tickets
3. Crash and restart
4. Verify all 10 recovered
5. Let Harvester close all 10
6. Verify clean shutdown with 0 positions

**Expected State**:
```json
{
  "position_tickets": {
    "12345678": {...},
    "12345679": {...},
    "12345680": {...},
    ...
    "12345687": {...}
  }
}
```

**Pass Criteria**:
- [ ] All 10 positions tracked with unique tickets
- [ ] State file size reasonable (<50KB)
- [ ] Crash recovery restores all 10
- [ ] Harvester closes each independently
- [ ] Final state shows 0 tickets

---

## Failure Scenarios

### No Ticket Extracted
**Symptom**: Log shows `No ticket - using ClOrdID fallback`

**Diagnosis**:
```bash
grep "No ticket" logs/python/ctrader_app.log
```

**Possible Causes**:
- Broker not sending Tag 721 (unlikely for cTrader)
- ExecutionReport parsing bug
- Order fill before ticket assigned

**Action**: Check raw FIX logs for Tag 721

---

### Wrong Position Closed
**Symptom**: Harvester closes position A but logs show position B closed

**Diagnosis**:
```bash
grep "Closing position ticket" logs/python/ctrader_app.log | tail -n 10
grep "Closed position ticket" logs/python/ctrader_app.log | tail -n 10
```

**Possible Causes**:
- `exit_order_to_ticket` mapping incorrect
- Tracker scan not finding correct ticket
- Multiple positions with same ticket (impossible)

**Action**: Add debug logging in `on_order_filled()` exit detection

---

### Recovery Failure
**Symptom**: Positions not recovered after restart

**Diagnosis**:
```bash
cat data/state/trade_integration_BTCUSD.json | jq '.position_tickets'
grep "Recovered position ticket" logs/python/ctrader_app.log
```

**Possible Causes**:
- State file corrupted (check `.backup`)
- Ticket format mismatch
- Recovery code not executing

**Action**: Check state file backup, verify recovery logs

---

## Success Summary

✅ **All Tests Passed** = Production Ready

**Checklist**:
- [ ] Tag 721 extraction working
- [ ] Single position tracked by ticket
- [ ] Multiple positions tracked independently
- [ ] Exit orders close correct position
- [ ] Crash recovery restores all positions
- [ ] Harvester closes recovered positions
- [ ] 10 position stress test passed

---

## Production Deployment

Once all tests pass:

1. **Deploy to Demo**: Run for 1 week with real trading
2. **Monitor**: Check logs daily for warnings
3. **Validate**: Verify state files consistent
4. **Go Live**: Deploy to production account

**Monitoring Commands**:
```bash
# Daily ticket tracking check
grep -E "ticket=" logs/python/ctrader_app.log | tail -n 50

# State file health check
jq '.position_tickets | length' data/state/trade_integration_BTCUSD.json

# Recovery validation (after restarts)
grep "Recovered position ticket" logs/python/ctrader_app.log | wc -l
```

---

## Rollback Plan

If implementation fails in production:

1. **Immediate**: Stop bot, close all positions manually
2. **Revert**: `git revert HEAD` (or restore from backup)
3. **Recovery**: Use legacy ClOrdID-based tracking
4. **Debug**: Analyze failure logs, fix issues
5. **Re-test**: Run full test suite again

**Manual Position Close**:
```bash
python scripts/close_all_positions.py --symbol 10028 --qty 0.1 --side LONG
python scripts/close_all_positions.py --symbol 10028 --qty 0.1 --side SHORT
```

---

## Notes

- Test suite execution time: ~30-60 minutes (depends on market activity)
- Requires live market data (M1 BTCUSD)
- Demo account recommended for initial tests
- H4 stress test may take hours/days to complete
