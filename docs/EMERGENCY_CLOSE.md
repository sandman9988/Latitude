# Emergency Position Closing - Circuit Breaker Integration

## Overview

Robust emergency position closing system integrated with circuit breakers for automated risk management.

## Components

### 1. EmergencyPositionCloser (`src/risk/emergency_close.py`)
- Closes ALL positions when called
- Handles hedging mode (close by broker ticket)
- Handles netting mode (close net position)
- Retry logic for failed closes
- Verification that positions closed

### 2. CircuitBreakerManager Integration
- Auto-close on breaker trip (configurable)
- Tracks close attempts to avoid duplicates
- Logs emergency actions to audit trail

### 3. Standalone Script (`scripts/emergency_close_positions.py`)
- Manual emergency close via FIX session
- Can run while bot is running or standalone
- Requires CTRADER_USERNAME and CTRADER_PASSWORD_TRADE env vars

## Usage

### Automated (Circuit Breaker)

Enable auto-close in bot initialization:

```python
self.circuit_breakers = CircuitBreakerManager(
    symbol=symbol,
    timeframe=self.timeframe_label,
    broker="default",
    param_manager=self.param_manager,
    auto_close_on_trip=True,  # ← Enable auto-close
)

# Set emergency closer (done automatically in main bot)
emergency_closer = create_emergency_closer(self.trade_integration)
self.circuit_breakers.set_emergency_closer(emergency_closer)
```

Or via environment variable:
```bash
export CTRADER_AUTO_CLOSE_ON_BREAKER=1
./run.sh
```

### Manual Emergency Close

**Method 1: Standalone FIX Session (Recommended)**
```bash
# Set credentials
export CTRADER_USERNAME="5179095"
export CTRADER_PASSWORD_TRADE="your_password"

# Run emergency closer
python3 scripts/emergency_close_positions.py --method fix
```

**Method 2: Via Running Bot**
```python
# In bot console or script
from src.risk.emergency_close import create_emergency_closer
emergency_closer = create_emergency_closer(app.trade_integration)
emergency_closer.close_all_positions(reason="MANUAL")
```

## Circuit Breaker Triggers

Emergency close activates when ANY breaker trips:

1. **Sortino Ratio** < 0.5 (configurable)
   - Risk-adjusted returns too low
   
2. **Excess Kurtosis** > 5.0 (configurable)
   - Fat tails detected (outlier risk)
   
3. **Drawdown** > 15% (configurable)
   - Maximum drawdown exceeded
   
4. **Consecutive Losses** >= 5 (configurable)
   - Loss streak protection

## Close Mechanisms

### Hedging Mode (Preferred)
1. Iterate through `position_tickets` dict
2. For each ticket, get position_id and tracker
3. Submit close order with `PosMaintRptID` (tag 721) set to ticket
4. Map exit order to ticket for cleanup on fill

### Netting Mode (Fallback)
1. Check TradeManager net position
2. Submit opposite market order for net quantity
3. Position closes when filled

### Last Resort
If tickets and trackers unavailable, scan MFE/MAE trackers for active positions.

## Verification

After emergency close:
```python
all_closed = emergency_closer.verify_all_closed()
if not all_closed:
    LOG.error("Manual intervention required!")
```

Checks:
- ✓ `mfe_mae_trackers` empty
- ✓ `position_tickets` empty  
- ✓ TradeManager shows zero positions

## Testing

```bash
# Test circuit breaker integration
python3 -c "from src.risk.circuit_breakers import CircuitBreakerManager; \
from src.risk.emergency_close import EmergencyPositionCloser; \
print('✓ Imports OK')"

# Simulate breaker trip (in test environment)
manager = CircuitBreakerManager(auto_close_on_trip=True)
emergency_closer = EmergencyPositionCloser(trade_integration)
manager.set_emergency_closer(emergency_closer)

# Force trip
manager.consecutive_losses_breaker.state.trip("TEST", 10, 5)
manager.check_all()  # Should trigger emergency close
```

## Monitoring

Circuit breaker events logged to:
- **Python logs**: `ctrader_py_logs/ctrader_*.log`
- **Audit log**: `log/trade_audit.jsonl`
- **Transaction log**: `log/transactions.jsonl`

Search for:
```bash
grep "CIRCUIT.*TRIPPED" ctrader_py_logs/*.log
grep "EMERGENCY CLOSE" ctrader_py_logs/*.log
grep "CIRCUIT_BREAKER" log/trade_audit.jsonl
```

## Recovery

After emergency close and cooldown:

1. **Reset breakers** (after fixing root cause):
```python
circuit_breakers.reset_all()
```

2. **Auto-reset after cooldown**:
```python
circuit_breakers.reset_if_cooldown_elapsed()
```

Cooldown periods:
- Sortino: 120 minutes
- Kurtosis: 60 minutes  
- Drawdown: 240 minutes
- Consecutive Losses: 180 minutes

## Safety Features

- ✅ Automatic close when breakers trip
- ✅ Idempotent (won't double-close)
- ✅ Audit trail of all emergency actions
- ✅ Verification of successful close
- ✅ Retry logic for failed closes
- ✅ Standalone script for manual intervention
- ✅ Works with both hedging and netting modes

## Troubleshooting

**Issue**: Emergency close fails

**Solutions**:
1. Check FIX session connectivity
2. Verify TradeManager initialized
3. Check for valid position tickets
4. Use standalone script as backup
5. Close manually via cTrader platform

**Issue**: Positions still open after close

**Check**:
```bash
# View persisted state
cat data/state/trade_integration_BTCUSD.json | jq '.data.position_tickets'

# Check audit log
tail -20 log/trade_audit.jsonl | jq 'select(.event_type=="POSITION_CLOSE")'

# Verify with broker
python3 scripts/emergency_close_positions.py --method fix
```

## Production Deployment

1. **Enable auto-close** (recommended for production):
   ```bash
   export CTRADER_AUTO_CLOSE_ON_BREAKER=1
   ```

2. **Set appropriate thresholds**:
   ```bash
   export CTRADER_SORTINO_THRESHOLD=0.5
   export CTRADER_KURTOSIS_THRESHOLD=5.0
   export CTRADER_MAX_DRAWDOWN=0.15
   export CTRADER_MAX_CONSECUTIVE_LOSSES=5
   ```

3. **Monitor circuit breaker status**:
   ```bash
   # In bot logs
   grep "Circuit Breaker" ctrader_py_logs/*.log | tail -20
   ```

4. **Test emergency close** before going live:
   ```bash
   # In demo environment
   python3 scripts/emergency_close_positions.py --method fix
   ```
