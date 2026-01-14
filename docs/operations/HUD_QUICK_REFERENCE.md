# Tabbed HUD Quick Reference Card

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **1** | Overview Tab - Quick snapshot |
| **2** | Performance Tab - Detailed metrics |
| **3** | Training Tab - Agent statistics |
| **4** | Risk Tab - Risk management |
| **5** | Market Tab - Microstructure |
| **6** | Decision Log Tab - Trade history |
| **Tab** | Next tab (cycle forward) |
| **Shift+Tab** | Previous tab (cycle backward) |
| **h** | Help screen |
| **s** | Select symbol/timeframe preset |
| **q** | Quit HUD |

## Tab Contents

### Tab 1: Overview
- Current position (LONG/SHORT/FLAT)
- Unrealized P&L
- Today's trade count, win rate, P&L
- Recent performance sparkline ▃▂▅▇
- Risk status (circuit breaker, regime, volatility)
- Agent buffer sizes
- Market conditions (spread, VPIN, imbalance)
- **System health summary** 🏥

### Tab 2: Performance
- Metrics across 4 timeframes:
  - Daily
  - Weekly
  - Monthly
  - Lifetime
- For each: Trades, Win%, PnL, Sharpe, Sortino, Omega, MaxDD
- Additional stats: Best/Worst/Avg trade, Profit Factor, Expectancy

### Tab 3: Training
- Multi-agent arena info (if applicable)
- Trigger agent stats: Buffer size, loss, TD-error, epsilon
- Harvester agent stats: Buffer size, loss, TD-error, epsilon
- Visual buffer fill indicators
- Last training time

### Tab 4: Risk
- **Circuit breaker status** (visual alert if active)
- VaR and kurtosis (tail risk)
- Realized volatility and regime
- Path geometry: Efficiency, gamma, jerk, runway
- Entry feasibility gauge

### Tab 5: Market
- Bid-ask spread
- Depth (bid vs ask) with visual bar
- VPIN (order flow toxicity) with z-score
- Order imbalance (buy/sell pressure)
- Visual gauges for VPIN and imbalance

### Tab 6: Decision Log
- Last 20 trading decisions
- Color-coded events:
  - 🟢 Green: OPEN entries
  - 🔴 Red: CLOSE exits
  - 🟡 Yellow: HOLD actions
- Timestamps and decision details

## Color Coding Legend

| Color | Meaning |
|-------|---------|
| 🟢 **Green** | Positive, good, profitable, normal |
| 🔴 **Red** | Negative, alert, loss, critical |
| 🟡 **Yellow** | Neutral, warning, hold |
| 🔵 **Blue** | Information, regime transitional |
| ⚪ **Gray** | Inactive, waiting |

## Footer Indicators

### Data Freshness
- ✓ Data fresh (X.Xs old) - Green, <5 seconds
- ⚡ Data aging (Xs old) - Yellow, 5-10 seconds
- ⚠️ Data stale (Xs old) - Red, >10 seconds
- ⏳ Waiting for data... - Gray, never updated

### Circuit Breaker Alert
When circuit breaker is ACTIVE, the header turns red:
```
╔══════════════════════════════════╗
║ ⚠️ CIRCUIT BREAKER ACTIVE ⚠️    ║
╚══════════════════════════════════╝
```

## System Health (Overview Tab)

4-metric health check:
- **Data Fresh:** ✓ OK / ⚡ Aging / ✗ Stale
- **Circuit Breaker:** ✓ OK / ⚠️ Active
- **Agent Buffers:** ✓ OK / ⚡ Low / ✗ Critical
- **Volatility:** ✓ Normal / ⚡ Elevated / ⚠️ High

## Sparkline (Overview Tab)

Visual trend of last 20 trades:
```
Recent: ▃▂▅▇▁▆█▄
```
- Higher bars = larger P&L
- Green = profitable trades
- Red = losing trades

## Data Files Required

HUD reads from `data/` directory:
- `bot_config.json` - Bot configuration
- `current_position.json` - Position details
- `performance_snapshot.json` - Performance metrics
- `training_stats.json` - Training statistics
- `risk_metrics.json` - Risk & market data
- `decision_log.json` - Decision history

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Data stale warning | Check if bot is running |
| Missing files error | Ensure bot is exporting data |
| Garbled display | Use terminal with UTF-8 & ANSI colors |
| Keys not working | Try different terminal emulator |
| HUD frozen | Press Ctrl+C to exit, restart |

## Tips

1. **Press 'h' anytime** for full help screen
2. **Watch data freshness** in footer to ensure bot is alive
3. **Check system health** in Overview for quick diagnostics
4. **Use Tab/Shift+Tab** to quickly cycle through tabs
5. **Monitor circuit breaker** - red header means trading halted
6. **Review decision log** to understand bot behavior
7. **Terminal size:** Minimum 80x24 recommended

## Symbol/Timeframe Presets

Press **'s'** to access preset selection:
1. Displays available trading profiles
2. Select by number
3. Updates `.env` file
4. Requires bot restart to apply

Presets defined in: `config/profile_presets.json`

---

**Quick Start:**
```bash
# Start HUD
python3 hud_tabbed.py

# Navigate
Press 1-6 for tabs, h for help, q to quit

# Monitor
Check footer for data freshness
Check Overview for system health
```

---

*Last Updated: 2026-01-11*
