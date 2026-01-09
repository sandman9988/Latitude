# cTrader DDQN Trading Bot

A dual FIX session trading bot for cTrader/Pepperstone that uses Deep Q-Network (DDQN) reinforcement learning to trade BTC/USD on M15 timeframes.

## Features

- **Dual FIX Sessions**: Separate QUOTE and TRADE sessions for market data and order execution
- **M15 Bar Building**: Constructs 15-minute candlestick bars from best bid/ask prices
- **DDQN Policy**: Optional deep reinforcement learning model for trading decisions
- **Fallback Strategy**: Simple moving average crossover strategy when no model is loaded
- **Position Management**: Automatic position tracking and target-based order execution

## Project Structure

```
ctrader_trading_bot/
├── ctrader_ddqn_paper.py         # Main trading bot application
├── config/                        # Configuration files
│   ├── ctrader_quote.cfg         # QUOTE session FIX config
│   ├── ctrader_trade.cfg         # TRADE session FIX config
│   └── cTraderAppTokens          # OAuth credentials (if needed)
├── scripts/                       # Utility scripts
│   └── ctrader_oauth_bootstrap.py # OAuth authentication helper
├── logs/                          # All log files
│   ├── python/                   # Python application logs
│   ├── fix_sessions/             # Combined FIX session logs
│   ├── fix_quote/                # QUOTE session FIX logs
│   └── fix_trade/                # TRADE session FIX logs
├── data/                          # Runtime data
│   └── sessions/                 # FIX session state
│       ├── store/                # Combined session store
│       ├── store_quote/          # QUOTE session store
│       └── store_trade/          # TRADE session store
├── docs/                          # Documentation
├── run.sh                         # Convenience launcher script
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore patterns
└── README.md                      # This file
```

## Requirements

- Python 3.12+
- QuickFIX Python bindings (compiled and installed)
- NumPy
- PyTorch (optional, for DDQN model)

### Installing QuickFIX

The QuickFIX library is already compiled in the `../quickfix` directory. To install the Python bindings:

```bash
cd ../quickfix
python3 -m pip install -e .
```

Or use the virtual environment:

```bash
source ../.venv/bin/activate
cd ../quickfix
pip install -e .
```

## Configuration

### Environment Variables

Required:
- `CTRADER_USERNAME` - Your cTrader account username (e.g., "5179095")
- `CTRADER_PASSWORD_QUOTE` - Password for QUOTE session
- `CTRADER_PASSWORD_TRADE` - Password for TRADE session

Optional:
- `CTRADER_CFG_QUOTE` - Path to QUOTE config (default: "config/ctrader_quote.cfg")
- `CTRADER_CFG_TRADE` - Path to TRADE config (default: "config/ctrader_trade.cfg")
- `CTRADER_BTC_SYMBOL_ID` - Symbol ID for BTC/USD (default: "10028")
- `CTRADER_QTY` - Order quantity (default: "0.10")
- `PY_LOGDIR` - Python log directory (default: "logs/python")
- `DDQN_MODEL_PATH` - Path to trained DDQN model (optional)

### FIX Configuration

The FIX configuration files (`ctrader_quote.cfg` and `ctrader_trade.cfg`) are pre-configured with:

- **Server**: demo-us-eqx-01.p.c-trader.com
- **Ports**: 5201 (QUOTE), 5202 (TRADE)
- **Protocol**: FIX 4.4 with SSL
- **TargetSubID**: Properly set to QUOTE/TRADE (required by cTrader)

## Usage

### Quick Start

```bash
cd ~/Documents/ctrader_trading_bot

# Set credentials
export CTRADER_USERNAME="5179095"
export CTRADER_PASSWORD_QUOTE="your_password"
export CTRADER_PASSWORD_TRADE="your_password"

# Run the bot
./run.sh
```

### Manual Start

```bash
# Activate virtual environment
source ../.venv/bin/activate

# Set environment variables
export CTRADER_USERNAME="5179095"
export CTRADER_PASSWORD_QUOTE="your_password"
export CTRADER_PASSWORD_TRADE="your_password"
export CTRADER_CFG_QUOTE="config/ctrader_quote.cfg"
export CTRADER_CFG_TRADE="config/ctrader_trade.cfg"

# Run
python3 ctrader_ddqn_paper.py
```

### With DDQN Model

```bash
export DDQN_MODEL_PATH="path/to/your/model.pth"
./run.sh
```

## Trading Strategy

### Without Model (Fallback)

Uses a simple moving average crossover strategy:
- **MA Fast**: 10-period moving average
- **MA Slow**: 30-period moving average
- **Long**: When MA diff > 0.2
- **Short**: When MA diff < -0.2
- **Flat**: Otherwise

### With DDQN Model

If a PyTorch model is provided via `DDQN_MODEL_PATH`, the bot uses:
- 64-bar lookback window
- 4 features: 1-bar return, 5-bar return, MA difference, volatility
- 3 actions: SHORT (0), FLAT (1), LONG (2)
- Convolutional neural network architecture

## Monitoring

### Logs

- **Python logs**: `logs/python/ctrader_YYYYMMDD_HHMMSS.log`
- **FIX logs**: `logs/fix_quote/` and `logs/fix_trade/`
- **Session state**: `data/sessions/store_quote/` and `data/sessions/store_trade/`

### Key Log Messages

- `[LOGON]` - Successful FIX session connection
- `[QUOTE]` - Market data subscription
- `[BAR]` - M15 bar close with trading decision
- `[TRADE]` - Order execution
- `[REJECT]` - Order or message rejection

## Troubleshooting

### Connection Issues

If you see `TargetSubID is assigned with the unexpected value` errors:
- ✅ Already fixed: Both config files now include `TargetSubID` field

### Module Not Found: quickfix

```bash
# Install QuickFIX Python bindings
source ../.venv/bin/activate
cd ../quickfix
pip install -e .
```

### No Market Data

- Check QUOTE session is logged in (`[LOGON]` message)
- Verify symbol ID is correct (default: 10028 for BTC/USD)
- Check FIX logs in `logs/fix_quote/`

### Orders Not Executing

- Check TRADE session is logged in
- Verify credentials are correct
- Check FIX logs in `logs/fix_trade/`
- Review execution reports (`[TRADE]` messages)

## Safety Notes

⚠️ **This is a demo trading bot**:
- Uses demo account credentials
- Default quantity is 0.10 BTC/USD
- No stop-loss or take-profit management
- No risk management beyond position targets
- Monitor positions manually

## Development

### Adding Features

1. Edit `ctrader_ddqn_paper.py`
2. Test with demo credentials
3. Monitor logs for issues

### Training DDQN Models

The bot supports loading pre-trained PyTorch models. Model architecture:
- Input: (batch, 64, 4) - 64 bars, 4 features
- Output: (batch, 3) - Q-values for SHORT/FLAT/LONG

## License

This is a research/demo project. Use at your own risk.

## Support

For issues related to:
- **cTrader FIX API**: Check Pepperstone/cTrader documentation
- **QuickFIX**: See https://www.quickfixengine.org/
- **Trading logic**: Review Python application logs

---

**Version**: 1.0  
**Last Updated**: January 9, 2026  
**Status**: Debugged and ready to run
