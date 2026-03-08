# cTrader DDQN Trading Bot

A dual FIX session trading bot for cTrader/Pepperstone that uses Deep Q-Network (DDQN) reinforcement learning with a dual-agent architecture (TriggerAgent + HarvesterAgent) to trade XAUUSD (Gold) on M5 timeframes.

## 🚀 Quick Start

**New to deployment?** See [docs/guides/DEPLOYMENT_QUICKSTART.md](docs/guides/DEPLOYMENT_QUICKSTART.md) for the production deployment guide.

**TL;DR:**

1. Run `scripts/testing/phase0_validate_system.sh` (2-4 hour paper validation)
2. Launch bot with `./run.sh` (set `QTY=0.001` for micro-position learning)
3. Monitor with `scripts/monitoring/monitor.sh`
4. Graduate to Phase 2 after 500+ profitable trades

**Why this approach?** Avoids RL complacency from paper training. Agents learn real friction costs (spread, slippage, requotes) from day one with tiny positions (~$2-3 max loss). See [docs/guides/PAPER_VS_LIVE_CONFIG.md](docs/guides/PAPER_VS_LIVE_CONFIG.md) for full rationale.

---

## 📚 Documentation

**System Status & Recent Changes:**

- 📄 [**CURRENT_STATE.md**](docs/CURRENT_STATE.md) - Latest fixes, parameters, and system health (updated Mar 8, 2026)
- 📁 [**INDEX.md**](docs/INDEX.md) - Complete navigation index for all documentation files

**Core Documentation:**

- 📖 [MASTER_HANDBOOK.md](MASTER_HANDBOOK.md) - Authoritative system design and architecture
- 🚀 [docs/00_START_HERE.md](docs/00_START_HERE.md) - Documentation entry point with organized guides
- 🏗 [docs/architecture/SYSTEM_ARCHITECTURE.md](docs/architecture/SYSTEM_ARCHITECTURE.md) - Technical architecture

---

## Features

- **Dual FIX Sessions**: Separate QUOTE and TRADE sessions for market data and order execution
- **M5 Bar Building**: Constructs 5-minute candlestick bars from best bid/ask prices
- **DDQN Policy**: Optional deep reinforcement learning model for trading decisions
- **Fallback Strategy**: Simple moving average crossover strategy when no model is loaded
- **Position Management**: Automatic position tracking and target-based order execution
- **Microstructure Signals**: Order book imbalance, VPIN, and depth analysis
- **Friction Modeling**: SymbolInfo-based spread/commission/slippage costs

### Phase 1: Defensive Programming (✅ Complete)

- **SafeMath/SafeArray**: Guards against NaN/Inf/bounds errors
- **Atomic Persistence**: CRC32-validated file writes with automatic backups
- **VaR Estimation**: Dynamic Value-at-Risk with regime/VPIN/kurtosis adjustments
- **Circuit Breakers**: Kurtosis-based automatic order cancellation
- **Adaptive Parameters**: Self-optimizing learned parameters with soft bounds
- **Self-Test**: Startup health checks (QuickFIX importable, model weights loadable, CB schema)

### Phase 2: Advanced RL Features (✅ Complete)

- **Activity Monitoring**: Prevents learned helplessness via trade frequency tracking
- **Counterfactual Analysis**: Compares actual exits vs optimal (MFE-based)
- **Non-Repaint Guards**: Enforces strict bar[0] discipline (prevents look-ahead bias)
- **Ring Buffers**: O(1) rolling statistics (6.8x faster than naive approach)
- **Ensemble Tracking**: Multi-model disagreement for epistemic uncertainty
- **Ensemble Policy**: Multi-model support with disagreement-based exploration
- **Enhanced Reward Shaping**: 6 components (capture, WTL, opportunity, activity, counterfactual, ensemble)
- **Safe Bar Access**: Helper methods with non-repaint discipline documentation

### Phase 3: Dual-Agent Architecture (✅ Complete)

- **TriggerAgent**: Entry specialist with runway prediction (10-50 pip MFE forecasts)
- **HarvesterAgent**: Exit specialist with capture optimization (MFE-aware exits)
- **DualPolicy**: Orchestrates trigger + harvester agents with position lifecycle tracking
- **Specialized State Spaces**: 7 features for entry, 10 features (7 market + 3 position) for exit
- **Backward Compatible**: Single-agent mode when `DDQN_DUAL_AGENT=0` (default)
- **Fallback Strategies**: Rule-based logic for model-free operation
  - Trigger: MA crossover + microstructure tilt (0.3 thresholds)
  - Harvester: 0.3% profit target, 0.12% stop loss, 50-bar time stop

## Project Structure

```
ctrader_trading_bot/
├── src/                           # Source code
│   ├── agents/                   # Agent implementations
│   │   ├── dual_policy.py        # Dual-agent orchestrator (Phase 3)
│   │   ├── trigger_agent.py      # Entry specialist (Phase 3)
│   │   └── harvester_agent.py    # Exit specialist (Phase 3)
│   ├── core/                     # Core trading system
│   │   ├── ctrader_ddqn_paper.py # ⭐ Main trading bot application
│   │   ├── trade_manager.py      # FIX order lifecycle management
│   │   ├── reward_shaper.py      # RL reward engineering (6 components)
│   │   ├── order_book.py         # L2 order book + VPIN calculator
│   │   └── self_test.py          # Startup health checks
│   ├── features/                 # Feature engineering
│   │   ├── feature_engine.py     # Feature computation pipeline
│   │   ├── regime_detector.py    # Market regime classification
│   │   └── event_time_features.py # Activity-time bar features
│   ├── monitoring/               # HUD and monitoring
│   │   ├── hud_tabbed.py         # Tabbed terminal dashboard
│   │   ├── performance_tracker.py # Trade metrics and analytics
│   │   └── trade_exporter.py     # JSON trade export
│   ├── persistence/              # State management
│   │   ├── atomic_persistence.py # CRC32-validated file operations (Phase 1)
│   │   └── learned_parameters.py # Adaptive parameter system
│   ├── risk/                     # Risk management
│   │   ├── risk_manager.py       # Position sizing & risk limits
│   │   ├── var_estimator.py      # VaR with kurtosis monitoring (Phase 1)
│   │   ├── circuit_breakers.py   # Kurtosis-based order cancellation
│   │   └── friction_costs.py     # Spread/commission/slippage modeling
│   └── utils/                    # Utilities
│       ├── safe_utils.py         # Defensive programming utilities (Phase 1)
│       ├── ring_buffer.py        # O(1) rolling statistics (Phase 2)
│       └── non_repaint_guards.py # Bar[0] discipline enforcement (Phase 2)
├── tests/                         # Test suite (124 files, 2506 passing)
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── validation/               # Validation tests
├── config/                        # Configuration files
│   ├── ctrader_quote.cfg         # QUOTE session FIX config
│   ├── ctrader_trade.cfg         # TRADE session FIX config
│   └── cTraderAppTokens          # OAuth credentials
├── scripts/                       # Utility scripts
│   ├── testing/                  # Test & validation scripts
│   │   └── phase0_validate_system.sh # 2-4 hour validation run
│   └── monitoring/               # Live monitoring dashboards
├── data/                          # Runtime data
│   ├── learned_parameters.json   # Atomic-persisted params (with backups)
│   └── sessions/                 # FIX session state
├── docs/                          # Documentation
│   ├── CURRENT_STATE.md          # Latest status (read this first)
│   ├── INDEX.md                  # Documentation navigation
│   ├── architecture/             # System design documents
│   ├── guides/                   # User & operator guides
│   └── operations/               # Runbooks
├── MASTER_HANDBOOK.md            # Authoritative system design (root)
├── run.sh                         # Convenience launcher script
├── train_offline.py               # Offline training pipeline
├── run_universe.py                # Universe management
├── requirements.txt               # Python dependencies
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
- Risk/microstructure gates (learned params override env):
	- `CTRADER_DEPTH_LEVELS` - Depth levels to evaluate (default: 5)
	- `CTRADER_DEPTH_BUFFER` - Required depth multiplier vs order qty (default: 2.0)
	- `CTRADER_SPREAD_RELAX` - Allowable spread = min_spread * relax (default: 3.0)
	- `CTRADER_VPIN_Z_LIMIT` - VPIN z-score threshold to gate entries (default: 2.5)
	- `CTRADER_VOL_REF` - Reference realized vol for size scaling (default: 0.005)
	- `CTRADER_VOL_CAP` - Block entries above this realized vol (default: 0.05)
	- `CTRADER_RISK_BUDGET_USD` - Max USD risk per 1-sigma move per order (default: 50.0)
	- `CTRADER_KURTOSIS_THRESHOLD` - Excess kurtosis threshold for circuit breaker (default: 3.0)

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
python3 -m src.core.ctrader_ddqn_paper
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
- 7 features: 1-bar return, 5-bar return, MA difference, volatility, imbalance, VPIN, depth_ratio
- 3 actions: SHORT (0), FLAT (1), LONG (2)
- Convolutional neural network architecture

#### Single-Agent Mode (Default - Phase 1/2)

```bash
export DDQN_MODEL_PATH="path/to/model.pth"
./run.sh
```
Uses `Policy.decide()` for all entry/exit decisions.

#### Dual-Agent Mode (Phase 3)

```bash
export DDQN_DUAL_AGENT=1
export DDQN_TRIGGER_MODEL="path/to/trigger_model.pth"  # optional
export DDQN_HARVESTER_MODEL="path/to/harvester_model.pth"  # optional
./run.sh
```
- **TriggerAgent** handles entry decisions (LONG/SHORT/NONE) with runway prediction
- **HarvesterAgent** handles exit decisions (HOLD/CLOSE) with capture optimization
- Falls back to rule-based strategies if models not provided
- Backward compatible: Set `DDQN_DUAL_AGENT=0` to use single-agent mode

#### Ensemble Mode (Multi-Model - Phase 2)

```bash
# Provide comma-separated model paths
export DDQN_MODEL_PATH="model1.pth,model2.pth,model3.pth"
export DDQN_MODEL_ENSEMBLE=1
./run.sh
```

**Ensemble Benefits:**

- Quantifies epistemic uncertainty via disagreement
- Exploration bonus when models disagree (high uncertainty)
- Performance-weighted voting for robust decisions
- Better sample efficiency during training

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
- Monitor positions manually
- **Circuit breakers active**: Orders auto-cancelled on kurtosis > 3.0
- **VaR-based sizing**: Position sizes adapt to volatility regime
- **Atomic persistence**: All parameter updates are crash-safe with CRC32
- **Defensive programming**: NaN/Inf guards on all float operations

## Advanced Features

### Kurtosis Circuit Breaker

The bot monitors excess kurtosis (fat tail risk) and automatically cancels all pending orders when kurtosis exceeds the threshold (default: 3.0). This protects against trading in unstable market conditions.

### VaR Estimation

Dynamic VaR calculation with multi-factor adjustments:

- **Base VaR**: Historical 95th percentile of returns
- **Regime multiplier**: 1.0 (ranging) to 2.0 (trending)
- **VPIN adjustment**: Scales with toxic flow (VPIN z-score)
- **Kurtosis adjustment**: Increases for fat-tail distributions
- **Volatility scaling**: Adapts to current vs reference volatility

### Atomic Persistence

All learned parameters are saved with:

- CRC32 checksums for corruption detection
- Automatic backup (keeps last 3 versions)
- Crash-safe write (temp file → atomic rename)
- Auto-restore from backup on CRC failure

### Defensive Programming

SafeMath utilities prevent runtime errors:

- Division by zero → returns default value
- NaN/Inf propagation → sanitized to valid numbers
- Array bounds checking → safe access with default values
- Soft bounds on learned parameters → tanh clamping

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

**Version**: 3.1  
**Last Updated**: February 22, 2026  
**Status**: Production-ready for paper trading — test suite green (2506 passed, 0 failed)  
**Handbook Alignment**: See [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md) for latest status
