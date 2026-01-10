# Pre-Launch Checklist

Complete this checklist BEFORE running `./phase0_validate_system.sh`

---

## 1. Environment Setup

### Python Dependencies
- [ ] Python 3.8+ installed: `python3 --version`
- [ ] All packages installed: `pip install -r requirements.txt`
- [ ] No import errors: `python3 -c "import quickfix; import numpy; import pandas; print('OK')"`

### Project Files
- [ ] All Phase 3 integration files present:
  - `safe_math.py`
  - `circuit_breakers.py`
  - `event_time_features.py`
  - `dual_policy.py`
  - `trigger_agent.py`
  - `harvester_agent.py`
- [ ] Main bot file: `ctrader_ddqn_paper.py`
- [ ] Deployment scripts executable:
  - `phase0_validate_system.sh`
  - `launch_micro_learning.sh`
  - `monitor_phase1.sh`

---

## 2. cTrader Credentials

### .env File
- [ ] `.env` file exists in project root
- [ ] Contains all required variables:
  ```bash
  CTRADER_USERNAME=<your_account_id>
  CTRADER_PASSWORD_QUOTE=<quote_password>
  CTRADER_PASSWORD_TRADE=<trade_password>
  ```
- [ ] Passwords match cTrader account settings
- [ ] No extra quotes or spaces around values

### FIX Configuration Files
- [ ] `config/ctrader_quote.cfg` exists
- [ ] `config/ctrader_trade.cfg` exists
- [ ] Both configs have correct:
  - `SenderCompID` (your account ID)
  - `TargetCompID` (usually `CSERVER` for cTrader)
  - `SocketConnectHost` (e.g., `h51.p.ctrader.com`)
  - `SocketConnectPort` (e.g., `5201` for quote, `5202` for trade)

### Test Credentials
- [ ] Can log into cTrader web/desktop app with same credentials
- [ ] Account has sufficient balance:
  - **Phase 0 (paper):** $0 required
  - **Phase 1 (micro):** $100-200 recommended minimum
  - **Phase 2 (mini):** $1,000+ recommended
  - **Phase 3 (standard):** $10,000+ recommended

---

## 3. Broker Configuration

### Symbol Verification
- [ ] XAUUSD (Gold) is available on your account
- [ ] Symbol ID is `41` (verify in cTrader: Settings → Symbols)
- [ ] Micro-lots allowed (0.001 minimum position size)
- [ ] Spreads reasonable (typically 2-8 pips for XAUUSD)

### Trading Hours
- [ ] Understand XAUUSD trading hours (23:00 Sunday - 22:00 Friday GMT)
- [ ] Know rollover times (typically 22:00-23:00 GMT for session transitions)
- [ ] Aware of high-impact news events (NFP, FOMC, etc.)

### Margin Requirements
- [ ] Understand leverage (typically 1:100 to 1:500 for gold)
- [ ] Calculated margin for QTY=0.001:
  - At $2000/oz gold: ~$2-20 margin per trade (depends on leverage)
- [ ] Account has 10x cushion (e.g., $200+ balance for Phase 1)

---

## 4. Network & System

### Internet Connection
- [ ] Stable internet (ping cTrader servers: `ping h51.p.ctrader.com`)
- [ ] Latency < 100ms to broker
- [ ] No frequent disconnections

### System Resources
- [ ] Available disk space: >1GB (for logs, models, trade exports)
- [ ] Available RAM: >2GB free
- [ ] CPU not maxed out (bot uses <10% single core)

### Firewall/VPN
- [ ] Ports 5201 (quote) and 5202 (trade) not blocked
- [ ] No VPN interfering with FIX connections (or VPN allows outbound TCP)
- [ ] Antivirus not blocking Python scripts

---

## 5. Directory Structure

### Required Directories
- [ ] `logs/` exists
- [ ] `logs/phase0_validation/` will be created automatically
- [ ] `logs/live_micro/` will be created automatically
- [ ] `config/` exists with FIX configs
- [ ] `data/` exists (for learned parameters)
- [ ] `trades/` exists or will be created (for trade CSV exports)

### Permissions
- [ ] Scripts are executable: `ls -l *.sh` shows `-rwxr-xr-x`
- [ ] Can write to logs: `touch logs/test.log && rm logs/test.log`
- [ ] Can create FIX session files: `touch data/sessions/test && rm data/sessions/test`

---

## 6. Safety Verifications

### Circuit Breakers
- [ ] Understand circuit breaker thresholds:
  - Sortino < 0.8 → position size reduced
  - Max drawdown > 12% → position size reduced
  - 3+ consecutive losses → cooldown period
- [ ] Know how to disable if too aggressive: `export DISABLE_GATES=1`

### Position Limits
- [ ] Phase 0 (paper): No real risk
- [ ] Phase 1 (micro): Max loss ~$2-3 per trade
- [ ] Comfortable with max drawdown: ~$50-100 over 500 trades
- [ ] Emergency stop command known: `pkill -f ctrader_ddqn_paper.py`

### Monitoring Plan
- [ ] Know how to view logs: `tail -f logs/live_micro/learning_*.log`
- [ ] Know how to run monitor: `./monitor_phase1.sh`
- [ ] Alerts configured (optional): SMS/email on critical errors
- [ ] Check-in schedule: Daily for first week, then weekly

---

## 7. Time Commitment

### Phase 0 (System Validation)
- [ ] Can dedicate 2-4 hours for validation run
- [ ] Available to monitor for crashes
- [ ] Can fix issues and restart if needed

### Phase 1 (Micro-Position Learning)
- [ ] Understand this runs 24/7 for 2-4 weeks
- [ ] Server/VPS available (or local machine stays on)
- [ ] Will check logs daily for first week
- [ ] Will run graduation check after 500+ trades

### Phase 2+ (Scaling)
- [ ] Committed to full 2-month progression (Phase 0 → Phase 3)
- [ ] Understand this is LIVE TRADING (real money at risk)
- [ ] Have exit plan if metrics don't improve

---

## 8. Understanding the Strategy

### Living Ecosystem Concept
- [ ] Read [PAPER_VS_LIVE_CONFIG.md](PAPER_VS_LIVE_CONFIG.md)
- [ ] Understand RL bootstrapping paradox
- [ ] Understand why paper training creates complacency
- [ ] Understand why micro-positions are the solution

### Graduation Criteria
- [ ] Read [PHASE1_GRADUATION_CHECKLIST.md](PHASE1_GRADUATION_CHECKLIST.md)
- [ ] Understand Sharpe > 1.0 requirement
- [ ] Understand Sortino > 0.8 requirement
- [ ] Understand win rate ≥ 45% requirement
- [ ] Understand 500+ trade minimum

### Risk Tolerance
- [ ] Comfortable losing $500-1500 during Phase 1 (worst case)
- [ ] Comfortable losing $5k-15k during Phase 2 (worst case)
- [ ] Understand this is EXPERIMENTAL (no guarantees)
- [ ] Have trading capital separate from living expenses

---

## 9. Documentation Review

### Core Docs Read
- [ ] [DEPLOYMENT_QUICKSTART.md](DEPLOYMENT_QUICKSTART.md) - Overview
- [ ] [PAPER_VS_LIVE_CONFIG.md](PAPER_VS_LIVE_CONFIG.md) - Strategy rationale
- [ ] [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md) - Current system state
- [ ] [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Technical details

### Know Where to Find
- [ ] Phase 1 graduation check script location
- [ ] How to adjust hyperparameters (epsilon, learning rate)
- [ ] How to disable circuit breakers if needed
- [ ] How to export trade history for analysis

---

## 10. Final Checks

### Pre-Launch Commands
- [ ] Stop any running bots: `pkill -f ctrader_ddqn_paper.py`
- [ ] Clear old logs if desired: `rm -rf logs/phase0_validation/*`
- [ ] Test .env loading: `source .env && echo $CTRADER_USERNAME`
- [ ] Verify Python path: `which python3` (should be /usr/bin/python3 or venv)

### Mental Preparation
- [ ] Understand Phase 0 is just validation (not training)
- [ ] Understand Phase 1 is where real learning happens
- [ ] Prepared for initially poor metrics (30-40% win rate OK early)
- [ ] Patience for 2-4 week learning period
- [ ] No panic selling on first few losses

### Support Resources
- [ ] Know how to check GitHub issues for bot repo
- [ ] Have cTrader support contact (for broker issues)
- [ ] Understand FIX protocol basics (for debugging connections)
- [ ] Backup plan if primary broker fails

---

## ✅ Ready to Launch

If ALL boxes above are checked, you're ready to proceed:

```bash
# Phase 0: System Validation (2-4 hours)
./phase0_validate_system.sh

# After validation passes:
# Phase 1: Live Micro-Position Learning (2-4 weeks)
./launch_micro_learning.sh

# In separate terminal:
./monitor_phase1.sh
```

---

## ❌ Not Ready Yet

If any boxes are unchecked:
1. Address missing items
2. Test credentials manually in cTrader app
3. Verify FIX connectivity with simple connection test
4. Review documentation thoroughly
5. Return to this checklist

**Don't skip steps.** Better to delay launch than to waste time debugging preventable issues during Phase 1.

---

## Emergency Contacts

- **Bot Issues:** Check logs first, then GitHub issues
- **Broker Issues:** cTrader support (support@spotware.com)
- **FIX Protocol:** QuickFIX documentation (http://www.quickfixengine.org/)
- **RL Questions:** Review MASTER_HANDBOOK.md for theoretical foundation

**Risk Disclaimer:** This is experimental trading software. Past performance does not guarantee future results. Only trade with capital you can afford to lose. The authors are not responsible for financial losses.
