# 📚 Living Ecosystem Documentation Index

**Quick navigation for all deployment-related documents**

---

## 🚀 Start Here (Priority Order)

### 1️⃣ First Time? Read This
**[LIVING_ECOSYSTEM_COMPLETE.md](LIVING_ECOSYSTEM_COMPLETE.md)** - Complete overview  
- What you have (all files explained)
- The strategy in one page
- Launch sequence (3 commands)
- Expected outcomes per phase
- Success/failure metrics

### 2️⃣ Ready to Launch? Check This
**[PRE_LAUNCH_CHECKLIST.md](PRE_LAUNCH_CHECKLIST.md)** - Pre-flight verification  
- 10 categories: environment, credentials, broker, network, etc.
- Must complete BEFORE running Phase 0
- Prevents common setup issues

### 3️⃣ Need Quick Start? Use This
**[DEPLOYMENT_QUICKSTART.md](DEPLOYMENT_QUICKSTART.md)** - Fast-track guide  
- 3-step process (validate → launch → monitor)
- Bash commands ready to copy/paste
- Graduation validation script
- Troubleshooting section

---

## 📖 Deep Dive Documentation

### Strategy & Philosophy
**[PAPER_VS_LIVE_CONFIG.md](PAPER_VS_LIVE_CONFIG.md)** - 5200 lines, comprehensive  
- The RL complacency problem (4 subsections)
- RL bootstrapping paradox (Catch-22 explained)
- Why paper training fails
- Why micro-positions work
- Three-phase deployment strategy
- Environment variables reference
- FAQ (20+ questions)

**Best for:** Understanding WHY we skip paper training

### Trade Logging & Analysis
**[TRADE_LOGGING_GUIDE.md](TRADE_LOGGING_GUIDE.md)** - Complete reference  
- Automatic export system (3-layer redundancy)
- Data columns and file formats
- Analysis tools (trade_analyzer.py)
- Phase 1 graduation workflow
- Troubleshooting common issues
- Best practices for backups

**Best for:** Understanding how trades are saved and analyzed

---

### Execution Scripts

**[phase0_validate_system.sh](phase0_validate_system.sh)** - System validation  
- **Duration:** 2-4 hours max
- **Mode:** Paper, learning disabled
- **Goal:** Verify no crashes before live
- **Usage:** `./phase0_validate_system.sh`
- **Logs:** `logs/phase0_validation/`

**[launch_micro_learning.sh](launch_micro_learning.sh)** ⭐ THE MAIN SCRIPT  
- **Duration:** 2-4 weeks (24/7)
- **Mode:** Live, learning enabled
- **Position Size:** QTY=0.001 ($0.10/pip)
- **Goal:** Build living ecosystem from real friction
- **Usage:** `./launch_micro_learning.sh`
- **Logs:** `logs/live_micro/`

**[monitor_phase1.sh](monitor_phase1.sh)** - Real-time dashboard  
- **Displays:** Win rate, PnL, circuit trips, FIX status
- **Refresh:** Every 5 seconds
- **Usage:** `./monitor_phase1.sh` (separate terminal)
- **Purpose:** Track learning progress without log tailing

**Best for:** Copy/paste commands to get started

---

### Validation & Graduation

**[PHASE1_GRADUATION_CHECKLIST.md](PHASE1_GRADUATION_CHECKLIST.md)** - Scaling gates  
- 4 graduation criteria (Sharpe, Sortino, win rate, trade count)
- Python script to generate performance report
- What to do if NOT ready (tuning guide)
- Phase 2 configuration
- Phase 3 requirements
- Common pitfalls

**Best for:** Knowing when to scale from Phase 1 → Phase 2

---

### System Status

**[INTEGRATION_STATUS.md](INTEGRATION_STATUS.md)** - Current state  
- Phase 3 integration complete
- Next: Phase 1 micro-position learning
- File modification history
- Commit references

**[README.md](README.md)** - Project overview  
- Updated with Quick Start section
- Links to deployment docs
- Feature list
- Project structure

**Best for:** Understanding what's already integrated

---

## 🎯 Use Case Lookup

### "I want to understand the strategy"
→ Read **LIVING_ECOSYSTEM_COMPLETE.md** (1-page strategy section)  
→ Deep dive: **PAPER_VS_LIVE_CONFIG.md** (full rationale)

### "I want to start trading NOW"
→ Check **PRE_LAUNCH_CHECKLIST.md** (verify readiness)  
→ Run **phase0_validate_system.sh** (2-4 hour validation)  
→ Run **launch_micro_learning.sh** (Phase 1 start)  
→ Monitor with **monitor_phase1.sh**

### "I want to know if I can scale"
→ Run graduation script from **PHASE1_GRADUATION_CHECKLIST.md**  
→ Check if 3 of 4 criteria pass  
→ If yes: Configure Phase 2 (QTY=0.01)  
→ If no: Continue Phase 1 or tune hyperparameters

### "I want to troubleshoot issues"
→ **DEPLOYMENT_QUICKSTART.md** (troubleshooting section)  
→ **PHASE1_GRADUATION_CHECKLIST.md** (common pitfalls)  
→ Check logs: `tail -f logs/live_micro/learning_*.log`

### "I want to understand the tech"
→ **SYSTEM_ARCHITECTURE.md** (technical deep dive)  
→ **MASTER_HANDBOOK.md** (RL theory)  
→ **INTEGRATION_STATUS.md** (what's been integrated)

---

## 📊 File Size Reference

| File | Lines | Purpose | Read Time |
|------|-------|---------|-----------|
| LIVING_ECOSYSTEM_COMPLETE.md | 600 | Overview + strategy | 15 min |
| PAPER_VS_LIVE_CONFIG.md | 5200 | Deep rationale | 60 min |
| DEPLOYMENT_QUICKSTART.md | 350 | Fast start guide | 10 min |
| PRE_LAUNCH_CHECKLIST.md | 450 | Pre-flight check | 15 min |
| PHASE1_GRADUATION_CHECKLIST.md | 400 | Validation gates | 12 min |
| phase0_validate_system.sh | 180 | Validation script | - |
| launch_micro_learning.sh | 150 | Phase 1 script | - |
| monitor_phase1.sh | 200 | Dashboard script | - |

**Total documentation:** ~7,500 lines  
**Total reading time:** ~2 hours (if you read everything)  
**Minimum reading time:** 30 min (LIVING_ECOSYSTEM_COMPLETE.md + DEPLOYMENT_QUICKSTART.md)

---

## 🔄 Workflow Diagram

```
START HERE
    ↓
┌─────────────────────────────────────┐
│ PRE_LAUNCH_CHECKLIST.md             │ ← Verify readiness
│ ✓ Credentials                       │
│ ✓ Broker config                     │
│ ✓ Network                           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ phase0_validate_system.sh           │ ← 2-4 hour paper validation
│ Mode: Paper, Learning OFF           │
│ Goal: No crashes                    │
└─────────────────────────────────────┘
    ↓
   PASS? ────────┐
    ↓           NO → Fix bugs, re-run
   YES            ↓
    ↓             └──┐
┌─────────────────────────────────────┐
│ launch_micro_learning.sh            │ ← Phase 1 (2-4 weeks)
│ Mode: LIVE, Learning ON             │
│ QTY: 0.001 ($0.10/pip)              │
│ Max Loss: $2-3 per trade            │
└─────────────────────────────────────┘
    ↓
    ├─→ monitor_phase1.sh (Track progress)
    ↓
┌─────────────────────────────────────┐
│ Wait 500-1000 trades                │ ← 2-4 weeks runtime
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ PHASE1_GRADUATION_CHECKLIST.md      │ ← Run graduation script
│ Check: Sharpe, Sortino, Win Rate    │
└─────────────────────────────────────┘
    ↓
   PASS? ────────┐
    ↓           NO → Continue Phase 1 or tune
   YES            ↓
    ↓             └──┐
┌─────────────────────────────────────┐
│ Phase 2: QTY=0.01 ($1/pip)          │ ← 2-4 weeks
│ Max Loss: $20-30 per trade          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Phase 3: QTY=0.10 ($10/pip)         │ ← Production
│ Learning OFF, Freeze weights        │
└─────────────────────────────────────┘
```

---

## 🏆 Key Takeaways

### What's Different About This Deployment?
- ✅ **No extended paper training** (2-4 hours validation only)
- ✅ **Live from day one** (with micro positions)
- ✅ **Real friction costs** (agents learn truth, not simulation)
- ✅ **Progressive scaling** (validate before increasing risk)
- ✅ **Living ecosystem** (agents make real mistakes at tiny cost)

### What Makes This Safe?
- ✅ **Micro positions** (QTY=0.001 = max $2-3 loss per trade)
- ✅ **Circuit breakers** (auto-reduce position size on bad metrics)
- ✅ **Graduation gates** (must pass 3 of 4 criteria to scale)
- ✅ **Monitoring dashboard** (real-time health checks)
- ✅ **Three-phase progression** (validate at each tier)

### What's the Expected Timeline?
- Week 1-2: High exploration, 30-40% win rate (learning friction)
- Week 3-4: Moderate exploration, 40-50% win rate (policies improving)
- Week 5-6: Low exploration, 45-55% win rate (consistent edge)
- Week 7-8: Graduation check → Phase 2 if passed

### What's the Total Risk?
- **Phase 1:** $500-1500 worst case (500 trades × $2-3 max loss)
- **Phase 2:** $5k-15k worst case (500 trades × $20-30 max loss)
- **Phase 3:** Production risk (standard position sizing)

---

## 📞 Support & Debugging

### Bot Won't Start
1. Check **PRE_LAUNCH_CHECKLIST.md** (Section 1: Environment Setup)
2. Verify credentials: `source .env && echo $CTRADER_USERNAME`
3. Check logs: `tail -100 logs/phase0_validation/validate_*.log`
4. Test FIX connectivity: `ping h51.p.ctrader.com`

### FIX Sessions Disconnect
1. Check broker status (cTrader servers online?)
2. Verify credentials (passwords correct in .env?)
3. Check network (firewall blocking ports 5201/5202?)
4. Review FIX logs: `ls -lh logs/fix_*/`

### Win Rate Stuck at 30%
1. **Normal for first 200 trades** (high exploration)
2. Check epsilon decay: Should be 0.30 → 0.05 over time
3. Verify PER buffer sampling: High-loss trades prioritized?
4. Extend Phase 1: Give agents more time (500 → 1000 trades)

### Circuit Breakers Trip Too Often
1. Check trip rate: Should be <5% of trades
2. If >10%: Relax thresholds (export CB_SORTINO_MIN=0.5)
3. If >20%: Agents not learning, extend Phase 1
4. Review logs for specific breaker types (Sortino vs Drawdown vs ConsecLoss)

### Agents Not Trading
1. Check feasibility threshold: `export FEAS_THRESHOLD=0.2` (lower = easier)
2. Check exploration: `export EPSILON_START=0.5` (higher = more random trades)
3. Check MAX_BARS_INACTIVE: Should be 100 (not too aggressive)
4. Review logs: Are signals firing but feasibility blocking?

---

## ✅ Deployment Readiness Status

You now have:
- ✅ Complete strategy documentation (7500+ lines)
- ✅ 3 executable scripts (validate, launch, monitor)
- ✅ Pre-launch checklist (10 categories)
- ✅ Graduation criteria (4 metrics)
- ✅ Phase 3 integration complete (circuit breakers, event features)
- ✅ Living ecosystem philosophy (documented)

**Next action:** Review **PRE_LAUNCH_CHECKLIST.md**, then run `./phase0_validate_system.sh`

**Good luck building the living ecosystem!** 🚀
