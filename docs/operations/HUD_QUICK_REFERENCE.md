# Tabbed HUD Quick Reference Card

**Last Updated:** 2026-04-23  
**Status:** Active  
**Audience:** Operators, Developers

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Overview tab |
| `2` | Performance tab |
| `3` | Training tab |
| `4` | Risk tab |
| `5` | Market tab |
| `6` | Decision Log tab |
| `7` | Trades tab |
| `Tab` | Next tab |
| `Shift+Tab` | Previous tab |
| `←` / `→` | Cycle tabs left/right |
| `s` | Select symbol/timeframe preset |
| `e` | Set/Clear stats epoch cutoff |
| `r` | Review/reset tripped circuit breakers |
| `h` | Help screen |
| `q` or `Ctrl+X` | Quit HUD |

## Tab Contents

### Tab 1: Overview
- Current bot snapshot and current position
- Fleet table (`ALL BOTS`) with canonical timeframe labels (`M1`, `M5`, `M15`, `M30`, `M60`, `M240`)
- Session metrics and Symbol/TF snapshot
- Health/freshness context

### Tab 2: Performance
- Period metrics (24h / 7d / 30d / lifetime or stats-epoch scope)
- Mode breakdown and symbol/TF/mode breakdown
- Trade quality and runway convergence panels

### Tab 3: Training
- Trigger + Harvester live training stats
- Runway model reliability diagnostics
- RL dynamic confidence floors:
  - Trigger: `entry_conf_dynamic_floor`
  - Harvester: `exit_conf_dynamic_floor`
- Arena/learning health blocks

### Tab 4: Risk
- Circuit breaker status
- VaR, volatility, regime, feasibility
- Risk budget and gating views

### Tab 5: Market
- Spread/depth/imbalance/VPIN synthesis
- Market feed freshness (`LIVE` / `AGING` / `STALE`)

### Tab 6: Decision Log
- Recent decision records and confidence context
- Session/trade correlation fields for review

### Tab 7: Trades
- Closed-trade ledger with pagination
- Capture% uses normalized `capture_ratio` and preserves negative values on losses
- Drill-down detail view per trade

## Footer Freshness States

- `Data fresh`: age <= 5s
- `Data aging`: 5s < age <= 15s
- `Data stale`: age > 15s

## Data Sources (Primary)

HUD reads from `data/` with per-bot preference where available:
- `paper_stats_<SYMBOL>_M<TF>.json`
- `training_stats_<SYMBOL>_M<TF>.json`
- `risk_metrics_<SYMBOL>_M<TF>.json`
- `current_position_<SYMBOL>_M<TF>.json`
- `trade_log.jsonl`
- `logs/audit/decisions.jsonl`

## Quick Start

```bash
# Attach HUD to running fleet
./run.sh --hud-only

# Start/restart fleet + HUD autostart (interactive shell)
./run.sh universe
```
