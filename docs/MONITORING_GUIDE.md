# Monitoring & Visualization Guide

**Comprehensive monitoring solutions for the Adaptive RL Trading Bot**

---

## Overview

This guide covers four monitoring approaches:

1. **Terminal HUD** - Real-time terminal dashboard (immediate, lightweight)
2. **Jupyter Notebooks** - Interactive analysis (post-trade, deep dive)
3. **Prometheus + Grafana** - Production monitoring (scalable, professional)
4. **Custom Web Dashboard** - Browser-based real-time UI (optional)

---

## 1. Terminal HUD (✅ Implemented)

### Quick Start

```bash
# Run standalone HUD with simulated data
python3 hud_display.py

# Integration with live bot (coming soon)
# The bot will export metrics to data/performance_snapshot.json
```

### Features

- **Live Position Tracking**: Entry price, MFE, MAE, unrealized PnL
- **Performance Metrics**: Win rate, Sharpe, drawdown, total PnL
- **Agent Training Stats**: Buffer sizes, loss, TD-errors
- **Risk Management**: VaR, kurtosis, circuit breakers
- **Market Microstructure**: VPIN, spread, depth, imbalance

### Screenshot

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ADAPTIVE RL TRADING BOT - LIVE HUD                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

🕐 2026-01-09 15:30:45 UTC

────────────────────────────────────────────────────────────────────────────────
📊 CURRENT POSITION
────────────────────────────────────────────────────────────────────────────────
  Direction:      LONG
  Entry Price:    100000.00
  Current Price:  100150.00
  MFE:            +200.00 (+0.20%)
  MAE:            +50.00 (+0.05%)
  Unrealized PnL: +150.00 USD
  Bars Held:      12

────────────────────────────────────────────────────────────────────────────────
📈 PERFORMANCE METRICS
────────────────────────────────────────────────────────────────────────────────
  Total Trades:   47
  Win Rate:       63.8%
  Total PnL:      +2340.50 USD
  Sharpe Ratio:   1.845
  Max Drawdown:   8.70%
  Avg Win:        $125.30
  Avg Loss:       $-78.20

────────────────────────────────────────────────────────────────────────────────
🧠 AGENT TRAINING STATUS
────────────────────────────────────────────────────────────────────────────────
  TriggerAgent:
    Buffer:       1,247 experiences
    Loss:         0.0234
    TD-Error:     0.0512
  HarvesterAgent:
    Buffer:       1,289 experiences
    Loss:         0.0187
    TD-Error:     0.0398
  Last Training:  2 minutes ago
```

### Integration

To integrate with live bot, add to `ctrader_ddqn_paper.py`:

```python
# In __init__():
self.hud_exporter = HUDDataExporter()

# In on_bar_close():
self.hud_exporter.export_snapshot({
    'position': {...},
    'metrics': {...},
    'training': {...},
    'risk': {...},
    'market': {...}
})
```

---

## 2. Jupyter Notebook Analysis (✅ Implemented)

### Quick Start

```bash
# Activate virtual environment (required on Ubuntu/Debian)
source .venv/bin/activate

# Install dependencies (already done if you followed setup)
pip install jupyter plotly pandas

# Launch notebook
jupyter notebook analysis_notebook.ipynb
```

**Note**: If you get "externally-managed-environment" error, you're not in the venv. Always run `source .venv/bin/activate` first.

### Features

1. **Trade Performance**: PnL curves, cumulative returns
2. **Win/Loss Distribution**: Histogram analysis
3. **MFE/MAE Analysis**: Path efficiency scatter plots
4. **Dual-Agent Attribution**: TriggerAgent vs HarvesterAgent quality
5. **Runway Accuracy**: TriggerAgent prediction analysis
6. **Direction Analysis**: LONG vs SHORT performance
7. **Interactive Visualizations**: Plotly charts with hover data

### Notebook Sections

```
1. Load Trade Data           → CSV import
2. Performance Summary        → Key metrics
3. PnL Curve & Drawdown      → Time series
4. Win/Loss Distribution     → Histogram
5. MFE/MAE Analysis          → Scatter plot
6. Dual-Agent Attribution    → Quality breakdown
7. Runway Prediction         → Accuracy distribution
8. Direction Analysis        → LONG vs SHORT
9. Export Summary Report     → JSON export
```

### Sample Outputs

- **PnL Curve**: Interactive line chart with drawdown
- **MFE/MAE Scatter**: Color-coded by winner/loser
- **Agent Quality Pies**: EXCELLENT/GOOD/POOR breakdown
- **Runway Histogram**: Distribution of prediction accuracy

---

## 3. Prometheus + Grafana (Production Setup)

### Architecture

```
Trading Bot → Prometheus Client → Prometheus Server → Grafana
                (port 8000)         (port 9090)       (port 3000)
# Activate venv first
source .venv/bin/activate

# Install Prometheus client
```

### Step 1: Install Prometheus Client

```bash
pip install prometheus-client
```

### Step 2: Add Metrics Exporter to Bot

Create `metrics_exporter.py`:

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time

# Metrics
trades_total = Counter('trades_total', 'Total trades executed', ['direction', 'outcome'])
pnl_total = Gauge('pnl_total', 'Total PnL')
position_pnl = Gauge('position_pnl', 'Current position PnL')
sharpe_ratio = Gauge('sharpe_ratio', 'Sharpe ratio')
win_rate = Gauge('win_rate', 'Win rate percentage')
mfe_histogram = Histogram('mfe', 'Maximum Favorable Excursion')
mae_histogram = Histogram('mae', 'Maximum Adverse Excursion')

# Agent metrics
trigger_buffer_size = Gauge('trigger_buffer_size', 'TriggerAgent buffer size')
harvester_buffer_size = Gauge('harvester_buffer_size', 'HarvesterAgent buffer size')
trigger_loss = Gauge('trigger_loss', 'TriggerAgent training loss')
harvester_loss = Gauge('harvester_loss', 'HarvesterAgent training loss')

# Risk metrics
var_estimate = Gauge('var_estimate', 'Value at Risk (95%)')
kurtosis = Gauge('kurtosis', 'Excess kurtosis')
circuit_breaker = Gauge('circuit_breaker', 'Circuit breaker status (1=active)')

# Market metrics
spread = Gauge('spread', 'Bid-ask spread')
vpin = Gauge('vpin', 'VPIN')
vpin_zscore = Gauge('vpin_zscore', 'VPIN z-score')

def start_metrics_server(port=8000):
    """Start Prometheus metrics server"""
    start_http_server(port)
    print(f"✓ Metrics server started on port {port}")

def update_trade_metrics(direction: str, pnl: float, mfe: float, mae: float):
    """Update metrics after trade completion"""
    outcome = 'winner' if pnl > 0 else 'loser'
    trades_total.labels(direction=direction, outcome=outcome).inc()
    mfe_histogram.observe(mfe)
    mae_histogram.observe(mae)
```

### Step 3: Integrate with Bot

In `ctrader_ddqn_paper.py`:

```python
from metrics_exporter import start_metrics_server, update_trade_metrics, pnl_total, sharpe_ratio

# In __init__():
start_metrics_server(port=8000)

# After trade completion:
update_trade_metrics(direction, pnl, mfe, mae)
pnl_total.set(self.performance.total_pnl)
sharpe_ratio.set(self.performance.sharpe)
```

### Step 4: Install Prometheus

```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*

# Create config
cat > prometheus.yml <<EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'trading_bot'
    static_configs:
      - targets: ['localhost:8000']
EOF

# Run Prometheus
./prometheus --config.file=prometheus.yml
```

Access: http://localhost:9090

### Step 5: Install Grafana

```bash
# Install Grafana (Ubuntu/Debian)
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana

# Start Grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

Access: http://localhost:3000 (default: admin/admin)

### Step 6: Create Grafana Dashboard

1. Add Prometheus data source:
   - URL: http://localhost:9090
   
2. Import dashboard or create panels:
   - **PnL Over Time**: `rate(pnl_total[5m])`
   - **Win Rate**: `win_rate`
   - **Trades by Outcome**: `trades_total`
   - **MFE/MAE Distribution**: Histogram from `mfe` and `mae`
   - **Agent Buffer Sizes**: `trigger_buffer_size`, `harvester_buffer_size`
   - **Training Loss**: `trigger_loss`, `harvester_loss`
   - **Circuit Breaker Status**: `circuit_breaker`
   - **VPIN Alert**: `vpin_zscore > 2.0`

### Sample Grafana Panels

```json
{
  "dashboard": {
    "title": "Trading Bot Performance",
    "panels": [
      {
        "title": "Total PnL",
        "targets": [{"expr": "pnl_total"}],
        "type": "graph"
      },
      {
        "title": "Trade Count (24h)",
        "targets": [{"expr": "increase(trades_total[24h])"}],
        "type": "stat"
      },
      {
        "title": "Win Rate",
        "targets": [{"expr": "win_rate"}],
        "type": "gauge",
        "thresholds": [50, 60, 70]
      },
      {
        "title": "Agent Training Loss",
        "targets": [
          {"expr": "trigger_loss", "legendFormat": "Trigger"},
          {"expr": "harvester_loss", "legendFormat": "Harvester"}
        ],
        "type": "graph"
      }
    ]
  }
}
```

---

## 4. Custom Web Dashboard (Optional)

For a custom browser-based real-time UI:

### Tech Stack

- **Backend**: Flask or FastAPI
- **Frontend**: React + Chart.js or D3.js
- **Real-time**: WebSockets or Server-Sent Events

### Quick Start (Flask + Plotly Dash)

```python
# dashboard_app.py
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas as pd
from flask import Flask

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

app.layout = html.Div([
    html.H1('Trading Bot Dashboard'),
    dcc.Interval(id='interval', interval=1000),  # Update every 1s
    dcc.Graph(id='pnl-graph'),
    dcc.Graph(id='position-gauge'),
])

@app.callback(
    Output('pnl-graph', 'figure'),
    Input('interval', 'n_intervals')
)
def update_pnl(n):
    # Load latest data
    df = pd.read_csv('exports/bot_trades_all.csv')
    df['cumulative_pnl'] = df['pnl'].cumsum()
    
    return {
        'data': [go.Scatter(
            x=df['exit_time'],
            y=df['cumulative_pnl'],
            mode='lines'
        )],
        'layout': go.Layout(title='Cumulative PnL')
    }

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

Run: `python dashboard_app.py`  
Access: http://localhost:8050

---

## Comparison Matrix

| Feature | Terminal HUD | Jupyter | Prometheus+Grafana | Web Dashboard |
|---------|--------------|---------|-------------------|---------------|
| **Setup Time** | 5 min | 10 min | 30-60 min | 1-2 hours |
| **Real-time** | ✅ | ❌ | ✅ | ✅ |
| **Historical** | ❌ | ✅ | ✅ | ✅ |
| **Interactivity** | ❌ | ✅ | ⚠️ (limited) | ✅ |
| **Scalability** | Low | Low | High | Medium |
| **Resource Cost** | Minimal | Low | Medium | Medium |
| **Production Ready** | ❌ | ❌ | ✅ | ⚠️ |
| **Best For** | Dev/Debug | Analysis | Production | Custom needs |

---

## Recommendations

### Development Phase
- **Terminal HUD** for quick feedback
- **Jupyter Notebook** for trade analysis

### Testing Phase
- Keep Terminal HUD
- Add Prometheus metrics exporter
- Optional: Simple web dashboard

### Production Phase
- **Prometheus + Grafana** for monitoring
- **Alerts** via Grafana (Slack, email, PagerDuty)
- **Jupyter** for weekly/monthly analysis
- **Terminal HUD** for debugging

---

## Next Steps

1. **Immediate**: Run Terminal HUD and Jupyter notebook
2. **Short-term**: Add Prometheus metrics exporter to bot
3. **Medium-term**: Set up Grafana dashboards
4. **Long-term**: Add alerting and anomaly detection

---

## Files Created

- ✅ `hud_display.py` - Terminal HUD
- ✅ `analysis_notebook.ipynb` - Jupyter analysis
- ✅ `docs/MONITORING_GUIDE.md` - This guide

---

## Resources

- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/
- **Plotly**: https://plotly.com/python/
- **Dash**: https://dash.plotly.com/

---

**Ready to monitor!** Choose the tools that fit your workflow. 🚀
