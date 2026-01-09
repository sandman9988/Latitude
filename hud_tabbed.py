#!/usr/bin/env python3
"""
Tabbed Trading HUD (Heads-Up Display)
=====================================
Terminal-based live dashboard with multiple tabs for organized data display.

Tabs:
  1 - Overview (compact summary)
  2 - Performance (detailed metrics)
  3 - Training (agent stats)
  4 - Risk (risk management)
  5 - Market (microstructure)

Press 1-5 to switch tabs, q to quit.
"""

import os
import sys
import time
import threading
import select
import tty
import termios
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
from pathlib import Path


class TabbedHUD:
    """Real-time tabbed HUD for trading bot monitoring"""
    
    TABS = {
        '1': 'overview',
        '2': 'performance', 
        '3': 'training',
        '4': 'risk',
        '5': 'market'
    }
    
    TAB_ORDER = ['overview', 'performance', 'training', 'risk', 'market']
    
    TAB_NAMES = {
        'overview': '📊 Overview',
        'performance': '📈 Performance',
        'training': '🧠 Training',
        'risk': '⚠️  Risk',
        'market': '🔬 Market'
    }
    
    def __init__(self, refresh_rate: float = 1.0):
        self.refresh_rate = refresh_rate
        self.running = False
        self.thread = None
        self.current_tab = 'overview'
        
        # Data sources
        self.data_dir = Path("data")
        
        # State
        self.position = {}
        self.metrics = {}
        self.training_stats = {}
        self.risk_stats = {}
        self.market_stats = {}
        self.bot_config = {}
        self.last_update = None
        
        # Time-based metrics
        self.daily_metrics = {}
        self.weekly_metrics = {}
        self.monthly_metrics = {}
        self.lifetime_metrics = {}
        
        # Heartbeat
        self.heartbeat_idx = 0
        self.heartbeat_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
        # Terminal settings for non-blocking input
        self.old_settings = None
        
    def start(self):
        """Start HUD"""
        self.running = True
        # Set terminal to raw mode for key input
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except:
            pass
        
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop HUD"""
        self.running = False
        # Restore terminal settings
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2)
    
    def _check_input(self):
        """Check for keyboard input (non-blocking)"""
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key in self.TABS:
                    self.current_tab = self.TABS[key]
                elif key == '\t':  # Tab key to cycle forward
                    idx = self.TAB_ORDER.index(self.current_tab)
                    self.current_tab = self.TAB_ORDER[(idx + 1) % len(self.TAB_ORDER)]
                elif key == '\x1b':  # Escape sequence (for Shift+Tab)
                    # Read the rest of the escape sequence
                    if select.select([sys.stdin], [], [], 0.01)[0]:
                        seq = sys.stdin.read(2)
                        if seq == '[Z':  # Shift+Tab
                            idx = self.TAB_ORDER.index(self.current_tab)
                            self.current_tab = self.TAB_ORDER[(idx - 1) % len(self.TAB_ORDER)]
                elif key.lower() == 'q':
                    self.running = False
        except:
            pass
    
    def _update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                self._check_input()
                self._refresh_data()
                self._render()
                time.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\033[0mHUD Error: {e}")
                time.sleep(2)
    
    def _refresh_data(self):
        """Refresh all data from bot exports"""
        self.last_update = datetime.now()
        self.heartbeat_idx = (self.heartbeat_idx + 1) % len(self.heartbeat_chars)
        
        # Bot config
        self._load_json("bot_config.json", "bot_config")
        
        # Position
        self._load_json("current_position.json", "position")
        
        # Performance
        perf_file = self.data_dir / "performance_snapshot.json"
        if perf_file.exists():
            try:
                with open(perf_file) as f:
                    data = json.load(f)
                    self.daily_metrics = data.get('daily', {})
                    self.weekly_metrics = data.get('weekly', {})
                    self.monthly_metrics = data.get('monthly', {})
                    self.lifetime_metrics = data.get('lifetime', {})
            except:
                pass
        
        # Training stats
        self._load_json("training_stats.json", "training_stats")
        
        # Risk metrics
        risk_file = self.data_dir / "risk_metrics.json"
        if risk_file.exists():
            try:
                with open(risk_file) as f:
                    data = json.load(f)
                    self.risk_stats = data
                    self.market_stats = {
                        'vpin': data.get('vpin', 0.0),
                        'vpin_z': data.get('vpin_zscore', 0.0),
                        'spread': data.get('spread', 0.0),
                        'imbalance': data.get('imbalance', 0.0),
                        'depth_bid': data.get('depth_bid', 0.0),
                        'depth_ask': data.get('depth_ask', 0.0)
                    }
            except:
                pass
    
    def _load_json(self, filename: str, attr: str):
        """Load JSON file into attribute"""
        filepath = self.data_dir / filename
        if filepath.exists():
            try:
                with open(filepath) as f:
                    setattr(self, attr, json.load(f))
            except:
                pass
    
    def _render(self):
        """Render current tab"""
        os.system('clear' if os.name != 'nt' else 'cls')
        
        # Header
        self._render_header()
        
        # Tab bar
        self._render_tab_bar()
        
        # Tab content
        if self.current_tab == 'overview':
            self._render_overview()
        elif self.current_tab == 'performance':
            self._render_performance()
        elif self.current_tab == 'training':
            self._render_training()
        elif self.current_tab == 'risk':
            self._render_risk()
        elif self.current_tab == 'market':
            self._render_market()
        
        # Footer
        self._render_footer()
    
    def _render_header(self):
        """Render header"""
        heartbeat = self.heartbeat_chars[self.heartbeat_idx]
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 18 + "ADAPTIVE RL TRADING BOT - TABBED HUD" + " " * 22 + "║")
        print("╚" + "═" * 78 + "╝")
        
        # Bot info
        symbol = self.bot_config.get('symbol', 'UNKNOWN')
        tf = self.bot_config.get('timeframe', '1m')
        uptime = self.bot_config.get('uptime_seconds', 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        
        price = self.position.get('current_price', 0)
        now = self.last_update or datetime.now()
        
        print(f"\n🎯 {symbol} @ {tf}    💰 {price:.2f}    ⏱  {hours:02d}h {minutes:02d}m")
        print(f"{heartbeat} {now.strftime('%Y-%m-%d %H:%M:%S')} UTC (Live)")
    
    def _render_tab_bar(self):
        """Render tab navigation bar"""
        print("\n" + "─" * 80)
        
        tabs = []
        for key, name in [('1', 'Overview'), ('2', 'Performance'), ('3', 'Training'), 
                          ('4', 'Risk'), ('5', 'Market')]:
            tab_id = self.TABS[key]
            if tab_id == self.current_tab:
                tabs.append(f"\033[7m [{key}] {name} \033[0m")  # Inverted
            else:
                tabs.append(f" [{key}] {name} ")
        
        print("".join(tabs))
        print("─" * 80)
    
    def _render_overview(self):
        """Render overview tab - compact summary"""
        # Position
        print("\n\033[1m📊 POSITION\033[0m")
        direction = self.position.get('direction', 'FLAT')
        entry = self.position.get('entry_price', 0)
        current = self.position.get('current_price', 0)
        pnl = self.position.get('unrealized_pnl', 0)
        bars = self.position.get('bars_held', 0)
        
        dir_color = '\033[92m' if direction == 'LONG' else ('\033[91m' if direction == 'SHORT' else '\033[93m')
        pnl_color = self._pnl_color(pnl)
        
        print(f"  {dir_color}{direction}\033[0m @ {entry:.2f} → {current:.2f}  |  "
              f"PnL: {pnl_color}{pnl:+.2f}\033[0m  |  Bars: {bars}")
        
        # Quick metrics
        print("\n\033[1m📈 TODAY'S STATS\033[0m")
        d = self.daily_metrics
        trades = d.get('total_trades', 0)
        wr = d.get('win_rate', 0) * 100
        day_pnl = d.get('total_pnl', 0)
        print(f"  Trades: {trades}  |  Win Rate: {wr:.1f}%  |  PnL: {self._pnl_color(day_pnl)}{day_pnl:+.2f}\033[0m")
        
        # Risk snapshot
        print("\n\033[1m⚠️  RISK STATUS\033[0m")
        cb = self.risk_stats.get('circuit_breaker', 'INACTIVE')
        regime = self.risk_stats.get('regime', 'UNKNOWN')
        vol = self.risk_stats.get('realized_vol', 0) * 100
        feas = self.risk_stats.get('feasibility', 0.5)
        
        cb_status = "\033[91m● ACTIVE\033[0m" if cb == 'ACTIVE' else "\033[92m● OK\033[0m"
        feas_color = '\033[92m' if feas > 0.7 else ('\033[93m' if feas > 0.5 else '\033[91m')
        
        print(f"  Circuit: {cb_status}  |  Regime: {regime}  |  Vol: {vol:.2f}%  |  "
              f"Feasibility: {feas_color}{feas:.2f}\033[0m")
        
        # Training snapshot
        print("\n\033[1m🧠 AGENT STATUS\033[0m")
        trig_buf = self.training_stats.get('trigger_buffer_size', 0)
        harv_buf = self.training_stats.get('harvester_buffer_size', 0)
        total_agents = self.training_stats.get('total_agents', 0)
        
        if total_agents > 0:
            print(f"  Arena: {total_agents} agents  |  Trigger: {trig_buf:,} exp  |  Harvester: {harv_buf:,} exp")
        else:
            print(f"  Trigger Buffer: {trig_buf:,}  |  Harvester Buffer: {harv_buf:,}")
        
        # Market snapshot
        print("\n\033[1m🔬 MARKET\033[0m")
        spread = self.market_stats.get('spread', 0)
        vpin = self.market_stats.get('vpin', 0)
        vpin_z = self.market_stats.get('vpin_z', 0)
        imb = self.market_stats.get('imbalance', 0)
        
        vpin_status = "\033[91m⚠️ HIGH\033[0m" if abs(vpin_z) > 2.0 else "\033[92m✓\033[0m"
        
        print(f"  Spread: {spread:.5f}  |  VPIN: {vpin:.3f} {vpin_status}  |  Imbalance: {imb:+.3f}")
    
    def _render_performance(self):
        """Render detailed performance metrics"""
        print("\n\033[1m📈 PERFORMANCE METRICS\033[0m\n")
        
        # Column headers
        print(f"  {'Period':<10} {'Trades':>8} {'Win%':>8} {'PnL':>12} {'Sharpe':>8} {'Sortino':>8} {'Omega':>8} {'MaxDD':>10}")
        print("  " + "─" * 76)
        
        for label, metrics in [('📅 Daily', self.daily_metrics), 
                                ('📆 Weekly', self.weekly_metrics),
                                ('🗓️ Monthly', self.monthly_metrics),
                                ('∞ Lifetime', self.lifetime_metrics)]:
            trades = metrics.get('total_trades', 0)
            wr = metrics.get('win_rate', 0) * 100
            pnl = metrics.get('total_pnl', 0)
            sharpe = metrics.get('sharpe_ratio', 0)
            sortino = metrics.get('sortino_ratio', 0)
            omega = metrics.get('omega_ratio', 0)
            maxdd = metrics.get('max_drawdown', 0) * 100
            
            pnl_color = self._pnl_color(pnl)
            
            print(f"  {label:<10} {trades:>8} {wr:>7.1f}% {pnl_color}{pnl:>+11.2f}\033[0m "
                  f"{sharpe:>8.2f} {sortino:>8.2f} {omega:>8.2f} {maxdd:>9.2f}%")
        
        # Additional stats if available
        print("\n\033[1m📊 ADDITIONAL METRICS\033[0m\n")
        
        lt = self.lifetime_metrics
        print(f"  Best Trade:       {lt.get('best_trade', 0):+.2f}")
        print(f"  Worst Trade:      {lt.get('worst_trade', 0):+.2f}")
        print(f"  Avg Trade:        {lt.get('avg_trade', 0):+.2f}")
        print(f"  Profit Factor:    {lt.get('profit_factor', 0):.2f}")
        print(f"  Expectancy:       {lt.get('expectancy', 0):+.4f}")
        print(f"  Avg Win:          {lt.get('avg_win', 0):+.2f}")
        print(f"  Avg Loss:         {lt.get('avg_loss', 0):+.2f}")
        print(f"  Consecutive Wins: {lt.get('max_consec_wins', 0)}")
        print(f"  Consecutive Loss: {lt.get('max_consec_losses', 0)}")
    
    def _render_training(self):
        """Render agent training status"""
        print("\n\033[1m🧠 AGENT TRAINING STATUS\033[0m\n")
        
        ts = self.training_stats
        
        # Arena info
        total_agents = ts.get('total_agents', 0)
        if total_agents > 0:
            diversity = ts.get('arena_diversity', {})
            trig_div = diversity.get('trigger_diversity', 0) if isinstance(diversity, dict) else 0
            harv_div = diversity.get('harvester_diversity', 0) if isinstance(diversity, dict) else 0
            agreement = ts.get('last_agreement_score', 0)
            consensus = ts.get('consensus_mode', 'unknown')
            
            print(f"  \033[1m🤖 MULTI-AGENT ARENA\033[0m")
            print(f"    Total Agents:     {total_agents}")
            print(f"    Consensus Mode:   {consensus}")
            print(f"    Agreement Score:  {agreement:.3f}")
            print(f"    Trigger Diversity:{trig_div:.3f}")
            print(f"    Harvester Div:    {harv_div:.3f}")
            print()
        
        # Trigger agents
        print(f"  \033[1m🎯 TRIGGER AGENTS (Entry)\033[0m")
        trig_buf = ts.get('trigger_buffer_size', 0)
        trig_loss = ts.get('trigger_loss', 0)
        trig_td = ts.get('trigger_td_error', 0)
        trig_eps = ts.get('trigger_epsilon', 0)
        
        buf_color = '\033[92m' if trig_buf > 1000 else ('\033[93m' if trig_buf > 100 else '\033[91m')
        
        print(f"    Experience Buffer: {buf_color}{trig_buf:>8,}\033[0m")
        print(f"    Training Loss:     {trig_loss:>12.6f}")
        if trig_td > 0:
            print(f"    TD-Error:          {trig_td:>12.6f}")
        if trig_eps > 0:
            print(f"    Epsilon:           {trig_eps:>12.4f}")
        
        # Progress bar for buffer
        max_buf = 10000
        pct = min(trig_buf / max_buf, 1.0)
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"    Buffer Fill:       [{bar}] {pct*100:.0f}%")
        
        print()
        
        # Harvester agents
        print(f"  \033[1m🌾 HARVESTER AGENTS (Exit)\033[0m")
        harv_buf = ts.get('harvester_buffer_size', 0)
        harv_loss = ts.get('harvester_loss', 0)
        harv_td = ts.get('harvester_td_error', 0)
        harv_eps = ts.get('harvester_epsilon', 0)
        
        buf_color = '\033[92m' if harv_buf > 1000 else ('\033[93m' if harv_buf > 100 else '\033[91m')
        
        print(f"    Experience Buffer: {buf_color}{harv_buf:>8,}\033[0m")
        print(f"    Training Loss:     {harv_loss:>12.6f}")
        if harv_td > 0:
            print(f"    TD-Error:          {harv_td:>12.6f}")
        if harv_eps > 0:
            print(f"    Epsilon:           {harv_eps:>12.4f}")
        
        pct = min(harv_buf / max_buf, 1.0)
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"    Buffer Fill:       [{bar}] {pct*100:.0f}%")
        
        print()
        
        # Training status
        last_train = ts.get('last_training_time', 'Never')
        print(f"  \033[1m⏱  TRAINING SCHEDULE\033[0m")
        print(f"    Last Training:     {last_train}")
        print(f"    Training Enabled:  {self.bot_config.get('training_enabled', False)}")
    
    def _render_risk(self):
        """Render risk management details"""
        print("\n\033[1m⚠️  RISK MANAGEMENT\033[0m\n")
        
        rs = self.risk_stats
        
        # Circuit breaker status
        cb = rs.get('circuit_breaker', 'INACTIVE')
        if cb == 'ACTIVE':
            print(f"  \033[91m╔════════════════════════════════════════╗")
            print(f"  ║     ⚠️  CIRCUIT BREAKER ACTIVE ⚠️       ║")
            print(f"  ╚════════════════════════════════════════╝\033[0m\n")
        else:
            print(f"  \033[92m✓ Circuit Breaker: INACTIVE\033[0m\n")
        
        # VaR and tail risk
        print(f"  \033[1m📉 TAIL RISK\033[0m")
        var = rs.get('var', 0)
        kurtosis = rs.get('kurtosis', 0)
        
        kurt_color = '\033[91m' if kurtosis > 3 else '\033[92m'
        
        print(f"    VaR (95%):         {var:>12.6f}")
        print(f"    Kurtosis:          {kurt_color}{kurtosis:>12.2f}\033[0m (>3 = fat tails)")
        
        print()
        
        # Volatility and regime
        print(f"  \033[1m📊 VOLATILITY & REGIME\033[0m")
        vol = rs.get('realized_vol', 0) * 100
        regime = rs.get('regime', 'UNKNOWN')
        zeta = rs.get('regime_zeta', 1.0)
        
        regime_colors = {
            'TRENDING': '\033[92m',
            'MEAN_REVERTING': '\033[93m',
            'TRANSITIONAL': '\033[94m',
            'UNKNOWN': '\033[90m'
        }
        regime_color = regime_colors.get(regime, '\033[90m')
        
        print(f"    Realized Vol (RS): {vol:>10.2f}%")
        print(f"    Market Regime:     {regime_color}{regime}\033[0m")
        print(f"    Regime Confidence: {zeta:>10.2f} (ζ)")
        
        print()
        
        # Path geometry
        print(f"  \033[1m📐 PATH GEOMETRY\033[0m")
        eff = rs.get('efficiency', 0)
        gamma = rs.get('gamma', 0)
        jerk = rs.get('jerk', 0)
        runway = rs.get('runway', 0.5)
        feas = rs.get('feasibility', 0.5)
        
        feas_color = '\033[92m' if feas > 0.7 else ('\033[93m' if feas > 0.5 else '\033[91m')
        
        print(f"    Efficiency:        {eff:>10.3f} (path directness)")
        print(f"    Gamma (γ):         {gamma:>+10.3f} (acceleration)")
        print(f"    Jerk:              {jerk:>+10.3f} (accel change rate)")
        print(f"    Runway:            {runway:>10.3f} (1/vol pressure)")
        print(f"    Entry Feasibility: {feas_color}{feas:>10.3f}\033[0m")
        
        # Feasibility gauge
        bar_len = 40
        feas_pct = max(0, min(1, feas))
        filled = int(bar_len * feas_pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n    [{bar}]")
        print(f"     LOW                                HIGH")
    
    def _render_market(self):
        """Render market microstructure"""
        print("\n\033[1m🔬 MARKET MICROSTRUCTURE\033[0m\n")
        
        ms = self.market_stats
        rs = self.risk_stats
        
        # Spread analysis
        print(f"  \033[1m💹 SPREAD & LIQUIDITY\033[0m")
        spread = ms.get('spread', 0)
        depth_bid = ms.get('depth_bid', 0)
        depth_ask = ms.get('depth_ask', 0)
        
        print(f"    Bid-Ask Spread:    {spread:>12.5f}")
        print(f"    Bid Depth:         {depth_bid:>12.2f}")
        print(f"    Ask Depth:         {depth_ask:>12.2f}")
        
        # Depth imbalance visualization
        total_depth = depth_bid + depth_ask
        if total_depth > 0:
            bid_pct = depth_bid / total_depth
            ask_pct = depth_ask / total_depth
            bar_len = 40
            bid_bar = int(bar_len * bid_pct)
            ask_bar = bar_len - bid_bar
            print(f"\n    Depth: \033[92m{'█' * bid_bar}\033[91m{'█' * ask_bar}\033[0m")
            print(f"           {'BID':<20}{'ASK':>20}")
        
        print()
        
        # Order flow toxicity
        print(f"  \033[1m☢️  ORDER FLOW TOXICITY (VPIN)\033[0m")
        vpin = ms.get('vpin', 0)
        vpin_z = ms.get('vpin_z', 0)
        
        if abs(vpin_z) > 2.0:
            vpin_status = "\033[91m⚠️  HIGH TOXICITY\033[0m"
        elif abs(vpin_z) > 1.0:
            vpin_status = "\033[93m⚡ ELEVATED\033[0m"
        else:
            vpin_status = "\033[92m✓ NORMAL\033[0m"
        
        print(f"    VPIN Value:        {vpin:>12.4f}")
        print(f"    VPIN Z-Score:      {vpin_z:>+12.2f}")
        print(f"    Status:            {vpin_status}")
        
        # VPIN gauge
        bar_len = 40
        # Normalize z-score to 0-1 range (-3 to +3 -> 0 to 1)
        z_norm = (vpin_z + 3) / 6
        z_norm = max(0, min(1, z_norm))
        pos = int(bar_len * z_norm)
        gauge = "░" * pos + "│" + "░" * (bar_len - pos - 1)
        print(f"\n    Z: [{gauge}]")
        print(f"       -3              0              +3")
        
        print()
        
        # Order imbalance
        print(f"  \033[1m⚖️  ORDER IMBALANCE\033[0m")
        imbalance = ms.get('imbalance', 0)
        
        if imbalance > 0.3:
            imb_status = "\033[92m🔺 BUY PRESSURE\033[0m"
        elif imbalance < -0.3:
            imb_status = "\033[91m🔻 SELL PRESSURE\033[0m"
        else:
            imb_status = "\033[93m⚖️  BALANCED\033[0m"
        
        print(f"    Imbalance:         {imbalance:>+12.4f}")
        print(f"    Status:            {imb_status}")
        
        # Imbalance visualization
        bar_len = 40
        mid = bar_len // 2
        imb_scaled = int(mid * imbalance)  # -1 to +1 -> -20 to +20
        
        if imbalance >= 0:
            bar = " " * mid + "\033[92m" + "█" * imb_scaled + "\033[0m" + " " * (mid - imb_scaled)
        else:
            bar = " " * (mid + imb_scaled) + "\033[91m" + "█" * (-imb_scaled) + "\033[0m" + " " * mid
        
        print(f"\n    [{bar}]")
        print(f"     SELL              ↕              BUY")
    
    def _render_footer(self):
        """Render footer with controls"""
        print("\n" + "─" * 80)
        print("  [1-5] Switch tabs  |  [Tab] Next  |  [Shift+Tab] Prev  |  [q] Quit")
        print("─" * 80)
    
    def _pnl_color(self, pnl: float) -> str:
        """Return color code for PnL"""
        if pnl > 0:
            return "\033[92m"
        elif pnl < 0:
            return "\033[91m"
        return "\033[93m"


def main():
    """Run tabbed HUD"""
    print("Starting Tabbed HUD...")
    print("Reading from: data/*.json")
    print()
    
    hud = TabbedHUD(refresh_rate=1.0)
    
    try:
        hud.start()
        while hud.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        hud.stop()
        print("\n\nHUD stopped.")


if __name__ == "__main__":
    main()
