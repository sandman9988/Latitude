#!/usr/bin/env python3
"""
Real-Time Trading HUD (Heads-Up Display)
========================================
Terminal-based live dashboard for monitoring trading bot performance.

Features:
- Live position tracking (entry, MFE, MAE, PnL)
- Performance metrics (win rate, Sharpe, drawdown)
- Agent training stats (buffer sizes, TD-errors)
- Risk metrics (VaR, kurtosis, circuit breakers)
- Market microstructure (VPIN, spread, depth)
- Heartbeat indicator
- Time-based metrics (daily, weekly, monthly, lifetime)
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
from pathlib import Path


class TradingHUD:
    """Real-time HUD for trading bot monitoring"""
    
    def __init__(self, refresh_rate: float = 1.0):
        """
        Initialize HUD
        
        Args:
            refresh_rate: Update frequency in seconds
        """
        self.refresh_rate = refresh_rate
        self.running = False
        self.thread = None
        
        # Data sources
        self.data_dir = Path("data")
        self.log_dir = Path("logs/python")
        
        # State
        self.position = None
        self.metrics = {}
        self.training_stats = {}
        self.risk_stats = {}
        self.market_stats = {}
        self.last_update = None
        
        # Bot configuration
        self.symbol = "UNKNOWN"
        self.timeframe = "1m"
        self.bot_uptime = None
        
        # Heartbeat
        self.heartbeat_char_index = 0
        self.heartbeat_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
        # Time-based metrics
        self.daily_metrics = {}
        self.weekly_metrics = {}
        self.monthly_metrics = {}
        self.lifetime_metrics = {}
        
    def start(self):
        """Start HUD update thread"""
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop HUD updates"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                self._refresh_data()
                self._render()
                time.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"HUD Error: {e}")
                time.sleep(5)
    
    def _refresh_data(self):
        """Refresh data from bot outputs"""
        # This would read from shared memory, files, or Redis in production
        # For now, simulate with placeholder data
        self.last_update = datetime.now()
        
        # Update heartbeat
        self.heartbeat_char_index = (self.heartbeat_char_index + 1) % len(self.heartbeat_chars)
        
        # Read performance data if available
        perf_file = self.data_dir / "performance_snapshot.json"
        if perf_file.exists():
            try:
                with open(perf_file) as f:
                    data = json.load(f)
                    self.daily_metrics = data.get('daily', {})
                    self.weekly_metrics = data.get('weekly', {})
                    self.monthly_metrics = data.get('monthly', {})
                    self.lifetime_metrics = data.get('lifetime', {})
                    # Set main metrics to lifetime for backward compatibility
                    self.metrics = self.lifetime_metrics
            except:
                pass
        
        # Read position data
        pos_file = self.data_dir / "current_position.json"
        if pos_file.exists():
            try:
                with open(pos_file) as f:
                    self.position = json.load(f)
            except:
                pass
        
        # Read training stats
        training_file = self.data_dir / "training_stats.json"
        if training_file.exists():
            try:
                with open(training_file) as f:
                    self.training_stats = json.load(f)
            except:
                pass
        
        # Read risk metrics
        risk_file = self.data_dir / "risk_metrics.json"
        if risk_file.exists():
            try:
                with open(risk_file) as f:
                    data = json.load(f)
                    self.risk_stats = {
                        'var': data.get('var', 0.0),
                        'kurtosis': data.get('kurtosis', 0.0),
                        'circuit_breaker': data.get('circuit_breaker', 'INACTIVE'),
                        'volatility': data.get('realized_vol', 0.0),
                        'regime': data.get('regime', 'UNKNOWN'),
                        'regime_zeta': data.get('regime_zeta', 1.0),
                        'efficiency': data.get('efficiency', 0.0),
                        'gamma': data.get('gamma', 0.0),
                        'jerk': data.get('jerk', 0.0),
                        'runway': data.get('runway', 0.5),
                        'feasibility': data.get('feasibility', 0.5)
                    }
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
        
        # Read bot configuration
        config_file = self.data_dir / "bot_config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    self.symbol = config.get('symbol', 'UNKNOWN')
                    self.timeframe = config.get('timeframe', '1m')
                    self.bot_uptime = config.get('uptime_seconds', 0)
            except:
                pass
    
    def _render(self):
        """Render HUD to terminal"""
        # Clear screen
        os.system('clear' if os.name != 'nt' else 'cls')
        
        # Header with heartbeat
        heartbeat = self.heartbeat_chars[self.heartbeat_char_index]
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "ADAPTIVE RL TRADING BOT - LIVE HUD" + " " * 24 + "║")
        print("╚" + "═" * 78 + "╝")
        
        # Bot info line
        uptime_str = ""
        if self.bot_uptime:
            hours = int(self.bot_uptime // 3600)
            minutes = int((self.bot_uptime % 3600) // 60)
            uptime_str = f"⏱  Uptime: {hours:02d}h {minutes:02d}m"
        
        print(f"\n🎯 {self.symbol} @ {self.timeframe}    {uptime_str}")
        
        # Timestamp with heartbeat
        now = self.last_update or datetime.now()
        print(f"{heartbeat} {now.strftime('%Y-%m-%d %H:%M:%S')} UTC (Live)")
        
        # Position Section
        print("\n" + "─" * 80)
        print("📊 CURRENT POSITION")
        print("─" * 80)
        
        if self.position:
            direction = self.position.get('direction', 'FLAT')
            entry_price = self.position.get('entry_price', 0)
            current_price = self.position.get('current_price', 0)
            mfe = self.position.get('mfe', 0)
            mae = self.position.get('mae', 0)
            unrealized_pnl = self.position.get('unrealized_pnl', 0)
            bars_held = self.position.get('bars_held', 0)
            
            pnl_color = self._color_pnl(unrealized_pnl)
            
            print(f"  Direction:     {direction}")
            print(f"  Entry Price:   {entry_price:.2f}")
            print(f"  Current Price: {current_price:.2f}")
            print(f"  MFE:           {mfe:+.4f} ({mfe/entry_price*100:+.2f}%)" if entry_price > 0 else "  MFE:           N/A")
            print(f"  MAE:           {mae:+.4f} ({mae/entry_price*100:+.2f}%)" if entry_price > 0 else "  MAE:           N/A")
            print(f"  Unrealized PnL: {pnl_color}{unrealized_pnl:+.2f}\033[0m USD")
            print(f"  Bars Held:     {bars_held}")
        else:
            print("  Status: FLAT (no open position)")
        
        # Performance Section - Time-based
        print("\n" + "─" * 80)
        print("📈 PERFORMANCE METRICS")
        print("─" * 80)
        
        # Display all time periods
        self._render_timeframe("Daily", self.daily_metrics)
        print()
        self._render_timeframe("Weekly", self.weekly_metrics)
        print()
        self._render_timeframe("Monthly", self.monthly_metrics)
        print()
        self._render_timeframe("Lifetime", self.lifetime_metrics)
        
        # Training Stats Section
        print("\n" + "─" * 80)
        print("🧠 AGENT TRAINING STATUS")
        print("─" * 80)
        
        trigger_buffer = self.training_stats.get('trigger_buffer_size', 0)
        harvester_buffer = self.training_stats.get('harvester_buffer_size', 0)
        trigger_loss = self.training_stats.get('trigger_loss', 0.0)
        harvester_loss = self.training_stats.get('harvester_loss', 0.0)
        trigger_td_error = self.training_stats.get('trigger_td_error', 0.0)
        harvester_td_error = self.training_stats.get('harvester_td_error', 0.0)
        last_training = self.training_stats.get('last_training_time', 'Never')
        
        # Arena-specific stats
        total_agents = self.training_stats.get('total_agents', 0)
        arena_diversity = self.training_stats.get('arena_diversity', 0.0)
        last_agreement = self.training_stats.get('last_agreement_score', 0.0)
        
        if total_agents > 0:
            print(f"  🤖 Multi-Agent Arena ({total_agents} agents)")
            print(f"    Diversity Score:  {arena_diversity:.3f}")
            print(f"    Last Agreement:   {last_agreement:.3f}")
            print()
        
        print(f"  Trigger Agents (Entry):")
        print(f"    Buffer:    {trigger_buffer:,} experiences")
        print(f"    Loss:      {trigger_loss:.4f}")
        if trigger_td_error > 0:
            print(f"    TD-Error:  {trigger_td_error:.4f}")
        
        print(f"\n  Harvester Agents (Exit):")
        print(f"    Buffer:    {harvester_buffer:,} experiences")
        print(f"    Loss:      {harvester_loss:.4f}")
        if harvester_td_error > 0:
            print(f"    TD-Error:  {harvester_td_error:.4f}")
        
        if last_training != 'Never':
            print(f"\n  Last Training: {last_training}")
        
        # Risk Section
        print("\n" + "─" * 80)
        print("⚠️  RISK MANAGEMENT")
        print("─" * 80)
        
        var = self.risk_stats.get('var', 0.0)
        kurtosis = self.risk_stats.get('kurtosis', 0.0)
        circuit_breaker = self.risk_stats.get('circuit_breaker', 'INACTIVE')
        vol = self.risk_stats.get('volatility', 0.0)
        regime = self.risk_stats.get('regime', 'UNKNOWN')
        regime_zeta = self.risk_stats.get('regime_zeta', 1.0)
        
        breaker_status = "\033[91m●\033[0m ACTIVE" if circuit_breaker == 'ACTIVE' else "\033[92m●\033[0m INACTIVE"
        
        # Regime color coding
        regime_color = {
            'TRENDING': '\033[92m',      # Green
            'MEAN_REVERTING': '\033[93m', # Yellow
            'TRANSITIONAL': '\033[94m',   # Blue
            'UNKNOWN': '\033[90m'          # Gray
        }.get(regime, '\033[90m')
        
        print(f"  VaR (95%):         {var:.6f}")
        print(f"  Kurtosis:          {kurtosis:.2f}")
        print(f"  Circuit Breaker:   {breaker_status}")
        print(f"  Realized Vol (RS): {vol*100:.2f}%")
        print(f"  Regime:            {regime_color}{regime}\033[0m (ζ={regime_zeta:.2f})")
        
        # Path Geometry (from handbook)
        print("\n" + "─" * 80)
        print("📐 PATH GEOMETRY (Entry Trigger)")
        print("─" * 80)
        
        efficiency = self.risk_stats.get('efficiency', 0.0)
        gamma = self.risk_stats.get('gamma', 0.0)
        jerk = self.risk_stats.get('jerk', 0.0)
        runway = self.risk_stats.get('runway', 0.5)
        feasibility = self.risk_stats.get('feasibility', 0.5)
        
        # Color code feasibility
        feas_color = '\033[92m' if feasibility > 0.7 else ('\033[93m' if feasibility > 0.5 else '\033[91m')
        
        print(f"  Efficiency:    {efficiency:.3f} (path direct/total)")
        print(f"  Gamma (γ):     {gamma:+.3f} (acceleration)")
        print(f"  Jerk:          {jerk:+.3f} (rate of accel change)")
        print(f"  Runway:        {runway:.3f} (1/volatility pressure)")
        print(f"  Feasibility:   {feas_color}{feasibility:.3f}\033[0m (composite entry score)")
        
        # Market Microstructure
        print("\n" + "─" * 80)
        print("🔬 MARKET MICROSTRUCTURE")
        print("─" * 80)
        
        spread = self.market_stats.get('spread', 0.0)
        vpin = self.market_stats.get('vpin', 0.0)
        vpin_z = self.market_stats.get('vpin_z', 0.0)
        imbalance = self.market_stats.get('imbalance', 0.0)
        depth_bid = self.market_stats.get('depth_bid', 0.0)
        depth_ask = self.market_stats.get('depth_ask', 0.0)
        
        vpin_status = "🔴 HIGH" if abs(vpin_z) > 2.0 else "🟢 NORMAL"
        
        print(f"  Spread:       {spread:.5f}")
        print(f"  VPIN:         {vpin:.4f} (z={vpin_z:+.2f}) {vpin_status}")
        print(f"  Imbalance:    {imbalance:+.4f}")
        print(f"  Depth (bid):  {depth_bid:.2f}")
        print(f"  Depth (ask):  {depth_ask:.2f}")
        
        # Footer
        print("\n" + "─" * 80)
        print("Press Ctrl+C to exit")
        print("─" * 80)
    
    def _color_pnl(self, pnl: float) -> str:
        """Return color code for PnL"""
        if pnl > 0:
            return "\033[92m"  # Green
        elif pnl < 0:
            return "\033[91m"  # Red
        return "\033[93m"  # Yellow
    
    def _color_sharpe(self, sharpe: float) -> str:
        """Return color code for Sharpe ratio"""
        if sharpe > 2.0:
            return "\033[92m"  # Green
        elif sharpe > 1.0:
            return "\033[93m"  # Yellow
        return "\033[91m"  # Red
    
    def _render_timeframe(self, label: str, metrics: Dict[str, Any]):
        """Render performance metrics for a specific timeframe"""
        if not metrics:
            # Use simulated data if no real data
            metrics = self.metrics
        
        trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0.0)
        pnl = metrics.get('total_pnl', 0.0)
        sharpe = metrics.get('sharpe_ratio', 0.0)
        sortino = metrics.get('sortino_ratio', 0.0)
        omega = metrics.get('omega_ratio', 0.0)
        max_dd = metrics.get('max_drawdown', 0.0)
        
        pnl_color = self._color_pnl(pnl)
        sharpe_color = self._color_sharpe(sharpe)
        
        # Format label with emoji
        emoji_map = {
            'Daily': '📅',
            'Weekly': '📆',
            'Monthly': '🗓️',
            'Lifetime': '∞'
        }
        emoji = emoji_map.get(label, '📊')
        
        print(f"  {emoji} {label:8s} → Trades: {trades:3d} | WR: {win_rate*100:5.1f}% | "
              f"PnL: {pnl_color}{pnl:+8.2f}\033[0m | Sharpe: {sharpe_color}{sharpe:5.2f}\033[0m | "
              f"Sortino: {sortino:5.2f} | Ω: {omega:5.2f}")


def main():
    """Run HUD in LIVE mode - reads data from bot exports"""
    print("╔═══════════════════════════════════════════════════════╗")
    print("║      TRADING HUD - LIVE MODE (Bot Data)              ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    print("📁 Reading from: data/*.json")
    print("⏳ Waiting for bot data export...")
    print()
    
    hud = TradingHUD(refresh_rate=1.0)
    
    # NO simulated data - HUD reads from JSON files exported by bot
    # Files will be created by bot's _export_hud_data() method
    
    try:
        hud.start()
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping HUD...")
        hud.stop()


if __name__ == "__main__":
    main()

