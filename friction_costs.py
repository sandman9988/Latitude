#!/usr/bin/env python3
"""
friction_costs.py

Friction Cost Calculator for cTrader
Tracks and models all trading costs: spread, commission, swap, slippage

Per handbook: Friction is asymmetric (buy vs sell different) and varies by:
- Time of day (wider spreads during illiquid hours)
- Volatility regime (wider spreads in volatile markets)
- Position size (larger orders have worse slippage)

This module extracts costs from cTrader's FIX messages and provides
real-time friction estimates for position sizing and reward calculations.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional
from collections import deque
from datetime import datetime, timezone
import statistics

LOG = logging.getLogger(__name__)


@dataclass
class SymbolCosts:
    """Trading costs for a specific symbol from cTrader."""
    symbol: str
    symbol_id: int
    
    # From cTrader symbol specification
    digits: int = 2  # Price precision (BTCUSD usually 2 decimals: 50000.00)
    pip_size: float = 1.0  # Size of 1 pip in price units (e.g., $1 for BTCUSD)
    tick_size: float = 0.01  # Minimum price movement
    pip_value_per_lot: float = 10.0  # USD value of 1 pip for 1 standard lot
    
    # Commission structure
    commission_per_lot: float = 0.0  # Fixed commission per lot
    commission_percentage: float = 0.0  # Percentage commission (e.g., 0.001 = 0.1%)
    commission_type: str = "ABSOLUTE"  # "ABSOLUTE" or "PERCENTAGE"
    min_commission: float = 0.0
    
    # Swap rates (overnight financing)
    swap_long: float = 0.0  # Swap for long positions (pips per lot per day)
    swap_short: float = 0.0  # Swap for short positions
    swap_type: str = "PIPS"  # "PIPS", "PERCENTAGE", or "POINTS"
    
    # Position limits
    min_volume: float = 0.01  # Minimum position size
    max_volume: float = 100.0  # Maximum position size
    volume_step: float = 0.01  # Lot size increment
    
    # Market hours (for spread modeling)
    trading_hours_start: int = 0  # UTC hour when market opens
    trading_hours_end: int = 24  # UTC hour when market closes
    
    # Observed characteristics
    avg_spread_pips: float = 0.0  # Average observed spread
    min_spread_pips: float = 0.0  # Tightest observed spread
    max_spread_pips: float = 0.0  # Widest observed spread
    
    # Update tracking
    last_updated: Optional[datetime] = None


class SpreadTracker:
    """
    Track real-time spreads to model spread patterns.
    
    Spreads vary by:
    - Time of day (wider during Asian session for BTC)
    - Volatility (wider during news events)
    - Liquidity (wider during holidays)
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.spreads = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Hourly spread buckets (0-23 UTC)
        self.hourly_spreads: Dict[int, deque] = {h: deque(maxlen=100) for h in range(24)}
        
    def update(self, bid: float, ask: float, pip_size: float = 1.0):
        """Record a new bid/ask spread."""
        if bid <= 0 or ask <= 0 or ask <= bid:
            LOG.warning("Invalid bid/ask: bid=%.2f ask=%.2f", bid, ask)
            return
            
        # Calculate spread in pips (for BTCUSD, $1 = 1 pip)
        spread_pips = (ask - bid) / pip_size
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        self.spreads.append(spread_pips)
        self.timestamps.append(now)
        self.hourly_spreads[hour].append(spread_pips)
        
    def get_current_spread(self) -> float:
        """Get most recent spread in pips."""
        if not self.spreads:
            return 0.0
        return self.spreads[-1]
        
    def get_avg_spread(self) -> float:
        """Get average spread over window."""
        if not self.spreads:
            return 0.0
        return statistics.mean(self.spreads)
        
    def get_min_spread(self) -> float:
        """Get tightest spread observed."""
        if not self.spreads:
            return 0.0
        return min(self.spreads)
        
    def get_max_spread(self) -> float:
        """Get widest spread observed."""
        if not self.spreads:
            return 0.0
        return max(self.spreads)
        
    def get_hourly_avg_spread(self, hour: int) -> float:
        """Get average spread for specific hour (0-23 UTC)."""
        if hour not in self.hourly_spreads or not self.hourly_spreads[hour]:
            return self.get_avg_spread()  # Fallback to overall average
        return statistics.mean(self.hourly_spreads[hour])
        
    def get_current_hour_spread(self) -> float:
        """Get average spread for current hour."""
        now = datetime.now(timezone.utc)
        return self.get_hourly_avg_spread(now.hour)


class SlippageModel:
    """
    Model slippage as a function of position size and market conditions.
    
    Per handbook: Slippage is ASYMMETRIC:
    - Market buy (aggressive): worse fill than ask
    - Market sell (aggressive): worse fill than bid
    - Limit orders: no slippage (but may not fill)
    
    Slippage increases with:
    - Position size (market impact)
    - Volatility (wider bid-ask, faster price movement)
    - Low liquidity (thinner order book)
    """
    
    def __init__(self):
        # Base slippage (pips) for 1 standard lot
        self.base_slippage_pips = 0.5
        
        # Slippage scaling factor (how much slippage increases with size)
        # slippage = base * (1 + scale * (size - 1))
        self.size_scale = 0.2  # 20% increase per additional lot
        
        # Volatility adjustment (higher vol = more slippage)
        self.volatility_multiplier = 1.0
        
        # Asymmetry: buys typically have worse slippage than sells
        self.buy_multiplier = 1.2  # Buys pay 20% more slippage
        self.sell_multiplier = 1.0
        
    def estimate_slippage(self, 
                         quantity: float, 
                         side: str = "BUY",
                         volatility_factor: float = 1.0) -> float:
        """
        Estimate slippage in pips for a given order.
        
        Args:
            quantity: Position size in lots
            side: "BUY" or "SELL"
            volatility_factor: Multiplier for volatility (1.0 = normal, 2.0 = 2x vol)
            
        Returns:
            Expected slippage in pips
        """
        # Base slippage increases with size (square root to avoid extreme scaling)
        size_factor = 1.0 + self.size_scale * (quantity ** 0.5 - 1.0)
        
        # Apply asymmetry
        side_mult = self.buy_multiplier if side.upper() == "BUY" else self.sell_multiplier
        
        # Total slippage
        slippage = self.base_slippage_pips * size_factor * side_mult * volatility_factor
        
        return max(0.0, slippage)


class FrictionCalculator:
    """
    Calculate total friction costs for a trade.
    
    Total friction = spread + commission + swap + slippage
    
    All costs are converted to USD (or account currency) for consistency.
    """
    
    def __init__(self, symbol: str = "BTCUSD", symbol_id: int = 10028):
        self.symbol = symbol
        self.symbol_id = symbol_id
        
        # Symbol cost specification
        self.costs = SymbolCosts(symbol=symbol, symbol_id=symbol_id)
        
        # Real-time spread tracking
        self.spread_tracker = SpreadTracker()
        
        # Slippage modeling
        self.slippage_model = SlippageModel()
        
        LOG.info("FrictionCalculator initialized for %s (id=%d)", symbol, symbol_id)
        
    def update_symbol_costs(self, **kwargs):
        """
        Update symbol cost parameters from cTrader SecurityDefinition.
        
        Args:
            **kwargs: Fields from FIX SecurityDefinition message
                - digits: Price precision
                - pip_value: Value of 1 pip
                - commission_per_lot: Fixed commission
                - commission_percentage: Percentage commission
                - swap_long: Long swap rate
                - swap_short: Short swap rate
                - min_volume: Minimum lot size
                - max_volume: Maximum lot size
                - etc.
        """
        for key, value in kwargs.items():
            if hasattr(self.costs, key):
                setattr(self.costs, key, value)
                
        self.costs.last_updated = datetime.now(timezone.utc)
        LOG.info("Updated symbol costs for %s: %s", self.symbol, kwargs)
        
    def update_spread(self, bid: float, ask: float):
        """Update current spread observation."""
        self.spread_tracker.update(bid, ask, self.costs.pip_size)
        
        # Update observed spread statistics
        self.costs.avg_spread_pips = self.spread_tracker.get_avg_spread()
        self.costs.min_spread_pips = self.spread_tracker.get_min_spread()
        self.costs.max_spread_pips = self.spread_tracker.get_max_spread()
        
    def calculate_spread_cost(self, quantity: float) -> float:
        """
        Calculate spread cost in USD.
        
        Args:
            quantity: Position size in lots
            
        Returns:
            Spread cost in USD
        """
        current_spread = self.spread_tracker.get_current_spread()
        if current_spread <= 0:
            current_spread = self.costs.avg_spread_pips or 2.0  # Fallback to 2 pips
            
        # For BTCUSD: 1 pip per standard lot = $10
        # spread_cost = spread_pips * pip_value_per_lot * quantity
        spread_cost = current_spread * self.costs.pip_value_per_lot * quantity
        
        return spread_cost
        
    def calculate_commission(self, quantity: float, price: float) -> float:
        """
        Calculate commission in USD.
        
        Args:
            quantity: Position size in lots
            price: Entry price
            
        Returns:
            Commission in USD
        """
        if self.costs.commission_type == "PERCENTAGE":
            # Percentage of notional
            notional = quantity * price * 100000  # 1 lot = 100,000 units
            commission = notional * self.costs.commission_percentage
        else:
            # Fixed per lot
            commission = quantity * self.costs.commission_per_lot
            
        # Apply min commission
        commission = max(commission, self.costs.min_commission)
        
        return commission
        
    def calculate_swap(self, quantity: float, side: str, holding_days: float = 1.0) -> float:
        """
        Calculate swap (overnight financing) cost in USD.
        
        Args:
            quantity: Position size in lots
            side: "BUY" (long) or "SELL" (short)
            holding_days: Expected holding period in days
            
        Returns:
            Swap cost in USD (negative = you pay, positive = you earn)
        """
        swap_rate = self.costs.swap_long if side.upper() == "BUY" else self.costs.swap_short
        
        if self.costs.swap_type == "PIPS":
            # Swap in pips per lot per day
            # For BTCUSD: 1 pip per lot = $10 per standard lot
            swap_cost = swap_rate * self.costs.pip_value_per_lot * quantity * holding_days
        elif self.costs.swap_type == "PERCENTAGE":
            # Swap as percentage per day (would need price)
            # For now, use simplified calculation
            swap_cost = 0.0  # TODO: implement percentage swap
        else:
            swap_cost = 0.0
            
        return swap_cost
        
    def calculate_slippage_cost(self, 
                               quantity: float, 
                               side: str,
                               volatility_factor: float = 1.0) -> float:
        """
        Calculate expected slippage cost in USD.
        
        Args:
            quantity: Position size in lots
            side: "BUY" or "SELL"
            volatility_factor: Volatility multiplier
            
        Returns:
            Expected slippage cost in USD
        """
        slippage_pips = self.slippage_model.estimate_slippage(
            quantity, side, volatility_factor
        )
        
        # Convert pips to USD
        # For BTCUSD: 1 pip per lot = $10 per standard lot
        slippage_cost = slippage_pips * self.costs.pip_value_per_lot * quantity
        
        return slippage_cost
        
    def calculate_total_friction(self,
                                quantity: float,
                                side: str,
                                price: float,
                                holding_days: float = 1.0,
                                volatility_factor: float = 1.0) -> Dict[str, float]:
        """
        Calculate all friction costs for a trade.
        
        Args:
            quantity: Position size in lots
            side: "BUY" or "SELL"
            price: Entry price
            holding_days: Expected holding period
            volatility_factor: Volatility multiplier for slippage
            
        Returns:
            Dictionary with breakdown of costs:
                - spread: Spread cost (USD)
                - commission: Commission cost (USD)
                - swap: Swap cost (USD, can be negative)
                - slippage: Expected slippage (USD)
                - total: Total friction (USD)
                - total_pips: Total friction in pips
        """
        spread = self.calculate_spread_cost(quantity)
        commission = self.calculate_commission(quantity, price)
        swap = self.calculate_swap(quantity, side, holding_days)
        slippage = self.calculate_slippage_cost(quantity, side, volatility_factor)
        
        total = spread + commission + swap + slippage
        
        # Convert total back to pips for reference
        if quantity > 0 and self.costs.pip_value_per_lot > 0:
            total_pips = total / (self.costs.pip_value_per_lot * quantity)
        else:
            total_pips = 0.0
        
        return {
            'spread': spread,
            'commission': commission,
            'swap': swap,
            'slippage': slippage,
            'total': total,
            'total_pips': total_pips,
            'quantity': quantity,
            'side': side,
            'price': price,
        }
        
    def get_friction_adjusted_pnl(self, 
                                 raw_pnl: float,
                                 quantity: float,
                                 side: str,
                                 entry_price: float,
                                 holding_days: float = 1.0) -> float:
        """
        Adjust raw P&L for friction costs.
        
        Args:
            raw_pnl: Raw P&L before costs
            quantity: Position size
            side: "BUY" or "SELL"
            entry_price: Entry price
            holding_days: How long position was held
            
        Returns:
            Net P&L after friction
        """
        friction = self.calculate_total_friction(
            quantity, side, entry_price, holding_days
        )
        
        net_pnl = raw_pnl - friction['total']
        
        LOG.debug("Friction-adjusted P&L: raw=%.2f friction=%.2f net=%.2f",
                 raw_pnl, friction['total'], net_pnl)
        
        return net_pnl
        
    def get_statistics(self) -> Dict:
        """Get friction cost statistics."""
        return {
            'symbol': self.symbol,
            'avg_spread_pips': self.costs.avg_spread_pips,
            'min_spread_pips': self.costs.min_spread_pips,
            'max_spread_pips': self.costs.max_spread_pips,
            'current_spread_pips': self.spread_tracker.get_current_spread(),
            'commission_per_lot': self.costs.commission_per_lot,
            'swap_long': self.costs.swap_long,
            'swap_short': self.costs.swap_short,
            'base_slippage': self.slippage_model.base_slippage_pips,
            'last_updated': self.costs.last_updated,
        }


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing FrictionCalculator module...")
    
    calc = FrictionCalculator(symbol="BTCUSD", symbol_id=10028)
    
    # Simulate cTrader symbol info
    print("\n=== Test 1: Update symbol costs from cTrader ===")
    calc.update_symbol_costs(
        digits=2,
        pip_size=1.0,  # For BTCUSD, $1 = 1 pip
        tick_size=0.01,  # Minimum price movement
        pip_value_per_lot=10.0,  # $10 per pip per standard lot
        commission_per_lot=7.0,  # $7 per lot round-trip
        commission_type="ABSOLUTE",
        swap_long=-0.5,  # Pay 0.5 pips per day
        swap_short=-0.3,
        min_volume=0.01,
        max_volume=100.0,
    )
    print(f"✓ Symbol costs updated: commission=${calc.costs.commission_per_lot}/lot")
    
    # Simulate spread updates (BTCUSD typical spreads are $1-5)
    print("\n=== Test 2: Track spreads ===")
    spreads_btc = [
        (50000.00, 50001.00),  # $1 spread
        (50000.00, 50001.50),  # $1.50 spread
        (50000.00, 50002.00),  # $2 spread
        (50000.00, 50002.50),  # $2.50 spread (wider)
    ]
    
    for bid, ask in spreads_btc:
        calc.update_spread(bid, ask)
        
    print(f"✓ Spreads tracked: avg={calc.costs.avg_spread_pips:.2f} pips")
    print(f"  Min: {calc.costs.min_spread_pips:.2f} pips")
    print(f"  Max: {calc.costs.max_spread_pips:.2f} pips")
    
    # Calculate friction for a trade
    print("\n=== Test 3: Calculate friction for BUY 0.10 lot @ $50,000 ===")
    friction = calc.calculate_total_friction(
        quantity=0.10,
        side="BUY",
        price=50000.0,
        holding_days=1.0,
        volatility_factor=1.0
    )
    
    print(f"Spread cost:     ${friction['spread']:.2f}")
    print(f"Commission:      ${friction['commission']:.2f}")
    print(f"Swap (1 day):    ${friction['swap']:.2f}")
    print(f"Slippage:        ${friction['slippage']:.2f}")
    print(f"─────────────────────────────")
    print(f"TOTAL FRICTION:  ${friction['total']:.2f}")
    print(f"Total in pips:   {friction['total_pips']:.2f} pips")
    
    # Test friction-adjusted P&L
    print("\n=== Test 4: Friction-adjusted P&L ===")
    raw_pnl = 100.0  # Made $100 gross
    net_pnl = calc.get_friction_adjusted_pnl(
        raw_pnl=raw_pnl,
        quantity=0.10,
        side="BUY",
        entry_price=50000.0,
        holding_days=1.0
    )
    print(f"Raw P&L:         ${raw_pnl:.2f}")
    print(f"Friction cost:   ${friction['total']:.2f}")
    print(f"Net P&L:         ${net_pnl:.2f}")
    print(f"Friction ratio:  {(friction['total']/raw_pnl)*100:.1f}% of gross profit")
    
    # Show statistics
    print("\n=== Test 5: Friction statistics ===")
    stats = calc.get_statistics()
    print(f"Symbol: {stats['symbol']}")
    print(f"Average spread: {stats['avg_spread_pips']:.2f} pips")
    print(f"Commission: ${stats['commission_per_lot']}/lot")
    print(f"Swap long: {stats['swap_long']:.2f} pips/day")
    print(f"Base slippage: {stats['base_slippage']:.2f} pips")
    
    print("\n✅ All tests complete")
