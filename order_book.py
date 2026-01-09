#!/usr/bin/env python3
"""
order_book.py

Lightweight L2 order book and VPIN calculator for microstructure signals.
- Maintains top-N bids/asks with sizes
- Provides imbalance and depth statistics
- Tracks trades and feeds VPIN (volume-synchronized probability of informed trading)
"""

from collections import deque
from typing import Dict, Optional, Tuple


class OrderBook:
    def __init__(self, depth: int = 10):
        self.depth = depth
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()

    def _prune(self, levels: Dict[float, float]) -> Dict[float, float]:
        # Keep only top-N by price (desc for bids, asc for asks handled externally)
        return dict(list(levels.items())[: self.depth])

    def update_level(self, side: str, price: float, size: float) -> None:
        """Update order book level with defensive validation."""
        # Defensive: Validate inputs
        import math
        if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            return  # Invalid price, skip silently
        if not isinstance(size, (int, float)) or not math.isfinite(size):
            return  # Invalid size, skip silently
        
        # Defensive: Validate side
        if side not in ("BID", "ASK"):
            return
        
        book = self.bids if side == "BID" else self.asks
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size
        # Sort and prune
        if side == "BID":
            book_sorted = dict(sorted(book.items(), key=lambda x: x[0], reverse=True))
        else:
            book_sorted = dict(sorted(book.items(), key=lambda x: x[0]))
        pruned = self._prune(book_sorted)
        if side == "BID":
            self.bids = pruned
        else:
            self.asks = pruned

    def best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        bid = next(iter(self.bids)) if self.bids else None
        ask = next(iter(self.asks)) if self.asks else None
        return bid, ask

    def spread(self) -> Optional[float]:
        """Calculate spread with defensive validation."""
        bid, ask = self.best_bid_ask()
        if bid is None or ask is None:
            return None
        # Defensive: Ensure bid < ask (crossed book should not happen)
        if bid >= ask:
            return 0.0  # Crossed book, return zero spread
        spread_value = ask - bid
        # Defensive: Validate result
        import math
        if not math.isfinite(spread_value) or spread_value < 0:
            return None
        return spread_value

    def depth_sum(self, levels: int = 5) -> Tuple[float, float]:
        b_sum = sum(list(self.bids.values())[:levels]) if self.bids else 0.0
        a_sum = sum(list(self.asks.values())[:levels]) if self.asks else 0.0
        return b_sum, a_sum

    def imbalance(self, levels: int = 5) -> float:
        b_sum, a_sum = self.depth_sum(levels)
        total = b_sum + a_sum
        if total <= 0:
            return 0.0
        return (b_sum - a_sum) / total


class VPINCalculator:
    """
    Volume-synchronized Probability of Informed Trading (VPIN).
    - Uses volume buckets (bucket_volume) instead of fixed time
    - Tracks rolling imbalance statistics (mean/std/zscore)
    - Requires trade side inference from tape or book
    """

    def __init__(self, bucket_volume: float = 1.0, window: int = 20):
        self.bucket_volume = max(bucket_volume, 1e-6)
        self.window = max(window, 1)
        self.reset()

    def reset(self) -> None:
        self.current_buy = 0.0
        self.current_sell = 0.0
        self.completed: deque[float] = deque(maxlen=self.window)

    def update(self, volume: float, side: str) -> Optional[float]:
        """Update VPIN with defensive validation."""
        # Defensive: Validate volume
        import math
        if not isinstance(volume, (int, float)) or not math.isfinite(volume) or volume <= 0:
            return None
        
        # Defensive: Cap extreme volumes (>1000x bucket size)
        max_volume = self.bucket_volume * 1000.0
        if volume > max_volume:
            volume = max_volume
        
        # Defensive: Validate side
        side_upper = side.upper()
        if side_upper not in ("BUY", "SELL"):
            return None
        
        if side_upper == "BUY":
            self.current_buy += volume
        else:
            self.current_sell += volume

        # If bucket filled, finalize imbalance
        bucket_total = self.current_buy + self.current_sell
        if bucket_total >= self.bucket_volume:
            imbalance = abs(self.current_buy - self.current_sell) / bucket_total
            self.completed.append(imbalance)
            # Carry over residual volume to next bucket
            residual = bucket_total - self.bucket_volume
            if residual > 0:
                # Assign residual to the dominating side
                if self.current_buy > self.current_sell:
                    self.current_buy = residual
                    self.current_sell = 0.0
                else:
                    self.current_sell = residual
                    self.current_buy = 0.0
            else:
                self.current_buy = 0.0
                self.current_sell = 0.0
            return imbalance
        return None

    def get_vpin(self) -> float:
        if not self.completed:
            return 0.0
        return sum(self.completed) / len(self.completed)

    def get_stats(self) -> dict:
        """Get VPIN statistics with defensive validation."""
        import math

        vpin = self.get_vpin()
        if not self.completed:
            return {"vpin": vpin, "mean": 0.0, "std": 0.0, "zscore": 0.0}
        mean = vpin
        
        # Defensive: Validate variance calculation
        try:
            variance = sum((x - mean) ** 2 for x in self.completed) / len(self.completed)
            if not math.isfinite(variance) or variance < 0:
                variance = 0.0
            std = math.sqrt(max(0.0, variance))
        except (ValueError, OverflowError):
            std = 0.0
        
        last = self.completed[-1]
        # Defensive: Division by zero protection with epsilon
        epsilon = 1e-8
        z = 0.0 if std < epsilon else (last - mean) / std
        
        # Defensive: Cap extreme z-scores
        z = max(-10.0, min(10.0, z))
        
        return {"vpin": vpin, "mean": mean, "std": std, "zscore": z}
