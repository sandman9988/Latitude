"""
ring_buffer.py
==============
O(1) rolling statistics using ring buffers.

Master Handbook alignment:
- Section: "Ring buffer stats"
- Purpose: Constant-time rolling mean, variance, min/max
- Defensive: Prevents O(N) recalculation on every bar

Key Features:
1. RingBuffer - Fixed-size circular buffer with O(1) append
2. RollingMean - Incremental mean update (no sum() calls)
3. RollingVariance - Welford's algorithm for numerically stable variance
4. RollingMinMax - Deque-based monotonic queue for efficient min/max

Performance:
- Traditional: O(N) per bar (recalculate mean/std over N bars)
- Ring buffer: O(1) per bar (incremental update)
- Critical for high-frequency systems with large lookback periods

Usage:
    mean_tracker = RollingMean(period=20)
    var_tracker = RollingVariance(period=20)

    for price in price_stream:
        mean_tracker.update(price)
        var_tracker.update(price)

        current_mean = mean_tracker.value
        current_std = var_tracker.std
"""

import logging
import math
import sys
from collections import deque

LOG = logging.getLogger(__name__)

# Test and validation constants
FLOATING_POINT_TOLERANCE: float = 1e-9
MIN_SAMPLES_FOR_VARIANCE: int = 2
MIN_SPEEDUP_EXPECTED: float = 1.5


class RingBuffer:
    """
    Fixed-size circular buffer with O(1) append.

    When buffer fills, oldest value is automatically discarded.
    Useful as base for rolling statistics.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of elements to store
        """
        if capacity < 1:
            raise ValueError(f"Capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.buffer: deque[float] = deque(maxlen=capacity)
        self.count = 0  # Total items added (not capped at capacity)

    def append(self, value: float):
        """Add new value, discarding oldest if full."""
        self.buffer.append(value)
        self.count += 1

    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return len(self.buffer) == self.capacity

    def __len__(self) -> int:
        """Return current number of elements."""
        return len(self.buffer)

    def __getitem__(self, index: int) -> float:
        """Access by index (0 = oldest, -1 = newest)."""
        return self.buffer[index]

    def __iter__(self):
        """Iterate from oldest to newest."""
        return iter(self.buffer)


class RollingMean:
    """
    O(1) rolling mean using incremental update.

    Instead of recalculating sum()/N every bar:
    1. Track running sum
    2. When new value arrives: sum += new_value
    3. If buffer full: sum -= discarded_value
    4. mean = sum / count

    Complexity: O(1) per update vs O(N) for naive approach.
    """

    def __init__(self, period: int):
        """
        Args:
            period: Rolling window size
        """
        self.period = period
        self.buffer = RingBuffer(period)
        self.sum = 0.0
        self.value = 0.0  # Current mean

    def update(self, new_value: float):
        """
        Add new value and update mean incrementally.

        Args:
            new_value: New data point
        """
        if self.buffer.is_full():
            # Remove oldest value from sum
            discarded = self.buffer[0]
            self.sum -= discarded

        self.sum += new_value
        self.buffer.append(new_value)

        if len(self.buffer) > 0:
            self.value = self.sum / len(self.buffer)
        else:
            self.value = 0.0

    def is_ready(self) -> bool:
        """Check if buffer is full (period samples collected)."""
        return self.buffer.is_full()

    def __repr__(self) -> str:
        return f"RollingMean(period={self.period}, value={self.value:.6f}, ready={self.is_ready()})"


class RollingVariance:
    """
    O(1) rolling variance using Welford's algorithm.

    Welford's algorithm:
    - Numerically stable (avoids catastrophic cancellation)
    - Incremental update (no need to store all values)
    - Tracks mean and M2 (sum of squared deviations)

    Key insight:
    variance = M2 / (N - 1)
    where M2 = sum((x - mean)^2)

    Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, period: int, min_periods: int = 2):
        """
        Args:
            period: Rolling window size
            min_periods: Minimum samples before variance is valid (default 2)
        """
        self.period = period
        self.min_periods = max(2, min_periods)
        self.buffer = RingBuffer(period)

        # Welford's algorithm state
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared deviations from mean

        self.variance = 0.0
        self.std = 0.0

    def update(self, new_value: float):
        """
        Add new value and update variance incrementally.

        Uses Welford's algorithm for numerical stability.

        Args:
            new_value: New data point
        """
        n = len(self.buffer)

        if self.buffer.is_full():
            # Remove oldest value using reverse Welford's
            old_value = self.buffer[0]

            # Adjust for removed value
            n_after = n - 1
            if n_after > 0:
                delta_old = old_value - self.mean
                self.mean = (self.mean * n - old_value) / n_after
                delta_old2 = old_value - self.mean
                self.m2 -= delta_old * delta_old2
            else:
                self.mean = 0.0
                self.m2 = 0.0

            n = n_after

        # Add new value
        self.buffer.append(new_value)
        n += 1

        delta = new_value - self.mean
        self.mean += delta / n
        delta2 = new_value - self.mean
        self.m2 += delta * delta2

        # Calculate variance and std
        if n >= self.min_periods:
            self.variance = self.m2 / (n - 1)
            self.std = math.sqrt(self.variance) if self.variance > 0 else 0.0
        else:
            self.variance = 0.0
            self.std = 0.0

    def is_ready(self) -> bool:
        """Check if enough samples for valid variance."""
        return len(self.buffer) >= self.min_periods

    def __repr__(self) -> str:
        return (
            f"RollingVariance(period={self.period}, mean={self.mean:.6f}, "
            f"std={self.std:.6f}, ready={self.is_ready()})"
        )


class RollingMinMax:
    """
    O(1) amortized rolling min/max using monotonic deque.

    Traditional approach: O(N) to find min/max over window
    Monotonic deque: O(1) amortized per update

    Key idea:
    - Maintain deque of (value, timestamp) in increasing order for min
    - When new value arrives, remove all larger values from back
    - Front of deque is always current min
    - Remove expired values from front

    For max: same logic but decreasing order.
    """

    def __init__(self, period: int):
        """
        Args:
            period: Rolling window size
        """
        self.period = period
        self.min_deque: deque[tuple[float, int]] = deque()  # (value, index)
        self.max_deque: deque[tuple[float, int]] = deque()
        self.index = 0

        self.min_value = float("inf")
        self.max_value = float("-inf")

    def update(self, new_value: float):
        """
        Add new value and update min/max.

        Args:
            new_value: New data point
        """
        current_idx = self.index
        self.index += 1

        # Remove expired values (outside window)
        cutoff_idx = current_idx - self.period

        while self.min_deque and self.min_deque[0][1] <= cutoff_idx:
            self.min_deque.popleft()

        while self.max_deque and self.max_deque[0][1] <= cutoff_idx:
            self.max_deque.popleft()

        # Maintain monotonic increasing deque for min
        while self.min_deque and self.min_deque[-1][0] >= new_value:
            self.min_deque.pop()
        self.min_deque.append((new_value, current_idx))

        # Maintain monotonic decreasing deque for max
        while self.max_deque and self.max_deque[-1][0] <= new_value:
            self.max_deque.pop()
        self.max_deque.append((new_value, current_idx))

        # Update values
        self.min_value = self.min_deque[0][0] if self.min_deque else float("inf")
        self.max_value = self.max_deque[0][0] if self.max_deque else float("-inf")

    def is_ready(self) -> bool:
        """Check if at least one sample collected."""
        return self.index > 0

    def __repr__(self) -> str:
        return f"RollingMinMax(period={self.period}, min={self.min_value:.6f}, max={self.max_value:.6f})"


class RollingStats:
    """
    Combined O(1) rolling statistics.

    Tracks mean, std, min, max simultaneously with constant-time updates.
    Ideal for real-time trading systems.
    """

    def __init__(self, period: int):
        """
        Args:
            period: Rolling window size
        """
        self.period = period
        self.mean_tracker = RollingMean(period)
        self.var_tracker = RollingVariance(period)
        self.minmax_tracker = RollingMinMax(period)

    def update(self, new_value: float):
        """Add new value and update all stats."""
        self.mean_tracker.update(new_value)
        self.var_tracker.update(new_value)
        self.minmax_tracker.update(new_value)

    @property
    def mean(self) -> float:
        return self.mean_tracker.value

    @property
    def std(self) -> float:
        return self.var_tracker.std

    @property
    def variance(self) -> float:
        return self.var_tracker.variance

    @property
    def min(self) -> float:
        return self.minmax_tracker.min_value

    @property
    def max(self) -> float:
        return self.minmax_tracker.max_value

    def is_ready(self) -> bool:
        """Check if all trackers are ready."""
        return self.mean_tracker.is_ready() and self.var_tracker.is_ready() and self.minmax_tracker.is_ready()

    def __repr__(self) -> str:
        return (
            f"RollingStats(period={self.period}, mean={self.mean:.6f}, "
            f"std={self.std:.6f}, min={self.min:.6f}, max={self.max:.6f}, "
            f"ready={self.is_ready()})"
        )


# ============================================
# Self-test
# ============================================
if __name__ == "__main__":
    import random
    import time

    import numpy as np

    print("=" * 70)
    print("Ring Buffer Stats Self-Test")
    print("=" * 70)

    # Test 1: RollingMean correctness
    print("\n[TEST 1] RollingMean correctness")
    period = 5
    mean_tracker = RollingMean(period)
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    for val in values:
        mean_tracker.update(val)

    # After 6 values with period=5, buffer contains [20, 30, 40, 50, 60]
    # Mean should be 40.0
    expected_mean = 40.0
    if abs(mean_tracker.value - expected_mean) < FLOATING_POINT_TOLERANCE:
        print(f"    ✓ RollingMean = {mean_tracker.value:.2f} (expected {expected_mean:.2f})")
    else:
        print(f"    ERROR: Expected {expected_mean}, got {mean_tracker.value}")
        sys.exit(1)

    # Test 2: RollingVariance correctness
    print("\n[TEST 2] RollingVariance correctness")
    var_tracker = RollingVariance(period)

    for val in values:
        var_tracker.update(val)

    # Verify against numpy
    window = [20.0, 30.0, 40.0, 50.0, 60.0]
    expected_mean = np.mean(window)
    expected_std = np.std(window, ddof=1)

    mean_error = abs(var_tracker.mean - expected_mean)
    std_error = abs(var_tracker.std - expected_std)

    if mean_error < FLOATING_POINT_TOLERANCE and std_error < FLOATING_POINT_TOLERANCE:
        print(f"    ✓ Mean = {var_tracker.mean:.2f} (expected {expected_mean:.2f})")
        print(f"    ✓ Std = {var_tracker.std:.2f} (expected {expected_std:.2f})")
    else:
        print(f"    ERROR: Mean error = {mean_error}, Std error = {std_error}")
        sys.exit(1)

    # Test 3: RollingMinMax correctness
    print("\n[TEST 3] RollingMinMax correctness")
    minmax_tracker = RollingMinMax(period)

    for val in values:
        minmax_tracker.update(val)

    expected_min = 20.0
    expected_max = 60.0

    if (
        abs(minmax_tracker.min_value - expected_min) < FLOATING_POINT_TOLERANCE
        and abs(minmax_tracker.max_value - expected_max) < FLOATING_POINT_TOLERANCE
    ):
        print(f"    ✓ Min = {minmax_tracker.min_value:.2f} (expected {expected_min:.2f})")
        print(f"    ✓ Max = {minmax_tracker.max_value:.2f} (expected {expected_max:.2f})")
    else:
        print(
            f"    ERROR: Min {minmax_tracker.min_value} vs {expected_min}, Max {minmax_tracker.max_value} vs {expected_max}"
        )
        sys.exit(1)

    # Test 4: Performance comparison
    print("\n[TEST 4] Performance comparison (O(1) vs O(N))")
    period = 100
    n_iterations = 10000

    # Generate random data
    random.seed(42)
    data = [random.gauss(100, 10) for _ in range(n_iterations)]

    # Benchmark O(1) ring buffer
    stats = RollingStats(period)
    start = time.perf_counter()
    for val in data:
        stats.update(val)
    ring_time = time.perf_counter() - start

    # Benchmark O(N) naive approach
    window = deque(maxlen=period)
    start = time.perf_counter()
    for val in data:
        window.append(val)
        if len(window) >= MIN_SAMPLES_FOR_VARIANCE:
            mean = sum(window) / len(window)
            std = (sum((x - mean) ** 2 for x in window) / (len(window) - 1)) ** 0.5
            min_val = min(window)
            max_val = max(window)
    naive_time = time.perf_counter() - start

    speedup = naive_time / ring_time
    print(f"    Ring buffer: {ring_time*1000:.2f} ms")
    print(f"    Naive O(N):  {naive_time*1000:.2f} ms")
    print(f"    ✓ Speedup: {speedup:.2f}x faster")

    if speedup < MIN_SPEEDUP_EXPECTED:
        print(f"    WARNING: Expected >{MIN_SPEEDUP_EXPECTED}x speedup, got {speedup:.2f}x")

    # Test 5: Numerical stability
    print("\n[TEST 5] Numerical stability (large values)")
    var_tracker = RollingVariance(10)

    # Add large values that could cause overflow in naive variance
    base = 1e10
    small_variance_data = [base + i * 0.1 for i in range(20)]

    for val in small_variance_data:
        var_tracker.update(val)

    # Should have small std (around 0.3) despite large mean (around 1e10)
    if var_tracker.std < 1.0:
        print(f"    ✓ Stable std = {var_tracker.std:.6f} with mean = {var_tracker.mean:.2e}")
    else:
        print(f"    ERROR: Std too large: {var_tracker.std}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✓ All Ring Buffer tests passed!")
    print("=" * 70)
