"""
non_repaint_guards.py
====================
Prevents look-ahead bias by enforcing non-repaint access to bar data.

Master Handbook alignment:
- Section: "Non-repaint discipline"
- Purpose: Prevent accessing bar[0] before bar close
- Defensive: Raises errors on premature bar[0] access

Key Features:
1. NonRepaintBarAccess - Wrapper for series data with strict bar[0] protection
2. is_bar_closed flag - Explicit permission for bar[0] access
3. safe_get_previous - Always safe to access historical bars (bar[1], bar[2], etc.)

Usage:
    close_series = NonRepaintBarAccess(close_prices, max_lookback=500)

    # During bar formation (on_tick):
    prev_close = close_series.safe_get_previous(1)  # OK: bar[1]
    # current = close_series.get_current()  # RAISES: Bar not closed yet

    # At bar close (on_bar_close):
    close_series.mark_bar_closed()
    current = close_series.get_current()  # OK: bar[0] access allowed
    close_series.mark_bar_opened()  # Reset for next bar
"""

import logging
import sys
from collections import deque

LOG = logging.getLogger(__name__)

# Test constants
TEST_BAR_VALUE_1: float = 101.0
TEST_BAR_VALUE_2: float = 102.0
TEST_BAR_VALUE_4: float = 104.0


class NonRepaintError(Exception):
    """Raised when attempting to access bar[0] before bar close."""

    pass


class NonRepaintBarAccess:
    """
    Wrapper for time-series data that prevents look-ahead bias.

    Enforces:
    - bar[0] (current incomplete bar) can ONLY be accessed after mark_bar_closed()
    - bar[1+] (historical bars) are always safe to access
    - Raises NonRepaintError on premature bar[0] access

    This prevents the common bug where:
    1. Strategy accesses close[0] during bar formation
    2. Close changes before bar actually closes
    3. Backtest results are over-optimistic (non-repaintable in live)
    """

    def __init__(self, name: str, max_lookback: int = 500):
        """
        Args:
            name: Series identifier for error messages (e.g., "close", "rsi")
            max_lookback: Maximum bars to store
        """
        self.name = name
        self.max_lookback = max_lookback
        self.data: deque[float] = deque(maxlen=max_lookback)
        self.is_bar_closed = False
        self.bar_count = 0

    def append(self, value: float):
        """
        Add new bar data. Called when new bar forms.

        Workflow:
        1. on_bar_close() calls mark_bar_closed()
        2. Strategy accesses bar[0] via get_current()
        3. New bar starts, mark_bar_opened() resets flag
        4. append() adds new incomplete bar[0]
        """
        self.data.append(value)
        self.bar_count += 1
        # Reset bar closed flag when new data arrives (new bar started)
        self.is_bar_closed = False

    def mark_bar_closed(self):
        """
        Explicitly mark current bar as closed.
        ONLY after this can get_current() be called.

        Call this in your on_bar_close() handler.
        """
        self.is_bar_closed = True
        LOG.debug("[%s] Bar %d marked closed - bar[0] access now permitted", self.name, self.bar_count)

    def mark_bar_opened(self):
        """
        Mark new bar as opened (incomplete).
        Call this after processing bar close, before next tick.
        """
        self.is_bar_closed = False
        LOG.debug("[%s] Bar %d opened - bar[0] access now restricted", self.name, self.bar_count + 1)

    def get_current(self, allow_incomplete: bool = False) -> float:
        """
        Get current bar value (bar[0]).

        Args:
            allow_incomplete: If True, allows access even during bar formation.
                             Use ONLY for monitoring, NOT for trading decisions.

        Returns:
            Current bar value (bar[0])

        Raises:
            NonRepaintError: If bar not closed and allow_incomplete=False
            IndexError: If no data available
        """
        if not self.data:
            raise IndexError(f"[{self.name}] No data available")

        if not self.is_bar_closed and not allow_incomplete:
            raise NonRepaintError(
                f"[{self.name}] Cannot access bar[0] before bar close. "
                f"Current bar #{self.bar_count} is still forming. "
                f"Call mark_bar_closed() first or use safe_get_previous(1)."
            )

        return self.data[-1]

    def safe_get_previous(self, bars_ago: int) -> float | None:
        """
        Safely get historical bar value.

        Args:
            bars_ago: How many bars back (1 = previous closed bar, 2 = 2 bars ago, etc.)
                     Must be >= 1 to avoid accessing incomplete bar[0]

        Returns:
            Historical bar value or None if not enough data

        Note:
            This is ALWAYS safe - historical bars never repaint.
            bars_ago=1 gets bar[1], which is the most recent CLOSED bar.
        """
        if bars_ago < 1:
            raise ValueError(f"[{self.name}] bars_ago must be >= 1 (got {bars_ago}). Use get_current() for bar[0].")

        if bars_ago >= len(self.data):
            return None

        # data[-1] is bar[0], data[-2] is bar[1], etc.
        # bars_ago=1 -> data[-2] (skip current bar[0])
        index = -(bars_ago + 1)
        return self.data[index]

    def get_series(self, count: int, offset: int = 1) -> list[float]:
        """
        Get historical series of bars.

        Args:
            count: Number of bars to retrieve
            offset: Starting offset (1 = skip current bar, 2 = skip 2 bars, etc.)
                   Default 1 ensures we skip incomplete bar[0]

        Returns:
            List of historical values [bar[offset], bar[offset+1], ..., bar[offset+count-1]]
            Returns shorter list if not enough data.

        Example:
            get_series(5, offset=1) -> [bar[1], bar[2], bar[3], bar[4], bar[5]]
        """
        if offset < 0:
            raise ValueError(f"[{self.name}] offset must be >= 0")

        # Special case: if bar is closed and offset=0, we can include bar[0]
        if offset == 0 and not self.is_bar_closed:
            raise NonRepaintError(
                f"[{self.name}] Cannot use offset=0 when bar not closed. "
                f"Use offset=1 or call mark_bar_closed() first."
            )

        result = []
        for i in range(count):
            bars_back = offset + i
            if bars_back == 0:
                # Accessing bar[0] - only allowed if closed
                if self.is_bar_closed and len(self.data) > 0:
                    result.append(self.data[-1])
                else:
                    break
            else:
                # Accessing historical bars
                val = self.safe_get_previous(bars_back)
                if val is None:
                    break
                result.append(val)
        return result

    def __len__(self) -> int:
        """Return number of bars stored."""
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"NonRepaintBarAccess(name={self.name}, bars={len(self.data)}, "
            f"closed={self.is_bar_closed}, bar_count={self.bar_count})"
        )


class NonRepaintIndicator:
    """
    Base class for indicators that respect non-repaint discipline.

    Ensures:
    - Indicators only update on bar close
    - Current indicator value reflects CLOSED bars only
    - No look-ahead bias in calculations

    Usage:
        class MyRSI(NonRepaintIndicator):
            def calculate(self, close_series: NonRepaintBarAccess) -> float:
                # Use close_series.safe_get_previous(1-14) for RSI calculation
                return rsi_value
    """

    def __init__(self, name: str, period: int):
        self.name = name
        self.period = period
        self.values: deque[float] = deque(maxlen=500)
        self.is_ready = False

    def update(self, new_value: float):
        """
        Update indicator with new value (called on bar close).

        Args:
            new_value: Newly calculated indicator value from closed bar
        """
        self.values.append(new_value)
        if len(self.values) >= self.period:
            self.is_ready = True

    def get_current(self) -> float | None:
        """Get current indicator value (from last closed bar)."""
        return self.values[-1] if self.values else None

    def get_previous(self, bars_ago: int) -> float | None:
        """Get historical indicator value."""
        if bars_ago < 1 or bars_ago >= len(self.values):
            return None
        return self.values[-(bars_ago + 1)]


# ============================================
# Self-test
# ============================================
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)

    print("=" * 70)
    print("NonRepaint Guards Self-Test")
    print("=" * 70)

    # Test 1: Basic non-repaint access
    print("\n[TEST 1] Basic non-repaint access")
    close = NonRepaintBarAccess("close", max_lookback=10)

    # Simulate 3 bars forming
    for i in range(3):
        close.append(100.0 + i)
        print(f"  Bar {i} formed: {100.0 + i}")

    # Try to access bar[0] before close -> should FAIL
    print("\n  Attempting bar[0] access before close...")
    try:
        val = close.get_current()
        print(f"    ERROR: Should have raised NonRepaintError, got {val}")
        sys.exit(1)
    except NonRepaintError as e:
        print(f"    ✓ Correctly blocked: {e}")

    # Access historical bars -> should SUCCEED
    print("\n  Accessing bar[1] (previous closed bar)...")
    val = close.safe_get_previous(1)
    if val == TEST_BAR_VALUE_1:
        print(f"    ✓ bar[1] = {val}")
    else:
        print(f"    ERROR: Expected {TEST_BAR_VALUE_1}, got {val}")
        sys.exit(1)

    # Mark bar closed and retry
    print("\n  Marking bar closed...")
    close.mark_bar_closed()
    val = close.get_current()
    if val == TEST_BAR_VALUE_2:
        print(f"    ✓ bar[0] = {val} (after mark_bar_closed)")
    else:
        print(f"    ERROR: Expected {TEST_BAR_VALUE_2}, got {val}")
        sys.exit(1)

    # Test 2: Series access
    print("\n[TEST 2] Series access")
    close.mark_bar_opened()
    close.append(103.0)
    close.mark_bar_closed()

    # Now data is: [100, 101, 102, 103]
    # bar[0]=103, bar[1]=102, bar[2]=101, bar[3]=100
    # get_series(3, offset=0) with bar closed should give [103, 102, 101]
    series = close.get_series(3, offset=0)
    expected = [103.0, 102.0, 101.0]
    if series == expected:
        print(f"    ✓ get_series(3, offset=0) = {series}")
    else:
        print(f"    ERROR: Expected {expected}, got {series}")
        sys.exit(1)

    # Test 3: Allow incomplete flag
    print("\n[TEST 3] Allow incomplete flag for monitoring")
    close.mark_bar_opened()
    close.append(TEST_BAR_VALUE_4)  # Incomplete bar

    val = close.get_current(allow_incomplete=True)
    if val == TEST_BAR_VALUE_4:
        print(f"    ✓ get_current(allow_incomplete=True) = {val}")
    else:
        print(f"    ERROR: Expected {TEST_BAR_VALUE_4}, got {val}")
        sys.exit(1)

    # Test 4: Invalid offset
    print("\n[TEST 4] Invalid offset detection")
    try:
        series = close.get_series(5, offset=0)  # offset=0 with bar not closed
        print("    ERROR: Should have raised NonRepaintError")
        sys.exit(1)
    except NonRepaintError as e:
        print(f"    ✓ Correctly rejected offset=0 when bar not closed: {str(e)[:80]}...")

    print("\n" + "=" * 70)
    print("✓ All NonRepaint Guards tests passed!")
    print("=" * 70)
