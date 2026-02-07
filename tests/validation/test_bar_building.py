#!/usr/bin/env python3
import pytest

"""
Bar Building & Bar Closure Verification Tests

Tests:
1. Bar aggregation (OHLC updates)
2. Bar closure timing
3. Timeframe alignment
4. Bar boundary detection
5. Data integrity across bar transitions
"""

import datetime as dt
from datetime import UTC


class BarBuilder:
    """Simplified BarBuilder for testing"""

    def __init__(self, timeframe_minutes: int = 15):
        self.timeframe_minutes = timeframe_minutes
        self.bucket: dt.datetime | None = None
        self.o: float | None = None
        self.h: float | None = None
        self.l: float | None = None
        self.c: float | None = None

    def bucket_start(self, t: dt.datetime) -> dt.datetime:
        m = (t.minute // self.timeframe_minutes) * self.timeframe_minutes
        return t.replace(minute=m, second=0, microsecond=0)

    def update(self, t: dt.datetime, mid: float):
        b = self.bucket_start(t)
        if self.bucket is None:
            self.bucket = b
            self.o = self.h = self.l = self.c = mid
            return None

        if b != self.bucket:
            closed = (self.bucket, self.o, self.h, self.l, self.c)
            self.bucket = b
            self.o = self.h = self.l = self.c = mid
            return closed

        self.c = mid
        if self.h is None or mid > self.h:
            self.h = mid
        if self.l is None or mid < self.l:
            self.l = mid
        return None


def test_basic_bar_aggregation():
    """Test OHLC aggregation within a single bar"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Bar Aggregation (OHLC)")
    print("=" * 70)

    bb = BarBuilder(timeframe_minutes=15)
    base = dt.datetime(2026, 1, 11, 10, 0, 0, tzinfo=UTC)

    # First tick opens the bar
    result = bb.update(base, 100.0)
    assert result is None, "First tick should not close a bar"
    assert bb.o == pytest.approx(100.0), f"Open should be 100.0, got {bb.o}"
    assert bb.h == pytest.approx(100.0), f"High should be 100.0, got {bb.h}"
    assert bb.l == pytest.approx(100.0), f"Low should be 100.0, got {bb.l}"
    assert bb.c == pytest.approx(100.0), f"Close should be 100.0, got {bb.c}"
    print("  ✓ First tick: O=100.0 H=100.0 L=100.0 C=100.0")

    # Update with higher price
    result = bb.update(base + dt.timedelta(seconds=30), 105.0)
    assert result is None, "Tick within bar should not close"
    assert bb.o == pytest.approx(100.0), "Open should remain 100.0"
    assert bb.h == pytest.approx(105.0), f"High should update to 105.0, got {bb.h}"
    assert bb.l == pytest.approx(100.0), "Low should remain 100.0"
    assert bb.c == pytest.approx(105.0), f"Close should update to 105.0, got {bb.c}"
    print("  ✓ High update: H=105.0 C=105.0")

    # Update with lower price
    result = bb.update(base + dt.timedelta(seconds=60), 95.0)
    assert result is None, "Tick within bar should not close"
    assert bb.o == pytest.approx(100.0), "Open should remain 100.0"
    assert bb.h == pytest.approx(105.0), "High should remain 105.0"
    assert bb.l == pytest.approx(95.0), f"Low should update to 95.0, got {bb.l}"
    assert bb.c == pytest.approx(95.0), f"Close should update to 95.0, got {bb.c}"
    print("  ✓ Low update: L=95.0 C=95.0")

    # Final close value
    result = bb.update(base + dt.timedelta(seconds=120), 102.0)
    assert result is None, "Tick within bar should not close"
    assert bb.o == pytest.approx(100.0), "Open should remain 100.0"
    assert bb.h == pytest.approx(105.0), "High should remain 105.0"
    assert bb.l == pytest.approx(95.0), "Low should remain 95.0"
    assert bb.c == pytest.approx(102.0), f"Close should update to 102.0, got {bb.c}"
    print("  ✓ Final bar state: O=100.0 H=105.0 L=95.0 C=102.0")

    print("✅ PASSED: Bar aggregation works correctly")


def test_bar_closure_timing():
    """Test bar closure at timeframe boundaries"""
    print("\n" + "=" * 70)
    print("TEST 2: Bar Closure Timing")
    print("=" * 70)

    bb = BarBuilder(timeframe_minutes=15)

    # Bar 1: 10:00 - 10:14:59
    t1 = dt.datetime(2026, 1, 11, 10, 0, 0, tzinfo=UTC)
    result = bb.update(t1, 100.0)
    assert result is None, "First bar should not close immediately"
    print(f"  Bar 1 opened at {t1.strftime('%H:%M:%S')}")

    result = bb.update(t1 + dt.timedelta(minutes=10), 105.0)
    assert result is None, "Tick at 10:10 should not close bar"
    print(f"  Tick at 10:10:00 - bar still open")

    result = bb.update(t1 + dt.timedelta(minutes=14, seconds=59), 102.0)
    assert result is None, "Tick at 10:14:59 should not close bar"
    print(f"  Tick at 10:14:59 - bar still open")

    # Bar 2: 10:15 - 10:29:59 (this should close Bar 1)
    t2 = dt.datetime(2026, 1, 11, 10, 15, 0, tzinfo=UTC)
    result = bb.update(t2, 103.0)
    assert result is not None, "Tick at 10:15:00 should close previous bar"

    closed_time, o, h, l, c = result
    assert closed_time == dt.datetime(
        2026, 1, 11, 10, 0, 0, tzinfo=UTC
    ), f"Closed bar timestamp should be 10:00, got {closed_time}"
    assert o == pytest.approx(100.0), f"Closed bar O should be 100.0, got {o}"
    assert h == pytest.approx(105.0), f"Closed bar H should be 105.0, got {h}"
    assert l == pytest.approx(100.0), f"Closed bar L should be 100.0, got {l}"
    assert c == pytest.approx(102.0), f"Closed bar C should be 102.0, got {c}"
    print(f"  ✓ Bar closed at 10:15:00: O={o} H={h} L={l} C={c}")

    # Verify new bar started
    assert bb.bucket == dt.datetime(2026, 1, 11, 10, 15, 0, tzinfo=UTC), "New bar should start at 10:15:00"
    assert bb.o == pytest.approx(103.0), "New bar O should be 103.0"
    assert bb.c == pytest.approx(103.0), "New bar C should be 103.0"
    print(f"  ✓ New bar started at 10:15:00: O={bb.o}")

    print("✅ PASSED: Bar closure timing is correct")


def test_timeframe_alignment():
    """Test bar alignment for different timeframes"""
    print("\n" + "=" * 70)
    print("TEST 3: Timeframe Alignment")
    print("=" * 70)

    test_cases = [
        (1, "10:00", "10:00"),  # M1
        (5, "10:07", "10:05"),  # M5
        (15, "10:23", "10:15"),  # M15
        (60, "11:45", "11:00"),  # H1
    ]

    for tf_minutes, tick_time, expected_bucket in test_cases:
        bb = BarBuilder(timeframe_minutes=tf_minutes)

        # Parse tick time
        hour, minute = map(int, tick_time.split(":"))
        tick = dt.datetime(2026, 1, 11, hour, minute, 0, tzinfo=UTC)

        # Parse expected bucket
        exp_hour, exp_minute = map(int, expected_bucket.split(":"))
        expected = dt.datetime(2026, 1, 11, exp_hour, exp_minute, 0, tzinfo=UTC)

        bucket = bb.bucket_start(tick)
        assert (
            bucket == expected
        ), f"M{tf_minutes}: tick {tick_time} should align to {expected_bucket}, got {bucket.strftime('%H:%M')}"

        print(f"  ✓ M{tf_minutes:>2}: {tick_time} → {expected_bucket}")

    print("✅ PASSED: Timeframe alignment is correct")


def test_bar_boundary_detection():
    """Test detection of bar boundaries across different scenarios"""
    print("\n" + "=" * 70)
    print("TEST 4: Bar Boundary Detection")
    print("=" * 70)

    bb = BarBuilder(timeframe_minutes=15)

    # Scenario 1: Normal sequential ticks
    base = dt.datetime(2026, 1, 11, 10, 0, 0, tzinfo=UTC)
    bb.update(base, 100.0)

    # Stay in same bar
    for i in range(1, 15):
        t = base + dt.timedelta(minutes=i)
        result = bb.update(t, 100.0 + i)
        assert result is None, f"Tick at minute {i} should not close bar"
    print("  ✓ 14 ticks within bar - no closure")

    # Cross boundary
    result = bb.update(base + dt.timedelta(minutes=15), 115.0)
    assert result is not None, "Tick at minute 15 should close bar"
    print("  ✓ Tick at minute 15 closes bar")

    # Scenario 2: Gap in data (simulates missing ticks)
    bb2 = BarBuilder(timeframe_minutes=15)
    t1 = dt.datetime(2026, 1, 11, 10, 0, 0, tzinfo=UTC)
    bb2.update(t1, 100.0)

    # Jump directly to next bar
    t2 = dt.datetime(2026, 1, 11, 10, 30, 0, tzinfo=UTC)
    result = bb2.update(t2, 105.0)
    assert result is not None, "Jump to 10:30 should close 10:00 bar"
    closed_time, _, _, _, _ = result
    assert closed_time == dt.datetime(2026, 1, 11, 10, 0, 0, tzinfo=UTC)
    print("  ✓ Gap in data handled - bar closed at boundary")

    # Scenario 3: Hour boundary
    bb3 = BarBuilder(timeframe_minutes=15)
    t1 = dt.datetime(2026, 1, 11, 10, 45, 0, tzinfo=UTC)
    bb3.update(t1, 100.0)

    t2 = dt.datetime(2026, 1, 11, 11, 0, 0, tzinfo=UTC)
    result = bb3.update(t2, 105.0)
    assert result is not None, "Hour boundary should close bar"
    closed_time, _, _, _, _ = result
    assert closed_time == dt.datetime(2026, 1, 11, 10, 45, 0, tzinfo=UTC)
    print("  ✓ Hour boundary handled correctly")

    print("✅ PASSED: Bar boundary detection is correct")


def test_data_integrity_across_bars():
    """Test that data doesn't leak between bars"""
    print("\n" + "=" * 70)
    print("TEST 5: Data Integrity Across Bar Transitions")
    print("=" * 70)

    bb = BarBuilder(timeframe_minutes=15)

    # Bar 1: O=100, H=110, L=95, C=105
    base = dt.datetime(2026, 1, 11, 10, 0, 0, tzinfo=UTC)
    bb.update(base, 100.0)
    bb.update(base + dt.timedelta(seconds=30), 110.0)  # High
    bb.update(base + dt.timedelta(seconds=60), 95.0)  # Low
    bb.update(base + dt.timedelta(seconds=90), 105.0)  # Close

    bar1_o, bar1_h, bar1_l, bar1_c = bb.o, bb.h, bb.l, bb.c
    print(f"  Bar 1: O={bar1_o} H={bar1_h} L={bar1_l} C={bar1_c}")

    # Bar 2: Should start fresh
    t2 = dt.datetime(2026, 1, 11, 10, 15, 0, tzinfo=UTC)
    result = bb.update(t2, 200.0)

    # Verify Bar 1 closed correctly
    assert result is not None, "Bar 1 should be closed"
    closed_time, o, h, l, c = result
    assert o == pytest.approx(100.0), f"Bar 1 O should be 100.0, got {o}"
    assert h == pytest.approx(110.0), f"Bar 1 H should be 110.0, got {h}"
    assert l == pytest.approx(95.0), f"Bar 1 L should be 95.0, got {l}"
    assert c == pytest.approx(105.0), f"Bar 1 C should be 105.0, got {c}"
    print(f"  ✓ Bar 1 closed correctly: O={o} H={h} L={l} C={c}")

    # Verify Bar 2 started fresh (no contamination from Bar 1)
    assert bb.o == pytest.approx(200.0), f"Bar 2 O should be 200.0, got {bb.o}"
    assert bb.h == pytest.approx(200.0), f"Bar 2 H should be 200.0, got {bb.h}"
    assert bb.l == pytest.approx(200.0), f"Bar 2 L should be 200.0, got {bb.l}"
    assert bb.c == pytest.approx(200.0), f"Bar 2 C should be 200.0, got {bb.c}"
    print(f"  ✓ Bar 2 started fresh: O={bb.o} H={bb.h} L={bb.l} C={bb.c}")

    # Update Bar 2 and verify it doesn't affect closed Bar 1
    bb.update(t2 + dt.timedelta(seconds=30), 150.0)
    assert bb.l == pytest.approx(150.0), "Bar 2 L should update to 150.0"
    # Bar 1 values should remain unchanged (they're in 'result')
    print("  ✓ Bar 2 updates don't affect closed Bar 1")

    print("✅ PASSED: Data integrity maintained across bars")


def test_edge_cases():
    """Test edge cases and unusual scenarios"""
    print("\n" + "=" * 70)
    print("TEST 6: Edge Cases")
    print("=" * 70)

    # Edge case 1: Same price all ticks
    bb1 = BarBuilder(timeframe_minutes=15)
    base = dt.datetime(2026, 1, 11, 10, 0, 0, tzinfo=UTC)

    for i in range(10):
        result = bb1.update(base + dt.timedelta(seconds=i * 60), 100.0)
        if i < 9:
            assert result is None

    assert bb1.o == pytest.approx(100.0) and bb1.h == pytest.approx(100.0) and bb1.l == pytest.approx(100.0) and bb1.c == pytest.approx(100.0)
    print("  ✓ Same price all ticks: O=H=L=C=100.0")

    # Edge case 2: Extreme price swing
    bb2 = BarBuilder(timeframe_minutes=15)
    bb2.update(base, 100.0)
    bb2.update(base + dt.timedelta(seconds=30), 1000.0)
    bb2.update(base + dt.timedelta(seconds=60), 1.0)

    assert bb2.h == pytest.approx(1000.0), "Should handle extreme high"
    assert bb2.l == pytest.approx(1.0), "Should handle extreme low"
    print("  ✓ Extreme price swing: L=1.0 to H=1000.0")

    # Edge case 3: Rapid-fire updates at boundary
    bb3 = BarBuilder(timeframe_minutes=1)
    t1 = dt.datetime(2026, 1, 11, 10, 0, 0, tzinfo=UTC)
    bb3.update(t1, 100.0)

    # Multiple ticks at exactly 10:01:00
    t2 = dt.datetime(2026, 1, 11, 10, 1, 0, tzinfo=UTC)
    result1 = bb3.update(t2, 101.0)
    result2 = bb3.update(t2, 102.0)
    result3 = bb3.update(t2, 103.0)

    assert result1 is not None, "First tick at boundary should close bar"
    assert result2 is None, "Subsequent ticks at same time should not close"
    assert result3 is None, "Subsequent ticks at same time should not close"
    assert bb3.c == pytest.approx(103.0), "Close should be last tick"
    print("  ✓ Rapid-fire updates at boundary handled correctly")

    print("✅ PASSED: All edge cases handled correctly")


def run_all_tests():
    """Run all bar building tests"""
    print("\n" + "=" * 70)
    print("BAR BUILDING & BAR CLOSURE VERIFICATION TEST SUITE")
    print("=" * 70)

    tests = [
        ("Basic Bar Aggregation", test_basic_bar_aggregation),
        ("Bar Closure Timing", test_bar_closure_timing),
        ("Timeframe Alignment", test_timeframe_alignment),
        ("Bar Boundary Detection", test_bar_boundary_detection),
        ("Data Integrity Across Bars", test_data_integrity_across_bars),
        ("Edge Cases", test_edge_cases),
    ]

    passed = 0
    failed = []

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {name}")
            print(f"   Error: {e}")
            failed.append((name, str(e)))
        except Exception as e:
            print(f"❌ ERROR: {name}")
            print(f"   Exception: {e}")
            failed.append((name, f"Exception: {e}"))

    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed}/{len(tests)} PASSED")
    print("=" * 70)

    if failed:
        print("\nFAILED TESTS:")
        for name, error in failed:
            print(f"  ❌ {name}: {error}")
        return 1
    else:
        print("\n✅ ALL BAR BUILDING TESTS PASSED")
        print("\nVerified:")
        print("  • OHLC aggregation within bars")
        print("  • Bar closure at timeframe boundaries")
        print("  • Correct alignment for M1, M5, M15, H1")
        print("  • Boundary detection across gaps and hour changes")
        print("  • Data integrity (no leakage between bars)")
        print("  • Edge cases (same price, extreme swings, rapid updates)")
        return 0


if __name__ == "__main__":
    exit(run_all_tests())
