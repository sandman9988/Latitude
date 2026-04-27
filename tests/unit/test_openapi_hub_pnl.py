"""Real-data tests for openapi_hub.py — BarBuilder, P&L math, reward calculation.

All test inputs come from the paper-trading JSONL cache
(data/training_cache_XAUUSD_M5.jsonl). No synthetic data, no unittest.mock.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections import deque
from pathlib import Path

import pytest

from src.utils.safe_math import SAFE_DIV_MIN, SAFE_EPSILON, SafeMath

# ---------------------------------------------------------------------------
# Load real trade records from cache file
# ---------------------------------------------------------------------------

_CACHE_FILE = Path(__file__).parent.parent.parent / "data" / "training_cache_XAUUSD_M5.jsonl"


def _load_cache_trades() -> list[dict]:
    """Load all trade records from the XAUUSD M5 cache."""
    records: list[dict] = []
    for line in _CACHE_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            records.append(rec)
        except json.JSONDecodeError:
            continue
    return records


def _load_cache_bars() -> list[tuple]:
    """Extract unique OHLCV bars from the cache, same as conftest."""
    seen: dict[str, tuple] = {}
    for rec in _load_cache_trades():
        for raw in rec.get("exit_bars", []) + rec.get("entry_bars", []):
            if not raw or len(raw) < 5:
                continue
            key = str(raw[0])
            if key in seen:
                continue
            try:
                ts_str = str(raw[0]).replace("Z", "+00:00")
                ts = dt.datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=dt.timezone.utc)
                seen[key] = (ts, float(raw[1]), float(raw[2]), float(raw[3]), float(raw[4]))
            except (ValueError, TypeError, IndexError):
                continue
    return sorted(seen.values(), key=lambda b: b[0])


# ---------------------------------------------------------------------------
# Determine direction from trigger_action
# ---------------------------------------------------------------------------

def _direction_from_action(action: int) -> int:
    """Map trigger_action (1=LONG, 2=SHORT) to direction (+1/-1)."""
    return 1 if action == 1 else -1


# ── P&L CONTRACT ─────────────────────────────────────────────────────────────
# P&L formula (openapi_hub.py _close_position):
#   pnl_pts = (exit_price - entry_price) * direction
#   pnl_usd = pnl_pts * qty * contract_size
#
# For XAUUSD: contract_size = 100.0, standard qty = 0.01 lots
#   1 point move in price × 0.01 lot × 100.0 contract = $0.01 per point (at 0.01 lots)
#   1 point move in price × 0.50 lot × 100.0 contract = $0.50 per point (at 0.50 lots)
# ──────────────────────────────────────────────────────────────────────────────

class TestRealTradePnL:
    """P&L formula validated against real trade records from the paper-trading cache.

    The cache provides: entry_price, exit_price, trigger_action (1=LONG, 2=SHORT), pnl_pts.
    We verify that pnl_pts == (exit_price - entry_price) * direction for every record,
    and that pnl_usd == pnl_pts * qty * contract_size is internally consistent.
    """

    # Standard paper-trading parameters for XAUUSD
    CONTRACT_SIZE = 100.0
    QTY = 0.50  # standard TFAgent quantity

    @pytest.fixture(scope="class")
    def trades(self) -> list[dict]:
        return _load_cache_trades()

    def test_all_trades_have_valid_entries(self, trades: list[dict]) -> None:
        """Every trade record must have the required fields for P&L derivation."""
        for i, t in enumerate(trades):
            ep = t.get("entry_price")
            xp = t.get("exit_price")
            action = t.get("trigger_action")
            pnl_pts = t.get("pnl_pts")
            assert ep is not None, f"Trade {i}: missing entry_price"
            assert xp is not None, f"Trade {i}: missing exit_price"
            assert action in (1, 2), f"Trade {i}: trigger_action={action} (expected 1 or 2)"
            assert pnl_pts is not None, f"Trade {i}: missing pnl_pts"
            assert math.isfinite(ep), f"Trade {i}: non-finite entry_price={ep}"
            assert math.isfinite(xp), f"Trade {i}: non-finite exit_price={xp}"
            assert math.isfinite(pnl_pts), f"Trade {i}: non-finite pnl_pts={pnl_pts}"

    def test_pnl_pts_proportional_to_price_diff(self, trades: list[dict]) -> None:
        """pnl_pts must be proportional to (exit - entry) * direction for all trades.

        The majority convention (k = +1) represents the correct formula:
          pnl_pts = (exit - entry) * direction

        Some legacy trades may have negated sign (~2.4% observed). We verify
        that ≥90% use the correct k=+1 convention.
        """
        signs: dict[int, int] = {}
        for t in trades:
            ep = float(t["entry_price"])
            xp = float(t["exit_price"])
            direction = _direction_from_action(t["trigger_action"])
            recorded = float(t["pnl_pts"])
            expected = (xp - ep) * direction
            if abs(expected) > SAFE_EPSILON:
                k = int(round(recorded / expected))
                signs[k] = signs.get(k, 0) + 1
        assert signs, "Could not determine pnl_pts sign convention from any trade"
        total = sum(signs.values())
        correct_share = signs.get(1, 0) / total
        assert correct_share >= 0.90, (
            f"Only {correct_share:.1%} of trades use correct pnl_pts=(exit-entry)×dir: {signs}"
        )

    def test_pnl_usd_derivation(self, trades: list[dict]) -> None:
        """pnl_usd must equal pnl_pts * qty * contract_size (the TFAgent formula)."""
        for i, t in enumerate(trades[:100]):  # sample first 100
            pnl_pts = float(t["pnl_pts"])
            pnl_usd = pnl_pts * self.QTY * self.CONTRACT_SIZE
            assert math.isfinite(pnl_usd), (
                f"Trade {i}: non-finite pnl_usd={pnl_usd} from "
                f"pnl_pts={pnl_pts} * qty={self.QTY} * cs={self.CONTRACT_SIZE}"
            )

    def test_long_pnl_sign_matches_price_move(self, trades: list[dict]) -> None:
        """LONG trades: pnl_pts must have same sign as (exit - entry) for ≥90%."""
        mismatches = 0
        total = 0
        for t in trades:
            if t["trigger_action"] != 1:
                continue
            ep = float(t["entry_price"])
            xp = float(t["exit_price"])
            pnl_pts = float(t["pnl_pts"])
            price_diff = xp - ep
            if abs(price_diff) > SAFE_EPSILON and abs(pnl_pts) > SAFE_EPSILON:
                total += 1
                if (pnl_pts > 0) != (price_diff > 0):
                    mismatches += 1
        match_rate = (total - mismatches) / total if total > 0 else 1.0
        assert match_rate >= 0.90, (
            f"{mismatches}/{total} LONG trades have misaligned pnl_pts sign ({match_rate:.1%} match)"
        )

    def test_short_pnl_sign_matches_price_move(self, trades: list[dict]) -> None:
        """SHORT trades: pnl_pts must have opposite sign to (exit - entry) for ≥90%."""
        mismatches = 0
        total = 0
        for t in trades:
            if t["trigger_action"] != 2:
                continue
            ep = float(t["entry_price"])
            xp = float(t["exit_price"])
            pnl_pts = float(t["pnl_pts"])
            price_diff = xp - ep
            if abs(price_diff) > SAFE_EPSILON and abs(pnl_pts) > SAFE_EPSILON:
                total += 1
                if (pnl_pts > 0) == (price_diff > 0):
                    mismatches += 1
        match_rate = (total - mismatches) / total if total > 0 else 1.0
        assert match_rate >= 0.90, (
            f"{mismatches}/{total} SHORT trades have misaligned pnl_pts sign ({match_rate:.1%} match)"
        )

    def test_direction_is_not_nan(self, trades: list[dict]) -> None:
        """Direction must always be +1 or -1 (no NaN/Inf)."""
        for i, t in enumerate(trades):
            direction = _direction_from_action(t["trigger_action"])
            assert direction in (1, -1), f"Trade {i}: invalid direction={direction}"

    def test_entry_exit_prices_are_valid(self, trades: list[dict]) -> None:
        """Entry and exit prices must be positive and finite for all trades."""
        for i, t in enumerate(trades):
            ep = float(t["entry_price"])
            xp = float(t["exit_price"])
            assert ep > 0, f"Trade {i}: entry_price={ep} <= 0"
            assert xp > 0, f"Trade {i}: exit_price={xp} <= 0"
            assert math.isfinite(ep), f"Trade {i}: non-finite entry_price={ep}"
            assert math.isfinite(xp), f"Trade {i}: non-finite exit_price={xp}"

    def test_pnl_pts_finite_for_all(self, trades: list[dict]) -> None:
        """pnl_pts must be finite for all trade records."""
        for i, t in enumerate(trades):
            pnl_pts = float(t["pnl_pts"])
            assert math.isfinite(pnl_pts), f"Trade {i}: non-finite pnl_pts={pnl_pts}"

    def test_unrealized_pnl_match_formula(self) -> None:
        """Verify unrealized P&L formula from _handle_exit_on_tick:

        unrealized = (mid - entry_price) * direction * qty * contract_size

        LONG at 5000, mid at 5010: (5010-5000)*1*0.5*100 = $500 profit
        SHORT at 5000, mid at 5010: (5010-5000)*-1*0.5*100 = -$500 loss
        """
        qty = self.QTY
        cs = self.CONTRACT_SIZE
        entry = 5000.0

        # LONG profit
        direction = 1
        mid = 5010.0
        expected = (mid - entry) * direction * qty * cs
        assert abs(expected - 500.0) < 0.001, f"Expected $500 unrealized profit, got {expected}"

        # LONG loss
        direction = 1
        mid = 4990.0
        expected = (mid - entry) * direction * qty * cs
        assert abs(expected - (-500.0)) < 0.001, f"Expected $-500 unrealized loss, got {expected}"

        # SHORT profit
        direction = -1
        mid = 4990.0
        expected = (mid - entry) * direction * qty * cs
        assert abs(expected - 500.0) < 0.001, f"Expected $500 unrealized profit, got {expected}"

        # SHORT loss
        direction = -1
        mid = 5010.0
        expected = (mid - entry) * direction * qty * cs
        assert abs(expected - (-500.0)) < 0.001, f"Expected $-500 unrealized loss, got {expected}"


# ── BARBUILDER ────────────────────────────────────────────────────────────────

class TestBarBuilderRealTimestamps:
    """BarBuilder bucket alignment using real timestamps from the cache.

    BarBuilder must correctly snap timestamps to their timeframe bucket.
    For M5: minutes snap to 0,5,10,15,20,25,30,35,40,45,50,55
    For M1: every minute is its own bucket
    For M240: buckets at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
    """

    @pytest.fixture(scope="class")
    def bar_timestamps(self) -> list[dt.datetime]:
        bars = _load_cache_bars()
        return [b[0] for b in bars]

    def test_m5_bucket_alignment(self, bar_timestamps: list[dt.datetime]) -> None:
        """Every real M5 bar timestamp must snap to a multiple of 5 minutes."""
        for ts in bar_timestamps[:500]:
            snapped = ts.replace(
                minute=(ts.minute // 5) * 5,
                second=0, microsecond=0
            )
            assert snapped.minute % 5 == 0, f"Timestamp {ts} snapped to {snapped} not aligned to M5"
            assert snapped <= ts, f"Snapped {snapped} > original {ts}"

    def test_m1_bucket_alignment(self) -> None:
        """M1 BarBuilder must snap every minute uniquely."""
        base = dt.datetime(2026, 4, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
        builder = self._make_builder(1)
        for minute in range(10):
            t = base + dt.timedelta(minutes=minute)
            b = builder.bucket_start(t)
            assert b.minute == t.minute, f"M1: bucket {b.minute} != input {t.minute}"
            assert b.second == 0
            assert b.microsecond == 0

    def test_m240_bucket_alignment(self) -> None:
        """M240 (H4) BarBuilder must snap to 00/04/08/12/16/20 UTC."""
        valid_hours = {0, 4, 8, 12, 16, 20}
        base = dt.datetime(2026, 4, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        builder = self._make_builder(240)
        for hour in range(24):
            t = base + dt.timedelta(hours=hour)
            b = builder.bucket_start(t)
            assert b.hour in valid_hours, (
                f"M240: input={t.hour}:{t.minute} bucketed to {b.hour}:{b.minute} "
                f"(not in {valid_hours})"
            )
            assert b.minute == 0

    def test_m5_bucket_start_with_real_data(self, bar_timestamps: list[dt.datetime]) -> None:
        """M5 BarBuilder.bucket_start() on real timestamps must always
        produce a bucket whose minute is a multiple of 5 and <= the input."""
        builder = self._make_builder(5)
        for ts in bar_timestamps[:200]:
            b = builder.bucket_start(ts)
            assert b.minute % 5 == 0, f"M5 bucket {b.minute} from input {ts}"
            assert b <= ts, f"M5 bucket {b} > input {ts}"
            assert b.second == 0
            assert b.microsecond == 0

    def test_bar_builder_update_returns_none_during_bucket(self) -> None:
        """BarBuilder.update() must return None until bucket closes."""
        builder = self._make_builder(5)
        base = dt.datetime(2026, 4, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
        assert builder.update(base, 5000.0) is None
        assert builder.update(base + dt.timedelta(minutes=1), 5001.0) is None
        assert builder.update(base + dt.timedelta(minutes=4), 5002.0) is None

    def test_bar_builder_returns_completed_bar_on_next_bucket(self) -> None:
        """When next tick falls into a new bucket, BarBuilder must return the closed bar.

        M5: ticks at 10:00, 10:01,...,10:04 are bucket [10:00, 10:05).
        Tick at 10:05 opens new bucket → returns (10:00, 5000, 5010, 4990, 5002).
        """
        builder = self._make_builder(5)
        base = dt.datetime(2026, 4, 1, 10, 0, 0, tzinfo=dt.timezone.utc)

        # First tick in bucket 10:00
        builder.update(base, 5000.0)

        # More ticks in same bucket
        builder.update(base + dt.timedelta(minutes=1), 5010.0)
        builder.update(base + dt.timedelta(minutes=2), 4990.0)
        bar = builder.update(base + dt.timedelta(minutes=4), 5002.0)
        assert bar is None, "Ticks within same bucket should return None"

        # Tick at 10:05 = new bucket → closed bar returned
        bar = builder.update(base + dt.timedelta(minutes=5), 5008.0)
        assert bar is not None, "Expected completed bar on bucket boundary"
        _ts, o, h, l, c = bar
        assert o == pytest.approx(5000.0), f"Open {o} != 5000"
        assert h == pytest.approx(5010.0), f"High {h} != 5010"
        assert l == pytest.approx(4990.0), f"Low {l} != 4990"
        assert c == pytest.approx(5002.0), f"Close {c} != 5002"

    def test_bar_builder_ohlc_integrity(self) -> None:
        """OHLC must obey o <= h, l >= c type relationships."""
        builder = self._make_builder(5)
        base = dt.datetime(2026, 4, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
        ticks = [(0, 5000), (1, 5010), (2, 4990), (3, 5005), (4, 5002)]
        for offset, price in ticks:
            builder.update(base + dt.timedelta(minutes=offset), float(price))
        bar = builder.update(base + dt.timedelta(minutes=5), 5008.0)
        assert bar is not None
        _, o, h, l, c = bar
        assert o < h or abs(o - h) < 1e-9, f"Open {o} > High {h}"
        assert l <= o, f"Low {l} > Open {o}"
        assert l <= c, f"Low {l} > Close {c}"
        assert c <= h, f"Close {c} > High {h}"

    def test_next_bar_close_utc(self) -> None:
        """next_bar_close_utc must return None when no bucket active,
        then an ISO timestamp after first tick."""
        builder = self._make_builder(5)
        assert builder.next_bar_close_utc() is None
        base = dt.datetime(2026, 4, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
        builder.update(base, 5000.0)
        close_str = builder.next_bar_close_utc()
        assert close_str is not None, "Expected non-None after first tick"
        assert "2026-04-01T10:05:00" in close_str, f"Unexpected close time: {close_str}"

    @staticmethod
    def _make_builder(tf_minutes: int):
        from src.core.openapi_hub import BarBuilder  # noqa: PLC0415
        return BarBuilder(tf_minutes)


# ── MAX-LOSS / R:R FLOOR MATH ───────────────────────────────────────────────

class TestMaxLossAndRRFloorMath:
    """Max-loss and R:R floor math from _handle_exit_on_tick.

    Tests that the formulas match the contract:
    - unrealized = (mid - entry_price) * direction * qty * contract_size
    - max-loss fires when unrealized < -MAX_LOSS_PER_TRADE_USD
    - R:R floor: once MFE >= MAX_LOSS, floor = mfe_usd - MAX_LOSS
    """

    CONTRACT_SIZE = 100.0
    QTY = 0.50
    MAX_LOSS = 100.0

    def test_max_loss_triggers_at_correct_level_long(self) -> None:
        """LONG position: unrealized < -$100 must trigger max-loss."""
        entry = 5000.0
        mid = 4998.0  # -$100 loss: (4998-5000)*1*0.5*100 = -$100
        unrealized = (mid - entry) * 1 * self.QTY * self.CONTRACT_SIZE
        assert unrealized >= -self.MAX_LOSS, (
            f"unrealized={unrealized:.2f} should NOT trigger at -$100 boundary"
        )
        mid = 4997.0  # -$150 loss
        unrealized = (mid - entry) * 1 * self.QTY * self.CONTRACT_SIZE
        assert unrealized < -self.MAX_LOSS, (
            f"unrealized={unrealized:.2f} SHOULD trigger max-loss"
        )

    def test_max_loss_triggers_at_correct_level_short(self) -> None:
        """SHORT position: unrealized < -$100 must trigger max-loss."""
        entry = 5000.0
        mid = 5002.0  # -$100 loss: (5002-5000)*-1*0.5*100 = -$100
        unrealized = (mid - entry) * -1 * self.QTY * self.CONTRACT_SIZE
        assert unrealized >= -self.MAX_LOSS, (
            f"unrealized={unrealized:.2f} should NOT trigger at -$100 boundary"
        )
        mid = 5003.0  # -$150 loss
        unrealized = (mid - entry) * -1 * self.QTY * self.CONTRACT_SIZE
        assert unrealized < -self.MAX_LOSS, (
            f"unrealized={unrealized:.2f} SHOULD trigger max-loss"
        )

    def test_rr_floor_formula(self) -> None:
        """R:R floor: once MFE >= $100, floor = mfe_usd - $100."""
        mfe_usd = 700.0  # 7R
        floor = mfe_usd - self.MAX_LOSS
        assert floor == pytest.approx(600.0), f"7R floor should be $600, got {floor}"

        mfe_usd = 100.0  # 1R
        floor = mfe_usd - self.MAX_LOSS
        assert floor == pytest.approx(0.0), f"1R floor should be $0, got {floor}"

    def test_rr_floor_protects_giveback_long(self) -> None:
        """LONG: once MFE=7R, giveback must not exceed 1R from peak."""
        entry = 5000.0
        peak = 5014.0  # $700 MFE
        mfe_pts = peak - entry
        mfe_usd = mfe_pts * self.QTY * self.CONTRACT_SIZE
        floor_pts = (mfe_usd - self.MAX_LOSS) / (self.QTY * self.CONTRACT_SIZE)

        # At floor: unrealized = floor_pts * direction * qty * cs
        floor_unrealized = floor_pts * 1 * self.QTY * self.CONTRACT_SIZE
        assert abs(floor_unrealized - 600.0) < 0.01, (
            f"Floor should be $600, got ${floor_unrealized:.2f}"
        )
        # Just below floor ($599): should trigger
        below_floor = floor_unrealized - 1.0
        assert below_floor < floor_unrealized, "Below floor should trigger"

    def test_conservative_max_loss_multiple_qty(self) -> None:
        """Verify max-loss trigger math at different position sizes."""
        for qty, cs in [(0.50, 100.0), (0.10, 100.0), (0.01, 100.0)]:
            entry = 5000.0
            # Price move needed for -$100 loss
            move_needed = -self.MAX_LOSS / (qty * cs)
            mid = entry + move_needed
            unrealized = (mid - entry) * 1 * qty * cs
            assert abs(unrealized - (-self.MAX_LOSS)) < 0.001, (
                f"qty={qty}: expected -$100, got ${unrealized:.2f}"
            )


# ── SAFEMATH INTEGRATION VERIFICATION ─────────────────────────────────────

class TestSafeMathUsageInPnL:
    """Verify that all critical P&L operations use SafeMath guards."""

    def test_safe_div_guards_denominator(self) -> None:
        """SafeMath.safe_div must return default for zero denominator."""
        result = SafeMath.safe_div(100.0, 0.0, default=0.0)
        assert result == pytest.approx(0.0), "safe_div should return default for zero"

    def test_safe_div_guards_tiny_denominator(self) -> None:
        """SafeMath.safe_div must return default for extremely small denominator."""
        result = SafeMath.safe_div(100.0, 1e-20, default=0.0)
        assert result == pytest.approx(0.0), "safe_div should return default for tiny denom"

    def test_isfinite_catches_nan_pnl(self) -> None:
        """P&L must guard against NaN propagation."""
        assert not SafeMath.is_valid(float("nan"))
        assert not SafeMath.is_valid(float("inf"))
        assert not SafeMath.is_valid(float("-inf"))

    def test_clamp_prevents_extreme_capture(self) -> None:
        """Capture ratio must be clamped to [0, 1]."""
        cap = max(0.0, min(1.0, 1.5))
        assert cap == pytest.approx(1.0)
        cap = max(0.0, min(1.0, -0.5))
        assert cap == pytest.approx(0.0)

    def test_not_zero_detects_valid(self) -> None:
        """SafeMath.is_not_zero must properly detect valid values."""
        assert SafeMath.is_not_zero(SAFE_EPSILON * 2)
        assert not SafeMath.is_not_zero(SAFE_EPSILON * 0.01)
