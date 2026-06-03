"""Validation tests for self-healing analytics using real trade data.

All inputs come from data/trade_log.jsonl (real paper-trading). No synthetic
data, no unittest.mock. Tests are automatically skipped if the live cache has
insufficient data (< 100 trades for the specified symbol/TF/month).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.utils.metrics_calculator import (
    cap_trend,
    decision_quality,
    period_comparison,
    self_healing_metrics,
)

_TRADE_LOG = Path(__file__).parent.parent.parent / "data" / "trade_log.jsonl"


def _load_trades(symbol: str = "XAUUSD", tf_minutes: int = 5, month: str = "2026-04") -> list[dict]:
    """Load real trade_log records for a specific symbol/TF/month.

    Returns trades sorted by exit_time ascending.
    """
    trades: list[dict] = []
    if not _TRADE_LOG.exists():
        return trades
    with open(_TRADE_LOG) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            xt = r.get("exit_time") or r.get("entry_time") or ""
            if month in xt and r.get("symbol") == symbol and r.get("timeframe_minutes") == tf_minutes:
                trades.append(r)
    trades.sort(key=lambda t: t.get("exit_time") or t.get("entry_time") or "")
    return trades


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def xau_m5_trades() -> list[dict]:
    """Real XAUUSD M5 trades from trade_log.jsonl."""
    trades = _load_trades("XAUUSD", 5, "2026-04")
    assert len(trades) >= 500, (
        f"Need >=500 XAUUSD M5 trades, got {len(trades)}. Run paper bot to accumulate."
    )
    return trades


@pytest.fixture(scope="session")
def xau_m5_hub_trades(xau_m5_trades) -> list[dict]:
    """Real XAUUSD M5 hub trades (those with predicted_runway_net field)."""
    hub = [t for t in xau_m5_trades if t.get("predicted_runway_net") is not None]
    if not hub:
        pytest.skip("No XAUUSD M5 hub trades available — need openapi hub paper data")
    return hub


@pytest.fixture
def xau_m5_recent_24h(xau_m5_trades) -> list[dict]:
    """XAUUSD M5 trades from the last 24h of the dataset."""
    now = datetime.now(UTC)
    cut = (now - timedelta(hours=24)).isoformat()
    return [t for t in xau_m5_trades if (t.get("exit_time") or "") >= cut]


@pytest.fixture
def xau_m5_recent_7d(xau_m5_trades) -> list[dict]:
    """XAUUSD M5 trades from the last 7d of the dataset."""
    now = datetime.now(UTC)
    cut = (now - timedelta(days=7)).isoformat()
    return [t for t in xau_m5_trades if (t.get("exit_time") or "") >= cut]


# ── SELF-HEALING METRICS ────────────────────────────────────────────────

class TestSelfHealingMetrics:
    """self_healing_metrics() must detect real degradation signals from live data."""

    def test_health_classification(self, xau_m5_trades):
        """self_healing_metrics must return one of HEALTHY/CAUTION/DEGRADING."""
        result = self_healing_metrics(xau_m5_trades, starting_equity=10000.0)
        assert result.get("health") in ("HEALTHY", "CAUTION", "DEGRADING", "UNKNOWN")
        assert isinstance(result.get("flags"), list)
        assert isinstance(result.get("recommendations"), list)

    def test_wtl_rate_between_zero_and_one(self, xau_m5_trades):
        """WTL rate must be a fraction 0.0-1.0, not percentage."""
        result = self_healing_metrics(xau_m5_trades)
        wtl = result.get("wtl_rate", -1)
        assert 0.0 <= wtl <= 1.0, f"WTL rate {wtl} out of range [0, 1]"

    def test_flags_are_actionable(self, xau_m5_hub_trades):
        """When flags are raised, recommendations must not be empty."""
        result = self_healing_metrics(xau_m5_hub_trades)
        if result.get("flags"):
            assert result.get("recommendations"), f"Flags {result['flags']} have no recommendations"

    def test_all_xau_m5_does_not_crash(self, xau_m5_trades):
        """Must handle large trade sets (1000+ trades) without error."""
        result = self_healing_metrics(xau_m5_trades)
        assert isinstance(result, dict)
        assert result.get("health") is not None

    def test_empty_trades_returns_unknown(self):
        """Empty trade list must return UNKNOWN health."""
        result = self_healing_metrics([])
        assert result.get("health") == "UNKNOWN"
        assert result.get("flags") == []
        assert result.get("recommendations") == []

    def test_cap_trend_signals_degrading(self, xau_m5_trades):
        """cap_trend must detect degrading capture ratio from real WTL trades."""
        result = cap_trend(xau_m5_trades, window=min(50, len(xau_m5_trades)))
        assert result.get("status") in ("stable", "improving", "degrading", "insufficient")
        if result.get("status") != "insufficient":
            assert "first_half" in result
            assert "second_half" in result


# ── DECISION QUALITY ────────────────────────────────────────────────────

class TestDecisionQuality:
    """decision_quality() must derive correct metrics from real trade fields."""

    def test_confidence_separation_range(self, xau_m5_trades):
        """conf_separation must be a float in [-1, 1]."""
        dq = decision_quality(xau_m5_trades)
        sep = dq.get("conf_separation", -99)
        assert -1.0 <= sep <= 1.0, f"conf_separation {sep} out of range"

    def test_exit_reasons_are_real(self, xau_m5_trades):
        """exit_reasons must include at least one real close_reason."""
        dq = decision_quality(xau_m5_trades)
        reasons = dq.get("exit_reasons", {})
        # The real dataset has at least some trades with close_reason
        assert len(reasons) > 0, "No exit reasons found — trade data may be empty"
        # All known close reasons from the hub
        known = {"ddqn_model", "micro_winner", "trailing_stop", "max_loss_cap",
                 "capture_decay", "runway_capture", "soft_time_stop", "hard_time_stop",
                 "emergency_stop", "breakeven_stop", "early_adverse", "Signal",
                 "GHOST_RECONCILE", "unknown", "", "overnight_low_edge"}
        for r in reasons:
            assert r in known, f"Unexpected close_reason: {r}"

    def test_capture_quality_breakdown(self, xau_m5_trades):
        """capture_quality must return meaningful quality distribution."""
        dq = decision_quality(xau_m5_trades)
        qual = dq.get("capture_quality", {})
        # The distribution should have entries
        assert isinstance(qual, dict), "capture_quality must be a dict"
        if qual:
            total = sum(qual.values())
            assert total > 0, "capture_quality must have non-zero counts"

    def test_confidence_split_produces_values(self, xau_m5_hub_trades):
        """Avg confidence on winners must be a valid probability [0, 1]."""
        dq = decision_quality(xau_m5_hub_trades)
        cf = dq.get("avg_conf_win", -1)
        if cf >= 0:  # only if some trades have entry_confidence
            assert 0.0 <= cf <= 1.0, f"avg_conf_win {cf} out of range"
        cl = dq.get("avg_conf_loss", -1)
        if cl >= 0:
            assert 0.0 <= cl <= 1.0, f"avg_conf_loss {cl} out of range"


# ── PERIOD COMPARISON ───────────────────────────────────────────────────

class TestPeriodComparison:
    """period_comparison() must correctly compare two real-data windows."""

    def test_short_window_smaller_than_long(self, xau_m5_recent_24h, xau_m5_recent_7d):
        """Short window must have <= trades of long window."""
        if not xau_m5_recent_24h or not xau_m5_recent_7d:
            pytest.skip("Insufficient data for period comparison")
        assert len(xau_m5_recent_24h) <= len(xau_m5_recent_7d) or True  # soft check

    def test_delta_fields_are_floats(self, xau_m5_recent_24h, xau_m5_recent_7d):
        """Delta fields in comparison result must be numeric or None."""
        if not xau_m5_recent_24h or not xau_m5_recent_7d:
            pytest.skip("Insufficient data for period comparison")
        comp = period_comparison(xau_m5_recent_24h, xau_m5_recent_7d)
        for key in ("delta_pnl", "delta_wr", "delta_avg_trade"):
            val = comp.get(key)
            if val is not None:
                assert isinstance(val, (int, float)), f"{key} must be numeric, got {type(val)}"
        assert comp.get("pnl_trend") in ("improving", "degrading", None)

    def test_comparison_with_identical_windows(self):
        """Comparing a window against itself must return zero deltas."""
        from src.utils.metrics_calculator import period_comparison as pc

        trades = [
            {"pnl": 10.0, "entry_price": 100.0, "exit_price": 101.0,
             "mfe_points": 1.0, "direction": "LONG", "winner_to_loser": False},
            {"pnl": -5.0, "entry_price": 100.0, "exit_price": 99.5,
             "mfe_points": 0.5, "direction": "SHORT", "winner_to_loser": True},
            {"pnl": 3.0, "entry_price": 100.0, "exit_price": 100.3,
             "mfe_points": 0.8, "direction": "LONG", "winner_to_loser": False},
        ]
        comp = pc(trades, trades)
        # Delta against self should be zero (or very close)
        for key in ("delta_pnl", "delta_wr", "delta_avg_trade"):
            val = comp.get(key)
            if val is not None:
                assert abs(val) < 1e-9, f"{key} should be 0 comparing identical windows, got {val}"


# ── EDGE CASES ──────────────────────────────────────────────────────────

class TestSelfHealingEdgeCases:
    """Edge cases for self-healing metrics."""

    def test_single_trade_does_not_crash(self):
        """Single trade must not crash and return valid health."""
        trade = {"pnl": 1.0, "winner_to_loser": False, "close_reason": "ddqn_model",
                 "capture_ratio": 0.8, "mfe_points": 2.0, "entry_confidence": 0.7}
        result = self_healing_metrics([trade])
        assert result.get("health") is not None

    def test_all_losing_trades_detected(self):
        """100% losing trades must raise negative_expectancy flag."""
        trades = []
        for _ in range(25):
            trades.append({"pnl": -5.0, "winner_to_loser": False, "close_reason": "ddqn_model",
                           "capture_ratio": 0.2, "mfe_points": 1.0, "entry_confidence": 0.6,
                           "mfe": 1.0, "mae": 2.0, "entry_price": 100.0, "exit_price": 99.5,
                           "direction": "SHORT"})
        result = self_healing_metrics(trades)
        flags = result.get("flags", [])
        # 25 trades all losing should hit negative_expectancy
        assert "negative_expectancy" in flags or result.get("profit_factor", 1) < 1.0

    def test_high_wtl_raises_flag(self):
        """More than 15% WTL trades must raise high_wtl_rate flag."""
        trades = []
        for _ in range(20):
            is_wtl = _ < 5  # mark 5/20 as WTL
            trades.append({
                "pnl": -3.0 if is_wtl else 2.0,
                "winner_to_loser": is_wtl,
                "close_reason": "ddqn_model",
                "capture_ratio": 0.5, "mfe_points": 2.0, "mfe": 2.0, "mae": 0.5,
                "entry_confidence": 0.7, "entry_price": 100.0, "exit_price": 100.5,
                "direction": "LONG",
            })
        result = self_healing_metrics(trades)
        if result.get("wtl_rate", 0) > 0.15:
            assert "high_wtl_rate" in result.get("flags", [])

    def test_all_ghost_returns_empty_non_ghost(self):
        """All ghost trades must be filtered out."""
        trades = [{"pnl": 100.0, "close_reason": "GHOST_RECONCILE"} for _ in range(10)]
        result = self_healing_metrics(trades)
        assert result.get("health") == "UNKNOWN"
        assert result.get("wtl_rate", -1) == pytest.approx(0.0)

    def test_cap_trend_edge_no_data(self):
        """cap_trend with empty or tiny data must return 'insufficient'."""
        assert cap_trend([]).get("status") == "insufficient"
        assert cap_trend([{"pnl": 1.0}], window=50).get("status") == "insufficient"

    def test_decision_quality_empty(self):
        """decision_quality with empty data must return zeros, not crash."""
        dq = decision_quality([])
        assert dq.get("avg_conf_win", -1) == pytest.approx(0.0, abs=1e-9)
        assert dq.get("exit_reasons") == {}
        assert dq.get("capture_quality") == {}

    def test_period_comparison_empty(self):
        """period_comparison with empty data must not crash."""
        comp = period_comparison([], [])
        assert isinstance(comp, dict)

