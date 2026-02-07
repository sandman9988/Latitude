"""Coverage batch 5: trade_analyzer + risk_manager edge cases.

Targets:
  - trade_analyzer.py   (98% → 99%): lines 176-178 (no predicted_runway/mfe), 184 (no capture_efficiency)
  - risk_manager.py     (94% → ~97%): lines 463, 477-480, 529, 642, 854, 957, 1230, 1322-1323
"""

import io
import time
from unittest.mock import patch

import numpy as np

rng = np.random.default_rng(42)

import pandas as pd
import pytest

# ── trade_analyzer ───────────────────────────────────────────────────────

from src.monitoring.trade_analyzer import TradeAnalyzer


class TestTradeAnalyzerGaps:
    """Cover else branches in analyze_dual_agents."""

    @staticmethod
    def _minimal_csv(extra_columns: dict | None = None) -> str:
        """Build a minimal CSV that satisfies TradeAnalyzer._validate_data."""
        rows = {
            "trade_num": [1, 2],
            "entry_time": ["2025-01-01 10:00:00", "2025-01-01 11:00:00"],
            "exit_time": ["2025-01-01 10:30:00", "2025-01-01 11:30:00"],
            "pnl": [10.0, -5.0],
            "result": ["WIN", "LOSS"],
            "trigger_quality": ["GOOD", "GOOD"],
            "harvester_quality": ["EXCELLENT", "FAIR"],
        }
        if extra_columns:
            rows.update(extra_columns)
        return pd.DataFrame(rows).to_csv(index=False)

    def test_no_predicted_runway_column(self, tmp_path):
        """Lines 177-178: No predicted_runway/mfe → avg_runway_error = None."""
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text(self._minimal_csv())
        ta = TradeAnalyzer(str(csv_file))
        result = ta.analyze_dual_agents()
        assert result["avg_runway_error"] is None
        assert result["avg_runway_error_pct"] is None

    def test_all_predicted_runway_zero(self, tmp_path):
        """Line 176: predicted_runway exists but all values <= 0 → df_valid empty → None."""
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text(
            self._minimal_csv(
                extra_columns={
                    "predicted_runway": [0.0, -1.0],
                    "mfe": [5.0, 3.0],
                }
            )
        )
        ta = TradeAnalyzer(str(csv_file))
        result = ta.analyze_dual_agents()
        assert result["avg_runway_error"] is None

    def test_no_capture_efficiency_column(self, tmp_path):
        """Line 184: No capture_efficiency → capture_by_harvester_quality = None."""
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text(self._minimal_csv())
        ta = TradeAnalyzer(str(csv_file))
        result = ta.analyze_dual_agents()
        assert result["capture_by_harvester_quality"] is None


# ── risk_manager ─────────────────────────────────────────────────────────

from src.risk.circuit_breakers import CircuitBreakerManager
from src.risk.risk_manager import RiskManager
from src.risk.var_estimator import VaREstimator


@pytest.fixture()
def _rm():
    """Build a RiskManager with warm VaR estimator."""
    est = VaREstimator(window=100, confidence=0.95)
    rng = np.random.default_rng(42)
    for ret in rng.normal(0.0, 0.01, 100):
        est.update_return(float(ret))
    cb = CircuitBreakerManager()
    return RiskManager(
        circuit_breakers=cb,
        var_estimator=est,
        risk_budget_usd=100.0,
        max_position_size=1.0,
        min_confidence_entry=0.6,
        min_confidence_exit=0.5,
    )


class TestRiskManagerGaps:
    """Cover scattered defensive/edge-case lines in RiskManager."""

    def test_validate_exit_unknown_type(self, _rm):
        """Line 463: Unknown exit_type → rejected."""
        result = _rm.validate_exit(action=1, exit_type="GARBAGE", current_position=0.5)
        assert not result.approved
        assert "Unknown exit_type" in result.reason

    def test_validate_exit_zero_volume(self, _rm):
        """Lines 477-480: volume ≤ 0 after calculation → rejected."""
        # FULL exit with current_position=0.0 but min_position_size=0.0 bypasses earlier guard
        result = _rm.validate_exit(
            action=1, exit_type="FULL", current_position=0.0, min_position_size=0.0
        )
        assert not result.approved
        assert "zero" in result.reason.lower() or "Calculated" in result.reason

    def test_update_exposure_clear_position(self, _rm):
        """Line 529: position_size=0 removes symbol from active_positions."""
        _rm.update_exposure("EURUSD", 1.0)
        assert "EURUSD" in _rm.active_positions
        _rm.update_exposure("EURUSD", 0)
        assert "EURUSD" not in _rm.active_positions

    def test_on_trade_complete_with_rl_feedback(self, _rm):
        """Line 642: With _last_decision_confidence, RL feedback fires."""
        _rm._last_decision_confidence = 0.8
        _rm._last_decision_type = "entry"
        _rm.on_trade_complete(pnl=10.0, equity=10010.0)
        assert _rm.total_trades == 1

    def test_get_risk_summary_no_assessment(self, _rm):
        """Line 854: No prior assess_risk → returns message string."""
        _rm._last_assessment = None
        summary = _rm.get_risk_summary()
        assert summary == "No risk assessment available"

    def test_get_confidence_bucket_high(self, _rm):
        """Line 957: confidence ≥ 0.95 → bucket 1.0."""
        bucket = _rm._get_confidence_bucket(0.99)
        assert bucket == pytest.approx(1.0)

    def test_check_correlation_breakdown_insufficient_data(self, _rm):
        """Line 1230: 2 symbols but only 1 has enough data → returns None."""
        _rm.returns_history = {
            "BTCUSD": list(rng.standard_normal(50)),
            "EURUSD": [0.01],  # too short
        }
        result = _rm.check_correlation_breakdown(current_time=time.time())
        assert result is None

    def test_allocate_capital_no_history(self, _rm):
        """Lines 1322-1323: correlation_matrix exists but < 2 symbols with data → equal split."""
        # Set correlation_matrix to non-None so first guard passes
        _rm.correlation_matrix = np.eye(3)
        _rm.returns_history = {"SYM_A": [0.01]}  # Only 1 symbol with data, not enough
        result = _rm.allocate_capital_by_correlation(
            symbols=["SYM_A", "SYM_B", "SYM_C"],
            total_capital=9000.0,
        )
        assert len(result) == 3
        assert all(abs(v - 3000.0) < 0.01 for v in result.values())
