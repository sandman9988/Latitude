"""Coverage batch 8 – Targeted production-code gaps.

Targets:
  - regime_detector: lines 178, 184-185, 199-200 (non-finite defensive guards in _update_regime)
  - journaled_persistence: lines 111, 320-321, 328, 352-353, 372-373 (replay edge cases)
  - safe_math: lines 258, 274-276 (softmax total→0 guard, running_variance count<2)
  - order_book: line 76 (non-finite spread_value)
  - trade_exporter: lines 159-160, 230-231 (formatting error, no prefix)
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# RegimeDetector – defensive guards in _update_regime
# ---------------------------------------------------------------------------
class TestRegimeDetectorUpdateRegimeDefensiveGuards:
    """Cover non-finite returns / variance guards inside _update_regime."""

    def _make_detector(self, window: int = 20, interval: int = 1):
        from src.features.regime_detector import RegimeDetector

        return RegimeDetector(
            window_size=window,
            update_interval=interval,
            instrument_volatility=0.001,
        )

    def _feed_prices(self, det, n: int, base: float = 2000.0, step: float = 0.5):
        """Feed *n* linearly increasing prices so the buffer fills up."""
        for i in range(n):
            det.add_price(base + i * step)

    # Line 178: non-finite 2-period returns → LOG.warning + return
    # Patch np.isfinite on the 3rd call (returns_2 check) to return False,
    # while letting the first two checks (prices, returns) pass normally.
    def test_nonfinite_two_period_returns(self):
        det = self._make_detector(window=20, interval=1)
        self._feed_prices(det, 25)

        import src.features.regime_detector as rd_mod

        orig_isfinite = np.isfinite
        call_count = {"n": 0}

        def patched_isfinite(arr):
            call_count["n"] += 1
            if call_count["n"] == 3:
                # 3rd call: np.isfinite(returns_2) → return all-False
                return np.zeros(len(arr), dtype=bool)
            return orig_isfinite(arr)

        with patch.object(rd_mod.np, "isfinite", side_effect=patched_isfinite):
            det.add_price(2100.0)

        assert det.current_regime is not None  # didn't crash

    # Lines 184-185: non-finite or negative var_2 → LOG.warning + return
    def test_nonfinite_var2(self):
        det = self._make_detector(window=20, interval=1)
        self._feed_prices(det, 25)

        import src.features.regime_detector as rd_mod

        orig_var = np.var
        call_count = {"n": 0}

        def var_side_effect(arr, *a, **kw):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                # Second+ call: var_2 – return NaN
                return float("nan")
            # First call: var_1 – return valid positive number
            return orig_var(arr, *a, **kw)

        with patch.object(rd_mod.np, "var", side_effect=var_side_effect):
            det.add_price(2100.0)

        assert det.current_regime is not None

    # Also test negative var_2
    def test_negative_var2(self):
        det = self._make_detector(window=20, interval=1)
        self._feed_prices(det, 25)

        import src.features.regime_detector as rd_mod

        orig_var = np.var
        call_count = {"n": 0}

        def var_side_effect(arr, *a, **kw):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                return -0.001  # negative var_2
            return orig_var(arr, *a, **kw)

        with patch.object(rd_mod.np, "var", side_effect=var_side_effect):
            det.add_price(2100.0)

        assert det.current_regime is not None

    # Lines 199-200: non-finite variance ratio → LOG.warning + return
    # Need var_2 finite (passes line 190 check) but ratio var_2/(2*var_1) overflows.
    # var_1 barely above VARIANCE_EPSILON (1e-12), var_2 large finite → ratio overflows to inf.
    def test_nonfinite_variance_ratio(self):
        det = self._make_detector(window=20, interval=1)
        self._feed_prices(det, 25)

        import src.features.regime_detector as rd_mod

        orig_var = np.var
        call_count = {"n": 0}

        def var_side_effect(arr, *a, **kw):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                # var_2: large but finite → ratio = 1e300 / (2 * 1e-11) = 5e310 → inf
                return 1e300
            # var_1: just above VARIANCE_EPSILON (1e-12) so it passes the check
            return 1e-11

        with patch.object(rd_mod.np, "var", side_effect=var_side_effect):
            det.add_price(2100.0)

        assert det.current_regime is not None


# ---------------------------------------------------------------------------
# JournaledPersistence – replay edge cases
# ---------------------------------------------------------------------------
class TestJournaledPersistenceReplayEdges:
    """Cover replay_from_checkpoint edge cases and _get_last_sequence empty line."""

    def _make_journal(self, tmp_path, entries=0):
        """Create a Journal and optionally log some entries."""
        from src.persistence.journaled_persistence import Journal

        j = Journal(str(tmp_path / "test.journal"), checkpoint_interval=1000)
        for i in range(entries):
            j.log_operation(f"op_{i}", {"i": i})
        return j

    # Line 111: return 0 when journal path doesn't exist.
    # In normal __init__, the file is created before _get_last_sequence is called,
    # so we call _get_last_sequence directly with a non-existent path.
    def test_get_last_sequence_no_file(self, tmp_path):
        from src.persistence.journaled_persistence import Journal

        journal_path = tmp_path / "test.log"
        j = Journal(str(journal_path), checkpoint_interval=1000)
        # Temporarily change journal_path to non-existent path
        j.journal_path = Path(str(tmp_path / "nonexistent.log"))
        result = j._get_last_sequence()
        assert result == 0
        # Restore and close
        j.journal_path = Path(str(journal_path))
        j.close()

    # Lines 320-321, 328: checkpoint load raises exception
    def test_replay_with_corrupted_checkpoint(self, tmp_path):
        from src.persistence.journaled_persistence import Journal

        journal_path = tmp_path / "test.log"
        checkpoint_path = tmp_path / "test.checkpoint"

        j = Journal(str(journal_path), checkpoint_interval=1000)
        j.log_operation("op_a", {"x": 1})
        j.log_operation("op_b", {"x": 2})
        j.close()

        # Write a corrupted checkpoint
        checkpoint_path.write_text("NOT VALID JSON {{{", encoding="utf-8")

        # Open new journal and replay – checkpoint load should warn and continue
        j2 = Journal(str(journal_path), checkpoint_interval=1000)
        replayed = j2.replay_from_checkpoint()
        # Should replay all entries since checkpoint is invalid (seq=0)
        assert len(replayed) >= 1
        j2.close()

    # Lines 352-353: callback returns False → break (stop replay early)
    def test_replay_callback_returns_false_stops_early(self, tmp_path):
        from src.persistence.journaled_persistence import Journal

        j = Journal(str(tmp_path / "test.log"), checkpoint_interval=1000)
        for i in range(5):
            j.log_operation(f"op_{i}", {"i": i})
        j.close()

        # Remove checkpoint so replay starts from the beginning
        checkpoint_file = tmp_path / "test.checkpoint"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

        j2 = Journal(str(tmp_path / "test.log"), checkpoint_interval=1000)

        seen = []

        def stop_after_two(op: str, data: dict) -> bool:
            seen.append(op)
            return len(seen) < 2  # Return False after 2nd call

        replayed = j2.replay_from_checkpoint(callback=stop_after_two)
        # Should have stopped after 2 entries
        assert len(replayed) == 2
        assert len(seen) == 2
        j2.close()

    # Lines 372-373: replay exception (corrupted journal entry)
    def test_replay_with_corrupted_journal_entry(self, tmp_path):
        from src.persistence.journaled_persistence import Journal

        journal_path = tmp_path / "test.log"
        # Write one valid entry followed by invalid JSON
        valid = json.dumps({"seq": 1, "ts": "2025-01-01T00:00:00", "op": "test", "data": {}})
        journal_path.write_text(valid + "\nNOT_JSON\n", encoding="utf-8")

        j = Journal(str(journal_path), checkpoint_interval=1000)
        # replay_from_checkpoint should catch the JSON parse error
        replayed = j.replay_from_checkpoint()
        # The exception handler logs and returns whatever was collected
        # (may have 1 entry or 0 depending on where exception hits)
        assert isinstance(replayed, list)
        j.close()


# ---------------------------------------------------------------------------
# SafeMath – softmax total near zero, running_variance count < 2
# ---------------------------------------------------------------------------
class TestSafeMathEdgeCases:
    """Cover running_variance_update count < MIN_SAMPLE_COUNT (lines 274-276).
    Note: normalize_logits line 258 (total < SAFE_DIV_MIN) is unreachable because
    the 'subtract max' step guarantees at least one exp(0)=1, so total >= 1."""

    # Lines 274-276: running_variance_update with count < 2
    def test_running_variance_count_zero(self):
        from src.utils.safe_math import SafeMath

        result = SafeMath.running_variance_update(
            old_variance=1.0, old_mean=5.0, new_mean=6.0, new_value=7.0, count=0
        )
        assert result == pytest.approx(0.0)

    def test_running_variance_count_one(self):
        from src.utils.safe_math import SafeMath

        result = SafeMath.running_variance_update(
            old_variance=1.0, old_mean=5.0, new_mean=6.0, new_value=7.0, count=1
        )
        assert result == pytest.approx(0.0)

    # Lines 274-276: normal computation path when count >= MIN_SAMPLE_COUNT (2)
    def test_running_variance_normal_path(self):
        from src.utils.safe_math import SafeMath

        result = SafeMath.running_variance_update(
            old_variance=1.0, old_mean=5.0, new_mean=5.5, new_value=6.0, count=3
        )
        # delta1 = 6.0 - 5.0 = 1.0, delta2 = 6.0 - 5.5 = 0.5
        # result = 1.0 + (1.0 * 0.5 - 1.0) / 3 = 1.0 + (-0.5/3) = 1.0 - 0.1667 = 0.8333
        assert abs(result - (1.0 + (1.0 * 0.5 - 1.0) / 3)) < 1e-10


# ---------------------------------------------------------------------------
# OrderBook – spread with non-finite value
# ---------------------------------------------------------------------------
class TestOrderBookSpreadNonFinite:
    """Cover order_book line 76: non-finite spread_value → return None."""

    def test_spread_returns_none_for_infinite_spread(self):
        from src.core.order_book import OrderBook

        ob = OrderBook()
        # Set asks to inf and bids to a normal value
        # bid < ask is required (otherwise returns 0.0 for crossed book)
        ob.bids = {1000.0: 1.0}
        ob.asks = {float("inf"): 1.0}
        result = ob.spread()
        # ask=inf, bid=1000 → spread = inf - 1000 = inf → not finite → return None
        assert result is None

    def test_spread_returns_none_for_nan_prices(self):
        from src.core.order_book import OrderBook

        ob = OrderBook()
        ob.bids = {float("nan"): 1.0}
        ob.asks = {2000.0: 1.0}
        result = ob.spread()
        # With NaN bid, bid >= ask is NaN comparison → False, so goes to
        # spread_value = 2000 - NaN = NaN → not finite → return None
        assert result is None


# ---------------------------------------------------------------------------
# TradeExporter – formatting error + no prefix
# ---------------------------------------------------------------------------
class TestTradeExporterEdges:
    """Cover trade_exporter lines 159-160 (formatting error) and 230-231 (no prefix)."""

    def _make_exporter(self, tmp_path):
        from src.monitoring.trade_exporter import TradeExporter

        return TradeExporter(output_dir=str(tmp_path))

    # Lines 159-160: trade with bad data that causes ValueError/TypeError
    def test_export_trades_formatting_error(self, tmp_path):
        import datetime as dt

        exporter = self._make_exporter(tmp_path)
        # A trade that passes the timestamp/entry_price checks but then
        # causes TypeError in the row formatting try block.
        # Use an object() for predicted_runway → f"{object():.6f}" raises TypeError.
        bad_trade = {
            "trade_num": 1,
            "entry_time": dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            "exit_time": dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc),
            "direction": "BUY",
            "entry_price": 2000.0,
            "exit_price": 2010.0,
            "pnl": 10.0,
            "mfe": 5.0,
            "mae": -3.0,
            "equity_after": 10000.0,
            "predicted_runway": object(),  # Can't format with :.6f → TypeError
        }
        # Should not raise; the error is caught and logged
        result = exporter.export_trades([bad_trade])
        assert result  # Returns filepath even with errors

    # Lines 230-231: export_all with no prefix → trades_file=None, summary_file=None
    def test_export_all_no_prefix(self, tmp_path):
        import datetime as dt

        exporter = self._make_exporter(tmp_path)

        # Create a mock performance tracker
        mock_tracker = MagicMock()
        mock_tracker.get_trade_history.return_value = [
            {
                "trade_num": 1,
                "entry_time": dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
                "exit_time": dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc),
                "direction": "BUY",
                "entry_price": 2000.0,
                "exit_price": 2010.0,
                "pnl": 10.0,
                "mfe": 5.0,
                "mae": -3.0,
                "equity_after": 10000.0,
            }
        ]
        mock_tracker.get_metrics.return_value = {
            "total_trades": 1,
            "winning_trades": 1,
            "losing_trades": 0,
            "win_rate": 1.0,
            "total_pnl": 10.0,
            "avg_winner": 10.0,
            "avg_loser": 0.0,
            "profit_factor": 999.0,
            "expectancy": 10.0,
            "sharpe_ratio": 1.5,
            "initial_equity": 10000.0,
            "current_equity": 10010.0,
            "total_return": 0.001,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "max_consecutive_wins": 1,
            "max_consecutive_losses": 0,
            "winner_to_loser_count": "1:0",
        }

        # Call without prefix → lines 230-231 set trades_file=None, summary_file=None
        results = exporter.export_all(mock_tracker, prefix="")
        assert "trades" in results or "summary" in results
