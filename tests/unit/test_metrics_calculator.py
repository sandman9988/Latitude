"""Regression tests for metrics_calculator.

These tests lock in the correct capture-ratio derivation so the unit-mismatch
bugs (BTCUSD mfe stored as price points, XAUUSD stored capture_ratio 100× too
high from bad normalization script) can never silently regress.
"""

import pytest

from src.utils.metrics_calculator import period_metrics, _trade_excursion_and_capture


# ── capture ratio derivation ──────────────────────────────────────────────────

class TestCaptureRatioDerivation:
    """Capture must always come from (exit-entry)/mfe_points, not stored fields."""

    def _btcusd_trade(self, entry, exit_, mfe_pts, pnl, qty=0.01):
        return {
            "symbol": "BTCUSD",
            "direction": "LONG",
            "entry_price": entry,
            "exit_price": exit_,
            "mfe_points": mfe_pts,
            "mfe": mfe_pts,          # old records: mfe == mfe_points (price pts)
            "pnl": pnl,
            "quantity": qty,
            "capture_ratio": pnl / mfe_pts,  # the WRONG stored value (pnl_usd / mfe_pts)
        }

    def _xauusd_trade(self, entry, exit_, mfe_pts, pnl, stored_cap):
        return {
            "symbol": "XAUUSD",
            "direction": "LONG",
            "entry_price": entry,
            "exit_price": exit_,
            "mfe_points": mfe_pts,
            "pnl": pnl,
            "capture_ratio": stored_cap,   # the WRONG stored value from normalise script
            "pnl_points": pnl / 2.0,       # wrong pnl_points written by old script
        }

    def test_btcusd_ignores_stored_capture_ratio(self):
        # entry=74138, exit=78013, mfe_pts=3917 → correct cap = (78013-74138)/3917 = 98.9%
        # stored capture_ratio = pnl_usd/mfe_pts = 38.75/3917 ≈ 0.99% ← WRONG
        trade = self._btcusd_trade(74138.0, 78013.0, 3917.0, 38.75)
        m = period_metrics([trade])
        # Correct: (78013-74138)/3917 = 98.9%
        assert m["avg_capture_ratio"] == pytest.approx(3875.0 / 3917.0, abs=0.01)

    def test_xauusd_ignores_stored_357_pct_capture(self):
        # entry=4804.71, exit=4809.02, mfe_pts=6.03 → correct cap = 4.31/6.03 = 71.5%
        # stored capture_ratio = 3.57 (357%) ← WRONG
        trade = self._xauusd_trade(4804.71, 4809.02, 6.03, 43.10, stored_cap=3.573797678)
        m = period_metrics([trade])
        assert m["avg_capture_ratio"] == pytest.approx(4.31 / 6.03, abs=0.01)

    def test_correct_capture_short_trade(self):
        trade = {
            "symbol": "XAUUSD",
            "direction": "SHORT",
            "entry_price": 4809.0,
            "exit_price": 4805.0,
            "mfe_points": 5.0,
            "pnl": 40.0,
            "capture_ratio": 99.0,   # wrong stored value — must be ignored
        }
        # pnl_pts = entry - exit = 4.0, cap = 4.0/5.0 = 80%
        m = period_metrics([trade])
        assert m["avg_capture_ratio"] == pytest.approx(0.80, abs=0.01)

    def test_no_mfe_points_yields_zero_capture(self):
        # When mfe_points is absent we cannot derive capture — result is 0, not stored garbage.
        trade = {"pnl": 10.0, "mfe": 0.1, "capture_ratio": 0.5}
        m = period_metrics([trade])
        assert m["avg_capture_ratio"] == pytest.approx(0.0)


# ── MFE/MAE USD conversion ────────────────────────────────────────────────────

class TestExcursionUsd:
    """_trade_excursion_and_capture must return USD values, not price points."""

    def test_btcusd_mfe_usd(self):
        trade = {
            "direction": "LONG",
            "entry_price": 74138.0,
            "exit_price": 78013.0,
            "mfe_points": 3917.0,
            "mae_points": 0.0,
            "pnl": 38.75,   # 3875 pts × 0.01 lots
        }
        mfe_usd, _, cap = _trade_excursion_and_capture(trade)
        # lot_value = 38.75 / 3875 = 0.01; mfe_usd = 3917 × 0.01 = 39.17
        assert mfe_usd == pytest.approx(39.17, abs=0.01)
        assert cap == pytest.approx(3875.0 / 3917.0, abs=0.01)

    def test_xauusd_mfe_usd(self):
        trade = {
            "direction": "LONG",
            "entry_price": 4804.71,
            "exit_price": 4809.02,
            "mfe_points": 6.03,
            "mae_points": 0.895,
            "pnl": 43.10,
        }
        mfe_usd, _, cap = _trade_excursion_and_capture(trade)
        # lot_value = 43.10 / 4.31 = 10.0; mfe_usd = 6.03 × 10 = 60.3
        assert mfe_usd == pytest.approx(60.3, abs=0.1)
        assert cap == pytest.approx(4.31 / 6.03, abs=0.01)

    def test_zero_mfe_points_returns_zero(self):
        trade = {"direction": "LONG", "entry_price": 1.0, "exit_price": 2.0,
                 "mfe_points": 0.0, "pnl": 1.0}
        mfe_usd, _, cap = _trade_excursion_and_capture(trade)
        assert mfe_usd == pytest.approx(0.0)
        assert cap is None


# ── avg_mfe in period_metrics ─────────────────────────────────────────────────

class TestAvgMfe:
    def test_avg_mfe_is_usd_not_points(self):
        # BTCUSD 0.01 lots: mfe_pts=3917, mfe_usd=39.17
        trade = {
            "direction": "LONG",
            "entry_price": 74138.0,
            "exit_price": 78013.0,
            "mfe_points": 3917.0,
            "mae_points": 0.0,
            "pnl": 38.75,
        }
        m = period_metrics([trade])
        assert m["avg_mfe"] == pytest.approx(39.17, abs=0.01)
        # Must NOT be 3917 (price points)
        assert m["avg_mfe"] < 100.0
