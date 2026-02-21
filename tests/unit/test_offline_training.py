"""
Tests for src/training — HistoricalLoader, OfflineTrainer, z_omega.
"""

from __future__ import annotations

import csv
import io
import json
import os
import textwrap
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.historical_loader import (
    _detect_columns,
    _parse_datetime,
    _sort_and_dedupe,
    bars_to_deque,
    load_csv,
    load_jsonl_cache,
    sliding_windows,
)
from src.training.offline_trainer import TrainResult, z_omega


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_csv(rows: list[dict], headers: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=headers)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _bar_rows(n: int = 20, base: float = 90000.0, step: float = 10.0) -> list[dict]:
    t0 = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
    rows = []
    for i in range(n):
        t = t0 + timedelta(minutes=i * 5)
        o = base + i * step
        rows.append({
            "Date & Time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "Open": f"{o:.2f}",
            "High": f"{o + 5:.2f}",
            "Low":  f"{o - 5:.2f}",
            "Close": f"{o + 2:.2f}",
        })
    return rows


# ── z_omega ───────────────────────────────────────────────────────────────────

class TestZOmega:

    def test_all_gains_returns_inf(self):
        """All positive returns → no losses → +inf."""
        returns = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = z_omega(returns)
        assert result == float("inf")

    def test_all_losses_returns_zero(self):
        """All negative returns → no gains → 0.0 (gains/losses = 0/positive)."""
        returns = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        result = z_omega(returns)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_mixed_positive_omega(self):
        """More gains than losses → ZOmega > 1."""
        returns = [10.0, 8.0, 6.0, -1.0, -1.0, -1.0, -1.0, 5.0]
        result = z_omega(returns)
        assert result > 1.0

    def test_mixed_negative_omega(self):
        """Large losses dominate → ZOmega < 1."""
        returns = [0.5, 0.5, 0.5, -10.0, -8.0, -6.0, -5.0]
        result = z_omega(returns)
        assert result < 1.0

    def test_too_few_returns_gives_zero(self):
        assert z_omega([1.0, 2.0, 3.0]) == 0.0

    def test_identical_returns_gives_one(self):
        """Zero std dev edge case returns neutral 1.0."""
        returns = [5.0] * 10
        assert z_omega(returns) == pytest.approx(1.0)

    def test_instrument_agnostic(self):
        """σ-normalisation makes ZOmega invariant to return scale.
        Scaling returns by a constant factor should not change ZOmega.
        """
        base = [1.0, 2.0, -0.5, 3.0, -1.0, 1.5, 2.5, -0.2, 0.8, 1.2]
        scaled = [r * 100.0 for r in base]
        assert z_omega(base) == pytest.approx(z_omega(scaled), abs=1e-4)


# ── _detect_columns ───────────────────────────────────────────────────────────

class TestDetectColumns:

    def test_ctrader_style(self):
        headers = ["Date & Time", "Open", "High", "Low", "Close"]
        col_map = _detect_columns(headers)
        assert col_map is not None
        assert col_map["o"] == "Open"
        assert col_map["c"] == "Close"

    def test_dukascopy_style(self):
        headers = ["Gmt time", "Open", "High", "Low", "Close", "Volume"]
        col_map = _detect_columns(headers)
        assert col_map is not None
        assert col_map["dt"] == "Gmt time"

    def test_mt4_date_time_split(self):
        headers = ["DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL"]
        col_map = _detect_columns(headers)
        assert col_map is not None
        assert "date" in col_map
        assert "time" in col_map

    def test_missing_close_returns_none(self):
        headers = ["Date & Time", "Open", "High", "Low"]  # no Close
        col_map = _detect_columns(headers)
        assert col_map is None

    def test_missing_datetime_returns_none(self):
        headers = ["Open", "High", "Low", "Close"]
        col_map = _detect_columns(headers)
        assert col_map is None


# ── _parse_datetime ───────────────────────────────────────────────────────────

class TestParseDatetime:

    def test_iso_format(self):
        dt = _parse_datetime("2026-01-05 10:30:00")
        assert dt is not None
        assert dt.hour == 10
        assert dt.tzinfo is not None

    def test_dukascopy_format(self):
        dt = _parse_datetime("05.01.2026 09:00:00.000")
        assert dt is not None
        assert dt.year == 2026

    def test_mt4_format(self):
        dt = _parse_datetime("2026.01.05 10:30")
        assert dt is not None
        assert dt.minute == 30

    def test_invalid_returns_none(self):
        dt = _parse_datetime("not-a-date")
        assert dt is None

    def test_aware_input_converted_to_utc(self):
        dt = _parse_datetime("2026-01-05T10:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None


# ── load_csv ──────────────────────────────────────────────────────────────────

class TestLoadCSV:

    def test_ctrader_csv(self, tmp_path):
        rows = _bar_rows(30)
        content = _make_csv(rows, ["Date & Time", "Open", "High", "Low", "Close"])
        f = tmp_path / "XAUUSD_M5.csv"
        f.write_text(content)

        bars = load_csv(str(f))
        assert len(bars) == 30
        # Each bar is (datetime, o, h, l, c, spread_pts)
        t, o, h, l, c, sp = bars[0]
        assert isinstance(t, datetime)
        assert h >= l
        assert sp == 0.0  # no spread column in this fixture

    def test_max_bars_truncation(self, tmp_path):
        rows = _bar_rows(50)
        content = _make_csv(rows, ["Date & Time", "Open", "High", "Low", "Close"])
        f = tmp_path / "test.csv"
        f.write_text(content)
        bars = load_csv(str(f), max_bars=20)
        assert len(bars) == 20

    def test_sorted_ascending(self, tmp_path):
        rows = list(reversed(_bar_rows(20)))  # Deliberately reversed
        content = _make_csv(rows, ["Date & Time", "Open", "High", "Low", "Close"])
        f = tmp_path / "test.csv"
        f.write_text(content)
        bars = load_csv(str(f))
        timestamps = [b[0] for b in bars]
        assert timestamps == sorted(timestamps)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path.csv")

    def test_malformed_ohlc_skipped(self, tmp_path):
        """Bars where high < low should be silently dropped."""
        rows = _bar_rows(10)
        rows[3]["High"] = "100.0"
        rows[3]["Low"]  = "200.0"   # Low > High → malformed
        content = _make_csv(rows, ["Date & Time", "Open", "High", "Low", "Close"])
        f = tmp_path / "test.csv"
        f.write_text(content)
        bars = load_csv(str(f))
        assert len(bars) == 9  # One dropped


# ── load_jsonl_cache ──────────────────────────────────────────────────────────

class TestLoadJSONLCache:

    def _make_cache(self, n_records: int, n_bars: int = 10) -> str:
        lines = []
        t0 = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
        for i in range(n_records):
            entry_bars = []
            for j in range(n_bars):
                t = t0 + timedelta(minutes=(i * n_bars + j) * 5)
                entry_bars.append([t.isoformat(), 90000.0 + j, 90005.0 + j,
                                   89995.0 + j, 90002.0 + j])
            lines.append(json.dumps({
                "version": 1,
                "ts_recorded": t0.isoformat(),
                "symbol": "XAUUSD",
                "timeframe_minutes": 5,
                "entry_bars": entry_bars,
                "exit_bars": entry_bars[-3:],
            }))
        return "\n".join(lines) + "\n"

    def test_basic_load(self, tmp_path):
        content = self._make_cache(5, n_bars=10)
        f = tmp_path / "training_cache.jsonl"
        f.write_text(content)
        bars = load_jsonl_cache(str(f))
        # 5 records × 10 bars, but exit_bars overlap → some deduplication expected
        assert len(bars) > 0
        assert all(isinstance(b[0], datetime) for b in bars)

    def test_corrupt_line_skipped(self, tmp_path):
        content = self._make_cache(3) + "this is not json\n"
        f = tmp_path / "training_cache.jsonl"
        f.write_text(content)
        bars = load_jsonl_cache(str(f))
        assert len(bars) > 0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl_cache("/nonexistent/cache.jsonl")


# ── sliding_windows ───────────────────────────────────────────────────────────

class TestSlidingWindows:

    def _fake_bars(self, n: int):
        t0 = datetime(2026, 1, 5, tzinfo=UTC)
        return [(t0 + timedelta(minutes=i*5), float(i), float(i)+1,
                 float(i)-1, float(i)) for i in range(n)]

    def test_window_count(self):
        bars = self._fake_bars(20)
        windows = list(sliding_windows(bars, window=10, step=1))
        assert len(windows) == 11  # 20 - 10 + 1

    def test_step_larger_than_one(self):
        bars = self._fake_bars(20)
        windows = list(sliding_windows(bars, window=10, step=2))
        assert len(windows) == 6

    def test_window_size_correct(self):
        bars = self._fake_bars(15)
        for w in sliding_windows(bars, window=8, step=3):
            assert len(w) == 8

    def test_window_exceeds_bars_yields_nothing(self):
        bars = self._fake_bars(5)
        windows = list(sliding_windows(bars, window=10))
        assert windows == []


# ── bars_to_deque ─────────────────────────────────────────────────────────────

class TestBarsToDeque:

    def test_deque_creation(self):
        t0 = datetime(2026, 1, 5, tzinfo=UTC)
        bars = [(t0 + timedelta(minutes=i*5), 100.0, 101.0, 99.0, 100.5) for i in range(20)]
        d = bars_to_deque(bars, maxlen=50)
        assert len(d) == 20
        assert isinstance(d, deque)

    def test_maxlen_respected(self):
        t0 = datetime(2026, 1, 5, tzinfo=UTC)
        bars = [(t0 + timedelta(minutes=i*5), 100.0, 101.0, 99.0, 100.5) for i in range(100)]
        d = bars_to_deque(bars, maxlen=30)
        assert d.maxlen == 30
        assert len(d) == 30


# ── BarExperienceCache ────────────────────────────────────────────────────────

class TestBarExperienceCache:

    def test_disabled_cache_writes_nothing(self, tmp_path):
        from src.training.bar_experience_cache import BarExperienceCache
        cache = BarExperienceCache(
            symbol="XAUUSD",
            cache_file=str(tmp_path / "cache.jsonl"),
            enabled=False,
        )
        cache.snapshot_entry(deque())
        cache.record_trade(
            bars=deque(),
            trigger_action=1, trigger_reward=0.1, capture_reward=0.2,
            entry_price=90000.0, exit_price=90100.0, pnl_pts=100.0,
            mfe=150.0, mae=0.0,
        )
        assert not (tmp_path / "cache.jsonl").exists()

    def test_record_written_to_jsonl(self, tmp_path):
        from src.training.bar_experience_cache import BarExperienceCache
        t0 = datetime(2026, 1, 5, tzinfo=UTC)
        bars = deque(maxlen=100)
        for i in range(30):
            bars.append((t0 + timedelta(minutes=i*5), 90000.0+i, 90005.0+i,
                         89995.0+i, 90002.0+i))

        cache_path = tmp_path / "cache.jsonl"
        cache = BarExperienceCache(
            symbol="XAUUSD",
            timeframe_minutes=5,
            cache_file=str(cache_path),
            enabled=True,
        )
        cache.snapshot_entry(bars)
        cache.record_trade(
            bars=bars,
            trigger_action=1, trigger_reward=0.05, capture_reward=0.3,
            entry_price=90000.0, exit_price=90100.0, pnl_pts=100.0,
            mfe=150.0, mae=20.0, regime="TRENDING", was_explore=False,
        )

        assert cache_path.exists()
        lines = [l for l in cache_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1

        rec = json.loads(lines[0])
        assert rec["symbol"] == "XAUUSD"
        assert rec["timeframe_minutes"] == 5
        assert rec["trigger_action"] == 1
        assert rec["mfe"] == pytest.approx(150.0)
        assert len(rec["exit_bars"]) > 0
        assert len(rec["entry_bars"]) > 0

    def test_multiple_trades_accumulate(self, tmp_path):
        from src.training.bar_experience_cache import BarExperienceCache
        cache_path = tmp_path / "cache.jsonl"
        cache = BarExperienceCache(cache_file=str(cache_path))
        for _ in range(5):
            cache.record_trade(
                bars=deque(), trigger_action=0, trigger_reward=0.0,
                capture_reward=0.0, entry_price=1.0, exit_price=1.0,
                pnl_pts=0.0, mfe=0.0, mae=0.0,
            )
        assert cache.record_count() == 5

    def test_entry_snapshot_cleared_after_record(self, tmp_path):
        from src.training.bar_experience_cache import BarExperienceCache
        cache = BarExperienceCache(cache_file=str(tmp_path / "c.jsonl"))
        bars = deque([(datetime(2026,1,5,tzinfo=UTC), 1.0, 1.1, 0.9, 1.0)])
        cache.snapshot_entry(bars)
        assert cache._entry_bars_snapshot is not None
        cache.record_trade(
            bars=bars, trigger_action=0, trigger_reward=0.0,
            capture_reward=0.0, entry_price=1.0, exit_price=1.0,
            pnl_pts=0.0, mfe=0.0, mae=0.0,
        )
        assert cache._entry_bars_snapshot is None

    def test_schema_version_in_record(self, tmp_path):
        from src.training.bar_experience_cache import BarExperienceCache, SCHEMA_VERSION
        cache_path = tmp_path / "cache.jsonl"
        cache = BarExperienceCache(cache_file=str(cache_path))
        cache.record_trade(
            bars=deque(), trigger_action=1, trigger_reward=0.1,
            capture_reward=0.2, entry_price=90000.0, exit_price=90100.0,
            pnl_pts=100.0, mfe=150.0, mae=0.0,
        )
        rec = json.loads(cache_path.read_text().strip())
        assert rec["version"] == SCHEMA_VERSION


# ── discover_jobs (from train_offline) ────────────────────────────────────────

class TestDiscoverJobs:

    def test_detect_symbol_and_tf_from_filename(self, tmp_path):
        from train_offline import discover_jobs
        f = tmp_path / "XAUUSD_M5.csv"
        f.write_text("Date & Time,Open,High,Low,Close\n")
        jobs = discover_jobs([str(tmp_path)])
        assert len(jobs) == 1
        assert jobs[0].symbol == "XAUUSD"
        assert jobs[0].timeframe_minutes == 5

    def test_symbol_filter(self, tmp_path):
        from train_offline import discover_jobs
        (tmp_path / "XAUUSD_M5.csv").write_text("Date & Time,Open,High,Low,Close\n")
        (tmp_path / "EURUSD_M5.csv").write_text("Date & Time,Open,High,Low,Close\n")
        jobs = discover_jobs([str(tmp_path)], symbol_filter=["XAUUSD"])
        assert all(j.symbol == "XAUUSD" for j in jobs)

    def test_tf_filter(self, tmp_path):
        from train_offline import discover_jobs
        (tmp_path / "XAUUSD_M5.csv").write_text("Date & Time,Open,High,Low,Close\n")
        (tmp_path / "XAUUSD_H1.csv").write_text("Date & Time,Open,High,Low,Close\n")
        jobs = discover_jobs([str(tmp_path)], tf_filter=["H1"])
        assert all(j.timeframe_minutes == 60 for j in jobs)

    def test_nonexistent_path_skipped(self, tmp_path):
        from train_offline import discover_jobs
        jobs = discover_jobs(["/nonexistent/path"])
        assert jobs == []
