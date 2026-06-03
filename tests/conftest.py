"""Shared pytest fixtures providing real market data from live JSONL caches.

All price-based tests should use these fixtures instead of synthetic bar sequences.
Data sourced from data/training_cache_XAUUSD_M5.jsonl (real paper-trading captures).
"""

from __future__ import annotations

import json
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _rocm_gfx_override():
    """Pin ROCm to gfx1100 ISA for the RX 7600 (gfx1102 has no TensileLibrary).

    Must be set before any torch import so every xdist worker and every
    ProcessPoolExecutor subprocess spawned during training tests inherits it.
    """
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

_CACHE_DIR = Path(__file__).parent.parent / "data"
_XAUUSD_M5_CACHE = _CACHE_DIR / "training_cache_XAUUSD_M5.jsonl"
_XAUUSD_M1_CACHE = _CACHE_DIR / "training_cache_XAUUSD_M1.jsonl"
_BTCUSD_M1_CACHE  = _CACHE_DIR / "training_cache_BTCUSD_M1.jsonl"


def _load_bars_from_jsonl(path: Path) -> list[tuple]:
    """Extract unique OHLCV bars from a BarExperienceCache JSONL file.

    Returns a list of (datetime, open, high, low, close) tuples sorted by time.
    """
    seen: dict[str, tuple] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for raw in rec.get("exit_bars", []) + rec.get("entry_bars", []):
            if not raw or len(raw) < 5:
                continue
            key = raw[0]
            if key in seen:
                continue
            try:
                ts_str = str(raw[0]).replace("Z", "+00:00")
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                seen[key] = (ts, float(raw[1]), float(raw[2]), float(raw[3]), float(raw[4]))
            except (ValueError, TypeError, IndexError):
                continue
    return sorted(seen.values(), key=lambda b: b[0])


def _load_trades_from_jsonl(path: Path) -> list[dict]:
    """Return the raw trade records from a BarExperienceCache JSONL file."""
    trades = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            trades.append(rec)
        except json.JSONDecodeError:
            continue
    return trades


# ---------------------------------------------------------------------------
# Module-level cached loads (parse once per session)
# ---------------------------------------------------------------------------

_xauusd_m5_bars_cache: list[tuple] | None = None
_xauusd_m1_bars_cache: list[tuple] | None = None
_btcusd_m1_bars_cache: list[tuple] | None = None
_xauusd_m5_trades_cache: list[dict] | None = None


def _get_xauusd_m5_bars() -> list[tuple]:
    global _xauusd_m5_bars_cache
    if _xauusd_m5_bars_cache is None:
        _xauusd_m5_bars_cache = _load_bars_from_jsonl(_XAUUSD_M5_CACHE)
    return _xauusd_m5_bars_cache


def _get_xauusd_m1_bars() -> list[tuple]:
    global _xauusd_m1_bars_cache
    if _xauusd_m1_bars_cache is None:
        _xauusd_m1_bars_cache = _load_bars_from_jsonl(_XAUUSD_M1_CACHE)
    return _xauusd_m1_bars_cache


def _get_btcusd_m1_bars() -> list[tuple]:
    global _btcusd_m1_bars_cache
    if _btcusd_m1_bars_cache is None:
        _btcusd_m1_bars_cache = _load_bars_from_jsonl(_BTCUSD_M1_CACHE)
    return _btcusd_m1_bars_cache


def _get_xauusd_m5_trades() -> list[dict]:
    global _xauusd_m5_trades_cache
    if _xauusd_m5_trades_cache is None:
        _xauusd_m5_trades_cache = _load_trades_from_jsonl(_XAUUSD_M5_CACHE)
    return _xauusd_m5_trades_cache


# ---------------------------------------------------------------------------
# Public fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def xauusd_m5_bars() -> list[tuple]:
    """All unique XAUUSD M5 bars from the live cache, sorted by time.

    Each bar is (datetime, open, high, low, close).
    Provides ~8 000+ real bars spanning Feb–Apr 2026.
    """
    bars = _get_xauusd_m5_bars()
    assert len(bars) >= 200, f"Expected >=200 real bars, got {len(bars)} — re-run the paper bot to accumulate more"
    return bars


@pytest.fixture(scope="session")
def xauusd_m5_bars_deque(xauusd_m5_bars) -> deque:
    """Real XAUUSD M5 bars as a deque(maxlen=2000), ready for DualPolicy._build_state()."""
    d = deque(maxlen=2000)
    d.extend(xauusd_m5_bars)
    return d


@pytest.fixture(scope="session")
def xauusd_m1_bars() -> list[tuple]:
    """All unique XAUUSD M1 bars from the live cache, sorted by time."""
    return _get_xauusd_m1_bars()


@pytest.fixture(scope="session")
def btcusd_m1_bars() -> list[tuple]:
    """All unique BTCUSD M1 bars from the live cache, sorted by time."""
    return _get_btcusd_m1_bars()


@pytest.fixture(scope="session")
def xauusd_m5_trades() -> list[dict]:
    """Raw trade records from the XAUUSD M5 live cache (BarExperienceCache schema)."""
    trades = _get_xauusd_m5_trades()
    assert len(trades) >= 10, "Need at least 10 real trades"
    return trades


@pytest.fixture(scope="session")
def xauusd_m5_cache_file() -> Path:
    """Path to the real XAUUSD M5 JSONL cache file."""
    assert _XAUUSD_M5_CACHE.exists(), f"Missing real data file: {_XAUUSD_M5_CACHE}"
    return _XAUUSD_M5_CACHE


@pytest.fixture(scope="session")
def xauusd_m1_cache_file() -> Path:
    """Path to the real XAUUSD M1 JSONL cache file."""
    assert _XAUUSD_M1_CACHE.exists(), f"Missing real data file: {_XAUUSD_M1_CACHE}"
    return _XAUUSD_M1_CACHE


@pytest.fixture
def xauusd_m5_bars_100(xauusd_m5_bars) -> deque:
    """Last 100 XAUUSD M5 bars as a deque — sufficient for warmup checks."""
    d = deque(maxlen=200)
    d.extend(xauusd_m5_bars[-100:])
    return d


@pytest.fixture
def xauusd_m5_bars_500(xauusd_m5_bars) -> deque:
    """Last 500 XAUUSD M5 bars as a deque — sufficient for VaR/kurtosis estimation."""
    d = deque(maxlen=1000)
    d.extend(xauusd_m5_bars[-500:])
    return d
