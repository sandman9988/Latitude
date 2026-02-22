#!/usr/bin/env python3
"""
HistoricalLoader
================
Load OHLC bars from disk into a list of (datetime, o, h, l, c) tuples
compatible with DualPolicy._build_state().

Supported file formats
----------------------
1. Standard CSV  — auto-detected by column headers
      Dukascopy:  Gmt time,Open,High,Low,Close,Volume
      cTrader:    Date & Time,Open,High,Low,Close
      MT4/MT5:    DATE,TIME,OPEN,HIGH,LOW,CLOSE,TICKVOL,...
      Generic:    any csv with recognisable o/h/l/c columns

2. training_cache.jsonl  — the BarExperienceCache format written by the live bot.
      Each record contains "entry_bars" and "exit_bars" lists; this loader
      reconstructs a flat bar sequence from those records so the same
      offline trainer can also replay live-captured experiences.

All timestamps are converted to UTC-aware datetime objects.
Bars are sorted ascending by time. Duplicates are dropped (last wins).
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import deque
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

Bar = tuple[datetime, float, float, float, float, float]   # t, o, h, l, c, spread_pts

# ── Recognised column-name patterns ─────────────────────────────────────────

_DATE_COLS   = re.compile(r"(date|time|gmt|timestamp)", re.IGNORECASE)
_TIME_COLS   = re.compile(r"^time$", re.IGNORECASE)
_OPEN_COLS   = re.compile(r"^open$", re.IGNORECASE)
_HIGH_COLS   = re.compile(r"^high$", re.IGNORECASE)
_LOW_COLS    = re.compile(r"^low$", re.IGNORECASE)
_CLOSE_COLS  = re.compile(r"^close$|^last$", re.IGNORECASE)
_SPREAD_COLS = re.compile(r"^spread$", re.IGNORECASE)

# Datetime format strings to try, most specific first
_DT_FORMATS = [
    "%d.%m.%Y %H:%M:%S.%f",   # Dukascopy with ms
    "%d.%m.%Y %H:%M:%S",      # Dukascopy
    "%Y.%m.%d %H:%M",         # MT4
    "%Y.%m.%d %H:%M:%S",      # MT4 with seconds
    "%Y-%m-%d %H:%M:%S",      # ISO-like (no tz)
    "%Y-%m-%d %H:%M:%S%z",    # ISO-like with tz offset (e.g. cTrader: +00:00)
    "%Y-%m-%dT%H:%M:%S",      # ISO 8601
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%m/%d/%Y %H:%M",         # US format
    "%d/%m/%Y %H:%M",         # EU format
]


# ── Public API ────────────────────────────────────────────────────────────────

def load_csv(
    path: str | Path,
    max_bars: int | None = None,
    timeframe_minutes: int | None = None,
) -> list[Bar]:
    """
    Load bars from a CSV file.

    Args:
        path:               Path to CSV file.
        max_bars:           If set, return only the most recent *max_bars* rows.
        timeframe_minutes:  When provided, bars that don't align to the TF
                            boundary are dropped (useful for mixed-TF exports).

    Returns:
        Sorted list of (datetime, o, h, l, c) bars, newest last.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    raw = list(_iter_csv(path))
    bars = _sort_and_dedupe(raw)
    if timeframe_minutes:
        bars = _filter_to_timeframe(bars, timeframe_minutes)
    if max_bars and len(bars) > max_bars:
        bars = bars[-max_bars:]

    LOG.info("[LOADER] %s → %d bars loaded", path.name, len(bars))
    return bars


def load_jsonl_cache(
    path: str | Path,
    max_bars: int | None = None,
) -> list[Bar]:
    """
    Reconstruct a flat bar sequence from a training_cache.jsonl file.

    The loader merges entry_bars and exit_bars from each record into one
    de-duplicated, sorted sequence.  This lets the offline trainer replay
    the same market context the live bot actually saw.

    Args:
        path:       Path to training_cache.jsonl.
        max_bars:   If set, return only the most recent *max_bars* rows.

    Returns:
        Sorted list of (datetime, o, h, l, c) bars.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL cache not found: {path}")

    seen: dict[datetime, Bar] = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                for bar_row in rec.get("entry_bars", []) + rec.get("exit_bars", []):
                    bar = _parse_jsonl_bar(bar_row)
                    if bar:
                        seen[bar[0]] = bar
            except (json.JSONDecodeError, Exception) as exc:
                LOG.debug("[LOADER] Line %d skipped: %s", lineno, exc)

    bars = sorted(seen.values(), key=lambda b: b[0])
    if max_bars and len(bars) > max_bars:
        bars = bars[-max_bars:]

    LOG.info("[LOADER] %s → %d unique bars reconstructed from JSONL", path.name, len(bars))
    return bars


def bars_to_deque(bars: list[Bar], maxlen: int = 500) -> deque:
    """Wrap a list of bars in a deque suitable for DualPolicy._build_state()."""
    d: deque = deque(maxlen=maxlen)
    d.extend(bars)
    return d


def sliding_windows(
    bars: list[Bar],
    window: int,
    step: int = 1,
) -> Iterator[list[Bar]]:
    """
    Yield overlapping windows of *window* bars, advancing by *step*.

    The offline trainer calls this to iterate through history as if the bot
    were running bar-by-bar, without rebuilding the full deque on every step.
    """
    n = len(bars)
    for start in range(0, n - window + 1, step):
        yield bars[start : start + window]


# ── CSV parsing ───────────────────────────────────────────────────────────────

def _iter_csv(path: Path) -> Iterator[Bar]:
    """Yield bars from a CSV file, auto-detecting column layout."""
    with open(path, encoding="utf-8", newline="") as fh:
        # Peek at first non-empty, non-comment line to detect delimiter
        sample = ""
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                sample = stripped
                break
        fh.seek(0)

        delimiter = "," if "," in sample else "\t" if "\t" in sample else ";"
        reader = csv.DictReader(fh, delimiter=delimiter)

        if reader.fieldnames is None:
            return

        col_map = _detect_columns(list(reader.fieldnames))
        if not col_map:
            raise ValueError(f"Could not detect OHLC columns in {path.name}. "
                             f"Headers: {reader.fieldnames}")

        for row in reader:
            bar = _parse_csv_row(row, col_map)
            if bar:
                yield bar


def _detect_datetime_columns(headers: list[str], norm: dict[str, str]) -> dict[str, str]:
    """
    Detect which header(s) carry datetime information.

    Handles three layouts:
    * Combined column   → ``result["dt"]``
    * Separate DATE+TIME → ``result["date"]`` + ``result["time"]``
    * Falls back to an empty dict when no datetime column is found.
    """
    result: dict[str, str] = {}
    for h in headers:
        if re.search(r"(date.*time|gmt.*time|timestamp)", h.strip("<> \t"), re.IGNORECASE):
            result["dt"] = h
            break
    if "dt" in result:
        return result

    date_h = next((norm[k] for k in norm if re.match(r"^date$", k, re.IGNORECASE)), None)
    time_h = next((norm[k] for k in norm if re.match(r"^time$", k, re.IGNORECASE)), None)
    if date_h and time_h:
        result["date"] = date_h
        result["time"] = time_h
    elif date_h:
        result["dt"] = date_h  # single "date" column with full datetime (yfinance)
    elif time_h:
        result["dt"] = time_h  # single "time" column with full datetime (cTrader CSV)
    return result


def _detect_columns(headers: list[str]) -> dict[str, str] | None:
    """Map logical field names to actual CSV column headers."""
    result: dict[str, str] = {}

    # Normalise MT4/MT5 angle-bracket headers: <DATE> → DATE, <OPEN> → OPEN …
    # We map normalised_name → original_name so lookups still use the real key.
    norm: dict[str, str] = {h.strip("<> \t"): h for h in headers}

    def _find(pattern: re.Pattern) -> str | None:
        """Return the original header for the first normalised key that matches."""
        return next((norm[k] for k in norm if pattern.match(k)), None)

    # Datetime: Dukascopy uses "Gmt time"; cTrader "Date & Time"; MT4 "DATE"
    result.update(_detect_datetime_columns(headers, norm))
    if "dt" not in result and "date" not in result:
        return None

    # OHLC columns (also strip brackets from normalised keys)
    for name, pattern in [("o", _OPEN_COLS), ("h", _HIGH_COLS),
                           ("l", _LOW_COLS),  ("c", _CLOSE_COLS)]:
        match = _find(pattern)
        if match:
            result[name] = match

    if not all(k in result for k in ("o", "h", "l", "c")):
        return None

    # Spread column is optional — present in cTrader / MT4 exports (broker points)
    sp_match = _find(_SPREAD_COLS)
    if sp_match:
        result["sp"] = sp_match

    return result


def _parse_csv_row(row: dict[str, str], col_map: dict[str, str]) -> Bar | None:
    try:
        if "dt" in col_map:
            dt_str = row[col_map["dt"]].strip()
        else:
            dt_str = row[col_map["date"]].strip() + " " + row[col_map["time"]].strip()

        ts = _parse_datetime(dt_str)
        if ts is None:
            return None

        o = float(row[col_map["o"]])
        h = float(row[col_map["h"]])
        lo = float(row[col_map["l"]])
        c = float(row[col_map["c"]])

        if not all(np.isfinite(v) for v in (o, h, lo, c)):
            return None
        if h < lo or h < o or h < c or lo > o or lo > c:
            return None  # Malformed OHLC

        sp = float(row[col_map["sp"]]) if "sp" in col_map else 0.0
        return (ts, o, h, lo, c, sp)
    except (KeyError, ValueError, TypeError):
        return None


def _parse_datetime(s: str) -> datetime | None:
    s = s.strip()
    # Fast path: try fromisoformat first (handles +00:00 / Z suffixes natively on Py3.11+)
    try:
        dt = datetime.fromisoformat(s)
        dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
        return dt
    except ValueError:
        pass
    for fmt in _DT_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
            return dt
        except ValueError:
            continue
    LOG.debug("[LOADER] Could not parse datetime: %r", s)
    return None


# ── JSONL parsing ─────────────────────────────────────────────────────────────

def _parse_jsonl_bar(row: list) -> Bar | None:
    """Parse a bar from the BarExperienceCache format: [iso_str, o, h, l, c]."""
    try:
        ts_str, o, h, lo, c = row[0], row[1], row[2], row[3], row[4]
        ts = _parse_datetime(str(ts_str))
        if ts is None:
            return None
        return (ts, float(o), float(h), float(lo), float(c), 0.0)
    except (IndexError, ValueError, TypeError):
        return None


# ── Utilities ─────────────────────────────────────────────────────────────────

def _sort_and_dedupe(bars: list[Bar]) -> list[Bar]:
    """Sort bars by timestamp, drop duplicates (keep last occurrence)."""
    by_ts: dict[datetime, Bar] = {}
    for b in bars:
        by_ts[b[0]] = b
    return sorted(by_ts.values(), key=lambda b: b[0])


def _filter_to_timeframe(bars: list[Bar], timeframe_minutes: int) -> list[Bar]:
    """Drop bars whose spacing doesn't match *timeframe_minutes*.

    Rather than testing epoch-alignment (which breaks for broker sessions that
    open at an offset, e.g. cTrader gold H4 starting at 02:00/06:00/...), we
    infer the actual bar-interval from the first pair of bars and only drop bars
    whose gap to the previous bar differs significantly from the expected step.
    """
    if len(bars) < 2:  # noqa: PLR2004 — need at least a pair to infer step
        return bars

    expected_secs = timeframe_minutes * 60
    # Detect actual session step from the first gap
    first_gap = (bars[1][0] - bars[0][0]).total_seconds()
    if abs(first_gap - expected_secs) > expected_secs * 0.1:
        # Actual interval doesn't match requested TF — conservative epoch-align fallback
        return [b for b in bars if int(b[0].timestamp()) % expected_secs == 0]

    # Interval matches: all bars belong to this TF, no filtering needed
    return bars
