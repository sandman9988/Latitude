"""
cTrader downloader — wraps the connector to produce pipeline.cleaner.Bar lists.
Supports M1 backfill + incremental, with price-scale auto-detection from Kinetra.

Usage:
    creds = CTraderCredentials.from_env()
    conn = CTraderConnector(creds)
    conn.start()
    bars = download_bars(conn, "XAUUSD", "M30", days=365)
    conn.stop()

Saves raw CSVs to data/ctrader/{symbol}/{timeframe}/ and returns Bar lists
ready for clean() + resample().
"""
from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from core.logger import get_logger
from pipeline.cleaner import Bar

logger = get_logger("downloader")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTER_REQUEST_SLEEP_S = float(os.getenv("CTRADER_INTER_REQUEST_SLEEP_S", "0.25"))
CHUNK_DAYS = 7


# ---------------------------------------------------------------------------
# Price decoding — ported from Kinetra with continuity-based scale selection
# ---------------------------------------------------------------------------

def _candidate_price_scales(digits: int, pip_position: int) -> list[float]:
    return sorted(
        {
            float(10 ** max(digits, 0)),
            float(10 ** max(pip_position + 1, 0)),
            100000.0,
        },
        reverse=True,
    )


def _decode_prices(low_raw: int, d_open: int, d_high: int, d_close: int, scale: float) -> tuple[float, float, float, float]:
    o = (low_raw + d_open) / scale
    h = (low_raw + d_high) / scale
    l = low_raw / scale
    c = (low_raw + d_close) / scale
    return o, h, l, c


def _select_price_scale(
    *,
    low_raw: int,
    d_open: int,
    d_high: int,
    d_close: int,
    digits: int,
    pip_position: int,
    prev_close: Optional[float] = None,
) -> float:
    candidates = []
    for scale in _candidate_price_scales(digits, pip_position):
        o, h, l, c = _decode_prices(low_raw, d_open, d_high, d_close, scale)
        if not (o > 0 and h > 0 and l > 0 and c > 0) or h < l:
            continue
        if prev_close is None:
            return scale
        score = min(abs(o - prev_close), abs(c - prev_close)) / max(abs(prev_close), 1e-12)
        candidates.append((score, scale))
    if candidates:
        candidates.sort(key=lambda x: (x[0], -x[1]))
        return candidates[0][1]
    return 100000.0


def _decode_trendbar(tb: object, digits: int, pip_position: int, prev_close: Optional[float] = None) -> Optional[dict]:
    low_raw = int(getattr(tb, "low", 0) or 0)
    if low_raw <= 0:
        return None
    d_open = int(getattr(tb, "deltaOpen", 0) or 0)
    d_high = int(getattr(tb, "deltaHigh", 0) or 0)
    d_close = int(getattr(tb, "deltaClose", 0) or 0)
    scale = _select_price_scale(
        low_raw=low_raw, d_open=d_open, d_high=d_high, d_close=d_close,
        digits=digits, pip_position=pip_position, prev_close=prev_close,
    )
    ts_min = int(getattr(tb, "utcTimestampInMinutes", 0) or 0)
    if ts_min > 0:
        ts = datetime.fromtimestamp(ts_min * 60, tz=timezone.utc)
    else:
        ts_ms = int(getattr(tb, "timestamp", 0) or 0)
        if ts_ms <= 0:
            return None
        ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

    o, h, l, c = _decode_prices(low_raw, d_open, d_high, d_close, scale)
    if not (o > 0 and h > 0 and l > 0 and c > 0):
        return None
    return {
        "timestamp": ts.timestamp(),
        "open": o, "high": h, "low": l, "close": c,
        "volume": float(getattr(tb, "volume", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Symbol precision
# ---------------------------------------------------------------------------

def _load_symbol_precision(conn, symbol_id: int) -> tuple[int, int]:
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
        req = api_msgs.ProtoOASymbolByIdReq()
        req.ctidTraderAccountId = conn.credentials.account_id
        req.symbolId.append(symbol_id)
        resp = conn.send_and_wait(req, timeout_s=20.0)
        if resp is None or hasattr(resp, "errorCode") or not getattr(resp, "symbol", []):
            return conn.get_digits(symbol_id), 1
        sym = resp.symbol[0]
        digits = int(getattr(sym, "digits", 0) or 0) or conn.get_digits(symbol_id)
        pip_pos = int(getattr(sym, "pipPosition", max(digits - 1, 0)) or max(digits - 1, 0))
        return digits, pip_pos
    except Exception:
        return 5, 4


# ---------------------------------------------------------------------------
# Trendbar period mapping
# ---------------------------------------------------------------------------

_TF_TO_PERIOD: Dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "H12": 720,
    "D1": 1440, "W1": 10080,
}
# Note: H2 and H8 are not native cTrader timeframes. Build them via
# pipeline.resampler.resample(bars, "M30", "H2") after downloading M30.

# cTrader ProtoOATrendbarPeriod enum values (from OpenApiModelMessages.proto)
_TF_TO_PROTO: Dict[str, int] = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M10": 6, "M15": 7,
    "M30": 8, "H1": 9, "H4": 10, "H12": 11, "D1": 12, "W1": 13, "MN1": 14,
}


# ---------------------------------------------------------------------------
# Core download
# ---------------------------------------------------------------------------

def download_bars(
    conn,
    symbol: str,
    timeframe: str = "M30",
    days: int = 365,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    output_dir: Optional[Path] = None,
) -> List[Bar]:
    """
    Download OHLCV bars from cTrader and return as List[Bar].
    Also saves a raw CSV to output_dir if provided.

    conn: connected CTraderConnector
    symbol: e.g. "XAUUSD", "DE40", "US500"
    timeframe: M1, M5, M15, M30, H1, H4, H12, D1, W1 (H2/H8 not natively supported; resample from M30)
    days: lookback if start_dt not provided
    """
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
    except ImportError:
        raise ImportError("ctrader_open_api not installed. Run: pip install ctrader-open-api")

    symbol = symbol.strip().upper()
    tf_upper = timeframe.upper()
    proto_period = _TF_TO_PROTO.get(tf_upper)
    if proto_period is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use: {list(_TF_TO_PROTO)}")

    if end_dt is None:
        end_dt = datetime.now(tz=timezone.utc)
    if start_dt is None:
        start_dt = end_dt - timedelta(days=days)

    symbol_id = conn.find_symbol_id(symbol)
    if symbol_id is None:
        raise ValueError(f"Symbol not found on broker: {symbol}")

    digits, pip_pos = _load_symbol_precision(conn, symbol_id)
    time.sleep(INTER_REQUEST_SLEEP_S)

    logger.info(
        f"Downloading {symbol} {tf_upper} from {start_dt.date()} to {end_dt.date()}",
        symbol=symbol, tf=tf_upper, component="downloader"
    )

    raw_rows = []
    cur = start_dt
    chunk_span = timedelta(days=CHUNK_DAYS)

    while cur < end_dt:
        nxt = min(cur + chunk_span, end_dt)
        req = api_msgs.ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = conn.credentials.account_id
        req.symbolId = symbol_id
        req.period = proto_period
        req.fromTimestamp = int(cur.timestamp() * 1000)
        req.toTimestamp = int(nxt.timestamp() * 1000)
        req.count = 10000

        resp = conn.send_and_wait(req, timeout_s=30.0)
        if resp and not hasattr(resp, "errorCode"):
            prev_close: Optional[float] = raw_rows[-1]["close"] if raw_rows else None
            for tb in getattr(resp, "trendbar", []):
                r = _decode_trendbar(tb, digits, pip_pos, prev_close=prev_close)
                if r:
                    raw_rows.append(r)
                    prev_close = r["close"]

        cur = nxt
        time.sleep(INTER_REQUEST_SLEEP_S)

    if not raw_rows:
        logger.warning(f"No data returned for {symbol} {tf_upper}", symbol=symbol, tf=tf_upper)
        return []

    # Sort and deduplicate
    raw_rows.sort(key=lambda r: r["timestamp"])
    seen: set[float] = set()
    deduped = []
    for r in raw_rows:
        if r["timestamp"] not in seen:
            seen.add(r["timestamp"])
            deduped.append(r)

    bars = [
        Bar(
            timestamp=r["timestamp"],
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            volume=r["volume"],
            symbol=symbol,
            timeframe=tf_upper,
        )
        for r in deduped
    ]

    logger.info(
        f"Downloaded {len(bars)} {tf_upper} bars for {symbol}",
        symbol=symbol, tf=tf_upper, component="downloader"
    )

    if output_dir:
        _save_csv(bars, symbol, tf_upper, output_dir)

    return bars


def _save_csv(bars: List[Bar], symbol: str, tf: str, output_dir: Path) -> Path:
    output_dir = output_dir / symbol / tf
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime.fromtimestamp(bars[0].timestamp, tz=timezone.utc).strftime("%Y%m%d%H%M")
    t1 = datetime.fromtimestamp(bars[-1].timestamp, tz=timezone.utc).strftime("%Y%m%d%H%M")
    path = output_dir / f"{symbol}_{tf}_{t0}_{t1}.csv"
    with open(path, "w") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for b in bars:
            dt = datetime.fromtimestamp(b.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{dt},{b.open},{b.high},{b.low},{b.close},{b.volume}\n")
    logger.info(f"Saved {len(bars)} bars to {path}", component="downloader")
    return path


def load_bars_from_csv(path: Path, symbol: str = "", tf: str = "") -> List[Bar]:
    """Load bars from a previously saved CSV."""
    bars = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                ts = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
                bars.append(Bar(
                    timestamp=ts,
                    open=float(parts[1]),
                    high=float(parts[2]),
                    low=float(parts[3]),
                    close=float(parts[4]),
                    volume=float(parts[5]),
                    symbol=symbol,
                    timeframe=tf,
                ))
            except (ValueError, IndexError):
                continue
    return bars
