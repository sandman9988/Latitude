"""
Resampler — aggregates lower TF bars into higher TFs.
M30 → H1 → H2 → H4. Bar synchronisation ensures alignment.
Symbol-agnostic.
"""
from __future__ import annotations

from typing import List, Dict
from .cleaner import Bar
from core.logger import get_logger

logger = get_logger("resampler")

# Timeframe durations in seconds
TF_SECONDS: Dict[str, int] = {
    "M1":   60,
    "M5":   300,
    "M15":  900,
    "M30":  1800,
    "H1":   3600,
    "H2":   7200,
    "H4":   14400,
    "H8":   28800,
    "D1":   86400,
    "W1":   604800,
}


def resample(bars: List[Bar], source_tf: str, target_tf: str) -> List[Bar]:
    """
    Aggregate bars from source_tf to target_tf.
    target_tf must be a multiple of source_tf.
    Returns list of aggregated bars.
    """
    src_secs = TF_SECONDS.get(source_tf.upper())
    tgt_secs = TF_SECONDS.get(target_tf.upper())

    if src_secs is None:
        raise ValueError(f"Unknown source timeframe: {source_tf}")
    if tgt_secs is None:
        raise ValueError(f"Unknown target timeframe: {target_tf}")
    if tgt_secs < src_secs:
        raise ValueError(f"Target TF {target_tf} must be >= source TF {source_tf}")
    if tgt_secs % src_secs != 0:
        raise ValueError(f"{target_tf} is not a multiple of {source_tf}")

    if not bars:
        return []

    ratio = tgt_secs // src_secs
    result: List[Bar] = []
    bucket: List[Bar] = []

    for bar in sorted(bars, key=lambda b: b.timestamp):
        bucket_start = int(bar.timestamp // tgt_secs) * tgt_secs
        if bucket and int(bucket[0].timestamp // tgt_secs) * tgt_secs != bucket_start:
            agg = _aggregate(bucket, target_tf)
            if agg:
                result.append(agg)
            bucket = []
        bucket.append(bar)

    if bucket:
        agg = _aggregate(bucket, target_tf)
        if agg:
            result.append(agg)

    logger.info(
        f"Resampled {len(bars)} {source_tf} bars → {len(result)} {target_tf} bars",
        symbol=bars[0].symbol if bars else "",
        tf=target_tf,
    )
    return result


def _aggregate(bars: List[Bar], target_tf: str) -> Bar | None:
    if not bars:
        return None
    return Bar(
        timestamp=bars[0].timestamp,
        open=bars[0].open,
        high=max(b.high for b in bars),
        low=min(b.low for b in bars),
        close=bars[-1].close,
        volume=sum(b.volume for b in bars),
        symbol=bars[0].symbol,
        timeframe=target_tf,
    )


def build_mtf(bars_m30: List[Bar]) -> Dict[str, List[Bar]]:
    """
    Build all required timeframes from M30 base data.
    Returns dict keyed by TF string.
    """
    symbol = bars_m30[0].symbol if bars_m30 else ""
    return {
        "M30": bars_m30,
        "H1":  resample(bars_m30, "M30", "H1"),
        "H2":  resample(bars_m30, "M30", "H2"),
        "H4":  resample(bars_m30, "M30", "H4"),
    }
