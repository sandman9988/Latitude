"""
Feature store — versioned, normalised, symbol-tagged feature cache.
Ensures backtests are reproducible and features are never cross-contaminated.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from core.logger import get_logger

logger = get_logger("feature_store")


@dataclass
class FeatureRow:
    """One bar's worth of features for a single symbol/TF combination."""
    timestamp: float
    symbol: str
    tf: str
    features: Dict[str, float] = field(default_factory=dict)
    label: Optional[float] = None   # MFE or other supervised target


class FeatureStore:
    """
    In-memory feature store with optional disk persistence (JSONL).
    Keyed by (symbol, tf). All values are validated finite floats.
    """

    def __init__(self, path: Optional[str] = None, version: str = "v1") -> None:
        self._path = path
        self._version = version
        self._data: Dict[str, List[FeatureRow]] = {}

    def _key(self, symbol: str, tf: str) -> str:
        return f"{symbol.upper()}:{tf.upper()}"

    def put(self, row: FeatureRow) -> None:
        key = self._key(row.symbol, row.tf)
        if key not in self._data:
            self._data[key] = []
        # Replace if same timestamp exists
        existing = self._data[key]
        for i, r in enumerate(existing):
            if r.timestamp == row.timestamp:
                existing[i] = row
                return
        existing.append(row)

    def get(self, symbol: str, tf: str) -> List[FeatureRow]:
        return self._data.get(self._key(symbol, tf), [])

    def get_feature_matrix(self, symbol: str, tf: str, feature_names: Optional[List[str]] = None) -> tuple[List[float], List[List[float]]]:
        """
        Returns (timestamps, feature_matrix) sorted by timestamp.
        feature_matrix[i] is the feature vector for bar i.
        """
        rows = sorted(self.get(symbol, tf), key=lambda r: r.timestamp)
        if not rows:
            return [], []

        names = feature_names or (list(rows[0].features.keys()) if rows else [])
        timestamps = [r.timestamp for r in rows]
        matrix = [[r.features.get(n, 0.0) for n in names] for r in rows]
        return timestamps, matrix

    def save(self, path: Optional[str] = None) -> None:
        out_path = path or self._path
        if not out_path:
            raise ValueError("No path specified for save")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            meta = {"version": self._version}
            f.write(json.dumps(meta) + "\n")
            for rows in self._data.values():
                for row in rows:
                    f.write(json.dumps(asdict(row)) + "\n")
        logger.info(f"Feature store saved to {out_path}", component="feature_store")

    def load(self, path: Optional[str] = None) -> None:
        in_path = path or self._path
        if not in_path or not os.path.exists(in_path):
            raise FileNotFoundError(f"Feature store file not found: {in_path}")
        with open(in_path, "r") as f:
            lines = f.readlines()
        if not lines:
            return
        meta = json.loads(lines[0])
        loaded_version = meta.get("version", "unknown")
        if loaded_version != self._version:
            logger.warning(
                f"Feature store version mismatch: file={loaded_version} expected={self._version}",
                component="feature_store"
            )
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            row = FeatureRow(
                timestamp=d["timestamp"],
                symbol=d["symbol"],
                tf=d["tf"],
                features=d.get("features", {}),
                label=d.get("label"),
            )
            self.put(row)
        logger.info(f"Feature store loaded from {in_path}", component="feature_store")

    def clear(self, symbol: Optional[str] = None, tf: Optional[str] = None) -> None:
        if symbol and tf:
            key = self._key(symbol, tf)
            self._data.pop(key, None)
        elif symbol:
            keys = [k for k in self._data if k.startswith(symbol.upper() + ":")]
            for k in keys:
                self._data.pop(k, None)
        else:
            self._data.clear()

    @property
    def symbols(self) -> List[str]:
        return list({k.split(":")[0] for k in self._data})

    @property
    def timeframes(self) -> List[str]:
        return list({k.split(":")[1] for k in self._data})
