"""
Centralised structured logger.
Every event tagged with symbol, timeframe, component, trade_id.
JSON output for easy ingestion into any reporting tool.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional, Any


_LOGGERS: dict[str, "LatitudeLogger"] = {}


class LatitudeLogger:
    """
    Structured logger — wraps stdlib logging with JSON formatting.
    Tags every message with context fields.
    """

    def __init__(self, name: str, level: int = logging.INFO) -> None:
        self._name = name
        self._logger = logging.getLogger(f"latitude.{name}")
        self._logger.setLevel(level)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(_JsonFormatter())
            self._logger.addHandler(handler)
            self._logger.propagate = False

    def _emit(
        self,
        level: str,
        msg: str,
        symbol: Optional[str] = None,
        tf: Optional[str] = None,
        component: Optional[str] = None,
        trade_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "component": component or self._name,
            "msg": msg,
        }
        if symbol:
            record["symbol"] = symbol
        if tf:
            record["tf"] = tf
        if trade_id:
            record["trade_id"] = trade_id
        if kwargs:
            record["data"] = kwargs

        log_fn = {
            "DEBUG": self._logger.debug,
            "INFO": self._logger.info,
            "WARNING": self._logger.warning,
            "ERROR": self._logger.error,
            "CRITICAL": self._logger.critical,
        }.get(level, self._logger.info)

        log_fn(json.dumps(record))

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._emit("DEBUG", msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._emit("INFO", msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._emit("WARNING", msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._emit("ERROR", msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._emit("CRITICAL", msg, **kwargs)

    def trade(
        self,
        event: str,
        symbol: str,
        trade_id: str,
        tf: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Dedicated trade event logger — always INFO level."""
        self._emit("INFO", event, symbol=symbol, tf=tf, trade_id=trade_id, **kwargs)

    def signal(self, symbol: str, tf: str, direction: str, score: float, **kwargs: Any) -> None:
        """Log a signal generation event."""
        self._emit(
            "INFO", "signal",
            symbol=symbol, tf=tf,
            component="signal",
            direction=direction,
            score=score,
            **kwargs,
        )

    def regime(self, symbol: str, tf: str, state: str, confidence: float, **kwargs: Any) -> None:
        """Log a regime classification."""
        self._emit(
            "INFO", "regime",
            symbol=symbol, tf=tf,
            component="regime",
            state=state,
            confidence=confidence,
            **kwargs,
        )


class _JsonFormatter(logging.Formatter):
    """Pass-through formatter — messages are already JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def get_logger(name: str, level: int = logging.INFO) -> LatitudeLogger:
    """Get or create a named logger. Loggers are singletons per name."""
    if name not in _LOGGERS:
        _LOGGERS[name] = LatitudeLogger(name, level)
    return _LOGGERS[name]


def set_global_level(level: int) -> None:
    """Set log level on all existing loggers."""
    for logger in _LOGGERS.values():
        logger._logger.setLevel(level)
