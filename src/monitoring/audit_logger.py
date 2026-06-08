#!/usr/bin/env python3
"""Audit Logger - Transaction & Decision Logging.
==============================================
Comprehensive audit trail for debugging and compliance.

Features:
- Transaction log: Append-only event log for audit trail
- Decision log: All agent decisions with context and confidence
- Thread-safe writes
- JSON Lines format for easy parsing
"""

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.persistence.json_io import append_jsonl_durable as _append_jsonl_durable

LOG = logging.getLogger(__name__)

# Rotate when the active log exceeds this size; keep one .1 backup.
_LOG_MAX_BYTES: int = 50 * 1024 * 1024   # 50 MB
# Check file size every N successful writes (cheap stat() call).
_LOG_ROTATE_CHECK_EVERY: int = 500


def _rotate_log_if_needed(path: Path, max_bytes: int = _LOG_MAX_BYTES) -> None:
    """Rename path → path.1 when the file exceeds max_bytes. No-op on error."""
    try:
        if path.exists() and path.stat().st_size >= max_bytes:
            backup = path.with_name(path.name + ".1")
            path.rename(backup)
            LOG.info("Rotated %s → %s (%.1f MB freed)", path.name, backup.name, backup.stat().st_size / 1e6)
    except Exception as e:
        LOG.warning("Log rotation failed for %s: %s", path, e)


def append_jsonl_durable(path: Path, entry: dict[str, Any], *, default: Any = str) -> None:
    """Append one JSONL record as a single durable O_APPEND write."""
    _append_jsonl_durable(path, entry, default=default)


class TransactionLogger:
    """Append-only transaction log for audit trail.

    Records all significant events in chronological order:
    - Session events (connect/disconnect)
    - Order submissions
    - Order fills/rejects
    - Position updates
    - Circuit breaker trips
    - Component health changes
    """

    def __init__(self, log_dir: str = "logs/audit", filename: str = "transactions.jsonl") -> None:
        """Initialize transaction logger.

        Args:
            log_dir: Directory for log files
            filename: Log filename (JSON Lines format)

        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
        self.lock = threading.Lock()
        self.session_id = f"session_{int(time.time())}"
        self._sequence = 0
        self._write_count = 0
        _rotate_log_if_needed(self.log_file)

        # Log session start
        self.log_event("SESSION_START", {"session_id": self.session_id})

    def log_event(self, event_type: str, data: dict[str, Any], severity: str = "INFO") -> None:
        """Log a transaction event.

        Args:
            event_type: Event type (SESSION_START, ORDER_SUBMIT, ORDER_FILL, etc.)
            data: Event-specific data
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)

        """
        try:
            with self.lock:
                entry = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "session": self.session_id,
                    "sequence": self._next_sequence(),
                    "event_type": event_type,
                    "severity": severity,
                    "data": data,
                }
                append_jsonl_durable(self.log_file, entry)
                self._write_count += 1
                if self._write_count % _LOG_ROTATE_CHECK_EVERY == 0:
                    _rotate_log_if_needed(self.log_file)
        except Exception as e:
            LOG.exception("[AUDIT] Failed to write transaction log: %s", e)

    def _next_sequence(self) -> int:
        self._sequence = int(getattr(self, "_sequence", 0)) + 1
        return self._sequence

    def log_order_submit(self, order_id: str, side: str, quantity: float, price: float | None = None) -> None:
        """Log order submission."""
        self.log_event(
            "ORDER_SUBMIT",
            {
                "order_id": order_id,
                "side": side,
                "quantity": quantity,
                "price": price,
            },
        )

    def log_order_fill(self, order_id: str, fill_price: float, fill_qty: float, position_id: str | None = None) -> None:
        """Log order fill."""
        self.log_event(
            "ORDER_FILL",
            {
                "order_id": order_id,
                "fill_price": fill_price,
                "fill_qty": fill_qty,
                "position_id": position_id,
            },
        )

    def log_order_reject(self, order_id: str, reason: str) -> None:
        """Log order rejection."""
        self.log_event(
            "ORDER_REJECT",
            {
                "order_id": order_id,
                "reason": reason,
            },
            severity="WARNING",
        )

    def log_position_update(self, position_id: str, net_qty: float, avg_price: float) -> None:
        """Log position update."""
        self.log_event(
            "POSITION_UPDATE",
            {
                "position_id": position_id,
                "net_qty": net_qty,
                "avg_price": avg_price,
            },
        )

    def log_position_close(self, position_id: str, pnl: float, mfe: float, mae: float) -> None:
        """Log position close."""
        self.log_event(
            "POSITION_CLOSE",
            {
                "position_id": position_id,
                "pnl": pnl,
                "mfe": mfe,
                "mae": mae,
            },
        )

    def log_circuit_breaker(self, breaker_name: str, tripped: bool, current_value: float, threshold: float) -> None:
        """Log circuit breaker state change."""
        self.log_event(
            "CIRCUIT_BREAKER",
            {
                "breaker": breaker_name,
                "tripped": tripped,
                "current": current_value,
                "threshold": threshold,
            },
            severity="WARNING" if tripped else "INFO",
        )

    def log_session_event(self, session_type: str, event: str, details: dict[str, Any] | None = None) -> None:
        """Log Open API session event (connect/disconnect/auth)."""
        self.log_event(
            "SESSION_EVENT",
            {
                "session_type": session_type,
                "event": event,
                "details": details or {},
            },
        )

    def log_component_health(self, component: str, healthy: bool, error_count: int = 0) -> None:
        """Log component health change."""
        self.log_event(
            "COMPONENT_HEALTH",
            {
                "component": component,
                "healthy": healthy,
                "error_count": error_count,
            },
            severity="ERROR" if not healthy else "INFO",
        )


class DecisionLogger:
    """Decision logging for all agents.

    Records:
    - Trigger agent decisions (entry/no-entry with confidence)
    - Harvester agent decisions (hold/exit with confidence)
    - State context at decision time
    - Reasoning/features that influenced decision
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        filename: str = "decisions.jsonl",
        trading_mode: str = "live",
        symbol: str | None = None,
        timeframe: str | None = None,
        timeframe_minutes: int | None = None,
    ) -> None:
        """Initialize decision logger.

        Args:
            log_dir: Directory for log files
            filename: Log filename (JSON Lines format)
            trading_mode: "paper" or "live" — stamped on every entry
            symbol/timeframe: optional bot scope stamped on every entry

        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
        self.lock = threading.Lock()
        self.session_id = f"session_{int(time.time())}"
        self.trading_mode = trading_mode
        self.symbol = symbol
        self.timeframe = timeframe
        self.timeframe_minutes = timeframe_minutes
        self._sequence = 0
        self._write_count = 0
        _rotate_log_if_needed(self.log_file)

    def log_decision(
        self,
        agent: str,
        decision: str,
        confidence: float,
        context: dict[str, Any],
        reasoning: dict[str, Any] | None = None,
        trade_id: str | None = None,
        position_id: list[str] | None = None,
    ) -> None:
        """Log an agent decision.

        Args:
            agent: Agent name (TriggerAgent, HarvesterAgent)
            decision: Decision made (LONG, SHORT, NO_ENTRY, HOLD, CLOSE)
            confidence: Decision confidence [0, 1]
            context: Market context (price, volatility, imbalance, etc.)
            reasoning: Features/factors that influenced decision
            trade_id: Correlation ID linking entry → hold(s) → close for one trade

        """
        # Build entry while holding lock to ensure atomic write from perspective of other threads
        with self.lock:
            entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "session": getattr(self, "session_id", None),
                "sequence": self._next_sequence(),
                "trading_mode": getattr(self, "trading_mode", "live"),
                "agent": agent,
                "decision": decision,
                "confidence": float(confidence),
                "context": context,
                "reasoning": reasoning or {},
            }
            symbol = getattr(self, "symbol", None)
            timeframe = getattr(self, "timeframe", None)
            timeframe_minutes = getattr(self, "timeframe_minutes", None)
            if symbol:
                entry["symbol"] = symbol
            if timeframe:
                entry["timeframe"] = timeframe
            if timeframe_minutes is not None:
                entry["timeframe_minutes"] = timeframe_minutes
            if trade_id is not None:
                entry["trade_id"] = trade_id
            if position_id is not None:
                entry["position_id"] = position_id if isinstance(position_id, list) else [position_id]

            # Serialize once, write once (atomic from perspective of other threads)
            try:
                append_jsonl_durable(self.log_file, entry)
                self._write_count += 1
                if self._write_count % _LOG_ROTATE_CHECK_EVERY == 0:
                    _rotate_log_if_needed(self.log_file)
            except Exception as e:
                LOG.exception("[DECISION] Failed to write decision log: %s", e)

    def _next_sequence(self) -> int:
        self._sequence = int(getattr(self, "_sequence", 0)) + 1
        return self._sequence

    def log_trigger_decision(
        self,
        decision: str,
        confidence: float,
        price: float,
        volatility: float,
        imbalance: float,
        vpin_z: float,
        regime: str,
        predicted_runway: float = 0.0,
        feasibility: float = 1.0,
        circuit_breakers_ok: bool = True,
        trade_id: str | None = None,
        position_id: list[str] | None = None,
        q_spread: float = 0.0,
    ) -> None:
        """Log TriggerAgent decision with full context."""
        self.log_decision(
            agent="TriggerAgent",
            decision=decision,
            confidence=confidence,
            context={
                "price": price,
                "volatility": volatility,
                "imbalance": imbalance,
                "vpin_z": vpin_z,
                "regime": regime,
            },
            reasoning={
                "predicted_runway": predicted_runway,
                "feasibility": feasibility,
                "circuit_breakers_ok": circuit_breakers_ok,
                "q_spread": q_spread,
            },
            trade_id=trade_id,
            position_id=position_id,
        )

    def log_harvester_decision(
        self,
        decision: str,
        confidence: float,
        price: float,
        entry_price: float,
        mfe: float,
        mae: float,
        ticks_held: int,
        unrealized_pnl: float,
        capture_ratio: float = 0.0,
        trade_id: str | None = None,
        in_position: bool = True,
        position_id: list[str] | None = None,
        q_spread: float = 0.0,
        regime: str = "UNKNOWN",
        realized_vol: float = 0.0,
        depth_ratio: float = 0.0,
        exit_floor: float = 0.0,
        trailing_stop_active: bool = False,
        trailing_stop_activation_pct: float = 0.0,
        trailing_stop_distance_pct: float = 0.0,
        breakeven_active: bool = False,
        breakeven_trigger_pct: float = 0.0,
        capture_decay_armed: bool = False,
        capture_decay_threshold: float = 0.0,
        close_reason: str = "",
    ) -> None:
        """Log HarvesterAgent decision with position context.

        Args:
            in_position: Must be True — HOLD/CLOSE are only valid when a position
                is open.  If False an error is logged and the entry is suppressed
                so the audit trail is never polluted with ghost HOLD entries.

        """
        if not in_position:
            LOG.error(
                "[DECISION] BUG: log_harvester_decision called while FLAT (decision=%s trade_id=%s) — suppressed",
                decision,
                trade_id,
            )
            return
        reasoning: dict[str, Any] = {
            "mfe": mfe,
            "mae": mae,
            "ticks_held": ticks_held,
            "capture_ratio": capture_ratio,
            "q_spread": q_spread,
            "regime": regime,
            "realized_vol": realized_vol,
            "depth_ratio": depth_ratio,
            "exit_floor": exit_floor,
            "trailing_stop_active": trailing_stop_active,
            "trailing_stop_activation_pct": trailing_stop_activation_pct,
            "trailing_stop_distance_pct": trailing_stop_distance_pct,
            "breakeven_active": breakeven_active,
            "breakeven_trigger_pct": breakeven_trigger_pct,
            "capture_decay_armed": capture_decay_armed,
            "capture_decay_threshold": capture_decay_threshold,
        }
        if close_reason:
            reasoning["close_reason"] = close_reason
        self.log_decision(
            agent="HarvesterAgent",
            decision=decision,
            confidence=confidence,
            context={
                "price": price,
                "entry_price": entry_price,
                "unrealized_pnl": unrealized_pnl,
            },
            reasoning=reasoning,
            trade_id=trade_id,
            position_id=position_id,
        )


# ==============================================================================
# Self-Test
# ==============================================================================

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Transaction Logger

        tx_log = TransactionLogger(log_dir=tmpdir, filename="test_transactions.jsonl")

        tx_log.log_order_submit("TEST001", "BUY", 0.1, 100000.0)
        tx_log.log_order_fill("TEST001", 100001.0, 0.1, "POS001")
        tx_log.log_position_close("POS001", 50.0, 75.0, 25.0)
        tx_log.log_circuit_breaker("Sortino", True, 0.3, 0.5)

        # Verify file created and readable
        log_file = Path(tmpdir) / "test_transactions.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            lines = f.readlines()

        for line in lines:
            entry = json.loads(line)

        # Test 2: Decision Logger

        dec_log = DecisionLogger(log_dir=tmpdir, filename="test_decisions.jsonl")

        dec_log.log_trigger_decision(
            decision="LONG",
            confidence=0.75,
            price=100000.0,
            volatility=0.005,
            imbalance=0.1,
            vpin_z=0.5,
            regime="TRENDING",
            predicted_runway=150.0,
            feasibility=0.85,
        )

        dec_log.log_harvester_decision(
            decision="CLOSE",
            confidence=0.90,
            price=100050.0,
            entry_price=100000.0,
            mfe=75.0,
            mae=25.0,
            ticks_held=10,
            unrealized_pnl=50.0,
            capture_ratio=0.67,
        )

        # Verify file created and readable
        log_file = Path(tmpdir) / "test_decisions.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            lines = f.readlines()

        for line in lines:
            entry = json.loads(line)
