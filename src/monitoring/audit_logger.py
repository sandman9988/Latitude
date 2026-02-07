#!/usr/bin/env python3
"""
Audit Logger - Transaction & Decision Logging
==============================================
GAP 10.1 FIX: Comprehensive audit trail for debugging and compliance.

Features:
- Transaction log: Append-only event log for audit trail
- Decision log: All agent decisions with context and confidence
- Thread-safe writes
- JSON Lines format for easy parsing
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


class TransactionLogger:
    """
    GAP 10.1 FIX: Append-only transaction log for audit trail.

    Records all significant events in chronological order:
    - Session events (logon/logout)
    - Order submissions
    - Order fills/rejects
    - Position updates
    - Circuit breaker trips
    - Component health changes
    """

    def __init__(self, log_dir: str = "log", filename: str = "transactions.jsonl"):
        """
        Initialize transaction logger.

        Args:
            log_dir: Directory for log files
            filename: Log filename (JSON Lines format)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
        self.lock = threading.Lock()
        self.session_id = f"session_{int(time.time())}"

        LOG.info("[AUDIT] Transaction logger initialized: %s", self.log_file)

        # Log session start
        self.log_event("SESSION_START", {"session_id": self.session_id})

    def log_event(self, event_type: str, data: dict[str, Any], severity: str = "INFO"):
        """
        Log a transaction event.

        Args:
            event_type: Event type (SESSION_START, ORDER_SUBMIT, ORDER_FILL, etc.)
            data: Event-specific data
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session": self.session_id,
            "event_type": event_type,
            "severity": severity,
            "data": data,
        }

        try:
            with self.lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            LOG.error("[AUDIT] Failed to write transaction log: %s", e)

    def log_order_submit(self, order_id: str, side: str, quantity: float, price: float = None):
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

    def log_order_fill(self, order_id: str, fill_price: float, fill_qty: float, position_id: str = None):
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

    def log_order_reject(self, order_id: str, reason: str):
        """Log order rejection."""
        self.log_event(
            "ORDER_REJECT",
            {
                "order_id": order_id,
                "reason": reason,
            },
            severity="WARNING",
        )

    def log_position_update(self, position_id: str, net_qty: float, avg_price: float):
        """Log position update."""
        self.log_event(
            "POSITION_UPDATE",
            {
                "position_id": position_id,
                "net_qty": net_qty,
                "avg_price": avg_price,
            },
        )

    def log_position_close(self, position_id: str, pnl: float, mfe: float, mae: float):
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

    def log_circuit_breaker(self, breaker_name: str, tripped: bool, current_value: float, threshold: float):
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

    def log_session_event(self, session_type: str, event: str, details: dict = None):
        """Log FIX session event (logon/logout/disconnect)."""
        self.log_event(
            "SESSION_EVENT",
            {
                "session_type": session_type,
                "event": event,
                "details": details or {},
            },
        )

    def log_component_health(self, component: str, healthy: bool, error_count: int = 0):
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
    """
    Decision logging for all agents.

    Records:
    - Trigger agent decisions (entry/no-entry with confidence)
    - Harvester agent decisions (hold/exit with confidence)
    - State context at decision time
    - Reasoning/features that influenced decision
    """

    def __init__(self, log_dir: str = "log", filename: str = "decisions.jsonl"):
        """
        Initialize decision logger.

        Args:
            log_dir: Directory for log files
            filename: Log filename (JSON Lines format)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
        self.lock = threading.Lock()
        self.session_id = f"session_{int(time.time())}"

        LOG.info("[DECISION] Decision logger initialized: %s", self.log_file)

    def log_decision(
        self, agent: str, decision: str, confidence: float, context: dict[str, Any], reasoning: dict[str, Any] = None
    ):
        """
        Log an agent decision.

        Args:
            agent: Agent name (TriggerAgent, HarvesterAgent)
            decision: Decision made (LONG, SHORT, NO_ENTRY, HOLD, CLOSE)
            confidence: Decision confidence [0, 1]
            context: Market context (price, volatility, imbalance, etc.)
            reasoning: Features/factors that influenced decision
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session": self.session_id,
            "agent": agent,
            "decision": decision,
            "confidence": float(confidence),
            "context": context,
            "reasoning": reasoning or {},
        }

        try:
            with self.lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            LOG.error("[DECISION] Failed to write decision log: %s", e)

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
    ):
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
            },
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
    ):
        """Log HarvesterAgent decision with position context."""
        self.log_decision(
            agent="HarvesterAgent",
            decision=decision,
            confidence=confidence,
            context={
                "price": price,
                "entry_price": entry_price,
                "unrealized_pnl": unrealized_pnl,
            },
            reasoning={
                "mfe": mfe,
                "mae": mae,
                "ticks_held": ticks_held,
                "capture_ratio": capture_ratio,
            },
        )


# ==============================================================================
# Self-Test
# ==============================================================================

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)
    print("=" * 80)
    print("AUDIT LOGGER - TEST SUITE")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Transaction Logger
        print("\n[Test 1] Transaction Logger")
        print("-" * 80)

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

        print(f"✓ Logged {len(lines)} transactions")
        for line in lines:
            entry = json.loads(line)
            print(f"  {entry['event_type']}: {entry['data']}")

        # Test 2: Decision Logger
        print("\n[Test 2] Decision Logger")
        print("-" * 80)

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
            bars_held=10,
            unrealized_pnl=50.0,
            capture_ratio=0.67,
        )

        # Verify file created and readable
        log_file = Path(tmpdir) / "test_decisions.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            lines = f.readlines()

        print(f"✓ Logged {len(lines)} decisions")
        for line in lines:
            entry = json.loads(line)
            print(f"  {entry['agent']}: {entry['decision']} (conf={entry['confidence']:.2f})")

        print("\n" + "=" * 80)
        print("✓ All audit logger tests passed!")
        print("=" * 80)
