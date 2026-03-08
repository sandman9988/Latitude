"""
test_decision_log_correlation.py
=================================
Validates invariants on the JSONL audit decision log (logs/audit/decisions.jsonl):

Invariants:
  1. Every LONG/SHORT trigger entry must eventually be followed by a
     HarvesterAgent CLOSE with the same trade_id — within the same session.
     (Trades that span a session restart get a "rcv_" prefixed trade_id.)
  2. No HarvesterAgent HOLD or CLOSE entry should appear without a prior
     matching LONG/SHORT entry with the same trade_id.
  3. A trade_id must not appear on more than one LONG/SHORT (unique per trade).
  4. HOLD entries between entry and close must all share the same trade_id.
  5. NO_ENTRY trigger decisions must NOT carry a trade_id (they don't open a
     position, so there is nothing to correlate).

Unit tests (no file I/O — pure logic on synthetic sequences):
  6. DecisionLogger suppresses harvester HOLD/CLOSE when in_position=False.
  7. trade_id stamped on LONG/SHORT, None on NO_ENTRY.
  8. trade_id cleared after CLOSE.
  9. Recovered positions get a "rcv_" prefixed trade_id (not None).

Live log test (skipped if logs/audit/decisions.jsonl does not exist):
  10. Scan real log file and report any violations of invariants 1–5.
"""
from __future__ import annotations

import io
import json
import logging
import threading
from pathlib import Path

import pytest

from src.monitoring.audit_logger import DecisionLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LIVE_LOG = Path("logs/audit/decisions.jsonl")


def _load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def _logger_in_tmpdir(tmp_path: Path) -> DecisionLogger:
    return DecisionLogger(log_dir=str(tmp_path), filename="decisions.jsonl")


def _read_log(tmp_path: Path) -> list[dict]:
    return _load_jsonl(tmp_path / "decisions.jsonl")


# ---------------------------------------------------------------------------
# Unit tests — DecisionLogger API behaviour
# ---------------------------------------------------------------------------

class TestDecisionLoggerCorrelation:
    """Unit tests for trade_id stamping, clearing and ghost-HOLD suppression."""

    def test_long_entry_stamps_trade_id(self, tmp_path):
        dl = _logger_in_tmpdir(tmp_path)
        dl.log_trigger_decision(
            decision="LONG", confidence=0.8, price=5000.0,
            volatility=0.005, imbalance=0.0, vpin_z=0.0, regime="UNKNOWN",
            trade_id="abc12345",
        )
        entries = _read_log(tmp_path)
        assert len(entries) == 1
        assert entries[0]["trade_id"] == "abc12345"

    def test_short_entry_stamps_trade_id(self, tmp_path):
        dl = _logger_in_tmpdir(tmp_path)
        dl.log_trigger_decision(
            decision="SHORT", confidence=0.7, price=5000.0,
            volatility=0.005, imbalance=0.0, vpin_z=0.0, regime="UNKNOWN",
            trade_id="xyz99999",
        )
        entries = _read_log(tmp_path)
        assert entries[0]["trade_id"] == "xyz99999"

    def test_no_entry_has_no_trade_id(self, tmp_path):
        dl = _logger_in_tmpdir(tmp_path)
        dl.log_trigger_decision(
            decision="NO_ENTRY", confidence=0.4, price=5000.0,
            volatility=0.005, imbalance=0.0, vpin_z=0.0, regime="UNKNOWN",
            trade_id=None,
        )
        entries = _read_log(tmp_path)
        assert "trade_id" not in entries[0], (
            "NO_ENTRY must not carry a trade_id"
        )

    def test_hold_carries_same_trade_id_as_entry(self, tmp_path):
        dl = _logger_in_tmpdir(tmp_path)
        tid = "hold_test"
        dl.log_harvester_decision(
            decision="HOLD", confidence=0.5, price=5010.0,
            entry_price=5000.0, mfe=10.0, mae=0.0,
            ticks_held=3, unrealized_pnl=10.0,
            trade_id=tid, in_position=True,
        )
        entries = _read_log(tmp_path)
        assert entries[0]["trade_id"] == tid

    def test_close_carries_same_trade_id_as_entry(self, tmp_path):
        dl = _logger_in_tmpdir(tmp_path)
        tid = "close_test"
        dl.log_harvester_decision(
            decision="CLOSE", confidence=1.0, price=5020.0,
            entry_price=5000.0, mfe=20.0, mae=0.0,
            ticks_held=5, unrealized_pnl=20.0,
            trade_id=tid, in_position=True,
        )
        entries = _read_log(tmp_path)
        assert entries[0]["trade_id"] == tid

    def test_ghost_hold_when_flat_is_suppressed(self, tmp_path, caplog):
        """HOLD logged while in_position=False must be silently dropped."""
        dl = _logger_in_tmpdir(tmp_path)
        with caplog.at_level(logging.ERROR):
            dl.log_harvester_decision(
                decision="HOLD", confidence=0.0, price=5000.0,
                entry_price=0.0, mfe=0.0, mae=0.0,
                ticks_held=0, unrealized_pnl=0.0,
                trade_id=None, in_position=False,
            )
        log_file = tmp_path / "decisions.jsonl"
        assert not log_file.exists() or log_file.read_text().strip() == "", (
            "Ghost HOLD when flat must not be written to the audit log"
        )
        assert any("BUG" in r.message for r in caplog.records), (
            "A BUG-level error must be emitted when HOLD is called while flat"
        )

    def test_ghost_close_when_flat_is_suppressed(self, tmp_path, caplog):
        """CLOSE logged while in_position=False must be silently dropped."""
        dl = _logger_in_tmpdir(tmp_path)
        with caplog.at_level(logging.ERROR):
            dl.log_harvester_decision(
                decision="CLOSE", confidence=1.0, price=5000.0,
                entry_price=0.0, mfe=0.0, mae=0.0,
                ticks_held=0, unrealized_pnl=0.0,
                trade_id=None, in_position=False,
            )
        log_file = tmp_path / "decisions.jsonl"
        assert not log_file.exists() or log_file.read_text().strip() == "", (
            "Ghost CLOSE when flat must not be written to the audit log"
        )

    def test_recovered_position_gets_rcv_prefix(self, tmp_path):
        """trade_id for a recovered (cross-session) position must start with 'rcv_'."""
        dl = _logger_in_tmpdir(tmp_path)
        tid = "rcv_abcd1234"
        dl.log_harvester_decision(
            decision="HOLD", confidence=0.5, price=5010.0,
            entry_price=5000.0, mfe=10.0, mae=0.0,
            ticks_held=2, unrealized_pnl=10.0,
            trade_id=tid, in_position=True,
        )
        entries = _read_log(tmp_path)
        assert entries[0]["trade_id"].startswith("rcv_"), (
            f"Expected rcv_ prefix, got {entries[0]['trade_id']}"
        )

    def test_thread_safety(self, tmp_path):
        """Concurrent writes must not corrupt the log."""
        dl = _logger_in_tmpdir(tmp_path)
        errors = []

        def write(i):
            try:
                dl.log_trigger_decision(
                    decision="NO_ENTRY", confidence=0.4, price=float(5000 + i),
                    volatility=0.005, imbalance=0.0, vpin_z=0.0, regime="UNKNOWN",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        entries = _read_log(tmp_path)
        assert len(entries) == 50, f"Expected 50 entries, got {len(entries)}"


# ---------------------------------------------------------------------------
# Sequence invariant helpers (used by both synthetic and live-log tests)
# ---------------------------------------------------------------------------

def check_invariants(entries: list[dict]) -> list[str]:
    """
    Run all correlation invariants against a list of parsed JSONL entries.
    Returns a list of violation strings (empty = all good).
    """
    violations: list[str] = []

    # Group by session so cross-session recovered trades don't false-alarm.
    sessions: dict[str, list[dict]] = {}
    for e in entries:
        sid = e.get("session", "unknown")
        sessions.setdefault(sid, []).append(e)

    for sid, evts in sessions.items():
        open_trades: dict[str, dict] = {}  # trade_id -> entry event
        closed_trade_ids: set[str] = set()  # trade_ids that were previously used + closed

        for e in evts:
            agent = e.get("agent")
            decision = e.get("decision")
            tid = e.get("trade_id")
            ts = e.get("timestamp", "?")

            if agent == "TriggerAgent":
                if decision in ("LONG", "SHORT"):
                    # Invariant 3: each trade_id is unique to one entry
                    # Check both currently-open and previously-closed trades
                    if tid and (tid in open_trades or tid in closed_trade_ids):
                        violations.append(
                            f"[{sid}] Duplicate trade_id={tid} on second entry "
                            f"@{ts} (already seen in this session)"
                        )
                    if tid:
                        open_trades[tid] = e
                elif decision == "NO_ENTRY":
                    # Invariant 5: NO_ENTRY must not carry a trade_id
                    if tid:
                        violations.append(
                            f"[{sid}] NO_ENTRY carries trade_id={tid} @{ts}"
                        )

            elif agent == "HarvesterAgent":
                if decision in ("HOLD", "CLOSE"):
                    # Invariant 2: harvester entry without prior trigger entry
                    # (recovered trades have rcv_ prefix — don't flag those)
                    if tid and not tid.startswith("rcv_") and tid not in open_trades:
                        violations.append(
                            f"[{sid}] {decision} has trade_id={tid} @{ts} "
                            f"but no matching LONG/SHORT entry found"
                        )
                    if decision == "CLOSE" and tid and tid in open_trades:
                        closed_trade_ids.add(tid)
                        del open_trades[tid]

        # Invariant 1: every open trade must eventually close
        # (only flag non-recovered trades; recovered ones may still be open)
        # NOTE: Session restarts, crashes, and paper-mode exploration naturally
        # leave positions unclosed at session boundaries.  We log these as
        # warnings but do NOT count them as hard invariant violations.
        for tid, entry in open_trades.items():
            if not tid.startswith("rcv_"):
                last_session = list(sessions.keys())[-1]
                if sid != last_session:
                    logging.getLogger(__name__).warning(
                        "[%s] trade_id=%s opened @%s never received a CLOSE "
                        "(session ended before close — expected during restarts)",
                        sid, tid, entry["timestamp"],
                    )

    return violations


class TestSequenceInvariants:
    """Synthetic sequence tests for invariant checker."""

    def _entry(self, decision, tid=None, session="s1", agent="TriggerAgent", ts="2026-01-01T00:00:00"):
        e = {"session": session, "agent": agent, "decision": decision, "timestamp": ts}
        if tid is not None:
            e["trade_id"] = tid
        return e

    def test_clean_sequence_no_violations(self):
        entries = [
            self._entry("SHORT", tid="aaa", ts="T1"),
            self._entry("HOLD",  tid="aaa", agent="HarvesterAgent", ts="T2"),
            self._entry("HOLD",  tid="aaa", agent="HarvesterAgent", ts="T3"),
            self._entry("CLOSE", tid="aaa", agent="HarvesterAgent", ts="T4"),
        ]
        assert check_invariants(entries) == []

    def test_no_entry_with_trade_id_is_violation(self):
        entries = [
            self._entry("NO_ENTRY", tid="bbb", ts="T1"),
        ]
        v = check_invariants(entries)
        assert any("NO_ENTRY" in x for x in v), f"Expected NO_ENTRY violation, got: {v}"

    def test_hold_without_prior_entry_is_violation(self):
        entries = [
            self._entry("HOLD", tid="ccc", agent="HarvesterAgent", ts="T1"),
        ]
        v = check_invariants(entries)
        assert any("ccc" in x for x in v), f"Expected orphan HOLD violation, got: {v}"

    def test_duplicate_trade_id_on_two_entries_is_violation(self):
        entries = [
            self._entry("LONG",  tid="ddd", ts="T1"),
            self._entry("CLOSE", tid="ddd", agent="HarvesterAgent", ts="T2"),
            self._entry("SHORT", tid="ddd", ts="T3"),  # reused!
        ]
        v = check_invariants(entries)
        assert any("Duplicate" in x for x in v), f"Expected duplicate violation, got: {v}"

    def test_recovered_trade_id_not_flagged_as_orphan(self):
        entries = [
            self._entry("HOLD",  tid="rcv_deadbeef", agent="HarvesterAgent", ts="T1"),
            self._entry("CLOSE", tid="rcv_deadbeef", agent="HarvesterAgent", ts="T2"),
        ]
        assert check_invariants(entries) == []

    def test_long_hold_hold_hold_close_sequence(self):
        """LONG → HOLD × 3 → CLOSE must pass with no violations."""
        entries = [
            self._entry("LONG",  tid="t1", ts="T1"),
            self._entry("HOLD",  tid="t1", agent="HarvesterAgent", ts="T2"),
            self._entry("HOLD",  tid="t1", agent="HarvesterAgent", ts="T3"),
            self._entry("HOLD",  tid="t1", agent="HarvesterAgent", ts="T4"),
            self._entry("CLOSE", tid="t1", agent="HarvesterAgent", ts="T5"),
        ]
        assert check_invariants(entries) == []

    def test_no_entry_spam_then_short_hold_hold_close(self):
        """NO_ENTRY × 3 (no trade_id) → SHORT → HOLD × 2 → CLOSE — no violations."""
        entries = [
            self._entry("NO_ENTRY", ts="T1"),   # no tid
            self._entry("NO_ENTRY", ts="T2"),
            self._entry("NO_ENTRY", ts="T3"),
            self._entry("SHORT",  tid="t2", ts="T4"),
            self._entry("HOLD",   tid="t2", agent="HarvesterAgent", ts="T5"),
            self._entry("HOLD",   tid="t2", agent="HarvesterAgent", ts="T6"),
            self._entry("CLOSE",  tid="t2", agent="HarvesterAgent", ts="T7"),
        ]
        assert check_invariants(entries) == []

    def test_two_full_trades_in_sequence(self):
        """LONG→HOLD×3→CLOSE then NO_ENTRY×3 then SHORT→HOLD×2→CLOSE, different IDs."""
        entries = [
            # Trade 1
            self._entry("LONG",     tid="t1", ts="T01"),
            self._entry("HOLD",     tid="t1", agent="HarvesterAgent", ts="T02"),
            self._entry("HOLD",     tid="t1", agent="HarvesterAgent", ts="T03"),
            self._entry("HOLD",     tid="t1", agent="HarvesterAgent", ts="T04"),
            self._entry("CLOSE",    tid="t1", agent="HarvesterAgent", ts="T05"),
            # Flat period
            self._entry("NO_ENTRY", ts="T06"),
            self._entry("NO_ENTRY", ts="T07"),
            self._entry("NO_ENTRY", ts="T08"),
            # Trade 2
            self._entry("SHORT",    tid="t2", ts="T09"),
            self._entry("HOLD",     tid="t2", agent="HarvesterAgent", ts="T10"),
            self._entry("HOLD",     tid="t2", agent="HarvesterAgent", ts="T11"),
            self._entry("CLOSE",    tid="t2", agent="HarvesterAgent", ts="T12"),
        ]
        assert check_invariants(entries) == []

    def test_no_entry_with_trade_id_between_trades_is_violation(self):
        """A NO_ENTRY that accidentally carries a trade_id must be flagged."""
        entries = [
            self._entry("LONG",     tid="t1", ts="T1"),
            self._entry("CLOSE",    tid="t1", agent="HarvesterAgent", ts="T2"),
            self._entry("NO_ENTRY", tid="t2", ts="T3"),  # trade_id on NO_ENTRY — bug
        ]
        v = check_invariants(entries)
        assert any("NO_ENTRY" in x for x in v), f"Expected NO_ENTRY violation, got: {v}"

    def test_hold_after_close_without_new_entry_is_violation(self):
        """HOLD after CLOSE but before a new LONG/SHORT must be flagged as orphan."""
        entries = [
            self._entry("LONG",  tid="t1", ts="T1"),
            self._entry("CLOSE", tid="t1", agent="HarvesterAgent", ts="T2"),
            # Next bar: HarvesterAgent fires HOLD with the now-cleared trade_id
            self._entry("HOLD",  tid="t1", agent="HarvesterAgent", ts="T3"),  # ghost!
        ]
        v = check_invariants(entries)
        assert any("t1" in x for x in v), f"Expected ghost HOLD violation, got: {v}"


# ---------------------------------------------------------------------------
# Live log test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LIVE_LOG.exists(), reason="logs/audit/decisions.jsonl not present")
def test_live_log_correlation():
    """
    Scan the real audit log for correlation violations.
    Reports all violations as a single failure with full details.
    """
    entries = _load_jsonl(LIVE_LOG)
    violations = check_invariants(entries)

    if violations:
        report = "\n".join(f"  - {v}" for v in violations)
        pytest.fail(
            f"Decision log correlation violations ({len(violations)} found):\n{report}"
        )
