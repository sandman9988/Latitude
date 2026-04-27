"""Session Selector — curses TUI wizard for configuring instrument / timeframe / mode.

3-step wizard:
  1. Select instruments   (multi-select from config/instruments.json)
  2. Select timeframes    (per instrument, multi-select)
  3. Set mode             (LIVE / PAPER / TRAIN per instrument)
  4. Review & save        → writes data/session.json

Usage (standalone):
    python3 -m src.monitoring.session_selector
    python3 src/monitoring/session_selector.py

Exit codes:
    0 — session saved successfully
    1 — cancelled / error

The selector also updates .env SYMBOL/SYMBOL_ID/TIMEFRAME_MINUTES/QTY to
reflect the primary live instrument (or first paper instrument as fallback),
keeping backward compatibility with the single-bot launcher.
"""

from __future__ import annotations

import contextlib
import curses
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

UTC = UTC

INSTRUMENTS_PATH = Path("config/instruments.json")
SESSION_PATH = Path("data/session.json")
ENV_PATH = Path(".env")

ALL_TIMEFRAMES: list[int] = [1, 5, 15, 30, 60, 240]
TF_LABEL: dict[int, str] = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "M60", 240: "M240"}
TF_DESC: dict[int, str] = {
    1: "1 minute",
    5: "5 minutes",
    15: "15 minutes",
    30: "30 minutes",
    60: "1 hour",
    240: "4 hours",
}
MODES: list[str] = ["LIVE", "PAPER", "TRAIN"]

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Instrument:
    symbol: str
    symbol_id: int
    label: str
    category: str
    default_qty: float
    available_timeframes: list[int]
    pip_size: float = 0.0001
    margin_rate: float = 0.01


@dataclass
class Selection:
    instrument: Instrument
    timeframes: list[int] = field(default_factory=list)
    mode: str = "PAPER"

    @property
    def tf_label(self) -> str:
        return "  ".join(TF_LABEL.get(tf, f"M{tf}") for tf in sorted(self.timeframes))


# ── Colour pair constants (set up in SessionSelector.__init__) ────────────────
_C_NORMAL  = 0
_C_GREEN   = 1
_C_YELLOW  = 2
_C_CYAN    = 3
_C_RED     = 4
_C_BLUE    = 5
_C_DIM     = 6
_C_INVERT  = 7  # black on white — selected row


# ── Main wizard class ─────────────────────────────────────────────────────────

class SessionSelector:
    """Multi-step curses TUI for building a session config."""

    STEP_INSTRUMENTS = 0
    STEP_TIMEFRAMES  = 1
    STEP_MODES       = 2
    STEP_REVIEW      = 3

    def __init__(self, stdscr: Any) -> None:
        self.stdscr = stdscr
        self.instruments = _load_instruments()

        # Step 1 state
        self.inst_selected: list[bool] = [False] * len(self.instruments)
        self.inst_cursor: int = 0

        # Step 2 state
        self.selections: list[Selection] = []
        self.tf_inst_idx: int = 0   # which instrument we're configuring
        self.tf_cursor: int = 0

        # Step 3 state
        self.mode_cursor: int = 0

        # Wizard position
        self.step: int = self.STEP_INSTRUMENTS
        self.result: dict | None = None

        # Curses setup
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(_C_GREEN,  curses.COLOR_GREEN,  -1)
        curses.init_pair(_C_YELLOW, curses.COLOR_YELLOW, -1)
        curses.init_pair(_C_CYAN,   curses.COLOR_CYAN,   -1)
        curses.init_pair(_C_RED,    curses.COLOR_RED,    -1)
        curses.init_pair(_C_BLUE,   curses.COLOR_BLUE,   -1)
        curses.init_pair(_C_DIM,    curses.COLOR_WHITE,  -1)
        curses.init_pair(_C_INVERT, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.curs_set(0)
        self.stdscr.keypad(True)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> dict | None:
        while True:
            self.stdscr.erase()
            if self.step == self.STEP_INSTRUMENTS:
                action = self._step_instruments()
            elif self.step == self.STEP_TIMEFRAMES:
                action = self._step_timeframes()
            elif self.step == self.STEP_MODES:
                action = self._step_modes()
            else:
                action = self._step_review()
            self.stdscr.refresh()
            if action == "quit":
                return None
            if action == "done":
                return self.result

    # ── Shared chrome ─────────────────────────────────────────────────────────

    def _header(self, title: str) -> int:
        """Draw title bar, return next free row."""
        _h, w = self.stdscr.getmaxyx()
        bar = f"  cTrader Session Selector  ·  {title}  "
        _safe_addstr(self.stdscr, 0, 0, bar.ljust(w), curses.color_pair(_C_CYAN) | curses.A_BOLD)
        return 2

    def _footer(self, hint: str) -> None:
        h, w = self.stdscr.getmaxyx()
        _safe_addstr(self.stdscr, h - 1, 0, f"  {hint}"[:w], curses.color_pair(_C_YELLOW))

    # ── Step 1 — instrument selection ─────────────────────────────────────────

    def _step_instruments(self) -> str:
        h, _ = self.stdscr.getmaxyx()
        row = self._header("Step 1 of 3  ·  Select Instruments")
        _safe_addstr(self.stdscr, row, 2,
                     "SPACE = toggle  ·  ↑/↓ = navigate  ·  ENTER = next  ·  q = quit",
                     curses.color_pair(_C_DIM) | curses.A_DIM)
        row += 2

        for i, inst in enumerate(self.instruments):
            if row + i >= h - 2:
                break
            checked = "✓" if self.inst_selected[i] else " "
            cursor  = "▶" if i == self.inst_cursor else " "
            line = f" {cursor} [{checked}]  {inst.symbol:<10}  {inst.label:<30}  {inst.category}"
            if i == self.inst_cursor:
                attr = curses.color_pair(_C_INVERT)
            elif self.inst_selected[i]:
                attr = curses.color_pair(_C_GREEN) | curses.A_BOLD
            else:
                attr = curses.color_pair(_C_DIM)
            _safe_addstr(self.stdscr, row + i, 0, line, attr)

        n = sum(self.inst_selected)
        status = f"  {n} instrument{'s' if n != 1 else ''} selected" if n else "  Select at least one instrument"
        _safe_addstr(self.stdscr, h - 3, 2, status,
                     curses.color_pair(_C_GREEN) if n else curses.color_pair(_C_RED))
        self._footer("↑/↓ navigate  │  SPACE toggle  │  ENTER next  │  q quit")

        key = self.stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        if key == curses.KEY_UP:
            self.inst_cursor = max(0, self.inst_cursor - 1)
        elif key == curses.KEY_DOWN:
            self.inst_cursor = min(len(self.instruments) - 1, self.inst_cursor + 1)
        elif key == ord(" "):
            self.inst_selected[self.inst_cursor] = not self.inst_selected[self.inst_cursor]
        elif key in (ord("\n"), curses.KEY_ENTER, 10, 13):
            if n == 0:
                pass  # need at least one
            else:
                self.selections = [
                    Selection(inst)
                    for i, inst in enumerate(self.instruments)
                    if self.inst_selected[i]
                ]
                self.tf_inst_idx = 0
                self.tf_cursor   = 0
                self.step = self.STEP_TIMEFRAMES
        return "continue"

    # ── Step 2 — timeframe selection ──────────────────────────────────────────

    def _step_timeframes(self) -> str:
        h, _ = self.stdscr.getmaxyx()
        sel  = self.selections[self.tf_inst_idx]
        inst = sel.instrument
        tfs  = inst.available_timeframes
        n_inst = len(self.selections)

        row = self._header(
            f"Step 2 of 3  ·  Timeframes  ·  {inst.symbol}  ({self.tf_inst_idx + 1}/{n_inst})",
        )
        _safe_addstr(self.stdscr, row, 2,
                     f"Select timeframes for {inst.symbol}  —  {inst.label}",
                     curses.color_pair(_C_DIM) | curses.A_DIM)
        row += 2

        for i, tf in enumerate(tfs):
            if row + i >= h - 4:
                break
            checked = "✓" if tf in sel.timeframes else " "
            cursor  = "▶" if i == self.tf_cursor else " "
            label   = TF_LABEL.get(tf, f"M{tf}")
            desc    = TF_DESC.get(tf, f"{tf} min")
            line = f" {cursor} [{checked}]  {label:<7}  {desc}"
            if i == self.tf_cursor:
                attr = curses.color_pair(_C_INVERT)
            elif tf in sel.timeframes:
                attr = curses.color_pair(_C_GREEN) | curses.A_BOLD
            else:
                attr = curses.color_pair(_C_DIM)
            _safe_addstr(self.stdscr, row + i, 0, line, attr)

        if sel.timeframes:
            summary = "  Selected: " + "  ".join(
                TF_LABEL.get(tf, f"M{tf}") for tf in sorted(sel.timeframes)
            )
            _safe_addstr(self.stdscr, h - 3, 2, summary, curses.color_pair(_C_GREEN))

        next_hint = (
            f"ENTER → next instrument ({self.tf_inst_idx + 2}/{n_inst})"
            if self.tf_inst_idx < n_inst - 1 else "ENTER → set modes"
        )
        back_hint = "b = back to instruments" if self.tf_inst_idx == 0 else "b = prev instrument"
        self._footer(f"↑/↓ navigate  │  SPACE toggle  │  {next_hint}  │  {back_hint}  │  q quit")

        key = self.stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        if key == curses.KEY_UP:
            self.tf_cursor = max(0, self.tf_cursor - 1)
        elif key == curses.KEY_DOWN:
            self.tf_cursor = min(len(tfs) - 1, self.tf_cursor + 1)
        elif key == ord(" "):
            tf = tfs[self.tf_cursor]
            if tf in sel.timeframes:
                sel.timeframes.remove(tf)
            else:
                sel.timeframes.append(tf)
        elif key in (ord("b"), ord("B"), curses.KEY_BACKSPACE, 127):
            if self.tf_inst_idx > 0:
                self.tf_inst_idx -= 1
                self.tf_cursor = 0
            else:
                self.step = self.STEP_INSTRUMENTS
        elif key in (ord("\n"), curses.KEY_ENTER, 10, 13):
            if not sel.timeframes:
                pass  # need at least one
            elif self.tf_inst_idx < n_inst - 1:
                self.tf_inst_idx += 1
                self.tf_cursor = 0
            else:
                for s in self.selections:
                    s.mode = "PAPER"
                self.mode_cursor = 0
                self.step = self.STEP_MODES
        return "continue"

    # ── Step 3 — mode selection ───────────────────────────────────────────────

    def _step_modes(self) -> str:
        h, _w = self.stdscr.getmaxyx()
        row = self._header("Step 3 of 3  ·  Trading Modes")
        _safe_addstr(self.stdscr, row, 2,
                     "↑/↓ rows  ·  ←/→ or SPACE cycle mode  ·  ENTER confirm  ·  b back",
                     curses.color_pair(_C_DIM) | curses.A_DIM)
        row += 2

        live_count = sum(1 for s in self.selections if s.mode == "LIVE")
        mode_attr  = {
            "LIVE":  curses.color_pair(_C_RED)    | curses.A_BOLD,
            "PAPER": curses.color_pair(_C_YELLOW) | curses.A_BOLD,
            "TRAIN": curses.color_pair(_C_BLUE)   | curses.A_BOLD,
        }

        for i, sel in enumerate(self.selections):
            if row + i >= h - 4:
                break
            is_cursor = (i == self.mode_cursor)
            cursor = "▶" if is_cursor else " "
            tf_str = sel.tf_label
            base   = f" {cursor} {sel.instrument.symbol:<10}  {tf_str:<28}  "
            base_attr = curses.color_pair(_C_INVERT) if is_cursor else curses.color_pair(_C_DIM)
            _safe_addstr(self.stdscr, row + i, 0, base, base_attr)

            col = len(base)
            for mode in MODES:
                if mode == sel.mode:
                    _safe_addstr(self.stdscr, row + i, col, f"[{mode}]", mode_attr[mode])
                    col += len(mode) + 2
                else:
                    _safe_addstr(self.stdscr, row + i, col, f" {mode} ",
                                 curses.color_pair(_C_DIM) | curses.A_DIM)
                    col += len(mode) + 2
                col += 1

        warn_row = row + len(self.selections) + 1
        if live_count > 1:
            _safe_addstr(self.stdscr, warn_row, 2,
                         "⚠  Only ONE LIVE instrument allowed (single FIX trade session)  —  fix before continuing",
                         curses.color_pair(_C_RED) | curses.A_BOLD)
        elif live_count == 0:
            _safe_addstr(self.stdscr, warn_row, 2,
                         "ℹ  No LIVE instrument — all bots will run in paper / training mode",
                         curses.color_pair(_C_YELLOW))
        else:
            live_sym = next(s.instrument.symbol for s in self.selections if s.mode == "LIVE")
            _safe_addstr(self.stdscr, warn_row, 2,
                         f"✓  Live trading: {live_sym}",
                         curses.color_pair(_C_GREEN) | curses.A_BOLD)

        self._footer("↑/↓ rows  │  ←/→ or SPACE cycle mode  │  ENTER confirm  │  b back  │  q quit")

        key = self.stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        if key == curses.KEY_UP:
            self.mode_cursor = max(0, self.mode_cursor - 1)
        elif key == curses.KEY_DOWN:
            self.mode_cursor = min(len(self.selections) - 1, self.mode_cursor + 1)
        elif key in (curses.KEY_RIGHT, ord(" ")):
            sel = self.selections[self.mode_cursor]
            sel.mode = MODES[(MODES.index(sel.mode) + 1) % len(MODES)]
        elif key == curses.KEY_LEFT:
            sel = self.selections[self.mode_cursor]
            sel.mode = MODES[(MODES.index(sel.mode) - 1) % len(MODES)]
        elif key in (ord("b"), ord("B")):
            self.tf_inst_idx = len(self.selections) - 1
            self.tf_cursor   = 0
            self.step = self.STEP_TIMEFRAMES
        elif key in (ord("\n"), curses.KEY_ENTER, 10, 13):
            if live_count > 1:
                pass  # must resolve conflict
            else:
                self.result = _build_session(self.selections)
                self.step = self.STEP_REVIEW
        return "continue"

    # ── Step 4 — review & save ────────────────────────────────────────────────

    def _step_review(self) -> str:
        h, _w = self.stdscr.getmaxyx()
        row = self._header("Review & Confirm")

        mode_attr = {
            "LIVE":  curses.color_pair(_C_RED)    | curses.A_BOLD,
            "PAPER": curses.color_pair(_C_YELLOW) | curses.A_BOLD,
            "TRAIN": curses.color_pair(_C_BLUE)   | curses.A_BOLD,
        }

        _safe_addstr(self.stdscr, row, 2, "Session configuration:", curses.A_BOLD)
        row += 2

        for sel in self.selections:
            if row >= h - 6:
                break
            _safe_addstr(self.stdscr, row, 4, f"{sel.mode:<8}", mode_attr.get(sel.mode, 0))
            _safe_addstr(self.stdscr, row, 13,
                         f"{sel.instrument.symbol:<12}  {sel.tf_label:<30}  qty {sel.instrument.default_qty}",
                         curses.color_pair(_C_DIM))
            row += 1

        row += 1
        _safe_addstr(self.stdscr, row, 2,
                     f"Saves to: {SESSION_PATH}  ·  Updates .env for backward compatibility",
                     curses.color_pair(_C_DIM) | curses.A_DIM)
        row += 2
        _safe_addstr(self.stdscr, row,   2, "  [ENTER / s]  Save session and exit", curses.color_pair(_C_GREEN) | curses.A_BOLD)
        _safe_addstr(self.stdscr, row+1, 2, "  [b]          Go back and edit",       curses.color_pair(_C_DIM))
        _safe_addstr(self.stdscr, row+2, 2, "  [q]          Cancel without saving",  curses.color_pair(_C_RED))

        self._footer("ENTER / s = save  │  b = back  │  q = cancel")

        key = self.stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        if key in (ord("b"), ord("B")):
            self.mode_cursor = 0
            self.step = self.STEP_MODES
        elif key in (ord("\n"), curses.KEY_ENTER, 10, 13, ord("s"), ord("S")):
            _save_session(self.result)
            return "done"
        return "continue"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_addstr(win: Any, y: int, x: int, text: str, attr: int = 0) -> None:
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    with contextlib.suppress(curses.error):
        win.addstr(y, x, text[: w - x], attr)


def _load_instruments() -> list[Instrument]:
    """Load from config/instruments.json; fall back to built-in defaults."""
    try:
        data = json.loads(INSTRUMENTS_PATH.read_text(encoding="utf-8"))
        out  = []
        for item in data.get("instruments", []):
            if "_note" in item and len(item) == 1:
                continue
            out.append(Instrument(
                symbol              = item["symbol"],
                symbol_id           = int(item["symbol_id"]),
                label               = item.get("label", item["symbol"]),
                category            = item.get("category", ""),
                default_qty         = float(item.get("default_qty", 0.1)),
                available_timeframes= list(item.get("available_timeframes", ALL_TIMEFRAMES)),
                pip_size            = float(item.get("pip_size", 0.0001)),
                margin_rate         = float(item.get("margin_rate", 0.01)),
            ))
        return out
    except Exception:
        return [
            Instrument("XAUUSD", 41,    "Gold vs USD",     "Metals",  0.5, ALL_TIMEFRAMES),
            Instrument("BTCUSD", 10028, "Bitcoin vs USD",  "Crypto",  0.1, ALL_TIMEFRAMES),
            Instrument("ETHUSD", 10031, "Ethereum vs USD", "Crypto",  0.5, ALL_TIMEFRAMES),
            Instrument("US30",   51017, "Dow Jones 30",    "Indices", 1.0, ALL_TIMEFRAMES),
        ]


def _build_session(selections: list[Selection]) -> dict:
    """Convert wizard selections into the session.json structure."""
    live: list[dict]  = []
    paper: list[dict] = []
    train_syms: list[str] = []
    train_tfs: set[int]   = set()

    for sel in selections:
        entry: dict[str, Any] = {
            "symbol":     sel.instrument.symbol,
            "symbol_id":  sel.instrument.symbol_id,
            "timeframes": sorted(sel.timeframes),
            "qty":        sel.instrument.default_qty,
        }
        if sel.mode == "LIVE":
            live.append(entry)
        elif sel.mode == "PAPER":
            paper.append(entry)
        elif sel.mode == "TRAIN":
            train_syms.append(sel.instrument.symbol)
            train_tfs.update(sel.timeframes)

    return {
        "version":    1,
        "created_at": datetime.now(UTC).isoformat(),
        "live":       live,
        "paper":      paper,
        "train": {
            "symbols":    train_syms,
            "timeframes": sorted(train_tfs),
        } if train_syms else {},
    }


def _save_session(session: dict) -> None:
    """Write data/session.json and patch .env for backward compat."""
    SESSION_PATH.parent.mkdir(exist_ok=True)
    SESSION_PATH.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")

    # Determine the primary instrument to write into .env
    # Priority: live[0] → paper[0] → none
    live  = session.get("live",  [])
    paper = session.get("paper", [])
    primary = live[0] if live else (paper[0] if paper else None)
    if primary:
        _update_env(primary)


def _update_env(entry: dict) -> None:
    """Patch SYMBOL / SYMBOL_ID / TIMEFRAME_MINUTES / QTY in .env."""
    if not ENV_PATH.exists():
        return
    tfs = sorted(entry.get("timeframes", [5]))
    updates = {
        "SYMBOL":             entry["symbol"],
        "SYMBOL_ID":          str(entry["symbol_id"]),
        "TIMEFRAME_MINUTES":  str(tfs[0]),   # primary TF for single-bot compat
        "QTY":                str(entry.get("qty", 0.1)),
    }
    try:
        lines   = ENV_PATH.read_text(encoding="utf-8").splitlines()
        seen    : set[str] = set()
        new_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in line:
                new_lines.append(line)
                continue
            key = line.partition("=")[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                seen.add(key)
            else:
                new_lines.append(line)
        for key, val in updates.items():
            if key not in seen:
                new_lines.append(f"{key}={val}")
        ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception:
        pass


# ── Public entry point ────────────────────────────────────────────────────────

def run_selector() -> int:
    """Launch the curses TUI wizard.
    Returns 0 if a session was saved, 1 if cancelled.
    """
    try:
        result: dict | None = curses.wrapper(lambda s: SessionSelector(s).run())
    except KeyboardInterrupt:
        return 1
    except Exception:
        return 1

    if result is None:
        return 1

    # Print post-curses summary
    live  = result.get("live",  [])
    paper = result.get("paper", [])
    train = result.get("train", {})
    for entry in live:
        "  ".join(TF_LABEL.get(tf, f"M{tf}") for tf in sorted(entry["timeframes"]))
    for entry in paper:
        "  ".join(TF_LABEL.get(tf, f"M{tf}") for tf in sorted(entry["timeframes"]))
    if train.get("symbols"):
        "  ".join(TF_LABEL.get(tf, f"M{tf}") for tf in sorted(train.get("timeframes", [])))
    return 0


if __name__ == "__main__":
    sys.exit(run_selector())
