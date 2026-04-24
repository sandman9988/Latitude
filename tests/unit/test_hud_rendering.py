"""Rendering-level regression tests for the tabbed HUD.

These tests drive ``TabbedHUD`` with a fully-mocked ``data_dir`` and assert
structural properties of the rendered frames:

* every table's column header, data rows, and horizontal separator have the
  same visible width (ANSI escapes stripped, wide emoji counted as 2 cells);
* data tables never emit the same logical row twice;
* the render pipeline's ``\\033[H`` / ``\\033[J`` clear sequences are present
  so the live repaint cannot leak stale content from longer previous frames.

The previous ``tests/validation/test_hud_plumbing.py`` only checked a handful
of helper attributes; those assertions are preserved here alongside the much
stronger rendering checks so we do not lose coverage.
"""
from __future__ import annotations

import io
import json
import re
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from src.monitoring.hud_tabbed import (
    TabbedHUD,
    _ANSI_G,
    _ANSI_R,
    _ANSI_Y,
    _strip_ansi,
    _truncate_visible,
    _visible_width,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def hud(tmp_path: Path) -> TabbedHUD:
    """A ``TabbedHUD`` bound to a mock data dir with enough files to exercise
    every table-rendering code path (fleet, performance periods, per-symbol,
    per-tf/mode, mode breakdown, decision log)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "bot_config.json").write_text(json.dumps({
        "symbol": "XAUUSD",
        "timeframe_minutes": 15,
        "trading_mode": "paper",
        "starting_equity": 10000,
    }))

    # Two running bots — drives the ALL BOTS panel rows.
    for sym, tf in [("XAUUSD", 15), ("EURUSD", 60)]:
        (data_dir / f"paper_stats_{sym}_M{tf}.json").write_text(json.dumps({
            "symbol": sym, "timeframe_minutes": tf, "trading_mode": "paper",
            "updated_at": "2026-04-23T01:00:00+00:00",
            "connection_healthy": True, "quote_ok": True,
            "bar_count": 123,
            "trigger_buffer": 500, "harvester_buffer": 250,
            "total_trades": 2 if sym == "XAUUSD" else 1,
            "total_pnl": 12345.67 if sym == "XAUUSD" else -12.34,
            "win_rate": 0.5 if sym == "XAUUSD" else 0.0,
        }))

    # Mixed-mode trade log — drives period/symbol/mode breakdown tables.
    trades = []
    for _ in range(5):
        trades.append({
            "symbol": "XAUUSD", "timeframe_minutes": 15, "trading_mode": "paper",
            "entry_time": "2026-04-20T00:00:00+00:00",
            "exit_time":  "2026-04-20T01:00:00+00:00",
            "pnl": 12.5,
            "mfe": 25.0,
            "capture_ratio": 0.5,
        })
    for _ in range(3):
        trades.append({
            "symbol": "EURUSD", "timeframe_minutes": 60, "trading_mode": "live",
            "entry_time": "2026-04-20T00:00:00+00:00",
            "exit_time":  "2026-04-20T01:00:00+00:00",
            "pnl": -7.2,
            "mfe": 10.0,
            "capture_ratio": -0.72,
        })
    tl = data_dir / "trade_log.jsonl"
    tl.write_text("\n".join(json.dumps(t) for t in trades) + "\n")

    # A single decision-log entry so the decision log tab renders its table
    audit_dir = data_dir / "logs" / "audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "decisions.jsonl").write_text(json.dumps({
        "timestamp": "2026-04-22T14:00:00+00:00",
        "trading_mode": "paper",
        "agent": "trigger",
        "decision": "LONG",
        "confidence": 0.834,
        "context": {"regime": "TREND", "vpin_z": 0.5},
        "reasoning": {"feasibility": 0.82, "predicted_runway": 0.6, "q_spread": 0.02},
        "trade_id": "abcd1234",
    }) + "\n")

    hud = TabbedHUD()
    hud.data_dir = data_dir
    hud._trade_log_reader = hud._trade_log_reader.__class__(tl)
    hud._stats_epoch = None
    hud._refresh_data()
    hud._compute_metrics_from_trade_log()
    return hud


def _render_tab(hud: TabbedHUD, tab: str) -> str:
    fn_map = {"log": "_render_decision_log"}
    fn = getattr(hud, fn_map.get(tab, f"_render_{tab}"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


# ─── Visible-width / ANSI helpers ─────────────────────────────────────────

class TestHelpers:
    def test_strip_ansi_removes_all_csi_sequences(self):
        raw = f"{_ANSI_G}hello{_ANSI_Y} world\x1b[0m"
        assert _strip_ansi(raw) == "hello world"

    def test_visible_width_ignores_colour_codes(self):
        assert _visible_width(f"{_ANSI_R}abc\x1b[0m") == 3

    def test_visible_width_counts_wide_emoji_as_two_cells(self):
        # 📄 and 💰 are both wide glyphs used in our layouts
        assert _visible_width("📄") == 2
        assert _visible_width("💰A") == 3

    def test_truncate_visible_preserves_width_with_ansi(self):
        raw = f"{_ANSI_G}abcdef{_ANSI_R}ghij{_ANSI_Y}klm{_ANSI_G}nop{_ANSI_Y}qrst{_ANSI_R}uvwxyz"
        truncated = _truncate_visible(raw, 12)

        assert _visible_width(truncated) <= 12
        assert _strip_ansi(truncated) == "abcdefghijkl"


# ─── Table alignment (structural) ─────────────────────────────────────────

def _extract_table(frame: str, header_regex: str) -> tuple[str, list[str], list[str]]:
    """Pull the header line matching *header_regex* and every line between it
    and the next blank line out of *frame*.  Returns (header, separators, rows)."""
    lines = frame.split("\n")
    for i, line in enumerate(lines):
        if re.search(header_regex, _strip_ansi(line)):
            header = line
            seps, rows = [], []
            j = i + 1
            while j < len(lines):
                bare = _strip_ansi(lines[j])
                if not bare.strip():
                    break
                if set(bare.strip()) <= {"─", "\u2500"}:
                    seps.append(lines[j])
                else:
                    rows.append(lines[j])
                j += 1
            return header, seps, rows
    raise AssertionError(f"header matching {header_regex!r} not found in frame")


class TestTableAlignment:
    """Every table's header, separator(s), and rows must share the same
    visible width — mismatched widths were the root of the reported
    'mixed tabs column width' HUD corruption."""

    def test_all_bots_panel_alignment(self, hud: TabbedHUD):
        frame = _render_tab(hud, "overview")
        header, seps, rows = _extract_table(frame, r"^\s+Bot\s+Status\s+Bars\s+Position")
        w = _visible_width(header)
        assert seps, "ALL BOTS table must emit a horizontal separator"
        for s in seps:
            assert _visible_width(s) == w, f"separator width {s!r}"
        for r in rows:
            assert _visible_width(r) == w, f"row width mismatch: {_strip_ansi(r)!r}"

    def test_symbol_tf_snapshot_alignment(self, hud: TabbedHUD):
        frame = _render_tab(hud, "overview")
        header, seps, rows = _extract_table(frame, r"^\s+Symbol\s+TF\s+Trades\s+Win%\s+PnL")
        w = _visible_width(header)
        for s in seps:
            assert _visible_width(s) == w
        for r in rows:
            assert _visible_width(r) == w, f"row width mismatch: {_strip_ansi(r)!r}"

    def test_performance_periods_alignment(self, hud: TabbedHUD):
        frame = _render_tab(hud, "performance")
        # May appear multiple times (paper, live, combined) — every block must align.
        lines = frame.split("\n")
        headers = [i for i, ln in enumerate(lines)
                   if re.search(r"^\s+Period\s+Trades\s+Win%\s+PnL \$\s+TQR", _strip_ansi(ln))]
        assert headers, "PERIOD table header not found"
        for idx in headers:
            w = _visible_width(lines[idx])
            j = idx + 1
            while j < len(lines) and _strip_ansi(lines[j]).strip():
                assert _visible_width(lines[j]) == w, (
                    f"period row/sep width mismatch at line {j}: {_strip_ansi(lines[j])!r}"
                )
                j += 1

    def test_mode_breakdown_alignment(self, hud: TabbedHUD):
        frame = _render_tab(hud, "performance")
        header, seps, rows = _extract_table(frame, r"^\s+Mode\s+Trades\s+Win%\s+PnL")
        w = _visible_width(header)
        assert seps, "MODE BREAKDOWN table must emit separators"
        for s in seps:
            assert _visible_width(s) == w
        for r in rows:
            assert _visible_width(r) == w, f"row width mismatch: {_strip_ansi(r)!r}"

    def test_per_symbol_tf_mode_alignment(self, hud: TabbedHUD):
        frame = _render_tab(hud, "performance")
        header, seps, rows = _extract_table(
            frame, r"^\s+Symbol\s+TF\s+Mode\s+Trades\s+Win%\s+PnL"
        )
        w = _visible_width(header)
        for s in seps:
            assert _visible_width(s) == w
        for r in rows:
            assert _visible_width(r) == w, f"row width mismatch: {_strip_ansi(r)!r}"

    def test_decision_log_alignment(self, hud: TabbedHUD):
        frame = _render_tab(hud, "log")
        lines = frame.split("\n")
        for i, ln in enumerate(lines):
            if re.search(r"^\s+Time\s+Mode\s+Agent\s+Decision\s+Conf\s+Detail",
                         _strip_ansi(ln)):
                # Separator on the very next non-blank line
                j = i + 1
                while j < len(lines) and not _strip_ansi(lines[j]).strip():
                    j += 1
                assert j < len(lines)
                sep = lines[j]
                # "Detail" column is variable width — assert the table *prefix*
                # up to and including the 2-space gutter before Detail is a
                # fixed 51 visible cells.
                prefix_width = 2 + 12 + 1 + 5 + 1 + 10 + 1 + 10 + 1 + 5 + 2
                assert _visible_width(ln) >= prefix_width
                assert _visible_width(sep) >= prefix_width
                return
        pytest.fail("decision-log header not found")


# ─── Duplicate-row checks ────────────────────────────────────────────────

class TestNoDuplicateRows:
    def test_all_bots_panel_has_one_row_per_bot(self, hud: TabbedHUD):
        frame = _render_tab(hud, "overview")
        bot_keys = []
        for line in frame.split("\n"):
            bare = _strip_ansi(line)
            # Rows look like "  XAUUSD/M15   ● STALE  ..."; the bot key is the
            # first slash-separated token.
            m = re.match(r"\s{2,}([A-Z]+/M\d+)\s", bare)
            if m:
                bot_keys.append(m.group(1))
        counts = Counter(bot_keys)
        assert counts, "expected at least one ALL BOTS row"
        dupes = {k: v for k, v in counts.items() if v > 1}
        assert not dupes, f"duplicate bot rows in ALL BOTS: {dupes}"

    def test_all_bots_session_columns_use_paper_stats_not_lifetime(self, hud: TabbedHUD):
        frame = _render_tab(hud, "overview")
        bare = _strip_ansi(frame)

        assert "SessTrd" in bare
        assert "+12345.67" in bare

    def test_per_symbol_tf_mode_has_unique_rows(self, hud: TabbedHUD):
        frame = _render_tab(hud, "performance")
        lines = frame.split("\n")
        start = None
        for i, ln in enumerate(lines):
            if re.search(r"^\s+Symbol\s+TF\s+Mode\s+Trades", _strip_ansi(ln)):
                start = i + 1
                break
        assert start is not None
        keys = []
        for ln in lines[start:]:
            bare = _strip_ansi(ln)
            if not bare.strip():
                break
            if set(bare.strip()) <= {"─", "\u2500"}:
                continue
            # Row: "  EURUSD    M60    LIVE    ..."
            m = re.match(r"\s+([A-Z]+)\s+(M\d+)\s+([A-Z]+)\s+", bare)
            if m:
                keys.append(m.groups())
        counts = Counter(keys)
        dupes = {k: v for k, v in counts.items() if v > 1}
        assert not dupes, f"duplicate rows in per-symbol/tf/mode table: {dupes}"

    def test_performance_period_rows_are_unique_per_block(self, hud: TabbedHUD):
        frame = _render_tab(hud, "performance")
        lines = frame.split("\n")
        block_keys: list[set[str]] = []
        current: set[str] = set()
        in_block = False
        for ln in lines:
            bare = _strip_ansi(ln)
            if re.search(r"^\s+Period\s+Trades\s+Win%", bare):
                if current:
                    block_keys.append(current)
                current, in_block = set(), True
                continue
            if in_block:
                if not bare.strip():
                    block_keys.append(current)
                    current, in_block = set(), False
                    continue
                m = re.match(r"\s+(\S+(?:\s\w+)?)\s+\d", bare)
                if m:
                    label = m.group(1).strip()
                    assert label not in current, (
                        f"duplicate period row {label!r} in performance block"
                    )
                    current.add(label)
        if current:
            block_keys.append(current)
        assert block_keys, "no performance period blocks found"


# ─── Frame-level render pipeline ──────────────────────────────────────────

class TestFramePipeline:
    def test_compose_viewport_keeps_footer_visible_when_body_is_long(self, hud: TabbedHUD):
        hud._term_height = lambda: 12  # type: ignore[method-assign]
        hud._term_width = lambda: 40  # type: ignore[method-assign]

        frame = hud._compose_viewport_frame(
            "HEADER\nTABBAR\n",
            "\n".join(f"body {i:02d}" for i in range(30)) + "\n",
            "FOOTER TOP\nFOOTER CONTROLS\nFOOTER BOTTOM\n",
        )
        lines = frame.splitlines()

        assert len(lines) == 12
        assert "FOOTER TOP" in lines[-3]
        assert "FOOTER CONTROLS" in lines[-2]
        assert "FOOTER BOTTOM" in lines[-1]
        assert any("body scroll" in _strip_ansi(line) for line in lines)
        assert any(line.endswith("│\x1b[0m") or line.endswith("█\x1b[0m") for line in lines)

    def test_mouse_click_on_tab_range_switches_tab(self, hud: TabbedHUD):
        hud._term_width = lambda: 120  # type: ignore[method-assign]
        hud._render_tab_bar()

        perf_range = next(r for r in hud._tab_click_ranges if r[2] == "performance")
        hud._handle_mouse_event(f"<0;{perf_range[0]};10M")

        assert hud.current_tab == "performance"
        assert hud._force_redraw is True

    def test_mouse_wheel_scrolls_body(self, hud: TabbedHUD):
        hud._body_scroll_max = 20
        hud._body_scroll_offsets[hud.current_tab] = 5

        hud._handle_mouse_event("<65;10;15M")

        assert hud._body_scroll_offsets[hud.current_tab] == 8

    def test_full_frame_render_emits_clear_sequences(self, hud: TabbedHUD, capsys):
        """Every actual paint must hard-clear the screen first.  The previous
        'in-place' repaint (ESC[H only) was the root cause of tab-bleed /
        duplicate content when long lines wrapped or the frame shrank."""
        hud._force_redraw = True
        hud.current_tab = "overview"
        hud._render()
        out = capsys.readouterr().out
        assert "\x1b[2J" in out, "every paint must clear the screen first"
        assert "\x1b[H" in out,  "must home the cursor after clearing"
        assert "\x1b[J" in out,  "must erase below after writing the frame"

    def test_tab_switch_repaints_from_a_clean_slate(self, hud: TabbedHUD, capsys):
        """Switching from a long tab (log/performance) to a short tab must
        never leave fragments of the previous tab visible on-screen."""
        hud.current_tab = "performance"
        hud._force_redraw = True
        hud._render()
        capsys.readouterr()
        hud.current_tab = "overview"
        hud._force_redraw = True
        hud._render()
        out = capsys.readouterr().out
        # The paint starts with a full clear, so any content from tab 2 is
        # wiped before tab 1 is written.
        assert out.startswith("\x1b[2J\x1b[H"), (
            "tab switch must begin with ESC[2J ESC[H"
        )

    def test_render_is_idempotent_when_frame_unchanged(self, hud: TabbedHUD, capsys):
        """Flicker prevention — when the rendered frame hasn't changed we
        must emit nothing at all."""
        hud.current_tab = "overview"
        hud._force_redraw = True
        hud._render()
        capsys.readouterr()
        hud._force_redraw = False
        hud._render()                     # same frame key — should be silent
        out = capsys.readouterr().out
        assert out == "", "idempotent re-render should write nothing"


# ─── Legacy plumbing sanity (from the retired test_hud_plumbing.py) ───────

class TestPlumbing:
    def test_sparkline_generation(self, hud: TabbedHUD):
        s = hud._create_sparkline([10, -5, 15, 20, -10, 25, 30])
        assert s and isinstance(s, str)

    def test_pnl_colour_coding(self, hud: TabbedHUD):
        assert hud._pnl_color(100)  == _ANSI_G
        assert hud._pnl_color(-50)  == _ANSI_R
        assert hud._pnl_color(0)    == _ANSI_Y

    def test_tab_configuration(self, hud: TabbedHUD):
        assert len(hud.TABS) == 7
        assert hud.TABS["6"] == "log"
        assert hud.TABS["7"] == "trades"
        assert hud.TAB_ORDER == [
            "overview", "performance", "training", "risk", "market", "log", "trades",
        ]

    def test_trade_history_uses_normalized_capture_ratio(self, hud: TabbedHUD):
        hud._load_all_trades_cached()
        frame = _render_tab(hud, "trades")
        bare = _strip_ansi(frame)

        assert "Cap%" in bare
        assert "+50%" in bare

    def test_capture_ratio_prefers_negative_derived_value_for_losses(self, hud: TabbedHUD):
        # Stored ratio can be stale/clamped at 0.0 in some historical records.
        # When pnl<0 and mfe>0, tab-7 must surface negative capture.
        ratio = hud._capture_ratio_for_trade({
            "pnl": -10.0,
            "mfe": 5.0,
            "capture_ratio": 0.0,
        })
        assert ratio == pytest.approx(-2.0)

    def test_capture_ratio_parses_string_fields(self, hud: TabbedHUD):
        ratio = hud._capture_ratio_for_trade({
            "pnl": "-7.5",
            "mfe": "3.0",
            "capture_ratio": "-2.5",
        })
        assert ratio == pytest.approx(-2.5)

    def test_training_tab_shows_dynamic_rl_confidence_floors(self, hud: TabbedHUD):
        hud.training_stats = {
            "trigger_ready": True,
            "harvester_ready": True,
            "trigger_training_steps": 10,
            "harvester_training_steps": 12,
            "trigger_confidence": 0.74,
            "harvester_confidence": 0.52,
            "entry_conf_dynamic_floor": 0.66,
            "exit_conf_dynamic_floor": 0.47,
        }

        frame = _render_tab(hud, "training")
        bare = _strip_ansi(frame)

        assert "dynamic entry floor" in bare
        assert "dynamic exit floor" in bare
        assert "0.660" in bare
        assert "0.470" in bare

    def test_training_tab_marks_rl_floor_pending_when_missing(self, hud: TabbedHUD):
        hud.training_stats = {
            "trigger_ready": True,
            "harvester_ready": True,
            "trigger_training_steps": 10,
            "harvester_training_steps": 12,
        }

        frame = _render_tab(hud, "training")
        bare = _strip_ansi(frame)

        assert bare.count("risk tuner pending") >= 2

    def test_trade_rows_per_page_adapts_to_terminal_height(self, hud: TabbedHUD):
        hud._term_height = lambda: 60  # type: ignore[method-assign]

        assert hud._trade_rows_per_page() == 40

    def test_offline_job_zo_shows_na_when_validation_trades_insufficient(self, hud: TabbedHUD):
        s = hud._offline_job_zo_str(0.0, "done", 3)
        assert "n/a<5" in _strip_ansi(s)

    def test_offline_job_zo_formats_value_when_validation_trades_sufficient(self, hud: TabbedHUD):
        s = hud._offline_job_zo_str(1.2345, "done", 8)
        assert "1.2345" in _strip_ansi(s)

    def test_trade_rows_per_page_keeps_small_terminal_minimum(self, hud: TabbedHUD):
        hud._term_height = lambda: 22  # type: ignore[method-assign]

        assert hud._trade_rows_per_page() == 3

    def test_trades_viewport_shows_table_rows_before_scroll_status(self, hud: TabbedHUD):
        hud._term_height = lambda: 24  # type: ignore[method-assign]
        hud._term_width = lambda: 100  # type: ignore[method-assign]
        hud._load_all_trades_cached()
        hud.current_tab = "trades"

        body = _render_tab(hud, "trades")
        frame = hud._compose_viewport_frame("h\n" * 11, body, "f\n" * 6)
        bare = _strip_ansi(frame)

        assert "body scroll" not in bare
        assert "XAUUSD" in bare
        assert "TRADE HISTORY" not in bare

    def test_control_file_broadcast_includes_isolated_runtime_dirs(self, hud: TabbedHUD):
        bot_dir = hud.data_dir / "paper_XAUUSD_M15"
        bot_dir.mkdir()

        payload = {"active": True, "timestamp": "2026-04-23T00:00:00+00:00"}
        hud._broadcast_control_file("kill_switch.json", payload, prefix=".kill_switch_")

        assert json.loads((hud.data_dir / "kill_switch.json").read_text()) == payload
        assert json.loads((bot_dir / "kill_switch.json").read_text()) == payload

    def test_check_input_drains_multiple_queued_tab_keys(self, hud: TabbedHUD, monkeypatch):
        hud.current_tab = "overview"
        queued = iter(["\t", "\t"])
        select_calls = {"n": 0}

        class _FakeStdin:
            @staticmethod
            def fileno() -> int:
                return 0

        def _fake_select(_r, _w, _e, _t):
            # Two ready events then empty queue.
            if select_calls["n"] < 2:
                select_calls["n"] += 1
                return ([0], [], [])
            return ([], [], [])

        monkeypatch.setattr("src.monitoring.hud_tabbed.sys.stdin", _FakeStdin())
        monkeypatch.setattr("src.monitoring.hud_tabbed.select.select", _fake_select)
        monkeypatch.setattr(hud, "_read_raw", lambda: next(queued))

        handled = hud._check_input()

        assert handled is True
        assert hud.current_tab == "training"
