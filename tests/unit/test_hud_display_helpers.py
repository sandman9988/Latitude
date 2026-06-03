"""Unit tests for HUD display helpers — offline training rendering, pipeline
card uptime formatting, status badges, and progress bars.

These tests cover features that were recently fixed (offline elapsed
computation from ``started_at``, pipeline uptime ``int()`` wrapping)
and the supporting helpers that compose them into the rendered frame.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.monitoring.hud_tabbed import (
    _ANSI_DIM,
    _ANSI_G,
    _ANSI_R,
    _ANSI_Y,
    TabbedHUD,
    _strip_ansi,
)

if TYPE_CHECKING:
    from pathlib import Path

UTC = UTC


# ═══════════════════════════════════════════════════════════════════════════
# Offline training elapsed
# ═══════════════════════════════════════════════════════════════════════════


class TestOfflineTrainingElapsed:
    """``_render_offline_training`` computes elapsed from ``started_at`` for
    running jobs and falls back to ``elapsed_s`` for stopped ones."""

    def test_running_job_computes_elapsed_from_started_at(self, tmp_path: Path):
        """A running job with a valid started_at should compute live elapsed
        based on the wall clock, not the stale ``elapsed_s`` field."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        from datetime import timedelta
        now = datetime.now(UTC)
        started = now - timedelta(minutes=45, seconds=5)  # ~45m 5s ago
        ofs = {
            "status": "running",
            "started_at": started.isoformat(),
            "elapsed_s": 0.0,  # stale — should be ignored
            "total_jobs": 5,
            "results": [
                {"symbol": "XAUUSD", "status": "done"},
                {"symbol": "XAUUSD", "status": "done"},
                {"symbol": "BTCUSD", "status": "running"},
            ],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        # Should NOT show "0s" from stale elapsed_s; should show "45m" range
        assert "0s" not in text
        assert "m " in text  # human-readable minutes
        assert "45m" in text  # ~45m elapsed
        assert "RUNNING (2/5 done)" in text

    def test_complete_job_uses_stored_elapsed_s(self, tmp_path: Path):
        """A complete/stopped job falls back to the stored ``elapsed_s`` value."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        ofs = {
            "status": "complete",
            "started_at": "2026-04-26T14:30:00+00:00",
            "completed_at": "2026-04-26T15:45:00+00:00",
            "elapsed_s": 4500.0,  # 1h 15m
            "total_jobs": 3,
            "results": [
                {"symbol": "XAUUSD", "status": "done"},
                {"symbol": "XAUUSD", "status": "done"},
                {"symbol": "XAUUSD", "status": "done"},
            ],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        assert "1h 15m" in text
        assert "COMPLETE" in text
        assert "2026-04-26 14:30" in text
        assert "2026-04-26 15:45" in text

    def test_idle_job_shows_zero_elapsed(self, tmp_path: Path):
        """An idle job with no started_at shows '0m 0s' elapsed."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        ofs = {"status": "idle", "elapsed_s": 0.0}
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        assert "idle" in text
        assert "0m 0s" in text or "0s" in text

    def test_running_no_started_at_falls_back_to_elapsed_s(self, tmp_path: Path):
        """A running job missing started_at should fall back to elapsed_s."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        ofs = {"status": "running", "elapsed_s": 123.0, "total_jobs": 1, "results": []}
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        assert "2m 3s" in text

    def test_running_invalid_started_at_falls_back_to_elapsed_s(self, tmp_path: Path):
        """A running job with an unparsable started_at falls back to elapsed_s."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        ofs = {"status": "running", "started_at": "not-a-date", "elapsed_s": 75.0, "total_jobs": 1, "results": []}
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        assert "1m 15s" in text

    def test_elapsed_format_under_one_hour(self, tmp_path: Path):
        """Elapsed < 1h shows 'Xm Ys' format."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        ofs = {"status": "complete", "elapsed_s": 183.0, "total_jobs": 2, "results": []}
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        assert "3m 3s" in text

    def test_elapsed_format_over_one_hour(self, tmp_path: Path):
        """Elapsed >= 1h shows 'Xh Ym' format."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        ofs = {"status": "complete", "elapsed_s": 7320.0, "total_jobs": 2, "results": []}
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        assert "2h 2m" in text

    def test_running_elapsed_surfaces_real_minutes_not_stale_zero(self, tmp_path: Path):
        """Regression: a running job whose status file has elapsed_s=0 must
        compute elapsed from started_at rather than showing '0m 0s'."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        from datetime import timedelta
        now = datetime.now(UTC)
        started = now - timedelta(minutes=10, seconds=5)  # ~10 min ago
        ofs = {
            "status": "running",
            "started_at": started.isoformat(),
            "elapsed_s": 0.0,  # stale from incomplete status write
            "total_jobs": 1,
            "results": [],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        assert "0m 0s" not in text
        assert "10m " in text or "9m " in text or "11m " in text


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline card uptime formatting
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineUptimeFormatting:
    """Pipeline bot cards must ``int()``-wrap ``uptime_seconds`` so float
    values never leak raw decimals into the rendered frame."""

    def test_uptime_float_is_rounded_to_int(self, tmp_path: Path):
        """A float uptime_seconds like 3661.7 must show as integer hours/mins."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        # _render_pipeline_card needs universe_stats to have the entry
        _entry = {
            "symbol": "XAUUSD",
            "timeframe_minutes": 5,
            "stage": "PAPER",
            "z_omega": 1.234,
            "paper_pid": 12345,
            "_pid_alive": True,
            "_bot_stats": {
                "uptime_seconds": 3661.7,  # float — must be int-wrapped
                "quote_ok": True,
                "trade_ok": True,
                "connection_healthy": True,
                "total_reconnects": 0,
                "total_trades": 10,
                "total_pnl": 50.0,
                "bar_count": 500,
                "trigger_buffer": 500,
                "harvester_buffer": 250,
            },
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_pipeline_card("XAUUSD", _entry)
        text = _strip_ansi(buf.getvalue())

        # Must show integers, not float decimals
        assert "1h 1m" in text
        assert "1.0167h" not in text  # float leak would look like this

    def test_uptime_zero_shows_dash(self, tmp_path: Path):
        """Zero uptime_seconds should show '—'."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        _entry = {
            "symbol": "XAUUSD",
            "timeframe_minutes": 15,
            "stage": "PAPER",
            "z_omega": 1.0,
            "paper_pid": None,
            "_pid_alive": False,
            "_bot_stats": {"uptime_seconds": 0, "quote_ok": False, "trade_ok": False, "connection_healthy": False},
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_pipeline_card("XAUUSD", _entry)
        text = _strip_ansi(buf.getvalue())

        assert "uptime —" in text

    def test_uptime_under_one_hour(self, tmp_path: Path):
        """Uptime < 1h shows 'Xm Ys'."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        _entry = {
            "symbol": "XAUUSD",
            "timeframe_minutes": 5,
            "stage": "PAPER",
            "z_omega": 0.0,
            "paper_pid": 1,
            "_pid_alive": True,
            "_bot_stats": {
                "uptime_seconds": 723.0,
                "quote_ok": True,
                "trade_ok": True,
                "connection_healthy": True,
                "total_reconnects": 0,
                "total_trades": 0,
                "total_pnl": 0,
                "bar_count": 0,
                "trigger_buffer": 0,
                "harvester_buffer": 0,
            },
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_pipeline_card("XAUUSD", _entry)
        text = _strip_ansi(buf.getvalue())

        assert "12m 3s" in text

    def test_uptime_over_one_hour(self, tmp_path: Path):
        """Uptime >= 1h shows 'Xh Ym'."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        _entry = {
            "symbol": "XAUUSD",
            "timeframe_minutes": 60,
            "stage": "PAPER",
            "z_omega": 2.5,
            "paper_pid": 42,
            "_pid_alive": True,
            "_bot_stats": {
                "uptime_seconds": 11100.0,
                "quote_ok": True,
                "trade_ok": True,
                "connection_healthy": True,
                "total_reconnects": 1,
                "total_trades": 5,
                "total_pnl": 100.0,
                "bar_count": 200,
                "trigger_buffer": 500,
                "harvester_buffer": 250,
            },
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_pipeline_card("XAUUSD", _entry)
        text = _strip_ansi(buf.getvalue())

        assert "3h 5m" in text


# ═══════════════════════════════════════════════════════════════════════════
# Offline status badge
# ═══════════════════════════════════════════════════════════════════════════


class TestOfflineStatusBadge:
    def test_running_badge(self):
        hud = TabbedHUD()
        badge = _strip_ansi(hud._offline_status_badge("running", 3, 10))
        assert "RUNNING (3/10 done)" in badge

    def test_complete_badge(self):
        hud = TabbedHUD()
        badge = _strip_ansi(hud._offline_status_badge("complete", 5, 5))
        assert "COMPLETE" in badge
        assert "5/5 jobs" in badge

    def test_idle_badge(self):
        hud = TabbedHUD()
        badge = _strip_ansi(hud._offline_status_badge("idle", 0, 0))
        assert "idle" in badge

    def test_unknown_status_badge(self):
        hud = TabbedHUD()
        badge = _strip_ansi(hud._offline_status_badge("error", 0, 3))
        assert "error" in badge


# ═══════════════════════════════════════════════════════════════════════════
# Offline progress bar
# ═══════════════════════════════════════════════════════════════════════════


class TestOfflineProgressBar:
    def test_zero_progress(self):
        hud = TabbedHUD()
        bar = _strip_ansi(hud._offline_progress_bar(0, 10))
        assert "0%" in bar
        assert "░" in bar

    def test_full_progress(self):
        hud = TabbedHUD()
        bar = _strip_ansi(hud._offline_progress_bar(10, 10))
        assert "100%" in bar
        assert "█" in bar

    def test_partial_progress(self):
        hud = TabbedHUD()
        bar = _strip_ansi(hud._offline_progress_bar(5, 10))
        assert "50%" in bar
        assert "█" in bar
        assert "░" in bar

    def test_no_total(self):
        hud = TabbedHUD()
        bar = _strip_ansi(hud._offline_progress_bar(0, 0))
        assert "0%" in bar


# ═══════════════════════════════════════════════════════════════════════════
# Offline job helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestOfflineJobStatus:
    def test_done_status(self):
        hud = TabbedHUD()
        col, badge = hud._offline_job_status("done")
        assert badge.strip() == "done"
        assert col == _ANSI_G

    def test_error_status(self):
        hud = TabbedHUD()
        col, badge = hud._offline_job_status("error")
        assert badge.strip() == "ERROR"
        assert col == _ANSI_R

    def test_running_status(self):
        hud = TabbedHUD()
        col, badge = hud._offline_job_status("running")
        assert badge.strip() == "running"
        assert col == _ANSI_Y

    def test_queued_status(self):
        hud = TabbedHUD()
        col, badge = hud._offline_job_status("queued")
        assert badge.strip() == "queued"
        assert col == _ANSI_DIM


class TestOfflineJobZoStr:
    def test_none_zo_returns_dash(self):
        hud = TabbedHUD()
        s = _strip_ansi(hud._offline_job_zo_str(None, "done", 10))
        assert "—" in s

    def test_not_done_returns_dash(self):
        hud = TabbedHUD()
        s = _strip_ansi(hud._offline_job_zo_str(1.5, "running", 10))
        assert "—" in s

    def test_insufficient_validation_trades_shows_na(self):
        hud = TabbedHUD()
        s = _strip_ansi(hud._offline_job_zo_str(0.5, "done", 3))
        assert "n/a<5" in s

    def test_high_zo_is_green(self):
        hud = TabbedHUD()
        _col, _ = hud._offline_job_status("done")
        s = hud._offline_job_zo_str(1.5, "done", 10)
        assert _ANSI_G in s
        assert "1.5000" in _strip_ansi(s)

    def test_warm_zo_is_yellow(self):
        hud = TabbedHUD()
        s = hud._offline_job_zo_str(0.3, "done", 10)
        assert _ANSI_Y in s or _ANSI_R in s

    def test_low_zo_is_red(self):
        hud = TabbedHUD()
        s = hud._offline_job_zo_str(0.01, "done", 10)
        assert _ANSI_R in s


class TestOfflineJobDetail:
    def test_done_detail(self):
        hud = TabbedHUD()
        detail = hud._offline_job_detail("done", {"train_trades": 500, "val_trades": 100, "total_train_steps": 3000})
        assert "tr=500" in detail
        assert "val=100" in detail
        assert "steps=3,000" in detail

    def test_error_detail(self):
        hud = TabbedHUD()
        detail = hud._offline_job_detail("error", {"train_trades": 200, "val_trades": 50, "total_train_steps": 1000})
        assert "tr=200" in detail
        assert "val=50" in detail

    def test_running_detail_with_progress(self):
        hud = TabbedHUD()
        hud.offline_job_progress = {("XAUUSD", 5): {"pct": 65.0, "epsilon": 0.15, "beta": 0.4}}
        detail = _strip_ansi(hud._offline_job_detail("running", {"symbol": "XAUUSD", "timeframe_minutes": 5}))
        assert "%" in detail
        assert "ε=0.150" in detail or "ε=0.150" in detail

    def test_running_detail_no_progress(self):
        hud = TabbedHUD()
        detail = hud._offline_job_detail("running", {"symbol": "XAUUSD", "timeframe_minutes": 15})
        assert "—" in detail

    def test_queued_detail(self):
        hud = TabbedHUD()
        detail = hud._offline_job_detail("queued", {})
        assert "—" in detail


# ═══════════════════════════════════════════════════════════════════════════
# Offline job comment column
# ═══════════════════════════════════════════════════════════════════════════


class TestOfflineJobComment:
    def test_running_with_progress_shows_run_xy(self):
        hud = TabbedHUD()
        hud.offline_job_progress = {("XAUUSD", 240): {"epoch": 1, "n_epochs": 6}}
        comment = _strip_ansi(hud._offline_job_comment("running", {"symbol": "XAUUSD", "timeframe_minutes": 240}))
        assert "run 1/6" in comment

    def test_running_without_progress_shows_dash(self):
        hud = TabbedHUD()
        comment = hud._offline_job_comment("running", {"symbol": "XAUUSD", "timeframe_minutes": 5})
        assert "—" in comment

    def test_running_progress_no_epoch(self):
        hud = TabbedHUD()
        hud.offline_job_progress = {("XAUUSD", 5): {"pct": 50.0}}
        comment = hud._offline_job_comment("running", {"symbol": "XAUUSD", "timeframe_minutes": 5})
        assert "—" in comment

    def test_done_with_accept_reason(self):
        hud = TabbedHUD()
        comment = _strip_ansi(hud._offline_job_comment("done", {"accepted": True, "accept_reason": "ZΩ>1.0"}))
        assert "ZΩ>1.0" in comment

    def test_done_candidate_not_better_than_champion_is_readable(self):
        hud = TabbedHUD()
        comment = _strip_ansi(
            hud._offline_job_comment(
                "done",
                {"accepted": False, "accept_reason": "candidate_not_better_than_champion"},
            ),
        )
        assert "kept champion" in comment

    def test_done_candidate_not_better_than_incumbent_is_readable(self):
        hud = TabbedHUD()
        comment = _strip_ansi(
            hud._offline_job_comment(
                "done",
                {"accepted": False, "accept_reason": "candidate_not_better_than_incumbent"},
            ),
        )
        assert "kept runtime" in comment

    def test_done_accepted_no_reason(self):
        hud = TabbedHUD()
        comment = _strip_ansi(hud._offline_job_comment("done", {"accepted": True}))
        assert "accepted" in comment

    def test_done_not_accepted(self):
        hud = TabbedHUD()
        comment = _strip_ansi(hud._offline_job_comment("done", {"accepted": False}))
        assert "not accepted" in comment

    def test_queued_shows_dash(self):
        hud = TabbedHUD()
        comment = hud._offline_job_comment("queued", {})
        assert "—" in comment

    def test_error_shows_dash(self):
        hud = TabbedHUD()
        comment = hud._offline_job_comment("error", {})
        assert "—" in comment


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline card — status display helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineCardStatus:
    def test_alive_pid_shows_green_pid(self, tmp_path: Path):
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        _entry = {
            "symbol": "XAUUSD",
            "timeframe_minutes": 5,
            "stage": "PAPER",
            "z_omega": 1.0,
            "paper_pid": 9999,
            "_pid_alive": True,
            "_bot_stats": {},
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_pipeline_card("XAUUSD", _entry)
        text = _strip_ansi(buf.getvalue())

        assert "PID 9999" in text
        assert "dead" not in text

    def test_dead_pid_shows_red_dead(self, tmp_path: Path):
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        _entry = {
            "symbol": "XAUUSD",
            "timeframe_minutes": 5,
            "stage": "PAPER",
            "z_omega": 1.0,
            "paper_pid": 8888,
            "_pid_alive": False,
            "_bot_stats": {},
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_pipeline_card("XAUUSD", _entry)
        text = _strip_ansi(buf.getvalue())

        assert "dead (8888)" in text

    def test_no_pid_shows_not_started(self, tmp_path: Path):
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        _entry = {
            "symbol": "XAUUSD",
            "timeframe_minutes": 5,
            "stage": "UNTRAINED",
            "z_omega": 0.0,
            "paper_pid": None,
            "_pid_alive": False,
            "_bot_stats": {},
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_pipeline_card("XAUUSD", _entry)
        text = _strip_ansi(buf.getvalue())

        assert "not started" in text


# ═══════════════════════════════════════════════════════════════════════════
# Trading pipeline header badge
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineHeaderBadge:
    def test_some_running_shows_count(self, tmp_path: Path):
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        hud.universe_stats = {
            "XAUUSD::M5": {
                "symbol": "XAUUSD",
                "timeframe_minutes": 5,
                "stage": "PAPER",
                "_pid_alive": True,
            },
            "XAUUSD::M15": {
                "symbol": "XAUUSD",
                "timeframe_minutes": 15,
                "stage": "PAPER",
                "_pid_alive": False,
            },
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_trading_pipeline()
        text = _strip_ansi(buf.getvalue())

        assert "1/2 running" in text

    def test_none_running_shows_red_zero(self, tmp_path: Path):
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        hud.universe_stats = {
            "XAUUSD::M5": {
                "symbol": "XAUUSD",
                "timeframe_minutes": 5,
                "stage": "PAPER",
                "_pid_alive": False,
            },
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_trading_pipeline()
        text = _strip_ansi(buf.getvalue())

        assert "0/1 running" in text

    def test_empty_universe_skips_rendering(self, tmp_path: Path):
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        hud.universe_stats = {}
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_trading_pipeline()
        assert buf.getvalue() == ""  # nothing printed


# ═══════════════════════════════════════════════════════════════════════════
# Offline training — full rendering
# ═══════════════════════════════════════════════════════════════════════════


class TestOfflineTrainingFullRender:
    def test_render_empty_offline(self, tmp_path: Path):
        """Empty offline state renders idle badge with no results table."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        ofs = {"status": "idle"}
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        assert "OFFLINE TRAINING" in text
        assert "idle" in text
        assert "Progress" in text
        assert "Elapsed" in text

    def test_render_offline_jobs_table_headers(self, tmp_path: Path):
        """The jobs table should show column headers and separators."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        results = [
            {"symbol": "XAUUSD", "timeframe_minutes": 5, "status": "done", "z_omega": 1.5, "val_trades": 10},
        ]
        ofs = {"status": "complete", "elapsed_s": 100.0, "total_jobs": 1, "results": results}
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        assert "Symbol" in text
        assert "TF" in text
        assert "Status" in text
        assert "Detail" in text
        assert "ZOmega" in text
        assert "Comment" in text

    def test_render_error_job_row(self, tmp_path: Path):
        """Error job rows should show ERROR badge and truncated error text."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        results = [
            {
                "symbol": "BTCUSD",
                "timeframe_minutes": 5,
                "status": "error",
                "z_omega": None,
                "error": "RuntimeError: CUDA out of memory — reduce batch size",
            },
        ]
        ofs = {"status": "complete", "elapsed_s": 50.0, "total_jobs": 1, "results": results}
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        assert "ERROR" in text
        # Error text is truncated to 30 chars in the row
        assert "CUDA out of memo" in text
        assert "RuntimeError" in text

    def test_render_mixed_job_statuses(self, tmp_path: Path):
        """Mixed done/running/queued jobs should all appear."""
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        results = [
            {"symbol": "XAUUSD", "timeframe_minutes": 5, "status": "done", "z_omega": 3.159, "val_trades": 50},
            {"symbol": "XAUUSD", "timeframe_minutes": 15, "status": "running"},
            {"symbol": "XAUUSD", "timeframe_minutes": 30, "status": "queued"},
        ]
        ofs = {"status": "running", "started_at": datetime.now(UTC).isoformat(), "total_jobs": 3, "results": results}
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_offline_training(ofs)
        text = _strip_ansi(buf.getvalue())

        assert "XAUUSD/M5" in text or "M5" in text
        assert "XAUUSD/M15" in text or "M15" in text
        assert "XAUUSD/M30" in text or "M30" in text
        assert "queued" in text


# ═══════════════════════════════════════════════════════════════════════════
# Self-healing analyzer HUD panel
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderHealthAnalyzer:
    """_render_health_analyzer reads _health_report and prints a status row."""

    def test_no_report_shows_placeholder(self):
        hud = TabbedHUD()
        hud._health_report = {}
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_health_analyzer()
        text = _strip_ansi(buf.getvalue())
        assert "SELF-HEAL" in text
        assert "no report yet" in text

    def test_healthy_report(self):
        hud = TabbedHUD()
        hud._health_report = {
            "overall_health": "HEALTHY",
            "generated_at": datetime.now(UTC).isoformat(),
            "analysis_window_hours": 4,
            "fleet": {"total_trades": 42, "win_rate": 0.55, "profit_factor": 1.35, "emergency_rate": 0.01},
            "anomalies": [],
            "corrections_applied": [],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_health_analyzer()
        text = _strip_ansi(buf.getvalue())
        assert "HEALTHY" in text
        assert "42 trades" in text
        assert "No anomalies" in text

    def test_degraded_report_shows_anomalies(self):
        hud = TabbedHUD()
        hud._health_report = {
            "overall_health": "DEGRADED",
            "generated_at": datetime.now(UTC).isoformat(),
            "analysis_window_hours": 4,
            "fleet": {"total_trades": 10, "win_rate": 0.20, "profit_factor": 0.8, "emergency_rate": 0.10},
            "anomalies": [
                {"symbol": "XAUUSD", "timeframe": "M5", "code": "DDQN_WIN_RATE_LOW"},
                {"symbol": "BTCUSD", "timeframe": "M30", "code": "EMERGENCY_RATE_HIGH"},
            ],
            "corrections_applied": [
                {"symbol": "XAUUSD", "timeframe": "M5", "parameter": "exit_confidence_threshold",
                 "old_value": 0.50, "new_value": 0.54},
            ],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_health_analyzer()
        text = _strip_ansi(buf.getvalue())
        assert "DEGRADED" in text
        assert "DDQN_WIN_RATE_LOW" in text
        assert "EMERGENCY_RATE_HIGH" in text
        assert "exit confidence threshold" in text or "exit_confidence_threshold" in text

    def test_health_report_accepts_string_corrections(self):
        hud = TabbedHUD()
        hud._health_report = {
            "overall_health": "CRITICAL",
            "generated_at": datetime.now(UTC).isoformat(),
            "analysis_window_hours": 4,
            "fleet": {"n_trades": 11, "win_rate": 0.55, "profit_factor": 1.1, "emergency_rate": 0.0},
            "anomalies": ["CB_LOCKOUT"],
            "corrections_applied": [
                "[BAD_RISK_REWARD] BTCUSD M60 data/learned_parameters.json: confidence_floor 0.5500 -> 0.5700",
            ],
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            hud._render_health_analyzer()
        text = _strip_ansi(buf.getvalue())

        assert "CRITICAL" in text
        assert "11 trades" in text
        assert "CB_LOCKOUT" in text
        assert "BAD_RISK_REWARD" in text

    def test_load_health_report_reads_file(self, tmp_path: Path):
        import json
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        report = {"overall_health": "HEALTHY", "fleet": {}, "anomalies": [], "corrections_applied": []}
        (tmp_path / "performance_health.json").write_text(json.dumps(report))
        hud._load_health_report()
        assert hud._health_report["overall_health"] == "HEALTHY"

    def test_load_health_report_missing_file(self, tmp_path: Path):
        hud = TabbedHUD()
        hud.data_dir = tmp_path
        hud._load_health_report()  # no file — should not raise
        assert hud._health_report == {}
