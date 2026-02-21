"""
Tests for run_universe.py
Covers: universe I/O, dotenv loader, process helpers, all cmd_* functions,
launch_paper_bots logic, CLI arg wiring.
All subprocess/os.kill calls are mocked — no real processes are spawned.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import run_universe as ru   # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_universe(path: Path, instruments: dict) -> None:
    path.write_text(json.dumps({"version": 1, "instruments": instruments}))


def _read_universe(path: Path) -> dict:
    return json.loads(path.read_text())


def _patch_universe(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(ru, "_UNIVERSE_PATH", path)


# ---------------------------------------------------------------------------
# _load_universe / _save_universe
# ---------------------------------------------------------------------------

class TestUniverseIO:

    def test_load_returns_empty_when_missing(self, tmp_path, monkeypatch):
        _patch_universe(monkeypatch, tmp_path / "universe.json")
        result = ru._load_universe()
        assert result == {"version": 1, "instruments": {}}

    def test_load_returns_contents(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _write_universe(uni, {"XAUUSD": {"stage": "PAPER"}})
        _patch_universe(monkeypatch, uni)
        result = ru._load_universe()
        assert result["instruments"]["XAUUSD"]["stage"] == "PAPER"

    def test_load_returns_empty_on_corrupt_json(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        uni.write_text("not valid json{{{")
        _patch_universe(monkeypatch, uni)
        result = ru._load_universe()
        assert result == {"version": 1, "instruments": {}}

    def test_save_writes_and_renames_atomically(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        ru._save_universe({"version": 1, "instruments": {"EURUSD": {"stage": "MICRO"}}})
        assert uni.exists()
        assert not uni.with_suffix(".tmp").exists()
        data = _read_universe(uni)
        assert data["instruments"]["EURUSD"]["stage"] == "MICRO"

    def test_round_trip(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        original = {"version": 1, "instruments": {"BTCUSD": {"stage": "PAPER", "timeframe_minutes": 15}}}
        ru._save_universe(original)
        loaded = ru._load_universe()
        assert loaded["instruments"]["BTCUSD"]["timeframe_minutes"] == 15


# ---------------------------------------------------------------------------
# _load_dotenv
# ---------------------------------------------------------------------------

class TestLoadDotenv:

    def test_parses_key_value(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text('SYMBOL=XAUUSD\nTIMEFRAME_MINUTES=240\n')
        monkeypatch.setattr(ru, "_ENV_PATH", env)
        result = ru._load_dotenv()
        assert result["SYMBOL"] == "XAUUSD"
        assert result["TIMEFRAME_MINUTES"] == "240"

    def test_strips_quotes(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text('KEY="quoted value"\nKEY2=\'single\'\n')
        monkeypatch.setattr(ru, "_ENV_PATH", env)
        result = ru._load_dotenv()
        assert result["KEY"] == "quoted value"
        assert result["KEY2"] == "single"

    def test_ignores_comments_and_blanks(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text('# comment\n\nVALID=yes\n')
        monkeypatch.setattr(ru, "_ENV_PATH", env)
        result = ru._load_dotenv()
        assert result == {"VALID": "yes"}

    def test_returns_empty_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ru, "_ENV_PATH", tmp_path / "no_such_file")
        result = ru._load_dotenv()
        assert result == {}


# ---------------------------------------------------------------------------
# _pid_alive
# ---------------------------------------------------------------------------

class TestPidAlive:

    def test_none_returns_false(self):
        assert ru._pid_alive(None) is False

    def test_zero_returns_false(self):
        assert ru._pid_alive(0) is False

    def test_live_pid_returns_true(self):
        # Current process is definitively alive
        assert ru._pid_alive(os.getpid()) is True

    def test_dead_pid_returns_false(self):
        # PID 1 always exists on Linux but is accessible — use a certainly-dead PID
        with patch("os.kill", side_effect=OSError):
            assert ru._pid_alive(999999) is False


# ---------------------------------------------------------------------------
# cmd_promote
# ---------------------------------------------------------------------------

class TestCmdPromote:

    def test_promotes_new_symbol(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {}}

        result = ru.cmd_promote(registry, "GBPUSD", 30)

        assert result["instruments"]["GBPUSD"]["stage"] == "PAPER"
        assert result["instruments"]["GBPUSD"]["timeframe_minutes"] == 30

    def test_promotes_preserves_existing_fields(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {"USDJPY": {"stage": "OFFLINE_TRAINING", "z_omega": 2.0}}}

        result = ru.cmd_promote(registry, "USDJPY", 1440)

        assert result["instruments"]["USDJPY"]["z_omega"] == pytest.approx(2.0)
        assert result["instruments"]["USDJPY"]["stage"] == "PAPER"

    def test_promotes_with_optional_symbol_id(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {}}

        result = ru.cmd_promote(registry, "EURUSD", 60, symbol_id=1)

        assert result["instruments"]["EURUSD"]["symbol_id"] == 1

    def test_saves_universe_file(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {}}

        ru.cmd_promote(registry, "AUDCAD", 60)

        assert uni.exists()
        data = _read_universe(uni)
        assert "AUDCAD" in data["instruments"]


# ---------------------------------------------------------------------------
# cmd_demote
# ---------------------------------------------------------------------------

class TestCmdDemote:

    def test_sets_stage_to_untrained(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {"XAUUSD": {"stage": "PAPER", "paper_pid": None}}}

        result = ru.cmd_demote(registry, "XAUUSD")

        assert result["instruments"]["XAUUSD"]["stage"] == "UNTRAINED"

    def test_clears_pid_field(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {"XAUUSD": {"stage": "PAPER", "paper_pid": 12345}}}

        with patch.object(ru, "_pid_alive", return_value=False):
            result = ru.cmd_demote(registry, "XAUUSD")

        assert result["instruments"]["XAUUSD"]["paper_pid"] is None

    def test_stops_running_bot(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {"BTCUSD": {"stage": "PAPER", "paper_pid": 9999}}}

        with patch.object(ru, "_pid_alive", return_value=True), \
             patch.object(ru, "_stop_pid") as mock_stop:
            ru.cmd_demote(registry, "BTCUSD")
            mock_stop.assert_called_once_with(9999, "BTCUSD paper bot")

    def test_noop_for_unknown_symbol(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {}}

        result = ru.cmd_demote(registry, "NONEXISTENT")

        assert result == {"version": 1, "instruments": {}}

    def test_saves_universe_file(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {"EURUSD": {"stage": "PAPER", "paper_pid": None}}}

        ru.cmd_demote(registry, "EURUSD")

        data = _read_universe(uni)
        assert data["instruments"]["EURUSD"]["stage"] == "UNTRAINED"


# ---------------------------------------------------------------------------
# cmd_stop_all
# ---------------------------------------------------------------------------

class TestCmdStopAll:

    def test_stops_all_alive_bots(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {
            "version": 1,
            "instruments": {
                "XAUUSD": {"stage": "PAPER", "paper_pid": 111},
                "BTCUSD": {"stage": "PAPER", "paper_pid": 222},
            },
        }
        stopped: list[int] = []

        def fake_pid_alive(pid):
            return pid in (111, 222)

        def fake_stop(pid, label=""):
            stopped.append(pid)

        with patch.object(ru, "_pid_alive", side_effect=fake_pid_alive), \
             patch.object(ru, "_stop_pid", side_effect=fake_stop):
            ru.cmd_stop_all(registry)

        assert sorted(stopped) == [111, 222]

    def test_clears_pids_in_registry(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {"XAUUSD": {"stage": "PAPER", "paper_pid": 111}}}

        with patch.object(ru, "_pid_alive", return_value=True), \
             patch.object(ru, "_stop_pid"):
            result = ru.cmd_stop_all(registry)

        assert result["instruments"]["XAUUSD"]["paper_pid"] is None

    def test_skips_dead_pids(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {"USDJPY": {"stage": "PAPER", "paper_pid": 777}}}

        with patch.object(ru, "_pid_alive", return_value=False), \
             patch.object(ru, "_stop_pid") as mock_stop:
            ru.cmd_stop_all(registry)
            mock_stop.assert_not_called()


# ---------------------------------------------------------------------------
# launch_paper_bots
# ---------------------------------------------------------------------------

class TestLaunchPaperBots:

    def _registry_with(self, symbol: str, stage: str, pid=None, tf=240) -> dict:
        return {
            "version": 1,
            "instruments": {
                symbol: {"stage": stage, "timeframe_minutes": tf, "paper_pid": pid},
            },
        }

    def test_launches_for_paper_stage(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = self._registry_with("XAUUSD", "PAPER", tf=240)
        specs = {"XAUUSD": {"symbol_id": 41, "min_volume": 0.01}}
        base_env: dict = {}

        with patch.object(ru, "_pid_alive", return_value=False), \
             patch.object(ru, "_launch_paper_bot", return_value=5555) as mock_launch:
            result = ru.launch_paper_bots(registry, specs, base_env)

        mock_launch.assert_called_once_with("XAUUSD", 240, 41, 0.01, base_env)
        assert result["instruments"]["XAUUSD"]["paper_pid"] == 5555

    def test_skips_non_paper_stages(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = self._registry_with("BTCUSD", "MICRO", tf=15)
        specs = {"BTCUSD": {"symbol_id": 10028, "min_volume": 0.01}}

        with patch.object(ru, "_pid_alive", return_value=False), \
             patch.object(ru, "_launch_paper_bot") as mock_launch:
            ru.launch_paper_bots(registry, specs, {})

        mock_launch.assert_not_called()

    def test_skips_already_running_bot(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = self._registry_with("EURUSD", "PAPER", pid=9999, tf=60)
        specs = {"EURUSD": {"symbol_id": 1, "min_volume": 0.01}}

        with patch.object(ru, "_pid_alive", return_value=True), \
             patch.object(ru, "_launch_paper_bot") as mock_launch:
            ru.launch_paper_bots(registry, specs, {})

        mock_launch.assert_not_called()

    def test_skips_missing_symbol_id(self, tmp_path, monkeypatch, caplog):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = self._registry_with("UNKNOWN_SYM", "PAPER", tf=5)
        specs: dict = {}   # no spec for this symbol

        with patch.object(ru, "_pid_alive", return_value=False), \
             patch.object(ru, "_launch_paper_bot") as mock_launch:
            import logging
            with caplog.at_level(logging.WARNING):
                ru.launch_paper_bots(registry, specs, {})

        mock_launch.assert_not_called()
        assert any("symbol_id" in msg for msg in caplog.messages)

    def test_skips_missing_timeframe(self, tmp_path, monkeypatch, caplog):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = {"version": 1, "instruments": {"XAUUSD": {"stage": "PAPER", "paper_pid": None}}}
        specs = {"XAUUSD": {"symbol_id": 41, "min_volume": 0.01}}

        with patch.object(ru, "_pid_alive", return_value=False), \
             patch.object(ru, "_launch_paper_bot") as mock_launch:
            import logging
            with caplog.at_level(logging.WARNING):
                ru.launch_paper_bots(registry, specs, {})

        mock_launch.assert_not_called()

    def test_saves_updated_registry_after_launch(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _patch_universe(monkeypatch, uni)
        registry = self._registry_with("GBPUSD", "PAPER", tf=30)
        specs = {"GBPUSD": {"symbol_id": 3, "min_volume": 0.01}}

        with patch.object(ru, "_pid_alive", return_value=False), \
             patch.object(ru, "_launch_paper_bot", return_value=1234):
            ru.launch_paper_bots(registry, specs, {})

        data = _read_universe(uni)
        assert data["instruments"]["GBPUSD"]["paper_pid"] == 1234


# ---------------------------------------------------------------------------
# CLI wiring (main)
# ---------------------------------------------------------------------------

class TestCLI:

    def test_list_exits_zero(self, tmp_path, monkeypatch, capsys):
        uni = tmp_path / "universe.json"
        _write_universe(uni, {})
        monkeypatch.setattr(ru, "_UNIVERSE_PATH", uni)
        monkeypatch.setattr(ru, "_load_dotenv", lambda: {})
        monkeypatch.setattr(ru, "_load_symbol_specs", lambda: {})

        ret = ru.main(["--list", "--universe", str(uni)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Universe is empty" in out or "Symbol" in out

    def test_promote_then_list(self, tmp_path, monkeypatch, capsys):
        uni = tmp_path / "universe.json"
        monkeypatch.setattr(ru, "_UNIVERSE_PATH", uni)
        monkeypatch.setattr(ru, "_load_dotenv", lambda: {})
        monkeypatch.setattr(ru, "_load_symbol_specs", lambda: {})

        # Promote creates universe entry; then one-shot launch pass runs but
        # launch_paper_bots will warn about missing symbol_id and skip —
        # that's fine, we just verify the CLI doesn't crash.
        with patch.object(ru, "launch_paper_bots", side_effect=lambda reg, *a, **kw: reg):
            ret = ru.main(["--promote", "AUDUSD", "--timeframe", "60", "--universe", str(uni)])

        assert ret == 0
        data = _read_universe(uni)
        assert data["instruments"]["AUDUSD"]["stage"] == "PAPER"

    def test_demote_cli(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _write_universe(uni, {"XAUUSD": {"stage": "PAPER", "paper_pid": None}})
        monkeypatch.setattr(ru, "_UNIVERSE_PATH", uni)
        monkeypatch.setattr(ru, "_load_dotenv", lambda: {})
        monkeypatch.setattr(ru, "_load_symbol_specs", lambda: {})

        ret = ru.main(["--demote", "XAUUSD", "--universe", str(uni)])

        assert ret == 0
        data = _read_universe(uni)
        assert data["instruments"]["XAUUSD"]["stage"] == "UNTRAINED"

    def test_stop_all_cli(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _write_universe(uni, {"BTCUSD": {"stage": "PAPER", "paper_pid": None}})
        monkeypatch.setattr(ru, "_UNIVERSE_PATH", uni)
        monkeypatch.setattr(ru, "_load_dotenv", lambda: {})
        monkeypatch.setattr(ru, "_load_symbol_specs", lambda: {})

        ret = ru.main(["--stop-all", "--universe", str(uni)])
        assert ret == 0

    def test_promote_requires_timeframe(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        monkeypatch.setattr(ru, "_UNIVERSE_PATH", uni)
        monkeypatch.setattr(ru, "_load_dotenv", lambda: {})
        monkeypatch.setattr(ru, "_load_symbol_specs", lambda: {})

        with pytest.raises(SystemExit) as exc_info:
            ru.main(["--promote", "EURUSD", "--universe", str(uni)])
        assert exc_info.value.code != 0
