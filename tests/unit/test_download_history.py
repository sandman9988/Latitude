"""
Unit tests for scripts/download_ctrader_history.py
(pure utility functions only — no network calls).
"""
from __future__ import annotations

import datetime
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts/ to path so we can import the module directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import download_ctrader_history as dch  # noqa: E402


# ---------------------------------------------------------------------------
# _load_tokens_file
# ---------------------------------------------------------------------------

class TestLoadTokensFile:
    def test_parses_export_lines(self, tmp_path):
        f = tmp_path / "tokens"
        f.write_text(
            'export CTRADER_CLIENT_ID="abc123"\n'
            'export CTRADER_CLIENT_SECRET="secret"\n'
        )
        result = dch._load_tokens_file(f)
        assert result["CTRADER_CLIENT_ID"] == "abc123"
        assert result["CTRADER_CLIENT_SECRET"] == "secret"

    def test_missing_file_returns_empty(self, tmp_path):
        result = dch._load_tokens_file(tmp_path / "nonexistent")
        assert result == {}

    def test_ignores_comments_and_blank_lines(self, tmp_path):
        f = tmp_path / "tokens"
        f.write_text(
            "# This is a comment\n"
            "\n"
            'export KEY="value"\n'
        )
        result = dch._load_tokens_file(f)
        assert result == {"KEY": "value"}

    def test_handles_values_without_quotes(self, tmp_path):
        f = tmp_path / "tokens"
        f.write_text("export MY_KEY=myvalue\n")
        result = dch._load_tokens_file(f)
        assert result["MY_KEY"] == "myvalue"


# ---------------------------------------------------------------------------
# _get_cred
# ---------------------------------------------------------------------------

class TestGetCred:
    def test_cli_takes_priority_over_env(self):
        tokens = {"MY_KEY": "from_file"}
        with patch.dict(os.environ, {"MY_KEY": "from_env"}):
            assert dch._get_cred("MY_KEY", tokens, "from_cli") == "from_cli"

    def test_env_over_file(self):
        tokens = {"MY_KEY": "from_file"}
        with patch.dict(os.environ, {"MY_KEY": "from_env"}):
            assert dch._get_cred("MY_KEY", tokens, None) == "from_env"

    def test_file_as_fallback(self):
        tokens = {"MY_KEY": "from_file"}
        env = {k: v for k, v in os.environ.items() if k != "MY_KEY"}
        with patch.dict(os.environ, env, clear=True):
            assert dch._get_cred("MY_KEY", tokens, None) == "from_file"

    def test_raises_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit, match="Missing credential"):
                dch._get_cred("MISSING_KEY", {}, None)


# ---------------------------------------------------------------------------
# _detect_account_id_from_fix_cfg
# ---------------------------------------------------------------------------

class TestDetectAccountId:
    def test_parses_demo_sendercompid(self, tmp_path):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "ctrader_quote.cfg").write_text(
            "[DEFAULT]\nSenderCompID=demo.pepperstone.5179095\n"
        )
        result = dch._detect_account_id_from_fix_cfg(tmp_path)
        assert result == "5179095"

    def test_parses_live_sendercompid(self, tmp_path):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "ctrader_quote.cfg").write_text(
            "[DEFAULT]\nSenderCompID=live.broker.9988776\n"
        )
        result = dch._detect_account_id_from_fix_cfg(tmp_path)
        assert result == "9988776"

    def test_missing_cfg_returns_none(self, tmp_path):
        result = dch._detect_account_id_from_fix_cfg(tmp_path)
        assert result is None

    def test_cfg_without_sendercompid(self, tmp_path):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir()
        (cfg_dir / "ctrader_quote.cfg").write_text("[DEFAULT]\nHeartBtInt=15\n")
        result = dch._detect_account_id_from_fix_cfg(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# _dt_to_ms / _ms_to_dt round-trip
# ---------------------------------------------------------------------------

class TestTimestampConversion:
    def test_epoch_zero(self):
        epoch = datetime.datetime(1970, 1, 1)
        assert dch._dt_to_ms(epoch) == 0

    def test_known_timestamp(self):
        dt = datetime.datetime(2024, 1, 1, 0, 0, 0)
        ms = dch._dt_to_ms(dt)
        assert ms == 1_704_067_200_000

    def test_round_trip_second_precision(self):
        dt = datetime.datetime(2024, 6, 15, 12, 30, 0)
        assert dch._ms_to_dt(dch._dt_to_ms(dt)) == dt

    def test_ms_gt_zero_for_recent_date(self):
        dt = datetime.datetime(2023, 1, 1)
        assert dch._dt_to_ms(dt) > 0


# ---------------------------------------------------------------------------
# _parse_date (CLI argument type)
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_iso_format(self):
        assert dch._parse_date("2024-01-15") == datetime.datetime(2024, 1, 15)

    def test_slash_format(self):
        assert dch._parse_date("2024/06/01") == datetime.datetime(2024, 6, 1)

    def test_dmy_format(self):
        assert dch._parse_date("15-01-2024") == datetime.datetime(2024, 1, 15)

    def test_invalid_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            dch._parse_date("not-a-date")


# ---------------------------------------------------------------------------
# Timeframe mapping completeness
# ---------------------------------------------------------------------------

class TestTimeframePeriodMap:
    def test_common_timeframes_present(self):
        for tf in (1, 5, 15, 30, 60, 240, 1440):
            assert tf in dch._TF_MINUTES_TO_PERIOD, f"M{tf} missing from period map"

    def test_periods_are_unique(self):
        values = list(dch._TF_MINUTES_TO_PERIOD.values())
        assert len(values) == len(set(values))

    def test_m1_is_period_1(self):
        assert dch._TF_MINUTES_TO_PERIOD[1] == 1

    def test_h1_is_period_9(self):
        assert dch._TF_MINUTES_TO_PERIOD[60] == 9

    def test_d1_is_period_12(self):
        assert dch._TF_MINUTES_TO_PERIOD[1440] == 12


# ---------------------------------------------------------------------------
# MAX_BARS_PER_REQUEST constant
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_bars_is_4096(self):
        assert dch.MAX_BARS_PER_REQUEST == 4_096

    def test_hosts_defined(self):
        assert "ctraderapi.com" in dch.LIVE_HOST
        assert "ctraderapi.com" in dch.DEMO_HOST
        assert dch.LIVE_HOST != dch.DEMO_HOST
