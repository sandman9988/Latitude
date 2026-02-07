import pytest

"""Tests for paper_mode configuration."""

import os

from src.core.paper_mode import setup_paper_mode, setup_live_mode, get_paper_settings


class TestSetupPaperMode:
    def test_sets_environment_variables(self):
        _result = setup_paper_mode()
        assert os.environ["PAPER_MODE"] == "1"
        assert os.environ["DISABLE_GATES"] == "1"
        assert os.environ["FEAS_THRESHOLD"] == "0.0"
        assert os.environ["EPSILON_START"] == "1.0"
        assert os.environ["EPSILON_END"] == "0.1"
        assert os.environ["FORCE_EXPLORATION"] == "1"

    def test_returns_config_dict(self):
        result = setup_paper_mode()
        assert result["paper_mode"] is True
        assert result["disable_gates"] is True
        assert result["feasibility_threshold"] == pytest.approx(0.0)
        assert result["epsilon_start"] == pytest.approx(1.0)
        assert result["epsilon_end"] == pytest.approx(0.1)
        assert result["exploration_boost"] == pytest.approx(0.5)
        assert result["force_exploration"] is True


class TestSetupLiveMode:
    def test_sets_environment_variables(self):
        _result = setup_live_mode()
        assert os.environ["PAPER_MODE"] == "0"
        assert os.environ["DISABLE_GATES"] == "0"
        assert os.environ["EPSILON_START"] == "0.05"
        assert os.environ["FORCE_EXPLORATION"] == "0"

    def test_returns_config_dict(self):
        result = setup_live_mode()
        assert result["paper_mode"] is False
        assert result["disable_gates"] is False
        assert result["confidence_floor"] == pytest.approx(0.55)
        assert result["epsilon_start"] == pytest.approx(0.05)
        assert result["force_exploration"] is False


class TestGetPaperSettings:
    def test_reads_paper_mode_env(self):
        setup_paper_mode()  # Ensure env is set
        settings = get_paper_settings()
        assert settings["paper_mode"] is True
        assert settings["disable_gates"] is True

    def test_reads_live_mode_env(self):
        setup_live_mode()
        settings = get_paper_settings()
        assert settings["paper_mode"] is False

    def test_defaults_when_no_env(self):
        # Clear relevant env vars
        for key in ["PAPER_MODE", "DISABLE_GATES", "FEAS_THRESHOLD",
                     "EPSILON_START", "EPSILON_END", "EPSILON_DECAY",
                     "EXPLORATION_BOOST", "MAX_BARS_INACTIVE",
                     "MIN_TRADES_PER_DAY", "FORCE_EXPLORATION"]:
            os.environ.pop(key, None)

        settings = get_paper_settings()
        assert settings["paper_mode"] is False
        assert settings["feasibility_threshold"] == pytest.approx(0.5)
