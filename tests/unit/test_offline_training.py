"""
Tests for src/training — HistoricalLoader, OfflineTrainer, z_omega.
"""

from __future__ import annotations

import csv
import io
import json
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import train_offline as to
from src.training.historical_loader import (
    _detect_columns,
    _parse_datetime,
    bars_to_deque,
    load_csv,
    load_jsonl_cache,
    sliding_windows,
)
from src.training.offline_trainer import OfflineTrainer, z_omega

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_csv(rows: list[dict], headers: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=headers)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _bar_rows(n: int = 20, base: float = 90000.0, step: float = 10.0) -> list[dict]:
    t0 = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
    rows = []
    for i in range(n):
        t = t0 + timedelta(minutes=i * 5)
        o = base + i * step
        rows.append(
            {
                "Date & Time": t.strftime("%Y-%m-%d %H:%M:%S"),
                "Open": f"{o:.2f}",
                "High": f"{o + 5:.2f}",
                "Low": f"{o - 5:.2f}",
                "Close": f"{o + 2:.2f}",
            },
        )
    return rows


# ── z_omega ───────────────────────────────────────────────────────────────────


class TestZOmega:
    def test_all_gains_returns_inf(self):
        """All positive returns → no losses → +inf."""
        returns = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = z_omega(returns)
        assert result == float("inf")

    def test_all_losses_returns_zero(self):
        """All negative returns → no gains → 0.0 (gains/losses = 0/positive)."""
        returns = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        result = z_omega(returns)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_mixed_positive_omega(self):
        """More gains than losses → ZOmega > 1."""
        returns = [10.0, 8.0, 6.0, -1.0, -1.0, -1.0, -1.0, 5.0]
        result = z_omega(returns)
        assert result > 1.0

    def test_mixed_negative_omega(self):
        """Large losses dominate → ZOmega < 1."""
        returns = [0.5, 0.5, 0.5, -10.0, -8.0, -6.0, -5.0]
        result = z_omega(returns)
        assert result < 1.0

    def test_too_few_returns_gives_zero(self):
        assert z_omega([1.0, 2.0, 3.0]) == 0.0

    def test_identical_returns_gives_one(self):
        """Zero std dev edge case returns neutral 1.0."""
        returns = [5.0] * 10
        assert z_omega(returns) == pytest.approx(1.0)

    def test_instrument_agnostic(self):
        """σ-normalisation makes ZOmega invariant to return scale.
        Scaling returns by a constant factor should not change ZOmega.
        """
        base = [1.0, 2.0, -0.5, 3.0, -1.0, 1.5, 2.5, -0.2, 0.8, 1.2]
        scaled = [r * 100.0 for r in base]
        assert z_omega(base) == pytest.approx(z_omega(scaled), abs=1e-4)


class TestOfflineStatusResume:
    def test_completed_resume_entries_returns_done_jobs_only(self, monkeypatch):
        monkeypatch.delenv("CTRADER_OFFLINE_RESUME_STATUS", raising=False)
        status = {
            "status": "running",
            "results": [
                {"symbol": "XAUUSD", "timeframe_minutes": 5, "status": "done", "z_omega": 1.2},
                {"symbol": "XAUUSD", "timeframe_minutes": 1, "status": "queued"},
                {"symbol": "BTCUSD", "timeframe_minutes": 1, "status": "error"},
            ],
        }

        entries = to._completed_resume_entries(status)

        assert set(entries) == {("XAUUSD", 5)}
        assert entries[("XAUUSD", 5)]["z_omega"] == pytest.approx(1.2)

    def test_completed_resume_entries_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("CTRADER_OFFLINE_RESUME_STATUS", "0")
        status = {
            "status": "running",
            "results": [{"symbol": "XAUUSD", "timeframe_minutes": 5, "status": "done"}],
        }

        assert to._completed_resume_entries(status) == {}

    def test_offline_supervisor_payload_is_restartable(self):
        payload = to._offline_supervisor_payload(["data/history", "--workers", "2"])

        assert payload["restartable"] is True
        assert Path(payload["argv"][0]).name == "train_offline.py"
        assert payload["argv"][1:] == ["data/history", "--workers", "2"]

    def test_parser_accepts_explicit_gpu_parallel_opt_in(self):
        parser = to._build_parser()
        args = parser.parse_args(["data", "--workers", "2", "--allow-gpu-parallel"])

        assert args.workers == 2
        assert args.allow_gpu_parallel is True


# ── _detect_columns ───────────────────────────────────────────────────────────


class TestDetectColumns:
    def test_ctrader_style(self):
        headers = ["Date & Time", "Open", "High", "Low", "Close"]
        col_map = _detect_columns(headers)
        assert col_map is not None
        assert col_map["o"] == "Open"
        assert col_map["c"] == "Close"

    def test_dukascopy_style(self):
        headers = ["Gmt time", "Open", "High", "Low", "Close", "Volume"]
        col_map = _detect_columns(headers)
        assert col_map is not None
        assert col_map["dt"] == "Gmt time"

    def test_mt4_date_time_split(self):
        headers = ["DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL"]
        col_map = _detect_columns(headers)
        assert col_map is not None
        assert "date" in col_map
        assert "time" in col_map

    def test_missing_close_returns_none(self):
        headers = ["Date & Time", "Open", "High", "Low"]  # no Close
        col_map = _detect_columns(headers)
        assert col_map is None

    def test_missing_datetime_returns_none(self):
        headers = ["Open", "High", "Low", "Close"]
        col_map = _detect_columns(headers)
        assert col_map is None


# ── _parse_datetime ───────────────────────────────────────────────────────────


class TestParseDatetime:
    def test_iso_format(self):
        dt = _parse_datetime("2026-01-05 10:30:00")
        assert dt is not None
        assert dt.hour == 10
        assert dt.tzinfo is not None

    def test_dukascopy_format(self):
        dt = _parse_datetime("05.01.2026 09:00:00.000")
        assert dt is not None
        assert dt.year == 2026

    def test_mt4_format(self):
        dt = _parse_datetime("2026.01.05 10:30")
        assert dt is not None
        assert dt.minute == 30

    def test_invalid_returns_none(self):
        dt = _parse_datetime("not-a-date")
        assert dt is None

    def test_aware_input_converted_to_utc(self):
        dt = _parse_datetime("2026-01-05T10:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None


# ── load_csv ──────────────────────────────────────────────────────────────────


class TestLoadCSV:
    def test_ctrader_csv(self, tmp_path):
        rows = _bar_rows(30)
        content = _make_csv(rows, ["Date & Time", "Open", "High", "Low", "Close"])
        f = tmp_path / "XAUUSD_M5.csv"
        f.write_text(content)

        bars = load_csv(str(f))
        assert len(bars) == 30
        # Each bar is (datetime, o, h, low, c, spread_pts)
        t, _o, h, low, _c, sp = bars[0]
        assert isinstance(t, datetime)
        assert h >= low
        assert sp == 0.0  # no spread column in this fixture

    def test_max_bars_truncation(self, tmp_path):
        rows = _bar_rows(50)
        content = _make_csv(rows, ["Date & Time", "Open", "High", "Low", "Close"])
        f = tmp_path / "test.csv"
        f.write_text(content)
        bars = load_csv(str(f), max_bars=20)
        assert len(bars) == 20

    def test_sorted_ascending(self, tmp_path):
        rows = list(reversed(_bar_rows(20)))  # Deliberately reversed
        content = _make_csv(rows, ["Date & Time", "Open", "High", "Low", "Close"])
        f = tmp_path / "test.csv"
        f.write_text(content)
        bars = load_csv(str(f))
        timestamps = [b[0] for b in bars]
        assert timestamps == sorted(timestamps)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path.csv")

    def test_malformed_ohlc_skipped(self, tmp_path):
        """Bars where high < low should be silently dropped."""
        rows = _bar_rows(10)
        rows[3]["High"] = "100.0"
        rows[3]["Low"] = "200.0"  # Low > High → malformed
        content = _make_csv(rows, ["Date & Time", "Open", "High", "Low", "Close"])
        f = tmp_path / "test.csv"
        f.write_text(content)
        bars = load_csv(str(f))
        assert len(bars) == 9  # One dropped


# ── load_jsonl_cache ──────────────────────────────────────────────────────────


class TestLoadJSONLCache:
    def test_basic_load(self, xauusd_m5_cache_file):
        """Load real XAUUSD M5 bars from the live BarExperienceCache JSONL file."""
        bars = load_jsonl_cache(str(xauusd_m5_cache_file))
        assert len(bars) >= 100
        assert all(isinstance(b[0], datetime) for b in bars)
        # Bars should be sorted and have realistic XAUUSD prices (> 1000)
        prices = [b[4] for b in bars]
        assert all(p > 1000.0 for p in prices), "Expected XAUUSD prices > $1000"
        timestamps = [b[0] for b in bars]
        assert timestamps == sorted(timestamps), "Bars must be time-sorted"

    def test_corrupt_line_skipped(self, xauusd_m5_cache_file, tmp_path):
        """Appending a corrupt line to real data should not affect valid bar count."""
        original = xauusd_m5_cache_file.read_text()
        corrupt_path = tmp_path / "corrupt_cache.jsonl"
        corrupt_path.write_text(original + "this is not json\n")
        bars_clean = load_jsonl_cache(str(xauusd_m5_cache_file))
        bars_corrupt = load_jsonl_cache(str(corrupt_path))
        assert len(bars_corrupt) == len(bars_clean)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl_cache("/nonexistent/cache.jsonl")


# ── sliding_windows ───────────────────────────────────────────────────────────


class TestSlidingWindows:
    def _fake_bars(self, n: int):
        t0 = datetime(2026, 1, 5, tzinfo=UTC)
        return [(t0 + timedelta(minutes=i * 5), float(i), float(i) + 1, float(i) - 1, float(i)) for i in range(n)]

    def test_window_count(self):
        bars = self._fake_bars(20)
        windows = list(sliding_windows(bars, window=10, step=1))
        assert len(windows) == 11  # 20 - 10 + 1

    def test_step_larger_than_one(self):
        bars = self._fake_bars(20)
        windows = list(sliding_windows(bars, window=10, step=2))
        assert len(windows) == 6

    def test_window_size_correct(self):
        bars = self._fake_bars(15)
        for w in sliding_windows(bars, window=8, step=3):
            assert len(w) == 8

    def test_window_exceeds_bars_yields_nothing(self):
        bars = self._fake_bars(5)
        windows = list(sliding_windows(bars, window=10))
        assert windows == []


# ── bars_to_deque ─────────────────────────────────────────────────────────────


class TestBarsToDeque:
    def test_deque_creation(self):
        t0 = datetime(2026, 1, 5, tzinfo=UTC)
        bars = [(t0 + timedelta(minutes=i * 5), 100.0, 101.0, 99.0, 100.5) for i in range(20)]
        d = bars_to_deque(bars, maxlen=50)
        assert len(d) == 20
        assert isinstance(d, deque)

    def test_maxlen_respected(self):
        t0 = datetime(2026, 1, 5, tzinfo=UTC)
        bars = [(t0 + timedelta(minutes=i * 5), 100.0, 101.0, 99.0, 100.5) for i in range(100)]
        d = bars_to_deque(bars, maxlen=30)
        assert d.maxlen == 30
        assert len(d) == 30


# ── BarExperienceCache ────────────────────────────────────────────────────────


class TestBarExperienceCache:
    def test_disabled_cache_writes_nothing(self, tmp_path):
        from src.training.bar_experience_cache import BarExperienceCache

        cache = BarExperienceCache(
            symbol="XAUUSD",
            cache_file=str(tmp_path / "cache.jsonl"),
            enabled=False,
        )
        cache.snapshot_entry(deque())
        cache.record_trade(
            bars=deque(),
            trigger_action=1,
            trigger_reward=0.1,
            capture_reward=0.2,
            entry_price=90000.0,
            exit_price=90100.0,
            pnl_pts=100.0,
            mfe=150.0,
            mae=0.0,
        )
        assert not (tmp_path / "cache.jsonl").exists()

    def test_record_written_to_jsonl(self, tmp_path):
        from src.training.bar_experience_cache import BarExperienceCache

        t0 = datetime(2026, 1, 5, tzinfo=UTC)
        bars = deque(maxlen=100)
        for i in range(30):
            bars.append((t0 + timedelta(minutes=i * 5), 90000.0 + i, 90005.0 + i, 89995.0 + i, 90002.0 + i))

        cache_path = tmp_path / "cache.jsonl"
        cache = BarExperienceCache(
            symbol="XAUUSD",
            timeframe_minutes=5,
            cache_file=str(cache_path),
            enabled=True,
        )
        cache.snapshot_entry(bars)
        cache.record_trade(
            bars=bars,
            trigger_action=1,
            trigger_reward=0.05,
            capture_reward=0.3,
            entry_price=90000.0,
            exit_price=90100.0,
            pnl_pts=100.0,
            mfe=150.0,
            mae=20.0,
            regime="TRENDING",
            was_explore=False,
        )

        assert cache_path.exists()
        lines = [line for line in cache_path.read_text().splitlines() if line.strip()]
        assert len(lines) == 1

        rec = json.loads(lines[0])
        assert rec["symbol"] == "XAUUSD"
        assert rec["timeframe_minutes"] == 5
        assert rec["trigger_action"] == 1
        assert rec["mfe"] == pytest.approx(150.0)
        assert len(rec["exit_bars"]) > 0
        assert len(rec["entry_bars"]) > 0

    def test_multiple_trades_accumulate(self, tmp_path):
        from src.training.bar_experience_cache import BarExperienceCache

        cache_path = tmp_path / "cache.jsonl"
        cache = BarExperienceCache(cache_file=str(cache_path))
        for _ in range(5):
            cache.record_trade(
                bars=deque(),
                trigger_action=0,
                trigger_reward=0.0,
                capture_reward=0.0,
                entry_price=1.0,
                exit_price=1.0,
                pnl_pts=0.0,
                mfe=0.0,
                mae=0.0,
            )
        assert cache.record_count() == 5

    def test_entry_snapshot_cleared_after_record(self, tmp_path):
        from src.training.bar_experience_cache import BarExperienceCache

        cache = BarExperienceCache(cache_file=str(tmp_path / "c.jsonl"))
        bars = deque([(datetime(2026, 1, 5, tzinfo=UTC), 1.0, 1.1, 0.9, 1.0)])
        cache.snapshot_entry(bars)
        assert cache._entry_bars_snapshot is not None
        cache.record_trade(
            bars=bars,
            trigger_action=0,
            trigger_reward=0.0,
            capture_reward=0.0,
            entry_price=1.0,
            exit_price=1.0,
            pnl_pts=0.0,
            mfe=0.0,
            mae=0.0,
        )
        assert cache._entry_bars_snapshot is None

    def test_schema_version_in_record(self, tmp_path):
        from src.training.bar_experience_cache import SCHEMA_VERSION, BarExperienceCache

        cache_path = tmp_path / "cache.jsonl"
        cache = BarExperienceCache(cache_file=str(cache_path))
        cache.record_trade(
            bars=deque(),
            trigger_action=1,
            trigger_reward=0.1,
            capture_reward=0.2,
            entry_price=90000.0,
            exit_price=90100.0,
            pnl_pts=100.0,
            mfe=150.0,
            mae=0.0,
        )
        rec = json.loads(cache_path.read_text().strip())
        assert rec["version"] == SCHEMA_VERSION

    def test_default_cache_path_uses_runtime_dir_and_scope(self, tmp_path, monkeypatch):
        from src.training.bar_experience_cache import BarExperienceCache

        monkeypatch.setenv("CTRADER_DATA_DIR", str(tmp_path / "runtime"))
        cache = BarExperienceCache(symbol="XAU/USD", timeframe_minutes=15, enabled=False)

        assert cache.cache_file == tmp_path / "runtime" / "training_cache_XAU-USD_M15.jsonl"


class TestRetrainEligibility:
    def test_threshold_mode_retries_below_threshold(self):
        assert to._retrain_eligible({"z_omega": 0.9, "error": None}, threshold=1.0, negative_only=False)

    def test_threshold_mode_skips_at_or_above_threshold(self):
        assert not to._retrain_eligible({"z_omega": 1.0, "error": None}, threshold=1.0, negative_only=False)
        assert not to._retrain_eligible({"z_omega": 1.2, "error": None}, threshold=1.0, negative_only=False)

    def test_retries_rejected_candidate_even_above_threshold(self):
        assert to._retrain_eligible(
            {"z_omega": 1.2, "accepted": False, "error": None},
            threshold=1.0,
            negative_only=False,
        )

    def test_negative_only_mode_retries_only_negative(self):
        assert to._retrain_eligible({"z_omega": -0.01, "error": None}, threshold=1.0, negative_only=True)
        assert not to._retrain_eligible({"z_omega": 0.0, "error": None}, threshold=1.0, negative_only=True)
        assert not to._retrain_eligible({"z_omega": 0.5, "error": None}, threshold=1.0, negative_only=True)

    def test_never_retries_missing_or_errored_results(self):
        assert not to._retrain_eligible(None, threshold=1.0, negative_only=False)
        assert not to._retrain_eligible({"z_omega": -1.0, "error": "boom"}, threshold=1.0, negative_only=True)


class TestOfflineAcceptance:
    def test_bot_checkpoint_dir_is_scoped(self, tmp_path):
        path = to._bot_checkpoint_dir(tmp_path / "ckpt", "XAU/USD+", 15)
        assert path == tmp_path / "ckpt" / "XAU_USD_M15"

    def test_candidate_checkpoint_dir_can_isolate_tournament_variants(self, tmp_path):
        path = to._candidate_checkpoint_dir(tmp_path / "ckpt", "XAUUSD", 5, "fresh/long")
        assert path == tmp_path / "ckpt" / "XAUUSD_M5" / "fresh_long"

    def test_select_best_skips_rejected_candidates(self):
        best = to.select_best(
            [
                {"symbol": "XAUUSD", "timeframe_minutes": 5, "z_omega": 10.0, "accepted": False},
                {"symbol": "XAUUSD", "timeframe_minutes": 15, "z_omega": 2.0, "accepted": True},
            ],
        )
        assert best["XAUUSD"]["timeframe_minutes"] == 15

    def test_select_best_per_bot_keeps_timeframes_separate(self):
        best = to.select_best_per_bot(
            [
                {"symbol": "XAUUSD", "timeframe_minutes": 1, "z_omega": 3.0, "accepted": True},
                {"symbol": "XAUUSD", "timeframe_minutes": 5, "z_omega": 1.6, "accepted": True},
                {"symbol": "XAUUSD", "timeframe_minutes": 5, "z_omega": 10.0, "accepted": False},
            ],
        )

        assert best[("XAUUSD", 1)]["z_omega"] == 3.0
        assert best[("XAUUSD", 5)]["z_omega"] == 1.6

    def test_copy_best_weights_preserves_timeframe_in_name(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        trigger = source / "trigger_ddqn_weights.pt"
        harvester = source / "harvester_ddqn_weights.pt"
        trigger.write_bytes(b"trigger")
        harvester.write_bytes(b"harvester")

        dest = tmp_path / "best"
        to.copy_best_weights(
            {
                ("XAUUSD", 5): {
                    "symbol": "XAUUSD",
                    "timeframe_minutes": 5,
                    "z_omega": 1.6,
                    "accepted_weights_path": f"{trigger};{harvester}",
                },
            },
            dest,
        )

        assert (dest / "XAUUSD_M5_trigger_offline.pt").read_bytes() == b"trigger"
        assert (dest / "XAUUSD_M5_harvester_offline.pt").read_bytes() == b"harvester"

    def test_acceptance_requires_beating_champion_when_champion_is_stricter(self):
        decision = to._decide_acceptance(
            candidate_score=1.1022,
            incumbent_score=1.0242,
            incumbent_loaded=True,
            champion_score=1.6088,
            acceptance_margin=0.0,
        )
        assert not decision.accepted
        assert decision.reason == "candidate_not_better_than_champion"
        assert decision.guard_z_omega == pytest.approx(1.6088)

    def test_acceptance_uses_incumbent_when_incumbent_is_stricter(self):
        decision = to._decide_acceptance(
            candidate_score=1.25,
            incumbent_score=1.2,
            incumbent_loaded=True,
            champion_score=1.1,
            acceptance_margin=0.0,
        )
        assert decision.accepted
        assert decision.reason == "candidate_better_than_incumbent"

    def test_load_offline_champion_ignores_stale_training_logs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "checkpoints").mkdir(parents=True)
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "train_offline.log").write_text(
            "2026-04-25 [INFO] train_offline: [MAIN] XAUUSD_M5 done -- ZOmega=9.9999 trades=99\n",
            encoding="utf-8",
        )
        (tmp_path / "data" / "universe.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "instruments": [
                        {
                            "symbol": "XAUUSD",
                            "timeframe_minutes": 5,
                            "z_omega": 0.8793465150180925,
                        },
                    ],
                },
            ),
            encoding="utf-8",
        )

        score, source = to._load_offline_champion("data/checkpoints", "XAUUSD", 5)

        assert score == pytest.approx(0.8793465150180925)
        assert source == "data/universe.json"

    def test_load_offline_champion_prefers_registry_over_universe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ckpt = tmp_path / "data" / "checkpoints"
        ckpt.mkdir(parents=True)
        (ckpt / "offline_champions.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "champions": {
                        "XAUUSD_M5": {
                            "symbol": "XAUUSD",
                            "timeframe_minutes": 5,
                            "z_omega": 1.6088,
                        },
                    },
                },
            ),
            encoding="utf-8",
        )
        (tmp_path / "data" / "universe.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "instruments": [
                        {
                            "symbol": "XAUUSD",
                            "timeframe_minutes": 5,
                            "z_omega": 0.8793465150180925,
                        },
                    ],
                },
            ),
            encoding="utf-8",
        )

        score, source = to._load_offline_champion("data/checkpoints", "XAUUSD", 5)

        assert score == pytest.approx(1.6088)
        assert source == "data/checkpoints/offline_champions.json"

    def test_copy_candidate_to_runtime_uses_live_checkpoint_names(self, tmp_path):
        cand = tmp_path / "candidate"
        cand.mkdir()
        trigger = cand / "XAUUSD_M5_trigger_offline.pt"
        harvester = cand / "XAUUSD_M5_harvester_offline.pt"
        trigger.write_bytes(b"trigger")
        harvester.write_bytes(b"harvester")

        runtime = tmp_path / "ckpt" / "XAUUSD_M5"
        copied = to._copy_candidate_to_runtime(f"{trigger};{harvester}", runtime)

        assert sorted(Path(p).name for p in copied) == ["harvester_ddqn_weights.pt", "trigger_ddqn_weights.pt"]
        assert (runtime / "trigger_ddqn_weights.pt").read_bytes() == b"trigger"
        assert (runtime / "harvester_ddqn_weights.pt").read_bytes() == b"harvester"

    def test_deferred_candidate_deploy_copies_selected_variant_to_runtime(self, tmp_path):
        cand = tmp_path / "candidate"
        cand.mkdir()
        trigger = cand / "XAUUSD_M5_trigger_offline.pt"
        harvester = cand / "XAUUSD_M5_harvester_offline.pt"
        trigger.write_bytes(b"trigger")
        harvester.write_bytes(b"harvester")

        result = {
            "symbol": "XAUUSD",
            "timeframe_minutes": 5,
            "accepted": True,
            "weights_path": f"{trigger};{harvester}",
            "candidate_id": "fresh_long",
        }

        assert to._deploy_candidate_result(result, tmp_path / "ckpt")
        runtime = tmp_path / "ckpt" / "XAUUSD_M5"
        assert (runtime / "trigger_ddqn_weights.pt").read_bytes() == b"trigger"
        assert (runtime / "harvester_ddqn_weights.pt").read_bytes() == b"harvester"
        assert result["accepted_weights_path"]
        assert result["candidate_deploy_deferred"] is False

    def test_training_variants_are_bounded_and_unique(self):
        args = type(
            "Args",
            (),
            {
                "tournament_variants": 6,
                "n_epochs": 3,
                "train_every": 4,
                "epsilon_start": 0.4,
                "epsilon_end": 0.05,
                "penalty_scale": 1.0,
                "focused_cap_passes": 2,
                "warm_start": True,
            },
        )()

        variants = to._build_training_variants(args)

        assert len(variants) == 6
        assert variants[0].name == "base"
        assert len({v.name for v in variants}) == 6
        assert any(not v.warm_start for v in variants)
        assert all(v.n_epochs >= 3 for v in variants)

    def test_max_bars_by_timeframe_overrides_global_limit(self, tmp_path):
        args = type(
            "Args",
            (),
            {"max_bars": 1_000_000, "max_bars_by_timeframe": "M1=500000,M240=250000"},
        )()
        m1 = to.Job("XAUUSD", 1, tmp_path / "XAUUSD_M1.csv", "csv")
        m5 = to.Job("XAUUSD", 5, tmp_path / "XAUUSD_M5.csv", "csv")
        m240 = to.Job("XAUUSD", 240, tmp_path / "XAUUSD_M240.csv", "csv")

        assert to._job_max_bars(args, m1) == 500_000
        assert to._job_max_bars(args, m5) == 1_000_000
        assert to._job_max_bars(args, m240) == 250_000

    def test_candidate_seed_is_stable_per_job_and_variant(self, tmp_path):
        job = to.Job("XAUUSD", 5, tmp_path / "XAUUSD_M5.jsonl", "jsonl")

        seed_a = to._candidate_seed(123, "offline_candidate_base", job)
        seed_b = to._candidate_seed(123, "offline_candidate_base", job)
        seed_c = to._candidate_seed(123, "offline_candidate_fresh", job)

        assert seed_a == seed_b
        assert seed_a != seed_c

    def test_optuna_storage_path_is_per_symbol_timeframe(self, tmp_path):
        path = to._optuna_storage_path(tmp_path / "optuna", "XAU/USD+", 240)

        assert path == tmp_path / "optuna" / "offline_XAU_USD_M240.db"

    def test_optuna_objective_penalizes_cents_only_sparse_candidates(self):
        good = {
            "z_omega": 1.4,
            "val_trades": 8,
            "val_net_pnl": 12.0,
            "val_avg_pnl": 1.5,
            "val_profit_factor": 1.8,
        }
        cents = {
            "z_omega": 1.4,
            "val_trades": 2,
            "val_net_pnl": 0.2,
            "val_avg_pnl": 0.1,
            "val_profit_factor": 1.1,
        }

        assert to._optuna_objective_score(good, min_val_trades=5) > to._optuna_objective_score(
            cents,
            min_val_trades=5,
        )

    def test_optuna_prefers_accepted_candidate_over_higher_rejected_objective(self):
        accepted = {"accepted": True, "optuna_objective": 1.0}
        rejected = {"accepted": False, "optuna_objective": 10.0}

        assert to._prefer_optuna_result(accepted, rejected)
        assert not to._prefer_optuna_result(rejected, accepted)

    def test_build_optuna_trial_spec_is_bounded(self, tmp_path):
        class FakeTrial:
            number = 3

            def suggest_float(self, _name, low, high, step=None):
                return high if step is None else low

            def suggest_int(self, _name, low, _high):
                return low

            def suggest_categorical(self, _name, choices):
                return choices[0]

        args = type(
            "Args",
            (),
            {
                "n_epochs": 2,
                "train_every": 4,
                "epsilon_start": 0.4,
                "epsilon_end": 0.05,
                "penalty_scale": 1.0,
                "focused_cap_passes": 1,
                "tournament_seed": 123,
            },
        )()
        job = to.Job("XAUUSD", 5, tmp_path / "XAUUSD_M5.jsonl", "jsonl")

        spec = to._build_optuna_trial_spec(FakeTrial(), args, job, trial_number=3)

        assert spec["candidate_id"] == "optuna_t0003"
        assert spec["n_epochs"] >= args.n_epochs
        assert 1 <= spec["train_every"] <= args.train_every * 2
        assert 0.01 <= spec["epsilon_end"] < spec["epsilon_start"] <= 1.0
        assert spec["deploy_candidate"] is False


# ── discover_jobs (from train_offline) ────────────────────────────────────────


class TestDiscoverJobs:
    def test_tf_label_uses_canonical_minute_labels(self):
        assert to._tf_label(60) == "M60"
        assert to._tf_label(240) == "M240"

    def test_detect_symbol_and_tf_from_filename(self, tmp_path):
        from train_offline import discover_jobs

        f = tmp_path / "XAUUSD_M5.csv"
        f.write_text("Date & Time,Open,High,Low,Close\n")
        jobs = discover_jobs([str(tmp_path)])
        assert len(jobs) == 1
        assert jobs[0].symbol == "XAUUSD"
        assert jobs[0].timeframe_minutes == 5

    def test_detect_symbol_from_scoped_training_cache_filename(self, tmp_path):
        from train_offline import discover_jobs

        f = tmp_path / "training_cache_XAUUSD_M15.jsonl"
        f.write_text('{"entry_bars": []}\n')
        jobs = discover_jobs([str(tmp_path)])
        assert len(jobs) == 1
        assert jobs[0].symbol == "XAUUSD"
        assert jobs[0].timeframe_minutes == 15

    def test_detect_m_minutes_from_scoped_training_cache_filename(self, tmp_path):
        from train_offline import discover_jobs

        f = tmp_path / "training_cache_XAUUSD_M240.jsonl"
        f.write_text('{"entry_bars": []}\n')
        jobs = discover_jobs([str(tmp_path)])
        assert len(jobs) == 1
        assert jobs[0].symbol == "XAUUSD"
        assert jobs[0].timeframe_minutes == 240

    def test_symbol_filter(self, tmp_path):
        from train_offline import discover_jobs

        (tmp_path / "XAUUSD_M5.csv").write_text("Date & Time,Open,High,Low,Close\n")
        (tmp_path / "EURUSD_M5.csv").write_text("Date & Time,Open,High,Low,Close\n")
        jobs = discover_jobs([str(tmp_path)], symbol_filter=["XAUUSD"])
        assert all(j.symbol == "XAUUSD" for j in jobs)

    def test_tf_filter(self, tmp_path):
        from train_offline import discover_jobs

        (tmp_path / "XAUUSD_M5.csv").write_text("Date & Time,Open,High,Low,Close\n")
        (tmp_path / "XAUUSD_M60.csv").write_text("Date & Time,Open,High,Low,Close\n")
        jobs = discover_jobs([str(tmp_path)], tf_filter=["M60"])
        assert all(j.timeframe_minutes == 60 for j in jobs)

    def test_legacy_h1_input_still_maps_to_m60(self, tmp_path):
        from train_offline import discover_jobs

        (tmp_path / "XAUUSD_H1.csv").write_text("Date & Time,Open,High,Low,Close\n")
        jobs = discover_jobs([str(tmp_path)], tf_filter=["H1"])
        assert len(jobs) == 1
        assert all(j.timeframe_minutes == 60 for j in jobs)

    def test_nonexistent_path_skipped(self, tmp_path):
        from train_offline import discover_jobs

        jobs = discover_jobs(["/nonexistent/path"])
        assert jobs == []

    def test_duplicate_jsonl_caches_are_merged_for_same_bot(self, tmp_path):
        from train_offline import discover_jobs

        root = tmp_path / "training_cache_XAUUSD_M1.jsonl"
        paper_dir = tmp_path / "paper_XAUUSD_M1"
        paper_dir.mkdir()
        paper = paper_dir / "training_cache_XAUUSD_M1.jsonl"
        root.write_text('{"entry_bars": []}\n' * 45)
        paper.write_text('{"entry_bars": []}\n' * 10)

        jobs = discover_jobs([str(paper), str(root)])

        assert len(jobs) == 1
        assert jobs[0].bars_file == root
        assert set(jobs[0].source_files) == {root, paper}
        good, errors = to.preflight_check(jobs, min_rows=50)
        assert good == jobs
        assert errors == []

    def test_focused_cap_replay_selects_weekly_best_and_worst(self, tmp_path):
        cache = tmp_path / "training_cache_XAUUSD_M5.jsonl"
        now = datetime.now(UTC)
        lines = []
        for idx in range(8):
            bars = []
            for bar_idx in range(80):
                ts = now - timedelta(minutes=(idx * 100 + bar_idx) * 5)
                px = 4800.0 + idx + bar_idx * 0.01
                bars.append([ts.isoformat(), px, px + 0.5, px - 0.5, px + 0.1])
            ratio = idx - 3
            lines.append(
                json.dumps(
                    {
                        "version": 1,
                        "ts_recorded": (now - timedelta(hours=idx)).isoformat(),
                        "symbol": "XAUUSD",
                        "timeframe_minutes": 5,
                        "pnl_pts": float(ratio),
                        "mfe": 1.0,
                        "entry_bars": bars,
                        "exit_bars": bars[-5:],
                    },
                ),
            )
        cache.write_text("\n".join(lines) + "\n")

        windows = to._load_focused_cap_replay_windows([str(cache)], "XAUUSD", 5, 7.0, 2)

        assert len(windows) == 4
        assert all(len(window) >= 80 for window in windows)


# ── OfflineTrainer new methods ────────────────────────────────────────────────


def _make_bars(n: int = 200) -> list:
    t0 = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
    bars = []
    for i in range(n):
        t = t0 + timedelta(minutes=i * 5)
        px = 90000.0 + i * 1.5
        bars.append((t, px, px + 5, px - 5, px + 1, 0.5))
    return bars


class TestEvaluateRuntimeCheckpoint:
    def test_returns_not_loaded_when_checkpoint_dir_empty(self, tmp_path):
        bars = _make_bars(200)
        trainer = OfflineTrainer(
            symbol="XAUUSD",
            timeframe_minutes=5,
            bars=bars,
            checkpoint_dir=str(tmp_path / "ckpt"),
            train_split=0.8,
        )
        score, trades, loaded = trainer.evaluate_runtime_checkpoint(tmp_path / "empty_dir")
        assert loaded is False
        assert score == pytest.approx(0.0)
        assert trades == 0

    def test_returns_not_loaded_when_no_val_bars(self, tmp_path):
        bars = _make_bars(10)
        trainer = OfflineTrainer(
            symbol="XAUUSD",
            timeframe_minutes=5,
            bars=bars,
            checkpoint_dir=str(tmp_path / "ckpt"),
            train_split=1.0,  # 100% train → empty val fold
        )
        score, _, loaded = trainer.evaluate_runtime_checkpoint(tmp_path)
        assert loaded is False
        assert score == pytest.approx(0.0)


class TestRunFocusedReplay:
    def test_no_op_when_windows_empty(self, tmp_path):
        bars = _make_bars(200)
        trainer = OfflineTrainer(
            symbol="XAUUSD",
            timeframe_minutes=5,
            bars=bars,
            checkpoint_dir=str(tmp_path),
            focused_replay_windows=[],
            focused_replay_passes=3,
        )
        from unittest.mock import MagicMock

        steps, trades = trainer._run_focused_replay(MagicMock(), "test")
        assert steps == 0
        assert trades == 0

    def test_no_op_when_passes_zero(self, tmp_path):
        bars = _make_bars(200)
        trainer = OfflineTrainer(
            symbol="XAUUSD",
            timeframe_minutes=5,
            bars=bars,
            checkpoint_dir=str(tmp_path),
            focused_replay_windows=[bars[:90]],
            focused_replay_passes=0,
        )
        from unittest.mock import MagicMock

        steps, trades = trainer._run_focused_replay(MagicMock(), "test")
        assert steps == 0
        assert trades == 0

    def test_no_op_when_window_too_short(self, tmp_path):
        bars = _make_bars(200)
        trainer = OfflineTrainer(
            symbol="XAUUSD",
            timeframe_minutes=5,
            bars=bars,
            checkpoint_dir=str(tmp_path),
            focused_replay_windows=[bars[:10]],  # fewer than MIN_BARS_FOR_ENTRY=80
            focused_replay_passes=2,
        )
        from unittest.mock import MagicMock

        steps, trades = trainer._run_focused_replay(MagicMock(), "test")
        assert steps == 0
        assert trades == 0
