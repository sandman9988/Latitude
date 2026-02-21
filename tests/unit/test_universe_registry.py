"""
Tests for the universe-registry additions in train_offline.py.

Design principle: no mocks for the training pipeline.
All CLI integration tests write real CSV files and execute the full
  discover_jobs → ProcessPoolExecutor → _run_job → OfflineTrainer
path so the promoted ZOmega values come from actual model training,
not fabricated return values.

Sections
--------
  TestRegisterUniverse  — unit tests for _register_universe() (file I/O only)
  TestStageOrder        — ordering guarantee for _STAGE_ORDER
  TestAutoPromoteCLI    — end-to-end integration: real CSV → real training
                          → universe.json written
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import train_offline as to   # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic bar generator
# Produces realistic OHLCV data that the OfflineTrainer can work with.
# Price is a random walk to mimic real market noise.
# N=400 gives the trainer enough bars for its warmup (80) plus meaningful
# train/val splits at the default 80/20 ratio.
# ---------------------------------------------------------------------------

def _synthetic_csv(path: Path, n: int = 400, seed: int = 42) -> None:
    """Write N synthetic XAUUSD-style bars to *path*."""
    rng = np.random.default_rng(seed)
    t0  = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    price = 2000.0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date & Time", "Open", "High", "Low", "Close", "Volume"])
        for i in range(n):
            ts  = t0 + timedelta(hours=4 * i)          # H4 bars
            ret = rng.normal(0, 0.003) * price          # ~0.3 % move per bar
            o   = round(price, 2)
            c   = round(price + ret, 2)
            h   = round(max(o, c) + abs(rng.normal(0, 0.001) * price), 2)
            lo  = round(min(o, c) - abs(rng.normal(0, 0.001) * price), 2)
            vol = int(rng.integers(100, 1000))
            w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), o, h, lo, c, vol])
            price = c  # random walk: next open = this close


def _read_universe(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_universe(path: Path, instruments: dict) -> None:
    path.write_text(json.dumps({"version": 1, "instruments": instruments}))


# ---------------------------------------------------------------------------
# _register_universe  (pure file-I/O unit tests — no training)
# ---------------------------------------------------------------------------

class TestRegisterUniverse:

    def test_creates_file_when_absent(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)

        to._register_universe("EURUSD", 60, z_omega=2.5, weights_path="data/checkpoints/best/EURUSD_trigger.npz")

        assert uni.exists()
        inst = _read_universe(uni)["instruments"]["EURUSD"]
        assert inst["stage"] == "PAPER"
        assert inst["timeframe_minutes"] == 60
        assert inst["z_omega"] == pytest.approx(2.5)
        assert inst["paper_pid"] is None

    def test_promotes_from_offline_training(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _write_universe(uni, {"XAUUSD": {"stage": "OFFLINE_TRAINING", "timeframe_minutes": 240}})
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)

        to._register_universe("XAUUSD", 240, z_omega=3.0, weights_path="")

        assert _read_universe(uni)["instruments"]["XAUUSD"]["stage"] == "PAPER"

    def test_does_not_demote_from_micro(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _write_universe(uni, {"BTCUSD": {"stage": "MICRO", "timeframe_minutes": 15, "z_omega": 4.0}})
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)

        to._register_universe("BTCUSD", 15, z_omega=1.0, weights_path="")

        assert _read_universe(uni)["instruments"]["BTCUSD"]["stage"] == "MICRO"

    def test_does_not_demote_from_live(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _write_universe(uni, {"EURUSD": {"stage": "LIVE", "timeframe_minutes": 5}})
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)

        to._register_universe("EURUSD", 5, z_omega=1.5, weights_path="")

        assert _read_universe(uni)["instruments"]["EURUSD"]["stage"] == "LIVE"

    def test_does_not_demote_from_paper(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        _write_universe(uni, {"GBPUSD": {"stage": "PAPER", "z_omega": 5.0, "timeframe_minutes": 30}})
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)

        to._register_universe("GBPUSD", 30, z_omega=0.5, weights_path="")

        assert _read_universe(uni)["instruments"]["GBPUSD"]["stage"] == "PAPER"
        # z_omega must NOT be overwritten
        assert _read_universe(uni)["instruments"]["GBPUSD"]["z_omega"] == pytest.approx(5.0)

    def test_atomic_write_leaves_no_tmp_file(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)

        to._register_universe("USDJPY", 1440, z_omega=1.1, weights_path="")

        assert not uni.with_suffix(".tmp").exists()
        assert uni.exists()

    def test_promoted_at_timestamp_present(self, tmp_path, monkeypatch):
        uni = tmp_path / "universe.json"
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)

        to._register_universe("AUDCAD", 60, z_omega=2.0, weights_path="")

        inst = _read_universe(uni)["instruments"]["AUDCAD"]
        assert "promoted_at" in inst
        # Should be a valid ISO timestamp
        datetime.fromisoformat(inst["promoted_at"])


# ---------------------------------------------------------------------------
# _STAGE_ORDER correctness
# ---------------------------------------------------------------------------

class TestStageOrder:

    def test_pipeline_order(self):
        s = to._STAGE_ORDER
        assert s.index("UNTRAINED")         < s.index("OFFLINE_TRAINING")
        assert s.index("OFFLINE_TRAINING")  < s.index("PAPER")
        assert s.index("PAPER")             < s.index("MICRO")
        assert s.index("MICRO")             < s.index("LIVE")

    def test_all_five_stages_present(self):
        for stage in ("UNTRAINED", "OFFLINE_TRAINING", "PAPER", "MICRO", "LIVE"):
            assert stage in to._STAGE_ORDER


# ---------------------------------------------------------------------------
# CLI integration — REAL training, no _run_job mock
#
# Each test writes a synthetic CSV, calls train_offline.main(), and checks the
# resulting universe.json.  The training is genuine: OfflineTrainer walks every
# bar, builds experiences in real time, and runs gradient updates.  ZOmega is
# whatever the model actually scores on the validation fold.
#
# --paper-threshold 0.0  → always promotes (any ZOmega ≥ 0.0)
# --paper-threshold 9999  → never promotes (ZOmega cannot reach 9999)
# --workers 1             → single worker keeps wall-clock time manageable
# ---------------------------------------------------------------------------

class TestAutoPromoteCLI:

    def test_real_training_promotes_to_paper(self, tmp_path, monkeypatch):
        """
        Full end-to-end: generate CSV → run actual DDQN training →
        assert XAUUSD promoted to PAPER stage in universe.json.
        """
        uni = tmp_path / "universe.json"
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)
        monkeypatch.setattr(to, "_STATUS_PATH", tmp_path / "status.json")

        csv_file = tmp_path / "XAUUSD_H4.csv"
        _synthetic_csv(csv_file, n=400, seed=1)

        ret = to.main([
            str(csv_file),
            "--checkpoint-dir", str(tmp_path / "ckpt"),
            "--auto-promote",
            "--paper-threshold", "0.0",   # promote regardless of ZOmega
            "--workers", "1",
        ])

        assert ret == 0
        assert uni.exists(), "universe.json was not created"
        data = _read_universe(uni)
        assert "XAUUSD" in data["instruments"], "XAUUSD not registered"
        assert data["instruments"]["XAUUSD"]["stage"] == "PAPER"
        zo = data["instruments"]["XAUUSD"]["z_omega"]
        assert isinstance(zo, float), "z_omega must be a float"

    def test_real_training_skips_below_threshold(self, tmp_path, monkeypatch):
        """
        Training completes, but --paper-threshold is impossibly high →
        no instrument should appear in universe.json.
        """
        uni = tmp_path / "universe.json"
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)
        monkeypatch.setattr(to, "_STATUS_PATH", tmp_path / "status.json")

        csv_file = tmp_path / "EURUSD_H1.csv"
        _synthetic_csv(csv_file, n=400, seed=2)

        to.main([
            str(csv_file),
            "--checkpoint-dir", str(tmp_path / "ckpt"),
            "--auto-promote",
            "--paper-threshold", "9999.0",  # impossible to reach on 400 bars
            "--workers", "1",
        ])

        if uni.exists():
            data = _read_universe(uni)
            assert "EURUSD" not in data.get("instruments", {}), (
                "EURUSD should not be promoted when ZOmega < 9999"
            )

    def test_real_training_without_flag_does_not_write_universe(self, tmp_path, monkeypatch):
        """
        --auto-promote is NOT passed → universe.json must not be created
        at all, regardless of training quality.
        """
        uni = tmp_path / "universe.json"
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)
        monkeypatch.setattr(to, "_STATUS_PATH", tmp_path / "status.json")

        csv_file = tmp_path / "BTCUSD_H4.csv"
        _synthetic_csv(csv_file, n=400, seed=3)

        to.main([
            str(csv_file),
            "--checkpoint-dir", str(tmp_path / "ckpt"),
            # NOTE: intentionally no --auto-promote
            "--workers", "1",
        ])

        assert not uni.exists(), "universe.json should not be created without --auto-promote"

    def test_two_symbols_both_promoted(self, tmp_path, monkeypatch):
        """
        Two separate CSV files, both trained in parallel → both appear in
        universe.json at PAPER stage.
        """
        uni = tmp_path / "universe.json"
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)
        monkeypatch.setattr(to, "_STATUS_PATH", tmp_path / "status.json")

        csv_xau = tmp_path / "XAUUSD_H4.csv"
        csv_btc = tmp_path / "BTCUSD_H4.csv"
        _synthetic_csv(csv_xau, n=400, seed=10)
        _synthetic_csv(csv_btc, n=400, seed=20)

        ret = to.main([
            str(csv_xau), str(csv_btc),
            "--checkpoint-dir", str(tmp_path / "ckpt"),
            "--auto-promote",
            "--paper-threshold", "0.0",
            "--workers", "2",
        ])

        assert ret == 0
        data = _read_universe(uni)
        assert data["instruments"]["XAUUSD"]["stage"] == "PAPER"
        assert data["instruments"]["BTCUSD"]["stage"] == "PAPER"

    def test_existing_live_instrument_not_demoted(self, tmp_path, monkeypatch):
        """
        If an instrument is already at LIVE stage, retraining it with
        --auto-promote must NOT demote it back to PAPER.
        """
        uni = tmp_path / "universe.json"
        _write_universe(uni, {"XAUUSD": {"stage": "LIVE", "timeframe_minutes": 240, "z_omega": 8.0}})
        monkeypatch.setattr(to, "_UNIVERSE_PATH", uni)
        monkeypatch.setattr(to, "_STATUS_PATH", tmp_path / "status.json")

        csv_file = tmp_path / "XAUUSD_H4.csv"
        _synthetic_csv(csv_file, n=400, seed=5)

        to.main([
            str(csv_file),
            "--checkpoint-dir", str(tmp_path / "ckpt"),
            "--auto-promote",
            "--paper-threshold", "0.0",
            "--workers", "1",
        ])

        assert _read_universe(uni)["instruments"]["XAUUSD"]["stage"] == "LIVE"
