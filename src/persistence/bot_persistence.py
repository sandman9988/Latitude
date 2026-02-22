#!/usr/bin/env python3
"""
Bot Persistence Manager
Atomic, crash-safe persistence for models, stats, and state
Organized by instrument, timeframe, and session
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from src.persistence.atomic_persistence import AtomicPersistence

# File parsing constants
AGENT_INDEX_POSITION: int = 3  # Position of agent index in filename parts

LOG = logging.getLogger(__name__)


class BotPersistenceManager:
    """
    Manages persistent state for trading bot with atomic writes

    Directory structure:
        store/
            {symbol}/
                {timeframe}/
                    models/
                        trigger_agent_0.pt
                        trigger_agent_1.pt
                        harvester_agent_0.pt
                        harvester_agent_1.pt
                    stats/
                        session_{timestamp}.json
                        cumulative_stats.json
                    checkpoints/
                        checkpoint_latest.json
            cumulative/
                all_instruments.json
                all_timeframes.json
                all_sessions.json
    """

    def __init__(self, base_dir: str = "store"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Atomic persistence handler
        self.persistence = AtomicPersistence(base_dir=str(self.base_dir))

    def get_instrument_dir(self, symbol: str, timeframe: str) -> Path:
        """Get directory for specific instrument/timeframe"""
        inst_dir = self.base_dir / symbol / timeframe
        inst_dir.mkdir(parents=True, exist_ok=True)
        return inst_dir

    # ========== MODEL PERSISTENCE ==========

    def save_agent_model(  # noqa: PLR0913
        self,
        agent_type: str,  # 'trigger' or 'harvester'
        agent_idx: int,
        model_state: dict[str, Any],
        symbol: str,
        timeframe: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Save agent neural network model with atomic write

        Args:
            agent_type: 'trigger' or 'harvester'
            agent_idx: Agent index in arena (0, 1, 2, ...)
            model_state: PyTorch state_dict
            symbol: Trading symbol (e.g., 'XAUUSD')
            timeframe: Bar period (e.g., '1m', '5m')
            metadata: Optional metadata (training steps, loss, etc.)
        """
        inst_dir = self.get_instrument_dir(symbol, timeframe)
        models_dir = inst_dir / "models"
        models_dir.mkdir(exist_ok=True)

        model_file = models_dir / f"{agent_type}_agent_{agent_idx}.pt"
        meta_file = models_dir / f"{agent_type}_agent_{agent_idx}_meta.json"

        try:
            # Save model weights with atomic tmp file
            temp_file = model_file.with_suffix(".pt.tmp")
            torch.save(model_state, temp_file)
            temp_file.replace(model_file)  # Atomic rename

            # Save metadata
            if metadata is None:
                metadata = {}
            metadata.update(
                {
                    "saved_at": datetime.now(UTC).isoformat() + "Z",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "agent_type": agent_type,
                    "agent_idx": agent_idx,
                }
            )

            self.persistence.save_json(metadata, str(meta_file.relative_to(self.base_dir)))

            LOG.info(f"[PERSISTENCE] Saved {agent_type} agent {agent_idx} model: " f"{symbol}/{timeframe}")
            return True

        except Exception as e:
            LOG.error(f"[PERSISTENCE] Failed to save model: {e}")
            return False

    def load_agent_model(self, agent_type: str, agent_idx: int, symbol: str, timeframe: str) -> dict[str, Any] | None:
        """Load agent model state_dict"""
        inst_dir = self.get_instrument_dir(symbol, timeframe)
        models_dir = inst_dir / "models"
        model_file = models_dir / f"{agent_type}_agent_{agent_idx}.pt"

        if not model_file.exists():
            LOG.warning(f"[PERSISTENCE] Model not found: {agent_type} {agent_idx} " f"for {symbol}/{timeframe}")
            return None

        try:
            state_dict = torch.load(model_file, map_location="cpu")
            LOG.info(f"[PERSISTENCE] Loaded {agent_type} agent {agent_idx}: " f"{symbol}/{timeframe}")
            return state_dict
        except Exception as e:
            LOG.error(f"[PERSISTENCE] Failed to load model: {e}")
            return None

    # ========== SESSION STATS ==========

    def save_session_stats(
        self, stats: dict[str, Any], symbol: str, timeframe: str, session_id: str | None = None
    ) -> bool:
        """
        Save statistics for current trading session

        Args:
            stats: Performance metrics, trade history, etc.
            symbol: Trading symbol
            timeframe: Bar period
            session_id: Unique session identifier (default: timestamp)
        """
        if session_id is None:
            session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        inst_dir = self.get_instrument_dir(symbol, timeframe)
        stats_dir = inst_dir / "stats"
        stats_dir.mkdir(exist_ok=True)

        session_file = f"session_{session_id}.json"

        # Add metadata
        stats["_metadata"] = {
        "session_id": session_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "saved_at": datetime.now(UTC).isoformat() + "Z",
        }

        success = self.persistence.save_json(stats, str((stats_dir / session_file).relative_to(self.base_dir)))

        if success:
            LOG.info(f"[PERSISTENCE] Saved session stats: {symbol}/{timeframe}/{session_id}")

        return success

    # ========== CUMULATIVE STATS ==========

    def save_cumulative_stats(self, stats: dict[str, Any], symbol: str, timeframe: str) -> bool:
        """
        Save cumulative statistics across all sessions

        This updates running totals, averages, etc. for the instrument/timeframe
        """
        inst_dir = self.get_instrument_dir(symbol, timeframe)
        stats_dir = inst_dir / "stats"
        stats_dir.mkdir(exist_ok=True)

        cumulative_file = stats_dir / "cumulative_stats.json"

        # Load existing cumulative stats
        existing = self.persistence.load_json(str(cumulative_file.relative_to(self.base_dir)))

        if existing is None:
            existing = {
                "total_sessions": 0,
                "total_trades": 0,
                "total_pnl": 0.0,
                "total_bars": 0,
                "first_session": datetime.now(UTC).isoformat() + "Z",
                "sessions": [],
            }

        # Merge new stats
        existing["total_sessions"] += stats.get("sessions", 1)
        existing["total_trades"] += stats.get("total_trades", 0)
        existing["total_pnl"] += stats.get("total_pnl", 0.0)
        existing["total_bars"] += stats.get("bars_processed", 0)
        existing["last_update"] = datetime.now(UTC).isoformat() + "Z"

        # Append session summary
        if "session_id" in stats:
            existing["sessions"].append(
                {
                    "session_id": stats["session_id"],
                    "trades": stats.get("total_trades", 0),
                    "pnl": stats.get("total_pnl", 0.0),
                    "timestamp": datetime.now(UTC).isoformat() + "Z",
                }
            )

        # Keep only last 100 sessions in history
        existing["sessions"] = existing["sessions"][-100:]

        success = self.persistence.save_json(existing, str(cumulative_file.relative_to(self.base_dir)))

        if success:
            LOG.info(f"[PERSISTENCE] Updated cumulative stats: {symbol}/{timeframe}")

        return success

    def load_cumulative_stats(self, symbol: str, timeframe: str) -> dict[str, Any] | None:
        """Load cumulative statistics for instrument/timeframe"""
        inst_dir = self.get_instrument_dir(symbol, timeframe)
        stats_dir = inst_dir / "stats"
        cumulative_file = stats_dir / "cumulative_stats.json"

        return self.persistence.load_json(str(cumulative_file.relative_to(self.base_dir)))

    # ========== CROSS-INSTRUMENT AGGREGATION ==========

    def save_global_stats(self, all_instruments: list[dict[str, Any]]) -> bool:
        """
        Save aggregated stats across all instruments, timeframes, sessions

        Args:
            all_instruments: List of instrument stats (symbol, timeframe, metrics)
        """
        cumulative_dir = self.base_dir / "cumulative"
        cumulative_dir.mkdir(exist_ok=True)

        # Aggregate by instrument
        by_instrument = {}
        by_timeframe = {}
        total_stats = {
            "total_instruments": 0,
            "total_timeframes": 0,
            "total_trades": 0,
            "total_pnl": 0.0,
            "updated_at": datetime.now(UTC).isoformat() + "Z",
        }

        for inst in all_instruments:
            symbol = inst.get("symbol", "UNKNOWN")
            timeframe = inst.get("timeframe", "UNKNOWN")

            # By instrument
            if symbol not in by_instrument:
                by_instrument[symbol] = {
                    "symbol": symbol,
                    "timeframes": [],
                    "total_trades": 0,
                    "total_pnl": 0.0,
                }

            by_instrument[symbol]["timeframes"].append(timeframe)
            by_instrument[symbol]["total_trades"] += inst.get("total_trades", 0)
            by_instrument[symbol]["total_pnl"] += inst.get("total_pnl", 0.0)

            # By timeframe
            if timeframe not in by_timeframe:
                by_timeframe[timeframe] = {
                    "timeframe": timeframe,
                    "instruments": [],
                    "total_trades": 0,
                    "total_pnl": 0.0,
                }

            by_timeframe[timeframe]["instruments"].append(symbol)
            by_timeframe[timeframe]["total_trades"] += inst.get("total_trades", 0)
            by_timeframe[timeframe]["total_pnl"] += inst.get("total_pnl", 0.0)

            # Totals
            total_stats["total_trades"] += inst.get("total_trades", 0)
            total_stats["total_pnl"] += inst.get("total_pnl", 0.0)

        total_stats["total_instruments"] = len(by_instrument)
        total_stats["total_timeframes"] = len(by_timeframe)

        # Save each aggregation
        self.persistence.save_json({"instruments": list(by_instrument.values())}, "cumulative/all_instruments.json")

        self.persistence.save_json({"timeframes": list(by_timeframe.values())}, "cumulative/all_timeframes.json")

        self.persistence.save_json(total_stats, "cumulative/all_sessions.json")

        LOG.info(
            f"[PERSISTENCE] Saved global stats: "
            f"{total_stats['total_instruments']} instruments, "
            f"{total_stats['total_timeframes']} timeframes, "
            f"{total_stats['total_trades']} trades, "
            f"${total_stats['total_pnl']:.2f} PnL"
        )

        return True

    # ========== CHECKPOINT MANAGEMENT ==========

    def save_checkpoint(
        self, state: dict[str, Any], symbol: str, timeframe: str, checkpoint_name: str = "latest"
    ) -> bool:
        """
        Save bot state checkpoint (positions, buffers, bars, etc.)

        Args:
            state: Complete bot state
            symbol: Trading symbol
            timeframe: Bar period
            checkpoint_name: Checkpoint identifier ('latest', 'backup', etc.)
        """
        inst_dir = self.get_instrument_dir(symbol, timeframe)
        checkpoint_dir = inst_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_name}.json"

        state["_checkpoint_metadata"] = {
            "checkpoint_name": checkpoint_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "saved_at": datetime.now(UTC).isoformat() + "Z",
        }

        success = self.persistence.save_json(state, str(checkpoint_file.relative_to(self.base_dir)))

        if success:
            LOG.info(f"[PERSISTENCE] Saved checkpoint '{checkpoint_name}': " f"{symbol}/{timeframe}")

        return success

    def load_checkpoint(self, symbol: str, timeframe: str, checkpoint_name: str = "latest") -> dict[str, Any] | None:
        """Load bot state checkpoint"""
        inst_dir = self.get_instrument_dir(symbol, timeframe)
        checkpoint_dir = inst_dir / "checkpoints"
        checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_name}.json"

        if not checkpoint_file.exists():
            LOG.warning(f"[PERSISTENCE] Checkpoint not found: {checkpoint_name} " f"for {symbol}/{timeframe}")
            return None

        return self.persistence.load_json(str(checkpoint_file.relative_to(self.base_dir)))

    # ========== UTILITY ==========

    def list_saved_models(self, symbol: str, timeframe: str) -> dict[str, list[int]]:
        """List available agent models for instrument/timeframe"""
        inst_dir = self.get_instrument_dir(symbol, timeframe)
        models_dir = inst_dir / "models"

        if not models_dir.exists():
            return {"trigger": [], "harvester": []}

        trigger_models = []
        harvester_models = []

        for model_file in models_dir.glob("*_agent_*.pt"):
            name = model_file.stem
            parts = name.split("_")
            if len(parts) >= AGENT_INDEX_POSITION:
                agent_type = parts[0]  # 'trigger' or 'harvester'
                try:
                    agent_idx = int(parts[2])
                    if agent_type == "trigger":
                        trigger_models.append(agent_idx)
                    elif agent_type == "harvester":
                        harvester_models.append(agent_idx)
                except ValueError:
                    pass

        return {"trigger": sorted(trigger_models), "harvester": sorted(harvester_models)}

    def get_storage_summary(self) -> dict[str, Any]:
        """Get summary of stored data"""
        summary = {
            "base_dir": str(self.base_dir),
            "instruments": {},
            "total_models": 0,
            "total_sessions": 0,
        }

        # Scan all symbols
        for symbol_dir in self.base_dir.iterdir():
            if symbol_dir.is_dir() and symbol_dir.name != "cumulative":
                symbol = symbol_dir.name
                summary["instruments"][symbol] = {}

                # Scan timeframes
                for tf_dir in symbol_dir.iterdir():
                    if tf_dir.is_dir():
                        timeframe = tf_dir.name

                        models_dir = tf_dir / "models"
                        stats_dir = tf_dir / "stats"

                        model_count = len(list(models_dir.glob("*.pt"))) if models_dir.exists() else 0
                        session_count = len(list(stats_dir.glob("session_*.json"))) if stats_dir.exists() else 0

                        summary["instruments"][symbol][timeframe] = {
                            "models": model_count,
                            "sessions": session_count,
                        }

                        summary["total_models"] += model_count
                        summary["total_sessions"] += session_count

        return summary
