from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.persistence.learned_parameters import LearnedParametersManager
from src.persistence.trade_log_reader import read_all_trades

LOG = logging.getLogger(__name__)
_CAPTURE_EFFICIENCY_FLOOR = 0.45


def _parse_iso_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass
class MonitorSuggestion:
    parameter: str
    direction: str
    reason: str
    current: float
    suggested: float


class RewardShapingMonitor:
    def __init__(self, symbol: str, param_manager: LearnedParametersManager, **kwargs):
        self.symbol = symbol
        self.param_manager = param_manager
        self._legacy_kwargs = dict(kwargs)
        self.timeframe = str(self._legacy_kwargs.pop("timeframe", "M5"))
        self.broker = str(self._legacy_kwargs.pop("broker", "default"))

        interval_seconds = self._env_int(
            "REWARD_SHAPING_MONITOR_INTERVAL_SECONDS",
            default=3600,
            min_value=300,
            max_value=86400,
        )
        interval_seconds = self._legacy_kwargs.pop("interval_seconds", interval_seconds)
        self.interval_seconds = max(300, min(86400, int(interval_seconds)))

        target_trending_participation = self._env_float(
            "REWARD_MONITOR_TRENDING_PARTICIPATION",
            default=0.60,
            min_value=0.1,
            max_value=1.0,
        )
        target_trending_participation = self._legacy_kwargs.pop(
            "target_trending_participation", target_trending_participation
        )
        self.target_trending_participation = max(0.1, min(1.0, float(target_trending_participation)))

        target_mean_reverting_participation = self._env_float(
            "REWARD_MONITOR_MEAN_REVERTING_PARTICIPATION",
            default=0.20,
            min_value=0.0,
            max_value=1.0,
        )
        target_mean_reverting_participation = self._legacy_kwargs.pop(
            "target_mean_reverting_participation", target_mean_reverting_participation
        )
        self.target_mean_reverting_participation = max(0.0, min(1.0, float(target_mean_reverting_participation)))

        self.trade_log_path = Path(
            self._legacy_kwargs.pop("trade_log_path", os.environ.get("REWARD_MONITOR_TRADE_LOG_PATH", "data/trade_log.jsonl"))
        )
        self.decision_log_path = Path(
            self._legacy_kwargs.pop(
                "decision_log_path", os.environ.get("REWARD_MONITOR_DECISION_LOG_PATH", "data/decision_log.json")
            )
        )
        self.risk_metrics_path = Path(
            self._legacy_kwargs.pop(
                "risk_metrics_path", os.environ.get("REWARD_MONITOR_RISK_METRICS_PATH", "data/risk_metrics.json")
            )
        )
        # Default output path is per-bot (symbol + timeframe keyed) so that
        # concurrent bots do not clobber a shared file.  Callers can override
        # via kwargs or REWARD_MONITOR_OUTPUT_PATH env var.
        _default_output = f"data/reward_shaping_monitor_{self.symbol}_{self.timeframe}.json"
        self.output_path = Path(
            self._legacy_kwargs.pop("output_path", os.environ.get("REWARD_MONITOR_OUTPUT_PATH", _default_output))
        )
        self.last_run_ts = 0.0
        if self._legacy_kwargs:
            LOG.warning("[REWARD_MONITOR] Ignoring unsupported init kwargs: %s", sorted(self._legacy_kwargs.keys()))

    def _env_int(self, key: str, default: int, min_value: int, max_value: int) -> int:
        raw = os.environ.get(key, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            LOG.warning("[REWARD_MONITOR] Invalid %s=%s", key, raw)
            return default
        return max(min_value, min(max_value, value))

    def _env_float(self, key: str, default: float, min_value: float, max_value: float) -> float:
        raw = os.environ.get(key, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            LOG.warning("[REWARD_MONITOR] Invalid %s=%s", key, raw)
            return default
        return max(min_value, min(max_value, value))

    def run_if_due(self, current_regime: str = "UNKNOWN") -> dict | None:
        now_ts = time.time()
        if now_ts - self.last_run_ts < self.interval_seconds:
            return None
        result = self.run(current_regime=current_regime, now_ts=now_ts)
        self.last_run_ts = now_ts
        return result

    def run(self, current_regime: str = "UNKNOWN", now_ts: float | None = None) -> dict:
        now_ts = now_ts or time.time()
        now_dt = datetime.fromtimestamp(now_ts, tz=UTC)
        window_start = now_dt.timestamp() - 3600
        trades = read_all_trades(self.trade_log_path)
        hourly_trades = [t for t in trades if self._trade_in_window(t, window_start)]
        opportunity_count = self._count_opportunities(window_start)
        regime = self._resolve_regime(current_regime)
        regime_bucket = self._normalize_regime(regime)
        trade_count = len(hourly_trades)
        avg_capture = self._avg_capture(hourly_trades)
        winner_to_loser_count = sum(1 for tr in hourly_trades if bool(tr.get("winner_to_loser", False)))
        suggestions = self._build_suggestions(
            regime_bucket=regime_bucket,
            trade_count=trade_count,
            opportunity_count=opportunity_count,
            avg_capture=avg_capture,
            winner_to_loser_count=winner_to_loser_count,
        )
        payload = {
            "timestamp": now_dt.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "regime": regime_bucket,
            "window_minutes": 60,
            "trade_count": trade_count,
            "opportunity_count": opportunity_count,
            "avg_capture_efficiency": avg_capture,
            "winner_to_loser_count": winner_to_loser_count,
            "target_trending_participation": self.target_trending_participation,
            "target_mean_reverting_participation": self.target_mean_reverting_participation,
            "recommendations": [s.__dict__ for s in suggestions],
        }
        self._atomic_write(payload)
        return payload

    def _trade_in_window(self, trade: dict, window_start_ts: float) -> bool:
        exit_ts = _parse_iso_ts(trade.get("exit_time"))
        if not exit_ts:
            return False
        return exit_ts.timestamp() >= window_start_ts

    def _count_opportunities(self, window_start_ts: float) -> int:
        entries = self._read_decision_entries()
        total = 0
        for entry in entries:
            ts = _parse_iso_ts(entry.get("timestamp"))
            if not ts or ts.timestamp() < window_start_ts:
                continue
            details = entry.get("details", {})
            action = details.get("action")
            if action in (1, 2, "LONG", "SHORT"):
                total += 1
        return total

    def _read_decision_entries(self) -> list[dict]:
        if not self.decision_log_path.exists():
            return []
        try:
            with open(self.decision_log_path, encoding="utf-8") as fh:
                payload = json.load(fh)
            return payload if isinstance(payload, list) else []
        except (OSError, ValueError):
            return []

    def _resolve_regime(self, current_regime: str) -> str:
        if current_regime and current_regime != "UNKNOWN":
            return current_regime
        if self.risk_metrics_path.exists():
            try:
                with open(self.risk_metrics_path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                if isinstance(payload, dict):
                    return str(payload.get("regime", "UNKNOWN"))
            except (OSError, ValueError):
                pass
        return "UNKNOWN"

    def _normalize_regime(self, regime: str) -> str:
        if regime == "TRENDING":
            return "TRENDING"
        if regime == "MEAN_REVERTING":
            return "MEAN_REVERTING"
        if regime in ("TRANSITIONAL", "RANGING"):
            return "RANGING"
        return "RANGING"

    def _avg_capture(self, trades: list[dict]) -> float:
        if not trades:
            return 0.0
        captures: list[float] = []
        for trade in trades:
            if "capture_ratio" in trade:
                captures.append(max(-1.0, min(1.0, float(trade.get("capture_ratio", 0.0) or 0.0))))
                continue
            pnl = float(trade.get("pnl", 0.0) or 0.0)
            mfe = abs(float(trade.get("mfe", 0.0) or 0.0))
            denom = max(1e-9, mfe)
            captures.append(max(-1.0, min(1.0, pnl / denom)))
        return float(sum(captures) / len(captures))

    def _build_suggestions(
        self,
        regime_bucket: str,
        trade_count: int,
        opportunity_count: int,
        avg_capture: float,
        winner_to_loser_count: int,
    ) -> list[MonitorSuggestion]:
        out: list[MonitorSuggestion] = []
        if regime_bucket == "TRENDING":
            threshold = max(1, math.floor(opportunity_count * self.target_trending_participation)) if opportunity_count > 0 else 1
            if trade_count < threshold:
                out.append(self._suggest("entry_confidence_threshold", -0.03, "raise_trade_participation_trending"))
                out.append(self._suggest("feasibility_threshold", -0.03, "reduce_entry_friction_trending"))
        elif regime_bucket == "RANGING":
            if trade_count > 0:
                out.append(self._suggest("entry_confidence_threshold", 0.05, "suppress_entries_in_ranging"))
                out.append(self._suggest("feasibility_threshold", 0.05, "enforce_zero_trades_ranging"))
        elif regime_bucket == "MEAN_REVERTING":
            mr_threshold = (
                max(1, math.floor(opportunity_count * self.target_mean_reverting_participation)) if opportunity_count > 0 else 1
            )
            if trade_count > mr_threshold:
                out.append(self._suggest("entry_confidence_threshold", 0.03, "increase_selectivity_mean_reverting"))
                out.append(self._suggest("feasibility_threshold", 0.03, "tighten_participation_mean_reverting"))

        if trade_count > 0 and avg_capture < _CAPTURE_EFFICIENCY_FLOOR:
            out.append(self._suggest("reward_weight_capture", 0.10, "improve_capture_efficiency"))
            out.append(self._suggest("reward_weight_opportunity", 0.05, "penalize_missed_runway"))

        if winner_to_loser_count > 0:
            out.append(self._suggest("reward_weight_wtl", 0.10, "penalize_winner_to_loser_paths"))

        unique: dict[str, MonitorSuggestion] = {}
        for item in out:
            unique[item.parameter] = item
        return list(unique.values())

    def _suggest(self, param_name: str, delta: float, reason: str) -> MonitorSuggestion:
        current = float(
            self.param_manager.get(
                self.symbol,
                param_name,
                timeframe=self.timeframe,
                broker=self.broker,
                default=0.0,
            )
        )
        proposed = current + delta
        spec = self.param_manager.param_specs.get(param_name)
        if spec:
            proposed = max(float(spec["min"]), min(float(spec["max"]), proposed))
        direction = "increase" if proposed >= current else "decrease"
        return MonitorSuggestion(
            parameter=param_name,
            direction=direction,
            reason=reason,
            current=round(current, 6),
            suggested=round(proposed, 6),
        )

    def _atomic_write(self, payload: dict) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.output_path.parent),
            prefix=".reward_shaping_monitor_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            Path(tmp_path).replace(self.output_path)
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
