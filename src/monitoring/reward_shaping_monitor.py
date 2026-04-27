from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.persistence.trade_log_reader import read_all_trades

if TYPE_CHECKING:
    from src.persistence.learned_parameters import LearnedParametersManager

LOG = logging.getLogger(__name__)
_CAPTURE_EFFICIENCY_FLOOR = 0.45
_QUALITY_MIN_SHORT_TRADES = 20
_QUALITY_MIN_BASELINE_TRADES = 80
_QUALITY_PF_DROP_PCT = -0.40
_QUALITY_PNL_PER_TRADE_DROP_PCT = -0.35
_QUALITY_CAPTURE_DROP_PCT = -0.15
_QUALITY_WINRATE_DROP_PCT = -0.06
_QUALITY_TRADES_PER_DAY_SPIKE_PCT = 0.20
_QUALITY_PAYOFF_DROP_PCT = -0.35
_NO_ENTRY_PRESSURE_MIN_DECISIONS = 40
_NO_ENTRY_PRESSURE_HIGH = 0.85
_NO_ENTRY_PRESSURE_EXTREME = 0.95
_NO_ENTRY_RELAX_PF_FLOOR = 1.10
_NO_ENTRY_RELAX_PAYOFF_FLOOR = 1.00
_NO_ENTRY_RELAX_EDGE_FLOOR = 0.0
_PATH_SCOPE_RE = re.compile(r"([A-Za-z0-9.-]+)_M(\d+)")


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
    def __init__(self, symbol: str, param_manager: LearnedParametersManager, **kwargs: Any) -> None:
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
            "target_trending_participation", target_trending_participation,
        )
        self.target_trending_participation = max(0.1, min(1.0, float(target_trending_participation)))

        target_mean_reverting_participation = self._env_float(
            "REWARD_MONITOR_MEAN_REVERTING_PARTICIPATION",
            default=0.20,
            min_value=0.0,
            max_value=1.0,
        )
        target_mean_reverting_participation = self._legacy_kwargs.pop(
            "target_mean_reverting_participation", target_mean_reverting_participation,
        )
        self.target_mean_reverting_participation = max(0.0, min(1.0, float(target_mean_reverting_participation)))
        self.compare_short_window_hours = self._env_int(
            "REWARD_MONITOR_COMPARE_SHORT_WINDOW_HOURS",
            default=24,
            min_value=6,
            max_value=72,
        )
        self.compare_short_window_hours = int(
            self._legacy_kwargs.pop("compare_short_window_hours", self.compare_short_window_hours),
        )
        self.compare_baseline_7d_days = self._env_int(
            "REWARD_MONITOR_COMPARE_BASELINE_7D_DAYS",
            default=7,
            min_value=2,
            max_value=30,
        )
        self.compare_baseline_7d_days = int(
            self._legacy_kwargs.pop("compare_baseline_7d_days", self.compare_baseline_7d_days),
        )
        self.compare_baseline_30d_days = self._env_int(
            "REWARD_MONITOR_COMPARE_BASELINE_30D_DAYS",
            default=30,
            min_value=7,
            max_value=120,
        )
        self.compare_baseline_30d_days = int(
            self._legacy_kwargs.pop("compare_baseline_30d_days", self.compare_baseline_30d_days),
        )
        self.apply_recommendations = bool(
            self._legacy_kwargs.pop(
                "apply_recommendations",
                self._env_bool("REWARD_MONITOR_APPLY_RECOMMENDATIONS", default=True),
            ),
        )

        data_dir = Path(os.environ.get("CTRADER_DATA_DIR", "data"))
        decision_log_path_explicit = "decision_log_path" in self._legacy_kwargs or bool(
            os.environ.get("REWARD_MONITOR_DECISION_LOG_PATH", "").strip(),
        )
        self.trade_log_path = Path(
            self._legacy_kwargs.pop(
                "trade_log_path",
                os.environ.get("REWARD_MONITOR_TRADE_LOG_PATH", str(data_dir / "trade_log.jsonl")),
            ),
        )
        self.decision_log_path = Path(
            self._legacy_kwargs.pop(
                "decision_log_path",
                os.environ.get(
                    "REWARD_MONITOR_DECISION_LOG_PATH",
                    str(data_dir / f"decision_log_{self.symbol}_M{self._timeframe_minutes()}.json"),
                ),
            ),
        )
        self._decision_log_path_explicit = decision_log_path_explicit
        self._allow_unscoped_decisions = self._env_bool(
            "REWARD_MONITOR_ALLOW_UNSCOPED_DECISIONS",
            default=False,
        )
        risk_metrics_path_explicit = "risk_metrics_path" in self._legacy_kwargs or bool(
            os.environ.get("REWARD_MONITOR_RISK_METRICS_PATH", "").strip(),
        )
        _risk_default = data_dir / f"risk_metrics_{self.symbol}_M{self._timeframe_minutes()}.json"
        if not _risk_default.exists():
            _risk_default = data_dir / "risk_metrics.json"
        self.risk_metrics_path = Path(
            self._legacy_kwargs.pop(
                "risk_metrics_path",
                os.environ.get("REWARD_MONITOR_RISK_METRICS_PATH", str(_risk_default)),
            ),
        )
        self._risk_metrics_path_explicit = risk_metrics_path_explicit
        self._allow_unscoped_risk_metrics = self._env_bool(
            "REWARD_MONITOR_ALLOW_UNSCOPED_RISK_METRICS",
            default=False,
        )
        # Default output path is per-bot (symbol + timeframe keyed) so that
        # concurrent bots do not clobber a shared file.  Callers can override
        # via kwargs or REWARD_MONITOR_OUTPUT_PATH env var.
        _default_output = data_dir / f"reward_shaping_monitor_{self.symbol}_{self.timeframe}.json"
        self.output_path = Path(
            self._legacy_kwargs.pop("output_path", os.environ.get("REWARD_MONITOR_OUTPUT_PATH", str(_default_output))),
        )
        self.last_run_ts = 0.0
        self._last_decision_log_source: str | None = None
        if self._legacy_kwargs:
            LOG.warning("[REWARD_MONITOR] Ignoring unsupported init kwargs: %s", sorted(self._legacy_kwargs.keys()))

    @staticmethod
    def _symbol_token(value: str | None) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())

    def _timeframe_minutes(self) -> int | None:
        text = str(self.timeframe or "").strip().upper()
        if not text:
            return None
        if text.startswith("M") and text[1:].isdigit():
            return int(text[1:])
        if text.startswith("H") and text[1:].isdigit():
            return int(text[1:]) * 60
        if text == "D1":
            return 1440
        if text == "W1":
            return 10080
        return None

    def _scope(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timeframe_minutes": self._timeframe_minutes(),
            "broker": self.broker,
        }

    def _path_has_any_scope(self, path: Path | str | None) -> bool:
        if path is None:
            return False
        return any(_PATH_SCOPE_RE.search(part) for part in Path(str(path)).parts)

    def _path_matches_scope(self, path: Path | str | None) -> bool:
        if path is None:
            return False
        target_tf = self._timeframe_minutes()
        if target_tf is None:
            return False
        target_symbol = self._symbol_token(self.symbol)
        for part in Path(str(path)).parts:
            for match in _PATH_SCOPE_RE.finditer(part):
                sym, tf = match.groups()
                try:
                    tf_minutes = int(tf)
                except (TypeError, ValueError):
                    continue
                if tf_minutes == target_tf and self._symbol_token(sym) == target_symbol:
                    return True
        return False

    def _extract_payload_scope(self, payload: dict) -> tuple[str, int]:
        details = payload.get("details", {}) if isinstance(payload.get("details"), dict) else {}
        context = payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}
        scope = payload.get("scope", {}) if isinstance(payload.get("scope"), dict) else {}
        sym = str(
            payload.get("symbol") or details.get("symbol") or context.get("symbol") or scope.get("symbol") or "",
        ).upper()

        raw_tfm = (
            payload.get("timeframe_minutes")
            or details.get("timeframe_minutes")
            or context.get("timeframe_minutes")
            or scope.get("timeframe_minutes")
        )
        try:
            tfm = int(raw_tfm or 0)
        except (TypeError, ValueError):
            tfm = 0
        if tfm <= 0:
            tf_label = (
                str(
                    payload.get("timeframe")
                    or details.get("timeframe")
                    or context.get("timeframe")
                    or scope.get("timeframe")
                    or "",
                )
                .strip()
                .upper()
            )
            tfm = self._timeframe_label_to_minutes(tf_label)
        return sym, tfm

    @staticmethod
    def _timeframe_label_to_minutes(tf_label: str) -> int:
        if not tf_label:
            return 0
        if tf_label.startswith("M") and tf_label[1:].isdigit():
            return int(tf_label[1:])
        if tf_label.startswith("H") and tf_label[1:].isdigit():
            return int(tf_label[1:]) * 60
        if tf_label == "D1":
            return 1440
        if tf_label == "W1":
            return 10080
        return 0

    def _payload_matches_scope(
        self,
        payload: dict,
        *,
        source_path: Path | str | None = None,
        explicit_unscoped_path: bool = False,
        allow_unscoped: bool = False,
    ) -> bool:
        sym, tfm = self._extract_payload_scope(payload)
        target_symbol = self._symbol_token(self.symbol)
        target_tf = self._timeframe_minutes()
        path_scoped = self._path_matches_scope(source_path)
        legacy_unscoped_allowed = bool(allow_unscoped or explicit_unscoped_path)

        if sym:
            if self._symbol_token(sym) != target_symbol:
                return False
        elif not (path_scoped or legacy_unscoped_allowed):
            return False

        if target_tf is None:
            return True
        if tfm > 0:
            return tfm == target_tf
        return path_scoped or legacy_unscoped_allowed

    def _trade_matches_scope(self, trade: dict) -> bool:
        sym = str(trade.get("symbol", "") or "").upper()
        if sym != self.symbol.upper():
            return False
        target_tf = self._timeframe_minutes()
        if target_tf is None:
            return True
        try:
            tfm = int(trade.get("timeframe_minutes", 0) or 0)
        except (TypeError, ValueError):
            tfm = 0
        if tfm > 0:
            return tfm == target_tf
        tf_label = str(trade.get("timeframe", "") or "").strip().upper()
        if tf_label:
            if tf_label.startswith("M") and tf_label[1:].isdigit():
                return int(tf_label[1:]) == target_tf
            if tf_label.startswith("H") and tf_label[1:].isdigit():
                return int(tf_label[1:]) * 60 == target_tf
            if tf_label == "D1":
                return target_tf == 1440
            if tf_label == "W1":
                return target_tf == 10080
        return False

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

    def _env_bool(self, key: str, default: bool) -> bool:
        raw = os.environ.get(key, "").strip().lower()
        if not raw:
            return default
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        LOG.warning("[REWARD_MONITOR] Invalid %s=%s", key, raw)
        return default

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
        all_trades = read_all_trades(self.trade_log_path)
        trades = [t for t in all_trades if self._trade_matches_scope(t)]
        hourly_trades = [t for t in trades if self._trade_in_window(t, window_start)]
        decision_stats = self._decision_stats(window_start)
        opportunity_count = int(decision_stats.get("entry_count", 0) or 0)
        regime = self._resolve_regime(current_regime)
        regime_bucket = self._normalize_regime(regime)
        trade_count = len(hourly_trades)
        avg_capture = self._avg_capture(hourly_trades)
        winner_to_loser_count = sum(1 for tr in hourly_trades if bool(tr.get("winner_to_loser", False)))
        window_comparison = self._build_window_comparison(now_dt, trades)
        suggestions = self._build_suggestions(
            regime_bucket=regime_bucket,
            trade_count=trade_count,
            opportunity_count=opportunity_count,
            avg_capture=avg_capture,
            winner_to_loser_count=winner_to_loser_count,
            window_comparison=window_comparison,
            decision_stats=decision_stats,
        )
        applied_adjustments = self._apply_suggestions(suggestions) if self.apply_recommendations else []
        payload = {
            "timestamp": now_dt.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "scope": self._scope(),
            "regime": regime_bucket,
            "window_minutes": 60,
            "trade_count": trade_count,
            "opportunity_count": opportunity_count,
            "avg_capture_efficiency": avg_capture,
            "winner_to_loser_count": winner_to_loser_count,
            "decision_stats": decision_stats,
            "target_trending_participation": self.target_trending_participation,
            "target_mean_reverting_participation": self.target_mean_reverting_participation,
            "window_comparison": window_comparison,
            "recommendations": [s.__dict__ for s in suggestions],
            "applied_adjustments": applied_adjustments,
        }
        self._atomic_write(payload)
        return payload

    def _trade_in_window(self, trade: dict, window_start_ts: float) -> bool:
        exit_ts = _parse_iso_ts(trade.get("exit_time"))
        if not exit_ts:
            return False
        return exit_ts.timestamp() >= window_start_ts

    def _count_opportunities(self, window_start_ts: float) -> int:
        return int(self._decision_stats(window_start_ts).get("entry_count", 0) or 0)

    def _decision_matches_scope(self, entry: dict) -> bool:
        """Strict per-bot decision scoping with controlled legacy fallback."""
        source_path = entry.get("_source_path")
        explicit_legacy = self._decision_log_path_explicit and not self._path_has_any_scope(self.decision_log_path)
        return self._payload_matches_scope(
            entry,
            source_path=source_path,
            explicit_unscoped_path=explicit_legacy,
            allow_unscoped=self._allow_unscoped_decisions,
        )

    def _normalize_action(self, action) -> str:
        text = str(action).strip().upper()
        if text in ("0", "NO_ENTRY", "NONE", "FLAT"):
            return "NO_ENTRY"
        if text in ("1", "LONG", "BUY"):
            return "LONG"
        if text in ("2", "SHORT", "SELL"):
            return "SHORT"
        return text

    def _decision_stats(self, window_start_ts: float) -> dict[str, float]:
        stats = {
            "total_count": 0,
            "entry_count": 0,
            "no_entry_count": 0,
            "no_entry_rate": 0.0,
            "confident_no_entry_count": 0,
            "confident_no_entry_rate": 0.0,
            "source_path": "",
        }
        entries = self._read_decision_entries()
        if self._last_decision_log_source:
            stats["source_path"] = self._last_decision_log_source
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ts = _parse_iso_ts(entry.get("timestamp"))
            if not ts or ts.timestamp() < window_start_ts or not self._decision_matches_scope(entry):
                continue
            details = entry.get("details", {}) if isinstance(entry.get("details"), dict) else {}
            reasoning = entry.get("reasoning", {}) if isinstance(entry.get("reasoning"), dict) else {}
            action = self._normalize_action(details.get("action", entry.get("action", entry.get("decision"))))
            stats["total_count"] += 1
            if action in ("LONG", "SHORT"):
                stats["entry_count"] += 1
                continue
            if action == "NO_ENTRY":
                stats["no_entry_count"] += 1
                confidence = float(details.get("confidence", entry.get("confidence", 0.0)) or 0.0)
                feasibility = float(
                    details.get("feasibility", entry.get("feasibility", reasoning.get("feasibility", 0.0))) or 0.0,
                )
                circuit_breaker = bool(
                    details.get(
                        "circuit_breaker",
                        entry.get("circuit_breaker", not bool(reasoning.get("circuit_breakers_ok", True))),
                    ),
                )
                if confidence >= 0.55 and feasibility >= 0.50 and not circuit_breaker:
                    stats["confident_no_entry_count"] += 1
        total = float(stats["total_count"])
        if total > 0.0:
            stats["no_entry_rate"] = stats["no_entry_count"] / total
            stats["confident_no_entry_rate"] = stats["confident_no_entry_count"] / total
        return stats

    def _read_decision_entries(self) -> list[dict]:
        log_path = self.decision_log_path
        self._last_decision_log_source = None
        if not log_path.exists():
            legacy_path = log_path.parent / "decision_log.json"
            if legacy_path.exists() and log_path.name.startswith("decision_log_"):
                log_path = legacy_path
            else:
                audit_path = log_path.parent / "logs" / "audit" / "decisions.jsonl"
                if audit_path.exists() and log_path.name.startswith("decision_log_"):
                    log_path = audit_path
                else:
                    return []
        if not log_path.exists():
            return []
        self._last_decision_log_source = str(log_path)
        try:
            with open(log_path, encoding="utf-8") as fh:
                raw = fh.read().strip()
            if not raw:
                return []
            if raw.startswith("["):
                payload = json.loads(raw)
                if not isinstance(payload, list):
                    return []
                rows = [row for row in payload if isinstance(row, dict)]
                for row in rows:
                    row.setdefault("_source_path", str(log_path))
                return rows
            payload: list[dict] = []
            for raw_line in raw.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    row.setdefault("_source_path", str(log_path))
                    payload.append(row)
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
                    explicit_legacy = self._risk_metrics_path_explicit and not self._path_has_any_scope(
                        self.risk_metrics_path,
                    )
                    if not self._payload_matches_scope(
                        payload,
                        source_path=self.risk_metrics_path,
                        explicit_unscoped_path=explicit_legacy,
                        allow_unscoped=self._allow_unscoped_risk_metrics,
                    ):
                        LOG.debug(
                            "[REWARD_MONITOR] Ignoring unscoped risk metrics for %s %s: %s",
                            self.symbol,
                            self.timeframe,
                            self.risk_metrics_path,
                        )
                        return "UNKNOWN"
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

    def _window_metrics(self, trades: list[dict]) -> dict[str, float]:
        count = len(trades)
        if count <= 0:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "pnl_total": 0.0,
                "pnl_per_trade": 0.0,
                "edge_per_trade": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "payoff_ratio": 0.0,
                "avg_capture_efficiency": 0.0,
            }
        pnl_values = [float(t.get("pnl", 0.0) or 0.0) for t in trades]
        gross_profit = sum(v for v in pnl_values if v > 0.0)
        gross_loss = sum(v for v in pnl_values if v < 0.0)
        if gross_loss < 0.0:
            profit_factor = gross_profit / abs(gross_loss)
        elif gross_profit > 0.0:
            # JSON payloads are written with allow_nan=False, so keep finite.
            profit_factor = 999.0
        else:
            profit_factor = 0.0
        wins = sum(1 for v in pnl_values if v > 0.0)
        win_values = [v for v in pnl_values if v > 0.0]
        loss_values = [abs(v) for v in pnl_values if v < 0.0]
        avg_win = sum(win_values) / len(win_values) if win_values else 0.0
        avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0.0
        if avg_loss > 0.0:
            payoff_ratio = avg_win / avg_loss
        elif avg_win > 0.0:
            payoff_ratio = 999.0
        else:
            payoff_ratio = 0.0
        pnl_per_trade = sum(pnl_values) / count
        avg_capture = self._avg_capture(trades)
        return {
            "trades": float(count),
            "win_rate": wins / count,
            "pnl_total": sum(pnl_values),
            "pnl_per_trade": pnl_per_trade,
            "edge_per_trade": pnl_per_trade,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": payoff_ratio,
            "avg_capture_efficiency": avg_capture,
        }

    @staticmethod
    def _relative_change(current: float, baseline: float) -> float | None:
        if baseline == 0.0:
            return None
        return (current - baseline) / abs(baseline)

    def _build_window_comparison(self, now_dt: datetime, trades: list[dict]) -> dict:
        short_start = now_dt.timestamp() - (self.compare_short_window_hours * 3600)
        b7_start = now_dt.timestamp() - (self.compare_baseline_7d_days * 86400)
        b30_start = now_dt.timestamp() - (self.compare_baseline_30d_days * 86400)
        short_trades = [t for t in trades if self._trade_in_window(t, short_start)]
        base7_trades = [t for t in trades if self._trade_in_window(t, b7_start)]
        base30_trades = [t for t in trades if self._trade_in_window(t, b30_start)]
        short_m = self._window_metrics(short_trades)
        base7_m = self._window_metrics(base7_trades)
        base30_m = self._window_metrics(base30_trades)

        def _delta_map(base: dict[str, float]) -> dict[str, float | None]:
            return {
                k: self._relative_change(float(short_m.get(k, 0.0)), float(base.get(k, 0.0)))
                for k in (
                    "win_rate",
                    "pnl_per_trade",
                    "edge_per_trade",
                    "profit_factor",
                    "payoff_ratio",
                    "avg_capture_efficiency",
                )
            }

        return {
            "short_window_hours": self.compare_short_window_hours,
            "baseline_7d_days": self.compare_baseline_7d_days,
            "baseline_30d_days": self.compare_baseline_30d_days,
            "short_window": short_m,
            "baseline_7d": base7_m,
            "baseline_30d": base30_m,
            "delta_vs_7d": _delta_map(base7_m),
            "delta_vs_30d": _delta_map(base30_m),
        }

    def _build_suggestions(
        self,
        regime_bucket: str,
        trade_count: int,
        opportunity_count: int,
        avg_capture: float,
        winner_to_loser_count: int,
        window_comparison: dict | None = None,
        decision_stats: dict | None = None,
    ) -> list[MonitorSuggestion]:
        out: list[MonitorSuggestion] = []
        if regime_bucket == "TRENDING":
            threshold = (
                max(1, math.floor(opportunity_count * self.target_trending_participation))
                if opportunity_count > 0
                else 1
            )
            if trade_count < threshold:
                out.append(self._suggest("entry_confidence_threshold", -0.03, "raise_trade_participation_trending"))
                out.append(self._suggest("feasibility_threshold", -0.03, "reduce_entry_friction_trending"))
        elif regime_bucket == "RANGING":
            if trade_count > 0:
                out.append(self._suggest("entry_confidence_threshold", 0.05, "suppress_entries_in_ranging"))
                out.append(self._suggest("feasibility_threshold", 0.05, "enforce_zero_trades_ranging"))
        elif regime_bucket == "MEAN_REVERTING":
            mr_threshold = (
                max(1, math.floor(opportunity_count * self.target_mean_reverting_participation))
                if opportunity_count > 0
                else 1
            )
            if trade_count > mr_threshold:
                out.append(self._suggest("entry_confidence_threshold", 0.03, "increase_selectivity_mean_reverting"))
                out.append(self._suggest("feasibility_threshold", 0.03, "tighten_participation_mean_reverting"))

        if trade_count > 0 and avg_capture < _CAPTURE_EFFICIENCY_FLOOR:
            out.append(self._suggest("reward_weight_capture", 0.10, "improve_capture_efficiency"))
            out.append(self._suggest("reward_weight_opportunity", 0.05, "penalize_missed_runway"))

        if winner_to_loser_count > 0:
            out.append(self._suggest("reward_weight_wtl", 0.10, "penalize_winner_to_loser_paths"))

        out.extend(self._build_quality_guard_suggestions(window_comparison))
        out.extend(self._build_no_entry_pressure_suggestions(decision_stats, window_comparison))

        unique: dict[str, MonitorSuggestion] = {}
        for item in out:
            existing = unique.get(item.parameter)
            if existing is None:
                unique[item.parameter] = item
                continue
            if abs(item.suggested - item.current) >= abs(existing.suggested - existing.current):
                unique[item.parameter] = item
        return list(unique.values())

    def _quality_is_collapsing(self, window_comparison: dict | None) -> bool:
        if not isinstance(window_comparison, dict):
            return False
        short = window_comparison.get("short_window", {}) or {}
        b7 = window_comparison.get("baseline_7d", {}) or {}
        d7 = window_comparison.get("delta_vs_7d", {}) or {}
        short_trades = int(short.get("trades", 0) or 0)
        base_trades = int(b7.get("trades", 0) or 0)
        if short_trades < _QUALITY_MIN_SHORT_TRADES or base_trades < _QUALITY_MIN_BASELINE_TRADES:
            return False
        d_pf = d7.get("profit_factor")
        d_edge = d7.get("edge_per_trade", d7.get("pnl_per_trade"))
        d_payoff = d7.get("payoff_ratio")
        return any(
            isinstance(v, (int, float)) and v <= limit
            for v, limit in (
                (d_pf, _QUALITY_PF_DROP_PCT),
                (d_edge, _QUALITY_PNL_PER_TRADE_DROP_PCT),
                (d_payoff, _QUALITY_PAYOFF_DROP_PCT),
            )
        )

    def _build_no_entry_pressure_suggestions(
        self,
        decision_stats: dict | None,
        window_comparison: dict | None,
    ) -> list[MonitorSuggestion]:
        if not isinstance(decision_stats, dict) or not isinstance(window_comparison, dict):
            return []
        total = int(decision_stats.get("total_count", 0) or 0)
        no_entry_rate = float(decision_stats.get("no_entry_rate", 0.0) or 0.0)
        if total < _NO_ENTRY_PRESSURE_MIN_DECISIONS or no_entry_rate < _NO_ENTRY_PRESSURE_HIGH:
            return []
        short = window_comparison.get("short_window", {}) or {}
        short_pf = float(short.get("profit_factor", 0.0) or 0.0)
        short_edge = float(short.get("edge_per_trade", short.get("pnl_per_trade", 0.0)) or 0.0)
        short_payoff = float(short.get("payoff_ratio", 0.0) or 0.0)
        if (
            self._quality_is_collapsing(window_comparison)
            or short_pf < _NO_ENTRY_RELAX_PF_FLOOR
            or short_edge <= _NO_ENTRY_RELAX_EDGE_FLOOR
            or short_payoff < _NO_ENTRY_RELAX_PAYOFF_FLOOR
        ):
            return []

        delta = -0.03 if no_entry_rate >= _NO_ENTRY_PRESSURE_EXTREME else -0.02
        reason = "reduce_no_entry_when_quality_favorable"
        return [
            self._suggest("confidence_floor", delta, reason),
            self._suggest("entry_confidence_threshold", delta, reason),
            self._suggest("feasibility_threshold", delta, reason),
        ]

    def _build_quality_guard_suggestions(self, window_comparison: dict | None) -> list[MonitorSuggestion]:
        """Add conservative quality-guard adjustments from rolling window deltas."""
        if not isinstance(window_comparison, dict):
            return []
        short = window_comparison.get("short_window", {}) or {}
        b7 = window_comparison.get("baseline_7d", {}) or {}
        d7 = window_comparison.get("delta_vs_7d", {}) or {}
        d30 = window_comparison.get("delta_vs_30d", {}) or {}
        short_trades = int(short.get("trades", 0) or 0)
        base_trades = int(b7.get("trades", 0) or 0)
        if short_trades < _QUALITY_MIN_SHORT_TRADES or base_trades < _QUALITY_MIN_BASELINE_TRADES:
            return []

        d_pf = d7.get("profit_factor")
        d_ppt = d7.get("pnl_per_trade")
        d_payoff = d7.get("payoff_ratio")
        d_cap = d7.get("avg_capture_efficiency")
        d_wr = d7.get("win_rate")
        d_tpd = self._relative_change(
            float(short.get("trades", 0.0) or 0.0),
            float(b7.get("trades", 0.0) / max(1.0, float(window_comparison.get("baseline_7d_days", 7))) or 0.0),
        )
        out: list[MonitorSuggestion] = []

        if (
            isinstance(d_pf, (int, float))
            and isinstance(d_ppt, (int, float))
            and (
                (d_pf <= _QUALITY_PF_DROP_PCT and d_ppt <= _QUALITY_PNL_PER_TRADE_DROP_PCT)
                or (
                    isinstance(d_payoff, (int, float))
                    and d_payoff <= _QUALITY_PAYOFF_DROP_PCT
                    and d_ppt <= _QUALITY_PNL_PER_TRADE_DROP_PCT
                )
                or d_ppt <= (_QUALITY_PNL_PER_TRADE_DROP_PCT - 0.20)
            )
        ):
            out.append(self._suggest("entry_confidence_threshold", 0.03, "quality_guard_pf_pnl_drop"))
            out.append(self._suggest("feasibility_threshold", 0.03, "quality_guard_pf_pnl_drop"))

        if isinstance(d_cap, (int, float)) and d_cap <= _QUALITY_CAPTURE_DROP_PCT:
            out.append(self._suggest("reward_weight_capture", 0.08, "quality_guard_capture_drop"))
            out.append(self._suggest("reward_weight_opportunity", 0.05, "quality_guard_capture_drop"))

        if (
            isinstance(d_wr, (int, float))
            and isinstance(d_tpd, (int, float))
            and d_wr <= _QUALITY_WINRATE_DROP_PCT
            and d_tpd >= _QUALITY_TRADES_PER_DAY_SPIKE_PCT
        ):
            out.append(self._suggest("entry_confidence_threshold", 0.02, "quality_guard_winrate_drop_with_overtrading"))

        # If both 7d and 30d show strong improvement, gently relax selectivity.
        if (
            isinstance(d_pf, (int, float))
            and isinstance(d_ppt, (int, float))
            and isinstance(d30.get("profit_factor"), (int, float))
            and isinstance(d30.get("pnl_per_trade"), (int, float))
            and d_pf >= 0.35
            and d_ppt >= 0.25
            and d30.get("profit_factor") >= 0.20
            and d30.get("pnl_per_trade") >= 0.15
        ):
            out.append(self._suggest("entry_confidence_threshold", -0.01, "quality_guard_sustained_improvement"))

        return out

    def _suggest(self, param_name: str, delta: float, reason: str) -> MonitorSuggestion:
        current = float(
            self.param_manager.get(
                self.symbol,
                param_name,
                timeframe=self.timeframe,
                broker=self.broker,
                default=0.0,
            ),
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

    def _apply_suggestions(self, suggestions: list[MonitorSuggestion]) -> list[dict]:
        applied: list[dict] = []
        for suggestion in suggestions:
            try:
                new_value = self.param_manager.set_value(
                    self.symbol,
                    suggestion.parameter,
                    suggestion.suggested,
                    timeframe=self.timeframe,
                    broker=self.broker,
                )
            except Exception as exc:
                LOG.warning(
                    "[REWARD_MONITOR] Failed to apply %s=%s: %s",
                    suggestion.parameter,
                    suggestion.suggested,
                    exc,
                )
                continue
            applied.append(
                {
                    "parameter": suggestion.parameter,
                    "reason": suggestion.reason,
                    "previous": suggestion.current,
                    "applied": round(float(new_value), 6),
                },
            )
        if applied:
            self.param_manager.save()
        return applied

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
