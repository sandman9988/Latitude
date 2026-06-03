"""Telemetry and snapshot I/O for :class:`TFAgent`.

Extracted verbatim from ``openapi_hub`` as a behaviour-preserving mixin.
All methods write state to disk or push metrics — none mutate trading state.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np

from src.persistence.json_io import write_json_async

LOG = logging.getLogger(__name__)

_CTRL_CB_RESET = "circuit_breaker_reset.json"


class TFAgentTelemetryMixin:
    """Snapshot writers, CB persistence, and production metrics flush."""

    def _write_telemetry(self) -> None:
        try:
            self._write_paper_stats()
            self._write_current_position()
            self._write_training_stats()
            self._write_risk_metrics()
            self._save_cb_state()
            self._flush_production_metrics()
        except Exception as e:
            LOG.debug("[%s %s] telemetry write error: %s", self.symbol, self.tf_label, e)
        self._check_cb_reset()

    def _save_cb_state(self) -> None:
        if self.circuit_breakers is None:
            return
        try:
            self.circuit_breakers.save_state(str(self.data_dir / "circuit_breakers.json"))
        except Exception as e:
            LOG.debug("[%s %s] cb save_state error: %s", self.symbol, self.tf_label, e)

    def _check_cb_reset(self) -> None:
        if self.circuit_breakers is None:
            return
        for _reset_path in (
            self.data_dir / _CTRL_CB_RESET,
            Path("data") / _CTRL_CB_RESET,
        ):
            if not _reset_path.exists():
                continue
            try:
                _reset_path.unlink(missing_ok=True)
                self.circuit_breakers.reset_all()
                LOG.info("[%s %s] Circuit breakers reset via HUD request", self.symbol, self.tf_label)
            except Exception as e:
                LOG.debug("[%s %s] cb reset error: %s", self.symbol, self.tf_label, e)
            break

    def _build_reward_shaping_block(self) -> dict:
        try:
            cs = self.reward_shaper.component_stats
            components: dict = {}
            for name, d in cs.items():
                count = int(d.get("count", 0) or 0)
                total = float(d.get("sum", 0.0) or 0.0)
                components[name] = {"count": count, "sum": total, "avg": (total / count) if count > 0 else 0.0}
            stats = self.reward_shaper.get_statistics()
            return {
                "total_rewards_calculated": stats.get("total_rewards_calculated", 0),
                "weights": stats.get("weights", {}),
                "parameters": stats.get("parameters", {}),
                "components": components,
            }
        except Exception:
            return {}

    def _compute_trade_log_metrics(self, now: dt.datetime) -> tuple[dict, dict, dict]:
        try:
            from datetime import timedelta

            from src.utils.metrics_calculator import (
                decision_quality,
                period_comparison,
                self_healing_metrics,
            )

            _all = self._trade_log_reader.trades
            _bot_trades = [t for t in _all if t.get("symbol") == self.symbol
                           and t.get("timeframe_minutes") == self.timeframe_minutes]
            _cut_24h = (now - timedelta(hours=24)).isoformat()
            _cut_7d = (now - timedelta(days=7)).isoformat()
            _24h = [t for t in _bot_trades if (t.get("exit_time") or "") >= _cut_24h]
            _7d = [t for t in _bot_trades if (t.get("exit_time") or "") >= _cut_7d]
            _self_heal = self_healing_metrics(_bot_trades, self.starting_equity)
            _comparison = period_comparison(_24h, _7d, self.starting_equity) if _24h and _7d else {}
            _dec_qual = decision_quality(_bot_trades)
            return _self_heal, _comparison, _dec_qual
        except Exception:
            return {}, {}, {}

    def _write_paper_stats(self) -> None:
        now = dt.datetime.now(dt.UTC)
        uptime = (now - self.start_time).total_seconds()

        ts_raw = self.policy.get_training_stats() if hasattr(self.policy, "get_training_stats") else {}
        trig = ts_raw.get("trigger") or {}
        harv = ts_raw.get("harvester") or {}

        wins = sum(1 for p in self.trades_pnl if p > 0)
        win_rate = wins / max(1, len(self.trades_pnl))
        total_pnl = sum(self.trades_pnl)

        avg_mfe = float(np.mean(list(self._rolling_mfe))) if self._rolling_mfe else 0.0
        avg_mae = float(np.mean(list(self._rolling_mae))) if self._rolling_mae else 0.0

        _self_heal, _comparison, _dec_qual = self._compute_trade_log_metrics(now)

        stats = {
            "symbol": self.symbol,
            "timeframe": self.tf_label,
            "timeframe_minutes": self.timeframe_minutes,
            "trading_mode": "paper",
            "uptime_seconds": uptime,
            "bar_count": self.bar_count,
            "quote_ok": True,
            "trade_ok": True,
            "connection_healthy": True,
            "total_reconnects": 0,
            "trigger_steps": trig.get("training_steps", 0),
            "trigger_epsilon": trig.get("epsilon", 1.0),
            "trigger_buffer": trig.get("buffer_size", 0),
            "trigger_loss": trig.get("loss", 0.0),
            "trigger_ready": trig.get("ready_to_train", False),
            "harvester_steps": harv.get("training_steps", 0),
            "harvester_beta": harv.get("beta", 0.4),
            "harvester_buffer": harv.get("buffer_size", 0),
            "harvester_loss": harv.get("loss", 0.0),
            "harvester_ready": harv.get("ready_to_train", False),
            "total_trades": self.total_trades,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "real_account_balance": self._broker_balance,
            "real_account_equity": self._broker_equity,
            "real_margin_free": self._broker_margin_free,
            "next_bar_close_utc": self.bar_builder.next_bar_close_utc(),
            "reward_shaping": self._build_reward_shaping_block(),
            "mfe_mae": {"avg_mfe": avg_mfe, "avg_mae": avg_mae, "samples": len(self._rolling_mfe)},
            "self_healing": _self_heal,
            "period_comparison": _comparison,
            "decision_quality": _dec_qual,
            "updated_at": now.isoformat(),
        }
        shared = Path("data")
        shared.mkdir(exist_ok=True)
        write_json_async(shared / f"paper_stats_{self.symbol}_M{self.timeframe_minutes}.json", stats)
        write_json_async(self.data_dir / "paper_stats.json", stats)

    def _write_current_position(self) -> None:
        now = dt.datetime.now(dt.UTC)
        mid = self.last_mid
        pos = self.position

        if pos is not None:
            direction = pos["direction"]
            entry_price = pos["entry_price"]
            unrealized = (mid - entry_price) * direction * self.qty * self.contract_size
            pos_metrics = self.policy.get_position_metrics() if hasattr(self.policy, "get_position_metrics") else {}
            mfe = float(pos_metrics.get("mfe", 0.0) or 0.0)
            mae = float(pos_metrics.get("mae", 0.0) or 0.0)
            data = {
                "symbol": self.symbol,
                "timeframe": self.tf_label,
                "timeframe_minutes": self.timeframe_minutes,
                "direction": "LONG" if direction == 1 else "SHORT",
                "position": direction,
                "entry_price": entry_price,
                "current_price": mid,
                "unrealized_pnl": unrealized,
                "mfe": mfe,
                "mae": mae,
                "qty": self.qty,
                "equity": self.equity + unrealized,
                "entry_time": pos["entry_time"].isoformat() if pos["entry_time"] else None,
                "updated_at": now.isoformat(),
            }
        else:
            data = {
                "symbol": self.symbol,
                "timeframe": self.tf_label,
                "timeframe_minutes": self.timeframe_minutes,
                "direction": "FLAT",
                "position": 0,
                "entry_price": 0.0,
                "current_price": mid,
                "unrealized_pnl": 0.0,
                "mfe": 0.0,
                "mae": 0.0,
                "qty": 0.0,
                "equity": self.equity,
                "entry_time": None,
                "updated_at": now.isoformat(),
            }

        shared = Path("data")
        shared.mkdir(exist_ok=True)
        write_json_async(shared / f"current_position_{self.symbol}_M{self.timeframe_minutes}.json", data)
        write_json_async(self.data_dir / "current_position.json", data)

    def _write_training_stats(self) -> None:
        ts_raw = self.policy.get_training_stats() if hasattr(self.policy, "get_training_stats") else {}
        trig = ts_raw.get("trigger") or {}
        harv = ts_raw.get("harvester") or {}
        last_train = trig.get("last_training_time") or harv.get("last_training_time") or "Never"
        stats = {
            "symbol": self.symbol,
            "timeframe": self.tf_label,
            "timeframe_minutes": self.timeframe_minutes,
            "trading_mode": "paper",
            "trigger_buffer_size": trig.get("buffer_size", 0),
            "harvester_buffer_size": harv.get("buffer_size", 0),
            "trigger_total_added": trig.get("total_added", 0),
            "harvester_total_added": harv.get("total_added", 0),
            "trigger_training_steps": trig.get("training_steps", 0),
            "harvester_training_steps": harv.get("training_steps", 0),
            "trigger_ready": trig.get("ready_to_train", False),
            "harvester_ready": harv.get("ready_to_train", False),
            "last_training_time": last_train,
            "trigger_loss": trig.get("loss", 0.0),
            "harvester_loss": harv.get("loss", 0.0),
            "trigger_tau": self._last_trigger_tau or trig.get("tau", 0.005),
            "harvester_tau": self._last_harvester_tau or harv.get("tau", 0.005),
            "trigger_grad_norm": self._last_trigger_grad_norm,
            "harvester_grad_norm": self._last_harvester_grad_norm,
            "trigger_epsilon": trig.get("epsilon", 1.0),
            "harvester_beta": harv.get("beta", 0.4),
            "trigger_epsilon_regime_factor": trig.get("epsilon_regime_factor", 1.0),
            "trigger_confidence": self._last_trigger_conf,
            "harvester_confidence": self._last_harvester_conf,
            "trigger_runway_cal_total_samples": trig.get("runway_cal_total_samples", 0),
            "trigger_runway_cal_active_buckets": trig.get("runway_cal_active_buckets", 0),
            "trigger_runway_predictor_reliable": trig.get("runway_predictor_reliable", False),
            "harvester_min_hold_ticks": harv.get("min_hold_ticks", 10),
            "harvester_regime_hold_mult": harv.get("regime_hold_mult", 1.0),
            "harvester_capture_decay_threshold": harv.get("capture_decay_threshold", 0.0),
            "harvester_micro_winner_giveback_pct": harv.get("micro_winner_giveback_pct", 0.0),
            "is_in_position": self.position is not None,
            "total_agents": 0,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        shared = Path("data")
        write_json_async(self.data_dir / "training_stats.json", stats)
        write_json_async(shared / f"training_stats_{self.symbol}_M{self.timeframe_minutes}.json", stats)

    def _compute_var_kurtosis(self) -> tuple[float, float]:
        if len(self.bars) < 20:
            return 0.0, 0.0
        closes = np.array([b[4] for b in list(self.bars)[-200:]], dtype=float)
        try:
            rets = np.diff(np.log(closes))
            if len(rets) < 10:
                return 0.0, 0.0
            var_95 = float(abs(np.percentile(rets, 5)))
            mu, sigma = np.mean(rets), np.std(rets)
            kurt = float(np.mean(((rets - mu) / sigma) ** 4) - 3.0) if sigma > 1e-10 else 0.0
            return var_95, kurt
        except Exception:
            return 0.0, 0.0

    def _log_lifecycle_events(self) -> None:
        current_regime = str(getattr(self.policy, "current_regime", "UNKNOWN") or "UNKNOWN")
        cb_tripped: list[str] = []
        if self.circuit_breakers is not None and self.circuit_breakers.is_any_tripped():
            cb_tripped = [b.name for b in self.circuit_breakers.get_tripped_breakers()]

        if current_regime != self._prev_regime:
            LOG.info("[%s %s] Regime change: %s → %s", self.symbol, self.tf_label,
                     self._prev_regime, current_regime)
            with contextlib.suppress(Exception):
                self.decision_log.log_decision(
                    agent="System",
                    decision="REGIME_CHANGE",
                    confidence=1.0,
                    context={"prev_regime": self._prev_regime, "new_regime": current_regime},
                    reasoning={},
                )
            self._prev_regime = current_regime

        if set(cb_tripped) != set(self._prev_cb_tripped):
            newly_tripped = [b for b in cb_tripped if b not in self._prev_cb_tripped]
            cleared = [b for b in self._prev_cb_tripped if b not in cb_tripped]
            LOG.info("[%s %s] CB state change: tripped=%s cleared=%s",
                     self.symbol, self.tf_label, newly_tripped, cleared)
            with contextlib.suppress(Exception):
                self.decision_log.log_decision(
                    agent="System",
                    decision="CIRCUIT_BREAKER" if newly_tripped else "CB_CLEARED",
                    confidence=1.0,
                    context={
                        "newly_tripped": newly_tripped,
                        "cleared": cleared,
                        "active_breakers": cb_tripped,
                    },
                    reasoning={},
                )
            self._prev_cb_tripped = cb_tripped

    def _write_risk_metrics(self) -> None:
        self._log_lifecycle_events()
        regime = str(getattr(self.policy, "current_regime", "UNKNOWN") or "UNKNOWN")
        zeta = float(getattr(self.policy, "current_zeta", 1.0) or 1.0)
        realized_vol = self._realized_vol()
        depth_ratio = self._depth_ratio()
        ts_raw = self.policy.get_training_stats() if hasattr(self.policy, "get_training_stats") else {}
        trig = ts_raw.get("trigger") or {}
        runway = float(trig.get("last_predicted_runway_net", 0.0) or 0.0)

        kurtosis_threshold = self._active_kurtosis_threshold()
        var_95, kurtosis = self._last_var_95, self._last_kurtosis
        cb_status = "INACTIVE"
        cb_tripped_names: list[str] = []
        cb_enabled = self.circuit_breakers is not None
        if cb_enabled and self.circuit_breakers.is_any_tripped():
            cb_status = "ACTIVE"
            cb_tripped_names = [b.name for b in self.circuit_breakers.get_tripped_breakers()]

        rs_vol_s = self._compute_rs_vol(10)
        rs_vol_l = self._compute_rs_vol(50)
        rs_vol_ratio = (rs_vol_s / rs_vol_l) if rs_vol_l > 0 else 1.0

        metrics = {
            "symbol": self.symbol,
            "timeframe": self.tf_label,
            "timeframe_minutes": self.timeframe_minutes,
            "circuit_breaker": cb_status,
            "circuit_breaker_enabled": cb_enabled,
            "circuit_breaker_tripped": cb_tripped_names,
            "kurtosis_gate_active": kurtosis > kurtosis_threshold,
            "kurtosis": kurtosis,
            "kurtosis_threshold": kurtosis_threshold,
            "depth_gate_active": (
                getattr(self.friction_calc, "depth_buffer", 0.0) > 0
                and self._last_depth_bid > 0 and self._last_depth_ask > 0
                and min(self._last_depth_bid, self._last_depth_ask)
                    < getattr(self.friction_calc, "depth_buffer", 0.0)
            ),
            "depth_floor": getattr(self.friction_calc, "depth_buffer", 0.0),
            "var": var_95,
            "realized_vol": realized_vol,
            "rs_vol_short": rs_vol_s,
            "rs_vol_long": rs_vol_l,
            "rs_vol_ratio": rs_vol_ratio,
            "regime": regime,
            "regime_zeta": zeta,
            "feasibility": zeta,
            "runway": runway,
            "path_geometry": self.path_geometry.last,
            "spread": self.last_half_spread * 2.0,
            "imbalance": self._entry_imbalance,
            "depth_bid": self._last_depth_bid,
            "depth_ask": self._last_depth_ask,
            "has_real_sizes": self._has_real_sizes,
            "depth_ratio": depth_ratio,
            "vpin": self._vpin_z,
            "vpin_zscore": self._vpin_z,
            "vpin_threshold": float(self._param_manager.get(
                self.symbol, "vpin_z_threshold",
                timeframe=self.tf_label, broker="default", default=2.5) or 2.5),
            "vol_cap": float(self._param_manager.get(
                self.symbol, "vol_cap", timeframe=self.tf_label, broker="default", default=0.05) or 0.05),
            "runway_delta_ema": self._runway_delta_ema,
            "runway_accuracy_ema": self._runway_accuracy_ema,
            "conf_calib_err_ema": self._conf_calib_err_ema,
            "entry_conf_dynamic_floor": self._entry_conf_dynamic_floor,
            "exit_conf_dynamic_floor": self._exit_conf_dynamic_floor,
            "win_rate_ema": self._win_rate_ema,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        shared = Path("data")
        write_json_async(shared / f"risk_metrics_{self.symbol}_M{self.timeframe_minutes}.json", metrics)

    def _flush_production_metrics(self) -> None:
        wins = sum(1 for p in self.trades_pnl if p > 0)
        total_pnl = sum(self.trades_pnl)
        win_rate = wins / max(1, len(self.trades_pnl))
        drawdown_current = max(0.0, (self.starting_equity - self.equity) / max(abs(self.starting_equity), 1.0))
        drawdown_max = max(0.0, 1.0 - min((self.equity / self.starting_equity), 1.0)) if self.trades_pnl else 0.0
        cb_tripped_names: list[str] = []
        if self.circuit_breakers is not None and self.circuit_breakers.is_any_tripped():
            cb_tripped_names = [b.name for b in self.circuit_breakers.get_tripped_breakers()]
        mins_since_trade = (
            (time.time() - self._last_trade_close_ts) / 60.0
            if self._last_trade_close_ts is not None else 0.0
        )
        regime = str(getattr(self.policy, "current_regime", "UNKNOWN"))
        try:
            self.prod_monitor.update_metrics(
                symbol=self.symbol,
                timeframe=self.tf_label,
                timeframe_minutes=self.timeframe_minutes,
                broker="default",
                trading_mode="paper",
                realized_pnl_day=total_pnl,
                realized_pnl_total=total_pnl,
                unrealized_pnl=0.0,
                drawdown_current=drawdown_current,
                drawdown_max=drawdown_max,
                trades_today=self.total_trades,
                trades_total=self.total_trades,
                win_rate=win_rate,
                avg_profit=float(
                    np.mean([p for p in self.trades_pnl if p > 0])
                    if any(p > 0 for p in self.trades_pnl)
                    else 0.0,
                ),
                avg_loss=float(
                    abs(np.mean([p for p in self.trades_pnl if p < 0]))
                    if any(p < 0 for p in self.trades_pnl)
                    else 0.0,
                ),
                last_trade_mins_ago=mins_since_trade,
                trigger_confidence_avg=self._last_trigger_conf,
                harvester_confidence_avg=self._last_harvester_conf,
                circuit_breakers_tripped=len(cb_tripped_names),
                circuit_breaker_names=cb_tripped_names,
                current_regime=regime,
                runway_delta_ema=self._runway_delta_ema,
                runway_accuracy_ema=self._runway_accuracy_ema,
                conf_calib_err_ema=self._conf_calib_err_ema,
            )
        except Exception as e:
            LOG.debug("[%s %s] prod_monitor error: %s", self.symbol, self.tf_label, e)
