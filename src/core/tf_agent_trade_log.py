"""Trade log writing for :class:`TFAgent`.

Extracted verbatim from ``openapi_hub`` as a behaviour-preserving mixin.
Handles the 66-field trade_log.jsonl record: runway bias correction, quality
classification, and durable append via audit_logger.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.persistence.json_io import append_jsonl_durable, json_default
from src.utils.safe_math import SAFE_EPSILON, SafeMath

LOG = logging.getLogger(__name__)

_RUNWAY_BIAS_LIMIT_POINTS = 12.0
_RUNWAY_ADJUST_MIN_SCALE = 0.35
_RUNWAY_ADJUST_MAX_SCALE = 1.5


class TFAgentTradeLogMixin:
    """Durable trade log append and quality classification."""

    @staticmethod
    def _classify_trigger_quality(predicted_runway_pts: float, actual_mfe_pts: float) -> str:
        if predicted_runway_pts <= 0:
            return "N/A"
        utilization = SafeMath.safe_div(actual_mfe_pts, predicted_runway_pts, 0.0)
        if actual_mfe_pts > 0 and 0.9 <= utilization <= 1.2:
            return "EXCELLENT"
        if utilization >= 1.2:
            return "UNDERPREDICTED"
        if utilization >= 0.7:
            return "GOOD"
        return "OVERPREDICTED"

    @staticmethod
    def _classify_harvester_quality(
        pnl_usd: float,
        mfe_usd: float,
        winner_to_loser: bool,
        bars_from_mfe_to_exit: int,
    ) -> str:
        if winner_to_loser:
            return "POOR_WTL"
        if pnl_usd <= 0:
            return "STOPPED_OUT"
        if mfe_usd <= 0:
            return "N/A"
        capture = SafeMath.safe_div(pnl_usd, mfe_usd, 0.0)
        if capture >= 0.8 and bars_from_mfe_to_exit <= 2:
            return "EXCELLENT"
        if capture >= 0.6:
            return "GOOD"
        if capture >= 0.35:
            return "FAIR"
        return "POOR"

    def _write_trade_log(
        self,
        direction: int,
        entry_price: float,
        exit_price: float,
        entry_time: dt.datetime,
        exit_time: dt.datetime,
        pnl_usd: float,
        mfe: float,
        mae: float,
        quantity: float | None = None,
        pnl_pts: float = 0.0,
        trigger_reward: float = 0.0,
        capture_reward: float = 0.0,
        regime: str = "UNKNOWN",
        predicted_runway_gross: float = 0.0,
        predicted_runway_net: float = 0.0,
        was_winner_to_loser: bool = False,
        reward_wtl_net_flag: bool = False,
        entry_vpin_z: float = 0.0,
        entry_var_95: float = 0.0,
        capture_ratio: float = 0.0,
        diag_cb_active: bool = False,
        diag_cb_tripped: list | None = None,
        trade_id: str | None = None,
        ticks_held: int = 0,
        exit_regime: str = "UNKNOWN",
        exit_vol: float = 0.0,
        exit_depth_ratio: float = 0.0,
        entry_dynamic_floor: float = 0.0,
        entry_conf_margin: float = 0.0,
        win_rate_ema_at_entry: float = 0.5,
        total_trades_at_entry: int = 0,
        equity_at_entry: float = 0.0,
        conf_calib_err_at_entry: float = 0.0,
        runway_accuracy_at_entry: float = 0.0,
        reward_capture_efficiency: float = 0.0,
        reward_wtl_penalty: float = 0.0,
        reward_opportunity_cost: float = 0.0,
        reward_session_quality: float = 1.0,
        reward_harvester_total: float = 0.0,
        reward_trigger_breakdown: dict | None = None,
        reward_harvester_breakdown: dict | None = None,
        trigger_data: dict | None = None,
        close_drawdown_pct: float = 0.0,
        close_cb_size_mult: float = 1.0,
        exit_data: dict | None = None,
        close_reason: str = "",
    ) -> bool:
        with self._trade_sequence_lock:
            self._trade_sequence += 1
            _seq = self._trade_sequence
            ticket = f"PAPER_{self._epoch_ts}_{_seq}"
        if isinstance(entry_time, str):
            entry_time = dt.datetime.fromisoformat(entry_time)
        hold_secs = (exit_time - entry_time).total_seconds() if entry_time else 0.0
        bars_held = round(hold_secs / max(self.timeframe_minutes * 60, 1))
        _price_ref = max(abs(entry_price), 1.0)
        entry_half_spread = float((trigger_data or {}).get("entry_half_spread", self.last_half_spread) or 0.0)
        spread_cost_pts = entry_half_spread + float(self.last_half_spread or 0.0)
        pnl_net = pnl_pts - spread_cost_pts
        trade_qty = float(quantity if quantity is not None else self.qty)
        contract_size = float(getattr(self, "contract_size", 1.0) or 1.0)
        lot_value = trade_qty * contract_size
        mfe_usd = float(mfe) * lot_value
        mae_usd = float(mae) * lot_value
        predicted_runway_net_points_raw = max(0.0, float(predicted_runway_net or 0.0)) * _price_ref
        predicted_runway_gross_points = max(0.0, float(predicted_runway_gross or 0.0)) * _price_ref
        runway_bias_ema_points = float(getattr(self, "_runway_delta_ema", 0.0) or 0.0)
        tf_gain = float(np.clip(15.0 / float(max(int(self.timeframe_minutes or 1), 1)), 0.6, 2.5))
        bias_clip = min(_RUNWAY_BIAS_LIMIT_POINTS, max(_price_ref * 0.003, 1.0))
        clipped_bias = float(np.clip(runway_bias_ema_points * tf_gain, -bias_clip, bias_clip))
        adjusted_runway_points = max(0.0, predicted_runway_net_points_raw - clipped_bias)
        runway_adjustment_scale = SafeMath.safe_div(
            adjusted_runway_points,
            max(predicted_runway_net_points_raw, 1e-6),
            1.0,
        )
        runway_adjustment_scale = float(
            np.clip(runway_adjustment_scale, _RUNWAY_ADJUST_MIN_SCALE, _RUNWAY_ADJUST_MAX_SCALE),
        )
        predicted_runway_net_points = predicted_runway_net_points_raw * runway_adjustment_scale
        runway_utilization = SafeMath.safe_div(float(mfe), predicted_runway_net_points, 0.0)
        runway_delta_points = predicted_runway_net_points - float(mfe)
        runway_error_pct = (
            abs(runway_delta_points) / max(predicted_runway_net_points, 1.0) * 100.0
            if predicted_runway_net_points > 0
            else 0.0
        )
        bars_from_mfe_to_exit = int((exit_data or {}).get("bars_from_mfe_to_exit", -1) or -1)
        mfe_bar_offset = int((exit_data or {}).get("mfe_bar_offset", -1) or -1)
        mae_bar_offset = int((exit_data or {}).get("mae_bar_offset", -1) or -1)
        trigger_quality = self._classify_trigger_quality(predicted_runway_net_points, float(mfe))
        harvester_quality = self._classify_harvester_quality(
            pnl_usd=pnl_usd,
            mfe_usd=mfe_usd,
            winner_to_loser=was_winner_to_loser,
            bars_from_mfe_to_exit=bars_from_mfe_to_exit,
        )
        diag_zero_mfe_loss = pnl_usd < 0 and mfe_usd <= SAFE_EPSILON
        diag_close_spread = float(self.last_half_spread or 0.0) * 2.0
        close_mid = float((exit_data or {}).get("exit_mid", exit_price) or exit_price)
        diag_close_spread_bps = (diag_close_spread / close_mid * 10_000.0) if close_mid > 0 else 0.0
        record = {
            "trade_id": _seq,
            "ticket": ticket,
            "position_id": f"{self.symbol_id}_ticket_{ticket}",
            "symbol": self.symbol,
            "timeframe": self.tf_label,
            "timeframe_minutes": self.timeframe_minutes,
            "trading_mode": "paper",
            "direction": "LONG" if direction == 1 else "SHORT",
            "quantity": trade_qty,
            "contract_size": contract_size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_time": entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time),
            "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time),
            "pnl": pnl_usd,
            "pnl_points": pnl_pts,
            "pnl_net_points": pnl_net,
            "mfe": mfe_usd,
            "mae": mae_usd,
            "mfe_points": mfe,
            "mae_points": mae,
            "close_reason": close_reason or "",
            "capture_ratio": float(capture_ratio),
            "ticks_held": ticks_held,
            "bars_held": bars_held,
            "hold_seconds": hold_secs,
            "entry_confidence": self._entry_conf,
            "trigger_reward": trigger_reward,
            "capture_reward": capture_reward,
            "predicted_runway_gross": predicted_runway_gross,
            "predicted_runway_net": predicted_runway_net,
            "predicted_runway_gross_points": predicted_runway_gross_points,
            "predicted_runway_net_points": predicted_runway_net_points,
            "predicted_runway_net_points_raw": predicted_runway_net_points_raw,
            "runway_bias_ema_points": runway_bias_ema_points,
            "runway_adjustment_scale": runway_adjustment_scale,
            "runway_delta_points": runway_delta_points,
            "runway_utilization": float(np.clip(runway_utilization, -2.0, 2.0)),
            "runway_error_pct": runway_error_pct,
            "runway_delta_ema": self._runway_delta_ema,
            "runway_accuracy_ema": self._runway_accuracy_ema,
            "trigger_quality": trigger_quality,
            "harvester_quality": harvester_quality,
            "mfe_bar_offset": mfe_bar_offset,
            "mae_bar_offset": mae_bar_offset,
            "bars_from_mfe_to_exit": bars_from_mfe_to_exit,
            "winner_to_loser": was_winner_to_loser,
            "reward_wtl_net_flag": reward_wtl_net_flag,
            "regime": regime,
            "entry_vpin_z": entry_vpin_z,
            "entry_var_95": entry_var_95,
            "entry_imbalance": self._entry_imbalance,
            "entry_dynamic_floor": entry_dynamic_floor,
            "entry_conf_margin": entry_conf_margin,
            "win_rate_ema_at_entry": win_rate_ema_at_entry,
            "total_trades_at_entry": total_trades_at_entry,
            "equity_at_entry": equity_at_entry,
            "conf_calib_err_at_entry": conf_calib_err_at_entry,
            "runway_accuracy_at_entry": runway_accuracy_at_entry,
            "exit_regime": exit_regime,
            "exit_vol": exit_vol,
            "exit_depth_ratio": exit_depth_ratio,
            "diag_circuit_breaker_active": diag_cb_active,
            "diag_circuit_breakers_tripped": diag_cb_tripped or [],
            "diag_zero_mfe_loss": diag_zero_mfe_loss,
            "diag_close_spread": diag_close_spread,
            "diag_close_spread_bps": diag_close_spread_bps,
            "spread_cost_points": spread_cost_pts,
            "balance_after": self.equity,
            "decision_trade_id": trade_id,
            "reward_capture_efficiency": reward_capture_efficiency,
            "reward_wtl_penalty": reward_wtl_penalty,
            "reward_opportunity_cost": reward_opportunity_cost,
            "reward_session_quality": reward_session_quality,
            "reward_harvester_total": reward_harvester_total,
            "reward_trigger_breakdown": reward_trigger_breakdown or {},
            "reward_harvester_breakdown": reward_harvester_breakdown or {},
            "trigger_data": trigger_data or {},
            "exit_data": exit_data or {},
            "close_drawdown_pct": close_drawdown_pct,
            "close_cb_size_mult": close_cb_size_mult,
        }
        try:
            log_path = Path("data") / "trade_log.jsonl"
            append_jsonl_durable(log_path, record, default=json_default)
            return True
        except Exception as e:
            LOG.debug("[%s %s] trade_log write error: %s", self.symbol, self.tf_label, e)
            return False
