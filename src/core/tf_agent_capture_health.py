"""Adaptive capture-health monitoring for :class:`TFAgent`.

Extracted verbatim from ``openapi_hub`` as a behaviour-preserving mixin. These
methods adjust harvester trailing-stop / capture-decay thresholds in response to
the rolling capture ratio: a two-tier intervention (immediate large-delta and
EMA-based alert/critical) plus a slow relax when capture is stably healthy.
"""

import logging
import time
from typing import Any

LOG = logging.getLogger(__name__)


class TFAgentCaptureHealthMixin:
    """Two-tier capture intervention and threshold relax/tighten."""

    # Minimum trades per TF before rolling EMA intervention fires.
    # Large-delta path bypasses this entirely and acts immediately.
    _CAPTURE_MIN_SAMPLES: dict = {1: 15, 5: 10, 15: 7, 30: 5, 60: 4, 240: 3}

    def _check_capture_health(self, capture_ratio: float, mfe: float, entry_price: float) -> None:
        """Two-tier capture intervention.

        Tier 1 — large delta (big MFE, low capture): act immediately, no gates.
        Tier 2 — rolling EMA: act after TF-adaptive minimum samples + 1-bar cooldown.
        Stable: relax slowly (3% per trade) only after 2× min samples to prevent whipsawing.
        """
        from src.constants import (
            CAPTURE_ALERT_THRESHOLD,
            CAPTURE_CRITICAL_THRESHOLD,
            CAPTURE_EMA_ALPHA,
            CAPTURE_LARGE_DELTA_CAP_MAX,
            CAPTURE_LARGE_DELTA_MFE_MULT,
            CAPTURE_RELAX_FACTOR,
            CAPTURE_STABLE_THRESHOLD,
            CAPTURE_TIGHTEN_ALERT,
            CAPTURE_TIGHTEN_IMMEDIATE,
        )

        self._capture_ema = (
            (1.0 - CAPTURE_EMA_ALPHA) * self._capture_ema
            + CAPTURE_EMA_ALPHA * max(-1.0, min(1.0, capture_ratio))
        )
        self._capture_ema_n += 1

        harv = getattr(getattr(self, "policy", None), "harvester", None)
        if harv is None:
            return

        mfe_pct = (mfe / max(abs(entry_price), 1.0)) * 100.0
        now = time.time()
        tf_cooldown = max(60.0, self.timeframe_minutes * 60.0)

        # ── Tier 1: IMMEDIATE — large delta wastes a significant move ──────────
        significant_mfe = (
            getattr(harv, "trailing_stop_activation_pct", 0.15) * CAPTURE_LARGE_DELTA_MFE_MULT
        )
        if mfe_pct > significant_mfe and capture_ratio < CAPTURE_LARGE_DELTA_CAP_MAX:
            LOG.warning(
                "[%s %s] CAPTURE DELTA: MFE=%.3f%% capture=%.1f%% "
                "(threshold=2×trail_act=%.3f%%) — immediate tighten",
                self.symbol, self.tf_label, mfe_pct, capture_ratio * 100, significant_mfe,
            )
            self._apply_capture_tighten(harv, factor=CAPTURE_TIGHTEN_IMMEDIATE)
            self._capture_last_intervention = now
            return

        # ── Tier 2: ROLLING EMA — needs min samples + cooldown ────────────────
        min_samples = self._CAPTURE_MIN_SAMPLES.get(self.timeframe_minutes, 5)
        if self._capture_ema_n < min_samples:
            return
        if now - self._capture_last_intervention < tf_cooldown:
            return

        if self._capture_ema < CAPTURE_CRITICAL_THRESHOLD:
            LOG.warning(
                "[%s %s] CAPTURE CRITICAL: rolling=%.1f%% — emergency reset",
                self.symbol, self.tf_label, self._capture_ema * 100,
            )
            self._apply_capture_emergency_reset(harv)
            self._capture_last_intervention = now
        elif self._capture_ema < CAPTURE_ALERT_THRESHOLD:
            LOG.warning(
                "[%s %s] CAPTURE ALERT: rolling=%.1f%% — tightening thresholds",
                self.symbol, self.tf_label, self._capture_ema * 100,
            )
            self._apply_capture_tighten(harv, factor=CAPTURE_TIGHTEN_ALERT)
            self._capture_last_intervention = now
        elif self._capture_ema > CAPTURE_STABLE_THRESHOLD and self._capture_ema_n >= min_samples * 2:
            # Stable performance: small relax, larger sample base to prevent whipsawing
            self._apply_capture_relax(harv, factor=CAPTURE_RELAX_FACTOR)

    def _apply_capture_tighten(self, harv: Any, factor: float) -> None:
        """Tighten trailing activation, stop distance, and capture decay threshold."""
        from src.constants import (
            TRAILING_STOP_ACTIVATION_PCT,
            TRAILING_STOP_DISTANCE_PCT,
        )
        tf_scale = harv._get_timeframe_scale() if hasattr(harv, "_get_timeframe_scale") else 1.0
        trail_floor = max(0.03, TRAILING_STOP_ACTIVATION_PCT * tf_scale * 0.40)
        dist_floor = max(0.01, TRAILING_STOP_DISTANCE_PCT * tf_scale * 0.30)

        harv.trailing_stop_activation_pct = max(
            trail_floor, harv.trailing_stop_activation_pct * factor,
        )
        harv.trailing_stop_distance_pct = max(
            dist_floor, harv.trailing_stop_distance_pct * factor,
        )
        # Raise capture_decay_threshold so capture-decay fires sooner on giveback
        harv.capture_decay_threshold = min(
            0.70, harv.capture_decay_threshold + (1.0 - factor) * 0.40,
        )
        LOG.info(
            "[%s %s] CAPTURE TIGHTEN (×%.2f): trail_act=%.3f%% dist=%.3f%% cd_thresh=%.3f",
            self.symbol, self.tf_label, factor,
            harv.trailing_stop_activation_pct,
            harv.trailing_stop_distance_pct,
            harv.capture_decay_threshold,
        )

    def _apply_capture_emergency_reset(self, harv: Any) -> None:
        """Emergency: reset harvester thresholds to tightest safe values (50% of default)."""
        from src.constants import (
            CAPTURE_DECAY_MIN_MFE_PCT,
            TRAILING_STOP_ACTIVATION_PCT,
            TRAILING_STOP_DISTANCE_PCT,
        )
        tf_scale = harv._get_timeframe_scale() if hasattr(harv, "_get_timeframe_scale") else 1.0
        harv.trailing_stop_activation_pct = TRAILING_STOP_ACTIVATION_PCT * tf_scale * 0.50
        harv.trailing_stop_distance_pct = TRAILING_STOP_DISTANCE_PCT * tf_scale * 0.50
        harv.capture_decay_threshold = 0.50
        harv.capture_decay_min_mfe_pct = CAPTURE_DECAY_MIN_MFE_PCT * tf_scale
        LOG.warning(
            "[%s %s] CAPTURE EMERGENCY RESET: trail_act=%.3f%% dist=%.3f%% cd_thresh=%.3f",
            self.symbol, self.tf_label,
            harv.trailing_stop_activation_pct,
            harv.trailing_stop_distance_pct,
            harv.capture_decay_threshold,
        )

    def _apply_capture_relax(self, harv: Any, factor: float) -> None:
        """Gently relax thresholds when capture is stably healthy (prevents over-tightening)."""
        from src.constants import (
            CAPTURE_DECAY_THRESHOLD,
            TRAILING_STOP_ACTIVATION_PCT,
            TRAILING_STOP_DISTANCE_PCT,
        )
        tf_scale = harv._get_timeframe_scale() if hasattr(harv, "_get_timeframe_scale") else 1.0
        trail_ceil = TRAILING_STOP_ACTIVATION_PCT * tf_scale * 1.50
        dist_ceil = TRAILING_STOP_DISTANCE_PCT * tf_scale * 1.50

        harv.trailing_stop_activation_pct = min(
            trail_ceil, harv.trailing_stop_activation_pct / factor,
        )
        harv.trailing_stop_distance_pct = min(
            dist_ceil, harv.trailing_stop_distance_pct / factor,
        )
        # Never relax capture_decay below original default
        harv.capture_decay_threshold = max(
            CAPTURE_DECAY_THRESHOLD, harv.capture_decay_threshold * factor,
        )
        LOG.debug(
            "[%s %s] CAPTURE RELAX (×%.3f): trail_act=%.3f%% dist=%.3f%%",
            self.symbol, self.tf_label, factor,
            harv.trailing_stop_activation_pct,
            harv.trailing_stop_distance_pct,
        )
