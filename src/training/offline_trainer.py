#!/usr/bin/env python3
"""
OfflineTrainer
==============
Walk-forward DDQN training on historical bars for one (symbol, timeframe) job.

Algorithm
----------
1. Split bars into train (80 %) + validation (20 %) sets.
2. Walk bar-by-bar through train set with a sliding deque:
      - If FLAT  → call policy.decide_entry(); if signal → record entry state
      - If LONG/SHORT → call policy.decide_exit(); if CLOSE → add both
        trigger + harvester experiences; call train_step every TRAIN_EVERY bars
3. After train pass: run the same simulation on the validation set (no gradient
   updates) and collect per-trade PnL-pts returns.
4. Compute ZOmega on validation returns.
5. Return a TrainResult with final weights path + ZOmega score.

ZOmega
------
Omega ratio on σ-normalised returns (scale-agnostic Omega):
    z_i  = r_i / σ_r              (σ-normalise, no mean subtraction)
    ZΩ   = Σ max(z_i, 0) / Σ max(-z_i, 0)

Mean subtraction is intentionally omitted: subtracting the mean makes
Σz_i = 0 by identity, forcing ZΩ = 1.0 for all inputs.  Dividing by σ
only removes scale while preserving the gain/loss structure — equivalent
to the standard Omega ratio on the same returns, but dimensionless across
different instruments and timeframes.

Caller: train_offline.py spawns one OfflineTrainer.run() per job in a
        separate process, so no shared state is needed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.core.reward_shaper import RewardShaper
from src.features.event_time_features import EventTimeFeatureEngine
from src.risk.path_geometry import PathGeometry
from src.utils.mfe_mae import MFEMAECalculator

LOG = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TRAIN_SPLIT: float = 0.80          # Fraction of bars used for training
TRAIN_EVERY: int = 4               # call train_step() every N bar-closes
MIN_BARS_FOR_ENTRY: int = 80       # Minimum deque depth before entry allowed
MAX_POSITION_BARS: int = 200       # Hard cap on simulated position hold
DEQUE_MAXLEN: int = 500            # Rolling bars deque max length
REWARD_CLIP: float = 2.0           # Hardcoded to match live bot
TRIGGER_REWARD_CLIP: float = 0.5
MIN_VALIDATION_TRADES: int = 5     # Below this ZOmega is not meaningful
_STD_FLOOR: float = 1e-10          # Minimum σ before treating as flat returns
_MFE_FLOOR: float = 1e-8           # Minimum MFE to compute capture ratio
_BAR_SPREAD_COL_IDX: int = 5       # Index of spread column in bar tuple

# ── Types ──────────────────────────────────────────────────────────────────────

class TradeRecord(NamedTuple):
    entry_bar_idx: int
    exit_bar_idx: int
    direction: int      # 1 LONG, -1 SHORT
    entry_price: float
    exit_price: float
    pnl_pts: float      # raw price-point P&L (instrument-agnostic)
    mfe: float
    mae: float
    trigger_action: int
    trigger_reward: float
    capture_reward: float


@dataclass
class TrainResult:
    symbol: str
    timeframe_minutes: int
    z_omega: float                   # Validation ZOmega (primary selection key)
    train_trades: int
    val_trades: int
    total_train_steps: int
    elapsed_s: float
    weights_path: str = ""           # Written by OfflineTrainer.run()
    error: str = ""                  # Non-empty if the job crashed


# ── ZOmega ────────────────────────────────────────────────────────────────────

def z_omega(returns: list[float], threshold: float = 0.0) -> float:
    """
    Compute Omega ratio on σ-normalised returns (scale-agnostic Omega).

    Normalises each return by the sample standard deviation of the window
    (WITHOUT subtracting the mean) so the metric is instrument- and
    timeframe-agnostic.  Mathematically equivalent to the standard Omega ratio
    but expressed in σ units, making scores comparable across symbol/TF jobs.

    Note: Full Z-scoring (subtract mean then divide by σ) yields trivial
    ZOmega = 1.0 for all datasets because Z-scores sum to zero by definition.

    Args:
        returns:    Per-trade PnL-pts returns.
        threshold:  Threshold in σ units (default 0 = breakeven).

    Returns:
        ZOmega ∈ (0, ∞).  +inf when there are no losses.
        Returns 0.0 when fewer than MIN_VALIDATION_TRADES trades.
    """
    if len(returns) < MIN_VALIDATION_TRADES:
        return 0.0

    arr = np.array(returns, dtype=np.float64)
    sig = arr.std()
    if sig < _STD_FLOOR:
        # All returns identical — edge case, treat as neutral
        return 1.0

    z = arr / sig              # σ-normalise only; threshold in sigma units
    gains  = np.maximum(z - threshold, 0.0).sum()
    losses = np.maximum(threshold - z, 0.0).sum()

    if losses < _STD_FLOOR:
        return float("inf")
    return float(gains / losses)


# ── Simulation ────────────────────────────────────────────────────────────────

class _Simulator:
    """
    Stateful bar-by-bar walker.

    Calls DualPolicy entry/exit methods and accumulates experiences.
    Does NOT call train_step — the caller decides when to update.
    """

    def __init__(self, policy, update_policy: bool = True, symbol_digits: int = 2,
                 event_engine: EventTimeFeatureEngine | None = None,
                 reward_clip_harvester: float = REWARD_CLIP,
                 reward_clip_trigger: float = TRIGGER_REWARD_CLIP,
                 capture_baseline: float = 0.5,
                 symbol: str = "XAUUSD",
                 timeframe: str = "M5",
                 penalty_scale: float = 1.0) -> None:
        self.policy = policy
        self.update_policy = update_policy  # False during validation pass
        self.bars: deque = deque(maxlen=DEQUE_MAXLEN)
        self._pt: float = 10 ** (-symbol_digits)  # broker points → price-unit multiplier
        self.event_engine = event_engine  # Compute event-time features from bar timestamps
        self._reward_clip_harvester = reward_clip_harvester
        self._reward_clip_trigger = reward_clip_trigger
        self._capture_baseline = capture_baseline

        # RewardShaper — canonical reward computation aligned with online training
        self._reward_shaper = RewardShaper(symbol=symbol, timeframe=timeframe)
        if event_engine is not None:
            self._reward_shaper.set_event_engine(event_engine)
        self._penalty_scale = penalty_scale  # soften WTL/timing penalties for curriculum training

        # Position state
        self.cur_pos: int = 0           # 0 flat, 1 long, -1 short
        self.entry_price: float = 0.0
        self.entry_bar_idx: int = 0
        self.entry_action: int = 0
        self._mfe_calc = MFEMAECalculator()  # single source of truth
        self.ticks_held: int = 0
        self.entry_spread_pts: float = 0.0  # spread at entry bar (broker points)
        self._cur_spread_pts: float = 0.0   # spread at current bar (broker points)
        self._predicted_runway: float = 0.0  # stored at entry for trigger reward

        # State snapshots for experience labelling
        self.entry_state: np.ndarray | None = None

        # Accumulated result
        self.trades: list[TradeRecord] = []

    def step(self, bar: tuple, bar_idx: int) -> None:
        """Process one bar."""
        self.bars.append(bar)
        self._cur_spread_pts = float(bar[_BAR_SPREAD_COL_IDX]) if len(bar) > _BAR_SPREAD_COL_IDX else 0.0

        if len(self.bars) < MIN_BARS_FOR_ENTRY:
            return

        current_price = float(bar[4])  # close

        if self.cur_pos == 0:
            self._try_entry(bar_idx, current_price)
        else:
            self._update_mfe_mae(current_price)
            self._try_exit(bar_idx, current_price)

    def _get_event_features(self) -> dict | None:
        """Compute event-time features from the latest bar timestamp."""
        if self.event_engine is None or len(self.bars) == 0:
            return None
        bar_time = self.bars[-1][0]
        if bar_time is None:
            return None
        return self.event_engine.calculate_features(bar_time)

    def _try_entry(self, bar_idx: int, current_price: float) -> None:
        try:
            action, _, _ = self.policy.decide_entry(
                self.bars, imbalance=0.0, vpin_z=0.0, depth_ratio=1.0,
                event_features=self._get_event_features(),
            )
        except Exception as exc:
            LOG.debug("[SIM] decide_entry failed at bar %d: %s", bar_idx, exc)
            return

        if action not in (1, 2):
            return

        direction = 1 if action == 1 else -1
        self.cur_pos = direction
        self.entry_price = current_price
        self.entry_spread_pts = self._cur_spread_pts
        self.entry_bar_idx = bar_idx
        self.entry_action = action
        self._mfe_calc.start(current_price, direction)
        self.ticks_held = 0
        self._predicted_runway = getattr(self.policy, "predicted_runway", 0.0)

        # Snapshot entry state for trigger experience
        if self.update_policy:
            try:
                trigger = self.policy.trigger
                self.entry_state = (
                    trigger.last_state.copy()
                    if hasattr(trigger, "last_state") and trigger.last_state is not None
                    else None
                )
            except Exception:
                self.entry_state = None

        try:
            self.policy.on_entry(
                direction=direction,
                entry_price=current_price,
                entry_time=self.bars[-1][0],
            )
        except Exception as exc:
            LOG.debug("[SIM] on_entry failed: %s", exc)

    def _try_exit(self, bar_idx: int, current_price: float) -> None:
        self.ticks_held += 1
        hard_stop = self.ticks_held > MAX_POSITION_BARS

        action = 0
        if not hard_stop:
            try:
                action, _ = self.policy.decide_exit(
                    self.bars,
                    current_price=current_price,
                    imbalance=0.0,
                    vpin_z=0.0,
                    depth_ratio=1.0,
                    event_features=self._get_event_features(),
                )
            except Exception as exc:
                LOG.debug("[SIM] decide_exit failed: %s", exc)
                action = 0

        if action == 1 or hard_stop:
            self._close_position(bar_idx, current_price)

    def _update_mfe_mae(self, current_price: float) -> None:
        self._mfe_calc.update(current_price)

    @property
    def mfe(self) -> float:
        return self._mfe_calc.mfe

    @property
    def mae(self) -> float:
        return self._mfe_calc.mae

    def _rs_volatility(self, window: int = 20) -> float:
        """Rogers-Satchell volatility from recent bars (matches live bot)."""
        n = min(len(self.bars), window)
        if n < 5:
            return 0.01
        rs_sum = 0.0
        for i in range(-n, 0):
            b = self.bars[i]
            o, h, l, c = float(b[1]), float(b[2]), float(b[3]), float(b[4])
            if o <= 0:
                continue
            ho = np.log(h / o) if h > 0 and o > 0 else 0.0
            lo = np.log(l / o) if l > 0 and o > 0 else 0.0
            hc = np.log(h / c) if h > 0 and c > 0 else 0.0
            lc = np.log(l / c) if l > 0 and c > 0 else 0.0
            rs_sum += ho * hc + lo * lc
        var = max(rs_sum / n, 1e-12)
        return float(np.sqrt(var))

    def _compute_trigger_reward(self, pnl_pts: float) -> float:
        """Trigger reward aligned with live bot's _calculate_trigger_reward().

        Components:
        1. Accuracy: how close predicted_runway was to actual MFE
        2. Magnitude: prefer identifying larger MFE opportunities
        3. False-positive penalty: loss despite positive prediction

        All computation in σ-of-price units (instrument-agnostic).
        VPIN toxic-flow penalty omitted (not available offline).
        """
        actual_mfe = self.mfe
        realized_vol = self._rs_volatility()
        entry_price = max(self.entry_price, 1.0)
        vol_pts = max(realized_vol * entry_price, 1e-6)
        predicted_runway_pts = self._predicted_runway * entry_price

        norm_mfe = actual_mfe / vol_pts
        norm_predicted = predicted_runway_pts / vol_pts

        # Component 1: Prediction accuracy
        prediction_error = abs(norm_mfe - norm_predicted)
        max_error = max(norm_mfe, norm_predicted, 1.0)
        runway_accuracy = 1.0 - (prediction_error / max_error)
        accuracy_reward = runway_accuracy * 2.0 - 1.0

        # Component 2: Magnitude bonus (cap at +0.5 for 3σ+ moves)
        magnitude_bonus = min(norm_mfe / 3.0, 1.0) * 0.5

        # Component 3: False-positive penalty
        false_positive_penalty = 0.0
        if self._predicted_runway > 0 and pnl_pts < 0:
            loss_severity = min(abs(pnl_pts) / vol_pts / 3.0, 1.0)
            false_positive_penalty = -0.2 - 0.5 * loss_severity

        total = accuracy_reward + magnitude_bonus + false_positive_penalty
        return float(np.clip(total, -self._reward_clip_trigger, self._reward_clip_trigger))

    def _close_position(self, bar_idx: int, exit_price: float) -> None:
        # Deduct round-trip spread cost (entry half-spread + exit half-spread = 1 full spread)
        spread_cost = (self.entry_spread_pts + self._cur_spread_pts) * self._pt
        pnl_pts = (exit_price - self.entry_price) * self.cur_pos - spread_cost
        capture = self.mfe > 0 and pnl_pts > 0
        capture_ratio = (pnl_pts / self.mfe) if self.mfe > _MFE_FLOOR else 0.0

        # ── Harvester reward via canonical RewardShaper ───────────────────
        # Aligned with online training: capture efficiency, WTL penalty,
        # MAE/MFE timing penalty, session quality, zero-MFE penalty.
        exit_time = ""
        if len(self.bars) > 0 and self.bars[-1][0] is not None:
            exit_time = str(self.bars[-1][0])
        harvester_result = self._reward_shaper.calculate_harvester_reward(
            exit_pnl=pnl_pts,
            mfe=self.mfe,
            was_wtl=not capture,
            bars_held=self.ticks_held,
            bars_from_mfe_to_exit=0,
            mae=self.mae,
            exit_time=exit_time,
        )
        # Curriculum scaling: keep capture efficiency at full strength but
        # soften penalty components so the agent can explore before being
        # punished heavily for WTL and poor timing.
        r_capture_eff = harvester_result.get("capture_efficiency", 0.0)
        r_wtl = harvester_result.get("wtl_penalty", 0.0) * self._penalty_scale
        r_timing = harvester_result.get("timing_penalty", 0.0) * self._penalty_scale
        session_mult = harvester_result.get("session_quality", 1.0)
        softened_reward = (r_capture_eff + r_wtl + r_timing) * session_mult
        capture_reward = float(np.clip(softened_reward,
                                       -self._reward_clip_harvester, self._reward_clip_harvester))

        # ── Trigger reward: prediction accuracy (matches live bot) ────────
        trigger_reward = self._compute_trigger_reward(pnl_pts)

        if self.update_policy and self.entry_state is not None:
            self._add_experiences(trigger_reward=trigger_reward, capture_reward=capture_reward)

        self.trades.append(TradeRecord(
            entry_bar_idx=self.entry_bar_idx,
            exit_bar_idx=bar_idx,
            direction=self.cur_pos,
            entry_price=self.entry_price,
            exit_price=exit_price,
            pnl_pts=float(pnl_pts),
            mfe=float(self.mfe),
            mae=float(self.mae),
            trigger_action=self.entry_action,
            trigger_reward=float(trigger_reward),
            capture_reward=float(capture_reward),
        ))

        with contextlib.suppress(Exception):
            self.policy.on_exit(
                exit_price=exit_price,
                capture_ratio=capture_ratio,
                was_wtl=not capture,
            )

        self.cur_pos = 0
        self.entry_price = 0.0
        self.entry_state = None
        self.ticks_held = 0
        self._predicted_runway = 0.0
        self._mfe_calc.reset()

    def _add_experiences(self, trigger_reward: float, capture_reward: float) -> None:
        try:
            trig = self.policy.trigger
            next_state = (
                trig.last_state.copy()
                if hasattr(trig, "last_state") and trig.last_state is not None
                else self.entry_state
            )
            self.policy.add_trigger_experience(
                state=self.entry_state,
                action=self.entry_action,
                reward=trigger_reward,
                next_state=next_state,
                done=True,
            )
        except Exception as exc:
            LOG.debug("[SIM] add_trigger_experience failed: %s", exc)

        try:
            harv = self.policy.harvester
            prev_state = (
                harv.last_state.copy()
                if hasattr(harv, "last_state") and harv.last_state is not None
                else None
            )
            if prev_state is not None:
                next_state = prev_state  # terminal step
                self.policy.add_harvester_experience(
                    state=prev_state,
                    action=1,  # CLOSE
                    reward=capture_reward,
                    next_state=next_state,
                    done=True,
                )
        except Exception as exc:
            LOG.debug("[SIM] add_harvester_experience failed: %s", exc)


# ── OfflineTrainer ─────────────────────────────────────────────────────────────

class OfflineTrainer:
    """
    Train one DualPolicy on historical bars and report ZOmega on hold-out.

    Args:
        symbol:             Instrument name (e.g. "XAUUSD").
        timeframe_minutes:  Bar timeframe in minutes.
        bars:               Full sorted bar list from HistoricalLoader.
        checkpoint_dir:     Where to save trained weights.
        train_split:        Fraction of bars used for training (default 0.80).
        train_every:        Number of bar-close steps between train_step() calls.
        policy_kwargs:      Extra kwargs forwarded to DualPolicy constructor.
        n_epochs:           Number of full passes over the training set (default 1).
                            Each epoch resets epsilon so the policy re-explores
                            with increasingly warm weights.
        warm_start:         If True, load existing per-label checkpoint weights
                            (``<checkpoint_dir>/<label>_trigger_offline.pt``) before
                            the first epoch so training continues from a prior run.
        epsilon_start:      Epsilon at the beginning of each training epoch
                            (overrides EPSILON_START env var, default 0.4).
        epsilon_end:        Epsilon floor across all epochs
                            (overrides EPSILON_END env var, default 0.05).
    """

    def __init__(  # noqa: PLR0913
        self,
        symbol: str,
        timeframe_minutes: int,
        bars: list,
        checkpoint_dir: str | Path = "data/checkpoints",
        train_split: float = TRAIN_SPLIT,
        train_every: int = TRAIN_EVERY,
        policy_kwargs: dict | None = None,
        n_epochs: int = 1,
        warm_start: bool = False,
        epsilon_start: float = 0.4,
        epsilon_end: float = 0.05,
        symbol_digits: int = 2,
        reward_clip_harvester: float = REWARD_CLIP,
        reward_clip_trigger: float = TRIGGER_REWARD_CLIP,
        capture_baseline: float = 0.5,
        penalty_scale: float = 1.0,
    ) -> None:
        self.symbol = symbol
        self.timeframe_minutes = timeframe_minutes
        self.bars = bars
        self.checkpoint_dir = Path(checkpoint_dir)
        self.train_split = train_split
        self.train_every = train_every
        self.policy_kwargs = policy_kwargs or {}
        self.n_epochs = max(1, int(n_epochs))
        self.warm_start = warm_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.symbol_digits = symbol_digits
        self._reward_clip_harvester = reward_clip_harvester
        self._reward_clip_trigger = reward_clip_trigger
        self._capture_baseline = capture_baseline
        self._penalty_scale = penalty_scale

    # ── Entry point ──────────────────────────────────────────────────────────

    def run(self) -> TrainResult:
        """Execute training + validation. Returns a TrainResult."""
        t0 = time.perf_counter()
        label = f"{self.symbol}_M{self.timeframe_minutes}"

        try:
            return self._run_inner(label, t0)
        except Exception as exc:
            LOG.error("[OFFLINE] %s crashed: %s", label, exc, exc_info=True)
            return TrainResult(
                symbol=self.symbol,
                timeframe_minutes=self.timeframe_minutes,
                z_omega=0.0,
                train_trades=0,
                val_trades=0,
                total_train_steps=0,
                elapsed_s=time.perf_counter() - t0,
                error=str(exc),
            )

    @staticmethod
    def _restore_env_vars(orig_start: str | None, orig_end: str | None, orig_gates: str | None) -> None:
        """Restore EPSILON_START, EPSILON_END, DISABLE_GATES env vars to their original values."""
        import os  # noqa: PLC0415
        for key, original in [("EPSILON_START", orig_start), ("EPSILON_END", orig_end), ("DISABLE_GATES", orig_gates)]:
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    def _warm_start_weights(self, policy, label: str) -> None:
        """Load pre-trained weights for both agents if warm_start is enabled."""
        if not self.warm_start:
            return
        for agent_name, agent in [("trigger", policy.trigger), ("harvester", policy.harvester)]:
            ckpt = self.checkpoint_dir / f"{label}_{agent_name}_offline.pt"
            if ckpt.exists() and agent.ddqn is not None:
                try:
                    agent.ddqn.load_weights(str(ckpt))
                    LOG.info("[OFFLINE] %s warm-started from %s", label, ckpt.name)
                except Exception as exc:
                    LOG.warning("[OFFLINE] %s could not load %s: %s", label, ckpt.name, exc)

    def _run_inner(self, label: str, t0: float) -> TrainResult:
        # Lazy import to avoid circular imports and allow multiprocessing fork
        import os  # noqa: PLC0415

        from src.agents.dual_policy import DualPolicy  # noqa: PLC0415

        n_total = len(self.bars)
        n_train = int(n_total * self.train_split)
        train_bars = self.bars[:n_train]
        val_bars   = self.bars[n_train:]

        LOG.info(
            "[OFFLINE] %s: %d train bars / %d val bars  (epochs=%d, warm_start=%s)",
            label, len(train_bars), len(val_bars), self.n_epochs, self.warm_start,
        )

        # Scale PER buffer to training set size: up to half the training bars,
        # capped at 20k to avoid unbounded memory. Explicit policy_kwargs override wins.
        offline_buffer_capacity = min(max(len(train_bars) // 2, 2_000), 20_000)
        policy_kwargs = {
            "trigger_buffer_capacity": offline_buffer_capacity,
            "harvester_buffer_capacity": offline_buffer_capacity,
            **self.policy_kwargs,
        }

        # Override epsilon schedule AND disable live gating for offline training.
        # Without DISABLE_GATES=1 the live confidence/feasibility gates block
        # every entry the untrained network attempts → 0 val trades → ZΩ=0.
        # Restore originals after policy construction so child processes don't
        # bleed into each other (each is a spawned process, so this is safe).
        _orig_eps_start, _orig_eps_end, _orig_disable_gates = self._set_offline_env_vars(os)

        # Build policy — training enabled, with full feature parity to online
        path_geometry = PathGeometry()
        self._event_engine = EventTimeFeatureEngine()
        policy = DualPolicy(
            symbol=self.symbol,
            timeframe=f"M{self.timeframe_minutes}",
            window=64,
            enable_training=True,
            enable_event_features=True,
            path_geometry=path_geometry,
            timeframe_minutes=self.timeframe_minutes,
            **policy_kwargs,
        )

        # Restore env vars immediately after construction
        self._restore_env_vars(_orig_eps_start, _orig_eps_end, _orig_disable_gates)

        # ── Warm start: load existing checkpoint weights ───────────────────────
        self._warm_start_weights(policy, label)

        # ── Multi-epoch training pass ──────────────────────────────────────────
        _progress_path = Path(f"data/offline_progress_{self.symbol}_M{self.timeframe_minutes}.json")
        _progress_path.parent.mkdir(parents=True, exist_ok=True)
        _total_bars_all_epochs = len(train_bars) * self.n_epochs
        _progress_every = max(50, len(train_bars) // 100)   # ~100 HUD updates per epoch

        total_train_steps, total_train_trades = self._run_training_epochs(
            policy,
            train_bars,
            label,
            _progress_path,
            _total_bars_all_epochs,
            _progress_every,
        )

        _progress_path.unlink(missing_ok=True)   # clean up when done

        LOG.info(
            "[OFFLINE] %s all epochs done: %d total trades, %d gradient steps",
            label, total_train_trades, total_train_steps,
        )

        # ── Validation pass ───────────────────────────────────────────────────
        # ── Validation pass (greedy: epsilon = epsilon_end so results are stable) ─
        # CRITICAL: reset stale position state from training. If the last epoch
        # ended with an open trade, policy.current_position stays non-zero,
        # causing trigger.decide_entry to return 0 for every val bar → val=0.
        score, val_trades = self._run_validation(policy, val_bars, label)

        # ── Save weights ───────────────────────────────────────────────────────
        weights_path = self._save_weights(policy, label)

        elapsed = time.perf_counter() - t0
        LOG.info("[OFFLINE] %s finished in %.1f s", label, elapsed)

        return TrainResult(
            symbol=self.symbol,
            timeframe_minutes=self.timeframe_minutes,
            z_omega=score,
            train_trades=total_train_trades,
            val_trades=val_trades,
            total_train_steps=total_train_steps,
            elapsed_s=elapsed,
            weights_path=weights_path,
        )

    def _set_offline_env_vars(self, os_module) -> tuple[str | None, str | None, str | None]:
        """Set offline training env vars and return original values."""
        orig_eps_start = os_module.environ.get("EPSILON_START")
        orig_eps_end = os_module.environ.get("EPSILON_END")
        orig_disable_gates = os_module.environ.get("DISABLE_GATES")
        os_module.environ["EPSILON_START"] = str(self.epsilon_start)
        os_module.environ["EPSILON_END"] = str(self.epsilon_end)
        os_module.environ["DISABLE_GATES"] = "1"  # bypass feasibility/conf gates
        return orig_eps_start, orig_eps_end, orig_disable_gates

    def _run_training_epochs(
        self,
        policy,
        train_bars: list,
        label: str,
        progress_path: Path,
        total_bars_all_epochs: int,
        progress_every: int,
    ) -> tuple[int, int]:
        """Run training epochs and return (total_train_steps, total_train_trades)."""
        total_train_steps = 0
        total_train_trades = 0
        for epoch in range(self.n_epochs):
            epoch_eps_start = self.epsilon_start * (0.7 ** epoch)
            epoch_eps_start = max(epoch_eps_start, self.epsilon_end * 2)
            policy.trigger.epsilon = epoch_eps_start
            if hasattr(policy, "harvester") and hasattr(policy.harvester, "epsilon"):
                policy.harvester.epsilon = epoch_eps_start

            LOG.info(
                "[OFFLINE] %s epoch %d/%d  ε_start=%.3f",
                label, epoch + 1, self.n_epochs, epoch_eps_start,
            )

            policy.current_position = 0
            sim = _Simulator(policy, update_policy=True, symbol_digits=self.symbol_digits,
                             event_engine=self._event_engine,
                             reward_clip_harvester=self._reward_clip_harvester,
                             reward_clip_trigger=self._reward_clip_trigger,
                             capture_baseline=self._capture_baseline,
                             symbol=self.symbol,
                             timeframe=f"M{self.timeframe_minutes}",
                             penalty_scale=self._penalty_scale)
            epoch_bar_offset = epoch * len(train_bars)

            for i, bar in enumerate(train_bars):
                sim.step(bar, i)
                total_train_steps = self._maybe_train_step(policy, i, total_train_steps)
                self._maybe_emit_progress(
                    policy,
                    sim,
                    i,
                    epoch,
                    epoch_bar_offset,
                    total_train_trades,
                    total_train_steps,
                    progress_path,
                    total_bars_all_epochs,
                    progress_every,
                )

            total_train_trades += len(sim.trades)
            LOG.info(
                "[OFFLINE] %s epoch %d/%d done: %d trades, %d gradient steps",
                label, epoch + 1, self.n_epochs, len(sim.trades), total_train_steps,
            )
        return total_train_steps, total_train_trades

    def _maybe_train_step(self, policy, bar_idx: int, total_train_steps: int) -> int:
        if bar_idx % self.train_every != 0 or bar_idx == 0:
            return total_train_steps
        try:
            policy.train_step()
            return total_train_steps + 1
        except Exception as exc:
            LOG.debug("[OFFLINE] train_step failed at bar %d: %s", bar_idx, exc)
            return total_train_steps

    def _maybe_emit_progress(
        self,
        policy,
        sim: _Simulator,
        bar_idx: int,
        epoch: int,
        epoch_bar_offset: int,
        total_train_trades: int,
        total_train_steps: int,
        progress_path: Path,
        total_bars_all_epochs: int,
        progress_every: int,
    ) -> None:
        if bar_idx % progress_every != 0:
            return
        _trig = policy.trigger
        _harv = policy.harvester
        _global_bar = epoch_bar_offset + bar_idx
        try:
            tmp_path = progress_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps({
                "symbol": self.symbol,
                "timeframe_minutes": self.timeframe_minutes,
                "bar": _global_bar,
                "total_bars": total_bars_all_epochs,
                "pct": round(_global_bar / total_bars_all_epochs * 100, 1),
                "epoch": epoch + 1,
                "n_epochs": self.n_epochs,
                "train_steps": total_train_steps,
                "trades": total_train_trades + len(sim.trades),
                "epsilon": round(float(_trig.epsilon), 4),
                "beta": round(float(_harv.buffer.beta) if _harv.buffer else 0.4, 4),
                "trigger_buf": int(_trig.buffer.size) if _trig.buffer else 0,
                "harvester_buf": int(_harv.buffer.size) if _harv.buffer else 0,
            }))
            tmp_path.replace(progress_path)
        except Exception:
            LOG.debug("[OFFLINE] Failed to write progress file: %s", progress_path, exc_info=True)

    def _prepare_validation_policy(self, policy) -> None:
        """Reset policy for validation pass."""
        policy.current_position = 0
        policy.trigger.epsilon = self.epsilon_end
        if hasattr(policy, "harvester") and hasattr(policy.harvester, "epsilon"):
            policy.harvester.epsilon = self.epsilon_end

    def _run_validation(self, policy, val_bars: list, label: str) -> tuple[float, int]:
        """Run validation pass and return (score, trade_count)."""
        self._prepare_validation_policy(policy)
        val_sim = _Simulator(policy, update_policy=False, symbol_digits=self.symbol_digits,
                             event_engine=self._event_engine,
                             reward_clip_harvester=self._reward_clip_harvester,
                             reward_clip_trigger=self._reward_clip_trigger,
                             capture_baseline=self._capture_baseline,
                             symbol=self.symbol,
                             timeframe=f"M{self.timeframe_minutes}",
                             penalty_scale=self._penalty_scale)
        for i, bar in enumerate(val_bars):
            val_sim.step(bar, i)
        val_pnl = [t.pnl_pts for t in val_sim.trades]
        score = z_omega(val_pnl)
        LOG.info(
            "[OFFLINE] %s val: %d trades, ZOmega=%.4f",
            label, len(val_sim.trades), score,
        )
        return score, len(val_sim.trades)

    def _save_weights(self, policy, label: str) -> str:
        """Save DDQN weights for both agents; return a summary path string."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        for agent_name, agent in [("trigger", policy.trigger), ("harvester", policy.harvester)]:
            if agent.ddqn is None:
                continue
            out = self.checkpoint_dir / f"{label}_{agent_name}_offline.pt"
            try:
                agent.ddqn.save_weights(str(out))
                paths.append(str(out))
                LOG.info("[OFFLINE] Saved %s", out.name)
            except Exception as exc:
                LOG.warning("[OFFLINE] Could not save %s weights: %s", agent_name, exc)

        return ";".join(paths)
