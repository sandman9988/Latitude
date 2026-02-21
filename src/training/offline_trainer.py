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

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

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

# ── Types ──────────────────────────────────────────────────────────────────────

class TradeRecord(NamedTuple):
    entry_bar_idx: int
    exit_bar_idx: int
    direction: int      # 1 = LONG, -1 = SHORT
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
    if sig < 1e-10:
        # All returns identical — edge case, treat as neutral
        return 1.0

    z = arr / sig              # σ-normalise only; threshold in sigma units
    gains  = np.maximum(z - threshold, 0.0).sum()
    losses = np.maximum(threshold - z, 0.0).sum()

    if losses < 1e-10:
        return float("inf")
    return float(gains / losses)


# ── Simulation ────────────────────────────────────────────────────────────────

class _Simulator:
    """
    Stateful bar-by-bar walker.

    Calls DualPolicy entry/exit methods and accumulates experiences.
    Does NOT call train_step — the caller decides when to update.
    """

    def __init__(self, policy, update_policy: bool = True) -> None:
        self.policy = policy
        self.update_policy = update_policy  # False during validation pass
        self.bars: deque = deque(maxlen=DEQUE_MAXLEN)

        # Position state
        self.cur_pos: int = 0           # 0=flat, 1=long, -1=short
        self.entry_price: float = 0.0
        self.entry_bar_idx: int = 0
        self.entry_action: int = 0
        self.mfe: float = 0.0
        self.mae: float = 0.0
        self.ticks_held: int = 0

        # State snapshots for experience labelling
        self.entry_state: np.ndarray | None = None

        # Accumulated result
        self.trades: list[TradeRecord] = []

    def step(self, bar: tuple, bar_idx: int) -> None:
        """Process one bar."""
        self.bars.append(bar)

        if len(self.bars) < MIN_BARS_FOR_ENTRY:
            return

        current_price = float(bar[4])  # close

        if self.cur_pos == 0:
            self._try_entry(bar_idx, current_price)
        else:
            self._update_mfe_mae(current_price)
            self._try_exit(bar_idx, current_price)

    def _try_entry(self, bar_idx: int, current_price: float) -> None:
        try:
            action, conf, runway = self.policy.decide_entry(
                self.bars, imbalance=0.0, vpin_z=0.0, depth_ratio=1.0
            )
        except Exception as exc:
            LOG.debug("[SIM] decide_entry failed at bar %d: %s", bar_idx, exc)
            return

        if action not in (1, 2):
            return

        direction = 1 if action == 1 else -1
        self.cur_pos = direction
        self.entry_price = current_price
        self.entry_bar_idx = bar_idx
        self.entry_action = action
        self.mfe = 0.0
        self.mae = 0.0
        self.ticks_held = 0

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
                action, _conf = self.policy.decide_exit(
                    self.bars,
                    current_price=current_price,
                    imbalance=0.0,
                    vpin_z=0.0,
                    depth_ratio=1.0,
                )
            except Exception as exc:
                LOG.debug("[SIM] decide_exit failed: %s", exc)
                action = 0

        if action == 1 or hard_stop:
            self._close_position(bar_idx, current_price, forced=hard_stop)

    def _update_mfe_mae(self, current_price: float) -> None:
        if self.entry_price <= 0:
            return
        excursion = (current_price - self.entry_price) * self.cur_pos
        if excursion > 0:
            self.mfe = max(self.mfe, excursion)
        else:
            self.mae = max(self.mae, -excursion)

    def _close_position(self, bar_idx: int, exit_price: float, forced: bool = False) -> None:
        pnl_pts = (exit_price - self.entry_price) * self.cur_pos
        capture = self.mfe > 0 and pnl_pts > 0
        capture_ratio = (pnl_pts / self.mfe) if self.mfe > 1e-8 else 0.0

        # Build rewards using the same formula as the live bot
        # Trigger reward: normalised outcome (no prediction accuracy in offline)
        vol_pts = max(self.entry_price * 0.005, 1e-6)  # ~0.5% default vol
        trigger_reward = float(np.clip(pnl_pts / vol_pts / 3.0,
                                       -TRIGGER_REWARD_CLIP, TRIGGER_REWARD_CLIP))

        # Capture reward: fraction of MFE captured, clipped
        capture_reward = float(np.clip(capture_ratio - 0.5, -REWARD_CLIP, REWARD_CLIP))

        if self.update_policy and self.entry_state is not None:
            self._add_experiences(
                trigger_reward=trigger_reward,
                capture_reward=capture_reward,
                exit_price=exit_price,
            )

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

        try:
            self.policy.on_exit(
                exit_price=exit_price,
                capture_ratio=capture_ratio,
                was_wtl=not capture,
            )
        except Exception:
            pass

        self.cur_pos = 0
        self.entry_price = 0.0
        self.entry_state = None
        self.ticks_held = 0
        self.mfe = 0.0
        self.mae = 0.0

    def _add_experiences(
        self, trigger_reward: float, capture_reward: float, exit_price: float
    ) -> None:
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
    """

    def __init__(
        self,
        symbol: str,
        timeframe_minutes: int,
        bars: list,
        checkpoint_dir: str | Path = "data/checkpoints",
        train_split: float = TRAIN_SPLIT,
        train_every: int = TRAIN_EVERY,
        policy_kwargs: dict | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe_minutes = timeframe_minutes
        self.bars = bars
        self.checkpoint_dir = Path(checkpoint_dir)
        self.train_split = train_split
        self.train_every = train_every
        self.policy_kwargs = policy_kwargs or {}

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

    def _run_inner(self, label: str, t0: float) -> TrainResult:
        # Lazy import to avoid circular imports and allow multiprocessing fork
        from src.agents.dual_policy import DualPolicy

        n_total = len(self.bars)
        n_train = int(n_total * self.train_split)
        train_bars = self.bars[:n_train]
        val_bars   = self.bars[n_train:]

        LOG.info(
            "[OFFLINE] %s: %d train bars / %d val bars",
            label, len(train_bars), len(val_bars),
        )

        # Scale PER buffer to training set size: up to half the training bars,
        # capped at 20k to avoid unbounded memory. Explicit policy_kwargs override wins.
        offline_buffer_capacity = min(max(len(train_bars) // 2, 2_000), 20_000)
        policy_kwargs = {
            "trigger_buffer_capacity": offline_buffer_capacity,
            "harvester_buffer_capacity": offline_buffer_capacity,
            **self.policy_kwargs,
        }

        # Build policy — training enabled, no checkpoint loading (start fresh)
        policy = DualPolicy(
            symbol=self.symbol,
            timeframe=f"M{self.timeframe_minutes}",
            window=64,
            enable_training=True,
            enable_event_features=False,   # event features require live-time context
            timeframe_minutes=self.timeframe_minutes,
            **policy_kwargs,
        )

        # ── Training pass ─────────────────────────────────────────────────────
        sim = _Simulator(policy, update_policy=True)
        total_train_steps = 0
        for i, bar in enumerate(train_bars):
            sim.step(bar, i)
            if i % self.train_every == 0 and i > 0:
                try:
                    policy.train_step()
                    total_train_steps += 1
                except Exception as exc:
                    LOG.debug("[OFFLINE] train_step failed at bar %d: %s", i, exc)

        LOG.info(
            "[OFFLINE] %s train done: %d trades, %d gradient steps",
            label, len(sim.trades), total_train_steps,
        )

        # ── Validation pass ───────────────────────────────────────────────────
        val_sim = _Simulator(policy, update_policy=False)
        for i, bar in enumerate(val_bars):
            val_sim.step(bar, i)

        val_pnl = [t.pnl_pts for t in val_sim.trades]
        score = z_omega(val_pnl)

        LOG.info(
            "[OFFLINE] %s val: %d trades, ZOmega=%.4f",
            label, len(val_sim.trades), score,
        )

        # ── Save weights ───────────────────────────────────────────────────────
        weights_path = self._save_weights(policy, label)

        elapsed = time.perf_counter() - t0
        LOG.info("[OFFLINE] %s finished in %.1f s", label, elapsed)

        return TrainResult(
            symbol=self.symbol,
            timeframe_minutes=self.timeframe_minutes,
            z_omega=score,
            train_trades=len(sim.trades),
            val_trades=len(val_sim.trades),
            total_train_steps=total_train_steps,
            elapsed_s=elapsed,
            weights_path=weights_path,
        )

    def _save_weights(self, policy, label: str) -> str:
        """Save DDQN weights for both agents; return a summary path string."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        for agent_name, agent in [("trigger", policy.trigger), ("harvester", policy.harvester)]:
            if agent.ddqn is None:
                continue
            out = self.checkpoint_dir / f"{label}_{agent_name}_offline.npz"
            try:
                agent.ddqn.save_weights(str(out))
                paths.append(str(out))
                LOG.info("[OFFLINE] Saved %s", out.name)
            except Exception as exc:
                LOG.warning("[OFFLINE] Could not save %s weights: %s", agent_name, exc)

        return ";".join(paths)
