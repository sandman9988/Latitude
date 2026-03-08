"""
Unified MFE / MAE calculator — single source of truth.

Used by:
- MFEMAETracker (live tick-level tracking in ctrader_ddqn_paper.py)
- DualPolicy._update_mfe_mae (per-bar tracking)
- offline_trainer._Simulator._update_mfe_mae (offline simulation)
"""

from __future__ import annotations

import logging

LOG = logging.getLogger(__name__)

# Percentage scaling factor (multiply by 100 for display).
_PCT = 100.0


class MFEMAECalculator:
    """Stateful MFE / MAE tracker.

    Attributes:
        mfe: Maximum Favorable Excursion (≥ 0, absolute price units)
        mae: Maximum Adverse Excursion (≥ 0, absolute price units)
        best_profit: Same as mfe (kept for backward-compat)
        worst_loss: Signed worst loss (≤ 0, absolute price units)
        winner_to_loser: True when trade was profitable but current PnL < 0
    """

    __slots__ = (
        "entry_price",
        "direction",
        "mfe",
        "mae",
        "best_profit",
        "worst_loss",
        "winner_to_loser",
    )

    def __init__(self) -> None:
        self.reset()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self, entry_price: float, direction: int) -> None:
        """Begin tracking a new position.

        Args:
            entry_price: Trade entry price.
            direction: 1 for LONG, -1 for SHORT.
        """
        try:
            if entry_price is None or entry_price <= 0:
                LOG.error("[MFE_MAE] Invalid entry_price=%.5f — cannot track",
                          entry_price or 0)
                return
        except TypeError:
            LOG.error("[MFE_MAE] Non-numeric entry_price=%r — cannot track",
                      entry_price)
            return
        if direction not in (1, -1):
            LOG.warning("[MFE_MAE] Invalid direction=%d — defaulting to LONG",
                        direction)
            direction = 1

        self.entry_price = float(entry_price)
        self.direction = int(direction)
        self.mfe = 0.0
        self.mae = 0.0
        self.best_profit = 0.0
        self.worst_loss = 0.0
        self.winner_to_loser = False

    def reset(self) -> None:
        """Clear all tracked values."""
        self.entry_price: float | None = None
        self.direction: int | None = None
        self.mfe = 0.0
        self.mae = 0.0
        self.best_profit = 0.0
        self.worst_loss = 0.0
        self.winner_to_loser = False

    # ── update ─────────────────────────────────────────────────────────────

    def update(self, current_price: float) -> None:
        """Update MFE/MAE from a new price tick.

        Args:
            current_price: Current market price.
        """
        if self.entry_price is None or self.entry_price <= 0:
            return
        try:
            if current_price is None or current_price <= 0:
                return
        except TypeError:
            return

        try:
            cp = float(current_price)
            ep = float(self.entry_price)
        except (TypeError, ValueError):
            return

        pnl = (cp - ep) if self.direction == 1 else (ep - cp)

        # MFE — best profit seen so far (≥ 0)
        if pnl > self.best_profit:
            self.best_profit = pnl
            self.mfe = pnl

        # MAE — worst drawdown (stored as positive magnitude)
        if pnl < self.worst_loss:
            self.worst_loss = pnl
            self.mae = abs(pnl)

        # Winner-to-Loser: live re-evaluation (clears when price recovers)
        if self.best_profit > 0:
            self.winner_to_loser = pnl < 0

    # ── convenience ────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Return a dict of current metrics."""
        return {
            "entry_price": self.entry_price,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "mfe": self.mfe,
            "mae": self.mae,
            "best_profit": self.best_profit,
            "worst_loss": self.worst_loss,
            "winner_to_loser": self.winner_to_loser,
        }
