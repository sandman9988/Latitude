"""
Tests for the unified MFE/MAE Calculator (src.utils.mfe_mae).

Covers:
- start / reset lifecycle
- update() for LONG and SHORT positions
- MFE accumulation (monotonically increasing)
- MAE accumulation (monotonically increasing)
- winner_to_loser detection and live re-evaluation
- Edge cases: zero entry, None price, direction validation
- Backward-compat: MFEMAETracker wrapper in ctrader_ddqn_paper.py
"""

import pytest

from src.utils.mfe_mae import MFEMAECalculator


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


class TestMFEMAELifecycle:
    """start → update → reset → start cycle."""

    def test_initial_state(self):
        calc = MFEMAECalculator()
        assert calc.entry_price is None
        assert calc.direction is None
        assert calc.mfe == 0.0
        assert calc.mae == 0.0
        assert calc.winner_to_loser is False

    def test_start_sets_fields(self):
        calc = MFEMAECalculator()
        calc.start(100.0, 1)
        assert calc.entry_price == 100.0
        assert calc.direction == 1
        assert calc.mfe == 0.0
        assert calc.mae == 0.0

    def test_reset_clears_everything(self):
        calc = MFEMAECalculator()
        calc.start(100.0, 1)
        calc.update(110.0)
        calc.reset()
        assert calc.entry_price is None
        assert calc.mfe == 0.0
        assert calc.mae == 0.0
        assert calc.winner_to_loser is False


# ---------------------------------------------------------------------------
# LONG position tracking
# ---------------------------------------------------------------------------


class TestLongPosition:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.calc = MFEMAECalculator()
        self.calc.start(100.0, 1)  # LONG at 100

    def test_price_above_entry(self):
        self.calc.update(105.0)
        assert self.calc.mfe == 5.0
        assert self.calc.mae == 0.0

    def test_price_below_entry(self):
        self.calc.update(95.0)
        assert self.calc.mfe == 0.0
        assert self.calc.mae == 5.0

    def test_mfe_is_monotonically_increasing(self):
        self.calc.update(110.0)  # MFE=10
        self.calc.update(105.0)  # price drops back
        assert self.calc.mfe == 10.0  # MFE stays at peak

    def test_mae_is_monotonically_increasing(self):
        self.calc.update(90.0)   # MAE=10
        self.calc.update(95.0)   # price recovers
        assert self.calc.mae == 10.0  # MAE stays at worst

    def test_mfe_and_mae_both_accumulate(self):
        self.calc.update(110.0)  # MFE=10
        self.calc.update(88.0)   # MAE=12
        self.calc.update(115.0)  # MFE=15
        self.calc.update(85.0)   # MAE=15
        assert self.calc.mfe == 15.0
        assert self.calc.mae == 15.0


# ---------------------------------------------------------------------------
# SHORT position tracking
# ---------------------------------------------------------------------------


class TestShortPosition:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.calc = MFEMAECalculator()
        self.calc.start(100.0, -1)  # SHORT at 100

    def test_price_below_entry_is_profit(self):
        self.calc.update(95.0)
        assert self.calc.mfe == 5.0
        assert self.calc.mae == 0.0

    def test_price_above_entry_is_loss(self):
        self.calc.update(105.0)
        assert self.calc.mfe == 0.0
        assert self.calc.mae == 5.0

    def test_mfe_monotonic_short(self):
        self.calc.update(90.0)   # MFE=10
        self.calc.update(95.0)   # price rebounds
        assert self.calc.mfe == 10.0

    def test_mae_monotonic_short(self):
        self.calc.update(110.0)  # MAE=10
        self.calc.update(105.0)  # price drops
        assert self.calc.mae == 10.0


# ---------------------------------------------------------------------------
# Winner-to-loser detection
# ---------------------------------------------------------------------------


class TestWinnerToLoser:
    def test_wtl_triggers_when_profitable_then_loss(self):
        calc = MFEMAECalculator()
        calc.start(100.0, 1)
        calc.update(110.0)  # was profitable
        calc.update(95.0)   # now losing
        assert calc.winner_to_loser is True

    def test_wtl_clears_on_recovery(self):
        calc = MFEMAECalculator()
        calc.start(100.0, 1)
        calc.update(110.0)  # profit
        calc.update(95.0)   # loss → WTL
        calc.update(105.0)  # recovery → WTL clears
        assert calc.winner_to_loser is False

    def test_wtl_never_set_if_never_profitable(self):
        calc = MFEMAECalculator()
        calc.start(100.0, 1)
        calc.update(90.0)
        assert calc.winner_to_loser is False

    def test_wtl_short_position(self):
        calc = MFEMAECalculator()
        calc.start(100.0, -1)
        calc.update(90.0)   # profit for short
        calc.update(105.0)  # loss → WTL
        assert calc.winner_to_loser is True


# ---------------------------------------------------------------------------
# Edge cases / defensive behaviour
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_start_with_zero_entry_does_not_crash(self):
        calc = MFEMAECalculator()
        calc.start(0.0, 1)
        assert calc.entry_price is None  # rejected

    def test_start_with_none_entry(self):
        calc = MFEMAECalculator()
        calc.start(None, 1)
        assert calc.entry_price is None

    def test_start_with_invalid_direction_defaults_long(self):
        calc = MFEMAECalculator()
        calc.start(100.0, 0)
        assert calc.direction == 1  # defaulted

    def test_update_before_start_is_noop(self):
        calc = MFEMAECalculator()
        calc.update(105.0)  # no crash
        assert calc.mfe == 0.0

    def test_update_with_none_price_is_noop(self):
        calc = MFEMAECalculator()
        calc.start(100.0, 1)
        calc.update(None)
        assert calc.mfe == 0.0

    def test_update_with_zero_price_is_noop(self):
        calc = MFEMAECalculator()
        calc.start(100.0, 1)
        calc.update(0.0)
        assert calc.mfe == 0.0

    def test_get_summary_includes_all_keys(self):
        calc = MFEMAECalculator()
        calc.start(100.0, 1)
        calc.update(110.0)
        s = calc.get_summary()
        assert set(s.keys()) == {
            "entry_price", "direction", "mfe", "mae",
            "best_profit", "worst_loss", "winner_to_loser",
        }
        assert s["direction"] == "LONG"
        assert s["mfe"] == 10.0


# ---------------------------------------------------------------------------
# MFEMAETracker wrapper (backward compat)
# ---------------------------------------------------------------------------


class TestMFEMAETrackerWrapper:
    """Ensure the wrapper in ctrader_ddqn_paper delegates correctly."""

    def test_tracker_delegates_start_tracking(self):
        from src.core.ctrader_ddqn_paper import MFEMAETracker

        t = MFEMAETracker(position_id="test-123")
        t.start_tracking(100.0, -1)
        assert t.entry_price == 100.0
        assert t.direction == -1

    def test_tracker_delegates_update(self):
        from src.core.ctrader_ddqn_paper import MFEMAETracker

        t = MFEMAETracker()
        t.start_tracking(100.0, 1)
        t.update(110.0)
        assert t.mfe == 10.0
        assert t.mae == 0.0

    def test_tracker_reset(self):
        from src.core.ctrader_ddqn_paper import MFEMAETracker

        t = MFEMAETracker()
        t.start_tracking(100.0, 1)
        t.update(110.0)
        t.reset()
        assert t.mfe == 0.0

    def test_tracker_get_summary(self):
        from src.core.ctrader_ddqn_paper import MFEMAETracker

        t = MFEMAETracker()
        t.start_tracking(100.0, 1)
        t.update(105.0)
        s = t.get_summary()
        assert s["mfe"] == 5.0
        assert s["direction"] == "LONG"

    def test_tracker_winner_to_loser(self):
        from src.core.ctrader_ddqn_paper import MFEMAETracker

        t = MFEMAETracker()
        t.start_tracking(100.0, 1)
        t.update(110.0)
        t.update(95.0)
        assert t.winner_to_loser is True
        t.update(105.0)
        assert t.winner_to_loser is False

    def test_tracker_best_profit_worst_loss_setters(self):
        """Setters required for recovery from persisted state."""
        from src.core.ctrader_ddqn_paper import MFEMAETracker

        t = MFEMAETracker(position_id="recovery-test")
        t.start_tracking(100.0, -1)
        # Simulate recovery: set best_profit and worst_loss from persisted data
        t.best_profit = 5.5
        t.worst_loss = -3.2
        assert t.best_profit == 5.5
        assert t.worst_loss == -3.2
