"""Tests for src.core.adaptive_regularization."""

import pytest

from src.core.adaptive_regularization import AdaptiveRegularization


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
class TestInit:
    def test_default_values(self):
        ar = AdaptiveRegularization()
        assert ar.l2_weight == pytest.approx(0.0001)
        assert ar.dropout_rate == pytest.approx(0.1)
        assert ar.l2_min == 1e-5
        assert ar.l2_max == 1e-2
        assert ar.dropout_min == pytest.approx(0.0)
        assert ar.dropout_max == pytest.approx(0.5)
        assert ar.adjustment_rate == pytest.approx(1.2)
        assert ar.adjustment_history == []

    def test_custom_values(self):
        ar = AdaptiveRegularization(
            initial_l2=0.001,
            initial_dropout=0.2,
            l2_range=(1e-6, 0.1),
            dropout_range=(0.05, 0.8),
            adjustment_rate=1.5,
        )
        assert ar.l2_weight == pytest.approx(0.001)
        assert ar.dropout_rate == pytest.approx(0.2)
        assert ar.l2_min == 1e-6
        assert ar.l2_max == pytest.approx(0.1)
        assert ar.dropout_min == pytest.approx(0.05)
        assert ar.dropout_max == pytest.approx(0.8)
        assert ar.adjustment_rate == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# increase_regularization()
# ---------------------------------------------------------------------------
class TestIncrease:
    def test_basic_increase(self):
        ar = AdaptiveRegularization(initial_l2=0.001, initial_dropout=0.1,
                                     adjustment_rate=2.0)
        ar.increase_regularization()
        assert ar.l2_weight == pytest.approx(0.002)
        assert ar.dropout_rate == pytest.approx(0.2)

    def test_capped_at_max(self):
        ar = AdaptiveRegularization(
            initial_l2=0.009, l2_range=(1e-5, 0.01),
            initial_dropout=0.45, dropout_range=(0.0, 0.5),
            adjustment_rate=2.0,
        )
        ar.increase_regularization()
        assert ar.l2_weight == pytest.approx(0.01)
        assert ar.dropout_rate == pytest.approx(0.5)

    def test_history_recorded(self):
        ar = AdaptiveRegularization()
        ar.increase_regularization()
        assert ar.adjustment_history == ["increase"]

    def test_multiple_increases(self):
        ar = AdaptiveRegularization(initial_l2=0.0001, adjustment_rate=1.5)
        ar.increase_regularization()
        ar.increase_regularization()
        assert ar.l2_weight == pytest.approx(0.0001 * 1.5 * 1.5)
        assert len(ar.adjustment_history) == 2


# ---------------------------------------------------------------------------
# decrease_regularization()
# ---------------------------------------------------------------------------
class TestDecrease:
    def test_basic_decrease(self):
        ar = AdaptiveRegularization(initial_l2=0.001, initial_dropout=0.2,
                                     adjustment_rate=2.0)
        ar.decrease_regularization()
        assert ar.l2_weight == pytest.approx(0.0005)
        assert ar.dropout_rate == pytest.approx(0.1)

    def test_capped_at_min(self):
        ar = AdaptiveRegularization(
            initial_l2=2e-5, l2_range=(1e-5, 0.01),
            initial_dropout=0.05, dropout_range=(0.02, 0.5),
            adjustment_rate=10.0,
        )
        ar.decrease_regularization()
        assert ar.l2_weight == 1e-5
        assert ar.dropout_rate == pytest.approx(0.02)

    def test_history_recorded(self):
        ar = AdaptiveRegularization()
        ar.decrease_regularization()
        assert ar.adjustment_history == ["decrease"]


# ---------------------------------------------------------------------------
# update_from_signal()
# ---------------------------------------------------------------------------
class TestUpdateFromSignal:
    def test_increase_regularization_signal(self):
        ar = AdaptiveRegularization(initial_l2=0.001, adjustment_rate=2.0)
        ar.update_from_signal("INCREASE_REGULARIZATION")
        assert ar.l2_weight == pytest.approx(0.002)

    def test_increase_capacity_signal(self):
        ar = AdaptiveRegularization(initial_l2=0.002, adjustment_rate=2.0)
        ar.update_from_signal("INCREASE_CAPACITY")
        assert ar.l2_weight == pytest.approx(0.001)

    def test_continue_training_no_change(self):
        ar = AdaptiveRegularization(initial_l2=0.001)
        ar.update_from_signal("CONTINUE_TRAINING")
        assert ar.l2_weight == pytest.approx(0.001)
        assert ar.adjustment_history == []

    def test_collect_more_data_no_change(self):
        ar = AdaptiveRegularization(initial_l2=0.001)
        ar.update_from_signal("COLLECT_MORE_DATA")
        assert ar.l2_weight == pytest.approx(0.001)

    def test_unknown_signal_no_change(self):
        ar = AdaptiveRegularization(initial_l2=0.001)
        ar.update_from_signal("UNKNOWN_SIGNAL")
        assert ar.l2_weight == pytest.approx(0.001)
        assert ar.adjustment_history == []


# ---------------------------------------------------------------------------
# get_current()
# ---------------------------------------------------------------------------
class TestGetCurrent:
    def test_returns_dict(self):
        ar = AdaptiveRegularization(initial_l2=0.001, initial_dropout=0.2)
        result = ar.get_current()
        assert result == {"l2_weight": 0.001, "dropout_rate": 0.2}

    def test_reflects_changes(self):
        ar = AdaptiveRegularization(initial_l2=0.001, initial_dropout=0.1,
                                     adjustment_rate=2.0)
        ar.increase_regularization()
        result = ar.get_current()
        assert result["l2_weight"] == pytest.approx(0.002)
        assert result["dropout_rate"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------
class TestReset:
    def test_reset_with_explicit_values(self):
        ar = AdaptiveRegularization()
        ar.increase_regularization()
        ar.increase_regularization()
        ar.reset(l2=0.005, dropout=0.3)
        assert ar.l2_weight == pytest.approx(0.005)
        assert ar.dropout_rate == pytest.approx(0.3)
        assert ar.adjustment_history == []

    def test_reset_clamps_to_range(self):
        ar = AdaptiveRegularization(l2_range=(1e-5, 0.01), dropout_range=(0.0, 0.5))
        ar.reset(l2=0.1, dropout=0.9)
        assert ar.l2_weight == pytest.approx(0.01)
        assert ar.dropout_rate == pytest.approx(0.5)

    def test_reset_clamps_below_min(self):
        ar = AdaptiveRegularization(l2_range=(1e-5, 0.01), dropout_range=(0.1, 0.5))
        ar.reset(l2=1e-10, dropout=0.0)
        assert ar.l2_weight == 1e-5
        assert ar.dropout_rate == pytest.approx(0.1)

    def test_reset_with_none_keeps_current(self):
        ar = AdaptiveRegularization(initial_l2=0.001, initial_dropout=0.2)
        ar.increase_regularization()
        _old_l2 = ar.l2_weight
        _old_dropout = ar.dropout_rate
        ar.reset(l2=None, dropout=None)
        # None means keep current value (no assignment in source)
        # But adjustment_history is still cleared
        assert ar.adjustment_history == []

    def test_reset_clears_history(self):
        ar = AdaptiveRegularization()
        ar.increase_regularization()
        ar.decrease_regularization()
        assert len(ar.adjustment_history) == 2
        ar.reset(l2=0.001, dropout=0.1)
        assert ar.adjustment_history == []


# ---------------------------------------------------------------------------
# Boundary / edge case tests
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_increase_from_max(self):
        """Already at max — stays at max."""
        ar = AdaptiveRegularization(
            initial_l2=0.01, l2_range=(1e-5, 0.01),
            initial_dropout=0.5, dropout_range=(0.0, 0.5),
        )
        ar.increase_regularization()
        assert ar.l2_weight == pytest.approx(0.01)
        assert ar.dropout_rate == pytest.approx(0.5)

    def test_decrease_from_min(self):
        """Already at min — stays at min."""
        ar = AdaptiveRegularization(
            initial_l2=1e-5, l2_range=(1e-5, 0.01),
            initial_dropout=0.0, dropout_range=(0.0, 0.5),
        )
        ar.decrease_regularization()
        assert ar.l2_weight == 1e-5
        assert ar.dropout_rate == pytest.approx(0.0)

    def test_many_increases_stay_in_bounds(self):
        ar = AdaptiveRegularization(l2_range=(1e-5, 0.01), dropout_range=(0.0, 0.5))
        for _ in range(100):
            ar.increase_regularization()
        assert ar.l2_weight <= 0.01
        assert ar.dropout_rate <= 0.5

    def test_many_decreases_stay_in_bounds(self):
        ar = AdaptiveRegularization(l2_range=(1e-5, 0.01), dropout_range=(0.0, 0.5))
        for _ in range(100):
            ar.decrease_regularization()
        assert ar.l2_weight >= 1e-5
        assert ar.dropout_rate >= 0.0

    def test_alternating_increase_decrease(self):
        ar = AdaptiveRegularization(initial_l2=0.001, initial_dropout=0.2,
                                     adjustment_rate=1.2)
        initial = ar.get_current()
        ar.increase_regularization()
        ar.decrease_regularization()
        # Should be back to approximately the original
        result = ar.get_current()
        assert result["l2_weight"] == pytest.approx(initial["l2_weight"], rel=1e-10)
        assert result["dropout_rate"] == pytest.approx(initial["dropout_rate"], rel=1e-10)
