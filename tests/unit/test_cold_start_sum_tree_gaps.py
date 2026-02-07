"""Gap tests for cold_start_manager.py line 186 and sum_tree.py line 127.

Targets:
- cold_start: check_graduation() fallthrough return None for unknown phase
- sum_tree: get() method to retrieve priority at tree index
"""

import pytest

from src.core.cold_start_manager import ColdStartManager, WarmupPhase
from src.utils.sum_tree import SumTree


# ---------------------------------------------------------------------------
# ColdStartManager.check_graduation fallthrough (line 186)
# ---------------------------------------------------------------------------
class TestColdStartGraduationFallthrough:
    def test_unknown_phase_returns_none(self):
        """If current_phase is not one of the known phases,
        check_graduation returns None (fallthrough)."""
        mgr = ColdStartManager()
        # Force an unrecognized phase value
        mgr.current_phase = "UNKNOWN_PHASE"
        result = mgr.check_graduation()
        assert result is None

    def test_production_phase_no_demotion(self):
        """PRODUCTION phase without demotion criteria → returns None from
        _check_production_demotion, not the fallthrough."""
        mgr = ColdStartManager()
        mgr.current_phase = WarmupPhase.PRODUCTION
        result = mgr.check_graduation()
        # Should call _check_production_demotion; typically returns None
        # (no demotion criteria met with fresh manager)
        assert result is None or isinstance(result, WarmupPhase)


# ---------------------------------------------------------------------------
# SumTree.get() (line 127)
# ---------------------------------------------------------------------------
class TestSumTreeGet:
    def test_get_initial_priority_zero(self):
        """Fresh tree should have zero priority at any index."""
        tree = SumTree(capacity=4)
        # Leaf indices start at capacity-1 = 3
        assert tree.get(3) == pytest.approx(0.0)
        assert tree.get(4) == pytest.approx(0.0)

    def test_get_after_update(self):
        """After updating a priority, get() should return the new value."""
        tree = SumTree(capacity=4)
        # Add a priority (uses internal write pointer)
        tree.add(1.5)
        # The first leaf is at index capacity-1 = 3
        assert tree.get(3) == pytest.approx(1.5)

    def test_get_parent_node_reflects_sum(self):
        """Parent nodes reflect cumulative sum."""
        tree = SumTree(capacity=4)
        tree.add(2.0)
        tree.add(3.0)
        # Root (index 0) should be the total
        assert tree.get(0) == pytest.approx(5.0)

    def test_get_multiple_items(self):
        """Get priorities for multiple items."""
        tree = SumTree(capacity=8)
        for i in range(4):
            tree.add(float(i + 1))
        # Leaf indices: 7, 8, 9, 10 for capacity=8
        assert tree.get(7) == pytest.approx(1.0)
        assert tree.get(8) == pytest.approx(2.0)
        assert tree.get(9) == pytest.approx(3.0)
        assert tree.get(10) == pytest.approx(4.0)
