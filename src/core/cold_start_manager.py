"""
Cold Start Manager

Implements graduated warm-up protocol for safe RL agent deployment.

The cold start problem in RL trading:
- Random/untrained agents make catastrophic decisions
- Real money at risk from first trade onwards
- No chance to learn safely before going live

Solution: 3-phase warm-up with strict graduation criteria

Phase 1: OBSERVATION (100+ bars)
  - No trading allowed
  - Agents observe market, build feature distributions
  - Learn what "normal" looks like
  - Graduation: Collect 100 bars of clean data

Phase 2: PAPER TRADING (500+ bars)
  - Agents make decisions, but trades are virtual only
  - Full RL learning loop active
  - Track virtual P&L and performance
  - Graduation: Sharpe >0.3, WinRate >45%, Max DD <20%

Phase 3: MICRO POSITIONS (1000+ bars)
  - Real trades with tiny position sizes (QTY=0.001)
  - Allows learning from real fills, spreads, slippage
  - Limits financial risk during early learning
  - Graduation: Sharpe >0.5, WinRate >48%, Avg trade profit >0

Phase 4: PRODUCTION
  - Full position sizing enabled
  - Continuous monitoring for degradation
  - Can be demoted back to Phase 3 if performance collapses
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class WarmupPhase(Enum):
    """Warmup phases with increasing risk."""

    OBSERVATION = 1  # Watch only, no trades
    PAPER_TRADING = 2  # Virtual trades
    MICRO_POSITIONS = 3  # Real but tiny trades
    PRODUCTION = 4  # Full size trades


@dataclass
class PhaseMetrics:
    """Performance metrics for a warmup phase."""

    phase: WarmupPhase
    bars_completed: int
    trades_completed: int
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    avg_trade_profit: float
    total_pnl: float


@dataclass
class DemotionThresholds:
    """Thresholds that trigger production demotion."""

    sharpe: float = 0.2
    win_rate: float = 0.40
    drawdown: float = 0.30


class ColdStartManager:
    """
    Manages graduated warm-up protocol for RL trading agents.

    Phases progress linearly: OBSERVATION → PAPER → MICRO → PRODUCTION
    Can be demoted back to earlier phases if performance degrades.
    """

    def __init__(
        self,
        # Phase 1: Observation
        observation_min_bars: int = 100,
        # Phase 2: Paper trading
        paper_min_bars: int = 500,
        paper_min_sharpe: float = 0.3,
        paper_min_win_rate: float = 0.45,
        paper_max_drawdown: float = 0.20,
        # Phase 3: Micro positions
        micro_min_bars: int = 1000,
        micro_position_size: float = 0.001,  # Fraction of normal size
        micro_min_sharpe: float = 0.5,
        micro_min_win_rate: float = 0.48,
        micro_min_avg_profit: float = 0.0,  # At least break-even
        # Production monitoring
        demotion: DemotionThresholds | None = None,
        # State persistence
        state_file: Path | None = None,
    ):
        # Phase thresholds
        self.observation_min_bars = observation_min_bars
        self.paper_min_bars = paper_min_bars
        self.paper_min_sharpe = paper_min_sharpe
        self.paper_min_win_rate = paper_min_win_rate
        self.paper_max_drawdown = paper_max_drawdown
        self.micro_min_bars = micro_min_bars
        self.micro_position_size = micro_position_size
        self.micro_min_sharpe = micro_min_sharpe
        self.micro_min_win_rate = micro_min_win_rate
        self.micro_min_avg_profit = micro_min_avg_profit
        _demotion = demotion or DemotionThresholds()
        self.production_demotion_sharpe = _demotion.sharpe
        self.production_demotion_win_rate = _demotion.win_rate
        self.production_demotion_drawdown = _demotion.drawdown

        # Current state
        self.current_phase = WarmupPhase.OBSERVATION
        self.bars_in_current_phase = 0
        self.trades_in_current_phase = 0

        # Performance tracking
        self.phase_history: list[PhaseMetrics] = []

        # Temporary storage for current phase metrics
        self.current_trades: list[dict] = []
        self.current_pnls: list[float] = []

        # Persistence
        self.state_file = state_file or Path("data/cold_start_manager.json")

    def update(
        self,
        new_bar: bool = False,
        trade_completed: dict | None = None,
    ):
        """
        Update cold start state.

        Args:
            new_bar: True if new bar arrived
            trade_completed: Dict with trade info if trade just completed:
                - pnl: float
                - is_paper: bool (True if virtual trade)
        """
        if new_bar:
            self.bars_in_current_phase += 1

        if trade_completed:
            self.trades_in_current_phase += 1
            self.current_trades.append(trade_completed)
            self.current_pnls.append(trade_completed.get("pnl", 0.0))

    def can_trade(self) -> bool:
        """Returns True if trading is allowed in current phase."""
        return self.current_phase != WarmupPhase.OBSERVATION

    def is_paper_only(self) -> bool:
        """Returns True if only paper trading allowed."""
        return self.current_phase == WarmupPhase.PAPER_TRADING

    def get_position_size_multiplier(self) -> float:
        """Returns position size multiplier for current phase."""
        if self.current_phase == WarmupPhase.OBSERVATION:
            return 0.0  # No trading
        elif self.current_phase == WarmupPhase.PAPER_TRADING:
            return 0.0  # Paper trades don't affect real positions
        elif self.current_phase == WarmupPhase.MICRO_POSITIONS:
            return self.micro_position_size  # Tiny positions
        else:  # PRODUCTION
            return 1.0  # Full size

    def check_graduation(self) -> WarmupPhase | None:
        """
        Check if current phase can graduate to next phase.

        Returns next phase if graduation criteria met, None otherwise.
        """
        if self.current_phase == WarmupPhase.OBSERVATION:
            return self._check_observation_graduation()
        elif self.current_phase == WarmupPhase.PAPER_TRADING:
            return self._check_paper_graduation()
        elif self.current_phase == WarmupPhase.MICRO_POSITIONS:
            return self._check_micro_graduation()
        elif self.current_phase == WarmupPhase.PRODUCTION:
            return self._check_production_demotion()
        return None

    def _check_observation_graduation(self) -> WarmupPhase | None:
        """Check if observation phase can graduate."""
        if self.bars_in_current_phase >= self.observation_min_bars:
            logger.info(
                "OBSERVATION phase complete (%d bars). Graduating to PAPER_TRADING.", self.bars_in_current_phase
            )
            return WarmupPhase.PAPER_TRADING
        return None

    def _check_paper_graduation(self) -> WarmupPhase | None:
        """Check if paper trading phase can graduate."""
        if self.bars_in_current_phase < self.paper_min_bars:
            return None  # Not enough bars

        if self.trades_in_current_phase < 10:
            return None  # Not enough trades to evaluate

        # Calculate metrics
        metrics = self._calculate_current_metrics()

        # Check all criteria
        if (
            metrics.sharpe_ratio >= self.paper_min_sharpe
            and metrics.win_rate >= self.paper_min_win_rate
            and metrics.max_drawdown <= self.paper_max_drawdown
        ):
            logger.info(
                "PAPER_TRADING phase complete. "
                "Sharpe=%.2f, WinRate=%.1f%%, DD=%.1f%%. "
                "Graduating to MICRO_POSITIONS.",
                metrics.sharpe_ratio,
                metrics.win_rate * 100,
                metrics.max_drawdown * 100,
            )
            return WarmupPhase.MICRO_POSITIONS

        logger.info(
            "PAPER_TRADING not ready to graduate. "
            "Sharpe=%.2f (need %.2f), "
            "WinRate=%.1f%% (need %.1f%%), "
            "DD=%.1f%% (max %.1f%%)",
            metrics.sharpe_ratio,
            self.paper_min_sharpe,
            metrics.win_rate * 100,
            self.paper_min_win_rate * 100,
            metrics.max_drawdown * 100,
            self.paper_max_drawdown * 100,
        )
        return None

    def _check_micro_graduation(self) -> WarmupPhase | None:
        """Check if micro positions phase can graduate."""
        if self.bars_in_current_phase < self.micro_min_bars:
            return None

        if self.trades_in_current_phase < 20:
            return None  # Need more real trades

        metrics = self._calculate_current_metrics()

        if (
            metrics.sharpe_ratio >= self.micro_min_sharpe
            and metrics.win_rate >= self.micro_min_win_rate
            and metrics.avg_trade_profit >= self.micro_min_avg_profit
        ):
            logger.warning(
                "MICRO_POSITIONS phase complete. "
                "Sharpe=%.2f, WinRate=%.1f%%, "
                "AvgProfit=%.4f. "
                "GRADUATING TO PRODUCTION!",
                metrics.sharpe_ratio,
                metrics.win_rate * 100,
                metrics.avg_trade_profit,
            )
            return WarmupPhase.PRODUCTION

        logger.info(
            "MICRO_POSITIONS not ready to graduate. "
            "Sharpe=%.2f (need %.2f), "
            "WinRate=%.1f%% (need %.1f%%), "
            "AvgProfit=%.4f (need %.4f)",
            metrics.sharpe_ratio,
            self.micro_min_sharpe,
            metrics.win_rate * 100,
            self.micro_min_win_rate * 100,
            metrics.avg_trade_profit,
            self.micro_min_avg_profit,
        )
        return None

    def _check_production_demotion(self) -> WarmupPhase | None:
        """
        Check if production should be demoted back to micro positions.

        Returns MICRO_POSITIONS if demotion needed, None otherwise.
        """
        if self.trades_in_current_phase < 50:
            return None  # Not enough data to judge

        metrics = self._calculate_current_metrics()

        # Check for catastrophic performance
        if (
            metrics.sharpe_ratio < self.production_demotion_sharpe
            or metrics.win_rate < self.production_demotion_win_rate
            or metrics.max_drawdown > self.production_demotion_drawdown
        ):
            logger.error(
                "PRODUCTION DEMOTION! "
                "Performance degraded: Sharpe=%.2f, "
                "WinRate=%.1f%%, DD=%.1f%%. "
                "Demoting to MICRO_POSITIONS.",
                metrics.sharpe_ratio,
                metrics.win_rate * 100,
                metrics.max_drawdown * 100,
            )
            return WarmupPhase.MICRO_POSITIONS

        return None

    def _calculate_current_metrics(self) -> PhaseMetrics:
        """Calculate metrics for current phase."""
        if not self.current_pnls:
            return PhaseMetrics(
                phase=self.current_phase,
                bars_completed=self.bars_in_current_phase,
                trades_completed=self.trades_in_current_phase,
                sharpe_ratio=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                avg_trade_profit=0.0,
                total_pnl=0.0,
            )

        # Sharpe ratio
        pnl_array = np.array(self.current_pnls)
        sharpe = 0.0
        if np.std(pnl_array) > 0:
            sharpe = np.mean(pnl_array) / (np.std(pnl_array) + 1e-8)

        # Win rate
        wins = sum(p > 0 for p in self.current_pnls)
        win_rate = wins / len(self.current_pnls) if self.current_pnls else 0.0

        # Max drawdown
        cumulative = np.cumsum(pnl_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (np.abs(running_max) + 1e-8)
        max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # Average trade profit
        avg_profit = np.mean(pnl_array)

        # Total P&L
        total = np.sum(pnl_array)

        return PhaseMetrics(
            phase=self.current_phase,
            bars_completed=self.bars_in_current_phase,
            trades_completed=self.trades_in_current_phase,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            max_drawdown=max_dd,
            avg_trade_profit=avg_profit,
            total_pnl=total,
        )

    def graduate(self, new_phase: WarmupPhase):
        """
        Graduate to new phase.

        Saves metrics from current phase and resets counters.
        """
        # Save current phase metrics
        final_metrics = self._calculate_current_metrics()
        self.phase_history.append(final_metrics)

        # Reset counters
        self.current_phase = new_phase
        self.bars_in_current_phase = 0
        self.trades_in_current_phase = 0
        self.current_trades = []
        self.current_pnls = []

        logger.warning("Phase transition to %s", new_phase.name)

    def get_status(self) -> dict:
        """Get current warmup status."""
        current_metrics = self._calculate_current_metrics()

        return {
            "current_phase": self.current_phase.name,
            "bars_in_phase": self.bars_in_current_phase,
            "trades_in_phase": self.trades_in_current_phase,
            "can_trade": self.can_trade(),
            "is_paper_only": self.is_paper_only(),
            "position_size_multiplier": self.get_position_size_multiplier(),
            "current_metrics": {
                "sharpe": current_metrics.sharpe_ratio,
                "win_rate": current_metrics.win_rate,
                "max_drawdown": current_metrics.max_drawdown,
                "avg_profit": current_metrics.avg_trade_profit,
                "total_pnl": current_metrics.total_pnl,
            },
            "phase_history_count": len(self.phase_history),
        }

    def save_state(self):
        """Persist state to disk."""
        state = {
            "current_phase": self.current_phase.name,
            "bars_in_current_phase": self.bars_in_current_phase,
            "trades_in_current_phase": self.trades_in_current_phase,
            "current_trades": self.current_trades[-100:],  # Keep last 100
            "current_pnls": self.current_pnls[-100:],
            "phase_history": [
                {
                    "phase": m.phase.name,
                    "bars": m.bars_completed,
                    "trades": m.trades_completed,
                    "sharpe": m.sharpe_ratio,
                    "win_rate": m.win_rate,
                    "max_dd": m.max_drawdown,
                    "avg_profit": m.avg_trade_profit,
                    "total_pnl": m.total_pnl,
                }
                for m in self.phase_history
            ],
        }

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> bool:
        """Load state from disk."""
        if not self.state_file.exists():
            return False

        try:
            with open(self.state_file, encoding="utf-8") as f:
                state = json.load(f)

            self.current_phase = WarmupPhase[state["current_phase"]]
            self.bars_in_current_phase = state.get("bars_in_current_phase", 0)
            self.trades_in_current_phase = state.get("trades_in_current_phase", 0)
            self.current_trades = state.get("current_trades", [])
            self.current_pnls = state.get("current_pnls", [])

            # Reconstruct phase history
            self.phase_history = [
                PhaseMetrics(
                    phase=WarmupPhase[h["phase"]],
                    bars_completed=h["bars"],
                    trades_completed=h["trades"],
                    sharpe_ratio=h["sharpe"],
                    win_rate=h["win_rate"],
                    max_drawdown=h["max_dd"],
                    avg_trade_profit=h["avg_profit"],
                    total_pnl=h["total_pnl"],
                )
                for h in state.get("phase_history", [])
            ]

            logger.info("Loaded ColdStartManager state: Phase=%s", self.current_phase.name)
            return True

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error("Failed to load state: %s", e)
            return False


# Self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)

    print("=== ColdStartManager Self-Test ===\n")

    # Test 1: Observation phase
    print("Test 1: Observation phase graduation")
    mgr = ColdStartManager(observation_min_bars=10)
    for _i in range(15):
        mgr.update(new_bar=True)
        next_phase = mgr.check_graduation()
        if next_phase:
            mgr.graduate(next_phase)
            break

    if mgr.current_phase == WarmupPhase.PAPER_TRADING:
        print(f"  ✓ Graduated to {mgr.current_phase.name}")
    else:
        print(f"  ✗ Failed to graduate, stuck in {mgr.current_phase.name}")

    # Test 2: Paper trading with good performance
    print("\nTest 2: Paper trading graduation (good performance)")
    mgr2 = ColdStartManager(paper_min_bars=20, paper_min_sharpe=0.3, paper_min_win_rate=0.5)
    mgr2.current_phase = WarmupPhase.PAPER_TRADING

    # Simulate 30 profitable trades
    for i in range(30):
        mgr2.update(new_bar=True)
        if i % 2 == 0:  # Trade every 2 bars
            pnl = rng.standard_normal() + 0.5  # Biased positive
            mgr2.update(trade_completed={"pnl": pnl, "is_paper": True})

    next_phase = mgr2.check_graduation()
    if next_phase == WarmupPhase.MICRO_POSITIONS:
        print(f"  ✓ Graduated to {next_phase.name}")
        status = mgr2.get_status()
        print(
            f"    Sharpe={status['current_metrics']['sharpe']:.2f}, WinRate={status['current_metrics']['win_rate']:.1%}"
        )
    else:
        print("  ✗ Failed to graduate")

    # Test 3: Paper trading with bad performance
    print("\nTest 3: Paper trading with bad performance (no graduation)")
    mgr3 = ColdStartManager(paper_min_bars=20, paper_min_sharpe=0.5)
    mgr3.current_phase = WarmupPhase.PAPER_TRADING

    # Simulate losing trades
    for i in range(30):
        mgr3.update(new_bar=True)
        if i % 2 == 0:
            pnl = rng.standard_normal() - 0.5  # Biased negative
            mgr3.update(trade_completed={"pnl": pnl, "is_paper": True})

    next_phase = mgr3.check_graduation()
    if next_phase is None:
        print("  ✓ Correctly blocked graduation due to poor performance")
    else:
        print("  ✗ Should not have graduated")

    # Test 4: Position size multipliers
    print("\nTest 4: Position size multipliers")
    test_cases = [
        (WarmupPhase.OBSERVATION, 0.0),
        (WarmupPhase.PAPER_TRADING, 0.0),
        (WarmupPhase.MICRO_POSITIONS, 0.001),
        (WarmupPhase.PRODUCTION, 1.0),
    ]

    all_correct = True
    for phase, expected_mult in test_cases:
        mgr_test = ColdStartManager()
        mgr_test.current_phase = phase
        actual = mgr_test.get_position_size_multiplier()
        if actual == expected_mult:
            print(f"  ✓ {phase.name}: multiplier={actual}")
        else:
            print(f"  ✗ {phase.name}: expected {expected_mult}, got {actual}")
            all_correct = False

    # Test 5: Production demotion
    print("\nTest 5: Production demotion on poor performance")
    mgr5 = ColdStartManager(
        demotion=DemotionThresholds(sharpe=0.5, win_rate=0.45),
    )
    mgr5.current_phase = WarmupPhase.PRODUCTION

    # Simulate many losing trades
    for i in range(100):
        mgr5.update(new_bar=True)
        if i % 2 == 0:
            pnl = rng.standard_normal() - 0.3  # Losing bias
            mgr5.update(trade_completed={"pnl": pnl, "is_paper": False})

    demote_to = mgr5.check_graduation()  # In production, "graduation" is actually demotion check
    if demote_to == WarmupPhase.MICRO_POSITIONS:
        print("  ✓ Correctly demoted to MICRO_POSITIONS")
    else:
        print(f"  ✗ Should have been demoted, got {demote_to}")

    # Test 6: State persistence
    print("\nTest 6: State persistence")
    import tempfile

    temp_file = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=False).name)
    mgr6 = ColdStartManager(state_file=temp_file)
    mgr6.current_phase = WarmupPhase.MICRO_POSITIONS
    mgr6.bars_in_current_phase = 123
    mgr6.save_state()

    mgr6_reload = ColdStartManager(state_file=temp_file)
    if mgr6_reload.load_state():
        if mgr6_reload.current_phase == WarmupPhase.MICRO_POSITIONS and mgr6_reload.bars_in_current_phase == 123:
            print("  ✓ State persistence working")
        else:
            print("  ✗ State mismatch")
    else:
        print("  ✗ Failed to load state")

    temp_file.unlink(missing_ok=True)

    print("\n=== Self-Test Complete ===")
