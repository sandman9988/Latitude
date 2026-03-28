"""
Walk-forward validator — prevents overfitting.
Splits data into in-sample (train) + out-of-sample (test) windows,
rolls forward, aggregates OOS results only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional
from core.math_utils import safe_div
from core.logger import get_logger
from pipeline.cleaner import Bar
from .metrics import compute_metrics

logger = get_logger("walk_forward")


@dataclass
class WFWindow:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class WFResult:
    windows: List[WFWindow] = field(default_factory=list)
    oos_metrics: Dict[str, float] = field(default_factory=dict)   # aggregated OOS
    efficiency: float = 0.0   # OOS win_rate / IS win_rate ratio. ~0.7-0.9 = healthy
    is_robust: bool = False


class WalkForwardValidator:
    """
    Anchored or rolling walk-forward.

    train_pct: fraction of available data used for training window
    test_pct: fraction used for testing window
    n_folds: number of WF folds
    anchored: if True, training window expands (anchored WF). If False, rolling.
    """

    def __init__(
        self,
        train_pct: float = 0.7,
        test_pct: float = 0.3,
        n_folds: int = 5,
        anchored: bool = False,
    ) -> None:
        self._train_pct = train_pct
        self._test_pct = test_pct
        self._n_folds = max(2, n_folds)
        self._anchored = anchored

    def validate(
        self,
        bars: List[Bar],
        strategy_factory: Callable,     # () -> (on_bar, engine)
        initial_balance: float = 10_000.0,
    ) -> WFResult:
        """
        Run walk-forward validation.
        strategy_factory: callable that returns a fresh (on_bar_fn, BacktestEngine) per fold.
        """
        n = len(bars)
        if n < 100:
            logger.warning(f"Too few bars for walk-forward: {n}")
            return WFResult()

        windows = self._build_windows(n)
        all_oos_trades = []
        all_oos_equity = []

        for window in windows:
            train_bars = bars[window.train_start:window.train_end]
            test_bars = bars[window.test_start:window.test_end]

            on_bar, engine = strategy_factory()

            # Train
            train_result = engine.run(train_bars, on_bar)
            window.train_metrics = train_result.metrics

            # Re-create fresh for OOS test (no bleed-over)
            on_bar, engine = strategy_factory()
            test_result = engine.run(test_bars, on_bar)
            window.test_metrics = test_result.metrics

            all_oos_trades.extend(test_result.trades)
            all_oos_equity.extend(test_result.equity_curve)

            logger.info(
                f"WF fold {window.fold}: IS win_rate={window.train_metrics.get('win_rate', 0):.2%} "
                f"OOS win_rate={window.test_metrics.get('win_rate', 0):.2%}"
            )

        result = WFResult(windows=windows)

        if all_oos_trades:
            result.oos_metrics = compute_metrics(all_oos_trades, initial_balance, all_oos_equity)

        # Efficiency: mean(OOS win_rate) / mean(IS win_rate)
        is_wr = safe_div(
            sum(w.train_metrics.get("win_rate", 0) for w in windows),
            len(windows)
        )
        oos_wr = safe_div(
            sum(w.test_metrics.get("win_rate", 0) for w in windows),
            len(windows)
        )
        result.efficiency = safe_div(oos_wr, is_wr)
        result.is_robust = result.efficiency >= 0.65 and oos_wr >= 0.50

        logger.info(
            f"Walk-forward complete: OOS win_rate={oos_wr:.2%} "
            f"efficiency={result.efficiency:.2f} robust={result.is_robust}"
        )

        return result

    def _build_windows(self, n: int) -> List[WFWindow]:
        windows = []
        fold_size = n // self._n_folds

        for fold in range(self._n_folds):
            if self._anchored:
                train_start = 0
                train_end = fold_size * (fold + 1)
            else:
                train_start = fold * fold_size
                train_end = train_start + int(fold_size * self._train_pct)

            test_start = train_end
            test_end = min(test_start + int(fold_size * self._test_pct), n)

            if test_start >= test_end or test_start >= n:
                break

            windows.append(WFWindow(
                fold=fold + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))

        return windows
