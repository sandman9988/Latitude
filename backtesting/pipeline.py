"""
BacktestPipeline — ML-aware walk-forward backtesting.

Extends the basic WalkForwardValidator with a proper IS-train / OOS-test loop
that supports the ML entry filter and self-calibrating runway predictor:

  For each WF fold:
    1. Run IS bars with heuristic-only strategy → collect feature_log
    2. Pre-compute ATR series on IS bars
    3. Label IS bars → binary labels (hit_tp=1, hit_sl=0)
    4. Match feature_log entries to labels → (X, y) training set
    5. Train EntryFilter on (X, y)
    6. Clone strategy config → reset state → run OOS with trained filter
    7. Record OOS metrics; feed runway calibration from closed trades

Optuna integration:
    pipeline.tune(bars, n_trials=50) → TuneResult with best StrategyConfig fields.
    Each trial runs the full ML-aware walk-forward as objective.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

from core.math_utils import safe_div
from core.logger import get_logger
from core.validator import BrokerSpec
from pipeline.cleaner import Bar
from pipeline.resampler import build_mtf
from pipeline.labeller import label_bars
from pipeline.features.volatility import ATR
from backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult
from backtesting.metrics import compute_metrics

# Deferred to avoid circular import:
# strategy.trend_strategy → backtesting.types → backtesting (pkg init) → pipeline → strategy
if TYPE_CHECKING:
    from strategy.trend_strategy import TrendStrategy, StrategyConfig


def _import_strategy():
    """Lazy import helper — call once per code path that needs TrendStrategy."""
    from strategy.trend_strategy import TrendStrategy, StrategyConfig, build_on_bar, FEATURE_NAMES  # noqa: F401
    return TrendStrategy, StrategyConfig, build_on_bar, FEATURE_NAMES

logger = get_logger("pipeline")


# ---------------------------------------------------------------------------
# Fold result
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    is_bars: int
    oos_bars: int
    is_metrics: Dict[str, float] = field(default_factory=dict)
    oos_metrics: Dict[str, float] = field(default_factory=dict)
    is_trades: int = 0
    oos_trades: int = 0
    ml_trained: bool = False
    ml_train_samples: int = 0


@dataclass
class PipelineResult:
    folds: List[FoldResult] = field(default_factory=list)
    oos_metrics: Dict[str, float] = field(default_factory=dict)   # aggregated across all OOS
    efficiency: float = 0.0     # mean OOS win_rate / mean IS win_rate
    is_robust: bool = False
    best_config: Optional[Any] = None


# ---------------------------------------------------------------------------
# BacktestPipeline
# ---------------------------------------------------------------------------

class BacktestPipeline:
    """
    Orchestrates the full IS-train / OOS-test walk-forward loop.

    Usage:
        pipeline = BacktestPipeline(spec, config)
        result = pipeline.run_walk_forward(m30_bars, n_folds=5)
        tune_result = pipeline.tune(m30_bars, n_trials=50)
    """

    def __init__(
        self,
        spec: BrokerSpec,
        config: Optional[Any] = None,
        initial_balance: float = 10_000.0,
    ) -> None:
        self._spec = spec
        if config is None:
            _, StrategyConfig, _, _ = _import_strategy()
            config = StrategyConfig()
        self._config = config
        self._initial_balance = initial_balance

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run_single(self, m30_bars: List[Bar]) -> BacktestResult:
        """
        Single pass over all bars. No ML training — heuristic gates only.
        Useful for a quick sanity check before walk-forward.
        """
        TrendStrategy, _, build_on_bar, _ = _import_strategy()
        mtf = build_mtf(m30_bars)
        htf_bars = mtf.get("H4", [])
        strategy = TrendStrategy(self._config, self._spec, htf_bars)
        engine = self._make_engine()
        return engine.run(m30_bars, build_on_bar(strategy))

    def run_walk_forward(
        self,
        m30_bars: List[Bar],
        n_folds: int = 5,
        anchored: bool = False,
    ) -> PipelineResult:
        """
        ML-aware walk-forward: IS fit → OOS evaluate, repeated n_folds times.
        """
        n = len(m30_bars)
        if n < 200:
            logger.warning(f"Too few bars for walk-forward: {n}", component="pipeline")
            return PipelineResult()

        mtf = build_mtf(m30_bars)
        windows = _build_wf_windows(n, n_folds, anchored)
        folds: List[FoldResult] = []
        all_oos_trades = []
        all_oos_equity = []

        for win in windows:
            train_bars = m30_bars[win["train_start"]:win["train_end"]]
            test_bars  = m30_bars[win["test_start"]:win["test_end"]]
            htf_train  = _slice_htf(mtf["H4"], m30_bars, win["train_start"], win["train_end"])
            htf_test   = _slice_htf(mtf["H4"], m30_bars, win["test_start"], win["test_end"])

            fold_result, oos_result = self._run_fold(
                fold=win["fold"],
                train_bars=train_bars,
                test_bars=test_bars,
                htf_train=htf_train,
                htf_test=htf_test,
            )
            folds.append(fold_result)
            all_oos_trades.extend(oos_result.trades)
            all_oos_equity.extend(oos_result.equity_curve)

            logger.info(
                f"Fold {win['fold']}: IS={fold_result.is_metrics.get('win_rate', 0):.1%} "
                f"OOS={fold_result.oos_metrics.get('win_rate', 0):.1%} "
                f"trades={fold_result.oos_trades} "
                f"ml={'yes' if fold_result.ml_trained else 'no (n=' + str(fold_result.ml_train_samples) + ')'}",
                component="pipeline",
            )

        result = PipelineResult(folds=folds)

        if all_oos_trades:
            result.oos_metrics = compute_metrics(
                all_oos_trades, self._initial_balance, all_oos_equity
            )

        is_wrs  = [f.is_metrics.get("win_rate",  0) for f in folds]
        oos_wrs = [f.oos_metrics.get("win_rate", 0) for f in folds]
        mean_is  = safe_div(sum(is_wrs),  len(is_wrs))
        mean_oos = safe_div(sum(oos_wrs), len(oos_wrs))
        result.efficiency = safe_div(mean_oos, mean_is)
        result.is_robust  = result.efficiency >= 0.65 and mean_oos >= 0.50

        logger.info(
            f"Walk-forward complete: OOS win_rate={mean_oos:.1%} "
            f"efficiency={result.efficiency:.2f} robust={result.is_robust}",
            component="pipeline",
        )
        return result

    def tune(
        self,
        m30_bars: List[Bar],
        n_trials: int = 50,
        n_folds: int = 4,
        timeout_s: Optional[int] = 3600,
        metric: str = "win_rate",
        output_dir: Optional[Path] = None,
    ) -> "TuneResult":
        """
        Optuna hyperparameter search. Each trial runs the full ML-aware
        walk-forward as the objective function.
        Returns best StrategyConfig fields + TuneResult.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("optuna not installed — run: pip install optuna")
            from backtesting.optuna_tuner import TuneResult
            return TuneResult()

        mtf = build_mtf(m30_bars)
        n = len(m30_bars)
        windows = _build_wf_windows(n, n_folds, anchored=False)

        def objective(trial) -> float:
            config = _suggest_config(trial)
            pipeline = BacktestPipeline(self._spec, config, self._initial_balance)
            all_oos_trades = []
            all_oos_equity = []

            for win in windows:
                train_bars = m30_bars[win["train_start"]:win["train_end"]]
                test_bars  = m30_bars[win["test_start"]:win["test_end"]]
                htf_train  = _slice_htf(mtf["H4"], m30_bars, win["train_start"], win["train_end"])
                htf_test   = _slice_htf(mtf["H4"], m30_bars, win["test_start"], win["test_end"])
                try:
                    _, oos_result = pipeline._run_fold(
                        fold=win["fold"],
                        train_bars=train_bars,
                        test_bars=test_bars,
                        htf_train=htf_train,
                        htf_test=htf_test,
                    )
                    all_oos_trades.extend(oos_result.trades)
                    all_oos_equity.extend(oos_result.equity_curve)
                except Exception as e:
                    logger.warning(f"Trial {trial.number} fold {win['fold']} failed: {e}")
                    return 0.0

            if len(all_oos_trades) < 20:
                return 0.0

            metrics = compute_metrics(all_oos_trades, self._initial_balance, all_oos_equity)
            value = float(metrics.get(metric, 0.0))

            # Penalise excessive drawdown
            if metrics.get("max_drawdown_pct", 0) > 0.30:
                value *= 0.5

            # Pruning: report intermediate value after each fold
            trial.report(value, step=n_folds)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return value

        study = optuna.create_study(
            direction="maximize",
            study_name=f"latitude_{self._spec.symbol}",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_s,
            show_progress_bar=True,
        )

        best_config = _params_to_config(study.best_params)

        logger.info(
            f"Optuna complete: best {metric}={study.best_value:.4f} "
            f"trials={len(study.trials)}",
            component="pipeline",
        )

        if output_dir:
            _save_tune_result(study, best_config, output_dir, self._spec.symbol)

        from backtesting.optuna_tuner import TuneResult
        return TuneResult(
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            all_trials=study.trials,
        )

    def save_result(self, result: PipelineResult, output_dir: Path) -> None:
        """Save walk-forward result metrics to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self._spec.symbol}_wf_result.json"
        data = {
            "symbol": self._spec.symbol,
            "oos_metrics": result.oos_metrics,
            "efficiency": result.efficiency,
            "is_robust": result.is_robust,
            "folds": [
                {
                    "fold": f.fold,
                    "is_trades": f.is_trades,
                    "oos_trades": f.oos_trades,
                    "ml_trained": f.ml_trained,
                    "ml_samples": f.ml_train_samples,
                    "is_win_rate": f.is_metrics.get("win_rate", 0),
                    "oos_win_rate": f.oos_metrics.get("win_rate", 0),
                    "oos_sharpe": f.oos_metrics.get("sharpe", 0),
                    "oos_profit_factor": f.oos_metrics.get("profit_factor", 0),
                    "oos_max_dd": f.oos_metrics.get("max_drawdown_pct", 0),
                }
                for f in result.folds
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved walk-forward result to {path}", component="pipeline")

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _run_fold(
        self,
        fold: int,
        train_bars: List[Bar],
        test_bars: List[Bar],
        htf_train: List[Bar],
        htf_test: List[Bar],
    ) -> Tuple[FoldResult, BacktestResult]:
        """
        Single IS-train / OOS-test fold.
          1. IS run: heuristic-only → feature_log + ATR series
          2. Label IS bars → binary outcomes
          3. Train entry filter on IS (X, y)
          4. OOS run with trained filter → BacktestResult
          5. Feed runway predictor from OOS closed trades
        """
        fold_result = FoldResult(fold=fold, is_bars=len(train_bars), oos_bars=len(test_bars))

        TrendStrategy, _, build_on_bar, _ = _import_strategy()

        # --- IS pass (heuristic only) ---
        is_strategy = TrendStrategy(self._config, self._spec, htf_train)
        is_engine = self._make_engine()
        is_result = is_engine.run(train_bars, build_on_bar(is_strategy))
        fold_result.is_metrics = is_result.metrics
        fold_result.is_trades = int(is_result.metrics.get("total_trades", 0))

        # --- Build (X, y) training set from IS feature_log ---
        atr_series = _compute_atr_series(train_bars, self._config.atr_period)
        X, y = _build_training_data(
            feature_log=is_strategy.get_feature_log(),
            bars=train_bars,
            atr_series=atr_series,
            sl_atr_mult=self._config.sl_atr_mult,
            tp_atr_mult=self._config.tp_atr_mult,
            horizon=30,
        )
        fold_result.ml_train_samples = len(y)

        # --- OOS pass with trained filter (or heuristic if not enough IS data) ---
        oos_strategy = TrendStrategy(self._config, self._spec, htf_test)  # TrendStrategy imported above
        if len(y) >= 30:
            oos_strategy.train(X, y)
            fold_result.ml_trained = True

        oos_engine = self._make_engine()
        oos_result = oos_engine.run(test_bars, build_on_bar(oos_strategy))
        fold_result.oos_metrics = oos_result.metrics
        fold_result.oos_trades  = int(oos_result.metrics.get("total_trades", 0))

        # --- Feed runway predictor from OOS closed trades ---
        for trade in oos_result.trades:
            oos_strategy.record_trade_outcome(trade.mfe)

        return fold_result, oos_result

    def _make_engine(self) -> BacktestEngine:
        config = BacktestConfig(
            initial_balance=self._initial_balance,
            lots_per_1000=self._config.lots_per_1000,
            use_dynamic_sizing=True,
            max_open_trades=3,
            max_pyramid_levels=self._config.max_pyramid_levels,
            pyramid_atr_step=self._config.pyramid_atr_step,
            allow_short=self._spec.short_selling_enabled,
        )
        return BacktestEngine(config, self._spec)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_wf_windows(n: int, n_folds: int, anchored: bool) -> List[Dict]:
    windows = []
    fold_size = n // n_folds
    for fold in range(n_folds):
        if anchored:
            train_start = 0
            train_end   = fold_size * (fold + 1)
        else:
            train_start = fold * fold_size
            train_end   = train_start + int(fold_size * 0.7)
        test_start = train_end
        test_end   = min(test_start + int(fold_size * 0.3), n)
        if test_start >= test_end or test_start >= n:
            break
        windows.append({
            "fold": fold + 1,
            "train_start": train_start,
            "train_end":   train_end,
            "test_start":  test_start,
            "test_end":    test_end,
        })
    return windows


def _slice_htf(htf_bars: List[Bar], m30_bars: List[Bar], start_idx: int, end_idx: int) -> List[Bar]:
    """Return HTF bars that fall within the same timestamp window as m30_bars[start:end]."""
    if not htf_bars or not m30_bars or start_idx >= len(m30_bars):
        return []
    t_start = m30_bars[start_idx].timestamp
    t_end   = m30_bars[min(end_idx, len(m30_bars) - 1)].timestamp
    return [b for b in htf_bars if t_start <= b.timestamp <= t_end]


def _compute_atr_series(bars: List[Bar], period: int = 14) -> List[float]:
    """Compute ATR value per bar. Returns list same length as bars."""
    atr = ATR(period=period)
    result = []
    for bar in bars:
        atr.update(bar.high, bar.low, bar.close)
        result.append(atr.value if atr.ready else 0.0)
    return result


def _build_training_data(
    feature_log: List[Dict],
    bars: List[Bar],
    atr_series: List[float],
    sl_atr_mult: float,
    tp_atr_mult: float,
    horizon: int = 30,
) -> Tuple[List[List[float]], List[int]]:
    """
    Match feature_log entries (each recorded at signal time) to forward labels.
    Returns (X, y) where y=1 if trade hit TP, 0 if SL or time exit.
    """
    if not feature_log or not bars:
        return [], []

    # Build timestamp → bar index map
    ts_to_idx: Dict[float, int] = {b.timestamp: i for i, b in enumerate(bars)}

    # Pre-compute labels for both directions using labeller
    long_labels  = {lbl.timestamp: lbl for lbl in
                    label_bars(bars, direction=1,  horizon=horizon,
                               sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                               atr_values=atr_series)}
    short_labels = {lbl.timestamp: lbl for lbl in
                    label_bars(bars, direction=-1, horizon=horizon,
                               sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                               atr_values=atr_series)}

    X: List[List[float]] = []
    y: List[int] = []

    for entry in feature_log:
        ts        = entry["timestamp"]
        direction = entry["direction"]
        features  = entry["features"]

        labels = long_labels if direction == 1 else short_labels
        lbl = labels.get(ts)
        if lbl is None:
            continue

        X.append(features)
        y.append(1 if lbl.hit_tp else 0)

    return X, y


def _suggest_config(trial) -> StrategyConfig:
    """Suggest a StrategyConfig from an Optuna trial."""
    return StrategyConfig(
        ker_period=trial.suggest_int("ker_period", 5, 20),
        ker_trend_threshold=trial.suggest_float("ker_threshold", 0.25, 0.65),
        vhf_period=trial.suggest_int("vhf_period", 14, 50),
        vhf_trend_threshold=trial.suggest_float("vhf_threshold", 0.20, 0.60),
        laguerre_gamma=trial.suggest_float("laguerre_gamma", 0.4, 0.95),
        laguerre_buy_threshold=trial.suggest_float("laguerre_buy", 0.10, 0.40),
        laguerre_sell_threshold=trial.suggest_float("laguerre_sell", 0.60, 0.90),
        atr_period=trial.suggest_int("atr_period", 7, 21),
        sl_atr_mult=trial.suggest_float("sl_atr_mult", 1.0, 3.0),
        tp_atr_mult=trial.suggest_float("tp_atr_mult", 1.5, 5.0),
        pyramid_atr_step=trial.suggest_float("pyramid_atr_step", 0.5, 2.5),
        max_pyramid_levels=trial.suggest_int("max_pyramid_levels", 1, 3),
        lots_per_1000=trial.suggest_float("lots_per_1000", 0.005, 0.05),
        entry_threshold=trial.suggest_float("entry_threshold", 0.45, 0.75),
        floor_multiplier=trial.suggest_float("floor_multiplier", 1.0, 4.0),
        dtw_template=trial.suggest_categorical("dtw_template", ["impulse", "breakout", "pullback"]),
        dtw_window=trial.suggest_int("dtw_window", 10, 30),
        warmup_bars=50,
    )


def _params_to_config(params: Dict[str, Any]) -> StrategyConfig:
    return StrategyConfig(
        ker_period=params.get("ker_period", 14),
        ker_trend_threshold=params.get("ker_threshold", 0.45),
        vhf_period=params.get("vhf_period", 28),
        vhf_trend_threshold=params.get("vhf_threshold", 0.35),
        laguerre_gamma=params.get("laguerre_gamma", 0.7),
        laguerre_buy_threshold=params.get("laguerre_buy", 0.2),
        laguerre_sell_threshold=params.get("laguerre_sell", 0.8),
        atr_period=params.get("atr_period", 14),
        sl_atr_mult=params.get("sl_atr_mult", 1.5),
        tp_atr_mult=params.get("tp_atr_mult", 3.0),
        pyramid_atr_step=params.get("pyramid_atr_step", 1.0),
        max_pyramid_levels=params.get("max_pyramid_levels", 2),
        lots_per_1000=params.get("lots_per_1000", 0.01),
        entry_threshold=params.get("entry_threshold", 0.55),
        floor_multiplier=params.get("floor_multiplier", 2.0),
        dtw_template=params.get("dtw_template", "impulse"),
        dtw_window=params.get("dtw_window", 20),
        warmup_bars=50,
    )


def _save_tune_result(study, best_config: StrategyConfig, output_dir: Path, symbol: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}_best_params.json"
    with open(path, "w") as f:
        json.dump({
            "symbol": symbol,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "params": study.best_params,
        }, f, indent=2)
    logger.info(f"Saved best params to {path}", component="pipeline")
