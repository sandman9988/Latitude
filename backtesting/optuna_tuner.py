"""
Optuna hyperparameter tuner.
Wraps Optuna trials with walk-forward validation to prevent overfitting.
Optimises over indicator periods, regime thresholds, runway params,
entry filter threshold, and position sizing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, List
from core.logger import get_logger

logger = get_logger("optuna_tuner")


@dataclass
class TunerConfig:
    n_trials: int = 100
    n_folds: int = 5
    timeout_seconds: Optional[int] = 3600
    metric: str = "win_rate"            # optimise for: win_rate, sharpe, profit_factor, calmar
    direction: str = "maximize"
    n_jobs: int = 1
    study_name: str = "latitude"
    pruning: bool = True                # prune bad trials early


@dataclass
class TuneResult:
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_value: float = 0.0
    n_trials: int = 0
    all_trials: list = field(default_factory=list)


class OptunaTuner:
    """
    Optuna-based hyperparameter search with walk-forward objective.

    Usage:
        tuner = OptunaTuner(config)
        result = tuner.tune(bars, strategy_factory_from_params)
        best_params = result.best_params

    strategy_factory_from_params: Callable[[dict], Callable[[], tuple[on_bar, engine]]]
    """

    def __init__(self, config: TunerConfig, spec=None, initial_balance: float = 10_000.0) -> None:
        self._config = config
        self._spec = spec
        self._initial_balance = initial_balance

    def tune(self, bars, strategy_factory_from_params: Callable) -> TuneResult:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("Optuna not installed. Run: pip install optuna")
            return TuneResult()

        from .walk_forward import WalkForwardValidator

        wf = WalkForwardValidator(n_folds=self._config.n_folds)

        def objective(trial) -> float:
            params = self._suggest_params(trial)
            try:
                factory = strategy_factory_from_params(params)
                result = wf.validate(bars, factory, self._initial_balance)
                oos = result.oos_metrics
                value = oos.get(self._config.metric, 0.0)

                # Penalise if too few trades
                n_trades = oos.get("total_trades", 0)
                if n_trades < 20:
                    return 0.0

                # Penalise large drawdowns
                max_dd = oos.get("max_drawdown_pct", 1.0)
                if max_dd > 0.30:
                    value *= 0.5

                return float(value)
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return 0.0

        pruner = optuna.pruners.MedianPruner() if self._config.pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(
            direction=self._config.direction,
            study_name=self._config.study_name,
            pruner=pruner,
        )

        study.optimize(
            objective,
            n_trials=self._config.n_trials,
            timeout=self._config.timeout_seconds,
            n_jobs=self._config.n_jobs,
            show_progress_bar=True,
        )

        logger.info(
            f"Optuna complete: best {self._config.metric}={study.best_value:.4f}",
            component="optuna"
        )

        return TuneResult(
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            all_trials=study.trials,
        )

    def _suggest_params(self, trial) -> Dict[str, Any]:
        """
        Default parameter search space.
        Override or extend in your strategy subclass.
        """
        return {
            # Smoother selection
            "smoother": trial.suggest_categorical("smoother", ["jma", "kama", "zlema", "alma"]),

            # Regime thresholds
            "ker_period": trial.suggest_int("ker_period", 5, 20),
            "vhf_period": trial.suggest_int("vhf_period", 14, 50),
            "ker_threshold": trial.suggest_float("ker_threshold", 0.25, 0.65),
            "vhf_threshold": trial.suggest_float("vhf_threshold", 0.35, 0.70),
            "cmo_trend_min": trial.suggest_float("cmo_trend_min", 10.0, 40.0),

            # Laguerre RSI
            "laguerre_gamma": trial.suggest_float("laguerre_gamma", 0.5, 0.95),

            # ATR period
            "atr_period": trial.suggest_int("atr_period", 7, 21),

            # SL/TP
            "sl_atr_mult": trial.suggest_float("sl_atr_mult", 1.0, 3.0),
            "tp_atr_mult": trial.suggest_float("tp_atr_mult", 1.5, 5.0),

            # Runway predictor
            "floor_multiplier": trial.suggest_float("floor_multiplier", 1.0, 4.0),

            # Entry filter
            "entry_threshold": trial.suggest_float("entry_threshold", 0.45, 0.75),

            # Position sizing
            "lots_per_1000": trial.suggest_float("lots_per_1000", 0.005, 0.05),

            # Pyramiding
            "pyramid_atr_step": trial.suggest_float("pyramid_atr_step", 0.5, 2.5),
            "max_pyramid_levels": trial.suggest_int("max_pyramid_levels", 1, 4),

            # DTW template window
            "dtw_window": trial.suggest_int("dtw_window", 10, 30),
        }
