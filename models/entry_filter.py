"""
ML Entry Filter — RF/GBM entry quality scorer.
Wraps scikit-learn RandomForest and LightGBM behind a common interface.
SHAP feature importance tracked per symbol/TF.

Usage:
    filter = EntryFilter(model_type="lgbm")
    filter.fit(X_train, y_train)
    score = filter.predict_proba(features)  # 0.0-1.0 entry quality
    importance = filter.feature_importance(feature_names)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from core.math_utils import safe_div
from core.numeric import clamp, is_valid_number
from core.logger import get_logger

logger = get_logger("entry_filter")


@dataclass
class FilterResult:
    score: float            # 0.0-1.0 entry quality probability
    allow: bool             # score >= threshold
    threshold: float
    model_type: str
    feature_names: List[str] = field(default_factory=list)
    importances: Dict[str, float] = field(default_factory=dict)


class EntryFilter:
    """
    Entry quality classifier.
    model_type: 'lgbm' (default, fastest), 'rf' (robust), 'xgb'
    threshold: minimum score to allow entry (tune via Optuna)
    """

    def __init__(
        self,
        model_type: str = "lgbm",
        threshold: float = 0.55,
        n_estimators: int = 100,
        max_depth: int = 4,
        symbol: str = "",
        tf: str = "",
    ) -> None:
        self._model_type = model_type.lower()
        self._threshold = clamp(threshold, 0.0, 1.0)
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._symbol = symbol
        self._tf = tf
        self._model = None
        self._feature_names: List[str] = []
        self._fitted = False
        self._shap_values: Optional[Any] = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, X: List[List[float]], y: List[float], feature_names: Optional[List[str]] = None) -> "EntryFilter":
        """
        Train the model.
        X: feature matrix [[f1, f2, ...], ...]
        y: binary labels [0/1] — 1 = good entry (MFE > threshold)
        """
        if not X or not y:
            logger.warning("Empty training data", symbol=self._symbol, tf=self._tf)
            return self

        self._feature_names = feature_names or [f"f{i}" for i in range(len(X[0]))]

        try:
            import numpy as np
            X_arr = np.array(X, dtype=np.float32)
            y_arr = np.array(y, dtype=np.int32)

            model = self._build_model()
            model.fit(X_arr, y_arr)
            self._model = model
            self._fitted = True

            logger.info(
                f"EntryFilter fitted: {self._model_type} n={len(y)} features={len(self._feature_names)}",
                symbol=self._symbol, tf=self._tf, component="entry_filter"
            )
        except ImportError as e:
            logger.error(f"Missing dependency: {e}. Install scikit-learn and lightgbm.")
        except Exception as e:
            logger.error(f"Fit failed: {e}", symbol=self._symbol, tf=self._tf)

        return self

    def predict_proba(self, features: List[float]) -> float:
        """Return probability of a good entry. 0.0-1.0."""
        if not self._fitted or self._model is None:
            return 0.5  # neutral when not calibrated

        try:
            import numpy as np
            x = np.array([features], dtype=np.float32)
            proba = self._model.predict_proba(x)[0]
            score = float(proba[1]) if len(proba) > 1 else float(proba[0])
            return clamp(score, 0.0, 1.0)
        except Exception as e:
            logger.error(f"predict_proba failed: {e}", symbol=self._symbol, tf=self._tf)
            return 0.5

    def evaluate(self, features: List[float]) -> FilterResult:
        score = self.predict_proba(features)
        return FilterResult(
            score=score,
            allow=score >= self._threshold,
            threshold=self._threshold,
            model_type=self._model_type,
            feature_names=self._feature_names,
        )

    def feature_importance(self) -> Dict[str, float]:
        """Return feature importances from the model."""
        if not self._fitted or self._model is None:
            return {}
        try:
            importances = self._model.feature_importances_
            total = sum(importances) or 1.0
            return {
                name: safe_div(float(imp), total)
                for name, imp in zip(self._feature_names, importances)
            }
        except Exception:
            return {}

    def shap_importance(self, X: List[List[float]]) -> Dict[str, float]:
        """
        Compute SHAP-based feature importances.
        Requires shap package: pip install shap
        Returns mean absolute SHAP value per feature.
        """
        if not self._fitted or self._model is None:
            return {}
        try:
            import shap
            import numpy as np
            explainer = shap.TreeExplainer(self._model)
            X_arr = np.array(X, dtype=np.float32)
            shap_vals = explainer.shap_values(X_arr)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # class 1 for binary
            mean_abs = [float(abs(shap_vals[:, i]).mean()) for i in range(shap_vals.shape[1])]
            total = sum(mean_abs) or 1.0
            return {
                name: safe_div(v, total)
                for name, v in zip(self._feature_names, mean_abs)
            }
        except ImportError:
            logger.warning("shap not installed — falling back to model importances")
            return self.feature_importance()
        except Exception as e:
            logger.error(f"SHAP failed: {e}")
            return self.feature_importance()

    def _build_model(self) -> Any:
        if self._model_type == "lgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=self._n_estimators,
                    max_depth=self._max_depth,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    verbose=-1,
                )
            except ImportError:
                logger.warning("LightGBM not available, falling back to RF")
                return self._build_rf()

        elif self._model_type == "xgb":
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=self._n_estimators,
                    max_depth=self._max_depth,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    verbosity=0,
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to RF")
                return self._build_rf()

        return self._build_rf()

    def _build_rf(self) -> Any:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )

    def set_threshold(self, threshold: float) -> None:
        self._threshold = clamp(threshold, 0.0, 1.0)

    def reset(self) -> None:
        self._model = None
        self._fitted = False
        self._shap_values = None
