"""
Model Layer — India Paddy Yield Prediction
Supports Random Forest and XGBoost with confidence intervals.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple, Optional
from .config import RF_PARAMS, XGB_PARAMS, MODEL_PATH, CV_FOLDS, RANDOM_STATE


class YieldPredictor:
    """
    Wrapper for crop yield regression model.
    Supports RandomForest (default) or XGBoost.
    """

    def __init__(self, model_type: str = 'random_forest', **kwargs):
        self.model_type = model_type
        self.feature_names = None
        self.is_trained = False

        if model_type == 'random_forest':
            params = {**RF_PARAMS, **kwargs}
            self.model = RandomForestRegressor(**params)

        elif model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
            except ImportError:
                raise ImportError("Install xgboost: pip install xgboost")
            params = {**XGB_PARAMS, **kwargs}
            self.model = XGBRegressor(**params)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # ------------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: list = None):
        print(f"\n{'='*60}")
        print(f"TRAINING  —  {self.model_type.upper()}")
        print(f"{'='*60}")
        print(f"  Samples : {X_train.shape[0]:,}")
        print(f"  Features: {X_train.shape[1]}")
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("  Training complete.")

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict(X)

    def predict_with_confidence(self,
                                X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean_prediction, std_dev) from tree ensemble."""
        self._check_trained()

        if self.model_type == 'random_forest':
            tree_preds = np.array(
                [t.predict(X) for t in self.model.estimators_]
            )
            return tree_preds.mean(axis=0), tree_preds.std(axis=0)

        elif self.model_type == 'xgboost':
            preds = self.model.predict(X)
            # XGBoost has no built-in uncertainty; use ±5% as proxy
            return preds, preds * 0.05

    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray,
                 y_test: np.ndarray) -> Dict[str, float]:
        self._check_trained()
        y_pred = self.predict(X_test)

        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae':  float(mean_absolute_error(y_test, y_pred)),
            'r2':   float(r2_score(y_test, y_pred)),
            'mape': float(
                np.mean(np.abs((y_test - y_pred) /
                               np.where(y_test == 0, 1, y_test))) * 100
            ),
        }

        print(f"\n{'='*60}")
        print(f"EVALUATION — {self.model_type.upper()}")
        print(f"{'='*60}")
        print(f"  Test samples : {len(y_test):,}")
        print(f"  RMSE : {metrics['rmse']:,.2f} kg/ha")
        print(f"  MAE  : {metrics['mae']:,.2f} kg/ha")
        print(f"  R²   : {metrics['r2']:.4f}")
        print(f"  MAPE : {metrics['mape']:.2f}%")
        print(f"{'='*60}\n")
        return metrics

    # ------------------------------------------------------------------
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       cv: int = CV_FOLDS) -> Dict[str, float]:
        self._check_trained()
        print(f"\nCross-validation ({cv}-fold) — {self.model_type}")

        neg_mse = cross_val_score(self.model, X, y,
                                  cv=cv, scoring='neg_mean_squared_error',
                                  n_jobs=-1)
        r2_scores = cross_val_score(self.model, X, y,
                                    cv=cv, scoring='r2', n_jobs=-1)
        rmse_scores = np.sqrt(-neg_mse)

        results = {
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std':  rmse_scores.std(),
            'cv_r2_mean':   r2_scores.mean(),
            'cv_r2_std':    r2_scores.std(),
        }
        print(f"  CV RMSE: {results['cv_rmse_mean']:,.2f} ± "
              f"{results['cv_rmse_std']:,.2f}")
        print(f"  CV R²  : {results['cv_r2_mean']:.4f} ± "
              f"{results['cv_r2_std']:.4f}")
        return results

    # ------------------------------------------------------------------
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        self._check_trained()

        if self.model_type == 'random_forest':
            importances = self.model.feature_importances_
        elif self.model_type == 'xgboost':
            importances = self.model.feature_importances_
        else:
            return None

        names = self.feature_names or [
            f'f{i}' for i in range(len(importances))
        ]
        df = pd.DataFrame({'feature': names, 'importance': importances})
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def print_feature_importance(self, top_n: int = 15):
        df = self.get_feature_importance()
        if df is None:
            return
        df = df.head(top_n)
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE")
        print(f"{'='*60}")
        for _, row in df.iterrows():
            bar = '█' * int(row['importance'] * 60)
            print(f"  {row['feature']:35s} {row['importance']:.4f}  {bar}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    def save(self, path: str = MODEL_PATH):
        self._check_trained()
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
        }
        joblib.dump(data, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> 'YieldPredictor':
        data = joblib.load(path)
        obj = cls(model_type=data['model_type'])
        obj.model = data['model']
        obj.feature_names = data['feature_names']
        obj.is_trained = data['is_trained']
        print(f"Model loaded from: {path}")
        return obj

    # ------------------------------------------------------------------
    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
