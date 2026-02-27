"""
Model Comparison — India Paddy Yield
Trains RandomForest and XGBoost with time-based split,
prints comparison table, returns best model.
"""

import numpy as np
import pandas as pd
from .model import YieldPredictor
from .config import TRAIN_END_YEAR, TEST_START_YEAR, ALL_FEATURES


def time_split(df: pd.DataFrame, target_col: str):
    """
    Time-based split — NO random shuffling, NO data leakage.
    Train: Year <= TRAIN_END_YEAR
    Test : Year >= TEST_START_YEAR
    """
    train = df[df['Year'] <= TRAIN_END_YEAR].copy()
    test  = df[df['Year'] >= TEST_START_YEAR].copy()
    print(f"\nTime-based split:")
    print(f"  Train: {train['Year'].min()}–{train['Year'].max()} "
          f"({len(train):,} rows)")
    print(f"  Test : {test['Year'].min()}–{test['Year'].max()} "
          f"({len(test):,} rows)")
    return train, test


def compare_models(X_train, y_train, X_test, y_test,
                   feature_names=None) -> tuple:
    """
    Train RF and XGBoost, evaluate, print comparison table.
    Returns (best_model_name, best_predictor, metrics_dict).
    """
    results = {}
    predictors = {}

    for mtype in ['random_forest', 'xgboost']:
        try:
            p = YieldPredictor(model_type=mtype)
            p.train(X_train, y_train, feature_names=feature_names)
            m = p.evaluate(X_test, y_test)
            results[mtype] = m
            predictors[mtype] = p
        except ImportError as e:
            print(f"  Skipping {mtype}: {e}")

    if not results:
        raise RuntimeError("No models trained successfully.")

    # ------ comparison table ------
    print("\n" + "="*65)
    print(f"  {'MODEL':<22} {'RMSE':>10} {'MAE':>10} {'R²':>8} {'MAPE%':>8}")
    print("="*65)
    for name, m in results.items():
        label = name.replace('_', ' ').title()
        print(f"  {label:<22} {m['rmse']:>10,.2f} {m['mae']:>10,.2f} "
              f"{m['r2']:>8.4f} {m['mape']:>8.2f}")
    print("="*65)

    # Best by R²
    best_name = max(results, key=lambda k: results[k]['r2'])
    print(f"\n  ✅ Best model: {best_name.replace('_',' ').title()} "
          f"(R² = {results[best_name]['r2']:.4f})")

    return best_name, predictors[best_name], results
