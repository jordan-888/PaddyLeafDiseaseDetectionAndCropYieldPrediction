"""
Training Pipeline — India Paddy Yield (Time-Aware, Real Data)
Loads clean data, time-splits, compares RF vs XGBoost,
saves best model + preprocessor, generates reports.
"""

import numpy as np
import pandas as pd
from .data_loader import load_clean_data, load_and_prepare
from .preprocessing import DataPreprocessor
from .model_comparison import compare_models, time_split
from .reporting import generate_all_reports
from .config import (
    TARGET_COLUMN, ALL_FEATURES, MODEL_PATH, PREPROCESSOR_PATH
)


def train_pipeline(force_rebuild: bool = False) -> dict:
    """
    Full training pipeline:
      1. Load (or build) clean data
      2. Time-based split
      3. Fit preprocessor on train only
      4. Compare RF vs XGBoost
      5. Save best model + preprocessor
      6. Generate reports
    Returns metrics dict.
    """
    print("\n" + "="*60)
    print("  INDIA PADDY YIELD — TRAINING PIPELINE")
    print("="*60)

    # Step 1 — data
    df = load_and_prepare() if force_rebuild else load_clean_data()

    # Step 2 — time split
    train_df, test_df = time_split(df, TARGET_COLUMN)

    # Step 3 — preprocessor (fit on train ONLY — no leakage)
    preprocessor = DataPreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    y_train = preprocessor.get_target(train_df)
    X_test  = preprocessor.transform(test_df)
    y_test  = preprocessor.get_target(test_df)

    print(f"\nX_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}   y_test : {y_test.shape}")

    # Step 4 — model comparison
    best_name, best_model, all_metrics = compare_models(
        X_train, y_train, X_test, y_test,
        feature_names=ALL_FEATURES
    )

    # Step 5 — save
    best_model.save(MODEL_PATH)
    preprocessor.save(PREPROCESSOR_PATH)

    # Step 6 — reports
    generate_all_reports(best_model, X_test, y_test,
                         feature_names=ALL_FEATURES)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  Best model : {best_name.replace('_',' ').title()}")
    m = all_metrics[best_name]
    print(f"  RMSE       : {m['rmse']:,.2f} kg/ha")
    print(f"  MAE        : {m['mae']:,.2f} kg/ha")
    print(f"  R²         : {m['r2']:.4f}")
    print(f"  MAPE       : {m['mape']:.2f}%")
    print("="*60 + "\n")

    return {'best_model': best_name, 'metrics': all_metrics}


if __name__ == '__main__':
    train_pipeline()
