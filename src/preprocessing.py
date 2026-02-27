"""
Preprocessing — India Paddy Yield (Real Data)
Encodes the State categorical and scales numeric features.
No synthetic feature generation.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple
from .config import (
    ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    TARGET_COLUMN, PREPROCESSOR_PATH
)


class DataPreprocessor:
    """
    Handles encoding and scaling for the India paddy yield model.
    Reproducible: fitted on train set, applied to test/inference.
    """

    def __init__(self):
        self.label_encoders: dict = {}   # one per categorical feature
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """Fit encoders and scaler on training DataFrame."""
        print("Fitting preprocessor...")

        # Encode categoricals
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
            print(f"  LabelEncoder '{col}': {len(le.classes_)} classes")

        # Build numeric matrix and fit scaler
        X_num = df[NUMERICAL_FEATURES].values.astype(float)
        self.scaler.fit(X_num)

        self.feature_names = ALL_FEATURES
        self.is_fitted = True
        print("Preprocessor fitted.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame → feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        # Encode categoricals
        cat_arrays = []
        for col in CATEGORICAL_FEATURES:
            le = self.label_encoders[col]
            encoded = le.transform(df[col].astype(str))
            cat_arrays.append(encoded.reshape(-1, 1))

        # Scale numerics
        X_num = df[NUMERICAL_FEATURES].values.astype(float)
        X_num_scaled = self.scaler.transform(X_num)

        return np.hstack(cat_arrays + [X_num_scaled])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience: fit then transform."""
        return self.fit(df).transform(df)

    def get_target(self, df: pd.DataFrame) -> np.ndarray:
        """Extract target column as numpy array."""
        return df[TARGET_COLUMN].values.astype(float)

    def save(self, path: str = PREPROCESSOR_PATH):
        """Persist preprocessor to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str = PREPROCESSOR_PATH) -> 'DataPreprocessor':
        """Load preprocessor from disk."""
        obj = joblib.load(path)
        print(f"Preprocessor loaded from: {path}")
        return obj


if __name__ == '__main__':
    from data_loader import load_clean_data
    df = load_clean_data()
    pp = DataPreprocessor()
    X = pp.fit_transform(df)
    y = pp.get_target(df)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
