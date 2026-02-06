"""
Preprocessing Module
Feature engineering, encoding, scaling, and train-test splitting
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
from .config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    SCALER_PATH,
    ENCODER_PATH
)


class DataPreprocessor:
    """
    Handles all preprocessing operations for crop yield prediction.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessors and transform data.
        
        Args:
            df: Input DataFrame with all features and target
            
        Returns:
            Tuple of (X_scaled, y) as numpy arrays
        """
        print("\n" + "="*60)
        print("PREPROCESSING DATA")
        print("="*60)
        
        # Encode categorical features
        df_encoded = df.copy()
        for cat_feature in CATEGORICAL_FEATURES:
            if cat_feature in df.columns:
                print(f"Encoding {cat_feature}...")
                le = LabelEncoder()
                df_encoded[f'{cat_feature}_encoded'] = le.fit_transform(df[cat_feature])
                self.label_encoders[cat_feature] = le
                print(f"  {len(le.classes_)} unique values: {le.classes_[:5]}...")
        
        # Prepare feature matrix
        feature_cols = NUMERICAL_FEATURES.copy()
        for cat_feature in CATEGORICAL_FEATURES:
            if cat_feature in df.columns:
                feature_cols.append(f'{cat_feature}_encoded')
        
        self.feature_names = feature_cols
        X = df_encoded[feature_cols].values
        y = df_encoded[TARGET].values
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Features: {feature_cols}")
        
        # Scale features
        print("\nScaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Scaling complete. Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
        print("="*60 + "\n")
        
        return X_scaled, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Scaled feature matrix
        """
        df_encoded = df.copy()
        
        # Encode categorical features
        for cat_feature in CATEGORICAL_FEATURES:
            if cat_feature in df.columns:
                le = self.label_encoders[cat_feature]
                df_encoded[f'{cat_feature}_encoded'] = le.transform(df[cat_feature])
        
        # Prepare features
        X = df_encoded[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def save(self, scaler_path: str = SCALER_PATH, encoder_path: str = ENCODER_PATH):
        """
        Save fitted preprocessors to disk.
        
        Args:
            scaler_path: Path to save scaler
            encoder_path: Path to save encoders
        """
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, encoder_path)
        print(f"Saved scaler to: {scaler_path}")
        print(f"Saved encoders to: {encoder_path}")
    
    @classmethod
    def load(cls, scaler_path: str = SCALER_PATH, encoder_path: str = ENCODER_PATH):
        """
        Load fitted preprocessors from disk.
        
        Args:
            scaler_path: Path to scaler file
            encoder_path: Path to encoders file
            
        Returns:
            DataPreprocessor instance with loaded preprocessors
        """
        preprocessor = cls()
        preprocessor.scaler = joblib.load(scaler_path)
        
        encoder_data = joblib.load(encoder_path)
        preprocessor.label_encoders = encoder_data['label_encoders']
        preprocessor.feature_names = encoder_data['feature_names']
        
        print(f"Loaded preprocessors from disk")
        return preprocessor


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    return X_train, X_test, y_train, y_test


def prepare_input_features(
    input_dict: Dict,
    preprocessor: DataPreprocessor
) -> np.ndarray:
    """
    Prepare input features from dictionary for prediction.
    
    Args:
        input_dict: Dictionary with feature values
        preprocessor: Fitted DataPreprocessor instance
        
    Returns:
        Scaled feature array ready for prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_dict])
    
    # Transform using preprocessor
    X_scaled = preprocessor.transform(df)
    
    return X_scaled


if __name__ == "__main__":
    # Test preprocessing
    from .data_loader import load_yield_data
    
    df = load_yield_data()
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("\nPreprocessing test complete!")
    print(f"Feature names: {preprocessor.feature_names}")
