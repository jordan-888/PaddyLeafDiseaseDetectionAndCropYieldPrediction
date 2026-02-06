"""
Model Module
Random Forest Regressor wrapper for crop yield prediction
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple
from .config import RF_PARAMS, MODEL_PATH, CV_FOLDS


class YieldPredictor:
    """
    Wrapper class for Random Forest crop yield prediction model.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the yield predictor.
        
        Args:
            **kwargs: Random Forest parameters (overrides config defaults)
        """
        # Merge default params with custom params
        params = {**RF_PARAMS, **kwargs}
        self.model = RandomForestRegressor(**params)
        self.feature_names = None
        self.is_trained = False
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Optional list of feature names
        """
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        print(f"Training samples: {X_train.shape[0]:,}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Model: Random Forest Regressor")
        print(f"Parameters: {self.model.get_params()}")
        
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print("Training complete!")
        print("="*60 + "\n")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted yield values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Uses prediction std from individual trees.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, std_deviations)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Calculate mean and std
        predictions = tree_predictions.mean(axis=0)
        std_devs = tree_predictions.std(axis=0)
        
        return predictions, std_devs
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        print(f"Test samples: {len(y_test):,}")
        print(f"\nPerformance Metrics:")
        print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:,.2f} hg/ha")
        print(f"  MAE  (Mean Absolute Error):     {metrics['mae']:,.2f} hg/ha")
        print(f"  R²   (Coefficient of Determination): {metrics['r2']:.4f}")
        print(f"  MAPE (Mean Absolute % Error):   {metrics['mape']:.2f}%")
        print("="*60 + "\n")
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = CV_FOLDS) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of folds
            
        Returns:
            Dictionary with CV scores
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        # Negative MSE for cross_val_score (it maximizes)
        neg_mse_scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rmse_scores = np.sqrt(-neg_mse_scores)
        
        r2_scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='r2',
            n_jobs=-1
        )
        
        results = {
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std(),
            'cv_r2_mean': r2_scores.mean(),
            'cv_r2_std': r2_scores.std()
        }
        
        print(f"  CV RMSE: {results['cv_rmse_mean']:,.2f} ± {results['cv_rmse_std']:,.2f}")
        print(f"  CV R²:   {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        
        if self.feature_names:
            feature_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            })
        else:
            feature_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importances))],
                'importance': importances
            })
        
        feature_df = feature_df.sort_values('importance', ascending=False)
        
        return feature_df
    
    def print_feature_importance(self, top_n: int = None):
        """
        Print feature importance in a formatted way.
        
        Args:
            top_n: Number of top features to show (None = all)
        """
        importance_df = self.get_feature_importance()
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        
        for idx, row in importance_df.iterrows():
            bar_length = int(row['importance'] * 50)
            bar = '█' * bar_length
            print(f"{row['feature']:30s} {row['importance']:.4f} {bar}")
        
        print("="*60 + "\n")
    
    def save_model(self, path: str = MODEL_PATH):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to: {path}")
    
    @classmethod
    def load_model(cls, path: str = MODEL_PATH):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            YieldPredictor instance with loaded model
        """
        model_data = joblib.load(path)
        
        predictor = cls()
        predictor.model = model_data['model']
        predictor.feature_names = model_data['feature_names']
        predictor.is_trained = model_data['is_trained']
        
        print(f"Model loaded from: {path}")
        return predictor


if __name__ == "__main__":
    # Test model creation
    predictor = YieldPredictor()
    print(f"Model created: {predictor.model}")
    print(f"Parameters: {predictor.model.get_params()}")
