"""
Training Pipeline
Complete workflow for training the crop yield prediction model
"""

import os
import sys
from .data_loader import load_yield_data, print_dataset_summary
from .preprocessing import DataPreprocessor, split_data
from .model import YieldPredictor
from .config import MODEL_DIR


def train_model(
    crop_filter: list = None,
    save_model: bool = True,
    perform_cv: bool = True
):
    """
    Complete training pipeline for crop yield prediction.
    
    Args:
        crop_filter: List of crops to include (None = all crops)
        save_model: Whether to save the trained model
        perform_cv: Whether to perform cross-validation
        
    Returns:
        Tuple of (predictor, preprocessor, metrics)
    """
    print("\n" + "="*70)
    print(" "*15 + "CROP YIELD PREDICTION - TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Load data
    print("\n[1/5] Loading dataset...")
    df = load_yield_data(crop_filter=crop_filter, add_synthetic=True)
    print_dataset_summary(df)
    
    # Step 2: Preprocess data
    print("\n[2/5] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 3: Train model
    print("\n[3/5] Training Random Forest model...")
    predictor = YieldPredictor()
    predictor.train(X_train, y_train, feature_names=preprocessor.feature_names)
    
    # Step 4: Evaluate model
    print("\n[4/5] Evaluating model...")
    metrics = predictor.evaluate(X_test, y_test)
    
    # Cross-validation
    if perform_cv:
        cv_metrics = predictor.cross_validate(X, y)
        metrics.update(cv_metrics)
    
    # Feature importance
    predictor.print_feature_importance(top_n=15)
    
    # Step 5: Save model
    if save_model:
        print("\n[5/5] Saving model and preprocessors...")
        
        # Create models directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        predictor.save_model()
        preprocessor.save()
        
        print("\nModel training complete!")
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING PIPELINE COMPLETE")
    print("="*70 + "\n")
    
    return predictor, preprocessor, metrics


def main():
    """
    Main entry point for training script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Crop Yield Prediction Model')
    parser.add_argument(
        '--crops',
        nargs='+',
        default=None,
        help='List of crops to include (default: all crops)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save the trained model'
    )
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Skip cross-validation'
    )
    
    args = parser.parse_args()
    
    # Train model
    predictor, preprocessor, metrics = train_model(
        crop_filter=args.crops,
        save_model=not args.no_save,
        perform_cv=not args.no_cv
    )
    
    # Print final summary
    print("\nFinal Model Performance:")
    print(f"  Test RMSE: {metrics['rmse']:,.2f} hg/ha")
    print(f"  Test R²:   {metrics['r2']:.4f}")
    if 'cv_rmse_mean' in metrics:
        print(f"  CV RMSE:   {metrics['cv_rmse_mean']:,.2f} ± {metrics['cv_rmse_std']:,.2f}")
        print(f"  CV R²:     {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")


if __name__ == "__main__":
    main()
