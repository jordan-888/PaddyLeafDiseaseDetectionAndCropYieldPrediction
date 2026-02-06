"""
Standalone Training Script
Run this to train the crop yield prediction model
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import joblib
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("\nPlease install required packages:")
    print("  pip3 install --user pandas numpy scikit-learn joblib")
    print("\nOr use:")
    print("  pip3 install --break-system-packages pandas numpy scikit-learn joblib")
    sys.exit(1)

# Configuration
DATA_PATH = '../archive/yield_df.csv'
MODEL_DIR = 'models'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Chennai synthetic data parameters
CHENNAI_DATA = {
    'humidity': {'mean': 75.0, 'std': 8.0, 'min': 60.0, 'max': 90.0},
    'nitrogen': {'mean': 280.0, 'std': 50.0, 'min': 200.0, 'max': 400.0},
    'phosphorus': {'mean': 45.0, 'std': 15.0, 'min': 20.0, 'max': 80.0},
    'potassium': {'mean': 220.0, 'std': 40.0, 'min': 150.0, 'max': 300.0},
    'ph': {'mean': 6.5, 'std': 0.5, 'min': 5.5, 'max': 7.5}
}


def main():
    print("\n" + "="*70)
    print(" "*15 + "CROP YIELD PREDICTION - TRAINING")
    print("="*70)
    
    # Step 1: Load data
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    print(f"Loaded {len(df):,} records")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle missing values
    df = df.dropna(subset=['hg/ha_yield'])
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Add synthetic features
    print("\nAdding synthetic Chennai features...")
    np.random.seed(RANDOM_STATE)
    n_samples = len(df)
    
    for feature, params in CHENNAI_DATA.items():
        values = np.random.normal(params['mean'], params['std'], n_samples)
        values = np.clip(values, params['min'], params['max'])
        df[feature] = values
        print(f"  Added {feature}")
    
    print(f"\nDataset: {len(df):,} records, {len(df.columns)} features")
    
    # Step 2: Preprocess
    print("\n[2/5] Preprocessing data...")
    
    # Encode categorical features
    le_area = LabelEncoder()
    le_item = LabelEncoder()
    df['Area_encoded'] = le_area.fit_transform(df['Area'])
    df['Item_encoded'] = le_item.fit_transform(df['Item'])
    
    # Prepare features
    feature_cols = [
        'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp',
        'humidity', 'nitrogen', 'phosphorus', 'potassium', 'ph',
        'Area_encoded', 'Item_encoded'
    ]
    
    X = df[feature_cols].values
    y = df['hg/ha_yield'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    print(f"Features: {len(feature_cols)}")
    
    # Step 3: Train model
    print("\n[3/5] Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # Step 4: Evaluate
    print("\n[4/5] Evaluating model...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\nPerformance Metrics:")
    print(f"  RMSE: {rmse:,.2f} hg/ha")
    print(f"  MAE:  {mae:,.2f} hg/ha")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    for feature, importance in feature_importance[:10]:
        bar_length = int(importance * 50)
        bar = '█' * bar_length
        print(f"{feature:30s} {importance:.4f} {bar}")
    
    # Step 5: Save model
    print("\n[5/5] Saving model and preprocessors...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_cols,
        'is_trained': True
    }
    
    encoder_data = {
        'label_encoders': {'Area': le_area, 'Item': le_item},
        'feature_names': feature_cols
    }
    
    joblib.dump(model_data, os.path.join(MODEL_DIR, 'rf_yield_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(encoder_data, os.path.join(MODEL_DIR, 'encoders.pkl'))
    
    print(f"✓ Model saved to: {MODEL_DIR}/rf_yield_model.pkl")
    print(f"✓ Scaler saved to: {MODEL_DIR}/scaler.pkl")
    print(f"✓ Encoders saved to: {MODEL_DIR}/encoders.pkl")
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Performance:")
    print(f"  Test RMSE: {rmse:,.2f} hg/ha")
    print(f"  Test R²:   {r2:.4f}")
    print(f"\nModel ready for predictions!")
    print("\n")


if __name__ == "__main__":
    main()
