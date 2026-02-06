"""
Configuration file for Crop Yield Prediction Module
Contains paths, model parameters, and synthetic data for Chennai region
"""

import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data files
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), 'archive', 'yield_df.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_yield_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoders.pkl')

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# ============================================================================
# FEATURES
# ============================================================================
NUMERICAL_FEATURES = [
    'Year',
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp',
    'humidity',      # Synthetic
    'nitrogen',      # Synthetic
    'phosphorus',    # Synthetic
    'potassium',     # Synthetic
    'ph'             # Synthetic
]

CATEGORICAL_FEATURES = ['Area', 'Item']
TARGET = 'hg/ha_yield'

# ============================================================================
# SYNTHETIC DATA - CHENNAI REGION
# ============================================================================
# Source: Average values for Chennai/Tamil Nadu agricultural region
# Based on typical paddy cultivation conditions

CHENNAI_SYNTHETIC_DATA = {
    # Humidity (%)
    # Chennai has tropical climate with high humidity
    # Monsoon season: 80-90%, Summer: 60-70%, Winter: 70-80%
    'humidity': {
        'mean': 75.0,
        'std': 8.0,
        'min': 60.0,
        'max': 90.0
    },
    
    # Soil Nutrients (kg/ha)
    # Based on Tamil Nadu soil health card data for paddy fields
    'nitrogen': {
        'mean': 280.0,      # Medium fertility
        'std': 50.0,
        'min': 200.0,
        'max': 400.0
    },
    
    'phosphorus': {
        'mean': 45.0,       # Medium fertility
        'std': 15.0,
        'min': 20.0,
        'max': 80.0
    },
    
    'potassium': {
        'mean': 220.0,      # Medium fertility
        'std': 40.0,
        'min': 150.0,
        'max': 300.0
    },
    
    # Soil pH
    # Paddy fields in Tamil Nadu typically have slightly acidic to neutral soil
    'ph': {
        'mean': 6.5,
        'std': 0.5,
        'min': 5.5,
        'max': 7.5
    }
}

# ============================================================================
# CROP TYPES SUPPORTED
# ============================================================================
SUPPORTED_CROPS = [
    'Maize',
    'Potatoes',
    'Rice, paddy',
    'Wheat',
    'Sorghum',
    'Soybeans',
    'Cassava',
    'Sweet potatoes'
]

# ============================================================================
# VALIDATION RANGES
# ============================================================================
FEATURE_RANGES = {
    'Year': (1990, 2030),
    'average_rain_fall_mm_per_year': (0, 5000),
    'pesticides_tonnes': (0, 500000),
    'avg_temp': (-10, 50),
    'humidity': (0, 100),
    'nitrogen': (0, 500),
    'phosphorus': (0, 150),
    'potassium': (0, 400),
    'ph': (4.0, 9.0)
}
