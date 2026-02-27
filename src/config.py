"""
Configuration — India Paddy Yield Prediction (Real Data)
No synthetic feature generation. All features from real measurements.
"""

import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

# Raw source dataset (real soil + weather + yield)
RAW_DATA_PATH = os.path.join(
    os.path.dirname(BASE_DIR), 'archive',
    'Custom_Crops_yield_Historical_Dataset.csv'
)

# Cleaned output dataset
CLEAN_DATA_PATH = os.path.join(DATA_DIR, 'india_paddy_clean.csv')

# Model artefacts
MODEL_PATH      = os.path.join(MODEL_DIR, 'yield_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')

# ============================================================================
# DATA SETTINGS
# ============================================================================
TARGET_CROP   = 'rice'
TARGET_COLUMN = 'Yield_kg_per_ha'

# Time-based train / test split — NO random split, no leakage
TRAIN_END_YEAR  = 2012
TEST_START_YEAR = 2013

# Cross-validation folds (within training window only)
CV_FOLDS = 5

# ============================================================================
# FEATURES  (all real — no synthetic)
# ============================================================================
NUMERICAL_FEATURES = [
    'Year',
    'Rainfall_mm',
    'Temperature_C',
    'Humidity_%',
    'N_req_kg_per_ha',
    'P_req_kg_per_ha',
    'K_req_kg_per_ha',
    'pH',
    'Wind_Speed_m_s',
    'Solar_Radiation_MJ_m2_day',
    # Lag features (time-aware)
    'Yield_t1',
    'Yield_t2',
    'Rainfall_t1',
    'Rainfall_t2',
    'Temp_t1',
]

CATEGORICAL_FEATURES = ['State']   # label-encoded state name

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
RANDOM_STATE = 42

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

# ============================================================================
# VALIDATION RANGES  (for prediction-time input checks)
# ============================================================================
FEATURE_RANGES = {
    'Year':                   (1960, 2030),
    'Rainfall_mm':            (100,  5000),
    'Temperature_C':          (10,   45),
    'Humidity_%':             (20,   100),
    'N_req_kg_per_ha':        (0,    200),
    'P_req_kg_per_ha':        (0,    100),
    'K_req_kg_per_ha':        (0,    200),
    'pH':                     (4.0,  9.0),
    'Wind_Speed_m_s':         (0,    20),
    'Solar_Radiation_MJ_m2_day': (5, 35),
    'Yield_t1':               (0,    10000),
    'Yield_t2':               (0,    10000),
    'Rainfall_t1':            (100,  5000),
    'Rainfall_t2':            (100,  5000),
    'Temp_t1':                (10,   45),
}

SUPPORTED_CROPS  = ['rice']
SUPPORTED_STATES = None   # populated at load time
