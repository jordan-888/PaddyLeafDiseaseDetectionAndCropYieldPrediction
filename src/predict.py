"""
Prediction API — India Paddy Yield (Real Data)
Accepts only real, measured features. No synthetic generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from .model import YieldPredictor
from .preprocessing import DataPreprocessor
from .config import (
    MODEL_PATH, PREPROCESSOR_PATH,
    ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    TARGET_COLUMN, FEATURE_RANGES
)

# ---------------------------------------------------------------------------
# Required input fields for prediction-time
# ---------------------------------------------------------------------------
REQUIRED_INPUT_FIELDS = (
    CATEGORICAL_FEATURES +           # ['State']
    [f for f in NUMERICAL_FEATURES   # all real + lag features
     if f != 'Year']
    + ['Year']
)


class YieldPredictionAPI:
    """High-level prediction interface for India paddy yield."""

    def __init__(self):
        self.predictor: YieldPredictor = None
        self.preprocessor: DataPreprocessor = None
        self.is_loaded = False

    def load_model(self):
        print("Loading model and preprocessor...")
        self.predictor    = YieldPredictor.load(MODEL_PATH)
        self.preprocessor = DataPreprocessor.load(PREPROCESSOR_PATH)
        self.is_loaded    = True
        print("Ready.")

    # ------------------------------------------------------------------
    def validate_input(self, features: Dict) -> Dict:
        """Check required fields and numeric ranges."""
        missing = [f for f in ALL_FEATURES if f not in features]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        for feat, (lo, hi) in FEATURE_RANGES.items():
            if feat in features:
                val = features[feat]
                if not (lo <= val <= hi):
                    raise ValueError(
                        f"{feat}={val} out of valid range [{lo}, {hi}]"
                    )
        return features

    # ------------------------------------------------------------------
    def predict_yield(self, input_features: Dict) -> Dict:
        """
        Predict paddy yield for one sample.

        Required keys (example):
            State              : 'Tamil Nadu'
            Year               : 2022
            Rainfall_mm        : 1100.0
            Temperature_C      : 27.5
            Humidity_%         : 78.0
            N_req_kg_per_ha    : 8.5
            P_req_kg_per_ha    : 4.0
            K_req_kg_per_ha    : 7.0
            pH                 : 6.5
            Wind_Speed_m_s     : 2.1
            Solar_Radiation_MJ_m2_day: 18.0
            Yield_t1           : 2500.0   # previous year yield
            Yield_t2           : 2350.0   # two years ago yield
            Rainfall_t1        : 1050.0
            Rainfall_t2        : 980.0
            Temp_t1            : 27.0

        Returns dict with predicted_yield, confidence_interval_95, unit.
        """
        if not self.is_loaded:
            raise RuntimeError("Call load_model() first.")

        validated = self.validate_input(input_features)
        df = pd.DataFrame([validated])
        X  = self.preprocessor.transform(df)

        preds, stds = self.predictor.predict_with_confidence(X)
        yield_val   = float(preds[0])
        std_val     = float(stds[0])

        return {
            'predicted_yield':        yield_val,
            'confidence_interval_95': (
                max(0.0, yield_val - 1.96 * std_val),
                yield_val + 1.96 * std_val,
            ),
            'std_deviation': std_val,
            'unit':          'kg/ha',
            'model_type':    self.predictor.model_type,
            'input_features': validated,
        }

    # ------------------------------------------------------------------
    def predict_batch(self, input_list: List[Dict]) -> List[Dict]:
        results = []
        for inp in input_list:
            try:
                results.append(self.predict_yield(inp))
            except Exception as e:
                results.append({'error': str(e), 'input_features': inp})
        return results


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def predict_yield(input_features: Dict) -> Dict:
    api = YieldPredictionAPI()
    api.load_model()
    return api.predict_yield(input_features)


def print_prediction_result(result: Dict):
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
        return

    print("\n" + "="*60)
    print("  INDIA PADDY YIELD PREDICTION")
    print("="*60)
    print(f"\n  State : {result['input_features'].get('State')}")
    print(f"  Year  : {result['input_features'].get('Year')}")
    print(f"\n  Predicted Yield : {result['predicted_yield']:,.2f} {result['unit']}")
    lo, hi = result['confidence_interval_95']
    print(f"  95% CI          : [{lo:,.2f}, {hi:,.2f}] {result['unit']}")
    print(f"  Std Deviation   : ±{result['std_deviation']:,.2f} {result['unit']}")
    print(f"  Model           : {result['model_type'].replace('_',' ').title()}")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Quick test — replace values with real state/year data
    example = {
        'State':                     'Tamil Nadu',
        'Year':                      2014,
        'Rainfall_mm':               1200.0,
        'Temperature_C':             28.0,
        'Humidity_%':                78.0,
        'N_req_kg_per_ha':           8.5,
        'P_req_kg_per_ha':           4.0,
        'K_req_kg_per_ha':           7.0,
        'pH':                        6.5,
        'Wind_Speed_m_s':            2.0,
        'Solar_Radiation_MJ_m2_day': 18.0,
        'Yield_t1':                  2400.0,
        'Yield_t2':                  2200.0,
        'Rainfall_t1':               1100.0,
        'Rainfall_t2':               1050.0,
        'Temp_t1':                   27.5,
    }
    try:
        result = predict_yield(example)
        print_prediction_result(result)
    except FileNotFoundError:
        print("❌ Model not found. Run train_standalone.py first.")
