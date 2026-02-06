"""
Prediction Interface
API for making crop yield predictions with trained model
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, List
from .model import YieldPredictor
from .preprocessing import DataPreprocessor
from .config import (
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    FEATURE_RANGES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


class YieldPredictionAPI:
    """
    High-level API for crop yield predictions.
    """
    
    def __init__(self):
        self.predictor = None
        self.preprocessor = None
        self.is_loaded = False
    
    def load_model(self):
        """
        Load trained model and preprocessors from disk.
        """
        print("Loading trained model and preprocessors...")
        self.predictor = YieldPredictor.load_model(MODEL_PATH)
        self.preprocessor = DataPreprocessor.load(SCALER_PATH, ENCODER_PATH)
        self.is_loaded = True
        print("Model loaded successfully!")
    
    def validate_input(self, input_features: Dict) -> Dict:
        """
        Validate and clean input features.
        
        Args:
            input_features: Dictionary of input features
            
        Returns:
            Validated and cleaned input dictionary
            
        Raises:
            ValueError: If required features are missing or out of range
        """
        # Check required features
        required_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        missing = [f for f in required_features if f not in input_features]
        
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Validate numerical ranges
        for feature, (min_val, max_val) in FEATURE_RANGES.items():
            if feature in input_features:
                value = input_features[feature]
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"{feature} = {value} is out of valid range [{min_val}, {max_val}]"
                    )
        
        return input_features
    
    def predict_yield(self, input_features: Dict) -> Dict:
        """
        Predict crop yield from input features.
        
        Args:
            input_features: Dictionary with the following keys:
                - Area: str (e.g., 'India', 'China', 'United States')
                - Item: str (e.g., 'Rice, paddy', 'Maize', 'Wheat')
                - Year: int (e.g., 2024)
                - average_rain_fall_mm_per_year: float
                - pesticides_tonnes: float
                - avg_temp: float (°C)
                - humidity: float (%)
                - nitrogen: float (kg/ha)
                - phosphorus: float (kg/ha)
                - potassium: float (kg/ha)
                - ph: float
                
        Returns:
            Dictionary with:
                - predicted_yield: float (hg/ha)
                - confidence_interval_95: tuple (lower, upper)
                - unit: str
                - input_features: dict (validated inputs)
                
        Example:
            >>> api = YieldPredictionAPI()
            >>> api.load_model()
            >>> result = api.predict_yield({
            ...     'Area': 'India',
            ...     'Item': 'Rice, paddy',
            ...     'Year': 2024,
            ...     'average_rain_fall_mm_per_year': 1200,
            ...     'pesticides_tonnes': 150,
            ...     'avg_temp': 28.5,
            ...     'humidity': 75,
            ...     'nitrogen': 280,
            ...     'phosphorus': 45,
            ...     'potassium': 220,
            ...     'ph': 6.5
            ... })
            >>> print(f"Predicted yield: {result['predicted_yield']:.2f} hg/ha")
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input
        validated_input = self.validate_input(input_features)
        
        # Create DataFrame
        df = pd.DataFrame([validated_input])
        
        # Preprocess
        X = self.preprocessor.transform(df)
        
        # Predict with confidence
        predictions, std_devs = self.predictor.predict_with_confidence(X)
        
        predicted_yield = predictions[0]
        std_dev = std_devs[0]
        
        # 95% confidence interval (1.96 * std)
        confidence_interval = (
            max(0, predicted_yield - 1.96 * std_dev),  # Yield can't be negative
            predicted_yield + 1.96 * std_dev
        )
        
        return {
            'predicted_yield': float(predicted_yield),
            'confidence_interval_95': confidence_interval,
            'std_deviation': float(std_dev),
            'unit': 'hg/ha',
            'input_features': validated_input
        }
    
    def predict_batch(self, input_list: List[Dict]) -> List[Dict]:
        """
        Predict yields for multiple inputs.
        
        Args:
            input_list: List of input feature dictionaries
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        for input_features in input_list:
            try:
                result = self.predict_yield(input_features)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'input_features': input_features
                })
        
        return results


def predict_yield(input_features: Dict) -> Dict:
    """
    Convenience function for single prediction.
    
    Loads model, makes prediction, and returns result.
    
    Args:
        input_features: Dictionary of input features
        
    Returns:
        Prediction result dictionary
    """
    api = YieldPredictionAPI()
    api.load_model()
    return api.predict_yield(input_features)


def print_prediction_result(result: Dict):
    """
    Print prediction result in a formatted way.
    
    Args:
        result: Prediction result dictionary
    """
    if 'error' in result:
        print(f"\n❌ Prediction Error: {result['error']}")
        return
    
    print("\n" + "="*60)
    print("CROP YIELD PREDICTION RESULT")
    print("="*60)
    
    # Input features
    print("\nInput Features:")
    for key, value in result['input_features'].items():
        print(f"  {key:30s}: {value}")
    
    # Prediction
    print(f"\nPredicted Yield:")
    print(f"  {result['predicted_yield']:,.2f} {result['unit']}")
    
    # Confidence interval
    ci_lower, ci_upper = result['confidence_interval_95']
    print(f"\n95% Confidence Interval:")
    print(f"  [{ci_lower:,.2f}, {ci_upper:,.2f}] {result['unit']}")
    print(f"  ± {result['std_deviation']:,.2f} {result['unit']}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Testing Crop Yield Prediction API\n")
    
    # Example input for Rice in India
    example_input = {
        'Area': 'India',
        'Item': 'Rice, paddy',
        'Year': 2024,
        'average_rain_fall_mm_per_year': 1200.0,
        'pesticides_tonnes': 150.0,
        'avg_temp': 28.5,
        'humidity': 75.0,
        'nitrogen': 280.0,
        'phosphorus': 45.0,
        'potassium': 220.0,
        'ph': 6.5
    }
    
    try:
        result = predict_yield(example_input)
        print_prediction_result(result)
    except FileNotFoundError:
        print("❌ Model not found. Please train the model first using train.py")
    except Exception as e:
        print(f"❌ Error: {e}")
