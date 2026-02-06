"""
Example Usage Scripts for Crop Yield Prediction Module
"""

# ============================================================================
# EXAMPLE 1: Basic Training
# ============================================================================

def example_1_basic_training():
    """Train the model on all crops"""
    from src.train import train_model
    
    predictor, preprocessor, metrics = train_model(
        crop_filter=None,  # All crops
        save_model=True,
        perform_cv=True
    )
    
    print(f"Model trained with R² = {metrics['r2']:.4f}")


# ============================================================================
# EXAMPLE 2: Train on Specific Crops
# ============================================================================

def example_2_specific_crops():
    """Train only on rice and wheat"""
    from src.train import train_model
    
    predictor, preprocessor, metrics = train_model(
        crop_filter=['Rice, paddy', 'Wheat'],
        save_model=True,
        perform_cv=True
    )


# ============================================================================
# EXAMPLE 3: Single Prediction
# ============================================================================

def example_3_single_prediction():
    """Make a single yield prediction"""
    from src.predict import predict_yield, print_prediction_result
    
    # Input features for rice in India
    input_features = {
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
    
    result = predict_yield(input_features)
    print_prediction_result(result)
    
    # Access specific values
    yield_value = result['predicted_yield']
    ci_lower, ci_upper = result['confidence_interval_95']
    
    print(f"Predicted: {yield_value:,.0f} hg/ha")
    print(f"Range: [{ci_lower:,.0f}, {ci_upper:,.0f}]")


# ============================================================================
# EXAMPLE 4: Batch Predictions
# ============================================================================

def example_4_batch_predictions():
    """Make predictions for multiple scenarios"""
    from src.predict import YieldPredictionAPI
    
    api = YieldPredictionAPI()
    api.load_model()
    
    # Multiple scenarios
    scenarios = [
        {
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
        },
        {
            'Area': 'China',
            'Item': 'Wheat',
            'Year': 2024,
            'average_rain_fall_mm_per_year': 600.0,
            'pesticides_tonnes': 200.0,
            'avg_temp': 15.0,
            'humidity': 65.0,
            'nitrogen': 300.0,
            'phosphorus': 50.0,
            'potassium': 250.0,
            'ph': 7.0
        }
    ]
    
    results = api.predict_batch(scenarios)
    
    for i, result in enumerate(results, 1):
        print(f"\nScenario {i}:")
        print(f"  Crop: {result['input_features']['Item']}")
        print(f"  Predicted Yield: {result['predicted_yield']:,.0f} hg/ha")


# ============================================================================
# EXAMPLE 5: Feature Importance Analysis
# ============================================================================

def example_5_feature_importance():
    """Analyze which features are most important"""
    from src.model import YieldPredictor
    
    # Load trained model
    predictor = YieldPredictor.load_model()
    
    # Get feature importance
    importance_df = predictor.get_feature_importance()
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Print formatted
    predictor.print_feature_importance(top_n=10)


# ============================================================================
# EXAMPLE 6: Comparing Different Scenarios
# ============================================================================

def example_6_scenario_comparison():
    """Compare yield under different conditions"""
    from src.predict import YieldPredictionAPI
    
    api = YieldPredictionAPI()
    api.load_model()
    
    # Base scenario
    base = {
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
    
    # Scenario with higher rainfall
    high_rainfall = base.copy()
    high_rainfall['average_rain_fall_mm_per_year'] = 1500.0
    
    # Scenario with better soil nutrients
    better_soil = base.copy()
    better_soil['nitrogen'] = 350.0
    better_soil['phosphorus'] = 60.0
    better_soil['potassium'] = 280.0
    
    # Compare
    base_result = api.predict_yield(base)
    rainfall_result = api.predict_yield(high_rainfall)
    soil_result = api.predict_yield(better_soil)
    
    print("\nScenario Comparison:")
    print(f"Base:           {base_result['predicted_yield']:,.0f} hg/ha")
    print(f"High Rainfall:  {rainfall_result['predicted_yield']:,.0f} hg/ha "
          f"({((rainfall_result['predicted_yield']/base_result['predicted_yield']-1)*100):+.1f}%)")
    print(f"Better Soil:    {soil_result['predicted_yield']:,.0f} hg/ha "
          f"({((soil_result['predicted_yield']/base_result['predicted_yield']-1)*100):+.1f}%)")


# ============================================================================
# EXAMPLE 7: Using Utilities
# ============================================================================

def example_7_utilities():
    """Use utility functions"""
    from src.utils import convert_hg_to_tonnes, format_yield, calculate_yield_change
    
    # Convert units
    yield_hg = 50000
    yield_tonnes = convert_hg_to_tonnes(yield_hg)
    print(f"{yield_hg} hg/ha = {yield_tonnes} tonnes/ha")
    
    # Format for display
    formatted = format_yield(yield_hg, unit='tonnes/ha')
    print(f"Formatted: {formatted}")
    
    # Calculate change
    old_yield = 45000
    new_yield = 50000
    change = calculate_yield_change(old_yield, new_yield)
    print(f"Change: {change['percent_change']:.1f}% {change['direction']}")


# ============================================================================
# EXAMPLE 8: Generate Prediction Report
# ============================================================================

def example_8_generate_report():
    """Generate a CSV report from multiple predictions"""
    from src.predict import YieldPredictionAPI
    from src.utils import generate_prediction_report
    
    api = YieldPredictionAPI()
    api.load_model()
    
    # Multiple predictions
    scenarios = [
        # ... (define multiple scenarios)
    ]
    
    results = api.predict_batch(scenarios)
    
    # Generate report
    report_df = generate_prediction_report(results, 'predictions_report.csv')
    print(f"Report saved with {len(report_df)} predictions")


# ============================================================================
# EXAMPLE 9: Replacing the Model
# ============================================================================

def example_9_replace_model():
    """Example of replacing Random Forest with XGBoost"""
    
    # In model.py, modify the YieldPredictor class:
    """
    from xgboost import XGBRegressor
    
    class YieldPredictor:
        def __init__(self, **kwargs):
            self.model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
    """
    
    # Then train as usual
    from src.train import train_model
    predictor, preprocessor, metrics = train_model()


# ============================================================================
# EXAMPLE 10: Custom Feature Ranges
# ============================================================================

def example_10_custom_validation():
    """Validate inputs with custom ranges"""
    from src.predict import YieldPredictionAPI
    
    api = YieldPredictionAPI()
    api.load_model()
    
    # This will raise ValueError if out of range
    try:
        result = api.predict_yield({
            'Area': 'India',
            'Item': 'Rice, paddy',
            'Year': 2050,  # Out of range
            # ... other features
        })
    except ValueError as e:
        print(f"Validation error: {e}")


if __name__ == "__main__":
    print("Crop Yield Prediction - Example Usage")
    print("\nRun individual examples by calling the functions above.")
    print("\nAvailable examples:")
    print("  1. Basic Training")
    print("  2. Train on Specific Crops")
    print("  3. Single Prediction")
    print("  4. Batch Predictions")
    print("  5. Feature Importance Analysis")
    print("  6. Scenario Comparison")
    print("  7. Using Utilities")
    print("  8. Generate Prediction Report")
    print("  9. Replacing the Model")
    print("  10. Custom Validation")
