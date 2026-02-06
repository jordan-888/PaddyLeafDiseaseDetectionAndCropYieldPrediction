"""
Live Demo Script - Crop Yield Prediction
Run this during your presentation
"""

import sys
sys.path.insert(0, 'src')

try:
    from predict import YieldPredictionAPI, print_prediction_result
except:
    print("⚠️  Model not trained yet. Run: python3 train_standalone.py")
    sys.exit(1)

print("\n" + "="*70)
print("🌾 CROP YIELD PREDICTION - LIVE DEMO")
print("="*70)

# Initialize API
print("\n[1] Loading trained model...")
api = YieldPredictionAPI()
api.load_model()
print("✓ Model loaded successfully!")

# Demo Scenario 1: Rice in India (Good conditions)
print("\n" + "="*70)
print("SCENARIO 1: Rice Farming in India (Optimal Conditions)")
print("="*70)

india_rice_good = {
    'Area': 'India',
    'Item': 'Rice, paddy',
    'Year': 2024,
    'average_rain_fall_mm_per_year': 1200.0,  # Good rainfall
    'pesticides_tonnes': 150.0,
    'avg_temp': 28.5,                          # Ideal temperature
    'humidity': 75.0,                          # Good humidity
    'nitrogen': 300.0,                         # High nitrogen
    'phosphorus': 50.0,                        # Good phosphorus
    'potassium': 250.0,                        # High potassium
    'ph': 6.5                                  # Optimal pH
}

result1 = api.predict_yield(india_rice_good)
print_prediction_result(result1)

input("\n⏸️  Press Enter to continue to Scenario 2...")

# Demo Scenario 2: Rice in India (Poor conditions)
print("\n" + "="*70)
print("SCENARIO 2: Rice Farming in India (Poor Conditions)")
print("="*70)

india_rice_poor = {
    'Area': 'India',
    'Item': 'Rice, paddy',
    'Year': 2024,
    'average_rain_fall_mm_per_year': 600.0,   # Low rainfall (drought)
    'pesticides_tonnes': 50.0,
    'avg_temp': 32.0,                          # Too hot
    'humidity': 60.0,                          # Low humidity
    'nitrogen': 200.0,                         # Low nitrogen
    'phosphorus': 25.0,                        # Low phosphorus
    'potassium': 150.0,                        # Low potassium
    'ph': 5.8                                  # Slightly acidic
}

result2 = api.predict_yield(india_rice_poor)
print_prediction_result(result2)

# Show comparison
print("\n" + "="*70)
print("📊 COMPARISON")
print("="*70)
yield_diff = result1['predicted_yield'] - result2['predicted_yield']
percent_diff = (yield_diff / result2['predicted_yield']) * 100

print(f"Optimal Conditions: {result1['predicted_yield']:,.0f} hg/ha")
print(f"Poor Conditions:    {result2['predicted_yield']:,.0f} hg/ha")
print(f"\nDifference: {yield_diff:,.0f} hg/ha ({percent_diff:.1f}% higher with optimal conditions)")

input("\n⏸️  Press Enter to continue to Scenario 3...")

# Demo Scenario 3: Different Crop (Wheat in China)
print("\n" + "="*70)
print("SCENARIO 3: Wheat Farming in China")
print("="*70)

china_wheat = {
    'Area': 'China',
    'Item': 'Wheat',
    'Year': 2024,
    'average_rain_fall_mm_per_year': 600.0,
    'pesticides_tonnes': 200.0,
    'avg_temp': 15.0,                          # Cooler for wheat
    'humidity': 65.0,
    'nitrogen': 320.0,
    'phosphorus': 55.0,
    'potassium': 270.0,
    'ph': 7.0                                  # Neutral pH
}

result3 = api.predict_yield(china_wheat)
print_prediction_result(result3)

print("\n" + "="*70)
print("✅ DEMO COMPLETE!")
print("="*70)
print("\nKey Takeaways:")
print("  • Model supports multiple crops and regions")
print("  • Provides confidence intervals for predictions")
print("  • Can compare different scenarios")
print("  • Helps farmers make data-driven decisions")
print("\n")
