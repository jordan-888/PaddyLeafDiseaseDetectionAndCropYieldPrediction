"""
Interactive Demo - Let audience input values
"""

import sys
sys.path.insert(0, 'src')

try:
    from predict import YieldPredictionAPI
except:
    print("⚠️  Model not trained yet. Run: python3 train_standalone.py")
    sys.exit(1)

print("\n" + "="*70)
print("🌾 INTERACTIVE CROP YIELD PREDICTOR")
print("="*70)

# Load model
print("\nLoading model...")
api = YieldPredictionAPI()
api.load_model()
print("✓ Ready!\n")

# Get user input
print("Enter crop and environmental conditions:\n")

area = input("Country (e.g., India, China, United States): ").strip()
crop = input("Crop type (e.g., Rice, paddy, Wheat, Maize): ").strip()

print("\nEnvironmental Conditions:")
rainfall = float(input("  Rainfall (mm/year, typical: 600-1500): "))
temp = float(input("  Temperature (°C, typical: 15-30): "))
pesticides = float(input("  Pesticides (tonnes, typical: 50-200): "))

print("\nSoil Conditions (or press Enter for Chennai defaults):")
humidity_input = input("  Humidity (%, default: 75): ")
nitrogen_input = input("  Nitrogen (kg/ha, default: 280): ")
phosphorus_input = input("  Phosphorus (kg/ha, default: 45): ")
potassium_input = input("  Potassium (kg/ha, default: 220): ")
ph_input = input("  pH (default: 6.5): ")

# Use defaults if empty
humidity = float(humidity_input) if humidity_input else 75.0
nitrogen = float(nitrogen_input) if nitrogen_input else 280.0
phosphorus = float(phosphorus_input) if phosphorus_input else 45.0
potassium = float(potassium_input) if potassium_input else 220.0
ph = float(ph_input) if ph_input else 6.5

# Make prediction
print("\n" + "="*70)
print("PREDICTION RESULT")
print("="*70)

try:
    result = api.predict_yield({
        'Area': area,
        'Item': crop,
        'Year': 2024,
        'average_rain_fall_mm_per_year': rainfall,
        'pesticides_tonnes': pesticides,
        'avg_temp': temp,
        'humidity': humidity,
        'nitrogen': nitrogen,
        'phosphorus': phosphorus,
        'potassium': potassium,
        'ph': ph
    })
    
    print(f"\n✅ Predicted Yield: {result['predicted_yield']:,.0f} hg/ha")
    print(f"   ({result['predicted_yield']/10000:.2f} tonnes/ha)")
    
    ci_lower, ci_upper = result['confidence_interval_95']
    print(f"\n📊 95% Confidence Interval:")
    print(f"   [{ci_lower:,.0f}, {ci_upper:,.0f}] hg/ha")
    print(f"   ({ci_lower/10000:.2f}, {ci_upper/10000:.2f} tonnes/ha)")
    
    print("\n" + "="*70)
    
except ValueError as e:
    print(f"\n❌ Error: {e}")
    print("Please check your input values and try again.")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")

print("\n")
