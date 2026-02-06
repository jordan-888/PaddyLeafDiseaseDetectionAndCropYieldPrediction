"""
Feature Importance Demo
Shows which factors matter most for crop yield
"""

import sys
sys.path.insert(0, 'src')

from model import YieldPredictor

print("\n" + "="*70)
print("🔍 FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Load model
print("\nLoading trained model...")
predictor = YieldPredictor.load_model()

# Show top features
predictor.print_feature_importance(top_n=10)

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("\n📊 What this means:")
print("  • Higher values = more important for yield prediction")
print("  • Year is often important (technology improvements over time)")
print("  • Temperature and rainfall are key environmental factors")
print("  • Soil nutrients (N, P, K) directly impact crop growth")
print("  • Area/Crop type affect baseline yield expectations")

print("\n💡 Practical Applications:")
print("  • Focus on controllable factors (soil nutrients, pesticides)")
print("  • Plan for uncontrollable factors (rainfall, temperature)")
print("  • Optimize resource allocation based on importance")
print("\n")
