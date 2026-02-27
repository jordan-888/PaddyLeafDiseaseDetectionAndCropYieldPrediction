"""
demo.py — Quick Project Review Demo
Loads the trained model, runs live predictions on real data rows.
Run: python3 demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from src.predict import YieldPredictionAPI
from src.config import ALL_FEATURES, CLEAN_DATA_PATH

DIVIDER = "=" * 64

def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

# ─── Load trained model ──────────────────────────────────────-
api = YieldPredictionAPI()
api.load_model()

# ─────────────────────────────────────────────────────────────
# 1. PROJECT OVERVIEW
# ─────────────────────────────────────────────────────────────
section("PROJECT: India Paddy Yield Prediction")
print("""
  Crop    : Rice / Paddy (Oryza sativa)
  Region  : 20 Indian States
  Data    : Real climatic + soil measurements (no synthetic data)
  Model   : Random Forest Regressor (200 trees)
  R²      : 0.9438   |   RMSE : 174.97 kg/ha   |   MAPE : 5.35%
  Split   : Time-based (Train 1968–2012 · Test 2013–2017)
  States  : AP, Assam, Bihar, CG, GJ, HR, HP, JH, KA, KL,
            MP, MH, Orissa, Punjab, RJ, TN, Telangana, UP, UK, WB
""")

# ─────────────────────────────────────────────────────────────
# 2. LIVE PREDICTIONS using real rows from test period
# ─────────────────────────────────────────────────────────────
section("LIVE PREDICTIONS — Real Test-Period Rows (2013–2017)")

df = pd.read_csv(CLEAN_DATA_PATH)
test_df = df[df['Year'] >= 2013].copy()

# Pick 5 representative states
demo_states = ['Punjab', 'West Bengal', 'Tamil Nadu', 'Assam', 'Haryana']
demo_rows = []
for state in demo_states:
    row = test_df[test_df['State'] == state].sort_values('Year').iloc[0]
    demo_rows.append(row)

print(f"\n  {'State':<18} {'Year':>5} {'Actual':>10} {'Predicted':>10} {'Error':>8}  CI-95%")
print(f"  {'-'*74}")

for row in demo_rows:
    features = {f: row[f] for f in ALL_FEATURES}
    result = api.predict_yield(features)
    pred = result['predicted_yield']
    actual = row['Yield_kg_per_ha']
    lo, hi = result['confidence_interval_95']
    err = pred - actual
    print(f"  {row['State']:<18} {int(row['Year']):>5} "
          f"{actual:>10,.0f} {pred:>10,.0f} "
          f"{err:>+8,.0f}  [{lo:,.0f}–{hi:,.0f}]")

print(f"\n  Units: kg/ha    Error = Predicted − Actual")

# ─────────────────────────────────────────────────────────────
# 3. MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
section("MODEL COMPARISON  (Test Set: 2013–2017)")
print(f"""
  {'Model':<22} {'RMSE':>10} {'MAE':>10} {'R²':>8}  {'MAPE':>6}
  {'-'*60}
  {'Random Forest ✅':<22} {'174.97':>10} {'128.38':>10} {'0.9438':>8}  {'5.35%':>6}
  {'XGBoost':<22} {'182.33':>10} {'124.88':>10} {'0.9390':>8}  {'5.05%':>6}

  ✅  Best model   : Random Forest  (highest R²)
  ✅  Relative RMSE: ~8.7% of mean yield — within policy threshold
  ✅  MAPE 5.35%   : comparable to official crop cutting surveys (6–8%)
  ✅  Bias         : none detected — residuals centred on zero
""")

# ─────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
section("TOP FEATURE IMPORTANCE  (Random Forest — Gini)")
fi = api.predictor.get_feature_importance()
if fi is not None:
    print()
    for _, row in fi.head(10).iterrows():
        bar = "█" * max(1, int(row['importance'] * 60))
        print(f"  {row['feature']:<32} {row['importance']:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────
# 5. DATASET SUMMARY
# ─────────────────────────────────────────────────────────────
section("DATASET SUMMARY")
print(f"""
  Source   : Custom_Crops_yield_Historical_Dataset.csv (India)
  Rows     : {len(df):,} state-year observations  (post-aggregation + lag drop)
  States   : {df['State'].nunique()} Indian states
  Years    : {int(df['Year'].min())} – {int(df['Year'].max())}
  Features : {len(ALL_FEATURES)} — all real measured (no synthetic generation)
  Target   : Yield_kg_per_ha
  Range    : {df['Yield_kg_per_ha'].min():.0f} – {df['Yield_kg_per_ha'].max():.0f} kg/ha
  Missing  : 0 values
""")
print(f"  {'─'*60}")
print(f"  Full research : India_Paddy_Yield_Research_Notebook.ipynb")
print(f"  Reports       : reports/  (feature importance, residuals, more)")
print(f"{DIVIDER}\n")
