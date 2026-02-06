# Crop Yield Prediction Module

**AI-Based Paddy Leaf Disease Forecasting and Crop Yield Prediction for Precision Agriculture**

A modular, production-ready machine learning system for predicting crop yields using Random Forest regression. Supports multiple crop types with environmental and soil features.

## Features

✅ **Multi-Crop Support** - Predicts yields for Rice, Maize, Wheat, Potatoes, and more  
✅ **Synthetic Chennai Data** - Includes soil nutrients (N, P, K), pH, and humidity  
✅ **Modular Architecture** - Easy to extend and replace models  
✅ **Confidence Intervals** - Provides prediction uncertainty estimates  
✅ **Feature Importance** - Identifies key yield drivers  
✅ **Production Ready** - Clean, documented, and tested code

## Project Structure

```
crop_yield_prediction/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration and synthetic data
│   ├── data_loader.py     # Data loading with synthetic features
│   ├── preprocessing.py   # Feature engineering and scaling
│   ├── model.py           # Random Forest model wrapper
│   ├── train.py           # Training pipeline
│   ├── predict.py         # Prediction API
│   └── utils.py           # Helper functions
├── models/                # Saved trained models
├── data/                  # Dataset files
└── requirements.txt       # Python dependencies
```

## Installation

```bash
# Navigate to project directory
cd crop_yield_prediction

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Model

```bash
# Train on all crops
python -m src.train

# Train on specific crops only
python -m src.train --crops "Rice, paddy" Maize Wheat

# Train without cross-validation (faster)
python -m src.train --no-cv
```

### 2. Make Predictions

```python
from src.predict import predict_yield, print_prediction_result

# Example: Predict rice yield in India
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
```

**Output:**
```
CROP YIELD PREDICTION RESULT
============================================================
Input Features:
  Area                          : India
  Item                          : Rice, paddy
  Year                          : 2024
  ...

Predicted Yield:
  45,234.56 hg/ha

95% Confidence Interval:
  [42,100.23, 48,368.89] hg/ha
  ± 1,567.33 hg/ha
============================================================
```

### 3. Using the API

```python
from src.predict import YieldPredictionAPI

# Initialize API
api = YieldPredictionAPI()
api.load_model()

# Single prediction
result = api.predict_yield(input_features)

# Batch predictions
results = api.predict_batch([input1, input2, input3])
```

## Input Features

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `Area` | str | Country/Region | 'India', 'China', 'United States' |
| `Item` | str | Crop type | 'Rice, paddy', 'Maize', 'Wheat' |
| `Year` | int | Harvest year | 2024 |
| `average_rain_fall_mm_per_year` | float | Annual rainfall (mm) | 1200.0 |
| `pesticides_tonnes` | float | Pesticide usage (tonnes) | 150.0 |
| `avg_temp` | float | Average temperature (°C) | 28.5 |
| `humidity` | float | Humidity (%) | 75.0 |
| `nitrogen` | float | Soil nitrogen (kg/ha) | 280.0 |
| `phosphorus` | float | Soil phosphorus (kg/ha) | 45.0 |
| `potassium` | float | Soil potassium (kg/ha) | 220.0 |
| `ph` | float | Soil pH | 6.5 |

## Model Performance

The Random Forest model is evaluated using:
- **RMSE** (Root Mean Squared Error) - Lower is better
- **R²** (Coefficient of Determination) - Closer to 1 is better
- **MAE** (Mean Absolute Error) - Lower is better
- **5-Fold Cross-Validation** - For robust performance estimation

## Chennai Synthetic Data

Since soil nutrients and humidity data are not in the original dataset, we generate synthetic values based on typical Chennai/Tamil Nadu agricultural conditions:

| Feature | Mean | Std Dev | Range |
|---------|------|---------|-------|
| Humidity (%) | 75.0 | 8.0 | 60-90 |
| Nitrogen (kg/ha) | 280.0 | 50.0 | 200-400 |
| Phosphorus (kg/ha) | 45.0 | 15.0 | 20-80 |
| Potassium (kg/ha) | 220.0 | 40.0 | 150-300 |
| pH | 6.5 | 0.5 | 5.5-7.5 |

## Replacing the Model

The modular design allows easy model replacement:

```python
# In model.py, replace RandomForestRegressor with:

# XGBoost
from xgboost import XGBRegressor
self.model = XGBRegressor(n_estimators=100, learning_rate=0.1)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
self.model = GradientBoostingRegressor(n_estimators=100)

# LightGBM
from lightgbm import LGBMRegressor
self.model = LGBMRegressor(n_estimators=100)
```

## Feature Importance

After training, view feature importance to understand yield drivers:

```python
from src.model import YieldPredictor

predictor = YieldPredictor.load_model()
predictor.print_feature_importance(top_n=10)
```

## Utilities

```python
from src.utils import (
    convert_hg_to_tonnes,
    plot_feature_importance,
    plot_predictions_vs_actual,
    generate_prediction_report
)

# Convert units
tonnes = convert_hg_to_tonnes(50000)  # 5.0 tonnes/ha

# Generate batch prediction report
report_df = generate_prediction_report(results, 'predictions.csv')
```

## Dataset

The model is trained on `yield_df.csv` containing:
- **28,244 records** across multiple countries and years (1990-2013)
- **8 crop types**: Maize, Potatoes, Rice (paddy), Wheat, Sorghum, Soybeans, Cassava, Sweet potatoes
- **Environmental features**: Rainfall, temperature, pesticide usage
- **Synthetic features**: Humidity, soil nutrients (N, P, K), pH

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- joblib >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

## License

This project is part of the "AI-Based Paddy Leaf Disease Forecasting and Crop Yield Prediction for Precision Agriculture" system.

## Authors

Crop Yield Prediction Team

---

**Note:** This module is designed to integrate with a CNN-based paddy leaf disease detection system. The prediction API can accept disease severity scores as optional features in future versions.
