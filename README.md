# Simple Property Price Prediction Model

A simplified price prediction model for Buenos Aires properties, based on the full training script.

## Overview

This is a basic implementation that:
- Loads preprocessed property data
- Uses simple features (area, bedrooms, bathrooms, location, amenities)
- Trains a Random Forest model
- Evaluates performance with RMSE, MAE, and R² metrics

## Features Used

- **Basic features**: `area`, `bedrooms`, `bathrooms`
- **Location**: `latitude`, `longitude`
- **Subway features**:
  - `distance_to_nearest_subway`: Distance to nearest subway station (km)
  - `subway_stations_nearby`: Count of subway stations within 1km
- **Bus stops**: `bus_stops_nearby`: Count of bus stops within 500m
- **Amenities**: First 5 boolean amenity features (`has_balcony`, `has_doorman`, `has_garage`, `has_grill`, `has_gym`)

**Total features**: 13 features

## Model

- **Algorithm**: Random Forest Regressor
- **Parameters**: 
  - `n_estimators=100`
  - `max_depth=10`
  - `random_state=42`

## Usage

```bash
python3 train_simple_model.py
```

## Output

The script will:
1. Load and clean the data
2. Split into train/test sets (80/20)
3. Scale features using StandardScaler
4. Train the model
5. Evaluate and print metrics
6. Save the model and scaler to `models/` directory

## Results

With subway stations and bus stops features:
- **Test RMSE**: ~$52,007
- **Test MAE**: ~$37,356
- **Test R²**: ~0.73

The model includes location-based features that capture proximity to public transportation, which is an important factor in property pricing in Buenos Aires.

## Files

- `train_simple_model.py`: Main training script
- `models/simple_price_model.joblib`: Trained model
- `models/simple_scaler.joblib`: Feature scaler

## Requirements

- pandas
- numpy
- scikit-learn
- joblib

