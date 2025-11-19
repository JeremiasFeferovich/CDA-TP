#!/usr/bin/env python3
"""
Test model performance with reduced area-related features
Compare different feature combinations to find optimal set
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load data (reuse logic from train_simple_model.py)
print("=" * 80)
print("TESTING REDUCED AREA FEATURES")
print("=" * 80)

df = pd.read_csv('/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/data/properati_preprocessed_normalized_20251016_204913.csv')
df = df[df['currency'] == 'USD'].copy()
df = df[(df['price'] >= 30_000) & (df['price'] <= 600_000)].copy()

# Basic filtering
price_per_sqm = df['price'] / df['area']
df = df[(price_per_sqm >= 800) & (price_per_sqm <= 5000)].copy()
df = df.dropna(subset=['latitude', 'longitude'])

print(f"Dataset size: {len(df):,} properties\n")

# Load existing trained model to get all 22 features
import joblib
ensemble = joblib.load('models/ensemble_price_model.joblib')
all_22_features = list(ensemble['xgb_model'].feature_names_in_)

print(f"Current 22 features from trained model:")
print(all_22_features)
print()

# Create minimal feature engineering (simplified from train_simple_model.py)
# We just need the area-related features to test
df['log_area'] = np.log1p(df['area'])
df['area_x_bedrooms'] = df['area'] * df['bedrooms']
df['area_x_bathrooms'] = df['area'] * df['bathrooms']
df['area_per_bedroom'] = df['area'] / df['bedrooms'].replace(0, 1)

# Identify area-related features
area_related = ['area', 'log_area', 'area_x_bedrooms', 'area_x_bathrooms', 'area_per_bedroom']
area_in_model = [f for f in area_related if f in all_22_features]

print(f"Area-related features in current model ({len(area_in_model)}/22 = {len(area_in_model)/22*100:.1f}%):")
for f in area_in_model:
    print(f"  - {f}")
print()

# Define different feature combinations to test
test_scenarios = {
    'Current (5 area features)': all_22_features,

    'Reduced Option 1: Keep only area_x_bathrooms + area_per_bedroom':
        [f for f in all_22_features if f not in ['area', 'log_area', 'area_x_bedrooms']],

    'Reduced Option 2: Keep only log_area + area_x_bathrooms':
        [f for f in all_22_features if f not in ['area', 'area_x_bedrooms', 'area_per_bedroom']],

    'Reduced Option 3: Keep only area_x_bathrooms':
        [f for f in all_22_features if f not in ['area', 'log_area', 'area_x_bedrooms', 'area_per_bedroom']],

    'Minimal: Keep only area (no derived features)':
        [f for f in all_22_features if f not in ['log_area', 'area_x_bedrooms', 'area_x_bathrooms', 'area_per_bedroom']],
}

# We need to create ALL features from train_simple_model.py to have them available
# This is a simplified version - in production you'd import the full pipeline

# For now, let's just test with the features we can easily create
# and subset to those actually in the dataset

print("\n" + "=" * 80)
print("TESTING DIFFERENT FEATURE COMBINATIONS")
print("=" * 80 + "\n")

results = []

for scenario_name, features in test_scenarios.items():
    print(f"Testing: {scenario_name}")
    print(f"  Features ({len(features)}): {features[:3]}... (showing first 3)")

    # Check which features are available in the dataframe
    available_features = [f for f in features if f in df.columns]

    if len(available_features) < len(features) * 0.5:  # If less than 50% available
        print(f"  ⚠️  Skipping - only {len(available_features)}/{len(features)} features available")
        print()
        continue

    try:
        # Prepare data
        X = df[available_features].fillna(df[available_features].median())
        y = df['price']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Log transform
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train simple XGBoost (fast evaluation)
        model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=2.0,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )

        model.fit(X_train_scaled, y_train_log,
                 eval_set=[(X_test_scaled, y_test_log)],
                 verbose=False)

        # Evaluate
        y_pred_log = model.predict(X_test_scaled)
        y_pred = np.expm1(y_pred_log)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"  ✅ RMSE: ${rmse:,.0f}, R²: {r2:.4f}")

        results.append({
            'Scenario': scenario_name,
            'Features': len(available_features),
            'Area Features': len([f for f in available_features if f in area_related]),
            'RMSE': rmse,
            'R²': r2
        })

    except Exception as e:
        print(f"  ❌ Error: {e}")

    print()

# Summary
if results:
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80 + "\n")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')

    print(results_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best = results_df.iloc[0]
    baseline = results_df[results_df['Scenario'].str.contains('Current')].iloc[0] if any(results_df['Scenario'].str.contains('Current')) else results_df.iloc[0]

    improvement = ((baseline['RMSE'] - best['RMSE']) / baseline['RMSE']) * 100

    print(f"\nBest scenario: {best['Scenario']}")
    print(f"  Features: {best['Features']} (vs {baseline['Features']} in current)")
    print(f"  Area features: {best['Area Features']} (vs {baseline['Area Features']} in current)")
    print(f"  RMSE: ${best['RMSE']:,.0f} (vs ${baseline['RMSE']:,.0f} in current)")
    print(f"  Improvement: {improvement:.2f}%")

    if improvement > 1:
        print(f"\n✅ Reducing area features improves RMSE by {improvement:.2f}%!")
    elif improvement < -1:
        print(f"\n⚠️  Current configuration is {-improvement:.2f}% better - keep it")
    else:
        print(f"\n→ Negligible difference ({improvement:.2f}%) - prefer simpler model")
