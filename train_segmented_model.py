#!/usr/bin/env python3
"""
Segmented Property Price Prediction Model

Trains separate models for different price ranges to improve RMSE.
Uses same feature engineering as train_simple_model.py but segments by price.
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SEGMENTED PROPERTY PRICE PREDICTION MODEL")
print("=" * 80)

# ============================================================================
# 1. LOAD CONFIGURATION
# ============================================================================

print("\n### 1. Loading configuration...")
with open('segment_config.json', 'r') as f:
    config = json.load(f)

print(f"Strategy: {config['strategy_name']}")
print(f"Number of segments: {len(config['segments'])}")
for seg in config['segments']:
    print(f"  - {seg['name']}: ${seg['price_min']:,} - ${seg['price_max']:,}")

# ============================================================================
# 2. LOAD AND PREPARE DATA
# ============================================================================

print("\n### 2. Loading data...")
# Use same data source as train_simple_model.py
csv_path = '/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/data/properati_preprocessed_20251016_204913.csv'
df = pd.read_csv(csv_path)

print(f"Loaded data shape: {df.shape}")

# Apply same basic filters as train_simple_model.py
df = df[df['currency'] == 'USD'].copy()
print(f"After USD filter: {df.shape}")

# Apply price range filter (full range for all segments combined)
df = df[(df['price'] >= 30_000) & (df['price'] <= 600_000)].copy()
print(f"After price filter ($30k-$600k): {df.shape}")

# Filter by price per sqm
price_per_sqm = df['price'] / df['area']
df = df[(price_per_sqm >= 800) & (price_per_sqm <= 5000)].copy()
print(f"After price/sqm filter: {df.shape}")

# Remove missing coordinates
df = df.dropna(subset=['latitude', 'longitude'])
print(f"After removing missing coordinates: {df.shape}")

print(f"\nFinal dataset: {len(df):,} properties")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

# We'll reuse the same feature engineering steps from train_simple_model.py
# For now, create a minimal set to get started

print("\n### 3. Feature Engineering...")

# Import feature engineering if needed, or inline basic features
# For this implementation, we'll do inline to keep it self-contained

# Basic composite features
df['luxury_score'] = (
    df.get('has_gym', 0).astype(int) +
    df.get('has_pool', 0).astype(int) +
    df.get('has_doorman', 0).astype(int) +
    df.get('has_security', 0).astype(int) +
    df.get('has_garage', 0).astype(int)
)

df['family_score'] = (
    df.get('has_balcony', 0).astype(int) +
    df.get('has_terrace', 0).astype(int) +
    df.get('has_grill', 0).astype(int) +
    (df.get('bedrooms', 0) >= 2).astype(int) +
    (df.get('bathrooms', 0) >= 2).astype(int)
)

amenity_cols = [col for col in df.columns if col.startswith('has_')]
df['total_amenities'] = df[amenity_cols].astype(int).sum(axis=1)

# Log transforms
df['log_area'] = np.log1p(df['area'])

# Interactions
df['area_x_bedrooms'] = df['area'] * df['bedrooms']
df['area_x_bathrooms'] = df['area'] * df['bathrooms']
df['area_per_bedroom'] = df['area'] / df['bedrooms'].replace(0, 1)

# Use subset of features that we know exist
# (Full feature engineering would add subway, POI, neighborhoods etc)
basic_features = ['area', 'bathrooms', 'bedrooms', 'latitude', 'longitude',
                  'luxury_score', 'family_score', 'total_amenities',
                  'log_area', 'area_x_bedrooms', 'area_x_bathrooms', 'area_per_bedroom']

# Add has_ features
for col in amenity_cols:
    if col in df.columns:
        basic_features.append(col)

# Filter to existing features
basic_features = [f for f in basic_features if f in df.columns]

print(f"Using {len(basic_features)} features")

# ============================================================================
# 4. SEGMENT DATA
# ============================================================================

print("\n### 4. Segmenting data by price...")

segments_data = {}

for seg_config in config['segments']:
    seg_name = seg_config['name']
    price_min = seg_config['price_min']
    price_max = seg_config['price_max']

    mask = (df['price'] >= price_min) & (df['price'] < price_max)
    seg_df = df[mask].copy()

    segments_data[seg_name] = {
        'data': seg_df,
        'config': seg_config,
        'count': len(seg_df)
    }

    print(f"  {seg_name:15s}: {len(seg_df):>6,} properties "
          f"(${price_min:>7,} - ${price_max:>7,})")

# ============================================================================
# 5. TRAIN MODELS FOR EACH SEGMENT
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING SEGMENT MODELS")
print("=" * 80)

segment_results = {}

for seg_name, seg_info in segments_data.items():
    print(f"\n### Training {seg_name} segment...")
    print("-" * 80)

    seg_df = seg_info['data']
    seg_config = seg_info['config']

    # Prepare features and target
    X = seg_df[basic_features].copy()
    y = seg_df['price'].copy()

    # Fill NaN
    X = X.fillna(X.median())

    print(f"Segment size: {len(X):,} properties")

    # Train-validation-test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42
    )

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Log transform target
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    print("  Training XGBoost...")
    xgb_params = seg_config['hyperparameters']['xgb']
    xgb_model = XGBRegressor(
        n_estimators=xgb_params['n_estimators'],
        max_depth=xgb_params['max_depth'],
        learning_rate=xgb_params['learning_rate'],
        subsample=xgb_params['subsample'],
        colsample_bytree=xgb_params['colsample_bytree'],
        reg_alpha=xgb_params['reg_alpha'],
        reg_lambda=xgb_params['reg_lambda'],
        random_state=xgb_params['random_state'],
        n_jobs=-1,
        verbosity=0
    )

    xgb_model.fit(
        X_train_scaled, y_train_log,
        eval_set=[(X_val_scaled, y_val_log)],
        verbose=False
    )

    # Train LightGBM
    print("  Training LightGBM...")
    lgbm_params = seg_config['hyperparameters']['lgbm']
    lgbm_model = LGBMRegressor(
        n_estimators=lgbm_params['n_estimators'],
        max_depth=lgbm_params['max_depth'],
        learning_rate=lgbm_params['learning_rate'],
        subsample=lgbm_params['subsample'],
        colsample_bytree=lgbm_params['colsample_bytree'],
        reg_alpha=lgbm_params['reg_alpha'],
        reg_lambda=lgbm_params['reg_lambda'],
        min_child_samples=lgbm_params['min_child_samples'],
        num_leaves=lgbm_params['num_leaves'],
        random_state=lgbm_params['random_state'],
        n_jobs=-1,
        verbose=-1
    )

    lgbm_model.fit(
        X_train_scaled, y_train_log,
        eval_set=[(X_val_scaled, y_val_log)],
        callbacks=[])

    # Optimize ensemble weights on validation set
    print("  Optimizing ensemble weights...")
    xgb_val_pred_log = xgb_model.predict(X_val_scaled)
    lgbm_val_pred_log = lgbm_model.predict(X_val_scaled)

    best_rmse = float('inf')
    best_weights = (0.5, 0.5)

    for w_xgb in np.arange(0.0, 1.05, 0.05):
        w_lgbm = 1.0 - w_xgb
        ensemble_pred_log = w_xgb * xgb_val_pred_log + w_lgbm * lgbm_val_pred_log
        ensemble_pred = np.expm1(ensemble_pred_log)
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = (w_xgb, w_lgbm)

    weight_xgb, weight_lgbm = best_weights
    print(f"  Best weights: XGBoost={weight_xgb:.2f}, LightGBM={weight_lgbm:.2f}")

    # Evaluate on test set
    xgb_test_pred_log = xgb_model.predict(X_test_scaled)
    lgbm_test_pred_log = lgbm_model.predict(X_test_scaled)
    ensemble_test_pred_log = weight_xgb * xgb_test_pred_log + weight_lgbm * lgbm_test_pred_log
    ensemble_test_pred = np.expm1(ensemble_test_pred_log)

    # Evaluate on train set (for overfitting check)
    xgb_train_pred_log = xgb_model.predict(X_train_scaled)
    lgbm_train_pred_log = lgbm_model.predict(X_train_scaled)
    ensemble_train_pred_log = weight_xgb * xgb_train_pred_log + weight_lgbm * lgbm_train_pred_log
    ensemble_train_pred = np.expm1(ensemble_train_pred_log)

    # Metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
    test_r2 = r2_score(y_test, ensemble_test_pred)
    test_mae = mean_absolute_error(y_test, ensemble_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, ensemble_train_pred))
    train_r2 = r2_score(y_train, ensemble_train_pred)

    overfitting_pct = ((test_rmse - train_rmse) / train_rmse) * 100

    print(f"\n  Results:")
    print(f"    Test RMSE:  ${test_rmse:>10,.2f}")
    print(f"    Test R²:    {test_r2:>10.4f}")
    print(f"    Test MAE:   ${test_mae:>10,.2f}")
    print(f"    Train RMSE: ${train_rmse:>10,.2f}")
    print(f"    Overfitting: {overfitting_pct:>9.1f}%")

    # Store results
    segment_results[seg_name] = {
        'xgb_model': xgb_model,
        'lgbm_model': lgbm_model,
        'scaler': scaler,
        'weight_xgb': weight_xgb,
        'weight_lgbm': weight_lgbm,
        'features': basic_features,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'overfitting_pct': overfitting_pct,
        'test_size': len(X_test),
        'y_test': y_test,
        'y_pred': ensemble_test_pred
    }

# ============================================================================
# 6. OVERALL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("OVERALL EVALUATION")
print("=" * 80)

# Combine all test predictions
all_y_test = []
all_y_pred = []
all_segments = []

for seg_name, results in segment_results.items():
    all_y_test.extend(results['y_test'].values)
    all_y_pred.extend(results['y_pred'])
    all_segments.extend([seg_name] * len(results['y_test']))

all_y_test = np.array(all_y_test)
all_y_pred = np.array(all_y_pred)

overall_rmse = np.sqrt(mean_squared_error(all_y_test, all_y_pred))
overall_r2 = r2_score(all_y_test, all_y_pred)
overall_mae = mean_absolute_error(all_y_test, all_y_pred)

print(f"\n📊 Overall Performance (Combined Test Sets):")
print(f"  RMSE: ${overall_rmse:>10,.2f}")
print(f"  R²:   {overall_r2:>10.4f}")
print(f"  MAE:  ${overall_mae:>10,.2f}")

# Compare to baseline
baseline_rmse = 49518  # From train_simple_model.py with 22 features
improvement = ((baseline_rmse - overall_rmse) / baseline_rmse) * 100

print(f"\n📈 Comparison to Baseline:")
print(f"  Baseline RMSE (single model): ${baseline_rmse:,}")
print(f"  Segmented RMSE:               ${overall_rmse:,.0f}")
print(f"  Improvement:                  {improvement:+.1f}%")

# Per-segment summary
print(f"\n📋 Per-Segment Performance:")
print("-" * 80)
print(f"{'Segment':<15} {'Test RMSE':>12} {'Test R²':>10} {'Properties':>12} {'Overfitting':>12}")
print("-" * 80)

for seg_name in config['segments']:
    seg_name = seg_name['name']
    results = segment_results[seg_name]
    print(f"{seg_name:<15} ${results['test_rmse']:>10,.0f} {results['test_r2']:>10.4f} "
          f"{results['test_size']:>12,} {results['overfitting_pct']:>11.1f}%")

# ============================================================================
# 7. SAVE MODELS
# ============================================================================

print("\n### 7. Saving models...")

# Create directory for segmented models
models_dir = Path('models/segmented')
models_dir.mkdir(parents=True, exist_ok=True)

# Save each segment model
for seg_name, results in segment_results.items():
    model_path = models_dir / f'{seg_name}_ensemble.joblib'

    model_data = {
        'xgb_model': results['xgb_model'],
        'lgbm_model': results['lgbm_model'],
        'scaler': results['scaler'],
        'weight_xgb': results['weight_xgb'],
        'weight_lgbm': results['weight_lgbm'],
        'features': results['features']
    }

    joblib.dump(model_data, model_path)
    print(f"  ✅ Saved {seg_name} model to {model_path}")

# Save metadata
metadata = {
    'config': config,
    'results': {
        seg_name: {
            'test_rmse': float(results['test_rmse']),
            'test_r2': float(results['test_r2']),
            'test_mae': float(results['test_mae']),
            'train_rmse': float(results['train_rmse']),
            'overfitting_pct': float(results['overfitting_pct']),
            'test_size': int(results['test_size'])
        }
        for seg_name, results in segment_results.items()
    },
    'overall': {
        'rmse': float(overall_rmse),
        'r2': float(overall_r2),
        'mae': float(overall_mae),
        'baseline_rmse': float(baseline_rmse),
        'improvement_pct': float(improvement)
    },
    'features': basic_features
}

metadata_path = models_dir / 'metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✅ Saved metadata to {metadata_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\n✅ Segmented model achieves ${overall_rmse:,.0f} RMSE")
print(f"✅ {improvement:+.1f}% improvement over baseline (${baseline_rmse:,})")
print(f"\nModels saved to: {models_dir}/")
