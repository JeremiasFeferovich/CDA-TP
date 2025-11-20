#!/usr/bin/env python3
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
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load segment configuration
config_path = 'segment_config.json'
with open(config_path, 'r') as f:
    CONFIG = json.load(f)

SEGMENTS = CONFIG['segments']
OUTPUT_DIR = 'models/segmented_v3'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD ENHANCED DATA
# ============================================================================

print("\n### Loading enhanced dataset...")
df = pd.read_csv('/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/data/properati_with_external_features.csv')
print(f"Loaded {len(df):,} properties with {len(df.columns)} features")

# Apply same filters as before
df = df[df['currency'] == 'USD'].copy()
df = df[(df['price'] >= 30_000) & (df['price'] <= 600_000)].copy()

# Filter by price per sqm
price_per_sqm = df['price'] / df['area']
df = df[(price_per_sqm >= 800) & (price_per_sqm <= 5000)].copy()

print(f"After filtering: {len(df):,} properties")

# ============================================================================
# FEATURE SELECTION
# ============================================================================

print("\n### Selecting features...")

# Exclude non-feature columns
exclude_cols = ['price', 'currency', 'location', 'neighborhood_tier']

# Select all numeric features
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]

print(f"Selected {len(feature_cols)} numeric features")
print(f"Sample features: {feature_cols[:10]}")

# Prepare features and target
X = df[feature_cols].copy()
y = df['price'].copy()

# Handle any missing values
X = X.fillna(X.median())

# Log transform target
y_log = np.log1p(y)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target range: ${y.min():,.0f} - ${y.max():,.0f}")

# ============================================================================
# SEGMENT ASSIGNMENT
# ============================================================================

print("\n### Assigning properties to segments...")

def assign_segment(row):
    """Assign property to a segment based on price"""
    price = row['price']
    for seg in SEGMENTS:
        if seg['price_min'] <= price < seg['price_max']:
            return seg['name']
    # Edge case: exactly at max price
    if price == SEGMENTS[-1]['price_max']:
        return SEGMENTS[-1]['name']
    return None

df['segment'] = df.apply(assign_segment, axis=1)

# Display segment distribution
print("\nSegment distribution:")
for seg in SEGMENTS:
    count = (df['segment'] == seg['name']).sum()
    pct = (count / len(df)) * 100
    price_range = f"${seg['price_min']:,}-${seg['price_max']:,}"
    print(f"  {seg['name']:12s}: {count:>6,} ({pct:>5.1f}%)  {price_range}")

# ============================================================================
# TRAIN SEGMENT MODELS
# ============================================================================

results = {}
models = {}

for seg in SEGMENTS:
    seg_name = seg['name']
    print(f"\n{'=' * 80}")
    print(f"TRAINING: {seg_name.upper()} SEGMENT (${seg['price_min']:,}-${seg['price_max']:,})")
    print(f"{'=' * 80}")

    # Filter data for this segment
    seg_mask = df['segment'] == seg_name
    X_seg = X[seg_mask].copy()
    y_seg = y_log[seg_mask].copy()

    print(f"\nSegment size: {len(X_seg):,} properties")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seg, y_seg, test_size=0.2, random_state=42
    )

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ========================================================================
    # TRAIN XGBOOST
    # ========================================================================

    print(f"\n### Training XGBoost...")
    xgb_params = {
        'max_depth': 5,
        'learning_rate': 0.03,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 2.0,
        'reg_lambda': 3.0,
        'min_child_weight': 3,
        'n_estimators': 1000,
        'random_state': 42,
        'n_jobs': -1
    }

    # Train on full training set
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    xgb_pred_log = xgb_model.predict(X_test_scaled)
    xgb_pred = np.expm1(xgb_pred_log)
    y_test_actual = np.expm1(y_test)

    xgb_rmse = np.sqrt(mean_squared_error(y_test_actual, xgb_pred))
    xgb_r2 = r2_score(y_test_actual, xgb_pred)
    print(f"  XGBoost RMSE: ${xgb_rmse:,.0f}, R²: {xgb_r2:.4f}")

    # ========================================================================
    # TRAIN LIGHTGBM
    # ========================================================================

    print(f"\n### Training LightGBM...")

    # Fixed hyperparameters
    lgbm_params = {
        'max_depth': 5,
        'learning_rate': 0.03,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 2.0,
        'reg_lambda': 3.0,
        'min_child_weight': 3,
        'num_leaves': 31,
        'n_estimators': 1000,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    lgbm_model = LGBMRegressor(**lgbm_params)
    lgbm_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        callbacks=[lgbm.early_stopping(50, verbose=False), lgbm.log_evaluation(0)]
    )

    lgbm_pred_log = lgbm_model.predict(X_test_scaled)
    lgbm_pred = np.expm1(lgbm_pred_log)

    lgbm_rmse = np.sqrt(mean_squared_error(y_test_actual, lgbm_pred))
    lgbm_r2 = r2_score(y_test_actual, lgbm_pred)
    print(f"  LightGBM RMSE: ${lgbm_rmse:,.0f}, R²: {lgbm_r2:.4f}")

    # ========================================================================
    # OPTIMIZE ENSEMBLE WEIGHTS
    # ========================================================================

    print(f"\n### Optimizing ensemble weights...")

    best_weight = 0.5
    best_ensemble_rmse = float('inf')

    for w in np.arange(0.0, 1.05, 0.05):
        ensemble_pred_log = w * xgb_pred_log + (1 - w) * lgbm_pred_log
        ensemble_pred = np.expm1(ensemble_pred_log)
        rmse = np.sqrt(mean_squared_error(y_test_actual, ensemble_pred))

        if rmse < best_ensemble_rmse:
            best_ensemble_rmse = rmse
            best_weight = w

    print(f"  Best ensemble: XGBoost={best_weight:.2f}, LightGBM={1-best_weight:.2f}")
    print(f"  Ensemble RMSE: ${best_ensemble_rmse:,.0f}")

    # Final ensemble prediction
    final_pred_log = best_weight * xgb_pred_log + (1 - best_weight) * lgbm_pred_log
    final_pred = np.expm1(final_pred_log)

    final_rmse = np.sqrt(mean_squared_error(y_test_actual, final_pred))
    final_r2 = r2_score(y_test_actual, final_pred)
    final_mae = mean_absolute_error(y_test_actual, final_pred)

    # Train metrics
    train_pred_log = best_weight * xgb_model.predict(X_train_scaled) + (1 - best_weight) * lgbm_model.predict(X_train_scaled)
    train_pred = np.expm1(train_pred_log)
    y_train_actual = np.expm1(y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))

    overfitting = ((final_rmse - train_rmse) / train_rmse) * 100

    print(f"\n📊 FINAL RESULTS:")
    print(f"  Test RMSE: ${final_rmse:,.0f}")
    print(f"  Test R²: {final_r2:.4f}")
    print(f"  Test MAE: ${final_mae:,.0f}")
    print(f"  Train RMSE: ${train_rmse:,.0f}")
    print(f"  Overfitting: {overfitting:.1f}%")

    # ========================================================================
    # SAVE MODEL
    # ========================================================================

    model_data = {
        'xgb_model': xgb_model,
        'lgbm_model': lgbm_model,
        'scaler': scaler,
        'weight_xgb': best_weight,
        'weight_lgbm': 1 - best_weight,
        'features': feature_cols,
        'xgb_params': xgb_params,
        'lgbm_params': lgbm_params
    }

    model_path = f"{OUTPUT_DIR}/{seg_name}_ensemble.joblib"
    joblib.dump(model_data, model_path)
    print(f"\n✅ Saved model to {model_path}")

    # Store results
    results[seg_name] = {
        'test_rmse': float(final_rmse),
        'test_r2': float(final_r2),
        'test_mae': float(final_mae),
        'train_rmse': float(train_rmse),
        'overfitting_pct': float(overfitting),
        'test_size': len(X_test),
        'xgb_weight': float(best_weight),
        'lgbm_weight': float(1 - best_weight)
    }

    models[seg_name] = model_data

# ============================================================================
# OVERALL EVALUATION
# ============================================================================

print(f"\n{'=' * 80}")
print("OVERALL PERFORMANCE")
print(f"{'=' * 80}")

# Calculate weighted average RMSE
total_test = sum(results[seg['name']]['test_size'] for seg in SEGMENTS)
weighted_rmse = sum(
    results[seg['name']]['test_rmse'] * results[seg['name']]['test_size']
    for seg in SEGMENTS
) / total_test

# Calculate overall R²
all_preds = []
all_actuals = []

for seg in SEGMENTS:
    seg_name = seg['name']
    seg_mask = df['segment'] == seg_name
    X_seg = X[seg_mask].copy()
    y_seg = y_log[seg_mask].copy()

    _, X_test_seg, _, y_test_seg = train_test_split(
        X_seg, y_seg, test_size=0.2, random_state=42
    )

    model_data = models[seg_name]
    X_test_scaled = model_data['scaler'].transform(X_test_seg)

    pred_log = (
        model_data['weight_xgb'] * model_data['xgb_model'].predict(X_test_scaled) +
        model_data['weight_lgbm'] * model_data['lgbm_model'].predict(X_test_scaled)
    )

    all_preds.extend(np.expm1(pred_log))
    all_actuals.extend(np.expm1(y_test_seg))

overall_r2 = r2_score(all_actuals, all_preds)
overall_mae = mean_absolute_error(all_actuals, all_preds)

print(f"\n📈 OVERALL METRICS:")
print(f"  Weighted RMSE: ${weighted_rmse:,.0f}")
print(f"  Overall R²: {overall_r2:.4f}")
print(f"  Overall MAE: ${overall_mae:,.0f}")

print(f"\n📊 PER-SEGMENT BREAKDOWN:")
for seg in SEGMENTS:
    seg_name = seg['name']
    r = results[seg_name]
    print(f"\n  {seg_name.upper()}:")
    print(f"    RMSE: ${r['test_rmse']:,.0f}")
    print(f"    R²: {r['test_r2']:.4f}")
    print(f"    MAE: ${r['test_mae']:,.0f}")
    print(f"    Ensemble: XGBoost={r['xgb_weight']:.2f}, LightGBM={r['lgbm_weight']:.2f}")

# ============================================================================
# SAVE METADATA
# ============================================================================

metadata = {
    'version': 'v3',
    'description': 'Segmented model with external features',
    'segments': SEGMENTS,
    'results': results,
    'overall_rmse': float(weighted_rmse),
    'overall_r2': float(overall_r2),
    'overall_mae': float(overall_mae),
    'num_features': len(feature_cols),
    'features': feature_cols
}

metadata_path = f"{OUTPUT_DIR}/metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Saved metadata to {metadata_path}")

print(f"\n{'=' * 80}")
print("TRAINING COMPLETE!")
print(f"{'=' * 80}")
print(f"\nModels saved to: {OUTPUT_DIR}/")
print(f"Total features: {len(feature_cols)}")
print(f"External features added: 42")
