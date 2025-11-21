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

OUTPUT_DIR = 'models/simple'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SIMPLE PROPERTY PRICE PREDICTION MODEL")
print("=" * 80)

# ============================================================================
# LOAD ENHANCED DATA
# ============================================================================

print("\n### Loading enhanced dataset...")
df = pd.read_csv('data/properati_with_external_features.csv')
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
# TRAIN MODEL
# ============================================================================

print(f"\n{'=' * 80}")
print("TRAINING MODEL")
print(f"{'=' * 80}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")

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

model_path = f"{OUTPUT_DIR}/ensemble_model.joblib"
joblib.dump(model_data, model_path)
print(f"\n✅ Saved model to {model_path}")

# ============================================================================
# SAVE METADATA
# ============================================================================

metadata = {
    'version': 'simple',
    'description': 'Simple ensemble model with external features',
    'results': {
        'test_rmse': float(final_rmse),
        'test_r2': float(final_r2),
        'test_mae': float(final_mae),
        'train_rmse': float(train_rmse),
        'overfitting_pct': float(overfitting),
        'test_size': len(X_test),
        'xgb_weight': float(best_weight),
        'lgbm_weight': float(1 - best_weight),
        'xgb_rmse': float(xgb_rmse),
        'xgb_r2': float(xgb_r2),
        'lgbm_rmse': float(lgbm_rmse),
        'lgbm_r2': float(lgbm_r2)
    },
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
print(f"\nModel saved to: {OUTPUT_DIR}/")
print(f"Total features: {len(feature_cols)}")
