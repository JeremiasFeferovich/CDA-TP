#!/usr/bin/env python3
"""
Analyze feature importance from trained model
Select top features to reduce overfitting
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load ensemble model
print("Loading ensemble model...")
ensemble = joblib.load('models/ensemble_price_model.joblib')
xgb_model = ensemble['xgb_model']

# Get feature importance
feature_names = xgb_model.feature_names_in_
feature_importance = xgb_model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Calculate cumulative importance
importance_df['cumulative'] = importance_df['importance'].cumsum()
total_importance = importance_df['importance'].sum()
importance_df['cumulative_pct'] = (importance_df['cumulative'] / total_importance) * 100

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

print(f"\nTotal features: {len(feature_names)}")
print(f"\nTop 25 Most Important Features:")
print("-" * 80)
for idx, row in importance_df.head(25).iterrows():
    print(f"{row['feature']:40s} {row['importance']:8.5f} ({row['importance']/total_importance*100:5.2f}%) - Cumulative: {row['cumulative_pct']:5.1f}%")

# Find features that contribute to 95% of importance
features_95 = importance_df[importance_df['cumulative_pct'] <= 95.0]
print(f"\nFeatures contributing to 95% of importance: {len(features_95)}")
print(f"Could reduce from {len(feature_names)} to {len(features_95)} features ({len(features_95)/len(feature_names)*100:.1f}%)")

# Identify low-importance features
low_importance = importance_df[importance_df['importance'] < 0.001]
print(f"\nLow-importance features (< 0.001): {len(low_importance)}")
if len(low_importance) > 0:
    print("Consider removing:")
    for idx, row in low_importance.iterrows():
        print(f"  - {row['feature']}")

# Save importance plot
plt.figure(figsize=(12, 10))
top_features = importance_df.head(30)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 30 Feature Importances (XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('data/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Feature importance plot saved to data/feature_importance_analysis.png")

# Save feature list for top features
top_features_list = features_95['feature'].tolist()
with open('data/top_features_95pct.txt', 'w') as f:
    f.write("# Features contributing to 95% of model importance\n")
    f.write(f"# Total: {len(top_features_list)} features\n\n")
    for feat in top_features_list:
        f.write(f"{feat}\n")
print(f"✅ Top features list saved to data/top_features_95pct.txt")
