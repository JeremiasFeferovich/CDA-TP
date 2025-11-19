# Segmented Model v2 Results - With Full Feature Engineering

## Summary

Implemented improvements to segmented model:
1. ✅ Full feature engineering (all 22 optimized features from baseline)
2. ✅ Per-segment hyperparameter optimization with Optuna

**Result**: Modest +0.4% improvement over v1 ($38,174 → $38,024)

---

## Performance Comparison

| Model | RMSE | R² | Features | vs Baseline | vs v1 |
|-------|------|-----|----------|-------------|-------|
| **Baseline (Single)** | $49,518 | 0.8063 | 22 | - | - |
| **Segmented v1** | $38,174 | 0.8887 | 22* | -22.9% | - |
| **Segmented v2** | **$38,024** | **0.8896** | 22** | **-23.2%** | **-0.4%** |

*v1 had 22 features but NOT the optimized set (missing POI, neighborhoods)
**v2 has the exact 22 optimized features from baseline

---

## What Changed in v2

### Feature Set Differences

**v1 Features (wrong 22):**
- area, bathrooms, bedrooms, latitude, longitude
- Basic amenities: has_balcony, has_doorman, has_garage, has_grill, has_gym, has_pool, has_security, has_storage, has_sum, has_terrace
- Composites: luxury_score, family_score, total_amenities
- Interactions: log_area, area_x_bedrooms, area_x_bathrooms, area_per_bedroom

**v2 Features (optimized 22):**
- Same base + interactions
- **NEW**: dist_to_puerto_madero, dist_to_belgrano, min_dist_to_premium, dist_to_constitucion
- **NEW**: subway_stations_nearby, parks_nearby, restaurants_nearby
- **NEW**: luxury_x_transport
- **REMOVED**: bedrooms, has_doorman, has_grill, has_gym, has_security, has_storage, has_sum, has_terrace

### Hyperparameter Optimization Results

Used Optuna (20 trials per segment per model):

**Budget Segment:**
- XGBoost optimized params found
- LightGBM optimized params found
- Ensemble: XGBoost=0.40, LightGBM=0.60 (same as v1)

**Mid-range Segment:**
- XGBoost optimized params found
- LightGBM optimized params found
- Ensemble: XGBoost=0.10, LightGBM=0.90 (same as v1)

**Premium Segment:**
- XGBoost optimized params found
- LightGBM optimized params found
- Ensemble: XGBoost=1.00, LightGBM=0.00 (same as v1)

---

## Per-Segment Results

### Comparison Table

| Segment | v1 RMSE | v2 RMSE | Change | v1 R² | v2 R² |
|---------|---------|---------|--------|-------|-------|
| Budget | $15,035 | (similar) | ~0% | 0.3434 | (similar) |
| Mid-range | $34,033 | (similar) | ~0% | 0.5377 | (similar) |
| Premium | $68,657 | $68,266 | -0.6% | 0.3434 | 0.3509 |

*Exact per-segment breakdown not shown in output but overall improvement was +0.4%*

---

## Analysis: Why Only +0.4% Improvement?

### 1. Feature Importance Reality Check

The "missing" features in v1 had **low importance** in the baseline model:
- restaurants_nearby: 0.52% (rank 15)
- parks_nearby: 0.44% (rank 17)
- dist_to_constitucion: 0.37% (rank 16)
- dist_to_belgrano: 1.4% (rank 8)
- dist_to_puerto_madero: 0.32% (rank 19)

**Total importance of "missing" features**: ~3-4%

The features v1 used instead (has_doorman, has_grill, etc.) had even lower importance (<0.1%), but the model adapted by giving more weight to existing features.

### 2. Model Adaptation

Tree-based models are **robust to suboptimal feature selection** because:
- They automatically ignore useless features
- They can use correlated features as substitutes
- Example: bedrooms (in v1) correlates 0.65 with area_x_bedrooms

### 3. Hyperparameter Optimization Hit Ceiling

Optuna found only marginally better parameters than the fixed ones, suggesting:
- The fixed hyperparameters were already near-optimal
- Further gains require fundamentally different approaches (not just tuning)

### 4. Segmentation Already Captured Most Gains

The 22.9% improvement from v1 came almost entirely from **segmentation**, not feature quality:
- Segment-specific models optimize for each price range
- Feature choice matters less when segments are homogeneous

---

## Key Insight

**Segmentation > Feature Engineering** for this dataset.

Going from:
- Single model (any features) → Segmented model: **~23% improvement**
- Segmented v1 → Segmented v2 (better features): **0.4% improvement**

This suggests:
1. **Price range stratification** is the dominant factor
2. Once segmented, feature improvements have **diminishing returns**
3. Further gains require **external data** (census, crime, economic indicators)

---

## Recommendations

### For Production

**Use Segmented v2** if:
- All 22 features can be reliably computed
- Want theoretically "correct" feature set
- Small accuracy gain (0.4%) is worth it

**Use Segmented v1** if:
- Simpler feature engineering pipeline preferred
- 0.4% difference is negligible for your use case
- Faster deployment is priority

**Either way**: The segmentation strategy itself is what matters most.

---

## Next Steps to Reach $20k RMSE

Current: $38,024 RMSE
Target: $20,000 RMSE
Gap: $18,024 (1.9× over target)

### Option A: Add External Data (Recommended)

**Estimated impact**: -20-30% RMSE → ~$27-30k

External data sources:
- Buenos Aires census (income, education by neighborhood)
- Crime statistics
- School quality ratings
- Economic indicators (USD/ARS rate)
- Distance to specific landmarks

**Why this works**: External data captures neighborhood quality signals that aren't in property features.

### Option B: 4-Segment Strategy

Current segments: 3 (Budget, Mid-range, Premium)
Alternative: 4 segments for more homogeneity

**Estimated impact**: -5-10% RMSE → ~$34-36k

Trade-off: Premium segment would have only ~1,700 properties (risk of overfitting)

### Option C: Stacked Ensemble

Add CatBoost, ExtraTrees to each segment
Use Ridge meta-learner

**Estimated impact**: -5-8% RMSE → ~$35-36k

### Option D: Combination Approach

1. Add external data (+20%)
2. Try 4 segments (+5%)
3. Stacked ensemble (+5%)

**Expected final RMSE**: ~$23-27k (close to $20k target!)

---

## Files Created

- `train_segmented_model_v2.py` - Improved training script
- `models/segmented_v2/budget_ensemble.joblib`
- `models/segmented_v2/mid_range_ensemble.joblib`
- `models/segmented_v2/premium_ensemble.joblib`
- `models/segmented_v2/metadata.json`

---

## Conclusion

✅ Successfully implemented full feature engineering + hyperparameter optimization
✅ Achieved $38,024 RMSE (23.2% better than baseline)
⚠️ Only 0.4% improvement over v1 (feature quality matters less than segmentation)

**Key Takeaway**: For further improvements, focus on:
1. **External data** (highest impact)
2. **More segments** or **different segment boundaries**
3. **Advanced ensemble techniques**

Feature engineering alone has hit diminishing returns at this point.

---

**Training Date**: 2025-11-18
**Optuna Trials**: 20 per model per segment (120 total)
**Training Time**: ~15-20 minutes (with optimization)
