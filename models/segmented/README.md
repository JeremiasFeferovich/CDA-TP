# Segmented Price Prediction Models

This directory contains three separate ensemble models trained for different price ranges of Buenos Aires properties.

## Model Performance

**Overall RMSE**: $38,174 (22.9% improvement over single-model baseline)

| Segment | Price Range | RMSE | R² | Properties |
|---------|-------------|------|-----|------------|
| Budget | $30k-$120k | $15,035 | 0.3434 | 1,389 |
| Mid-range | $120k-$300k | $34,033 | 0.5377 | 2,765 |
| Premium | $300k-$600k | $68,657 | 0.3434 | 779 |

## Files

### Model Files
- `budget_ensemble.joblib` - Budget segment model
- `mid_range_ensemble.joblib` - Mid-range segment model
- `premium_ensemble.joblib` - Premium segment model
- `metadata.json` - Configuration and performance metrics

### Model Contents

Each `.joblib` file contains:
```python
{
    'xgb_model': XGBRegressor,      # XGBoost model
    'lgbm_model': LGBMRegressor,    # LightGBM model
    'scaler': StandardScaler,        # Feature scaler
    'weight_xgb': float,             # XGBoost weight in ensemble
    'weight_lgbm': float,            # LightGBM weight in ensemble
    'features': list                 # Feature names (22 features)
}
```

## Usage

### Python API

```python
from predict_segmented import SegmentedPricePredictor

# Load all models
predictor = SegmentedPricePredictor('models/segmented')

# Predict for a single property
property_features = {
    'area': 85,
    'bathrooms': 2,
    'bedrooms': 2,
    'latitude': -34.58,
    'longitude': -58.42,
    # ... other features (22 total)
}

result = predictor.predict(property_features)
print(f"Price: ${result['price']:,.0f}")
print(f"Segment: {result['segment']}")
```

### Direct Model Loading

```python
import joblib

# Load specific segment
model = joblib.load('models/segmented/mid_range_ensemble.joblib')

# Predict
X_scaled = model['scaler'].transform(features)
xgb_pred = model['xgb_model'].predict(X_scaled)
lgbm_pred = model['lgbm_model'].predict(X_scaled)
price_log = (model['weight_xgb'] * xgb_pred +
             model['weight_lgbm'] * lgbm_pred)
price = np.expm1(price_log)
```

## Segment Selection Logic

Properties are routed to segments based on:

**Budget** ($30k-$120k):
- Small properties (<80m²)
- Low luxury score (<2 amenities)

**Premium** ($300k-$600k):
- Large properties (>150m²)
- High luxury score (≥4 amenities)
- 3+ bathrooms

**Mid-range** ($120k-$300k):
- Everything else

## Training Details

### Hyperparameters

All segments use:
- `n_estimators=1000` (with early stopping)
- `subsample=0.75`
- `colsample_bytree=0.75`

Segment-specific:

**Budget & Mid-range**:
- `max_depth=5`
- `learning_rate=0.03`
- `reg_alpha=2.0`, `reg_lambda=3.0`

**Premium**:
- `max_depth=6` (deeper trees)
- `learning_rate=0.02` (slower learning)
- `reg_alpha=3.0`, `reg_lambda=4.0` (stronger regularization)

### Ensemble Weights

Optimized on validation set:

- **Budget**: 40% XGBoost, 60% LightGBM
- **Mid-range**: 10% XGBoost, 90% LightGBM
- **Premium**: 100% XGBoost, 0% LightGBM

## Features (22 total)

1. area
2. bathrooms
3. bedrooms
4. latitude
5. longitude
6. luxury_score
7. family_score
8. total_amenities
9. log_area
10. area_x_bedrooms
11. area_x_bathrooms
12. area_per_bedroom
13. has_balcony
14. has_doorman
15. has_garage
16. has_grill
17. has_gym
18. has_pool
19. has_security
20. has_storage
21. has_sum
22. has_terrace

## Retraining

To retrain models:

```bash
python train_segmented_model.py
```

Requires:
- `segment_config.json` - Configuration file
- Data file at specified path
- ~8 minutes training time
- ~16GB RAM

## Monitoring

Track these metrics in production:

1. **Per-segment RMSE** - Monitor each segment separately
2. **Segment distribution** - Alert if % per segment shifts >10%
3. **Overall RMSE** - Should stay <$40k
4. **Prediction failures** - Properties outside training ranges

Alert if:
- Any segment RMSE increases >15%
- Overall RMSE exceeds $45k
- Segment distribution changes significantly

## Version History

- **v1.0** (2025-11-18): Initial release
  - 3 segments ($30k-$120k, $120k-$300k, $300k-$600k)
  - Overall RMSE: $38,174
  - 22 features
  - XGBoost + LightGBM ensemble per segment

## See Also

- `../SEGMENTED_RESULTS.md` - Complete analysis and results
- `../segment_analyzer.py` - Segmentation analysis tool
- `../segment_config.json` - Configuration file
- `../predict_segmented.py` - Inference script
