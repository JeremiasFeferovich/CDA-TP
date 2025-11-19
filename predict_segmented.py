#!/usr/bin/env python3
"""
Segmented Model Prediction Script

Loads segmented models and makes predictions for new properties.
Automatically routes properties to the appropriate price segment model.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

class SegmentedPricePredictor:
    """
    Predictor that uses multiple models segmented by price range
    """

    def __init__(self, models_dir='models/segmented'):
        """
        Load all segment models and metadata

        Parameters:
        -----------
        models_dir : str
            Directory containing segmented model files
        """
        self.models_dir = Path(models_dir)

        # Load metadata
        with open(self.models_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Load segment models
        self.segments = {}
        for seg_config in self.metadata['config']['segments']:
            seg_name = seg_config['name']
            model_path = self.models_dir / f'{seg_name}_ensemble.joblib'

            model_data = joblib.load(model_path)

            self.segments[seg_name] = {
                'config': seg_config,
                'model': model_data,
                'price_min': seg_config['price_min'],
                'price_max': seg_config['price_max']
            }

        print(f"✅ Loaded {len(self.segments)} segment models")
        for seg_name in self.segments.keys():
            seg = self.segments[seg_name]
            print(f"   - {seg_name}: ${seg['price_min']:,} - ${seg['price_max']:,}")

    def _determine_segment(self, features):
        """
        Determine which price segment a property belongs to

        For now, uses a simple heuristic based on features.
        In production, could use a classifier or predicted price estimate.

        Parameters:
        -----------
        features : dict or pd.Series
            Property features

        Returns:
        --------
        str : segment name
        """
        # Simple heuristic: use area and luxury_score to estimate segment
        area = features.get('area', 0)
        luxury_score = features.get('luxury_score', 0)
        bathrooms = features.get('bathrooms', 1)

        # Budget: smaller properties with fewer amenities
        if area < 80 and luxury_score < 2:
            return 'budget'

        # Premium: large properties or high luxury
        elif area > 150 or luxury_score >= 4 or bathrooms >= 3:
            return 'premium'

        # Mid-range: everything else
        else:
            return 'mid_range'

    def predict(self, property_features, segment=None):
        """
        Predict price for a single property

        Parameters:
        -----------
        property_features : dict
            Dictionary of feature values
        segment : str, optional
            Force prediction with specific segment (for testing)

        Returns:
        --------
        dict with:
            - price: predicted price
            - segment: segment used for prediction
            - confidence: estimated confidence (based on segment match)
        """
        # Determine segment
        if segment is None:
            segment = self._determine_segment(property_features)

        seg_info = self.segments[segment]
        model_data = seg_info['model']

        # Get features in correct order
        features_list = model_data['features']
        feature_values = [property_features.get(f, 0) for f in features_list]

        # Create DataFrame (required by sklearn)
        X = pd.DataFrame([feature_values], columns=features_list)

        # Scale
        X_scaled = model_data['scaler'].transform(X)

        # Predict with ensemble
        xgb_pred_log = model_data['xgb_model'].predict(X_scaled)
        lgbm_pred_log = model_data['lgbm_model'].predict(X_scaled)

        ensemble_pred_log = (
            model_data['weight_xgb'] * xgb_pred_log +
            model_data['weight_lgbm'] * lgbm_pred_log
        )

        # Transform back from log scale
        price = np.expm1(ensemble_pred_log)[0]

        return {
            'price': price,
            'segment': segment,
            'price_range': f"${seg_info['price_min']:,} - ${seg_info['price_max']:,}"
        }

    def predict_batch(self, properties_df):
        """
        Predict prices for multiple properties

        Parameters:
        -----------
        properties_df : pd.DataFrame
            DataFrame with property features

        Returns:
        --------
        pd.DataFrame with predictions
        """
        predictions = []

        for idx, row in properties_df.iterrows():
            result = self.predict(row.to_dict())
            predictions.append(result)

        return pd.DataFrame(predictions)


def main():
    """
    Demo usage of the segmented predictor
    """
    print("=" * 80)
    print("SEGMENTED PRICE PREDICTOR - DEMO")
    print("=" * 80)

    # Load predictor
    predictor = SegmentedPricePredictor()

    # Example properties
    print("\n### Example Predictions:")
    print("-" * 80)

    examples = [
        {
            'name': 'Budget apartment',
            'area': 45,
            'bedrooms': 1,
            'bathrooms': 1,
            'latitude': -34.59,
            'longitude': -58.42,
            'luxury_score': 0,
            'family_score': 1,
            'total_amenities': 2,
            'log_area': np.log1p(45),
            'area_x_bedrooms': 45,
            'area_x_bathrooms': 45,
            'area_per_bedroom': 45,
            'has_balcony': 1,
            'has_doorman': 0,
            'has_garage': 0,
            'has_grill': 0,
            'has_gym': 0,
            'has_pool': 0,
            'has_security': 0,
            'has_storage': 0,
            'has_sum': 0,
            'has_terrace': 0
        },
        {
            'name': 'Mid-range family apartment',
            'area': 85,
            'bedrooms': 2,
            'bathrooms': 2,
            'latitude': -34.58,
            'longitude': -58.42,
            'luxury_score': 2,
            'family_score': 3,
            'total_amenities': 5,
            'log_area': np.log1p(85),
            'area_x_bedrooms': 170,
            'area_x_bathrooms': 170,
            'area_per_bedroom': 42.5,
            'has_balcony': 1,
            'has_doorman': 1,
            'has_garage': 1,
            'has_grill': 1,
            'has_gym': 0,
            'has_pool': 0,
            'has_security': 1,
            'has_storage': 0,
            'has_sum': 0,
            'has_terrace': 0
        },
        {
            'name': 'Premium penthouse',
            'area': 180,
            'bedrooms': 3,
            'bathrooms': 3,
            'latitude': -34.59,
            'longitude': -58.40,
            'luxury_score': 5,
            'family_score': 4,
            'total_amenities': 8,
            'log_area': np.log1p(180),
            'area_x_bedrooms': 540,
            'area_x_bathrooms': 540,
            'area_per_bedroom': 60,
            'has_balcony': 1,
            'has_doorman': 1,
            'has_garage': 1,
            'has_grill': 1,
            'has_gym': 1,
            'has_pool': 1,
            'has_security': 1,
            'has_storage': 1,
            'has_sum': 1,
            'has_terrace': 1
        }
    ]

    for example in examples:
        name = example.pop('name')
        result = predictor.predict(example)

        print(f"\n{name}:")
        print(f"  Area: {example['area']}m², {int(example['bedrooms'])} bed, {int(example['bathrooms'])} bath")
        print(f"  Predicted Price: ${result['price']:,.0f}")
        print(f"  Segment: {result['segment']} ({result['price_range']})")

    # Show overall stats
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 80)

    overall_results = predictor.metadata['overall']
    print(f"\nOverall Performance:")
    print(f"  RMSE:        ${overall_results['rmse']:>10,.0f}")
    print(f"  R²:          {overall_results['r2']:>10.4f}")
    print(f"  MAE:         ${overall_results['mae']:>10,.0f}")
    print(f"\nImprovement over baseline:")
    print(f"  Baseline:    ${overall_results['baseline_rmse']:>10,.0f}")
    print(f"  Improvement: {overall_results['improvement_pct']:>10.1f}%")

    print("\nPer-Segment Performance:")
    print("-" * 80)
    print(f"{'Segment':<15} {'RMSE':>12} {'R²':>10} {'Test Size':>12}")
    print("-" * 80)

    for seg_name, seg_results in predictor.metadata['results'].items():
        print(f"{seg_name:<15} ${seg_results['test_rmse']:>10,.0f} "
              f"{seg_results['test_r2']:>10.4f} {seg_results['test_size']:>12,}")


if __name__ == '__main__':
    main()
