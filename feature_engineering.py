"""
Feature Engineering Module

Reusable feature engineering functions for property price prediction.
Used by both train_simple_model.py and train_segmented_model.py
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import os

# Premium neighborhoods in Buenos Aires
PREMIUM_NEIGHBORHOODS = {
    'Palermo': (-34.5889, -58.4197),
    'Puerto_Madero': (-34.6118, -58.3636),
    'Recoleta': (-34.5875, -58.3943),
    'Belgrano': (-34.5629, -58.4582),
}

LOW_VALUE_NEIGHBORHOODS = {
    'La_Boca': (-34.6345, -58.3631),
    'Constitucion': (-34.6277, -58.3817),
}

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two points on earth (in km)"""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km


def calculate_distance_to_neighborhood(lat, lon, neighborhood_coords):
    """Calculate distance to a specific neighborhood center"""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    return haversine(lon, lat, neighborhood_coords[1], neighborhood_coords[0])


def calculate_min_distance_to_premium_areas(lat, lon):
    """Calculate minimum distance to any premium neighborhood"""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    distances = [haversine(lon, lat, coords[1], coords[0])
                 for coords in PREMIUM_NEIGHBORHOODS.values()]
    return min(distances)


def engineer_all_features(df, verbose=True):
    """
    Apply all feature engineering steps to a dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with raw property data
    verbose : bool
        Print progress messages

    Returns:
    --------
    pd.DataFrame
        DataFrame with all engineered features
    list
        List of feature names to use for modeling
    """

    if verbose:
        print("\n### Feature Engineering...")

    # Create domain-specific composite features
    if verbose:
        print("  Creating composite features...")

    # Luxury score (amenities indicating premium property)
    df['luxury_score'] = (
        df.get('has_gym', 0).astype(int) +
        df.get('has_pool', 0).astype(int) +
        df.get('has_doorman', 0).astype(int) +
        df.get('has_security', 0).astype(int) +
        df.get('has_garage', 0).astype(int)
    )

    # Family score (features families care about)
    df['family_score'] = (
        df.get('has_balcony', 0).astype(int) +
        df.get('has_terrace', 0).astype(int) +
        df.get('has_grill', 0).astype(int) +
        (df.get('bedrooms', 0) >= 2).astype(int) +
        (df.get('bathrooms', 0) >= 2).astype(int)
    )

    # Total amenities
    amenity_cols = [col for col in df.columns if col.startswith('has_')]
    df['total_amenities'] = df[amenity_cols].astype(int).sum(axis=1)

    # Log transformations
    if verbose:
        print("  Creating log-transformed features...")
    df['log_area'] = np.log1p(df['area'])

    # Interaction features
    if verbose:
        print("  Creating interaction features...")
    df['area_x_bedrooms'] = df['area'] * df['bedrooms']
    df['area_x_bathrooms'] = df['area'] * df['bathrooms']
    df['area_per_bedroom'] = df['area'] / df['bedrooms'].replace(0, 1)

    # Location-based features (if coordinates exist)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        if verbose:
            print("  Creating location-based features...")

        # Premium neighborhood distances
        for name, coords in PREMIUM_NEIGHBORHOODS.items():
            df[f'dist_to_{name.lower()}'] = df.apply(
                lambda row: calculate_distance_to_neighborhood(
                    row['latitude'], row['longitude'], coords
                ), axis=1
            )

        # Min distance to any premium area
        df['min_dist_to_premium'] = df.apply(
            lambda row: calculate_min_distance_to_premium_areas(
                row['latitude'], row['longitude']
            ), axis=1
        )

        # Low-value neighborhood distances
        for name, coords in LOW_VALUE_NEIGHBORHOODS.items():
            df[f'dist_to_{name.lower()}'] = df.apply(
                lambda row: calculate_distance_to_neighborhood(
                    row['latitude'], row['longitude'], coords
                ), axis=1
            )

    # Luxury × transport interaction (if transport features exist)
    if 'luxury_score' in df.columns:
        # Create simple transport score if not exists
        if 'transport_score' not in df.columns:
            transport_components = []
            if 'subway_stations_nearby' in df.columns:
                transport_components.append(df['subway_stations_nearby'] * 10)
            if 'bus_stops_nearby' in df.columns:
                transport_components.append(df['bus_stops_nearby'])

            if transport_components:
                df['transport_score'] = sum(transport_components)
            else:
                df['transport_score'] = 0

        df['luxury_x_transport'] = df['luxury_score'] * df['transport_score']

    if verbose:
        print(f"  ✅ Feature engineering complete")

    return df


def get_feature_list(df):
    """
    Get list of features to use for modeling (matching train_simple_model.py)

    Returns list of 22 features used in the optimized model
    """

    # Top 22 features from feature importance analysis
    top_22_features = [
        'area_x_bathrooms',
        'log_area',
        'area',
        'has_pool',
        'latitude',
        'luxury_x_transport',
        'luxury_score',
        'dist_to_belgrano',
        'area_per_bedroom',
        'min_dist_to_premium',
        'family_score',
        'has_garage',
        'longitude',
        'total_amenities',
        'restaurants_nearby',
        'dist_to_constitucion',
        'parks_nearby',
        'subway_stations_nearby',
        'dist_to_puerto_madero',
        'has_balcony',
        'area_x_bedrooms',
        'bathrooms'
    ]

    # Filter to only features that exist in dataframe
    available_features = [f for f in top_22_features if f in df.columns]

    return available_features
