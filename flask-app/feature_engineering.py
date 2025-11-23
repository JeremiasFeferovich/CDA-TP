import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import os

# Load reference data for neighborhood statistics (lazy loading)
_ref_data = None

# Landmark coordinates (Buenos Aires, Argentina)
LANDMARKS = {
    'obelisco': (-34.603722, -58.381592),
    'plaza_de_mayo': (-34.608295, -58.370373),
    'microcentro': (-34.603722, -58.381592),  # Approximate center
    'calle_florida': (-34.603722, -58.375000),
    'av_santa_fe_alto_palermo': (-34.589722, -58.410833),
    'av_cabildo_belgrano': (-34.563056, -58.458333),
    'retiro_station': (-34.592778, -58.375556),
    'constitucion_station': (-34.625833, -58.390278),
    'teatro_colon': (-34.601389, -58.383611),
    'la_boca_caminito': (-34.635278, -58.363889),
    'uba_law': (-34.600556, -58.383056),
    'uca': (-34.589722, -58.410833),
    'bosques_de_palermo': (-34.587222, -58.410833),
    'reserva_ecologica': (-34.610833, -58.360833),
    'puerto_madero_north': (-34.610833, -58.360833),
}

# Commercial areas (approximate coordinates)
COMMERCIAL_AREAS = [
    (-34.603722, -58.381592),  # Obelisco area
    (-34.589722, -58.410833),  # Alto Palermo
    (-34.563056, -58.458333),  # Belgrano
]

# Green spaces (approximate coordinates)
GREEN_SPACES = [
    (-34.587222, -58.410833),  # Bosques de Palermo
    (-34.610833, -58.360833),  # Reserva Ecológica
]

# Cultural landmarks
CULTURAL_LANDMARKS = [
    (-34.601389, -58.383611),  # Teatro Colón
    (-34.635278, -58.363889),  # La Boca Caminito
]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula (in km)."""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    
    return c * r


def load_reference_data():
    """Load reference data for computing neighborhood statistics."""
    global _ref_data
    if _ref_data is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'data', 'properati_with_external_features.csv')
        _ref_data = pd.read_csv(csv_path)
    return _ref_data


def compute_neighborhood_stats(lat, lon, ref_data, radius_km=2.0):
    """Compute neighborhood statistics from nearby properties."""
    # Calculate distances
    distances = ref_data.apply(
        lambda row: haversine_distance(lat, lon, row['latitude'], row['longitude']),
        axis=1
    )
    
    # Filter properties within radius
    nearby = ref_data[distances <= radius_km]
    
    if len(nearby) == 0:
        # Use global medians if no nearby properties
        return {
            'area_mean': ref_data['area'].median(),
            'area_std': ref_data['area'].std(),
            'density': len(ref_data) / 1000.0,  # Approximate density
        }
    
    return {
        'area_mean': nearby['area'].mean(),
        'area_std': nearby['area'].std() if len(nearby) > 1 else 0,
        'density': len(nearby) / (np.pi * radius_km**2),  # Properties per km²
    }


def compute_features(area, bedrooms, bathrooms, latitude, longitude, 
                    property_type='departamento', balcony_count=0):
    """
    Compute all 40 features from minimal inputs.
    
    Parameters:
    -----------
    area : float
        Property area in square meters
    bedrooms : int
        Number of bedrooms
    bathrooms : float
        Number of bathrooms
    latitude : float
        Latitude coordinate
    longitude : float
        Longitude coordinate
    property_type : str
        'departamento' or 'casa'
    balcony_count : int
        Number of balconies
    
    Returns:
    --------
    dict : Dictionary with all 40 features in the order expected by the model
    """
    # Load reference data
    ref_data = load_reference_data()
    
    # Basic features
    features = {
        'area': float(area),
        'balcony_count': int(balcony_count),
        'bathrooms': float(bathrooms),
        'bedrooms': int(bedrooms),
        'latitude': float(latitude),
        'longitude': float(longitude),
        'is_departamento': 1 if property_type.lower() == 'departamento' else 0,
        'is_casa': 1 if property_type.lower() == 'casa' else 0,
    }
    
    # Distance features
    features['dist_to_obelisco'] = haversine_distance(
        latitude, longitude, *LANDMARKS['obelisco']
    )
    features['dist_to_plaza_de_mayo'] = haversine_distance(
        latitude, longitude, *LANDMARKS['plaza_de_mayo']
    )
    features['dist_to_microcentro'] = haversine_distance(
        latitude, longitude, *LANDMARKS['microcentro']
    )
    features['dist_to_calle_florida'] = haversine_distance(
        latitude, longitude, *LANDMARKS['calle_florida']
    )
    features['dist_to_av_santa_fe_alto_palermo'] = haversine_distance(
        latitude, longitude, *LANDMARKS['av_santa_fe_alto_palermo']
    )
    features['dist_to_av_cabildo_belgrano'] = haversine_distance(
        latitude, longitude, *LANDMARKS['av_cabildo_belgrano']
    )
    features['dist_to_retiro_station'] = haversine_distance(
        latitude, longitude, *LANDMARKS['retiro_station']
    )
    features['dist_to_constitucion_station'] = haversine_distance(
        latitude, longitude, *LANDMARKS['constitucion_station']
    )
    features['dist_to_teatro_colon'] = haversine_distance(
        latitude, longitude, *LANDMARKS['teatro_colon']
    )
    features['dist_to_la_boca_caminito'] = haversine_distance(
        latitude, longitude, *LANDMARKS['la_boca_caminito']
    )
    features['dist_to_uba_law'] = haversine_distance(
        latitude, longitude, *LANDMARKS['uba_law']
    )
    features['dist_to_uca'] = haversine_distance(
        latitude, longitude, *LANDMARKS['uca']
    )
    features['dist_to_bosques_de_palermo'] = haversine_distance(
        latitude, longitude, *LANDMARKS['bosques_de_palermo']
    )
    features['dist_to_reserva_ecologica'] = haversine_distance(
        latitude, longitude, *LANDMARKS['reserva_ecologica']
    )
    features['dist_to_puerto_madero_north'] = haversine_distance(
        latitude, longitude, *LANDMARKS['puerto_madero_north']
    )
    
    # Minimum distances
    commercial_dists = [
        haversine_distance(latitude, longitude, lat, lon)
        for lat, lon in COMMERCIAL_AREAS
    ]
    features['min_dist_to_commercial'] = min(commercial_dists) if commercial_dists else 10.0
    
    landmark_dists = [
        features['dist_to_obelisco'],
        features['dist_to_plaza_de_mayo'],
        features['dist_to_teatro_colon'],
    ]
    features['min_dist_to_landmark'] = min(landmark_dists)
    
    green_dists = [
        haversine_distance(latitude, longitude, lat, lon)
        for lat, lon in GREEN_SPACES
    ]
    features['min_dist_to_green'] = min(green_dists) if green_dists else 10.0
    
    cultural_dists = [
        haversine_distance(latitude, longitude, lat, lon)
        for lat, lon in CULTURAL_LANDMARKS
    ]
    features['min_dist_to_culture'] = min(cultural_dists) if cultural_dists else 10.0
    
    # Accessibility scores (normalized inverse distances)
    features['city_centrality_score'] = 1.0 / (1.0 + features['dist_to_microcentro'])
    features['commercial_accessibility'] = 1.0 / (1.0 + features['min_dist_to_commercial'])
    features['transport_hub_score'] = 1.0 / (
        1.0 + min(features['dist_to_retiro_station'], features['dist_to_constitucion_station'])
    )
    features['accessibility_score'] = (
        features['city_centrality_score'] + 
        features['commercial_accessibility'] + 
        features['transport_hub_score']
    ) / 3.0
    
    # Waterfront and tourist proximity (simplified)
    features['is_waterfront_nearby'] = 1 if features['dist_to_puerto_madero_north'] < 3.0 else 0
    features['tourist_area_proximity'] = 1.0 / (1.0 + features['dist_to_la_boca_caminito'])
    
    # Neighborhood statistics
    neighborhood_stats = compute_neighborhood_stats(latitude, longitude, ref_data)
    features['neighborhood_area_mean'] = neighborhood_stats['area_mean']
    features['neighborhood_area_std'] = neighborhood_stats['area_std']
    
    # Derived features
    features['area_vs_neighborhood'] = (
        features['area'] / features['neighborhood_area_mean']
        if features['neighborhood_area_mean'] > 0 else 1.0
    )
    features['local_property_density'] = neighborhood_stats['density']
    
    # Interaction features (simplified - using defaults for tier)
    # These would ideally use neighborhood_tier, but we'll use approximations
    tier_premium = 0  # Default to non-premium
    features['premium_x_area'] = tier_premium * features['area']
    features['premium_x_accessibility'] = tier_premium * features['accessibility_score']
    features['centrality_x_tier_premium'] = features['city_centrality_score'] * tier_premium
    
    return features

