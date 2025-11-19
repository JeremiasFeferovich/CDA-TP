# -*- coding: utf-8 -*-
"""
Simple Property Price Prediction Model

A simplified version of the full training script for basic price prediction.
Includes subway stations and bus stops features for improved performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
from math import radians, cos, sin, asin, sqrt
import joblib
import os
import time

print("=" * 80)
print("SIMPLE PROPERTY PRICE PREDICTION MODEL")
print("=" * 80)

# ============================================================================
# SUBWAY STATIONS AND BUS STOPS DATA
# ============================================================================

# Premium neighborhoods in Buenos Aires (known high-value areas)
PREMIUM_NEIGHBORHOODS = {
    'Palermo': (-34.5889, -58.4197),      # Trendy area with parks, restaurants, nightlife
    'Puerto_Madero': (-34.6118, -58.3636), # Waterfront, modern, most expensive
    'Recoleta': (-34.5875, -58.3943),      # Historic, elegant, cemetery, museums
    'Belgrano': (-34.5629, -58.4582),      # Residential, family-friendly, upscale
}

# Low-value reference neighborhoods
LOW_VALUE_NEIGHBORHOODS = {
    'La_Boca': (-34.6345, -58.3631),       # Tourist area but lower property values
    'Constitucion': (-34.6277, -58.3817),  # Transportation hub, lower-end
}

# Buenos Aires subway stations coordinates
SUBWAY_STATIONS = [
    (-34.635750, -58.398928),  # CASEROS - Line H
    (-34.629376, -58.400970),  # INCLAN - MEZQUITA AL AHMAD - Line H
    (-34.623092, -58.402323),  # HUMBERTO 1° - Line H
    (-34.615242, -58.404732),  # VENEZUELA - Line H
    (-34.608935, -58.406036),  # ONCE - 30 DE DICIEMBRE - Line H
    (-34.604245, -58.380574),  # 9 DE JULIO - Line D
    (-34.599757, -58.397924),  # FACULTAD DE MEDICINA - Line D
    (-34.601587, -58.385142),  # TRIBUNALES - TEATRO COLÓN - Line D
    (-34.591628, -58.407161),  # AGÜERO - Line D
    (-34.585156, -58.415955),  # R.SCALABRINI ORTIZ - Line D
    (-34.581411, -58.421196),  # PLAZA ITALIA - Line D
    (-34.578422, -58.425711),  # PALERMO - Line D
    (-34.591194, -58.374018),  # RETIRO - Line C
    (-34.601770, -58.378156),  # LAVALLE - Line C
    (-34.604844, -58.379530),  # DIAGONAL NORTE - Line C
    (-34.608983, -58.380611),  # AV. DE MAYO - Line C
    (-34.612617, -58.380444),  # MORENO - Line C
    (-34.618126, -58.380174),  # INDEPENDENCIA - Line C
    (-34.627619, -58.381434),  # CONSTITUCION - Line C
    (-34.603297, -58.375072),  # FLORIDA - Line B
    (-34.603637, -58.380715),  # C. PELLEGRINI - Line B
    (-34.604094, -58.387296),  # URUGUAY - Line B
    (-34.604420, -58.392314),  # CALLAO - MAESTRO ALFREDO BRAVO - Line B
    (-34.604643, -58.399474),  # PASTEUR - AMIA - Line B
    (-34.604581, -58.405399),  # PUEYRREDON - Line B
    (-34.604080, -58.411763),  # CARLOS GARDEL - Line B
    (-34.603165, -58.420962),  # ALMAGRO - MEDRANO - Line B
    (-34.602162, -58.431274),  # ANGEL GALLARDO - Line B
    (-34.598967, -58.439771),  # MALABIA - OSVALDO PUGLIESE - Line B
    (-34.591718, -58.447573),  # DORREGO - Line B
    (-34.608559, -58.374268),  # PERU - Line A
    (-34.608882, -58.379085),  # PIEDRAS - Line A
    (-34.609100, -58.382232),  # LIMA - Line A
    (-34.609413, -58.386777),  # SAENZ PEÑA - Line A
    (-34.609226, -58.392669),  # CONGRESO - PDTE. DR. RAÚL R. ALFONSÍN - Line A
    (-34.609646, -58.398427),  # PASCO - Line A
    (-34.609834, -58.401208),  # ALBERTI - Line A
    (-34.609817, -58.406707),  # PLAZA DE MISERERE - Line A
    (-34.610782, -58.415186),  # LORIA - Line A
    (-34.611770, -58.421816),  # CASTRO BARROS - Line A
    (-34.615206, -58.429500),  # RIO DE JANEIRO - Line A
    (-34.618280, -58.436429),  # ACOYTE - Line A
    (-34.620405, -58.441178),  # PRIMERA JUNTA - Line A
    (-34.609242, -58.373684),  # BOLIVAR - Line E
    (-34.612849, -58.377581),  # BELGRANO - Line E
    (-34.617937, -58.381535),  # INDEPENDENCIA - Line E
    (-34.622339, -58.385149),  # SAN JOSE - Line E
    (-34.622720, -58.391512),  # ENTRE RIOS - RODOLFO WALSH - Line E
    (-34.623110, -58.397068),  # PICHINCHA - Line E
    (-34.623866, -58.402937),  # JUJUY - Line E
    (-34.624654, -58.409391),  # URQUIZA - Line E
    (-34.628018, -58.433816),  # JOSE MARIA MORENO - Line E
    (-34.631042, -58.442171),  # EMILIO MITRE - Line E
    (-34.588237, -58.411294),  # BULNES - Line D
    (-34.594426, -58.402395),  # PUEYRREDON - Line D
    (-34.599640, -58.393125),  # CALLAO - Line D
    (-34.595057, -58.377819),  # SAN MARTIN - Line C
    (-34.621917, -58.379921),  # SAN JUAN - Line C
    (-34.636389, -58.450278),  # MEDALLA MILAGROSA - Line E
    (-34.627015, -58.426789),  # AV. LA PLATA - Line E
    (-34.625366, -58.415533),  # BOEDO - Line E
    (-34.575178, -58.435014),  # MINISTRO CARRANZA - MIGUEL ABUELO - Line D
    (-34.570012, -58.444668),  # OLLEROS - Line D
    (-34.566215, -58.452126),  # JOSE HERNANDEZ - Line D
    (-34.562309, -58.456489),  # JURAMENTO - Line D
    (-34.587198, -58.455029),  # FEDERICO LACROZE - Line B
    (-34.643312, -58.461652),  # PLAZA DE LOS VIRREYES - EVA PERON - Line E
    (-34.640137, -58.457892),  # VARELA - Line E
    (-34.607802, -58.373956),  # CATEDRAL - Line D
    (-34.608810, -58.370968),  # PLAZA DE MAYO - Line A
    (-34.555642, -58.462378),  # CONGRESO DE TUCUMAN - Line D
    (-34.602989, -58.369930),  # LEANDRO N. ALEM - Line B
    (-34.584095, -58.466227),  # TRONADOR - VILLA ORTÚZAR - Line B
    (-34.581249, -58.474241),  # DE LOS INCAS -PQUE. CHAS - Line B
    (-34.626667, -58.456710),  # CARABOBO - Line A
    (-34.623529, -58.448648),  # PUAN - Line A
    (-34.604490, -58.405450),  # CORRIENTES - Line H
    (-34.638406, -58.405795),  # PARQUE PATRICIOS - Line H
    (-34.641269, -58.412385),  # HOSPITALES - Line H
    (-34.577797, -58.481014),  # ECHEVERRÍA - Line B
    (-34.574319, -58.486385),  # JUAN MANUEL DE ROSAS - VILLA URQUIZA - Line B
    (-34.630707, -58.469640),  # SAN PEDRITO - Line A
    (-34.629087, -58.463541),  # SAN JOSÉ DE FLORES - Line A
    (-34.598455, -58.403722),  # CÓRDOBA - Line H
    (-34.587462, -58.397216),  # LAS HERAS - Line H
    (-34.594525, -58.402376),  # SANTA FE - CARLOS JAUREGUI - Line H
    (-34.583036, -58.391019),  # FACULTAD DE DERECHO - JULIETA LANTERI - Line H
    (-34.592114, -58.375850),  # RETIRO - Line E
    (-34.596597, -58.371700),  # CATALINAS - Line E
    (-34.603014, -58.370413),  # CORREO CENTRAL - Line E
]

# Load bus stops data
print("Loading bus stops data...")
bus_stops_path = '/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/data/bus_stops_coords.npy'
if os.path.exists(bus_stops_path):
    BUS_STOPS = np.load(bus_stops_path)
    print(f"✅ Loaded {len(BUS_STOPS):,} bus stops from OpenStreetMap")
else:
    print("⚠️  Bus stops file not found, skipping bus stop features")
    BUS_STOPS = None

# Load additional POI data (parks, schools, hospitals, etc.)
print("\nLoading additional Points of Interest...")
POI_DATA = {}
poi_types = ['parks', 'schools', 'hospitals', 'supermarkets', 'restaurants', 'green_spaces']

for poi_type in poi_types:
    poi_path = f'data/{poi_type}_coords.npy'
    if os.path.exists(poi_path):
        POI_DATA[poi_type] = np.load(poi_path)
        print(f"✅ Loaded {len(POI_DATA[poi_type]):,} {poi_type}")
    else:
        print(f"⚠️  {poi_type} file not found")
        POI_DATA[poi_type] = None

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r

def calculate_distance_to_nearest_subway(lat, lon):
    """Calculate distance to nearest subway station in km"""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    
    distances = [haversine(lon, lat, station[1], station[0]) 
                 for station in SUBWAY_STATIONS]
    return min(distances)

def count_subway_stations_nearby(lat, lon, radius_km=1.0):
    """Count number of subway stations within radius_km"""
    if pd.isna(lat) or pd.isna(lon):
        return 0
    
    count = sum(1 for station in SUBWAY_STATIONS 
                if haversine(lon, lat, station[1], station[0]) <= radius_km)
    return count

def count_bus_stops_nearby(lat, lon, radius_km=0.5):
    """
    Count number of bus stops within radius_km (default 500m = 0.5km)
    Uses vectorized numpy operations for better performance
    """
    if BUS_STOPS is None or pd.isna(lat) or pd.isna(lon):
        return 0
    
    # Vectorized haversine calculation for all bus stops at once
    lon1, lat1 = radians(lon), radians(lat)
    lon2 = np.radians(BUS_STOPS[:, 1])
    lat2 = np.radians(BUS_STOPS[:, 0])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = 6371 * c  # Earth radius in km
    
    return int(np.sum(distances <= radius_km))

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

def calculate_distance_to_nearest_poi(lat, lon, poi_coords):
    """Calculate distance to nearest POI of a given type"""
    if poi_coords is None or len(poi_coords) == 0 or pd.isna(lat) or pd.isna(lon):
        return np.nan

    # Vectorized calculation for all POIs
    lon1, lat1 = radians(lon), radians(lat)
    lon2 = np.radians(poi_coords[:, 1])
    lat2 = np.radians(poi_coords[:, 0])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = 6371 * c  # Earth radius in km

    return float(np.min(distances))

def count_pois_nearby(lat, lon, poi_coords, radius_km=0.5):
    """Count POIs within radius"""
    if poi_coords is None or len(poi_coords) == 0 or pd.isna(lat) or pd.isna(lon):
        return 0

    # Vectorized calculation
    lon1, lat1 = radians(lon), radians(lat)
    lon2 = np.radians(poi_coords[:, 1])
    lat2 = np.radians(poi_coords[:, 0])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = 6371 * c

    return int(np.sum(distances <= radius_km))

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n### 1. Loading data...")
csv_path = '/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/data/properati_preprocessed_20251016_204913.csv'
df = pd.read_csv(csv_path)
print(f"Loaded data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# 2. BASIC DATA CLEANING
# ============================================================================
print("\n### 2. Basic data cleaning...")

# Filter for USD only
if 'currency' in df.columns:
    print(f"Currency distribution: {df['currency'].value_counts().to_dict()}")
    df = df[df['currency'] == 'USD'].copy()
    df = df.drop(columns=['currency'])
    print(f"Filtered for USD only. New shape: {df.shape}")

# Filter reasonable price range - EXPANDED to capture more data
print(f"Price range before filtering: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
df = df[(df['price'] >= 30_000) & (df['price'] <= 600_000)].copy()
print(f"Price range after filtering: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"Final dataset size: {len(df):,} properties")

# Filter by price per sqm (domain knowledge: Buenos Aires typical range)
print(f"\n### Filtering by price per square meter...")
df['price_per_sqm_temp'] = df['price'] / df['area']
before_ppsqm = len(df)
df = df[(df['price_per_sqm_temp'] >= 800) & (df['price_per_sqm_temp'] <= 5000)].copy()
df = df.drop(columns=['price_per_sqm_temp'])
after_ppsqm = len(df)
print(f"Removed {before_ppsqm - after_ppsqm:,} properties with unusual price/m² (outside $800-$5,000/m²)")
print(f"Final dataset size: {len(df):,} properties")

# Drop rows with missing critical features
critical_features = ['area', 'bedrooms', 'bathrooms', 'price']
before_drop = len(df)
df = df.dropna(subset=critical_features)
after_drop = len(df)
print(f"Removed {before_drop - after_drop} rows with missing critical features")
print(f"Final dataset size: {len(df):,} properties")

# ============================================================================
# 3. FEATURE ENGINEERING - ADD SUBWAY AND BUS STOP FEATURES
# ============================================================================
print("\n### 3. Engineering location features (subway stations & bus stops)...")

# Ensure we have coordinates - DROP rows instead of filling (avoid artificial clusters)
if 'latitude' in df.columns and 'longitude' in df.columns:
    before_coords = len(df)
    df = df.dropna(subset=['latitude', 'longitude'])
    after_coords = len(df)
    print(f"  Removed {before_coords - after_coords:,} properties with missing coordinates")
    print(f"  Remaining: {after_coords:,} properties")
    
    # Calculate distance to nearest subway station
    print("  Calculating distance to nearest subway station...")
    calc_start = time.time()
    df['distance_to_nearest_subway'] = df.apply(
        lambda row: calculate_distance_to_nearest_subway(row['latitude'], row['longitude']),
        axis=1
    )
    print(f"  ✅ Completed in {time.time() - calc_start:.1f} seconds")
    print(f"    Mean distance: {df['distance_to_nearest_subway'].mean():.3f} km")
    
    # Count subway stations nearby (within 1km)
    print("  Counting nearby subway stations (within 1km)...")
    calc_start = time.time()
    df['subway_stations_nearby'] = df.apply(
        lambda row: count_subway_stations_nearby(row['latitude'], row['longitude'], radius_km=1.0),
        axis=1
    )
    print(f"  ✅ Completed in {time.time() - calc_start:.1f} seconds")
    print(f"    Properties with 0 nearby stations: {(df['subway_stations_nearby'] == 0).sum()}")
    print(f"    Properties with 1+ nearby stations: {(df['subway_stations_nearby'] > 0).sum()}")
    
    # Count bus stops nearby (within 500m) if data available
    if BUS_STOPS is not None:
        print("  Counting nearby bus stops (within 500m)...")
        calc_start = time.time()
        df['bus_stops_nearby'] = df.apply(
            lambda row: count_bus_stops_nearby(row['latitude'], row['longitude'], radius_km=0.5),
            axis=1
        )
        print(f"  ✅ Completed in {time.time() - calc_start:.1f} seconds")
        print(f"    Mean bus stops within 500m: {df['bus_stops_nearby'].mean():.2f}")
    else:
        df['bus_stops_nearby'] = 0
        print("  ⚠️  Skipping bus stops (data not available)")

    # Distance to premium neighborhoods (high-value areas)
    print("\n  Calculating distances to premium neighborhoods...")
    calc_start = time.time()

    # Distance to each premium neighborhood
    for name, coords in PREMIUM_NEIGHBORHOODS.items():
        df[f'dist_to_{name.lower()}'] = df.apply(
            lambda row: calculate_distance_to_neighborhood(row['latitude'], row['longitude'], coords),
            axis=1
        )
    print(f"  ✅ Created distances to {len(PREMIUM_NEIGHBORHOODS)} premium neighborhoods in {time.time() - calc_start:.1f} seconds")

    # Minimum distance to any premium area
    df['min_dist_to_premium'] = df.apply(
        lambda row: calculate_min_distance_to_premium_areas(row['latitude'], row['longitude']),
        axis=1
    )
    print(f"    Mean min distance to premium areas: {df['min_dist_to_premium'].mean():.2f} km")

    # Distance to low-value neighborhoods (inverse indicator)
    for name, coords in LOW_VALUE_NEIGHBORHOODS.items():
        df[f'dist_to_{name.lower()}'] = df.apply(
            lambda row: calculate_distance_to_neighborhood(row['latitude'], row['longitude'], coords),
            axis=1
        )
    print(f"  ✅ Created distances to {len(LOW_VALUE_NEIGHBORHOODS)} reference neighborhoods")

    # ========================================================================
    # NEW: Additional POI Features (parks, schools, hospitals, etc.)
    # ========================================================================
    print("\n  Calculating POI-based features...")

    # Distance to nearest POI of each type
    poi_distance_features = {}
    for poi_type, poi_coords in POI_DATA.items():
        if poi_coords is not None and len(poi_coords) > 0:
            print(f"    Processing {poi_type}...")
            calc_start = time.time()

            # Distance to nearest
            df[f'dist_to_nearest_{poi_type}'] = df.apply(
                lambda row: calculate_distance_to_nearest_poi(row['latitude'], row['longitude'], poi_coords),
                axis=1
            )

            # Count nearby (within 1km for most, 0.5km for restaurants/supermarkets)
            radius = 0.5 if poi_type in ['restaurants', 'supermarkets'] else 1.0
            df[f'{poi_type}_nearby'] = df.apply(
                lambda row: count_pois_nearby(row['latitude'], row['longitude'], poi_coords, radius_km=radius),
                axis=1
            )

            poi_distance_features[poi_type] = f'dist_to_nearest_{poi_type}'
            print(f"    ✅ Completed in {time.time() - calc_start:.1f}s (mean dist: {df[f'dist_to_nearest_{poi_type}'].mean():.2f} km)")

    # Create walkability score (composite of nearby amenities)
    walkability_components = []
    if 'restaurants_nearby' in df.columns:
        walkability_components.append(df['restaurants_nearby'] / 10)  # Normalize
    if 'supermarkets_nearby' in df.columns:
        walkability_components.append(df['supermarkets_nearby'] * 2)  # Weight higher
    if 'parks_nearby' in df.columns:
        walkability_components.append(df['parks_nearby'] * 3)  # Weight higher
    if 'schools_nearby' in df.columns:
        walkability_components.append(df['schools_nearby'])

    if walkability_components:
        df['walkability_score'] = sum(walkability_components)
        print(f"\n  ✅ Created walkability_score (mean: {df['walkability_score'].mean():.2f})")

else:
    print("  ⚠️  No coordinates available, skipping location features")
    df['distance_to_nearest_subway'] = np.nan
    df['subway_stations_nearby'] = 0
    df['bus_stops_nearby'] = 0
    df['min_dist_to_premium'] = np.nan

# ============================================================================
# 4. DOMAIN-SPECIFIC COMPOSITE FEATURES (NO LEAKAGE)
# ============================================================================
print("\n### 4. Creating domain-specific composite features...")

# Luxury score - premium amenities that indicate high-end properties
if all(col in df.columns for col in ['has_gym', 'has_pool', 'has_doorman', 'has_security', 'has_garage']):
    df['luxury_score'] = (
        df.get('has_gym', 0).astype(int) +
        df.get('has_pool', 0).astype(int) +
        df.get('has_doorman', 0).astype(int) +
        df.get('has_security', 0).astype(int) +
        df.get('has_garage', 0).astype(int)
    )
    print(f"  Created luxury_score (range: {df['luxury_score'].min()}-{df['luxury_score'].max()})")
else:
    df['luxury_score'] = 0
    print("  ⚠️  Some luxury amenity columns missing, luxury_score set to 0")

# Family-friendly score - features attractive to families
df['family_score'] = 0
if 'has_balcony' in df.columns:
    df['family_score'] += df['has_balcony']
if 'has_terrace' in df.columns:
    df['family_score'] += df['has_terrace']
if 'has_grill' in df.columns:
    df['family_score'] += df['has_grill']
if 'bedrooms' in df.columns:
    df['family_score'] += (df['bedrooms'] >= 2).astype(int)
if 'bathrooms' in df.columns:
    df['family_score'] += (df['bathrooms'] >= 2).astype(int)
print(f"  Created family_score (range: {df['family_score'].min()}-{df['family_score'].max()})")

# Transportation accessibility score - inverse of distance + nearby stations
if 'distance_to_nearest_subway' in df.columns and 'subway_stations_nearby' in df.columns:
    df['transport_score'] = (
        (1 / (df['distance_to_nearest_subway'] + 0.1)) * 10 +  # Normalized inverse distance
        df['subway_stations_nearby'] +
        df['bus_stops_nearby'] / 10  # Normalize bus stops (can be >50)
    )
    print(f"  Created transport_score (mean: {df['transport_score'].mean():.2f})")
else:
    df['transport_score'] = 0
    print("  ⚠️  Transport features missing, transport_score set to 0")

# Total amenities count
amenity_cols = [col for col in df.columns if col.startswith('has_')]
df['total_amenities'] = df[amenity_cols].sum(axis=1)
print(f"  Created total_amenities from {len(amenity_cols)} amenity columns")

# Log transform for skewed features (right-skewed distributions common in real estate)
print("\n### 5. Log-transforming skewed features...")
df['log_area'] = np.log1p(df['area'])  # log(1 + area) to handle zeros
print(f"  Created log_area (mean: {df['log_area'].mean():.2f})")

if 'distance_to_nearest_subway' in df.columns:
    df['log_distance_subway'] = np.log1p(df['distance_to_nearest_subway'])
    print(f"  Created log_distance_subway (mean: {df['log_distance_subway'].mean():.2f})")

# Simple but effective interaction features
print("\n### 5b. Creating interaction features...")
# Area interactions (size matters differently with rooms)
df['area_x_bedrooms'] = df['area'] * df['bedrooms']
df['area_x_bathrooms'] = df['area'] * df['bathrooms']
print(f"  Created area × bedrooms and area × bathrooms")

# Luxury properties in good locations
if 'luxury_score' in df.columns and 'transport_score' in df.columns:
    df['luxury_x_transport'] = df['luxury_score'] * df['transport_score']
    print(f"  Created luxury_score × transport_score")

# Area per room ratios (spaciousness)
df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 0.5)  # Add 0.5 to avoid division by zero
df['area_per_bathroom'] = df['area'] / (df['bathrooms'] + 0.5)
print(f"  Created area_per_bedroom and area_per_bathroom (spaciousness indicators)")

# ============================================================================
# 6. SELECT FEATURES
# ============================================================================
print("\n### 6. Selecting features...")

# Use basic numeric features
basic_features = ['area', 'bedrooms', 'bathrooms']

# Add location coordinates if available
if 'latitude' in df.columns and 'longitude' in df.columns:
    basic_features.extend(['latitude', 'longitude'])

# Add location-based features
if 'distance_to_nearest_subway' in df.columns:
    basic_features.append('distance_to_nearest_subway')
if 'subway_stations_nearby' in df.columns:
    basic_features.append('subway_stations_nearby')
if 'bus_stops_nearby' in df.columns:
    basic_features.append('bus_stops_nearby')

# Add neighborhood distance features (NEW - premium and low-value areas)
neighborhood_features = []
# Add premium neighborhood distances
for name in PREMIUM_NEIGHBORHOODS.keys():
    feat = f'dist_to_{name.lower()}'
    if feat in df.columns:
        basic_features.append(feat)
        neighborhood_features.append(feat)
# Add min distance to any premium area
if 'min_dist_to_premium' in df.columns:
    basic_features.append('min_dist_to_premium')
    neighborhood_features.append('min_dist_to_premium')
# Add low-value neighborhood distances
for name in LOW_VALUE_NEIGHBORHOODS.keys():
    feat = f'dist_to_{name.lower()}'
    if feat in df.columns:
        basic_features.append(feat)
        neighborhood_features.append(feat)
if neighborhood_features:
    print(f"Added {len(neighborhood_features)} neighborhood distance features")

# Add POI features (NEW - external OpenStreetMap data)
poi_features = []
for poi_type in poi_types:
    # Distance to nearest
    feat_dist = f'dist_to_nearest_{poi_type}'
    if feat_dist in df.columns:
        basic_features.append(feat_dist)
        poi_features.append(feat_dist)
    # Count nearby
    feat_count = f'{poi_type}_nearby'
    if feat_count in df.columns:
        basic_features.append(feat_count)
        poi_features.append(feat_count)
# Walkability score
if 'walkability_score' in df.columns:
    basic_features.append('walkability_score')
    poi_features.append('walkability_score')
if poi_features:
    print(f"Added {len(poi_features)} POI-based features (parks, schools, hospitals, etc.)")

# Add composite features (NEW - high impact, no leakage)
composite_features = ['luxury_score', 'family_score', 'transport_score', 'total_amenities']
for feat in composite_features:
    if feat in df.columns:
        basic_features.append(feat)
print(f"Added {len([f for f in composite_features if f in df.columns])} composite features")

# Add log-transformed features (NEW - captures non-linearity)
log_features = ['log_area', 'log_distance_subway']
for feat in log_features:
    if feat in df.columns:
        basic_features.append(feat)
print(f"Added {len([f for f in log_features if f in df.columns])} log-transformed features")

# Add interaction features (NEW - capture multiplicative relationships)
interaction_features = ['area_x_bedrooms', 'area_x_bathrooms', 'luxury_x_transport',
                        'area_per_bedroom', 'area_per_bathroom']
for feat in interaction_features:
    if feat in df.columns:
        basic_features.append(feat)
print(f"Added {len([f for f in interaction_features if f in df.columns])} interaction features")

# Add individual amenity features (keep all, not just first 5)
amenity_cols = [col for col in df.columns if col.startswith('has_')]
if amenity_cols:
    basic_features.extend(amenity_cols)
    print(f"Added {len(amenity_cols)} individual amenity features")

# Ensure all features exist
basic_features = [f for f in basic_features if f in df.columns]
print(f"All available features ({len(basic_features)})")

# Load top 22 features (95% importance) for optimized model
print("\n### Applying feature selection (top 22 features contributing 95% importance)...")
top_features_file = 'data/top_features_95pct.txt'
if os.path.exists(top_features_file):
    with open(top_features_file, 'r') as f:
        top_22_features = [line.strip() for line in f if not line.startswith('#') and line.strip()]
    print(f"Loaded {len(top_22_features)} top features from {top_features_file}")

    # Filter basic_features to only include top 22 that are actually in the data
    basic_features = [f for f in basic_features if f in top_22_features]
    print(f"Reduced to {len(basic_features)} features (some may be missing from dataset)")
else:
    print(f"⚠️  Warning: {top_features_file} not found. Using all {len(basic_features)} features.")

print(f"Selected features ({len(basic_features)}): {basic_features}")

# Prepare feature matrix and target
X = df[basic_features].copy()
y = df['price'].copy()

# Handle any remaining NaN values
X = X.fillna(X.median())

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================================
# 7. TRAIN-TEST SPLIT (with validation set for early stopping)
# ============================================================================
print("\n### 7. Splitting data into train, validation, and test sets...")
# First split: 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Second split: 85% train, 15% validation (of the 80%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Apply log transform to target variable (prices are right-skewed)
print("\n### 8. Applying log transform to target variable...")
print(f"Price range before log: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)
print(f"Log price range: {y_train_log.min():.3f} - {y_train_log.max():.3f}")
print("Note: Will transform predictions back using np.expm1()")

# ============================================================================
# 9. SCALE FEATURES
# ============================================================================
print("\n### 9. Scaling features...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
print("✅ Features scaled with StandardScaler")

# ============================================================================
# 10. TRAIN MODEL - XGBoost with Bayesian Hyperparameter Optimization
# ============================================================================
print("\n### 10. Training model...")
print("Using XGBoost Regressor (gradient boosting with regularization)")
print("Training on log-transformed prices for better performance on wide price range")

if BAYESIAN_OPT_AVAILABLE:
    print("\n🔍 Performing Bayesian hyperparameter optimization...")
    print("This will take 3-5 minutes but should find better parameters")

    # Define search space
    search_spaces = {
        'n_estimators': Integer(200, 800),
        'max_depth': Integer(4, 8),
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'subsample': Real(0.6, 0.9),
        'colsample_bytree': Real(0.6, 0.9),
        'reg_alpha': Real(0.5, 5.0, prior='log-uniform'),
        'reg_lambda': Real(1.0, 10.0, prior='log-uniform'),
        'min_child_weight': Integer(1, 10),
        'gamma': Real(0.0, 0.5),
    }

    # Base model
    base_model = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    # Bayesian optimization
    opt = BayesSearchCV(
        base_model,
        search_spaces,
        n_iter=20,  # 20 iterations (reasonable for time vs performance trade-off)
        cv=3,  # 3-fold CV
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    opt.fit(X_train_scaled, y_train_log)

    print(f"\n✅ Bayesian optimization complete!")
    print(f"Best parameters found: {opt.best_params_}")
    print(f"Best CV score (MSE): {-opt.best_score_:.2f}")

    # Use best model and retrain with early stopping
    best_params = opt.best_params_
    model = XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50
    )

    model.fit(
        X_train_scaled, y_train_log,
        eval_set=[(X_val_scaled, y_val_log)],
        verbose=False
    )
    print(f"✅ Model retrained with best parameters (stopped at {model.best_iteration} iterations)")

else:
    # Fallback to fixed hyperparameters if Bayesian opt not available
    print("\nUsing fixed hyperparameters (install scikit-optimize for Bayesian optimization)")

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=2.0,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50
    )

    print(f"Model parameters: n_estimators={model.n_estimators}, max_depth={model.max_depth}, " +
          f"learning_rate={model.learning_rate}, reg_alpha={model.reg_alpha}, reg_lambda={model.reg_lambda}")
    print("Using early stopping with validation set to prevent overfitting")

    # Train on log-transformed target with validation set for early stopping
    model.fit(
        X_train_scaled, y_train_log,
        eval_set=[(X_val_scaled, y_val_log)],
        verbose=False
    )
    print(f"✅ Model trained on log-transformed prices (stopped at {model.best_iteration} iterations)")

#Store XGBoost model
xgb_model = model

# ============================================================================
# 10b. TRAIN LIGHTGBM MODEL FOR ENSEMBLE
# ============================================================================
print("\n### 10b. Training LightGBM model for ensemble...")

lgbm_model = LGBMRegressor(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.75,
    colsample_bytree=0.75,
    reg_alpha=2.0,
    reg_lambda=3.0,
    min_child_samples=20,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgbm_model.fit(
    X_train_scaled, y_train_log,
    eval_set=[(X_val_scaled, y_val_log)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)
    ]
)
print(f"✅ LightGBM model trained (stopped at {lgbm_model.best_iteration_} iterations)")

# ============================================================================
# 11. CREATE ENSEMBLE & EVALUATE
# ============================================================================
print("\n### 11. Creating ensemble and evaluating...")

# Get predictions from both models (log scale)
xgb_train_pred_log = xgb_model.predict(X_train_scaled)
xgb_test_pred_log = xgb_model.predict(X_test_scaled)

lgbm_train_pred_log = lgbm_model.predict(X_train_scaled)
lgbm_test_pred_log = lgbm_model.predict(X_test_scaled)

# Optimize ensemble weights based on validation set performance
xgb_val_pred_log = xgb_model.predict(X_val_scaled)
lgbm_val_pred_log = lgbm_model.predict(X_val_scaled)

# Calculate validation RMSE for each model
xgb_val_rmse = np.sqrt(mean_squared_error(y_val_log, xgb_val_pred_log))
lgbm_val_rmse = np.sqrt(mean_squared_error(y_val_log, lgbm_val_pred_log))

# Inverse RMSE weights (better models get higher weight)
weight_xgb = (1 / xgb_val_rmse) / ((1 / xgb_val_rmse) + (1 / lgbm_val_rmse))
weight_lgbm = (1 / lgbm_val_rmse) / ((1 / xgb_val_rmse) + (1 / lgbm_val_rmse))

print(f"Ensemble weights: XGBoost={weight_xgb:.3f}, LightGBM={weight_lgbm:.3f}")

# Ensemble predictions (log scale)
y_train_pred_log = weight_xgb * xgb_train_pred_log + weight_lgbm * lgbm_train_pred_log
y_test_pred_log = weight_xgb * xgb_test_pred_log + weight_lgbm * lgbm_test_pred_log

# Transform predictions back to original scale
y_train_pred = np.expm1(y_train_pred_log)
y_test_pred = np.expm1(y_test_pred_log)

# Also get individual model predictions for comparison
xgb_test_pred = np.expm1(xgb_test_pred_log)
lgbm_test_pred = np.expm1(lgbm_test_pred_log)

# Calculate metrics on original scale
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Calculate individual model metrics for comparison
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
xgb_test_r2 = r2_score(y_test, xgb_test_pred)
lgbm_test_rmse = np.sqrt(mean_squared_error(y_test, lgbm_test_pred))
lgbm_test_r2 = r2_score(y_test, lgbm_test_pred)

print("\n" + "=" * 80)
print("MODEL EVALUATION RESULTS")
print("=" * 80)

print(f"\n📊 Individual Models (Test Set):")
print(f"  XGBoost:  RMSE=${xgb_test_rmse:,.2f}, R²={xgb_test_r2:.4f}")
print(f"  LightGBM: RMSE=${lgbm_test_rmse:,.2f}, R²={lgbm_test_r2:.4f}")

print(f"\n🏆 Ensemble (Test Set):")
print(f"  RMSE: ${test_rmse:,.2f}")
print(f"  MAE:  ${test_mae:,.2f}")
print(f"  R²:   {test_r2:.4f}")

print(f"\n📈 Training Set (Ensemble):")
print(f"  RMSE: ${train_rmse:,.2f}")
print(f"  MAE:  ${train_mae:,.2f}")
print(f"  R²:   {train_r2:.4f}")

# Calculate improvement over best individual model
best_individual_rmse = min(xgb_test_rmse, lgbm_test_rmse)
improvement = ((best_individual_rmse - test_rmse) / best_individual_rmse) * 100
print(f"\n✨ Ensemble improvement over best individual: {improvement:.2f}%")

# ============================================================================
# 12. SAVE ENSEMBLE MODEL
# ============================================================================
print("\n### 12. Saving ensemble model...")
os.makedirs('models', exist_ok=True)

# Save both models and weights for ensemble
ensemble_data = {
    'xgb_model': xgb_model,
    'lgbm_model': lgbm_model,
    'weight_xgb': weight_xgb,
    'weight_lgbm': weight_lgbm,
    'scaler': scaler
}

ensemble_path = 'models/ensemble_price_model.joblib'
joblib.dump(ensemble_data, ensemble_path)
print(f"✅ Ensemble model saved: {ensemble_path}")

# Also save individual models for compatibility
joblib.dump(xgb_model, 'models/xgb_model.joblib')
joblib.dump(lgbm_model, 'models/lgbm_model.joblib')
joblib.dump(scaler, 'models/simple_scaler.joblib')
print(f"✅ Individual models saved for reference")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

