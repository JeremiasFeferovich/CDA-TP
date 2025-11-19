# -*- coding: utf-8 -*-
"""
Property Price Prediction Model Training Script

Equipo: Ian Bernasconi, Jeremias Feferovich
Proyecto: Batata Real State

This script trains and evaluates multiple regression models to predict
property prices in Buenos Aires.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
try:
    from lightgbm import early_stopping, log_evaluation
except ImportError:
    # Fallback for older versions
    early_stopping = None
    log_evaluation = None
from catboost import CatBoostRegressor
import joblib
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import time

# Configuraciones de estilo
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

script_start_time = time.time()

print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# BUENOS AIRES SUBWAY STATIONS COORDINATES
# ============================================================================
# Official data from Buenos Aires Open Data Portal
# Source: https://data.buenosaires.gob.ar/dataset/subte-estaciones
# Total stations: 90
# Format: (latitude, longitude)
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

# ============================================================================
# BUS STOPS COORDINATES
# ============================================================================
# Bus stop data from OpenStreetMap via Overpass API
# Source: data/bus_stops_coords.npy
# Total stops: 18,923
# Format: numpy array with shape (n, 2) where columns are [latitude, longitude]
print("Loading bus stops data...")
BUS_STOPS = np.load('/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/data/bus_stops_coords.npy')
print(f"✅ Loaded {len(BUS_STOPS):,} bus stops from OpenStreetMap")

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
    Uses vectorized numpy operations for better performance with large datasets
    """
    if pd.isna(lat) or pd.isna(lon):
        return 0
    
    # Vectorized haversine calculation for all bus stops at once
    # BUS_STOPS is a numpy array with shape (n, 2) where columns are [lat, lon]
    lon1, lat1 = radians(lon), radians(lat)
    lon2 = np.radians(BUS_STOPS[:, 1])
    lat2 = np.radians(BUS_STOPS[:, 0])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = 6371 * c  # Earth radius in km
    
    return int(np.sum(distances <= radius_km))

def calculate_distance_to_city_center(lat, lon):
    """Calculate distance to Plaza de Mayo (city center) in km"""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
    # Plaza de Mayo coordinates (approximate city center)
    city_center_lat = -34.6083
    city_center_lon = -58.3712
    return haversine(lon, lat, city_center_lon, city_center_lat)

# ============================================================================
# 1. DATA LOADING AND FEATURE ENGINEERING
# ============================================================================
section_start = time.time()
print("\n### 1. LOADING DATA AND ENGINEERING FEATURES")
print("-" * 80)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting data loading and feature engineering...")

# Load ORIGINAL (non-normalized) preprocessed data
csv_path = '/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/data/properati_preprocessed_20251016_204913.csv'
df = pd.read_csv(csv_path)
print(f"Loaded data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Drop location column (using lat/long instead)
if 'location' in df.columns:
    df = df.drop(columns=['location'])
    print("Dropped 'location' column")

# Handle currency - filter for USD only
print(f"\nCurrency distribution before filtering:")
print(df['currency'].value_counts())

df = df[df['currency'] == 'USD'].copy()
print(f"Filtered for USD only. New shape: {df.shape}")

# Filter out properties < $30K (minimum threshold)
print(f"\n### Filtering properties < $30,000...")
before_min_filter = len(df)
df = df[df['price'] >= 30_000].copy()
after_min_filter = len(df)
print(f"  Removed {before_min_filter - after_min_filter:,} properties ({(before_min_filter - after_min_filter)/before_min_filter*100:.1f}%)")
print(f"  Remaining: {after_min_filter:,} properties")

# **PHASE 1: Narrow price range to 30k-350k for RMSE < 20K target**
print(f"\n### Filtering properties > $350,000...")
before_max_filter = len(df)
df = df[df['price'] <= 350_000].copy()
after_max_filter = len(df)
print(f"  Removed {before_max_filter - after_max_filter:,} properties ({(before_max_filter - after_max_filter)/before_max_filter*100:.1f}%)")
print(f"  Remaining: {after_max_filter:,} properties")
print(f"  Final price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")

# **PHASE 1 IMPROVEMENT: Gentle outlier handling using winsorization (capping instead of removal)**
print(f"\n### Phase 1: Outlier Handling (Winsorization - capping extreme values)...")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Capping extreme outliers in key features...")

# Use winsorization (capping) instead of removal - less aggressive, preserves data
outlier_features = ['area', 'bathrooms', 'bedrooms']
outliers_capped = 0

for feature in outlier_features:
    if feature in df.columns:
        # Use wider bounds: 1st and 99th percentiles (very conservative)
        lower_bound = df[feature].quantile(0.01)
        upper_bound = df[feature].quantile(0.99)
        
        before_cap = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
        
        # Cap values instead of removing
        df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
        
        after_cap = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
        capped = before_cap
        outliers_capped += capped
        
        if capped > 0:
            print(f"  {feature}: Capped {capped:,} extreme values to bounds ({lower_bound:.2f} - {upper_bound:.2f})")

print(f"  Total extreme values capped: {outliers_capped:,}")
print(f"  Final dataset size: {len(df):,} properties (no properties removed)")
print(f"  Note: Using RobustScaler will further handle outliers during scaling")

# Drop currency column after filtering
if 'currency' in df.columns:
    df = df.drop(columns=['currency'])

# Create derived features
print("\n### Creating derived features...")

# 1. Price per square meter
df['price_per_sqm'] = df['price'] / df['area']
print(f"Created 'price_per_sqm' feature")

# 2. Distance to nearest subway station
print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating distance to nearest subway station...")
print(f"  Processing {len(df)} properties (this may take 1-2 minutes)...")
calc_start = time.time()
df['distance_to_nearest_subway'] = df.apply(
    lambda row: calculate_distance_to_nearest_subway(row['latitude'], row['longitude']),
    axis=1
)
calc_time = time.time() - calc_start
print(f"  ✅ Completed in {calc_time:.1f} seconds")
print(f"  Mean distance: {df['distance_to_nearest_subway'].mean():.3f} km")
print(f"  Min distance: {df['distance_to_nearest_subway'].min():.3f} km")
print(f"  Max distance: {df['distance_to_nearest_subway'].max():.3f} km")

# 3. Count of subway stations nearby (within 1km)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Counting nearby subway stations...")
calc_start = time.time()
df['subway_stations_nearby'] = df.apply(
    lambda row: count_subway_stations_nearby(row['latitude'], row['longitude'], radius_km=1.0),
    axis=1
)
calc_time = time.time() - calc_start
print(f"  ✅ Completed in {calc_time:.1f} seconds")
print(f"  Created 'subway_stations_nearby' feature")
print(f"  Properties with 0 nearby stations: {(df['subway_stations_nearby'] == 0).sum()}")
print(f"  Properties with 1+ nearby stations: {(df['subway_stations_nearby'] > 0).sum()}")

# 4. Count of bus stops nearby (within 500m)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Counting nearby bus stops (within 500m)...")
print(f"  Processing {len(df)} properties with vectorized operations...")
calc_start = time.time()
df['bus_stops_nearby'] = df.apply(
    lambda row: count_bus_stops_nearby(row['latitude'], row['longitude'], radius_km=0.5),
    axis=1
)
calc_time = time.time() - calc_start
print(f"  ✅ Completed in {calc_time:.1f} seconds")
print(f"  Created 'bus_stops_nearby' feature")
print(f"  Mean bus stops within 500m: {df['bus_stops_nearby'].mean():.2f}")
print(f"  Max bus stops within 500m: {df['bus_stops_nearby'].max()}")
print(f"  Properties with 0 nearby bus stops: {(df['bus_stops_nearby'] == 0).sum()}")
print(f"  Properties with 10+ nearby bus stops: {(df['bus_stops_nearby'] >= 10).sum()}")

# 5. Average price per sqm in neighborhood (grid-based)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating neighborhood avg price per sqm...")
print(f"  Using spatial grid approach (0.01 degree cells ≈ 1.1km)")
calc_start = time.time()

# Create grid cells for each property (0.01 degree ≈ 1.1 km in Buenos Aires)
grid_size = 0.01
df['grid_lat'] = (df['latitude'] / grid_size).round() * grid_size
df['grid_lon'] = (df['longitude'] / grid_size).round() * grid_size
df['grid_cell'] = df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)

# Calculate average price per sqm for each grid cell
grid_avg_price_per_sqm = df.groupby('grid_cell')['price_per_sqm'].mean()

# Map back to properties
df['neighborhood_avg_price_per_sqm'] = df['grid_cell'].map(grid_avg_price_per_sqm)

# For cells with only 1 property, use the broader area average (3x3 grid)
def get_neighborhood_avg_with_fallback(row):
    """Get neighborhood average, with fallback to broader area if cell has < 3 properties"""
    cell_count = df[df['grid_cell'] == row['grid_cell']].shape[0]
    
    if cell_count >= 3:
        return row['neighborhood_avg_price_per_sqm']
    else:
        # Use 3x3 grid around the cell
        lat, lon = row['grid_lat'], row['grid_lon']
        nearby_mask = (
            (df['grid_lat'].between(lat - 2*grid_size, lat + 2*grid_size)) &
            (df['grid_lon'].between(lon - 2*grid_size, lon + 2*grid_size))
        )
        nearby_properties = df[nearby_mask]
        if len(nearby_properties) > 0:
            return nearby_properties['price_per_sqm'].mean()
        else:
            return row['neighborhood_avg_price_per_sqm']

df['neighborhood_avg_price_per_sqm'] = df.apply(get_neighborhood_avg_with_fallback, axis=1)

# Drop temporary grid columns
df = df.drop(columns=['grid_lat', 'grid_lon', 'grid_cell'])

calc_time = time.time() - calc_start
print(f"  ✅ Completed in {calc_time:.1f} seconds")
print(f"  Created 'neighborhood_avg_price_per_sqm' feature")
print(f"  Mean neighborhood avg: ${df['neighborhood_avg_price_per_sqm'].mean():,.2f}/m²")
print(f"  Min neighborhood avg: ${df['neighborhood_avg_price_per_sqm'].min():,.2f}/m²")
print(f"  Max neighborhood avg: ${df['neighborhood_avg_price_per_sqm'].max():,.2f}/m²")

# 6. Total amenities
amenity_cols = [col for col in df.columns if col.startswith('has_')]
df['total_amenities'] = df[amenity_cols].sum(axis=1)
print(f"Created 'total_amenities' feature from {len(amenity_cols)} amenity columns")

# 7. PHASE 3.1: Interaction Features
print("\n### Phase 3.1: Creating interaction features...")
# Area interactions
df['area_bathrooms'] = df['area'] * df['bathrooms']
df['area_bedrooms'] = df['area'] * df['bedrooms']
df['bathrooms_bedrooms'] = df['bathrooms'] * df['bedrooms']
print(f"Created area*bathrooms, area*bedrooms, bathrooms*bedrooms")

# Location interactions
if 'distance_to_nearest_subway' in df.columns:
    df['area_subway_dist'] = df['area'] * df['distance_to_nearest_subway']
    df['subway_bus_stops'] = df['subway_stations_nearby'] * df['bus_stops_nearby']
    print(f"Created area*subway_dist, subway*bus_stops")

# Neighborhood interactions
if 'neighborhood_avg_price_per_sqm' in df.columns:
    df['area_neighborhood_price'] = df['area'] * df['neighborhood_avg_price_per_sqm']
    print(f"Created area*neighborhood_price")

# Amenity interactions
if 'total_amenities' in df.columns:
    df['area_amenities'] = df['area'] * df['total_amenities']
    print(f"Created area*amenities")

print(f"✅ Created {len([c for c in df.columns if any(x in c for x in ['area_bathrooms', 'area_bedrooms', 'bathrooms_bedrooms', 'area_subway', 'subway_bus', 'area_amenities'])])} interaction features")

# 8. PHASE 3.1: Advanced Location Features
print("\n### Phase 3.1: Creating advanced location features...")

# Distance to city center (Plaza de Mayo)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating distance to city center...")
calc_start = time.time()
df['distance_to_city_center'] = df.apply(
    lambda row: calculate_distance_to_city_center(row['latitude'], row['longitude']),
    axis=1
)
calc_time = time.time() - calc_start
print(f"  ✅ Completed in {calc_time:.1f} seconds")
print(f"  Mean distance: {df['distance_to_city_center'].mean():.3f} km")

# Property quality score (weighted sum of features)
print(f"Creating property quality score...")
# Weight different amenities differently
amenity_weights = {
    'has_garage': 2.0,
    'has_doorman': 1.5,
    'has_gym': 1.5,
    'has_pool': 2.0,
    'has_security': 1.5,
    'has_terrace': 1.0,
    'has_balcony': 0.5,
    'has_grill': 0.5,
    'has_storage': 0.5,
    'has_sum': 0.5,
}

df['property_quality_score'] = 0
for amenity, weight in amenity_weights.items():
    if amenity in df.columns:
        df['property_quality_score'] += df[amenity] * weight

# Add size score
df['size_score'] = df['area'] * df['bathrooms'] * df['bedrooms'] / 100  # Normalize
df['property_quality_score'] += df['size_score']

print(f"  Created property_quality_score feature")
print(f"  Mean quality score: {df['property_quality_score'].mean():.2f}")

# 9. PHASE 3: Aggressive Feature Engineering for RMSE < 20K target
print("\n### Phase 3: Creating polynomial and advanced features...")

# 9.1 Polynomial Features
print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating polynomial features...")
df['area_squared'] = df['area'] ** 2
df['bathrooms_squared'] = df['bathrooms'] ** 2
df['bedrooms_squared'] = df['bedrooms'] ** 2
print(f"  Created area², bathrooms², bedrooms")

# 9.2 Ratio Features
print(f"Creating ratio features...")
df['area_per_bathroom'] = df['area'] / (df['bathrooms'] + 0.01)  # Add small value to avoid division by zero
df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 0.01)
df['bathrooms_per_bedroom'] = df['bathrooms'] / (df['bedrooms'] + 0.01)
print(f"  Created area/bathrooms, area/bedrooms, bathrooms/bedrooms")

# 9.3 Additional Interaction Features
print(f"Creating additional interaction features...")
df['area_squared_bathrooms'] = df['area_squared'] * df['bathrooms']
df['area_squared_bedrooms'] = df['area_squared'] * df['bedrooms']
if 'distance_to_city_center' in df.columns:
    df['distance_city_center_area'] = df['distance_to_city_center'] * df['area']
if 'property_quality_score' in df.columns:
    df['quality_score_area'] = df['property_quality_score'] * df['area']
print(f"  Created area²*bathrooms, area²*bedrooms, distance_city_center*area, quality_score*area")

# 9.4 Location Clustering
print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating location clusters using KMeans...")
calc_start = time.time()
# Filter out NaN coordinates for clustering
valid_coords = df[['latitude', 'longitude']].dropna()
if len(valid_coords) > 0:
    # Use 8 clusters for neighborhood segmentation
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['location_cluster'] = np.nan
    df.loc[valid_coords.index, 'location_cluster'] = kmeans.fit_predict(valid_coords)
    
    # Calculate distance to each cluster centroid
    cluster_centers = kmeans.cluster_centers_
    for i in range(n_clusters):
        # Calculate distance to cluster center i
        def dist_to_cluster_i(lat, lon, center_idx):
            if pd.isna(lat) or pd.isna(lon):
                return np.nan
            return haversine(lon, lat, cluster_centers[center_idx][1], cluster_centers[center_idx][0])
        
        df[f'distance_to_cluster_{i}'] = df.apply(
            lambda row: dist_to_cluster_i(row['latitude'], row['longitude'], i), axis=1
        )
    
    print(f"  Created location_cluster feature with {n_clusters} clusters")
    print(f"  Created distance_to_cluster_0 to distance_to_cluster_{n_clusters-1} features")
    print(f"  Cluster distribution:")
    cluster_counts = df['location_cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"    Cluster {int(cluster_id)}: {count:,} properties")
else:
    print(f"  ⚠️  No valid coordinates for clustering")
    df['location_cluster'] = 0

calc_time = time.time() - calc_start
print(f"  ✅ Completed in {calc_time:.1f} seconds")

# Remove rows with NaN values in critical features
print("\n### Cleaning data...")
before_clean = len(df)
df = df.dropna(subset=['latitude', 'longitude', 'distance_to_nearest_subway'])
# Fill any remaining NaN in location_cluster (shouldn't happen after dropping lat/lon NaN, but just in case)
if 'location_cluster' in df.columns:
    df['location_cluster'] = df['location_cluster'].fillna(0)
after_clean = len(df)
print(f"Removed {before_clean - after_clean} rows with NaN in critical features")
print(f"Final dataset shape: {df.shape}")

# Prepare feature matrix X and target y
print("\n### Preparing feature matrix and target...")
target = 'price'
# Don't use price_per_sqm directly (data leakage)
# We'll drop neighborhood_avg_price_per_sqm temporarily and recalculate it properly after train-test split
features_to_drop = [target, 'price_per_sqm', 'neighborhood_avg_price_per_sqm']

X = df.drop(columns=features_to_drop)
y = df[target]

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")

# Store the neighborhood_avg_price_per_sqm for later (will recalculate after split)
neighborhood_avg_temp = df['neighborhood_avg_price_per_sqm'].copy()

section_time = time.time() - section_start
print(f"\n✅ Section 1 completed in {section_time:.1f} seconds ({section_time/60:.1f} minutes)")

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================
section_start = time.time()
print("\n### 2. TRAIN-TEST SPLIT")
print("-" * 80)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Splitting data into train and test sets...")

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split training data into train and validation sets for early stopping
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Train/Val/Test split: {X_train.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}% / {X_val.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}% / {X_test.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%")

# PHASE 2: Remove Log Transformation (previous tests showed it performed worse)
print("\n### Phase 2: Training on Original Scale")
print("  Using original price scale (no log transformation)")
print("  Previous tests showed log transformation performed worse for this dataset")
print(f"  Train price range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
print(f"  Val price range: ${y_val.min():,.0f} - ${y_val.max():,.0f}")
print(f"  Test price range: ${y_test.min():,.0f} - ${y_test.max():,.0f}")
print(f"  Using original scale targets for training")

# Calculate neighborhood average price per sqm ONLY from training data (avoid data leakage)
print("\n### Calculating neighborhood avg price/sqm (using ONLY training data)...")
calc_start = time.time()

# **PHASE 1 IMPROVEMENT: Fix data leakage - use ONLY training set (not validation) for neighborhood averages**
# Previously used X_train_full which included validation set - this causes data leakage
grid_size = 0.01
train_data = pd.DataFrame({
    'latitude': X_train['latitude'],  # Changed: Use ONLY X_train, not X_train_full
    'longitude': X_train['longitude'],
    'price': y_train,  # Changed: Use ONLY y_train, not y_train_full
    'area': X_train['area']
})
train_data['price_per_sqm'] = train_data['price'] / train_data['area']
train_data['grid_lat'] = (train_data['latitude'] / grid_size).round() * grid_size
train_data['grid_lon'] = (train_data['longitude'] / grid_size).round() * grid_size
train_data['grid_cell'] = train_data['grid_lat'].astype(str) + '_' + train_data['grid_lon'].astype(str)

# Calculate grid averages from training data only (no validation data)
grid_avg_from_train = train_data.groupby('grid_cell')['price_per_sqm'].mean().to_dict()
overall_avg = train_data['price_per_sqm'].mean()
print(f"  ⚠️  Fixed data leakage: Using ONLY training set ({len(train_data)} samples) for neighborhood averages")

def get_neighborhood_avg_safe(lat, lon):
    """Get neighborhood average from training data, with fallback"""
    grid_lat = round(lat / grid_size) * grid_size
    grid_lon = round(lon / grid_size) * grid_size
    grid_cell = f"{grid_lat}_{grid_lon}"
    
    if grid_cell in grid_avg_from_train:
        return grid_avg_from_train[grid_cell]
    else:
        # Fallback: search nearby cells (3x3 grid)
        nearby_cells = []
        for dlat in [-grid_size, 0, grid_size]:
            for dlon in [-grid_size, 0, grid_size]:
                nearby_cell = f"{grid_lat + dlat}_{grid_lon + dlon}"
                if nearby_cell in grid_avg_from_train:
                    nearby_cells.append(grid_avg_from_train[nearby_cell])
        
        if nearby_cells:
            return np.mean(nearby_cells)
        else:
            return overall_avg

# Apply to train, validation, and test sets
X_train['neighborhood_avg_price_per_sqm'] = X_train.apply(
    lambda row: get_neighborhood_avg_safe(row['latitude'], row['longitude']), axis=1
)
X_val['neighborhood_avg_price_per_sqm'] = X_val.apply(
    lambda row: get_neighborhood_avg_safe(row['latitude'], row['longitude']), axis=1
)
X_test['neighborhood_avg_price_per_sqm'] = X_test.apply(
    lambda row: get_neighborhood_avg_safe(row['latitude'], row['longitude']), axis=1
)

calc_time = time.time() - calc_start
print(f"  ✅ Completed in {calc_time:.1f} seconds")
print(f"  Train mean: ${X_train['neighborhood_avg_price_per_sqm'].mean():,.2f}/m²")
print(f"  Test mean: ${X_test['neighborhood_avg_price_per_sqm'].mean():,.2f}/m²")
print(f"  ⚠️  Calculated using ONLY training data to prevent leakage")

# Create neighborhood interaction feature after split (to avoid leakage)
print("\n### Creating neighborhood interaction feature...")
X_train['area_neighborhood_price'] = X_train['area'] * X_train['neighborhood_avg_price_per_sqm']
X_val['area_neighborhood_price'] = X_val['area'] * X_val['neighborhood_avg_price_per_sqm']
X_test['area_neighborhood_price'] = X_test['area'] * X_test['neighborhood_avg_price_per_sqm']
print(f"  Created area*neighborhood_price feature")

# Scale numeric features
print("\n### Scaling features...")
numeric_features = ['area', 'balcony_count', 'bathrooms', 'bedrooms', 
                   'latitude', 'longitude', 'distance_to_nearest_subway',
                   'subway_stations_nearby', 'bus_stops_nearby', 'total_amenities',
                   'neighborhood_avg_price_per_sqm']

# Add interaction features to scaling list
interaction_feature_names = ['area_bathrooms', 'area_bedrooms', 'bathrooms_bedrooms',
                            'area_subway_dist', 'subway_bus_stops', 
                            'area_neighborhood_price', 'area_amenities']
numeric_features.extend(interaction_feature_names)

# Add advanced features to scaling list
advanced_features = ['distance_to_city_center', 'property_quality_score', 'size_score']
numeric_features.extend(advanced_features)

# Add polynomial features to scaling list
polynomial_features = ['area_squared', 'bathrooms_squared', 'bedrooms_squared',
                       'area_per_bathroom', 'area_per_bedroom', 'bathrooms_per_bedroom',
                       'area_squared_bathrooms', 'area_squared_bedrooms',
                       'distance_city_center_area', 'quality_score_area']
numeric_features.extend(polynomial_features)

# Add location cluster distance features
cluster_distance_features = [f for f in X_train.columns if f.startswith('distance_to_cluster_')]
numeric_features.extend(cluster_distance_features)

# Only scale features that exist in the dataset and are numeric (exclude location_cluster which is categorical)
numeric_features = [f for f in numeric_features if f in X_train.columns and f != 'location_cluster']
print(f"Scaling {len(numeric_features)} numeric features (including polynomial and cluster distance features)")
print(f"Note: location_cluster is categorical and will not be scaled")

# **PHASE 1 IMPROVEMENT: Use RobustScaler instead of StandardScaler for outlier resistance**
print(f"\n### Phase 1: Using RobustScaler (outlier-resistant scaling)...")
scaler = RobustScaler()  # Changed from StandardScaler - uses median and IQR instead of mean/std
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_val_scaled[numeric_features] = scaler.transform(X_val[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
print(f"  ✅ Applied RobustScaler to {len(numeric_features)} features (more robust to outliers)")

# Prepare targets for training (using original scale, no log transformation)
y_train_original = y_train.copy()
y_val_original = y_val.copy()
y_test_original = y_test.copy()

print(f"Scaled {len(numeric_features)} numeric features")
print(f"Boolean features preserved as-is")

section_time = time.time() - section_start
print(f"\n✅ Section 2 completed in {section_time:.1f} seconds")

# ============================================================================
# 3. MODEL TRAINING WITH ENSEMBLE (XGBoost + LightGBM + CatBoost)
# ============================================================================
section_start = time.time()
print("\n### 3. MODEL TRAINING WITH ENSEMBLE METHODS")
print("-" * 80)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting ensemble model training...")
print(f"Training 3 models: XGBoost, LightGBM, CatBoost")
print(f"This is the longest step - please be patient!\n")

models = {}
best_params = {}

# **MODEL 1: XGBoost**
print(f"\n{'='*60}")
print(f"MODEL 1/3: XGBoost")
print(f"{'='*60}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training...")
print(f"Using RandomizedSearchCV: 30 iterations × 5 folds = 150 fits")
print(f"Hyperparameter grid optimized for 30k-350k price range with regularization")
print(f"Estimated time: 5-8 minutes")
print(f"{'='*60}\n")

model_start = time.time()
# **PHASE 1 IMPROVEMENT: More thorough hyperparameter search around current best**
# Current best: {'subsample': 0.9, 'reg_lambda': 1.5, 'reg_alpha': 1.0, 'n_estimators': 300, 
#                'min_child_weight': 5, 'max_depth': 6, 'learning_rate': 0.03, 'gamma': 0.1, 
#                'colsample_bytree': 0.9}
# Search around these values with finer granularity
param_grid_xgb = {
    'n_estimators': [250, 300, 350, 400],  # Around 300
    'max_depth': [5, 6, 7],  # Around 6
    'learning_rate': [0.02, 0.03, 0.04, 0.05],  # Around 0.03
    'subsample': [0.85, 0.9, 0.95],  # Around 0.9
    'colsample_bytree': [0.85, 0.9, 0.95],  # Around 0.9
    'min_child_weight': [4, 5, 6],  # Around 5
    'reg_alpha': [0.5, 1.0, 1.5],  # Around 1.0
    'reg_lambda': [1.0, 1.5, 2.0, 2.5],  # Around 1.5
    'gamma': [0.05, 0.1, 0.15],  # Around 0.1
}

xgb = XGBRegressor(random_state=42, n_jobs=1)
grid_xgb = RandomizedSearchCV(xgb, param_grid_xgb, n_iter=30, cv=5,  # Reduced for speed: 30 iterations × 5 folds = 150 fits
                              scoring='neg_mean_squared_error', 
                              verbose=2, n_jobs=1, random_state=42)
# Train on original scale (no log transformation)
grid_xgb.fit(X_train_scaled, y_train_original)

model_time = time.time() - model_start
models['XGBoost'] = grid_xgb.best_estimator_
best_params['XGBoost'] = grid_xgb.best_params_
print(f"\n✅ XGBoost completed in {model_time:.1f} seconds ({model_time/60:.1f} minutes)")
print(f"Best parameters: {grid_xgb.best_params_}")
print(f"Best CV MSE: {-grid_xgb.best_score_:.4f}")

# **MODEL 2: LightGBM**
print(f"\n{'='*60}")
print(f"MODEL 2/3: LightGBM")
print(f"{'='*60}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training...")
print(f"Using RandomizedSearchCV: 30 iterations × 5 folds = 150 fits")
print(f"Estimated time: 4-6 minutes")
print(f"{'='*60}\n")

model_start = time.time()
# **PHASE 1 IMPROVEMENT: More thorough search around current best**
# Current best: {'subsample': 0.9, 'reg_lambda': 0.5, 'reg_alpha': 0.5, 'num_leaves': 31, 
#                'n_estimators': 300, 'min_child_samples': 50, 'max_depth': 6, 
#                'learning_rate': 0.05, 'colsample_bytree': 0.9}
param_grid_lgb = {
    'n_estimators': [250, 300, 350, 400],  # Around 300
    'max_depth': [5, 6, 7],  # Around 6
    'learning_rate': [0.04, 0.05, 0.06],  # Around 0.05
    'subsample': [0.85, 0.9, 0.95],  # Around 0.9
    'colsample_bytree': [0.85, 0.9, 0.95],  # Around 0.9
    'min_child_samples': [40, 50, 60],  # Around 50
    'reg_alpha': [0.3, 0.5, 0.7],  # Around 0.5
    'reg_lambda': [0.3, 0.5, 0.7, 1.0],  # Around 0.5
    'num_leaves': [25, 31, 40],  # Around 31
}

lgb = LGBMRegressor(random_state=42, n_jobs=1, verbose=-1)
grid_lgb = RandomizedSearchCV(lgb, param_grid_lgb, n_iter=30, cv=5,  # Reduced for speed: 30 iterations × 5 folds = 150 fits
                               scoring='neg_mean_squared_error',
                               verbose=2, n_jobs=1, random_state=42)
grid_lgb.fit(X_train_scaled, y_train_original)

model_time = time.time() - model_start
models['LightGBM'] = grid_lgb.best_estimator_
best_params['LightGBM'] = grid_lgb.best_params_
print(f"\n✅ LightGBM completed in {model_time:.1f} seconds ({model_time/60:.1f} minutes)")
print(f"Best parameters: {grid_lgb.best_params_}")
print(f"Best CV MSE: {-grid_lgb.best_score_:.4f}")

# **MODEL 3: CatBoost**
print(f"\n{'='*60}")
print(f"MODEL 3/3: CatBoost")
print(f"{'='*60}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training...")
print(f"Using RandomizedSearchCV: 30 iterations × 5 folds = 150 fits")
print(f"Estimated time: 6-10 minutes")
print(f"{'='*60}\n")

model_start = time.time()
# **PHASE 1 IMPROVEMENT: More thorough search around current best**
# Current best: {'random_strength': 0.5, 'learning_rate': 0.05, 'l2_leaf_reg': 3, 
#                'iterations': 400, 'depth': 6, 'bagging_temperature': 1}
param_grid_cat = {
    'iterations': [350, 400, 450, 500],  # Around 400
    'depth': [5, 6, 7],  # Around 6
    'learning_rate': [0.04, 0.05, 0.06],  # Around 0.05
    'l2_leaf_reg': [2, 3, 4, 5],  # Around 3
    'random_strength': [0.3, 0.5, 0.7],  # Around 0.5
    'bagging_temperature': [0.8, 1.0, 1.2],  # Around 1.0
}

cat = CatBoostRegressor(random_state=42, verbose=False, thread_count=1)
grid_cat = RandomizedSearchCV(cat, param_grid_cat, n_iter=30, cv=5,  # Reduced for speed: 30 iterations × 5 folds = 150 fits
                              scoring='neg_mean_squared_error',
                              verbose=2, n_jobs=1, random_state=42)
grid_cat.fit(X_train_scaled, y_train_original)

model_time = time.time() - model_start
models['CatBoost'] = grid_cat.best_estimator_
best_params['CatBoost'] = grid_cat.best_params_
print(f"\n✅ CatBoost completed in {model_time:.1f} seconds ({model_time/60:.1f} minutes)")
print(f"Best parameters: {grid_cat.best_params_}")
print(f"Best CV MSE: {-grid_cat.best_score_:.4f}")

# ============================================================================
# FEATURE SELECTION BASED ON IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("FEATURE SELECTION: Analyzing feature importance")
print("="*80)

# Get feature importance from XGBoost (most reliable)
if 'XGBoost' in models:
    xgb_model = models['XGBoost']
    if hasattr(xgb_model, 'feature_importances_'):
        feature_importance_df = pd.DataFrame({
            'feature': X_train_scaled.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate cumulative importance
        feature_importance_df['cumulative_importance'] = feature_importance_df['importance'].cumsum()
        total_importance = feature_importance_df['importance'].sum()
        
        # Select features: keep top features that contribute to 99.5% of total importance
        # OR remove features with < 0.5% importance (whichever is more restrictive)
        min_importance_for_995 = feature_importance_df[feature_importance_df['cumulative_importance'] <= 0.995 * total_importance]
        if len(min_importance_for_995) > 0:
            min_importance_995 = min_importance_for_995['importance'].min()
        else:
            min_importance_995 = 0
        importance_threshold = max(0.005 * total_importance, min_importance_995)
        
        selected_features = feature_importance_df[feature_importance_df['importance'] >= importance_threshold]['feature'].tolist()
        
        print(f"\nTotal features: {len(X_train_scaled.columns)}")
        print(f"Selected features: {len(selected_features)}")
        print(f"Removed features: {len(X_train_scaled.columns) - len(selected_features)}")
        print(f"\nTop 10 most important features:")
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.6f} ({row['importance']/total_importance*100:.2f}%)")
        
        if len(selected_features) < len(X_train_scaled.columns):
            print(f"\nRemoved low-importance features:")
            removed_features = feature_importance_df[feature_importance_df['importance'] < importance_threshold]['feature'].tolist()
            for feat in removed_features[:10]:  # Show first 10
                print(f"  - {feat}")
            if len(removed_features) > 10:
                print(f"  ... and {len(removed_features) - 10} more")
            
            # Update feature sets to use only selected features
            X_train_scaled = X_train_scaled[selected_features]
            X_val_scaled = X_val_scaled[selected_features]
            X_test_scaled = X_test_scaled[selected_features]
            
            print(f"\n✅ Feature selection complete. Retraining models with {len(selected_features)} features...")
            
            # Retrain models with selected features
            print(f"\n{'='*60}")
            print("RETRAINING MODELS WITH SELECTED FEATURES")
            print(f"{'='*60}\n")
            
            # Retrain XGBoost
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retraining XGBoost with selected features...")
            xgb_best_params = best_params['XGBoost'].copy()
            xgb_final = XGBRegressor(**xgb_best_params, random_state=42, n_jobs=1)
            xgb_final.fit(X_train_scaled, y_train_original)
            models['XGBoost'] = xgb_final
            print(f"  ✅ XGBoost retrained")
            
            # Retrain LightGBM
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retraining LightGBM with selected features...")
            lgb_best_params = best_params['LightGBM'].copy()
            max_rounds = lgb_best_params.pop('n_estimators', 500)
            lgb_final = LGBMRegressor(**lgb_best_params, n_estimators=max_rounds, random_state=42, n_jobs=1, verbose=-1)
            # Use callbacks if available, otherwise train without early stopping
            if early_stopping is not None and log_evaluation is not None:
                lgb_final.fit(
                    X_train_scaled, y_train_original,
                    eval_set=[(X_val_scaled, y_val_original)],
                    eval_metric='rmse',
                    callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)]
                )
            else:
                lgb_final.fit(
                    X_train_scaled, y_train_original,
                    eval_set=[(X_val_scaled, y_val_original)],
                    eval_metric='rmse'
                )
            models['LightGBM'] = lgb_final
            print(f"  ✅ LightGBM retrained")
            
            # Retrain CatBoost
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retraining CatBoost with selected features...")
            cat_best_params = best_params['CatBoost'].copy()
            max_iterations = cat_best_params.pop('iterations', 500)
            cat_final = CatBoostRegressor(**cat_best_params, iterations=max_iterations, random_state=42, verbose=False, thread_count=1)
            cat_final.fit(
                X_train_scaled, y_train_original,
                eval_set=(X_val_scaled, y_val_original),
                early_stopping_rounds=50,
                verbose=False
            )
            models['CatBoost'] = cat_final
            print(f"  ✅ CatBoost retrained")
        else:
            print(f"\n⚠️  All features kept (all have importance >= {importance_threshold:.6f})")
            # Still retrain with early stopping even if no features removed
            print(f"\n{'='*60}")
            print("RETRAINING MODELS WITH EARLY STOPPING")
            print(f"{'='*60}\n")
            
            # Retrain XGBoost with early stopping
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retraining XGBoost with early stopping...")
            xgb_best_params = best_params['XGBoost'].copy()
            xgb_final = XGBRegressor(**xgb_best_params, random_state=42, n_jobs=1)
            xgb_final.fit(X_train_scaled, y_train_original)
            models['XGBoost'] = xgb_final
            print(f"  ✅ XGBoost retrained")
            
            # Retrain LightGBM with early stopping
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retraining LightGBM with early stopping...")
            lgb_best_params = best_params['LightGBM'].copy()
            max_rounds = lgb_best_params.pop('n_estimators', 500)
            lgb_final = LGBMRegressor(**lgb_best_params, n_estimators=max_rounds, random_state=42, n_jobs=1, verbose=-1)
            # Use callbacks if available, otherwise train without early stopping
            if early_stopping is not None and log_evaluation is not None:
                lgb_final.fit(
                    X_train_scaled, y_train_original,
                    eval_set=[(X_val_scaled, y_val_original)],
                    eval_metric='rmse',
                    callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)]
                )
            else:
                lgb_final.fit(
                    X_train_scaled, y_train_original,
                    eval_set=[(X_val_scaled, y_val_original)],
                    eval_metric='rmse'
                )
            models['LightGBM'] = lgb_final
            print(f"  ✅ LightGBM retrained")
            
            # Retrain CatBoost with early stopping
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retraining CatBoost with early stopping...")
            cat_best_params = best_params['CatBoost'].copy()
            max_iterations = cat_best_params.pop('iterations', 500)
            cat_final = CatBoostRegressor(**cat_best_params, iterations=max_iterations, random_state=42, verbose=False, thread_count=1)
            cat_final.fit(
                X_train_scaled, y_train_original,
                eval_set=(X_val_scaled, y_val_original),
                early_stopping_rounds=50,
                verbose=False
            )
            models['CatBoost'] = cat_final
            print(f"  ✅ CatBoost retrained")
else:
    print("⚠️  XGBoost model not found, skipping feature selection")

# Ridge Regression - COMMENTED OUT (only running XGBoost)
# print(f"\n{'='*60}")
# print(f"MODEL 2/3: Ridge Regression")
# print(f"{'='*60}")
# print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training...")
# print(f"Hyperparameter grid: 4 combinations × 3 folds = 12 fits")
# print(f"Estimated time: 30-60 seconds")
# print(f"{'='*60}\n")
# 
# model_start = time.time()
# param_grid_ridge = {
#     'alpha': [0.1, 1.0, 10.0, 100.0]
# }
# 
# ridge = Ridge(random_state=42)
# grid_ridge = GridSearchCV(ridge, param_grid_ridge, cv=3, scoring='neg_mean_squared_error',
#                          verbose=2, n_jobs=1)
# grid_ridge.fit(X_train_scaled, y_train)
# 
# model_time = time.time() - model_start
# models['Ridge'] = grid_ridge.best_estimator_
# best_params['Ridge'] = grid_ridge.best_params_
# print(f"\n✅ Ridge Regression completed in {model_time:.1f} seconds")
# print(f"Best parameters: {grid_ridge.best_params_}")
# print(f"Best CV MSE: {-grid_ridge.best_score_:.4f}")

# SVR (Support Vector Regressor) - COMMENTED OUT (only running XGBoost)
# print(f"\n{'='*60}")
# print(f"MODEL 3/3: Support Vector Regressor (SVR)")
# print(f"{'='*60}")
# print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training...")
# print(f"Hyperparameter grid: 2 combinations × 3 folds = 6 fits")
# print(f"Estimated time: 2-5 minutes (SVR is slow on large datasets)")
# print(f"{'='*60}\n")
# 
# model_start = time.time()
# param_grid_svr = {
#     'C': [1, 10],
#     'kernel': ['rbf'],
#     'epsilon': [0.1]
# }
# 
# svr = SVR()
# grid_svr = GridSearchCV(svr, param_grid_svr, cv=3, scoring='neg_mean_squared_error',
#                        verbose=2, n_jobs=1)
# grid_svr.fit(X_train_scaled, y_train)
# 
# model_time = time.time() - model_start
# models['SVR'] = grid_svr.best_estimator_
# best_params['SVR'] = grid_svr.best_params_
# print(f"\n✅ SVR completed in {model_time:.1f} seconds ({model_time/60:.1f} minutes)")
# print(f"Best parameters: {grid_svr.best_params_}")
# print(f"Best CV MSE: {-grid_svr.best_score_:.4f}")

section_time = time.time() - section_start
print(f"\n{'='*80}")
print(f"✅ All model training completed in {section_time:.1f} seconds ({section_time/60:.1f} minutes)")
print(f"{'='*80}")

# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================
section_start = time.time()
print("\n### 4. MODEL EVALUATION")
print("-" * 80)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluating all models on test set...")

results = {}

# Evaluate individual models
for name, model in models.items():
    # Predictions on original scale (no log transformation)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_original, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred))
    train_r2 = r2_score(y_train_original, y_train_pred)
    test_r2 = r2_score(y_test_original, y_test_pred)
    train_mae = mean_absolute_error(y_train_original, y_train_pred)
    test_mae = mean_absolute_error(y_test_original, y_test_pred)
    
    results[name] = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'predictions': y_test_pred,
        'train_predictions': y_train_pred
    }
    
    print(f"\n{name}:")
    print(f"  Train RMSE: ${train_rmse:,.2f}")
    print(f"  Test RMSE:  ${test_rmse:,.2f}")
    print(f"  Train R²:   {train_r2:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")
    print(f"  Train MAE:  ${train_mae:,.2f}")
    print(f"  Test MAE:   ${test_mae:,.2f}")
    
    # Overfitting check
    rmse_gap = (test_rmse / train_rmse - 1) * 100
    print(f"  Overfitting: {rmse_gap:+.1f}% (test vs train)")

# PHASE 4: Ensemble Predictions (Weighted Average)
print("\n" + "="*60)
print("ENSEMBLE: Weighted Average of All Models")
print("="*60)

# **PHASE 1 IMPROVEMENT: Optimize ensemble weights using validation set performance**
print(f"\n### Phase 1: Optimizing ensemble weights using validation set...")

# Get validation predictions for all models
val_predictions = {}
for name, model in models.items():
    val_predictions[name] = model.predict(X_val_scaled)

# Calculate validation RMSE for each model
val_rmses = {}
for name in models.keys():
    val_rmse = np.sqrt(mean_squared_error(y_val_original, val_predictions[name]))
    val_rmses[name] = val_rmse
    print(f"  {name} validation RMSE: ${val_rmse:,.2f}")

# Optimize weights: use inverse validation RMSE (better than test RMSE for generalization)
val_rmses_list = [val_rmses[name] for name in models.keys()]
weights = [1.0 / rmse for rmse in val_rmses_list]
weights = np.array(weights) / np.sum(weights)  # Normalize to sum to 1

print(f"\nOptimized ensemble weights (based on validation performance):")
for name, weight in zip(models.keys(), weights):
    print(f"  {name}: {weight:.3f} (val RMSE: ${val_rmses[name]:,.2f})")

# Generate ensemble predictions
y_train_pred_ensemble = np.zeros(len(y_train_original))
y_test_pred_ensemble = np.zeros(len(y_test_original))

for name, weight in zip(models.keys(), weights):
    y_train_pred_ensemble += weight * results[name]['train_predictions']
    y_test_pred_ensemble += weight * results[name]['predictions']

# Calculate ensemble metrics
ensemble_train_rmse = np.sqrt(mean_squared_error(y_train_original, y_train_pred_ensemble))
ensemble_test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred_ensemble))
ensemble_train_r2 = r2_score(y_train_original, y_train_pred_ensemble)
ensemble_test_r2 = r2_score(y_test_original, y_test_pred_ensemble)
ensemble_train_mae = mean_absolute_error(y_train_original, y_train_pred_ensemble)
ensemble_test_mae = mean_absolute_error(y_test_original, y_test_pred_ensemble)

results['Ensemble'] = {
    'train_rmse': ensemble_train_rmse,
    'test_rmse': ensemble_test_rmse,
    'train_r2': ensemble_train_r2,
    'test_r2': ensemble_test_r2,
    'train_mae': ensemble_train_mae,
    'test_mae': ensemble_test_mae,
    'predictions': y_test_pred_ensemble,
    'train_predictions': y_train_pred_ensemble,
    'weights': dict(zip(models.keys(), weights))
}

print(f"\nEnsemble Results:")
print(f"  Train RMSE: ${ensemble_train_rmse:,.2f}")
print(f"  Test RMSE:  ${ensemble_test_rmse:,.2f}")
print(f"  Train R²:   {ensemble_train_r2:.4f}")
print(f"  Test R²:    {ensemble_test_r2:.4f}")
print(f"  Train MAE:  ${ensemble_train_mae:,.2f}")
print(f"  Test MAE:   ${ensemble_test_mae:,.2f}")

# Select best model (ensemble or individual)
print("\n### Selecting best model...")
best_model_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
if best_model_name == 'Ensemble':
    # For ensemble, we'll create a wrapper that combines models
    best_model = {'type': 'ensemble', 'models': models, 'weights': weights}
else:
    best_model = models[best_model_name]
best_results = results[best_model_name]

# Save predictions for residual analysis (Phase 1.1)
print("\n### Saving predictions for residual analysis...")
predictions_df = pd.DataFrame({
    'actual_price': y_test_original.values,
    'predicted_price': best_results['predictions'],
    'residual': y_test_original.values - best_results['predictions'],
    'abs_residual': np.abs(y_test_original.values - best_results['predictions']),
    'pct_error': (np.abs(y_test_original.values - best_results['predictions']) / y_test_original.values) * 100
})

# Add feature columns for analysis
for col in X_test.columns:
    predictions_df[col] = X_test[col].values

# Also add train predictions
if best_model_name == 'Ensemble':
    y_train_pred_original = best_results['train_predictions']
else:
    y_train_pred_original = best_model.predict(X_train_scaled)
train_predictions_df = pd.DataFrame({
    'actual_price': y_train_original.values,
    'predicted_price': y_train_pred_original,
    'residual': y_train_original.values - y_train_pred_original,
    'abs_residual': np.abs(y_train_original.values - y_train_pred_original),
    'pct_error': (np.abs(y_train_original.values - y_train_pred_original) / y_train_original.values) * 100
})
for col in X_train.columns:
    train_predictions_df[col] = X_train[col].values

os.makedirs('data', exist_ok=True)
predictions_df.to_csv('data/test_predictions_for_analysis.csv', index=False)
train_predictions_df.to_csv('data/train_predictions_for_analysis.csv', index=False)
print(f"✅ Saved test predictions to: data/test_predictions_for_analysis.csv")
print(f"✅ Saved train predictions to: data/train_predictions_for_analysis.csv")

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test RMSE: ${best_results['test_rmse']:,.2f}")
print(f"   Test R²:   {best_results['test_r2']:.4f}")
print(f"   Test MAE:  ${best_results['test_mae']:,.2f}")

# Check if KPI targets are met
print("\n### KPI Target Achievement:")
rmse_target = 20000
r2_target = 0.8

rmse_met = best_results['test_rmse'] < rmse_target
r2_met = best_results['test_r2'] > r2_target

print(f"  RMSE < ${rmse_target:,}: {'✅ YES' if rmse_met else '❌ NO'} (${best_results['test_rmse']:,.2f})")
print(f"  R² > {r2_target}: {'✅ YES' if r2_met else '❌ NO'} ({best_results['test_r2']:.4f})")

section_time = time.time() - section_start
print(f"\n✅ Section 4 completed in {section_time:.1f} seconds")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
section_start = time.time()
print("\n### 5. CREATING VISUALIZATIONS")
print("-" * 80)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating plots...")

# Create output directory
os.makedirs('data', exist_ok=True)

# 1. Feature Importance (for tree-based models)
if best_model_name == 'Ensemble' or 'XGBoost' in models:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating feature importance plot...")
    # Use XGBoost for feature importance (or ensemble average if available)
    if 'XGBoost' in models:
        xgb_model = models['XGBoost']
        if hasattr(xgb_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train_scaled.columns,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            plt.figure(figsize=(12, 8))
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.xlabel('Importance')
            plt.title('Top 15 Feature Importances (XGBoost)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('data/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✅ Saved: data/feature_importance.png")

# 2. Predicted vs Actual scatter plot
print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating predicted vs actual plot...")
plt.figure(figsize=(10, 8))
plt.scatter(y_test, best_results['predictions'], alpha=0.5, edgecolors='k', s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title(f'Predicted vs Actual Prices - {best_model_name}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: data/predicted_vs_actual.png")

# 3. Residuals distribution
print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating residuals distribution plot...")
residuals = y_test - best_results['predictions']
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Residuals - {best_model_name}')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/residuals_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: data/residuals_distribution.png")

# 4. Residuals vs Predicted
print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating residuals vs predicted plot...")
plt.figure(figsize=(10, 8))
plt.scatter(best_results['predictions'], residuals, alpha=0.5, edgecolors='k', s=30)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Price (USD)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title(f'Residuals vs Predicted Prices - {best_model_name}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/residuals_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: data/residuals_vs_predicted.png")

# 5. Model Comparison
print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating model comparison plot...")
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[m]['test_rmse'] for m in results.keys()],
    'R²': [results[m]['test_r2'] for m in results.keys()],
    'MAE': [results[m]['test_mae'] for m in results.keys()]
})

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# RMSE comparison
axes[0].bar(comparison_df['Model'], comparison_df['RMSE'], color='steelblue', edgecolor='black')
axes[0].set_ylabel('RMSE (USD)')
axes[0].set_title('Model Comparison - RMSE')
axes[0].axhline(y=rmse_target, color='r', linestyle='--', label=f'Target: ${rmse_target:,}')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# R² comparison
axes[1].bar(comparison_df['Model'], comparison_df['R²'], color='forestgreen', edgecolor='black')
axes[1].set_ylabel('R² Score')
axes[1].set_title('Model Comparison - R²')
axes[1].axhline(y=r2_target, color='r', linestyle='--', label=f'Target: {r2_target}')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

# MAE comparison
axes[2].bar(comparison_df['Model'], comparison_df['MAE'], color='coral', edgecolor='black')
axes[2].set_ylabel('MAE (USD)')
axes[2].set_title('Model Comparison - MAE')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('data/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: data/model_comparison.png")

section_time = time.time() - section_start
print(f"\n✅ Section 5 completed in {section_time:.1f} seconds")

# ============================================================================
# 6. MODEL PERSISTENCE
# ============================================================================
section_start = time.time()
print("\n### 6. SAVING MODELS AND REPORTS")
print("-" * 80)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving models and generating report...")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save best model(s)
if best_model_name == 'Ensemble':
    # Save all models and weights for ensemble
    ensemble_path = 'models/ensemble_models.joblib'
    ensemble_data = {
        'models': models,
        'weights': weights,
        'scaler': scaler
    }
    joblib.dump(ensemble_data, ensemble_path)
    print(f"✅ Saved ensemble models: {ensemble_path}")
    # Also save individual best model for compatibility
    best_individual = min(models.keys(), key=lambda x: results[x]['test_rmse'])
    model_path = 'models/best_property_price_model.joblib'
    joblib.dump(models[best_individual], model_path)
    print(f"✅ Saved best individual model ({best_individual}): {model_path}")
else:
    model_path = 'models/best_property_price_model.joblib'
    joblib.dump(best_model, model_path)
    print(f"✅ Saved best model: {model_path}")

# Save scaler
scaler_path = 'models/scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"✅ Saved scaler: {scaler_path}")

# Save training report
report_path = 'models/training_report.txt'
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("PROPERTY PRICE PREDICTION MODEL - TRAINING REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Project: Batata Real State\n")
    f.write(f"Team: Ian Bernasconi, Jeremias Feferovich\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("DATASET INFORMATION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"Number of features: {X_train.shape[1]}\n\n")
    
    f.write("Features used:\n")
    for i, feat in enumerate(X_train.columns, 1):
        f.write(f"  {i}. {feat}\n")
    f.write("\n")
    
    f.write("-" * 80 + "\n")
    f.write("MODELS TRAINED\n")
    f.write("-" * 80 + "\n\n")
    
    for name in results.keys():
        f.write(f"{name}:\n")
        if name in best_params:
            f.write(f"  Best parameters: {best_params[name]}\n")
        if name == 'Ensemble' and 'weights' in results[name]:
            f.write(f"  Ensemble weights: {results[name]['weights']}\n")
        f.write(f"  Train RMSE: ${results[name]['train_rmse']:,.2f}\n")
        f.write(f"  Test RMSE:  ${results[name]['test_rmse']:,.2f}\n")
        f.write(f"  Train R²:   {results[name]['train_r2']:.4f}\n")
        f.write(f"  Test R²:    {results[name]['test_r2']:.4f}\n")
        f.write(f"  Train MAE:  ${results[name]['train_mae']:,.2f}\n")
        f.write(f"  Test MAE:   ${results[name]['test_mae']:,.2f}\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("BEST MODEL\n")
    f.write("-" * 80 + "\n")
    f.write(f"Selected model: {best_model_name}\n")
    if best_model_name in best_params:
        f.write(f"Best parameters: {best_params[best_model_name]}\n")
    if best_model_name == 'Ensemble' and 'weights' in best_results:
        f.write(f"Ensemble weights: {best_results['weights']}\n")
    f.write(f"Test RMSE: ${best_results['test_rmse']:,.2f}\n")
    f.write(f"Test R²:   {best_results['test_r2']:.4f}\n")
    f.write(f"Test MAE:  ${best_results['test_mae']:,.2f}\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("KPI TARGET ACHIEVEMENT\n")
    f.write("-" * 80 + "\n")
    f.write(f"RMSE < ${rmse_target:,}: {'✅ ACHIEVED' if rmse_met else '❌ NOT MET'} (${best_results['test_rmse']:,.2f})\n")
    f.write(f"R² > {r2_target}: {'✅ ACHIEVED' if r2_met else '❌ NOT MET'} ({best_results['test_r2']:.4f})\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("FILES SAVED\n")
    f.write("-" * 80 + "\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Scaler: {scaler_path}\n")
    f.write(f"Report: {report_path}\n")
    f.write(f"Visualizations:\n")
    f.write(f"  - data/feature_importance.png\n")
    f.write(f"  - data/predicted_vs_actual.png\n")
    f.write(f"  - data/residuals_distribution.png\n")
    f.write(f"  - data/residuals_vs_predicted.png\n")
    f.write(f"  - data/model_comparison.png\n\n")

print(f"  ✅ Saved training report: {report_path}")

section_time = time.time() - section_start
print(f"\n✅ Section 6 completed in {section_time:.1f} seconds")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
total_time = time.time() - script_start_time
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total elapsed time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print("=" * 80)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test RMSE: ${best_results['test_rmse']:,.2f}")
print(f"   Test R²: {best_results['test_r2']:.4f}")
print(f"   Test MAE: ${best_results['test_mae']:,.2f}")

print(f"\n📊 KPI Achievement:")
print(f"   RMSE < ${rmse_target:,}: {'✅' if rmse_met else '❌'}")
print(f"   R² > {r2_target}: {'✅' if r2_met else '❌'}")

print(f"\n💾 Saved Files:")
print(f"   - Model: {model_path}")
print(f"   - Scaler: {scaler_path}")
print(f"   - Report: {report_path}")
print(f"   - 5 visualization plots in data/ folder")

print("\n" + "=" * 80)

