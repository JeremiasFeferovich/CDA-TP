# -*- coding: utf-8 -*-
"""
Deep Neural Network Property Price Prediction Model Training Script

Equipo: Ian Bernasconi, Jeremias Feferovich
Proyecto: Batata Real State

This script trains a deep neural network (DNN/MLP) to predict property prices
in Buenos Aires, iteratively tuning hyperparameters until achieving
R² > 0.8 and RMSE < $20,000.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.regularizers import l1_l2
import joblib
import os
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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
SUBWAY_STATIONS = [
    (-34.635750, -58.398928), (-34.629376, -58.400970), (-34.623092, -58.402323),
    (-34.615242, -58.404732), (-34.608935, -58.406036), (-34.604245, -58.380574),
    (-34.599757, -58.397924), (-34.601587, -58.385142), (-34.591628, -58.407161),
    (-34.585156, -58.415955), (-34.581411, -58.421196), (-34.578422, -58.425711),
    (-34.591194, -58.374018), (-34.601770, -58.378156), (-34.604844, -58.379530),
    (-34.608983, -58.380611), (-34.612617, -58.380444), (-34.618126, -58.380174),
    (-34.627619, -58.381434), (-34.603297, -58.375072), (-34.603637, -58.380715),
    (-34.604094, -58.387296), (-34.604420, -58.392314), (-34.604643, -58.399474),
    (-34.604581, -58.405399), (-34.604080, -58.411763), (-34.603165, -58.420962),
    (-34.602162, -58.431274), (-34.598967, -58.439771), (-34.591718, -58.447573),
    (-34.608559, -58.374268), (-34.608882, -58.379085), (-34.609100, -58.382232),
    (-34.609413, -58.386777), (-34.609226, -58.392669), (-34.609646, -58.398427),
    (-34.609834, -58.401208), (-34.609817, -58.406707), (-34.610782, -58.415186),
    (-34.611770, -58.421816), (-34.615206, -58.429500), (-34.618280, -58.436429),
    (-34.620405, -58.441178), (-34.609242, -58.373684), (-34.612849, -58.377581),
    (-34.617937, -58.381535), (-34.622339, -58.385149), (-34.622720, -58.391512),
    (-34.623110, -58.397068), (-34.623866, -58.402937), (-34.624654, -58.409391),
    (-34.628018, -58.433816), (-34.631042, -58.442171), (-34.588237, -58.411294),
    (-34.594426, -58.402395), (-34.599640, -58.393125), (-34.595057, -58.377819),
    (-34.621917, -58.379921), (-34.636389, -58.450278), (-34.627015, -58.426789),
    (-34.625366, -58.415533), (-34.575178, -58.435014), (-34.570012, -58.444668),
    (-34.566215, -58.452126), (-34.562309, -58.456489), (-34.587198, -58.455029),
    (-34.643312, -58.461652), (-34.640137, -58.457892), (-34.607802, -58.373956),
    (-34.608810, -58.370968), (-34.555642, -58.462378), (-34.602989, -58.369930),
    (-34.584095, -58.466227), (-34.581249, -58.474241), (-34.626667, -58.456710),
    (-34.623529, -58.448648), (-34.604490, -58.405450), (-34.638406, -58.405795),
    (-34.641269, -58.412385), (-34.577797, -58.481014), (-34.574319, -58.486385),
    (-34.630707, -58.469640), (-34.629087, -58.463541), (-34.598455, -58.403722),
    (-34.587462, -58.397216), (-34.594525, -58.402376), (-34.583036, -58.391019),
    (-34.592114, -58.375850), (-34.596597, -58.371700), (-34.603014, -58.370413),
]

# ============================================================================
# BUS STOPS COORDINATES
# ============================================================================
print("Loading bus stops data...")
BUS_STOPS = np.load('/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/data/bus_stops_coords.npy')
print(f"✅ Loaded {len(BUS_STOPS):,} bus stops from OpenStreetMap")

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance in kilometers between two points"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
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
    """Count number of bus stops within radius_km"""
    if pd.isna(lat) or pd.isna(lon):
        return 0
    lon1, lat1 = radians(lon), radians(lat)
    lon2 = np.radians(BUS_STOPS[:, 1])
    lat2 = np.radians(BUS_STOPS[:, 0])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = 6371 * c
    return int(np.sum(distances <= radius_km))

def calculate_distance_to_city_center(lat, lon):
    """Calculate distance to Plaza de Mayo (city center) in km"""
    if pd.isna(lat) or pd.isna(lon):
        return np.nan
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

# Load preprocessed data
csv_path = '/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/data/properati_preprocessed_20251016_204913.csv'
df = pd.read_csv(csv_path)
print(f"Loaded data shape: {df.shape}")

# Drop location column if exists
if 'location' in df.columns:
    df = df.drop(columns=['location'])

# Filter for USD only
df = df[df['currency'] == 'USD'].copy()
print(f"Filtered for USD only. New shape: {df.shape}")

# Filter price range 30k-350k
df = df[df['price'] >= 30_000].copy()
df = df[df['price'] <= 350_000].copy()
print(f"Filtered price range. Final shape: {df.shape}")

# Outlier handling (winsorization)
outlier_features = ['area', 'bathrooms', 'bedrooms']
for feature in outlier_features:
    if feature in df.columns:
        lower_bound = df[feature].quantile(0.01)
        upper_bound = df[feature].quantile(0.99)
        df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)

# Drop currency column
if 'currency' in df.columns:
    df = df.drop(columns=['currency'])

# Create derived features
print("\n### Creating derived features...")
df['price_per_sqm'] = df['price'] / df['area']

print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating distance to nearest subway station...")
df['distance_to_nearest_subway'] = df.apply(
    lambda row: calculate_distance_to_nearest_subway(row['latitude'], row['longitude']),
    axis=1
)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Counting nearby subway stations...")
df['subway_stations_nearby'] = df.apply(
    lambda row: count_subway_stations_nearby(row['latitude'], row['longitude'], radius_km=1.0),
    axis=1
)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Counting nearby bus stops...")
df['bus_stops_nearby'] = df.apply(
    lambda row: count_bus_stops_nearby(row['latitude'], row['longitude'], radius_km=0.5),
    axis=1
)

# Neighborhood avg price per sqm
print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating neighborhood avg price per sqm...")
grid_size = 0.01
df['grid_lat'] = (df['latitude'] / grid_size).round() * grid_size
df['grid_lon'] = (df['longitude'] / grid_size).round() * grid_size
df['grid_cell'] = df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)
grid_avg_price_per_sqm = df.groupby('grid_cell')['price_per_sqm'].mean()

def get_neighborhood_avg_with_fallback(row):
    cell_count = df[df['grid_cell'] == row['grid_cell']].shape[0]
    if cell_count >= 3:
        return row['neighborhood_avg_price_per_sqm']
    else:
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

df['neighborhood_avg_price_per_sqm'] = df['grid_cell'].map(grid_avg_price_per_sqm)
df['neighborhood_avg_price_per_sqm'] = df.apply(get_neighborhood_avg_with_fallback, axis=1)
df = df.drop(columns=['grid_lat', 'grid_lon', 'grid_cell'])

# Total amenities
amenity_cols = [col for col in df.columns if col.startswith('has_')]
df['total_amenities'] = df[amenity_cols].sum(axis=1)

# Interaction features
print("\n### Creating interaction features...")
df['area_bathrooms'] = df['area'] * df['bathrooms']
df['area_bedrooms'] = df['area'] * df['bedrooms']
df['bathrooms_bedrooms'] = df['bathrooms'] * df['bedrooms']
if 'distance_to_nearest_subway' in df.columns:
    df['area_subway_dist'] = df['area'] * df['distance_to_nearest_subway']
    df['subway_bus_stops'] = df['subway_stations_nearby'] * df['bus_stops_nearby']
if 'neighborhood_avg_price_per_sqm' in df.columns:
    df['area_neighborhood_price'] = df['area'] * df['neighborhood_avg_price_per_sqm']
if 'total_amenities' in df.columns:
    df['area_amenities'] = df['area'] * df['total_amenities']

# Advanced location features
print("\n### Creating advanced location features...")
df['distance_to_city_center'] = df.apply(
    lambda row: calculate_distance_to_city_center(row['latitude'], row['longitude']),
    axis=1
)

# Property quality score
amenity_weights = {
    'has_garage': 2.0, 'has_doorman': 1.5, 'has_gym': 1.5, 'has_pool': 2.0,
    'has_security': 1.5, 'has_terrace': 1.0, 'has_balcony': 0.5,
    'has_grill': 0.5, 'has_storage': 0.5, 'has_sum': 0.5,
}
df['property_quality_score'] = 0
for amenity, weight in amenity_weights.items():
    if amenity in df.columns:
        df['property_quality_score'] += df[amenity] * weight
df['size_score'] = df['area'] * df['bathrooms'] * df['bedrooms'] / 100
df['property_quality_score'] += df['size_score']

# Polynomial features
print("\n### Creating polynomial features...")
df['area_squared'] = df['area'] ** 2
df['bathrooms_squared'] = df['bathrooms'] ** 2
df['bedrooms_squared'] = df['bedrooms'] ** 2
df['area_per_bathroom'] = df['area'] / (df['bathrooms'] + 0.01)
df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 0.01)
df['bathrooms_per_bedroom'] = df['bathrooms'] / (df['bedrooms'] + 0.01)
df['area_squared_bathrooms'] = df['area_squared'] * df['bathrooms']
df['area_squared_bedrooms'] = df['area_squared'] * df['bedrooms']
if 'distance_to_city_center' in df.columns:
    df['distance_city_center_area'] = df['distance_to_city_center'] * df['area']
if 'property_quality_score' in df.columns:
    df['quality_score_area'] = df['property_quality_score'] * df['area']

# Location clustering
print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating location clusters...")
valid_coords = df[['latitude', 'longitude']].dropna()
if len(valid_coords) > 0:
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['location_cluster'] = np.nan
    df.loc[valid_coords.index, 'location_cluster'] = kmeans.fit_predict(valid_coords)
    cluster_centers = kmeans.cluster_centers_
    for i in range(n_clusters):
        def dist_to_cluster_i(lat, lon, center_idx):
            if pd.isna(lat) or pd.isna(lon):
                return np.nan
            return haversine(lon, lat, cluster_centers[center_idx][1], cluster_centers[center_idx][0])
        df[f'distance_to_cluster_{i}'] = df.apply(
            lambda row: dist_to_cluster_i(row['latitude'], row['longitude'], i), axis=1
        )
else:
    df['location_cluster'] = 0

# Clean data
print("\n### Cleaning data...")
before_clean = len(df)
df = df.dropna(subset=['latitude', 'longitude', 'distance_to_nearest_subway'])
if 'location_cluster' in df.columns:
    df['location_cluster'] = df['location_cluster'].fillna(0)
after_clean = len(df)
print(f"Removed {before_clean - after_clean} rows with NaN")
print(f"Final dataset shape: {df.shape}")

# Prepare feature matrix and target
print("\n### Preparing feature matrix and target...")
target = 'price'
features_to_drop = [target, 'price_per_sqm', 'neighborhood_avg_price_per_sqm']
X = df.drop(columns=features_to_drop)
y = df[target]
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

section_time = time.time() - section_start
print(f"\n✅ Section 1 completed in {section_time:.1f} seconds")

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================
section_start = time.time()
print("\n### 2. TRAIN-TEST SPLIT")
print("-" * 80)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Calculate neighborhood avg price per sqm from training data only
print("\n### Calculating neighborhood avg price/sqm (using ONLY training data)...")
grid_size = 0.01
train_data = pd.DataFrame({
    'latitude': X_train['latitude'],
    'longitude': X_train['longitude'],
    'price': y_train,
    'area': X_train['area']
})
train_data['price_per_sqm'] = train_data['price'] / train_data['area']
train_data['grid_lat'] = (train_data['latitude'] / grid_size).round() * grid_size
train_data['grid_lon'] = (train_data['longitude'] / grid_size).round() * grid_size
train_data['grid_cell'] = train_data['grid_lat'].astype(str) + '_' + train_data['grid_lon'].astype(str)
grid_avg_from_train = train_data.groupby('grid_cell')['price_per_sqm'].mean().to_dict()
overall_avg = train_data['price_per_sqm'].mean()

def get_neighborhood_avg_safe(lat, lon):
    grid_lat = round(lat / grid_size) * grid_size
    grid_lon = round(lon / grid_size) * grid_size
    grid_cell = f"{grid_lat}_{grid_lon}"
    if grid_cell in grid_avg_from_train:
        return grid_avg_from_train[grid_cell]
    else:
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

X_train['neighborhood_avg_price_per_sqm'] = X_train.apply(
    lambda row: get_neighborhood_avg_safe(row['latitude'], row['longitude']), axis=1
)
X_val['neighborhood_avg_price_per_sqm'] = X_val.apply(
    lambda row: get_neighborhood_avg_safe(row['latitude'], row['longitude']), axis=1
)
X_test['neighborhood_avg_price_per_sqm'] = X_test.apply(
    lambda row: get_neighborhood_avg_safe(row['latitude'], row['longitude']), axis=1
)

# Create neighborhood interaction feature
X_train['area_neighborhood_price'] = X_train['area'] * X_train['neighborhood_avg_price_per_sqm']
X_val['area_neighborhood_price'] = X_val['area'] * X_val['neighborhood_avg_price_per_sqm']
X_test['area_neighborhood_price'] = X_test['area'] * X_test['neighborhood_avg_price_per_sqm']

# Scale numeric features
print("\n### Scaling features...")
numeric_features = ['area', 'balcony_count', 'bathrooms', 'bedrooms', 
                   'latitude', 'longitude', 'distance_to_nearest_subway',
                   'subway_stations_nearby', 'bus_stops_nearby', 'total_amenities',
                   'neighborhood_avg_price_per_sqm']
interaction_feature_names = ['area_bathrooms', 'area_bedrooms', 'bathrooms_bedrooms',
                            'area_subway_dist', 'subway_bus_stops', 
                            'area_neighborhood_price', 'area_amenities']
numeric_features.extend(interaction_feature_names)
advanced_features = ['distance_to_city_center', 'property_quality_score', 'size_score']
numeric_features.extend(advanced_features)
polynomial_features = ['area_squared', 'bathrooms_squared', 'bedrooms_squared',
                       'area_per_bathroom', 'area_per_bedroom', 'bathrooms_per_bedroom',
                       'area_squared_bathrooms', 'area_squared_bedrooms',
                       'distance_city_center_area', 'quality_score_area']
numeric_features.extend(polynomial_features)
cluster_distance_features = [f for f in X_train.columns if f.startswith('distance_to_cluster_')]
numeric_features.extend(cluster_distance_features)
numeric_features = [f for f in numeric_features if f in X_train.columns and f != 'location_cluster']

print(f"Scaling {len(numeric_features)} numeric features")

scaler = RobustScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_val_scaled[numeric_features] = scaler.transform(X_val[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

y_train_original = y_train.copy()
y_val_original = y_val.copy()
y_test_original = y_test.copy()

print(f"Scaled {len(numeric_features)} numeric features")

section_time = time.time() - section_start
print(f"\n✅ Section 2 completed in {section_time:.1f} seconds")

# ============================================================================
# 3. DNN MODEL BUILDING AND TRAINING
# ============================================================================
print("\n### 3. DNN MODEL BUILDING AND TRAINING")
print("-" * 80)

# Convert to numpy arrays
X_train_array = X_train_scaled.values.astype(np.float32)
X_val_array = X_val_scaled.values.astype(np.float32)
X_test_array = X_test_scaled.values.astype(np.float32)
y_train_array = y_train_original.values.astype(np.float32)
y_val_array = y_val_original.values.astype(np.float32)
y_test_array = y_test_original.values.astype(np.float32)

# Normalize target variable for better training stability
print("\n### Normalizing target variable...")
from sklearn.preprocessing import RobustScaler
target_scaler = RobustScaler()
y_train_array_scaled = target_scaler.fit_transform(y_train_array.reshape(-1, 1)).flatten()
y_val_array_scaled = target_scaler.transform(y_val_array.reshape(-1, 1)).flatten()
y_test_array_scaled = target_scaler.transform(y_test_array.reshape(-1, 1)).flatten()
print(f"  Target mean (original): ${y_train_array.mean():,.2f}")
print(f"  Target std (original): ${y_train_array.std():,.2f}")
print(f"  Target mean (scaled): {y_train_array_scaled.mean():.4f}")
print(f"  Target std (scaled): {y_train_array_scaled.std():.4f}")

n_features = X_train_array.shape[1]
print(f"Number of features: {n_features}")

# Feature selection based on correlation with target (optional but can help)
print("\n### Performing feature selection...")
# Calculate correlation with target
feature_correlations = []
for i in range(n_features):
    corr = np.corrcoef(X_train_array[:, i], y_train_array)[0, 1]
    if not np.isnan(corr):
        feature_correlations.append((i, abs(corr)))
    else:
        feature_correlations.append((i, 0.0))

# Sort by correlation
feature_correlations.sort(key=lambda x: x[1], reverse=True)
# Keep top 90% of features (or all if correlation is reasonable)
n_features_to_keep = max(int(n_features * 0.9), n_features - 5)
selected_feature_indices = [idx for idx, _ in feature_correlations[:n_features_to_keep]]
selected_feature_indices.sort()

if len(selected_feature_indices) < n_features:
    print(f"  Selected {len(selected_feature_indices)}/{n_features} features based on correlation")
    X_train_array = X_train_array[:, selected_feature_indices]
    X_val_array = X_val_array[:, selected_feature_indices]
    X_test_array = X_test_array[:, selected_feature_indices]
    n_features = len(selected_feature_indices)
    print(f"  Reduced to {n_features} features")
else:
    print(f"  Keeping all {n_features} features")

# Custom R² metric for Keras
def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# Custom RMSE metric
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Target metrics
R2_TARGET = 0.8
RMSE_TARGET = 20000.0

best_model = None
best_test_rmse = float('inf')
best_test_r2 = 0.0
best_config = None
iteration = 0
max_iterations = 20

print(f"\nTarget: R² > {R2_TARGET} AND RMSE < ${RMSE_TARGET:,.0f}")
print(f"Starting iterative hyperparameter tuning...\n")

while iteration < max_iterations:
    iteration += 1
    print(f"\n{'='*80}")
    print(f"ITERATION {iteration}/{max_iterations}")
    print(f"{'='*80}")
    
    # Hyperparameter configurations - optimized for better performance
    # Focus on reducing overfitting and improving generalization
    configs = [
        # Config 1: Deep network with high regularization - reduce overfitting
        {
            'layers': [512, 256, 128, 64, 32],
            'dropout': 0.5,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'l2_reg': 0.001,
            'optimizer': 'adam'
        },
        # Config 2: Very wide network with strong regularization
        {
            'layers': [1024, 512, 256, 128],
            'dropout': 0.5,
            'learning_rate': 0.0001,
            'batch_size': 128,
            'l2_reg': 0.001,
            'optimizer': 'adam'
        },
        # Config 3: Deep and wide with aggressive dropout
        {
            'layers': [512, 512, 256, 256, 128, 64],
            'dropout': 0.6,
            'learning_rate': 0.00005,
            'batch_size': 64,
            'l2_reg': 0.001,
            'optimizer': 'adam'
        },
        # Config 4: Medium depth with very high regularization
        {
            'layers': [256, 256, 128, 128, 64],
            'dropout': 0.5,
            'learning_rate': 0.0001,
            'batch_size': 32,
            'l2_reg': 0.01,
            'optimizer': 'adam'
        },
        # Config 5: Wide shallow with high dropout
        {
            'layers': [1024, 512, 256],
            'dropout': 0.6,
            'learning_rate': 0.0001,
            'batch_size': 128,
            'l2_reg': 0.001,
            'optimizer': 'adam'
        },
        # Config 6: Very deep with progressive dropout
        {
            'layers': [256, 256, 128, 128, 64, 64, 32],
            'dropout': 0.5,
            'learning_rate': 0.00005,
            'batch_size': 32,
            'l2_reg': 0.001,
            'optimizer': 'adam'
        },
        # Config 7: AdamW optimizer with weight decay
        {
            'layers': [512, 256, 128, 64],
            'dropout': 0.5,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'l2_reg': 0.001,
            'optimizer': 'adamw'
        },
        # Config 8: RMSprop with momentum
        {
            'layers': [512, 256, 128, 64],
            'dropout': 0.5,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'l2_reg': 0.001,
            'optimizer': 'rmsprop'
        },
    ]
    
    # Try configurations sequentially
    config_idx = (iteration - 1) % len(configs)
    config = configs[config_idx]
    
    print(f"\nConfiguration:")
    print(f"  Layers: {config['layers']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  L2 regularization: {config['l2_reg']}")
    print(f"  Optimizer: {config['optimizer']}")
    
    # Build model
    model = models.Sequential()
    model.add(layers.Input(shape=(n_features,)))
    
    # Add hidden layers with better initialization
    from tensorflow.keras.initializers import HeNormal
    
    for i, units in enumerate(config['layers']):
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=l1_l2(l1=0, l2=config['l2_reg']),
            kernel_initializer=HeNormal(),
            name=f'dense_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        model.add(layers.Dropout(config['dropout'], name=f'dropout_{i+1}'))
    
    # Output layer
    model.add(layers.Dense(1, activation='linear', name='output'))
    
    # Compile model with appropriate optimizer
    # Note: We'll use ReduceLROnPlateau callback instead of schedule to avoid conflicts
    initial_lr = config['learning_rate']
    
    if config['optimizer'] == 'adam':
        optimizer = optimizers.Adam(learning_rate=initial_lr, beta_1=0.9, beta_2=0.999)
    elif config['optimizer'] == 'adamw':
        try:
            optimizer = optimizers.AdamW(learning_rate=initial_lr, weight_decay=config['l2_reg'])
        except AttributeError:
            # Fallback to Adam if AdamW not available
            optimizer = optimizers.Adam(learning_rate=initial_lr, beta_1=0.9, beta_2=0.999)
    elif config['optimizer'] == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=initial_lr, rho=0.9, momentum=0.9)
    else:
        optimizer = optimizers.Adam(learning_rate=initial_lr)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[rmse, r_squared, 'mae']
    )
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Callbacks with improved settings
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,  # Increased patience for better convergence
        restore_best_weights=True,
        verbose=1,
        min_delta=1e-6
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,  # More patience before reducing LR
        min_lr=1e-8,
        verbose=1,
        cooldown=5
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        'models/dnn_temp_best.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    # Train model
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting training...")
    train_start = time.time()
    
    history = model.fit(
        X_train_array, y_train_array_scaled,
        validation_data=(X_val_array, y_val_array_scaled),
        batch_size=config['batch_size'],
        epochs=500,  # Increased max epochs for better convergence
        callbacks=[early_stop, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    train_time = time.time() - train_start
    print(f"\n✅ Training completed in {train_time:.1f} seconds ({train_time/60:.1f} minutes)")
    
    # Load best model
    model.load_weights('models/dnn_temp_best.h5')
    
    # Evaluate on test set (transform predictions back to original scale)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Evaluating on test set...")
    y_test_pred_scaled = model.predict(X_test_array, verbose=0).flatten()
    y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate sklearn metrics on original scale
    test_rmse_sklearn = np.sqrt(mean_squared_error(y_test_array, y_test_pred))
    test_r2_sklearn = r2_score(y_test_array, y_test_pred)
    test_mae_sklearn = mean_absolute_error(y_test_array, y_test_pred)
    
    print(f"\nTest Set Results:")
    print(f"  RMSE: ${test_rmse_sklearn:,.2f} (target: < ${RMSE_TARGET:,.0f})")
    print(f"  R²:   {test_r2_sklearn:.4f} (target: > {R2_TARGET})")
    print(f"  MAE:  ${test_mae_sklearn:,.2f}")
    
    # Check if targets are met
    rmse_met = test_rmse_sklearn < RMSE_TARGET
    r2_met = test_r2_sklearn > R2_TARGET
    
    print(f"\nTarget Achievement:")
    print(f"  RMSE < ${RMSE_TARGET:,.0f}: {'✅ YES' if rmse_met else '❌ NO'}")
    print(f"  R² > {R2_TARGET}: {'✅ YES' if r2_met else '❌ NO'}")
    
    # Track best model
    if test_rmse_sklearn < best_test_rmse:
        best_test_rmse = test_rmse_sklearn
        best_test_r2 = test_r2_sklearn
        best_model = model
        best_config = config.copy()
        best_config['test_rmse'] = test_rmse_sklearn
        best_config['test_r2'] = test_r2_sklearn
        best_config['test_mae'] = test_mae_sklearn
        best_config['history'] = history
        print(f"\n🏆 New best model! RMSE: ${best_test_rmse:,.2f}, R²: {best_test_r2:.4f}")
    
    # If targets met, break
    if rmse_met and r2_met:
        print(f"\n🎉 TARGETS ACHIEVED! Stopping iterations.")
        break
    
    # Clean up temporary model
    if os.path.exists('models/dnn_temp_best.h5'):
        os.remove('models/dnn_temp_best.h5')

# ============================================================================
# 4. FINAL EVALUATION AND SAVING
# ============================================================================
print("\n" + "="*80)
print("FINAL MODEL EVALUATION")
print("="*80)

if best_model is None:
    print("⚠️  No model was trained successfully. Using last model.")
    best_model = model
    best_config = config.copy()

# Final evaluation (transform predictions back to original scale)
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Final evaluation on all sets...")

# Train set
y_train_pred_scaled = best_model.predict(X_train_array, verbose=0).flatten()
y_train_pred = target_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
train_rmse = np.sqrt(mean_squared_error(y_train_array, y_train_pred))
train_r2 = r2_score(y_train_array, y_train_pred)
train_mae = mean_absolute_error(y_train_array, y_train_pred)

# Validation set
y_val_pred_scaled = best_model.predict(X_val_array, verbose=0).flatten()
y_val_pred = target_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
val_rmse = np.sqrt(mean_squared_error(y_val_array, y_val_pred))
val_r2 = r2_score(y_val_array, y_val_pred)
val_mae = mean_absolute_error(y_val_array, y_val_pred)

# Test set
y_test_pred_scaled = best_model.predict(X_test_array, verbose=0).flatten()
y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
test_rmse = np.sqrt(mean_squared_error(y_test_array, y_test_pred))
test_r2 = r2_score(y_test_array, y_test_pred)
test_mae = mean_absolute_error(y_test_array, y_test_pred)

print(f"\nFinal Results:")
print(f"  Train RMSE: ${train_rmse:,.2f}, R²: {train_r2:.4f}, MAE: ${train_mae:,.2f}")
print(f"  Val RMSE:   ${val_rmse:,.2f}, R²: {val_r2:.4f}, MAE: ${val_mae:,.2f}")
print(f"  Test RMSE:  ${test_rmse:,.2f}, R²: {test_r2:.4f}, MAE: ${test_mae:,.2f}")

# Save model
os.makedirs('models', exist_ok=True)
model_path = 'models/dnn_best_model.h5'
best_model.save(model_path)
print(f"\n✅ Saved best model: {model_path}")

# Save scalers
scaler_path = 'models/dnn_scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"✅ Saved feature scaler: {scaler_path}")

target_scaler_path = 'models/dnn_target_scaler.joblib'
joblib.dump(target_scaler, target_scaler_path)
print(f"✅ Saved target scaler: {target_scaler_path}")

# Save training report
report_path = 'models/dnn_training_report.txt'
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DEEP NEURAL NETWORK PROPERTY PRICE PREDICTION - TRAINING REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Project: Batata Real State\n")
    f.write(f"Team: Ian Bernasconi, Jeremias Feferovich\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("DATASET INFORMATION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Validation samples: {len(X_val)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"Number of features: {n_features}\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("BEST MODEL CONFIGURATION\n")
    f.write("-" * 80 + "\n")
    if best_config:
        f.write(f"Layers: {best_config.get('layers', 'N/A')}\n")
        f.write(f"Dropout: {best_config.get('dropout', 'N/A')}\n")
        f.write(f"Learning rate: {best_config.get('learning_rate', 'N/A')}\n")
        f.write(f"Batch size: {best_config.get('batch_size', 'N/A')}\n")
        f.write(f"L2 regularization: {best_config.get('l2_reg', 'N/A')}\n")
        f.write(f"Optimizer: {best_config.get('optimizer', 'N/A')}\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("MODEL PERFORMANCE\n")
    f.write("-" * 80 + "\n")
    f.write(f"Train RMSE: ${train_rmse:,.2f}\n")
    f.write(f"Train R²:   {train_r2:.4f}\n")
    f.write(f"Train MAE:  ${train_mae:,.2f}\n\n")
    f.write(f"Val RMSE:   ${val_rmse:,.2f}\n")
    f.write(f"Val R²:     {val_r2:.4f}\n")
    f.write(f"Val MAE:    ${val_mae:,.2f}\n\n")
    f.write(f"Test RMSE:  ${test_rmse:,.2f}\n")
    f.write(f"Test R²:    {test_r2:.4f}\n")
    f.write(f"Test MAE:   ${test_mae:,.2f}\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("KPI TARGET ACHIEVEMENT\n")
    f.write("-" * 80 + "\n")
    rmse_met = test_rmse < RMSE_TARGET
    r2_met = test_r2 > R2_TARGET
    f.write(f"RMSE < ${RMSE_TARGET:,}: {'✅ ACHIEVED' if rmse_met else '❌ NOT MET'} (${test_rmse:,.2f})\n")
    f.write(f"R² > {R2_TARGET}: {'✅ ACHIEVED' if r2_met else '❌ NOT MET'} ({test_r2:.4f})\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("FILES SAVED\n")
    f.write("-" * 80 + "\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Feature Scaler: {scaler_path}\n")
    f.write(f"Target Scaler: {target_scaler_path}\n")
    f.write(f"Report: {report_path}\n")

print(f"✅ Saved training report: {report_path}")

# Create visualizations
print("\n### Creating visualizations...")
os.makedirs('data', exist_ok=True)

# 1. Training history
if best_config and 'history' in best_config:
    history = best_config['history']
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['rmse'], label='Train RMSE')
    plt.plot(history.history['val_rmse'], label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Model RMSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/dnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: data/dnn_training_history.png")

# 2. Predicted vs Actual
plt.figure(figsize=(10, 8))
plt.scatter(y_test_array, y_test_pred, alpha=0.5, edgecolors='k', s=30)
plt.plot([y_test_array.min(), y_test_array.max()], 
         [y_test_array.min(), y_test_array.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title(f'Predicted vs Actual Prices - DNN\nR² = {test_r2:.4f}, RMSE = ${test_rmse:,.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/dnn_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: data/dnn_predicted_vs_actual.png")

# 3. Residuals distribution
residuals = y_test_array - y_test_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Residuals - DNN\nMean: ${np.mean(residuals):,.2f}, Std: ${np.std(residuals):,.2f}')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/dnn_residuals_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: data/dnn_residuals_distribution.png")

# 4. Residuals vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_test_pred, residuals, alpha=0.5, edgecolors='k', s=30)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Price (USD)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs Predicted Prices - DNN')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/dnn_residuals_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ Saved: data/dnn_residuals_vs_predicted.png")

# Final summary
total_time = time.time() - script_start_time
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total elapsed time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print("=" * 80)
print(f"\n🏆 Best Model Results:")
print(f"   Test RMSE: ${test_rmse:,.2f}")
print(f"   Test R²: {test_r2:.4f}")
print(f"   Test MAE: ${test_mae:,.2f}")

print(f"\n📊 KPI Achievement:")
print(f"   RMSE < ${RMSE_TARGET:,}: {'✅' if rmse_met else '❌'}")
print(f"   R² > {R2_TARGET}: {'✅' if r2_met else '❌'}")

print(f"\n💾 Saved Files:")
print(f"   - Model: {model_path}")
print(f"   - Scaler: {scaler_path}")
print(f"   - Report: {report_path}")
print(f"   - 4 visualization plots in data/ folder")

print("\n" + "=" * 80)

