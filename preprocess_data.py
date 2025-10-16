# -*- coding: utf-8 -*-
"""
Data Preprocessing Script for Properati Dataset
Adapted from Colab notebook to run locally

Equipo: Ian Bernasconi, Jeremias Feferovich
Proyecto: Batata Real State
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime

# Configuraciones de estilo
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("=" * 80)
print("PREPROCESSING PROPERATI DATASET")
print("=" * 80)

# Load data from local CSV file
print("\n### 🧾 Loading data...")
csv_path = '/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/properati_dataset_v4_20250924_211821.csv'
df = pd.read_csv(csv_path)
print(f"Initial shape: {df.shape}")
print(f"\nFirst rows:\n{df.head()}")

# Data info
print("\n### 🔍 Dataset Information")
df.info()

# Drop rows with NaN in price (target variable)
print("\n### Removing rows with NaN in 'price'...")
df = df.dropna(subset=["price"])

# Check for duplicates by property_id
print("\n### Checking for duplicates...")
print(f"Shape before removing duplicates: {df.shape}")
df = df.drop_duplicates(subset="property_id", keep="first")
print(f"Shape after removing duplicates: {df.shape}")

# Numeric distributions
print("\n### 🔢 Analyzing numeric variable distributions...")
cols = ["price", "surface_total", "bathrooms", "bedrooms", "area", "balcony_count"]
df_numeric = df[cols]

fig = df_numeric.hist(bins=10, figsize=(15, 10), layout=(4, 2))
plt.suptitle('Distribución de variables numéricas')
plt.tight_layout()
plt.savefig('distributions.png')
print("Saved: distributions.png")
plt.close()

# Boxplots for outlier detection
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
columnas = df_numeric.columns
for i, ax in enumerate(axes.flat):
    sns.boxplot(y=df[columnas[i]], ax=ax)
    ax.set_title(columnas[i])
plt.suptitle('Boxplots de variables numéricas', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('boxplots.png')
print("Saved: boxplots.png")
plt.close()

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlación")
plt.savefig('correlation_matrix.png')
print("Saved: correlation_matrix.png")
plt.close()

# Surface vs price scatter
plt.figure(figsize=(10, 6))
plt.scatter(df["surface_total"], df["price"], alpha=0.3, edgecolor="k", s=20)
plt.title("Distribución del precio según la superficie")
plt.xlabel("Superficie (m²)")
plt.ylabel("Precio")
plt.xticks(rotation=45)
plt.locator_params(axis="x", nbins=10)
plt.savefig('surface_vs_price_before.png')
print("Saved: surface_vs_price_before.png")
plt.close()

# ### 🧪 TRANSFORMATIONS
print("\n" + "=" * 80)
print("### 🧪 STARTING TRANSFORMATIONS")
print("=" * 80)

# Drop irrelevant columns
print("\n### Dropping irrelevant columns...")
cols_to_drop = [
    'amenities', 'rooms', 'expenses', 'orientation', 'floor_number', 'parking_spaces',
    'property_status', 'surface_covered', 'total_floors',
    'agency_name', 'description', 'detail_url', 'photo_count',
    'published_date', 'publisher', 'title',
    'page_number', 'property_id', 'scraping_date',
    'full_address', 'neighborhood', 'neighborhood_slug', 'scraped_neighborhood',  # Fixed: added comma
    'surface_total'  # keeping only 'area'
]

df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')
print(f"Total columns after dropping: {df.shape[1]}")

# Handle missing values
print("\n### Handling missing values...")
cols_imputar = ['currency']

print("🔍 Nulls before imputation:")
for col in cols_imputar:
    print(f" - {col}: {df[col].isna().sum()} nulls")

df['currency'] = df['currency'].fillna('USD')

print("\nNulls after imputation:")
for col in cols_imputar:
    print(f" - {col}: {df[col].isna().sum()} remaining nulls")

# Remove rows with nulls in key columns
cols_eliminar_si_nulos = ['area', 'bathrooms', 'bedrooms', 'currency']

print("\n🔍 Nulls before removing rows:")
for col in cols_eliminar_si_nulos:
    print(f" - {col}: {df[col].isna().sum()} nulls")

before_drop = df.shape[0]
df = df.dropna(subset=cols_eliminar_si_nulos)
after_drop = df.shape[0]

print(f"\n🧹 Rows removed: {before_drop - after_drop}")
print(f"✅ Total remaining rows: {after_drop}")

# Remove rows with 0 values in key columns
print("\n### Removing rows with 0 values in key columns...")
cols_sin_ceros = ['area', 'bathrooms', 'bedrooms', 'price']

print("🔍 Zero values before removal:")
for col in cols_sin_ceros:
    print(f" - {col}: {(df[col] == 0).sum()} rows with 0")

before_drop = df.shape[0]
df = df[~(df[cols_sin_ceros] == 0).any(axis=1)]
after_drop = df.shape[0]

print(f"\nRows removed: {before_drop - after_drop}")
print(f"Total remaining rows: {after_drop}")

# ### OUTLIER TREATMENT
print("\n" + "=" * 80)
print("### 🧊 OUTLIER TREATMENT")
print("=" * 80)

numeric_cols = ['price', 'area', 'bathrooms', 'bedrooms', 'balcony_count']

print("\nBefore outlier treatment:\n")
for col in numeric_cols:
    print(f"→ {col}")
    print(f"   Mean: {df[col].mean():,.2f}")
    print(f"   Std: {df[col].std():,.2f}")
    print(f"   Min: {df[col].min():,.2f}")
    print(f"   Max: {df[col].max():,.2f}\n")

# Price: minimum threshold
min_valid_price = 10000
before_min_filter = len(df)
df = df[df['price'] >= min_valid_price]
after_min_filter = len(df)

print(f"\n✂️ Price filter applied (< {min_valid_price:,} USD)")
print(f"Rows removed: {before_min_filter - after_min_filter}")

# Price: maximum threshold
q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr

print("\n📊 High price analysis:")
print(f" - Q3 (75th percentile): {q3:,.2f}")
print(f" - Upper limit (Q3 + 1.5*IQR): {upper_limit:,.2f}")
print(f" - Current max: {df['price'].max():,.2f}")

top_prices = df.nlargest(10, 'price')[['price', 'area', 'bedrooms', 'bathrooms', 'location', 'currency']]
print("\n🏠 Top 10 most expensive properties:\n")
print(top_prices)

max_valid_price = 12000000
before = len(df)
df = df[df['price'] <= max_valid_price]
after = len(df)

print(f"\n✂️ Removed {before - after} properties with price > {max_valid_price:,} USD")
print(f"New max price: {df['price'].max():,.2f} USD")

# Area: minimum threshold
min_area = 30
menos_30 = (df['area'] < 30).sum()
total = len(df)
porcentaje = (menos_30 / total) * 100
print(f"\n📏 Properties with area < 30 m²: {menos_30} ({porcentaje:.2f}% of total)")

df = df[(df['area'] >= min_area)]
print(f"✂️ Area filter applied | Final range: {df['area'].min():,.2f} - {df['area'].max():,.2f}")

# Bathrooms: maximum threshold
max_bathrooms = 8
before_drop = len(df)
df = df[df['bathrooms'] <= max_bathrooms]
after_drop = len(df)
print(f"\n🚿 Removed {before_drop - after_drop} properties with > {max_bathrooms} bathrooms")

# Bedrooms: maximum threshold
max_bedrooms = 11
before_drop = len(df)
df = df[df['bedrooms'] <= max_bedrooms]
after_drop = len(df)
print(f"🛏️  Removed {before_drop - after_drop} properties with > {max_bedrooms} bedrooms")

print(f"\n✅ Total records after transformations: {len(df)}")

# Plot area vs price after cleaning
plt.figure(figsize=(10, 6))
plt.scatter(df["area"], df["price"], alpha=0.3, edgecolor="k", s=20)
plt.title("Distribución del precio según el área (después de limpieza)")
plt.xlabel("Area (m²)")
plt.ylabel("Precio")
plt.xticks(rotation=45)
plt.locator_params(axis="x", nbins=10)
plt.savefig('area_vs_price_after.png')
print("\nSaved: area_vs_price_after.png")
plt.close()

# ### TYPE CONVERSION
print("\n" + "=" * 80)
print("### 🔄 TYPE CONVERSION")
print("=" * 80)

# Parse coordinates
def parse_coordinates(coord):
    if isinstance(coord, str):
        match = re.findall(r'-?\d+\.\d+', coord)
        if len(match) == 2:
            return float(match[0]), float(match[1])
    return (np.nan, np.nan)

print("\nParsing coordinates...")
print("Before parsing:")
print(df[['coordinates', 'latitude', 'longitude']].head(5))

df[['latitude', 'longitude']] = df['coordinates'].apply(
    lambda x: pd.Series(parse_coordinates(x))
)

print("\nAfter parsing:")
print(f" - latitude nulls: {df['latitude'].isna().sum()}")
print(f" - longitude nulls: {df['longitude'].isna().sum()}")
print(df[['coordinates', 'latitude', 'longitude']].head(5))

df.drop(columns=['coordinates'], inplace=True)

# ### TEXT NORMALIZATION
print("\n### 📝 Text normalization...")
df['location'] = df['location'].str.lower().str.strip()
df['currency'] = df['currency'].str.upper().str.strip()
df['property_type'] = df['property_type'].str.lower().str.strip()

# ### NEW VARIABLE GENERATION
print("\n### ➕ Generating new variables...")

# Labels
df['labels'] = df['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
df['is_destacado'] = df['labels'].apply(lambda x: 'DESTACADO' in x)
df['is_nuevo'] = df['labels'].apply(lambda x: 'NUEVO' in x)
df.drop(columns=['labels'], inplace=True)

# Property type
df['property_type'] = df['property_type'].str.lower().str.strip()
df['is_departamento'] = df['property_type'].apply(lambda x: 1 if 'departamento' in x else 0)
df['is_casa'] = df['property_type'].apply(lambda x: 1 if 'casa' in x else 0)

print("\nNew binary columns created:")
print(df[['is_departamento', 'is_casa', 'is_destacado', 'is_nuevo']].head(5))

df.drop(columns=['property_type'], inplace=True)

# ### NORMALIZATION AND STANDARDIZATION
print("\n### 📊 Normalization and standardization...")

scaler_minmax = MinMaxScaler()
df_norm = df.copy()
col_norm = ['area']
df_norm[col_norm] = scaler_minmax.fit_transform(df_norm[col_norm])

print("\nNormalized 'area' column (first 5 rows):")
print(df_norm[['area']].head())

# ### SAVE PREPROCESSED DATA
print("\n" + "=" * 80)
print("### 💾 SAVING PREPROCESSED DATA")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save both versions (original and normalized)
output_path = f'/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/properati_preprocessed_{timestamp}.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Saved preprocessed data (original scale): {output_path}")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

output_path_norm = f'/home/jeremias.feferovich/Desktop/ITBA/CDA/CDA-TP/properati_preprocessed_normalized_{timestamp}.csv'
df_norm.to_csv(output_path_norm, index=False)
print(f"\n✅ Saved preprocessed data (normalized area): {output_path_norm}")
print(f"   Shape: {df_norm.shape}")
print(f"   Columns: {list(df_norm.columns)}")

# Print summary statistics
print("\n" + "=" * 80)
print("### 📈 FINAL SUMMARY STATISTICS")
print("=" * 80)
print("\nFinal dataset info:")
print(df.info())
print("\nNumeric columns statistics:")
print(df.describe())

print("\n" + "=" * 80)
print("✅ PREPROCESSING COMPLETE!")
print("=" * 80)

