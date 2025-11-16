#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process bus stops data and create Python constant for train_model.py

This script reads the downloaded bus stop coordinates and generates
a Python constant that can be imported or copied into train_model.py
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("PROCESSING BUS STOPS DATA")
print("=" * 80)

# Load bus stop data
input_file = 'data/bus_stops_coordinates.csv'
print(f"\n### Loading data from: {input_file}")

df = pd.read_csv(input_file)
print(f"✅ Loaded {len(df)} bus stops")

# Remove any rows with missing coordinates
print("\n### Cleaning data...")
before = len(df)
df = df.dropna(subset=['latitude', 'longitude'])
after = len(df)
print(f"Removed {before - after} stops with missing coordinates")

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"Removed {before - after} duplicate stops")
print(f"✅ Final count: {len(df)} unique bus stops")

# Statistics
print(f"\n### Statistics:")
print(f"  Latitude range:  {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
print(f"  Longitude range: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
print(f"  Mean latitude:   {df['latitude'].mean():.6f}")
print(f"  Mean longitude:  {df['longitude'].mean():.6f}")

# Generate Python constant
print("\n### Generating Python constant...")

# Create the BUS_STOPS constant as a list of tuples
output_py = 'data/bus_stops_constant.py'
with open(output_py, 'w') as f:
    f.write('# -*- coding: utf-8 -*-\n')
    f.write('"""\n')
    f.write('Bus stops coordinates for Buenos Aires, Argentina\n')
    f.write('Source: OpenStreetMap via Overpass API\n')
    f.write(f'Total stops: {len(df)}\n')
    f.write('Format: (latitude, longitude)\n')
    f.write('"""\n\n')
    f.write('BUS_STOPS = [\n')
    
    for idx, row in df.iterrows():
        f.write(f'    ({row["latitude"]:.6f}, {row["longitude"]:.6f}),\n')
        
        # Add progress indicator for large datasets
        if (idx + 1) % 1000 == 0:
            f.write(f'    # ... {idx + 1} stops so far ...\n')
    
    f.write(']\n')

print(f"✅ Saved Python constant to: {output_py}")
print(f"   File contains {len(df)} bus stop coordinates")

# Also save a compact numpy version for faster loading
print("\n### Creating compact numpy version...")
coords_array = df[['latitude', 'longitude']].values
np.save('data/bus_stops_coords.npy', coords_array)
print(f"✅ Saved numpy array to: data/bus_stops_coords.npy")
print(f"   Shape: {coords_array.shape}")

# Create a summary
summary_file = 'data/bus_stops_summary.txt'
with open(summary_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("BUS STOPS DATA SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Source: OpenStreetMap (Overpass API)\n")
    f.write(f"Region: Buenos Aires, Argentina\n")
    f.write(f"Downloaded: {pd.Timestamp.now()}\n\n")
    f.write(f"Total bus stops: {len(df)}\n\n")
    f.write(f"Coordinate ranges:\n")
    f.write(f"  Latitude:  {df['latitude'].min():.6f} to {df['latitude'].max():.6f}\n")
    f.write(f"  Longitude: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}\n\n")
    f.write(f"Files generated:\n")
    f.write(f"  - {output_py} (Python constant)\n")
    f.write(f"  - data/bus_stops_coords.npy (NumPy array)\n")
    f.write(f"  - {input_file} (CSV source)\n\n")
    f.write("=" * 80 + "\n")

print(f"✅ Saved summary to: {summary_file}")

print("\n" + "=" * 80)
print("✅ PROCESSING COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("1. Review the generated files in the data/ directory")
print("2. Use the BUS_STOPS constant in train_model.py")
print("3. Add bus stop density feature to the model")


