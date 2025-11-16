#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download bus stops data from OpenStreetMap via Overpass API
Buenos Aires boundaries: approximately -34.7 to -34.5 lat, -58.5 to -58.3 lon
"""

import requests
import json
import pandas as pd
import time

print("=" * 80)
print("DOWNLOADING BUS STOPS DATA FROM OPENSTREETMAP")
print("Buenos Aires, Argentina")
print("=" * 80)

# Overpass API endpoint
overpass_url = "http://overpass-api.de/api/interpreter"

# Query for bus stops in Buenos Aires
# Using bbox: south, west, north, east
overpass_query = """
[out:json][timeout:90];
(
  node["highway"="bus_stop"](-34.75,-58.55,-34.50,-58.30);
  node["public_transport"="platform"]["bus"="yes"](-34.75,-58.55,-34.50,-58.30);
  node["public_transport"="stop_position"]["bus"="yes"](-34.75,-58.55,-34.50,-58.30);
);
out body;
"""

print("\n### Querying OpenStreetMap Overpass API...")
print("This may take 30-60 seconds...")

try:
    response = requests.post(
        overpass_url,
        data={'data': overpass_query},
        timeout=120
    )
    
    if response.status_code != 200:
        print(f"❌ API returned status code: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        exit(1)
    
    data = response.json()
    
    if 'elements' not in data:
        print(f"❌ Unexpected API response format")
        print(f"Response: {json.dumps(data)[:500]}")
        exit(1)
    
    bus_stops = data['elements']
    print(f"✅ Retrieved {len(bus_stops)} bus stops from OpenStreetMap")
    
    # Extract coordinates
    bus_stop_data = []
    for stop in bus_stops:
        if 'lat' in stop and 'lon' in stop:
            bus_stop_data.append({
                'id': stop.get('id'),
                'latitude': stop['lat'],
                'longitude': stop['lon'],
                'name': stop.get('tags', {}).get('name', ''),
                'ref': stop.get('tags', {}).get('ref', ''),
                'network': stop.get('tags', {}).get('network', ''),
                'operator': stop.get('tags', {}).get('operator', '')
            })
    
    # Create DataFrame
    df = pd.DataFrame(bus_stop_data)
    
    print(f"\n### Processing data...")
    print(f"Total bus stops with coordinates: {len(df)}")
    print(f"\nFirst few stops:")
    print(df.head(10))
    
    print(f"\nCoordinate ranges:")
    print(f"  Latitude:  {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
    print(f"  Longitude: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
    
    # Save to CSV
    output_file = 'data/colectivos_paradas_osm.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")
    
    # Also save a simplified version with just coordinates
    df_coords = df[['latitude', 'longitude']].copy()
    df_coords = df_coords.drop_duplicates()
    output_coords = 'data/bus_stops_coordinates.csv'
    df_coords.to_csv(output_coords, index=False)
    print(f"✅ Saved coordinates to: {output_coords}")
    print(f"   Unique locations: {len(df_coords)}")
    
    print("\n" + "=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 80)
    
except requests.exceptions.Timeout:
    print("❌ Request timed out. The Overpass API might be busy.")
    print("Please try again in a few minutes.")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


