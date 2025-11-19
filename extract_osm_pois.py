#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract Points of Interest from OpenStreetMap for Buenos Aires
Downloads parks, schools, hospitals, restaurants, and other amenities
"""

import requests
import json
import numpy as np
import time

# Buenos Aires bounding box (approximate)
# Format: [min_lat, min_lon, max_lat, max_lon]
BUENOS_AIRES_BBOX = [-34.75, -58.55, -34.50, -58.30]

# Overpass API endpoint
OVERPASS_URL = "http://overpass-api.de/api/interpreter"

def query_overpass(query, description=""):
    """Query Overpass API with rate limiting"""
    print(f"Querying OpenStreetMap for {description}...")

    try:
        response = requests.post(OVERPASS_URL, data={'data': query}, timeout=180)
        response.raise_for_status()
        data = response.json()
        print(f"  ✅ Found {len(data.get('elements', []))} items")
        return data
    except requests.exceptions.Timeout:
        print(f"  ⚠️  Timeout - query took too long")
        return {'elements': []}
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  Error: {e}")
        return {'elements': []}

def extract_coordinates(elements):
    """Extract coordinates from Overpass API response"""
    coords = []
    for element in elements:
        if element['type'] == 'node':
            coords.append((element['lat'], element['lon']))
        elif element['type'] == 'way' and 'center' in element:
            coords.append((element['center']['lat'], element['center']['lon']))
        elif element['type'] == 'relation' and 'center' in element:
            coords.append((element['center']['lat'], element['center']['lon']))
    return coords

# Build Overpass QL queries
bbox_str = f"{BUENOS_AIRES_BBOX[0]},{BUENOS_AIRES_BBOX[1]},{BUENOS_AIRES_BBOX[2]},{BUENOS_AIRES_BBOX[3]}"

queries = {
    'parks': f"""
        [out:json][timeout:180];
        (
          node["leisure"="park"]({bbox_str});
          way["leisure"="park"]({bbox_str});
          relation["leisure"="park"]({bbox_str});
        );
        out center;
    """,

    'schools': f"""
        [out:json][timeout:180];
        (
          node["amenity"="school"]({bbox_str});
          way["amenity"="school"]({bbox_str});
          relation["amenity"="school"]({bbox_str});
          node["amenity"="university"]({bbox_str});
          way["amenity"="university"]({bbox_str});
          relation["amenity"="university"]({bbox_str});
        );
        out center;
    """,

    'hospitals': f"""
        [out:json][timeout:180];
        (
          node["amenity"="hospital"]({bbox_str});
          way["amenity"="hospital"]({bbox_str});
          relation["amenity"="hospital"]({bbox_str});
          node["amenity"="clinic"]({bbox_str});
          way["amenity"="clinic"]({bbox_str});
        );
        out center;
    """,

    'supermarkets': f"""
        [out:json][timeout:180];
        (
          node["shop"="supermarket"]({bbox_str});
          way["shop"="supermarket"]({bbox_str});
          node["shop"="mall"]({bbox_str});
          way["shop"="mall"]({bbox_str});
        );
        out center;
    """,

    'restaurants': f"""
        [out:json][timeout:180];
        (
          node["amenity"="restaurant"]({bbox_str});
          way["amenity"="restaurant"]({bbox_str});
          node["amenity"="cafe"]({bbox_str});
          way["amenity"="cafe"]({bbox_str});
          node["amenity"="bar"]({bbox_str});
          way["amenity"="bar"]({bbox_str});
        );
        out center;
    """,

    'green_spaces': f"""
        [out:json][timeout:180];
        (
          node["leisure"="garden"]({bbox_str});
          way["leisure"="garden"]({bbox_str});
          node["landuse"="recreation_ground"]({bbox_str});
          way["landuse"="recreation_ground"]({bbox_str});
        );
        out center;
    """
}

print("=" * 80)
print("EXTRACTING POINTS OF INTEREST FROM OPENSTREETMAP")
print("=" * 80)
print(f"\nBuenos Aires bounding box: {BUENOS_AIRES_BBOX}")
print("This will take several minutes due to API rate limiting...\n")

# Extract all POIs
all_pois = {}

for poi_type, query in queries.items():
    data = query_overpass(query, poi_type)
    coords = extract_coordinates(data.get('elements', []))
    all_pois[poi_type] = coords

    # Save to file
    if coords:
        coords_array = np.array(coords)
        filename = f'data/{poi_type}_coords.npy'
        np.save(filename, coords_array)
        print(f"  💾 Saved to {filename}\n")
    else:
        print(f"  ⚠️  No data to save\n")

    # Rate limiting - be nice to Overpass API
    if poi_type != list(queries.keys())[-1]:  # Don't wait after last query
        print("  ⏳ Waiting 3 seconds (API rate limiting)...")
        time.sleep(3)

# Summary
print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
print("\nSummary:")
for poi_type, coords in all_pois.items():
    print(f"  {poi_type:15s}: {len(coords):5,} locations")

total_pois = sum(len(coords) for coords in all_pois.values())
print(f"\nTotal POIs extracted: {total_pois:,}")
print("\nFiles saved to data/ directory")
