#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download bus stops data from Buenos Aires Open Data Portal
"""

import requests
import json
import pandas as pd
import sys

print("=" * 80)
print("DOWNLOADING BUS STOPS DATA FROM BUENOS AIRES OPEN DATA PORTAL")
print("=" * 80)

# Try different approaches to get the data

# Approach 1: Direct CSV download
print("\n### Approach 1: Trying direct CSV download...")
csv_url = "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/transporte/colectivos-paradas/colectivos-paradas.csv"
try:
    response = requests.get(csv_url, timeout=30)
    if response.status_code == 200 and len(response.content) > 1000:
        with open('data/colectivos_paradas.csv', 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded {len(response.content)} bytes")
        df = pd.read_csv('data/colectivos_paradas.csv')
        print(f"✅ Loaded {len(df)} bus stops")
        print(f"Columns: {list(df.columns)}")
        print(df.head())
        sys.exit(0)
except Exception as e:
    print(f"❌ Failed: {e}")

# Approach 2: CKAN API with pagination
print("\n### Approach 2: Trying CKAN API with pagination...")
base_url = "https://data.buenosaires.gob.ar/api/3/action/datastore_search"
resource_id = "f1625d17-c447-4d8c-a63e-b16d20644eb2"

all_records = []
offset = 0
limit = 1000

try:
    while True:
        params = {
            'resource_id': resource_id,
            'limit': limit,
            'offset': offset
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ API returned status code: {response.status_code}")
            break
        
        data = response.json()
        
        if not data.get('success'):
            print(f"❌ API request not successful")
            break
        
        records = data['result']['records']
        
        if not records:
            break
        
        all_records.extend(records)
        print(f"  Downloaded {len(all_records)} records so far...")
        
        offset += limit
        
        # Stop if we've got all records
        if len(records) < limit:
            break
        
        # Safety limit
        if offset > 50000:
            break
    
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv('data/colectivos_paradas.csv', index=False)
        print(f"\n✅ Downloaded {len(all_records)} bus stops via API")
        print(f"Columns: {list(df.columns)}")
        print(df.head())
        sys.exit(0)
    
except Exception as e:
    print(f"❌ API approach failed: {e}")

# Approach 3: Try alternative URLs
print("\n### Approach 3: Trying alternative data sources...")
alternative_urls = [
    "https://data.buenosaires.gob.ar/dataset/1c8f7b30-a3df-4a39-81e8-e39ae29e2e4b/resource/f1625d17-c447-4d8c-a63e-b16d20644eb2/download/colectivos-paradas.csv",
    "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/colectivos-paradas/colectivos-paradas.csv",
]

for url in alternative_urls:
    try:
        print(f"\nTrying: {url}")
        response = requests.get(url, timeout=30, allow_redirects=True)
        if response.status_code == 200 and len(response.content) > 1000:
            with open('data/colectivos_paradas.csv', 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded {len(response.content)} bytes")
            df = pd.read_csv('data/colectivos_paradas.csv')
            print(f"✅ Loaded {len(df)} bus stops")
            print(f"Columns: {list(df.columns)}")
            print(df.head())
            sys.exit(0)
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n" + "=" * 80)
print("❌ ALL DOWNLOAD APPROACHES FAILED")
print("=" * 80)
print("\nPlease manually download the dataset from:")
print("https://data.buenosaires.gob.ar/dataset/colectivos-paradas")
print("And save it as: data/colectivos_paradas.csv")
sys.exit(1)


