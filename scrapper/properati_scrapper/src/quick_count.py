#!/usr/bin/env python3
"""
Quick script to count unique property IDs in a JSON file
Usage: python3 quick_count.py <filename.json>
"""

import json
import sys
from pathlib import Path

def quick_count(file_path):
    """Quick count of unique IDs in a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        total = len(data)
        unique_ids = len(set(prop.get('property_id', '') for prop in data))
        duplicates = total - unique_ids
        efficiency = (unique_ids / total * 100) if total > 0 else 0
        
        print(f"📊 {Path(file_path).name}")
        print(f"   Total: {total} | Únicos: {unique_ids} | Duplicados: {duplicates}")
        print(f"   Eficiencia: {efficiency:.1f}%")
        
        if efficiency == 100:
            print("   ✅ Perfecto!")
        elif efficiency >= 90:
            print("   👍 Muy bueno")
        elif efficiency >= 70:
            print("   ⚠️  Moderado")
        else:
            print("   ❌ Muchos duplicados")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 quick_count.py <filename.json>")
        sys.exit(1)
    
    quick_count(sys.argv[1])
