import requests
import json

url = "http://127.0.0.1:5000/predict"

data = {
    "area": 80.0,
    "bedrooms": 2,
    "bathrooms": 1.5,
    "latitude": -34.589722,
    "longitude": -58.410833,
    "property_type": "departamento",
    "balcony_count": 1,
    "segment": "mid_range"
}

print("Sending request to API...")
print(f"Data: {json.dumps(data, indent=2)}")
print()

response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
