import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "sepal_length": 6.2,
    "sepal_width": 2.9,
    "petal_length": 4.3,
    "petal_width": 1.3
}

response = requests.post(url, json=data)
print(response.json())
