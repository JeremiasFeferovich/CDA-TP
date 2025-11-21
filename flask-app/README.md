
# 🧠 Iris Model Web App

Este proyecto demuestra cómo desplegar un modelo de Machine Learning (clasificación de flores *Iris*) usando Flask, con una interfaz web y una API REST.

---

## 🚀 Estructura del proyecto

```
├── app.py                  # App Flask principal (formulario + API)
├── test_api.py             # Script para probar la API con requests
├── models/
│   └── modelo_iris_mejor.joblib  # Modelo entrenado y guardado con joblib
├── templates/
│   └── index.html          # Formulario web
├── static/
│   └── style.css           # Estilo visual para el formulario
```

---

## 🧪 ¿Qué incluye?

| Interfaz         | Ruta           | Uso                                 |
|------------------|----------------|--------------------------------------|
| Formulario web   | `/`            | Ingreso manual de valores por HTML   |
| API REST         | `/predict`     | Recepción de JSON, devuelve predicción |
| Script de prueba | `test_api.py`  | Envía un JSON y recibe predicción    |

---

## 📦 Requisitos

Instalar las dependencias necesarias:

```
pip install flask pandas scikit-learn joblib
```

---

## 🧑‍💻 Ejecución local

1. Asegurate de tener el modelo guardado en `models/modelo_iris_mejor.joblib`.
2. Ejecutá la app Flask:

```
python app.py
```

3. Accedé al navegador en [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📡 Uso de la API

### Ejemplo de request desde Python (`test_api.py`)

```python
import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "sepal_length": 6.2,
    "sepal_width": 2.9,
    "petal_length": 4.3,
    "petal_width": 1.3
}

response = requests.post(url, json=data)
print(response.json())  # {'prediction': 'versicolor'}
```

---

## 🎓 Pensado para enseñanza

Esta app muestra cómo llevar un modelo desde un notebook hasta una aplicación real de forma incremental:

1. [Entrenamiento y guardado en notebook](https://colab.research.google.com/drive/1obfnUN9vHLPNcC6U_JmYhvHBTwBN7Af5).
2. [Predicción en notebook](https://colab.research.google.com/drive/1bTmAV8yLe9W3GKpjs4FSvBW9X2uN0c36).
3. [Prototipo visual (Streamlit)](https://iris-svm.streamlit.app/).
4. Web real (Flask con formulario).
5. API REST (Flask + JSON).
6. Prueba externa (requests) (`test_api.py`).

---

## ✨ Autor

Desarrollado para el curso de Ciencia de Datos Aplicada – ITBA.


---

## 🔁 Instalación rápida desde cero

```bash
# Cloná el repositorio
git clone https://github.com/usuario/proyecto-iris-flask.git
cd proyecto-iris-flask

# Instalá las dependencias
pip install -r requirements.txt

# Ejecutá la app
python app.py
```

La app estará disponible en [http://127.0.0.1:5000](http://127.0.0.1:5000)
