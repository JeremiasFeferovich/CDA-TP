from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo
modelo = joblib.load("models/modelo_iris_mejor.joblib")
clases = ['setosa', 'versicolor', 'virginica']

# Variables para el formulario (con nombres largos)
columnas_modelo = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]

# Variables para la API (con nombres cortos)
columnas_api = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width'
]

# Ruta para formulario
@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    valores = {}
    if request.method == "POST":
        try:
            valores = {col: float(request.form[col]) for col in columnas_modelo}
            df = pd.DataFrame([valores.values()], columns=columnas_modelo)
            pred = modelo.predict(df)[0]
            resultado = clases[pred]
        except Exception as e:
            resultado = f"Error: {str(e)}"
    return render_template("index.html", columnas=columnas_modelo, resultado=resultado, valores=valores)

# Ruta para llamar API
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        # Reordenamos los valores según el orden esperado
        valores = [data[col] for col in columnas_api]
        df = pd.DataFrame([valores], columns=columnas_modelo)
        pred = modelo.predict(df)[0]
        return jsonify({"prediction": clases[pred]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
