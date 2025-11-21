from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import os
from feature_engineering import compute_features

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'segmented')
METADATA_PATH = os.path.join(MODELS_DIR, 'metadata.json')

with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

models = {}
for seg in metadata['segments']:
    seg_name = seg['name']
    model_path = os.path.join(MODELS_DIR, f"{seg_name}_ensemble.joblib")
    models[seg_name] = joblib.load(model_path)
    print(f"Loaded {seg_name} model")

FEATURE_COLS = metadata['features']

PROPERTY_FIELDS = [
    ('area', 'Area (m²)', 'number'),
    ('bedrooms', 'Bedrooms', 'number'),
    ('bathrooms', 'Bathrooms', 'number'),
    ('latitude', 'Latitude', 'number'),
    ('longitude', 'Longitude', 'number'),
    ('property_type', 'Property Type', 'select'),
    ('balcony_count', 'Balcony Count', 'number'),
    ('segment', 'Price Segment', 'select'),
]

SEGMENTS = [seg['name'] for seg in metadata['segments']]
PROPERTY_TYPES = ['departamento', 'casa']

SEGMENT_NAMES = {
    'budget': 'Económico',
    'mid_range': 'Medio',
    'premium': 'Premium'
}

# Información de segmentos con rangos de precio
SEGMENT_INFO = {}
for seg in metadata['segments']:
    seg_name = seg['name']
    price_min = seg['price_min']
    price_max = seg['price_max']
    name_es = SEGMENT_NAMES.get(seg_name, seg_name)
    price_range = f"${price_min:,.0f} - ${price_max:,.0f}"
    SEGMENT_INFO[seg_name] = {
        'name_es': name_es,
        'price_min': price_min,
        'price_max': price_max,
        'price_range': price_range,
        'display': f"{name_es} ({price_range})"
    }

def predict_price(features_dict, segment):
    if segment not in models:
        raise ValueError(f"Segmento desconocido: {segment}")
    
    model_data = models[segment]
    
    feature_vector = [features_dict[col] for col in FEATURE_COLS]
    X = pd.DataFrame([feature_vector], columns=FEATURE_COLS)
    
    X = X.fillna(X.median())
    
    X_scaled = model_data['scaler'].transform(X)
    
    xgb_pred_log = model_data['xgb_model'].predict(X_scaled)[0]
    lgbm_pred_log = model_data['lgbm_model'].predict(X_scaled)[0]
    
    pred_log = (
        model_data['weight_xgb'] * xgb_pred_log +
        model_data['weight_lgbm'] * lgbm_pred_log
    )
    
    price = np.expm1(pred_log)
    
    return float(price)


# Ruta para formulario
@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    valores = {}
    if request.method == "POST":
        try:
            area = float(request.form.get('area', 0))
            bedrooms = int(request.form.get('bedrooms', 0))
            bathrooms = float(request.form.get('bathrooms', 0))
            latitude = float(request.form.get('latitude', 0))
            longitude = float(request.form.get('longitude', 0))
            property_type = request.form.get('property_type', 'departamento')
            balcony_count = int(request.form.get('balcony_count', 0))
            segment = request.form.get('segment', 'mid_range')
            
            # Store values for form
            valores = {
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'latitude': latitude,
                'longitude': longitude,
                'property_type': property_type,
                'balcony_count': balcony_count,
                'segment': segment,
            }
            
            features = compute_features(
                area=area,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                latitude=latitude,
                longitude=longitude,
                property_type=property_type,
                balcony_count=balcony_count
            )
            
            price = predict_price(features, segment)
            resultado = f"${price:,.0f} USD"
            
        except Exception as e:
            resultado = f"Error: {str(e)}"
    
    return render_template(
        "index.html",
        fields=PROPERTY_FIELDS,
        segments=SEGMENTS,
        segment_info=SEGMENT_INFO,
        property_types=PROPERTY_TYPES,
        resultado=resultado,
        valores=valores
    )


# Ruta para llamar API
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        
        required_fields = ['area', 'bedrooms', 'bathrooms', 'latitude', 'longitude']
        field_names_es = {
            'area': 'área',
            'bedrooms': 'dormitorios',
            'bathrooms': 'baños',
            'latitude': 'latitud',
            'longitude': 'longitud'
        }
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Campo requerido faltante: {field_names_es.get(field, field)}"}), 400
        
        # Get values with defaults
        area = float(data['area'])
        bedrooms = int(data['bedrooms'])
        bathrooms = float(data['bathrooms'])
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        property_type = data.get('property_type', 'departamento')
        balcony_count = int(data.get('balcony_count', 0))
        segment = data.get('segment', 'mid_range')
        
        # Validate segment
        if segment not in models:
            return jsonify({"error": f"Segmento inválido: {segment}. Debe ser uno de: {', '.join(SEGMENTS)}"}), 400
        
        # Compute all features
        features = compute_features(
            area=area,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            latitude=latitude,
            longitude=longitude,
            property_type=property_type,
            balcony_count=balcony_count
        )
        
        # Predict price
        price = predict_price(features, segment)
        
        # Get segment info
        seg_info = next(s for s in metadata['segments'] if s['name'] == segment)
        
        return jsonify({
            "prediction": price,
            "segment": segment,
            "segment_description": seg_info['description'],
            "price_range": f"${seg_info['price_min']:,}-${seg_info['price_max']:,}"
        })
        
    except ValueError as e:
        return jsonify({"error": f"Entrada inválida: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error del servidor: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
