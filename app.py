from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the model and label encoder
try:
    model = joblib.load("model.joblib")
    le = joblib.load("label_encoder.joblib")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    le = None

# Define valid ranges for inputs based on the dataset statistics
INPUT_RANGES = {
    'N': (0, 140),      # Nitrogen
    'P': (0, 145),      # Phosphorus
    'K': (0, 205),      # Potassium
    'temperature': (8.0, 44.0),  # Temperature in Celsius
    'humidity': (14.0, 100.0),   # Humidity percentage
    'ph': (3.5, 10.0),  # pH level
    'rainfall': (20.0, 300.0)    # Rainfall in mm
}

@app.route('/')
def home():
    return render_template('index.html')

def validate_input(value, field_name):
    try:
        num_value = float(value)
        min_val, max_val = INPUT_RANGES[field_name]
        if not (min_val <= num_value <= max_val):
            return False, f"{field_name} must be between {min_val} and {max_val}"
        return True, num_value
    except ValueError:
        return False, f"Invalid {field_name} value"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or le is None:
        return jsonify({'error': 'Model not loaded. Please contact administrator.'}), 500

    try:
        # Validate all inputs
        data = {}
        for field in INPUT_RANGES.keys():
            if field not in request.form:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            
            is_valid, result = validate_input(request.form[field], field)
            if not is_valid:
                return jsonify({'error': result}), 400
            data[field] = [result]
        
        # Create DataFrame
        input_df = pd.DataFrame(data)
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Get crop name from prediction
        predicted_crop = le.inverse_transform(prediction)[0]
        
        return jsonify({
            'prediction': predicted_crop,
            'status': 'success',
            'input_values': data
        })
    
    except Exception as e:
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 