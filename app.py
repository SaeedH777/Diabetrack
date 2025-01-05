from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'model/Diabetes_predictor_project.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = [float(x) for x in request.form.values()]
        data_array = np.array(data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(data_array)
        result = 'Diabetes Positive' if prediction[0] == 1 else 'Diabetes Negative'
        
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
