from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('diamond_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# We need to encode the categorical features.
# For simplicity we use a basic mapping; in production you might need a robust logic.
# NOTE: Update these mappings as per the ones used during training.
cut_mapping = {'Ideal': 0, 'Premium': 1, 'Good': 2, 'Very Good': 3, 'Fair': 4}
color_mapping = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
clarity_mapping = {'IF': 0, 'VVS1': 1, 'VVS2': 2, 'VS1': 3, 'VS2': 4, 'SI1': 5, 'SI2': 6, 'I1': 7}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        carat = float(request.form['Carat(Weight of Daimond)'])
        cut = request.form['Cut(Quality)']
        color = request.form['Color']
        clarity = request.form['Clarity']
        depth = float(request.form['Depth'])
        table = float(request.form['Table'])
        x = float(request.form['X(length)'])
        y = float(request.form['Y(width)'])
        z = float(request.form['Z(Depth)'])
        
        # Convert categorical features using mapping
        # If key not found, return error message.
        if cut not in cut_mapping:
            return 'Cut value not recognized. Available values: ' + ', '.join(cut_mapping.keys())
        if color not in color_mapping:
            return 'Color value not recognized. Available values: ' + ', '.join(color_mapping.keys())
        if clarity not in clarity_mapping:
            return 'Clarity value not recognized. Available values: ' + ', '.join(clarity_mapping.keys())
            
        cut_val = cut_mapping[cut]
        color_val = color_mapping[color]
        clarity_val = clarity_mapping[clarity]
        
        # Create input DataFrame, ensuring feature order is consistent with training
        # Expected order: Carat(Weight of Daimond), Cut(Quality), Color, Clarity, Depth, Table, X(length), Y(width), Z(Depth)
        input_data = pd.DataFrame([[carat, cut_val, color_val, clarity_val, depth, table, x, y, z]],
                                  columns=['Carat(Weight of Daimond)', 'Cut(Quality)', 'Color', 'Clarity', 'Depth', 'Table', 'X(length)', 'Y(width)', 'Z(Depth)'])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Predict price
        prediction = model.predict(input_scaled)[0]
        return 'Predicted Diamond Price (in US dollars): {:.2f}'.format(prediction)
    except Exception as e:
        return 'Error occurred: ' + str(e)

if __name__ == '__main__':
    app.run(debug=True)