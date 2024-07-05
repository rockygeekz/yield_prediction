from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_file = 'models/crop_yield_model.pkl'
model = joblib.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()

    # Prepare input data as DataFrame
    input_data = pd.DataFrame([data])

    # Perform prediction
    prediction = model.predict(input_data)

    # Return prediction as JSON response
    return jsonify({'predicted_yield': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
