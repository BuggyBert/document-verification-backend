
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Route to handle document verification
@app.route('/verify', methods=['POST'])
def verify_document():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)  # Assuming you pass features as a list

    # Predict using the trained model
    prediction = model.predict(features)

    # Return the result
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
