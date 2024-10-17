from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and encoder
model = joblib.load('fraud_detect_model')
type_encoder = joblib.load('type_encoder')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            transaction_type = request.form['type']
            amount = float(request.form['amount'])
            oldbalanceOrg = float(request.form['oldbalanceOrg'])
            newbalanceDest = float(request.form['newbalanceDest'])

            # Encode the transaction type (checking if transform returns a scalar or array)
            encoded_type = type_encoder.transform([[transaction_type]])

            # Check if the encoder returned an array or scalar
            if isinstance(encoded_type, np.ndarray):
                encoded_type = encoded_type[0]  # In case it returns a 2D array

            # Prepare the features for prediction
            features = np.array([[encoded_type, amount, oldbalanceOrg, newbalanceDest]])

            # Make prediction
            prediction = model.predict(features)

            # Output
            if prediction[0] == 1:
                result = "The model predicts that this is Fraud."
            else:
                result = "The model predicts that this is No Fraud."

            return render_template('index.html', prediction_text=result)
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
