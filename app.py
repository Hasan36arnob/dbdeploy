from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
try:
    model = joblib.load('diabetes_model.pkl')  # Replace with your model file path
except Exception as e:
    raise ValueError(f"Error loading the model: {str(e)}")

# Load the LabelEncoders
try:
    label_encoders = joblib.load('label_encoders.pkl')  # Replace with your label encoders file path
except Exception as e:
    raise ValueError(f"Error loading the label encoders: {str(e)}")

# Define all feature names with 'Age' as the first feature
feature_names = [
    'Age',  # Age is now the first feature
    'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia',
    'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis',
    'muscle stiffness', 'Alopecia', 'Obesity'
]

# Separate categorical and numerical features
categorical_fields = [
    'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
    'weakness', 'Polyphagia', 'visual blurring', 'Itching',
    'Irritability', 'delayed healing', 'partial paresis',
    'muscle stiffness', 'Alopecia', 'Obesity'
]
numerical_fields = ['Age']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate that all required fields are present
        if not all(field in data for field in feature_names):
            return jsonify({'error': 'Missing fields in input data'}), 400

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([data])

        # Ensure the features are in the correct order
        input_data = input_data[feature_names]

        # Apply Label Encoding to the categorical fields
        for field in categorical_fields:
            encoder = label_encoders[field]  # Get the encoder for the specific field
            input_data[field] = encoder.transform([input_data[field].values[0]])[0]

        # Make predictions using the loaded model
        prediction = model.predict(input_data)

        # Return the prediction as JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        # Log the full error message
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)