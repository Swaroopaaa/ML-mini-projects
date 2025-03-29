from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model_path = "model/logistic_regression.pkl"  # Logistic Regression model
scaler_path = "model/scaler.pkl"

try:
    print("🔄 Loading Logistic Regression model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    exit(1)

# Feature order as expected by the model
FEATURE_ORDER = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", 
    "RestECG", "MaxHeartRate", "ExerciseAngina", "Oldpeak", "Slope", "MajorVessels", "Thalassemia"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Ensure feature order
        df = df[FEATURE_ORDER]

        # Scale input features
        df_scaled = scaler.transform(df)

        # Make prediction
        probability = model.predict_proba(df_scaled)[0][1]  # Probability of Heart Disease

        # Custom Threshold (Optional)
        threshold = 0.5  # Standard threshold for Logistic Regression
        prediction = 1 if probability >= threshold else 0

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return jsonify({"prediction": result, "probability": probability})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "_main_":
    print("🔥 Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)