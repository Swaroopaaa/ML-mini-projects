import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define dataset path
dataset_path = "Heart_Disease_Prediction.xlsx"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found. Please check the file path.")

try:
    # Load dataset
    df = pd.read_excel(dataset_path)
except Exception as e:
    raise Exception(f"Error reading the dataset: {e}")

# Check if dataset is empty
if df.empty:
    raise ValueError("The dataset is empty. Please provide a valid dataset.")

# Ensure 'HeartDisease' column exists
if 'HeartDisease' not in df.columns:
    raise KeyError("The dataset does not contain a 'HeartDisease' column.")

# Features & Target
X = df.drop(columns=['HeartDisease'], errors='ignore')  # Ensure 'HeartDisease' column exists
y = df['HeartDisease']

# Handle missing values
X.fillna(X.median(numeric_only=True), inplace=True)

# Ensure enough data
if len(X) < 10:
    raise ValueError("Not enough data points available for training.")

# Ensure at least two unique classes in target
if y.nunique() < 2:
    raise ValueError("Dataset must contain at least two classes in the target column.")

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "logistic_regression.pkl")
joblib.dump(model, model_path)
print(f"Model saved successfully at: {model_path}")