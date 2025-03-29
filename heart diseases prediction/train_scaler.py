import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_excel("Heart_Disease_Prediction.xlsx")  # Ensure correct dataset path

# Define features & target
X = df.drop(columns=["HeartDisease"])  # Adjust based on your dataset
y = df["HeartDisease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model & scaler
joblib.dump(model, "model/logistic_regression.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model and scaler saved successfully.")