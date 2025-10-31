import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler.pkl"

def train_isolation_forest():
    # Ensure data exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found at {DATA_PATH}. Run feature_extraction first.")
        return

    df = pd.read_csv(DATA_PATH)

    # Select features for training
    features = ["url_length", "query_params", "special_char_density", "method_encoded", "status_category"]
    X = df[features]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ✅ Adaptive Contamination (Novel Feature)
    contamination = adaptive_contamination(X_scaled)

    print(f"🔧 Using adaptive contamination: {contamination:.4f}")

    # Train Isolation Forest
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(X_scaled)

    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"✅ Isolation Forest Model saved to {MODEL_PATH}")
    print(f"✅ Scaler saved to {SCALER_PATH}")

def adaptive_contamination(X):
    """
    Adaptive contamination based on anomaly scores distribution.
    Automatically adjusts threshold based on data variance.
    """
    var = np.var(X)
    if var < 0.5:
        return 0.01  # Very stable traffic
    elif var < 1.0:
        return 0.02  # Slightly variable
    else:
        return 0.05  # Highly variable → more anomalies expected

if __name__ == "__main__":
    train_isolation_forest()
