import os
import joblib
import numpy as np
import pandas as pd

# Paths to saved models and objects
IF_MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler.pkl"
CLF_MODEL_PATH = "models/classifier.pkl"
FEATURES_PATH = "models/features.pkl"

# ---------------------------------------------------
# Load Stage-1 Isolation Forest and Scaler
# ---------------------------------------------------
def load_if_and_scaler():
    if not os.path.exists(IF_MODEL_PATH):
        raise FileNotFoundError("Isolation Forest model not found. Train Stage 1 first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found. Train Stage 1 first.")

    if_model = joblib.load(IF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return if_model, scaler

# ---------------------------------------------------
# Load Stage-2 Classifier
# ---------------------------------------------------
def load_clf():
    if not os.path.exists(CLF_MODEL_PATH):
        raise FileNotFoundError("Classifier model not found. Train Stage 2 first.")
    clf = joblib.load(CLF_MODEL_PATH)
    return clf

# ---------------------------------------------------
# Load feature names used for training
# ---------------------------------------------------
def load_feature_list():
    if os.path.exists(FEATURES_PATH):
        return joblib.load(FEATURES_PATH)
    else:
        return None

# ---------------------------------------------------
# Compute anomaly score using Stage-1
# ---------------------------------------------------
def compute_anomaly_score(if_model, scaler, X):
    """
    X: DataFrame or numpy array with ONE row of features
    Returns: (score, is_anom)
    """
    try:
        X_scaled = scaler.transform(X)
        score = if_model.decision_function(X_scaled)[0]   # anomaly score
        is_anom = if_model.predict(X_scaled)[0] == -1     # anomaly detected if -1
        return score, is_anom
    except Exception as e:
        print("Error computing anomaly score:", e)
        return None, False

# ---------------------------------------------------
# Make stage-2 classification prediction
# ---------------------------------------------------
def classify_instance(clf, X):
    """
    X: DataFrame or numpy array with ONE row of features
    Returns: predicted label
    """
    try:
        pred = clf.predict(X)[0]
        return pred
    except Exception as e:
        print("Error in classifier prediction:", e)
        return None

# ---------------------------------------------------
# Optional: Feature extraction placeholder
# ---------------------------------------------------
def run_feature_extraction(input_df):
    """
    You should already have your feature extraction logic
    in another file. If the dashboard uses this function,
    keep it as a wrapper. Otherwise, it's safe to leave.
    """
    # Example: directly return input if features are preprocessed
    return input_df
