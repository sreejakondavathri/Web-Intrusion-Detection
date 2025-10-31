# src/stage2_classifier.py
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/classifier.pkl"
ATTACK_MAP_PATH = "models/attack_mapping.pkl"

def heuristic_labeling(df):
    """Create demo labels: simple heuristics marking suspicious requests as malicious."""
    def is_malicious(row):
        url = str(row.get("url","")).lower()
        if ("or '1'='1" in url) or ("<script" in url) or ("../" in url) or ("%3cscript" in url):
            return 1
        if row.get("special_char_density",0) > 0.05:
            return 1
        # else benign
        return 0

    df["label"] = df.apply(is_malicious, axis=1)
    return df

def train_classifier():
    if not os.path.exists(DATA_PATH):
        print(f"❌ {DATA_PATH} missing. Run feature_extraction first.")
        return

    df = pd.read_csv(DATA_PATH)
    df = heuristic_labeling(df)

    features = ["url_length", "query_params", "special_char_density", "method_encoded", "status_category"]
    X = df[features].fillna(0)
    y = df["label"].astype(int)

    if len(np.unique(y)) == 1 or (len(y) > 1 and min(y.value_counts()) < 2):
        print("⚠ Not enough label variety for fully stratified training. Proceeding with a simple split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # Save classifier
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Classifier saved to {MODEL_PATH}")

    # Save a small attack mapping (keeps compatibility). We keep it simple since we
    # infer the specific attack type in the dashboard via heuristics.
    attack_mapping = {
        0: "Normal",
        1: "Malicious"
    }
    joblib.dump(attack_mapping, ATTACK_MAP_PATH)
    print(f"✅ Attack mapping saved to {ATTACK_MAP_PATH}")

if __name__ == "__main__":
    train_classifier()
