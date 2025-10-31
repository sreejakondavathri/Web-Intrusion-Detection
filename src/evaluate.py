# src/evaluate.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

# Utilities / paths
DATA_PATH = "data/processed_data.csv"
CLF_PATH = "models/classifier.pkl"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_processed():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Processed data missing. Run src/feature_extraction.py first.")
    df = pd.read_csv(DATA_PATH)
    return df

def evaluate_and_plot():
    df = load_processed()
    # heuristic labeling (same as stage2 classifier)
    def is_malicious(row):
        url = str(row.get("url","")).lower()
        if ("or '1'='1" in url) or ("<script" in url) or ("../" in url) or ("%3Cscript" in url):
            return 1
        if row.get("special_char_density",0) > 0.05:
            return 1
        return 0
    df["label"] = df.apply(is_malicious, axis=1)
    features = ["url_length", "query_params", "special_char_density", "method_encoded", "status_category"]
    X = df[features].fillna(0)
    y = df["label"].astype(int)

    # require classifier
    if not os.path.exists(CLF_PATH):
        raise FileNotFoundError("Classifier not found. Run stage2_classifier.py first.")
    clf = joblib.load(CLF_PATH)

    # do a full-predict evaluation (use cross-validation style but simple)
    preds = clf.predict(X)
    probs = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[:,1]
    else:
        # use decision_function (not all classifiers have it)
        try:
            probs = clf.predict_proba(X)[:,1]
        except:
            probs = np.zeros(len(preds))

    acc = accuracy_score(y, preds)
    cr = classification_report(y, preds, zero_division=0, output_dict=True)
    cm = confusion_matrix(y, preds)
    auc = None
    try:
        auc = roc_auc_score(y, probs)
    except:
        auc = None

    # Save metrics to a summary CSV/JSON
    metrics = {
        "accuracy": acc,
        "roc_auc": auc,
        "n_total": int(len(y)),
        "n_malicious": int(y.sum()),
        "n_benign": int(len(y)-y.sum()),
        "confusion_matrix": cm.tolist(),
        "classification_report": cr
    }
    import json
    with open(os.path.join(REPORTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print("✅ Metrics saved to reports/metrics_summary.json")

    # Confusion matrix plot
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    for (i,j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha='center', va='center')
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.clf()
    print(f"✅ Confusion matrix saved to {cm_path}")

    # ROC Curve if auc exists
    if auc is not None:
        fpr, tpr, _ = roc_curve(y, probs)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title(f"ROC Curve (AUC={auc:.3f})")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
        plt.savefig(roc_path, bbox_inches="tight")
        plt.clf()
        print(f"✅ ROC curve saved to {roc_path}")

    # Save a small dataset of top false positives/negatives for the report
    df["pred"] = preds
    fp = df[(df["pred"]==1) & (df["label"]==0)].head(10)
    fn = df[(df["pred"]==0) & (df["label"]==1)].head(10)
    fp.to_csv(os.path.join(REPORTS_DIR, "false_positives_sample.csv"), index=False)
    fn.to_csv(os.path.join(REPORTS_DIR, "false_negatives_sample.csv"), index=False)
    print("✅ Saved FP/FN samples")

    return metrics

if __name__ == "__main__":
    metrics = evaluate_and_plot()
    print("Evaluation complete. Summary:", metrics)
