import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import time
from src import utils
from src.explainability import build_shap_explainer, explain_instance
from src.forensic_report import create_forensic_report
from src.project_report import build_project_report
from src.synthetic_attack_generator import generate as generate_synthetic
from src import feature_extraction
from src import stage1_isolation_forest, stage2_classifier
from src.evaluate import evaluate_and_plot
from src.file_parser import parse_uploaded_file  # NEW ✅
import joblib
import re
import os

st.set_page_config(page_title="X-WIDS Dashboard", layout="wide")
st.title("X-WIDS — Explainable Hybrid Web Intrusion Detection (Complete Mode)")


# ------------------ Helper: infer attack type from URL/payload ------------------
def infer_attack_type(url: str) -> str:
    """
    Lightweight heuristic to map a suspicious URL to an attack type name.
    This does not change model outputs — it only provides a human-readable label.
    """
    if not isinstance(url, str):
        return "Unknown"

    u = url.lower()

    # SQL Injection patterns
    if re.search(r"(\bor\b|\band\b).+?=.*'|union\s+select|drop\s+table|--", u) or "'" in u and "or" in u:
        return "SQL Injection"

    # XSS patterns
    if "<script" in u or "%3cscript" in u or "onerror=" in u or "alert(" in u or "<img" in u:
        return "Cross-Site Scripting (XSS)"

    # Directory traversal
    if "../" in u or "..%2f" in u or "etc/passwd" in u:
        return "Directory Traversal"

    # File upload / LFI/RFI patterns
    if "upload" in u and ("file" in u or ".php" in u or "windows" in u):
        return "File Upload / LFI"

    # API abuse or admin probing
    if "/admin" in u or "wp-admin" in u or "/login" in u and ("id=" in u and "'" in u):
        return "Brute Force / Auth Abuse"

    # generic suspicious (fallback)
    return "Generic Exploit"


# ------------------ SIDEBAR CONTROLS ------------------
st.sidebar.header("Controls")

if st.sidebar.button("Append synthetic attacks to sample_logs.txt"):
    generate_synthetic(num_benign=200, num_mal=80)
    st.sidebar.success("Synthetic logs added. Now run feature extraction and train models.")

if st.sidebar.button("Run feature extraction"):
    feature_extraction.extract_features("data/sample_logs.txt")
    st.sidebar.success("Feature extraction complete.")

if st.sidebar.button("Train Stage1 (IsolationForest)"):
    stage1_isolation_forest.train_isolation_forest()
    st.sidebar.success("Stage 1 trained successfully.")

if st.sidebar.button("Train Stage2 (RandomForest)"):
    stage2_classifier.train_classifier()
    st.sidebar.success("Stage 2 trained successfully.")

if st.sidebar.button("Run Evaluation (train/eval)"):
    try:
        evaluate_and_plot()
        st.sidebar.success("Evaluation completed. Check Metrics tab.")
    except Exception as e:
        st.sidebar.error(f"Evaluation error: {e}")

if st.sidebar.button("Build final project report (PDF)"):
    forensic_files = sorted([os.path.join("reports",f) for f in os.listdir("reports") if f.startswith("forensic_")], reverse=True)
    example = forensic_files[0] if forensic_files else None
    out = build_project_report(example_forensic_pdf=example)
    st.sidebar.success(f"Report created: {out}")
    st.sidebar.markdown(f"[Download Report]({out})")


# ------------------ LOAD MODELS ------------------
try:
    if_model, scaler = utils.load_if_and_scaler()
except Exception as e:
    st.warning(f"Stage-1 model not loaded. Train Stage-1. {e}")
    if_model, scaler = None, None

try:
    clf = utils.load_clf()
except Exception as e:
    st.warning(f"Stage-2 classifier not loaded. Train Stage-2. {e}")
    clf = None

explainer = None
if clf is not None:
    try:
        explainer = build_shap_explainer()
    except Exception as e:
        st.warning(f"SHAP explainer not loaded: {e}")


# ------------------ TABS ------------------
tab1, tab2, tab3 = st.tabs(["Live Detection", "Metrics", "Reports"])


# ------------------ TAB 1: LIVE DETECTION ------------------
with tab1:
    st.subheader("Upload Log File for Detection")

    uploaded = st.file_uploader(
        "Upload log file (TXT, LOG, CSV, JSON, PCAP):",
        type=["txt", "log", "csv", "json", "pcap"]
    )

    df = None
    if uploaded:
        try:
            df = parse_uploaded_file(uploaded)
            st.success(f"File parsed successfully as {uploaded.name.split('.')[-1].upper()} format.")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")

    else:
        st.info("No file uploaded. Using default sample_logs.txt")
        if os.path.exists("data/sample_logs.txt"):
            with open("data/sample_logs.txt", "r", encoding="utf-8") as f:
                content = f.read()
            # create a fake UploadedFile-like object for reuse of parser
            class Tmp:
                name = "sample_logs.txt"
                def getvalue(self): return content.encode()
            df = parse_uploaded_file(Tmp())
        else:
            st.error("No default log file found.")

    if df is not None and not df.empty:
        st.write("Parsed Logs Preview:")
        st.dataframe(df.head())

        # Feature computation
        try:
            df["url_length"] = df["url"].apply(lambda x: len(str(x)))
            df["query_params"] = df["url"].apply(lambda x: str(x).count("="))
            special_chars = ["'", '"', "<", ">", ";", "--", "#", "%", "(", ")"]
            df["special_char_density"] = df["url"].apply(
                lambda x: sum(c in str(x) for c in special_chars) / max(len(str(x)), 1)
            )
            df["method_encoded"] = df["method"].apply(lambda x: 1 if str(x).upper() == "POST" else (0 if str(x).upper() == "GET" else 2))
            df["status_category"] = df["status"].apply(lambda x: int(x) // 100 if pd.notnull(x) else 0)

            # Apply Stage-1 and Stage-2
            results = []
            features = ["url_length", "query_params", "special_char_density", "method_encoded", "status_category"]

            for _, row in df.iterrows():
                feat_row = row[features].fillna(0)
                # compute anomaly score
                anomaly_score, is_anom = utils.compute_anomaly_score(if_model, scaler, pd.DataFrame([feat_row], columns=features))
                
                result = row.to_dict()
                result["anomaly_score"] = anomaly_score
                result["is_anomaly"] = bool(is_anom)
                result["stage2_label"] = None
                result["attack_type"] = "Normal"
                result["suggested_action"] = "none"
                result["shap_img"] = None

                if is_anom and clf is not None:
                    pred = int(clf.predict(pd.DataFrame([feat_row], columns=features))[0])
                    result["stage2_label"] = pred
                    if pred == 1:
                        # infer a human-friendly attack type from the URL/payload
                        atk = infer_attack_type(result.get("url", ""))
                        result["attack_type"] = atk
                        result["suggested_action"] = "Block IP & create WAF rule"
                        if explainer is not None:
                            _, shap_img = explain_instance(explainer, pd.DataFrame([feat_row], columns=features), feature_names=features)
                            # shap_img could be a path (string) or None
                            if shap_img:
                                result["shap_img"] = shap_img
                    else:
                        result["attack_type"] = "Normal"

                results.append(result)

            results_df = pd.DataFrame(results)

            # Ensure attack_type column exists
            if "attack_type" not in results_df.columns:
                results_df["attack_type"] = "Normal"

            st.subheader("Detection Results")
            # Show the new attack_type column beside stage2_label
            st.dataframe(results_df[["ip", "url", "anomaly_score", "is_anomaly", "stage2_label", "attack_type", "suggested_action"]])

            # Detailed view
            idx = st.selectbox("Select an index to view details", results_df.index)
            sel = results_df.loc[idx]
            st.write(sel.to_dict())

            shap_img = sel.get("shap_img")
            if shap_img and isinstance(shap_img, str) and shap_img.endswith(".png") and os.path.exists(shap_img):
                st.image(shap_img, caption="SHAP Explanation")
            else:
                st.info("No SHAP explanation available for this instance.")

            if st.button("Generate Forensic Report for this Event"):
                if sel.get("stage2_label") == 1:
                    pdf = create_forensic_report(sel.to_dict(), shap_img_path=sel.get("shap_img"))
                    st.success(f"Forensic Report Generated Successfully!")

                    with open(pdf, "rb") as file:
                        st.download_button(
                            label="📄 Download Forensic Report",
                            data=file,
                            file_name=os.path.basename(pdf),
                            mime="application/pdf"
                        )
                else:
                    st.warning("Only malicious predictions can generate forensic reports.")

        except Exception as e:
            st.error(f"Error during detection processing: {e}")


# ------------------ TAB 2: METRICS ------------------
with tab2:
    st.subheader("Evaluation Metrics")
    metrics_file = os.path.join("reports", "metrics_summary.json")
    if os.path.exists(metrics_file):
        import json
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        st.metric("Accuracy", metrics.get("accuracy", "-"))
        st.metric("ROC AUC", metrics.get("roc_auc", "-"))
        st.write("Confusion Matrix:")
        st.image(os.path.join("reports", "confusion_matrix.png"))
        if os.path.exists(os.path.join("reports", "roc_curve.png")):
            st.write("ROC Curve:")
            st.image(os.path.join("reports", "roc_curve.png"))
    else:
        st.warning("Run Evaluation from sidebar to generate metrics.")


# ------------------ TAB 3: REPORTS ------------------
with tab3:
    st.subheader("Generated Reports & Files")
    files = sorted([f for f in os.listdir("reports")] , reverse=True)
    if files:
        for f in files:
            st.markdown(f"- [{f}](reports/{f})")
    else:
        st.info("No reports generated yet.")
