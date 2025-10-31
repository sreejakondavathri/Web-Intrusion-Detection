# Patent Draft — X-WIDS (Provisional Draft)

**Title:** Adaptive Explainable Hybrid Web Intrusion Detection System (X-WIDS)

**Field:** Cybersecurity, Intrusion Detection Systems, Machine Learning

**Background / Problem:**  
Existing log-based web intrusion detection systems either use static anomaly thresholds or static supervised classifiers. High false positive rates and lack of explainability reduce operational adoption.

**Summary of Invention:**  
A hybrid two-stage detection pipeline: (1) an unsupervised anomaly detector (Isolation Forest) with an adaptive contamination threshold that changes dynamically based on short-term traffic variance; (2) a supervised classifier (Random Forest) that labels only the anomalies from stage (1); and (3) an explainability module producing SHAP-based feature attribution and an automated forensic report. Optionally, the pipeline supports honeypot-driven auto-labeling for continuous model refinement.

**Key Novel Elements / Claims (high-level):**  
1. Adaptive contamination algorithm for unsupervised anomaly detection that computes contamination based on short-term statistical characteristics of the feature distribution.  
2. A two-stage hybrid detection process where only anomalies from the unsupervised stage are passed to a supervised classifier, reducing false positives.  
3. Automated explainability and minimal forensic report generation (SHAP-based feature attribution mapped to actionable suggestions).  
4. (Optional) Honeypot-driven auto-labelling loop to continuously add high-confidence malicious samples to the supervised training set.

**Potential Independent Claims:**  
- A computer-implemented method for web intrusion detection comprising: receiving HTTP access logs; extracting a plurality of behavioral and content features; applying an unsupervised anomaly detector with a dynamically computed contamination value; filtering anomalies to a supervised classifier; and generating an explainability report for each flagged event.

**Advantages:**  
- Reduced false positive rate compared to single-stage detectors  
- Human-friendly explanations accelerating triage  
- Continuous improvement via optional honeypot feedback

**Example Implementation:** Python, IsolationForest, RandomForest, SHAP, Streamlit.

**Priority Considerations:** This file is a draft for provisional filing. Work with an attorney to prepare the final claims and filing.

