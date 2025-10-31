# X-WIDS — Project Documentation (Short)

## Abstract
X-WIDS is an explainable hybrid web intrusion detection system that combines unsupervised anomaly detection with supervised classification and SHAP-based explanations. It introduces an adaptive contamination mechanism and produces automated forensic reports for every flagged event.

## Methodology
1. Collect HTTP access logs in common log format.  
2. Extract engineered features: url_length, query_params, special_char_density, method_encoded, status_category.  
3. Train IsolationForest with adaptive contamination to find anomalies.  
4. For anomalies, run a RandomForest classifier (trained on heuristic + honeypot data).  
5. Generate SHAP explanations and a one-page forensic PDF.

## How to run (quick)
1. Install requirements: `pip install -r requirements.txt`  
2. Generate synthetic logs (optional): `python src/synthetic_attack_generator.py`  
3. Extract features: `python src/feature_extraction.py`  
4. Train stage1: `python src/stage1_isolation_forest.py`  
5. Train stage2: `python src/stage2_classifier.py`  
6. Evaluate: `python src/evaluate.py`  
7. Run dashboard: `streamlit run dashboard/dashboard.py`

## Files
- `dashboard/` — Streamlit UI  
- `src/` — core code (feature extraction, models, explainability)  
- `data/` — logs + processed CSV  
- `models/` — saved models  
- `reports/` — evaluation images, forensic PDFs, project report

## Results
After evaluation the system produces:
- `reports/metrics_summary.json`  
- `reports/confusion_matrix.png`  
- `reports/roc_curve.png` (if available)  
- forensic PDFs for flagged events

## Future work
- Add honeypot auto-labeling loop  
- Federated updates for privacy-preserving model sharing  
- WAF integration for automated blocking
