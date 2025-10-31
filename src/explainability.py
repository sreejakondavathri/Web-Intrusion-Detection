# src/explainability.py
import joblib
import shap
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "models/classifier.pkl"
SHAP_EXPLAINER_PATH = "models/shap_explainer.pkl"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def build_shap_explainer(sample_X=None):
    """
    Build and return a SHAP explainer for the classifier.
    If possible uses TreeExplainer for tree models, otherwise falls back to shap.Explainer.
    sample_X (optional): a DataFrame or numpy array used when kernel-like explainers need a background.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Classifier model not found. Train the classifier first.")

    clf = joblib.load(MODEL_PATH)

    # Prefer TreeExplainer for tree models (fast)
    try:
        explainer = shap.TreeExplainer(clf)
    except Exception:
        # Fallback to generic Explainer; supply background data if given
        try:
            if sample_X is not None:
                explainer = shap.Explainer(clf, sample_X)
            else:
                explainer = shap.Explainer(clf)
        except Exception as e:
            raise RuntimeError(f"Could not create SHAP explainer: {e}")

    return explainer


def explain_instance(explainer, X_row, feature_names=None, show_plot=False):
    """
    Explain a single instance (one row).
    Parameters:
      - explainer: a shap.Explainer/TreeExplainer instance
      - X_row: pandas.DataFrame (one row) or 2D array of shape (1, n_features)
      - feature_names: optional list of feature names (not passed to shap.plots.bar because API changed)
      - show_plot: if True, show matplotlib figure (not used in Streamlit)
    Returns:
      - shap_explanation: the SHAP Explanation object (or shap values array fallback)
      - fig_path: path to saved PNG of the SHAP bar plot (or None if plotting failed)
    """
    # Normalize input to DataFrame (SHAP prefers DataFrame to preserve feature names)
    if isinstance(X_row, np.ndarray):
        X_df = pd.DataFrame(X_row)
        if feature_names:
            X_df.columns = feature_names
    elif isinstance(X_row, pd.DataFrame):
        X_df = X_row.copy()
        if feature_names and (len(feature_names) == X_df.shape[1]):
            X_df.columns = feature_names
    else:
        # try to coerce
        X_df = pd.DataFrame([X_row])

    # Compute SHAP values / Explanation object
    try:
        explanation = explainer(X_df)
    except Exception as e:
        # Some older versions expect explainer.shap_values, try that as a fallback
        try:
            shap_vals = explainer.shap_values(X_df)
            # wrap into a simple container for compatibility
            explanation = shap_vals
        except Exception as e2:
            raise RuntimeError(f"Failed to compute SHAP values: {e}; fallback failed: {e2}")

    # Prepare to plot
    fig_path = None
    try:
        # Clear existing figures
        plt.clf()
        plt.close("all")

        # Newer SHAP returns an Explanation object; older returns arrays/lists.
        # shap.plots.bar accepts either an Explanation or arrays in recent versions.
        if hasattr(explanation, "__len__") and not hasattr(explanation, "values"):
            # explanation could be a list of arrays (older shap). Choose class-1 if present
            if isinstance(explanation, list) and len(explanation) > 1:
                to_plot = explanation[1]  # class 1
            else:
                to_plot = explanation
            # shap.plots.bar can accept numpy arrays in some versions
            shap.plots.bar(to_plot, show=False)
        elif hasattr(explanation, "values"):
            # Explanation object — handle multi-output vs single-output
            # If multi-output (shape: (n_outputs, n_samples, n_features)) choose first output or the positive class
            vals = explanation.values
            if vals.ndim == 3:
                # e.g., (n_outputs, n_samples, n_features)
                # Choose the first output index that seems appropriate (commonly index 1 is positive class)
                # If only one sample, pick [:,0,:] to plot
                # Try to pick the "positive" class (index 1) if possible
                idx = 1 if vals.shape[0] > 1 else 0
                # build a small Explanation object for that output if possible, else pass array
                try:
                    single_exp = explanation[idx]
                    shap.plots.bar(single_exp, show=False)
                except Exception:
                    shap.plots.bar(vals[idx][0], show=False)
            else:
                # vals.ndim == 2 probably (n_samples, n_features)
                shap.plots.bar(explanation, show=False)
        else:
            # last resort: try to call shap.plots.bar directly on the object
            shap.plots.bar(explanation, show=False)

        # Save figure to file
        fig_path = os.path.join(REPORTS_DIR, "temp_shap.png")
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight")
        plt.clf()
        plt.close("all")
    except TypeError as te:
        # Known issue: some shap versions don't accept certain argument shapes.
        # As a fallback, create a simple bar chart using the raw values.
        try:
            # Try to extract numeric values
            if isinstance(explanation, list):
                vals = np.array(explanation[0]).flatten()
            elif hasattr(explanation, "values"):
                vals = explanation.values
                if vals.ndim == 3:
                    vals = vals[0][0]  # pick first output and first sample
                elif vals.ndim == 2:
                    vals = vals[0]
                else:
                    vals = vals.flatten()
            else:
                vals = np.array(explanation).flatten()

            # feature names
            if feature_names and len(feature_names) == len(vals):
                names = feature_names
            else:
                names = [f"f{i}" for i in range(len(vals))]

            plt.figure(figsize=(6, 3))
            y_pos = np.arange(len(vals))
            plt.barh(y_pos, vals)
            plt.yticks(y_pos, names)
            plt.xlabel("SHAP value (impact)")
            plt.gca().invert_yaxis()  # largest on top
            fig_path = os.path.join(REPORTS_DIR, "temp_shap.png")
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches="tight")
            plt.clf()
            plt.close("all")
        except Exception as e:
            print("Fallback SHAP plotting failed:", e)
            fig_path = None
    except Exception as e:
        print("SHAP plot generation error:", e)
        fig_path = None

    return explanation, fig_path
