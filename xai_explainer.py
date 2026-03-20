"""
xai_explainer.py
SHAP-based feature importance on top of the Root Cause Classifier.
"""
import warnings

import numpy as np
import pandas as pd
import shap

from anomaly_detector import FEATURE_NAMES

warnings.filterwarnings("ignore")

# Human-readable feature display names
FEATURE_DISPLAY_NAMES = {
    "v_mean":          "Voltage mean (V)",
    "v_std":           "Voltage std dev",
    "v_range":         "Voltage range",
    "v_skew":          "Voltage skewness",
    "v_max_diff":      "Voltage max Δ/sample",
    "i_mean":          "Current mean (A)",
    "i_std":           "Current std dev",
    "i_range":         "Current range",
    "i_skew":          "Current skewness",
    "i_max_diff":      "Current max Δ/sample",
    "f_std":           "Frequency std dev",
    "f_range":         "Frequency range",
    "pf_mean":         "Power factor mean",
    "pf_std":          "Power factor std dev",
    "pf_trend":        "Power factor trend",
    "thd_mean":        "THD mean (%)",
    "thd_std":         "THD std dev",
    "vi_corr":         "V–I correlation",
    "v_stuck":         "Voltage flatness",
    "cross_corr_break":"Correlation break",
}


class XAIExplainer:
    def __init__(self, classifier):
        self.classifier  = classifier
        self._explainer  = None

    # ------------------------------------------------------------------
    def fit(self) -> "XAIExplainer":
        """Build SHAP TreeExplainer from the fitted RF model."""
        self._explainer = shap.TreeExplainer(self.classifier.model)
        return self

    # ------------------------------------------------------------------
    def shap_values_for_sample(
        self, features: np.ndarray, class_idx: int
    ) -> np.ndarray:
        """Return |SHAP| values for a single sample and a given class."""
        sv = self._explainer.shap_values(features.reshape(1, -1))
        # sv shape varies by shap version — always flatten to 1D
        if isinstance(sv, list):
            vals = np.abs(np.array(sv[class_idx]).flatten())
        else:
            vals = np.abs(np.array(sv).flatten())
        # if flattened length > n_features, take last n_features (multi-output edge case)
        n = len(FEATURE_NAMES)
        if len(vals) > n:
            vals = vals[-n:]
        return vals

    # ------------------------------------------------------------------
    def get_top_features(
        self,
        features: np.ndarray,
        predicted_cause: str,
        top_n: int = 8,
    ) -> pd.DataFrame:
        """Return top-N features sorted by |SHAP| for the predicted class."""
        features = np.array(features).flatten()
        class_idx = self.classifier.get_class_index(predicted_cause)
        shap_vals = self.shap_values_for_sample(features, class_idx)

        df = pd.DataFrame({
            "feature":      FEATURE_NAMES,
            "display_name": [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_NAMES],
            "shap_value":   shap_vals,
            "raw_value":    features,
        })
        return df.nlargest(top_n, "shap_value").reset_index(drop=True)

    # ------------------------------------------------------------------
    def full_feature_df(self, features: np.ndarray, predicted_cause: str) -> pd.DataFrame:
        """Full feature table with SHAP values (all 20 features)."""
        features = np.array(features).flatten()
        class_idx = self.classifier.get_class_index(predicted_cause)
        shap_vals = self.shap_values_for_sample(features, class_idx)

        return pd.DataFrame({
            "Feature":       [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_NAMES],
            "Raw name":      FEATURE_NAMES,
            "Value":         [f"{v:.4f}" for v in features],
            "|SHAP|":        [f"{v:.4f}" for v in shap_vals],
        }).sort_values("|SHAP|", ascending=False).reset_index(drop=True)
