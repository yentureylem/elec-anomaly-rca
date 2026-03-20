"""
anomaly_detector.py
Feature extraction + Isolation Forest anomaly detection for electrical measurements.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Feature names  (must stay in sync with extract_features return order)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "v_mean",        "v_std",       "v_range",    "v_skew",    "v_max_diff",
    "i_mean",        "i_std",       "i_range",    "i_skew",    "i_max_diff",
    "f_std",         "f_range",
    "pf_mean",       "pf_std",      "pf_trend",
    "thd_mean",      "thd_std",
    "vi_corr",       "v_stuck",     "cross_corr_break",
]


def extract_features(window: pd.DataFrame) -> np.ndarray:
    """Extract a 20-dimensional feature vector from a measurement window."""
    v   = window["voltage"].values
    i   = window["current"].values
    f   = window["frequency"].values
    pf  = window["power_factor"].values
    thd = window["thd"].values

    safe_corr = (
        float(np.corrcoef(v, i)[0, 1])
        if np.std(v) > 1e-6 and np.std(i) > 1e-6
        else 0.0
    )

    features = [
        # Voltage
        np.mean(v), np.std(v), float(np.ptp(v)), float(stats.skew(v)),
        float(np.max(np.abs(np.diff(v)))),
        # Current
        np.mean(i), np.std(i), float(np.ptp(i)), float(stats.skew(i)),
        float(np.max(np.abs(np.diff(i)))),
        # Frequency
        np.std(f), float(np.ptp(f)),
        # Power factor
        np.mean(pf), np.std(pf),
        float(np.polyfit(np.arange(len(pf)), pf, 1)[0]),
        # THD
        np.mean(thd), np.std(thd),
        # Cross-channel
        safe_corr,
        1.0 / (np.std(v) + 1e-6),            # high = voltage stuck/flat
        abs(safe_corr - 0.95),                 # deviation from normal V-I correlation
    ]

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    def __init__(self, contamination: float = 0.05, window_size: int = 30):
        self.contamination = contamination
        self.window_size   = window_size
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
        )
        self.score_min = 0.0
        self.score_max = 1.0
        self._fitted   = False

    # ------------------------------------------------------------------
    def fit_on_normal(self, normal_data: pd.DataFrame) -> "AnomalyDetector":
        """Fit Isolation Forest on clean (normal) data windows."""
        features, step = [], max(1, self.window_size // 2)

        for start in range(0, len(normal_data) - self.window_size, step):
            w = normal_data.iloc[start : start + self.window_size]
            features.append(extract_features(w))

        X = np.array(features)
        self.model.fit(X)

        raw = -self.model.score_samples(X)
        self.score_min = float(raw.min())
        self.score_max = float(raw.max())
        self._fitted   = True
        return self

    # ------------------------------------------------------------------
    def _normalize(self, raw: float) -> float:
        return float(
            np.clip((raw - self.score_min) / (self.score_max - self.score_min + 1e-6), 0, 1)
        )

    # ------------------------------------------------------------------
    def compute_anomaly_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Return per-sample anomaly score in [0, 1] via rolling windows."""
        n      = len(data)
        scores = np.zeros(n, dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)
        step   = max(1, self.window_size // 4)

        for start in range(0, n - self.window_size, step):
            w    = data.iloc[start : start + self.window_size]
            feat = extract_features(w).reshape(1, -1)
            raw  = -self.model.score_samples(feat)[0]
            norm = self._normalize(raw)
            scores[start : start + self.window_size] += norm
            counts[start : start + self.window_size] += 1

        counts = np.where(counts == 0, 1, counts)
        return scores / counts

    # ------------------------------------------------------------------
    def get_window_features(self, data: pd.DataFrame, center: int) -> np.ndarray:
        """Extract features from the window centred at *center*."""
        half  = self.window_size // 2
        start = max(0, center - half)
        end   = min(len(data), center + half)
        return extract_features(data.iloc[start:end])
