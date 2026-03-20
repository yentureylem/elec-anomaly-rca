"""
root_cause_classifier.py
Random Forest classifier trained on synthetic labelled electrical data.
Classes: Normal | Sensor Fault | Cyber Attack | Equipment Fault
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from data_generator import (
    generate_normal_signal,
    inject_sensor_fault,
    inject_cyber_attack,
    inject_equipment_fault,
)
from anomaly_detector import extract_features

warnings.filterwarnings("ignore")

CAUSE_LABELS = ["Normal", "Sensor Fault", "Cyber Attack", "Equipment Fault"]

CAUSE_COLORS = {
    "Normal":           "#1D9E75",
    "Sensor Fault":     "#185FA5",
    "Cyber Attack":     "#A32D2D",
    "Equipment Fault":  "#BA7517",
}

CAUSE_ICONS = {
    "Normal":           "✅",
    "Sensor Fault":     "🔴",
    "Cyber Attack":     "🟠",
    "Equipment Fault":  "🟡",
}


def _generate_training_data(n_per_class: int = 400):
    """Build a labelled dataset from synthetic windows."""
    X, y    = [], []
    WIN     = 30

    for seed in range(n_per_class):
        rng = np.random.RandomState(seed)

        # ── Normal ──────────────────────────────────────────────────────
        data   = generate_normal_signal(n_samples=200, seed=seed)
        start  = rng.randint(0, 170)
        X.append(extract_features(data.iloc[start : start + WIN]))
        y.append("Normal")

        # ── Sensor Fault ────────────────────────────────────────────────
        data   = generate_normal_signal(n_samples=200, seed=seed + 10_000)
        fs     = rng.randint(20, 60)
        ft     = rng.choice(["stuck", "offset", "noise"])
        ch     = rng.choice(["voltage", "current"])
        data   = inject_sensor_fault(data, start_idx=fs, fault_type=ft, affected_channel=ch)
        start  = int(np.clip(rng.randint(fs + 5, fs + 60), 0, 169))
        X.append(extract_features(data.iloc[start : start + WIN]))
        y.append("Sensor Fault")

        # ── Cyber Attack ────────────────────────────────────────────────
        data   = generate_normal_signal(n_samples=200, seed=seed + 20_000)
        atkS   = rng.randint(20, 80)
        atkE   = int(np.clip(atkS + rng.randint(20, 60), atkS + 10, 185))
        data   = inject_cyber_attack(data, start_idx=atkS, end_idx=atkE)
        start  = int(np.clip(rng.randint(atkS + 2, atkE - 4), 0, 169))
        X.append(extract_features(data.iloc[start : start + WIN]))
        y.append("Cyber Attack")

        # ── Equipment Fault ─────────────────────────────────────────────
        data   = generate_normal_signal(n_samples=200, seed=seed + 30_000)
        eqs    = rng.randint(20, 80)
        data   = inject_equipment_fault(data, start_idx=eqs)
        rs     = int(np.clip(eqs + rng.randint(10, 40), 0, 169))
        X.append(extract_features(data.iloc[rs : rs + WIN]))
        y.append("Equipment Fault")

    return np.array(X, dtype=np.float32), np.array(y)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class RootCauseClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.label_encoder = LabelEncoder()
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, n_per_class: int = 400) -> "RootCauseClassifier":
        X, y = _generate_training_data(n_per_class)
        self.label_encoder.fit(CAUSE_LABELS)
        y_enc = self.label_encoder.transform(y)
        self.model.fit(X, y_enc)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_proba_dict(self, features: np.ndarray) -> dict:
        """Return {cause_label: probability} for all classes."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        proba   = self.model.predict_proba(features.reshape(1, -1))[0]
        classes = self.label_encoder.inverse_transform(np.arange(len(proba)))
        return {str(c): float(p) for c, p in zip(classes, proba)}

    def predict_cause(self, features: np.ndarray) -> str:
        d = self.predict_proba_dict(features)
        return max(d, key=d.get)

    def get_class_index(self, cause: str) -> int:
        return int(self.label_encoder.transform([cause])[0])

    def feature_importances_df(self) -> pd.DataFrame:
        from anomaly_detector import FEATURE_NAMES
        return (
            pd.DataFrame({
                "feature":    FEATURE_NAMES,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
