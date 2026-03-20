# ⚡ XAI-Driven Root Cause Analysis for Electrical Measurement Anomalies

**XAI-Driven Root Cause Analysis for Electrical Measurement Anomalies**

XAI methods such as SHAP provide feature importance values but do not directly determine the actual cause of anomalies. This project proposes a two-stage approach combining anomaly detection, explainability, and supervised root cause classification to infer anomaly causes in electrical measurement systems.

## Pipeline

```
Synthetic data  →  Feature extraction  →  Isolation Forest  →  SHAP  →  RF Root Cause Classifier
```

| Stage | Method | Output |
|---|---|---|
| Data ingestion | Synthetic V / I / f / PF / THD signals + anomaly injection | pd.DataFrame |
| Anomaly detection | Isolation Forest (rolling window) | Anomaly score [0–1] |
| XAI | SHAP TreeExplainer | Feature importance per class |
| Root cause | Random Forest classifier | Cause + confidence |

## Anomaly scenarios

| Scenario | Description |
|---|---|
| Normal | Clean signal, no fault |
| Sensor Fault | Stuck value, offset, or excessive noise on one channel |
| Cyber Attack | Coordinated multi-channel data injection → breaks V–I correlation |
| Equipment Fault | Gradual voltage drop + PF degradation + rising THD |

## Quick start

```bash
# 1. Clone / unzip the project
cd elec_anomaly_rca

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
```

The app opens at **http://localhost:8501** automatically.

> First launch trains models on synthetic data (~20 s). Subsequent launches use the cache.

## Project structure

```
elec_anomaly_rca/
├── app.py                    # Streamlit dashboard
├── data_generator.py         # Synthetic signal + anomaly injection
├── anomaly_detector.py       # Feature extraction + Isolation Forest
├── root_cause_classifier.py  # Random Forest root cause model
├── xai_explainer.py          # SHAP wrapper
└── requirements.txt
```

## Tech stack

- `scikit-learn` — Isolation Forest, Random Forest
- `shap` — SHAP TreeExplainer for interpretability
- `streamlit` — interactive dashboard
- `plotly` — interactive charts
- `scipy` — statistical feature extraction
