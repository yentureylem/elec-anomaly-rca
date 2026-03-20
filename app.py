"""
app.py  —  Streamlit dashboard
Electrical Anomaly Root Cause Analysis  |  XAI Pipeline Demo
"""

import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Elec Anomaly RCA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached so it only runs once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on synthetic data… (one-time, ~20 s)")
def load_models():
    from data_generator import generate_normal_signal
    from anomaly_detector import AnomalyDetector
    from root_cause_classifier import RootCauseClassifier
    from xai_explainer import XAIExplainer

    detector   = AnomalyDetector(contamination=0.05, window_size=30)
    clean_data = generate_normal_signal(n_samples=600, seed=0)
    detector.fit_on_normal(clean_data)

    classifier = RootCauseClassifier()
    classifier.fit(n_per_class=400)

    explainer  = XAIExplainer(classifier)
    explainer.fit()

    return detector, classifier, explainer


detector, classifier, explainer = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Elec Anomaly RCA")
st.sidebar.caption("XAI-based root cause analysis for electrical measurement anomalies")
st.sidebar.divider()

SCENARIO_LABELS = {
    "normal":           "✅  Normal operation",
    "sensor_fault":     "🔴  Sensor fault",
    "cyber_attack":     "🟠  Cyber attack",
    "equipment_fault":  "🟡  Equipment fault",
}

scenario  = st.sidebar.selectbox(
    "Inject anomaly scenario",
    list(SCENARIO_LABELS.keys()),
    format_func=lambda k: SCENARIO_LABELS[k],
    index=1,
)
seed      = st.sidebar.slider("Random seed", 0, 99, 42)
threshold = st.sidebar.slider("Anomaly threshold", 0.20, 0.90, 0.50, 0.05)

st.sidebar.divider()
st.sidebar.markdown("**Pipeline steps**")
st.sidebar.markdown(
    "1. Synthetic data generation\n"
    "2. Feature extraction (rolling window)\n"
    "3. Isolation Forest — anomaly score\n"
    "4. SHAP — feature importance\n"
    "5. Random Forest — root cause\n"
)
st.sidebar.divider()
st.sidebar.caption("Fraunhofer IOSB · ODAL Project Demo")

# ─────────────────────────────────────────────────────────────────────────────
# Data + scores
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_results(scenario, seed, threshold):
    from data_generator import get_scenario_data

    data, anom_start = get_scenario_data(scenario, seed=seed)
    scores = detector.compute_anomaly_scores(data)

    peak_idx   = int(np.argmax(scores))
    peak_score = float(scores[peak_idx])
    is_anomaly = peak_score > threshold

    features = detector.get_window_features(data, center=peak_idx)
    proba    = classifier.predict_proba_dict(features)
    cause    = max(proba, key=proba.get)

    top_feats  = explainer.get_top_features(features, predicted_cause=cause, top_n=8)
    full_feats = explainer.full_feature_df(features, predicted_cause=cause)

    return data, scores, anom_start, peak_idx, peak_score, is_anomaly, features, proba, cause, top_feats, full_feats


data, scores, anom_start, peak_idx, peak_score, is_anomaly, features, proba, cause, top_feats, full_feats = \
    get_results(scenario, seed, threshold)

CAUSE_COLORS = {
    "Normal":           "#1D9E75",
    "Sensor Fault":     "#185FA5",
    "Cyber Attack":     "#C04040",
    "Equipment Fault":  "#BA7517",
}

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚡ Electrical Anomaly Root Cause Analysis")
st.markdown(
    "Transparent AI pipeline: **Isolation Forest** anomaly detection → "
    "**SHAP** feature importance → **Random Forest** root cause classification"
)

# ── KPI cards ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

status_icon = "🔴" if is_anomaly else "🟢"
with c1:
    st.metric("System status", f"{status_icon} {'ANOMALY' if is_anomaly else 'NORMAL'}")
with c2:
    st.metric("Peak anomaly score", f"{peak_score:.3f}", delta=f"threshold: {threshold:.2f}")
with c3:
    display_cause = cause if is_anomaly else "Normal"
    st.metric("Predicted cause", display_cause)
with c4:
    conf = proba.get(cause, 0)
    st.metric("Confidence", f"{conf * 100:.1f}%")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Time series
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Electrical Measurements")

def fault_annotation(fig, anom_start, data, **kwargs):
    if anom_start > 0 and anom_start < len(data):
        tx = float(data["time"].iloc[anom_start])
        fig.add_vrect(x0=tx, x1=float(data["time"].iloc[-1]),
                      fillcolor="rgba(226,75,74,0.07)", line_width=0, **kwargs)
        fig.add_vline(x=tx, line_dash="dash", line_color="rgba(226,75,74,0.55)",
                      line_width=1.5,
                      annotation_text="fault start",
                      annotation_position="top left",
                      **kwargs)

LAYOUT = dict(
    height=280,
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.12, font_size=12),
)
GRID = dict(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.15)")

tab_vi, tab_f, tab_pf = st.tabs(["Voltage & Current", "Frequency", "Power Factor & THD"])

with tab_vi:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=data["time"], y=data["voltage"],
                             name="Voltage (V)",
                             line=dict(color="#185FA5", width=1.5),
                             hovertemplate="%{x:.1f}s: %{y:.2f} V<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=data["time"], y=data["current"],
                             name="Current (A)",
                             line=dict(color="#1D9E75", width=1.5),
                             hovertemplate="%{x:.1f}s: %{y:.3f} A<extra></extra>"),
                  secondary_y=True)
    fault_annotation(fig, anom_start, data)
    fig.update_layout(**LAYOUT)
    fig.update_xaxes(title="Time (s)", **GRID)
    fig.update_yaxes(title="Voltage (V)", secondary_y=False, **GRID)
    fig.update_yaxes(title="Current (A)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

with tab_f:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["time"], y=data["frequency"],
                             line=dict(color="#BA7517", width=1.5), name="Frequency (Hz)"))
    fig.add_hline(y=50.0, line_dash="dot", line_color="rgba(128,128,128,0.4)",
                  annotation_text="50 Hz", annotation_position="top right")
    fault_annotation(fig, anom_start, data)
    fig.update_layout(**LAYOUT)
    fig.update_xaxes(title="Time (s)", **GRID)
    fig.update_yaxes(title="Frequency (Hz)", **GRID)
    st.plotly_chart(fig, use_container_width=True)

with tab_pf:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=data["time"], y=data["power_factor"],
                             name="Power factor",
                             line=dict(color="#534AB7", width=1.5)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=data["time"], y=data["thd"],
                             name="THD (%)",
                             line=dict(color="#D85A30", width=1.5)),
                  secondary_y=True)
    fault_annotation(fig, anom_start, data)
    fig.update_layout(**LAYOUT)
    fig.update_xaxes(title="Time (s)", **GRID)
    fig.update_yaxes(title="Power factor", secondary_y=False, **GRID)
    fig.update_yaxes(title="THD (%)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Anomaly score | SHAP | RCA
# ─────────────────────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)

# ── Anomaly score ────────────────────────────────────────────────────────────
with col_a:
    st.subheader("🔍 Anomaly Score")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time"], y=scores,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#185FA5", width=1.5),
        fillcolor="rgba(24,95,165,0.10)",
        name="Score",
        hovertemplate="%{x:.1f}s: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=threshold, line_dash="dash",
                  line_color="rgba(226,75,74,0.75)", line_width=1.5,
                  annotation_text=f"threshold {threshold:.2f}",
                  annotation_position="top right")
    if peak_idx < len(data):
        fig.add_vline(x=float(data["time"].iloc[peak_idx]),
                      line_dash="dot", line_color="rgba(226,75,74,0.6)",
                      line_width=1,
                      annotation_text="peak",
                      annotation_position="top left")
    fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                      showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Time (s)", **GRID)
    fig.update_yaxes(title="Score [0–1]", range=[0, 1.08], **GRID)
    st.plotly_chart(fig, use_container_width=True)

# ── SHAP feature importance ──────────────────────────────────────────────────
with col_b:
    st.subheader("🧠 SHAP Feature Importance")
    rev  = top_feats[::-1].reset_index(drop=True)
    fig  = go.Figure(go.Bar(
        x=rev["shap_value"],
        y=rev["display_name"],
        orientation="h",
        marker_color=CAUSE_COLORS.get(cause, "#185FA5"),
        text=[f"{v:.3f}" for v in rev["shap_value"]],
        textposition="outside",
        textfont_size=11,
    ))
    fig.update_layout(height=260, margin=dict(l=0,r=50,t=10,b=0),
                      showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="|SHAP value|", **GRID)
    fig.update_yaxes(tickfont_size=11)
    st.plotly_chart(fig, use_container_width=True)

# ── Root cause classification ────────────────────────────────────────────────
with col_c:
    st.subheader("🎯 Root Cause")

    cause_order = ["Normal", "Sensor Fault", "Cyber Attack", "Equipment Fault"]
    probs  = [proba.get(c, 0.0) for c in cause_order]
    colors = [CAUSE_COLORS[c] for c in cause_order]

    fig = go.Figure(go.Bar(
        x=probs,
        y=cause_order,
        orientation="h",
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont_size=11,
    ))
    fig.add_vline(x=0.5, line_dash="dot",
                  line_color="rgba(128,128,128,0.4)", line_width=1)
    fig.update_layout(height=180, margin=dict(l=0,r=60,t=10,b=0),
                      showlegend=False, xaxis_range=[0, 1.18],
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Probability", tickformat=".0%", **GRID)
    fig.update_yaxes(tickfont_size=12)
    st.plotly_chart(fig, use_container_width=True)

    # Verdict card
    c = CAUSE_COLORS.get(cause, "#888888")
    st.markdown(
        f"""<div style="border:1px solid {c}55; border-left:4px solid {c};
             border-radius:8px; padding:12px 16px; background:{c}14; margin-top:4px">
            <div style="font-size:11px; color:gray; margin-bottom:4px; text-transform:uppercase; letter-spacing:.5px">Verdict</div>
            <div style="font-size:17px; font-weight:600; color:{c}">{cause}</div>
            <div style="font-size:13px; color:gray; margin-top:4px">Confidence: {conf*100:.1f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Feature table + model info
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📊 Full feature vector (extracted from peak window)"):
    st.dataframe(full_feats, use_container_width=True, hide_index=True)

with st.expander("🌲 Random Forest feature importances (global)"):
    rf_imp = classifier.feature_importances_df()
    from xai_explainer import FEATURE_DISPLAY_NAMES
    rf_imp["display_name"] = rf_imp["feature"].map(
        lambda f: FEATURE_DISPLAY_NAMES.get(f, f)
    )
    fig = go.Figure(go.Bar(
        x=rf_imp["importance"].values[:10][::-1],
        y=rf_imp["display_name"].values[:10][::-1],
        orientation="h",
        marker_color="#534AB7",
    ))
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                      showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Importance", **GRID)
    st.plotly_chart(fig, use_container_width=True)
