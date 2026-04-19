"""
dashboard.py
Streamlit clinical monitoring dashboard for ICU Patient Deterioration Risk.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from features import build_feature_matrix, compute_features_for_patient
from inference import ICURiskPredictor, RISK_TIERS, score_to_tier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Risk Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0d1117; }
  .risk-card {
    padding: 1.2rem 1.5rem; border-radius: 10px; margin: 0.5rem 0;
    border-left: 5px solid;
  }
  .risk-low    { background: #0d2118; border-color: #3fb950; }
  .risk-medium { background: #2d1f00; border-color: #ffa657; }
  .risk-high   { background: #2d0f0f; border-color: #f78166; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.8} }
  .metric-box {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 0.8rem; text-align: center;
  }
  .stSelectbox label { color: #8b949e !important; }
  h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data & model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return ICURiskPredictor()

@st.cache_data
def load_predictions():
    return pd.read_csv("outputs/predictions.csv")

@st.cache_data
def load_raw():
    return pd.read_csv("data/icu_vitals_raw.csv")

@st.cache_data
def load_metrics():
    with open("models/metrics.json") as f:
        return json.load(f)

predictor   = load_model()
predictions = load_predictions()
raw_df      = load_raw()
metrics     = load_metrics()
best_model  = metrics["best_model"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 ICU Risk Monitor")
    st.markdown("---")

    view_mode = st.radio("View", ["Patient Monitor", "Population Overview", "Model Performance"])

    if view_mode == "Patient Monitor":
        st.markdown("### Patient Selection")
        tier_filter = st.multiselect(
            "Filter by risk tier",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
        filtered = predictions[predictions["risk_tier"].isin(tier_filter)]
        patient_ids = filtered["patient_id"].tolist()
        selected_pid = st.selectbox("Select patient", patient_ids)

    st.markdown("---")
    st.markdown(f"**Model:** {best_model}")
    st.markdown(f"**AUROC:** {metrics[best_model]['auroc']:.3f}")
    st.markdown(f"**Recall:** {metrics[best_model]['recall']:.3f}")
    st.markdown(f"**Threshold:** {metrics['threshold']:.2f}")
    st.markdown(f"**Patients:** {len(predictions):,}")
    st.markdown("---")
    st.caption("Synthetic data — not for clinical use")


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 1: PATIENT MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
if view_mode == "Patient Monitor":
    # Get patient data
    patient_raw = raw_df[raw_df["patient_id"] == selected_pid].sort_values("hour")
    patient_pred = predictions[predictions["patient_id"] == selected_pid].iloc[0]

    risk_score = patient_pred["risk_score"]
    tier = patient_pred["risk_tier"]
    tier_color = RISK_TIERS[tier]["color"]

    # ── Header ─────────────────────────────────────────────────────────────
    col_title, col_score = st.columns([3, 1])
    with col_title:
        st.markdown(f"## Patient {selected_pid}")
        st.markdown(f"Age: **{int(patient_pred['age'])}** · "
                    f"APACHE-II: **{int(patient_pred['apache_ii'])}** · "
                    f"NEWS2: **{int(patient_pred['news2_total'])}** · "
                    f"Hours observed: **{len(patient_raw)}**")

    with col_score:
        tier_css = tier.lower()
        actual_label = "⚠️ Deteriorated" if patient_pred["actual_outcome"] == 1 else "✓ Stable (actual)"
        st.markdown(f"""
        <div class="risk-card risk-{tier_css}">
          <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.1em">RISK LEVEL</div>
          <div style="font-size:2rem;font-weight:700;color:{tier_color}">{tier.upper()}</div>
          <div style="font-size:1.1rem;color:#e6edf3">{risk_score*100:.1f}%</div>
          <div style="font-size:0.8rem;color:#8b949e;margin-top:4px">{actual_label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Clinical recommendation ─────────────────────────────────────────────
    action = RISK_TIERS[tier]["action"]
    escalation = RISK_TIERS[tier]["escalation"]
    if tier == "High":
        st.error(f"🚨 **IMMEDIATE ACTION REQUIRED** · {action}")
    elif tier == "Medium":
        st.warning(f"⚠️ **ATTENTION** · {action}")
    else:
        st.success(f"✓ **STABLE** · {action}")

    # ── Vital sign trend charts ─────────────────────────────────────────────
    st.markdown("### Vital Sign Trends (Last 24h)")

    vitals_to_plot = [
        ("hr",   "Heart Rate",      "bpm",  50, 100,  "#58a6ff"),
        ("sbp",  "Systolic BP",     "mmHg", 90, 160,  "#3fb950"),
        ("rr",   "Resp. Rate",      "/min", 12, 20,   "#ffa657"),
        ("spo2", "SpO2",            "%",    94, 100,  "#d2a8ff"),
        ("temp", "Temperature",     "°C",   36, 38.5, "#f78166"),
        ("gcs",  "GCS",             "pts",  8,  15,   "#79c0ff"),
    ]

    col1, col2 = st.columns(2)
    cols_cycle = [col1, col2, col1, col2, col1, col2]

    display_df = patient_raw.tail(24)

    fig_vitals, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig_vitals.patch.set_facecolor("#0d1117")
    axes_flat = axes.flatten()

    for ax, (vcol, vlabel, vunit, vlo, vhi, vcolor) in zip(axes_flat, vitals_to_plot):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values(): spine.set_edgecolor("#30363d")

        hours = display_df["hour"].values
        vals  = display_df[vcol].ffill().bfill().values

        ax.plot(hours, vals, color=vcolor, lw=2, zorder=3)
        ax.fill_between(hours, vals, alpha=0.15, color=vcolor)
        ax.axhline(vlo, color="#f78166", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(vhi, color="#f78166", lw=0.8, ls="--", alpha=0.6)
        ax.axhspan(vlo, vhi, alpha=0.06, color="#3fb950")

        last_val = vals[-1]
        alert_color = "#f78166" if (last_val < vlo or last_val > vhi) else "#3fb950"
        ax.scatter([hours[-1]], [last_val], color=alert_color, s=60, zorder=5)

        ax.set_title(f"{vlabel} ({vunit})", color="#e6edf3", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.set_xlabel("Hour", color="#8b949e", fontsize=8)

    fig_vitals.suptitle("Vital Sign Trends with Clinical Normal Ranges",
                         color="#e6edf3", fontsize=12, fontweight="bold")
    fig_vitals.tight_layout()
    st.pyplot(fig_vitals, use_container_width=True)
    plt.close()

    # ── Feature contribution table ──────────────────────────────────────────
    st.markdown("### Top Risk Contributing Features")

    feat_imp = pd.read_csv("outputs/feature_importances.csv", index_col=0)
    feat_imp.columns = ["importance"]
    feat_names_top = feat_imp.head(15).index.tolist()

    patient_features = compute_features_for_patient(patient_raw)
    alert = predictor.predict_patient(patient_features)

    contrib_rows = []
    for feat in feat_names_top:
        val = patient_features.get(feat, np.nan)
        imp = feat_imp.loc[feat, "importance"] if feat in feat_imp.index else 0
        contrib_rows.append({"Feature": feat, "Value": round(val, 3) if not np.isnan(val) else "—",
                              "Importance": round(imp, 4)})

    contrib_df = pd.DataFrame(contrib_rows).head(10)
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    # ── Alert reasons ──────────────────────────────────────────────────────
    if alert["top_risk_factors"]:
        st.markdown("### Clinical Alert Summary")
        for reason in alert["top_risk_factors"][:5]:
            st.markdown(f"  · {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 2: POPULATION OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Population Overview":
    st.markdown("## Population Risk Overview")
    st.markdown(f"**{len(predictions):,} patients** currently monitored")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    tier_counts = predictions["risk_tier"].value_counts()

    with c1:
        n = tier_counts.get("High", 0)
        st.metric("🔴 High Risk", n, f"{n/len(predictions)*100:.1f}%")
    with c2:
        n = tier_counts.get("Medium", 0)
        st.metric("🟡 Medium Risk", n, f"{n/len(predictions)*100:.1f}%")
    with c3:
        n = tier_counts.get("Low", 0)
        st.metric("🟢 Low Risk", n, f"{n/len(predictions)*100:.1f}%")
    with c4:
        actual_pos = predictions["actual_outcome"].sum()
        st.metric("Actual Deteriorations", actual_pos, f"{actual_pos/len(predictions)*100:.1f}%")

    st.markdown("---")

    # Risk score distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")

    ax = axes[0]
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")

    colors = {"High": "#f78166", "Medium": "#ffa657", "Low": "#3fb950"}
    for tier_name, grp in predictions.groupby("risk_tier"):
        ax.hist(grp["risk_score"], bins=30, alpha=0.7, color=colors.get(tier_name, "#58a6ff"),
                label=tier_name, density=True)
    ax.axvline(0.30, color="white", lw=1.2, ls="--")
    ax.axvline(0.65, color="white", lw=1.2, ls="--")
    ax.set_xlabel("Risk Score", color="#8b949e"); ax.set_ylabel("Density", color="#8b949e")
    ax.set_title("Risk Score Distribution", color="#e6edf3", fontweight="bold")
    ax.tick_params(colors="#8b949e")
    ax.legend(labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d")

    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    for spine in ax2.spines.values(): spine.set_edgecolor("#30363d")
    tier_order = ["Low", "Medium", "High"]
    bar_colors = ["#3fb950", "#ffa657", "#f78166"]
    counts = [tier_counts.get(t, 0) for t in tier_order]
    bars = ax2.bar(tier_order, counts, color=bar_colors, alpha=0.85)
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(cnt), ha="center", color="#e6edf3", fontsize=12, fontweight="bold")
    ax2.set_title("Patients by Risk Tier", color="#e6edf3", fontweight="bold")
    ax2.tick_params(colors="#8b949e")
    ax2.set_ylabel("Count", color="#8b949e")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # High-risk patient table
    st.markdown("### High Risk Patients — Immediate Review")
    high_risk = predictions[predictions["risk_tier"] == "High"].sort_values(
        "risk_score", ascending=False
    )[["patient_id","risk_score","age","apache_ii","news2_total","shock_index","actual_outcome"]]
    high_risk.columns = ["Patient ID","Risk Score","Age","APACHE-II","NEWS2","Shock Index","Deteriorated"]
    st.dataframe(high_risk.head(20), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Model Performance":
    st.markdown("## Model Performance Report")

    c1, c2, c3, c4, c5 = st.columns(5)
    m = metrics[best_model]
    c1.metric("AUROC",     f"{m['auroc']:.3f}")
    c2.metric("AUPRC",     f"{m['auprc']:.3f}")
    c3.metric("Recall",    f"{m['recall']:.3f}")
    c4.metric("Precision", f"{m['precision']:.3f}")
    c5.metric("F1 Score",  f"{m['f1']:.3f}")

    st.markdown("---")
    st.image("outputs/evaluation_report.png", use_container_width=True)

    st.markdown("### Why these metrics matter clinically")
    st.markdown("""
    | Metric | Clinical relevance |
    |--------|-------------------|
    | **Recall (sensitivity)** | Proportion of deteriorating patients correctly identified. Maximized to avoid missing critical events. |
    | **Precision (PPV)** | Fraction of alerts that are true deteriorations. Balances alert fatigue. |
    | **AUROC** | Overall discriminative ability across all thresholds — gold standard for clinical risk models. |
    | **AUPRC** | More informative than AUROC under class imbalance (only ~28% deteriorate). |
    | **Classification threshold = 0.32** | Deliberately lowered from 0.5 to increase recall — in ICU, a missed deterioration is far more costly than a false alarm. |
    """)
