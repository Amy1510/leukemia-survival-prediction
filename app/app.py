"""
streamlit_app.py — Leukemia Survival Predictor
Usage: streamlit run app/streamlit_app.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from src.features import CYTO_CLASS_ORDER
from src.config import (
    MODEL_RSF_CLIN,
    PIPELINE_CLINICAL,
    MODEL_RSF_FULL,
    PIPELINE_FULL,
    FEAT_COLS_FULL,
)

st.set_page_config(
    page_title="Leukemia Survival Predictor", page_icon="🩺", layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_RSF_CLIN), joblib.load(PIPELINE_CLINICAL)


try:
    model, pipeline = load_model()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    load_error = str(e)

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.header("🩺 Patient clinical profile")
cyto_class = st.sidebar.selectbox(
    "Cytogenetic risk group (ELN 2022)",
    CYTO_CLASS_ORDER,
    index=CYTO_CLASS_ORDER.index("Normal"),
)
cyto_ord = CYTO_CLASS_ORDER.index(cyto_class)

st.sidebar.subheader("Hematological markers")
bm_blast = st.sidebar.slider("BM_BLAST — Bone marrow blasts (%)", 0.0, 100.0, 5.0, 0.5)
wbc = st.sidebar.slider("WBC — White blood cells (G/L)", 0.1, 200.0, 6.5, 0.1)
anc = st.sidebar.slider("ANC — Neutrophils (G/L)", 0.0, 120.0, 3.0, 0.1)
monocytes = st.sidebar.slider("MONOCYTES (G/L)", 0.0, 50.0, 1.0, 0.1)
hb = st.sidebar.slider("HB — Hemoglobin (g/dL)", 4.0, 18.0, 10.0, 0.1)
plt_count = st.sidebar.slider("PLT — Platelets (G/L)", 2.0, 600.0, 167.0, 1.0)
predict_btn = st.sidebar.button(
    "🔮 Predict survival", type="primary", use_container_width=True
)

# ── Header ───────────────────────────────────────────────────────────────
st.title("Leukemia Survival Predictor")
st.markdown(
    "> **Disclaimer** — For research and educational purposes only. Not for clinical use."
)

if not model_loaded:
    st.error(f"Model not found. Run the notebook first.\n\n{load_error}")
    st.stop()

# ── Risk badge ───────────────────────────────────────────────────────────
favorable = ["APL t(15;17)", "inv(16)", "t(8;21)", "Normal"]
adverse = ["Monosomy 7", "Complex"]
risk_badge = (
    "🟢 Favourable"
    if cyto_class in favorable
    else ("🔴 Adverse" if cyto_class in adverse else "🟡 Intermediate")
)

col1, col2 = st.columns([1.4, 1], gap="large")
with col1:
    st.subheader("Patient summary")
    c1, c2 = st.columns(2)
    c1.metric("Cytogenetic class", cyto_class)
    c1.metric("ELN risk group", risk_badge)
    c2.metric("BM blasts", f"{bm_blast:.1f} %")
    c2.metric("WBC", f"{wbc:.1f} G/L")
    st.dataframe(
        pd.DataFrame(
            {
                "Marker": [
                    "BM_BLAST",
                    "WBC",
                    "ANC",
                    "MONOCYTES",
                    "HB",
                    "PLT",
                    "CYTO_CLASS",
                ],
                "Value": [bm_blast, wbc, anc, monocytes, hb, plt_count, cyto_class],
                "Unit": ["%", "G/L", "G/L", "G/L", "g/dL", "G/L", "—"],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with col2:
    st.subheader("Model information")
    st.markdown("""
| | |
|---|---|
| **Algorithm** | Random Survival Forest |
| **Discrimination score** | 0.741 [0.728 – 0.754] |
| **Dataset** | QRT 2025 — 2456 patients |
| **Validation** | Hold-out 20% test set |
""")

# ── Prediction ───────────────────────────────────────────────────────────
if predict_btn:
    feat_names = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT", "CYTO_CLASS_ORD"]
    X_df = pd.DataFrame(
        [[bm_blast, wbc, anc, monocytes, hb, plt_count, cyto_ord]], columns=feat_names
    )
    try:
        X_scaled = pipeline.transform(X_df)
    except Exception:
        X_scaled = pipeline.transform(X_df.values)

    surv_fn = model.predict_survival_function(X_scaled)[0]
    times, probs = surv_fn.x, surv_fn(surv_fn.x)

    below_50 = times[probs <= 0.5]
    median_surv = below_50[0] if len(below_50) > 0 else None
    t1 = float(surv_fn(np.array([1.0]))[0])
    t3 = float(surv_fn(np.array([3.0]))[0])

    st.divider()
    st.subheader("Predicted survival")
    m1, m2, m3 = st.columns(3)
    m1.metric("1-year survival", f"{t1:.1%}")
    m2.metric("3-year survival", f"{t3:.1%}")
    m3.metric(
        "Median survival", f"{median_surv:.1f} years" if median_surv else "> 10 years"
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(times, probs, where="post", color="#2563eb", lw=2.5)
    ax.fill_between(times, probs, alpha=0.1, color="#2563eb", step="post")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="50% survival")
    if median_surv:
        ax.axvline(
            median_surv,
            color="#dc2626",
            ls=":",
            lw=1.2,
            label=f"Median OS ≈ {median_surv:.1f} yr",
        )
    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_title(f"Predicted survival — {cyto_class} · {risk_badge}", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    st.info(
        "**How to read:** The curve shows the probability the patient is still alive at each time point."
    )
else:
    st.info(
        "👈 Set the patient's clinical values in the sidebar and click **Predict survival**."
    )
